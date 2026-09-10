use std::sync::Arc;
use std::time::{Duration, Instant};

use camera::{
    CameraSystem, DeliveryPolicy, DeviceId, FrameRate, MemoryBudget, OutputFormat, StreamRequest,
};
use image::RgbImage;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, watch};

use crate::{Error, ModelBundle};
use crate::{
    GazeEngine, Result, StabilizerConfig, TargetStabilizer, TrackerEvent, TrackerSnapshot,
    TrackingState, WorkspaceMapper,
};

/// Camera shown to the ArcRelay desktop UI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CameraDescriptor {
    pub id: String,
    pub name: String,
    pub description: String,
}

/// Capture and inference scheduling policy.
#[derive(Clone, Debug)]
pub struct TrackerConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub inference_interval: Duration,
    pub include_preview: bool,
    pub stabilizer: StabilizerConfig,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 360,
            fps: 30,
            inference_interval: Duration::from_millis(66),
            include_preview: false,
            stabilizer: StabilizerConfig::default(),
        }
    }
}

/// Shareable factory and workspace mapping state for gaze sessions.
pub struct GazeTracker {
    engine: Arc<GazeEngine>,
    camera_system: CameraSystem,
    mapper: Arc<RwLock<Option<Arc<WorkspaceMapper>>>>,
}

impl GazeTracker {
    #[must_use]
    pub fn new(engine: GazeEngine) -> Self {
        Self {
            engine: Arc::new(engine),
            camera_system: CameraSystem::new(),
            mapper: Arc::new(RwLock::new(None)),
        }
    }

    pub fn with_bundled_models(threads: usize) -> Result<Self> {
        Ok(Self::new(GazeEngine::bundled(threads)?))
    }

    pub fn with_runtime(runtime: &mnn_runtime::Runtime) -> Result<Self> {
        Ok(Self::new(GazeEngine::new(runtime, ModelBundle::bundled())?))
    }

    pub async fn cameras(&self) -> Result<Vec<CameraDescriptor>> {
        camera_descriptors(&self.camera_system).await
    }

    /// Enumerate cameras without loading any MNN model. Desktop settings use
    /// this path so opening the device picker remains cheap and side-effect free.
    pub async fn available_cameras() -> Result<Vec<CameraDescriptor>> {
        camera_descriptors(&CameraSystem::new()).await
    }

    pub fn set_workspace_mapper(&self, mapper: Option<WorkspaceMapper>) {
        *self.mapper.write() = mapper.map(Arc::new);
    }

    pub async fn start(&self, camera_id: &str, config: TrackerConfig) -> Result<TrackerSession> {
        if config.fps == 0 || config.inference_interval.is_zero() {
            return Err(Error::InvalidConfig(
                "fps and inference interval must be positive".into(),
            ));
        }
        let devices = self.camera_system.devices().await?;
        let device = devices
            .into_iter()
            .find(|device| device.id.to_string() == camera_id)
            .ok_or_else(|| camera::CameraError::DeviceNotFound(camera_id.into()))?;
        self.start_device(device.id, device.name, config).await
    }

    async fn start_device(
        &self,
        device_id: DeviceId,
        device_name: String,
        config: TrackerConfig,
    ) -> Result<TrackerSession> {
        let mut camera = self.camera_system.open(&device_id).await?;
        let request = StreamRequest::builder()
            .resolution(config.width, config.height)
            .frame_rate(FrameRate::new(config.fps, 1)?)
            .output(OutputFormat::Rgb8)
            .delivery(DeliveryPolicy::Latest)
            .memory_budget(MemoryBudget {
                buffers: 4,
                bytes: 32 * 1024 * 1024,
            })
            .startup_timeout(Duration::from_secs(5))
            .build()?;
        let capture = camera.start(request).await?;
        let negotiated = capture.negotiated_config().capture.clone();
        let camera_id = device_id.to_string();
        let initial = TrackerSnapshot {
            state: TrackingState::Starting,
            camera_id: Some(camera_id.clone()),
            camera_name: Some(device_name.clone()),
            frame_width: negotiated.width,
            frame_height: negotiated.height,
            ..TrackerSnapshot::default()
        };
        let (snapshot_tx, snapshot_rx) = watch::channel(initial);
        let (event_tx, _) = broadcast::channel(4);
        let (stop_tx, mut stop_rx) = watch::channel(false);
        let engine = self.engine.clone();
        let mapper = self.mapper.clone();
        let events = event_tx.clone();
        let task = tokio::spawn(async move {
            let mut receiver = capture.subscribe();
            let mut stabilizer = TargetStabilizer::new(config.stabilizer);
            let mut last_inference = None;
            let mut snapshot = snapshot_tx.borrow().clone();
            snapshot.state = TrackingState::Tracking;
            snapshot_tx.send_replace(snapshot.clone());
            loop {
                tokio::select! {
                    changed = stop_rx.changed() => {
                        if changed.is_err() || *stop_rx.borrow() {
                            break;
                        }
                    }
                    frame = receiver.next() => {
                        let frame = match frame {
                            Ok(frame) => frame,
                            Err(camera::CameraError::StreamStopped) => break,
                            Err(error) => {
                                snapshot.state = TrackingState::Failed;
                                snapshot.error = Some(error.to_string());
                                snapshot_tx.send_replace(snapshot.clone());
                                let _ = events.send(TrackerEvent { snapshot, rgb_preview: None });
                                let _ = capture.stop().await;
                                return;
                            }
                        };
                        snapshot.captured_frames = snapshot.captured_frames.saturating_add(1);
                        let now = Instant::now();
                        if last_inference.is_some_and(|last: Instant| {
                            now.saturating_duration_since(last) < config.inference_interval
                        }) {
                            continue;
                        }
                        last_inference = Some(now);
                        let layout = frame.layout();
                        let pixels = frame.bytes().to_vec();
                        let preview = config.include_preview.then(|| Arc::<[u8]>::from(pixels.clone()));
                        let Some(image) = RgbImage::from_raw(layout.width, layout.height, pixels) else {
                            snapshot.state = TrackingState::Failed;
                            snapshot.error = Some("camera returned an invalid RGB frame".into());
                            snapshot_tx.send_replace(snapshot.clone());
                            continue;
                        };
                        let inference_engine = engine.clone();
                        let observation = match tokio::task::spawn_blocking(move || inference_engine.infer(&image)).await {
                            Ok(Ok(observation)) => observation,
                            Ok(Err(error)) => {
                                snapshot.state = TrackingState::Failed;
                                snapshot.error = Some(error.to_string());
                                snapshot_tx.send_replace(snapshot.clone());
                                break;
                            }
                            Err(error) => {
                                snapshot.state = TrackingState::Failed;
                                snapshot.error = Some(format!("gaze inference worker: {error}"));
                                snapshot_tx.send_replace(snapshot.clone());
                                break;
                            }
                        };
                        snapshot.inferred_frames = snapshot.inferred_frames.saturating_add(1);
                        snapshot.dropped_frames = receiver.dropped_frames();
                        snapshot.frame_width = layout.width;
                        snapshot.frame_height = layout.height;
                        snapshot.error = None;
                        snapshot.observation = observation.clone();
                        let mapped = observation
                            .as_ref()
                            .and_then(|observation| mapper.read().as_ref().and_then(|mapper| mapper.map(observation).ok().flatten()));
                        snapshot.target = stabilizer.update(mapped, now);
                        snapshot.state = if observation.is_none() {
                            TrackingState::FaceLost
                        } else if mapper.read().is_none() {
                            TrackingState::Uncalibrated
                        } else {
                            TrackingState::Tracking
                        };
                        snapshot_tx.send_replace(snapshot.clone());
                        let _ = events.send(TrackerEvent { snapshot: snapshot.clone(), rgb_preview: preview });
                    }
                }
            }
            snapshot.state = TrackingState::Stopping;
            snapshot_tx.send_replace(snapshot.clone());
            if let Err(error) = capture.stop().await {
                snapshot.state = TrackingState::Failed;
                snapshot.error = Some(error.to_string());
            } else {
                snapshot.state = TrackingState::Stopped;
            }
            snapshot_tx.send_replace(snapshot.clone());
            let _ = events.send(TrackerEvent {
                snapshot,
                rgb_preview: None,
            });
        });
        Ok(TrackerSession {
            camera_id,
            camera_name: device_name,
            snapshots: snapshot_rx,
            events: event_tx,
            stop: stop_tx,
            task: Some(task),
        })
    }
}

async fn camera_descriptors(system: &CameraSystem) -> Result<Vec<CameraDescriptor>> {
    system.devices().await.map_err(Error::from).map(|devices| {
        devices
            .into_iter()
            .map(|device| CameraDescriptor {
                id: device.id.to_string(),
                name: device.name,
                description: device.description,
            })
            .collect()
    })
}

/// Owns capture lifetime. Explicit stop waits for native camera cleanup.
pub struct TrackerSession {
    camera_id: String,
    camera_name: String,
    snapshots: watch::Receiver<TrackerSnapshot>,
    events: broadcast::Sender<TrackerEvent>,
    stop: watch::Sender<bool>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl TrackerSession {
    #[must_use]
    pub fn camera_id(&self) -> &str {
        &self.camera_id
    }

    #[must_use]
    pub fn camera_name(&self) -> &str {
        &self.camera_name
    }

    #[must_use]
    pub fn snapshot(&self) -> TrackerSnapshot {
        self.snapshots.borrow().clone()
    }

    #[must_use]
    pub fn subscribe_snapshots(&self) -> watch::Receiver<TrackerSnapshot> {
        self.snapshots.clone()
    }

    #[must_use]
    pub fn subscribe_events(&self) -> broadcast::Receiver<TrackerEvent> {
        self.events.subscribe()
    }

    pub async fn stop(&mut self) -> Result<()> {
        let _ = self.stop.send(true);
        if let Some(task) = self.task.take() {
            task.await
                .map_err(|error| Error::Worker(format!("join gaze tracker: {error}")))?;
        }
        Ok(())
    }
}

impl Drop for TrackerSession {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
    }
}
