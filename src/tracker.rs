use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use camera::{
    CameraErrorKind, CameraSystem, CaptureRequest, ConversionRequest, DeviceId, DeviceSelector,
    FrameRate, MemoryBudget, RgbConverter, SubscriptionOptions,
};
use image::RgbImage;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, watch};

use crate::observation_filter::ObservationFilter;
use crate::pipeline::InferencePlan;
use crate::presence::{
    classify_presence, FaceEmbedding, PresenceEnrollment, PresenceStabilizer,
    PRESENCE_ENROLLMENT_REQUIRED_SAMPLES,
};
use crate::{Error, ModelBundle};
use crate::{
    GazeEngine, ObservationFilterConfig, PresenceEnrollmentStatus, PresenceProfile, Result,
    StabilizerConfig, TargetStabilizer, TrackerEvent, TrackerSnapshot, TrackingState,
    WorkspaceMapper,
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
    pub stable_inference_interval: Duration,
    pub idle_inference_interval: Duration,
    pub face_detection_interval: Duration,
    pub presence_recheck_interval: Duration,
    pub include_preview: bool,
    pub presence_enabled: bool,
    pub observation_filter: ObservationFilterConfig,
    pub stabilizer: StabilizerConfig,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 360,
            fps: 15,
            inference_interval: Duration::from_millis(90),
            stable_inference_interval: Duration::from_millis(150),
            idle_inference_interval: Duration::from_millis(300),
            face_detection_interval: Duration::from_millis(500),
            presence_recheck_interval: Duration::from_secs(3),
            include_preview: false,
            presence_enabled: true,
            observation_filter: ObservationFilterConfig::default(),
            stabilizer: StabilizerConfig::default(),
        }
    }
}

/// Shareable factory and workspace mapping state for gaze sessions.
pub struct GazeTracker {
    engine: Arc<GazeEngine>,
    camera_system: CameraSystem,
    mapper: Arc<RwLock<Option<Arc<WorkspaceMapper>>>>,
    presence_profile: Arc<RwLock<Option<PresenceProfile>>>,
    presence_enrollment: Arc<Mutex<Option<PresenceEnrollment>>>,
    completed_presence_profile: Arc<Mutex<Option<PresenceProfile>>>,
    preview_enabled: Arc<AtomicBool>,
}

impl GazeTracker {
    #[must_use]
    pub fn new(engine: GazeEngine) -> Self {
        Self {
            engine: Arc::new(engine),
            camera_system: CameraSystem::new(),
            mapper: Arc::new(RwLock::new(None)),
            presence_profile: Arc::new(RwLock::new(None)),
            presence_enrollment: Arc::new(Mutex::new(None)),
            completed_presence_profile: Arc::new(Mutex::new(None)),
            preview_enabled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn from_model_directory(directory: &Path, threads: usize) -> Result<Self> {
        Ok(Self::new(GazeEngine::from_model_directory(
            directory, threads,
        )?))
    }

    pub fn with_runtime(runtime: &mnn_runtime::Runtime, models: ModelBundle<'_>) -> Result<Self> {
        Ok(Self::new(GazeEngine::new(runtime, models)?))
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

    pub fn set_presence_profile(&self, mut profile: Option<PresenceProfile>) -> Result<()> {
        if let Some(profile) = profile.as_mut() {
            profile.validate()?;
        }
        *self.presence_profile.write() = profile;
        Ok(())
    }

    #[must_use]
    pub fn presence_profile(&self) -> Option<PresenceProfile> {
        self.presence_profile.read().clone()
    }

    pub fn begin_presence_enrollment(&self, display_name: impl Into<String>) -> Result<()> {
        *self.presence_enrollment.lock() = Some(PresenceEnrollment::new(display_name)?);
        self.completed_presence_profile.lock().take();
        Ok(())
    }

    pub fn cancel_presence_enrollment(&self) {
        self.presence_enrollment.lock().take();
    }

    pub fn take_completed_presence_profile(&self) -> Option<PresenceProfile> {
        self.completed_presence_profile.lock().take()
    }

    /// Enable or disable in-memory preview frames without restarting capture.
    /// Preview pixels remain process-local and are never persisted by this crate.
    pub fn set_preview_enabled(&self, enabled: bool) {
        self.preview_enabled.store(enabled, Ordering::Release);
    }

    pub async fn start(&self, camera_id: &str, config: TrackerConfig) -> Result<TrackerSession> {
        if config.fps == 0
            || config.inference_interval.is_zero()
            || config.stable_inference_interval.is_zero()
            || config.idle_inference_interval.is_zero()
            || config.face_detection_interval.is_zero()
            || config.presence_recheck_interval.is_zero()
        {
            return Err(Error::InvalidConfig(
                "fps and inference intervals must be positive".into(),
            ));
        }
        let devices = self.camera_system.devices().await?;
        let device = devices
            .into_iter()
            .find(|device| device.id.to_string() == camera_id)
            .ok_or_else(|| camera::CameraError::device_not_found(camera_id.into()))?;
        self.start_device(device.id, device.name, config).await
    }

    async fn start_device(
        &self,
        device_id: DeviceId,
        device_name: String,
        config: TrackerConfig,
    ) -> Result<TrackerSession> {
        let mut camera = self
            .camera_system
            .open(DeviceSelector::Id(device_id.clone()))
            .await?;
        let request = CaptureRequest::builder()
            .preferred_resolution(config.width, config.height)
            .preferred_frame_rate(FrameRate::new(config.fps, 1)?)
            .memory_budget(MemoryBudget {
                buffers: 4,
                bytes: 32 * 1024 * 1024,
            })
            .startup_timeout(Duration::from_secs(5))
            .build()?;
        let capture = camera.start(request).await?;
        let negotiated = capture.negotiated().capture.clone();
        let mut receiver = capture.subscribe(SubscriptionOptions::latest())?;
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
        let presence_profile = self.presence_profile.clone();
        let presence_enrollment = self.presence_enrollment.clone();
        let completed_presence_profile = self.completed_presence_profile.clone();
        let preview_enabled = self.preview_enabled.clone();
        let events = event_tx.clone();
        let task = tokio::spawn(async move {
            let mut converter = RgbConverter::new();
            let mut rgb_buffer = Vec::new();
            let mut observation_filter = ObservationFilter::new(config.observation_filter.clone());
            let mut stabilizer = TargetStabilizer::new(config.stabilizer);
            let mut presence_stabilizer = PresenceStabilizer::new(Instant::now());
            let mut last_inference = None;
            let mut last_detection = None;
            let mut last_identity = None;
            let mut cached_identity: Option<(u64, FaceEmbedding)> = None;
            let mut previous_luma: Option<Vec<u8>> = None;
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
                            Err(error) if error.kind() == CameraErrorKind::StreamStopped => break,
                            Err(error) => {
                                snapshot.state = TrackingState::Failed;
                                snapshot.error = Some(error.to_string());
                                snapshot_tx.send_replace(snapshot.clone());
                                let _ = events.send(TrackerEvent { snapshot, rgb_preview: None });
                                let _ = capture.close().await;
                                return;
                            }
                        };
                        snapshot.captured_frames = snapshot.captured_frames.saturating_add(1);
                        let now = Instant::now();
                        let inference_interval = adaptive_inference_interval(&config, &snapshot);
                        if last_inference.is_some_and(|last: Instant| {
                            now.saturating_duration_since(last) < inference_interval
                        }) {
                            continue;
                        }
                        last_inference = Some(now);
                        let include_preview = config.include_preview || preview_enabled.load(Ordering::Acquire);
                        let prior_luma = previous_luma.take();
                        let detection_due = snapshot.observation.is_none()
                            || last_detection.is_none_or(|last: Instant| now.saturating_duration_since(last) >= config.face_detection_interval);
                        let identity_due = presence_enrollment.lock().is_some() || cached_identity.is_none()
                            || last_identity.is_none_or(|last: Instant| now.saturating_duration_since(last) >= config.presence_recheck_interval);
                        let identity_tracking = config.presence_enabled;
                        let inference_engine = engine.clone();
                        let frame_width = frame.layout().width;
                        let frame_height = frame.layout().height;
                        // There is one outstanding worker per session. Converter
                        // state and its RGB allocation return for the next frame.
                        let work = tokio::task::spawn_blocking(move || -> Result<_> {
                            let conversion_started = Instant::now();
                            let layout = frame.layout();
                            let conversion = ConversionRequest::for_layout(layout);
                            let length = conversion.output_len()?;
                            if length > 32 * 1024 * 1024 {
                                return Err(Error::InvalidConfig("camera RGB frame exceeds the conversion budget".into()));
                            }
                            rgb_buffer.resize(length, 0);
                            converter.convert_into(&frame, conversion, &mut rgb_buffer)?;
                            let image = RgbImage::from_raw(layout.width, layout.height, rgb_buffer)
                                .ok_or_else(|| Error::Worker("camera returned an invalid RGB frame".into()))?;
                            let luma = sampled_luma(&image);
                            let scene_changed = prior_luma.as_deref().is_some_and(|previous| luma_changed(previous, &luma));
                            let conversion_us = conversion_started.elapsed().as_micros() as u64;
                            let inference_started = Instant::now();
                            let inference = inference_engine.infer_planned(&image, InferencePlan {
                                detect_faces: detection_due || scene_changed,
                                estimate_gaze: true,
                                identity_tracking,
                                extract_identity: identity_tracking && (identity_due || scene_changed),
                            })?;
                            let inference_us = inference_started.elapsed().as_micros() as u64;
                            let preview = include_preview.then(|| Arc::<[u8]>::from(image.as_raw().as_slice()));
                            tracing::trace!(event = "gaze.frame.processed", conversion_us, inference_us, rgb_bytes = length, preview = include_preview);
                            Ok((converter, image.into_raw(), luma, inference, preview))
                        }).await;
                        let (returned_converter, pixels, luma, frame_inference, preview) = match work {
                            Ok(Ok(result)) => result,
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
                        converter = returned_converter;
                        rgb_buffer = pixels;
                        previous_luma = Some(luma);
                        if frame_inference.detection_ran {
                            last_detection = Some(now);
                            snapshot.detected_frames = snapshot.detected_frames.saturating_add(1);
                        } else {
                            snapshot.tracked_frames = snapshot.tracked_frames.saturating_add(1);
                        }
                        if !frame_inference.track_continuous
                            || cached_identity
                                .as_ref()
                                .is_some_and(|(track_id, _)| Some(*track_id) != frame_inference.track_id)
                        {
                            cached_identity = None;
                            last_identity = None;
                            presence_stabilizer.invalidate_track(
                                Instant::now(),
                                presence_profile.read().is_some(),
                            );
                        }
                        if frame_inference.identity_ran {
                            last_identity = Some(now);
                            snapshot.identity_frames = snapshot.identity_frames.saturating_add(1);
                            cached_identity = frame_inference
                                .track_id
                                .zip(frame_inference.embeddings.first().cloned());
                        }
                        let filtered_at = Instant::now();
                        let observation = frame_inference.observation
                            .map(|observation| observation_filter.update(observation, filtered_at));
                        if config.presence_enabled
                            && observation.as_ref().is_some_and(enrollment_quality_is_acceptable)
                        {
                            let completed = {
                                let mut enrollment = presence_enrollment.lock();
                                enrollment
                                    .as_mut()
                                    .map(|enrollment| {
                                        enrollment.push(
                                            frame_inference.face_count,
                                            &frame_inference.embeddings,
                                            observation
                                                .as_ref()
                                                .expect("enrollment quality checked")
                                                .head_pose,
                                        )
                                    })
                                    .transpose()
                            };
                            match completed {
                                Ok(Some(Some(profile))) => {
                                    *presence_profile.write() = Some(profile.clone());
                                    *completed_presence_profile.lock() = Some(profile);
                                    presence_enrollment.lock().take();
                                }
                                Ok(_) => {}
                                Err(error) => {
                                    tracing::warn!(%error, "presence enrollment was cancelled");
                                    presence_enrollment.lock().take();
                                }
                            }
                        }
                        let cached_embeddings = cached_identity
                            .as_ref()
                            .filter(|(track_id, _)| Some(*track_id) == frame_inference.track_id)
                            .map(|(_, embedding)| std::slice::from_ref(embedding))
                            .unwrap_or_default();
                        let raw_presence = if config.presence_enabled {
                            classify_presence(
                                presence_profile.read().as_ref(),
                                frame_inference.face_count,
                                cached_embeddings,
                                observation.as_ref().map(|observation| observation.head_pose),
                            )
                        } else {
                            Default::default()
                        };
                        snapshot.presence = presence_stabilizer.update(raw_presence, filtered_at);
                        snapshot.presence_enrollment = presence_enrollment
                            .lock()
                            .as_ref()
                            .map(PresenceEnrollment::status)
                            .unwrap_or_else(|| PresenceEnrollmentStatus {
                                required_samples: PRESENCE_ENROLLMENT_REQUIRED_SAMPLES,
                                ..PresenceEnrollmentStatus::default()
                            });
                        snapshot.inferred_frames = snapshot.inferred_frames.saturating_add(1);
                        snapshot.dropped_frames = receiver.dropped_frames();
                        snapshot.frame_width = frame_width;
                        snapshot.frame_height = frame_height;
                        snapshot.error = None;
                        snapshot.observation = observation.clone();
                        let mapped = observation
                            .as_ref()
                            .and_then(|observation| mapper.read().as_ref().and_then(|mapper| mapper.map(observation).ok().flatten()));
                        snapshot.target = stabilizer.update(mapped, filtered_at);
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
            if let Err(error) = capture.close().await {
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

fn adaptive_inference_interval(config: &TrackerConfig, snapshot: &TrackerSnapshot) -> Duration {
    if snapshot.presence_enrollment.active {
        return config.inference_interval;
    }
    if snapshot.observation.is_none() {
        return config.idle_inference_interval;
    }
    if snapshot
        .target
        .as_ref()
        .is_some_and(|target| target.stable_for_ms >= 1_500)
        || snapshot.presence.stable_for_ms >= 2_000
    {
        config.stable_inference_interval
    } else {
        config.inference_interval
    }
}

const LUMA_WIDTH: usize = 32;
const LUMA_HEIGHT: usize = 18;
const LUMA_CHANGE_THRESHOLD: u64 = 14;

fn sampled_luma(image: &RgbImage) -> Vec<u8> {
    let mut luma = Vec::with_capacity(LUMA_WIDTH * LUMA_HEIGHT);
    for y in 0..LUMA_HEIGHT {
        let source_y =
            (y as u32 * image.height() / LUMA_HEIGHT as u32).min(image.height().saturating_sub(1));
        for x in 0..LUMA_WIDTH {
            let source_x =
                (x as u32 * image.width() / LUMA_WIDTH as u32).min(image.width().saturating_sub(1));
            let pixel = image.get_pixel(source_x, source_y).0;
            let value =
                (u16::from(pixel[0]) * 77 + u16::from(pixel[1]) * 150 + u16::from(pixel[2]) * 29)
                    >> 8;
            luma.push(value as u8);
        }
    }
    luma
}

fn luma_changed(previous: &[u8], current: &[u8]) -> bool {
    if previous.len() != current.len() || current.is_empty() {
        return true;
    }
    let difference = previous
        .iter()
        .zip(current)
        .map(|(left, right)| u64::from(left.abs_diff(*right)))
        .sum::<u64>()
        / current.len() as u64;
    difference >= LUMA_CHANGE_THRESHOLD
}

fn enrollment_quality_is_acceptable(observation: &crate::GazeObservation) -> bool {
    observation.face_confidence >= 0.75
        && observation.head_pose.yaw.abs() <= 35.0
        && observation.head_pose.pitch.abs() <= 25.0
        && observation.face.width >= observation.frame_width as f32 * 0.12
        && observation.face.height >= observation.frame_height as f32 * 0.18
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_schedule_slows_only_stable_or_idle_tracking() {
        let config = TrackerConfig::default();
        let mut snapshot = TrackerSnapshot::default();
        assert_eq!(
            adaptive_inference_interval(&config, &snapshot),
            config.idle_inference_interval
        );

        snapshot.observation = Some(crate::GazeObservation {
            frame_width: 640,
            frame_height: 360,
            face: crate::Rect {
                x: 100.0,
                y: 60.0,
                width: 160.0,
                height: 200.0,
            },
            face_confidence: 0.9,
            landmarks: Vec::new(),
            left_eye: crate::Rect::default(),
            right_eye: crate::Rect::default(),
            left_eye_open: true,
            right_eye_open: true,
            head_pose: crate::HeadPose::default(),
            gaze: crate::Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            inference_ms: 10.0,
        });
        assert_eq!(
            adaptive_inference_interval(&config, &snapshot),
            config.inference_interval
        );

        snapshot.presence.stable_for_ms = 2_000;
        assert_eq!(
            adaptive_inference_interval(&config, &snapshot),
            config.stable_inference_interval
        );
    }

    #[test]
    fn sampled_luma_detects_material_scene_changes() {
        let black = RgbImage::from_pixel(64, 36, image::Rgb([0, 0, 0]));
        let white = RgbImage::from_pixel(64, 36, image::Rgb([255, 255, 255]));
        let black_luma = sampled_luma(&black);
        assert!(!luma_changed(&black_luma, &sampled_luma(&black)));
        assert!(luma_changed(&black_luma, &sampled_luma(&white)));
    }
}
