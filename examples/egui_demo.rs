use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use arcrelay_gaze::{
    CameraDescriptor, GazeObservation, GazeTracker, TrackerConfig, TrackerSnapshot,
};
use eframe::egui::{self, Color32, Pos2, Rect as EguiRect, Stroke, Vec2};

enum WorkerEvent {
    Frame(TrackerSnapshot, Arc<[u8]>),
    State(TrackerSnapshot),
    Error(String),
}

struct CameraWorker {
    receiver: mpsc::Receiver<WorkerEvent>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl CameraWorker {
    fn start(camera_id: String, threads: usize) -> Self {
        let (sender, receiver) = mpsc::sync_channel(2);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let thread = thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = sender.send(WorkerEvent::Error(error.to_string()));
                    return;
                }
            };
            runtime.block_on(async move {
                let tracker = match GazeTracker::with_bundled_models(threads) {
                    Ok(tracker) => tracker,
                    Err(error) => {
                        let _ = sender.send(WorkerEvent::Error(error.to_string()));
                        return;
                    }
                };
                let config = TrackerConfig {
                    include_preview: true,
                    ..TrackerConfig::default()
                };
                let mut session = match tracker.start(&camera_id, config).await {
                    Ok(session) => session,
                    Err(error) => {
                        let _ = sender.send(WorkerEvent::Error(error.to_string()));
                        return;
                    }
                };
                let mut events = session.subscribe_events();
                while !worker_stop.load(Ordering::Acquire) {
                    match tokio::time::timeout(Duration::from_millis(100), events.recv()).await {
                        Ok(Ok(event)) => {
                            let message = match event.rgb_preview {
                                Some(preview) => WorkerEvent::Frame(event.snapshot, preview),
                                None => WorkerEvent::State(event.snapshot),
                            };
                            let _ = sender.try_send(message);
                        }
                        Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => {}
                        Ok(Err(tokio::sync::broadcast::error::RecvError::Closed)) => break,
                        Err(_) => {}
                    }
                }
                if let Err(error) = session.stop().await {
                    let _ = sender.try_send(WorkerEvent::Error(error.to_string()));
                }
            });
        });
        Self {
            receiver,
            stop,
            thread: Some(thread),
        }
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for CameraWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

struct DemoApp {
    cameras: Vec<CameraDescriptor>,
    selected_camera: usize,
    worker: Option<CameraWorker>,
    snapshot: TrackerSnapshot,
    texture: Option<egui::TextureHandle>,
    frame_size: [u32; 2],
    status: String,
    threads: usize,
}

impl Default for DemoApp {
    fn default() -> Self {
        let mut app = Self {
            cameras: Vec::new(),
            selected_camera: 0,
            worker: None,
            snapshot: TrackerSnapshot::default(),
            texture: None,
            frame_size: [0, 0],
            status: "点击刷新，选择 camera-rs 枚举到的摄像头。".into(),
            threads: 4,
        };
        app.refresh_cameras();
        app
    }
}

impl DemoApp {
    fn refresh_cameras(&mut self) {
        let result = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| error.to_string())
            .and_then(|runtime| {
                let tracker =
                    GazeTracker::with_bundled_models(1).map_err(|error| error.to_string())?;
                runtime
                    .block_on(tracker.cameras())
                    .map_err(|error| error.to_string())
            });
        self.cameras = result.unwrap_or_else(|error| {
            self.status = format!("枚举摄像头失败：{error}");
            Vec::new()
        });
        self.selected_camera = self
            .selected_camera
            .min(self.cameras.len().saturating_sub(1));
        if !self.cameras.is_empty() {
            self.status = format!("发现 {} 个摄像头。", self.cameras.len());
        }
    }

    fn start(&mut self) {
        self.stop();
        let Some(camera) = self.cameras.get(self.selected_camera) else {
            self.status = "没有可用摄像头。".into();
            return;
        };
        self.status = format!("正在打开 {} 并加载五个 MNN 模型…", camera.name);
        self.worker = Some(CameraWorker::start(camera.id.clone(), self.threads));
    }

    fn stop(&mut self) {
        if let Some(mut worker) = self.worker.take() {
            worker.stop();
        }
    }

    fn receive(&mut self, context: &egui::Context) {
        let mut latest = None;
        if let Some(worker) = &self.worker {
            while let Ok(event) = worker.receiver.try_recv() {
                latest = Some(event);
            }
        }
        match latest {
            Some(WorkerEvent::Frame(snapshot, pixels)) => {
                self.update_texture(
                    context,
                    snapshot.frame_width,
                    snapshot.frame_height,
                    &pixels,
                );
                self.status = format!("{:?}", snapshot.state);
                self.snapshot = snapshot;
            }
            Some(WorkerEvent::State(snapshot)) => {
                self.status = format!("{:?}", snapshot.state);
                self.snapshot = snapshot;
            }
            Some(WorkerEvent::Error(error)) => {
                self.status = error;
                self.stop();
            }
            None => {}
        }
    }

    fn update_texture(&mut self, context: &egui::Context, width: u32, height: u32, rgb: &[u8]) {
        if rgb.len() != width as usize * height as usize * 3 {
            return;
        }
        self.frame_size = [width, height];
        let image = egui::ColorImage::from_rgb([width as usize, height as usize], rgb);
        if let Some(texture) = &mut self.texture {
            texture.set(image, egui::TextureOptions::LINEAR);
        } else {
            self.texture = Some(context.load_texture(
                "camera-rs-preview",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
    }

    fn draw_overlay(&self, ui: &egui::Ui, image_rect: EguiRect) {
        let Some(observation) = &self.snapshot.observation else {
            return;
        };
        if self.frame_size[0] == 0 || self.frame_size[1] == 0 {
            return;
        }
        let map = |point: arcrelay_gaze::Point| {
            Pos2::new(
                image_rect.left() + point.x / self.frame_size[0] as f32 * image_rect.width(),
                image_rect.top() + point.y / self.frame_size[1] as f32 * image_rect.height(),
            )
        };
        let map_rect = |rect: arcrelay_gaze::Rect| {
            EguiRect::from_min_max(
                map(arcrelay_gaze::Point {
                    x: rect.x,
                    y: rect.y,
                }),
                map(arcrelay_gaze::Point {
                    x: rect.x + rect.width,
                    y: rect.y + rect.height,
                }),
            )
        };
        let painter = ui.painter();
        painter.rect_stroke(
            map_rect(observation.face),
            0.0,
            Stroke::new(2.0_f32, Color32::GREEN),
            egui::StrokeKind::Inside,
        );
        for landmark in &observation.landmarks {
            painter.circle_filled(map(*landmark), 1.5, Color32::YELLOW);
        }
        for (eye, open) in [
            (observation.left_eye, observation.left_eye_open),
            (observation.right_eye, observation.right_eye_open),
        ] {
            painter.rect_stroke(
                map_rect(eye),
                0.0,
                Stroke::new(
                    2.0_f32,
                    if open {
                        Color32::LIGHT_BLUE
                    } else {
                        Color32::RED
                    },
                ),
                egui::StrokeKind::Inside,
            );
        }
        if observation.left_eye_open && observation.right_eye_open {
            let left = observation.left_eye.center();
            let right = observation.right_eye.center();
            let start = map(arcrelay_gaze::Point {
                x: (left.x + right.x) * 0.5,
                y: (left.y + right.y) * 0.5,
            });
            let length = image_rect.width().min(image_rect.height()) * 0.22;
            painter.arrow(
                start,
                Vec2::new(observation.gaze.x * length, -observation.gaze.y * length),
                Stroke::new(3.0_f32, Color32::LIGHT_GREEN),
            );
        }
    }

    fn metrics(ui: &mut egui::Ui, observation: &GazeObservation) {
        ui.label(format!("推理：{:.1} ms", observation.inference_ms));
        ui.label(format!("人脸置信度：{:.3}", observation.face_confidence));
        ui.label(format!(
            "头部 yaw/pitch/roll：{:.1}° / {:.1}° / {:.1}°",
            observation.head_pose.yaw, observation.head_pose.pitch, observation.head_pose.roll
        ));
        ui.label(format!(
            "视线 x/y/z：{:+.3} / {:+.3} / {:+.3}",
            observation.gaze.x, observation.gaze.y, observation.gaze.z
        ));
    }
}

impl eframe::App for DemoApp {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        self.receive(context);
        context.request_repaint_after(Duration::from_millis(33));
        egui::SidePanel::left("controls")
            .resizable(false)
            .default_width(320.0)
            .show(context, |ui| {
                ui.heading("ArcRelay Gaze Lab");
                ui.label("camera-rs · Intel OMZ 五阶段流水线 · MNN");
                ui.label(format!("MNN {}", mnn_runtime::Runtime::native_version()));
                ui.separator();
                if ui.button("刷新摄像头").clicked() {
                    self.refresh_cameras();
                }
                egui::ComboBox::from_label("摄像头")
                    .selected_text(
                        self.cameras
                            .get(self.selected_camera)
                            .map(|camera| camera.name.as_str())
                            .unwrap_or("未发现"),
                    )
                    .show_ui(ui, |ui| {
                        for (index, camera) in self.cameras.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_camera, index, &camera.name);
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label("MNN 线程");
                    ui.add(egui::DragValue::new(&mut self.threads).range(1..=16));
                });
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(self.worker.is_none(), egui::Button::new("启动"))
                        .clicked()
                    {
                        self.start();
                    }
                    if ui
                        .add_enabled(self.worker.is_some(), egui::Button::new("停止"))
                        .clicked()
                    {
                        self.stop();
                        self.status = "已停止".into();
                    }
                });
                ui.separator();
                ui.label(&self.status);
                ui.label(format!(
                    "采集 / 推理 / 丢帧：{} / {} / {}",
                    self.snapshot.captured_frames,
                    self.snapshot.inferred_frames,
                    self.snapshot.dropped_frames
                ));
                if let Some(observation) = &self.snapshot.observation {
                    ui.separator();
                    Self::metrics(ui, observation);
                }
            });
        egui::CentralPanel::default().show(context, |ui| {
            let Some(texture) = &self.texture else {
                ui.centered_and_justified(|ui| ui.label("摄像头画面会显示在这里。"));
                return;
            };
            let available = ui.available_size();
            let source = Vec2::new(self.frame_size[0] as f32, self.frame_size[1] as f32);
            let scale = (available.x / source.x).min(available.y / source.y);
            let response = ui.add(egui::Image::new((texture.id(), source * scale)));
            self.draw_overlay(ui, response.rect);
        });
    }
}

impl Drop for DemoApp {
    fn drop(&mut self) {
        self.stop();
    }
}

fn main() -> eframe::Result<()> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if arguments.iter().any(|argument| argument == "--self-test") {
        match arcrelay_gaze::GazeEngine::bundled(4).and_then(|engine| engine.self_test()) {
            Ok(report) => {
                println!("MNN gaze pipeline self-test passed");
                for line in report {
                    println!("  {line}");
                }
                return Ok(());
            }
            Err(error) => {
                eprintln!("MNN gaze pipeline self-test failed: {error}");
                std::process::exit(1);
            }
        }
    }
    if let Some(position) = arguments
        .iter()
        .position(|argument| argument == "--camera-test")
    {
        let requested_id = arguments.get(position + 1).cloned();
        if let Err(error) = run_camera_test(requested_id.as_deref()) {
            eprintln!("camera-rs gaze test failed: {error}");
            std::process::exit(1);
        }
        return Ok(());
    }
    eframe::run_native(
        "ArcRelay Gaze Lab",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1180.0, 760.0])
                .with_min_inner_size([800.0, 540.0]),
            ..Default::default()
        },
        Box::new(|_| Ok(Box::new(DemoApp::default()))),
    )
}

fn run_camera_test(requested_id: Option<&str>) -> Result<(), String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(async {
        let tracker = GazeTracker::with_bundled_models(4).map_err(|error| error.to_string())?;
        let cameras = tracker.cameras().await.map_err(|error| error.to_string())?;
        let camera = requested_id
            .and_then(|id| cameras.iter().find(|camera| camera.id == id))
            .or_else(|| cameras.first())
            .ok_or_else(|| "camera-rs did not enumerate any camera".to_string())?;
        println!("camera: {} ({})", camera.name, camera.id);
        let mut session = tracker
            .start(&camera.id, TrackerConfig::default())
            .await
            .map_err(|error| error.to_string())?;
        let mut snapshots = session.subscribe_snapshots();
        let result = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                snapshots
                    .changed()
                    .await
                    .map_err(|error| error.to_string())?;
                let snapshot = snapshots.borrow().clone();
                if let Some(error) = snapshot.error {
                    return Err(error);
                }
                if snapshot.inferred_frames >= 20 {
                    return Ok(snapshot);
                }
            }
        })
        .await
        .map_err(|_| "timed out waiting for 20 inferred frames".to_string())?;
        session.stop().await.map_err(|error| error.to_string())?;
        let snapshot = result?;
        println!(
            "captured={}, inferred={}, dropped={}, face={}, last_inference_ms={:.1}",
            snapshot.captured_frames,
            snapshot.inferred_frames,
            snapshot.dropped_frames,
            snapshot.observation.is_some(),
            snapshot
                .observation
                .as_ref()
                .map_or(0.0, |observation| observation.inference_ms)
        );
        Ok(())
    })
}
