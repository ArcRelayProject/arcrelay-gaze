use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{PresenceEnrollmentStatus, PresenceObservation};

/// Point in a captured camera frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub(crate) fn midpoint(self, other: Self) -> Self {
        Self {
            x: (self.x + other.x) * 0.5,
            y: (self.y + other.y) * 0.5,
        }
    }

    pub(crate) fn distance(self, other: Self) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// Rectangle in a captured camera frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub(crate) fn clamp(self, image_width: u32, image_height: u32) -> Option<Self> {
        let left = self.x.max(0.0).min(image_width as f32);
        let top = self.y.max(0.0).min(image_height as f32);
        let right = (self.x + self.width).max(left).min(image_width as f32);
        let bottom = (self.y + self.height).max(top).min(image_height as f32);
        (right - left >= 2.0 && bottom - top >= 2.0).then_some(Self {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        })
    }

    pub fn center(self) -> Point {
        Point {
            x: self.x + self.width * 0.5,
            y: self.y + self.height * 0.5,
        }
    }
}

/// Normalized 3D gaze vector in camera coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub(crate) fn normalized(self) -> Self {
        let length = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if length <= f32::EPSILON {
            return self;
        }
        Self {
            x: self.x / length,
            y: self.y / length,
            z: self.z / length,
        }
    }
}

/// Estimated head orientation in degrees.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HeadPose {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

/// One complete inference result before desktop calibration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GazeObservation {
    pub frame_width: u32,
    pub frame_height: u32,
    pub face: Rect,
    pub face_confidence: f32,
    pub landmarks: Vec<Point>,
    pub left_eye: Rect,
    pub right_eye: Rect,
    pub left_eye_open: bool,
    pub right_eye_open: bool,
    pub head_pose: HeadPose,
    pub gaze: Vec3,
    pub inference_ms: f32,
}

impl GazeObservation {
    #[must_use]
    pub fn usable_for_targeting(&self) -> bool {
        self.left_eye_open
            && self.right_eye_open
            && self.face_confidence >= 0.6
            && self.gaze.x.is_finite()
            && self.gaze.y.is_finite()
            && self.gaze.z.is_finite()
    }

    #[must_use]
    pub fn usable_for_head_targeting(&self) -> bool {
        self.face_confidence >= 0.6
            && self.head_pose.yaw.is_finite()
            && self.head_pose.pitch.is_finite()
            && self.head_pose.roll.is_finite()
            && self.head_pose.yaw.abs() <= 80.0
            && self.head_pose.pitch.abs() <= 60.0
            && self.head_pose.roll.abs() <= 50.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TargetingSource {
    #[default]
    Eye,
    HeadFallback,
}

/// A calibrated point on one ArcRelay display.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GazeTarget {
    pub device_id: String,
    pub display_id: String,
    pub desk_x_um: i64,
    pub desk_y_um: i64,
    pub logical_x: f64,
    pub logical_y: f64,
    pub confidence: f32,
    #[serde(default)]
    pub source: TargetingSource,
}

/// Target after temporal hysteresis and dwell filtering.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StabilizedTarget {
    pub target: GazeTarget,
    pub stable_for_ms: u64,
    pub changed: bool,
}

/// High-level tracker lifecycle state exposed to the desktop host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TrackingState {
    #[default]
    Idle,
    Starting,
    Tracking,
    FaceLost,
    Uncalibrated,
    Stopping,
    Stopped,
    Failed,
}

/// Latest state suitable for Tauri IPC and diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackerSnapshot {
    pub state: TrackingState,
    pub camera_id: Option<String>,
    pub camera_name: Option<String>,
    pub frame_width: u32,
    pub frame_height: u32,
    pub captured_frames: u64,
    pub inferred_frames: u64,
    pub detected_frames: u64,
    pub identity_frames: u64,
    pub tracked_frames: u64,
    pub dropped_frames: u64,
    pub observation: Option<GazeObservation>,
    pub target: Option<StabilizedTarget>,
    pub presence: PresenceObservation,
    pub presence_enrollment: PresenceEnrollmentStatus,
    pub error: Option<String>,
}

impl Default for TrackerSnapshot {
    fn default() -> Self {
        Self {
            state: TrackingState::Idle,
            camera_id: None,
            camera_name: None,
            frame_width: 0,
            frame_height: 0,
            captured_frames: 0,
            inferred_frames: 0,
            detected_frames: 0,
            identity_frames: 0,
            tracked_frames: 0,
            dropped_frames: 0,
            observation: None,
            target: None,
            presence: PresenceObservation::default(),
            presence_enrollment: PresenceEnrollmentStatus::default(),
            error: None,
        }
    }
}

/// In-process event. Preview pixels are never serialized or persisted.
#[derive(Clone, Debug)]
pub struct TrackerEvent {
    pub snapshot: TrackerSnapshot,
    pub rgb_preview: Option<Arc<[u8]>>,
}
