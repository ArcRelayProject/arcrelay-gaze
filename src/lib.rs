//! Local-first gaze estimation, calibration, and ArcRelay display targeting.
//!
//! Camera frames stay in process memory and are never persisted or transmitted.
//! Face detection is an internal preprocessing stage; this crate performs no
//! face recognition or identity matching.

mod calibration;
mod error;
mod image_ops;
mod mapping;
mod model;
mod pipeline;
mod smoothing;
#[cfg(feature = "camera")]
mod tracker;
mod types;

pub use calibration::{CalibrationProfile, CalibrationSample, Calibrator};
pub use error::{Error, Result};
pub use mapping::{layout_signature, WorkspaceMapper};
pub use pipeline::{GazeEngine, ModelBundle};
pub use smoothing::{StabilizerConfig, TargetStabilizer};
#[cfg(feature = "camera")]
pub use tracker::{CameraDescriptor, GazeTracker, TrackerConfig, TrackerSession};
pub use types::{
    GazeObservation, GazeTarget, HeadPose, Point, Rect, StabilizedTarget, TargetingSource,
    TrackerEvent, TrackerSnapshot, TrackingState, Vec3,
};
