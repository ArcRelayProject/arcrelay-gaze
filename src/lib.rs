//! Local-first gaze estimation, calibration, and ArcRelay display targeting.
//!
//! Camera frames stay in process memory and are never persisted or transmitted.
//! Face detection and optional owner-presence matching are local preprocessing
//! stages. Raw frames and embeddings are never serialized or transmitted.

mod calibration;
mod error;
mod image_ops;
mod mapping;
mod model;
mod model_bundle;
mod observation_filter;
mod pipeline;
mod presence;
mod smoothing;
#[cfg(feature = "camera")]
mod tracker;
mod types;

pub use calibration::{CalibrationProfile, CalibrationSample, Calibrator, HeadRegionProfile};
pub use error::{Error, Result};
pub use mapping::{layout_signature, WorkspaceMapper};
pub use model_bundle::{ModelFile, OwnedModelBundle, MODEL_BUNDLE_VERSION, MODEL_FILES};
pub use observation_filter::ObservationFilterConfig;
pub use pipeline::{GazeEngine, ModelBundle};
pub use presence::{
    PresenceEnrollmentStatus, PresenceObservation, PresencePose, PresencePoseTemplate,
    PresenceProfile, PresenceState, FACE_EMBEDDING_DIMENSIONS,
    PRESENCE_ENROLLMENT_REQUIRED_SAMPLES, PRESENCE_PROFILE_VERSION,
};
pub use smoothing::{StabilizerConfig, TargetStabilizer};
#[cfg(feature = "camera")]
pub use tracker::{CameraDescriptor, GazeTracker, TrackerConfig, TrackerSession};
pub use types::{
    GazeObservation, GazeTarget, HeadPose, Point, Rect, StabilizedTarget, TargetingSource,
    TrackerEvent, TrackerSnapshot, TrackingState, Vec3,
};
