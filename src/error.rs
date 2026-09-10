/// Errors produced by gaze inference, calibration, capture, and mapping.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid gaze configuration: {0}")]
    InvalidConfig(String),
    #[error("gaze model error: {0}")]
    Model(String),
    #[error("gaze calibration error: {0}")]
    Calibration(String),
    #[error("workspace mapping error: {0}")]
    Mapping(String),
    #[cfg(feature = "camera")]
    #[error("camera error: {0}")]
    Camera(#[from] camera::CameraError),
    #[error("gaze worker stopped")]
    WorkerStopped,
    #[error("gaze worker failed: {0}")]
    Worker(String),
}

/// Result type used by this crate.
pub type Result<T> = std::result::Result<T, Error>;
