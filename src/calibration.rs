use arcrelay_input::DeskPointUm;
use serde::{Deserialize, Serialize};

use crate::{Error, GazeObservation, Result};

const FEATURE_COUNT: usize = 8;
const HEAD_FEATURE_COUNT: usize = 7;
const MINIMUM_SAMPLES: usize = 9;

#[derive(Clone, Debug)]
struct HeadCalibrationSample {
    features: [f64; HEAD_FEATURE_COUNT],
    desk_x_um: i64,
    desk_y_um: i64,
}

/// One known point observed during calibration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationSample {
    pub features: [f64; FEATURE_COUNT],
    pub desk_x_um: i64,
    pub desk_y_um: i64,
}

impl CalibrationSample {
    #[must_use]
    pub fn from_observation(observation: &GazeObservation, target: DeskPointUm) -> Self {
        Self {
            features: features(observation),
            desk_x_um: target.x,
            desk_y_um: target.y,
        }
    }
}

/// Persistable calibration tied to one camera and one physical display layout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationProfile {
    pub version: u32,
    pub camera_id: String,
    pub layout_signature: String,
    pub coefficients_x: [f64; FEATURE_COUNT],
    pub coefficients_y: [f64; FEATURE_COUNT],
    pub rms_error_um: f64,
    pub sample_count: usize,
    #[serde(default)]
    pub eye_sample_count: Option<usize>,
    #[serde(default)]
    pub head_coefficients_x: Option<[f64; HEAD_FEATURE_COUNT]>,
    #[serde(default)]
    pub head_coefficients_y: Option<[f64; HEAD_FEATURE_COUNT]>,
    #[serde(default)]
    pub head_rms_error_um: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProjectionSource {
    Eye,
    HeadFallback,
}

impl CalibrationProfile {
    /// Project a usable gaze observation into ArcRelay's physical desk space.
    pub fn project(&self, observation: &GazeObservation) -> Result<DeskPointUm> {
        self.project_with_source(observation)
            .map(|(point, _)| point)
    }

    pub(crate) fn project_with_source(
        &self,
        observation: &GazeObservation,
    ) -> Result<(DeskPointUm, ProjectionSource)> {
        let eye_available = self.eye_sample_count.unwrap_or(self.sample_count) >= MINIMUM_SAMPLES;
        let (x, y, source) = if eye_available && observation.usable_for_targeting() {
            let values = features(observation);
            (
                dot(&self.coefficients_x, &values),
                dot(&self.coefficients_y, &values),
                ProjectionSource::Eye,
            )
        } else if observation.usable_for_head_targeting() {
            let coefficients_x = self.head_coefficients_x.as_ref().ok_or_else(|| {
                Error::Calibration("head-direction fallback was not calibrated".into())
            })?;
            let coefficients_y = self.head_coefficients_y.as_ref().ok_or_else(|| {
                Error::Calibration("head-direction fallback was not calibrated".into())
            })?;
            let values = head_features(observation);
            (
                dot(coefficients_x, &values),
                dot(coefficients_y, &values),
                ProjectionSource::HeadFallback,
            )
        } else {
            return Err(Error::Calibration(
                "observation is not usable for eye or head mapping".into(),
            ));
        };
        if !x.is_finite() || !y.is_finite() {
            return Err(Error::Calibration(
                "calibration produced a non-finite point".into(),
            ));
        }
        Ok((
            DeskPointUm {
                x: x.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64,
                y: y.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64,
            },
            source,
        ))
    }
}

/// Accumulates screen points and fits a regularized affine/polynomial mapping.
#[derive(Clone, Debug)]
pub struct Calibrator {
    camera_id: String,
    layout_signature: String,
    samples: Vec<CalibrationSample>,
    head_samples: Vec<HeadCalibrationSample>,
}

impl Calibrator {
    #[must_use]
    pub fn new(camera_id: impl Into<String>, layout_signature: impl Into<String>) -> Self {
        Self {
            camera_id: camera_id.into(),
            layout_signature: layout_signature.into(),
            samples: Vec::new(),
            head_samples: Vec::new(),
        }
    }

    pub fn push(&mut self, observation: &GazeObservation, target: DeskPointUm) -> Result<()> {
        if !observation.usable_for_head_targeting() {
            return Err(Error::Calibration(
                "a confident face and usable head pose are required".into(),
            ));
        }
        self.head_samples.push(HeadCalibrationSample {
            features: head_features(observation),
            desk_x_um: target.x,
            desk_y_um: target.y,
        });
        if observation.usable_for_targeting() {
            self.samples
                .push(CalibrationSample::from_observation(observation, target));
        }
        Ok(())
    }

    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.head_samples.len()
    }

    pub fn finish(self) -> Result<CalibrationProfile> {
        if self.head_samples.len() < MINIMUM_SAMPLES {
            return Err(Error::Calibration(format!(
                "at least {MINIMUM_SAMPLES} samples are required, got {}",
                self.head_samples.len()
            )));
        }
        let eye_sample_count = self.samples.len();
        let (coefficients_x, coefficients_y, eye_rms_error_um) =
            if eye_sample_count >= MINIMUM_SAMPLES {
                let x = fit(&self.samples, |sample| sample.desk_x_um as f64)?;
                let y = fit(&self.samples, |sample| sample.desk_y_um as f64)?;
                let error = rms_error(&self.samples, &x, &y);
                (x, y, Some(error))
            } else {
                ([0.0; FEATURE_COUNT], [0.0; FEATURE_COUNT], None)
            };
        let head_coefficients_x = fit_head(&self.head_samples, |sample| sample.desk_x_um as f64)?;
        let head_coefficients_y = fit_head(&self.head_samples, |sample| sample.desk_y_um as f64)?;
        let head_rms_error_um = rms_head_error(
            &self.head_samples,
            &head_coefficients_x,
            &head_coefficients_y,
        );
        let rms_error_um = eye_rms_error_um.unwrap_or(head_rms_error_um);
        Ok(CalibrationProfile {
            version: 2,
            camera_id: self.camera_id,
            layout_signature: self.layout_signature,
            coefficients_x,
            coefficients_y,
            rms_error_um,
            sample_count: self.head_samples.len(),
            eye_sample_count: Some(eye_sample_count),
            head_coefficients_x: Some(head_coefficients_x),
            head_coefficients_y: Some(head_coefficients_y),
            head_rms_error_um: Some(head_rms_error_um),
        })
    }
}

fn features(observation: &GazeObservation) -> [f64; FEATURE_COUNT] {
    let center = observation.face.center();
    let width = observation.frame_width.max(1) as f64;
    let height = observation.frame_height.max(1) as f64;
    let gaze_x = f64::from(observation.gaze.x);
    let gaze_y = f64::from(observation.gaze.y);
    [
        1.0,
        gaze_x,
        gaze_y,
        f64::from(observation.head_pose.yaw) / 45.0,
        f64::from(observation.head_pose.pitch) / 45.0,
        f64::from(center.x) / width,
        f64::from(center.y) / height,
        gaze_x * gaze_y,
    ]
}

fn head_features(observation: &GazeObservation) -> [f64; HEAD_FEATURE_COUNT] {
    let center = observation.face.center();
    let width = observation.frame_width.max(1) as f64;
    let height = observation.frame_height.max(1) as f64;
    let yaw = f64::from(observation.head_pose.yaw) / 45.0;
    let pitch = f64::from(observation.head_pose.pitch) / 45.0;
    [
        1.0,
        yaw,
        pitch,
        f64::from(center.x) / width,
        f64::from(center.y) / height,
        yaw * pitch,
        f64::from(observation.face.width * observation.face.height) / (width * height),
    ]
}

fn fit(
    samples: &[CalibrationSample],
    target: impl Fn(&CalibrationSample) -> f64,
) -> Result<[f64; FEATURE_COUNT]> {
    let mut matrix = [[0.0; FEATURE_COUNT]; FEATURE_COUNT];
    let mut rhs = [0.0; FEATURE_COUNT];
    for sample in samples {
        let y = target(sample);
        for row in 0..FEATURE_COUNT {
            rhs[row] += sample.features[row] * y;
            for column in 0..FEATURE_COUNT {
                matrix[row][column] += sample.features[row] * sample.features[column];
            }
        }
    }
    for (index, row) in matrix.iter_mut().enumerate() {
        row[index] += 1e-6;
    }
    solve(matrix, rhs)
}

fn fit_head(
    samples: &[HeadCalibrationSample],
    target: impl Fn(&HeadCalibrationSample) -> f64,
) -> Result<[f64; HEAD_FEATURE_COUNT]> {
    fit_features(samples, |sample| &sample.features, target)
}

fn fit_features<const N: usize, T>(
    samples: &[T],
    features: impl Fn(&T) -> &[f64; N],
    target: impl Fn(&T) -> f64,
) -> Result<[f64; N]> {
    let mut matrix = [[0.0; N]; N];
    let mut rhs = [0.0; N];
    for sample in samples {
        let values = features(sample);
        let y = target(sample);
        for row in 0..N {
            rhs[row] += values[row] * y;
            for column in 0..N {
                matrix[row][column] += values[row] * values[column];
            }
        }
    }
    for (index, row) in matrix.iter_mut().enumerate() {
        row[index] += 1e-4;
    }
    solve(matrix, rhs)
}

fn rms_error(
    samples: &[CalibrationSample],
    x: &[f64; FEATURE_COUNT],
    y: &[f64; FEATURE_COUNT],
) -> f64 {
    let sum = samples
        .iter()
        .map(|sample| {
            let dx = dot(x, &sample.features) - sample.desk_x_um as f64;
            let dy = dot(y, &sample.features) - sample.desk_y_um as f64;
            dx * dx + dy * dy
        })
        .sum::<f64>();
    (sum / samples.len() as f64).sqrt()
}

fn rms_head_error(
    samples: &[HeadCalibrationSample],
    x: &[f64; HEAD_FEATURE_COUNT],
    y: &[f64; HEAD_FEATURE_COUNT],
) -> f64 {
    let sum = samples
        .iter()
        .map(|sample| {
            let dx = dot(x, &sample.features) - sample.desk_x_um as f64;
            let dy = dot(y, &sample.features) - sample.desk_y_um as f64;
            dx * dx + dy * dy
        })
        .sum::<f64>();
    (sum / samples.len() as f64).sqrt()
}

fn solve<const N: usize>(mut matrix: [[f64; N]; N], mut rhs: [f64; N]) -> Result<[f64; N]> {
    for pivot in 0..N {
        let best = (pivot..N)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .abs()
                    .total_cmp(&matrix[right][pivot].abs())
            })
            .expect("nonempty pivot range");
        if matrix[best][pivot].abs() < 1e-12 {
            return Err(Error::Calibration(
                "calibration samples do not span the workspace".into(),
            ));
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let scale = matrix[pivot][pivot];
        for column in pivot..N {
            matrix[pivot][column] /= scale;
        }
        rhs[pivot] /= scale;
        for row in 0..N {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..N {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Ok(rhs)
}

fn dot<const N: usize>(left: &[f64; N], right: &[f64; N]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_a_full_rank_synthetic_workspace() {
        let expected_x = [10.0, 2.0, -3.0, 4.0, 0.5, 1.5, -2.5, 3.5];
        let expected_y = [-5.0, 1.0, 2.0, -1.0, 3.0, 2.5, 1.5, -2.0];
        let mut samples = Vec::new();
        for index in 0..24 {
            let t = index as f64 / 7.0;
            let features = [
                1.0,
                t,
                t * t,
                (t * 1.7).sin(),
                (t * 0.9).cos(),
                (index % 3) as f64,
                (index % 5) as f64,
                (index % 7) as f64,
            ];
            samples.push(CalibrationSample {
                features,
                desk_x_um: dot(&expected_x, &features).round() as i64,
                desk_y_um: dot(&expected_y, &features).round() as i64,
            });
        }
        let head_samples = samples
            .iter()
            .map(|sample| HeadCalibrationSample {
                features: sample.features[..HEAD_FEATURE_COUNT].try_into().unwrap(),
                desk_x_um: sample.desk_x_um,
                desk_y_um: sample.desk_y_um,
            })
            .collect();
        let calibrator = Calibrator {
            camera_id: "camera".into(),
            layout_signature: "layout".into(),
            samples,
            head_samples,
        };
        let profile = calibrator.finish().expect("fit calibration");
        assert_eq!(profile.sample_count, 24);
        assert!(profile.rms_error_um < 1.0);
        assert!(profile.coefficients_x.iter().all(|value| value.is_finite()));
        assert!(profile.coefficients_y.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rejects_an_underdetermined_calibration() {
        let calibrator = Calibrator::new("camera", "layout");
        assert!(matches!(
            calibrator.finish(),
            Err(Error::Calibration(message)) if message.contains("at least")
        ));
    }

    #[test]
    fn head_only_profile_has_finite_serializable_error() {
        let mut calibrator = Calibrator::new("camera", "layout");
        for index in 0..12 {
            let yaw = -36.0 + index as f32 * 6.0;
            let pitch = -18.0 + (index % 4) as f32 * 12.0;
            let observation = GazeObservation {
                frame_width: 1280,
                frame_height: 720,
                face: crate::Rect {
                    x: 420.0 + index as f32 * 5.0,
                    y: 180.0 + (index % 3) as f32 * 8.0,
                    width: 260.0,
                    height: 300.0,
                },
                face_confidence: 0.95,
                landmarks: Vec::new(),
                left_eye: crate::Rect::default(),
                right_eye: crate::Rect::default(),
                left_eye_open: false,
                right_eye_open: false,
                head_pose: crate::HeadPose {
                    yaw,
                    pitch,
                    roll: 0.0,
                },
                gaze: crate::Vec3::default(),
                inference_ms: 4.0,
            };
            calibrator
                .push(
                    &observation,
                    DeskPointUm {
                        x: index as i64 * 100_000,
                        y: (index % 4) as i64 * 120_000,
                    },
                )
                .expect("accept head-only sample");
        }

        let profile = calibrator.finish().expect("fit head-only profile");
        assert_eq!(profile.eye_sample_count, Some(0));
        assert!(profile.rms_error_um.is_finite());
        serde_json::to_vec(&profile).expect("serialize head-only profile");
    }
}
