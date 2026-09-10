use arcrelay_input::DeskPointUm;
use serde::{Deserialize, Serialize};

use crate::{Error, GazeObservation, Result};

const FEATURE_COUNT: usize = 8;
const MINIMUM_SAMPLES: usize = 9;

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
}

impl CalibrationProfile {
    /// Project a usable gaze observation into ArcRelay's physical desk space.
    pub fn project(&self, observation: &GazeObservation) -> Result<DeskPointUm> {
        if !observation.usable_for_targeting() {
            return Err(Error::Calibration(
                "observation is not usable for calibration mapping".into(),
            ));
        }
        let values = features(observation);
        let x = dot(&self.coefficients_x, &values);
        let y = dot(&self.coefficients_y, &values);
        if !x.is_finite() || !y.is_finite() {
            return Err(Error::Calibration(
                "calibration produced a non-finite point".into(),
            ));
        }
        Ok(DeskPointUm {
            x: x.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64,
            y: y.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64,
        })
    }
}

/// Accumulates screen points and fits a regularized affine/polynomial mapping.
#[derive(Clone, Debug)]
pub struct Calibrator {
    camera_id: String,
    layout_signature: String,
    samples: Vec<CalibrationSample>,
}

impl Calibrator {
    #[must_use]
    pub fn new(camera_id: impl Into<String>, layout_signature: impl Into<String>) -> Self {
        Self {
            camera_id: camera_id.into(),
            layout_signature: layout_signature.into(),
            samples: Vec::new(),
        }
    }

    pub fn push(&mut self, observation: &GazeObservation, target: DeskPointUm) -> Result<()> {
        if !observation.usable_for_targeting() {
            return Err(Error::Calibration(
                "both eyes and a confident face are required".into(),
            ));
        }
        self.samples
            .push(CalibrationSample::from_observation(observation, target));
        Ok(())
    }

    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn finish(self) -> Result<CalibrationProfile> {
        if self.samples.len() < MINIMUM_SAMPLES {
            return Err(Error::Calibration(format!(
                "at least {MINIMUM_SAMPLES} samples are required, got {}",
                self.samples.len()
            )));
        }
        let coefficients_x = fit(&self.samples, |sample| sample.desk_x_um as f64)?;
        let coefficients_y = fit(&self.samples, |sample| sample.desk_y_um as f64)?;
        let squared_error = self
            .samples
            .iter()
            .map(|sample| {
                let dx = dot(&coefficients_x, &sample.features) - sample.desk_x_um as f64;
                let dy = dot(&coefficients_y, &sample.features) - sample.desk_y_um as f64;
                dx * dx + dy * dy
            })
            .sum::<f64>();
        Ok(CalibrationProfile {
            version: 1,
            camera_id: self.camera_id,
            layout_signature: self.layout_signature,
            coefficients_x,
            coefficients_y,
            rms_error_um: (squared_error / self.samples.len() as f64).sqrt(),
            sample_count: self.samples.len(),
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

fn solve(
    mut matrix: [[f64; FEATURE_COUNT]; FEATURE_COUNT],
    mut rhs: [f64; FEATURE_COUNT],
) -> Result<[f64; FEATURE_COUNT]> {
    for pivot in 0..FEATURE_COUNT {
        let best = (pivot..FEATURE_COUNT)
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
        for column in pivot..FEATURE_COUNT {
            matrix[pivot][column] /= scale;
        }
        rhs[pivot] /= scale;
        for row in 0..FEATURE_COUNT {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..FEATURE_COUNT {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Ok(rhs)
}

fn dot(left: &[f64; FEATURE_COUNT], right: &[f64; FEATURE_COUNT]) -> f64 {
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
        let calibrator = Calibrator {
            camera_id: "camera".into(),
            layout_signature: "layout".into(),
            samples,
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
}
