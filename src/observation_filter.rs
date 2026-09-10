use std::time::{Duration, Instant};

use crate::{GazeObservation, HeadPose, Rect, Vec3};

/// Adaptive smoothing policy for the continuous values emitted by the gaze model.
///
/// Low motion is smoothed aggressively so calibration can settle. Fast eye or
/// head motion raises the cutoff automatically, keeping cross-screen turns
/// responsive instead of adding a fixed moving-average delay.
#[derive(Clone, Debug)]
pub struct ObservationFilterConfig {
    pub gaze_min_cutoff_hz: f32,
    pub gaze_beta: f32,
    pub head_min_cutoff_hz: f32,
    pub head_beta: f32,
    pub geometry_min_cutoff_hz: f32,
    pub geometry_beta: f32,
    pub confidence_min_cutoff_hz: f32,
    pub derivative_cutoff_hz: f32,
    pub reset_after: Duration,
}

impl Default for ObservationFilterConfig {
    fn default() -> Self {
        Self {
            gaze_min_cutoff_hz: 1.35,
            gaze_beta: 0.55,
            head_min_cutoff_hz: 0.8,
            head_beta: 0.02,
            geometry_min_cutoff_hz: 0.85,
            geometry_beta: 0.002,
            confidence_min_cutoff_hz: 1.6,
            derivative_cutoff_hz: 1.0,
            reset_after: Duration::from_millis(450),
        }
    }
}

pub(crate) struct ObservationFilter {
    config: ObservationFilterConfig,
    last_observation_at: Option<Instant>,
    gaze: Vec3Filter,
    head: HeadPoseFilter,
    face: RectFilter,
    left_eye: RectFilter,
    right_eye: RectFilter,
    confidence: OneEuro,
    left_eye_open: BoolDebounce,
    right_eye_open: BoolDebounce,
}

impl ObservationFilter {
    pub(crate) fn new(config: ObservationFilterConfig) -> Self {
        Self {
            config,
            last_observation_at: None,
            gaze: Vec3Filter::default(),
            head: HeadPoseFilter::default(),
            face: RectFilter::default(),
            left_eye: RectFilter::default(),
            right_eye: RectFilter::default(),
            confidence: OneEuro::default(),
            left_eye_open: BoolDebounce::default(),
            right_eye_open: BoolDebounce::default(),
        }
    }

    pub(crate) fn update(
        &mut self,
        mut observation: GazeObservation,
        at: Instant,
    ) -> GazeObservation {
        if self
            .last_observation_at
            .is_some_and(|last| at.saturating_duration_since(last) > self.config.reset_after)
        {
            self.reset();
        }
        self.last_observation_at = Some(at);

        observation.gaze = self.gaze.update(
            observation.gaze,
            at,
            self.config.gaze_min_cutoff_hz,
            self.config.gaze_beta,
            self.config.derivative_cutoff_hz,
        );
        observation.head_pose = self.head.update(
            observation.head_pose,
            at,
            self.config.head_min_cutoff_hz,
            self.config.head_beta,
            self.config.derivative_cutoff_hz,
        );
        observation.face = self.face.update(
            observation.face,
            at,
            self.config.geometry_min_cutoff_hz,
            self.config.geometry_beta,
            self.config.derivative_cutoff_hz,
        );
        observation.left_eye = self.left_eye.update(
            observation.left_eye,
            at,
            self.config.geometry_min_cutoff_hz,
            self.config.geometry_beta,
            self.config.derivative_cutoff_hz,
        );
        observation.right_eye = self.right_eye.update(
            observation.right_eye,
            at,
            self.config.geometry_min_cutoff_hz,
            self.config.geometry_beta,
            self.config.derivative_cutoff_hz,
        );
        observation.face_confidence = self.confidence.update(
            observation.face_confidence,
            at,
            self.config.confidence_min_cutoff_hz,
            0.0,
            self.config.derivative_cutoff_hz,
        );
        observation.left_eye_open = self.left_eye_open.update(observation.left_eye_open);
        observation.right_eye_open = self.right_eye_open.update(observation.right_eye_open);
        observation
    }

    fn reset(&mut self) {
        self.last_observation_at = None;
        self.gaze = Vec3Filter::default();
        self.head = HeadPoseFilter::default();
        self.face = RectFilter::default();
        self.left_eye = RectFilter::default();
        self.right_eye = RectFilter::default();
        self.confidence = OneEuro::default();
        self.left_eye_open = BoolDebounce::default();
        self.right_eye_open = BoolDebounce::default();
    }
}

#[derive(Default)]
struct OneEuro {
    filtered: Option<f32>,
    filtered_derivative: Option<f32>,
    last_raw: Option<f32>,
    last_at: Option<Instant>,
    raw_window: [f32; 3],
    raw_count: usize,
    raw_cursor: usize,
}

impl OneEuro {
    fn update(
        &mut self,
        raw: f32,
        at: Instant,
        min_cutoff_hz: f32,
        beta: f32,
        derivative_cutoff_hz: f32,
    ) -> f32 {
        if !raw.is_finite() {
            *self = Self::default();
            return raw;
        }
        let raw = self.median_prefilter(raw);
        let (Some(previous_raw), Some(previous_at), Some(previous_filtered)) =
            (self.last_raw, self.last_at, self.filtered)
        else {
            self.filtered = Some(raw);
            self.filtered_derivative = Some(0.0);
            self.last_raw = Some(raw);
            self.last_at = Some(at);
            return raw;
        };
        let dt = at
            .saturating_duration_since(previous_at)
            .as_secs_f32()
            .clamp(1.0 / 240.0, 0.25);
        let derivative = (raw - previous_raw) / dt;
        let derivative_alpha = smoothing_alpha(derivative_cutoff_hz, dt);
        let filtered_derivative = low_pass(
            self.filtered_derivative.unwrap_or(derivative),
            derivative,
            derivative_alpha,
        );
        let cutoff = min_cutoff_hz + beta * filtered_derivative.abs();
        let filtered = low_pass(previous_filtered, raw, smoothing_alpha(cutoff, dt));
        self.filtered = Some(filtered);
        self.filtered_derivative = Some(filtered_derivative);
        self.last_raw = Some(raw);
        self.last_at = Some(at);
        filtered
    }

    fn median_prefilter(&mut self, raw: f32) -> f32 {
        self.raw_window[self.raw_cursor] = raw;
        self.raw_cursor = (self.raw_cursor + 1) % self.raw_window.len();
        self.raw_count = (self.raw_count + 1).min(self.raw_window.len());
        if self.raw_count < self.raw_window.len() {
            return raw;
        }
        let mut sorted = self.raw_window;
        sorted.sort_by(f32::total_cmp);
        sorted[1]
    }
}

fn smoothing_alpha(cutoff_hz: f32, dt: f32) -> f32 {
    let cutoff_hz = cutoff_hz.max(0.001);
    let time_constant = 1.0 / (std::f32::consts::TAU * cutoff_hz);
    (dt / (dt + time_constant)).clamp(0.0, 1.0)
}

fn low_pass(previous: f32, current: f32, alpha: f32) -> f32 {
    previous + alpha * (current - previous)
}

#[derive(Default)]
struct Vec3Filter {
    x: OneEuro,
    y: OneEuro,
    z: OneEuro,
}

impl Vec3Filter {
    fn update(
        &mut self,
        value: Vec3,
        at: Instant,
        min_cutoff_hz: f32,
        beta: f32,
        derivative_cutoff_hz: f32,
    ) -> Vec3 {
        let filtered = Vec3 {
            x: self
                .x
                .update(value.x, at, min_cutoff_hz, beta, derivative_cutoff_hz),
            y: self
                .y
                .update(value.y, at, min_cutoff_hz, beta, derivative_cutoff_hz),
            z: self
                .z
                .update(value.z, at, min_cutoff_hz, beta, derivative_cutoff_hz),
        };
        if filtered.x.is_finite() && filtered.y.is_finite() && filtered.z.is_finite() {
            filtered.normalized()
        } else {
            filtered
        }
    }
}

#[derive(Default)]
struct HeadPoseFilter {
    yaw: OneEuro,
    pitch: OneEuro,
    roll: OneEuro,
}

impl HeadPoseFilter {
    fn update(
        &mut self,
        value: HeadPose,
        at: Instant,
        min_cutoff_hz: f32,
        beta: f32,
        derivative_cutoff_hz: f32,
    ) -> HeadPose {
        HeadPose {
            yaw: self
                .yaw
                .update(value.yaw, at, min_cutoff_hz, beta, derivative_cutoff_hz),
            pitch: self
                .pitch
                .update(value.pitch, at, min_cutoff_hz, beta, derivative_cutoff_hz),
            roll: self
                .roll
                .update(value.roll, at, min_cutoff_hz, beta, derivative_cutoff_hz),
        }
    }
}

#[derive(Default)]
struct RectFilter {
    x: OneEuro,
    y: OneEuro,
    width: OneEuro,
    height: OneEuro,
}

impl RectFilter {
    fn update(
        &mut self,
        value: Rect,
        at: Instant,
        min_cutoff_hz: f32,
        beta: f32,
        derivative_cutoff_hz: f32,
    ) -> Rect {
        Rect {
            x: self
                .x
                .update(value.x, at, min_cutoff_hz, beta, derivative_cutoff_hz),
            y: self
                .y
                .update(value.y, at, min_cutoff_hz, beta, derivative_cutoff_hz),
            width: self
                .width
                .update(value.width, at, min_cutoff_hz, beta, derivative_cutoff_hz),
            height: self
                .height
                .update(value.height, at, min_cutoff_hz, beta, derivative_cutoff_hz),
        }
    }
}

#[derive(Default)]
struct BoolDebounce {
    value: Option<bool>,
    disagreement_count: u8,
}

impl BoolDebounce {
    fn update(&mut self, raw: bool) -> bool {
        let Some(current) = self.value else {
            self.value = Some(raw);
            return raw;
        };
        if raw == current {
            self.disagreement_count = 0;
            return current;
        }
        self.disagreement_count = self.disagreement_count.saturating_add(1);
        if self.disagreement_count >= 2 {
            self.value = Some(raw);
            self.disagreement_count = 0;
            raw
        } else {
            current
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;

    fn observation(gaze_x: f32, yaw: f32) -> GazeObservation {
        GazeObservation {
            frame_width: 640,
            frame_height: 360,
            face: Rect {
                x: 210.0,
                y: 70.0,
                width: 220.0,
                height: 250.0,
            },
            face_confidence: 0.92,
            landmarks: vec![Point { x: 320.0, y: 180.0 }],
            left_eye: Rect::default(),
            right_eye: Rect::default(),
            left_eye_open: true,
            right_eye_open: true,
            head_pose: HeadPose {
                yaw,
                pitch: 0.0,
                roll: 0.0,
            },
            gaze: Vec3 {
                x: gaze_x,
                y: 0.0,
                z: 1.0,
            }
            .normalized(),
            inference_ms: 8.0,
        }
    }

    #[test]
    fn suppresses_stationary_model_jitter() {
        let started = Instant::now();
        let mut filter = ObservationFilter::new(ObservationFilterConfig::default());
        let mut raw = Vec::new();
        let mut filtered = Vec::new();
        for index in 0..24 {
            let gaze = if index % 2 == 0 { -0.07 } else { 0.07 };
            raw.push(gaze);
            filtered.push(
                filter
                    .update(
                        observation(gaze, gaze * 65.0),
                        started + Duration::from_millis(index * 66),
                    )
                    .gaze
                    .x,
            );
        }
        let raw_range = range(&raw[12..]);
        let filtered_range = range(&filtered[12..]);
        assert!(filtered_range < raw_range * 0.45);
    }

    #[test]
    fn follows_a_deliberate_head_turn_without_long_fixed_lag() {
        let started = Instant::now();
        let mut filter = ObservationFilter::new(ObservationFilterConfig::default());
        for index in 0..8 {
            filter.update(
                observation(0.0, 0.0),
                started + Duration::from_millis(index * 66),
            );
        }
        let mut output = 0.0;
        for index in 8..12 {
            output = filter
                .update(
                    observation(0.45, 32.0),
                    started + Duration::from_millis(index * 66),
                )
                .head_pose
                .yaw;
        }
        assert!(output > 26.0, "filtered yaw was {output}");
    }

    #[test]
    fn ignores_one_frame_eye_state_flicker() {
        let started = Instant::now();
        let mut filter = ObservationFilter::new(ObservationFilterConfig::default());
        filter.update(observation(0.0, 0.0), started);
        let mut closed = observation(0.0, 0.0);
        closed.left_eye_open = false;
        let first = filter.update(closed.clone(), started + Duration::from_millis(66));
        let second = filter.update(closed, started + Duration::from_millis(132));
        assert!(first.left_eye_open);
        assert!(!second.left_eye_open);
    }

    #[test]
    fn rejects_a_single_frame_head_pose_outlier() {
        let started = Instant::now();
        let mut filter = ObservationFilter::new(ObservationFilterConfig::default());
        for index in 0..8 {
            filter.update(
                observation(0.0, 2.0),
                started + Duration::from_millis(index * 66),
            );
        }
        let outlier = filter.update(
            observation(0.0, 70.0),
            started + Duration::from_millis(8 * 66),
        );
        assert!(outlier.head_pose.yaw < 8.0);
    }

    fn range(values: &[f32]) -> f32 {
        let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        maximum - minimum
    }
}
