use std::cmp::Ordering;
use std::time::Instant;

use image::RgbImage;
use mnn_runtime::{Runtime, RuntimeConfig};

use crate::image_ops::{crop, eye_box, rotate_rgb, to_nchw_bgr};
use crate::model::{find_output, find_output_len, MnnModel};
use crate::{Error, GazeObservation, HeadPose, Point, Rect, Result, Vec3};

const FACE_SIZE: u32 = 300;
const ATTRIBUTE_SIZE: u32 = 60;
const EYE_STATE_SIZE: u32 = 32;
const FACE_CONFIDENCE: f32 = 0.60;
const NMS_THRESHOLD: f32 = 0.45;
const EYE_BOX_SCALE: f32 = 1.8;

/// Bytes for the five networks used by the Intel-style gaze pipeline.
#[derive(Clone, Copy)]
pub struct ModelBundle<'a> {
    pub face: &'a [u8],
    pub landmarks: &'a [u8],
    pub head_pose: &'a [u8],
    pub eye_state: &'a [u8],
    pub gaze: &'a [u8],
}

impl ModelBundle<'static> {
    /// Models bundled with the crate, including their third-party notices.
    #[must_use]
    pub fn bundled() -> Self {
        Self {
            face: include_bytes!("../models/face-detection-retail-0004.mnn"),
            landmarks: include_bytes!("../models/facial-landmarks-35-adas-0002.mnn"),
            head_pose: include_bytes!("../models/head-pose-estimation-adas-0001.mnn"),
            eye_state: include_bytes!("../models/open-closed-eye-0001.mnn"),
            gaze: include_bytes!("../models/gaze-estimation-adas-0002-packed.mnn"),
        }
    }
}

/// Synchronous gaze inference engine. MNN sessions themselves are worker-confined.
pub struct GazeEngine {
    face: MnnModel,
    landmarks: MnnModel,
    head_pose: MnnModel,
    eye_state: MnnModel,
    gaze: MnnModel,
    roll_align: bool,
}

impl GazeEngine {
    /// Load all models into a caller-supplied MNN runtime.
    pub fn new(runtime: &Runtime, bundle: ModelBundle<'_>) -> Result<Self> {
        let engine = Self {
            face: MnnModel::load(runtime, "face detector", bundle.face)?,
            landmarks: MnnModel::load(runtime, "facial landmarks", bundle.landmarks)?,
            head_pose: MnnModel::load(runtime, "head pose", bundle.head_pose)?,
            eye_state: MnnModel::load(runtime, "eye state", bundle.eye_state)?,
            gaze: MnnModel::load(runtime, "gaze estimation", bundle.gaze)?,
            roll_align: true,
        };
        engine.validate_shapes()?;
        Ok(engine)
    }

    /// Load bundled models with a serialized CPU runtime.
    pub fn bundled(threads: usize) -> Result<Self> {
        let runtime = Runtime::new(RuntimeConfig::new().with_threads(threads))
            .map_err(|error| Error::Model(format!("create MNN runtime: {error}")))?;
        Self::new(&runtime, ModelBundle::bundled())
    }

    /// Run zero-filled inputs through all five models and report graph outputs.
    pub fn self_test(&self) -> Result<Vec<String>> {
        let face = self.face.run(vec![0.0; 3 * 300 * 300])?;
        if find_output_len(&face, 12_996).is_none() || find_output_len(&face, 6_498).is_none() {
            return Err(Error::Model(format!(
                "face detector returned unexpected outputs: {face:?}"
            )));
        }
        let landmarks = self.landmarks.run(vec![0.0; 3 * 60 * 60])?;
        if find_output_len(&landmarks, 70).is_none() {
            return Err(Error::Model(format!(
                "landmarks returned unexpected outputs: {landmarks:?}"
            )));
        }
        let pose = self.head_pose.run(vec![0.0; 3 * 60 * 60])?;
        for name in ["fc_y", "fc_p", "fc_r"] {
            if find_output(&pose, name).is_none() {
                return Err(Error::Model(format!("head pose is missing {name}")));
            }
        }
        let eye_state = self.eye_state.run(vec![0.0; 3 * 32 * 32])?;
        if find_output_len(&eye_state, 2).is_none() {
            return Err(Error::Model(format!(
                "eye state returned unexpected outputs: {eye_state:?}"
            )));
        }
        let gaze = self.gaze.run(vec![0.0; 21_603])?;
        if find_output_len(&gaze, 3).is_none() {
            return Err(Error::Model(format!(
                "gaze returned unexpected outputs: {gaze:?}"
            )));
        }
        Ok([
            ("face", face),
            ("landmarks", landmarks),
            ("head_pose", pose),
            ("eye_state", eye_state),
            ("gaze", gaze),
        ]
        .into_iter()
        .map(|(model, outputs)| {
            let tensors = outputs
                .iter()
                .map(|output| format!("{} {:?}", output.name, output.shape))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{model}: {tensors}")
        })
        .collect())
    }

    fn validate_shapes(&self) -> Result<()> {
        let expected = [
            (
                "face detector",
                self.face.input_shape(),
                &[1, 3, 300, 300][..],
            ),
            (
                "landmarks",
                self.landmarks.input_shape(),
                &[1, 3, 60, 60][..],
            ),
            (
                "head pose",
                self.head_pose.input_shape(),
                &[1, 3, 60, 60][..],
            ),
            (
                "eye state",
                self.eye_state.input_shape(),
                &[1, 3, 32, 32][..],
            ),
            ("gaze", self.gaze.input_shape(), &[1, 21_603][..]),
        ];
        for (name, actual, wanted) in expected {
            if actual != wanted {
                return Err(Error::Model(format!(
                    "{name} input is {actual:?}, expected {wanted:?}"
                )));
            }
        }
        Ok(())
    }

    /// Estimate gaze from one packed RGB frame.
    pub fn infer(&self, frame: &RgbImage) -> Result<Option<GazeObservation>> {
        let started = Instant::now();
        let Some((face, confidence)) = self.detect_primary_face(frame)? else {
            return Ok(None);
        };
        let face_image = crop(frame, face)
            .ok_or_else(|| Error::Model("detected face crop is invalid".into()))?;

        let landmarks_raw = self.landmarks.run_one(to_nchw_bgr(
            &face_image,
            ATTRIBUTE_SIZE,
            ATTRIBUTE_SIZE,
            false,
        ))?;
        if landmarks_raw.len() != 70 {
            return Err(Error::Model(format!(
                "landmark output has {}, expected 70",
                landmarks_raw.len()
            )));
        }
        let landmarks = landmarks_raw
            .chunks_exact(2)
            .map(|xy| Point {
                x: face.x + xy[0] * face.width,
                y: face.y + xy[1] * face.height,
            })
            .collect::<Vec<_>>();

        let pose_outputs = self.head_pose.run(to_nchw_bgr(
            &face_image,
            ATTRIBUTE_SIZE,
            ATTRIBUTE_SIZE,
            false,
        ))?;
        let angle = |needle: &str| -> Result<f32> {
            find_output(&pose_outputs, needle)
                .and_then(|output| output.values.first().copied())
                .ok_or_else(|| Error::Model(format!("missing head-pose output {needle}")))
        };
        let head_pose = HeadPose {
            yaw: angle("fc_y")?,
            pitch: angle("fc_p")?,
            roll: angle("fc_r")?,
        };

        let left_eye = eye_box(landmarks[0], landmarks[1], EYE_BOX_SCALE, frame)
            .ok_or_else(|| Error::Model("left eye crop is outside the frame".into()))?;
        let right_eye = eye_box(landmarks[2], landmarks[3], EYE_BOX_SCALE, frame)
            .ok_or_else(|| Error::Model("right eye crop is outside the frame".into()))?;
        let mut left_image =
            crop(frame, left_eye).ok_or_else(|| Error::Model("crop left eye".into()))?;
        let mut right_image =
            crop(frame, right_eye).ok_or_else(|| Error::Model("crop right eye".into()))?;

        let left_eye_open = self.eye_is_open(&rotate_rgb(&left_image, head_pose.roll))?;
        let right_eye_open = self.eye_is_open(&rotate_rgb(&right_image, head_pose.roll))?;
        let mut gaze = Vec3::default();
        if left_eye_open && right_eye_open {
            let mut pose_for_gaze = head_pose;
            if self.roll_align {
                left_image = rotate_rgb(&left_image, head_pose.roll);
                right_image = rotate_rgb(&right_image, head_pose.roll);
                pose_for_gaze.roll = 0.0;
            }
            let mut packed = to_nchw_bgr(&left_image, ATTRIBUTE_SIZE, ATTRIBUTE_SIZE, false);
            packed.extend(to_nchw_bgr(
                &right_image,
                ATTRIBUTE_SIZE,
                ATTRIBUTE_SIZE,
                false,
            ));
            packed.extend([pose_for_gaze.yaw, pose_for_gaze.pitch, pose_for_gaze.roll]);
            let gaze_raw = self.gaze.run_one(packed)?;
            if gaze_raw.len() != 3 {
                return Err(Error::Model(format!(
                    "gaze output has {}, expected 3",
                    gaze_raw.len()
                )));
            }
            gaze = Vec3 {
                x: gaze_raw[0],
                y: gaze_raw[1],
                z: gaze_raw[2],
            }
            .normalized();
            if self.roll_align {
                let roll = head_pose.roll.to_radians();
                let (sin, cos) = roll.sin_cos();
                gaze = Vec3 {
                    x: gaze.x * cos + gaze.y * sin,
                    y: -gaze.x * sin + gaze.y * cos,
                    z: gaze.z,
                };
            }
        }

        Ok(Some(GazeObservation {
            frame_width: frame.width(),
            frame_height: frame.height(),
            face,
            face_confidence: confidence,
            landmarks,
            left_eye,
            right_eye,
            left_eye_open,
            right_eye_open,
            head_pose,
            gaze,
            inference_ms: started.elapsed().as_secs_f32() * 1_000.0,
        }))
    }

    fn eye_is_open(&self, image: &RgbImage) -> Result<bool> {
        let output =
            self.eye_state
                .run_one(to_nchw_bgr(image, EYE_STATE_SIZE, EYE_STATE_SIZE, true))?;
        if output.len() != 2 {
            return Err(Error::Model(format!(
                "eye-state output has {}, expected 2",
                output.len()
            )));
        }
        Ok(output[1] > output[0])
    }

    fn detect_primary_face(&self, frame: &RgbImage) -> Result<Option<(Rect, f32)>> {
        let outputs = self
            .face
            .run(to_nchw_bgr(frame, FACE_SIZE, FACE_SIZE, false))?;
        let locations = find_output_len(&outputs, 12_996)
            .ok_or_else(|| Error::Model("face detector locations output is missing".into()))?;
        let confidences = find_output_len(&outputs, 6_498)
            .ok_or_else(|| Error::Model("face detector confidence output is missing".into()))?;
        let mut detections = decode_ssd(&locations.values, &confidences.values, FACE_CONFIDENCE);
        detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        Ok(nms(detections, NMS_THRESHOLD)
            .into_iter()
            .next()
            .and_then(|detection| {
                Rect {
                    x: detection.x_min * frame.width() as f32,
                    y: detection.y_min * frame.height() as f32,
                    width: (detection.x_max - detection.x_min) * frame.width() as f32,
                    height: (detection.y_max - detection.y_min) * frame.height() as f32,
                }
                .clamp(frame.width(), frame.height())
                .map(|rect| (rect, detection.confidence))
            }))
    }
}

#[derive(Clone, Copy, Debug)]
struct Detection {
    confidence: f32,
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
}

fn decode_ssd(locations: &[f32], confidences: &[f32], threshold: f32) -> Vec<Detection> {
    const WIDTHS: [f32; 9] = [9.4, 25.1, 14.7, 34.7, 143.0, 77.4, 128.8, 51.1, 75.6];
    const HEIGHTS: [f32; 9] = [15.0, 39.6, 25.5, 63.2, 227.5, 162.9, 124.5, 105.1, 72.6];
    let mut result = Vec::new();
    for y in 0..19 {
        for x in 0..19 {
            for prior in 0..9 {
                let index = (y * 19 + x) * 9 + prior;
                let confidence = confidences[index * 2 + 1];
                if confidence < threshold {
                    continue;
                }
                let location = &locations[index * 4..index * 4 + 4];
                let prior_width = WIDTHS[prior] / FACE_SIZE as f32;
                let prior_height = HEIGHTS[prior] / FACE_SIZE as f32;
                let prior_x = (x as f32 + 0.5) * 16.0 / FACE_SIZE as f32;
                let prior_y = (y as f32 + 0.5) * 16.0 / FACE_SIZE as f32;
                let center_x = prior_x + location[0] * 0.1 * prior_width;
                let center_y = prior_y + location[1] * 0.1 * prior_height;
                let width = (location[2] * 0.2).exp() * prior_width;
                let height = (location[3] * 0.2).exp() * prior_height;
                result.push(Detection {
                    confidence,
                    x_min: (center_x - width * 0.5).clamp(0.0, 1.0),
                    y_min: (center_y - height * 0.5).clamp(0.0, 1.0),
                    x_max: (center_x + width * 0.5).clamp(0.0, 1.0),
                    y_max: (center_y + height * 0.5).clamp(0.0, 1.0),
                });
            }
        }
    }
    result
}

fn nms(mut detections: Vec<Detection>, threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
    });
    detections.truncate(400);
    let mut kept: Vec<Detection> = Vec::new();
    'candidate: for detection in detections {
        for other in &kept {
            if intersection_over_union(detection, *other) > threshold {
                continue 'candidate;
            }
        }
        kept.push(detection);
        if kept.len() == 200 {
            break;
        }
    }
    kept
}

fn intersection_over_union(a: Detection, b: Detection) -> f32 {
    let width = (a.x_max.min(b.x_max) - a.x_min.max(b.x_min)).max(0.0);
    let height = (a.y_max.min(b.y_max) - a.y_min.max(b.y_min)).max(0.0);
    let intersection = width * height;
    let area_a = (a.x_max - a.x_min).max(0.0) * (a.y_max - a.y_min).max(0.0);
    let area_b = (b.x_max - b.x_min).max(0.0) * (b.y_max - b.y_min).max(0.0);
    intersection / (area_a + area_b - intersection).max(f32::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_suppresses_overlapping_boxes() {
        let boxes = vec![
            Detection {
                confidence: 0.9,
                x_min: 0.0,
                y_min: 0.0,
                x_max: 0.5,
                y_max: 0.5,
            },
            Detection {
                confidence: 0.8,
                x_min: 0.05,
                y_min: 0.05,
                x_max: 0.5,
                y_max: 0.5,
            },
            Detection {
                confidence: 0.7,
                x_min: 0.6,
                y_min: 0.6,
                x_max: 0.9,
                y_max: 0.9,
            },
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].confidence, 0.9);
    }
}
