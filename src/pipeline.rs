use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use image::RgbImage;
use mnn_runtime::{Runtime, RuntimeConfig};
use parking_lot::Mutex;

use crate::image_ops::{align_face_for_embedding, crop, eye_box, rotate_rgb, to_nchw_bgr};
use crate::model::{find_output, find_output_len, MnnModel};
use crate::presence::FaceEmbedding;
use crate::{Error, GazeObservation, HeadPose, Point, Rect, Result, Vec3};

const FACE_SIZE: u32 = 300;
const ATTRIBUTE_SIZE: u32 = 60;
const EYE_STATE_SIZE: u32 = 32;
const FACE_EMBEDDING_SIZE: u32 = 128;
const FACE_CONFIDENCE: f32 = 0.60;
const NMS_THRESHOLD: f32 = 0.45;
const EYE_BOX_SCALE: f32 = 1.8;

/// Bytes for the six networks used by gaze and local presence recognition.
#[derive(Clone, Copy)]
pub struct ModelBundle<'a> {
    pub face: &'a [u8],
    pub landmarks: &'a [u8],
    pub head_pose: &'a [u8],
    pub eye_state: &'a [u8],
    pub gaze: &'a [u8],
    pub face_embedding: &'a [u8],
}

/// Synchronous gaze inference engine. MNN sessions themselves are worker-confined.
pub struct GazeEngine {
    face: MnnModel,
    landmarks: MnnModel,
    head_pose: MnnModel,
    eye_state: MnnModel,
    gaze: MnnModel,
    face_embedding: MnnModel,
    roll_align: bool,
    primary_track: Mutex<PrimaryTrackState>,
}

#[derive(Clone, Copy, Debug)]
struct PrimaryTrack {
    id: u64,
    face: Rect,
    confidence: f32,
    face_count: usize,
}

#[derive(Default)]
struct PrimaryTrackState {
    current: Option<PrimaryTrack>,
    next_id: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct InferencePlan {
    pub(crate) detect_faces: bool,
    pub(crate) estimate_gaze: bool,
    pub(crate) identity_tracking: bool,
    pub(crate) extract_identity: bool,
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
            face_embedding: MnnModel::load(
                runtime,
                "face reidentification",
                bundle.face_embedding,
            )?,
            roll_align: true,
            primary_track: Mutex::new(PrimaryTrackState::default()),
        };
        engine.validate_shapes()?;
        Ok(engine)
    }

    /// Load a verified, architecture-independent model bundle from disk.
    pub fn from_model_directory(directory: &Path, threads: usize) -> Result<Self> {
        let models = crate::OwnedModelBundle::load_from_directory(directory)?;
        let runtime = Runtime::new(RuntimeConfig::new().with_threads(threads))
            .map_err(|error| Error::Model(format!("create MNN runtime: {error}")))?;
        Self::new(&runtime, models.as_borrowed())
    }

    /// Run zero-filled inputs through all six models and report graph outputs.
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
        let face_embedding = self.face_embedding.run(vec![
            0.0;
            3 * FACE_EMBEDDING_SIZE as usize
                * FACE_EMBEDDING_SIZE as usize
        ])?;
        if find_output_len(&face_embedding, 256).is_none() {
            return Err(Error::Model(format!(
                "face reidentification returned unexpected outputs: {face_embedding:?}"
            )));
        }
        Ok([
            ("face", face),
            ("landmarks", landmarks),
            ("head_pose", pose),
            ("eye_state", eye_state),
            ("gaze", gaze),
            ("face_embedding", face_embedding),
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
            (
                "face reidentification",
                self.face_embedding.input_shape(),
                &[1, 3, 128, 128][..],
            ),
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

    /// Estimate gaze from one packed RGB frame without exporting biometric data.
    pub fn infer(&self, frame: &RgbImage) -> Result<Option<GazeObservation>> {
        self.infer_frame(
            frame,
            InferencePlan {
                detect_faces: true,
                estimate_gaze: true,
                identity_tracking: false,
                extract_identity: false,
            },
        )
        .map(|result| result.observation)
    }

    pub(crate) fn infer_planned(
        &self,
        frame: &RgbImage,
        plan: InferencePlan,
    ) -> Result<FrameInference> {
        self.infer_frame(frame, plan)
    }

    fn infer_frame(&self, frame: &RgbImage, plan: InferencePlan) -> Result<FrameInference> {
        let started = Instant::now();
        let (track, track_continuous) = if plan.detect_faces {
            let mut faces = self.detect_faces(frame)?;
            self.update_primary_track(&mut faces)
        } else {
            (self.primary_track.lock().current, true)
        };
        let Some(track) = track else {
            return Ok(FrameInference {
                observation: None,
                face_count: 0,
                embeddings: Vec::new(),
                detection_ran: plan.detect_faces,
                identity_ran: false,
                track_id: None,
                track_continuous,
            });
        };
        let face = track.face;
        let confidence = track.confidence;
        let face_count = track.face_count;
        let face_image = crop(frame, face)
            .ok_or_else(|| Error::Model("detected face crop is invalid".into()))?;
        let landmarks = self.face_landmarks(&face_image, face)?;

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

        // Eye-state and gaze previously rotated both crops independently twice.
        // Keep one aligned copy per eye and share it between the two stages.
        let aligned_left = rotate_rgb(&left_image, head_pose.roll);
        let aligned_right = rotate_rgb(&right_image, head_pose.roll);
        let left_eye_open = self.eye_is_open(&aligned_left)?;
        let right_eye_open = self.eye_is_open(&aligned_right)?;
        let mut gaze = Vec3::default();
        if plan.estimate_gaze && left_eye_open && right_eye_open {
            let mut pose_for_gaze = head_pose;
            if self.roll_align {
                left_image = aligned_left;
                right_image = aligned_right;
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

        let observation = GazeObservation {
            frame_width: frame.width(),
            frame_height: frame.height(),
            face,
            face_confidence: confidence,
            landmarks: landmarks.clone(),
            left_eye,
            right_eye,
            left_eye_open,
            right_eye_open,
            head_pose,
            gaze,
            inference_ms: started.elapsed().as_secs_f32() * 1_000.0,
        };
        let mut embeddings = Vec::new();
        let mut identity_ran = false;
        // Multiple people is already a fail-closed presence verdict. Computing
        // landmarks and embeddings for every secondary face cannot change that
        // verdict, so identity work is restricted to a single continuous track.
        if plan.identity_tracking && (plan.extract_identity || !track_continuous) && face_count == 1
        {
            identity_ran = true;
            if let Some(embedding) = self.face_embedding(frame, &landmarks)? {
                embeddings.push(embedding);
            }
        }
        Ok(FrameInference {
            observation: Some(observation),
            face_count,
            embeddings,
            detection_ran: plan.detect_faces,
            identity_ran,
            track_id: Some(track.id),
            track_continuous,
        })
    }

    fn face_landmarks(&self, face_image: &RgbImage, face: Rect) -> Result<Vec<Point>> {
        let landmarks_raw = self.landmarks.run_one(to_nchw_bgr(
            face_image,
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
        Ok(landmarks_raw
            .chunks_exact(2)
            .map(|xy| Point {
                x: face.x + xy[0] * face.width,
                y: face.y + xy[1] * face.height,
            })
            .collect())
    }

    fn face_embedding(
        &self,
        frame: &RgbImage,
        landmarks: &[Point],
    ) -> Result<Option<FaceEmbedding>> {
        let Some(aligned) = align_face_for_embedding(frame, landmarks) else {
            return Ok(None);
        };
        self.face_embedding
            .run_one(to_nchw_bgr(
                &aligned,
                FACE_EMBEDDING_SIZE,
                FACE_EMBEDDING_SIZE,
                false,
            ))
            .and_then(FaceEmbedding::new)
            .map(Some)
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

    fn detect_faces(&self, frame: &RgbImage) -> Result<Vec<(Rect, f32)>> {
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
            .filter_map(|detection| {
                Rect {
                    x: detection.x_min * frame.width() as f32,
                    y: detection.y_min * frame.height() as f32,
                    width: (detection.x_max - detection.x_min) * frame.width() as f32,
                    height: (detection.y_max - detection.y_min) * frame.height() as f32,
                }
                .clamp(frame.width(), frame.height())
                .map(|rect| (rect, detection.confidence))
            })
            .collect())
    }

    fn update_primary_track(&self, faces: &mut [(Rect, f32)]) -> (Option<PrimaryTrack>, bool) {
        let mut state = self.primary_track.lock();
        if faces.is_empty() {
            let continuous = state.current.is_none();
            state.current = None;
            return (None, continuous);
        }
        let previous = state.current;
        let selected = previous
            .and_then(|previous| {
                faces
                    .iter()
                    .enumerate()
                    .map(|(index, (face, _))| (index, rect_iou(previous.face, *face)))
                    .max_by(|left, right| left.1.total_cmp(&right.1))
                    .filter(|(_, overlap)| *overlap >= 0.15)
                    .map(|(index, _)| index)
            })
            .unwrap_or(0);
        faces.swap(0, selected);
        let (raw, confidence) = faces[0];
        let track_continuous =
            previous.is_some_and(|previous| rect_iou(previous.face, raw) >= 0.15);
        let stable = previous
            .filter(|previous| track_continuous && rect_iou(previous.face, raw) >= 0.15)
            .map(|previous| blend_rect(previous.face, raw, 0.38))
            .unwrap_or(raw);
        faces[0].0 = stable;
        let id = if track_continuous {
            previous.expect("continuous track has previous state").id
        } else {
            state.next_id = state.next_id.saturating_add(1).max(1);
            state.next_id
        };
        let track = PrimaryTrack {
            id,
            face: stable,
            confidence,
            face_count: faces.len(),
        };
        state.current = Some(track);
        (Some(track), track_continuous)
    }
}

fn blend_rect(previous: Rect, current: Rect, alpha: f32) -> Rect {
    let blend = |a: f32, b: f32| a + (b - a) * alpha;
    Rect {
        x: blend(previous.x, current.x),
        y: blend(previous.y, current.y),
        width: blend(previous.width, current.width),
        height: blend(previous.height, current.height),
    }
}

fn rect_iou(a: Rect, b: Rect) -> f32 {
    let left = a.x.max(b.x);
    let top = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let union = a.width * a.height + b.width * b.height - intersection;
    intersection / union.max(f32::EPSILON)
}

pub(crate) struct FrameInference {
    pub(crate) observation: Option<GazeObservation>,
    pub(crate) face_count: usize,
    pub(crate) embeddings: Vec<FaceEmbedding>,
    pub(crate) detection_ran: bool,
    pub(crate) identity_ran: bool,
    pub(crate) track_id: Option<u64>,
    pub(crate) track_continuous: bool,
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
