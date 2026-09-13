use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{Error, HeadPose, Result};

pub const FACE_EMBEDDING_DIMENSIONS: usize = 256;
pub const PRESENCE_PROFILE_VERSION: u32 = 2;
const DEFAULT_OWNER_SIMILARITY: f32 = 0.65;
const UNKNOWN_MARGIN: f32 = 0.05;
const ENROLLMENT_SAMPLE_SIMILARITY: f32 = 0.60;
const ENROLLMENT_CROSS_POSE_SIMILARITY: f32 = 0.52;
const SAMPLES_PER_POSE: usize = 4;
const ENROLLMENT_POSES: [PresencePose; 5] = [
    PresencePose::Frontal,
    PresencePose::Left,
    PresencePose::Right,
    PresencePose::Up,
    PresencePose::Down,
];
pub const PRESENCE_ENROLLMENT_REQUIRED_SAMPLES: usize = SAMPLES_PER_POSE * ENROLLMENT_POSES.len();

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PresencePose {
    #[default]
    Frontal,
    Left,
    Right,
    Up,
    Down,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PresencePoseTemplate {
    pub pose: PresencePose,
    pub template: Vec<f32>,
    pub sample_count: usize,
    pub owner_threshold: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PresenceState {
    Absent,
    OwnerPresent,
    UnknownPresent,
    MultiplePeople,
    #[default]
    Uncertain,
}

impl PresenceState {
    #[must_use]
    pub fn is_private(self) -> bool {
        matches!(
            self,
            Self::UnknownPresent | Self::MultiplePeople | Self::Uncertain
        )
    }

    #[must_use]
    pub fn event_kind(self) -> &'static str {
        match self {
            Self::Absent => "presence.absent",
            Self::OwnerPresent => "presence.ownerPresent",
            Self::UnknownPresent => "presence.unknownPresent",
            Self::MultiplePeople => "presence.multiplePeople",
            Self::Uncertain => "presence.uncertain",
        }
    }
}

/// Local biometric template. Desktop persistence must keep this file private and
/// must never transmit it through ArcRelay's protocol layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PresenceProfile {
    pub version: u32,
    pub display_name: String,
    /// Legacy v1 centroid. It is migrated to a frontal pose template on load.
    #[serde(default)]
    pub template: Vec<f32>,
    #[serde(default)]
    pub templates: Vec<PresencePoseTemplate>,
    pub sample_count: usize,
}

impl PresenceProfile {
    pub fn new(
        display_name: impl Into<String>,
        template: Vec<f32>,
        sample_count: usize,
    ) -> Result<Self> {
        let display_name = display_name
            .into()
            .trim()
            .chars()
            .take(80)
            .collect::<String>();
        if display_name.is_empty() {
            return Err(Error::InvalidConfig(
                "presence profile name is empty".into(),
            ));
        }
        if sample_count == 0 {
            return Err(Error::InvalidConfig(
                "presence profile contains no enrollment samples".into(),
            ));
        }
        let template = normalize_embedding(template)?;
        Ok(Self {
            version: PRESENCE_PROFILE_VERSION,
            display_name,
            template: Vec::new(),
            templates: vec![PresencePoseTemplate {
                pose: PresencePose::Frontal,
                template,
                sample_count,
                owner_threshold: DEFAULT_OWNER_SIMILARITY,
            }],
            sample_count,
        })
    }

    fn from_pose_templates(
        display_name: impl Into<String>,
        templates: Vec<PresencePoseTemplate>,
    ) -> Result<Self> {
        let sample_count = templates.iter().map(|template| template.sample_count).sum();
        let mut profile = Self {
            version: PRESENCE_PROFILE_VERSION,
            display_name: display_name.into(),
            template: Vec::new(),
            templates,
            sample_count,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&mut self) -> Result<()> {
        if self.version == 1 && self.templates.is_empty() && !self.template.is_empty() {
            let template = normalize_embedding(std::mem::take(&mut self.template))?;
            self.templates.push(PresencePoseTemplate {
                pose: PresencePose::Frontal,
                template,
                sample_count: self.sample_count,
                owner_threshold: DEFAULT_OWNER_SIMILARITY,
            });
            self.version = PRESENCE_PROFILE_VERSION;
        }
        if self.version != PRESENCE_PROFILE_VERSION {
            return Err(Error::InvalidConfig(format!(
                "unsupported presence profile version {}",
                self.version
            )));
        }
        self.display_name = self
            .display_name
            .trim()
            .chars()
            .take(80)
            .collect::<String>();
        if self.display_name.is_empty() || self.templates.is_empty() {
            return Err(Error::InvalidConfig(
                "presence profile name or samples are invalid".into(),
            ));
        }
        self.template.clear();
        for template in &mut self.templates {
            if template.sample_count == 0 {
                return Err(Error::InvalidConfig(
                    "presence pose template contains no samples".into(),
                ));
            }
            template.template = normalize_embedding(std::mem::take(&mut template.template))?;
            template.owner_threshold = template
                .owner_threshold
                .clamp(DEFAULT_OWNER_SIMILARITY, 0.85);
        }
        self.sample_count = self
            .templates
            .iter()
            .map(|template| template.sample_count)
            .sum();
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FaceEmbedding {
    pub values: Vec<f32>,
}

impl FaceEmbedding {
    pub(crate) fn new(values: Vec<f32>) -> Result<Self> {
        Ok(Self {
            values: normalize_embedding(values)?,
        })
    }

    pub(crate) fn similarity(&self, template: &[f32]) -> Option<f32> {
        cosine_similarity(&self.values, template)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PresenceObservation {
    pub state: PresenceState,
    pub face_count: usize,
    pub owner_similarity: Option<f32>,
    pub stable_for_ms: u64,
    pub profile_enrolled: bool,
}

pub(crate) fn classify_presence(
    profile: Option<&PresenceProfile>,
    face_count: usize,
    embeddings: &[FaceEmbedding],
    head_pose: Option<HeadPose>,
) -> PresenceObservation {
    if face_count == 0 {
        return PresenceObservation {
            state: PresenceState::Absent,
            face_count,
            profile_enrolled: profile.is_some(),
            ..PresenceObservation::default()
        };
    }
    if face_count > 1 {
        let owner_similarity = profile.and_then(|profile| {
            embeddings
                .iter()
                .filter_map(|embedding| best_owner_match(profile, embedding, None).map(|m| m.0))
                .max_by(f32::total_cmp)
        });
        return PresenceObservation {
            state: PresenceState::MultiplePeople,
            face_count,
            owner_similarity,
            profile_enrolled: profile.is_some(),
            ..PresenceObservation::default()
        };
    }
    let Some(profile) = profile else {
        return PresenceObservation {
            state: PresenceState::Uncertain,
            face_count,
            profile_enrolled: false,
            ..PresenceObservation::default()
        };
    };
    let owner_match = embeddings
        .first()
        .and_then(|embedding| best_owner_match(profile, embedding, head_pose.map(pose_for_head)));
    let owner_similarity = owner_match.map(|matched| matched.0);
    let state = match owner_match {
        Some((similarity, threshold)) if similarity >= threshold => PresenceState::OwnerPresent,
        Some((similarity, threshold)) if similarity <= threshold - UNKNOWN_MARGIN => {
            PresenceState::UnknownPresent
        }
        _ => PresenceState::Uncertain,
    };
    PresenceObservation {
        state,
        face_count,
        owner_similarity,
        profile_enrolled: true,
        stable_for_ms: 0,
    }
}

fn best_owner_match(
    profile: &PresenceProfile,
    embedding: &FaceEmbedding,
    pose: Option<PresencePose>,
) -> Option<(f32, f32)> {
    let compatible = |template: &&PresencePoseTemplate| {
        pose.is_none_or(|pose| template.pose == pose || template.pose == PresencePose::Frontal)
    };
    let candidates = profile
        .templates
        .iter()
        .filter(compatible)
        .collect::<Vec<_>>();
    let candidates = if candidates.is_empty() {
        profile.templates.iter().collect::<Vec<_>>()
    } else {
        candidates
    };
    candidates
        .into_iter()
        .filter_map(|template| {
            embedding
                .similarity(&template.template)
                .map(|similarity| (similarity, template.owner_threshold))
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

fn pose_for_head(head_pose: HeadPose) -> PresencePose {
    let horizontal = head_pose.yaw.abs() / 16.0;
    let vertical = head_pose.pitch.abs() / 13.0;
    if horizontal < 0.65 && vertical < 0.65 {
        PresencePose::Frontal
    } else if horizontal >= vertical {
        if head_pose.yaw < 0.0 {
            PresencePose::Left
        } else {
            PresencePose::Right
        }
    } else if head_pose.pitch < 0.0 {
        PresencePose::Up
    } else {
        PresencePose::Down
    }
}

pub(crate) struct PresenceStabilizer {
    stable: PresenceObservation,
    stable_since: Instant,
    candidate: Option<(PresenceObservation, Instant)>,
}

impl PresenceStabilizer {
    pub(crate) fn new(now: Instant) -> Self {
        Self {
            stable: PresenceObservation::default(),
            stable_since: now,
            candidate: None,
        }
    }

    pub(crate) fn update(
        &mut self,
        mut observation: PresenceObservation,
        now: Instant,
    ) -> PresenceObservation {
        if observation.state == self.stable.state {
            self.stable.face_count = observation.face_count;
            self.stable.owner_similarity = observation.owner_similarity;
            self.stable.profile_enrolled = observation.profile_enrolled;
            self.candidate = None;
        } else if self
            .candidate
            .as_ref()
            .is_some_and(|(candidate, _)| candidate.state == observation.state)
        {
            let (_, since) = self.candidate.as_ref().expect("candidate checked above");
            if now.saturating_duration_since(*since) >= transition_delay(observation.state) {
                observation.stable_for_ms = 0;
                self.stable = observation;
                self.stable_since = now;
                self.candidate = None;
            } else if let Some((candidate, _)) = self.candidate.as_mut() {
                candidate.face_count = observation.face_count;
                candidate.owner_similarity = observation.owner_similarity;
                candidate.profile_enrolled = observation.profile_enrolled;
            }
        } else {
            self.candidate = Some((observation, now));
        }
        self.stable.stable_for_ms = now
            .saturating_duration_since(self.stable_since)
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        self.stable.clone()
    }
}

fn transition_delay(state: PresenceState) -> Duration {
    match state {
        PresenceState::UnknownPresent | PresenceState::MultiplePeople => Duration::from_millis(250),
        PresenceState::Uncertain => Duration::from_millis(400),
        PresenceState::OwnerPresent => Duration::from_millis(600),
        PresenceState::Absent => Duration::from_millis(1_200),
    }
}

pub(crate) struct PresenceEnrollment {
    display_name: String,
    pose_index: usize,
    samples: Vec<(PresencePose, Vec<Vec<f32>>)>,
    rejected_frames: usize,
}

impl PresenceEnrollment {
    pub(crate) fn new(display_name: impl Into<String>) -> Result<Self> {
        let display_name = display_name
            .into()
            .trim()
            .chars()
            .take(80)
            .collect::<String>();
        if display_name.is_empty() {
            return Err(Error::InvalidConfig(
                "presence profile name is empty".into(),
            ));
        }
        Ok(Self {
            display_name,
            pose_index: 0,
            samples: ENROLLMENT_POSES
                .into_iter()
                .map(|pose| (pose, Vec::with_capacity(SAMPLES_PER_POSE)))
                .collect(),
            rejected_frames: 0,
        })
    }

    pub(crate) fn push(
        &mut self,
        face_count: usize,
        embeddings: &[FaceEmbedding],
        head_pose: HeadPose,
    ) -> Result<Option<PresenceProfile>> {
        if face_count != 1 || embeddings.len() != 1 {
            self.rejected_frames = self.rejected_frames.saturating_add(1);
            return Ok(None);
        }
        let expected_pose = ENROLLMENT_POSES[self.pose_index];
        if !pose_matches(expected_pose, head_pose) {
            return Ok(None);
        }
        let embedding = &embeddings[0].values;
        let same_pose_reference = self.samples[self.pose_index].1.first();
        let identity_reference = same_pose_reference
            .or_else(|| self.samples.iter().find_map(|(_, samples)| samples.first()));
        if let Some(reference) = identity_reference {
            let minimum_similarity = if same_pose_reference.is_some() {
                ENROLLMENT_SAMPLE_SIMILARITY
            } else {
                // A genuine embedding can move noticeably between frontal and
                // profile views. Keep a lower cross-pose guard, then apply the
                // stricter consistency check to the remaining frames in that pose.
                ENROLLMENT_CROSS_POSE_SIMILARITY
            };
            if cosine_similarity(reference, embedding)
                .is_none_or(|similarity| similarity < minimum_similarity)
            {
                self.rejected_frames = self.rejected_frames.saturating_add(1);
                return Ok(None);
            }
        }
        self.samples[self.pose_index].1.push(embedding.clone());
        if self.samples[self.pose_index].1.len() < SAMPLES_PER_POSE {
            return Ok(None);
        }
        self.pose_index += 1;
        if self.pose_index < ENROLLMENT_POSES.len() {
            return Ok(None);
        }
        let templates = self
            .samples
            .iter()
            .map(|(pose, samples)| build_pose_template(*pose, samples))
            .collect::<Result<Vec<_>>>()?;
        PresenceProfile::from_pose_templates(self.display_name.clone(), templates).map(Some)
    }

    pub(crate) fn status(&self) -> PresenceEnrollmentStatus {
        PresenceEnrollmentStatus {
            active: true,
            collected_samples: self.samples.iter().map(|(_, samples)| samples.len()).sum(),
            required_samples: PRESENCE_ENROLLMENT_REQUIRED_SAMPLES,
            rejected_frames: self.rejected_frames,
            required_pose: ENROLLMENT_POSES.get(self.pose_index).copied(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PresenceEnrollmentStatus {
    pub active: bool,
    pub collected_samples: usize,
    pub required_samples: usize,
    pub rejected_frames: usize,
    pub required_pose: Option<PresencePose>,
}

fn pose_matches(expected: PresencePose, head_pose: HeadPose) -> bool {
    match expected {
        PresencePose::Frontal => head_pose.yaw.abs() <= 8.0 && head_pose.pitch.abs() <= 8.0,
        PresencePose::Left => {
            (-32.0..=-13.0).contains(&head_pose.yaw) && head_pose.pitch.abs() <= 16.0
        }
        PresencePose::Right => {
            (13.0..=32.0).contains(&head_pose.yaw) && head_pose.pitch.abs() <= 16.0
        }
        PresencePose::Up => {
            (-25.0..=-10.0).contains(&head_pose.pitch) && head_pose.yaw.abs() <= 18.0
        }
        PresencePose::Down => {
            (10.0..=25.0).contains(&head_pose.pitch) && head_pose.yaw.abs() <= 18.0
        }
    }
}

fn build_pose_template(pose: PresencePose, samples: &[Vec<f32>]) -> Result<PresencePoseTemplate> {
    let mut average = vec![0.0; FACE_EMBEDDING_DIMENSIONS];
    for sample in samples {
        for (value, sample_value) in average.iter_mut().zip(sample) {
            *value += sample_value;
        }
    }
    let template = normalize_embedding(average)?;
    let minimum_genuine_similarity = samples
        .iter()
        .filter_map(|sample| cosine_similarity(sample, &template))
        .min_by(f32::total_cmp)
        .unwrap_or(DEFAULT_OWNER_SIMILARITY);
    Ok(PresencePoseTemplate {
        pose,
        template,
        sample_count: samples.len(),
        owner_threshold: (minimum_genuine_similarity - 0.08).clamp(DEFAULT_OWNER_SIMILARITY, 0.80),
    })
}

fn normalize_embedding(mut values: Vec<f32>) -> Result<Vec<f32>> {
    if values.len() != FACE_EMBEDDING_DIMENSIONS || values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Model(format!(
            "face embedding has {} values; expected {FACE_EMBEDDING_DIMENSIONS} finite values",
            values.len()
        )));
    }
    let length = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !length.is_finite() || length <= f32::EPSILON {
        return Err(Error::Model("face embedding has zero magnitude".into()));
    }
    for value in &mut values {
        *value /= length;
    }
    Ok(values)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    (left.len() == FACE_EMBEDDING_DIMENSIONS && right.len() == FACE_EMBEDDING_DIMENSIONS)
        .then(|| left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>())
        .filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(index: usize) -> FaceEmbedding {
        let mut values = vec![0.0; FACE_EMBEDDING_DIMENSIONS];
        values[index] = 1.0;
        FaceEmbedding::new(values).unwrap()
    }

    fn profile() -> PresenceProfile {
        PresenceProfile::new("Owner", embedding(0).values, 12).unwrap()
    }

    #[test]
    fn classifies_owner_unknown_multiple_and_absent() {
        let profile = profile();
        assert_eq!(
            classify_presence(Some(&profile), 1, &[embedding(0)], None).state,
            PresenceState::OwnerPresent
        );
        assert_eq!(
            classify_presence(Some(&profile), 1, &[embedding(1)], None).state,
            PresenceState::UnknownPresent
        );
        assert_eq!(
            classify_presence(Some(&profile), 2, &[embedding(0), embedding(1)], None).state,
            PresenceState::MultiplePeople
        );
        assert_eq!(
            classify_presence(Some(&profile), 0, &[], None).state,
            PresenceState::Absent
        );
        assert_eq!(
            classify_presence(None, 1, &[embedding(0)], None).state,
            PresenceState::Uncertain
        );
    }

    #[test]
    fn protective_state_transitions_faster_than_owner_or_absence() {
        let started = Instant::now();
        let mut stabilizer = PresenceStabilizer::new(started);
        let unknown = classify_presence(Some(&profile()), 1, &[embedding(1)], None);
        stabilizer.update(unknown.clone(), started);
        assert_eq!(
            stabilizer
                .update(unknown, started + Duration::from_millis(251))
                .state,
            PresenceState::UnknownPresent
        );
        let owner = classify_presence(Some(&profile()), 1, &[embedding(0)], None);
        stabilizer.update(owner.clone(), started + Duration::from_millis(300));
        assert_eq!(
            stabilizer
                .update(owner.clone(), started + Duration::from_millis(800))
                .state,
            PresenceState::UnknownPresent
        );
        assert_eq!(
            stabilizer
                .update(owner, started + Duration::from_millis(901))
                .state,
            PresenceState::OwnerPresent
        );
    }

    #[test]
    fn enrollment_rejects_different_people_and_builds_normalized_template() {
        let mut enrollment = PresenceEnrollment::new("Owner").unwrap();
        assert!(enrollment
            .push(1, &[embedding(1)], head_for_pose(PresencePose::Frontal))
            .unwrap()
            .is_none());
        assert!(enrollment
            .push(1, &[embedding(2)], head_for_pose(PresencePose::Frontal))
            .unwrap()
            .is_none());
        assert_eq!(enrollment.status().rejected_frames, 1);
        let mut complete = None;
        for pose in ENROLLMENT_POSES {
            let already_collected = usize::from(pose == PresencePose::Frontal);
            for _ in already_collected..SAMPLES_PER_POSE {
                complete = enrollment
                    .push(1, &[embedding(1)], head_for_pose(pose))
                    .unwrap();
            }
        }
        let complete = complete.expect("profile should complete");
        assert_eq!(complete.sample_count, PRESENCE_ENROLLMENT_REQUIRED_SAMPLES);
        assert_eq!(complete.templates.len(), ENROLLMENT_POSES.len());
        assert!(complete
            .templates
            .iter()
            .all(|template| (template.template[1] - 1.0).abs() < 1e-6));
    }

    #[test]
    fn migrates_a_legacy_centroid_to_a_frontal_pose_template() {
        let mut legacy = PresenceProfile {
            version: 1,
            display_name: "Owner".into(),
            template: embedding(3).values,
            templates: Vec::new(),
            sample_count: 12,
        };
        legacy.validate().unwrap();
        assert_eq!(legacy.version, PRESENCE_PROFILE_VERSION);
        assert!(legacy.template.is_empty());
        assert_eq!(legacy.templates.len(), 1);
        assert_eq!(legacy.templates[0].pose, PresencePose::Frontal);
    }

    fn head_for_pose(pose: PresencePose) -> HeadPose {
        match pose {
            PresencePose::Frontal => HeadPose::default(),
            PresencePose::Left => HeadPose {
                yaw: -20.0,
                ..HeadPose::default()
            },
            PresencePose::Right => HeadPose {
                yaw: 20.0,
                ..HeadPose::default()
            },
            PresencePose::Up => HeadPose {
                pitch: -15.0,
                ..HeadPose::default()
            },
            PresencePose::Down => HeadPose {
                pitch: 15.0,
                ..HeadPose::default()
            },
        }
    }
}
