use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

pub const FACE_EMBEDDING_DIMENSIONS: usize = 256;
pub const PRESENCE_PROFILE_VERSION: u32 = 1;
const DEFAULT_OWNER_SIMILARITY: f32 = 0.65;
const UNKNOWN_MARGIN: f32 = 0.05;
const ENROLLMENT_SAMPLE_SIMILARITY: f32 = 0.55;
const REQUIRED_ENROLLMENT_SAMPLES: usize = 12;

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
    pub template: Vec<f32>,
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
            template,
            sample_count,
        })
    }

    pub fn validate(&mut self) -> Result<()> {
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
        if self.display_name.is_empty() || self.sample_count == 0 {
            return Err(Error::InvalidConfig(
                "presence profile name or samples are invalid".into(),
            ));
        }
        self.template = normalize_embedding(std::mem::take(&mut self.template))?;
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
                .filter_map(|embedding| embedding.similarity(&profile.template))
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
    let owner_similarity = embeddings
        .first()
        .and_then(|embedding| embedding.similarity(&profile.template));
    let state = match owner_similarity {
        Some(similarity) if similarity >= DEFAULT_OWNER_SIMILARITY => PresenceState::OwnerPresent,
        Some(similarity) if similarity <= DEFAULT_OWNER_SIMILARITY - UNKNOWN_MARGIN => {
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
    samples: Vec<Vec<f32>>,
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
            samples: Vec::with_capacity(REQUIRED_ENROLLMENT_SAMPLES),
            rejected_frames: 0,
        })
    }

    pub(crate) fn push(
        &mut self,
        face_count: usize,
        embeddings: &[FaceEmbedding],
    ) -> Result<Option<PresenceProfile>> {
        if face_count != 1 || embeddings.len() != 1 {
            self.rejected_frames = self.rejected_frames.saturating_add(1);
            return Ok(None);
        }
        let embedding = &embeddings[0].values;
        if let Some(reference) = self.samples.first() {
            if cosine_similarity(reference, embedding)
                .is_none_or(|similarity| similarity < ENROLLMENT_SAMPLE_SIMILARITY)
            {
                self.rejected_frames = self.rejected_frames.saturating_add(1);
                return Ok(None);
            }
        }
        self.samples.push(embedding.clone());
        if self.samples.len() < REQUIRED_ENROLLMENT_SAMPLES {
            return Ok(None);
        }
        let mut average = vec![0.0; FACE_EMBEDDING_DIMENSIONS];
        for sample in &self.samples {
            for (value, sample_value) in average.iter_mut().zip(sample) {
                *value += sample_value;
            }
        }
        PresenceProfile::new(self.display_name.clone(), average, self.samples.len()).map(Some)
    }

    pub(crate) fn status(&self) -> PresenceEnrollmentStatus {
        PresenceEnrollmentStatus {
            active: true,
            collected_samples: self.samples.len(),
            required_samples: REQUIRED_ENROLLMENT_SAMPLES,
            rejected_frames: self.rejected_frames,
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
            classify_presence(Some(&profile), 1, &[embedding(0)]).state,
            PresenceState::OwnerPresent
        );
        assert_eq!(
            classify_presence(Some(&profile), 1, &[embedding(1)]).state,
            PresenceState::UnknownPresent
        );
        assert_eq!(
            classify_presence(Some(&profile), 2, &[embedding(0), embedding(1)]).state,
            PresenceState::MultiplePeople
        );
        assert_eq!(
            classify_presence(Some(&profile), 0, &[]).state,
            PresenceState::Absent
        );
        assert_eq!(
            classify_presence(None, 1, &[embedding(0)]).state,
            PresenceState::Uncertain
        );
    }

    #[test]
    fn protective_state_transitions_faster_than_owner_or_absence() {
        let started = Instant::now();
        let mut stabilizer = PresenceStabilizer::new(started);
        let unknown = classify_presence(Some(&profile()), 1, &[embedding(1)]);
        stabilizer.update(unknown.clone(), started);
        assert_eq!(
            stabilizer
                .update(unknown, started + Duration::from_millis(251))
                .state,
            PresenceState::UnknownPresent
        );
        let owner = classify_presence(Some(&profile()), 1, &[embedding(0)]);
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
        assert!(enrollment.push(1, &[embedding(1)]).unwrap().is_none());
        assert!(enrollment.push(1, &[embedding(2)]).unwrap().is_none());
        assert_eq!(enrollment.status().rejected_frames, 1);
        let mut complete = None;
        for _ in 1..REQUIRED_ENROLLMENT_SAMPLES {
            complete = enrollment.push(1, &[embedding(1)]).unwrap();
        }
        let complete = complete.expect("profile should complete");
        assert_eq!(complete.sample_count, REQUIRED_ENROLLMENT_SAMPLES);
        assert!((complete.template[1] - 1.0).abs() < 1e-6);
    }
}
