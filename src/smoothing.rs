use std::time::{Duration, Instant};

use crate::{GazeTarget, StabilizedTarget};

/// Dwell, loss timeout, and exponential smoothing policy.
#[derive(Clone, Copy, Debug)]
pub struct StabilizerConfig {
    pub dwell: Duration,
    pub loss_timeout: Duration,
    pub switch_cooldown: Duration,
    pub minimum_confidence: f32,
    pub smoothing_alpha: f64,
}

impl Default for StabilizerConfig {
    fn default() -> Self {
        Self {
            dwell: Duration::from_millis(550),
            loss_timeout: Duration::from_millis(900),
            switch_cooldown: Duration::from_millis(1_200),
            minimum_confidence: 0.50,
            smoothing_alpha: 0.28,
        }
    }
}

/// Turns noisy per-frame display hits into an intentional, stable target.
pub struct TargetStabilizer {
    config: StabilizerConfig,
    candidate: Option<(GazeTarget, Instant)>,
    active: Option<(GazeTarget, Instant)>,
    last_seen: Option<Instant>,
    last_switch: Option<Instant>,
}

impl TargetStabilizer {
    #[must_use]
    pub fn new(config: StabilizerConfig) -> Self {
        Self {
            config,
            candidate: None,
            active: None,
            last_seen: None,
            last_switch: None,
        }
    }

    pub fn update(&mut self, target: Option<GazeTarget>, now: Instant) -> Option<StabilizedTarget> {
        let Some(target) =
            target.filter(|target| target.confidence >= self.config.minimum_confidence)
        else {
            if self
                .last_seen
                .is_some_and(|seen| now.saturating_duration_since(seen) >= self.config.loss_timeout)
            {
                self.candidate = None;
                self.active = None;
            }
            return self.current(now, false);
        };
        self.last_seen = Some(now);
        match &mut self.candidate {
            Some((candidate, _)) if same_surface(candidate, &target) => {
                smooth(candidate, &target, self.config.smoothing_alpha);
            }
            _ => self.candidate = Some((target, now)),
        }
        let (candidate, since) = self.candidate.as_ref().expect("candidate was set");
        if now.saturating_duration_since(*since) < self.config.dwell {
            return self.current(now, false);
        }
        let changed = self
            .active
            .as_ref()
            .is_none_or(|(active, _)| !same_surface(active, candidate));
        if changed
            && self.last_switch.is_some_and(|last_switch| {
                now.saturating_duration_since(last_switch) < self.config.switch_cooldown
            })
        {
            return self.current(now, false);
        }
        let activated_at = if changed {
            now
        } else {
            self.active.as_ref().expect("active target").1
        };
        self.active = Some((candidate.clone(), activated_at));
        if changed {
            self.last_switch = Some(now);
        }
        self.current(now, changed)
    }

    pub fn reset(&mut self) {
        self.candidate = None;
        self.active = None;
        self.last_seen = None;
        self.last_switch = None;
    }

    fn current(&self, now: Instant, changed: bool) -> Option<StabilizedTarget> {
        self.active
            .as_ref()
            .map(|(target, since)| StabilizedTarget {
                target: target.clone(),
                stable_for_ms: now
                    .saturating_duration_since(*since)
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX),
                changed,
            })
    }
}

fn same_surface(left: &GazeTarget, right: &GazeTarget) -> bool {
    left.device_id == right.device_id && left.display_id == right.display_id
}

fn smooth(current: &mut GazeTarget, next: &GazeTarget, alpha: f64) {
    let alpha = alpha.clamp(0.0, 1.0);
    let blend = |left: f64, right: f64| left + (right - left) * alpha;
    current.desk_x_um = blend(current.desk_x_um as f64, next.desk_x_um as f64).round() as i64;
    current.desk_y_um = blend(current.desk_y_um as f64, next.desk_y_um as f64).round() as i64;
    current.logical_x = blend(current.logical_x, next.logical_x);
    current.logical_y = blend(current.logical_y, next.logical_y);
    current.confidence = blend(f64::from(current.confidence), f64::from(next.confidence)) as f32;
    current.source = next.source;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(display: &str) -> GazeTarget {
        GazeTarget {
            device_id: "device".into(),
            display_id: display.into(),
            desk_x_um: 10,
            desk_y_um: 20,
            logical_x: 1.0,
            logical_y: 2.0,
            confidence: 0.9,
            source: crate::TargetingSource::Eye,
        }
    }

    #[test]
    fn requires_dwell_before_switching_displays() {
        let started = Instant::now();
        let mut filter = TargetStabilizer::new(StabilizerConfig::default());
        assert!(filter.update(Some(target("a")), started).is_none());
        let active = filter
            .update(Some(target("a")), started + Duration::from_millis(600))
            .unwrap();
        assert!(active.changed);
        assert_eq!(active.target.display_id, "a");
        let held = filter
            .update(Some(target("b")), started + Duration::from_millis(610))
            .unwrap();
        assert!(!held.changed);
        assert_eq!(held.target.display_id, "a");
    }

    #[test]
    fn rejects_low_confidence_targets_without_dropping_the_active_surface() {
        let started = Instant::now();
        let mut filter = TargetStabilizer::new(StabilizerConfig::default());
        filter.update(Some(target("a")), started);
        let active = filter
            .update(Some(target("a")), started + Duration::from_millis(600))
            .unwrap();
        assert_eq!(active.target.display_id, "a");
        let mut uncertain = target("b");
        uncertain.confidence = 0.2;
        let held = filter
            .update(Some(uncertain), started + Duration::from_millis(700))
            .unwrap();
        assert_eq!(held.target.display_id, "a");
    }

    #[test]
    fn switches_after_both_dwell_and_cooldown_have_elapsed() {
        let started = Instant::now();
        let mut filter = TargetStabilizer::new(StabilizerConfig::default());
        filter.update(Some(target("a")), started);
        filter.update(Some(target("a")), started + Duration::from_millis(600));
        filter.update(Some(target("b")), started + Duration::from_millis(700));
        let held = filter
            .update(Some(target("b")), started + Duration::from_millis(1_300))
            .unwrap();
        assert_eq!(held.target.display_id, "a");
        let switched = filter
            .update(Some(target("b")), started + Duration::from_millis(1_801))
            .unwrap();
        assert!(switched.changed);
        assert_eq!(switched.target.display_id, "b");
    }
}
