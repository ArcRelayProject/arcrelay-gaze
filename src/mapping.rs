use arcrelay_input::{DeskPointUm, DisplaySurface, WorkspaceLayout};
use sha2::{Digest, Sha256};

use crate::calibration::ProjectionSource;
use crate::{CalibrationProfile, Error, GazeObservation, GazeTarget, Result, TargetingSource};

/// Stable digest of the camera-relevant physical display arrangement.
#[must_use]
pub fn layout_signature(layout: &WorkspaceLayout) -> String {
    let mut hash = Sha256::new();
    hash.update(layout.workspace_id.as_str().as_bytes());
    for (id, display) in &layout.displays {
        hash.update(id.as_str().as_bytes());
        hash.update(display.device_id.as_str().as_bytes());
        hash.update(display.fingerprint.as_str().as_bytes());
        for value in [
            display.desk_rect_um.x,
            display.desk_rect_um.y,
            display.desk_rect_um.width,
            display.desk_rect_um.height,
        ] {
            hash.update(value.to_le_bytes());
        }
        hash.update(format!("{:?}", display.rotation));
    }
    format!("{:x}", hash.finalize())
}

/// Maps calibrated observations to the existing Arc Input workspace topology.
pub struct WorkspaceMapper {
    layout: WorkspaceLayout,
    profile: CalibrationProfile,
}

impl WorkspaceMapper {
    pub fn new(layout: WorkspaceLayout, profile: CalibrationProfile) -> Result<Self> {
        layout
            .validate()
            .map_err(|error| Error::Mapping(error.to_string()))?;
        let actual = layout_signature(&layout);
        if profile.layout_signature != actual {
            return Err(Error::Mapping(
                "calibration does not match the current display arrangement".into(),
            ));
        }
        Ok(Self { layout, profile })
    }

    #[must_use]
    pub fn layout(&self) -> &WorkspaceLayout {
        &self.layout
    }

    #[must_use]
    pub fn profile(&self) -> &CalibrationProfile {
        &self.profile
    }

    pub fn map(&self, observation: &GazeObservation) -> Result<Option<GazeTarget>> {
        if !self.profile.head_regions.is_empty() {
            return self.map_head_region(observation);
        }
        if !observation.usable_for_targeting() && !observation.usable_for_head_targeting() {
            return Ok(None);
        }
        let (point, source) = self.profile.project_with_source(observation)?;
        let Some(display) = self
            .layout
            .displays
            .values()
            .find(|display| contains(display, point))
            .or_else(|| {
                (source == ProjectionSource::HeadFallback)
                    .then(|| {
                        self.layout.displays.values().min_by_key(|display| {
                            let center = display_center(display);
                            let dx = i128::from(center.x) - i128::from(point.x);
                            let dy = i128::from(center.y) - i128::from(point.y);
                            dx * dx + dy * dy
                        })
                    })
                    .flatten()
            })
        else {
            return Ok(None);
        };
        let point = if source == ProjectionSource::HeadFallback {
            display_center(display)
        } else {
            point
        };
        let logical = display.logical_point_from_desk(point);
        let error_um = match source {
            ProjectionSource::Eye => self.profile.rms_error_um,
            ProjectionSource::HeadFallback => self.profile.head_rms_error_um.unwrap_or(150_000.0),
        };
        let error_scale = (1.0 / (1.0 + error_um / 50_000.0)) as f32;
        Ok(Some(GazeTarget {
            device_id: display.device_id.to_string(),
            display_id: display.display_id.to_string(),
            desk_x_um: point.x,
            desk_y_um: point.y,
            logical_x: logical.x,
            logical_y: logical.y,
            confidence: (observation.face_confidence * error_scale).clamp(
                0.0,
                if source == ProjectionSource::HeadFallback {
                    0.65
                } else {
                    1.0
                },
            ),
            source: match source {
                ProjectionSource::Eye => TargetingSource::Eye,
                ProjectionSource::HeadFallback => TargetingSource::HeadFallback,
            },
        }))
    }

    fn map_head_region(&self, observation: &GazeObservation) -> Result<Option<GazeTarget>> {
        let Some((region, region_confidence)) = self.profile.classify_head_region(observation)
        else {
            return Ok(None);
        };
        if region_confidence < 0.35 {
            return Ok(None);
        }
        let display = self
            .layout
            .displays
            .values()
            .find(|display| display.display_id.as_str() == region.display_id)
            .ok_or_else(|| {
                Error::Mapping(format!(
                    "calibrated display {} is missing from the workspace",
                    region.display_id
                ))
            })?;
        let eye_point = (self.profile.eye_sample_count.unwrap_or(0) >= 9
            && observation.usable_for_targeting())
        .then(|| self.profile.project_with_source(observation))
        .transpose()?
        .and_then(|(point, source)| (source == ProjectionSource::Eye).then_some(point));
        let (point, source) = eye_point
            .map(|point| (clamp_to_display(display, point), TargetingSource::Eye))
            .unwrap_or_else(|| (display_center(display), TargetingSource::HeadFallback));
        let logical = display.logical_point_from_desk(point);
        let calibration_quality = if source == TargetingSource::Eye {
            (1.0 / (1.0 + self.profile.rms_error_um / 80_000.0)) as f32
        } else {
            0.7
        };
        // Region classification already rejects ambiguous and out-of-distribution
        // observations. Blend the remaining evidence instead of multiplying it,
        // which would make normal calibration error suppress every valid target.
        let confidence = 0.62 * region_confidence
            + 0.23 * observation.face_confidence
            + 0.15 * calibration_quality;
        Ok(Some(GazeTarget {
            device_id: display.device_id.to_string(),
            display_id: display.display_id.to_string(),
            desk_x_um: point.x,
            desk_y_um: point.y,
            logical_x: logical.x,
            logical_y: logical.y,
            confidence: confidence.clamp(0.0, 0.92),
            source,
        }))
    }
}

fn contains(display: &DisplaySurface, point: DeskPointUm) -> bool {
    let rect = display.desk_rect_um;
    point.x >= rect.x
        && point.x < rect.x.saturating_add(rect.width)
        && point.y >= rect.y
        && point.y < rect.y.saturating_add(rect.height)
}

fn display_center(display: &DisplaySurface) -> DeskPointUm {
    DeskPointUm {
        x: display
            .desk_rect_um
            .x
            .saturating_add(display.desk_rect_um.width / 2),
        y: display
            .desk_rect_um
            .y
            .saturating_add(display.desk_rect_um.height / 2),
    }
}

fn clamp_to_display(display: &DisplaySurface, point: DeskPointUm) -> DeskPointUm {
    let rect = display.desk_rect_um;
    let margin_x = (rect.width / 50).max(1);
    let margin_y = (rect.height / 50).max(1);
    DeskPointUm {
        x: point.x.clamp(
            rect.x.saturating_add(margin_x),
            rect.x.saturating_add(rect.width).saturating_sub(margin_x),
        ),
        y: point.y.clamp(
            rect.y.saturating_add(margin_y),
            rect.y.saturating_add(rect.height).saturating_sub(margin_y),
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use arcrelay_input::{
        DeskRectUm, DisplayFingerprint, DisplayId, DisplayRotation, GeometryConfidence,
        InventoryRevision, LogicalRect, ScaleFactor, ServiceInstanceId, SizeI64, SizeU32,
        TopologyRevision, WorkspaceId,
    };

    use super::*;
    use crate::{HeadPose, HeadRegionProfile, Point, Rect, Vec3};

    fn layout() -> WorkspaceLayout {
        let display_id = DisplayId::parse("local-display").unwrap();
        let display = DisplaySurface {
            display_id: display_id.clone(),
            device_id: ServiceInstanceId::parse("local-device").unwrap(),
            fingerprint: DisplayFingerprint::parse("panel-a").unwrap(),
            name: "Main".into(),
            pixel_size: SizeU32 {
                width: 1920,
                height: 1080,
            },
            logical_bounds: LogicalRect {
                x: 0.0,
                y: 0.0,
                width: 1920.0,
                height: 1080.0,
            },
            scale_factor: ScaleFactor(1.0),
            physical_size_um: SizeI64 {
                width: 600_000,
                height: 340_000,
            },
            rotation: DisplayRotation::Degrees0,
            desk_rect_um: DeskRectUm {
                x: 0,
                y: 0,
                width: 600_000,
                height: 340_000,
            },
            geometry_confidence: GeometryConfidence::HardwareReported,
            inventory_revision: InventoryRevision(1),
        };
        WorkspaceLayout {
            workspace_id: WorkspaceId::parse("desk").unwrap(),
            revision: TopologyRevision(1),
            displays: BTreeMap::from([(display_id, display)]),
            portals: Vec::new(),
        }
    }

    fn observation() -> GazeObservation {
        GazeObservation {
            frame_width: 640,
            frame_height: 360,
            face: Rect {
                x: 200.0,
                y: 80.0,
                width: 200.0,
                height: 220.0,
            },
            face_confidence: 0.9,
            landmarks: vec![Point { x: 250.0, y: 150.0 }],
            left_eye: Rect {
                x: 240.0,
                y: 140.0,
                width: 30.0,
                height: 20.0,
            },
            right_eye: Rect {
                x: 330.0,
                y: 140.0,
                width: 30.0,
                height: 20.0,
            },
            left_eye_open: true,
            right_eye_open: true,
            head_pose: HeadPose::default(),
            gaze: Vec3 {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            },
            inference_ms: 10.0,
        }
    }

    #[test]
    fn maps_physical_point_to_logical_display_coordinates() {
        let layout = layout();
        let profile = CalibrationProfile {
            version: 1,
            camera_id: "camera".into(),
            layout_signature: layout_signature(&layout),
            coefficients_x: [300_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            coefficients_y: [170_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            rms_error_um: 1_000.0,
            sample_count: 9,
            eye_sample_count: None,
            head_coefficients_x: None,
            head_coefficients_y: None,
            head_rms_error_um: None,
            head_regions: Vec::new(),
        };
        let target = WorkspaceMapper::new(layout, profile)
            .unwrap()
            .map(&observation())
            .unwrap()
            .unwrap();
        assert_eq!(target.device_id, "local-device");
        assert_eq!(target.display_id, "local-display");
        assert!((target.logical_x - 960.0).abs() < 0.1);
        assert!((target.logical_y - 540.0).abs() < 0.1);
    }

    #[test]
    fn falls_back_to_head_direction_when_eye_gaze_is_unavailable() {
        let layout = layout();
        let profile = CalibrationProfile {
            version: 2,
            camera_id: "camera".into(),
            layout_signature: layout_signature(&layout),
            coefficients_x: [0.0; 8],
            coefficients_y: [0.0; 8],
            rms_error_um: f64::INFINITY,
            sample_count: 9,
            eye_sample_count: Some(0),
            head_coefficients_x: Some([300_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            head_coefficients_y: Some([170_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            head_rms_error_um: Some(20_000.0),
            head_regions: Vec::new(),
        };
        let mut observation = observation();
        observation.left_eye_open = false;
        observation.right_eye_open = false;
        observation.head_pose.yaw = 38.0;
        let target = WorkspaceMapper::new(layout, profile)
            .unwrap()
            .map(&observation)
            .unwrap()
            .unwrap();
        assert_eq!(target.source, TargetingSource::HeadFallback);
        assert_eq!(target.display_id, "local-display");
        assert!((target.logical_x - 960.0).abs() < 0.1);
        assert!(target.confidence <= 0.65);
    }

    #[test]
    fn head_only_region_profile_selects_screen_center() {
        let mut layout = layout();
        let mut right = layout.displays.values().next().unwrap().clone();
        right.display_id = DisplayId::parse("right-display").unwrap();
        right.device_id = ServiceInstanceId::parse("right-device").unwrap();
        right.fingerprint = DisplayFingerprint::parse("panel-b").unwrap();
        right.name = "Right".into();
        right.logical_bounds.x = 1920.0;
        right.desk_rect_um.x = 600_000;
        layout.displays.insert(right.display_id.clone(), right);
        let profile = CalibrationProfile {
            version: 4,
            camera_id: "camera".into(),
            layout_signature: layout_signature(&layout),
            coefficients_x: [0.0; 8],
            coefficients_y: [0.0; 8],
            rms_error_um: 0.0,
            sample_count: 18,
            eye_sample_count: Some(0),
            head_coefficients_x: None,
            head_coefficients_y: None,
            head_rms_error_um: None,
            head_regions: vec![
                HeadRegionProfile {
                    display_id: "local-display".into(),
                    centroid: [-0.48, 0.0, 0.48, 0.5, 0.19],
                    scale: [0.07, 0.07, 0.03, 0.03, 0.02],
                    gaze_centroid: None,
                    gaze_scale: None,
                    acceptance_radius: 3.0,
                    sample_count: 9,
                },
                HeadRegionProfile {
                    display_id: "right-display".into(),
                    centroid: [0.52, 0.0, 0.48, 0.5, 0.19],
                    scale: [0.07, 0.07, 0.03, 0.03, 0.02],
                    gaze_centroid: None,
                    gaze_scale: None,
                    acceptance_radius: 3.0,
                    sample_count: 9,
                },
            ],
        };
        let mut observation = observation();
        observation.gaze.x = -0.8;
        observation.head_pose.yaw = 42.0;
        let target = WorkspaceMapper::new(layout, profile)
            .unwrap()
            .map(&observation)
            .unwrap()
            .unwrap();
        assert_eq!(target.display_id, "right-display");
        assert_eq!(target.device_id, "right-device");
        assert!((target.logical_x - 2_880.0).abs() < 0.1);
        assert_eq!(target.source, TargetingSource::HeadFallback);
    }

    #[test]
    fn fused_region_profile_uses_eye_regression_within_selected_screen() {
        let mut layout = layout();
        let mut right = layout.displays.values().next().unwrap().clone();
        right.display_id = DisplayId::parse("right-display").unwrap();
        right.device_id = ServiceInstanceId::parse("right-device").unwrap();
        right.fingerprint = DisplayFingerprint::parse("panel-b").unwrap();
        right.name = "Right".into();
        right.logical_bounds.x = 1920.0;
        right.desk_rect_um.x = 600_000;
        layout.displays.insert(right.display_id.clone(), right);
        let profile = CalibrationProfile {
            version: 4,
            camera_id: "camera".into(),
            layout_signature: layout_signature(&layout),
            coefficients_x: [900_000.0, 200_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            coefficients_y: [170_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            rms_error_um: 20_000.0,
            sample_count: 90,
            eye_sample_count: Some(90),
            head_coefficients_x: None,
            head_coefficients_y: None,
            head_rms_error_um: None,
            head_regions: vec![
                HeadRegionProfile {
                    display_id: "local-display".into(),
                    centroid: [-0.48, 0.0, 0.48, 0.5, 0.19],
                    scale: [0.07, 0.07, 0.03, 0.03, 0.02],
                    gaze_centroid: Some([0.3, 0.0]),
                    gaze_scale: Some([0.1, 0.1]),
                    acceptance_radius: 3.0,
                    sample_count: 45,
                },
                HeadRegionProfile {
                    display_id: "right-display".into(),
                    centroid: [0.56, 0.0, 0.48, 0.5, 0.19],
                    scale: [0.07, 0.07, 0.03, 0.03, 0.02],
                    gaze_centroid: Some([-0.3, 0.0]),
                    gaze_scale: Some([0.1, 0.1]),
                    acceptance_radius: 3.0,
                    sample_count: 45,
                },
            ],
        };
        let mut observation = observation();
        observation.head_pose.yaw = 42.0;
        observation.gaze.x = -0.3;
        let target = WorkspaceMapper::new(layout, profile)
            .unwrap()
            .map(&observation)
            .unwrap()
            .unwrap();
        assert_eq!(target.display_id, "right-display");
        assert_eq!(target.source, TargetingSource::Eye);
        assert_eq!(target.desk_x_um, 840_000);
        assert!((target.logical_x - 2_688.0).abs() < 0.1);
        assert!(target.confidence >= 0.55);
    }

    #[test]
    fn invalidates_profile_when_layout_changes() {
        let layout = layout();
        let profile = CalibrationProfile {
            version: 1,
            camera_id: "camera".into(),
            layout_signature: "stale".into(),
            coefficients_x: [0.0; 8],
            coefficients_y: [0.0; 8],
            rms_error_um: 0.0,
            sample_count: 9,
            eye_sample_count: None,
            head_coefficients_x: None,
            head_coefficients_y: None,
            head_rms_error_um: None,
            head_regions: Vec::new(),
        };
        assert!(matches!(
            WorkspaceMapper::new(layout, profile),
            Err(Error::Mapping(message)) if message.contains("does not match")
        ));
    }
}
