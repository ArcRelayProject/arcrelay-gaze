use arcrelay_input::{DeskPointUm, DisplaySurface, WorkspaceLayout};
use sha2::{Digest, Sha256};

use crate::{CalibrationProfile, Error, GazeObservation, GazeTarget, Result};

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
        if !observation.usable_for_targeting() {
            return Ok(None);
        }
        let point = self.profile.project(observation)?;
        let Some(display) = self
            .layout
            .displays
            .values()
            .find(|display| contains(display, point))
        else {
            return Ok(None);
        };
        let logical = display.logical_point_from_desk(point);
        let error_scale = (1.0 / (1.0 + self.profile.rms_error_um / 50_000.0)) as f32;
        Ok(Some(GazeTarget {
            device_id: display.device_id.to_string(),
            display_id: display.display_id.to_string(),
            desk_x_um: point.x,
            desk_y_um: point.y,
            logical_x: logical.x,
            logical_y: logical.y,
            confidence: (observation.face_confidence * error_scale).clamp(0.0, 1.0),
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use arcrelay_input::{
        DeskRectUm, DisplayFingerprint, DisplayId, DisplayRotation, GeometryConfidence,
        InventoryRevision, LogicalRect, ScaleFactor, ServiceInstanceId, SizeI64, SizeU32,
        TopologyRevision, WorkspaceId,
    };

    use super::*;
    use crate::{HeadPose, Point, Rect, Vec3};

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
        };
        assert!(matches!(
            WorkspaceMapper::new(layout, profile),
            Err(Error::Mapping(message)) if message.contains("does not match")
        ));
    }
}
