use std::path::Path;

use sha2::{Digest, Sha256};

use crate::{Error, ModelBundle, Result};

/// Version of the architecture-independent model bundle expected by this crate.
pub const MODEL_BUNDLE_VERSION: &str = "1.0.0";

/// Metadata used to validate one downloaded model before MNN sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelFile {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// Complete list of files in the gaze model bundle.
pub const MODEL_FILES: [ModelFile; 6] = [
    ModelFile {
        name: "face-detection-retail-0004.mnn",
        size: 2_363_656,
        sha256: "7f73526600c83ecf30b480e283ed612f0f8b9716a0257b0822d41cf0de8d7550",
    },
    ModelFile {
        name: "face-reidentification-retail-0095.mnn",
        size: 4_454_904,
        sha256: "455a55d7ebced693ee44230a7c832f7321674bf2f47604380f7f14a602dfd92e",
    },
    ModelFile {
        name: "facial-landmarks-35-adas-0002.mnn",
        size: 18_431_648,
        sha256: "c956b4a7f41eccbffe3f016aedbf0ae04c1a735db13665a0359f7cd9ed38537d",
    },
    ModelFile {
        name: "gaze-estimation-adas-0002-packed.mnn",
        size: 7_541_236,
        sha256: "4d9d3fe0a752a6afda45de76a73907d1f8dad3183230aec14ce01ab752d76d53",
    },
    ModelFile {
        name: "head-pose-estimation-adas-0001.mnn",
        size: 7_660_080,
        sha256: "2b0c4786462845997cd11622854a137ba31e1024fcd9bfa4d0ea107761e8d1b7",
    },
    ModelFile {
        name: "open-closed-eye-0001.mnn",
        size: 47_040,
        sha256: "f196cf4f906a3554515d0b82983e80339e6cc44b59b818458f3ca0f190a76f18",
    },
];

/// Model bytes owned by the caller while MNN creates its model workers.
pub struct OwnedModelBundle {
    face: Vec<u8>,
    landmarks: Vec<u8>,
    head_pose: Vec<u8>,
    eye_state: Vec<u8>,
    gaze: Vec<u8>,
    face_embedding: Vec<u8>,
}

impl OwnedModelBundle {
    /// Read and verify every model in an extracted bundle directory.
    pub fn load_from_directory(directory: &Path) -> Result<Self> {
        let read = |name| {
            let specification = MODEL_FILES
                .iter()
                .find(|file| file.name == name)
                .expect("every pipeline model must have metadata");
            read_verified(directory, specification)
        };
        Ok(Self {
            face: read("face-detection-retail-0004.mnn")?,
            landmarks: read("facial-landmarks-35-adas-0002.mnn")?,
            head_pose: read("head-pose-estimation-adas-0001.mnn")?,
            eye_state: read("open-closed-eye-0001.mnn")?,
            gaze: read("gaze-estimation-adas-0002-packed.mnn")?,
            face_embedding: read("face-reidentification-retail-0095.mnn")?,
        })
    }

    #[must_use]
    pub fn as_borrowed(&self) -> ModelBundle<'_> {
        ModelBundle {
            face: &self.face,
            landmarks: &self.landmarks,
            head_pose: &self.head_pose,
            eye_state: &self.eye_state,
            gaze: &self.gaze,
            face_embedding: &self.face_embedding,
        }
    }
}

fn read_verified(directory: &Path, specification: &ModelFile) -> Result<Vec<u8>> {
    let path = directory.join(specification.name);
    let bytes = std::fs::read(&path)
        .map_err(|error| Error::Model(format!("read {}: {error}", path.display())))?;
    if bytes.len() as u64 != specification.size {
        return Err(Error::Model(format!(
            "{} has {} bytes, expected {}",
            specification.name,
            bytes.len(),
            specification.size
        )));
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != specification.sha256 {
        return Err(Error::Model(format!(
            "{} failed SHA-256 verification",
            specification.name
        )));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_metadata_is_complete_and_unique() {
        let names = MODEL_FILES.iter().map(|file| file.name).collect::<Vec<_>>();
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), MODEL_FILES.len());
        assert!(MODEL_FILES
            .iter()
            .all(|file| file.size > 0 && file.sha256.len() == 64));
    }
}
