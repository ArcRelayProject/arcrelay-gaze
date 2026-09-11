use std::path::PathBuf;
use std::time::Instant;

use arcrelay_gaze::{OwnedModelBundle, MODEL_BUNDLE_VERSION};
use mnn_runtime::{Runtime, RuntimeConfig, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MNN {}", Runtime::native_version());
    let directory = std::env::var_os("ARCRELAY_GAZE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));
    println!(
        "gaze model bundle {MODEL_BUNDLE_VERSION}: {}",
        directory.display()
    );
    let bundle = OwnedModelBundle::load_from_directory(&directory)?;
    let bundle = bundle.as_borrowed();
    let runtime = Runtime::new(RuntimeConfig::new().with_threads(4))?;
    let models = [
        ("face detector", bundle.face),
        ("facial landmarks", bundle.landmarks),
        ("head pose", bundle.head_pose),
        ("eye state", bundle.eye_state),
        ("gaze estimation", bundle.gaze),
        ("face reidentification", bundle.face_embedding),
    ];
    let mut loaded = Vec::with_capacity(models.len());
    for (name, bytes) in models {
        let started = Instant::now();
        println!("loading {name} ({} bytes)", bytes.len());
        let model = runtime.load_bytes(bytes.to_vec())?;
        println!(
            "loaded {name} in {} ms ({} inputs, {} outputs)",
            started.elapsed().as_millis(),
            model.info().inputs().len(),
            model.info().outputs().len()
        );
        let inputs = model
            .info()
            .inputs()
            .iter()
            .map(|input| {
                let shape = input.concrete_shape()?;
                let values = vec![0.0; shape.iter().product()];
                Tensor::new(input.name(), shape, values)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let started = Instant::now();
        println!("running {name}");
        let outputs = model.run_owned(inputs)?;
        println!(
            "ran {name} in {} ms ({} outputs)",
            started.elapsed().as_millis(),
            outputs.len()
        );
        loaded.push(model);
    }
    println!("loaded all {} models", loaded.len());
    Ok(())
}
