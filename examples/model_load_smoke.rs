use std::time::Instant;

use mnn_runtime::{Runtime, RuntimeConfig, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MNN {}", Runtime::native_version());
    let runtime = Runtime::new(RuntimeConfig::new().with_threads(4))?;
    let models: [(&str, &[u8]); 6] = [
        (
            "face detector",
            include_bytes!("../models/face-detection-retail-0004.mnn"),
        ),
        (
            "facial landmarks",
            include_bytes!("../models/facial-landmarks-35-adas-0002.mnn"),
        ),
        (
            "head pose",
            include_bytes!("../models/head-pose-estimation-adas-0001.mnn"),
        ),
        (
            "eye state",
            include_bytes!("../models/open-closed-eye-0001.mnn"),
        ),
        (
            "gaze estimation",
            include_bytes!("../models/gaze-estimation-adas-0002-packed.mnn"),
        ),
        (
            "face reidentification",
            include_bytes!("../models/face-reidentification-retail-0095.mnn"),
        ),
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
