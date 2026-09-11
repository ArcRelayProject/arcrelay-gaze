use std::time::Instant;

use mnn_runtime::{Model, Runtime, Tensor};

use crate::{Error, Result};

pub(crate) struct MnnModel {
    model: Model,
    input_name: String,
    input_shape: Vec<usize>,
}

impl MnnModel {
    pub(crate) fn load(runtime: &Runtime, name: &str, bytes: &[u8]) -> Result<Self> {
        let started = Instant::now();
        tracing::info!(model = name, bytes = bytes.len(), "loading MNN model");
        let model = runtime.load_bytes(bytes.to_vec()).map_err(|error| {
            tracing::error!(model = name, %error, "failed to load MNN model");
            Error::Model(format!("load {name}: {error}"))
        })?;
        let inputs = model.info().inputs();
        if inputs.len() != 1 {
            return Err(Error::Model(format!(
                "{name} has {} inputs; this converted stage expects one",
                inputs.len()
            )));
        }
        let input = &inputs[0];
        if input.is_channel_last() {
            return Err(Error::Model(format!(
                "{name} input {:?} is channel-last; NCHW is required",
                input.name()
            )));
        }
        let input_name = input.name().to_owned();
        let input_shape = input
            .concrete_shape()
            .map_err(|error| Error::Model(format!("resolve {name} input: {error}")))?;
        tracing::info!(
            model = name,
            elapsed_ms = started.elapsed().as_millis() as u64,
            ?input_shape,
            "loaded MNN model"
        );
        Ok(Self {
            model,
            input_name,
            input_shape,
        })
    }

    pub(crate) fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub(crate) fn run(&self, input: Vec<f32>) -> Result<Vec<NamedTensor>> {
        let expected = self.input_shape.iter().product::<usize>();
        if input.len() != expected {
            return Err(Error::Model(format!(
                "input has {} values, expected {expected} for {:?}",
                input.len(),
                self.input_shape
            )));
        }
        let tensor = Tensor::new(self.input_name.clone(), self.input_shape.clone(), input)
            .map_err(|error| Error::Model(error.to_string()))?;
        self.model
            .run_owned(vec![tensor])
            .map_err(|error| Error::Model(error.to_string()))
            .map(|outputs| {
                outputs
                    .into_iter()
                    .map(|output| NamedTensor {
                        name: output.name().to_owned(),
                        shape: output.shape().to_vec(),
                        values: output.into_data(),
                    })
                    .collect()
            })
    }

    pub(crate) fn run_one(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        let mut outputs = self.run(input)?;
        if outputs.len() != 1 {
            return Err(Error::Model(format!(
                "expected one output, got {}",
                outputs.len()
            )));
        }
        Ok(outputs.remove(0).values)
    }
}

#[derive(Debug)]
pub(crate) struct NamedTensor {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) values: Vec<f32>,
}

pub(crate) fn find_output<'a>(outputs: &'a [NamedTensor], needle: &str) -> Option<&'a NamedTensor> {
    outputs.iter().find(|output| output.name.contains(needle))
}

pub(crate) fn find_output_len(outputs: &[NamedTensor], length: usize) -> Option<&NamedTensor> {
    outputs.iter().find(|output| output.values.len() == length)
}
