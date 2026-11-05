use anyhow::{Result, ensure};
use opencv::{
    core::{Mat, Vector},
    dnn,
    prelude::*,
};
fn main() -> Result<()> {
    let read = |path| -> Result<Vec<f32>> {
        Ok(std::fs::read(path)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    };
    let input = read("models/motionagformer-xs.input.f32")?;
    let expected = read("models/motionagformer-xs.expected.f32")?;
    let mut net = opencv::opencv_branch_5! { { dnn::read_net_from_onnx_def("models/motionagformer-xs.onnx")? } else { dnn::read_net_from_onnx("models/motionagformer-xs.onnx")? } };
    let mat = Mat::from_slice(&input)?;
    let blob = mat.reshape_nd(1, &[1, 27, 17, 3])?;
    net.set_input_def(&blob)?;
    let mut outputs = Vector::<Mat>::new();
    net.forward(&mut outputs, &net.get_unconnected_out_layers_names()?)?;
    let output = outputs.get(0)?;
    println!("output shape: {:?}", output.mat_size());
    let values = output.data_typed::<f32>()?;
    let error = values
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    ensure!(
        values.len() == expected.len() && error < 0.001,
        "Runtime parity failed: {error}"
    );
    println!("OpenCV/PyTorch max absolute error: {error}");
    Ok(())
}
