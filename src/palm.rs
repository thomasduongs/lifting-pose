//! Independent MediaPipe palm detection for reliable hand-model crops.
use anyhow::{Result, ensure};
use opencv::{
    core::{self, Mat, Size, Vector},
    dnn, imgproc,
    prelude::*,
};
use std::path::Path;

pub struct Palm {
    pub landmarks: [[f32; 2]; 7],
    bbox: [f32; 4],
    pub score: f32,
}
pub struct PalmDetector {
    net: dnn::Net,
    names: Vector<String>,
}
impl PalmDetector {
    pub fn new(path: &Path) -> Result<Self> {
        ensure!(
            path.is_file(),
            "Palm detector missing at {}. Download the documented MediaPipe palm model.",
            path.display()
        );
        let path = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid palm model path"))?;
        let mut net = opencv::opencv_branch_5! {{dnn::read_net_from_onnx_def(path)}else{dnn::read_net_from_onnx(path)}}?;
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        opencv::not_opencv_branch_5! {net.set_preferable_target(dnn::DNN_TARGET_CPU)?;}
        let names = net.get_unconnected_out_layers_names()?;
        Ok(Self { net, names })
    }
    pub fn infer(&mut self, image: &Mat) -> Result<Vec<Palm>> {
        let size = image.size()?;
        let scale = 192.0 / f64::from(size.width.max(size.height));
        let pad = [
            (192.0 - f64::from(size.width) * scale) / 2.0,
            (192.0 - f64::from(size.height) * scale) / 2.0,
        ];
        let affine = Mat::from_slice_2d(&[[scale, 0.0, pad[0]], [0.0, scale, pad[1]]])?;
        let mut crop = Mat::default();
        imgproc::warp_affine_def(image, &mut crop, &affine, Size::new(192, 192))?;
        let mut rgb = Mat::default();
        imgproc::cvt_color_def(&crop, &mut rgb, imgproc::COLOR_BGR2RGB)?;
        let mut float = Mat::default();
        rgb.convert_to(&mut float, core::CV_32F, 1.0 / 255.0, 0.0)?;
        self.net
            .set_input_def(&float.reshape_nd(1, &[1, 192, 192, 3])?)?;
        let mut outputs = Vector::<Mat>::new();
        self.net.forward(&mut outputs, &self.names)?;
        let mut boxes = None;
        let mut scores = None;
        for output in outputs {
            match output.total() {
                36288 => boxes = Some(output.data_typed::<f32>()?.to_vec()),
                2016 => scores = Some(output.data_typed::<f32>()?.to_vec()),
                _ => (),
            }
        }
        ensure!(
            boxes.is_some() && scores.is_some(),
            "Unexpected MediaPipe palm detector outputs"
        );
        Ok(decode(
            &boxes.unwrap(),
            &scores.unwrap(),
            scale as f32,
            pad.map(|v| v as f32),
        ))
    }
}
fn anchor(index: usize) -> [f32; 2] {
    let (cell, grid) = if index < 1152 {
        (index / 2, 24)
    } else {
        ((index - 1152) / 6, 12)
    };
    [
        ((cell % grid) as f32 + 0.5) / grid as f32,
        ((cell / grid) as f32 + 0.5) / grid as f32,
    ]
}
fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    intersection
        / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection).max(1e-6)
}
fn decode(boxes: &[f32], scores: &[f32], scale: f32, pad: [f32; 2]) -> Vec<Palm> {
    let mut candidates = Vec::new();
    for (i, &logit) in scores.iter().enumerate() {
        let score = 1.0 / (1.0 + (-logit.clamp(-80.0, 80.0)).exp());
        if !logit.is_finite() || score < 0.6 {
            continue;
        }
        let b = &boxes[i * 18..i * 18 + 18];
        if b.iter().any(|v| !v.is_finite()) || b[2] <= 0.0 || b[3] <= 0.0 {
            continue;
        }
        let anchor = anchor(i).map(|v| v * 192.0);
        let center = [anchor[0] + b[0], anchor[1] + b[1]];
        let bbox = [
            (center[0] - b[2] / 2.0 - pad[0]) / scale,
            (center[1] - b[3] / 2.0 - pad[1]) / scale,
            (center[0] + b[2] / 2.0 - pad[0]) / scale,
            (center[1] + b[3] / 2.0 - pad[1]) / scale,
        ];
        let landmarks = std::array::from_fn(|j| {
            [
                (anchor[0] + b[4 + j * 2] - pad[0]) / scale,
                (anchor[1] + b[5 + j * 2] - pad[1]) / scale,
            ]
        });
        candidates.push(Palm {
            bbox,
            landmarks,
            score,
        });
    }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Palm> = Vec::new();
    for palm in candidates {
        if kept.iter().all(|p| iou(p.bbox, palm.bbox) < 0.3) {
            kept.push(palm);
            if kept.len() == 6 {
                break;
            }
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anchors_decode_letterboxing_and_suppress_duplicate_palms() {
        assert_eq!(anchor(0), [0.5 / 24.0, 0.5 / 24.0]);
        assert_eq!(anchor(1152), [0.5 / 12.0, 0.5 / 12.0]);
        assert_eq!(anchor(2015), [11.5 / 12.0, 11.5 / 12.0]);
        let mut boxes = vec![0.0; 2016 * 18];
        let mut scores = vec![-100.0; 2016];
        for i in [0, 1] {
            scores[i] = 5.0;
            boxes[i * 18] = 92.0;
            boxes[i * 18 + 1] = 92.0;
            boxes[i * 18 + 2] = 40.0;
            boxes[i * 18 + 3] = 40.0;
        }
        let palms = decode(&boxes, &scores, 0.15, [0.0, 42.0]);
        assert_eq!(palms.len(), 1);
        assert!((palms[0].bbox[0] - 506.66666).abs() < 0.01);
        assert!((palms[0].bbox[1] - 226.66666).abs() < 0.01);
    }
}
