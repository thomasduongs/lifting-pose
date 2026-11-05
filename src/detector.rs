//! YOLOX-s COCO person detection using OpenCV Zoo's 2022nov ONNX export.
//! Preprocessing/decoding: https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox
use crate::pose::Bbox;
use anyhow::{Context, Result, ensure};
use opencv::{
    core::{self, Mat, Rect, Scalar, Size, Vector},
    dnn, imgproc,
    prelude::*,
};
use std::path::Path;

const INPUT: i32 = 640;
const ROWS: usize = 8400;
const COLUMNS: usize = 85;

#[derive(Clone, Copy, Debug)]
pub struct Detection {
    pub bbox: Bbox,
    pub score: f32,
}

pub struct PersonDetector {
    net: dnn::Net,
    output_names: Vector<String>,
}

impl PersonDetector {
    pub fn new(path: &Path) -> Result<Self> {
        ensure!(
            path.is_file(),
            "Person detector not found at {}. See README.md for the YOLOX-s model, or use --bbox / --no-detection.",
            path.display()
        );
        let mut net = opencv::opencv_branch_5! {
            { dnn::read_net_from_onnx_def(path) } else { dnn::read_net_from_onnx(path) }
        }
        .with_context(|| format!("Unable to load person detector {}", path.display()))?;
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        opencv::not_opencv_branch_5! { net.set_preferable_target(dnn::DNN_TARGET_CPU)?; }
        let output_names = net.get_unconnected_out_layers_names()?;
        ensure!(output_names.len() == 1, "Expected one raw YOLOX-s output");
        Ok(Self { net, output_names })
    }

    pub fn detect(
        &mut self,
        frame: &Mat,
        threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<Detection>> {
        ensure!(!frame.empty(), "Cannot detect people in an empty image");
        let scale = (INPUT as f32 / frame.cols() as f32).min(INPUT as f32 / frame.rows() as f32);
        let size = Size::new(
            (frame.cols() as f32 * scale) as i32,
            (frame.rows() as f32 * scale) as i32,
        );
        ensure!(
            size.width > 0 && size.height > 0,
            "Image aspect ratio is too extreme for person detection"
        );
        let mut resized = Mat::default();
        imgproc::resize(frame, &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;
        // Top-left alignment, 114 padding, RGB float32, and no normalization.
        let mut padded =
            Mat::new_rows_cols_with_default(INPUT, INPUT, core::CV_8UC3, Scalar::all(114.0))?;
        let mut roi = Mat::roi_mut(&mut padded, Rect::new(0, 0, size.width, size.height))?;
        resized.copy_to(&mut roi)?;
        let blob = dnn::blob_from_image(
            &padded,
            1.0,
            Size::new(INPUT, INPUT),
            Scalar::all(0.0),
            true,
            false,
            core::CV_32F,
        )?;
        self.net.set_input_def(&blob)?;
        let mut outputs = Vector::<Mat>::new();
        self.net.forward(&mut outputs, &self.output_names)?;
        ensure!(
            outputs.len() == 1,
            "YOLOX-s returned an unexpected output count"
        );
        let output = outputs.get(0)?;
        ensure!(
            output.mat_size().as_ref() == [1, ROWS as i32, COLUMNS as i32],
            "Expected raw YOLOX-s output [1,8400,85]; got {:?}",
            output.mat_size()
        );
        let candidates = decode(
            output.data_typed::<f32>()?,
            scale,
            frame.cols(),
            frame.rows(),
            threshold,
        )?;
        Ok(closest(suppress(candidates, nms_threshold))
            .into_iter()
            .collect())
    }
}

fn decode(
    data: &[f32],
    scale: f32,
    width: i32,
    height: i32,
    threshold: f32,
) -> Result<Vec<Detection>> {
    ensure!(
        data.len() == ROWS * COLUMNS,
        "Invalid YOLOX-s tensor length"
    );
    let mut candidates = Vec::new();
    let mut index = 0;
    for stride in [8, 16, 32] {
        let grid = INPUT / stride;
        for gy in 0..grid {
            for gx in 0..grid {
                let row = &data[index * COLUMNS..(index + 1) * COLUMNS];
                index += 1;
                let score = row[4] * row[5]; // objectness × COCO person score (class 0)
                if !score.is_finite() || score < threshold || row[6..].iter().any(|&v| v > row[5]) {
                    continue;
                }
                let cx = (row[0] + gx as f32) * stride as f32 / scale;
                let cy = (row[1] + gy as f32) * stride as f32 / scale;
                let bw = row[2].exp() * stride as f32 / scale;
                let bh = row[3].exp() * stride as f32 / scale;
                if ![cx, cy, bw, bh].iter().all(|v| v.is_finite()) {
                    continue;
                }
                let left = (cx - bw / 2.0).clamp(0.0, width as f32);
                let top = (cy - bh / 2.0).clamp(0.0, height as f32);
                let right = (cx + bw / 2.0).clamp(0.0, width as f32);
                let bottom = (cy + bh / 2.0).clamp(0.0, height as f32);
                if right - left >= 1.0 && bottom - top >= 1.0 {
                    candidates.push(Detection {
                        bbox: Bbox {
                            x: left,
                            y: top,
                            width: right - left,
                            height: bottom - top,
                        },
                        score,
                    });
                }
            }
        }
    }
    Ok(candidates)
}

fn iou(a: Bbox, b: Bbox) -> f32 {
    let width = ((a.x + a.width).min(b.x + b.width) - a.x.max(b.x)).max(0.0);
    let height = ((a.y + a.height).min(b.y + b.height) - a.y.max(b.y)).max(0.0);
    let intersection = width * height;
    intersection / (a.width * a.height + b.width * b.height - intersection).max(f32::EPSILON)
}

fn suppress(mut candidates: Vec<Detection>, threshold: f32) -> Vec<Detection> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for candidate in candidates {
        if kept
            .iter()
            .all(|p| iou(p.bbox, candidate.bbox) <= threshold)
        {
            kept.push(candidate);
        }
    }
    kept
}

/// Largest visible person box approximates proximity in a monocular image.
/// Confidence only breaks ties; apply NMS before selecting the person.
fn closest(candidates: Vec<Detection>) -> Option<Detection> {
    candidates.into_iter().max_by(|a, b| {
        (a.bbox.width * a.bbox.height)
            .total_cmp(&(b.bbox.width * b.bbox.height))
            .then_with(|| a.score.total_cmp(&b.score))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_filters_classes_and_undoes_letterbox_scale() {
        let mut data = vec![0.0; ROWS * COLUMNS];
        // Anchor (0,0), stride 8: center=(80,160), size=(80,160).
        data[..7].copy_from_slice(&[10.0, 20.0, 10.0_f32.ln(), 20.0_f32.ln(), 0.9, 0.8, 0.1]);
        // High-confidence non-person must be ignored.
        data[COLUMNS..COLUMNS + 7].copy_from_slice(&[10.0, 20.0, 0.0, 0.0, 1.0, 0.6, 0.9]);
        let detections = decode(&data, 0.5, 1280, 720, 0.4).unwrap();
        assert_eq!(detections.len(), 1);
        let bbox = detections[0].bbox;
        assert!((bbox.x - 80.0).abs() < 0.001);
        assert!((bbox.y - 160.0).abs() < 0.001);
        assert!((bbox.width - 160.0).abs() < 0.001);
        assert!((bbox.height - 320.0).abs() < 0.001);
        assert!((detections[0].score - 0.72).abs() < 1e-6);
        assert!(decode(&data, 0.5, 1280, 720, 0.99).unwrap().is_empty());
    }

    #[test]
    fn nms_removes_duplicate_boxes_but_keeps_separate_people() {
        let make = |x, score| Detection {
            bbox: Bbox {
                x,
                y: 0.0,
                width: 100.0,
                height: 200.0,
            },
            score,
        };
        let input = vec![make(0.0, 0.7), make(5.0, 0.9), make(300.0, 0.8)];
        let result = suppress(input, 0.45);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].score, 0.9);
        assert_eq!(result[1].bbox.x, 300.0);
        assert!(suppress(Vec::new(), 0.45).is_empty());
    }
    #[test]
    fn closest_selects_largest_box_over_highest_confidence() {
        let make = |width, height, score| Detection {
            bbox: Bbox {
                x: 0.0,
                y: 0.0,
                width,
                height,
            },
            score,
        };
        let selected = closest(vec![make(100.0, 200.0, 0.99), make(200.0, 300.0, 0.6)]).unwrap();
        assert_eq!(selected.bbox.width, 200.0);
        assert_eq!(selected.score, 0.6);
        assert_eq!(
            closest(vec![make(200.0, 300.0, 0.7), make(200.0, 300.0, 0.9)])
                .unwrap()
                .score,
            0.9
        );
        assert!(closest(Vec::new()).is_none());
    }
}
