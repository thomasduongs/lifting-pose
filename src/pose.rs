use crate::landmarks::{BODY_KEYPOINTS, BONES, NAMES, WHOLEBODY_KEYPOINTS};
use anyhow::{Context, Result, ensure};
use opencv::{
    core::{self, Mat, Point, Scalar, Size, Vec3f, Vector},
    dnn, imgproc,
    prelude::*,
};
use serde::Serialize;
use std::path::Path;

const WIDTH: i32 = 192;
const HEIGHT: i32 = 256;
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Keypoint {
    pub name: &'static str,
    pub x: f32,
    pub y: f32,
    /// Raw SimCC score, not a calibrated probability.
    pub score: f32,
}

/// Person rectangle in original-frame pixel coordinates.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Bbox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Copy, Debug)]
struct Crop {
    cx: f32,
    cy: f32,
    width: f32,
    height: f32,
}

impl Crop {
    fn new(bbox: Bbox) -> Self {
        let aspect = WIDTH as f32 / HEIGHT as f32;
        let width = bbox.width.max(bbox.height * aspect) * 1.25;
        let height = bbox.height.max(bbox.width / aspect) * 1.25;
        Self {
            cx: bbox.x + bbox.width / 2.0,
            cy: bbox.y + bbox.height / 2.0,
            width,
            height,
        }
    }

    fn to_frame(self, x: f32, y: f32) -> (f32, f32) {
        (
            x / WIDTH as f32 * self.width + self.cx - self.width / 2.0,
            y / HEIGHT as f32 * self.height + self.cy - self.height / 2.0,
        )
    }
}

pub struct PoseEstimator {
    net: dnn::Net,
    output_names: Vector<String>,
    mediapipe: bool,
    direct_pose: Option<crate::motion3d::Pose3d>,
    upper_body: bool,
    mediapipe_center: Option<f32>,
}

impl PoseEstimator {
    pub fn new(path: &Path) -> Result<Self> {
        ensure!(
            path.is_file(),
            "Model not found at {}. Place a supported MediaPipe Pose, RTMW-m or RTMPose-m ONNX model there or specify --model /path/to/model.onnx. See README.md for the model source.",
            path.display()
        );
        let mut net = opencv::opencv_branch_5! {
            { dnn::read_net_from_onnx_def(path) } else
            { dnn::read_net_from_onnx(path) }
        }
        .with_context(|| format!("Unable to load ONNX model {}", path.display()))?;
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        // OpenCV 5's graph engine runs on CPU and does not implement target selection.
        opencv::not_opencv_branch_5! { net.set_preferable_target(dnn::DNN_TARGET_CPU)?; }
        let output_names = net.get_unconnected_out_layers_names()?;
        ensure!(
            matches!(output_names.len(), 2 | 5),
            "Expected MediaPipe Pose (five outputs) or RTM SimCC (two outputs); found {}.",
            output_names.len()
        );
        Ok(Self {
            mediapipe: output_names.len() == 5,
            net,
            output_names,
            direct_pose: None,
            upper_body: true,
            mediapipe_center: None,
        })
    }

    pub fn is_mediapipe(&self) -> bool {
        self.mediapipe
    }
    pub fn set_upper_body(&mut self, upper_body: bool) {
        self.upper_body = upper_body;
    }
    pub fn take_pose3d(&mut self) -> Option<crate::motion3d::Pose3d> {
        self.direct_pose.take()
    }

    pub fn infer(&mut self, image: &Mat, bbox: Bbox) -> Result<Vec<Keypoint>> {
        ensure!(!image.empty(), "Cannot estimate pose in an empty image");
        ensure!(
            bbox.width > 0.0 && bbox.height > 0.0,
            "Crop must have positive dimensions"
        );
        self.direct_pose = None;
        if self.mediapipe {
            let mut roi = bbox;
            let center = self
                .mediapipe_center
                .filter(|x| *x >= bbox.x && *x <= bbox.x + bbox.width);
            if let Some(center) = center {
                roi.x = center - bbox.width / 2.0;
            }
            let (mut points, mut pose) = crate::mediapipe::infer(
                &mut self.net,
                &self.output_names,
                image,
                roi,
                self.upper_body,
            )?;
            let shoulder_center = |points: &[Keypoint]| {
                (points[5].score.min(points[6].score) >= 0.6)
                    .then_some((points[5].x + points[6].x) / 2.0)
            };
            // Bootstrap a torso-centered ROI when a raised arm makes the
            // detector box asymmetric. Later frames reuse the measured center.
            if self.upper_body && center.is_none() {
                if let Some(center) = shoulder_center(&points) {
                    if (center - (bbox.x + bbox.width / 2.0)).abs() > bbox.height * 0.15 {
                        roi.x = center - bbox.width / 2.0;
                        let refined = crate::mediapipe::infer(
                            &mut self.net,
                            &self.output_names,
                            image,
                            roi,
                            self.upper_body,
                        )?;
                        if refined.1.is_some() {
                            (points, pose) = refined;
                        }
                    }
                }
            }
            self.mediapipe_center = shoulder_center(&points);
            self.direct_pose = pose;
            return Ok(points);
        }
        let crop = Crop::new(bbox);
        let sx = f64::from(WIDTH) / f64::from(crop.width);
        let sy = f64::from(HEIGHT) / f64::from(crop.height);
        let affine = Mat::from_slice_2d(&[
            [sx, 0.0, f64::from(WIDTH) / 2.0 - sx * f64::from(crop.cx)],
            [0.0, sy, f64::from(HEIGHT) / 2.0 - sy * f64::from(crop.cy)],
        ])?;
        let mut warped = Mat::default();
        // Defaults are bilinear interpolation, constant zero border, and no inverse map.
        imgproc::warp_affine_def(image, &mut warped, &affine, Size::new(WIDTH, HEIGHT))?;
        let mut rgb = Mat::default();
        imgproc::cvt_color_def(&warped, &mut rgb, imgproc::COLOR_BGR2RGB)?;
        let mut normalized = Mat::default();
        rgb.convert_to(&mut normalized, core::CV_32F, 1.0, 0.0)?;
        for pixel in normalized.data_typed_mut::<Vec3f>()? {
            for channel in 0..3 {
                pixel[channel] = (pixel[channel] - MEAN[channel]) / STD[channel];
            }
        }
        let blob = dnn::blob_from_image(
            &normalized,
            1.0,
            Size::new(WIDTH, HEIGHT),
            Scalar::all(0.0),
            false,
            false,
            core::CV_32F,
        )?;
        self.net.set_input_def(&blob)?;
        let mut outputs = Vector::<Mat>::new();
        self.net.forward(&mut outputs, &self.output_names)?;
        ensure!(
            outputs.len() == 2,
            "Inference did not return two SimCC outputs"
        );
        let a = outputs.get(0)?;
        let b = outputs.get(1)?;
        // Identify axes by shape rather than relying on ONNX output ordering.
        let (x, y) = if a.mat_size().last() == Some(&(WIDTH * 2)) {
            (&a, &b)
        } else {
            (&b, &a)
        };
        let xs = x.mat_size();
        let ys = y.mat_size();
        ensure!(
            xs.len() == 3
                && ys.len() == 3
                && xs[0] == 1
                && ys[0] == 1
                && xs[1] == ys[1]
                && matches!(xs[1], 17 | 133)
                && xs[2] == WIDTH * 2
                && ys[2] == HEIGHT * 2,
            "Expected SimCC [1,K,384] and [1,K,512], K=17 or 133; got {:?} and {:?}",
            a.mat_size(),
            b.mat_size()
        );
        decode(
            x.data_typed::<f32>()?,
            y.data_typed::<f32>()?,
            crop,
            xs[1] as usize,
        )
    }
}

fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .copied()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
}

fn decode(x: &[f32], y: &[f32], crop: Crop, count: usize) -> Result<Vec<Keypoint>> {
    ensure!(
        matches!(count, BODY_KEYPOINTS | WHOLEBODY_KEYPOINTS),
        "Unsupported landmark count: {count}"
    );
    let nx = (WIDTH * 2) as usize;
    let ny = (HEIGHT * 2) as usize;
    ensure!(
        x.len() == count * nx && y.len() == count * ny,
        "Invalid SimCC tensor lengths"
    );
    ensure!(
        x.iter().chain(y).all(|v| v.is_finite()),
        "Model produced non-finite SimCC values"
    );
    Ok((0..count)
        .map(|k| {
            let (ix, sx) = argmax(&x[k * nx..(k + 1) * nx]);
            let (iy, sy) = argmax(&y[k * ny..(k + 1) * ny]);
            let score = sx.min(sy);
            let (px, py) = if score > 0.0 {
                crop.to_frame(ix as f32 / 2.0, iy as f32 / 2.0)
            } else {
                (-1.0, -1.0)
            };
            Keypoint {
                name: NAMES[k],
                x: px,
                y: py,
                score,
            }
        })
        .collect())
}

pub fn draw(image: &mut Mat, points: &[Keypoint], threshold: f32) -> Result<()> {
    ensure!(
        matches!(points.len(), BODY_KEYPOINTS | WHOLEBODY_KEYPOINTS),
        "Unsupported landmark count for drawing"
    );
    let visible = |p: &Keypoint| {
        p.score > 0.0
            && p.score >= threshold
            && p.x >= 0.0
            && p.x < image.cols() as f32
            && p.y >= 0.0
            && p.y < image.rows() as f32
    };
    let visibility: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(i, p)| visible(p) && !matches!(i,0..=4 | 17..=90))
        .collect();
    let point = |p: &Keypoint| Point::new(p.x.round() as i32, p.y.round() as i32);
    for &(a, b) in BONES {
        if a < points.len() && b < points.len() && visibility[a] && visibility[b] {
            imgproc::line(
                image,
                point(&points[a]),
                point(&points[b]),
                landmark_color(b),
                2,
                imgproc::LINE_AA,
                0,
            )?;
        }
    }
    for (index, (p, visible)) in points.iter().zip(visibility).enumerate() {
        if visible {
            imgproc::circle(
                image,
                point(p),
                if index < 23 { 4 } else { 2 },
                landmark_color(index),
                -1,
                imgproc::LINE_AA,
                0,
            )?;
        }
    }
    Ok(())
}

fn landmark_color(index: usize) -> Scalar {
    match index {
        17..=22 => Scalar::new(255.0, 220.0, 40.0, 0.0), // Feet
        23..=90 => Scalar::new(180.0, 180.0, 180.0, 0.0), // Face
        91..=111 => Scalar::new(40.0, 170.0, 255.0, 0.0), // Left hand
        112..=132 => Scalar::new(255.0, 100.0, 220.0, 0.0), // Right hand
        _ => Scalar::new(70.0, 220.0, 70.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_maps_simcc_peaks_back_to_original_crop() {
        let crop = Crop::new(Bbox {
            x: 100.0,
            y: 200.0,
            width: 300.0,
            height: 400.0,
        });
        let mut x = vec![0.0; BODY_KEYPOINTS * 384];
        let mut y = vec![0.0; BODY_KEYPOINTS * 512];
        x[192] = 0.8;
        y[256] = 0.6;
        let points = decode(&x, &y, crop, BODY_KEYPOINTS).unwrap();
        assert_eq!((points[0].x, points[0].y), (250.0, 400.0));
        assert_eq!(points[0].score, 0.6);
        assert_eq!((points[1].x, points[1].y), (-1.0, -1.0));
        let (left, top) = crop.to_frame(0.0, 0.0);
        assert_eq!((left, top), (62.5, 150.0));
    }

    #[test]
    fn crop_preserves_model_aspect_ratio_for_wide_frames() {
        let crop = Crop::new(Bbox {
            x: 0.0,
            y: 0.0,
            width: 1920.0,
            height: 1080.0,
        });
        assert!((crop.width / crop.height - 0.75).abs() < 1e-6);
        assert_eq!(crop.width, 2400.0);
    }

    #[test]
    fn wholebody_decodes_hands_and_feet_and_draws_both_layouts() {
        let crop = Crop::new(Bbox {
            x: 0.0,
            y: 0.0,
            width: 192.0,
            height: 256.0,
        });
        let mut x = vec![0.0; WHOLEBODY_KEYPOINTS * 384];
        let mut y = vec![0.0; WHOLEBODY_KEYPOINTS * 512];
        for k in 0..WHOLEBODY_KEYPOINTS {
            x[k * 384 + 96 + k] = 0.9;
            y[k * 512 + 128 + k] = 0.7;
        }
        let points = decode(&x, &y, crop, WHOLEBODY_KEYPOINTS).unwrap();
        assert_eq!(points.len(), 133);
        for (index, name) in [
            (17, "left_big_toe"),
            (18, "left_small_toe"),
            (19, "left_heel"),
            (20, "right_big_toe"),
            (21, "right_small_toe"),
            (22, "right_heel"),
            (91, "left_hand_root"),
            (95, "left_thumb4"),
            (111, "left_pinky_finger4"),
            (112, "right_hand_root"),
            (116, "right_thumb4"),
            (132, "right_pinky_finger4"),
        ] {
            assert_eq!(points[index].name, name);
            let expected = crop.to_frame((96 + index) as f32 / 2.0, (128 + index) as f32 / 2.0);
            assert_eq!((points[index].x, points[index].y), expected);
            assert_eq!(points[index].score, 0.7);
        }
        let mut image =
            Mat::new_rows_cols_with_default(256, 192, core::CV_8UC3, Scalar::all(0.0)).unwrap();
        draw(&mut image, &points, 0.3).unwrap();
        draw(&mut image, &points[..BODY_KEYPOINTS], 0.3).unwrap();
    }

    #[test]
    fn decode_rejects_wrong_landmark_count_or_tensor_lengths() {
        let crop = Crop::new(Bbox {
            x: 0.0,
            y: 0.0,
            width: 192.0,
            height: 256.0,
        });
        let x = vec![0.0; BODY_KEYPOINTS * 384];
        let y = vec![0.0; BODY_KEYPOINTS * 512];
        assert!(decode(&x, &y, crop, WHOLEBODY_KEYPOINTS).is_err());
        assert!(decode(&x, &y, crop, 21).is_err());
    }
    #[test]
    fn overlay_hides_face_and_feet_but_keeps_hand_landmarks() {
        let mut points = vec![
            Keypoint {
                name: "test",
                x: -1.0,
                y: -1.0,
                score: 0.0
            };
            133
        ];
        for i in [0, 1, 17, 23] {
            points[i] = Keypoint {
                name: "detail",
                x: 20.0 + i as f32,
                y: 20.0,
                score: 1.0,
            };
        }
        let mut image =
            Mat::new_rows_cols_with_default(160, 160, core::CV_8UC3, Scalar::all(0.0)).unwrap();
        draw(&mut image, &points, 0.3).unwrap();
        assert!(image.data_bytes().unwrap().iter().all(|v| *v == 0));
        points[91] = Keypoint {
            name: "hand",
            x: 100.0,
            y: 100.0,
            score: 1.0,
        };
        draw(&mut image, &points, 0.3).unwrap();
        assert!(image.data_bytes().unwrap().iter().any(|v| *v != 0));
    }
}
