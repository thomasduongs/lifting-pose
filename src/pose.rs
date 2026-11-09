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
    accelerated: crate::accelerated::Inference,
    model_path: std::path::PathBuf,
    output_names: Vector<String>,
    mediapipe: bool,
    direct_pose: Option<crate::motion3d::Pose3d>,
    upper_body: bool,
    mediapipe_center: Option<f32>,
    image_motion: crate::image_motion::ImageMotion,
    hand_crop_wrists: Option<[Keypoint; 2]>,
    arm_refiner: Option<Box<PoseEstimator>>,
    arm_refined: Vec<usize>,
    joint_tracker: crate::joint_tracking::JointTracker,
    stabilization: bool,
    smoothing: bool,
}

impl PoseEstimator {
    pub fn enable_acceleration(&mut self) {
        // Native MediaPipe's five-output path still uses its existing decoder;
        // the live 2D path uses RTM when available.
        if self.mediapipe {
            return;
        }
        if let Err(error) = self.accelerated.enable(&self.model_path) {
            eprintln!(
                "Native inference unavailable for {}: {error:#}; using OpenCV",
                self.model_path.display()
            );
        }
    }
    pub fn confirm_hands(&mut self, hands: &[crate::details::HandPose]) {
        self.image_motion.confirm_hands(hands);
    }
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
            accelerated: crate::accelerated::Inference::default(),
            model_path: path.to_path_buf(),
            mediapipe: output_names.len() == 5,
            net,
            output_names,
            direct_pose: None,
            upper_body: true,
            mediapipe_center: None,
            image_motion: crate::image_motion::ImageMotion::default(),
            hand_crop_wrists: None,
            arm_refiner: None,
            arm_refined: Vec::new(),
            joint_tracker: crate::joint_tracking::JointTracker::default(),
            stabilization: true,
            smoothing: true,
        })
    }

    /// Independent image measurements improve arm localization without replacing
    /// MediaPipe's torso/head or pretending that the auxiliary model measures depth.
    pub fn set_arm_refiner(&mut self, path: &Path) -> Result<()> {
        ensure!(
            self.mediapipe,
            "Arm refinement requires a MediaPipe primary model"
        );
        let model = Self::new(path)?;
        ensure!(
            !model.mediapipe,
            "Arm refinement requires an RTMPose/RTMW SimCC model"
        );
        self.arm_refiner = Some(Box::new(model));
        Ok(())
    }
    pub fn arm_refined(&self) -> &[usize] {
        &self.arm_refined
    }

    pub fn set_stabilization(&mut self, enabled: bool, smoothing: bool) {
        self.stabilization = enabled;
        self.smoothing = smoothing;
        self.joint_tracker.reset();
    }
    pub fn stabilize(
        &mut self,
        image: &Mat,
        points: &mut [Keypoint],
        time: f64,
        bbox: Bbox,
        hands: &[crate::details::HandPose],
        occluded: &[bool; 17],
    ) -> Result<crate::joint_tracking::TrackingFrame> {
        if !self.stabilization {
            return Ok(crate::joint_tracking::TrackingFrame::default());
        }
        let wrists = ["left", "right"]
            .map(|side| hands.iter().any(|h| h.side == side && h.confidence >= 0.85));
        let result = self.joint_tracker.apply(
            image,
            points,
            time,
            bbox,
            wrists,
            occluded,
            self.smoothing,
        )?;
        let corrections: Vec<_> = result
            .updates
            .iter()
            .filter(|u| u.source == "image_tracked" || u.source == "predicted")
            .map(|u| u.joint)
            .collect();
        if let Some(pose) = &mut self.direct_pose {
            for coco in corrections {
                let mapped: &[usize] = match coco {
                    0 => &[9],
                    3 | 4 => &[10],
                    5 => &[11, 8],
                    6 => &[14, 8],
                    7 => &[12],
                    8 => &[15],
                    9 => &[13],
                    10 => &[16],
                    11 => &[4, 0, 7],
                    12 => &[1, 0, 7],
                    13 => &[5],
                    14 => &[2],
                    15 => &[6],
                    16 => &[3],
                    _ => &[],
                };
                for &joint in mapped {
                    pose.joints[joint].input_score = 0.0;
                }
            }
        }
        Ok(result)
    }
    pub fn warm_up(&mut self) -> Result<()> {
        let image = Mat::new_rows_cols_with_default(256, 256, core::CV_8UC3, Scalar::all(0.0))?;
        self.infer(
            &image,
            Bbox {
                x: 0.0,
                y: 0.0,
                width: 256.0,
                height: 256.0,
            },
        )?;
        self.mediapipe_center = None;
        self.direct_pose = None;
        self.hand_crop_wrists = None;
        self.image_motion = crate::image_motion::ImageMotion::default();
        self.joint_tracker.reset();
        self.arm_refined.clear();
        Ok(())
    }
    pub fn is_mediapipe(&self) -> bool {
        self.mediapipe
    }
    pub fn set_upper_body(&mut self, upper_body: bool) {
        self.upper_body = upper_body;
    }
    pub fn image_torso_reference_reset(&self) -> bool {
        self.image_motion.torso_reference_reset
    }
    pub fn image_torso_span(&self) -> usize {
        self.image_motion.torso_span
    }
    pub fn image_torso_scale(&self) -> Option<f64> {
        self.image_motion.torso_scale
    }
    pub fn observed_native_scores(&self) -> Option<Vec<f32>> {
        self.direct_pose
            .as_ref()
            .map(|p| p.joints.iter().map(|j| j.input_score).collect())
    }
    pub fn take_pose3d(&mut self) -> Option<crate::motion3d::Pose3d> {
        self.direct_pose.take()
    }
    pub fn hand_crop_wrists(&self) -> Option<[Keypoint; 2]> {
        self.hand_crop_wrists
    }

    pub fn infer(&mut self, image: &Mat, bbox: Bbox) -> Result<Vec<Keypoint>> {
        ensure!(!image.empty(), "Cannot estimate pose in an empty image");
        ensure!(
            bbox.width > 0.0 && bbox.height > 0.0,
            "Crop must have positive dimensions"
        );
        self.direct_pose = None;
        self.hand_crop_wrists = None;
        self.arm_refined.clear();
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
            if self.upper_body && pose.is_none() {
                // Retain upper-body resolution normally; reacquire in a wider
                // virtual full-body crop when a turn defeats the tighter crop.
                for factor in [2.5, 3.0] {
                    let mut wider = roi;
                    wider.height *= factor / crate::mediapipe::UPPER_BODY_CROP_FACTOR;
                    let recovered = crate::mediapipe::infer(
                        &mut self.net,
                        &self.output_names,
                        image,
                        wider,
                        true,
                    )?;
                    if recovered.1.is_some() {
                        (points, pose) = recovered;
                        break;
                    }
                }
            }
            let mut crop_points = points.clone();
            crate::occlusion::mask_arm_overlap(&mut crop_points, image.size()?);
            let corrected = self.image_motion.apply(image, &mut points)?;
            self.hand_crop_wrists = corrected
                .iter()
                .any(|i| matches!(i, 9 | 10))
                .then_some([crop_points[9], crop_points[10]]);
            if let Some(pose) = &mut pose {
                invalidate_flow_arm_depth(pose, &corrected);
            }
            if let Some(refiner) = &mut self.arm_refiner {
                let mut measured = refiner.infer(image, bbox)?;
                crate::occlusion::mask_arm_overlap(&mut measured, image.size()?);
                self.arm_refined =
                    refine_arms(&mut points, &measured, &corrected, &mut pose, image.size()?);
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
        let outputs = self
            .accelerated
            .forward(&mut self.net, &blob, &self.output_names)?;
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

/// Image correspondence measures pixels, not the depth of a different neural
/// landmark. Keep the 2D constraint without calibrating/fitting that stale XYZ.
fn invalidate_flow_arm_depth(pose: &mut crate::motion3d::Pose3d, corrected: &[usize]) {
    for i in corrected {
        let node = match i {
            7 => 12,
            8 => 15,
            9 => 13,
            10 => 16,
            _ => continue,
        };
        if let Some(joint) = pose.joints.get_mut(node) {
            joint.input_score = 0.0;
        }
    }
}

/// Prefer fresh auxiliary image measurements, except where validated image flow
/// already recovered a wrist. Dedicated hand measurements are merged afterwards.
fn refine_arms(
    points: &mut [Keypoint],
    measured: &[Keypoint],
    flow_corrected: &[usize],
    pose: &mut Option<crate::motion3d::Pose3d>,
    size: Size,
) -> Vec<usize> {
    if points.len() < 17 || measured.len() < 17 {
        return Vec::new();
    }
    let width = (points[5].x - points[6].x).hypot(points[5].y - points[6].y);
    let mut changed = Vec::new();
    let mut stale_depth = Vec::new();
    for i in [7, 8, 9, 10] {
        let q = measured[i];
        if q.score < 0.65
            || !q.score.is_finite()
            || !q.x.is_finite()
            || !q.y.is_finite()
            || q.x < 0.0
            || q.y < 0.0
            || q.x >= size.width as f32
            || q.y >= size.height as f32
            || (i >= 9 && flow_corrected.contains(&i))
        {
            continue;
        }
        if (q.x - points[i].x).hypot(q.y - points[i].y) > (width * 0.15).max(20.0) {
            stale_depth.push(i);
        }
        points[i] = q;
        changed.push(i);
    }
    if let Some(pose) = pose {
        invalidate_flow_arm_depth(pose, &stale_depth);
    }
    changed
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
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires LIFTING_LIVE_EVAL, LIFTING_LIVE_SOURCE and native macOS inference; no camera or UI"]
    fn native_pose_matches_opencv_on_visible_recorded_joints() -> Result<()> {
        use opencv::videoio;
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_LIVE_EVAL")?);
        let rows: Vec<serde_json::Value> = std::fs::read_to_string(directory.join("neural.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        let mut capture =
            videoio::VideoCapture::from_file_def(&std::env::var("LIFTING_LIVE_SOURCE")?)?;
        let mut reference = PoseEstimator::new(Path::new("models/rtmpose-m.onnx"))?;
        let mut native = PoseEstimator::new(Path::new("models/rtmpose-m.onnx"))?;
        native.enable_acceleration();
        native.accelerated.force_native();
        ensure!(
            native.accelerated.active(),
            "Native provider did not initialize"
        );
        let mut offsets = Vec::new();
        for frame in [30, 60, 83, 90, 94, 114] {
            capture.set(videoio::CAP_PROP_POS_FRAMES, frame as f64)?;
            let mut image = Mat::default();
            ensure!(capture.read(&mut image)?, "Missing frame");
            let b = &rows[frame]["people"][0]["bbox"];
            let bbox = Bbox {
                x: b["x"].as_f64().unwrap() as f32,
                y: b["y"].as_f64().unwrap() as f32,
                width: b["width"].as_f64().unwrap() as f32,
                height: b["height"].as_f64().unwrap() as f32,
            };
            let a = reference.infer(&image, bbox)?;
            let b = native.infer(&image, bbox)?;
            ensure!(
                native.accelerated.active(),
                "Native inference fell back to OpenCV"
            );
            for (a, b) in a.iter().zip(&b) {
                if a.score >= 0.65 && b.score >= 0.65 {
                    offsets.push((a.x - b.x).hypot(a.y - b.y));
                }
                ensure!(
                    (a.score - b.score).abs() < 0.05,
                    "Native confidence diverged for {}",
                    a.name
                );
            }
        }
        let max = offsets.iter().copied().fold(0.0f32, f32::max);
        let mean = offsets.iter().sum::<f32>() / offsets.len() as f32;
        println!(
            "native vs OpenCV: {} visible joints, mean {mean:.3}px, max {max:.3}px",
            offsets.len()
        );
        ensure!(
            mean < 2.0 && max < 6.0,
            "Native pose diverged from reference"
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires selected local recording and RTMW model; auxiliary pose comparison"]
    fn evaluate_recorded_auxiliary_pose() -> Result<()> {
        use opencv::{imgcodecs, videoio};
        use std::collections::BTreeSet;
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let input: Vec<serde_json::Value> =
            std::fs::read_to_string(directory.join("verifiedarms.jsonl"))?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
        let mut frames: BTreeSet<usize> = (0..input.len()).step_by(15).collect();
        if std::env::var_os("LIFTING_EVAL_ALL").is_some() {
            frames.extend(0..input.len());
        }
        let keys: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(directory.join("keyframes.json"))?)?;
        frames.extend(
            keys.iter()
                .filter_map(|v| v["frame"].as_u64().map(|n| n as usize)),
        );
        let mut estimator = PoseEstimator::new(Path::new("models/rtmw-m.onnx"))?;
        let mut capture =
            videoio::VideoCapture::from_file_def(directory.join("source.mp4").to_str().unwrap())?;
        let mut result = Vec::new();
        for n in frames {
            let p = &input[n]["people"][0];
            let b = &p["bbox"];
            let bbox = Bbox {
                x: b["x"].as_f64().unwrap() as f32,
                y: b["y"].as_f64().unwrap() as f32,
                width: b["width"].as_f64().unwrap() as f32,
                height: b["height"].as_f64().unwrap() as f32,
            };
            capture.set(videoio::CAP_PROP_POS_FRAMES, n as f64)?;
            let mut image = Mat::default();
            ensure!(capture.read(&mut image)?, "Missing frame");
            let mut points = estimator.infer(&image, bbox)?;
            crate::scope::mask_details(&mut points);
            crate::scope::mask_lower(&mut points);
            draw(&mut image, &points, 0.3)?;
            if n % 15 == 0 || keys.iter().any(|v| v["frame"].as_u64() == Some(n as u64)) {
                imgcodecs::imwrite(
                    directory
                        .join(format!("auxrtmw-{n:04}.jpg"))
                        .to_str()
                        .unwrap(),
                    &image,
                    &Vector::new(),
                )?;
            }
            result.push(serde_json::json!({"frame":n,"keypoints":points}));
        }
        std::fs::write(directory.join("auxrtmw.json"), serde_json::to_vec(&result)?)?;
        Ok(())
    }
    #[test]
    fn flow_arm_pixels_do_not_reuse_neural_depth_or_change_other_joints() {
        use crate::motion3d::{Joint3d, Pose3d};
        let mut pose = Pose3d {
            ear_head: true,
            model: "MediaPipe-Pose",
            coordinate_space: "root_relative_camera",
            units: "estimated_meters",
            history_frames: 1,
            angles: Vec::new(),
            joints: (0..17)
                .map(|i| Joint3d {
                    name: crate::motion3d::NAMES[i],
                    x: i as f32,
                    y: 1.0,
                    z: 2.0,
                    input_score: 0.95,
                })
                .collect(),
        };
        invalidate_flow_arm_depth(&mut pose, &[5, 7, 9]);
        for (i, j) in pose.joints.iter().enumerate() {
            assert_eq!((j.x, j.y, j.z), (i as f32, 1.0, 2.0));
            assert_eq!(
                j.input_score,
                if [12, 13].contains(&i) { 0.0 } else { 0.95 }
            );
        }
    }

    #[test]
    fn arm_refinement_preserves_flow_and_ignores_unreliable_or_outside_points() {
        let original = Keypoint {
            name: "joint",
            x: 100.0,
            y: 100.0,
            score: 0.95,
        };
        let mut points = vec![original; 17];
        points[5].x = 60.0;
        points[6].x = 140.0;
        let mut measured = points.clone();
        measured[7].x = 130.0;
        measured[8].score = 0.64;
        measured[9].x = 200.0;
        measured[10].x = f32::NAN;
        let mut pose = None;
        assert_eq!(
            refine_arms(&mut points, &measured, &[9], &mut pose, Size::new(640, 480)),
            vec![7]
        );
        assert_eq!(points[7].x, 130.0);
        for i in [8, 9, 10] {
            assert_eq!(points[i].x, original.x);
        }
        assert_eq!(points[5].x, 60.0);
        assert_eq!(points[6].x, 140.0);
        assert_eq!(points[0].x, original.x);
    }

    #[test]
    fn arm_refinement_drops_conflicting_depth_but_keeps_matching_depth() {
        use crate::motion3d::{Joint3d, Pose3d};
        let original = Keypoint {
            name: "joint",
            x: 100.0,
            y: 100.0,
            score: 0.95,
        };
        let mut points = vec![original; 17];
        points[5].x = 60.0;
        points[6].x = 140.0;
        let mut measured = points.clone();
        measured[7].x += 30.0;
        measured[8].x += 5.0;
        let mut pose = Some(Pose3d {
            ear_head: true,
            model: "MediaPipe-Pose",
            coordinate_space: "root_relative_camera",
            units: "estimated_meters",
            history_frames: 1,
            angles: Vec::new(),
            joints: (0..17)
                .map(|i| Joint3d {
                    name: crate::motion3d::NAMES[i],
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                    input_score: 0.95,
                })
                .collect(),
        });
        refine_arms(&mut points, &measured, &[], &mut pose, Size::new(640, 480));
        let pose = pose.unwrap();
        for (i, joint) in pose.joints.iter().enumerate() {
            assert_eq!(joint.input_score, if i == 12 { 0.0 } else { 0.95 });
        }
    }

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
    /// Diagnostic integration run against an explicitly selected local recording.
    /// Keeps model observations reusable and writes overlays for visual inspection.
    #[test]
    #[ignore = "requires LIFTING_EVAL_DIR containing source.mp4 and baseline.jsonl"]
    fn evaluate_recorded_pose_samples() -> Result<()> {
        use opencv::{imgcodecs, videoio};
        use std::collections::BTreeSet;
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let factor: f32 = std::env::var("LIFTING_EVAL_CROP_FACTOR")
            .unwrap_or("3.0".into())
            .parse()?;
        ensure!(
            factor.is_finite() && (1.5..=4.0).contains(&factor),
            "Invalid evaluation crop factor"
        );
        let tag = std::env::var("LIFTING_EVAL_TAG").unwrap_or("sample".into());
        ensure!(
            tag.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "Invalid evaluation tag"
        );
        let source = std::env::var("LIFTING_EVAL_INPUT").unwrap_or("baseline".into());
        ensure!(
            source
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "Invalid input tag"
        );
        let mirror = std::env::var_os("LIFTING_EVAL_MIRROR").is_some();
        let input: Vec<serde_json::Value> =
            std::fs::read_to_string(directory.join(format!("{source}.jsonl")))?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
        let mut frames: BTreeSet<usize> = (0..input.len()).step_by(15).collect();
        if std::env::var_os("LIFTING_EVAL_ALL").is_some() {
            frames.extend(0..input.len());
        }
        if directory.join("keyframes.json").exists() {
            let keys: Vec<serde_json::Value> =
                serde_json::from_slice(&std::fs::read(directory.join("keyframes.json"))?)?;
            frames.extend(
                keys.iter()
                    .filter_map(|v| v["frame"].as_u64().map(|i| i as usize)),
            );
        }
        let mut estimator = PoseEstimator::new(Path::new("models/mediapipe-pose.onnx"))?;
        let mut capture =
            videoio::VideoCapture::from_file_def(directory.join("source.mp4").to_str().unwrap())?;
        let mut results = Vec::new();
        for index in frames {
            let Some(person) = input[index]["people"].as_array().and_then(|x| x.first()) else {
                continue;
            };
            let b = &person["bbox"];
            let mut bbox = Bbox {
                x: b["x"].as_f64().unwrap() as f32,
                y: b["y"].as_f64().unwrap() as f32,
                width: b["width"].as_f64().unwrap() as f32,
                height: b["height"].as_f64().unwrap() as f32,
            };
            bbox.height *= factor / crate::mediapipe::UPPER_BODY_CROP_FACTOR;
            estimator.mediapipe_center = if index > 0 {
                input[index - 1]["people"]
                    .as_array()
                    .and_then(|x| x.first())
                    .and_then(|p| {
                        let k = &p["keypoints"];
                        (k[5]["score"]
                            .as_f64()
                            .unwrap_or(0.0)
                            .min(k[6]["score"].as_f64().unwrap_or(0.0))
                            >= 0.6)
                            .then(|| {
                                ((k[5]["x"].as_f64().unwrap() + k[6]["x"].as_f64().unwrap()) / 2.0)
                                    as f32
                            })
                    })
            } else {
                None
            };
            capture.set(videoio::CAP_PROP_POS_FRAMES, index as f64)?;
            let mut image = Mat::default();
            ensure!(
                capture.read(&mut image)?,
                "Cannot read source frame {index}"
            );
            // Sparse sample comparison is a CNN/ROI ablation, not a consecutive
            // image-motion sequence. Validate flow separately on all frames.
            estimator.image_motion = crate::image_motion::ImageMotion::default();
            if mirror {
                let mut reflected = Mat::default();
                core::flip(&image, &mut reflected, 1)?;
                image = reflected;
                bbox.x = image.cols() as f32 - bbox.x - bbox.width;
                estimator.mediapipe_center = estimator
                    .mediapipe_center
                    .map(|x| image.cols() as f32 - 1.0 - x);
            }
            let mut points = estimator.infer(&image, bbox)?;
            if mirror {
                let reflected = points.clone();
                for (i, j) in [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
                    .into_iter()
                    .enumerate()
                {
                    points[i] = Keypoint {
                        name: crate::landmarks::NAMES[i],
                        x: image.cols() as f32 - 1.0 - reflected[j].x,
                        ..reflected[j]
                    };
                }
                for (to, from) in [(91, 112), (112, 91)] {
                    for i in 0..21 {
                        points[to + i] = Keypoint {
                            name: crate::landmarks::NAMES[to + i],
                            x: image.cols() as f32 - 1.0 - reflected[from + i].x,
                            ..reflected[from + i]
                        };
                    }
                }
                let mut original = Mat::default();
                core::flip(&image, &mut original, 1)?;
                image = original;
            }
            let pose = estimator.take_pose3d();
            // This ablation compares restored original-image 2D coordinates;
            // mirrored world XYZ has not been converted to that convention.
            let pose = if mirror { None } else { pose };
            let mut overlay = image.clone();
            draw(&mut overlay, &points, 0.3)?;
            imgcodecs::imwrite(
                directory
                    .join(format!("{tag}-{index:04}.jpg"))
                    .to_str()
                    .unwrap(),
                &overlay,
                &Vector::new(),
            )?;
            results.push(serde_json::json!({"frame":index,"keypoints":points,"pose3d":pose}));
        }
        std::fs::write(
            directory.join(format!("{tag}.json")),
            serde_json::to_vec(&results)?,
        )?;
        Ok(())
    }
}
