//! MotionAGFormer-XS H36M (27 frames). COCO conversion and screen normalization
//! follow https://github.com/TaatiTeam/MotionAGFormer/tree/master/demo/lib.
use crate::{
    angles::JointAngle,
    pose::{Bbox, Keypoint},
};
use anyhow::{Context, Result, ensure};
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, Vector},
    dnn, imgproc,
    prelude::*,
};
use serde::Serialize;
use std::{collections::VecDeque, path::Path};

const FRAMES: usize = 27;
const JOINTS: usize = 17;
type PoseInput = [[f32; 3]; JOINTS];
pub(crate) const NAMES: [&str; JOINTS] = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "nose",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
];
const BONES: [(usize, usize); 16] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
];
const ANGLES: [(&str, [usize; 3]); 8] = [
    ("left_shoulder", [12, 11, 4]),
    ("right_shoulder", [15, 14, 1]),
    ("left_elbow", [11, 12, 13]),
    ("right_elbow", [14, 15, 16]),
    ("left_hip", [11, 4, 5]),
    ("right_hip", [14, 1, 2]),
    ("left_knee", [4, 5, 6]),
    ("right_knee", [1, 2, 3]),
];

#[derive(Debug, Serialize)]
pub struct Joint3d {
    pub name: &'static str,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Confidence of the corresponding 2D input, not a 3D confidence estimate.
    pub input_score: f32,
}

#[derive(Debug, Serialize)]
pub struct Pose3d {
    /// Head slot is the ear midpoint instead of the legacy eye midpoint.
    pub ear_head: bool,
    pub model: &'static str,
    pub coordinate_space: &'static str,
    pub units: &'static str,
    pub history_frames: usize,
    pub joints: Vec<Joint3d>,
    pub angles: Vec<JointAngle>,
}
impl Pose3d {
    pub(crate) fn has_metric_camera_coordinates(&self) -> bool {
        self.coordinate_space == "root_relative_camera" && self.units == "estimated_meters"
    }
}

#[derive(Default)]
struct History {
    upper_body: bool,
    frames: VecDeque<PoseInput>,
    previous_coco: Option<PoseInput>,
    bbox: Option<Bbox>,
    size: Option<Size>,
}

impl History {
    fn clear(&mut self) {
        let upper_body = self.upper_body;
        *self = Self {
            upper_body,
            ..Self::default()
        };
    }

    fn tensor(&self) -> Vec<f32> {
        let mut values = Vec::with_capacity(FRAMES * JOINTS * 3);
        if let Some(first) = self.frames.front() {
            // Left-pad a short history with its earliest pose; never wait for future frames.
            for _ in self.frames.len()..FRAMES {
                values.extend(first.iter().flatten());
            }
            for frame in &self.frames {
                values.extend(frame.iter().flatten());
            }
        }
        values
    }

    fn push(&mut self, points: &[Keypoint], bbox: Bbox, size: Size, threshold: f32) -> bool {
        if points.len() < JOINTS || size.width <= 0 || size.height <= 0 {
            self.clear();
            return false;
        }
        let valid = |p: &Keypoint| {
            p.x.is_finite()
                && p.y.is_finite()
                && p.score.is_finite()
                && p.score > 0.0
                && p.x >= 0.0
                && p.y >= 0.0
                && p.x < size.width as f32
                && p.y < size.height as f32
        };
        let torso = if self.upper_body {
            &[5, 6][..]
        } else {
            &[5, 6, 11, 12][..]
        };
        let reliable = torso
            .iter()
            .filter(|&&i| valid(&points[i]) && points[i].score >= threshold)
            .count();
        let needed = if self.upper_body { 2 } else { 3 };
        if reliable < needed
            || (self.previous_coco.is_none() && torso.iter().any(|&i| !valid(&points[i])))
        {
            self.clear();
            return false;
        }
        if self.size != Some(size)
            || self
                .bbox
                .is_some_and(|previous| overlap(previous, bbox) < 0.15)
        {
            self.clear();
        }
        let pelvis = [
            (points[11].x + points[12].x) / 2.0,
            (points[11].y + points[12].y) / 2.0,
            0.0,
        ];
        let mut coco = [pelvis; JOINTS];
        for (i, p) in points.iter().take(JOINTS).enumerate() {
            if valid(p) {
                coco[i] = [p.x, p.y, p.score.clamp(0.0, 1.0)];
            } else if let Some(previous) = self.previous_coco {
                coco[i] = [previous[i][0], previous[i][1], 0.0];
            }
        }
        if self.upper_body {
            let width = (points[5].x - points[6].x)
                .hypot(points[5].y - points[6].y)
                .max(20.0);
            let cx = (points[5].x + points[6].x) / 2.0;
            let cy = (points[5].y + points[6].y) / 2.0;
            // Neutral low-confidence context required by the full-body lifting network.
            for (i, dx, dy) in [
                (11, 0.3, 1.0),
                (12, -0.3, 1.0),
                (13, 0.3, 2.0),
                (14, -0.3, 2.0),
                (15, 0.3, 3.0),
                (16, -0.3, 3.0),
            ] {
                coco[i] = [cx + dx * width, cy + dy * width, 0.0];
            }
        }
        let mut converted = coco_to_h36m(coco);
        if self.upper_body {
            for p in &mut converted[..8] {
                p[2] = 0.0;
            }
        }
        for p in &mut converted {
            p[0] = p[0] / size.width as f32 * 2.0 - 1.0;
            p[1] = p[1] / size.width as f32 * 2.0 - size.height as f32 / size.width as f32;
        }
        self.frames.push_back(converted);
        if self.frames.len() > FRAMES {
            self.frames.pop_front();
        }
        self.previous_coco = Some(coco);
        self.bbox = Some(bbox);
        self.size = Some(size);
        true
    }
}

fn overlap(a: Bbox, b: Bbox) -> f32 {
    let area = ((a.x + a.width).min(b.x + b.width) - a.x.max(b.x)).max(0.0)
        * ((a.y + a.height).min(b.y + b.height) - a.y.max(b.y)).max(0.0);
    area / (a.width * a.height + b.width * b.height - area).max(f32::EPSILON)
}

/// Exact official demo coordinate mapping, including synthetic torso/head landmarks.
pub(crate) fn coco_to_h36m(c: PoseInput) -> PoseInput {
    let mut h = [[0.0; 3]; JOINTS];
    for (target, source) in [
        (9, 0),
        (11, 5),
        (14, 6),
        (12, 7),
        (15, 8),
        (13, 9),
        (16, 10),
        (4, 11),
        (1, 12),
        (5, 13),
        (2, 14),
        (6, 15),
        (3, 16),
    ] {
        h[target] = c[source];
    }
    let mean = |indices: &[usize], axis: usize| {
        indices.iter().map(|&i| c[i][axis]).sum::<f32>() / indices.len() as f32
    };
    for axis in 0..3 {
        h[0][axis] = mean(&[11, 12], axis);
        h[8][axis] = mean(&[5, 6], axis);
        h[7][axis] = (h[0][axis] + h[8][axis]) / 2.0;
        h[10][axis] = mean(&[1, 2, 3, 4], axis);
    }
    h[10][1] = c[1][1] + c[2][1] - c[0][1];
    for axis in 0..2 {
        h[8][axis] += (c[0][axis] - h[8][axis]) / 3.0;
        h[7][axis] = mean(&[5, 6, 11, 12], axis);
        h[9][axis] -= (h[9][axis] - mean(&[5, 6], axis)) / 4.0;
    }
    h[7][0] += 2.0 * (h[7][0] - (h[0][0] + h[8][0]) / 2.0);
    h[8][1] -= (mean(&[1, 2], 1) - c[0][1]) * 2.0 / 3.0;
    h
}

pub struct MotionLifter {
    net: dnn::Net,
    output_names: Vector<String>,
    history: History,
}

impl MotionLifter {
    pub fn new(path: &Path) -> Result<Self> {
        ensure!(
            path.is_file(),
            "MotionAGFormer model not found at {}. See README.md for export instructions, or use --no-3d.",
            path.display()
        );
        let mut net = opencv::opencv_branch_5! {
            { dnn::read_net_from_onnx_def(path) } else { dnn::read_net_from_onnx(path) }
        }
        .with_context(|| format!("Cannot load MotionAGFormer export {}", path.display()))?;
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        opencv::not_opencv_branch_5! { net.set_preferable_target(dnn::DNN_TARGET_CPU)?; }
        let output_names = net.get_unconnected_out_layers_names()?;
        ensure!(
            output_names.len() == 1,
            "Expected a single MotionAGFormer output"
        );
        Ok(Self {
            net,
            output_names,
            history: History::default(),
        })
    }

    pub fn set_upper_body(&mut self, upper_body: bool) {
        self.history.upper_body = upper_body;
        self.history.clear();
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }

    pub fn infer(
        &mut self,
        points: &[Keypoint],
        bbox: Bbox,
        size: Size,
        threshold: f32,
    ) -> Result<Option<Pose3d>> {
        if !self.history.push(points, bbox, size, threshold) {
            return Ok(None);
        }
        self.infer_history(FRAMES - 1, threshold).map(Some)
    }

    fn infer_history(&mut self, frame_index: usize, threshold: f32) -> Result<Pose3d> {
        ensure!(
            frame_index < FRAMES && !self.history.frames.is_empty(),
            "Invalid temporal output index"
        );
        let input = self.history.tensor();
        let mat = Mat::from_slice(&input)?;
        let blob = mat.reshape_nd(1, &[1, FRAMES as i32, JOINTS as i32, 3])?;
        self.net.set_input_def(&blob)?;
        let mut outputs = Vector::<Mat>::new();
        self.net.forward(&mut outputs, &self.output_names).context(
            "MotionAGFormer inference failed; use OpenCV 5 for the validated graph engine",
        )?;
        ensure!(outputs.len() == 1, "Unexpected MotionAGFormer output count");
        let output = outputs.get(0)?;
        ensure!(
            output.mat_size().as_ref() == [1, FRAMES as i32, JOINTS as i32, 3],
            "Expected MotionAGFormer [1,27,17,3]; got {:?}",
            output.mat_size()
        );
        let values = output.data_typed::<f32>()?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "MotionAGFormer produced non-finite coordinates"
        );
        let latest = &values[frame_index * JOINTS * 3..(frame_index + 1) * JOINTS * 3];
        let padding = FRAMES - self.history.frames.len();
        let scores = &self.history.frames[frame_index.saturating_sub(padding)];
        let joints: Vec<_> = (0..JOINTS)
            .map(|i| Joint3d {
                name: NAMES[i],
                x: latest[i * 3] - latest[0],
                y: latest[i * 3 + 1] - latest[1],
                z: latest[i * 3 + 2] - latest[2],
                input_score: scores[i][2],
            })
            .collect();
        let angles = angles3d(&joints, threshold);
        Ok(Pose3d {
            ear_head: false,
            model: "MotionAGFormer-XS-H36M",
            coordinate_space: "root_relative_camera",
            units: "model_normalized",
            history_frames: self.history.frames.len(),
            joints,
            angles,
        })
    }
}

pub(crate) fn angles3d(joints: &[Joint3d], threshold: f32) -> Vec<JointAngle> {
    let upper = joints[..8].iter().all(|p| p.input_score == 0.0);
    ANGLES
        .iter()
        .filter(|(name, _)| !upper || !(name.contains("hip") || name.contains("knee")))
        .map(|&(joint, [a, b, mut c])| {
            if upper && joint.contains("shoulder") {
                c = 8;
            }
            let (a, b, c) = (&joints[a], &joints[b], &joints[c]);
            let u = [
                f64::from(a.x - b.x),
                f64::from(a.y - b.y),
                f64::from(a.z - b.z),
            ];
            let v = [
                f64::from(c.x - b.x),
                f64::from(c.y - b.y),
                f64::from(c.z - b.z),
            ];
            let nu = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let degrees = if [a, b, c]
                .iter()
                .all(|p| p.input_score > 0.0 && p.input_score >= threshold)
                && nu > 1e-6
                && nv > 1e-6
            {
                Some(
                    (u.iter().zip(v).map(|(x, y)| x * y).sum::<f64>() / (nu * nv))
                        .clamp(-1.0, 1.0)
                        .acos()
                        .to_degrees(),
                )
            } else {
                None
            };
            JointAngle { joint, degrees }
        })
        .collect()
}

pub fn draw(image: &mut Mat, pose: &Pose3d) -> Result<()> {
    // Oblique projection for display only; exported XYZ remain camera-relative.
    let extent = Size::new(220.min(image.cols() - 8), 220.min(image.rows() - 8));
    if extent.width < 120 || extent.height < 120 {
        return Ok(());
    }
    let origin = Point::new(image.cols() - extent.width - 4, 4);
    imgproc::rectangle(
        image,
        Rect::new(origin.x, origin.y, extent.width, extent.height),
        Scalar::new(25.0, 25.0, 25.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    let upper = pose.joints[..8].iter().all(|p| p.input_score == 0.0);
    let projected: Vec<_> = pose
        .joints
        .iter()
        .map(|j| (j.x + 0.5 * j.z, j.y - 0.3 * j.z))
        .collect();
    let visible: Vec<_> = projected.iter().copied().collect();
    let minx = visible.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let maxx = visible
        .iter()
        .map(|p| p.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let miny = visible.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let maxy = visible
        .iter()
        .map(|p| p.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let scale = ((extent.width - 30) as f32 / (maxx - minx).max(1e-4))
        .min((extent.height - 50) as f32 / (maxy - miny).max(1e-4));
    let pixel = |i: usize| {
        Point::new(
            origin.x + extent.width / 2 + ((projected[i].0 - (minx + maxx) / 2.0) * scale) as i32,
            origin.y
                + 30
                + (extent.height - 30) / 2
                + ((projected[i].1 - (miny + maxy) / 2.0) * scale) as i32,
        )
    };
    let color = Scalar::new(255.0, 210.0, 70.0, 0.0);
    for (a, b) in BONES {
        let unobserved = upper && (!crate::scope::body_upper(a) || !crate::scope::body_upper(b));
        imgproc::line(
            image,
            pixel(a),
            pixel(b),
            if unobserved {
                Scalar::all(100.0)
            } else {
                color
            },
            2,
            imgproc::LINE_AA,
            0,
        )?;
    }
    for i in 0..JOINTS {
        let unobserved = upper && !crate::scope::body_upper(i);
        imgproc::circle(
            image,
            pixel(i),
            3,
            if unobserved {
                Scalar::all(100.0)
            } else {
                color
            },
            if unobserved { 1 } else { -1 },
            imgproc::LINE_AA,
            0,
        )?;
    }
    imgproc::put_text(
        image,
        &if pose.model == "skeletal_twin_fk" {
            "Twin (fixed bone fit)".to_owned()
        } else {
            format!("3D body {}/27 (inferred)", pose.history_frames)
        },
        Point::new(origin.x + 7, origin.y + 18),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.4,
        Scalar::all(240.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires local ONNX model and reference tensor files"]
    fn temporal_onnx_reference_preserves_center_and_latest_alignment() -> Result<()> {
        let read = |path: &str| -> Result<Vec<f32>> {
            let bytes = std::fs::read(path)?;
            ensure!(bytes.len() % 4 == 0, "Invalid tensor bytes");
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        };
        let input = read("models/motionagformer-xs.input.f32")?;
        let expected = read("models/motionagformer-xs.expected.f32")?;
        ensure!(
            input.len() == FRAMES * JOINTS * 3 && expected.len() == input.len(),
            "Wrong tensor shape"
        );
        let mut lifter = MotionLifter::new(Path::new("models/motionagformer-xs.onnx"))?;
        for frame in input.chunks_exact(JOINTS * 3) {
            lifter.history.frames.push_back(std::array::from_fn(|i| {
                [frame[i * 3], frame[i * 3 + 1], frame[i * 3 + 2]]
            }));
        }
        for index in [FRAMES / 2, FRAMES - 1] {
            let pose = lifter.infer_history(index, 0.3)?;
            let source = &expected[index * JOINTS * 3..(index + 1) * JOINTS * 3];
            for (i, joint) in pose.joints.iter().enumerate() {
                for (axis, value) in [joint.x, joint.y, joint.z].into_iter().enumerate() {
                    assert!(
                        (value - (source[i * 3 + axis] - source[axis])).abs() < 1e-4,
                        "Frame {index} joint {i} axis {axis}"
                    );
                }
                assert_eq!(joint.input_score, input[index * JOINTS * 3 + i * 3 + 2]);
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires local recording/model; cached-landmark temporal lifting comparison"]
    fn evaluate_recorded_temporal_lifting() -> Result<()> {
        use std::io::Write;
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let original: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("twin.json"))?)?;
        let meta = &original["recording"];
        let times: Vec<f64> = serde_json::from_value(meta["timestamps_seconds"].clone())?;
        let mut lifter = MotionLifter::new(Path::new("models/motionagformer-xs.onnx"))?;
        lifter.set_upper_body(true);
        let mut recorder = crate::twin::TwinRecorder::default();
        recorder.set_upright_torso(true);
        let centered = std::env::var_os("LIFTING_EVAL_CENTERED").is_some();
        let calibrated = std::env::var_os("LIFTING_EVAL_CONTEXT").is_some();
        let metric = std::env::var_os("LIFTING_EVAL_METRIC").is_some();
        ensure!(
            !metric || calibrated,
            "Metric-prior comparison requires calibrated context"
        );
        ensure!(
            !calibrated || centered,
            "Calibrated-context diagnostic requires centered windows"
        );
        let tag = if metric {
            "motionmetric"
        } else if calibrated {
            "motionheight"
        } else if centered {
            "motioncenter"
        } else {
            "motion27"
        };
        let native: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("verifiedarms-twin.json"))?)?;
        let body_width = native["subjects"][0]["person"]["shoulder_width"]
            .as_f64()
            .unwrap();
        let input: Vec<serde_json::Value> =
            std::fs::read_to_string(directory.join("verifiedarms.jsonl"))?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
        let get = |person: &serde_json::Value| {
            let points: Vec<_> = person["keypoints"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, k)| Keypoint {
                    name: crate::landmarks::NAMES[i],
                    x: k["x"].as_f64().unwrap() as f32,
                    y: k["y"].as_f64().unwrap() as f32,
                    score: k["score"].as_f64().unwrap() as f32,
                })
                .collect();
            let b = &person["bbox"];
            let bbox = Bbox {
                x: b["x"].as_f64().unwrap() as f32,
                y: b["y"].as_f64().unwrap() as f32,
                width: b["width"].as_f64().unwrap() as f32,
                height: b["height"].as_f64().unwrap() as f32,
            };
            (points, bbox)
        };
        let mut output = std::io::BufWriter::new(std::fs::File::create(
            directory.join(format!("{tag}.jsonl")),
        )?);
        for (index, source) in input.iter().enumerate() {
            let mut row = source.clone();
            let n = row["frame"].as_u64().unwrap();
            let width = row["width"].as_i64().unwrap() as i32;
            let height = row["height"].as_i64().unwrap() as i32;
            for person in row["people"].as_array_mut().unwrap() {
                let (points, bbox) = get(person);
                let mut pose = if centered {
                    lifter.history.clear();
                    for offset in 0..FRAMES {
                        let sample = (index as isize + offset as isize - (FRAMES / 2) as isize)
                            .clamp(0, input.len() as isize - 1)
                            as usize;
                        let (context, roi) = get(&input[sample]["people"][0]);
                        lifter
                            .history
                            .push(&context, roi, Size::new(width, height), 0.3);
                        if calibrated {
                            // Diagnostic intervention only: preserve foreshortened
                            // horizontal context, but infer vertical extent from
                            // shared body width and the existing distance fit.
                            let frame = &native["subjects"][0]["frames"][sample];
                            ensure!(
                                frame["frame"].as_u64() == Some(sample as u64),
                                "Mismatched context timing"
                            );
                            let vertical = (frame["focal_length_px"].as_f64().unwrap() * body_width
                                / frame["root_position"][2].as_f64().unwrap())
                                as f32;
                            let horizontal = (context[5].x - context[6].x)
                                .hypot(context[5].y - context[6].y)
                                .max(20.0);
                            let cx = (context[5].x + context[6].x) / 2.0;
                            let cy = (context[5].y + context[6].y) / 2.0;
                            let mut coco = lifter.history.previous_coco.unwrap();
                            for (i, dx, dy) in [
                                (11, 0.3, 1.0),
                                (12, -0.3, 1.0),
                                (13, 0.3, 2.0),
                                (14, -0.3, 2.0),
                                (15, 0.3, 3.0),
                                (16, -0.3, 3.0),
                            ] {
                                coco[i] = [cx + dx * horizontal, cy + dy * vertical, 0.0];
                            }
                            let mut converted = coco_to_h36m(coco);
                            for p in &mut converted[..8] {
                                p[2] = 0.0;
                            }
                            for p in &mut converted {
                                p[0] = p[0] / width as f32 * 2.0 - 1.0;
                                p[1] = p[1] / width as f32 * 2.0 - height as f32 / width as f32;
                            }
                            *lifter.history.frames.back_mut().unwrap() = converted;
                        }
                    }
                    ensure!(
                        lifter.history.frames.len() == FRAMES,
                        "Incomplete centered context"
                    );
                    Some(lifter.infer_history(FRAMES / 2, 0.3)?)
                } else {
                    lifter.infer(&points, bbox, Size::new(width, height), 0.3)?
                };
                if metric {
                    if let Some(pose) = &mut pose {
                        let native = &person["pose3d"]["joints"];
                        let a = &pose.joints[11];
                        let b = &pose.joints[14];
                        let center = [(a.x + b.x) / 2.0, (a.y + b.y) / 2.0, (a.z + b.z) / 2.0];
                        let scale = body_width as f32
                            / (a.x - b.x).hypot(a.y - b.y).hypot(a.z - b.z).max(1e-6);
                        let anchor = [
                            native[8]["x"].as_f64().unwrap() as f32,
                            native[8]["y"].as_f64().unwrap() as f32,
                            native[8]["z"].as_f64().unwrap() as f32,
                        ];
                        for joint in &mut pose.joints {
                            joint.x = (joint.x - center[0]) * scale + anchor[0];
                            joint.y = (joint.y - center[1]) * scale + anchor[1];
                            joint.z = (joint.z - center[2]) * scale + anchor[2];
                        }
                        // Keep the native physical shoulder-center/head anchors;
                        // model H36M thorax/neck slots have different semantics.
                        for i in [8, 9, 10] {
                            pose.joints[i].x = native[i]["x"].as_f64().unwrap() as f32;
                            pose.joints[i].y = native[i]["y"].as_f64().unwrap() as f32;
                            pose.joints[i].z = native[i]["z"].as_f64().unwrap() as f32;
                            pose.joints[i].input_score =
                                native[i]["input_score"].as_f64().unwrap() as f32;
                        }
                        pose.ear_head = true;
                        pose.model = "MotionAGFormer-NativeScale-Diagnostic";
                        pose.units = "estimated_meters";
                    }
                }
                if let Some(pose) = &pose {
                    let hands: Vec<crate::details::HandPose> =
                        serde_json::from_value(person["hands"].clone())?;
                    let head = serde_json::from_value(person["head"].clone())?;
                    recorder.observe(
                        n,
                        times[n as usize],
                        pose,
                        &points,
                        &hands,
                        head,
                        bbox,
                        Size::new(width, height),
                        0.3,
                        None,
                    );
                }
                person["pose3d"] = serde_json::to_value(pose)?;
            }
            serde_json::to_writer(&mut output, &row)?;
            output.write_all(b"\n")?;
        }
        output.flush()?;
        recorder.set_timing(crate::twin::RecordingTiming {
            fps: meta["fps"].as_f64().unwrap(),
            frame_count: times.len() as u64,
            timestamps_seconds: times,
            source_video: Some("source.mp4".into()),
            trim_start_seconds: 3.0,
            trim_end_seconds: 3.0,
            upper_body: true,
            assume_upright_torso: true,
        });
        recorder.save(&directory.join(format!("{tag}-twin.json")))?;
        Ok(())
    }
    fn points() -> Vec<Keypoint> {
        (0..17)
            .map(|i| Keypoint {
                name: "test",
                x: 100.0 + i as f32 * 3.0,
                y: 80.0 + i as f32 * 4.0,
                score: 0.9,
            })
            .collect()
    }
    fn bbox() -> Bbox {
        Bbox {
            x: 50.0,
            y: 20.0,
            width: 200.0,
            height: 400.0,
        }
    }

    #[test]
    fn mapping_normalization_and_padding_match_official_layout() {
        let points = points();
        let mut history = History::default();
        assert!(history.push(&points, bbox(), Size::new(640, 480), 0.3));
        let input = history.tensor();
        assert_eq!(input.len(), 27 * 17 * 3);
        assert_eq!(&input[..51], &input[26 * 51..]);
        let right_hip = history.frames.back().unwrap()[1];
        assert!((right_hip[0] - (points[12].x / 640.0 * 2.0 - 1.0)).abs() < 1e-6);
        assert!((right_hip[1] - (points[12].y / 640.0 * 2.0 - 0.75)).abs() < 1e-6);
        let pelvis = history.frames.back().unwrap()[0];
        assert!(
            (pelvis[0] - (((points[11].x + points[12].x) / 2.0) / 640.0 * 2.0 - 1.0)).abs() < 1e-6
        );
    }

    #[test]
    fn temporal_history_is_bounded_and_resets_on_switch_missing_torso_or_resize() {
        let mut h = History::default();
        let mut p = points();
        for _ in 0..35 {
            assert!(h.push(&p, bbox(), Size::new(640, 480), 0.3));
        }
        assert_eq!(h.frames.len(), 27);
        let mut switched = bbox();
        switched.x = 400.0;
        assert!(h.push(&p, switched, Size::new(640, 480), 0.3));
        assert_eq!(h.frames.len(), 1);
        p[5].score = 0.1;
        p[6].score = 0.1;
        assert!(!h.push(&p, switched, Size::new(640, 480), 0.3));
        assert!(h.frames.is_empty());
        p[5].score = 0.9;
        p[6].score = 0.9;
        h.push(&p, bbox(), Size::new(640, 480), 0.3);
        h.push(&p, bbox(), Size::new(1280, 720), 0.3);
        assert_eq!(h.frames.len(), 1);
    }

    #[test]
    fn three_dimensional_knee_angle_uses_depth_and_confidence() {
        let mut joints: Vec<_> = NAMES
            .iter()
            .map(|&name| Joint3d {
                name,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                input_score: 0.9,
            })
            .collect();
        joints[4].z = 1.0;
        joints[6].x = 1.0;
        let knee = angles3d(&joints, 0.3)
            .into_iter()
            .find(|a| a.joint == "left_knee")
            .unwrap();
        assert_eq!(knee.degrees, Some(90.0));
        joints[6].input_score = 0.1;
        assert!(angles3d(&joints, 0.3)[6].degrees.is_none());
    }
    #[test]
    fn one_uncertain_torso_landmark_preserves_temporal_pose_without_raising_confidence() {
        let mut history = History::default();
        let mut p = points();
        assert!(history.push(&p, bbox(), Size::new(640, 480), 0.3));
        p[5].score = 0.1;
        assert!(history.push(&p, bbox(), Size::new(640, 480), 0.3));
        assert_eq!(history.frames.len(), 2);
        assert_eq!(history.frames.back().unwrap()[11][2], 0.1);
        p[6].score = 0.1;
        assert!(!history.push(&p, bbox(), Size::new(640, 480), 0.3));
    }
    #[test]
    fn upper_body_history_does_not_require_or_trust_hips_and_legs() {
        let mut history = History {
            upper_body: true,
            ..History::default()
        };
        let mut p = points();
        crate::scope::mask_lower(&mut p);
        assert!(history.push(&p, bbox(), Size::new(640, 480), 0.3));
        assert!(
            history.frames.back().unwrap()[..8]
                .iter()
                .all(|p| p[2] == 0.0)
        );
        assert!(history.frames.back().unwrap()[11][2] > 0.0);
        history.clear();
        assert!(history.upper_body);
        assert!(history.push(&p, bbox(), Size::new(640, 480), 0.3));
        p[5].score = 0.1;
        assert!(!history.push(&p, bbox(), Size::new(640, 480), 0.3));
    }
}
