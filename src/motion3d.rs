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
    pub model: &'static str,
    pub coordinate_space: &'static str,
    pub units: &'static str,
    pub history_frames: usize,
    pub joints: Vec<Joint3d>,
    pub angles: Vec<JointAngle>,
}

#[derive(Default)]
struct History {
    frames: VecDeque<PoseInput>,
    previous_coco: Option<PoseInput>,
    bbox: Option<Bbox>,
    size: Option<Size>,
}

impl History {
    fn clear(&mut self) {
        *self = Self::default();
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
        // Without a reliable torso there is no dependable person reference for lifting.
        if [5, 6, 11, 12]
            .iter()
            .any(|&i| !valid(&points[i]) || points[i].score < threshold)
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
        let mut converted = coco_to_h36m(coco);
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
        let latest = &values[(FRAMES - 1) * JOINTS * 3..];
        let scores = self.history.frames.back().unwrap();
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
        Ok(Some(Pose3d {
            model: "MotionAGFormer-XS-H36M",
            coordinate_space: "root_relative_camera",
            units: "model_normalized",
            history_frames: self.history.frames.len(),
            joints,
            angles,
        }))
    }
}

pub(crate) fn angles3d(joints: &[Joint3d], threshold: f32) -> Vec<JointAngle> {
    ANGLES
        .iter()
        .map(|&(joint, [a, b, c])| {
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
    let projected: Vec<_> = pose
        .joints
        .iter()
        .map(|j| (j.x + 0.5 * j.z, j.y - 0.3 * j.z))
        .collect();
    let minx = projected.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let maxx = projected
        .iter()
        .map(|p| p.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let miny = projected.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let maxy = projected
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
        imgproc::line(image, pixel(a), pixel(b), color, 2, imgproc::LINE_AA, 0)?;
    }
    for i in 0..JOINTS {
        imgproc::circle(image, pixel(i), 3, color, -1, imgproc::LINE_AA, 0)?;
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
        assert!(!h.push(&p, switched, Size::new(640, 480), 0.3));
        assert!(h.frames.is_empty());
        p[5].score = 0.9;
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
}
