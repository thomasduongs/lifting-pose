//! Shared body dimensions plus per-frame rotations. Online calibration is
//! provisional; final export re-solves every frame using one final shape per subject.
use crate::{
    motion3d::{self, Joint3d, Pose3d},
    pose::{Bbox, Keypoint},
};
use anyhow::{Context, Result, ensure};
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use opencv::core::Size;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

type V = Vector3<f64>;
type Q = UnitQuaternion<f64>;
const N: usize = 17;
const PARENT: [usize; N] = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15];
const DIMENSIONS: [&str; 10] = [
    "pelvis_width",
    "shoulder_width",
    "upper_leg",
    "lower_leg",
    "upper_arm",
    "forearm",
    "pelvis_to_spine",
    "spine_to_thorax",
    "thorax_to_nose",
    "nose_to_head",
];

#[derive(Clone, Copy)]
struct Dimensions([f64; 10]);
impl Dimensions {
    fn offsets(self) -> [V; N] {
        let d = self.0;
        [
            V::zeros(),
            V::new(-d[0] / 2.0, 0.0, 0.0),
            V::new(0.0, d[2], 0.0),
            V::new(0.0, d[3], 0.0),
            V::new(d[0] / 2.0, 0.0, 0.0),
            V::new(0.0, d[2], 0.0),
            V::new(0.0, d[3], 0.0),
            V::new(0.0, -d[6], 0.0),
            V::new(0.0, -d[7], 0.0),
            V::new(0.0, -d[8], 0.0),
            V::new(0.0, -d[9], 0.0),
            V::new(d[1] / 2.0, 0.0, 0.0),
            V::new(d[4], 0.0, 0.0),
            V::new(d[5], 0.0, 0.0),
            V::new(-d[1] / 2.0, 0.0, 0.0),
            V::new(-d[4], 0.0, 0.0),
            V::new(-d[5], 0.0, 0.0),
        ]
    }
}

#[derive(Clone)]
struct Pose {
    root: V,
    orientation: Q,
    local: [Q; N],
}
impl Default for Pose {
    fn default() -> Self {
        Self {
            root: V::zeros(),
            orientation: Q::identity(),
            local: [Q::identity(); N],
        }
    }
}

/// Rotations belong to joints and rotate their outgoing bones, not incoming offsets.
fn forward(dimensions: Dimensions, pose: &Pose) -> ([V; N], [Q; N]) {
    let offsets = dimensions.offsets();
    let mut positions = [V::zeros(); N];
    let mut rotations = [Q::identity(); N];
    positions[0] = pose.root;
    rotations[0] = unit(pose.orientation);
    for i in 1..N {
        let parent = PARENT[i];
        positions[i] = positions[parent] + rotations[parent] * offsets[i];
        rotations[i] = unit(rotations[parent] * pose.local[i]);
    }
    (positions, rotations)
}

fn measurements(p: &[V; N], scores: &[f64; N], minimum: f64) -> [Vec<(f64, f64)>; 10] {
    let pairs: [&[(usize, usize)]; 10] = [
        &[(1, 4)],
        &[(11, 14)],
        &[(1, 2), (4, 5)],
        &[(2, 3), (5, 6)],
        &[(11, 12), (14, 15)],
        &[(12, 13), (15, 16)],
        &[(0, 7)],
        &[(7, 8)],
        &[(8, 9)],
        &[(9, 10)],
    ];
    std::array::from_fn(|i| {
        pairs[i]
            .iter()
            .filter_map(|&(a, b)| {
                let weight = scores[a].min(scores[b]);
                let length = (p[b] - p[a]).norm();
                (weight >= minimum && length.is_finite() && length > 1e-5)
                    .then_some((length, weight * weight))
            })
            .collect()
    })
}

struct Estimate {
    value: f64,
    weight: f64,
    samples: u64,
    bootstrap: Vec<(f64, f64)>,
}
impl Estimate {
    fn new(value: f64) -> Self {
        Self {
            value,
            weight: 0.0,
            samples: 0,
            bootstrap: Vec::new(),
        }
    }
    fn add(&mut self, value: f64, weight: f64) {
        if self.samples < 31 {
            self.bootstrap.push((value, weight));
            let mut sorted = self.bootstrap.clone();
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
            self.weight = sorted.iter().map(|p| p.1).sum();
            let mut accumulated = 0.0;
            for &(v, w) in &sorted {
                accumulated += w;
                if accumulated >= self.weight / 2.0 {
                    self.value = v;
                    break;
                }
            }
        } else {
            // Bounded-influence running estimate: outliers cannot resize a limb abruptly.
            let residual = (value - self.value).clamp(-self.value * 0.15, self.value * 0.15);
            self.weight += weight;
            self.value += weight / self.weight * residual;
        }
        self.samples += 1;
    }
}

struct Calibration {
    estimates: [Estimate; 10],
    revision: u64,
    frames: u64,
}
impl Calibration {
    fn new(obs: &Observation) -> Self {
        let raw = measurements(&obs.points, &[1.0; N], 0.0);
        let scale = (obs.points[8] - obs.points[0]).norm().max(0.05);
        let fallback = [
            scale * 0.6,
            scale * 0.9,
            scale,
            scale,
            scale * 0.7,
            scale * 0.6,
            scale / 2.0,
            scale / 2.0,
            scale * 0.3,
            scale * 0.2,
        ];
        let estimates = std::array::from_fn(|i| {
            let value = if raw[i].is_empty() {
                fallback[i]
            } else {
                raw[i].iter().map(|p| p.0).sum::<f64>() / raw[i].len() as f64
            };
            Estimate::new(value.max(1e-4))
        });
        Self {
            estimates,
            revision: 0,
            frames: 0,
        }
    }
    fn update(&mut self, obs: &Observation, minimum: f64) {
        // MotionAGFormer startup repeats poses. Do not count those as calibration evidence.
        if obs.history_frames < 27 {
            return;
        }
        let data = measurements(&obs.points, &obs.scores, minimum);
        let before = self.dimensions().0;
        let mut accepted = false;
        for (estimate, samples) in self.estimates.iter_mut().zip(data) {
            for (value, weight) in samples {
                estimate.add(value, weight);
                accepted = true;
            }
        }
        if accepted {
            self.frames += 1;
        }
        if self
            .dimensions()
            .0
            .iter()
            .zip(before)
            .any(|(a, b)| (a - b).abs() > 1e-9)
        {
            self.revision += 1;
        }
    }
    fn dimensions(&self) -> Dimensions {
        Dimensions(std::array::from_fn(|i| self.estimates[i].value))
    }
    fn parameters(&self) -> PersonParameters {
        let dimensions = self.dimensions();
        PersonParameters {
            units: "motionagformer_model_units",
            revision: self.revision,
            pelvis_width: dimensions.0[0],
            shoulder_width: dimensions.0[1],
            segment_lengths: (2..10).map(|i| (DIMENSIONS[i], dimensions.0[i])).collect(),
            rest_offsets: dimensions.offsets().map(array),
            bilateral_symmetry: true,
            calibration_frames: self.frames,
            measurement_counts: (0..10)
                .map(|i| (DIMENSIONS[i], self.estimates[i].samples))
                .collect(),
        }
    }
}

#[derive(Serialize)]
pub struct PersonParameters {
    units: &'static str,
    revision: u64,
    pelvis_width: f64,
    shoulder_width: f64,
    segment_lengths: BTreeMap<&'static str, f64>,
    /// These offsets are shared by all frames of the finalized subject.
    rest_offsets: [[f64; 3]; N],
    bilateral_symmetry: bool,
    calibration_frames: u64,
    measurement_counts: BTreeMap<&'static str, u64>,
}

#[derive(Clone)]
struct Observation {
    frame: u64,
    points: [V; N],
    scores: [f64; N],
    pixels: [[f64; 2]; N],
    pixel_scores: [f64; N],
    size: Size,
    focal: f64,
    assumed_focal: bool,
    history_frames: usize,
}

fn descendant(mut node: usize, ancestor: usize) -> bool {
    while node != 0 {
        node = PARENT[node];
        if node == ancestor {
            return true;
        }
    }
    false
}
fn robust_weight(residual: f64, delta: f64) -> f64 {
    if residual <= delta {
        1.0
    } else {
        delta / residual
    }
}
fn cost(positions: &[V; N], obs: &Observation, delta: f64) -> f64 {
    (1..N)
        .map(|i| {
            let r = (positions[i] - obs.points[i]).norm();
            let loss = if r <= delta {
                r * r / 2.0
            } else {
                delta * (r - delta / 2.0)
            };
            obs.scores[i].powi(2) * loss
        })
        .sum()
}
fn array(v: V) -> [f64; 3] {
    [v.x, v.y, v.z]
}
fn unit(q: Q) -> Q {
    Q::new_normalize(*q.quaternion())
}
fn quaternion(q: Q) -> [f64; 4] {
    let q = unit(q);
    let q = q.quaternion();
    [q.i, q.j, q.k, q.w]
}

fn root_orientation(obs: &Observation) -> Q {
    let x = obs.points[4] - obs.points[1];
    let down = obs.points[0] - obs.points[8];
    if x.norm() < 1e-6 || down.norm() < 1e-6 {
        return Q::identity();
    }
    let x = x.normalize();
    let z = x.cross(&down);
    if z.norm() < 1e-6 {
        return Q::identity();
    }
    let z = z.normalize();
    let y = z.cross(&x).normalize();
    Q::from_rotation_matrix(&Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[
        x, y, z,
    ])))
}

fn initial_pose(dimensions: Dimensions, obs: &Observation) -> Pose {
    let mut pose = Pose {
        orientation: root_orientation(obs),
        ..Pose::default()
    };
    for node in 1..N {
        if let Some(child) = ((node + 1)..N).find(|&i| PARENT[i] == node && obs.scores[i] > 0.0) {
            let (positions, world) = forward(dimensions, &pose);
            let from = positions[child] - positions[node];
            let to = obs.points[child] - obs.points[node];
            if let Some(delta) = Q::rotation_between(&from, &to) {
                pose.local[node] = unit(world[PARENT[node]].inverse() * delta * world[node]);
            }
        }
    }
    pose
}

/// Block coordinate damped least-squares IK with analytic rotational Jacobians.
/// Each step changes rotations only: dimensions remain fixed for the entire solve.
fn inverse(dimensions: Dimensions, obs: &Observation, previous: Option<&Pose>) -> Pose {
    let initialized = initial_pose(dimensions, obs);
    let delta = dimensions.0[2] * 0.2;
    let mut pose = match previous {
        Some(previous)
            if cost(&forward(dimensions, previous).0, obs, delta)
                < cost(&forward(dimensions, &initialized).0, obs, delta) =>
        {
            previous.clone()
        }
        _ => initialized,
    };
    pose.root = V::zeros();
    let damping = dimensions.0[2].powi(2) * 0.002;
    for _ in 0..8 {
        let before = cost(&forward(dimensions, &pose).0, obs, delta);
        for node in (0..N).rev() {
            if !(1..N).any(|i| descendant(i, node)) {
                continue;
            }
            let (positions, world) = forward(dimensions, &pose);
            let mut a = Matrix3::identity() * damping;
            let mut b = V::zeros();
            for i in 1..N {
                if !descendant(i, node) || obs.scores[i] <= 0.0 {
                    continue;
                }
                let arm = positions[i] - positions[node];
                let residual = obs.points[i] - positions[i];
                let weight = obs.scores[i].powi(2) * robust_weight(residual.norm(), delta);
                // delta_position = delta_rotation × arm.
                let jacobian =
                    Matrix3::new(0.0, arm.z, -arm.y, -arm.z, 0.0, arm.x, arm.y, -arm.x, 0.0);
                a += weight * jacobian.transpose() * jacobian;
                b += weight * jacobian.transpose() * residual;
            }
            let Some(mut step) = a.lu().solve(&b) else {
                continue;
            };
            if step.norm() > 0.3 {
                step *= 0.3 / step.norm();
            }
            if !step.iter().all(|v| v.is_finite()) {
                continue;
            }
            let old_cost = cost(&positions, obs, delta);
            for factor in [1.0, 0.5, 0.25] {
                let mut candidate = pose.clone();
                let rotation = Q::from_scaled_axis(step * factor);
                if node == 0 {
                    candidate.orientation = unit(rotation * pose.orientation);
                } else {
                    candidate.local[node] =
                        unit(world[PARENT[node]].inverse() * rotation * world[node]);
                }
                if cost(&forward(dimensions, &candidate).0, obs, delta) < old_cost {
                    pose = candidate;
                    break;
                }
            }
        }
        let after = cost(&forward(dimensions, &pose).0, obs, delta);
        if before - after < 1e-9 {
            break;
        }
    }
    pose
}

#[derive(Serialize)]
pub struct JointRotation {
    joint: &'static str,
    quaternion_xyzw: [f64; 4],
}
#[derive(Serialize)]
pub struct FrameParameters {
    frame: u64,
    root_position: [f64; 3],
    root_orientation_xyzw: [f64; 4],
    joint_rotations: Vec<JointRotation>,
    root_translation_valid: bool,
    root_translation_model: &'static str,
    focal_length_px: f64,
    focal_length_assumed: bool,
    fit_rmse: f64,
    angles: Vec<crate::angles::JointAngle>,
}

/// Weak-perspective translation: model scale/depth is a gauge, not metric camera tracking.
fn camera_translation(obs: &Observation, positions: &[V; N]) -> Option<V> {
    let mut sw = 0.0;
    let mut model = V::zeros();
    let mut image = V::zeros();
    for (i, p) in positions.iter().enumerate() {
        let w = obs.scores[i].min(obs.pixel_scores[i]).powi(2);
        sw += w;
        model += p * w;
        image += V::new(obs.pixels[i][0], obs.pixels[i][1], 0.0) * w;
    }
    if sw < 1e-6 {
        return None;
    }
    model /= sw;
    image /= sw;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (i, p) in positions.iter().enumerate() {
        let w = obs.scores[i].min(obs.pixel_scores[i]).powi(2);
        let p = p - model;
        numerator += w * (p.x * (obs.pixels[i][0] - image.x) + p.y * (obs.pixels[i][1] - image.y));
        denominator += w * (p.x * p.x + p.y * p.y);
    }
    if denominator < 1e-9 {
        return None;
    }
    let scale = numerator / denominator;
    if !scale.is_finite() || scale <= 1e-6 {
        return None;
    }
    let tx = image.x - scale * model.x;
    let ty = image.y - scale * model.y;
    let root = V::new(
        (tx - obs.size.width as f64 / 2.0) / scale,
        (ty - obs.size.height as f64 / 2.0) / scale,
        obs.focal / scale,
    );
    root.iter().all(|v| v.is_finite()).then_some(root)
}
fn frame_parameters(dimensions: Dimensions, obs: &Observation, pose: &Pose) -> FrameParameters {
    let positions = forward(dimensions, pose).0;
    let fitted_joints: Vec<_> = (0..N)
        .map(|i| Joint3d {
            name: motion3d::NAMES[i],
            x: positions[i].x as f32,
            y: positions[i].y as f32,
            z: positions[i].z as f32,
            input_score: obs.scores[i] as f32,
        })
        .collect();
    let sum: f64 = obs.scores.iter().map(|s| s * s).sum();
    let fit_rmse = ((0..N)
        .map(|i| obs.scores[i].powi(2) * (positions[i] - obs.points[i]).norm_squared())
        .sum::<f64>()
        / sum.max(1e-9))
    .sqrt();
    let translation = camera_translation(obs, &positions);
    FrameParameters {
        frame: obs.frame,
        root_position: array(translation.unwrap_or(V::zeros())),
        root_orientation_xyzw: quaternion(pose.orientation),
        joint_rotations: (0..N)
            .map(|i| JointRotation {
                joint: motion3d::NAMES[i],
                quaternion_xyzw: quaternion(pose.local[i]),
            })
            .collect(),
        root_translation_valid: translation.is_some(),
        root_translation_model: "weak_perspective_approximation",
        focal_length_px: obs.focal,
        focal_length_assumed: obs.assumed_focal,
        fit_rmse,
        angles: motion3d::angles3d(&fitted_joints, 0.0),
    }
}

#[derive(Serialize)]
pub struct TwinPreview {
    subject_segment: usize,
    provisional: bool,
    person: PersonParameters,
    frame: FrameParameters,
    #[serde(skip)]
    pub fitted_pose: Pose3d,
}

impl TwinPreview {
    /// A learned shape is a tracking prior only after it has real calibration support.
    pub fn tracking_projection(&self, size: Size) -> Option<[[f32; 2]; N]> {
        if self.person.calibration_frames < 5 || !self.frame.root_translation_valid {
            return None;
        }
        let leg = self.person.segment_lengths["upper_leg"];
        if self.frame.fit_rmse > leg * 0.3 {
            return None;
        }
        let root = self.frame.root_position;
        let scale = self.frame.focal_length_px / root[2];
        if !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        let projection = std::array::from_fn(|i| {
            let p = &self.fitted_pose.joints[i];
            [
                ((f64::from(p.x) + root[0]) * scale + f64::from(size.width) / 2.0) as f32,
                ((f64::from(p.y) + root[1]) * scale + f64::from(size.height) / 2.0) as f32,
            ]
        });
        projection
            .iter()
            .flatten()
            .all(|v| v.is_finite())
            .then_some(projection)
    }

    pub fn subject_segment(&self) -> usize {
        self.subject_segment
    }
}
struct Subject {
    calibration: Calibration,
    observations: Vec<Observation>,
    previous: Option<Pose>,
}
#[derive(Default)]
pub struct TwinRecorder {
    subjects: Vec<Subject>,
    last_frame: Option<u64>,
    last_bbox: Option<Bbox>,
    last_size: Option<Size>,
    recording: Option<RecordingTiming>,
}

#[derive(Serialize)]
pub struct RecordingTiming {
    pub fps: f64,
    pub frame_count: u64,
    pub timestamps_seconds: Vec<f64>,
    pub source_video: Option<String>,
}
fn overlap(a: Bbox, b: Bbox) -> f32 {
    let intersection = ((a.x + a.width).min(b.x + b.width) - a.x.max(b.x)).max(0.0)
        * ((a.y + a.height).min(b.y + b.height) - a.y.max(b.y)).max(0.0);
    intersection / (a.width * a.height + b.width * b.height - intersection).max(f32::EPSILON)
}
impl TwinRecorder {
    pub fn set_timing(&mut self, recording: RecordingTiming) {
        self.recording = Some(recording);
    }
    pub fn observe(
        &mut self,
        frame: u64,
        pose: &Pose3d,
        points: &[Keypoint],
        bbox: Bbox,
        size: Size,
        threshold: f32,
        focal: Option<f64>,
    ) -> Option<TwinPreview> {
        if pose.joints.len() != N || points.len() < N {
            return None;
        }
        let coco = std::array::from_fn(|i| [points[i].x, points[i].y, points[i].score]);
        let screen = motion3d::coco_to_h36m(coco);
        let obs = Observation {
            frame,
            points: std::array::from_fn(|i| {
                V::new(
                    pose.joints[i].x as f64,
                    pose.joints[i].y as f64,
                    pose.joints[i].z as f64,
                )
            }),
            scores: std::array::from_fn(|i| {
                let s = pose.joints[i].input_score as f64;
                if s.is_finite() && s >= threshold as f64 {
                    s.clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }),
            pixels: screen.map(|p| {
                if p[0].is_finite() && p[1].is_finite() {
                    [p[0] as f64, p[1] as f64]
                } else {
                    [0.0, 0.0]
                }
            }),
            pixel_scores: screen.map(|p| {
                if p.iter().all(|v| v.is_finite()) && p[0] >= 0.0 && p[1] >= 0.0 {
                    p[2].clamp(0.0, 1.0) as f64
                } else {
                    0.0
                }
            }),
            size,
            focal: focal.unwrap_or(size.width as f64),
            assumed_focal: focal.is_none(),
            history_frames: pose.history_frames,
        };
        if obs.points.iter().any(|p| !p.iter().all(|x| x.is_finite()))
            || obs.scores.iter().filter(|&&s| s > 0.0).count() < 4
        {
            return None;
        }
        let new = self.subjects.is_empty()
            || self.last_size != Some(size)
            || self
                .last_frame
                .is_some_and(|previous| frame.saturating_sub(previous) > 30)
            || self
                .last_bbox
                .is_some_and(|previous| overlap(previous, bbox) < 0.15);
        if new {
            self.subjects.push(Subject {
                calibration: Calibration::new(&obs),
                observations: Vec::new(),
                previous: None,
            });
        }
        self.last_frame = Some(frame);
        self.last_bbox = Some(bbox);
        self.last_size = Some(size);
        let subject_segment = self.subjects.len() - 1;
        let subject = self.subjects.last_mut().unwrap();
        subject
            .calibration
            .update(&obs, f64::from(threshold.max(0.6)));
        let dimensions = subject.calibration.dimensions();
        let solved = inverse(dimensions, &obs, subject.previous.as_ref());
        let parameters = frame_parameters(dimensions, &obs, &solved);
        let fitted = forward(dimensions, &solved).0;
        let joints: Vec<_> = (0..N)
            .map(|i| Joint3d {
                name: motion3d::NAMES[i],
                x: fitted[i].x as f32,
                y: fitted[i].y as f32,
                z: fitted[i].z as f32,
                input_score: pose.joints[i].input_score,
            })
            .collect();
        let fitted_pose = Pose3d {
            model: "skeletal_twin_fk",
            coordinate_space: "root_relative_camera",
            units: pose.units,
            history_frames: pose.history_frames,
            angles: motion3d::angles3d(&joints, threshold),
            joints,
        };
        subject.previous = Some(solved);
        subject.observations.push(obs);
        Some(TwinPreview {
            subject_segment,
            provisional: true,
            person: subject.calibration.parameters(),
            frame: parameters,
            fitted_pose,
        })
    }

    /// Final pass: one constant shape per subject segment, refit every recorded pose.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = BufWriter::new(
            File::create(path)
                .with_context(|| format!("Cannot create twin recording {}", path.display()))?,
        );
        writer.write_all(b"{\"schema_version\":1,\"quaternion_convention\":\"xyzw_active_parent_local\",\"coordinate_axes\":\"camera_x_right_y_down_z_forward\",\"recording\":")?;
        serde_json::to_writer(&mut writer, &self.recording)?;
        writer.write_all(b",\"joint_names\":")?;
        serde_json::to_writer(&mut writer, &motion3d::NAMES)?;
        writer.write_all(b",\"parents\":")?;
        serde_json::to_writer(
            &mut writer,
            &PARENT
                .map(Some)
                .iter()
                .enumerate()
                .map(|(i, p)| if i == 0 { None } else { *p })
                .collect::<Vec<_>>(),
        )?;
        writer.write_all(b",\"subjects\":[")?;
        for (index, subject) in self.subjects.iter().enumerate() {
            if index > 0 {
                writer.write_all(b",")?;
            }
            write!(writer, "{{\"subject_segment\":{index},\"person\":")?;
            serde_json::to_writer(&mut writer, &subject.calibration.parameters())?;
            writer.write_all(b",\"frames\":[")?;
            let dimensions = subject.calibration.dimensions();
            let mut previous = None;
            for (index, obs) in subject.observations.iter().enumerate() {
                if index > 0 {
                    writer.write_all(b",")?;
                }
                let pose = inverse(dimensions, obs, previous.as_ref());
                let frame = frame_parameters(dimensions, obs, &pose);
                ensure!(frame.fit_rmse.is_finite(), "Non-finite skeletal twin fit");
                serde_json::to_writer(&mut writer, &frame)?;
                previous = Some(pose);
            }
            writer.write_all(b"]}")?;
        }
        writer.write_all(b"]}")?;
        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dimensions() -> Dimensions {
        Dimensions([0.24, 0.4, 0.42, 0.43, 0.28, 0.25, 0.12, 0.18, 0.13, 0.08])
    }
    fn observation(frame: u64, pose: &Pose) -> Observation {
        let positions = forward(dimensions(), pose).0;
        Observation {
            frame,
            points: positions,
            scores: [0.9; N],
            pixels: positions.map(|p| [320.0 + p.x * 500.0, 240.0 + p.y * 500.0]),
            pixel_scores: [0.9; N],
            size: Size::new(640, 480),
            focal: 800.0,
            assumed_focal: false,
            history_frames: 27,
        }
    }
    fn pose() -> Pose {
        let mut p = Pose {
            orientation: Q::from_euler_angles(0.2, -0.3, 0.1),
            ..Pose::default()
        };
        p.local[1] = Q::from_euler_angles(0.1, 0.0, 0.3);
        p.local[2] = Q::from_euler_angles(0.5, 0.0, 0.0);
        p.local[11] = Q::from_euler_angles(0.3, 0.1, 0.8);
        p.local[12] = Q::from_euler_angles(0.7, 0.0, 0.2);
        p
    }
    #[test]
    fn forward_kinematics_preserves_lengths_and_pelvis_width_under_rotations() {
        let dimensions = dimensions();
        let mut p = pose();
        p.root = V::new(3.0, 2.0, 5.0);
        let (positions, _) = forward(dimensions, &p);
        for (i, offset) in dimensions.offsets().iter().enumerate().skip(1) {
            assert!(((positions[i] - positions[PARENT[i]]).norm() - offset.norm()).abs() < 1e-12);
        }
        assert!(((positions[4] - positions[1]).norm() - dimensions.0[0]).abs() < 1e-12);
        assert!(((positions[11] - positions[14]).norm() - dimensions.0[1]).abs() < 1e-12);
    }
    #[test]
    fn inverse_kinematics_reduces_observation_error_without_stretching_bones() {
        let obs = observation(0, &pose());
        let dimensions = dimensions();
        let delta = 0.1;
        let before = cost(&forward(dimensions, &Pose::default()).0, &obs, delta);
        let solved = inverse(dimensions, &obs, None);
        let positions = forward(dimensions, &solved).0;
        let after = cost(&positions, &obs, delta);
        assert!(
            after < before * 0.05,
            "IK residual did not improve enough: {before} -> {after}"
        );
        for i in 1..N {
            assert!(
                ((positions[i] - positions[PARENT[i]]).norm() - dimensions.offsets()[i].norm())
                    .abs()
                    < 1e-12
            );
        }
    }
    #[test]
    fn calibration_uses_warm_reliable_frames_and_resists_outliers() {
        let obs = observation(0, &pose());
        let mut calibration = Calibration::new(&obs);
        let mut cold = obs.clone();
        cold.history_frames = 1;
        calibration.update(&cold, 0.6);
        assert_eq!(calibration.frames, 0);
        for i in 0..200 {
            let mut noisy = obs.clone();
            for j in 0..N {
                noisy.points[j] += V::new(
                    ((i * 17 + j * 3) as f64).sin() * 0.002,
                    ((i * 11 + j * 7) as f64).cos() * 0.002,
                    0.0,
                );
            }
            calibration.update(&noisy, 0.6);
        }
        let before = calibration.dimensions().0;
        let mut bad = obs.clone();
        bad.points[2] *= 10.0;
        bad.points[3] *= 10.0;
        calibration.update(&bad, 0.6);
        let after = calibration.dimensions().0;
        assert!((after[2] - before[2]).abs() < before[2] * 0.01);
        for (actual, expected) in after.iter().zip(dimensions().0) {
            assert!((actual - expected).abs() < expected * 0.05);
        }
        let mut unreliable = obs;
        unreliable.scores = [0.1; N];
        let count = calibration.frames;
        calibration.update(&unreliable, 0.6);
        assert_eq!(calibration.frames, count);
    }
    #[test]
    fn root_translation_recovers_known_weak_perspective_camera_and_image_motion() {
        let mut obs = observation(0, &Pose::default());
        let root = camera_translation(&obs, &obs.points).unwrap();
        assert!(root.x.abs() < 1e-12 && root.y.abs() < 1e-12 && (root.z - 1.6).abs() < 1e-12);
        for p in &mut obs.pixels {
            p[0] += 50.0;
            p[1] -= 25.0;
        }
        let root = camera_translation(&obs, &obs.points).unwrap();
        assert!((root.x - 0.1).abs() < 1e-12 && (root.y + 0.05).abs() < 1e-12);
    }

    #[test]
    fn tracking_projection_waits_for_calibration_and_rejects_poor_fits() {
        let obs = observation(0, &Pose::default());
        let calibration = Calibration::new(&obs);
        let joints: Vec<_> = (0..N)
            .map(|i| Joint3d {
                name: motion3d::NAMES[i],
                x: obs.points[i].x as f32,
                y: obs.points[i].y as f32,
                z: obs.points[i].z as f32,
                input_score: 0.9,
            })
            .collect();
        let mut preview = TwinPreview {
            subject_segment: 0,
            provisional: true,
            person: calibration.parameters(),
            frame: frame_parameters(dimensions(), &obs, &Pose::default()),
            fitted_pose: Pose3d {
                model: "test",
                coordinate_space: "root_relative_camera",
                units: "model_normalized",
                history_frames: 27,
                angles: Vec::new(),
                joints,
            },
        };
        assert!(preview.tracking_projection(obs.size).is_none());
        preview.person.calibration_frames = 5;
        let projected = preview.tracking_projection(obs.size).unwrap();
        for (actual, expected) in projected.iter().zip(obs.pixels) {
            assert!((f64::from(actual[0]) - expected[0]).abs() < 1e-4);
            assert!((f64::from(actual[1]) - expected[1]).abs() < 1e-4);
        }
        preview.frame.fit_rmse = 1.0;
        assert!(preview.tracking_projection(obs.size).is_none());
    }
    #[test]
    fn final_recording_has_one_shape_and_reconstructable_rotations_for_all_frames() {
        let mut recorder = TwinRecorder::default();
        let first = observation(0, &pose());
        let mut subject = Subject {
            calibration: Calibration::new(&first),
            observations: Vec::new(),
            previous: None,
        };
        for frame in 0..4 {
            let mut p = pose();
            p.local[2] = Q::from_euler_angles(frame as f64 * 0.2, 0.0, 0.0);
            let obs = observation(frame, &p);
            subject.calibration.update(&obs, 0.6);
            subject.observations.push(obs);
        }
        recorder.subjects.push(subject);
        let path =
            std::env::temp_dir().join(format!("lifting-twin-test-{}.json", std::process::id()));
        recorder.save(&path).unwrap();
        let recording: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        std::fs::remove_file(path).unwrap();
        let subject = &recording["subjects"][0];
        assert_eq!(subject["frames"].as_array().unwrap().len(), 4);
        let vec3 = |v: &serde_json::Value| {
            V::new(
                v[0].as_f64().unwrap(),
                v[1].as_f64().unwrap(),
                v[2].as_f64().unwrap(),
            )
        };
        let quat = |v: &serde_json::Value| {
            let norm = v
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap().powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
            Q::from_quaternion(nalgebra::Quaternion::new(
                v[3].as_f64().unwrap(),
                v[0].as_f64().unwrap(),
                v[1].as_f64().unwrap(),
                v[2].as_f64().unwrap(),
            ))
        };
        let offsets: [V; N] = std::array::from_fn(|i| vec3(&subject["person"]["rest_offsets"][i]));
        // Reconstruct using only the exported contract, independently of forward().
        for frame in subject["frames"].as_array().unwrap() {
            assert!(frame.get("person").is_none());
            let mut positions = [V::zeros(); N];
            let mut world = [Q::identity(); N];
            positions[0] = vec3(&frame["root_position"]);
            world[0] = quat(&frame["root_orientation_xyzw"]);
            for i in 1..N {
                positions[i] = positions[PARENT[i]] + world[PARENT[i]] * offsets[i];
                world[i] = world[PARENT[i]] * quat(&frame["joint_rotations"][i]["quaternion_xyzw"]);
                assert!(
                    ((positions[i] - positions[PARENT[i]]).norm() - offsets[i].norm()).abs()
                        < 1e-10
                );
            }
        }
    }
    #[test]
    fn repeated_fits_keep_rotations_normalized_and_fk_lengths_constant() {
        let dimensions = dimensions();
        let mut previous = None;
        for i in 0..60 {
            let mut source = pose();
            source.orientation = Q::from_euler_angles(0.2, (i as f64 * 0.1).sin(), 0.1);
            source.local[12] = Q::from_euler_angles(i as f64 * 0.03, 0.0, 0.2);
            let obs = observation(i, &source);
            let fitted = inverse(dimensions, &obs, previous.as_ref());
            let frame = frame_parameters(dimensions, &obs, &fitted);
            for rotation in &frame.joint_rotations {
                let norm = rotation
                    .quaternion_xyzw
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt();
                assert!((norm - 1.0).abs() < 1e-12);
            }
            let p = forward(dimensions, &fitted).0;
            for j in 1..N {
                assert!(
                    ((p[j] - p[PARENT[j]]).norm() - dimensions.offsets()[j].norm()).abs() < 1e-12
                );
            }
            previous = Some(fitted);
        }
    }
}
