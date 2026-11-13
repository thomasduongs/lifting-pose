//! Shared body dimensions plus per-frame rotations. Online calibration is
//! provisional; final export re-solves every frame using one final shape per subject.
use crate::details::{HAND_PARENT, HandFrame, HandPose, HandShape, HeadPose};
use crate::{
    motion3d::{self, Joint3d, Pose3d},
    pose::{Bbox, Keypoint},
};
use anyhow::{Context, Result, ensure};
use nalgebra::{Matrix2x3, Matrix3, Rotation3, UnitQuaternion, Vector2, Vector3};
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
const MODEL_N: usize = 17;
const HEAD_PIVOT: usize = MODEL_N;
const N: usize = MODEL_N + 1;
const PARENT: [usize; N] = [
    0, 0, 1, 2, 0, 4, 5, 0, 7, HEAD_PIVOT, 9, 8, 11, 12, 8, 14, 15, 8,
];
// Keep model indices stable internally; exports remain parent-before-child.
const ORDER: [usize; N] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, HEAD_PIVOT, 9, 10, 11, 12, 13, 14, 15, 16,
];
fn joint_name(i: usize) -> &'static str {
    if i == HEAD_PIVOT {
        "head_pivot"
    } else {
        motion3d::NAMES[i]
    }
}
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
fn head_dimensions(body: Dimensions) -> Dimensions {
    #[cfg(test)]
    if let Ok(value) = std::env::var("LIFTING_EVAL_HEAD_LENGTH") {
        let value: f64 = value.parse().unwrap();
        assert!(value.is_finite() && value > 0.01 && value < 0.3);
        let mut result = body;
        result.0[9] = value;
        return result;
    }
    body
}
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
            V::zeros(),
        ]
    }
}

#[derive(Clone)]
struct Pose {
    time_seconds: f64,
    root: V,
    orientation: Q,
    local: [Q; N],
    camera: Option<V>,
    depth_inferred: bool,
}
impl Default for Pose {
    fn default() -> Self {
        Self {
            time_seconds: 0.0,
            root: V::zeros(),
            orientation: Q::identity(),
            local: [Q::identity(); N],
            camera: None,
            depth_inferred: false,
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
    for &i in &ORDER[1..] {
        let parent = PARENT[i];
        positions[i] = positions[parent] + rotations[parent] * offsets[i];
        rotations[i] = if i == HEAD_PIVOT && pose.local[i] == Q::identity() {
            rotations[parent]
        } else {
            unit(rotations[parent] * pose.local[i])
        };
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
    ear_head: bool,
    head_radius: Option<Estimate>,
    upper_body: bool,
    direct_depth: bool,
    estimates: [Estimate; 10],
    revision: u64,
    frames: u64,
}
impl Calibration {
    fn new(obs: &Observation) -> Self {
        let upper_body = is_upper(obs);
        let raw = if upper_body {
            measurements(&obs.points, &obs.scores, 0.1)
        } else {
            measurements(&obs.points, &[1.0; N], 0.0)
        };
        let scale = if upper_body {
            (obs.points[11] - obs.points[14]).norm().max(0.05)
        } else {
            (obs.points[8] - obs.points[0]).norm().max(0.05)
        };
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
            upper_body,
            ear_head: obs.ear_head,
            head_radius: obs.head_radius.map(Estimate::new),
            direct_depth: obs.direct_depth,
            estimates,
            revision: 0,
            frames: 0,
        }
    }
    fn update(&mut self, obs: &Observation, minimum: f64) {
        let old_head_radius = self.head_radius.as_ref().map(|e| e.value);
        if let Some(radius) = obs.head_radius {
            self.head_radius
                .get_or_insert_with(|| Estimate::new(radius))
                .add(radius, 1.0);
        }
        // MotionAGFormer startup repeats poses. Do not count those as calibration evidence.
        if !obs.direct_depth && obs.history_frames < 27 {
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
            || old_head_radius != self.head_radius.as_ref().map(|e| e.value)
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
            ear_head: self.ear_head,
            head_radius: self.head_radius.as_ref().map(|e| e.value),
            units: if self.direct_depth {
                "mediapipe_estimated_meters"
            } else {
                "motionagformer_model_units"
            },
            revision: self.revision,
            pelvis_width: (!self.upper_body).then_some(dimensions.0[0]),
            body_scope: if self.upper_body {
                "upper_body"
            } else {
                "full_body"
            },
            active_joint_indices: ORDER
                .iter()
                .enumerate()
                .filter(|(_, i)| {
                    **i != HEAD_PIVOT && (!self.upper_body || crate::scope::body_upper(**i))
                })
                .map(|(export, _)| export)
                .collect(),
            shoulder_width: dimensions.0[1],
            segment_lengths: (2..10)
                .filter(|i| !self.upper_body || [4, 5, 8, 9].contains(i))
                .map(|i| (DIMENSIONS[i], dimensions.0[i]))
                .collect(),
            rest_offsets: ORDER.map(|i| array(dimensions.offsets()[i])),
            bilateral_symmetry: true,
            calibration_frames: self.frames,
            hands: BTreeMap::new(),
            measurement_counts: (0..10)
                .filter(|i| !self.upper_body || [1, 4, 5, 8, 9].contains(i))
                .map(|i| (DIMENSIONS[i], self.estimates[i].samples))
                .chain(self.head_radius.iter().map(|e| ("head_radius", e.samples)))
                .collect(),
        }
    }
}

#[derive(Serialize)]
pub struct PersonParameters {
    ear_head: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    head_radius: Option<f64>,
    units: &'static str,
    revision: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pelvis_width: Option<f64>,
    body_scope: &'static str,
    active_joint_indices: Vec<usize>,
    shoulder_width: f64,
    segment_lengths: BTreeMap<&'static str, f64>,
    /// These offsets are shared by all frames of the finalized subject.
    rest_offsets: [[f64; 3]; N],
    bilateral_symmetry: bool,
    calibration_frames: u64,
    measurement_counts: BTreeMap<&'static str, u64>,
    hands: BTreeMap<String, HandShape>,
}

#[derive(Clone)]
struct Observation {
    ear_head: bool,
    head_radius: Option<f64>,
    frame: u64,
    time_seconds: f64,
    direct_depth: bool,
    torso_scale: Option<f64>,
    torso_span: usize,
    depth_hint: Option<f64>,
    heading_hint: Option<f64>,
    assume_upright_torso: bool,
    points: [V; N],
    scores: [f64; N],
    pixels: [[f64; 2]; N],
    pixel_scores: [f64; N],
    size: Size,
    focal: f64,
    assumed_focal: bool,
    history_frames: usize,
    hands: Vec<HandPose>,
    head: Option<HeadPose>,
}

fn is_upper(obs: &Observation) -> bool {
    obs.scores[..8].iter().all(|s| *s == 0.0)
}

/// Estimate skull width from near-frontal image evidence at the body's scale.
/// Profile faces/shoulders cannot supply a width measurement. This is shared
/// shape calibration, not a per-frame resize or an anatomical measurement.
fn head_radius_measurement(pose: &Pose3d, points: &[Keypoint]) -> Option<f64> {
    if !pose.ear_head || [0, 3, 4, 5, 6].into_iter().any(|i| points[i].score < 0.8) {
        return None;
    }
    let a = &pose.joints[11];
    let b = &pose.joints[14];
    let width = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt();
    if (a.x - b.x).abs() / width.max(1e-6) < 0.85 {
        return None;
    }
    let ears = (points[3].x - points[4].x).hypot(points[3].y - points[4].y);
    let shoulders = (points[5].x - points[6].x).hypot(points[5].y - points[6].y);
    if ears < 10.0
        || shoulders < 30.0
        || (points[0].x - (points[3].x + points[4].x) / 2.0).abs() > ears * 0.2
    {
        return None;
    }
    let radius = (ears / shoulders * width * 0.55) as f64;
    (radius.is_finite() && (0.025..0.2).contains(&radius)).then_some(radius)
}
fn align_upper_anchor(dimensions: Dimensions, pose: &mut Pose, obs: &Observation) {
    if is_upper(obs) {
        pose.root = V::zeros();
        pose.root = obs.points[8] - forward(dimensions, pose).0[8];
    }
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

fn project(p: V, camera: V, obs: &Observation) -> Option<(Vector2<f64>, Matrix2x3<f64>)> {
    let p = p + camera;
    if p.z <= 0.01 {
        return None;
    }
    let f = obs.focal;
    Some((
        Vector2::new(
            f * p.x / p.z + obs.size.width as f64 / 2.0,
            f * p.y / p.z + obs.size.height as f64 / 2.0,
        ),
        Matrix2x3::new(
            f / p.z,
            0.0,
            -f * p.x / (p.z * p.z),
            0.0,
            f / p.z,
            -f * p.y / (p.z * p.z),
        ),
    ))
}
fn huber(r: f64, delta: f64) -> f64 {
    if r <= delta {
        r * r / 2.0
    } else {
        delta * (r - delta / 2.0)
    }
}
fn image_weight(obs: &Observation, i: usize) -> f64 {
    // Keep camera and rotational updates on exactly the same image likelihood.
    // Reliable arm pixels constrain visible motion more directly than the
    // monocular world prediction. Blend confidence continuously so a score
    // crossing a threshold cannot abruptly change the objective.
    obs.pixel_scores[i].powi(2)
        * if [12, 13, 15, 16].contains(&i) {
            1.0 + 2.0 * ((obs.pixel_scores[i] - 0.6) / 0.3).clamp(0.0, 1.0)
        } else {
            1.0
        }
}
struct FitContext<'a> {
    obs: &'a Observation,
    camera: Option<V>,
    previous: Option<[V; N]>,
    temporal: f64,
    sequence_target: Option<[V; N]>,
    delta: f64,
    image_unit: f64,
}
fn relative_depth_jacobian(positions: &[V; N], i: usize, node: usize) -> V {
    let pivot = if descendant(8, node) || node == 8 {
        positions[8]
    } else {
        positions[node]
    };
    let arm = positions[i] - pivot;
    V::new(arm.y, -arm.x, 0.0)
}
impl FitContext<'_> {
    fn unmeasured_depth_weight(&self, i: usize) -> f64 {
        // A measured pixel does not measure depth. Stabilize only rejected
        // native arm depth, relative to the current thorax; leave image motion
        // and accepted native depth observations free of temporal averaging.
        if self.obs.direct_depth
            && [12, 13, 15, 16].contains(&i)
            && self.obs.scores[i] == 0.0
            && self.obs.pixel_scores[i] > 0.0
        {
            self.temporal * 2.0
        } else {
            0.0
        }
    }
    fn temporal_weight(&self, i: usize) -> f64 {
        // Measured joints must not be pulled toward an older pose. Retain
        // continuity only where neither the 3D nor image position is observed.
        if i == HEAD_PIVOT
            || (is_upper(self.obs) && i < 8)
            || self.obs.scores[i] > 0.0
            || self.obs.pixel_scores[i] > 0.0
        {
            0.0
        } else {
            self.temporal * (1.0 + 2.0 * (1.0 - self.obs.scores[i]))
        }
    }
    fn sequence_weight(&self, i: usize) -> f64 {
        if i == HEAD_PIVOT {
            0.0
        } else {
            0.12 + 0.18 * (1.0 - self.obs.scores[i].max(self.obs.pixel_scores[i]))
        }
    }
    fn cost(&self, positions: &[V; N]) -> f64 {
        let native = self.obs.direct_depth;
        let mut result = cost(positions, self.obs, self.delta) * if native { 0.4 } else { 1.0 };
        for i in 1..N {
            if let Some(camera) = self.camera {
                if let Some((pixel, _)) = project(positions[i], camera, self.obs) {
                    let r = (pixel - Vector2::from(self.obs.pixels[i])).norm();
                    result +=
                        2.0 * image_weight(self.obs, i) * huber(r, 12.0) * self.image_unit.powi(2);
                } else if self.obs.pixel_scores[i] > 0.0 {
                    return f64::INFINITY;
                }
            }
            if let Some(target) = &self.sequence_target {
                result += self.sequence_weight(i) * (positions[i] - target[i]).norm_squared() / 2.0;
            }
            if let Some(previous) = &self.previous {
                result +=
                    self.temporal_weight(i) * (positions[i] - previous[i]).norm_squared() / 2.0;
                let dz = (positions[i].z - positions[8].z) - (previous[i].z - previous[8].z);
                result += self.unmeasured_depth_weight(i) * dz * dz / 2.0;
            }
        }
        result
    }
}

/// Prevent a fitted elbow/knee from folding beyond a broad anatomical envelope.
/// The bend plane remains free; this does not impose an uncalibrated hinge axis.
fn limit_bends(pose: &mut Pose) {
    for node in [2, 5, 12, 15] {
        let axis = if node == 12 {
            V::x()
        } else if node == 15 {
            -V::x()
        } else {
            V::y()
        };
        let direction = pose.local[node] * axis;
        let angle = axis.dot(&direction).clamp(-1.0, 1.0).acos();
        let max = 165.0_f64.to_radians();
        if angle > max {
            let mut normal = axis.cross(&direction);
            if normal.norm() < 1e-8 {
                normal = V::z();
            }
            let target = Q::from_scaled_axis(normal.normalize() * max) * axis;
            if let Some(correction) = Q::rotation_between(&direction, &target) {
                pose.local[node] = unit(correction * pose.local[node]);
            }
        }
    }
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
    let x = if is_upper(obs) {
        obs.points[11] - obs.points[14]
    } else {
        obs.points[4] - obs.points[1]
    };
    let down = if is_upper(obs) {
        V::y()
    } else {
        obs.points[0] - obs.points[8]
    };
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

/// Remove unobservable torso lean while preserving the chest's world rotation.
/// Node 8 still controls the shoulder girdle/neck, so arms and head remain free.
fn constrain_upright_torso(pose: &mut Pose, upper_body: bool) {
    let chest = unit(pose.orientation * pose.local[7] * pose.local[8]);
    let right = if upper_body {
        chest * V::x()
    } else {
        pose.orientation * V::x()
    };
    let heading = if right.x.hypot(right.z) > 1e-8 {
        (-right.z).atan2(right.x)
    } else {
        let old = pose.orientation * V::x();
        (-old.z).atan2(old.x)
    };
    pose.orientation = Q::from_axis_angle(&V::y_axis(), heading);
    pose.local[7] = Q::identity();
    pose.local[8] = unit(pose.orientation.inverse() * chest);
}

fn constrain_observed_torso(pose: &mut Pose, obs: &Observation) {
    constrain_upright_torso(pose, is_upper(obs));
    if let Some(heading) = obs.heading_hint {
        pose.orientation = Q::from_axis_angle(&V::y_axis(), heading);
    }
}

fn initial_pose(dimensions: Dimensions, obs: &Observation) -> Pose {
    let mut pose = Pose {
        time_seconds: obs.time_seconds,
        orientation: root_orientation(obs),
        ..Pose::default()
    };
    for &node in &ORDER[1..] {
        if node == HEAD_PIVOT {
            continue;
        }
        if is_upper(obs) && node < 8 {
            continue;
        }
        if let Some(child) = (1..N)
            .filter(|&i| node != 8 || i == 11)
            .find(|&i| PARENT[i] == node && obs.scores[i] > 0.0)
        {
            let (positions, world) = forward(dimensions, &pose);
            let from = positions[child] - positions[node];
            let to = obs.points[child] - obs.points[node];
            if let Some(delta) = Q::rotation_between(&from, &to) {
                pose.local[node] = unit(world[PARENT[node]].inverse() * delta * world[node]);
            }
        }
    }
    if obs.assume_upright_torso {
        constrain_observed_torso(&mut pose, obs);
    }
    pose
}

/// Block coordinate damped least-squares IK with analytic rotational Jacobians.
/// Each step changes rotations only: dimensions remain fixed for the entire solve.
fn inverse(dimensions: Dimensions, obs: &Observation, previous: Option<&Pose>) -> Pose {
    inverse_sequence(dimensions, obs, previous, None)
}
fn inverse_sequence(
    dimensions: Dimensions,
    obs: &Observation,
    previous: Option<&Pose>,
    sequence_target: Option<[V; N]>,
) -> Pose {
    let mut initialized = initial_pose(dimensions, obs);
    align_upper_anchor(dimensions, &mut initialized, obs);
    let scale = if is_upper(obs) {
        dimensions.0[4]
    } else {
        dimensions.0[2]
    };
    let delta = scale * 0.2;
    let mut previous = previous.cloned();
    if previous.as_ref().is_some_and(|p| {
        obs.time_seconds - p.time_seconds > 0.5 || obs.time_seconds < p.time_seconds
    }) {
        previous = None;
    }
    if let Some(p) = previous.as_mut() {
        if obs.direct_depth && is_upper(obs) && obs.scores[11].min(obs.scores[14]) >= 0.5 {
            let observed = obs.points[11] - obs.points[14];
            let before = forward(dimensions, p).0;
            let old = before[11] - before[14];
            if observed.x.hypot(observed.z) > 1e-6 && old.x.hypot(old.z) > 1e-6 {
                let difference = (-observed.z).atan2(observed.x) - (-old.z).atan2(old.x);
                let yaw = difference.sin().atan2(difference.cos());
                // Missing limbs should follow the measured torso turn, not resist
                // it by staying fixed in camera coordinates.
                p.orientation = unit(Q::from_axis_angle(&V::y_axis(), yaw) * p.orientation);
            }
        }
        if obs.assume_upright_torso {
            constrain_observed_torso(p, obs);
        }
        let old_root = p.root;
        align_upper_anchor(dimensions, p, obs);
        // Rebase the camera seed when the model's arbitrary hip-relative origin
        // moves. Its camera-space body position must not move with that origin.
        p.camera = p.camera.map(|camera| camera + old_root - p.root);
        // Both starting candidates must carry unobserved arm articulation.
        // Otherwise a good new visible-arm fit can reset the hidden arm to rest.
        for (node, child) in [(11, 12), (12, 13), (14, 15), (15, 16)] {
            if obs.scores[child] == 0.0 && obs.pixel_scores[child] == 0.0 {
                initialized.local[node] = p.local[node];
            }
        }
    }
    let dt = previous
        .as_ref()
        .map(|p| obs.time_seconds - p.time_seconds)
        .unwrap_or(1.0 / 30.0)
        .max(1.0 / 240.0);
    let seed = previous.as_ref().and_then(|p| p.camera);
    let measured_camera = if obs.direct_depth {
        fit_camera_seeded(obs, &obs.points, seed)
    } else {
        None
    };
    let fixed_depth = obs.depth_hint.map(|depth| depth - obs.points[8].z);
    let mut context = FitContext {
        sequence_target,
        obs,
        delta,
        image_unit: fixed_depth
            .or(measured_camera.map(|p| p.z))
            .map(|z| z + if is_upper(obs) { obs.points[8].z } else { 0.0 })
            .unwrap_or(1.0)
            .max(0.05)
            / obs.focal,
        camera: if fixed_depth.is_some() {
            fit_camera_constrained(obs, &obs.points, seed, fixed_depth)
        } else {
            measured_camera
        },
        previous: if obs.direct_depth {
            previous.as_ref().map(|p| forward(dimensions, p).0)
        } else {
            None
        },
        temporal: if obs.direct_depth {
            (0.10 * (1.0 / 30.0 / dt).powi(2)).min(0.6)
        } else {
            0.0
        },
    };
    let mut pose = match previous {
        Some(previous)
            if context.cost(&forward(dimensions, &previous).0)
                < context.cost(&forward(dimensions, &initialized).0) =>
        {
            previous
        }
        _ => initialized,
    };
    if !is_upper(obs) {
        pose.root = V::zeros();
    }
    if obs.direct_depth {
        limit_bends(&mut pose);
    }
    let damping = scale.powi(2) * 0.002;
    // More coordinate sweeps let coupled shoulder/elbow/camera updates converge.
    // The cost-improvement exit below still stops already converged fits early.
    for _ in 0..36 {
        let points = forward(dimensions, &pose).0;
        if context.camera.is_some() {
            let before = context.cost(&points);
            let old = context.camera;
            context.camera = fit_camera_objective(obs, &points, old, fixed_depth, true);
            if context.cost(&points) > before {
                context.camera = old;
            }
        }
        let before = context.cost(&forward(dimensions, &pose).0);
        for &node in ORDER.iter().rev() {
            if node == HEAD_PIVOT {
                continue;
            }
            if obs.assume_upright_torso && (node == 7 || (is_upper(obs) && node == 0)) {
                continue;
            }
            if !(1..N).any(|i| descendant(i, node)) {
                continue;
            }
            let (positions, world) = forward(dimensions, &pose);
            let mut a = Matrix3::identity() * damping;
            let mut b = V::zeros();
            for i in 1..N {
                if !descendant(i, node) {
                    continue;
                }
                let pivot = if is_upper(obs) && (node == 0 || node == 7) {
                    positions[8]
                } else {
                    positions[node]
                };
                let arm = positions[i] - pivot;
                let residual = obs.points[i] - positions[i];
                let weight = obs.scores[i].powi(2)
                    * robust_weight(residual.norm(), delta)
                    * if obs.direct_depth { 0.4 } else { 1.0 };
                // delta_position = delta_rotation × arm.
                let jacobian =
                    Matrix3::new(0.0, arm.z, -arm.y, -arm.z, 0.0, arm.x, arm.y, -arm.x, 0.0);
                a += weight * jacobian.transpose() * jacobian;
                b += weight * jacobian.transpose() * residual;
                if let Some(camera) = context.camera {
                    if let Some((pixel, derivative)) = project(positions[i], camera, obs) {
                        let residual = Vector2::from(obs.pixels[i]) - pixel;
                        let units = context.image_unit;
                        let j = derivative * jacobian * units;
                        let weight =
                            2.0 * image_weight(obs, i) * robust_weight(residual.norm(), 12.0);
                        a += weight * j.transpose() * j;
                        b += weight * j.transpose() * (residual * units);
                    }
                }
                if let Some(target) = &context.sequence_target {
                    let weight = context.sequence_weight(i);
                    a += weight * jacobian.transpose() * jacobian;
                    b += weight * jacobian.transpose() * (target[i] - positions[i]);
                }
                if let Some(previous) = &context.previous {
                    let weight = context.temporal_weight(i);
                    a += weight * jacobian.transpose() * jacobian;
                    b += weight * jacobian.transpose() * (previous[i] - positions[i]);
                    let weight = context.unmeasured_depth_weight(i);
                    let jz = relative_depth_jacobian(&positions, i, node);
                    let residual =
                        (previous[i].z - previous[8].z) - (positions[i].z - positions[8].z);
                    a += weight * jz * jz.transpose();
                    b += weight * jz * residual;
                }
            }
            let Some(mut step) = a.lu().solve(&b) else {
                continue;
            };
            if obs.assume_upright_torso && node == 0 {
                step = V::new(0.0, b.y / a[(1, 1)].max(1e-12), 0.0);
            }
            if step.norm() > 0.3 {
                step *= 0.3 / step.norm();
            }
            if !step.iter().all(|v| v.is_finite()) {
                continue;
            }
            let old_cost = context.cost(&positions);
            for factor in [1.0, 0.5, 0.25] {
                let mut candidate = pose.clone();
                let rotation = Q::from_scaled_axis(step * factor);
                if node == 0 {
                    candidate.orientation = unit(rotation * pose.orientation);
                } else {
                    candidate.local[node] =
                        unit(world[PARENT[node]].inverse() * rotation * world[node]);
                }
                if obs.direct_depth {
                    limit_bends(&mut candidate);
                }
                if obs.assume_upright_torso {
                    constrain_observed_torso(&mut candidate, obs);
                }
                align_upper_anchor(dimensions, &mut candidate, obs);
                if context.cost(&forward(dimensions, &candidate).0) < old_cost {
                    pose = candidate;
                    break;
                }
            }
        }
        let after = context.cost(&forward(dimensions, &pose).0);
        if before - after < 1e-9 {
            break;
        }
    }
    pose.time_seconds = obs.time_seconds;
    pose.camera = context.camera;
    pose.depth_inferred = fixed_depth.is_some();
    pose
}

/// Refine selected branches while keeping body/camera/person dimensions fixed.
fn refine_fixed_branches(
    dimensions: Dimensions,
    obs: &Observation,
    base: &Pose,
    nodes: &[usize],
    head_direction: Option<(V, f64)>,
) -> Pose {
    let camera = base.camera.unwrap_or_else(V::zeros);
    let positions = forward(dimensions, base).0;
    let context = FitContext {
        sequence_target: None,
        obs,
        camera: Some(camera),
        previous: None,
        temporal: 0.0,
        delta: dimensions.0[4] * 0.2,
        image_unit: (positions[8].z + camera.z).max(0.05) / obs.focal,
    };
    let mut pose = base.clone();
    let head_depth_target = positions[10].z;
    let head_position_gain = if nodes.contains(&HEAD_PIVOT) && obs.ear_head {
        16.0
    } else {
        1.0
    };
    // The current body fit disambiguates head distance. Image observations
    // locate the skull; they do not justify switching to a distant depth branch.
    let head_depth_gain = if nodes.contains(&HEAD_PIVOT) && obs.ear_head {
        4.0
    } else {
        0.0
    };
    let cost = |positions: &[V; N]| {
        context.cost(positions)
            + head_direction.map_or(0.0, |(target, w)| {
                w * ((positions[10] - positions[9]) - target).norm_squared() / 2.0
            })
            + project(positions[10], camera, obs).map_or(0.0, |(pixel, _)| {
                2.0 * (head_position_gain - 1.0)
                    * image_weight(obs, 10)
                    * huber((pixel - Vector2::from(obs.pixels[10])).norm(), 12.0)
                    * context.image_unit.powi(2)
            })
            + head_depth_gain * (positions[10].z - head_depth_target).powi(2) / 2.0
    };
    let damping = dimensions.0[4].powi(2) * 0.002;
    for _ in 0..36 {
        let before = cost(&forward(dimensions, &pose).0);
        for &node in nodes {
            if !(1..N)
                .any(|i| descendant(i, node) && (obs.scores[i] > 0.0 || obs.pixel_scores[i] > 0.0))
            {
                continue;
            }
            let (positions, world) = forward(dimensions, &pose);
            let mut a = Matrix3::identity() * damping;
            let mut b = V::zeros();
            for i in 1..N {
                if !descendant(i, node) {
                    continue;
                }
                let arm = positions[i] - positions[node];
                let j = Matrix3::new(0.0, arm.z, -arm.y, -arm.z, 0.0, arm.x, arm.y, -arm.x, 0.0);
                let residual = obs.points[i] - positions[i];
                let weight = (if obs.direct_depth { 0.4 } else { 1.0 })
                    * obs.scores[i].powi(2)
                    * robust_weight(residual.norm(), context.delta);
                a += weight * j.transpose() * j;
                b += weight * j.transpose() * residual;
                if let Some((pixel, derivative)) = project(positions[i], camera, obs) {
                    let residual = Vector2::from(obs.pixels[i]) - pixel;
                    let jac = derivative * j * context.image_unit;
                    let w = 2.0
                        * image_weight(obs, i)
                        * robust_weight(residual.norm(), 12.0)
                        * if i == 10 { head_position_gain } else { 1.0 };
                    a += w * jac.transpose() * jac;
                    b += w * jac.transpose() * (residual * context.image_unit);
                }
            }
            if let Some((target, w)) = head_direction {
                let arm = positions[10] - positions[9];
                let j = Matrix3::new(0.0, arm.z, -arm.y, -arm.z, 0.0, arm.x, arm.y, -arm.x, 0.0);
                a += w * j.transpose() * j;
                b += w * j.transpose() * (target - arm);
            }
            if head_depth_gain > 0.0 {
                let arm = positions[10] - positions[node];
                let jac = V::new(arm.y, -arm.x, 0.0);
                a += head_depth_gain * jac * jac.transpose();
                b += head_depth_gain * jac * (head_depth_target - positions[10].z);
            }
            let Some(mut step) = a.lu().solve(&b) else {
                continue;
            };
            if !step.iter().all(|v| v.is_finite()) {
                continue;
            }
            if step.norm() > 0.3 {
                step *= 0.3 / step.norm();
            }
            let old = cost(&positions);
            for factor in [1.0, 0.5, 0.25] {
                let mut candidate = pose.clone();
                candidate.local[node] = unit(
                    world[PARENT[node]].inverse()
                        * Q::from_scaled_axis(step * factor)
                        * world[node],
                );
                if nodes.iter().any(|n| [11, 12, 14, 15].contains(n)) {
                    limit_bends(&mut candidate);
                }
                if cost(&forward(dimensions, &candidate).0) < old {
                    pose = candidate;
                    break;
                }
            }
        }
        if before - cost(&forward(dimensions, &pose).0) < 1e-9 {
            break;
        }
    }
    pose
}

#[cfg(test)]
fn refine_fixed_body_arms(dimensions: Dimensions, obs: &Observation, base: &Pose) -> Pose {
    refine_fixed_branches(dimensions, obs, base, &[12, 11, 15, 14], None)
}

fn refine_head(
    dimensions: Dimensions,
    obs: &Observation,
    base: &Pose,
    previous: Option<&Pose>,
) -> Pose {
    if base.camera.is_none() {
        return base.clone();
    }
    let mut seed = base.clone();
    if let Some(previous) =
        previous.filter(|p| (0.0..=0.5).contains(&(obs.time_seconds - p.time_seconds)))
    {
        if obs.scores[10] == 0.0 && obs.pixel_scores[10] == 0.0 {
            seed.local[9] = previous.local[9];
        }
        if (obs.ear_head && obs.scores[10] == 0.0 && obs.pixel_scores[10] == 0.0)
            || [9, 10]
                .iter()
                .all(|&i| obs.scores[i] == 0.0 && obs.pixel_scores[i] == 0.0)
        {
            seed.local[HEAD_PIVOT] = previous.local[HEAD_PIVOT];
        }
    }
    let direction = if obs.ear_head && obs.direct_depth {
        (obs.points[10] - obs.points[9])
            .try_normalize(1e-8)
            .map(|axis| {
                (
                    axis * dimensions.0[9],
                    obs.scores[9].min(obs.scores[10]).powi(2),
                )
            })
    } else {
        None
    };
    let mut head_obs = obs.clone();
    if obs.ear_head {
        // The ear midpoint locates the skull. The nose supplies direction,
        // rather than a second positional anchor for an approximate head shape.
        head_obs.scores[9] = 0.0;
        head_obs.pixel_scores[9] = 0.0;
    }
    refine_fixed_branches(dimensions, &head_obs, &seed, &[9, HEAD_PIVOT], direction)
}

/// Front-facing upper-body fallback: intersect a detected joint's camera ray
/// with the fixed-length sphere around its parent. This recovers foreshortening
/// that a nearly planar lifting prediction misses. The two depth branches are
/// still ambiguous; use the learned direction, then history, then a forward reach.
fn frontal_arm_observation(
    dimensions: Dimensions,
    obs: &Observation,
    previous: Option<&Pose>,
) -> Observation {
    let mut result = obs.clone();
    let shoulder = obs.points[11] - obs.points[14];
    if !is_upper(obs) || shoulder.z.abs() > shoulder.norm() * 0.5 {
        return result;
    }
    let mut torso = obs.clone();
    for i in [12, 13, 15, 16] {
        torso.scores[i] = 0.0;
    }
    let Some(translation) = camera_translation(&torso, &obs.points) else {
        return result;
    };
    let previous_points = previous.map(|p| forward(dimensions, p).0);
    for (parent, child, length) in [
        (11, 12, dimensions.0[4]),
        (12, 13, dimensions.0[5]),
        (14, 15, dimensions.0[4]),
        (15, 16, dimensions.0[5]),
    ] {
        if obs.scores[parent]
            .min(obs.scores[child])
            .min(obs.pixel_scores[parent])
            .min(obs.pixel_scores[child])
            < 0.6
        {
            continue;
        }
        let ray = V::new(
            (obs.pixels[child][0] - obs.size.width as f64 / 2.0) / obs.focal,
            (obs.pixels[child][1] - obs.size.height as f64 / 2.0) / obs.focal,
            1.0,
        );
        let center = result.points[parent] + translation;
        let a = ray.norm_squared();
        let b = ray.dot(&center);
        let discriminant = b * b - a * (center.norm_squared() - length * length);
        if discriminant <= 0.0 {
            continue;
        }
        let near = ray * ((b - discriminant.sqrt()) / a) - translation;
        let far = ray * ((b + discriminant.sqrt()) / a) - translation;
        let learned = obs.points[child].z - obs.points[parent].z;
        let direction = if learned.abs() > length * 0.15 {
            learned
        } else if let Some(points) = &previous_points {
            points[child].z - points[parent].z
        } else {
            -length
        };
        let target = if direction > length * 0.05 { far } else { near };
        // Only supplement a weak depth estimate, and never infer beyond the camera.
        if target.z + translation.z > 0.0
            && (target.z - result.points[parent].z).abs() > learned.abs() * 1.25
        {
            result.points[child] = target;
        }
    }
    result
}

// Palm segments avoid finger-flexion changes. Store their first reliable lengths
// rather than allowing the hand network to resize the depth reference each frame.
const PALM_PAIRS: [(usize, usize); 8] = [
    (0, 5),
    (0, 9),
    (0, 13),
    (0, 17),
    (5, 9),
    (9, 13),
    (13, 17),
    (5, 17),
];
struct HandDepthReference {
    lengths: [f64; 8],
    initial_scale: f64,
    depth_per_shoulder: f64,
    filtered_scale: f64,
}
#[derive(Default)]
struct DepthReferences {
    hands: BTreeMap<String, HandDepthReference>,
    body: BTreeMap<(usize, usize), BodyDepthReference>,
}
struct BodyDepthReference {
    initial_scale: f64,
    filtered_scale: f64,
    depth_per_shoulder: f64,
    last_frame: u64,
}

/// Size means landmark span, not a silhouette width. Compensate each span for
/// its lifted orientation before interpreting its scale change as depth.
fn body_size_observation(
    dimensions: Dimensions,
    obs: &Observation,
    camera: V,
    references: &mut BTreeMap<(usize, usize), BodyDepthReference>,
    corrected: &mut Observation,
) {
    let offsets = dimensions.offsets();
    let mut targets = [V::zeros(); N];
    let mut weights = [0.0; N];
    let pairs = (1..N)
        .map(|i| {
            (
                if PARENT[i] == HEAD_PIVOT {
                    8
                } else {
                    PARENT[i]
                },
                i,
                offsets[i].norm(),
            )
        })
        .chain([(11, 14, dimensions.0[1]), (1, 4, dimensions.0[0])]);
    for (a, b, length) in pairs {
        let confidence = obs.scores[a]
            .min(obs.scores[b])
            .min(obs.pixel_scores[a])
            .min(obs.pixel_scores[b]);
        let direction = obs.points[b] - obs.points[a];
        let orientation = direction.xy().norm() / direction.norm().max(1e-9);
        let pixels =
            (obs.pixels[b][0] - obs.pixels[a][0]).hypot(obs.pixels[b][1] - obs.pixels[a][1]);
        if confidence < 0.6 || orientation < 0.45 || pixels < 8.0 || length < 1e-5 {
            continue;
        }
        let scale = pixels / (length * orientation);
        let midpoint_depth = camera.z + (obs.points[a].z + obs.points[b].z) / 2.0;
        if !scale.is_finite() || !midpoint_depth.is_finite() || midpoint_depth <= 0.0 {
            continue;
        }
        let reference = references.entry((a, b)).or_insert(BodyDepthReference {
            initial_scale: scale,
            filtered_scale: scale,
            depth_per_shoulder: midpoint_depth / dimensions.0[1],
            last_frame: obs.frame,
        });
        // Skip smoothing across long detection gaps; never use an old size as a
        // fresh observation. Bound innovations during continuous tracking.
        if obs.frame.saturating_sub(reference.last_frame) > 30 {
            reference.filtered_scale = scale;
        } else {
            reference.filtered_scale *= (scale / reference.filtered_scale)
                .clamp(0.8, 1.25)
                .powf(0.35);
        }
        reference.last_frame = obs.frame;
        let depth = reference.depth_per_shoulder * dimensions.0[1] * reference.initial_scale
            / reference.filtered_scale;
        let model_midpoint = (obs.points[a].z + obs.points[b].z) / 2.0;
        for node in [a, b] {
            let z = depth + obs.points[node].z - model_midpoint;
            // Reject implausible candidates instead of moving a joint through the camera.
            if !z.is_finite() || z < camera.z * 0.1 || z > camera.z * 4.0 {
                continue;
            }
            let target = V::new(
                (obs.pixels[node][0] - obs.size.width as f64 / 2.0) * z / obs.focal,
                (obs.pixels[node][1] - obs.size.height as f64 / 2.0) * z / obs.focal,
                z,
            ) - camera;
            if !target.iter().all(|v| v.is_finite()) {
                continue;
            }
            let weight = confidence.powi(2) * orientation.powi(2);
            targets[node] += target * weight;
            weights[node] += weight;
        }
    }
    for i in 0..N {
        if weights[i] > 0.0 {
            // Size is an additional cue: retain the lifting prior and let FK/IK
            // reconcile overlapping segment estimates without stretching bones.
            corrected.points[i] = corrected.points[i] * 0.4 + targets[i] / weights[i] * 0.6;
        }
    }
}
fn palm_scale(hand: &HandPose, lengths: &[f64; 8]) -> Option<f64> {
    let mut scales = Vec::new();
    for (i, (a, b)) in PALM_PAIRS.into_iter().enumerate() {
        let delta = V::from_column_slice(&hand.world_landmarks[b].map(f64::from))
            - V::from_column_slice(&hand.world_landmarks[a].map(f64::from));
        let foreshortening = delta.xy().norm() / delta.norm().max(1e-9);
        let pixels = (f64::from(hand.image_landmarks[b][0] - hand.image_landmarks[a][0])).hypot(
            f64::from(hand.image_landmarks[b][1] - hand.image_landmarks[a][1]),
        );
        if foreshortening >= 0.45 && pixels >= 5.0 && lengths[i] > 1e-5 {
            let scale = pixels / (foreshortening * lengths[i]);
            if scale.is_finite() {
                scales.push(scale);
            }
        }
    }
    if scales.len() < 4 {
        return None;
    }
    scales.sort_by(f64::total_cmp);
    let median = scales[scales.len() / 2];
    // Reject inconsistent segment scales (bad landmarks or an unreliable palm).
    (scales
        .iter()
        .filter(|s| (**s / median - 1.0).abs() < 0.3)
        .count()
        >= 4)
        .then_some(median)
}
fn inverse_with_size(
    dimensions: Dimensions,
    obs: &Observation,
    previous: Option<&Pose>,
    references: &mut DepthReferences,
) -> (Pose, Observation) {
    if obs.direct_depth {
        return (inverse(dimensions, obs, previous), obs.clone());
    }
    let mut corrected = frontal_arm_observation(dimensions, obs, previous);
    let mut torso = obs.clone();
    for i in [12, 13, 15, 16] {
        torso.scores[i] = 0.0;
    }
    if let Some(camera) = camera_translation(&torso, &obs.points) {
        body_size_observation(
            dimensions,
            obs,
            camera,
            &mut references.body,
            &mut corrected,
        );
        for hand in &obs.hands {
            let (shoulder, elbow, wrist) = match hand.side.as_str() {
                "left" => (11, 12, 13),
                "right" => (14, 15, 16),
                _ => continue,
            };
            if hand.confidence < 0.8 || obs.scores[shoulder].min(obs.scores[wrist]) < 0.6 {
                continue;
            }
            let lengths = PALM_PAIRS.map(|(a, b)| {
                (V::from_column_slice(&hand.world_landmarks[b].map(f64::from))
                    - V::from_column_slice(&hand.world_landmarks[a].map(f64::from)))
                .norm()
            });
            let Some(scale) = palm_scale(
                hand,
                references
                    .hands
                    .get(&hand.side)
                    .map(|r| &r.lengths)
                    .unwrap_or(&lengths),
            ) else {
                continue;
            };
            let depth = camera.z + obs.points[wrist].z;
            if depth <= 0.0 {
                continue;
            }
            let reference =
                references
                    .hands
                    .entry(hand.side.clone())
                    .or_insert(HandDepthReference {
                        lengths,
                        initial_scale: scale,
                        filtered_scale: scale,
                        depth_per_shoulder: depth / dimensions.0[1],
                    });
            // Bound scale innovations and smooth multiplicative measurement noise.
            let ratio = (scale / reference.filtered_scale).clamp(0.75, 1.33);
            reference.filtered_scale *= ratio.powf(0.4);
            let depth = reference.depth_per_shoulder * dimensions.0[1] * reference.initial_scale
                / reference.filtered_scale;
            let pixel = hand.image_landmarks[0];
            let target = V::new(
                (f64::from(pixel[0]) - obs.size.width as f64 / 2.0) * depth / obs.focal,
                (f64::from(pixel[1]) - obs.size.height as f64 / 2.0) * depth / obs.focal,
                depth,
            ) - camera;
            if !target.iter().all(|v| v.is_finite()) {
                continue;
            }
            let anchor = corrected.points[shoulder];
            let delta = target - anchor;
            let distance = delta.norm();
            if distance < 1e-6 {
                continue;
            }
            let upper = dimensions.0[4];
            let lower = dimensions.0[5];
            let reach = distance.clamp((upper - lower).abs() + 1e-5, (upper + lower) * 0.995);
            let axis = delta / distance;
            let along = (upper * upper - lower * lower + reach * reach) / (2.0 * reach);
            let height = (upper * upper - along * along).max(0.0).sqrt();
            let prior = corrected.points[elbow] - anchor;
            let mut bend = prior - axis * prior.dot(&axis);
            if bend.norm() < 1e-6 {
                bend = V::y() - axis * axis.y;
            }
            if bend.norm() < 1e-6 {
                bend = V::x() - axis * axis.x;
            }
            corrected.points[wrist] = anchor + axis * reach;
            corrected.points[elbow] = anchor + axis * along + bend.normalize() * height;
        }
    }
    (inverse(dimensions, &corrected, previous), corrected)
}

#[derive(Serialize)]
pub struct JointRotation {
    joint: &'static str,
    quaternion_xyzw: [f64; 4],
    confidence: f64,
}
#[derive(Serialize)]
pub struct FrameParameters {
    frame: u64,
    root_position: [f64; 3],
    root_orientation_xyzw: [f64; 4],
    joint_rotations: Vec<JointRotation>,
    root_translation_valid: bool,
    root_depth_observed: bool,
    root_translation_model: &'static str,
    focal_length_px: f64,
    focal_length_assumed: bool,
    image_scale_px_per_unit: Option<f64>,
    fit_rmse: f64,
    reprojection_rmse_px: Option<f64>,
    angles: Vec<crate::angles::JointAngle>,
    hands: Vec<HandFrame>,
    head: Option<HeadPose>,
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
#[cfg(test)]
fn fit_camera(obs: &Observation, positions: &[V; N]) -> Option<V> {
    fit_camera_seeded(obs, positions, None)
}
fn profile_depth_ambiguous(obs: &Observation, positions: &[V; N]) -> bool {
    if !is_upper(obs) || !obs.direct_depth {
        return false;
    }
    let shoulder = positions[11] - positions[14];
    let width = (Vector2::from(obs.pixels[11]) - Vector2::from(obs.pixels[14])).norm();
    let head_height = ((obs.pixels[11][1] + obs.pixels[14][1]) / 2.0 - obs.pixels[9][1]).abs();
    shoulder.x.abs() < shoulder.norm() * 0.8
        || (obs.pixel_scores[9]
            .min(obs.pixel_scores[11])
            .min(obs.pixel_scores[14])
            >= 0.5
            && width < head_height * 1.3)
}
fn fit_camera_seeded(obs: &Observation, positions: &[V; N], seed: Option<V>) -> Option<V> {
    fit_camera_constrained(obs, positions, seed, None)
}
fn fit_camera_constrained(
    obs: &Observation,
    positions: &[V; N],
    seed: Option<V>,
    fixed_depth: Option<f64>,
) -> Option<V> {
    fit_camera_objective(obs, positions, seed, fixed_depth, false)
}
fn fit_camera_objective(
    obs: &Observation,
    positions: &[V; N],
    seed: Option<V>,
    fixed_depth: Option<f64>,
    joint_objective: bool,
) -> Option<V> {
    let mut camera = camera_translation(obs, positions).or(seed)?;
    if !obs.direct_depth {
        return Some(camera);
    }
    let profile = profile_depth_ambiguous(obs, &obs.points);
    // A foreshortened/occluded shoulder baseline cannot identify body depth.
    // Preserve the previous depth instead of using waving arms as a ruler.
    let hold_depth = (profile && seed.is_some()) || fixed_depth.is_some();
    if hold_depth {
        camera = seed.unwrap_or(camera);
        if let Some(depth) = fixed_depth {
            camera.z = depth;
        }
    }
    let weight = |i: usize| {
        if joint_objective {
            return image_weight(obs, i);
        }
        obs.pixel_scores[i].min(obs.scores[i]).powi(2)
            * if [0, 7, 8, 9, 10, 11, 14].contains(&i) {
                1.0
            } else if hold_depth {
                0.0
            } else {
                0.25
            }
    };
    let error = |camera: V| -> f64 {
        (0..N)
            .filter(|&i| weight(i) > 0.0)
            .map(|i| {
                project(positions[i], camera, obs)
                    .map(|(pixel, _)| {
                        weight(i) * huber((pixel - Vector2::from(obs.pixels[i])).norm(), 12.0)
                    })
                    .unwrap_or(f64::INFINITY)
            })
            .sum()
    };
    if let Some(seed) = seed {
        if !hold_depth && error(seed) < error(camera) {
            camera = seed;
        }
    }
    for _ in 0..8 {
        let mut a = Matrix3::identity() * 1e-4;
        let mut b = V::zeros();
        for i in 0..N {
            if weight(i) <= 0.0 {
                continue;
            }
            if let Some((pixel, j)) = project(positions[i], camera, obs) {
                let residual = Vector2::from(obs.pixels[i]) - pixel;
                let w = weight(i) * robust_weight(residual.norm(), 12.0);
                a += w * j.transpose() * j;
                b += w * j.transpose() * residual;
            }
        }
        if hold_depth {
            // Solve image-plane translation with the unobservable depth fixed.
            a[(0, 2)] = 0.0;
            a[(1, 2)] = 0.0;
            a[(2, 0)] = 0.0;
            a[(2, 1)] = 0.0;
            a[(2, 2)] = 1.0;
            b.z = 0.0;
        }
        let Some(mut step) = a.lu().solve(&b) else {
            break;
        };
        if !step.iter().all(|v| v.is_finite()) {
            break;
        }
        let max_step = camera.z * 0.2;
        if step.norm() > max_step {
            step *= max_step / step.norm();
        }
        let before = error(camera);
        let mut accepted = false;
        for scale in [1.0, 0.5, 0.25] {
            let candidate = camera + step * scale;
            if candidate.z > 0.05 && error(candidate) < before {
                camera = candidate;
                accepted = true;
                break;
            }
        }
        if !accepted || step.norm() < 1e-6 {
            break;
        }
    }
    Some(camera)
}
fn frame_parameters(dimensions: Dimensions, obs: &Observation, pose: &Pose) -> FrameParameters {
    let positions = forward(dimensions, pose).0;
    let fitted_joints: Vec<_> = (0..MODEL_N)
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
    let translation = pose
        .camera
        .or_else(|| fit_camera_seeded(obs, &positions, None));
    let reprojection_rmse_px = translation.and_then(|camera| {
        let mut sum = 0.0;
        let mut weights = 0.0;
        for i in 0..N {
            if let Some((pixel, _)) = project(positions[i], camera, obs) {
                // Pixel observations remain valid when independently measured
                // wrists have no trustworthy native 3D depth.
                let weight = obs.pixel_scores[i].powi(2);
                sum += weight * (pixel - Vector2::from(obs.pixels[i])).norm_squared();
                weights += weight;
            }
        }
        (weights > 0.0).then_some((sum / weights.max(1e-9)).sqrt())
    });
    FrameParameters {
        frame: obs.frame,
        root_position: array(translation.map(|t| t + pose.root).unwrap_or(V::zeros())),
        root_orientation_xyzw: quaternion(pose.orientation),
        joint_rotations: ORDER
            .into_iter()
            .map(|i| JointRotation {
                joint: joint_name(i),
                quaternion_xyzw: quaternion(pose.local[i]),
                confidence: obs.scores[i],
            })
            .collect(),
        root_translation_valid: translation.is_some(),
        root_depth_observed: translation.is_some()
            && !pose.depth_inferred
            && !profile_depth_ambiguous(obs, &positions),
        root_translation_model: if obs.direct_depth {
            "perspective_translation_fit"
        } else {
            "weak_perspective_approximation"
        },
        focal_length_px: obs.focal,
        focal_length_assumed: obs.assumed_focal,
        image_scale_px_per_unit: translation.map(|t| obs.focal / t.z),
        fit_rmse,
        reprojection_rmse_px,
        angles: motion3d::angles3d(&fitted_joints, 0.0),
        hands: Vec::new(),
        head: obs.head.clone(),
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
    pub fn tracking_projection(&self, size: Size) -> Option<[[f32; 2]; MODEL_N]> {
        if self.person.calibration_frames < 5 || !self.frame.root_translation_valid {
            return None;
        }
        let leg = self.person.segment_lengths[if self.person.body_scope == "upper_body" {
            "upper_arm"
        } else {
            "upper_leg"
        }];
        if self.frame.fit_rmse > leg * 0.3 {
            return None;
        }
        let root = self.frame.root_position;
        let scale = self
            .frame
            .image_scale_px_per_unit
            .unwrap_or(self.frame.focal_length_px / root[2]);
        if !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        let projection = std::array::from_fn(|i| {
            let p = &self.fitted_pose.joints[i];
            let scale = if self.frame.root_translation_model == "perspective_translation_fit" {
                self.frame.focal_length_px / (f64::from(p.z) + root[2]).max(0.01)
            } else {
                scale
            };
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
    hand_shapes: BTreeMap<String, Vec<Estimate>>,
    depth_references: DepthReferences,
    observations: Vec<Observation>,
    previous: Option<Pose>,
    previous_head: Option<Pose>,
}
fn registered_vectors(
    hand: &HandPose,
    frame: &FrameParameters,
    wrist_depth: f64,
) -> Option<[V; 21]> {
    if !frame.root_translation_valid || frame.root_position[2] <= 0.0 {
        return None;
    }
    let world: [V; 21] = std::array::from_fn(|i| {
        V::new(
            hand.world_landmarks[i][0] as f64,
            hand.world_landmarks[i][1] as f64,
            hand.world_landmarks[i][2] as f64,
        )
    });
    let relative = world.map(|p| p - world[0]);
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (i, p) in relative.iter().enumerate().skip(1) {
        numerator += p.x * f64::from(hand.image_landmarks[i][0] - hand.image_landmarks[0][0])
            + p.y * f64::from(hand.image_landmarks[i][1] - hand.image_landmarks[0][1]);
        denominator += p.x * p.x + p.y * p.y;
    }
    let pixels_per_unit = if frame.root_translation_model == "perspective_translation_fit" {
        if wrist_depth <= 0.01 {
            return None;
        }
        frame.focal_length_px / wrist_depth
    } else {
        frame
            .image_scale_px_per_unit
            .unwrap_or(frame.focal_length_px / frame.root_position[2])
    };
    let scale = numerator / denominator.max(1e-12) / pixels_per_unit;
    if !scale.is_finite() || !(0.001..100.0).contains(&scale) {
        return None;
    }
    let result = relative.map(|p| p * scale);
    (1..21)
        .all(|i| (result[i] - result[HAND_PARENT[i]]).norm() > 1e-7)
        .then_some(result)
}
impl Subject {
    fn parameters(&self) -> PersonParameters {
        let mut parameters = self.calibration.parameters();
        let dimensions = head_dimensions(self.calibration.dimensions());
        parameters.rest_offsets = ORDER.map(|i| array(dimensions.offsets()[i]));
        parameters
            .segment_lengths
            .insert("nose_to_head", dimensions.0[9]);
        parameters.hands = self
            .hand_shapes
            .iter()
            .map(|(side, lengths)| {
                (
                    side.clone(),
                    HandShape {
                        bone_lengths: lengths.iter().map(|e| e.value).collect(),
                        measurement_count: lengths
                            .iter()
                            .skip(1)
                            .map(|e| e.samples)
                            .min()
                            .unwrap_or(0),
                    },
                )
            })
            .collect();
        parameters
    }
    fn learn_hands(&mut self, obs: &Observation, pose: &Pose) {
        let frame = frame_parameters(self.calibration.dimensions(), obs, pose);
        let body = forward(self.calibration.dimensions(), pose).0;
        for hand in &obs.hands {
            let Some(points) = registered_vectors(
                hand,
                &frame,
                frame.root_position[2] + body[if hand.side == "left" { 13 } else { 16 }].z
                    - pose.root.z,
            ) else {
                continue;
            };
            let lengths = self
                .hand_shapes
                .entry(hand.side.clone())
                .or_insert_with(|| {
                    (0..21)
                        .map(|i| {
                            Estimate::new(if i == 0 {
                                0.0
                            } else {
                                (points[i] - points[HAND_PARENT[i]]).norm()
                            })
                        })
                        .collect()
                });
            if (obs.direct_depth || obs.history_frames >= 27) && hand.confidence >= 0.8 {
                for i in 1..21 {
                    lengths[i].add(
                        (points[i] - points[HAND_PARENT[i]]).norm(),
                        f64::from(hand.confidence).powi(2),
                    );
                }
            }
        }
    }
    fn frame(&self, obs: &Observation, pose: &Pose) -> FrameParameters {
        let mut frame = frame_parameters(head_dimensions(self.calibration.dimensions()), obs, pose);
        let body = forward(self.calibration.dimensions(), pose).0;
        for hand in &obs.hands {
            if !self.hand_shapes.contains_key(&hand.side) {
                continue;
            }
            let Some(points) = registered_vectors(
                hand,
                &frame,
                frame.root_position[2] + body[if hand.side == "left" { 13 } else { 16 }].z
                    - pose.root.z,
            ) else {
                continue;
            };
            let rotations = (0..21)
                .map(|i| {
                    let rotation = if i == 0 {
                        Q::identity()
                    } else {
                        Q::rotation_between(&V::y(), &(points[i] - points[HAND_PARENT[i]]))
                            .unwrap_or(Q::identity())
                    };
                    quaternion(rotation)
                })
                .collect();
            frame.hands.push(HandFrame {
                side: hand.side.clone(),
                confidence: hand.confidence,
                bone_rotations_xyzw: rotations,
            });
        }
        frame
    }
}

#[derive(Default)]
pub struct TwinRecorder {
    streaming: bool,
    streaming_segment: usize,
    subjects: Vec<Subject>,
    last_frame: Option<u64>,
    last_bbox: Option<Bbox>,
    last_size: Option<Size>,
    recording: Option<RecordingTiming>,
    free_torso: bool,
    next_torso_scale: Option<(f64, usize)>,
    no_stabilization: bool,
    no_smoothing: bool,
}

#[derive(Serialize)]
pub struct RecordingTiming {
    pub fps: f64,
    pub frame_count: u64,
    pub timestamps_seconds: Vec<f64>,
    pub source_video: Option<String>,
    pub trim_start_seconds: f64,
    pub trim_end_seconds: f64,
    pub upper_body: bool,
    pub assume_upright_torso: bool,
}
fn overlap(a: Bbox, b: Bbox) -> f32 {
    let intersection = ((a.x + a.width).min(b.x + b.width) - a.x.max(b.x)).max(0.0)
        * ((a.y + a.height).min(b.y + b.height) - a.y.max(b.y)).max(0.0);
    intersection / (a.width * a.height + b.width * b.height - intersection).max(f32::EPSILON)
}
impl TwinRecorder {
    /// Preserve calibration and continuity without retaining a recording in memory.
    pub fn streaming() -> Self {
        Self {
            streaming: true,
            ..Self::default()
        }
    }

    pub fn set_stabilization(&mut self, enabled: bool, smooth: bool) {
        self.no_stabilization = !enabled;
        self.no_smoothing = !smooth;
    }
    pub fn set_image_torso_scale(&mut self, scale: Option<f64>, span: usize) {
        self.next_torso_scale = scale
            .filter(|s| s.is_finite() && (0.8..1.25).contains(s) && (1..=12).contains(&span))
            .map(|s| (s, span));
    }
    pub fn set_upright_torso(&mut self, assume: bool) {
        self.free_torso = !assume;
    }
    pub fn set_timing(&mut self, recording: RecordingTiming) {
        self.recording = Some(recording);
    }
    pub fn observe(
        &mut self,
        frame: u64,
        time_seconds: f64,
        pose: &Pose3d,
        points: &[Keypoint],
        hands: &[HandPose],
        head: Option<HeadPose>,
        bbox: Bbox,
        size: Size,
        threshold: f32,
        focal: Option<f64>,
    ) -> Option<TwinPreview> {
        if pose.joints.len() != MODEL_N || points.len() < MODEL_N {
            return None;
        }
        let coco = std::array::from_fn(|i| [points[i].x, points[i].y, points[i].score]);
        let screen = if pose.model == "MediaPipe-Pose" {
            crate::mediapipe::skeleton_pixels_for(points, pose.ear_head)
        } else {
            motion3d::coco_to_h36m(coco)
        };
        let screen: [_; N] = std::array::from_fn(|i| {
            if i == HEAD_PIVOT {
                [screen[8][0], screen[8][1], 0.0]
            } else {
                screen[i]
            }
        });
        let torso = self.next_torso_scale.take();
        let obs = Observation {
            ear_head: pose.ear_head,
            head_radius: head_radius_measurement(pose, points),
            frame,
            time_seconds,
            direct_depth: pose.has_metric_camera_coordinates(),
            torso_scale: torso.map(|v| v.0),
            torso_span: torso.map_or(1, |v| v.1),
            depth_hint: None,
            heading_hint: None,
            assume_upright_torso: !self.free_torso,
            points: std::array::from_fn(|i| {
                V::new(
                    pose.joints[if i == HEAD_PIVOT { 8 } else { i }].x as f64,
                    pose.joints[if i == HEAD_PIVOT { 8 } else { i }].y as f64,
                    pose.joints[if i == HEAD_PIVOT { 8 } else { i }].z as f64,
                )
            }),
            scores: std::array::from_fn(|i| {
                if i == HEAD_PIVOT {
                    return 0.0;
                }
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
                if p.iter().all(|v| v.is_finite())
                    && p[0] >= 0.0
                    && p[1] >= 0.0
                    && p[2] >= threshold
                {
                    p[2].clamp(0.0, 1.0) as f64
                } else {
                    0.0
                }
            }),
            size,
            focal: focal.unwrap_or(size.width as f64),
            assumed_focal: focal.is_none(),
            history_frames: pose.history_frames,
            hands: hands.to_vec(),
            head,
        };
        if obs.points.iter().any(|p| !p.iter().all(|x| x.is_finite()))
            || (obs.scores.iter().filter(|&&s| s > 0.0).count() < 4
                && !(is_upper(&obs) && obs.scores[11].min(obs.scores[14]) >= 0.5)
                && !(obs.pixel_scores.iter().filter(|&&s| s >= 0.3).count() >= 4
                    && obs.pixel_scores[11].min(obs.pixel_scores[14]) >= 0.3))
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
            if self.streaming && !self.subjects.is_empty() {
                self.subjects.clear();
                self.streaming_segment += 1;
            }
            self.subjects.push(Subject {
                calibration: Calibration::new(&obs),
                hand_shapes: BTreeMap::new(),
                depth_references: DepthReferences::default(),
                observations: Vec::new(),
                previous: None,
                previous_head: None,
            });
        }
        self.last_frame = Some(frame);
        self.last_bbox = Some(bbox);
        self.last_size = Some(size);
        let subject_segment = if self.streaming {
            self.streaming_segment
        } else {
            self.subjects.len() - 1
        };
        let subject = self.subjects.last_mut().unwrap();
        subject
            .calibration
            .update(&obs, f64::from(threshold.max(0.6)));
        let dimensions = subject.calibration.dimensions();
        let (solved, fitted_observation) = inverse_with_size(
            dimensions,
            &obs,
            subject.previous.as_ref(),
            &mut subject.depth_references,
        );
        subject.learn_hands(&fitted_observation, &solved);
        let head_solved = refine_head(
            head_dimensions(dimensions),
            &fitted_observation,
            &solved,
            subject.previous_head.as_ref(),
        );
        let parameters = subject.frame(&fitted_observation, &head_solved);
        let fitted = forward(head_dimensions(dimensions), &head_solved).0;
        let joints: Vec<_> = (0..MODEL_N)
            .map(|i| Joint3d {
                name: motion3d::NAMES[i],
                x: (fitted[i].x - solved.root.x) as f32,
                y: (fitted[i].y - solved.root.y) as f32,
                z: (fitted[i].z - solved.root.z) as f32,
                input_score: pose.joints[i].input_score,
            })
            .collect();
        let fitted_pose = Pose3d {
            ear_head: pose.ear_head,
            model: "skeletal_twin_fk",
            coordinate_space: "root_relative_camera",
            units: pose.units,
            history_frames: pose.history_frames,
            angles: motion3d::angles3d(&joints, threshold),
            joints,
        };
        subject.previous = Some(solved);
        subject.previous_head = Some(head_solved);
        if !self.streaming {
            subject.observations.push(obs);
        }
        Some(TwinPreview {
            subject_segment,
            provisional: true,
            person: subject.parameters(),
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
        writer.write_all(b"{\"schema_version\":2,\"quaternion_convention\":\"xyzw_active_parent_local\",\"coordinate_axes\":\"camera_x_right_y_down_z_forward\",\"recording\":")?;
        serde_json::to_writer(&mut writer, &self.recording)?;
        writer.write_all(b",\"joint_names\":")?;
        serde_json::to_writer(&mut writer, &ORDER.map(joint_name))?;
        writer.write_all(b",\"parents\":")?;
        serde_json::to_writer(
            &mut writer,
            &ORDER
                .iter()
                .map(|&i| {
                    if i == 0 {
                        None
                    } else {
                        ORDER.iter().position(|&j| j == PARENT[i])
                    }
                })
                .collect::<Vec<_>>(),
        )?;
        writer.write_all(b",\"motion_processing\":")?;
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "joint_continuity": !self.no_stabilization,
                "sequence_ik": !self.no_stabilization,
                "sequence_window_frames": 5,
                "adaptive_smoothing": !self.no_stabilization && !self.no_smoothing,
                "maximum_image_filter_offset_px": 2.5,
                "short_prediction_limit_seconds": 0.16,
            }),
        )?;
        writer.write_all(b",\"subjects\":[")?;
        for (index, subject) in self.subjects.iter().enumerate() {
            if index > 0 {
                writer.write_all(b",")?;
            }
            write!(writer, "{{\"subject_segment\":{index},\"person\":")?;
            serde_json::to_writer(&mut writer, &subject.parameters())?;
            writer.write_all(b",\"frames\":[")?;
            let dimensions = subject.calibration.dimensions();
            let mut fitted = fit_recording(dimensions, &subject.observations);
            if !self.no_stabilization {
                stabilize_sequence(dimensions, &mut fitted, !self.no_smoothing);
            }
            for (index, (pose, fitted_observation)) in fitted.iter().enumerate() {
                if index > 0 {
                    writer.write_all(b",")?;
                }
                let frame = subject.frame(fitted_observation, pose);
                ensure!(frame.fit_rmse.is_finite(), "Non-finite skeletal twin fit");
                serde_json::to_writer(&mut writer, &frame)?;
            }
            writer.write_all(b"]}")?;
        }
        writer.write_all(b"]}")?;
        writer.flush()?;
        Ok(())
    }
}

/// Shape is fixed before both passes. Depth is a latent trajectory; observed
/// image joints remain per-frame constraints during the second IK solve.
// Largest torso distance for which the fixed arm can reach a hand's image ray.
// This supplies an inequality, not a measured hand depth or an extended-arm pose.
#[cfg(test)]
fn reach_depth_upper_bound(thorax: V, offset: V, hand_ray: V, reach: f64) -> Option<f64> {
    if thorax.z <= 0.1 || reach <= 0.0 {
        return None;
    }
    let ray = hand_ray.try_normalize(1e-10)?;
    let torso_ray = thorax / thorax.z;
    let a = torso_ray - ray * torso_ray.dot(&ray);
    let b = offset - ray * offset.dot(&ray);
    let aa = a.norm_squared();
    let ab = a.dot(&b);
    let discriminant = ab * ab - aa * (b.norm_squared() - reach * reach);
    if aa < 1e-10 || discriminant <= 0.0 {
        return None;
    }
    let upper = (-ab + discriminant.sqrt()) / aa;
    (upper.is_finite() && upper > 0.1 && (torso_ray * upper + offset).dot(&ray) > 0.01)
        .then_some(upper)
}

#[cfg(test)]
fn hand_reach_depth_bounds(
    dimensions: Dimensions,
    observations: &[Observation],
    fitted: &[(Pose, Observation)],
) -> Vec<(usize, f64, f64)> {
    type HandSample = (u64, [f64; 2], f64);
    let mut history: [[Option<HandSample>; 2]; 2] = [[None; 2]; 2];
    let mut bounds = Vec::new();
    for (index, (obs, (pose, _))) in observations.iter().zip(fitted).enumerate() {
        let before = history;
        history = [[None; 2]; 2];
        let Some(camera) = pose.camera else { continue };
        let positions = forward(dimensions, pose).0;
        for hand in &obs.hands {
            let side = match hand.side.as_str() {
                "left" => 0,
                "right" => 1,
                _ => continue,
            };
            let pixel = hand.image_landmarks[0].map(f64::from);
            if hand.confidence < 0.65
                || !pixel.iter().all(|x| x.is_finite())
                || pixel[0] < 0.0
                || pixel[1] < 0.0
                || pixel[0] >= obs.size.width as f64
                || pixel[1] >= obs.size.height as f64
            {
                continue;
            }
            let current = (obs.frame, pixel, f64::from(hand.confidence));
            history[side] = [before[side][1], Some(current)];
            let [Some(a), Some(b)] = before[side] else {
                continue;
            };
            let span = (hand.image_landmarks[5][0] - hand.image_landmarks[17][0])
                .hypot(hand.image_landmarks[5][1] - hand.image_landmarks[17][1])
                as f64;
            let limit = (span * 2.0).max(40.0);
            if a.0 + 2 != obs.frame
                || b.0 + 1 != obs.frame
                || [a, b]
                    .iter()
                    .any(|h| (Vector2::from(h.1) - Vector2::from(pixel)).norm() > limit)
                || obs.pixel_scores[11].min(obs.pixel_scores[14]) < 0.8
                || !profile_depth_ambiguous(obs, &obs.points)
            {
                continue;
            }
            let shoulder = if side == 0 { 11 } else { 14 };
            let Some((projected, _)) = project(positions[shoulder], camera, obs) else {
                continue;
            };
            if (projected - Vector2::from(obs.pixels[shoulder])).norm() > 15.0 {
                continue;
            }
            let ray = V::new(
                (pixel[0] - obs.size.width as f64 / 2.0) / obs.focal,
                (pixel[1] - obs.size.height as f64 / 2.0) / obs.focal,
                1.0,
            );
            // Allow for uncertain wrist/shoulder localization before declaring
            // the current camera distance incompatible with this person's reach.
            let reach = dimensions.0[4] + dimensions.0[5] + dimensions.0[1] * 0.1;
            if let Some(upper) = reach_depth_upper_bound(
                positions[8] + camera,
                positions[shoulder] - positions[8],
                ray,
                reach,
            ) {
                bounds.push((index, upper, current.2.min(a.2).min(b.2).powi(2)));
            }
        }
    }
    bounds
}

fn fit_recording(dimensions: Dimensions, observations: &[Observation]) -> Vec<(Pose, Observation)> {
    let finish = |mut fitted: Vec<(Pose, Observation)>| {
        if observations.iter().all(|o| o.direct_depth) {
            complete_occluded_arms(&mut fitted);
        }
        let mut previous = None;
        for (pose, obs) in &mut fitted {
            *pose = refine_head(head_dimensions(dimensions), obs, pose, previous.as_ref());
            previous = Some(pose.clone());
        }
        fitted
    };
    let pass = |input: &[Observation]| {
        let mut previous = None;
        let mut references = DepthReferences::default();
        let mut fitted = Vec::with_capacity(input.len());
        for obs in input {
            let (pose, measurement) =
                inverse_with_size(dimensions, obs, previous.as_ref(), &mut references);
            previous = Some(pose.clone());
            fitted.push((pose, measurement));
        }
        fitted
    };
    let first = pass(observations);
    if observations.len() < 3
        || !observations
            .iter()
            .all(|o| o.direct_depth && o.assume_upright_torso && is_upper(o))
    {
        return finish(first);
    }
    let samples: Option<Vec<_>> = observations
        .iter()
        .zip(&first)
        .enumerate()
        .map(|(i, (obs, (pose, _)))| {
            let estimate = pose.camera?.z + pose.root.z;
            let confidence = if profile_depth_ambiguous(obs, &obs.points) {
                0.0
            } else {
                obs.scores[11]
                    .min(obs.scores[14])
                    .min(obs.pixel_scores[11])
                    .min(obs.pixel_scores[14])
                    .powi(2)
            };
            let scale = obs.torso_scale.and_then(|scale| {
                let reference = i.checked_sub(obs.torso_span)?;
                (obs.frame - observations[reference].frame == obs.torso_span as u64)
                    .then_some((reference, scale))
            });
            Some(crate::depth_path::Sample {
                time: obs.time_seconds,
                estimate,
                confidence,
                scale,
            })
        })
        .collect();
    let Some(path) = samples.and_then(|s| {
        #[cfg(test)]
        if let Ok(path) = std::env::var("LIFTING_EVAL_REACH_BOUNDS") {
            let bounds: Vec<(usize, f64, f64)> =
                serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
            return crate::depth_path::estimate_with_upper_bounds(&s, dimensions.0[1], &bounds);
        }
        #[cfg(test)]
        if std::env::var_os("LIFTING_EVAL_NATIVE_REACH").is_some() {
            let bounds = hand_reach_depth_bounds(dimensions, observations, &first);
            return crate::depth_path::estimate_with_upper_bounds(&s, dimensions.0[1], &bounds);
        }
        crate::depth_path::estimate(&s, dimensions.0[1])
    }) else {
        return finish(first);
    };
    let mut constrained = observations.to_vec();
    for (obs, depth) in constrained.iter_mut().zip(path) {
        obs.depth_hint = Some(depth);
    }
    let headings: Vec<_> = observations
        .iter()
        .zip(&first)
        .map(|(obs, (pose, _))| {
            let right = pose.orientation * V::x();
            crate::heading_path::Sample {
                time: obs.time_seconds,
                heading: (-right.z).atan2(right.x),
                confidence: obs.scores[11]
                    .min(obs.scores[14])
                    .min(obs.pixel_scores[11])
                    .min(obs.pixel_scores[14])
                    .powi(2),
            }
        })
        .collect();
    if let Some(path) = crate::heading_path::estimate(&headings) {
        for (obs, heading) in constrained.iter_mut().zip(path) {
            obs.heading_hint = Some(heading);
        }
    }
    finish(pass(&constrained))
}

/// Fill only unobserved outgoing arm rotations, in parent/body coordinates.
/// Any observed descendant keeps its fitted pose exactly, including a visible
/// wrist whose elbow is hidden. No measured arm samples are averaged.
fn complete_occluded_arms(fitted: &mut [(Pose, Observation)]) {
    for node in [11, 12, 14, 15] {
        let anchors: Vec<_> = fitted
            .iter()
            .enumerate()
            .filter_map(|(frame, (_, obs))| {
                ((node + 1)..N)
                    .any(|i| {
                        descendant(i, node) && (obs.scores[i] > 0.0 || obs.pixel_scores[i] > 0.0)
                    })
                    .then_some(frame)
            })
            .collect();
        for endpoints in anchors.windows(2) {
            let (a, b) = (endpoints[0], endpoints[1]);
            let start = fitted[a].0.local[node];
            let end = fitted[b].0.local[node];
            let t0 = fitted[a].1.time_seconds;
            let duration = fitted[b].1.time_seconds - t0;
            for i in a + 1..b {
                let t = if duration > 1e-6 {
                    (fitted[i].1.time_seconds - t0) / duration
                } else {
                    (i - a) as f64 / (b - a) as f64
                };
                fitted[i].0.local[node] = start.slerp(&end, t.clamp(0.0, 1.0));
            }
        }
    }
}

/// Bound any temporal edit against the original per-frame image fit. The model
/// remains a rigid FK tree; no joint positions or segment lengths are averaged.
fn image_edit_allowed(
    dimensions: Dimensions,
    obs: &Observation,
    original: &Pose,
    candidate: &Pose,
    budget: f64,
) -> bool {
    let (Some(a), Some(b)) = (original.camera, candidate.camera) else {
        return false;
    };
    let before = forward(dimensions, original).0;
    let after = forward(dimensions, candidate).0;
    (1..N).filter(|&i| obs.pixel_scores[i] >= 0.65).all(|i| {
        match (project(before[i], a, obs), project(after[i], b, obs)) {
            (Some((p, _)), Some((q, _))) => {
                (q - Vector2::from(obs.pixels[i])).norm()
                    <= (p - Vector2::from(obs.pixels[i])).norm() + budget
            }
            _ => false,
        }
    })
}
fn blend_pose(a: &Pose, b: &Pose, t: f64) -> Pose {
    let mut p = a.clone();
    p.orientation = unit(a.orientation.slerp(&b.orientation, t));
    p.local = std::array::from_fn(|i| unit(a.local[i].slerp(&b.local[i], t)));
    p.root = a.root * (1.0 - t) + b.root * t;
    p.camera = a.camera.zip(b.camera).map(|(x, y)| x * (1.0 - t) + y * t);
    p
}
fn bounded_temporal_edit(
    dimensions: Dimensions,
    obs: &Observation,
    original: &Pose,
    candidate: &Pose,
) -> Pose {
    for gain in [1.0, 0.5, 0.25, 0.125] {
        let mut p = blend_pose(original, candidate, gain);
        if obs.direct_depth {
            limit_bends(&mut p);
        }
        if obs.assume_upright_torso {
            constrain_observed_torso(&mut p, obs);
        }
        align_upper_anchor(dimensions, &mut p, obs);
        if image_edit_allowed(dimensions, obs, original, &p, 2.5) {
            return p;
        }
    }
    original.clone()
}
fn short_interval(a: &Observation, b: &Observation) -> bool {
    let dt = b.time_seconds - a.time_seconds;
    dt > 0.0 && dt <= 0.16 && b.frame > a.frame && b.frame - a.frame <= 5
}
fn rotational_gain(a: Q, b: Q, dt: f64) -> f64 {
    let speed = a.angle_to(&b) / dt.max(1e-3);
    0.40 / (1.0 + (speed / 2.0).powi(2))
}
/// Two bounded block-coordinate sweeps over a five-frame moving window. Every
/// frame is re-solved against its own observations plus neighboring FK poses;
/// this is a sequence objective, not just a filter applied to exported pixels.
fn stabilize_sequence(dimensions: Dimensions, fitted: &mut [(Pose, Observation)], smooth: bool) {
    if fitted.len() < 3 {
        return;
    }
    let original = fitted.to_vec();
    for _ in 0..2 {
        let snapshot = fitted.to_vec();
        for i in 1..snapshot.len() - 1 {
            if !short_interval(&snapshot[i - 1].1, &snapshot[i].1)
                || !short_interval(&snapshot[i].1, &snapshot[i + 1].1)
            {
                continue;
            }
            let obs = &snapshot[i].1;
            let mut target = [V::zeros(); N];
            let mut total = 0.0;
            let anchor = if is_upper(obs) { 8 } else { 0 };
            for radius in 1..=2 {
                if i < radius || i + radius >= snapshot.len() {
                    continue;
                }
                if !(i - radius..i + radius)
                    .all(|j| short_interval(&snapshot[j].1, &snapshot[j + 1].1))
                {
                    continue;
                }
                let a = &snapshot[i - radius];
                let b = &snapshot[i + radius];
                let t =
                    (obs.time_seconds - a.1.time_seconds) / (b.1.time_seconds - a.1.time_seconds);
                let pa = forward(dimensions, &a.0).0;
                let pb = forward(dimensions, &b.0).0;
                let weight = if radius == 1 { 0.75 } else { 0.25 };
                for j in 0..N {
                    target[j] += ((pa[j] - pa[anchor]) * (1.0 - t)
                        + (pb[j] - pb[anchor]) * t
                        + obs.points[anchor])
                        * weight;
                }
                total += weight;
            }
            if total == 0.0 {
                continue;
            }
            for p in &mut target {
                *p /= total;
            }
            let candidate =
                inverse_sequence(dimensions, obs, Some(&snapshot[i - 1].0), Some(target));
            // Head refinement has a separate pivot and must not be discarded by
            // a body-only solve. Its final adaptive edit is applied below.
            let mut candidate = candidate;
            for node in [HEAD_PIVOT, 9, 10] {
                candidate.local[node] = snapshot[i].0.local[node];
            }
            fitted[i].0 = bounded_temporal_edit(dimensions, obs, &original[i].0, &candidate);
        }
    }
    complete_occluded_arms(fitted);
    if !smooth {
        return;
    }
    let snapshot = fitted.to_vec();
    for i in 1..snapshot.len() - 1 {
        let (a, obs, b) = (&snapshot[i - 1], &snapshot[i].1, &snapshot[i + 1]);
        if !short_interval(&a.1, obs) || !short_interval(obs, &b.1) {
            continue;
        }
        let duration = b.1.time_seconds - a.1.time_seconds;
        let t = (obs.time_seconds - a.1.time_seconds) / duration;
        let base = &snapshot[i].0;
        let mut candidate = base.clone();
        for node in 1..N {
            let gain = rotational_gain(a.0.local[node], b.0.local[node], duration);
            candidate.local[node] =
                unit(base.local[node].slerp(&a.0.local[node].slerp(&b.0.local[node], t), gain));
        }
        let gain = rotational_gain(a.0.orientation, b.0.orientation, duration);
        candidate.orientation = unit(
            base.orientation
                .slerp(&a.0.orientation.slerp(&b.0.orientation, t), gain),
        );
        if let (Some(ca), Some(cb), Some(current)) = (a.0.camera, b.0.camera, base.camera) {
            let pa = ca + a.0.root;
            let pb = cb + b.0.root;
            let speed = (pb - pa).norm() / duration / dimensions.0[1].max(0.01);
            let gain = 0.4 / (1.0 + (speed / 1.5).powi(2));
            let target = pa * (1.0 - t) + pb * t;
            candidate.camera =
                Some((current + base.root) * (1.0 - gain) + target * gain - base.root);
        }
        // All temporal passes together share the same per-frame error budget.
        fitted[i].0 = bounded_temporal_edit(dimensions, obs, &original[i].0, &candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sequence_stabilization_keeps_fixed_bones_and_bounds_visible_joint_edits() {
        let mut fitted = Vec::new();
        for i in 0..15 {
            let mut p = Pose {
                time_seconds: i as f64 / 30.0,
                camera: Some(V::new(0.0, 0.0, 2.5)),
                ..Pose::default()
            };
            p.local[11] = Q::from_axis_angle(&V::z_axis(), if i % 2 == 0 { 0.02 } else { -0.02 });
            let mut obs = observation(i, &p);
            obs.direct_depth = true;
            for j in 0..N {
                obs.pixels[j] = project(obs.points[j], p.camera.unwrap(), &obs)
                    .unwrap()
                    .0
                    .into();
            }
            fitted.push((p, obs));
        }
        let original = fitted.clone();
        let jitter = |frames: &[(Pose, Observation)]| {
            frames
                .windows(2)
                .map(|w| w[0].0.local[11].angle_to(&w[1].0.local[11]).powi(2))
                .sum::<f64>()
        };
        let before = jitter(&fitted);
        stabilize_sequence(dimensions(), &mut fitted, true);
        assert!(
            jitter(&fitted) < before * 0.8,
            "sequence failed to attenuate jitter"
        );
        for ((p, obs), (base, _)) in fitted.iter().zip(&original) {
            assert!(image_edit_allowed(dimensions(), obs, base, p, 2.5001));
            let xyz = forward(dimensions(), p).0;
            for j in 1..N {
                assert!(
                    ((xyz[j] - xyz[PARENT[j]]).norm() - dimensions().offsets()[j].norm()).abs()
                        < 1e-12
                );
                assert!((p.local[j].norm() - 1.0).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn temporal_filter_is_time_based_and_does_not_cross_long_gaps_or_erase_fast_motion() {
        assert!(
            rotational_gain(Q::identity(), Q::from_axis_angle(&V::y_axis(), 1.0), 0.066) < 0.01
        );
        assert!(rotational_gain(Q::identity(), Q::identity(), 0.066) > 0.3);
        let mut frames = Vec::new();
        for i in 0..3 {
            let mut p = Pose {
                time_seconds: i as f64,
                camera: Some(V::new(0.0, 0.0, 2.5)),
                ..Pose::default()
            };
            p.local[11] = Q::from_axis_angle(&V::z_axis(), i as f64 * 0.2);
            let mut obs = observation(i, &p);
            obs.time_seconds = i as f64;
            frames.push((p, obs));
        }
        let rotations: Vec<_> = frames.iter().map(|f| f.0.local).collect();
        stabilize_sequence(dimensions(), &mut frames, true);
        assert!(frames.iter().zip(rotations).all(|(p, q)| p.0.local == q));
    }

    #[test]
    fn observed_pixels_keep_free_motion_but_unmeasured_depth_has_relative_continuity() {
        let base = pose();
        let points = forward(dimensions(), &base).0;
        let mut obs = observation(1, &base);
        obs.direct_depth = true;
        obs.scores.fill(1.0);
        obs.scores[13] = 0.0;
        obs.pixel_scores.fill(0.0);
        obs.pixel_scores[13] = 0.9;
        let make = |temporal| FitContext {
            sequence_target: None,
            obs: &obs,
            camera: None,
            previous: Some(points),
            temporal,
            delta: 0.1,
            image_unit: 0.001,
        };
        let context = make(0.1);
        assert_eq!(context.temporal_weight(13), 0.0);
        assert!(context.unmeasured_depth_weight(13) > 0.0);
        let mut xy = points;
        xy[13].x += 0.2;
        xy[13].y += 0.1;
        assert_eq!(context.cost(&xy) - make(0.0).cost(&xy), 0.0);
        let mut z = points;
        z[13].z += 0.1;
        let penalty = context.cost(&z) - make(0.0).cost(&z);
        assert!(penalty > 0.0);
        let shifted = z.map(|p| p + V::new(3.0, 4.0, 5.0));
        assert!(((context.cost(&shifted) - make(0.0).cost(&shifted)) - penalty).abs() < 1e-12);
        obs.scores[13] = 0.8;
        assert_eq!(
            make_context_for_depth_test(&obs, points).unmeasured_depth_weight(13),
            0.0
        );
        obs.scores[13] = 0.0;
        obs.pixel_scores[13] = 0.0;
        assert_eq!(
            make_context_for_depth_test(&obs, points).unmeasured_depth_weight(13),
            0.0
        );
    }
    fn make_context_for_depth_test(obs: &Observation, points: [V; N]) -> FitContext<'_> {
        FitContext {
            sequence_target: None,
            obs,
            camera: None,
            previous: Some(points),
            temporal: 0.1,
            delta: 0.1,
            image_unit: 0.001,
        }
    }
    #[test]
    fn relative_arm_depth_jacobian_matches_root_torso_and_arm_rotations() {
        let base = pose();
        let (points, world) = forward(dimensions(), &base);
        for node in [0, 7, 8, 11, 12] {
            let analytic = relative_depth_jacobian(&points, 13, node);
            for axis in 0..3 {
                let mut delta = V::zeros();
                delta[axis] = 1e-6;
                let mut moved = base.clone();
                if node == 0 {
                    moved.orientation = unit(Q::from_scaled_axis(delta) * base.orientation);
                } else {
                    moved.local[node] = unit(
                        world[PARENT[node]].inverse() * Q::from_scaled_axis(delta) * world[node],
                    );
                }
                let after = forward(dimensions(), &moved).0;
                let numerical = ((after[13].z - after[8].z) - (points[13].z - points[8].z)) / 1e-6;
                assert!(
                    (numerical - analytic[axis]).abs() < 1e-6,
                    "node {node} axis {axis}"
                );
            }
        }
    }
    #[test]
    fn ear_based_head_uses_center_position_and_carries_it_when_unobserved() {
        let base = Pose {
            camera: Some(V::new(0.0, 0.0, 2.4)),
            ..Pose::default()
        };
        let mut target = base.clone();
        target.local[HEAD_PIVOT] = Q::from_euler_angles(0.0, 0.0, 0.25);
        let mut obs = observation(0, &target);
        obs.ear_head = true;
        obs.direct_depth = true;
        obs.scores[..8].fill(0.0);
        obs.pixel_scores[..8].fill(0.0);
        obs.pixels = obs
            .points
            .map(|p| project(p, base.camera.unwrap(), &obs).unwrap().0.into());
        // A conflicting nose pixel must not displace the observed skull center.
        obs.pixels[9][0] += 100.0;
        let fitted = refine_head(dimensions(), &obs, &base, None);
        let positions = forward(dimensions(), &fitted).0;
        let pixel = project(positions[10], base.camera.unwrap(), &obs)
            .unwrap()
            .0;
        assert!((pixel - Vector2::from(obs.pixels[10])).norm() < 1.0);
        assert_eq!(fitted.camera, base.camera);
        assert_eq!(fitted.root, base.root);
        obs.time_seconds = 1.0 / 30.0;
        obs.scores[10] = 0.0;
        obs.pixel_scores[10] = 0.0;
        let missing = refine_head(dimensions(), &obs, &base, Some(&fitted));
        assert_eq!(missing.local[HEAD_PIVOT], fitted.local[HEAD_PIVOT]);
        assert_eq!(missing.local[9], fitted.local[9]);
    }
    #[test]
    fn head_pivot_moves_head_without_moving_shoulders_or_arms() {
        let base = pose();
        let before = forward(dimensions(), &base).0;
        let mut turned = base.clone();
        turned.local[HEAD_PIVOT] = Q::from_euler_angles(0.35, 0.2, -0.25);
        let after = forward(dimensions(), &turned).0;
        for i in 0..N {
            if ![9, 10].contains(&i) {
                assert!((before[i] - after[i]).norm() < 1e-12);
            }
            if i > 0 {
                assert!(
                    ((after[i] - after[PARENT[i]]).norm() - dimensions().offsets()[i].norm()).abs()
                        < 1e-12
                );
            }
        }
        assert!((after[9] - before[9]).norm() > 0.02);
        assert!((after[10] - before[10]).norm() > 0.02);
    }
    #[test]
    fn head_direction_jacobian_matches_both_head_rotations() {
        let mut base = pose();
        base.local[HEAD_PIVOT] = Q::from_euler_angles(0.2, -0.1, 0.3);
        base.local[9] = Q::from_euler_angles(-0.3, 0.4, 0.1);
        let (points, world) = forward(dimensions(), &base);
        let bone = points[10] - points[9];
        let jacobian = Matrix3::new(
            0.0, bone.z, -bone.y, -bone.z, 0.0, bone.x, bone.y, -bone.x, 0.0,
        );
        for node in [9, HEAD_PIVOT] {
            for axis in 0..3 {
                let mut delta = V::zeros();
                delta[axis] = 1e-6;
                let mut moved = base.clone();
                moved.local[node] =
                    unit(world[PARENT[node]].inverse() * Q::from_scaled_axis(delta) * world[node]);
                let after = forward(dimensions(), &moved).0;
                let numerical = ((after[10] - after[9]) - bone) / 1e-6;
                assert!((numerical - jacobian.column(axis)).norm() < 1e-6);
            }
        }
    }
    #[test]
    fn head_refinement_preserves_body_and_carries_only_unobserved_head_motion() {
        let base = Pose {
            camera: Some(V::new(0.1, -0.1, 2.4)),
            ..Pose::default()
        };
        let mut target = base.clone();
        target.local[HEAD_PIVOT] = Q::from_euler_angles(0.4, 0.1, -0.25);
        target.local[9] = Q::from_euler_angles(0.0, 0.2, 0.0);
        let mut obs = observation(0, &target);
        obs.direct_depth = true;
        obs.scores[..8].fill(0.0);
        obs.pixel_scores[..8].fill(0.0);
        obs.pixels = obs
            .points
            .map(|p| project(p, base.camera.unwrap(), &obs).unwrap().0.into());
        let fitted = refine_head(dimensions(), &obs, &base, None);
        assert_eq!(fitted.root, base.root);
        assert_eq!(fitted.orientation, base.orientation);
        assert_eq!(fitted.camera, base.camera);
        let before = forward(dimensions(), &base).0;
        let after = forward(dimensions(), &fitted).0;
        for i in 0..N {
            if ![9, 10].contains(&i) {
                assert_eq!(before[i], after[i]);
            }
        }
        assert!((after[9] - obs.points[9]).norm() < (before[9] - obs.points[9]).norm() * 0.1);
        for i in [9, 10] {
            obs.scores[i] = 0.0;
            obs.pixel_scores[i] = 0.0;
        }
        obs.time_seconds = 1.0 / 30.0;
        let missing = refine_head(dimensions(), &obs, &base, Some(&fitted));
        assert_eq!(missing.local[HEAD_PIVOT], fitted.local[HEAD_PIVOT]);
        assert_eq!(missing.local[9], fitted.local[9]);
    }
    #[test]
    fn pixel_error_includes_a_wrist_without_native_depth() {
        let base = Pose {
            camera: Some(V::new(0.0, 0.0, 3.0)),
            ..Pose::default()
        };
        let mut obs = observation(0, &base);
        obs.pixels = obs
            .points
            .map(|p| project(p, base.camera.unwrap(), &obs).unwrap().0.into());
        obs.scores[13] = 0.0;
        obs.pixels[13][0] += 50.0;
        let frame = frame_parameters(dimensions(), &obs, &base);
        assert!(
            (frame.reprojection_rmse_px.unwrap() - 50.0 / (MODEL_N as f64).sqrt()).abs() < 1e-9
        );
        obs.pixel_scores[13] = 0.0;
        assert!(
            frame_parameters(dimensions(), &obs, &base)
                .reprojection_rmse_px
                .unwrap()
                < 1e-9
        );
    }
    #[test]
    fn reach_bound_is_a_one_sided_geometric_limit() {
        let thorax = V::new(0.0, 0.0, 2.0);
        let offset = V::new(0.2, 0.0, 0.0);
        let ray = V::new(-0.3, 0.0, 1.0);
        let upper = reach_depth_upper_bound(thorax, offset, ray, 0.5).unwrap();
        let unit_ray = ray.normalize();
        let distance = |z: f64| {
            let p = V::new(0.2, 0.0, z);
            (p - unit_ray * p.dot(&unit_ray)).norm()
        };
        assert!((distance(upper) - 0.5).abs() < 1e-10);
        assert!(distance(upper + 0.1) > 0.5);
        assert!(distance(upper - 0.1) < 0.5);
        assert!(reach_depth_upper_bound(thorax, offset, V::z(), 0.5).is_none());
    }
    #[test]
    fn reach_bounds_require_three_consecutive_independent_hands() {
        let mut base = Pose {
            orientation: Q::from_axis_angle(&V::y_axis(), std::f64::consts::FRAC_PI_2),
            camera: Some(V::new(0.0, 0.0, 2.4)),
            ..Pose::default()
        };
        let mut observations = Vec::new();
        let mut fitted = Vec::new();
        for n in 0..3 {
            base.time_seconds = n as f64 / 30.0;
            let mut obs = observation(n, &base);
            obs.direct_depth = true;
            obs.scores[..8].fill(0.0);
            obs.pixel_scores[..8].fill(0.0);
            obs.pixels = obs
                .points
                .map(|p| project(p, base.camera.unwrap(), &obs).unwrap().0.into());
            obs.hands = vec![HandPose {
                side: "right".into(),
                confidence: 0.9,
                image_landmarks: [[100.0, 140.0]; 21],
                world_landmarks: [[0.0; 3]; 21],
            }];
            observations.push(obs.clone());
            fitted.push((base.clone(), obs));
        }
        let bounds = hand_reach_depth_bounds(dimensions(), &observations, &fitted);
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0].0, 2);
        observations[1].hands[0].confidence = 0.64;
        assert!(hand_reach_depth_bounds(dimensions(), &observations, &fitted).is_empty());
        observations[1].hands.clear();
        assert!(hand_reach_depth_bounds(dimensions(), &observations, &fitted).is_empty());
    }
    fn dimensions() -> Dimensions {
        Dimensions([0.24, 0.4, 0.42, 0.43, 0.28, 0.25, 0.12, 0.18, 0.13, 0.08])
    }
    fn observation(frame: u64, pose: &Pose) -> Observation {
        let positions = forward(dimensions(), pose).0;
        Observation {
            ear_head: false,
            head_radius: None,
            frame,
            time_seconds: frame as f64 / 30.0,
            direct_depth: false,
            torso_scale: None,
            torso_span: 1,
            depth_hint: None,
            heading_hint: None,
            assume_upright_torso: false,
            points: positions,
            scores: std::array::from_fn(|i| if i == HEAD_PIVOT { 0.0 } else { 0.9 }),
            pixels: positions.map(|p| [320.0 + p.x * 500.0, 240.0 + p.y * 500.0]),
            pixel_scores: std::array::from_fn(|i| if i == HEAD_PIVOT { 0.0 } else { 0.9 }),
            size: Size::new(640, 480),
            focal: 800.0,
            assumed_focal: false,
            history_frames: 27,
            hands: Vec::new(),
            head: None,
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
    fn arm_only_refinement_preserves_body_camera_and_unobserved_arm() -> Result<()> {
        let mut base = Pose::default();
        base.camera = Some(V::new(0.1, -0.1, 2.4));
        let mut target = base.clone();
        target.local[11] = Q::from_euler_angles(0.2, 0.1, -0.8);
        target.local[12] = Q::from_euler_angles(0.0, 0.5, 0.0);
        let mut obs = observation(0, &target);
        obs.direct_depth = true;
        obs.scores[..8].fill(0.0);
        obs.pixel_scores[..8].fill(0.0);
        for i in [15, 16] {
            obs.scores[i] = 0.0;
            obs.pixel_scores[i] = 0.0;
        }
        for i in 0..N {
            obs.pixels[i] = project(obs.points[i], base.camera.unwrap(), &obs)
                .unwrap()
                .0
                .into();
        }
        let fitted = refine_fixed_body_arms(dimensions(), &obs, &base);
        assert_eq!(fitted.root, base.root);
        assert_eq!(fitted.orientation, base.orientation);
        assert_eq!(fitted.camera, base.camera);
        for i in 0..N {
            if ![11, 12].contains(&i) {
                assert!((fitted.local[i].inverse() * base.local[i]).angle() < 1e-12);
            }
        }
        let before = forward(dimensions(), &base).0;
        let after = forward(dimensions(), &fitted).0;
        assert!((after[13] - obs.points[13]).norm() < (before[13] - obs.points[13]).norm() * 0.1);
        Ok(())
    }
    #[test]
    fn reliable_arm_pixels_resist_a_noisy_native_world_prediction() {
        let mut truth = Pose {
            orientation: Q::from_axis_angle(&V::y_axis(), 0.4),
            ..Pose::default()
        };
        truth.local[11] = Q::from_euler_angles(0.2, 0.0, -0.8);
        truth.local[12] = Q::from_euler_angles(0.0, 0.8, 0.0);
        let mut obs = observation(0, &truth);
        obs.direct_depth = true;
        obs.assume_upright_torso = true;
        obs.heading_hint = Some(0.4);
        obs.scores[..8].fill(0.0);
        obs.pixel_scores[..8].fill(0.0);
        let camera = V::new(0.12, -0.08, 2.4);
        obs.depth_hint = Some(camera.z + obs.points[8].z);
        for i in 0..N {
            obs.pixels[i] = project(obs.points[i], camera, &obs).unwrap().0.into();
        }
        for i in [12, 13] {
            obs.points[i].x += 0.1;
            obs.points[i].z += 0.1;
        }
        let fit = inverse(dimensions(), &obs, None);
        let positions = forward(dimensions(), &fit).0;
        for i in [12, 13] {
            let pixel = project(positions[i], fit.camera.unwrap(), &obs).unwrap().0;
            assert!(
                (pixel - Vector2::from(obs.pixels[i])).norm() < 8.0,
                "Joint {i} lost its visible position"
            );
        }
    }
    #[test]
    fn head_size_calibrates_only_when_face_and_shoulders_show_their_width() {
        let positions = forward(dimensions(), &Pose::default()).0;
        let mut body = Pose3d {
            ear_head: true,
            model: "MediaPipe-Pose",
            coordinate_space: "root_relative_camera",
            units: "estimated_meters",
            history_frames: 1,
            angles: Vec::new(),
            joints: positions
                .iter()
                .take(MODEL_N)
                .enumerate()
                .map(|(i, p)| Joint3d {
                    name: motion3d::NAMES[i],
                    x: p.x as f32,
                    y: p.y as f32,
                    z: p.z as f32,
                    input_score: 0.95,
                })
                .collect(),
        };
        let mut points = vec![
            Keypoint {
                name: "test",
                x: 320.0,
                y: 200.0,
                score: 0.95
            };
            133
        ];
        points[3].x = 290.0;
        points[4].x = 350.0;
        points[5].x = 400.0;
        points[6].x = 240.0;
        assert!(head_radius_measurement(&body, &points).is_some());
        points[0].x = 380.0;
        assert!(head_radius_measurement(&body, &points).is_none());
        points[0].x = 320.0;
        body.joints[11].x = 0.0;
        body.joints[14].x = 0.0;
        body.joints[11].z = 0.16;
        body.joints[14].z = -0.16;
        assert!(head_radius_measurement(&body, &points).is_none());
    }
    #[test]
    fn constrained_torso_heading_still_fits_observed_arm_motion() {
        let mut truth = pose();
        truth.orientation = Q::from_axis_angle(&V::y_axis(), 1.3);
        truth.local[7] = Q::identity();
        truth.local[8] = Q::identity();
        let mut obs = observation(0, &truth);
        obs.direct_depth = true;
        obs.assume_upright_torso = true;
        obs.scores[..8].fill(0.0);
        obs.pixel_scores[..8].fill(0.0);
        let camera = V::new(0.12, -0.08, 2.4);
        for i in 0..N {
            obs.pixels[i] = project(obs.points[i], camera, &obs).unwrap().0.into();
        }
        let spin = Q::from_axis_angle(&V::y_axis(), 0.5);
        for i in [11, 14] {
            obs.points[i] = obs.points[8] + spin * (obs.points[i] - obs.points[8]);
        }
        obs.heading_hint = Some(1.3);
        let fitted = inverse(dimensions(), &obs, None);
        let right = fitted.orientation * V::x();
        assert!(((-right.z).atan2(right.x) - 1.3).abs() < 1e-8);
        let positions = forward(dimensions(), &fitted).0;
        for i in [13, 16] {
            let pixel = project(positions[i], fitted.camera.unwrap(), &obs)
                .unwrap()
                .0;
            assert!((pixel - Vector2::from(obs.pixels[i])).norm() < 15.0);
        }
    }
    #[test]
    fn upright_spin_tracks_a_full_turn_with_only_shoulders_during_the_back_view() {
        let mut previous = None;
        let mut last = None;
        let mut unwrapped = 0.0;
        for i in 0..=72 {
            let angle = -(i as f64) * std::f64::consts::TAU / 72.0;
            let truth = Pose {
                orientation: Q::from_axis_angle(&V::y_axis(), angle),
                ..Pose::default()
            };
            let mut obs = observation(i, &truth);
            obs.direct_depth = true;
            obs.assume_upright_torso = true;
            obs.scores[..8].fill(0.0);
            obs.pixel_scores[..8].fill(0.0);
            if (11..62).contains(&i) {
                obs.scores.fill(0.0);
                obs.pixel_scores.fill(0.0);
                for j in [8, 11, 14] {
                    obs.scores[j] = 0.9;
                    obs.pixel_scores[j] = 0.9;
                }
            }
            for j in 0..N {
                obs.pixels[j] = project(obs.points[j], V::new(0.12, -0.08, 2.4), &obs)
                    .unwrap()
                    .0
                    .into();
            }
            let fitted = inverse(dimensions(), &obs, previous.as_ref());
            let right = fitted.orientation * V::x();
            let heading = (-right.z).atan2(right.x);
            assert!(
                (heading - angle).sin().atan2((heading - angle).cos()).abs() < 0.12,
                "Frame {i}: lost heading"
            );
            if let Some(old) = last {
                let delta: f64 = heading - old;
                unwrapped += delta.sin().atan2(delta.cos());
            }
            last = Some(heading);
            assert!((fitted.orientation * V::y() - V::y()).norm() < 1e-10);
            assert!(
                fitted.camera.is_some(),
                "Lost camera estimate at profile frame {i}"
            );
            previous = Some(fitted);
        }
        assert!((unwrapped + std::f64::consts::TAU).abs() < 0.15);
    }
    #[test]
    fn shoulder_only_observations_are_kept_without_face_or_hand_landmarks() {
        let mut source = observation(0, &Pose::default());
        source.scores.fill(0.0);
        for i in [8, 11, 14] {
            source.scores[i] = 0.9;
        }
        let pose = Pose3d {
            ear_head: false,
            model: "MediaPipe-Pose",
            coordinate_space: "root_relative_camera",
            units: "estimated_meters",
            history_frames: 1,
            joints: (0..MODEL_N)
                .map(|i| Joint3d {
                    name: motion3d::NAMES[i],
                    x: source.points[i].x as f32,
                    y: source.points[i].y as f32,
                    z: source.points[i].z as f32,
                    input_score: source.scores[i] as f32,
                })
                .collect(),
            angles: Vec::new(),
        };
        let mut points = vec![
            Keypoint {
                name: "unseen",
                x: -1.0,
                y: -1.0,
                score: 0.0
            };
            17
        ];
        for (c, h) in [(5, 11), (6, 14)] {
            points[c] = Keypoint {
                name: "shoulder",
                x: source.pixels[h][0] as f32,
                y: source.pixels[h][1] as f32,
                score: 0.9,
            };
        }
        points[7] = Keypoint {
            name: "weak_elbow",
            x: 100.0,
            y: 100.0,
            score: 0.1,
        };
        let mut recorder = TwinRecorder::default();
        assert!(
            recorder
                .observe(
                    0,
                    0.0,
                    &pose,
                    &points,
                    &[],
                    None,
                    Bbox {
                        x: 0.0,
                        y: 0.0,
                        width: 640.0,
                        height: 480.0
                    },
                    source.size,
                    0.3,
                    Some(source.focal)
                )
                .is_some()
        );
        assert_eq!(recorder.subjects[0].observations[0].pixel_scores[12], 0.0);
    }
    #[test]
    fn streaming_matches_recording_fit_without_retaining_frames() {
        let source = observation(0, &Pose::default());
        let pose = Pose3d {
            ear_head: false,
            model: "MediaPipe-Pose",
            coordinate_space: "root_relative_camera",
            units: "estimated_meters",
            history_frames: 1,
            angles: Vec::new(),
            joints: (0..MODEL_N)
                .map(|i| Joint3d {
                    name: motion3d::NAMES[i],
                    x: source.points[i].x as f32,
                    y: source.points[i].y as f32,
                    z: source.points[i].z as f32,
                    input_score: if [9, 10, 14].contains(&i) { 0.9 } else { 0.0 },
                })
                .collect(),
        };
        let mut points = vec![
            Keypoint {
                name: "joint",
                x: 320.0,
                y: 240.0,
                score: 0.0
            };
            17
        ];
        for (c, h) in [(5, 11), (6, 14), (7, 12), (8, 15), (9, 13), (10, 16)] {
            points[c].x = source.pixels[h][0] as f32;
            points[c].y = source.pixels[h][1] as f32;
            points[c].score = 0.65;
        }
        let mut recording = TwinRecorder::default();
        let mut streaming = TwinRecorder::streaming();
        let bbox = Bbox {
            x: 0.0,
            y: 0.0,
            width: 640.0,
            height: 480.0,
        };
        for frame in 0..8 {
            let a = recording
                .observe(
                    frame,
                    frame as f64 / 30.0,
                    &pose,
                    &points,
                    &[],
                    None,
                    bbox,
                    source.size,
                    0.3,
                    Some(source.focal),
                )
                .unwrap();
            let b = streaming
                .observe(
                    frame,
                    frame as f64 / 30.0,
                    &pose,
                    &points,
                    &[],
                    None,
                    bbox,
                    source.size,
                    0.3,
                    Some(source.focal),
                )
                .unwrap();
            assert_eq!(
                serde_json::to_value(&a).unwrap(),
                serde_json::to_value(&b).unwrap()
            );
            for (a, b) in a.fitted_pose.joints.iter().zip(&b.fitted_pose.joints) {
                assert_eq!(
                    [a.x, a.y, a.z, a.input_score],
                    [b.x, b.y, b.z, b.input_score]
                );
            }
            assert!(streaming.subjects[0].observations.is_empty());
        }
        assert_eq!(recording.subjects[0].observations.len(), 8);
        assert_eq!(
            recording.subjects[0].calibration.frames,
            streaming.subjects[0].calibration.frames
        );
        let next = streaming
            .observe(
                100,
                4.0,
                &pose,
                &points,
                &[],
                None,
                bbox,
                source.size,
                0.3,
                Some(source.focal),
            )
            .unwrap();
        assert_eq!(next.subject_segment(), 1);
        assert_eq!(streaming.subjects.len(), 1);
        assert!(streaming.subjects[0].observations.is_empty());
    }
    #[test]
    fn visible_body_is_retained_when_native_joint_depth_has_been_rejected() {
        let source = observation(0, &Pose::default());
        let pose = Pose3d {
            ear_head: false,
            model: "MediaPipe-Pose",
            coordinate_space: "root_relative_camera",
            units: "estimated_meters",
            history_frames: 1,
            angles: Vec::new(),
            joints: (0..MODEL_N)
                .map(|i| Joint3d {
                    name: motion3d::NAMES[i],
                    x: source.points[i].x as f32,
                    y: source.points[i].y as f32,
                    z: source.points[i].z as f32,
                    input_score: if [9, 10, 14].contains(&i) { 0.9 } else { 0.0 },
                })
                .collect(),
        };
        let mut points = vec![
            Keypoint {
                name: "joint",
                x: 320.0,
                y: 240.0,
                score: 0.0
            };
            17
        ];
        for (c, h) in [(5, 11), (6, 14), (7, 12), (8, 15), (9, 13), (10, 16)] {
            points[c].x = source.pixels[h][0] as f32;
            points[c].y = source.pixels[h][1] as f32;
            points[c].score = 0.65;
        }
        let mut recorder = TwinRecorder::default();
        assert!(
            recorder
                .observe(
                    0,
                    0.0,
                    &pose,
                    &points,
                    &[],
                    None,
                    Bbox {
                        x: 0.0,
                        y: 0.0,
                        width: 640.0,
                        height: 480.0
                    },
                    source.size,
                    0.3,
                    Some(source.focal)
                )
                .is_some()
        );
        let obs = &recorder.subjects[0].observations[0];
        assert_eq!(obs.scores[12], 0.0);
        assert!(obs.pixel_scores[12] > 0.6);
    }
    #[test]
    fn upright_torso_keeps_spine_vertical_without_moving_upper_body_observations() {
        let dimensions = dimensions();
        let mut p = pose();
        p.local[7] = Q::from_euler_angles(0.25, 0.1, -0.15);
        p.local[8] = Q::from_euler_angles(-0.1, 0.2, 0.12);
        let before = forward(dimensions, &p).0;
        constrain_upright_torso(&mut p, true);
        let chest = forward(dimensions, &p).0[8];
        p.root += before[8] - chest;
        let after = forward(dimensions, &p).0;
        assert!((p.orientation * V::y() - V::y()).norm() < 1e-12);
        for i in [7, 8] {
            assert!((after[i] - after[0]).x.abs() < 1e-12);
            assert!((after[i] - after[0]).z.abs() < 1e-12);
        }
        for i in 8..N {
            assert!(
                (after[i] - before[i]).norm() < 1e-12,
                "Upper joint {i} moved"
            );
        }
        for i in 1..N {
            assert!(
                ((after[i] - after[PARENT[i]]).norm() - dimensions.offsets()[i].norm()).abs()
                    < 1e-12
            );
        }
    }
    #[test]
    fn upright_constraint_survives_ik_updates_and_a_tilted_previous_pose() {
        let mut obs = observation(1, &pose());
        obs.scores[..8].fill(0.0);
        obs.pixel_scores[..8].fill(0.0);
        obs.assume_upright_torso = true;
        let fit = inverse(dimensions(), &obs, Some(&pose()));
        let positions = forward(dimensions(), &fit).0;
        assert!((fit.orientation * V::y() - V::y()).norm() < 1e-12);
        assert!(fit.local[7].angle() < 1e-12);
        assert!((positions[8] - positions[0]).cross(&V::y()).norm() < 1e-12);
        assert!((positions[8] - obs.points[8]).norm() < 1e-12);
    }
    #[test]
    fn changing_the_unseen_hip_origin_does_not_move_a_profile_twin() {
        let mut truth = Pose::default();
        truth.orientation = Q::from_axis_angle(&V::y_axis(), std::f64::consts::FRAC_PI_2);
        let mut obs = observation(0, &truth);
        obs.direct_depth = true;
        obs.assume_upright_torso = true;
        obs.scores[..8].fill(0.0);
        obs.pixel_scores[..8].fill(0.0);
        for i in 0..N {
            obs.pixels[i] = project(obs.points[i], V::new(0.1, -0.08, 2.4), &obs)
                .unwrap()
                .0
                .into();
        }
        let first = inverse(dimensions(), &obs, None);
        let first_root = first.root + first.camera.unwrap();
        obs.time_seconds = 1.0 / 30.0;
        for p in &mut obs.points {
            *p += V::new(0.05, 0.03, 0.2);
        }
        let second = inverse(dimensions(), &obs, Some(&first));
        assert!((second.root + second.camera.unwrap() - first_root).norm() < 1e-5);
    }
    #[test]
    fn hidden_arm_keeps_articulation_when_the_visible_arm_changes_and_reacquires() {
        let mut truth = Pose::default();
        truth.local[14] = Q::from_euler_angles(0.2, 0.4, -0.6);
        truth.local[15] = Q::from_euler_angles(0.0, 0.0, -0.8);
        let make_observation = |frame, truth: &Pose| {
            let mut obs = observation(frame, truth);
            obs.direct_depth = true;
            obs.assume_upright_torso = true;
            obs.scores[..8].fill(0.0);
            obs.pixel_scores[..8].fill(0.0);
            for i in 0..N {
                obs.pixels[i] = project(obs.points[i], V::new(0.0, 0.0, 2.4), &obs)
                    .unwrap()
                    .0
                    .into();
            }
            obs
        };
        let before = inverse(dimensions(), &make_observation(0, &truth), None);
        truth.local[11] = Q::from_euler_angles(0.0, 0.0, 1.2);
        let mut hidden = make_observation(1, &truth);
        for (hidden_index, visible_index) in [(15, 12), (16, 13)] {
            hidden.points[hidden_index] = hidden.points[visible_index];
            hidden.pixels[hidden_index] = hidden.pixels[visible_index];
            hidden.scores[hidden_index] = 0.0;
            hidden.pixel_scores[hidden_index] = 0.0;
        }
        let carried = inverse(dimensions(), &hidden, Some(&before));
        for node in [14, 15] {
            assert!((before.local[node].inverse() * carried.local[node]).angle() < 0.08);
        }
        truth.local[14] = Q::from_euler_angles(0.2, 0.4, -0.2);
        let visible = make_observation(2, &truth);
        let recovered = inverse(dimensions(), &visible, Some(&carried));
        let target = visible.points[16];
        assert!(
            (forward(dimensions(), &recovered).0[16] - target).norm()
                < (forward(dimensions(), &carried).0[16] - target).norm()
        );
    }
    #[test]
    fn recording_predicts_only_unobserved_arm_articulation_between_observed_anchors() {
        let mut fitted = Vec::new();
        for frame in 0..=10 {
            let mut truth = Pose::default();
            truth.local[14] = Q::from_axis_angle(&V::z_axis(), frame as f64 / 10.0);
            let mut obs = observation(frame, &truth);
            obs.direct_depth = true;
            let mut pose = truth.clone();
            if (1..10).contains(&frame) {
                for i in [15, 16] {
                    obs.scores[i] = 0.0;
                    obs.pixel_scores[i] = 0.0;
                }
                pose.local[14] = Q::identity();
            }
            fitted.push((pose, obs));
        }
        let before = fitted.clone();
        complete_occluded_arms(&mut fitted);
        assert!((fitted[5].0.local[14].angle() - 0.5).abs() < 1e-10);
        for (i, (pose, obs)) in fitted.iter().enumerate() {
            let positions = forward(dimensions(), pose).0;
            let original = forward(dimensions(), &before[i].0).0;
            for j in 0..N {
                if obs.scores[j] > 0.0 || obs.pixel_scores[j] > 0.0 {
                    assert!((positions[j] - original[j]).norm() < 1e-12);
                }
            }
        }
        // A visible wrist constrains the entire chain even when its elbow is
        // unobserved. Do not replace that fitted articulation with interpolation.
        let mut partial = before;
        for (_, obs) in &mut partial {
            obs.scores[16] = 0.9;
        }
        complete_occluded_arms(&mut partial);
        assert_eq!(partial[5].0.local[14], Q::identity());
    }
    #[test]
    fn previous_pose_penalty_applies_only_to_unobserved_joints() {
        let mut obs = observation(1, &pose());
        obs.scores[12] = 0.2;
        obs.pixel_scores[12] = 0.0;
        obs.scores[13] = 0.0;
        obs.pixel_scores[13] = 0.2;
        obs.scores[16] = 0.0;
        obs.pixel_scores[16] = 0.0;
        let context = FitContext {
            sequence_target: None,
            obs: &obs,
            camera: None,
            previous: Some(forward(dimensions(), &pose()).0),
            temporal: 0.6,
            delta: 0.1,
            image_unit: 1.0,
        };
        assert_eq!(context.temporal_weight(12), 0.0);
        assert_eq!(context.temporal_weight(13), 0.0);
        assert!(context.temporal_weight(16) > 0.0);
    }
    #[test]
    fn profile_arm_motion_does_not_change_depth_but_image_plane_motion_is_kept() {
        let mut body = pose();
        body.orientation = Q::from_axis_angle(&V::y_axis(), std::f64::consts::FRAC_PI_2);
        let mut obs = observation(0, &body);
        obs.direct_depth = true;
        obs.scores[..8].fill(0.0);
        obs.pixel_scores[..8].fill(0.0);
        let target = V::new(0.12, -0.08, 2.4);
        for i in 0..N {
            obs.pixels[i] = project(obs.points[i], target, &obs).unwrap().0.into();
        }
        for i in [12, 13, 15, 16] {
            obs.pixels[i][0] += 250.0;
            obs.pixels[i][1] -= 150.0;
        }
        let camera = fit_camera_seeded(&obs, &obs.points, Some(V::new(0.0, 0.0, 2.4))).unwrap();
        assert!(profile_depth_ambiguous(&obs, &obs.points));
        assert!((camera - target).norm() < 1e-6);
        assert_eq!(camera.z, 2.4);
    }
    #[test]
    fn perspective_camera_and_projection_jacobian_match_known_geometry() {
        let mut obs = observation(0, &pose());
        obs.direct_depth = true;
        let camera = V::new(0.12, -0.08, 2.4);
        for i in 0..N {
            obs.pixels[i] = project(obs.points[i], camera, &obs).unwrap().0.into();
        }
        let recovered = fit_camera(&obs, &obs.points).unwrap();
        assert!((recovered - camera).norm() < 1e-5);
        let context = FitContext {
            sequence_target: None,
            obs: &obs,
            camera: Some(camera),
            previous: None,
            temporal: 0.0,
            delta: 0.05,
            image_unit: camera.z / obs.focal,
        };
        let mut invalid = obs.points;
        invalid[13].z = -camera.z;
        assert!(!context.cost(&invalid).is_finite());
        let p = obs.points[13];
        let (_, j) = project(p, camera, &obs).unwrap();
        for axis in 0..3 {
            let mut offset = V::zeros();
            offset[axis] = 1e-6;
            let finite = (project(p + offset, camera, &obs).unwrap().0
                - project(p - offset, camera, &obs).unwrap().0)
                / 2e-6;
            assert!((finite - j.column(axis)).norm() < 1e-5);
        }
    }
    #[test]
    fn image_alignment_corrects_a_bad_depth_prediction_without_resizing_limbs() {
        let mut obs = observation(0, &pose());
        let camera = V::new(0.12, -0.08, 2.4);
        for i in 0..N {
            obs.pixels[i] = project(obs.points[i], camera, &obs).unwrap().0.into();
        }
        obs.points[13].z += 0.2;
        let old = inverse(dimensions(), &obs, None);
        let error = |p: &Pose, obs: &Observation| {
            let positions = forward(dimensions(), p).0;
            (1..N)
                .map(|i| {
                    (project(positions[i], camera, &obs).unwrap().0 - Vector2::from(obs.pixels[i]))
                        .norm_squared()
                })
                .sum::<f64>()
        };
        let before = error(&old, &obs);
        obs.direct_depth = true;
        let fitted = inverse(dimensions(), &obs, None);
        let after = error(&fitted, &obs);
        assert!(
            after < before * 0.65,
            "reprojection: before {before}, after {after}"
        );
        let points = forward(dimensions(), &fitted).0;
        for i in 1..N {
            assert!(
                ((points[i] - points[PARENT[i]]).norm() - dimensions().offsets()[i].norm()).abs()
                    < 1e-10
            );
        }
    }
    #[test]
    fn upper_body_root_ignores_head_bobbing_and_bends_stay_bounded() {
        let mut obs = observation(0, &pose());
        obs.scores[..8].fill(0.0);
        let before = root_orientation(&obs);
        obs.points[10] += V::new(0.3, 0.0, 0.3);
        assert!((before.inverse() * root_orientation(&obs)).angle() < 1e-10);
        let mut folded = Pose::default();
        folded.local[12] = Q::from_euler_angles(0.0, 0.0, 175.0_f64.to_radians());
        limit_bends(&mut folded);
        let bend = V::x()
            .dot(&(folded.local[12] * V::x()))
            .clamp(-1.0, 1.0)
            .acos();
        assert!((bend - 165.0_f64.to_radians()).abs() < 1e-8);
    }
    #[test]
    fn occluded_chest_does_not_switch_upper_body_fit_to_hidden_legs() {
        let mut obs = observation(0, &pose());
        obs.scores[..8].fill(0.0);
        obs.scores[8] = 0.0;
        obs.scores[11] = 0.0;
        assert!(is_upper(&obs));
        let person = Calibration::new(&obs).parameters();
        assert_eq!(person.body_scope, "upper_body");
        assert!(person.pelvis_width.is_none());
    }
    #[test]
    fn native_depth_bypasses_size_overrides_and_calibrates_without_temporal_warmup() {
        let mut obs = observation(0, &pose());
        obs.direct_depth = true;
        obs.history_frames = 1;
        let mut references = DepthReferences::default();
        let (_, fitted) = inverse_with_size(dimensions(), &obs, None, &mut references);
        assert_eq!(fitted.points, obs.points);
        assert!(references.body.is_empty() && references.hands.is_empty());
        let mut calibration = Calibration::new(&obs);
        calibration.update(&obs, 0.6);
        assert_eq!(calibration.frames, 1);
        assert_eq!(calibration.parameters().units, "mediapipe_estimated_meters");
    }
    #[test]
    fn body_size_depth_covers_head_torso_arms_and_visible_legs() {
        let mut obs = observation(0, &Pose::default());
        let camera = camera_translation(&obs, &obs.points).unwrap();
        let mut references = BTreeMap::new();
        let mut baseline = obs.clone();
        body_size_observation(dimensions(), &obs, camera, &mut references, &mut baseline);
        assert_eq!(references.len(), 18);
        for pixel in &mut obs.pixels {
            pixel[0] = 320.0 + (pixel[0] - 320.0) * 1.25;
            pixel[1] = 240.0 + (pixel[1] - 240.0) * 1.25;
        }
        let mut closer = obs.clone();
        for frame in 1..16 {
            obs.frame = frame;
            closer = obs.clone();
            body_size_observation(dimensions(), &obs, camera, &mut references, &mut closer);
        }
        for i in 0..MODEL_N {
            assert!(
                closer.points[i].z < baseline.points[i].z - 0.1,
                "Missing size cue for joint {i}"
            );
        }
        // Shrinking spans reverse the cue for every observed joint.
        for pixel in &mut obs.pixels {
            pixel[0] = 320.0 + (pixel[0] - 320.0) * 0.6;
            pixel[1] = 240.0 + (pixel[1] - 240.0) * 0.6;
        }
        let mut farther = obs.clone();
        for frame in 16..32 {
            obs.frame = frame;
            farther = obs.clone();
            body_size_observation(dimensions(), &obs, camera, &mut references, &mut farther);
        }
        for i in 0..MODEL_N {
            assert!(farther.points[i].z > closer.points[i].z + 0.2);
        }
        let fit = inverse(dimensions(), &closer, None);
        let points = forward(dimensions(), &fit).0;
        for i in 1..N {
            assert!(
                ((points[i] - points[PARENT[i]]).norm() - dimensions().offsets()[i].norm()).abs()
                    < 1e-9
            );
        }
    }
    #[test]
    fn body_size_skips_hidden_parts_and_compensates_rotation() {
        let mut obs = observation(0, &Pose::default());
        obs.scores[..8].fill(0.0);
        let camera = camera_translation(&obs, &obs.points).unwrap();
        let mut references = BTreeMap::new();
        let mut baseline = obs.clone();
        body_size_observation(dimensions(), &obs, camera, &mut references, &mut baseline);
        assert!(references.keys().all(|(a, b)| *a >= 8 && *b >= 8));
        // Rotate one forearm: the projected span shrinks but its corrected scale
        // and inferred depth remain unchanged.
        let old_scale = references[&(12, 13)].filtered_scale;
        let delta = Q::from_euler_angles(0.0, 0.7, 0.0) * (obs.points[13] - obs.points[12]);
        let center = (obs.points[12] + obs.points[13]) / 2.0;
        obs.points[12] = center - delta / 2.0;
        obs.points[13] = center + delta / 2.0;
        for i in [12, 13] {
            obs.pixels[i] = [
                320.0 + obs.points[i].x * 500.0,
                240.0 + obs.points[i].y * 500.0,
            ];
        }
        obs.frame = 1;
        let mut turned = obs.clone();
        body_size_observation(dimensions(), &obs, camera, &mut references, &mut turned);
        assert!((references[&(12, 13)].filtered_scale / old_scale - 1.0).abs() < 1e-9);
        assert!((turned.points[13].z - obs.points[13].z).abs() < 1e-9);
        obs.pixel_scores.fill(0.0);
        let mut missing = obs.clone();
        body_size_observation(dimensions(), &obs, camera, &mut references, &mut missing);
        assert_eq!(missing.points, obs.points);
    }
    #[test]
    fn palm_size_drives_wrist_depth_and_ik_without_stretching_arms() {
        let mut obs = observation(0, &Pose::default());
        obs.scores[..8].fill(0.0);
        let hand = |scale: f64, yaw: f64| {
            let mut world = [[0.0_f32; 3]; 21];
            for (i, p) in [
                (5, [-0.035, -0.07, 0.0]),
                (9, [-0.012, -0.09, 0.0]),
                (13, [0.012, -0.085, 0.0]),
                (17, [0.035, -0.06, 0.0]),
            ] {
                let p = Q::from_euler_angles(0.0, yaw, 0.0) * V::from(p);
                world[i] = [p.x as f32, p.y as f32, p.z as f32];
            }
            HandPose {
                side: "left".into(),
                confidence: 0.95,
                image_landmarks: world.map(|p| {
                    [
                        (520.0 + f64::from(p[0]) * scale) as f32,
                        (90.0 + f64::from(p[1]) * scale) as f32,
                    ]
                }),
                world_landmarks: world,
            }
        };
        obs.hands = vec![hand(1000.0, 0.0)];
        let mut references = DepthReferences::default();
        let (initial, baseline) = inverse_with_size(dimensions(), &obs, None, &mut references);
        let mut previous = initial.clone();
        obs.hands = vec![hand(1250.0, 0.0)];
        let mut closer = baseline.clone();
        for _ in 0..12 {
            let (pose, target) =
                inverse_with_size(dimensions(), &obs, Some(&previous), &mut references);
            previous = pose;
            closer = target;
        }
        assert!(closer.points[13].z < baseline.points[13].z - 0.15);
        let fitted = forward(dimensions(), &previous).0;
        assert!(fitted[13].z < forward(dimensions(), &initial).0[13].z - 0.1);
        assert!((fitted[13] - closer.points[13]).norm() < 0.03);
        for (a, b, length) in [(11, 12, dimensions().0[4]), (12, 13, dimensions().0[5])] {
            assert!(((fitted[b] - fitted[a]).norm() - length).abs() < 1e-10);
        }
        obs.hands = vec![hand(800.0, 0.0)];
        let mut farther = closer.clone();
        for _ in 0..12 {
            let (pose, target) =
                inverse_with_size(dimensions(), &obs, Some(&previous), &mut references);
            previous = pose;
            farther = target;
        }
        assert!(farther.points[13].z > closer.points[13].z + 0.2);
        // A rotated palm at the same distance keeps its orientation-corrected scale.
        let fixed_lengths = references.hands["left"].lengths;
        let rotated = palm_scale(&hand(1000.0, 0.7), &fixed_lengths).unwrap();
        assert!((rotated / 1000.0 - 1.0).abs() < 1e-5);
        obs.hands[0].confidence = 0.2;
        let (_, missing) = inverse_with_size(dimensions(), &obs, None, &mut references);
        obs.hands.clear();
        let (_, body_only) = inverse_with_size(dimensions(), &obs, None, &mut references);
        assert_eq!(missing.points, body_only.points);
    }
    #[test]
    fn frontal_reach_recovers_wrist_depth_from_foreshortening() {
        let mut obs = observation(0, &Pose::default());
        obs.scores[..8].fill(0.0);
        // The lifter predicts a flat arm, but the detected wrist lies much closer
        // to the elbow in the image, as during a reach toward a frontal camera.
        obs.pixels[13][0] = obs.pixels[12][0] + 40.0;
        let corrected = frontal_arm_observation(dimensions(), &obs, None);
        assert!(corrected.points[13].z < corrected.points[12].z - 0.15);
        assert!(
            ((corrected.points[13] - corrected.points[12]).norm() - dimensions().0[5]).abs() < 1e-9
        );
        let mut torso = obs.clone();
        for i in [12, 13, 15, 16] {
            torso.scores[i] = 0.0;
        }
        let camera = camera_translation(&torso, &obs.points).unwrap();
        let wrist = corrected.points[13] + camera;
        assert!((obs.focal * wrist.x / wrist.z + 320.0 - obs.pixels[13][0]).abs() < 1e-8);
        assert!((obs.focal * wrist.y / wrist.z + 240.0 - obs.pixels[13][1]).abs() < 1e-8);
        // An unreliable wrist must not invent depth from its pixel position.
        obs.pixel_scores[13] = 0.0;
        let missing = frontal_arm_observation(dimensions(), &obs, None);
        assert_eq!(missing.points[13], obs.points[13]);
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
        let joints: Vec<_> = (0..MODEL_N)
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
                ear_head: false,
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
            hand_shapes: BTreeMap::new(),
            depth_references: DepthReferences::default(),
            observations: Vec::new(),
            previous: None,
            previous_head: None,
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
                let parent = recording["parents"][i].as_u64().unwrap() as usize;
                positions[i] = positions[parent] + world[parent] * offsets[i];
                world[i] = world[parent] * quat(&frame["joint_rotations"][i]["quaternion_xyzw"]);
                assert!(
                    ((positions[i] - positions[parent]).norm() - offsets[i].norm()).abs() < 1e-10
                );
            }
            assert_eq!(positions[9], positions[8]);
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
    #[test]
    fn upper_body_fit_and_calibration_ignore_hidden_lower_body_geometry() {
        let mut obs = observation(0, &pose());
        for i in 0..8 {
            obs.scores[i] = 0.0;
            obs.pixel_scores[i] = 0.0;
        }
        let mut calibration = Calibration::new(&obs);
        for _ in 0..40 {
            calibration.update(&obs, 0.6);
        }
        let person = serde_json::to_value(calibration.parameters()).unwrap();
        assert_eq!(person["body_scope"], "upper_body");
        assert!(person.get("pelvis_width").is_none());
        assert!(person["segment_lengths"].get("upper_leg").is_none());
        assert!(person["measurement_counts"].get("lower_leg").is_none());
        let original = dimensions();
        let mut altered = original;
        altered.0[0] *= 4.0;
        altered.0[2] *= 10.0;
        altered.0[3] *= 10.0;
        altered.0[6] *= 3.0;
        altered.0[7] *= 3.0;
        let a = inverse(original, &obs, None);
        let b = inverse(altered, &obs, None);
        let pa = forward(original, &a).0;
        let pb = forward(altered, &b).0;
        assert!((pa[8] - obs.points[8]).norm() < 1e-12);
        assert!((pb[8] - obs.points[8]).norm() < 1e-12);
        for i in 8..17 {
            assert!(
                (pa[i] - pb[i]).norm() < 1e-6,
                "Hidden dimensions changed visible joint {i}"
            );
        }
        let frame = frame_parameters(original, &obs, &a);
        assert!(
            frame
                .angles
                .iter()
                .all(|a| !a.joint.contains("hip") && !a.joint.contains("knee"))
        );
    }
    /// Refit cached model observations without repeating camera/model inference.
    #[test]
    #[ignore = "requires LIFTING_EVAL_DIR and cached recording observations"]
    fn evaluate_recorded_twin_fit() -> Result<()> {
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let input = std::env::var("LIFTING_EVAL_INPUT").unwrap_or("candidate".into());
        let output = std::env::var("LIFTING_EVAL_TAG").unwrap_or("refit".into());
        ensure!(
            [&input, &output].iter().all(|s| s
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')),
            "Invalid diagnostic tag"
        );
        let snapshot: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("twin.json"))?)?;
        let timing = &snapshot["recording"];
        let times: Vec<f64> = serde_json::from_value(timing["timestamps_seconds"].clone())?;
        let mut recorder = TwinRecorder::default();
        recorder.set_upright_torso(timing["assume_upright_torso"].as_bool().unwrap_or(true));
        for line in std::fs::read_to_string(directory.join(format!("{input}.jsonl")))?.lines() {
            let row: serde_json::Value = serde_json::from_str(line)?;
            let frame = row["frame"].as_u64().unwrap();
            for person in row["people"].as_array().unwrap() {
                let raw = &person["pose3d"];
                if raw.is_null() {
                    continue;
                }
                ensure!(
                    raw["units"] == "estimated_meters"
                        && raw["coordinate_space"] == "root_relative_camera",
                    "Evaluation requires estimated camera-space meter coordinates"
                );
                let model = match raw["model"].as_str() {
                    Some("MediaPipe-Pose") => "MediaPipe-Pose",
                    Some("VideoPose3D-Arms-NativeShape-Diagnostic") => {
                        "VideoPose3D-Arms-NativeShape-Diagnostic"
                    }
                    Some("VideoPose3D-NativeScale-Diagnostic") => {
                        "VideoPose3D-NativeScale-Diagnostic"
                    }
                    _ => anyhow::bail!("Unsupported diagnostic model"),
                };
                let pose = Pose3d {
                    ear_head: raw["ear_head"].as_bool().unwrap_or(false),
                    model,
                    coordinate_space: "root_relative_camera",
                    units: "estimated_meters",
                    history_frames: 1,
                    angles: Vec::new(),
                    joints: raw["joints"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, p)| Joint3d {
                            name: motion3d::NAMES[i],
                            x: p["x"].as_f64().unwrap() as f32,
                            y: p["y"].as_f64().unwrap() as f32,
                            z: p["z"].as_f64().unwrap() as f32,
                            input_score: p["input_score"].as_f64().unwrap() as f32,
                        })
                        .collect(),
                };
                let points: Vec<_> = person["keypoints"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, p)| Keypoint {
                        name: crate::landmarks::NAMES[i],
                        x: p["x"].as_f64().unwrap() as f32,
                        y: p["y"].as_f64().unwrap() as f32,
                        score: p["score"].as_f64().unwrap() as f32,
                    })
                    .collect();
                let b = &person["bbox"];
                let hands: Vec<HandPose> = serde_json::from_value(person["hands"].clone())?;
                let head: Option<HeadPose> = serde_json::from_value(person["head"].clone())?;
                recorder.set_image_torso_scale(
                    person["image_torso_scale"].as_f64(),
                    person["image_torso_span"].as_u64().unwrap_or(1) as usize,
                );
                recorder.observe(
                    frame,
                    times[frame as usize],
                    &pose,
                    &points,
                    &hands,
                    head,
                    Bbox {
                        x: b["x"].as_f64().unwrap() as f32,
                        y: b["y"].as_f64().unwrap() as f32,
                        width: b["width"].as_f64().unwrap() as f32,
                        height: b["height"].as_f64().unwrap() as f32,
                    },
                    Size::new(
                        row["width"].as_i64().unwrap() as i32,
                        row["height"].as_i64().unwrap() as i32,
                    ),
                    0.3,
                    None,
                );
            }
        }
        if std::env::var_os("LIFTING_EVAL_FIXED_BODY").is_some() {
            let base_tag = std::env::var("LIFTING_EVAL_BASE").unwrap_or("verifiedarms".into());
            ensure!(
                base_tag
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
                "Invalid base tag"
            );
            let native_prior = std::env::var_os("LIFTING_EVAL_NATIVE_PRIOR").is_some();
            let mut original: serde_json::Value = serde_json::from_slice(&std::fs::read(
                directory.join(format!("{base_tag}-twin.json")),
            )?)?;
            let offsets: Vec<[f64; 3]> =
                serde_json::from_value(original["subjects"][0]["person"]["rest_offsets"].clone())?;
            let model_to_export: [Option<usize>; N] = std::array::from_fn(|i| {
                original["joint_names"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .position(|n| n.as_str() == Some(joint_name(i)))
            });
            let length = |i: usize| V::from(offsets[model_to_export[i].unwrap()]).norm();
            let mut dimensions = Dimensions([
                length(1) * 2.0,
                length(11) * 2.0,
                length(2),
                length(3),
                length(12),
                length(13),
                length(7),
                length(8),
                length(9),
                length(10),
            ]);
            let arm_scale: f64 = std::env::var("LIFTING_EVAL_ARM_SCALE")
                .unwrap_or("1".into())
                .parse()?;
            ensure!(
                arm_scale.is_finite() && (0.8..=1.5).contains(&arm_scale),
                "Invalid arm scale"
            );
            dimensions.0[4] *= arm_scale;
            dimensions.0[5] *= arm_scale;
            if arm_scale != 1.0 {
                for (i, offset) in dimensions.offsets().iter().enumerate() {
                    if let Some(export) = model_to_export[i] {
                        original["subjects"][0]["person"]["rest_offsets"][export] =
                            serde_json::to_value(array(*offset))?;
                    }
                }
                original["subjects"][0]["person"]["segment_lengths"]["upper_arm"] =
                    serde_json::json!(dimensions.0[4]);
                original["subjects"][0]["person"]["segment_lengths"]["forearm"] =
                    serde_json::json!(dimensions.0[5]);
            }
            let rotation = |v: &serde_json::Value| {
                let q: [f64; 4] = serde_json::from_value(v.clone()).unwrap();
                Q::from_quaternion(nalgebra::Quaternion::new(q[3], q[0], q[1], q[2]))
            };
            for frame in original["subjects"][0]["frames"].as_array_mut().unwrap() {
                let n = frame["frame"].as_u64().unwrap();
                let Some(source) = recorder.subjects[0]
                    .observations
                    .iter()
                    .find(|o| o.frame == n)
                else {
                    continue;
                };
                let base = Pose {
                    time_seconds: source.time_seconds,
                    root: V::from(serde_json::from_value::<[f64; 3]>(
                        frame["root_position"].clone(),
                    )?),
                    orientation: rotation(&frame["root_orientation_xyzw"]),
                    local: std::array::from_fn(|i| {
                        model_to_export[i]
                            .map(|export| {
                                rotation(&frame["joint_rotations"][export]["quaternion_xyzw"])
                            })
                            .unwrap_or_else(Q::identity)
                    }),
                    camera: Some(V::zeros()),
                    depth_inferred: !frame["root_depth_observed"].as_bool().unwrap_or(false),
                };
                let positions = forward(dimensions, &base).0;
                let mut obs = source.clone();
                let shift = positions[8] - obs.points[8];
                for p in &mut obs.points {
                    *p += shift;
                }
                let directions: Vec<_> = [(11, 12), (12, 13), (14, 15), (15, 16)]
                    .into_iter()
                    .map(|(a, b)| (obs.points[b] - obs.points[a]).normalize())
                    .collect();
                if !native_prior {
                    for (k, (a, b, bone_length)) in [
                        (11, 12, dimensions.0[4]),
                        (12, 13, dimensions.0[5]),
                        (14, 15, dimensions.0[4]),
                        (15, 16, dimensions.0[5]),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        obs.points[b] = if [11, 14].contains(&a) {
                            positions[a]
                        } else {
                            obs.points[a]
                        } + directions[k] * bone_length;
                    }
                }
                let fitted = refine_fixed_body_arms(dimensions, &obs, &base);
                let result = serde_json::to_value(frame_parameters(dimensions, &obs, &fitted))?;
                for i in [11, 12, 14, 15] {
                    frame["joint_rotations"][model_to_export[i].unwrap()]["quaternion_xyzw"] =
                        result["joint_rotations"][ORDER.iter().position(|&j|j==i).unwrap()]["quaternion_xyzw"].clone();
                }
                for key in ["angles", "fit_rmse", "reprojection_rmse_px"] {
                    frame[key] = result[key].clone();
                }
            }
            std::fs::write(
                directory.join(format!("{output}-twin.json")),
                serde_json::to_vec(&original)?,
            )?;
            std::fs::write(directory.join(format!("{output}-source.txt")), &input)?;
            return Ok(());
        }
        recorder.set_timing(RecordingTiming {
            fps: timing["fps"].as_f64().unwrap(),
            frame_count: timing["frame_count"].as_u64().unwrap(),
            timestamps_seconds: times,
            source_video: timing["source_video"].as_str().map(String::from),
            trim_start_seconds: timing["trim_start_seconds"].as_f64().unwrap(),
            trim_end_seconds: timing["trim_end_seconds"].as_f64().unwrap(),
            upper_body: timing["upper_body"].as_bool().unwrap(),
            assume_upright_torso: timing["assume_upright_torso"].as_bool().unwrap_or(true),
        });
        recorder.save(&directory.join(format!("{output}-twin.json")))?;
        std::fs::write(directory.join(format!("{output}-source.txt")), input)?;
        Ok(())
    }
}
