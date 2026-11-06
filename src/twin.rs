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
    time_seconds: f64,
    root: V,
    orientation: Q,
    local: [Q; N],
    camera: Option<V>,
}
impl Default for Pose {
    fn default() -> Self {
        Self {
            time_seconds: 0.0,
            root: V::zeros(),
            orientation: Q::identity(),
            local: [Q::identity(); N],
            camera: None,
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
            direct_depth: obs.direct_depth,
            estimates,
            revision: 0,
            frames: 0,
        }
    }
    fn update(&mut self, obs: &Observation, minimum: f64) {
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
            active_joint_indices: (0..N)
                .filter(|&i| !self.upper_body || crate::scope::body_upper(i))
                .collect(),
            shoulder_width: dimensions.0[1],
            segment_lengths: (2..10)
                .filter(|i| !self.upper_body || [4, 5, 8, 9].contains(i))
                .map(|i| (DIMENSIONS[i], dimensions.0[i]))
                .collect(),
            rest_offsets: dimensions.offsets().map(array),
            bilateral_symmetry: true,
            calibration_frames: self.frames,
            hands: BTreeMap::new(),
            measurement_counts: (0..10)
                .filter(|i| !self.upper_body || [1, 4, 5, 8, 9].contains(i))
                .map(|i| (DIMENSIONS[i], self.estimates[i].samples))
                .collect(),
        }
    }
}

#[derive(Serialize)]
pub struct PersonParameters {
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
    frame: u64,
    time_seconds: f64,
    direct_depth: bool,
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

#[derive(Default)]
struct HandSmoother {
    world: [crate::smoothing::AdaptivePoint; 21],
    pixels: [crate::smoothing::AdaptivePoint; 21],
}
#[derive(Default)]
struct ObservationSmoother {
    points: [crate::smoothing::AdaptivePoint; N],
    pixels: [crate::smoothing::AdaptivePoint; N],
    hands: BTreeMap<String, HandSmoother>,
    head: crate::smoothing::AdaptiveRotation,
}
impl ObservationSmoother {
    fn apply(&mut self, obs: &Observation) -> Observation {
        let mut output = obs.clone();
        if !obs.direct_depth {
            return output;
        }
        let anchor = if is_upper(obs) {
            obs.points[8]
        } else {
            obs.points[0]
        };
        let scale = (obs.points[11] - obs.points[14]).norm().max(0.1);
        for i in 0..N {
            if obs.scores[i] > 0.0 {
                let speed = scale
                    * if [12, 13, 15, 16].contains(&i) {
                        20.0
                    } else {
                        10.0
                    };
                output.points[i] = anchor
                    + self.points[i].filter(
                        obs.points[i] - anchor,
                        obs.time_seconds,
                        scale,
                        speed,
                        1.5,
                    );
            }
            if obs.pixel_scores[i] > 0.0 {
                let p = self.pixels[i].filter(
                    V::new(obs.pixels[i][0], obs.pixels[i][1], 0.0),
                    obs.time_seconds,
                    obs.size.width as f64,
                    obs.size.width as f64 * 4.0,
                    15.0,
                );
                output.pixels[i] = [p.x, p.y];
            }
        }
        for hand in &mut output.hands {
            let state = self.hands.entry(hand.side.clone()).or_default();
            let world = hand.world_landmarks[0].map(f64::from);
            let pixels = hand.image_landmarks[0].map(f64::from);
            for i in 1..21 {
                let relative = V::from(hand.world_landmarks[i].map(f64::from)) - V::from(world);
                let p = V::from(world)
                    + state.world[i].filter(relative, obs.time_seconds, 0.08, 2.0, 1.0);
                hand.world_landmarks[i] = [p.x as f32, p.y as f32, p.z as f32];
                let relative = V::new(
                    hand.image_landmarks[i][0] as f64 - pixels[0],
                    hand.image_landmarks[i][1] as f64 - pixels[1],
                    0.0,
                );
                let p = state.pixels[i].filter(relative, obs.time_seconds, 80.0, 2000.0, 1.0);
                hand.image_landmarks[i] = [(pixels[0] + p.x) as f32, (pixels[1] + p.y) as f32];
            }
        }
        if let Some(head) = &mut output.head {
            head.orientation_xyzw = quaternion(self.head.filter(
                Q::new_normalize(nalgebra::Quaternion::new(
                    head.orientation_xyzw[3],
                    head.orientation_xyzw[0],
                    head.orientation_xyzw[1],
                    head.orientation_xyzw[2],
                )),
                obs.time_seconds,
            ));
        }
        output
    }
}
fn smoothed_observations(observations: &[Observation]) -> Vec<Observation> {
    let mut state = ObservationSmoother::default();
    let mut forward: Vec<_> = observations.iter().map(|o| state.apply(o)).collect();
    let mut state = ObservationSmoother::default();
    for (raw, output) in observations.iter().zip(&mut forward).rev() {
        let backward = state.apply(raw);
        if !raw.direct_depth {
            continue;
        }
        for i in 0..N {
            output.points[i] = (output.points[i] + backward.points[i]) / 2.0;
            for axis in 0..2 {
                output.pixels[i][axis] = (output.pixels[i][axis] + backward.pixels[i][axis]) / 2.0;
            }
        }
        for hand in &mut output.hands {
            if let Some(other) = backward.hands.iter().find(|h| h.side == hand.side) {
                for i in 0..21 {
                    for axis in 0..3 {
                        hand.world_landmarks[i][axis] =
                            (hand.world_landmarks[i][axis] + other.world_landmarks[i][axis]) / 2.0;
                    }
                    for axis in 0..2 {
                        hand.image_landmarks[i][axis] =
                            (hand.image_landmarks[i][axis] + other.image_landmarks[i][axis]) / 2.0;
                    }
                }
            }
        }
        if let (Some(head), Some(other)) = (&mut output.head, &backward.head) {
            let q =
                |a: [f64; 4]| Q::new_normalize(nalgebra::Quaternion::new(a[3], a[0], a[1], a[2]));
            head.orientation_xyzw =
                quaternion(q(head.orientation_xyzw).slerp(&q(other.orientation_xyzw), 0.5));
        }
    }
    forward
}

fn is_upper(obs: &Observation) -> bool {
    obs.scores[..8].iter().all(|s| *s == 0.0)
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
struct FitContext<'a> {
    obs: &'a Observation,
    camera: Option<V>,
    previous: Option<[V; N]>,
    temporal: f64,
    delta: f64,
}
impl FitContext<'_> {
    fn temporal_weight(&self, i: usize) -> f64 {
        if is_upper(self.obs) && i < 8 {
            0.0
        } else {
            self.temporal * (1.0 + 2.0 * (1.0 - self.obs.scores[i]))
        }
    }
    fn cost(&self, positions: &[V; N]) -> f64 {
        let native = self.obs.direct_depth;
        let mut result = cost(positions, self.obs, self.delta) * if native { 0.4 } else { 1.0 };
        for i in 1..N {
            if let Some(camera) = self.camera {
                if let Some((pixel, _)) = project(positions[i], camera, self.obs) {
                    let r = (pixel - Vector2::from(self.obs.pixels[i])).norm();
                    result += 2.0
                        * self.obs.pixel_scores[i].powi(2)
                        * huber(r, 12.0)
                        * (camera.z / self.obs.focal).powi(2);
                } else if self.obs.pixel_scores[i] > 0.0 {
                    return f64::INFINITY;
                }
            }
            if let Some(previous) = &self.previous {
                result +=
                    self.temporal_weight(i) * (positions[i] - previous[i]).norm_squared() / 2.0;
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

fn initial_pose(dimensions: Dimensions, obs: &Observation) -> Pose {
    let mut pose = Pose {
        time_seconds: obs.time_seconds,
        orientation: root_orientation(obs),
        ..Pose::default()
    };
    for node in 1..N {
        if is_upper(obs) && node < 8 {
            continue;
        }
        if let Some(child) = ((node + 1)..N)
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
        constrain_upright_torso(&mut pose, is_upper(obs));
    }
    pose
}

/// Block coordinate damped least-squares IK with analytic rotational Jacobians.
/// Each step changes rotations only: dimensions remain fixed for the entire solve.
fn inverse(dimensions: Dimensions, obs: &Observation, previous: Option<&Pose>) -> Pose {
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
            constrain_upright_torso(p, is_upper(obs));
        }
        align_upper_anchor(dimensions, p, obs);
    }
    let dt = previous
        .as_ref()
        .map(|p| obs.time_seconds - p.time_seconds)
        .unwrap_or(1.0 / 30.0)
        .max(1.0 / 240.0);
    let context = FitContext {
        obs,
        delta,
        camera: if obs.direct_depth {
            fit_camera_seeded(obs, &obs.points, previous.as_ref().and_then(|p| p.camera))
        } else {
            None
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
    for _ in 0..12 {
        let before = context.cost(&forward(dimensions, &pose).0);
        for node in (0..N).rev() {
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
                        let units = camera.z / obs.focal;
                        let j = derivative * jacobian * units;
                        let weight = 2.0
                            * obs.pixel_scores[i].powi(2)
                            * robust_weight(residual.norm(), 12.0);
                        a += weight * j.transpose() * j;
                        b += weight * j.transpose() * (residual * units);
                    }
                }
                if let Some(previous) = &context.previous {
                    let weight = context.temporal_weight(i);
                    a += weight * jacobian.transpose() * jacobian;
                    b += weight * jacobian.transpose() * (previous[i] - positions[i]);
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
                    constrain_upright_torso(&mut candidate, is_upper(obs));
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
    pose.camera = if obs.direct_depth {
        fit_camera_seeded(obs, &forward(dimensions, &pose).0, context.camera)
    } else {
        None
    };
    pose
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
        .map(|i| (PARENT[i], i, offsets[i].norm()))
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
fn fit_camera_seeded(obs: &Observation, positions: &[V; N], seed: Option<V>) -> Option<V> {
    let mut camera = camera_translation(obs, positions).or(seed)?;
    if !obs.direct_depth {
        return Some(camera);
    }
    let weight = |i: usize| {
        obs.pixel_scores[i].min(obs.scores[i]).powi(2)
            * if [0, 7, 8, 9, 10, 11, 14].contains(&i) {
                1.0
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
    let translation = fit_camera_seeded(obs, &positions, pose.camera);
    let reprojection_rmse_px = translation.and_then(|camera| {
        let mut sum = 0.0;
        let mut weights = 0.0;
        for i in 0..N {
            if let Some((pixel, _)) = project(positions[i], camera, obs) {
                let weight = obs.pixel_scores[i].min(obs.scores[i]).powi(2);
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
        joint_rotations: (0..N)
            .map(|i| JointRotation {
                joint: motion3d::NAMES[i],
                quaternion_xyzw: quaternion(pose.local[i]),
                confidence: obs.scores[i],
            })
            .collect(),
        root_translation_valid: translation.is_some(),
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
    pub fn tracking_projection(&self, size: Size) -> Option<[[f32; 2]; N]> {
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
    smoother: ObservationSmoother,
    observations: Vec<Observation>,
    previous: Option<Pose>,
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
        let mut frame = frame_parameters(self.calibration.dimensions(), obs, pose);
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
    subjects: Vec<Subject>,
    last_frame: Option<u64>,
    last_bbox: Option<Bbox>,
    last_size: Option<Size>,
    recording: Option<RecordingTiming>,
    free_torso: bool,
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
        if pose.joints.len() != N || points.len() < N {
            return None;
        }
        let coco = std::array::from_fn(|i| [points[i].x, points[i].y, points[i].score]);
        let screen = if pose.model == "MediaPipe-Pose" {
            crate::mediapipe::skeleton_pixels(points)
        } else {
            motion3d::coco_to_h36m(coco)
        };
        let obs = Observation {
            frame,
            time_seconds,
            direct_depth: pose.model == "MediaPipe-Pose",
            assume_upright_torso: !self.free_torso,
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
            hands: hands.to_vec(),
            head,
        };
        if obs.points.iter().any(|p| !p.iter().all(|x| x.is_finite()))
            || (obs.scores.iter().filter(|&&s| s > 0.0).count() < 4
                && !(is_upper(&obs) && obs.scores[11].min(obs.scores[14]) >= 0.5))
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
                hand_shapes: BTreeMap::new(),
                depth_references: DepthReferences::default(),
                smoother: ObservationSmoother::default(),
                observations: Vec::new(),
                previous: None,
            });
        }
        self.last_frame = Some(frame);
        self.last_bbox = Some(bbox);
        self.last_size = Some(size);
        let subject_segment = self.subjects.len() - 1;
        let subject = self.subjects.last_mut().unwrap();
        let smoothed = subject.smoother.apply(&obs);
        subject
            .calibration
            .update(&smoothed, f64::from(threshold.max(0.6)));
        let dimensions = subject.calibration.dimensions();
        let (solved, fitted_observation) = inverse_with_size(
            dimensions,
            &smoothed,
            subject.previous.as_ref(),
            &mut subject.depth_references,
        );
        subject.learn_hands(&fitted_observation, &solved);
        let parameters = subject.frame(&fitted_observation, &solved);
        let fitted = forward(dimensions, &solved).0;
        let joints: Vec<_> = (0..N)
            .map(|i| Joint3d {
                name: motion3d::NAMES[i],
                x: (fitted[i].x - solved.root.x) as f32,
                y: (fitted[i].y - solved.root.y) as f32,
                z: (fitted[i].z - solved.root.z) as f32,
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
            serde_json::to_writer(&mut writer, &subject.parameters())?;
            writer.write_all(b",\"frames\":[")?;
            let dimensions = subject.calibration.dimensions();
            let mut previous = None;
            let mut depth_references = DepthReferences::default();
            let observations = smoothed_observations(&subject.observations);
            for (index, obs) in observations.iter().enumerate() {
                if index > 0 {
                    writer.write_all(b",")?;
                }
                let (pose, fitted_observation) =
                    inverse_with_size(dimensions, obs, previous.as_ref(), &mut depth_references);
                let frame = subject.frame(&fitted_observation, &pose);
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
            time_seconds: frame as f64 / 30.0,
            direct_depth: false,
            assume_upright_torso: false,
            points: positions,
            scores: [0.9; N],
            pixels: positions.map(|p| [320.0 + p.x * 500.0, 240.0 + p.y * 500.0]),
            pixel_scores: [0.9; N],
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
            model: "MediaPipe-Pose",
            coordinate_space: "root_relative_camera",
            units: "estimated_meters",
            history_frames: 1,
            joints: (0..N)
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
            obs: &obs,
            camera: Some(camera),
            previous: None,
            temporal: 0.0,
            delta: 0.05,
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
        for i in 0..N {
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
        for i in 0..N {
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
            hand_shapes: BTreeMap::new(),
            depth_references: DepthReferences::default(),
            smoother: ObservationSmoother::default(),
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
}
