//! Model-free skeletal-twin playback: deserialize shared offsets + local quaternions,
//! reconstruct with FK, and render from an orbitable perspective camera.
use crate::details::{HAND_PARENT, HandFrame, HandShape, HeadPose};
use anyhow::{Context, Result, ensure};
use nalgebra::{Quaternion, Rotation3, UnitQuaternion, Vector3};
use opencv::{
    core::{Mat, Point, Scalar, Size},
    highgui, imgproc,
    prelude::*,
    videoio,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::{
    fs::File,
    path::Path,
    time::{Duration, Instant},
};

type V = Vector3<f64>;
type Q = UnitQuaternion<f64>;
const N: usize = 17;
#[derive(Clone, Deserialize)]
struct Recording {
    schema_version: u32,
    quaternion_convention: String,
    joint_names: Vec<String>,
    parents: Vec<Option<usize>>,
    #[serde(default)]
    recording: Option<Timing>,
    subjects: Vec<Subject>,
}
#[derive(Clone, Deserialize)]
struct Timing {
    fps: f64,
    frame_count: u64,
    timestamps_seconds: Vec<f64>,
    #[serde(default)]
    source_video: Option<String>,
    #[serde(default = "default_trim")]
    trim_start_seconds: f64,
    #[serde(default = "default_trim")]
    trim_end_seconds: f64,
    #[serde(default)]
    upper_body: bool,
}
#[derive(Clone, Deserialize)]
struct Subject {
    subject_segment: usize,
    person: Person,
    frames: Vec<Frame>,
}
#[derive(Clone, Deserialize)]
struct Person {
    rest_offsets: Vec<[f64; 3]>,
    #[serde(default)]
    hands: BTreeMap<String, HandShape>,
}
#[derive(Clone, Default, Deserialize)]
struct Prediction {
    body: bool,
    hands: bool,
    head: bool,
}
#[derive(Clone, Deserialize)]
struct Frame {
    #[serde(default)]
    prediction: Prediction,
    frame: u64,
    root_position: [f64; 3],
    root_orientation_xyzw: [f64; 4],
    joint_rotations: Vec<JointRotation>,
    #[serde(default)]
    hands: Vec<HandFrame>,
    #[serde(default)]
    head: Option<HeadPose>,
}
#[derive(Clone, Deserialize)]
struct JointRotation {
    joint: String,
    quaternion_xyzw: [f64; 4],
}

pub struct Options<'a> {
    pub output: Option<&'a Path>,
    pub no_display: bool,
    pub subject: Option<usize>,
    pub fps: Option<f64>,
    pub max_frames: Option<u64>,
    pub trim_start: Option<f64>,
    pub trim_end: Option<f64>,
    pub full_body: bool,
}

fn default_trim() -> f64 {
    3.0
}

fn vector(p: [f64; 3]) -> V {
    V::new(p[0], p[1], p[2])
}
fn rotation(p: [f64; 4]) -> Result<Q> {
    ensure!(
        p.iter().all(|v| v.is_finite()),
        "Non-finite replay rotation"
    );
    let q = Quaternion::new(p[3], p[0], p[1], p[2]);
    ensure!(
        q.norm() > 1e-9 && q.norm() < 1e9,
        "Invalid replay quaternion"
    );
    Ok(Q::from_quaternion(q))
}
impl Recording {
    fn load(path: &Path) -> Result<Self> {
        let record: Self = serde_json::from_reader(
            File::open(path).with_context(|| format!("Cannot open twin {}", path.display()))?,
        )
        .context("Invalid skeletal twin JSON")?;
        ensure!(
            record.schema_version == 1,
            "Unsupported skeletal twin schema version"
        );
        ensure!(
            record.quaternion_convention == "xyzw_active_parent_local",
            "Unsupported quaternion convention"
        );
        ensure!(
            record.joint_names.len() == N && record.parents.len() == N,
            "Expected a 17-joint twin hierarchy"
        );
        ensure!(record.parents[0].is_none(), "Root must have no parent");
        for i in 1..N {
            ensure!(
                record.parents[i].is_some_and(|p| p < i),
                "Hierarchy must be in parent-before-child order"
            );
        }
        for subject in &record.subjects {
            ensure!(
                subject.person.rest_offsets.len() == N,
                "Missing shared rest offsets"
            );
            ensure!(
                subject
                    .person
                    .rest_offsets
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite() && v.abs() < 1e9),
                "Invalid person dimensions"
            );
            for shape in subject.person.hands.values() {
                ensure!(
                    shape.bone_lengths.len() == 21
                        && shape
                            .bone_lengths
                            .iter()
                            .all(|v| v.is_finite() && *v >= 0.0 && *v < 10.0),
                    "Invalid shared hand lengths"
                );
            }
            let mut last = None;
            for f in &subject.frames {
                ensure!(
                    last.is_none_or(|previous| f.frame > previous),
                    "Replay frames must be strictly ordered"
                );
                last = Some(f.frame);
                ensure!(
                    f.root_position
                        .iter()
                        .all(|v| v.is_finite() && v.abs() < 1e9),
                    "Invalid root position"
                );
                rotation(f.root_orientation_xyzw)?;
                if let Some(head) = &f.head {
                    rotation(head.orientation_xyzw)?;
                }
                for hand in &f.hands {
                    ensure!(
                        subject.person.hands.contains_key(&hand.side)
                            && matches!(hand.side.as_str(), "left" | "right"),
                        "Missing/invalid hand shape"
                    );
                    ensure!(
                        hand.bone_rotations_xyzw.len() == 21,
                        "Expected 21 hand bone rotations"
                    );
                    for &q in &hand.bone_rotations_xyzw {
                        rotation(q)?;
                    }
                }
                ensure!(
                    f.joint_rotations.len() == N,
                    "Missing per-frame joint rotations"
                );
                for (i, q) in f.joint_rotations.iter().enumerate() {
                    ensure!(
                        q.joint == record.joint_names[i],
                        "Joint rotation order does not match the hierarchy"
                    );
                    rotation(q.quaternion_xyzw)?;
                }
            }
        }
        if let Some(t) = &record.recording {
            ensure!(
                t.fps.is_finite() && t.fps > 0.0 && t.fps <= 1000.0,
                "Invalid recording frame rate"
            );
            ensure!(
                t.timestamps_seconds.len() as u64 == t.frame_count,
                "Recording timestamp count does not match frame count"
            );
            ensure!(
                t.timestamps_seconds
                    .iter()
                    .all(|t| t.is_finite() && *t >= 0.0)
                    && t.timestamps_seconds.windows(2).all(|p| p[1] > p[0]),
                "Recording times must be finite and strictly increasing"
            );
            ensure!(
                record
                    .subjects
                    .iter()
                    .flat_map(|s| &s.frames)
                    .all(|f| f.frame < t.frame_count),
                "Pose frame exceeds recording timeline"
            );
        }
        Ok(record)
    }

    fn positions(&self, subject: &Subject, frame: &Frame, follow: bool) -> Result<[V; N]> {
        let mut positions = [V::zeros(); N];
        let mut world = [Q::identity(); N];
        positions[0] = if follow {
            V::zeros()
        } else {
            vector(frame.root_position)
        };
        world[0] = rotation(frame.root_orientation_xyzw)?;
        for i in 1..N {
            let parent = self.parents[i].unwrap();
            positions[i] =
                positions[parent] + world[parent] * vector(subject.person.rest_offsets[i]);
            world[i] = Q::new_normalize(
                *(world[parent] * rotation(frame.joint_rotations[i].quaternion_xyzw)?).quaternion(),
            );
        }
        Ok(positions)
    }
    fn time(&self, frame: u64, fps: f64, override_fps: bool) -> f64 {
        if !override_fps {
            if let Some(t) = &self.recording {
                return t.timestamps_seconds[frame as usize];
            }
        }
        frame as f64 / fps
    }
}

fn hand_positions(subject: &Subject, frame: &Frame, body: &[V; N]) -> Result<Vec<[V; 21]>> {
    let mut result = Vec::new();
    for hand in &frame.hands {
        let shape = &subject.person.hands[&hand.side];
        let mut points = [V::zeros(); 21];
        points[0] = body[if hand.side == "left" { 13 } else { 16 }];
        for i in 1..21 {
            points[i] = points[HAND_PARENT[i]]
                + rotation(hand.bone_rotations_xyzw[i])? * V::y() * shape.bone_lengths[i];
        }
        result.push(points);
    }
    Ok(result)
}

fn display_positions(
    record: &Recording,
    subject: &Subject,
    frame: &Frame,
    follow: bool,
    upper: bool,
) -> Result<[V; N]> {
    let mut positions = record.positions(subject, frame, follow)?;
    if follow && upper {
        let chest = positions[8];
        for p in &mut positions {
            *p -= chest;
        }
    }
    Ok(positions)
}

struct Bounds {
    center: V,
    radius: f64,
    floor: f64,
}
fn bounds(
    record: &Recording,
    subjects: &[&Subject],
    follow: bool,
    upper_body: bool,
) -> Result<Bounds> {
    let mut min = V::repeat(f64::INFINITY);
    let mut max = V::repeat(f64::NEG_INFINITY);
    for subject in subjects {
        for frame in &subject.frames {
            for p in &display_positions(record, subject, frame, follow, upper_body)? {
                min = min.inf(p);
                max = max.sup(p);
            }
        }
    }
    Ok(Bounds {
        center: (min + max) / 2.0,
        radius: ((max - min).norm() / 2.0).max(0.1),
        floor: max.y,
    })
}
struct View {
    yaw: f64,
    pitch: f64,
    zoom: f64,
    follow: bool,
    names: bool,
    upper_body: bool,
}
impl Default for View {
    fn default() -> Self {
        Self {
            yaw: -0.55,
            pitch: 0.22,
            zoom: 1.0,
            follow: true,
            names: false,
            upper_body: true,
        }
    }
}

fn render(
    record: &Recording,
    subjects: &[&Subject],
    frame: u64,
    scene: &Bounds,
    view: &View,
    seconds: f64,
    speed: f64,
    paused: bool,
) -> Result<Mat> {
    let size = Size::new(1024, 768);
    let mut image = Mat::new_rows_cols_with_default(
        size.height,
        size.width,
        opencv::core::CV_8UC3,
        Scalar::new(22.0, 18.0, 16.0, 0.0),
    )?;
    let orbit = Rotation3::from_euler_angles(view.pitch, view.yaw, 0.0);
    let distance = scene.radius * 3.5 / view.zoom;
    let project = |p: V| -> Option<Point> {
        let p = orbit * (p - scene.center);
        let depth = p.z + distance;
        if depth <= scene.radius * 0.01 {
            return None;
        }
        let x = 512.0 + 620.0 * p.x / depth;
        let y = 370.0 + 620.0 * p.y / depth;
        (x.is_finite() && y.is_finite() && x.abs() < 100000.0 && y.abs() < 100000.0)
            .then(|| Point::new(x.round() as i32, y.round() as i32))
    };
    // Fixed visual floor under the recording, including the estimated lower-body
    // scaffold. This is a spatial reference, not a measured ground/contact plane.
    let floor_extent = scene.radius * 1.4;
    let corners: Option<Vec<Point>> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .into_iter()
        .map(|(x, z)| {
            project(V::new(
                scene.center.x + x * floor_extent,
                scene.floor,
                scene.center.z + z * floor_extent,
            ))
        })
        .collect();
    if let Some(corners) = corners {
        let polygon = opencv::core::Vector::<Point>::from_iter(corners);
        imgproc::fill_convex_poly(
            &mut image,
            &polygon,
            Scalar::new(36.0, 31.0, 28.0, 0.0),
            imgproc::LINE_AA,
            0,
        )?;
    }
    for i in -7..=7 {
        let offset = i as f64 * scene.radius / 5.0;
        for (a, b) in [
            (
                V::new(
                    scene.center.x - floor_extent,
                    scene.floor,
                    scene.center.z + offset,
                ),
                V::new(
                    scene.center.x + floor_extent,
                    scene.floor,
                    scene.center.z + offset,
                ),
            ),
            (
                V::new(
                    scene.center.x + offset,
                    scene.floor,
                    scene.center.z - floor_extent,
                ),
                V::new(
                    scene.center.x + offset,
                    scene.floor,
                    scene.center.z + floor_extent,
                ),
            ),
        ] {
            if let (Some(a), Some(b)) = (project(a), project(b)) {
                let shade = if i == 0 { 100.0 } else { 62.0 };
                imgproc::line(&mut image, a, b, Scalar::all(shade), 1, imgproc::LINE_AA, 0)?;
            }
        }
    }
    let current = subjects.iter().find_map(|subject| {
        subject
            .frames
            .binary_search_by_key(&frame, |f| f.frame)
            .ok()
            .map(|i| (*subject, &subject.frames[i]))
    });
    if let Some((subject, pose)) = current {
        let positions = display_positions(record, subject, pose, view.follow, view.upper_body)?;
        for i in 1..N {
            let unobserved = view.upper_body
                && (!crate::scope::body_upper(i)
                    || !crate::scope::body_upper(record.parents[i].unwrap()));
            let color = if unobserved {
                Scalar::all(100.0)
            } else if pose.prediction.body {
                Scalar::new(165.0, 165.0, 165.0, 0.0)
            } else if record.joint_names[i].starts_with("left_") {
                Scalar::new(60.0, 180.0, 255.0, 0.0)
            } else if record.joint_names[i].starts_with("right_") {
                Scalar::new(255.0, 205.0, 65.0, 0.0)
            } else {
                Scalar::all(220.0)
            };
            if let (Some(a), Some(b)) = (
                project(positions[record.parents[i].unwrap()]),
                project(positions[i]),
            ) {
                imgproc::line(
                    &mut image,
                    a,
                    b,
                    color,
                    if unobserved { 2 } else { 6 },
                    imgproc::LINE_AA,
                    0,
                )?;
            }
        }
        // Head position follows the 3D body; sparse face fitting supplies orientation when available.
        let orientation = if let Some(head) = &pose.head {
            rotation(head.orientation_xyzw)?
        } else {
            rotation(pose.root_orientation_xyzw)?
                * rotation(pose.joint_rotations[7].quaternion_xyzw)?
                * rotation(pose.joint_rotations[8].quaternion_xyzw)?
                * rotation(pose.joint_rotations[9].quaternion_xyzw)?
        };
        let center = (positions[9] + positions[10]) / 2.0;
        let radius = ((positions[10] - positions[9]).norm() * 0.85).max(0.015);
        let head_point = |p: V| project(center + orientation * p);
        for plane in 0..3 {
            let mut previous = None;
            for step in 0..=32 {
                let angle = step as f64 * std::f64::consts::TAU / 32.0;
                let (a, b) = (angle.cos() * radius, angle.sin() * radius);
                let p = match plane {
                    0 => V::new(a * 0.8, b * 1.1, 0.0),
                    1 => V::new(a * 0.8, 0.0, b * 0.8),
                    _ => V::new(0.0, a * 1.1, b * 0.8),
                };
                let current = head_point(p);
                if let (Some(a), Some(b)) = (previous, current) {
                    imgproc::line(&mut image, a, b, Scalar::all(225.0), 2, imgproc::LINE_AA, 0)?;
                }
                previous = current;
            }
        }
        if let (Some(a), Some(b)) = (project(center), head_point(V::new(0.0, 0.0, -radius * 1.5))) {
            imgproc::line(
                &mut image,
                a,
                b,
                Scalar::new(90.0, 240.0, 90.0, 0.0),
                3,
                imgproc::LINE_AA,
                0,
            )?;
        }
        for hand in hand_positions(subject, pose, &positions)? {
            let color = if pose.prediction.hands {
                Scalar::all(150.0)
            } else {
                Scalar::new(220.0, 90.0, 230.0, 0.0)
            };
            for i in 1..21 {
                if let (Some(a), Some(b)) = (project(hand[HAND_PARENT[i]]), project(hand[i])) {
                    imgproc::line(&mut image, a, b, color, 2, imgproc::LINE_AA, 0)?;
                }
                if let Some(p) = project(hand[i]) {
                    imgproc::circle(&mut image, p, 2, color, -1, imgproc::LINE_AA, 0)?;
                }
            }
            for pair in [0, 5, 9, 13, 17, 0].windows(2) {
                if let (Some(a), Some(b)) = (project(hand[pair[0]]), project(hand[pair[1]])) {
                    imgproc::line(&mut image, a, b, color, 2, imgproc::LINE_AA, 0)?;
                }
            }
        }
        for (i, p) in positions.iter().enumerate() {
            let unobserved = view.upper_body && !crate::scope::body_upper(i);
            if let Some(p) = project(*p) {
                imgproc::circle(
                    &mut image,
                    p,
                    6,
                    Scalar::all(if unobserved { 100.0 } else { 245.0 }),
                    if unobserved { 1 } else { -1 },
                    imgproc::LINE_AA,
                    0,
                )?;
                if view.names {
                    imgproc::put_text(
                        &mut image,
                        &record.joint_names[i],
                        Point::new(p.x + 8, p.y - 6),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.38,
                        Scalar::all(235.0),
                        1,
                        imgproc::LINE_AA,
                        false,
                    )?;
                }
            }
        }
    } else {
        imgproc::put_text(
            &mut image,
            "No reliable pose for this frame",
            Point::new(310, 370),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            Scalar::all(180.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    let mode = if view.follow {
        "in place"
    } else {
        "approximate root motion"
    };
    for (text, y, scale) in [
        (
            format!(
                "Skeletal twin | subject {} | frame {} | {:.2}s",
                current
                    .map(|(s, _)| s.subject_segment.to_string())
                    .unwrap_or_else(|| "none".into()),
                frame,
                seconds
            ),
            30,
            0.65,
        ),
        (
            format!(
                "{} | {:.2}x | {} | {}",
                if paused { "Paused" } else { "Playing" },
                speed,
                mode,
                if current.is_some_and(|(_, p)| p.prediction.body
                    || p.prediction.hands
                    || p.prediction.head)
                {
                    "Predicted motion"
                } else {
                    "Observed fit"
                }
            ),
            58,
            0.5,
        ),
        (
            if view.upper_body {
                "Gray lower body: unobserved estimate"
            } else {
                ""
            }
            .into(),
            82,
            0.45,
        ),
        (
            "Space play/pause | J/L step | A/D orbit | W/S tilt | +/- zoom".into(),
            716,
            0.45,
        ),
        (
            "F root motion | N names | 1/2/3 view | R reset | [/] speed | Q quit".into(),
            743,
            0.45,
        ),
    ] {
        imgproc::put_text(
            &mut image,
            &text,
            Point::new(18, y),
            imgproc::FONT_HERSHEY_SIMPLEX,
            scale,
            Scalar::all(225.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(image)
}

fn playback_range(record: &Recording, subjects: &[&Subject], filtered: bool) -> (u64, u64) {
    let first = if filtered {
        subjects
            .iter()
            .filter_map(|s| s.frames.first().map(|f| f.frame))
            .min()
            .unwrap_or(0)
    } else {
        0
    };
    let last = subjects
        .iter()
        .filter_map(|s| s.frames.last().map(|f| f.frame + 1))
        .max()
        .unwrap_or(0);
    let end = if filtered {
        last
    } else {
        record
            .recording
            .as_ref()
            .map(|t| t.frame_count)
            .unwrap_or(last)
    };
    (first, end)
}

fn q_array(q: Q) -> [f64; 4] {
    let q = Q::new_normalize(*q.quaternion());
    let q = q.quaternion();
    [q.i, q.j, q.k, q.w]
}
fn blend_rotation(a: [f64; 4], b: [f64; 4], amount: f64) -> Result<[f64; 4]> {
    Ok(q_array(rotation(a)?.slerp(&rotation(b)?, amount)))
}
fn bracket<T>(items: &[(u64, T)], frame: u64) -> (usize, usize) {
    let i = items.partition_point(|p| p.0 < frame);
    if i < items.len() && items[i].0 == frame {
        (i, i)
    } else {
        (i.saturating_sub(1), i.min(items.len() - 1))
    }
}
fn amount(record: &Recording, a: u64, b: u64, frame: u64, fps: f64, override_fps: bool) -> f64 {
    if a == b {
        0.0
    } else {
        ((record.time(frame, fps, override_fps) - record.time(a, fps, override_fps))
            / (record.time(b, fps, override_fps) - record.time(a, fps, override_fps)))
        .clamp(0.0, 1.0)
    }
}

/// Offline motion completion. A single shared shape is used throughout playback;
/// rotations are interpolated, never joint coordinates or bone lengths.
fn complete_motion(
    record: &Recording,
    subjects: &[&Subject],
    fps: f64,
    override_fps: bool,
) -> Result<Subject> {
    let reference = subjects.iter().max_by_key(|s| s.frames.len()).unwrap();
    let mut person = reference.person.clone();
    for subject in subjects {
        for (side, shape) in &subject.person.hands {
            if person
                .hands
                .get(side)
                .is_none_or(|old| shape.measurement_count > old.measurement_count)
            {
                person.hands.insert(side.clone(), shape.clone());
            }
        }
    }
    let mut anchors: Vec<_> = subjects
        .iter()
        .flat_map(|s| s.frames.iter().map(|f| (f.frame, f)))
        .collect();
    anchors.sort_by_key(|a| a.0);
    anchors.dedup_by_key(|a| a.0);
    let first = anchors.first().unwrap().0;
    let end = record
        .recording
        .as_ref()
        .map(|t| t.frame_count)
        .unwrap_or(anchors.last().unwrap().0 + 1);
    let hand_anchors: Vec<_> = ["left", "right"]
        .iter()
        .map(|side| {
            anchors
                .iter()
                .filter_map(|(i, f)| f.hands.iter().find(|h| h.side == *side).map(|h| (*i, h)))
                .collect::<Vec<_>>()
        })
        .collect();
    let head_anchors: Vec<_> = anchors
        .iter()
        .filter_map(|(i, f)| f.head.as_ref().map(|h| (*i, h)))
        .collect();
    let mut frames = Vec::new();
    for frame in first..end {
        let (ai, bi) = bracket(&anchors, frame);
        let a = anchors[ai].1;
        let b = anchors[bi].1;
        let t = amount(record, a.frame, b.frame, frame, fps, override_fps);
        let mut result = a.clone();
        result.frame = frame;
        result.prediction = Prediction::default();
        if a.frame != frame {
            result.prediction.body = true;
            result.root_orientation_xyzw =
                blend_rotation(a.root_orientation_xyzw, b.root_orientation_xyzw, t)?;
            for i in 0..N {
                result.joint_rotations[i].quaternion_xyzw = blend_rotation(
                    a.joint_rotations[i].quaternion_xyzw,
                    b.joint_rotations[i].quaternion_xyzw,
                    t,
                )?;
            }
            // Monotone Hermite root motion: surrounding velocities, bounded against overshoot.
            let dt = (record.time(b.frame, fps, override_fps)
                - record.time(a.frame, fps, override_fps))
            .max(1e-6);
            let previous = anchors[ai.saturating_sub(1)].1;
            let next = anchors[(bi + 1).min(anchors.len() - 1)].1;
            for axis in 0..3 {
                let start = a.root_position[axis];
                let finish = b.root_position[axis];
                let slope = (finish - start) / dt;
                let mut v0 = (finish - previous.root_position[axis])
                    / (record.time(b.frame, fps, override_fps)
                        - record.time(previous.frame, fps, override_fps))
                    .max(1e-6);
                let mut v1 = (next.root_position[axis] - start)
                    / (record.time(next.frame, fps, override_fps)
                        - record.time(a.frame, fps, override_fps))
                    .max(1e-6);
                if slope.abs() < 1e-9 {
                    v0 = 0.0;
                    v1 = 0.0;
                } else {
                    let mut alpha = (v0 / slope).max(0.0);
                    let mut beta = (v1 / slope).max(0.0);
                    let norm = alpha.hypot(beta);
                    if norm > 3.0 {
                        alpha *= 3.0 / norm;
                        beta *= 3.0 / norm;
                    }
                    v0 = alpha * slope;
                    v1 = beta * slope;
                }
                result.root_position[axis] = ((2.0 * t.powi(3) - 3.0 * t * t + 1.0) * start
                    + (t.powi(3) - 2.0 * t * t + t) * dt * v0
                    + (-2.0 * t.powi(3) + 3.0 * t * t) * finish
                    + (t.powi(3) - t * t) * dt * v1)
                    .clamp(start.min(finish), start.max(finish));
            }
        }
        result.hands.clear();
        for items in &hand_anchors {
            if items.is_empty() {
                continue;
            }
            let (a, b) = bracket(items, frame);
            let t = amount(record, items[a].0, items[b].0, frame, fps, override_fps);
            let mut hand = items[a].1.clone();
            for i in 0..21 {
                hand.bone_rotations_xyzw[i] = blend_rotation(
                    items[a].1.bone_rotations_xyzw[i],
                    items[b].1.bone_rotations_xyzw[i],
                    t,
                )?;
            }
            result.prediction.hands |= items[a].0 != frame;
            result.hands.push(hand);
        }
        if !head_anchors.is_empty() {
            let (a, b) = bracket(&head_anchors, frame);
            let t = amount(
                record,
                head_anchors[a].0,
                head_anchors[b].0,
                frame,
                fps,
                override_fps,
            );
            let mut head = head_anchors[a].1.clone();
            head.orientation_xyzw = blend_rotation(
                head_anchors[a].1.orientation_xyzw,
                head_anchors[b].1.orientation_xyzw,
                t,
            )?;
            result.head = Some(head);
            result.prediction.head = head_anchors[a].0 != frame;
        }
        frames.push(result);
    }
    // Leading frames have no earlier observation: nearest-pose hold, explicitly predicted.
    if first > 0 {
        let mut leading = Vec::new();
        for frame in 0..first {
            let mut pose = frames[0].clone();
            pose.frame = frame;
            pose.prediction = Prediction {
                body: true,
                hands: !pose.hands.is_empty(),
                head: pose.head.is_some(),
            };
            leading.push(pose);
        }
        leading.extend(frames);
        frames = leading;
    }
    Ok(Subject {
        subject_segment: reference.subject_segment,
        person,
        frames,
    })
}

fn trimmed_range(
    record: &Recording,
    first: u64,
    end: u64,
    fps: f64,
    override_fps: bool,
    start: f64,
    tail: f64,
) -> Result<(u64, u64)> {
    let origin = record.time(first, fps, override_fps);
    let finish = record.time(end - 1, fps, override_fps) + 1.0 / fps;
    ensure!(
        finish - origin > start + tail,
        "Recording is too short for a {start}s start/{tail}s end trim. Record longer or set --trim-start 0 --trim-end 0."
    );
    let begin = (first..end)
        .find(|&f| record.time(f, fps, override_fps) >= origin + start)
        .unwrap_or(end);
    let end = (begin..end)
        .find(|&f| record.time(f, fps, override_fps) >= finish - tail)
        .unwrap_or(end);
    ensure!(end > begin, "Trim leaves no frames");
    Ok((begin, end))
}

pub fn run(path: &Path, options: Options<'_>) -> Result<()> {
    ensure!(
        !options.no_display || options.output.is_some(),
        "Headless replay needs --output replay.mp4"
    );
    if let Some(output) = options.output {
        ensure!(
            output != path
                && !(output.exists() && output.canonicalize()? == path.canonicalize()?),
            "Replay output must differ from twin JSON"
        );
    }
    let record = Recording::load(path)?;
    let subjects: Vec<_> = record
        .subjects
        .iter()
        .filter(|s| options.subject.is_none_or(|id| s.subject_segment == id))
        .filter(|s| !s.frames.is_empty())
        .collect();
    ensure!(
        !subjects.is_empty(),
        "No reliable poses for the requested recording/subject"
    );
    let fps = options
        .fps
        .or(record.recording.as_ref().map(|t| t.fps))
        .unwrap_or(30.0);
    ensure!(
        fps.is_finite() && fps > 0.0 && fps <= 1000.0,
        "Replay FPS must be in (0,1000]"
    );
    let (first, end) = playback_range(&record, &subjects, options.subject.is_some());
    let start = options
        .trim_start
        .or(record.recording.as_ref().map(|t| t.trim_start_seconds))
        .unwrap_or(3.0);
    let tail = options
        .trim_end
        .or(record.recording.as_ref().map(|t| t.trim_end_seconds))
        .unwrap_or(3.0);
    let trim_origin = record.time(first, fps, options.fps.is_some()) + start;
    let trim_finish = record.time(end - 1, fps, options.fps.is_some()) + 1.0 / fps - tail;
    let (first, mut end) =
        trimmed_range(&record, first, end, fps, options.fps.is_some(), start, tail)?;
    let mut continuous = complete_motion(&record, &subjects, fps, options.fps.is_some())?;
    if let Some(limit) = options.max_frames {
        end = end.min(first.saturating_add(limit));
    }
    ensure!(end > first, "Replay timeline is empty");
    continuous
        .frames
        .retain(|f| f.frame >= first && f.frame < end);
    eprintln!(
        "Replay continuity: {} frames, {} interpolated/held body poses",
        continuous.frames.len(),
        continuous
            .frames
            .iter()
            .filter(|f| f.prediction.body)
            .count()
    );
    let subjects = vec![&continuous];
    let upper_body = !options.full_body || record.recording.as_ref().is_some_and(|t| t.upper_body);
    let follow_bounds = bounds(&record, &subjects, true, upper_body)?;
    let world_bounds = bounds(&record, &subjects, false, upper_body)?;
    let default_view = View {
        upper_body,
        ..View::default()
    };
    let time = |f| record.time(f, fps, options.fps.is_some());
    let origin = trim_origin;
    let duration = if options.max_frames.is_some() {
        trim_finish.min(time(end - 1) + 1.0 / fps) - origin
    } else {
        trim_finish - origin
    };
    if let Some(output) = options.output {
        if let Some(source) = record
            .recording
            .as_ref()
            .and_then(|t| t.source_video.as_ref())
        {
            let source = path.parent().unwrap_or(Path::new(".")).join(source);
            ensure!(
                !(source.exists()
                    && output.exists()
                    && source.canonicalize()? == output.canonicalize()?),
                "Replay output must differ from the recorded source video"
            );
        }
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = videoio::VideoWriter::new(
            output,
            videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?,
            fps,
            Size::new(1024, 768),
            true,
        )?;
        ensure!(
            writer.is_opened()?,
            "Cannot create replay video {}",
            output.display()
        );
        let mut cursor = first;
        let count = (duration * fps - 1e-7).ceil().max(1.0) as u64;
        for i in 0..count {
            let seconds = i as f64 / fps;
            while cursor + 1 < end && time(cursor + 1) - origin <= seconds + 1e-9 {
                cursor += 1;
            }
            writer.write(&render(
                &record,
                &subjects,
                cursor,
                &follow_bounds,
                &default_view,
                seconds,
                1.0,
                false,
            )?)?;
        }
        writer.release()?;
        eprintln!("Saved {count} 3D replay frames to {}", output.display());
    }
    if options.no_display {
        return Ok(());
    }
    let mut cursor = first;
    let mut view = default_view;
    let mut paused = false;
    let mut speed = 1.0;
    let mut deadline = Instant::now();
    loop {
        let scene = if view.follow {
            &follow_bounds
        } else {
            &world_bounds
        };
        highgui::imshow(
            "3D human twin replay",
            &render(
                &record,
                &subjects,
                cursor,
                scene,
                &view,
                time(cursor) - origin,
                speed,
                paused,
            )?,
        )?;
        let key = highgui::wait_key_ex(10)?;
        let key = if key >= 0 { key & 0xff } else { -1 };
        match key {
            27 | 113 => break,
            32 => {
                if cursor + 1 >= end {
                    cursor = first;
                }
                paused = !paused;
                deadline = Instant::now();
            }
            106 => {
                cursor = cursor.saturating_sub(1).max(first);
                paused = true;
            }
            108 => {
                cursor = (cursor + 1).min(end - 1);
                paused = true;
            }
            97 => view.yaw -= 0.12,
            100 => view.yaw += 0.12,
            119 => view.pitch = (view.pitch - 0.1).max(-1.4),
            115 => view.pitch = (view.pitch + 0.1).min(1.4),
            43 | 61 => view.zoom = (view.zoom * 1.1).min(3.0),
            45 => view.zoom = (view.zoom / 1.1).max(0.3),
            102 => view.follow = !view.follow,
            110 => view.names = !view.names,
            49 => {
                view.yaw = 0.0;
                view.pitch = 0.0;
            }
            50 => {
                view.yaw = std::f64::consts::FRAC_PI_2;
                view.pitch = 0.0;
            }
            51 => {
                view.yaw = -0.55;
                view.pitch = 0.12;
            }
            114 => {
                view = View {
                    upper_body,
                    ..View::default()
                }
            }
            91 => {
                speed = (speed / 2.0).max(0.25);
                deadline = Instant::now();
            }
            93 => {
                speed = (speed * 2.0).min(4.0);
                deadline = Instant::now();
            }
            _ => {}
        }
        while !paused {
            let delay = if cursor + 1 < end {
                (time(cursor + 1) - time(cursor)) / speed
            } else {
                1.0 / fps / speed
            };
            if Instant::now() < deadline + Duration::from_secs_f64(delay.max(0.001)) {
                break;
            }
            {
                deadline += Duration::from_secs_f64(delay.max(0.001));
                if cursor + 1 < end {
                    cursor += 1;
                } else {
                    paused = true;
                }
            }
        }
    }
    highgui::destroy_all_windows()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_fk_preserves_shared_lengths_and_root_transform() {
        let parents = vec![None, Some(0), Some(1)];
        let mut record = Recording {
            schema_version: 1,
            quaternion_convention: "xyzw_active_parent_local".into(),
            joint_names: (0..17).map(|i| format!("j{i}")).collect(),
            parents,
            recording: None,
            subjects: Vec::new(),
        };
        while record.parents.len() < 17 {
            record.parents.push(Some(0));
        }
        let mut offsets = vec![[0.0, 1.0, 0.0]; 17];
        offsets[0] = [0.0; 3];
        let subject = Subject {
            subject_segment: 0,
            person: Person {
                rest_offsets: offsets,
                hands: BTreeMap::new(),
            },
            frames: Vec::new(),
        };
        let mut frame = Frame {
            prediction: Prediction::default(),
            frame: 0,
            root_position: [2.0, 3.0, 4.0],
            root_orientation_xyzw: [0.0, 0.0, 0.0, 1.0],
            joint_rotations: (0..17)
                .map(|i| JointRotation {
                    joint: format!("j{i}"),
                    quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
                })
                .collect(),
            hands: Vec::new(),
            head: None,
        };
        frame.joint_rotations[1].quaternion_xyzw = [0.0, 0.0, 0.5_f64.sqrt(), 0.5_f64.sqrt()];
        let positions = record.positions(&subject, &frame, false).unwrap();
        assert_eq!(positions[0], V::new(2.0, 3.0, 4.0));
        assert!(((positions[2] - positions[1]).norm() - 1.0).abs() < 1e-12);
        assert!((positions[2].x - 1.0).abs() < 1e-12);
        assert_eq!(
            record.positions(&subject, &frame, true).unwrap()[0],
            V::zeros()
        );
        assert!(rotation([0.0; 4]).is_err());
    }
    #[test]
    fn recorded_timing_is_preserved_and_can_be_overridden() {
        let record = Recording {
            schema_version: 1,
            quaternion_convention: "xyzw_active_parent_local".into(),
            joint_names: Vec::new(),
            parents: Vec::new(),
            recording: Some(Timing {
                fps: 30.0,
                frame_count: 3,
                timestamps_seconds: vec![0.1, 0.2, 0.35],
                source_video: None,
                trim_start_seconds: 3.0,
                trim_end_seconds: 3.0,
                upper_body: false,
            }),
            subjects: Vec::new(),
        };
        assert_eq!(record.time(2, 30.0, false), 0.35);
        assert_eq!(record.time(2, 10.0, true), 0.2);
    }

    #[test]
    fn loader_validates_hierarchy_rotations_and_timeline() {
        let path =
            std::env::temp_dir().join(format!("lifting-replay-loader-{}.json", std::process::id()));
        let names: Vec<_> = (0..N).map(|i| format!("j{i}")).collect();
        let mut value = serde_json::json!({
            "schema_version": 1, "quaternion_convention": "xyzw_active_parent_local",
            "joint_names": names, "parents": (0..N).map(|i| if i==0 {None}else{Some(0)}).collect::<Vec<_>>(),
            "recording": {"fps": 15.0,"frame_count": 1,"timestamps_seconds": [0.0]},
            "subjects": [{"subject_segment": 0,"person": {"rest_offsets": vec![[0.0,1.0,0.0];N]},
                "frames": [{"frame":0,"root_position":[0.0,0.0,0.0],"root_orientation_xyzw":[0.0,0.0,0.0,1.0],
                    "joint_rotations": (0..N).map(|i| serde_json::json!({"joint":format!("j{i}"),"quaternion_xyzw":[0.0,0.0,0.0,1.0]})).collect::<Vec<_>>() }]}]
        });
        let write = |value: &serde_json::Value| {
            std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap()
        };
        write(&value);
        assert!(Recording::load(&path).is_ok());
        value["parents"][1] = serde_json::json!(2);
        write(&value);
        assert!(Recording::load(&path).is_err());
        value["parents"][1] = serde_json::json!(0);
        value["subjects"][0]["frames"][0]["root_orientation_xyzw"] =
            serde_json::json!([0.0, 0.0, 0.0, 0.0]);
        write(&value);
        assert!(Recording::load(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn default_replay_keeps_full_capture_even_when_tracking_is_split() {
        let frame = |frame| Frame {
            prediction: Prediction::default(),
            frame,
            root_position: [0.0; 3],
            root_orientation_xyzw: [0.0, 0.0, 0.0, 1.0],
            joint_rotations: (0..17)
                .map(|i| JointRotation {
                    joint: format!("j{i}"),
                    quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
                })
                .collect(),
            hands: Vec::new(),
            head: None,
        };
        let subject = |id, a, b| Subject {
            subject_segment: id,
            person: Person {
                rest_offsets: vec![[0.0, 1.0, 0.0]; 17],
                hands: BTreeMap::new(),
            },
            frames: vec![frame(a), frame(b)],
        };
        let record = Recording {
            schema_version: 1,
            quaternion_convention: "xyzw_active_parent_local".into(),
            joint_names: (0..17).map(|i| format!("j{i}")).collect(),
            parents: (0..17)
                .map(|i| if i == 0 { None } else { Some(0) })
                .collect(),
            recording: Some(Timing {
                fps: 30.0,
                frame_count: 333,
                timestamps_seconds: (0..333).map(|i| i as f64 * 11.795 / 332.0).collect(),
                source_video: None,
                trim_start_seconds: 3.0,
                trim_end_seconds: 3.0,
                upper_body: false,
            }),
            subjects: vec![
                subject(0, 0, 81),
                subject(1, 120, 191),
                subject(2, 292, 332),
            ],
        };
        let selected: Vec<_> = record.subjects.iter().collect();
        assert_eq!(playback_range(&record, &selected, false), (0, 333));
        assert_eq!(playback_range(&record, &selected[..1], true), (0, 82));
        assert!(record.time(332, 30.0, false) + 1.0 / 30.0 > 11.8);
    }
    #[test]
    fn hand_replay_uses_shared_lengths_and_attaches_to_body_wrist() {
        let mut shapes = BTreeMap::new();
        shapes.insert(
            "left".into(),
            HandShape {
                bone_lengths: vec![0.02; 21],
                measurement_count: 20,
            },
        );
        let subject = Subject {
            subject_segment: 0,
            person: Person {
                rest_offsets: Vec::new(),
                hands: shapes,
            },
            frames: Vec::new(),
        };
        let q = Q::from_euler_angles(0.6, 0.0, 0.0);
        let q = q.quaternion();
        let frame = Frame {
            prediction: Prediction::default(),
            frame: 0,
            root_position: [0.0; 3],
            root_orientation_xyzw: [0.0, 0.0, 0.0, 1.0],
            joint_rotations: Vec::new(),
            head: None,
            hands: vec![HandFrame {
                side: "left".into(),
                confidence: 0.9,
                bone_rotations_xyzw: vec![[q.i, q.j, q.k, q.w]; 21],
            }],
        };
        let body = [V::new(1.0, 2.0, 3.0); 17];
        let hands = hand_positions(&subject, &frame, &body).unwrap();
        assert_eq!(hands[0][0], body[13]);
        for i in 1..21 {
            assert!(((hands[0][i] - hands[0][HAND_PARENT[i]]).norm() - 0.02).abs() < 1e-12);
        }
        assert!(hands[0][4].z > hands[0][0].z);
    }
    fn continuity_fixture() -> Recording {
        let frame = |frame, angle| Frame {
            prediction: Prediction::default(),
            frame,
            root_position: [frame as f64 / 10.0, 0.0, 0.0],
            root_orientation_xyzw: q_array(Q::from_euler_angles(0.0, angle, 0.0)),
            joint_rotations: (0..17)
                .map(|i| JointRotation {
                    joint: format!("j{i}"),
                    quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
                })
                .collect(),
            hands: Vec::new(),
            head: None,
        };
        let subject = |id, length, frames| Subject {
            subject_segment: id,
            person: Person {
                rest_offsets: vec![[0.0, length, 0.0]; 17],
                hands: BTreeMap::new(),
            },
            frames,
        };
        Recording {
            schema_version: 1,
            quaternion_convention: "xyzw_active_parent_local".into(),
            joint_names: (0..17).map(|i| format!("j{i}")).collect(),
            parents: (0..17)
                .map(|i| if i == 0 { None } else { Some(0) })
                .collect(),
            recording: Some(Timing {
                fps: 10.0,
                frame_count: 100,
                timestamps_seconds: (0..100)
                    .map(|i| {
                        if (10..=20).contains(&i) {
                            1.0 + ((i - 10) as f64 / 10.0).powi(2)
                        } else {
                            i as f64 / 10.0
                        }
                    })
                    .collect(),
                source_video: None,
                trim_start_seconds: 3.0,
                trim_end_seconds: 3.0,
                upper_body: false,
            }),
            subjects: vec![
                subject(0, 1.0, vec![frame(0, 0.0), frame(10, 0.0)]),
                subject(1, 2.0, vec![frame(20, std::f64::consts::FRAC_PI_2)]),
            ],
        }
    }
    #[test]
    fn motion_completion_fills_all_gaps_using_time_and_preserves_one_body_shape() {
        let record = continuity_fixture();
        let selected: Vec<_> = record.subjects.iter().collect();
        let completed = complete_motion(&record, &selected, 10.0, false).unwrap();
        assert_eq!(completed.frames.len(), 100);
        let middle = &completed.frames[15];
        assert!(middle.prediction.body);
        assert!((middle.root_position[0] - 1.25).abs() < 1e-9);
        assert!(
            (rotation(middle.root_orientation_xyzw).unwrap().angle()
                - std::f64::consts::FRAC_PI_2 * 0.25)
                .abs()
                < 1e-9
        );
        assert!(!completed.frames[10].prediction.body);
        assert_eq!(
            completed.frames[99].root_position,
            completed.frames[20].root_position
        );
        for frame in &completed.frames {
            let positions = record.positions(&completed, frame, false).unwrap();
            for i in 1..17 {
                assert!(((positions[i] - positions[0]).norm() - 1.0).abs() < 1e-12);
            }
            assert!((0.0..=2.0).contains(&frame.root_position[0]));
        }
    }
    #[test]
    fn trim_removes_three_seconds_at_each_end_and_rejects_empty_clips() {
        let record = continuity_fixture();
        assert_eq!(
            trimmed_range(&record, 0, 100, 10.0, false, 3.0, 3.0).unwrap(),
            (30, 70)
        );
        assert!(trimmed_range(&record, 0, 50, 10.0, false, 3.0, 3.0).is_err());
        assert_eq!(
            trimmed_range(&record, 0, 50, 10.0, false, 0.0, 0.0).unwrap(),
            (0, 50)
        );
    }
    #[test]
    fn independent_hand_and_head_gaps_are_interpolated_not_dropped() {
        let mut record = continuity_fixture();
        for subject in &mut record.subjects {
            subject.person.hands.insert(
                "left".into(),
                HandShape {
                    bone_lengths: vec![0.02; 21],
                    measurement_count: 10,
                },
            );
        }
        for (subject, index, angle) in [(0, 0, 0.0), (1, 0, 1.0)] {
            let q = q_array(Q::from_euler_angles(angle, 0.0, 0.0));
            record.subjects[subject].frames[index].hands = vec![HandFrame {
                side: "left".into(),
                confidence: 0.9,
                bone_rotations_xyzw: vec![q; 21],
            }];
            record.subjects[subject].frames[index].head = Some(HeadPose {
                orientation_xyzw: q,
                confidence: 0.9,
                source: "test".into(),
            });
        }
        let selected: Vec<_> = record.subjects.iter().collect();
        let completed = complete_motion(&record, &selected, 10.0, false).unwrap();
        let frame = &completed.frames[10];
        assert!(!frame.prediction.body && frame.prediction.hands && frame.prediction.head);
        assert_eq!(frame.hands.len(), 1);
        assert!(
            (rotation(frame.head.as_ref().unwrap().orientation_xyzw)
                .unwrap()
                .angle()
                - 0.5)
                .abs()
                < 1e-9
        );
        let body = record.positions(&completed, frame, true).unwrap();
        let hands = hand_positions(&completed, frame, &body).unwrap();
        for i in 1..21 {
            assert!(((hands[0][i] - hands[0][HAND_PARENT[i]]).norm() - 0.02).abs() < 1e-12);
        }
    }
    #[test]
    fn upper_body_camera_bounds_include_unobserved_skeleton() {
        let record = continuity_fixture();
        let selected: Vec<_> = record.subjects.iter().collect();
        let bounds = bounds(&record, &selected, true, true).unwrap();
        for subject in selected {
            for frame in &subject.frames {
                for p in display_positions(&record, subject, frame, true, true).unwrap() {
                    assert!((p - bounds.center).norm() <= bounds.radius + 1e-9);
                }
            }
        }
    }
    #[test]
    fn upper_body_follow_view_anchors_at_chest_without_distorting_arm_lengths() {
        let record = continuity_fixture();
        let subject = &record.subjects[0];
        let frame = &subject.frames[0];
        let original = record.positions(subject, frame, true).unwrap();
        let visible = display_positions(&record, subject, frame, true, true).unwrap();
        assert_eq!(visible[8], V::zeros());
        for i in 8..17 {
            assert!(
                ((visible[i] - visible[8]).norm() - (original[i] - original[8]).norm()).abs()
                    < 1e-12
            );
        }
    }
}
