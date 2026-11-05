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
#[derive(Deserialize)]
struct Recording {
    schema_version: u32,
    quaternion_convention: String,
    joint_names: Vec<String>,
    parents: Vec<Option<usize>>,
    #[serde(default)]
    recording: Option<Timing>,
    subjects: Vec<Subject>,
}
#[derive(Deserialize)]
struct Timing {
    fps: f64,
    frame_count: u64,
    timestamps_seconds: Vec<f64>,
    #[serde(default)]
    source_video: Option<String>,
}
#[derive(Deserialize)]
struct Subject {
    subject_segment: usize,
    person: Person,
    frames: Vec<Frame>,
}
#[derive(Deserialize)]
struct Person {
    rest_offsets: Vec<[f64; 3]>,
    #[serde(default)]
    hands: BTreeMap<String, HandShape>,
}
#[derive(Deserialize)]
struct Frame {
    frame: u64,
    root_position: [f64; 3],
    root_orientation_xyzw: [f64; 4],
    joint_rotations: Vec<JointRotation>,
    #[serde(default)]
    hands: Vec<HandFrame>,
    #[serde(default)]
    head: Option<HeadPose>,
}
#[derive(Deserialize)]
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

struct Bounds {
    center: V,
    radius: f64,
    floor: f64,
}
fn bounds(record: &Recording, subjects: &[&Subject], follow: bool) -> Result<Bounds> {
    let mut min = V::repeat(f64::INFINITY);
    let mut max = V::repeat(f64::NEG_INFINITY);
    for subject in subjects {
        for frame in &subject.frames {
            for p in record.positions(subject, frame, follow)? {
                min = min.inf(&p);
                max = max.sup(&p);
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
}
impl Default for View {
    fn default() -> Self {
        Self {
            yaw: -0.55,
            pitch: 0.12,
            zoom: 1.0,
            follow: true,
            names: false,
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
    // A visual reference plane; no physical ground/contact estimate is implied.
    for i in -5..=5 {
        let offset = i as f64 * scene.radius / 5.0;
        for (a, b) in [
            (
                V::new(
                    scene.center.x - scene.radius,
                    scene.floor,
                    scene.center.z + offset,
                ),
                V::new(
                    scene.center.x + scene.radius,
                    scene.floor,
                    scene.center.z + offset,
                ),
            ),
            (
                V::new(
                    scene.center.x + offset,
                    scene.floor,
                    scene.center.z - scene.radius,
                ),
                V::new(
                    scene.center.x + offset,
                    scene.floor,
                    scene.center.z + scene.radius,
                ),
            ),
        ] {
            if let (Some(a), Some(b)) = (project(a), project(b)) {
                imgproc::line(&mut image, a, b, Scalar::all(48.0), 1, imgproc::LINE_AA, 0)?;
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
        let positions = record.positions(subject, pose, view.follow)?;
        for i in 1..N {
            let color = if record.joint_names[i].starts_with("left_") {
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
                imgproc::line(&mut image, a, b, color, 6, imgproc::LINE_AA, 0)?;
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
            let color = Scalar::new(220.0, 90.0, 230.0, 0.0);
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
            if let Some(p) = project(*p) {
                imgproc::circle(
                    &mut image,
                    p,
                    6,
                    Scalar::all(245.0),
                    -1,
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
                "{} | {:.2}x | {}",
                if paused { "Paused" } else { "Playing" },
                speed,
                mode
            ),
            58,
            0.5,
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
    let (first, mut end) = playback_range(&record, &subjects, options.subject.is_some());
    if let Some(limit) = options.max_frames {
        end = end.min(first.saturating_add(limit));
    }
    ensure!(end > first, "Replay timeline is empty");
    let follow_bounds = bounds(&record, &subjects, true)?;
    let world_bounds = bounds(&record, &subjects, false)?;
    let time = |f| record.time(f, fps, options.fps.is_some());
    let origin = time(first);
    let duration = time(end - 1) - origin + 1.0 / fps;
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
                &View::default(),
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
    let mut view = View::default();
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
            114 => view = View::default(),
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
}
