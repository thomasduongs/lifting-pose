mod angles;
mod bundle;
mod details;
mod detection_cache;
mod detector;
mod landmarks;
mod live_preview;
mod mediapipe;
mod motion3d;
mod palm;
mod pose;
mod recording;
mod replay;
mod scope;
mod tracking;
mod twin;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use detection_cache::DetectionCache;
use detector::PersonDetector;
use motion3d::MotionLifter;
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, Vector},
    highgui, imgcodecs, imgproc,
    prelude::*,
    videoio,
};
use pose::{Bbox, Keypoint, PoseEstimator};
use serde::Serialize;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Closest-person body, hand and foot estimation with OpenCV (CPU)"
)]
struct Args {
    /// Restore full-body tracking; default focuses on the visible upper body.
    #[arg(long)]
    full_body: bool,
    /// Allow torso lean while fitting new poses; the default assumes an upright spine.
    #[arg(long)]
    free_torso: bool,
    /// Record into one reusable save file (default: output/recording.twin).
    #[arg(long, num_args = 0..=1, default_missing_value = bundle::DEFAULT_SAVE, conflicts_with_all = ["replay", "no_3d", "output"])]
    record: Option<PathBuf>,
    /// Remove this many seconds from the start (default: 3).
    #[arg(long, value_parser = parse_nonnegative)]
    trim_start: Option<f64>,
    /// Remove this many seconds from the end (default: 3).
    #[arg(long, value_parser = parse_nonnegative)]
    trim_end: Option<f64>,
    /// Stop camera recording after this many seconds; otherwise q / Esc stops it.
    #[arg(long, requires = "record", value_parser = parse_positive)]
    record_seconds: Option<f64>,
    /// Replay the reusable save (or legacy twin JSON); overwrites output/replay.mp4.
    #[arg(long, num_args = 0..=1, default_missing_value = bundle::DEFAULT_SAVE, conflicts_with_all = ["record", "input", "camera"])]
    replay: Option<PathBuf>,
    /// Replay only this subject segment; default replays the full recording.
    #[arg(long)]
    replay_subject: Option<usize>,
    /// Override recorded replay timing with a constant frame rate.
    #[arg(long, requires = "replay", value_parser = parse_positive)]
    replay_fps: Option<f64>,
    /// Dedicated 21-point 3D hand model (OpenCV Zoo MediaPipe).
    #[arg(long, default_value = "models/handpose.onnx")]
    hand_model: PathBuf,
    #[arg(long)]
    no_hands: bool,
    /// Opt in to sparse-face head orientation; head position is tracked without it.
    #[arg(long)]
    head_orientation: bool,
    #[arg(long, default_value = "models/mediapipe-pose.onnx")]
    model: PathBuf,
    /// MotionAGFormer-XS H36M ONNX export for 3D body lifting.
    #[arg(long, default_value = "models/motionagformer-xs.onnx")]
    motion_model: PathBuf,
    /// Disable 3D lifting and use only 2D pose/angles.
    #[arg(long)]
    no_3d: bool,
    /// Disable dimension-based landmark stabilization and automatic crop following.
    #[arg(long)]
    no_dimension_tracking: bool,
    /// Final skeletal twin recording (one shared shape per subject segment).
    #[arg(long, default_value = "output/twin.json")]
    twin_output: PathBuf,
    /// Calibrated camera focal length in pixels; otherwise frame width is assumed.
    #[arg(long, value_parser = parse_positive)]
    focal_length_px: Option<f64>,
    /// YOLOX-s detector (OpenCV Zoo raw 640x640 export).
    #[arg(long, default_value = "models/yolox-s.onnx")]
    detector_model: PathBuf,
    /// Skip detection and estimate one full-frame pose.
    #[arg(long, conflicts_with = "bbox")]
    no_detection: bool,
    /// Run person detection every N frames; reuse the selected crop between refreshes.
    #[arg(long, default_value_t = 15, value_parser = clap::value_parser!(u64).range(1..))]
    detection_interval: u64,
    /// Minimum person detection confidence (objectness × class score).
    #[arg(long, default_value_t = 0.4, value_parser = parse_probability)]
    detection_threshold: f32,
    /// IoU threshold for suppressing duplicate person boxes.
    #[arg(long, default_value_t = 0.45, value_parser = parse_probability)]
    nms_threshold: f32,
    /// Image or video path. If omitted, open a camera.
    #[arg(long, conflicts_with = "camera")]
    input: Option<PathBuf>,
    /// Camera index (default: 0).
    #[arg(long)]
    camera: Option<i32>,
    /// Annotated image or video destination (.mp4 uses mp4v).
    #[arg(long)]
    output: Option<PathBuf>,
    /// Write one JSON object per frame, including all model keypoints (133 with RTMW-m).
    #[arg(long)]
    json: Option<PathBuf>,
    /// Manual person crop: x,y,width,height. Bypasses automatic detection.
    #[arg(long, value_parser = parse_bbox)]
    bbox: Option<Bbox>,
    #[arg(long, default_value_t = 0.3, value_parser = parse_threshold)]
    threshold: f32,
    /// Disable preview window (useful on servers).
    #[arg(long)]
    no_display: bool,
    /// Stop after this many frames.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    max_frames: Option<u64>,
}

fn parse_nonnegative(value: &str) -> std::result::Result<f64, String> {
    let value: f64 = value.parse().map_err(|_| "Expected a nonnegative number")?;
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err("Value must be finite and nonnegative".into())
    }
}

fn parse_positive(value: &str) -> std::result::Result<f64, String> {
    let value: f64 = value.parse().map_err(|_| "Expected a positive number")?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("Value must be finite and positive".into())
    }
}

fn parse_probability(value: &str) -> std::result::Result<f32, String> {
    let value: f32 = value
        .parse()
        .map_err(|_| "Expected a numeric probability")?;
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        Err("Value must be greater than 0 and at most 1".into())
    }
}

fn parse_threshold(value: &str) -> std::result::Result<f32, String> {
    let value: f32 = value.parse().map_err(|_| "Expected a numeric threshold")?;
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err("Threshold must be finite and nonnegative".into())
    }
}

fn parse_bbox(value: &str) -> std::result::Result<Bbox, String> {
    let values: Vec<f32> = value
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| "Expected x,y,width,height")?;
    if values.len() != 4
        || !values.iter().all(|x| x.is_finite())
        || values[0] < 0.0
        || values[1] < 0.0
        || values[2] <= 0.0
        || values[3] <= 0.0
    {
        return Err("Expected nonnegative x,y and positive width,height, all finite".into());
    }
    Ok(Bbox {
        x: values[0],
        y: values[1],
        width: values[2],
        height: values[3],
    })
}

#[derive(Serialize)]
struct PersonPose {
    /// Per-frame index, not a persistent track ID.
    person_index: usize,
    bbox: Bbox,
    /// Null for manual/full-frame crops.
    detection_score: Option<f32>,
    keypoints: Vec<Keypoint>,
    angles: Vec<angles::JointAngle>,
    pose3d: Option<motion3d::Pose3d>,
    twin: Option<twin::TwinPreview>,
    tracking: Option<tracking::TrackingResult>,
    hands: Vec<details::HandPose>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    predicted_hands: Vec<details::HandPose>,
    head: Option<details::HeadPose>,
}

#[derive(Serialize)]
struct FramePose<'a> {
    frame: u64,
    width: i32,
    height: i32,
    inference_ms: f64,
    detection_ran: bool,
    people: &'a [PersonPose],
}

fn process(
    frame: &mut Mat,
    estimator: &mut PoseEstimator,
    detector: &mut Option<PersonDetector>,
    detection_cache: &mut DetectionCache,
    lifter: &mut Option<MotionLifter>,
    hand_model: &mut Option<details::HandModel>,
    twin_recorder: &mut twin::TwinRecorder,
    dimension_tracker: &mut tracking::DimensionTracker,
    args: &Args,
    index: u64,
    time_seconds: f64,
    json: &mut Option<BufWriter<File>>,
) -> Result<()> {
    let start = Instant::now();
    let mut detection_ran = false;
    let crops = if let Some(detector) = detector {
        detection_ran = detection_cache.update(index, frame.size()?, || {
            detector.detect(frame, args.detection_threshold, args.nms_threshold)
        })?;
        detection_cache
            .detections()
            .iter()
            .map(|d| (d.bbox, Some(d.score)))
            .collect::<Vec<_>>()
    } else {
        let bbox = args.bbox.unwrap_or(Bbox {
            x: 0.0,
            y: 0.0,
            width: frame.cols() as f32,
            height: frame.rows() as f32,
        });
        ensure!(
            bbox.x + bbox.width <= frame.cols() as f32
                && bbox.y + bbox.height <= frame.rows() as f32,
            "Crop extends outside the {}x{} frame",
            frame.cols(),
            frame.rows()
        );
        vec![(bbox, None)]
    };
    if crops.is_empty() {
        if let Some(lifter) = lifter.as_mut() {
            lifter.reset();
        }
    }
    // Infer every crop from the original frame before adding any annotations.
    let mut people = Vec::with_capacity(crops.len());
    for (person_index, (bbox, detection_score)) in crops.into_iter().enumerate() {
        let mut keypoints = estimator.infer(frame, bbox)?;
        scope::mask_details(&mut keypoints);
        if !args.full_body {
            scope::mask_lower(&mut keypoints);
        }
        let body_wrists = [keypoints[9], keypoints[10]];
        let hands = match hand_model.as_mut() {
            Some(model) => model.infer(frame, &keypoints, time_seconds)?,
            None => Vec::new(),
        };
        let predicted_hands = hand_model
            .as_ref()
            .map(|model| model.predictions(time_seconds))
            .unwrap_or_default();
        if estimator.is_mediapipe() {
            details::merge_hand_landmarks(&mut keypoints, &hands);
        }
        let head = if args.head_orientation {
            details::head_pose(&keypoints, args.threshold)
        } else {
            None
        };
        let angles = if args.full_body {
            angles::calculate(&keypoints, args.threshold, frame.size()?)
        } else {
            angles::calculate_scoped(&keypoints, args.threshold, frame.size()?, true)
        };
        let mut pose3d = if args.no_3d {
            None
        } else if estimator.is_mediapipe() {
            estimator.take_pose3d()
        } else {
            match lifter.as_mut() {
                Some(lifter) => lifter.infer(&keypoints, bbox, frame.size()?, args.threshold)?,
                None => None,
            }
        };
        if estimator.is_mediapipe() {
            if let Some(pose) = &mut pose3d {
                let width =
                    (keypoints[5].x - keypoints[6].x).hypot(keypoints[5].y - keypoints[6].y);
                for hand in &hands {
                    let side = if hand.side == "left" { 0 } else { 1 };
                    let old = body_wrists[side];
                    let wrist = hand.image_landmarks[0];
                    if (old.x - wrist[0]).hypot(old.y - wrist[1]) > (width * 0.3).max(40.0) {
                        // The hand detector has located the wrist elsewhere. Keep
                        // its reliable 2D constraint without trusting stale body depth.
                        pose.joints[if side == 0 { 13 } else { 16 }].input_score = 0.0;
                    }
                }
            }
        }
        let twin = pose3d.as_ref().and_then(|pose| {
            twin_recorder.observe(
                index,
                time_seconds,
                pose,
                &keypoints,
                &hands,
                head.clone(),
                bbox,
                Size::new(frame.cols(), frame.rows()),
                args.threshold,
                args.focal_length_px,
            )
        });
        let tracking = if args.no_dimension_tracking || args.no_3d || estimator.is_mediapipe() {
            None
        } else {
            Some(dimension_tracker.refine(
                &keypoints,
                twin.as_ref(),
                bbox,
                frame.size()?,
                args.threshold,
            ))
        };
        if detector.is_some() {
            if let Some(next_crop) = tracking.as_ref().and_then(|t| t.next_crop) {
                detection_cache.follow(next_crop);
            }
        }
        people.push(PersonPose {
            person_index,
            bbox,
            detection_score,
            keypoints,
            angles,
            pose3d,
            twin,
            tracking,
            hands,
            predicted_hands,
            head,
        });
    }
    let inference_ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(writer) = json {
        serde_json::to_writer(
            &mut *writer,
            &FramePose {
                frame: index,
                width: frame.cols(),
                height: frame.rows(),
                inference_ms,
                detection_ran,
                people: &people,
            },
        )?;
        writeln!(writer)?;
    }
    if people.is_empty() {
        if let Some(model) = hand_model.as_ref() {
            details::draw_predicted_hands(frame, &model.predictions(time_seconds))?;
        }
    }
    for person in &people {
        if let Some(tracking) = &person.tracking {
            tracking::draw(frame, tracking, args.threshold)?;
        } else {
            pose::draw(frame, &person.keypoints, args.threshold)?;
        }
        details::draw_hands(frame, &person.hands)?;
        details::draw_predicted_hands(frame, &person.predicted_hands)?;
        let b = person.bbox;
        let left = b.x.floor() as i32;
        let top = b.y.floor() as i32;
        let right = (b.x + b.width).ceil().min(frame.cols() as f32) as i32;
        let bottom = (b.y + b.height).ceil().min(frame.rows() as f32) as i32;
        let color = Scalar::new(255.0, 180.0, 40.0, 0.0);
        imgproc::rectangle(
            frame,
            Rect::new(left, top, right - left, bottom - top),
            color,
            2,
            imgproc::LINE_AA,
            0,
        )?;
        let label = match person.detection_score {
            Some(score) => format!("person {} {:.0}%", person.person_index, score * 100.0),
            None => format!("person {}", person.person_index),
        };
        imgproc::put_text(
            frame,
            &label,
            Point::new(left, (top - 6).max(16)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    for person in &people {
        let display_points = person
            .tracking
            .as_ref()
            .map(|t| &t.keypoints)
            .unwrap_or(&person.keypoints);
        let display_angles = person
            .tracking
            .as_ref()
            .map(|t| &t.angles)
            .unwrap_or(&person.angles);
        let display_pose = person
            .twin
            .as_ref()
            .map(|twin| &twin.fitted_pose)
            .or(person.pose3d.as_ref());
        if let Some(pose3d) = display_pose {
            // The 3D model has no finger/toe landmarks; keep wrist/ankle labels in 2D.
            let remaining: Vec<_> = display_angles
                .iter()
                .filter(|a| !pose3d.angles.iter().any(|b| b.joint == a.joint))
                .cloned()
                .collect();
            angles::draw(frame, display_points, &remaining)?;
            angles::draw_labeled(frame, display_points, &pose3d.angles, "3D")?;
            motion3d::draw(frame, pose3d)?;
        } else {
            angles::draw(frame, display_points, display_angles)?;
        }
    }
    if index == 0 || index.is_multiple_of(30) {
        eprintln!(
            "Frame {index}: {} people, {inference_ms:.1} ms",
            people.len()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    if let Some(path) = &args.replay {
        let output = args
            .output
            .clone()
            .unwrap_or_else(|| PathBuf::from(bundle::DEFAULT_REPLAY));
        ensure!(
            output != *path
                && !(output.exists() && output.canonicalize()? == path.canonicalize()?),
            "Replay output must differ from recording save"
        );
        let unpacked = if bundle::is_bundle(path)? {
            Some(bundle::open_for_replay(path)?)
        } else {
            None
        };
        let twin = unpacked
            .as_ref()
            .map(|(_, p)| p.as_path())
            .unwrap_or(path.as_path());
        return replay::run(
            twin,
            replay::Options {
                output: Some(&output),
                no_display: args.no_display,
                subject: args.replay_subject,
                fps: args.replay_fps,
                max_frames: args.max_frames,
                trim_start: args.trim_start,
                trim_end: args.trim_end,
                full_body: args.full_body,
            },
        );
    }
    let mut estimator = PoseEstimator::new(&args.model)?;
    estimator.set_upper_body(!args.full_body);
    let session_workspace = if args.record.is_some() {
        Some(tempfile::tempdir()?)
    } else {
        None
    };
    let captured = if let Some(target) = &args.record {
        ensure!(
            !target.is_dir(),
            "--record now takes one .twin save file. Use --record by itself for output/recording.twin."
        );
        for input in args.input.iter().chain([
            &args.model,
            &args.motion_model,
            &args.hand_model,
            &args.detector_model,
        ]) {
            ensure!(
                !(target.exists()
                    && input.exists()
                    && target.canonicalize()? == input.canonicalize()?),
                "Recording save must differ from input/model files"
            );
        }
        let directory = session_workspace.as_ref().unwrap().path();
        for path in std::iter::once(&args.model)
            .chain((!estimator.is_mediapipe() && !args.no_3d).then_some(&args.motion_model))
        {
            ensure!(
                path.is_file(),
                "Model not found: {}. Set up models before recording.",
                path.display()
            );
        }
        if !args.no_hands {
            ensure!(
                args.hand_model.is_file(),
                "Hand model missing: {}",
                args.hand_model.display()
            );
            ensure!(
                args.hand_model.with_file_name("palmdet.onnx").is_file(),
                "Palm model missing beside hand model: palmdet.onnx"
            );
        }
        if !args.no_detection && args.bbox.is_none() {
            ensure!(
                args.detector_model.is_file(),
                "Person detector not found: {}",
                args.detector_model.display()
            );
        }
        let mut preview = if args.no_display {
            None
        } else {
            Some(live_preview::LivePreview::new(
                &args.model,
                (!args.no_hands).then_some(args.hand_model.as_path()),
                if args.no_detection || args.bbox.is_some() {
                    None
                } else {
                    Some((
                        args.detector_model.as_path(),
                        args.detection_threshold,
                        args.nms_threshold,
                    ))
                },
                args.bbox,
                args.detection_interval,
                args.threshold,
                args.full_body,
            )?)
        };
        let mut draw_preview = |frame: &mut Mat, index: u64| -> Result<()> {
            if let Some(preview) = &mut preview {
                preview.draw(frame, index)?;
            }
            Ok(())
        };
        let captured = recording::capture(
            directory,
            args.input.as_deref(),
            args.camera.unwrap_or(0),
            args.no_display,
            args.record_seconds,
            args.max_frames,
            if args.no_display {
                None
            } else {
                Some(&mut draw_preview)
            },
        )?;
        recording::export_trimmed(
            &captured,
            &directory.join("recording.mp4"),
            args.trim_start.unwrap_or(3.0),
            args.trim_end.unwrap_or(3.0),
        )?;
        args.input = Some(captured.source.clone());
        args.camera = None;
        args.twin_output = directory.join("twin.json");
        args.output = Some(directory.join("tracked.mp4"));
        if args.json.is_some() {
            args.json = Some(directory.join("poses.jsonl"));
        }
        // The recording preview already closed. Analyze every saved frame without a second window.
        args.no_display = true;
        Some(captured)
    } else {
        args.input
            .as_deref()
            .map(recording::existing_timing)
            .transpose()?
            .flatten()
    };
    // Avoid truncating an input or model through an accidentally reused output path.
    for destination in args
        .output
        .iter()
        .chain(args.json.iter())
        .chain(std::iter::once(&args.twin_output))
    {
        for source in args.input.iter().chain([
            &args.model,
            &args.detector_model,
            &args.motion_model,
            &args.hand_model,
        ]) {
            if destination.exists() && source.exists() {
                ensure!(
                    destination.canonicalize()? != source.canonicalize()?,
                    "Output must differ from input/model"
                );
            }
        }
    }
    if let (Some(output), Some(json)) = (&args.output, &args.json) {
        ensure!(
            output != json
                && !(output.exists()
                    && json.exists()
                    && output.canonicalize()? == json.canonicalize()?),
            "Image/video and JSON destinations must differ"
        );
    }
    for other in args.output.iter().chain(args.json.iter()) {
        ensure!(
            other != &args.twin_output
                && !(other.exists()
                    && args.twin_output.exists()
                    && other.canonicalize()? == args.twin_output.canonicalize()?),
            "Twin, video/image, and JSON Lines output paths must differ"
        );
    }
    if let Some(input) = &args.input {
        ensure!(input.is_file(), "Input not found: {}", input.display());
    }
    let mut detector = if args.bbox.is_some() || args.no_detection {
        None
    } else {
        Some(PersonDetector::new(&args.detector_model)?)
    };
    let mut detection_cache = DetectionCache::new(args.detection_interval);
    let mut lifter = if args.no_3d || estimator.is_mediapipe() {
        None
    } else {
        Some(MotionLifter::new(&args.motion_model)?)
    };
    let mut hand_model = if args.no_hands {
        None
    } else {
        Some(details::HandModel::new(&args.hand_model)?)
    };
    let mut twin_recorder = twin::TwinRecorder::default();
    twin_recorder.set_upright_torso(!args.free_torso);
    if let Some(lifter) = lifter.as_mut() {
        lifter.set_upper_body(!args.full_body);
    }
    let mut dimension_tracker = tracking::DimensionTracker::default();
    dimension_tracker.set_upper_body(!args.full_body);
    let mut json = args
        .json
        .as_ref()
        .map(|p| {
            File::create(p)
                .map(BufWriter::new)
                .with_context(|| format!("Cannot create {}", p.display()))
        })
        .transpose()?;

    let is_image = args
        .input
        .as_ref()
        .map(|p| imgcodecs::have_image_reader(p))
        .transpose()?
        .unwrap_or(false);
    let mut timeline = Vec::new();
    let mut timeline_fps = 1.0;
    if is_image {
        timeline.push(0.0);
        let input = args.input.as_ref().unwrap();
        let mut image = imgcodecs::imread(input, imgcodecs::IMREAD_COLOR)?;
        ensure!(!image.empty(), "Cannot read image {}", input.display());
        process(
            &mut image,
            &mut estimator,
            &mut detector,
            &mut detection_cache,
            &mut lifter,
            &mut hand_model,
            &mut twin_recorder,
            &mut dimension_tracker,
            &args,
            0,
            0.0,
            &mut json,
        )?;
        if let Some(path) = &args.output {
            ensure!(
                imgcodecs::imwrite(path, &image, &Vector::new())?,
                "Failed to write {}",
                path.display()
            );
        }
        if !args.no_display {
            highgui::imshow("lifting-pose (press any key)", &image)?;
            highgui::wait_key(0)?;
        }
    } else {
        let mut capture = match &args.input {
            Some(path) => videoio::VideoCapture::from_file_def(path)?,
            None => videoio::VideoCapture::new(args.camera.unwrap_or(0), videoio::CAP_ANY)?,
        };
        ensure!(
            capture.is_opened()?,
            "Cannot open video/camera. Check the source and camera permissions."
        );
        let fps = capture.get(videoio::CAP_PROP_FPS)?;
        let fps = if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            30.0
        };
        timeline_fps = fps;
        let mut writer: Option<videoio::VideoWriter> = None;
        let mut index = 0;
        let started = Instant::now();
        loop {
            let mut frame = Mat::default();
            if !capture.read(&mut frame)? || frame.empty() {
                break;
            }
            timeline.push(if args.input.is_some() {
                index as f64 / fps
            } else {
                started.elapsed().as_secs_f64()
            });
            process(
                &mut frame,
                &mut estimator,
                &mut detector,
                &mut detection_cache,
                &mut lifter,
                &mut hand_model,
                &mut twin_recorder,
                &mut dimension_tracker,
                &args,
                index,
                captured
                    .as_ref()
                    .and_then(|c| c.timestamps_seconds.get(index as usize))
                    .copied()
                    .unwrap_or(*timeline.last().unwrap()),
                &mut json,
            )?;
            if let Some(path) = &args.output {
                if writer.is_none() {
                    let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
                    let out = videoio::VideoWriter::new(
                        path,
                        fourcc,
                        fps,
                        Size::new(frame.cols(), frame.rows()),
                        true,
                    )?;
                    ensure!(
                        out.is_opened()?,
                        "Cannot open video output {} (use .mp4 or .avi)",
                        path.display()
                    );
                    writer = Some(out);
                }
                writer.as_mut().unwrap().write(&frame)?;
            }
            index += 1;
            if !args.no_display {
                highgui::imshow("lifting-pose (q / Esc to quit)", &frame)?;
                let key = highgui::wait_key(1)? & 0xff;
                if key == 27 || key == i32::from(b'q') {
                    break;
                }
            }
            if args.max_frames.is_some_and(|limit| index >= limit) {
                break;
            }
        }
        if index == 0 {
            bail!("Source did not yield any frames");
        }
        if let Some(mut writer) = writer {
            writer.release()?;
        }
        capture.release()?;
        eprintln!("Processed {index} frames");
    }
    if let Some(mut writer) = json {
        writer.flush()?;
    }
    if !args.no_display {
        highgui::destroy_all_windows()?;
    }
    if !args.no_3d {
        eprintln!("Finalizing skeletal twin with one fixed shape per subject segment...");
        let source_video = if is_image {
            None
        } else {
            args.input
                .as_ref()
                .map(|source| {
                    let source = source.canonicalize()?;
                    let directory = args
                        .twin_output
                        .parent()
                        .filter(|p| !p.as_os_str().is_empty())
                        .unwrap_or(std::path::Path::new("."));
                    std::fs::create_dir_all(directory)?;
                    let directory = directory.canonicalize()?;
                    Ok::<_, anyhow::Error>(
                        source
                            .strip_prefix(directory)
                            .unwrap_or(&source)
                            .to_string_lossy()
                            .into_owned(),
                    )
                })
                .transpose()?
        };
        let timestamps_seconds = if let Some(captured) = &captured {
            captured
                .timestamps_seconds
                .iter()
                .copied()
                .take(timeline.len())
                .collect()
        } else {
            timeline
        };
        twin_recorder.set_timing(twin::RecordingTiming {
            fps: captured.as_ref().map(|c| c.fps).unwrap_or(timeline_fps),
            frame_count: timestamps_seconds.len() as u64,
            timestamps_seconds,
            source_video,
            trim_start_seconds: args.trim_start.unwrap_or(3.0),
            trim_end_seconds: args.trim_end.unwrap_or(3.0),
            upper_body: !args.full_body,
            assume_upright_torso: !args.free_torso,
        });
        twin_recorder.save(&args.twin_output)?;
        if args.record.is_none() {
            eprintln!("Saved skeletal twin: {}", args.twin_output.display());
        }
        if let Some(target) = &args.record {
            bundle::save(session_workspace.as_ref().unwrap().path(), target)?;
            eprintln!("Replaced recording save: {}", target.display());
            if target == std::path::Path::new(bundle::DEFAULT_SAVE) {
                eprintln!("Replay with: cargo run --release -- --replay");
            } else {
                eprintln!("Replay using --replay with this save: {}", target.display());
            }
        }
    }
    Ok(())
}
