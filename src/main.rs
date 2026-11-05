mod detector;
mod landmarks;
mod pose;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use detector::PersonDetector;
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
    #[arg(long, default_value = "models/rtmw-m.onnx")]
    model: PathBuf,
    /// YOLOX-s detector (OpenCV Zoo raw 640x640 export).
    #[arg(long, default_value = "models/yolox-s.onnx")]
    detector_model: PathBuf,
    /// Skip detection and estimate one full-frame pose.
    #[arg(long, conflicts_with = "bbox")]
    no_detection: bool,
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
}

#[derive(Serialize)]
struct FramePose<'a> {
    frame: u64,
    width: i32,
    height: i32,
    inference_ms: f64,
    people: &'a [PersonPose],
}

fn process(
    frame: &mut Mat,
    estimator: &mut PoseEstimator,
    detector: &mut Option<PersonDetector>,
    args: &Args,
    index: u64,
    json: &mut Option<BufWriter<File>>,
) -> Result<()> {
    let start = Instant::now();
    let crops = if let Some(detector) = detector {
        detector
            .detect(frame, args.detection_threshold, args.nms_threshold)?
            .into_iter()
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
    // Infer every crop from the original frame before adding any annotations.
    let mut people = Vec::with_capacity(crops.len());
    for (person_index, (bbox, detection_score)) in crops.into_iter().enumerate() {
        let keypoints = estimator.infer(frame, bbox)?;
        people.push(PersonPose {
            person_index,
            bbox,
            detection_score,
            keypoints,
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
                people: &people,
            },
        )?;
        writeln!(writer)?;
    }
    for person in &people {
        pose::draw(frame, &person.keypoints, args.threshold)?;
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
    if index == 0 || index.is_multiple_of(30) {
        eprintln!(
            "Frame {index}: {} people, {inference_ms:.1} ms",
            people.len()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Avoid truncating an input or model through an accidentally reused output path.
    for destination in [&args.output, &args.json].into_iter().flatten() {
        for source in args.input.iter().chain([&args.model, &args.detector_model]) {
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
    if let Some(input) = &args.input {
        ensure!(input.is_file(), "Input not found: {}", input.display());
    }
    let mut estimator = PoseEstimator::new(&args.model)?;
    let mut detector = if args.bbox.is_some() || args.no_detection {
        None
    } else {
        Some(PersonDetector::new(&args.detector_model)?)
    };
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
    if is_image {
        let input = args.input.as_ref().unwrap();
        let mut image = imgcodecs::imread(input, imgcodecs::IMREAD_COLOR)?;
        ensure!(!image.empty(), "Cannot read image {}", input.display());
        process(
            &mut image,
            &mut estimator,
            &mut detector,
            &args,
            0,
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
        let mut writer: Option<videoio::VideoWriter> = None;
        let mut index = 0;
        loop {
            let mut frame = Mat::default();
            if !capture.read(&mut frame)? || frame.empty() {
                break;
            }
            process(
                &mut frame,
                &mut estimator,
                &mut detector,
                &args,
                index,
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
    Ok(())
}
