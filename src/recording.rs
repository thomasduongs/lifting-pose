//! Capture first; model inference runs afterward on the saved footage.
use anyhow::{Context, Result, ensure};
use opencv::{
    core::{Mat, Point, Scalar, Size},
    highgui, imgproc,
    prelude::*,
    videoio,
};
use serde::Serialize;
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Serialize)]
pub struct Capture {
    pub source: PathBuf,
    pub fps: f64,
    pub timestamps_seconds: Vec<f64>,
}

pub fn capture(
    directory: &Path,
    input: Option<&Path>,
    camera: i32,
    no_display: bool,
    seconds: Option<f64>,
    max_frames: Option<u64>,
) -> Result<Capture> {
    ensure!(
        !no_display || seconds.is_some() || max_frames.is_some() || input.is_some(),
        "Headless camera recording needs --record-seconds or --max-frames"
    );
    std::fs::create_dir_all(directory)?;
    let source = directory.join("source.mp4");
    for name in ["source.mp4", "capture.json", "twin.json", "tracked.mp4"] {
        ensure!(
            !directory.join(name).exists(),
            "Session already contains {name}. Choose a new --record directory."
        );
    }
    let mut capture = if let Some(input) = input {
        videoio::VideoCapture::from_file_def(input)?
    } else {
        videoio::VideoCapture::new(camera, videoio::CAP_ANY)?
    };
    ensure!(
        capture.is_opened()?,
        "Cannot open recording source; check camera permissions/source path"
    );
    let fps = capture.get(videoio::CAP_PROP_FPS)?;
    let fps = if fps.is_finite() && fps > 0.0 {
        fps
    } else {
        30.0
    };
    let mut writer: Option<videoio::VideoWriter> = None;
    let mut timestamps_seconds = Vec::new();
    let mut start = None;
    eprintln!(
        "Recording to {}. Press q or Esc to stop and build the twin.",
        source.display()
    );
    loop {
        let mut frame = Mat::default();
        if !capture.read(&mut frame)? || frame.empty() {
            break;
        }
        let start = start.get_or_insert_with(Instant::now);
        let elapsed = if input.is_some() {
            timestamps_seconds.len() as f64 / fps
        } else {
            start.elapsed().as_secs_f64()
        };
        if seconds.is_some_and(|limit| elapsed >= limit) {
            break;
        }
        if writer.is_none() {
            // MP4 codecs require even dimensions. Crop at most one border pixel.
            let width = frame.cols() / 2 * 2;
            let height = frame.rows() / 2 * 2;
            ensure!(width > 0 && height > 0, "Recording source is too small");
            let out = videoio::VideoWriter::new(
                &source,
                videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?,
                fps,
                Size::new(width, height),
                true,
            )?;
            ensure!(
                out.is_opened()?,
                "Cannot create recording {}",
                source.display()
            );
            writer = Some(out);
        }
        let cropped = Mat::roi(
            &frame,
            opencv::core::Rect::new(0, 0, frame.cols() / 2 * 2, frame.rows() / 2 * 2),
        )?;
        writer.as_mut().unwrap().write(&cropped)?;
        timestamps_seconds.push(elapsed);
        if !no_display {
            // Labels belong only to the preview; source.mp4 contains unannotated video.
            imgproc::put_text(
                &mut frame,
                &format!("REC {:.1}s | q / Esc: stop", elapsed),
                Point::new(12, 28),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.65,
                Scalar::new(30.0, 30.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )?;
            highgui::imshow("Record human twin", &frame)?;
            let key = highgui::wait_key(1)? & 0xff;
            if key == 27 || key == i32::from(b'q') {
                break;
            }
        }
        if max_frames.is_some_and(|limit| timestamps_seconds.len() as u64 >= limit) {
            break;
        }
    }
    if let Some(mut writer) = writer {
        writer.release()?;
    }
    capture.release()?;
    if !no_display {
        highgui::destroy_all_windows()?;
    }
    ensure!(
        !timestamps_seconds.is_empty(),
        "No video frames were recorded"
    );
    let result = Capture {
        source,
        fps,
        timestamps_seconds,
    };
    serde_json::to_writer_pretty(
        File::create(directory.join("capture.json")).context("Cannot save capture timing")?,
        &result,
    )?;
    eprintln!(
        "Recorded {} frames. Building the skeletal twin from saved footage...",
        result.timestamps_seconds.len()
    );
    Ok(result)
}
