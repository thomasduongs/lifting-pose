//! Capture first; model inference runs afterward on the saved footage.
use anyhow::{Context, Result, ensure};
use opencv::{
    core::{Mat, Point, Scalar, Size},
    highgui, imgproc,
    prelude::*,
    videoio,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Serialize, Deserialize)]
pub struct Capture {
    pub source: PathBuf,
    pub fps: f64,
    pub timestamps_seconds: Vec<f64>,
}

/// Reprocessing a recorded source must retain acquisition times, not its nominal codec FPS.
pub fn existing_timing(input: &Path) -> Result<Option<Capture>> {
    if input.file_name().and_then(|p| p.to_str()) != Some("source.mp4") {
        return Ok(None);
    }
    let metadata = input
        .parent()
        .unwrap_or(Path::new("."))
        .join("capture.json");
    if !metadata.is_file() {
        return Ok(None);
    }
    let mut timing: Capture = serde_json::from_reader(File::open(metadata)?)
        .context("Invalid capture timing metadata")?;
    ensure!(
        timing.fps.is_finite()
            && timing.fps > 0.0
            && !timing.timestamps_seconds.is_empty()
            && timing
                .timestamps_seconds
                .iter()
                .all(|t| t.is_finite() && *t >= 0.0)
            && timing.timestamps_seconds.windows(2).all(|p| p[1] > p[0]),
        "Invalid capture timestamps"
    );
    timing.source = input.to_owned();
    Ok(Some(timing))
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
    for name in [
        "source.mp4",
        "capture.json",
        "twin.json",
        "tracked.mp4",
        "recording.mp4",
    ] {
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
    let original_timing = input.map(existing_timing).transpose()?.flatten();
    let fps = original_timing
        .as_ref()
        .map(|t| t.fps)
        .unwrap_or(capture.get(videoio::CAP_PROP_FPS)?);
    let fps = if fps.is_finite() && fps > 0.0 {
        fps
    } else {
        30.0
    };
    let mut writer: Option<videoio::VideoWriter> = None;
    let mut timestamps_seconds = Vec::new();
    let mut start = None;
    eprintln!("Recording. Press q or Esc to stop and build the twin.");
    loop {
        let mut frame = Mat::default();
        if !capture.read(&mut frame)? || frame.empty() {
            break;
        }
        let start = start.get_or_insert_with(Instant::now);
        let elapsed = if input.is_some() {
            original_timing
                .as_ref()
                .and_then(|t| t.timestamps_seconds.get(timestamps_seconds.len()).copied())
                .unwrap_or(timestamps_seconds.len() as f64 / fps)
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

/// Finished video, sampled in acquisition time. Full source.mp4 remains a backup.
pub fn export_trimmed(capture: &Capture, output: &Path, head: f64, tail: f64) -> Result<()> {
    let times = &capture.timestamps_seconds;
    ensure!(!times.is_empty(), "No captured frames to trim");
    let begin = times[0] + head;
    let finish = times[times.len() - 1] + 1.0 / capture.fps - tail;
    ensure!(
        finish > begin,
        "Recording is too short to trim {head}s from the start and {tail}s from the end. Use --trim-start 0 --trim-end 0 for short clips."
    );
    ensure!(
        output != capture.source
            && !(output.exists() && output.canonicalize()? == capture.source.canonicalize()?),
        "Trim output must differ from source backup"
    );
    let mut reader = videoio::VideoCapture::from_file_def(&capture.source)?;
    ensure!(reader.is_opened()?, "Cannot read source video for trimming");
    let first = times.partition_point(|&t| t < begin).min(times.len() - 1);
    let mut index = 0;
    let mut frame = Mat::default();
    for _ in 0..=first {
        ensure!(
            reader.read(&mut frame)? && !frame.empty(),
            "Source video is shorter than capture metadata"
        );
    }
    index += first;
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = videoio::VideoWriter::new(
        output,
        videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?,
        capture.fps,
        Size::new(frame.cols(), frame.rows()),
        true,
    )?;
    ensure!(writer.is_opened()?, "Cannot create trimmed recording");
    let count = ((finish - begin) * capture.fps - 1e-7).ceil() as u64;
    for i in 0..count {
        let time = begin + i as f64 / capture.fps;
        while index + 1 < times.len() && times[index + 1] <= time {
            ensure!(
                reader.read(&mut frame)? && !frame.empty(),
                "Source video is shorter than capture metadata"
            );
            index += 1;
        }
        writer.write(&frame)?;
    }
    writer.release()?;
    reader.release()?;
    eprintln!("Trimmed video: {:.2}s", count as f64 / capture.fps);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finished_video_trims_both_ends_and_preserves_source() {
        let directory =
            std::env::temp_dir().join(format!("lifting-trim-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("source.mp4");
        let output = directory.join("recording.mp4");
        let mut writer = videoio::VideoWriter::new(
            &source,
            videoio::VideoWriter::fourcc('m', 'p', '4', 'v').unwrap(),
            10.0,
            Size::new(32, 32),
            true,
        )
        .unwrap();
        for i in 0..100 {
            let frame = Mat::new_rows_cols_with_default(
                32,
                32,
                opencv::core::CV_8UC3,
                Scalar::all(i as f64 * 2.0),
            )
            .unwrap();
            writer.write(&frame).unwrap();
        }
        writer.release().unwrap();
        let capture = Capture {
            source: source.clone(),
            fps: 10.0,
            timestamps_seconds: (0..100).map(|i| i as f64 / 10.0).collect(),
        };
        export_trimmed(&capture, &output, 3.0, 3.0).unwrap();
        let mut reader = videoio::VideoCapture::from_file_def(&output).unwrap();
        assert_eq!(reader.get(videoio::CAP_PROP_FRAME_COUNT).unwrap(), 40.0);
        let mut frame = Mat::default();
        reader.read(&mut frame).unwrap();
        let value = frame.data_typed::<opencv::core::Vec3b>().unwrap()[0][0];
        assert!((45..80).contains(&value));
        reader.release().unwrap();
        let reader = videoio::VideoCapture::from_file_def(&source).unwrap();
        assert_eq!(reader.get(videoio::CAP_PROP_FRAME_COUNT).unwrap(), 100.0);
        drop(reader);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
