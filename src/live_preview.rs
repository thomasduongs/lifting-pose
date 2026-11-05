//! Best-effort 2D inference on a worker, independent of recording and final 3D fitting.
use crate::{
    angles,
    detection_cache::DetectionCache,
    detector::PersonDetector,
    pose::{self, Bbox, Keypoint, PoseEstimator},
    scope,
};
use anyhow::{Context, Result};
use opencv::{
    core::{Mat, Point, Scalar},
    imgproc,
    prelude::*,
};
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

struct Estimate {
    captured: Instant,
    points: Vec<Keypoint>,
    predicted_hands: Vec<(crate::details::HandPose, Instant)>,
}
pub struct LivePreview {
    input: Option<SyncSender<(u64, Mat, Instant)>>,
    output: Receiver<std::result::Result<Estimate, String>>,
    worker: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    latest: Option<Estimate>,
    error: Option<String>,
    threshold: f32,
    upper_body: bool,
}
impl LivePreview {
    pub fn new(
        model: &Path,
        hand_model: Option<&Path>,
        detector: Option<(&Path, f32, f32)>,
        bbox: Option<Bbox>,
        interval: u64,
        threshold: f32,
        full_body: bool,
    ) -> Result<Self> {
        // Load before capture starts, so model startup cannot consume recording time.
        let mut estimator = PoseEstimator::new(model)?;
        estimator.set_upper_body(!full_body);
        let mut hands = hand_model.map(crate::details::HandModel::new).transpose()?;
        let detection_threshold = detector.map(|(_, score, _)| score).unwrap_or(0.4);
        let nms_threshold = detector.map(|(_, _, nms)| nms).unwrap_or(0.45);
        let mut detector = detector
            .map(|(path, _, _)| PersonDetector::new(path))
            .transpose()?;
        let mut cache = DetectionCache::new(interval);
        let (input, frames) = mpsc::sync_channel::<(u64, Mat, Instant)>(0);
        let (results, output) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let mut smoother = crate::smoothing::LandmarkSmoother::default();
        let preview_start = Instant::now();
        let worker = thread::Builder::new()
            .name("recording-pose-preview".into())
            .spawn(move || {
                while let Ok((index, image, captured)) = frames.recv() {
                    if worker_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let estimate = (|| -> Result<Estimate> {
                        let size = image.size()?;
                        let crop = if let Some(detector) = &mut detector {
                            cache.update(index, size, || {
                                detector.detect(&image, detection_threshold, nms_threshold)
                            })?;
                            cache.detections().first().map(|person| person.bbox)
                        } else {
                            Some(bbox.unwrap_or(Bbox {
                                x: 0.0,
                                y: 0.0,
                                width: size.width as f32,
                                height: size.height as f32,
                            }))
                        };
                        let mut points = if let Some(crop) = crop {
                            estimator.infer(&image, crop)?
                        } else {
                            Vec::new()
                        };
                        scope::mask_details(&mut points);
                        if !full_body {
                            scope::mask_lower(&mut points);
                        }
                        let mut predicted_hands = Vec::new();
                        if let Some(model) = &mut hands {
                            let time = captured.duration_since(preview_start).as_secs_f64();
                            let estimates = model.infer(&image, &points, time)?;
                            predicted_hands = model
                                .predictions(time)
                                .into_iter()
                                .map(|hand| {
                                    let deadline =
                                        model.prediction_deadline(&hand.side).unwrap_or(time);
                                    (hand, preview_start + Duration::from_secs_f64(deadline))
                                })
                                .collect();
                            crate::details::merge_hand_landmarks(&mut points, &estimates);
                        } else if estimator.is_mediapipe() {
                            crate::details::merge_hand_landmarks(&mut points, &[]);
                        }
                        smoother.apply(
                            &mut points,
                            captured.duration_since(preview_start).as_secs_f64(),
                            size.width as f64,
                        );
                        // Keep head anchors; omit dense facial landmarks from the overlay.
                        for point in points.iter_mut().take(91).skip(23) {
                            point.score = 0.0;
                        }
                        Ok(Estimate {
                            captured,
                            points,
                            predicted_hands,
                        })
                    })()
                    .map_err(|e| format!("{e:#}"));
                    let failed = estimate.is_err();
                    let _ = results.send(estimate);
                    if failed {
                        break;
                    }
                }
            })
            .context("Cannot start live recording preview")?;
        Ok(Self {
            input: Some(input),
            output,
            worker: Some(worker),
            stop,
            latest: None,
            error: None,
            threshold,
            upper_body: !full_body,
        })
    }
    pub fn draw(&mut self, frame: &mut Mat, index: u64) -> Result<()> {
        while let Ok(result) = self.output.try_recv() {
            match result {
                Ok(estimate) => self.latest = Some(estimate),
                Err(error) => {
                    eprintln!("Live pose preview failed: {error}. Recording continues.");
                    self.error = Some(error);
                    self.latest = None;
                }
            }
        }
        // Never wait for inference or queue stale frames. A ready worker receives
        // the current camera frame; busy workers simply skip preview updates.
        // Final processing still estimates every frame in the clean saved source.
        if self.error.is_none() {
            if let Some(input) = &self.input {
                let _ = input.try_send((index, frame.try_clone()?, Instant::now()));
            }
        }
        let status = if self.error.is_some() {
            "Pose preview unavailable; recording continues".into()
        } else if let Some(estimate) = &self.latest {
            let predictions: Vec<_> = estimate
                .predicted_hands
                .iter()
                .filter(|(_, deadline)| Instant::now() <= *deadline)
                .map(|(hand, _)| hand.clone())
                .collect();
            if estimate.captured.elapsed() > Duration::from_millis(750) {
                "Pose preview catching up...".into()
            } else if estimate.points.is_empty() {
                crate::details::draw_predicted_hands(frame, &predictions)?;
                "Live 2D pose: looking for person".into()
            } else {
                pose::draw(frame, &estimate.points, self.threshold)?;
                crate::details::draw_predicted_hands(frame, &predictions)?;
                let angles = angles::calculate_scoped(
                    &estimate.points,
                    self.threshold,
                    frame.size()?,
                    self.upper_body,
                );
                angles::draw(frame, &estimate.points, &angles)?;
                format!(
                    "Live 2D pose | {} ms old",
                    estimate.captured.elapsed().as_millis()
                )
            }
        } else {
            "Starting live pose preview...".into()
        };
        imgproc::put_text(
            frame,
            &status,
            Point::new(12, 54),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::all(245.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
        Ok(())
    }
}
impl Drop for LivePreview {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.input.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the local RTMW ONNX model; headless, no camera or windows"]
    fn worker_infers_and_overlay_keeps_source_pixels_clean() -> Result<()> {
        check_worker(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/rtmw-m.onnx"
        )))
    }
    #[test]
    #[ignore = "requires local MediaPipe Pose ONNX; headless, no camera or windows"]
    fn mediapipe_worker_preserves_clean_source() -> Result<()> {
        check_worker(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/mediapipe-pose.onnx"
        )))
    }
    #[test]
    #[ignore = "requires local MediaPipe pose, palm and hand models; no camera or windows"]
    fn mediapipe_hand_worker_runs_independent_detector_without_drawing_on_source() -> Result<()> {
        let model = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/mediapipe-pose.onnx"
        ));
        let hands = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/models/handpose.onnx"));
        let preview = LivePreview::new(model, Some(hands), None, None, 15, 0.3, false)?;
        let source =
            Mat::new_rows_cols_with_default(256, 192, opencv::core::CV_8UC3, Scalar::all(0.0))?;
        preview
            .input
            .as_ref()
            .unwrap()
            .send((0, source.try_clone()?, Instant::now()))?;
        let estimate = preview
            .output
            .recv_timeout(Duration::from_secs(10))?
            .map_err(anyhow::Error::msg)?;
        assert!(estimate.points[17..].iter().all(|p| p.score == 0.0));
        assert!(source.data_bytes()?.iter().all(|v| *v == 0));
        Ok(())
    }
    fn check_worker(model: &Path) -> Result<()> {
        let mut preview = LivePreview::new(model, None, None, None, 15, 0.3, false)?;
        let source =
            Mat::new_rows_cols_with_default(256, 192, opencv::core::CV_8UC3, Scalar::all(0.0))?;
        preview
            .input
            .as_ref()
            .unwrap()
            .send((0, source.try_clone()?, Instant::now()))?;
        let estimate = preview
            .output
            .recv_timeout(Duration::from_secs(10))?
            .map_err(anyhow::Error::msg)?;
        assert_eq!(estimate.points.len(), 133);
        assert!(estimate.points[11..23].iter().all(|p| p.score == 0.0));
        assert!(estimate.points[23..91].iter().all(|p| p.score == 0.0));
        preview.latest = Some(estimate);
        let mut display = source.try_clone()?;
        preview.draw(&mut display, 1)?;
        assert!(source.data_bytes()?.iter().all(|v| *v == 0));
        assert!(display.data_bytes()?.iter().any(|v| *v != 0));
        Ok(())
    }
}
