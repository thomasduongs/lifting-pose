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

struct LiveDetector {
    input: Option<SyncSender<(u64, Mat)>>,
    worker: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    sent: Option<(u64, opencv::core::Size)>,
    interval: u64,
}
type DetectorOutput = Receiver<std::result::Result<(opencv::core::Size, Option<Bbox>), String>>;
impl LiveDetector {
    fn start(
        mut detector: PersonDetector,
        threshold: f32,
        nms: f32,
        interval: u64,
    ) -> Result<(Self, DetectorOutput)> {
        let (input, frames) = mpsc::sync_channel::<(u64, Mat)>(0);
        let (results, output) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let shutdown = stop.clone();
        let worker = thread::Builder::new()
            .name("recording-person-detection".into())
            .spawn(move || {
                let mut cache = DetectionCache::new(1);
                while let Ok((index, image)) = frames.recv() {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    let result = (|| -> Result<(opencv::core::Size, Option<Bbox>)> {
                        cache.update(index, image.size()?, || {
                            detector.detect(&image, threshold, nms)
                        })?;
                        Ok((image.size()?, cache.detections().first().map(|p| p.bbox)))
                    })()
                    .map_err(|e| format!("{e:#}"));
                    if results.send(result).is_err() {
                        break;
                    }
                }
            })
            .context("Cannot start live person detector")?;
        Ok((
            Self {
                input: Some(input),
                worker: Some(worker),
                stop,
                sent: None,
                interval: interval.max(1),
            },
            output,
        ))
    }
    fn submit(&mut self, image: &Mat, index: u64) -> Result<()> {
        let size = image.size()?;
        if self.sent.is_none_or(|(last, previous)| {
            previous != size || index.saturating_sub(last) >= self.interval
        }) {
            if let Some(input) = &self.input {
                if input.try_send((index, image.try_clone()?)).is_ok() {
                    self.sent = Some((index, size));
                }
            }
        }
        Ok(())
    }
}
impl Drop for LiveDetector {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.input.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Default)]
struct WristIdentity {
    previous: [Option<(f64, [f32; 2])>; 2],
}
impl WristIdentity {
    fn confirm(
        &mut self,
        points: &mut [Keypoint],
        hands: &[crate::details::HandPose],
        time: f64,
        body_only: bool,
    ) -> [bool; 2] {
        if points.len() < 17 {
            self.previous = [None; 2];
            return [false; 2];
        }
        let before = self.previous;
        self.previous = [None; 2];
        ["left", "right"].map(|side| {
            let i = usize::from(side == "right");
            let sample = if body_only {
                let p = points[9 + i];
                (p.score >= 0.75).then_some((
                    [p.x, p.y],
                    p.score,
                    (points[5].x - points[6].x).hypot(points[5].y - points[6].y) * 0.3,
                ))
            } else {
                hands
                    .iter()
                    .find(|h| h.side == side && h.confidence >= 0.65)
                    .map(|h| {
                        let p = &h.image_landmarks;
                        (
                            p[0],
                            h.confidence,
                            (p[5][0] - p[17][0]).hypot(p[5][1] - p[17][1]).max(8.0),
                        )
                    })
            };
            let Some((wrist, confidence, span)) = sample else {
                return false;
            };
            self.previous[i] = Some((time, wrist));
            let consistent = before[i].is_some_and(|(t, p)| {
                time > t
                    && time - t <= 0.3
                    && (p[0] - wrist[0]).hypot(p[1] - wrist[1]) <= (span * 2.0).max(40.0)
            });
            let confirmed = (!body_only && confidence >= 0.85) || consistent;
            if confirmed && !body_only {
                points[9 + i].x = wrist[0];
                points[9 + i].y = wrist[1];
                points[9 + i].score = confidence;
            }
            confirmed
        })
    }
}

struct Estimate {
    captured: Instant,
    points: Vec<Keypoint>,
    index: u64,
    bbox: Option<Bbox>,
    wrists: [bool; 2],
    occluded: [bool; 17],
    #[cfg(test)]
    timings_ms: [f64; 3],
    #[cfg(test)]
    complete: bool,
}
pub struct LivePreview {
    input: Option<SyncSender<(u64, Mat, Instant)>>,
    detector: Option<LiveDetector>,
    output: Receiver<std::result::Result<Estimate, String>>,
    worker: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    latest: Option<Estimate>,
    error: Option<String>,
    tracking_error: Option<String>,
    threshold: f32,
    upper_body: bool,
    image_tracking: crate::preview_tracking::PreviewTracking,
    started: Instant,
    stabilization: bool,
}
impl LivePreview {
    pub fn new(
        model: &Path,
        arm_model: Option<&Path>,
        hand_model: Option<&Path>,
        detector: Option<(&Path, f32, f32)>,
        bbox: Option<Bbox>,
        interval: u64,
        threshold: f32,
        full_body: bool,
        stabilization: bool,
        smoothing: bool,
    ) -> Result<Self> {
        // Load before capture starts, so model startup cannot consume recording time.
        // The overlay needs image joints, not two body networks or discarded
        // world coordinates. Use the installed RTM image model directly here;
        // the recording's 3D pass still uses the selected primary model.
        let live_model = arm_model.unwrap_or(model);
        let mut estimator = PoseEstimator::new(live_model)?;
        estimator.set_upper_body(!full_body);
        estimator.set_stabilization(stabilization, smoothing);
        let mut hands = hand_model.map(crate::details::HandModel::new).transpose()?;
        let detection_threshold = detector.map(|(_, score, _)| score).unwrap_or(0.4);
        let nms_threshold = detector.map(|(_, _, nms)| nms).unwrap_or(0.45);
        let mut detector = detector
            .map(|(path, _, _)| PersonDetector::new(path))
            .transpose()?;
        eprintln!("Preparing live tracking models...");
        estimator.enable_acceleration();
        estimator.warm_up()?;
        if let Some(model) = &mut hands {
            model.enable_acceleration();
            model.enable_fast_tracking();
            model.warm_up()?;
        }
        if let Some(model) = &mut detector {
            model.enable_acceleration();
            let image =
                Mat::new_rows_cols_with_default(640, 640, opencv::core::CV_8UC3, Scalar::all(0.0))?;
            model.detect(&image, detection_threshold, nms_threshold)?;
        }
        let (live_detector, detector_results) = if let Some(model) = detector {
            let (worker, output) =
                LiveDetector::start(model, detection_threshold, nms_threshold, interval)?;
            (Some(worker), Some(output))
        } else {
            (None, None)
        };
        let (input, frames) = mpsc::sync_channel::<(u64, Mat, Instant)>(0);
        let (results, output) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let preview_start = Instant::now();
        let worker = thread::Builder::new()
            .name("recording-pose-preview".into())
            .spawn(move || {
                let mut detected_crop = None;
                let mut wrist_identity = WristIdentity::default();
                while let Ok((index, image, captured)) = frames.recv() {
                    if worker_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let estimate = (|| -> Result<Estimate> {
                        #[cfg(test)]
                        let stage_start = Instant::now();
                        let size = image.size()?;
                        let crop = if let Some(results) = &detector_results {
                            while let Ok(result) = results.try_recv() {
                                detected_crop = Some(result.map_err(anyhow::Error::msg)?);
                            }
                            detected_crop
                                .filter(|(source_size, _)| *source_size == size)
                                .and_then(|(_, crop)| crop)
                        } else {
                            Some(bbox.unwrap_or(Bbox {
                                x: 0.0,
                                y: 0.0,
                                width: size.width as f32,
                                height: size.height as f32,
                            }))
                        };
                        #[cfg(test)]
                        let detection_done = Instant::now();
                        let mut points = if let Some(crop) = crop {
                            estimator.infer(&image, crop)?
                        } else {
                            Vec::new()
                        };
                        if !points.is_empty() {
                            while points.len() < 133 {
                                let i = points.len();
                                points.push(Keypoint {
                                    name: crate::landmarks::NAMES[i],
                                    x: -1.0,
                                    y: -1.0,
                                    score: 0.0,
                                });
                            }
                        }
                        scope::mask_details(&mut points);
                        if !full_body {
                            scope::mask_lower(&mut points);
                        }
                        let mut occluded = crate::occlusion::mask_arm_overlap(&mut points, size);
                        #[cfg(test)]
                        let pose_done = Instant::now();
                        // Publish the fresh body measurement immediately. A
                        // difficult hand search must not hold the body hostage.
                        if crop.is_some() && !points.is_empty() {
                            let mut body_points = points.clone();
                            for p in body_points.iter_mut().skip(91) {
                                p.score = 0.0;
                            }
                            let _ = results.send(Ok(Estimate {
                                captured,
                                index,
                                points: body_points,
                                bbox: crop,
                                wrists: [false; 2],
                                occluded,
                                #[cfg(test)]
                                timings_ms: [
                                    (detection_done - stage_start).as_secs_f64() * 1000.0,
                                    (pose_done - detection_done).as_secs_f64() * 1000.0,
                                    0.0,
                                ],
                                #[cfg(test)]
                                complete: false,
                            }));
                        }
                        let mut confirmed_hands = Vec::new();
                        if let Some(model) = &mut hands {
                            let time = captured.duration_since(preview_start).as_secs_f64();
                            let references = model.occlusion_references(time);
                            let estimates = model.infer(
                                &image,
                                &points,
                                time,
                                crop,
                                estimator.hand_crop_wrists().as_ref(),
                            )?;
                            let recent = crate::occlusion::mask_recent_hand_overlap(
                                &mut points,
                                size,
                                &references,
                            );
                            for i in 0..17 {
                                occluded[i] |= recent[i];
                            }
                            estimator.confirm_hands(&estimates);
                            confirmed_hands = estimates.clone();
                            model.merge_landmarks(&mut points, &estimates, time);
                        } else if estimator.is_mediapipe() {
                            crate::details::merge_hand_landmarks(&mut points, &[]);
                        }
                        let wrists = wrist_identity.confirm(
                            &mut points,
                            &confirmed_hands,
                            captured.duration_since(preview_start).as_secs_f64(),
                            hands.is_none(),
                        );
                        // Current-image tracking is performed once, on camera
                        // frames by PreviewTracking. Repeating it on sparse NN
                        // frames here adds latency and weaker correspondences.
                        // Keep head anchors; omit dense facial landmarks from the overlay.
                        for point in points.iter_mut().take(91).skip(23) {
                            point.score = 0.0;
                        }
                        Ok(Estimate {
                            captured,
                            points,
                            index,
                            bbox: crop,
                            wrists,
                            occluded,
                            #[cfg(test)]
                            complete: true,
                            #[cfg(test)]
                            timings_ms: [
                                (detection_done - stage_start).as_secs_f64() * 1000.0,
                                (pose_done - detection_done).as_secs_f64() * 1000.0,
                                pose_done.elapsed().as_secs_f64() * 1000.0,
                            ],
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
            detector: live_detector,
            output,
            worker: Some(worker),
            stop,
            latest: None,
            error: None,
            tracking_error: None,
            threshold,
            upper_body: !full_body,
            image_tracking: crate::preview_tracking::PreviewTracking::default(),
            started: preview_start,
            stabilization,
        })
    }
    pub fn draw(&mut self, frame: &mut Mat, index: u64) -> Result<()> {
        let captured = Instant::now();
        if let Some(detector) = &mut self.detector {
            detector.submit(frame, index)?;
        }
        if self.stabilization {
            if let Err(error) = self.image_tracking.push(
                frame,
                index,
                captured.duration_since(self.started).as_secs_f64(),
            ) {
                eprintln!("Live image tracking reset: {error}. Recording continues.");
                self.tracking_error = Some(error.to_string());
                self.image_tracking.clear();
            } else {
                self.tracking_error = None;
            }
        }
        while let Ok(result) = self.output.try_recv() {
            match result {
                Ok(estimate) => {
                    if self.stabilization {
                        if let Some(bbox) = estimate.bbox {
                            if let Err(error) = self.image_tracking.anchor(
                                estimate.index,
                                &estimate.points,
                                bbox,
                                estimate.wrists,
                                estimate.occluded,
                            ) {
                                eprintln!(
                                    "Live image alignment reset: {error}. Recording continues."
                                );
                                self.tracking_error = Some(error.to_string());
                                self.image_tracking.clear();
                            }
                        }
                    }
                    self.latest = Some(estimate);
                }
                Err(error) => {
                    eprintln!("Live pose preview failed: {error}. Recording continues.");
                    self.error = Some(error);
                    self.latest = None;
                    self.image_tracking.clear();
                }
            }
        }
        // Never wait for inference or queue stale frames. A ready worker receives
        // the current camera frame; busy workers simply skip preview updates.
        // Final processing still estimates every frame in the clean saved source.
        if self.error.is_none() {
            if let Some(input) = &self.input {
                let _ = input.try_send((index, frame.try_clone()?, captured));
            }
        }
        let status = if self.error.is_some() {
            "Pose preview unavailable; recording continues".into()
        } else if self.tracking_error.is_some() {
            "Live image tracking: reacquiring".into()
        } else if let Some(estimate) = &self.latest {
            let points = if self.stabilization {
                &self.image_tracking.points
            } else {
                &estimate.points
            };
            if estimate.captured.elapsed() > Duration::from_millis(750) {
                "Pose preview catching up...".into()
            } else if points.is_empty() || !points.iter().any(|p| p.score >= self.threshold) {
                crate::joint_tracking::draw_predictions(frame, &self.image_tracking.predicted)?;
                "Live tracking: reacquiring joints".into()
            } else {
                pose::draw(frame, points, self.threshold)?;
                crate::joint_tracking::draw_predictions(frame, &self.image_tracking.predicted)?;
                let angles = angles::calculate_scoped(
                    points,
                    self.threshold,
                    frame.size()?,
                    self.upper_body,
                );
                angles::draw(frame, points, &angles)?;
                format!(
                    "Live image tracking | neural update {} ms old",
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
    fn moderate_hands_confirm_wrist_identity_at_actual_neural_intervals_without_inflating_confidence()
     {
        let mut identity = WristIdentity::default();
        let mut points = vec![
            Keypoint {
                name: "joint",
                x: 100.0,
                y: 100.0,
                score: 0.9
            };
            133
        ];
        let mut hand = crate::details::HandPose {
            side: "left".into(),
            confidence: 0.7,
            image_landmarks: [[100.0, 100.0]; 21],
            world_landmarks: [[0.0; 3]; 21],
        };
        hand.image_landmarks[5][0] = 120.0;
        hand.image_landmarks[17][0] = 80.0;
        assert!(!identity.confirm(&mut points, std::slice::from_ref(&hand), 0.0, false)[0]);
        hand.image_landmarks[0] = [110.0, 100.0];
        assert!(identity.confirm(&mut points, std::slice::from_ref(&hand), 0.15, false)[0]);
        assert_eq!(points[9].x, 110.0);
        assert_eq!(points[9].score, 0.7);
        identity.confirm(&mut points, &[], 0.2, false);
        assert!(!identity.confirm(&mut points, std::slice::from_ref(&hand), 0.25, false)[0]);
    }
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
        let preview = LivePreview::new(
            model,
            None,
            Some(hands),
            None,
            None,
            15,
            0.3,
            false,
            true,
            true,
        )?;
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
        let mut preview =
            LivePreview::new(model, None, None, None, None, 15, 0.3, false, true, true)?;
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

#[cfg(test)]
mod realtime_evaluation {
    use super::*;
    use opencv::videoio;
    #[test]
    #[ignore = "requires LIFTING_LIVE_SOURCE and local pose/arm/hand/detector models; camera-free real-time video simulation"]
    fn evaluate_realtime_neural_preview() -> Result<()> {
        let requested = std::env::var("LIFTING_CV_THREADS")
            .ok()
            .map(|v| v.parse())
            .transpose()?;
        opencv::core::set_num_threads(crate::opencv_thread_budget(requested))?;
        let source = std::env::var("LIFTING_LIVE_SOURCE")?;
        let mut capture = videoio::VideoCapture::from_file_def(&source)?;
        let mut preview = LivePreview::new(
            Path::new("models/mediapipe-pose.onnx"),
            Some(Path::new("models/rtmpose-m.onnx")),
            Some(Path::new("models/handpose.onnx")),
            Some((Path::new("models/yolox-s.onnx"), 0.4, 0.45)),
            None,
            15,
            0.3,
            false,
            true,
            false,
        )?;
        let started = Instant::now();
        let mut processing = Vec::new();
        let mut sources = std::collections::BTreeSet::new();
        let mut stage_times = Vec::new();
        let mut completed = std::collections::BTreeSet::new();
        let mut joint_support = [0usize; 6];
        let mut supported = 0;
        let mut tested = 0;
        let mut ages = Vec::new();
        for index in 0..120 {
            let deadline = started + Duration::from_secs_f64(index as f64 / 30.0);
            if let Some(wait) = deadline.checked_duration_since(Instant::now()) {
                thread::sleep(wait);
            }
            let mut image = Mat::default();
            if !capture.read(&mut image)? {
                break;
            }
            let begin = Instant::now();
            preview.draw(&mut image, index)?;
            processing.push(begin.elapsed().as_secs_f64() * 1000.0);
            if let Some(estimate) = &preview.latest {
                if estimate.points.iter().any(|p| p.score >= 0.3) {
                    sources.insert(estimate.index);
                    if estimate.complete && completed.insert(estimate.index) {
                        stage_times.push(estimate.timings_ms);
                    }
                }
                ages.push(estimate.captured.elapsed().as_secs_f64() * 1000.0);
            }
            if index >= 15 {
                tested += 1;
                for (slot, joint) in (5..11).enumerate() {
                    if preview
                        .image_tracking
                        .points
                        .get(joint)
                        .is_some_and(|p| p.score >= 0.3)
                    {
                        joint_support[slot] += 1;
                    }
                }
                if preview
                    .image_tracking
                    .points
                    .iter()
                    .enumerate()
                    .filter(|(j, p)| (5..11).contains(j) && p.score >= 0.3)
                    .count()
                    >= 2
                {
                    supported += 1;
                }
            }
        }
        let stats = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            serde_json::json!({"mean":v.iter().sum::<f64>()/v.len().max(1) as f64,"p95":v.get((v.len()as f64*0.95)as usize),"max":v.last()})
        };
        let metrics = serde_json::json!({"capture_frames":processing.len(),"elapsed_seconds":started.elapsed().as_secs_f64(),"neural_updates":sources.len(),"stage_ms":(0..3).map(|i|stats(stage_times.iter().map(|v|v[i]).collect())).collect::<Vec<_>>(),"cv_threads":opencv::core::get_num_threads()?,"main_thread_ms":stats(processing),"neural_age_ms":stats(ages),"joint_support_after_warmup":joint_support,"tracked_after_warmup":supported,"tested_after_warmup":tested,"worker_error":preview.error,"image_error":preview.tracking_error});
        if let Ok(directory) = std::env::var("LIFTING_LIVE_EVAL") {
            std::fs::write(
                Path::new(&directory).join("realtime-metrics.json"),
                serde_json::to_vec_pretty(&metrics)?,
            )?;
        }
        println!("{metrics}");
        assert!(preview.error.is_none());
        assert!(preview.tracking_error.is_none());
        assert!(sources.len() > 5);
        assert!(
            supported * 10 >= tested * 9,
            "Live tracking failed to maintain current-image support on at least 90% of frames"
        );
        Ok(())
    }
}
