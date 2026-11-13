//! Asynchronous live camera capture + MediaPipe inference for `--live-biomech`.
//!
//! Two threads: capture (camera/file -> newest `VideoFrame`) and inference
//! (newest frame -> `PoseFrame`). Both hand off through single-slot "latest value"
//! mailboxes, so there is never an unbounded queue and the preview keeps refreshing
//! while inference runs. Nothing is written to disk and nothing is transmitted.

use crate::detector::PersonDetector;
use crate::pose::{Bbox, PoseEstimator};
use biomech_lite::live_fit::{Landmark, Observation};
use opencv::{
    core::{Mat, MatTraitConst, MatTraitConstManual},
    imgcodecs, imgproc,
    prelude::*,
    videoio,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MAX_READ_FAILURES: u32 = 60;
const CAMERA_OPEN_TIMEOUT: Duration = Duration::from_secs(25);

pub struct Config {
    pub camera: i32,
    pub input: Option<PathBuf>,
    pub pose_model: PathBuf,
    pub detector_model: PathBuf,
    pub no_detection: bool,
    pub arm_model: Option<PathBuf>,
    pub hand_model: Option<PathBuf>,
    pub motion_model: PathBuf,
    pub options: crate::tracking_pipeline::Options,
    pub stabilization: bool,
    pub smoothing: bool,
    pub upright_torso: bool,
    pub acceleration: bool,
    pub bbox: Option<Bbox>,
    pub detection_interval: u64,
    pub detection_threshold: f32,
    pub nms_threshold: f32,
    pub opencv_threads: i32,
}

/// Unmirrored 0x00RRGGBB pixels.
pub struct VideoFrame {
    pub index: u64,
    pub captured: Instant,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u32>,
}

pub struct PoseFrame {
    pub source_video: Arc<VideoFrame>,
    pub index: u64,
    pub captured: Instant,
    pub observation: Observation,
    pub points: Vec<(String, f32, f32, f32)>,
    pub inference_ms: f64,
    pub hands: Vec<crate::details::HandPose>,
    pub predicted_hands: Vec<crate::details::HandPose>,
    /// Image-constrained depths are skeletal estimates, never native measurements.
    pub image_constrained: usize,
    pub subject_segment: Option<usize>,
    pub refined_joints: usize,
}

#[derive(Clone, Default)]
pub struct CaptureState {
    pub video: Option<Arc<VideoFrame>>,
    pub pose: Option<Arc<PoseFrame>>,
    pub status: String,
    pub error: Option<String>,
    pub ended: bool,
}

struct Frame {
    video: Arc<VideoFrame>,
    index: u64,
    captured: Instant,
    image: Mat,
}

struct Inner {
    state: Mutex<CaptureState>,
    /// Single-slot mailbox: newest frame replaces any unconsumed one.
    pending: Mutex<Option<Frame>>,
    wake: Condvar,
    stop: AtomicBool,
    capture_done: AtomicBool,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Inner {
    fn set_status(&self, status: impl Into<String>) {
        lock(&self.state).status = status.into();
    }
    fn fail(&self, message: impl Into<String>) {
        let message = message.into();
        let mut s = lock(&self.state);
        s.status = "error".into();
        s.error = Some(message);
    }
}

pub struct LiveCapture {
    inner: Arc<Inner>,
    threads: Vec<JoinHandle<()>>,
}

impl LiveCapture {
    pub fn start(config: Config) -> Result<Self, String> {
        if !config.pose_model.is_file() {
            return Err(format!(
                "Pose model not found: {}",
                config.pose_model.display()
            ));
        }
        if !config.no_detection && !config.detector_model.is_file() {
            return Err(format!(
                "Detector model not found: {} (use --no-detection to skip)",
                config.detector_model.display()
            ));
        }
        if let Some(input) = &config.input {
            if !input.is_file() {
                return Err(format!("Input file not found: {}", input.display()));
            }
        }
        let source = if config.input.is_some() {
            "file input"
        } else {
            "camera"
        };
        let inner = Arc::new(Inner {
            state: Mutex::new(CaptureState {
                status: format!("opening {source}"),
                ..Default::default()
            }),
            pending: Mutex::new(None),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            capture_done: AtomicBool::new(false),
        });
        if config.opencv_threads > 0 {
            let _ = opencv::core::set_num_threads(config.opencv_threads.clamp(2, 4));
        }
        let mut threads = Vec::new();
        let capture_inner = inner.clone();
        let camera = config.camera;
        let input = config.input.clone();
        threads.push(
            thread::Builder::new()
                .name("biomech-capture".into())
                .spawn(move || capture_loop(capture_inner, camera, input))
                .map_err(|e| format!("Cannot start capture thread: {e}"))?,
        );
        let infer_inner = inner.clone();
        let infer = thread::Builder::new()
            .name("biomech-inference".into())
            .spawn(move || {
                if let Err(e) = inference_loop(&infer_inner, &config) {
                    infer_inner.fail(e);
                }
            });
        match infer {
            Ok(handle) => threads.push(handle),
            Err(e) => {
                inner.stop.store(true, Ordering::SeqCst);
                for t in threads {
                    let _ = t.join();
                }
                return Err(format!("Cannot start inference thread: {e}"));
            }
        }
        Ok(Self { inner, threads })
    }

    /// Snapshot of the newest pointers; never blocks on camera or model work.
    pub fn latest(&self) -> CaptureState {
        lock(&self.inner.state).clone()
    }
}

impl Drop for LiveCapture {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::SeqCst);
        self.inner.wake.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn bgr_to_rgb_u32(image: &Mat) -> opencv::Result<(usize, usize, Vec<u32>)> {
    let mut bgr = Mat::default();
    let src: &Mat = if image.typ() == opencv::core::CV_8UC3 {
        image
    } else {
        imgproc::cvt_color_def(image, &mut bgr, imgproc::COLOR_GRAY2BGR)?;
        &bgr
    };
    let (w, h) = (src.cols() as usize, src.rows() as usize);
    let owned;
    let src = if src.is_continuous() {
        src
    } else {
        owned = src.try_clone()?;
        &owned
    };
    let bytes = src.data_bytes()?;
    let rgb = bytes
        .chunks_exact(3)
        .map(|p| ((p[2] as u32) << 16) | ((p[1] as u32) << 8) | p[0] as u32)
        .collect();
    Ok((w, h, rgb))
}

fn publish(inner: &Inner, index: u64, captured: Instant, image: Mat) {
    match bgr_to_rgb_u32(&image) {
        Ok((width, height, rgb)) if width * height == rgb.len() => {
            let frame = Arc::new(VideoFrame {
                index,
                captured,
                width,
                height,
                rgb,
            });
            lock(&inner.state).video = Some(frame.clone());
            *lock(&inner.pending) = Some(Frame {
                video: frame,
                index,
                captured,
                image,
            });
            inner.wake.notify_one();
        }
        Ok(_) => inner.fail("Unexpected frame layout from video source"),
        Err(e) => inner.fail(format!("Frame conversion failed: {e}")),
    }
}

fn capture_loop(inner: Arc<Inner>, camera: i32, input: Option<PathBuf>) {
    let is_file = input.is_some();
    let is_image = input.as_ref().is_some_and(|p| {
        p.extension()
            .and_then(|e| e.to_str())
            .map(|e| {
                matches!(
                    e.to_ascii_lowercase().as_str(),
                    "png" | "jpg" | "jpeg" | "bmp" | "webp" | "tif" | "tiff"
                )
            })
            .unwrap_or(false)
    });
    let finish = |inner: &Inner| {
        inner.capture_done.store(true, Ordering::SeqCst);
        inner.wake.notify_all();
    };

    if is_image {
        let path = input.unwrap();
        match imgcodecs::imread(path.to_str().unwrap_or(""), imgcodecs::IMREAD_COLOR) {
            Ok(img) if !img.empty() => {
                publish(&inner, 0, Instant::now(), img);
                let mut s = lock(&inner.state);
                s.status = "file input (image)".into();
                s.ended = true;
            }
            Ok(_) => inner.fail(format!("Cannot decode image {}", path.display())),
            Err(e) => inner.fail(format!("Cannot read image {}: {e}", path.display())),
        }
        finish(&inner);
        return;
    }

    let mut cap = match &input {
        Some(path) => match videoio::VideoCapture::from_file_def(path.to_str().unwrap_or("")) {
            Ok(c) if c.is_opened().unwrap_or(false) => c,
            Ok(_) => {
                inner.fail("Cannot open video file");
                finish(&inner);
                return;
            }
            Err(e) => {
                inner.fail(format!("Cannot open video source: {e}"));
                finish(&inner);
                return;
            }
        },
        None => {
            // First use on macOS: OpenCV's AVFoundation backend requests permission and
            // returns "not opened" immediately, so retry for a bounded time while the
            // user answers the prompt. Stays responsive to stop.
            #[cfg(target_os = "macos")]
            let api = videoio::CAP_AVFOUNDATION;
            #[cfg(not(target_os = "macos"))]
            let api = videoio::CAP_ANY;
            let begun = Instant::now();
            let mut last_err = String::new();
            let mut found = None;
            'retry: while begun.elapsed() < CAMERA_OPEN_TIMEOUT {
                match videoio::VideoCapture::new(camera, api) {
                    Ok(c) if c.is_opened().unwrap_or(false) => {
                        found = Some(c);
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => last_err = format!(" ({e})"),
                }
                inner.set_status(format!(
                    "waiting for camera {camera} / permission ({:.0}s of {:.0}s)",
                    begun.elapsed().as_secs_f64(),
                    CAMERA_OPEN_TIMEOUT.as_secs_f64()
                ));
                for _ in 0..5 {
                    if inner.stop.load(Ordering::SeqCst) {
                        break 'retry;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            }
            match found {
                Some(c) => c,
                None => {
                    if !inner.stop.load(Ordering::SeqCst) {
                        inner.fail(format!(
                            "Cannot open camera {camera} after {}s{last_err}. Check that a camera is \
                             connected and that camera permission is granted to this app \
                             (System Settings > Privacy & Security > Camera), then restart.",
                            CAMERA_OPEN_TIMEOUT.as_secs()
                        ));
                    }
                    finish(&inner);
                    return;
                }
            }
        }
    };
    if !is_file {
        // Use the camera's widescreen mode rather than the tighter 4:3 mode.
        // Presentation letterboxes the complete returned frame; it never crops it.
        let _ = cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0);
        let _ = cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0);
        let _ = cap.set(videoio::CAP_PROP_FPS, 30.0);
    }
    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap_or(0.0);
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap_or(0.0);
    let fps = cap.get(videoio::CAP_PROP_FPS).unwrap_or(0.0);
    let label = if is_file { "file input" } else { "camera" };
    inner.set_status(format!("{label} {width:.0}x{height:.0} @ {fps:.0}fps"));
    let frame_period = if is_file {
        let f = if fps.is_finite() && fps >= 1.0 {
            fps
        } else {
            30.0
        };
        Some(Duration::from_secs_f64(1.0 / f))
    } else {
        None
    };

    let mut index = 0u64;
    let mut failures = 0u32;
    let mut next_deadline = Instant::now();
    while !inner.stop.load(Ordering::SeqCst) {
        if let Some(period) = frame_period {
            let now = Instant::now();
            if next_deadline > now {
                thread::sleep((next_deadline - now).min(Duration::from_millis(50)));
                continue;
            }
            next_deadline = (next_deadline + period).max(now.checked_sub(period).unwrap_or(now));
        }
        let mut image = Mat::default();
        match cap.read(&mut image) {
            Ok(true) if !image.empty() => {
                failures = 0;
                let captured = Instant::now();
                publish(&inner, index, captured, image);
                index += 1;
            }
            Ok(_) | Err(_) if is_file => {
                // End of file: keep the last frame visible, honestly flagged.
                let mut s = lock(&inner.state);
                s.status = "file input ended".into();
                s.ended = true;
                break;
            }
            Ok(_) | Err(_) => {
                failures += 1;
                if failures >= MAX_READ_FAILURES {
                    inner.fail("Camera stopped delivering frames");
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
        }
    }
    let _ = cap.release();
    finish(&inner);
}

fn inference_loop(inner: &Arc<Inner>, config: &Config) -> Result<(), String> {
    let mut estimator =
        PoseEstimator::new(&config.pose_model).map_err(|e| format!("Pose model failed: {e}"))?;
    estimator.set_upper_body(!config.options.full_body);
    estimator.set_stabilization(config.stabilization, config.smoothing);
    if let Some(path) = &config.arm_model {
        if estimator.is_mediapipe() {
            estimator.set_arm_refiner(path).map_err(|e| e.to_string())?;
        }
    }
    let mut hands = config
        .hand_model
        .as_ref()
        .map(|p| crate::details::HandModel::new(p))
        .transpose()
        .map_err(|e| e.to_string())?;
    let mut lifter = if estimator.is_mediapipe() {
        None
    } else {
        Some(crate::motion3d::MotionLifter::new(&config.motion_model).map_err(|e| e.to_string())?)
    };
    if let Some(lifter) = &mut lifter {
        lifter.set_upper_body(!config.options.full_body);
    }
    let mut twin = crate::twin::TwinRecorder::streaming();
    twin.set_upright_torso(config.upright_torso);
    twin.set_stabilization(config.stabilization, config.smoothing);
    let mut dimensions = crate::tracking::DimensionTracker::default();
    dimensions.set_upper_body(!config.options.full_body);
    let mut detector = if config.no_detection {
        None
    } else {
        Some(
            PersonDetector::new(&config.detector_model)
                .map_err(|e| format!("Detector model failed: {e}"))?,
        )
    };
    let origin = Instant::now();
    if config.acceleration {
        estimator.enable_acceleration();
        if let Some(hands) = &mut hands {
            hands.enable_acceleration();
        }
        if let Some(detector) = &mut detector {
            detector.enable_acceleration();
        }
    }
    let mut detection_cache =
        crate::detection_cache::DetectionCache::new(config.detection_interval);
    let mut last_index: Option<u64> = None;
    let mut motion = crate::live_motion::LiveMotion::default();
    loop {
        let frame = {
            let mut slot = lock(&inner.pending);
            loop {
                if inner.stop.load(Ordering::SeqCst) {
                    return Ok(());
                }
                if let Some(f) = slot.take() {
                    break f;
                }
                if inner.capture_done.load(Ordering::SeqCst) {
                    return Ok(());
                }
                slot = inner
                    .wake
                    .wait_timeout(slot, Duration::from_millis(100))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        };
        if last_index == Some(frame.index) {
            continue;
        }
        last_index = Some(frame.index);
        let started = Instant::now();
        let timestamp = frame
            .captured
            .saturating_duration_since(origin)
            .as_secs_f64();
        let result = (|| -> anyhow::Result<Option<crate::PersonPose>> {
            let size = frame.image.size()?;
            let (bbox, score) = if let Some(detector) = &mut detector {
                detection_cache.update(frame.index, size, || {
                    detector.detect(
                        &frame.image,
                        config.detection_threshold,
                        config.nms_threshold,
                    )
                })?;
                let Some(person) = detection_cache.detections().first() else {
                    if let Some(lifter) = &mut lifter {
                        lifter.reset();
                    }
                    return Ok(None);
                };
                (person.bbox, Some(person.score))
            } else {
                (
                    config.bbox.unwrap_or(Bbox {
                        x: 0.0,
                        y: 0.0,
                        width: size.width as f32,
                        height: size.height as f32,
                    }),
                    None,
                )
            };
            anyhow::ensure!(
                bbox.x >= 0.0
                    && bbox.y >= 0.0
                    && bbox.x + bbox.width <= size.width as f32
                    && bbox.y + bbox.height <= size.height as f32,
                "Crop extends outside frame"
            );
            let person = crate::tracking_pipeline::track_person(
                &frame.image,
                &mut estimator,
                &mut lifter,
                &mut hands,
                &mut twin,
                &mut dimensions,
                config.options,
                frame.index,
                timestamp,
                0,
                bbox,
                score,
            )?;
            if let Some(crop) = person.tracking.as_ref().and_then(|t| t.next_crop) {
                detection_cache.follow(crop);
            }
            Ok(Some(person))
        })();
        match result {
            Ok(Some(mut person)) => {
                let (mut observation, image_constrained) = anatomical_observation(
                    &person,
                    timestamp,
                    config.options.threshold,
                    frame.image.size().map_err(|e| e.to_string())?,
                );
                if config.stabilization && config.smoothing {
                    motion.apply(&mut observation, &mut person.keypoints, frame.image.rows());
                }
                let points = person
                    .keypoints
                    .iter()
                    .map(|k| (k.name.to_string(), k.x, k.y, k.score))
                    .collect();
                let pose = Arc::new(PoseFrame {
                    source_video: frame.video,
                    index: frame.index,
                    captured: frame.captured,
                    observation,
                    hands: person.hands,
                    predicted_hands: person.predicted_hands,
                    image_constrained,
                    subject_segment: person.twin.as_ref().map(|t| t.subject_segment()),
                    refined_joints: person.arm_refined.len(),
                    points,
                    inference_ms: started.elapsed().as_secs_f64() * 1000.0,
                });
                let mut s = lock(&inner.state);
                s.pose = Some(pose);
                if s.status != "error" {
                    s.error = None; // only clears transient inference errors
                }
                if !s.ended && s.status != "error" {
                    s.status = "tracking".into();
                }
            }
            Ok(None) => {
                // No person / no 3D pose: drop the pose rather than leave a stale one.
                let mut s = lock(&inner.state);
                s.pose = None;
                s.status = if detector.is_some() && detection_cache.detections().is_empty() {
                    "no person detected".into()
                } else {
                    "pose unavailable for this frame".into()
                };
            }
            Err(e) => {
                let mut s = lock(&inner.state);
                s.pose = None;
                s.error = Some(e.to_string());
            }
        }
    }
}

/// Feed the shared skeletal fit (which solves reliable image constraints as well as
/// native depth) to the anatomical retargeter. Confidence gates stay attached:
/// held/predicted joints never become observations. Pixel-only depths are labelled
/// as inferred and receive lower confidence than independent native depth.
fn anatomical_observation(
    person: &crate::PersonPose,
    timestamp: f64,
    threshold: f32,
    size: opencv::core::Size,
) -> (Observation, usize) {
    let fitted = person
        .twin
        .as_ref()
        .map(|t| &t.fitted_pose)
        .or(person.pose3d.as_ref());
    let mut image_constrained = 0;
    let landmarks = fitted
        .map(|pose| {
            pose.joints
                .iter()
                .enumerate()
                .map(|(i, j)| {
                    let coco = match i {
                        11 => Some(5),
                        12 => Some(7),
                        13 => Some(9),
                        14 => Some(6),
                        15 => Some(8),
                        16 => Some(10),
                        _ => None,
                    };
                    let predicted = coco.is_some_and(|c| {
                        person
                            .joint_tracking
                            .iter()
                            .any(|u| u.joint == c && u.source == "predicted")
                    });
                    let native = person.pose3d.as_ref().and_then(|p| p.joints.get(i));
                    let pixel = coco.and_then(|c| person.keypoints.get(c));
                    // Native depth is not visibility evidence by itself. Reject offscreen /
                    // weak image joints rather than letting the body network invent an arm.
                    let visible = pixel.is_some_and(|p| {
                        crate::tracking_pipeline::image_supported(p, size, threshold.max(0.55))
                    });
                    let shoulder = match i {
                        11..=13 => Some(11),
                        14..=16 => Some(14),
                        _ => None,
                    };
                    let own_shoulder_visible = shoulder.is_some_and(|idx| {
                        let c = if idx == 11 { 5 } else { 6 };
                        person.keypoints.get(c).is_some_and(|p| {
                            crate::tracking_pipeline::image_supported(p, size, threshold.max(0.6))
                        })
                    });
                    let native_score = native.map_or(0.0, |p| p.input_score);
                    let (confidence, inferred) =
                        if coco.is_some() && (!visible || !own_shoulder_visible) {
                            (0.0, false)
                        } else {
                            constraint_confidence(
                                native_score,
                                pixel,
                                predicted,
                                person.twin.is_some(),
                                threshold.max(0.55),
                            )
                        };
                    // Independent native limbs must not inherit a global skeletal-fit
                    // correction driven by an uncertain opposite arm.
                    let position = if !inferred {
                        native
                            .map(|p| [p.x as f64, p.y as f64, p.z as f64])
                            .unwrap_or([j.x as f64, j.y as f64, j.z as f64])
                    } else {
                        let fitted_shoulder = shoulder.and_then(|i| pose.joints.get(i));
                        let native_shoulder = shoulder
                            .and_then(|i| person.pose3d.as_ref().and_then(|p| p.joints.get(i)));
                        match (fitted_shoulder, native_shoulder) {
                            (Some(a), Some(b)) => [
                                (j.x - a.x + b.x) as f64,
                                (j.y - a.y + b.y) as f64,
                                (j.z - a.z + b.z) as f64,
                            ],
                            _ => [j.x as f64, j.y as f64, j.z as f64],
                        }
                    };
                    image_constrained += usize::from(inferred);
                    Landmark {
                        name: j.name.to_string(),
                        position,
                        confidence,
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    (
        Observation {
            timestamp_seconds: timestamp,
            landmarks,
        },
        image_constrained,
    )
}

fn constraint_confidence(
    native: f32,
    pixel: Option<&crate::pose::Keypoint>,
    predicted: bool,
    fitted: bool,
    threshold: f32,
) -> (f64, bool) {
    if predicted {
        return (0.0, false);
    }
    if native.is_finite() && native > 0.0 && native >= threshold {
        return (f64::from(native.clamp(0.0, 1.0)), false);
    }
    if fitted {
        if let Some(p) = pixel {
            if p.score.is_finite()
                && p.score >= threshold
                && p.x.is_finite()
                && p.y.is_finite()
                && p.x >= 0.0
                && p.y >= 0.0
            {
                return (f64::from(p.score).min(0.49), true);
            }
        }
    }
    (0.0, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_depth_needs_current_image_support_and_a_skeletal_fit() {
        let mut pixel = crate::pose::Keypoint {
            name: "left_wrist",
            x: 100.0,
            y: 200.0,
            score: 0.9,
        };
        assert_eq!(
            constraint_confidence(0.0, Some(&pixel), false, true, 0.3),
            (0.49, true)
        );
        assert_eq!(
            constraint_confidence(0.0, Some(&pixel), false, false, 0.3),
            (0.0, false)
        );
        assert_eq!(
            constraint_confidence(0.9, Some(&pixel), true, true, 0.3),
            (0.0, false)
        );
        pixel.score = 0.0;
        assert_eq!(
            constraint_confidence(0.0, Some(&pixel), false, true, 0.3),
            (0.0, false)
        );
        pixel.score = 0.9;
        pixel.x = f32::NAN;
        assert_eq!(
            constraint_confidence(f32::NAN, Some(&pixel), false, true, 0.3),
            (0.0, false)
        );
        assert_eq!(constraint_confidence(0.8, None, false, false, 0.3).1, false);
    }

    #[test]
    fn hidden_arm_native_predictions_are_rejected_without_affecting_visible_arm() {
        let point = crate::pose::Keypoint {
            name: "joint",
            x: 100.0,
            y: 100.0,
            score: 0.9,
        };
        let mut person = crate::PersonPose {
            person_index: 0,
            bbox: Bbox {
                x: 0.0,
                y: 0.0,
                width: 640.0,
                height: 480.0,
            },
            detection_score: None,
            keypoints: vec![point; 17],
            observed_keypoints: vec![],
            observed_native_scores: None,
            joint_tracking: vec![],
            predicted_joints: vec![],
            body_wrists: [point; 2],
            arm_refined: vec![],
            angles: vec![],
            pose3d: Some(crate::motion3d::Pose3d {
                ear_head: false,
                model: "MediaPipe-Pose",
                coordinate_space: "root_relative_camera",
                units: "estimated_meters",
                history_frames: 1,
                angles: vec![],
                joints: crate::motion3d::NAMES
                    .iter()
                    .map(|&name| crate::motion3d::Joint3d {
                        name,
                        x: 0.1,
                        y: 0.2,
                        z: 0.3,
                        input_score: 0.95,
                    })
                    .collect(),
            }),
            twin: None,
            tracking: None,
            hands: vec![],
            predicted_hands: vec![],
            image_torso_scale: None,
            image_torso_reference_reset: false,
            image_torso_span: 1,
            head: None,
        };
        let size = opencv::core::Size::new(640, 480);
        let baseline = anatomical_observation(&person, 0.0, 0.3, size).0;
        for i in [6, 8, 10] {
            person.keypoints[i].score = 0.0;
        }
        crate::tracking_pipeline::gate_image_support(
            person.pose3d.as_mut().unwrap(),
            &person.keypoints,
            size,
            0.55,
        );
        for i in [14, 15, 16] {
            assert_eq!(person.pose3d.as_ref().unwrap().joints[i].input_score, 0.0);
        }
        let hidden = anatomical_observation(&person, 0.1, 0.3, size).0;
        for i in [14, 15, 16] {
            assert_eq!(hidden.landmarks[i].confidence, 0.0);
        }
        for i in [11, 12, 13] {
            assert_eq!(
                hidden.landmarks[i].confidence,
                baseline.landmarks[i].confidence
            );
            assert_eq!(hidden.landmarks[i].position, baseline.landmarks[i].position);
        }
        person.keypoints[7].x = 700.0;
        assert_eq!(
            anatomical_observation(&person, 0.2, 0.3, size).0.landmarks[12].confidence,
            0.0
        );
    }

    /// Exercises real model inference and the exact shared recording pipeline,
    /// without opening a camera or a window. Uses an existing local fixture only.
    #[test]
    #[ignore = "requires output/live-biomech/test-source.mp4 and installed ONNX models"]
    fn full_pipeline_file_input() {
        let capture = LiveCapture::start(Config {
            camera: 0,
            input: Some("output/live-biomech/test-source.mp4".into()),
            pose_model: "models/mediapipe-pose.onnx".into(),
            detector_model: "models/yolox-s.onnx".into(),
            no_detection: false,
            arm_model: Some("models/rtmpose-m.onnx".into()),
            hand_model: Some("models/handpose.onnx".into()),
            motion_model: "models/motionagformer-xs.onnx".into(),
            options: crate::tracking_pipeline::Options {
                full_body: false,
                image_support_floor: Some(0.55),
                no_3d: false,
                head_orientation: false,
                no_dimension_tracking: false,
                threshold: 0.3,
                focal_length_px: None,
            },
            stabilization: true,
            smoothing: true,
            upright_torso: true,
            acceleration: true,
            bbox: None,
            detection_interval: 15,
            detection_threshold: 0.4,
            nms_threshold: 0.45,
            opencv_threads: 4,
        })
        .unwrap();
        let model = Arc::new(
            biomech_lite::model::Model::load(std::path::Path::new(
                "assets/biomech-lite/model.json",
            ))
            .unwrap(),
        );
        let mut fitter = biomech_lite::live_fit::LiveFitter::new(model.clone()).unwrap();
        let mut fingers = biomech_lite::hand_fit::HandFitter::new(model.clone()).unwrap();
        let mut rigged = 0;
        let mut moved_fingers = false;
        let mut last = None;
        let (mut frames, mut refined, mut hands, mut inferred, mut supported) = (0, 0, 0, 0, 0);
        let start = Instant::now();
        let mut latency = Vec::new();
        while start.elapsed() < Duration::from_secs(25) {
            let state = capture.latest();
            assert!(state.error.is_none(), "{:?}", state.error);
            if let Some(pose) = state.pose {
                if last != Some(pose.index) {
                    last = Some(pose.index);
                    frames += 1;
                    refined += pose.refined_joints;
                    hands += pose.hands.len();
                    inferred += pose.image_constrained;
                    latency.push(pose.inference_ms);
                    let mut fit = fitter.update(&pose.observation).unwrap();
                    let hand_observations: Vec<_> = pose
                        .hands
                        .iter()
                        .map(|h| biomech_lite::hand_fit::HandObservation {
                            side: usize::from(h.side == "right"),
                            confidence: h.confidence as f64,
                            points: h.world_landmarks.map(|p| p.map(f64::from)),
                        })
                        .collect();
                    let hand_fit = fingers
                        .apply(
                            &mut fit.qpos,
                            &hand_observations,
                            pose.observation.timestamp_seconds,
                        )
                        .unwrap();
                    rigged += hand_fit.iter().filter(|&&v| v).count();
                    for name in [
                        "mcp2_flexion_l",
                        "mcp2_flexion_r",
                        "cmc_flexion_l",
                        "cmc_flexion",
                    ] {
                        let a = model.joints[model.joint_id(name).unwrap()].qposadr;
                        moved_fingers |= (fit.qpos[a] - model.standing_qpos[a]).abs() > 0.05;
                    }
                    supported += usize::from(fit.supported_arms.iter().any(|&v| v));
                    assert_eq!(fit.qpos.len(), model.nq);
                    assert!(fit.qpos.iter().all(|v| v.is_finite()));
                }
            }
            if frames >= 20 {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        eprintln!(
            "full pipeline: {frames} frames, {supported} supported fits, {refined} refinements, {hands} hands, {inferred} image-depth constraints; inference ms: {latency:?}"
        );
        assert!(
            frames >= 5 && supported >= 5,
            "insufficient supported tracking"
        );
        eprintln!("hand rig: {rigged} supported hands; fingers moved: {moved_fingers}");
        assert!(
            rigged > 0 && moved_fingers,
            "tracked hands did not articulate the model"
        );
        assert!(refined > 0, "arm refiner was never consumed");
        assert!(hands > 0, "dedicated hand tracking was never consumed");
    }
}
