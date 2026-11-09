//! Dedicated 3D hand inference and sparse-face head orientation.
//! Hand export/preprocessing: OpenCV Zoo handpose_estimation_mediapipe (2023feb).
use crate::pose::Keypoint;
use anyhow::{Context, Result, ensure};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use opencv::{
    core::{self, Mat, Point, Scalar, Size, Vector},
    dnn, imgproc,
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

pub const HAND_PARENT: [usize; 21] = [
    0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,
];
#[derive(Clone, Serialize, Deserialize)]
pub struct HandPose {
    pub side: String,
    pub confidence: f32,
    pub image_landmarks: [[f32; 2]; 21],
    /// Learned relative 3D hand coordinates, in the model's meter convention.
    pub world_landmarks: [[f32; 3]; 21],
}
#[derive(Clone, Serialize, Deserialize)]
pub struct HeadPose {
    pub orientation_xyzw: [f64; 4],
    pub confidence: f32,
    pub source: String,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct HandFrame {
    pub side: String,
    pub confidence: f32,
    /// Camera-space incoming-bone rotations, mapping +Y onto each bone direction.
    pub bone_rotations_xyzw: Vec<[f64; 4]>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct HandShape {
    pub bone_lengths: Vec<f64>,
    pub measurement_count: u64,
}

pub fn merge_hand_landmarks(points: &mut [Keypoint], hands: &[HandPose]) {
    merge_supported_hand_landmarks(points, hands, [false; 2]);
}

fn merge_supported_hand_landmarks(
    points: &mut [Keypoint],
    hands: &[HandPose],
    confirmed: [bool; 2],
) {
    if points.len() < 133 {
        return;
    }
    // Coarse body-model finger guesses are crop hints, not full hand detections.
    for point in &mut points[91..133] {
        point.x = -1.0;
        point.y = -1.0;
        point.score = 0.0;
    }
    for hand in hands {
        let base = if hand.side == "left" { 91 } else { 112 };
        let wrist = if hand.side == "left" { 9 } else { 10 };
        // An isolated marginal crop must not displace a confident body wrist.
        // Consistent independent hand observations can establish wrist support.
        let side = usize::from(hand.side == "right");
        if hand.confidence >= 0.85
            || points[wrist].score < 0.8
            || (confirmed[side] && hand.confidence >= 0.65)
        {
            points[wrist].x = hand.image_landmarks[0][0];
            points[wrist].y = hand.image_landmarks[0][1];
            points[wrist].score = hand.confidence;
        }
        for (i, pixel) in hand.image_landmarks.iter().enumerate() {
            points[base + i].x = pixel[0];
            points[base + i].y = pixel[1];
            points[base + i].score = hand.confidence;
        }
    }
}

// Geometrically duplicate palm crops can disagree on handedness. They still
// represent one observation and must never be assigned to both arms.
fn same_hand_detection(a: &HandPose, b: &HandPose) -> bool {
    let scale = palm_span(a).max(palm_span(b)).max(1.0);
    let distance = |i: usize| {
        (a.image_landmarks[i][0] - b.image_landmarks[i][0])
            .hypot(a.image_landmarks[i][1] - b.image_landmarks[i][1])
    };
    [0, 5, 9, 17]
        .into_iter()
        .all(|i| distance(i) < scale * 0.35)
        || ([0, 5, 9, 17]
            .into_iter()
            .all(|i| distance(i) < scale * 0.75)
            && (0..21).map(distance).sum::<f32>() / 21.0 < scale * 0.5)
}

struct HandTrack {
    hand: HandPose,
    previous: Option<(HandPose, f64)>,
    time: f64,
    strong_time: f64,
}
fn palm_span(hand: &HandPose) -> f32 {
    let p = &hand.image_landmarks;
    (p[5][0] - p[17][0])
        .hypot(p[5][1] - p[17][1])
        .max((p[0][0] - p[9][0]).hypot(p[0][1] - p[9][1]))
        .max(8.0)
}
impl HandTrack {
    fn occlusion_reference(
        &self,
        side: usize,
        time: f64,
    ) -> Option<crate::occlusion::HandReference> {
        let age = time - self.time;
        let (previous, previous_time) = self.previous.as_ref()?;
        let dt = self.time - previous_time;
        if !(0.0..=0.2).contains(&age)
            || !(1e-4..=0.1).contains(&dt)
            || self.hand.confidence < 0.65
            || previous.confidence < 0.65
        {
            return None;
        }
        let span = palm_span(&self.hand);
        let a = self.hand.image_landmarks[0];
        let b = previous.image_landmarks[0];
        if (a[0] - b[0]).hypot(a[1] - b[1]) > (span * 2.0).max(40.0) {
            return None;
        }
        Some(crate::occlusion::HandReference {
            side,
            wrist: self.prediction(time)?.image_landmarks[0],
            palm_span: span,
        })
    }
    fn prediction(&self, time: f64) -> Option<HandPose> {
        let gap = time - self.time;
        if !(0.0..=0.3).contains(&gap) {
            return None;
        }
        let mut hand = self.hand.clone();
        if let Some((previous, previous_time)) = &self.previous {
            let dt = self.time - previous_time;
            if dt > 1e-4 && dt < 0.5 {
                // Damped rigid hand motion only: do not invent finger articulation.
                let horizon = 0.10 * (1.0 - (-gap / 0.10).exp());
                let factor = (horizon / dt) as f32;
                let old = self.hand.image_landmarks[0];
                let before = previous.image_landmarks[0];
                let mut shift = [(old[0] - before[0]) * factor, (old[1] - before[1]) * factor];
                let limit = palm_span(&self.hand) * 1.5;
                let bound = (limit / shift[0].hypot(shift[1]).max(1e-6)).min(1.0);
                shift[0] *= bound;
                shift[1] *= bound;
                let scale = ((palm_span(&self.hand) / palm_span(previous)).ln() * factor)
                    .exp()
                    .clamp(0.8, 1.3);
                for p in &mut hand.image_landmarks {
                    *p = [
                        old[0] + shift[0] + (p[0] - old[0]) * scale,
                        old[1] + shift[1] + (p[1] - old[1]) * scale,
                    ];
                }
            }
        }
        hand.confidence *= (-gap / 0.15).exp() as f32;
        Some(hand)
    }
}
// Screen position cannot establish anatomical side after the person turns.
// Without a wrist observation or recent same-side track, leave the hand
// unassigned rather than letting it steal a detection from the observed arm.
fn hand_assignment_cost(
    hand: &HandPose,
    handedness: f32,
    guide: &Keypoint,
    previous: Option<&HandTrack>,
    scale: f32,
    side: &str,
    time: f64,
) -> f32 {
    // The relaxed recovery threshold belongs to an existing same-side track.
    // A marginal crop proposed for one arm must not create a new other hand.
    if hand.confidence < 0.65 && !previous.is_some_and(|track| time - track.strong_time <= 0.3) {
        return f32::INFINITY;
    }
    let wrist = hand.image_landmarks[0];
    let separation = |p: [f32; 2]| (wrist[0] - p[0]).hypot(wrist[1] - p[1]) / scale;
    let body = if guide.score >= 0.2
        && guide.x >= 0.0
        && guide.y >= 0.0
        && guide.x.is_finite()
        && guide.y.is_finite()
    {
        separation([guide.x, guide.y])
    } else if let Some(track) = previous {
        separation(track.hand.image_landmarks[0])
    } else {
        return f32::INFINITY;
    };
    // A marginal detection needs agreement with this person's wrist or hand
    // history. Otherwise a background hand can start or hijack a track.
    // Strong independent hand detections can still correct a bad body wrist.
    if hand.confidence < 0.85
        && body > 0.35
        && !previous.is_some_and(|track| separation(track.hand.image_landmarks[0]) <= 0.35)
    {
        return f32::INFINITY;
    }
    let distance = previous.map_or(body, |track| {
        0.6 * body + 0.4 * separation(track.hand.image_landmarks[0])
    });
    distance
        + 0.1 * (1.0 - hand.confidence)
        + 0.02
            * if side == "left" {
                handedness
            } else {
                1.0 - handedness
            }
}
pub struct HandModel {
    net: dnn::Net,
    accelerated: crate::accelerated::Inference,
    model_path: std::path::PathBuf,
    names: Vector<String>,
    palms: crate::palm::PalmDetector,
    previous: BTreeMap<String, HandTrack>,
    fast_tracking: bool,
}
impl HandModel {
    pub fn enable_acceleration(&mut self) {
        if let Err(error) = self.accelerated.enable_crop(&self.model_path) {
            eprintln!(
                "Native inference unavailable for {}: {error:#}; using OpenCV",
                self.model_path.display()
            );
        }
        self.palms.enable_acceleration();
    }
    pub fn new(path: &Path) -> Result<Self> {
        ensure!(
            path.is_file(),
            "Hand model missing at {}. Download the documented OpenCV Zoo model or use --no-hands.",
            path.display()
        );
        let mut net=opencv::opencv_branch_5! {{dnn::read_net_from_onnx_def(path)}else{dnn::read_net_from_onnx(path)}}.context("Cannot load hand model")?;
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        opencv::not_opencv_branch_5! {net.set_preferable_target(dnn::DNN_TARGET_CPU)?;}
        let names = net.get_unconnected_out_layers_names()?;
        let palms = crate::palm::PalmDetector::new(&path.with_file_name("palmdet.onnx"))?;
        Ok(Self {
            net,
            names,
            accelerated: crate::accelerated::Inference::default(),
            model_path: path.to_path_buf(),
            palms,
            previous: BTreeMap::new(),
            fast_tracking: false,
        })
    }
    pub fn enable_fast_tracking(&mut self) {
        self.fast_tracking = true;
    }
    pub fn warm_up(&mut self) -> Result<()> {
        let image = Mat::new_rows_cols_with_default(256, 256, core::CV_8UC3, Scalar::all(0.0))?;
        self.palms.infer(&image)?;
        self.infer_crop(
            &image,
            &HandCrop {
                ex: [1.0, 0.0],
                ey: [0.0, 1.0],
                center: [128.0, 128.0],
                side: 224.0,
            },
            1.1,
        )?;
        Ok(())
    }
    /// Missing-hand predictions are deliberately returned separately from detections.
    pub fn predictions(&self, time: f64) -> Vec<HandPose> {
        self.previous
            .values()
            .filter(|track| time - track.time > 1e-6)
            .filter_map(|track| track.prediction(time))
            .collect()
    }
    pub fn occlusion_references(&self, time: f64) -> Vec<crate::occlusion::HandReference> {
        [("left", 0), ("right", 1)]
            .into_iter()
            .filter_map(|(name, side)| self.previous.get(name)?.occlusion_reference(side, time))
            .collect()
    }
    /// Only fresh detections can replace a body wrist. Two consistent hand
    /// observations can establish support without promoting their confidence.
    pub fn merge_landmarks(&self, points: &mut [Keypoint], hands: &[HandPose], time: f64) {
        let confirmed = [("left", 0), ("right", 1)].map(|(name, side)| {
            self.previous.get(name).is_some_and(|track| {
                (track.time - time).abs() < 1e-6 && track.occlusion_reference(side, time).is_some()
            })
        });
        merge_supported_hand_landmarks(points, hands, confirmed);
    }
    pub fn infer(
        &mut self,
        image: &Mat,
        points: &[Keypoint],
        time: f64,
        bbox: Option<crate::pose::Bbox>,
        crop_wrists: Option<&[Keypoint; 2]>,
    ) -> Result<Vec<HandPose>> {
        if points.len() < 133 {
            return Ok(Vec::new());
        }
        self.previous
            .retain(|_, track| time >= track.time && time - track.time <= 0.5);
        // Retain the CNN crop proposals as well as flow-corrected wrists.
        // Moving a crop can lose a fist even when its center becomes better.
        let mut crop_points = points.to_vec();
        if let Some(wrists) = crop_wrists {
            crop_points[9..11].copy_from_slice(wrists);
        }
        let region = hand_detection_region(image.size()?, &crop_points, bbox);
        let mut candidates: Vec<(HandPose, f32)> = Vec::new();
        if self.fast_tracking {
            // Search the known hand directly, but require fresh, confident NN
            // landmarks. Forecasts only propose the crop; they are never output.
            let proposals: Vec<_> = ["left", "right"]
                .into_iter()
                .filter_map(|side| {
                    let track = self.previous.get(side)?;
                    if time - track.time > 0.2 || time - track.strong_time > 0.3 {
                        return None;
                    }
                    let hand = track.prediction(time)?;
                    let mut crop = HandCrop::from_hand(&hand)?;
                    crop.side *= 1.2;
                    Some((crop, hand.image_landmarks[0]))
                })
                .collect();
            for (crop, wrist) in proposals {
                if let Some(hand) = self.infer_crop(image, &crop, 0.75)? {
                    if (hand.0.image_landmarks[0][0] - wrist[0])
                        .hypot(hand.0.image_landmarks[0][1] - wrist[1])
                        <= crop.side * 0.3
                        && !candidates
                            .iter()
                            .any(|(old, _)| same_hand_detection(old, &hand.0))
                    {
                        candidates.push(hand);
                    }
                }
            }
        }
        let complete = candidates.len() == 2;
        let mut palms = if complete {
            Vec::new()
        } else if bbox.is_some() {
            self.palms.infer_region(image, region)?
        } else {
            self.palms.infer(image)?
        };
        for palm in &palms {
            if self.fast_tracking
                && candidates.iter().any(|(h, _)| {
                    (h.image_landmarks[0][0] - palm.landmarks[0][0])
                        .hypot(h.image_landmarks[0][1] - palm.landmarks[0][1])
                        < palm_span(h) * 0.6
                })
            {
                continue;
            }
            if let Some(crop) = HandCrop::from_palm(palm) {
                if let Some(hand) = self.infer_crop(image, &crop, 0.65)? {
                    if !candidates
                        .iter()
                        .any(|(old, _)| same_hand_detection(old, &hand.0))
                    {
                        candidates.push(hand);
                    }
                }
            }
        }
        // A small fist can disappear when the whole person is reduced to the
        // palm detector's 192px input. Reacquire near reliable body wrists only
        // when the broad crop has not already found that hand.
        let shoulder_span = (points[5].x - points[6].x)
            .hypot(points[5].y - points[6].y)
            .max(region[2] * 0.2);
        let mut wrist_proposals: Vec<_> = crop_points[9..11].iter().collect();
        for (side, wrist) in points[9..11].iter().enumerate() {
            let original = &crop_points[9 + side];
            if (wrist.x - original.x).hypot(wrist.y - original.y) > 1.0 {
                wrist_proposals.push(wrist);
            }
        }
        for wrist in wrist_proposals {
            if complete {
                break;
            }
            if wrist.score < 0.5
                || wrist.x < 0.0
                || wrist.y < 0.0
                || !wrist.x.is_finite()
                || !wrist.y.is_finite()
            {
                continue;
            }
            let size = (shoulder_span * 1.2).clamp(160.0, 480.0);
            if candidates.iter().any(|(hand, _)| {
                (hand.image_landmarks[0][0] - wrist.x).hypot(hand.image_landmarks[0][1] - wrist.y)
                    < size * 0.3
            }) {
                continue;
            }
            palms = self.palms.infer_region(
                image,
                [wrist.x - size / 2.0, wrist.y - size / 2.0, size, size],
            )?;
            for palm in &palms {
                if let Some(crop) = HandCrop::from_palm(palm) {
                    if let Some(hand) = self.infer_crop(image, &crop, 0.65)? {
                        if !candidates
                            .iter()
                            .any(|(old, _)| same_hand_detection(old, &hand.0))
                        {
                            candidates.push(hand);
                        }
                    }
                }
            }
        }
        // Reacquire even when coarse body fingers vanish. Previous hand bounds
        // are crop proposals only; every accepted hand gets fresh inference.
        for (side, base) in [("left", 91), ("right", 112)] {
            let previous = self
                .previous
                .get(side)
                .and_then(|track| track.prediction(time));
            let threshold = if self
                .previous
                .get(side)
                .is_some_and(|track| time - track.strong_time <= 0.3)
            {
                0.5
            } else {
                0.65
            };
            let crop = previous
                .as_ref()
                .and_then(|hand| HandCrop::from_hand(hand))
                .or_else(|| HandCrop::from_points(&points[base..base + 21]))
                .or_else(|| HandCrop::from_sparse_points(&points[base..base + 21]));
            if let Some(mut crop) = crop {
                if previous.is_some() {
                    crop.side *= 1.15;
                }
                let wrist = previous
                    .as_ref()
                    .map(|hand| hand.image_landmarks[0])
                    .unwrap_or(crop.center);
                let already = candidates.iter().any(|(h, _)| {
                    h.confidence >= 0.75
                        && (h.image_landmarks[0][0] - wrist[0])
                            .hypot(h.image_landmarks[0][1] - wrist[1])
                            < crop.side * 0.15
                });
                if !already {
                    let mut recovered = self.infer_crop(image, &crop, threshold)?;
                    if recovered.is_none() && previous.is_some() {
                        crop.side *= 1.4;
                        recovered = self.infer_crop(image, &crop, threshold)?;
                    }
                    if let Some(hand) = recovered {
                        if hand.0.confidence < 0.65
                            && (hand.0.image_landmarks[0][0] - wrist[0])
                                .hypot(hand.0.image_landmarks[0][1] - wrist[1])
                                > crop.side * 0.3
                        {
                            continue;
                        }
                        let duplicate = candidates
                            .iter()
                            .any(|(old, _)| same_hand_detection(old, &hand.0));
                        if !duplicate {
                            candidates.push(hand);
                        }
                    }
                }
            }
        }
        let width = image.cols() as f32;
        let scale = (points[5].x - points[6].x)
            .hypot(points[5].y - points[6].y)
            .max(width * 0.15);
        let cost = |side: &str, index: usize| {
            let (hand, handedness) = &candidates[index];
            hand_assignment_cost(
                hand,
                *handedness,
                &points[if side == "left" { 9 } else { 10 }],
                self.previous.get(side),
                scale,
                side,
                time,
            )
        };
        let mut best = (None, None, 4.0);
        for left in 0..=candidates.len() {
            for right in 0..=candidates.len() {
                if left < candidates.len()
                    && right < candidates.len()
                    && (left == right
                        || same_hand_detection(&candidates[left].0, &candidates[right].0))
                {
                    continue;
                }
                let score = if left < candidates.len() {
                    cost("left", left)
                } else {
                    2.0
                } + if right < candidates.len() {
                    cost("right", right)
                } else {
                    2.0
                };
                if score < best.2 {
                    best = (
                        if left < candidates.len() {
                            Some(left)
                        } else {
                            None
                        },
                        if right < candidates.len() {
                            Some(right)
                        } else {
                            None
                        },
                        score,
                    );
                }
            }
        }
        let mut hands = Vec::new();
        for (side, index) in [("left", best.0), ("right", best.1)] {
            if let Some(index) = index {
                let mut hand = candidates[index].0.clone();
                hand.side = side.into();
                let old = self.previous.remove(side);
                let strong_time = if hand.confidence >= 0.65 {
                    time
                } else {
                    old.as_ref().map(|track| track.strong_time).unwrap_or(time)
                };
                self.previous.insert(
                    side.into(),
                    HandTrack {
                        hand: hand.clone(),
                        time,
                        strong_time,
                        previous: old.map(|track| (track.hand, track.time)),
                    },
                );
                hands.push(hand);
            }
        }
        Ok(hands)
    }
    fn infer_crop(
        &mut self,
        image: &Mat,
        crop: &HandCrop,
        threshold: f32,
    ) -> Result<Option<(HandPose, f32)>> {
        let affine = Mat::from_slice_2d(&crop.affine())?;
        let mut warped = Mat::default();
        imgproc::warp_affine_def(image, &mut warped, &affine, Size::new(224, 224))?;
        let mut rgb = Mat::default();
        imgproc::cvt_color_def(&warped, &mut rgb, imgproc::COLOR_BGR2RGB)?;
        let mut float = Mat::default();
        rgb.convert_to(&mut float, core::CV_32F, 1.0 / 255.0, 0.0)?;
        let blob = float.reshape_nd(1, &[1, 224, 224, 3])?;
        let outputs = self
            .accelerated
            .forward(&mut self.net, &blob, &self.names)?;
        let output = |name: &str| -> Result<Mat> {
            let index = self
                .names
                .iter()
                .position(|n| n == name)
                .with_context(|| format!("Missing hand output {name}"))?;
            Ok(outputs.get(index)?)
        };
        let confidence = output("Identity_1")?.data_typed::<f32>()?[0];
        if !confidence.is_finite() || confidence < threshold {
            return Ok(None);
        }
        let screen = output("Identity")?;
        let world = output("Identity_3")?;
        let screen = screen.data_typed::<f32>()?;
        let world = world.data_typed::<f32>()?;
        ensure!(
            screen.len() == 63 && world.len() == 63,
            "Unexpected hand landmark shapes"
        );
        if screen.iter().chain(world).any(|v| !v.is_finite()) {
            return Ok(None);
        }
        let image_landmarks = std::array::from_fn(|i| crop.image(screen[i * 3], screen[i * 3 + 1]));
        let world_landmarks = std::array::from_fn(|i| {
            [
                world[i * 3] * crop.ex[0] + world[i * 3 + 1] * crop.ey[0],
                world[i * 3] * crop.ex[1] + world[i * 3 + 1] * crop.ey[1],
                world[i * 3 + 2],
            ]
        });
        let handedness = output("Identity_2")?.data_typed::<f32>()?[0];
        Ok(Some((
            HandPose {
                side: String::new(),
                confidence,
                image_landmarks,
                world_landmarks,
            },
            handedness,
        )))
    }
}

fn hand_detection_region(
    size: Size,
    points: &[Keypoint],
    bbox: Option<crate::pose::Bbox>,
) -> [f32; 4] {
    let Some(bbox) = bbox.filter(|b| {
        [b.x, b.y, b.width, b.height].iter().all(|v| v.is_finite())
            && b.width > 0.0
            && b.height > 0.0
    }) else {
        return [0.0, 0.0, size.width as f32, size.height as f32];
    };
    let mut min = [bbox.x - bbox.width * 0.15, bbox.y - bbox.height * 0.15];
    let mut max = [bbox.x + bbox.width * 1.15, bbox.y + bbox.height * 1.15];
    let margin = (bbox.width * 0.25).max(60.0);
    for p in points.iter().take(11).skip(5).filter(|p| {
        p.score >= 0.5 && p.x.is_finite() && p.y.is_finite() && p.x >= 0.0 && p.y >= 0.0
    }) {
        min[0] = min[0].min(p.x - margin);
        min[1] = min[1].min(p.y - margin);
        max[0] = max[0].max(p.x + margin);
        max[1] = max[1].max(p.y + margin);
    }
    min[0] = min[0].clamp(0.0, (size.width - 1).max(0) as f32);
    min[1] = min[1].clamp(0.0, (size.height - 1).max(0) as f32);
    max[0] = max[0].clamp(min[0] + 1.0, size.width as f32);
    max[1] = max[1].clamp(min[1] + 1.0, size.height as f32);
    [min[0], min[1], max[0] - min[0], max[1] - min[1]]
}

struct HandCrop {
    ex: [f32; 2],
    ey: [f32; 2],
    /// Center in original image coordinates (not in the rotated basis).
    center: [f32; 2],
    side: f32,
}
impl HandCrop {
    /// A fresh-inference proposal when the palm detector cannot resolve a fist.
    /// Forearm direction is a crop guide, not a measured finger orientation.
    #[cfg(test)]
    fn from_forearm(
        elbow: Keypoint,
        wrist: Keypoint,
        shoulder_span: f32,
        factor: f32,
    ) -> Option<Self> {
        if elbow.score < 0.5
            || wrist.score < 0.5
            || [elbow.x, elbow.y, wrist.x, wrist.y, shoulder_span]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return None;
        }
        let dx = elbow.x - wrist.x;
        let dy = elbow.y - wrist.y;
        let length = dx.hypot(dy);
        if length < 10.0 {
            return None;
        }
        let ey = [dx / length, dy / length];
        let ex = [ey[1], -ey[0]];
        let side = (length * factor)
            .max(shoulder_span * 0.35)
            .clamp(48.0, 480.0);
        Some(Self {
            ex,
            ey,
            center: [wrist.x - ey[0] * side * 0.22, wrist.y - ey[1] * side * 0.22],
            side,
        })
    }
    fn from_palm(palm: &crate::palm::Palm) -> Option<Self> {
        let p = &palm.landmarks;
        let dx = p[0][0] - p[2][0];
        let dy = p[0][1] - p[2][1];
        let length = dx.hypot(dy);
        if length < 3.0 {
            return None;
        }
        let ey = [dx / length, dy / length];
        let ex = [ey[1], -ey[0]];
        let projected = p.map(|p| [p[0] * ex[0] + p[1] * ex[1], p[0] * ey[0] + p[1] * ey[1]]);
        let minx = projected.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let maxx = projected
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let miny = projected.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let maxy = projected
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let center = [
            (minx + maxx) / 2.0,
            (miny + maxy) / 2.0 - 0.4 * (maxy - miny),
        ];
        let side = (maxx - minx).max(maxy - miny) * 3.0;
        if !(24.0..4000.0).contains(&side) {
            return None;
        }
        Some(Self {
            ex,
            ey,
            side,
            center: [
                center[0] * ex[0] + center[1] * ey[0],
                center[0] * ex[1] + center[1] * ey[1],
            ],
        })
    }
    fn from_hand(hand: &HandPose) -> Option<Self> {
        let points: Vec<_> = hand
            .image_landmarks
            .iter()
            .map(|p| Keypoint {
                name: "hand",
                x: p[0],
                y: p[1],
                score: 1.0, // Crop guides carry no measurement confidence.
            })
            .collect();
        Self::from_points(&points)
    }
    fn from_sparse_points(p: &[Keypoint]) -> Option<Self> {
        // MediaPipe Pose provides wrist, thumb, index and pinky anchors. Use
        // interpolated crop guides internally; never export them as finger joints.
        if [0, 4, 8, 20]
            .iter()
            .any(|&i| p[i].score < 0.2 || !p[i].x.is_finite() || !p[i].y.is_finite())
        {
            return None;
        }
        let mut guides = p.to_vec();
        for (target, a, b) in [(9, 8, 20), (5, 0, 8), (17, 0, 20)] {
            guides[target].x = (p[a].x + p[b].x) / 2.0;
            guides[target].y = (p[a].y + p[b].y) / 2.0;
            guides[target].score = p[a].score.min(p[b].score);
        }
        Self::from_points(&guides)
    }
    fn from_points(p: &[Keypoint]) -> Option<Self> {
        let valid = |p: &Keypoint| {
            p.score >= 0.2 && p.x.is_finite() && p.y.is_finite() && p.x >= 0.0 && p.y >= 0.0
        };
        if !valid(&p[0]) || !valid(&p[9]) {
            return None;
        }
        let dx = p[0].x - p[9].x;
        let dy = p[0].y - p[9].y;
        let length = dx.hypot(dy);
        if length < 5.0 {
            return None;
        }
        let ey = [dx / length, dy / length];
        let ex = [ey[1], -ey[0]];
        let usable: Vec<_> = p
            .iter()
            .filter(|p| valid(p))
            .map(|p| [p.x * ex[0] + p.y * ex[1], p.x * ey[0] + p.y * ey[1]])
            .collect();
        if usable.len() < 6 {
            return None;
        }
        let minx = usable.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let maxx = usable
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let miny = usable.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let maxy = usable
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let side = (maxx - minx).max(maxy - miny) * 1.5;
        if !(16.0..2000.0).contains(&side) {
            return None;
        }
        Some(Self {
            ex,
            ey,
            center: [
                (minx + maxx) / 2.0 * ex[0] + (miny + maxy) / 2.0 * ey[0],
                (minx + maxx) / 2.0 * ex[1] + (miny + maxy) / 2.0 * ey[1],
            ],
            side,
        })
    }
    fn affine(&self) -> [[f64; 3]; 2] {
        let s = 224.0 / self.side as f64;
        [
            [
                s * self.ex[0] as f64,
                s * self.ex[1] as f64,
                112.0 - s * (self.ex[0] * self.center[0] + self.ex[1] * self.center[1]) as f64,
            ],
            [
                s * self.ey[0] as f64,
                s * self.ey[1] as f64,
                112.0 - s * (self.ey[0] * self.center[0] + self.ey[1] * self.center[1]) as f64,
            ],
        ]
    }
    fn image(&self, x: f32, y: f32) -> [f32; 2] {
        let u = (x - 112.0) * self.side / 224.0;
        let v = (y - 112.0) * self.side / 224.0;
        [
            self.center[0] + u * self.ex[0] + v * self.ey[0],
            self.center[1] + u * self.ex[1] + v * self.ey[1],
        ]
    }
}

/// Generic sparse face fit under weak perspective. This is approximate head
/// orientation; it is not facial-surface reconstruction or measured neck rotation.
pub fn head_pose(points: &[Keypoint], threshold: f32) -> Option<HeadPose> {
    if points.len() < 5 {
        return None;
    }
    let canonical = [
        Vector3::new(0.0, 0.0, -0.5),
        Vector3::new(0.35, -0.25, -0.2),
        Vector3::new(-0.35, -0.25, -0.2),
        Vector3::new(0.65, 0.0, 0.0),
        Vector3::new(-0.65, 0.0, 0.0),
    ];
    let usable: Vec<_> = (0..5)
        .filter(|&i| {
            points[i].score >= threshold.max(0.4)
                && points[i].x >= 0.0
                && points[i].y >= 0.0
                && points[i].x.is_finite()
                && points[i].y.is_finite()
        })
        .collect();
    if usable.len() < 4 {
        return None;
    }
    let score: f64 = usable
        .iter()
        .map(|&i| points[i].score.clamp(0.0, 1.0) as f64)
        .sum::<f64>()
        / usable.len() as f64;
    let fit = |q: UnitQuaternion<f64>| {
        let n = usable.len() as f64;
        let mean = usable
            .iter()
            .map(|&i| q * canonical[i])
            .fold(Vector3::zeros(), |a, b| a + b)
            / n;
        let center = usable
            .iter()
            .map(|&i| Vector3::new(points[i].x as f64, points[i].y as f64, 0.0))
            .fold(Vector3::zeros(), |a, b| a + b)
            / n;
        let numerator = usable
            .iter()
            .map(|&i| {
                let p = q * canonical[i] - mean;
                p.x * (points[i].x as f64 - center.x) + p.y * (points[i].y as f64 - center.y)
            })
            .sum::<f64>();
        let denominator = usable
            .iter()
            .map(|&i| {
                let p = q * canonical[i] - mean;
                p.x * p.x + p.y * p.y
            })
            .sum::<f64>();
        let scale = numerator / denominator.max(1e-9);
        let translation = center - mean * scale;
        let residual = usable
            .iter()
            .map(|&i| {
                let p = q * canonical[i] * scale + translation;
                (p.x - points[i].x as f64).powi(2) + (p.y - points[i].y as f64).powi(2)
            })
            .sum::<f64>();
        (residual, scale, translation)
    };
    let mut best = None;
    for yaw in [-0.8, 0.0, 0.8] {
        let mut q = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
        for _ in 0..15 {
            let (cost, scale, t) = fit(q);
            if scale <= 2.0 {
                break;
            }
            let mut a = Matrix3::identity() * 0.01;
            let mut b = Vector3::zeros();
            for &i in &usable {
                let p = q * canonical[i];
                let r = Vector3::new(
                    points[i].x as f64 - (p.x * scale + t.x),
                    points[i].y as f64 - (p.y * scale + t.y),
                    0.0,
                );
                let j = Matrix3::new(
                    0.0,
                    p.z * scale,
                    -p.y * scale,
                    -p.z * scale,
                    0.0,
                    p.x * scale,
                    0.0,
                    0.0,
                    0.0,
                );
                a += j.transpose() * j;
                b += j.transpose() * r;
            }
            let Some(mut delta) = a.lu().solve(&b) else {
                break;
            };
            if delta.norm() > 0.2 {
                delta *= 0.2 / delta.norm();
            }
            let candidate = UnitQuaternion::from_scaled_axis(delta) * q;
            if fit(candidate).0 < cost {
                q = candidate;
            } else {
                break;
            }
        }
        let (cost, scale, _) = fit(q);
        if scale > 2.0
            && cost.is_finite()
            && cost.sqrt() / scale < 0.35
            && best.as_ref().is_none_or(|(c, _)| cost < *c)
        {
            best = Some((cost, q));
        }
    }
    let (_, q) = best?;
    let q = q.quaternion();
    Some(HeadPose {
        orientation_xyzw: [q.i, q.j, q.k, q.w],
        confidence: score as f32,
        source: "sparse_face_weak_perspective".into(),
    })
}

pub fn draw_hands(image: &mut Mat, hands: &[HandPose]) -> Result<()> {
    draw_hands_colored(image, hands, Scalar::new(220.0, 90.0, 230.0, 0.0))
}
pub fn draw_predicted_hands(image: &mut Mat, hands: &[HandPose]) -> Result<()> {
    draw_hands_colored(image, hands, Scalar::all(145.0))?;
    for hand in hands {
        let p = hand.image_landmarks[0];
        imgproc::put_text(
            image,
            "pred",
            Point::new(p[0] as i32 + 5, p[1] as i32 + 12),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.35,
            Scalar::all(170.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(())
}
fn draw_hands_colored(image: &mut Mat, hands: &[HandPose], color: Scalar) -> Result<()> {
    for hand in hands {
        let point = |i: usize| {
            Point::new(
                hand.image_landmarks[i][0] as i32,
                hand.image_landmarks[i][1] as i32,
            )
        };
        for i in 1..21 {
            imgproc::line(
                image,
                point(HAND_PARENT[i]),
                point(i),
                color,
                2,
                imgproc::LINE_AA,
                0,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlap_reference_requires_two_consistent_recent_hand_detections() {
        let hand = |x| HandPose {
            side: "left".into(),
            confidence: 0.7,
            image_landmarks: std::array::from_fn(|i| [x + i as f32, 100.0]),
            world_landmarks: [[0.0; 3]; 21],
        };
        let mut track = HandTrack {
            hand: hand(110.0),
            previous: Some((hand(100.0), 0.0)),
            time: 1.0 / 30.0,
            strong_time: 1.0 / 30.0,
        };
        assert!(track.occlusion_reference(0, 0.1).is_some());
        assert!(track.occlusion_reference(0, 0.24).is_none());
        assert!(track.occlusion_reference(0, -0.1).is_none());
        track.hand.confidence = 0.64;
        assert!(track.occlusion_reference(0, 0.1).is_none());
        track.hand.confidence = 0.7;
        track.previous = Some((hand(0.0), 0.0));
        assert!(track.occlusion_reference(0, 0.1).is_none());
        track.previous = Some((hand(100.0), -0.2));
        assert!(track.occlusion_reference(0, 0.1).is_none());
        track.previous = None;
        assert!(track.occlusion_reference(0, 0.1).is_none());
    }

    #[test]
    #[ignore = "requires cached recording observations; evaluates short hand overlap gating"]
    fn evaluate_recent_hand_overlap() -> Result<()> {
        use serde_json::Value;
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let mut history: BTreeMap<String, HandTrack> = BTreeMap::new();
        let mut output = String::new();
        let mut changes = Vec::new();
        let video = opencv::videoio::VideoCapture::from_file_def(
            directory.join("source.mp4").to_str().unwrap(),
        )?;
        let fps = video.get(opencv::videoio::CAP_PROP_FPS)?;
        for line in std::fs::read_to_string(directory.join("handflow3.jsonl"))?.lines() {
            let mut row: Value = serde_json::from_str(line)?;
            let time = row["frame"].as_u64().unwrap() as f64 / fps;
            let size = Size::new(
                row["width"].as_i64().unwrap() as i32,
                row["height"].as_i64().unwrap() as i32,
            );
            for person in row["people"].as_array_mut().unwrap() {
                let mut points: Vec<Keypoint> = person["keypoints"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| Keypoint {
                        name: "diagnostic",
                        x: p["x"].as_f64().unwrap() as f32,
                        y: p["y"].as_f64().unwrap() as f32,
                        score: p["score"].as_f64().unwrap() as f32,
                    })
                    .collect();
                for (side, wrist) in [9, 10].into_iter().enumerate() {
                    let p = &person["body_wrists"][side];
                    points[wrist].x = p["x"].as_f64().unwrap() as f32;
                    points[wrist].y = p["y"].as_f64().unwrap() as f32;
                    points[wrist].score = p["score"].as_f64().unwrap() as f32;
                }
                let references: Vec<_> = [("left", 0), ("right", 1)]
                    .into_iter()
                    .filter_map(|(name, side)| history.get(name)?.occlusion_reference(side, time))
                    .collect();
                let rejected =
                    crate::occlusion::mask_recent_hand_overlap(&mut points, size, &references);
                let hands: Vec<HandPose> = serde_json::from_value(person["hands"].clone())?;
                for (side, wrist, joint) in [(0, 9, 13), (1, 10, 16)] {
                    if rejected[wrist] {
                        changes.push(serde_json::json!({"time":time, "wrist":wrist, "x":points[wrist].x, "y":points[wrist].y, "fresh_hands":hands.iter().map(|h|h.side.as_str()).collect::<Vec<_>>()}));
                        person["body_wrists"][side]["score"] = 0.0.into();
                        person["pose3d"]["joints"][joint]["input_score"] = 0.0.into();
                    }
                }
                merge_hand_landmarks(&mut points, &hands);
                for i in [9, 10] {
                    person["keypoints"][i]["x"] = points[i].x.into();
                    person["keypoints"][i]["y"] = points[i].y.into();
                    person["keypoints"][i]["score"] = points[i].score.into();
                }
                history.retain(|_, track| time >= track.time && time - track.time <= 0.5);
                for hand in hands {
                    let side = hand.side.clone();
                    let old = history.remove(&side);
                    let strong_time = if hand.confidence >= 0.65 {
                        time
                    } else {
                        old.as_ref().map(|t| t.strong_time).unwrap_or(time)
                    };
                    history.insert(
                        side,
                        HandTrack {
                            hand,
                            previous: old.map(|t| (t.hand, t.time)),
                            time,
                            strong_time,
                        },
                    );
                }
            }
            output.push_str(&serde_json::to_string(&row)?);
            output.push('\n');
        }
        std::fs::write(directory.join("handguard.jsonl"), output)?;
        std::fs::write(
            directory.join("handguard-changes.json"),
            serde_json::to_vec_pretty(&changes)?,
        )?;
        println!("{} wrist observations gated", changes.len());
        Ok(())
    }
    #[test]
    #[ignore = "requires local recording/models; evaluates fresh coarse-hand crop proposals"]
    fn evaluate_coarse_hand_recovery() -> Result<()> {
        use opencv::{imgcodecs, videoio};
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let baseline: Vec<serde_json::Value> =
            std::fs::read_to_string(directory.join("handflow3.jsonl"))?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
        let mut pose = crate::pose::PoseEstimator::new(Path::new("models/mediapipe-pose.onnx"))?;
        let mut model = HandModel::new(Path::new("models/handpose.onnx"))?;
        let mut video =
            videoio::VideoCapture::from_file_def(directory.join("source.mp4").to_str().unwrap())?;
        let mut result = Vec::new();
        for (n, row) in baseline.iter().enumerate() {
            let mut image = Mat::default();
            ensure!(video.read(&mut image)?, "Missing frame {n}");
            let person = &row["people"][0];
            let b = &person["bbox"];
            let bbox = crate::pose::Bbox {
                x: b["x"].as_f64().unwrap() as f32,
                y: b["y"].as_f64().unwrap() as f32,
                width: b["width"].as_f64().unwrap() as f32,
                height: b["height"].as_f64().unwrap() as f32,
            };
            let mut points = pose.infer(&image, bbox)?;
            crate::scope::mask_details(&mut points);
            crate::scope::mask_lower(&mut points);
            crate::occlusion::mask_arm_overlap(&mut points, image.size()?);
            let old: Vec<HandPose> = serde_json::from_value(person["hands"].clone())?;
            pose.confirm_hands(&old);
            for (side, base) in [("left", 91), ("right", 112)] {
                if old.iter().any(|h| h.side == side && h.confidence >= 0.75) {
                    continue;
                }
                let hand = if let Some(crop) = HandCrop::from_points(&points[base..base + 21])
                    .or_else(|| HandCrop::from_sparse_points(&points[base..base + 21]))
                {
                    model.infer_crop(&image, &crop, 0.65)?.map(|(mut h, _)| {
                        h.side = side.into();
                        h
                    })
                } else {
                    None
                };
                let duplicate = hand.as_ref().is_some_and(|h| {
                    old.iter()
                        .any(|o| o.side != side && same_hand_detection(h, o))
                });
                if let Some(h) = &hand {
                    let mut overlay = image.try_clone()?;
                    draw_hands_colored(
                        &mut overlay,
                        &[h.clone()],
                        Scalar::new(0.0, 255.0, 255.0, 0.0),
                    )?;
                    imgcodecs::imwrite(
                        directory
                            .join(format!("coarsehand-{side}-{n:04}.jpg"))
                            .to_str()
                            .unwrap(),
                        &overlay,
                        &Vector::new(),
                    )?;
                }
                result.push(serde_json::json!({"frame":n,"side":side,"hand":hand,
                    "opposite_duplicate":duplicate,"baseline_present":old.iter().any(|h|h.side==side)}));
            }
        }
        std::fs::write(
            directory.join("coarse-hand-recovery.json"),
            serde_json::to_vec(&result)?,
        )?;
        Ok(())
    }
    #[test]
    fn sideways_punch_hand_is_not_claimed_by_an_unobserved_arm() {
        // Recorded frame 293: the visible right hand is on screen right.
        let hand = HandPose {
            side: String::new(),
            confidence: 0.6616,
            image_landmarks: [[896.78, 335.62]; 21],
            world_landmarks: [[0.0; 3]; 21],
        };
        let hidden = Keypoint {
            name: "left",
            x: 882.97,
            y: 415.32,
            score: 0.1333,
        };
        let visible = Keypoint {
            name: "right",
            x: 906.68,
            y: 338.02,
            score: 0.5963,
        };
        let track = HandTrack {
            hand: HandPose {
                image_landmarks: [[927.0, 329.0]; 21],
                ..hand.clone()
            },
            previous: None,
            time: 0.0,
            strong_time: 0.0,
        };
        assert!(hand_assignment_cost(&hand, 0.5, &hidden, None, 192.0, "left", 0.0).is_infinite());
        assert!(
            hand_assignment_cost(&hand, 0.5, &visible, Some(&track), 192.0, "right", 0.0) < 0.2
        );
        // Frame 298: a marginal background hand is too far from this
        // person's wrist to initialize a new left-hand track.
        let background = HandPose {
            confidence: 0.688,
            image_landmarks: [[806.0, 497.0]; 21],
            ..hand.clone()
        };
        let body = Keypoint {
            x: 860.37,
            y: 431.94,
            score: 0.548,
            ..visible
        };
        assert!(
            hand_assignment_cost(&background, 0.5, &body, None, 192.0, "left", 0.0).is_infinite()
        );
        assert!(
            hand_assignment_cost(
                &background,
                0.5,
                &visible,
                Some(&track),
                192.0,
                "right",
                0.0
            )
            .is_infinite()
        );
        let strong = HandPose {
            confidence: 0.95,
            ..background.clone()
        };
        assert!(hand_assignment_cost(&strong, 0.5, &body, None, 192.0, "left", 0.0).is_finite());
        let marginal = HandPose {
            confidence: 0.60,
            ..hand.clone()
        };
        assert!(
            hand_assignment_cost(&marginal, 0.5, &visible, None, 192.0, "left", 0.1).is_infinite()
        );
        assert!(
            hand_assignment_cost(&marginal, 0.5, &visible, Some(&track), 192.0, "right", 0.1)
                .is_finite()
        );
        assert!(
            hand_assignment_cost(&marginal, 0.5, &visible, Some(&track), 192.0, "right", 0.4)
                .is_infinite()
        );
        // A recent same-side track still permits genuine wrist-dropout recovery.
        assert!(hand_assignment_cost(&hand, 0.5, &hidden, Some(&track), 192.0, "left", 0.0) < 0.3);
    }
    #[test]
    #[ignore = "requires local recording and hand model; compares crop proposals only"]
    fn evaluate_hand_crop_context() -> Result<()> {
        use opencv::videoio;
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let baseline: Vec<serde_json::Value> =
            std::fs::read_to_string(directory.join("verifiedarms.jsonl"))?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
        let mut model = HandModel::new(Path::new("models/handpose.onnx"))?;
        let mut video =
            videoio::VideoCapture::from_file_def(directory.join("source.mp4").to_str().unwrap())?;
        let mut result = Vec::new();
        let mut image = Mat::default();
        for n in 0..baseline.len() {
            ensure!(video.read(&mut image)?, "Missing frame {n}");
            if n == 0 {
                continue;
            }
            let previous: Vec<HandPose> =
                serde_json::from_value(baseline[n - 1]["people"][0]["hands"].clone())?;
            let current: Vec<HandPose> =
                serde_json::from_value(baseline[n]["people"][0]["hands"].clone())?;
            for old in previous {
                let Some(base) = HandCrop::from_hand(&old) else {
                    continue;
                };
                for (name, scale, shift) in [
                    ("current", 1.15, 0.0),
                    ("wide", 2.0 / 1.5, 0.0),
                    ("wide_shift", 2.0 / 1.5, -0.1),
                ] {
                    let mut crop = HandCrop {
                        ex: base.ex,
                        ey: base.ey,
                        center: base.center,
                        side: base.side,
                    };
                    crop.side *= scale;
                    for axis in 0..2 {
                        crop.center[axis] += crop.ey[axis] * base.side / 1.5 * shift;
                    }
                    let hand = model.infer_crop(&image, &crop, 0.5)?.map(|(h, _)| h);
                    let duplicate = hand.as_ref().is_some_and(|h| {
                        current
                            .iter()
                            .any(|c| c.side != old.side && same_hand_detection(h, c))
                    });
                    result.push(serde_json::json!({"frame":n,"side":old.side,
                        "variant":name,"hand":hand,"opposite_duplicate":duplicate,
                        "baseline_present":current.iter().any(|h|h.side==old.side)}));
                }
            }
        }
        std::fs::write(
            directory.join("hand-crop-context.json"),
            serde_json::to_vec(&result)?,
        )?;
        Ok(())
    }
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires LIFTING_LIVE_EVAL, LIFTING_LIVE_SOURCE and native hand inference; no camera or UI"]
    fn native_hand_matches_opencv_on_the_same_recorded_crop() -> Result<()> {
        use opencv::videoio;
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_LIVE_EVAL")?);
        let row: serde_json::Value = serde_json::from_str(
            std::fs::read_to_string(directory.join("neural.jsonl"))?
                .lines()
                .nth(60)
                .context("Missing frame")?,
        )?;
        let hands: Vec<HandPose> = serde_json::from_value(row["people"][0]["hands"].clone())?;
        let hand = hands
            .iter()
            .find(|h| h.confidence > 0.8)
            .context("Missing reference hand")?;
        let crop = HandCrop::from_hand(hand).context("Missing crop")?;
        let mut capture =
            videoio::VideoCapture::from_file_def(&std::env::var("LIFTING_LIVE_SOURCE")?)?;
        capture.set(videoio::CAP_PROP_POS_FRAMES, 60.0)?;
        let mut image = Mat::default();
        ensure!(capture.read(&mut image)?, "Missing image");
        let mut reference = HandModel::new(Path::new("models/handpose.onnx"))?;
        let mut native = HandModel::new(Path::new("models/handpose.onnx"))?;
        native.enable_acceleration();
        native.accelerated.force_native();
        ensure!(
            native.accelerated.active(),
            "Native hand provider did not initialize"
        );
        let (a, ha) = reference
            .infer_crop(&image, &crop, 0.0)?
            .context("Missing reference output")?;
        let (b, hb) = native
            .infer_crop(&image, &crop, 0.0)?
            .context("Missing native output")?;
        ensure!(
            native.accelerated.active(),
            "Native hand fell back to OpenCV"
        );
        let max = a
            .image_landmarks
            .iter()
            .zip(b.image_landmarks)
            .map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]))
            .fold(0.0f32, f32::max);
        let world = a
            .world_landmarks
            .iter()
            .flatten()
            .zip(b.world_landmarks.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "native hand vs OpenCV: max {max:.3}px, world-component max {world:.6}m, confidence delta {:.5}",
            (a.confidence - b.confidence).abs()
        );
        ensure!(
            max < 3.0
                && world < 0.002
                && (a.confidence - b.confidence).abs() < 0.02
                && (ha - hb).abs() < 0.02,
            "Native hand diverged from reference"
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires auxiliary RTMW observations and local recording/models"]
    fn evaluate_auxiliary_hand_crops() -> Result<()> {
        use opencv::{imgcodecs, videoio};
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let input: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(directory.join("auxrtmw.json"))?)?;
        let baseline: Vec<serde_json::Value> =
            std::fs::read_to_string(directory.join("verifiedarms.jsonl"))?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
        let mut model = HandModel::new(Path::new("models/handpose.onnx"))?;
        let mut video =
            videoio::VideoCapture::from_file_def(directory.join("source.mp4").to_str().unwrap())?;
        let mut result = Vec::new();
        for row in input {
            let n = row["frame"].as_u64().unwrap() as usize;
            video.set(videoio::CAP_PROP_POS_FRAMES, n as f64)?;
            let mut image = Mat::default();
            ensure!(video.read(&mut image)?, "Missing frame");
            let points: Vec<_> = row["keypoints"]
                .as_array()
                .unwrap()
                .iter()
                .map(|k| Keypoint {
                    name: "guide",
                    x: k["x"].as_f64().unwrap() as f32,
                    y: k["y"].as_f64().unwrap() as f32,
                    score: k["score"].as_f64().unwrap() as f32,
                })
                .collect();
            let old: Vec<HandPose> =
                serde_json::from_value(baseline[n]["people"][0]["hands"].clone())?;
            for (side, base) in [("left", 91), ("right", 112)] {
                if old.iter().any(|h| h.side == side) {
                    continue;
                }
                let hand = if let Some(crop) = HandCrop::from_points(&points[base..base + 21]) {
                    model.infer_crop(&image, &crop, 0.65)?.map(|(mut h, _)| {
                        h.side = side.into();
                        h
                    })
                } else {
                    None
                };
                if let Some(h) = &hand {
                    let mut overlay = image.try_clone()?;
                    draw_hands_colored(
                        &mut overlay,
                        &[h.clone()],
                        Scalar::new(0.0, 255.0, 255.0, 0.0),
                    )?;
                    imgcodecs::imwrite(
                        directory
                            .join(format!("auxhand-{side}-{n:04}.jpg"))
                            .to_str()
                            .unwrap(),
                        &overlay,
                        &Vector::new(),
                    )?;
                }
                result.push(serde_json::json!({"frame":n,"side":side,"hand":hand}));
            }
        }
        std::fs::write(
            directory.join("auxhand-crops.json"),
            serde_json::to_vec(&result)?,
        )?;
        Ok(())
    }
    #[test]
    #[ignore = "requires local recording and hand model; no camera or windows"]
    fn evaluate_forearm_hand_proposals() -> Result<()> {
        use opencv::{imgcodecs, videoio};
        use std::collections::BTreeSet;
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let input: Vec<serde_json::Value> =
            std::fs::read_to_string(directory.join("verifiedarms.jsonl"))?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
        let mut frames: BTreeSet<usize> = (0..input.len()).step_by(15).collect();
        if std::env::var_os("LIFTING_EVAL_ALL").is_some() {
            frames.extend(0..input.len());
        }
        let keys: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(directory.join("keyframes.json"))?)?;
        frames.extend(
            keys.iter()
                .filter_map(|v| v["frame"].as_u64().map(|n| n as usize)),
        );
        let mut model = HandModel::new(Path::new("models/handpose.onnx"))?;
        let mut video =
            videoio::VideoCapture::from_file_def(directory.join("source.mp4").to_str().unwrap())?;
        let mut result = Vec::new();
        for n in frames {
            video.set(videoio::CAP_PROP_POS_FRAMES, n as f64)?;
            let mut image = Mat::default();
            ensure!(video.read(&mut image)?, "Missing frame");
            let p = &input[n]["people"][0];
            let k = &p["keypoints"];
            let get = |v: &serde_json::Value| Keypoint {
                name: "guide",
                x: v["x"].as_f64().unwrap() as f32,
                y: v["y"].as_f64().unwrap() as f32,
                score: v["score"].as_f64().unwrap() as f32,
            };
            let span = (get(&k[5]).x - get(&k[6]).x)
                .hypot(get(&k[5]).y - get(&k[6]).y)
                .max(p["bbox"]["width"].as_f64().unwrap() as f32 * 0.3);
            for (side, elbow, wrist) in [("left", 7, 0), ("right", 8, 1)] {
                for factor in [1.2, 1.6, 2.0] {
                    if let Some(crop) = HandCrop::from_forearm(
                        get(&k[elbow]),
                        get(&p["body_wrists"][wrist]),
                        span,
                        factor,
                    ) {
                        let hand = model.infer_crop(&image, &crop, 0.65)?.map(|(mut h, _)| {
                            h.side = side.into();
                            h
                        });
                        let mut palm_supported = false;
                        if let Some(h) = &hand {
                            if p["hands"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .any(|old| old["side"] == side)
                            {
                                result.push(serde_json::json!({"frame":n,"side":side,"factor":factor,"hand":hand}));
                                continue;
                            }
                            let mut aligned = Mat::default();
                            imgproc::warp_affine_def(
                                &image,
                                &mut aligned,
                                &Mat::from_slice_2d(&crop.affine())?,
                                Size::new(224, 224),
                            )?;
                            for palm in model.palms.infer(&aligned)? {
                                let wrist = crop.image(palm.landmarks[0][0], palm.landmarks[0][1]);
                                if (wrist[0] - h.image_landmarks[0][0])
                                    .hypot(wrist[1] - h.image_landmarks[0][1])
                                    <= (palm_span(h) * 0.5).max(15.0)
                                {
                                    palm_supported = true;
                                }
                            }
                            let mut overlay = image.try_clone()?;
                            draw_hands_colored(
                                &mut overlay,
                                &[h.clone()],
                                Scalar::new(0.0, 255.0, 255.0, 0.0),
                            )?;
                            imgcodecs::imwrite(
                                directory
                                    .join(format!("forearm-{side}-{factor}-{n:04}.jpg"))
                                    .to_str()
                                    .unwrap(),
                                &overlay,
                                &Vector::new(),
                            )?;
                        }
                        result.push(
                            serde_json::json!({"frame":n,"side":side,"factor":factor,"hand":hand,"palm_supported":palm_supported}),
                        );
                    }
                }
            }
        }
        std::fs::write(
            directory.join("forearm-proposals.json"),
            serde_json::to_vec(&result)?,
        )?;
        Ok(())
    }
    #[test]
    #[ignore = "requires local recording and palm/hand models; headless diagnostic"]
    fn evaluate_recorded_hand_regions() -> Result<()> {
        use opencv::{imgcodecs, videoio};
        use std::collections::BTreeSet;
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let input: Vec<serde_json::Value> = std::fs::read_to_string(directory.join("full.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        let mut frames: BTreeSet<usize> = (0..input.len()).step_by(15).collect();
        let keys: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(directory.join("keyframes.json"))?)?;
        frames.extend(
            keys.iter()
                .filter_map(|v| v["frame"].as_u64().map(|i| i as usize)),
        );
        let mut model = HandModel::new(Path::new("models/handpose.onnx"))?;
        let mut capture =
            videoio::VideoCapture::from_file_def(directory.join("source.mp4").to_str().unwrap())?;
        let mut results = Vec::new();
        for index in frames {
            let Some(person) = input[index]["people"].as_array().and_then(|p| p.first()) else {
                continue;
            };
            capture.set(videoio::CAP_PROP_POS_FRAMES, index as f64)?;
            let mut image = Mat::default();
            ensure!(capture.read(&mut image)?, "Cannot read frame {index}");
            let b = &person["bbox"];
            let (x, y, w, h) = (
                b["x"].as_f64().unwrap() as f32,
                b["y"].as_f64().unwrap() as f32,
                b["width"].as_f64().unwrap() as f32,
                b["height"].as_f64().unwrap() as f32,
            );
            let mut regions = vec![
                ("full", [0.0, 0.0, image.cols() as f32, image.rows() as f32]),
                ("person", [x - w * 0.15, y - h * 0.15, w * 1.3, h * 1.3]),
            ];
            let k = &person["keypoints"];
            let shoulder = ((k[5]["x"].as_f64().unwrap() - k[6]["x"].as_f64().unwrap())
                .hypot(k[5]["y"].as_f64().unwrap() - k[6]["y"].as_f64().unwrap())
                as f32)
                .max(w * 0.3);
            for (side, i) in [("left", 9), ("right", 10)] {
                if k[i]["score"].as_f64().unwrap_or(0.0) < 0.5 {
                    continue;
                }
                let size = (shoulder * 1.2).clamp(160.0, 480.0);
                regions.push((
                    side,
                    [
                        k[i]["x"].as_f64().unwrap() as f32 - size / 2.0,
                        k[i]["y"].as_f64().unwrap() as f32 - size / 2.0,
                        size,
                        size,
                    ],
                ));
            }
            for (tag, region) in regions {
                let palms = model.palms.infer_region(&image, region)?;
                let mut hands = Vec::new();
                for (palm_index, palm) in palms.iter().enumerate() {
                    if let Some(crop) = HandCrop::from_palm(palm) {
                        if index == 144 || index == 279 {
                            let mut warped = Mat::default();
                            imgproc::warp_affine_def(
                                &image,
                                &mut warped,
                                &Mat::from_slice_2d(&crop.affine())?,
                                Size::new(224, 224),
                            )?;
                            imgcodecs::imwrite(
                                directory
                                    .join(format!("hand-input-{tag}-{index:04}-{palm_index}.jpg"))
                                    .to_str()
                                    .unwrap(),
                                &warped,
                                &Vector::new(),
                            )?;
                        }
                        if let Some((mut hand, _)) = model.infer_crop(&image, &crop, 0.65)? {
                            hand.side = tag.into();
                            hands.push(hand);
                        }
                    }
                }
                let mut overlay = image.try_clone()?;
                draw_hands_colored(&mut overlay, &hands, Scalar::new(0.0, 255.0, 255.0, 0.0))?;
                imgcodecs::imwrite(
                    directory
                        .join(format!("hand-{tag}-{index:04}.jpg"))
                        .to_str()
                        .unwrap(),
                    &overlay,
                    &Vector::new(),
                )?;
                results.push(serde_json::json!({"frame":index,"region":tag,"palms":palms.len(),"hands":hands}));
            }
        }
        std::fs::write(
            directory.join("hand-regions.json"),
            serde_json::to_vec(&results)?,
        )?;
        Ok(())
    }
    #[test]
    fn duplicate_palm_crops_are_one_detection_even_with_opposite_side_labels() {
        let hand = HandPose {
            side: "left".into(),
            confidence: 0.95,
            image_landmarks: std::array::from_fn(|i| [100.0 + i as f32 * 2.0, 100.0 + i as f32]),
            world_landmarks: [[0.0; 3]; 21],
        };
        let mut duplicate = hand.clone();
        duplicate.side = "right".into();
        for p in &mut duplicate.image_landmarks {
            p[0] += 2.0;
            p[1] -= 1.0;
        }
        assert!(same_hand_detection(&hand, &duplicate));
        // Different crops can shift a fist by part of its palm width. Keep one
        // observation instead of attaching that same visible fist to both arms.
        let mut shifted = hand.clone();
        shifted.side = "right".into();
        let shift = palm_span(&hand) * 0.45;
        for p in &mut shifted.image_landmarks {
            p[0] += shift;
        }
        assert!(same_hand_detection(&hand, &shifted));
        for p in &mut duplicate.image_landmarks {
            p[0] += 100.0;
        }
        assert!(!same_hand_detection(&hand, &duplicate));
    }
    #[test]
    fn short_hand_predictions_decay_expire_and_preserve_finger_pose() {
        let make = |x: f32| HandPose {
            side: "left".into(),
            confidence: 0.95,
            image_landmarks: std::array::from_fn(|i| [x + i as f32, 100.0 + i as f32 * 0.5]),
            world_landmarks: [[0.0; 3]; 21],
        };
        let track = HandTrack {
            hand: make(110.0),
            previous: Some((make(100.0), 0.0)),
            time: 1.0 / 30.0,
            strong_time: 1.0 / 30.0,
        };
        let prediction = track.prediction(0.1).unwrap();
        assert!(prediction.image_landmarks[0][0] > 110.0);
        assert!(prediction.confidence < track.hand.confidence);
        for i in 1..21 {
            let a = prediction.image_landmarks[i];
            let b = prediction.image_landmarks[HAND_PARENT[i]];
            let c = track.hand.image_landmarks[i];
            let d = track.hand.image_landmarks[HAND_PARENT[i]];
            assert!(
                ((a[0] - b[0]).hypot(a[1] - b[1]) - (c[0] - d[0]).hypot(c[1] - d[1])).abs() < 1e-4
            );
        }
        assert!(track.prediction(0.34).is_none());
        assert!(track.prediction(-1.0).is_none());
        let faded = track.prediction(0.32).unwrap();
        assert!(faded.confidence < 0.2);
        assert!(
            HandCrop::from_hand(&faded).is_some(),
            "Faded display confidence must not disable recovery crops"
        );
    }
    #[test]
    fn dedicated_hands_replace_coarse_guesses_and_connect_to_body_wrists() {
        let mut points = vec![
            Keypoint {
                name: "test",
                x: 10.0,
                y: 10.0,
                score: 0.9
            };
            133
        ];
        let hand = HandPose {
            side: "left".into(),
            confidence: 0.95,
            image_landmarks: std::array::from_fn(|i| [100.0 + i as f32, 200.0 - i as f32]),
            world_landmarks: [[0.0; 3]; 21],
        };
        merge_hand_landmarks(&mut points, &[hand]);
        assert_eq!((points[9].x, points[9].y), (100.0, 200.0));
        assert_eq!((points[111].x, points[111].y), (120.0, 180.0));
        assert!(points[112..].iter().all(|p| p.score == 0.0));
        merge_hand_landmarks(&mut points, &[]);
        assert!(points[91..].iter().all(|p| p.score == 0.0));
    }
    #[test]
    fn consistent_fresh_hand_can_replace_a_confident_body_wrist_without_inflating_score() {
        let mut points = vec![
            Keypoint {
                name: "test",
                x: 100.0,
                y: 200.0,
                score: 0.99
            };
            133
        ];
        let mut hand = HandPose {
            side: "left".into(),
            confidence: 0.7,
            image_landmarks: [[140.0, 210.0]; 21],
            world_landmarks: [[0.0; 3]; 21],
        };
        merge_supported_hand_landmarks(&mut points, &[hand.clone()], [false; 2]);
        assert_eq!((points[9].x, points[9].score), (100.0, 0.99));
        merge_supported_hand_landmarks(&mut points, &[hand.clone()], [false, true]);
        assert_eq!((points[9].x, points[9].score), (100.0, 0.99));
        hand.confidence = 0.64;
        merge_supported_hand_landmarks(&mut points, &[hand.clone()], [true, false]);
        assert_eq!((points[9].x, points[9].score), (100.0, 0.99));
        hand.confidence = 0.7;
        merge_supported_hand_landmarks(&mut points, &[hand], [true, false]);
        assert_eq!(
            (points[9].x, points[9].y, points[9].score),
            (140.0, 210.0, 0.7)
        );
        assert_eq!((points[10].x, points[10].score), (100.0, 0.99));
        let before = points[9];
        merge_supported_hand_landmarks(&mut points, &[], [true; 2]);
        assert_eq!((points[9].x, points[9].score), (before.x, before.score));
        assert!(points[91..].iter().all(|p| p.score == 0.0));
    }
    #[test]
    fn marginal_hand_keeps_a_confident_body_wrist() {
        let mut points = vec![
            Keypoint {
                name: "test",
                x: 100.0,
                y: 200.0,
                score: 0.95
            };
            133
        ];
        let mut hand = HandPose {
            side: "left".into(),
            confidence: 0.7,
            image_landmarks: [[140.0, 210.0]; 21],
            world_landmarks: [[0.0; 3]; 21],
        };
        merge_hand_landmarks(&mut points, &[hand.clone()]);
        assert_eq!(
            (points[9].x, points[9].y, points[9].score),
            (100.0, 200.0, 0.95)
        );
        assert_eq!(points[91].score, 0.7);
        points[9].score = 0.3;
        merge_hand_landmarks(&mut points, &[hand.clone()]);
        assert_eq!(points[9].x, 140.0);
        points[9].x = 100.0;
        points[9].score = 0.95;
        hand.confidence = 0.99;
        merge_hand_landmarks(&mut points, &[hand]);
        assert_eq!(points[9].x, 140.0);
    }
    #[test]
    fn palm_crop_keeps_a_rotated_hand_centered_and_roundtrips() {
        // Palm constructors and landmark constructors must use the same image
        // space. A roundtrip alone can hide a consistently misplaced crop.
        for angle in [0.0_f32, 0.7, 1.57, -2.1] {
            let (s, c) = angle.sin_cos();
            let center = [650.0, 350.0];
            let landmarks = std::array::from_fn(|i| {
                let local = [
                    [0.0, 30.0],
                    [-25.0, 0.0],
                    [0.0, -15.0],
                    [15.0, 0.0],
                    [30.0, 5.0],
                    [-20.0, 20.0],
                    [-30.0, 10.0],
                ][i];
                [
                    center[0] + c * local[0] - s * local[1],
                    center[1] + s * local[0] + c * local[1],
                ]
            });
            let palm = crate::palm::Palm::from_test_landmarks(landmarks);
            let crop = HandCrop::from_palm(&palm).unwrap();
            let affine = crop.affine();
            let project = |p: [f32; 2]| {
                [
                    (affine[0][0] * p[0] as f64 + affine[0][1] * p[1] as f64 + affine[0][2]) as f32,
                    (affine[1][0] * p[0] as f64 + affine[1][1] * p[1] as f64 + affine[1][2]) as f32,
                ]
            };
            let mapped = project(crop.center);
            assert!((mapped[0] - 112.0).abs() < 0.001 && (mapped[1] - 112.0).abs() < 0.001);
            for p in landmarks {
                let mapped = project(p);
                assert!(
                    mapped.iter().all(|v| (0.0..224.0).contains(v)),
                    "Palm outside crop: {mapped:?}"
                );
                let back = crop.image(mapped[0], mapped[1]);
                assert!((back[0] - p[0]).abs() < 0.001 && (back[1] - p[1]).abs() < 0.001);
            }
        }
    }
    #[test]
    fn oriented_hand_crop_roundtrips_image_coordinates() {
        let p: Vec<_> = (0..21)
            .map(|i| Keypoint {
                name: "test",
                x: 100.0 + (i % 5) as f32 * 5.0,
                y: 200.0 - i as f32 * 4.0,
                score: 0.9,
            })
            .collect();
        let crop = HandCrop::from_points(&p).unwrap();
        let affine = crop.affine();
        for p in p {
            let x = (affine[0][0] * p.x as f64 + affine[0][1] * p.y as f64 + affine[0][2]) as f32;
            let y = (affine[1][0] * p.x as f64 + affine[1][1] * p.y as f64 + affine[1][2]) as f32;
            let reconstructed = crop.image(x, y);
            assert!((reconstructed[0] - p.x).abs() < 1e-4 && (reconstructed[1] - p.y).abs() < 1e-4);
        }
    }
    #[test]
    fn sparse_face_fit_tracks_a_known_head_turn_and_rejects_missing_face() {
        let canonical = [
            Vector3::new(0.0, 0.0, -0.5),
            Vector3::new(0.35, -0.25, -0.2),
            Vector3::new(-0.35, -0.25, -0.2),
            Vector3::new(0.65, 0.0, 0.0),
            Vector3::new(-0.65, 0.0, 0.0),
        ];
        let q = UnitQuaternion::from_euler_angles(0.15, 0.4, 0.1);
        let mut p: Vec<_> = canonical
            .iter()
            .map(|p| {
                let p = q * p;
                Keypoint {
                    name: "face",
                    x: (320.0 + 80.0 * p.x) as f32,
                    y: (200.0 + 80.0 * p.y) as f32,
                    score: 0.9,
                }
            })
            .collect();
        let fitted = head_pose(&p, 0.3).unwrap();
        let a = fitted.orientation_xyzw;
        let fitted =
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(a[3], a[0], a[1], a[2]));
        assert!(fitted.angle_to(&q) < 0.3);
        p[1].score = 0.0;
        p[2].score = 0.0;
        assert!(head_pose(&p, 0.3).is_none());
    }
}
