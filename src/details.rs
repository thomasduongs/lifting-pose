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
        points[wrist].x = hand.image_landmarks[0][0];
        points[wrist].y = hand.image_landmarks[0][1];
        points[wrist].score = hand.confidence;
        for (i, pixel) in hand.image_landmarks.iter().enumerate() {
            points[base + i].x = pixel[0];
            points[base + i].y = pixel[1];
            points[base + i].score = hand.confidence;
        }
    }
}

pub struct HandModel {
    net: dnn::Net,
    names: Vector<String>,
    palms: crate::palm::PalmDetector,
    previous: BTreeMap<String, (HandPose, u8)>,
}
impl HandModel {
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
            palms,
            previous: BTreeMap::new(),
        })
    }
    pub fn infer(&mut self, image: &Mat, points: &[Keypoint]) -> Result<Vec<HandPose>> {
        if points.len() < 133 {
            return Ok(Vec::new());
        }
        for (_, age) in self.previous.values_mut() {
            *age = age.saturating_add(1);
        }
        self.previous.retain(|_, (_, age)| *age <= 3);
        let palms = self.palms.infer(image)?;
        let mut candidates = Vec::new();
        for palm in &palms {
            if let Some(crop) = HandCrop::from_palm(palm) {
                if let Some(hand) = self.infer_crop(image, &crop)? {
                    candidates.push(hand);
                }
            }
        }
        // Reacquire even when coarse body fingers vanish. Previous hand bounds
        // are crop proposals only; every accepted hand gets fresh inference.
        for (side, base) in [("left", 91), ("right", 112)] {
            let previous = self.previous.get(side).map(|(hand, _)| hand.clone());
            let crop = previous
                .as_ref()
                .and_then(|hand| HandCrop::from_hand(hand))
                .or_else(|| HandCrop::from_points(&points[base..base + 21]))
                .or_else(|| HandCrop::from_sparse_points(&points[base..base + 21]));
            if let Some(crop) = crop {
                let already = candidates.iter().any(|(h, _)| {
                    (h.image_landmarks[0][0] - crop.center[0])
                        .hypot(h.image_landmarks[0][1] - crop.center[1])
                        < crop.side * 0.65
                });
                if !already {
                    if let Some(hand) = self.infer_crop(image, &crop)? {
                        let duplicate = candidates.iter().any(|(h, _)| {
                            (h.image_landmarks[0][0] - hand.0.image_landmarks[0][0])
                                .hypot(h.image_landmarks[0][1] - hand.0.image_landmarks[0][1])
                                < crop.side * 0.35
                        });
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
            let wrist = hand.image_landmarks[0];
            let guide = &points[if side == "left" { 9 } else { 10 }];
            let body = if guide.score >= 0.2 && guide.x >= 0.0 {
                (wrist[0] - guide.x).hypot(wrist[1] - guide.y) / scale
            } else {
                (wrist[0] - width * if side == "left" { 0.7 } else { 0.3 }).abs() / scale
            };
            let distance = if let Some((old, _)) = self.previous.get(side) {
                0.6 * body
                    + 0.4
                        * (wrist[0] - old.image_landmarks[0][0])
                            .hypot(wrist[1] - old.image_landmarks[0][1])
                        / scale
            } else {
                body
            };
            distance
                + 0.1 * (1.0 - hand.confidence)
                + 0.02
                    * if side == "left" {
                        *handedness
                    } else {
                        1.0 - *handedness
                    }
        };
        let mut best = (None, None, 4.0);
        for left in 0..=candidates.len() {
            for right in 0..=candidates.len() {
                if left == right && left < candidates.len() {
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
                self.previous.insert(side.into(), (hand.clone(), 0));
                hands.push(hand);
            }
        }
        Ok(hands)
    }
    fn infer_crop(&mut self, image: &Mat, crop: &HandCrop) -> Result<Option<(HandPose, f32)>> {
        let affine = Mat::from_slice_2d(&crop.affine())?;
        let mut warped = Mat::default();
        imgproc::warp_affine_def(image, &mut warped, &affine, Size::new(224, 224))?;
        let mut rgb = Mat::default();
        imgproc::cvt_color_def(&warped, &mut rgb, imgproc::COLOR_BGR2RGB)?;
        let mut float = Mat::default();
        rgb.convert_to(&mut float, core::CV_32F, 1.0 / 255.0, 0.0)?;
        let blob = float.reshape_nd(1, &[1, 224, 224, 3])?;
        self.net.set_input_def(&blob)?;
        let mut outputs = Vector::<Mat>::new();
        self.net.forward(&mut outputs, &self.names)?;
        let output = |name: &str| -> Result<Mat> {
            let index = self
                .names
                .iter()
                .position(|n| n == name)
                .with_context(|| format!("Missing hand output {name}"))?;
            Ok(outputs.get(index)?)
        };
        let confidence = output("Identity_1")?.data_typed::<f32>()?[0];
        if !confidence.is_finite() || confidence < 0.65 {
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

struct HandCrop {
    ex: [f32; 2],
    ey: [f32; 2],
    center: [f32; 2],
    side: f32,
}
impl HandCrop {
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
                score: hand.confidence,
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
            center: [(minx + maxx) / 2.0, (miny + maxy) / 2.0],
            side,
        })
    }
    fn affine(&self) -> [[f64; 3]; 2] {
        let s = 224.0 / self.side as f64;
        [
            [
                s * self.ex[0] as f64,
                s * self.ex[1] as f64,
                112.0 - s * self.center[0] as f64,
            ],
            [
                s * self.ey[0] as f64,
                s * self.ey[1] as f64,
                112.0 - s * self.center[1] as f64,
            ],
        ]
    }
    fn image(&self, x: f32, y: f32) -> [f32; 2] {
        let u = (x - 112.0) * self.side / 224.0 + self.center[0];
        let v = (y - 112.0) * self.side / 224.0 + self.center[1];
        [
            u * self.ex[0] + v * self.ey[0],
            u * self.ex[1] + v * self.ey[1],
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
    for hand in hands {
        let color = Scalar::new(220.0, 90.0, 230.0, 0.0);
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
