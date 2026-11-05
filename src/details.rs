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
use std::path::Path;

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
#[derive(Serialize, Deserialize)]
pub struct HandShape {
    pub bone_lengths: Vec<f64>,
    pub measurement_count: u64,
}

pub struct HandModel {
    net: dnn::Net,
    names: Vector<String>,
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
        Ok(Self { net, names })
    }
    pub fn infer(&mut self, image: &Mat, points: &[Keypoint]) -> Result<Vec<HandPose>> {
        if points.len() < 133 {
            return Ok(Vec::new());
        }
        let mut hands = Vec::new();
        for (side, base) in [("left", 91), ("right", 112)] {
            let Some(crop) = HandCrop::from_points(&points[base..base + 21]) else {
                continue;
            };
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
                continue;
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
                continue;
            }
            let image_landmarks =
                std::array::from_fn(|i| crop.image(screen[i * 3], screen[i * 3 + 1]));
            let world_landmarks = std::array::from_fn(|i| {
                [
                    world[i * 3] * crop.ex[0] + world[i * 3 + 1] * crop.ey[0],
                    world[i * 3] * crop.ex[1] + world[i * 3 + 1] * crop.ey[1],
                    world[i * 3 + 2],
                ]
            });
            hands.push(HandPose {
                side: side.into(),
                confidence,
                image_landmarks,
                world_landmarks,
            });
        }
        Ok(hands)
    }
}
struct HandCrop {
    ex: [f32; 2],
    ey: [f32; 2],
    center: [f32; 2],
    side: f32,
}
impl HandCrop {
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
