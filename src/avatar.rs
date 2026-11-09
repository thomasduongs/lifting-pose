//! Procedural baseline human surface driven by the replay's existing FK joints.
//! Circumferences are visual defaults, not measurements inferred from the video.
use anyhow::Result;
use nalgebra::{Rotation3, UnitQuaternion, Vector3};
use opencv::{
    core::{Mat, Vec3b},
    prelude::*,
};

type V = Vector3<f64>;
const SKIN: [f64; 3] = [190.0, 177.0, 155.0]; // BGR, neutral clay
const INFERRED: [f64; 3] = [145.0, 139.0, 131.0];

pub struct Body<'a> {
    pub joints: &'a [V; 17],
    pub shoulder_width: f64,
    pub pelvis_width: f64,
    pub orientation: UnitQuaternion<f64>,
    pub head_center: V,
    pub head_orientation: UnitQuaternion<f64>,
    pub head_radius: f64,
    pub observed: [bool; 17],
    pub hands: &'a [(bool, [V; 21])], // true = left
}
#[derive(Clone, Copy)]
struct Vertex {
    p: V,
    n: V,
}
struct Triangle {
    v: [Vertex; 3],
    color: [f64; 3],
}
#[derive(Default)]
pub struct Mesh {
    triangles: Vec<Triangle>,
}

fn basis(axis: V) -> (V, V, V) {
    let y = axis.try_normalize(1e-9).unwrap_or_else(V::y);
    let guide = if y.x.abs() < 0.8 { V::x() } else { V::z() };
    let z = guide.cross(&y).normalize();
    (y.cross(&z).normalize(), y, z)
}
impl Mesh {
    fn ellipsoid(&mut self, center: V, axes: [V; 3], radii: V, color: [f64; 3]) {
        let vertex = |latitude: usize, longitude: usize| {
            let a = std::f64::consts::PI * latitude as f64 / 8.0;
            let b = std::f64::consts::TAU * longitude as f64 / 12.0;
            let unit = V::new(a.sin() * b.cos(), a.cos(), a.sin() * b.sin());
            let local = unit.component_mul(&radii);
            Vertex {
                p: center + axes[0] * local.x + axes[1] * local.y + axes[2] * local.z,
                n: (axes[0] * (unit.x / radii.x)
                    + axes[1] * (unit.y / radii.y)
                    + axes[2] * (unit.z / radii.z))
                    .normalize(),
            }
        };
        for j in 0..8 {
            for i in 0..12 {
                let a = vertex(j, i);
                let b = vertex(j, i + 1);
                let c = vertex(j + 1, i);
                let d = vertex(j + 1, i + 1);
                self.triangles.push(Triangle {
                    v: [a, c, b],
                    color,
                });
                self.triangles.push(Triangle {
                    v: [b, c, d],
                    color,
                });
            }
        }
    }
    fn ball(&mut self, p: V, radius: f64, color: [f64; 3]) {
        self.ellipsoid(p, [V::x(), V::y(), V::z()], V::repeat(radius), color);
    }
    fn limb(&mut self, a: V, b: V, ra: f64, rb: f64, color: [f64; 3]) {
        let delta = b - a;
        let length = delta.norm();
        if length < 1e-8 {
            return;
        }
        let (x, y, z) = basis(delta);
        let slope = (ra - rb) / length;
        let ring = |end: bool, i: usize| {
            let angle = std::f64::consts::TAU * i as f64 / 12.0;
            let radial = x * angle.cos() + z * angle.sin();
            Vertex {
                p: if end {
                    b + radial * rb
                } else {
                    a + radial * ra
                },
                n: (radial + y * slope).normalize(),
            }
        };
        for i in 0..12 {
            let a = ring(false, i);
            let b = ring(false, i + 1);
            let c = ring(true, i);
            let d = ring(true, i + 1);
            self.triangles.push(Triangle {
                v: [a, c, b],
                color,
            });
            self.triangles.push(Triangle {
                v: [b, c, d],
                color,
            });
        }
        self.ball(a, ra, color);
        self.ball(b, rb, color);
    }
    pub fn human(body: &Body<'_>) -> Self {
        let mut mesh = Self::default();
        let p = body.joints;
        let s = body.shoulder_width.max(0.02);
        let color = |indices: &[usize]| {
            if indices.iter().all(|&i| body.observed[i]) {
                SKIN
            } else {
                INFERRED
            }
        };
        let down = (p[0] - p[8]).try_normalize(1e-8).unwrap_or_else(V::y);
        let right = (p[11] - p[14]).try_normalize(1e-8).unwrap_or_else(V::x);
        let front = right.cross(&down).try_normalize(1e-8).unwrap_or_else(V::z);
        let right = down.cross(&front).normalize();
        // Elliptical torso rings share one stable cross-section profile. Centers
        // follow the spine; the shoulders/hips determine the transverse frame.
        let rings = [
            (0.0, (body.pelvis_width * 0.64).max(s * 0.37), s * 0.28),
            (0.2, s * 0.38, s * 0.26),
            (0.48, s * 0.40, s * 0.26),
            (0.75, s * 0.47, s * 0.29),
            (0.96, s * 0.48, s * 0.26),
            (1.06, s * 0.16, s * 0.14),
        ];
        let torso_vertex = |r: usize, i: usize| {
            let (t, width, depth) = rings[r];
            let a = std::f64::consts::TAU * i as f64 / 20.0;
            let center = p[0] * (1.0 - t) + p[8] * t;
            Vertex {
                p: center + right * (width * a.cos()) + front * (depth * a.sin()),
                n: (right * (a.cos() / width) + front * (a.sin() / depth)).normalize(),
            }
        };
        for r in 0..rings.len() - 1 {
            for i in 0..20 {
                let a = torso_vertex(r, i);
                let b = torso_vertex(r, i + 1);
                let c = torso_vertex(r + 1, i);
                let d = torso_vertex(r + 1, i + 1);
                // Torso volume is a baseline even where the spine/hips are unseen.
                mesh.triangles.push(Triangle {
                    v: [a, c, b],
                    color: SKIN,
                });
                mesh.triangles.push(Triangle {
                    v: [b, c, d],
                    color: SKIN,
                });
            }
        }
        for r in [0, rings.len() - 1] {
            let center = Vertex {
                p: p[0] * (1.0 - rings[r].0) + p[8] * rings[r].0,
                n: if r == 0 { down } else { -down },
            };
            for i in 0..20 {
                mesh.triangles.push(Triangle {
                    v: [center, torso_vertex(r, i), torso_vertex(r, i + 1)],
                    color: SKIN,
                });
            }
        }
        mesh.ellipsoid(
            p[0],
            [right, down, front],
            V::new(rings[0].1, s * 0.22, s * 0.28),
            INFERRED,
        );
        for (shoulder, elbow, wrist) in [(11, 12, 13), (14, 15, 16)] {
            mesh.limb(
                p[shoulder],
                p[elbow],
                s * 0.125,
                s * 0.085,
                color(&[shoulder, elbow]),
            );
            mesh.limb(
                p[elbow],
                p[wrist],
                s * 0.09,
                s * 0.055,
                color(&[elbow, wrist]),
            );
            let left = shoulder == 11;
            if let Some((_, hand)) = body.hands.iter().find(|(side, _)| *side == left) {
                mesh.hand(hand, s, color(&[wrist]));
            } else {
                let hand = neutral_hand(p[wrist], p[wrist] - p[elbow], right, s, left);
                mesh.hand(&hand, s, INFERRED);
            }
        }
        let foot_axes = [body.orientation * V::x(), V::y(), body.orientation * V::z()];
        for (hip, knee, ankle) in [(1, 2, 3), (4, 5, 6)] {
            mesh.limb(p[hip], p[knee], s * 0.175, s * 0.115, color(&[hip, knee]));
            mesh.limb(
                p[knee],
                p[ankle],
                s * 0.115,
                s * 0.07,
                color(&[knee, ankle]),
            );
            let center = p[ankle] - V::y() * s * 0.075 - foot_axes[2] * s * 0.13;
            mesh.ellipsoid(
                center,
                foot_axes,
                V::new(s * 0.12, s * 0.075, s * 0.30),
                color(&[ankle]),
            );
        }
        let q = body.head_orientation;
        let axes = [q * V::x(), q * V::y(), q * V::z()];
        let r = body.head_radius.max(s * 0.12);
        mesh.limb(
            p[8] - down * s * 0.02,
            body.head_center + axes[1] * r * 0.65,
            s * 0.105,
            s * 0.09,
            SKIN,
        );
        let head_color = color(&[9, 10]);
        mesh.ellipsoid(
            body.head_center,
            axes,
            V::new(r * 0.80, r * 1.10, r * 0.85),
            head_color,
        );
        mesh.ellipsoid(
            body.head_center + q * V::new(0.0, r * 0.54, -r * 0.1),
            axes,
            V::new(r * 0.58, r * 0.65, r * 0.66),
            head_color,
        );
        mesh.ellipsoid(
            body.head_center + q * V::new(0.0, r * 0.10, -r * 0.85),
            axes,
            V::new(r * 0.13, r * 0.21, r * 0.20),
            head_color,
        );
        for side in [-1.0, 1.0] {
            mesh.ellipsoid(
                body.head_center + q * V::new(side * r * 0.79, r * 0.03, 0.0),
                axes,
                V::new(r * 0.12, r * 0.25, r * 0.15),
                head_color,
            );
            mesh.ellipsoid(
                body.head_center + q * V::new(side * r * 0.28, -r * 0.16, -r * 0.78),
                axes,
                V::new(r * 0.075, r * 0.04, r * 0.04),
                [48.0, 47.0, 45.0],
            );
        }
        mesh
    }
    fn hand(&mut self, p: &[V; 21], s: f64, color: [f64; 3]) {
        let middle = (p[5] + p[9] + p[13] + p[17]) / 4.0;
        let (fallback, y, _) = basis(middle - p[0]);
        let x = (p[5] - p[17] - y * (p[5] - p[17]).dot(&y))
            .try_normalize(1e-8)
            .unwrap_or(fallback);
        let z = x.cross(&y).normalize();
        self.ellipsoid(
            (middle + p[0]) / 2.0,
            [x, y, z],
            V::new(
                (p[5] - p[17]).norm() * 0.61,
                (middle - p[0]).norm() * 0.70,
                s * 0.032,
            )
            .map(|v| v.max(s * 0.015)),
            color,
        );
        for i in 1..21 {
            let joint = (i - 1) % 4;
            let radius = s * (0.023 - joint as f64 * 0.003);
            self.limb(p[HAND_PARENT[i]], p[i], radius, radius * 0.85, color);
        }
    }
    /// Smooth normal shading and a depth buffer give real surface occlusion,
    /// including crossing arms and fingers. Camera-space +Y points downward.
    pub fn draw(
        &self,
        image: &mut Mat,
        orbit: &Rotation3<f64>,
        center: V,
        distance: f64,
    ) -> Result<()> {
        let width = image.cols() as usize;
        let height = image.rows() as usize;
        let pixels = image.data_typed_mut::<Vec3b>()?;
        let mut depth = vec![f64::INFINITY; width * height];
        let light = V::new(-0.4, -0.65, -1.0).normalize();
        for t in &self.triangles {
            let v = t.v.map(|v| {
                let p = orbit * (v.p - center) + V::new(0.0, 0.0, distance);
                let brightness = 0.30 + 0.70 * (orbit * v.n).dot(&light).max(0.0);
                (
                    V::new(512.0 + 620.0 * p.x / p.z, 370.0 + 620.0 * p.y / p.z, p.z),
                    brightness,
                )
            });
            if v.iter()
                .any(|(p, _)| p.z <= 0.001 || !p.iter().all(|x| x.is_finite()))
            {
                continue;
            }
            let edge =
                |a: V, b: V, x: f64, y: f64| (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
            let area = edge(v[0].0, v[1].0, v[2].0.x, v[2].0.y);
            if area.abs() < 1e-8 {
                continue;
            }
            let minx = v
                .iter()
                .map(|(p, _)| p.x)
                .fold(f64::INFINITY, f64::min)
                .floor()
                .max(0.0) as usize;
            let maxx = v
                .iter()
                .map(|(p, _)| p.x)
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil()
                .min((width - 1) as f64)
                .max(0.0) as usize;
            let miny = v
                .iter()
                .map(|(p, _)| p.y)
                .fold(f64::INFINITY, f64::min)
                .floor()
                .max(0.0) as usize;
            let maxy = v
                .iter()
                .map(|(p, _)| p.y)
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil()
                .min((height - 1) as f64)
                .max(0.0) as usize;
            for y in miny..=maxy {
                for x in minx..=maxx {
                    let a = edge(v[1].0, v[2].0, x as f64 + 0.5, y as f64 + 0.5) / area;
                    let b = edge(v[2].0, v[0].0, x as f64 + 0.5, y as f64 + 0.5) / area;
                    let c = 1.0 - a - b;
                    if a < -1e-9 || b < -1e-9 || c < -1e-9 {
                        continue;
                    }
                    let iz = a / v[0].0.z + b / v[1].0.z + c / v[2].0.z;
                    let z = 1.0 / iz;
                    let index = y * width + x;
                    if z >= depth[index] {
                        continue;
                    }
                    depth[index] = z;
                    let brightness =
                        (a * v[0].1 / v[0].0.z + b * v[1].1 / v[1].0.z + c * v[2].1 / v[2].0.z)
                            / iz;
                    pixels[index] = Vec3b::from(
                        t.color
                            .map(|channel| (channel * brightness).clamp(0.0, 255.0) as u8),
                    );
                }
            }
        }
        Ok(())
    }
}
use crate::details::HAND_PARENT;
fn neutral_hand(wrist: V, direction: V, width_axis: V, s: f64, left: bool) -> [V; 21] {
    let (_, y, _) = basis(direction);
    let x = (width_axis - y * width_axis.dot(&y))
        .try_normalize(1e-8)
        .unwrap_or_else(|| basis(y).0);
    let z = x.cross(&y).normalize();
    let mut p = [wrist; 21];
    for finger in 0..5 {
        let base = 1 + finger * 4;
        let lateral = if finger == 0 {
            0.12
        } else {
            0.095 - (finger - 1) as f64 * 0.063
        };
        let lateral = if left { lateral } else { -lateral };
        let lengths = if finger == 0 {
            [0.10, 0.075, 0.065, 0.055]
        } else {
            [0.23, 0.10, 0.065, 0.05]
        };
        for j in 0..4 {
            let offset = if j == 0 { x * lateral * s } else { V::zeros() };
            p[base + j] =
                p[HAND_PARENT[base + j]] + offset + y * lengths[j] * s + z * (j as f64 * 0.008 * s);
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn depth_buffer_keeps_the_front_surface_in_either_triangle_order() -> Result<()> {
        use opencv::core::{self, Scalar};
        let triangle = |z: f64, color| Triangle {
            v: [[-0.1, -0.1], [0.1, -0.1], [0.0, 0.1]].map(|p| Vertex {
                p: V::new(p[0] * z, p[1] * z, z),
                n: -V::z(),
            }),
            color,
        };
        let mut samples = Vec::new();
        for reverse in [false, true] {
            let mut triangles = vec![
                triangle(2.0, [200.0, 0.0, 0.0]),
                triangle(3.0, [0.0, 0.0, 200.0]),
            ];
            if reverse {
                triangles.reverse();
            }
            let mut image =
                Mat::new_rows_cols_with_default(768, 1024, core::CV_8UC3, Scalar::all(0.0))?;
            Mesh { triangles }.draw(&mut image, &Rotation3::identity(), V::zeros(), 0.0)?;
            samples.push(*image.at_2d::<Vec3b>(370, 512)?);
        }
        assert_eq!(samples[0], samples[1]);
        assert!(samples[0][0] > 100 && samples[0][2] == 0);
        Ok(())
    }
    #[test]
    fn generated_surface_is_finite_and_moves_with_its_rig() {
        let mut joints = [V::zeros(); 17];
        for (i, p) in [
            (0, [0., 0., 0.]),
            (8, [0., -0.5, 0.]),
            (11, [0.2, -0.5, 0.]),
            (14, [-0.2, -0.5, 0.]),
            (12, [0.5, -0.5, 0.]),
            (13, [0.75, -0.5, 0.]),
            (15, [-0.5, -0.5, 0.]),
            (16, [-0.75, -0.5, 0.]),
            (1, [-0.1, 0., 0.]),
            (2, [-0.1, 0.4, 0.]),
            (3, [-0.1, 0.8, 0.]),
            (4, [0.1, 0., 0.]),
            (5, [0.1, 0.4, 0.]),
            (6, [0.1, 0.8, 0.]),
        ] {
            joints[i] = V::from(p);
        }
        let make = |joints, head_center| {
            Mesh::human(&Body {
                joints,
                shoulder_width: 0.4,
                pelvis_width: 0.2,
                orientation: UnitQuaternion::identity(),
                head_center,
                head_orientation: UnitQuaternion::identity(),
                head_radius: 0.09,
                observed: [true; 17],
                hands: &[],
            })
        };
        let a = make(&joints, V::new(0., -0.75, 0.));
        let shift = V::new(2., 3., 4.);
        let moved = joints.map(|p| p + shift);
        let b = make(&moved, V::new(0., -0.75, 0.) + shift);
        assert!(a.triangles.len() > 1000);
        assert_eq!(a.triangles.len(), b.triangles.len());
        for (a, b) in a.triangles.iter().zip(b.triangles.iter()) {
            for (a, b) in a.v.iter().zip(b.v.iter()) {
                assert!(a.p.iter().chain(a.n.iter()).all(|v| v.is_finite()));
                assert!((a.n.norm() - 1.).abs() < 1e-9);
                assert!((b.p - a.p - shift).norm() < 1e-9);
            }
        }
    }
}
