//! Spatial tendon traversal and sphere/cylinder wrapping.
//!
//! Rust port of MuJoCo 3.17.0 `mju_wrap`, `wrap_circle`, `wrap_inside`, `is_intersect`,
//! `length_circle` (`src/engine/engine_util_misc.c`) and the spatial branch of `mj_tendon`
//! (`src/engine/engine_core_smooth.c`). Copyright 2021 DeepMind Technologies Limited,
//! licensed under the Apache License, Version 2.0. Modified: translated to Rust, restricted to
//! site/sphere/cylinder paths (no pulleys, no fixed tendons), Jacobians not computed here.

use crate::kinematics::{Kinematics, Mat3, Vec3, add3, dist3, dot3, mat_t_vec, mat_vec, norm3, normalize3, scl3, sub3, cross3};
use crate::model::Model;
use crate::{MJ_MINVAL, WRAP_CYLINDER, WRAP_SITE, WRAP_SPHERE};
use std::f64::consts::PI;

fn dot2(a: &[f64], b: &[f64]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn norm2(a: &[f64]) -> f64 {
    dot2(a, a).sqrt()
}
/// `mju_normalize` for n=2.
fn normalize2(a: &mut [f64; 2]) -> f64 {
    let n = norm2(a);
    if n < MJ_MINVAL {
        *a = [1.0, 0.0];
    } else {
        a[0] /= n;
        a[1] /= n;
    }
    n
}

/// Intersection of 2D segments p1-p2 and p3-p4.
fn is_intersect(p1: &[f64], p2: &[f64], p3: &[f64], p4: &[f64]) -> bool {
    let u = [p2[0] - p1[0], p2[1] - p1[1]];
    let v = [p4[0] - p3[0], p4[1] - p3[1]];
    let det = v[1] * u[0] - v[0] * u[1];
    if det.abs() < MJ_MINVAL {
        return false;
    }
    let s = if det > 0.0 { 1.0 } else { -1.0 };
    let w = [p1[0] - p3[0], p1[1] - p3[1]];
    let d = [p2[0] - p4[0], p2[1] - p4[1]];
    let a0 = (v[0] * w[1] - v[1] * w[0]) * s;
    let a1 = (v[0] * d[1] - v[1] * d[0]) * s;
    let b0 = (u[0] * w[1] - u[1] * w[0]) * s;
    let b1 = (u[0] * d[1] - u[1] * d[0]) * s;
    a0 >= 0.0 && a1 <= 0.0 && b0 >= 0.0 && b1 <= 0.0
}

fn length_circle(p0: &[f64], p1: &[f64], ind: bool, radius: f64) -> f64 {
    let cross = p0[1] * p1[0] - p0[0] * p1[1];
    let mut angle = cross.abs().atan2(dot2(p0, p1));
    if (cross > 0.0 && ind) || (cross < 0.0 && !ind) {
        angle = 2.0 * PI - angle;
    }
    radius * angle
}

/// 2D circle wrap; returns wrap arc length or -1 (no wrap), tangent points in `pnt`.
fn wrap_circle(pnt: &mut [f64; 4], end: &[f64; 4], side: Option<&[f64; 2]>, radius: f64) -> f64 {
    let sqlen0 = end[0] * end[0] + end[1] * end[1];
    let sqlen1 = end[2] * end[2] + end[3] * end[3];
    let sqrad = radius * radius;
    if sqlen0 < sqrad || sqlen1 < sqrad || radius < MJ_MINVAL {
        return -1.0;
    }
    let dif = [end[2] - end[0], end[3] - end[1]];
    let dd = dif[0] * dif[0] + dif[1] * dif[1];
    if dd < MJ_MINVAL {
        return -1.0;
    }
    let a = (-(dif[0] * end[0] + dif[1] * end[1]) / dd).clamp(0.0, 1.0);
    let tmp = [a * dif[0] + end[0], a * dif[1] + end[1]];
    if tmp[0] * tmp[0] + tmp[1] * tmp[1] > sqrad && side.is_none_or(|s| dot2(s, &tmp) >= 0.0) {
        return -1.0;
    }
    let sqrt0 = (sqlen0 - sqrad).sqrt();
    let sqrt1 = (sqlen1 - sqrad).sqrt();

    let mut sol = [[[0.0f64; 2]; 2]; 2];
    let mut good = [0.0f64; 2];
    for i in 0..2 {
        let sgn = if i == 0 { 1.0 } else { -1.0 };
        sol[i][0][0] = (end[0] * sqrad + sgn * radius * end[1] * sqrt0) / sqlen0;
        sol[i][0][1] = (end[1] * sqrad - sgn * radius * end[0] * sqrt0) / sqlen0;
        sol[i][1][0] = (end[2] * sqrad - sgn * radius * end[3] * sqrt1) / sqlen1;
        sol[i][1][1] = (end[3] * sqrad + sgn * radius * end[2] * sqrt1) / sqlen1;
        if let Some(s) = side {
            let mut t = [sol[i][0][0] + sol[i][1][0], sol[i][0][1] + sol[i][1][1]];
            normalize2(&mut t);
            good[i] = dot2(&t, s);
        } else {
            let t = [sol[i][0][0] - sol[i][1][0], sol[i][0][1] - sol[i][1][1]];
            good[i] = -dot2(&t, &t);
        }
        if is_intersect(&end[0..2], &sol[i][0], &end[2..4], &sol[i][1]) {
            good[i] = -10000.0;
        }
    }
    let i = if good[0] > good[1] { 0 } else { 1 };
    *pnt = [sol[i][0][0], sol[i][0][1], sol[i][1][0], sol[i][1][1]];
    if is_intersect(&end[0..2], &pnt[0..2], &end[2..4], &pnt[2..4]) {
        return -1.0;
    }
    length_circle(&sol[i][0], &sol[i][1], i == 1, radius)
}

/// 2D inside wrap (side site inside the circle); returns 0 if wrap, -1 if no wrap.
fn wrap_inside(pnt: &mut [f64; 4], end: &[f64; 4], radius: f64) -> f64 {
    const MAXITER: usize = 20;
    const TOLERANCE: f64 = 1e-6;
    let len0 = norm2(&end[0..2]);
    let len1 = norm2(&end[2..4]);
    let dif = [end[2] - end[0], end[3] - end[1]];
    let dd = dif[0] * dif[0] + dif[1] * dif[1];
    if len0 <= radius || len1 <= radius || radius < MJ_MINVAL || len0 < MJ_MINVAL || len1 < MJ_MINVAL {
        return -1.0;
    }
    if dd > MJ_MINVAL {
        let a = -(dif[0] * end[0] + dif[1] * end[1]) / dd;
        if a > 0.0 && a < 1.0 {
            let tmp = [end[0] + a * dif[0], end[1] + a * dif[1]];
            if norm2(&tmp) <= radius {
                return -1.0;
            }
        }
    }
    // default for numerical failure: normalized average
    let mut avg = [0.5 * (end[0] + end[2]), 0.5 * (end[1] + end[3])];
    normalize2(&mut avg);
    *pnt = [avg[0] * radius, avg[1] * radius, avg[0] * radius, avg[1] * radius];

    let a_ = radius / len0;
    let b_ = radius / len1;
    let cos_g = (len0 * len0 + len1 * len1 - dd) / (2.0 * len0 * len1);
    if cos_g < -1.0 + MJ_MINVAL {
        return -1.0;
    } else if cos_g > 1.0 - MJ_MINVAL {
        return 0.0;
    }
    let g = cos_g.acos();
    let mut phi = PI / 2.0;
    let mut z = phi.sin();
    let mut f = (a_ * z).asin() + (b_ * z).asin() - 2.0 * phi + g;
    if f > 0.0 {
        return 0.0;
    }
    let mut iter = 0;
    while iter < MAXITER && f.abs() > TOLERANCE {
        let df = phi.cos()
            * (a_ / MJ_MINVAL.max((1.0 - z * z * a_ * a_).sqrt())
                + b_ / MJ_MINVAL.max((1.0 - z * z * b_ * b_).sqrt()))
            - 2.0;
        if df > -MJ_MINVAL {
            return 0.0;
        }
        let phi1 = phi - f / df;
        if phi1 > phi {
            return 0.0;
        }
        phi = phi1;
        z = phi.sin();
        f = (a_ * z).asin() + (b_ * z).asin() - 2.0 * phi + g;
        if f > TOLERANCE {
            return 0.0;
        }
        iter += 1;
    }
    if iter >= MAXITER {
        return 0.0;
    }
    let (mut vec, ang) = if end[0] * end[3] - end[1] * end[2] > 0.0 {
        ([end[0], end[1]], phi - (a_ * z).asin())
    } else {
        ([end[2], end[3]], phi - (b_ * z).asin())
    };
    normalize2(&mut vec);
    let (s, c) = ang.sin_cos();
    let p = [radius * (c * vec[0] - s * vec[1]), radius * (s * vec[0] + c * vec[1])];
    *pnt = [p[0], p[1], p[0], p[1]];
    0.0
}

/// `mju_wrap`: wrap segment x0-x1 around a sphere/cylinder at (xpos, xmat) with `radius`.
/// Returns the wrap length (>= 0) and the two 3D tangent points, or `None` for no wrap.
pub fn wrap(x0: &Vec3, x1: &Vec3, xpos: &Vec3, xmat: &Mat3, radius: f64, kind: i32,
            side: Option<&Vec3>) -> Option<(f64, [Vec3; 2])> {
    debug_assert!(kind == WRAP_SPHERE || kind == WRAP_CYLINDER);
    let p = [mat_t_vec(xmat, &sub3(x0, xpos)), mat_t_vec(xmat, &sub3(x1, xpos))];
    if norm3(&p[0]) < MJ_MINVAL || norm3(&p[1]) < MJ_MINVAL {
        return None;
    }
    let axis: [Vec3; 2] = if kind == WRAP_SPHERE {
        let mut a0 = p[0];
        normalize3(&mut a0);
        let mut normal = cross3(&p[0], &p[1]);
        let nrm = normalize3(&mut normal);
        if nrm < MJ_MINVAL {
            let mut i = 0;
            if a0[1].abs() > a0[0].abs() && a0[1].abs() > a0[2].abs() {
                i = 1;
            }
            if a0[2].abs() > a0[0].abs() && a0[2].abs() > a0[1].abs() {
                i = 2;
            }
            let mut a1 = [1.0, 1.0, 1.0];
            a1[i] = 0.0;
            normal = cross3(&a0, &a1);
            normalize3(&mut normal);
        }
        let mut a1 = cross3(&normal, &a0);
        normalize3(&mut a1);
        [a0, a1]
    } else {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    };
    let d = [dot3(&p[0], &axis[0]), dot3(&p[0], &axis[1]), dot3(&p[1], &axis[0]), dot3(&p[1], &axis[1])];

    let mut s_local = [0.0; 3];
    let mut sd = [0.0; 2];
    if let Some(sp) = side {
        s_local = mat_t_vec(xmat, &sub3(sp, xpos));
        sd = [dot3(&s_local, &axis[0]), dot3(&s_local, &axis[1])];
        normalize2(&mut sd);
        sd = [sd[0] * radius, sd[1] * radius];
    }
    let mut pnt = [0.0; 4];
    let mut wlen = if side.is_some() && norm3(&s_local) < radius {
        wrap_inside(&mut pnt, &d, radius)
    } else {
        wrap_circle(&mut pnt, &d, side.map(|_| &sd), radius)
    };
    if wlen < 0.0 {
        return None;
    }
    let mut res = [[0.0; 3]; 2];
    for i in 0..2 {
        res[i] = add3(&scl3(&axis[0], pnt[2 * i]), &scl3(&axis[1], pnt[2 * i + 1]));
    }
    if kind == WRAP_CYLINDER {
        let l0 = ((p[0][0] - res[0][0]).powi(2) + (p[0][1] - res[0][1]).powi(2)).sqrt();
        let l1 = ((p[1][0] - res[1][0]).powi(2) + (p[1][1] - res[1][1]).powi(2)).sqrt();
        res[0][2] = p[0][2] + (p[1][2] - p[0][2]) * l0 / (l0 + wlen + l1);
        res[1][2] = p[0][2] + (p[1][2] - p[0][2]) * (l0 + wlen) / (l0 + wlen + l1);
        let height = (res[1][2] - res[0][2]).abs();
        wlen = (wlen * wlen + height * height).sqrt();
    }
    let w0 = add3(&mat_vec(xmat, &res[0]), xpos);
    let w1 = add3(&mat_vec(xmat, &res[1]), xpos);
    Some((wlen, [w0, w1]))
}

/// Path of one spatial tendon: total length and the world-space polyline
/// (sites and, where wrapping is active, the two tangent points; arcs are not tessellated).
#[derive(Debug, Clone)]
pub struct TendonPath {
    pub length: f64,
    pub points: Vec<Vec3>,
}

impl Model {
    /// Length and path points of tendon `t` (validated site/sphere/cylinder path) at `k`.
    pub fn tendon_path(&self, k: &Kinematics, t: usize) -> Result<TendonPath, String> {
        let tn = self.tendons.get(t).ok_or(format!("tendon {t} out of range"))?;
        self.check_kin(k)?;
        let path = &tn.path;
        let n = path.len();
        let mut length = 0.0;
        let mut points = Vec::with_capacity(n + 2);
        let mut j = 0;
        while j + 1 < n {
            let (w0, w1) = (&path[j], &path[j + 1]);
            if w0.kind != WRAP_SITE {
                return Err(format!("tendon {}: path[{j}] must be a site", tn.name));
            }
            let x0 = k.site_pos[w0.object as usize];
            points.push(x0);
            if w1.kind == WRAP_SPHERE || w1.kind == WRAP_CYLINDER {
                let w2 = &path[j + 2];
                let x1 = k.site_pos[w2.object as usize];
                let g = w1.object as usize;
                let sideid = w1.prm.round() as i64;
                let side = (sideid >= 0).then(|| &k.site_pos[sideid as usize]);
                match wrap(&x0, &x1, &k.geom_pos[g], &k.geom_mat[g], self.geoms[g].size[0], w1.kind, side) {
                    Some((wlen, [a, b])) => {
                        length += dist3(&x0, &a) + wlen + dist3(&b, &x1);
                        points.push(a);
                        points.push(b);
                    }
                    None => length += dist3(&x0, &x1),
                }
                j += 2;
            } else {
                length += dist3(&x0, &k.site_pos[w1.object as usize]);
                j += 1;
            }
        }
        points.push(k.site_pos[path[n - 1].object as usize]);
        Ok(TendonPath { length, points })
    }

    /// Lengths of all tendons, tendon order.
    pub fn tendon_lengths(&self, k: &Kinematics) -> Result<Vec<f64>, String> {
        (0..self.tendons.len()).map(|t| self.tendon_path(k, t).map(|p| p.length)).collect()
    }

    /// Tendon velocities `dL/dt` along `qvel` at `q`, by centered differences of the exact
    /// lengths on the configuration manifold with step `h` (e.g. 1e-6 s). `qvel` is used as
    /// given (apply [`Model::project_velocity`] first for coupling-consistent velocities).
    pub fn tendon_velocities(&self, q: &[f64], qvel: &[f64], h: f64) -> Result<Vec<f64>, String> {
        if !(h > 0.0 && h.is_finite()) {
            return Err(format!("finite-difference step must be positive, got {h}"));
        }
        let qp = self.integrate_pos(q, qvel, h)?;
        let qm = self.integrate_pos(q, qvel, -h)?;
        let lp = self.tendon_lengths(&self.kinematics(&qp)?)?;
        let lm = self.tendon_lengths(&self.kinematics(&qm)?)?;
        Ok(lp.iter().zip(&lm).map(|(a, b)| (a - b) / (2.0 * h)).collect())
    }

    fn check_kin(&self, k: &Kinematics) -> Result<(), String> {
        if k.site_pos.len() != self.sites.len() || k.geom_pos.len() != self.geoms.len()
            || k.geom_mat.len() != self.geoms.len() {
            return Err("kinematics does not match this model".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const I3: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    #[test]
    fn sphere_wrap_matches_analytic_symmetric_case() {
        // endpoints at (+-2, -0.5... ) symmetric about y axis, unit sphere at origin,
        // segment passes through sphere: tangent length sqrt(d^2-1) each side + arc.
        let x0 = [-2.0, 0.0, 0.0];
        let x1 = [2.0, 0.0, 0.0];
        let side = [0.0, 1.0, 0.0];
        let (wlen, pts) = wrap(&x0, &x1, &[0.0; 3], &I3, 1.0, WRAP_SPHERE, Some(&side)).unwrap();
        // tangent point angle from x axis: acos(1/2) = 60deg -> arc between them 180-2*60 = 60deg
        assert!((wlen - PI / 3.0).abs() < 1e-12, "{wlen}");
        assert!(pts[0][1] > 0.0 && pts[1][1] > 0.0);
        let total = dist3(&x0, &pts[0]) + wlen + dist3(&pts[1], &x1);
        assert!((total - (2.0 * 3f64.sqrt() + PI / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn no_wrap_when_segment_misses() {
        let r = wrap(&[-2.0, 2.0, 0.0], &[2.0, 2.0, 0.0], &[0.0; 3], &I3, 1.0, WRAP_SPHERE, None);
        assert!(r.is_none());
    }

    #[test]
    fn cylinder_wrap_adds_helical_height() {
        let x0 = [-2.0, 0.0, -1.0];
        let x1 = [2.0, 0.0, 1.0];
        let side = [0.0, 1.0, 0.0];
        let (wlen, pts) = wrap(&x0, &x1, &[0.0; 3], &I3, 1.0, WRAP_CYLINDER, Some(&side)).unwrap();
        let planar = PI / 3.0;
        let t = 3f64.sqrt();
        let h = 2.0 * planar / (2.0 * t + planar);
        assert!((wlen - (planar * planar + h * h).sqrt()).abs() < 1e-12);
        assert!((pts[0][2] - (-1.0 + 2.0 * t / (2.0 * t + planar))).abs() < 1e-12);
    }

    #[test]
    fn inside_wrap_returns_point_on_circle() {
        // side site inside the sphere -> wrap_inside: path touches the circle once, from inside
        // the convex side; symmetric endpoints reflect at (0, 1, 0).
        let x0 = [-2.0, 1.5, 0.0];
        let x1 = [2.0, 1.5, 0.0];
        let side = [0.0, 0.2, 0.0];
        let (wlen, pts) = wrap(&x0, &x1, &[0.0; 3], &I3, 1.0, WRAP_SPHERE, Some(&side)).unwrap();
        assert_eq!(wlen, 0.0);
        assert!((norm3(&pts[0]) - 1.0).abs() < 1e-9);
        assert_eq!(pts[0], pts[1]);
        assert!(dist3(&pts[0], &[0.0, 1.0, 0.0]) < 1e-5, "{:?}", pts[0]);
    }
}
