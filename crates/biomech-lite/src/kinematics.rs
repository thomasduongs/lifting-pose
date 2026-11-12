//! Forward kinematics: a Rust port of MuJoCo 3.17.0 `mj_kinematics1/2`
//! (`src/engine/engine_core_smooth.c`, Copyright 2021 DeepMind Technologies Limited,
//! Apache-2.0) restricted to free/slide/hinge joints, no mocap bodies, no sleeping.

use crate::model::Model;
use crate::{JNT_FREE, JNT_HINGE, JNT_SLIDE, MJ_MINVAL};

pub type Vec3 = [f64; 3];
pub type Quat = [f64; 4];
pub type Mat3 = [f64; 9];

/// World-frame poses at one configuration (native index order, world body 0 included).
#[derive(Debug, Clone)]
pub struct Kinematics {
    pub body_pos: Vec<Vec3>,
    pub body_quat: Vec<Quat>,
    pub site_pos: Vec<Vec3>,
    pub geom_pos: Vec<Vec3>,
    pub geom_quat: Vec<Quat>,
    /// Row-major body rotation matrices (MuJoCo `xmat`).
    pub body_mat: Vec<Mat3>,
    /// Row-major geom rotation matrices (MuJoCo `geom_xmat`).
    pub geom_mat: Vec<Mat3>,
    /// Body inertial (COM) frame positions/orientations (MuJoCo `xipos`, quat of `ximat`).
    pub inertia_pos: Vec<Vec3>,
    pub inertia_quat: Vec<Quat>,
    /// Joint anchors and axes in world frame (MuJoCo `xanchor`, `xaxis`).
    pub joint_anchor: Vec<Vec3>,
    pub joint_axis: Vec<Vec3>,
}

// ------------------------------------------------------------------ spatial helpers (mju_*)

pub fn add3(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub3(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scl3(a: &Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot3(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross3(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm3(a: &Vec3) -> f64 {
    dot3(a, a).sqrt()
}
pub fn dist3(a: &Vec3, b: &Vec3) -> f64 {
    norm3(&sub3(a, b))
}
/// `mju_normalize3`: returns the original norm; tiny vectors become (1,0,0).
pub fn normalize3(a: &mut Vec3) -> f64 {
    let n = norm3(a);
    if n < MJ_MINVAL {
        *a = [1.0, 0.0, 0.0];
    } else {
        let s = 1.0 / n;
        a.iter_mut().for_each(|x| *x *= s);
    }
    n
}
/// `mju_mulMatVec3` (row-major).
pub fn mat_vec(m: &Mat3, v: &Vec3) -> Vec3 {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}
/// `mju_mulMatTVec3`.
pub fn mat_t_vec(m: &Mat3, v: &Vec3) -> Vec3 {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}
/// `mju_normalize4`: tiny quaternions become identity.
pub fn normalize4(q: &mut Quat) -> f64 {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < MJ_MINVAL {
        *q = [1.0, 0.0, 0.0, 0.0];
    } else if (n - 1.0).abs() > MJ_MINVAL {
        let s = 1.0 / n;
        q.iter_mut().for_each(|x| *x *= s);
    }
    n
}
/// `mju_mulQuat` (Hamilton product, wxyz).
pub fn quat_mul(a: &Quat, b: &Quat) -> Quat {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
pub fn quat_conj(q: &Quat) -> Quat {
    [q[0], -q[1], -q[2], -q[3]]
}
/// `mju_axisAngle2Quat`.
pub fn axis_angle_quat(axis: &Vec3, angle: f64) -> Quat {
    if angle == 0.0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let (s, c) = (0.5 * angle).sin_cos();
    [c, axis[0] * s, axis[1] * s, axis[2] * s]
}
/// `mju_rotVecQuat`: rotate `v` by unit quaternion `q`.
pub fn quat_rotate(q: &Quat, v: &Vec3) -> Vec3 {
    if q[0] == 1.0 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
        return *v;
    }
    let u = [q[1], q[2], q[3]];
    let t = add3(&scl3(v, q[0]), &cross3(&u, v));
    add3(v, &scl3(&cross3(&u, &t), 2.0))
}
/// `mju_quat2Mat` (row-major).
pub fn quat_to_mat(q: &Quat) -> Mat3 {
    if q[0] == 1.0 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let (q00, q01, q02, q03) = (q[0] * q[0], q[0] * q[1], q[0] * q[2], q[0] * q[3]);
    let (q11, q12, q13) = (q[1] * q[1], q[1] * q[2], q[1] * q[3]);
    let (q22, q23, q33) = (q[2] * q[2], q[2] * q[3], q[3] * q[3]);
    [
        q00 + q11 - q22 - q33, 2.0 * (q12 - q03), 2.0 * (q13 + q02),
        2.0 * (q12 + q03), q00 - q11 + q22 - q33, 2.0 * (q23 - q01),
        2.0 * (q13 - q02), 2.0 * (q23 + q01), q00 - q11 - q22 + q33,
    ]
}
/// `mju_quatIntegrate`: `q <- q * exp(omega * h)` with `omega` in the local frame.
pub fn quat_integrate(q: &Quat, omega: &Vec3, h: f64) -> Quat {
    let mut axis = *omega;
    let ang = normalize3(&mut axis) * h;
    let mut r = quat_mul(q, &axis_angle_quat(&axis, ang));
    normalize4(&mut r);
    r
}
/// Rotation vector `log(q)` (radians * axis) of a unit quaternion, shortest path.
pub fn quat_log(q: &Quat) -> Vec3 {
    let mut q = *q;
    if q[0] < 0.0 {
        q = [-q[0], -q[1], -q[2], -q[3]];
    }
    let s = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if s < MJ_MINVAL {
        return [2.0 * q[1], 2.0 * q[2], 2.0 * q[3]];
    }
    let ang = 2.0 * s.atan2(q[0]);
    [q[1] / s * ang, q[2] / s * ang, q[3] / s * ang]
}

/// Frame composition `mj_local2Global`: world pose of a local (pos, quat) attached to a body.
fn local_to_global(bpos: &Vec3, bquat: &Quat, bmat: &Mat3, pos: &Vec3, quat: &Quat) -> (Vec3, Quat, Mat3) {
    let p = add3(&mat_vec(bmat, pos), bpos);
    let mut q = quat_mul(bquat, quat);
    normalize4(&mut q);
    (p, q, quat_to_mat(&q))
}

impl Model {
    /// Forward kinematics of the whole model at `q` (length nq). `q` is used as given; call
    /// [`Model::project`] first if couplings must hold.
    pub fn kinematics(&self, q: &[f64]) -> Result<Kinematics, String> {
        if q.len() != self.nq {
            return Err(format!("q length {} != nq {}", q.len(), self.nq));
        }
        if let Some(i) = q.iter().position(|x| !x.is_finite()) {
            return Err(format!("q[{i}] is not finite"));
        }
        let nb = self.bodies.len();
        let nj = self.joints.len();
        let ident: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut xpos = vec![[0.0; 3]; nb];
        let mut xquat = vec![[1.0, 0.0, 0.0, 0.0]; nb];
        let mut xmat = vec![ident; nb];
        let mut xanchor = vec![[0.0; 3]; nj];
        let mut xaxis = vec![[0.0; 3]; nj];

        for i in 1..nb {
            let b = &self.bodies[i];
            let (mut p, mut qt);
            if b.joints.len() == 1 && self.joints[b.joints[0]].kind == JNT_FREE {
                let j = &self.joints[b.joints[0]];
                let a = j.qposadr;
                p = [q[a], q[a + 1], q[a + 2]];
                qt = [q[a + 3], q[a + 4], q[a + 5], q[a + 6]];
                normalize4(&mut qt);
                xanchor[b.joints[0]] = p;
                xaxis[b.joints[0]] = j.axis;
            } else {
                let pid = b.parent;
                if pid != 0 {
                    p = add3(&mat_vec(&xmat[pid], &b.pos), &xpos[pid]);
                    qt = quat_mul(&xquat[pid], &b.quat);
                } else {
                    p = b.pos;
                    qt = b.quat;
                }
                for &jid in &b.joints {
                    let j = &self.joints[jid];
                    let a = j.qposadr;
                    let axis = quat_rotate(&qt, &j.axis);
                    let anchor = add3(&quat_rotate(&qt, &j.pos), &p);
                    match j.kind {
                        JNT_SLIDE => {
                            p = add3(&p, &scl3(&axis, q[a] - self.qpos0[a]));
                        }
                        JNT_HINGE => {
                            let qloc = axis_angle_quat(&j.axis, q[a] - self.qpos0[a]);
                            qt = quat_mul(&qt, &qloc);
                            let v = quat_rotate(&qt, &j.pos);
                            p = sub3(&anchor, &v);
                        }
                        k => return Err(format!("joint {}: unsupported kind {k}", j.name)),
                    }
                    xanchor[jid] = anchor;
                    xaxis[jid] = axis;
                }
            }
            normalize4(&mut qt);
            xpos[i] = p;
            xquat[i] = qt;
            xmat[i] = quat_to_mat(&qt);
        }

        let mut inertia_pos = vec![[0.0; 3]; nb];
        let mut inertia_quat = vec![[1.0, 0.0, 0.0, 0.0]; nb];
        for i in 1..nb {
            let b = &self.bodies[i];
            let (p, qq, _) = local_to_global(&xpos[i], &xquat[i], &xmat[i], &b.ipos, &b.iquat);
            inertia_pos[i] = p;
            inertia_quat[i] = qq;
        }
        let ng = self.geoms.len();
        let mut geom_pos = Vec::with_capacity(ng);
        let mut geom_quat = Vec::with_capacity(ng);
        let mut geom_mat = Vec::with_capacity(ng);
        for g in &self.geoms {
            let (p, qq, m) = local_to_global(&xpos[g.body], &xquat[g.body], &xmat[g.body], &g.pos, &g.quat);
            geom_pos.push(p);
            geom_quat.push(qq);
            geom_mat.push(m);
        }
        let site_pos = self.sites.iter()
            .map(|s| add3(&mat_vec(&xmat[s.body], &s.pos), &xpos[s.body]))
            .collect();

        Ok(Kinematics {
            body_pos: xpos, body_quat: xquat, site_pos, geom_pos, geom_quat, body_mat: xmat,
            geom_mat, inertia_pos, inertia_quat, joint_anchor: xanchor, joint_axis: xaxis,
        })
    }

    /// `mj_integratePos`: `q + h * qvel` on the configuration manifold (free-joint quaternion
    /// integrated with local angular velocity, scalar joints added).
    pub fn integrate_pos(&self, q: &[f64], qvel: &[f64], h: f64) -> Result<Vec<f64>, String> {
        if q.len() != self.nq || qvel.len() != self.nv {
            return Err(format!("q/qvel lengths {}/{} != nq/nv {}/{}", q.len(), qvel.len(), self.nq, self.nv));
        }
        let mut out = q.to_vec();
        for j in &self.joints {
            let (a, d) = (j.qposadr, j.dofadr);
            if j.kind == JNT_FREE {
                for k in 0..3 {
                    out[a + k] += h * qvel[d + k];
                }
                let qt = [q[a + 3], q[a + 4], q[a + 5], q[a + 6]];
                let r = quat_integrate(&qt, &[qvel[d + 3], qvel[d + 4], qvel[d + 5]], h);
                out[a + 3..a + 7].copy_from_slice(&r);
            } else {
                out[a] += h * qvel[d];
            }
        }
        Ok(out)
    }
}
