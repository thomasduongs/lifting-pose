//! Live constrained anatomical arm fitting from camera landmarks (see
//! `docs/live-biomech-contract.md`, "Fitting library API").
//!
//! Camera-driven constrained kinematics only: the eight independent native arm coordinates
//! (`model.dynamic_joints`: elv_angle, shoulder_elv, shoulder_rot, elbow_flexion per side) are
//! fitted to *directions* of the observed upper arm / forearm, retargeted onto the reference
//! segment lengths. Root, torso, wrists, fingers and lower body are held at the standing prior.
//! Every native coupling is applied exactly ([`Model::project`]) on every residual evaluation and
//! every accepted state satisfies all scalar joint ranges (masters are boxed to the effective
//! interval that keeps every dependent inside its own range). Nothing here is a measurement of
//! muscle activation or force.
//!
//! Frames:
//! * camera: native MediaPipe x-right / y-down / z-forward (metres; hip relative or not, only
//!   differences are used). Torso basis columns: lateral = normalize(L_shoulder - R_shoulder),
//!   up = [0,-1,0] orthogonalised against lateral, forward = lateral x up.
//! * model: the same construction from the standing glenohumeral anchors with up = +z.
//!
//! Camera directions are mapped by `R = B_model * B_camera^T`, a proper rotation (both bases are
//! right-handed), so left/right are never reflected.

use std::sync::Arc;

use nalgebra::{Matrix4, Vector4};

use crate::kinematics::{
    Mat3, Quat, Vec3, add3, axis_angle_quat, cross3, dot3, mat_vec, norm3, normalize4, quat_mul,
    quat_rotate, quat_to_mat, scl3, sub3,
};
use crate::model::{Model, coupling_poly};
use crate::{JNT_FREE, JNT_HINGE, JNT_SLIDE};

/// Landmarks below this (clipped) confidence are treated as missing.
pub const SCORE_FLOOR: f64 = 0.3;
/// Huber threshold on per-marker residual norm (metres).
pub const HUBER_DELTA_M: f64 = 0.04;
/// Total LM trial steps per arm per update (warm solve + optional reseed).
pub const MAX_ITERATIONS: usize = 12;
/// Explicit rate limit on every fitted coordinate (rad/s). Applied after the solve.
pub const MAX_RATE_RAD_S: f64 = 10.0;
/// Shortest accepted observed segment (shoulder line, upper arm, forearm), metres.
pub const MIN_SEGMENT_M: f64 = 0.05;

const WARM_ITERS: usize = 8;
const RESEED_ITERS: usize = MAX_ITERATIONS - WARM_ITERS;
/// Temporal prior, metres-equivalent per radian of change from the previous fit.
const CONTINUITY: f64 = 0.08;
/// Neutral prior towards the standing pose (weak), metres-equivalent per radian.
const NEUTRAL: f64 = 0.01;
/// Extra temporal weight on axially unobservable coordinates (plane when the arm hangs,
/// humeral rotation when the elbow is straight).
const AX_GAIN: f64 = 8.0;
/// Warm-fit marker RMS above which a coarse global reseed is attempted.
const RESEED_RMS_M: f64 = 0.06;
const FD_STEP: f64 = 1e-5;
const DT_MIN: f64 = 1.0 / 240.0;
const DT_MAX: f64 = 0.1;
const DT_NOMINAL: f64 = 1.0 / 30.0;
/// |up orthogonalised against the shoulder line| below this = shoulder line too vertical.
const MIN_UP_ORTHO: f64 = 0.3;

const SIDE_NAMES: [&str; 2] = ["left", "right"];
const SIDE_SFX: [&str; 2] = ["_l", "_r"];
/// Index of each coordinate within an arm's 4-vector.
const PLANE: usize = 0;
const ELEV: usize = 1;
const ROT: usize = 2;
const FLEX: usize = 3;
const MASTER_STEMS: [&str; 4] = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion"];

#[derive(Clone, Debug)]
pub struct Landmark {
    pub name: String,
    pub position: [f64; 3],
    pub confidence: f64,
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub timestamp_seconds: f64,
    pub landmarks: Vec<Landmark>,
}

#[derive(Clone, Debug)]
pub struct FitResult {
    /// Full projected configuration (length nq).
    pub qpos: Vec<f64>,
    /// RMS of shoulder-relative elbow/wrist offset errors over targets used this update.
    pub rmse_m: Option<f64>,
    /// LEFT then RIGHT: true when that arm was fitted to observations this update.
    pub supported_arms: [bool; 2],
    pub status: String,
    /// LM trial steps this update, summed over both arms.
    pub iterations: usize,
}

/// Native marker positions per side (LEFT, RIGHT): [shoulder, elbow, wrist] in model world.
pub type ArmMarkers = [[Vec3; 3]; 2];

#[derive(Clone, Debug)]
struct ArmSpec {
    joints: [usize; 4],
    qadr: [usize; 4],
    /// Effective box: native range intersected with what keeps every dependent in range.
    lo: [f64; 4],
    hi: [f64; 4],
    shoulder_joint: usize,
    elbow_sites: [usize; 2],
    wrist_joint: usize,
    /// Ancestor-closed bodies needed for this arm's markers, ascending (parents first).
    chain: Vec<usize>,
    /// Couplings whose root master is one of this arm's masters, in native order.
    couplings: Vec<usize>,
    upper_len: f64,
    fore_len: f64,
}

#[derive(Clone, Debug)]
struct Fk {
    xpos: Vec<Vec3>,
    xquat: Vec<Quat>,
    xmat: Vec<Mat3>,
    anchor: Vec<Vec3>,
}

impl Fk {
    fn new(m: &Model) -> Self {
        let ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        Fk {
            xpos: vec![[0.0; 3]; m.bodies.len()],
            xquat: vec![[1.0, 0.0, 0.0, 0.0]; m.bodies.len()],
            xmat: vec![ident; m.bodies.len()],
            anchor: vec![[0.0; 3]; m.joints.len()],
        }
    }
}

/// Per-arm least-squares problem for one update.
#[derive(Clone, Debug)]
struct ArmProblem {
    tgt_elbow: Vec3,
    tgt_wrist: Option<Vec3>,
    w_elbow: f64,
    w_wrist: f64,
    x_prev: [f64; 4],
    x_neutral: [f64; 4],
    /// Per-coordinate temporal weight (CONTINUITY * unobservability multiplier).
    lam_t: [f64; 4],
    free: [bool; 4],
}

const NRES: usize = 14;

/// Immutable fitting data shared by all updates.
#[derive(Clone, Debug)]
struct Core {
    model: Arc<Model>,
    arms: [ArmSpec; 2],
    /// Projected standing configuration (the fixed prior for everything not fitted).
    standing: Vec<f64>,
    /// Model torso basis, row-major, columns (lateral, up, forward).
    b_model: Mat3,
    /// Limited scalar joints to verify on every accepted state: (qposadr, lo, hi).
    limited: Vec<(usize, f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct LiveFitter {
    core: Core,
    fk: Fk,
    q: Vec<f64>,
    x: [[f64; 4]; 2],
    last_fit_t: [Option<f64>; 2],
    camera_lateral: Option<Vec3>,
    last_targets: [Option<(Vec3, Option<Vec3>)>; 2],
    settled: [u8; 2],
}

// ------------------------------------------------------------------ small helpers

fn finite3(v: &Vec3) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn unit(v: &Vec3, min_len: f64) -> Option<Vec3> {
    let n = norm3(v);
    if !n.is_finite() || n < min_len {
        return None;
    }
    Some(scl3(v, 1.0 / n))
}

/// Row-major matrix with the given columns.
fn from_cols(c0: &Vec3, c1: &Vec3, c2: &Vec3) -> Mat3 {
    [
        c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2],
    ]
}

/// Columns (lateral, up, forward) from a lateral vector and a reference up.
fn torso_basis(lateral: &Vec3, up_ref: &Vec3) -> Option<Mat3> {
    let lat = unit(lateral, MIN_SEGMENT_M)?;
    let up = sub3(up_ref, &scl3(&lat, dot3(up_ref, &lat)));
    let up = unit(&up, MIN_UP_ORTHO)?;
    let fwd = cross3(&lat, &up);
    Some(from_cols(&lat, &up, &fwd))
}

/// `a * b^T` for row-major 3x3 matrices.
fn mul_abt(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[3 * i + j] = (0..3).map(|k| a[3 * i + k] * b[3 * j + k]).sum();
        }
    }
    r
}

fn huber(r: f64) -> f64 {
    if r <= HUBER_DELTA_M {
        0.5 * r * r
    } else {
        HUBER_DELTA_M * (r - 0.5 * HUBER_DELTA_M)
    }
}

/// IRLS scale on a residual of norm `r` so that 0.5*|s e|^2 matches Huber's local quadratic.
fn huber_scale(r: f64) -> f64 {
    if r <= HUBER_DELTA_M {
        1.0
    } else {
        (HUBER_DELTA_M / r).sqrt()
    }
}

// ------------------------------------------------------------------ core

impl Core {
    fn new(model: Arc<Model>) -> Result<Self, String> {
        let m = &*model;
        let mut standing = m.standing_qpos.clone();
        m.project(&mut standing)?;
        let limited: Vec<(usize, f64, f64)> = m
            .joints
            .iter()
            .filter(|j| j.limited && j.is_scalar())
            .map(|j| (j.qposadr, j.range[0], j.range[1]))
            .collect();
        for &(a, lo, hi) in &limited {
            if !(lo - 1e-12..=hi + 1e-12).contains(&standing[a]) {
                return Err(format!("standing prior violates joint range at qpos[{a}]"));
            }
        }
        // Root master of every coupling (couplings are topologically sorted by validation).
        let mut root_of_dep: Vec<Option<usize>> = vec![None; m.joints.len()];
        let mut coupling_root: Vec<Option<usize>> = Vec::with_capacity(m.couplings.len());
        for c in &m.couplings {
            let root = c.master.map(|mj| root_of_dep[mj].unwrap_or(mj));
            root_of_dep[c.dependent] = root;
            coupling_root.push(root);
        }

        let mut arms = Vec::with_capacity(2);
        for side in 0..2 {
            let sfx = SIDE_SFX[side];
            let mut joints = [0usize; 4];
            for (k, stem) in MASTER_STEMS.iter().enumerate() {
                let name = format!("{stem}{sfx}");
                let j = m
                    .joint_id(&name)
                    .ok_or(format!("model lacks joint {name}"))?;
                if !m.dynamic_joints.contains(&j) {
                    return Err(format!("joint {name} is not one of model.dynamic_joints"));
                }
                joints[k] = j;
            }
            let qadr = joints.map(|j| m.joints[j].qposadr);
            let shoulder_joint = m
                .joint_id(&format!("shoulder_elv{sfx}"))
                .ok_or(format!("model lacks joint shoulder_elv{sfx}"))?;
            let wrist_joint = m
                .joint_id(&format!("flexion{sfx}"))
                .ok_or(format!("model lacks joint flexion{sfx}"))?;
            let mut elbow_sites = [0usize; 2];
            for (k, nm) in ["Elbow_Lateral_marker", "Elbow_Medial_marker"]
                .iter()
                .enumerate()
            {
                let name = format!("{nm}{sfx}");
                elbow_sites[k] = m.site_id(&name).ok_or(format!("model lacks site {name}"))?;
            }
            let mut need = vec![false; m.bodies.len()];
            for b0 in [
                m.joints[shoulder_joint].body,
                m.joints[wrist_joint].body,
                m.sites[elbow_sites[0]].body,
                m.sites[elbow_sites[1]].body,
            ] {
                let mut b = b0;
                while b != 0 && !need[b] {
                    need[b] = true;
                    b = m.bodies[b].parent;
                }
            }
            let chain: Vec<usize> = (1..m.bodies.len()).filter(|&b| need[b]).collect();
            let couplings: Vec<usize> = (0..m.couplings.len())
                .filter(|&c| coupling_root[c].is_some_and(|r| joints.contains(&r)))
                .collect();
            arms.push(ArmSpec {
                joints,
                qadr,
                lo: [0.0; 4],
                hi: [0.0; 4],
                shoulder_joint,
                elbow_sites,
                wrist_joint,
                chain,
                couplings,
                upper_len: 0.0,
                fore_len: 0.0,
            });
        }
        let arms: [ArmSpec; 2] = [arms[0].clone(), arms[1].clone()];
        let mut core = Core {
            model,
            arms,
            standing,
            b_model: [0.0; 9],
            limited,
        };

        // Effective master boxes (dependent ranges can be tighter than the master's own range).
        for side in 0..2 {
            for k in 0..4 {
                let (lo, hi) = core.effective_bounds(side, k)?;
                core.arms[side].lo[k] = lo;
                core.arms[side].hi[k] = hi;
            }
        }
        // Reference segment lengths and model torso basis from the standing pose.
        let mk = core.markers_full(&core.standing.clone())?;
        for side in 0..2 {
            let up = norm3(&sub3(&mk[side][1], &mk[side][0]));
            let fore = norm3(&sub3(&mk[side][2], &mk[side][1]));
            if !(up > MIN_SEGMENT_M && fore > MIN_SEGMENT_M) {
                return Err(format!(
                    "degenerate {} reference arm segments",
                    SIDE_NAMES[side]
                ));
            }
            core.arms[side].upper_len = up;
            core.arms[side].fore_len = fore;
        }
        core.b_model = torso_basis(&sub3(&mk[0][0], &mk[1][0]), &[0.0, 0.0, 1.0])
            .ok_or("degenerate model shoulder line")?;
        Ok(core)
    }

    /// Feasible interval of one master (others at standing) containing its standing value.
    fn effective_bounds(&self, side: usize, k: usize) -> Result<(f64, f64), String> {
        let m = &*self.model;
        let arm = &self.arms[side];
        let jt = &m.joints[arm.joints[k]];
        let a = arm.qadr[k];
        let s = self.standing[a];
        let (nlo, nhi) = if jt.limited {
            (jt.range[0], jt.range[1])
        } else {
            (
                s - 2.0 * std::f64::consts::PI,
                s + 2.0 * std::f64::consts::PI,
            )
        };
        let mut q = self.standing.clone();
        let mut feasible = |v: f64| {
            q[a] = v;
            self.project_arm(side, &mut q);
            arm.couplings.iter().all(|&c| {
                let d = &m.joints[m.couplings[c].dependent];
                !d.limited || (d.range[0]..=d.range[1]).contains(&q[d.qposadr])
            })
        };
        if !feasible(s) {
            return Err(format!(
                "standing value of {} makes a coupled dependent infeasible",
                jt.name
            ));
        }
        let mut bound = |end: f64| -> f64 {
            let n = (((end - s).abs() / 1e-3).ceil() as usize).max(1);
            let mut good = s;
            for i in 1..=n {
                let v = s + (end - s) * i as f64 / n as f64;
                if feasible(v) {
                    good = v;
                } else {
                    let mut bad = v;
                    for _ in 0..60 {
                        let mid = 0.5 * (good + bad);
                        if feasible(mid) { good = mid } else { bad = mid }
                    }
                    return good;
                }
            }
            good
        };
        let hi = bound(nhi);
        let lo = bound(nlo);
        Ok((lo, hi))
    }

    /// Apply this arm's couplings in native order (same arithmetic as [`Model::project`]).
    fn project_arm(&self, side: usize, q: &mut [f64]) {
        let m = &*self.model;
        for &ci in &self.arms[side].couplings {
            let c = &m.couplings[ci];
            let dq = m.joints[c.dependent].qposadr;
            q[dq] = match c.master {
                None => m.qpos0[dq] + c.coef[0],
                Some(mj) => {
                    let mq = m.joints[mj].qposadr;
                    m.qpos0[dq] + coupling_poly(&c.coef, q[mq] - m.qpos0[mq]).0
                }
            };
        }
    }

    /// Port of the body loop of [`Model::kinematics`] restricted to an ancestor-closed chain.
    fn chain_fk(&self, q: &[f64], side: usize, fk: &mut Fk) {
        let m = &*self.model;
        for &i in &self.arms[side].chain {
            let b = &m.bodies[i];
            let (mut p, mut qt);
            if b.joints.len() == 1 && m.joints[b.joints[0]].kind == JNT_FREE {
                let a = m.joints[b.joints[0]].qposadr;
                p = [q[a], q[a + 1], q[a + 2]];
                qt = [q[a + 3], q[a + 4], q[a + 5], q[a + 6]];
                normalize4(&mut qt);
                fk.anchor[b.joints[0]] = p;
            } else {
                let pid = b.parent;
                if pid != 0 {
                    p = add3(&mat_vec(&fk.xmat[pid], &b.pos), &fk.xpos[pid]);
                    qt = quat_mul(&fk.xquat[pid], &b.quat);
                } else {
                    p = b.pos;
                    qt = b.quat;
                }
                for &jid in &b.joints {
                    let j = &m.joints[jid];
                    let a = j.qposadr;
                    let anchor = add3(&quat_rotate(&qt, &j.pos), &p);
                    match j.kind {
                        JNT_SLIDE => {
                            let axis = quat_rotate(&qt, &j.axis);
                            p = add3(&p, &scl3(&axis, q[a] - m.qpos0[a]));
                        }
                        JNT_HINGE => {
                            qt = quat_mul(&qt, &axis_angle_quat(&j.axis, q[a] - m.qpos0[a]));
                            p = sub3(&anchor, &quat_rotate(&qt, &j.pos));
                        }
                        _ => {}
                    }
                    fk.anchor[jid] = anchor;
                }
            }
            normalize4(&mut qt);
            fk.xpos[i] = p;
            fk.xquat[i] = qt;
            fk.xmat[i] = quat_to_mat(&qt);
        }
    }

    fn arm_markers(&self, q: &[f64], side: usize, fk: &mut Fk) -> [Vec3; 3] {
        self.chain_fk(q, side, fk);
        let m = &*self.model;
        let arm = &self.arms[side];
        let site = |s: usize| {
            let st = &m.sites[s];
            add3(&mat_vec(&fk.xmat[st.body], &st.pos), &fk.xpos[st.body])
        };
        let e = scl3(
            &add3(&site(arm.elbow_sites[0]), &site(arm.elbow_sites[1])),
            0.5,
        );
        [fk.anchor[arm.shoulder_joint], e, fk.anchor[arm.wrist_joint]]
    }

    fn markers_full(&self, q: &[f64]) -> Result<ArmMarkers, String> {
        if q.len() != self.model.nq {
            return Err(format!("qpos length {} != nq {}", q.len(), self.model.nq));
        }
        if q.iter().any(|v| !v.is_finite()) {
            return Err("qpos is not finite".into());
        }
        let mut fk = Fk::new(&self.model);
        Ok([
            self.arm_markers(q, 0, &mut fk),
            self.arm_markers(q, 1, &mut fk),
        ])
    }

    /// Shoulder-relative (elbow, wrist) offsets at arm coordinates `x`; `q` is a scratch copy of
    /// the current configuration (its arm masters/dependents are overwritten).
    fn offsets(&self, side: usize, x: &[f64; 4], q: &mut [f64], fk: &mut Fk) -> (Vec3, Vec3) {
        for k in 0..4 {
            q[self.arms[side].qadr[k]] = x[k];
        }
        self.project_arm(side, q);
        let mk = self.arm_markers(q, side, fk);
        (sub3(&mk[1], &mk[0]), sub3(&mk[2], &mk[0]))
    }

    fn arm_feasible(&self, side: usize, q: &[f64]) -> bool {
        let m = &*self.model;
        let arm = &self.arms[side];
        let ok = |j: usize| {
            let jt = &m.joints[j];
            let v = q[jt.qposadr];
            v.is_finite() && (!jt.limited || (jt.range[0]..=jt.range[1]).contains(&v))
        };
        arm.joints.iter().all(|&j| ok(j))
            && arm.couplings.iter().all(|&c| ok(m.couplings[c].dependent))
    }

    // -------------------------------------------------------------- objective

    fn marker_cost(p: &ArmProblem, off: &(Vec3, Vec3)) -> f64 {
        let mut c = p.w_elbow * huber(norm3(&sub3(&off.0, &p.tgt_elbow)));
        if let Some(tw) = p.tgt_wrist {
            c += p.w_wrist * huber(norm3(&sub3(&off.1, &tw)));
        }
        c
    }

    fn cost(p: &ArmProblem, x: &[f64; 4], off: &(Vec3, Vec3)) -> f64 {
        let mut c = Self::marker_cost(p, off);
        for k in 0..4 {
            if p.free[k] {
                c += 0.5 * (p.lam_t[k] * (x[k] - p.x_prev[k])).powi(2)
                    + 0.5 * (NEUTRAL * (x[k] - p.x_neutral[k])).powi(2);
            }
        }
        c
    }

    fn irls(p: &ArmProblem, off: &(Vec3, Vec3)) -> [f64; 2] {
        let we = (p.w_elbow * huber_scale(norm3(&sub3(&off.0, &p.tgt_elbow)))).sqrt();
        let ww = match p.tgt_wrist {
            Some(tw) => (p.w_wrist * huber_scale(norm3(&sub3(&off.1, &tw)))).sqrt(),
            None => 0.0,
        };
        [we, ww]
    }

    fn residual(p: &ArmProblem, x: &[f64; 4], off: &(Vec3, Vec3), w: &[f64; 2]) -> [f64; NRES] {
        let mut r = [0.0; NRES];
        let de = sub3(&off.0, &p.tgt_elbow);
        for i in 0..3 {
            r[i] = w[0] * de[i];
        }
        if let Some(tw) = p.tgt_wrist {
            let dw = sub3(&off.1, &tw);
            for i in 0..3 {
                r[3 + i] = w[1] * dw[i];
            }
        }
        for k in 0..4 {
            if p.free[k] {
                r[6 + k] = p.lam_t[k] * (x[k] - p.x_prev[k]);
                r[10 + k] = NEUTRAL * (x[k] - p.x_neutral[k]);
            }
        }
        r
    }

    /// Bounded Levenberg-Marquardt with a central finite-difference Jacobian and IRLS Huber
    /// weights. Returns (x, cost, trials). Only accepted states are returned; each is boxed to
    /// the effective bounds and verified against every coupled dependent range.
    fn solve(
        &self,
        side: usize,
        p: &ArmProblem,
        x0: [f64; 4],
        max_iter: usize,
        q: &mut [f64],
        fk: &mut Fk,
    ) -> ([f64; 4], f64, usize) {
        let arm = &self.arms[side];
        let mut x = x0;
        let mut off = self.offsets(side, &x, q, fk);
        let mut cost = Self::cost(p, &x, &off);
        let mut mu = 1e-3;
        let mut jac: Option<(Matrix4<f64>, Vector4<f64>)> = None;
        let mut trials = 0;
        while trials < max_iter {
            trials += 1;
            let (h, g) = match jac {
                Some(v) => v,
                None => {
                    let w = Self::irls(p, &off);
                    let r0 = Self::residual(p, &x, &off, &w);
                    let mut jm = [[0.0; NRES]; 4];
                    for k in 0..4 {
                        if !p.free[k] {
                            continue;
                        }
                        let (mut xp, mut xm) = (x, x);
                        xp[k] += FD_STEP;
                        xm[k] -= FD_STEP;
                        let op = self.offsets(side, &xp, q, fk);
                        let om = self.offsets(side, &xm, q, fk);
                        let rp = Self::residual(p, &xp, &op, &w);
                        let rm = Self::residual(p, &xm, &om, &w);
                        for i in 0..NRES {
                            jm[k][i] = (rp[i] - rm[i]) / (2.0 * FD_STEP);
                        }
                    }
                    let mut h = Matrix4::zeros();
                    let mut g = Vector4::zeros();
                    for a in 0..4 {
                        g[a] = (0..NRES).map(|i| jm[a][i] * r0[i]).sum();
                        for b in 0..4 {
                            h[(a, b)] = (0..NRES).map(|i| jm[a][i] * jm[b][i]).sum();
                        }
                    }
                    jac = Some((h, g));
                    (h, g)
                }
            };
            let mut a = h;
            let mut rhs = -g;
            for k in 0..4 {
                if p.free[k] {
                    a[(k, k)] += mu * (h[(k, k)] + 1e-9);
                } else {
                    for j in 0..4 {
                        a[(k, j)] = 0.0;
                        a[(j, k)] = 0.0;
                    }
                    a[(k, k)] = 1.0;
                    rhs[k] = 0.0;
                }
            }
            let Some(step) = a.cholesky().map(|c| c.solve(&rhs)) else {
                mu *= 4.0;
                continue;
            };
            let mut xn = x;
            for k in 0..4 {
                if p.free[k] && step[k].is_finite() {
                    xn[k] = (x[k] + step[k]).clamp(arm.lo[k], arm.hi[k]);
                }
            }
            let moved = (0..4).map(|k| (xn[k] - x[k]).abs()).fold(0.0, f64::max);
            if moved < 1e-8 {
                break;
            }
            let on = self.offsets(side, &xn, q, fk);
            let cn = Self::cost(p, &xn, &on);
            if cn.is_finite() && cn < cost && self.arm_feasible(side, q) {
                let gain = cost - cn;
                x = xn;
                off = on;
                cost = cn;
                jac = None;
                mu = (mu / 3.0).max(1e-7);
                if gain < 1e-12 {
                    break;
                }
            } else {
                mu *= 4.0;
                if mu > 1e8 {
                    break;
                }
            }
        }
        (x, cost, trials)
    }
}

// ------------------------------------------------------------------ public API

enum ArmObs {
    Fit {
        tgt_elbow: Vec3,
        tgt_wrist: Option<Vec3>,
        w_elbow: f64,
        w_wrist: f64,
    },
    Held(&'static str),
}

impl LiveFitter {
    pub fn new(model: Arc<Model>) -> Result<Self, String> {
        let core = Core::new(model)?;
        let fk = Fk::new(&core.model);
        let mut f = LiveFitter {
            q: core.standing.clone(),
            x: [[0.0; 4]; 2],
            fk,
            core,
            last_fit_t: [None; 2],
            camera_lateral: None,
            last_targets: [None; 2],
            settled: [0; 2],
        };
        f.reset();
        Ok(f)
    }

    /// Return to the standing prior and forget timing/warm-start history.
    pub fn reset(&mut self) {
        self.q = self.core.standing.clone();
        for side in 0..2 {
            self.x[side] = self.core.arms[side].qadr.map(|a| self.q[a]);
        }
        self.last_fit_t = [None; 2];
        self.camera_lateral = None;
        self.last_targets = [None; 2];
        self.settled = [0; 2];
    }

    /// Current projected configuration (the last result's qpos).
    pub fn qpos(&self) -> &[f64] {
        &self.q
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.core.model
    }

    /// Native marker positions (shoulder = shoulder_elv anchor, elbow = mean epicondyle markers,
    /// wrist = flexion anchor) per side at `qpos` (use a projected configuration).
    pub fn markers(&self, qpos: &[f64]) -> Result<ArmMarkers, String> {
        self.core.markers_full(qpos)
    }

    /// Effective (lo, hi) box of each fitted coordinate, per side, order
    /// [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion].
    pub fn master_bounds(&self) -> [[(f64, f64); 4]; 2] {
        let b =
            |s: usize| std::array::from_fn(|k| (self.core.arms[s].lo[k], self.core.arms[s].hi[k]));
        [b(0), b(1)]
    }

    /// Joint ids of the fitted coordinates per side (same order as [`Self::master_bounds`]).
    pub fn fitted_joints(&self) -> [[usize; 4]; 2] {
        [self.core.arms[0].joints, self.core.arms[1].joints]
    }

    /// Fit one observation. Errors (non-finite timestamp) leave the state untouched. Missing or
    /// low-confidence data never fails: the affected arm holds independently.
    /// Partial views retain the learned torso frame, or use a frontal prior at startup.
    pub fn update(&mut self, obs: &Observation) -> Result<FitResult, String> {
        let t = obs.timestamp_seconds;
        if !t.is_finite() {
            return Err("observation timestamp is not finite".into());
        }
        // Best confident, finite landmark per name.
        let get = |name: &str| -> Option<(Vec3, f64)> {
            obs.landmarks
                .iter()
                .filter(|l| l.name == name && finite3(&l.position) && l.confidence.is_finite())
                .map(|l| (l.position, l.confidence.clamp(0.0, 1.0)))
                .filter(|(_, c)| *c >= SCORE_FLOOR)
                .max_by(|a, b| a.1.total_cmp(&b.1))
        };
        let sh = [get("left_shoulder"), get("right_shoulder")];
        let held = |me: &Self, why: &str| FitResult {
            qpos: me.q.clone(),
            rmse_m: None,
            supported_arms: [false, false],
            status: format!("held: {why}"),
            iterations: 0,
        };
        // A missing contralateral shoulder must not stop the visible arm. Learn
        // the torso orientation only from two confident shoulders, and retain it
        // during partial views. A fresh one-arm session uses an explicit frontal prior.
        if let (Some((ls, lc)), Some((rs, rc))) = (sh[0], sh[1]) {
            if lc.min(rc) >= 0.6 {
                if let Some(lateral) = unit(&sub3(&ls, &rs), MIN_SEGMENT_M) {
                    if torso_basis(&lateral, &[0.0, -1.0, 0.0]).is_some() {
                        self.camera_lateral = Some(match self.camera_lateral {
                            Some(old)
                                if dot3(&old, &lateral) < 0.5
                                    || dot3(&old, &lateral) > 0.9999875 =>
                            {
                                old
                            }
                            Some(old) => {
                                unit(&add3(&scl3(&old, 0.85), &scl3(&lateral, 0.15)), 0.01)
                                    .unwrap_or(old)
                            }
                            None => lateral,
                        });
                    }
                }
            }
        }
        let Some(b_cam) = torso_basis(
            &self.camera_lateral.unwrap_or([1.0, 0.0, 0.0]),
            &[0.0, -1.0, 0.0],
        ) else {
            return Ok(held(self, "torso basis unavailable"));
        };
        let rot = mul_abt(&self.core.b_model, &b_cam);

        let mut supported = [false; 2];
        let mut iterations = 0;
        let mut err2 = Vec::with_capacity(4);
        let mut side_status = [String::new(), String::new()];
        for side in 0..2 {
            let arm = &self.core.arms[side];
            let Some((s_pos, s_conf)) = sh[side] else {
                side_status[side] = "held(shoulder missing/low confidence)".into();
                continue;
            };
            let ao = match get(&format!("{}_elbow", SIDE_NAMES[side])) {
                None => ArmObs::Held("elbow missing/low confidence"),
                Some((e_pos, e_conf)) => match unit(&sub3(&e_pos, &s_pos), MIN_SEGMENT_M) {
                    None => ArmObs::Held("degenerate upper arm"),
                    Some(u_dir) => {
                        let tgt_elbow = scl3(&mat_vec(&rot, &u_dir), arm.upper_len);
                        let w_elbow = s_conf.min(e_conf);
                        let wrist = get(&format!("{}_wrist", SIDE_NAMES[side])).and_then(
                            |(w_pos, w_conf)| {
                                unit(&sub3(&w_pos, &e_pos), MIN_SEGMENT_M).map(|f_dir| {
                                    (
                                        add3(
                                            &tgt_elbow,
                                            &scl3(&mat_vec(&rot, &f_dir), arm.fore_len),
                                        ),
                                        w_elbow.min(w_conf),
                                    )
                                })
                            },
                        );
                        ArmObs::Fit {
                            tgt_elbow,
                            tgt_wrist: wrist.map(|w| w.0),
                            w_elbow,
                            w_wrist: wrist.map_or(0.0, |w| w.1),
                        }
                    }
                },
            };
            let ArmObs::Fit {
                tgt_elbow,
                tgt_wrist,
                w_elbow,
                w_wrist,
            } = ao
            else {
                if let ArmObs::Held(why) = ao {
                    side_status[side] = format!("held({why})");
                }
                continue;
            };
            let same_target = self.last_targets[side].is_some_and(|(e, w)| {
                norm3(&sub3(&e, &tgt_elbow)) < 0.001
                    && match (w, tgt_wrist) {
                        (Some(a), Some(b)) => norm3(&sub3(&a, &b)) < 0.001,
                        (None, None) => true,
                        _ => false,
                    }
            });
            if !same_target {
                self.last_targets[side] = Some((tgt_elbow, tgt_wrist));
                self.settled[side] = 0;
            }
            if same_target && self.settled[side] >= 3 {
                // Constant evidence must not repeatedly reseed or creep along an
                // unobservable shoulder rotation. Resume as soon as targets move.
                supported[side] = true;
                self.last_fit_t[side] = Some(t);
                side_status[side] = "still".into();
                let mut qs = self.q.clone();
                let off = self
                    .core
                    .offsets(side, &self.x[side], &mut qs, &mut self.fk);
                err2.push(norm3(&sub3(&off.0, &tgt_elbow)).powi(2));
                if let Some(w) = tgt_wrist {
                    err2.push(norm3(&sub3(&off.1, &w)).powi(2));
                }
                continue;
            }
            let xp = self.x[side];
            let mut lam_t = [CONTINUITY; 4];
            lam_t[PLANE] *= 1.0 + AX_GAIN * (-(xp[ELEV] / 0.25).powi(2)).exp();
            lam_t[ROT] *= 1.0 + AX_GAIN * (-(xp[FLEX] / 0.3).powi(2)).exp();
            let wrist_seen = tgt_wrist.is_some();
            // Without a forearm direction, humeral rotation and elbow flexion are unobserved:
            // they are held exactly (not fitted, not pulled to neutral).
            let free = [true, true, wrist_seen, wrist_seen];
            let mut prob = ArmProblem {
                tgt_elbow,
                tgt_wrist,
                w_elbow,
                w_wrist,
                x_prev: xp,
                x_neutral: arm.qadr.map(|a| self.core.standing[a]),
                lam_t,
                free,
            };
            let mut qs = self.q.clone();
            let (mut x, _, it) =
                self.core
                    .solve(side, &prob, xp, WARM_ITERS, &mut qs, &mut self.fk);
            iterations += it;
            let mut tags: Vec<&str> = Vec::new();
            let off = self.core.offsets(side, &x, &mut qs, &mut self.fk);
            let mcost = Core::marker_cost(&prob, &off);
            if self.marker_rms(&prob, &off) > RESEED_RMS_M {
                // Coarse global seed on marker cost, then a short LM with a weak temporal prior.
                let mut best = (f64::INFINITY, x);
                for &pl in &[-1.0, 0.0, 1.0, 2.0] {
                    for &el in &[0.3, 1.0, 1.7, 2.4, 3.0] {
                        for &ro in &[xp[ROT], -0.8, 0.8] {
                            for &fl in &[xp[FLEX], 0.4, 1.2, 2.0] {
                                let mut s = [pl, el, ro, fl];
                                for k in 0..4 {
                                    s[k] = if free[k] {
                                        s[k].clamp(arm.lo[k], arm.hi[k])
                                    } else {
                                        xp[k]
                                    };
                                }
                                let o = self.core.offsets(side, &s, &mut qs, &mut self.fk);
                                let c = Core::marker_cost(&prob, &o);
                                if c < best.0 && self.core.arm_feasible(side, &qs) {
                                    best = (c, s);
                                }
                            }
                        }
                    }
                }
                prob.lam_t = prob.lam_t.map(|l| 0.1 * l);
                let (xr, _, it2) =
                    self.core
                        .solve(side, &prob, best.1, RESEED_ITERS, &mut qs, &mut self.fk);
                iterations += it2;
                let or = self.core.offsets(side, &xr, &mut qs, &mut self.fk);
                if Core::marker_cost(&prob, &or) < 0.7 * mcost && self.core.arm_feasible(side, &qs)
                {
                    x = xr;
                    tags.push("reseeded");
                }
            }
            // Explicit rate limit (box is convex, so the limited state stays feasible).
            let dt = match self.last_fit_t[side] {
                Some(t0) if t > t0 => (t - t0).clamp(DT_MIN, DT_MAX),
                Some(_) => DT_NOMINAL,
                // Initialize from the first usable pose; later reacquisitions stay rate-limited.
                None => 0.5,
            };
            let max_step = MAX_RATE_RAD_S * dt;
            for k in 0..4 {
                let d = x[k] - xp[k];
                if d.abs() > max_step {
                    x[k] = xp[k] + max_step.copysign(d);
                    if !tags.contains(&"rate-limited") {
                        tags.push("rate-limited");
                    }
                }
            }
            let off = self.core.offsets(side, &x, &mut qs, &mut self.fk);
            if !self.core.arm_feasible(side, &qs) || !finite3(&off.0) || !finite3(&off.1) {
                side_status[side] = "held(solver produced invalid state)".into();
                continue;
            }
            err2.push(norm3(&sub3(&off.0, &tgt_elbow)).powi(2));
            if let Some(tw) = tgt_wrist {
                err2.push(norm3(&sub3(&off.1, &tw)).powi(2));
            } else {
                tags.push("elbow-only");
            }
            if (0..4).all(|k| (x[k] - xp[k]).abs() < 0.001) {
                self.settled[side] = self.settled[side].saturating_add(1);
            } else {
                self.settled[side] = 0;
            }
            self.x[side] = x;
            self.last_fit_t[side] = Some(t);
            supported[side] = true;
            side_status[side] = if tags.is_empty() {
                "fit".into()
            } else {
                format!("fit({})", tags.join(","))
            };
        }

        // Rebuild the full configuration from the standing prior + fitted masters, exactly.
        let mut q = self.core.standing.clone();
        for side in 0..2 {
            for k in 0..4 {
                q[self.core.arms[side].qadr[k]] = self.x[side][k];
            }
        }
        self.core.model.project(&mut q)?;
        if let Some(&(a, _, _)) = self
            .core
            .limited
            .iter()
            .find(|&&(a, lo, hi)| !(lo..=hi).contains(&q[a]))
        {
            return Err(format!(
                "internal: fitted state violates range at qpos[{a}]"
            ));
        }
        self.q = q;
        let rmse_m =
            (!err2.is_empty()).then(|| (err2.iter().sum::<f64>() / err2.len() as f64).sqrt());
        Ok(FitResult {
            qpos: self.q.clone(),
            rmse_m,
            supported_arms: supported,
            status: format!("L:{} R:{}", side_status[0], side_status[1]),
            iterations,
        })
    }

    fn marker_rms(&self, p: &ArmProblem, off: &(Vec3, Vec3)) -> f64 {
        let mut s = norm3(&sub3(&off.0, &p.tgt_elbow)).powi(2);
        let mut n = 1.0;
        if let Some(tw) = p.tgt_wrist {
            s += norm3(&sub3(&off.1, &tw)).powi(2);
            n += 1.0;
        }
        (s / n).sqrt()
    }
}
