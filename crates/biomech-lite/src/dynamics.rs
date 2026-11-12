//! Reduced fixed-torso arm dynamics (eight coordinates, everything else held/prescribed).
//!
//! The eight dynamic coordinates `x` are `model.dynamic_joints` (right then left
//! `elv_angle, shoulder_elv, shoulder_rot, elbow_flexion`). Every other independent coordinate
//! and the free root stay at the values stored in `State::qpos`; all native polynomial
//! couplings are re-projected exactly, so dependent joints follow their masters.
//!
//! Algorithm (docs/rust-biomech-contract.md, "Settled dynamics algorithm and API"):
//! * analytic Jacobians of body COM and WORLD angular velocity from one FK (joint axes/anchors of
//!   the body and its ancestors times the analytic coupling chain matrix `C`); finite differences
//!   (inner step [`H_JAC`]) remain only for the tendon-length Jacobian;
//! * `M = sum_b(m Jv^T Jv + Jw^T I_world Jw) + C^T diag(armature) C` with `C` the scalar-dof
//!   chain-rule matrix (so it equals native `C^T M_full C`);
//! * `gravity` = required holding torque `-sum_b Jv^T m g` (native `qfrc_bias` at rest);
//! * Coriolis from centered differences (outer step [`H_MASS`]) of the reduced mass matrix with
//!   Christoffel symbols of the first kind;
//! * muscle and passive tendon forces through the tendon-length Jacobian.
//!
//! Numerical limitations: finite-difference Jacobians (tendon Jacobian, mass derivatives; worse
//! right at a wrapping branch switch); hard-bound handling is an approximation of MuJoCo's soft
//! limits; there are no contacts, ground reaction, balance control or activation inference.

use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3};

use crate::activation_dot;
use crate::kinematics::quat_to_mat;
use crate::model::{coupling_poly, Model};

/// Inner finite-difference step for Jacobians (radians / metres of the independent coordinate).
pub const H_JAC: f64 = 1e-6;
/// Outer finite-difference step for mass-matrix derivatives.
pub const H_MASS: f64 = 1e-4;
/// Largest accepted integration step in seconds.
pub const MAX_DT: f64 = 0.002;
/// Number of dynamic coordinates.
pub const NX: usize = 8;
/// Tolerated joint-range violation (radians / metres).
pub const BOUND_TOL: f64 = 1e-9;

type Mat38 = SMatrix<f64, 3, 8>;
type Mat8 = SMatrix<f64, 8, 8>;

/// Dynamic state: full `qpos` (nq), velocity of the eight dynamic coordinates, muscle activation.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub qpos: Vec<f64>,
    pub velocity: Vec<f64>,
    pub activation: Vec<f64>,
}

impl State {
    /// Rest state at `qpos` (projected), zero velocity, zero activation.
    pub fn at_rest(model: &Model, qpos: &[f64]) -> Result<State, String> {
        let mut q = qpos.to_vec();
        model.project(&mut q)?;
        Ok(State { qpos: q, velocity: vec![0.0; NX], activation: vec![0.0; model.muscles.len()] })
    }

    /// Rest state at the model's standing keyframe.
    pub fn standing(model: &Model) -> Result<State, String> {
        State::at_rest(model, &model.standing_qpos)
    }

    /// Current values of the eight dynamic coordinates.
    pub fn coordinates(&self, model: &Model) -> Vec<f64> {
        dyn_qadr(model).iter().map(|&a| self.qpos[a]).collect()
    }
}

fn dyn_qadr(model: &Model) -> Vec<usize> {
    model.dynamic_joints.iter().map(|&j| model.joints[j].qposadr).collect()
}

fn check_q(model: &Model, q: &[f64]) -> Result<(), String> {
    if q.len() != model.nq {
        return Err(format!("q length {} != nq {}", q.len(), model.nq));
    }
    if model.dynamic_joints.len() != NX {
        return Err("model must have eight dynamic joints".into());
    }
    if q.iter().any(|x| !x.is_finite()) {
        return Err("non-finite q".into());
    }
    Ok(())
}

fn check_v(v: &[f64]) -> Result<(), String> {
    if v.len() != NX {
        return Err(format!("velocity length {} != {NX}", v.len()));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err("non-finite velocity".into());
    }
    Ok(())
}

/// `q` with dynamic coordinate `k` shifted by `delta`, couplings re-projected.
fn perturbed(model: &Model, q: &[f64], k: usize, delta: f64) -> Result<Vec<f64>, String> {
    let mut p = q.to_vec();
    p[model.joints[model.dynamic_joints[k]].qposadr] += delta;
    model.project(&mut p)?;
    Ok(p)
}

/// Scalar-dof chain-rule matrix `C` (nv x 8), analytic: dynamic masters are unit rows, locked and
/// non-dynamic independent rows (and the free root) are zero, each dependent row is the quartic
/// slope at its master times the master's row (couplings are in dependency order).
/// `q` must satisfy the couplings.
fn chain_matrix(model: &Model, q: &[f64]) -> Result<DMatrix<f64>, String> {
    let mut c = DMatrix::zeros(model.nv, NX);
    for (k, &j) in model.dynamic_joints.iter().enumerate() {
        c[(model.joints[j].dofadr, k)] = 1.0;
    }
    for cp in &model.couplings {
        let dd = model.joints[cp.dependent].dofadr;
        match cp.master {
            None => {
                for k in 0..NX {
                    c[(dd, k)] = 0.0;
                }
            }
            Some(m) => {
                let jm = &model.joints[m];
                let slope = coupling_poly(&cp.coef, q[jm.qposadr] - model.qpos0[jm.qposadr]).1;
                for k in 0..NX {
                    c[(dd, k)] = slope * c[(jm.dofadr, k)];
                }
            }
        }
    }
    Ok(c)
}

struct BodyJac {
    mass: f64,
    jv: Mat38,
    jw: Mat38,
    inertia_world: Matrix3<f64>,
}

/// Analytic reduced body Jacobians from one FK: each scalar joint on the body and its ancestors
/// contributes `axis x (COM - anchor)` (hinge, plus angular `axis`) or `axis` (slide) times its
/// row of `C`. The free root is prescribed and contributes nothing.
fn jacobians(model: &Model, q: &[f64]) -> Result<Vec<BodyJac>, String> {
    let k0 = model.kinematics(q)?;
    let c = chain_matrix(model, q)?;
    let mut out = Vec::new();
    for (b, body) in model.bodies.iter().enumerate().skip(1) {
        if body.mass <= crate::MJ_MINVAL {
            continue;
        }
        let com = Vector3::from(k0.inertia_pos[b]);
        let mut jv = Mat38::zeros();
        let mut jw = Mat38::zeros();
        let mut cur = b;
        while cur != 0 {
            for &ji in &model.bodies[cur].joints {
                let j = &model.joints[ji];
                if !j.is_scalar() || (0..NX).all(|k| c[(j.dofadr, k)] == 0.0) {
                    continue;
                }
                let axis = Vector3::from(k0.joint_axis[ji]);
                let (lin, ang) = if j.kind == crate::JNT_HINGE {
                    (axis.cross(&(com - Vector3::from(k0.joint_anchor[ji]))), axis)
                } else {
                    (axis, Vector3::zeros())
                };
                for k in 0..NX {
                    let w = c[(j.dofadr, k)];
                    for r in 0..3 {
                        jv[(r, k)] += lin[r] * w;
                        jw[(r, k)] += ang[r] * w;
                    }
                }
            }
            cur = model.bodies[cur].parent;
        }
        let m = quat_to_mat(&k0.inertia_quat[b]);
        let rot = Matrix3::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
        let diag = Matrix3::from_diagonal(&Vector3::new(body.inertia[0], body.inertia[1], body.inertia[2]));
        out.push(BodyJac { mass: body.mass, jv, jw, inertia_world: rot * diag * rot.transpose() });
    }
    Ok(out)
}

fn mass8(model: &Model, q: &[f64]) -> Result<Mat8, String> {
    let mut m = Mat8::zeros();
    for b in jacobians(model, q)? {
        m += b.mass * b.jv.transpose() * b.jv + b.jw.transpose() * b.inertia_world * b.jw;
    }
    let c = chain_matrix(model, q)?;
    for j in model.joints.iter().filter(|j| j.is_scalar() && j.armature != 0.0) {
        for a in 0..NX {
            for bb in 0..NX {
                m[(a, bb)] += j.armature * c[(j.dofadr, a)] * c[(j.dofadr, bb)];
            }
        }
    }
    Ok(m)
}

fn to_dmatrix(m: &Mat8) -> DMatrix<f64> {
    DMatrix::from_fn(NX, NX, |i, j| m[(i, j)])
}

/// Reduced 8x8 mass matrix at `q` (includes native armature of every scalar joint).
pub fn mass_matrix(model: &Model, q: &[f64]) -> Result<DMatrix<f64>, String> {
    check_q(model, q)?;
    Ok(to_dmatrix(&mass8(model, q)?))
}

/// Required holding torque against gravity, `-sum_b Jv^T m g` (native `C^T qfrc_bias` at rest).
pub fn gravity(model: &Model, q: &[f64]) -> Result<Vec<f64>, String> {
    check_q(model, q)?;
    let g = Vector3::new(model.gravity[0], model.gravity[1], model.gravity[2]);
    let mut t = nalgebra::SVector::<f64, 8>::zeros();
    for b in jacobians(model, q)? {
        t -= b.jv.transpose() * (b.mass * g);
    }
    Ok(t.iter().copied().collect())
}

/// Bias demand `gravity + Coriolis/centrifugal` for velocity `v` (the quantity a native
/// `C^T (qfrc_bias + M Cdot xdot)` fixture holds).
pub fn bias(model: &Model, q: &[f64], v: &[f64]) -> Result<Vec<f64>, String> {
    check_q(model, q)?;
    check_v(v)?;
    let mut out = gravity(model, q)?;
    if v.iter().all(|x| *x == 0.0) {
        return Ok(out);
    }
    // dm[k] = dM/dx_k
    let mut dm: Vec<Mat8> = Vec::with_capacity(NX);
    for k in 0..NX {
        let mp = mass8(model, &perturbed(model, q, k, H_MASS)?)?;
        let mm = mass8(model, &perturbed(model, q, k, -H_MASS)?)?;
        dm.push((mp - mm) / (2.0 * H_MASS));
    }
    for i in 0..NX {
        let mut c = 0.0;
        for j in 0..NX {
            for k in 0..NX {
                c += (dm[k][(i, j)] - 0.5 * dm[i][(j, k)]) * v[j] * v[k];
            }
        }
        out[i] += c;
    }
    Ok(out)
}

/// Tendon lengths, tendon-length Jacobian (ntendon x 8) and length velocities for velocity `v`.
fn tendon_state(model: &Model, q: &[f64], v: &[f64]) -> Result<(Vec<f64>, DMatrix<f64>, Vec<f64>), String> {
    let len = model.tendon_lengths(&model.kinematics(q)?)?;
    let mut jt = DMatrix::zeros(model.tendons.len(), NX);
    for k in 0..NX {
        let lp = model.tendon_lengths(&model.kinematics(&perturbed(model, q, k, H_JAC)?)?)?;
        let lm = model.tendon_lengths(&model.kinematics(&perturbed(model, q, k, -H_JAC)?)?)?;
        for t in 0..model.tendons.len() {
            jt[(t, k)] = (lp[t] - lm[t]) / (2.0 * H_JAC);
        }
    }
    let vel = (&jt * DVector::from_column_slice(v)).iter().copied().collect();
    Ok((len, jt, vel))
}

/// Generalized torque of the muscle actuators (`-dL/dx * positive tension`), `C^T qfrc_actuator`.
pub fn muscle_torque(model: &Model, q: &[f64], v: &[f64], a: &[f64]) -> Result<Vec<f64>, String> {
    check_q(model, q)?;
    check_v(v)?;
    if a.len() != model.muscles.len() {
        return Err(format!("activation length {} != {} muscles", a.len(), model.muscles.len()));
    }
    let (len, jt, vel) = tendon_state(model, q, v)?;
    muscle_torque_from(model, &len, &jt, &vel, a)
}

fn muscle_torque_from(model: &Model, len: &[f64], jt: &DMatrix<f64>, vel: &[f64], a: &[f64])
    -> Result<Vec<f64>, String> {
    let tension = model.muscle_forces(len, vel, a)?;
    let mut t = vec![0.0; NX];
    for (m, mu) in model.muscles.iter().enumerate() {
        for k in 0..NX {
            t[k] -= jt[(mu.tendon, k)] * tension[m];
        }
    }
    Ok(t)
}

/// Passive torque: tendon dead-band springs/damping (e.g. the glenohumeral ligament routes) and
/// scalar joint spring/damping projected through the couplings.
pub fn passive_torque(model: &Model, q: &[f64], v: &[f64]) -> Result<Vec<f64>, String> {
    check_q(model, q)?;
    check_v(v)?;
    let (len, jt, vel) = tendon_state(model, q, v)?;
    passive_from(model, q, v, &len, &jt, &vel)
}

fn passive_from(model: &Model, q: &[f64], v: &[f64], len: &[f64], jt: &DMatrix<f64>, vel: &[f64])
    -> Result<Vec<f64>, String> {
    let mut t = vec![0.0; NX];
    for (i, tn) in model.tendons.iter().enumerate() {
        if tn.stiffness == 0.0 && tn.damping == 0.0 {
            continue;
        }
        let l = len[i];
        let tension = tn.stiffness * (l - l.clamp(tn.springlength[0], tn.springlength[1]))
            + tn.damping * vel[i];
        for k in 0..NX {
            t[k] -= jt[(i, k)] * tension;
        }
    }
    if model.joints.iter().any(|j| j.is_scalar() && (j.stiffness != 0.0 || j.damping != 0.0)) {
        let c = chain_matrix(model, q)?;
        let cv = &c * DVector::from_column_slice(v);
        for j in model.joints.iter().filter(|j| j.is_scalar()) {
            let f = -j.stiffness * (q[j.qposadr] - j.springref) - j.damping * cv[j.dofadr];
            if f != 0.0 {
                for k in 0..NX {
                    t[k] += c[(j.dofadr, k)] * f;
                }
            }
        }
    }
    Ok(t)
}

fn violation(model: &Model, q: &[f64]) -> Vec<f64> {
    model.joints.iter().map(|j| {
        if j.is_scalar() && j.limited {
            let x = q[j.qposadr];
            (j.range[0] - x).max(x - j.range[1]).max(0.0)
        } else {
            0.0
        }
    }).collect()
}

/// Independent (dynamic) coordinate index feeding scalar joint `j`, following the coupling chain.
fn root_master(model: &Model, mut j: usize) -> usize {
    for _ in 0..model.couplings.len() + 1 {
        match model.couplings.iter().find(|c| c.dependent == j) {
            Some(c) => match c.master {
                Some(m) => j = m,
                None => return j,
            },
            None => return j,
        }
    }
    j
}

/// One semi-implicit Euler step with forward-Euler activation. `excitation` is per muscle
/// (clamped to each muscle's `ctrlrange`). `0 < dt <= MAX_DT`. On error `state` is unchanged.
///
/// Joint bounds are hard. An initial state outside any native joint range is rejected. A master
/// reaching its range is clamped with zero velocity; if a dependent joint would leave its range,
/// only the masters feeding it have their displacement bisected back to the boundary and lose
/// velocity, while the other coordinates keep `q += dt*v`. This is a hard-limit approximation of
/// MuJoCo's soft limits.
pub fn step(model: &Model, state: &mut State, excitation: &[f64], dt: f64) -> Result<(), String> {
    check_q(model, &state.qpos)?;
    check_v(&state.velocity)?;
    if state.activation.len() != model.muscles.len() || excitation.len() != model.muscles.len() {
        return Err("activation/excitation length must equal the number of muscles".into());
    }
    if !dt.is_finite() || dt <= 0.0 || dt > MAX_DT {
        return Err(format!("dt must satisfy 0 < dt <= {MAX_DT}, got {dt}"));
    }
    if state.activation.iter().chain(excitation).any(|x| !x.is_finite()) {
        return Err("non-finite activation/excitation".into());
    }
    if let Some(j) = violation(model, &state.qpos).iter().position(|v| *v > BOUND_TOL) {
        return Err(format!("initial state violates the range of joint {}", model.joints[j].name));
    }
    let q = &state.qpos;
    let v = &state.velocity;
    let (len, jt, vel) = tendon_state(model, q, v)?;
    let tau_m = muscle_torque_from(model, &len, &jt, &vel, &state.activation)?;
    let tau_p = passive_from(model, q, v, &len, &jt, &vel)?;
    let bias_t = bias(model, q, v)?;
    let rhs = DVector::from_fn(NX, |i, _| tau_m[i] + tau_p[i] - bias_t[i]);
    let qdd = mass_matrix(model, q)?
        .cholesky().ok_or("mass matrix is not positive definite")?
        .solve(&rhs);

    let mut next = state.clone();
    for (m, mu) in model.muscles.iter().enumerate() {
        let u = excitation[m].clamp(mu.ctrlrange[0], mu.ctrlrange[1]);
        let d = activation_dot(u, state.activation[m], mu.dynprm);
        next.activation[m] = (state.activation[m] + dt * d).clamp(0.0, 1.0);
    }
    let adr = dyn_qadr(model);
    let mut v_new: Vec<f64> = (0..NX).map(|i| v[i] + dt * qdd[i]).collect();
    let x0: Vec<f64> = adr.iter().map(|&a| q[a]).collect();
    let mut disp: Vec<f64> = (0..NX).map(|i| dt * v_new[i]).collect();
    // master bounds
    for i in 0..NX {
        let jt_ = &model.joints[model.dynamic_joints[i]];
        if jt_.limited {
            let x = (x0[i] + disp[i]).clamp(jt_.range[0], jt_.range[1]);
            if x != x0[i] + disp[i] {
                disp[i] = x - x0[i];
                v_new[i] = 0.0;
            }
        }
    }
    // dependent bounds: only the masters feeding a violated joint are backtracked (bisection) and
    // lose velocity; unaffected coordinates keep q += dt*v.
    let propose = |scale: &[f64]| -> Result<Vec<f64>, String> {
        let mut p = q.clone();
        for i in 0..NX {
            p[adr[i]] = x0[i] + scale[i] * disp[i];
        }
        model.project(&mut p)?;
        Ok(p)
    };
    let mut scale = vec![1.0; NX];
    let mut proposal = propose(&scale)?;
    let bad = |p: &[f64]| -> Vec<usize> {
        violation(model, p).iter().enumerate().filter(|(_, v)| **v > BOUND_TOL).map(|(j, _)| j).collect()
    };
    let first_bad = bad(&proposal);
    if !first_bad.is_empty() {
        let mut affected = vec![false; NX];
        for j in first_bad {
            let root = root_master(model, j);
            if let Some(i) = model.dynamic_joints.iter().position(|&d| d == root) {
                affected[i] = true;
            }
        }
        if !affected.iter().any(|a| *a) {
            return Err("joint bound violated by a non-dynamic coordinate".into());
        }
        for i in 0..NX {
            if affected[i] {
                v_new[i] = 0.0;
            }
        }
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            for i in 0..NX {
                scale[i] = if affected[i] { mid } else { 1.0 };
            }
            if bad(&propose(&scale)?).is_empty() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        for i in 0..NX {
            scale[i] = if affected[i] { lo } else { 1.0 };
        }
        proposal = propose(&scale)?;
        if !bad(&proposal).is_empty() {
            return Err("cannot resolve joint bounds for this step".into());
        }
    }
    next.qpos = proposal;
    next.velocity = v_new;
    if next.qpos.iter().chain(&next.velocity).chain(&next.activation).any(|x| !x.is_finite()) {
        return Err("non-finite state after step".into());
    }
    *state = next;
    Ok(())
}
