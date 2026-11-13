//! Reduced inverse dynamics and bounded muscle recruitment for the eight fixed-torso arm
//! coordinates (`docs/muscle-estimation-contract.md`).
//!
//! The tracked skeleton prescribes `q`, velocity and acceleration; this module only ESTIMATES the
//! muscle activation that would produce the demanded torque. Nothing feeds back into motion.
//! Assumptions: generic reference anatomy, unloaded hands, no external or contact forces.
//! Recruitment is non-unique (more muscles than coordinates); the effort and continuity
//! regularizers pick one solution. Any demand the muscles cannot meet stays visible in
//! `residual_torque_nm` - there is no hidden reserve actuator.
//!
//! Sign convention (same as `dynamics::step`): `M qdd + bias = muscle + passive`, so
//! `required = M a + bias`, and a muscle with tension `T >= 0` contributes `-dL/dx * T`.

use std::sync::Arc;

use nalgebra::{DMatrix, DVector};

use crate::dynamics::{self, NX};
use crate::model::Model;

const LAMBDA_EFFORT: f64 = 1e-4;
const LAMBDA_CONTINUITY: f64 = 1e-5;
const MAX_SWEEPS: usize = 250;
const CONVERGENCE: f64 = 1e-5;
const SUPPORT_EPS: f64 = 1e-9;
const RANGE_TOL: f64 = 1e-6;
const NEWTON_ITERS: usize = 80;
const HOMOTOPY_STAGES: usize = 6;
const DUAL_TOL: f64 = 1e-8;
const KKT_TOL: f64 = 1e-6;
const LAMBDA: f64 = LAMBDA_EFFORT + LAMBDA_CONTINUITY;

#[derive(Clone, Debug)]
pub struct Recruitment {
    /// Model muscle order, bounded to [0, 1].
    pub activation: Vec<f64>,
    /// Passive + active positive tension, muscle order.
    pub tension_n: Vec<f64>,
    /// Muscle has a nonzero moment about at least one observed dynamic coordinate.
    pub supported_muscles: Vec<bool>,
    /// `M*a + bias`, coordinate order, unloaded hands.
    pub required_torque_nm: Vec<f64>,
    /// Passive joint / non-actuator tendon torque (not passive muscle).
    pub passive_torque_nm: Vec<f64>,
    /// Total passive + active muscle torque.
    pub muscle_torque_nm: Vec<f64>,
    /// `required - passive - muscle`.
    pub residual_torque_nm: Vec<f64>,
    pub observed_coordinates: [bool; 8],
    /// Norm over observed rows only.
    pub residual_norm_nm: f64,
    /// `residual_norm / max(1, norm(required - passive))` over observed rows.
    pub relative_residual: f64,
    /// Optimizer convergence, not physical accuracy.
    pub converged: bool,
    pub iterations: usize,
}

pub struct Recruiter {
    model: Arc<Model>,
    /// Last supported activation (warm start and continuity target).
    last: Vec<f64>,
}

impl Recruiter {
    pub fn new(model: Arc<Model>) -> Result<Self, String> {
        model.validate()?;
        if model.dynamic_joints.len() != NX {
            return Err(format!("model must have {NX} dynamic joints, has {}", model.dynamic_joints.len()));
        }
        let n = model.muscles.len();
        Ok(Self { model, last: vec![0.0; n] })
    }

    /// Forget the warm start (call on discontinuities such as a new generation).
    pub fn reset(&mut self) {
        self.last.iter_mut().for_each(|a| *a = 0.0);
    }

    pub fn solve(
        &mut self,
        q: &[f64],
        velocity: &[f64],
        acceleration: &[f64],
        observed: [bool; 8],
    ) -> Result<Recruitment, String> {
        let model = &*self.model;
        // Validate everything before touching state.
        dynamics::check_q(model, q)?;
        dynamics::check_v(velocity)?;
        if acceleration.len() != NX {
            return Err(format!("acceleration length {} != {NX}", acceleration.len()));
        }
        if acceleration.iter().any(|x| !x.is_finite()) {
            return Err("non-finite acceleration".into());
        }
        if !observed.iter().any(|o| *o) {
            return Err("at least one observed coordinate is required".into());
        }
        let mut qp = q.to_vec();
        model.project(&mut qp)?;
        if let Some(j) = dynamics::violation(model, &qp).iter().position(|v| *v > RANGE_TOL) {
            return Err(format!("q violates the range of joint {}", model.joints[j].name));
        }
        // Unobserved coordinates carry no motion into cross terms.
        let v: Vec<f64> = (0..NX).map(|k| if observed[k] { velocity[k] } else { 0.0 }).collect();
        let acc: Vec<f64> = (0..NX).map(|k| if observed[k] { acceleration[k] } else { 0.0 }).collect();

        let nm = model.muscles.len();
        // Each primitive once per solve.
        let (len, jt, vel) = dynamics::tendon_state(model, &qp, &v)?;
        let passive = dynamics::passive_from(model, &qp, &v, &len, &jt, &vel)?;
        let mass = dynamics::mass_matrix(model, &qp)?;
        let bias = dynamics::bias(model, &qp, &v)?;
        let required: Vec<f64> = (0..NX)
            .map(|i| (0..NX).map(|j| mass[(i, j)] * acc[j]).sum::<f64>() + bias[i])
            .collect();
        let f0 = model.muscle_forces(&len, &vel, &vec![0.0; nm])?;
        let f1 = model.muscle_forces(&len, &vel, &vec![1.0; nm])?;

        // A[k,m] = -dL/dx_k (F1-F0); passive muscle torque P[k] = -sum J F0.
        let mut a_mat = DMatrix::<f64>::zeros(NX, nm);
        let mut p_muscle = vec![0.0; NX];
        for (m, mu) in model.muscles.iter().enumerate() {
            for k in 0..NX {
                let j = jt[(mu.tendon, k)];
                a_mat[(k, m)] = -j * (f1[m] - f0[m]);
                p_muscle[k] -= j * f0[m];
            }
        }
        let rows: Vec<usize> = (0..NX).filter(|k| observed[*k]).collect();
        let supported: Vec<bool> = (0..nm)
            .map(|m| rows.iter().any(|&k| a_mat[(k, m)].abs() > SUPPORT_EPS))
            .collect();

        // Row-scaled least squares over observed rows only.
        let scale: Vec<f64> = (0..NX).map(|k| 1.0 / (required[k] - passive[k]).abs().max(1.0)).collect();
        let target: Vec<f64> = (0..NX).map(|k| (required[k] - passive[k] - p_muscle[k]) * scale[k]).collect();
        let a_s = |k: usize, m: usize| a_mat[(k, m)] * scale[k];

        let mut act: Vec<f64> = (0..nm)
            .map(|m| if supported[m] { self.last[m].clamp(0.0, 1.0) } else { 0.0 })
            .collect();
        let prev = act.clone();
        // Incremental residual e = target - A_s a over observed rows.
        let mut e = target.clone();
        for &k in &rows {
            for m in 0..nm {
                e[k] -= a_s(k, m) * act[m];
            }
        }
        let denom: Vec<f64> = (0..nm)
            .map(|m| rows.iter().map(|&k| a_s(k, m).powi(2)).sum::<f64>() + LAMBDA_EFFORT + LAMBDA_CONTINUITY)
            .collect();
        let mut iterations = 0;
        let mut converged = false;
        // Primary solver: dual semismooth Newton over the (<= 8) observed rows. It solves the
        // same convex objective exactly and shares redundant muscles evenly, unlike
        // coordinate descent which is order-biased on duplicated columns.
        let sup: Vec<usize> = (0..nm).filter(|&m| supported[m]).collect();
        if !sup.is_empty() && !rows.is_empty() {
            let a_dense = DMatrix::from_fn(rows.len(), sup.len(), |r, c| a_s(rows[r], sup[c]));
            let b_vec = DVector::from_fn(rows.len(), |r, _| target[rows[r]]);
            let c_vec = DVector::from_fn(sup.len(), |c, _| LAMBDA_CONTINUITY * prev[sup[c]]);
            let y0 = DVector::from_fn(rows.len(), |r, _| -e[rows[r]]);
            let out = dual_newton(&a_dense, &b_vec, &c_vec, y0);
            iterations = out.iterations;
            if out.a.iter().all(|x| x.is_finite()) {
                for (c, &m) in sup.iter().enumerate() {
                    act[m] = out.a[c];
                }
                e = target.clone();
                for &k in &rows {
                    for m in 0..nm {
                        e[k] -= a_s(k, m) * act[m];
                    }
                }
                converged = out.converged;
            }
        }
        let fallback = !converged;
        for sweep in 0..if converged { 0 } else { MAX_SWEEPS } {
            iterations += 1;
            let _ = sweep;
            let mut max_delta = 0.0f64;
            for m in 0..nm {
                if !supported[m] {
                    continue;
                }
                let grad: f64 = rows.iter().map(|&k| a_s(k, m) * e[k]).sum::<f64>()
                    - LAMBDA_EFFORT * act[m]
                    - LAMBDA_CONTINUITY * (act[m] - prev[m]);
                let new = (act[m] + grad / denom[m]).clamp(0.0, 1.0);
                let d = new - act[m];
                if d != 0.0 {
                    for &k in &rows {
                        e[k] -= a_s(k, m) * d;
                    }
                    act[m] = new;
                    max_delta = max_delta.max(d.abs());
                }
            }
            if max_delta < CONVERGENCE {
                converged = true;
                break;
            }
        }

        if fallback {
            // Small coordinate steps alone can hide ill-conditioning. Verify
            // projected KKT conditions before reporting fallback convergence.
            converged = (0..nm).filter(|&m| supported[m]).all(|m| {
                let gp = -rows.iter().map(|&k| a_s(k,m)*e[k]).sum::<f64>()
                    + LAMBDA_EFFORT*act[m] + LAMBDA_CONTINUITY*(act[m]-prev[m]);
                if act[m] <= 0.0 { gp >= -KKT_TOL }
                else if act[m] >= 1.0 { gp <= KKT_TOL }
                else { gp.abs() <= KKT_TOL }
            });
        }

        let tension: Vec<f64> = (0..nm).map(|m| f0[m] + (f1[m] - f0[m]) * act[m]).collect();
        let muscle: Vec<f64> = (0..NX)
            .map(|k| p_muscle[k] + (0..nm).map(|m| a_mat[(k, m)] * act[m]).sum::<f64>())
            .collect();
        let residual: Vec<f64> = (0..NX).map(|k| required[k] - passive[k] - muscle[k]).collect();
        let residual_norm = rows.iter().map(|&k| residual[k].powi(2)).sum::<f64>().sqrt();
        let demand = rows.iter().map(|&k| (required[k] - passive[k]).powi(2)).sum::<f64>().sqrt();
        if residual.iter().chain(&act).chain(&tension).any(|x| !x.is_finite()) {
            return Err("non-finite recruitment result".into());
        }
        self.last = act.clone();
        Ok(Recruitment {
            activation: act,
            tension_n: tension,
            supported_muscles: supported,
            required_torque_nm: required,
            passive_torque_nm: passive,
            muscle_torque_nm: muscle,
            residual_torque_nm: residual,
            observed_coordinates: observed,
            residual_norm_nm: residual_norm,
            relative_residual: residual_norm / demand.max(1.0),
            converged,
            iterations,
        })
    }
}

struct DualResult {
    a: Vec<f64>,
    iterations: usize,
    converged: bool,
}

/// Dual value and primal point for `y`: `s = c - A^T y`, `a = clamp(s / lambda, 0, 1)`,
/// `g(y) = 1/2 y.y + b.y + sum(s a - lambda a^2 / 2)`.
fn dual_eval(a: &DMatrix<f64>, b: &DVector<f64>, c: &DVector<f64>, y: &DVector<f64>, lam: f64) -> (f64, DVector<f64>) {
    let s = c - a.transpose() * y;
    let act = s.map(|v| (v / lam).clamp(0.0, 1.0));
    let mut g = 0.5 * y.dot(y) + b.dot(y);
    for i in 0..s.len() {
        g += s[i] * act[i] - 0.5 * lam * act[i] * act[i];
    }
    (g, act)
}

/// Damped semismooth Newton on the strictly convex, C1 dual. Gradient is `y + b - A a`; the
/// Hessian is `I + sum_free A_i A_i^T / lambda` (SPD). Convergence is the gradient infinity norm
/// <= 1e-8, or a verified projected primal KKT residual <= 1e-6; anything else is reported false.
fn dual_newton(a: &DMatrix<f64>, b: &DVector<f64>, c: &DVector<f64>, y0: DVector<f64>) -> DualResult {
    // Homotopy in lambda: heavily regularized stages change the active set gradually and warm
    // start the exact (final) stage, which is the contract objective.
    let mut y = y0;
    let mut total = 0;
    for k in (1..=HOMOTOPY_STAGES).rev() {
        let lam = LAMBDA * 10f64.powi(k as i32);
        let (yk, it, _) = newton_stage(a, b, c, y, lam, 30);
        y = yk;
        total += it;
    }
    let (y, it, act) = newton_stage(a, b, c, y, LAMBDA, NEWTON_ITERS);
    total += it;
    finish(a, b, c, y, act, total)
}

fn newton_stage(a: &DMatrix<f64>, b: &DVector<f64>, c: &DVector<f64>, y0: DVector<f64>, lam: f64, max_iter: usize) -> (DVector<f64>, usize, DVector<f64>) {
    let mut y = y0;
    if y.iter().any(|v| !v.is_finite()) {
        y.fill(0.0);
    }
    let (mut g, mut act) = dual_eval(a, b, c, &y, lam);
    let mut iterations = 0;
    let mut _converged = false;
    for it in 0..max_iter {
        iterations = it + 1;
        let grad = &y + b - a * &act;
        if grad.iter().any(|v| !v.is_finite()) {
            break;
        }
        if grad.amax() <= DUAL_TOL {
            _converged = true;
            break;
        }
        let s = c - a.transpose() * &y;
        let mut h = DMatrix::<f64>::identity(y.len(), y.len());
        for i in 0..s.len() {
            let v = s[i] / lam;
            if v > 0.0 && v < 1.0 {
                let col = a.column(i);
                h += (&col * col.transpose()) / lam;
            }
        }
        let Some(chol) = h.cholesky() else { break };
        let d = chol.solve(&(-&grad));
        let slope = grad.dot(&d);
        if !(slope < 0.0) {
            break;
        }
        let mut t = 1.0;
        let mut accepted = false;
        for _ in 0..40 {
            let trial = &y + &d * t;
            let (gt, at) = dual_eval(a, b, c, &trial, lam);
            if gt.is_finite() && gt <= g + 1e-4 * t * slope {
                y = trial;
                g = gt;
                act = at;
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    (y, iterations, act)
}

fn finish(a: &DMatrix<f64>, b: &DVector<f64>, c: &DVector<f64>, y: DVector<f64>, act: DVector<f64>, iterations: usize) -> DualResult {
    let grad_inf = (&y + b - a * &act).amax();
    let mut converged = grad_inf <= DUAL_TOL;
    if !converged {
        // Verify the projected primal KKT conditions directly.
        let r = a * &act - b;
        let ok = (0..act.len()).all(|i| {
            let gp = a.column(i).dot(&r) + LAMBDA * act[i] - c[i];
            if act[i] <= 0.0 {
                gp >= -KKT_TOL
            } else if act[i] >= 1.0 {
                gp <= KKT_TOL
            } else {
                gp.abs() <= KKT_TOL
            }
        });
        converged = ok && act.iter().all(|v| v.is_finite());
    }
    DualResult { a: act.iter().copied().collect(), iterations, converged }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(a: &DMatrix<f64>, b: &[f64], prev: &[f64]) -> DualResult {
        let bv = DVector::from_column_slice(b);
        let c = DVector::from_fn(prev.len(), |i, _| LAMBDA_CONTINUITY * prev[i]);
        dual_newton(a, &bv, &c, DVector::zeros(b.len()))
    }

    #[test]
    fn duplicate_muscles_share_equally() {
        let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let r = solve(&a, &[1.0], &[0.0, 0.0]);
        assert!(r.converged);
        assert!((r.a[0] - 0.5).abs() < 1e-3 && (r.a[1] - 0.5).abs() < 1e-3, "{:?}", r.a);
        assert!((r.a[0] - r.a[1]).abs() < 1e-9, "must not favour the first column");
    }

    #[test]
    fn bounds_hold_when_demand_exceeds_capacity() {
        let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let r = solve(&a, &[3.0], &[0.0, 0.0]);
        assert!(r.converged);
        assert!(r.a.iter().all(|x| (x - 1.0).abs() < 1e-9), "{:?}", r.a);
        let r0 = solve(&a, &[-2.0], &[0.0, 0.0]);
        assert!(r0.a.iter().all(|x| x.abs() < 1e-9), "{:?}", r0.a);
    }

    #[test]
    fn permutation_invariant() {
        let a = DMatrix::from_row_slice(2, 4, &[1.0, 0.5, 0.2, 0.9, 0.1, 0.7, 0.4, 0.3]);
        let b = [0.8, 0.6];
        let r = solve(&a, &b, &[0.0; 4]);
        let perm = [2usize, 0, 3, 1];
        let ap = DMatrix::from_fn(2, 4, |i, j| a[(i, perm[j])]);
        let rp = solve(&ap, &b, &[0.0; 4]);
        assert!(r.converged && rp.converged);
        for (j, &p) in perm.iter().enumerate() {
            assert!((rp.a[j] - r.a[p]).abs() < 1e-7, "col {j}: {} vs {}", rp.a[j], r.a[p]);
        }
    }

    #[test]
    fn continuity_pulls_toward_previous() {
        let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let r = solve(&a, &[1.0], &[1.0, 0.0]);
        assert!(r.a[0] > r.a[1], "{:?}", r.a);
    }
}
