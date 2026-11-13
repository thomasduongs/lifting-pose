//! Reduced inverse dynamics / bounded recruitment (`docs/muscle-estimation-contract.md`).
//! `cargo test --release --test inverse -- --nocapture` prints solve timing.

use biomech_lite::dynamics::{self, NX};
use biomech_lite::inverse::Recruiter;
use biomech_lite::Model;
use nalgebra::DVector;
use std::path::PathBuf;
use std::sync::Arc;

fn model() -> Arc<Model> {
    let dir = std::env::var_os("BIOMECH_LITE_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/biomech-lite"));
    Arc::new(Model::load(dir.join("model.json")).expect("model.json"))
}

/// Standing pose with every dynamic coordinate moved by `delta`, clamped inside its range.
fn pose(m: &Model, delta: f64) -> Vec<f64> {
    let mut q = m.standing_qpos.clone();
    for &j in &m.dynamic_joints {
        let jt = &m.joints[j];
        let x = q[jt.qposadr] + delta;
        q[jt.qposadr] = if jt.limited { x.clamp(jt.range[0], jt.range[1]) } else { x };
    }
    m.project(&mut q).unwrap();
    q
}

const ALL: [bool; 8] = [true; 8];

#[test]
fn reconstructs_known_activation_demand() {
    let m = model();
    let q = pose(&m, 0.25);
    let zero = vec![0.0; NX];
    let a_true: Vec<f64> = (0..m.muscles.len()).map(|i| if i % 4 == 0 { 0.15 } else { 0.0 }).collect();
    let tau_m = dynamics::muscle_torque(&m, &q, &zero, &a_true).unwrap();
    let tau_p = dynamics::passive_torque(&m, &q, &zero).unwrap();
    let bias = dynamics::bias(&m, &q, &zero).unwrap();
    let rhs = DVector::from_fn(NX, |i, _| tau_m[i] + tau_p[i] - bias[i]);
    let qdd = dynamics::mass_matrix(&m, &q).unwrap().cholesky().unwrap().solve(&rhs);
    let acc: Vec<f64> = qdd.iter().copied().collect();

    let mut r = Recruiter::new(m.clone()).unwrap();
    let out = r.solve(&q, &zero, &acc, ALL).unwrap();
    assert!(out.residual_norm_nm < 0.5, "residual {} Nm", out.residual_norm_nm);
    assert!(out.activation.iter().all(|a| a.is_finite() && (0.0..=1.0).contains(a)));
    // required torque equals M a + bias by construction of the synthetic demand.
    for i in 0..NX {
        let want = tau_m[i] + tau_p[i];
        assert!((out.required_torque_nm[i] - want).abs() < 1e-6, "row {i}");
    }
}

#[test]
fn force_identity_matches_muscle_torque() {
    let m = model();
    let q = pose(&m, 0.2);
    let zero = vec![0.0; NX];
    let mut r = Recruiter::new(m.clone()).unwrap();
    let out = r.solve(&q, &zero, &zero, ALL).unwrap();
    let want = dynamics::muscle_torque(&m, &q, &zero, &out.activation).unwrap();
    for k in 0..NX {
        assert!((out.muscle_torque_nm[k] - want[k]).abs() < 1e-6, "row {k}: {} vs {}", out.muscle_torque_nm[k], want[k]);
        let resid = out.required_torque_nm[k] - out.passive_torque_nm[k] - out.muscle_torque_nm[k];
        assert!((resid - out.residual_torque_nm[k]).abs() < 1e-9);
    }
    assert!(out.tension_n.iter().all(|t| t.is_finite()));
}

#[test]
fn static_gravity_residual_is_small_or_reported() {
    let m = model();
    let zero = vec![0.0; NX];
    let mut r = Recruiter::new(m.clone()).unwrap();
    for delta in [0.0, 0.3, 0.6] {
        let q = pose(&m, delta);
        let out = r.solve(&q, &zero, &zero, ALL).unwrap();
        let demand: f64 = (0..NX)
            .map(|k| (out.required_torque_nm[k] - out.passive_torque_nm[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        eprintln!(
            "delta {delta}: demand {demand:.2} Nm residual {:.3} Nm rel {:.3} iters {} converged {}",
            out.residual_norm_nm, out.relative_residual, out.iterations, out.converged
        );
        assert!(out.residual_norm_nm.is_finite());
        assert!(out.relative_residual >= 0.0);
        assert!((out.residual_norm_nm / demand.max(1.0) - out.relative_residual).abs() < 1e-9);
    }
}

#[test]
fn impossible_demand_keeps_bounds_and_exposes_residual() {
    let m = model();
    let q = pose(&m, 0.2);
    let zero = vec![0.0; NX];
    let acc = vec![5000.0; NX];
    let mut r = Recruiter::new(m.clone()).unwrap();
    let out = r.solve(&q, &zero, &acc, ALL).unwrap();
    assert!(out.activation.iter().all(|a| (0.0..=1.0).contains(a)));
    assert!(out.residual_norm_nm > 100.0, "residual {}", out.residual_norm_nm);
    assert!(out.relative_residual > 0.1);
}

#[test]
fn missing_side_excludes_that_sides_muscles() {
    let m = model();
    let q = pose(&m, 0.2);
    let zero = vec![0.0; NX];
    // Dynamic joints are native order (right then left); derive sides from names.
    let mut obs = [false; 8];
    for (k, &j) in m.dynamic_joints.iter().enumerate() {
        obs[k] = m.joints[j].name.ends_with("_r");
    }
    assert!(obs.iter().any(|o| *o) && obs.iter().any(|o| !*o), "both sides expected");
    let mut r = Recruiter::new(m.clone()).unwrap();
    let both = r.solve(&q, &zero, &zero, ALL).unwrap();
    r.reset();
    let right = r.solve(&q, &zero, &zero, obs).unwrap();
    assert_eq!(right.observed_coordinates, obs);
    let n_both = both.supported_muscles.iter().filter(|s| **s).count();
    let n_right = right.supported_muscles.iter().filter(|s| **s).count();
    assert!(n_right > 0 && n_right < n_both, "{n_right} vs {n_both}");
    for (i, s) in right.supported_muscles.iter().enumerate() {
        if !s {
            assert_eq!(right.activation[i], 0.0, "unsupported muscle {i} must stay zero");
        }
    }
    // Unobserved rows do not count toward the residual norm.
    let obs_norm: f64 = (0..NX).filter(|k| obs[*k]).map(|k| right.residual_torque_nm[k].powi(2)).sum::<f64>().sqrt();
    assert!((obs_norm - right.residual_norm_nm).abs() < 1e-9);
}

#[test]
fn invalid_inputs_are_rejected_and_q_untouched() {
    let m = model();
    let q = pose(&m, 0.2);
    let before = q.clone();
    let zero = vec![0.0; NX];
    let mut r = Recruiter::new(m.clone()).unwrap();
    assert!(r.solve(&q[..q.len() - 1], &zero, &zero, ALL).is_err());
    assert!(r.solve(&q, &zero[..7], &zero, ALL).is_err());
    assert!(r.solve(&q, &zero, &zero[..7], ALL).is_err());
    let mut nan = zero.clone();
    nan[2] = f64::NAN;
    assert!(r.solve(&q, &nan, &zero, ALL).is_err());
    assert!(r.solve(&q, &zero, &nan, ALL).is_err());
    let mut bad_q = q.clone();
    bad_q[0] = f64::INFINITY;
    assert!(r.solve(&bad_q, &zero, &zero, ALL).is_err());
    assert!(r.solve(&q, &zero, &zero, [false; 8]).is_err());
    assert_eq!(q, before);
    // A rejected call must not disturb a following valid solve.
    assert!(r.solve(&q, &zero, &zero, ALL).is_ok());
}

#[test]
fn unobserved_motion_does_not_leak_into_observed_rows() {
    let m = model();
    let q = pose(&m, 0.2);
    let zero = vec![0.0; NX];
    let mut obs = [false; 8];
    for (k, &j) in m.dynamic_joints.iter().enumerate() {
        obs[k] = m.joints[j].name.ends_with("_r");
    }
    let mut wild = zero.clone();
    for k in 0..NX {
        if !obs[k] {
            wild[k] = 3.0;
        }
    }
    let mut r = Recruiter::new(m.clone()).unwrap();
    let a = r.solve(&q, &zero, &zero, obs).unwrap();
    r.reset();
    let b = r.solve(&q, &wild, &wild, obs).unwrap();
    for k in 0..NX {
        if obs[k] {
            assert!((a.required_torque_nm[k] - b.required_torque_nm[k]).abs() < 1e-9);
        }
    }
}

#[test]
fn benchmark_real_solves() {
    let m = model();
    let q = pose(&m, 0.3);
    let zero = vec![0.0; NX];
    let vel = vec![0.3; NX];
    let acc = vec![1.0; NX];
    let mut r = Recruiter::new(m.clone()).unwrap();
    let n = 5;
    let t = std::time::Instant::now();
    let mut last = None;
    for _ in 0..n {
        last = Some(r.solve(&q, &vel, &acc, ALL).unwrap());
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let out = last.unwrap();
    eprintln!(
        "recruitment solve: {ms:.1} ms avg over {n} (muscles {}, iters {}, converged {}, residual {:.2} Nm)",
        m.muscles.len(),
        out.iterations,
        out.converged,
        out.residual_norm_nm
    );
    let _ = zero;
    assert!(ms.is_finite());
}
