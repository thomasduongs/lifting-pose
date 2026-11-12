//! Core parity against the independent native MuJoCo reference (`reference.json`, written by
//! `scripts/export_biomech_lite.py` with mj_forward). Asset directory: `$BIOMECH_LITE_ASSETS`
//! or `<repo>/assets/biomech-lite`. Missing assets are a test failure, not a skip.
//! Run with `cargo test --release --test core_parity -- --nocapture` to see observed max errors.

use biomech_lite::Model;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Case {
    name: String,
    qpos: Vec<f64>,
    body_pos: Vec<[f64; 3]>,
    body_quat: Vec<[f64; 4]>,
    site_pos: Vec<[f64; 3]>,
    tendon_lengths: Vec<f64>,
    tendon_velocities: Vec<f64>,
    activation: Vec<f64>,
    muscle_tensions: Vec<f64>,
    activation_derivative: Vec<f64>,
    control: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    schema_version: u32,
    cases: Vec<Case>,
}

fn asset_dir() -> PathBuf {
    std::env::var_os("BIOMECH_LITE_ASSETS").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/biomech-lite")
    })
}

fn load() -> (Model, Reference) {
    let dir = asset_dir();
    let model = Model::load(dir.join("model.json")).unwrap_or_else(|e| {
        panic!("{e}\n(export assets first: .venv-biomech/bin/python scripts/export_biomech_lite.py)")
    });
    let text = std::fs::read_to_string(dir.join("reference.json"))
        .unwrap_or_else(|e| panic!("cannot read {}/reference.json: {e}", dir.display()));
    let r: Reference = serde_json::from_str(&text).expect("reference.json schema");
    assert_eq!(r.schema_version, 1);
    assert!(!r.cases.is_empty(), "reference has no cases");
    (model, r)
}

/// Running maximum with the case/index where it occurred.
struct MaxErr {
    what: &'static str,
    tol: f64,
    val: f64,
    at: String,
}
impl MaxErr {
    fn new(what: &'static str, tol: f64) -> Self {
        MaxErr { what, tol, val: 0.0, at: String::new() }
    }
    fn see(&mut self, e: f64, case: &str, i: usize) {
        assert!(e.is_finite(), "{}: non-finite error in {case}[{i}]", self.what);
        if e > self.val {
            self.val = e;
            self.at = format!("{case}[{i}]");
        }
    }
    fn report(&self) -> bool {
        let ok = self.val <= self.tol;
        eprintln!("{:<28} max {:>10.3e} (tol {:.0e}) at {} {}", self.what, self.val, self.tol, self.at,
            if ok { "ok" } else { "FAIL" });
        ok
    }
}

fn d3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[test]
fn model_counts_and_support() {
    let (m, r) = load();
    assert_eq!(m.qpos0.len(), m.nq);
    assert_eq!(r.cases[0].qpos.len(), m.nq);
    assert_eq!(r.cases[0].body_pos.len(), m.bodies.len());
    assert_eq!(r.cases[0].site_pos.len(), m.sites.len());
    assert_eq!(r.cases[0].tendon_lengths.len(), m.tendons.len());
    assert_eq!(r.cases[0].muscle_tensions.len(), m.muscles.len());
    eprintln!("bodies {} joints {} sites {} geoms {} couplings {} tendons {} muscles {} cases {}",
        m.bodies.len(), m.joints.len(), m.sites.len(), m.geoms.len(), m.couplings.len(),
        m.tendons.len(), m.muscles.len(), r.cases.len());
    // standing pose is coupling-consistent
    let res = m.coupling_residuals(&m.standing_qpos);
    assert!(res.iter().all(|x| x.abs() < 1e-9), "standing_qpos violates couplings");
}

#[test]
fn kinematics_wrapping_muscle_parity() {
    let (m, r) = load();
    let mut e_proj = MaxErr::new("coupling projection (rad|m)", 1e-10);
    let mut e_body = MaxErr::new("body position (m)", 1e-8);
    let mut e_quat = MaxErr::new("body quaternion", 1e-8);
    let mut e_site = MaxErr::new("site position (m)", 1e-8);
    let mut e_len = MaxErr::new("tendon length (m)", 1e-7);
    let mut e_frc = MaxErr::new("muscle tension (N)", 1e-5);
    let mut e_act = MaxErr::new("activation rate (rel)", 1e-9);
    let mut e_pas = MaxErr::new("passive<=total (N)", 1e-9);
    for c in &r.cases {
        let n = c.name.as_str();
        // reference qpos is already projected natively; Rust projection must not move it
        let mut q = c.qpos.clone();
        m.project(&mut q).unwrap();
        for i in 0..m.nq {
            e_proj.see((q[i] - c.qpos[i]).abs(), n, i);
        }
        let k = m.kinematics(&c.qpos).unwrap();
        for (i, (a, b)) in k.body_pos.iter().zip(&c.body_pos).enumerate() {
            e_body.see(d3(a, b), n, i);
        }
        for (i, (a, b)) in k.body_quat.iter().zip(&c.body_quat).enumerate() {
            let dp: f64 = (0..4).map(|j| (a[j] - b[j]).powi(2)).sum::<f64>().sqrt();
            let dm: f64 = (0..4).map(|j| (a[j] + b[j]).powi(2)).sum::<f64>().sqrt();
            e_quat.see(dp.min(dm), n, i);
        }
        for (i, (a, b)) in k.site_pos.iter().zip(&c.site_pos).enumerate() {
            e_site.see(d3(a, b), n, i);
        }
        let lens = m.tendon_lengths(&k).unwrap();
        for (i, (a, b)) in lens.iter().zip(&c.tendon_lengths).enumerate() {
            e_len.see((a - b).abs(), n, i);
        }
        // forces evaluated from the reference's native tendon state isolate the muscle model;
        // forces from Rust lengths check the composed pipeline
        let f_ref = m.muscle_forces(&c.tendon_lengths, &c.tendon_velocities, &c.activation).unwrap();
        let f_rust = m.muscle_forces(&lens, &c.tendon_velocities, &c.activation).unwrap();
        for i in 0..m.muscles.len() {
            e_frc.see((f_ref[i] - c.muscle_tensions[i]).abs(), n, i);
            e_frc.see((f_rust[i] - c.muscle_tensions[i]).abs(), n, i);
        }
        let passive = m.passive_muscle_forces(&lens).unwrap();
        for (i, p) in passive.iter().enumerate() {
            e_pas.see((-p).max(0.0), n, i);
        }
        let ad = m.activation_rates(&c.control, &c.activation).unwrap();
        for i in 0..m.muscles.len() {
            let want = c.activation_derivative[i];
            e_act.see((ad[i] - want).abs() / want.abs().max(1.0), n, i);
        }
    }
    let ok = [&e_proj, &e_body, &e_quat, &e_site, &e_len, &e_frc, &e_act, &e_pas]
        .iter().map(|e| e.report()).fold(true, |a, b| a & b);
    assert!(ok, "parity outside tolerance (see report above)");
}

#[test]
fn finite_difference_velocities_are_consistent() {
    // dL/dt along a coupled velocity equals the derivative of lengths along the same path
    let (m, _) = load();
    let mut q = m.standing_qpos.clone();
    m.project(&mut q).unwrap();
    let mut v = vec![0.0; m.nv];
    for (k, &j) in m.dynamic_joints.iter().enumerate() {
        v[m.joints[j].dofadr] = 0.3 + 0.1 * k as f64;
    }
    m.project_velocity(&q, &mut v).unwrap();
    let vel = m.tendon_velocities(&q, &v, 1e-6).unwrap();
    let h = 1e-4;
    let qp = m.integrate_pos(&q, &v, h).unwrap();
    let qm = m.integrate_pos(&q, &v, -h).unwrap();
    let lp = m.tendon_lengths(&m.kinematics(&qp).unwrap()).unwrap();
    let lm = m.tendon_lengths(&m.kinematics(&qm).unwrap()).unwrap();
    let mut worst: f64 = 0.0;
    for i in 0..vel.len() {
        worst = worst.max((vel[i] - (lp[i] - lm[i]) / (2.0 * h)).abs());
    }
    eprintln!("tendon velocity step consistency max {worst:.3e} m/s");
    assert!(worst < 1e-5);
}

#[test]
fn unsupported_features_fail_clearly() {
    let (m, _) = load();
    let mut bad = m.clone();
    let j = bad.dynamic_joints[0];
    bad.joints[j].kind = biomech_lite::JNT_BALL;
    assert!(bad.validate().unwrap_err().contains("ball"));
    let mut bad = m.clone();
    bad.tendons[0].path[0].kind = 6; // pulley
    assert!(bad.validate().unwrap_err().contains("not supported"));
    let mut bad = m.clone();
    bad.schema_version = 2;
    assert!(bad.validate().is_err());
    assert!(m.kinematics(&vec![0.0; m.nq - 1]).is_err());
}
