//! Reduced-dynamics parity against independent native MuJoCo fixtures
//! (`assets/biomech-lite/reference.json`) plus short stability checks.

use biomech_lite::dynamics::{self, State};
use biomech_lite::Model;
use serde_json::Value;
use std::path::PathBuf;

fn assets() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/biomech-lite")
}

fn model() -> Model {
    Model::load(assets().join("model.json")).expect("model.json")
}

fn vecf(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases() -> Vec<Value> {
    let text = std::fs::read_to_string(assets().join("reference.json")).expect("reference.json");
    let v: Value = serde_json::from_str(&text).unwrap();
    v["cases"].as_array().unwrap().clone()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn mass_gravity_bias_torque_match_native() {
    let model = model();
    let (mut mass_rel, mut grav, mut bias, mut torque, mut passive) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for c in cases() {
        let name = c["name"].as_str().unwrap().to_string();
        let q = vecf(&c["qpos"]);
        let v = vecf(&c["reduced_velocity"]);
        let a = vecf(&c["activation"]);

        let m = dynamics::mass_matrix(&model, &q).unwrap();
        let rm = c["reduced_mass"].as_array().unwrap();
        let mut scale = 0.0f64;
        let mut err = 0.0f64;
        for i in 0..8 {
            let row = vecf(&rm[i]);
            for j in 0..8 {
                scale = scale.max(row[j].abs());
                err = err.max((m[(i, j)] - row[j]).abs());
            }
        }
        mass_rel = mass_rel.max(err / scale);

        grav = grav.max(max_abs_diff(&dynamics::gravity(&model, &q).unwrap(), &vecf(&c["reduced_gravity"])));
        let b = dynamics::bias(&model, &q, &v).unwrap();
        let be = max_abs_diff(&b, &vecf(&c["reduced_bias"]));
        bias = bias.max(be);
        let t = dynamics::muscle_torque(&model, &q, &v, &a).unwrap();
        let te = max_abs_diff(&t, &vecf(&c["reduced_muscle_torque"]));
        torque = torque.max(te);
        let pe = max_abs_diff(&dynamics::passive_torque(&model, &q, &v).unwrap(), &vecf(&c["reduced_passive_torque"]));
        passive = passive.max(pe);
        eprintln!("{name}: bias err {be:.3e} torque err {te:.3e}");
    }
    eprintln!("max mass rel {mass_rel:.3e}, gravity {grav:.3e} Nm, bias {bias:.3e} Nm, muscle torque {torque:.3e} Nm, passive {passive:.3e} Nm");
    assert!(passive < 1e-4, "passive torque error {passive}");
    assert!(mass_rel < 1e-10, "mass relative error {mass_rel}");
    assert!(grav < 1e-10, "gravity error {grav}");
    assert!(bias < 1e-7, "bias error {bias}");
    assert!(torque < 1e-5, "muscle torque error {torque}");
}

#[test]
fn step_is_finite_and_rejects_bad_input() {
    let model = model();
    let mut st = State::standing(&model).unwrap();
    let n = model.muscles.len();
    let mut ex = vec![0.0; n];
    if let Some(i) = model.muscle_id("BIClong") {
        ex[i] = 0.25;
    }
    for _ in 0..50 {
        dynamics::step(&model, &mut st, &ex, 0.0005).unwrap();
    }
    assert!(st.qpos.iter().chain(&st.velocity).chain(&st.activation).all(|x| x.is_finite()));
    assert!(st.activation.iter().all(|a| (0.0..=1.0).contains(a)));

    let before = st.clone();
    assert!(dynamics::step(&model, &mut st, &ex, 0.0).is_err());
    assert!(dynamics::step(&model, &mut st, &ex, 0.01).is_err());
    assert!(dynamics::step(&model, &mut st, &ex[1..], 0.001).is_err());
    st.velocity[0] = f64::NAN;
    assert!(dynamics::step(&model, &mut st, &ex, 0.001).is_err());
    st.velocity[0] = before.velocity[0];
    assert_eq!(st, before, "failed steps must preserve the input state");
}

/// Timing of the documented 500-step run (dt 0.0005 s, BIClong excitation 0.25).
#[test]
fn bench_500_steps() {
    let model = model();
    let mut st = State::standing(&model).unwrap();
    let mut ex = vec![0.0; model.muscles.len()];
    ex[model.muscle_id("BIClong").unwrap()] = 0.25;
    let t0 = std::time::Instant::now();
    for _ in 0..500 {
        dynamics::step(&model, &mut st, &ex, 0.0005).unwrap();
    }
    eprintln!("500 steps: {:.3} s", t0.elapsed().as_secs_f64());
}

fn in_range(model: &Model, q: &[f64]) -> bool {
    model.joints.iter().all(|j| !(j.is_scalar() && j.limited)
        || (q[j.qposadr] >= j.range[0] - 1e-9 && q[j.qposadr] <= j.range[1] + 1e-9))
}

/// Drive each coordinate hard into both of its limits: bounds hold every step (masters clamp,
/// dependent joints backtrack) and an out-of-range start is rejected.
#[test]
fn bounds_hold_when_driven_into_limits() {
    let model = model();
    let ex = vec![0.0; model.muscles.len()];
    let mut clamped = 0;
    for i in 0..8 {
        for sign in [-1.0, 1.0] {
            let mut st = State::standing(&model).unwrap();
            st.velocity[i] = 8.0 * sign;
            for _ in 0..300 {
                dynamics::step(&model, &mut st, &ex, 0.002).unwrap();
                assert!(in_range(&model, &st.qpos), "coordinate {i} sign {sign} left its range");
                assert!(st.velocity.iter().all(|v| v.is_finite()));
            }
            let jt = &model.joints[model.dynamic_joints[i]];
            let x = st.qpos[jt.qposadr];
            if (x - jt.range[0]).abs() < 1e-6 || (x - jt.range[1]).abs() < 1e-6 {
                clamped += 1;
            }
        }
    }
    eprintln!("coordinates resting on a master bound: {clamped}/16");
    assert!(clamped > 0, "no run reached a bound; test does not exercise clamping");

    let mut st = State::standing(&model).unwrap();
    let j = &model.joints[model.dynamic_joints[1]];
    st.qpos[j.qposadr] = j.range[1] + 0.1;
    let before = st.clone();
    assert!(dynamics::step(&model, &mut st, &ex, 0.001).is_err());
    assert_eq!(st, before);
}
