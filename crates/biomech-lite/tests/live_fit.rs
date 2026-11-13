//! Live arm fitting: synthetic native-FK marker poses mapped through a proper camera basis.
//! Asset directory: `$BIOMECH_LITE_ASSETS` or `<repo>/assets/biomech-lite` (missing = failure).
//! `cargo test --release --test live_fit -- --nocapture` prints timing and errors.

use biomech_lite::Model;
use biomech_lite::live_fit::{Landmark, LiveFitter, MAX_ITERATIONS, Observation};
use std::path::PathBuf;
use std::sync::Arc;

type V3 = [f64; 3];

fn model() -> Arc<Model> {
    let dir = std::env::var_os("BIOMECH_LITE_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/biomech-lite")
        });
    Arc::new(Model::load(dir.join("model.json")).expect("model.json"))
}

fn sub(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn norm(a: &V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
fn cross(a: &V3, b: &V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(a: &V3) -> V3 {
    let n = norm(a);
    [a[0] / n, a[1] / n, a[2] / n]
}

const STEMS: [&str; 4] = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion"];
const SFX: [&str; 2] = ["_l", "_r"];

/// Independent marker definitions straight from full `Model::kinematics`.
fn native_markers(m: &Model, q: &[f64]) -> [[V3; 3]; 2] {
    let k = m.kinematics(q).unwrap();
    let mut out = [[[0.0; 3]; 3]; 2];
    for s in 0..2 {
        let sh = k.joint_anchor[m.joint_id(&format!("shoulder_elv{}", SFX[s])).unwrap()];
        let a = k.site_pos[m
            .site_id(&format!("Elbow_Lateral_marker{}", SFX[s]))
            .unwrap()];
        let b = k.site_pos[m
            .site_id(&format!("Elbow_Medial_marker{}", SFX[s]))
            .unwrap()];
        let wr = k.joint_anchor[m.joint_id(&format!("flexion{}", SFX[s])).unwrap()];
        out[s] = [
            sh,
            [
                (a[0] + b[0]) / 2.0,
                (a[1] + b[1]) / 2.0,
                (a[2] + b[2]) / 2.0,
            ],
            wr,
        ];
    }
    out
}

fn standing(m: &Model) -> Vec<f64> {
    let mut q = m.standing_qpos.clone();
    m.project(&mut q).unwrap();
    q
}

fn pose(m: &Model, arms: [[f64; 4]; 2]) -> Vec<f64> {
    let mut q = standing(m);
    for s in 0..2 {
        for k in 0..4 {
            q[m.joints[m.joint_id(&format!("{}{}", STEMS[k], SFX[s])).unwrap()].qposadr] =
                arms[s][k];
        }
    }
    m.project(&mut q).unwrap();
    q
}

fn masters(m: &Model, q: &[f64]) -> [[f64; 4]; 2] {
    let mut out = [[0.0; 4]; 2];
    for s in 0..2 {
        for k in 0..4 {
            out[s][k] =
                q[m.joints[m.joint_id(&format!("{}{}", STEMS[k], SFX[s])).unwrap()].qposadr];
        }
    }
    out
}

/// Synthetic camera: the person faces the camera with torso yaw `yaw` (about camera up = -y) and
/// is `scale` times the reference size (retargeting must not care). Model -> camera is a PROPER
/// rotation: model lateral(left) -> camera ~+x, model up(+z) -> camera -y.
struct Cam {
    r: [[f64; 3]; 3],
    scale: f64,
    origin: V3,
}

impl Cam {
    fn new(m: &Model, yaw: f64, scale: f64) -> Self {
        let mk = native_markers(m, &standing(m));
        let lat_m = unit(&sub(&mk[0][0], &mk[1][0]));
        let up0 = [0.0, 0.0, 1.0];
        let d = lat_m[2];
        let up_m = unit(&[
            up0[0] - d * lat_m[0],
            up0[1] - d * lat_m[1],
            up0[2] - d * lat_m[2],
        ]);
        let fw_m = cross(&lat_m, &up_m);
        let lat_c = [yaw.cos(), 0.0, yaw.sin()];
        let up_c = [0.0, -1.0, 0.0];
        let fw_c = cross(&lat_c, &up_c);
        // r = Bc * Bm^T
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = lat_c[i] * lat_m[j] + up_c[i] * up_m[j] + fw_c[i] * fw_m[j];
            }
        }
        Cam {
            r,
            scale,
            origin: [0.0, 0.0, 1.0],
        }
    }
    fn map(&self, p: &V3) -> V3 {
        let mut o = [0.0; 3];
        for i in 0..3 {
            o[i] = self.origin[i] + self.scale * (0..3).map(|j| self.r[i][j] * p[j]).sum::<f64>();
        }
        o
    }
}

fn observe(m: &Model, cam: &Cam, q: &[f64], t: f64) -> Observation {
    let mk = native_markers(m, q);
    let mut landmarks = Vec::new();
    for (s, side) in ["left", "right"].iter().enumerate() {
        for (j, joint) in ["shoulder", "elbow", "wrist"].iter().enumerate() {
            landmarks.push(Landmark {
                name: format!("{side}_{joint}"),
                position: cam.map(&mk[s][j]),
                confidence: 0.9,
            });
        }
    }
    Observation {
        timestamp_seconds: t,
        landmarks,
    }
}

fn set_lm(obs: &mut Observation, name: &str, f: impl Fn(&mut Landmark)) {
    for l in obs.landmarks.iter_mut().filter(|l| l.name == name) {
        f(l);
    }
}

/// Shoulder-relative elbow/wrist offset error (m) between two configurations, per side.
fn offset_err(m: &Model, a: &[f64], b: &[f64]) -> [f64; 2] {
    let (ma, mb) = (native_markers(m, a), native_markers(m, b));
    std::array::from_fn(|s| {
        (1..3)
            .map(|j| norm(&sub(&sub(&ma[s][j], &ma[s][0]), &sub(&mb[s][j], &mb[s][0]))))
            .fold(0.0, f64::max)
    })
}

/// Couplings exact, all scalar ranges respected, everything that is neither a fitted master nor
/// a dependent equals the standing prior bit-for-bit.
fn check_invariants(m: &Model, q: &[f64]) {
    assert_eq!(q.len(), m.nq);
    let worst = m
        .coupling_residuals(q)
        .iter()
        .fold(0.0f64, |a, r| a.max(r.abs()));
    assert!(worst < 1e-12, "coupling residual {worst}");
    for j in &m.joints {
        if j.limited && j.is_scalar() {
            let v = q[j.qposadr];
            assert!(
                v >= j.range[0] && v <= j.range[1],
                "{} = {v} outside {:?}",
                j.name,
                j.range
            );
        }
    }
    let st = standing(m);
    let mut free = vec![false; m.nq];
    for &j in &m.dynamic_joints {
        free[m.joints[j].qposadr] = true;
    }
    for c in &m.couplings {
        free[m.joints[c.dependent].qposadr] = true;
    }
    for i in 0..m.nq {
        if !free[i] {
            assert_eq!(
                q[i].to_bits(),
                st[i].to_bits(),
                "fixed prior qpos[{i}] changed"
            );
        }
    }
}

#[test]
fn chain_markers_match_full_kinematics() {
    let m = model();
    let f = LiveFitter::new(m.clone()).unwrap();
    for arms in [
        [[0.0, 0.1, 0.0, 0.0]; 2],
        [[0.7, 1.9, -0.6, 1.3], [-0.9, 0.6, 1.2, 0.2]],
        [[2.0, 3.0, 1.8, 2.2], [-1.5, 2.5, -1.4, 1.0]],
    ] {
        let q = pose(&m, arms);
        let a = f.markers(&q).unwrap();
        let b = native_markers(&m, &q);
        for s in 0..2 {
            for j in 0..3 {
                assert!(
                    norm(&sub(&a[s][j], &b[s][j])) < 1e-12,
                    "marker {s}/{j} mismatch"
                );
            }
        }
    }
}

#[test]
fn effective_bounds_keep_dependents_in_range() {
    let m = model();
    let f = LiveFitter::new(m.clone()).unwrap();
    let b = f.master_bounds();
    for s in 0..2 {
        for k in 0..4 {
            let jt = &m.joints[f.fitted_joints()[s][k]];
            assert!(b[s][k].0 >= jt.range[0] && b[s][k].1 <= jt.range[1] && b[s][k].0 < b[s][k].1);
        }
        // native dependent ranges are tighter than the masters' own (shoulder_elv*0.242 <= 0.75)
        assert!(
            b[s][1].1 < 3.1 && b[s][1].1 > 3.0,
            "shoulder_elv eff hi {}",
            b[s][1].1
        );
        let mut corners = [[0.0; 4]; 2];
        for k in 0..4 {
            corners[s][k] = b[s][k].1;
        }
        let q = pose(&m, corners);
        check_invariants(&m, &q);
    }
}

#[test]
fn recovers_asymmetric_pose_without_reflection() {
    let m = model();
    // LEFT: abducted, mild flexion. RIGHT: hanging, elbow strongly flexed. Torso yawed 25 deg,
    // subject 15% larger than the reference.
    let truth = [[0.2, 1.4, 0.3, 0.4], [0.6, 0.35, 0.1, 1.6]];
    let q_true = pose(&m, truth);
    let cam = Cam::new(&m, 25f64.to_radians(), 1.15);
    let mut f = LiveFitter::new(m.clone()).unwrap();
    let mut last = None;
    for i in 0..40 {
        let r = f
            .update(&observe(&m, &cam, &q_true, i as f64 / 30.0))
            .unwrap();
        assert_eq!(r.supported_arms, [true, true], "{}", r.status);
        assert!(r.iterations <= 2 * MAX_ITERATIONS);
        check_invariants(&m, &r.qpos);
        last = Some(r);
    }
    let r = last.unwrap();
    let x = masters(&m, &r.qpos);
    let err = offset_err(&m, &r.qpos, &q_true);
    println!(
        "recovered {x:?}\ntruth     {truth:?}\noffset err {err:?} rmse {:?} status {}",
        r.rmse_m, r.status
    );
    // The fit itself converges to ~mm (rmse_m); the remaining few cm vs truth are the documented
    // approximation of the contract's STANDING model basis: asymmetric scapular elevation tilts the
    // observed shoulder line (camera basis) relative to the standing GH line (model basis).
    assert!(r.rmse_m.unwrap() < 0.01, "fit residual {:?}", r.rmse_m);
    assert!(err[0] < 0.05 && err[1] < 0.05, "offset errors {err:?}");
    // no left/right reflection: the abducted arm is the LEFT model arm, the bent elbow the RIGHT
    assert!(
        x[0][1] > 1.1 && x[1][1] < 0.7,
        "elevations {:?}",
        (x[0][1], x[1][1])
    );
    assert!(
        x[1][3] > 1.3 && x[0][3] < 0.8,
        "flexions {:?}",
        (x[0][3], x[1][3])
    );
    assert!((x[0][1] - truth[0][1]).abs() < 0.15 && (x[1][3] - truth[1][3]).abs() < 0.15);
}

#[test]
fn mirrored_input_is_not_silently_accepted_as_same_pose() {
    // Swapping the L/R landmark NAMES (a mirrored feed) must move the opposite model arm.
    let m = model();
    let q_true = pose(&m, [[0.2, 1.4, 0.3, 0.4], [0.0, 0.1, 0.0, 0.0]]);
    let cam = Cam::new(&m, 0.0, 1.0);
    let mut f = LiveFitter::new(m.clone()).unwrap();
    let mut r = None;
    for i in 0..30 {
        let mut o = observe(&m, &cam, &q_true, i as f64 / 30.0);
        for l in &mut o.landmarks {
            l.name = if l.name.starts_with("left") {
                l.name.replacen("left", "right", 1)
            } else {
                l.name.replacen("right", "left", 1)
            };
        }
        r = Some(f.update(&o).unwrap());
    }
    let x = masters(&m, &r.unwrap().qpos);
    assert!(
        x[1][1] > 1.1 && x[0][1] < 0.5,
        "elevations {:?}",
        (x[0][1], x[1][1])
    );
}

#[test]
fn tracks_motion_sequence_with_exact_constraints() {
    let m = model();
    let cam = Cam::new(&m, -15f64.to_radians(), 0.9);
    let mut f = LiveFitter::new(m.clone()).unwrap();
    let mut worst_late = 0.0f64;
    let mut moved = [0.0f64; 2];
    let mut prev = masters(&m, f.qpos());
    for i in 0..120 {
        let t = i as f64 / 30.0;
        let ph = 2.0 * std::f64::consts::PI * t / 3.0;
        let truth = [
            [
                0.4 + 0.5 * ph.sin(),
                0.9 + 0.7 * ph.sin(),
                0.2,
                0.6 + 0.5 * (ph * 1.3).sin(),
            ],
            [
                1.0 + 0.4 * ph.cos(),
                0.6 + 0.4 * (1.0 - ph.cos()),
                -0.2,
                1.2 + 0.6 * ph.sin(),
            ],
        ];
        let q_true = pose(&m, truth);
        let r = f.update(&observe(&m, &cam, &q_true, t)).unwrap();
        check_invariants(&m, &r.qpos);
        assert_eq!(r.supported_arms, [true, true]);
        let x = masters(&m, &r.qpos);
        for s in 0..2 {
            moved[s] += (0..4).map(|k| (x[s][k] - prev[s][k]).abs()).sum::<f64>();
        }
        prev = x;
        if i >= 15 {
            let e = offset_err(&m, &r.qpos, &q_true);
            worst_late = worst_late.max(e[0]).max(e[1]);
        }
    }
    println!("sequence worst offset error after warm-up {worst_late:.4} m, motion {moved:?}");
    assert!(worst_late < 0.05, "tracking error {worst_late}");
    assert!(
        moved[0] > 2.0 && moved[1] > 2.0,
        "fit did not follow motion {moved:?}"
    );
}

#[test]
fn unsupported_side_holds_and_other_side_moves() {
    let m = model();
    let cam = Cam::new(&m, 0.0, 1.0);
    let mut f = LiveFitter::new(m.clone()).unwrap();
    let a = pose(&m, [[0.3, 1.2, 0.0, 0.5], [0.5, 0.8, 0.2, 1.0]]);
    for i in 0..20 {
        f.update(&observe(&m, &cam, &a, i as f64 / 30.0)).unwrap();
    }
    let held = masters(&m, f.qpos())[1];
    let b = pose(&m, [[0.0, 0.4, 0.0, 1.5], [1.5, 2.0, -0.5, 0.1]]);
    for (k, spoil) in [
        |l: &mut Landmark| l.confidence = 0.1,
        |l: &mut Landmark| l.position = [f64::NAN, 0.0, 0.0],
        |l: &mut Landmark| l.confidence = f64::NAN,
    ]
    .iter()
    .enumerate()
    {
        let mut o = observe(&m, &cam, &b, 1.0 + k as f64);
        set_lm(&mut o, "right_elbow", spoil);
        let r = f.update(&o).unwrap();
        assert_eq!(r.supported_arms, [true, false], "{}", r.status);
        assert!(r.status.contains("R:held"), "{}", r.status);
        assert_eq!(masters(&m, &r.qpos)[1], held, "unsupported right arm moved");
        check_invariants(&m, &r.qpos);
    }
    let x = masters(&m, f.qpos());
    assert!(
        x[0][3] > 1.0,
        "supported left arm should follow: {:?}",
        x[0]
    );

    // Missing wrist: rotation and flexion are unobserved and held exactly.
    let before = masters(&m, f.qpos())[0];
    let mut o = observe(&m, &cam, &a, 10.0);
    set_lm(&mut o, "left_wrist", |l| l.confidence = 0.0);
    let r = f.update(&o).unwrap();
    let after = masters(&m, &r.qpos)[0];
    assert!(r.status.contains("elbow-only"), "{}", r.status);
    assert_eq!((after[2], after[3]), (before[2], before[3]));
    assert!(
        after[1] != before[1],
        "elevation should still follow the elbow"
    );
}

#[test]
fn missing_opposite_shoulder_does_not_interrupt_visible_arm() {
    let m = model();
    let cam = Cam::new(&m, 0.0, 1.0);
    let mut f = LiveFitter::new(m.clone()).unwrap();
    let initial = pose(&m, [[0.3, 1.2, 0.0, 0.5], [0.5, 0.8, 0.2, 1.0]]);
    for i in 0..10 {
        f.update(&observe(&m, &cam, &initial, i as f64 / 30.0))
            .unwrap();
    }
    let held = masters(&m, f.qpos())[1];
    let mut previous = masters(&m, f.qpos())[0];
    for i in 10..40 {
        let moving = pose(
            &m,
            [
                [0.3, 1.2 + (i - 10) as f64 * 0.015, 0.0, 0.5],
                [1.5, 2.0, -0.5, 0.1],
            ],
        );
        let mut obs = observe(&m, &cam, &moving, i as f64 / 30.0);
        for l in &mut obs.landmarks {
            if l.name.starts_with("right") {
                l.confidence = 0.0;
                l.position = [i as f64 * 100.0, -999.0, 999.0];
            }
        }
        let fit = f.update(&obs).unwrap();
        assert_eq!(fit.supported_arms, [true, false]);
        let x = masters(&m, &fit.qpos);
        assert_eq!(x[1], held);
        assert!((x[0][1] - previous[1]).abs() < 0.15, "visible arm jumped");
        previous = x[0];
        check_invariants(&m, &fit.qpos);
    }
    assert!(previous[1] > 1.4, "visible arm stopped following");
    let q = f.qpos().to_vec();
    let empty = f
        .update(&Observation {
            timestamp_seconds: 2.0,
            landmarks: vec![],
        })
        .unwrap();
    assert_eq!(empty.qpos, q);
    assert_eq!(empty.supported_arms, [false, false]);
}

#[test]
fn invalid_timestamp_rejected_and_reset_restores_standing() {
    let m = model();
    let cam = Cam::new(&m, 0.0, 1.0);
    let mut f = LiveFitter::new(m.clone()).unwrap();
    assert_eq!(f.qpos(), standing(&m).as_slice());
    let a = pose(&m, [[0.3, 1.2, 0.0, 0.5], [0.5, 0.8, 0.2, 1.0]]);
    f.update(&observe(&m, &cam, &a, 0.0)).unwrap();
    let q0 = f.qpos().to_vec();
    assert!(f.update(&observe(&m, &cam, &a, f64::NAN)).is_err());
    assert!(f.update(&observe(&m, &cam, &a, f64::INFINITY)).is_err());
    assert_eq!(f.qpos(), q0.as_slice());
    f.reset();
    assert_eq!(f.qpos(), standing(&m).as_slice());
}

#[test]
fn extreme_targets_stay_within_bounds_and_rate_limited() {
    let m = model();
    let mut f = LiveFitter::new(m.clone()).unwrap();
    // Arms straight overhead and slightly behind, wrists folded back: beyond reachable ranges.
    let mk = |t: f64| {
        let pt = |n: &str, p: V3| Landmark {
            name: n.into(),
            position: p,
            confidence: 1.0,
        };
        Observation {
            timestamp_seconds: t,
            landmarks: vec![
                pt("left_shoulder", [0.18, -0.4, 0.0]),
                pt("right_shoulder", [-0.18, -0.4, 0.0]),
                pt("left_elbow", [0.2, -0.7, 0.05]),
                pt("right_elbow", [-0.2, -0.7, 0.05]),
                pt("left_wrist", [0.2, -0.75, 0.3]),
                pt("right_wrist", [-0.2, -0.75, 0.3]),
            ],
        }
    };
    let mut prev = masters(&m, f.qpos());
    let mut prev_t: Option<f64> = None;
    for i in 0..60 {
        let t = i as f64 / 30.0;
        let r = f.update(&mk(t)).unwrap();
        check_invariants(&m, &r.qpos);
        let x = masters(&m, &r.qpos);
        let dt = prev_t.map_or(0.5, |p| t - p);
        for s in 0..2 {
            for k in 0..4 {
                assert!(
                    (x[s][k] - prev[s][k]).abs() <= 10.0 * dt + 1e-9,
                    "rate limit"
                );
            }
        }
        prev = x;
        prev_t = Some(t);
    }
    let x = masters(&m, f.qpos());
    assert!(
        x[0][1] > 2.5 && x[1][1] > 2.5,
        "arms should be elevated overhead: {x:?}"
    );
}

#[test]
fn update_latency() {
    let m = model();
    let cam = Cam::new(&m, 0.2, 1.0);
    let mut f = LiveFitter::new(m.clone()).unwrap();
    let obs: Vec<Observation> = (0..90)
        .map(|i| {
            let t = i as f64 / 30.0;
            let ph = 2.0 * std::f64::consts::PI * t / 2.0;
            observe(
                &m,
                &cam,
                &pose(
                    &m,
                    [
                        [
                            0.5 * ph.sin(),
                            1.0 + 0.8 * ph.sin(),
                            0.3,
                            1.0 + 0.8 * ph.cos(),
                        ],
                        [0.8, 0.5 + 0.4 * ph.cos(), 0.0, 0.8 + 0.6 * ph.sin()],
                    ],
                ),
                t,
            )
        })
        .collect();
    let start = std::time::Instant::now();
    let mut worst = 0.0f64;
    for o in &obs {
        let t0 = std::time::Instant::now();
        f.update(o).unwrap();
        worst = worst.max(t0.elapsed().as_secs_f64());
    }
    let mean = start.elapsed().as_secs_f64() / obs.len() as f64;
    println!(
        "LiveFitter::update mean {:.3} ms worst {:.3} ms",
        mean * 1e3,
        worst * 1e3
    );
    if !cfg!(debug_assertions) {
        assert!(mean < 0.010, "mean update {mean} s exceeds 10 ms");
    }
}

#[test]
fn one_arm_can_start_without_ever_observing_the_other_shoulder() {
    let m = model();
    let cam = Cam::new(&m, 0.0, 1.0);
    for missing in ["left", "right"] {
        let mut fitter = LiveFitter::new(m.clone()).unwrap();
        let truth = pose(&m, [[0.3, 1.3, 0.2, 0.8], [0.3, 1.3, 0.2, 0.8]]);
        for i in 0..15 {
            let mut obs = observe(&m, &cam, &truth, i as f64 / 30.0);
            obs.landmarks.retain(|l| !l.name.starts_with(missing));
            let result = fitter.update(&obs).unwrap();
            assert_eq!(
                result.supported_arms,
                [missing != "left", missing != "right"]
            );
        }
    }
}

#[test]
fn stationary_targets_stop_solver_creep_and_release_on_deliberate_motion() {
    let m = model();
    let cam = Cam::new(&m, 0.0, 1.0);
    let mut f = LiveFitter::new(m.clone()).unwrap();
    let q = pose(&m, [[0.3, 1.2, 0.2, 0.8], [0.5, 0.6, 0.1, 1.2]]);
    for i in 0..120 {
        f.update(&observe(&m, &cam, &q, i as f64 / 30.0)).unwrap();
    }
    let held = f.qpos().to_vec();
    for i in 120..180 {
        let result = f.update(&observe(&m, &cam, &q, i as f64 / 30.0)).unwrap();
        assert_eq!(result.qpos, held, "stationary input moved at frame {i}");
    }
    let moving = pose(&m, [[0.3, 1.8, 0.2, 0.8], [0.5, 0.6, 0.1, 1.2]]);
    for i in 180..186 {
        f.update(&observe(&m, &cam, &moving, i as f64 / 30.0))
            .unwrap();
    }
    assert!(
        (masters(&m, f.qpos())[0][1] - masters(&m, &held)[0][1]).abs() > 0.15,
        "stillness lock prevented movement"
    );
}
