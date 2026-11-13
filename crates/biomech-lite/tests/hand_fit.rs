use biomech_lite::{
    Model,
    hand_fit::{HandFitter, HandObservation},
};
use std::{path::PathBuf, sync::Arc, time::Instant};
fn model() -> Arc<Model> {
    Arc::new(
        Model::load(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/biomech-lite/model.json"),
        )
        .unwrap(),
    )
}
fn error(a: &[[f64; 3]; 21], b: &[[f64; 3]; 21]) -> f64 {
    (0..5)
        .flat_map(|f| (0..3).map(move |i| (1 + f * 4 + i, 2 + f * 4 + i)))
        .map(|(i, j)| {
            let da: [f64; 3] = std::array::from_fn(|k| a[j][k] - a[i][k]);
            let db: [f64; 3] = std::array::from_fn(|k| b[j][k] - b[i][k]);
            let na = da.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb = db.iter().map(|x| x * x).sum::<f64>().sqrt();
            (0..3)
                .map(|k| (da[k] / na - db[k] / nb).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / 15.0
}
#[test]
fn tracks_all_five_digits_on_both_hands_and_holds_missing_side() {
    let m = model();
    let mut fitter = HandFitter::new(m.clone()).unwrap();
    let mut truth = m.standing_qpos.clone();
    // Names are inherited from the source model, including its unsuffixed right joints.
    let finger_joints: Vec<usize> = m
        .joints
        .iter()
        .enumerate()
        .filter(|(_, j)| {
            [
                "mcp",
                "pm2_",
                "pm3_",
                "pm4_",
                "pm5_",
                "md2_",
                "md3_",
                "md4_",
                "md5_",
                "cmc_",
                "mp_flexion",
                "ip_flexion",
            ]
            .iter()
            .any(|s| j.name.starts_with(s))
        })
        .map(|(i, _)| i)
        .collect();
    for &i in &finger_joints {
        let j = &m.joints[i];
        truth[j.qposadr] = if j.name.contains("abduction") {
            0.12
        } else if j.name.starts_with("ip_") {
            -0.55
        } else {
            0.65
        };
    }
    m.project(&mut truth).unwrap();
    let source = [
        fitter.landmarks(0, &truth).unwrap(),
        fitter.landmarks(1, &truth).unwrap(),
    ];
    // A proper camera rotation and arbitrary size/translation must leave local curls unchanged.
    let obs: Vec<_> = (0..2)
        .map(|side| HandObservation {
            side,
            confidence: 0.95,
            points: source[side].map(|p| [2.0 * p[2] + 3.0, 2.0 * p[1] - 4.0, -2.0 * p[0] + 5.0]),
        })
        .collect();
    let mut q = m.standing_qpos.clone();
    let initial = error(&fitter.landmarks(0, &q).unwrap(), &source[0]);
    let start = Instant::now();
    for i in 0..20 {
        assert_eq!(
            fitter.apply(&mut q, &obs, i as f64 / 30.0).unwrap(),
            [true, true]
        );
    }
    eprintln!(
        "two-hand IK average {:.2} ms",
        start.elapsed().as_secs_f64() * 1000.0 / 20.0
    );
    for side in 0..2 {
        let err = error(&fitter.landmarks(side, &q).unwrap(), &source[side]);
        assert!(
            err < 0.002,
            "side {side} direction error {err}, initial {initial}"
        );
    }
    for (i, j) in m.joints.iter().enumerate() {
        if !finger_joints.contains(&i) {
            assert_eq!(
                q[j.qposadr], m.standing_qpos[j.qposadr],
                "non-finger {} changed",
                j.name
            );
        }
        if j.limited {
            assert!((j.range[0] - 1e-8..=j.range[1] + 1e-8).contains(&q[j.qposadr]));
        }
    }
    let held = q.clone();
    // No observations: maintain the pose even if the arm fitter supplies a fresh standing prior.
    q = m.standing_qpos.clone();
    assert_eq!(fitter.apply(&mut q, &[], 1.0).unwrap(), [false, false]);
    assert_eq!(q, held);
    let mut only_left = obs[0].clone();
    only_left.points = fitter.landmarks(0, &m.standing_qpos).unwrap();
    for i in 0..20 {
        assert_eq!(
            fitter
                .apply(&mut q, &[only_left.clone()], 1.1 + i as f64 / 30.0)
                .unwrap(),
            [true, false]
        );
    }
    for &i in &finger_joints {
        let j = &m.joints[i];
        if j.body < m.joints[m.joint_id("flexion_l").unwrap()].body {
            assert_eq!(q[j.qposadr], held[j.qposadr]);
        }
    }
    assert!(
        error(
            &fitter.landmarks(0, &q).unwrap(),
            &fitter.landmarks(0, &m.standing_qpos).unwrap()
        ) < 0.002
    );
}
#[test]
fn rejects_weak_and_degenerate_hands_without_changing_fingers() {
    let m = model();
    let mut f = HandFitter::new(m.clone()).unwrap();
    let mut q = m.standing_qpos.clone();
    let valid = f.landmarks(0, &q).unwrap();
    for points in [[[0.0; 3]; 21], [[f64::NAN; 3]; 21]] {
        assert_eq!(
            f.apply(
                &mut q,
                &[HandObservation {
                    side: 0,
                    confidence: 0.95,
                    points
                }],
                0.0
            )
            .unwrap(),
            [false, false]
        );
    }
    assert_eq!(
        f.apply(
            &mut q,
            &[HandObservation {
                side: 0,
                confidence: 0.2,
                points: valid
            }],
            0.1
        )
        .unwrap(),
        [false, false]
    );
    assert_eq!(q, m.standing_qpos);
}
