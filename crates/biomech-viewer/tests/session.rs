use biomech_lite::{JNT_FREE, Model};
use biomech_viewer::session::{Motion, SimCommand, Simulation, slerp};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn assets() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/biomech-lite")
}

fn model() -> Model {
    Model::load(assets().join("model.json")).expect("model")
}

fn two_frame_motion(m: &Model, t1: f64) -> Motion {
    let a = m.standing_qpos.clone();
    let mut b = a.clone();
    // move the first scalar (non-free, non-dependent) dynamic joint
    let adr = m.joints[m.dynamic_joints[1]].qposadr;
    b[adr] = a[adr] + 0.2;
    Motion::new(vec![1.0, 1.0 + t1], vec![0, 1], vec![a, b], m).unwrap()
}

#[test]
fn loads_real_motion_and_clamps_endpoints() {
    let m = model();
    let mo = Motion::load(assets().join("motion.json"), &m).unwrap();
    assert!(mo.duration() > 0.0);
    let first = mo.sample(&m, -5.0).unwrap();
    let last = mo.sample(&m, mo.duration() + 5.0).unwrap();
    let mut f0 = mo.qpos[0].clone();
    m.project(&mut f0).unwrap();
    let mut fl = mo.qpos.last().unwrap().clone();
    m.project(&mut fl).unwrap();
    assert_eq!(first, f0);
    assert_eq!(last, fl);
    // exact recorded stamp returns the recorded frame (last frame preserved)
    let k = mo.len() / 2;
    let s = mo.sample(&m, mo.frame_time(k)).unwrap();
    for (x, y) in s.iter().zip(&mo.qpos[k]) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
    assert_eq!(mo.frame_at(mo.duration()), mo.len() - 1);
}

#[test]
fn irregular_clock_interpolates_by_time() {
    let m = model();
    let mo = two_frame_motion(&m, 4.0);
    let adr = m.joints[m.dynamic_joints[1]].qposadr;
    let mid = mo.sample(&m, 1.0).unwrap(); // 25% of the span
    let expect = mo.qpos[0][adr] + 0.25 * 0.2;
    assert!((mid[adr] - expect).abs() < 1e-12);
}

#[test]
fn couplings_are_exactly_reprojected() {
    let m = model();
    let mo = Motion::load(assets().join("motion.json"), &m).unwrap();
    for k in [0, 1, mo.len() / 3] {
        let t = 0.5 * (mo.frame_time(k) + mo.frame_time(k + 1));
        let q = mo.sample(&m, t).unwrap();
        let res = m.coupling_residuals(&q);
        assert!(res.iter().all(|r| r.abs() < 1e-9), "residual {:?}", res.iter().fold(0.0f64, |a, r| a.max(r.abs())));
    }
}

#[test]
fn slerp_takes_shortest_path_and_stays_unit() {
    let a = [1.0, 0.0, 0.0, 0.0];
    let half = std::f64::consts::FRAC_PI_4; // 90 degrees about z
    let b = [half.cos(), 0.0, 0.0, half.sin()];
    let nb = [-b[0], -b[1], -b[2], -b[3]]; // same rotation, opposite sign
    let r = slerp(&a, &nb, 0.5);
    let n: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((n - 1.0).abs() < 1e-12);
    let h = (std::f64::consts::FRAC_PI_4 / 2.0).sin();
    assert!((r[3].abs() - h).abs() < 1e-12, "{r:?}");
    assert!(r[0] > 0.0);
}

#[test]
fn free_root_quaternion_slerped_in_sample() {
    let m = model();
    let free = m.joints.iter().find(|j| j.kind == JNT_FREE).map(|j| j.qposadr);
    let Some(adr) = free else { return };
    let mut a = m.standing_qpos.clone();
    let mut b = a.clone();
    a[adr + 3..adr + 7].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    let h = std::f64::consts::FRAC_PI_4;
    b[adr + 3..adr + 7].copy_from_slice(&[-h.cos(), 0.0, 0.0, -h.sin()]);
    let mo = Motion::new(vec![0.0, 1.0], vec![0, 1], vec![a, b], &m).unwrap();
    let q = mo.sample(&m, 0.5).unwrap();
    let n: f64 = q[adr + 3..adr + 7].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((n - 1.0).abs() < 1e-12);
    assert!(q[adr + 3] > 0.9, "took long path: {:?}", &q[adr + 3..adr + 7]);
}

#[test]
fn malformed_motion_rejected() {
    let m = model();
    let q = m.standing_qpos.clone();
    assert!(Motion::new(vec![], vec![], vec![], &m).is_err());
    assert!(Motion::new(vec![0.0, 0.0], vec![0, 1], vec![q.clone(), q.clone()], &m).is_err());
    assert!(Motion::new(vec![0.0, 1.0], vec![1, 1], vec![q.clone(), q.clone()], &m).is_err());
    assert!(Motion::new(vec![0.0], vec![0], vec![q[..q.len() - 1].to_vec()], &m).is_err());
    let mut nan = q.clone();
    nan[0] = f64::NAN;
    assert!(Motion::new(vec![0.0], vec![0], vec![nan], &m).is_err());
    if let Some(adr) = m.joints.iter().find(|j| j.kind == JNT_FREE).map(|j| j.qposadr) {
        let mut bad = q.clone();
        bad[adr + 3..adr + 7].copy_from_slice(&[2.0, 0.0, 0.0, 0.0]);
        assert!(Motion::new(vec![0.0], vec![0], vec![bad], &m).is_err());
    }
}

fn wait_snapshot(sim: &mut Simulation, secs: f64) -> Option<biomech_viewer::session::SimSnapshot> {
    let end = Instant::now() + Duration::from_secs_f64(secs);
    while Instant::now() < end {
        if let Some(s) = sim.latest() {
            return Some(s);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    None
}

#[test]
fn simulation_starts_paused_runs_and_resets() {
    let m = Arc::new(model());
    let mut sim = Simulation::spawn(Arc::clone(&m)).unwrap();
    let s0 = wait_snapshot(&mut sim, 2.0).expect("initial snapshot");
    assert_eq!(s0.time_seconds, 0.0);
    // paused: no time advance
    std::thread::sleep(Duration::from_millis(200));
    assert!(sim.latest().is_none());
    sim.send(SimCommand::Pause(false)).unwrap();
    let mut last = None;
    let end = Instant::now() + Duration::from_secs(5);
    while Instant::now() < end {
        if let Some(s) = sim.latest() {
            let t = s.time_seconds;
            last = Some(s);
            if t >= 0.04 {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let s = last.expect("running snapshot");
    assert!(s.time_seconds >= 0.02, "t={}", s.time_seconds);
    assert!(s.error.is_none());
    assert_eq!(s.qpos.len(), m.nq);
    sim.send(SimCommand::Pause(true)).unwrap();
    // drain until paused snapshot, then verify it freezes
    std::thread::sleep(Duration::from_millis(300));
    let _ = sim.latest();
    std::thread::sleep(Duration::from_millis(200));
    assert!(sim.latest().is_none(), "paused simulation kept publishing");
    sim.send(SimCommand::Reset).unwrap();
    let r = wait_snapshot(&mut sim, 2.0).expect("reset snapshot");
    assert_eq!(r.time_seconds, 0.0);
    assert!(r.activation.iter().all(|a| *a == 0.0));
}

#[test]
fn simulation_shutdown_is_prompt_even_while_running() {
    let m = Arc::new(model());
    let sim = Simulation::spawn(m).unwrap();
    sim.send(SimCommand::Pause(false)).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let t = Instant::now();
    drop(sim);
    assert!(t.elapsed() < Duration::from_secs(2), "drop took {:?}", t.elapsed());
}
