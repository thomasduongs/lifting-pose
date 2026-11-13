//! Asynchronous, latest-only muscle recruitment for the live anatomical twin.
//!
//! One worker thread consumes a single pending sample (newer samples replace older ones), so
//! there is no unbounded history. The worker owns all derivative history; callers only see the
//! published `State`. Generation tokens make pre-reset work unable to publish after `reset`.
//! No camera, window or UI access happens here.

use biomech_lite::inverse::{Recruiter, Recruitment};
use biomech_lite::Model;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Instant;

const NX: usize = 8;
const ARM_COORDS: usize = NX / 2;
/// Velocity above this magnitude (rad/s) is an implausible input, not motion.
const MAX_SPEED: f64 = 10.0;
/// Acceleration above this magnitude (rad/s^2) is an implausible input, not motion.
const MAX_ACCEL: f64 = 120.0;
/// Gaps longer than this restart derivative history on that side.
const GAP_RESET_S: f64 = 0.5;
/// Consecutive valid samples a side needs before its derivatives are used.
const WARM_SAMPLES: usize = 3;
const VEL_TAU_S: f64 = 0.08;
const ACC_TAU_S: f64 = 0.12;

/// One pose observation handed to the worker.
#[derive(Clone)]
pub struct Sample {
    pub index: u64,
    pub captured: Instant,
    pub timestamp_seconds: f64,
    pub qpos: Vec<f64>,
    /// [LEFT, RIGHT]
    pub supported_arms: [bool; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EstimateMode {
    /// Static (zero-velocity) solve: first samples, gaps, or a side not yet warmed.
    QuasiStaticWarmup,
    /// Velocity and acceleration from finite differences of every supported side.
    Dynamic,
}

/// A recruitment result plus the sample it was computed from.
#[derive(Clone)]
pub struct Estimate {
    pub sample: Sample,
    pub recruitment: Recruitment,
    pub mode: EstimateMode,
    pub computation_ms: f64,
    /// Per-arm derivative rejections for this frame (the other arm may still be valid).
    /// `sample.supported_arms` holds the EFFECTIVE arms that were actually solved.
    pub warnings: Vec<String>,
}

#[derive(Clone, Default)]
pub struct State {
    pub estimate: Option<Arc<Estimate>>,
    pub error: Option<String>,
    pub computing: bool,
}

type Inner = (Mutex<Shared>, Condvar);

struct Shared {
    generation: u64,
    pending: Option<(u64, Sample)>,
    state: State,
    shutdown: bool,
}

pub struct Estimator {
    inner: Arc<Inner>,
    handle: Option<JoinHandle<()>>,
}

impl Estimator {
    pub fn start(model: Arc<Model>) -> Result<Self, String> {
        let layout = layout(&model)?;
        let recruiter = Recruiter::new(model.clone())?;
        let inner: Arc<Inner> = Arc::new((
            Mutex::new(Shared {
                generation: 0,
                pending: None,
                state: State::default(),
                shutdown: false,
            }),
            Condvar::new(),
        ));
        let worker_inner = inner.clone();
        let handle = std::thread::Builder::new()
            .name("biomech-estimate".into())
            .spawn(move || run(model, layout, recruiter, worker_inner))
            .map_err(|e| format!("cannot start estimate worker: {e}"))?;
        Ok(Self {
            inner,
            handle: Some(handle),
        })
    }

    /// Replaces any pending sample; the worker only ever computes the newest one.
    pub fn submit(&self, sample: Sample) {
        let (mutex, cv) = &*self.inner;
        let mut shared = lock(mutex);
        let generation = shared.generation;
        shared.pending = Some((generation, sample));
        shared.state.computing = true;
        cv.notify_one();
    }

    pub fn latest(&self) -> State {
        lock(&self.inner.0).state.clone()
    }

    /// Drops pending work and warm state; work already running cannot publish.
    pub fn reset(&self) {
        let (mutex, _) = &*self.inner;
        let mut shared = lock(mutex);
        shared.generation += 1;
        shared.pending = None;
        shared.state = State::default();
    }
}

impl Drop for Estimator {
    fn drop(&mut self) {
        {
            let mut shared = lock(&self.inner.0);
            shared.shutdown = true;
            shared.pending = None;
        }
        self.inner.1.notify_all();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn lock(mutex: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// One dynamic coordinate: its index in `model.dynamic_joints` plus its state addresses.
#[derive(Clone, Copy, Debug)]
struct Coord {
    k: usize,
    qposadr: usize,
}

/// Maps the eight dynamic coordinates to [LEFT, RIGHT] by joint-name suffix, in the order
/// `model.dynamic_joints` lists them (not assumed to be left-first or right-first).
fn layout(model: &Model) -> Result<[Vec<Coord>; 2], String> {
    if model.dynamic_joints.len() != NX {
        return Err(format!(
            "model has {} dynamic joints, expected {NX}",
            model.dynamic_joints.len()
        ));
    }
    let mut sides: [Vec<Coord>; 2] = [Vec::new(), Vec::new()];
    for (k, &j) in model.dynamic_joints.iter().enumerate() {
        let joint = model
            .joints
            .get(j)
            .ok_or_else(|| format!("dynamic joint index {j} out of range"))?;
        let side = if joint.name.ends_with("_l") {
            0
        } else if joint.name.ends_with("_r") {
            1
        } else {
            return Err(format!("dynamic joint {} has no _l/_r suffix", joint.name));
        };
        sides[side].push(Coord {
            k,
            qposadr: joint.qposadr,
        });
    }
    if sides.iter().any(|side| side.len() != ARM_COORDS) {
        return Err(format!("each arm must have exactly {ARM_COORDS} dynamic coordinates"));
    }
    Ok(sides)
}

/// Finite-difference history for one arm. Velocity is an EMA of finite differences at capture
/// timestamps; acceleration is an EMA of differences of that filtered velocity.
#[derive(Default)]
struct SideHistory {
    /// Consecutive valid samples, including the most recent one.
    streak: usize,
    last_time: f64,
    last_q: Vec<f64>,
    vel: Vec<f64>,
    acc: Vec<f64>,
}

impl SideHistory {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn warm(&self) -> bool {
        self.streak >= WARM_SAMPLES
    }

    /// Advances with one valid sample. Any error leaves this arm's history reset.
    fn advance(&mut self, t: f64, q: &[f64]) -> Result<(), String> {
        let n = q.len();
        if self.streak == 0 || t - self.last_time > GAP_RESET_S {
            *self = Self {
                streak: 1,
                last_time: t,
                last_q: q.to_vec(),
                vel: vec![0.0; n],
                acc: vec![0.0; n],
            };
            return Ok(());
        }
        let dt = t - self.last_time;
        if dt <= 0.0 {
            self.reset();
            return Err(format!("duplicate or reversed timestamp (dt {dt:.6} s)"));
        }
        let alpha_v = dt / (VEL_TAU_S + dt);
        let alpha_a = dt / (ACC_TAU_S + dt);
        let mut vel = vec![0.0; n];
        for i in 0..n {
            let raw = (q[i] - self.last_q[i]) / dt;
            if !raw.is_finite() || raw.abs() > MAX_SPEED {
                self.reset();
                return Err(format!(
                    "coordinate {i} speed {raw:.2} rad/s exceeds {MAX_SPEED} rad/s"
                ));
            }
            vel[i] = if self.streak == 1 {
                raw
            } else {
                self.vel[i] + alpha_v * (raw - self.vel[i])
            };
        }
        let mut acc = vec![0.0; n];
        if self.streak >= 2 {
            for i in 0..n {
                let raw = (vel[i] - self.vel[i]) / dt;
                if !raw.is_finite() || raw.abs() > MAX_ACCEL {
                    self.reset();
                    return Err(format!(
                        "coordinate {i} acceleration {raw:.1} rad/s^2 exceeds {MAX_ACCEL} rad/s^2"
                    ));
                }
                acc[i] = if self.streak == 2 {
                    raw
                } else {
                    self.acc[i] + alpha_a * (raw - self.acc[i])
                };
            }
        }
        self.streak += 1;
        self.last_time = t;
        self.last_q = q.to_vec();
        self.vel = vel;
        self.acc = acc;
        Ok(())
    }
}

const SIDE_NAMES: [&str; 2] = ["left", "right"];

/// Result of one frame: the solve plus which arms were actually used and why others were not.
struct Frame {
    recruitment: Recruitment,
    mode: EstimateMode,
    effective: [bool; 2],
    warnings: Vec<String>,
}

/// One frame: validate globally (fatal on failure), update per-arm history (a failing arm is
/// masked and reset, the other arm still solves), then solve with each warmed arm's own
/// derivatives. Returns `Ok(None)` when no arm is supported.
fn solve_frame(
    model: &Model,
    layout: &[Vec<Coord>; 2],
    recruiter: &mut Recruiter,
    histories: &mut [SideHistory; 2],
    sample: &Sample,
) -> Result<Option<Frame>, String> {
    if sample.qpos.len() != model.nq {
        return Err(format!(
            "qpos length {} does not match model nq {}",
            sample.qpos.len(),
            model.nq
        ));
    }
    if !sample.timestamp_seconds.is_finite() {
        return Err("non-finite sample timestamp".into());
    }
    if sample.qpos.iter().any(|x| !x.is_finite()) {
        return Err("non-finite qpos".into());
    }
    let mut q = sample.qpos.clone();
    model.project(&mut q)?;

    let mut effective = [false; 2];
    let mut warnings = Vec::new();
    let mut any_supported = false;
    for side in 0..2 {
        if !sample.supported_arms[side] {
            histories[side].reset();
            continue;
        }
        any_supported = true;
        let values: Vec<f64> = layout[side].iter().map(|c| q[c.qposadr]).collect();
        match histories[side].advance(sample.timestamp_seconds, &values) {
            Ok(()) => effective[side] = true,
            Err(e) => warnings.push(format!("{} arm masked: {e}", SIDE_NAMES[side])),
        }
    }
    if !any_supported {
        return Ok(None);
    }
    if !effective.iter().any(|e| *e) {
        return Err(format!("no valid arm: {}", warnings.join("; ")));
    }

    // Unobserved arms carry no information: hold them at the standing prior so their (possibly
    // implausible) values cannot make the recruiter reject the healthy arm's solve.
    let mut changed = false;
    for side in 0..2 {
        if !effective[side] {
            for c in &layout[side] {
                q[c.qposadr] = model.standing_qpos[c.qposadr];
                changed = true;
            }
        }
    }
    if changed {
        model.project(&mut q)?;
    }

    // Reduced layout: length NX, indexed by position in model.dynamic_joints.
    let mut qvel = vec![0.0; NX];
    let mut qacc = vec![0.0; NX];
    let mut observed = [false; NX];
    let mut all_warm = true;
    for side in 0..2 {
        if !effective[side] {
            continue;
        }
        let warm = histories[side].warm();
        all_warm &= warm;
        for (i, c) in layout[side].iter().enumerate() {
            observed[c.k] = true;
            if warm {
                qvel[c.k] = histories[side].vel[i];
                qacc[c.k] = histories[side].acc[i];
            }
        }
    }
    let mode = if all_warm {
        EstimateMode::Dynamic
    } else {
        EstimateMode::QuasiStaticWarmup
    };
    let recruitment = recruiter.solve(&q, &qvel, &qacc, observed)?;
    Ok(Some(Frame {
        recruitment,
        mode,
        effective,
        warnings,
    }))
}

fn run(model: Arc<Model>, layout: [Vec<Coord>; 2], mut recruiter: Recruiter, inner: Arc<Inner>) {
    let mut histories: [SideHistory; 2] = Default::default();
    let mut worker_generation = 0u64;
    loop {
        let (generation, sample) = {
            let (mutex, cv) = &*inner;
            let mut shared = lock(mutex);
            loop {
                if shared.shutdown {
                    return;
                }
                if let Some(job) = shared.pending.take() {
                    break job;
                }
                shared = cv.wait(shared).unwrap_or_else(|e| e.into_inner());
            }
        };

        if generation != worker_generation {
            worker_generation = generation;
            histories = Default::default();
            recruiter.reset();
        }

        let started = Instant::now();
        let outcome = solve_frame(&model, &layout, &mut recruiter, &mut histories, &sample);
        if outcome.is_err() {
            // Fatal frame: never let it poison the next derivative.
            histories = Default::default();
            recruiter.reset();
        }
        let computation_ms = started.elapsed().as_secs_f64() * 1e3;

        let (mutex, _) = &*inner;
        let mut shared = lock(mutex);
        if shared.generation != generation {
            continue;
        }
        shared.state.computing = shared.pending.is_some();
        match outcome {
            Ok(Some(frame)) => {
                let mut sample = sample;
                sample.supported_arms = frame.effective;
                shared.state.estimate = Some(Arc::new(Estimate {
                    sample,
                    recruitment: frame.recruitment,
                    mode: frame.mode,
                    computation_ms,
                    warnings: frame.warnings,
                }));
                shared.state.error = None;
            }
            Ok(None) => {
                shared.state.estimate = None;
                shared.state.error = None;
            }
            Err(message) => {
                shared.state.estimate = None;
                shared.state.error = Some(message);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Arc<Model> {
        Arc::new(
            Model::load(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/biomech-lite/model.json"))
                .expect("load biomech-lite model"),
        )
    }

    fn sample(model: &Model, t: f64, supported: [bool; 2]) -> Sample {
        Sample {
            index: 0,
            captured: Instant::now(),
            timestamp_seconds: t,
            qpos: model.qpos0.clone(),
            supported_arms: supported,
        }
    }

    #[test]
    fn layout_follows_joint_names_not_array_order() {
        let model = model();
        let sides = layout(&model).expect("layout");
        let mut seen = [false; NX];
        for (side, coords) in sides.iter().enumerate() {
            let suffix = if side == 0 { "_l" } else { "_r" };
            for c in coords {
                let name = &model.joints[model.dynamic_joints[c.k]].name;
                assert!(name.ends_with(suffix), "{name} on wrong side");
                assert!(!seen[c.k], "coordinate {} mapped twice", c.k);
                seen[c.k] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "every dynamic coordinate is mapped");
    }

    #[test]
    fn static_samples_have_zero_derivatives_and_warm_up() {
        let mut h = SideHistory::default();
        let q = [0.1, 0.2, 0.3, 0.4];
        for t in [0.0, 0.1, 0.2] {
            h.advance(t, &q).expect("advance");
        }
        assert!(h.warm());
        assert!(h.vel.iter().all(|&v| v == 0.0));
        assert!(h.acc.iter().all(|&a| a == 0.0));
    }

    #[test]
    fn gap_restarts_history_with_zero_velocity() {
        let mut h = SideHistory::default();
        let q = [0.0; 4];
        for t in [0.0, 0.1, 0.2] {
            h.advance(t, &q).expect("advance");
        }
        h.advance(1.0, &[0.5; 4]).expect("advance after gap");
        assert_eq!(h.streak, 1);
        assert!(!h.warm());
        assert!(h.vel.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn reversed_time_is_an_error_and_resets_the_arm() {
        let mut h = SideHistory::default();
        h.advance(0.0, &[0.0; 4]).expect("first");
        h.advance(0.1, &[0.0; 4]).expect("second");
        assert!(h.advance(0.05, &[0.0; 4]).is_err());
        assert_eq!(h.streak, 0);
    }

    #[test]
    fn missing_side_resets_only_that_side() {
        let model = model();
        let layout = layout(&model).expect("layout");
        let mut recruiter = Recruiter::new(model.clone()).expect("recruiter");
        let mut histories: [SideHistory; 2] = Default::default();
        for t in [0.0, 0.1, 0.2] {
            let s = sample(&model, t, [true, false]);
            solve_frame(&model, &layout, &mut recruiter, &mut histories, &s).expect("frame");
        }
        assert_eq!(histories[0].streak, 3);
        assert_eq!(histories[1].streak, 0);
    }

    #[test]
    fn dynamic_solve_uses_reduced_eight_layout() {
        let model = model();
        let layout = layout(&model).expect("layout");
        let mut recruiter = Recruiter::new(model.clone()).expect("recruiter");
        let mut histories: [SideHistory; 2] = Default::default();
        let mut last = None;
        for i in 0..4 {
            let mut s = sample(&model, i as f64 * 0.05, [true, true]);
            for side in &layout {
                let c = side[1];
                s.qpos[c.qposadr] += 0.002 * i as f64;
            }
            last = solve_frame(&model, &layout, &mut recruiter, &mut histories, &s).expect("frame");
        }
        let frame = last.expect("estimate");
        assert_eq!(frame.mode, EstimateMode::Dynamic);
        assert_eq!(frame.effective, [true, true]);
        assert_eq!(frame.recruitment.required_torque_nm.len(), NX);
    }

    #[test]
    fn one_arm_derivative_failure_masks_only_that_arm() {
        let model = model();
        let layout = layout(&model).expect("layout");
        let mut recruiter = Recruiter::new(model.clone()).expect("recruiter");
        let mut histories: [SideHistory; 2] = Default::default();
        for t in [0.0, 0.1, 0.2] {
            let s = sample(&model, t, [true, true]);
            solve_frame(&model, &layout, &mut recruiter, &mut histories, &s).expect("warm");
        }
        assert!(histories[0].warm() && histories[1].warm());
        // Throw the LEFT arm to the far end of every range: an implausible speed.
        let mut s = sample(&model, 0.3, [true, true]);
        for c in &layout[0] {
            let j = &model.joints[model.dynamic_joints[c.k]];
            let q = s.qpos[c.qposadr];
            s.qpos[c.qposadr] = if q - j.range[0] > j.range[1] - q { j.range[0] } else { j.range[1] };
        }
        let frame = solve_frame(&model, &layout, &mut recruiter, &mut histories, &s)
            .expect("frame")
            .expect("estimate");
        assert_eq!(frame.effective, [false, true], "only the right arm is valid");
        assert!(!frame.warnings.is_empty() && frame.warnings[0].contains("left"));
        assert_eq!(histories[0].streak, 0, "failed arm history is reset");
        assert!(histories[1].warm(), "healthy arm keeps its history");
        assert_eq!(frame.mode, EstimateMode::Dynamic, "remaining arm is warm");
        for (k, obs) in frame.recruitment.observed_coordinates.iter().enumerate() {
            assert_eq!(*obs, layout[1].iter().any(|c| c.k == k));
        }
    }

    #[test]
    fn warm_arm_keeps_derivatives_while_other_arm_warms() {
        let model = model();
        let layout = layout(&model).expect("layout");
        let mut recruiter = Recruiter::new(model.clone()).expect("recruiter");
        let mut histories: [SideHistory; 2] = Default::default();
        for t in [0.0, 0.1, 0.2] {
            let s = sample(&model, t, [false, true]);
            solve_frame(&model, &layout, &mut recruiter, &mut histories, &s).expect("right only");
        }
        let s = sample(&model, 0.3, [true, true]);
        let frame = solve_frame(&model, &layout, &mut recruiter, &mut histories, &s)
            .expect("frame")
            .expect("estimate");
        assert_eq!(frame.mode, EstimateMode::QuasiStaticWarmup, "left is cold");
        assert!(histories[1].warm());
    }

    #[test]
    fn worker_first_sample_produces_recruitment() {
        let model = model();
        let est = Estimator::start(model.clone()).expect("start");
        est.submit(sample(&model, 0.0, [true, true]));
        let begun = Instant::now();
        let state = loop {
            let state = est.latest();
            if state.estimate.is_some() || state.error.is_some() {
                break state;
            }
            assert!(begun.elapsed().as_secs() < 30, "worker never published");
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        assert!(state.error.is_none(), "{:?}", state.error);
        let e = state.estimate.expect("estimate");
        assert_eq!(e.mode, EstimateMode::QuasiStaticWarmup);
        assert_eq!(e.recruitment.activation.len(), model.muscles.len());
        assert_eq!(e.sample.supported_arms, [true, true]);
    }
}
