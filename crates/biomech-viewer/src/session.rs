//! Recorded-motion playback (irregular clock, shortest-path SLERP, exact coupling projection)
//! and a background live simulation built on `biomech_lite::dynamics::step`.

use biomech_lite::dynamics::{self, State};
use biomech_lite::kinematics::{Quat, normalize4};
use biomech_lite::{JNT_FREE, Model};
use serde::Deserialize;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Recorded motion
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct MotionFile {
    schema_version: u32,
    timestamps_seconds: Vec<f64>,
    frame_indices: Vec<u64>,
    qpos: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct Motion {
    pub timestamps_seconds: Vec<f64>,
    pub frame_indices: Vec<u64>,
    pub qpos: Vec<Vec<f64>>,
}

fn free_adrs(model: &Model) -> Vec<usize> {
    model.joints.iter().filter(|j| j.kind == JNT_FREE).map(|j| j.qposadr).collect()
}

impl Motion {
    pub fn load(path: impl AsRef<Path>, model: &Model) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read motion {}: {e}", path.display()))?;
        let f: MotionFile = serde_json::from_str(&text)
            .map_err(|e| format!("invalid motion JSON {}: {e}", path.display()))?;
        if f.schema_version != 1 {
            return Err(format!("motion schema_version {} unsupported (expected 1)", f.schema_version));
        }
        Self::new(f.timestamps_seconds, f.frame_indices, f.qpos, model)
    }

    /// Validates with the same semantics as the core CLI.
    pub fn new(
        timestamps_seconds: Vec<f64>,
        frame_indices: Vec<u64>,
        qpos: Vec<Vec<f64>>,
        model: &Model,
    ) -> Result<Self, String> {
        let n = qpos.len();
        if n == 0 {
            return Err("motion has no frames".into());
        }
        if timestamps_seconds.len() != n || frame_indices.len() != n {
            return Err("motion arrays timestamps_seconds, frame_indices and qpos differ in length".into());
        }
        let free = free_adrs(model);
        for i in 0..n {
            let (t, q) = (timestamps_seconds[i], &qpos[i]);
            if !t.is_finite() {
                return Err(format!("motion frame {i}: non-finite timestamp"));
            }
            if i > 0 {
                if t <= timestamps_seconds[i - 1] {
                    return Err(format!("motion frame {i}: timestamps not strictly increasing"));
                }
                if frame_indices[i] <= frame_indices[i - 1] {
                    return Err(format!("motion frame {i}: frame_indices not strictly increasing"));
                }
            }
            if q.len() != model.nq {
                return Err(format!("motion frame {i}: qpos has {} entries, model nq={}", q.len(), model.nq));
            }
            if q.iter().any(|v| !v.is_finite()) {
                return Err(format!("motion frame {i}: non-finite qpos"));
            }
            for &a in &free {
                let norm = q[a + 3..a + 7].iter().map(|v| v * v).sum::<f64>().sqrt();
                if (norm - 1.0).abs() > 1e-6 {
                    return Err(format!("motion frame {i}: free-root quaternion norm {norm} is not unit"));
                }
            }
        }
        Ok(Motion { timestamps_seconds, frame_indices, qpos })
    }

    pub fn len(&self) -> usize {
        self.qpos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.qpos.is_empty()
    }

    /// Time span from first to last recorded stamp.
    pub fn duration(&self) -> f64 {
        match (self.timestamps_seconds.first(), self.timestamps_seconds.last()) {
            (Some(a), Some(b)) => b - a,
            _ => 0.0,
        }
    }

    /// Recorded time of frame `i` relative to the first stamp.
    pub fn frame_time(&self, i: usize) -> f64 {
        self.timestamps_seconds[i] - self.timestamps_seconds[0]
    }

    /// Index of the last frame whose time is <= `t` (clamped to valid range).
    pub fn frame_at(&self, t: f64) -> usize {
        let abs = self.timestamps_seconds[0] + t;
        self.timestamps_seconds.partition_point(|&s| s <= abs).saturating_sub(1)
    }

    /// Pose at `t` seconds after the first stamp. Endpoints clamp; scalar coordinates are
    /// linearly interpolated, free quaternions use shortest-path SLERP, then couplings are
    /// re-projected exactly.
    pub fn sample(&self, model: &Model, t: f64) -> Result<Vec<f64>, String> {
        if !t.is_finite() {
            return Err("non-finite sample time".into());
        }
        let n = self.qpos.len();
        let ts = &self.timestamps_seconds;
        let abs = ts[0] + t;
        let mut q = if n == 1 || abs <= ts[0] {
            self.qpos[0].clone()
        } else if abs >= ts[n - 1] {
            self.qpos[n - 1].clone()
        } else {
            let hi = ts.partition_point(|&s| s <= abs); // ts[hi-1] <= abs < ts[hi]
            let lo = hi - 1;
            let u = (abs - ts[lo]) / (ts[hi] - ts[lo]);
            let (a, b) = (&self.qpos[lo], &self.qpos[hi]);
            let mut q: Vec<f64> = a.iter().zip(b).map(|(x, y)| x + u * (y - x)).collect();
            for adr in free_adrs(model) {
                let qa: Quat = [a[adr + 3], a[adr + 4], a[adr + 5], a[adr + 6]];
                let qb: Quat = [b[adr + 3], b[adr + 4], b[adr + 5], b[adr + 6]];
                q[adr + 3..adr + 7].copy_from_slice(&slerp(&qa, &qb, u));
            }
            q
        };
        model.project(&mut q)?;
        Ok(q)
    }
}

/// Shortest-path spherical interpolation of unit quaternions.
pub fn slerp(a: &Quat, b: &Quat, u: f64) -> Quat {
    let mut b = *b;
    let mut dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    if dot < 0.0 {
        dot = -dot;
        for v in &mut b {
            *v = -*v;
        }
    }
    let (wa, wb) = if dot > 0.9995 {
        (1.0 - u, u) // nearly parallel: nlerp
    } else {
        let theta = dot.clamp(-1.0, 1.0).acos();
        let s = theta.sin();
        (((1.0 - u) * theta).sin() / s, (u * theta).sin() / s)
    };
    let mut r = [0.0; 4];
    for i in 0..4 {
        r[i] = wa * a[i] + wb * b[i];
    }
    normalize4(&mut r);
    r
}

// ---------------------------------------------------------------------------------------------
// Background simulation
// ---------------------------------------------------------------------------------------------

pub const SIM_DT: f64 = 0.001;
pub const DEFAULT_EXCITATION: f64 = 0.25;
/// Simulated seconds between snapshots (50 Hz).
pub const SNAPSHOT_PERIOD: f64 = 0.02;

#[derive(Debug, Clone, PartialEq)]
pub enum SimCommand {
    Pause(bool),
    Reset,
    Muscle(usize),
    Excitation(f64),
    Quit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimSnapshot {
    pub time_seconds: f64,
    pub qpos: Vec<f64>,
    pub activation: Vec<f64>,
    pub velocity: Vec<f64>,
    pub error: Option<String>,
}

pub struct Simulation {
    tx: Sender<SimCommand>,
    /// Single-slot (bounded) mailbox; the worker overwrites, the UI takes.
    slot: Arc<Mutex<Option<SimSnapshot>>>,
    worker: Option<JoinHandle<()>>,
}

fn snapshot(t: f64, s: &State, error: Option<String>) -> SimSnapshot {
    SimSnapshot {
        time_seconds: t,
        qpos: s.qpos.clone(),
        activation: s.activation.clone(),
        velocity: s.velocity.clone(),
        error,
    }
}

impl Simulation {
    pub fn spawn(model: Arc<Model>) -> Result<Self, String> {
        if model.muscles.is_empty() {
            return Err("model has no muscles".into());
        }
        let state = State::standing(&model)?;
        let muscle = model.muscle_id("BIClong").unwrap_or(0);
        let (tx, rx) = mpsc::channel();
        let slot = Arc::new(Mutex::new(Some(snapshot(0.0, &state, None))));
        let slot_w = Arc::clone(&slot);
        let worker = thread::Builder::new()
            .name("biomech-sim".into())
            .spawn(move || worker_loop(model, rx, slot_w, state, muscle))
            .map_err(|e| format!("cannot spawn simulation thread: {e}"))?;
        Ok(Simulation { tx, slot, worker: Some(worker) })
    }

    pub fn send(&self, command: SimCommand) -> Result<(), String> {
        self.tx.send(command).map_err(|_| "simulation thread has stopped".to_string())
    }

    /// Newest snapshot not yet taken, if any. Never blocks on simulation work.
    pub fn latest(&mut self) -> Option<SimSnapshot> {
        self.slot.lock().ok().and_then(|mut g| g.take())
    }
}

impl Drop for Simulation {
    fn drop(&mut self) {
        let _ = self.tx.send(SimCommand::Quit);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn worker_loop(
    model: Arc<Model>,
    rx: Receiver<SimCommand>,
    slot: Arc<Mutex<Option<SimSnapshot>>>,
    initial: State,
    mut muscle: usize,
) {
    let publish = |snap: SimSnapshot| {
        if let Ok(mut g) = slot.lock() {
            *g = Some(snap);
        }
    };
    let mut state = initial.clone();
    let mut time = 0.0f64;
    let mut paused = true;
    let mut level = DEFAULT_EXCITATION;
    let mut failed = false;
    let steps_per_snapshot = (SNAPSHOT_PERIOD / SIM_DT).round() as usize;
    let mut ex = vec![0.0; model.muscles.len()];

    macro_rules! apply {
        ($cmd:expr) => {
            match $cmd {
                SimCommand::Quit => return,
                SimCommand::Pause(p) => {
                    paused = p;
                    publish(snapshot(time, &state, None));
                }
                SimCommand::Reset => {
                    state = initial.clone();
                    time = 0.0;
                    failed = false;
                    publish(snapshot(time, &state, None));
                }
                SimCommand::Muscle(m) => {
                    if m < model.muscles.len() {
                        muscle = m;
                    }
                    state = initial.clone();
                    time = 0.0;
                    failed = false;
                    publish(snapshot(time, &state, None));
                }
                SimCommand::Excitation(e) => {
                    if e.is_finite() {
                        level = e.clamp(0.0, 1.0);
                    }
                }
            }
        };
    }

    loop {
        if paused || failed {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(c) => apply!(c),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
            continue;
        }
        let started = Instant::now();
        for _ in 0..steps_per_snapshot {
            loop {
                match rx.try_recv() {
                    Ok(c) => apply!(c),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return,
                }
            }
            if paused || failed {
                break;
            }
            ex.iter_mut().for_each(|e| *e = 0.0);
            ex[muscle] = level;
            match dynamics::step(&model, &mut state, &ex, SIM_DT) {
                Ok(()) => time += SIM_DT,
                Err(e) => {
                    failed = true;
                    publish(snapshot(time, &state, Some(e)));
                    break;
                }
            }
        }
        if !failed && !paused {
            publish(snapshot(time, &state, None));
            // Pace to real time; when slower there is no catch-up (UI shows the sim clock).
            let target = Duration::from_secs_f64(SNAPSHOT_PERIOD);
            if let Some(rest) = target.checked_sub(started.elapsed()) {
                thread::sleep(rest);
            }
        }
    }
}
