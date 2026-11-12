//! `biomech-lite` command line: inspect the model, replay the prescribed motion, and run a
//! short fixed-torso forward simulation. Runtime is Rust only; no Python is needed.
//!
//! See `docs/rust-biomechanics.md` for the command reference and limitations.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use biomech_lite::dynamics::{self, State};
use biomech_lite::Model;
use serde::{Deserialize, Serialize};

const DEFAULT_MODEL: &str = "assets/biomech-lite/model.json";
const DEFAULT_MOTION: &str = "assets/biomech-lite/motion.json";
/// Largest accepted fixed step; the Rust integrator is semi-implicit Euler.
const MAX_DT: f64 = 0.002;
/// Upper bound on simulated time per run (seconds).
const MAX_DURATION: f64 = 600.0;
/// Upper bound on integration steps per run, to keep output and runtime bounded.
const MAX_STEPS: usize = 2_000_000;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            if e.starts_with("usage") {
                ExitCode::from(2)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

const USAGE: &str = "\
usage: biomech-lite <command> [options]

commands:
  inspect   --model PATH                         print model summary
  replay    --model PATH --motion PATH --output PATH
                                                 prescribed replay of the recorded frames,
                                                 with passive (isometric) muscle forces; JSON
  simulate  --model PATH --output PATH --duration SECONDS --dt SECONDS
            --muscle NAME --excitation VALUE     fixed-torso forward dynamics; JSON

defaults: --model assets/biomech-lite/model.json, --motion assets/biomech-lite/motion.json
(relative to the current directory, normally the project root).";

fn run(args: &[String]) -> Result<(), String> {
    let Some((cmd, rest)) = args.split_first() else {
        return Err(format!("usage\n{USAGE}"));
    };
    match cmd.as_str() {
        "inspect" => cmd_inspect(&Opts::parse(rest, &["model"])?),
        "replay" => cmd_replay(&Opts::parse(rest, &["model", "motion", "output"])?),
        "simulate" => cmd_simulate(&Opts::parse(
            rest,
            &["model", "output", "duration", "dt", "muscle", "excitation"],
        )?),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("usage: unknown command '{other}'\n{USAGE}")),
    }
}

/// Minimal `--key value` option parser (no external CLI dependency).
struct Opts {
    values: HashMap<String, String>,
}

impl Opts {
    fn parse(args: &[String], allowed: &[&str]) -> Result<Self, String> {
        let mut values = HashMap::new();
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            let key = flag
                .strip_prefix("--")
                .ok_or_else(|| format!("usage: expected an option, got '{flag}'\n{USAGE}"))?;
            if !allowed.contains(&key) {
                return Err(format!("usage: unknown option '--{key}'\n{USAGE}"));
            }
            let value = it
                .next()
                .ok_or_else(|| format!("usage: option '--{key}' needs a value\n{USAGE}"))?;
            if values.insert(key.to_string(), value.clone()).is_some() {
                return Err(format!("usage: option '--{key}' given twice"));
            }
        }
        Ok(Self { values })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    fn path_or(&self, key: &str, default: &str) -> PathBuf {
        PathBuf::from(self.get(key).unwrap_or(default))
    }

    fn required(&self, key: &str) -> Result<&str, String> {
        self.get(key)
            .ok_or_else(|| format!("usage: missing required option '--{key}'\n{USAGE}"))
    }

    fn f64_required(&self, key: &str) -> Result<f64, String> {
        let raw = self.required(key)?;
        parse_finite(key, raw)
    }
}

fn parse_finite(key: &str, raw: &str) -> Result<f64, String> {
    let v: f64 = raw
        .parse()
        .map_err(|_| format!("'--{key}' is not a number: '{raw}'"))?;
    if !v.is_finite() {
        return Err(format!("'--{key}' must be finite, got {raw}"));
    }
    Ok(v)
}

fn load_model(path: &Path) -> Result<Model, String> {
    Model::load(path)
}

// ---------------------------------------------------------------------------------------------
// inspect
// ---------------------------------------------------------------------------------------------

fn cmd_inspect(opts: &Opts) -> Result<(), String> {
    let model_path = opts.path_or("model", DEFAULT_MODEL);
    let model = load_model(&model_path)?;
    let dyn_names: Vec<&str> = model
        .dynamic_joints
        .iter()
        .map(|&j| model.joints[j].name.as_str())
        .collect();
    println!("model: {}", model_path.display());
    println!("schema_version: {}", model.schema_version);
    println!("generalized coordinates nq={} nv={}", model.nq, model.nv);
    println!("bodies: {} (index 0 is world)", model.bodies.len());
    println!("joints: {}", model.joints.len());
    println!("couplings: {}", model.couplings.len());
    println!("sites: {}", model.sites.len());
    println!("geoms: {}", model.geoms.len());
    println!("tendons: {}", model.tendons.len());
    println!("muscles: {}", model.muscles.len());
    println!("reduced fixed-torso coordinates ({}):", dyn_names.len());
    for name in &dyn_names {
        println!("  {name}");
    }
    println!("gravity: {:?}", model.gravity);
    println!("muscle names (valid for --muscle):");
    for m in &model.muscles {
        println!("  {}", m.name);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// replay (prescribed motion, passive forces)
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct MotionFile {
    schema_version: u32,
    timestamps_seconds: Vec<f64>,
    frame_indices: Vec<u64>,
    qpos: Vec<Vec<f64>>,
}

fn load_motion(path: &Path, model: &Model) -> Result<MotionFile, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read motion {}: {e}", path.display()))?;
    let motion: MotionFile = serde_json::from_str(&text)
        .map_err(|e| format!("invalid motion JSON {}: {e}", path.display()))?;
    if motion.schema_version != 1 {
        return Err(format!(
            "motion schema_version {} unsupported (expected 1)",
            motion.schema_version
        ));
    }
    validate_motion(&motion, model)?;
    Ok(motion)
}

/// Checks a parsed motion before any output is written. Timestamps are kept exactly as recorded
/// (irregular spacing is allowed); they must only be finite and strictly increasing.
fn validate_motion(motion: &MotionFile, model: &Model) -> Result<(), String> {
    let n = motion.qpos.len();
    if n == 0 {
        return Err("motion has no frames".to_string());
    }
    if motion.timestamps_seconds.len() != n || motion.frame_indices.len() != n {
        return Err("motion arrays timestamps_seconds, frame_indices and qpos differ in length"
            .to_string());
    }
    let free_adr: Vec<usize> = model
        .joints
        .iter()
        .filter(|j| j.kind == biomech_lite::JNT_FREE)
        .map(|j| j.qposadr)
        .collect();
    for i in 0..n {
        let (t, q) = (motion.timestamps_seconds[i], &motion.qpos[i]);
        if !t.is_finite() {
            return Err(format!("motion frame {i}: non-finite timestamp"));
        }
        if i > 0 {
            if t <= motion.timestamps_seconds[i - 1] {
                return Err(format!("motion frame {i}: timestamps not strictly increasing"));
            }
            if motion.frame_indices[i] <= motion.frame_indices[i - 1] {
                return Err(format!("motion frame {i}: frame_indices not strictly increasing"));
            }
        }
        if q.len() != model.nq {
            return Err(format!(
                "motion frame {i}: qpos has {} entries, model nq={}",
                q.len(),
                model.nq
            ));
        }
        if q.iter().any(|v| !v.is_finite()) {
            return Err(format!("motion frame {i}: non-finite qpos"));
        }
        for &adr in &free_adr {
            // Free joint quaternion is qpos[adr+3..adr+7] (wxyz).
            let norm = q[adr + 3..adr + 7].iter().map(|v| v * v).sum::<f64>().sqrt();
            if (norm - 1.0).abs() > 1e-6 {
                return Err(format!(
                    "motion frame {i}: free-root quaternion norm {norm} is not unit"
                ));
            }
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct ReplayFrame<'a> {
    frame_index: u64,
    timestamp_seconds: f64,
    qpos: &'a [f64],
    body_pos: &'a [[f64; 3]],
    site_pos: &'a [[f64; 3]],
    tendon_lengths: &'a [f64],
    /// Passive (activation 0, tendon velocity 0) muscle tensions, muscle order.
    muscle_tensions: &'a [f64],
}

fn cmd_replay(opts: &Opts) -> Result<(), String> {
    let model = load_model(&opts.path_or("model", DEFAULT_MODEL))?;
    let motion = load_motion(&opts.path_or("motion", DEFAULT_MOTION), &model)?;
    let output = PathBuf::from(opts.required("output")?);

    let started = Instant::now();
    let mut out = create_output(&output)?;

    let ntendon = model.tendons.len();
    let nmuscle = model.muscles.len();
    let zero_vel = vec![0.0; ntendon];
    let zero_act = vec![0.0; nmuscle];

    write_str(&mut out, "{\"schema_version\":1,\"mode\":\"prescribed\",")?;
    write_str(
        &mut out,
        "\"label\":\"prescribed replay of recorded frames; muscle_tensions are passive \
         isometric evaluations (activation 0, tendon velocity 0), not inferred forces\",",
    )?;
    write_str(&mut out, "\"frames\":[")?;

    for (i, q_in) in motion.qpos.iter().enumerate() {
        let mut q = q_in.clone();
        model.project(&mut q)?;
        let kin = model.kinematics(&q)?;
        let lengths = model.tendon_lengths(&kin)?;
        let tensions = model.muscle_forces(&lengths, &zero_vel, &zero_act)?;
        let finite = lengths.iter().chain(&tensions).all(|v| v.is_finite())
            && kin.body_pos.iter().flatten().all(|v| v.is_finite())
            && kin.site_pos.iter().flatten().all(|v| v.is_finite());
        if !finite {
            return Err(format!("frame {i}: non-finite kinematics or forces"));
        }
        let frame = ReplayFrame {
            frame_index: motion.frame_indices[i],
            timestamp_seconds: motion.timestamps_seconds[i],
            qpos: &q,
            body_pos: &kin.body_pos,
            site_pos: &kin.site_pos,
            tendon_lengths: &lengths,
            muscle_tensions: &tensions,
        };
        if i > 0 {
            write_str(&mut out, ",")?;
        }
        serde_json::to_writer(&mut out, &frame).map_err(|e| e.to_string())?;
    }

    let elapsed = started.elapsed().as_secs_f64();
    write_str(&mut out, "],")?;
    write_str(
        &mut out,
        &format!(
            "\"summary\":{{\"frames\":{},\"elapsed_seconds\":{}}}}}",
            motion.qpos.len(),
            elapsed
        ),
    )?;
    out.flush().map_err(|e| e.to_string())?;
    drop(out);

    let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "replay: {} frames -> {} ({} bytes) in {:.3} s",
        motion.qpos.len(),
        output.display(),
        bytes,
        elapsed
    );
    Ok(())
}

/// Creates the output file, making missing parent directories first.
fn create_output(path: &Path) -> Result<BufWriter<File>, String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create directory {}: {e}", parent.display()))?;
    }
    let file = File::create(path)
        .map_err(|e| format!("cannot create output {}: {e}", path.display()))?;
    Ok(BufWriter::new(file))
}

fn write_str(out: &mut impl Write, s: &str) -> Result<(), String> {
    out.write_all(s.as_bytes()).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------------------------
// simulate (fixed-torso forward dynamics)
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
struct SimFrame<'a> {
    frame_index: usize,
    time_seconds: f64,
    /// Reduced fixed-torso coordinates (model.dynamic_joints order).
    coords: &'a [f64],
    /// Reduced velocities (same order).
    velocity: &'a [f64],
    /// Activation of the selected muscle.
    activation: f64,
}

fn cmd_simulate(opts: &Opts) -> Result<(), String> {
    let model = load_model(&opts.path_or("model", DEFAULT_MODEL))?;
    let output = PathBuf::from(opts.required("output")?);
    let duration = opts.f64_required("duration")?;
    let dt = opts.f64_required("dt")?;
    let excitation = opts.f64_required("excitation")?;
    let muscle_name = opts.required("muscle")?;

    if duration <= 0.0 || duration > MAX_DURATION {
        return Err(format!("'--duration' must be in (0, {MAX_DURATION}] s, got {duration}"));
    }
    if dt <= 0.0 || dt > MAX_DT {
        return Err(format!("'--dt' must be in (0, {MAX_DT}] s, got {dt}"));
    }
    // Round up so the run reaches `duration`; the last step is shortened so that simulated
    // time ends exactly at `duration`.
    let steps_f = (duration / dt - 1e-9).ceil();
    if steps_f < 1.0 || steps_f > MAX_STEPS as f64 {
        return Err(format!(
            "duration/dt gives {steps_f} steps; allowed range is 1..={MAX_STEPS}"
        ));
    }
    let steps = steps_f as usize;

    let muscle_idx = model
        .muscles
        .iter()
        .position(|m| m.name == muscle_name)
        .ok_or_else(|| format!("unknown muscle '{muscle_name}' (see `inspect`)"))?;
    let range = model.muscles[muscle_idx].ctrlrange;
    if excitation < range[0] || excitation > range[1] {
        return Err(format!(
            "excitation {excitation} outside muscle ctrlrange [{}, {}]",
            range[0], range[1]
        ));
    }

    let nmuscle = model.muscles.len();
    let mut control = vec![0.0; nmuscle];
    control[muscle_idx] = excitation;

    let mut qpos = model.standing_qpos.clone();
    model.project(&mut qpos)?;
    let mut state = State {
        qpos,
        velocity: vec![0.0; model.dynamic_joints.len()],
        activation: vec![0.0; nmuscle],
    };
    let qadr: Vec<usize> = model
        .dynamic_joints
        .iter()
        .map(|&j| model.joints[j].qposadr)
        .collect();

    let started = Instant::now();
    let mut out = create_output(&output)?;
    write_str(&mut out, "{\"schema_version\":1,\"mode\":\"forward_dynamics\",")?;
    write_str(
        &mut out,
        "\"label\":\"fixed-torso forward dynamics from the standing pose, one selected muscle \
         driven at constant excitation, others at 0\",",
    )?;
    write_str(
        &mut out,
        &format!(
            "\"muscle\":{},\"excitation\":{},\"dt\":{},\"steps\":{},\"frames\":[",
            serde_json::to_string(muscle_name).map_err(|e| e.to_string())?,
            excitation,
            dt,
            steps
        ),
    )?;

    let mut coords = vec![0.0; qadr.len()];
    for step in 0..=steps {
        let t = (step as f64 * dt).min(duration);
        for (c, &adr) in coords.iter_mut().zip(&qadr) {
            *c = state.qpos[adr];
        }
        if !coords.iter().all(|v| v.is_finite()) || !state.velocity.iter().all(|v| v.is_finite())
        {
            return Err(format!("non-finite state at step {step}"));
        }
        let frame = SimFrame {
            frame_index: step,
            time_seconds: t,
            coords: &coords,
            velocity: &state.velocity,
            activation: state.activation[muscle_idx],
        };
        if step > 0 {
            write_str(&mut out, ",")?;
        }
        serde_json::to_writer(&mut out, &frame).map_err(|e| e.to_string())?;
        if step < steps {
            let h = dt.min(duration - t);
            dynamics::step(&model, &mut state, &control, h)
                .map_err(|e| format!("integration failed at step {step}: {e}"))?;
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    write_str(
        &mut out,
        &format!(
            "],\"summary\":{{\"steps\":{},\"simulated_seconds\":{},\"elapsed_seconds\":{}}}}}",
            steps,
            duration,
            elapsed
        ),
    )?;
    out.flush().map_err(|e| e.to_string())?;
    drop(out);

    let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "simulate: {} steps (dt={dt}, {:.3} s simulated) -> {} ({} bytes) in {:.3} s",
        steps,
        duration,
        output.display(),
        bytes,
        elapsed
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Option<Model> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/biomech-lite/model.json");
        Model::load(path).ok()
    }

    fn free_adrs(model: &Model) -> Vec<usize> {
        model
            .joints
            .iter()
            .filter(|j| j.kind == biomech_lite::JNT_FREE)
            .map(|j| j.qposadr)
            .collect()
    }

    fn motion(model: &Model, times: Vec<f64>, idx: Vec<u64>) -> MotionFile {
        let mut q = model.standing_qpos.clone();
        for adr in free_adrs(model) {
            q[adr + 3..adr + 7].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        }
        let qpos = vec![q; times.len()];
        MotionFile { schema_version: 1, timestamps_seconds: times, frame_indices: idx, qpos }
    }

    #[test]
    fn numeric_options_reject_nonfinite_and_garbage() {
        assert!(parse_finite("dt", "nan").is_err());
        assert!(parse_finite("dt", "inf").is_err());
        assert!(parse_finite("dt", "-inf").is_err());
        assert!(parse_finite("dt", "abc").is_err());
        assert_eq!(parse_finite("dt", "0.001").unwrap(), 0.001);
    }

    #[test]
    fn unknown_and_duplicate_options_are_rejected() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(Opts::parse(&a(&["--bogus", "1"]), &["dt"]).is_err());
        assert!(Opts::parse(&a(&["--dt"]), &["dt"]).is_err());
        assert!(Opts::parse(&a(&["--dt", "1", "--dt", "2"]), &["dt"]).is_err());
        assert!(Opts::parse(&a(&["dt", "1"]), &["dt"]).is_err());
    }

    #[test]
    fn motion_validation_rejects_empty_reversed_and_bad_quaternion() {
        let Some(m) = model() else { return };
        assert!(validate_motion(&motion(&m, vec![], vec![]), &m).is_err());
        assert!(validate_motion(&motion(&m, vec![0.0, 0.5, 0.25], vec![0, 1, 2]), &m).is_err());
        assert!(validate_motion(&motion(&m, vec![0.0, 0.5, 1.0], vec![0, 2, 1]), &m).is_err());
        assert!(validate_motion(&motion(&m, vec![0.0, 0.5, 1.0], vec![0, 1, 2]), &m).is_ok());
        let mut bad = motion(&m, vec![0.0], vec![0]);
        for adr in free_adrs(&m) {
            bad.qpos[0][adr + 3] = 2.0;
        }
        assert!(validate_motion(&bad, &m).is_err());
    }
}
