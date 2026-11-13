# Skeleton-driven muscle recruitment: implementation contract

Final solver refinement (C2): the same bounded convex objective is now solved with an
eight-dimensional dual semismooth Newton method, continuation and Armijo backtracking.
Coordinate descent remains a fallback with projected KKT verification. The original algorithm
sketch below records the initial design; final APIs and signs remain unchanged. W2 also adds
Estimate.warnings and effective per-arm support, with reduced velocity vectors of length eight.

Implement option 1: existing tracked skeleton and finger IK prescribe qpos. Inverse dynamics
and bounded muscle recruitment estimate loads; they NEVER feed force or motion back into qpos.
This is estimated activation/tension for the existing fixed-torso EIGHT arm coordinates,
not individual fibers, not EMG, not full-body inverse dynamics. Assumptions: generic reference
anatomy, unloaded hands, no external/contact forces, all non-arm coordinates held/prescribed.
Wrist/finger inertial motion and torso motion are outside reduced dynamics. Unknown muscles
must be dimmed, not colored as measured inactive. Preserve all current tracking/stillness work.

## Core API: crates/biomech-lite/src/inverse.rs (owner C1)

Use existing dynamics/muscle code and nalgebra; no new dependencies. Declare pub mod inverse.
Public API exactly:

```
#[derive(Clone, Debug)]
pub struct Recruitment {
 pub activation: Vec<f64>,         // model.muscles order, bounded [0,1]
 pub tension_n: Vec<f64>,          // passive + active positive tension, muscle order
 pub supported_muscles: Vec<bool>,// nonzero moment in at least one observed dynamic row
 pub required_torque_nm: Vec<f64>,// M*a + bias, NX order; unloaded hands
 pub passive_torque_nm: Vec<f64>, // passive joint/tendon torque (not passive muscle)
 pub muscle_torque_nm: Vec<f64>,  // total passive+active muscle torque
 pub residual_torque_nm: Vec<f64>,// required - passive - muscle, NX order
 pub observed_coordinates: [bool; 8],
 pub residual_norm_nm: f64,       // norm over observed rows only
 pub relative_residual: f64,     // /max(1, norm(required-passive) observed)
 pub converged: bool,            // optimization convergence, not physical accuracy
 pub iterations: usize,
}
pub struct Recruiter;
impl Recruiter {
 pub fn new(model: std::sync::Arc<crate::Model>) -> Result<Self,String>;
 pub fn reset(&mut self);
 pub fn solve(&mut self,q:&[f64],velocity:&[f64],acceleration:&[f64],observed:[bool;8]) -> Result<Recruitment,String>;
}
```

Reuse dynamics::tendon_state and passive_from by exposing pub(crate), if needed. Compute each
primitive once per solve. Check lengths, all finite, projected q ranges, NX=8, at least one
observed coordinate. Do not mutate caller q and reject invalid input before state mutation.
`required=M(q)*acceleration + dynamics::bias(q,velocity)`.
`passive=dynamics::passive_from(...)` (joint damping/springs and passive non-actuator tendons).
Get F0=model.muscle_forces(length,velocity,zeros) and F1=...ones.
For muscle m, A[k,m]=-dL_t/dx_k*(F1[m]-F0[m]); passive muscle torque P[k]=-sum J[t,k]*F0[m].
Solve A*a ~= required - passive - P over ONLY observed rows; per-row scale
1/max(1,abs(required[k]-passive[k])). Penalize activation effort with lambda=1e-4 and continuity
with lambda_t=1e-5 toward last supported a. Coordinate descent bounded [0,1], warm start,
max 250 sweeps; diagonal denominator sum A_scaled^2 + lambda + lambda_t; incremental residual.
Set unsupported-muscle activation=0, mask false, do not pretend their physical passive force is
estimated motion activity. Convergence max coordinate delta <1e-5 (report failure honestly).
Compute total tension F0+(F1-F0)*a, full generalized torque and residual using original units.
No hidden reserve actuation: residual explicitly displays missing demand / infeasibility.
Unobserved velocity/acceleration MUST be 0 before calculating cross terms; ignore unsupported
rows in objective, residual norm, support masks. Activations are estimates, not excitations.
Use existing coupled-coordinate analytic geometry and finite-difference tendon Jacobians;
no FD of individual muscle activations, and no black-box optimizer dependency.

Core tests in tests/inverse.rs: known-activation synthetic demand using M^-1 to choose accel
(zero v) must reconstruct observed torque with <0.5 Nm residual; finite bounded activations and
force identity must match existing muscle_torque; static gravity support has small residual or
explicitly reported residual (do not assume every pose feasible); impossible large demand
retains bounds and exposes large residual; missing side mask excludes that side's muscles;
invalid inputs rejected; q unchanged; benchmark a small number of real model solves in release.
Do not run the long forward dynamics suite. Don't change existing parity algorithms.

## Async live adapter: src/biomech_estimate.rs (owner W1)

Do not edit main.rs or biomech_live.rs (lead integrates). Independent module with no UI/camera
access. API exactly:

```
#[derive(Clone)]
pub struct Sample { pub index:u64, pub captured:std::time::Instant,
 pub timestamp_seconds:f64, pub qpos:Vec<f64>, pub supported_arms:[bool;2] }
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EstimateMode { QuasiStaticWarmup, Dynamic }
#[derive(Clone)]
pub struct Estimate { pub sample:Sample, pub recruitment:biomech_lite::inverse::Recruitment,
 pub mode:EstimateMode, pub computation_ms:f64 }
#[derive(Clone,Default)]
pub struct State { pub estimate:Option<std::sync::Arc<Estimate>>, pub error:Option<String>,
 pub computing:bool }
pub struct Estimator;
impl Estimator {
 pub fn start(model:std::sync::Arc<biomech_lite::Model>) -> Result<Self,String>;
 pub fn submit(&self,sample:Sample);
 pub fn latest(&self)->State;
 pub fn reset(&self);
}
```

One worker thread, latest-only pending mailbox, no unbounded history. Drop stops/notifies/joins.
Do not hold mutex while computing. Clone Arc model. Native dynamic_joints order RIGHT then LEFT;
map each coordinate via model.joints[id].name suffix _l/_r, NOT assumed left-first array order.
Track derivative history independently per side. Sample qpos stays untouched for display;
project a private q copy. Unsupported arms invalidate their histories and rows. First valid
samples and gaps >0.5 s -> zero velocity/acceleration and QuasiStaticWarmup label. After at least
3 consecutive samples each supported side, finite differences at capture timestamps give qdot;
EMA velocity tau .08 s, acceleration from consecutive filtered velocities, EMA tau .12 s.
Reset on duplicate/reversed time, nonfinite, implausible input speed >10 rad/s or acceleration
>120 rad/s^2: report error and clear old estimate for that frame, do NOT silently clip motion to
look valid. Support partial arms independently. When no arms observed, publish estimate None
and clear warm state. `Dynamic` only if all currently supported sides have warmed.
Latest estimate retains its source sample and capture time so lead can reject stale/mismatched
values. Warmup computes a clearly quasi-static solve, allowing one-image file snapshots.
reset uses a generation token: pending/completed pre-reset work cannot publish after reset.
Worker must call Recruiter.reset when generation changes. Add focused module tests for helper
source-order mapping, static derivatives, gap reset and independent missing side; avoid NN/UI.
Build compile can await C1 if its file isn't ready; do not create a competing inverse stub.

## Renderer: crates/biomech-viewer/src/render.rs (owner V1)

Preserve Options Copy and existing APIs. Add renderer field Option<Vec<Option<f64>>> and
`pub fn set_muscle_activity(&mut self, activity:Option<&[Option<f64>]>) -> Result<(),String>`.
Some array must equal model muscle count? Renderer has no model at setter; validate finite
values [0,1] at setter, validate length against model.muscles.len() in render. None disables
coloring and preserves existing uniform muscle red. Some array uses None per muscle as unknown
(dark gray); Some(0) muted gold, Some(1) bright red, interpolate continuously. Expose
`pub fn muscle_activity_color(activity:Option<f64>) -> u32` for legend/tests. Selected muscle
cyan highlight remains unchanged. Add focused module tests color endpoints/midpoint and invalid
setter input/length. No other files owned; don't rustfmt entire crate or rewrite unrelated code.

## Lead integration I1

Default live viewer submits tracked qpos (with finger fit) and reliable supported arm masks
(shoulder/elbow/wrist all confident) to background Estimator. Never call recruitment on UI or
capture thread. Poll even when no new pose; discard results >0.5 s old (files use source frame
age), pre-reset results, errors, unsupported arms, or force-infeasible output; muscle colors
unknown for these (never hold bright stale colors). A moderate residual may be shown but label
unexplained torque; large residual >max(1 Nm, 10% required supported torque norm) is infeasible.
UI: label "Estimated muscle activation | unloaded arms" plus warmup/dynamic/source-age/latency;
show residual Nm, optimizer convergence, up to 4 supported high-activation muscles with % and N.
Legend explains gray unknown, gold low, red high; clearly not EMG/fiber FEM. Existing split 50/50
and full camera no crop persist. Muscle analysis toggle E optional; don't overload camera panel.
Snapshots pair anatomy/force result with exact source sample (bounded wait, no second webcam).
Explicit PNG fixture tests allowed. No real camera or GUI verification by workers.

## Documentation (owner N1)

Write docs/muscle-estimates.md: what is built, API, signs, effort objective, velocity filtering,
reference sources and commands; state reduced eight-DOF unloaded assumptions, nonunique
recruitment, generic anatomy, no finger/wholebody force inference, no EMG/individual fibers and
visible residual when demand cannot be matched. Evidence reported only from actual test logs.
Sources: OpenSim "How Static Optimization Works" and "Overview of OpenSim Workflows" official
pages; OpenSim Moco (Dembia et al 2020, DOI 10.1371/journal.pcbi.1008493). No claim implementing
Moco/CMC or that rigid-tendon MuJoCo actuator is detailed muscle fiber/tissue simulation.
