# Muscle activation estimates for the live anatomical twin

This document describes the reduced muscle-recruitment estimator added to the biomech-lite
crate and its asynchronous live adapter. It is an estimate of what muscle activation would
be needed to produce the demanded arm torques. It is not a measurement, not EMG, and not a
force reading.

## What is built

| Part | Location | Role |
| --- | --- | --- |
| Core recruitment | `crates/biomech-lite/src/inverse.rs` | Per-frame solve: demanded torque to bounded muscle activations |
| Core tests | `crates/biomech-lite/tests/inverse.rs` | Synthetic and invariant tests (see Evidence) |
| Live adapter | `src/biomech_estimate.rs` | Background worker; velocity and acceleration from pose history |
| Renderer | `crates/biomech-viewer/src/render.rs` | Colors each muscle path by its estimated activation |

The tracked skeleton still prescribes position, velocity and acceleration. The estimator only
reads them. Nothing feeds back into the motion.

## Scope and assumptions

- **Reduced model:** eight fixed-torso arm coordinates (the `model.dynamic_joints`), with
  all other coordinates prescribed at their current tracked/reference poses. Their velocities
  and accelerations are outside this reduced inverse-dynamics model.
- **Unloaded hands:** no object or hand load is modelled.
- **No external or contact forces:** gravity acts through the model; nothing else does.
- **Generic reference anatomy:** the biomech-lite reference model, not a person's measured geometry.
- **Non-unique recruitment:** with more muscles than coordinates, many activation patterns
  produce the same torque. The effort and continuity regularizers select one of them. The
  selection is a modelling choice, not a physiological finding.
- **Not covered:** finger or whole-body force inference, individual fibers, EMG, and any
  claim that a muscle is fatigued, loaded or injured.

## Equations

Sign convention (same as `dynamics::step`): `M qdd + bias = muscle + passive`.

- Required torque: `required = M(q) a + bias(q, v)`, where `a` is the demanded acceleration.
- Passive joint and tendon torque: `passive = dynamics::passive_from(...)` (joint damping and
  springs, plus non-actuator passive tendons). Passive muscle force is handled separately.
- Muscle tension: `T_m = F0_m + (F1_m - F0_m) * a_m`, with `F0` and `F1` the muscle forces at
  activation 0 and 1 (`model.muscle_forces`). The muscle contributes `-dL/dx * T` to each coordinate.
- Muscle influence matrix: `A[k, m] = -dL_t/dx_k * (F1_m - F0_m)`.
- Passive muscle torque: `P[k] = -sum_m J[t, k] * F0_m`.
- Fit target: solve `A a ~ required - passive - P` over the **observed rows only**, with each
  row scaled by `1 / max(1, |required_k - passive_k|)`.

Residual reporting:

- `residual_torque_nm = required - passive - muscle_torque`, per coordinate.
- `residual_norm_nm` and `relative_residual` use observed rows only.
- Unobserved coordinates are excluded from the objective and residual norm. Muscles without a
  moment about any observed coordinate receive a `supported_muscles = false` mask; their
  stored zero activation is not an estimate that the real muscle is inactive.

## Objective and solver

Minimize the scaled residual plus two small penalties:

- Effort: `lambda = 1e-4` on activation.
- Continuity: `lambda_t = 1e-5` toward the last supported activation (warm start).

The primary solver is a convex dual semismooth Newton method over the observed torque
rows (at most eight). Activations are recovered with the exact `[0,1]` box projection.
The dual Hessian is positive definite, with damped steps and Armijo backtracking. Six
continuation stages warm-start the final solve at the exact stated regularization weights;
the final stage allows up to 80 iterations. A bounded coordinate-descent fallback allows
250 sweeps and verifies projected optimality conditions before reporting convergence.

The dual gradient tolerance is `1e-8`; projected primal KKT tolerance is `1e-6`. Duplicate
muscle columns share load equally, and muscle ordering does not select the result. No external
optimizer dependency is used. Geometry uses the existing coupled-coordinate kinematics and
finite-difference tendon Jacobians. Individual activations are not finite-differenced.

`converged` describes numerical optimality for this objective, not physiological accuracy.
The activation solution remains dependent on model parameters and recruitment assumptions.

## Live adapter (`src/biomech_estimate.rs`)

- One worker thread with a latest-only pending slot. A new sample replaces an unprocessed one,
  so history is bounded. Dropping the estimator stops and joins the thread.
- The worker never holds the state lock while computing.
- `reset()` bumps a generation token. Work that started before the reset cannot publish after it,
  and the worker resets its recruiter and histories when it sees a new generation.
- Arms are mapped to coordinates by joint-name suffix (`_l` left, `_r` right), not by array order.
- Each arm keeps its own derivative history:
  - Velocity: finite differences at capture timestamps, EMA with tau 0.08 s.
  - Acceleration: finite differences of the filtered velocity, EMA with tau 0.12 s.
  - An arm is warm after 3 consecutive valid samples. Before that its velocity and acceleration are zero.
  - A gap longer than 0.5 s restarts that arm's history (zero derivatives).
  - Duplicate or reversed timestamps, non-finite input, speed above 10 rad/s, or acceleration
    above 120 rad/s^2 reject that arm and report a per-arm warning; a valid opposite arm
    continues. Fatal global errors clear the estimate and derivative histories. Motion is never clipped to look valid.
- Mode: `Dynamic` only when every currently supported arm is warm. Otherwise
  `QuasiStaticWarmup`, labelled "motion warming; cold arms static". A warm arm keeps its
  own motion derivatives when the opposite arm is still warming.
- When no arm is observed, the estimate is cleared and warm state is dropped.
- Each published estimate carries its source sample and capture time, so the caller can reject stale values.

The adapter uses vectors of length eight indexed by `dynamic_joints`, matching the core
reduced API. Each estimate carries the effective supported arm mask and per-arm warnings.
A rejected/unsupported arm is held at the standing prior only inside the estimation copy;
the displayed skeleton is unchanged.

## Usage

```bash
./scripts/live_biomech.sh
```

Estimates are on by default. Press **E** or click **EST** to toggle. The model panel shows
activation percentage, estimated tension in N, unexplained torque in Nm, source age, and
optimizer status. Blue/green/yellow/orange/red muscle paths show 0/25/50/75/100% estimated activation. Gray means
unknown, including stale/unsupported/paused estimates or demand that cannot be matched.
The camera remains the full uncropped input.


Core (from Rust, inside the workspace):

```rust
let mut recruiter = biomech_lite::inverse::Recruiter::new(model.clone())?;
let r = recruiter.solve(&q, &velocity, &acceleration, observed)?; // observed: [bool; 8]
// r.activation, r.tension_n, r.residual_torque_nm, r.converged, r.iterations
```

Live adapter:

```rust
let estimator = biomech_estimate::Estimator::start(model.clone())?;
estimator.submit(sample);             // non-blocking
let state = estimator.latest();       // estimate, error, computing
```

Renderer:

```rust
renderer.set_muscle_activity(Some(activations.as_slice()))?; // Option<&[Option<f64>]>, values in [0, 1]
// None entries are drawn dark gray (unknown); None disables coloring.
```

Colors: blue at 0%, green at 25%, yellow at 50%, orange at 75%, red at 100%,
with smooth interpolation between these points; dark gray means unknown. The selected
muscle highlight is unchanged.

## Evidence

Reported from actual release runs:

- Eight inverse-dynamics/recruitment integration tests passed: known-activation torque
  reconstruction, force identity, capacity failure with explicit residual, confidence masks,
  invalid inputs and exclusion of unsupported motion.
- Four optimizer tests passed: equal sharing for redundant muscles, box bounds, permutation
  invariance and continuity regularization.
- All three tested static poses converged (27–37 total Newton iterations) with torque residuals
  below 0.001 Nm. Average measured core solve time was about 3 ms for the 416-muscle model.
- A real-model file test produced 33 source-tagged estimates from 35 fitted frames: 28 dynamic
  and 23 within the displayed feasibility threshold. Larger residuals remain visible and their
  muscle colors are withheld. These counts are integration evidence, not biological accuracy.

Lead verification now includes nine live-worker tests, thirteen live-view tests and two
renderer color/validation tests, all passing in release. A packaged file-input preview ran the
actual pose/hand models, skeletal fit and inverse muscle solve with exact source-frame pairing.
No real-camera hardware verification was performed for this change. Logs and final optimizer
measurements are recorded under `output/muscle-estimates/` and `docs/live-biomech-results.md`.

## Sources

- [OpenSim: How Static Optimization Works](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089619/How%2BStatic%2BOptimization%2BWorks).
- [OpenSim: Overview of OpenSim Workflows](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53084226/Overview%2Bof%2BOpenSim%2BWorkflows).
- [Dembia et al. (2020): OpenSim Moco](https://doi.org/10.1371/journal.pcbi.1008493).

This work is a reduced, unloaded, fixed-torso estimator in the spirit of static optimization.
It does not implement Moco, CMC, or any optimal-control solver. The MuJoCo actuator used here is
rigid-tendon and is not a detailed muscle-fiber or tissue simulation.
