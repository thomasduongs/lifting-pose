# Rust biomechanics implementation and validation

Implemented a standalone Rust library and executable in `crates/biomech-lite` from
the applicable [MuJoCo 3.17.0 source](https://github.com/google-deepmind/mujoco/tree/3.17.0)
and the compiled [MyoSim](https://github.com/MyoHub/myo_sim) 0.2.5 model. This is a
source-derived port with Apache-2.0 attribution, rather than a wrapper around libmujoco.
It implements the subset needed by the current shoulder/elbow recording workflow.

## Delivered runtime

- Whole-model forward kinematics: 104 computational bodies including world and phantom
  frames, 123 native joints, 51 polynomial/locked joint equalities, and 2,042 sites.
- All 424 site/sphere/cylinder tendon paths, including side-site and inside-wrap behavior.
- All 416 sourced muscle actuators: activation dynamics, active force-length-velocity
  law and passive force. Series tendons use the source's stiff-tendon approximation.
- Eight glenohumeral ligament routes, four with the previously selected Boardman1996
  spring profile active. These are reference parameters, with uncalibrated slack lengths.
- Eight-coordinate forward arm dynamics: source masses/inertias, analytically reduced
  body Jacobians, gravity, Coriolis, armature, muscle torques, ligament springs and
  scalar joint damping. The torso/root and other independent coordinates are held.
- Headless prescribed replay and forward simulation, streaming JSON output.

The runtime requires the executable and `model.json`; replay additionally needs
`motion.json`. The current model and motion assets are already exported. Python,
MuJoCo, OpenCV, ONNX, model meshes and the reference fixture JSON are unnecessary
at runtime. New recording fitting and model export still use the Python tools.

`otool -L` on the release executable lists only macOS `libSystem.B.dylib`.
The executable is 789,088 bytes, model JSON 519,172 bytes and recording JSON 261,087
bytes: approximately 1.57 MB combined, excluding source notices. The independent
reference fixture file and upstream notices are retained for development and attribution.

## Commands

Run from `/Users/thomasduong/Downloads/lifting-pose`:

```sh
cargo build --release --manifest-path crates/biomech-lite/Cargo.toml
crates/biomech-lite/target/release/biomech-lite inspect
crates/biomech-lite/target/release/biomech-lite replay --output output/biomech-lite/replay.json
crates/biomech-lite/target/release/biomech-lite simulate --output output/biomech-lite/simulation.json --duration 0.5 --dt 0.001 --muscle BIClong --excitation 0.25
```

`replay` evaluates all fitted recording frames and preserves their irregular timestamps.
Its force values are passive, zero-activation, zero-velocity evaluations at each pose.
`simulate` starts at the standing reference pose and integrates the arms with the selected
muscle excitation. The output is numerical JSON, without an interactive graphics window.
The existing Python anatomical viewer and original Rust skeletal viewer remain separate.

## Numerical evidence

Validated November 12, 2026, using Rust 1.99.0 on macOS arm64. Native reference engine:
MuJoCo 3.17.0. Source model SHA256:
`c333dcc65c144a8775f83fa9cab057bc00d78b24b1b57c507ab9ebd6247f13a5`.
Recording SHA256:
`c35cbe0df2afa6a31ac1db8a9be01350847a7e3a192c20fa0ee2d34ce511a54e`.

The exporter calls native MuJoCo for independent fixtures. There are 39 cases covering
standing, shoulder/elbow sweeps, asymmetric poses, early and late recorded frames,
nonzero velocities, activation rise/decay, and loaded ligament poses. Native reduced
mass is `C.T*M*C`; bias includes coupling curvature `C.T*(bias + M*Cdot*xdot)`.
Ligament/joint passive torques are tested separately from actuator passive muscle force.

| Quantity | Largest observed difference across 39 cases |
|---|---:|
| Body position, Euclidean distance | 1.34e-15 m |
| Site position, Euclidean distance | 1.45e-15 m |
| Tendon length | 9.99e-16 m |
| Muscle tension | 1.50e-11 N |
| Activation derivative, normalized by max(1, native magnitude) | 1.10e-14 |
| Reduced mass matrix, max absolute entry error / max native entry | 4.87e-15 |
| Gravity demand | 2.67e-14 Nm |
| Gravity plus Coriolis/centrifugal demand | 1.54e-9 Nm |
| Muscle generalized torque | 9.29e-7 Nm |
| Ligament and joint passive generalized torque | 1.78e-7 Nm |

An additional independent native comparison checks every body position, site position,
tendon length and passive muscle force across all 222 fitted frames (8.059309708 seconds).
Maximum component errors: body/site position 1.33e-15 m, tendon length 1.22e-15 m,
passive muscle tension 4.22e-11 N. Original frame indices, timestamps and prescribed
joint coordinates also match. Report: `output/biomech-lite/replay-parity.json`.

Final checks: **17 Rust tests and 5 exporter tests passed**. The 500-step CLI run
produced 501 finite states, ended at exactly 0.5 seconds, and had zero native scalar
joint-limit violations after projecting every dependent joint. A separate fractional
duration run ended at exactly 0.0025 seconds. Failed dynamic steps preserve their input
state; tests also drive all eight coordinates toward both bounds.

Reproduce the checks:

```sh
cargo test --release --manifest-path crates/biomech-lite/Cargo.toml -- --nocapture
.venv-biomech/bin/python scripts/test_export_biomech_lite.py
.venv-biomech/bin/python scripts/validate_biomech_lite.py --output output/biomech-lite/replay-parity.json
```

Logs are in `output/biomech-lite/rust-tests.log` and `export-tests.log`.

## Performance and remaining limits

All 222 recording frames compute and serialize in roughly 0.10–0.14 seconds, producing
32.3 MB JSON. The same 500-step dynamics benchmark improved from 3.265 to 1.174 seconds
after replacing finite-difference body Jacobians with analytic Jacobians (about 2.8x).
The half-second CLI simulation at dt=0.001 took 1.18 seconds in the worker measurement
and 1.37 seconds during final concurrent checks. Forward dynamics are currently slower
than real time. Tendon Jacobians and Coriolis mass derivatives remain numerical.

This is generic reference anatomy with a monocular kinematic fit. Numerical agreement
with MuJoCo establishes implementation fidelity, not subject-specific anatomical or
physiological accuracy. The previous fit limitations, including uncertain late-frame
motion, carry forward. Actual muscle activations, tissue stiffness, tendon slack lengths,
and subject forces have not been identified from the recording.

The runtime excludes contact, frictional ground reaction, whole-body balance, independent
neck motion, compliant tendon dynamics and continuum muscle FEM. Exact polynomial
parameterization and hard joint bounds replace native soft equality/limit constraints;
semi-implicit Euler trajectories therefore are not claimed identical to MuJoCo stepping.
Current clinical ligament-force interpretations remain unsupported.

## Duo ownership

Codex handled source research, contracts, integration and whole-recording validation.
Claude Opus implemented the Rust core; Sonnet handled export/reference data, dynamics,
performance and ligament torque validation; Haiku implemented CLI commands and docs.
Claude owned seven of nine tasks. Worker-reported cumulative usage estimate: **$4.88**,
excluding Codex lead usage. Source and model notices are in
`assets/biomech-lite/NOTICE.md` and `assets/biomech-lite/provenance/`.
