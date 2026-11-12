# Biomechanical simulation workflow

This workflow exports a sourced full-body musculoskeletal reference model, fits a
recording's joint-centre markers to its independent joints, and renders the result
headlessly. It runs in an isolated Python 3.13 environment. Interactive replay is
an optional command that the user launches; export, fitting and rendering open no windows.

**Scope.** The anatomy is a *reference* (generic adult) model from MyoSim 0.2.5
(MyoArm/MoBL, MyoLeg/Rajagopal 2016, MyoBack/Walia 2025) on MuJoCo 3.17.0. It is not
subject-specific anatomy. The fit measures the recording's kinematics against that
reference, not the subject's bones, muscles or ligaments. See [Limitations](#limitations).

## Setup

Use Python 3.10 or newer. The workflow was built with Python 3.13 in
`.venv-biomech`, which is git-ignored.

```sh
python3.13 -m venv .venv-biomech
.venv-biomech/bin/python -m pip install -r scripts/requirements-biomech.txt
```

`scripts/requirements-biomech.txt` pins the versions used. Upstream model sources
are not edited. Their LICENSE/NOTICE files and per-model READMEs are copied into the
export's provenance.

## 1. Export the reference model

```sh
.venv-biomech/bin/python scripts/biomech_model.py --output output/biomech-reference
```

This writes `model.mjb` (version-specific MuJoCo binary), a portable `model.xml`
with its mesh assets, `anatomy.json` (bodies, joints, actuators, ligament routes,
geometry scope) and `provenance.json` (versions, hashes, reconstruction recipe).
Ligaments are off by default. To enable tension-only springs on the eight existing
glenohumeral/coracohumeral routes, pass `--ligament-stiffness N` and
`--ligament-slack-scale S`. Both values are exploratory priors, not measured tissue
properties.

## 2. Fit a recording

```sh
.venv-biomech/bin/python scripts/fit_biomech.py output/recording.twin \
    --output output/biomech-twin --ligament-profile boardman1996 --mechanics
```

The fit uses `scripts/fit_biomech.py` and writes a fit directory with:

- `motion.npz`: `qpos`, `timestamps_seconds`, `frame_indices`, `joint_names`,
  `marker_positions`, `marker_names`, `fit_rmse`, `fitted_dof_mask`,
  `dependent_dof_mask`, `derived_dof_mask`, `support_3d`, `support_2d`,
  `reprojection_rmse_px`, `fit_status` and root-heading diagnostics.
- `fit_report.json`: observed vs. fixed DOFs, input hash, units, alignments,
  per-frame support, limit and coupling checks, optimizer status.
- `model.mjb`: the model snapshot that matches the `qpos` interpretation.
- `mechanics.npz` (with `--mechanics`): musculotendon lengths, velocities, moment
  arms, passive generalized forces and inverse-dynamics demand under the prescribed
  motion. Contacts are excluded. These are kinematic demand, not measured forces.

Eight independent shoulder/elbow coordinates are optimized, with native couplings
enforced on every residual evaluation. The fit combines supported 3D targets with
fresh image observations from `poses.jsonl` when available; it preserves the
distinction between observed image location and inferred depth. Intrinsics may be
assumed, so this is not camera calibration. `--no-image-observations` disables that
term. Source body heading supplies a derived root yaw; `--no-root-yaw` disables it.
Root translation, trunk/neck articulation, forearm twist, wrists, fingers,
independent scapular motion and lower-body joints remain reference priors. The
recorded per-finger quaternions are not copied.

`--ligament-profile boardman1996` uses published aggregate linear stiffness for
bilateral SGHL (17,400 N/m) and CHL (36,700 N/m), from
[Boardman et al.](https://pubmed.ncbi.nlm.nih.gov/8872921/). Four routes become
tension-only springs; the other four have no matched stiffness in this profile.
This is an optional reference approximation with uncalibrated slack, not measured
personal tissue mechanics. It cannot be combined with a nonzero uniform
`--ligament-stiffness`. The fitted `model.mjb` includes the selected profile.
The separate reference export's portable XML retains its own stated configuration.
The delivered fit also includes a portable XML/assets snapshot with the same
four-route profile, checked against its binary. The fitter itself updates the
binary; use a separate output directory when changing model configuration so a
previously packaged XML/provenance snapshot is not mistaken for a new export.

## 3. Render the fitted subject

```sh
.venv-biomech/bin/python scripts/render_biomech.py output/biomech-twin \
    --output output/biomech-twin/front.mp4 --view front --fps 30 --size 720
```

Options: `--view front|side`, `--max-frames N` (0 renders all frames),
`--snapshot frame0.png` (also writes the first frame as PNG for inspection).

The renderer is an orthographic projection that does not require a GL context.
It resamples the original irregular timestamps with previous-pose holds and
includes the final source pose. `--max-frames` limits source poses, not output
clock ticks. Text identifies how many ligament routes actually have stiffness.
It draws:

- bones and solid tissue: the sourced mesh geoms, at the fitted pose, in grey;
- muscle-tendon paths: red polylines along the engine's solved route (via-points and wrap points);
- ligament routes: orange polylines (the eight upstream routes);
- fitted joint centres: green dots.

The lower body is grey and labeled **unobserved**, because the recording does not
observe it and it stays at the reference standing pose.

Muscle and ligament lines use the engine's solved path after `mj_forward`: the
flattened wrap points `data.wrap_xpos.reshape(-1, 3)[ten_wrapadr[t] : ten_wrapadr[t] + ten_wrapnum[t]]`.
These include site via-points and sphere/cylinder tangent points. At standing,
DELT1 and BIClong match the engine's `ten_length` to better than 1e-6 m, and every
route's endpoints match its first and last site.

Rendering limitation: a wrapped arc is drawn as a chord between its solved tangent
points. The polyline differs from the engine's `ten_length` by up to about 5 mm for
some routes at standing, and the difference changes with pose. The renderer does not
claim exact arc geometry.

Rendering cost: each triangle is filled separately, far to near, which takes about
1 s per frame on the full mesh at 720 px. A 222-frame recording takes a few minutes.
Use `--max-frames` for a quick preview.

## Replay the fitted motion in the native viewer

Four things share the word "replay" or "video". They are different:

| Command | What it shows | Simulation |
| --- | --- | --- |
| `cargo run --release -- --replay` (Rust, [README](../README.md)) | The original recording's twin, from the `.twin` file | Rust pose pipeline; not this model |
| `scripts/render_fem_twin.py` | Inverse-FEM scaffold video | Experimental FEM, not this model |
| `scripts/render_biomech.py` | Headless MP4/PNG of the sourced reference anatomy at the fitted pose | None (a picture of the kinematics) |
| `scripts/replay_biomech.py` | Live native MuJoCo playback of the sourced reference model at the fitted pose | Kinematic: the joints follow the prescribed fit |

Force-driven simulation (muscle activation, inverse-dynamics demand, contacts) is
not part of this replay. The fitted motion is prescribed, so the model is moved by
the fit and not by muscle forces. Use `mechanics.npz` for kinematic demand only.

```sh
# Headless check of the inputs. Prints counts and timing, opens no viewer.
.venv-biomech/bin/python scripts/replay_biomech.py output/biomech-twin --inspect

# Interactive playback: run it yourself. On macOS the viewer needs mjpython.
.venv-biomech/bin/mjpython scripts/replay_biomech.py output/biomech-twin --speed 1.0 --loop
```

`--subject NAME` selects a subject subdirectory when the fit wrote several subjects.
`--speed` must be positive. Playback keeps the recorded timestamps, including
irregular gaps. It interpolates with MuJoCo's own differentiation and integration,
which handles the free-root quaternion correctly, and then re-enforces the native
joint couplings, so dependent joints are never interpolated on their own.

## Limitations

- Reference anatomy only. No subject-specific bone, muscle volume, fascicle, moment
  arm or ligament geometry.
- Muscles use MuJoCo's Hill-type force-length-velocity model on a stiff series
  tendon. Tendon and aponeurosis compliance, fascicle architecture and 3D continuum
  muscle strain are not modeled.
- Only the eight upstream glenohumeral/coracohumeral ligament routes exist. They
  produce no force unless a profile or stiffness is set; other ligaments are not
  modeled. The literature profile covers only four of these routes. Linear
  springs omit nonlinear toe behavior, viscoelasticity and tissue failure.
- The recording observes the upper body. The lower body, wrists, fingers and
  scapula are priors, not measurements.
- Inverse dynamics are under prescribed motion and exclude contacts. Nothing here
  identifies muscle forces or activations from video.
- Landmark mismatch: MediaPipe/H36M keypoints are surface and learned landmarks; the
  model markers are joint centres. See `LANDMARK_MISMATCH_NOTE` in
  `scripts/biomech_model.py`.
- Fit residuals and estimated joint rates remain large in some frames. Pose and
  inverse-dynamics results are estimates under the stated priors, not validated
  personal kinematics or clinical force predictions. The source head is rigid to
  the torso; detailed cervical articulation is absent.

## Force-production and mobility experiments

```sh
.venv-biomech/bin/python scripts/biomech_experiment.py \
    --output output/biomech-experiment/BIClong_elbow_flexion_r \
    --muscle BIClong --joint elbow_flexion_r
.venv-biomech/bin/python scripts/biomech_experiment.py \
    --output output/biomech-experiment/TRIlong_elbow_flexion_r \
    --muscle TRIlong --joint elbow_flexion_r
.venv-biomech/bin/python scripts/biomech_ligaments.py --output output/biomech-ligaments
```

The isometric bench sweeps angle and prescribed activation, checks moment arms
against tendon-length derivatives, and integrates the native activation response
with kinematics clamped. It exports `sweep.csv`, `activation_step.csv`,
`ligaments.csv`, `experiment.npz` and `diagnostics.json`. Tension is positive for
pulling; moment arm is `-dL/dq`; torque is moment arm times tension. Partial
native-coordinate and total coupled-coordinate moment arms are distinguished.
This is controlled reference-model force production, not force identification
from video or a force-driven reproduction of the recording.

The ligament bench produces `ligament_sensitivity.csv/json`, varying source slack
and published stiffness values on the four supported routes. It exposes parameter
sensitivity; large simulated forces can lie outside the experimental load range
and must not be interpreted as injury or personal tissue strength.

[Research comparison](biomechanics-research.md) and
[completed-run results](biomechanics-results.md) describe what was implemented,
the artifacts and the remaining accuracy limits.

## Verification

Offline tests use the real MuJoCo engine at small scale:

```sh
.venv-biomech/bin/python -m unittest discover -s scripts -p 'test_*biomech*.py'
```

Covered: model compilation and inventory, coupling residuals, tension-only ligament
springs, fitting on synthetic known poses, joint limits, missing data and irregular
timing, renderer axes plus an end-to-end MP4/PNG smoke test, and replay timing,
interpolation, input validation and `--inspect`. The replay tests open no viewer. These tests are
not run against the Python 3.9 FEM workflow.

The force demo and ligament profile have additional native-engine consistency
tests. Integration also checks all exported poses, model snapshots and timing.

## Licenses

MyoSim, MyoArm, MyoLeg and MyoBack source models and MuJoCo are third-party. Their
license and notice files are copied into `output/biomech-reference/provenance`. Follow
those terms when you redistribute the model or its assets.
