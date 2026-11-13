# Biomechanical human simulation contract

The user prioritizes **biomechanical simulation**. Build a real research-model
workflow, not another capsule scaffold. Preserve source recordings and existing
Rust recording/replay behavior. Do not open a browser, native viewer, or GUI for verification.
Use `.venv-biomech/bin/python` (Python 3.13) for all new work. Lead has installed
myo-sim==0.2.5, mujoco==3.17.0, NumPy, SciPy and OpenCV in that isolated environment.

## Anatomical base and scope

Use `myo_sim.load_spec('myofullbody', build_kwargs={'postures': True})` or
`myo_sim.load('myofullbody')`. Actual local compile: 104 bodies including world,
123 joints, nq129/nv128, 416 muscle actuators, 424 spatial tendons, 104 meshes.
Reset to the `standing` keyframe. Actual DOF and model content must be inventoried,
not inferred from advertised counts (some joints are dependent/coupled).

The eight unactuated tendons named s/m/i_glenohum_ligament and coracohum_ligament
(bilateral suffix _l) are real upstream ligament routes, but **zero stiffness,
damping and disabled limits** in the base model. Do not call these force-producing
ligaments unless mechanics are explicitly enabled and tested. No claim to include
all human ligaments. No invented attachment coordinates or rebranding actuator
paths as ligament anatomy. Provide optional tension-only native tendon springs
on these existing routes (`springlength=[0,Lslack]`) with documented configurable
stiffness and slack; defaults are exploratory priors, not measured properties.
Compare enabled/disabled passive force at a pose; below slack must produce zero.
If reference slack lengths over-tension standing, document and offer neutral-pose
slack calibration as an explicit model choice, never a measured calibration.

Model anatomy derives from MyoSim/MoBL, Rajagopal and MyoBack sources. Pin versions,
copy LICENSE/NOTICE and original per-model README sources into output provenance.
Package assets are installed in `.venv-biomech`, ignored by git. Do not edit those
upstream files. Keep a reproducible export (`model.mjb`, portable `model.xml` plus
required mesh assets if feasible) in `output/biomech-twin`. A compiled MJB alone
is version-specific; include exact reconstruction recipe and dependency lock.

Muscles use the engine's sourced Hill-type force-length-velocity/activation model.
MuJoCo's tendon is a spatial musculotendon path. The imported model uses stiff
series tendons and does NOT solve tissue elasticity, intramuscular fascicles,
ligament injury, cartilage or subject-specific anatomical geometry. State this.

## Files and APIs

`scripts/biomech_model.py` (Claude Opus):

* `load_model(ligament_stiffness=0.0, ligament_slack_scale=1.0)` -> (model,data),
  standing state; validates material options. Enable only named existing ligament
  routes when stiffness>0, and preserve all416 muscle actuator definitions.
* `enforce_couplings(model,data)` -> updates dependent scalar joint qpos from
  native equality-polynomial definitions in dependency order; preserve source
  joint references and conventions. Expose any unsupported equality type.
* `marker_positions(model,data)` -> dict str:world xyz; at least
  left/right_shoulder, left/right_elbow, left/right_wrist, thorax, head.
  Use meaningful joint centers from the native model (average medial/lateral
  elbow markers; wrist center from radius/ulna/carpal frame; head from native
  head frame). State landmark mismatch vs MediaPipe. No fabricated anatomy.
* `inventory(model)` -> JSON-safe structures: counts, body/joint/actuator names,
  source limits, muscle parameters, ligament routes/mechanics, geometry scope.
* CLI `--output DIR` default output/biomech-twin writes model assets,
  anatomy.json and provenance.json. Keep saved model and analysis model identical
  with respect to ligament config. Provide `--ligament-stiffness` / slack flags.
* `scripts/test_biomech_model.py`: finite compilation, native coupling residual,
  muscle force varies with activation, tension-only ligaments slack/taut, names.
* `scripts/requirements-biomech.txt`: pin successful dependency versions.

`scripts/fit_biomech.py` (Claude Sonnet):

* CLI `.twin|twin.json --output DIR [--stride N] [--max-frames N]
  [--ligament-stiffness N] [--ligament-slack-scale S] [--max-nfev N]`.
* Reuse `skeleton_recording.py` file/schema/timestamp/FK helpers where suitable.
  Fit positions to anatomical joint-center markers using confidence-weighted
  robust least_squares and actual anatomical joint limits. Optimize independent
  shoulder/elbow coordinates only (plus constrained root alignment if useful);
  **enforce native shoulder polynomial couplings on every residual evaluation**.
  Never independently optimize all dependent joints. Underobserved forearm twist,
  wrists/fingers, scapular pose and lower body remain explicit priors. Do not copy
  the recorded glitching per-finger quaternion rotations into anatomical joints.
* Use a documented proper camera->model basis, name-based left/right matching,
  per-subject scale/alignment of RECORDING targets to the reference model. No
  mirrored handedness. Source coordinates are inferred meters; this alignment is
  NOT measurement of personalized bones/muscle anatomy. Report scale and what is
  held fixed; don't force knees/feet from an upper-body recording.
* Confidence>=? continuous weighted targets; missing elbow native depth may retain
  usable image observations but do not invent measured 3D confidence. Use temporal
  prior for unobserved DOF and continuity, with robust residuals and no arbitrary
  frame deletion. Robust shared fit and bounded optimization should reduce
  reacquisition snapping; report remaining jumps and residuals rather than hide.
* Output `motion.npz`: qpos(F,nq), timestamps_seconds(F,), frame_indices(F,),
  joint_names(J,), source_joint_confidence(F,J), marker_positions(F,K,3),
  marker_names(K,), fit_rmse(F,), fitted_dof_mask(nv,). Output `fit_report.json`
  with observed vs fixed DOFs, input sha, units, alignments, per-frame support,
  limits/coupling checks and optimizer status. Separate subject dirs if >1.
  Fit every frame by default, preserve irregular timestamps. Do not overwrite
  source recordings. Save model snapshot matching qpos interpretation.
* Add optional mechanics output `mechanics.npz` containing musculotendon lengths,
  velocities (actual nonuniform derivative or engine Jacobian*qvel), moment arms,
  passive generalized forces and inverse-dynamics generalized demand, with units.
  If inverse dynamics under prescribed motion is provided, exclude contacts and
  say so; it does NOT identify forces from the video. Never call static
  optimization activations or real tissue forces measured.
* `scripts/test_fit_biomech.py`: independent synthetic known-pose recovery,
  joint-limits/coupling respect, missing data behavior, irregular timing and no
  hand flip copying. Tests should exercise real engine at small scale.

`scripts/render_biomech.py` + docs (Claude Haiku):

* Headless CLI `OUTPUT_DIR --output MP4 [--view front|side] [--fps 30]
  [--max-frames N] [--size 720]`; read `model.mjb` and `motion.npz` (or its per-subject
  directory, coordinate with Sonnet). Render actual sourced bone mesh geometry
  and real routed musculotendon paths, label muscle lines vs solid tissue.
  Prefer MuJoCo Renderer offscreen; if unavailable on macOS use code-only
  projection of `model.mesh_vert/face` through body/geom transforms and tendon
  wrap points. Never open mjviewer, browser, or windows. Reference lower body
  should be visibly labeled as unobserved. Show skeleton/MTU paths and ligaments
  with distinct colors; activation only if actual simulated activation supplied.
* `docs/biomechanics.md`: reproducible setup using .venv-biomech Python>=3.10,
  model export, recording fit, rendering, force demo, limitations and licenses.
  Update README with concise link and command. No claimed verified status until
  integration reports it. Do not clobber user's existing README edits.
* `scripts/test_render_biomech.py`: tiny offline smoke/axis checks only.

## Research sources already verified by lead

* User M3 lab: https://engineering.virginia.edu/labs-groups/multiscale-muscle-mechanophysiology-m3-lab
* Blemker/Pinsky/Delp2005 biceps continuum mechanics: https://pubmed.ncbi.nlm.nih.gov/15713285/
* Blemker/Delp2005 MRI muscle architecture: https://pubmed.ncbi.nlm.nih.gov/15981866/
* Handsfield et al2014 size/volume MRI study: https://pubmed.ncbi.nlm.nih.gov/24368144/
* MyoSim sources: https://github.com/MyoHub/myo_sim (installed PyPI0.2.5)
* Muscle paths and joints: Holzbaur2005 DOI10.1007/s10439-005-3320-7;
  Rajagopal2016 DOI10.1109/TBME.2016.2586891; MyoBack2025 source README.
* Simulation: https://arxiv.org/abs/2205.13600 and MuJoCo official docs.
* Ligament approach/limitations: https://arxiv.org/html/2410.05931v1
  (not reimplemented whole paper; upstream actual glenohumeral routes preferred).

No fabricated muscle volume, material constants, anatomy claims, EMG/GRF/load
observations. No claim to reproduce MRI-based continuum muscle strain from RGB.
User wants accuracy: use reference-anatomical assets and mechanics, quantify the
fit, and state the remaining unobservable subject-specific quantities.
