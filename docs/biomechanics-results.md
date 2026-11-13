# Biomechanical human model: delivered results

A working **full-body musculoskeletal reference simulator** now sits beside the
original pose pipeline. It includes sourced anatomical meshes, articulated joints,
musculotendon routes and physiological actuation, with explicit shoulder-ligament
coverage. The user's selected priority was biomechanical simulation.

This is not a validated reconstruction of the recorded person's internal anatomy.
The distinction matters: accurate reference mechanics and a small image residual
do not identify individual muscle geometry, strength, tendon compliance or ligament
properties from RGB video.

## Play the new anatomical model

From the repository folder:

```sh
.venv-biomech/bin/mjpython scripts/replay_biomech.py output/biomech-twin --loop
```

This is the new MuJoCo anatomical playback. `cargo run --release -- --replay`
continues to show the original skeletal twin. Playback prescribes fitted joint
angles; the recording is not reproduced by a solved muscle-control policy.

Artifacts:

* [Front animation](../output/biomech-twin/front.mp4)
* [Side animation](../output/biomech-twin/side.mp4)
* [Anatomy preview](../output/biomech-twin/preview.png)
* [Fitted model XML](../output/biomech-twin/model.xml),
  [native binary](../output/biomech-twin/model.mjb),
  [anatomy inventory](../output/biomech-twin/anatomy.json) and
  [provenance](../output/biomech-twin/provenance.json)
* [Motion and mechanics fit report](../output/biomech-twin/fit_report.json)
* [Ligament sensitivity results](../output/biomech-ligaments/ligament_sensitivity.json)

## Anatomical and computational coverage

| Component | Delivered content |
| --- | --- |
| Reference library | myo-sim 0.2.5, MuJoCo 3.17.0 |
| Computational bodies | 104, including world, phantom frames and grouped anatomy |
| Native joints | 123; this is not the independent anatomical DOF count |
| Native equality constraints | 51: 49 joint relationships and two locks |
| Independent coordinates | 71 scalar coordinates plus six free-root velocity coordinates |
| Actuation | 416 muscle actuators with native activation and force-length-velocity behavior |
| Spatial paths | 424: 416 musculotendon paths and eight named shoulder-ligament routes |
| Geometry | 104 source meshes, 102 mesh geoms |
| Reference total mass | 84.265 kg; not the recorded person's measured mass |
| Shoulder ligament mechanics | Four optional sourced SGHL/CHL springs; four other routes unparameterized by this profile |

The skeleton includes the shoulder girdle, arms, forearms, wrist/hand bones,
lumbar/trunk structures, pelvis and legs. These component counts must not be
presented as a count of all 206 human bones or all distinct human muscles.
The source head is rigidly attached; detailed cervical articulation is absent.
Muscle compartments are force generators acting along wrapped paths, not scanned
solid muscle bellies or resolved fascicles.

## Recording fit

Eight independent shoulder/elbow coordinates were fit using native joint limits
and couplings. Root yaw is derived from estimated shoulder geometry. Root
translation, trunk/neck articulation, forearm twist, wrist/finger pose and lower
body remain reference priors. The original unstable finger quaternions are not
copied into the new anatomical joints.

The fitter combines supported 3D targets and fresh 2D image observations from the
saved log. When depth is rejected, the image point can still constrain projection;
its depth remains inferred. All observations retain provenance and support masks.

| Quantity | Final result |
| --- | --- |
| Source frames | 222 / 222 |
| Recorded time span | 8.0593 s, original irregular timestamps retained |
| Frames with supported 3D arm targets | 147 |
| Frames with valid 2D arm observations | 221 |
| Accepted arm-joint image observations | 806 |
| Frames with an arm held for lack of supported observations | 6 |
| Root-yaw supported / held frames | 207 / 15 |
| Reference-model / recording arm-length scale | 1.0592; recording-to-reference alignment, not anatomical scaling |
| Supported 3D joint residual, median / mean | 17.0 / 28.6 mm in reference-model coordinates |
| Supported 3D residual, 95th percentile / maximum | 106.9 / 230.0 mm |
| Image reprojection residual, median / RMS | 7.42 / 20.18 pixels |
| Image reprojection residual, 95th percentile / maximum | 28.93 / 191.22 pixels |
| Native joint-limit violations in exported poses | 0 |
| Native coupling residual | 0 |
| Estimated fitted-coordinate transitions above 20 rad/s | 11 |

The outliers are material. This is not an accuracy claim for every frame, and the
remaining fast coordinate changes can reflect ambiguous orientation, tracking
error or motion. They are not verified true human joint velocities. Image
observations themselves are network estimates; camera intrinsics may be assumed.
Neither the residuals nor the smoothness of playback establish 3D ground truth.
No hard angular-speed cap was used to disguise incompatible observations.

## Ligament mechanics and uncertainty

The native library contains eight anatomical routes but gives them zero stiffness.
The delivered optional `boardman1996` profile assigns bilateral superior
glenohumeral and coracohumeral routes aggregate cadaver linear stiffness values
from [Boardman et al.](https://pubmed.ncbi.nlm.nih.gov/8872921/): 17.4 and 36.7 N/mm,
converted to 17,400 and 36,700 N/m. MGHL/IGHL stiffness is not inferred from those
unrelated tissues and remains disabled in this profile.

Native tension-only springs use `F=k*max(0,L-Lslack)`. Slack lengths are inherited
reference choices, not measured personal values. The sensitivity experiment
exports 972 route/configuration/pose rows; passive generalized force agrees with
the tendon-Jacobian calculation. It varies assumed slack and published stiffness
variation to expose uncertainty rather than hide it.

In some imposed poses the model produces approximately 245 N on a left SGHL route
and 409 N on a left CHL route. Those exceed the corresponding **mean ultimate
loads** reported by the reference experiment. The unbounded linear-spring model
does not model tissue failure, so these are extrapolations, not valid injury or
personal-strength predictions. A clinically meaningful passive-tissue model
would require calibrated geometry/slack and an appropriate nonlinear material law.

## Verification and portability

The initial combined biomechanics suite passed **63 tests**. A new irregular-clock
regression brought coverage to **64 distinct tests**; all ten renderer tests
passed after that integration change. The 23 anatomy/ligament tests passed again
after fixing metadata that could otherwise retain a pre-profile stiffness setting.
No broad Rust or unrelated FEM suite was required.

Export checks confirmed all 222 source indices, finite poses/mechanics, unit root
quaternions, strictly increasing timestamps, zero source-joint-limit violations,
and matching native couplings. Replay's `--inspect` passed without opening a UI.
The two H.264 videos have 243 frames at 30 fps, with previous-pose holds on the
original clock and the last source pose included.

The fitted binary is the exact MuJoCo-version-specific model snapshot. Its portable
XML/assets copy carries the same four-route profile and matches checked model
parameters within 1e-6 absolute serialization tolerance. Sampled muscle-path
differences were below 0.25 micrometers; sampled passive-force differences were
negligible. Third-party licenses, notices and model READMEs accompany both exports.
See [workflow](biomechanics.md) for setup and commands.

## Research, ownership and remaining accuracy requirements

[Research notes](biomechanics-research.md) trace decisions to the UVA M3 lab,
Blemker's muscle-architecture studies, Holzbaur/MoBL, Rajagopal, MyoSim/MyoSuite,
and the ligament literature. This implementation does not claim to reproduce
MRI-based continuum muscle strain or to recover all human connective tissues.
Series tendon compliance, cartilage and detailed fascicle architecture are absent.

Codex led research, contracts, integration and numerical/export corrections.
Claude completed nine of eleven work items: Opus built anatomy and force
experiments, Sonnet built fitting and ligament profiles, and Haiku built rendering,
native replay and their documentation. Worker CLI usage estimates total about
**$8.26**, excluding the Codex lead session.

Further personalization needs measured anatomy/anthropometry, calibrated motion
evidence, task loads/contact data and independent validation appropriate to the
intended use. The delivered reference simulator is usable for controlled
experiments; its existence does not establish a highly accurate personal twin.
