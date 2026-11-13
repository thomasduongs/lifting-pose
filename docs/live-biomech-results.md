# Live camera / Rust anatomical twin

Implemented and opened a native 1440x900 live window: **Rust anatomical model on
the left half, actual webcam feed on the right half**. Both content panes have equal
width. No recording/export/replay step is required.

## Open it

```sh
./scripts/live_biomech.sh
./scripts/view_biomech.sh --live
```

Or double-click `output/Live Biomech.app`. The camera-enabled app was built, signed
ad hoc with its camera usage description sealed, and checked on this Mac. Camera
authorization was refreshed specifically for the rebuilt Live Biomech app through
the normal macOS Camera settings and Quit & Reopen flow. It was left running with
the actual webcam; no microphone permission was requested.

Space freezes pose-following while the camera stays live. R resets the fit; F
toggles full/upper-body framing. Drag or scroll in the left pane to orbit/zoom.
Camera presets and anatomical layers are available through the top toolbar.
The video is unmirrored and letterboxed to preserve its aspect ratio.

## How it works

The root Rust binary reuses the existing OpenCV YOLOX person detector and MediaPipe
pose decoder. Separate camera and inference threads use single-slot latest-frame
mailboxes, so inference does not build a queue or freeze video presentation.

`biomech-lite::live_fit::LiveFitter` fits the eight native shoulder/elbow coordinates
from observed limb directions. A proper camera-torso-to-model-torso rotation avoids
left/right reflection, and directions are retargeted onto the reference segment
lengths. The solver uses a bounded, warm-start robust least-squares fit with temporal
priors and a 10 rad/s coordinate rate limit. Every accepted pose obeys native joint
bounds and exact polynomial couplings. An optimized marker-chain FK is tested against
the complete simulator FK. The viewer uses the existing Rust anatomical renderer.

No supported elbow: that arm holds its prior. Missing wrist: unobservable forearm
rotation/flexion hold while the supported upper arm can update. Missing or degenerate
shoulder basis holds the pose. Stale webcam poses older than 0.5 seconds are rejected
for updates. The UI distinguishes tracking, partial fit reasons, no supported arms,
paused, held/stale, lost and errors, and reports pose age and timing.

First-use macOS authorization may make OpenCV's AVFoundation opener return before
permission is granted. Camera startup therefore retries, with a visible waiting
message, every 0.5 seconds for up to 25 seconds. It uses AVFoundation explicitly on
macOS and preserves errors if a delayed inference result arrives. Camera errors
wrap into visible lines. Camera close/drop signals and joins workers and releases
the device. File-input tests are labeled as file input.

## Checks and observed performance

Validated November 12, 2026:

- **10 targeted release fitter tests passed**: marker-chain/full-FK parity, bounds,
  asymmetric pose recovery without reflection, mirrored-name rejection, a 120-frame
  constrained sequence, fixed priors, unsupported/NaN side hold, missing basis,
  invalid timestamps, extreme/rate-limited targets and fitting latency.
- **8 targeted release live-view tests passed**: equal split layout, aspect-preserving
  letterboxing, held/lost states, unsupported-arm status, mixed model/camera pixels,
  error text wrapping and explicit PNG file creation.
- The existing source video was processed through the real detector, pose model,
  fitter and shared renderer with `--live-snapshot`; visual inspection confirmed
  the model-left/video-right layout and supported-arm updates. It was not treated
  as proof of webcam operation.
- The **actual webcam was then verified in the native packaged application**. The
  right feed refreshed while left anatomical arm estimates changed. Space froze
  the model while the live video continued. Reset and no-supported-arm behavior
  were also observed. No live webcam frames were saved for this check.
- Observed webcam presentation about **29–30 fps**, ordinary inference **21–40 ms**,
  fitting about **0.1–0.3 ms**, and model rendering about **11–16 ms**. Pose age was
  about **28–53 ms** during fresh tracking and rose to approximately **284 ms**
  during detector refresh. These are sampled UI timings, not guaranteed throughput.
- File test: about 29 fps, inference 19 ms, fitting 0.3 ms, rendering 19 ms.
- CLI guards reject a snapshot without live mode, a live snapshot without file input,
  and windowed live mode with `--no-display`. These checks did not open a camera.
- Final bundle signature verification and `git diff --check` passed. An unrelated
  slow debug dynamics stress run was intentionally stopped; only the completed
  targeted release checks above are claimed for this change.

Reproduce the focused checks:

```sh
cargo test --release --manifest-path crates/biomech-lite/Cargo.toml --test live_fit
cargo test --release --bin lifting-pose biomech_live
./scripts/live_biomech.sh --input output/live-biomech/test-source.mp4 --live-snapshot output/live-biomech/file-split.png
./scripts/package_live_biomech.sh
```

Logs: `output/live-biomech/fit-tests.log`, `ui-tests.log` and `package-final.log`.
The source-video fixture is extracted locally from the existing recording; it is
not required to use the webcam. The packaged executable is **25,249,888 bytes** and
the app bundle is **83,738,995 bytes**, including ONNX weights, meshes and notices.
It uses this Mac's installed Homebrew OpenCV and is not a portable OpenCV distribution.
Python and MuJoCo are unnecessary at runtime.

## Practical limits

This is **live camera-driven constrained kinematics**, not identified muscle-force
simulation. Anatomy remains generic. Root, torso, hands and lower body are standing
priors; only the eight arm coordinates are fitted. Monocular depth, occlusion and
landmark misidentification can produce incorrect estimates even when confidence is
high. A torso/face close-up does not establish full-arm visibility: keep shoulders,
elbows and wrists in view. Large residuals and elbow-only fits are displayed; do not
interpret them as personal biomechanical accuracy. Synthetic strongly asymmetric
shoulder poses can retain several centimetres of marker error.

The app does not automatically record or upload camera frames. P and explicit
`--live-snapshot` are opt-in PNG writes. The old recorded viewer and headless simulator
are still available through their own commands.

## Duo ownership and usage

Claude owned **6 of 8 tasks**: Opus implemented and tested the constrained fitter;
Sonnet implemented capture, split UI and macOS camera startup handling; Haiku wired
CLI flags, launchers, packaging and controls documentation. Codex handled the contract,
integration, file-input checks and actual webcam/native-window verification.
Worker-reported API-equivalent usage estimate: **$3.16**, excluding Codex lead usage.


## Full tracking pipeline connection (2026-11-12)

Live inference now calls the same `tracking_pipeline::track_person` function as recording
analysis. The packaged app includes RTMPose, palm/hand and MotionAGFormer weights in addition
to the existing body and detector models. Camera framing stays 1280×720 with no display crop.

Validation after this integration:

- Root release suite: **145 passed, 22 ignored, zero failures**. This includes confidence
  gating and a streaming-vs-recording equivalence test; streaming retains no observations
  and only the current subject while preserving calibration and continuity.
- Explicit real-model file test: **20 distinct frames, 20 supported anatomical fits,
  57 accepted arm refinements, 31 hand observations, 41 image-constrained depths**.
  These counts describe that local clip segment, not accuracy against ground truth.
- First inference included model preparation (~933 ms). Subsequent samples were ~40–101 ms.
  Camera capture uses a separate latest-frame mailbox, independent of this tracking latency.
- After adding exact source-frame pairing for file snapshots and hiding stale live overlays,
  targeted live tests: **9 passed, 1 fixture test ignored by default**. The rebuilt packaged
  executable also rendered the file snapshot successfully from outside the project directory.
- Inferred depths are capped at lower confidence. Predicted/missing points stay excluded.
  The anatomical fit still updates eight arm coordinates; finger/torso/lower-body coordinates
  remain reference priors, and no forces or activation are measured.

Reproduce the explicit model test:

```bash
cargo test --release --bin lifting-pose full_pipeline_file_input -- --ignored --nocapture
```

Integration logs: `output/live-biomech/full-pipeline-tests.log`,
`output/live-biomech/full-pipeline-e2e.log`, `output/live-biomech/package-full-pipeline.log`.


## Partial-arm stability and finger rigging (2026-11-12)

- Arm tracking retains a stable torso frame when one shoulder disappears and permits
  either arm to start independently with a frontal orientation prior. Missing arms hold.
- Weak/offscreen image joints invalidate native depth before the shared skeletal fit.
  Reliable native observations stay independent of opposite-arm skeletal corrections.
  Detector crop segment changes no longer reset the live fit during continuous tracking.
- Confirmed hand world landmarks drive all forty native finger coordinates, including
  both thumbs. Palm-local direction IK preserves reference bone geometry, range bounds,
  smooth per-hand updates and missing-hand holds; predictions are excluded.
- Core release tests: 11 arm tests plus 2 hand tests passed. The synthetic two-hand IK
  benchmark averaged about 1.1 ms/update. Tests cover one-arm startup, opposite-shoulder
  disappearance, bilateral curls/spread, held missing hands and unchanged non-finger joints.
- Real-model file test: 20 frames, 19 supported arm fits, 35 supported hand fits; finger
  coordinates moved from reference. This is integration evidence, not ground-truth accuracy
  or a live webcam validation of the reported motion.

Logs: `output/live-biomech/arm-hand-tests.log`, `arm-hand-ui-tests.log`, `arm-hand-e2e.log`,
`package-arm-hand.log`. The anatomical model still holds root, torso, wrist orientation
and lower body; camera motion does not measure muscle force or activation.


## Stationary-arm filtering (2026-11-12)

Live anatomical input previously used raw native depth even when stabilized image joints
were quiet. Added independent image-anchored stillness filtering, resolution-scaled five-pixel
noise bands, speed-dependent smoothing, and retention through wrist-confidence dropouts.
The camera image remains uncropped; only tracking overlays/kinematics are filtered. The arm
solver now retains converged coordinates for unchanged targets instead of repeatedly
optimizing unobservable rotation. Deliberate movement releases the hold.

Targeted release validation: 3 stillness-filter tests and 12 arm-fitting tests passed.
Cases include independent 8 cm synthetic depth noise with stationary image evidence,
confidence dropouts, missing-arm independence, and resumed deliberate motion after a hold.
These are synthetic/integration checks, not a measurement of live webcam jitter reduction.
Logs: `output/live-biomech/stillness-tests.log`, `stillness-core.log`,
`stillness-e2e.log`, `package-stillness.log`.


## Skeleton-driven inverse muscle estimation (2026-11-12)

Implemented option 1: camera tracking/finger IK prescribe the skeleton; an independent
latest-only worker estimates required arm torque, bounded muscle activation and tension.
No estimated force feeds back into the pose. The live model panel colors supported muscle
paths by estimated activation and lists tension, unexplained torque, source age and optimizer
status. E / EST toggles analysis. Unsupported, stale, paused or infeasible estimates remain
unknown. Assumptions stay visible: generic anatomy, unloaded hands, fixed torso, eight reduced
shoulder/elbow coordinates; no measured EMG, fiber FEM, whole-body or finger-force inference.

Validation:

- Root release suite: **161 passed, 23 ignored, zero failures**.
- Numerical recruitment: **8 integration tests + 4 optimizer tests passed**. Known torque
  reconstruction and force identity, infeasible-demand residuals, masks, redundant muscle
  sharing, permutation invariance and bounded activations are covered.
- Renderer activity color/validation: **2 passed**. Nine worker tests and thirteen live tests
  are included in the root result, covering reduced velocity indexing, independent per-arm
  warm-up/failure handling, stale-result gating and unchanged prescribed qpos.
- Three static model poses converged with torque residuals below 0.001 Nm. Core solve average
  was about 3 ms for the 416-muscle model. This validates numerical consistency, not biological
  accuracy or subject-specific recruitment.
- Real-model file pipeline: **35 fitted frames, 33 estimates, 28 dynamic, 23 feasible**.
  Larger unexplained torques are exposed and those activations are not displayed. Actual
  solve samples were about 2.2–5.9 ms. The test checked that estimates never move qpos.
- Rebuilt the signed local app and rendered a packaged file-input snapshot from outside the
  project directory, pairing muscle results and anatomy with the exact source frame.
  No real camera or desktop verification was used for this change.

Reproduce:

```bash
cargo test --release --bin lifting-pose
cargo test --release --manifest-path crates/biomech-lite/Cargo.toml --test inverse
cargo test --release --manifest-path crates/biomech-lite/Cargo.toml --lib inverse::tests
cargo test --release --bin lifting-pose tracked_file_drives_dynamic_muscle_estimates -- --ignored --nocapture
./scripts/live_biomech.sh
```

Logs and packaged preview: `output/muscle-estimates/`. Technical assumptions/equations:
`docs/muscle-estimates.md`. Codex designed and integrated the pipeline; Claude Sonnet built
the inverse solver/optimizer and corrected the runtime adapter; Claude Haiku built the initial
worker, rendering colors and documentation. Claude's reported incremental API-equivalent
usage for this run was approximately **$2.08**, excluding the Codex lead. This is a reported
API-equivalent amount, not a subscription-usage percentage or billing charge.
