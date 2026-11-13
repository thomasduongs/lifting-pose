# Live Biomech: live anatomical twin

A native Rust window with the anatomical model on the **left half** and the camera on the
**right half**. The camera drives constrained kinematics of the model's arms through the
shared recording/tracking pipeline (YOLOX, MediaPipe, RTMPose arm refinement, dedicated
hand tracking, occlusion checks, image stabilization, and skeletal fitting). No recording or replay step is required. Camera
frames stay in memory; PNG files are written only when you request P or `--live-snapshot`.

- Left: "Model - live pose fit", the biomech-lite reference model (held lower body and reference anatomy).
- Right: "Camera - live", the raw camera image.
- Header: "LIVE ANATOMICAL TWIN".

The muscle panel now shows **estimated** activation and tension for supported unloaded arms,
using reduced inverse dynamics and bounded muscle recruitment. These are model estimates;
the viewer does not measure muscle activity or run muscle-driven forward motion.
The fit is kinematic. It places the model's upper-limb joints to match the camera's shoulder,
elbow and wrist positions, within the model's joint limits. Eight arm coordinates and forty finger coordinates are fitted. Root, torso, wrist
orientation and lower body keep their reference priors.

## Muscle estimates

Live mode uses the tracked skeleton as prescribed motion, then estimates the torques and muscle
recruitment needed for it. Muscle paths are colored by estimated activation: **blue 0%, green 25%, yellow 50%, orange 75%, red 100%, gray unknown**. The model pane lists up to four muscles with activation percentage and
estimated tension in newtons, the unexplained torque in Nm, optimizer status and source age.
Press **E** or the **EST** button to toggle this analysis; movement and camera tracking continue.

The background estimator uses a single pending pose, bounded derivative history and independent
arm confidence gates. First samples use a labelled quasi-static estimate while motion history
warms up. Unsupported, stale, paused and physically infeasible estimates are shown as unknown.
A solver iteration limit is reported explicitly even when torque matching is already close.

The current mechanical scope is the reference model's eight shoulder/elbow coordinates,
fixed torso, unloaded hands and no external contacts. Finger motion is tracked visually but
finger forces, whole-body loading and actual personal muscle recruitment are not inferred.
See [muscle-estimates.md](muscle-estimates.md) for equations, validation and assumptions.

## Tracking connection

Both live mode and recording analysis call `tracking_pipeline::track_person`; there is no
separate simplified inference path. The detector uses the existing closest-person selection
and detection cache. Each observation runs body inference, independent arm refinement,
palm/21-landmark hand inference, arm/hand occlusion gates, hand wrist merging, image motion
tracking, confidence gating, adaptive smoothing, 3D estimation, and the original skeletal twin
fit/calibration. RTM primary models use MotionAGFormer and the existing dimension tracker;
MediaPipe uses its native depth and original direct-depth tracking rules.

The anatomical arm retargeter uses reliable native limb observations and accepted skeletal-fit
image corrections. Native depth is visibility-gated before skeletal fitting, so an invisible
limb cannot contribute confident depth to the shared fit. Each arm updates independently.
The last stable torso frame is retained during partial views; a fresh one-arm session starts
with a frontal torso prior. Detector crop changes preserve live fitting state.
The anatomical arm retargeter consumes these gated observations. Accepted 2D corrections can constrain
that fit when native depth is rejected; those resulting depths are **inferred**, shown in the
status, and capped at lower confidence. Missing or display-only predicted joints do not become
fresh constraints. Confirmed hands appear green in the camera overlay; predicted hands are gray.
Confirmed 21-landmark hands also drive the model's native finger hinges: MCP flexion/spread,
PIP/DIP flexion and thumb CMC/MCP/IP motion. Each digit is fitted from palm-local bone
directions, respecting the source model's joint ranges. Missing hands hold their last pose;
predicted hand overlays never drive these joints. Wrist orientation remains a reference prior.

Stillness filtering compares each arm's current image landmarks with a retained anchor.
Within a five-pixel band at 720p (scaled for capture height), uncertain depth updates cannot
move the model or its arm overlay. Displacement accumulates against that anchor, so small
real movement eventually releases the hold; faster movement uses less smoothing. Wrist
confidence dropouts do not reset the elbow anchor. Missing observations remain missing.
The arm solver stops updating converged, unchanged targets. `--no-smoothing` or
`--no-stabilization` bypasses the image-based stillness filter.

The streaming skeletal fitter retains calibration and continuity state without storing frame
history. Separate camera/inference workers retain only the latest pending frame, so slow
inference cannot create an expanding backlog. The full camera frame remains uncropped.

## Run it

```bash
scripts/live_biomech.sh                      # builds the release binary, opens the webcam (camera 0)
scripts/live_biomech.sh --camera 1           # another camera index
scripts/view_biomech.sh --live               # same as live_biomech.sh; the old viewer stays the default
```

Flags are forwarded to the binary. Useful ones:

| Flag | Meaning |
| --- | --- |
| `--input PATH` | Video or image file instead of the webcam. Never opens a camera. |
| `--live-snapshot PATH` | Headless: render one processed frame to a PNG and exit. Requires `--input`. |
| `--max-frames N` | Stop after N processed frames (smoke test). |
| `--biomech-model PATH` | biomech-lite `model.json` (default `assets/biomech-lite/model.json`). |
| `--biomech-visual PATH` | `visual.json` (default `assets/biomech-lite/visual.json`). |
| `--model PATH` | MediaPipe pose ONNX (default `models/mediapipe-pose.onnx`). |
| `--detector-model PATH` | YOLOX person detector (default `models/yolox-s.onnx`). |
| `--arm-model PATH` / `--no-arm-refinement` | RTMPose refinement model or explicit opt-out. Enabled by default. |
| `--hand-model PATH` / `--no-hands` | Dedicated hand tracking model or explicit opt-out. Enabled by default. |
| `--no-stabilization` / `--no-smoothing` | The same tracking/filter controls as recording analysis. |
| `--full-body` | Track lower-body observations too; the anatomical retargeter still fits eight arm coordinates. |
| `--free-torso` / `--focal-length-px N` | Existing skeletal-fit assumptions and camera calibration. |
| `--head-orientation` | Explicitly opt into sparse head orientation; off by default. |
| `--no-detection` | Skip the person detector and infer over the full frame. |
| `--opencv-threads N` | OpenCV CPU threads (default at most 4, to leave room for rendering). |

`--live-biomech` conflicts with `--record`, `--replay`, `--output` and `--json`. Live mode
does not save automatically. `--live-snapshot` without `--live-biomech` is an error, and
`--no-3d` is rejected because anatomical fitting requires 3D.
`--no-display` is rejected in live mode unless `--live-snapshot` is also given.

## Controls

| Key or mouse | Action |
| --- | --- |
| E / EST | Toggle background muscle estimates. |
| Space | Pause pose-following. The camera keeps running live. |
| R | Reset the fit. |
| F | Toggle full-body / upper-body framing. |
| 1 / 2 / 3 | Switch views. |
| B / M / G | Toggle bones, muscle, and ligament layers. |
| P | Save a PNG to `Pictures/Biomech Viewer`. This is the only write the window makes, and only on request. |
| Esc | Exit and release the camera. |
| Mouse drag / scroll (left half) | Orbit and zoom the model. |

Status shows camera FPS, pose age, which arms are supported, and inference and fit latency.
If no pose arrives for more than 0.5 s the model shows **HELD** or **LOST**. The camera keeps
refreshing while inference runs. The camera image is never mirrored.
Capture requests 1280×720 widescreen. The complete returned frame is letterboxed
in the camera pane, with no display crop or digital zoom.

Framing is fixed to a standing full-body view with arm margin, and it is not refitted every
frame. Use Space or R when the subject moves out of frame or the fit drifts.

## Privacy

- Processing is local. The app does not record or upload frames. PNG saving is explicit.
- No microphone access is requested.
- The first run triggers the macOS camera prompt for the app (or the terminal, when run from
  `scripts/live_biomech.sh`). Deny it and the window shows a permission message instead of video.

## Testing without a webcam

Use file input to test the whole path, including the split layout and fit, without touching hardware:

```bash
scripts/live_biomech.sh --input output/live-biomech/test-source.mp4
scripts/live_biomech.sh --input output/live-biomech/test-source.mp4 --live-snapshot /tmp/live.png
```

The file input is labelled as file input in the UI. It is not a webcam test. The source clip
has an upper-body person and estimated depth, so do not read it as proof of full-body fit.

## macOS app

```bash
scripts/package_live_biomech.sh
open "output/Live Biomech.app"
```

The app is a local, unsigned bundle (`local.liftingpose.live-biomech`):

- `Contents/MacOS/live-biomech` is a native copy of the release binary. It is not a shell wrapper,
  so the camera permission belongs to the app. The binary enables live mode when its file name is
  `live-biomech`.
- `Contents/Resources` holds the ONNX pose and detector weights, `assets/biomech-lite` model,
  visual data, NOTICE and provenance. Assets resolve in this order: the path as given, then
  `Contents/Resources/<path>`, then the project root. An explicit path that does not exist is an error.
- `Contents/Frameworks` holds any ONNX Runtime dylibs the binary links through `@rpath`.
- OpenCV is **not** bundled. The app uses the Homebrew OpenCV installed on this Mac, so it is
  not portable to another machine. The packager prints any other local dependency it finds.

Rebuilding the package moves an existing `Live Biomech.app` aside. It never overwrites it.

Development builds use an ad-hoc signature. After rebuilding, macOS may require
camera authorization for the new executable even if Live Biomech still appears enabled.
In that case, refresh Live Biomech in System Settings > Privacy & Security > Camera
and choose Quit & Reopen. The app waits up to 25 seconds for first-use authorization.

## Performance

Target is an interactive preview at 20 to 30 fps. The fit may run slower than the camera. Measured
rates are shown in the status line and should be reported as measured, not assumed.

## First camera use (macOS)

On first launch macOS shows the camera permission prompt. The app keeps retrying to open camera 0 (AVFoundation) for up to 25 s and shows "waiting for camera / permission" in the right pane. If it still cannot open, a wrapped error explains how to grant access in System Settings > Privacy & Security > Camera; restart afterwards. The packaged app is signed ad hoc as a whole bundle (identifier `local.liftingpose.live-biomech`) so the camera usage description is sealed. File input is unaffected.
