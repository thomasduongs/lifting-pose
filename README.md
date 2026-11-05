# lifting-pose

Rust CLI using **OpenCV** for capture, preprocessing, ONNX inference, and skeleton
rendering with **RTMW-m WholeBody** (default) or **RTMPose-m**, followed by
**MotionAGFormer-XS** temporal 3D body lifting. Runs on CPU; no Python inference environment or
separate ONNX Runtime installation is required.

A skeletal twin fits the 3D observations using shared body dimensions and
per-frame rotations, preserving bone lengths through forward kinematics.

## Record and replay a 3D twin

Record your webcam into a new session directory:

```sh
cargo run --release -- --record output/session-01 --camera 0
```

The recording preview opens immediately after the camera is ready. Press **q**
or **Esc** to stop recording; the app then processes the entire saved video and
finalizes the twin. Capture runs without neural inference so model processing
does not reduce capture frame rate. Analysis takes additional time after capture.
Use a new session directory for each recording; existing session artifacts are
not overwritten by record mode.

The session contains:

- `source.mp4`: unannotated footage (silent; at most one border row/column is
  cropped when the source dimensions are odd, for MP4 encoding).
- `capture.json`: reported capture FPS and per-frame timestamps.
- `tracked.mp4`: annotated pose/twin preview.
- `twin.json`: shared body dimensions, per-frame rotations, root estimates, and
  recording timing used by replay.

For a timed capture, add `--record-seconds 10`. A headless camera recording
(`--no-display`) requires `--record-seconds` or `--max-frames`. Stop normally with
q/Esc or the duration/frame limit so video and twin files can be finalized.
Recording needs the same local model files as tracking.

You can also import an existing video into a session and build its twin:

```sh
cargo run --release -- --record output/session-02 --input lifting.mp4 --no-display
```

Replay the finalized 3D skeleton:

```sh
cargo run --release -- --replay output/session-01/twin.json
```

Replay reconstructs joint positions using the saved hierarchy, shared rest
offsets, and quaternion forward kinematics. **No neural models, Python, camera,
or original video are required for replay.** Recorded timestamps determine
playback timing. Frames without a reliable saved pose show an empty scene;
no poses are invented across tracking gaps.

| Replay key | Action |
| --- | --- |
| Space | Pause / play; restart after the end |
| J / L | Previous / next frame (pauses playback) |
| A / D | Orbit left / right |
| W / S | Tilt the view |
| + / - | Zoom |
| 1 / 2 / 3 | Front / side / oblique view |
| F | Toggle in-place view / approximate root motion |
| N | Toggle joint names |
| [ / ] | Decrease / increase playback speed |
| R | Reset view |
| Q / Esc | Close replay |

Select another tracked subject segment with `--replay-subject 1`. Replay covers
the selected segment's first through last reliable frame, retaining internal
gaps. Older twin files without timing metadata default to 30 FPS; override timing
with `--replay-fps 15`. The 3D view shows the 17-joint skeletal body, not a skin
mesh or 3D fingers/toes. Root motion remains the approximate camera-space estimate
described below.

Export a shareable video of the 3D replay without opening a window:

```sh
cargo run --release -- --replay output/session-01/twin.json \
  --output output/session-01/replay3d.mp4 --no-display
```

Replay video exports use a 1024×768 oblique in-place view. Irregular capture
timestamps are resampled to constant FPS for MP4. Remove `--no-display` to export
first and then open interactive replay. `--max-frames` limits the source-frame
span for preview/export. The source video and twin JSON are protected against
being selected as the replay output.

Supports images, video files, and webcams, annotated image/video output, and
JSON Lines keypoint export. The default model uses 192×256 input (width × height)
and predicts 133 COCO-WholeBody landmarks:

- 17 body points.
- 6 foot points: heel, big toe, and small toe on each foot.
- 42 hand points: a hand root and 20 finger joints on each hand.
- 68 face points.

Hand skeletons include finger chains and palm connections. These landmarks can
help estimate palm position/orientation, but the model does not output a palm
surface or 3D rotation. Feet are cyan, left hands orange, right hands pink,
and face points gray. Small or occluded hands/feet can have low confidence;
use a close, clear view and a person crop where possible.

**Automatic person detection is enabled by default.** OpenCV Zoo's YOLOX-s
detector finds people in each frame, filters out other object classes, and removes
duplicate boxes with non-maximum suppression. Only the person with the largest
visible bounding-box area is passed to the pose model, producing at most one
bounding box and skeleton per frame. Box size approximates who is closest to the
camera; it does not measure depth and can be affected by body size, posture, or
occlusion. Detection confidence breaks ties between equally sized boxes.

Use `--bbox x,y,width,height` to bypass detection with a fixed manual crop, or
`--no-detection` to estimate one full-frame pose. Detection runs on the first
frame and every **15 frames** thereafter by default. Between refreshes, the learned
skeletal twin can move the selected crop gradually to follow the person while
pose estimation and angles update every frame. Before calibration is ready, or
when the fit is unreliable, the last crop is reused. The selected person can change at a refresh when someone else has a larger
box. Empty detections are also cached until the next refresh; resolution changes
force immediate detection. A moving person can leave the cached crop between
refreshes, so use a smaller interval for fast motion. Persistent identity tracking and lifting form analysis are not implemented.

Joint angles are displayed automatically beside the selected person's joints.
With 3D lifting enabled, yellow shoulder, elbow, hip, and knee labels use fitted
**3D interior angles** and carry a `3D` suffix. Wrist and ankle angles retain a
`2D` suffix because MotionAGFormer has no finger or toe outputs. A small inset
shows an oblique view of the fitted skeletal twin. The existing JSON
`angles` field still contains all twelve 2D angles; 3D angles are separate.

Use `--no-3d` to disable lifting and display only 2D labels. These are unsigned **2D interior
angles** (0–180°) measured in the image plane: a straight joint is 180°, not 0°.
Perspective and camera angle affect the measurements; these are not 3D joint
rotations or anatomical flexion/extension measurements.

| Angle (each side) | Landmark triplet (middle landmark is the vertex) |
| --- | --- |
| Shoulder | Elbow → shoulder → hip |
| Elbow | Shoulder → elbow → wrist |
| Wrist | Elbow → wrist → first middle-finger joint |
| Hip | Shoulder → hip → knee |
| Knee | Hip → knee → ankle |
| Ankle | Knee → ankle → big toe |

Wrist and ankle measurements require the whole-body model. Ankle angles use the
ankle-to-big-toe direction as a foot approximation. All three landmarks must be
inside the frame and meet `--threshold`; missing, low-confidence, or coincident
landmarks produce no label and a JSON `degrees: null` value.

## MotionAGFormer 3D body estimation

The default 3D model is the official **MotionAGFormer-XS H36M** checkpoint,
with 27 frames and 17 Human3.6M joints. The first 17 COCO body landmarks from
RTMW/RTMPose are converted to H36M, including synthesized pelvis, spine, thorax,
and head input landmarks, then normalized using the frame width and aspect ratio.
Conversion follows the [official demo preprocessing](https://github.com/TaatiTeam/MotionAGFormer/blob/master/demo/lib/preprocess.py).

The app keeps the latest 27 poses and uses the model's newest-frame output.
Short histories repeat the earliest pose on the left; the inset displays the
number of real history frames collected. This is a streaming adaptation of the
official full-clip demo and uses no future frames or flip augmentation. Its
accuracy may differ from the published offline evaluation. It adds computation:
in a short local CPU test, cached-detector frames with 3D lifting took about
72–90 ms, versus roughly 37 ms without lifting in an earlier test. These are
illustrative timings, not a controlled benchmark.

`pose3d` in each person's JSON contains 17 named XYZ `joints`, eight body `angles`,
`history_frames`, and coordinate metadata. The pelvis is the origin; coordinates
are **root-relative camera-space model units**, not meters, world coordinates,
or camera distance. Each joint's `input_score` comes from the 2D input; it is not
a confidence produced by the 3D model. The model predicts body structure, not
3D fingers/toes or measured spine curvature. Raw `keypoints` and `pose3d` are
preserved; dimension-guided image-space results are exported separately under
`tracking`. The model does not supply persistent person identities.

History resets when no person is selected, the torso becomes unreliable, frame
resolution changes, or the selected box has little overlap with its predecessor.
That overlap check reduces mixing across person switches but is not an identity
tracker. Invalid limb inputs reuse the previous coordinate with zero input
confidence (or the pelvis on the first frame). If the torso is unreliable,
`pose3d` is null and the display falls back to 2D angles. Low-confidence input
joints suppress affected 3D angle labels even when the model predicts XYZ.

## Skeletal twin: person dimensions and frame pose

The twin stores **one person model per subject segment** and a sequence of frame
parameters. During capture, reliable new observations refine the shared person
estimate. These revisions are provisional estimates of one body, not independent
limb lengths per frame. On normal completion (including **q** / **Esc**), the app
refits every saved observation against the **same final dimensions** and writes
`output/twin.json`. Use `--twin-output path.json` to choose another destination.
The destination is overwritten and its parent directories are created if needed.
`--no-3d` disables both lifting and twin recording.

| Stored data | Parameters |
| --- | --- |
| Person (shared across finalized frames) | Pelvis width, shoulder width, segment lengths, rest offsets, calibration revision and sample counts |
| Frame | Root position/orientation, local joint quaternions, fitted 3D angles, fit error, and camera approximation metadata |

Dimensions use the MotionAGFormer model's relative units. They are not calibrated
centimeters or meters. Left/right upper legs, lower legs, upper arms, and forearms
share lengths as a bilateral symmetry prior. The remaining shared segments are
pelvis→spine, spine→thorax, thorax→nose, and nose→head; these are the model's joint
definitions, not measurements of vertebrae or head surfaces.

Calibration starts from the first valid 3D observation. Refinement begins once
MotionAGFormer has 27 real history frames. Both endpoints must meet confidence
`max(--threshold, 0.6)`. Each dimension starts with a confidence-weighted median
over its first 31 accepted measurements, then uses a running estimate with
bounded residuals to reduce outlier influence. Sample counts expose which
dimensions have supporting observations; zero-count dimensions retain their
initial estimate. More observations can reduce noise but do not remove systematic
errors in the 3D model or guarantee true physical proportions.

**Forward kinematics** uses the shared rest offsets and parent hierarchy:

```text
position[root] = root_position
rotation[root] = root_orientation
position[j] = position[parent[j]] + rotation[parent[j]] * rest_offset[j]
rotation[j] = rotation[parent[j]] * local_rotation[j]
```

Rotations act on outgoing bones. The root local rotation is identity; its global
orientation is stored separately. Quaternions use **[x, y, z, w]**, active
rotations, and parent-local composition. The exported `joint_names`, `parents`,
and shared `rest_offsets` are sufficient to reconstruct every frame without the
neural model. The root has a null parent.

**Inverse kinematics** fits rotations to confidence-weighted 3D observations
with a robust loss, damped least-squares rotational updates, and bounded steps.
It warms up from the previous solution when helpful and never changes dimensions
inside a pose solve. `fit_rmse` reports the weighted observation residual in model
units. Missing/low-confidence observations have zero weight. Joint-center fitting
does not identify axial twist uniquely, and the current model uses unconstrained
rotational joints; anatomical joint limits, meshes, skin, and contact constraints
are not implemented.

MotionAGFormer removes global root translation. The twin therefore approximates
root position by fitting a **weak-perspective camera** to fitted 3D joints and
their 2D image observations. It assumes square pixels and the image center as
principal point. By default, focal length equals image width; provide a calibrated
value with `--focal-length-px 900`. Each frame records whether focal length was
assumed and whether the translation fit was valid. An invalid translation uses
zero position with `root_translation_valid: false`. Positions are in an
approximate camera coordinate system (x right, y down, z forward), not measured
world trajectories. Camera movement is not separated from person movement, and
supplying focal length alone does not establish metric scale.

Streaming `--json` output keeps the raw `pose3d` observations and adds a `twin`
preview containing the current provisional person parameters and fitted frame.
Finalized `--twin-output` contains `subjects`, each with one `person` and many
`frames`; person dimensions are never duplicated or adjusted inside those frames.
Only frames with valid 3D observations are included. A large selected-box jump,
resolution change, or a gap over 30 processed frames starts a separate subject
segment to reduce mixing between people. This is a heuristic, not re-identification.
3D observations are retained in memory until completion, and final refitting adds
processing time proportional to the recording length. Abrupt termination before
normal completion does not finalize the twin file.

```sh
cargo run --release -- --input lifting.mp4 --no-display \
  --json output/preview.jsonl --twin-output output/twin.json
```

### Using dimensions during tracking

Dimension-guided tracking is enabled by default. Once the twin has at least five
accepted calibration frames (after the 27-frame 3D warmup), its fixed-length
fitted skeleton is projected into the image using the estimated camera. Feedback
is rejected if the 3D fit error is high or the projected torso disagrees with the
observed torso. Three reliable torso landmarks are required for this check.

- Strong 2D body observations are preserved. Uncertain visible body points can
  blend toward the fitted skeleton with a bounded correction.
- Missing body points can use model-inferred positions for up to five consecutive
  frames with a trusted projection. They appear as gray hollow markers/links,
  keep zero confidence, and do not produce measured 2D angles.
- A padded crop follows the fitted body and reliable observed extremities.
  Movement and resizing are gradual and clipped to the image. The detector still
  runs every 15 frames and refreshes the selected person. Manual `--bbox` and
  `--no-detection` crops remain fixed.

The projection respects 3D segment lengths, rather than enforcing constant 2D
pixel lengths, which would be incorrect for limbs pointing toward the camera.
Fine hand/foot landmarks and facial points retain their raw positions; the
correction stage currently targets the twelve shoulder-to-ankle body landmarks.

Each person's streaming JSON includes `tracking.dimension_prior_active`,
`tracking.keypoints`, derived `tracking.angles`, an `adjustments` list identifying
blended/inferred joints, and an optional `next_crop`. Original raw `keypoints`,
2D `angles`, and `pose3d` remain available. The correction stage does not feed
its synthesized positions into MotionAGFormer or dimension calibration; those
continue to use raw network observations. Image accuracy is not guaranteed by a
consistent skeleton, particularly during occlusion or a poor depth estimate.

Use `--no-dimension-tracking` to compare against raw 2D overlays and fixed cached
crops while retaining the 3D twin. `--no-3d` also disables dimension feedback.

## Setup

Requires Rust 1.88+ and **OpenCV 5** for the validated MotionAGFormer graph engine,
with DNN, videoio, highgui, imgproc, and
imgcodecs modules, plus Clang/libclang for generating Rust bindings.
The 2D-only pipeline (`--no-3d`) also supports OpenCV 4.8+.

macOS (Homebrew):

```sh
brew install opencv llvm pkg-config
# Only needed if libclang is not discovered automatically:
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
```

Ubuntu/Debian:

```sh
sudo apt-get update
sudo apt-get install libopencv-dev clang libclang-dev pkg-config
```

Then, from this directory:

```sh
cargo build --release
```

Model files are excluded from Git. Download the default pose model and detector
before the first run; the body-only model is optional:

| Model | File | Landmarks | Official archive |
| --- | --- | --- | --- |
| RTMW-m (default) | `models/rtmw-m.onnx` (~59 MiB) | 133: body, feet, face, hands | [Download](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip) |
| RTMPose-m | `models/rtmpose-m.onnx` (~52 MiB) | 17: body only | [Download](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip) |
| YOLOX-s (detector) | `models/yolox-s.onnx` (~34 MiB) | Person bounding boxes | [Download ONNX](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx) |
| MotionAGFormer-XS | `models/motionagformer-xs.onnx` (~9 MiB) | 17 inferred 3D body joints | [Official H36M checkpoint](https://drive.google.com/file/d/1Pab7cPvnWG8NOVd0nnL1iqAfYCUY4hDH/view) (export below) |

On another machine, download the required archive and extract its `end2end.onnx`
file to the corresponding model path above. For YOLOX-s, download the ONNX file
directly to `models/yolox-s.onnx`. The MotionAGFormer ONNX file is already exported
in this workspace. For a fresh clone, export its checkpoint as described below,
or pass `--no-3d` to run the 2D pipeline.
Models and generated outputs are excluded from Git; `Cargo.lock` is included for
reproducible Rust dependencies.
Use `--model /path/to/model.onnx` to load a compatible model. The supported export
has a float32 `[1,3,256,192]` input and two raw float32 SimCC outputs,
`[1,K,384]` and `[1,K,512]`, where K is 17 or 133. The layout is selected from the
output shapes. Other input resolutions and landmark layouts require code changes.
`--detector-model /path/to/yolox-s.onnx` selects the person detector; it must use
the documented 640×640 export with raw float32 `[1,8400,85]` output.

### One-time MotionAGFormer export

Python is needed only to convert the official PyTorch checkpoint. Inference in
the app remains Rust/OpenCV, with no ONNX Runtime dependency. Use Python 3.13 and
the verified export dependency versions:

```sh
python3.13 -m venv .venv-motion
.venv-motion/bin/pip install -r scripts/requirements-motion-export.txt
mkdir -p output models
git clone https://github.com/TaatiTeam/MotionAGFormer.git output/MotionAGFormer
git -C output/MotionAGFormer checkout 4756fd1eb7cc73f0e991f091ff2280e030ab85f3
curl -fL 'https://drive.google.com/uc?export=download&id=1Pab7cPvnWG8NOVd0nnL1iqAfYCUY4hDH' \
  -o models/motionagformer-xs.pth
.venv-motion/bin/python scripts/export_motionagformer.py \
  --source output/MotionAGFormer --checkpoint models/motionagformer-xs.pth
```

The exporter checks the ONNX graph and compares ONNX Runtime output with PyTorch.
It also writes binary validation fixtures beside the model. To check Rust/OpenCV
against the same PyTorch fixture, run `cargo run --example check_motion_export`.
The fixed float32 input/output shapes are `[1,27,17,3]`; the input channels are
normalized x, normalized y, and 2D confidence. Select a compatible export with
`--motion-model /path/to/motionagformer-xs.onnx`.
Upstream MotionAGFormer is [Apache-2.0 licensed](https://github.com/TaatiTeam/MotionAGFormer/blob/master/LICENSE).

If OpenCV discovery fails, ensure its pkg-config directory is on
`PKG_CONFIG_PATH`, for example on Homebrew:

```sh
export PKG_CONFIG_PATH="$(brew --prefix opencv)/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
```

The Rust crate's [build documentation](https://github.com/twistedfall/opencv-rust#opencv-rust)
covers manual library paths and platform setup. Older OpenCV DNN versions can
reject ONNX operators in some exports; use a newer OpenCV build if model loading
fails.

## Run

Webcam preview with automatic detection (press **q** or **Esc** to stop):

```sh
cargo run --release -- --camera 0
```

Adjust detection frequency, confidence, and duplicate suppression:

```sh
cargo run --release -- --camera 0 --detection-interval 15 --detection-threshold 0.5 \
  --nms-threshold 0.45
```

`--detection-interval 1` restores detection on every frame. Higher intervals reduce
average detector cost but delay crop updates and switching to a closer person.
Still images always run detection once. `--bbox` and `--no-detection` bypass the
schedule.

The detection threshold uses objectness × person-class confidence (default 0.4).
NMS uses bounding-box intersection-over-union (default 0.45). These settings are
separate from `--threshold`, which controls keypoint visibility.

Use the original body-only model:

```sh
cargo run --release -- --model models/rtmpose-m.onnx --camera 0
```

Image, without opening a window:

```sh
mkdir -p output
cargo run --release -- --input person.jpg --output output/pose.jpg \
  --json output/pose.jsonl --no-display
```

Video with a person crop and frame limit:

```sh
cargo run --release -- --input lifting.mp4 --bbox 100,40,500,700 \
  --output output/pose.mp4 --json output/pose.jsonl --max-frames 300 --no-display
```

Create destination directories before running. File outputs are overwritten.
Video output uses `mp4v` and the input's reported FPS, falling back to 30 FPS;
audio is not copied. Webcam capture may require camera permission for your terminal.

`--threshold 0.3` controls skeleton visibility. Each JSON Lines record includes a
zero-based frame index, frame dimensions, total inference time in milliseconds,
`detection_ran` (whether the detector ran on that frame), and a `people` array with zero or one entry. The selected person has a zero-based `person_index` valid only for
that frame, a `bbox` (`x`, `y`, `width`, `height`), a `detection_score`, and a
`keypoints` array with `name`, `x`, `y`, `score` per point. Each person also has an
`angles` array of 12 objects such as `{"joint":"left_knee","degrees":124.5}`;
unavailable angles have `degrees: null`. All keypoints are
exported, including low-scoring ones. Frames without detections emit `people: []`
and skip pose inference. On cached frames, `bbox` and `detection_score` come from
the last refresh; keypoints and angles are newly calculated. Manual/full-frame
crops have `detection_score: null` and `detection_ran: false`.
This replaces the previous top-level `keypoints` field with per-person results.
Coordinates
refer to the original frame and may lie outside its bounds; invalid nonpositive
predictions use `(-1,-1)`. Scores are raw SimCC values, not probabilities. Inference
time includes person detection, pose estimation for the selected person, preprocessing,
and decoding; it excludes drawing and file I/O.

```sh
cargo run -- --help
cargo test
```

## Implementation and sources

- `src/pose.rs`: padded aspect-preserving affine crop, RGB mean/std normalization,
  NCHW blob, OpenCV DNN CPU inference, SimCC argmax decoding, inverse crop mapping,
  and COCO skeleton rendering.
- `src/detection_cache.rs`: periodic detector scheduling and selected-crop caching.
- `src/angles.rs`: confidence-gated 2D joint angle calculations and 2D/3D degree labels.
- `src/motion3d.rs`: COCO-to-H36M conversion, temporal buffering, native
  MotionAGFormer inference, root-relative XYZ, 3D angles, and skeleton inset.
- `src/twin.rs`: shared dimension estimation, quaternion FK, robust rotation-only
  IK, approximate camera root translation, and final fixed-shape recording export.
- `src/tracking.rs`: learned-dimension landmark stabilization, bounded inferred
  joint display, and crop feedback between detector refreshes.
- `src/recording.rs`: unannotated capture/import and capture timing, before inference.
- `src/replay.rs`: model-free twin loading, FK reconstruction, timed interactive
  playback, orbit/zoom controls, and headless 3D video export.
- `scripts/export_motionagformer.py`: one-time checkpoint-to-ONNX conversion and
  numerical parity validation; not used by the running app.
- `src/detector.rs`: YOLOX-s RGB letterbox preprocessing, person-class decoding,
  coordinate rescaling/clipping, non-maximum suppression, and closest-person selection. Based on the
  [official OpenCV Zoo inference implementation](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox).
- `src/landmarks.rs`: COCO-WholeBody landmark names and body/foot/hand skeleton
  connections, using the [official MMPose dataset metadata](https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py).
- `src/main.rs`: CLI, capture/preview, image/video saving, JSON Lines output.

Models: [official RTMPose/RTMW model zoo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).
The default is RTMW-m cocktail14, 270 epochs, 256×192, `20231122`;
the original is RTMPose-m body7, 420 epochs, 256×192, `e48f03d0_20230504`.
Preprocessing follows each selected archive's `pipeline.json` (including
`to_rgb: true`, padding 1.25, and split ratio 2.0). Refer to
[MMPose's inference example](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/examples/onnxruntime/main.py)
for the affine/SimCC pipeline and to the upstream project for model licensing.
