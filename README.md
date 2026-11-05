# lifting-pose

Rust CLI using **OpenCV** for capture, preprocessing, ONNX inference, and skeleton
rendering with **RTMW-m WholeBody** (default) or **RTMPose-m**, followed by
**MotionAGFormer-XS** temporal 3D body lifting. Runs on CPU; no Python inference environment or
separate ONNX Runtime installation is required.

A skeletal twin fits the 3D observations using shared body dimensions and
per-frame rotations, preserving bone lengths through forward kinematics.

## Upper-body focus (default)

Tracking now focuses on the visible **head, neck, shoulders, arms, wrists, and
hands**. Both shoulders must be reliable, but hips and legs are not required.
The 3D inset and replay still display the full skeleton. Unobserved lower-body
joints appear in gray as an estimated scaffold; missing leg data is fine and
never blocks upper-body tracking. The 2D overlay only draws available measurements.
Older recordings also use upper-body tracking scope by default.
Replay places a shaded perspective floor grid beneath the skeleton as a spatial
reference. Its height follows the recording's lowest skeleton point; it is not a
measured floor or a foot-contact estimate.

The 3D lifting network still has a 17-joint interface. Neutral lower-body context
is supplied internally with **zero confidence**; those slots are not treated as
observations, used to calibrate lower-body dimensions, or used to move the crop.
Upper-body IK is anchored at the chest and uses the shoulder/head reference frame,
so changing hidden leg/spine scaffold lengths cannot move the fitted visible joints.
Upper-body saves identify their active joints and omit unobserved pelvis/leg
measurements from their named person parameters. Compatibility rest-offset slots
remain in the file, but are not leg measurements.

Angles shown in this mode are shoulder, elbow, and wrist angles. The 2D shoulder
reference is the opposite shoulder; the 3D reference is the chest. Hip, knee, and
ankle angles are omitted. Head and 3D hand tracking remain enabled. Add
`--full-body` when recording/tracking to restore the full-body pipeline later.
An upper-body-only save cannot turn its hidden slots into measured legs merely
by passing `--full-body` to replay.

For a camera facing your chest, upper-body fitting supplements weak elbow/wrist
depth predictions using fixed arm lengths and perspective camera rays. It uses
the detected foreshortening only for reliable joints and a roughly frontal torso.
The learned depth direction selects the bend; previous rotations resolve weak
predictions, with a reach toward the camera as the initial ambiguous default.
This is a depth estimate, not a depth measurement. New recordings use this fit;
existing saved rotations are unchanged. A calibrated focal length improves the
camera model; the default focal length is approximate.

With hand tracking enabled, changes in **apparent palm size** also constrain wrist
depth: a larger palm implies a smaller camera distance, and vice versa. Each hand
gets its own reference from its first reliable detection. The fit compares several
palm segments, compensates for their estimated 3D orientation, and rejects weak
or inconsistent measurements. It smooths scale changes and feeds the resulting
wrist target through fixed-length shoulder/elbow IK, clamping targets to the arm's
reach. The same calculation runs during live fitting and the final recording fit.
Missing hand detections fall back to body-based fitting.

Start with both palms clearly visible in a comfortable pose to establish the
references. Initial absolute depth still comes from the body/camera estimate;
relative hand-size changes improve reaching motion but do not provide metric depth.
Palm turns, occlusion, and hand-model errors can still affect the estimate. This
requires a new recording; replaying old saved rotations does not refit them.

Using body axes **X forward/back, Y side-to-side, Z up/down**, a frontal camera
maps forward toward the camera. The saved format retains camera coordinates for
compatibility: body `(X, Y, Z) = (-camera_z, camera_x, -camera_y)` for aligned
axes. Relabeling axes does not add depth information.

## Record and replay a 3D twin

There is one reusable recording save and one reusable replay video:

- **`output/recording.twin`** contains the current videos, timing, and skeletal twin.
- **`output/replay.mp4`** contains the latest 3D replay.

Every new recording replaces the previous save after processing succeeds. Every
replay overwrites the previous replay video. No numbered session folders or
permanent intermediate files are created.

Record:

```sh
cargo run --release -- --record --camera 0
```

Press **q** or **Esc** to stop capture and build the twin. Capture runs without
neural inference; analysis takes place afterward. Video, metadata, and raw twin
observations are kept in temporary storage, bundled into `recording.twin`, and
cleaned up. The old save is replaced atomically only after the new one is complete;
a failed or too-short recording leaves it intact.

The `.twin` file is a portable ZIP container. It includes the full source backup,
the finished video trimmed by three seconds at each end, annotated preview,
capture timing, and final person/frame parameters. These are archive entries,
not separate permanent files. Optional JSON Lines data is also kept inside the save.
You can extract the archive with a ZIP tool if you need the original footage.

Add `--record-seconds 10` for a timed capture. Headless capture (`--no-display`)
requires a duration or frame limit. `--trim-start 0 --trim-end 0` disables trimming;
clips of six seconds or less need a smaller trim. Recording requires the local
pose models. Import an existing video into the same reusable save with:

```sh
cargo run --release -- --record --input lifting.mp4 --no-display
```

Replay the current save:

```sh
cargo run --release -- --replay
```

This overwrites `output/replay.mp4` and opens the interactive viewer. For export
only, use `--replay --no-display`. Replay extracts only the twin parameters into
a temporary directory; it needs no models, Python, camera, or extracted videos.
Custom `.twin` paths are supported after `--record` or `--replay`; legacy standalone
`twin.json` inputs also remain supported. `--output other.mp4` overrides the replay
video destination. Record mode keeps all generated assets inside the save.

Recorded times control replay. Missing body, head, and hand poses are completed
from surrounding observations; rotations are interpolated while FK keeps shared
bone lengths fixed. Bounded Hermite root motion avoids overshoot. With only one
usable endpoint, the nearest pose is held. Gray parts and the "Predicted motion"
label identify predictions. Long gaps remain uncertain, and synthesized frames
are not fed back into dimension calibration.

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

By default replay covers the capture timeline **minus three seconds at each end**,
including all tracking segments and completed gaps in the retained portion. It
uses one shared reference body shape (from the segment with most valid frames)
and the best-supported hand shapes across the selected recording so segment
boundaries cannot resize the skeleton. This assumes one continuing person;
select a specific subject if the recording contains a real identity change. Use `--replay-subject 1` to explicitly filter and trim to one segment.
Older files without timing default to 30 FPS; `--replay-fps 15` overrides timing.
The 3D view includes the 17-joint body, an outlined head with a forward-direction
cue, and 21 landmarks per reliably detected hand. It has no skin mesh or 3D toes.

Dedicated 3D hands use OpenCV Zoo's MediaPipe HandPose model, with RTMW hand
landmarks providing aligned crops. Each subject stores shared hand bone lengths;
each frame stores incoming-bone quaternions in camera coordinates. Replay attaches
the palm and fingers to the fitted body wrist. Learned hand depth is registered
to body model units using image scale, so these are not metric measurements.
Low-confidence/unseen hand observations are omitted from the raw twin data;
replay interpolates hand gaps independently when usable hand observations exist. `--no-hands` disables the additional
model while retaining RTMW's existing 2D hand landmarks. Very small or poorly
observed hands may not yield a 3D result.

Head position follows the 3D body. Approximate orientation is fitted from reliable
nose, eye, and ear landmarks against a generic sparse face. When this fit is
unavailable, replay uses the head's body-joint orientation. This is not a facial
surface reconstruction or a measured neck rotation. Old recordings still load
but need reprocessing for dedicated hand and face-orientation data.

Legacy folder recordings can still be processed directly using `--input`,
`--output`, and `--twin-output`. A legacy `source.mp4` next to `capture.json`
reuses its acquisition times. For new recordings, use the bundled workflow above
to avoid creating separate files.

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
a confidence produced by the 3D model. The body model predicts body structure;
a separate model supplies 3D hands. Neither supplies 3D toes or measured spine curvature. Raw `keypoints` and `pose3d` are
preserved; dimension-guided image-space results are exported separately under
`tracking`. The model does not supply persistent person identities.

One uncertain shoulder/hip is permitted when the other three torso points are
reliable. Its confidence stays low and it is excluded from confident dimension
measurements. History resets when no person is selected, fewer than three torso
points are reliable, frame
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
the `twin.json` entry inside `output/recording.twin` in record mode. Direct
tracking without `--record` remains available for debugging and writes
`output/twin.json`; `--twin-output path.json` overrides that debug destination.
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
| MediaPipe HandPose | `models/handpose.onnx` (~4 MiB) | 21 learned 3D points per hand | [Download ONNX](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx) |

On another machine, download the required archive and extract its `end2end.onnx`
file to the corresponding model path above. For YOLOX-s, download the ONNX file
directly to `models/yolox-s.onnx`; download the hand ONNX directly to
`models/handpose.onnx` (or use `--no-hands`). The MotionAGFormer ONNX file is already exported
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

The additional head/hand implementation is in `src/details.rs`; the hand model
interface follows the [official OpenCV Zoo example](https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe).

`src/bundle.rs` packages a recording as one atomically replaced `.twin` save and
extracts only temporary twin data during replay.

`src/scope.rs` defines active upper-body landmarks and suppresses lower-body observations.
