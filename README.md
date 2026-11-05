# lifting-pose

Rust CLI using **OpenCV** for capture, preprocessing, ONNX inference, and skeleton
rendering with **RTMW-m WholeBody** (default) or **RTMPose-m**. Runs on CPU; no Python inference environment or
separate ONNX Runtime installation is required.

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
frame and every **15 frames** thereafter by default. Between refreshes, the last
selected bounding box is reused while pose estimation and angles update every
frame. The selected person can change at a refresh when someone else has a larger
box. Empty detections are also cached until the next refresh; resolution changes
force immediate detection. A moving person can leave the cached crop between
refreshes, so use a smaller interval for fast motion. Persistent identity tracking and lifting form analysis are not implemented.

Joint angles are displayed automatically beside the selected person's joints.
Yellow labels show left/right shoulder, elbow, wrist, hip, knee, and ankle angles
in degrees, when their landmarks are reliable. These are unsigned **2D interior
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

## Setup

Requires Rust 1.88+, OpenCV 4.8+ or 5.x with DNN, videoio, highgui, imgproc, and
imgcodecs modules, plus Clang/libclang for generating Rust bindings.

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

On another machine, download the required archive and extract its `end2end.onnx`
file to the corresponding model path above. For YOLOX-s, download the ONNX file
directly to `models/yolox-s.onnx`. No download script is required.
Models and generated outputs are excluded from Git; `Cargo.lock` is included for
reproducible Rust dependencies.
Use `--model /path/to/model.onnx` to load a compatible model. The supported export
has a float32 `[1,3,256,192]` input and two raw float32 SimCC outputs,
`[1,K,384]` and `[1,K,512]`, where K is 17 or 133. The layout is selected from the
output shapes. Other input resolutions and landmark layouts require code changes.
`--detector-model /path/to/yolox-s.onnx` selects the person detector; it must use
the documented 640×640 export with raw float32 `[1,8400,85]` output.

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
- `src/angles.rs`: confidence-gated 2D joint angle calculations and degree labels.
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
