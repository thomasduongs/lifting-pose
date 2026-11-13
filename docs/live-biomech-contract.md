# Live split-view anatomical twin

**Muscle analysis extension:** `docs/muscle-estimation-contract.md` defines the new
skeleton-driven inverse dynamics and estimated muscle activation panel. Tracking still
prescribes the skeleton; source muscle activity is never claimed measured.

**Current integration:** live and recording now share `src/tracking_pipeline.rs`, including
RTMPose, dedicated hands, occlusion, stabilization and the original skeletal fit. The API
sketches below describe the initial implementation; see `docs/live-biomech.md` and the
current source for expanded capture configuration and provenance fields. The anatomical
retargeter consumes the skeletal fit, with pixel-only depths explicitly labelled inferred.

Deliver a native Rust window with the model on the LEFT HALF and camera on the
RIGHT HALF. No record/export/replay step and no Python/MuJoCo runtime. Use the
existing OpenCV MediaPipe/YOLOX pipeline and biomech-lite FK/constraints, plus the
existing biomech-viewer Renderer/Camera/VisualModel. Retain old workflows. No frame
recording, file writes of camera images or external transmission by default.

This is camera-driven constrained kinematics. Do not label inferred pose as measured
muscle activation, forces or autonomous forward dynamics. UI: "LIVE ANATOMICAL TWIN",
"Model - live pose fit" left, "Camera - live" right; held lower body/reference anatomy.

## Integration strategy / ownership

Add `--live-biomech` to the EXISTING root `lifting-pose` binary and early-dispatch
to `biomech_live::run`. That binary already has OpenCV, pose, detector and all their
private module dependencies. Reuse `crate::pose::PoseEstimator` and
`crate::detector::PersonDetector`; do not duplicate MediaPipe decoding or force a
new public root library. Root Cargo adds path deps biomech-lite and biomech-viewer,
plus minifb0.29. The small viewer and headless simulator crates keep no OpenCV dep.

Root modules: `src/biomech_live_capture.rs` (capture worker), `src/biomech_live.rs`
(native split UI). Main defines both modules and exposes a `--live-biomech` bool;
the existing --camera/--input/--model/--detector-model flags are reused. Root CLI
for live additionally accepts --biomech-model PATH, --biomech-visual PATH, and
--live-snapshot PATH (headless one processed frame; with --input video/image),
--max-frames N (smoke exit). Do not run normal recording/finalization in live mode.
Main passes explicit config fields, no dependency on private Args type in modules.

## Fitting library API (F1 owner, Opus)

Create `crates/biomech-lite/src/live_fit.rs`, declare pub mod live_fit in lib.rs.
Own its meaningful tests `tests/live_fit.rs`. No OpenCV dependency here.

```
#[derive(Clone,Debug)]
pub struct Landmark { pub name:String,pub position:[f64;3],pub confidence:f64 }
pub struct Observation { pub timestamp_seconds:f64,pub landmarks:Vec<Landmark> }
pub struct FitResult { pub qpos:Vec<f64>,pub rmse_m:Option<f64>,
 pub supported_arms:[bool;2], // LEFT then RIGHT
 pub status:String,pub iterations:usize }
pub struct LiveFitter
LiveFitter::new(model:std::sync::Arc<Model>)->Result<Self,String>
LiveFitter::update(&mut self,obs:&Observation)->Result<FitResult,String>
LiveFitter::reset(&mut self)
```

Inputs: native MediaPipe camera coordinates x-right/y-down/z-forward, hip relative
estimated metres. Landmarks named left/right_shoulder,elbow,wrist (at least these).
Only source eight independent arm coords (model.dynamic_joints) fitted. Root,
torso, wrists/fingers/lower body remain standing priors. All model polynomial
couplings exactly projected every residual; all scalar bounds respected. Missing
or low-confidence observations HOLD prior pose, do not invent observations.

Use native marker definitions from scripts/biomech_model.py:
shoulder=joint anchor shoulder_elv_l/r; elbow=mean Elbow_Lateral_marker_l/r and
Elbow_Medial_marker_l/r sites; wrist=joint anchor flexion_l/r. Native camera basis:
lateral=normalize(leftShoulder-rightShoulder), up=orthogonalized [0,-1,0],
forward=lateral.cross(up). Native model basis from standing GH lateral and [0,0,1].
Map camera directions by Bmodel*Bcamera.T. Require confident nondegenerate shoulder
line for basis; otherwise hold and report. This removes camera-axis reflection.

Retarget upper/lower limb directions to reference segment lengths: target elbow
offset = mapped normalized(elbow-shoulder)*native upper length; wrist offset adds
mapped normalized(wrist-elbow)*native forearm length. Predicted offsets subtract
CURRENT model GH anchor, not standing global shoulder. This is generic anatomy,
not personal geometry calibration. Score floor0.3, clip confidence[0,1], reject
nonfinite/degenerate segments. An unsupported side must not update its master DOFs.

Implement bounded warm-start damped GN/LM with central FD Jacobian of FK residual,
robust residual (Huber~0.04m), modest temporal/neutral priors for unidentifiable twist.
At most ~12 iterations/update; backtrack invalid master/dependent bounds and poor
steps; reject invalid/nonfinite input safely. Rate/jitter protection must be explicit
and not convert held poses to measured observations. No forcing missing limbs to zero.
Whole torso/yaw fitting is outside initial live subset. Tests: synthetic marker poses
mapped through camera basis recover supported limb motion without left/right reflection;
unsupported/NaN side holds; bounds and coupling exact over sequence, fixed priors stay fixed.

## Capture/inference worker API (C1 owner, Sonnet)

`src/biomech_live_capture.rs`:

```
pub struct Config { pub camera:i32,pub input:Option<PathBuf>,pub pose_model:PathBuf,
 pub detector_model:PathBuf,pub no_detection:bool,pub opencv_threads:i32 }
pub struct VideoFrame { pub index:u64,pub captured:Instant,pub width:usize,
 pub height:usize,pub rgb:Vec<u32> } // unmirrored 0x00RRGGBB
pub struct PoseFrame { pub index:u64,pub captured:Instant,pub observation:Observation,
 pub points:Vec<(String,f32,f32,f32)>,pub inference_ms:f64 }
pub struct CaptureState { pub video:Option<Arc<VideoFrame>>,pub pose:Option<Arc<PoseFrame>>,
 pub status:String,pub error:Option<String>,pub ended:bool }
pub struct LiveCapture
LiveCapture::start(config:Config)->Result<Self,String>
LiveCapture::latest(&self)->CaptureState // snapshots newest pointers, no blocking work
```

Bounded latest-value mailboxes with separate capture and inference threads. Camera
preview must keep refreshing while inference runs; inference consumes freshest frame,
never an unbounded frame queue. `PoseEstimator::new`, set_upper_body(true), infer,
take_pose3d. For YOLOX use existing detector without acceleration initially, detect
every~15 processed frames then reuse crop; choose closest/large person. No-detection
mode may infer full-image rectangle, but no person/low scores must be clearly reported.
Set OpenCV thread budget2-4 to leave CPU for renderer. No hands/refiner initial scope.
MediaPipe pose3d joints have name/x/y/z/input_score; convert directly to Landmark.
No fabricated zero 3D for failed inference. 2D points may be used for camera overlay.

Camera default0, request around640x480/30fps; VideoCapture CAP_ANY, read actual dims.
`--input` supports video/image for deterministic testing, clearly labeled as file
input and never presented as an actual webcam test. File reader paced to source FPS;
at EOF retain last frame with ended=true. UI remains responsive on open/read/model
errors and displays useful permission/device/model messages. Stop/drop signals threads,
release capture and join cleanly; avoid staleFrame queues and camera leaked on close.
Workers MUST NOT open actual webcam or GUI; lead owns hardware/permission verification.

## Split UI (U1 owner, Sonnet, independent new module)

```
pub struct Config { pub capture:CaptureConfig,pub model:PathBuf,pub visual:PathBuf,
 pub max_frames:Option<u64>,pub snapshot:Option<PathBuf> }
pub fn run(config:Config)->anyhow::Result<()>
```

Native minifb main-thread window, default1280x800 or1440x900. Header60 and footer40
shared, content rectangle split EXACTLY50/50 left model, right video. No persistent
sidebar consuming a half. Camera image letterboxed to maintain aspect; RGB conversion
and resize predictable, no hidden mirror. Compact overlaid/button toolbar for camera
presets, fit/reset and bones/muscle/ligament layers. Space pauses pose-following
(camera remainslive); R resetsfit; 1/2/3views, F frame, B/M/Glayers, P savesPNG to
Pictures/Biomech Viewer, Esc exits. Mouse orbit/zoom confined toleftviewport. Show
camera FPS/pose age/support/inference+fit latency or equivalent meaningful status.
Stale pose>0.5s shows HELD/LOST conspicuously; do not freeze webcam with inference.
Use safe default standing full-body framing plus arms margin; never camera-fit each
frame (causes motion/zoom). Fit on F/reset only; user may choose upper body.

Process newest distinct PoseFrame with LiveFitter once, keep UI responsive; if fitting
proves too slow, move it to a third bounded worker. Use Rust model.kinematics and
existing Renderer for anatomical pixels. Do not copy an unrelated old skeletal twin.
Camera view raw/live plus optional native2D arm overlay; any overlay must use its
matching pose/frame or explicitly label estimated-overlay age. No activation/force
inference. No simulated prerecorded values in camera mode.

--live-snapshot requires file input (no hidden webcam opening), waits for usable
processed frame or givesbounded timeout with useful error, renders same split UI
headlessly without minifbwindow and exits; --max-framessmoke respects finiteupdates.
SavePNG createsparents/no numberedoverwrite; errors visible, no panic. Standalone
GUIclose stopsworkers/releasecamera. Tests:50/50layout,aspectletterbox,stale status,
syntheticposeupdatesRustmodel andnonblankmixedcamera/model pixels; do not fake hardware.

## CLI/launch/package (A1+L1 owner, Haiku)

A1 owns root Cargo.toml, main.rs plumbing (newmod declarations,Argsflags,earlydispatch
immediatelyafterArgsparse and before any normaloutputs). Addpathdeps viewer/core and
minifb without changing olddefaults. New--live-biomech conflicts record/replay and
unrelatedoutput/json options or clearly bypasses them. Reuse--no-detection and
opencvthreads flags. Camera/input CLIconflictsalreadyexist.

L1 afterU1: executable `scripts/live_biomech.sh` builds root binary release with
--bin lifting-pose then exec --live-biomech --camera0 bydefault (do NOT forcecamera0
if--inputgiven). Forwardflags exactly. Make `scripts/view_biomech.sh --live` route
to this launcher; oldviewerdefaultkept. Document livecontrols,localonly/noautosave,
camera/privacypermission andfileinputtesting clearly. Provide macOS
`scripts/package_live_biomech.sh` or extendexistingpackage with--live to produce
`output/Live Biomech.app`, idlocal.liftingpose.live-biomech,
NSCameraUsageDescription describing local tracking and no recording. Bundle executable
plus model/visual data+ONNX pose/detector weights+notices. Launch bundle withliveargs
withoutrecording: shellentry resolvesresources/cd then exec realrootbinary--live-biomech;
nativebinaryCFBundleExecutablewrapper identity shouldpreserve maccamerausage metadata.
For macOS package, prefer a native CFBundleExecutable `live-biomech` (copy of the
root binary) instead of a shell entry point: main automatically enables live mode
when current_exe filename is `live-biomech`. A1 implements this opt-in alias only.
Resolve live assets: explicit existing paths, app Contents/Resources/{relativepath},
then project-manifest-root/{relativepath}; preserve explicit missing-path error.
This keeps camera permission identity in the native app bundle. Inspect `otool -L`
and copy required local ONNX Runtime dylibs if linked; system Homebrew OpenCV is
already installed, so do not claim a portable OpenCV bundle unless actually bundled.
No actualcamera/GUIlaunchbyworkers. Root owns finalREADME/results andhardwaretest.

## Acceptance

Release livewindow opens with real webcam onRIGHT andmodelLEFT. Supportedcamera
shoulder/elbow motion updates sourcedanatomy without saving/replayingfiles. Show
no-person/lost/stale/error states honestly. Targetinteractivepreview20-30fps even
ifposefitslower;reportmeasuredrates,notassumedrealtime. Validatewith syntheticFK
observations andsourcevideo first;then leadactualwebcam test plus nativewindowvisual
inspection. User requestedlivecamera, so cameraaccesspromptisexplicitlyauthorized.
Do not enable microphone or change unrelatedprivacysettings. Ifhardwareblocked,
shipreviewablefeature andreportexactblockerwithoutclaimingwebcamverified.
