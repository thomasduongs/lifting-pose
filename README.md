# lifting-pose

Rust CLI using **OpenCV** for capture, preprocessing, ONNX inference, and skeleton
rendering with **MediaPipe Pose** (default), using its native 3D body landmarks.
**RTMW-m / RTMPose-m + MotionAGFormer-XS** remain available as an alternative. Runs on CPU; no Python inference environment or
separate ONNX Runtime installation is required.

A skeletal twin fits the 3D observations using shared body dimensions and
per-frame rotations, preserving bone lengths through forward kinematics.


## MediaPipe Pose (default)

The default body model is OpenCV Zoo's [MediaPipe Pose ONNX export](https://github.com/opencv/opencv_zoo/tree/main/models/pose_estimation_mediapipe).
It takes 256×256 RGB input and estimates 33 image and hip-relative 3D landmarks.
YOLOX still selects the closest person every 15 frames; the pose model runs every
processed frame. Upper-body mode adds virtual crop space below the image for the
unseen legs and recenters the crop on measured shoulders. Unseen legs keep zero
confidence and are excluded from fitting/calibration.

The native 3D landmarks go directly into fixed-length twin IK. MediaPipe mode
bypasses MotionAGFormer, the experimental size/depth overrides described below,
and projected-landmark stabilization. This lets you compare the new body model
without those corrections overriding its predictions. Body units are inferred
meters, not camera-calibrated measurements. Live recording uses the new model too.
Hands use a dedicated MediaPipe palm detector, followed by an oriented crop and
21-point hand inference. Palm detection runs in a padded crop around the selected
person, including reliable wrists outside an older person box. A missing hand gets
a closer palm-detection pass around a reliable body wrist. Crop transforms keep
their centers in original image coordinates at every hand orientation. Palm
detection runs independently of the body's coarse finger guesses.
Previous hand bounds and body anchors provide recovery crops;
every accepted hand is inferred afresh. Left/right assignment uses body wrists
and recent hand positions. Strong hand wrists replace the corresponding body
wrist image positions, so finger skeletons connect to the arms. A body depth prior
that strongly disagrees with the hand's location is excluded from wrist fitting.
Direct hand-landmark replacement below 0.85 does not displace a body wrist with confidence at least
0.8 unless two fresh, consistent hand detections both have confidence at least 0.65.
This confirmation uses the current measured hand position and keeps its original
confidence; it does not average positions or promote a prediction to an observation.
Other weaker crops can still supply finger observations.
Unconfirmed coarse finger guesses are not drawn.
Tracking crops extrapolate recent wrist motion and palm scale, add a margin, and
retry with a wider crop after a miss. A high-confidence nearby hand only suppresses
a recovery attempt when its wrist matches the expected wrist. Palm NMS permits
more overlap, while matching landmark geometry removes duplicate detections even
when the crops disagree on handedness.
New hands still require 0.65 presence confidence. A recently reliable hand may
continue at 0.5 for up to 0.3 seconds, with a bound on wrist displacement.

Brief losses are shown as **gray fingers marked `pred`**, using damped translation
and scale of the last observed finger pose. They expire after 0.3 seconds; finger
articulation is held, not extrapolated. Predictions are display-only: they never
update body wrists, dimensions, IK observations, or recorded hand measurements.
Optional JSON exports put them in a separate `predicted_hands` field. Replay still
fills saved observation gaps independently from actual hand detections.

Face and detailed foot tracking/overlays are disabled for now. Sparse head anchors
remain internal to body fitting, and head position still follows the body model.
Sparse-face head orientation is off by default; `--head-orientation` opts back in.
The 3D viewer can still display the estimated lower-body scaffold.

MediaPipe upper-body inference uses a torso-centered virtual full-body crop twice
the visible detector-box height, retaining more arm detail than the old 3x crop.
If pose detection fails during a turn, it retries 2.5x and 3x crops. Its 64x64
landmark heatmap refines image coordinates with MediaPipe's same-frame 7x7 spatial
centroid method; this does not smooth motion over time or boost visibility scores.
Reference: [MediaPipe pose postprocessing](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt).

MediaPipe's image and world landmarks now use the same physical chest/head
definitions. Fitting combines a robust 3D prior, perspective image reprojection,
and continuity for unobserved joints only. Camera translation is refined against
image landmarks with the chosen focal length. Elbow/knee bends are limited to a
broad 165-degree flexion envelope; bend planes remain free. Upper-body root
orientation uses the shoulder line and camera vertical so head bobbing does not
rotate the entire torso.

Raw body/hand landmarks and head orientations are retained without speed clamping.
The bounded continuity and adaptive filtering stages are described below. Abrupt shoulder, elbow, or wrist detection errors can
be replaced by current-frame image-motion evidence when forward/backward optical flow agrees;
this gate does not keep dragging a joint onto an old shirt patch. The live overlay
uses the same measurement gate. An arm point replaced by image correspondence contributes
only its 2D position; its inconsistent neural depth is excluded from fitting and
dimension calibration. Missing points remain unobserved, and a failed
correspondence leaves the detector's observation intact. Wrist correspondence
starts at an independently detected hand: one strong detection or two consecutive,
consistent moderate detections. It can correct a weak but present body wrist,
without increasing its confidence. A confident body landmark on a shirt cannot
seed wrist tracking. Each confirmed hand supports at most two successive wrist
corrections, each requiring a new forward/backward image check and a detector
innovation. A failed check ends the chain immediately; image correspondence
cannot renew its own two-step allowance. Hand detection retains the original body-wrist crop and
also searches near a corrected wrist, so a crop shift does not discard the
original search location.
For file processing and the final recording pass, MediaPipe automatically uses
`models/rtmpose-m.onnx` as an additional arm image model when that file is installed.
Only sufficiently supported, in-frame elbow/wrist measurements are adopted;
validated wrist image flow and subsequent dedicated hand detections retain their
roles. Torso, head, and native body depth still come from MediaPipe. If an arm image
correction conflicts substantially with its native 3D location, that joint's stale
depth is excluded from fitting/calibration. The landmark log records adopted RTM
body observations in `arm_refined` (COCO indices, before dedicated hand merging
and final occlusion gates). This reduces pose jumps caused by an
elbow or wrist landing on the torso during blurred movement, without averaging
away real motion.

The extra model runs on each processed frame and increases processing time.
The live recording overlay uses that RTM model directly for 2D body joints,
plus the dedicated hand model. It does not run two body networks merely to display
2D landmarks. The saved 3D pass still uses MediaPipe with RTM arm refinement.
`--no-arm-refinement` keeps the selected primary model in both paths;
`--arm-model /path/to/rtmpose.onnx` chooses the image model explicitly. No additional model
is downloaded automatically. MediaPipe-only installations continue to work.

Joint continuity now adds a second, bounded check after dedicated hand merging.
It tracks a 3x3 set of image patches around each body joint, requires spatially
distributed forward/backward-consistent matches, and compares that motion with a
short velocity forecast before replacing a conflicting detector location. Explicit
arm-overlap masks block image-based reacquisition of the hidden arm. Strong fresh
hand detections can reacquire the wrist immediately. Corrected image measurements
cannot supply native depth or calibrate segment lengths from that stale depth.

If correspondence also fails, a decaying velocity forecast may draw a gray joint
for at most 160 ms. These predictions have zero measurement confidence and are
excluded from calibration and observed IK constraints. Tracking resets after a
long gap, non-increasing timestamps, resolution changes, or a different person crop.
The JSON Lines log retains `observed_keypoints` before this continuity stage,
`joint_tracking` provenance/age/inlier counts, and separate `predicted_joints`.
`observed_native_scores` preserves native confidence before the new depth rejection
gates; the native joint coordinates remain available in `pose3d`.

An adaptive time-based filter suppresses small stationary jitter while responding
more quickly to fast motion. It can move a body landmark by at most 2.5 pixels
from its accepted image measurement (1.5 pixels for finger landmarks). It does not
cap movement speed. File processing uses this causal continuity/filtering stage.
The live recording overlay uses current-frame image correspondence directly;
it does not repeat optical flow or adaptive averaging in the slower neural worker.

During recording, person detection has its own worker and refreshes every 15
camera frames by default; it cannot stall the body/hand worker. Known hands first
get fresh hand-model inference in their tracked regions. A wider palm search runs
when that fails; predictions are crop proposals, not measured hand landmarks.
Models warm up before capture starts. Two consistent moderate hand detections can
confirm wrist identity even when inference intervals exceed 100 ms.

The camera overlay follows image features on **each captured frame**, including
fingers. A short clean-image history aligns delayed neural results to the current
image before using them. Matching neural observations renew joint identity while
current image correspondence supplies the location; missing partial observations
do not erase a still-visible finger track. Contradictory neural jumps cannot
immediately displace verified tracks, and neither optical flow nor predictions
can renew their own neural support indefinitely. Live image tracks have a bounded
350 ms semantic lease with fresh forward/backward checks on every image. Pure
velocity forecasts retain the shorter 160 ms limit and zero confidence.
The image tracker works at up to 640 pixels wide and returns original-image
coordinates; neural inference retains its normal full-frame crop resolution.
All of this works with `--no-smoothing`, without a finished recording or replay.

Fresh body observations are published as soon as body inference finishes; they
are not held until hand reacquisition completes. Body and hand updates from the
same source frame are fused independently, without discarding partial updates.

On macOS, the live networks can use ONNX Runtime with CoreML. Warm-up compares
native and OpenCV inference on each model and retains the faster backend; a
runtime failure falls back to OpenCV. The model, crop preprocessing, and output
decoding remain unchanged. `--no-acceleration` forces the OpenCV path. No Python
runtime is required. The first Cargo build fetches the pinned native runtime;
CoreML's first model compilation may take several minutes before capture starts.
Compiled models are cached under `~/Library/Caches/lifting-pose/coreml` using model
content in the cache key, so changing weights cannot reuse the wrong compiled model.
The selected recording/3D model still uses its existing inference path.

OpenCV defaults to at most four CPU threads so detection, hand/body inference,
and camera image tracking can share the CPU. `--opencv-threads N` overrides it.
Live image-tracking errors trigger reacquisition while recording continues.

Final twin export additionally performs two five-frame sequence IK sweeps against
neighboring FK poses and each frame's own observations, with one fixed person
shape. A centered adaptive rotation/root filter follows: slow motion receives more
filtering, fast motion less. All edits in this recording-only pass together may increase any
strong visible joint's image-fit error by at most 2.5 pixels relative to its
input per-frame IK fit. Long gaps are not included in the temporal window.
This recording-only pass can use future observations without adding live latency.
The twin JSON describes its settings in `motion_processing`.

`--no-smoothing` disables the adaptive jitter filters while retaining rejection,
short-gap tracking and sequence IK. `--no-stabilization` disables all four new
stabilization stages for comparison. Previously saved twins must be reprocessed
from their source video to receive sequence fitting; replay alone does not alter
stored joint rotations.

Camera translation and joint rotations optimize the same reprojection objective;
image residual scale stays fixed during each
solve, and a later camera refit cannot invalidate the fitted joint projection.
High-confidence visible elbow/wrist image constraints get more weight than
uncertain monocular world predictions. Their weight changes continuously with
confidence, and additional coordinate sweeps let coupled arm/camera updates
converge; already converged fits still stop early.
Previous poses provide initialization and continuity for joints without
usable observations. A visible arm joint with rejected native depth retains
continuity in its unmeasured depth relative to the thorax; measured image motion
is not averaged. This depth prior is time based and follows torso turns rather
than holding the person fixed in space. Replay predicts missing frames. Fixed segment lengths,
joint limits, and confidence weighting remain active. Frames include
`reprojection_rmse_px` as an alignment diagnostic, not ground-truth pose accuracy.
It includes observed image landmarks even when their native 3D depth is rejected,
so a visible wrist cannot disappear from the error report merely because its
depth is uncertain.

Final saved upright upper-body recordings also estimate a continuous torso-heading
trajectory from the per-frame fits. A robust angular-acceleration prior reduces
isolated shoulder-depth/identity spikes, with weaker heading evidence in profile.
The original image observations are then fitted again with this torso heading;
observed arm coordinates are not averaged. This is a temporal constraint on the
derived torso rotation, so it can change estimated heading near a turn. It does
not establish the true orientation of a hidden torso or arm. The provisional
live fit does not have future frames and does not use this batch constraint.

New MediaPipe fits use the midpoint between the ears for the head-center anchor,
with the nose as a direction reference. The replay draws the skull at that center
and an inferred gray neck connection, rather than drawing the chest-to-nose fit
as an anatomical neck. Skull radius is one shared person parameter, calibrated
from near-frontal face/shoulder widths; profile samples cannot resize it. This is
an approximate head model, not a measured cervical spine or cranial surface.
Saves mark the anchor convention with `ear_head`; older saves retain their
original eye-based convention when replayed.

The head has an independent, zero-offset `head_pivot` control at the thorax.
The body and camera are fitted first; a separate head fit keeps the body fixed.
The ear midpoint supplies skull position; the nose supplies direction. The
current frame's body fit supplies a head-depth prior. The native nose-to-head
direction helps reject ambiguous depth flips without averaging observed head
motion. Brief unobserved head articulation carries the last estimate in body
coordinates.

New saves use schema 2 with 18 stored joints, including that virtual control.
Replay still accepts schema 1's 17-joint saves. Consumers should read
`joint_names` and `parents`; the new control shifts subsequent exported indices.
All rotations retain the standard outgoing-bone, parent-local FK convention.

Compatibility output retains the 133 landmark slots and the model's 17-joint
skeleton. Missing slots have zero confidence. MediaPipe supplies heel and foot-index
points; it does not distinguish big and small toes, so separate small-toe slots
are unobserved. Dense face tracking is absent. Old saves replay unchanged; record
again to use the new estimator.

To compare the previous pipeline:

```sh
cargo run --release -- --record --camera 0 --model models/rtmw-m.onnx
```

## Upper-body focus (default)

The torso spine is **assumed upright by default**. Pelvis-to-spine-to-chest bones
stay vertical relative to the replay floor; torso forward/back and side lean are
not estimated. Turning is allowed. Shoulder-girdle/neck and arm rotations remain
free, so the upright constraint does not force the observed head or hands into a
neutral pose. The assumption is recorded as `assume_upright_torso` in timing
metadata. Use `--free-torso` on a new tracking/recording run to allow lean again.
This is an imposed reference, not a measured gravity/camera calibration.
In upright upper-body mode, body yaw is assigned to the root from the fitted
shoulder direction; the chest joint cannot absorb the whole turn while the body
root stays facing forward. Reliable shoulders can sustain tracking when the face
and hands disappear in a back view. The motion prior follows measured torso yaw,
carrying unobserved limb poses in body-relative coordinates. At foreshortened
profile views, instantaneous depth is ambiguous. Final native upper-body saves
estimate a shared body-distance trajectory from the perspective fits, robust
shirt-feature scale changes, and a timestamp-based acceleration prior. This
prior applies to inferred body distance, not measured arm landmarks or joint
rotations. Depth can change through a side view; weak/missing depth evidence is
predicted using neighboring recording frames. `root_depth_observed: false`
identifies inferred depth. Clothing deformation, occlusion, and monocular scale
ambiguity still limit accuracy. Online previews remain provisional.
Camera initialization
is rebased when the model's unseen hip origin changes, keeping that origin change
from shifting the twin. Pixel landmarks below `--threshold` have zero fitting
weight, so weak occluded arm points use missing-joint continuity instead of
pulling the skeleton. These safeguards do not average measured poses over time.

For MediaPipe, a weak elbow/wrist landmark on a strongly observed opposite arm
is treated as occluded. Its 2D/3D fitting weights are zero until observations
return, and it does not calibrate segment lengths. Confident crossing arms and
independently visible wrists remain measurements. Unobserved arm articulation is
carried into both IK starting poses, including when the other arm moves. Confirmed
hand observations can still restore a wrist's 2D constraint; duplicate palm crops
cannot supply both hands even if their handedness predictions disagree. New hand
detections need a body wrist or recent same-side hand track to establish their
left/right identity; screen position alone is unreliable during turns. A
moderately confident hand must also agree with that arm's wrist or recent hand location.
Two consistent independent hand detections also provide a reference for up to
0.2 seconds: a moderately confident body wrist that jumps far from that reference
onto the strongly observed opposite arm is excluded. This reference only rejects
an observation; it never supplies a measured joint position. A fresh hand detection
can restore the wrist immediately, and confident crossing wrists remain observations.
Lower-confidence recovery is reserved for the same arm's recent hand track. These
checks do not smooth visible motion or recover ground-truth hidden joint positions.
Final recordings interpolate an unobserved arm chain between neighboring observed
rotations in body coordinates. A visible wrist keeps its fitted chain even if the
elbow is hidden. Predicted joint confidence remains zero.

Saved joint rotations include their observation `confidence`; replay dims joints
with little 3D observation support. These are model confidences, not calibrated
angle uncertainties. Missing hand/head orientations are interpolated relative to
the turning body and converted back to camera coordinates; measured orientations
remain unchanged. This keeps inferred detail from staying camera-facing through a spin.


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

In the RTM/MotionAGFormer alternative, for a camera facing your chest, upper-body fitting supplements weak elbow/wrist
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

The same relative-size principle also runs across every **observed skeletal
segment**, including head/neck, shoulders/torso, arms, pelvis, thighs, and shins
through the ankles. Each segment has its own reference size. The tracker measures
its image landmark span, compensates for the lifted segment's orientation, and
uses the change in scale to propose camera depth for its endpoints. Overlapping
estimates are confidence-weighted, smoothed, and combined with the 3D lifting
prior before fixed-length IK. Palm-size targets provide the more detailed wrist
constraint when hand detections are available. This works without the hand model.

These measurements are landmark spans, not body silhouette widths. Rotation
compensation still depends on inferred 3D orientation, so turning and moving in
depth can remain ambiguous. Hidden or unreliable segments contribute no size
cue; default upper-body mode does not measure unseen legs. Separate toe/heel
depth or body-surface thickness is not represented by the landmark skeleton.

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

The recording window shows a live **2D skeleton, hand landmarks, and joint
angles**. Preview inference runs on a separate worker without queued frames, so capture
never waits for the pose model and the worker starts from a fresh camera frame. The overlay shows its measurement age
and hides estimates older than 750 ms. Preview updates may be slower than camera
frames; every saved frame is still analyzed by the full 3D pipeline afterward.
Your closest-person selection, detection interval, crop, and upper-body settings
also apply to the preview. `--no-display` skips the live preview entirely.

Press **q** or **Esc** to stop capture and build the twin. The source footage is
saved before drawing overlays, so annotations never become model input. Video,
metadata, and raw twin observations are kept in temporary storage, bundled into `recording.twin`, and
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
New recordings also keep their raw landmark log inside that same save.

Replay defaults to a shaded baseline human body around the calibrated skeleton:
torso, pelvis, articulated limbs, head, palms, and fingers. The surface follows the
existing FK joints and fixed segment lengths. Body thickness uses stable baseline
ratios to shoulder/pelvis width; it is not a body scan or a measured muscle/fat model.
Missing hand detail uses a neutral hand, and unseen lower-body parts remain gray.
Press **B** to cycle body, skeleton, and body-with-skeleton views.

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
| B | Cycle body / skeleton / body with skeleton |
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
The body view includes a procedural surface and neutral facial features for
orientation. Skeleton view retains the outlined head, direction cue, and 21
landmarks per reliably detected hand. Feet use baseline shapes, without tracked toes.

Dedicated 3D hands use OpenCV Zoo's MediaPipe HandPose model, with MediaPipe Pose
palm detections providing aligned crops and previous hand landmarks providing recovery crops. Each subject stores shared hand bone lengths;
each frame stores incoming-bone quaternions in camera coordinates. Replay attaches
the palm and fingers to the fitted body wrist. Learned hand depth is registered
to body model units using image scale, so these are not metric measurements.
Low-confidence/unseen hand observations are omitted from the raw twin data;
replay interpolates hand gaps independently when usable hand observations exist. `--no-hands` disables the additional
model while retaining RTMW's existing 2D hand landmarks. Very small or poorly
observed hands may not yield a 3D result.

Head position follows the 3D body. With `--head-orientation`, approximate orientation is fitted from reliable
nose, eye, and ear landmarks against a generic sparse face. When this fit is
unavailable, replay uses the head's body-joint orientation. This is not a facial
surface reconstruction or a measured neck rotation. Old recordings still load
but need reprocessing for dedicated hand and face-orientation data.

Legacy folder recordings can still be processed directly using `--input`,
`--output`, and `--twin-output`. A legacy `source.mp4` next to `capture.json`
reuses its acquisition times. For new recordings, use the bundled workflow above
to avoid creating separate files.

Replay opens and exports in a 1024×768 straight-on view showing recorded root
movement through a fixed scene, with zero camera yaw and tilt. Reset (R) restores this front view and root movement; F
toggles the optional in-place view. Translation does not change the saved joint
angles, which are calculated from relative bone directions. Irregular capture
timestamps are resampled to constant FPS with time-based root/quaternion
interpolation; original observed samples and fixed bone lengths are preserved.
Remove `--no-display` to export
first and then open interactive replay. `--max-frames` limits the source-frame
span for preview/export. The source video and twin JSON are protected against
being selected as the replay output.

Supports images, video files, and webcams, annotated image/video output, and
JSON Lines keypoint export. The optional RTMW model uses 192×256 input (width × height)
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
frame and every **15 frames** thereafter by default. In the RTM alternative, between refreshes the learned
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
shows a straight-on view of the fitted skeletal twin. The existing JSON
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

Calibration starts from the first valid 3D observation. MediaPipe refinement uses
reliable filtered observations immediately; MotionAGFormer refinement begins once
it has 27 real history frames. Both endpoints must meet confidence
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

**Inverse kinematics** fits rotations to confidence-weighted observations with a
robust loss, damped least-squares rotational updates, and bounded steps. MediaPipe
adds perspective reprojection to the 3D prior, with motion continuity only for
unobserved joints.
The previous solution initializes the solve when helpful; dimensions never change
inside a pose solve. `fit_rmse` reports the weighted observation residual in model
units. Missing/low-confidence observations have zero weight. Joint-center fitting
does not identify axial twist uniquely. MediaPipe applies broad elbow/knee bend
limits; a calibrated anatomical hinge model, meshes, skin, and contact constraints
are not implemented.

The twin approximates root translation from fitted 3D joints and their image
observations. MediaPipe refines a perspective translation fit; the MotionAGFormer
alternative uses a weak-perspective approximation. It assumes square pixels and the image center as
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
| MediaPipe Pose (default) | `models/mediapipe-pose.onnx` (~5 MiB) | 33 native 2D/3D body points | [Download ONNX](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx) |
| RTMW-m | `models/rtmw-m.onnx` (~59 MiB) | 133: body, feet, face, hands | [Download](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip) |
| RTMPose-m | `models/rtmpose-m.onnx` (~52 MiB) | 17: body only | [Download](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip) |
| YOLOX-s (detector) | `models/yolox-s.onnx` (~34 MiB) | Person bounding boxes | [Download ONNX](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx) |
| MotionAGFormer-XS | `models/motionagformer-xs.onnx` (~9 MiB) | 17 inferred 3D body joints | [Official H36M checkpoint](https://drive.google.com/file/d/1Pab7cPvnWG8NOVd0nnL1iqAfYCUY4hDH/view) (export below) |
| MediaPipe Palm | `models/palmdet.onnx` (~4 MiB) | Palm boxes and 7 crop anchors | [Download ONNX](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx) |
| MediaPipe HandPose | `models/handpose.onnx` (~4 MiB) | 21 learned 3D points per hand | [Download ONNX](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx) |

For the default pipeline, download the MediaPipe Pose ONNX file directly to
`models/mediapipe-pose.onnx`. MotionAGFormer is not required for MediaPipe Pose.
For RTM alternatives, download the archive and extract its `end2end.onnx`
file to the corresponding model path above. For YOLOX-s, download the ONNX file
directly to `models/yolox-s.onnx`; download the hand ONNX directly to
`models/handpose.onnx` and the palm ONNX to `models/palmdet.onnx` (or use `--no-hands`).
For a custom hand-model directory, keep `palmdet.onnx` beside the hand model. The MotionAGFormer ONNX file is already exported
in this workspace. For a fresh clone, export its checkpoint as described below,
or pass `--no-3d` to run the 2D pipeline.
Models and generated outputs are excluded from Git; `Cargo.lock` is included for
reproducible Rust dependencies.
Use `--model /path/to/model.onnx` to load a compatible model. MediaPipe uses the documented NHWC `[1,256,256,3]` export. RTM exports
have a float32 `[1,3,256,192]` input and two raw float32 SimCC outputs,
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

### VideoPose3D evaluation

VideoPose3D currently has a local evaluation harness; recordings continue to use
the current MediaPipe fit. `scripts/export_videopose3d.py` converts the
[official COCO checkpoint](https://github.com/facebookresearch/VideoPose3D/blob/main/INFERENCE.md)
using an upstream source checkout. The one-time export environment needs PyTorch,
NumPy, ONNX, and ONNX Runtime. The Rust evaluation runs inference with OpenCV.
Upstream code and weights retain their
[CC BY-NC license](https://github.com/facebookresearch/VideoPose3D/blob/main/LICENSE).

```sh
VP_EVAL=/tmp/lifting-pose-vp-eval
python scripts/export_videopose3d.py \
  --source /path/to/VideoPose3D \
  --checkpoint /path/to/pretrained_h36m_detectron_coco.bin \
  --output "$VP_EVAL/videopose3d/videopose3d.onnx"
LIFTING_EVAL_DIR="$VP_EVAL" cargo test --release \
  videopose3d_matches_export_reference -- --ignored
```

The exporter verifies reference/optimized PyTorch and ONNX predictions and saves
reference tensors beside the model. The Rust check compares the same tensors.
Its fixed input is `[1,243,17,2]` normalized COCO image coordinates; output is `[1,1,17,3]`
inferred Human3.6M joints. Recording comparisons use acquisition timestamps and
centered model windows. Missing joints and neutral lower-body context stay
unobserved in the fit; a model producing all joints does not verify hidden poses.

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
- `src/avatar.rs`: baseline human surface, articulated hands, shading, and depth-buffer rendering.
- `src/joint_tracking.rs`: multi-patch joint correspondence, short predictions, and adaptive image filtering.
- `src/replay.rs`: model-free twin loading, FK reconstruction, timed interactive
  playback, orbit/zoom controls, and headless 3D video export.
- `scripts/export_motionagformer.py`: one-time checkpoint-to-ONNX conversion and
  numerical parity validation; not used by the running app.
- `src/detector.rs`: YOLOX-s RGB letterbox preprocessing, person-class decoding,
  coordinate rescaling/clipping, non-maximum suppression, and closest-person selection. Based on the
  [official OpenCV Zoo inference implementation](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox).
- `src/landmarks.rs`: COCO-WholeBody landmark names and body/foot/hand skeleton
  connections, using the [official MMPose dataset metadata](https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/coco_wholebody.py).
- `src/mediapipe.rs`: MediaPipe Pose preprocessing, sparse-landmark mapping, native 3D observations.
- `src/live_preview.rs`: independent live detection and pose workers with fresh hand-region inference.
- `src/preview_tracking.rs`: current-image joint tracking and delayed neural-result alignment.
- `src/accelerated.rs`: native macOS execution, model warm-up selection, and OpenCV fallback.
- `src/main.rs`: CLI, capture/preview, image/video saving, JSON Lines output.

Alternative models: [official RTMPose/RTMW model zoo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).
The RTMW alternative is cocktail14, 270 epochs, 256×192, `20231122`;
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
