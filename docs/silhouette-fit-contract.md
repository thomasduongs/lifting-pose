# Video surface-evidence extension

Research direction: Kadlecek et al. 2016 inverse body modeling uses surface
observations, not just skeletal targets. Here add a deliberately conservative
silhouette-width adaptation; do not claim a full paper reproduction.

Implement `scripts/twin_silhouette.py` and integrate into `reconstruct_twin.py`
behind `--fit-silhouette` with `--segmentation-model models/mediapipe-pose.onnx`.
Optional cv2 dependency; core FEM stays NumPy/SciPy only. On this machine Python
cv2 4.10 and the model are already installed. Emit JSON-serializable shape evidence
in subject manifest, including accepted/rejected counts, per-bone prior/fitted
radii and provenance. No silent fallback on a missing requested model/video.

Use the saved bundle's full `source.mp4` when its frame count agrees with
`recording.frame_count` and original pose/twin frame indices are in range. The raw
capture preserves source frame identity despite irregular capture timestamps.
`recording.mp4` is trimmed/resampled (63 frames in this bundle versus 222 raw
frames) and MUST NOT be indexed by original frame id. A fallback to that clip
requires timestamp/trim alignment, explicitly recorded. Read only fixed members
into a temporary directory (no arbitrary archive extraction). For legacy JSON
without saved video/observations, report actionable unsupported-input error.
Sample up to 24 evenly spaced available subject frames; do not run every frame
just for shape calibration. Match pose log frame and person, preferably
person_index 0 (recording tracks closest subject); never mix multiple people.

For each selected frame reproduce Rust MediaPipe crop from its logged bbox:
upper body: side=height*2, left=x+width/2-side/2, top=y;
full body: side=max(width,height)*1.25, center=(x+w/2,y+h/2).
cv2.warpAffine to 256 square with black padding; RGB float32 [0,1], NHWC [1,256,256,3].
Use cv2.dnn.readNetFromONNX, named outputs, identify 65536-value segmentation
logits and 1-value presence. Segmentation output is logits, not probabilities;
foreground >0.0 is p>0.5. Reject presence <0.7, and nonfinite model outputs.

Fit only exposed, supported arm widths initially (left/right elbow child = upper
arm; left/right wrist child = forearm). For each sample use original measured
image keypoints at shoulder/elbow/wrist with score>=0.7, both in image. Require
projected segment length >=30 px. Sample 5 centers at t=.25,.375,.5,.625,.75 along
bone. Trace each perpendicular direction in mask pixels until first background;
require center foreground, both boundaries found before image/crop edges, widths
>=2 crop pixels, symmetry ratio <=2.0. Reject centerlines near another nonincident
limb segment or torso (to avoid fitting an overlap as fat). Use conservative
minimum half-width (not total envelope) for overlap robustness. Pixel radius in
original image = crop radius*side/256. Convert to existing inferred body units
with the mean posed segment camera Z / fitted focal_length_px. No calibration
claim. Reject Z<=0 and radii outside [.015,.14] inferred units or beyond [.4,2.5]
times prior. Aggregate robustly over frames (median, MAD rejection); require
support from >=3 distinct frames per segment. If insufficient keep prior and
explicitly say why. Keep bilateral bones separate unless explicitly reporting
symmetry pooling; do not promote hidden/conflicting evidence to observations.

Radii fitting happens BEFORE build_mesh so fitted geometry feeds the FEM solve.
Store raw accepted samples/support frame indices and radius source. Surface
widths describe the clothed visible silhouette, NOT anatomical tissue dimensions.
Keep priors for unobserved torso/head/depth axes. Do not imply this refits poses
or reduces upstream image landmark error. Add targeted tests for synthetic binary
capsule width recovery, invalid/truncated/overlapping masks, and missing members.

Run actual recording calibration and communicate evidence/coverage before claiming
success. A valid result may have unsupported segments, with honest diagnostics.
