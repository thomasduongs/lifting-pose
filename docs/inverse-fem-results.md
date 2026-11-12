# Recording-driven inverse FEM reconstruction

The delivered workflow creates a coarse upper-body volumetric twin from the saved
recording. It reconstructs a displacement field using the existing skeletal
estimates plus limited visible-width evidence. Anatomical accuracy has **not**
been established by this recording.

## Reproduce

From the repository root:

```sh
python3 -m pip install -r scripts/requirements-fem-video.txt
python3 scripts/reconstruct_twin.py output/recording.twin \
  --output output/fem-twin --fit-silhouette --spacing 0.04 --max-iterations 120
python3 scripts/render_fem_twin.py output/fem-twin \
  --output output/fem-twin/front.mp4
python3 scripts/render_fem_twin.py output/fem-twin \
  --output output/fem-twin/side.mp4 --view side
```

The source is `output/recording.twin`, SHA256
`c35cbe0df2afa6a31ac1db8a9be01350847a7e3a192c20fa0ee2d34ce511a54e`.
It has 222 raw frames, a schema-2 rig with a separate head pivot, and irregular
capture timestamps. Units are `mediapipe_estimated_meters`, not camera-calibrated
meters. The full raw video has 222 frames; the separately trimmed preview has 63.
Calibration verifies raw frame count and dimensions before matching original
frame indices. The original bundle and existing Rust behavior are preserved.

## Surface evidence

The installed MediaPipe Pose ONNX model supplies a person-segmentation logit map.
We sample up to 24 frames and use supported, untruncated, reasonably symmetric
arm cross-sections. Rejected observations and unsupported radii remain explicit
in `manifest.json`. Visible widths constrain the clothed silhouette, not tissue.
One forearm has sufficient support in this recording. All other radius estimates
remain declared priors. No new model downloads or training are required.

## Numerical method

The scaffold is a connected conforming union of voxel tetrahedra around active
bones. The inverse objective combines corotated elastic energy, weighted distances
to skeletal skinning targets, and a determinant log barrier. Sparse projected
Newton steps use a finite-difference approximation of the analytic stress
derivative, positive-semidefinite element-Hessian projection, and feasibility plus
Armijo backtracking. The default determinant floor is 0.05. Frame-to-frame warm
starts improve convergence without changing the observations.

The solve can deviate from skinning targets to keep tetrahedra valid. Element
validity does not prevent global surface intersections, and a determinant floor
of 0.05 still permits substantial local compression. This is not a tissue
equilibrium, contact, muscle, force, stress, or injury model.

## Verification

`python3 -m unittest discover -s scripts -p 'test_*.py'` passed 45 tests,
including analytic-gradient finite differences, rigid motion, positive reference
volumes, bent-elbow inversion regression, representative real recording frames,
name-based forward kinematics, irregular timing, archive input, conservative
silhouette fitting, video alignment, and renderer camera orientation.
Eight recording-pipeline tests also passed after the final integration changes
to progress logging, validity reporting and metadata. Rust files were unchanged;
no broad Rust suite or browser/desktop verification was needed.

## Research and task ownership

[Paper comparisons and exact implementation boundaries](inverse-fem-research.md)
cover Kadlecek et al.'s inverse anatomical modeling, Smith et al.'s robust flesh
simulation, PPR, and segmentation-supervised shape reconstruction. We did not
identify a specific Baek lab reverse-FEM paper and do not claim to reproduce one.

Codex led research, API contracts, numerical-design guidance, and integration.
Claude Opus implemented and repaired the FEM solver; Claude Sonnet built the
recording pipeline and video calibration; Claude Haiku wrote documentation and
the headless renderer. Claude owns eight of the ten completed/planned work items.

## Accuracy boundary

The FEM target error measures agreement with the saved estimated skeleton.
The original mean image-joint reprojection error is approximately 11.57 pixels;
the FEM stage does not refit those image joints. No independent 3D ground truth,
surface scans, measured tissue dimensions, or loading data are present. A
highly accurate anatomical twin cannot be verified from these inputs alone.
