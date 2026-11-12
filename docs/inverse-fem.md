# Inverse FEM twin reconstruction

An offline Python workflow that runs beside the Rust estimator. It reads a saved
recording and fits a coarse tetrahedral elastic scaffold to the **existing
estimated skeletal motion**. It does not modify `output/recording.twin`, the
videos, or any Rust behavior. No GUI or browser is involved.

Interface and scope rules: [inverse-fem-contract.md](inverse-fem-contract.md).
Literature notes: [inverse-fem-research.md](inverse-fem-research.md).

## What it estimates, and what it does not

* **Estimated:** a volumetric displacement field per frame, from minimizing a
  corotated tetrahedral FEM energy plus weighted skeletal/skinning target residuals.
* **Inputs are inherited estimates.** Targets come from the upstream skeleton
  (joint rotations and positions). They are not a scan, not ground truth, and not
  a video re-fit. The upstream `reprojection_rmse_px` is reported separately.
* **Priors:** unobserved radii, Young's modulus, and Poisson ratio. Optional
  silhouette fitting can replace supported visible-arm radius priors with width
  estimates from the video; these are clothed surface widths, not tissue girths.
* **Not claimed:** anatomical accuracy, tissue stiffness, forces, stress, or an
  unloaded reference shape. A low residual only means the scaffold follows the
  targets.
* **Upper-body recordings** exclude unobserved legs from the mesh.
* **Hidden joints** are not called measured. Their confidence is carried into the export.
* **Element validity:** a determinant barrier keeps each tetrahedron above
  `det(F)=0.05` by default. This prevents inversion at the accepted states; it does
  not prevent self-intersection or establish anatomical plausibility. Convergence
  and element validity are separate diagnostics.

## Install

From the repository root:

```sh
python3 -m pip install -r scripts/requirements-fem.txt
```

The solver needs only NumPy and SciPy. Video calibration and rendering need OpenCV:

```sh
python3 -m pip install -r scripts/requirements-fem-video.txt
```

## Reconstruct

```sh
python3 scripts/reconstruct_twin.py output/recording.twin
```

For the finer mesh and visible-width calibration used in the delivered run:

```sh
python3 scripts/reconstruct_twin.py output/recording.twin \
  --output output/fem-twin --fit-silhouette --spacing 0.04 --max-iterations 120
```

This is an offline optimization and can take several minutes on a CPU. Progress
is printed on the first frame, every 20 frames, and on the last frame.

Input is a positional `.twin` ZIP or a legacy `twin.json`. Output defaults to
`output/fem-twin`. The run refuses to write over or into the input location.

| Option | Purpose |
| --- | --- |
| `--output DIR` | Output directory (default `output/fem-twin`) |
| `--subject N` | Process one `subject_segment` id (integer; default all subjects) |
| `--stride N` | Use every N-th frame (default 1) |
| `--max-frames N` | Cap the number of frames |
| `--spacing M` | Voxel spacing for the mesh (default 0.06) |
| `--young E` | Young's modulus prior (default 1.0) |
| `--poisson V` | Poisson ratio prior, in [0, 0.49] (default 0.35) |
| `--data-weight W` | Target-residual weight, a regularization choice (default 100.0) |
| `--max-iterations N` | Optimizer iteration limit (default 60) |

`--fit-silhouette` replaces supported visible-arm radii with conservative
clothed-silhouette widths measured from the saved source video and a
segmentation model (`--segmentation-model`, default `models/mediapipe-pose.onnx`).
It needs the `.twin` file as input and optional `opencv-python`. Evidence, accepted
and rejected counts, and radius provenance go in the subject's entry in
`manifest.json`. Widths describe the visible silhouette, not anatomical dimensions.
The pipeline prefers `source.mp4` after checking frame count, dimensions and
frame-index coverage against the saved observations. It uses original frame
indices directly, preserving correspondence despite irregular capture times.
The trimmed `recording.mp4` is a fallback with explicit timestamp alignment.
Calibration on the supplied recording supported one forearm; other arm widths
kept their priors. See [run results](inverse-fem-results.md).
Design: [silhouette-fit-contract.md](silhouette-fit-contract.md). Without the flag,
all radii are the declared priors.

Behavior:

* Schema 1 and 2 are accepted. Hierarchy, dimensions, timestamps, and quaternions
  are validated. Rotations match joints **by name**, and quaternions are normalized.
* Each subject is processed independently. Segments are never mixed silently.
* Timing comes from the original frame indices. Legacy schema-1 files without
  timestamps can explicitly fall back to `frame/fps`; missing schema-2 timing is
  rejected.
* The input is read directly; archive members are never extracted to arbitrary paths.
* Frames are processed in the supplied recording. Mesh resolution can be reduced
  to keep a run tractable, and the summary says so.

## Outputs

Each subject gets a directory named `subject_<NN>_segment_<id>`:

* `mesh.npz`: `rest_vertices`, `tetrahedra`, `surface_triangles`, `weights`
* `motion.npz`: `frame_indices`, `timestamps_seconds`, `vertices`,
  `joint_positions`, `joint_confidence`, `joint_names`
* `rest.obj`: rest scaffold surface
* `frame_000000.obj`: first posed frame
* `diagnostics.json`: `aggregate` summary plus per-frame solver diagnostics
  (`converged`, `target_rmse`, `min_det_f`, `inverted_elements`, and others)

At the top level:

* `manifest.json`: provenance, assumptions, parameters, input SHA256, units,
  per-subject bone radius priors, and `failures`

Frames that cannot be solved (for example, all active-joint confidences zero) are
recorded as skipped with a reason. Invalid/skipped frames cause a nonzero CLI exit
while diagnostic artifacts remain available. A valid solve may reach its iteration
budget without converging; inspect `converged_frames` separately.

The default solver is sparse projected Newton: finite differences of the analytic
element stress provide a symmetrized element Hessian, whose negative eigenvalues
are clamped before sparse assembly. A determinant feasibility check and Armijo
line search guard each update. The elastic energy is corotated isotropic FEM with
a log barrier on element determinants. Geometry, material, barrier threshold and
confidence floors are declared modeling choices. This is a displacement inverse
problem, not inverse material identification or a dynamic equilibrium simulation.

## Render a mesh animation

```sh
python3 scripts/render_fem_twin.py output/fem-twin --output output/fem-twin/front.mp4
python3 scripts/render_fem_twin.py output/fem-twin --output output/fem-twin/side.mp4 --view side
```

* Reads `manifest.json`, then the subject's `mesh.npz` and `motion.npz`. Opens no windows.
* Software orthographic camera. Triangles are flat-shaded and drawn far to near,
  with dark edges. Bounds are fixed across all frames, so the camera does not move.
* `--view front` (default) or `--view side`.
* `--fps` (default 30): output is resampled by holding the previous reconstructed
  mesh. Irregular timestamps are kept and nothing is interpolated or smoothed.
* `--subject DIRECTORY`: required when the manifest lists more than one subject.
  Each video shows one subject.
* Every frame is labeled as the experimental FEM scaffold.

The renderer uses the twin's declared convention, `camera_x_right_y_down_z_forward`:
+y points down in the image, and +z points away from the camera. The front view
draws (x, y) with larger z farther away. The side view draws (z, y) with larger x
nearer the viewer. Frame labels show the source `frame_indices` from `motion.npz`.

## Validation

```sh
python3 -m unittest discover -s scripts -p "test_*.py"
```

This runs the solver, reconstruction, silhouette and rendering tests (45 tests).
They cover positive
mesh volumes, rigid-motion energy near zero, analytic gradients against finite
differences, residual improvement over rest, rejection of invalid materials and
degenerate elements, name-based FK with the schema-2 head pivot, ZIP input with
irregular timing, and upper-body scope.

The renderer also has synthetic two-frame coverage (a cube with one
rotated frame, timestamps 0 and 0.1 s). It wrote 4 frames at 30 fps for both views
and reports a clear error for an unknown subject. Final recording artifacts and
their measured diagnostics are listed in [run results](inverse-fem-results.md).

## Limitations

* Monocular upstream targets constrain the skeleton, not the volume. Many
  displacement fields fit the same data, so the scaffold is not identifiable.
* Geometry and material are assumed, not measured.
* The specific Baek reverse-FEM paper referenced in the project has **not been
  identified**. This workflow does not claim to reproduce any Baek method.
  Related work is discussed in the research note.
