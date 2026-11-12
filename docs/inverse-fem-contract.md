# Recording-driven inverse FEM contract

This adds an offline Python workflow alongside the existing Rust estimator. Preserve
the original `output/recording.twin`, videos, and Rust behavior. No GUI or browser.

## Scientific scope

Fit a coarse tetrahedral elastic scaffold to the **existing estimated skeletal
motion**, not to a scan or ground truth. The unknown is a volumetric displacement
field. Minimize a corotated tetrahedral FEM energy plus weighted skeletal/skinning
target residuals. This is inverse displacement reconstruction with assumed geometry
and material, not identification of anatomy, tissue stiffness, forces, or an
unloaded reference shape. Radii, Young's modulus, and Poisson ratio are priors.
Report inversion, convergence, and residuals; never label low residual anatomical
accuracy. Upper-body recordings must exclude unobserved legs from the mesh.

## Modules and interfaces (NumPy + SciPy only)

`scripts/twin_fem.py` exports:

* `build_mesh(rest_joints, parents, active_bones, radii, spacing=0.06)` -> dict
  with `vertices` (N,3 float), `tetrahedra` (T,4 int), `surface_triangles` (S,3 int),
  `weights` (N,J float, rows sum to one). `rest_joints` are global rest positions;
  parents are integer indices with root -1; active_bones are child indices.
  Construct a connected conforming voxel tetrahedral union of bone capsules;
  deterministic positive nondegenerate elements. Include all intersecting cells
  needed for thin limbs. Weights associate vertices with bone child transforms.
* `skin_targets(mesh, rest_joints, posed_joints, world_rotations)` -> (N,3).
  A bone child's transform uses its parent's world rotation and the child's
  global joint position: x' = R_parent (x - rest_child) + posed_child.
* `solve_frame(mesh, targets, target_weights=None, young=1.0, poisson=0.35,
  data_weight=100.0, max_iterations=60)` -> dict with `vertices` and `diagnostics`.
  Default weights are all ones. Optimize total energy using an analytic gradient
  and proper polar rotations (SVD with det correction); report actual optimizer
  convergence, objective, target_rmse, min_det_f, inverted_elements, iterations.
  Require finite valid inputs and reject unsupported material parameters.
  No force/stress claims. Target weighting is a regularization choice.
  Large rotations require corotation, not plain small-strain linear elasticity.

`scripts/reconstruct_twin.py` CLI:

* input positional `.twin` ZIP or legacy `twin.json`; output `--output DIR`
  default `output/fem-twin`; optional `--subject`, `--stride`, `--max-frames`,
  `--spacing`, `--young`, `--poisson`, `--data-weight`, `--max-iterations`.
* Validate schema 1/2, hierarchy, finite dimensions, timestamps and quaternions.
  Match rotations **by joint name**, normalize valid nonzero xyzw quaternions;
  FK follows exported parent-before-child hierarchy and outgoing-bone convention.
  Position[i]=position[parent]+world[parent]@offset[i];
  world[i]=world[parent]@local[i], root world=root_orientation@local[root].
  Root position is exported camera translation. Rest joints use identity FK.
* Read ZIP member directly, never extract arbitrary paths. Process each subject
  independently. Active bones: full body if full-body; upper-body: thorax-to-head,
  shoulders/elbows/wrists, optional spine-to-thorax torso scaffold but NO legs.
  Skip zero-length head_pivot bone; retain hierarchy for FK. Radii are explicitly
  declared priors proportional to measured segment dimensions, not fitted girths.
* Write `manifest.json` provenance/assumptions/parameters/input SHA256/units,
  per-subject `mesh.npz` (rest_vertices, tetrahedra, surface_triangles, weights),
  `motion.npz` (frame_indices, timestamps_seconds, vertices, joint_positions),
  `rest.obj`, first posed `frame_000000.obj`, and `diagnostics.json` per frame and
  aggregate. Avoid silently mixing subject segments or using FPS for irregular
  timestamps. Timing lookup uses original frame indices; reject missing timing
  unless explicitly documented fallback to frame/fps for legacy no-timing files.
* Explicitly mark targets as inherited skeletal estimates. Report upstream
  reprojection_rmse_px separately when present; no claim of improving video fit.
  Preserve per-frame joint confidence in export. All-zero unsupported frames
  should fail/skip visibly. Mesh target confidence from bone endpoint confidence;
  do not call hidden inferred joints measured. Keep a small numerical anchoring
  floor documented as a prior for unsupported vertices.
* Print concise summary and failures. Refuse overwriting input/source. Run all
  frames in the supplied recording for the deliverable, reducing mesh resolution
  if needed rather than pretending a single frame is the complete recording.

## Validation

Targeted Python unittest cases: positive mesh volumes and conforming boundary,
rigid motion has near-zero FEM energy, analytic gradient finite differences,
nonrigid data residual improves over rest, invalid materials/degenerate elements,
name-based FK with schema-2 head pivot, ZIP input and irregular timing, upper-body
scope. Tests belong in `scripts/test_twin_fem.py` and
`scripts/test_reconstruct_twin.py`, owned by their respective implementation task.

## Research

UVA confirms Baek lab works on physics-aware learning and sports biomechanics:
https://datascience.virginia.edu/labs/baek-lab
https://datascience.virginia.edu/news/qa-stephen-baek-ai-extreme-physics-and-future-innovation
The exact user-referenced reverse-FEM paper has not been identified. Do not claim
to reproduce a Baek method. Related primary work: Ichim's thesis,
https://infoscience.epfl.ch/entities/publication/c8e08f4c-260c-4927-9ef9-bd5d54b11a91
and PhysTwin https://arxiv.org/abs/2503.17973 (deformable objects, multiple views,
not evidence that monocular landmarks identify human tissue parameters).
