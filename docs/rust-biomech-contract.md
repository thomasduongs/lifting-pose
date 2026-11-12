# Standalone Rust biomechanics subset

The runtime must be Rust without Python, libmujoco, MuJoCo FFI, OpenCV, or ONNX.
Create independent `crates/biomech-lite/Cargo.toml` (edition 2024), library and CLI.
Offline Python export and MuJoCo numerical reference generation are development tools.
Preserve existing project files and prior biomechanics results.

## Scope and source

Use installed MyoSim 0.2.5 and MuJoCo 3.17.0, `output/biomech-twin/model.mjb`.
Runtime forward kinematics covers the whole sourced model. Dynamic coordinates are
exactly `elv_angle_r, shoulder_elv_r, shoulder_rot_r, elbow_flexion_r` and left
equivalents (native left names must be inspected, not guessed). Other independent
coordinates and the free root are prescribed. Polynomial dependencies are exact
parameterizations, rather than soft constraints. Dynamic integration has no contacts,
ground reaction, balance control, or muscle activation inference. Do not claim
person-specific anatomy or forces from this monocular recording.

Study and port the applicable Apache-2.0 MuJoCo algorithms, attributing derived code.
Canonical source tag: https://github.com/google-deepmind/mujoco/tree/3.17.0
`src/engine/engine_util_misc.c`: circle/inside/sphere/cylinder tendon wrapping,
muscle gain, bias, activation. `engine_core_smooth.c`: kinematics and tendon traversal.
Already downloaded locally to `/tmp/lifting-mujoco-misc.c` and
`/tmp/lifting-mujoco-smooth.c`. Export original licenses/notices with the data.
Keep source-specific restrictions visible; unsupported model features must fail clearly.

## Model JSON v1 (exporter and Rust serde MUST share these exact keys)

`assets/biomech-lite/model.json`:

```
{
 "schema_version":1, "source":{...provenance...},
 "nq":129,"nv":128,"qpos0":[...],"standing_qpos":[...],"gravity":[0,0,-9.81],
 "dynamic_joints":[native joint index, ...8],
 "bodies":[{"name":str,"parent":usize,"pos":[3],"quat":[4],
   "ipos":[3],"iquat":[4],"mass":f64,"inertia":[3],"joints":[joint indices]}],
 "joints":[{"name":str,"kind":int,"body":usize,"qposadr":usize,"dofadr":usize,
   "pos":[3],"axis":[3],"limited":bool,"range":[2],
   "damping":f64,"armature":f64,"stiffness":f64,"springref":f64}],
 "sites":[{"name":str,"body":usize,"pos":[3]}],
 "geoms":[{"name":str,"body":usize,"kind":int,"pos":[3],"quat":[4],"size":[3]}],
 "couplings":[{"dependent":joint index,"master":joint index or null,"coef":[5]}],
 "tendons":[{"name":str,"path":[{"kind":int,"object":int,"prm":f64}],
   "stiffness":f64,"damping":f64,"springlength":[2]}],
 "muscles":[{"name":str,"tendon":usize,"gainprm":[9],"biasprm":[9],
   "dynprm":[3],"lengthrange":[2],"acc0":f64,"ctrlrange":[2]}]
}
```

All floats f64, all quaternions wxyz, SI model reference units. Native enum integers:
joint free0 ball1 slide2 hinge3; wrap site3 sphere4 cylinder5. `prm` is native
wrap_prm (side site id for wraps; -1 means no side site). Preserve native indexing,
including world body0. Geom kind is native mjtGeom. Couplings topologically sorted
using `biomech_model._couplings`. Tendon springlength is native deadband [low,high].
Scalar springref is model.qpos_spring at jnt_qposadr; damping/armature native dof.
All 416 muscle actuators and 424 paths may be retained; no mesh duplication required
for physics. Native joints scalar except free root; reject ball if not implemented.

`assets/biomech-lite/motion.json`:
`{"schema_version":1,"timestamps_seconds":[F],"frame_indices":[F],"qpos":[[nq]...],"source":{...}}`.
Use all 222 fitted frames including irregular timestamps. Do not silently smooth them.

## Library interface (core owner creates these public APIs)

`pub struct Model` with public serde fields as above; `Model::load(path)->Result<Self,String>`;
`Model::validate()->Result<(),String>`; `Model::project(&self,q:&mut[f64])->Result<(),String>`;
`Model::kinematics(&self,q:&[f64])->Result<Kinematics,String>`;
`Model::tendon_lengths(&self,k:&Kinematics)->Result<Vec<f64>,String>`;
`Model::muscle_forces(&self,lengths:&[f64],velocities:&[f64],activation:&[f64])->Result<Vec<f64>,String>`
returns positive total muscle tensions (passive + active), vectors in muscle order;
`pub fn activation_dot(ctrl:f64,act:f64,prm:[f64;3])->f64`.
`Kinematics { pub body_pos:Vec<[f64;3]>,pub body_quat:Vec<[f64;4]>,
pub site_pos:Vec<[f64;3]>,pub geom_pos:Vec<[f64;3]>,pub geom_quat:Vec<[f64;4]> }`.
Lengths vector is tendon order, velocities vector tendon order, activation muscle order.
Model::muscle_forces selects each muscle's own tendon and uses gainprm/biasprm separately.
Core owner owns Cargo.toml, src/lib.rs, src/model.rs, src/kinematics.rs,
src/wrapping.rs, src/muscle.rs, tests/core_parity.rs. May use nalgebra 0.33, serde,
serde_json and std; add anyhow/clap for CLI owner if requested. Do not add FFI dependencies.
Announce any interface change before implementing it.

## Dynamics and CLI (separate owner)

Own src/dynamics.rs and src/main.rs, tests/dynamics_parity.rs, docs/rust-biomechanics.md.
Core lib declares `pub mod dynamics;` (empty stub until implementation arrives).
Build reduced 8x8 mass using COM and angular Jacobians of body frames at q,
source inertias transformed by body*iquat, and native armature via full scalar
coordinate Jacobian (couplings included). Gravity via body COM Jacobians.
Use centered differences with a documented step (e.g. 1e-6) for Jacobians; angular
Jacobian from relative quaternions/log, not component derivatives.
Compute centrifugal/Coriolis from finite differences of reduced mass matrix when
velocity is nonzero. Include muscle torque = -dL/dx * positive tension, tendon
deadband spring forces, scalar joint spring/damping projected through couplings.
No arbitrary fake masses or surrogate inertia. Semi-implicit Euler with explicit
step size, limit handling, finite-state rejection; document numerical limitations.

CLI subcommands: `inspect --model PATH`; `replay --model PATH --motion PATH --output PATH`
(headless CSV/JSON trajectory with clock, body/site positions, lengths and passive
forces; name it prescribed replay); `simulate --model PATH --output PATH --duration
SECONDS --dt SECONDS --muscle BIClong --excitation 0.25` (fixed-torso dynamic run).
CLI defaults paths assets/biomech-lite when invoked from project root. Provide clear
commands, output sizes, no GUI requirement. If core ready, native replay performance
benchmark should run all222frames. No need modify existing OpenCV app for this task.

## Export and independent numerical fixtures (separate owner)

Own scripts/export_biomech_lite.py, assets/biomech-lite/*,
scripts/test_export_biomech_lite.py. Export actual compiled arrays, no reimplementing
reference formulas. Include MuJoCo and MyoSim source license/NOTICE/component credits.
Record SHA256 of input MJB and recording, package versions, pinned source URLs.

`reference.json` top-level `{"schema_version":1,"cases":[...]}` each case:
`{"name":str,"qpos":[nq],"body_pos":[[3]],"body_quat":[[4]],"site_pos":[[3]],
"tendon_lengths":[ntendon],"tendon_velocities":[ntendon],"activation":[nu],
"muscle_tensions":[nu],"activation_derivative":[nu],"control":[nu],
"reduced_mass":[[8]],"reduced_gravity":[8]}`.
Cases: standing, several shoulder/elbow sweeps, left/right asymmetric poses,
representative recorded frames incl late frames, lengthening/shortening velocity,
partial/full activation. Use native mj_forward + helpers, exact coupling projection.
For mass: native mj_fullM then C.T M C, C is nv x8 chain-rule polynomial joint
parameterization derivative. Root and all nonselected independent coordinates held.
Gravity: native qfrc_bias at qvel=0 then C.T *bias (this is demand opposing gravity).
velocity cases assign full qvel=C*chosen8velocity before mj_forward; use ten_velocity.
Native muscles return actuator_force <=0; export -actuator_force. `act_dot` must
be populated from forward dynamics, not zeros. Preserve tension/torque sign.

Also export `reduced_velocity:[8]`, `reduced_bias:[8]`, and
`reduced_muscle_torque:[8]` for each reference case. The bias is required generalized
demand including gravity and Coriolis on the exactly parameterized manifold:
`C.T * (qfrc_bias + M * Cdot * xdot)`, where `Cdot` can be centered-differenced
along xdot (recompute C at x +/- h*xdot, h=1e-5). Do not use qfrc_inverse, which
also includes passive/constraint effects. Muscle torque is `C.T*qfrc_actuator`.
These fixtures exercise nonzero-velocity coupling curvature and force signs.
Dynamics tests should compare these too; force/Jacobian error near a wrap branch
must be identified and reported, rather than hidden by broad tolerance changes.
Tests validate schema, deterministic export and native comparison internally.

## Acceptance

Rust unit/integration tests compare ALL body/site positions, all path lengths,
muscle forces and activation rates against independent exported reference cases.
Aim kinematics <1e-8m, wrapping lengths <1e-7m, force <1e-5N, reduced mass relative
<1e-5 and gravity <1e-4Nm. Investigate errors rather than broadening tolerances.
Exact parity is limited to implemented subset; report observed max error and unsupported
features. Replay all222fittedframes in Rust; run short finite dynamic excitation.
Deliver release binary size and timing, Rust-only runtime command, source attribution,
and clear distinction between prescribed recording and forward dynamic simulation.

## Settled dynamics algorithm and API

Implementation decisions are fixed here so D1 needs no further physics design.
Use nalgebra dynamic matrices internally. Expose:

```
pub struct State { pub qpos:Vec<f64>, pub velocity:Vec<f64>, pub activation:Vec<f64> }
pub fn mass_matrix(model:&Model,q:&[f64])->Result<nalgebra::DMatrix<f64>,String>;
pub fn gravity(model:&Model,q:&[f64])->Result<Vec<f64>,String>;
pub fn bias(model:&Model,q:&[f64],v:&[f64])->Result<Vec<f64>,String>;
pub fn muscle_torque(model:&Model,q:&[f64],v:&[f64],a:&[f64])->Result<Vec<f64>,String>;
pub fn step(model:&Model,state:&mut State,excitation:&[f64],dt:f64)->Result<(),String>;
```

For each of eight x coordinates, perturb its scalar qpos by +/-h (h=1e-6), project
all couplings and call kinematics. For each body compute world COM as
`body_pos + rotate(body_quat, body.ipos)`; its Jv column is centered COM difference.
Jw column is the scaled-axis/log vector of `Rplus * Rminus.transpose()` divided
by 2h (WORLD angular velocity). Use normalized UnitQuaternion and scaled_axis.
Compute C's scalar rows from centered qpos differences using joint qposadr/dofadr;
the held free-root rows are zero. All scalar joint rows, including dependencies,
contribute native armature, spring and damping; do not only sum eight masters.

At central q, Iworld = Rbody*Riquat*diag(inertia)*(Rbody*Riquat).transpose().
`M = sum_b(mass_b*Jv_b.T*Jv_b + Jw_b.T*Iworld_b*Jw_b) + C.T*diag(armature)*C`.
`gravity = -sum_b(Jv_b.T * mass_b * model.gravity)` (required holding torque).
Skip world body; massless bodies contribute no rigid inertia, but their descendant
Jacobians still matter. This matches `C.T * native_M * C`.

For nonzero v, get dM/dx[k] by centered mass_matrix differences with OUTER step
1e-4 to avoid cancellation from the inner 1e-6 Jacobians. Coriolis component i is
`sum_jk (dM/dx[k][i,j] - 0.5*dM/dx[i][j,k])*v[j]*v[k]`.
`bias = gravity + Coriolis`; v=0 bypasses mass derivative calls.

Tendon Jacobian Jt (ntendon x8) is centered length differences at the same projected
q+/-h. Length velocity = Jt*v. Muscle tensions come from Model::muscle_forces;
sum torques `-Jt[tendon,:]*tension`. Spring tension is
`stiffness*(length-clamp(length,low,high)) + damping*length_velocity`, and its torque
also uses `-Jt.T`. Scalar passive spring/damping is projected `C.T*(-k*(q-qref)-d*C*v)`.

Step: validate dimensions, all finite, dt>0 and dt<=0.002; activation is forward
Euler with native activation_dot and clamp [0,1]. Forces use the pre-step activation,
matching the source model's actearly=false; activation is advanced for the next step.
Compute qdd by Cholesky solve
of `M*qdd = muscle_torque + passive_torque - bias`. Semi-implicit velocity then
selected scalar coordinate update, then project. Reject nonfinite state. If a master
hits a native bound, clamp it and set its velocity0. If a dependent bound is crossed,
backtrack the proposed displacement until every scalar dependent bound is valid,
setting affected velocities0. Return an error rather than emitting invalid bounds if
this cannot be resolved. Stage updates in a clone so failed steps preserve input state.
This is a documented hard-bound approximation to native soft constraints.

## P1 analytic body Jacobian optimization (supersedes finite body Jacobians)

Use the existing `Kinematics.joint_anchor` and `joint_axis` world frames. Build C
analytically: selected master rows are unit vectors, locked rows zero, then each
dependent row = quartic slope at its master times that master's C row, in dependency
order. For each body's COM, visit that body and every ancestor's scalar joints.
A hinge contributes world linear Jacobian `axis.cross(COM-anchor)` and angular
Jacobian `axis`; a slide contributes linear `axis`, angular zero. Multiply each
contribution by that joint's C row and sum into the reduced body Jv/Jw. Free root
is prescribed and contributes zero. Keep the same inertia/armature/gravity formula.
This needs ONE FK per mass/gravity evaluation instead of seventeen. Keep centered
mass derivatives for Coriolis and tendon-length derivatives initially. Public APIs
and model JSON do not change. Native fixture parity must still pass. Tighten bias
and torque assertions to 1e-4 Nm or better if measured results support it. Record
before/after timing for the same 500-step command in docs; no speed claim without
measurement. Address pending bounds comments while in dynamics, preserving atomic
failed steps. This is a performance task owned by claude-mid.

P1 also adds `reduced_passive_torque:[8] = C.T*qfrc_passive` to native reference
cases and compares Rust `passive_torque` against it. This separately checks the
four active ligament springs and scalar damping, including moving poses. Native
actuator passive muscle bias belongs to `reduced_muscle_torque`, not this field.
