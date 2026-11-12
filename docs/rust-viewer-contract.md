# Native Rust viewer contract

Build `crates/biomech-viewer`, an independent Rust executable/library depending on
`../biomech-lite`. No Python, MuJoCo, OpenCV, ONNX or browser runtime. Offline visual
asset export may use the installed native reference engine. Preserve core physics
and its existing tests; all transforms/paths/forward simulation use biomech-lite.

## Architecture and appearance

Use minifb 0.29 native window, font8x8 0.3, png 0.18 for screenshots. Native window
defaults 1280x860, with 280px left controls and a 1000x760 3D viewport under a 60px
header and above a 40px timeline. Resizing must remain usable (min size or scale).
Use a dark navy background, ivory shaded bone meshes, muted red muscle paths,
amber ligament paths, cyan selected path, a subtle ground grid and clear text.
The bone geometry must be the actual sourced meshes, not stick replacements.
Muscles are MTU paths, not volumetric muscle bellies; label that honestly.

Software 3D renderer: real z-buffered triangles, flat diffuse lighting, orbit camera,
perspective projection, near-plane handling (clip or reject triangles safely),
depth-tested thick path lines (optional x-ray paths toggle), no painter-only sorting.
Use f64 simulator math then f32/f64 screen math. Native z-up, subject faces -Y.
Default front camera sees face; front/side/back presets, orbit, pan, scroll zoom,
reset and full-body/upper-body framing. 330k triangles is the source model scale;
aim responsive release rendering at 1000x760, report actual timings. Do not drop
anatomy or silently decimate if slow; first cache buffers and optimize raster work.

## Offline visual asset schema (A1 owns exporter/data)

`assets/biomech-lite/visual.json` (separate from physics model):

```
{"schema_version":1,"source_model_sha256":"MJB SHA256",
 "source":{"mujoco":"3.17.0","myo_sim":"0.2.5","note":"compiled mesh frames"},
 "meshes":[{"name":str,"vertices":[[x,y,z],...],"faces":[[i,j,k],...]}],
 "geoms":[{"geom":native_geom_id,"mesh":native_mesh_id,"body":native_body_id,
             "rgba":[r,g,b,a],"unobserved":bool}],
 "reference":{"qpos":[nq],"samples":[{"geom":id,"vertex":local_mesh_vertex_id,
                                      "world_position":[x,y,z]}]}}
```

Keep native mesh indexing (all104 meshes), all102 mesh geoms. Vertices are compiled
`mesh_vert` slices, faces local indices from `mesh_face` slices. Runtime world transform
is `kin.geom_pos[g] + kin.geom_mat[g]*vertex`; do not apply mesh_pos/quat a second time.
Validate face indices, finite arrays, and source MJB hash matching model.source.mjb_sha256.
Reference samples: at standing native qpos, select first/middle/last vertex for each
mesh geom, transform with native geom_xmat/geom_xpos. These independently verify geometry
attachment. Unobserved lower body same set as scripts/render_biomech.py. Retain source
notices; no network fetch at runtime. Export atomically and deterministically.

## Renderer public API (R1 owns renderer modules)

Renderer owner owns `src/scene.rs`, `src/camera.rs`, `src/render.rs` and renderer tests.
Coordinator owner creates Cargo.toml/lib.rs with public modules; renderer may not edit
them without coordinating. Source model types from biomech_lite.

```
scene::VisualModel::load(path:impl AsRef<Path>,model:&Model)->Result<Self,String>
camera::Camera::front()->Self
Camera::preset(&mut self,name:&str) // front,side,back
Camera::fit(&mut self,model:&Model,kin:&Kinematics,visual:&VisualModel,upper:bool)
Camera::orbit(&mut self,dx:f64,dy:f64)
Camera::pan(&mut self,dx:f64,dy:f64)
Camera::zoom(&mut self,delta:f64)
pub struct render::Options { pub bones:bool,pub muscles:bool,pub ligaments:bool,
 pub joints:bool,pub xray:bool,pub selected_muscle:Option<usize> }
impl Default for Options
pub struct Renderer // cached reusable buffers
Renderer::new(width:usize,height:usize)->Self
Renderer::resize(&mut self,width:usize,height:usize)
Renderer::render(&mut self,model:&Model,kin:&Kinematics,visual:&VisualModel,
 camera:&Camera,options:&Options)->Result<(),String>
Renderer::pixels(&self)->&[u32] // 0x00RRGGBB, width*height
```

Any additional methods fine; announce needed changes. Renderer calls model.tendon_path
at current kinematics, selects actuated paths by muscle.tendon, unactuated routes as
ligaments. Wrapped arcs may initially be visual chords; physics retains exact arc
length. Show true world joint anchors for optional joints. Bones hold source geometry.
Perspective-correct reciprocal depth for triangle/line visibility if perspective used.
Tests: mesh sample attachment vs native reference, front/side visible nonempty images,
occlusion/clipping on synthetic triangles, camera finite orbit/zoom. Headless PNG render
must share this exact renderer with interactive mode; no fake separate preview renderer.

## Session/playback API (P1 owns crate manifests/lib and session.rs)

Create serde/nalgebra/path biomech-lite dependencies plus minifb/font8x8/png. Edition2024,
Apache2.0 attribution compatible. Release LTO thin. `lib.rs` declares camera/scene/render/session.
`session::Motion::load(path,model:&Model)->Result<Self,String>` with public
`timestamps_seconds:Vec<f64>, frame_indices:Vec<u64>, qpos:Vec<Vec<f64>>`.
Reject malformed clock, nonfinite qpos, wrong dimensions/quaternionnorms; same semantics
as core CLI. `Motion::duration()->f64`, `Motion::sample(&self,model:&Model,t:f64)->Result<Vec<f64>,String>`
where t is relative to first stamp, clamp endpoints, interpolate scalar coordinates and
free quaternion shortest-path SLERP, then exact project couplings. Handle irregular
timestamps correctly and preserve last frame. No smoothing or invented support.

Simulation should run on a background thread using biomech_lite::dynamics::step,
dt=0.001, fixed torso. Freeze all other independent coordinates at standing. Public:
`session::Simulation::spawn(model:std::sync::Arc<Model>)->Result<Self,String>`;
`Simulation::send(command:SimCommand)->Result<(),String>`;
`Simulation::latest(&mut self)->Option<SimSnapshot>`;
`SimCommand::{Pause(bool),Reset,Muscle(usize),Excitation(f64),Quit}`;
`SimSnapshot { pub time_seconds:f64,pub qpos:Vec<f64>,pub activation:Vec<f64>,
 pub velocity:Vec<f64>,pub error:Option<String> }`.
Bounded snapshot channel, responsive command handling, stop/join worker on Drop. Start
paused; muscle default BIClong; excitation0.25. Reset time/state on muscle selection;
excitation changes apply to next step. Snapshot at <=50 simulated Hz with current
complete state. Pause commands produce a current snapshot. No catch-up giant variable
dt; slower-than-real-time simulation is allowed and must be visible in UI clock.
Do not use imported simulation JSON to fake live forward dynamics.

## Native app (U1 same owner as session)

`cargo run --release --manifest-path crates/biomech-viewer/Cargo.toml` opens the viewer
with model/visual/motion defaults in assets/biomech-lite (resolve project-root assets
robustly, including invocation outside root via CARGO_MANIFEST_DIR fallback).
CLI --model/--visual/--motion, --simulation, --snapshot PATH, --time SEC, --view front|side|back,
--upper-body and --frames N optional smoke exit (native GUI only). `--snapshot` is
headless, no window; render frame at --time using the shared renderer and overlay.
Errors should be clear, no panics for file/window failures. --help lists controls.

Controls must be both visible and usable: Space play/pause, left/right step recorded
frames, Home reset, L loop, 1/2/3 front/side/back, F frame, U upper/full-body, B/M/G/J
toggle bones/muscles/ligaments/joints, X xray paths, Tab recording/simulation,
[ and ] select muscle, +/- excitation (bounded[0,1]), P save PNG. Mouse drag orbit,
right-drag pan, wheel zoom. Clickable sidebar buttons for play/reset/mode/layers,
muscle previous/next, excitation plus/minus, camera presets; clickable bottom scrubber
only in recording mode. Show current mode/time, speed, selected muscle, activation
(simulation only), layer legend, and lower body held/reference anatomy note. Avoid
printing force labels if values are not computed at correct velocity/activation.
Recording supports speed0.25x/0.5x/1x/2x via visible buttons or shortcuts and looping.
Playback starts playing in recording mode; simulation starts paused until play.
App mode switch pauses old mode. Headless snapshot gives a usable full screenshot.

## Documentation/launcher (D1 A1 owner after app ready)

Create executable `scripts/view_biomech.sh` that resolves root, cargo builds viewer in
release and execs with all supplied args. Default opens recording; --simulation opens
live arm simulator. Update docs/rust-viewer.md with exact commands, controls, runtime
dependencies/sourced anatomy limitations and headless preview. Root owns README and
docs/rust-viewer-results.md. Never delete or overwrite previous outputs.

## Acceptance

Native Rust window shows actual bones and paths attached to Rust FK at replay poses,
scrub/loop/lastframe timing correct; simulation displays actual Rust dynamic snapshots.
Release build and focused tests; independent source mesh attachment, headless screenshots
front/side/late frame visually inspected. Native smoke launch closes itself with --frames,
and leave a final normal viewer open for user after verification if environment permits.
User explicitly authorized computer control in this session; no need new permission
for viewer/window verification. Do not drive unrelated apps. Report measured timing,
binary and mesh data sizes. Retain pure headless core crate functionality.
