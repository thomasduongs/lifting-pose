# Native Rust viewer: delivered result

The native viewer is built and was opened on the development Mac. It renders the
MyoSim anatomy using the existing Rust `biomech-lite` forward kinematics, tendon
wrapping and forward dynamics. Python, MuJoCo, OpenCV, ONNX and a browser are not
needed at runtime.

## Open it

```sh
./scripts/view_biomech.sh
./scripts/view_biomech.sh --simulation
```

The first command plays the fitted recording, looping by default. The second starts
the live Rust arm simulator paused; press Space to run it. Tab switches modes. Drag
to orbit, right-drag to pan, scroll to zoom, and use the sidebar and bottom timeline
for layers, camera views, muscle selection, excitation and recording scrubbing.

The self-contained macOS application is `output/Biomech Viewer.app`; double-click
it in Finder. Rebuild it with `./scripts/package_biomech_viewer.sh`. Its resources
include the model, motion, meshes and upstream notices. Bundled model/motion/visual
SHA256 values were checked against the source assets. `Info.plist` passed `plutil`.
Automatic screenshots go to `~/Pictures/Biomech Viewer/biomech_viewer_NNN.png`,
without overwriting existing numbered images. Explicit `--snapshot PATH` chooses
a headless output location.

## Implemented behavior

- Actual source bone meshes: 104 native mesh assets, with all 102 visual mesh
  instances represented. There are 329,974 triangles in the complete mesh array;
  the displayed instances contain 287,542 triangles.
- Z-buffered, shaded, perspective 3D rendering with near-plane clipping, depth-tested
  paths, optional x-ray paths, selected-muscle highlighting and a ground grid.
- Front, side and back views, upper/full-body framing, layer toggles and optional
  joint markers. The recording camera fits the envelope of all 222 poses so raised
  hands remain in view. It does not zoom independently on each frame.
- Original irregular recording clock, shortest-path free-root SLERP and exact
  polynomial joint coupling projection. Playback supports frame stepping, scrubbing,
  looping and 0.25x/0.5x/1x/2x speeds, including the final recorded pose.
- Background simulation calls `biomech_lite::dynamics::step` at dt=0.001 s, with
  selected muscle/excitation controls. A bounded latest-state mailbox keeps window
  controls independent of physics work. The UI shows actual simulation time,
  activation and its speed relative to wall time. Mode changes pause the old mode.
- Headless PNG rendering shares the renderer and UI composition used by the native
  window. Paused, unchanged scenes cache the 3D image.

## Verification

Final functional checks on November 12, 2026, macOS arm64 / Rust 1.99:

- **26 Rust viewer tests passed**, covering 7 app tests, 11 renderer tests and 8
  session tests. One manual rendering timing test is intentionally ignored in the
  ordinary test run.
- **6 visual-export tests passed**, including native geometry arrays, attachment
  transforms, source hashes, deterministic output and atomic asset writing.
- All **306 native reference mesh attachment samples** match Rust transforms within
  1e-9 m. The visual source hash must match the physics model's MJB hash.
- Raster tests check draw-order-independent occlusion, intersecting-triangle depth,
  line occlusion/x-ray behavior, near clipping and a camera inside the body. Output
  is deterministic across renderer thread counts.
- App tests check looping, endpoint stopping, frame stepping, scrubbing, mode
  pausing, excitation bounds, clickable controls, resizing and stable camera framing.
  Framing tests project all mesh vertices at representative early/mid/late poses
  in front/side/back and full/upper-body views.
- Inspected front, side and late upper-body PNGs. A raised-hand clipping issue in
  the first-frame camera fit was corrected using a cached whole-motion envelope.
- Native window smoke test (`--frames 5`) exited successfully. The packaged app was
  then opened and checked directly: recording playback, switching to simulation,
  starting/pausing dynamics and PNG saving. Live simulation visibly advanced to
  11.267 simulated seconds with BIClong activation 0.250 at excitation 0.25.
- Finder launch exposed a read-only-working-directory screenshot bug. Automatic
  saves were changed to the Pictures folder; after rebuilding/relaunching, P saved
  `~/Pictures/Biomech Viewer/biomech_viewer_001.png` successfully. The packaged viewer
  was left open playing the recording.

Logs: `output/biomech-viewer/rust-tests.log` and `export-tests.log`. The screenshot
path correction subsequently passed the 7 targeted app tests and was checked in
the actual packaged window. Final release build is warning-free.

```sh
cargo test --release --manifest-path crates/biomech-viewer/Cargo.toml
.venv-biomech/bin/python scripts/test_export_biomech_visual.py
./scripts/view_biomech.sh --snapshot output/biomech-viewer/front.png
./scripts/view_biomech.sh --snapshot output/biomech-viewer/side.png --view side --time 4
./scripts/view_biomech.sh --snapshot output/biomech-viewer/late.png --time 8.059309708 --upper-body
```

## Size, performance and scope

The final release executable is **1,165,472 bytes**. `visual.json` is **16,280,649
bytes**, and the complete macOS app bundle, including anatomy/motion data and notices,
is **18,493,814 bytes**. At a 1000x760 viewport, checked headless frames took about
**10–12 ms** to render; the focused renderer benchmark measured 11.4 ms. The rasterizer
uses eight horizontal bands with deterministic depth and pixel output. These are
viewport timings, not a promise about window refresh rate.

Live simulation remains slower than real time on this machine (about 0.35x observed
while verifying the window). It runs in the background, without substituting recorded
poses for computed dynamics. The underlying headless simulator remains a separate crate
and gains no windowing dependencies.

Anatomy is generic reference anatomy. Muscles are MTU paths, not volumetric bellies;
wrapped arcs may be visualized as chords while physics retains exact lengths. The
recording fit retains its existing monocular uncertainty. Lower body is unobserved
and held at reference; dynamic simulation covers eight shoulder/elbow coordinates,
with the torso and other independent coordinates held. Contact, balance and personal
muscle activation/force identification are outside this version.

## Duo ownership and usage

Codex owned the viewer contract, integration and visual/native-window checks. Claude
owned **8 of 10 tasks**: Opus implemented rendering/camera/scene validation; Sonnet
implemented sessions, live simulation, app controls and motion-envelope framing;
Haiku implemented source mesh export, launch/docs, macOS packaging and the Finder
screenshot fix. Worker-reported cumulative API-equivalent usage estimate: **$4.28**,
excluding Codex lead usage. Original notices remain in the assets and app resources.
