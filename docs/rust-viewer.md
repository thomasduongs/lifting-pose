# Native biomech viewer

`crates/biomech-viewer` is a native Rust window and headless renderer for the MyoSim full-body
reference model. It draws the sourced bone meshes, muscle-tendon paths and ligament routes at
replayed poses, and can run a live arm simulation. It needs no Python, MuJoCo, OpenCV, ONNX or
browser at runtime.

## Launch

```sh
./scripts/view_biomech.sh                          # recorded-motion replay (default)
./scripts/view_biomech.sh --simulation             # live arm simulator, paused until Space
./scripts/view_biomech.sh --view side --upper-body # start from side view, upper-body framing
./scripts/view_biomech.sh --help                   # all options and controls
```

The launcher builds the viewer with `cargo build --release` (first build takes a while; later
runs only rebuild what changed), then runs the binary from the project root. Any arguments are
forwarded unchanged. Build directly with:

```sh
cargo run --release --manifest-path crates/biomech-viewer/Cargo.toml -- [options]
```

Default data files are read from `assets/biomech-lite/` (`model.json`, `visual.json`,
`motion.json`). Override with `--model`, `--visual` and `--motion`.

## Double-click app (macOS)

```sh
./scripts/package_biomech_viewer.sh
```

This builds the release binary and writes `output/Biomech Viewer.app`, containing
`Contents/MacOS/biomech-viewer`, `Contents/Resources/assets/biomech-lite/` (model, visual and
motion JSON, `NOTICE.md`, `provenance/`) and an `Info.plist` with bundle id
`local.liftingpose.biomech-viewer`. Double-click it in Finder, or run `open "output/Biomech Viewer.app"`.
The bundle finds its own assets, so it works without the project folder. It is not installed,
not signed, and no global settings are changed. If a bundle already exists it is moved to
`Biomech Viewer.app.previous-<timestamp>`, not overwritten. The raw command-line binary does
not register as an app; use the bundle for a Dock or Finder launch.

## Headless preview

`--snapshot PATH` renders one full-UI frame with the same renderer as the window, writes a PNG
and exits. No window is opened.

```sh
./scripts/view_biomech.sh --snapshot /tmp/front.png --time 2.0 --view front
./scripts/view_biomech.sh --snapshot /tmp/side.png --time 2.0 --view side --upper-body
```

`--time SEC` sets the recording time used for the frame (default 0; clamped to the recording
duration). `--frames N` opens the window, renders N frames and exits, for a smoke test.

## Options

| Option | Meaning |
| --- | --- |
| `--model PATH` | physics model JSON (default `assets/biomech-lite/model.json`) |
| `--visual PATH` | visual mesh JSON (default `assets/biomech-lite/visual.json`) |
| `--motion PATH` | recorded motion JSON (default `assets/biomech-lite/motion.json`) |
| `--simulation` | start in live arm-simulation mode |
| `--snapshot PATH` | headless: write one PNG and exit |
| `--time SEC` | recording time for `--snapshot` (default 0) |
| `--view NAME` | `front`, `side` or `back` (default `front`) |
| `--upper-body` | frame the upper body instead of the full body |
| `--frames N` | open the window, render N frames, then exit |
| `-h`, `--help` | print options and controls |

## Controls

Keyboard:

| Key | Action |
| --- | --- |
| Space | play / pause |
| Home | reset |
| L | loop on / off |
| Tab | switch recording / simulation |
| Left / Right | step one recorded frame |
| `,` / `.` | slower / faster playback (recording: 0.25x, 0.5x, 1x, 2x) |
| 1 / 2 / 3 | front / side / back camera |
| F | frame all |
| U | upper body / full body |
| B / M / G / J | toggle bones / muscles / ligaments / joints |
| X | x-ray (draw paths through bones) |
| `[` / `]` | select previous / next muscle |
| `+` / `-` | excitation up / down (bounded to 0..1; simulation) |
| P | save PNG to `~/Pictures/Biomech Viewer/biomech_viewer_NNN.png` (never overwrites; `%USERPROFILE%` if `HOME` is unset). Independent of the launch directory, so it works when started from Finder |
| Esc | quit |

Mouse: drag to orbit, right-drag to pan, wheel to zoom. The sidebar has clickable buttons for
play, reset, mode, layers, muscle previous/next, excitation, and camera presets. In recording
mode the bottom timeline is a clickable scrubber.

## What is shown, and its limits

- **Bones** are the sourced compiled meshes from the MyoSim model (native mesh indexing, 104
  meshes, 102 visual mesh geoms), not stick figures. The triangle data is about 330k triangles.
- **Muscles** are muscle-tendon (MTU) paths computed from the model's wrapping geometry. They are
  not volumetric muscle bellies, and the viewer labels them as paths.
- **Ligaments** are unactuated tendon routes. Wrapped arcs are drawn as visual chords; the
  physics keeps the exact arc length.
- **Lower body** is not observed in the recording. It is drawn at the reference standing pose,
  marked as unobserved (16 geoms on 13 bodies: pelvis, femurs, patellae, tibiae, talī, calcanei,
  toes).
- **Recording** is a monocular kinematic fit of the source motion, not measured forces or
  measured muscle activation. Irregular timestamps are kept as fitted and interpolated with
  SLERP for free rotations.
- **Simulation** runs the arm on a background thread with the Rust dynamics (`dt` = 0.001 s).
  Eight dynamic coordinates move (shoulder and elbow of both arms, elevation angles and
  rotations); the torso is fixed and all other coordinates are held at standing. If the
  simulation runs slower than real time, the UI clock shows it. Force labels are not printed
  because they are not computed at the correct velocity and activation.
- **Anatomy** is generic reference anatomy, not person-specific.

## Runtime dependencies

Rust toolchain (edition 2024) to build. At runtime the binary reads only the three JSON files in
`assets/biomech-lite/`. Window and text use `minifb` 0.29 and `font8x8`; PNG output uses `png`
0.18. Licences and upstream notices are in `assets/biomech-lite/NOTICE.md` and
`assets/biomech-lite/provenance/`.

Regenerating the assets needs the Python environment in `scripts/requirements-biomech.txt`:
`.venv-biomech/bin/python scripts/export_biomech_lite.py` (physics model and motion) and
`.venv-biomech/bin/python scripts/export_biomech_visual.py` (meshes and attachment fixtures).
The viewer does not need Python to run.
