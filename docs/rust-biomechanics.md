# biomech-lite command line

`biomech-lite` is a standalone Rust binary (crate `crates/biomech-lite`) for the reduced
MyoSim full-body arm model. It runs without Python. The Python exporter
(`scripts/export_biomech_lite.py`) is an offline step that writes the JSON assets read here.
See `docs/rust-biomech-contract.md` for the model schema and the numerical design.

Build and run from the project root:

```
cargo build --release --manifest-path crates/biomech-lite/Cargo.toml
crates/biomech-lite/target/release/biomech-lite <command> [options]
```

The binary is written under the crate's own `target/` directory, not the repository root.

Paths default to `assets/biomech-lite/model.json` and `assets/biomech-lite/motion.json`,
relative to the current directory.

## Commands

### `inspect`

```
biomech-lite inspect [--model assets/biomech-lite/model.json]
```

Loads and validates the model, then prints counts (bodies, joints, couplings, sites, tendons,
muscles), the 8 reduced fixed-torso coordinates and gravity. Unsupported native features are
rejected at load time with a message naming the feature.

### `replay` (prescribed motion, not dynamics)

```
biomech-lite replay --output out/replay.json \
    [--model assets/biomech-lite/model.json] [--motion assets/biomech-lite/motion.json]
```

Projects each recorded frame onto the coupling manifold, then computes full forward kinematics,
all tendon lengths and passive muscle tensions. Writes one JSON document with
`"mode":"prescribed"`. All 222 fitted frames are used with their original, irregular
timestamps; nothing is smoothed. Frames are streamed to the output file as they are computed.

Per frame: `frame_index`, `timestamp_seconds`, `qpos` (nq), `body_pos` (all native bodies,
world included), `site_pos`, `tendon_lengths`, `muscle_tensions`. `summary.elapsed_seconds`
is at the end of the file.

`muscle_tensions` are **passive isometric evaluations**: activation is 0 and tendon velocity is
0. They are not forces inferred from the recording. The recording has no muscle activations, so
replay cannot say what the muscles were doing.

### `simulate` (fixed-torso forward dynamics)

```
biomech-lite simulate --output out/sim.json --duration 0.5 --dt 0.001 \
    --muscle BIClong --excitation 0.25 [--model assets/biomech-lite/model.json]
```

Starts from the standing pose with zero reduced velocity and zero activation. One selected muscle
receives constant excitation; all other muscles stay at 0. The integrator is semi-implicit
Euler on the 8 reduced coordinates. Writes `"mode":"forward_dynamics"`, with one frame per
step (including t=0) holding reduced coordinates, reduced velocities and the selected muscle's
activation.

| Option | Rule |
|---|---|
| `--duration` | finite, in (0, 600] s |
| `--dt` | finite, in (0, 0.002] s |
| `duration/dt` | steps = ceil(duration/dt), 1 to 2,000,000; the last step is shortened so simulated time ends exactly at `--duration` |
| `--excitation` | finite, inside the muscle's `ctrlrange` |
| `--muscle` | must exist (see `inspect`) |

Any non-finite state stops the run with an error. No partial "valid-looking" output is produced
after that point, but the output file is left partly written.

## Scope and limitations

- **Prescribed replay and forward simulation are different things.** Replay reproduces the
  recorded kinematics and evaluates passive lengths and forces at them. Simulate integrates the
  reduced fixed-torso arm under muscle and passive forces from a chosen excitation. Neither one
  is a fitted or validated motion model.
- Reduced dynamics cover only the 8 coordinates listed by `inspect`. The torso and the free root
  are held fixed. Contact, balance and all other independent coordinates are excluded; the
  unobserved ones are prescribed.
- Joints: free root, slide and hinge. Ball joints are rejected at load.
- Couplings are exact parameterizations, not soft equality constraints.
- Dynamics use hard-bound clamping on coordinates, a documented approximation to MuJoCo's soft
  limits. Body COM and angular Jacobians and the coupling matrix are analytic (one forward
  kinematics pass per mass or gravity evaluation). The tendon-length Jacobian is still a centered
  finite difference (step 1e-6, 16 kinematics passes with wrapping). The Coriolis term uses
  centered differences of the mass matrix with a 1e-4 outer step.
- Simulation is fixed-step only (`dt <= 0.002`), with no adaptive stepping and no contact.
- Output is JSON only. Output size grows with frames and bodies; `replay` writes all bodies.

## Attribution and licenses

The kinematics, wrapping and muscle code are Rust ports of MuJoCo 3.17.0
(Copyright 2021 DeepMind Technologies Limited, Apache-2.0). The model data is derived from
MyoSim (Apache-2.0) and its cited source models. See `assets/biomech-lite/NOTICE.md` and
`assets/biomech-lite/provenance/`.

## Results

Measured on the development Mac, `cargo build --release` (`lto = "thin"`, one codegen unit):

- Release binary `biomech-lite`: 789,088 bytes (about 0.8 MB).
- `inspect`: loads and validates the 129-coordinate model (104 bodies, 424 tendons, 416 muscles)
  and lists the muscle names accepted by `--muscle`.
- `replay` over all 222 fitted frames: 0.10 s in Rust. Output is 32.3 MB of JSON, dominated by
  the 2,042 sites and 104 bodies per frame.
- `simulate --duration 0.5 --dt 0.001 --muscle BIClong --excitation 0.25`: 500 steps in 1.18 s
  (about 2.3 ms per step), down from 3.33 s before the analytic Jacobians. The output file is
  197 KB. The remaining cost is the Coriolis term (16 mass evaluations per step) and the
  finite-difference tendon Jacobian (16 kinematics passes with wrapping) per step.
- Error handling checked: `--dt nan` is rejected with a clear message and exit status 1; an
  unknown `--muscle` is rejected and points to `inspect`.

Numerical parity against the exported reference cases is reported by the dynamics and export
tests, not by this document.
