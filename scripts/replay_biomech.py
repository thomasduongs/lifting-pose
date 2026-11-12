#!/usr/bin/env python3
"""Native MuJoCo replay of a fitted biomechanical subject (kinematic, prescribed motion).

This is NOT the Rust ``--replay`` (which replays the original recording's twin) and not the
FEM video. It plays the sourced MyoSim reference model at the fitted joint angles from
``motion.npz``. The motion is prescribed: no muscle force, activation or dynamics are
simulated here, so the viewer shows kinematics only.

On macOS the passive viewer needs ``mjpython``; run it yourself from a terminal:

    .venv-biomech/bin/mjpython scripts/replay_biomech.py output/biomech-twin --speed 1.0 --loop

``--inspect`` checks the inputs headlessly, prints a JSON summary and exits without
opening a viewer. Automated checks should use ``--inspect``.

Playback interpolates between fitted frames with MuJoCo's own differentiation and
integration (``mj_differentiatePos``/``mj_integratePos``, correct for the free-root
quaternion), then re-enforces the native joint couplings, so dependent joints are never
interpolated independently. Irregular timestamps are kept as recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biomech_model as bm  # noqa: E402  (coupling projection, same pinned MuJoCo)


class ReplayError(Exception):
    pass


def load_run(run_dir: Path, subject: str | None = None):
    """Return (model, qpos, timestamps, info). ``subject`` selects a subdirectory."""
    run_dir = Path(run_dir)
    if subject:
        run_dir = run_dir / subject
    if not run_dir.is_dir():
        raise ReplayError(f"not a directory: {run_dir}")
    model_path = run_dir / "model.mjb"
    motion_path = run_dir / "motion.npz"
    if not model_path.is_file() or not motion_path.is_file():
        subdirs = sorted(p.name for p in run_dir.iterdir()
                         if p.is_dir() and (p / "motion.npz").is_file()) if run_dir.is_dir() else []
        hint = f" (subject directories: {', '.join(subdirs)}; pass --subject)" if subdirs else ""
        raise ReplayError(f"{run_dir} needs model.mjb and motion.npz{hint}")
    try:
        model = mujoco.MjModel.from_binary_path(str(model_path))
    except Exception as exc:
        raise ReplayError(f"cannot load {model_path} with MuJoCo {mujoco.__version__}: {exc}") from exc
    motion = np.load(motion_path, allow_pickle=False)
    qpos = np.asarray(motion["qpos"], dtype=np.float64)
    ts = np.asarray(motion["timestamps_seconds"], dtype=np.float64)
    validate(model, qpos, ts)
    info = {
        "run_dir": str(run_dir),
        "frames": int(qpos.shape[0]),
        "nq": int(model.nq), "nv": int(model.nv), "njnt": int(model.njnt),
        "nu": int(model.nu), "ntendon": int(model.ntendon), "neq": int(model.neq),
        "duration_seconds": float(ts[-1] - ts[0]),
        "dt_min_seconds": float(np.min(np.diff(ts))) if ts.size > 1 else None,
        "dt_max_seconds": float(np.max(np.diff(ts))) if ts.size > 1 else None,
        "coupling_residual_max": float(np.max(np.abs(bm.coupling_residuals(model, qpos[0]))))
        if model.neq else 0.0,
        "mujoco_version": mujoco.__version__,
    }
    return model, qpos, ts, info


def validate(model, qpos, ts):
    if qpos.ndim != 2 or qpos.shape[0] < 1:
        raise ReplayError(f"qpos must be (frames, nq) with at least one frame, got {qpos.shape}")
    if qpos.shape[1] != model.nq:
        raise ReplayError(f"qpos has {qpos.shape[1]} columns, model nq={model.nq}")
    if ts.shape != (qpos.shape[0],):
        raise ReplayError(f"timestamps_seconds shape {ts.shape} must be ({qpos.shape[0]},)")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(ts)):
        raise ReplayError("qpos or timestamps contain non-finite values")
    if ts.size > 1 and np.any(np.diff(ts) <= 0):
        raise ReplayError("timestamps must be strictly increasing")


def check_speed(speed: float) -> float:
    if not (isinstance(speed, (int, float)) and math.isfinite(speed) and speed > 0):
        raise ReplayError(f"speed must be a positive finite number, got {speed!r}")
    return float(speed)


def playback_time(elapsed_wall: float, speed: float, duration: float, loop: bool) -> float:
    """Recording-time offset in [0, duration] for a wall-clock elapsed time (seconds)."""
    t = elapsed_wall * speed
    if duration <= 0:
        return 0.0
    if loop:
        return math.fmod(t, duration)
    return min(t, duration)


def interpolate_qpos(model, qpos, ts, t_rel: float) -> np.ndarray:
    """qpos at recording time ``ts[0] + t_rel``, interpolated per frame pair in the Lie-group sense."""
    t = ts[0] + t_rel
    if t <= ts[0]:
        q = qpos[0].copy()
    elif t >= ts[-1]:
        q = qpos[-1].copy()
    else:
        i = int(np.searchsorted(ts, t, side="right") - 1)
        i = min(max(i, 0), qpos.shape[0] - 2)
        h = ts[i + 1] - ts[i]
        alpha = (t - ts[i]) / h
        diff = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, diff, 1.0, qpos[i], qpos[i + 1])
        q = qpos[i].copy()
        mujoco.mj_integratePos(model, q, alpha * diff, 1.0)
    bm.project_couplings_qpos(model, q)
    return q


def run_viewer(model, qpos, ts, speed: float, loop: bool):
    try:
        from mujoco import viewer
    except ImportError as exc:
        raise ReplayError(f"mujoco.viewer unavailable: {exc}") from exc
    data = mujoco.MjData(model)
    duration = float(ts[-1] - ts[0])
    try:
        handle = viewer.launch_passive(model, data)
    except RuntimeError as exc:
        raise ReplayError(
            f"cannot open the viewer ({exc}). On macOS run this with mjpython: "
            ".venv-biomech/bin/mjpython scripts/replay_biomech.py ..."
        ) from exc
    with handle:
        start = time.monotonic()
        while handle.is_running():
            wall = time.monotonic() - start
            if not loop and wall * speed >= duration:
                break
            rel = playback_time(wall, speed, duration, loop)
            data.qpos[:] = interpolate_qpos(model, qpos, ts, rel)
            mujoco.mj_forward(model, data)
            handle.sync()
            time.sleep(1.0 / 60.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path, help="fitted run directory (model.mjb, motion.npz)")
    p.add_argument("--subject", default=None, help="subject subdirectory of run_dir, if any")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed factor (>0)")
    p.add_argument("--loop", action="store_true", help="repeat the recording")
    p.add_argument("--inspect", action="store_true",
                   help="validate inputs, print a JSON summary and exit (no viewer)")
    args = p.parse_args(argv)
    try:
        check_speed(args.speed)
        model, qpos, ts, info = load_run(args.run_dir, args.subject)
        if args.inspect:
            print(json.dumps(info, indent=1))
            return 0
        run_viewer(model, qpos, ts, args.speed, args.loop)
    except ReplayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
