#!/usr/bin/env python3
"""Independently compare an entire Rust prescribed replay against pinned MuJoCo.

Development-only: the Rust runtime does not import this script or require Python.
Run after `biomech-lite replay --output output/biomech-lite/replay.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def validate(replay_path, model_path, motion_path):
    replay = json.loads(Path(replay_path).read_text())
    motion = json.loads(Path(motion_path).read_text())
    model = mujoco.MjModel.from_binary_path(str(model_path))
    frames = replay["frames"]
    if len(frames) != len(motion["qpos"]) or not frames:
        raise ValueError("Rust replay must contain every source frame")
    if replay.get("mode") != "prescribed":
        raise ValueError("expected prescribed replay (passive isometric force evaluation)")
    errors = {}
    for index, frame in enumerate(frames):
        if frame["frame_index"] != motion["frame_indices"][index]:
            raise ValueError(f"frame index mismatch at {index}")
        if abs(frame["timestamp_seconds"] - motion["timestamps_seconds"][index]) > 1e-12:
            raise ValueError(f"timestamp mismatch at {index}")
        q = np.asarray(frame["qpos"], dtype=float)
        if q.shape != (model.nq,) or not np.isfinite(q).all():
            raise ValueError(f"invalid qpos at {index}")
        # Input is already coupled; a mismatch indicates the replay changed the fit.
        if np.max(np.abs(q - np.asarray(motion["qpos"][index]))) > 1e-10:
            raise ValueError(f"replay changed prescribed qpos at {index}")
        data = mujoco.MjData(model)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)  # zero qvel, activation and control
        for key, expected, tol in (
            ("body_pos", data.xpos, 1e-8),
            ("site_pos", data.site_xpos, 1e-8),
            ("tendon_lengths", data.ten_length, 1e-7),
            ("muscle_tensions", -data.actuator_force, 1e-5),
        ):
            actual = np.asarray(frame[key], dtype=float)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError(f"invalid {key} at {index}: shape {actual.shape}")
            delta = np.abs(actual - expected)
            worst = float(delta.max(initial=0))
            if key not in errors or worst > errors[key]["max_absolute_component_error"]:
                errors[key] = {
                    "max_absolute_component_error": worst,
                    "frame_index": frame["frame_index"],
                    "array_index": list(np.unravel_index(int(delta.argmax()), delta.shape)),
                    "tolerance": tol,
                }
            if worst > tol:
                raise ValueError(f"{key} mismatch at frame {index}: {worst} > {tol}")
    return {
        "passed": True,
        "frames": len(frames),
        "duration_seconds": motion["timestamps_seconds"][-1] - motion["timestamps_seconds"][0],
        "comparison": "independent native MuJoCo; zero-velocity, zero-activation passive evaluation",
        "mujoco_version": mujoco.__version__,
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=Path("output/biomech-lite/replay.json"))
    parser.add_argument("--model", type=Path, default=Path("output/biomech-twin/model.mjb"))
    parser.add_argument("--motion", type=Path, default=Path("assets/biomech-lite/motion.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.replay, args.model, args.motion)
    text = json.dumps(report, indent=2, default=lambda x: int(x)) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
