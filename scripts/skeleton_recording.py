"""Saved-skeleton loading, validation, timing and forward kinematics for offline anatomy fitting.

These helpers are shared by the retained asset-generation tools. They do not build
soft-tissue meshes or reconstruct material properties.
"""
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

class TwinError(Exception):
    pass


def load_twin(path):
    """Return (twin dict, sha256 of input bytes). Reads the ZIP member directly."""
    path = Path(path)
    if not path.is_file():
        raise TwinError(f"Input not found: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            if "twin.json" not in z.namelist():
                raise TwinError("ZIP has no twin.json member")
            raw = z.read("twin.json")
            capture = None
            if "capture.json" in z.namelist():
                try:
                    capture = json.loads(z.read("capture.json"))
                except ValueError:
                    capture = None
    else:
        raw = path.read_bytes()
        capture = None
    try:
        twin = json.loads(raw)
    except ValueError as e:
        raise TwinError(f"twin.json is not valid JSON: {e}")
    if not isinstance(twin, dict):
        raise TwinError("twin.json must be an object")
    twin["_capture"] = capture
    return twin, digest.hexdigest()


def _finite(a):
    return bool(np.all(np.isfinite(a)))


def validate_hierarchy(twin):
    schema = twin.get("schema_version")
    if schema not in (1, 2):
        raise TwinError(f"Unsupported schema_version {schema!r} (need 1 or 2)")
    if schema == 2 or any(k in twin for k in ("quaternion_convention", "coordinate_axes")):
        for key, expected in (("quaternion_convention", "xyzw_active_parent_local"),
                              ("coordinate_axes", "camera_x_right_y_down_z_forward")):
            if twin.get(key) != expected:
                raise TwinError(f"{key} must be {expected!r}, got {twin.get(key)!r}")
    names = twin.get("joint_names")
    parents = twin.get("parents")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise TwinError("joint_names must be a non-empty list of unique names")
    if not isinstance(parents, list) or len(parents) != len(names):
        raise TwinError("parents must match joint_names in length")
    par = []
    roots = 0
    for i, p in enumerate(parents):
        if p is None or p == -1:
            par.append(-1)
            roots += 1
        elif isinstance(p, int) and not isinstance(p, bool) and 0 <= p < i:
            par.append(p)  # parent-before-child ordering
        else:
            raise TwinError(f"Joint {names[i]}: parent {p!r} invalid or not before child")
    if roots != 1 or par[0] != -1:
        raise TwinError("Hierarchy needs exactly one root at index 0")
    return names, par


def timestamps_for(twin, frame_indices):
    """Seconds per original frame index; fallback to frame/fps only if no timing exists."""
    rec = twin.get("recording") or {}
    ts = rec.get("timestamps_seconds")
    cap = twin.get("_capture") or {}
    if ts is None:
        ts = cap.get("timestamps_seconds")
    if ts is not None:
        ts = np.asarray(ts, dtype=float)
        if ts.ndim != 1 or not _finite(ts) or np.any(np.diff(ts) <= 0):
            raise TwinError("timestamps_seconds must be finite and strictly increasing")
        idx = np.asarray(frame_indices)
        if idx.size and (idx.min() < 0 or idx.max() >= ts.size):
            raise TwinError("A frame index has no timestamp (index outside timestamps_seconds)")
        return ts[idx], "recording_timestamps_by_original_frame_index"
    if twin.get("schema_version") == 1:
        fps = rec.get("fps") or cap.get("fps")
        if isinstance(fps, (int, float)) and fps > 0 and math.isfinite(fps):
            return np.asarray(frame_indices, dtype=float) / float(fps), "legacy_fallback_frame_over_fps"
    raise TwinError("Recording has no timestamps (fallback to fps only for legacy schema 1)")


def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not _finite(q):
        raise TwinError(f"Invalid quaternion {q!r}")
    n = np.linalg.norm(q)
    if n < 1e-8:
        raise TwinError("Zero-norm quaternion")
    return q / n


def quat_to_matrix(q):
    x, y, z, w = normalize_quat(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def forward_kinematics(parents, offsets, local_rotations, root_orientation, root_position):
    """position[i]=position[p]+world[p]@offset[i]; world[i]=world[p]@local[i];
    root world = root_orientation @ local[root]. Returns (positions (J,3), world (J,3,3))."""
    J = len(parents)
    pos = np.zeros((J, 3))
    world = np.zeros((J, 3, 3))
    for i in range(J):
        p = parents[i]
        if p < 0:
            world[i] = root_orientation @ local_rotations[i]
            pos[i] = root_position
        else:
            world[i] = world[p] @ local_rotations[i]
            pos[i] = pos[p] + world[p] @ offsets[i]
    return pos, world


def frame_rotations(frame, names):
    """Match rotations by joint name; returns (local (J,3,3), confidence (J,))."""
    by_name = {}
    for entry in frame.get("joint_rotations") or []:
        n = entry.get("joint")
        if n in by_name:
            raise TwinError(f"Duplicate rotation for joint {n}")
        by_name[n] = entry
    missing = [n for n in names if n not in by_name]
    if missing:
        raise TwinError(f"Frame {frame.get('frame')}: missing rotations for {missing}")
    local = np.zeros((len(names), 3, 3))
    conf = np.zeros(len(names))
    for i, n in enumerate(names):
        local[i] = quat_to_matrix(by_name[n]["quaternion_xyzw"])
        c = by_name[n].get("confidence", 0.0)
        if not isinstance(c, (int, float)) or not math.isfinite(c):
            raise TwinError(f"Joint {n}: invalid confidence")
        conf[i] = min(max(float(c), 0.0), 1.0)
    return local, conf


def safe_name(segment, index):
    return f"subject_{index:02d}_segment_{segment}"
