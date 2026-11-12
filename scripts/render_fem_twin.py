#!/usr/bin/env python3
"""Headless mesh animation export for reconstruct_twin.py output.

Reads `manifest.json` and per-subject `motion.npz` / `mesh.npz` from a
reconstruction directory and writes an MP4. Software orthographic camera,
flat-shaded triangles drawn far-to-near, fixed bounds across all frames.
Opens no windows (no cv2.imshow/highgui).

Timing: frames are resampled at --fps by holding the previous reconstructed
mesh (no interpolation, no smoothing). Irregular source timestamps are kept.

Axes follow the twin's declared convention, camera_x_right_y_down_z_forward:
+x points right, +y points DOWN in the image, +z points forward (away from the
camera). Front view draws (x, y) with z as depth (larger z is farther). Side view
draws (z, y), with x as depth (larger x is nearer the viewer). Shading uses
absolute normal dot products, so it does not depend on the sign of the normals.

This visualizes the EXPERIMENTAL FEM scaffold. It is not a measurement of tissue,
anatomy, or video fit.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - reported to the user
    cv2 = None

LABEL = "EXPERIMENTAL FEM scaffold - inherited skeletal targets, not measurements"
LIGHT = np.array([0.4, 0.6, -0.7]) / np.linalg.norm([0.4, 0.6, -0.7])
BASE_COLOR = np.array([200.0, 170.0, 140.0])  # BGR-free RGB base tone
EDGE_COLOR = (30, 30, 30)  # dark edges (BGR)


class RenderError(Exception):
    pass


def load_subject(root, subject_dir):
    sdir = root / subject_dir
    try:
        mesh = np.load(sdir / "mesh.npz")
        motion = np.load(sdir / "motion.npz")
    except (OSError, ValueError) as e:
        raise RenderError(f"cannot read NPZ assets in {sdir}: {e}") from e
    tris = np.asarray(mesh["surface_triangles"], dtype=np.int64)
    verts = np.asarray(motion["vertices"], dtype=float)
    stamps = np.asarray(motion["timestamps_seconds"], dtype=float)
    src_frames = np.asarray(motion["frame_indices"]).astype(np.int64)
    if verts.ndim != 3 or verts.shape[2] != 3 or verts.shape[0] == 0:
        raise RenderError(f"{subject_dir}: motion vertices must be (F,N,3) with F>0")
    if stamps.shape != (verts.shape[0],) or not np.all(np.isfinite(stamps)):
        raise RenderError(f"{subject_dir}: timestamps missing or not finite")
    if src_frames.shape != (verts.shape[0],):
        raise RenderError(f"{subject_dir}: frame_indices length does not match vertices")
    if np.any(np.diff(stamps) < 0):
        raise RenderError(f"{subject_dir}: timestamps are not nondecreasing")
    if not np.all(np.isfinite(verts)):
        raise RenderError(f"{subject_dir}: non-finite vertices")
    if tris.ndim != 2 or tris.shape[1] != 3 or tris.size == 0:
        raise RenderError(f"{subject_dir}: surface triangles must be (S,3) and non-empty")
    if tris.min() < 0 or tris.max() >= verts.shape[1]:
        raise RenderError(f"{subject_dir}: surface triangles do not index the vertices")
    return verts, tris, stamps, src_frames


def project(verts, view):
    """Return (F,N,2) image-plane coords (+y down) and (F,N) depth (larger = farther)."""
    if view == "front":
        xy = verts[..., [0, 1]]
        depth = verts[..., 2]
    else:  # side: horizontal axis is z (forward), depth is -x (larger x is nearer)
        xy = verts[..., [2, 1]]
        depth = -verts[..., 0]
    return xy, depth


def fixed_transform(xy_all, width, height, margin=0.08):
    """One scale and offset for all frames so the camera never moves."""
    lo = xy_all.reshape(-1, 2).min(axis=0)
    hi = xy_all.reshape(-1, 2).max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = (1.0 - 2 * margin) * min(width, height) / span.max()
    center_world = (lo + hi) / 2.0
    return scale, center_world


def to_pixels(xy, scale, center_world, width, height):
    px = (xy - center_world) * scale
    x = px[..., 0] + width / 2.0
    y = height / 2.0 + px[..., 1]  # +y (camera y-down) maps to larger image y
    return np.stack([x, y], axis=-1)


def timeline(stamps, fps):
    t0, t1 = stamps[0], stamps[-1]
    count = int(np.floor((t1 - t0) * fps + 1e-9)) + 1
    times = t0 + np.arange(count) / fps
    # Previous reconstructed mesh: largest index with stamp <= t (no look-ahead).
    idx = np.searchsorted(stamps, times, side="right") - 1
    return np.clip(idx, 0, len(stamps) - 1), times


def render_frame(canvas, pts2d, tris, depth, normals_world):
    tri_depth = depth[tris].mean(axis=1)
    order = np.argsort(-tri_depth, kind="stable")  # far first, near last
    shade = 0.35 + 0.65 * np.abs(normals_world @ LIGHT)
    for t in order:
        poly = np.round(pts2d[tris[t]]).astype(np.int32)
        color = np.clip(BASE_COLOR * shade[t], 0, 255)
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.fillConvexPoly(canvas, poly, bgr)
        cv2.polylines(canvas, [poly], True, EDGE_COLOR, 1, cv2.LINE_AA)


def face_normals(verts, tris):
    a = verts[:, tris[:, 0]]
    b = verts[:, tris[:, 1]]
    c = verts[:, tris[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norm, 1e-12)


def render(root, out_path, subject, view, fps, size):
    if cv2 is None:
        raise RenderError("opencv (cv2) is required for rendering; pip install opencv-python")
    if not np.isfinite(fps) or fps <= 0:
        raise RenderError("--fps must be a finite positive number")
    if size < 128:
        raise RenderError("--size must be at least 128 pixels")
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as e:
        raise RenderError(f"cannot read {manifest_path}: {e}") from e
    subjects = manifest.get("subjects") or []
    if not subjects:
        raise RenderError("manifest lists no subjects")
    if subject is None:
        if len(subjects) != 1:
            names = [s.get("directory") for s in subjects]
            raise RenderError(f"multiple subjects {names}; choose one with --subject DIRECTORY")
        chosen = subjects[0]
    else:
        matches = [s for s in subjects if s.get("directory") == subject]
        if not matches:
            raise RenderError(f"no subject directory named {subject!r}")
        chosen = matches[0]
    verts, tris, stamps, src_frames = load_subject(root, chosen["directory"])
    width = height = int(size)
    xy_all, depth_all = project(verts, view)
    scale, center = fixed_transform(xy_all, width, height)
    idx, times = timeline(stamps, fps)
    normals = face_normals(verts, tris)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RenderError(f"cannot open video writer for {out_path}")
    try:
        cache = {}
        for k, fi in enumerate(idx):
            if fi not in cache:
                cache = {fi: to_pixels(xy_all[fi], scale, center, width, height)}
            pts = cache[fi]
            canvas = np.full((height, width, 3), 245, np.uint8)
            render_frame(canvas, pts, tris, depth_all[fi], normals[fi])
            cv2.putText(canvas, LABEL, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"view={view}  t={times[k]:.3f}s  src_frame={int(src_frames[fi])}",
                        (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    return len(idx), out_path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("reconstruction", help="directory written by reconstruct_twin.py (contains manifest.json)")
    p.add_argument("--output", required=True, help="output .mp4 path")
    p.add_argument("--subject", default=None, help="subject directory name from the manifest")
    p.add_argument("--view", choices=["front", "side"], default="front")
    p.add_argument("--fps", type=float, default=30.0, help="output frame rate (resampled, default 30)")
    p.add_argument("--size", type=int, default=720, help="square frame size in pixels")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        n, out = render(Path(args.reconstruction), args.output, args.subject, args.view, args.fps, args.size)
    except RenderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {out} ({n} frames at {args.fps:g} fps, view={args.view})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
