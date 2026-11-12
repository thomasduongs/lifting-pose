#!/usr/bin/env python3
"""Headless anatomical renderer for the biomechanical twin (no window, no GL context).

Reads a MuJoCo ``model.mjb`` and a ``motion.npz`` written by ``fit_biomech.py`` from
one subject directory, and renders the sourced reference anatomy frame by frame:

* bones / solid tissue: the upstream mesh geoms, transformed by the engine's own
  ``geom_xpos``/``geom_xmat`` and painter-sorted per geom (flat colour);
* muscle-tendon paths (red polylines): the actuated spatial tendons, drawn from the
  engine's solved path (site via-points plus wrap points, ``data.wrap_xpos``);
* ligament routes (orange polylines): the eight upstream glenohumeral/coracohumeral
  routes. Their stiffness is 0 unless the model was exported with it enabled;
* fitted joint-centre markers (green dots) from ``motion.npz``;
* the lower body (pelvis, legs, feet) is labeled UNOBSERVED: the recording does not
  observe it, so it is held at the reference standing pose.

Limitations: sphere/cylinder wrap arcs are drawn as chords between the engine's
solved tangent points. The polyline length differs from the engine's ``ten_length``
by up to about 5 mm at standing for some routes; DELT1 and BIClong match to <1e-6 m
(see docs/biomechanics.md). Output is an
orthographic projection, not a rendered camera image. Activation is not shown
because no activation is simulated by this renderer.

Usage:
    .venv-biomech/bin/python scripts/render_biomech.py output/biomech-twin/<subject> \
        --output output/biomech-twin/<subject>/front.mp4 [--view front|side] [--fps 30] \
        [--max-frames N] [--size 720] [--snapshot frame0.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import mujoco
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biomech_model as bm  # noqa: E402  (ligament names; same pinned MuJoCo)

UP = np.array([0.0, 0.0, 1.0])  # world z is up in the MyoSim model
# Subject faces -y in the reference model; left = +x. Camera forward vectors:
VIEWS = {
    "front": np.array([0.0, 1.0, 0.0]),   # viewer on -y side, subject faces the viewer
    "side": np.array([-1.0, 0.0, 0.0]),   # viewer on +x (subject's left) side
}

LOWER_BODY = {
    "pelvis", "myolegs_root",
    "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r", "patella_r",
    "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l", "patella_l",
}

# BGR colours
BG = (247, 247, 247)
BONE = (176, 170, 164)
BONE_LOWER = (206, 206, 206)
MUSCLE = (52, 52, 196)
LIGAMENT = (0, 150, 255)
MARKER = (40, 160, 40)
TEXT = (30, 30, 30)
UNOBSERVED = (120, 90, 40)

MARKER_SHOW = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "thorax", "head")


class RenderError(Exception):
    pass


def view_basis(view: str):
    """Return (right, up, forward) unit vectors for an orthographic view."""
    if view not in VIEWS:
        raise RenderError(f"unknown view {view!r}; choose from {sorted(VIEWS)}")
    fwd = VIEWS[view]
    right = np.cross(fwd, UP)  # viewer's right-hand direction
    return right, UP.copy(), fwd


class Camera:
    """Orthographic camera; ``project`` returns pixel (x, y) and depth along the view."""

    def __init__(self, view: str, points: np.ndarray, size: int):
        self.right, self.up, self.fwd = view_basis(view)
        self.size = size
        u = points @ self.right
        v = points @ self.up
        lo_u, hi_u = float(u.min()), float(u.max())
        lo_v, hi_v = float(v.min()), float(v.max())
        header = min(115.0, 0.16 * size)
        bottom = min(35.0, 0.07 * size)
        self.scale = min(0.9 * size / max(hi_u - lo_u, 1e-6),
                         (size - header - bottom) / max(hi_v - lo_v, 1e-6))
        self.y_center = (header + size - bottom) / 2
        self.cu = 0.5 * (lo_u + hi_u)
        self.cv = 0.5 * (lo_v + hi_v)

    def project(self, pts: np.ndarray) -> np.ndarray:
        x = (pts @ self.right - self.cu) * self.scale + self.size / 2
        y = self.y_center - (pts @ self.up - self.cv) * self.scale
        return np.stack([x, y], axis=-1)

    def depth(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.fwd


def mesh_geoms(model):
    """Visual mesh geoms as (geom_id, body_name, local_vertices, faces)."""
    out = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_group[g] > 2:  # MuJoCo's default visual groups are 0-2
            continue
        mid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float64)
        faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int64)
        if faces.size and faces.max() >= vn:
            raise RenderError(f"mesh {mid} face index exceeds its vertex count")
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        out.append((g, body, verts, faces))
    return out


def tendon_sites(model, t):
    """Site ids of the via-points of tendon t, in path order (sites only)."""
    adr, num = model.tendon_adr[t], model.tendon_num[t]
    return [int(model.wrap_objid[w]) for w in range(adr, adr + num)
            if model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_SITE]


def tendon_path(model, data, t):
    """Native solved path of tendon t after mj_forward, world xyz (k, 3).

    The engine stores wrap points as (nwrap, 6) rows; the path of tendon t is the
    flattened 3-vectors ``wrap_xpos.reshape(-1, 3)[a:a+n]`` with
    ``a = ten_wrapadr[t]`` and ``n = ten_wrapnum[t]``. Site via-points are part of
    this path, and sphere/cylinder tangent points are solved by the engine. Falls back
    to the site positions when the engine path is shorter than two points."""
    a, n = int(data.ten_wrapadr[t]), int(data.ten_wrapnum[t])
    if n >= 2:
        pts = data.wrap_xpos.reshape(-1, 3)[a:a + n]
        if np.all(np.isfinite(pts)):
            return np.asarray(pts, dtype=np.float64)
    sites = tendon_sites(model, t)
    return np.array([data.site_xpos[s] for s in sites], dtype=np.float64).reshape(-1, 3)


def has_geometric_wrap(model, t):
    adr, num = model.tendon_adr[t], model.tendon_num[t]
    geometric = (int(mujoco.mjtWrap.mjWRAP_SPHERE), int(mujoco.mjtWrap.mjWRAP_CYLINDER))
    return any(int(model.wrap_type[w]) in geometric for w in range(adr, adr + num))


def tendon_roles(model):
    """(muscle_tendon_ids, ligament_tendon_ids) by actuator and by upstream name."""
    muscle = set()
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON:
            muscle.add(int(model.actuator_trnid[a][0]))
    lig = set()
    for name in bm.LIGAMENT_NAMES:
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
        if tid >= 0:
            lig.add(tid)
    muscle -= lig
    return sorted(muscle), sorted(lig)


def load_run(run_dir: Path):
    model_path = run_dir / "model.mjb"
    motion_path = run_dir / "motion.npz"
    if not model_path.is_file() or not motion_path.is_file():
        raise RenderError(f"{run_dir} needs model.mjb and motion.npz (a fitted subject directory)")
    try:
        model = mujoco.MjModel.from_binary_path(str(model_path))
    except Exception as exc:  # version-specific binary
        raise RenderError(f"cannot load {model_path} with MuJoCo {mujoco.__version__}: {exc}") from exc
    motion = np.load(motion_path, allow_pickle=False)
    qpos = np.asarray(motion["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise RenderError(f"qpos shape {qpos.shape} does not match model nq={model.nq}")
    stamps = np.asarray(motion["timestamps_seconds"], dtype=np.float64)
    markers = np.asarray(motion["marker_positions"], dtype=np.float64)
    names = [str(n) for n in motion["marker_names"]]
    return model, qpos, stamps, markers, names


def frame_points(model, data, muscles, ligaments, qpos_frames):
    """Mesh vertices (subsampled) and tendon sites over all frames, for framing the camera."""
    pts = []
    meshes = mesh_geoms(model)
    for f, q in enumerate(qpos_frames):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        if f == 0:
            for g, _, verts, _ in meshes:
                w = data.geom_xpos[g] + verts[::8] @ data.geom_xmat[g].reshape(3, 3).T
                pts.append(w)
        for t in muscles + ligaments:
            pts.append(tendon_path(model, data, t))
    return np.concatenate(pts, axis=0)


def draw_frame(img, model, data, cam, meshes, muscles, ligaments, marker_xyz, marker_names):
    mujoco.mj_forward(model, data)  # qpos was set by the caller
    # Bones / solid tissue: every triangle is filled separately, far to near (global painter
    # sort on depth). One fillPoly over all triangles is not used: overlapping polygons in
    # one call do not composite reliably.
    tri_px, tri_depth, tri_col = [], [], []
    for g, body, verts, faces in meshes:
        world = data.geom_xpos[g] + verts @ data.geom_xmat[g].reshape(3, 3).T
        tri_px.append(cam.project(world)[faces])                       # (nf, 3, 2)
        tri_depth.append(cam.depth(world)[faces].mean(axis=1))         # (nf,)
        col = BONE_LOWER if body in LOWER_BODY else BONE
        tri_col.append(np.broadcast_to(np.array(col, dtype=np.uint8), (faces.shape[0], 3)))
    if tri_px:
        px = np.concatenate(tri_px)
        depth = np.concatenate(tri_depth)
        cols = np.concatenate(tri_col)
        order = np.argsort(-depth)  # far first
        px = np.round(px[order]).astype(np.int32)
        cols = cols[order]
        for k in range(px.shape[0]):
            cv2.fillConvexPoly(img, px[k], (int(cols[k, 0]), int(cols[k, 1]), int(cols[k, 2])))

    # Muscle-tendon paths and ligament routes: native solved path (wrap points included).
    for t in muscles:
        pts = cam.project(tendon_path(model, data, t))
        if pts.shape[0] < 2:
            continue
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, MUSCLE, 1, cv2.LINE_AA)
    for t in ligaments:
        pts = cam.project(tendon_path(model, data, t))
        if pts.shape[0] < 2:
            continue
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, LIGAMENT, 3, cv2.LINE_AA)

    # Fitted joint-centre markers.
    for name, xyz in zip(marker_names, marker_xyz):
        if name in MARKER_SHOW:
            x, y = np.round(cam.project(xyz[None, :])[0]).astype(int)
            cv2.circle(img, (int(x), int(y)), 4, MARKER, -1, cv2.LINE_AA)


def annotate(img, size, t_seconds, frame_idx, n_frames, view, ligament_stiffness, enabled_routes=0):
    lines = [
        "Reference anatomy (MyoSim/MuJoCo), not subject-specific",
        f"view: {view}   frame {frame_idx + 1}/{n_frames}   t = {t_seconds:0.3f} s",
        "bone (grey, solid mesh)   muscle-tendon path (red)   ligament route (orange)",
        ("ligament routes: zero stiffness in this model (no force)"
         if not ligament_stiffness else f"ligament springs: {enabled_routes}/8 routes enabled"),
        "green dots: fitted joint centres",
    ]
    y = 22
    for line in lines:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
        y += 20
    cv2.putText(img, "LOWER BODY (grey): UNOBSERVED in recording, reference standing pose, not fitted",
                (12, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, UNOBSERVED, 1, cv2.LINE_AA)


def presentation_frames(stamps, fps):
    """Previous-pose hold on a uniform output clock, including the final source pose."""
    stamps = np.asarray(stamps, dtype=float)
    if not np.isfinite(fps) or fps <= 0:
        raise RenderError("--fps must be finite and positive")
    if stamps.ndim != 1 or len(stamps) == 0 or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
        raise RenderError("timestamps must be finite and strictly increasing")
    count = int(np.ceil((stamps[-1] - stamps[0]) * fps - 1e-9)) + 1
    times = stamps[0] + np.arange(count) / fps
    indices = np.clip(np.searchsorted(stamps, times, side="right") - 1, 0, len(stamps) - 1)
    return indices, times


def run(run_dir: Path, output: Path, view: str, fps: float, max_frames: int, size: int,
        snapshot: Path | None):
    if size < 128:
        raise RenderError("--size must be at least 128")
    if not np.isfinite(fps) or fps <= 0:
        raise RenderError("--fps must be finite and positive")
    model, qpos, stamps, markers, marker_names = load_run(run_dir)
    n_frames = qpos.shape[0] if max_frames <= 0 else min(max_frames, qpos.shape[0])
    if n_frames < 1:
        raise RenderError("no frames to render")
    data = mujoco.MjData(model)
    muscles, ligaments = tendon_roles(model)
    meshes = mesh_geoms(model)
    frames = qpos[:n_frames]
    if stamps.shape != (len(qpos),) or not np.isfinite(qpos).all():
        raise RenderError("invalid qpos or timestamp dimensions")
    presentation_indices, presentation_times = presentation_frames(stamps[:n_frames], fps)
    points = frame_points(model, data, muscles, ligaments, frames)
    cam = Camera(view, points, size)
    ligament_stiffness = float(np.max(model.tendon_stiffness[ligaments])) if ligaments else 0.0
    enabled_routes = int(np.count_nonzero(model.tendon_stiffness[ligaments] > 0))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise RenderError(f"cannot open video writer for {output}")
    try:
        cached_index, cached_img = -1, None
        for k, (f, t) in enumerate(zip(presentation_indices, presentation_times)):
            if f != cached_index:
                data.qpos[:] = qpos[f]
                cached_img = np.full((size, size, 3), BG, dtype=np.uint8)
                draw_frame(cached_img, model, data, cam, meshes, muscles, ligaments,
                           markers[f] if markers.size else np.zeros((0, 3)), marker_names)
                cached_index = f
            img = cached_img.copy()
            annotate(img, size, float(t), int(f), n_frames, view, ligament_stiffness, enabled_routes)
            if k == 0 and snapshot is not None:
                cv2.imwrite(str(snapshot), img)
            writer.write(img)
    finally:
        writer.release()
    return {"frames": len(presentation_indices), "source_frames": n_frames,
            "timing": "previous-pose hold on original timestamps; final source pose included",
            "output": str(output), "view": view,
            "muscle_tendons": len(muscles), "ligament_routes": len(ligaments),
            "wrapped_tendons_native_points": int(sum(has_geometric_wrap(model, t) for t in muscles + ligaments))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path, help="fitted subject directory with model.mjb and motion.npz")
    p.add_argument("--output", required=True, type=Path, help="output MP4 path")
    p.add_argument("--view", choices=sorted(VIEWS), default="front")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-frames", type=int, default=0, help="0 renders every frame")
    p.add_argument("--size", type=int, default=720, help="square frame size in pixels")
    p.add_argument("--snapshot", type=Path, default=None, help="also write the first frame as PNG")
    args = p.parse_args(argv)
    try:
        info = run(args.run_dir, args.output, args.view, args.fps, args.max_frames, args.size,
                   args.snapshot)
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {info['output']} ({info['frames']} frames, view={info['view']}, "
          f"muscle paths={info['muscle_tendons']}, ligament routes={info['ligament_routes']}, "
          f"wrapped paths from native wrap points={info['wrapped_tendons_native_points']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
