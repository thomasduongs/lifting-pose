#!/usr/bin/env python3
"""Export the source bone meshes and independent attachment fixtures to
``assets/biomech-lite/visual.json`` for the native Rust viewer (docs/rust-viewer-contract.md).

Reads the compiled mesh arrays straight from ``output/biomech-twin/model.mjb``. Vertices are the
compiled ``mesh_vert`` slices and faces the local indices from ``mesh_face`` slices (native mesh
indexing, all 104 meshes). Runtime world transform is ``geom_xpos + geom_xmat * vertex``; mesh
pos/quat are already compiled into the vertices and must not be applied again.

Reference samples (first, middle, last vertex of every visual mesh geom) are taken from native
MuJoCo kinematics at the standing keyframe, so the Rust side can verify its FK independently.

Run: ``.venv-biomech/bin/python scripts/export_biomech_visual.py [--output assets/biomech-lite/visual.json]``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco  # noqa: E402

import export_biomech_lite as ex  # noqa: E402  (sha256, load_native, standing_qpos, TWIN, ExportError)
import render_biomech as rb  # noqa: E402  (LOWER_BODY: the unobserved set, same as the renderer)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "assets" / "biomech-lite" / "visual.json"
SCHEMA_VERSION = 1


def mesh_geom_ids(m):
    """Visual mesh geoms in native order, using the same group filter as render_biomech.mesh_geoms."""
    return [g for g in range(m.ngeom)
            if m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH) and m.geom_group[g] <= 2]


def build_meshes(m):
    meshes = []
    for mid in range(m.nmesh):
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        verts = np.asarray(m.mesh_vert[va:va + vn], dtype=np.float64)
        faces = np.asarray(m.mesh_face[fa:fa + fn], dtype=np.int64)
        if not np.isfinite(verts).all():
            raise ex.ExportError("mesh %d has nonfinite vertices" % mid)
        if faces.size == 0 or faces.min() < 0 or faces.max() >= vn:
            raise ex.ExportError("mesh %d face indices out of range" % mid)
        meshes.append({"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or "",
                       "vertices": verts.tolist(), "faces": faces.tolist()})
    return meshes


def build_geoms(m, geom_ids):
    geoms = []
    for g in geom_ids:
        body = int(m.geom_bodyid[g])
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        geoms.append({"geom": int(g), "mesh": int(m.geom_dataid[g]), "body": body,
                      "rgba": [float(x) for x in m.geom_rgba[g]],
                      "unobserved": body_name in rb.LOWER_BODY})
    return geoms


def build_reference(m, geom_ids):
    q = ex.standing_qpos(m)
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    samples = []
    for g in geom_ids:
        mid = int(m.geom_dataid[g])
        vn = int(m.mesh_vertnum[mid])
        va = int(m.mesh_vertadr[mid])
        R = d.geom_xmat[g].reshape(3, 3)
        for v in sorted({0, vn // 2, vn - 1}):
            local = np.asarray(m.mesh_vert[va + v], dtype=np.float64)
            world = d.geom_xpos[g] + R @ local
            if not np.isfinite(world).all():
                raise ex.ExportError("nonfinite reference sample for geom %d" % g)
            samples.append({"geom": int(g), "vertex": int(v), "world_position": world.tolist()})
    return {"qpos": q.tolist(), "samples": samples}


def build_visual(mjb_path=ex.TWIN / "model.mjb"):
    m = ex.load_native(mjb_path)
    geom_ids = mesh_geom_ids(m)
    if m.nmesh != 104 or len(geom_ids) != 102:
        raise ex.ExportError("expected 104 meshes and 102 visual mesh geoms, got %d and %d"
                             % (m.nmesh, len(geom_ids)))
    return {
        "schema_version": SCHEMA_VERSION,
        "source_model_sha256": ex.sha256(mjb_path),
        "source": {"mujoco": metadata.version("mujoco"), "myo_sim": metadata.version("myo-sim"),
                   "note": "compiled mesh frames"},
        "meshes": build_meshes(m),
        "geoms": build_geoms(m, geom_ids),
        "reference": build_reference(m, geom_ids),
    }


def write_atomic(obj, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=out.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return out


def export_visual(out=DEFAULT_OUT, mjb_path=ex.TWIN / "model.mjb"):
    visual = build_visual(mjb_path)
    write_atomic(visual, out)
    return {"out": str(out), "meshes": len(visual["meshes"]), "geoms": len(visual["geoms"]),
            "reference_samples": len(visual["reference"]["samples"]),
            "unobserved_geoms": sum(g["unobserved"] for g in visual["geoms"]),
            "bytes": Path(out).stat().st_size, "source_model_sha256": visual["source_model_sha256"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    try:
        print(json.dumps(export_visual(a.output), indent=1))
    except ex.ExportError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
