#!/usr/bin/env python3
"""Recording-driven inverse FEM reconstruction (see docs/inverse-fem-contract.md).

Reads a .twin ZIP (or legacy twin.json), recomputes skeletal forward kinematics
by joint name, and fits a corotated tetrahedral FEM scaffold per frame to the
*inherited skeletal estimates*. The source recording is never modified.
Needs NumPy + SciPy only (solver lives in scripts/twin_fem.py).
"""
import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

UNITS_NOTE = "Lengths are the recording's own units (mediapipe_estimated_meters when present)."
BARRIER_WEIGHT = 1.0  # declared anti-inversion prior (IPC-style log barrier on det F), x mu
BARRIER_MIN_DET = 0.05
CONF_FLOOR = 0.05  # documented numerical-anchoring prior for unsupported vertices
UPPER_BONE_NAMES = (
    "thorax", "nose", "head", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)


class TwinError(Exception):
    pass


# ----------------------------------------------------------------- loading
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


# ----------------------------------------------------------------- kinematics
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


# ----------------------------------------------------------------- scaffold
def select_active_bones(names, parents, person, rec):
    """Return (bone child indices, scope string)."""
    idx = {n: i for i, n in enumerate(names)}
    scope = person.get("body_scope")
    upper = scope == "upper_body" or (scope is None and bool(rec.get("upper_body")))
    if upper:
        # 'thorax' bone = spine->thorax torso scaffold; pelvis/legs are unobserved, excluded.
        bones = sorted(idx[n] for n in UPPER_BONE_NAMES if n in idx)
        return bones, "upper_body"
    return [i for i in range(len(names)) if parents[i] >= 0], "full_body"


def bone_radii(names, parents, offsets, person, bones):
    """Declared radius priors (per child joint), proportional to measured dimensions."""
    seg = person.get("segment_lengths") or {}
    shoulder = float(person.get("shoulder_width") or 0.0)
    head_r = float(person.get("head_radius") or 0.0)
    J = len(names)
    radii = np.zeros(J)
    for i in bones:
        n = names[i]
        length = float(np.linalg.norm(offsets[i]))
        if n in ("spine", "thorax"):
            r = 0.30 * shoulder if shoulder > 0 else 0.15 * length
        elif n in ("head", "nose", "head_pivot"):
            r = 0.75 * head_r if head_r > 0 else 0.3 * length
        elif "shoulder" in n:
            r = 0.30 * float(seg.get("upper_arm", length)) if "upper_arm" in seg else 0.12 * length
        elif "elbow" in n:
            r = 0.16 * float(seg.get("upper_arm", length))
        elif "wrist" in n:
            r = 0.12 * float(seg.get("forearm", length))
        elif "hip" in n or "knee" in n:
            r = 0.16 * length
        elif "ankle" in n:
            r = 0.12 * length
        else:
            r = 0.15 * length
        radii[i] = max(r, 1e-3)
    return radii


def weld_bones(names, parents, offsets, bones):
    """Drop zero-length bones (e.g. head_pivot) from the scaffold; FK keeps them."""
    return [i for i in bones if float(np.linalg.norm(offsets[i])) > 1e-9]


# ----------------------------------------------------------------- output
def write_obj(path, vertices, triangles, comment):
    with open(path, "w") as f:
        f.write(f"# {comment}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in triangles:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def safe_name(segment, index):
    return f"subject_{index:02d}_segment_{segment}"


def fit_silhouette_radii(twin, subject, args, names, parents, offsets, bones, radii, frames, upper_body):
    """Opt-in arm-width calibration from the bundle's recording.mp4 masks (before meshing)."""
    try:
        import twin_silhouette
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import twin_silhouette
    source = getattr(args, "_source", None)
    if source is None:
        raise TwinError("--fit-silhouette needs the input path")
    by_frame = {f["frame"]: f for f in subject.get("frames") or []}

    def geometry(frame):
        local, _ = frame_rotations(frame, names)
        posed, _ = forward_kinematics(parents, offsets, local, quat_to_matrix(frame["root_orientation_xyzw"]),
                                      np.asarray(frame["root_position"], dtype=float))
        return posed, frame.get("focal_length_px")

    priors = {names[b]: float(radii[b]) for b in bones if names[b] in twin_silhouette.ARM_SEGMENTS}
    subject_all = dict(subject, frames=list(by_frame.values()))  # all frames, not just strided ones
    try:
        return twin_silhouette.fit_arm_radii(source, subject_all, names, priors, upper_body,
                                             args.segmentation_model, geometry,
                                             timestamps=(twin.get("recording") or {}).get("timestamps_seconds"),
                                             trim_start=(twin.get("recording") or {}).get("trim_start_seconds") or 0.0,
                                             frame_count=(twin.get("recording") or {}).get("frame_count"))
    except twin_silhouette.SilhouetteError as e:
        raise TwinError(str(e))


def reconstruct_subject(twin, subject, index, args, out_dir, twin_fem):
    names, parents = validate_hierarchy(twin)
    J = len(names)
    rec = twin.get("recording") or {}
    person = subject.get("person") or {}
    offsets = np.asarray(person.get("rest_offsets"), dtype=float)
    if offsets.shape != (J, 3) or not _finite(offsets):
        raise TwinError("person.rest_offsets must be finite (J,3)")
    frames = subject.get("frames") or []
    if not frames:
        raise TwinError("Subject has no frames")
    for k in ("frame",):
        if any(not isinstance(f.get(k), int) or isinstance(f.get(k), bool) or f.get(k) < 0 for f in frames):
            raise TwinError("Every frame needs a non-negative integer 'frame' index")
    if len({f["frame"] for f in frames}) != len(frames):
        raise TwinError("Duplicate frame indices in subject")
    frames = sorted(frames, key=lambda f: f["frame"])
    frames = frames[:: max(1, args.stride)]
    if args.max_frames:
        frames = frames[: args.max_frames]
    frame_idx = np.array([f["frame"] for f in frames])
    stamps, timing_source = timestamps_for(twin, frame_idx)

    bones, scope = select_active_bones(names, parents, person, rec)
    bones = weld_bones(names, parents, offsets, bones)
    if not bones:
        raise TwinError("No active bones for scaffold")
    radii = bone_radii(names, parents, offsets, person, bones)

    # Rest pose: identity FK.
    eye = np.tile(np.eye(3), (J, 1, 1))
    rest_joints, _ = forward_kinematics(parents, offsets, eye, np.eye(3), np.zeros(3))
    prior_radii = radii.copy()
    silhouette = None
    if getattr(args, "fit_silhouette", False):
        silhouette = fit_silhouette_radii(twin, subject, args, names, parents, offsets, bones,
                                          radii, frames, scope == "upper_body")
        for child, r in silhouette["radii"].items():
            radii[names.index(child)] = r
    mesh = twin_fem.build_mesh(rest_joints, parents, bones, radii, spacing=args.spacing)
    rest_v = np.asarray(mesh["vertices"], dtype=float)
    weights = np.asarray(mesh["weights"], dtype=float)

    sdir = out_dir / safe_name(subject.get("subject_segment"), index)
    sdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sdir / "mesh.npz", rest_vertices=rest_v, tetrahedra=mesh["tetrahedra"],
        surface_triangles=mesh["surface_triangles"], weights=weights,
    )
    write_obj(sdir / "rest.obj", rest_v, mesh["surface_triangles"], "rest scaffold (assumed geometry)")

    kept, verts, joints_out, confs, per_frame, skipped = [], [], [], [], [], []
    prev_vertices = None
    for f, t in zip(frames, stamps):
        fi = f["frame"]
        try:
            ro = f.get("root_orientation_xyzw")
            rp = np.asarray(f.get("root_position"), dtype=float)
            if rp.shape != (3,) or not _finite(rp):
                raise TwinError("invalid root_position")
            local, conf = frame_rotations(f, names)
            posed, world = forward_kinematics(parents, offsets, local, quat_to_matrix(ro), rp)
            # Bone endpoint confidence: mean of the bone's parent and child joint confidence.
            bone_conf = np.zeros(J)
            for b in bones:
                bone_conf[b] = 0.5 * (conf[b] + conf[parents[b]])
            if not np.any(bone_conf > 0):
                raise TwinError("all active-joint confidences are zero (unsupported frame)")
            targets = twin_fem.skin_targets(mesh, rest_joints, posed, world)
            vconf = weights @ bone_conf
            tw = CONF_FLOOR + (1.0 - CONF_FLOOR) * np.clip(vconf, 0.0, 1.0)
            solve_kwargs = dict(target_weights=tw, young=args.young, poisson=args.poisson,
                                data_weight=args.data_weight, max_iterations=args.max_iterations,
                                barrier_weight=BARRIER_WEIGHT, min_det=BARRIER_MIN_DET)
            if prev_vertices is not None:
                solve_kwargs["initial_vertices"] = prev_vertices  # warm start from previous solved frame
            res = twin_fem.solve_frame(mesh, targets, **solve_kwargs)
            v = np.asarray(res["vertices"], dtype=float)
            if not _finite(v):
                raise TwinError("solver returned non-finite vertices")
        except (TwinError, ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
            skipped.append({"frame": fi, "reason": str(e)})
            print(f"  skip frame {fi}: {e}", file=sys.stderr)
            continue
        kept.append((fi, t))
        prev_vertices = v
        verts.append(v)
        joints_out.append(posed)
        confs.append(conf)
        diag = dict(res["diagnostics"])
        diag = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in diag.items()}
        diag["valid"] = bool(diag.get("valid", True) and diag.get("inverted_elements", 0) == 0)
        diag.update({
            "frame": fi, "timestamp_seconds": float(t),
            "mean_target_confidence": float(vconf.mean()),
            "upstream_reprojection_rmse_px": f.get("reprojection_rmse_px"),
            "upstream_root_depth_observed": f.get("root_depth_observed"),
        })
        per_frame.append(diag)
        if len(kept) == 1 or len(kept) % 20 == 0 or len(kept) == len(frames):
            print(f"  frame {fi}: {len(kept)}/{len(frames)}, "
                  f"target RMSE={diag.get('target_rmse', float('nan')):.5f}, "
                  f"valid={diag['valid']}, converged={diag.get('converged')}", flush=True)
        if len(kept) == 1:
            write_obj(sdir / "frame_000000.obj", v, mesh["surface_triangles"], f"posed frame {fi}")

    if not kept:
        raise TwinError(f"No frame could be reconstructed ({len(skipped)} skipped)")
    np.savez_compressed(
        sdir / "motion.npz",
        frame_indices=np.array([k[0] for k in kept]),
        timestamps_seconds=np.array([k[1] for k in kept]),
        vertices=np.stack(verts), joint_positions=np.stack(joints_out),
        joint_confidence=np.stack(confs), joint_names=np.array(names),
    )
    def agg(key, fn):
        vals = [d[key] for d in per_frame if isinstance(d.get(key), (int, float))]
        return float(fn(vals)) if vals else None
    summary = {
        "frames_requested": len(frames), "frames_reconstructed": len(kept),
        "frames_skipped": skipped,
        "converged_frames": sum(1 for d in per_frame if d.get("converged") is True),
        "invalid_frames": sum(1 for d in per_frame if not d["valid"]),
        "mean_target_rmse": agg("target_rmse", np.mean),
        "max_target_rmse": agg("target_rmse", np.max),
        "min_det_f_overall": agg("min_det_f", np.min),
        "inverted_elements_total": int(sum(d.get("inverted_elements", 0) or 0 for d in per_frame)),
        "upstream_reprojection_rmse_px_mean": agg("upstream_reprojection_rmse_px", np.mean),
    }
    with open(sdir / "diagnostics.json", "w") as f:
        json.dump({"aggregate": summary, "frames": per_frame}, f, indent=1)
    info = {
        "subject_segment": subject.get("subject_segment"), "directory": sdir.name,
        "units": person.get("units", "unspecified_recording_units"),
        "scope": scope, "bones": [names[b] for b in bones],
        "bone_radii_prior": {names[b]: float(prior_radii[b]) for b in bones},
        "bone_radii_used": {names[b]: float(radii[b]) for b in bones},
        "silhouette_fit": silhouette["evidence"] if silhouette else None,
        "vertices": int(len(rest_v)), "tetrahedra": int(len(mesh["tetrahedra"])),
        "timing": timing_source, "summary": summary,
    }
    print(f"{sdir.name}: {len(kept)}/{len(frames)} frames, scope={scope}, "
          f"V={len(rest_v)} T={len(mesh['tetrahedra'])}, "
          f"mean target RMSE={summary['mean_target_rmse']}, inverted={summary['inverted_elements_total']}")
    return info


def build_parser():
    p = argparse.ArgumentParser(description="Inverse FEM displacement reconstruction from a .twin recording.")
    p.add_argument("input", help=".twin ZIP or legacy twin.json")
    p.add_argument("--output", default="output/fem-twin", help="output directory")
    p.add_argument("--subject", type=int, default=None, help="subject_segment to process (default all)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--spacing", type=float, default=0.06)
    p.add_argument("--young", type=float, default=1.0)
    p.add_argument("--poisson", type=float, default=0.35)
    p.add_argument("--data-weight", type=float, default=100.0)
    p.add_argument("--max-iterations", type=int, default=60)
    p.add_argument("--fit-silhouette", action="store_true",
                   help="calibrate visible arm radii from recording.mp4 segmentation (needs cv2)")
    p.add_argument("--segmentation-model", default="models/mediapipe-pose.onnx")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    fin = math.isfinite
    if (args.stride < 1 or not (fin(args.spacing) and args.spacing > 0) or args.max_iterations < 1
            or not (fin(args.young) and args.young > 0) or not (fin(args.data_weight) and args.data_weight >= 0)
            or not (fin(args.poisson) and 0 <= args.poisson <= 0.49)
            or (args.max_frames is not None and args.max_frames < 1)):
        print("error: need stride>=1, spacing>0, young>0, 0<=poisson<=0.49, data-weight>=0, "
              "max-iterations>=1, max-frames>=1 (when given), all finite", file=sys.stderr)
        return 2
    src = Path(args.input).resolve()
    out = Path(args.output).resolve()
    args._source = src
    if out == src or out in src.parents:
        print("error: refusing to write over/into the input location", file=sys.stderr)
        return 2
    try:
        import twin_fem  # solver module in the same directory
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            import twin_fem
        except ImportError as e:
            print(f"error: cannot import twin_fem: {e}", file=sys.stderr)
            return 2
    try:
        twin, sha = load_twin(src)
        validate_hierarchy(twin)
        subjects = twin.get("subjects")
        if not isinstance(subjects, list) or not subjects:
            raise TwinError("Recording has no subjects")
        out.mkdir(parents=True, exist_ok=True)
        infos, failures = [], []
        for i, s in enumerate(subjects):
            if args.subject is not None and s.get("subject_segment") != args.subject:
                continue
            try:
                infos.append(reconstruct_subject(twin, s, i, args, out, twin_fem))
            except (TwinError, ValueError) as e:  # ValueError: mesh/solver rejected input
                failures.append({"subject_segment": s.get("subject_segment"), "error": str(e)})
                print(f"subject {s.get('subject_segment')}: FAILED: {e}", file=sys.stderr)
        manifest = {
            "tool": "reconstruct_twin.py", "input": str(src), "input_sha256": sha,
            "schema_version": twin.get("schema_version"), "units": UNITS_NOTE,
            "coordinate_axes": twin.get("coordinate_axes", "camera_x_right_y_down_z_forward"),
            "quaternion_convention": twin.get("quaternion_convention", "xyzw_active_parent_local"),
            "parameters": {k: getattr(args, k) for k in (
                "subject", "stride", "max_frames", "spacing", "young", "poisson",
                "data_weight", "max_iterations", "fit_silhouette", "segmentation_model")},
            "barrier_prior": {"barrier_weight": BARRIER_WEIGHT, "min_det": BARRIER_MIN_DET,
                              "note": "anti-inversion regularization prior, not a measured quantity"},
            "assumptions": [
                "Targets are INHERITED skeletal/skinning estimates from the recording, not measurements of tissue or video.",
                "Geometry is an assumed union of bone capsules; radii are declared priors proportional to measured segment dimensions unless --fit-silhouette replaces supported arm radii with visible clothed-silhouette widths (not tissue/anatomy).",
                "Young's modulus and Poisson ratio are priors; no tissue stiffness, force or stress is identified.",
                f"Per-vertex target weight = {CONF_FLOOR} + {1 - CONF_FLOOR} * bone-endpoint confidence; the {CONF_FLOOR} floor is a numerical anchoring prior.",
                "Low residual does not imply anatomical accuracy. Upstream reprojection_rmse_px is reported separately and is not improved here.",
                "Upper-body recordings exclude unobserved legs from the mesh.",
            ],
            "subjects": infos, "failures": failures,
        }
        with open(out / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)
    except TwinError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {out}; {len(infos)} subject(s), {len(failures)} failure(s)")
    incomplete = any(s["summary"]["invalid_frames"] or s["summary"]["frames_skipped"] for s in infos)
    if incomplete:
        print("WARNING: invalid or skipped frames; inspect diagnostics.json", file=sys.stderr)
    return 1 if failures or not infos or incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
