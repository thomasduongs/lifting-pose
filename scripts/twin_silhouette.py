"""Conservative silhouette-width calibration of arm radii (docs/silhouette-fit-contract.md).

Measures the *clothed visible silhouette* half-width of upper arms / forearms from
the MediaPipe segmentation output on the verified aligned video, converts it to the
recording's inferred body units, and returns radii to use BEFORE meshing. This is
surface evidence only: it does not refit poses, reduce landmark error or measure
tissue. Optional dependency: cv2 (the FEM core stays NumPy/SciPy only).
"""
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np

CROP = 256
UPPER_BODY_CROP_FACTOR = 2.0
MAX_SAMPLE_FRAMES = 24
FRACTIONS = (0.25, 0.375, 0.5, 0.625, 0.75)
MIN_KEYPOINT_SCORE = 0.7
MIN_PRESENCE = 0.7
MIN_SEGMENT_PX = 30.0
MIN_WIDTH_CROP_PX = 2.0
MAX_SYMMETRY = 2.0
RADIUS_RANGE = (0.015, 0.14)
PRIOR_RATIO_RANGE = (0.4, 2.5)
MIN_SAMPLES_PER_FRAME = 3
MIN_SUPPORT_FRAMES = 3
VIDEOS = ("source.mp4", "recording.mp4")  # source.mp4 = raw full recording; recording.mp4 = trimmed/resampled
# child joint -> (parent keypoint, child keypoint, label)
ARM_SEGMENTS = {
    "left_elbow": ("left_shoulder", "left_elbow", "upper_arm"),
    "left_wrist": ("left_elbow", "left_wrist", "forearm"),
    "right_elbow": ("right_shoulder", "right_elbow", "upper_arm"),
    "right_wrist": ("right_elbow", "right_wrist", "forearm"),
}


class SilhouetteError(Exception):
    pass


def _import_cv2():
    try:
        import cv2
    except ImportError as e:
        raise SilhouetteError(f"--fit-silhouette needs OpenCV (pip install -r scripts/requirements-fem-video.txt): {e}")
    return cv2


# ------------------------------------------------------------- geometry
def crop_geometry(bbox, upper_body):
    """Reproduce the Rust MediaPipe crop: returns (left, top, side)."""
    x, y, w, h = (float(bbox[k]) for k in ("x", "y", "width", "height"))
    if upper_body:
        side = h * UPPER_BODY_CROP_FACTOR
        return x + w / 2 - side / 2, y, side
    side = max(w, h) * 1.25
    return x + w / 2 - side / 2, y + h / 2 - side / 2, side


def preprocess(cv2, image_bgr, left, top, side):
    scale = CROP / side
    affine = np.array([[scale, 0, -left * scale], [0, scale, -top * scale]], dtype=np.float64)
    crop = cv2.warpAffine(image_bgr, affine, (CROP, CROP))  # black padding
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.reshape(1, CROP, CROP, 3)


def _point_segment_distance(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else min(max(float((p - a) @ ab) / denom, 0.0), 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _point_in_triangle_or_quad(p, poly):
    sign = 0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        s = 1 if cross > 0 else -1 if cross < 0 else 0
        if s != 0:
            if sign == 0:
                sign = s
            elif s != sign:
                return False
    return True


def torso_polygon(l_sh, r_sh):
    """Approximate torso region below the shoulders (image y down); hips are unobserved."""
    across = r_sh - l_sh
    width = float(np.linalg.norm(across))
    down = np.array([-across[1], across[0]])
    if down[1] < 0:
        down = -down
    n = np.linalg.norm(down)
    down = down / n if n > 0 else np.array([0.0, 1.0])
    return [l_sh, r_sh, r_sh + down * 1.5 * width, l_sh + down * 1.5 * width]


def trace_half_width(mask, center, direction, max_steps=256):
    """Walk from center along direction to the first background pixel.
    Returns half-width in crop px, or None if the crop edge is reached first."""
    step = 0.5
    for k in range(1, max_steps):
        s = k * step
        x, y = center + direction * s
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < CROP and 0 <= yi < CROP):
            return None
        if not mask[yi, xi]:
            return s - step / 2
    return None


def measure_segment(mask, a, b, others=(), torso=None):
    """Measure min half-width (crop px) at the five fixed positions along a->b.
    Returns (samples, rejects): samples are dicts, rejects are reason strings."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return [], ["degenerate_segment"]
    perp = np.array([-axis[1], axis[0]]) / length
    samples, rejects = [], []
    for t in FRACTIONS:
        c = a + t * axis
        xi, yi = int(round(c[0])), int(round(c[1]))
        if not (0 <= xi < CROP and 0 <= yi < CROP) or not mask[yi, xi]:
            rejects.append("center_background")
            continue
        d1 = trace_half_width(mask, c, perp)
        d2 = trace_half_width(mask, c, -perp)
        if d1 is None or d2 is None:
            rejects.append("truncated_at_crop_edge")
            continue
        if d1 + d2 < MIN_WIDTH_CROP_PX:
            rejects.append("too_narrow")
            continue
        if max(d1, d2) / max(min(d1, d2), 1e-6) > MAX_SYMMETRY:
            rejects.append("asymmetric")
            continue
        margin = 1.25 * max(d1, d2) + 2.0
        if any(_point_segment_distance(c, np.asarray(o0, float), np.asarray(o1, float)) < margin
               for o0, o1 in others):
            rejects.append("near_other_limb")
            continue
        if torso is not None:
            inside = _point_in_triangle_or_quad(c, torso)
            edges = [_point_segment_distance(c, torso[i], torso[(i + 1) % 4]) for i in (1, 2, 3)]
            if inside or min(edges) < margin:
                rejects.append("near_torso")
                continue
        samples.append({"t": t, "half_width_crop_px": min(d1, d2), "d1": d1, "d2": d2})
    return samples, rejects


# ------------------------------------------------------------- aggregation
def robust_aggregate(values, frames):
    """Median with MAD rejection. Returns (value or None, kept_frames, rejected_frames)."""
    v = np.asarray(values, float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    mad = max(mad, 0.05 * abs(med))
    keep = np.abs(v - med) <= 3.0 * mad
    kept = [f for f, k in zip(frames, keep) if k]
    rej = [f for f, k in zip(frames, keep) if not k]
    if keep.sum() == 0:
        return None, [], list(frames)
    return float(np.median(v[keep])), kept, rej


def aggregate_bone(per_frame, prior):
    """per_frame: {frame: [radius_units,...]} of accepted samples. Returns evidence dict."""
    frame_values = {}
    for f, vals in per_frame.items():
        if len(vals) >= MIN_SAMPLES_PER_FRAME:
            frame_values[f] = float(np.median(vals))
    out = {"prior_radius": float(prior), "support_frames_before_mad": sorted(frame_values)}
    if len(frame_values) < MIN_SUPPORT_FRAMES:
        out.update(source="prior", fitted_radius=None, radius=float(prior),
                   unsupported_reason=f"only {len(frame_values)} frame(s) with >= {MIN_SAMPLES_PER_FRAME} accepted "
                                      f"samples; need {MIN_SUPPORT_FRAMES}")
        return out
    frames = sorted(frame_values)
    value, kept, rej = robust_aggregate([frame_values[f] for f in frames], frames)
    if value is None or len(kept) < MIN_SUPPORT_FRAMES:
        out.update(source="prior", fitted_radius=None, radius=float(prior),
                   unsupported_reason=f"only {len(kept)} frame(s) survive MAD rejection; need {MIN_SUPPORT_FRAMES}",
                   mad_rejected_frames=rej)
        return out
    out.update(source="silhouette_median", fitted_radius=value, radius=value,
               support_frames=kept, mad_rejected_frames=rej, unsupported_reason=None)
    return out


# ------------------------------------------------------------- IO
def extract_members(path, directory):
    """Copy only the fixed members needed (poses.jsonl plus the available fixed-name videos)
    into directory. Never extracts arbitrary archive paths."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise SilhouetteError("--fit-silhouette needs a .twin bundle with video and poses.jsonl; "
                              "legacy twin.json has no saved video or observations")
    out = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if "poses.jsonl" not in names:
            raise SilhouetteError("Bundle is missing required member poses.jsonl for --fit-silhouette")
        if not any(v in names for v in VIDEOS):
            raise SilhouetteError("Bundle has neither source.mp4 nor recording.mp4 for --fit-silhouette")
        for m in ("poses.jsonl",) + tuple(v for v in VIDEOS if v in names):
            dest = Path(directory) / m  # fixed constant names only
            with z.open(m) as src, open(dest, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            out[m] = dest
    return out


def read_pose_log(path, subject_segment):
    """frame -> person dict (person_index 0, matching subject_segment when logged)."""
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for p in rec.get("people") or []:
                if p.get("person_index") != 0:
                    continue
                seg = (p.get("twin") or {}).get("subject_segment")
                if seg is not None and subject_segment is not None and seg != subject_segment:
                    continue
                result[rec["frame"]] = {"person": p, "width": rec.get("width"), "height": rec.get("height")}
    return result


def load_net(cv2, model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise SilhouetteError(f"Segmentation model not found: {model_path}")
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
    except cv2.error as e:
        raise SilhouetteError(f"Cannot load segmentation model {model_path}: {e}")
    return net, list(net.getUnconnectedOutLayersNames())


def run_segmentation(cv2, net, names, blob):
    net.setInput(blob)
    outs = net.forward(names)
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    seg = presence = None
    for o in outs:
        arr = np.asarray(o)
        if not np.all(np.isfinite(arr)):
            raise SilhouetteError("Non-finite segmentation model output")
        if arr.size == CROP * CROP:
            seg = arr.reshape(CROP, CROP)
        elif arr.size == 1:
            presence = float(arr.reshape(-1)[0])
    if seg is None or presence is None:
        raise SilhouetteError("Unexpected segmentation model outputs (need 65536 logits and 1 presence value)")
    return seg > 0.0, presence  # logits: > 0 <=> p > 0.5


def select_frames(available, limit=MAX_SAMPLE_FRAMES):
    available = sorted(available)
    if len(available) <= limit:
        return available
    idx = np.unique(np.round(np.linspace(0, len(available) - 1, limit)).astype(int))
    return [available[i] for i in idx]


def map_video_to_twin(video_frames, video_fps, timestamps, trim_start):
    """Map recording.mp4 frame k -> twin frame index by time (video time + trim_start vs twin
    timestamps). recording.mp4 may be a trimmed clip of a longer pose log, so index equality is
    not assumed; without timestamps, identity is used only when frame counts match."""
    if timestamps is None:
        raise SilhouetteError("Recording has no timestamps to align recording.mp4 with twin frames")
    ts = np.asarray(timestamps, float)
    if not (video_fps > 0 and video_frames > 0):
        raise SilhouetteError("recording.mp4 reports no frame rate/frame count")
    mapping = {}
    tol = 1.0 / video_fps
    for k in range(video_frames):
        t = float(trim_start) + k / video_fps
        j = int(np.argmin(np.abs(ts - t)))
        if abs(ts[j] - t) <= tol:
            mapping[k] = j
    return mapping, {"method": "nearest_timestamp", "trim_start_seconds": float(trim_start),
                     "video_fps": video_fps, "mapped_video_frames": len(mapping),
                     "note": "video frame k time = trim_start + k/fps; +-1 frame uncertainty"}


def choose_video(cv2, members, frame_count, pose_log, timestamps, trim_start):
    """Pick the video whose frame alignment with twin frame ids can be proven.

    Prefer source.mp4 (raw): only when its decoded-count equals twin recording.frame_count, the
    pose-log frame ids are in range and the pose-log image size matches the video; then video
    frame k == twin frame k (irregular capture timestamps do not imply a remap for raw source).
    Fall back to recording.mp4 mapped by timestamp. Returns (cap, mapping, alignment)."""
    tried = []
    if "source.mp4" in members and frame_count:
        cap = cv2.VideoCapture(str(members["source.mp4"]))
        if cap.isOpened():
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            sizes = {(e["width"], e["height"]) for e in pose_log.values()}
            ids_ok = all(0 <= f < n for f in pose_log)
            if n == int(frame_count) and ids_ok and sizes <= {(w, h)}:
                return cap, {k: k for k in range(n)}, {
                    "video": "source.mp4", "method": "source_frame_index_direct", "video_frames": n,
                    "twin_frame_count": int(frame_count), "size": [w, h],
                    "proof": "decoded frame count == recording.frame_count, pose-log ids in range, image size matches"}
            tried.append(f"source.mp4 not provably aligned (frames {n} vs {frame_count}, ids_ok={ids_ok}, "
                         f"size {w}x{h} vs {sorted(sizes)})")
        cap.release()
    if "recording.mp4" in members:
        cap = cv2.VideoCapture(str(members["recording.mp4"]))
        if cap.isOpened():
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mapping, info = map_video_to_twin(n, float(cap.get(cv2.CAP_PROP_FPS)), timestamps, trim_start)
            info.update(video="recording.mp4", video_frames=n, fallback_reasons=tried)
            return cap, mapping, info
        tried.append("recording.mp4 unreadable")
    raise SilhouetteError("No video with provable frame alignment: " + "; ".join(tried or ["none available"]))


# ------------------------------------------------------------- main entry
def fit_arm_radii(bundle_path, subject, names, priors, upper_body, model_path, geometry,
                  timestamps=None, trim_start=0.0, frame_count=None):
    """Measure arm silhouette radii.

    priors: {child joint name: prior radius}. geometry(frame_dict) -> (posed (J,3), focal_px).
    Returns dict: radii {child name: radius used}, plus JSON-serializable evidence.
    """
    cv2 = _import_cv2()
    net, out_names = load_net(cv2, model_path)
    idx = {n: i for i, n in enumerate(names)}
    frames_by_id = {f["frame"]: f for f in subject.get("frames") or []}
    with tempfile.TemporaryDirectory() as tmp:
        members = extract_members(bundle_path, tmp)
        pose_log = read_pose_log(members["poses.jsonl"], subject.get("subject_segment"))
        cap, mapping, alignment = choose_video(cv2, members, frame_count, pose_log, timestamps, trim_start)
        try:
            video_frames = int(alignment["video_frames"])
            usable = {k: f for k, f in mapping.items() if f in frames_by_id and f in pose_log}
            if not usable:
                raise SilhouetteError("No video frame maps to a subject twin frame with a pose-log entry")
            chosen = select_frames(sorted(usable))
            wanted = {k: usable[k] for k in chosen}
            per_bone = {c: {} for c in ARM_SEGMENTS if c in idx}
            stats = {"frames_selected": len(wanted), "frames_processed": 0, "frames_rejected": {},
                     "sample_rejects": {}, "samples_accepted": 0, "video_frames": video_frames,
                     "alignment": alignment}
            raw = []
            n = 0
            while n <= max(wanted):
                ok, image = cap.read()
                if not ok:
                    stats["frames_rejected"]["video_frame_unreadable"] = \
                        sum(1 for w in wanted if w >= n)
                    break
                if n in wanted:
                    f = wanted[n]
                    _process_frame(cv2, net, out_names, image, f, pose_log[f], frames_by_id[f],
                                   upper_body, names, idx, priors, geometry, per_bone, stats, raw)
                n += 1
        finally:
            cap.release()
    bones = {}
    radii = {}
    for child, per_frame in per_bone.items():
        ev = aggregate_bone(per_frame, priors[child])
        ev["segment"] = ARM_SEGMENTS[child][2]
        ev["keypoints"] = list(ARM_SEGMENTS[child][:2])
        bones[child] = ev
        radii[child] = ev["radius"]
    return {"radii": radii, "evidence": {
        "method": "silhouette_min_half_width_v1",
        "caveat": "Visible clothed silhouette width at fixed fractions of the arm in the verified aligned video; "
                  "not anatomical tissue size, no pose refit, no landmark-error reduction. Depth axis, torso and "
                  "head keep priors.",
        "model": str(model_path), "stats": stats, "bones": bones, "raw_accepted_samples": raw}}


def _process_frame(cv2, net, out_names, image, n, entry, frame, upper_body, names, idx, priors,
                   geometry, per_bone, stats, raw):
    person = entry["person"]
    rej = stats["frames_rejected"]

    def reject(reason):
        rej[reason] = rej.get(reason, 0) + 1

    height, width = image.shape[:2]
    try:
        left, top, side = crop_geometry(person["bbox"], upper_body)
    except (KeyError, TypeError, ValueError):
        return reject("bad_bbox")
    if not (side > 0 and math.isfinite(side)):
        return reject("bad_bbox")
    blob = preprocess(cv2, image, left, top, side)
    mask, presence = run_segmentation(cv2, net, out_names, blob)
    if presence < MIN_PRESENCE:
        return reject("presence_below_0.7")
    kps = {k["name"]: k for k in person.get("observed_keypoints") or person.get("keypoints") or []}
    pts = {}
    for name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"):
        k = kps.get(name)
        if k and k.get("score", 0) >= MIN_KEYPOINT_SCORE and 0 <= k["x"] < width and 0 <= k["y"] < height:
            pts[name] = np.array([k["x"], k["y"]], float)
    posed, focal = geometry(frame)
    stats["frames_processed"] += 1
    scale = CROP / side
    crop_pts = {k: (v - [left, top]) * scale for k, v in pts.items()}
    # Segments for nonincident-limb checks (all four arm bones, when both ends are available).
    all_segs = {}
    for child, (p, c, _) in ARM_SEGMENTS.items():
        if p in crop_pts and c in crop_pts:
            all_segs[child] = (p, c)
    torso = torso_polygon(crop_pts["left_shoulder"], crop_pts["right_shoulder"]) \
        if "left_shoulder" in crop_pts and "right_shoulder" in crop_pts else None
    for child, (p, c, label) in ARM_SEGMENTS.items():
        if child not in per_bone:
            continue
        if child not in all_segs:
            stats["sample_rejects"]["keypoint_unobserved_or_low_score"] = \
                stats["sample_rejects"].get("keypoint_unobserved_or_low_score", 0) + 1
            continue
        if float(np.linalg.norm(pts[p] - pts[c])) < MIN_SEGMENT_PX:
            stats["sample_rejects"]["segment_shorter_than_30px"] = \
                stats["sample_rejects"].get("segment_shorter_than_30px", 0) + 1
            continue
        others = [(crop_pts[a], crop_pts[b]) for ch, (a, b) in all_segs.items()
                  if ch != child and not ({a, b} & {p, c})]
        samples, rejects = measure_segment(mask, crop_pts[p], crop_pts[c], others, torso)
        for r in rejects:
            stats["sample_rejects"][r] = stats["sample_rejects"].get(r, 0) + 1
        z = 0.5 * (posed[idx[p]][2] + posed[idx[c]][2])
        if not (z > 0 and focal and focal > 0):
            stats["sample_rejects"]["nonpositive_depth"] = stats["sample_rejects"].get("nonpositive_depth", 0) + 1
            continue
        for s in samples:
            radius_px = s["half_width_crop_px"] * side / CROP
            radius = radius_px * z / focal
            prior = priors[child]
            if not (RADIUS_RANGE[0] <= radius <= RADIUS_RANGE[1]) or \
                    not (PRIOR_RATIO_RANGE[0] * prior <= radius <= PRIOR_RATIO_RANGE[1] * prior):
                stats["sample_rejects"]["radius_out_of_range"] = stats["sample_rejects"].get("radius_out_of_range", 0) + 1
                continue
            per_bone[child].setdefault(n, []).append(radius)
            stats["samples_accepted"] += 1
            raw.append({"bone": child, "frame": n, "t": s["t"], "radius_px": radius_px, "radius": radius})
