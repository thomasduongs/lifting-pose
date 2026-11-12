import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twin_silhouette as ts

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def capsule_mask(a, b, radius):
    m = np.zeros((256, 256), np.uint8)
    cv2.line(m, tuple(int(v) for v in a), tuple(int(v) for v in b), 1, int(2 * radius), cv2.LINE_8)
    return m.astype(bool)


@unittest.skipIf(cv2 is None, "cv2 not installed")
class Width(unittest.TestCase):
    def test_recovers_capsule_half_width(self):
        a, b = np.array([60.0, 60.0]), np.array([180.0, 120.0])
        samples, rejects = ts.measure_segment(capsule_mask(a, b, 12), a, b)
        self.assertEqual(len(samples), 5, rejects)
        for s in samples:
            self.assertAlmostEqual(s["half_width_crop_px"], 12, delta=1.5)

    def test_truncated_mask_rejected(self):
        a, b = np.array([100.0, 8.0]), np.array([100.0, 200.0])
        samples, rejects = ts.measure_segment(capsule_mask(a, b, 14) | (np.arange(256)[None, :] < 90) &
                                              (np.arange(256)[:, None] > 0), a, b)
        self.assertEqual(samples, [])
        self.assertTrue(set(rejects) <= {"truncated_at_crop_edge", "center_background", "asymmetric"})

    def test_invalid_masks(self):
        a, b = np.array([60.0, 60.0]), np.array([180.0, 60.0])
        samples, rejects = ts.measure_segment(np.zeros((256, 256), bool), a, b)
        self.assertEqual((samples, set(rejects)), ([], {"center_background"}))
        samples, rejects = ts.measure_segment(np.ones((256, 256), bool), a, b)  # fills crop: edge truncation
        self.assertEqual(samples, [])
        thin = capsule_mask(a, b, 0.5)
        self.assertEqual(ts.measure_segment(thin, a, b)[0], [])

    def test_overlap_with_nonincident_limb_rejected(self):
        a, b = np.array([60.0, 100.0]), np.array([190.0, 100.0])
        mask = capsule_mask(a, b, 10)
        other = (np.array([60.0, 112.0]), np.array([190.0, 112.0]))
        samples, rejects = ts.measure_segment(mask, a, b, others=[other])
        self.assertEqual(samples, [])
        self.assertIn("near_other_limb", rejects)

    def test_asymmetric_overlap_rejected(self):
        a, b = np.array([60.0, 100.0]), np.array([190.0, 100.0])
        mask = capsule_mask(a, b, 10) | capsule_mask((60, 125), (190, 125), 14)  # merged blob
        samples, rejects = ts.measure_segment(mask, a, b)
        self.assertEqual(samples, [])

    def test_crop_geometry_matches_rust(self):
        bbox = {"x": 100.0, "y": 50.0, "width": 200.0, "height": 300.0}
        self.assertEqual(ts.crop_geometry(bbox, True), (200 - 300, 50.0, 600.0))
        left, top, side = ts.crop_geometry(bbox, False)
        self.assertEqual((left, top, side), (200 - 187.5, 200 - 187.5, 375.0))


class Aggregate(unittest.TestCase):
    def test_needs_three_frames(self):
        out = ts.aggregate_bone({1: [0.04] * 4, 2: [0.04] * 4}, 0.04)
        self.assertEqual(out["source"], "prior")
        self.assertIn("need 3", out["unsupported_reason"])

    def test_median_and_mad(self):
        pf = {f: [v] * 4 for f, v in {1: .040, 2: .042, 3: .041, 4: .039, 5: .12}.items()}
        out = ts.aggregate_bone(pf, 0.04)
        self.assertEqual(out["source"], "silhouette_median")
        self.assertIn(5, out["mad_rejected_frames"])
        self.assertAlmostEqual(out["radius"], 0.0405, delta=0.001)

    def test_frames_with_few_samples_dropped(self):
        out = ts.aggregate_bone({1: [0.04, 0.04], 2: [0.04] * 3, 3: [0.04] * 3}, 0.04)
        self.assertEqual(out["source"], "prior")


class Members(unittest.TestCase):
    def test_legacy_json_unsupported(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "twin.json"
            p.write_text("{}")
            with self.assertRaises(ts.SilhouetteError) as cm:
                ts.extract_members(p, d)
            self.assertIn("legacy", str(cm.exception))

    def test_missing_member_and_fixed_names_only(self):
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "a.twin"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("poses.jsonl", "")
                zf.writestr("../evil.txt", "x")
            out = Path(d) / "out"
            out.mkdir()
            with self.assertRaises(ts.SilhouetteError):
                ts.extract_members(z, out)
            with zipfile.ZipFile(z, "a") as zf:
                zf.writestr("recording.mp4", "v")
            ts.extract_members(z, out)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["poses.jsonl", "recording.mp4"])
            self.assertFalse((Path(d) / "evil.txt").exists())

    def test_missing_model(self):
        if cv2 is None:
            self.skipTest("cv2")
        with self.assertRaises(ts.SilhouetteError):
            ts.load_net(cv2, "/nonexistent/model.onnx")

    def test_video_alignment_by_timestamp(self):
        ts_list = [i / 30 for i in range(100)]
        mapping, info = ts.map_video_to_twin(30, 30.0, ts_list, 1.0)
        self.assertEqual(mapping[0], 30)
        self.assertEqual(mapping[29], 59)
        with self.assertRaises(ts.SilhouetteError):
            ts.map_video_to_twin(30, 30.0, None, 0.0)

    def _fake_cv2(self, counts, size=(1280, 720)):
        import types
        props = {"FRAME_COUNT": 7, "FRAME_WIDTH": 3, "FRAME_HEIGHT": 4, "FPS": 5}
        class Cap:
            def __init__(self, path):
                self.n = counts[Path(path).name]
            def isOpened(self):
                return True
            def get(self, k):
                return {7: self.n, 3: size[0], 4: size[1], 5: 30.0}[k]
            def release(self):
                pass
        return types.SimpleNamespace(VideoCapture=Cap, CAP_PROP_FRAME_COUNT=7, CAP_PROP_FRAME_WIDTH=3,
                                     CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5)

    def test_choose_video_prefers_matching_source(self):
        members = {"source.mp4": Path("source.mp4"), "recording.mp4": Path("recording.mp4")}
        log = {f: {"width": 1280, "height": 720} for f in (0, 10, 221)}
        stamps = [i / 30 for i in range(222)]
        cv = self._fake_cv2({"source.mp4": 222, "recording.mp4": 63})
        _, mapping, info = ts.choose_video(cv, members, 222, log, stamps, 3.0)
        self.assertEqual(info["video"], "source.mp4")
        self.assertEqual(mapping[100], 100)

    def test_choose_video_falls_back_to_trimmed_by_timestamp(self):
        members = {"source.mp4": Path("source.mp4"), "recording.mp4": Path("recording.mp4")}
        log = {f: {"width": 1280, "height": 720} for f in (0, 10)}
        stamps = [i / 30 for i in range(222)]
        cv = self._fake_cv2({"source.mp4": 200, "recording.mp4": 63})  # count mismatch
        _, mapping, info = ts.choose_video(cv, members, 222, log, stamps, 3.0)
        self.assertEqual(info["video"], "recording.mp4")
        self.assertEqual(mapping[0], 90)
        self.assertTrue(info["fallback_reasons"])
        # size mismatch also rejects the raw source
        log = {0: {"width": 640, "height": 480}}
        cv = self._fake_cv2({"source.mp4": 222, "recording.mp4": 63})
        self.assertEqual(ts.choose_video(cv, members, 222, log, stamps, 3.0)[2]["video"], "recording.mp4")
        with self.assertRaises(ts.SilhouetteError):
            ts.choose_video(cv, {"source.mp4": Path("source.mp4")}, 222, log, stamps, 3.0)

    def test_select_frames(self):
        self.assertEqual(len(ts.select_frames(list(range(222)))), 24)
        self.assertEqual(ts.select_frames([3, 1, 2]), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
