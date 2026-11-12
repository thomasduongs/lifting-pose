"""Smoke and axis tests for render_biomech. Run:
    .venv-biomech/bin/python -m unittest discover -s scripts -p 'test_*biomech*.py'
Projection axes are checked offline; the end-to-end test uses the real MuJoCo model
at a tiny size and skips when the biomech environment is not available."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
    import mujoco  # noqa: F401
except ImportError as exc:  # run with .venv-biomech/bin/python
    raise unittest.SkipTest(f"biomech renderer needs .venv-biomech (mujoco, opencv): {exc}")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_biomech as rb  # noqa: E402


class AxisTests(unittest.TestCase):
    def test_front_view_right_is_world_plus_x_and_up_is_z(self):
        right, up, fwd = rb.view_basis("front")
        np.testing.assert_allclose(right, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(up, [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(fwd, [0, 1, 0], atol=1e-12)

    def test_side_view_right_is_world_plus_y(self):
        right, up, _ = rb.view_basis("side")
        np.testing.assert_allclose(right, [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(up, [0, 0, 1], atol=1e-12)

    def test_projection_puts_plus_x_right_and_plus_z_up(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        cam = rb.Camera("front", pts, size=200)
        centre, px, pz = cam.project(pts)
        self.assertGreater(px[0], centre[0])  # +x is screen-right
        self.assertLess(pz[1], centre[1])     # +z is screen-up (smaller y pixel)

    def test_unknown_view_is_rejected(self):
        with self.assertRaises(rb.RenderError):
            rb.view_basis("top")


class NativePathTests(unittest.TestCase):
    """Route geometry from the engine's solved wrap points, at the standing pose."""

    @classmethod
    def setUpClass(cls):
        try:
            import biomech_model as bm
        except ImportError as exc:
            raise unittest.SkipTest(f"biomech model unavailable: {exc}")
        cls.model, cls.data = bm.load_model()
        mujoco.mj_forward(cls.model, cls.data)

    def _tendon(self, name):
        tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, name)
        self.assertGreaterEqual(tid, 0, f"tendon {name} missing")
        return tid

    def _polyline(self, t):
        pts = rb.tendon_path(self.model, self.data, t)
        return pts, float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def test_standing_routes_match_engine_length(self):
        for name in ("DELT1_tendon", "BIClong_tendon"):
            t = self._tendon(name)
            pts, length = self._polyline(t)
            self.assertGreaterEqual(pts.shape[0], 2)
            self.assertLess(abs(length - self.data.ten_length[t]), 1e-6, name)

    def test_route_endpoints_are_the_engine_endpoints(self):
        for name in ("DELT1_tendon", "BIClong_tendon"):
            t = self._tendon(name)
            pts, _ = self._polyline(t)
            sites = rb.tendon_sites(self.model, t)
            np.testing.assert_allclose(pts[0], self.data.site_xpos[sites[0]], atol=1e-9)
            np.testing.assert_allclose(pts[-1], self.data.site_xpos[sites[-1]], atol=1e-9)

    def test_all_routes_are_finite_and_close_to_engine_length(self):
        for t in range(self.model.ntendon):
            pts, length = self._polyline(t)
            self.assertTrue(np.all(np.isfinite(pts)), t)
            # Chord approximation of wrap arcs: a few mm at standing (measured max 4.6 mm).
            self.assertLess(abs(length - self.data.ten_length[t]), 0.01, t)


def _write_subject(run_dir: Path):
    import biomech_model as bm

    model, data = bm.load_model()
    mujoco.mj_forward(model, data)
    mujoco.mj_saveModel(model, str(run_dir / "model.mjb"), None)
    qpos = np.tile(data.qpos.copy(), (2, 1))
    markers = []
    for i in range(2):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        markers.append(list(bm.marker_positions(model, data).values()))
    names = list(bm.marker_positions(model, data).keys())
    np.savez(run_dir / "motion.npz", qpos=qpos, timestamps_seconds=np.array([0.0, 1 / 30]),
             marker_positions=np.array(markers), marker_names=np.array(names))


class PresentationTimingTests(unittest.TestCase):
    def test_irregular_timestamps_are_held_and_final_pose_is_included(self):
        indices, times = rb.presentation_frames([0.0, 0.51, 0.57], 30.0)
        self.assertEqual(len(times), 19)
        self.assertTrue(np.all(indices[times < 0.51] == 0))
        self.assertEqual(indices[-1], 2)
        self.assertGreaterEqual(times[-1], 0.57)
        for bad in ([0.0, 0.0], [0.0, np.nan]):
            with self.assertRaises(rb.RenderError):
                rb.presentation_frames(bad, 30.0)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import biomech_model  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"biomech model unavailable: {exc}")

    def test_renders_mp4_and_png_from_fitted_subject_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "subject"
            run_dir.mkdir()
            _write_subject(run_dir)
            out = Path(tmp) / "front.mp4"
            png = Path(tmp) / "frame0.png"
            info = rb.run(run_dir, out, "front", fps=30.0, max_frames=2, size=160, snapshot=png)
            self.assertEqual(info["frames"], 2)
            self.assertGreater(info["muscle_tendons"], 0)
            self.assertEqual(info["ligament_routes"], 8)
            self.assertTrue(out.is_file() and out.stat().st_size > 0)
            self.assertTrue(png.is_file())
            img = cv2.imread(str(png))
            self.assertEqual(img.shape[:2], (160, 160))
            # The figure must cover pixels that are neither background nor text-only.
            self.assertGreater(int((np.abs(img.astype(int) - 247).sum(axis=2) > 30).sum()), 200)
            cap = cv2.VideoCapture(str(out))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            cap.release()

    def test_missing_motion_is_reported_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(rb.RenderError):
                rb.run(Path(tmp), Path(tmp) / "x.mp4", "front", 30.0, 1, 160, None)


if __name__ == "__main__":
    unittest.main()
