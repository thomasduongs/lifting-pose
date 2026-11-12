"""Headless tests for replay_biomech: loading, validation, timing and interpolation.
No viewer is opened. Run:
    .venv-biomech/bin/python -m unittest discover -s scripts -p 'test_*biomech*.py'"""
import io
import json
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:  # run with .venv-biomech/bin/python
    raise unittest.SkipTest(f"biomech replay needs .venv-biomech (mujoco): {exc}")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biomech_model as bm  # noqa: E402
import replay_biomech as rp  # noqa: E402


def _write_run(run_dir: Path, qpos, ts):
    model, data = bm.load_model()
    mujoco.mj_forward(model, data)
    mujoco.mj_saveModel(model, str(run_dir / "model.mjb"), None)
    np.savez(run_dir / "motion.npz", qpos=qpos, timestamps_seconds=ts)


def _standing_frames(n=3):
    model, data = bm.load_model()
    mujoco.mj_forward(model, data)
    return model, np.tile(data.qpos.copy(), (n, 1))


class TimingTests(unittest.TestCase):
    def test_speed_scales_recording_time(self):
        self.assertAlmostEqual(rp.playback_time(2.0, 0.5, 10.0, loop=False), 1.0)
        self.assertAlmostEqual(rp.playback_time(2.0, 2.0, 10.0, loop=False), 4.0)

    def test_hold_at_end_without_loop(self):
        self.assertAlmostEqual(rp.playback_time(100.0, 1.0, 10.0, loop=False), 10.0)

    def test_loop_wraps_into_duration(self):
        self.assertAlmostEqual(rp.playback_time(12.5, 1.0, 10.0, loop=True), 2.5)

    def test_speed_must_be_positive_and_finite(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(rp.ReplayError):
                rp.check_speed(bad)
        self.assertEqual(rp.check_speed(1), 1.0)


class InterpolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.standing = _standing_frames(1)

    def test_midpoint_of_identical_frames_is_identity(self):
        qpos = np.tile(self.standing[0], (2, 1))
        ts = np.array([0.0, 0.5])
        q = rp.interpolate_qpos(self.model, qpos, ts, 0.25)
        np.testing.assert_allclose(q, self.standing[0], atol=1e-9)

    def test_midpoint_of_hinge_joint_is_average(self):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion_r")
        if jid < 0:
            self.skipTest("elbow_flexion_r not present in this model")
        adr = self.model.jnt_qposadr[jid]
        a = self.standing[0].copy()
        b = a.copy()
        a[adr] = 0.2
        b[adr] = 0.6
        q = rp.interpolate_qpos(self.model, np.vstack([a, b]), np.array([0.0, 1.0]), 0.5)
        self.assertAlmostEqual(float(q[adr]), 0.4, places=6)

    def test_irregular_timestamps_are_respected(self):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion_r")
        if jid < 0:
            self.skipTest("elbow_flexion_r not present in this model")
        adr = self.model.jnt_qposadr[jid]
        frames = np.tile(self.standing[0], (3, 1))
        frames[0, adr], frames[1, adr], frames[2, adr] = 0.0, 0.2, 1.0
        ts = np.array([0.0, 0.1, 1.0])
        q = rp.interpolate_qpos(self.model, frames, ts, 0.55)  # halfway in the 0.1..1.0 segment
        self.assertAlmostEqual(float(q[adr]), 0.2 + (1.0 - 0.2) * (0.55 - 0.1) / 0.9, places=6)

    def test_free_root_quaternion_stays_unit(self):
        a = self.standing[0].copy()
        b = a.copy()
        b[3:7] = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]  # 90 deg about x
        q = rp.interpolate_qpos(self.model, np.vstack([a, b]), np.array([0.0, 1.0]), 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(q[3:7])), 1.0, places=6)
        self.assertTrue(np.all(np.isfinite(q)))


class LoadTests(unittest.TestCase):
    def test_inspect_summary_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            model, qpos = _standing_frames(3)
            _write_run(run_dir, qpos, np.array([0.0, 1 / 30, 0.1]))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = rp.main([str(run_dir), "--inspect"])
            self.assertEqual(code, 0)
            info = json.loads(buf.getvalue())
            self.assertEqual(info["frames"], 3)
            self.assertEqual(info["nq"], model.nq)
            self.assertAlmostEqual(info["dt_max_seconds"], 0.1 - 1 / 30, places=9)
            self.assertLess(info["coupling_residual_max"], 1e-6)

    def test_rejects_non_increasing_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _, qpos = _standing_frames(2)
            _write_run(run_dir, qpos, np.array([0.5, 0.5]))
            with self.assertRaises(rp.ReplayError):
                rp.load_run(run_dir)

    def test_rejects_wrong_nq_and_nonfinite(self):
        model, qpos = _standing_frames(2)
        with self.assertRaises(rp.ReplayError):
            rp.validate(model, qpos[:, :-1], np.array([0.0, 1.0]))
        bad = qpos.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(rp.ReplayError):
            rp.validate(model, bad, np.array([0.0, 1.0]))

    def test_missing_files_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(rp.ReplayError):
                rp.load_run(Path(tmp))


if __name__ == "__main__":
    unittest.main()
