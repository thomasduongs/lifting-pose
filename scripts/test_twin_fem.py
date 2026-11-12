#!/usr/bin/env python3
"""Numerical tests for scripts/twin_fem.py. Run: python3 -m unittest scripts/test_twin_fem.py"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_fem as tf  # noqa: E402


def rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


# Root(0) -> shoulder(1) -> elbow(2) -> wrist(3); plus a thin branch 0 -> 4.
JOINTS = np.array([[0, 0, 0], [0.15, 0, 0], [0.45, 0, 0], [0.72, 0, 0], [0.0, 0.3, 0.0]], float)
PARENTS = np.array([-1, 0, 1, 2, 0])


def arm_mesh(spacing=0.05, radii=(0.06, 0.05, 0.04)):
    return tf.build_mesh(JOINTS[:4], PARENTS[:4], [1, 2, 3], list(radii), spacing=spacing)


def contained(points, mesh):
    v, t = mesh["vertices"], mesh["tetrahedra"]
    x0 = v[t[:, 0]]
    inv = np.linalg.inv(np.stack([v[t[:, 1]] - x0, v[t[:, 2]] - x0, v[t[:, 3]] - x0], axis=2))
    out = []
    for p in points:
        b = np.einsum("tij,tj->ti", inv, p - x0)
        bary = np.concatenate([1 - b.sum(1, keepdims=True), b], 1)
        out.append(bool(np.any(np.all(bary >= -1e-9, axis=1))))
    return np.array(out)


class MeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = arm_mesh()

    def test_positive_volumes_and_shapes(self):
        m = self.mesh
        vol, _ = tf._signed_volumes(m["vertices"], m["tetrahedra"])
        self.assertTrue(np.all(vol > 0))
        np.testing.assert_allclose(vol, 0.05 ** 3 / 6, rtol=1e-9)
        self.assertEqual(m["weights"].shape, (len(m["vertices"]), 4))
        np.testing.assert_allclose(m["weights"].sum(1), 1.0)
        self.assertTrue(np.all(m["weights"][:, 0] == 0))  # root is not a bone
        self.assertEqual(len(np.unique(m["tetrahedra"])), len(m["vertices"]))  # no unused vertices

    def test_conforming_closed_outward_boundary(self):
        m = self.mesh
        faces = np.sort(m["tetrahedra"][:, tf._FACE_LOCAL].reshape(-1, 3), axis=1)
        _, counts = np.unique(faces, axis=0, return_counts=True)
        self.assertTrue(np.all(counts <= 2))
        surf = m["surface_triangles"]
        self.assertEqual(len(surf), int(np.sum(counts == 1)))
        # Closed surface: every directed edge is matched by its reverse.
        edges = np.concatenate([surf[:, [0, 1]], surf[:, [1, 2]], surf[:, [2, 0]]])
        fwd = {tuple(e) for e in edges}
        self.assertTrue(all((b, a) in fwd for a, b in fwd))
        # Outward orientation: divergence theorem gives the total volume.
        v = m["vertices"]
        a, b, c = v[surf[:, 0]], v[surf[:, 1]], v[surf[:, 2]]
        enclosed = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
        vol, _ = tf._signed_volumes(v, m["tetrahedra"])
        self.assertAlmostEqual(enclosed, vol.sum(), places=10)

    def test_deterministic(self):
        other = arm_mesh()
        for key in ("vertices", "tetrahedra", "surface_triangles", "weights"):
            np.testing.assert_array_equal(self.mesh[key], other[key])

    def test_thin_limb_is_covered_and_connected(self):
        m = tf.build_mesh(JOINTS, PARENTS, [1, 2, 3, 4], [0.06, 0.05, 0.04, 0.004], spacing=0.06)
        for child in (1, 2, 3, 4):
            a, b = JOINTS[PARENTS[child]], JOINTS[child]
            pts = a + np.linspace(0, 1, 25)[:, None] * (b - a)
            self.assertTrue(np.all(contained(pts, m)), f"bone {child} not covered")

    def test_disconnected_bones_rejected(self):
        joints = np.array([[0, 0, 0], [0.1, 0, 0], [1.0, 0, 0], [1.2, 0, 0]], float)
        with self.assertRaises(ValueError):
            tf.build_mesh(joints, [-1, 0, 1, 2], [1, 3], 0.03, spacing=0.05)

    def test_invalid_mesh_inputs(self):
        with self.assertRaises(ValueError):
            tf.build_mesh(JOINTS[:4], PARENTS[:4], [0], 0.05)  # root bone
        with self.assertRaises(ValueError):
            tf.build_mesh(JOINTS[:4], PARENTS[:4], [1], -0.05)
        with self.assertRaises(ValueError):
            tf.build_mesh(JOINTS[:4], [-1, 2, 1, 2], [1], 0.05)  # cycle
        bad = JOINTS[:4].copy()
        bad[2, 0] = np.nan
        with self.assertRaises(ValueError):
            tf.build_mesh(bad, PARENTS[:4], [1, 2], 0.05)


class SkinTests(unittest.TestCase):
    def test_identity_and_global_rigid(self):
        m = arm_mesh()
        rot = np.repeat(np.eye(3)[None], 4, 0)
        np.testing.assert_allclose(tf.skin_targets(m, JOINTS[:4], JOINTS[:4], rot), m["vertices"], atol=1e-12)
        r = rotation([1, 2, 3], 1.1)
        shift = np.array([0.3, -0.2, 1.0])
        posed = JOINTS[:4] @ r.T + shift
        out = tf.skin_targets(m, JOINTS[:4], posed, np.repeat(r[None], 4, 0))
        np.testing.assert_allclose(out, m["vertices"] @ r.T + shift, atol=1e-12)
        # xyzw quaternion form: rotation about z by pi/2.
        q = np.repeat([[0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]], 4, 0)
        rz = rotation([0, 0, 1], np.pi / 2)
        out = tf.skin_targets(m, JOINTS[:4], JOINTS[:4] @ rz.T, q)
        np.testing.assert_allclose(out, m["vertices"] @ rz.T, atol=1e-12)


def bent_targets(mesh, angle=np.pi / 2):
    """Elbow flexion: the forearm (bone 3, parent elbow 2) rotates about z."""
    rot = np.repeat(np.eye(3)[None], 4, 0)
    rot[2] = rotation([0, 0, 1], angle)
    posed = JOINTS[:4].copy()
    posed[3] = posed[2] + rot[2] @ (JOINTS[3] - JOINTS[2])
    return tf.skin_targets(mesh, JOINTS[:4], posed, rot)


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = arm_mesh()

    def test_rigid_motion_zero_energy(self):
        p = tf.FemProblem(self.mesh)
        for angle in (0.3, np.pi / 2, 2.9):
            r = rotation([0.2, -1, 0.5], angle)
            x = self.mesh["vertices"] @ r.T + np.array([1.0, 2.0, -3.0])
            energy, grad, _ = p.elastic(x)
            self.assertLess(energy, 1e-20)
            self.assertLess(np.abs(grad).max(), 1e-10)
        res = tf.solve_frame(self.mesh, x)
        d = res["diagnostics"]
        self.assertLess(d["objective"], 1e-18)
        self.assertLess(d["target_rmse"], 1e-9)
        self.assertEqual(d["inverted_elements"], 0)
        self.assertAlmostEqual(d["min_det_f"], 1.0, places=8)

    def check_gradient(self, x, **kw):
        rng = np.random.default_rng(1)
        t = bent_targets(self.mesh, 0.7)
        w = rng.uniform(0.1, 1.0, len(x))
        kw = dict(young=2.0, poisson=0.3, data_weight=10.0, **kw)
        _, grad = tf.fem_objective(self.mesh, x, t, w, **kw)
        self.assertIsNotNone(grad)
        eps = 1e-6
        direction = rng.standard_normal(x.shape)
        ep, _ = tf.fem_objective(self.mesh, x + eps * direction, t, w, **kw)
        em, _ = tf.fem_objective(self.mesh, x - eps * direction, t, w, **kw)
        fd = (ep - em) / (2 * eps)
        self.assertAlmostEqual(fd, float(np.sum(grad * direction)), delta=1e-6 * max(1.0, abs(fd)))
        for idx in rng.choice(x.size, 6, replace=False):  # per-coordinate checks
            e = np.zeros(x.size)
            e[idx] = eps
            e = e.reshape(x.shape)
            fp, _ = tf.fem_objective(self.mesh, x + e, t, w, **kw)
            fm, _ = tf.fem_objective(self.mesh, x - e, t, w, **kw)
            self.assertAlmostEqual((fp - fm) / (2 * eps), grad.reshape(-1)[idx], delta=1e-6)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        rest = self.mesh["vertices"]
        # Unconstrained energy, including some inverted elements.
        x = rest + 0.02 * rng.standard_normal(rest.shape)
        _, _, f = tf.FemProblem(self.mesh).elastic(x)
        self.assertTrue(np.any(np.linalg.det(f) <= 0))
        self.check_gradient(x, barrier_weight=0.0)
        # Barrier active: compress z so 0.05 < det F < 0.55 everywhere.
        x = rest * np.array([1.0, 1.0, 0.4]) + 0.001 * rng.standard_normal(rest.shape)
        dets = np.linalg.det(tf.FemProblem(self.mesh).deformation(x))
        self.assertTrue(np.all((dets > 0.1) & (dets < 0.55)))
        p = tf.FemProblem(self.mesh)
        self.assertGreater(p.terms(x, x, np.ones(len(x)))["barrier"], 0.0)
        self.check_gradient(x)

    def test_newton_hessian_matches_gradient_differences(self):
        # Near rest the exact Hessian is PSD, so projection must leave it unchanged.
        rng = np.random.default_rng(3)
        rest = self.mesh["vertices"]
        x = rest + 1e-4 * rng.standard_normal(rest.shape)
        t = bent_targets(self.mesh, 0.5)
        w = rng.uniform(0.2, 1.0, len(x))
        p = tf.FemProblem(self.mesh, young=2.0, poisson=0.3, data_weight=10.0)
        hv = p.hessian(x, w) @ (v := rng.standard_normal(x.size))
        eps = 1e-6
        gp = p.objective(x + eps * v.reshape(x.shape), t, w)[1]
        gm = p.objective(x - eps * v.reshape(x.shape), t, w)[1]
        fd = ((gp - gm) / (2 * eps)).reshape(-1)
        # Forward-difference dP/dF (eps 1e-7) and clamping of near-null rotational
        # modes bound the agreement at ~1e-3; the Hessian only steers directions.
        self.assertLess(np.linalg.norm(hv - fd) / np.linalg.norm(fd), 3e-3)

    def test_newton_and_lbfgs_agree(self):
        t = bent_targets(self.mesh, 0.8)
        newton = tf.solve_frame(self.mesh, t, max_iterations=100)["diagnostics"]
        lbfgs = tf.solve_frame(self.mesh, t, max_iterations=2000, method="lbfgs")["diagnostics"]
        self.assertTrue(newton["converged"])
        self.assertLess(newton["iterations"], 50)
        self.assertAlmostEqual(newton["objective"], lbfgs["objective"], delta=1e-6 * max(1.0, newton["objective"]))
        with self.assertRaises(ValueError):
            tf.solve_frame(self.mesh, t, method="cg")

    def test_barrier_is_infinite_when_inverted(self):
        x = self.mesh["vertices"].copy()
        x[:, 2] *= 0.03  # det F = 0.03 < min_det
        energy, grad = tf.fem_objective(self.mesh, x, x)
        self.assertEqual(energy, np.inf)
        self.assertIsNone(grad)

    def test_nonrigid_targets_improve_over_rest(self):
        t = bent_targets(self.mesh)
        rest = self.mesh["vertices"]
        rest_obj, _ = tf.fem_objective(self.mesh, rest, t)
        rest_rmse = np.sqrt(np.mean(np.sum((rest - t) ** 2, 1)))
        res = tf.solve_frame(self.mesh, t, max_iterations=200)
        d = res["diagnostics"]
        self.assertLess(d["objective"], rest_obj)
        self.assertLessEqual(d["objective"], d["initial_objective"])
        self.assertLess(d["target_rmse"], 0.1 * rest_rmse)
        self.assertGreater(d["elastic_energy"], 0.0)  # bending is genuinely nonrigid
        # A 90 degree LBS elbow inverts inner elements of the *targets*; the barrier
        # keeps every solved element above min_det regardless.
        self.assertGreater(d["target_inverted_elements"], 0)
        dets = np.linalg.det(tf.FemProblem(self.mesh).deformation(res["vertices"]))
        self.assertGreater(dets.min(), 0.05)
        self.assertAlmostEqual(d["min_det_f"], dets.min(), places=12)
        self.assertEqual(d["inverted_elements"], 0)
        self.assertTrue(d["seed"].startswith("rigid_fit"))
        # Without the barrier the same targets leave inverted elements, reported as such.
        free = tf.solve_frame(self.mesh, t, max_iterations=200, barrier_weight=0.0)["diagnostics"]
        self.assertGreater(free["inverted_elements"], 0)
        self.assertLess(free["min_det_f"], 0.0)
        obj, _ = tf.fem_objective(self.mesh, res["vertices"], t)
        self.assertAlmostEqual(obj, d["objective"], places=12)
        # From a rest-pose start the solver must still move toward the targets.
        cold = tf.solve_frame(self.mesh, t, max_iterations=300, initial_vertices=rest)["diagnostics"]
        self.assertLess(cold["target_rmse"], 0.5 * rest_rmse)

    def test_lower_data_weight_trades_fit_for_smoothness(self):
        t = bent_targets(self.mesh)
        stiff = tf.solve_frame(self.mesh, t, data_weight=1.0, max_iterations=300)["diagnostics"]
        loose = tf.solve_frame(self.mesh, t, data_weight=1000.0, max_iterations=300)["diagnostics"]
        self.assertGreater(stiff["target_rmse"], loose["target_rmse"])
        self.assertLess(stiff["elastic_energy"], loose["elastic_energy"])

    def test_invalid_materials_and_inputs(self):
        t = self.mesh["vertices"]
        for kwargs in ({"young": 0.0}, {"young": -1.0}, {"young": np.nan}, {"poisson": 0.5},
                       {"poisson": -0.1}, {"poisson": np.inf}, {"data_weight": -1.0}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                tf.solve_frame(self.mesh, t, **kwargs)
        bad = t.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            tf.solve_frame(self.mesh, bad)
        with self.assertRaises(ValueError):
            tf.solve_frame(self.mesh, t[:-1])
        with self.assertRaises(ValueError):
            tf.solve_frame(self.mesh, t, target_weights=-np.ones(len(t)))

    def test_degenerate_elements_rejected(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], float)
        flat = {"vertices": verts, "tetrahedra": np.array([[0, 1, 2, 4], [0, 1, 2, 3]])}
        with self.assertRaises(ValueError):
            tf.solve_frame(flat, verts)
        inverted = {"vertices": verts, "tetrahedra": np.array([[0, 2, 1, 4]])}
        with self.assertRaises(ValueError):
            tf.solve_frame(inverted, verts)
        repeated = {"vertices": verts, "tetrahedra": np.array([[0, 1, 1, 4]])}
        with self.assertRaises(ValueError):
            tf.solve_frame(repeated, verts)


RECORDING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "recording.twin")


@unittest.skipUnless(os.path.exists(RECORDING), "output/recording.twin not present")
class RealRecordingRegression(unittest.TestCase):
    def test_representative_frames_stay_valid(self):
        import json
        import tempfile
        import reconstruct_twin

        with tempfile.TemporaryDirectory() as out:
            reconstruct_twin.main([RECORDING, "--output", out, "--max-frames", "3", "--spacing", "0.06",
                                   "--max-iterations", "120"])
            subject_dirs = [d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))]
            self.assertTrue(subject_dirs)
            for sub in subject_dirs:
                with open(os.path.join(out, sub, "diagnostics.json")) as fh:
                    frames = json.load(fh)["frames"]
                self.assertTrue(frames)
                # The skinned targets of this recording invert elements; the solve must not.
                self.assertTrue(any(f["target_inverted_elements"] > 0 for f in frames))
                for f in frames:
                    self.assertTrue(f["valid"], f["validity_reason"])
                    self.assertEqual(f["inverted_elements"], 0)
                    self.assertGreater(f["min_det_f"], 0.05)
                    self.assertLess(f["target_rmse"], 0.02)


if __name__ == "__main__":
    unittest.main()
