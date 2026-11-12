import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_fem_twin as rf  # noqa: E402


def cube_assets(root, tris=None, frame_indices=(3, 9)):
    import itertools
    c = np.array(list(itertools.product([-0.1, 0.1], repeat=3)), float)
    faces = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                      (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    if tris is not None:
        faces = tris
    sdir = root / "subj_a"
    sdir.mkdir(parents=True)
    frames = np.stack([c, c + np.array([0.0, 0.0, 0.02])])
    np.savez(sdir / "mesh.npz", surface_triangles=faces, rest_vertices=c)
    np.savez(sdir / "motion.npz", frame_indices=np.array(frame_indices),
             timestamps_seconds=np.array([0.0, 0.1]), vertices=frames)
    (root / "manifest.json").write_text(json.dumps({"subjects": [{"directory": "subj_a"}]}))


class AxisConventionTest(unittest.TestCase):
    def test_head_with_lower_y_renders_above_thorax(self):
        # Camera convention: +y points DOWN. A head above the thorax has smaller y.
        verts = np.array([[[0.0, 0.0, 1.0],     # thorax at y=0
                           [0.0, -0.5, 1.0]]])  # head at y=-0.5 (higher in the image)
        for view in ("front", "side"):
            xy, _ = rf.project(verts, view)
            scale, center = rf.fixed_transform(xy, 720, 720)
            px = rf.to_pixels(xy[0], scale, center, 720, 720)
            self.assertLess(px[1, 1], px[0, 1], f"{view}: head must be above thorax")

    def test_fixed_bounds_keep_frames_inside_canvas(self):
        verts = np.array([[[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
                          [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]]])
        xy, _ = rf.project(verts, "front")
        scale, center = rf.fixed_transform(xy, 200, 200)
        for f in range(2):
            px = rf.to_pixels(xy[f], scale, center, 200, 200)
            self.assertTrue(np.all(px >= 0) and np.all(px <= 200))


class ValidationTest(unittest.TestCase):
    def test_rejects_bad_options_and_triangles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cube_assets(root)
            out = root / "o.mp4"
            with self.assertRaises(rf.RenderError):
                rf.render(root, out, None, "front", 0.0, 720)
            with self.assertRaises(rf.RenderError):
                rf.render(root, out, None, "front", float("nan"), 720)
            with self.assertRaises(rf.RenderError):
                rf.render(root, out, None, "front", 30.0, 64)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cube_assets(root, tris=np.array([(0, 1, -1)]))
            with self.assertRaises(rf.RenderError):
                rf.render(root, root / "o.mp4", None, "front", 30.0, 720)


@unittest.skipIf(rf.cv2 is None, "opencv not installed")
class RenderSmokeTest(unittest.TestCase):
    def test_renders_frames_and_labels_source_indices(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cube_assets(root, frame_indices=(3, 9))
            n, out = rf.render(root, root / "o.mp4", None, "front", 30.0, 256)
            self.assertEqual(n, 4)  # 0 to 0.1 s at 30 fps
            self.assertTrue(out.exists() and out.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
