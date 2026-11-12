import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruct_twin as rt

NAMES = ['pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
         'spine', 'thorax', 'head_pivot', 'nose', 'head', 'left_shoulder', 'left_elbow', 'left_wrist',
         'right_shoulder', 'right_elbow', 'right_wrist']
PARENTS = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 10, 8, 12, 13, 8, 15, 16]
OFF = [[0, 0, 0], [-.1, 0, 0], [0, .33, 0], [0, .33, 0], [.1, 0, 0], [0, .33, 0], [0, .33, 0],
       [0, -.16, 0], [0, -.16, 0], [0, 0, 0], [0, -.25, 0], [0, -.1, 0], [.18, 0, 0], [.25, 0, 0],
       [.27, 0, 0], [-.18, 0, 0], [-.25, 0, 0], [-.27, 0, 0]]


def make_twin(n=4, conf=0.9, timestamps=True):
    frames = []
    for i in range(n):
        # Reversed order and unnormalized quaternions: matching must be by name.
        rots = [{'joint': nm, 'quaternion_xyzw': [0, 0, 0, 2.0], 'confidence': conf} for nm in NAMES[::-1]]
        frames.append({'frame': i, 'root_position': [0, 0.5, 1.7], 'root_orientation_xyzw': [0, 0, 0, 1],
                       'joint_rotations': rots, 'reprojection_rmse_px': 3.0})
    rec = {'fps': 30.0, 'frame_count': n, 'upper_body': True}
    if timestamps:
        rec['timestamps_seconds'] = [0.0, 0.1, 0.35, 0.4, 0.9, 1.0][:n]
    return {'schema_version': 2, 'quaternion_convention': 'xyzw_active_parent_local',
            'coordinate_axes': 'camera_x_right_y_down_z_forward', 'recording': rec, 'joint_names': NAMES, 'parents': list(PARENTS),
            'subjects': [{'subject_segment': 0, 'person': {
                'body_scope': 'upper_body', 'rest_offsets': OFF, 'shoulder_width': .36, 'head_radius': .08,
                'segment_lengths': {'upper_arm': .25, 'forearm': .27}}, 'frames': frames}]}


def stub_fem():
    m = types.ModuleType('twin_fem')

    def build_mesh(rest, parents, bones, radii, spacing=0.06):
        assert 0 not in bones and 9 not in bones and 2 not in bones  # no legs/pelvis/zero-length pivot
        v = np.random.RandomState(0).rand(5, 3)
        w = np.zeros((5, len(rest)))
        w[:, bones[0]] = 1
        return {'vertices': v, 'tetrahedra': np.array([[0, 1, 2, 3]]),
                'surface_triangles': np.array([[0, 1, 2]]), 'weights': w}

    m.build_mesh = build_mesh
    m.skin_targets = lambda mesh, r, p, w: mesh['vertices'] + 0.01
    m.solve_frame = lambda mesh, t, target_weights=None, **k: {
        'vertices': t, 'diagnostics': {'converged': True, 'target_rmse': 0.0, 'min_det_f': 1.0,
                                       'inverted_elements': 0, 'iterations': 1, 'objective': 0.0}}
    return m


def write_zip(path, twin):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('save-format.json', '{"format":"lifting-pose","version":1}')
        z.writestr('twin.json', json.dumps(twin))


class FK(unittest.TestCase):
    def test_name_based_fk_head_pivot(self):
        twin = make_twin()
        names, par = rt.validate_hierarchy(twin)
        local, conf = rt.frame_rotations(twin['subjects'][0]['frames'][0], names)
        self.assertTrue(np.allclose(local, np.eye(3)))
        pos, _ = rt.forward_kinematics(par, np.array(OFF), local, np.eye(3), np.array([0, .5, 1.7]))
        self.assertTrue(np.allclose(pos[9], pos[8]))  # zero-length head pivot
        self.assertTrue(np.allclose(pos[10], pos[8] + [0, -.25, 0]))
        self.assertTrue(np.allclose(pos[13], pos[12] + [.25, 0, 0]))

    def test_rotation_propagates(self):
        names, par = rt.validate_hierarchy(make_twin())
        local = np.tile(np.eye(3), (18, 1, 1))
        local[12] = rt.quat_to_matrix([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90 deg about z
        pos, _ = rt.forward_kinematics(par, np.array(OFF), local, np.eye(3), np.zeros(3))
        # Outgoing-bone convention: the shoulder's own rotation turns its child offset.
        self.assertTrue(np.allclose(pos[13] - pos[12], [0, .25, 0], atol=1e-9))
        self.assertTrue(np.allclose(pos[14] - pos[13], [0, .27, 0], atol=1e-9))

    def test_bad_inputs(self):
        with self.assertRaises(rt.TwinError):
            rt.quat_to_matrix([0, 0, 0, 0])
        with self.assertRaises(rt.TwinError):
            rt.quat_to_matrix([np.nan, 0, 0, 1])
        t = make_twin()
        t['parents'][3] = 5
        with self.assertRaises(rt.TwinError):
            rt.validate_hierarchy(t)
        t = make_twin()
        t['quaternion_convention'] = 'wxyz'
        with self.assertRaises(rt.TwinError):
            rt.validate_hierarchy(t)
        t = make_twin()
        t['schema_version'] = 9
        with self.assertRaises(rt.TwinError):
            rt.validate_hierarchy(t)

    def test_timing(self):
        t = make_twin(4)
        s, _ = rt.timestamps_for(t, [0, 2, 3])
        self.assertTrue(np.allclose(s, [0, .35, .4]))
        t = make_twin(4, timestamps=False)
        with self.assertRaises(rt.TwinError):
            rt.timestamps_for(t, [0])
        t['schema_version'] = 1
        s, src = rt.timestamps_for(t, [3])
        self.assertAlmostEqual(s[0], .1)
        self.assertIn('fallback', src)
        t = make_twin(4)
        t['recording']['timestamps_seconds'][2] = 0.05
        with self.assertRaises(rt.TwinError):
            rt.timestamps_for(t, [0])

    def test_scope(self):
        t = make_twin()
        names, par = rt.validate_hierarchy(t)
        bones, scope = rt.select_active_bones(names, par, t['subjects'][0]['person'], t['recording'])
        self.assertEqual(scope, 'upper_body')
        self.assertTrue(all(b >= 8 for b in bones))
        self.assertNotIn(9, rt.weld_bones(names, par, np.array(OFF), bones))


class CLI(unittest.TestCase):
    def run_cli(self, twin, extra=()):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        src = Path(self.tmp.name) / 'rec.twin'
        write_zip(src, twin)
        sys.modules['twin_fem'] = stub_fem()
        self.addCleanup(sys.modules.pop, 'twin_fem', None)
        out = Path(self.tmp.name) / 'out'
        before = src.read_bytes()
        code = rt.main([str(src), '--output', str(out), *extra])
        self.assertEqual(src.read_bytes(), before)
        return code, out

    def test_zip_irregular_timing_outputs(self):
        code, out = self.run_cli(make_twin(4))
        self.assertEqual(code, 0)
        man = json.load(open(out / 'manifest.json'))
        self.assertEqual(len(man['input_sha256']), 64)
        sd = out / man['subjects'][0]['directory']
        m = np.load(sd / 'motion.npz')
        self.assertTrue(np.allclose(m['timestamps_seconds'], [0, .1, .35, .4]))
        self.assertEqual(m['vertices'].shape, (4, 5, 3))
        self.assertEqual(m['joint_positions'].shape, (4, 18, 3))
        for f in ('mesh.npz', 'rest.obj', 'frame_000000.obj', 'diagnostics.json'):
            self.assertTrue((sd / f).exists())
        d = json.load(open(sd / 'diagnostics.json'))
        self.assertEqual(d['frames'][0]['upstream_reprojection_rmse_px'], 3.0)

    def test_zero_confidence_frames_fail_visibly(self):
        code, _ = self.run_cli(make_twin(3, conf=0.0))
        self.assertEqual(code, 1)

    def test_stride_and_refuse_overwrite(self):
        code, out = self.run_cli(make_twin(4), ['--stride', '2'])
        man = json.load(open(out / 'manifest.json'))
        self.assertEqual(man['subjects'][0]['summary']['frames_reconstructed'], 2)
        self.assertEqual(rt.main([str(Path(self.tmp.name) / 'rec.twin'), '--output', self.tmp.name]), 2)


if __name__ == '__main__':
    unittest.main()
