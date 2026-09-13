"""CPU-only structural checks; no pretrained model or dataset is loaded."""
from dataclasses import replace
import unittest

import torch

from rl3dsr.models.wan.geometry_conditioning import CameraBatch
from rl3dsr.models.wan.lr_fusion import LRViewFusion, patch_fundamental_matrices


def camera(views=3):
    k = torch.tensor([[24., 0., 16.], [0., 20., 12.], [0., 0., 1.]]).repeat(1, views, 1, 1)
    t = torch.eye(4).repeat(1, views, 1, 1)
    t[0, :, 0, 3] = torch.arange(views) * .25
    return CameraBatch(k, t, (24, 32), 'multiview')


def active(mode='epipolar', chunk=3):
    net = LRViewFusion(12, 12, 3, mode=mode, query_chunk_size=chunk)
    torch.nn.init.normal_(net.output.weight, std=.1)
    return net


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(1, 3, 6, 12)
        self.cam = camera()

    def test_known_projection_in_patch_coordinates(self):
        # Unequal image axes and a rotated second camera catch coordinate mistakes.
        c = camera(2)
        t = c.T_world_from_camera.clone()
        angle = torch.tensor(.2)
        t[0, 1, :3, :3] = torch.tensor([[angle.cos(), 0, angle.sin()], [0, 1, 0], [-angle.sin(), 0, angle.cos()]])
        c = replace(c, T_world_from_camera=t)
        f, valid = patch_fundamental_matrices(c, (3, 8))
        world = torch.tensor([.3, -.2, 3., 1.])
        projected = []
        for v in range(2):
            xyz = torch.linalg.inv(t[0, v]) @ world
            uv = c.K[0, v] @ xyz[:3]
            uv = uv / uv[2]
            uv[:2] /= torch.tensor([4., 8.])
            projected.append(uv)
        residual = projected[1] @ f[0, 0, 1] @ projected[0]
        self.assertLess(abs(float(residual)), 1e-5)
        self.assertTrue(bool(valid[0, 0, 1]))
        self.assertFalse(bool(valid[0, 0, 0]))

    def test_noop_and_bypasses(self):
        for mode in ('off', 'same_view', 'visual', 'epipolar', 'epipolar_local'):
            model = LRViewFusion(12, 12, 3, mode=mode)
            self.assertTrue(torch.equal(model(self.x, self.cam, (2, 3)), self.x))
        model = active()
        self.assertIs(model(self.x, replace(self.cam, sequence_kind='temporal'), (2, 3)), self.x)
        single = self.x[:, :1]
        self.assertIs(model(single, camera(1), (2, 3)), single)

    def test_mask_blocks_auxiliary_evidence(self):
        model = active('visual')
        changed = self.x.clone()
        changed[:, 1:] += 20
        mask = torch.tensor([[True, False, False]])
        a = model(self.x, self.cam, (2, 3), source_mask=mask)
        b = model(changed, self.cam, (2, 3), source_mask=mask)
        self.assertTrue(torch.allclose(a[:, 0], b[:, 0], atol=1e-6))
        self.assertFalse(torch.allclose(model(self.x, self.cam, (2, 3))[:, 0], model(changed, self.cam, (2, 3))[:, 0]))
        empty = torch.zeros_like(mask)
        self.assertTrue(torch.equal(model(self.x, self.cam, (2, 3), source_mask=empty), self.x))

    def test_chunk_equivalence_and_permutation(self):
        model = active()
        full = active(chunk=100)
        full.load_state_dict(model.state_dict())
        expected = model(self.x, self.cam, (2, 3))
        self.assertTrue(torch.allclose(expected, full(self.x, self.cam, (2, 3)), atol=1e-6))
        order = torch.tensor([2, 0, 1])
        c = replace(self.cam, K=self.cam.K[:, order], T_world_from_camera=self.cam.T_world_from_camera[:, order])
        self.assertTrue(torch.allclose(expected[:, order], model(self.x[:, order], c, (2, 3)), atol=1e-6))

    def test_same_view_is_independent(self):
        model = active('same_view')
        changed = self.x.clone()
        changed[:, 1:] += 20
        self.assertTrue(torch.allclose(model(self.x, self.cam, (2, 3))[:, 0], model(changed, self.cam, (2, 3))[:, 0], atol=1e-6))

    def test_gradients_and_initialization(self):
        model = LRViewFusion(12, 12, 3)
        model(self.x, self.cam, (2, 3)).square().sum().backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)
        self.assertEqual(float(model.qkv.weight.grad.abs().sum()), 0)
        model = active()
        model(self.x, self.cam, (2, 3)).square().sum().backward()
        self.assertGreater(float(model.qkv.weight.grad.abs().sum()), 0)

    def test_local_epipolar_mode_is_finite_and_uses_a_band(self):
        model = active('epipolar_local')
        output = model(self.x, self.cam, (2, 3))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertEqual(model.epipolar_band, 1.5)
        with self.assertRaises(ValueError):
            LRViewFusion(12, 12, 3, mode='epipolar_local', epipolar_band=0)

    def test_optional_diagnostics_partition_attention_and_bound_keys(self):
        model = active('epipolar_local')
        model.record_diagnostics = True
        model(self.x, self.cam, (2, 3))
        stats = model.last_diagnostics['fusion']
        self.assertAlmostEqual(float(stats['attention_mass_total']), 1.0, places=5)
        self.assertGreaterEqual(float(stats['same_view_attention_mass']), 0.0)
        self.assertGreaterEqual(float(stats['cross_view_attention_mass']), 0.0)
        self.assertGreaterEqual(float(stats['null_attention_mass']), 0.0)
        self.assertGreaterEqual(float(stats['retained_key_ratio']), 0.0)
        self.assertLessEqual(float(stats['retained_key_ratio']), 1.0)
        self.assertAlmostEqual(float(stats['active_auxiliary_source_ratio']), 1.0, places=6)

    def test_diagnostics_are_disabled_by_default(self):
        model = active('epipolar_local')
        model(self.x, self.cam, (2, 3))
        self.assertEqual(model.last_diagnostics, {})

    def test_bfloat16_features_with_float32_weights(self):
        model = active()
        features = self.x.to(torch.bfloat16)
        actual = model(features, self.cam, (2, 3))
        self.assertEqual(actual.dtype, features.dtype)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        actual.float().square().sum().backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.qkv.weight.grad.abs().sum()), 0)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        self.assertTrue(torch.equal(model(features, self.cam, (2, 3), source_mask=mask), features))

    def test_noninteger_dimensions_rejected(self):
        for config in ({'feature_dim': 12.5}, {'hidden_dim': 12.0}, {'heads': 3.0},
                       {'query_chunk_size': 2.5}, {'heads': True}):
            with self.assertRaises(ValueError):
                LRViewFusion(**config)
        with self.assertRaises(ValueError):
            active()(self.x, self.cam, (2., 3))

    def test_degenerate_and_invalid_inputs(self):
        model = active()
        c = replace(self.cam, T_world_from_camera=torch.eye(4).repeat(1, 3, 1, 1))
        self.assertTrue(bool(torch.isfinite(model(self.x, c, (2, 3))).all()))
        with self.assertRaises(ValueError):
            model(self.x, None, (2, 3))
        with self.assertRaises(ValueError):
            model(self.x, self.cam, (2, 4))
        with self.assertRaises(ValueError):
            model(self.x * float('nan'), self.cam, (2, 3))
        with self.assertRaises(ValueError):
            model(self.x, self.cam, (2, 3), source_mask=torch.ones(1, 3))


if __name__ == '__main__':
    unittest.main()
