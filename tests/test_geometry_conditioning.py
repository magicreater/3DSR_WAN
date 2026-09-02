from dataclasses import replace

import torch

from rl3dsr.models.wan.geometry_conditioning import CameraBatch, GeometryConditioner


def _camera(sequence=4, kind="multiview"):
    k = torch.eye(3).repeat(1, sequence, 1, 1)
    k[..., 0, 0] = 16
    k[..., 1, 1] = 16
    k[..., 0, 2] = 8
    k[..., 1, 2] = 8
    t = torch.eye(4).repeat(1, sequence, 1, 1)
    t[..., 0, 3] = torch.arange(sequence, dtype=torch.float32)
    return CameraBatch(k, t, (16, 16), kind)


def test_rre_and_plucker_shapes_and_zero_init():
    camera = _camera()
    for representation, dim in (("rre", 23), ("plucker", 17)):
        module = GeometryConditioner(representation=representation, hidden_dim=16, attention_heads=4)
        assert module.input_dim == dim
        hidden = module.encode(camera, (4, 4, 4))
        assert hidden.shape == (1, 4 * 4, 16)
        residuals = module.residuals(camera, (4, 4, 4), torch.tensor([800.0]))
        assert set(residuals) == {0, 1, 2, 3}
        assert all(value.shape == (1, 16, 1536) for value in residuals.values())
        assert all(torch.equal(value, torch.zeros_like(value)) for value in residuals.values())


def test_camera_change_changes_encoded_geometry():
    first = _camera()
    second = _camera()
    changed = second.T_world_from_camera.clone()
    changed[:, 1:, 1, 3] += 0.25
    second = replace(second, T_world_from_camera=changed)
    module = GeometryConditioner(hidden_dim=16, attention_heads=4)
    assert not torch.equal(module.encode(first, (4, 4, 4)), module.encode(second, (4, 4, 4)))


def test_geometry_gradients_are_finite_after_output_unfreeze():
    camera = _camera()
    module = GeometryConditioner(hidden_dim=16, attention_heads=4)
    with torch.no_grad():
        module.output["0"].weight.normal_(std=1e-3)
    residual = module.residuals(camera, (4, 4, 4), torch.tensor([500.0]))[0]
    loss = residual.square().mean()
    loss.backward()
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    assert trainable
    non_null = [parameter for parameter in trainable if parameter.grad is not None]
    assert non_null
    assert all(torch.isfinite(parameter.grad).all() for parameter in non_null)


def test_temporal_sequence_uses_same_shape_contract():
    camera = _camera(sequence=5, kind="temporal")
    module = GeometryConditioner(hidden_dim=16, attention_heads=4)
    residuals = module.residuals(camera, (5, 4, 4), torch.tensor([200.0]))
    assert residuals[0].shape == (1, 5 * 4, 1536)
