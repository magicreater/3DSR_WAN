import torch
import pytest

from stage3_3_fixed_merge import interpolate_adapters


def test_fixed_merge_arithmetic_and_source_immutability():
    left = {name: {"weight": torch.tensor([0.0, 2.0])}
            for name in ("bridge", "geometry", "fusion")}
    right = {name: {"weight": torch.tensor([2.0, 4.0])}
             for name in ("bridge", "geometry", "fusion")}
    merged = interpolate_adapters(left, right)
    for name in left:
        assert torch.equal(merged[name]["weight"], torch.tensor([1.0, 3.0]))
        assert torch.equal(left[name]["weight"], torch.tensor([0.0, 2.0]))
        assert torch.equal(right[name]["weight"], torch.tensor([2.0, 4.0]))
    with pytest.raises(ValueError, match="nonfinite"):
        interpolate_adapters(left, {**right, "fusion": {"weight": torch.tensor([float("nan"), 4.0])}})
    with pytest.raises(ValueError, match="shape or dtype"):
        interpolate_adapters(left, {**right, "fusion": {"weight": torch.tensor([2.0])}})
