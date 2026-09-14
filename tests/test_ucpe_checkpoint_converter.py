from __future__ import annotations

import hashlib
import json

import pytest
import torch

from rl3dsr.models.wan.geometry_conditioning import (
    OFFICIAL_UCPE_COMMIT,
    FullRREConditioner,
    convert_official_ucpe_checkpoint,
)


def _official_state(module: FullRREConditioner) -> dict[str, torch.Tensor]:
    rename = {
        "camera_encoder": "cam_encoder",
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "output": "out_proj",
    }
    result = {}
    for target_name, value in module.state_dict().items():
        _, branch, layer, suffix = target_name.split(".")
        source_name = f"pipe.dit.blocks.{branch}.cam_self_attn.{rename[layer]}.{suffix}"
        result[source_name] = value.clone()
    return result


def _write_official(path, state):
    torch.save({"state_dict": state}, path)


def test_converter_maps_every_branch_tensor_and_records_file_provenance(tmp_path):
    module = FullRREConditioner(
        feature_dim=32,
        hidden_dim=8,
        attention_heads=1,
        branch_count=2,
        compression=4,
    )
    source = tmp_path / "official.ckpt"
    destination = tmp_path / "converted.pt"
    official = _official_state(module)
    _write_official(source, official)

    manifest = convert_official_ucpe_checkpoint(source, destination, module=module)

    converted = torch.load(destination, map_location="cpu", weights_only=True)
    assert list(converted["geometry"]) == list(module.state_dict())
    assert all(
        torch.equal(converted["geometry"][name], value)
        for name, value in module.state_dict().items()
    )
    provenance_path = destination.with_suffix(destination.suffix + ".provenance.json")
    assert json.loads(provenance_path.read_text()) == manifest
    assert manifest == {
        "source_commit": OFFICIAL_UCPE_COMMIT,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape"])
def test_converter_rejects_nonexact_official_branch_state(tmp_path, mutation):
    module = FullRREConditioner(
        feature_dim=32,
        hidden_dim=8,
        attention_heads=1,
        branch_count=2,
        compression=4,
    )
    state = _official_state(module)
    if mutation == "missing":
        state.pop(next(iter(state)))
    elif mutation == "unexpected":
        state["pipe.dit.blocks.2.cam_self_attn.q_proj.bias"] = torch.zeros(8)
    else:
        key = "pipe.dit.blocks.0.cam_self_attn.q_proj.weight"
        state[key] = state[key][:-1]
    source = tmp_path / "official.ckpt"
    _write_official(source, state)

    with pytest.raises(RuntimeError, match=mutation):
        convert_official_ucpe_checkpoint(source, tmp_path / "converted.pt", module=module)
