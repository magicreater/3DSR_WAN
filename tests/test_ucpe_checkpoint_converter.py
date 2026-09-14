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


def _official_state(
    module: FullRREConditioner,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    rename = {
        "camera_encoder": "cam_encoder",
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "output": "out_proj",
    }
    result = {}
    expected_markers = {}
    for marker, (target_name, value) in enumerate(module.state_dict().items(), start=1):
        _, branch, layer, suffix = target_name.split(".")
        source_name = f"pipe.dit.blocks.{branch}.cam_self_attn.{rename[layer]}.{suffix}"
        marker_value = float(marker)
        result[source_name] = torch.tensor(marker_value, dtype=value.dtype).as_strided(
            value.shape, (0,) * value.ndim
        )
        expected_markers[target_name] = marker_value
    return result, expected_markers


def _write_official(path, state):
    torch.save({"state_dict": state}, path)


def test_converter_maps_every_branch_tensor_and_records_file_provenance(tmp_path):
    with torch.device("meta"):
        module = FullRREConditioner()
    source = tmp_path / "official.ckpt"
    destination = tmp_path / "converted.pt"
    official, expected_markers = _official_state(module)
    _write_official(source, official)

    manifest = convert_official_ucpe_checkpoint(
        source,
        destination,
        module=module,
        asserted_source_commit=OFFICIAL_UCPE_COMMIT,
    )

    converted = torch.load(destination, map_location="cpu", weights_only=True)
    assert list(converted["geometry"]) == list(module.state_dict())
    assert {
        name: tuple(value.shape) for name, value in converted["geometry"].items()
    } == {
        name: tuple(value.shape) for name, value in module.state_dict().items()
    }
    for target_name, marker in expected_markers.items():
        value = converted["geometry"][target_name]
        assert value[(0,) * value.ndim].item() == marker
    assert converted["asserted_source_commit"] == OFFICIAL_UCPE_COMMIT
    assert "source_commit" not in converted
    provenance_path = destination.with_suffix(destination.suffix + ".provenance.json")
    assert json.loads(provenance_path.read_text()) == manifest
    assert manifest == {
        "asserted_source_commit": OFFICIAL_UCPE_COMMIT,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape"])
def test_converter_rejects_nonexact_official_branch_state(tmp_path, mutation):
    with torch.device("meta"):
        module = FullRREConditioner()
    state, _ = _official_state(module)
    if mutation == "missing":
        state.pop(next(iter(state)))
    elif mutation == "unexpected":
        state["pipe.dit.blocks.30.cam_self_attn.q_proj.bias"] = torch.zeros(192)
    else:
        key = "pipe.dit.blocks.0.cam_self_attn.q_proj.weight"
        state[key] = torch.zeros(1).as_strided((191, 1536), (0, 0))
    source = tmp_path / "official.ckpt"
    _write_official(source, state)

    with pytest.raises(RuntimeError, match=mutation):
        convert_official_ucpe_checkpoint(
            source,
            tmp_path / "converted.pt",
            module=module,
            asserted_source_commit=OFFICIAL_UCPE_COMMIT,
        )


def test_converter_rejects_noncanonical_branch_count_before_mapping(tmp_path):
    source = tmp_path / "official.ckpt"
    _write_official(source, {})
    module = FullRREConditioner(
        feature_dim=32,
        hidden_dim=8,
        attention_heads=1,
        branch_count=2,
        compression=4,
    )
    with pytest.raises(RuntimeError, match="canonical.*30 branches"):
        convert_official_ucpe_checkpoint(
            source,
            tmp_path / "converted.pt",
            module=module,
            asserted_source_commit=OFFICIAL_UCPE_COMMIT,
        )


def test_converter_requires_the_pinned_source_commit_assertion(tmp_path):
    source = tmp_path / "official.ckpt"
    _write_official(source, {})
    with pytest.raises(RuntimeError, match="source commit assertion"):
        convert_official_ucpe_checkpoint(
            source,
            tmp_path / "converted.pt",
            asserted_source_commit="not-the-pinned-commit",
        )
