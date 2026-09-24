"""One synthetic check of the frozen Stage 3 decision boundary."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage3_final_review import candidate_gate, select


def test_candidate_requires_all_correspondence_and_permutation_gates():
    probes = tuple(f"lego:{i:03d}" for i in (0, 33, 66, 99))
    modes = {"correct": (30.0, .95), "target_drop": (25.0, .90),
             "mispaired_lr": (29.9, .949), "mispaired_camera": (29.9, .949),
             "shuffle_fusion": (29.9, .949),
             "target_drop_shuffle_fusion": (24.9, .899),
             "aux_permute": (29.99, .9499), "joint_permute": (29.0, .94)}
    index = {(p, mode): {"psnr": psnr, "ssim": ssim}
             for p in probes for mode, (psnr, ssim) in modes.items()}
    candidate = {"probes": probes, "index": index, "correct": {"psnr": 30, "ssim": .95},
                 "bicubic_gain": 2.0}
    control = {"probes": probes, "correct": {"psnr": 30.05, "ssim": .9505}}
    passed = candidate_gate(candidate, control)
    assert passed["pass"]
    assert select(passed, {**passed, "pass": False}, {})[0] == 4
    index[(probes[0], "aux_permute")]["psnr"] = 29.8
    assert not candidate_gate(candidate, control)["pass"]
