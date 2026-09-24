#!/usr/bin/env python3
"""Separate Stage 4 readiness decision after selected Stage 3 replication."""
from __future__ import annotations

import json
from pathlib import Path

from stage3_final_campaign import OUT, ROOT, SOURCE, cell_name, sha
from stage3_final_review import candidate_gate, read_cell


def main():
    initial = json.loads((OUT / "analysis/four_arm_review.json").read_text())
    selected = initial.get("selected_views")
    if initial.get("status") != "STAGE3_REPLICATION_REQUIRED" or selected not in (4, 8):
        raise RuntimeError("initial Stage 3 gate did not select a configuration")
    protocol = json.loads((OUT / "replication_protocol.json").read_text())
    if any(sha(ROOT / path) != expected for path, expected in protocol["source_sha256"].items()):
        raise RuntimeError("replication source hash mismatch")
    comparisons = {}
    for scene, seed in (("lego", 43), ("chair", 42)):
        candidate = read_cell(cell_name("new", selected, scene, seed), protocol)
        control = read_cell(cell_name("a5", selected, scene, seed), protocol)
        comparisons[f"{scene}_s{seed}"] = candidate_gate(candidate, control)
    ready = all(x["pass"] for x in comparisons.values())
    result = {
        "status": "READY_FOR_STAGE4" if ready else "HOLD",
        "STAGE4_READY": ready,
        "selected_views": selected,
        "initial_review_sha256": sha(OUT / "analysis/four_arm_review.json"),
        "replication_protocol_sha256": sha(OUT / "replication_protocol.json"),
        "replications": comparisons,
        "scope": "seen-view quality and correspondence; held-out scenes and downstream 3D remain Stage 4 work",
        "review_source_sha256": sha(Path(__file__)),
    }
    destination = OUT / "analysis" / "stage4_readiness.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "STAGE4_READY": ready,
                      "selected_views": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
