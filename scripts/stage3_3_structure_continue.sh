#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/linzizhuo/RL3DSR_WAN_REMO-stage3-3-structure
DATA=/data/linzizhuo/RL3DSR_WAN_REMO
OUT="$ROOT/artifacts/stage3_3_structure_20260923"
PY=/home/linzizhuo/miniconda3/envs/rl3dsr-stcdit/bin/python
LQ=/home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py
LQ_CKPT=/home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt
PARENT="$DATA/artifacts/stage3_2_camera_causality_20260913/phase_c/train/rank/seed42/stage3_step_1000.pt"
cd "$ROOT"
export PYTHONPATH=src

wait_gpu() {
  local gpu="$1" memory
  while true; do
    memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((gpu + 1))p" | tr -d ' ')"
    if [[ "$memory" =~ ^[0-9]+$ ]] && ((memory < 500)); then return; fi
    sleep 60
  done
}

run_eval_and_review() {
  local name="$1" gpu="$2" scene="$3"
  local checkpoint="$OUT/train/$name/stage3_step_1000.pt"
  local eval_dir="$OUT/eval/$name" config="$OUT/config/$name.json" manifest="$OUT/manifest/$name.json"
  wait_gpu "$gpu"
  if [[ ! -f "$eval_dir/evaluation_summary.json" ]]; then
    if [[ -d "$eval_dir" ]] && [[ -n "$(ls -A "$eval_dir")" ]]; then
      echo "$name: incomplete evaluation directory" >&2; return 1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/stage3_experiment.py seen-eval \
      --config "$config" --dataset-root "$DATA/datasets/nerf_synthetic" \
      --model-dir "$DATA/models/Wan2.1-T2V-1.3B" --lq-source "$LQ" \
      --lq-checkpoint "$LQ_CKPT" --bridge-checkpoint "$DATA/artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt" \
      --device cuda --checkpoint "$checkpoint" --seen-manifest "$manifest" \
      --subset probe --group-ids "$scene:000" "$scene:033" "$scene:066" "$scene:099" \
      --inference-seeds 3302 --modes correct correct_repeat remove target_drop shuffle_fusion \
      target_drop_shuffle_fusion mispaired_lr mispaired_camera joint_permute \
      fusion_camera_dose_half fusion_camera_dose_full shuffle_geometry \
      --save-diagnostics --save-images --output-dir "$eval_dir" \
      > "$OUT/logs/${name}_eval.log" 2>&1
  fi
  "$PY" scripts/stage3_3_structure_review.py "$name" > "$OUT/logs/${name}_review.log" 2>&1
}

train_replica() {
  local name="$1" gpu="$2" seed="$3"
  local train_dir="$OUT/train/$name" checkpoint="$OUT/train/$name/stage3_step_1000.pt"
  wait_gpu "$gpu"
  if [[ ! -f "$checkpoint" ]]; then
    if [[ -d "$train_dir" ]] && [[ -n "$(ls -A "$train_dir")" ]]; then
      echo "$name: incomplete training directory; inspect before resume" >&2; return 1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/stage3_experiment.py train \
      --config "$OUT/config/$name.json" --dataset-root "$DATA/datasets/nerf_synthetic" \
      --model-dir "$DATA/models/Wan2.1-T2V-1.3B" --lq-source "$LQ" \
      --lq-checkpoint "$LQ_CKPT" --bridge-checkpoint "$DATA/artifacts/stage1/c_campaign_20260902/c_main/best_dev.pt" \
      --device cuda --seed "$seed" --output-dir "$train_dir" \
      --init-checkpoint "$PARENT" --init-reset-fusion \
      > "$OUT/logs/${name}_train.log" 2>&1
  fi
}

finalize() {
  "$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import stage3_3_ucpe_rre_fusion as prior

root = Path(sys.argv[1])
names = ("s1_lego_seed42", "s1_lego_seed43", "s1_chair_seed42")
cells = {}
for name in names:
    path = root / "analysis" / name / "summary.json"
    cells[name] = None if not path.is_file() else json.loads(path.read_text())
passed = all(item is not None and item["pass"] for item in cells.values())
result = {"STAGE4_READY": False, "verdict": "REVIEW_STAGE4" if passed else "HOLD",
          "next": "REVIEW_STAGE4_ONLY" if passed else "STOP_AT_HARD_GATE",
          "protocol_sha256": prior.sha256_file(root / "protocol.json"),
          "cells": {name: None if item is None else {
              "pass": item["pass"], "failure_category": item["failure_category"],
              "checkpoint_sha256": item["checkpoint_sha256"],
              "summary_sha256": prior.sha256_file(root / "analysis" / name / "summary.json")
          } for name, item in cells.items()}}
prior.write_frozen_json(root / "machine_verdict.json", result)
lines = ["# W3 目标图结构项：Stage 4 前评审", "",
         f"结论：{result['verdict']}；Stage 4 HOLD。", "",
         "| 单元 | 质量 | LR | 相机三项 | 状态 |", "|---|---|---|---|---|"]
for name, item in cells.items():
    if item is None:
        lines.append(f"| {name} | — | — | — | 未运行（阶段止损） |")
    else:
        checks = item["hard_gate"]
        lines.append(f"| {name} | {checks['quality']} | {checks['mispaired_lr']} | "
                     f"{all(checks[key] for key in ('mispaired_camera', 'shuffle_fusion', 'target_drop_shuffle_fusion'))} | "
                     f"{item['failure_category'] or 'PASS'} |")
prior.write_frozen_text(root / "stage4_review" / "stage4_review.md", "\n".join(lines) + "\n")
print(json.dumps(result, indent=2))
PY
}

mkdir -p "$OUT/logs"
echo "Waiting for Lego seed42 checkpoint: $(date -Is)"
for _ in $(seq 1 360); do
  if [[ -f "$OUT/train/s1_lego_seed42/stage3_step_1000.pt" ]]; then break; fi
  sleep 60
done
if [[ ! -f "$OUT/train/s1_lego_seed42/stage3_step_1000.pt" ]]; then
  echo "Lego seed42 did not finish within six hours" >&2; exit 1
fi
wait_gpu 3
run_eval_and_review s1_lego_seed42 3 lego
"$PY" scripts/stage3_3_structure_images.py > "$OUT/logs/s1_lego_seed42_images.log" 2>&1

if ! "$PY" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' \
     "$OUT/analysis/s1_lego_seed42/summary.json"; then
  echo "Lego seed42 failed the hard gate; stopping replicas" >&2
  finalize
  exit 0
fi

(
  train_replica s1_lego_seed43 1 43
  run_eval_and_review s1_lego_seed43 1 lego
) > "$OUT/logs/s1_lego_seed43_runner.log" 2>&1 &
pid_lego=$!
(
  train_replica s1_chair_seed42 3 42
  run_eval_and_review s1_chair_seed42 3 chair
) > "$OUT/logs/s1_chair_seed42_runner.log" 2>&1 &
pid_chair=$!
set +e
wait "$pid_lego"; lego_status=$?
wait "$pid_chair"; chair_status=$?
set -e
echo "Replication exits: Lego43=$lego_status Chair42=$chair_status"
finalize
if ((lego_status || chair_status)); then exit 1; fi
