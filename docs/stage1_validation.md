# Stage 1 — LR Fidelity Conditioning Validation

## Verdict

**PASS — LR 因果可行性；端到端 Stage 1 收尾见实验 C 报告**

在 Wan2.1-T2V-1.3B 的 VAE、DiT、text embedding 全部冻结时，训练 2,360,832 个 LR bridge 参数即可让真实 Wan prediction 稳定依赖正确 LR observation。3D 与 4D 都满足：

- 相同 `x_t / timestep / noise / text / checkpoint`，只改变 LR 会显著改变 prediction；
- mean `loss(correct LR) < loss(shuffled LR)`；
- mean `loss(correct LR) < loss(neutral LR)`；
- 关闭 adapter 后误差增大；
- Stage 0 的 3D independent-view VAE 与 4D native temporal VAE 语义未改变。

## Repository Changes

- `src/rl3dsr/models/wan/lq_conditioning.py`: FlashVSR LQ projector 的本地 checkpoint-compatible 定义、3D/4D 对齐、zero bridge、adapter checkpoint。
- `src/rl3dsr/models/wan/flow.py`: 与 Wan flow-matching 一致的最小训练 pair 和 FP32 MSE。
- `src/rl3dsr/models/wan/dit.py`: 在 Wan patch embedding 后、block 0 前接受可选 token residual；默认 forward 不变。
- `src/rl3dsr/data/temporal_fixture.py`: deterministic coherent-motion 4D fixture。
- `scripts/stage1_experiment.py`: 数据缓存、baseline/gradient audit、micro-overfit、causal evaluation、checkpoint/reload。
- `tests/test_stage1_conditioning.py`, `tests/test_stage1_wan_injection.py`, `tests/test_wan_stage1.py`: lightweight 与 real-checkpoint regression。
- `tests/test_temporal_fixture.py`、package exports、`.gitignore`: 配套更新。

未修改 `third_party/wan2_1/`，未加入 pose、LoRA 或 Stage 2+ 功能。

## Final Architecture

```text
HR target ──Wan VAE──> clean latent z0 ──flow noise──> xt
   │
   └─4x bicubic+antialias downsample──> LR 32x32
          └─bicubic resize to conditioning 128x128
             └─frozen FlashVSR Causal_LQ4x_Proj (1 output layer)
                └─tokens [B,L,1536]
                   └─zero-init FP32 Linear(1536,1536), trainable
                      └─add after Wan patch embedding, before block 0
                         └─frozen Wan DiT prediction
```

采用现有 FlashVSR `LQ_proj_in.ckpt`，因为它本身就是为 Wan2.1 1.3B 构造的 causal LR projector，输出维度 1536 与 Wan token width 一致。官方 FlashVSR 代码也使用 `Causal_LQ4x_Proj(3,1536,layer_num=1)`；本地等价实现对相同 checkpoint/input 与参考实现逐元素完全一致：max/mean abs diff 都为 0。

只注入 block 0 前的一处 residual，避免复制 Wan block 或训练 287.8M projector。zero-init bridge 让初始 token residual 精确为零，保持 pretrained baseline。这个 residual 接口以后可以并列接收 Pose Adapter，但本 Stage 没有 pose 逻辑。

### Alignment

- **3D:** `[B,3,V,LR_H,LR_W] → [B*V,3,1,H,W]`，每个 view 单独过 LQ projector，再按 Wan 的 `F/H/W` token 顺序 regroup 为 `[B,V*(h/2)*(w/2),1536]`。LQ projector 不跨 view 混合。
- **4D:** `[B,3,T,LR_H,LR_W]` 保持 native video，不 flatten frame。conditioning branch 在开头外加 4 个 first-frame warm-up copy，以补偿 FlashVSR causal projector 的首 chunk warm-up；原始 T=1/5/9/17 对齐 Wan latent T′=1/2/3/5。
- target 3D VAE 仍逐 view encode；target 4D VAE 仍使用 native temporal encode。warm-up copy 只存在于 LR condition branch。

## Training Semantics

使用实际 Wan/FlowMatch 语义：

```text
x_sigma = (1 - sigma) * z_clean + sigma * epsilon
model timestep = 1000 * sigma
training target = epsilon - z_clean
loss = FP32 MSE(prediction, target)
```

`sigma = sigmoid(N(0,1))`。所有样本使用相同的 deterministic all-zero BF16 context `[B,512,4096]`；text 不训练。

LR 仅由 HR 经独立的 deterministic 4x bicubic antialiased downsample 得到。condition branch 后续只读取 LR tensor并做 spatial resize；target latent 只进入 diffusion target path。测试确认 LR 与 HR 不共享 storage，且没有把 HR tensor传入 projector。

## Parameters and Gradient Isolation

| Module | Parameters | Trainable |
|---|---:|---:|
| Wan VAE | 126,892,531 | 0 |
| Wan DiT including text embedding | 1,418,996,800 | 0 |
| FlashVSR LQ projector | 287,845,888 | 0 |
| LR bridge | 2,360,832 | 2,360,832 |
| **Total** | **1,836,096,051** | **2,360,832 (0.1286%)** |

Optimizer parameter IDs 与 `bridge.weight`、`bridge.bias` 精确相等。真实 backward audit 中：

- 3D bridge weight/bias grad norm: 0.3672 / 0.1095；
- 4D bridge weight/bias grad norm: 0.8528 / 0.5530；
- Wan VAE、Wan DiT、text embedding、LQ projector 的 non-null gradient tensor count 全部为 0。

## Baseline Preservation

固定 model、seed、latent、noise、timestep、text 和 shape，在训练前比较原 Stage 0 Wan 与存在 LR branch 但 bridge 为 zero-init 的模型：

| Mode | Max abs diff | Mean abs diff | Tolerance | Result |
|---|---:|---:|---:|---|
| 3D | 0.0 | 0.0 | max 1e-6, mean 1e-7 | PASS |
| 4D | 0.0 | 0.0 | max 1e-6, mean 1e-7 | PASS |

## 3D Micro-overfit

- Dataset: NeRF Synthetic `chair`, train split。
- 8 个 evenly spaced real views，索引 `[0,14,28,42,56,70,84,99]`；构成两个 V=4 observations。
- HR/LR: 128×128 / 32×32；4x bicubic antialiased；conditioning 128×128。
- Wan latent: `[1,16,4,16,16]`；LQ tokens: `[1,256,1536]`。
- BF16 Wan/projector，FP32 bridge，AdamW lr 3e-4，weight decay 0.01，seed 42。
- 50 steps；loss first/last/min = 0.1161 / 0.0837 / 0.0756。

固定 2 observations × 4 sigmas (`0.2,0.5,0.8,0.95`) 的 final evaluation：

| Condition | Mean flow loss |
|---|---:|
| Correct LR | **0.10164** |
| Shuffled LR | 0.12700 |
| Neutral LR | 0.16017 |
| Adapter disabled | 0.16458 |

Correct LR 比 shuffled 低 19.96%，比 neutral 低 36.54%，比 disabled 低 38.24%。8/8 evaluation rows 都满足 correct 最优。

Same-noise intervention (`correct` vs `shuffled`)：mean abs 0.07533，max abs 1.79102，relative L2 0.10196，远高于 numerical noise。

## 4D Micro-overfit

- Fixture: 4 个 deterministic coherent-motion clips，seed 0–3；平移纹理背景与移动彩色物体，非独立随机 frame。
- T=9，native Wan VAE latent T′=3；没有把 frame flatten 成图片。
- HR/LR: 128×128 / 32×32；4x bicubic antialiased；conditioning 128×128。
- Wan latent: `[1,16,3,16,16]`；LQ tokens: `[1,192,1536]`。
- 与 3D 相同 optimizer/precision/seed；50 steps；loss first/last/min = 0.4894 / 0.09052 / 0.08954。

固定 4 clips × 4 sigmas 的 final evaluation：

| Condition | Mean flow loss |
|---|---:|
| Correct LR | **0.25573** |
| Shuffled LR | 0.28215 |
| Neutral LR | 0.29661 |
| Adapter disabled | 0.49147 |

Correct LR 比 shuffled 低 9.36%，比 neutral 低 13.78%，比 disabled 低 47.97%。13/16 rows 同时满足 correct 优于 shuffled、neutral 和 disabled，direction fraction 0.8125，超过 0.75 门槛。

Same-noise intervention：mean abs 0.17345，max abs 2.55469，relative L2 0.18093。

## Adapter Ablation and Checkpointing

两种模式中，adapter disabled 的 aggregate loss 都显著高于 enabled correct LR。Neutral LR 也劣于 correct LR，说明收益不是单纯来自非零 residual 或 bridge bias。

只保存 bridge 的 adapter checkpoint，各 9.1 MiB；未复制 Wan 或 frozen LQ projector：

- `/data/linzizhuo/RL3DSR_WAN_REMO/artifacts/stage1/repro/3d_adapter.pt`
- `/data/linzizhuo/RL3DSR_WAN_REMO/artifacts/stage1/repro/4d_adapter.pt`

另保存 optimizer/training state。两类 checkpoint 都通过 save → zero bridge → 带 expected config、Wan identity 与 LQ SHA256 校验的 reload → same input prediction 检查，max/mean abs diff 为 0，bit-exact。PASS 状态同时要求 causal evaluation 和 reload equivalence 通过。

## Reproducibility

- Seed: 42；训练 sigma/noise 与 evaluation noise 使用分离 generator。
- LQ checkpoint SHA256: `d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2`。
- FlashVSR source commit: `b527c6f285fb30df530f5febc8b45764a789c961`。
- Wan vendored commit: `9737cba9c1c3c4d04b33fcad41c111989865d315`。
- Wan checkpoint identity 写入 adapter metadata。
- 当前实现的完整 repeat run 中，3D 与 4D 再次 PASS；direction fraction 分别为 1.0 和 0.8125。
- Peak allocated GPU memory: 3D 4697.7 MiB；4D 4667.4 MiB，单张 RTX 4090。

### Reproduction commands

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/linzizhuo/miniconda3/envs/rl3dsr-stcdit/bin/python scripts/stage1_experiment.py --kind 3d --model-dir models/Wan2.1-T2V-1.3B --scene datasets/nerf_synthetic/chair --lq-source /home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py --lq-checkpoint /home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt --output-dir artifacts/stage1/repro --max-steps 50 --eval-interval 25
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/linzizhuo/miniconda3/envs/rl3dsr-stcdit/bin/python scripts/stage1_experiment.py --kind 4d --model-dir models/Wan2.1-T2V-1.3B --scene datasets/nerf_synthetic/chair --lq-source /home/linzizhuo/rl3dsr-new/data/rl3dsr/external/FlashVSR/examples/WanVSR/utils/utils.py --lq-checkpoint /home/linzizhuo/rl3dsr-new/data/models/rl3dsr/FlashVSR-v1.1/LQ_proj_in.ckpt --output-dir artifacts/stage1/repro --max-steps 50 --eval-interval 25
```

## Regression and Tests

- Lightweight suite: 107 passed, 7 real-checkpoint tests skipped when model env vars are absent。
- Stage 1 real-checkpoint integration: 2 passed；同时验证本地 LQ projector 与 reference FlashVSR 实现逐元素完全一致，并验证 3D/4D LQ alignment、真实 VAE/DiT、zero baseline 和冻结状态。
- Stage 0 real validation script PASS：
  - 3D V=1/4/8/16 shape 与 real DiT forward 正常；cross-view changed delta 0.25635，unchanged views delta 0；
  - 4D T=1/5/9/17 → T′=1/2/3/5，native temporal VAE 与 real DiT forward 正常；
  - temporal impulse validation 仍执行成功。
- Raw metrics、training state 和 Stage 0 regression report 位于 ignored `artifacts/stage1/`。

## Design Basis

- [FlashVSR](https://arxiv.org/abs/2510.12747) 与其[官方 WanVSR inference](https://github.com/OpenImagingLab/FlashVSR/blob/main/examples/WanVSR/infer_flashvsr_v1.1_full.py)支持复用已训练的 causal `LQ_proj_in`，而不是从零设计大 LR encoder。
- [ControlNet](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper)支持用 zero-initialized residual 保留 frozen pretrained baseline。
- [T2I-Adapter](https://arxiv.org/abs/2302.08453)支持冻结大生成模型、只训练小 condition adapter 的路线。
- [Wan technical report](https://arxiv.org/abs/2503.20314)与[官方 repository](https://github.com/Wan-Video/Wan2.1)支持保留 Wan VAE/DiT 与 flow-matching formulation；最终 objective 以 vendored scheduler/model semantics 为准。

## Limitations

- 这是 feasibility micro-overfit，不证明跨 scene、跨 degradation 或真实分布泛化。
- 4D 使用 deterministic synthetic coherent-motion clips；仓库尚无真实 pose-annotated 4D dataset，但 Stage 1 不使用 pose。
- 4D 在两个 `sigma=0.95` rows 中 correct 没有胜过所有 ablation；aggregate、13/16 direction 和 repeat run 均通过预先设定门槛，但高噪声端仍是后续监督训练要观察的弱点。
- CUDA BF16/attention backward 不是 bitwise deterministic：相同 seed 的完整重新训练指标有小幅差异，但两次独立 run 都 PASS；固定 checkpoint 的 forward 与 checkpoint reload 是 bit-exact。
- 复用的 frozen LQ projector 有 287.8M 参数，虽然不训练且峰值显存低，但它不是极小的运行时模块；本 Stage 证明的是 2.36M trainable bridge 的可行性。
- 使用 deterministic neutral context，没有加载官方 empty T5 embedding。

没有开始 Pose Adapter、Stage 2、cross-view fusion、LoRA 或大规模训练。

## Decoded-space follow-up

See [Stage 1 decoded-space evaluation](stage1_decoded_evaluation.md) for paired RGB metrics, contact sheets, videos, and the short-trajectory limitation analysis.


## Stage 1 C closeout

实验 C 的四层时间门控 bridge 在开发 seed 上通过候选门槛，冻结 checkpoint 的四 seed 最终严格质量验收仍为 **FAIL**。双层状态为：**Stage 1 feasibility PASS；robust SR quality FAIL；overall status CONDITIONAL_PASS**。本状态承认 LR 条件机制与单样本可行性已经验证，同时保留原严格质量规则及其 `FINAL_FAIL` 结果。完整中文报告、曲线、逐 seed/view 指标和视觉证据见 `artifacts/stage1/c_campaign_20260902/report_zh.md`。
