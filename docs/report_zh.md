# 实验 C 与 Stage 1 验收报告

日期：2026-09-02。双层结论：**Stage 1 feasibility PASS；robust SR quality FAIL；overall status CONDITIONAL_PASS。**

这里的 feasibility PASS 表示 LR 条件机制和单样本端到端可行性已经得到验证；robust SR quality FAIL 表示冻结候选在预先登记的四 seed 稳定质量规则下未通过。原始严格判定 `FINAL_FAIL` 保留，不修改任何指标、阈值或 checkpoint。

C 的四层独立残差与时间门控在开发 seed 1201 的第 2000 步通过候选门槛；冻结该 checkpoint 后，最终 seeds 2201–2204 未通过像素保真度和配对胜出数门槛。按预先约定的规则，最终失败后停止调参，保留失败结论。没有启动 Stage 2、Pose、LoRA、主干解冻或新 loss。

## 双层目标判定

|研究目标|结论|证据解释|
|---|---|---|
|LR 条件是否真正控制 Wan 预测|PASS|correct 相对 shuffled、neutral、disabled 的因果差距成立，原 causal gate 通过。|
|四层时间门控 bridge 是否能完成单样本开发拟合|PASS|开发 seed 1201 在第 2000 步达到 PSNR 28.39740、SSIM 0.935704、LPIPS 0.041602。该 seed 用于候选选择。|
|冻结候选在未参与选择的 seeds 上稳定超过 bicubic|FAIL|最终平均 PSNR 27.05601，低于 bicubic 27.64006；三指标同时胜出 6/16。最终 seeds 用于稳定性验收。|
|Stage 1 综合状态|CONDITIONAL_PASS|feasibility PASS 与 robust SR quality FAIL 并存，不能合并成完整严格 PASS。|

开发 seed 只用于候选选择，最终 seeds 只用于稳定性验收；二者不能合并成一个单一 PASS。

## 1. 实际执行与边界

- 数据：NeRF Synthetic chair，训练视角 `[0,33,66,99]`，单样本 V=4；HR 256×256，LR 64×64，4× bicubic antialias。图中 view 0–3 对应上述四个原始视角。
- 冻结 Wan VAE、DiT、text embedding、FlashVSR projector；固定全零 text context。
- 训练 bridge：blocks 0、1、2、3 输入处分别注入独立 FP32 零初始化投影；256 维 Wan sinusoidal timestep embedding 经 SiLU 与零初始化线性层生成 `1+tanh(...)` 门控。采样每一步用实际 Wan timestep 重算。
- 11,022,336 个可训练参数；A/B 旧单层 bridge 为 2,360,832。projector 逐视角处理与 token 顺序保留，4D 保留原生时间路径。
- seed=42，AdamW，weight decay 0.01，梯度裁剪 1；前 750 步学习率 3e-4，之后 3e-5；balanced sigma；每 250 步保存并解码。
- 推理从纯高斯噪声开始，官方 UniPC 50 步、shift=5。正确／打乱／中性／关闭条件使用配对噪声。HR 不用于生产采样初始化。
- GPU0 RTX4090 串行运行。全流程按 GPU 子进程完整墙钟时间保守计费：**818.89 秒，约 13.65 分钟 / 0.2275 GPU 小时**，包括 smoke、续训 smoke、两次预检、主实验、四-seed 最终验收和最终回归，低于 6 GPU 小时。

主实验 2000 步开发通过后，按执行规则直接冻结候选。多层无门控、单层时间门控、permutation sigma、八层深度扩展均未触发；因此本轮只证明组合结构的效果，不能分离多层与门控各自贡献。没有因为最终 seeds 失败而继续尝试这些配置。

## 2. A/B/C 开发结果与训练曲线

下表均使用开发 seed 1201、2000 步。各自按预定排序选出的最佳 checkpoint 恰好也都是 2000 步；[CSV](analysis/experiment_summary.csv)仍分别保存“同一步数”和“各自最佳”两种比较。

|配置 / 参考|PSNR ↑|SSIM ↑|LPIPS ↓|
|---|---:|---:|---:|
|Bicubic|27.64006|0.908945|0.124076|
|VAE 重建参考|33.33161|0.970262|0.015187|
|A：旧单层，permutation|25.03954|0.893256|0.083027|
|B：旧单层，balanced|25.16557|0.893307|0.091262|
|C：四层＋时间门控，balanced|28.39740|0.935704|0.041602|

C 相比同预算 B 的开发 PSNR 提高 3.23183 dB，SSIM、LPIPS 同时改善。相比 bicubic，C 开发 PSNR +0.75734 dB、SSIM +0.026759、LPIPS 降低 66.47%。这支持组合升级有效，但开发结果不替代最终验收。

![A/B/C 开发曲线](analysis/across_experiment_curves.png)

250→500→750→1000→1250→1500→1750→2000 步的 C 开发 PSNR 为 24.5175、25.1558、26.3245、27.3300、27.5399、27.8734、27.8252、28.3974 dB。SSIM 较早达到门槛，PSNR 是候选选择瓶颈。750 步降学习率后解码质量继续改善，但不能仅凭这一条轨迹断言学习率变化具有独立因果贡献。

[完整训练曲线](c_main/training_curves.png)分别记录 loss、sigma、学习率、梯度、显存、每步时间、三项解码指标与条件消融差距；原始点保留，loss 的 50 步均线单独标注。固定 sigma 验证使用相同噪声和条件：[分 sigma loss](analysis/selected_fixed_sigma_losses.png)。[门控和残差诊断](c_main/bridge_diagnostics.png)保存各块 gate min/mean/max 与残差／输入 RMS 比；这些是分布摘要，不是完整直方图，不能据此计算门控饱和比例。峰值 allocated 显存 6399.06 MiB。

## 3. 冻结候选后的最终验收

最终四个噪声 seeds × 四视角，96 条六条件指标记录完整；其中 correct 有 16 个配对组合。逐行重算严格判定与原 JSON 完全一致。

|最终聚合|PSNR ↑|SSIM ↑|LPIPS ↓|
|---|---:|---:|---:|
|C correct|27.05601|0.932097|0.050189|
|Bicubic|27.64006|0.908945|0.124076|
|Shuffled LR|15.73409|0.729549|0.238736|
|Disabled|4.82886|0.244875|0.837148|

|门槛|实际结果|判定|
|---|---|---|
|比 bicubic PSNR ≥ +0.25 dB|−0.58405 dB；距门槛 0.83405 dB|FAIL|
|比 bicubic SSIM ≥ +0.005|+0.023152|PASS|
|比 bicubic LPIPS 至少降低 5%|降低 59.55%|PASS|
|比 shuffled PSNR ≥ +1 dB / SSIM ≥ +0.02 / LPIPS 降低 ≥10%|+11.32192 dB / +0.202548 / 降低 78.98%|PASS|
|三指标同时胜过 bicubic ≥12/16|6/16|FAIL|
|三指标同时胜过 shuffled ≥12/16|16/16|PASS|
|四个 seed 均三指标优于 bicubic|2201、2204 的 PSNR 方向不满足|FAIL|
|每个 seed 三指标优于 shuffled、disabled|全部满足|PASS|

### Seed 波动

|seed|PSNR|SSIM|LPIPS|PSNR 比 bicubic|三指标胜出 / 4|
|---|---:|---:|---:|---:|---:|
|2201|25.34722|0.927547|0.058186|−2.29284|0|
|2202|27.75727|0.932902|0.052732|+0.11720|2|
|2203|28.36376|0.934997|0.041052|+0.72370|4|
|2204|26.75578|0.932941|0.048786|−0.88428|0|

四个 seed 均值的样本标准差：PSNR **1.31809 dB**，SSIM 0.003187，LPIPS 0.007208；PSNR 极差 **3.01654 dB**。仅有 seed 2203 的聚合 PSNR 达到 +0.25 dB，开发点未代表稳定的初始化表现。

### 视角与最差组合

|view / 原视角|跨 seed 平均 PSNR|SSIM|LPIPS|三指标胜出 / 4|
|---|---:|---:|---:|---:|
|0 / 0|28.13292|0.956091|0.041922|1|
|1 / 33|28.83675|0.962433|0.038802|1|
|2 / 66|25.57462|0.905155|0.060878|2|
|3 / 99|25.67975|0.904710|0.059155|2|

最差 PSNR、SSIM 和 LPIPS 均出现在 seed 2201 / view 2：**23.93968 dB / 0.899451 / 0.068432**。四个视角跨 seed 的平均 PSNR 都低于各自 bicubic。全部 16 个组合的 SSIM 和 LPIPS 都优于 bicubic，PSNR 仅 6 个胜出，说明结构／感知指标改善与像素保真度并不一致。

明细：[最终原始指标](analysis/final_raw_metrics.csv)、[逐 seed 汇总](analysis/final_seed_summary.csv)、[逐 view 汇总](analysis/final_view_summary.csv)、[全部配对差值](analysis/per_seed_view_metrics.csv)、[重新聚合 JSON](analysis/final_summary.json)。这些 seeds 是同一场景的噪声重复，16 个组合不是独立场景；不进行场景泛化显著性推断。

## 4. 定性检查

已逐一检查四个最终 seeds 的全部四视角完整图与纹理放大图，以及开发 seed 的 B/C 固定中心裁剪和统一误差图。不是仅展示最好 seed。

- **轮廓与细结构**：C 保留正确视角和整体椅子形状，椅腿、扶手及外框较 bicubic 清晰；相较 B，外框附近颗粒与杂色减少。极细腿端、花饰和高频边缘仍有局部偏差，不能称为逐像素恢复。
- **绿色织物**：C 恢复可辨认的纹理，但局部花纹与 HR 并不完全重合。B/C 误差图在织物和边界处仍有明显残差；LPIPS 的改善不能证明细节全部忠实。
- **颜色与对比度**：开发图 C 改善 B 的椅背偏绿问题；最终 seed 2201 的织物明显偏深、对比更强，2204 也较鲜艳。2203 的大面积颜色更接近 HR。它们与 PSNR 对 seed 敏感的现象一致；这里只是观察关联，尚未分解误差来源。
- **噪声与条件作用**：correct 输出整体干净；shuffled 输出对应其他椅子视角，disabled 则为与椅子无关的色场和纹理。条件因果作用很强，但这本身不构成稳定 SR 质量通过。

![开发 seed 全部四视角 HR/LR/bicubic/B/C/消融](analysis/qualitative_all_views.png)

[B/C 统一中心裁剪](analysis/qualitative_fixed_center_crops.png)；[B/C 绝对 RGB 误差图，统一 0–0.25](analysis/qualitative_absrgb_error_0_025.png)；[最终全部 16 个组合](analysis/final_all_seeds_views.png)；[最终统一中心裁剪](analysis/final_all_seeds_fixed_center_crops.png)。误差图采用 PNG 量化后的 RGB 每通道误差可视化；正式数值来自浮点指标 JSON。B 图已核对来自 `optimized_sigma_balanced` 第 2000 步。

## 5. 验证、复现与证据身份

- 完整回归 **176 passed**，52 条上游 autocast 弃用警告；报告模块随后追加的 **20 项 CPU 检查通过**（与全套有重叠，不相加）。包含多块注入与异常 hook 清理、关闭/零残差、首次投影梯度及更新后 gate 梯度、冻结模块无梯度、旧 A/B 兼容与新结构严格加载、真实 Wan 3D/4D 回归。
- 主实验正式权重审计：零残差与 baseline max/mean diff = 0；首次投影梯度有限，临时投影更新后各 gate 获得非零梯度，审计后恢复初始 bridge；冻结模块梯度计数均为 0。
- checkpoint 重载 max/mean diff = 0；原 causal gate 通过，四个 sigma 下 correct loss 均优于 shuffled、neutral、disabled。
- Oracle UniPC final max error = **0.0009951591 < 0.001**，mean error = **0.0002121160 < 0.0003**。保持原门槛，max 余量较小。
- 完整性审计核对 C 的 2000 步、smoke 的 8 步 JSONL/CSV 逐字段一致；10 组 checkpoint/state 与三个源码归档的 194 个文件 SHA 通过；冻结候选与 best 一致。smoke 原索引备份后，仅从 checkpoint/launch 补齐遗漏的来源和父 checkpoint 元数据。
- 历史限制：smoke 启动完成记录的训练耗时漏计末尾评估，原记录保留；预算统一采用 campaign 完整子进程墙钟时间，未少算 GPU 预算。首次 manifest 与首 launch 一致，但没有独立的旧文件摘要，故不声称历史上逐字节从未变化。
- 原始数据集和基础模型保持原位。服务器完整证据位于 `/data/linzizhuo/RL3DSR_WAN_REMO/artifacts/stage1/c_campaign_20260902`；报告、图表、CSV/JSON 和绘图脚本同步到本地相应目录。

冻结候选：`c_main/3d_step_2000.pt`，SHA256 `af964951c53f35e0bed55a1af5229546fb63a4732e84f96d97105fa9a06ff83d`。

主实验实际源码内容集合 SHA256：`0553d24ddb2dffc41ef27ccda2d7273f37b7fc318bc945f02a7eebfaf2fe711b`；归档 `c_main/source_0553d24ddb2d.tar.gz`，配套逐文件 SHA、Git diff、status。Git revision 为 `e4c96dbbafd4f239254be3a8d07f8a6335501a18`，但当时存在未提交代码，因此以源码快照为实际运行身份。报告修正另有 closeout 源码归档，未改训练时归档。

运行恢复身份保存于 `run_manifest.json`、`launches.jsonl`、checkpoint payload 和 training state；新旧 bridge 不隐式转换，optimizer 不跨不兼容配置复用。主实验本次未续训，恢复路径通过 4→8 步 smoke 验证日志与状态连续；没有额外宣称已完成长程 uninterrupted-vs-resumed 位级等价实验。

在服务器仓库根目录，以既有环境重建报告数据和图表（CPU）：

```bash
export PYTHONPATH=src:scripts
PY=/home/linzizhuo/miniconda3/envs/rl3dsr-stcdit/bin/python
$PY scripts/stage1_c_analysis.py --campaign artifacts/stage1/c_campaign_20260902
$PY scripts/stage1_c_summary.py --campaign artifacts/stage1/c_campaign_20260902
```

逐步曲线可由 `scripts/stage1_reporting.py` 的 `plot_training_curves` 读取 `train_steps.jsonl`、`checkpoint_metrics.jsonl` 重建。缺失旧字段保持 unknown，不补零；缺失/重复配对、非有限指标或聚合不一致会报错或被标为 invalid。

## 6. 借鉴依据与本轮结论

[ControlNet 的零初始化连接](https://github.com/lllyasviel/ControlNet/blob/main/docs/faq.md)提供了初始不扰动预训练网络的思路；[DiT 源码](https://github.com/facebookresearch/DiT/blob/main/models.py)提供 timestep 调制参考。本项目的 `1+tanh`、四个输入残差与全冻结 Wan 是具体实验设计，不是对这些方法的完整复现。

[FlashVSR](https://arxiv.org/html/2510.12747v1)提供 LR 特征注入路线，但其训练包含 LoRA。因此不能从 FlashVSR 的结果推导本项目全冻结四层 bridge 必然足够。C 实证表明该组合在相同开发 seed 上明显优于旧单层桥接，同时暴露了最终初始噪声下的颜色／像素保真度波动。

本轮结论为 **LR 因果作用 PASS；开发单样本候选 PASS；最终四-seed 稳定端到端质量 FAIL；Stage 1 未结束**。下一次若重新设计实验，应采用新的预先登记开发/验证协议，并将本次四个最终 seeds 视为已使用数据；本次不据其结果继续调参或放宽阈值。
