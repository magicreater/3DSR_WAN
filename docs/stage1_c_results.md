# Stage 1 实验 C 收尾：机制与可行性通过，稳定质量未通过

本轮采用双层结论：**Stage 1 feasibility PASS；robust SR quality FAIL；overall status CONDITIONAL_PASS。**

这表示 Stage 1 的 LR 条件机制和单样本端到端可行性已经得到证据支持，但稳定超过 bicubic 的质量目标尚未达到。原先登记的严格验收规则仍保留，原始严格结果仍为 `FINAL_FAIL`。

四层独立残差与时间门控 bridge 在开发 seed 1201、第 2000 步达到候选门槛（PSNR 28.39740 dB，SSIM 0.935704，LPIPS 0.041602）。冻结该 checkpoint 后，seeds 2201–2204 的最终均值为 PSNR 27.05601 dB、SSIM 0.932097、LPIPS 0.050189。相对 bicubic，SSIM 和 LPIPS 达标，但 PSNR 为 −0.58405 dB，三指标同时胜出仅 6/16（要求至少 12/16）；相对 shuffled LR 的三项均达标且为 16/16。

最终结论区分为：LR 条件因果作用 PASS，开发单样本候选 PASS，最终稳定端到端质量 FAIL。最终失败后按预先登记规则停止调参；未启动 Stage 2、Pose、LoRA、主干解冻或新 loss。

|研究目标|当前结论|依据|
|---|---|---|
|LR 条件是否真正控制 Wan 预测|PASS|correct 对 shuffled、neutral、disabled 的因果差距和消融结果均通过|
|四层时间门控 bridge 是否能完成单样本开发拟合|PASS|seed 1201 第 2000 步三项指标均达到开发候选门槛|
|纯噪声采样是否稳定超过 bicubic|FAIL|最终四 seed 平均 PSNR 低于 bicubic，三指标同时胜出 6/16|
|Stage 1 综合状态|CONDITIONAL_PASS|机制/可行性通过，稳定质量未通过|

完整中文报告、训练曲线、逐 seed/view CSV/JSON、最终四 seed × 四视角图像和证据索引见 [实验 C 报告](../artifacts/stage1/c_campaign_20260902/report_zh.md)。服务器保留完整 checkpoint、training state、源码快照、Git diff 和 SHA256 证据。
