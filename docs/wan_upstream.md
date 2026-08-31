# Wan2.1 Stage 0 Upstream

Stage 0 vendors the VAE, DiT, and attention modules from the official
`Wan-Video/Wan2.1` repository at commit
`9737cba9c1c3c4d04b33fcad41c111989865d315`.

The checkpoint is external and must be supplied with
`RL3DSR_WAN_MODEL_DIR`; the expected files are `config.json`,
`diffusion_pytorch_model.safetensors`, and `Wan2.1_VAE.pth`.

The vendored implementation is used without architectural changes. The RL3dSR
wrappers only convert 3D independent-view and 4D native-video tensor layouts.
All Wan parameters are frozen and Stage 0 never constructs an optimizer.

The T2V text encoder is intentionally not loaded in Stage 0. DiT smoke tests
use a deterministic neutral context with shape `[B, 512, 4096]`.
