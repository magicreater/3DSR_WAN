# Stage 0 Validation Report

## Status

This report is generated from real Wan2.1 VAE and DiT execution. No training, optimizer, backward pass, adapter, pose encoder, or geometry input is used.

## Environment and Sources

- Wan upstream commit: `9737cba9c1c3c4d04b33fcad41c111989865d315`
- Checkpoint: `/data/linzizhuo/RL3DSR_WAN_REMO/models/Wan2.1-T2V-1.3B`
- Runtime: `3.10.19`, PyTorch `2.4.0+cu124`, CUDA `12.4`, device `NVIDIA GeForce RTX 4090`
- Datasets: NeRF Synthetic `/data/linzizhuo/RL3DSR_WAN_REMO/datasets/nerf_synthetic/chair`; Mip-NeRF 360 `/data/linzizhuo/RL3DSR_WAN_REMO/datasets/360_v2/counter`
- Camera: `T_world_from_camera`, OpenCV camera axes (+X right, +Y down, +Z forward); source world units preserved.
- DiT context: deterministic neutral tensor `[B,512,4096]`; T5 is not loaded.

## Measured Results

```json
{
  "platform": "Linux-5.4.0-216-generic-x86_64-with-glibc2.31",
  "python": "3.10.19",
  "torch": "2.4.0+cu124",
  "cuda": "12.4",
  "gpu": "NVIDIA GeForce RTX 4090",
  "wan_upstream_commit": "9737cba9c1c3c4d04b33fcad41c111989865d315",
  "checkpoint": "/data/linzizhuo/RL3DSR_WAN_REMO/models/Wan2.1-T2V-1.3B",
  "camera_convention": "T_world_from_camera, canonical OpenCV +X right +Y down +Z forward",
  "datasets": {
    "nerf_synthetic": "/data/linzizhuo/RL3DSR_WAN_REMO/datasets/nerf_synthetic/chair",
    "mipnerf360": "/data/linzizhuo/RL3DSR_WAN_REMO/datasets/360_v2/counter"
  },
  "real_rgb_samples": {
    "nerf_first": [
      800,
      800,
      3
    ],
    "mip_first": [
      519,
      779,
      3
    ]
  },
  "3d": {
    "source_observations": 200,
    "mip_test_observations": 30,
    "V=1": {
      "input": {
        "shape": [
          1,
          3,
          1,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "latents": {
        "shape": [
          1,
          16,
          1,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded": {
        "shape": [
          1,
          3,
          1,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "dit": {
        "shape": [
          1,
          16,
          1,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded_range": [
        -0.78515625,
        1.0
      ],
      "decoded_std": 0.25299638509750366
    },
    "V=4": {
      "input": {
        "shape": [
          1,
          3,
          4,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "latents": {
        "shape": [
          1,
          16,
          4,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded": {
        "shape": [
          1,
          3,
          4,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "dit": {
        "shape": [
          1,
          16,
          4,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded_range": [
        -0.83984375,
        1.0
      ],
      "decoded_std": 0.25969111919403076
    },
    "V=8": {
      "input": {
        "shape": [
          1,
          3,
          8,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "latents": {
        "shape": [
          1,
          16,
          8,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded": {
        "shape": [
          1,
          3,
          8,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "dit": {
        "shape": [
          1,
          16,
          8,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded_range": [
        -0.83984375,
        1.0
      ],
      "decoded_std": 0.2724159061908722
    },
    "V=16": {
      "input": {
        "shape": [
          1,
          3,
          16,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "latents": {
        "shape": [
          1,
          16,
          16,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded": {
        "shape": [
          1,
          3,
          16,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "dit": {
        "shape": [
          1,
          16,
          16,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded_range": [
        -0.83984375,
        1.0
      ],
      "decoded_std": 0.3002243638038635
    },
    "cross_view_isolation": {
      "changed_view": 2,
      "changed_view_max_abs_delta": 0.25634765625,
      "unchanged_views_max_abs_delta": 0.0,
      "passed": true
    }
  },
  "4d": {
    "T=1": {
      "input": {
        "shape": [
          1,
          3,
          1,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "latents": {
        "shape": [
          1,
          16,
          1,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded": {
        "shape": [
          1,
          3,
          1,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "dit": {
        "shape": [
          1,
          16,
          1,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "measured_T_prime": 1,
      "decoded_range": [
        -1.0,
        1.0
      ],
      "decoded_std": 0.3308880627155304
    },
    "T=5": {
      "input": {
        "shape": [
          1,
          3,
          5,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "latents": {
        "shape": [
          1,
          16,
          2,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded": {
        "shape": [
          1,
          3,
          5,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "dit": {
        "shape": [
          1,
          16,
          2,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "measured_T_prime": 2,
      "decoded_range": [
        -1.0,
        1.0
      ],
      "decoded_std": 0.3311874568462372
    },
    "T=9": {
      "input": {
        "shape": [
          1,
          3,
          9,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "latents": {
        "shape": [
          1,
          16,
          3,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded": {
        "shape": [
          1,
          3,
          9,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "dit": {
        "shape": [
          1,
          16,
          3,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "measured_T_prime": 3,
      "decoded_range": [
        -1.0,
        1.0
      ],
      "decoded_std": 0.33024272322654724
    },
    "T=17": {
      "input": {
        "shape": [
          1,
          3,
          17,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "latents": {
        "shape": [
          1,
          16,
          5,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "decoded": {
        "shape": [
          1,
          3,
          17,
          64,
          64
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "dit": {
        "shape": [
          1,
          16,
          5,
          8,
          8
        ],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "finite": true
      },
      "measured_T_prime": 5,
      "decoded_range": [
        -1.0,
        1.0
      ],
      "decoded_std": 0.3277554214000702
    }
  },
  "temporal_impulse": {
    "seed": 0,
    "lengths": {
      "1": {
        "latent_length": 1,
        "positions": {
          "0": {
            "affected_latent_positions": [
              0
            ],
            "mean_abs_delta": [
              0.15417930483818054
            ],
            "threshold": 1.5417930483818056e-05
          }
        }
      },
      "5": {
        "latent_length": 2,
        "positions": {
          "0": {
            "affected_latent_positions": [
              0,
              1
            ],
            "mean_abs_delta": [
              0.15417930483818054,
              0.06144480034708977
            ],
            "threshold": 1.5417930483818056e-05
          },
          "1": {
            "affected_latent_positions": [
              1
            ],
            "mean_abs_delta": [
              0.0,
              0.13826265931129456
            ],
            "threshold": 1.3826265931129456e-05
          },
          "2": {
            "affected_latent_positions": [
              1
            ],
            "mean_abs_delta": [
              0.0,
              0.14541122317314148
            ],
            "threshold": 1.4541122317314148e-05
          },
          "3": {
            "affected_latent_positions": [
              1
            ],
            "mean_abs_delta": [
              0.0,
              0.15652808547019958
            ],
            "threshold": 1.5652808547019958e-05
          },
          "4": {
            "affected_latent_positions": [
              1
            ],
            "mean_abs_delta": [
              0.0,
              0.1519857943058014
            ],
            "threshold": 1.519857943058014e-05
          }
        }
      },
      "9": {
        "latent_length": 3,
        "positions": {
          "0": {
            "affected_latent_positions": [
              0,
              1,
              2
            ],
            "mean_abs_delta": [
              0.15417930483818054,
              0.06144480034708977,
              0.03837144747376442
            ],
            "threshold": 1.5417930483818056e-05
          },
          "1": {
            "affected_latent_positions": [
              1,
              2
            ],
            "mean_abs_delta": [
              0.0,
              0.13826265931129456,
              0.037634532898664474
            ],
            "threshold": 1.3826265931129456e-05
          },
          "2": {
            "affected_latent_positions": [
              1,
              2
            ],
            "mean_abs_delta": [
              0.0,
              0.14541122317314148,
              0.033445436507463455
            ],
            "threshold": 1.4541122317314148e-05
          },
          "3": {
            "affected_latent_positions": [
              1,
              2
            ],
            "mean_abs_delta": [
              0.0,
              0.15652808547019958,
              0.04669918492436409
            ],
            "threshold": 1.5652808547019958e-05
          },
          "4": {
            "affected_latent_positions": [
              1,
              2
            ],
            "mean_abs_delta": [
              0.0,
              0.1519857943058014,
              0.05043643340468407
            ],
            "threshold": 1.519857943058014e-05
          },
          "5": {
            "affected_latent_positions": [
              2
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.15397711098194122
            ],
            "threshold": 1.5397711098194124e-05
          },
          "6": {
            "affected_latent_positions": [
              2
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.15778282284736633
            ],
            "threshold": 1.5778282284736636e-05
          },
          "7": {
            "affected_latent_positions": [
              2
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.18237793445587158
            ],
            "threshold": 1.8237793445587158e-05
          },
          "8": {
            "affected_latent_positions": [
              2
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.17558425664901733
            ],
            "threshold": 1.7558425664901734e-05
          }
        }
      },
      "17": {
        "latent_length": 5,
        "positions": {
          "0": {
            "affected_latent_positions": [
              0,
              1,
              2,
              3,
              4
            ],
            "mean_abs_delta": [
              0.15417930483818054,
              0.06144480034708977,
              0.03837144747376442,
              0.02084704488515854,
              0.007985614240169525
            ],
            "threshold": 1.5417930483818056e-05
          },
          "1": {
            "affected_latent_positions": [
              1,
              2,
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.13826265931129456,
              0.037634532898664474,
              0.01945795863866806,
              0.0061772167682647705
            ],
            "threshold": 1.3826265931129456e-05
          },
          "2": {
            "affected_latent_positions": [
              1,
              2,
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.14541122317314148,
              0.033445436507463455,
              0.023752722889184952,
              0.0053608715534210205
            ],
            "threshold": 1.4541122317314148e-05
          },
          "3": {
            "affected_latent_positions": [
              1,
              2,
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.15652808547019958,
              0.04669918492436409,
              0.034453317523002625,
              0.00935211032629013
            ],
            "threshold": 1.5652808547019958e-05
          },
          "4": {
            "affected_latent_positions": [
              1,
              2,
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.1519857943058014,
              0.05043643340468407,
              0.026326604187488556,
              0.00838925689458847
            ],
            "threshold": 1.519857943058014e-05
          },
          "5": {
            "affected_latent_positions": [
              2,
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.15397711098194122,
              0.042052797973155975,
              0.02516641467809677
            ],
            "threshold": 1.5397711098194124e-05
          },
          "6": {
            "affected_latent_positions": [
              2,
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.15778282284736633,
              0.03767671436071396,
              0.027393724769353867
            ],
            "threshold": 1.5778282284736636e-05
          },
          "7": {
            "affected_latent_positions": [
              2,
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.18237793445587158,
              0.046829357743263245,
              0.0399128757417202
            ],
            "threshold": 1.8237793445587158e-05
          },
          "8": {
            "affected_latent_positions": [
              2,
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.17558425664901733,
              0.05338449031114578,
              0.02859395742416382
            ],
            "threshold": 1.7558425664901734e-05
          },
          "9": {
            "affected_latent_positions": [
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.0,
              0.15406271815299988,
              0.03922785073518753
            ],
            "threshold": 1.540627181529999e-05
          },
          "10": {
            "affected_latent_positions": [
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.0,
              0.1662488430738449,
              0.03539963811635971
            ],
            "threshold": 1.6624884307384493e-05
          },
          "11": {
            "affected_latent_positions": [
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.0,
              0.182113915681839,
              0.05307245999574661
            ],
            "threshold": 1.82113915681839e-05
          },
          "12": {
            "affected_latent_positions": [
              3,
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.0,
              0.17378990352153778,
              0.05522869527339935
            ],
            "threshold": 1.737899035215378e-05
          },
          "13": {
            "affected_latent_positions": [
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.15714150667190552
            ],
            "threshold": 1.5714150667190552e-05
          },
          "14": {
            "affected_latent_positions": [
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.1541149616241455
            ],
            "threshold": 1.541149616241455e-05
          },
          "15": {
            "affected_latent_positions": [
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.18344759941101074
            ],
            "threshold": 1.8344759941101074e-05
          },
          "16": {
            "affected_latent_positions": [
              4
            ],
            "mean_abs_delta": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.1737498939037323
            ],
            "threshold": 1.7374989390373232e-05
          }
        }
      }
    }
  }
}
```

## Reproduction

```bash
python scripts/stage0_validate.py --model-dir /data/linzizhuo/RL3DSR_WAN_REMO/models/Wan2.1-T2V-1.3B --nerf-scene /data/linzizhuo/RL3DSR_WAN_REMO/datasets/nerf_synthetic/chair --mip-scene /data/linzizhuo/RL3DSR_WAN_REMO/datasets/360_v2/counter --resolution 64 --report docs/stage0_validation.md
```

## Limitations

- The 4D input is a deterministic synthetic video fixture because the repository has no real pose-annotated 4D video dataset.
- The 3D VAE path isolates views; the joint DiT receives the regrouped view axis as its joint sequence axis.
