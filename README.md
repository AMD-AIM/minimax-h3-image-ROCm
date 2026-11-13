# MiniMax H3-Image INT8 on AMD ROCm

This `int8` branch preserves the **single-GPU INT8 ConvRot baseline** for MiniMax-H3-Image on AMD Radeon AI PRO R9700. It is based on the frozen `v0.1.0-image-int8-first-image` release; inference code, workflow, model revisions, and runtime locks are unchanged from that snapshot.

The [`main` branch](https://github.com/AMD-AIM/minimax-h3-image-ROCm/tree/main) contains the current original-precision BF16/FP32 four-GPU pipeline. The INT8 measurements below describe this branch's separate workload.

[Setup guide](docs/QUICKSTART.md) · [Model sources](models/README.md) · [First-image report](docs/FIRST_IMAGE.md) · [Frozen release](docs/releases/v0.1.0-image-int8-first-image.md)

## Validated baseline

| Item | Recorded result |
|---|---|
| Hardware | One AMD Radeon AI PRO R9700, RDNA4 / `gfx1201` |
| Image | 2048 × 2048, batch 1, seed 42 |
| Sampling | Six steps, CFG=1, Euler/simple, denoise=1 |
| Weights | INT8 ConvRot DiT and encoder; FP16 Image VAE |
| Engine | Pinned ComfyUI v0.45.0 `PromptExecutor` |
| Runtime | ROCm 10.1; Torch `2.14.0+rocm10.1.0`; Triton `3.8.0+git669b31ac.rocm10.1.0` |
| Full workflow request | **17.642 s** |
| Sampled whole-card VRAM peak | **31.618 GiB** |
| Peak process RSS | **2.857 GiB** |

This is one instrumented first-image observation, including loading, synchronization, tensor capture, and profiling overhead. DynamicVRAM and asynchronous weight offload are enabled. It does not establish warm serving throughput, lossless quantization, editing, 4K, or multi-GPU support for this INT8 path. See the [full measurement definitions](docs/FIRST_IMAGE.md).

## Reproduce the INT8 request

Follow the [setup guide](docs/QUICKSTART.md) to prepare the workspace launcher, pinned application environment, and three model files. Set `SETUP_GPU_ID` to an available physical GPU ID on your machine, then run from the project root:

```bash
python3 tools/run_rocm10.py \
  --gpu "${SETUP_GPU_ID:?Set SETUP_GPU_ID to an available physical GPU ID}" \
  --run-dir runs/image-int8-2048-seed42 \
  -- .runtime/venv/bin/python tools/run_image.py
```

Use a new run directory each time. The supplied launcher depends on the shared workspace infrastructure described in the setup guide; this branch does not provision that infrastructure.

## Source and repository boundaries

The original snapshot is preserved at [tag `v0.1.0-image-int8-first-image`](https://github.com/AMD-AIM/minimax-h3-image-ROCm/tree/v0.1.0-image-int8-first-image). The [freeze manifest](configs/releases/v0.1.0-image-int8-first-image.json) retains its original identities and report hashes. The reports on this branch are English translations; verify the historical report hashes against files at the frozen tag.

Only source, configuration, and prepared documentation are public. Model weights, environments, caches, raw images/tensors/logs, `runs/`, `vibe/`, and `research/` stay local. Under `models/`, only `README.md` is tracked. Models and third-party code retain their respective licenses.

The previous personal repository, [zihaomu/minimax-h3-image-ROCm](https://github.com/zihaomu/minimax-h3-image-ROCm), is retained as an archive. Active development is hosted by AMD-AIM.
