# INT8 environment setup and image generation

[Branch overview](../README.md) · [Model sources](../models/README.md) · [First-image report](FIRST_IMAGE.md)

This guide covers the single-GPU INT8 ConvRot branch. The original-precision four-GPU pipeline is on [`main`](https://github.com/AMD-AIM/minimax-h3-image-ROCm/tree/main).

## Launcher prerequisites

These commands target the existing R9700 workspace foundation: Linux, Docker, `rocm-smi`, the parent workspace's `rocm10-launch.sh`, its locked image, and shared reservation and GPU-lock directories. The launcher uses `deepinfer_workspace/artifacts/amd/rocm10/` and `minimax_h3/.standard-opt-state/locks/` under that parent workspace. This repository does not provision those components on a new machine.

Run commands from the project root. Every `--run-dir` must be a new directory under `runs/`. Set `SETUP_GPU_ID` to one available physical GPU ID. The frozen launcher accepts IDs 0, 1, 2, 4, 5, 6, and 7; physical GPU3 is excluded by the development-host policy. Occupancy, shared reservations, and ownership must pass before execution.

## Prepare the pinned sources

The following checkouts stay under the ignored `.runtime/` directory. Setup tools verify their commits.

```bash
git clone --depth 1 --branch v0.45.0 https://github.com/Comfy-Org/ComfyUI.git .runtime/engines/ComfyUI-image
git clone --depth 1 --branch v0.2.41 https://github.com/Comfy-Org/comfy-kitchen.git .runtime/sources/comfy-kitchen
git clone --depth 1 --branch v0.6.0 https://github.com/Comfy-Org/comfy-aimdo.git .runtime/sources/comfy-aimdo
```

The ComfyUI commit is `989d93d0d968e704c589ce0146bb18c77f6b50cc`. The [runtime lock](../configs/runtime-lock.json) records the exact foundation/framework pairing, extension commits, and previously verified binary hashes.

## Prepare the models

Download the three pinned INT8/FP16 files using the links in [model sources](../models/README.md), place them in the documented `models/` subdirectories, and verify their full SHA256 values.

The environment check verifies loader paths and file sizes against the runtime lock. It does not reread and hash all 49.29 GB on every launch. Weights, partial files, download caches, and local verification records remain ignored.

## Install and check the environment

```bash
python3 tools/run_rocm10.py --gpu "${SETUP_GPU_ID:?Set SETUP_GPU_ID to an available physical GPU ID}" --run-dir runs/setup-image-env -- python tools/prepare_environment.py --install
python3 tools/run_rocm10.py --gpu "$SETUP_GPU_ID" --run-dir runs/build-image-native -- .runtime/venv/bin/python tools/build_native.py
python3 tools/run_rocm10.py --gpu "$SETUP_GPU_ID" --run-dir runs/check-int8-image-env -- .runtime/venv/bin/python tools/verify_environment.py
```

The project venv references foundation packages through a `.pth` file. Dependency preparation rejects a plan that would replace protected Torch, Triton, TorchVision, or ROCm packages. HIP extensions are rebuilt from pinned sources inside the same foundation for `gfx1201`. Application packages, build artifacts, and JIT caches remain project-local. Rebuild extensions if the ABI changes.

## Generate an image

```bash
python3 tools/run_rocm10.py \
  --gpu "${SETUP_GPU_ID:?Set SETUP_GPU_ID to an available physical GPU ID}" \
  --run-dir runs/image-int8-2048-seed42 \
  -- .runtime/venv/bin/python tools/run_image.py
```

The [API workflow](../configs/workflows/t2i_2048_api.json) fixes 2048 × 2048, batch 1, seed 42, six steps, CFG=1, Euler/simple, and denoise=1. It uses a detailed product-photography prompt and disables prompt expansion, the Comfy compiler, and node-result caching. Native DynamicVRAM is enabled.

The generated PNG is under the run's `output/` directory. The run also retains `result.json`, console logs, six-step latents, conditioning, noise, sigmas, sampled VRAM, and resource-release records. The entry point synchronizes stages and captures diagnostic tensors and selected CPU profiler records. It is not an uninstrumented warm-serving benchmark or a persistent Web UI.

## Validation scope

The retained baseline is one complete single-GPU request. A/A, a multi-prompt/multi-seed matrix, unprofiled warm-request statistics, editing, and multi-GPU execution are unqualified for this INT8 branch. Refer to the [first-image report](FIRST_IMAGE.md) and [frozen release report](releases/v0.1.0-image-int8-first-image.md).

Source, locks, and workflows are public. Images, weights, raw records, local plans, and research remain outside Git.
