# Tool reference

Run commands from the project root. See the [setup and generation guide](../docs/QUICKSTART.md) for complete commands and the [configuration reference](../configs/README.md) for pinned inputs.

## Host tools

| Tool | Purpose | Output |
|---|---|---|
| [`run_rocm10.py`](run_rocm10.py) | Reserve ordered physical GPUs, invoke the shared ROCm 10.1 foundation, and monitor ownership/resources | A new local `runs/` directory with launch, hardware, monitoring, and exit records |
| [`download_native_models.py`](download_native_models.py) | Download and verify the selected original BF16/FP32 files at fixed revisions; requires Python `requests`; honors `HF_ENDPOINT` for a mirror | Weights under `models/native/`; local verification under `models/.download/native/` |
| [`summarize_native_run.py`](summarize_native_run.py) | Summarize an existing native request using the Python standard library | `summary.json` and `summary.md` in that run directory |
| [`summarize_resolution_sweep.py`](summarize_resolution_sweep.py) | Validate at least two completed requests per 1K/2K/4K resolution, matching identities and exact A/A artifacts | A prepared JSON summary on stdout; no inference |

The launcher takes `--gpus "$GPU_IDS"` for an ordered four-card (R9700) or two-card (W7900) group, or `--gpu "$SETUP_GPU_ID"` for one card. `--hardware r9700|w7900` (default `r9700`, or `H3I_HARDWARE`) selects the [hardware profile](../configs/hardware/) used for device checks and passed to every tool in the container. Set these variables to available physical GPU IDs on your machine and choose a new `--run-dir` each time. The [setup guide](../docs/QUICKSTART.md) documents launcher prerequisites and device policy.

## Tools inside the foundation

Invoke these through `run_rocm10.py`. Setup starts with the foundation's Python; subsequent commands use `.runtime/venv/bin/python`.

| Tool | Purpose |
|---|---|
| [`prepare_environment.py`](prepare_environment.py) | Create the project venv and review/install application dependencies while preserving protected foundation packages |
| [`build_native.py`](build_native.py) | Build pinned comfy-kitchen and comfy-aimdo HIP extensions for the profile's architecture (`gfx1201` or `gfx1100`) |
| [`probe_native_runtime.py`](probe_native_runtime.py) | Check imports, one-, two- or four-device mapping/copies, and optionally a forced Flash call at a 4K token shape |
| [`run_native_image.py`](run_native_image.py) | Run the original-precision image pipeline with stage/resource evidence; `four-static` is the published R9700 path and `two-static` the published W7900 path |

`run_native_image.py` also retains `single-offload`, the diagnostic baseline mode. The published 1K/2K/4K performance tables cover `four-static` (R9700) and `two-static` (W7900).

## Internal helper

[`encode_native_conditioning.py`](encode_native_conditioning.py) is spawned by the native pipeline. It consumes an owned stage request, streams the original text encoder on the first GPU, and saves the BF16 conditioning artifact before exiting. Use the parent image entry point to obtain a complete request.

## Historical INT8 workflow

| Tool | Purpose |
|---|---|
| [`run_image.py`](run_image.py) | Execute the frozen single-card INT8 ConvRot ComfyUI workflow |
| [`verify_environment.py`](verify_environment.py) | Check legacy ConvRot/DynamicVRAM support and the three historical model files |
| [`comfy_runtime.py`](comfy_runtime.py) | Internal ComfyUI bootstrap shared by those legacy entry points |

See the [frozen INT8 release](../docs/releases/v0.1.0-image-int8-first-image.md). These legacy files and entry-point paths are retained for reproducibility.

Exploratory diagnostic scripts are archived in the development workspace's ignored `research/` directory. Raw images, tensors, logs, and local reports belong in ignored run or research directories.
