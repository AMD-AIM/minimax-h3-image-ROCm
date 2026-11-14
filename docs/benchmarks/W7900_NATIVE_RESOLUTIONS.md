# W7900 two-GPU original-precision resolution benchmark

Measured on **November 14, 2026**. This report compares 1024 × 1024, 2048 × 2048, and 4096 × 4096 text-to-image requests on **two AMD Radeon PRO W7900D GPUs** (`gfx1100`, 48 GB each) using `--mode two-static` and the same implementation and workload as the [R9700 benchmark](R9700_NATIVE_RESOLUTIONS.md).

| Resolution | Full request (s) | Six-step denoise (s) | Peak VRAM / GPU (GiB) | Peak CPU RAM, RSS (GiB) | Host RAM growth (GiB) |
|---|---:|---:|---:|---:|---:|
| 1K · 1024 × 1024 | **72.39** | 5.03 | 32.05 | 6.37 | 10.54 |
| 2K · 2048 × 2048 | **92.48** | 18.15 | 32.43 | 6.56 | 9.56 |
| 4K · 4096 × 4096 | **314.26** | 229.68 | 34.00 | 6.49 | 40.44 |

1K has **three** completed fresh-process requests; 2K and 4K have **two** each. Timings are arithmetic means. Memory values are the maximum observed across the samples. The 4K host RAM growth is 40.44 GiB in one sample and 9.03 GiB in the other; this whole-host metric includes unrelated jobs running on the same machine and is not a requirement of this workload.

## Fixed workload and environment

- **Hardware:** two W7900D GPUs, RDNA3 / `gfx1100`, 51,522,830,336 bytes of VRAM each, ordered physical IDs `4,5` on the same NUMA node, PCIe without XGMI. The host reported 503.56 GiB of total RAM.
- **Runtime:** [`runtime-lock.w7900.json`](../../configs/runtime-lock.w7900.json); ROCm 10.1 foundation with Torch `2.14.0+rocm10.1.0`, Triton `3.8.0+git669b31ac.rocm10.1.0`, Python `3.13.13` and HIP `7.16.26385`, the same versions as the R9700 lock. comfy-kitchen and comfy-aimdo were rebuilt from the pinned sources for `gfx1100`. The foundation provides transformers `5.18.0` (R9700: `5.16.1`), which is used only for the tokenizer.
- **Models, precision and request:** identical to the R9700 benchmark: original BF16 encoder and DiT weights, released FP32 modules and PDD heads, FP32 Euler updates and FP32 VAE; product-photography prompt, seed 42, batch 1, six model evaluations, CFG=1. Weight quantization is disabled.
- **Placement:** `two-static`. The isolated text encoder streams CPU weights to the first GPU. DiT blocks 0–23 (30.41 GiB with the input modules) and 24–49 (31.33 GiB with the output heads) stay on the two GPUs during denoising. The first partition is released before tiled VAE decoding on the first GPU.
- **Attention policy:** unchanged. The 2K and 4K DiT requests each record 300 strict PyTorch Flash calls (sequence lengths 4,229 and 16,517); the 1K request uses ordinary SDPA dispatch.
- **Process and cache policy:** each sample starts a new container/process and recomputes conditioning. File and kernel caches may be warm.
- **Monitoring:** 0.5-second whole-card VRAM and process-tree RAM samples, with ownership checks throughout the request and after exit. No external process used the selected GPUs during the reported samples.

## Timing ranges

| Resolution | Request mean (range), s | Launcher mean (range), s | Denoise mean (range), s |
|---|---:|---:|---:|
| 1K · 1024 × 1024 | 72.39 (67.66–75.47) | 78.43 (73.57–82.47) | 5.03 (5.01–5.06) |
| 2K · 2048 × 2048 | 92.48 (92.24–92.71) | 97.77 (97.52–98.02) | 18.15 (18.13–18.17) |
| 4K · 4096 × 4096 | 314.26 (312.38–316.13) | 320.18 (318.33–322.03) | 229.68 (228.60–230.76) |

Request and launcher time follow the [R9700 definitions](R9700_NATIVE_RESOLUTIONS.md#timing-ranges). The 1K spread comes from text encoding and DiT loading, which depend on host CPU and storage load; denoising varied by less than 1%.

## Stage breakdown

Arithmetic mean seconds:

| Resolution | Text encoding | DiT loading | Six-step denoise | VAE loading | VAE decoding | PNG saving |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 33.22 | 30.62 | 5.03 | 1.50 | 0.96 | 0.38 |
| 2K | 33.88 | 32.09 | 18.15 | 1.51 | 4.31 | 1.86 |
| 4K | 31.77 | 28.23 | 229.68 | 1.47 | 16.06 | 6.33 |

Each 4K denoising step takes about 38.3 s. A one-call microbenchmark on the same GPU model measured the 4K-shaped Flash SDPA call (`[1, 56, 16517, 128]`, BF16) at 250 ms (about 31 TFLOPS) and the four block GEMMs at 135 ms (about 86 TFLOPS), so attention and GEMMs account for roughly half of the 4K step time. The rest grows faster than the token count and has not yet been profiled.

## Memory measurements

All values are **GiB (2³⁰ bytes)**, taking the maximum over the samples.

| Resolution | GPU4 VRAM | GPU5 VRAM | PyTorch allocated, L0 / L1 | Process-tree RSS | Sum of process VmHWM | Host RAM growth |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 31.14 | 32.05 | 30.66 / 31.58 | 6.37 | 6.55 | 10.54 |
| 2K | 31.61 | 32.43 | 31.00 / 31.92 | 6.56 | 6.56 | 9.56 |
| 4K | 33.42 | 34.00 | 32.36 / 33.29 | 6.49 | 6.55 | 40.44 |

Metric definitions match the [R9700 report](R9700_NATIVE_RESOLUTIONS.md#memory-measurements). At 4K the busiest card keeps about 14 GiB of its 48 GiB free.

## Reproducibility and qualification

- **Two-GPU placement versus single-GPU streaming:** a 2K `single-offload` request on one W7900D, which streams DiT blocks from the CPU, reproduced the `two-static` conditioning, initial noise, all six step latents, and the final PNG bit for bit. This exceeds the frozen gate (step relative L2 ≤ 1e-6, max absolute ≤ 1e-5).
- **A/A:** for each resolution, all samples produced byte-identical conditioning, initial noise, step latents, and PNGs.

| Resolution | PNG SHA256, identical for all samples |
|---|---|
| 1K | `edbdffe277df1ebc9980b4279b324c81d4708f6193168259f1f191b42a651bad` |
| 2K | `2630189b04cdf200302a01c18d0cc30262a637b2d5c41ed1c6633b2b8e532b8a` |
| 4K | `bcf82c831c349849b836e240bd5ed6b591003078d035ee53a36e237b374ebc35` |

- **4K functional matrix:** all three prompts × two seeds in [`o5_qualification_matrix.json`](../../configs/o5_qualification_matrix.json) completed with 300 strict Flash calls, non-constant output and peak VRAM of 33.42 / 34.00 GiB.
- **Visual check:** an automated Qwen3-VL-8B-Instruct review of the six matrix images and the three benchmark images found no generation defects and confirmed 42 of 45 expected elements. The three it did not confirm are the stone footbridge in both architecture seeds and the specimen jar in one illustration seed. These are prompt-adherence details, not numerical corruption.
- W7900 and R9700 outputs come from different GPU architectures and are not expected to be pixel-identical.

The supported claim is **single-image text-to-image generation at original precision on two W7900-class (`gfx1100`, 48 GB) GPUs with CPU offload for the text encoder**. Concurrent serving, throughput scaling, and an all-GPU-resident pipeline remain unqualified.

## Evidence and regeneration

The [machine-readable summary](w7900-native-resolutions.json) contains unrounded measurements, per-sample stage and memory data, output/tensor hashes, raw-evidence hashes, model revisions, environment identity, and hashes of the 13 runtime source files. All samples ran at commit `38e5322`.

With the local evidence present, regenerate it from the project root:

```bash
python3 tools/summarize_resolution_sweep.py \
  runs/w7900-bench-1024-seed42-r1 \
  runs/w7900-bench-1024-seed42-r2 \
  runs/w7900-bench-1024-seed42-r3 \
  runs/w7900-bench-2048-seed42-r1 \
  runs/w7900-bench-2048-seed42-r2 \
  runs/w7900-bench-4096-seed42-r1 \
  runs/w7900-bench-4096-seed42-r2
```

An earlier set of 1K/2K/4K samples was rejected before inference by the launcher preflight because another job had started using one of the selected GPUs; those attempts are excluded and retained locally.
