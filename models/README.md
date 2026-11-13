# INT8 model sources and pinned revision

This branch uses the ComfyUI INT8 ConvRot components for its single-GPU baseline. The original BF16/FP32 components used by the four-GPU pipeline are documented on [`main`](https://github.com/AMD-AIM/minimax-h3-image-ROCm/blob/main/models/README.md).

Download the files into the subdirectories below, relative to `models/`. Only this README is tracked by Git; weights, partial downloads, caches, and verification reports stay local.

- Original model: [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
- Repacked weights: [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)
- Pinned revision: `0bb149f51e3156ed4b90dd45d5620346bfe1df8f`

| Component | Local path | Size, GB / GiB | Pinned download |
|---|---|---:|---|
| DiT (INT8 ConvRot) | `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors` | 20.98 / 19.54 | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors) |
| Text/reference-image encoder (INT8 ConvRot) | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.14 / 25.28 | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) |
| Image VAE (FP16) | `vae/minimax_h3_image_vae_fp16.safetensors` | 1.17 / 1.09 | [Download](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/vae/minimax_h3_image_vae_fp16.safetensors) |

The three files total **49,289,162,266 bytes = 49.29 GB = 45.90 GiB**. Sizes come from file metadata at the pinned revision and describe disk storage, not runtime VRAM.

## Weight SHA256 checksums

```text
83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622  diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors
bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af  vae/minimax_h3_image_vae_fp16.safetensors
```

## Usage and validation scope

The workflow uses these three components with a fixed prompt. The optional prompt enhancer is outside the download. Select the `int8_convrot` encoder and `minimax_h3_image_vae_fp16` VAE variants.

File-size, SHA256, and safetensors checks establish download integrity. The [first-image report](../docs/FIRST_IMAGE.md) records inference validation and its limits. The three files total the disk sizes listed above; those sizes are not runtime VRAM requirements.

Follow the [setup guide](../docs/QUICKSTART.md) and read the licenses at the pinned upstream revisions before use.
