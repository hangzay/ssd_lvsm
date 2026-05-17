# SSD-LVSM: Semantic-Spatial Decoupled LVSM

Official code release for **Resolving Representation Ambiguity in Feedforward Novel View Synthesis Transformer via Semantic-Spatial Decoupling**.

SSD-LVSM separates RGB semantic tokens from Plucker-ray spatial tokens while preserving shared attention routing, reducing representation ambiguity in feedforward novel view synthesis transformers.

**Author:** Yihang Wu<br>
**Contact:** [yh048172@gmail.com](mailto:yh048172@gmail.com)<br>
**Project page:** <https://hangzay.github.io/ssd_lvsm/><br>
**Code:** <https://github.com/hangzay/ssd_lvsm>

![SSD-LVSM model overview](assets/model.png)

## Highlights

- **Semantic-spatial token decoupling:** RGB patch tokens form a semantic branch and Plucker-ray patch tokens form a spatial branch.
- **Independent-V attention:** semantic and spatial branches share Q/K attention routing while keeping branch-specific value updates.
- **Branch-specific supervision:** DINOv3/iREPA supervision targets the semantic branch, while DA3-derived geometric consistency supervises the spatial branch.
- **Bidirectional modulation:** lightweight semantic-to-spatial and spatial-to-semantic modulation improves controlled cross-branch conditioning.
- **Training-only helpers:** the frozen DINOv3 teacher, iREPA projector, and DA3 supervision caches are used only during training and are removed at inference.

## Model Variants

This release exposes three experiment families.

| Family | Config | Public variants |
| --- | --- | --- |
| RealEstate10K decoder-only | `configs/re10k_decoder_only.yaml` | `training.variant=basic` and `training.variant=full` |
| RealEstate10K encoder-decoder | `configs/re10k_encoder_decoder.yaml` | full design |
| Objaverse decoder-only | `configs/objaverse_decoder_only.yaml` | full design |

For `re10k_decoder_only`, `basic` enables the base semantic-spatial decoupled model. `full` adds branch-specific iREPA supervision, spatial supervision, and bidirectional modulation. The encoder-decoder and Objaverse released configs use the full design.

## Controlled Reimplementation Results

The following numbers are **controlled reimplementation results**, not direct comparisons against official large-scale LVSM checkpoints. Within each benchmark, the baseline and SSD-LVSM variants use the same codebase, data split, view sampling protocol, 256x256 resolution, 50K-step schedule, and training budget. These tables are intended to isolate the effect of semantic-spatial decoupling under a fixed budget.

iLRM experiments are not included in the README main results because the iLRM code path is not open-sourced in this release.

**Training configuration for the main table**

| Benchmark | Backbone | Views | Resolution | GPUs | Steps | Time |
| --- | --- | --- | --- | --- | --- | --- |
| RE10K decoder-only | 12-layer decoder-only, hidden dim 768 | 2 input / 6 target | 256x256 | 4x A100 80G | 50K | about 8 h / 32 GPU-hours |
| Objaverse decoder-only | 12-layer decoder-only, hidden dim 768 | 4 input / 8 target | 256x256 | 8x A100 80G | 50K | about 15 h / 120 GPU-hours |
| RE10K encoder-decoder | 12-layer encoder + 12-layer decoder, hidden dim 768 | 2 input / 6 target | 256x256 | 4x A100 80G | 50K | about 12 h / 48 GPU-hours |

| Architecture | Dataset | Model | PSNR up | SSIM up | LPIPS down |
| --- | --- | --- | ---: | ---: | ---: |
| Decoder-only | RE10K | Baseline | 26.10 | 0.839 | 0.144 |
| Decoder-only | RE10K | SSD-LVSM full | **27.21** | **0.869** | **0.125** |
| Decoder-only | Objaverse | Baseline | 23.75 | 0.864 | 0.150 |
| Decoder-only | Objaverse | SSD-LVSM full | **26.46** | **0.899** | **0.101** |
| Encoder-decoder | RE10K | Baseline | 24.06 | 0.775 | 0.206 |
| Encoder-decoder | RE10K | SSD-LVSM full | **25.31** | **0.806** | **0.154** |

**Fixed-budget RE10K decoder-only component control**

This table uses the RE10K decoder-only training configuration above: 12 decoder layers, 2 input / 6 target views, 256x256 resolution, 4x A100 80G, 50K training steps, about 8 wall-clock hours.

| Configuration | PSNR up | SSIM up | LPIPS down |
| --- | ---: | ---: | ---: |
| Decouple | 26.70 | 0.851 | 0.138 |
| Decouple + supervision | 26.91 | 0.857 | 0.134 |
| Decouple + modulation | 27.06 | 0.860 | 0.131 |
| Decouple + supervision + modulation | **27.21** | **0.869** | **0.125** |

## Why Low Overhead

The base decoupled design preserves the original token dimension and keeps shared Q/K attention routing, so it adds negligible inference overhead relative to the entangled decoder-only backbone. The full model adds lightweight bidirectional modulation. The DINOv3 teacher, iREPA projection head, DA3 point-map cache, and branch-specific supervision losses are training-only modules and are not required at inference.

## Preparation

Create the environment and install dependencies from the project root:

```bash
conda create -n decoupled-nvs python=3.11
conda activate decoupled-nvs
pip install -r requirements.txt
```

The training code uses CUDA, distributed `torchrun`, and `xformers` memory-efficient attention. Prepare `configs/api_keys.yaml` from `configs/api_keys_example.yaml` before WandB logging. The DINOv3 ViT-B/16 teacher is loaded from the local Hugging Face cache first and downloaded automatically when missing; gated model access must be approved before the first run.

## Data

Download RealEstate10K from the pixelSplat mirror: <http://schadenfreude.csail.mit.edu:8000/>. Preprocess raw `.torch` files with:

```bash
python process_data.py --base_path path_to_raw_re10k/ --output_dir path_to_re10k_processed/ --mode train
```

Download Objaverse from <https://objaverse.allenai.org/> and render/preprocess it into the layout referenced by `configs/objaverse_decoder_only.yaml`. Replace `path_to_objaverse_rendered_data/`, `path_to_objaverse_da3_cache/`, and `path_to_objaverse_scene_lists/` with your machine-specific paths.

Spatial supervision expects a DA3 cache per scene or object under `training.da3_cache_root`:

```text
<da3_cache_root>/<scene_id>/depth.zarr
<da3_cache_root>/<scene_id>/camera_pose.zarr
<da3_cache_root>/<scene_id>/camera_intrinsics.zarr
```

For RealEstate10K, replace `path_to_re10k_processed/` and `path_to_re10k_da3_cache/` in the configs before launching training or evaluation.

## Training

Training is intended for remote GPU machines. RealEstate10K scripts use 4 GPUs; Objaverse uses 8 GPUs.

Run the full RealEstate10K decoder-only model with the script defaults:

```bash
bash scripts/train_re10k_decoder_full.sh
```

Override layer count and per-GPU batch size directly when launching:

```bash
torchrun --nproc_per_node 4 --nnodes 1 \
  --rdzv_id 18636 --rdzv_backend c10d --rdzv_endpoint localhost:29503 \
  train.py --config configs/re10k_decoder_only.yaml \
  training.variant=full \
  model.transformer.n_layer=12 \
  training.batch_size_per_gpu=4
```

Other released training entry points:

```bash
bash scripts/train_re10k_decoder_basic.sh
bash scripts/train_re10k_encoder_decoder.sh
bash scripts/train_objaverse_decoder.sh
```

## Evaluation

Evaluate a RealEstate10K decoder-only checkpoint with:

```bash
bash scripts/eval_re10k_decoder.sh
```

Other released evaluation entry points:

```bash
bash scripts/eval_re10k_encoder_decoder.sh
bash scripts/eval_objaverse_decoder.sh
```

Set `training.checkpoint_dir=path_to_checkpoint_or_dir/` and `inference_out_dir=path_to_eval_output/` as command-line overrides when needed. Evaluation writes images, metrics, and optional HTML pages under `inference_out_dir`.

## Project Page

The GitHub Pages site lives in `docs/` and is designed to be served from:

```text
https://hangzay.github.io/ssd_lvsm/
```

After pushing to GitHub, enable Pages from the repository settings with branch `main` and directory `/docs`.

## Third-Party Components and Licenses

- DINOv3 helper code under `dinov3/` follows the DINOv3 license in `dinov3/LICENSE.md`.
- DA3 outputs are expected as external training-time supervision caches; this repository does not redistribute DA3 checkpoints or generated caches.
- RealEstate10K and Objaverse data are not redistributed here. Follow the original dataset licenses and terms.
- Camera utilities in `utils/camera_utils.py` include Nerfstudio-derived Apache License 2.0 code.

## Citation

If this code is useful for your work, please cite the Semantic-Spatial Decoupled NVS paper. Citation metadata will be added when a public record is available.

## License

See `LICENSE.md` for project, DINOv3, dataset, and third-party utility license notes.
