# Semantic-Spatial Decoupled NVS

Official code release for **Resolving Representation Ambiguity in Feedforward Novel View Synthesis Transformer via Semantic-Spatial Decoupling**.

Contact: [yh048172@gmail.com](mailto:yh048172@gmail.com)

![Model overview](assets/model.png)

## Highlights

This repository implements semantic-spatial decoupled feedforward novel view synthesis. RGB patch tokens form the semantic branch, Plucker-ray tokens form the spatial branch, and Independent-V attention shares Q/K routing while keeping branch-specific value updates. The public release keeps three experiment families:

- RealEstate10K decoder-only: `configs/re10k_decoder_only.yaml`
- RealEstate10K encoder-decoder: `configs/re10k_encoder_decoder.yaml`
- Objaverse decoder-only: `configs/objaverse_decoder_only.yaml`

The decoder-only RealEstate10K config exposes two variants: `training.variant=basic` for the base decoupled model and `training.variant=full` for decoupling with iREPA, spatial supervision, and bidirectional modulation. The other released configs use the full design.

## Preparation

Create the environment and install dependencies:

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

Download Objaverse from <https://objaverse.allenai.org/> and render/preprocess it into the layout referenced by `configs/objaverse_decoder_only.yaml`. Replace `path_to_objaverse_rendered_data/`, `path_to_objaverse_da3_cache/`, and `path_to_objaverse_scene_lists/` with your remote-machine paths.

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

Set `training.checkpoint_dir=path_to_checkpoint_or_dir/` and `inference_out_dir=path_to_eval_output/` as command-line overrides when needed. Evaluation writes images, metrics, and optional HTML pages under `inference_out_dir`.

## Paper Materials

The main architecture figure is included above. Additional qualitative inference material is available in `assets/infer-cropped.pdf`.

## Citation and Acknowledgements

If this code is useful for your work, please cite the Semantic-Spatial Decoupled NVS paper. The release uses DINOv3 ViT-B/16 as the frozen iREPA teacher and follows the DINOv3 license for the bundled helper code under `dinov3/`. The spatial correspondence objective is inspired by CAMEO:

```bibtex
@article{kwon2025cameo,
  title={CAMEO: Correspondence-Attention Alignment for Multi-View Diffusion Models},
  author={Kwon, Minkyung and Choi, Jinhyeok and Park, Jiho and Jeon, Seonghu and Jang, Jinhyuk and Seo, Junyoung and Kwak, Min-Seop and Kim, Jin-Hwa and Kim, Seungryong},
  journal={arXiv preprint arXiv:2512.03045},
  year={2025}
}
```

## License

See `LICENSE.md` for project, DINOv3, and third-party utility license notes.
