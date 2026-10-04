# Resolving Representation Ambiguity in Feedforward Novel View Synthesis Transformer via Semantic-Spatial Decoupling

[Paper](https://arxiv.org/abs/2605.18599) · [Project page](https://hangzay.github.io/ssd_lvsm/) · [Experiment settings](BASELINES.md)

Semantic-Spatial Decoupling separates RGB and camera-ray features into two branches, with shared full-token Q/K attention and independent values, normalization, and FFNs. Optional bidirectional FiLM, DINOv3 semantic supervision, and geometric correspondence supervision complete the full model.

## Installation

```bash
conda create -n sslvsm python=3.11
conda activate sslvsm
pip install -r requirements.txt
```

Training and inference require CUDA and xFormers. W&B runs offline by default; set `WANDB_MODE=online` and `WANDB_API_KEY` for online logging. An optional local key file can be copied from `configs/api_keys_example.yaml`; it is ignored by Git.

## Data and training assets

Replace all `path_to_*` placeholders in the selected config before running:

| Asset | Config field |
| --- | --- |
| [RealEstate10K](https://google.github.io/realestate10k/) scene JSON list | `training.dataset_path` |
| [Objaverse](https://objaverse.allenai.org/) object manifest or scene list | `training.dataset_path` |
| Objaverse black-background renders and poses | `training.objaverse_root`, `inference.test_objaverse_root` |
| Depth/camera cache, needed only for geometric supervision | `training.depth_camera_cache_root` |
| DINOv3 ViT-B/16 weights, needed only for semantic supervision | `model.irepa.weight_path` |

`process_data.py` converts RealEstate10K `.torch` chunks into scene JSON files, RGB images, and train/test lists:

```bash
python process_data.py --base_path path_to_raw_re10k --output_dir path_to_re10k --mode train
python process_data.py --base_path path_to_raw_re10k --output_dir path_to_re10k --mode test
```

Objaverse expects `rendered/<shard>/<object_id>/*.png` with matching world-to-camera `.npy` poses, or the same scene directories directly under the render root. The committed 100-object evaluation list is excluded from training. Rendering and depth/camera cache generation are not included.

The loaders read DA3 depth and camera arrays from the following cache layout, then align cameras and back-project points during training:

```text
<cache_root>/<scene_id>/depth.zarr
<cache_root>/<scene_id>/camera_pose.zarr
<cache_root>/<scene_id>/camera_intrinsics.zarr
```

Objaverse also supports a shard directory before `<scene_id>`. The DINOv3 architecture is loaded through Torch Hub with local weights. For offline use, set `model.irepa.repo=path_to_dinov3_source`. VGG perceptual training uses `metric_checkpoint/imagenet-vgg-verydeep-19.mat`, downloaded on first use if absent. Datasets, teacher assets, keys, and checkpoints are not distributed here.

## Training

All four configs default to the full model at 256×256, patch size 8, width 768, and 50K optimizer updates. The decoder-only has 12 layers; the encoder-decoder has 12 encoder and 12 decoder layers.

| Dataset | Decoder-only | Encoder-decoder | Paper setting |
| --- | --- | --- | --- |
| RealEstate10K | `configs/re10k_decoder_only.yaml` | `configs/re10k_encoder_decoder.yaml` | 4 A100 80GB, batch/GPU 4, 2 input + 6 target views |
| Objaverse | `configs/objaverse_decoder_only.yaml` | `configs/objaverse_encoder_decoder.yaml` | 8 A100 80GB, batch/GPU 4, 4 input + 8 target views |

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/re10k_decoder_only.yaml
torchrun --standalone --nproc_per_node=8 train.py --config configs/objaverse_encoder_decoder.yaml
```

Choose any config from the table. The four independent switches select the method:

| Variant | `model.transformer.decouple` | `model.transformer.film` | `training.dino` | `training.spatial_supervision` |
| --- | --- | --- | --- | --- |
| Entangled LVSM baseline | false | false | false | false |
| Decouple only | true | false | false | false |
| Full model (default) | true | true | true | true |

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/re10k_decoder_only.yaml \
  model.transformer.film=false training.dino=false training.spatial_supervision=false \
  training.checkpoint_dir=./experiments/checkpoints/re10k_decouple_only \
  training.wandb_exp_name=re10k_decouple_only
```

Use a separate checkpoint directory for every variant: training automatically resumes from that directory. FiLM and both auxiliary losses require decoupling. Effective batch is `GPU count × batch_size_per_gpu × grad_accum_steps`. `checkpoint_every: 0` saves only the final checkpoint; set it to a positive forward-step interval for intermediate saves. The old `training.variant` / `model.transformer.variant` interfaces are replaced by the switches above; checkpoints must match the current architecture.

## Camera encodings

Both architectures support `model.camera_encoding.ray_encoding=plucker` (default, `[o × d, d]`) or `raymap` (`[o, d]`). Decoder-only additionally supports `model.camera_encoding.attention_encoding=prope`, applying projective camera transforms around attention; `gta` is its pose-only control. Encoder-decoder scene latents have no per-view camera assignment, so attention-level camera encodings are rejected.

For the paper's Raymap or PRoPE controlled comparisons, disable FiLM and both auxiliary losses, and compare `decouple=false` with `decouple=true` under identical settings:

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/re10k_decoder_only.yaml \
  model.camera_encoding.ray_encoding=raymap model.transformer.decouple=true \
  model.transformer.film=false training.dino=false training.spatial_supervision=false \
  training.checkpoint_dir=./experiments/checkpoints/re10k_raymap_decouple
```

For PRoPE, use `ray_encoding=plucker` and `attention_encoding=prope`. [BASELINES.md](BASELINES.md) records the manuscript's comparison protocols; external baseline implementations are not included.

## Inference and evaluation

Use the config and switches that match the checkpoint. No teacher or depth cache is needed at inference. RealEstate10K uses the committed fixed-view index (2 input, 3 target views):

```bash
torchrun --standalone --nproc_per_node=1 inference.py --config configs/re10k_decoder_only.yaml \
  inference.if_inference=true inference.compute_metrics=true \
  training.dataset_path=path_to_re10k/test/eval_list.txt \
  training.target_has_input=false training.num_input_views=2 \
  training.num_target_views=3 training.num_views=5 \
  training.checkpoint_dir=path_to_checkpoint_or_directory \
  inference_out_dir=./outputs/re10k
```

For Objaverse, select its config, retain 4 input / 8 target views, set `training.target_has_input=false`, and configure `inference.test_objaverse_root`; the committed test list and view index select the held-out objects. Evaluation exports RGB images, PSNR/SSIM/LPIPS, and an HTML browser. Set `inference.render_video=true` for camera-path videos.

## Validation and attribution

```bash
python tools/smoke_models.py --forward
python tools/smoke_camera_encodings.py
```

These use small synthetic inputs to check model construction, CUDA forwards, and camera-encoding gradients; they do not reproduce the paper's training runs. The training-only projector and teacher are omitted from inference.

Project-authored code retains the repository's [MIT license](LICENSE.md). See [third-party notices](THIRD_PARTY_NOTICES.md) for LVSM-derived components, PRoPE, and separately obtained models and datasets.

## Citation

```bibtex
@misc{wu2026resolvingrepresentationambiguityfeedforward,
  title={Resolving Representation Ambiguity in Feedforward Novel View Synthesis Transformer via Semantic-Spatial Decoupling},
  author={Yihang Wu and Yihang Sun and Shaofeng Zhang and Zuxuan Wu and Junchi Yan and Xiaosong Jia and Yu-gang Jiang},
  year={2026},
  eprint={2605.18599},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.18599}
}
```
