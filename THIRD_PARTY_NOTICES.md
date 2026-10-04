# Third-party notices

Project-authored code uses the repository's [MIT license](LICENSE.md). The components below retain their upstream terms and copyright notices.

| Component | Source | Corresponding files and use |
| --- | --- | --- |
| LVSM | [Haian-Jin/LVSM](https://github.com/Haian-Jin/LVSM) | Backbone and transformer foundation in `model/decoder_only.py`, `model/encoder_decoder.py`, and `model/transformer.py`; training/evaluation and data-processing foundation in `train.py`, `inference.py`, `process_data.py`, `data/dataset_scene.py`, `utils/data_utils.py`, `utils/training_utils.py`, and `model/loss.py`. This project extends them with decoupling, modulation, and training supervision. |
| PRoPE | [liruilong940607/prope](https://github.com/liruilong940607/prope), specifically [`prope/torch.py`](https://github.com/liruilong940607/prope/blob/main/prope/torch.py) | Projective attention transforms, 2D rotary coefficients, and the intrinsics-free GTA control in `model/camera_conditioning.py`; integrated with LVSM through `model/transformer.py` and `model/decoder_only.py`. |
| Nerfstudio / MultiNeRF | [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [google-research/multinerf](https://github.com/google-research/multinerf) | `utils/camera_utils.py`, inherited through LVSM. The original copyright header and function-specific MultiNeRF links are preserved. |
| DINOv3 | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | Frozen semantic teacher loaded through Torch Hub by `utils/dino_utils.py`; used by both model architectures during training. Source and pretrained weights are obtained separately and are not redistributed. |
| Depth Anything 3 (DA3) | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | External depth/camera cache generation. `utils/data_utils.py` reads the cache and aligns/back-projects geometry; `model/spatial_loss.py` uses it for correspondence supervision. DA3 source, weights, and cache-generation code are not included. |

LVSM's source-reference CC BY-NC-SA 4.0 license is retained in [LICENSES/LVSM-CC-BY-NC-SA-4.0.md](LICENSES/LVSM-CC-BY-NC-SA-4.0.md). PRoPE's MIT notice, including its upstream copyright attribution, is retained in [LICENSES/PRoPE-MIT.txt](LICENSES/PRoPE-MIT.txt). The Apache-2.0 license for the inherited camera utilities is retained in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

The raw-raymap switch (`[o, d]`) in `encode_rays` and the LVSM camera-context adapter are local implementations. The PRoPE source attribution applies to the attention transforms and their helpers.

External baseline training settings are documented in [BASELINES.md](BASELINES.md); their implementations are not bundled. RealEstate10K imagery, Objaverse assets, and separately obtained perceptual/teacher weights remain subject to their providers' terms.
