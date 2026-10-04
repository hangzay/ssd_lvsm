# Baseline experiment settings

This document records the protocols stated in the manuscript, *Resolving Representation Ambiguity in Feedforward Novel View Synthesis Transformer via Semantic-Spatial Decoupling* (Implementation Details and Appendix: Implementation Details, iLRM, and Complementary Generative Comparisons). It provides experiment settings only; external model code, datasets, and pretrained weights are not included.

## Controlled feedforward reconstruction

The LVSM baseline and decoupled variants share the implementation, data splits, view sampling, image resolution, update budget, and fixed evaluation views. Their reported scores are controlled reimplementations, not the original models' published large-scale results.

| Setting | RealEstate10K | Objaverse |
| --- | --- | --- |
| Resolution | 256×256 | 256×256 |
| Input / target views | 2 / 6 | 4 / 8 |
| GPUs | 4 A100 80GB | 8 A100 80GB |
| Batch per GPU | 4 | 4 |
| Effective batch (accumulation 1) | 16 | 32 |
| Optimizer updates | 50,000 | 50,000 |
| LVSM learning rate | 0.0004 | 0.0004 |
| LVSM warmup updates | 3,000 | 2,500 |
| LVSM schedule | Constant after warmup | Constant after warmup |
| LVSM precision | BF16 | BF16 |
| LVSM RGB objective | L2 + 0.5 × VGG perceptual | L2 + 1.0 × VGG perceptual |

LVSM uses AdamW with betas (0.9, 0.95) and weight decay 0.05; LPIPS is disabled in its training objective. The decoder-only backbone has 12 layers (24 layers for the scaling experiment); the encoder-decoder has 12 encoder and 12 decoder layers. Both use width 768. Optional semantic and spatial losses each have weight 0.5. Semantic supervision uses frozen DINOv3 ViT-B/16 layer-8 features, spatial normalization gamma 0.60, and Smooth L1 matching through a 3×3 projector. Spatial supervision uses DA3-aligned cross-view cosine consistency; the minimum valid correspondence counts are 100 for RealEstate10K and 50 for Objaverse.

| Comparison method | Manuscript-specific setting |
| --- | --- |
| LVSM | Entangled baseline; disable FiLM, DINO, and spatial supervision using the switches in README. |
| Raw Raymap | Replace Plucker `[o × d, d]` with `[o, d]`; keep the architecture and protocol fixed for the baseline/decouple pair. |
| PRoPE | Controlled LVSM variant with projective camera conditioning in attention. |
| MVSplat | Camera intrinsics and poses; common RealEstate10K data, sampling, resolution, hardware, batch, 50K updates, and fixed evaluation views. |
| DepthSplat | Official architecture and pretrained encoder; common RealEstate10K comparison protocol. |
| pixelSplat | Official architecture and pretrained encoder; common RealEstate10K comparison protocol. |
| LagerNVS-DA3-S/4 | Replace VGGT-1B with the DA3-S DINOv2 backbone and reduce renderer depth from 12 to 4 layers to match parameter scale; common RealEstate10K comparison protocol. |
| LGM | Common Objaverse training and evaluation protocol. |

External reconstruction methods retain their native losses, optimizers, and pretrained components. Matching the data, hardware, update budget, and evaluation does not equate pretrained priors. Hyperparameters absent from the manuscript are not supplied as claimed paper settings. Use each external method's implementation for training; ssLVSM entrypoints run the LVSM variants only.

## iLRM transfer experiment

The baseline and decoupled model share 2 context views, 4 target views, 256×256 resolution, 4 A100 80GB GPUs, batch/GPU 32, and 5,000 updates. Both use AdamW, learning rate 0.0004, 2,500 warmup updates, gradient clipping 1.0, BF16, and the native RGB MSE objective. The output remains the original Gaussian head. The decoupled model uses branch-wise normalization, separate FFNs, and bidirectional modulation. iLRM implementation code is not released here.

## Complementary generative references

These experiments have different input protocols, model sizes, and pretrained priors, and are separate from the controlled deterministic reconstruction comparison.

| Adapted model | Settings specified in the manuscript |
| --- | --- |
| GEN3C-MobileI2V | Replace the large video-diffusion backbone with MobileI2V; retain independent 3D-cache construction and in-layer cache fusion; adapt latent and temporal/spatial interfaces; train on the same RealEstate10K split for 50K iterations. |
| Lyra | Static single-source-image reconstruction; use frozen GEN3C-MobileI2V to generate RealEstate10K trajectories; train the 3DGS decoder for 50K iterations, retaining Plucker conditioning, multi-trajectory fusion, and static reconstruction. |

Published full-scale GEN3C and Lyra scores are contextual references under their original protocols.
