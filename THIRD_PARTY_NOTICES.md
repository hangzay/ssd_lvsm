# Third-party notices

- This project builds on [LVSM](https://github.com/haian-jin/LVSM). Preserve the LVSM copyright notices in derived code. The source reference's CC BY-NC-SA 4.0 license is retained in [LICENSES/LVSM-CC-BY-NC-SA-4.0.md](LICENSES/LVSM-CC-BY-NC-SA-4.0.md); the project MIT license does not replace these upstream terms.
- The projective camera transforms in `model/camera_conditioning.py` are adapted from [PRoPE](https://github.com/liruilong940607/prope), which declares an MIT license. Its notice is retained in [LICENSES/PRoPE-MIT.txt](LICENSES/PRoPE-MIT.txt).
- [DINOv3](https://github.com/facebookresearch/dinov3) is loaded separately through Torch Hub. DINOv3 source and pretrained weights retain their own terms; this repository does not redistribute them. DA3 cache generation is external to this release.
- RealEstate10K source imagery, Objaverse assets, and separately obtained perceptual/teacher weights remain subject to their respective providers' terms. They are not covered by the project license.
