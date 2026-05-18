# Demo Assets

The project page uses one showcase manifest:

```text
assets/demo/showcase/manifest.json
```

Required for the full demo:

- `input_01.png`, `input_02.png`, ...: individual posed input views. Add or remove entries in `manifest.json` to match the number of source views.
- `target.png`: the held-out target view or target query reference image.
- `rendered_video.mp4`: the generated novel-view trajectory rendered by the model.

After placing the files, fill the empty `src` fields in `manifest.json`, for example `assets/demo/showcase/input_01.png` and `assets/demo/showcase/rendered_video.mp4`.

Recommended:

- `poster.jpg`: still frame used before the video plays.
- `gt_vs_pred.png`: optional target/prediction strip exported by `utils.metric_utils.export_results`.

The `camera` and `trajectory` arrays in `manifest.json` are normalized display coordinates for the website's spatial map, not a training-data format. If you have real camera poses, project them into a readable 2D layout and place the normalized coordinates there. If you only have exported images, manually arrange the coordinates so the source cameras, target camera, and video path communicate the view synthesis process clearly.

Current inference utilities commonly export `input.png`, `gt_vs_pred.png`, and `rendered_video.mp4`. For this website demo, either split `input.png` into per-view images named in the manifest or point multiple manifest entries to the same input strip as a temporary placeholder.
