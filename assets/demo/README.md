# Demo Assets

The project page uses one showcase manifest:

```text
assets/demo/showcase/manifest.json
```

Required for the full demo:

- `input_01.png`, `input_02.png`: the two posed RGB input views used by the demo.
- `output_01.png`, `output_02.png`, `output_03.png`: the three generated novel views, aligned with the three target Plucker-ray queries in `manifest.json`.
- `rendered_video.mp4`: the generated novel-view trajectory rendered by the model.

After rendering the trajectory video, set `video.src` in `manifest.json` to `assets/demo/showcase/rendered_video.mp4`.

Recommended:

- `poster.jpg`: still frame used before the video plays.

The `camera` and `trajectory` arrays in `manifest.json` are normalized display coordinates for the website's 3D spatial map, not a training-data format. If you have real camera poses, project them into a readable isometric layout and place the normalized coordinates there. If you only have exported images, manually arrange the coordinates so the two source cameras, three output cameras, and video path communicate the view synthesis process clearly.

Current inference utilities commonly export `input.png` and `rendered_video.mp4`. For this website demo, split `input.png` into the per-view images named in the manifest if needed.
