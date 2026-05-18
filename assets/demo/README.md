# Demo Assets

The project page looks for one model-generation showcase folder:

```text
assets/demo/showcase/
```

Minimum files per scene:

- `input.png`: source-view strip exported by `utils.metric_utils.export_results`
- `rendered_video.mp4`: trajectory render exported when `inference.render_video=true`
- `gt_vs_pred.png`: target/prediction strip exported by `utils.metric_utils.export_results`

Use the same filenames in this folder. Update `docs/demo.js` only if a different path is needed.
After adding the files, set the `available` field to `true` in `docs/demo.js`.
