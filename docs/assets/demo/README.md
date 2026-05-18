# Demo Assets

The project page looks for one folder per demo scene:

```text
assets/demo/re10k/
assets/demo/objaverse/
```

Minimum files per scene:

- `input.png`: source-view strip exported by `utils.metric_utils.export_results`
- `rendered_video.mp4`: trajectory render exported when `inference.render_video=true`
- `gt_vs_pred.png`: target/prediction strip exported by `utils.metric_utils.export_results`

Use the same filenames for each scene folder. Update `docs/demo.js` only if a different scene name or path is needed.
After adding the files, set that scene's `available` field to `true` in `docs/demo.js`.
