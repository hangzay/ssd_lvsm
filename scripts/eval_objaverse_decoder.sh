torchrun --nproc_per_node 1 --nnodes 1 \
  --rdzv_id 18639 --rdzv_backend c10d --rdzv_endpoint localhost:29507 \
  inference.py --config configs/objaverse_decoder_only.yaml \
  training.batch_size_per_gpu=4 \
  training.target_has_input=false \
  training.square_crop=true \
  training.num_input_views=4 \
  training.num_target_views=8 \
  training.num_views=12 \
  training.checkpoint_dir=./experiments/checkpoints/objaverse_decoder_only \
  inference.if_inference=true \
  inference.compute_metrics=true \
  inference.render_video=false \
  inference_out_dir=./experiments/eval/objaverse_decoder_only
