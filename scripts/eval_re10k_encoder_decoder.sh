torchrun --nproc_per_node 1 --nnodes 1 \
  --rdzv_id 18642 --rdzv_backend c10d --rdzv_endpoint localhost:29509 \
  inference.py --config configs/re10k_encoder_decoder.yaml \
  training.batch_size_per_gpu=4 \
  training.target_has_input=false \
  training.square_crop=true \
  training.num_input_views=2 \
  training.num_target_views=3 \
  training.num_views=5 \
  training.checkpoint_dir=./experiments/checkpoints/re10k_encoder_decoder \
  inference.if_inference=true \
  inference.compute_metrics=true \
  inference.render_video=false \
  inference_out_dir=./experiments/eval/re10k_encoder_decoder
