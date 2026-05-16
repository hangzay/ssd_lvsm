torchrun --nproc_per_node 8 --nnodes 1 \
  --rdzv_id 18638 --rdzv_backend c10d --rdzv_endpoint localhost:29505 \
  train.py --config configs/objaverse_decoder_only.yaml \
  training.wandb_exp_name=objaverse_decoder_only \
  training.checkpoint_dir=./experiments/checkpoints/objaverse_decoder_only \
  model.transformer.n_layer=12 \
  training.batch_size_per_gpu=4
