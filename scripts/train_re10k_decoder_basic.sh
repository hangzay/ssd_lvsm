torchrun --nproc_per_node 4 --nnodes 1 \
  --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
  train.py --config configs/re10k_decoder_only.yaml \
  training.variant=basic \
  training.wandb_exp_name=re10k_decoder_only_basic \
  training.checkpoint_dir=./experiments/checkpoints/re10k_decoder_only_basic \
  model.transformer.n_layer=12 \
  training.batch_size_per_gpu=4
