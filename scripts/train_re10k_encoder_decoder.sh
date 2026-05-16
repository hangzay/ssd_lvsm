torchrun --nproc_per_node 4 --nnodes 1 \
  --rdzv_id 18637 --rdzv_backend c10d --rdzv_endpoint localhost:29504 \
  train.py --config configs/re10k_encoder_decoder.yaml \
  training.wandb_exp_name=re10k_encoder_decoder \
  training.checkpoint_dir=./experiments/checkpoints/re10k_encoder_decoder \
  model.transformer.encoder_n_layer=12 \
  model.transformer.decoder_n_layer=12 \
  training.batch_size_per_gpu=4
