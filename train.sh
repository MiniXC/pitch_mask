accelerate launch train.py \
--wandb_mode="offline" \
--eval_every=1000 \
--resume_from_checkpoint="checkpoints/model_20000.pt" \
--dropout=0.2