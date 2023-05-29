accelerate launch train.py \
--wandb_mode="online" \
--eval_every=1000 \
--resume_from_checkpoint="checkpoints/model_5000.pt" \
--dropout=0.25