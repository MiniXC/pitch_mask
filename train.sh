accelerate launch train.py \
--wandb_mode="offline" \
--eval_every=100 \
--dropout=0.2 \
--wandb_run_name="256 buckets" \
--do_bucketize=True \
--use_energy=True \
--energy_buckets=256 \
--pitch_buckets=256

# --resume_from_checkpoint="checkpoints/model_20000.pt" \