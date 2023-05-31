from dataclasses import dataclass

@dataclass
class Args:
    # vocex
    vocex_url: str = "https://github.com/MiniXC/vocex/raw/main/models/vocex_600k.pt"
    vocex_path: str = "vocex"
    # data loading
    dataset: str = "cdminix/libritts-aligned"
    train_split: str = "train"
    eval_split: str = "dev"
    num_workers: int = 96
    prefetch_factor: int = 2
    # model
    n_layers: int = 8
    depthwise: bool = False
    filter_size: int = 256
    kernel_size: int = 3
    dropout: float = 0.1
    downsample_factor: int = 4
    do_bucketize: bool = True
    pitch_buckets: int = 128
    energy_buckets: int = 128
    use_energy: bool = False
    # training
    max_epochs: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    log_every: int = 500
    eval_every: int = 5000
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    gradient_sync_every: int = 100
    bf16: bool = False
    resume_from_checkpoint: str = None
    strict_load: bool = False
    max_grad_norm: float = 2.0
    train_loss_logging_sum_steps: int = 100
    # audio
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    sample_rate: int = 22050
    max_frames: int = 512
    f_min: int = 0
    f_max: int = 8000
    # wandb
    wandb_project: str = "pitch_mask"
    wandb_run_name: str = None
    wandb_mode: str = "online"