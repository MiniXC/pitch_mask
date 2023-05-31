import json
import sys
import os

from accelerate import Accelerator
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from speech_collator import SpeechCollator
from transformers import HfArgumentParser
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from transformers import get_linear_schedule_with_warmup
import requests
from nnAudio.features.mel import MelSpectrogram
import torchaudio
import torchaudio.transforms as AT
from librosa.filters import mel as librosa_mel
import librosa
from sklearn.decomposition import PCA

from vocex import VocexModel
from arguments import Args
from pitch_mask.model import PitchMask


def eval_loop(accelerator, model, eval_ds, step):
    loss = 0.0
    i = 0
    progress_bar = tqdm(range(len(eval_ds)), desc="eval")
    dvectors = []
    speakers = []
    embeddings = []
    for batch in eval_ds:
        with torch.no_grad():
            outputs = model(mel=batch["mel"], mask=batch["mask"], padding_mask=batch["padding_mask"], return_embedding=True)
            if i == 0:
                reconstructed_pitch = outputs["pitch"].cpu().numpy()
                pitch = outputs["real_pitch"].cpu().numpy()
                mask = outputs["mask"].cpu().numpy()
                padding_mask = outputs["padding_mask"].cpu().numpy()
                # plot the first batch
                fig, ax = plt.subplots(pitch.shape[0], 1, figsize=(10, 10))
                for j in range(pitch.shape[0]):
                    pitch[j][padding_mask[j] == 0] = np.nan
                    ax[j].plot(pitch[j])
                    reconstructed_pitch[j][mask[j] == 0] = np.nan
                    reconstructed_pitch[j][padding_mask[j] == 0] = np.nan
                    ax[j].plot(reconstructed_pitch[j])
                wandb.log({"pitch": wandb.Image(fig)})
            loss += outputs["loss"].item()
            i += 1
            dvector = model.vocex(batch["mel"], inference=True)["dvector"]
            dvectors += list(dvector.cpu().numpy())
            speakers += batch["speaker"]
            mean_embedding_per_sample = outputs["embedding"].mean(dim=1).cpu().numpy()
            embeddings += list(mean_embedding_per_sample)
            progress_bar.update(1)
    # visualize dvectors using PCA
    dvectors = np.array(dvectors)
    dvectors = (dvectors - dvectors.mean()) / dvectors.std()
    pca = PCA(n_components=2)
    dvectors_pca = pca.fit_transform(dvectors)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=dvectors_pca[:, 0], y=dvectors_pca[:, 1], hue=speakers, ax=ax)
    wandb.log({"dvectors": wandb.Image(fig)})
    # visualize embeddings using PCA
    embeddings = np.array(embeddings)
    embeddings = (embeddings - embeddings.mean()) / embeddings.std()
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], hue=speakers, ax=ax)
    wandb.log({"embeddings": wandb.Image(fig)})
    # get intra-speaker distance
    ## embeddings
    speaker_to_embedding = {}
    for speaker, embedding in zip(speakers, embeddings):
        if speaker not in speaker_to_embedding:
            speaker_to_embedding[speaker] = []
        speaker_to_embedding[speaker].append(embedding)
    intra_speaker_distances = []
    for speaker, embeddings in speaker_to_embedding.items():
        for i, embedding in enumerate(embeddings):
            for j in range(i+1, len(embeddings)):
                intra_speaker_distances.append(np.linalg.norm(embedding - embeddings[j]))
    intra_speaker_distances = np.array(intra_speaker_distances)
    intra_speaker_distance_mean = np.mean(intra_speaker_distances)
    ## dvectors
    speaker_to_dvector = {}
    for speaker, dvector in zip(speakers, dvectors):
        if speaker not in speaker_to_dvector:
            speaker_to_dvector[speaker] = []
        speaker_to_dvector[speaker].append(dvector)
    intra_speaker_distances_dvector = []
    for speaker, dvectors in speaker_to_dvector.items():
        for i, dvector in enumerate(dvectors):
            for j in range(i+1, len(dvectors)):
                intra_speaker_distances_dvector.append(np.linalg.norm(dvector - dvectors[j]))
    intra_speaker_distances_dvector = np.array(intra_speaker_distances_dvector)
    intra_speaker_distance_mean_dvector = np.mean(intra_speaker_distances_dvector)
    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(intra_speaker_distances, ax=ax, label="embedding")
    sns.histplot(intra_speaker_distances_dvector, ax=ax, label="dvector")
    ax.legend()
    wandb.log({"intra-speaker distance": wandb.Image(fig)})
    # log
    wandb.log({"intra-speaker distance embedding": intra_speaker_distance_mean, "step": step})
    wandb.log({"intra-speaker distance dvector": intra_speaker_distance_mean_dvector, "step": step})
    print(f"eval loss: {loss / i}")
    print(f"intra-speaker distance embedding: {intra_speaker_distance_mean}")
    print(f"intra-speaker distance dvector: {intra_speaker_distance_mean_dvector}")
    # inter-speaker distance
    inter_speaker_distances_dvec = []
    for speaker, dvectors in speaker_to_dvector.items():
        for speaker2, dvectors2 in speaker_to_dvector.items():
            if speaker == speaker2:
                continue
            dvector_mean = np.array(dvectors).mean(axis=0)
            dvector2_mean = np.array(dvectors2).mean(axis=0)
            inter_speaker_distances_dvec.append(np.linalg.norm(dvector_mean - dvector2_mean))
    inter_speaker_distances_dvec = np.array(inter_speaker_distances_dvec)
    inter_speaker_distance_mean_dvec = np.mean(inter_speaker_distances_dvec)
    ## embeddings   
    inter_speaker_distances_emb = []
    for speaker, embeddings in speaker_to_embedding.items():
        for speaker2, embeddings2 in speaker_to_embedding.items():
            if speaker == speaker2:
                continue
            # take mean of embedding and embedding2, then compute distance
            embedding_mean = np.array(embeddings).mean(axis=0)
            embedding2_mean = np.array(embeddings2).mean(axis=0)
            inter_speaker_distances_emb.append(np.linalg.norm(embedding_mean - embedding2_mean))
    inter_speaker_distances_emb = np.array(inter_speaker_distances_emb)
    inter_speaker_distance_mean_emb = np.mean(inter_speaker_distances_emb)
    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(inter_speaker_distances_emb, ax=ax, label="embedding")
    sns.histplot(inter_speaker_distances_dvec, ax=ax, label="dvector")
    ax.legend()
    wandb.log({"inter-speaker distance": wandb.Image(fig)})
    # log
    wandb.log({"inter-speaker distance embedding": inter_speaker_distance_mean_emb, "step": step})
    wandb.log({"inter-speaker distance dvector": inter_speaker_distance_mean_dvec, "step": step})
    print(f"inter-speaker distance embedding: {inter_speaker_distance_mean_emb}")
    print(f"inter-speaker distance dvector: {inter_speaker_distance_mean_dvec}")
    loss /= i
    wandb.log({"eval_loss": loss, "step": step})


class MelCollator():
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        f_min=0,
        f_max=8000,
        sample_rate=22050,
        max_frames=256,
    ):
        self.sampling_rate = sample_rate
        self.max_frames = max_frames
        self.mel_spectrogram = AT.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=0,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.mel_basis = librosa_mel(
            sr=self.sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()

    @staticmethod
    def drc(x, C=1, clip_val=1e-7):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def collate_fn(self, batch):
        for i, row in enumerate(batch):
            sr = self.sampling_rate
            audio_path = row["audio"]
            # load audio with torch audio and then resample
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sampling_rate:
                audio = torchaudio.transforms.Resample(sr, self.sampling_rate)(audio)
            audio = audio[0]
            audio = audio / torch.abs(audio).max()
            mel = self.mel_spectrogram(audio).unsqueeze(0)
            mel = torch.sqrt(mel[0])
            mel = torch.matmul(self.mel_basis, mel)
            mel = MelCollator.drc(mel)
            mel = mel.T
            if mel.shape[0] > self.max_frames:
                mel = mel[: self.max_frames]
            elif mel.shape[0] < self.max_frames:
                # pad dimension 0
                mel = torch.nn.functional.pad(
                    mel, (0, 0, 0, self.max_frames - mel.shape[0])
                )
            batch[i]["mel"] = mel
            # "For masking, we sample p = 0.065 of all time-steps to be
            # starting indices and mask the subsequent M = 10 time-steps. This results in approximately 49% of
            # all time steps to be masked with a mean span length of 14.7, or 299ms (see Appendix A for more
            # details on masking)." from wav2vec2
            # https://arxiv.org/pdf/2006.11477.pdf
            # we change this to 20 time steps as frames are ~10ms long
            mask = torch.zeros(mel.shape[0])
            p = 0.065 / 2
            M = 20
            for j in range(mel.shape[0]):
                if torch.rand(1) < p:
                    mask[j : j + M] = 1
            batch[i]["mask"] = mask
            batch[i]["padding_mask"] = (mel.sum(dim=-1) != 0).to(torch.float32)
        # stack
        batch = {
            "mel": torch.stack([row["mel"] for row in batch]),
            "mask": torch.stack([row["mask"] for row in batch]),
            "padding_mask": torch.stack([row["padding_mask"] for row in batch]),
            "speaker": [row["speaker"] for row in batch],
        }
        return batch

def main():
    parser = HfArgumentParser([Args])

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(sys.argv[1])[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    wandb.init(
        name=args.wandb_run_name,
        project=args.wandb_project,
        mode=args.wandb_mode,
    )
    wandb.config.update(args)

    if not args.bf16:
        accelerator = Accelerator()
    else:
        accelerator = Accelerator(mixed_precision="bf16")

    with accelerator.main_process_first():
        libritts = load_dataset(args.dataset)

    train_ds = libritts[args.train_split].shuffle(seed=42)
    eval_ds = libritts[args.eval_split]

    collator = MelCollator(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
    )

    vocex_model = VocexModel(measures="energy,pitch,srmr,snr,voice_activity_binary".split(","))
    vocex_url = args.vocex_url
    filename = vocex_url.split("/")[-1]
    # download model
    if not os.path.exists(args.vocex_path + "/" + filename):
        print("Downloading vocex model")
        if not os.path.exists(args.vocex_path):
            os.makedirs(args.vocex_path)
        r = requests.get(vocex_url, allow_redirects=True)
        open(os.path.join(args.vocex_path, filename), "wb").write(r.content)
    vocex_model.load_state_dict(torch.load(os.path.join(args.vocex_path, filename)))
    for scaler_key in vocex_model.scalers.keys():
        vocex_model.scalers[scaler_key].is_fit = True

    model = PitchMask(
        in_channels=1,
        out_channels=1,
        kernel_size=args.kernel_size,
        depthwise=args.depthwise,
        n_layers=args.n_layers,
        dropout=args.dropout,
        vocex=vocex_model,
        downsample_factor=args.downsample_factor,
        do_bucketize=args.do_bucketize,
        pitch_buckets=args.pitch_buckets,
        energy_buckets=args.energy_buckets,
        use_energy=args.use_energy,
    )


    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.resume_from_checkpoint:
        try:
            model.load_state_dict(torch.load(args.resume_from_checkpoint), strict=True)
        except RuntimeError as e:
            if args.strict_load:
                raise e
            else:
                print("Could not load model from checkpoint. Trying without strict loading, and removing mismatched keys.")
                current_model_dict = model.state_dict()
                loaded_state_dict = torch.load(args.resume_from_checkpoint)
                new_state_dict={
                    k:v if v.size()==current_model_dict[k].size() 
                    else current_model_dict[k] 
                    for k,v 
                    in zip(current_model_dict.keys(), loaded_state_dict.values())
                }
                model.load_state_dict(new_state_dict, strict=False)
        for scaler_key in model.vocex.scalers.keys():
            model.vocex.scalers[scaler_key].is_fit = True

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator.collate_fn,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator.collate_fn,
        prefetch_factor=args.prefetch_factor,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=[0.9, 0.999],
        eps=1e-8,
    )

    num_epochs = args.max_epochs
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps), desc="training", disable=not accelerator.is_local_main_process)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    model.train()

    step = 0

    losses = deque(maxlen=100)

    print(f"number of parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                step += 1
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), args.max_grad_norm)
                if step % args.gradient_sync_every == 0:
                    outputs = model(mel=batch["mel"], mask=batch["mask"], padding_mask=batch["padding_mask"])
                    loss = outputs["loss"] / args.gradient_accumulation_steps
                    accelerator.backward(loss)
                else:
                    with accelerator.no_sync(model):
                        outputs = model(mel=batch["mel"], mask=batch["mask"], padding_mask=batch["padding_mask"])
                        loss = outputs["loss"] / args.gradient_accumulation_steps
                        accelerator.backward(loss)
                ## add to queues
                losses.append(outputs["loss"])

                lr_scheduler.step()
                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                ## log losses
                if step % args.log_every == 0 and accelerator.is_local_main_process:
                    last_lr = lr_scheduler.get_last_lr()[0]
                    log_loss_dict = {}
                    log_loss_dict["train/loss"] = sum([l.item() for l in losses])/len(losses)
                    wandb.log(log_loss_dict, step=step)
                    wandb.log({"train/global_step": step}, step=step)
                    print(f"step={step}, lr={last_lr:.8f}:")
                    print({k.split('/')[1]: np.round(v, 4) for k, v in log_loss_dict.items()})
                ## evaluate
                if step % args.eval_every == 0 and accelerator.is_local_main_process:
                    model.eval()
                    with torch.no_grad():
                        eval_loop(accelerator, model, eval_dataloader, step)
                    model.train()
                ## save checkpoint
                if step % args.save_every == 0 and accelerator.is_local_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(unwrapped_model.state_dict(), f"{args.checkpoint_dir}/model_{step}.pt")
                progress_bar.update(1)
                # set description
                progress_bar.set_description(f"epoch {epoch+1}/{num_epochs}")

if __name__ == "__main__":
    main()