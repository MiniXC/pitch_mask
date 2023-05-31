import torch
from torch import nn
from tqdm.auto import tqdm
from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer

class PitchMask(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        filter_size=256,
        kernel_size=3,
        dropout=0.1,
        depthwise=True,
        n_layers=8,
        vocex=None,
        downsample_factor=2,
        do_bucketize=True,
        pitch_buckets=1024,
        energy_buckets=1024,
        use_energy=True,
    ):
        super().__init__()
        in_channels = in_channels
        filter_size = filter_size
        kernel_size = kernel_size
        dropout = dropout
        depthwise = depthwise
        num_outputs = out_channels
        
        if not do_bucketize:
            if not use_energy:
                self.in_layer = nn.Linear(in_channels, filter_size)
            else:
                self.in_layer = nn.Linear(in_channels+1, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

        self.layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                2,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=depthwise,
            ),
            num_layers=n_layers,
        )

        if do_bucketize:
            num_outputs = pitch_buckets
            if use_energy:
                num_outputs = pitch_buckets + energy_buckets
        elif use_energy:
            num_outputs = 2

        self.linear = nn.Sequential(
            nn.Linear(filter_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs),
        )

        self.apply(self._init_weights)

        self.vocex = vocex

        self.downsample_factor = downsample_factor

        if do_bucketize:
            self.pitch_embedding = nn.Embedding(pitch_buckets, filter_size)
            self.pitch_bins = torch.linspace(-2.5, 2.5, pitch_buckets)
            if use_energy:
                self.energy_embedding = nn.Embedding(energy_buckets, filter_size)
                self.energy_bins = torch.linspace(-2.5, 2.5, energy_buckets)

        self.use_energy = use_energy
        self.do_bucketize = do_bucketize

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, mel, mask, padding_mask, return_embedding=False):
        with torch.no_grad():
            measures = self.vocex(mel, inference=True)["measures"]
            # normalize pitch
            pitch = measures["pitch"].unsqueeze(-1)
            pitch = (pitch - pitch.mean()) / pitch.std()
            mask = mask.unsqueeze(-1)
            masked_pitch = pitch * (1-mask)
            if self.downsample_factor > 1:
                masked_pitch = masked_pitch[:, ::self.downsample_factor]
                mask = mask[:, ::self.downsample_factor]
                padding_mask = padding_mask[:, ::self.downsample_factor]
                pitch = pitch[:, ::self.downsample_factor]
            if self.use_energy:
                energy = measures["energy"].unsqueeze(-1)
                energy = (energy - energy.mean()) / energy.std()
                if self.downsample_factor > 1:
                    energy = energy[:, ::self.downsample_factor]
                masked_energy = energy * (1-mask)
             
        if self.do_bucketize:
            masked_pitch = masked_pitch.squeeze(-1)
            pitch = pitch.squeeze(-1)
            masked_pitch = torch.bucketize(masked_pitch, self.pitch_bins)
            pitch = torch.bucketize(pitch, self.pitch_bins)
            masked_pitch = self.pitch_embedding(masked_pitch)
            mask = mask.squeeze(-1)
            padding_mask = padding_mask.squeeze(-1)
            if self.use_energy:
                masked_energy = masked_energy.squeeze(-1)
                energy = energy.squeeze(-1)
                energy = torch.bucketize(energy, self.energy_bins)
                masked_energy = torch.bucketize(masked_energy, self.energy_bins)
                masked_energy = self.energy_embedding(masked_energy)
                x = 0.5 * masked_pitch + 0.5 * masked_energy
            else:
                x = masked_pitch
        else:
            x = self.in_layer(masked_pitch)

        if return_embedding:
            result_dict = {
                "embedding": x,
            }
        else:
            result_dict = {}
        
        x = self.positional_encoding(x)
        x = self.layers(x, src_key_padding_mask=padding_mask)
        x = self.linear(x)

        if self.do_bucketize:
            num_pitch_bins = self.pitch_embedding.num_embeddings
            pitch_loss = nn.functional.cross_entropy(x.transpose(1, 2)[:, :num_pitch_bins], pitch, reduction="none")
            pitch_loss = (pitch_loss * mask * padding_mask)
            if self.use_energy:
                num_energy_bins = self.energy_embedding.num_embeddings
                energy_loss = nn.functional.cross_entropy(x.transpose(1, 2)[:, num_energy_bins:], energy, reduction="none")
                energy_loss = (energy_loss * mask * padding_mask)
                loss = 0.5 * pitch_loss + 0.5 * energy_loss
            else:
                loss = pitch_loss
            loss = loss.sum() / (mask * padding_mask).sum()
            reconstructed_pitch = self.pitch_bins.to(x.device)[x.argmax(-1)]
            real_pitch = self.pitch_bins.to(x.device)[pitch]
            result_dict.update({
                "loss": loss,
                "pitch": reconstructed_pitch,
                "real_pitch": real_pitch,
                "padding_mask": padding_mask,
                "mask": mask,
            })
            return result_dict
        else:
            padding_mask = padding_mask.unsqueeze(-1)
            loss = nn.functional.mse_loss(x*mask*padding_mask, pitch*mask*padding_mask)
            reconstructed_pitch = x * mask + masked_pitch
            result_dict.update({
                "loss": loss,
                "pitch": reconstructed_pitch,
                "real_pitch": pitch,
                "padding_mask": padding_mask,
                "mask": mask,
            })
            return result_dict