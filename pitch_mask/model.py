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
    ):
        super().__init__()
        in_channels = in_channels
        filter_size = filter_size
        kernel_size = kernel_size
        dropout = dropout
        depthwise = depthwise
        num_outputs = out_channels
        
        self.in_layer = nn.Linear(in_channels, filter_size)

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

        self.linear = nn.Sequential(
            nn.Linear(filter_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs),
        )

        self.apply(self._init_weights)

        self.vocex = vocex

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, mel, mask, padding_mask):
        with torch.no_grad():
            pitch = self.vocex(mel, inference=True)["measures"]["pitch"].unsqueeze(-1)
            # normalize pitch
            pitch = (pitch - pitch.mean()) / pitch.std()
            mask = mask.unsqueeze(-1)
            masked_pitch = pitch * (1-mask)
             
        x = self.in_layer(masked_pitch)
        x = self.positional_encoding(x)
        x = self.layers(x, src_key_padding_mask=padding_mask)
        x = self.linear(x)

        padding_mask = padding_mask.unsqueeze(-1)

        loss = nn.functional.mse_loss(x*mask*padding_mask, pitch*mask*padding_mask)

        reconstructed_pitch = x * mask + masked_pitch

        return {
            "loss": loss,
            "pitch": reconstructed_pitch,
            "real_pitch": pitch,
        }