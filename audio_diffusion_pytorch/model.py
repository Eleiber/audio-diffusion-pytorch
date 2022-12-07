from random import randint
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn
from tqdm import tqdm

from .diffusion import XDiffusion
from .modules import XUNet1d, rand_bool
from .utils import downsample, exists, groupby, upsample

"""
Diffusion Classes (generic for 1d data)
"""


class Model1d(nn.Module):
    def __init__(self, unet_type: str = "base", **kwargs):
        super().__init__()
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)
        self.unet = XUNet1d(type=unet_type, **kwargs)
        self.diffusion = XDiffusion(net=self.unet, **diffusion_kwargs)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.diffusion(x, **kwargs)

    def sample(self, *args, **kwargs) -> Tensor:
        return self.diffusion.sample(*args, **kwargs)


class DiffusionAR1d(Model1d):
    def __init__(
        self,
        in_channels: int,
        chunk_length: int,
        upsample: int = 0,
        dropout: float = 0.05,
        verbose: int = 0,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.chunk_length = chunk_length
        self.dropout = dropout
        self.upsample = upsample
        self.verbose = verbose
        super().__init__(
            in_channels=in_channels,
            context_channels=[in_channels * (2 if upsample > 0 else 1)],
            **kwargs,
        )

    def reupsample(self, x: Tensor) -> Tensor:
        x = x.clone()
        x = downsample(x, factor=self.upsample)
        x = upsample(x, factor=self.upsample)
        return x

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        b, _, t, device = *x.shape, x.device
        cl, num_chunks = self.chunk_length, t // self.chunk_length
        assert num_chunks >= 2, "Input tensor length must be >= chunk_length * 2"

        # Get prev and current target chunks
        chunk_index = randint(0, num_chunks - 2)
        chunk_pos = cl * (chunk_index + 1)
        chunk_prev = x[:, :, cl * chunk_index : chunk_pos]
        chunk_curr = x[:, :, chunk_pos : cl * (chunk_index + 2)]

        # Randomly dropout source chunks to allow for zero AR start
        if self.dropout > 0:
            batch_mask = rand_bool(shape=(b, 1, 1), proba=self.dropout, device=device)
            chunk_zeros = torch.zeros_like(chunk_prev)
            chunk_prev = torch.where(batch_mask, chunk_zeros, chunk_prev)

        # Condition on previous chunk and reupsampled current if required
        if self.upsample > 0:
            chunk_reupsampled = self.reupsample(chunk_curr)
            channels_list = [torch.cat([chunk_prev, chunk_reupsampled], dim=1)]
        else:
            channels_list = [chunk_prev]

        # Diffuse current current chunk
        return self.diffusion(chunk_curr, channels_list=channels_list, **kwargs)

    def sample(self, x: Tensor, start: Optional[Tensor] = None, **kwargs) -> Tensor:  # type: ignore # noqa
        noise = x

        if self.upsample > 0:
            # In this case we assume that x is the downsampled audio instead of noise
            upsampled = upsample(x, factor=self.upsample)
            noise = torch.randn_like(upsampled)

        b, c, t, device = *noise.shape, noise.device
        cl, num_chunks = self.chunk_length, t // self.chunk_length
        assert c == self.in_channels
        assert t % cl == 0, "noise must be divisible by chunk_length"

        # Initialize previous chunk
        if exists(start):
            chunk_prev = start[:, :, -cl:]
        else:
            chunk_prev = torch.zeros(b, c, cl).to(device)

        # Computed chunks
        chunks = []

        for i in tqdm(range(num_chunks), disable=(self.verbose == 0)):
            # Chunk noise
            chunk_start, chunk_end = cl * i, cl * (i + 1)
            noise_curr = noise[:, :, chunk_start:chunk_end]

            # Condition on previous chunk and artifically upsampled current if required
            if self.upsample > 0:
                chunk_upsampled = upsampled[:, :, chunk_start:chunk_end]
                channels_list = [torch.cat([chunk_prev, chunk_upsampled], dim=1)]
            else:
                channels_list = [chunk_prev]
            default_kwargs = dict(channels_list=channels_list)

            # Sample current chunk
            chunk_curr = super().sample(noise_curr, **{**default_kwargs, **kwargs})

            # Save chunk and use current as prev
            chunks += [chunk_curr]
            chunk_prev = chunk_curr

        return rearrange(chunks, "l b c t -> b c (l t)")
