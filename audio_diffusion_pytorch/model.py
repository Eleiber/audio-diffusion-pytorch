from math import pi
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm

from .modules import UNet1d
from .utils import default


class DiffusionAR1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        length: int,
        num_splits: int,
        unet_type: Type[UNet1d] = UNet1d,
        **kwargs,
    ):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.split_length = length // num_splits

        self.net = unet_type(
            in_channels=in_channels,
            context_channels=[1],
            length=length,
            **kwargs,
        )

    @property
    def device(self):
        return next(self.net.parameters()).device

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Returns diffusion loss of v-objective with different noises per split"""
        b, _, t, device, dtype = *x.shape, x.device, x.dtype
        assert t == self.length, "input length must match length"
        # Sample amount of noise to add for each split
        sigmas = torch.rand((b, 1, self.num_splits), device=device, dtype=dtype)
        sigmas = repeat(sigmas, "b 1 n -> b 1 (n l)", l=self.split_length)
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Denoise and return loss
        v_pred = self.net(x_noisy, channels_list=[sigmas], **kwargs)
        return F.mse_loss(v_pred, v_target)

    def sample(
        self,
        num_chunks: int,
        num_items: int,
        num_steps: int,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        """Samples autoregressively `num_chunks` splits"""
        b, c, t, n = num_items, self.in_channels, self.length, self.num_splits
        start = default(start, lambda: self.sample_all(num_items, num_steps, **kwargs))
        assert start.shape == (b, c, t), "start has wrong shape"
        assert num_steps >= n, "num_steps must be greater than num_splits"

        s, l = num_steps // self.num_splits, self.split_length  # noqa
        sigmas = torch.linspace(1, 0, s * n, device=self.device)
        sigmas = repeat(sigmas, "(n s) -> s b 1 (n l)", b=b, l=l, n=n)
        sigmas = torch.flip(sigmas, dims=[-1])  # Lowest noise level first
        sigmas = F.pad(sigmas, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # Add index i+1
        sigmas[-1, :, :, l:] = sigmas[0, :, :, :-l]  # Loop back at index i+1
        alphas, betas = self.get_alpha_beta(sigmas)

        # Noise start and set as starting chunks
        start_noisy = alphas[0] * start + betas[0] * torch.randn_like(start)
        chunks = list(start_noisy.chunk(chunks=n, dim=-1))

        progress_bar = tqdm(range(num_chunks), disable=not show_progress)
        for _ in progress_bar:
            # Get last n chunks
            x_noisy = torch.cat(chunks[-n:], dim=-1)
            print(x_noisy.shape)
            # Decrease noise by one level
            for i in range(s):
                v_pred = self.net(x_noisy, channels_list=[sigmas[i]], **kwargs)
                x_pred = alphas[i] * x_noisy - betas[i] * v_pred
                noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
                x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred

            # Update chunks
            chunks[-n:] = list(x_noisy.chunk(chunks=n, dim=-1))
            # Add fresh noise chunk
            chunks += [torch.randn((b, c, l), device=self.device)]

        return torch.cat(chunks[:-n], dim=-1)

    def sample_all(
        self, num_items: int, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        """Samples a single block of length `length` in one go"""
        b, c, t = num_items, self.in_channels, self.length
        sigmas = torch.linspace(1, 0, num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, "i -> i b 1 t", b=b, t=t)
        alphas, betas = self.get_alpha_beta(sigmas)
        x_noisy = torch.randn((b, c, t), device=self.device) * sigmas[0]
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            v_pred = self.net(x_noisy, channels_list=[sigmas[i]], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0,0,0]:.2f})")

        return x_noisy


""" Normal V-diffusion for comparison """


class Diffusion(nn.Module):
    def __init__(
        self, in_channels: int, length: int, unet_type: Type[UNet1d] = UNet1d, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.length = length
        self.net = unet_type(
            in_channels=self.in_channels,
            length=length,
            **kwargs,
        )

    @property
    def device(self):
        return next(self.net.parameters()).device

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        assert x.shape[-1] == self.length, "input length must match length"
        b, device, dtype = x.shape[0], x.device, x.dtype
        # Sample amount of noise to add for each split
        sigmas = torch.rand((b,), device=device, dtype=dtype)
        sigmas_batch = rearrange(sigmas, "b -> b 1 1")
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x
        # Denoise and return loss
        v_pred = self.net(x_noisy, time=sigmas, **kwargs)
        return F.mse_loss(v_pred, v_target)

    def sample(
        self, num_items: int, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        """Samples a single block of length `length` in one go"""
        b, c, t = num_items, self.in_channels, self.length
        sigmas = torch.linspace(1, 0, num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = rearrange(sigmas, "i b -> i b 1 1")
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = torch.randn((b, c, t), device=self.device) * sigmas_batch[0]
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            v_pred = self.net(x_noisy, time=sigmas[i], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")

        return x_noisy
