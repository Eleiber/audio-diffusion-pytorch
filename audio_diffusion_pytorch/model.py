from math import pi
from typing import Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from a_unet import (
    Attention,
    Conv,
    Downsample,
    FeedForward,
    Module,
    Packed,
    Repeat,
    ResnetBlock,
    Select,
    Sequential,
    Skip,
    T,
    Ts,
    Upsample,
)
from einops import repeat
from torch import Tensor
from tqdm import tqdm

from .utils import default, downsample, upsample

"""
Models
"""


def UNet1d(
    in_channels: int,
    embedding_features: int,
    channels: Sequence[int],
    factors: Sequence[int],
    blocks: Sequence[int],
    attentions: Sequence[int],
    attention_heads: int,
    attention_features: int,
    attention_multiplier: int,
    out_channels: Optional[int] = None,
):
    # Check that all lists have matching lengths
    n_layers = len(channels)
    assert all(len(xs) == n_layers for xs in (factors, blocks, attentions))
    out_channels = default(out_channels, in_channels)

    # Selects only first module input, ignores context
    S = Select(lambda x, context: x)

    # Pre-initalize attention and feed-forward types with parameters
    A = T(Attention)(head_features=attention_features, num_heads=attention_heads)
    C = T(A)(context_features=embedding_features)  # Same as A but with context features
    F = T(FeedForward)(multiplier=attention_multiplier)

    def Stack(channels: int, n_blocks: int, n_attentions: int):
        # Build resnet stack type
        Block = T(ResnetBlock)(dim=1, in_channels=channels, out_channels=channels)
        ResnetStack = S(Repeat(Block, times=n_blocks))
        # Build attention, cross att, and feed forward types (ignoring context in A & F)
        Attention = T(S(A))(features=channels)
        CrossAttention = T(C)(features=channels)
        FeedForward = T(S(F))(features=channels)
        # Build transformer type
        Transformer = Ts(Sequential)(Attention, CrossAttention, FeedForward)
        TransformerStack = Repeat(Transformer, times=n_attentions)
        # Instantiate sequential resnet stack and transformer stack
        return Sequential(ResnetStack(), Packed(TransformerStack()))

    def Net(i: int):
        if i == n_layers:
            return S(nn.Identity)()
        ch_in, ch_out = channels[i - 1], channels[i - 1]
        if i == 0:
            ch_in, ch_out = in_channels, out_channels  # type: ignore
        f, ch = factors[i], channels[i]
        net = Sequential(
            S(Downsample)(dim=1, factor=f, in_channels=ch_in, out_channels=ch),
            Stack(channels=channels[i], n_blocks=blocks[i], n_attentions=attentions[i]),
            Net(i + 1),
            Stack(channels=channels[i], n_blocks=blocks[i], n_attentions=attentions[i]),
            S(Upsample)(dim=1, factor=f, in_channels=ch, out_channels=ch_out),
        )
        if i == 0:
            return net
        skip = Skip(lambda x, y: torch.cat([x, y], dim=1))(net)
        merge = S(Conv)(dim=1, in_channels=ch_in * 2, out_channels=ch_in, kernel_size=1)
        return Sequential(skip, merge)

    unet = Net(0)

    def fn(x: Tensor, embedding: Tensor) -> Tensor:
        return unet(x, embedding)

    return Module([unet], fn)


UNet1dT = T(UNet1d)


class T5Embedder(nn.Module):
    def __init__(self, model: str = "t5-base", max_length: int = 64):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.transformer = T5EncoderModel.from_pretrained(model)
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, texts: Sequence[str]) -> Tensor:

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        device = next(self.transformer.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        self.transformer.eval()

        embedding = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )["last_hidden_state"]

        return embedding


"""
Diffusion
"""


class DiffusionAR1d(nn.Module):
    """Autoregressive Diffusion Generation"""

    def __init__(
        self,
        in_channels: int,
        length: int,
        num_splits: int,
        unet_type: Type[nn.Module] = UNet1d,  # type: ignore
        **kwargs,
    ):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.split_length = length // num_splits
        self.net = unet_type(
            in_channels=in_channels + 1,  # type: ignore
            out_channels=in_channels,  # type: ignore
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
        # Denoise and return loss while conditioning on reupsampled channels
        channels = torch.cat([x_noisy, sigmas], dim=1)
        v_pred = self.net(channels, **kwargs)
        return F.mse_loss(v_pred, v_target)

    def get_sigmas_ladder(self, num_items: int, num_steps_per_split: int) -> Tensor:
        b, n, l, i = num_items, self.num_splits, self.split_length, num_steps_per_split
        n_half = n // 2  # Only half ladder, rest is zero, to leave some context
        sigmas = torch.linspace(1, 0, i * n_half, device=self.device)
        sigmas = repeat(sigmas, "(n i) -> i b 1 (n l)", b=b, l=l, n=n_half)
        sigmas = torch.flip(sigmas, dims=[-1])  # Lowest noise level first
        sigmas = F.pad(sigmas, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # Add index i+1
        sigmas[-1, :, :, l:] = sigmas[0, :, :, :-l]  # Loop back at index i+1
        return torch.cat([torch.zeros_like(sigmas), sigmas], dim=-1)

    def sample_loop(
        self, current: Tensor, sigmas: Tensor, show_progress: bool = False, **kwargs
    ) -> Tensor:
        num_steps = sigmas.shape[0] - 1
        alphas, betas = self.get_alpha_beta(sigmas)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            channels = torch.cat([current, sigmas[i]], dim=1)
            v_pred = self.net(channels, **kwargs)
            x_pred = alphas[i] * current - betas[i] * v_pred
            noise_pred = betas[i] * current + alphas[i] * v_pred
            current = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0,0,0]:.2f})")

        return current

    def sample_start(self, num_items: int, num_steps: int, **kwargs) -> Tensor:
        b, c, t = num_items, self.in_channels, self.length
        # Same sigma schedule over all chunks
        sigmas = torch.linspace(1, 0, num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, "i -> i b 1 t", b=b, t=t)
        noise = torch.randn((b, c, t), device=self.device) * sigmas[0]
        # Sample start
        return self.sample_loop(current=noise, sigmas=sigmas, **kwargs)

    def sample(
        self,
        num_items: int,
        num_chunks: int,
        num_steps: int,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        assert_message = f"required at least {self.num_splits} chunks"
        assert num_chunks >= self.num_splits, assert_message

        # Sample initial chunks
        start = self.sample_start(num_items=num_items, num_steps=num_steps, **kwargs)
        # Return start if only num_splits chunks
        if num_chunks == self.num_splits:
            return start

        # Get sigmas for autoregressive ladder
        b, n = num_items, self.num_splits
        assert num_steps >= n, "num_steps must be greater than num_splits"
        sigmas = self.get_sigmas_ladder(
            num_items=b,
            num_steps_per_split=num_steps // self.num_splits,
        )
        alphas, betas = self.get_alpha_beta(sigmas)

        # Noise start to match ladder and set starting chunks
        start_noise = alphas[0] * start + betas[0] * torch.randn_like(start)
        chunks = list(start_noise.chunk(chunks=n, dim=-1))

        # Loop over ladder shifts
        num_shifts = num_chunks  # - self.num_splits
        progress_bar = tqdm(range(num_shifts), disable=not show_progress)

        for j in progress_bar:
            # Decrease ladder noise of last n chunks
            updated = self.sample_loop(
                current=torch.cat(chunks[-n:], dim=-1), sigmas=sigmas, **kwargs
            )
            # Update chunks
            chunks[-n:] = list(updated.chunk(chunks=n, dim=-1))
            # Add fresh noise chunk
            shape = (b, self.in_channels, self.split_length)
            chunks += [torch.randn(shape, device=self.device)]

        return torch.cat(chunks[:num_chunks], dim=-1)


class DiffusionARU1d(nn.Module):
    """Autoregressive Diffusion Upsampler"""

    def __init__(
        self,
        in_channels: int,
        length: int,
        num_splits: int,
        upsample: int,
        unet_type: Type[nn.Module] = UNet1d,  # type: ignore
        **kwargs,
    ):
        super().__init__()
        assert length % num_splits == 0, "length must be divisible by num_splits"
        self.length = length
        self.in_channels = in_channels
        self.num_splits = num_splits
        self.upsample = upsample
        self.split_length = length // num_splits
        self.net = unet_type(
            in_channels=in_channels * 2 + 1,  # type: ignore
            out_channels=in_channels,  # type: ignore
            **kwargs,
        )

    @property
    def device(self):
        return next(self.net.parameters()).device

    def reupsample(self, x: Tensor) -> Tensor:
        x = x.clone()
        x = downsample(x, factor=self.upsample)
        x = upsample(x, factor=self.upsample)
        return x

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
        # Denoise and return loss while conditioning on reupsampled channels
        channels = torch.cat([x_noisy, self.reupsample(x), sigmas], dim=1)
        v_pred = self.net(channels, **kwargs)
        return F.mse_loss(v_pred, v_target)

    def sample_loop(
        self,
        current: Tensor,
        resampled: Tensor,
        sigmas: Tensor,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        num_steps = sigmas.shape[0] - 1
        alphas, betas = self.get_alpha_beta(sigmas)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            channels = torch.cat([current, resampled, sigmas[i]], dim=1)
            v_pred = self.net(channels, **kwargs)
            x_pred = alphas[i] * current - betas[i] * v_pred
            noise_pred = betas[i] * current + alphas[i] * v_pred
            current = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0,0,0]:.2f})")

        return current

    def get_sigmas_ladder(self, num_items: int, num_steps_per_split: int) -> Tensor:
        b, n, l, i = num_items, self.num_splits, self.split_length, num_steps_per_split
        n_half = n // 2  # Only half ladder, rest is zero, to leave some context
        sigmas = torch.linspace(1, 0, i * n_half, device=self.device)
        sigmas = repeat(sigmas, "(n i) -> i b 1 (n l)", b=b, l=l, n=n_half)
        sigmas = torch.flip(sigmas, dims=[-1])  # Lowest noise level first
        sigmas = F.pad(sigmas, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # Add index i+1
        sigmas[-1, :, :, l:] = sigmas[0, :, :, :-l]  # Loop back at index i+1
        return torch.cat([torch.zeros_like(sigmas), sigmas], dim=-1)

    def sample_start(self, resampled: Tensor, num_steps: int, **kwargs) -> Tensor:
        b, c, t = resampled.shape[0], self.in_channels, self.length
        # Same sigma schedule over all chunks
        sigmas = torch.linspace(1, 0, num_steps + 1, device=self.device)
        sigmas = repeat(sigmas, "i -> i b 1 t", b=b, t=t)
        noise = torch.randn((b, c, t), device=self.device) * sigmas[0]
        # Sample start
        return self.sample_loop(
            current=noise, resampled=resampled, sigmas=sigmas, **kwargs
        )

    def sample(
        self,
        undersampled: Tensor,
        num_steps: int,
        start: Optional[Tensor] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        c, sl = self.in_channels, self.split_length
        # Compute num_chunks from undersampled length
        num_chunks = (undersampled.shape[-1] * self.upsample) // self.split_length
        assert_message = f"required at least {self.num_splits} chunks"
        assert num_chunks >= self.num_splits, assert_message

        # Sample initial chunks
        t = self.length
        resampled = upsample(undersampled, factor=self.upsample)
        start = self.sample_start(
            resampled=resampled[:, :, :t], num_steps=num_steps, **kwargs
        )
        # Return start if only num_splits chunks
        if num_chunks == self.num_splits:
            return start

        # Get sigmas for autoregressive ladder
        b, n = undersampled.shape[0], self.num_splits
        assert num_steps >= n, "num_steps must be greater than num_splits"
        sigmas = self.get_sigmas_ladder(
            num_items=b,
            num_steps_per_split=num_steps // self.num_splits,
        )
        alphas, betas = self.get_alpha_beta(sigmas)

        # Set starting chunks
        start_noise = alphas[0] * start + betas[0] * torch.randn_like(start)
        chunks = list(start_noise.chunk(chunks=n, dim=-1))
        chunks_resampled = list(resampled.chunk(chunks=num_chunks, dim=-1))
        # Zero chunks at the end
        chunks_resampled += [torch.zeros_like(chunks_resampled[0])] * (n // 2)

        # Loop over ladder shifts
        progress_bar = tqdm(range(num_chunks - (n // 2)), disable=not show_progress)
        for j in progress_bar:
            # Decrease ladder noise of last n chunks
            updated = self.sample_loop(
                current=torch.cat(chunks[-n:], dim=-1),
                resampled=torch.cat(chunks_resampled[j : j + n], dim=-1),
                sigmas=sigmas,
                **kwargs,
            )
            # Update chunks
            chunks[-n:] = list(updated.chunk(chunks=n, dim=-1))
            # Add fresh noise chunk
            chunks += [torch.randn((b, c, sl), device=self.device)]

        return torch.cat(chunks[:num_chunks], dim=-1)
