from itertools import accumulate
from math import floor, log, pi
from typing import Any, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
from torch import Tensor, einsum

from .utils import closest_power_2, default, exists, groupby, to_list

"""
Convolutional Blocks
"""


def Conv1d(*args, **kwargs) -> nn.Module:
    return nn.Conv1d(*args, **kwargs)


def Downsample1d(in_channels: int, out_channels: int, factor: int) -> nn.Module:
    return Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor,
        stride=factor,
    )


def Upsample1d(in_channels: int, out_channels: int, factor: int) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=factor, mode="nearest"),
        Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        ),
    )


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 1,
        num_groups: int = 8,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            if use_norm
            else nn.Identity()
        )
        self.activation = nn.SiLU()
        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )

    def forward(
        self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tensor:
        x = self.groupnorm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.activation(x)
        return self.project(x)


class MappingToScaleShift(nn.Module):
    def __init__(
        self,
        features: int,
        channels: int,
    ):
        super().__init__()

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=features, out_features=channels * 2),
        )

    def forward(self, mapping: Tensor) -> Tuple[Tensor, Tensor]:
        scale_shift = self.to_scale_shift(mapping)
        scale_shift = rearrange(scale_shift, "b c -> b c 1")
        scale, shift = scale_shift.chunk(2, dim=1)
        return scale, shift


class ResnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        use_norm: bool = True,
        num_groups: int = 8,
        context_mapping_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_mapping = exists(context_mapping_features)

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            use_norm=use_norm,
            num_groups=num_groups,
        )

        if self.use_mapping:
            assert exists(context_mapping_features)
            self.to_scale_shift = MappingToScaleShift(
                features=context_mapping_features, channels=out_channels
            )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            use_norm=use_norm,
            num_groups=num_groups,
        )

        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None) -> Tensor:
        assert_message = "context mapping required if context_mapping_features > 0"
        assert not (self.use_mapping ^ exists(mapping)), assert_message

        h = self.block1(x)

        scale_shift = None
        if self.use_mapping:
            scale_shift = self.to_scale_shift(mapping)

        h = self.block2(h, scale_shift=scale_shift)

        return h + self.to_out(x)


class Patcher(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        context_mapping_features: Optional[int] = None,
    ):
        super().__init__()
        assert_message = f"out_channels must be divisible by patch_size ({patch_size})"
        assert out_channels % patch_size == 0, assert_message
        self.patch_size = patch_size

        self.block = ResnetBlock1d(
            in_channels=in_channels,
            out_channels=out_channels // patch_size,
            num_groups=1,
            context_mapping_features=context_mapping_features,
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None) -> Tensor:
        x = self.block(x, mapping)
        x = rearrange(x, "b c (l p) -> b (c p) l", p=self.patch_size)
        return x


class Unpatcher(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        context_mapping_features: Optional[int] = None,
    ):
        super().__init__()
        assert_message = f"in_channels must be divisible by patch_size ({patch_size})"
        assert in_channels % patch_size == 0, assert_message
        self.patch_size = patch_size

        self.block = ResnetBlock1d(
            in_channels=in_channels // patch_size,
            out_channels=out_channels,
            num_groups=1,
            context_mapping_features=context_mapping_features,
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None) -> Tensor:
        x = rearrange(x, " b (c p) l -> b c (l p) ", p=self.patch_size)
        x = self.block(x, mapping)
        return x


"""
Attention Components
"""


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )


class FixedEmbedding(nn.Module):
    def __init__(self, max_length: int, features: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, length, device = *x.shape[0:2], x.device
        assert_message = "Input sequence length must be <= max_length"
        assert length <= self.max_length, assert_message
        position = torch.arange(length, device=device)
        fixed_embedding = self.embedding(position)
        fixed_embedding = repeat(fixed_embedding, "n d -> b n d", b=batch_size)
        return fixed_embedding


class AttentionBase(nn.Module):
    def __init__(self, features: int, *, head_features: int, num_heads: int):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        mid_features = head_features * num_heads
        self.to_out = nn.Linear(in_features=mid_features, out_features=features)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)
        # Compute similarity matrix
        sim = einsum("... n d, ... m d -> ... n m", q, k)
        sim = sim * self.scale
        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1)
        # Compute values
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        context_features: Optional[int] = None,
        use_positional_embedding: bool = False,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self.context_features = context_features
        self.use_positional_embedding = use_positional_embedding
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.max_length = max_length
        if use_positional_embedding:
            assert exists(max_length)
            self.positional_embedding = FixedEmbedding(
                max_length=max_length, features=features
            )

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        self.attention = AttentionBase(
            features, num_heads=num_heads, head_features=head_features
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        if self.use_positional_embedding:
            x = x + self.positional_embedding(x)
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        # Compute and return attention
        return self.attention(q, k, v)


"""
Transformer Blocks
"""


class TransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_features: Optional[int] = None,
        use_positional_embedding: bool = False,
        max_length: Optional[int] = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = Attention(
            features=features,
            num_heads=num_heads,
            head_features=head_features,
            use_positional_embedding=use_positional_embedding,
            max_length=max_length,
        )

        if self.use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_positional_embedding=use_positional_embedding,
                max_length=max_length,
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        x = self.attention(x) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context) + x
        x = self.feed_forward(x) + x
        return x


"""
Transformers
"""


class Transformer1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_features: Optional[int] = None,
        use_positional_embedding: bool = False,
        max_length: Optional[int] = None,
    ):
        super().__init__()

        self.to_in = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True),
            Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
            Rearrange("b c t -> b t c"),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    features=channels,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    context_features=context_features,
                    use_positional_embedding=use_positional_embedding,
                    max_length=max_length,
                )
                for i in range(num_layers)
            ]
        )

        self.to_out = nn.Sequential(
            Rearrange("b t c -> b c t"),
            Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        x = self.to_in(x)
        for block in self.blocks:
            x = block(x, context=context)
        x = self.to_out(x)
        return x


"""
Time Embeddings
"""


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device, half_dim = x.device, self.dim // 2
        emb = torch.tensor(log(10000) / (half_dim - 1), device=device)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


"""
Encoder/Decoder Components
"""


class DownsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_groups: int,
        num_layers: int,
        use_pre_downsample: bool = True,
        use_skip: bool = False,
        extract_channels: int = 0,
        context_channels: int = 0,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        attention_use_positional_embedding: bool = False,
        attention_max_length: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
    ):
        super().__init__()
        self.use_pre_downsample = use_pre_downsample
        self.use_skip = use_skip
        self.use_transformer = num_transformer_blocks > 0
        self.use_extract = extract_channels > 0
        self.use_context = context_channels > 0

        channels = out_channels if use_pre_downsample else in_channels

        self.downsample = Downsample1d(
            in_channels=in_channels, out_channels=out_channels, factor=factor
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + context_channels if i == 0 else channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    context_mapping_features=context_mapping_features,
                )
                for i in range(num_layers)
            ]
        )

        if self.use_transformer:
            assert (
                exists(attention_heads)
                and exists(attention_features)
                and exists(attention_multiplier)
            )
            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
                use_positional_embedding=attention_use_positional_embedding,
                max_length=attention_max_length,
            )

        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups,
            )

    def forward(
        self,
        x: Tensor,
        *,
        mapping: Optional[Tensor] = None,
        channels: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor, List[Tensor]], Tensor]:

        if self.use_pre_downsample:
            x = self.downsample(x)

        if self.use_context and exists(channels):
            x = torch.cat([x, channels], dim=1)

        skips = []
        for block in self.blocks:
            x = block(x, mapping=mapping)
            skips += [x] if self.use_skip else []

        if self.use_transformer:
            x = self.transformer(x, context=embedding)
            skips += [x] if self.use_skip else []

        if not self.use_pre_downsample:
            x = self.downsample(x)

        if self.use_extract:
            extracted = self.to_extracted(x)
            return x, extracted

        return (x, skips) if self.use_skip else x


class UpsampleBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int,
        num_layers: int,
        num_groups: int,
        use_pre_upsample: bool = False,
        use_skip: bool = False,
        skip_channels: int = 0,
        use_skip_scale: bool = False,
        extract_channels: int = 0,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        attention_use_positional_embedding: bool = False,
        attention_max_length: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_extract = extract_channels > 0
        self.use_pre_upsample = use_pre_upsample
        self.use_transformer = num_transformer_blocks > 0
        self.use_skip = use_skip
        self.skip_scale = 2 ** -0.5 if use_skip_scale else 1.0

        channels = out_channels if use_pre_upsample else in_channels

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + skip_channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    context_mapping_features=context_mapping_features,
                )
                for _ in range(num_layers)
            ]
        )

        if self.use_transformer:
            assert (
                exists(attention_heads)
                and exists(attention_features)
                and exists(attention_multiplier)
            )
            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
                use_positional_embedding=attention_use_positional_embedding,
                max_length=attention_max_length,
            )

        self.upsample = Upsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
        )

        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups,
            )

    def add_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([x, skip * self.skip_scale], dim=1)

    def forward(
        self,
        x: Tensor,
        *,
        skips: Optional[List[Tensor]] = None,
        mapping: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:

        if self.use_pre_upsample:
            x = self.upsample(x)

        for block in self.blocks:
            x = self.add_skip(x, skip=skips.pop()) if exists(skips) else x
            x = block(x, mapping=mapping)

        if self.use_transformer:
            x = self.transformer(x, context=embedding)

        if not self.use_pre_upsample:
            x = self.upsample(x)

        if self.use_extract:
            extracted = self.to_extracted(x)
            return x, extracted

        return x


class BottleneckBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_groups: int,
        num_transformer_blocks: int = 0,
        attention_heads: Optional[int] = None,
        attention_features: Optional[int] = None,
        attention_multiplier: Optional[int] = None,
        attention_use_positional_embedding: bool = False,
        attention_max_length: Optional[int] = None,
        context_mapping_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
    ):
        super().__init__()
        self.use_transformer = num_transformer_blocks > 0

        self.pre_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            context_mapping_features=context_mapping_features,
        )

        if self.use_transformer:
            assert (
                exists(attention_heads)
                and exists(attention_features)
                and exists(attention_multiplier)
            )
            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
                use_positional_embedding=attention_use_positional_embedding,
                max_length=attention_max_length,
            )

        self.post_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            context_mapping_features=context_mapping_features,
        )

    def forward(
        self,
        x: Tensor,
        *,
        mapping: Optional[Tensor] = None,
        embedding: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.pre_block(x, mapping=mapping)
        if self.use_transformer:
            x = self.transformer(x, context=embedding)
        x = self.post_block(x, mapping=mapping)
        return x


"""
UNet
"""


class UNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        attentions: Sequence[int],
        patch_size: int = 1,
        resnet_groups: int = 8,
        use_context_time: bool = True,
        use_skip_scale: bool = True,
        out_channels: Optional[int] = None,
        context_features: Optional[int] = None,
        context_features_multiplier: int = 4,
        context_channels: Optional[Sequence[int]] = None,
        context_embedding_features: Optional[int] = None,
        length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        # Args
        out_channels = default(out_channels, in_channels)
        attention_kwargs, kwargs = groupby("attention_", kwargs, keep_prefix=True)
        factors = to_list(factors)

        # Number of layers with checks
        num_layers = len(multipliers) - 1
        self.num_layers = num_layers
        assert (
            len(factors) == num_layers
            and len(attentions) >= num_layers
            and len(num_blocks) == num_layers
        )

        # Context time and context features
        self.use_context_time = use_context_time
        use_context_features = exists(context_features)
        self.use_context_features = use_context_features
        self.context_features = context_features

        context_mapping_features = None
        if use_context_time or use_context_features:
            context_mapping_features = channels * context_features_multiplier

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )

        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=channels, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(
                    in_features=context_features, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        # Context channels
        context_channels = list(default(context_channels, []))
        use_context_channels = len(context_channels) > 0
        self.use_context_channels = use_context_channels
        context_channels_pad_length = num_layers + 1 - len(context_channels)
        context_channels = context_channels + [0] * context_channels_pad_length
        self.context_channels = context_channels
        self.context_embedding_features = context_embedding_features
        if use_context_channels:
            has_context = [c > 0 for c in context_channels]
            self.has_context = has_context
            self.channels_ids = [sum(has_context[:i]) for i in range(len(has_context))]

        # Layer factor and length (used for fixed length posemb in attention)
        factors_cumulative = [*accumulate([patch_size] + factors, lambda x, y: x * y)]
        layer_lengths = [default(length, 0) // f for f in factors_cumulative]
        self.layer_lengths = layer_lengths

        # Check that all kwargs have been used
        assert not kwargs, f"Unknown arguments: {', '.join(list(kwargs.keys()))}"

        self.to_in = Patcher(
            in_channels=in_channels + context_channels[0],
            out_channels=channels * multipliers[0],
            patch_size=patch_size,
            context_mapping_features=context_mapping_features,
        )

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    context_mapping_features=context_mapping_features,
                    context_channels=context_channels[i + 1],
                    context_embedding_features=context_embedding_features,
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    num_groups=resnet_groups,
                    use_pre_downsample=True,
                    use_skip=True,
                    num_transformer_blocks=attentions[i],
                    attention_use_positional_embedding=(
                        attentions[i] > 0 and exists(length)
                    ),
                    attention_max_length=layer_lengths[i + 1],
                    **attention_kwargs,
                )
                for i in range(num_layers)
            ]
        )

        self.bottleneck = BottleneckBlock1d(
            channels=channels * multipliers[-1],
            context_mapping_features=context_mapping_features,
            context_embedding_features=context_embedding_features,
            num_groups=resnet_groups,
            num_transformer_blocks=attentions[-1],
            attention_use_positional_embedding=attentions[-1] > 0 and exists(length),
            attention_max_length=layer_lengths[-1],
            **attention_kwargs,
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i + 1],
                    out_channels=channels * multipliers[i],
                    context_mapping_features=context_mapping_features,
                    context_embedding_features=context_embedding_features,
                    num_layers=num_blocks[i] + (1 if attentions[i] else 0),
                    factor=factors[i],
                    num_groups=resnet_groups,
                    use_skip_scale=use_skip_scale,
                    use_pre_upsample=False,
                    use_skip=True,
                    skip_channels=channels * multipliers[i + 1],
                    num_transformer_blocks=attentions[i],
                    attention_use_positional_embedding=(
                        attentions[i] > 0 and exists(length)
                    ),
                    attention_max_length=layer_lengths[i + 1],
                    **attention_kwargs,
                )
                for i in reversed(range(num_layers))
            ]
        )

        self.to_out = Unpatcher(
            in_channels=channels * multipliers[0],
            out_channels=out_channels,
            patch_size=patch_size,
            context_mapping_features=context_mapping_features,
        )

    def get_channels(
        self, channels_list: Optional[Sequence[Tensor]] = None, layer: int = 0
    ) -> Optional[Tensor]:
        """Gets context channels at `layer` and checks that shape is correct"""
        use_context_channels = self.use_context_channels and self.has_context[layer]
        if not use_context_channels:
            return None
        assert exists(channels_list), "Missing context"
        # Get channels index (skipping zero channel contexts)
        channels_id = self.channels_ids[layer]
        # Get channels
        channels = channels_list[channels_id]
        message = f"Missing context for layer {layer} at index {channels_id}"
        assert exists(channels), message
        # Check channels
        num_channels = self.context_channels[layer]
        message = f"Expected context with {num_channels} channels at idx {channels_id}"
        assert channels.shape[1] == num_channels, message
        return channels

    def get_mapping(
        self, time: Optional[Tensor] = None, features: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]
        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = reduce(torch.stack(items), "n b m -> b m", "sum")
            mapping = self.to_mapping(mapping)
        return mapping

    def forward(
        self,
        x: Tensor,
        time: Optional[Tensor] = None,
        *,
        features: Optional[Tensor] = None,
        channels_list: Optional[Sequence[Tensor]] = None,
        embedding: Optional[Tensor] = None,
    ) -> Tensor:
        channels = self.get_channels(channels_list, layer=0)
        # Concat context channels at layer 0 if provided
        x = torch.cat([x, channels], dim=1) if exists(channels) else x
        # Compute mapping from time and features
        mapping = self.get_mapping(time, features)
        x = self.to_in(x, mapping)
        skips_list = [x]

        for i, downsample in enumerate(self.downsamples):
            channels = self.get_channels(channels_list, layer=i + 1)
            x, skips = downsample(
                x, mapping=mapping, channels=channels, embedding=embedding
            )
            skips_list += [skips]

        x = self.bottleneck(x, mapping=mapping, embedding=embedding)

        for i, upsample in enumerate(self.upsamples):
            skips = skips_list.pop()
            x = upsample(x, skips=skips, mapping=mapping, embedding=embedding)

        x += skips_list.pop()
        x = self.to_out(x, mapping)
        return x


""" Conditioning Modules """


def rand_bool(shape: Any, proba: float, device: Any = None) -> Tensor:
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


def UNetCFG1dType(
    context_embedding_max_length: int,
    context_embedding_features: int,
    type: Type[UNet1d] = UNet1d,
) -> Type[UNet1d]:
    """UNet1d with Classifier-Free Guidance"""

    class UNetCFG1d(type):  # type: ignore
        def __init__(self, **kwargs):
            super().__init__(
                context_embedding_features=context_embedding_features, **kwargs
            )
            self.fixed_embedding = FixedEmbedding(
                max_length=context_embedding_max_length,
                features=context_embedding_features,
            )

        def forward(  # type: ignore
            self,
            x: Tensor,
            time: Optional[Tensor] = None,
            *,
            embedding: Optional[Tensor] = None,
            embedding_scale: float = 1.0,
            embedding_mask_proba: float = 0.0,
            **kwargs,
        ) -> Tensor:
            assert exists(embedding), "embedding required when using CFG"
            b, device = embedding.shape[0], embedding.device
            fixed_embedding = self.fixed_embedding(embedding)

            if embedding_mask_proba > 0.0:
                # Randomly mask embedding
                batch_mask = rand_bool(
                    shape=(b, 1, 1), proba=embedding_mask_proba, device=device
                )
                embedding = torch.where(batch_mask, fixed_embedding, embedding)

            if embedding_scale != 1.0:
                # Compute both normal and fixed embedding outputs
                out = super().forward(x, time, embedding=embedding, **kwargs)
                out_masked = super().forward(
                    x, time, embedding=fixed_embedding, **kwargs
                )
                # Scale conditional output using classifier-free guidance
                return out_masked + (out - out_masked) * embedding_scale
            else:
                return super().forward(x, time, embedding=embedding, **kwargs)

    return UNetCFG1d


def UNetCQT1dType(
    num_octaves: int,
    num_bins_per_octave: int,
    power_of_2_length: bool = True,
    use_on_context: bool = True,
    type: Type[UNet1d] = UNet1d,
    **transform_kwargs,
) -> Type[UNet1d]:
    class UNetCQT1d(type):  # type: ignore
        def __init__(
            self,
            in_channels: int,
            length: int,
            context_channels: Optional[Sequence[int]] = None,
            **kwargs,
        ):
            from cqt_pytorch import CQT

            transform = CQT(
                num_octaves=num_octaves,
                num_bins_per_octave=num_bins_per_octave,
                power_of_2_length=power_of_2_length,
                **transform_kwargs,
            )
            self.in_channels = in_channels

            if use_on_context and exists(context_channels) and context_channels[0] > 0:
                context_channels[0] *= num_octaves * num_bins_per_octave * 2  # type: ignore # noqa

            super().__init__(
                in_channels=in_channels * num_octaves * num_bins_per_octave * 2,
                length=length // transform.block_length * transform.max_window_length,
                context_channels=context_channels,
                **kwargs,
            )
            self.transform = transform

        def encode(self, x: Tensor) -> Tensor:
            x = torch.view_as_real(self.transform.encode(x))
            x = rearrange(x, "b c k l i -> b (c k i) l")
            return x

        def decode(self, x: Tensor) -> Tensor:
            x = rearrange(x, "b (c k i) l -> b c k l i", c=self.in_channels, i=2)
            x = self.transform.decode(torch.view_as_complex(x.contiguous()))
            return x

        def forward(
            self,
            x: Tensor,
            *args,
            channels_list: Optional[Sequence[Tensor]] = None,
            **kwargs,
        ) -> Tensor:
            if use_on_context and exists(channels_list) and len(channels_list) > 0:
                channels_list[0] = self.encode(channels_list[0])  # type: ignore # noqa
            x = self.encode(x)
            y = super().forward(x, *args, channels_list=channels_list, **kwargs)
            y = self.decode(y)
            return y

    return UNetCQT1d


class LT(nn.Module):
    """Learned Transform"""

    def __init__(
        self,
        num_channels: int,
        num_filters: int,
        window_length: int,
        stride: int,
    ):
        super().__init__()
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=window_length,
            stride=stride,
            padding=0,
            bias=False,
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.conv(x)

    def decode(self, x: Tensor) -> Tensor:
        return F.conv_transpose1d(input=x, weight=self.conv.weight, stride=self.stride)


def UNetLT1dType(
    num_filters: int,
    window_length: int,
    use_on_context: bool = True,
    type: Type[UNet1d] = UNet1d,
    **transform_kwargs,
) -> Type[UNet1d]:
    class UNetLT1d(type):  # type: ignore
        def __init__(
            self,
            in_channels: int,
            length: int,
            context_channels: Optional[Sequence[int]] = None,
            **kwargs,
        ):
            transform = LT(
                num_channels=in_channels,
                num_filters=num_filters,
                window_length=window_length,
                stride=window_length,
                **transform_kwargs,
            )

            if use_on_context and exists(context_channels) and context_channels[0] > 0:
                assert context_channels[0] == in_channels
                context_channels[0] = num_filters  # type: ignore # noqa

            super().__init__(
                in_channels=num_filters,
                length=length // window_length,
                context_channels=context_channels,
                **kwargs,
            )
            self.transform = transform

        def forward(
            self,
            x: Tensor,
            *args,
            channels_list: Optional[Sequence[Tensor]] = None,
            **kwargs,
        ) -> Tensor:
            if use_on_context and exists(channels_list) and len(channels_list) > 0:
                channels_list[0] = self.transform.encode(channels_list[0])  # type: ignore # noqa
            x = self.transform.encode(x)
            y = super().forward(x, *args, channels_list=channels_list, **kwargs)
            y = self.transform.decode(y)
            return y

    return UNetLT1d


class STFT(nn.Module):
    """Helper for torch stft and istft"""

    def __init__(
        self,
        num_fft: int = 1023,
        hop_length: int = 256,
        window_length: Optional[int] = None,
        length: Optional[int] = None,
        use_complex: bool = False,
    ):
        super().__init__()
        self.num_fft = num_fft
        self.hop_length = default(hop_length, floor(num_fft // 4))
        self.window_length = default(window_length, num_fft)
        self.length = length
        self.register_buffer("window", torch.hann_window(self.window_length))
        self.use_complex = use_complex

    def encode(self, wave: Tensor) -> Tuple[Tensor, Tensor]:
        b = wave.shape[0]
        wave = rearrange(wave, "b c t -> (b c) t")

        stft = torch.stft(
            wave,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            return_complex=True,
            normalized=True,
        )

        if self.use_complex:
            # Returns real and imaginary
            stft_a, stft_b = stft.real, stft.imag
        else:
            # Returns magnitude and phase matrices
            magnitude, phase = torch.abs(stft), torch.angle(stft)
            stft_a, stft_b = magnitude, phase

        return rearrange_many((stft_a, stft_b), "(b c) f l -> b c f l", b=b)

    def decode(self, stft_a: Tensor, stft_b: Tensor) -> Tensor:
        b, l = stft_a.shape[0], stft_a.shape[-1]  # noqa
        length = closest_power_2(l * self.hop_length)

        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b c f l -> (b c) f l")

        if self.use_complex:
            real, imag = stft_a, stft_b
        else:
            magnitude, phase = stft_a, stft_b
            real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)

        stft = torch.stack([real, imag], dim=-1)

        wave = torch.istft(
            stft,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            length=default(self.length, length),
            normalized=True,
        )

        return rearrange(wave, "(b c) t -> b c t", b=b)

    def encode1d(
        self, wave: Tensor, stacked: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        stft_a, stft_b = self.encode(wave)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b c f l -> b (c f) l")
        return torch.cat((stft_a, stft_b), dim=1) if stacked else (stft_a, stft_b)

    def decode1d(self, stft_pair: Tensor) -> Tensor:
        f = self.num_fft // 2 + 1
        stft_a, stft_b = stft_pair.chunk(chunks=2, dim=1)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b (c f) l -> b c f l", f=f)
        return self.decode(stft_a, stft_b)


def UNetSTFT1dType(
    num_fft: int,
    hop_length: int,
    use_on_context: bool = True,
    type: Type[UNet1d] = UNet1d,
    **transform_kwargs,
) -> Type[UNet1d]:
    class UNetSTFT1d(type):  # type: ignore
        def __init__(
            self,
            in_channels: int,
            length: int,
            context_channels: Optional[Sequence[int]] = None,
            **kwargs,
        ):
            transform = STFT(num_fft=num_fft, hop_length=hop_length, **transform_kwargs)
            transform_channels = (num_fft // 2 + 1) * 2
            if use_on_context and exists(context_channels) and context_channels[0] > 0:
                context_channels[0] *= transform_channels  # type: ignore # noqa

            super().__init__(
                in_channels=in_channels * transform_channels,
                length=length // hop_length,
                context_channels=context_channels,
                **kwargs,
            )
            self.transform = transform

        def forward(
            self,
            x: Tensor,
            *args,
            channels_list: Optional[Sequence[Tensor]] = None,
            **kwargs,
        ) -> Tensor:
            if use_on_context and exists(channels_list) and len(channels_list) > 0:
                channels_list[0] = self.transform.encode1d(channels_list[0])  # type: ignore # noqa
            x = self.transform.encode1d(x)  # type: ignore
            y = super().forward(x, *args, channels_list=channels_list, **kwargs)
            y = self.transform.decode1d(y)
            return y

    return UNetSTFT1d


def XUNet1dType(types: Sequence[Type[UNet1d]]) -> Type[UNet1d]:
    class XUNet1d(*types):  # type: ignore
        pass

    return XUNet1d


def XUNet1d(*types: Tuple[Type[UNet1d]], **kwargs) -> UNet1d:
    return XUNet1dType(*types)(**kwargs)


class T5Embedder(nn.Module):
    def __init__(self, model: str = "t5-base", max_length: int = 64):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.transformer = T5EncoderModel.from_pretrained(model)
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, texts: List[str]) -> Tensor:

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


class NumberEmbedder(nn.Module):
    def __init__(
        self,
        features: int,
        dim: int = 256,
    ):
        super().__init__()
        self.features = features
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: Union[List[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        x = embedding.view(*shape, self.features)
        return x  # type: ignore
