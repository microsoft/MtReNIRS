from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int = 4, num_bands: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.num_bands = num_bands
        frequencies = (2.0 ** torch.arange(num_bands).float()) * torch.pi
        self.register_buffer("freqs", frequencies, persistent=False)

    @property
    def out_dim(self) -> int:
        return self.input_dim * self.num_bands * 2

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projection = values.unsqueeze(-1) * self.freqs.view(*([1] * values.dim()), -1)
        projection = projection.flatten(start_dim=-2)
        return torch.cat([projection.sin(), projection.cos()], dim=-1)


class SpatialChromophoreEmbedding(nn.Module):
    def __init__(self, input_dim: int, num_bands: int, hidden: int, out_dim: int):
        super().__init__()
        self.fourier = FourierFeatures(input_dim, num_bands)
        self.mlp = nn.Sequential(
            nn.Linear(self.fourier.out_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.fourier(coordinates))


class SignalEmbedding(nn.Module):
    def __init__(self, samples: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(samples, out_dim)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.proj(signal)


class TokenEmbedding(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model = config["model"]
        dimension = int(model["d_model"])
        self.signal = SignalEmbedding(int(config["data"]["t_win"]), dimension)
        self.spatial = SpatialChromophoreEmbedding(
            int(model["fourier_input_dim"]),
            int(model["fourier_num_bands"]),
            int(model["spatial_mlp_hidden"]),
            dimension,
        )

    def forward(self, signal: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        return self.signal(signal) + self.spatial(coordinates)


def _rope_angles(positions: torch.Tensor, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inverse = 1.0 / (
        base ** (torch.arange(half, device=positions.device, dtype=torch.float32) / half)
    )
    angles = positions.float().unsqueeze(-1) * inverse.view(1, 1, half)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)


def _apply_rope(values: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    half = values.shape[-1] // 2
    rotated = torch.cat([-values[..., half:], values[..., :half]], dim=-1)
    return values * cosine + rotated * sine


class MultiHeadAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float, rope_base: float, qk_norm: bool):
        super().__init__()
        if dimension % heads:
            raise ValueError("model dimension must be divisible by attention heads")
        self.d_model = dimension
        self.n_heads = heads
        self.head_dim = dimension // heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.proj = nn.Linear(dimension, dimension)
        self.dropout = dropout
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, tokens, _ = values.shape
        qkv = self.qkv(values).reshape(batch, tokens, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        query, key = self.q_norm(query), self.k_norm(key)
        cosine, sine = _rope_angles(positions, self.head_dim, self.rope_base)
        query, key = _apply_rope(query, cosine, sine), _apply_rope(key, cosine, sine)
        bias = None
        if key_padding_mask is not None:
            bias = torch.zeros(batch, 1, 1, tokens, device=values.device, dtype=query.dtype)
            bias = bias.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        return self.proj(output.transpose(1, 2).reshape(batch, tokens, self.d_model))


class TransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model, encoder = config["model"], config["model"]["encoder"]
        dimension = int(model["d_model"])
        dropout = float(encoder["dropout"])
        self.norm1 = nn.LayerNorm(dimension)
        self.attn = MultiHeadAttention(
            dimension,
            int(encoder["n_heads"]),
            dropout,
            float(model["rope_base"]),
            bool(model.get("qk_norm", False)),
        )
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, int(encoder["mlp_hidden"])),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(encoder["mlp_hidden"]), dimension),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values + self.drop(self.attn(self.norm1(values), positions, mask))
        return values + self.mlp(self.norm2(values))


class TransformerEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        encoder = config["model"]["encoder"]
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(int(encoder["depth"]))])
        self.norm = nn.LayerNorm(int(config["model"]["d_model"]))

    def forward(self, values: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values, positions, mask)
        return self.norm(values)


class MtReNIRSEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed = TokenEmbedding(config)
        self.encoder = TransformerEncoder(config)

    def forward(
        self,
        signal: torch.Tensor,
        coordinates: torch.Tensor,
        window_indices: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.embed(signal, coordinates)
        encoded = self.encoder(tokens, window_indices, valid_tokens)
        return encoded * valid_tokens.unsqueeze(-1)
