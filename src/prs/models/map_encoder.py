"""Map encoding (design doc §18): a PointNet/VectorNet-style polyline encoder, and attention
pooling of the polyline embeddings with the target embedding as the query."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PolylineEncoder(nn.Module):
    """Shared point MLP, masked max pool over each chunk's points, then a projection plus a
    map-type embedding.

    Padded points are zeroed before the MLP and excluded from the pool, and the embeddings of
    all-padding chunks are zeroed, so padded values (and padded chunks' types) never reach the
    output. Returns embeddings [B, M, out_dim] and the chunk mask [B, M].
    """

    def __init__(self, input_dim: int, point_dim: int, out_dim: int, num_types: int) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, point_dim // 2),
            nn.LayerNorm(point_dim // 2),
            nn.ReLU(),
            nn.Linear(point_dim // 2, point_dim),
            nn.LayerNorm(point_dim),
            nn.ReLU(),
        )
        self.projection = nn.Linear(point_dim, out_dim)
        self.type_embedding = nn.Embedding(num_types, out_dim)

    def forward(self, polylines: Tensor, valid: Tensor, types: Tensor) -> tuple[Tensor, Tensor]:
        """polylines [B, M, P, F], valid [B, M, P], types [B, M]."""
        points = self.point_mlp(polylines * valid[..., None].to(polylines.dtype))
        points = points.masked_fill(~valid[..., None], float("-inf"))
        chunk_valid = valid.any(dim=-1)
        pooled = points.max(dim=-2).values
        pooled = torch.where(chunk_valid[..., None], pooled, torch.zeros_like(pooled))
        embedding = self.projection(pooled) + self.type_embedding(types)
        embedding = torch.where(chunk_valid[..., None], embedding, torch.zeros_like(embedding))
        return embedding, chunk_valid


class MapAttentionPool(nn.Module):
    """Attention from the target embedding over the map chunks plus a learned null token.

    The null token is always valid, so a sample with no valid chunk attends only to it and
    still gets a finite context (design doc §14.3 check 10).
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.null_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, query: Tensor, tokens: Tensor, token_valid: Tensor) -> Tensor:
        """query [B, D], tokens [B, M, D], token_valid [B, M] -> context [B, D]."""
        batch = len(tokens)
        keys = torch.cat([self.null_token.expand(batch, -1, -1), tokens], dim=1)
        valid = torch.cat([token_valid.new_ones(batch, 1), token_valid], dim=1)
        context, _ = self.attention(
            query[:, None], keys, keys, key_padding_mask=~valid, need_weights=False
        )
        return context[:, 0]
