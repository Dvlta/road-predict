"""The map encoder's masking (design doc §18)."""

import torch

from prs.data.map_types import NUM_MAP_TYPES, MapType
from prs.models.map_encoder import MapAttentionPool, PolylineEncoder


def _encoder() -> PolylineEncoder:
    torch.manual_seed(0)
    return PolylineEncoder(input_dim=4, point_dim=16, out_dim=8, num_types=NUM_MAP_TYPES).eval()


def test_chunk_embedding_ignores_padded_point_slots():
    """A chunk encodes the same with or without trailing padded slots: padded points take no
    part in the max pool. Short chunks (1-3 points) make a leaked padded point visible."""
    encoder = _encoder()
    types = torch.full((1, 1), int(MapType.CROSSWALK))
    for num_points in (1, 2, 3):
        points = torch.randn(1, 1, num_points, 4)
        padded = torch.cat([points, torch.zeros(1, 1, 20 - num_points, 4)], dim=2)
        valid = torch.arange(20).expand(1, 1, 20) < num_points
        with torch.no_grad():
            alone, _ = encoder(points, torch.ones(1, 1, num_points, dtype=torch.bool), types)
            within, _ = encoder(padded, valid, types)
        # Shapes differ, so matmul rounding may differ (~1e-7); a leaked point shifts ~0.1.
        torch.testing.assert_close(alone, within, rtol=0, atol=1e-6)


def test_padded_chunks_are_zero_and_masked():
    encoder = _encoder()
    valid = torch.zeros(2, 3, 4, dtype=torch.bool)
    valid[0, 0, :2] = True
    types = torch.tensor([[3, 0, 0], [0, 0, 0]])
    with torch.no_grad():
        embedding, chunk_valid = encoder(torch.randn(2, 3, 4, 4), valid, types)
    assert chunk_valid.tolist() == [[True, False, False], [False, False, False]]
    assert embedding[~chunk_valid].abs().sum() == 0


def test_attention_pool_ignores_masked_tokens_and_survives_none():
    torch.manual_seed(0)
    pool = MapAttentionPool(dim=8, num_heads=2).eval()
    query, tokens = torch.randn(2, 8), torch.randn(2, 4, 8)
    valid = torch.tensor([[True, True, False, False], [False, False, False, False]])
    noisy = tokens.clone()
    noisy[~valid] = torch.randn(int((~valid).sum()), 8) * 100
    with torch.no_grad():
        context, noisy_context = pool(query, tokens, valid), pool(query, noisy, valid)
    torch.testing.assert_close(context, noisy_context, rtol=0, atol=0)
    assert torch.isfinite(context).all()  # row 1 has only the null token
