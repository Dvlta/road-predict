"""Multi-agent interaction model (design doc §19): the target, its neighbors and the map chunks
as tokens of one Transformer encoder, whose target token is decoded into the future."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from prs.models.base import ForecastModel
from prs.models.lstm_forecaster import TrajectoryEncoder
from prs.models.map_encoder import PolylineEncoder

TARGET_SLOT = 1  # token order: [null] [target] [neighbor_1..N] [map_1..M]


def relative_position(xy: Tensor) -> Tensor:
    """Positions [..., 2] in the target frame -> [distance, sin(bearing), cos(bearing)]."""
    bearing = torch.atan2(xy[..., 1], xy[..., 0])
    return torch.stack([xy.norm(dim=-1), bearing.sin(), bearing.cos()], dim=-1)


class InteractionForecaster(ForecastModel):
    """Agent tokens: the §17 trajectory encoder, shared by the target and every neighbor, plus
    summed tags (agent type, is-SDC, is-target, position at t0). Map tokens: the §18 polyline
    encoder plus the position of the chunk's centroid. A learned null token is prepended and
    always valid. Pre-norm Transformer layers mix all tokens. The decoder reads the target's
    trajectory encoding concatenated with its Transformer output (the §18 fusion): decoding the
    Transformer output alone trained markedly worse than the plain LSTM on the same inputs.

    `use_map` and `use_neighbors` switch the two context token sets on or off, giving the four
    variants of the §19 ablation with one architecture. Masking: padded neighbor slots (no
    valid frame) never reach the encoder, padded tokens are zeroed before the first layer and
    after every layer, and they are excluded as attention keys, so padded values cannot change
    the output (design doc §14.2).
    """

    num_modes = 1

    def __init__(
        self,
        agent_features: int,
        map_features: int,
        num_agent_types: int,
        num_map_types: int,
        future_steps: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        decoder_hidden_dim: int,
        use_map: bool,
        use_neighbors: bool,
        transformer_layers: int,
        transformer_heads: int,
        transformer_ff_dim: int,
        map_point_dim: int | None = None,
    ) -> None:
        super().__init__()
        if use_map and map_point_dim is None:
            raise ValueError("use_map needs map_point_dim")
        self.future_steps = future_steps
        self.use_map, self.use_neighbors = use_map, use_neighbors
        self.input_fields = (
            ("target_history", "target_history_valid", "target_type")
            + (("neighbors", "neighbor_valid", "neighbor_types", "neighbor_is_sdc")
               if use_neighbors else ())
            + (("map_polylines", "map_valid", "map_types") if use_map else ())
        )  # fmt: skip
        self.agent_encoder = TrajectoryEncoder(agent_features, hidden_dim, num_layers, dropout)
        self.type_embedding = nn.Embedding(num_agent_types, hidden_dim)
        self.sdc_embedding = nn.Embedding(2, hidden_dim)
        self.target_embedding = nn.Embedding(2, hidden_dim)
        # Tags start small, like the null token. At PyTorch's default N(0, 1) the summed tags
        # have norm ≈ 28 against ≈ 0.7 for the trajectory encoding, so after the first
        # LayerNorm the target token barely depends on its history, and learning is slow.
        for embedding in (self.type_embedding, self.sdc_embedding, self.target_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
        self.position_mlp = nn.Sequential(  # shared by agents (t0) and map chunks (centroid)
            nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.map_encoder = (
            PolylineEncoder(map_features, map_point_dim, hidden_dim, num_map_types)
            if use_map
            else None
        )
        self.null_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                hidden_dim,
                transformer_heads,
                transformer_ff_dim,
                dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(transformer_layers)
        )
        self.norm = nn.LayerNorm(hidden_dim)  # pre-norm layers leave the output unnormalized
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, future_steps * 2),
        )

    def forward(
        self,
        target_history: Tensor,
        target_history_valid: Tensor,
        target_type: Tensor,
        neighbors: Tensor | None = None,
        neighbor_valid: Tensor | None = None,
        neighbor_types: Tensor | None = None,
        neighbor_is_sdc: Tensor | None = None,
        map_polylines: Tensor | None = None,
        map_valid: Tensor | None = None,
        map_types: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch = len(target_history)
        histories, valid = target_history[:, None], target_history_valid[:, None]
        types, is_sdc = target_type[:, None], torch.zeros_like(target_history_valid[:, :1])
        if self.use_neighbors:
            if neighbors is None or neighbor_valid is None or neighbor_types is None:
                raise ValueError("use_neighbors needs neighbors, neighbor_valid, neighbor_types")
            if neighbor_is_sdc is None:
                raise ValueError("use_neighbors needs neighbor_is_sdc")
            histories = torch.cat([histories, neighbors], dim=1)
            valid = torch.cat([valid, neighbor_valid], dim=1)
            types = torch.cat([types, neighbor_types], dim=1)
            is_sdc = torch.cat([is_sdc, neighbor_is_sdc], dim=1)
        # The target is always a token; a neighbor slot is one if any frame is valid.
        agent_valid = valid.any(dim=-1)
        agent_valid[:, 0] = True
        is_target = torch.zeros_like(types)
        is_target[:, 0] = 1

        encoded = self._encode_agents(histories, valid, agent_valid)
        at_t0 = histories[:, :, -1, :2] * valid[:, :, -1, None].to(histories.dtype)
        agents = (
            encoded
            + self.type_embedding(types)
            + self.sdc_embedding(is_sdc.long())
            + self.target_embedding(is_target)
            + self.position_mlp(relative_position(at_t0))
        )
        tokens = [self.null_token.expand(batch, -1, -1), _zero_padded(agents, agent_valid)]
        token_valid = [agent_valid.new_ones(batch, 1), agent_valid]

        if self.use_map:
            if map_polylines is None or map_valid is None or map_types is None:
                raise ValueError("use_map needs map_polylines, map_valid and map_types")
            chunks, chunk_valid = self.map_encoder(map_polylines, map_valid, map_types)
            points = map_polylines[..., :2] * map_valid[..., None].to(map_polylines.dtype)
            centroid = points.sum(dim=-2) / map_valid.sum(dim=-1, keepdim=True).clamp(min=1)
            chunks = chunks + self.position_mlp(relative_position(centroid))
            tokens.append(_zero_padded(chunks, chunk_valid))
            token_valid.append(chunk_valid)

        x, mask = torch.cat(tokens, dim=1), torch.cat(token_valid, dim=1)
        for layer in self.layers:
            x = _zero_padded(layer(x, src_key_padding_mask=~mask), mask)
        decoded = self.decoder(torch.cat([encoded[:, 0], self.norm(x[:, TARGET_SLOT])], dim=-1))
        trajectories = decoded.view(-1, 1, self.future_steps, 2)
        return {
            "trajectories": trajectories,
            "logits": torch.zeros(len(trajectories), 1, device=trajectories.device),
        }

    def _encode_agents(self, histories: Tensor, valid: Tensor, agent_valid: Tensor) -> Tensor:
        """Run the trajectory encoder on the real agents only ([B, A, H, F] -> [B, A, D]);
        padded neighbor slots (about a third of them) get zeros and cost nothing."""
        flat = agent_valid.flatten()
        encoded = self.agent_encoder(histories.flatten(0, 1)[flat], valid.flatten(0, 1)[flat])
        out = encoded.new_zeros(flat.numel(), encoded.shape[-1])
        out[flat] = encoded
        return out.view(*agent_valid.shape, -1)


def _zero_padded(tokens: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
