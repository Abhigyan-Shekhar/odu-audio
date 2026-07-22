"""Lightweight BiLSTM temporal fusion model for Phase 2."""

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalBiLSTMConfig:
    """Configurable BiLSTM architecture parameters."""

    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"


class TemporalBiLSTM(nn.Module):
    """Bidirectional LSTM classifier over fused feature sequences."""

    def __init__(self, config: TemporalBiLSTMConfig) -> None:
        super().__init__()
        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim * 2, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Return one agitation logit per sequence.

        Args:
            features: Tensor shaped [batch, time, feature].
        """
        outputs, _ = self.lstm(features)
        final_state = outputs[:, -1, :]
        return cast(
            torch.Tensor, self.classifier(self.dropout(final_state)).squeeze(-1)
        )

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Return agitation probabilities for feature sequences."""
        return cast(torch.Tensor, torch.sigmoid(self.forward(features)))
