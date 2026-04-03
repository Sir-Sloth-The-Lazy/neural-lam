# Standard library
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

# Third-party
import torch
from torch import nn


@dataclass(frozen=True)
class ForecastResult:
    """
    Structured forecast output.

    This keeps the prediction tensor as the primary API while allowing future
    probabilistic forecasters to attach additional training-time outputs
    without overloading the tuple contract.
    """

    prediction: torch.Tensor
    pred_std: Optional[torch.Tensor] = None
    aux_data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ensemble_prediction(self) -> bool:
        """Whether the prediction carries an explicit sample dimension."""
        return self.prediction.ndim == 5


class Forecaster(nn.Module, ABC):
    """
    Generic forecaster capable of mapping from a set of initial states,
    forcing and forces and previous states into a full forecast of the
    requested length.
    """

    @property
    @abstractmethod
    def predicts_std(self) -> bool:
        """Whether this forecaster outputs a predicted standard deviation."""

    @abstractmethod
    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
    ) -> ForecastResult | tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        boundary_states: (B, pred_steps, num_grid_nodes, d_f)
        Returns:
            prediction: either (B, pred_steps, num_grid_nodes, d_f) for
                deterministic outputs or (B, S, pred_steps, num_grid_nodes,
                d_f) for sample-based ensemble outputs.
            pred_std: same shape as prediction, or (d_f,), or None
        """
        pass
