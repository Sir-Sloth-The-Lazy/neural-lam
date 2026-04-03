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
            prediction: (B, pred_steps, num_grid_nodes, d_f)
            pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)
        """
        pass
