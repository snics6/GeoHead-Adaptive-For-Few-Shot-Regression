"""MLP encoder ``phi_theta`` for GeoHead.

Implements ``docs/design.md`` sec. 7:

    Encoder phi_theta:  MLP(d_x=16 -> 64 -> 64 -> p=32),  ReLU + LayerNorm
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

__all__ = ["MLPEncoder"]


class MLPEncoder(nn.Module):
    """Trainable MLP encoder ``phi_theta : R^{d_x} -> R^{p}``.

    Architecture
    ------------
    ``d_x -> hidden[0] -> ... -> hidden[-1] -> p``.  Each hidden block
    applies ``Linear -> ReLU -> LayerNorm`` (post-activation norm).
    The output layer is a bare ``Linear`` with no activation and no
    normalisation so that downstream linear operations (head, Gram
    matrices, DARE-GRAM regularizer) see unmodified features.

    Parameters
    ----------
    d_x: input dimension.
    hidden: sizes of the hidden layers (at least one required).
    p: output feature dimension.
    """

    def __init__(
        self,
        d_x: int = 16,
        hidden: Sequence[int] = (64, 64),
        p: int = 32,
    ) -> None:
        super().__init__()
        if len(hidden) < 1:
            raise ValueError("MLPEncoder requires at least one hidden layer")

        self.d_x = int(d_x)
        self.p = int(p)
        self.hidden: tuple[int, ...] = tuple(int(h) for h in hidden)

        dims = [self.d_x, *self.hidden, self.p]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(dims[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[-1] != self.d_x:
            raise ValueError(
                f"expected input of shape (B, {self.d_x}), got {tuple(x.shape)}"
            )
        return self.net(x)

    def extra_repr(self) -> str:
        return f"d_x={self.d_x}, hidden={self.hidden}, p={self.p}"
