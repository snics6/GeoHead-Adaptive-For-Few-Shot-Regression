"""Linear regression head ``beta`` for GeoHead.

Implements ``docs/design.md`` sec. 7:

    Head beta:  Linear(p=32 -> 1),  no bias  (for theory alignment)
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["LinearHead"]


class LinearHead(nn.Module):
    """Linear regression head ``beta in R^p``: ``y_hat = z beta`` (no bias).

    Internally a ``nn.Linear(p, 1, bias=False)`` so that all standard
    PyTorch plumbing (optimisers, state_dict, higher's differentiable
    rewrites) works out of the box.  The ``beta`` property exposes the
    weight as a 1-D vector of shape ``(p,)`` to match the theoretical
    notation used throughout ``docs/design.md``.
    """

    def __init__(self, p: int = 32) -> None:
        super().__init__()
        self.p = int(p)
        self.linear = nn.Linear(self.p, 1, bias=False)

    @property
    def beta(self) -> Tensor:
        """Head weight as a 1-D tensor of shape ``(p,)``."""
        return self.linear.weight.squeeze(0)

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim != 2 or z.shape[-1] != self.p:
            raise ValueError(
                f"expected input of shape (B, {self.p}), got {tuple(z.shape)}"
            )
        return self.linear(z).squeeze(-1)

    def extra_repr(self) -> str:
        return f"p={self.p}, bias=False"
