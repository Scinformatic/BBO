from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array


@dataclass
class BBOOutput:
    """Bounding box optimization output."""
    rotation: Array
    box: Array
    points: Array
    volume: float | Array
