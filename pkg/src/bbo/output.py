from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax.numpy import ndarray


@dataclass
class BBOOutput:
    """Bounding box optimization output.

    Attributes
    ----------
    points
        Rotated points in minimal bounding box alignment.
    box
        Corners of the minimal bounding box in the original coordinate system.
    rotation
        Rotation matrix aligning points to minimal bounding box axes.
    volume
        Volume of the minimal bounding box.
    """
    points: ndarray
    box: ndarray
    rotation: ndarray
    volume: ndarray
