"""BBO: Bounding Box Optimization Algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from bbo import hull, pca

if TYPE_CHECKING:
    from typing import Literal
    from numpy.typing import ArrayLike
    from bbo.output import BBOOutput

__all__ = ["run", "hull", "pca"]


def run(points: ArrayLike, method: Literal["hull", "pca", "best"] = "best") -> BBOOutput:
    if method == "hull":
        return hull.run(points)
    if method == "pca":
        return pca.run(points)
    if method == "best":
        points = np.asarray(points)
        if points.shape[-1] == 2:
            # Exact solution for 2D points
            return hull.run(points)
        hull_output = hull.run(points)
        pca_output = pca.run(points)
        if hull_output.volume.ndim == 0:
            return hull_output if hull_output.volume < pca_output.volume else pca_output
        points = np.empty(hull_output.points.shape)
        box = np.empty(hull_output.box.shape)
        rotation = np.empty(hull_output.rotation.shape)
        volume = np.empty(hull_output.volume.shape)
        for i in range(hull_output.points.shape[0]):
            selected = hull_output if hull_output.volume[i] < pca_output.volume[i] else pca_output
            points[i] = selected.points[i]
            box[i] = selected.box[i]
            rotation[i] = selected.rotation[i]
            volume[i] = selected.volume[i]
        return BBOOutput(
            points=jnp.asarray(points),
            box=jnp.asarray(box),
            rotation=jnp.asarray(rotation),
            volume=jnp.asarray(volume)
        )
