"""BBO: Bounding Box Optimization Algorithms."""

import numpy as np
import jax.numpy as jnp
from typing import Literal

from bbo import hull, pca, exception
from bbo.output import BBOOutput
from bbo.typing import atypecheck, Num, Array


__all__ = ["run", "hull", "pca"]


@atypecheck
def run(
    points: Num[Array, "*n_batches n_samples n_features"],
    method: Literal["hull", "pca", "best"] = "best",
) -> BBOOutput:
    if method == "hull":
        return hull.run(points)
    if method == "pca":
        return pca.run(points)
    if method == "best":
        if points.shape[-1] == 2:
            # Exact solution for 2D points
            return hull.run(points)
        hull_output = hull.run(points)
        pca_output = pca.run(points)
        if hull_output.volume.ndim == 0:
            return hull_output if hull_output.volume < pca_output.volume else pca_output
        hull_is_better = hull_output.volume < pca_output.volume
        mask_shape = hull_is_better.shape
        mask_ndim = hull_is_better.ndim
        points_rotated, box, rotation, volume = [
            jnp.where(
                hull_is_better.reshape(
                    *mask_shape, *(1,) * (getattr(hull_output, attr_name).ndim - mask_ndim)
                ),
                getattr(hull_output, attr_name),
                getattr(pca_output, attr_name),
            ) for attr_name in ("points", "box", "rotation", "volume")
        ]
        return BBOOutput(
            points=points_rotated,
            box=box,
            rotation=rotation,
            volume=volume,
        )
    raise exception.InputError(
        name="method",
        value=method,
        problem=f"Method '{method}' is not recognized. Use 'hull', 'pca', or 'best'."
    )
