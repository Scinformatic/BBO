from __future__ import annotations

from typing import TYPE_CHECKING

from arrayer.pca import pca_single as _pca_single
import jax
import jax.numpy as jnp

from bbo import exception, util
from bbo.output import BBOOutput

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def run(points: ArrayLike) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate the oriented minimum-volume bounding box (MVBB) of a set of points (or batch thereof).

    Parameters
    ----------
    points
        Points as an array of shape `(n_points, n_dimensions)`
        or `(n_batches, n_points, n_dimensions)`.
    """
    points = jnp.asarray(points)
    if points.shape[-2] < 2:
        raise exception.InputError(
            name="points",
            value=points,
            problem=f"At least 2 points are required, but got {points.shape[0]}."
        )
    if points.shape[-1] < 2:
        raise exception.InputError(
            name="points",
            value=points,
            problem=f"At least 2 features are required, but got {points.shape[1]}."
        )
    if points.ndim == 2:
        func = run_single
    elif points.ndim == 3:
        func = run_batch
    else:
        raise exception.InputError(
            name="points",
            value=points,
            problem=f"Points must be a 2D or 3D array, but is {points.ndim}D."
        )
    return BBOOutput(*func(points))


@jax.jit
def run_single(points: jnp.ndarray):
    """Calculate the oriented minimum-volume bounding box (MVBB) of a set of points.

    Parameters
    ----------
    points
        Points as an array of shape `(n_points, n_dimensions)`.
    """
    # Calculate the PCA version
    points_transformed, components, _, translation = _pca_single(points)
    rotation_transpose = components
    rotation = rotation_transpose.T
    lower_bounds_pca = jnp.min(points_transformed, axis=0)
    upper_bounds_pca = jnp.max(points_transformed, axis=0)
    volume_pca = jnp.prod(upper_bounds_pca - lower_bounds_pca)
    bbox_vertices_pca = util.box_vertices_from_bounds(lower_bounds_pca, upper_bounds_pca)
    bbox_vertices_pca = (bbox_vertices_pca @ rotation_transpose) - translation
    points_rotated_pca = points @ rotation

    # Calculate the original version
    lower_bounds_orig = jnp.min(points, axis=0)
    upper_bounds_orig = jnp.max(points, axis=0)
    volume_orig = jnp.prod(upper_bounds_orig - lower_bounds_orig)
    bbox_vertices_orig = util.box_vertices_from_bounds(lower_bounds_orig, upper_bounds_orig)
    rotation_orig = jnp.eye(points.shape[1])

    # Select between PCA and original results
    pred = volume_pca < volume_orig
    points_rotated = jax.lax.select(pred, points_rotated_pca, points)
    bbox_vertices = jax.lax.select(pred, bbox_vertices_pca, bbox_vertices_orig)
    rotation_final = jax.lax.select(pred, rotation, rotation_orig)
    volume_final = jax.lax.select(pred, volume_pca, volume_orig)
    return points_rotated, bbox_vertices, rotation_final, volume_final


run_batch = jax.jit(jax.vmap(run_single))
