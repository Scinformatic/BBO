"""Principal Component Analysis ([PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)) based OBB optimization."""

from arrayer.pca import pca_single
import jax
import jax.numpy as jnp

from bbo import exception, util
from bbo.output import BBOOutput
from bbo.typing import atypecheck, Array, JAXArray, Num


@atypecheck
def run(points: Num[Array, "*n_batches n_samples n_features"]) -> BBOOutput:
    """Calculate the oriented minimum-volume bounding box (MVBB) of a set of points (or batch thereof).

    Parameters
    ----------
    points
        Points as an array of shape `(n_points, n_dimensions)`
        or `(*n_batches, n_points, n_dimensions)`.
    """
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
        return BBOOutput(*run_single(points))
    elif points.ndim == 3:
        return BBOOutput(*run_batch(points))
    points_reshaped = points.reshape(-1, *points.shape[-2:])
    points_rotated, bbox_vertices, rotation_final, volume_final = run_batch(points_reshaped)
    batch_shape = points.shape[:-2]
    return BBOOutput(
        points=points_rotated.reshape(*points.shape),
        box=bbox_vertices.reshape(*batch_shape, *bbox_vertices.shape[-2:]),
        rotation=rotation_final.reshape(*batch_shape, *rotation_final.shape[-2:]),
        volume=volume_final.reshape(*batch_shape),
    )


@jax.jit
@atypecheck
def run_single(points: Num[Array, "n_samples n_features"]) -> tuple[
    Num[JAXArray, "n_samples n_features"],
    Num[JAXArray, "2**n_features n_features"],
    Num[JAXArray, "n_features n_features"],
    Num[JAXArray, ""],
]:
    """Calculate the oriented minimum-volume bounding box (MVBB) of a set of points.

    Parameters
    ----------
    points
        Points as an array of shape `(n_points, n_dimensions)`.
    """
    points = jnp.asarray(points)

    # Calculate the PCA version
    points_transformed, components, _, translation = pca_single(points)
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
    rotation_orig = jnp.eye(points.shape[1], dtype=points.dtype)

    # Select between PCA and original results
    pred = volume_pca < volume_orig
    points_rotated = jax.lax.select(pred, points_rotated_pca, points)
    bbox_vertices = jax.lax.select(pred, bbox_vertices_pca, bbox_vertices_orig)
    rotation_final = jax.lax.select(pred, rotation, rotation_orig)
    volume_final = jax.lax.select(pred, volume_pca, volume_orig)
    return points_rotated, bbox_vertices, rotation_final, volume_final


@jax.jit
@atypecheck
def run_batch(points: Num[Array, "n_batches n_samples n_features"]) -> tuple[
    Num[JAXArray, "n_batches n_samples n_features"],
    Num[JAXArray, "n_batches 2**n_features n_features"],
    Num[JAXArray, "n_batches n_features n_features"],
    Num[JAXArray, "n_batches"],
]:
    points = jnp.asarray(points)
    return _run_batch(points)


_run_batch = jax.vmap(run_single)
