"""Optimize the oriented minimum-volume bounding box (MVBB) using convex hull method."""

import logging

import ray
import jax.numpy as jnp
import jax
import numpy as np
import scipy as sp

from bbo import exception, util
from bbo.output import BBOOutput
from bbo.typing import atypecheck, Array, JAXArray, Num, Integer


__all__ = [
    "run",
    "run_single",
    "run_batch",
    "hull_simplices_single",
    "hull_simplices_batch"
]


@atypecheck
def run(points: Num[Array, "*n_batches n_samples n_features"]) -> BBOOutput:
    """Calculate the oriented minimum-volume bounding box (MVBB) of a set of points (or batch thereof).

    Parameters
    ----------
    points
        Points as an array of shape `(n_points, n_dimensions)`
        or `(n_batches, n_points, n_dimensions)`.
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
        simplices = hull_simplices_single(points)
        return BBOOutput(*run_single(points, simplices))
    elif points.ndim == 3:
        simplices = hull_simplices_batch(points)
        return BBOOutput(*run_batch(points, simplices))
    points_reshaped = points.reshape(-1, *points.shape[-2:])
    simplices = hull_simplices_batch(points_reshaped)
    points_rotated, bbox_vertices, rotation_final, volume_final = run_batch(points_reshaped, simplices)
    batch_shape = points.shape[:-2]
    return BBOOutput(
        points=points_rotated.reshape(points.shape),
        box=bbox_vertices.reshape(*batch_shape, *bbox_vertices.shape[-2:]),
        rotation=rotation_final.reshape(*batch_shape, *rotation_final.shape[-2:]),
        volume=volume_final.reshape(*batch_shape),
    )


@atypecheck
def hull_simplices_batch(
    points: Num[Array, "n_batches n_samples n_features"],
    array_out: bool = True,
) -> list[np.ndarray] | Integer[Array, "n_batches nmax_simplices n_features"]:
    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
    futures = [_hull_simplices_single.remote(batch) for batch in points]
    simplices_list = ray.get(futures)
    if not array_out:
        return simplices_list
    # Infer simplex dimensionality
    simplex_ndim = simplices_list[0].shape[1]
    # Pad simplices to uniform shape
    max_faces = max(len(s) for s in simplices_list)
    simplices_padded = np.zeros((len(simplices_list), max_faces, simplex_ndim), dtype=np.int32)
    for i, s in enumerate(simplices_list):
        simplices_padded[i, :len(s)] = s
    return jnp.asarray(simplices_padded)


@atypecheck
def hull_simplices_single(points: Num[Array, "n_samples n_features"]) -> Integer[Array, "n_simplices n_features"]:
    """Calculate the convex hull of a set of 3D points.

    References
    ----------
    - [`scipy.spatial.ConvexHull`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)
    """
    # Indices of `points` forming each
    # triangular face of the convex hull.
    return sp.spatial.ConvexHull(points).simplices  # (n_faces, n_dims)


@ray.remote
def _hull_simplices_single(points: Num[Array, "n_samples n_features"]) -> Integer[Array, "n_simplices n_features"]:
    return hull_simplices_single(points)


@jax.jit
@atypecheck
def run_single(
    points: Num[Array, "n_samples n_features"],
    simplices: Integer[Array, "n_simplices n_features"],
) -> tuple[
    Num[JAXArray, "n_samples n_features"],
    Num[JAXArray, "2**n_features n_features"],
    Num[JAXArray, "n_features n_features"],
    Num[JAXArray, ""],
]:
    """Calculate the oriented minimum-volume bounding box (MVBB) of a set of points.

    Parameters
    ----------
    points
        Point coordinates as an array of shape `(n_points, n_dimensions)`.
    simplices
        Indices of points forming each triangular face (from ConvexHull)
        as an integer array of shape `(n_faces_or_edges, n_dimensions)`.
    """
    points = jnp.asarray(points)
    simplices = jnp.asarray(simplices)
    points_ndim = points.shape[-1]

    if points_ndim == 2:
        # Extract edge vertices for all edges
        lines = points[simplices]  # (n_edges, 2, 2)

        # Compute edge vectors
        edges = lines[:, 1] - lines[:, 0]  # (n_edges, 2)

        # Compute norms
        edge_norms = jnp.linalg.norm(edges, axis=1)  # (n_edges,)

        # Mask for valid (non-degenerate) edges (to avoid division by zero)
        valid_mask = edge_norms > 1e-12

        # Filter valid edges
        safe_norms = jnp.where(valid_mask, edge_norms, jnp.nan)

        # Compute orthonormal axes
        x_axes = edges / safe_norms[:, None]  # (n_edges, 2)
        # Perpendicular vectors (rotated 90 degrees CCW)
        y_axes = jnp.stack([-x_axes[:, 1], x_axes[:, 0]], axis=1)  # (n_edges, 2)

        # Stack into rotation matrices (x_axis and y_axis as columns)
        rotations = jnp.stack([x_axes, y_axes], axis=-1)  # (n_edges, 2, 2)

    elif points_ndim == 3:
        # Extract triangle vertices for all faces
        triangles = points[simplices]  # (n_faces, n_dims, n_dims)

        # Compute edge vectors
        edge1 = triangles[:, 1] - triangles[:, 0]  # (n_faces, n_dims)
        edge2 = triangles[:, 2] - triangles[:, 0]  # (n_faces, n_dims)

        # Compute normals
        normals = jnp.cross(edge1, edge2)  # (n_faces, n_dims)
        norm_lengths = jnp.linalg.norm(normals, axis=1)  # (n_faces,)

        # Mask for valid (non-degenerate) triangles (to avoid division by zero)
        valid_mask = norm_lengths > 1e-12

        # Filter valid triangles
        # Instead of using the mask directly (e.g., `normals[valid_mask]`),
        # we use `jnp.where` to ensure the shape remains consistent
        # so that the function can be JIT-compiled.
        normals = jnp.where(valid_mask[:, None], normals, jnp.nan)
        edge1 = jnp.where(valid_mask[:, None], edge1, jnp.nan)

        # Compute orthonormal axes
        z_axes = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)  # (n_faces, n_dims)
        x_axes = edge1 / jnp.linalg.norm(edge1, axis=1, keepdims=True)  # (n_faces, n_dims)
        y_axes = jnp.cross(z_axes, x_axes)
        y_axes = y_axes / jnp.linalg.norm(y_axes, axis=1, keepdims=True)  # (n_faces, n_dims)
        # Gram-Schmidt refinement for x-axes
        x_axes = jnp.cross(y_axes, z_axes)

        # Stack into rotation matrices (x, y, z as columns)
        rotations = jnp.stack([x_axes, y_axes, z_axes], axis=-1)  # (n_faces, n_dims, n_dims)

        # Append identity matrix as fallback candidate,
        # so that if all other rotations increase the volume,
        # we return the identity rotation and the original points.
        identity_rotation = jnp.eye(points_ndim, dtype=points.dtype)
        rotations = jnp.concatenate([rotations, identity_rotation[None]], axis=0)  # (n_faces + 1, n_dims, n_dims)

        # Ensure right-handed coordinate systems (i.e., no reflection)
        # by flipping the last axis if the determinant is negative.
        # Again, here we can't use the mask directly
        # (i.e., `rotations.at[flip_mask, :, -1].multiply(-1.0)`),
        # as we will get a `NonConcreteBooleanIndexError`.
        # See: https://docs.jax.dev/en/latest/errors.html#jax.errors.NonConcreteBooleanIndexError
        dets = jnp.linalg.det(rotations)  # (n_faces,)
        flip_mask = dets < 0
        scale = jnp.where(flip_mask, -1.0, 1.0).reshape(-1, 1)
        z_axes_flipped = rotations[:, :, -1] * scale
        rotations = rotations.at[:, :, -1].set(z_axes_flipped)

    # Rotate points for all rotations
    rotated_points = jnp.einsum('nj,fji->fni', points, rotations)  # (n_faces, n_points, n_dims)

    # Compute AABB bounds in rotated space
    lower_bounds = jnp.min(rotated_points, axis=1)  # (n_faces, n_dims)
    upper_bounds = jnp.max(rotated_points, axis=1)  # (n_faces, n_dims)

    # Compute volumes of AABBs
    volumes = jnp.prod(upper_bounds - lower_bounds, axis=1)  # (n_faces,)

    # Find minimal volume index (ignoring NaNs)
    idx_min_vol = jnp.nanargmin(volumes)

    # Extract best rotation and aligned points
    best_rotation = rotations[idx_min_vol]
    best_volume = volumes[idx_min_vol]
    best_lower_bounds = lower_bounds[idx_min_vol]
    best_upper_bounds = upper_bounds[idx_min_vol]
    best_points = rotated_points[idx_min_vol]

    # Bounding box vertices (in rotated space)
    bbox_vertices_rotated = util.box_vertices_from_bounds(best_lower_bounds, best_upper_bounds)

    # Rotate bbox back to original space
    best_bbox = bbox_vertices_rotated @ best_rotation.T

    return best_points, best_bbox, best_rotation, best_volume


@jax.jit
@atypecheck
def run_batch(
    points: Num[Array, "n_batches n_samples n_features"],
    simplices: Integer[Array, "n_batches n_simplices n_features"],
) -> tuple[
    Num[JAXArray, "n_batches n_samples n_features"],
    Num[JAXArray, "n_batches 2**n_features n_features"],
    Num[JAXArray, "n_batches n_features n_features"],
    Num[JAXArray, "n_batches"],
]:
    points = jnp.asarray(points)
    simplices = jnp.asarray(simplices)
    return _run_batch(points, simplices)


_run_batch = jax.vmap(run_single, in_axes=(0, 0))
