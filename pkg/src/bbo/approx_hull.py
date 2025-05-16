from __future__ import annotations

from typing import TYPE_CHECKING
import logging

import ray
import jax.numpy as jnp
import jax
import numpy as np
import scipy as sp

from bbo import exception
from bbo.output import BBOOutput

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


__all__ = [
    "approx_hull",
    "approx_hull_single",
    "approx_hull_batch",
    "hull_simplices_single",
    "hull_simplices_batch"
]


def approx_hull(points: ArrayLike) -> BBOOutput:
    """Calculate the oriented minimum-volume bounding box (MVBB) of a set of points (or batch thereof).

    Parameters
    ----------
    points
        Points in 3D space as an array of shape `(n_points, 3)`
        or `(n_batches, n_points, 3)`.
    """
    points = np.asarray(points)
    if points.ndim == 2:
        simplices = jnp.asarray(hull_simplices_single(points))
        points = jnp.asarray(points)
        out = approx_hull_single(points, simplices)
    elif points.ndim == 3:
        simplices = hull_simplices_batch(points)
        points = jnp.asarray(points)
        out = approx_hull_batch(points, simplices)
    else:
        raise exception.InputError(
            name="points",
            value=points,
            problem=f"Expected 2D or 3D input, but got shape {points.shape}"
        )
    return BBOOutput(*out)


def hull_simplices_batch(points: np.ndarray, array_out: bool = True) -> list[np.ndarray] | jnp.ndarray:
    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
    futures = [_hull_simplices_single.remote(pc) for pc in points]
    simplices_list = ray.get(futures)
    if not array_out:
        return simplices_list
    # Pad simplices to uniform shape
    max_faces = max(len(s) for s in simplices_list)
    simplices_padded = np.zeros((len(simplices_list), max_faces, 3), dtype=np.int32)
    for i, s in enumerate(simplices_list):
        simplices_padded[i, :len(s)] = s
    return jnp.asarray(simplices_padded)


def hull_simplices_single(points: ArrayLike) -> np.ndarray:
    """Calculate the convex hull of a set of 3D points.

    References
    ----------
    - [`scipy.spatial.ConvexHull`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)
    """
    # Indices of `points` forming each
    # triangular face of the convex hull.
    return sp.spatial.ConvexHull(points).simplices  # (n_faces, n_dims)


@ray.remote
def _hull_simplices_single(points: ArrayLike) -> np.ndarray:
    return hull_simplices_single(points)


@jax.jit
def approx_hull_single(points: jnp.ndarray, simplices: jnp.ndarray):
    """Calculate the oriented minimum-volume bounding box (MVBB) of a set of 3D points.

    Parameters
    ----------
    points
        Point coordinates as an array of shape `(n_points, n_dims)`.
    simplices
        Indices of points forming each triangular face (from ConvexHull)
        as an integer array of shape `(n_faces, n_dims)`.

    Returns
    -------
    rotation_matrix : (3, 3) float
        Matrix aligning points to minimal bounding box axes.
    bounding_box : (8, 3) float
        Corners of the minimal bounding box.
    volume : float
        Volume of the minimal bounding box.
    final_points : (n_points, 3) float
        Rotated points in minimal bounding box alignment.
    """

    ndim = points.shape[1]

    # Extract triangle vertices for all faces
    triangles = points[simplices]  # (n_faces, n_dims, n_dims)

    # Compute edge vectors
    edge1 = triangles[:, 1] - triangles[:, 0]  # (n_faces, n_dims)
    edge2 = triangles[:, 2] - triangles[:, 0]  # (n_faces, n_dims)

    # Compute normals
    normals = jnp.cross(edge1, edge2)  # (n_faces, n_dims)
    norm_lengths = jnp.linalg.norm(normals, axis=1)  # (n_faces,)

    # Mask for valid (non-degenerate) triangles
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
    min_coords = jnp.min(rotated_points, axis=1)  # (n_faces, n_dims)
    max_coords = jnp.max(rotated_points, axis=1)  # (n_faces, n_dims)

    # Compute volumes
    volumes = jnp.prod(max_coords - min_coords, axis=1)  # (n_faces,)

    # Find minimal volume index (ignoring NaNs)
    min_idx = jnp.nanargmin(volumes)
    min_volume = jnp.nanmin(volumes)

    # Extract best rotation and aligned points
    best_rotation = rotations[min_idx]
    best_min = min_coords[min_idx]
    best_max = max_coords[min_idx]
    final_points = rotated_points[min_idx]

    # Bounding box corners (in rotated space)
    # For each axis, create a list: [min, max]
    choices = jnp.stack([best_min, best_max], axis=0)  # (2, n_dims)
    # Generate all index combinations (0 or 1 per axis) — Cartesian product
    grid = jnp.indices((2,) * ndim).reshape(ndim, -1).T  # (2^n_dims, n_dims)
    # Select min/max per axis using the grid indices
    bbox_corners = choices[grid, jnp.arange(ndim)]  # (2^n_dims, n_dims)

    # Rotate bbox back to original space
    best_bbox = bbox_corners @ best_rotation.T

    return final_points, best_bbox, best_rotation, min_volume


approx_hull_batch = jax.vmap(approx_hull_single, in_axes=(0, 0))
