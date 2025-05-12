from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax
import numpy as np
import scipy as sp

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def approx_hull(points: ArrayLike):
    """Calculate the oriented minimum-volume bounding box (MVBB) of a set of 3D points.

    Parameters
    ----------
    points
        Points in 3D space as an array of shape `(n_points, 3)`.

    Returns
    -------
    rotation_matrix : ndarray
        (3, 3) matrix aligning points to minimal bounding box axes.
    bounding_box : ndarray
        (8, 3) corners of the minimal bounding box.
    volume : float
        Volume of the minimal bounding box.
    final_points : ndarray
        Rotated points in minimal bounding box alignment.
    """
    hull = sp.spatial.ConvexHull(points)
    # Indices of `points` forming each
    # triangular face of the convex hull.
    simplices = hull.simplices  # (n_faces, n_dims)
    return _approx_hull_jax(points, simplices)

@jax.jit
def _approx_hull_jax(points: jnp.ndarray, simplices: jnp.ndarray):
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
    bbox_corners = jnp.array([
        [best_min[0], best_min[1], best_min[2]],
        [best_min[0], best_min[1], best_max[2]],
        [best_min[0], best_max[1], best_min[2]],
        [best_min[0], best_max[1], best_max[2]],
        [best_max[0], best_min[1], best_min[2]],
        [best_max[0], best_min[1], best_max[2]],
        [best_max[0], best_max[1], best_min[2]],
        [best_max[0], best_max[1], best_max[2]],
    ])

    # Rotate bbox back to original space
    best_bbox = bbox_corners @ best_rotation.T

    return best_rotation, best_bbox, min_volume, final_points
