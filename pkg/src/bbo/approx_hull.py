from __future__ import annotations

from typing import TYPE_CHECKING
import logging

import ray
import jax.numpy as jnp
import jax
import numpy as np
import scipy as sp

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


__all__ = ["approx_hull"]


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
    simplices_list = _parallel_convex_hull(points)
    # Step 2: Pad simplices to uniform shape
    max_faces = max(len(s) for s in simplices_list)
    simplices_padded = np.zeros((len(simplices_list), max_faces, 3), dtype=np.int32)
    for i, s in enumerate(simplices_list):
        simplices_padded[i, :len(s)] = s

    # Step 3: JAX MVBB Computation
    points_jax = jnp.asarray(points)
    simplices_jax = jnp.asarray(simplices_padded)
    rotations, bboxes, volumes, aligned_points = _approx_hull_batched(points_jax, simplices_jax)
    return rotations, bboxes, volumes, aligned_points


def _parallel_convex_hull(point_clouds: np.ndarray) -> list[np.ndarray]:
    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
    futures = [_convex_hull_simplices.remote(pc) for pc in point_clouds]
    simplices_list = ray.get(futures)
    return simplices_list


@ray.remote
def _convex_hull_simplices(points: ArrayLike) -> np.ndarray:
    """Calculate the convex hull of a set of 3D points.

    References
    ----------
    - [`scipy.spatial.ConvexHull`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)
    """
    # Indices of `points` forming each
    # triangular face of the convex hull.
    return sp.spatial.ConvexHull(points).simplices  # (n_faces, n_dims)


@jax.jit
def _approx_hull(points: jnp.ndarray, simplices: jnp.ndarray):
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

    return best_rotation, best_bbox, min_volume, final_points


_approx_hull_batched = jax.vmap(_approx_hull, in_axes=(0, 0))


def _approx_hull_non_vectorized(points: np.ndarray):
    """Calculate the oriented minimum-volume bounding box (MVBB) of a set of 3D points.

    Parameters
    ----------
    points
        Points in 3D space as an array of shape `(n_points, 3)`.

    Returns
    -------
    rotation_matrix (ndarray): (3, 3) matrix aligning points to minimal bounding box axes.
    bounding_box (ndarray): (8, 3) corners of the minimal bounding box.
    volume (float): Volume of the minimal bounding box.
    """

    min_volume = np.inf
    best_rotation: np.ndarray = None
    best_bbox: tuple[np.ndarray, np.ndarray] = None
    final_points: np.ndarray = None

    hull = sp.spatial.ConvexHull(points)
    # Iterate over hull faces.
    # `hull.simplices` contains indices of the `points`
    # forming each triangular face of the convex hull.
    for simplex in hull.simplices:
        # Compute normal vector of the face
        triangle = points[simplex]
        edge1 = triangle[1] - triangle[0]
        edge2 = triangle[2] - triangle[0]
        normal = np.cross(edge1, edge2)
        if np.linalg.norm(normal) <= 1e-12:
            # Degenerate triangle, skip it
            continue
        normal /= np.linalg.norm(normal)

        # Build rotation matrix aligning z-axis to normal
        z_axis = normal
        x_axis = edge1 / np.linalg.norm(edge1)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        rotation = np.vstack([x_axis, y_axis, z_axis]).T
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1  # Flip last axis to ensure right-handed frame

        # Rotate points
        rotated_points = points @ rotation

        # Compute axis-aligned bounding box in rotated space
        min_coords = rotated_points.min(axis=0)
        max_coords = rotated_points.max(axis=0)
        volume = np.prod(max_coords - min_coords)

        if volume < min_volume:
            min_volume = volume
            best_rotation = rotation
            bbox_corners = np.array(
                [
                    [min_coords[0], min_coords[1], min_coords[2]],
                    [min_coords[0], min_coords[1], max_coords[2]],
                    [min_coords[0], max_coords[1], min_coords[2]],
                    [min_coords[0], max_coords[1], max_coords[2]],
                    [max_coords[0], min_coords[1], min_coords[2]],
                    [max_coords[0], min_coords[1], max_coords[2]],
                    [max_coords[0], max_coords[1], min_coords[2]],
                    [max_coords[0], max_coords[1], max_coords[2]],
                ]
            )
            best_bbox = bbox_corners @ rotation.T  # Rotate corners back to original orientation
            final_points = rotated_points
    return best_rotation, best_bbox, min_volume, final_points
