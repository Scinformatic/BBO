"""Approximate convex hull for n-dimensional points."""

from __future__ import annotations

from typing import TYPE_CHECKING
import itertools

import jax.numpy as jnp
import jax


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

    n_dims = points.shape[1]

    # Extract triangle vertices for all faces
    triangles = points[simplices]  # (n_faces, n_dims, n_dims)

    # Compute edge vectors relative to first vertex of each simplex
    edges = triangles[:, 1:, :] - triangles[:, 0:1, :]  # (n_faces, n_dims-1, n_dims)

    # Complete to full nD basis by adding the normal vector
    # Compute normal vector via null space of edge matrix
    normals = _compute_normal_batched(edges)  # (n_faces, n_dims)

    # Compute simplex pseudo-volumes using norm of the normal vector
    norm_lengths = jnp.linalg.norm(normals, axis=1)  # (n_faces,)

    # Mask for valid (non-degenerate) simplices
    valid_mask = norm_lengths > 1e-12

    # Mask out degenerate simplices
    # Instead of using the mask directly (e.g., `normals[valid_mask]`),
    # we use `jnp.where` to ensure the shape remains consistent
    # so that the function can be JIT-compiled.
    normals = jnp.where(valid_mask[:, None], normals, jnp.nan)
    edges = jnp.where(valid_mask[:, None, None], edges, jnp.nan)

    # Orthonormalize edge vectors via Gram-Schmidt to form basis
    axes = _gram_schmidt_batched(edges)  # (n_faces, n_dims-1, n_dims)

    # Assemble rotation matrices: (n_faces, n_dims, n_dims)
    rotations = jnp.concatenate([axes, normals[:, None, :]], axis=1)

    # Ensure right-handed coordinate systems (i.e., no reflection)
    # by flipping the last axis if the determinant is negative.
    # Again, here we can't use the mask directly
    # (i.e., `rotations.at[flip_mask, :, -1].multiply(-1.0)`),
    # as we will get a `NonConcreteBooleanIndexError`.
    # See: https://docs.jax.dev/en/latest/errors.html#jax.errors.NonConcreteBooleanIndexError
    dets = jnp.linalg.det(rotations)  # (n_faces,)
    flip_mask = dets < 0
    scale = jnp.where(flip_mask[:, None], -1.0, 1.0)
    normals_flipped = normals * scale
    rotations = rotations.at[:, -1, :].set(normals_flipped)


@jax.jit
def gram_schmidt(vectors: jnp.ndarray) -> jnp.ndarray:
    """Perform Gram-Schmidt orthonormalization on (n_vectors, n_dims)."""
    def body_fn(i, orthonormal_vectors):
        vi = vectors[i]
        mask = jnp.arange(orthonormal_vectors.shape[0]) < i  # (n_vectors,)
        mask = mask[:, None]  # (n_vectors, 1)
        vectors_masked = orthonormal_vectors * mask  # (n_vectors, n_dims)
        projections = jnp.sum(vectors_masked * vi, axis=1, keepdims=True) * vectors_masked  # (n_vectors, n_dims)
        vi_orth = vi - jnp.sum(projections, axis=0)  # (n_dims,)
        vi_orth /= jnp.linalg.norm(vi_orth)
        return orthonormal_vectors.at[i].set(vi_orth)

    init_basis = vectors.at[0].set(vectors[0] / jnp.linalg.norm(vectors[0]))
    orthonormal_vectors = jax.lax.fori_loop(1, vectors.shape[0], body_fn, init_basis)
    return orthonormal_vectors  # (n_vectors, n_dims)


@jax.jit
def compute_normal(edges_face: jnp.ndarray) -> jnp.ndarray:
    """Compute normal vector as last column of QR decomposition (null space)."""
    q, _ = jnp.linalg.qr(edges_face.T)
    return q[:, -1]  # Last column corresponds to normal vector


_approx_hull_batched = jax.vmap(_approx_hull, in_axes=(0, 0))
_gram_schmidt_batched = jax.vmap(gram_schmidt, in_axes=(0,))
_compute_normal_batched = jax.vmap(compute_normal, in_axes=(0,))
