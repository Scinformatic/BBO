import numpy as np
import scipy as sp


def approx_hull(points: np.ndarray):
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

    # Extract triangle vertices for all faces
    triangles = points[simplices]  # (n_faces, n_dims, n_dims)

    # Compute edge vectors
    edge1 = triangles[:, 1] - triangles[:, 0]  # (n_faces, n_dims)
    edge2 = triangles[:, 2] - triangles[:, 0]  # (n_faces, n_dims)

    # Compute normals
    normals = np.cross(edge1, edge2)  # (n_faces, n_dims)
    norm_lengths = np.linalg.norm(normals, axis=1)  # (n_faces,)

    # Mask degenerate triangles
    valid_mask = norm_lengths > 1e-12  # Avoid divide by zero
    if not np.any(valid_mask):
        raise ValueError("All triangles are degenerate.")

    normals = normals[valid_mask]
    edge1 = edge1[valid_mask]

    # Normalize normals to get z-axes
    z_axes = normals / np.linalg.norm(normals, axis=1, keepdims=True)  # (n_valid_faces, n_dims)

    # Normalize edge1 to get x-axes
    x_axes = edge1 / np.linalg.norm(edge1, axis=1, keepdims=True)  # (n_valid_faces, n_dims)

    # Compute y-axes via cross product
    y_axes = np.cross(z_axes, x_axes)
    y_axes /= np.linalg.norm(y_axes, axis=1, keepdims=True)

    # Recompute orthogonal x-axes (Gram-Schmidt refinement)
    x_axes = np.cross(y_axes, z_axes)

    # Stack into rotation matrices
    rotations = np.stack([x_axes, y_axes, z_axes], axis=-1)  # (n_valid_faces, n_dims, n_dims)

    # Ensure right-handed coordinate systems (i.e., no reflection)
    dets = np.linalg.det(rotations)  # (n_valid_faces,)
    rotations[dets < 0, :, -1] *= -1  # Flip last axis if needed

    # Rotate points for all rotations: (n_valid_faces, n_points, n_dims)
    rotated_points = np.einsum('nj,fji->fni', points, rotations)

    # Compute AABB min and max for each rotation
    min_coords = rotated_points.min(axis=1)  # (n_valid_faces, n_dims)
    max_coords = rotated_points.max(axis=1)  # (n_valid_faces, n_dims)

    # Compute volumes
    volumes = np.prod(max_coords - min_coords, axis=1)  # (n_valid_faces,)

    # Find the minimal volume
    min_idx = np.argmin(volumes)
    min_volume = volumes[min_idx]
    best_rotation = rotations[min_idx]
    best_min = min_coords[min_idx]
    best_max = max_coords[min_idx]
    final_points = rotated_points[min_idx]

    # Compute bounding box corners in rotated space
    bbox_corners = np.array([
        [best_min[0], best_min[1], best_min[2]],
        [best_min[0], best_min[1], best_max[2]],
        [best_min[0], best_max[1], best_min[2]],
        [best_min[0], best_max[1], best_max[2]],
        [best_max[0], best_min[1], best_min[2]],
        [best_max[0], best_min[1], best_max[2]],
        [best_max[0], best_max[1], best_min[2]],
        [best_max[0], best_max[1], best_max[2]],
    ])  # (2^n_dims, n_dims)

    # Rotate bbox corners back to original space
    best_bbox = bbox_corners @ best_rotation.T  # (2^n_dims, n_dims)

    return best_rotation, best_bbox, min_volume, final_points
