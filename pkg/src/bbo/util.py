import jax
import jax.numpy as jnp


@jax.jit
def box_vertices_from_bounds(lower_bounds: jnp.ndarray, upper_bounds: jnp.ndarray) -> jnp.ndarray:
    """Generate the vertices of a box given its lower and upper bounds.

    The vertices are ordered consistently for 2D (counter-clockwise) and 3D (face-order).

    Parameters
    ----------
    lower_bounds
        Lower bounds of the box
        as an array of shape `(n_dims,)`.
    upper_bounds
        Upper bounds of the box
        as an array of shape `(n_dims,)`.

    Returns
    -------
    Vertices of the box
    as an array of shape `(2^n_dims, n_dims)`.
    """
    ndim = lower_bounds.size

    # Generate binary grid for all index combinations (0 or 1 per axis) — Cartesian product
    grid = jnp.indices((2,) * ndim).reshape(ndim, -1).T  # (2^n_dims, n_dims)
    # Select lower/upper bounds per axis using the grid indices
    choices = jnp.stack([lower_bounds, upper_bounds], axis=0)  # (2, n_dims)
    vertices = choices[grid, jnp.arange(ndim)]  # (2^n_dims, n_dims)
    # Alternative:
    # vertices_signs = jnp.array(list(itertools.product(*[[-1, 1]] * ndim)))
    # vertices = jnp.where(vertices_signs < 0, lower_bounds, upper_bounds)

    if ndim == 2:
        # Order 2D vertices CCW for plotting
        center = jnp.mean(vertices, axis=0)
        angles = jnp.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
        sort_order = jnp.argsort(angles)
        vertices = vertices[sort_order]
    elif ndim == 3:
        # Reorder vertices to standard cuboid convention:
        # Bottom face (z=min): 000, 100, 110, 010 (CCW)
        # Top face    (z=max): 001, 101, 111, 011 (CCW)
        # Binary indices → map to desired order
        order = jnp.array([0b000, 0b100, 0b110, 0b010,  # Bottom face CCW
                           0b001, 0b101, 0b111, 0b011], dtype=jnp.int32)
        # grid indices are lexicographic, so this reorders them to face-wise order
        vertices = vertices[order]
    return vertices
