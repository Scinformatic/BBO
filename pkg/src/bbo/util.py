import jax
import jax.numpy as jnp


@jax.jit
def box_vertices_from_bounds(lower_bounds: jnp.ndarray, upper_bounds: jnp.ndarray) -> jnp.ndarray:
    """Generate the vertices of a box given its lower and upper bounds.

    Parameters
    ----------
    lower_bounds
        Lower bounds of the box.
    upper_bounds
        Upper bounds of the box.
    """
    ndim = lower_bounds.size
    choices = jnp.stack([lower_bounds, upper_bounds], axis=0)  # (2, n_dims)
    # Generate all index combinations (0 or 1 per axis) — Cartesian product
    grid = jnp.indices((2,) * ndim).reshape(ndim, -1).T  # (2^n_dims, n_dims)
    # Select min/max per axis using the grid indices
    return choices[grid, jnp.arange(ndim)]  # (2^n_dims, n_dims)
