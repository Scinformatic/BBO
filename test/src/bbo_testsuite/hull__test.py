import jax.numpy as jnp
import numpy as np
import arrayer

import bbo


def hull__batch_vs_single__test():
    """Test hull batch vs single outputs."""
    n_batches = 10
    n_points = 100
    points = np.random.rand(n_batches, n_points, 3)
    output_batch = bbo.approx_hull(points)

    # Verify output shapes
    output_batch_shapes = get_expected_output_shapes(n_batches, n_points, 3)
    for output_name, expected_shape in output_batch_shapes.items():
        value = getattr(output_batch, output_name)
        assert value.shape == expected_shape, f"Shape mismatch in {output_name}: expected {expected_shape}, got {value.shape}."

    # Verify batch vs single output
    for batch_idx in range(n_batches):
        output_single = bbo.approx_hull(points[batch_idx])
        for output_name in output_batch_shapes.keys():
            value_batch = getattr(output_batch, output_name)[batch_idx]
            value_single = getattr(output_single, output_name)
            assert jnp.allclose(value_batch, value_single, atol=1e-6), f"Value mismatch in {output_name} for batch {batch_idx}: expected {value_single}, got {value_batch}."


def hull__test():
    """Test hull output."""
    n_batches = 100
    n_points = 100
    points = np.random.rand(n_batches, n_points, 3)
    output = bbo.approx_hull(points)
    assert np.all(arrayer.matrix.is_rotation(output.rotation)), "Rotation matrices are not pure rotation."
    assert np.allclose(output.points, points @ output.rotation), "Points are not aligned with the rotation matrix."

    lower_bounds = jnp.min(points, axis=1)
    upper_bounds = jnp.max(points, axis=1)
    volumes = jnp.prod(upper_bounds - lower_bounds, axis=1)
    assert np.all(output.volume <= volumes), "Volume of the bounding box is larger than the volume of the points."
    return


def get_expected_output_shapes(n_batches: int | None, n_samples: int, n_features: int) -> dict:
    """Get expected output shapes for hull."""
    shape_single = {
        "points": (n_samples, n_features),
        "box": (2 ** n_features, n_features),
        "rotation": (n_features, n_features),
        "volume": (),
    }
    return shape_single if n_batches is None else {k: (n_batches, *v) for k, v in shape_single.items()}
