"""Utility functions for testing."""


def expected_output_shapes(n_batches: int | None, n_samples: int, n_features: int) -> dict:
    """Get expected output shapes from input shape."""
    shape_single = {
        "points": (n_samples, n_features),
        "box": (2 ** n_features, n_features),
        "rotation": (n_features, n_features),
        "volume": (),
    }
    return shape_single if n_batches is None else {
        k: (n_batches, *v) for k, v in shape_single.items()
    }
