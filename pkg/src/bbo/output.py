from dataclasses import dataclass

from bbo.typing import atypecheck, JAXArray, Num


@atypecheck
@dataclass
class BBOOutput:
    """Bounding box optimization output.

    Attributes
    ----------
    points
        Rotated points in minimal bounding box alignment.
    box
        Corners of the minimal bounding box in the original coordinate system.
    rotation
        Rotation matrix aligning points to minimal bounding box axes.
    volume
        Volume of the minimal bounding box.
    """
    points: Num[JAXArray, "*n_batches n_samples n_features"]
    box: Num[JAXArray, "*n_batches 2**n_features n_features"]
    rotation: Num[JAXArray, "*n_batches n_features n_features"]
    volume: Num[JAXArray, "*n_batches"]
