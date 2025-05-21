# Overview


### PCA Method

This aligns the bounding box to the principal axes of the point cloud.
This is a fast approximation, but can fail for non-ellipsoidal shapes.

This method uses principal component analysis (PCA) to find the rotation.
It is not guaranteed to find the optimal rotation,
but it is a good approximation
(cf. [On the bounding boxes obtained by principal component analysis](https://www.researchgate.net/publication/235758825_On_the_bounding_boxes_obtained_by_principal_component_analysis)).

### Convex Hull Based Methods

For 2D points, the **rotating calipers** method is used to find the global minimum as follows:

1. For each edge of the convex hull, consider that edge as the bottom edge of the bounding box.
2. Rotate the points so that this edge is aligned with the x-axis.
3. Compute the axis-aligned bounding box (AABB) of the rotated points.
4. Compute the area of the AABB.
5. Keep track of the minimum area found.
6. Return the values that correspond to the minimum area.

In 2D, the minimum-area bounding box has always one side aligned with an edge of the convex hull.
Therefore, the algorithm is guaranteed to find the global minimum.
On the other hand, in 3D, the minimum-volume bounding box can be aligned with
a face normal, an edge direction, or a combination of both.