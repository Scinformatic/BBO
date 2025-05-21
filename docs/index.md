

For other algorithms, see:
- [Minimum bounding box algorithms - Wikipedia](https://en.wikipedia.org/wiki/Minimum_bounding_box_algorithms)
- https://perso.uclouvain.be/chia-tche.chang/resources/CGM11_paper.pdf
- https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
- https://math.stackexchange.com/questions/2342844/how-to-find-the-rotation-which-minimizes-the-volume-of-the-bounding-box


## References

### General
- [Bounding Volume - Wikipedia](https://en.wikipedia.org/wiki/Bounding_volume)
- [Minimum bounding box - Wikipedia](https://en.wikipedia.org/wiki/Minimum_bounding_box)


### Algorithms
- [Minimum bounding box algorithms - Wikipedia](https://en.wikipedia.org/wiki/Minimum_bounding_box_algorithms)
- [Minimum bounding box algorithms - GH Issues](https://github.com/VolkerH/np_obb/issues)
- [O'Rourke, J. Finding minimal enclosing boxes. International Journal of Computer and Information Sciences 14, 183–199 (1985).](https://doi.org/10.1007/BF00991005)
  This is the original algorithm by O'Rourke.
- [Fast oriented bounding box optimization on the rotation group SO(3, R)](https://doi.org/10.1145/2019627.2019641)
  This [paper](https://perso.uclouvain.be/chia-tche.chang/resources/CGM11_paper.pdf)
  describes an algorithm to compute an optimal 3D oriented bounding box,
  as a faster and more reliable alternative to the original O'Rourke algorithm.
  The computation of the minimal-volume OBB is formulated as an unconstrained
  optimization problem on the rotation group SO(3, R).
  It is solved using a hybrid method combining the genetic and Nelder-Mead algorithms.
- [Convex hull - SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)

### Code
- [Open3D Python Package](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Bounding-volumes)
- [github.com/varunagrawal/bbox](https://github.com/varunagrawal/bbox)
  Python library for 2D and 3D bounding boxes, providing a set of flexible primitives and functions.
- [github.com/VolkerH/np_obb](https://github.com/VolkerH/np_obb)
  PCA implementation to calculate the oriented bounding box.
- [github.com/chadogome/OptimalOBB](https://github.com/chadogome/OptimalOBB)
  Matlab scripts implementing algorithms for finding the optimal oriented bounding box of a set of points.
  It includes both the original O'Rourke algorithm, as well as Chang's hybrid optimization algorithm
  (in "Fast oriented bounding box optimization on the rotation group SO(3, R)").
- [github.com/cansik/LongLiveTheSquare](https://github.com/cansik/LongLiveTheSquare)
  Algorithm in C# to find the arbitrarily oriented minimum bounding box in R².
  This also has a [C++ implementation](https://github.com/schmidt9/MinimalBoundingBox).
- [CARLA Simulator: Bounding Boxes](https://carla.readthedocs.io/en/0.9.15/tuto_G_bounding_boxes/)
- [OpenCV tutorial: Creating Bounding boxes and circles for contours](https://docs.opencv.org/4.x/da/d0c/tutorial_bounding_rects_circles.html)
- [Matlab implementation of Korsawe's algorithm](https://www.mathworks.com/matlabcentral/fileexchange/18264-minimal-bounding-box)
- https://bitbucket.org/william_rusnack/minimumboundingbox/src/master/


### Discussions
- [Algorithm to find the minimum-area-rectangle for given points in order to compute the major and minor axis length - Stackoverflow](https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/14675742#14675742)
- [Minimum oriented bounding box. Implementation in Grasshopper + Python script node](https://discourse.mcneel.com/t/minimum-oriented-bounding-box-implementation-in-grasshopper-python-script-node/64344)
- [Create the Oriented Bounding-box (OBB) with Python and NumPy](https://stackoverflow.com/questions/32892932/create-the-oriented-bounding-box-obb-with-python-and-numpy)
- [How do I find the smallest possible bounding box for a solid? - Rhinoceros Forums](https://discourse.mcneel.com/t/how-do-i-find-the-smallest-possible-bounding-box-for-a-solid/90008)
- [Seeking Generalisation Strategies for Building Outlines in PostGIS? - StackExchange](https://gis.stackexchange.com/questions/3739/seeking-generalisation-strategies-for-building-outlines-in-postgis)