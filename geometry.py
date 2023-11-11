import numpy as np
from scipy.spatial import Voronoi


def are_collinear(vs, tol=0.01):

    if len(vs) < 2:
        return True
    v_0 = vs[0]
    for v in vs[1:]:
        if sum(abs(np.cross(v_0, v))) > tol:
            return False
    return True


def are_coplanar(vs, tol=0.01):

    if len(vs) < 3:
        return True
    v_1 = vs[0]
    for v in vs[1:]:
        if not are_collinear([v_1, v]):
            ab = np.cross(v_1, v)
            break
    else:
        return True
    for v in vs[1:]:
        if abs(np.dot(ab, v)) > tol:
            return False
    return True


def find_vneighbors(points, central_points, key=1):

    """
     Parameter mod can takes values 1, 2, or 3 that correspond to the
    search for domains adjacent by vertices, edges or faces.
    """

    neighbors = {i: None for i in central_points}
    vor = Voronoi(points)
    for i in central_points:
        region = vor.regions[vor.point_region[i]]
        if -1 in region:
            raise ValueError("The domain for \"" + str(i) + "\" point is not closed!")
        local_neighbors = []
        for j in range(len(points)):
            numb_common_vertices = len(np.intersect1d(region, vor.regions[vor.point_region[j]]))
            if i != j and numb_common_vertices >= key:
                local_neighbors.append(j)
        neighbors[i] = local_neighbors
    return neighbors







