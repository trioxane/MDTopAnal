import numpy as np
from scipy.spatial import Voronoi
import warnings


def calc_angles(ps1, ps2, ps3):

    vs1 = np.array(ps1) - ps2
    vs2 = np.array(ps3) - ps2
    a = np.array(list(map(np.linalg.norm, vs1))) * list(map(np.linalg.norm, vs2))
    a[a == 0] = None
    with warnings.catch_warnings():
        cos = np.array([np.dot(v1, v2) / a for v1, v2, a in zip(vs1, vs2, a)])
        cos[cos > 1] = 1.0
        cos[cos < -1] = - 1.0
        angles = np.arccos(cos)
        angles[angles > np.pi] = np.pi
    return angles


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


def is_inside(vertices, pt, tol=0.001):


    inside = False
    n_v = len(vertices)
    centroid = sum(vertices) / n_v
    v1 = pt - centroid
    if sum(abs(v1)) < tol:
        return True
    v2, v3 = vertices[:2] - centroid
    norm = np.cross(v2, v3)
    norm /= np.linalg.norm(norm)
    if abs(np.dot(v1 / np.linalg.norm(v1), norm)) > tol:
        return False
    angles_sum = sum(calc_angles(vertices, [pt]*len(vertices), [*vertices[1:], vertices[0]])) / np.pi
    if abs(angles_sum - round(angles_sum)) < 1e-9:
        return True
    return inside


class Voro:

    #Central points should be first in the points list

    def __init__(self, points, num_central):

        self.points = np.array(points)
        self.num_central = num_central
        self.vor = Voronoi(points)
        self.p_adjacency = self.calc_p_adjacency()

    def calc_p_adjacency(self):

        p_adjacency = [[] for _ in range(self.num_central)]
        for (p1, p2), vertices in self.vor.ridge_dict.items():
            if -1 not in vertices and (p1 < self.num_central or p2 < self.num_central):
                if is_inside(self.vor.vertices[vertices], sum(self.vor.points[[p1, p2]]) / 2):
                    contact_type = "direct"
                else:
                    contact_type = "indirect"
                if p1 < self.num_central:
                    p_adjacency[p1] += [(p2, contact_type)]
                if p2 < self.num_central:
                    p_adjacency[p2] += [(p1, contact_type)]
        self.p_adjacency = p_adjacency
        return self.p_adjacency

def unit_vector(vector):
    """
    Returns the unit vector of the vector
    """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """
    Returns the angle in degrees between vectors v1 and v2
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

