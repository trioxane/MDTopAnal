import numpy as np
from scipy.spatial import Voronoi

from collections import defaultdict
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

def calculate_solid_angle(center, coords):
    """
    Helper method to calculate the solid angle of a set of coords from the
    center.

    Args:
        center (3x1 array): Center to measure solid angle from.
        coords (Nx3 array): List of coords to determine solid angle.

    Returns:
        The solid angle.
    """
    # Compute the displacement from the center
    r = [np.subtract(c, center) for c in coords]

    # Compute the magnitude of each vector
    r_norm = [np.linalg.norm(i) for i in r]

    # Compute the solid angle for each tetrahedron that makes up the facet
    #  Following: https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
    angle = 0
    for i in range(1, len(r) - 1):
        j = i + 1
        tp = np.abs(np.dot(r[0], np.cross(r[i], r[j])))
        de = (
            r_norm[0] * r_norm[i] * r_norm[j]
            + r_norm[j] * np.dot(r[0], r[i])
            + r_norm[i] * np.dot(r[0], r[j])
            + r_norm[0] * np.dot(r[i], r[j])
        )
        my_angle = (0.5 * pi if tp > 0 else -0.5 * pi) if de == 0 else np.arctan(tp / de)
        angle += (my_angle if my_angle > 0 else my_angle + np.pi) * 2

    return angle

class Voro:

    #Central points should be first in the points list
    def __init__(self, points, num_central):

        self.points = np.array(points)
        self.num_central = num_central
        self.vor = Voronoi(points)
        self.p_adjacency = self.calc_p_adjacency()

    def get_VDP_data(self, omega_threshold: float = 0.05) -> dict:
        """
        Constructs VDP for each point and returns its VDP characteristics
        as well as contact types with neighbours
        Args:
            omega_threshold: threshold solid angle; small VDP faces will be ignored

        Returns:
            neighbours_dict: dict with neighbours characteristics
        """
        p_neighbours_dict = defaultdict(list)
        all_vertices = self.vor.vertices

        for (p1, p2), vertices in self.vor.ridge_dict.items():

            if -1 not in vertices and (p1 < self.num_central or p2 < self.num_central):

                # Get the solid angle of the face and check if it is not too small
                facets = [all_vertices[i] for i in vertices]
                p1_coords = self.points[p1]
                solid_angle_p1 = calculate_solid_angle(p1_coords, facets)
                if solid_angle_p1 < omega_threshold:
                    continue

                if is_inside(all_vertices[vertices], self.points[[p1, p2]].sum(axis=0) / 2):
                    contact_type = "direct"
                else:
                    contact_type = "indirect"

                if p1 < self.num_central:
                    p_neighbours_dict[p1].append((p2, solid_angle_p1, contact_type))
                if p2 < self.num_central:
                    p_neighbours_dict[p2].append((p1, solid_angle_p1, contact_type))

        neighbours_dict = {
            p1: {
                'neighbours_indices': [nd[0] for nd in neighbours],
                'SA': [nd[1] for nd in neighbours],
                'N_direct_neighbours': len([nd[2] for nd in neighbours if nd[2] == 'direct']),
                'N_indirect_neighbours': len([nd[2] for nd in neighbours if nd[2] == 'indirect'])
            }
            for p1, neighbours in p_neighbours_dict.items()
        }

        return neighbours_dict

    def calc_p_adjacency(self, omega_threshold=0.25, check_direct=False):
        """
        OMEGA_THRESHOLD: (0.1-0.2)
        """
        p_adjacency = [[] for _ in range(self.num_central)]
        # Get the coordinates of every vertex
        all_vertices = self.vor.vertices
        for (p1, p2), vertices in self.vor.ridge_dict.items():
            if -1 not in vertices and (p1 < self.num_central or p2 < self.num_central):

                if omega_threshold is not None:
                    # Get the solid angle of the face and check if it is not too small
                    facets = [all_vertices[i] for i in vertices]
                    p1_coords = self.points[p1]
                    solid_angle_p1 = calculate_solid_angle(p1_coords, facets)
                    if solid_angle_p1 < omega_threshold:
                        continue

                if check_direct:
                    if is_inside(self.vor.vertices[vertices], sum(self.vor.points[[p1, p2]]) / 2):
                        contact_type = "direct"
                    else:
                        contact_type = "indirect"
                else:
                    contact_type = "direct"

                if p1 < self.num_central:
                    p_adjacency[p1] += [(p2, contact_type)]
                if p2 < self.num_central:
                    p_adjacency[p2] += [(p1, contact_type)]
        self.p_adjacency = p_adjacency
        return self.p_adjacency
