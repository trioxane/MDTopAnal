import numpy as np
from geometry import are_collinear, are_coplanar


class PeriodicGraph:

    def __init__(self,
                 vertices=[],
                 edges=[],
                 translations=[],
                 vertex_attributes={},
                 edges_attributes={}):

        self.vertices = vertices
        self.edges = np.array(edges)
        self.translations = np.array(translations, dtype=int)
        self.vertex_attributes = {k: np.array(v) for k, v in vertex_attributes.items()}
        self.edges_attributes = {k: np.array(v) for k, v in edges_attributes.items()}
        self.adjacency_list = None
        self.calc_adjacency()

    def calc_adjacency(self):

        self.adjacency_list = [[] for _ in range(len(self.vertices))]
        for i, (v_1, v_2) in enumerate(self.edges):
            self.adjacency_list[v_1].append([v_2, self.translations[i]])
            self.adjacency_list[v_2].append([v_1, -self.translations[i]])
        return self.adjacency_list

    def get_neighbors(self, vertex, translation):
        return [(n, t + translation) for n, t in self.adjacency_list[vertex]]

    def calc_periodicity(self):

        traversed = [None for _ in range(self.v_number)]
        for i in range(self.v_number):
            if traversed[i] is None:
                translations = []
                traversed[i] = np.array([0, 0, 0])
                vertices = [(i, np.array([0, 0, 0]))]
                for j, (v, t_1) in enumerate(vertices):
                    for n, t_2 in self.get_neighbors(v, t_1):
                        translation_image = traversed[n]
                        if translation_image is None:
                            traversed[n] = t_2
                            vertices.append((n, t_2))
                        elif (translation_image != t_2).any():
                            translations.append(t_2 - translation_image)
                yield vertices, self.get_periodicity(translations)

    @staticmethod
    def get_periodicity(translations):

        translations
        if len(translations) == 0:
            return 0
        elif are_collinear(translations, tol=0):
            return 1
        elif are_coplanar(translations, tol=0):
            return 2
        return 3











