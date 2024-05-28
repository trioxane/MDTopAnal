import pandas as pd
import numpy as np
import networkx as nx
import dynetx as dn
import matplotlib.pyplot as plt

from collections import Counter, defaultdict


class TemporalNetworkAnalysis:

    def __init__(self, data_list: list):

        G_list = [data['G'] for data in data_list]
        nodes_data_list = [pd.DataFrame(data['nodes_data']) for data in data_list]

        self.dG, self._all_edges = self._create_dynamic_graph(G_list)
        self._create_weighted_static_graph(G_list[0])
        self._collect_nodes_list_data(nodes_data_list)

    def _create_dynamic_graph(self, G_list):
        """
        Create dynamic graph object from a list of snapshot graphs
        as well as collecting all edges ever encountered in the snapshots
        """
        all_edges = []
        dynamic_G = dn.DynGraph()

        for i, snapshot_G in enumerate(G_list):
            snapshot_edges = list(snapshot_G.edges(keys=True))
            dynamic_G.add_interactions_from(snapshot_edges, t=i)
            all_edges.extend([(*sorted(e[:2]), e[2]) for e in snapshot_edges])

        return dynamic_G, all_edges

    def _create_weighted_static_graph(self, G0):
        """
        Create static representation of the dynamic graph as a weighted graph
        """
        # G0 - first graph in a list used for weighted static graph construction -
        # stores node coordinates for visualisation
        wsG = G0.copy()
        wsG.remove_edges_from(list(wsG.edges()))
        self.wsG = wsG

        # at first make a list with all edges ever encountered along the trajectory
        # and group edges between the same nodes pair but with different translations
        counter = Counter(self._all_edges)
        edges_locations = defaultdict(list)

        for edge, count in counter.items():
            n1, n2, t = edge
            n1 = n1 if n1 < n2 else n2
            n2 = n2 if n2 > n1 else n1
            edges_locations[(n1, n2)].append((edge, count))


        # now if the edge between a pair of nodes occured with different translations
        # select the translation which appeared most frequently
        selected_edges = []
        all_edges = []

        for nodes_pair, edges_data in edges_locations.items():

            edge_persistence = self.dG.edge_contribution(*nodes_pair)

            if len(edges_data) > 1:
                most_common_edge = sorted(edges_data, key=lambda x: x[1])[-1]
                selected_edges.append([*most_common_edge[0], edge_persistence])
                all_edges.extend([[*ed, edge_persistence] for ed in edges_data])
            else:
                selected_edges.append([*edges_data[0][0], edge_persistence])
                all_edges.append([*edges_data[0][0], edge_persistence])

        self._all_edges = all_edges
        self._edge_persistence_list = sorted(selected_edges, key=lambda x: x[-1], reverse=True)

        for n1, n2, translation, persistence in self._edge_persistence_list:
            self.wsG.add_edge(n1, n2, key=translation, weight=persistence)

    def _collect_nodes_list_data(self, nodes_list):
        """
        Combine the node VDP characteristics from each frame into a single dataframe
        """
        df = pd.concat(nodes_list, axis=1).T
        df = df.astype({
            'frame': 'int32',
            'neighbours_SA_CV': 'float',
            'N_direct_neighbours': 'int8',
            'N_indirect_neighbours': 'int8',
            'N_neighbours': 'int8',
            'neighbours_indices': 'object',
            'MCN': 'int8',
            })

        self.node_characteristics = df

        # equal number of molecules should be in each frame throughout trajectory
        if len(df.groupby('frame')['MCN'].count().value_counts()) > 1:
            print('!!!! WARNING: number of molecules in a frame throughout trajectory ARE NOT EQUAL !!!!')
            print('N_molecules: N_frames with N_molecules')
            print('\n'.join(
                [f"{k}: {v}" for k, v in df.groupby('frame')['MCN'].count().value_counts().to_dict().items()]
            )
                 )

    def show_edge_contribution_info(self, plot_edge_weigts=True, return_persistence_df=False, title=''):
        """
        Show data on the edge persistence
        """
        df = pd.DataFrame(
            self._edge_persistence_list,
            columns=['node1', 'node2', 'translation', 'persistence']
        ).sort_values(by='persistence', ascending=False)

        if plot_edge_weigts:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharex=True)
            fig.suptitle(title)
            df['persistence'].hist(bins=40, density=True, label='hist', ax=ax1)
            df['persistence'].plot.kde(bw_method=0.035, lw=2.0, color='navy', label='kde', ax=ax1)
            ax1.set_title('Edge persistence distribution')
            ax1.set_xlabel('persistence')
            ax1.set_xlim(-0.01, 1.01)
            data_sorted = np.sort(df['persistence'].values)
            y = np.arange(len(data_sorted)) / len(data_sorted)
            ax2.set_title('Edge persistence CDF')
            ax2.plot(data_sorted, y, 'k-', lw=2.5)
            ax2.set_xlabel('persistence')
            ax1.legend()
            plt.show()

        if return_persistence_df:
            return df

    def get_filtered_weighted_static_graph(self, edge_weight_threshold=0.9):
        """
        Return graph with edges filtered to have weight >= edge_weight_threshold.
        Nodes are taken from the first snapshot graph in the G_list
        """

        filtered_wsG = self.wsG.copy()
        filtered_wsG.remove_edges_from(list(filtered_wsG.edges()))

        for edge in self.wsG.edges(keys=True, data=True):
            n1, n2, translation, data = edge
            if data['weight'] >= edge_weight_threshold:
                filtered_wsG.add_edge(n1, n2, key=translation, weight=data['weight'])

        return filtered_wsG
