import numpy as np
import networkx as nx
from random import choice
def jaccard_similarity(set1, set2):
    s1 = set(set1)
    s2 = set(set2)

    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))

    if union == 0:
        return 0.0
    else:
        return intersection / union


def matriz_densidade_entre_comunidades(G, comunidades):
    n = len(comunidades)
    matriz = np.zeros((n, n))

    for i, Ci in enumerate(comunidades):
        for j, Cj in enumerate(comunidades):
            edges_between = sum(1 for u, v in G.edges() if
                                (u in Ci and v in Cj) or
                                (v in Ci and u in Cj))
            matriz[i, j] = edges_between / (len(Ci) * len(Cj))
    return matriz

# def calcula_metrica(G, node, chosen_nodes, max_distance):
#     custom_metric = {}
#     for node_in_graph in G.nodes():
#         for chosen_node in chosen_nodes:
#             try:
#                 distance = nx.shortest_path_length(G, source=node_in_graph, target=chosen_node)
#                 if 0 < distance <= max_distance:
#                     custom_metric[node_in_graph] += (1 / distance)
#             except nx.NetworkXNoPath:
#                 pass

#     return custom_metric

def gera_metricas(G, chosen_nodes = None, max_distance = 3):
    if chosen_nodes is None:
        chosen_nodes = [choice(list(G.nodes())) for _ in range(10)]
    
    custom_metric = {}
    for node in G.nodes():
        custom_metric[node] = 0

    print(f"Custom metric initialized for {len(custom_metric)} nodes.")
    print(f"Chosen nodes: {chosen_nodes}")
    print(f"Maximum distance for calculations: {max_distance}")

    for node in G.nodes():
        for chosen_node in chosen_nodes:
            try:
                distance = nx.shortest_path_length(G, source=node, target=chosen_node)
                if 0 < distance <= max_distance:
                    custom_metric[node] += (1 / distance)
            except nx.NetworkXNoPath:
                pass

    return custom_metric
