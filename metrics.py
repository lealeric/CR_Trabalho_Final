import numpy as np
from math import pow
import networkx as nx
from random import choice

def jaccard_similarity(set1, set2):
    """
    Calcula a similaridade de Jaccard entre dois conjuntos.

    Args:
        set1 (set): O primeiro conjunto.
        set2 (set): O segundo conjunto.

    Returns:
        float: A similaridade de Jaccard entre os conjuntos.
    """
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

def calcula_metrica(G, node, chosen_nodes, max_distance):
    """
    Calcula a métrica personalizada para um nó.

    Args:
        G (nx.Graph): O grafo original.
        node (int): O nó para calcular a métrica.
        chosen_nodes (list): Lista de nós semente.
        max_distance (int): Distância máxima para considerar.

    Returns:
        dict: Dicionário com a métrica personalizada para cada nó.
    """
    custom_metric = {}
    for node_in_graph in G.nodes():
        for chosen_node in chosen_nodes:
            try:
                distance = nx.shortest_path_length(G, source=node_in_graph, target=chosen_node)
                if 0 < distance <= max_distance:
                    custom_metric[node_in_graph] += (1 / distance)
            except nx.NetworkXNoPath:
                pass

    return custom_metric

def gera_metricas(G, chosen_nodes = None, max_distance = np.inf, depreciation_factor = 'proporcional'):
    """
    Gera métricas personalizadas para todos os nós.

    Args:
        G (nx.Graph): O grafo original.
        chosen_nodes (list, optional): Lista de nós semente. Se None, gera 10 nós aleatórios.
        max_distance (int, optional): Distância máxima para considerar.

    Returns:
        dict: Dicionário com a métrica personalizada para cada nó.
    """
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
                    custom_metric[node] += pow(distance, -10)
                    if depreciation_factor == 'proporcional':
                        custom_metric[node] += (1 / distance)
                    elif depreciation_factor == 'exponencial':
                        custom_metric[node] += pow(2, -distance)
                    elif depreciation_factor == 'linear':
                        custom_metric[node] += (1 - distance / max_distance)
                    else:
                        custom_metric[node] += (1 / pow(distance, depreciation_factor))
            except nx.NetworkXNoPath:
                pass

    with open("custom_metric.txt", "w") as f:
        sorted_custom_metric = sorted(custom_metric.items(), key=lambda x: x[1], reverse=True)
        for node, value in sorted_custom_metric:
            f.write(f"{node}: {value}\n")

    return custom_metric


def novelty_metric(G,recommended_nodes, famous_nodes):
    """
    Calcula a métrica de novidade para um conjunto de nós recomendados.

    Args:
        G (nx.Graph): O grafo original.
        recommended_nodes (list): Lista de nós recomendados.
        famous_nodes (list): Lista de nós famosos.

    Returns:
        float: A métrica de novidade.
    """
    
    node_popularity_counts = dict.fromkeys(G.nodes(), 0)
    for node in famous_nodes:
        node_popularity_counts[node] += 1

    max_popularity_count = max(node_popularity_counts.values()) if node_popularity_counts else 0

    if max_popularity_count == 0:
        normalized_popularity_scores = dict.fromkeys(G.nodes(), 0.0)
    else:
        normalized_popularity_scores = {node: count / max_popularity_count for node, count in node_popularity_counts.items()}

    if not recommended_nodes:
        return 0.0

    sum_one_minus_popularity = 0
    for node_id in recommended_nodes:
        sum_one_minus_popularity += (1 - normalized_popularity_scores.get(node_id, 0.0))

    novelty = sum_one_minus_popularity / len(recommended_nodes)
    return novelty

def catalog_coverage(G, recommended_nodes):
    """
    Calcula a métrica de cobertura do catálogo para um conjunto de nós recomendados.

    Args:
        G (nx.Graph): O grafo original.
        recommended_nodes (list): Lista de nós recomendados.

    Returns:
        float: A métrica de cobertura do catálogo.
    """

    recommended_nodes_set = set(recommended_nodes)
    total_nodes = len(G.nodes())
    
    return (len(recommended_nodes_set) / total_nodes)*100