import networkx as nx
import numpy as np
from utils import generate_random_2d_coordinate

def create_geometric_graph(n, radius, x_max=1000, y_max=1000):
    """
    Gera um grafo geométrico aleatório.

    Args:
        n (int): Número de nós.
        radius (float): Raio da vizinhança.
        x_max (int): Limite máximo em x.
        y_max (int): Limite máximo em y.

    Returns:
        tuple: Tupla contendo o grafo e as posições dos nós.
    """
    pos = {i: generate_random_2d_coordinate(0, x_max, 0, y_max) for i in range(n)}
    G = nx.random_geometric_graph(n, radius, pos=pos)
    return G, pos

def get_communities(G):
    """
    Detecta comunidades usando a modularidade gulosa.

    Args:
        G (nx.Graph): O grafo original.

    Returns:
        list: Lista de comunidades.
    """
    return nx.community.greedy_modularity_communities(G)

def get_seed_nodes(G, num_seeds=10):
    """
    Seleciona nós semente aleatórios.

    Args:
        G (nx.Graph): O grafo original.
        num_seeds (int): Número de nós semente.

    Returns:
        list: Lista de nós semente.
    """
    return list(np.random.choice(G.nodes(), num_seeds, replace=False))

def get_induced_subgraph(G, seed_nodes, n_hops=3):
    """
    Gera um subgrafo induzido por nós dentro de n_hops de seed_nodes.

    Args:
        G (nx.Graph): O grafo original.
        seed_nodes (list): Lista de nós semente.
        n_hops (int): Número de hops para considerar.

    Returns:
        nx.Graph: Subgrafo induzido.
    """
    nodes_in_n_hops = set()
    for seed_node in seed_nodes:
        reachable_nodes = nx.single_source_shortest_path_length(G, seed_node, cutoff=n_hops)
        nodes_in_n_hops.update(reachable_nodes.keys())
    
    return G.subgraph(list(nodes_in_n_hops))

def calculate_subgraph_centrality(subgraph):
    """
    Calcula a centralidade para o subgrafo. 
    Usa a centralidade de proximidade global.

    Args:
        subgraph (nx.Graph): O subgrafo para calcular a centralidade.

    Returns:
        dict: Dicionário com a centralidade de cada nó.
    """
    centrality = nx.closeness_centrality(subgraph)
    
    return centrality
