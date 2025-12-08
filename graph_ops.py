import networkx as nx
import numpy as np
from utils import generate_random_2d_coordinate

def create_geometric_graph(n, radius, x_max=1000, y_max=1000):
    """
    Creates a random geometric graph and returns the graph and positions.
    """
    pos = {i: generate_random_2d_coordinate(0, x_max, 0, y_max) for i in range(n)}
    G = nx.random_geometric_graph(n, radius, pos=pos)
    return G, pos

def get_communities(G):
    """
    Detects communities using greedy modularity.
    """
    return nx.community.greedy_modularity_communities(G)

def get_seed_nodes(G, num_seeds=10):
    """
    Selects random seed nodes.
    """
    return list(np.random.choice(G.nodes(), num_seeds, replace=False))

def get_induced_subgraph(G, seed_nodes, n_hops=3):
    """
    Returns a subgraph induced by nodes within n_hops of seed_nodes.
    """
    nodes_in_n_hops = set()
    for seed_node in seed_nodes:
        # single_source_shortest_path_length returns {node: distance}
        reachable_nodes = nx.single_source_shortest_path_length(G, seed_node, cutoff=n_hops)
        nodes_in_n_hops.update(reachable_nodes.keys())
    
    return G.subgraph(list(nodes_in_n_hops))

def calculate_subgraph_centrality(subgraph):
    """
    Calculates centrality for the subgraph. 
    Uses eigenvetor centrality per connected component to avoid convergence issues 
    on disconnected graphs, and closeness centrality globally.
    """
    centrality = nx.closeness_centrality(subgraph)
    
    centrality_sub = {}
    for sub in nx.connected_components(subgraph):
        # Create a view/subgraph for the component to calculate eigenvector centrality
        comp = subgraph.subgraph(sub)
        try:
            # eigenvector_centrality_numpy is generally more robust for this than eigenvector_centrality
            eigen = nx.eigenvector_centrality_numpy(comp)
            for node in sub:
                centrality_sub[node] = eigen[node]
        except Exception:
             # Fallback if eigenvector fails (though numpy version usually works)
             for node in sub:
                centrality_sub[node] = 0
                
    # Merge specific centrality if needed, but original code used 'centrality' (closeness) 
    # and updated it with 'centrality_sub' (eigenvector).
    # This seems like a specific requirement of the original logic: 
    # "metrics based on eigenvector for components, but closeness for structure?"
    # Actually, let's look at the original code carefully:
    # centrality = nx.closeness_centrality(induced_subgraph)
    # ... loop components ...
    #    centrality_sub[node] = nx.eigenvector_centrality_numpy(...)
    # for node in centrality_sub: centrality[node] = centrality_sub[node]
    # So it OVERWRITES closeness with eigenvector where available.
    
    for node in centrality_sub:
        centrality[node] = centrality_sub[node]
        
    return centrality
