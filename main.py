import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from utils import generate_random_2d_coordinate, generate_random_hex_colors
# Import the new function
from visualization import draw_community_edges, get_node_colors_by_centrality

def main():
    n = 1000
    pos = {i: generate_random_2d_coordinate(0, 1000, 0, 1000) for i in range(n)}
    G = nx.random_geometric_graph(n, 58, pos=pos)

    if not nx.is_connected(G):
        list(nx.connected_components(G)) # Just to check connectivity

    comm = nx.community.greedy_modularity_communities(G)

    node_colors = {}
    node_labels = {}
    cores = generate_random_hex_colors(len(comm))
    for i, c in enumerate(comm):
        for node in c:
            node_colors[node] = cores[i]
            node_labels[node] = i

    plt.figure(figsize=(7, 5))
    nx.draw(G, with_labels=False, font_weight='bold', pos=pos, node_size=50, node_color=[node_colors[n] for n in G.nodes()], labels=node_labels, font_color='blue', font_size=20)
    plt.savefig('graph.png')
    plt.close()

    # --- Seed Logic ---
    num_seed_nodes = 10
    seed_nodes = list(np.random.choice(G.nodes(), num_seed_nodes, replace=False))

    n_hops = 3
    nodes_in_n_hops = set()
    for seed_node in seed_nodes:
        reachable_nodes = nx.single_source_shortest_path_length(G, seed_node, cutoff=n_hops)
        nodes_in_n_hops.update(reachable_nodes.keys())
    nodes_in_n_hops_list = list(nodes_in_n_hops)
    
    induced_subgraph = G.subgraph(nodes_in_n_hops_list)

    centrality = nx.closeness_centrality(induced_subgraph)
    
    centrality_sub = {}
    for sub in nx.connected_components(induced_subgraph):
        for node in sub:
            centrality_sub[node] = nx.eigenvector_centrality_numpy(induced_subgraph.subgraph(sub))[node]
    for node in centrality_sub:
        centrality[node] = centrality_sub[node]

    # --- Intermediate Plot: Induced Subgraph ---
    # REFACTORED: Use helper function
    node_colors_induced, min_bc, max_bc = get_node_colors_by_centrality(
        induced_subgraph.nodes(), centrality, cmap_name='viridis', seed_nodes=seed_nodes
    )
    
    cmap = cm.get_cmap('viridis')

    plt.figure(figsize=(10, 8))
    nx.draw(induced_subgraph,
            pos=pos,
            node_color=[node_colors_induced[n] for n in induced_subgraph.nodes()],
            with_labels=False,
            node_size=50,
            alpha=0.7)
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=min_bc, vmax=max_bc))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=plt.gca())
    cb.set_label('Centrality')
    red_patch = mpatches.Patch(color='red', label='Seed Nodes')
    plt.legend(handles=[red_patch])
    plt.title('Induced Subgraph: Seed Nodes and Centrality')
    plt.savefig('induced_subgraph.png')
    plt.close()

    # --- FINAL VISUALIZATION UPDATE ---
    
    # REFACTORED: Use helper function for the full graph
    # We want non-induced nodes to be 'lightgray', induced nodes colored by centrality, and seed nodes red.
    # The helper function handles centrality and seed nodes.
    # We can pass G.nodes() but ONLY allow centrality mapping for induced nodes.
    
    # Filter centrality map to only include induced nodes (which it strictly does already, but good to be explicit mentally)
    # The helper handles "missing" nodes by assigning default_color ('lightgray').
    
    combined_node_colors, min_bc_induced, max_bc_induced = get_node_colors_by_centrality(
        G.nodes(), centrality, cmap_name='viridis', default_color='lightgray', seed_nodes=seed_nodes
    )
    
    final_node_colors_list = [combined_node_colors[node] for node in G.nodes()]

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    nx.draw_networkx_nodes(G,
            pos=pos,
            node_color=final_node_colors_list,
            node_size=50,
            alpha=0.7,
            ax=ax)

    draw_community_edges(G, comm, pos, ax=ax)

    handles = []
    red_patch = mpatches.Patch(color='red', label='Seed Nodes')
    handles.append(red_patch)
    gray_patch = mpatches.Patch(color='lightgray', label='Other Nodes')
    handles.append(gray_patch)

    if centrality and (max_bc_induced > 0):
        sm_bc = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=min_bc_induced, vmax=max_bc_induced))
        sm_bc.set_array([])
        cb_bc = plt.colorbar(sm_bc, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
        cb_bc.set_label('Induced Subgraph Node Centrality')

    plt.legend(handles=handles, loc='upper left')
    plt.title('Graph Visualization: Communities (Hulls), Seed Nodes, and Centrality')
    plt.savefig('graph_visualization.png')
    plt.close()

if __name__ == "__main__":
    main()
