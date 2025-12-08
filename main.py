import networkx as nx
import graph_ops
import visualization
import metrics

def main():
    n = 1000
    G, pos = graph_ops.create_geometric_graph(n, radius=58, x_max=1000, y_max=1000)

    if not nx.is_connected(G):
        list(nx.connected_components(G)) 

    comm = graph_ops.get_communities(G)
    visualization.plot_graph_communities(G, comm, pos, filename='graph.png')

    # Tentativa de usar centralidade para recomendar nós
    # seed_nodes = graph_ops.get_seed_nodes(G, num_seeds=10)
    # induced_subgraph = graph_ops.get_induced_subgraph(G, seed_nodes, n_hops=3)
    # centrality = graph_ops.calculate_subgraph_centrality(induced_subgraph)

    # node_colors_induced, min_bc, max_bc = visualization.get_node_colors_by_centrality(
    #     induced_subgraph.nodes(), centrality, cmap_name='viridis', seed_nodes=seed_nodes
    # )
    # visualization.plot_induced_subgraph(
    #     induced_subgraph, pos, node_colors_induced, min_bc, max_bc, filename='induced_subgraph.png'
    # )
    
    # combined_node_colors, min_bc_induced, max_bc_induced = visualization.get_node_colors_by_centrality(
    #     G.nodes(), centrality, cmap_name='viridis', default_color='lightgray', seed_nodes=seed_nodes
    # )
    
    # visualization.plot_final_visualization(
    #     G, comm, pos, seed_nodes, 
    #     min_bc_induced, max_bc_induced, 
    #     combined_node_colors, 
    #     filename='graph_visualization.png'
    # )

    # Tentativa de usar métrica personalizada
    nos_escolhidos = graph_ops.get_seed_nodes(G, num_seeds=10)
    metrica = metrics.gera_metricas(G, nos_escolhidos, 10)
    visualization.plotar_metrica(G, metrica, nos_escolhidos, filename='metrica.png', pos=pos)
    
    # Tentativa de usar métrica personalizada para recomendar nós
    

if __name__ == "__main__":
    main()
