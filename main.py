import networkx as nx
import graph_ops
import visualization
import metrics
import numpy as np
import pandas as pd


def main():
    try:
        G = nx.read_gml('graph.gml')
        pos = nx.get_node_attributes(G, 'pos')
    except FileNotFoundError:
        n = 1000
        G, pos = graph_ops.create_geometric_graph(n, radius=58, x_max=1000, y_max=1000)
        nx.write_gml(G, 'graph.gml')
    print(nx.diameter(G))

    famous_nodes = []

    rng = np.random.default_rng(42)
    famous_nodes = [str(n) for n in rng.choice(list(G.nodes()), size=100, replace=True)]
    print(famous_nodes)

    if not nx.is_connected(G):
        list(nx.connected_components(G)) 

    comm = graph_ops.get_communities(G)
    visualization.plot_graph_communities(G, comm, pos, filename='resultados/graph.png')

    # Tentativa de usar centralidade para recomendar nós
    seed_nodes = graph_ops.get_seed_nodes(G, num_seeds=10)
    induced_subgraph = graph_ops.get_induced_subgraph(G, seed_nodes, n_hops=3)
    centrality = graph_ops.calculate_subgraph_centrality(induced_subgraph)

    node_colors_induced, min_bc, max_bc = visualization.get_node_colors_by_centrality(
        induced_subgraph.nodes(), centrality, cmap_name='viridis', seed_nodes=seed_nodes
    )
    visualization.plot_induced_subgraph(
        induced_subgraph, pos, node_colors_induced, min_bc, max_bc, filename='resultados/induced_subgraph.png'
    )
    
    combined_node_colors, min_bc_induced, max_bc_induced = visualization.get_node_colors_by_centrality(
        G.nodes(), centrality, cmap_name='viridis', default_color='lightgray', seed_nodes=seed_nodes
    )
    
    visualization.plot_final_visualization(
        G, comm, pos, seed_nodes, 
        min_bc_induced, max_bc_induced, 
        combined_node_colors, 
        filename='resultados/graph_visualization.png'
    )

    ### Tentativa de usar métrica personalizada para recomendar nós

    nos_escolhidos = [str(node) for node in [167, 58, 737, 538, 541, 596, 732, 915, 434, 540]]
    print(nos_escolhidos)

    parametros = ['proporcional', 'exponencial', 'linear', 2, 5, 10, 20, 30, nx.diameter(G)]
    ranking_recomendados = []
    novidade = []
    cobertura = []

    df = pd.DataFrame(columns=['Parametro', 'Novidade', 'Cobertura', 'Recomendados'])


    for parametro in parametros:
        metrica = metrics.gera_metricas(G, nos_escolhidos, max_distance=10, depreciation_factor=parametro)
        visualization.plotar_metrica(G, metrica, nos_escolhidos, famous_nodes, filename=f'resultados/metrica_{parametro}.png', pos=pos)

        maior_metrica = max(metrica.values())
        print(f"Maior métrica: {maior_metrica}")

        recomendados = [node for node, value in metrica.items() if value >= maior_metrica*0.5]
        print(f"{len(recomendados)} nós com métrica >= 1: {recomendados}")
        
        novidade_metrica = metrics.novelty_metric(G, nos_escolhidos, famous_nodes)
        cobertura_metrica = metrics.catalog_coverage(G, recomendados)

        print(f"Novidade: {novidade_metrica}")
        print(f"Cobertura do catálogo: {cobertura_metrica:.3f}%")

        ranking_recomendados.append(recomendados)
        novidade.append(novidade_metrica)
        cobertura.append(cobertura_metrica)

        df = df._append({'Parametro': parametro, 'Novidade': novidade_metrica, 'Cobertura': cobertura_metrica, 'Recomendados': recomendados}, ignore_index=True)

    df.to_csv('resultados/resultados.csv', index=False)

    visualization.plot_summary_metrics(parametros, ranking_recomendados, novidade, cobertura, filename='resultados/resultados.png')

    print(famous_nodes)




if __name__ == "__main__":
    main()
