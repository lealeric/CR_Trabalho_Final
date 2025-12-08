import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx

def plot_degree_distribution_complementar(grafos, titles=None):
  """Plota o gráfico de dispersão da distribuição complementar do grau.
  Cria um layout 2x2: cada grafo é plotado em escala linear (esquerda) e log-log (direita).
  Primeiro grafo na parte superior, segundo grafo na parte inferior.

  Args:
      grafos (list or nx.Graph): Lista de grafos do NetworkX ou um único grafo.
      titles (list): Lista de títulos para cada grafo.
      output_path (str): Caminho de saída do arquivo.
      output_file_name (str): Nome do arquivo de saída.
      show (bool): Se o gráfico deve ser exibido.
  """
  # Se apenas um grafo for passado, converte para lista
  if not isinstance(grafos, list):
      grafos = [grafos]

  if titles is None:
      titles = [f"Grafo {i+1}" for i in range(len(grafos))]

  # Determina o número de linhas baseado no número de grafos
  num_grafos = len(grafos)

  # Cria subplots em layout 2x2 (ou 1x2 se apenas um grafo)
  _, axes = plt.subplots(num_grafos, 2, figsize=(10, 5 * num_grafos))

  # Se apenas um grafo, axes precisa ser reformatado para 2D
  if num_grafos == 1:
      axes = axes.reshape(1, -1)

  for i, (grafo, title) in enumerate(zip(grafos, titles)):
      degrees = [degree for node, degree in grafo.degree()]
      degrees.sort()
      prob = [1 - (j/len(degrees)) for j in range(len(degrees))]

      # Gráfico linear (coluna da esquerda)
      axes[i, 0].scatter(degrees, prob, alpha=0.7)
      axes[i, 0].set_xlabel("Grau")
      axes[i, 0].set_ylabel("Probabilidade Complementar")
      axes[i, 0].set_title(f"{title} - Escala Linear")
      axes[i, 0].grid(True, alpha=0.3)
      axes[i, 0].set_ylim(0, 1)

      # Gráfico log-log (coluna da direita)
      axes[i, 1].scatter(degrees, prob, alpha=0.7)
      axes[i, 1].set_xscale('log')
      axes[i, 1].set_yscale('log')
      axes[i, 1].set_xlabel("Grau (log)")
      axes[i, 1].set_ylabel("Probabilidade Complementar (log)")
      axes[i, 1].set_title(f"{title} - Escala Log-Log")
      axes[i, 1].grid(True, alpha=0.3)

  plt.tight_layout()
  # plt.savefig(f"{output_path}{output_file_name}", dpi=300, bbox_inches='tight')

  # print(f"Gráfico salvo em: {output_path}{output_file_name}")

  # if show:
  #     plt.show()

  plt.show()
  plt.close()


def plotar_matriz_densidade(matriz):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz, cmap="viridis")

    n = matriz.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{i}" for i in range(n)])
    ax.set_yticklabels([f"C{i}" for i in range(n)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)

    ax.set_title("Matriz de Densidade entre Comunidades")
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()

def draw_community_edges(G, communities, pos, ax=None, colors=None, inter_community_color='red', alpha=0.5):
    """
    Desenha arestas com cores baseadas na estrutura da comunidade.
    Arestas dentro de uma comunidade recebem a cor da comunidade.
    Arestas entre comunidades diferentes recebem a cor inter_community_color.
    """
    if ax is None:
        ax = plt.gca()
    
    if colors is None:
        # Generate random colors if not provided
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        colors = [cmap(i) for i in np.linspace(0, 1, len(communities))]
        
    # Map each node to its community index
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i
            
    edge_colors = []
    for u, v in G.edges():
        comm_u = node_to_community.get(u)
        comm_v = node_to_community.get(v)
        
        if comm_u is not None and comm_v is not None and comm_u == comm_v:
            edge_colors.append(colors[comm_u % len(colors)])
        else:
            edge_colors.append(inter_community_color)
            
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=alpha, ax=ax)

def get_node_colors_by_centrality(nodes, centrality_map, cmap_name='viridis', default_color='lightgray', seed_nodes=None):
    """
    Gera um dicionário de cores dos nós com base nos valores de centralidade.
    
    Args:
        nodes: Iterável de nós para colorir.
        centrality_map: Dicionário mapeando nós para valores de centralidade.
        cmap_name: Nome do mapa de cores do matplotlib a ser usado.
        default_color: Cor para nós sem valores de centralidade (se houver).
        seed_nodes: Lista/conjunto de nós para colorir estritamente como 'red'.
        
    Returns:
        dict: node -> cor
        (min_val, max_val): Tupla de valores mínimo e máximo de centralidade para escala.
    """
    if not centrality_map:
        return {n: default_color for n in nodes}, 0, 0

    values = [centrality_map[n] for n in nodes if n in centrality_map]
    if not values:
         return {n: default_color for n in nodes}, 0, 0

    min_val = min(values)
    max_val = max(values)
    
    cmap = cm.get_cmap(cmap_name)
    colors = {}
    
    seed_nodes_set = set(seed_nodes) if seed_nodes else set()
    
    for node in nodes:
        if node in seed_nodes_set:
            colors[node] = 'red'
            continue
            
        if node in centrality_map:
            val = centrality_map[node]
            if max_val == min_val:
                norm_val = 0.5
            else:
                norm_val = (val - min_val) / (max_val - min_val)
            colors[node] = cmap(norm_val)
        else:
            colors[node] = default_color
            
    return colors, min_val, max_val


def plotar_metrica(G, metrica, seed_nodes=None, filename='metrica.png', pos=None):
    node_colors = {}
    node_labels = {}

    metric_values = [score for node, score in metrica.items() if node not in seed_nodes]

    if metric_values:
        min_metric = min(metric_values)
        max_metric = max(metric_values)
    else:
        min_metric = 0
        max_metric = 0

    cmap = cm.YlGnBu

    for node in G.nodes():
        if node in seed_nodes:
            node_colors[node] = 'red'
            node_labels[node] = f'{node}'
        else:
            if max_metric == min_metric:
                normalized_metric = 0.5
            else:
                normalized_metric = (metrica[node] - min_metric) / (max_metric - min_metric)
            node_colors[node] = cmap(normalized_metric)
            node_labels[node] = f'{metrica[node]:.2f}'

    if pos is None:
        pos = nx.spring_layout(G)

    plt.figure(figsize=(12, 10))
    nx.draw(G,
            pos=pos,
            node_color=[node_colors[n] for n in G.nodes()],
            with_labels=True,
            labels=node_labels,
            font_size=8,
            font_weight='bold',
            node_size=400,
            alpha=0.8)

    if metric_values:
        sm = cm.ScalarMappable(cmap=cmap,
                            norm=mcolors.Normalize(vmin=min_metric, vmax=max_metric))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', fraction=0.02, pad=0.02)
        cb.set_label('Score da Métrica')

    red_patch = mpatches.Patch(color='red', label='Nós escolhidos')
    plt.legend(handles=[red_patch])

    plt.title('Visualização da Métrica')
    plt.savefig(filename)
    plt.close()

def plot_graph_communities(G, comm, pos, filename='graph.png'):
    """
    Plots the graph with communities colored.
    """
    # Import locally to avoid potential circular dependency issues if utils imports visualization
    from utils import generate_random_hex_colors 
    
    node_colors = {}
    node_labels = {}
    cores = generate_random_hex_colors(len(comm))
    for i, c in enumerate(comm):
        for node in c:
            node_colors[node] = cores[i]
            node_labels[node] = i

    plt.figure(figsize=(7, 5))
    sorted_nodes = list(G.nodes())
    # Use get with default to be safe
    colors_list = [node_colors.get(n, 'lightgray') for n in sorted_nodes]
    
    nx.draw(G, with_labels=False, font_weight='bold', pos=pos, node_size=50, 
            node_color=colors_list, 
            labels=node_labels, font_color='blue', font_size=20)
    plt.savefig(filename)
    plt.close()

def plot_induced_subgraph(subgraph, pos, node_colors, min_bc, max_bc, filename='induced_subgraph.png'):
    """
    Plots the induced subgraph with centrality heatmap.
    """
    cmap = cm.get_cmap('viridis')
    plt.figure(figsize=(10, 8))
    
    sorted_nodes = list(subgraph.nodes())
    colors_list = [node_colors.get(n, 'lightgray') for n in sorted_nodes]
    
    nx.draw(subgraph,
            pos=pos,
            node_color=colors_list,
            with_labels=False,
            node_size=50,
            alpha=0.7)
            
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=min_bc, vmax=max_bc))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=plt.gca())
    cb.set_label('Centralidade')
    
    red_patch = mpatches.Patch(color='red', label='Nós Sementes')
    plt.legend(handles=[red_patch])
    plt.title('Subgrafo Induzido: Nós Sementes e Centralidade')
    plt.savefig(filename)
    plt.close()

def plot_final_visualization(G, comm, pos, seed_nodes, min_bc_induced, max_bc_induced, final_node_colors, filename='graph_visualization.png'):
    """
    Plots the final visualization with complex overlays.
    """
    cmap = cm.get_cmap('viridis')
    
    sorted_nodes = list(G.nodes())
    final_node_colors_list = [final_node_colors.get(n, 'lightgray') for n in sorted_nodes]

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
    red_patch = mpatches.Patch(color='red', label='Nós Sementes')
    handles.append(red_patch)
    gray_patch = mpatches.Patch(color='lightgray', label='Outros Nós')
    handles.append(gray_patch)

    # Check if we have valid range for colorbar
    if min_bc_induced is not None and max_bc_induced is not None and max_bc_induced > min_bc_induced:
        sm_bc = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=min_bc_induced, vmax=max_bc_induced))
        sm_bc.set_array([])
        cb_bc = plt.colorbar(sm_bc, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
        cb_bc.set_label('Centralidade dos Nós do Subgrafo Induzido')

    plt.legend(handles=handles, loc='upper left')
    plt.title('Visualização do Grafo: Comunidades, Nós Sementes e Centralidade')
    plt.savefig(filename)
    plt.close()