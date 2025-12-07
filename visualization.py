import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
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
    Draws edges with colors based on community structure.
    Edges within a community get the community color.
    Edges between different communities get the inter_community_color.
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
    Generates a dictionary of node colors based on centrality values.
    
    Args:
        nodes: Iterable of nodes to color.
        centrality_map: Dictionary mapping nodes to centrality values.
        cmap_name: Name of the matplotlib colormap to use.
        default_color: Color for nodes without centrality values (if any).
        seed_nodes: List/set of nodes to color strictly as 'red'.
        
    Returns:
        dict: node -> color
        (min_val, max_val): Tuple of min and max centrality values for scaling.
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
