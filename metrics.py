import numpy as np

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
