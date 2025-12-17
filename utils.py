from random import uniform, randint

def generate_random_2d_coordinate(x_min, x_max, y_min, y_max):
    """
    Gera uma coordenada 2D aleatória.

    Args:
        x_min (float): Limite mínimo em x.
        x_max (float): Limite máximo em x.
        y_min (float): Limite mínimo em y.
        y_max (float): Limite máximo em y.

    Returns:
        tuple: Coordenada 2D aleatória.
    """
    x = uniform(x_min, x_max)
    y = uniform(y_min, y_max)
    return (x, y)

def generate_random_hex_colors(n):
    """
    Gera uma lista de cores hexadecimais aleatórias.

    Args:
        n (int): Número de cores a gerar.

    Returns:
        list: Lista de cores hexadecimais aleatórias.
    """
    colors = []
    for _ in range(n):
        hex_color = '#%06x' % randint(0, 0xFFFFFF)
        colors.append(hex_color)
    return colors
