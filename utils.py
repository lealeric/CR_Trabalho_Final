from random import uniform, randint

def generate_random_2d_coordinate(x_min, x_max, y_min, y_max):
    x = uniform(x_min, x_max)
    y = uniform(y_min, y_max)
    return (x, y)

def generate_random_hex_colors(n):
    colors = []
    for _ in range(n):
        hex_color = '#%06x' % randint(0, 0xFFFFFF)
        colors.append(hex_color)
    return colors
