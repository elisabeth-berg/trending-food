import matplotlib.pyplot as plt
import numpy as np

def make_plot(food):
    fig, ax = plt.subplots()
    x = np.linspace(-10, 10)
    y = x**2
    ax.plot(x, y)
    ax.set_title(food)
    return fig
