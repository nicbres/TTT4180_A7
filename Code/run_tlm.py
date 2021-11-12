import imageio
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from tlm.tlm import TLM
from tlm.source import Source


def harmonic_function(time: float):
    return np.cos(2 * np.pi * 171 * time)


def impulse_function(time: float):
    if time == 0:
        return 1.0
    else:
        return 0.


if __name__ == "__main__":
    sources = [
        Source(coordinates=(0, 0), function=harmonic_function),
    ]

    res_tlm = TLM(
        maximum_frequency=2000,
        sound_speed=343,
        length=2,
        width=0.2,
        sources=sources,
        wavelength_delta_x_ratio=20,
        reflection_coefficient_right=1.0,
    )

    fig, ax = plt.subplots()

    norm_max = 0.35
    norm = colors.Normalize(vmin=-norm_max, vmax=norm_max)
    colormap = plt.get_cmap('inferno')
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
    #im = ax.imshow(res_tlm.get_pressure_layer(), norm=norm)

    """
    def animate(*args, **kwargs):
        for _ in np.arange(4):
            res_tlm.update()
        ax.clear()
        pressure_layer = res_tlm.get_pressure_layer()[::-1, :]
        if image_count < 100:
            imageio.imwrite(f'./images/pipe_{image_count}', scalar_mappable.to_rgba(pressure_layer))
            image_count += 1
        im = ax.imshow(pressure_layer, norm=norm)
        return im,
    """

    for i in range(20):
        for _ in np.arange(80):
            res_tlm.update()
        pressure_layer = res_tlm.get_pressure_layer()[::-1, :]
        image = scalar_mappable.to_rgba(pressure_layer)
        imageio.imwrite(f'./images/pipe_{i}.png', image)

"""
    ani = animation.FuncAnimation(
        fig, animate, interval=5,
    )

    plt.show()
"""
