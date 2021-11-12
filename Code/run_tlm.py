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


def plot(tlm: TLM, cnorm: colors.Normalize):
    fig, ax = plt.subplots()
    im = ax.imshow(tlm.get_pressure_layer(), norm=cnorm)

    def animate(*args, **kwargs):
        for _ in np.arange(4):
            res_tlm.update()
        ax.clear()
        pressure_layer = res_tlm.get_pressure_layer()[::-1, :]
        im = ax.imshow(pressure_layer, norm=cnorm)
        return im,

    ani = animation.FuncAnimation(
        fig, animate, interval=5,
    )

    plt.show()


def generate_images(
    tlm: TLM,
    cnorm: colors.Normalize,
    colormap: str = 'inferno',
    number_of_images: int = 10,
    skip_rate: int = 80,
):
    colormap = plt.get_cmap(colormap)
    scalar_mappable = cm.ScalarMappable(norm=cnorm, cmap=colormap)

    for i in range(number_of_images):
        for _ in np.arange(skip_rate):
            tlm.update()
        pressure_layer = tlm.get_pressure_layer()[::-1, :]
        image = scalar_mappable.to_rgba(pressure_layer)
        imageio.imwrite(f'./images/pipe_{i}.png', image)


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
        wavelength_delta_x_ratio=10,
        reflection_coefficient_right=0.1,
    )

    norm_max = 0.35
    norm = colors.Normalize(vmin=-norm_max, vmax=norm_max)

    #plot(tlm=res_tlm, cnorm=norm)
    generate_images(tlm=res_tlm, cnorm=norm)
