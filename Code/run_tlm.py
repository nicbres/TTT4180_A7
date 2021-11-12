import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from tlm.tlm import TLM
from tlm.source import Source


def harmonic_function(time: float):
    return np.sin(2 * np.pi * 500 * time)


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
        wavelength_delta_x_ratio=10,
        reflection_coefficient_right=1.0,
    )

    fig, ax = plt.subplots()

    norm_max = 0.05
    norm = colors.Normalize(vmin=-norm_max, vmax=norm_max)
    im = ax.imshow(res_tlm.get_pressure_layer(), norm=norm)

    def animate(*args, **kwargs):
        for _ in np.arange(1):
            res_tlm.update()
        ax.clear()
        im = ax.imshow(res_tlm.get_pressure_layer()[::-1,:], norm=norm)
        return im,


    ani = animation.FuncAnimation(
        fig, animate, interval=5,
    )

    plt.show()
