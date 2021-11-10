import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from tlm.tlm import TLM
from tlm.source import Source


def harmonic_function(time: float):
    return np.cos(2 * np.pi * 500 * time)


if __name__ == "__main__":
    sources = [
        Source(coordinates=(0, 0), function=harmonic_function),
    ]

    res_tlm = TLM(
        maximum_frequency=2000,
        sound_speed=343,
        length=1,
        width=0.2,
        sources=sources,
        wavelength_delta_x_ratio=10,
    )

    fig, ax = plt.subplots()

    norm = colors.Normalize(vmin=-2.0, vmax=2.0)
    im = ax.imshow(res_tlm.get_pressure_layer().T, norm=norm)

    def animate(*args, **kwargs):
        res_tlm.update()
        im = ax.imshow(res_tlm.get_pressure_layer().T, norm=norm)
        return im,


    ani = animation.FuncAnimation(
        fig, animate, interval=1, save_count=30,
    )

    plt.show()
