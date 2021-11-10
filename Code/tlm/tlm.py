from typing import Iterable, Union

import numpy as np

from tlm.source import Source


class TLM:
    def __init__(
        self,
        maximum_frequency: float,  # [Hz]
        length: float,  # [m]
        width: float,  # [m]
        sources: Iterable[Source] = [],
        sound_speed: float = 343,  # [m/s]
        reflection_coefficient_bottom: float = 1.0,
        reflection_coefficient_left: float = 1.0,
        reflection_coefficient_right: float = 1.0,
        reflection_coefficient_top: float = 1.0,
    ):
        # Dimensions should not be changed after initialization
        self._maximum_frequency = maximum_frequency
        self._sound_speed = sound_speed
        self._length = length
        self._width = width

        # Reflection Coefficient could be changed after
        self.reflection_coefficient_bottom = reflection_coefficient_bottom
        self.reflection_coefficient_left = reflection_coefficient_left
        self.reflection_coefficient_right = reflection_coefficient_right
        self.reflection_coefficient_top = reflection_coefficient_top

        maximum_wavelength = sound_speed / maximum_frequency
        self._delta_x = maximum_wavelength / 10
        self._delta_t = self._delta_x / self.sound_speed

        x_max = int(np.ceil(self.length / self._delta_x))
        y_max = int(np.ceil(self.width / self._delta_x))

        # 3rd Dimension:
        #   1x Incident per Branch -> 4x
        #   1x Scattering per Branch -> 4x
        #   1x Source Layer
        #   => 9x Layers
        # 4th Dimension:
        #   1x Layer for current values
        #   2x Layer for next values
        self._layers = np.zeros((x_max, y_max, 9, 2))

        # reverse lookup table
        self._layer_lookup = {
            "incident_0": 0,
            "incident_1": 1,
            "incident_2": 2,
            "incident_3": 3,
            "scatter_0": 4,
            "scatter_1": 5,
            "scatter_2": 6,
            "scatter_3": 7,
            "source": 8,
        }
        self._layer_lookup.update(
            {value: key for key, value in self._layer_lookup.items()}
        )

        self.step_count = 0
        self._current_layer = self.step_count
        self._next_layer = self.step_count + 1

        self.sources = sources

    @property
    def sound_speed(self):
        return self._sound_speed

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    @property
    def shape(self):
        return np.shape(self._layers[:, :, :, self._current_layer])

    @property
    def x_max(self):
        return self.shape[0] - 1

    @property
    def y_max(self):
        return self.shape[1] - 1

    def get_layer(self, index: Union[str, int]):
        if type(index) == str:
            index = self._layer_lookup[index]
        return self._layers[:, :, index, self._current_layer]

    def _update_source_layer(self):
        src_index = self._layer_lookup["source"]
        current_time = self.step_count * self._delta_t

        for src in self.sources:
            self._layers[src.x, src.y, src_index, self._next_layer] = src.value(
                current_time
            )

    def _calculate_next_scattering(self, x: int, y: int) -> np.array:
        inc_0_index = self._layer_lookup["incident_0"]
        inc_1_index = self._layer_lookup["incident_1"]
        inc_2_index = self._layer_lookup["incident_2"]
        inc_3_index = self._layer_lookup["incident_3"]
        src_index = self._layer_lookup["source"]

        current_vector = np.array(
            [
                self._layers[x, y, inc_0_index, self._current_layer],  # I0
                self._layers[x, y, inc_1_index, self._current_layer],  # I1
                self._layers[x, y, inc_2_index, self._current_layer],  # I2
                self._layers[x, y, inc_3_index, self._current_layer],  # I3
                self._layers[x, y, src_index, self._next_layer],  # S
            ]
        )

        scattering_matrix = np.array(
            [
                [-1, 1, 1, 1, 1],
                [1, -1, 1, 1, 1],
                [1, 1, -1, 1, 1],
                [1, 1, 1, -1, 1],
            ]
        )

        return 1 / 2 * np.dot(scattering_matrix, current_vector)

    def _calculate_next_incoming_for_branch(self, x: int, y: int, branch: int) -> float:
        if (x == 0) and (branch == 0):
            # left boundary
            scat_index = self._layer_lookup["scatter_0"]
            scat_value = self._layers[x, y, scat_index, self._current_layer]
            return self.reflection_coefficient_left * scat_value
        elif (x == self.x_max) and (branch == 2):
            # right boundary
            scat_index = self._layer_lookup["scatter_2"]
            scat_value = self._layers[x, y, scat_index, self._current_layer]
            return self.reflection_coefficient_right * scat_value
        elif (y == 0) and (branch == 1):
            # top boundary
            scat_index = self._layer_lookup["scatter_1"]
            scat_value = self._layers[x, y, scat_index, self._current_layer]
            return self.reflection_coefficient_top * scat_value
        elif (y == self.y_max) and (branch == 3):
            # bottom boundary
            scat_index = self._layer_lookup["scatter_3"]
            scat_value = self._layers[x, y, scat_index, self._current_layer]
            return self.reflection_coefficient_bottom * scat_value

        scat_index = self._layer_lookup[f"scatter_{(branch + 2) % 4}"]
        if branch == 0:
            x_scat = x - 1
            y_scat = y
        elif branch == 1:
            x_scat = x
            y_scat = y - 1
        elif branch == 2:
            x_scat = x + 1
            y_scat = y
        elif branch == 3:
            x_scat = x
            y_scat = y + 1
        scat_value = self._layers[x_scat, y_scat, scat_index, self._current_layer]

        return scat_value

    def _calculate_next_incoming(self, x: int, y: int) -> np.array:
        return np.array(
            [
                self._calculate_next_incoming_for_branch(x=x, y=y, branch=branch)
                for branch in np.arange(4)
            ]
        )

    def _increment_step(self):
        self.step_count += 1
        self._current_layer = self.step_count % 2
        self._next_layer = (self.step_count + 1) % 2

    def update(self):
        self._update_source_layer()
        inc_indices = np.array(
            [
                self._layer_lookup["incident_0"],
                self._layer_lookup["incident_1"],
                self._layer_lookup["incident_2"],
                self._layer_lookup["incident_3"],
            ]
        )

        scat_indices = np.array(
            [
                self._layer_lookup["scatter_0"],
                self._layer_lookup["scatter_1"],
                self._layer_lookup["scatter_2"],
                self._layer_lookup["scatter_3"],
            ]
        )

        # calculate next step incoming and scattering
        for x in np.arange(self.shape[0]):
            for y in np.arange(self.shape[1]):
                scattering_values = self._calculate_next_scattering(x, y)
                self._layers[x, y, scat_indices, self._next_layer] = scattering_values

                incoming_values = self._calculate_next_incoming(x, y)
                self._layers[x, y, inc_indices, self._next_layer] = incoming_values

        self._increment_step()

    def get_pressure_layer(self):
        all_indices = np.array(
            [
                self._layer_lookup["incident_0"],
                self._layer_lookup["incident_1"],
                self._layer_lookup["incident_2"],
                self._layer_lookup["incident_3"],
                self._layer_lookup["scatter_0"],
                self._layer_lookup["scatter_1"],
                self._layer_lookup["scatter_2"],
                self._layer_lookup["scatter_3"],
            ]
        )

        return np.sum(self._layers[:, :, all_indices, self._current_layer], axis=2)
