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
        wavelength_delta_x_ratio: float = 10.0,
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

        maximum_wavelength = self._sound_speed / maximum_frequency
        self._delta_x = maximum_wavelength / wavelength_delta_x_ratio
        self._delta_t = 1 / (wavelength_delta_x_ratio * maximum_frequency)

        self.x_max = int(np.ceil(self._length / self._delta_x))
        self.y_max = int(np.ceil(self._width / self._delta_x))

        # 3rd Dimension:
        #   1x Incident per Branch -> 4x
        #   1x Scattering per Branch -> 4x
        #   1x Source Layer
        #   => 9x Layers
        # 4th Dimension:
        #   1x Layer for current values
        #   2x Layer for next values
        self._layers = np.zeros((self.x_max, self.y_max, 9, 2))

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

        self._scatter_propagation_lookup = {
            0: (-1, 0),
            1: (0, -1),
            2: (1, 0),
            3: (0, 1),
        }

    @property
    def shape(self):
        return np.shape(self._layers[:, :, :, self._current_layer])

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

    def _update_next_incoming_horizontal(self, x: int):
        scat_layer_0_index = self._layer_lookup["scatter_0"]
        scat_layer_2_index = self._layer_lookup["scatter_2"]

        inc_layer_0_index = self._layer_lookup["incident_0"]
        inc_layer_2_index = self._layer_lookup["incident_2"]

        scat_index_branch_0 = scat_layer_2_index
        scat_index_branch_2 = scat_layer_0_index
        x_branch_0 = x + self._scatter_propagation_lookup[0][0]
        x_branch_2 = x + self._scatter_propagation_lookup[2][0]
        factor_branch_0 = 1
        factor_branch_2 = 1

        if x == 0:
            scat_index_branch_0 = scat_layer_0_index
            x_branch_0 = 0
            factor_branch_0 = self.reflection_coefficient_left
        elif x == (self.x_max - 1):
            scat_index_branch_2 = scat_layer_2_index
            x_branch_2 = self.x_max - 1
            factor_branch_2 = self.reflection_coefficient_right

        self._layers[x, :, inc_layer_0_index, self._next_layer] = (
            factor_branch_0
            * self._layers[
                x_branch_0,
                :,
                scat_index_branch_0,
                self._current_layer,
            ]
        )

        self._layers[x, :, inc_layer_2_index, self._next_layer] = (
            factor_branch_2
            * self._layers[
                x_branch_2,
                :,
                scat_index_branch_2,
                self._current_layer,
            ]
        )

    def _update_next_incoming_vertical(self, y: int):
        scat_layer_1_index = self._layer_lookup["scatter_1"]
        scat_layer_3_index = self._layer_lookup["scatter_3"]

        inc_layer_1_index = self._layer_lookup["incident_1"]
        inc_layer_3_index = self._layer_lookup["incident_3"]

        scat_index_branch_1 = scat_layer_3_index
        scat_index_branch_3 = scat_layer_1_index
        y_branch_1 = y + self._scatter_propagation_lookup[1][1]
        y_branch_3 = y + self._scatter_propagation_lookup[3][1]
        factor_branch_1 = 1
        factor_branch_3 = 1

        if y == 0:
            scat_index_branch_1 = scat_layer_1_index
            y_branch_1 = 0
            factor_branch_1 = self.reflection_coefficient_top
        elif y == (self.y_max - 1):
            scat_index_branch_3 = scat_layer_3_index
            y_branch_3 = self.y_max - 1
            factor_branch_3 = self.reflection_coefficient_bottom

        self._layers[:, y, inc_layer_1_index, self._next_layer] = (
            factor_branch_1
            * self._layers[
                :,
                y_branch_1,
                scat_index_branch_1,
                self._current_layer,
            ]
        )

        self._layers[:, y, inc_layer_3_index, self._next_layer] = (
            factor_branch_3
            * self._layers[
                :,
                y_branch_3,
                scat_index_branch_3,
                self._current_layer,
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

        for x in np.arange(self.shape[0]):
            self._update_next_incoming_horizontal(x)

        for y in np.arange(self.shape[1]):
            self._update_next_incoming_vertical(y)

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
