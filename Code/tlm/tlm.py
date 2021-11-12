"""Provides the main function for running TLM.

The function provided here can be used for computing wave propagation in a
wave guide using TLM.

Typical usage:
    from tlm.source import Source
    from tlm.tlm import TLM

    def source_function(time: float) -> float:
        return np.sin(2 * np.pi * 500 * time)

    sources = [
        Source(coordinates=(10, 13), function=harmonic_function),
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

    res_tlm.update()
    pressure_matrix = res_tlm.get_pressure_layer()
"""
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
        self.layers = np.zeros((self.x_max, self.y_max, 9, 2))

        # layer indices lookup table
        self._layer_lookup = {
            "incident_0": 0,
            "incident_1": 1,
            "incident_2": 2,
            "incident_3": 3,
            "scattering_0": 4,
            "scattering_1": 5,
            "scattering_2": 6,
            "scattering_3": 7,
            "source": 8,
        }

        self.step_count = 0
        self._current_layer = self.step_count
        self._next_layer = self.step_count + 1

        self.sources = sources

    @property
    def incident_indices(self) -> np.array:
        return np.array([self._layer_lookup[f"incident_{i}"] for i in np.arange(4)])

    @property
    def scattering_indices(self) -> np.array:
        return np.array([self._layer_lookup[f"scattering_{i}"] for i in np.arange(4)])

    def _update_source_layer(self):
        src_index = self._layer_lookup["source"]
        current_time = self.step_count * self._delta_t

        for src in self.sources:
            amplitude = src.value(current_time)
            self.layers[src.x, src.y, src_index, self._next_layer] = amplitude

    def _update_next_scattering(self, x: int) -> np.array:
        inc_0_index = self._layer_lookup["incident_0"]
        inc_1_index = self._layer_lookup["incident_1"]
        inc_2_index = self._layer_lookup["incident_2"]
        inc_3_index = self._layer_lookup["incident_3"]
        src_index = self._layer_lookup["source"]

        current_vector = np.array(
            [
                self.layers[x, :, inc_0_index, self._current_layer],  # I0
                self.layers[x, :, inc_1_index, self._current_layer],  # I1
                self.layers[x, :, inc_2_index, self._current_layer],  # I2
                self.layers[x, :, inc_3_index, self._current_layer],  # I3
                self.layers[x, :, src_index, self._next_layer],  # S
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

        # Computes all scattering values for a whole x-vector from the x,y plane
        # by using multidimensional np.dot()
        result = 1 / 2 * np.dot(scattering_matrix, current_vector)
        self.layers[x, :, self.scattering_indices, self._next_layer] = result

    def _update_next_incident_horizontal(self, x: int):
        # prepares all necessary indices
        scat_layer_0_index = self._layer_lookup["scattering_0"]
        scat_layer_2_index = self._layer_lookup["scattering_2"]

        inc_layer_0_index = self._layer_lookup["incident_0"]
        inc_layer_2_index = self._layer_lookup["incident_2"]

        scat_index_branch_0 = scat_layer_2_index
        scat_index_branch_2 = scat_layer_0_index

        x_branch_0 = x - 1
        x_branch_2 = x + 1

        factor_branch_0 = 1
        factor_branch_2 = 1

        # handling for horizontal boundaries
        if x == 0:
            scat_index_branch_0 = scat_layer_0_index
            x_branch_0 = 0
            factor_branch_0 = self.reflection_coefficient_left
        elif x == (self.x_max - 1):
            scat_index_branch_2 = scat_layer_2_index
            x_branch_2 = self.x_max - 1
            factor_branch_2 = self.reflection_coefficient_right

        # update the actual branch layers
        self.layers[x, :, inc_layer_0_index, self._next_layer] = (
            factor_branch_0
            * self.layers[
                x_branch_0,
                :,
                scat_index_branch_0,
                self._current_layer,
            ]
        )

        self.layers[x, :, inc_layer_2_index, self._next_layer] = (
            factor_branch_2
            * self.layers[
                x_branch_2,
                :,
                scat_index_branch_2,
                self._current_layer,
            ]
        )

    def _update_next_incident_vertical(self, y: int):
        # prepares all necessary indices
        scat_layer_1_index = self._layer_lookup["scattering_1"]
        scat_layer_3_index = self._layer_lookup["scattering_3"]

        inc_layer_1_index = self._layer_lookup["incident_1"]
        inc_layer_3_index = self._layer_lookup["incident_3"]

        scat_index_branch_1 = scat_layer_3_index
        scat_index_branch_3 = scat_layer_1_index

        y_branch_1 = y - 1
        y_branch_3 = y + 1

        factor_branch_1 = 1
        factor_branch_3 = 1

        # handling for vertical boundaries
        if y == 0:
            scat_index_branch_1 = scat_layer_1_index
            y_branch_1 = 0
            factor_branch_1 = self.reflection_coefficient_top
        elif y == (self.y_max - 1):
            scat_index_branch_3 = scat_layer_3_index
            y_branch_3 = self.y_max - 1
            factor_branch_3 = self.reflection_coefficient_bottom

        # update the actual branch layers
        self.layers[:, y, inc_layer_1_index, self._next_layer] = (
            factor_branch_1
            * self.layers[
                :,
                y_branch_1,
                scat_index_branch_1,
                self._current_layer,
            ]
        )

        self.layers[:, y, inc_layer_3_index, self._next_layer] = (
            factor_branch_3
            * self.layers[
                :,
                y_branch_3,
                scat_index_branch_3,
                self._current_layer,
            ]
        )

    def _increment_step_count(self):
        self.step_count += 1
        self._current_layer = self.step_count % 2
        self._next_layer = (self.step_count + 1) % 2

    def update(self):
        """Update all layers and increment step count.

        This function first updates the amplitudes on the source layer. Then
        it calculates the scattering and incident values for all nodes and
        branches, before finally updating step count and switching out the
        current layers with the computed next layers.
        """
        self._update_source_layer()

        # calculate next step incident and scattering
        for x in np.arange(self.x_max):
            self._update_next_scattering(x)
            self._update_next_incident_horizontal(x)

        for y in np.arange(self.y_max):
            self._update_next_incident_vertical(y)

        self._increment_step_count()

    def get_pressure_layer(self):
        """Computes the pressure from superposition of incident and scattering waves.

        Returns:
            A numpy array with a shape corresponding to delta_x and the given physical
            length and width.
        """
        incident_pressure = np.sum(
            self.layers[:, :, self.incident_indices, self._current_layer],
            axis=2,
        )
        """
        scattering_pressure = (-1.) * np.sum(
            self.layers[:, :, self.scattering_indices, self._current_layer],
            axis=2,
        )

        return np.sum(np.dstack((incident_pressure, scattering_pressure)), axis=2).T 
        """
        return np.sum(np.dstack((incident_pressure, self.layers[:,:,8,self._current_layer])), axis=2).T
