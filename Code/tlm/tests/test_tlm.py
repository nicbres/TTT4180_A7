import numpy as np
import pytest

from tlm.tlm import TLM
from tlm.source import Source


class TestTLM:
    @staticmethod
    def test_given_initial_values_then_initialization_successful():
        TLM(
            maximum_frequency=2000,
            length=2,
            width=0.2,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "freq, sound_speed, length, width, expected_shape",
        [
            [1, 10, 100, 10, (100, 10, 9, 2)],  # Simple test values
            [2000, 343, 2, 0.2, (117, 12, 9, 2)],  # Assignment values
        ],
    )
    def test_given_initial_values_then_shape_as_expected(
        freq,
        sound_speed,
        length,
        width,
        expected_shape,
    ):
        ref_shape = expected_shape

        res_tlm = TLM(
            maximum_frequency=freq,
            sound_speed=sound_speed,
            length=length,
            width=width,
        )

        assert res_tlm.layers.shape == ref_shape

    @staticmethod
    def test_given_no_source_when_update_then_no_error():
        res_tlm = TLM(
            maximum_frequency=2000,
            sound_speed=343,
            length=2,
            width=0.2,
        )

        res_tlm.update()

    @staticmethod
    def test_given_source_when_update_then_no_error():
        def harmonic_function(time: float):
            return np.cos(2 * np.pi * 500 * time)

        sources = [Source(coordinates=(0, 0), function=harmonic_function)]

        res_tlm = TLM(
            maximum_frequency=2000,
            sound_speed=343,
            length=2,
            width=0.2,
            sources=sources,
        )

        res_tlm.update()

    @staticmethod
    def test_given_source_when_get_pressure_layer_then_expected_shape():
        def harmonic_function(time: float):
            return np.cos(2 * np.pi * 500 * time)

        sources = [Source(coordinates=(0, 0), function=harmonic_function)]

        ref_shape = (12, 117)

        res_tlm = TLM(
            maximum_frequency=2000,
            sound_speed=343,
            length=2,
            width=0.2,
            sources=sources,
        )

        res_tlm.update()
        res_shape = np.shape(res_tlm.get_pressure_layer())

        assert res_shape == ref_shape
