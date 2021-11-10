import numpy as np
import pytest

from tlm.source import Source


def _dummy_function(time: float) -> float:
    return 42.0


class TestSource:
    @staticmethod
    @pytest.mark.parametrize(
        "coordinates",
        [
            (0, 0),
            (10, 20),
        ],
    )
    def test_given_initial_values_then_coordinates_as_expected(coordinates):
        res_source = Source(
            coordinates=coordinates,
            function=_dummy_function,
        )

        assert (res_source.x, res_source.y) == coordinates

    @staticmethod
    @pytest.mark.parametrize(
        "time, result",
        [
            [0, 1],
            [0.25, 0],
        ],
    )
    def test_given_function_is_cosine_when_get_value_then_expected_value(
        time,
        result,
    ):
        def cosine(time: float) -> float:
            frequency = 1
            return np.cos(2 * np.pi * frequency * time)

        res_source = Source(
            function=cosine,
        )

        assert abs(res_source.value(time=time) - result) < 10 ** (-6)
