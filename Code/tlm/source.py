"""Provides a class to use as sources for TLM.

This class can be provided with a callable that generates the amplitudes
to pipe into the TLM. Additionally coordinates can be given to place this
source somewhere else than the origin.
"""
from typing import Callable, Tuple

import numpy as np


class Source:
    def __init__(
        self,
        function: Callable,
        coordinates: Tuple[int, int] = (0, 0),  # x, y
    ):
        self._x = coordinates[0]
        self._y = coordinates[1]
        self._func = function

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def value(self, time: float) -> float:
        return self._func(time)
