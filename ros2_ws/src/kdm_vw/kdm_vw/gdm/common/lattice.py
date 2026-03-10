import copy
from dataclasses import dataclass

import numpy as np


@dataclass
class LatticeScalar:
    dimensions: int
    shape: tuple
    init_value: float = 0.0

    def __post_init__(self):
        self._data = np.full(self.shape, self.init_value, dtype=float)

    @classmethod
    def fromMatrix(cls, matrix):
        instance = cls(dimensions=len(matrix.shape), shape=matrix.shape)
        instance.loadMatrix(matrix)
        return instance

    def toMatrix(self):
        return copy.deepcopy(self._data)

    def loadMatrix(self, matrix):
        assert matrix.shape == self.shape
        self._data = np.array(matrix, dtype=float, copy=True)
        return self

    def getCell(self, cell):
        return float(self._data[cell])

    def setCell(self, cell, value):
        self._data[cell] = value
        return self

    def min(self):
        return float(np.min(self._data))

    def max(self):
        return float(np.max(self._data))
