import copy
import math

import numpy as np

from .lattice import LatticeScalar


class DiscreteMap:
    def __init__(self, dimensions, size, resolution=1.0, offset=0):
        assert dimensions == len(size)
        assert resolution > 0.0

        if offset == 0:
            offset = tuple(np.zeros(dimensions).tolist())
        else:
            offset = tuple(offset)

        shape = np.zeros(dimensions)
        for axis in range(dimensions):
            if axis == 0:
                shape[1] = int(math.ceil(size[0] / resolution))
            elif axis == 1:
                shape[0] = int(math.ceil(size[1] / resolution))
            else:
                shape[axis] = int(math.ceil(size[axis] / resolution))

        self.dimensions = dimensions
        self.size = tuple(float(value) for value in size)
        self.resolution = float(resolution)
        self.offset = offset
        self.shape = tuple(shape.astype(int).tolist())

    def isPositionValid(self, position):
        if type(position) is tuple:
            assert len(position) == self.dimensions
            for axis in range(self.dimensions):
                minimum = 0.0 - self.offset[axis]
                maximum = self.size[axis] - self.offset[axis]
                assert minimum <= position[axis] <= maximum
        elif type(position) is list:
            for item in position:
                self.isPositionValid(item)
        return True

    def _clipPosition(self, position):
        if type(position) is tuple:
            clipped = []
            for axis in range(self.dimensions):
                minimum = 0.0 - self.offset[axis]
                maximum = self.size[axis] - self.offset[axis]
                clipped.append(max(min(position[axis], maximum), minimum))
            return tuple(clipped)
        return [self._clipPosition(item) for item in position]

    def _convertPositionToCell(self, position, fix_position=False):
        if fix_position:
            position = self._clipPosition(position)
        else:
            self.isPositionValid(position)

        if type(position) is tuple:
            shifted = np.array(position) + np.array(self.offset)
            fixed = np.zeros(self.dimensions)
            for axis in range(self.dimensions):
                value = shifted[axis]
                minimum = 0.0 + self.resolution / 10.0
                maximum = self.size[axis] - self.resolution / 10.0
                fixed[axis] = max(min(value, maximum), minimum)
            shifted = tuple(fixed.tolist())

            cell = np.zeros(self.dimensions, dtype=int)
            for axis in range(self.dimensions):
                if axis == 0:
                    cell[1] = int(np.floor(shifted[0] / self.resolution))
                elif axis == 1:
                    cell[0] = int(np.floor((self.size[1] - shifted[1]) / self.resolution))
                else:
                    cell[axis] = int(np.floor(shifted[axis] / self.resolution))

            for axis in range(self.dimensions):
                if cell[axis] >= self.shape[axis]:
                    cell[axis] = self.shape[axis] - 1
                if cell[axis] < 0:
                    cell[axis] = 0

            return tuple(cell.tolist())

        return [self._convertPositionToCell(item, fix_position=fix_position) for item in position]

    def _convertCellToPosition(self, cell):
        if type(cell) is tuple:
            position = np.zeros(self.dimensions)
            for axis in range(self.dimensions):
                if axis == 0:
                    position[1] = self.size[1] - (cell[0] + 0.5) * self.resolution
                elif axis == 1:
                    position[0] = (cell[1] + 0.5) * self.resolution
                else:
                    position[axis] = (cell[axis] + 0.5) * self.resolution
            position = np.round(position - np.array(self.offset), 6)
            return tuple(position.tolist())

        return [self._convertCellToPosition(item) for item in cell]


class DiscreteScalarMap(DiscreteMap, LatticeScalar):
    def __init__(self, dimensions, size, resolution=1.0, init_value=0.0, offset=0):
        DiscreteMap.__init__(self, dimensions=dimensions, size=size, resolution=resolution, offset=offset)
        LatticeScalar.__init__(self, dimensions=dimensions, shape=self.shape, init_value=init_value)

    @classmethod
    def fromMatrix(cls, matrix, resolution=1.0, offset=0):
        dimensions = len(matrix.shape)
        size = np.zeros(dimensions)
        for axis in range(dimensions):
            if axis == 0:
                size[1] = matrix.shape[0] * resolution
            elif axis == 1:
                size[0] = matrix.shape[1] * resolution
            else:
                size[axis] = matrix.shape[axis] * resolution
        instance = cls(dimensions=dimensions, size=tuple(size.tolist()), resolution=resolution, offset=offset)
        instance.loadMatrix(matrix)
        return instance

    def __getitem__(self, key):
        return DiscreteScalarMap.fromMatrix(self._data[key], resolution=self.resolution, offset=self.offset)

    def toMatrix(self):
        return copy.deepcopy(self._data)
