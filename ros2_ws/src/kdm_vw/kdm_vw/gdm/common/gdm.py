from abc import abstractmethod

from .map import DiscreteScalarMap
from .observation import Observation


class DistributionMapper:
    def __init__(self, dimensions, size):
        self.dimensions = dimensions
        self.size = size
        self._observations = []
        self._estimate_valid = False
        self._uncertainty_valid = False

    def addObservation(self, observation):
        if isinstance(observation, list):
            self._observations.extend(observation)
        elif isinstance(observation, Observation):
            self._observations.append(observation)
        else:
            return self

        self._estimate_valid = False
        self._uncertainty_valid = False
        return self

    def estimate(self):
        if not self._estimate_valid and self._observations:
            self._estimate()
            self._estimate_valid = True
        return self

    @abstractmethod
    def _estimate(self):
        raise NotImplementedError


class NormalGasDistributionMapper(DistributionMapper):
    def __init__(self, dimensions, size, resolution, offset=0):
        super().__init__(dimensions=dimensions, size=size)
        self.resolution = resolution
        self.offset = offset if offset != 0 else tuple(0.0 for _ in range(dimensions))
        self.shape = DiscreteScalarMap(dimensions, size, resolution, offset=self.offset).shape
        self._domain = DiscreteScalarMap(dimensions, size, resolution, offset=self.offset)
        self._gas = DiscreteScalarMap(dimensions, size, resolution, offset=self.offset)
        self._gas_uncertainty = DiscreteScalarMap(dimensions, size, resolution, init_value=float("inf"), offset=self.offset)

    def _convertPositionToCell(self, position):
        return self._domain._convertPositionToCell(position, fix_position=True)

    def _convertCellToPosition(self, cell):
        return self._domain._convertCellToPosition(cell)

    def getGasEstimate(self):
        self.estimate()
        return self._getGasEstimate()

    def getGasUncertainty(self):
        self.estimate()
        if not self._uncertainty_valid and self._observations:
            self._computeUncertainty()
            self._uncertainty_valid = True
        return self._getGasUncertainty()

    def _getGasEstimate(self):
        return self._gas

    def _getGasUncertainty(self):
        return self._gas_uncertainty

    @abstractmethod
    def _computeUncertainty(self):
        raise NotImplementedError
