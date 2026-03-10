import numpy as np
from scipy.stats import multivariate_normal

from ..common import DiscreteScalarMap, NormalGasDistributionMapper, Observation


DEFAULT_KERNEL_STD = 0.38
DEFAULT_SCALING_STD = 4.956
DEFAULT_WIND_STRECH = 0.33


def wind_covariance(std, wind_speed, wind_direction, stretch):
    assert std > 0.0
    a_axis = std + stretch * wind_speed
    b_axis = (std * std) / (std + stretch * wind_speed)
    covariance = np.array(((a_axis * a_axis, 0.0), (0.0, b_axis * b_axis)))

    if wind_speed > 0.01:
        rotation = np.array(
            (
                (np.cos(wind_direction), -np.sin(wind_direction)),
                (np.sin(wind_direction), np.cos(wind_direction)),
            )
        )
        rotated_inverse = np.linalg.inv(rotation).dot(np.linalg.inv(covariance).dot(rotation))
        covariance = np.linalg.inv(rotated_inverse)

    return covariance


class KDM_VW(NormalGasDistributionMapper):
    def __init__(
        self,
        domain_map,
        kernel_std=DEFAULT_KERNEL_STD,
        scaling_std=DEFAULT_SCALING_STD,
        wind_strech=DEFAULT_WIND_STRECH,
    ):
        assert kernel_std > 0.0
        assert scaling_std > 0.0
        assert wind_strech > 0.0

        super().__init__(
            dimensions=2,
            size=domain_map.size,
            resolution=domain_map.resolution,
            offset=domain_map.offset,
        )

        self._k_s = kernel_std
        self._o_s = scaling_std
        self._stretch = wind_strech
        self._boundary = 10

        self._omega = DiscreteScalarMap(dimensions=2, size=self.size, resolution=self.resolution, offset=self.offset)
        self._R = DiscreteScalarMap(dimensions=2, size=self.size, resolution=self.resolution, offset=self.offset)
        self._alpha = DiscreteScalarMap(dimensions=2, size=self.size, resolution=self.resolution, offset=self.offset)

    def _estimate(self):
        omega = np.zeros(self.shape)
        weighted_sum = np.zeros(self.shape)
        total_gas = 0.0
        num_samples = 0

        for sample in self._observations:
            if not sample.hasGas():
                continue

            total_gas += sample.gas
            num_samples += 1

            wind = sample.wind if sample.hasWind() else (0.0, 0.0)
            wind_speed = float(np.hypot(wind[0], wind[1]))
            wind_direction = float(np.arctan2(wind[1], wind[0]))
            covariance = wind_covariance(self._k_s, wind_speed, wind_direction, self._stretch)

            sample_cell = self._convertPositionToCell(sample.position)
            padding = self._boundary
            i_scale = (padding * max(np.sqrt(abs(covariance[0, 0])), np.sqrt(abs(covariance[1, 0])))) / self._k_s
            j_scale = (padding * max(np.sqrt(abs(covariance[0, 1])), np.sqrt(abs(covariance[1, 1])))) / self._k_s
            start_i = max(0, int(sample_cell[0] - i_scale * self._k_s / self.resolution))
            start_j = max(0, int(sample_cell[1] - j_scale * self._k_s / self.resolution))
            end_i = min(self.shape[0], int(sample_cell[0] + 1 + i_scale * self._k_s / self.resolution) + 1)
            end_j = min(self.shape[1], int(sample_cell[1] + 1 + j_scale * self._k_s / self.resolution) + 1)

            index_i = [index for index in range(start_i, end_i)]
            index_j = [index for index in range(start_j, end_j)]
            cells = np.array(np.meshgrid(index_i, index_j)).T.reshape(-1, 2)
            distances = self.resolution * (cells - sample_cell)

            local_pdf = multivariate_normal.pdf(distances, cov=covariance).reshape(end_i - start_i, end_j - start_j)
            omega[start_i:end_i, start_j:end_j] += local_pdf
            weighted_sum[start_i:end_i, start_j:end_j] += local_pdf * sample.gas

        average_gas = total_gas / (num_samples + 1e-4)
        alpha = 1.0 - np.exp(-(omega ** 2) / (self._o_s ** 2))
        gas = alpha * (weighted_sum / (omega + 1e-4)) + (1.0 - alpha) * average_gas

        self._gas.loadMatrix(gas)
        self._R.loadMatrix(weighted_sum)
        self._omega.loadMatrix(omega)
        self._alpha.loadMatrix(alpha)
        self._uncertainty_valid = False
        return self

    def _computeUncertainty(self):
        alpha = self._alpha.toMatrix()
        uncertainty = 1.0 / (alpha + 1e-5) - 1.0
        uncertainty[uncertainty < 1e-5] = 1e-5
        self._gas_uncertainty.loadMatrix(uncertainty)
        return self

    def _getCell(self, cell):
        gas = self.getGasEstimate().getCell(cell)
        position = self._convertCellToPosition(cell)
        return Observation(position=position, gas=max(0.0, gas), data_type="gas")
