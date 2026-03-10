from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Observation:
    position: Tuple[float, float]
    gas: Optional[float] = None
    wind: Optional[Tuple[float, float]] = None
    time: float = 0.0
    data_type: str = "gas+wind"

    def hasGas(self) -> bool:
        return self.gas is not None

    def hasWind(self) -> bool:
        return self.wind is not None
