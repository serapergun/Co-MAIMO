"""Battery feasibility guard used identically across B1-B6."""
from dataclasses import dataclass

@dataclass
class BEVState:
    capacity_kwh: float
    energy_kwh: float
    reserve_soc: float

    @property
    def soc(self):
        if self.capacity_kwh <= 0:
            raise ValueError("Battery capacity must be positive.")
        return self.energy_kwh / self.capacity_kwh

    def feasible(self, estimated_required_kwh):
        reserve = self.reserve_soc * self.capacity_kwh
        return estimated_required_kwh + reserve <= self.energy_kwh

    def validate(self):
        if not 0 <= self.reserve_soc <= 1:
            raise ValueError("reserve_soc must be in [0,1].")
        if not 0 <= self.energy_kwh <= self.capacity_kwh:
            raise ValueError("energy_kwh outside battery bounds.")
