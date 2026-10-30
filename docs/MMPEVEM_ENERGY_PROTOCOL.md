# Physical MMPEVEM energy protocol

This pipeline accepts only outputs from an executed SUMO simulation. Former
normalized energy-proxy values must never be converted to physical units.

For each run, the vehicle type must explicitly use MMPEVEM and the SUMO battery
device. Tripinfo is parsed for route length, total electrical energy consumed,
and total regenerated energy.

Primary manuscript metric:

net energy [kWh] = (sum consumed Wh - sum regenerated Wh) / 1000

energy intensity [kWh/vehicle-km] = net energy / fleet vehicle-km

A run is rejected for physical-energy reporting if a required field is absent,
an energy or distance record is invalid, total vehicle distance is non-positive,
or net energy is invalid.

B1-B6 are comparable only under the same accepted Barcelona network, demand
realization, initial fleet state, EV parameterization, SUMO version, and paired
test seed. The physical-energy result remains pending until an actual SUMO run
passes these checks.
