# SUMO/TraCI execution protocol

Co-MAIMO uses the SUMO taxi device with external TraCI dispatch. SUMO manages reservations and pickup/drop-off routing after taxi dispatch; Co-MAIMO selects vehicle/request assignments.

Taxi tripinfo provides customers, occupied distance and occupied time. With the battery device configured, tripinfo also provides depleted events, actual battery capacity, total energy consumed and regenerated. Net electrical energy is reported as consumed minus regenerated energy.

Empty vehicle distance is derived as total vehicle route distance minus occupied taxi distance and is QC-checked before manuscript use.

The initial executable B1 dispatch selects an idle taxi using current SUMO route travel time. This is an integration milestone. The final B1 will use the common feasible sequential-insertion engine shared by B1-B6.
