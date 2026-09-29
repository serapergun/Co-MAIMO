# EMEF 2023 verification record

ATM lists 2023 public Displacements microdata and a Dictionary of variables and metadata. Institut Metropoli describes EMEF 2023 as weekday mobility of residents aged 16+ in the Barcelona integrated mobility area. The official questionnaire collects trip destination/location, start time, duration, purpose, transport mode, occupancy and other mode-dependent attributes.

## Open verification item
The exact public microdata column names and whether both trip origin and destination expose Barcelona district identifiers have not yet been verified from the public dictionary.

Therefore district_od_verified remains false. A 10x10 matrix must not be described as observed/empirically derived district OD until the official dictionary and displacement file pass schema audit. If only coarser geography is public, a documented spatial-disaggregation layer is required and its output must be labelled EMEF-calibrated, spatially disaggregated OD.
