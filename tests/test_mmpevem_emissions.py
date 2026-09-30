from pathlib import Path
import pytest
from src.evaluation.parse_mmpevem_emissions import mmpevem_emission_energy

def test_signed_mmpevem_energy(tmp_path: Path):
    p=tmp_path/"emissions.xml"
    p.write_text("""<emission-export><timestep time="0"><vehicle id="v" electricity="100"/></timestep><timestep time="1"><vehicle id="v" electricity="-20"/></timestep></emission-export>""")
    m=mmpevem_emission_energy(p)
    assert m["gross_energy_kWh"] == pytest.approx(.1)
    assert m["regenerated_energy_kWh"] == pytest.approx(.02)
    assert m["net_energy_kWh"] == pytest.approx(.08)

def test_nan_rejected(tmp_path: Path):
    p=tmp_path/"emissions.xml"
    p.write_text("""<emission-export><timestep time="0"><vehicle id="v" electricity="nan"/></timestep></emission-export>""")
    with pytest.raises(ValueError):
        mmpevem_emission_energy(p)
