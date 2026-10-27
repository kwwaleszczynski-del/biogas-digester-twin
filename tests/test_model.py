"""The simulator: ADM1 against the published BSM2 steady state, the agricultural plant, the lab model."""
import numpy as np
import pytest

from biogas_twin.adm1 import BSM2_STEADY_STATE, IDX
from biogas_twin.plant import T_SET, Plant
from biogas_twin.sensors import fos_tac, titration, vfa_mg_hac
from biogas_twin.substrates import CATTLE_SLURRY, MAIZE_SILAGE
from biogas_twin.verify import bsm2_run, deviations


@pytest.fixture(scope="module")
def plant():
    pl = Plant()
    return pl, pl.steady_state()


def test_adm1_reproduces_bsm2_steady_state():
    x, out = bsm2_run()
    dev = deviations(x, out)
    # all 26 liquid states, the gas phase, pH, ions and gas flow within 1e-5 of the published values
    assert max(dev.values()) < 1e-5, sorted(dev.items(), key=lambda kv: -kv[1])[:3]
    assert out["pH"] == pytest.approx(BSM2_STEADY_STATE["pH"], abs=1e-4)
    assert out["q_gas"] == pytest.approx(2955.70, rel=1e-5)


def test_plant_operates_like_a_maize_slurry_plant(plant):
    pl, xs = plant
    o = pl.model.outputs(xs, T_SET)
    assert 7.3 < o["pH"] < 7.9
    assert 0.50 < o["ch4_frac_dry"] < 0.56
    # about 0.5 MWel at 40 % electrical efficiency (9.97 kWh per Nm3 CH4)
    assert 450 < o["q_ch4_N"] * 9.97 * 0.40 / 24 < 650
    assert 1.5 < xs[IDX["S_IN"]] * 14 < 3.5          # TAN g N/L
    assert 300 < vfa_mg_hac(xs) < 1500                 # VFA as acetic acid mg/L


def test_specific_methane_yield_in_published_range(plant):
    """Methane per oDM of the mix sits between the slurry and the silage standard values."""
    pl, xs = plant
    o = pl.model.outputs(xs, T_SET)
    odm = 26 * MAIZE_SILAGE.odm_per_t() + 30 * CATTLE_SLURRY.odm_per_t()  # kg/d
    smy = o["q_ch4_N"] / odm
    assert 0.27 < smy < 0.35


def test_fos_tac_of_the_plant_and_of_pure_buffers(plant):
    pl, xs = plant
    tt = titration(xs)
    fos, tac = fos_tac(tt["A"], tt["B"])
    assert 8000 < tac < 15000 and 0.1 < fos / tac < 0.3
    # a bicarbonate-only sample: TAC is its alkalinity, FOS the known bicarbonate interference
    s = np.zeros(29)
    s[IDX["S_IC"]] = 0.25
    s[IDX["S_cat"]] = 0.2
    tt0 = titration(s)
    fos0, tac0 = fos_tac(tt0["A"], tt0["B"])
    assert tac0 == pytest.approx(0.2 * 50_000 * 0.95, rel=0.05)
    # adding 1 g/L acetic acid (as acetate) raises FOS and lowers the sample pH
    s[IDX["S_ac"]] = 64.0 / 60.05
    s[IDX["S_cat"]] += 1.0 / 60.05
    tt1 = titration(s)
    fos1, _ = fos_tac(tt1["A"], tt1["B"])
    assert 500 < fos1 - fos0 < 1500
