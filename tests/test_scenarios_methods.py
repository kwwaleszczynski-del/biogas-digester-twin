"""Upsets, sensors, methods and scores on short runs (offline, a few seconds each)."""
import numpy as np
import pandas as pd
import pytest

from biogas_twin import evaluate as ev
from biogas_twin.am2 import EKF, ReducedModel, calibrate, feed_inputs
from biogas_twin.methods import PCAMonitor, YieldCharts, fostac_relative
from biogas_twin.plant import NOMINAL_FEED, T_SET, Plant
from biogas_twin.realdata import _dates, _feed
from biogas_twin.scenarios import Scenario, apply, draw, run, yield_loss_onset
from biogas_twin.sensors import daily, feed_log, lab, online


@pytest.fixture(scope="module")
def plant():
    pl = Plant()
    return pl, pl.steady_state()


def observe(res, seed=1, missing=0.1):
    return daily(online(res, np.random.default_rng(seed)), lab(res, np.random.default_rng(seed + 1), missing),
                 feed_log(res["inputs"], np.random.default_rng(seed + 2)))


def test_heating_failure_costs_yield_and_shows_on_the_temperature_probe(plant):
    pl, xs = plant
    sc = Scenario("temperature", seed=1, t0=20, days=40, severity=1.0, duration=8)
    res = run(sc, pl, xs)
    assert res["yield_loss_day"] is not None and 20 <= res["yield_loss_day"] < 30
    d = observe(res)
    assert d.loc[27, "temp"] < T_SET - 273.15 - 5


def test_sensor_faults_leave_the_process_alone(plant):
    pl, xs = plant
    sc = Scenario("flow_stuck", seed=2, t0=15, days=30, severity=1.0, duration=10)
    res = run(sc, pl, xs)
    assert res["yield_loss_day"] is None
    d = observe(res)
    assert (d.loc[16:24, "flow_sd"].dropna() == 0).all()
    assert (d.loc[2:13, "flow_sd"].dropna() > 0).all()
    sc2 = Scenario("ch4_drift", seed=2, t0=10, days=30, severity=0.3, duration=20)
    d2 = observe(run(sc2, pl, xs))
    assert d2["ch4"].iloc[25:29].mean() - d2["ch4"].iloc[5:9].mean() == pytest.approx(0.3 * 18, abs=1.5)


def test_scenario_draws_are_reproducible_and_inject():
    a, b = draw("ammonia", 3), draw("ammonia", 3)
    assert a == b
    from biogas_twin.plant import nominal_inputs
    base = nominal_inputs(a.days, seed=3)
    inj = apply(a, base)
    assert inj.feed["poultry manure"].iloc[: a.t0].sum() == 0
    assert inj.feed["poultry manure"].iloc[-1] > 0
    assert inj.feed["maize silage"].iloc[-1] < base.feed["maize silage"].iloc[-1]


def test_yield_loss_onset_uses_the_trailing_week():
    healthy = np.full(60, 100.0)
    ch4 = healthy.copy()
    ch4[30:] = 90.0
    assert yield_loss_onset(ch4, healthy, t0=30) == 33  # 7-day mean below 95 % after 4 days
    assert yield_loss_onset(healthy, healthy, t0=30) is None


def test_yield_charts_and_pca_detect_a_step_not_noise():
    rng = np.random.default_rng(0)
    n = 200
    def frame(shift):
        x = pd.DataFrame({"smy": 0.31 + 0.003 * rng.standard_normal(n), "gas_per_odm": 0.6 + 0.01 * rng.standard_normal(n),
                          "ch4_f": 53.5 + 0.2 * rng.standard_normal(n), "pH_f": 7.58 + 0.01 * rng.standard_normal(n),
                          "temp_f": 40 + 0.1 * rng.standard_normal(n)})
        x.loc[150:, "smy"] -= shift
        x.loc[150:, "ch4_f"] -= 100 * shift
        return x
    train = [frame(0.0) for _ in range(3)]
    ch = YieldCharts().fit(train)
    pca = PCAMonitor().fit(train)
    test = frame(0.03)
    assert ch.ewma(test)[:150].sum() <= 3
    assert ch.ewma(test)[152:].mean() > 0.9
    assert ch.cusum(test)[150:160].any()
    assert pca.alarm(test)[155:].mean() > 0.9


def test_relative_fos_tac_rule():
    days = np.arange(60)
    ft = np.full(60, np.nan)
    ft[::3] = 0.17
    ft[45] = 0.25
    x = pd.DataFrame({"day": days, "fos_tac": ft})
    x["fos_tac_last"] = x["fos_tac"].ffill()
    a = fostac_relative(x)
    assert not a[:45].any() and a[45:48].all() and not a[48:].any()


def test_scores():
    alarm = np.zeros(100, bool)
    alarm[[10, 11, 12, 30, 60, 61]] = True
    assert list(ev.onsets(alarm)) == [10, 30, 60]
    assert ev.episodes(alarm, 0, 100) == 3
    assert ev.episodes(alarm, 20, 100) == 2
    sc = Scenario("ammonia", seed=0, t0=50, days=100, severity=0.5, duration=50)
    r = ev.detection(alarm, None, sc, yield_loss_day=70)
    assert r["detected"] and r["delay"] == 10 and r["lead"] == 10 and r["before_loss"]
    cls = np.array(["normal"] * 100, dtype=object)
    cls[10:13] = "substrate"
    cls[55:58] = "ammonia"
    assert list(ev.class_onsets(cls)) == [10, 55]


def test_reduced_model_matches_adm1_and_ekf_stays_calm(plant):
    pl, xs = plant
    p, x0 = calibrate(pl, xs)
    rm = ReducedModel(p)
    u = feed_inputs(NOMINAL_FEED)
    y = rm.measure(x0, u, T_SET, p.tan0, lab=True)
    o = pl.model.outputs(xs, T_SET)
    assert y[0] == pytest.approx(o["q_biogas_N"], rel=0.01)
    assert y[1] == pytest.approx(100 * o["ch4_frac_dry"], abs=0.3)
    assert y[2] == pytest.approx(o["pH"], abs=0.02)
    res = run(draw("normal", 5, pre=30, post=0), pl, xs)
    est = EKF(rm, x0).run(observe(res))
    assert est["m"].abs().max() < 0.15
    assert est["b"].abs().max() < 1.0


def test_real_data_parsers():
    from datetime import datetime
    # spreadsheet wrote 10 Jan as 2024-10-01 and 7 Feb as 2024-07-02; strings are day/month/year
    got = _dates([datetime(2024, 10, 1), datetime(2024, 7, 2), "13/2/2024", datetime(2024, 2, 14)])
    assert got == [datetime(2024, 1, 10), datetime(2024, 2, 7), datetime(2024, 2, 13), datetime(2024, 2, 14)]
    assert _feed("20kg") == 20 and _feed("140kg") == 140 and _feed("nl") == 0 and _feed(" ") == 0
