"""What the operator sees: online sensors with noise, drift and gaps, lab FOS/TAC, the feed log.

Online, hourly: biogas flow (Nm3/h, dry), CH4 content (%), pH, digester temperature.
Lab: FOS/TAC by the Nordmann titration twice a week (Monday, Thursday), ammonium nitrogen weekly.
Feed log: tonnes of each substrate per day (weighing error 2 %).

The FOS/TAC titration is simulated on the ADM1 acid-base state: a 20 mL sample is titrated with
0.1 N H2SO4 to pH 5.0 (volume A) and on to pH 4.4 (volume B); TAC = 250 A [mg CaCO3/L] and
FOS = (1.66 B - 0.15) 500 [mg acetic acid/L] (Nordmann 1977, as used in German and Polish plants).
CO2 is assumed to stay in the sample during the titration.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .adm1 import IDX, Constants
from .plant import Inputs, day_odm
from .scenarios import Scenario

T_LAB = 298.15
pKa_lab = Constants.at(T_LAB)


def _acid_added(pH: float, tot: dict, c: Constants) -> float:
    """Strong acid [eq/L] added to bring the sample to pH (charge balance; offsets cancel)."""
    h = 10.0 ** -pH
    nh4 = tot["N"] * h / (c.K_a_IN + h)
    hco3 = tot["C"] * c.K_a_co2 / (c.K_a_co2 + h)
    vfa = (tot["ac"] * c.K_a_ac / (c.K_a_ac + h) + tot["pro"] * c.K_a_pro / (c.K_a_pro + h)
           + tot["bu"] * c.K_a_bu / (c.K_a_bu + h) + tot["va"] * c.K_a_va / (c.K_a_va + h))
    return tot["sid"] + nh4 + h - hco3 - vfa - c.K_w / h


def titration(state: np.ndarray | pd.Series) -> dict:
    """Nordmann FOS/TAC of a sample taken from the digester state (exact, no lab error)."""
    s = np.asarray(state, dtype=float)
    tot = {
        "ac": s[IDX["S_ac"]] / 64.0, "pro": s[IDX["S_pro"]] / 112.0, "bu": s[IDX["S_bu"]] / 160.0,
        "va": s[IDX["S_va"]] / 208.0, "C": s[IDX["S_IC"]], "N": s[IDX["S_IN"]],
        "sid": s[IDX["S_cat"]] - s[IDX["S_an"]],
    }
    c = pKa_lab
    # sample pH at lab temperature: the root of the charge balance
    lo, hi = 2.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _acid_added(mid, tot, c) > 0:
            lo = mid
        else:
            hi = mid
    pH0 = 0.5 * (lo + hi)
    a5 = _acid_added(5.0, tot, c) - _acid_added(pH0, tot, c)
    a44 = _acid_added(4.4, tot, c) - _acid_added(pH0, tot, c)
    A = a5 * 0.020 / 1e-4         # mL of 0.1 N acid for 20 mL sample
    B = (a44 - a5) * 0.020 / 1e-4
    return {"A": A, "B": B, "pH_lab": pH0}


def fos_tac(A: float, B: float) -> tuple[float, float]:
    tac = 250.0 * A
    fos = (1.66 * B - 0.15) * 500.0
    return fos, tac


def vfa_mg_hac(state) -> float:
    """Total volatile fatty acids as acetic acid equivalent [mg/L] (what a GC would report)."""
    s = np.asarray(state, dtype=float)
    mol = (s[IDX["S_ac"]] / 64.0 + s[IDX["S_pro"]] / 112.0 + s[IDX["S_bu"]] / 160.0
           + s[IDX["S_va"]] / 208.0)
    return mol * 60.05e3


def online(res: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Hourly sensor readings with noise, drift, gaps and the injected sensor faults."""
    sc: Scenario = res["scenario"]
    h = res["hourly"]
    n = len(h)
    day = h["day"].to_numpy()
    t = day + h["hour"].to_numpy() / 24.0

    # biogas flow: 2 % noise, a fixed calibration factor of the meter
    flow = h["biogas_Nm3h"].to_numpy() * rng.uniform(0.98, 1.02) * (1 + 0.02 * rng.standard_normal(n))
    # CH4: 0.3 %-pt noise, slow drift re-zeroed at a monthly calibration
    drift = np.zeros(n)
    walk = np.cumsum(0.02 / math.sqrt(24) * rng.standard_normal(n))
    cal_start = (day // 30) * 30 * 24
    drift = walk - walk[np.minimum(cal_start, n - 1)]
    ch4 = 100 * h["ch4_frac"].to_numpy() + drift + 0.3 * rng.standard_normal(n)
    # pH electrode: 0.02 noise, slow drift, recalibrated monthly
    pwalk = np.cumsum(0.003 / math.sqrt(24) * rng.standard_normal(n))
    pdrift = pwalk - pwalk[np.minimum(cal_start, n - 1)]
    pH = h["pH"].to_numpy() + pdrift + 0.02 * rng.standard_normal(n)
    temp = h["T_K"].to_numpy() - 273.15 + 0.1 * rng.standard_normal(n)

    if sc.kind == "ch4_drift":
        on = t >= sc.t0
        ch4 = ch4 + np.where(on, sc.severity * (t - sc.t0), 0.0)
    if sc.kind == "flow_stuck":
        on = (t >= sc.t0) & (t < sc.t_end)
        i0 = int(np.argmax(on))
        flow = np.where(on, flow[max(i0 - 1, 0)], flow)

    df = pd.DataFrame({"day": day, "hour": h["hour"].to_numpy(), "flow": flow,
                       "ch4": np.clip(ch4, 0, 100), "pH": pH, "temp": temp})
    # communication gaps: a few outages of 2-36 h per quarter, all channels
    for _ in range(rng.poisson(n / 24 / 45)):
        start = int(rng.integers(0, n))
        df.loc[start:start + int(rng.integers(2, 37)), ["flow", "ch4", "pH", "temp"]] = np.nan
    # single-channel dropouts
    for col in ["flow", "ch4", "pH", "temp"]:
        mask = rng.random(n) < 0.01
        df.loc[mask, col] = np.nan
    return df


def lab(res: dict, rng: np.random.Generator, missing: float = 0.1) -> pd.DataFrame:
    """Lab results per day (NaN on days without a sample). Monday and Thursday FOS/TAC; Monday TAN.

    The random draws do not depend on `missing`, so a higher missing share removes samples from the
    same series (nested), which keeps the robustness comparison paired.
    """
    states = res["states"]
    n = len(states)
    rows = []
    for d in range(n):
        u, eA, eB, eT = rng.random(), rng.standard_normal(), rng.standard_normal(), rng.standard_normal()
        fos = tac = tan = np.nan
        if d % 7 in (0, 3) and u >= missing:
            tt = titration(states.iloc[d])
            fos, tac = fos_tac(tt["A"] + 0.1 * eA, tt["B"] + 0.05 * eB)
            if d % 7 == 0:
                tan = states.iloc[d]["S_IN"] * 14.0 * (1 + 0.05 * eT)
        rows.append((d, fos, tac, tan))
    return pd.DataFrame(rows, columns=["day", "fos", "tac", "tan"])


def feed_log(inp: Inputs, rng: np.random.Generator) -> pd.DataFrame:
    """Logged feed per substrate [t/d] (2 % weighing error) and the oDM the operator books."""
    f = inp.feed * (1 + 0.02 * rng.standard_normal(inp.feed.shape))
    f = f.clip(lower=0)
    out = f.add_prefix("feed_")
    # the operator books oDM with the nominal silage dry matter (measured only monthly)
    nominal = Inputs(feed=f, temperature=inp.temperature, maize_quality=inp.maize_quality,
                     dm_maize=np.full(inp.days, 0.33))
    out["odm_t"] = [day_odm(nominal, d) for d in range(inp.days)]
    out.insert(0, "day", np.arange(inp.days))
    return out


def daily(on: pd.DataFrame, labd: pd.DataFrame, feed: pd.DataFrame) -> pd.DataFrame:
    """Daily table the methods work on."""
    g = on.groupby("day")
    n_ok = g["flow"].count()
    d = pd.DataFrame({
        "biogas": g["flow"].mean() * 24.0,
        "ch4": g["ch4"].mean(),
        "pH": g["pH"].mean(),
        "temp": g["temp"].mean(),
        # spread of the hourly flow: a frozen meter repeats the same value
        "flow_sd": g["flow"].std() / g["flow"].mean(),
    })
    d.loc[n_ok < 6, ["biogas", "flow_sd"]] = np.nan
    d.loc[g["ch4"].count() < 6, "ch4"] = np.nan
    d.loc[g["pH"].count() < 6, "pH"] = np.nan
    d.loc[g["temp"].count() < 6, "temp"] = np.nan
    d = d.reset_index()
    d = d.merge(labd, on="day", how="left").merge(feed, on="day", how="left")
    d["ch4_Nm3"] = d["biogas"] * d["ch4"] / 100.0
    d["fos_tac"] = d["fos"] / d["tac"]
    return d
