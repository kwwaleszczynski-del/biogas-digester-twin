"""A synthetic agricultural co-digestion plant on ADM1 (maize silage + cattle slurry, ~0.5 MWel).

One mesophilic CSTR (main digester) with a membrane gas roof, fed once a day with a feed that the
operator logs. Inputs are piecewise constant per day; the plant state is integrated with the stiff
BDF solver and sampled every hour.

Agricultural parameter set (assumptions on top of BSM2, each stated in the README):

- 3000 m3 liquid, 600 m3 gas, 40 C set point, gas outlet sized for ~10 mbar over-pressure;
- first-order hydrolysis 0.25 1/d (the rate-limiting step for silage);
- kinetics scale exp(theta (T - 35 C)): theta 0.035 1/K for hydrolysis, acidogenesis and decay,
  0.069 1/K (rates halve per 10 K) for acetogenesis and methanogenesis;
- free-ammonia inhibition of acetoclastic methanogens for a culture adapted to ammonia:
  K_I = 0.02 kmol N/m3 (280 mg NH3-N/L), Hill exponent 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .adm1 import ADM1, BSM2, IDX, STATE_NAMES, Params, with_params
from .substrates import MAIZE_SILAGE, POULTRY_MANURE, SUBSTRATES, degraded, influent, odm_fed

AGRI = with_params(
    BSM2,
    V_liq=3000.0, V_gas=600.0, k_p=5e5,
    k_hyd_ch=0.25, k_hyd_pr=0.25, k_hyd_li=0.25,
    k_m_pro=1.2, k_m_ac=4.5,
    T_ref=308.15, theta_hyd=0.035, theta_acid=0.035, theta_acet=0.069, theta_meth=0.069,
    theta_dec=0.035,
)
T_SET = 313.15
NOMINAL_FEED = {"maize silage": 26.0, "cattle slurry": 30.0}
FEEDS = list(SUBSTRATES)
HOURS = np.arange(24) / 24.0


@dataclass
class Inputs:
    """Daily plant inputs (index = day number)."""

    feed: pd.DataFrame                 # t fresh matter per day, one column per substrate
    temperature: np.ndarray            # digester temperature [K], daily mean
    maize_quality: np.ndarray          # degradability factor of the maize silage fed that day
    dm_maize: np.ndarray               # dry matter content of the maize silage that day

    @property
    def days(self) -> int:
        return len(self.temperature)

    def copy(self) -> "Inputs":
        return Inputs(self.feed.copy(), self.temperature.copy(), self.maize_quality.copy(),
                      self.dm_maize.copy())


def nominal_inputs(days: int, seed: int | None = None, variability: bool = True) -> Inputs:
    """Normal operation: the operator feeds the nominal recipe with everyday variation.

    Variation (seeded): silage mass +-3 % day to day, slurry +-12 % (barn cleaning), silage dry
    matter drifting between clamps (31-35 %), digester temperature +-0.2 K around the set point, and
    every 6-17 weeks a harmless recipe change: 5-15 % of the silage oDM as poultry manure for 1-4
    weeks, 5-15 t/d of sugar beet for 1-2 days, or 8-20 % more silage for 2-6 weeks. These keep the
    yield within the normal band; they are there so that a method cannot call an upset from the
    feed log alone.
    """
    rng = np.random.default_rng(seed)
    feed = pd.DataFrame(0.0, index=np.arange(days), columns=FEEDS)
    if variability:
        feed["maize silage"] = NOMINAL_FEED["maize silage"] * (1 + 0.03 * rng.standard_normal(days))
        feed["cattle slurry"] = NOMINAL_FEED["cattle slurry"] * np.clip(
            1 + 0.12 * rng.standard_normal(days), 0.6, 1.4)
        dm = np.empty(days)
        level = MAIZE_SILAGE.dm + 0.01 * rng.standard_normal()
        for d in range(days):
            if d % 60 == 0:  # a new clamp section every two months
                level = np.clip(MAIZE_SILAGE.dm + 0.012 * rng.standard_normal(), 0.31, 0.35)
            dm[d] = level + 0.004 * rng.standard_normal()
        temp = T_SET + 0.2 * np.tanh(np.cumsum(0.05 * rng.standard_normal(days)))
        benign_changes(feed, dm, np.random.default_rng(None if seed is None else seed + 100_003))
    else:
        feed["maize silage"] = NOMINAL_FEED["maize silage"]
        feed["cattle slurry"] = NOMINAL_FEED["cattle slurry"]
        dm = np.full(days, MAIZE_SILAGE.dm)
        temp = np.full(days, T_SET)
    return Inputs(feed=feed, temperature=temp, maize_quality=np.ones(days), dm_maize=dm)


def benign_changes(feed: pd.DataFrame, dm: np.ndarray, rng: np.random.Generator) -> None:
    """Harmless recipe changes of everyday operation (in place)."""
    days = len(feed)
    d = int(rng.integers(20, 70))
    while d < days:
        kind = rng.choice(["poultry", "beet", "more"])
        if kind == "poultry":
            dur = int(rng.integers(7, 29))
            share = rng.uniform(0.05, 0.15)
            idx = np.arange(d, min(d + dur, days))
            maize_odm = feed.loc[idx, "maize silage"].to_numpy() * dm[idx] * MAIZE_SILAGE.odm
            feed.loc[idx, "maize silage"] *= 1 - share
            feed.loc[idx, "poultry manure"] += share * maize_odm / (POULTRY_MANURE.dm * POULTRY_MANURE.odm)
        elif kind == "beet":
            dur = int(rng.integers(1, 3))
            feed.loc[np.arange(d, min(d + dur, days)), "sugar beet"] += rng.uniform(5.0, 15.0)
        else:
            dur = int(rng.integers(14, 43))
            f = rng.uniform(1.08, 1.2)
            idx = np.arange(d, min(d + dur, days))
            ramp = np.clip((idx - d + 1) / 3.0, 0, 1)
            feed.loc[idx, "maize silage"] *= 1 + (f - 1) * ramp
        d += dur + int(rng.integers(40, 120))


def day_influent(inp: Inputs, d: int) -> tuple[float, np.ndarray]:
    subs = dict(SUBSTRATES)
    maize = degraded(MAIZE_SILAGE, inp.maize_quality[d])
    subs["maize silage"] = type(maize)(**{**maize.__dict__, "dm": float(inp.dm_maize[d])})
    feed = {k: float(v) for k, v in inp.feed.iloc[d].items() if v > 0}
    return influent(feed, subs)


def day_odm(inp: Inputs, d: int) -> float:
    subs = dict(SUBSTRATES)
    subs["maize silage"] = type(MAIZE_SILAGE)(**{**MAIZE_SILAGE.__dict__, "dm": float(inp.dm_maize[d])})
    return odm_fed({k: float(v) for k, v in inp.feed.iloc[d].items() if v > 0}, subs)


class Plant:
    def __init__(self, params: Params = AGRI):
        self.params = params
        self.model = ADM1(params)

    def steady_state(self, days: int = 600) -> np.ndarray:
        """Steady state under the nominal feed (constant, no variability)."""
        inp = nominal_inputs(1, variability=False)
        q, x_in = day_influent(inp, 0)
        x0 = np.concatenate([x_in * 0.05, [1e-5, 1.6, 0.012]])
        x0[IDX["X_I"]] = x_in[IDX["X_I"]]
        x0[16:23] = [1.0, 1.0, 0.5, 0.5, 0.3, 1.5, 0.6]
        x0[IDX["S_IC"]] = 0.2
        x0[IDX["S_IN"]] = x_in[IDX["S_IN"]] * 2
        x0[IDX["S_cat"]], x0[IDX["S_an"]] = x_in[IDX["S_cat"]], x_in[IDX["S_an"]]
        sol = solve_ivp(lambda t, x: self.model.rhs(t, x, q, x_in, T_SET), (0, days), x0,
                        method="BDF", rtol=1e-6, atol=1e-9)
        return sol.y[:, -1]

    def run(self, inp: Inputs, x0: np.ndarray, uninhibited: bool = False,
            lab_hour: int = 8) -> dict:
        """Simulate day by day; hourly outputs and the full state at the daily lab sampling hour."""
        m = self.model
        x = np.array(x0, dtype=float)
        n = inp.days
        hourly = np.empty((n * 24, 5))
        states = np.empty((n, len(STATE_NAMES)))
        ch4_day = np.empty(n)
        for d in range(n):
            q, x_in = day_influent(inp, d)
            T = float(inp.temperature[d])
            sol = solve_ivp(lambda t, y: m.rhs(t, y, q, x_in, T, uninhibited), (0.0, 1.0), x,
                            method="BDF", rtol=1e-5, atol=1e-8, t_eval=np.append(HOURS, 1.0))
            if not sol.success:
                raise RuntimeError(f"day {d}: {sol.message}")
            ys = np.maximum(sol.y, 0.0)
            for h in range(24):
                o = m.outputs(ys[:, h], T)
                hourly[d * 24 + h] = (o["q_biogas_N"] / 24.0, o["ch4_frac_dry"], o["pH"], T,
                                      o["q_ch4_N"] / 24.0)
            states[d] = ys[:, lab_hour]
            ch4_day[d] = hourly[d * 24:(d + 1) * 24, 4].sum()
            x = ys[:, -1]
        h = pd.DataFrame(hourly, columns=["biogas_Nm3h", "ch4_frac", "pH", "T_K", "ch4_Nm3h"])
        h["day"] = np.repeat(np.arange(n), 24)
        h["hour"] = np.tile(np.arange(24), n)
        return {"hourly": h, "states": pd.DataFrame(states, columns=STATE_NAMES),
                "ch4_day": ch4_day, "x_end": x}
