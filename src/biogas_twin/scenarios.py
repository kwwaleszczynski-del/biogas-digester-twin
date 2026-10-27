"""Process upsets and sensor faults injected into normal operation, with ground truth.

Each scenario starts from the plant's nominal steady state, runs `pre` days of normal operation (the
methods' own history, also counted for false alarms) and then injects one upset at day t0.

Ground truth per scenario: the class, t0, the end of the cause, and the **yield-loss onset**: the first
day on which the true methane production, averaged over the trailing 7 days, falls more than 5 %
below what a
healthy twin produces from the same feed. The healthy twin is the same ADM1 plant without the upset
mechanism: no inhibition terms, the normal temperature, the nominal silage quality. Both are normalised
by their ratio over the 14 days before t0. Sensor faults have no yield loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .plant import T_SET, Inputs, Plant, nominal_inputs
from .substrates import MAIZE_SILAGE, POULTRY_MANURE

PROCESS = ["overload", "acidification", "ammonia", "temperature", "substrate"]
SENSOR = ["ch4_drift", "flow_stuck"]
CLASSES = ["normal"] + PROCESS + SENSOR
LABELS = {
    "normal": "normal",
    "overload": "organic overload",
    "acidification": "acidification (shock load)",
    "ammonia": "ammonia inhibition",
    "temperature": "heating failure",
    "substrate": "substrate change",
    "ch4_drift": "CH4 sensor drift",
    "flow_stuck": "stuck gas flow meter",
}
YIELD_LOSS = 0.05


@dataclass
class Scenario:
    kind: str
    seed: int
    t0: int
    days: int
    severity: float
    duration: int          # days the cause is active (fault ends at t0 + duration)
    params: dict = field(default_factory=dict)

    @property
    def t_end(self) -> int:
        return min(self.t0 + self.duration, self.days)

    @property
    def is_sensor(self) -> bool:
        return self.kind in SENSOR


def draw(kind: str, seed: int, pre: int = 90, post: int = 90) -> Scenario:
    """A random instance of one upset type (severity and timing drawn from the seed)."""
    rng = np.random.default_rng(seed * 7919 + CLASSES.index(kind))
    t0 = pre + int(rng.integers(0, 7))
    days = pre + post
    if kind == "overload":
        sev = float(rng.uniform(1.4, 2.0))            # silage multiplier
        return Scenario(kind, seed, t0, days, sev, days - t0, {"ramp": int(rng.integers(1, 8))})
    if kind == "acidification":
        sev = float(rng.uniform(40.0, 90.0))          # extra sugar beet t/d
        return Scenario(kind, seed, t0, days, sev, int(rng.integers(2, 5)))
    if kind == "ammonia":
        sev = float(rng.uniform(0.4, 0.9))            # share of the silage oDM replaced by poultry manure
        return Scenario(kind, seed, t0, days, sev, days - t0, {"ramp": int(rng.integers(1, 15))})
    if kind == "temperature":
        sev = float(rng.uniform(0.5, 1.0))            # cooling K/d
        return Scenario(kind, seed, t0, days, sev, int(rng.integers(5, 15)))
    if kind == "substrate":
        sev = float(rng.uniform(0.65, 0.88))          # degradability factor of the new silage
        return Scenario(kind, seed, t0, days, sev, days - t0)
    if kind == "ch4_drift":
        sev = float(rng.uniform(0.08, 0.3)) * float(rng.choice([-1.0, 1.0]))  # %-points per day
        return Scenario(kind, seed, t0, days, sev, days - t0)
    if kind == "flow_stuck":
        sev = 1.0
        return Scenario(kind, seed, t0, days, sev, int(rng.integers(7, 45)))
    if kind == "normal":
        return Scenario(kind, seed, days, days, 0.0, 0)
    raise ValueError(kind)


def apply(sc: Scenario, inp: Inputs) -> Inputs:
    """Inputs with the process upset injected (sensor faults leave the process untouched)."""
    out = inp.copy()
    d = np.arange(out.days)
    act = (d >= sc.t0) & (d < sc.t_end)
    if sc.kind == "overload":
        ramp = np.clip((d - sc.t0 + 1) / sc.params["ramp"], 0, 1)
        f = np.where(act, 1 + (sc.severity - 1) * ramp, 1.0)
        out.feed["maize silage"] *= f
    elif sc.kind == "acidification":
        out.feed.loc[act, "sugar beet"] += sc.severity
    elif sc.kind == "ammonia":
        ramp = np.clip((d - sc.t0 + 1) / sc.params["ramp"], 0, 1)
        share = np.where(act, sc.severity * ramp, 0.0)
        maize_odm = out.feed["maize silage"] * out.dm_maize * MAIZE_SILAGE.odm
        out.feed["maize silage"] *= 1 - share
        out.feed["poultry manure"] += share * maize_odm / (POULTRY_MANURE.dm * POULTRY_MANURE.odm)
    elif sc.kind == "temperature":
        temp = out.temperature.copy()
        cool = sc.severity
        level = None
        for i in range(sc.t0, out.days):
            if i < sc.t_end:
                level = (temp[i - 1] if level is None else level) - cool
                temp[i] = level
            else:
                level = min(level + 0.5, T_SET) if level is not None else temp[i]
                if level >= temp[i]:
                    break
                temp[i] = level
        out.temperature = temp
    elif sc.kind == "substrate":
        out.maize_quality = np.where(d >= sc.t0, sc.severity, 1.0)
    return out


def healthy_inputs(sc: Scenario, base: Inputs, applied: Inputs) -> Inputs:
    """Twin inputs: the same feed, but the normal temperature and the nominal silage quality."""
    out = applied.copy()
    out.temperature = base.temperature.copy()
    out.maize_quality = np.ones(out.days)
    return out


def yield_loss_onset(ch4: np.ndarray, ch4_healthy: np.ndarray, t0: int,
                     threshold: float = YIELD_LOSS) -> int | None:
    ratio = ch4 / np.maximum(ch4_healthy, 1e-9)
    base = ratio[max(t0 - 14, 0):t0].mean()
    r7 = np.convolve(ratio / base, np.ones(7) / 7, mode="full")[:len(ratio)]
    for d in range(t0, len(ratio)):
        if r7[d] < 1 - threshold:
            return d
    return None


def run(sc: Scenario, plant: Plant, x0: np.ndarray, base: Inputs | None = None) -> dict:
    """Simulate a scenario and its healthy twin; return true trajectories and ground truth."""
    base = base if base is not None else nominal_inputs(sc.days, seed=sc.seed)
    inp = apply(sc, base)
    res = plant.run(inp, x0)
    onset = None
    if sc.kind in PROCESS:
        twin = plant.run(healthy_inputs(sc, base, inp), x0, uninhibited=True)
        res["ch4_healthy"] = twin["ch4_day"]
        onset = yield_loss_onset(res["ch4_day"], twin["ch4_day"], sc.t0)
    else:
        res["ch4_healthy"] = res["ch4_day"].copy()
    res["inputs"] = inp
    res["scenario"] = sc
    res["yield_loss_day"] = onset
    return res
