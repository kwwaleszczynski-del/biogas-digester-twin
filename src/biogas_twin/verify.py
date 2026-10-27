"""ADM1 against the published BSM2 steady state (Rosen & Jeppsson 2006)."""
from __future__ import annotations

import numpy as np

from .adm1 import (ADM1, BSM2_INFLUENT, BSM2_Q, BSM2_STEADY_STATE, BSM2_T, STATE_NAMES,
                   simulate_constant)

OUTPUTS = ["pH", "S_hco3_ion", "S_nh3", "S_ac_ion", "q_gas", "p_gas_ch4", "p_gas_co2"]


def bsm2_run(days: float = 400.0) -> tuple[np.ndarray, dict]:
    """Start far from the reference (biomass halved, solubles doubled, less inert) and integrate the
    constant BSM2 influent until the state settles."""
    ref = BSM2_STEADY_STATE
    x0 = np.array([ref[n] for n in STATE_NAMES], dtype=float)
    x0[16:23] *= 0.5
    x0[:12] *= 2.0
    x0[23] = 15.0
    m = ADM1()
    sol = simulate_constant(m, x0, BSM2_Q, BSM2_INFLUENT, BSM2_T, days)
    x = sol.y[:, -1]
    return x, m.outputs(x, BSM2_T)


def deviations(x: np.ndarray, out: dict) -> dict[str, float]:
    ref = BSM2_STEADY_STATE
    dev = {n: abs(x[i] - ref[n]) / abs(ref[n]) for i, n in enumerate(STATE_NAMES) if ref.get(n)}
    dev.update({k: abs(out[k] - ref[k]) / abs(ref[k]) for k in OUTPUTS})
    return dev


def bsm2_check() -> float:
    x, out = bsm2_run()
    return max(deviations(x, out).values())
