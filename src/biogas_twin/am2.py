"""State estimation: an extended Kalman filter on a reduced AM2-type model.

The reduced model follows the two-step structure of AM2 (Bernard et al. 2001, Biotechnol. Bioeng.
75, 424-438) with a hydrolysis step in front, as in AM2 extensions for particulate substrates:

    particulate COD XT --hydrolysis--> (acidogenesis, taken as instantaneous) --> VFA S2 + H2
    VFA S2 --methanogens X2--> CH4 + CO2

States [XT, S2, X2, C, Z] (kg COD/m3; inorganic carbon C and alkalinity Z in kmol/m3), augmented with
three slowly varying unknowns the filter estimates:

    m  log activity of the methanogenic step (inhibition by ammonia, pH, cold, overload: m < 0)
    s  degradability factor of the feed relative to the nominal recipe (substrate change: s < 1)
    b  bias of the CH4 analyser [%-points], with its drift rate beta [%-points/d] (sensor drift)

The parameters come from lumping the ADM1 plant at its nominal steady state (no fitting to the test
scenarios). Measurements: daily biogas, CH4 %, pH; FOS and TAC on lab days, through the same Nordmann
titration model applied to the reduced state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .adm1 import IDX, Constants
from .plant import NOMINAL_FEED, T_SET, Plant
from .sensors import _acid_added, fos_tac
from .substrates import SUBSTRATES, influent

N_X = 5
N_A = 9
NM3_PER_KMOL = 22.414


@dataclass
class RMParams:
    V: float
    k_h: float
    Y1: float
    f_h2: float
    Y_h2: float
    k2: float
    K2: float
    Y2: float
    kd: float
    c1: float
    c2: float
    n_N: float
    kLa: float
    theta_h: float = 0.035
    theta_m: float = 0.069
    T_ref: float = 308.15
    cod_per_kmol_vfa: float = 70.0
    tan0: float = 0.168
    ret: float = 0.0         # decayed biomass returning to XT, per unit X2 (ADM1: via X_c)
    kH_factor: float = 1.0   # effective CO2 solubility, matches the ADM1 gas composition
    pH_offset: float = 0.0   # the lumped charge balance misses minor ions; matched to ADM1


def lump(x: np.ndarray) -> np.ndarray:
    return np.array([
        x[IDX["X_ch"]] + x[IDX["X_pr"]] + x[IDX["X_li"]],
        x[IDX["S_va"]] + x[IDX["S_bu"]] + x[IDX["S_pro"]] + x[IDX["S_ac"]],
        x[IDX["X_c4"]] + x[IDX["X_pro"]] + x[IDX["X_ac"]],
        x[IDX["S_IC"]],
        x[IDX["S_cat"]] - x[IDX["S_an"]] + x[IDX["S_IN"]],
    ])


def feed_inputs(feed: dict[str, float]) -> dict:
    """Reduced-model inputs from a logged feed, with the nominal substrate composition."""
    q, xi = influent({k: v for k, v in feed.items() if v > 0}, SUBSTRATES)
    return {
        "q": q,
        "XT": xi[IDX["X_ch"]] + xi[IDX["X_pr"]] + xi[IDX["X_li"]],
        "S1": xi[IDX["S_su"]] + xi[IDX["S_aa"]] + xi[IDX["S_fa"]],
        "S2": xi[IDX["S_va"]] + xi[IDX["S_bu"]] + xi[IDX["S_pro"]] + xi[IDX["S_ac"]],
        "C": xi[IDX["S_IC"]],
        "Z": xi[IDX["S_cat"]] - xi[IDX["S_an"]] + xi[IDX["S_IN"]],
    }


class ReducedModel:
    def __init__(self, p: RMParams):
        self.p = p
        self._consts: dict[float, Constants] = {}

    def const(self, T: float) -> Constants:
        c = self._consts.get(T)
        if c is None:
            c = self._consts[T] = Constants.at(T)
        return c

    def factors(self, T: float) -> tuple[float, float]:
        d = T - self.p.T_ref
        return math.exp(self.p.theta_h * d), math.exp(self.p.theta_m * d)

    def chem(self, xs: np.ndarray, T: float, tan: float) -> tuple[float, float, float]:
        """(pH, HCO3-, dissolved CO2) from C, Z, S2 by the charge balance
        Z - NH3(pH) - VFA- - HCO3-(pH) = 0, solved by bisection on pH."""
        p = self.p
        c = self.const(T)
        XT, S2, X2, C, Z = xs[:N_X]
        vfa = S2 / p.cod_per_kmol_vfa
        C = max(C, 1e-6)
        ka_n, ka_c = c.K_a_IN, c.K_a_co2
        lo, hi = 4.0, 10.5
        pH = 7.5
        for _ in range(30):  # Newton on pH, kept inside the bisection bracket
            h = 10.0 ** -pH
            f = Z - tan * ka_n / (ka_n + h) - vfa - C * ka_c / (ka_c + h)
            if f > 0:
                lo = pH
            else:
                hi = pH
            # df/dpH = ln10 * h * (tan ka_n/(ka_n+h)^2 + C ka_c/(ka_c+h)^2) * (-1) * (-1)
            df = -math.log(10) * h * (tan * ka_n / (ka_n + h) ** 2 + C * ka_c / (ka_c + h) ** 2)
            new = pH - f / df if df != 0 else 0.5 * (lo + hi)
            if not lo < new < hi:
                new = 0.5 * (lo + hi)
            if abs(new - pH) < 1e-9:
                pH = new
                break
            pH = new
        lo = hi = pH
        pH = 0.5 * (lo + hi)
        h = 10.0 ** -pH
        hco3 = C * c.K_a_co2 / (c.K_a_co2 + h)
        return pH + p.pH_offset, hco3, C - hco3

    def rates(self, xa: np.ndarray, u: dict, T: float, tan: float) -> dict:
        p = self.p
        XT, S2, X2, C, Z = np.maximum(xa[:N_X], 1e-9)
        m, s = xa[5], xa[6]
        fh, fm = self.factors(T)
        D = u["q"] / p.V
        pH, hco3, co2 = self.chem(xa, T, tan)
        sh = 10 ** -pH
        lim = 10 ** -6.5
        I_pH = lim ** 3 / (sh ** 3 + lim ** 3)
        r_h = p.k_h * fh * XT
        r_ac = (1 - p.Y1) * (r_h + D * u["S1"])
        r2 = math.exp(m) * p.k2 * fm * S2 / (p.K2 + S2) * X2 * I_pH
        ch4_cod = p.f_h2 * (1 - p.Y_h2) * r_ac + (1 - p.Y2) * r2
        F_ch4 = ch4_cod * p.V / 64.0                     # kmol/d
        c = self.const(T)
        a = p.kLa * p.V
        Pd = 1.01 - c.p_h2o
        b = F_ch4 - a * co2 + a * p.kH_factor * c.K_H_co2 * Pd
        F_co2 = (-b + math.sqrt(b * b + 4 * a * co2 * F_ch4)) / 2.0
        return {"D": D, "r_h": r_h, "r_ac": r_ac, "r2": r2, "F_ch4": F_ch4, "F_co2": F_co2,
                "pH": pH, "s": s}

    def deriv(self, xa: np.ndarray, u: dict, T: float, tan: float) -> np.ndarray:
        p = self.p
        r = self.rates(xa, u, T, tan)
        XT, S2, X2, C, Z = xa[:N_X]
        D = r["D"]
        d = np.zeros_like(xa)
        d[0] = D * (r["s"] * u["XT"] - XT) - r["r_h"] + p.ret * X2
        d[1] = D * (u["S2"] - S2) + (1 - p.f_h2) * r["r_ac"] - r["r2"]
        d[2] = p.Y2 * r["r2"] - (D + p.kd) * X2
        d[3] = D * (u["C"] - C) + p.c1 * r["r_ac"] + p.c2 * r["r2"] - r["F_co2"] / p.V
        d[4] = D * (u["Z"] - Z) + p.n_N * r["r_h"]
        return d

    def step(self, xa: np.ndarray, u: dict, T: float, tan: float, n_sub: int = 12) -> np.ndarray:
        h = 1.0 / n_sub
        x = xa.copy()
        for _ in range(n_sub):
            k1 = self.deriv(x, u, T, tan)
            k2 = self.deriv(x + 0.5 * h * k1, u, T, tan)
            k3 = self.deriv(x + 0.5 * h * k2, u, T, tan)
            k4 = self.deriv(x + h * k3, u, T, tan)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x[:N_X] = np.maximum(x[:N_X], 1e-6)
        x[7] += x[8]  # analyser bias drifts at rate beta
        return x

    def measure(self, xa: np.ndarray, u: dict, T: float, tan: float, lab: bool) -> np.ndarray:
        r = self.rates(xa, u, T, tan)
        F = r["F_ch4"] + r["F_co2"]
        y = [F * NM3_PER_KMOL, 100 * r["F_ch4"] / F + xa[7], r["pH"]]
        if lab:
            y.extend(self.lab(xa, tan))
        return np.array(y)

    def lab(self, xa: np.ndarray, tan: float) -> tuple[float, float]:
        """FOS and TAC of the reduced state through the Nordmann titration model."""
        XT, S2, X2, C, Z = xa[:N_X]
        tot = {"ac": 0.8 * S2 / 64.0, "pro": 0.2 * S2 / 112.0, "bu": 0.0, "va": 0.0,
               "C": C, "N": tan, "sid": Z - tan}
        c = Constants.at(298.15)
        lo, hi = 2.0, 12.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _acid_added(mid, tot, c) > 0:
                lo = mid
            else:
                hi = mid
        a0 = _acid_added(0.5 * (lo + hi), tot, c)
        A = (_acid_added(5.0, tot, c) - a0) * 200.0
        B = (_acid_added(4.4, tot, c) - _acid_added(5.0, tot, c)) * 200.0
        return fos_tac(A, B)


def calibrate(plant: Plant, xs: np.ndarray) -> tuple[RMParams, np.ndarray]:
    """Reduced-model parameters by lumping the ADM1 plant at its nominal steady state."""
    ap = plant.params
    T = T_SET
    o = plant.model.outputs(xs, T)
    u = feed_inputs(NOMINAL_FEED)
    D = u["q"] / ap.V_liq
    lx = lump(xs)
    XT, S2, X2, C, Z = lx
    fh = math.exp(0.035 * (T - ap.T_ref))
    fm = math.exp(0.069 * (T - ap.T_ref))
    kd = ap.k_dec * math.exp(ap.theta_dec * (T - ap.T_ref))
    r_h = ap.k_hyd_ch * fh * XT
    ch4_cod = o["q_ch4_N"] / 0.35 / ap.V_liq             # kg COD/m3/d to methane (0.35 Nm3/kg COD)
    Y_h2 = ap.Y_h2
    f_h2 = 0.15                                            # COD share of acidogenesis going to H2 (ADM1)
    # methanogenic yield from the X2 balance, acidogenic yield Y1 so that the methane balance closes:
    # ch4 = (1-Y1) h [f (1-Yh2) + (1-f)(1-Y2)] + (1-Y2) D (S2in - S2),  h = r_h + D S1in
    hsrc = r_h + D * u["S1"]
    r2_guess = D * (u["S2"] - S2) + 0.9 * (1 - f_h2) * hsrc
    Y2 = (D + kd) * X2 / r2_guess
    Y1 = 1 - (ch4_cod - (1 - Y2) * D * (u["S2"] - S2)) / (hsrc * (f_h2 * (1 - Y_h2) + (1 - f_h2) * (1 - Y2)))
    r_ac = (1 - Y1) * hsrc
    r2 = D * (u["S2"] - S2) + (1 - f_h2) * r_ac
    Y2 = (D + kd) * X2 / r2
    K2 = 0.3
    lim = 10 ** -6.5
    sh = 10 ** -o["pH"]
    I_pH = lim ** 3 / (sh ** 3 + lim ** 3)
    k2 = r2 / (fm * S2 / (K2 + S2) * X2 * I_pH)
    c2 = 0.0313 - 0.95 * 0.0156 - 0.05 * 0.0313
    F_co2 = o["q_gas"] * o["p_gas_co2"] / o["P_gas"] * ap.p_atm / (0.083145 * T)  # kmol/d
    c1 = (F_co2 / ap.V_liq - D * (u["C"] - C) - c2 * r2) / r_ac
    n_N = D * (Z - u["Z"]) / r_h
    p = RMParams(V=ap.V_liq, k_h=ap.k_hyd_ch, Y1=Y1, f_h2=f_h2, Y_h2=Y_h2, k2=k2, K2=K2, Y2=Y2,
                 kd=kd, c1=c1, c2=c2, n_N=n_N, kLa=10.0, T_ref=ap.T_ref,
                 tan0=xs[IDX["S_IN"]])
    # gas transfer slowed to 10 1/d (ADM1: 200) so that a daily explicit step stays stable;
    # the CO2 solubility factor below restores the ADM1 gas composition
    p.ret = (r_h - D * (u["XT"] - XT)) / X2
    x0 = np.concatenate([lx, [0.0, 1.0, 0.0, 0.0]])
    rm = ReducedModel(p)
    # match pH and the CO2 outflow of ADM1 at the lumped state
    p.pH_offset = o["pH"] - rm.chem(x0, T, p.tan0)[0]
    lo, hi = 0.2, 5.0
    for _ in range(60):
        p.kH_factor = 0.5 * (lo + hi)
        if rm.rates(x0, u, T, p.tan0)["F_co2"] > F_co2:
            lo = p.kH_factor
        else:
            hi = p.kH_factor
    # the reduced model's own steady state under the nominal feed is the filter's starting point
    for _ in range(400):
        x0 = rm.step(x0, u, T, p.tan0)
    return p, x0


class EKF:
    """Daily extended Kalman filter; Jacobians by finite differences."""

    def __init__(self, model: ReducedModel, x0: np.ndarray):
        self.m = model
        self.x0 = x0
        # relative process noise per day: XT, S2, X2, C, Z. Inorganic carbon and alkalinity are
        # driven by the feed and kept stiff: small changes in C move CO2 and pH a lot (ill-conditioned)
        rel_q = np.array([0.02, 0.04, 0.01, 0.002, 0.002])
        rel_p0 = np.array([0.05, 0.1, 0.05, 0.005, 0.005])
        self.Q = np.diag(np.concatenate([(rel_q * x0[:N_X]) ** 2, [0.03 ** 2, 0.006 ** 2, 0.02 ** 2, 0.01 ** 2]]))
        self.P0 = np.diag(np.concatenate([(rel_p0 * x0[:N_X]) ** 2, [0.05 ** 2, 0.02 ** 2, 0.5 ** 2, 0.02 ** 2]]))
        self.R_online = np.array([0.02, 0.3, 0.02])  # relative biogas, %-pt CH4, pH
        self.R_lab = np.array([150.0, 200.0])         # FOS, TAC [mg/L]

    def _jac(self, f, x: np.ndarray, eps_rel: float = 1e-4) -> np.ndarray:
        y0 = f(x)
        J = np.empty((len(y0), len(x)))
        for j in range(len(x)):
            h = eps_rel * max(abs(x[j]), 1e-3)
            xp = x.copy()
            xp[j] += h
            J[:, j] = (f(xp) - y0) / h
        return J

    def run(self, d: pd.DataFrame) -> pd.DataFrame:
        """Filter one run's daily table; returns the estimates and normalised innovations."""
        x = self.x0.copy()
        P = self.P0.copy()
        feed_cols = [c for c in d.columns if c.startswith("feed_")]
        out = []
        tan = self.m.p.tan0
        for i in range(len(d)):
            row = d.iloc[i]
            feed = {c[5:]: float(row[c]) for c in feed_cols if pd.notna(row[c])}
            u = feed_inputs(feed)
            T = float(row["temp"]) + 273.15 if pd.notna(row["temp"]) else T_SET
            if pd.notna(row.get("tan", np.nan)):
                tan = float(row["tan"]) / 14.0
            # predict
            f = lambda z: self.m.step(z, u, T, tan)
            F = self._jac(f, x)
            x = f(x)
            P = F @ P @ F.T + self.Q
            # update
            lab = pd.notna(row["fos"]) and pd.notna(row["tac"])
            ym = [row["biogas"], row["ch4"], row["pH"]] + ([row["fos"], row["tac"]] if lab else [])
            ym = np.array(ym, dtype=float)
            sd = np.concatenate([[self.R_online[0] * max(row["biogas"], 1.0) if pd.notna(row["biogas"]) else 1.0],
                                 self.R_online[1:], self.R_lab if lab else []])
            ok = np.isfinite(ym)
            nis = np.nan
            if ok.any():
                h = lambda z: self.m.measure(z, u, T, tan, lab)
                y = h(x)
                H = self._jac(h, x)[ok]
                R = np.diag(sd[ok] ** 2)
                S = H @ P @ H.T + R
                innov = ym[ok] - y[ok]
                nis = float(innov @ np.linalg.solve(S, innov)) / ok.sum()
                if nis > 9.0:  # robust update: an outlying day gets a proportionally weaker update
                    R = R * nis / 9.0
                    S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.solve(S, np.eye(len(S)))
                x = x + K @ innov
                x[:N_X] = np.maximum(x[:N_X], 1e-6)
                P = (np.eye(len(x)) - K @ H) @ P
                P = 0.5 * (P + P.T)
            out.append({"day": int(row["day"]), "XT": x[0], "S2": x[1], "X2": x[2], "C": x[3],
                        "Z": x[4], "m": x[5], "s": x[6], "b": x[7], "beta": x[8], "sd_m": math.sqrt(P[5, 5]),
                        "sd_s": math.sqrt(P[6, 6]), "sd_b": math.sqrt(P[7, 7]), "nis": nis})
        return pd.DataFrame(out)


def ekf_alarm(est: pd.DataFrame, x: pd.DataFrame, s2_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """Alarms and classes from the filter's estimates (and the same feed log as the rules).

    VFA accumulation: S2 above 1.6x its nominal level, or methanogenic activity down by more than
    26 % (m < -0.3); lost activity without VFA (m < -0.15, S2 < 1.2x) or degradability s < 0.93 is a
    poorer substrate; |b| above 1.5 %-points is the CH4 analyser; frozen hourly flow is the meter.
    """
    n = len(est)
    m, s, b, S2 = est["m"].to_numpy(), est["s"].to_numpy(), est["b"].to_numpy(), est["S2"].to_numpy()
    flat = (x["flow_sd"] < 1e-6).fillna(False).to_numpy()
    cold = (x["temp_f"] < T_SET - 273.15 - 1.5).fillna(False).to_numpy()
    vfa_up = (S2 > 1.6 * s2_ref) | (m < -0.3)
    subst = (s < 0.93) | ((m < -0.15) & (S2 < 1.2 * s2_ref))
    drift = np.abs(b) > 1.5
    poultry = x.get("feed_poultry manure", pd.Series(0.0, index=x.index)).rolling(14, min_periods=1).sum().to_numpy()
    beet = x.get("feed_sugar beet", pd.Series(0.0, index=x.index)).rolling(10, min_periods=1).sum().to_numpy()
    odm_rel = (x["odm7"] / x["odm7"].shift(7).rolling(28, min_periods=14).median()).to_numpy()
    tan_hi = (x["tan_last"] > 3.2).fillna(False).to_numpy()
    cls = np.array(["normal"] * n, dtype=object)
    for i in range(n):
        if flat[i]:
            cls[i] = "flow_stuck"
        elif cold[i]:
            cls[i] = "temperature"
        elif vfa_up[i]:
            if beet[i] > 0:
                cls[i] = "acidification"
            elif poultry[i] > 0 or tan_hi[i]:
                cls[i] = "ammonia"
            elif np.isfinite(odm_rel[i]) and odm_rel[i] > 1.2:
                cls[i] = "overload"
            else:
                cls[i] = "overload"
        elif drift[i]:
            cls[i] = "ch4_drift"
        elif subst[i]:
            cls[i] = "substrate"
    raw = cls != "normal"
    alarm = raw & np.roll(raw, 1)
    alarm[0] = False
    cls[~alarm] = "normal"
    return alarm, cls
