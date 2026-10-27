"""IWA Anaerobic Digestion Model No. 1 with the BSM2 implementation details.

Model: Batstone et al. (2002), ADM1, IWA Scientific and Technical Report No. 13.
Implementation details: Rosen & Jeppsson (2006), "Aspects on ADM1 implementation within the
BSM2 framework", Lund University (IEA); Rosen et al. (2006), Water Sci. Technol. 54(4), 11-19.

The acid-base system is solved algebraically (the BSM2 "DAE" variant): the ionised fractions follow
from equilibrium and S_H+ from the charge balance by Newton-Raphson at every right-hand-side call.
Dissolved hydrogen stays a differential state and is handled by the stiff integrator.

Two additions, both inactive with the BSM2 parameter set, are needed for the agricultural plant:

- kinetic rates scale with temperature as exp(theta * (T - T_ref)) (BSM2: theta = 0);
- free-ammonia inhibition of acetoclastic methanogens is a Hill function with exponent n_nh3
  (BSM2: n_nh3 = 1, the non-competitive form of ADM1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

# 0-11 soluble, 12-23 particulate, 24-25 strong ions, 26-28 gas phase
STATE_NAMES = [
    "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC", "S_IN", "S_I",
    "X_c", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
    "S_cat", "S_an", "S_gas_h2", "S_gas_ch4", "S_gas_co2",
]
N_LIQ = 26
IDX = {name: i for i, name in enumerate(STATE_NAMES)}

R_GAS = 0.083145  # bar m3 / (kmol K)
T_BASE = 298.15


@dataclass(frozen=True)
class Params:
    """ADM1 parameters; defaults are the BSM2 values (Rosen & Jeppsson 2006, tables 1-3)."""

    # stoichiometry
    f_sI_xc: float = 0.1
    f_xI_xc: float = 0.2
    f_ch_xc: float = 0.2
    f_pr_xc: float = 0.2
    f_li_xc: float = 0.3
    N_xc: float = 0.0376 / 14.0
    N_I: float = 0.06 / 14.0
    N_aa: float = 0.007
    C_xc: float = 0.02786
    C_sI: float = 0.03
    C_ch: float = 0.0313
    C_pr: float = 0.03
    C_li: float = 0.022
    C_xI: float = 0.03
    C_su: float = 0.0313
    C_aa: float = 0.03
    f_fa_li: float = 0.95
    C_fa: float = 0.0217
    f_h2_su: float = 0.19
    f_bu_su: float = 0.13
    f_pro_su: float = 0.27
    f_ac_su: float = 0.41
    N_bac: float = 0.08 / 14.0
    C_bu: float = 0.025
    C_pro: float = 0.0268
    C_ac: float = 0.0313
    C_bac: float = 0.0313
    Y_su: float = 0.1
    f_h2_aa: float = 0.06
    f_va_aa: float = 0.23
    f_bu_aa: float = 0.26
    f_pro_aa: float = 0.05
    f_ac_aa: float = 0.40
    C_va: float = 0.024
    Y_aa: float = 0.08
    Y_fa: float = 0.06
    Y_c4: float = 0.06
    Y_pro: float = 0.04
    C_ch4: float = 0.0156
    Y_ac: float = 0.05
    Y_h2: float = 0.06
    # biochemical kinetics (at T_ref)
    k_dis: float = 0.5
    k_hyd_ch: float = 10.0
    k_hyd_pr: float = 10.0
    k_hyd_li: float = 10.0
    K_S_IN: float = 1e-4
    k_m_su: float = 30.0
    K_S_su: float = 0.5
    pH_UL_aa: float = 5.5
    pH_LL_aa: float = 4.0
    k_m_aa: float = 50.0
    K_S_aa: float = 0.3
    k_m_fa: float = 6.0
    K_S_fa: float = 0.4
    K_I_h2_fa: float = 5e-6
    k_m_c4: float = 20.0
    K_S_c4: float = 0.2
    K_I_h2_c4: float = 1e-5
    k_m_pro: float = 13.0
    K_S_pro: float = 0.1
    K_I_h2_pro: float = 3.5e-6
    k_m_ac: float = 8.0
    K_S_ac: float = 0.15
    K_I_nh3: float = 0.0018
    n_nh3: float = 1.0
    pH_UL_ac: float = 7.0
    pH_LL_ac: float = 6.0
    k_m_h2: float = 35.0
    K_S_h2: float = 7e-6
    pH_UL_h2: float = 6.0
    pH_LL_h2: float = 5.0
    k_dec: float = 0.02
    # temperature dependence of the kinetics: k(T) = k(T_ref) * exp(theta * (T - T_ref))
    T_ref: float = 308.15
    theta_hyd: float = 0.0   # disintegration, hydrolysis
    theta_acid: float = 0.0  # sugar, amino acid, LCFA uptake (acidogenesis)
    theta_acet: float = 0.0  # valerate, butyrate, propionate uptake (acetogenesis)
    theta_meth: float = 0.0  # acetate and hydrogen uptake (methanogenesis)
    theta_dec: float = 0.0
    # physico-chemical
    k_L_a: float = 200.0
    k_p: float = 5e4
    p_atm: float = 1.013
    # reactor
    V_liq: float = 3400.0
    V_gas: float = 300.0


BSM2 = Params()


def _hill_ph(pH_UL: float, pH_LL: float) -> tuple[float, float]:
    """BSM2 Hill-function form of the pH inhibition: (n, S_H limit)."""
    n = 3.0 / (pH_UL - pH_LL)
    lim = 10.0 ** (-(pH_UL + pH_LL) / 2.0)
    return n, lim


@dataclass
class Constants:
    """Temperature-dependent constants (van 't Hoff, BSM2 table 3)."""

    T: float
    K_w: float
    K_a_va: float
    K_a_bu: float
    K_a_pro: float
    K_a_ac: float
    K_a_co2: float
    K_a_IN: float
    K_H_co2: float
    K_H_ch4: float
    K_H_h2: float
    p_h2o: float

    @classmethod
    def at(cls, T: float) -> "Constants":
        f = 1.0 / T_BASE - 1.0 / T
        return cls(
            T=T,
            K_w=1e-14 * math.exp(55900.0 / (100.0 * R_GAS) * f),
            K_a_va=10 ** -4.86,
            K_a_bu=10 ** -4.82,
            K_a_pro=10 ** -4.88,
            K_a_ac=10 ** -4.76,
            K_a_co2=10 ** -6.35 * math.exp(7646.0 / (100.0 * R_GAS) * f),
            K_a_IN=10 ** -9.25 * math.exp(51965.0 / (100.0 * R_GAS) * f),
            K_H_co2=0.035 * math.exp(-19410.0 / (100.0 * R_GAS) * f),
            K_H_ch4=0.0014 * math.exp(-14240.0 / (100.0 * R_GAS) * f),
            K_H_h2=7.8e-4 * math.exp(-4180.0 / (100.0 * R_GAS) * f),
            p_h2o=0.0313 * math.exp(5290.0 * f),
        )


def solve_sh(x: np.ndarray, c: Constants, sh0: float = 1e-7) -> float:
    """S_H+ [kmol/m3] from the charge balance, Newton-Raphson on S_H+ (BSM2 DAE)."""
    S_va, S_bu, S_pro, S_ac = x[3], x[4], x[5], x[6]
    S_IC, S_IN, S_cat, S_an = x[9], x[10], x[24], x[25]
    sh = sh0 if sh0 > 0 else 1e-7
    for _ in range(50):
        va = c.K_a_va * S_va / (c.K_a_va + sh)
        bu = c.K_a_bu * S_bu / (c.K_a_bu + sh)
        pro = c.K_a_pro * S_pro / (c.K_a_pro + sh)
        ac = c.K_a_ac * S_ac / (c.K_a_ac + sh)
        hco3 = c.K_a_co2 * S_IC / (c.K_a_co2 + sh)
        nh3 = c.K_a_IN * S_IN / (c.K_a_IN + sh)
        oh = c.K_w / sh
        f = (S_cat + (S_IN - nh3) + sh - hco3 - ac / 64.0 - pro / 112.0 - bu / 160.0 - va / 208.0
             - oh - S_an)
        df = (1.0 + c.K_a_IN * S_IN / (c.K_a_IN + sh) ** 2
              + c.K_a_co2 * S_IC / (c.K_a_co2 + sh) ** 2
              + c.K_a_ac * S_ac / (64.0 * (c.K_a_ac + sh) ** 2)
              + c.K_a_pro * S_pro / (112.0 * (c.K_a_pro + sh) ** 2)
              + c.K_a_bu * S_bu / (160.0 * (c.K_a_bu + sh) ** 2)
              + c.K_a_va * S_va / (208.0 * (c.K_a_va + sh) ** 2)
              + c.K_w / sh ** 2)
        step = f / df
        new = sh - step
        if new <= 0:
            new = sh / 10.0
        if abs(new - sh) < 1e-12 * sh:
            return new
        sh = new
    return sh


class ADM1:
    """ADM1 right-hand side for one CSTR with a headspace."""

    def __init__(self, p: Params = BSM2):
        self.p = p
        self._c: Constants | None = None
        self._sh = 1e-7
        self.n_aa, self.lim_aa = _hill_ph(p.pH_UL_aa, p.pH_LL_aa)
        self.n_ac, self.lim_ac = _hill_ph(p.pH_UL_ac, p.pH_LL_ac)
        self.n_h2, self.lim_h2 = _hill_ph(p.pH_UL_h2, p.pH_LL_h2)
        q = p
        # carbon balance coefficients s1..s13 (BSM2)
        self.s = [
            -q.C_xc + q.f_sI_xc * q.C_sI + q.f_ch_xc * q.C_ch + q.f_pr_xc * q.C_pr
            + q.f_li_xc * q.C_li + q.f_xI_xc * q.C_xI,
            -q.C_ch + q.C_su,
            -q.C_pr + q.C_aa,
            -q.C_li + (1 - q.f_fa_li) * q.C_su + q.f_fa_li * q.C_fa,
            -q.C_su + (1 - q.Y_su) * (q.f_bu_su * q.C_bu + q.f_pro_su * q.C_pro + q.f_ac_su * q.C_ac)
            + q.Y_su * q.C_bac,
            -q.C_aa + (1 - q.Y_aa) * (q.f_va_aa * q.C_va + q.f_bu_aa * q.C_bu + q.f_pro_aa * q.C_pro
                                      + q.f_ac_aa * q.C_ac) + q.Y_aa * q.C_bac,
            -q.C_fa + (1 - q.Y_fa) * 0.7 * q.C_ac + q.Y_fa * q.C_bac,
            -q.C_va + (1 - q.Y_c4) * 0.54 * q.C_pro + (1 - q.Y_c4) * 0.31 * q.C_ac + q.Y_c4 * q.C_bac,
            -q.C_bu + (1 - q.Y_c4) * 0.8 * q.C_ac + q.Y_c4 * q.C_bac,
            -q.C_pro + (1 - q.Y_pro) * 0.57 * q.C_ac + q.Y_pro * q.C_bac,
            -q.C_ac + (1 - q.Y_ac) * q.C_ch4 + q.Y_ac * q.C_bac,
            (1 - q.Y_h2) * q.C_ch4 + q.Y_h2 * q.C_bac,
            -q.C_bac + q.C_xc,
        ]

    def constants(self, T: float) -> Constants:
        if self._c is None or self._c.T != T:
            self._c = Constants.at(T)
        return self._c

    def rates_factor(self, T: float) -> tuple[float, float, float, float, float]:
        p = self.p
        d = T - p.T_ref
        return (math.exp(p.theta_hyd * d), math.exp(p.theta_acid * d), math.exp(p.theta_acet * d),
                math.exp(p.theta_meth * d), math.exp(p.theta_dec * d))

    def rhs(self, t: float, x: np.ndarray, q_in: float, x_in: np.ndarray, T: float,
            uninhibited: bool = False) -> np.ndarray:
        p = self.p
        c = self.constants(T)
        fh, fa, fc, fm, fd = self.rates_factor(T)
        x = np.maximum(x, 0.0)
        (S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I,
         X_c, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I,
         S_cat, S_an, G_h2, G_ch4, G_co2) = x

        sh = solve_sh(x, c, self._sh)
        self._sh = sh
        hco3 = c.K_a_co2 * S_IC / (c.K_a_co2 + sh)
        nh3 = c.K_a_IN * S_IN / (c.K_a_IN + sh)
        co2 = S_IC - hco3

        if uninhibited:
            I_aa = I_ac = I_h2pH = I_nh3 = I_h2fa = I_h2c4 = I_h2pro = 1.0
        else:
            I_aa = self.lim_aa ** self.n_aa / (sh ** self.n_aa + self.lim_aa ** self.n_aa)
            I_ac = self.lim_ac ** self.n_ac / (sh ** self.n_ac + self.lim_ac ** self.n_ac)
            I_h2pH = self.lim_h2 ** self.n_h2 / (sh ** self.n_h2 + self.lim_h2 ** self.n_h2)
            I_nh3 = 1.0 / (1.0 + (nh3 / p.K_I_nh3) ** p.n_nh3)
            I_h2fa = 1.0 / (1.0 + S_h2 / p.K_I_h2_fa)
            I_h2c4 = 1.0 / (1.0 + S_h2 / p.K_I_h2_c4)
            I_h2pro = 1.0 / (1.0 + S_h2 / p.K_I_h2_pro)
        I_IN = 1.0 / (1.0 + p.K_S_IN / S_IN) if S_IN > 0 else 0.0
        I5 = I_aa * I_IN
        I_c4 = I5 * I_h2c4

        r1 = fh * p.k_dis * X_c
        r2 = fh * p.k_hyd_ch * X_ch
        r3 = fh * p.k_hyd_pr * X_pr
        r4 = fh * p.k_hyd_li * X_li
        r5 = fa * p.k_m_su * S_su / (p.K_S_su + S_su) * X_su * I5
        r6 = fa * p.k_m_aa * S_aa / (p.K_S_aa + S_aa) * X_aa * I5
        r7 = fa * p.k_m_fa * S_fa / (p.K_S_fa + S_fa) * X_fa * I5 * I_h2fa
        sv = S_bu + S_va + 1e-6
        r8 = fc * p.k_m_c4 * S_va / (p.K_S_c4 + S_va) * X_c4 * S_va / sv * I_c4
        r9 = fc * p.k_m_c4 * S_bu / (p.K_S_c4 + S_bu) * X_c4 * S_bu / sv * I_c4
        r10 = fc * p.k_m_pro * S_pro / (p.K_S_pro + S_pro) * X_pro * I5 * I_h2pro
        r11 = fm * p.k_m_ac * S_ac / (p.K_S_ac + S_ac) * X_ac * I_ac * I_IN * I_nh3
        r12 = fm * p.k_m_h2 * S_h2 / (p.K_S_h2 + S_h2) * X_h2 * I_h2pH * I_IN
        kd = fd * p.k_dec
        r13, r14, r15, r16 = kd * X_su, kd * X_aa, kd * X_fa, kd * X_c4
        r17, r18, r19 = kd * X_pro, kd * X_ac, kd * X_h2
        r_dec = r13 + r14 + r15 + r16 + r17 + r18 + r19

        RT = R_GAS * T
        p_h2 = G_h2 * RT / 16.0
        p_ch4 = G_ch4 * RT / 64.0
        p_co2 = G_co2 * RT
        P = p_h2 + p_ch4 + p_co2 + c.p_h2o
        q_gas = max(p.k_p * (P - p.p_atm), 0.0)
        rT8 = p.k_L_a * (S_h2 - 16.0 * c.K_H_h2 * p_h2)
        rT9 = p.k_L_a * (S_ch4 - 64.0 * c.K_H_ch4 * p_ch4)
        rT10 = p.k_L_a * (co2 - c.K_H_co2 * p_co2)

        D = q_in / p.V_liq
        dx = np.empty(29)
        dx[:N_LIQ] = D * (x_in - x[:N_LIQ])
        dx[0] += r2 + (1 - p.f_fa_li) * r4 - r5
        dx[1] += r3 - r6
        dx[2] += p.f_fa_li * r4 - r7
        dx[3] += (1 - p.Y_aa) * p.f_va_aa * r6 - r8
        dx[4] += (1 - p.Y_su) * p.f_bu_su * r5 + (1 - p.Y_aa) * p.f_bu_aa * r6 - r9
        dx[5] += ((1 - p.Y_su) * p.f_pro_su * r5 + (1 - p.Y_aa) * p.f_pro_aa * r6
                  + (1 - p.Y_c4) * 0.54 * r8 - r10)
        dx[6] += ((1 - p.Y_su) * p.f_ac_su * r5 + (1 - p.Y_aa) * p.f_ac_aa * r6
                  + (1 - p.Y_fa) * 0.7 * r7 + (1 - p.Y_c4) * 0.31 * r8 + (1 - p.Y_c4) * 0.8 * r9
                  + (1 - p.Y_pro) * 0.57 * r10 - r11)
        dx[7] += ((1 - p.Y_su) * p.f_h2_su * r5 + (1 - p.Y_aa) * p.f_h2_aa * r6
                  + (1 - p.Y_fa) * 0.3 * r7 + (1 - p.Y_c4) * 0.15 * r8 + (1 - p.Y_c4) * 0.2 * r9
                  + (1 - p.Y_pro) * 0.43 * r10 - r12 - rT8)
        dx[8] += (1 - p.Y_ac) * r11 + (1 - p.Y_h2) * r12 - rT9
        s = self.s
        sigma = (s[0] * r1 + s[1] * r2 + s[2] * r3 + s[3] * r4 + s[4] * r5 + s[5] * r6 + s[6] * r7
                 + s[7] * r8 + s[8] * r9 + s[9] * r10 + s[10] * r11 + s[11] * r12 + s[12] * r_dec)
        dx[9] += -sigma - rT10
        dx[10] += (-p.Y_su * p.N_bac * r5 + (p.N_aa - p.Y_aa * p.N_bac) * r6 - p.Y_fa * p.N_bac * r7
                   - p.Y_c4 * p.N_bac * (r8 + r9) - p.Y_pro * p.N_bac * r10 - p.Y_ac * p.N_bac * r11
                   - p.Y_h2 * p.N_bac * r12 + (p.N_bac - p.N_xc) * r_dec
                   + (p.N_xc - p.f_xI_xc * p.N_I - p.f_sI_xc * p.N_I - p.f_pr_xc * p.N_aa) * r1)
        dx[11] += p.f_sI_xc * r1
        dx[12] += -r1 + r_dec
        dx[13] += p.f_ch_xc * r1 - r2
        dx[14] += p.f_pr_xc * r1 - r3
        dx[15] += p.f_li_xc * r1 - r4
        dx[16] += p.Y_su * r5 - r13
        dx[17] += p.Y_aa * r6 - r14
        dx[18] += p.Y_fa * r7 - r15
        dx[19] += p.Y_c4 * (r8 + r9) - r16
        dx[20] += p.Y_pro * r10 - r17
        dx[21] += p.Y_ac * r11 - r18
        dx[22] += p.Y_h2 * r12 - r19
        dx[23] += p.f_xI_xc * r1
        vr = p.V_liq / p.V_gas
        dx[26] = -G_h2 * q_gas / p.V_gas + rT8 * vr
        dx[27] = -G_ch4 * q_gas / p.V_gas + rT9 * vr
        dx[28] = -G_co2 * q_gas / p.V_gas + rT10 * vr
        return dx

    def outputs(self, x: np.ndarray, T: float) -> dict:
        """Algebraic outputs: pH, ions, partial pressures, gas flow (BSM2 definitions)."""
        p = self.p
        c = Constants.at(T)
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        sh = solve_sh(x, c)
        RT = R_GAS * T
        p_h2 = x[26] * RT / 16.0
        p_ch4 = x[27] * RT / 64.0
        p_co2 = x[28] * RT
        P = p_h2 + p_ch4 + p_co2 + c.p_h2o
        q_head = max(p.k_p * (P - p.p_atm), 0.0)  # leaves the headspace at P_gas
        q_gas = q_head * P / p.p_atm               # the same gas at atmospheric pressure
        hco3 = c.K_a_co2 * x[9] / (c.K_a_co2 + sh)
        nh3 = c.K_a_IN * x[10] / (c.K_a_IN + sh)
        return {
            "pH": -math.log10(sh),
            "S_H": sh,
            "S_va_ion": c.K_a_va * x[3] / (c.K_a_va + sh),
            "S_bu_ion": c.K_a_bu * x[4] / (c.K_a_bu + sh),
            "S_pro_ion": c.K_a_pro * x[5] / (c.K_a_pro + sh),
            "S_ac_ion": c.K_a_ac * x[6] / (c.K_a_ac + sh),
            "S_hco3_ion": hco3,
            "S_co2": x[9] - hco3,
            "S_nh3": nh3,
            "S_nh4_ion": x[10] - nh3,
            "p_gas_h2": p_h2,
            "p_gas_ch4": p_ch4,
            "p_gas_co2": p_co2,
            "P_gas": P,
            "q_head": q_head,
            "q_gas": q_gas,
            # methane and dry biogas at normal conditions (273.15 K, 1.01325 bar)
            "q_ch4_N": q_gas * p_ch4 / P * p.p_atm / 1.01325 * 273.15 / T,
            "q_biogas_N": q_gas * (P - c.p_h2o) / P * p.p_atm / 1.01325 * 273.15 / T,
            "ch4_frac_dry": p_ch4 / (P - c.p_h2o),
        }


def simulate_constant(model: ADM1, x0: np.ndarray, q_in: float, x_in: np.ndarray, T: float,
                      days: float, t_eval=None, rtol: float = 1e-7, atol: float = 1e-10):
    """Integrate with constant influent (used for the BSM2 verification)."""
    return solve_ivp(lambda t, x: model.rhs(t, x, q_in, x_in, T), (0.0, days), x0, method="BDF",
                     rtol=rtol, atol=atol, t_eval=t_eval)


def with_params(p: Params, **kw) -> Params:
    return replace(p, **kw)


# --- BSM2 reference case --------------------------------------------------------------------------
# Constant influent of the BSM2 digester (Rosen & Jeppsson 2006), T = 35 C, Q = 170 m3/d.
BSM2_INFLUENT = np.array([
    0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 1e-8, 1e-5, 0.04, 0.01, 0.02,
    2.0, 5.0, 20.0, 5.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 25.0,
    0.04, 0.02,
])
BSM2_Q = 170.0
BSM2_T = 308.15

# Published steady state of that case (Rosen & Jeppsson 2006, steady-state results table).
BSM2_STEADY_STATE = {
    "S_su": 0.0119548297170, "S_aa": 0.0053147401716, "S_fa": 0.0986214009308,
    "S_va": 0.0116250064639, "S_bu": 0.0132507296663, "S_pro": 0.0157836662845,
    "S_ac": 0.1976297169375, "S_h2": 0.0000002359451, "S_ch4": 0.0550887764460,
    "S_IC": 0.1526778706263, "S_IN": 0.1302298158037, "S_I": 0.3286976637215,
    "X_c": 0.3086976637215, "X_ch": 0.0279472404350, "X_pr": 0.1025741061067,
    "X_li": 0.0294830497073, "X_su": 0.4201659824546, "X_aa": 1.1791717989237,
    "X_fa": 0.2430353447194, "X_c4": 0.4319211056360, "X_pro": 0.1373059089340,
    "X_ac": 0.7605626583132, "X_h2": 0.3170229533613, "X_I": 25.6173953274430,
    "S_cat": 0.04, "S_an": 0.02,
    "pH": 7.4655377698929, "S_hco3_ion": 0.1427774793921, "S_nh3": 0.0040909284584,
    "S_ac_ion": 0.1972411554365, "S_gas_h2": 0.0000102410356, "S_gas_ch4": 1.6256072099814,
    "S_gas_co2": 0.0141505346784, "q_gas": 2955.70345419378, "p_gas_ch4": 0.6507796328232,
    "p_gas_co2": 0.3625527133281,
}
