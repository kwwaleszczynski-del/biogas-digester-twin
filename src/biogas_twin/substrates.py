"""Substrate characterisation: Weender-type composition -> ADM1 influent.

Values are typical figures chosen inside the ranges reported for German and Polish agricultural plants
(FNR Leitfaden Biogas; KTBL standard biogas yields; feed-value tables). They describe an illustrative
plant, not a measured one; the README lists the sources and the resulting methane yields, which a test
keeps inside the published standard ranges.

Mapping (common practice for agricultural ADM1 applications, e.g. Lübken et al. 2007; Koch et al. 2010):

- crude protein, lipids and carbohydrates enter directly as X_pr, X_li and X_ch (no X_c step);
  silage acids and slurry volatile fatty acids enter as dissolved S_ac / S_pro / S_bu;
- a substrate-specific anaerobic degradability splits the particulate COD into degradable and X_I;
- COD factors per kg: carbohydrate 1.185 (C6H10O5), protein 1.50, lipid 2.90, lactic and acetic
  acid 1.067, sucrose 1.123;
- ammonium nitrogen enters as S_IN; protein nitrogen follows ADM1's N_aa when proteins hydrolyse;
- potassium and sodium are the monovalent cations S_cat, the strong anions (chloride, sulphate,
  phosphate) S_an, both in kmol charge per m3.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .adm1 import IDX, N_LIQ

COD_CH = 1.185
COD_PR = 1.50
COD_LI = 2.90
COD_ACID = 1.067   # lactic and acetic acid
COD_SUC = 1.123    # sucrose
COD_PRO = 1.514    # propionic acid
COD_BU = 1.818     # butyric acid


@dataclass(frozen=True)
class Substrate:
    """Composition per kg dry matter (DM) unless stated otherwise."""

    name: str
    dm: float            # kg DM / kg fresh matter
    odm: float           # kg organic DM / kg DM
    protein: float       # crude protein, kg / kg DM (organic N x 6.25)
    lipid: float         # crude fat, kg / kg DM
    sugar: float         # free sugars, kg / kg DM (dissolved, enter as S_su)
    acids: float         # lactic + acetic acid (silage) or VFA (slurry), kg / kg DM, as acetate COD
    degradability: float  # anaerobically degradable share of the particulate organic matter
    tan: float           # ammonium N, kg N / t fresh matter
    cations: float       # K+ + Na+ as charge, kmol / t fresh matter
    anions: float        # strong anions as charge, kmol / t fresh matter
    inorganic_c: float = 0.0  # dissolved inorganic carbon, kmol C / t fresh matter

    @property
    def carbohydrate(self) -> float:
        return self.odm - self.protein - self.lipid - self.sugar - self.acids

    def cod_per_t(self) -> dict[str, float]:
        """kg COD per t fresh matter of each ADM1 input (degradable particulates split from X_I)."""
        dm = 1000.0 * self.dm
        f = self.degradability
        ch = dm * self.carbohydrate * COD_CH
        pr = dm * self.protein * COD_PR
        li = dm * self.lipid * COD_LI
        return {
            "X_ch": f * ch, "X_pr": f * pr, "X_li": f * li,
            "X_I": (1.0 - f) * (ch + pr + li),
            "S_su": dm * self.sugar * COD_SUC,
            "S_ac": dm * self.acids * COD_ACID,
        }

    def odm_per_t(self) -> float:
        return 1000.0 * self.dm * self.odm


MAIZE_SILAGE = Substrate(
    name="maize silage", dm=0.33, odm=0.955, protein=0.085, lipid=0.032, sugar=0.010,
    acids=0.065, degradability=0.90, tan=0.4, cations=0.090, anions=0.030,
)
CATTLE_SLURRY = Substrate(
    name="cattle slurry", dm=0.09, odm=0.80, protein=0.15, lipid=0.035, sugar=0.0,
    acids=0.045, degradability=0.50, tan=2.6, cations=0.115, anions=0.035, inorganic_c=0.06,
)
# N-rich co-substrate for the ammonia scenario
POULTRY_MANURE = Substrate(
    name="poultry manure", dm=0.45, odm=0.72, protein=0.26, lipid=0.025, sugar=0.0,
    acids=0.02, degradability=0.65, tan=6.0, cations=0.20, anions=0.06, inorganic_c=0.05,
)
# sugar-rich co-substrate for the acidification scenario (whole sugar beet)
SUGAR_BEET = Substrate(
    name="sugar beet", dm=0.23, odm=0.92, protein=0.05, lipid=0.005, sugar=0.68,
    acids=0.0, degradability=0.95, tan=0.1, cations=0.035, anions=0.010,
)

SUBSTRATES = {s.name: s for s in (MAIZE_SILAGE, CATTLE_SLURRY, POULTRY_MANURE, SUGAR_BEET)}


def degraded(sub: Substrate, factor: float) -> Substrate:
    """The same substrate with its degradability scaled (a poorer silage clamp)."""
    return replace(sub, degradability=sub.degradability * factor)


def influent(feed: dict[str, float], subs: dict[str, Substrate] | None = None) -> tuple[float, np.ndarray]:
    """Daily feed {substrate name: t fresh matter/d} -> (q_in m3/d, ADM1 influent concentrations).

    One tonne of fresh matter is taken as one cubic metre of liquid.
    """
    subs = subs or SUBSTRATES
    q = sum(feed.values())
    load = np.zeros(N_LIQ)
    for name, t in feed.items():
        if t <= 0:
            continue
        s = subs[name]
        for key, cod in s.cod_per_t().items():
            load[IDX[key]] += t * cod  # kg COD/d
        load[IDX["S_IN"]] += t * s.tan / 14.0  # kmol N/d
        load[IDX["S_cat"]] += t * s.cations
        load[IDX["S_an"]] += t * s.anions
        load[IDX["S_IC"]] += t * s.inorganic_c
    x_in = load / q if q > 0 else load
    # traces that keep the ADM1 Monod and Hill terms well defined
    x_in[IDX["S_h2"]] = max(x_in[IDX["S_h2"]], 1e-8)
    x_in[IDX["S_ch4"]] = max(x_in[IDX["S_ch4"]], 1e-5)
    return q, x_in


def odm_fed(feed: dict[str, float], subs: dict[str, Substrate] | None = None) -> float:
    """Organic dry matter fed [t oDM/d]."""
    subs = subs or SUBSTRATES
    return sum(t * subs[name].odm_per_t() / 1000.0 for name, t in feed.items())
