"""Detection and classification methods. Each works on the daily table of one run, causally.

Detectors return a boolean alarm per day; classifiers also return a class per day. The in-control
statistics (EWMA, CUSUM, PCA) and the gradient boosting model are fitted on training runs only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .plant import T_SET

ONLINE = ["biogas", "ch4", "pH", "temp"]


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    """Causal derived signals: gaps bridged up to 3 days, 7-day specific methane yield."""
    x = d.copy()
    for c in ONLINE + ["flow_sd"]:
        x[c + "_f"] = x[c].ffill(limit=3)
    ch4_nm3 = x["biogas_f"] * x["ch4_f"] / 100.0
    x["ch4_nm3_f"] = ch4_nm3
    odm7 = x["odm_t"].rolling(7, min_periods=5).mean()
    x["odm7"] = odm7
    x["smy"] = ch4_nm3.rolling(7, min_periods=5).mean() / (1000.0 * odm7)   # Nm3 CH4 / kg oDM
    x["gas_per_odm"] = x["biogas_f"] / (1000.0 * odm7)
    # last lab values carried forward and their age
    for c in ["fos", "tac", "fos_tac", "tan"]:
        x[c + "_last"] = x[c].ffill()
    lab_day = x["day"].where(x["fos"].notna()).ffill()
    x["lab_age"] = x["day"] - lab_day
    return x


# --- 1. FOS/TAC thresholds (industry practice) ------------------------------------------------------
def fostac_absolute(x: pd.DataFrame, limit: float = 0.3) -> np.ndarray:
    """Alarm from a lab result above `limit` until the next result."""
    return (x["fos_tac_last"] > limit).to_numpy()


def fostac_relative(x: pd.DataFrame, rise: float = 0.3, n_ref: int = 8) -> np.ndarray:
    """Plant-specific: the latest FOS/TAC more than `rise` (30 %) above the median of the previous
    `n_ref` results."""
    lab = x.loc[x["fos_tac"].notna(), ["day", "fos_tac"]]
    ref = lab["fos_tac"].shift(1).rolling(n_ref, min_periods=4).median()
    flag = pd.Series(lab["fos_tac"].to_numpy() > (1 + rise) * ref.to_numpy(), index=lab["day"])
    out = pd.Series(np.nan, index=x["day"])
    out.loc[flag.index] = flag.astype(float)
    return out.ffill().fillna(0).astype(bool).to_numpy()


# --- 2. EWMA and CUSUM on the specific methane yield ---------------------------------------------
class YieldCharts:
    """Two-sided EWMA (lambda 0.2, L = 3) and tabular CUSUM (k = 0.5 sigma, h = 5 sigma) on the
    7-day specific methane yield, in-control mean and sigma from normal training runs."""

    def __init__(self, lam: float = 0.2, L: float = 3.0, k: float = 0.5, h: float = 5.0):
        self.lam, self.L, self.k, self.h = lam, L, k, h

    def fit(self, runs: list[pd.DataFrame]) -> "YieldCharts":
        s = pd.concat([r["smy"] for r in runs]).dropna()
        self.mu = float(s.mean())
        # sigma of the daily series (autocorrelated): from the spread of 7-day blocks is too small;
        # the plain standard deviation keeps the false-alarm rate of normal operation in check
        self.sd = float(s.std())
        return self

    def z(self, x: pd.DataFrame) -> np.ndarray:
        return ((x["smy"] - self.mu) / self.sd).ffill().fillna(0.0).to_numpy()

    def ewma(self, x: pd.DataFrame) -> np.ndarray:
        z = self.z(x)
        e = 0.0
        lim = self.L * np.sqrt(self.lam / (2 - self.lam))
        out = np.zeros(len(z), dtype=bool)
        for i, v in enumerate(z):
            e = self.lam * v + (1 - self.lam) * e
            out[i] = abs(e) > lim
        return out

    def cusum(self, x: pd.DataFrame) -> np.ndarray:
        z = self.z(x)
        hi = lo = 0.0
        out = np.zeros(len(z), dtype=bool)
        for i, v in enumerate(z):
            hi = max(0.0, hi + v - self.k)
            lo = max(0.0, lo - v - self.k)
            out[i] = hi > self.h or lo > self.h
            if out[i]:  # restart after a signal, as a chart would be after an investigation
                hi = lo = 0.0
        return out


# --- 3. PCA T2 / SPE --------------------------------------------------------------------------------
PCA_VARS = ["smy", "gas_per_odm", "ch4_f", "pH_f", "temp_f", "d_ch4", "d_pH", "d_smy"]


def pca_frame(x: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=x.index)
    f["smy"] = x["smy"]
    f["gas_per_odm"] = x["gas_per_odm"]
    f["ch4_f"] = x["ch4_f"]
    f["pH_f"] = x["pH_f"]
    f["temp_f"] = x["temp_f"]
    f["d_ch4"] = x["ch4_f"] - x["ch4_f"].shift(7)
    f["d_pH"] = x["pH_f"] - x["pH_f"].shift(7)
    f["d_smy"] = x["smy"] - x["smy"].shift(7)
    return f[PCA_VARS]


class PCAMonitor:
    """PCA on standardised daily signals; alarm when T2 or SPE exceed their 99.5 % training quantile
    on two consecutive days."""

    def __init__(self, var_explained: float = 0.9, q: float = 0.995):
        self.var_explained, self.q = var_explained, q

    def fit(self, runs: list[pd.DataFrame]) -> "PCAMonitor":
        X = pd.concat([pca_frame(r) for r in runs]).dropna().to_numpy()
        self.mu, self.sd = X.mean(0), X.std(0)
        Z = (X - self.mu) / self.sd
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        ev = S ** 2 / (len(Z) - 1)
        self.k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), self.var_explained) + 1)
        self.P = Vt[: self.k].T
        self.lam = ev[: self.k]
        t2, spe = self._stats(Z)
        self.t2_lim = float(np.quantile(t2, self.q))
        self.spe_lim = float(np.quantile(spe, self.q))
        return self

    def _stats(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = Z @ self.P
        t2 = np.sum(T ** 2 / self.lam, axis=1)
        res = Z - T @ self.P.T
        return t2, np.sum(res ** 2, axis=1)

    def scores(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        F = pca_frame(x).ffill().fillna(pd.Series(self.mu, index=PCA_VARS))
        Z = (F.to_numpy() - self.mu) / self.sd
        return self._stats(Z)

    def alarm(self, x: pd.DataFrame) -> np.ndarray:
        t2, spe = self.scores(x)
        raw = (t2 > self.t2_lim) | (spe > self.spe_lim)
        out = raw & np.roll(raw, 1)
        out[0] = False
        return out


# --- 4. Expert symptom rules (detector + classifier) -------------------------------------------------
def rules(x: pd.DataFrame, charts: YieldCharts) -> tuple[np.ndarray, np.ndarray]:
    """The checks an experienced operator runs every morning, top to bottom.

    Returns (alarm, class) per day. The feed log is part of the operator's knowledge.
    """
    n = len(x)
    cls = np.array(["normal"] * n, dtype=object)
    feed_odm7 = x["odm7"]
    base_odm = feed_odm7.shift(7).rolling(28, min_periods=14).median()
    poultry = x.get("feed_poultry manure", pd.Series(0.0, index=x.index)).rolling(7, min_periods=1).sum()
    beet = x.get("feed_sugar beet", pd.Series(0.0, index=x.index)).rolling(10, min_periods=1).sum()
    ewma = charts.ewma(x)
    fostac_rel = fostac_relative(x)
    ch4_ref = x["ch4_f"].shift(7).rolling(21, min_periods=7).median()
    pH_ref = x["pH_f"].shift(7).rolling(21, min_periods=7).median()
    smy_ref = x["smy"].shift(7).rolling(21, min_periods=7).median()
    tan_hi = x["tan_last"] > 3.2
    for i in range(n):
        flat = x["flow_sd"].iloc[i] < 1e-6 if pd.notna(x["flow_sd"].iloc[i]) else False
        if flat:
            cls[i] = "flow_stuck"
            continue
        if pd.notna(x["temp_f"].iloc[i]) and x["temp_f"].iloc[i] < T_SET - 273.15 - 1.5:
            cls[i] = "temperature"
            continue
        upset = ewma[i] or fostac_rel[i] or (
            pd.notna(pH_ref.iloc[i]) and x["pH_f"].iloc[i] < pH_ref.iloc[i] - 0.08)
        ch4_jump = pd.notna(ch4_ref.iloc[i]) and abs(x["ch4_f"].iloc[i] - ch4_ref.iloc[i]) > 2.0
        if not upset and not ch4_jump:
            continue
        if beet.iloc[i] > 0:
            cls[i] = "acidification"
        elif poultry.iloc[i] > 0 or bool(tan_hi.iloc[i]):
            cls[i] = "ammonia"
        elif pd.notna(base_odm.iloc[i]) and feed_odm7.iloc[i] > 1.2 * base_odm.iloc[i]:
            cls[i] = "overload"
        elif ch4_jump and not fostac_rel[i] and abs(x["pH_f"].iloc[i] - (pH_ref.iloc[i] if pd.notna(pH_ref.iloc[i]) else x["pH_f"].iloc[i])) < 0.04:
            cls[i] = "ch4_drift"
        elif pd.notna(smy_ref.iloc[i]) and x["smy"].iloc[i] < 0.95 * smy_ref.iloc[i] and not fostac_rel[i]:
            cls[i] = "substrate"
        elif fostac_rel[i]:
            cls[i] = "overload"
        else:
            cls[i] = "substrate"
    return cls != "normal", cls


# --- 5. Gradient boosting classifier ------------------------------------------------------------------
def gbm_features(x: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=x.index)
    for c in ["smy", "gas_per_odm", "ch4_f", "pH_f", "temp_f"]:
        s = x[c]
        f[c] = s
        for lag in (3, 7, 14):
            f[f"{c}_d{lag}"] = s - s.shift(lag)
        f[c + "_sd7"] = s.rolling(7, min_periods=3).std()
        f[c + "_dev28"] = s - s.shift(7).rolling(28, min_periods=10).median()
    f["flow_sd"] = x["flow_sd_f"]
    f["flow_sd_min3"] = x["flow_sd_f"].rolling(3, min_periods=1).min()
    f["fos_tac"] = x["fos_tac_last"]
    f["fos"] = x["fos_last"]
    f["tac"] = x["tac_last"]
    f["tan"] = x["tan_last"]
    f["lab_age"] = x["lab_age"]
    lab = x.loc[x["fos_tac"].notna(), "fos_tac"]
    f["fos_tac_trend"] = (lab - lab.shift(1)).reindex(x.index).ffill()
    f["odm_rel"] = x["odm7"] / x["odm7"].shift(7).rolling(28, min_periods=10).median()
    for c in ["feed_poultry manure", "feed_sugar beet"]:
        if c in x:
            f[c.replace("feed_", "f_").replace(" ", "_") + "_7"] = x[c].rolling(7, min_periods=1).sum()
            f[c.replace("feed_", "f_").replace(" ", "_") + "_28"] = x[c].rolling(28, min_periods=1).sum()
    f["silage_share"] = x["feed_maize silage"] / x[[c for c in x if c.startswith("feed_")]].sum(axis=1)
    return f


class GBMClassifier:
    """HistGradientBoosting on daily features; alarm when P(normal) < 0.5 on two consecutive days,
    the class is the most likely upset at that day."""

    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                    max_leaf_nodes=31, l2_regularization=1.0,
                                                    class_weight="balanced", random_state=seed)

    def fit(self, frames: list[pd.DataFrame], labels: list[np.ndarray]) -> "GBMClassifier":
        X = pd.concat([gbm_features(f) for f in frames])
        y = np.concatenate(labels)
        self.columns = list(X.columns)
        self.model.fit(X.to_numpy(), y)
        return self

    def predict(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = gbm_features(x).reindex(columns=self.columns)
        P = self.model.predict_proba(X.to_numpy())
        classes = list(self.model.classes_)
        p_norm = P[:, classes.index("normal")]
        raw = p_norm < 0.5
        alarm = raw & np.roll(raw, 1)
        alarm[0] = False
        Pu = P.copy()
        Pu[:, classes.index("normal")] = -1
        cls = np.array([classes[j] for j in Pu.argmax(1)], dtype=object)
        cls[~alarm] = "normal"
        return alarm, cls, P


def day_labels(sc, n: int, temp: np.ndarray | None = None) -> np.ndarray:
    """Training labels: the upset class while its cause (or its direct after-effect) is present."""
    y = np.array(["normal"] * n, dtype=object)
    if sc.kind == "normal":
        return y
    end = n
    if sc.kind == "acidification":
        end = min(n, sc.t_end + 14)
    elif sc.kind == "flow_stuck":
        end = sc.t_end
    elif sc.kind == "temperature" and temp is not None:
        back = np.where((np.arange(n) >= sc.t_end) & (temp >= T_SET - 0.3))[0]
        end = int(back[0]) + 3 if len(back) else n
    y[sc.t0:end] = sc.kind
    return y
