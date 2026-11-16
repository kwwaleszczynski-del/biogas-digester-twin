"""Detection and classification methods. Each works on the daily table of one run, causally.

Every detector also returns a daily score scaled so that its published operating point alarms at
score > 1; another threshold on the same score moves it along its false-alarm/detection curve (used
to compare all detectors at one common false-alarm rate). Classifiers also return a class per day.
The in-control statistics (EWMA, CUSUM, PCA) and the gradient boosting model are fitted on training
runs only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

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


def persist2(score: np.ndarray) -> np.ndarray:
    """Score of a two-consecutive-days rule: day t qualifies only if days t-1 and t both do."""
    s = np.asarray(score, dtype=float)
    out = np.minimum(s, np.concatenate([[0.0], s[:-1]]))
    out[0] = 0.0
    return out


# --- 1. FOS/TAC thresholds (industry practice) ------------------------------------------------------
def fostac_absolute_score(x: pd.DataFrame, limit: float = 0.3) -> np.ndarray:
    """Latest lab FOS/TAC over `limit`, held until the next result."""
    return (x["fos_tac_last"] / limit).fillna(0.0).to_numpy()


def fostac_absolute(x: pd.DataFrame, limit: float = 0.3) -> np.ndarray:
    """Alarm from a lab result above `limit` until the next result."""
    return fostac_absolute_score(x, limit) > 1.0


def fostac_relative_score(x: pd.DataFrame, rise: float = 0.3, n_ref: int = 8) -> np.ndarray:
    """Rise of the latest FOS/TAC over the median of the previous `n_ref` results, in units of
    `rise`, held until the next result."""
    lab = x.loc[x["fos_tac"].notna(), ["day", "fos_tac"]]
    ref = lab["fos_tac"].shift(1).rolling(n_ref, min_periods=4).median()
    rel = pd.Series((lab["fos_tac"].to_numpy() / ref.to_numpy() - 1.0) / rise, index=lab["day"])
    out = pd.Series(np.nan, index=x["day"])
    out.loc[rel.index] = rel.fillna(0.0)
    return out.ffill().fillna(0.0).to_numpy()


def fostac_relative(x: pd.DataFrame, rise: float = 0.3, n_ref: int = 8) -> np.ndarray:
    """Plant-specific: the latest FOS/TAC more than `rise` (30 %) above the median of the previous
    `n_ref` results."""
    return fostac_relative_score(x, rise, n_ref) > 1.0


# --- 2. EWMA and CUSUM on the specific methane yield ---------------------------------------------
class YieldCharts:
    """Two-sided EWMA (lambda 0.2, L = 3) and tabular CUSUM (k = 0.5 sigma, h = 5 sigma) on the
    7-day specific methane yield, in-control mean and sigma from normal training runs. The CUSUM is
    not restarted after a signal: it stays in alarm while the shift lasts, so one shift is one alarm
    (a restart would raise a new alarm every few days of the same shift)."""

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

    def ewma_score(self, x: pd.DataFrame) -> np.ndarray:
        z = self.z(x)
        e = 0.0
        lim = self.L * np.sqrt(self.lam / (2 - self.lam))
        out = np.zeros(len(z))
        for i, v in enumerate(z):
            e = self.lam * v + (1 - self.lam) * e
            out[i] = abs(e) / lim
        return out

    def ewma(self, x: pd.DataFrame) -> np.ndarray:
        return self.ewma_score(x) > 1.0

    def cusum_score(self, x: pd.DataFrame) -> np.ndarray:
        z = self.z(x)
        hi = lo = 0.0
        out = np.zeros(len(z))
        for i, v in enumerate(z):
            hi = max(0.0, hi + v - self.k)
            lo = max(0.0, lo - v - self.k)
            out[i] = max(hi, lo) / self.h
        return out

    def cusum(self, x: pd.DataFrame) -> np.ndarray:
        return self.cusum_score(x) > 1.0


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

    def score(self, x: pd.DataFrame) -> np.ndarray:
        t2, spe = self.scores(x)
        return persist2(np.maximum(t2 / self.t2_lim, spe / self.spe_lim))

    def alarm(self, x: pd.DataFrame) -> np.ndarray:
        return self.score(x) > 1.0


# --- 4. Expert symptom rules (detector + classifier) -------------------------------------------------
def rules(x: pd.DataFrame, charts: YieldCharts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The checks an experienced operator runs every morning, top to bottom.

    Returns (alarm, class, score) per day. The feed log is part of the operator's knowledge. The
    score is the largest of the symptom checks relative to its limit (frozen meter 10, cold digester,
    EWMA, FOS/TAC rise, pH drop of 0.08, CH4 jump of 2 %-points), so alarm == score > 1.
    """
    n = len(x)
    cls = np.array(["normal"] * n, dtype=object)
    feed_odm7 = x["odm7"]
    base_odm = feed_odm7.shift(7).rolling(28, min_periods=14).median()
    poultry = x.get("feed_poultry manure", pd.Series(0.0, index=x.index)).rolling(7, min_periods=1).sum()
    beet = x.get("feed_sugar beet", pd.Series(0.0, index=x.index)).rolling(10, min_periods=1).sum()
    ewma_k = charts.ewma_score(x)
    ewma = ewma_k > 1.0
    fostac_k = fostac_relative_score(x)
    fostac_rel = fostac_k > 1.0
    ch4_ref = x["ch4_f"].shift(7).rolling(21, min_periods=7).median()
    pH_ref = x["pH_f"].shift(7).rolling(21, min_periods=7).median()
    flat_k = np.where((x["flow_sd"] < 1e-6).fillna(False).to_numpy(), 10.0, 0.0)
    cold_k = ((T_SET - 273.15 - x["temp_f"]) / 1.5).fillna(0.0).to_numpy()
    pH_k = ((pH_ref - x["pH_f"]) / 0.08).fillna(0.0).to_numpy()
    ch4_k = ((x["ch4_f"] - ch4_ref).abs() / 2.0).fillna(0.0).to_numpy()
    score = np.maximum.reduce([flat_k, cold_k, ewma_k, fostac_k, pH_k, ch4_k])
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
    return cls != "normal", cls, score


# --- 5. Gradient boosting classifier ------------------------------------------------------------------
FEED_FEATURES = ["f_poultry_manure_7", "f_poultry_manure_28", "f_sugar_beet_7", "f_sugar_beet_28",
                 "silage_share", "odm_rel"]


def gbm_features(x: pd.DataFrame) -> pd.DataFrame:
    """Daily features. FEED_FEATURES come from the feed log (what and how much was fed); the yield
    features divide by the logged oDM as every plant's KPI does."""
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
    the class is the most likely upset at that day. Fixed iterations (no early stopping) and one
    thread, so that a refit gives the same model bit for bit. `feed_log=False` drops the features
    that say what and how much was fed (the check for learning upsets from feed sizes)."""

    def __init__(self, seed: int = 0, feed_log: bool = True):
        self.feed_log = feed_log
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                    max_leaf_nodes=31, l2_regularization=1.0,
                                                    early_stopping=False,
                                                    class_weight="balanced", random_state=seed)

    def fit(self, frames: list[pd.DataFrame], labels: list[np.ndarray]) -> "GBMClassifier":
        X = pd.concat([gbm_features(f) for f in frames])
        if not self.feed_log:
            X = X.drop(columns=[c for c in FEED_FEATURES if c in X])
        y = np.concatenate(labels)
        self.columns = list(X.columns)
        with threadpool_limits(1):
            self.model.fit(X.to_numpy(), y)
        return self

    def predict(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(alarm, class, score): score = (1 - P(normal)) / 0.5, held on two days."""
        X = gbm_features(x).reindex(columns=self.columns)
        with threadpool_limits(1):
            P = self.model.predict_proba(X.to_numpy())
        classes = list(self.model.classes_)
        p_norm = P[:, classes.index("normal")]
        score = persist2((1.0 - p_norm) / 0.5)
        alarm = score > 1.0
        Pu = P.copy()
        Pu[:, classes.index("normal")] = -1
        cls = np.array([classes[j] for j in Pu.argmax(1)], dtype=object)
        cls[~alarm] = "normal"
        return alarm, cls, score


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
