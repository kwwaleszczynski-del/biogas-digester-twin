"""Scores on the daily alarm series of the test runs: detection against chance, false alarms, lead
time before a material yield loss, classification, bootstrap intervals.

One event rule for everything: an **alarm onset** is a day in alarm after a day without. A run's
upset is detected if an onset falls within WINDOW days from t0 (an alarm already running at t0 does
not count). False alarms are onsets per year on upset-free days after the warm-up: the normal years
and the history before t0 of every upset run. **Chance** applies the detection rule at every possible
fake t0 on those upset-free days: the share of WINDOW-day windows that contain an onset by
coincidence. **Excess** = detected - chance: what a method finds beyond its own alarm habit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scenarios import CLASSES, PROCESS, SENSOR

WARMUP = 35        # days of history a method needs before its alarms count
WINDOW = 14        # an upset counts as detected if an alarm starts within this many days from t0
LONG_WINDOW = 60   # the window of the first version, kept as a sensitivity column
MATERIAL = 0.02    # material yield loss: at least 2 % of the healthy twin's methane lost after t0
MAX_W = 90
COMMON_RATE = 2.0  # false alarms per year at which all detectors are compared
N_BOOT = 1000
# the upset type whose yield loss builds up after the cause, so that a lead time means early warning
# (a substrate change loses yield with the first poorer silage; the others are mild and transient)
LEAD_KINDS = ("ammonia",)


def onsets(alarm: np.ndarray) -> np.ndarray:
    return np.where(onset_mask(alarm))[0]


def onset_mask(alarm: np.ndarray) -> np.ndarray:
    a = np.asarray(alarm, dtype=bool)
    return a & ~np.concatenate([[False], a[:-1]])


@dataclass
class Run:
    """One test run as the scores see it: ground truth and each method's daily series."""

    kind: str
    seed: int
    missing: float
    n: int
    t0: int | None
    loss_day: int | None
    material: bool
    alarm: dict            # method -> bool array
    cls: dict              # method -> object array (classifiers only)
    score: dict            # method -> float array

    @property
    def stop(self) -> int:
        """End of the upset-free part: t0, or the end of a normal run."""
        return self.t0 if self.t0 is not None else self.n


# --- per run --------------------------------------------------------------------------------------
def first_onset(alarm: np.ndarray, start: int, stop: int) -> int | None:
    o = onsets(alarm)
    o = o[(o >= start) & (o < stop)]
    return int(o[0]) if len(o) else None


def named_class(cls: np.ndarray, d: int) -> str:
    """The class named over the first three alarm days (majority, ties to the first)."""
    names = [c for c in cls[d:d + 3] if c != "normal"] or [cls[d]]
    return max(dict.fromkeys(names), key=names.count)


def window_hits(alarm: np.ndarray, start: int, stop: int, w: int) -> tuple[int, int]:
    """(windows with an onset, windows) over every fake t0 in [start, stop - w]."""
    if stop - w < start:
        return 0, 0
    c = np.concatenate([[0], np.cumsum(onset_mask(alarm))])
    t = np.arange(start, stop - w + 1)
    return int(np.count_nonzero(c[t + w] - c[t])), len(t)


def free_counts(alarm: np.ndarray, start: int, stop: int) -> tuple[int, int, int]:
    """(onsets, alarm days, days) on the upset-free days [start, stop)."""
    if stop <= start:
        return 0, 0, 0
    a = np.asarray(alarm, dtype=bool)
    return int(onset_mask(a)[start:stop].sum()), int(a[start:stop].sum()), stop - start


# --- tables ---------------------------------------------------------------------------------------
def detections(runs: list[Run], method: str, window: int = WINDOW) -> pd.DataFrame:
    """One row per upset run: detected within `window`, delay, class named, lead before a loss."""
    rows = []
    for r in runs:
        if r.t0 is None:
            continue
        a = r.alarm[method]
        d = first_onset(a, r.t0, min(r.t0 + window, r.n))
        row = {"kind": r.kind, "seed": r.seed, "missing": r.missing, "method": method, "t0": r.t0,
               "yield_loss_day": r.loss_day, "material_loss": r.material, "detected": d is not None,
               "delay": np.nan if d is None else d - r.t0, "class": None,
               "before_loss": np.nan, "lead": np.nan, "chance_window": np.nan}
        if d is not None and method in r.cls:
            row["class"] = named_class(r.cls[method], d)
        if r.material and r.loss_day is not None:
            b = first_onset(a, r.t0, r.loss_day)
            row["before_loss"] = b is not None
            row["lead"] = np.nan if b is None else r.loss_day - b
            row["chance_window"] = r.loss_day - r.t0
        rows.append(row)
    return pd.DataFrame(rows)


def chance_curve(runs: list[Run], method: str) -> np.ndarray:
    """p[w]: share of fake w-day windows on upset-free days that contain an onset (w = 0..MAX_W)."""
    hits = np.zeros(MAX_W + 1)
    wins = np.zeros(MAX_W + 1)
    for r in runs:
        a = r.alarm[method]
        for w in range(1, MAX_W + 1):
            h, n = window_hits(a, WARMUP, r.stop, w)
            hits[w] += h
            wins[w] += n
    return np.divide(hits, wins, out=np.zeros_like(hits), where=wins > 0)


def run_stats(runs: list[Run], method: str, window: int = WINDOW, alarm_of=None) -> pd.DataFrame:
    """Per run: detection within the window, chance windows and false-alarm counts (for the summary
    and the bootstrap)."""
    rows = []
    for r in runs:
        a = r.alarm[method] if alarm_of is None else alarm_of(r)
        h, nw = window_hits(a, WARMUP, r.stop, window)
        on, ad, days = free_counts(a, WARMUP, r.stop)
        det = np.nan
        if r.t0 is not None:
            det = float(first_onset(a, r.t0, min(r.t0 + window, r.n)) is not None)
        rows.append({"kind": r.kind, "seed": r.seed, "process": r.kind in PROCESS,
                     "sensor": r.kind in SENSOR, "detected": det, "chance_hits": h, "chance_windows": nw,
                     "onsets": on, "alarm_days": ad, "free_days": days, "normal_year": r.t0 is None})
    return pd.DataFrame(rows)


def _rates(s: pd.DataFrame) -> dict:
    proc, sens = s[s["process"]], s[s["sensor"]]
    det = proc["detected"].mean()
    ch = s["chance_hits"].sum() / max(s["chance_windows"].sum(), 1)
    return {"detected_process": det, "detected_sensor": sens["detected"].mean(), "chance": ch,
            "excess_process": det - ch,
            "false_alarms_per_year": 365.0 * s["onsets"].sum() / max(s["free_days"].sum(), 1),
            "alarm_day_share": s["alarm_days"].sum() / max(s["free_days"].sum(), 1)}


def bootstrap(s: pd.DataFrame, n: int = N_BOOT, seed: int = 0) -> dict:
    """95 % intervals of the summary rates, resampling runs with replacement within each kind
    (every run has its own feed history, so runs are independent)."""
    rng = np.random.default_rng(seed)
    s = s.reset_index(drop=True)
    groups = [g.index.to_numpy() for _, g in s.groupby("kind")]
    draws = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        draws.append(_rates(s.loc[idx]))
    d = pd.DataFrame(draws)
    out = {}
    for c in d:
        out[f"{c}_lo"] = float(d[c].quantile(0.025))
        out[f"{c}_hi"] = float(d[c].quantile(0.975))
    return out


def _chance_before(p: np.ndarray, loss: pd.DataFrame) -> float:
    if not len(loss):
        return np.nan
    return float(np.mean(p[np.clip(loss["chance_window"].astype(int), 0, MAX_W)]))


def summary(runs: list[Run], methods: list[str], classifiers: list[str], boot: bool = True) -> pd.DataFrame:
    """One row per method at its published operating point."""
    rows = []
    for m in methods:
        s = run_stats(runs, m)
        row = {"method": m, **_rates(s)}
        free, pre = s[s["normal_year"]], s[~s["normal_year"]]
        for name, g in (("normal_years", free), ("before_t0", pre)):
            row[f"false_alarms_per_year_{name}"] = 365.0 * g["onsets"].sum() / max(g["free_days"].sum(), 1)
            row[f"alarm_day_share_{name}"] = g["alarm_days"].sum() / max(g["free_days"].sum(), 1)
        det = detections(runs, m)
        proc = det[det["kind"].isin(PROCESS)]
        row["median_delay_d"] = proc.loc[proc["detected"], "delay"].median()
        p = chance_curve(runs, m)
        loss = det[det["kind"].isin(LEAD_KINDS) & det["material_loss"] & det["yield_loss_day"].notna()]
        row["material_loss_runs"] = len(loss)
        row["before_loss"] = loss["before_loss"].astype(float).mean() if len(loss) else np.nan
        row["chance_before_loss"] = _chance_before(p, loss)
        row["median_lead_d"] = loss["lead"].median() if len(loss) else np.nan
        r60 = _rates(run_stats(runs, m, LONG_WINDOW))
        row["detected_process_60d"], row["chance_60d"] = r60["detected_process"], r60["chance"]
        if m in classifiers:
            row.update(class_scores(det))
        if boot:
            row.update(bootstrap(s))
        rows.append(row)
    return pd.DataFrame(rows)


def by_class(runs: list[Run], methods: list[str]) -> pd.DataFrame:
    """Per method and upset type: detection within the window, delay, and (only where the upset
    costs a material share of methane) alarms before the loss and the lead time."""
    rows = []
    for m in methods:
        det = detections(runs, m)
        p = chance_curve(runs, m)
        for k, g in det.groupby("kind", sort=False):
            loss = g[g["material_loss"] & g["yield_loss_day"].notna()]
            rows.append({
                "method": m, "kind": k, "runs": len(g), "detected": g["detected"].mean(),
                "chance": p[WINDOW], "median_delay_d": g.loc[g["detected"], "delay"].median(),
                "material_loss_runs": len(loss),
                "before_loss": loss["before_loss"].astype(float).mean() if len(loss) else np.nan,
                "chance_before_loss": _chance_before(p, loss),
                "median_lead_d": loss["lead"].median() if len(loss) else np.nan,
            })
    return pd.DataFrame(rows)


def class_scores(det: pd.DataFrame) -> dict:
    g = det[det["detected"]]
    if not len(g):
        return {"accuracy": np.nan, "process_vs_sensor": np.nan, "classified": 0}
    acc = float((g["kind"] == g["class"]).mean())
    sens_true = g["kind"].isin(SENSOR)
    sens_pred = g["class"].isin(SENSOR)
    return {"accuracy": acc, "process_vs_sensor": float((sens_true == sens_pred).mean()), "classified": len(g)}


def confusion(runs: list[Run], method: str) -> pd.DataFrame:
    det = detections(runs, method)
    g = det[det["detected"]]
    labels = CLASSES[1:]
    m = pd.crosstab(pd.Categorical(g["kind"], labels), pd.Categorical(g["class"], labels), dropna=False)
    m.index.name, m.columns.name = "true", "predicted"
    return m


# --- all detectors at one false-alarm rate --------------------------------------------------------
def threshold_curve(runs: list[Run], method: str, n_thr: int = 120) -> pd.DataFrame:
    """Detection, chance and false alarms as the threshold on the method's score moves (alarm =
    score > threshold; the published operating point is threshold 1)."""
    free = np.concatenate([r.score[method][WARMUP:r.stop] for r in runs])
    free = free[np.isfinite(free)]
    qs = np.quantile(free, np.linspace(0.5, 1.0, n_thr))
    thr = np.unique(np.round(np.concatenate([qs, [1.0], np.linspace(0.2, 3.0, 57)]), 6))
    rows = []
    for t in thr:
        s = run_stats(runs, method, alarm_of=lambda r, t=t: r.score[method] > t)
        rows.append({"method": method, "threshold": float(t), **_rates(s)})
    return pd.DataFrame(rows)


def at_common_rate(curve: pd.DataFrame, rate: float = COMMON_RATE) -> dict:
    """The threshold with the highest detection among those whose false-alarm rate does not exceed
    `rate` (ties: the higher threshold). A threshold so low that the method is in alarm all the time
    has no onsets, hence no false alarms but no detections either, and is never chosen."""
    ok = curve[curve["false_alarms_per_year"] <= rate]
    if not len(ok):
        return {"method": curve["method"].iloc[0], "threshold": np.nan}
    return ok.sort_values(["detected_process", "threshold"], ascending=False).iloc[0].to_dict()
