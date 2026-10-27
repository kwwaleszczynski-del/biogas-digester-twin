"""Scores: detection, lead time before yield loss, false alarms per year, classification."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scenarios import CLASSES, PROCESS, SENSOR

WARMUP = 35        # days of history a method needs before its alarms count
WINDOW = 60        # an upset counts as detected if an alarm starts within this many days after t0
EPISODE_GAP = 7    # alarm days less than a week apart belong to one alarm episode


def onsets(alarm: np.ndarray) -> np.ndarray:
    a = np.asarray(alarm, dtype=bool)
    prev = np.concatenate([[False], a[:-1]])
    return np.where(a & ~prev)[0]


def episodes(alarm: np.ndarray, start: int, stop: int) -> int:
    """Alarm episodes beginning in [start, stop): an onset after at least a week without alarm."""
    days = np.where(np.asarray(alarm, dtype=bool))[0]
    n, last = 0, -10 ** 9
    for d in days:
        if d - last > EPISODE_GAP and start <= d < stop:
            n += 1
        last = d
    return n


def class_onsets(cls: np.ndarray) -> np.ndarray:
    """Days on which a classifier names a new upset (from normal or from another class)."""
    c = np.asarray(cls, dtype=object)
    prev = np.concatenate([["normal"], c[:-1]])
    return np.where((c != "normal") & (c != prev))[0]


def detection(alarm: np.ndarray, cls: np.ndarray | None, sc, yield_loss_day: int | None) -> dict:
    """First alarm onset in [t0, t0 + WINDOW) (an alarm already running at t0 does not count;
    for a classifier a new diagnosis is an onset too)."""
    on = onsets(alarm) if cls is None else class_onsets(cls)
    hits = on[(on >= sc.t0) & (on < sc.t0 + WINDOW)]
    out = {"kind": sc.kind, "seed": sc.seed, "t0": sc.t0, "severity": sc.severity,
           "yield_loss_day": yield_loss_day, "detected": len(hits) > 0, "delay": np.nan,
           "lead": np.nan, "before_loss": False, "class": None}
    if len(hits):
        d = int(hits[0])
        out["delay"] = d - sc.t0
        if yield_loss_day is not None:
            out["lead"] = yield_loss_day - d
            out["before_loss"] = d < yield_loss_day
        if cls is not None:
            # the class named over the first three alarm days (majority, ties to the first)
            names = [c for c in cls[d:d + 3] if c != "normal"] or [cls[d]]
            out["class"] = max(dict.fromkeys(names), key=names.count)
    return out


def false_alarms(alarm: np.ndarray, stop: int) -> tuple[int, int]:
    """(alarm episodes, normal days) between the warm-up and `stop` (t0 or the end of a normal run)."""
    return episodes(alarm, WARMUP, stop), max(stop - WARMUP, 0)


def summary(det: pd.DataFrame, fa: pd.DataFrame) -> pd.DataFrame:
    """One row per method: detection, delay, lead time before yield loss, false alarms per year."""
    rows = []
    for m, g in det.groupby("method", sort=False):
        proc = g[g["kind"].isin(PROCESS)]
        sens = g[g["kind"].isin(SENSOR)]
        loss = proc[proc["yield_loss_day"].notna()]
        f = fa[fa["method"] == m]
        rows.append({
            "method": m,
            "detected_process": proc["detected"].mean(),
            "detected_sensor": sens["detected"].mean(),
            "median_delay_d": proc.loc[proc["detected"], "delay"].median(),
            "before_yield_loss": loss["before_loss"].mean() if len(loss) else np.nan,
            "median_lead_d": loss.loc[loss["detected"], "lead"].median() if len(loss) else np.nan,
            "false_alarms_per_year": 365.0 * f["episodes"].sum() / max(f["days"].sum(), 1),
        })
    return pd.DataFrame(rows)


def by_class(det: pd.DataFrame) -> pd.DataFrame:
    g = det.groupby(["method", "kind"], sort=False)
    out = g.agg(detected=("detected", "mean"), delay=("delay", "median"),
                before_loss=("before_loss", "mean"), lead=("lead", "median"),
                with_loss=("yield_loss_day", lambda s: s.notna().mean())).reset_index()
    return out


def confusion(det: pd.DataFrame, method: str) -> pd.DataFrame:
    g = det[(det["method"] == method) & det["detected"]]
    labels = CLASSES[1:]
    m = pd.crosstab(pd.Categorical(g["kind"], labels), pd.Categorical(g["class"], labels),
                    dropna=False)
    m.index.name, m.columns.name = "true", "predicted"
    return m


def class_scores(det: pd.DataFrame, method: str) -> dict:
    g = det[(det["method"] == method) & det["detected"]]
    if not len(g):
        return {"accuracy": np.nan, "process_vs_sensor": np.nan, "n": 0}
    acc = float((g["kind"] == g["class"]).mean())
    sens_true = g["kind"].isin(SENSOR)
    sens_pred = g["class"].isin(SENSOR)
    return {"accuracy": acc, "process_vs_sensor": float((sens_true == sens_pred).mean()), "n": len(g)}
