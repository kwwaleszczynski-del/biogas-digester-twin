"""Real-data check on open farm-scale digester monitoring data (false alarms and plausibility only).

Data: D. Mulat (ILRI), "Process monitoring and weather datasets on farm-scale biogas digesters",
Mendeley Data V1, 2025, doi:10.17632/gk3f363sfg.1, CC BY 4.0. Four small manure-fed digesters
(two flexible-bag, two fixed-dome) logged daily in 2024: gas meter reading, manure and water fed,
digester and air temperature, operator comments. pH, VFA and FOS/TAC columns are empty, CH4 has
one value per digester, so only the gas- and temperature-based methods can be applied.

Changes made to the data (all in code, the raw file is not modified): dates repaired where the
spreadsheet swapped day and month, the meter reading differenced per elapsed day, negative or
implausible differences treated as metering errors, feed strings ("20kg", "140kg", "nl") parsed.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev

SHEETS = {"RI-FLEX": "R1 flexible bag", "R2-FLEX": "R2 flexible bag",
          "R3-FIXED DOME": "R3 fixed dome", "R4-FIXED DOME": "R4 fixed dome"}
FILE = "Farm-scale Biodigester 2024_Performance.xlsx"
CALIBRATION_DAYS = 56
OPERATIONAL = re.compile(r"repa|pipe|valve|meter|revers|swap|swarp|flar|experiment", re.I)


def find_file(root: Path) -> Path:
    hits = list(root.rglob(FILE))
    if not hits:
        raise FileNotFoundError(f"{FILE} not found under {root}; run `python -m biogas_twin fetch`")
    return hits[0]


def _dates(col: pd.Series) -> list[datetime | None]:
    """Chronological dates; spreadsheet values whose day and month were swapped are repaired."""
    out, prev = [], None
    for v in col:
        cands = []
        if isinstance(v, (datetime, pd.Timestamp)):
            v = pd.Timestamp(v).to_pydatetime()
            cands.append(v)
            if v.day <= 12:
                try:
                    cands.append(v.replace(month=v.day, day=v.month))
                except ValueError:
                    pass
        elif isinstance(v, str) and re.match(r"\d{1,2}/\d{1,2}/\d{4}", v.strip()):
            d, m, y = (int(x) for x in v.strip().split("/"))
            cands.append(datetime(y, m, d))
        if not cands:
            out.append(None)
            continue
        if prev is None:
            best = min(cands)
        else:
            after = [c for c in cands if c > prev]
            best = min(after, key=lambda c: c - prev) if after else min(cands, key=lambda c: abs(c - prev))
        out.append(best)
        prev = best
    return out


def _feed(v) -> float:
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*kg", str(v))
    return float(m.group(1)) if m else 0.0


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    """Daily table of one digester: gas per day (meter units, litres), feed, temperatures, comments."""
    raw = pd.read_excel(path, sheet_name=sheet)
    raw.columns = [str(c).strip() for c in raw.columns]
    dcol = raw.columns[0]
    tcol = next((c for c in raw.columns if c.lower().startswith("digestertemp")), None)
    df = pd.DataFrame({
        "date": _dates(raw[dcol]),
        "meter": pd.to_numeric(raw["Biogas Reading (No.)time 09:10am"], errors="coerce"),
        "feed_kg": raw["Manure fed (kg)"].map(_feed),
        "air_temp": pd.to_numeric(raw["Air Temp (C)time 09:15am"], errors="coerce"),
        "temp": pd.to_numeric(raw[tcol], errors="coerce") if tcol else np.nan,
        "comment": raw["Comment"].astype("string").str.strip(),
    }).dropna(subset=["date"])
    # physically implausible logger values (0 C, 55 C in an unheated digester) are typing errors
    df["temp"] = df["temp"].where(df["temp"].between(10, 45))
    df["air_temp"] = df["air_temp"].where(df["air_temp"].between(0, 45))
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    s = df.set_index("date").reindex(days)
    # meter differenced per elapsed day; a reading that goes backwards or jumps beyond 5x the
    # median daily gas is a metering event, not gas
    reading = s["meter"].interpolate(limit_area="inside")
    gas = reading.diff()
    med = gas[gas > 0].median()
    s["gas"] = gas.where((gas >= 0) & (gas < 5 * med))
    s["meter_error"] = (gas < 0) | (gas >= 5 * med)
    s["feed_kg"] = s["feed_kg"].fillna(0.0)
    s.index.name = "date"
    return s.reset_index()


def start_day(d: pd.DataFrame) -> int:
    """First day after start-up: the 7-day mean gas reaches a quarter of the digester's median."""
    g7 = d["gas"].rolling(7, min_periods=4).mean()
    ok = np.where(g7 >= 0.25 * d["gas"][d["gas"] > 0].median())[0]
    return int(ok[0]) if len(ok) else 0


def monitor(d: pd.DataFrame, lam: float = 0.2, L: float = 3.0, k: float = 0.5, h: float = 5.0) -> dict:
    """The twin study's gas-based detectors, calibrated on the digester's own first 8 weeks after
    start-up."""
    feed7 = d["feed_kg"].rolling(7, min_periods=1).mean()
    sy = d["gas"].rolling(7, min_periods=4).mean() / feed7.where(feed7 > 0)  # gas per kg manure
    s0 = start_day(d)
    cal = sy.iloc[s0:s0 + CALIBRATION_DAYS].dropna()
    mu, sd = cal.mean(), cal.std()
    z = ((sy - mu) / sd).ffill().fillna(0.0).to_numpy()
    e, hi, lo = 0.0, 0.0, 0.0
    ewma = np.zeros(len(z), bool)
    cusum = np.zeros(len(z), bool)
    lim = L * np.sqrt(lam / (2 - lam))
    for i, v in enumerate(z):
        e = lam * v + (1 - lam) * e
        ewma[i] = abs(e) > lim
        hi, lo = max(0.0, hi + v - k), max(0.0, lo - v - k)
        cusum[i] = hi > h or lo > h
        if cusum[i]:
            hi = lo = 0.0
    out = {"EWMA yield": ewma, "CUSUM yield": cusum}
    # flat or backwards meter: the data-quality rule of the expert system
    flat = (d["gas"].fillna(-1) == 0) & (d["feed_kg"].rolling(3, min_periods=1).sum() > 0)
    out["Meter rule"] = (flat | d["meter_error"]).to_numpy()
    return out


def temperature_sensitivity(d: pd.DataFrame, events: pd.Series) -> dict | None:
    """theta in gas ~ exp(theta * T) from weekly means (feed constant), with a 95 % interval.
    Days within three days of a logged repair, valve or meter event are left out."""
    if d["temp"].notna().sum() < 60:
        return None
    near = events.astype(float).rolling(7, center=True, min_periods=1).max().astype(bool)
    w = d.loc[~near.to_numpy()].set_index("date")[["gas", "temp", "feed_kg"]].resample("W").mean()
    w = w[(w["gas"] > 0) & w["temp"].notna() & (w["feed_kg"] > 0)]
    if len(w) < 12:
        return None
    X = np.column_stack([np.ones(len(w)), w["temp"]])
    y = np.log(w["gas"] / w["feed_kg"])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(w) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return {"theta": float(beta[1]), "lo": float(beta[1] - 1.96 * se), "hi": float(beta[1] + 1.96 * se),
            "weeks": len(w), "temp_range": [float(w["temp"].min()), float(w["temp"].max())]}


def episode_starts(alarm: np.ndarray, start: int) -> list[int]:
    """First day of each alarm episode (no alarm in the week before) from `start` on."""
    days = np.where(np.asarray(alarm, dtype=bool))[0]
    out, last = [], -10 ** 9
    for d in days:
        if d - last > ev.EPISODE_GAP and d >= start:
            out.append(int(d))
        last = d
    return out


def check(root: Path) -> dict:
    """False alarms per year after calibration, their overlap with logged operational events, the
    temperature sensitivity of gas production."""
    path = find_file(root)
    rows, series, sens = [], {}, {}
    for sheet, name in SHEETS.items():
        d = read_sheet(path, sheet)
        alarms = monitor(d)
        first = start_day(d) + CALIBRATION_DAYS
        # operator notes only; the meter rule's own trigger (an impossible reading) is not an event
        events = d["comment"].fillna("").str.contains(OPERATIONAL)
        ev_days = np.where(events.to_numpy())[0]
        n_days = len(d) - first
        for m, a in alarms.items():
            starts = episode_starts(a, first)
            near = sum(1 for o in starts if len(ev_days) and np.min(np.abs(ev_days - o)) <= 3)
            rows.append({"digester": name, "method": m, "days": n_days, "episodes": len(starts),
                         "per_year": 365.0 * len(starts) / n_days,
                         "share_of_days_in_alarm": float(np.mean(a[first:])), "near_logged_event": near})
        sens[name] = temperature_sensitivity(d, events)
        series[name] = {
            "dates": d["date"].dt.strftime("%Y-%m-%d").tolist(),
            "gas": [None if not np.isfinite(v) else round(float(v), 1) for v in d["gas"]],
            "temp": [None if not np.isfinite(v) else round(float(v), 1) for v in d["temp"]],
            "feed": [round(float(v), 1) for v in d["feed_kg"]],
            "events": [int(i) for i in ev_days],
            "monitored_from": first,
            "alarms": {m: [int(i) for i in np.where(a)[0]] for m, a in alarms.items()},
        }
    return {"table": pd.DataFrame(rows), "series": series, "sensitivity": sens}
