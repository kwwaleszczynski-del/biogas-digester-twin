"""The comparison: simulate training and test runs, observe them, fit, apply every method, score.

Training: 2 normal years and 8 instances of every upset type (seeds 0-7).
Test: 3 normal years and 20 instances of every upset type (seeds 100-119), never seen in fitting.
Every test run is observed three times with the same online data and 10 %, 50 % and 100 % of the
lab samples missing (the robustness check).
"""
from __future__ import annotations

import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from .am2 import EKF, ReducedModel, calibrate, ekf_alarm
from .methods import (GBMClassifier, PCAMonitor, YieldCharts, day_labels, fostac_absolute,
                      fostac_relative, prepare, rules)
from .plant import Plant
from .scenarios import CLASSES, PROCESS, SENSOR, draw, run
from .sensors import daily, feed_log, lab, online, vfa_mg_hac

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BIOGAS_TWIN_DATA", ROOT / "data")) / "processed"
RESULTS = ROOT / "results"

KINDS = PROCESS + SENSOR
TRAIN_NORMAL, TEST_NORMAL = [1001, 1002], [2001, 2002, 2003]
TRAIN_SEEDS, TEST_SEEDS = range(0, 8), range(100, 120)
PRE, POST = 90, 90
MISSING = [0.1, 0.5, 1.0]
METHODS = ["FOS/TAC > 0.3", "FOS/TAC +30 %", "EWMA yield", "CUSUM yield", "PCA T2/SPE",
           "Expert rules", "Gradient boosting", "EKF twin"]
CLASSIFIERS = ["Expert rules", "Gradient boosting", "EKF twin"]

_PLANT: dict = {}


def plant_state():
    if not _PLANT:
        pl = Plant()
        xs = pl.steady_state()
        p, x0 = calibrate(pl, xs)
        _PLANT.update(plant=pl, xs=xs, rm=ReducedModel(p), x0=x0)
    return _PLANT


def specs(split: str) -> list[tuple[str, int]]:
    normal = TRAIN_NORMAL if split == "train" else TEST_NORMAL
    seeds = TRAIN_SEEDS if split == "train" else TEST_SEEDS
    return [("normal", s) for s in normal] + [(k, s) for k in KINDS for s in seeds]


def scenario(kind: str, seed: int):
    if kind == "normal":
        return draw("normal", seed, pre=365, post=0)
    return draw(kind, seed, pre=PRE, post=POST)


def sim_path(kind: str, seed: int) -> Path:
    return DATA / "sim" / f"{kind}_{seed}.pkl"


def simulate_one(spec: tuple[str, int]) -> str:
    kind, seed = spec
    path = sim_path(kind, seed)
    if path.exists():
        return str(path)
    st = plant_state()
    res = run(scenario(kind, seed), st["plant"], st["xs"])
    res["hourly"] = res["hourly"].astype({c: "float32" for c in ["biogas_Nm3h", "ch4_frac", "pH", "T_K", "ch4_Nm3h"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return str(path)


def load(kind: str, seed: int) -> dict:
    # pickle only for this project's own simulation cache in data/processed (never downloaded)
    with open(sim_path(kind, seed), "rb") as f:
        return pickle.load(f)


def observe(res: dict, missing: float = 0.1) -> pd.DataFrame:
    sc = res["scenario"]
    base = 1_000_003 * sc.seed + 7919 * CLASSES.index(sc.kind)
    on = online(res, np.random.default_rng(base + 1))
    lb = lab(res, np.random.default_rng(base + 2), missing=missing)
    fd = feed_log(res["inputs"], np.random.default_rng(base + 3))
    return daily(on, lb, fd)


@dataclass
class Fitted:
    charts: YieldCharts
    pca: PCAMonitor
    gbm: GBMClassifier
    s2_ref: float


def fit(train: list[tuple[dict, pd.DataFrame]]) -> Fitted:
    normal = [prepare(d) for r, d in train if r["scenario"].kind == "normal"]
    charts = YieldCharts().fit(normal)
    pca = PCAMonitor().fit(normal)
    frames, labels = [], []
    for r, d in train:
        x = prepare(d)
        frames.append(x)
        labels.append(day_labels(r["scenario"], len(x), r["inputs"].temperature))
    gbm = GBMClassifier().fit(frames, labels)
    return Fitted(charts, pca, gbm, float(plant_state()["x0"][1]))


def apply_methods(d: pd.DataFrame, f: Fitted) -> tuple[dict, pd.DataFrame]:
    x = prepare(d)
    out = {
        "FOS/TAC > 0.3": (fostac_absolute(x, 0.3), None),
        "FOS/TAC +30 %": (fostac_relative(x), None),
        "EWMA yield": (f.charts.ewma(x), None),
        "CUSUM yield": (f.charts.cusum(x), None),
        "PCA T2/SPE": (f.pca.alarm(x), None),
    }
    out["Expert rules"] = rules(x, f.charts)
    a, c, _ = f.gbm.predict(x)
    out["Gradient boosting"] = (a, c)
    st = plant_state()
    est = EKF(st["rm"], st["x0"]).run(d)
    out["EKF twin"] = ekf_alarm(est, x, f.s2_ref)
    return out, est


def evaluate_one(args: tuple) -> dict:
    kind, seed, missing, f = args
    res = load(kind, seed)
    d = observe(res, missing)
    out, est = apply_methods(d, f)
    sc = res["scenario"]
    det, fa = [], []
    stop = sc.t0 if kind != "normal" else len(d)
    for m, (alarm, cls) in out.items():
        e, n = ev.false_alarms(alarm, stop)
        fa.append({"method": m, "kind": kind, "seed": seed, "missing": missing, "episodes": e, "days": n})
        if kind != "normal":
            row = ev.detection(alarm, cls, sc, res["yield_loss_day"])
            row.update(method=m, missing=missing)
            det.append(row)
    replay = None
    if missing == 0.1 and seed in (TEST_SEEDS[0], TEST_NORMAL[0]):
        replay = replay_record(res, d, out, est)
    return {"det": det, "fa": fa, "replay": replay}


def replay_record(res: dict, d: pd.DataFrame, out: dict, est: pd.DataFrame) -> dict:
    sc = res["scenario"]
    st = res["states"]
    n = len(d) if sc.kind != "normal" else 180
    true_pH = res["hourly"].groupby("day")["pH"].mean().to_numpy()
    r = lambda a, k=3: [None if not np.isfinite(v) else round(float(v), k) for v in np.asarray(a, float)[:n]]
    return {
        "kind": sc.kind, "seed": sc.seed, "t0": int(sc.t0) if sc.kind != "normal" else None,
        "t_end": int(sc.t_end) if sc.kind != "normal" else None, "severity": round(sc.severity, 3),
        "yield_loss_day": res["yield_loss_day"],
        "biogas": r(d["biogas"], 0), "ch4": r(d["ch4"], 2), "pH": r(d["pH"], 3), "temp": r(d["temp"], 2),
        "fos_tac": r(d["fos_tac"], 3), "tan": r(d["tan"], 2),
        "odm": r(d["odm_t"], 2),
        "true_ch4": r(res["ch4_day"], 0), "healthy_ch4": r(res["ch4_healthy"], 0),
        "true_vfa": r([vfa_mg_hac(st.iloc[i]) for i in range(len(st))], 0),
        "true_pH": r(true_pH, 3),
        "ekf_vfa": r(est["S2"] / 70.0 * 60.05e3, 0),
        "ekf_m": r(est["m"], 3), "ekf_s": r(est["s"], 3), "ekf_b": r(est["b"], 2),
        "alarms": {m: [int(i) for i in np.where(np.asarray(a)[:n])[0]] for m, (a, c) in out.items()},
        "classes": {m: [str(v) for v in np.asarray(c)[:n]] for m, (a, c) in out.items() if c is not None},
    }


def run_study(workers: int = 6) -> dict:
    """The whole comparison; returns the result tables and writes them to results/."""
    plant_state()
    all_specs = specs("train") + specs("test")
    log.info("simulating %d runs (cached in %s)", len(all_specs), DATA / "sim")
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(simulate_one, all_specs))
    train = []
    for kind, seed in specs("train"):
        res = load(kind, seed)
        train.append((res, observe(res)))
    f = fit(train)
    log.info("fitted: SMY %.3f +- %.3f Nm3/kg oDM; PCA %d components", f.charts.mu, f.charts.sd, f.pca.k)
    jobs = [(k, s, m, f) for m in MISSING for k, s in specs("test")]
    det, fa, replays = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(evaluate_one, jobs, chunksize=2)):
            det += r["det"]
            fa += r["fa"]
            if r["replay"]:
                replays.append(r["replay"])
            if i % 50 == 0:
                log.info("evaluated %d/%d", i + 1, len(jobs))
    det = pd.DataFrame(det)
    fa = pd.DataFrame(fa)
    unseen = unseen_classes(train, f)
    truth = scenario_table()
    RESULTS.mkdir(exist_ok=True)
    det.to_csv(RESULTS / "detections.csv", index=False)
    fa.to_csv(RESULTS / "false_alarms.csv", index=False)
    truth.to_csv(RESULTS / "scenarios.csv", index=False)
    unseen.to_csv(RESULTS / "gbm_unseen_class.csv", index=False)
    tables = {"det": det, "fa": fa, "truth": truth, "unseen": unseen, "replays": replays, "fitted": f}
    with open(DATA / "study.pkl", "wb") as fh:
        pickle.dump({k: v for k, v in tables.items() if k != "fitted"}, fh)
    return tables


def unseen_classes(train: list[tuple[dict, pd.DataFrame]], f: Fitted) -> pd.DataFrame:
    """Gradient boosting trained without one upset type, tested on that type: what a supervised
    model does with an upset it has never seen."""
    test = [(k, s) for k, s in specs("test") if k != "normal"]
    obs = {(k, s): prepare(observe(load(k, s))) for k, s in test}
    rows = []
    for held in KINDS:
        frames, labels = [], []
        for r, d in train:
            if r["scenario"].kind == held:
                continue
            x = prepare(d)
            frames.append(x)
            labels.append(day_labels(r["scenario"], len(x), r["inputs"].temperature))
        gbm = GBMClassifier().fit(frames, labels)
        for (k, s), x in obs.items():
            if k != held:
                continue
            res = load(k, s)
            a, c, _ = gbm.predict(x)
            row = ev.detection(a, c, res["scenario"], res["yield_loss_day"])
            seen = f.gbm.predict(x)
            row_seen = ev.detection(seen[0], seen[1], res["scenario"], res["yield_loss_day"])
            rows.append({"kind": k, "seed": s, "detected_unseen": row["detected"],
                         "delay_unseen": row["delay"], "class_unseen": row["class"],
                         "detected_seen": row_seen["detected"], "delay_seen": row_seen["delay"]})
    return pd.DataFrame(rows)


def scenario_table() -> pd.DataFrame:
    rows = []
    for kind, seed in specs("test"):
        if kind == "normal":
            continue
        res = load(kind, seed)
        sc = res["scenario"]
        st = res["states"]
        vfa = np.array([vfa_mg_hac(st.iloc[i]) for i in range(len(st))])
        ratio = res["ch4_day"] / np.maximum(res["ch4_healthy"], 1e-9)
        ratio = ratio / ratio[sc.t0 - 14:sc.t0].mean()
        rows.append({"kind": kind, "seed": seed, "t0": sc.t0, "t_end": sc.t_end,
                     "severity": sc.severity, "yield_loss_day": res["yield_loss_day"],
                     "loss_after_d": (res["yield_loss_day"] - sc.t0) if res["yield_loss_day"] else np.nan,
                     "max_vfa_mg_l": vfa.max(), "min_yield_ratio": ratio[sc.t0:].min(),
                     "lost_ch4_share": 1 - res["ch4_day"][sc.t0:].sum() / res["ch4_healthy"][sc.t0:].sum()})
    return pd.DataFrame(rows)
