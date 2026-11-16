"""The comparison: simulate training and test runs, observe them, fit, apply every method, score.

Training: 2 normal years and 8 instances of every upset type (seeds 0-7).
Test: 3 normal years and 20 instances of every upset type (seeds 101-120), never seen in fitting or
in setting thresholds (seed 100 was used in development runs and is left out). Every run has its own
feed history.
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
from .am2 import EKF, ReducedModel, calibrate, ekf_alarm, perturbed
from .methods import (GBMClassifier, PCAMonitor, YieldCharts, day_labels, fostac_absolute_score,
                      fostac_relative_score, prepare, rules)
from .plant import Plant
from .scenarios import CLASSES, PROCESS, SENSOR, draw, methane_lost, run
from .sensors import daily, feed_log, lab, online, vfa_mg_hac

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BIOGAS_TWIN_DATA", ROOT / "data")) / "processed"
RESULTS = ROOT / "results"

KINDS = PROCESS + SENSOR
TRAIN_NORMAL, TEST_NORMAL = [1001, 1002], [2001, 2002, 2003]
TRAIN_SEEDS, TEST_SEEDS = range(0, 8), range(101, 121)
PRE, POST = 90, 90
MISSING = [0.1, 0.5, 1.0]
METHODS = ["FOS/TAC > 0.3", "FOS/TAC +30 %", "EWMA yield", "CUSUM yield", "PCA T2/SPE",
           "Expert rules", "Gradient boosting", "EKF twin"]
CLASSIFIERS = ["Expert rules", "Gradient boosting", "EKF twin"]
# checks run on the 10 %-missing observation only: gradient boosting without the feed log, and a
# filter whose model parameters are off by ~20 % and re-zeroed on each run's first weeks
VARIANTS = ["Gradient boosting (no feed log)", "EKF twin (perturbed)"]

_PLANT: dict = {}


def plant_state():
    if not _PLANT:
        pl = Plant()
        xs = pl.steady_state()
        p, x0 = calibrate(pl, xs)
        q, xq = perturbed(p, x0)
        _PLANT.update(plant=pl, xs=xs, rm=ReducedModel(p), x0=x0, rm_pert=ReducedModel(q), x0_pert=xq)
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
    gbm_nofeed: GBMClassifier
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
    gbm_nofeed = GBMClassifier(feed_log=False).fit(frames, labels)
    return Fitted(charts, pca, gbm, gbm_nofeed, float(plant_state()["x0"][1]))


def _detector(score: np.ndarray) -> tuple[np.ndarray, None, np.ndarray]:
    s = np.asarray(score, dtype=float)
    return s > 1.0, None, s


def apply_methods(d: pd.DataFrame, f: Fitted, variants: bool = False) -> tuple[dict, pd.DataFrame]:
    """Every method on one run: {method: (alarm, class or None, score)} and the filter's estimates."""
    x = prepare(d)
    out = {
        "FOS/TAC > 0.3": _detector(fostac_absolute_score(x, 0.3)),
        "FOS/TAC +30 %": _detector(fostac_relative_score(x)),
        "EWMA yield": _detector(f.charts.ewma_score(x)),
        "CUSUM yield": _detector(f.charts.cusum_score(x)),
        "PCA T2/SPE": _detector(f.pca.score(x)),
    }
    out["Expert rules"] = rules(x, f.charts)
    out["Gradient boosting"] = f.gbm.predict(x)
    st = plant_state()
    est = EKF(st["rm"], st["x0"]).run(d)
    out["EKF twin"] = ekf_alarm(est, x, f.s2_ref)
    if variants:
        out["Gradient boosting (no feed log)"] = f.gbm_nofeed.predict(x)
        est_p = EKF(st["rm_pert"], st["x0_pert"]).run(d)
        out["EKF twin (perturbed)"] = ekf_alarm(est_p, x, f.s2_ref, rebaseline=True)
    return out, est


def evaluate_one(args: tuple) -> dict:
    """Apply every method to one observed test run; keep the daily series for the scores."""
    kind, seed, missing, f = args
    res = load(kind, seed)
    d = observe(res, missing)
    out, est = apply_methods(d, f, variants=missing == MISSING[0])
    sc = res["scenario"]
    lost = methane_lost(res)
    rec = {"kind": kind, "seed": seed, "missing": missing, "n": len(d),
           "t0": int(sc.t0) if kind != "normal" else None, "loss_day": res["yield_loss_day"],
           "material": bool(lost >= ev.MATERIAL),
           "alarm": {m: np.asarray(a, dtype=bool) for m, (a, c, s) in out.items()},
           "cls": {m: np.asarray(c, dtype=object) for m, (a, c, s) in out.items() if c is not None},
           "score": {m: np.asarray(s, dtype=np.float32) for m, (a, c, s) in out.items()}}
    replay = None
    if missing == MISSING[0] and seed in (TEST_SEEDS[0], TEST_NORMAL[0]):
        replay = replay_record(res, d, out, est)
    return {"run": rec, "replay": replay}


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
        "alarms": {m: [int(i) for i in np.where(np.asarray(out[m][0])[:n])[0]] for m in METHODS},
        "classes": {m: [str(v) for v in np.asarray(out[m][1])[:n]] for m in CLASSIFIERS},
    }


def run_study(workers: int = 6) -> dict:
    """The whole comparison; returns the result tables and writes them to results/."""
    plant_state()
    all_specs = specs("train") + specs("test")
    log.info("simulating %d runs (cached in %s)", len(all_specs), DATA / "sim")
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(simulate_one, all_specs))
    train, f = train_and_fit()
    jobs = evaluation_jobs(f)
    results = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(evaluate_one, jobs, chunksize=2)):
            results.append(r)
            if i % 50 == 0:
                log.info("evaluated %d/%d", i + 1, len(jobs))
    return finish(train, f, results)


def train_and_fit() -> tuple[list[tuple[dict, pd.DataFrame]], Fitted]:
    train = []
    for kind, seed in specs("train"):
        res = load(kind, seed)
        train.append((res, observe(res)))
    f = fit(train)
    log.info("fitted: SMY %.3f +- %.3f Nm3/kg oDM; PCA %d components", f.charts.mu, f.charts.sd, f.pca.k)
    return train, f


def evaluation_jobs(f: Fitted) -> list[tuple]:
    """Every test run at every lab-missing level, in a fixed order."""
    return [(k, s, m, f) for m in MISSING for k, s in specs("test")]


def finish(train: list[tuple[dict, pd.DataFrame]], f: Fitted, results: list[dict]) -> dict:
    """Collect the evaluated runs (in job order), add the unseen-type experiment and the ground-truth
    table, write them."""
    runs = [r["run"] for r in results]
    replays = [r["replay"] for r in results if r["replay"]]
    unseen = unseen_classes(train, f)
    truth = scenario_table()
    RESULTS.mkdir(exist_ok=True)
    truth.to_csv(RESULTS / "scenarios.csv", index=False)
    unseen.to_csv(RESULTS / "gbm_unseen_class.csv", index=False)
    tables = {"runs": runs, "truth": truth, "unseen": unseen, "replays": replays}
    with open(DATA / "study.pkl", "wb") as fh:
        pickle.dump(tables, fh)
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
            t0 = load(k, s)["scenario"].t0
            row = {"kind": k, "seed": s}
            for tag, model in (("unseen", gbm), ("seen", f.gbm)):
                a, c, _ = model.predict(x)
                d = ev.first_onset(a, t0, t0 + ev.WINDOW)
                row[f"detected_{tag}"] = d is not None
                row[f"delay_{tag}"] = np.nan if d is None else d - t0
                row[f"class_{tag}"] = None if d is None else ev.named_class(c, d)
            rows.append(row)
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
        r7 = pd.Series(ratio).rolling(7, min_periods=1).mean().to_numpy()
        lost = methane_lost(res)
        rows.append({"kind": kind, "seed": seed, "t0": sc.t0, "t_end": sc.t_end,
                     "severity": sc.severity, "yield_loss_day": res["yield_loss_day"],
                     "loss_after_d": (res["yield_loss_day"] - sc.t0) if res["yield_loss_day"] else np.nan,
                     "max_vfa_mg_l": vfa.max(), "min_yield_ratio_7d": r7[sc.t0:].min(),
                     "lost_ch4_share": lost, "material_loss": bool(lost >= ev.MATERIAL)})
    return pd.DataFrame(rows)
