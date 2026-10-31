"""Result tables (results/), figures (docs/img) and the dashboard data (docs/data.js)."""
from __future__ import annotations

import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import evaluate as ev
from .scenarios import CLASSES, LABELS, PROCESS, SENSOR
from .study import CLASSIFIERS, DATA, METHODS, MISSING, RESULTS, ROOT

DOCS = ROOT / "docs"
IMG = DOCS / "img"
FAMILY = {
    "FOS/TAC > 0.3": "lab", "FOS/TAC +30 %": "lab",
    "EWMA yield": "stat", "CUSUM yield": "stat", "PCA T2/SPE": "stat",
    "Expert rules": "model", "Gradient boosting": "model", "EKF twin": "model",
}
FAMILY_LABEL = {"lab": "lab FOS/TAC (industry practice)", "stat": "statistical monitoring",
                "model": "diagnosis: rules, ML, state estimation"}
COLORS = {"lab": "#eb6834", "stat": "#2a78d6", "model": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)


def load_study() -> dict:
    # this project's own cache, written by study.run_study
    with open(DATA / "study.pkl", "rb") as f:
        return pickle.load(f)


def tables(st: dict) -> dict:
    det, fa = st["det"], st["fa"]
    base = det[det["missing"] == MISSING[0]]
    fbase = fa[fa["missing"] == MISSING[0]]
    summ = ev.summary(base, fbase)
    cls = pd.DataFrame([{"method": m, **ev.class_scores(base, m)} for m in CLASSIFIERS])
    summ = summ.merge(cls[["method", "accuracy", "process_vs_sensor"]], on="method", how="left")
    summ = summ.set_index("method").loc[METHODS].reset_index()
    rob = pd.concat([ev.summary(det[det["missing"] == m], fa[fa["missing"] == m]).assign(lab_missing=m)
                     for m in MISSING])
    rob_cls = pd.DataFrame([{"method": c, "lab_missing": m, **ev.class_scores(det[det["missing"] == m], c)}
                            for m in MISSING for c in CLASSIFIERS])
    rob = rob.merge(rob_cls[["method", "lab_missing", "accuracy"]], on=["method", "lab_missing"], how="left")
    byc = ev.by_class(base)
    conf = {m: ev.confusion(base, m) for m in CLASSIFIERS}
    return {"summary": summ, "robustness": rob, "by_class": byc, "confusion": conf,
            "unseen": st.get("unseen"), "truth": st["truth"]}


def write_tables(t: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    t["summary"].to_csv(RESULTS / "method_summary.csv", index=False, float_format="%.4g")
    t["robustness"].to_csv(RESULTS / "robustness_lab.csv", index=False, float_format="%.4g")
    t["by_class"].to_csv(RESULTS / "method_by_class.csv", index=False, float_format="%.4g")
    long = pd.concat([c.stack().rename("n").reset_index().assign(method=m)
                      for m, c in t["confusion"].items()])
    long[["method", "true", "predicted", "n"]].to_csv(RESULTS / "confusion.csv", index=False)


# --- figures ----------------------------------------------------------------------------------------
def fig_tradeoff(summ: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ycol, ylab in [(axes[0], "median_delay_d", "median detection delay [days after the cause]"),
                           (axes[1], "before_yield_loss", "share of yield losses alarmed before they start")]:
        _style(ax)
        for _, r in summ.iterrows():
            c = COLORS[FAMILY[r["method"]]]
            ax.scatter(r["false_alarms_per_year"], r[ycol], s=60, color=c, edgecolor="white", lw=2, zorder=3)
            # the two lab rules can sit at the same height next to each other: one label goes below its point
            below = r["method"] == "FOS/TAC > 0.3" and ycol == "median_delay_d"
            ax.annotate(r["method"], (r["false_alarms_per_year"], r[ycol]), xytext=(6, -12 if below else 4),
                        textcoords="offset points", fontsize=8.5, color=INK2)
        ax.margins(y=0.08)
        ax.set_xlabel("false alarm episodes per year of normal operation", color=INK2, fontsize=9)
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
    axes[1].set_ylim(-0.03, 1.05)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[k], label=v) for k, v in FAMILY_LABEL.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(IMG / "tradeoff.png", dpi=150)
    plt.close(fig)


def fig_lead(det: pd.DataFrame) -> None:
    d = det[(det["missing"] == MISSING[0]) & det["kind"].isin(PROCESS) & det["yield_loss_day"].notna()]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    _style(ax)
    rng = np.random.default_rng(0)
    for i, m in enumerate(METHODS):
        g = d[d["method"] == m]
        lead = g["lead"].to_numpy(dtype=float)
        miss = (~g["detected"]).sum()
        y = i + rng.uniform(-0.25, 0.25, len(lead))
        ax.scatter(np.clip(lead, -40, 40), y, s=14, color=COLORS[FAMILY[m]], alpha=0.75, lw=0)
        med = np.nanmedian(lead) if np.isfinite(lead).any() else np.nan
        if np.isfinite(med):
            ax.plot([med, med], [i - 0.35, i + 0.35], color=INK, lw=2)
        ax.text(42, i, f"{miss} missed", va="center", fontsize=8, color=MUTED)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(METHODS)), METHODS)
    ax.invert_yaxis()
    ax.set_xlim(-42, 52)
    ax.set_xlabel("days from the first alarm to the start of yield loss (positive = alarm came first)",
                  color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "lead_time.png", dpi=150)
    plt.close(fig)


def fig_confusion(conf: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    labels = CLASSES[1:]
    short = ["overload", "acidif.", "ammonia", "heating", "substrate", "CH4 drift", "flow stuck"]
    for ax, (m, c) in zip(axes, conf.items()):
        a = c.to_numpy(dtype=float)
        rows = a.sum(1, keepdims=True)
        share = np.divide(a, rows, out=np.zeros_like(a), where=rows > 0)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if a[i, j]:
                    ax.text(j, i, int(a[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if share[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), short, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), short, fontsize=8)
        acc = np.trace(a) / max(a.sum(), 1)
        ax.set_title(f"{m}: {acc:.0%} of detected upsets named right", fontsize=9.5, color=INK)
        ax.set_xlabel("named at the first alarm", fontsize=8.5, color=INK2)
    axes[0].set_ylabel("true upset", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(IMG / "confusion.png", dpi=150)
    plt.close(fig)


def fig_robustness(rob: pd.DataFrame) -> None:
    ms = ["FOS/TAC +30 %", "Expert rules", "Gradient boosting", "EKF twin"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, lab in [(axes[0], "detected_process", "process upsets detected"),
                         (axes[1], "median_delay_d", "median delay [days]")]:
        _style(ax)
        for i, m in enumerate(ms):
            g = rob[rob["method"] == m].sort_values("lab_missing")
            ax.plot(100 * g["lab_missing"], g[col], marker="o", lw=2, ms=7, label=m,
                    color=["#eb6834", "#2a78d6", "#1baf7a", "#6250d6"][i])
        ax.set_xlabel("lab samples missing [%]", fontsize=9, color=INK2)
        ax.set_ylabel(lab, fontsize=9, color=INK2)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(IMG / "robustness.png", dpi=150)
    plt.close(fig)


def fig_replay(rep: dict) -> None:
    n = len(rep["biogas"])
    t = np.arange(n)
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1, 1, 1.4]})
    f = lambda a: np.array([np.nan if v is None else v for v in a], dtype=float)
    ax = axes[0]
    _style(ax)
    ax.plot(t, f(rep["true_ch4"]), color="#2a78d6", lw=2, label="methane, true [Nm3/d]")
    ax.plot(t, f(rep["healthy_ch4"]), color=MUTED, lw=1.5, ls="--", label="healthy twin, same feed")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axes[1]
    _style(ax)
    ax.plot(t, f(rep["pH"]), color="#2a78d6", lw=1.5, label="pH (online)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axes[2]
    _style(ax)
    ft = f(rep["fos_tac"])
    ax.plot(t[np.isfinite(ft)], ft[np.isfinite(ft)], "o", color="#eb6834", ms=4, label="FOS/TAC (lab)")
    ax2 = ax.twinx()
    ax2.set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[3]
    _style(ax)
    for i, m in enumerate(METHODS):
        days = rep["alarms"].get(m, [])
        ax.scatter(days, np.full(len(days), i), marker="|", s=60, color=COLORS[FAMILY[m]])
    ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8)
    ax.invert_yaxis()
    for a in axes:
        if rep["t0"] is not None:
            a.axvline(rep["t0"], color=INK, lw=1)
        if rep["yield_loss_day"] is not None:
            a.axvline(rep["yield_loss_day"], color="#e34948", lw=1, ls="--")
    axes[0].set_title(f"{LABELS[rep['kind']]}: cause at day {rep['t0']} (solid), yield loss from day "
                      f"{rep['yield_loss_day']} (dashed), alarms below", fontsize=10, color=INK)
    axes[-1].set_xlabel("day", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(IMG / f"replay_{rep['kind']}.png", dpi=150)
    plt.close(fig)


# --- real data --------------------------------------------------------------------------------------
def write_real(res: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    res["table"].to_csv(RESULTS / "real_data_alarms.csv", index=False, float_format="%.4g")
    weekly = {}
    for name, s in res["series"].items():
        d = pd.DataFrame({"date": pd.to_datetime(s["dates"]), "gas": s["gas"], "temp": s["temp"]})
        d["gas"] = pd.to_numeric(d["gas"])
        d["temp"] = pd.to_numeric(d["temp"])
        w = d.set_index("date").resample("W").mean()
        to_week = lambda days: sorted({int((pd.Timestamp(s["dates"][i]) - w.index[0]).days // 7 + 1)
                                       for i in days if i < len(s["dates"])})
        weekly[name] = {
            "week": w.index.strftime("%Y-%m-%d").tolist(),
            "gas": [None if not np.isfinite(v) else round(float(v), 0) for v in w["gas"]],
            "temp": [None if not np.isfinite(v) else round(float(v), 1) for v in w["temp"]],
            "event_weeks": to_week(s["events"]),
            "monitored_from_week": to_week([s["monitored_from"]])[0] if s["monitored_from"] < len(s["dates"]) else None,
            "alarm_weeks": {m: to_week(a) for m, a in s["alarms"].items()},
        }
    out = {"table": res["table"].round(3).to_dict(orient="records"), "weekly": weekly,
           "sensitivity": res["sensitivity"]}
    (RESULTS / "real_data.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharey=False)
    for ax, (name, w) in zip(axes.ravel(), weekly.items()):
        _style(ax)
        x = np.arange(len(w["week"]))
        ax.plot(x, [np.nan if v is None else v for v in w["gas"]], color="#2a78d6", lw=1.8)
        top = np.nanmax([v for v in w["gas"] if v is not None] or [1])
        for k, m in enumerate(["EWMA yield", "CUSUM yield", "Meter rule"]):
            wk = w["alarm_weeks"].get(m, [])
            ax.scatter(wk, np.full(len(wk), top * (1.08 + 0.07 * k)), marker="|", s=40,
                       color=["#2a78d6", "#6250d6", "#eb6834"][k], label=m)
        ev_w = w["event_weeks"]
        ax.scatter(ev_w, np.full(len(ev_w), top * 1.32), marker="v", s=18, color="#e34948",
                   label="operator logged repair / valve / meter")
        ax.set_title(name, fontsize=9.5, color=INK)
        ax.set_xticks(x[::8], [w["week"][i][:7] for i in x[::8]], fontsize=8)
        ax.set_ylabel("gas per day (weekly mean)", fontsize=8, color=INK2)
        if w["monitored_from_week"] is not None:
            ax.axvspan(-0.5, w["monitored_from_week"] - 0.5, color=GRID, alpha=0.5, lw=0)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=4, frameon=False, fontsize=8.5)
    fig.text(0.99, 0.01, "shaded: start-up and the 8 calibration weeks", fontsize=8, color=MUTED, ha="right")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / "real_data.png", dpi=150)
    plt.close(fig)


# --- dashboard data ---------------------------------------------------------------------------------
def dashboard(st: dict, t: dict) -> dict:
    rob = t["robustness"].replace({np.nan: None})
    byc = t["by_class"].replace({np.nan: None})
    real = json.loads((RESULTS / "real_data.json").read_text(encoding="utf-8")) \
        if (RESULTS / "real_data.json").exists() else None
    order = {k: i for i, k in enumerate(CLASSES)}
    replays = sorted(st["replays"], key=lambda r: order[r["kind"]])
    truth = t["truth"].replace({np.nan: None})
    return {
        "methods": METHODS, "family": FAMILY, "family_label": FAMILY_LABEL, "classifiers": CLASSIFIERS,
        "classes": CLASSES, "labels": LABELS, "process": PROCESS, "sensor": SENSOR,
        "robustness": rob.round(4).to_dict(orient="records"),
        "by_class": byc.round(3).to_dict(orient="records"),
        "confusion": {m: c.to_numpy().tolist() for m, c in t["confusion"].items()},
        "unseen": t["unseen"].replace({np.nan: None}).to_dict(orient="records") if t["unseen"] is not None else [],
        "truth": truth.round(3).to_dict(orient="records"),
        "replays": replays,
        "real": real,
    }


def write_all() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    st = load_study()
    t = tables(st)
    write_tables(t)
    fig_tradeoff(t["summary"])
    fig_lead(st["det"])
    fig_confusion(t["confusion"])
    fig_robustness(t["robustness"])
    for rep in st["replays"]:
        if rep["kind"] in ("ammonia", "ch4_drift"):
            fig_replay(rep)
    data = dashboard(st, t)
    js = "window.DATA = " + json.dumps(data, separators=(",", ":"), allow_nan=False) + ";\n"
    (DOCS / "data.js").write_text(js, encoding="utf-8")
