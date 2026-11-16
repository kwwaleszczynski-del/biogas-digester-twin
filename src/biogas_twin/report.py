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
from .study import CLASSIFIERS, DATA, METHODS, MISSING, RESULTS, ROOT, VARIANTS

DOCS = ROOT / "docs"
IMG = DOCS / "img"
FAMILY = {
    "FOS/TAC > 0.3": "lab", "FOS/TAC +30 %": "lab",
    "EWMA yield": "stat", "CUSUM yield": "stat", "PCA T2/SPE": "stat",
    "Expert rules": "model", "Gradient boosting": "model", "EKF twin": "model",
    "Gradient boosting (no feed log)": "model", "EKF twin (perturbed)": "model",
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


def runs_at(st: dict, missing: float) -> list[ev.Run]:
    return [ev.Run(**r) for r in st["runs"] if abs(r["missing"] - missing) < 1e-9]


def tables(st: dict) -> dict:
    base = runs_at(st, MISSING[0])
    summ = ev.summary(base, METHODS, CLASSIFIERS)
    var = ev.summary(base, VARIANTS, VARIANTS)
    rob = pd.concat([ev.summary(runs_at(st, m), METHODS, CLASSIFIERS, boot=False).assign(lab_missing=m)
                     for m in MISSING], ignore_index=True)
    byc = ev.by_class(base, METHODS + VARIANTS)
    conf = {m: ev.confusion(base, m) for m in CLASSIFIERS + VARIANTS}
    curves = pd.concat([ev.threshold_curve(base, m) for m in METHODS + VARIANTS], ignore_index=True)
    common = pd.DataFrame([ev.at_common_rate(curves[curves["method"] == m]) for m in METHODS + VARIANTS])
    det = pd.concat([ev.detections(runs_at(st, mi), m) for mi in MISSING for m in METHODS]
                    + [ev.detections(base, m) for m in VARIANTS], ignore_index=True)
    fa = pd.concat([ev.run_stats(runs_at(st, mi), m).assign(method=m, missing=mi)
                    for mi in MISSING for m in METHODS], ignore_index=True)
    fa = fa[["method", "missing", "kind", "seed", "normal_year", "free_days", "onsets", "alarm_days",
             "chance_windows", "chance_hits"]]
    return {"summary": summ, "variants": var, "robustness": rob, "by_class": byc, "confusion": conf,
            "curves": curves, "common": common, "det": det, "fa": fa,
            "unseen": st.get("unseen"), "truth": st["truth"]}


def write_tables(t: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    t["summary"].to_csv(RESULTS / "method_summary.csv", index=False, float_format="%.4g")
    t["variants"].to_csv(RESULTS / "variants.csv", index=False, float_format="%.4g")
    t["robustness"].to_csv(RESULTS / "robustness_lab.csv", index=False, float_format="%.4g")
    t["by_class"].to_csv(RESULTS / "method_by_class.csv", index=False, float_format="%.4g")
    t["curves"].to_csv(RESULTS / "threshold_curves.csv", index=False, float_format="%.4g")
    t["common"].to_csv(RESULTS / "common_false_alarm_rate.csv", index=False, float_format="%.4g")
    t["det"].to_csv(RESULTS / "detections.csv", index=False, float_format="%.4g")
    t["fa"].to_csv(RESULTS / "false_alarms.csv", index=False)
    long = pd.concat([c.stack().rename("n").reset_index().assign(method=m)
                      for m, c in t["confusion"].items()])
    long[["method", "true", "predicted", "n"]].to_csv(RESULTS / "confusion.csv", index=False)


# --- figures ----------------------------------------------------------------------------------------
def fig_tradeoff(summ: pd.DataFrame, common: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    _style(ax)
    for _, r in summ.iterrows():
        c = COLORS[FAMILY[r["method"]]]
        x = r["false_alarms_per_year"]
        ax.plot([x, x], [r["chance"], r["detected_process"]], color=c, lw=2, alpha=0.6, zorder=2)
        ax.scatter(x, r["chance"], s=34, facecolor="white", edgecolor=c, lw=1.5, zorder=3)
        ax.scatter(x, r["detected_process"], s=60, color=c, edgecolor="white", lw=2, zorder=4)
        # labels that would run into a neighbour's point or line go to the left of their point
        left = r["method"] in ("Gradient boosting", "CUSUM yield", "FOS/TAC > 0.3")
        ax.annotate(r["method"], (x, r["detected_process"]), xytext=(-6 if left else 6, 4),
                    textcoords="offset points", fontsize=8.5, color=INK2, ha="right" if left else "left",
                    zorder=5, bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85})
    ax.set_xlim(-1.8, max(summ["false_alarms_per_year"].max() * 1.3, 5))
    ax.set_xticks([t for t in range(0, int(ax.get_xlim()[1]) + 1, 2)])
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("false alarms (alarm onsets) per year of normal operation", color=INK2, fontsize=9)
    ax.set_ylabel(f"process upsets: alarm within {ev.WINDOW} days", color=INK2, fontsize=9)
    ax.set_title("each method at its own threshold (open circle: chance)", fontsize=9.5, color=INK)
    ax = axes[1]
    _style(ax)
    c = common.set_index("method").loc[METHODS].reset_index().sort_values("excess_process")
    y = np.arange(len(c))
    cols = [COLORS[FAMILY[m]] for m in c["method"]]
    ax.barh(y, c["detected_process"], color=cols, alpha=0.85, height=0.62)
    ax.barh(y, c["chance"], color="#d6d5cc", height=0.62)
    for i, (_, r) in enumerate(c.iterrows()):
        ax.text(r["detected_process"] + 0.015, i, f"{r['excess_process']:+.0%}", va="center", fontsize=8.5,
                color=INK2)
    ax.set_yticks(y, c["method"], fontsize=8.5)
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(f"alarm within {ev.WINDOW} days (grey: chance; label: excess over chance)", color=INK2,
                  fontsize=9)
    ax.set_title(f"all methods at {ev.COMMON_RATE:.0f} false alarms per year", fontsize=9.5, color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[k], label=v) for k, v in FAMILY_LABEL.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(IMG / "tradeoff.png", dpi=150)
    plt.close(fig)


def fig_lead(det: pd.DataFrame, summ: pd.DataFrame) -> None:
    d = det[(det["missing"] == MISSING[0]) & det["kind"].isin(ev.LEAD_KINDS) & det["material_loss"]
            & det["yield_loss_day"].notna()]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    _style(ax)
    rng = np.random.default_rng(0)
    top = max(float(np.nanmax(d["lead"].to_numpy(dtype=float))), 10.0) + 2
    for i, m in enumerate(METHODS):
        g = d[d["method"] == m]
        lead = g["lead"].to_numpy(dtype=float)
        hit = int(g["before_loss"].astype(bool).sum())
        y = i + rng.uniform(-0.25, 0.25, len(lead))
        ax.scatter(lead, y, s=14, color=COLORS[FAMILY[m]], alpha=0.75, lw=0)
        if np.isfinite(lead).any():
            med = np.nanmedian(lead)
            ax.plot([med, med], [i - 0.35, i + 0.35], color=INK, lw=2)
        ch = summ.set_index("method").loc[m, "chance_before_loss"]
        ax.text(top + 1, i, f"{hit} of {len(g)} before the loss (chance {ch:.0%})", va="center", fontsize=8,
                color=MUTED)
    ax.set_yticks(range(len(METHODS)), METHODS)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, top + 12)
    ax.set_xlabel("ammonia inhibition: days from the first alarm to the start of yield loss", color=INK2,
                  fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "lead_time.png", dpi=150)
    plt.close(fig)


def fig_confusion(conf: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    labels = CLASSES[1:]
    short = ["overload", "acidif.", "ammonia", "heating", "substrate", "CH4 drift", "flow stuck"]
    for ax, m in zip(axes, CLASSIFIERS):
        c = conf[m]
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
        ax.set_title(f"{m}: {acc:.0%} of {int(a.sum())} detections named right", fontsize=9.5, color=INK)
        ax.set_xlabel("named at the first alarm", fontsize=8.5, color=INK2)
    axes[0].set_ylabel("true upset", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(IMG / "confusion.png", dpi=150)
    plt.close(fig)


def fig_robustness(rob: pd.DataFrame) -> None:
    ms = ["FOS/TAC +30 %", "Expert rules", "Gradient boosting", "EKF twin"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, lab in [(axes[0], "excess_process", f"process upsets: alarm within {ev.WINDOW} d, minus chance"),
                         (axes[1], "false_alarms_per_year", "false alarms per year")]:
        _style(ax)
        for i, m in enumerate(ms):
            g = rob[rob["method"] == m].sort_values("lab_missing")
            ax.plot(100 * g["lab_missing"], g[col], marker="o", lw=2, ms=7, label=m,
                    color=["#eb6834", "#2a78d6", "#1baf7a", "#6250d6"][i])
        ax.set_xlabel("lab samples missing [%]", fontsize=9, color=INK2)
        ax.set_ylabel(lab, fontsize=9, color=INK2)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
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
def _records(df: pd.DataFrame, digits: int = 4) -> list[dict]:
    return df.round(digits).replace({np.nan: None}).to_dict(orient="records")


def dashboard(st: dict, t: dict) -> dict:
    real = json.loads((RESULTS / "real_data.json").read_text(encoding="utf-8")) \
        if (RESULTS / "real_data.json").exists() else None
    order = {k: i for i, k in enumerate(CLASSES)}
    replays = sorted(st["replays"], key=lambda r: order[r["kind"]])
    return {
        "methods": METHODS, "variants": VARIANTS, "family": FAMILY, "family_label": FAMILY_LABEL,
        "classifiers": CLASSIFIERS, "classes": CLASSES, "labels": LABELS, "process": PROCESS,
        "sensor": SENSOR, "window": ev.WINDOW, "common_rate": ev.COMMON_RATE, "lead_kinds": list(ev.LEAD_KINDS),
        "summary": _records(t["summary"]), "variant_rows": _records(t["variants"]),
        "robustness": _records(t["robustness"]), "common": _records(t["common"]),
        "by_class": _records(t["by_class"], 3),
        "confusion": {m: c.to_numpy().tolist() for m, c in t["confusion"].items()},
        "unseen": _records(t["unseen"]) if t["unseen"] is not None else [],
        "truth": _records(t["truth"], 3),
        "replays": replays,
        "real": real,
    }


def write_all() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    st = load_study()
    t = tables(st)
    write_tables(t)
    fig_tradeoff(t["summary"], t["common"])
    fig_lead(t["det"], t["summary"])
    fig_confusion(t["confusion"])
    fig_robustness(t["robustness"])
    for rep in st["replays"]:
        if rep["kind"] in ("ammonia", "ch4_drift"):
            fig_replay(rep)
    data = dashboard(st, t)
    js = "window.DATA = " + json.dumps(data, separators=(",", ":"), allow_nan=False) + ";\n"
    (DOCS / "data.js").write_text(js, encoding="utf-8")
