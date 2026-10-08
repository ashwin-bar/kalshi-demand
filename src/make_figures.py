"""Figures for the README and technical post (row 148). Six are computed from the scored test predictions of
release v1.0.0 (read only); the ablation figure uses the validation results of Block 60 (row 129).
Run from the project folder:   .venv\\Scripts\\python src\\make_figures.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from kalshi_api import DATA_ROOT
from model_v1 import rows_table, rank_metrics

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, BENCH, V1, V1B, GRID = "#1f2937", "#9ca3af", "#2563eb", "#93c5fd", "#e5e7eb"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})
PCT = PercentFormatter(1.0, decimals=0)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", name)


def weekly_wape(d, col, keys):
    """WAPE of forecasts summed to `keys`, computed per week and averaged equally (as in the holdout plan)."""
    a = d.groupby(keys)[["vol", col]].sum().reset_index()
    a["week"] = a["event_date"] - pd.to_timedelta(a["event_date"].dt.weekday, unit="D")
    a["err"] = (a[col] - a["vol"]).abs()
    w = a.groupby("week")[["err", "vol"]].sum()
    return (w["err"] / w["vol"]).mean()


def pooled_wape(d, col):
    return (d[col] - d["vol"]).abs().sum() / d["vol"].sum()


t = pd.read_parquet(DATA_ROOT / "processed" / "preds_test_frozen_v1.0.0.parquet")
t["horizon"] = t["horizon"].astype(str)
t["category"] = t["category"].astype(str)
t.loc[t["series"].str.startswith("KXWC"), "category"] = "Sports (World Cup)"
warm = t[t["med10"].notna()]
t1 = warm[warm["horizon"] == "T-1d"].copy()

# 1. Benchmark vs v1 by row
wt = rows_table(t1, ["med28d", "med10", "m"], "wape")
rows = [("HEAD TOTAL", "High-volume series"), ("head: Sports", "Sports"), ("head: Crypto", "Crypto"),
        ("head: Sports (World Cup)", "World Cup 2026"), ("TAIL", "Long tail")]
totals = {"HEAD TOTAL", "TAIL"}
bench = [wt.loc[r, "med10"] if r == "TAIL" else wt.loc[r, "med28d"] for r, _ in rows]
v1 = [wt.loc[r, "m"] for r, _ in rows]
labels = [f"{name}\n{int(wt.loc[r, 'series']):,} series" for r, name in rows]
fig, ax = plt.subplots(figsize=(9, 5.2))
y = np.array([5.4, 4.2, 3.2, 2.2, 0.8])                       # gap separates the head breakdown from the tail
ax.barh(y + 0.19, bench, 0.36, color=BENCH, label="Benchmark: median of the series' recent events")
ax.barh(y - 0.19, v1, 0.36, color=V1, label="v1 model (change vs benchmark in brackets)")
for yi, b, v in zip(y, bench, v1):
    ax.text(b + 0.006, yi + 0.19, f"{b:.1%}", va="center", fontsize=9, color=INK)
    ax.text(v + 0.006, yi - 0.19, f"{v:.1%}  ({v / b - 1:+.0%})", va="center", fontsize=9, color=INK,
            fontweight="bold")
ax.set_yticks(y, labels)
for tick, (r, _) in zip(ax.get_yticklabels(), rows):
    tick.set_fontweight("bold" if r in totals else "normal")
ax.axhline(1.5, color=GRID, linewidth=1)
ax.text(0.003, 4.75, "of which:", fontsize=8, color="#6b7280", style="italic")
ax.xaxis.set_major_formatter(PCT)
ax.set_xlim(0, max(bench + v1) * 1.32)
ax.set_xlabel("Weighted absolute percentage error (WAPE), lower is better")
ax.set_title("Forecast error on 30 held-out weeks\nForecasts made 1 day before each event; weeks never used to "
             "train or tune the model", loc="left", fontsize=11)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2, frameon=False, fontsize=9)
ax.grid(axis="y", visible=False)
save(fig, "01_error_by_row.png")

# 2. Error by hierarchy level
levels = [("Event", ["event_date", "unit_id"]), ("Series-day", ["event_date", "series"]),
          ("Category-day", ["event_date", "category"]), ("Total-day", ["event_date"])]
lines = {"v1 median": ("m", V1, "-"), "v1 mean (Tweedie)": ("mu", V1, "--"),
         "Trailing median (med28d)": ("med28d", BENCH, "-"), "Trailing mean (mean28d)": ("mean28d", BENCH, "--")}
fig, ax = plt.subplots(figsize=(8, 4.4))
x = np.arange(len(levels))
for label, (col, colour, ls) in lines.items():
    vals = [weekly_wape(t1, col, keys) for _, keys in levels]
    ax.plot(x, vals, ls, color=colour, marker="o", label=label, linewidth=2)
    if col in ("mu", "mean28d"):
        ax.text(x[-1] + 0.08, vals[-1], f"{vals[-1]:.1%}", va="center", fontsize=9, color=INK)
ax.set_xticks(x, [name for name, _ in levels])
ax.yaxis.set_major_formatter(PCT)
ax.set_ylim(0, None)
ax.set_ylabel("WAPE, lower is better")
ax.set_title("Error shrinks as forecasts are summed up the hierarchy (held-out weeks, T−1d)", loc="left", fontsize=12)
ax.legend(frameon=False)
save(fig, "02_error_by_level.png")

# 3. Ranking: top-decile capture
groups = [("All categories", t1), ("Sports", t1[t1["category"].str.startswith("Sports")])]
fig, ax = plt.subplots(figsize=(6.5, 4))
x = np.arange(len(groups))
for i, (col, colour, label) in enumerate([("med28d", BENCH, "Trailing-median benchmark"), ("m", V1, "v1")]):
    vals = [rank_metrics(sub, ["med28d", "m"]).loc[col, "top_decile_capture"] for _, sub in groups]
    bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=colour, label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=10, color=INK)
ax.set_xticks(x, [g for g, _ in groups])
ax.yaxis.set_major_formatter(PCT)
ax.set_ylim(0, 1)
ax.set_ylabel("Share of the busiest 10% of events\ncorrectly placed in the top 10%")
ax.set_title("Finding the biggest events in advance", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="x", visible=False)
save(fig, "03_ranking.png")

# 4. One NBA week: forecast vs actual with 80% intervals (week chosen by rule: the most games in the test period)
nba = t1[t1["series"] == "KXNBAGAME"].copy()
nba["week"] = nba["event_date"] - pd.to_timedelta(nba["event_date"].dt.weekday, unit="D")
wk = nba.groupby("week")["unit_id"].nunique().idxmax()
g = nba[nba["week"] == wk].sort_values(["event_date", "vol"]).reset_index(drop=True)
g["label"] = g["event_date"].dt.strftime("%a %d %b") + "  " + g["unit_id"].str.split("-").str[1].str[7:]
inside = ((g["vol"] >= g["lo"]) & (g["vol"] <= g["hi"])).sum()
nba_cov = ((nba["vol"] >= nba["lo"]) & (nba["vol"] <= nba["hi"])).mean()
fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.32 * len(g) + 1.5)))
yy = np.arange(len(g))[::-1]
ax.hlines(yy, g["lo"], g["hi"], color=V1B, linewidth=5, label="80% interval")
ax.scatter(g["m"], yy, color=V1, zorder=3, s=30, label="Forecast (median, 1 day ahead)")
ax.scatter(g["vol"], yy, color=INK, marker="x", zorder=4, s=36, label="Actual contracts traded")
ax.set_xscale("log")
ax.set_yticks(yy, g["label"], fontsize=8)
ax.set_xlabel("Contracts traded per game (log scale)")
ax.set_title(f"NBA game-winner markets, week of {wk:%d %b %Y} (the week with the most games in the test period)\n"
             f"{inside} of {len(g)} games inside the 80% interval; over the whole test period: {nba_cov:.0%}",
             loc="left", fontsize=10)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.grid(axis="y", visible=False)
save(fig, "04_nba_week.png")

# 5. Interval coverage and width by category
seg = t1["segment"].astype(str)
t1["covered"] = (t1["vol"] >= t1["lo"]) & (t1["vol"] <= t1["hi"])
t1["width"] = np.log1p(t1["hi"]) - np.log1p(t1["lo"])
cats = [(c, (seg == "head") & (t1["category"] == c)) for c in t1.loc[seg == "head", "category"].unique()]
cats = [(f"Head: {c}", m) for c, m in cats if t1.loc[m, "unit_id"].nunique() >= 100 and t1.loc[m, "series"].nunique() >= 5]
cats += [("Head total", seg == "head"), ("Long tail", seg == "tail")]
cov = [t1.loc[m, "covered"].mean() for _, m in cats]
wid = [np.expm1(t1.loc[m, "width"].median()) + 1 for _, m in cats]
fig, ax = plt.subplots(figsize=(9, 4.6))
y = np.arange(len(cats))[::-1]
ax.barh(y, cov, 0.6, color=[V1 if "total" in n or "tail" in n else V1B for n, _ in cats])
ax.axvline(0.8, color=INK, linestyle="--", linewidth=1)
ax.text(0.81, -0.75, "target 80%", fontsize=9, color=INK)
for yi, c, w in zip(y, cov, wid):
    ax.text(c + 0.01, yi, f"{c:.0%} covered · width {w:.0f}×", va="center", fontsize=9, color=INK)
ax.set_yticks(y, [n for n, _ in cats])
ax.set_ylim(-1, len(cats) - 0.4)
ax.xaxis.set_major_formatter(PCT)
ax.set_xlim(0, 1.4)
ax.set_xlabel("Share of actual outcomes inside the 80% interval   (width = high end ÷ low end)")
ax.set_title("80% prediction intervals: calibrated overall, but wide (held-out weeks, T−1d)", loc="left", fontsize=12)
ax.grid(axis="y", visible=False)
save(fig, "05_intervals.png")

# 6. Ablations (validation weeks, Block 60 / row 129): increase in WAPE when each feature group is removed
abl = {"Event characteristics": (7.68, 7.94, 7.74), "Series history": (5.31, 5.49, 13.40),
       "Team / player popularity": (4.14, 4.59, 0.52), "Market context": (-0.15, -0.32, -0.19)}
fig, ax = plt.subplots(figsize=(8.5, 4.4))
y = np.arange(len(abl))[::-1]
for i, (label, colour) in enumerate([("Head total", V1), ("Head: sports", V1B), ("Long tail", BENCH)]):
    vals = [v[i] for v in abl.values()]
    ax.barh(y + (1 - i) * 0.26, vals, 0.26, color=colour, label=label)
ax.axvline(0, color=INK, linewidth=0.8)
ax.set_yticks(y, list(abl.keys()))
ax.set_xlabel("Increase in WAPE (percentage points) when the group is removed")
ax.set_title("What each feature group contributes (validation weeks)", loc="left", fontsize=12)
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", visible=False)
save(fig, "06_ablations.png")

# 7. World Cup by week: error and mean-model bias
wc = t1[t1["category"] == "Sports (World Cup)"].copy()
wc["week"] = wc["event_date"] - pd.to_timedelta(wc["event_date"].dt.weekday, unit="D")
wcw = wc.groupby("week").apply(lambda d: pd.Series({"v1": pooled_wape(d, "m"), "med10": pooled_wape(d, "med10"),
                                                    "med28d": pooled_wape(d, "med28d"),
                                                    "bias": d["mu"].sum() / d["vol"].sum() - 1}),
                               include_groups=False)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
x = np.arange(len(wcw))
a1.plot(x, wcw["v1"], "-o", color=V1, linewidth=2, label="v1 median")
a1.plot(x, wcw["med10"], "-o", color=BENCH, linewidth=2, label="Last 10 events (med10)")
a1.plot(x, wcw["med28d"], "--o", color=BENCH, linewidth=1.5, label="Trailing 28 days (med28d)")
a1.yaxis.set_major_formatter(PCT)
a1.set_ylabel("WAPE")
a1.legend(frameon=False, fontsize=9)
a1.set_title("World Cup 2026: the mean model overshoots the opening weeks;\n"
             "the most recent games (med10) are the better guide until the final week", loc="left", fontsize=11)
a2.bar(x, wcw["bias"], color=[V1 if b >= 0 else BENCH for b in wcw["bias"]])
a2.axhline(0, color=INK, linewidth=0.8)
a2.yaxis.set_major_formatter(PCT)
a2.set_ylabel("Mean-model bias")
a2.set_xticks(x, [f"w/c {d:%d %b}" for d in wcw.index])
for a in (a1, a2):
    a.axvspan(-0.5, 2.5, color=GRID, alpha=0.5, zorder=0)
a1.text(1, a1.get_ylim()[1] * 0.95, "first three weeks", ha="center", fontsize=9, color=INK)
save(fig, "07_world_cup.png")
print(f"\nAll figures written to {OUT}")