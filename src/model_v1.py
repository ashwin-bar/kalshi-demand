"""LightGBM v1 (rows 38, 44, 62, 63, 67, 89, 90, 96, 99). Weekly retraining; every forecast uses the latest model
trained only on units that closed before that forecast's issue time."""
import subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import G1, G2, G3
from pathlib import Path
ET = "America/New_York"
TRAIN_START = pd.Timestamp("2025-09-01")                                   # row 44
PARAMS = dict(learning_rate=0.08, n_estimators=400, num_leaves=63, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1)
FEATURE_SETS = {"all": G1 + G2 + G3,
                "no_G1_event": G2 + G3,
                "no_G2_history": G1 + G3,
                "no_G3_context": G1 + G2}
MIN_UNITS, MIN_SERIES = 100, 5                                             # row 90: smaller rows are reported, not interpreted


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=str(Path(__file__).resolve().parents[1]), text=True).strip()
    except Exception:
        return "unknown"


def week_cutoffs(start, end, lead_weeks=2, every=1):
    """Training cutoffs: Mondays 00:00 New York time, every `every` weeks, from lead_weeks before start (UTC)."""
    days = pd.date_range(pd.Timestamp(start) - pd.Timedelta(weeks=lead_weeks), end, freq=f"{7 * every}D")
    return days.tz_localize(ET).tz_convert("UTC").as_unit("ns")


def _to_ns(ts):
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ns")).to_numpy()


def assign_cutoff(issue_ts, cutoffs):
    """Latest cutoff at or before each issue time (earliest cutoff if none)."""
    idx = np.searchsorted(cutoffs.asi8, _to_ns(issue_ts), side="right") - 1
    return pd.Series(cutoffs[np.clip(idx, 0, len(cutoffs) - 1)], index=issue_ts.index)


def fit_predict(df, test_mask, feats, objective, cutoffs, params=None, weight_col=None, tweedie_power=1.5):
    """One model per cutoff, trained on units closed before it; predicts the rows assigned to it.
    l1: predicts the log-ratio target (column yhat). tweedie: predicts mean volume (column mu), with
    log1p(baseline) as an offset entering through init_score (row 89)."""
    params = {**PARAMS, **(params or {})}
    test = df[test_mask].copy()
    test["cutoff"] = assign_cutoff(test["issue_ts"], cutoffs)
    out = []
    for c, part in test.groupby("cutoff"):
        tr = df[(df["event_date"] >= TRAIN_START) & (df["close_ts"] < c) & df["y"].notna()]
        w = tr[weight_col].to_numpy() if weight_col else None
        p = part[["unit_id", "horizon"]].copy()
        if objective == "tweedie":
            m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=tweedie_power, **params)
            m.fit(tr[feats], tr["vol"], sample_weight=w, init_score=np.log1p(tr["denom"].to_numpy()))
            p["mu"] = np.exp(m.predict(part[feats], raw_score=True) + np.log1p(part["denom"].to_numpy()))
        else:
            m = lgb.LGBMRegressor(objective=objective, **params).fit(tr[feats], tr["y"], sample_weight=w)
            p["yhat"] = m.predict(part[feats])
        p["cutoff"], p["n_train"] = c, len(tr)
        out.append(p)
    return pd.concat(out, ignore_index=True)


def score_cols(d, cols, by=()):
    """Per week (and groups): volume-weighted WAPE, bias, median per-series WAPE, for each forecast column."""
    d = d.copy()
    d["week"] = d["event_date"] - pd.to_timedelta(d["event_date"].dt.weekday, unit="D")
    rows = []
    for key, g in d.groupby(["horizon", "week", *by], observed=True):
        y = g["vol"]
        if y.sum() <= 0:
            continue
        for c in cols:
            err = (g[c] - y).abs()
            ser = (err.groupby(g["series"]).sum() / y.groupby(g["series"]).sum()).replace(np.inf, np.nan)
            rows.append(dict(zip(["horizon", "week", *by], key), col=c, wape=err.sum() / y.sum(),
                             bias=g[c].sum() / y.sum() - 1, series_wape_med=ser.median()))
    return pd.DataFrame(rows)


def rows_table(d, cols, metric):
    """Row-73 results table: head by category, head total, tail, pooled; with unit and series counts and
    whether the row is large enough to interpret (row 90)."""
    a = score_cols(d, cols, by=["segment", "category"])
    a = a[a["segment"].astype(str) == "head"].assign(row=lambda x: "head: " + x["category"].astype(str))
    b = score_cols(d, cols, by=["segment"]).assign(row=lambda x: x["segment"].astype(str).map({"head": "HEAD TOTAL", "tail": "TAIL"}))
    c = score_cols(d, cols).assign(row="pooled (not headline)")
    t = pd.concat([a, b, c]).groupby(["row", "col"])[metric].mean().unstack()[cols]
    seg, cat = d["segment"].astype(str), d["category"].astype(str)
    masks = {f"head: {k}": (seg == "head") & (cat == k) for k in cat[seg == "head"].unique()}
    masks.update({"HEAD TOTAL": seg == "head", "TAIL": seg == "tail", "pooled (not headline)": seg == seg})
    n = pd.DataFrame({r: {"units": d.loc[m, "unit_id"].nunique(), "series": d.loc[m, "series"].nunique()}
                      for r, m in masks.items()}).T
    t = n.join(t, how="inner")
    t["interpret"] = np.where((t["units"] >= MIN_UNITS) & (t["series"] >= MIN_SERIES), "yes", "no")
    order = [f"head: {k}" for k in d[seg == "head"].groupby(cat[seg == "head"])["vol"].sum().sort_values(ascending=False).index]
    order += ["HEAD TOTAL", "TAIL", "pooled (not headline)"]
    return t.loc[[r for r in order if r in t.index]]


def rank_metrics(d, cols, min_n=10):
    """Row 96: within each issue-day x category with >= min_n units, Spearman rank correlation of forecast vs
    actual, and top-decile capture (share of the actual top 10% that the forecast also puts in its top 10%)."""
    d = d.copy()
    d["issue_day"] = d["issue_ts"].dt.tz_convert(ET).dt.date
    res = {c: {"rho": [], "top10": []} for c in cols}
    for _, g in d.groupby(["issue_day", "category"], observed=True):
        if len(g) < min_n:
            continue
        k = max(1, int(round(len(g) * 0.1)))
        top_actual = set(g.nlargest(k, "vol").index)
        for c in cols:
            res[c]["rho"].append(g[c].corr(g["vol"], method="spearman"))
            res[c]["top10"].append(len(top_actual & set(g.nlargest(k, c).index)) / k)
    return pd.DataFrame({c: {"spearman": np.nanmean(v["rho"]), "top_decile_capture": np.mean(v["top10"]),
                             "groups": len(v["top10"])} for c, v in res.items()}).T