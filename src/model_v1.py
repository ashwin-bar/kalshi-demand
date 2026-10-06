"""LightGBM v1 (rows 38, 44, 62, 63, 67). Weekly retraining; every forecast uses the latest model trained
only on units that closed before that forecast's issue time."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import G1, G2, G3

ET = "America/New_York"
TRAIN_START = pd.Timestamp("2025-09-01")                                   # row 44
PARAMS = dict(learning_rate=0.08, n_estimators=400, num_leaves=63, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1)
FEATURE_SETS = {"all": G1 + G2 + G3,
                "no_G1_event": G2 + G3,
                "no_G2_history": G1 + G3,
                "no_G3_context": G1 + G2}


def week_cutoffs(start, end, lead_weeks=2):
    """Training cutoffs: Mondays 00:00 New York time, from lead_weeks before start to end (UTC)."""
    days = pd.date_range(pd.Timestamp(start) - pd.Timedelta(weeks=lead_weeks), end, freq="7D")
    return days.tz_localize(ET).tz_convert("UTC").as_unit("ns")


def _to_ns(ts):
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ns")).to_numpy()


def assign_cutoff(issue_ts, cutoffs):
    """Latest cutoff at or before each issue time (earliest cutoff if none)."""
    idx = np.searchsorted(cutoffs.asi8, _to_ns(issue_ts), side="right") - 1
    return pd.Series(cutoffs[np.clip(idx, 0, len(cutoffs) - 1)], index=issue_ts.index)


def fit_predict(df, test_mask, feats, objective, cutoffs):
    """Train one model per cutoff on units closed before it; predict the rows assigned to it."""
    test = df[test_mask].copy()
    test["cutoff"] = assign_cutoff(test["issue_ts"], cutoffs)
    out = []
    for c, part in test.groupby("cutoff"):
        tr = df[(df["event_date"] >= TRAIN_START) & (df["close_ts"] < c) & df["y"].notna()]
        model = lgb.LGBMRegressor(objective=objective, **PARAMS).fit(tr[feats], tr["y"])
        p = part[["unit_id", "horizon"]].copy()
        p["yhat"] = model.predict(part[feats])
        if objective == "l2":                                              # Duan smearing for the mean
            smp = tr.sample(min(len(tr), 200_000), random_state=0)
            p["smear"] = float(np.mean(np.exp(smp["y"] - model.predict(smp[feats]))))
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