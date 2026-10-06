"""Baseline ladder for unit-level demand. A unit's volume counts as known only after its close, so every
forecast uses units that closed strictly before its issue time."""
import numpy as np
import pandas as pd

HORIZONS_H = {"listing": 168, "T-1d": 24, "T-2h": 2}     # "listing" = T-7d, which precedes listing for ~97% of units (row 65)
BASELINES = ["last", "med10", "med28d", "weekly", "mean10", "mean28d", "mean28d_g"]


def issue_times(units):
    """One row per unit and horizon. Issue time = max(anchor_eff - H, listing time) (row 19),
    capped 1s before the unit's close so its own volume is never known (row 50)."""
    cap = units["close_ts"] - pd.Timedelta(seconds=1)
    out = []
    for name, H in HORIZONS_H.items():
        it = units["anchor_eff"] - pd.Timedelta(hours=H)
        listed_later = it < units["open_ts"]
        it = it.where(~listed_later, units["open_ts"])
        it = it.where(it <= cap, cap)
        out.append(units[["unit_id", "series", "category", "event_date", "vol", "hf", "multiday"]]
                   .assign(horizon=name, issue_ts=it, at_listing=listed_later))
    return pd.concat(out, ignore_index=True)


def history_stats(units):
    """Per series, in close order, evaluated at each unit's close: its volume, count of completed units,
    median/mean of the last 10, trailing 28-day median/mean, and a 14-day-over-14-day growth ratio."""
    h = units[["series", "close_ts", "vol"]].sort_values(["series", "close_ts"]).reset_index(drop=True)
    g = h.groupby("series")["vol"]
    h["n_prior"] = h.groupby("series").cumcount() + 1
    h["med10"] = g.transform(lambda s: s.rolling(10, min_periods=1).median())
    h["mean10"] = g.transform(lambda s: s.rolling(10, min_periods=1).mean())
    t = h.set_index("close_ts").groupby("series")["vol"]
    h["med28d"] = t.transform(lambda s: s.rolling("28D").median()).to_numpy()
    h["mean28d"] = t.transform(lambda s: s.rolling("28D").mean()).to_numpy()
    sum14 = t.transform(lambda s: s.rolling("14D").sum()).to_numpy()
    sum28 = t.transform(lambda s: s.rolling("28D").sum()).to_numpy()
    prev = sum28 - sum14
    h["growth"] = pd.Series(np.where(prev > 0, sum14 / np.where(prev > 0, prev, 1), np.nan)).clip(0.25, 4).fillna(1.0).to_numpy()
    return h.rename(columns={"vol": "last", "close_ts": "hist_close_ts"})


def add_baselines(iss, units):
    """Attach each baseline and the history stratum as known at the issue time (strictly earlier closes only)."""
    h = history_stats(units).sort_values("hist_close_ts")
    out = pd.merge_asof(iss.sort_values("issue_ts"), h, by="series", left_on="issue_ts",
                        right_on="hist_close_ts", direction="backward", allow_exact_matches=False)
    w = (units.groupby(["series", "event_date"])
              .agg(wk_med=("vol", "median"), wk_close=("close_ts", "max")).reset_index())
    w["event_date"] = w["event_date"] + pd.Timedelta(days=7)
    out = out.merge(w, on=["series", "event_date"], how="left")
    out.loc[out["wk_close"] >= out["issue_ts"], "wk_med"] = np.nan
    out["weekly_pure"] = out["wk_med"].notna()
    out["weekly"] = out["wk_med"].fillna(out["med10"])
    out["mean28d_g"] = out["mean28d"] * out["growth"]
    out["stratum"] = np.select([out["n_prior"].isna(), out["n_prior"] < 20], ["cold", "lukewarm"], "warm")
    return out


def score(fc, by=()):
    """Per week (and optional groups): volume-weighted WAPE, bias, and median per-series WAPE.
    Only units with a forecast (non-cold) are scored."""
    fc = fc[fc["med10"].notna()].copy()
    fc["week"] = fc["event_date"] - pd.to_timedelta(fc["event_date"].dt.weekday, unit="D")
    rows = []
    for key, d in fc.groupby(["horizon", "week", *by]):
        y = d["vol"]
        if y.sum() <= 0:
            continue
        for b in BASELINES:
            err = (d[b] - y).abs()
            ser = (err.groupby(d["series"]).sum() / y.groupby(d["series"]).sum()).replace(np.inf, np.nan)
            rows.append(dict(zip(["horizon", "week", *by], key), baseline=b, wape=err.sum() / y.sum(),
                             bias=d[b].sum() / y.sum() - 1, series_wape_med=ser.median()))
    return pd.DataFrame(rows)