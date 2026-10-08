"""Rolling conformal intervals (rows 125, 131): empirical quantiles of log-ratio residuals by category x horizon,
from units that closed in the 8 weeks before the issue day; pooled across categories if a pool is too small."""
import numpy as np
import pandas as pd

ET = "America/New_York"
WINDOW_DAYS = 56
MIN_POOL = 50
QUANTILES = (0.10, 0.90)


def _day(ts):
    return ts.dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()


def residual_pool(oos):
    """Out-of-sample residuals r = y - yhat with the New York day each unit closed."""
    p = oos.dropna(subset=["y", "yhat"]).copy()
    p["r"] = p["y"] - p["yhat"]
    p["close_day"] = _day(p["close_ts"])
    p["category"] = p["category"].astype(str)
    p["horizon"] = p["horizon"].astype(str)
    return p[["category", "horizon", "close_day", "r"]]


def interval_table(pool, days, categories, horizons):
    """q10/q90 per (category, horizon, issue day) from residuals closing in [day - 56 days, day)."""
    rows = []
    for h in horizons:
        ph = pool[pool["horizon"] == h]
        for d in days:
            win = ph[(ph["close_day"] >= d - pd.Timedelta(days=WINDOW_DAYS)) & (ph["close_day"] < d)]
            q_all = np.quantile(win["r"], QUANTILES) if len(win) >= MIN_POOL else (np.nan, np.nan)
            for c in categories:
                wc = win.loc[win["category"] == c, "r"]
                own = len(wc) >= MIN_POOL
                q = np.quantile(wc, QUANTILES) if own else q_all
                rows.append((c, h, d, q[0], q[1], own))
    return pd.DataFrame(rows, columns=["category", "horizon", "issue_day", "q_lo", "q_hi", "own_pool"])


def add_intervals(rows, pool):
    """Attach 80% interval bounds (in contracts) to forecast rows that carry denom and yhat."""
    rows = rows.copy()
    rows["category"] = rows["category"].astype(str)
    rows["horizon"] = rows["horizon"].astype(str)
    rows["issue_day"] = _day(rows["issue_ts"])
    qt = interval_table(pool, sorted(rows["issue_day"].unique()), sorted(rows["category"].unique()),
                        sorted(rows["horizon"].unique()))
    rows = rows.merge(qt, on=["category", "horizon", "issue_day"], how="left")
    base = np.log1p(rows["denom"]) + rows["yhat"]
    rows["lo"] = np.expm1(base + rows["q_lo"]).clip(lower=0)
    rows["hi"] = np.expm1(base + rows["q_hi"]).clip(lower=0)
    return rows