"""Feature table v1 (rows 38, 63, 67, 69): one row per unit x horizon. Every feature uses only information
available at the issue time: history via strictly earlier closes, counts via listings at or before issue."""
import numpy as np
import pandas as pd

ET = "America/New_York"
G1 = ["horizon", "dow", "hour", "dow_hour", "month", "lead_h", "to_anchor_h", "since_listing_h",
      "strikes_so_far", "fixture_units", "fixture_series", "slate_size", "multiday", "hf", "category", "segment"]
G2 = ["med28d", "mean28d", "med10", "last", "weekly", "n_prior", "growth", "skew28", "trend10", "since_last_h"]
G3 = ["exch_7d", "exch_growth", "cat_7d", "cat_med28d"]
CATEGORICAL = ["horizon", "category", "segment"]


def base_frame(fc, units):
    u = units[["unit_id", "anchor_eff", "anchor_capped", "open_ts", "close_ts", "fixture_id", "n_mkts"]]
    return fc.merge(u, on="unit_id", how="left")


def add_calendar(df):
    ok = ~df["anchor_capped"].astype(bool)                     # capped anchors use the close time: hindsight
    a = df["anchor_eff"].dt.tz_convert(ET)
    df["dow"] = a.dt.dayofweek.where(ok)
    df["hour"] = a.dt.hour.where(ok)
    df["dow_hour"] = df["dow"] * 24 + df["hour"]
    df["month"] = df["event_date"].dt.month
    df["lead_h"] = ((df["anchor_eff"] - df["open_ts"]).dt.total_seconds() / 3600).where(ok)
    df["to_anchor_h"] = ((df["anchor_eff"] - df["issue_ts"]).dt.total_seconds() / 3600).where(ok)
    df["since_listing_h"] = (df["issue_ts"] - df["open_ts"]).dt.total_seconds() / 3600
    return df


def add_history(df):
    df["skew28"] = (df["mean28d"] / df["med28d"]).replace([np.inf, -np.inf], np.nan)
    df["trend10"] = (df["med10"] / df["med28d"]).replace([np.inf, -np.inf], np.nan)
    df["since_last_h"] = (df["issue_ts"] - df["hist_close_ts"]).dt.total_seconds() / 3600
    return df


def add_context(df, units):
    """Exchange-wide and category-wide recent volume, from units closed strictly before the issue time."""
    h = units[["category", "close_ts", "vol"]].sort_values("close_ts").reset_index(drop=True)
    ex = h.set_index("close_ts")["vol"]
    s7, s14 = ex.rolling("7D").sum().to_numpy(), ex.rolling("14D").sum().to_numpy()
    exd = pd.DataFrame({"ctx_ts": h["close_ts"], "exch_7d": s7, "exch_prev7d": s14 - s7})
    t = h.set_index("close_ts").groupby("category")["vol"]
    catd = pd.DataFrame({"category": h["category"], "cat_ts": h["close_ts"],
                         "cat_7d": t.transform(lambda s: s.rolling("7D").sum()).to_numpy(),
                         "cat_med28d": t.transform(lambda s: s.rolling("28D").median()).to_numpy()})
    df = pd.merge_asof(df.sort_values("issue_ts"), exd, left_on="issue_ts", right_on="ctx_ts",
                       direction="backward", allow_exact_matches=False)
    df = pd.merge_asof(df.sort_values("issue_ts"), catd.sort_values("cat_ts"), by="category",
                       left_on="issue_ts", right_on="cat_ts", direction="backward", allow_exact_matches=False)
    df["exch_growth"] = (df["exch_7d"] / df["exch_prev7d"]).replace([np.inf, -np.inf], np.nan)
    return df.drop(columns=["ctx_ts", "cat_ts", "exch_prev7d"])


def add_counts(df, units, con, clean_globs):
    """Strikes listed so far (event units), slate size and fixture size, all counted at the issue time."""
    iss = df[["unit_id", "horizon", "series", "event_date", "issue_ts", "fixture_id", "hf"]].copy()
    con.register("_iss", iss)
    con.register("_u", units[["unit_id", "series", "event_date", "fixture_id", "open_ts"]])
    union = " UNION ALL ".join(f"SELECT event_ticker, open_ts FROM '{g}'" for g in clean_globs)
    strikes = con.execute(f"""
      SELECT i.unit_id, i.horizon, count(*) AS strikes_so_far
      FROM _iss i JOIN ({union}) m ON m.event_ticker = i.unit_id AND m.open_ts <= i.issue_ts
      WHERE NOT i.hf GROUP BY ALL""").df()
    slate = con.execute("""
      SELECT i.unit_id, i.horizon, count(*) AS slate_size
      FROM _iss i JOIN _u u ON u.series = i.series AND u.event_date = i.event_date AND u.open_ts <= i.issue_ts
      GROUP BY ALL""").df()
    fixture = con.execute("""
      SELECT i.unit_id, i.horizon, count(*) AS fixture_units, count(DISTINCT u.series) AS fixture_series
      FROM _iss i JOIN _u u ON u.fixture_id = i.fixture_id AND u.open_ts <= i.issue_ts
      WHERE i.fixture_id IS NOT NULL GROUP BY ALL""").df()
    for t in [strikes, slate, fixture]:
        df = df.merge(t, on=["unit_id", "horizon"], how="left")
    return df


def add_target(df):
    """Row 38: log ratio of volume to a baseline known at issue time (series med28d, else category median)."""
    df["denom"] = df["med28d"].fillna(df["cat_med28d"])
    df["y"] = np.log1p(df["vol"]) - np.log1p(df["denom"])
    return df


def build_features(fc, units, con, clean_globs):
    df = base_frame(fc, units)
    df = add_calendar(df)
    df = add_history(df)
    df = add_context(df, units)
    df = add_counts(df, units, con, clean_globs)
    df = add_target(df)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df.sort_values(["unit_id", "horizon"]).reset_index(drop=True)