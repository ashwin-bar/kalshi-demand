"""Daily demand per series: exact volume from candles vs a cheap shortcut built from lifetime volume."""
import json
import time
import pandas as pd
from kalshi_api import paginate, get, to_float, DATA_ROOT

ET = "America/New_York"
SETTLED = {"finalized", "settled", "determined"}


def _to_et(s):
    return pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce").dt.tz_convert(ET)


def _et_day(ts_et):
    """New York timestamps -> midnight of their New York calendar day (time zone dropped)."""
    return ts_et.dt.tz_localize(None).dt.normalize()


def load_markets(series, refresh=False):
    """All markets for a series from both tiers. Pulled once, then read from D:."""
    frames = []
    for tier, path in [("historical", "/historical/markets"), ("live", "/markets")]:
        f = DATA_ROOT / "raw" / "markets" / f"{series}__{tier}.json"
        if refresh or not f.exists():
            f.write_text(json.dumps(paginate(path, "markets", {"series_ticker": series})))
        frames.append(pd.DataFrame(json.loads(f.read_text())).assign(tier=tier))
    m = pd.concat(frames, ignore_index=True)
    if m.empty:
        return m
    m = m.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    m["open_et"] = _to_et(m["open_time"])
    m["close_et"] = _to_et(m["close_time"])
    m["close_day"] = _et_day(m["close_et"])
    m["listed_vol"] = m["volume_fp"].map(to_float)
    return m


def fetch_hist_candles(m, series, pause=0.2):
    """Daily candles for archived markets, one file per market. Files already on disk are skipped,
    so an interrupted run resumes where it stopped."""
    out = DATA_ROOT / "raw" / "candles" / series
    out.mkdir(parents=True, exist_ok=True)
    hist = m[m["tier"] == "historical"]
    todo = [r for r in hist.itertuples() if not (out / f"{r.ticker}.json").exists()]
    print(f"  candles: {len(hist) - len(todo)} already saved, {len(todo)} to fetch")
    for i, r in enumerate(todo, 1):
        params = {"start_ts": int(r.open_et.timestamp()),
                  "end_ts": int(r.close_et.timestamp()) + 86400,
                  "period_interval": 1440}
        resp = get(f"/historical/markets/{r.ticker}/candlesticks", params)
        (out / f"{r.ticker}.json").write_text(json.dumps(resp))
        time.sleep(pause)
        if i % 100 == 0:
            print(f"    {i}/{len(todo)}")


def load_candles(m, series):
    """Volume per market per New York day, from saved historical candle files."""
    rows = []
    d = DATA_ROOT / "raw" / "candles" / series
    for t in m["ticker"]:
        f = d / f"{t}.json"
        if not f.exists():
            continue
        for cd in json.loads(f.read_text()).get("candlesticks", []):
            rows.append((t, cd["end_period_ts"], to_float(cd["volume"])))   # historical-tier field name
    c = pd.DataFrame(rows, columns=["ticker", "end_period_ts", "volume"])
    end_et = pd.to_datetime(c["end_period_ts"], unit="s", utc=True).dt.tz_convert(ET)
    c["day"] = _et_day(end_et - pd.Timedelta(seconds=1))   # a candle ending at midnight covers the day before
    return c


def offset_profile(c):
    """Share of volume by whole-day offset from each market's close day (needs an 'offset' column)."""
    p = c.groupby("offset")["volume"].sum()
    return p / p.sum()


def shortcut_daily(m, profile):
    """Spread each market's lifetime volume over days relative to its close day, using a profile."""
    parts = [pd.Series(m["listed_vol"].to_numpy() * w, index=m["close_day"] + pd.Timedelta(days=int(off)))
             for off, w in profile.items()]
    return pd.concat(parts).groupby(level=0).sum()


def compare(exact, approx):
    df = pd.concat({"exact": exact, "approx": approx}, axis=1).fillna(0)
    err = df["approx"] - df["exact"]
    nz = df["exact"] > 0
    ape = err[nz].abs() / df.loc[nz, "exact"]
    return {"days": len(df), "corr": df["exact"].corr(df["approx"]),
            "wape": err.abs().sum() / df["exact"].sum(),
            "median_ape": ape.median(), "worst_ape": ape.max()}


def pick_windows(m, n_windows, days):
    """Start dates of n windows of consecutive days, spread evenly over the archived history."""
    h = m[(m["tier"] == "historical") & m["status"].isin(SETTLED)]
    pool = pd.Series(h["close_day"].unique()).sort_values().reset_index(drop=True)
    ok = pool[pool <= pool.max() - pd.Timedelta(days=days)].reset_index(drop=True)
    idx = pd.Series(range(n_windows)) * (len(ok) - 1) // max(n_windows - 1, 1)
    return [ok.iloc[i] for i in idx]


def validate_series(series, n_windows=3, days=14, fetch=True):
    """Compare exact daily volume with two shortcuts on sampled windows. Each window's profile
    comes from the OTHER windows, so the shortcut is never tested on the data it was fitted to."""
    m = load_markets(series)
    starts = pick_windows(m, n_windows, days)
    base = m[(m["tier"] == "historical") & m["status"].isin(SETTLED)]
    sm = pd.concat([base[(base["close_day"] >= s) & (base["close_day"] < s + pd.Timedelta(days=days))].assign(window=i)
                    for i, s in enumerate(starts)], ignore_index=True)
    print(f"{series}: {len(m):,} markets in total | validating on {len(sm):,} markets, "
          f"{n_windows} windows of {days} days")
    if fetch:
        fetch_hist_candles(sm, series)

    c = load_candles(sm, series).merge(sm[["ticker", "window", "close_day"]], on="ticker")
    c["offset"] = (c["day"] - c["close_day"]).dt.days
    n_back = int(-c["offset"].min())   # days before close that volume can land on

    results, profiles = [], {}
    for i, s in enumerate(starts):
        cw, mw = c[c["window"] == i], sm[sm["window"] == i]
        others = c[c["window"] != i]
        prof = offset_profile(others if len(others) else cw)
        profiles[f"{s:%Y-%m-%d}"] = offset_profile(cw)
        # Only score days whose volume comes entirely from markets inside the window
        core = pd.date_range(s, s + pd.Timedelta(days=days - 1 - n_back))
        exact = cw.groupby("day")["volume"].sum().reindex(core, fill_value=0)
        for name, p in [("single_day", pd.Series({prof.idxmax(): 1.0})), ("profile", prof)]:
            approx = shortcut_daily(mw, p).reindex(core, fill_value=0)
            results.append({"window": f"{s:%Y-%m-%d}", "method": name, **compare(exact, approx)})

    return pd.DataFrame(results), pd.DataFrame(profiles).fillna(0).sort_index(), sm, c