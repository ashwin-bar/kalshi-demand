"""Background pull for the pickup subset (row 42): hourly candles (daily for the Fed) for every market of the
major game-winner series + KXFEDDECISION closing since 2025-08-01. One JSON file per market; safe to stop and rerun.
Run from the project folder:   .venv\\Scripts\\python src\\pull_pickup_candles.py"""
import json
import time
import pandas as pd
import requests
from kalshi_api import get, DATA_ROOT

SERIES = ["KXNBAGAME", "KXMLBGAME", "KXATPMATCH", "KXNFLGAME", "KXNCAAMBGAME", "KXNCAAFGAME", "KXWTAMATCH",
          "KXWCGAME", "KXNHLGAME", "KXUFCFIGHT", "KXWNBAGAME", "KXIPLGAME", "KXEPLGAME", "KXMARMAD",
          "KXLALIGAGAME", "KXUCLGAME", "KXFEDDECISION"]
DAILY = {"KXFEDDECISION"}                       # some Fed markets live for hundreds of days (row 42)
SINCE = pd.Timestamp("2025-08-01", tz="UTC")
MAX_CANDLES = 4800                              # per call; the API caps a request at 5,000 candles
OUT = DATA_ROOT / "raw" / "candles_pickup"

m = pd.read_parquet(DATA_ROOT / "interim" / "markets_clean.parquet",
                    columns=["series", "ticker", "tier", "open_ts", "close_ts"],
                    filters=[("series", "in", SERIES)])
m["open_ts"] = pd.to_datetime(m["open_ts"], utc=True)
m["close_ts"] = pd.to_datetime(m["close_ts"], utc=True)
m = m[m["close_ts"] >= SINCE].sort_values(["series", "close_ts"])


def fetch(r, period):
    """All candles for one market, in chunks; tries the tier seen at crawl time, then the other."""
    step = MAX_CANDLES * period * 60
    start, end = int(r.open_ts.timestamp()), int(r.close_ts.timestamp()) + period * 60
    paths = {"historical": f"/historical/markets/{r.ticker}/candlesticks",
             "live": f"/series/{r.series}/markets/{r.ticker}/candlesticks"}
    for tier in [r.tier, "live" if r.tier == "historical" else "historical"]:
        candles, ok = [], True
        try:
            for a in range(start, end, step):
                resp = get(paths[tier], {"start_ts": a, "end_ts": min(a + step, end), "period_interval": period})
                candles += resp.get("candlesticks", [])
        except requests.HTTPError:
            ok = False
        if ok and candles:
            return tier, candles
    return None, []


todo = [r for r in m.itertuples() if not (OUT / r.series / f"{r.ticker}.json").exists()]
print(f"{len(m):,} markets in scope, {len(todo):,} still to pull", flush=True)
t0, errors = time.time(), 0
for i, r in enumerate(todo, 1):
    period = 1440 if r.series in DAILY else 60
    try:
        tier, candles = fetch(r, period)
    except Exception as e:
        errors += 1
        print(f"  {r.ticker}: ERROR {e!r}", flush=True)
        continue
    uniq = {c["end_period_ts"]: c for c in candles}            # chunk boundaries can repeat a candle
    d = OUT / r.series
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{r.ticker}.json").write_text(json.dumps({"ticker": r.ticker, "series": r.series, "tier": tier,
                                                    "period": period, "candlesticks": [uniq[k] for k in sorted(uniq)]}))
    time.sleep(0.05)
    if i % 500 == 0:
        print(f"  {i:,}/{len(todo):,} markets | {(time.time() - t0) / 60:.0f} min | errors {errors}", flush=True)
print(f"Done: {len(todo) - errors:,} pulled, {errors} errors, {(time.time() - t0) / 60:.0f} min", flush=True)