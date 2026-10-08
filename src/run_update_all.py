"""Bring every active series up to date with the daily updater (rows 93, 110) and measure the run time.
Active = series with a unit closing in the last 45 days, plus series new to Kalshi's catalogue.
Run from the project folder:   .venv\\Scripts\\python src\\run_update_all.py"""
import json
import time
import pandas as pd
import duckdb
from kalshi_api import get, to_float, DATA_ROOT
from crawl import STATE_DIR
from updater import run_update

t0 = time.time()
now = pd.Timestamp.now(tz="UTC")
stamp = now.strftime("%Y%m%d%H%M")

# 1. Active series from the unit table, plus series new to the catalogue
u = pd.read_parquet(DATA_ROOT / "processed" / "units_v6.parquet", columns=["series", "close_ts"])
u["close_ts"] = pd.to_datetime(u["close_ts"], utc=True)
active = set(u.loc[u["close_ts"] >= now - pd.Timedelta(days=45), "series"])

resp = get("/series", {"include_volume": "true"})
(DATA_ROOT / "raw" / "series" / f"series_{now:%Y%m%d}.json").write_text(json.dumps(resp))
sz = pd.read_parquet(DATA_ROOT / "interim" / "crawl_sizing.parquet")
excluded_head = set(sz.loc[~sz["in_scope"], "series"])          # one-off / outright head series (row 17/21)
new = {s["ticker"] for s in resp["series"]
       if to_float(s.get("volume_fp")) > 0 and not s["ticker"].startswith("KXMVE")
       and s["ticker"] not in excluded_head and not (STATE_DIR / f"{s['ticker']}.json").exists()}
series_list = sorted(active) + sorted(new)
print(f"Active series: {len(active):,} | new to catalogue: {len(new):,} | total: {len(series_list):,}", flush=True)

# 2. Lifespan margin per series (row 46): 1.5 x longest market life + 1 day; 30 days for new series
con = duckdb.connect()
globs = [(DATA_ROOT / "interim" / "markets_clean.parquet").as_posix(),
         (DATA_ROOT / "interim" / "markets_clean_tail" / "*.parquet").as_posix()]
union = " UNION ALL ".join(f"SELECT series, created_ts, close_ts FROM '{g}'" for g in globs)
con.register("_act", pd.DataFrame({"series": sorted(active)}))
mg = con.execute(f"""SELECT series, ceil(1.5 * max(date_diff('second', created_ts, close_ts)) / 86400.0) + 1 AS margin
                     FROM ({union}) JOIN _act USING (series) GROUP BY series""").df()
margins = dict(zip(mg["series"], mg["margin"].astype(int)))

# 3. Update everything; the cutoff is recorded only if every series succeeds
res, info = run_update(series_list, margins, save_state=True, stamp=stamp,
                       log=lambda m: print(m, flush=True))
out = DATA_ROOT / "logs" / f"update_all_{stamp}.csv"
res.to_csv(out, index=False)
errs = int(res["error"].notna().sum()) if "error" in res.columns else 0
print(f"\nDone in {(time.time() - t0) / 60:.0f} min | cutoff {info['cutoff_prev']} -> {info['cutoff_now']} "
      f"(advanced: {info['advanced']}) | state saved: {info['state_saved']} | errors: {errs}", flush=True)
for col in ["archive_added", "events_pulled"]:
    if col in res.columns:
        print(f"   total {col}: {int(res[col].fillna(0).sum()):,}", flush=True)
if "new_series" in res.columns:
    print(f"   new series crawled: {int(res['new_series'].fillna(False).sum()):,}", flush=True)
print(f"   results saved to {out}", flush=True)