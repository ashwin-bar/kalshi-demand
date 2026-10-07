"""Daily data update for the prospective runner (row 93), reusing the backtest crawler (row 91).
- Archive: only when Kalshi's historical cutoff has advanced since the last run. Crawl newest-created pages
  back to (previous cutoff - series margin) and store markets not already held (rows 25, 46). Done BEFORE
  the live refresh, so markets moving from live to archive are never lost.
- Live tier: re-read for active series; old live pages kept as dated snapshots (row 28).
- Events: pull events closing after (now - 3 days), with milestones, and merge into the cached event file.
- Series new to the catalogue get a full crawl."""
import json
import time
import pandas as pd
from kalshi_api import get, get_cutoff, DATA_ROOT
from crawl import COLS, CRAWL_DIR, STATE_DIR, refresh_live, crawl_series, _ts
from events import EVENTS_DIR

RUNNER_STATE = DATA_ROOT / "logs" / "runner_state.json"
BASE_CUTOFF = "2026-08-04T00:00:00Z"            # cutoff at the base crawls (logs/crawl_runs.jsonl)


def _held_archive_tickers(series):
    files = sorted((CRAWL_DIR / series).glob("historical_*.parquet"))
    if not files:
        return set()
    return set(pd.concat([pd.read_parquet(f, columns=["ticker"]) for f in files])["ticker"])


def archive_catch_up(series, prev_cutoff, margin_days, stamp, pause=0.1):
    """Store archived markets created after (prev_cutoff - margin) that are not held yet."""
    stop_before = pd.Timestamp(prev_cutoff) - pd.Timedelta(days=margin_days)
    (CRAWL_DIR / series).mkdir(parents=True, exist_ok=True)
    held = _held_archive_tickers(series)
    params, page, added = {"series_ticker": series, "limit": 1000}, 0, 0
    while True:
        r = get("/historical/markets", params)
        mk = r.get("markets", [])
        if not mk:
            break
        df = pd.DataFrame(mk).reindex(columns=COLS)
        new = df[~df["ticker"].isin(held)]
        if len(new):
            new.to_parquet(CRAWL_DIR / series / f"historical_{stamp}{page:04d}.parquet", index=False)
            added += len(new)
        page += 1
        if _ts(df["created_time"]).max() < stop_before or not r.get("cursor"):
            break
        params["cursor"] = r["cursor"]
        time.sleep(pause)
    return page, added


def events_update(series, since_ts, pause=0.1):
    """Pull events with a market closing after since_ts (with milestones) and merge into the cached file."""
    f = EVENTS_DIR / f"{series}__events.json"
    old = json.loads(f.read_text()) if f.exists() else {"events": [], "milestones": []}
    params = {"series_ticker": series, "limit": 200, "with_milestones": "true", "min_close_ts": int(since_ts.timestamp())}
    ev, ms = [], []
    while True:
        r = get("/events", params)
        ev += r.get("events", [])
        ms += r.get("milestones") or []
        if not r.get("cursor"):
            break
        params["cursor"] = r["cursor"]
        time.sleep(pause)
    evd = {e["event_ticker"]: e for e in old["events"]}
    evd.update({e["event_ticker"]: e for e in ev})
    msd = {m["id"]: m for m in old["milestones"]}
    msd.update({m["id"]: m for m in ms})
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"events": list(evd.values()), "milestones": list(msd.values())}))
    return len(ev)


def run_update(series_list, margins, save_state=True, stamp=None, log=print):
    """Update a list of series. The new cutoff is recorded only if save_state and every series succeeded."""
    stamp = stamp or pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M")
    st = json.loads(RUNNER_STATE.read_text()) if RUNNER_STATE.exists() else {}
    cut_prev = st.get("market_settled_ts", BASE_CUTOFF)
    cut_now = get_cutoff()["market_settled_ts"]
    advanced = pd.Timestamp(cut_now) > pd.Timestamp(cut_prev)
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
    rows = []
    for i, s in enumerate(series_list, 1):
        row = {"series": s}
        try:
            if not (STATE_DIR / f"{s}.json").exists():                    # new series: full crawl (row 46 margin)
                crawl_series(s, margin_days=400)
                row["new_series"] = True
            else:
                if advanced:
                    row["archive_pages"], row["archive_added"] = archive_catch_up(s, cut_prev, margins.get(s, 30), stamp)
                refresh_live(s)
            row["events_pulled"] = events_update(s, since)
        except Exception as e:
            row["error"] = repr(e)
        rows.append(row)
        if i % 100 == 0:
            log(f"  {i:,}/{len(series_list):,} series updated")
    out = pd.DataFrame(rows)
    ok = "error" not in out.columns or out["error"].isna().all()
    if save_state and ok:
        st.update({"market_settled_ts": cut_now, "last_update": stamp})
        RUNNER_STATE.parent.mkdir(parents=True, exist_ok=True)
        RUNNER_STATE.write_text(json.dumps(st, indent=2))
    return out, {"cutoff_prev": cut_prev, "cutoff_now": cut_now, "advanced": advanced, "state_saved": save_state and ok}