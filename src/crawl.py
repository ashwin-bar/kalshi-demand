"""Resumable market-list crawler: one parquet file per page, progress saved after every page.
The historical tier is sorted newest-first by created_time (verified: 0 breaks across 130 series)."""
import json
import shutil
import time
import pandas as pd
from kalshi_api import get, DATA_ROOT

COLS = ["ticker", "event_ticker", "status", "created_time", "open_time", "close_time", "settlement_ts",
        "volume_fp", "result", "strike_type", "floor_strike", "cap_strike", "yes_sub_title", "can_close_early"]
TRAIN_START = pd.Timestamp("2025-01-01", tz="UTC")
# Live first: a market that settles mid-crawl moves live -> historical, so it's caught twice, never missed
TIERS = [("live", "/markets"), ("historical", "/historical/markets")]
CRAWL_DIR = DATA_ROOT / "raw" / "markets_crawl"
SNAP_DIR = DATA_ROOT / "raw" / "markets_crawl_snapshots"
STATE_DIR = DATA_ROOT / "logs" / "crawl"


def _ts(s):
    return pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce")


def _load_state(series):
    sp = STATE_DIR / f"{series}.json"
    return json.loads(sp.read_text()) if sp.exists() else {}


def _save_state(series, state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{series}.json").write_text(json.dumps(state))


def _new_tier_state():
    return {"cursor": None, "pages": 0, "markets": 0, "done": False,
            "order_breaks_created": 0, "prev_min_created": None}


def crawl_series(series, max_pages=None, pause=0.1, deadline=None, margin_days=30):
    """Crawl one series, both tiers; resumes from the saved cursor. The historical tier stops once a
    whole page was created before TRAIN_START - margin_days: the archive is sorted by created_time,
    so nothing later can close inside the training window if markets live under margin_days."""
    out = CRAWL_DIR / series
    out.mkdir(parents=True, exist_ok=True)
    state = _load_state(series)
    stop_before = TRAIN_START - pd.Timedelta(days=margin_days)

    for tier, path in TIERS:
        st = state.setdefault(tier, _new_tier_state())
        while (not st["done"]
               and (max_pages is None or st["pages"] < max_pages)
               and (deadline is None or time.time() < deadline)):
            params = {"series_ticker": series, "limit": 1000}
            if st["cursor"]:
                params["cursor"] = st["cursor"]
            r = get(path, params)
            mk = r.get("markets", [])
            too_old = False
            if mk:
                df = pd.DataFrame(mk).reindex(columns=COLS)
                df.to_parquet(out / f"{tier}_{st['pages']:05d}.parquet", index=False)
                if tier == "historical":
                    cr = _ts(df["created_time"])
                    prev = st.get("prev_min_created")
                    if prev and cr.max() > pd.Timestamp(prev):
                        st["order_breaks_created"] = st.get("order_breaks_created", 0) + 1
                    st["prev_min_created"] = cr.min().isoformat()
                    too_old = cr.max() < stop_before
            st["pages"] += 1
            st["markets"] += len(mk)
            st["cursor"] = r.get("cursor") or None
            if not st["cursor"] or not mk or too_old:
                st["done"] = True
            _save_state(series, state)                   # progress saved after every page
            time.sleep(pause)
    return state


def reopen_historical(series):
    """Mark the historical tier unfinished so crawl_series continues from its saved cursor.
    Returns the page count at which the earlier crawl had stopped."""
    state = _load_state(series)
    h = state["historical"]
    h.setdefault("pages_before_reopen", h["pages"])
    h["done"] = False
    _save_state(series, state)
    return h["pages_before_reopen"]


def refresh_live(series, pause=0.1):
    """Re-crawl the live tier from scratch. Previous live pages move to a dated snapshot folder."""
    old = sorted((CRAWL_DIR / series).glob("live_*.parquet"))
    if old:
        dest = SNAP_DIR / series / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
        dest.mkdir(parents=True, exist_ok=True)
        for f in old:
            shutil.move(str(f), str(dest / f.name))
    state = _load_state(series)
    state["live"] = _new_tier_state()
    _save_state(series, state)
    return crawl_series(series, pause=pause)


def load_crawl(series, dedupe=True):
    """All crawled pages for a series. With dedupe, a market seen in both tiers keeps its historical
    (settled, final) record."""
    files = sorted((CRAWL_DIR / series).glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=COLS + ["tier"])
    df = pd.concat([pd.read_parquet(f).assign(tier=f.name.split("_")[0]) for f in files], ignore_index=True)
    if dedupe:
        df = (df.assign(_rank=(df["tier"] == "historical").astype(int))
                .sort_values("_rank").drop_duplicates("ticker", keep="last").drop(columns="_rank"))
    return df.reset_index(drop=True)