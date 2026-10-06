"""Events and milestones per series. Pulled once per series, then read from D:."""
import json
import time
import pandas as pd
from kalshi_api import get, DATA_ROOT

EVENTS_DIR = DATA_ROOT / "raw" / "events"


def pull_events(series, pause=0.1):
    """All events for a series with milestones attached (same file format as Block 11)."""
    f = EVENTS_DIR / f"{series}__events.json"
    if f.exists():
        return json.loads(f.read_text())
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    params = {"series_ticker": series, "limit": 200, "with_milestones": "true"}
    events, milestones = [], []
    while True:
        r = get("/events", params)
        events += r.get("events", [])
        milestones += r.get("milestones") or []
        if not r.get("cursor"):
            break
        params["cursor"] = r["cursor"]
        time.sleep(pause)
    out = {"events": events, "milestones": milestones}
    f.write_text(json.dumps(out))                 # written only once the whole series is pulled
    return out


def event_milestones(raw):
    """One row per event with its earliest related milestone: id (= fixture id), start and end (UTC)."""
    ev = pd.DataFrame({"event_ticker": [e["event_ticker"] for e in raw["events"]]})
    ms = pd.DataFrame(raw["milestones"])
    if ms.empty or "related_event_tickers" not in ms.columns:
        ev["milestone_id"] = None
        for col in ["ms_start", "ms_end"]:
            ev[col] = pd.Series(pd.NaT, index=ev.index, dtype="datetime64[ns, UTC]")
        return ev
    ms = ms.explode("related_event_tickers").dropna(subset=["related_event_tickers"])
    ms["ms_start"] = pd.to_datetime(ms["start_date"], utc=True, format="ISO8601", errors="coerce")
    ms["ms_end"] = (pd.to_datetime(ms["end_date"], utc=True, format="ISO8601", errors="coerce")
                    if "end_date" in ms.columns else pd.NaT)
    first = (ms.sort_values("ms_start").drop_duplicates("related_event_tickers")
               [["related_event_tickers", "id", "ms_start", "ms_end"]]
               .rename(columns={"related_event_tickers": "event_ticker", "id": "milestone_id"}))
    return ev.merge(first, on="event_ticker", how="left")