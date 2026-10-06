"""Kalshi public API helpers and project paths."""
import time
from pathlib import Path
import requests

PROJECT_DIR = Path(r"C:\Users\ashwi\kalshi-demand")
DATA_ROOT = Path(r"D:\kalshi_demand")
BASE = "https://api.elections.kalshi.com/trade-api/v2"

_session = requests.Session()


def get(path, params=None, max_retries=6, timeout=30):
    """GET a public endpoint, retrying with growing waits on rate limits (429),
    server errors (5xx) and network drops. Kalshi gives no Retry-After header."""
    for attempt in range(max_retries):
        wait = 2 ** attempt
        try:
            r = _session.get(f"{BASE}{path}", params=params, timeout=timeout)
        except requests.RequestException as e:
            print(f"  network error on {path}: {e!r}; retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            print(f"  HTTP {r.status_code} on {path}; retry in {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Gave up on {path} after {max_retries} attempts")


def paginate(path, key, params=None, limit=1000, max_pages=None, pause=0.1):
    """Follow Kalshi's page cursors and return every item listed under `key`."""
    params = dict(params or {})
    params["limit"] = limit
    items, pages = [], 0
    while True:
        resp = get(path, params)
        items.extend(resp.get(key, []))
        pages += 1
        cursor = resp.get("cursor")
        if not cursor or (max_pages and pages >= max_pages):
            return items
        params["cursor"] = cursor
        time.sleep(pause)


def get_cutoff():
    """Timestamps where the live tier ends and the historical tier begins."""
    return get("/historical/cutoff")


def to_float(x):
    """Kalshi sends counts and prices as text like '830.14'. Empty or missing becomes NaN."""
    return float(x) if x not in (None, "") else float("nan")