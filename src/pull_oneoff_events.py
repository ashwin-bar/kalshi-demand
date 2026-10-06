"""Background pull: events + milestones for tail series with 1-19 events since 2025 (one-offs and young series).
Run from the project folder:   .venv\\Scripts\\python src\\pull_oneoff_events.py
Safe to stop with Ctrl+C and rerun: series already saved are skipped."""
import time
import pandas as pd
from kalshi_api import DATA_ROOT
from events import pull_events, EVENTS_DIR

prof = pd.read_parquet(DATA_ROOT / "interim" / "tail_profile.parquet")
targets = (prof[(prof["ev_2025"] >= 1) & (prof["ev_2025"] < 20)]
             .sort_values("vol_2025", ascending=False)["series"].tolist())
todo = [s for s in targets if not (EVENTS_DIR / f"{s}__events.json").exists()]
print(f"{len(targets):,} target series, {len(todo):,} still to pull", flush=True)

t0, errors = time.time(), 0
for i, s in enumerate(todo, 1):
    try:
        pull_events(s, pause=0.1)
    except Exception as e:
        errors += 1
        print(f"  {s}: ERROR {e!r}", flush=True)
    time.sleep(0.05)
    if i % 250 == 0:
        print(f"  {i:,}/{len(todo):,} series | {(time.time() - t0) / 60:.0f} min | errors {errors}", flush=True)
print(f"Done: {len(todo) - errors:,} pulled, {errors} errors, {(time.time() - t0) / 60:.0f} min", flush=True)