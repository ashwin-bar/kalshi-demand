"""Event dating as reusable code (decision rows 14-15, 20, 29-34, 41, 49, 68, 72, 86).
Rules are chosen and validated on completed, non-voided in-window events. The same qualification
test chooses rules (assign_rules) and re-checks saved rules on fresh data (check_rules, row 43).
No series-specific overrides: every rule is general."""
import numpy as np
import pandas as pd
from events import pull_events, event_milestones

ET = "America/New_York"
MENTION_WINDOW_H = 36            # mention markets resolve next morning (row 34)
MIN_EVENTS_FOR_PATTERN = 5       # a series-level pattern (fixed offset) needs at least this many completed events (row 86)
MAX_OFFSET_DAYS = 7              # an event date more than a week from the close is not a dated occurrence near trading (row 86)
MAX_MULTIDAY_H = 168             # multi-day events up to a week (tournaments, playoff series); longer = out of scope (row 86)
PRIORITY = ["milestone", "milestone_multiday", "milestone_multiday_partial", "name_date", "close_offset"]


def _ns(s):
    return s.dt.as_unit("ns")


def load_events(con, clean_glob, series_list):
    """One row per in-window event: open/close, volume, completeness and voided flags."""
    con.register("_dating_scope", pd.DataFrame({"series": list(series_list)}))
    ev = con.execute(f"""
      SELECT m.series, m.event_ticker, min(m.open_ts) AS open_ts, max(m.close_ts) AS close_ts,
             count(*) AS n_mkts, sum(m.volume) AS vol,
             bool_and(m.status NOT IN ('initialized', 'inactive', 'active')) AS complete,
             bool_and(m.result IS NULL OR m.result = '') AS no_result
      FROM '{clean_glob}' m JOIN _dating_scope USING (series) GROUP BY ALL
      HAVING max(m.close_ts) >= TIMESTAMPTZ '2025-01-01 00:00:00 America/New_York'""").df()
    for c in ["open_ts", "close_ts"]:
        ev[c] = _ns(pd.to_datetime(ev[c], utc=True))
    ev["voided"] = ev["complete"] & ev["no_result"]
    return ev


def add_date_inputs(ev, hf_series):
    """Attach milestones (from event files on D:) and every candidate date."""
    hf_series = set(hf_series)
    nonhf = sorted(set(ev["series"]) - hf_series)
    parts = [event_milestones(pull_events(s)) for s in nonhf]
    em = (pd.concat(parts, ignore_index=True).drop_duplicates("event_ticker") if parts
          else pd.DataFrame(columns=["event_ticker", "milestone_id", "ms_start", "ms_end"]))
    ev = ev.merge(em, on="event_ticker", how="left")
    for c in ["ms_start", "ms_end"]:
        ev[c] = _ns(pd.to_datetime(ev[c], utc=True))
    ev["hf"] = ev["series"].isin(hf_series)
    ev["close_day"] = _ns(pd.to_datetime(ev["close_ts"].dt.tz_convert(ET).dt.date))
    ev["ms_day"] = _ns(pd.to_datetime(ev["ms_start"].dt.tz_convert(ET).dt.date))
    ev["name_day"] = _ns(pd.to_datetime(ev["event_ticker"].str.split("-").str[1].str[:7],
                                        format="%y%b%d", errors="coerce"))
    ev["ms_off"] = (ev["ms_day"] - ev["close_day"]).dt.days
    ev["name_off"] = (ev["name_day"] - ev["close_day"]).dt.days
    ev["close_after_h"] = (ev["close_ts"] - ev["ms_start"]).dt.total_seconds() / 3600
    ev["dur_h"] = (ev["ms_end"] - ev["ms_start"]).dt.total_seconds() / 3600
    return ev


def _mode_share(x):
    x = x.dropna()
    if x.empty:
        return pd.Series({"mode": np.nan, "share": np.nan})
    m = x.mode().iloc[0]
    return pd.Series({"mode": m, "share": (x == m).mean()})


def series_stats(ev):
    """Per-series evidence for each candidate rule, from completed, non-voided, non-hourly events."""
    b = ev[~ev["hf"] & ev["complete"] & ~ev["voided"]]
    s, g = b["series"], b.groupby("series")
    has = b["ms_start"].notna()
    st = pd.DataFrame({"events": g.size(), "volume": g["vol"].sum(),
                       "ms_cov": has.groupby(s).mean(), "dur_med_h": g["dur_h"].median(),
                       "name_cov": b["name_day"].notna().groupby(s).mean()})
    window = b["dur_h"].where(b["dur_h"] > 0).fillna(s.map(st["dur_med_h"])).fillna(0) + 12
    window = window.where(~s.str.contains("MENTION"), MENTION_WINDOW_H)
    st["timing_ok"] = (has & b["close_after_h"].between(0, window)).groupby(s).sum() / has.groupby(s).sum()
    st["started_ok"] = (has & (b["close_after_h"] >= 0)).groupby(s).sum() / has.groupby(s).sum()
    named = b["name_day"].notna()
    st["name_in_window"] = (named & b["name_off"].isin([-1, 0])).groupby(s).sum() / named.groupby(s).sum()
    st = st.join(g["ms_off"].apply(_mode_share).unstack().add_prefix("ms_off_"))
    st["exp_misdate"] = (1 - st["ms_cov"]) * (1 - st["ms_off_share"].fillna(0))
    return st


def _qualifies(rule, r):
    """The single validation test for each rule (used to choose rules and to re-check them)."""
    multi = r["dur_med_h"] > 24
    week = r["dur_med_h"] <= MAX_MULTIDAY_H
    ms_ok = r["timing_ok"] >= 0.85 and (r["ms_cov"] >= 0.95 or (r["ms_cov"] >= 0.80 and r["exp_misdate"] <= 0.05))
    if rule == "milestone":
        return bool(ms_ok and not multi)
    if rule == "milestone_multiday":
        return bool(ms_ok and multi and week)
    if rule == "milestone_multiday_partial":
        return bool(multi and week and r["ms_cov"] >= 0.50 and r["started_ok"] >= 0.85)
    if rule == "name_date":
        return bool(r["name_cov"] >= 0.99 and r["name_in_window"] >= 0.99)
    if rule.startswith("close") and rule != "close+0 (hourly)":
        return bool(r["events"] >= MIN_EVENTS_FOR_PATTERN and r["ms_cov"] >= 0.20
                    and r["ms_off_share"] >= 0.99 and abs(r["ms_off_mode"]) <= MAX_OFFSET_DAYS)
    return rule == "close+0 (hourly)"


def _auto_rule(r):
    for rule in PRIORITY:
        if rule == "close_offset":
            if _qualifies("close+0", r):
                return f"close{int(r['ms_off_mode']):+d}"
        elif _qualifies(rule, r):
            return rule
    return "review"


def assign_rules(ev):
    """Rule per series from the evidence; hourly-or-faster series use close day (row 24)."""
    cfg = series_stats(ev)
    cfg["rule"] = cfg.apply(_auto_rule, axis=1)
    hf = (ev[ev["hf"]].groupby("series").agg(events=("event_ticker", "size"), volume=("vol", "sum"))
            .assign(rule="close+0 (hourly)"))
    cfg = pd.concat([cfg, hf])
    cfg.index.name = "series"
    return cfg


def date_events(ev, cfg):
    """Event date, date source, anchor (capped at close), multi-day flag and fixture id for every event."""
    ev = ev.copy()
    ev["rule"] = ev["series"].map(cfg["rule"]).fillna("review")
    fb = pd.to_timedelta(ev["series"].map(cfg.get("ms_off_mode")).fillna(0), unit="D")
    ev["event_date"] = pd.Series(pd.NaT, index=ev.index, dtype="datetime64[ns]")
    ev["date_source"] = "review"
    ms_rule = ev["rule"].isin(["milestone", "milestone_multiday", "milestone_multiday_partial"])
    has_ms = ev["ms_day"].notna()

    sel = ms_rule & has_ms
    ev.loc[sel, "event_date"], ev.loc[sel, "date_source"] = ev.loc[sel, "ms_day"], "milestone"
    sel = ev["rule"].isin(["milestone", "milestone_multiday"]) & ~has_ms          # fallback (row 31)
    ev.loc[sel, "event_date"], ev.loc[sel, "date_source"] = ev.loc[sel, "close_day"] + fb[sel], "milestone_fallback"
    sel = ev["rule"] == "name_date"
    ev.loc[sel, "event_date"], ev.loc[sel, "date_source"] = ev.loc[sel, "name_day"], "name_date"
    sel = ev["rule"].str.match(r"close[+-]\d+$")
    k = pd.to_timedelta(ev.loc[sel, "rule"].str.extract(r"close([+-]\d+)")[0].astype(int), unit="D")
    ev.loc[sel, "event_date"], ev.loc[sel, "date_source"] = ev.loc[sel, "close_day"] + k, "close_offset"
    sel = ev["hf"]
    ev.loc[sel, "event_date"], ev.loc[sel, "date_source"] = ev.loc[sel, "close_day"], "close_day_hourly"

    stale = (ev["date_source"] == "milestone") & ((ev["ms_start"] - ev["close_ts"]).dt.total_seconds() > 3600)
    ev.loc[stale, "event_date"] = ev.loc[stale, "close_day"] + fb[stale]
    ev.loc[stale, "date_source"] = "milestone_stale_fallback"

    midnight = ev["event_date"].dt.tz_localize(ET).dt.tz_convert("UTC")
    anchor = ev["ms_start"].where(ev["date_source"] == "milestone", midnight)
    ev["anchor_capped"] = anchor > ev["close_ts"]
    ev["anchor_eff"] = anchor.where(~ev["anchor_capped"], ev["close_ts"])
    ev["multiday"] = ev["rule"].isin(["milestone_multiday", "milestone_multiday_partial"])
    ev["fixture_id"] = ev["milestone_id"].where(ms_rule)
    return ev


def check_rules(ev, saved_rules):
    """Automated test (row 43): recompute the evidence on current data; flag series whose saved rule
    no longer qualifies under exactly the test used to choose it."""
    st = series_stats(ev).join(saved_rules.rename("rule"), how="inner")
    fails = [(s, r["rule"]) for s, r in st.iterrows()
             if r["rule"] != "review" and not _qualifies(r["rule"], r)]
    return pd.DataFrame(fails, columns=["series", "rule"])


def review_reason(r):
    """Why a series failed every rule (for reporting excluded volume by reason, row 72)."""
    if r["ms_cov"] < 0.20 and r["name_cov"] < 0.99:
        return "no usable dates"
    if r["dur_med_h"] > MAX_MULTIDAY_H:
        return "longer than a week (season-long)"
    if r["dur_med_h"] > 24:
        return "multi-day / delayed resolution"
    if r["timing_ok"] < 0.85:
        return "milestone timing fails"
    if r["events"] < MIN_EVENTS_FOR_PATTERN and r["ms_cov"] >= 0.20:
        return "too few events for an offset pattern"
    if r["ms_cov"] < 0.95:
        return "partial milestone coverage"
    return "inconsistent name dates"