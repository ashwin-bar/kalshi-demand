"""Forecasting units from dated events (rows 24, 33, 41, 47, 49, 59): one unit per event for ordinary
series, one per series-day for hourly-or-faster series. Returns the units and an exclusion log."""
import pandas as pd

ET = "America/New_York"
COLS = ["unit_id", "series", "category", "event_date", "anchor_eff", "open_ts", "close_ts", "vol",
        "n_mkts", "n_events", "hf", "multiday", "date_source", "fixture_id", "anchor_capped"]


def build_units(dated, category):
    ev = dated.copy()
    ev["category"] = ev["series"].map(category)
    bad_days = (ev.loc[ev["hf"] & ~ev["complete"], ["series", "event_date"]]
                  .drop_duplicates().assign(partial_day=True))
    ev = ev.merge(bad_days, on=["series", "event_date"], how="left")
    ev["partial_day"] = ev["partial_day"].fillna(False).astype(bool)

    reasons = {
        "incomplete (row 33)":       ~ev["complete"],
        "voided (row 47)":           ev["voided"] & ev["complete"],
        "undated or review (59)":    ev["event_date"].isna() | (ev["date_source"] == "review"),
        "hourly partial day":        ev["partial_day"] & ev["complete"],
    }
    drop = pd.Series(False, index=ev.index)
    log = []
    for k, m in reasons.items():
        log.append({"reason": k, "events": int(m.sum()), "vol_share": ev.loc[m, "vol"].sum() / ev["vol"].sum()})
        drop |= m
    keep = ev[~drop]

    n = keep[~keep["hf"]].copy()
    n["unit_id"] = n["event_ticker"]
    n["n_events"] = 1

    h = (keep[keep["hf"]].groupby(["series", "category", "event_date"], as_index=False)
           .agg(open_ts=("open_ts", "min"), close_ts=("close_ts", "max"), vol=("vol", "sum"),
                n_mkts=("n_mkts", "sum"), n_events=("event_ticker", "size")))
    h["unit_id"] = h["series"] + "|" + h["event_date"].dt.strftime("%Y-%m-%d")
    midnight = h["event_date"].dt.tz_localize(ET).dt.tz_convert("UTC").dt.as_unit("ns")
    h["anchor_capped"] = midnight > h["close_ts"]
    h["anchor_eff"] = midnight.where(~h["anchor_capped"], h["close_ts"])
    h["hf"], h["multiday"], h["date_source"], h["fixture_id"] = True, False, "close_day_hourly", None

    units = (pd.concat([n[COLS], h[COLS]], ignore_index=True)
               .sort_values(["series", "anchor_eff"]).reset_index(drop=True))
    for c in ["anchor_eff", "open_ts", "close_ts"]:
        units[c] = units[c].dt.as_unit("ns")
    units["event_date"] = units["event_date"].dt.as_unit("ns")
    return units, pd.DataFrame(log)