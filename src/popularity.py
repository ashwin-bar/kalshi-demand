"""Team/player popularity features (row 67, tier 3), built only from our own data and only from events
that closed strictly before each forecast's issue time (row 69)."""
import pandas as pd

WINNER_SERIES_REGEX = r"(?:GAME|MATCH|FIGHT)$"   # game-winner series: one market per side
NON_PARTICIPANT = {"TIE", "DRAW", "NONE"}
K = 10                                            # popularity = mean relative size of the last K completed events
G4 = ["pop_max", "pop_mean", "pop_min", "pop_games_min", "n_participants"]


def participants(con, clean_globs, units):
    """Participant of every market in winner-series events: the ticker suffix after the last '-', keyed by series."""
    union = " UNION ALL ".join(f"SELECT series, event_ticker, ticker FROM '{g}'" for g in clean_globs)
    win = units.loc[~units["hf"] & units["series"].str.contains(WINNER_SERIES_REGEX, regex=True), ["unit_id"]]
    con.register("_win_units", win)
    p = con.execute(f"""
        SELECT m.series, m.event_ticker AS unit_id, regexp_extract(m.ticker, '-([^-]+)$', 1) AS code
        FROM ({union}) m JOIN _win_units w ON m.event_ticker = w.unit_id""").df()
    p = p[(p["code"] != "") & ~p["code"].str.upper().isin(NON_PARTICIPANT)]
    p["participant"] = p["series"] + ":" + p["code"]
    return p[["unit_id", "participant"]].drop_duplicates()


def popularity_history(parts, feat):
    """Per participant, in close order: mean relative size (realised y at T-1d) of its last K completed winner events."""
    rel = feat.loc[feat["horizon"].astype(str) == "T-1d", ["unit_id", "close_ts", "y"]].dropna(subset=["y"])
    h = parts.merge(rel, on="unit_id").sort_values(["participant", "close_ts"]).reset_index(drop=True)
    h["pop_mean_k"] = h.groupby("participant")["y"].rolling(K, min_periods=1).mean().reset_index(level=0, drop=True)
    h["pop_games"] = h.groupby("participant").cumcount() + 1
    return h[["participant", "close_ts", "pop_mean_k", "pop_games"]].rename(columns={"close_ts": "pop_ts"})


def add_popularity(feat, units, parts):
    """For every feature row on a fixture: popularity of the fixture's participants, as known at the issue time."""
    fx = units.loc[units["fixture_id"].notna(), ["unit_id", "fixture_id"]]
    fix_parts = parts.merge(fx, on="unit_id")[["fixture_id", "participant"]].drop_duplicates()
    rows = feat.loc[feat["fixture_id"].notna(), ["unit_id", "horizon", "issue_ts", "fixture_id"]]
    rp = rows.merge(fix_parts, on="fixture_id")
    hist = popularity_history(parts, feat).sort_values("pop_ts")
    rp = pd.merge_asof(rp.sort_values("issue_ts"), hist, by="participant", left_on="issue_ts", right_on="pop_ts",
                       direction="backward", allow_exact_matches=False)          # strictly earlier closes only
    agg = (rp.groupby(["unit_id", "horizon"], observed=True)
             .agg(pop_max=("pop_mean_k", "max"), pop_mean=("pop_mean_k", "mean"), pop_min=("pop_mean_k", "min"),
                  pop_games_min=("pop_games", "min"), n_participants=("participant", "nunique"))
             .reset_index())
    return feat.merge(agg, on=["unit_id", "horizon"], how="left")