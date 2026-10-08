# Model card: Kalshi event demand forecaster, v1.0.0

## Overview

| | |
|---|---|
| **What it forecasts** | Total contracts traded per event on Kalshi (per series-day for hourly and 15-minute series) |
| **Forecasts per event** | At listing, 1 day before, and 2 hours before the event |
| **Outputs** | Median forecast, mean forecast, 80% interval |
| **Intended use** | Demonstrating demand forecasting for event contracts: ranking which events will draw volume (listing and liquidity decisions) and forecasting daily totals (capacity and finance). Not trading advice |
| **Version** | Git tag `v1.0.0`; configuration in `src/frozen_config.py` |
| **Data** | Kalshi public API only; crawl of 4 Oct 2026 (backtest data frozen, row 116) |

## Data

- **Scope:** 11,273 series crawled (130 head + 11,143 tail), about 17.2M markets. Combo (multivariate) markets, never-traded series, one-off and outright-winner head series, and series whose events can't be dated by a validated rule are excluded.
- **Units:** 429,201 forecasting units (184,369 head, 244,832 tail), each an event or a series-day, dated by validated rules: milestone start, date in the event name, or a close-day offset with at least 5 events of evidence (rows 75, 86).
- **Periods:** history only before 2025-09; training from 2025-09-01; validation 2026-01-05 to 03-01; test 2026-03-02 to 09-27; holdout after the freeze (see `docs/holdout_plan.md`).

## Target and models

- **Target:** y = log1p(volume) − log1p(baseline), where the baseline is the series' median unit volume over the 28 days before the issue time (category median for brand-new series). The model learns how far an event will land above or below its series' recent norm.
- **Median model:** LightGBM, L1 loss on y. Tuned settings in `src/frozen_config.py` (seed 42, deterministic).
- **Mean model:** LightGBM Tweedie on volume, log link, log1p(baseline) as offset, variance power 1.2.
- **Intervals:** rolling conformal: 10th/90th percentiles of residuals from units closed in the 8 weeks before the issue day, by category × horizon.
- **Training:** weekly retraining; every forecast uses only a model trained on units that closed before its issue time.

## Features (35)

| Group | Count | Examples | Validation ablation (head WAPE when removed) |
|---|---|---|---|
| G1 event characteristics | 16 | Day and hour of the event, listing lead, strikes listed so far, fixture size, slate size | +7.7 points |
| G2 series history | 10 | 28-day and 10-unit medians, weekly naive, growth, skew, units completed | +5.3 points (tail +13.4) |
| G3 market context | 4 | Exchange and category volume over the last 7 days | −0.15 (within noise; crypto +1.7) |
| G4 team/player popularity | 5 | Mean relative size of each participant's last 10 completed events | +4.1 points (sports +4.6) |

Every feature uses only information available at the issue time (row 69). Leakage checks pass.

## Validation results (T−1d, 8 weeks, seed 42)

| Row | Benchmark | v1 | Relative |
|---|---|---|---|
| Head total | 65.7% (med28d) | 51.0% | −22% |
| Head: sports | 67.9% | 53.7% | −21% |
| Head: crypto | 26.6% (best baseline) | 22.8% | −14% |
| Tail | 67.1% (med10) | 60.6% | −10% |

- **Seed noise band:** objective spread 0.0012 across 3 seeds; tuning gain 0.0095.
- **Ranking:** top-decile capture 64.6% overall, about 70% in sports (benchmark about 55%).
- **Mean model totals:** in-scope total-day WAPE 14.8% vs mean28d 25.9%.
- **Intervals:** pooled 80% coverage 79.2% (T−1d); median width 13× (high ÷ low).
- **Test results:** to be added after the single test scoring.

## Known limitations

1. **Launch-ramp bias in the mean model:** it under-forecasts fast-growing new series in their first weeks (crypto −14% on validation, concentrated in early January).
2. **Interval calibration varies by segment:** head over-covered (83%), tail under-covered (74%), crypto far too wide (99%), because pools are by category rather than category × segment.
3. **Wide unit-level uncertainty:** single-event forecasts are inherently noisy; use rankings and aggregates for decisions.
4. **Excluded markets:** one-off "will X happen by…" questions without a scheduled date; 68 tail series (5.0% of tail volume) that fail every dating rule (row 75).
5. **Multi-day events** (golf tournaments, playoff series) are dated by their start.
6. **Small rows** (fewer than 100 units or 5 series, e.g. economics = one Fed meeting) are reported but not interpreted.
7. **Popularity is keyed by series:** the same team in different series builds separate histories.
8. **Market context features** add nothing overall on validation; kept because the feature set was fixed before tuning (row 130).
9. **Holdout is pseudo-prospective:** the pipeline is frozen in advance, but the data is collected afterwards and may include small after-the-fact corrections.
10. **Public data only:** Kalshi's own data (user activity, order flow) would support a stronger model.