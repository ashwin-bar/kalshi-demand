# Holdout plan: frozen-pipeline (pseudo-prospective) evaluation

**Status:** fixed at release `v1.0.0`, before any holdout outcome exists (decision-log rows 118, 119).
**Nothing in this plan may change after the release.** If a bug forces a change, see "Bug policy".

## 1. What is evaluated

The pipeline tagged `v1.0.0`: code in `src/` at the tagged commit, configuration in `src/frozen_config.py`, dating rules and guardrails in `src/dating.py`, units in `src/units.py`, features in `src/features.py` and `src/popularity.py`, intervals in `src/intervals.py`. No retuning, no feature changes, no rule changes.

## 2. Windows (by event date, New York time)

| Window | Event dates | Scoring date |
|---|---|---|
| **Short (primary)** | 2026-10-20 to 2026-12-13 | On or after 2026-12-19 (≥5-day buffer for late settlements) |
| Extension (optional) | 2026-12-14 to 2027-01-31 | On or after 2027-02-06 |
| Not scored | 2026-09-28 to 2026-10-19 | Data seen during development; reported as unscored |

A unit belongs to a window by its **event date** (anchor date), not by its issue time. Units with event dates in the window but issue times before 2026-10-20 are included: the pipeline was frozen before any of their outcomes existed.

## 3. Data collection

- Collected after the window ends with the tagged crawler and updater (`src/crawl.py`, `src/updater.py`): archive catch-up before live refresh; series new to the catalogue crawled in full.
- The scope rule, dating rules (re-derived by the tagged `src/dating.py`; saved rules for existing series re-checked with `check_rules`), unit building and feature building are run exactly as tagged.
- **Data revisions** found after the fact (for example rescheduled games with updated milestones) are logged with counts and volume. The main case is already handled by row 49.

## 4. Forecasting procedure

- Weekly retraining on Mondays 00:00 New York time, continuing through the window; each forecast uses the latest model trained only on units that closed before its issue time.
- Training data from 2025-09-01 up to each cutoff, including post-freeze units as they complete.
- Three forecasts per unit (horizons "listing", "T-1d", "T-2h"), issue time = max(anchor − H, listing time), capped 1 second before close (rows 19, 50, 65).
- Outputs per row: median forecast (L1 model), mean forecast (Tweedie, power 1.2), 80% interval (rolling conformal, 8-week residual window by category × horizon).

## 5. Units scored

Event date in the window; complete (no market initialized, inactive or active); not voided; dated by a validated rule (not "review"). Cold units (no completed history in the series at issue time) are scored separately.

## 6. Metrics

| Metric | Applies to | Benchmark |
|---|---|---|
| Volume-weighted WAPE, per week, weeks averaged equally | Median forecast | Head: med28d. Tail: med10. Plus the best baseline in each row |
| Median per-series WAPE | Median forecast | Same |
| Relative error reduction vs benchmark | Median forecast | Same |
| Bias (forecast total ÷ actual − 1) | Median and mean forecasts | mean28d |
| WAPE of totals at series-day, category-day and in-scope total-day | Mean forecast (summed bottom-up) | mean28d summed |
| Spearman rank correlation and top-decile capture within issue-day × category | Median forecast | med28d, weekly |
| 80% interval coverage and median width (high ÷ low) | Intervals | Target 80% |
| Cold units: WAPE | Median forecast | Category median and launch analogue baseline |

## 7. Reporting rows and strata

- Head by category; head total; tail; cold units split by eventual type (one-off / launch); pooled (shown, not used as the headline).
- Horizons reported separately, with the effective horizon distribution.
- Strata: warm (≥20 completed units at issue), lukewarm (1–19), cold (0).
- **Rows with fewer than 100 units or fewer than 5 series are reported but not interpreted.**
- Differences smaller than the validation seed noise band (row 128) are not treated as findings.

## 8. Bug policy

If a bug is found while scoring: fix it, log it in the decision log with what changed and why, and **report both the pre-fix and post-fix scores**.

## 9. What would count as success

Stated in advance: v1 beats the head benchmark (med28d) and the tail benchmark (med10) on volume-weighted WAPE in the short window, and the Tweedie mean's total-day WAPE beats summed mean28d. Results are reported whether or not this holds.