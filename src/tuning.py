"""Light HPO for the median model (rows 92, 97, 102, 111). Objective declared in advance (row 111):
mean over horizons of (0.5 x head-total WAPE + 0.5 x tail WAPE), validation weeks, warm + lukewarm rows."""
import numpy as np
import pandas as pd
import optuna
from model_v1 import PARAMS, week_cutoffs, fit_predict, score_cols

VAL_START, VAL_END = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-03-01")
TUNED_KEYS = ["num_leaves", "min_child_samples", "learning_rate", "n_estimators", "subsample", "colsample_bytree", "reg_lambda"]


def journal_storage(path):
    """File-based Optuna storage on D: (no database), resumable; uses an open-file lock that works on Windows."""
    try:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
        return JournalStorage(JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path))))
    except ImportError:
        from optuna.storages import JournalStorage, JournalFileStorage, JournalFileOpenLock
        return JournalStorage(JournalFileStorage(str(path), lock_obj=JournalFileOpenLock(str(path))))


def objective_value(preds):
    d = preds[preds["med10"].notna()].copy()
    d["segment"] = d["segment"].astype(str)
    w = score_cols(d, ["m"], by=["segment"]).groupby(["horizon", "segment"], observed=True)["wape"].mean().unstack()
    return float((0.5 * w["head"] + 0.5 * w["tail"]).mean()), w


def evaluate(df, val_mask, feats, params, weight_mode, every):
    """Median model on the validation weeks with retraining every `every` weeks; returns (objective, table, preds)."""
    p = fit_predict(df, val_mask, feats, "l1", week_cutoffs(VAL_START, VAL_END, every=every), params=params,
                    weight_col="_w" if weight_mode == "baseline" else None)
    keep = ["unit_id", "horizon", "segment", "category", "series", "event_date", "issue_ts", "vol", "denom",
            "med10", "med28d"]
    preds = df.loc[val_mask, keep].merge(p[["unit_id", "horizon", "yhat"]], on=["unit_id", "horizon"])
    preds["m"] = np.expm1(np.log1p(preds["denom"]) + preds["yhat"]).clip(lower=0)
    val, w = objective_value(preds)
    return val, w, preds


def make_objective(df, val_mask, feats, every=4, on_trial=None):
    def objective(trial):
        params = dict(num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
                      min_child_samples=trial.suggest_int("min_child_samples", 50, 1000, log=True),
                      learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                      n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
                      subsample=trial.suggest_float("subsample", 0.6, 1.0),
                      colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                      reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        wm = trial.suggest_categorical("weight_mode", ["none", "baseline"])
        val, w, _ = evaluate(df, val_mask, feats, params, wm, every)
        trial.set_user_attr("head", float(w["head"].mean()))
        trial.set_user_attr("tail", float(w["tail"].mean()))
        if on_trial:
            on_trial(trial.number, params, wm, val, w)
        return val
    return objective


def current_trial():
    """The current (Block 51) settings, enqueued as trial 0 so the search competes against them."""
    t = {k: PARAMS[k] for k in TUNED_KEYS}
    t["weight_mode"] = "none"
    return t