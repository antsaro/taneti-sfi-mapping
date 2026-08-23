"""Metrics, leakage-safe imputation and Optuna-based RF tuning shared by the
nested-CV, kriging and RFE scripts. Pulled out into one place because the
original notebook had three near-identical copies of each function.
"""

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from config import N_JOBS_CPU

optuna.logging.set_verbosity(optuna.logging.WARNING)


def calc_metrics(obs, pred):
    obs, pred = np.asarray(obs, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    if len(obs) < 2:
        return dict(N=len(obs), R2=np.nan, RMSE=np.nan, MAE=np.nan, RPIQ=np.nan, CCC=np.nan)

    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae = mean_absolute_error(obs, pred)
    r2 = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)
    iqr = np.percentile(obs, 75) - np.percentile(obs, 25)
    rpiq = iqr / rmse if rmse > 0 else 0.0

    mx, my = obs.mean(), pred.mean()
    vx, vy = obs.var(ddof=0), pred.var(ddof=0)
    sxy = np.cov(obs, pred, ddof=0)[0, 1]
    ccc_den = vx + vy + (mx - my) ** 2
    ccc = (2 * sxy) / ccc_den if ccc_den > 0 else 0.0

    return dict(N=int(len(obs)), R2=r2, RMSE=rmse, MAE=mae, RPIQ=rpiq, CCC=ccc)


def impute_train_test(X_tr, X_te, covariate_names):
    """Median-impute using train-fold medians only, applied to both splits.
    Falls back to 0 for a covariate that's entirely NaN in the train fold."""
    tr_df = pd.DataFrame(X_tr, columns=covariate_names)
    te_df = pd.DataFrame(X_te, columns=covariate_names)
    medians = tr_df.median()
    medians[medians.isna()] = 0.0
    return tr_df.fillna(medians).values, te_df.fillna(medians).values


def make_rf_objective(X_np, y_np, n_inner):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 15),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=42,
            n_jobs=-1,
        )
        inner_kf = KFold(n_splits=n_inner, shuffle=True, random_state=0)
        scores = []
        for tr, va in inner_kf.split(X_np):
            model = RandomForestRegressor(**params)
            model.fit(X_np[tr], y_np[tr])
            scores.append(np.sqrt(mean_squared_error(y_np[va], model.predict(X_np[va]))))
        return float(np.mean(scores))
    return objective


def tune_rf(X_np, y_np, n_trials, n_inner):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_rf_objective(X_np, y_np, n_inner), n_trials=n_trials,
                    n_jobs=N_JOBS_CPU, show_progress_bar=False)
    return {**study.best_params, "random_state": 42, "n_jobs": -1}
