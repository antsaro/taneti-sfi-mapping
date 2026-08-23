"""Nested cross-validation for the Random Forest models, one per SFI depth
layer. The outer loop gives an honest, unbiased estimate of generalization
error; the inner loop is where Optuna searches for hyperparameters.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from config import (COVARIATE_NAMES, FINAL_GPKG, N_FOLDS_INNER, N_FOLDS_OUTER,
                     OPTUNA_TRIALS_CV, RESPONSES, RESULTS_DIR, SEED)
from modeling.common import calc_metrics, impute_train_test, tune_rf


def run_nested_cv(df, response, covariates, outer_k=N_FOLDS_OUTER,
                   n_trials=OPTUNA_TRIALS_CV, seed=SEED):
    # response NaNs are dropped, not imputed: imputing a target manufactures
    # observations that were never measured and shrinks its variance.
    sub = df.dropna(subset=[response]).reset_index(drop=True)
    X_all, y_all = sub[covariates].values, sub[response].values.astype(np.float64)

    kf = KFold(n_splits=outer_k, shuffle=True, random_state=seed)
    metrics, predictions, params_used = [], [], []

    for i, (tr, te) in enumerate(kf.split(X_all), 1):
        X_tr, X_te = impute_train_test(X_all[tr], X_all[te], covariates)
        y_tr, y_te = y_all[tr], y_all[te]

        best_params = tune_rf(X_tr, y_tr, n_trials, N_FOLDS_INNER)
        rf = RandomForestRegressor(**best_params)
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_te)

        m = calc_metrics(y_te, preds)
        m["fold"] = i
        metrics.append(m)
        predictions.append(pd.DataFrame({"fold": i, "obs": y_te, "pred": preds}))
        params_used.append({**best_params, "fold": i})

    return dict(
        metrics=pd.DataFrame(metrics),
        predictions=pd.concat(predictions, ignore_index=True),
        best_params=pd.DataFrame(params_used),
    )


if __name__ == "__main__":
    df = gpd.read_file(FINAL_GPKG).drop(columns="geometry")
    cv_results = {r: run_nested_cv(df, r, COVARIATE_NAMES) for r in RESPONSES}

    summary = pd.DataFrame([
        {"response": r,
         **{f"{k}_mean": cv_results[r]["metrics"][k].mean() for k in ["R2", "RMSE", "MAE", "RPIQ", "CCC"]},
         **{f"{k}_sd": cv_results[r]["metrics"][k].std() for k in ["R2", "RMSE", "MAE", "RPIQ", "CCC"]}}
        for r in RESPONSES
    ])
    summary.to_csv(os.path.join(RESULTS_DIR, "cv_summary.csv"), index=False)

    best_params = pd.concat(
        [cv_results[r]["best_params"].assign(response=r) for r in RESPONSES], ignore_index=True
    ).sort_values(["response", "fold"])
    best_params.to_csv(os.path.join(RESULTS_DIR, "best_params_per_fold.csv"), index=False)

    print(summary)
