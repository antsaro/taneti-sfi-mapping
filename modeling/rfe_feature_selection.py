"""Nested CV with an RFECV feature-selection step inside each outer training
fold, to see whether trimming the covariate set changes performance and
which covariates get kept most consistently across folds.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold

from config import (COVARIATE_NAMES, FINAL_GPKG, N_FOLDS_INNER, N_FOLDS_OUTER, N_JOBS_CPU,
                     OPTUNA_TRIALS_CV, RESPONSES, RESULTS_DIR, SEED)
from modeling.common import calc_metrics, tune_rf

RFE_STEP = 1
RFE_MIN_FEATURES = 3


def run_nested_cv_rfe(df, response, covariates, outer_k=N_FOLDS_OUTER, inner_k=N_FOLDS_INNER,
                       n_trials=OPTUNA_TRIALS_CV, seed=SEED, min_features=RFE_MIN_FEATURES, step=RFE_STEP):
    X_all = df[covariates].values
    y_all = df[response].values.astype(np.float64)
    kf = KFold(n_splits=outer_k, shuffle=True, random_state=seed)

    metrics, predictions, selected_features = [], [], []

    for i, (tr, te) in enumerate(kf.split(X_all), 1):
        X_tr, X_te = X_all[tr], X_all[te]
        y_tr, y_te = y_all[tr], y_all[te]

        inner_kf = KFold(n_splits=inner_k, shuffle=True, random_state=0)
        rfecv = RFECV(estimator=RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
                      step=step, cv=inner_kf, scoring="neg_root_mean_squared_error",
                      min_features_to_select=min_features, n_jobs=N_JOBS_CPU)
        rfecv.fit(X_tr, y_tr)

        selected = [c for c, keep in zip(covariates, rfecv.support_) if keep]
        X_tr_sel, X_te_sel = X_tr[:, rfecv.support_], X_te[:, rfecv.support_]

        best_params = tune_rf(X_tr_sel, y_tr, n_trials, inner_k)
        rf = RandomForestRegressor(**best_params)
        rf.fit(X_tr_sel, y_tr)
        preds = rf.predict(X_te_sel)

        m = calc_metrics(y_te, preds)
        m["fold"], m["n_features"] = i, len(selected)
        metrics.append(m)
        predictions.append(pd.DataFrame({"fold": i, "obs": y_te, "pred": preds}))
        selected_features.append({"fold": i, "response": response, "n_features": len(selected),
                                   "features": ";".join(selected)})
        print(f"  {response} fold {i}/{outer_k}  R2={m['R2']:.3f}  n_features={len(selected)}")

    return dict(metrics=pd.DataFrame(metrics), predictions=pd.concat(predictions, ignore_index=True),
                selected_features=pd.DataFrame(selected_features))


def feature_selection_frequency(selected_df, covariates, outer_k):
    counts = {c: 0 for c in covariates}
    for feats in selected_df["features"]:
        for f in feats.split(";"):
            if f in counts:
                counts[f] += 1
    freq = pd.DataFrame({"feature": list(counts.keys()), "times_selected": list(counts.values())})
    freq["selection_rate"] = freq["times_selected"] / outer_k
    return freq.sort_values("selection_rate", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = gpd.read_file(FINAL_GPKG).drop(columns="geometry")
    results = {r: run_nested_cv_rfe(df, r, COVARIATE_NAMES) for r in RESPONSES}

    summary = pd.DataFrame([
        {"response": r,
         **{f"{k}_mean": results[r]["metrics"][k].mean() for k in ["R2", "RMSE", "MAE", "RPIQ", "CCC"]},
         "n_features_mean": results[r]["metrics"]["n_features"].mean()}
        for r in RESPONSES
    ])
    summary.to_csv(os.path.join(RESULTS_DIR, "cv_rfe_summary.csv"), index=False)

    freq = pd.concat([
        feature_selection_frequency(results[r]["selected_features"], COVARIATE_NAMES, N_FOLDS_OUTER).assign(response=r)
        for r in RESPONSES
    ], ignore_index=True)
    freq.to_csv(os.path.join(RESULTS_DIR, "rfe_feature_frequency.csv"), index=False)

    print(summary)
