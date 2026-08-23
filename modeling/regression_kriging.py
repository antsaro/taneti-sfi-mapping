"""Regression kriging: krige the out-of-fold residuals of the RF trend
model and add the kriged surface back on top, at each outer-fold test set.
Compared against the plain RF trend to see whether the residuals still
carry usable spatial structure once the covariates have done their part.

Needs pykrige: pip install pykrige
"""

import os

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from config import (COVARIATE_NAMES, FINAL_GPKG, N_FOLDS_INNER, N_FOLDS_OUTER, N_JOBS_CPU,
                     OPTUNA_TRIALS_CV, RESPONSES, RESULTS_DIR, SEED)
from modeling.common import calc_metrics, impute_train_test, tune_rf

SOURCE_EPSG_WGS84 = 4326
TARGET_EPSG_UTM = 32738  # WGS84 / UTM 38S, Madagascar -- use 32739 east of 48E

VARIOGRAM_MODELS = ["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"]
VARIOGRAM_OPTUNA_TRIALS = 200

RK_RESULTS_DIR = os.path.join(RESULTS_DIR, "RF_Kriging")
os.makedirs(RK_RESULTS_DIR, exist_ok=True)


def utm_coords(gdf):
    transformer = Transformer.from_crs(f"EPSG:{SOURCE_EPSG_WGS84}", f"EPSG:{TARGET_EPSG_UTM}", always_xy=True)
    x, y = transformer.transform(gdf.geometry.x.values, gdf.geometry.y.values)
    return np.column_stack([x, y])


def oof_predictions(X, y, rf_params, n_inner):
    """In-fold OOF predictions on a train split, used to build residuals for
    kriging without letting the RF see its own targets."""
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=42)
    pred = np.full(len(y), np.nan)
    for tr, va in kf.split(X):
        scaler = StandardScaler()
        Xtr, Xva = scaler.fit_transform(X[tr]), scaler.transform(X[va])
        model = RandomForestRegressor(**rf_params)
        model.fit(Xtr, y[tr])
        pred[va] = model.predict(Xva)
    return pred


def krige(coords_tr, resid_tr, coords_te, params):
    krig = OrdinaryKriging(coords_tr[:, 0], coords_tr[:, 1], resid_tr,
                            variogram_model=params["variogram_model"], nlags=params["nlags"],
                            weight=params["weight"], enable_plotting=False, verbose=False)
    z, ss = krig.execute("points", coords_te[:, 0], coords_te[:, 1])
    return np.asarray(z), np.asarray(ss)


def tune_variogram(coords, residuals, n_inner, n_trials):
    def objective(trial):
        params = dict(
            variogram_model=trial.suggest_categorical("variogram_model", VARIOGRAM_MODELS),
            nlags=trial.suggest_int("nlags", 6, 20),
            weight=trial.suggest_categorical("weight", [True, False]),
        )
        kf = KFold(n_splits=n_inner, shuffle=True, random_state=42)
        rmses = []
        for tr, va in kf.split(coords):
            try:
                z, _ = krige(coords[tr], residuals[tr], coords[va], params)
                if np.any(np.isnan(z)):
                    return float("inf")
                rmses.append(np.sqrt(mean_squared_error(residuals[va], z)))
            except Exception:
                return float("inf")
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, "kriging_method": "ordinary"}


def run_nested_rk(response, covariates, y_full, X_full, coords_full,
                   outer_k=N_FOLDS_OUTER, inner_k=N_FOLDS_INNER,
                   rf_trials=OPTUNA_TRIALS_CV, variogram_trials=VARIOGRAM_OPTUNA_TRIALS, seed=SEED):
    kf = KFold(n_splits=outer_k, shuffle=True, random_state=seed)
    rf_metrics, rk_metrics, oof_rows = [], [], []

    for i, (tr, te) in enumerate(kf.split(X_full), 1):
        y_tr, y_te = y_full[tr], y_full[te]
        coords_tr, coords_te = coords_full[tr], coords_full[te]
        X_tr, X_te = impute_train_test(X_full[tr], X_full[te], covariates)

        rf_params = tune_rf(X_tr, y_tr, rf_trials, inner_k)
        scaler = StandardScaler()
        X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_tr_s, y_tr)
        rf_pred = rf.predict(X_te_s)
        m_rf = calc_metrics(y_te, rf_pred)
        m_rf["fold"] = i
        rf_metrics.append(m_rf)

        resid_tr = y_tr - oof_predictions(X_tr, y_tr, rf_params, inner_k)
        vparams = tune_variogram(coords_tr, resid_tr, inner_k, variogram_trials)
        try:
            z_resid, _ = krige(coords_tr, resid_tr, coords_te, vparams)
        except Exception:
            z_resid = np.zeros(len(te))

        rk_pred = rf_pred + z_resid
        m_rk = calc_metrics(y_te, rk_pred)
        m_rk["fold"] = i
        rk_metrics.append(m_rk)

        oof_rows.append(pd.DataFrame({"fold": i, "obs": y_te, "rf_pred": rf_pred, "rk_pred": rk_pred}))
        print(f"  {response} fold {i}/{outer_k}  RF R2={m_rf['R2']:.3f}  RF+RK R2={m_rk['R2']:.3f}")

    return dict(rf_metrics=pd.DataFrame(rf_metrics), rk_metrics=pd.DataFrame(rk_metrics),
                oof=pd.concat(oof_rows, ignore_index=True))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    gdf = gpd.read_file(FINAL_GPKG).reset_index(drop=True)
    coords_all = utm_coords(gdf)
    df = gdf.drop(columns="geometry")

    summary_rows = []
    for response in RESPONSES:
        mask = df[response].notna()
        sub = df.loc[mask].reset_index(drop=True)
        coords_sub = coords_all[mask.values]
        X_full = sub[COVARIATE_NAMES].values
        y_full = sub[response].values.astype(np.float64)

        out_dir = os.path.join(RK_RESULTS_DIR, f"results_{response}")
        os.makedirs(out_dir, exist_ok=True)
        result = run_nested_rk(response, COVARIATE_NAMES, y_full, X_full, coords_sub)
        result["oof"].to_csv(os.path.join(out_dir, f"oof_predictions_{response}.csv"), index=False)

        row = {"response": response, "n_obs": len(y_full)}
        for k in ["R2", "RMSE", "MAE", "RPIQ", "CCC"]:
            row[f"RF_{k}_mean"] = result["rf_metrics"][k].mean()
            row[f"RF_RK_{k}_mean"] = result["rk_metrics"][k].mean()
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(os.path.join(RK_RESULTS_DIR, "summary_RF_vs_RK.csv"), index=False)
