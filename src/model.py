from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import Any

import geopandas as gpd
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    ARTIFACT_DIR,
    BASE_X,
    CONTINUOUS_VARS,
    DATA_PATH,
    EPSG_MODEL,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    KNN_K,
    ML_N,
    MODEL_ARTIFACT_PATH,
    SEED,
    USER_CONTINUOUS,
)


def _safe_numeric(series, fill_value: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().any():
        fill_value = float(values.median())
    return values.fillna(fill_value)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["furniture_idx", "other_app_idx", "media_idx"]:
        if col not in out:
            out[col] = 0
        out[col] = _safe_numeric(out[col], 0)
    for col in BASE_X:
        if col not in out and not col.startswith("ln_dist") and not col.startswith("ln_"):
            out[col] = 0
    out["sum_equip_idx"] = out[["furniture_idx", "other_app_idx", "media_idx"]].sum(axis=1)
    out["inter_equip"] = _safe_numeric(out.get("pet_friendly", 0), 0) * out["sum_equip_idx"]
    out["inter_apt"] = _safe_numeric(out.get("pet_friendly", 0), 0) * _safe_numeric(out.get("apartment", 0), 0)
    out["inter_elev"] = _safe_numeric(out.get("pet_friendly", 0), 0) * _safe_numeric(out.get("elevator_building", 0), 0)
    out["inter_core"] = _safe_numeric(out.get("pet_friendly", 0), 0) * _safe_numeric(out.get("core_zone", 0), 0)
    return out


def _compute_knn_lags(gdf: gpd.GeoDataFrame, k: int = KNN_K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    tree = cKDTree(coords)
    query_k = min(k + 1, len(coords))
    _, idx = tree.query(coords, k=query_k)
    neighbors = idx.reshape(-1, 1) if query_k == 1 else idx[:, 1:]
    y = gdf["ln_rent"].to_numpy(dtype=float)
    pet = _safe_numeric(gdf["pet_friendly"], 0).to_numpy(dtype=float)
    return y[neighbors].mean(axis=1), pet[neighbors].mean(axis=1), coords


def _feature_stats(raw: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in USER_CONTINUOUS:
        series = _safe_numeric(raw[col], 0)
        q01 = float(series.quantile(0.01))
        q99 = float(series.quantile(0.99))
        if math.isclose(q01, q99):
            q01, q99 = float(series.min()), float(series.max())
        step = 0.5 if col in {"area_pings", "deposit_months"} else 1.0
        stats[col] = {
            "min": round(q01, 2),
            "max": round(q99, 2),
            "median": round(float(series.median()), 2),
            "step": step,
        }
    return stats


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MAPE_pct": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 2),
    }


def _prepare_training_frame() -> tuple[gpd.GeoDataFrame, StandardScaler, dict[str, dict[str, float]], np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到訓練資料：{DATA_PATH}")
    gdf = gpd.read_file(DATA_PATH).to_crs(EPSG_MODEL)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = add_engineered_features(gdf)

    required = ["ln_rent", *BASE_X]
    missing = [col for col in required if col not in gdf.columns]
    if missing:
        raise ValueError(f"訓練資料缺少必要欄位：{missing}")

    for col in BASE_X + ["ln_rent"]:
        if col in gdf.columns:
            gdf[col] = _safe_numeric(gdf[col], 0)
    gdf = gdf.dropna(subset=["ln_rent"]).copy()

    stats = _feature_stats(gdf)
    scaler = StandardScaler()
    gdf_s = gdf.copy()
    gdf_s[CONTINUOUS_VARS] = scaler.fit_transform(gdf[CONTINUOUS_VARS])
    wy, w_pet, coords = _compute_knn_lags(gdf_s)
    gdf_s["Wy"] = wy
    gdf_s["W_pet_friendly"] = w_pet
    return gdf_s, scaler, stats, coords


def train_and_save_model(artifact_path=MODEL_ARTIFACT_PATH, ml_n: int = ML_N) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    gdf_s, scaler, stats, coords = _prepare_training_frame()
    n = min(int(ml_n), len(gdf_s))
    rng = np.random.default_rng(SEED)
    idx_ml = rng.choice(len(gdf_s), size=n, replace=False)
    gdf_ml = gdf_s.iloc[idx_ml].reset_index(drop=True)
    y = gdf_ml["ln_rent"].to_numpy(dtype=float)

    idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=SEED)
    X_train = gdf_ml.iloc[idx_train][FEATURE_COLUMNS]
    X_test = gdf_ml.iloc[idx_test][FEATURE_COLUMNS]
    y_train = y[idx_train]
    y_test = y[idx_test]

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = _metrics(y_test, pred)

    feature_importance = pd.DataFrame(
        {"Feature": FEATURE_COLUMNS, "LightGBM_importance": model.feature_importances_}
    ).sort_values("LightGBM_importance", ascending=False)

    shap_importance = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Mean_abs_SHAP": np.nan})
    try:
        import shap

        sample_n = min(3000, len(X_test))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.iloc[:sample_n], check_additivity=False)
        shap_importance = pd.DataFrame(
            {"Feature": FEATURE_COLUMNS, "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0)}
        ).sort_values("Mean_abs_SHAP", ascending=False)
    except Exception as exc:
        shap_importance["note"] = f"SHAP 計算未完成：{exc}"

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "continuous_vars": CONTINUOUS_VARS,
        "metrics": metrics,
        "feature_stats": stats,
        "feature_importance": feature_importance.reset_index(drop=True),
        "shap_importance": shap_importance.reset_index(drop=True),
        "rental_coords": coords,
        "rental_y": gdf_s["ln_rent"].to_numpy(dtype=float),
        "rental_pet": gdf_s["pet_friendly"].to_numpy(dtype=float),
        "trained_rows": n,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(bundle, artifact_path)
    return _hydrate_bundle(bundle)


def _hydrate_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    bundle = dict(bundle)
    bundle["rental_tree"] = cKDTree(bundle["rental_coords"])
    return bundle


@lru_cache(maxsize=1)
def load_or_train_bundle() -> dict[str, Any]:
    if MODEL_ARTIFACT_PATH.exists():
        return _hydrate_bundle(joblib.load(MODEL_ARTIFACT_PATH))
    return train_and_save_model(MODEL_ARTIFACT_PATH)


def estimate_spatial_lags(bundle: dict[str, Any], x: float, y: float, k: int = KNN_K) -> dict[str, float]:
    query_k = min(k, len(bundle["rental_coords"]))
    _, idx = bundle["rental_tree"].query([[x, y]], k=query_k)
    idx_arr = np.atleast_1d(idx[0])
    return {
        "Wy": float(np.mean(bundle["rental_y"][idx_arr])),
        "W_pet_friendly": float(np.mean(bundle["rental_pet"][idx_arr])),
    }


def _build_raw_feature_row(
    bundle: dict[str, Any],
    user_inputs: dict[str, float],
    location_features: dict[str, float],
    x: float,
    y: float,
) -> dict[str, float]:
    row: dict[str, float] = {}
    row.update({key: float(value) for key, value in user_inputs.items()})
    row.update({key: float(value) for key, value in location_features.items()})
    row.update(estimate_spatial_lags(bundle, x, y))
    row["inter_equip"] = row["pet_friendly"] * row["sum_equip_idx"]
    row["inter_apt"] = row["pet_friendly"] * row["apartment"]
    row["inter_elev"] = row["pet_friendly"] * row["elevator_building"]
    row["inter_core"] = row["pet_friendly"] * row["core_zone"]
    missing = [col for col in FEATURE_COLUMNS if col not in row]
    if missing:
        raise ValueError(f"預測特徵缺漏：{missing}")
    return row


def predict_rent_per_ping(
    bundle: dict[str, Any],
    user_inputs: dict[str, float],
    location_features: dict[str, float],
    x: float,
    y: float,
) -> dict[str, Any]:
    raw_row = _build_raw_feature_row(bundle, user_inputs, location_features, x, y)
    model_row = pd.DataFrame([raw_row])
    model_row[CONTINUOUS_VARS] = bundle["scaler"].transform(model_row[CONTINUOUS_VARS])
    ln_pred = float(bundle["model"].predict(model_row[FEATURE_COLUMNS])[0])
    return {
        "ln_rent": ln_pred,
        "rent_per_ping": float(np.exp(ln_pred)),
        "raw_features": raw_row,
        "model_features": model_row[FEATURE_COLUMNS].iloc[0].to_dict(),
    }


def feature_description_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"Feature": col, "說明": FEATURE_DESCRIPTIONS.get(col, "")} for col in FEATURE_COLUMNS]
    )
