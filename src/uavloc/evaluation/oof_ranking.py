"""Reusable deterministic out-of-fold ranking helpers.

Designed for development-only diagnostics with group/query-level folds. This
module does not know anything about ground truth semantics or particular stages.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def numeric_feature(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip().str.lower()
    boolish = text.map({
        "true": 1.0, "false": 0.0,
        "yes": 1.0, "no": 0.0,
        "y": 1.0, "n": 0.0,
    })
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.notna(), boolish).astype(float)


def fit_preprocessor(
    frame: pd.DataFrame,
    features: list[str],
    *,
    clip_value: float,
) -> dict[str, Any]:
    specs: dict[str, dict[str, float]] = {}
    kept: list[str] = []
    for feature in features:
        values = numeric_feature(frame[feature]).replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if finite.empty:
            continue
        median = float(finite.median())
        scale = float(finite.quantile(0.75) - finite.quantile(0.25))
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = float(finite.std(ddof=0))
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        specs[feature] = {
            "median": median,
            "scale": scale,
            "clip": float(clip_value),
        }
        kept.append(feature)
    if not kept:
        raise RuntimeError("No usable features")
    return {"features": kept, "specs": specs}


def transform(frame: pd.DataFrame, prep: dict[str, Any]) -> np.ndarray:
    columns = []
    for feature in prep["features"]:
        spec = prep["specs"][feature]
        values = numeric_feature(frame[feature]).replace([np.inf, -np.inf], np.nan)
        values = values.fillna(spec["median"]).astype(float)
        z = (values.to_numpy() - spec["median"]) / spec["scale"]
        z = np.clip(z, -spec["clip"], spec["clip"])
        columns.append(z)
    return np.column_stack(columns).astype(float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    max_iterations: int,
    tolerance: float,
    balanced: bool,
) -> np.ndarray:
    n, d = X.shape
    design = np.column_stack([np.ones(n), X])
    beta = np.zeros(d + 1, dtype=float)

    y = np.asarray(y, dtype=float)
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if positives <= 0 or negatives <= 0:
        raise RuntimeError("Training fold requires both classes")

    if balanced:
        sample_weight = np.where(
            y > 0.5,
            len(y) / (2.0 * positives),
            len(y) / (2.0 * negatives),
        )
    else:
        sample_weight = np.ones(len(y), dtype=float)

    regularizer = np.eye(d + 1, dtype=float) * float(l2)
    regularizer[0, 0] = 0.0

    for _ in range(int(max_iterations)):
        probability = sigmoid(design @ beta)
        gradient = (
            design.T @ (sample_weight * (probability - y))
            + regularizer @ beta
        )
        curvature = sample_weight * probability * (1.0 - probability)
        hessian = (
            design.T @ (curvature[:, None] * design)
            + regularizer
        )
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta -= delta
        if float(np.max(np.abs(delta))) <= float(tolerance):
            break
    return beta


def predict_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X)), X])
    return sigmoid(design @ beta)


def binary_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    frame = pd.DataFrame({
        "label": labels.astype(bool),
        "score": pd.to_numeric(scores, errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    positives = int(frame["label"].sum())
    negatives = int((~frame["label"]).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = frame["score"].rank(method="average", ascending=True)
    rank_sum = float(ranks[frame["label"]].sum())
    return float(
        (
            rank_sum - positives * (positives + 1) / 2.0
        ) / (positives * negatives)
    )


def query_rank_hits(
    frame: pd.DataFrame,
    *,
    score_col: str,
    label_col: str,
    topks: list[int],
) -> dict[str, int]:
    hits = {f"top{k}": 0 for k in topks}
    for _, group in frame.groupby("query_id", sort=True):
        ordered = group.sort_values(
            [score_col, "pool_rank", "tile_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        for k in topks:
            hits[f"top{k}"] += int(
                ordered.head(int(k))[label_col].astype(bool).any()
            )
    return hits


def contiguous_folds(query_ids: list[int], folds: int) -> list[set[int]]:
    arrays = np.array_split(np.array(sorted(query_ids), dtype=int), int(folds))
    return [set(int(value) for value in array.tolist()) for array in arrays]


def run_contiguous_oof(
    frame: pd.DataFrame,
    *,
    features: list[str],
    label_col: str,
    folds: int,
    clip_value: float,
    l2: float,
    max_iterations: int,
    tolerance: float,
    balanced: bool,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    query_ids = sorted(frame["query_id"].astype(int).unique().tolist())
    fold_sets = contiguous_folds(query_ids, int(folds))
    scores = pd.Series(index=frame.index, dtype=float)
    metadata: list[dict[str, Any]] = []

    for fold_index, test_ids in enumerate(fold_sets, start=1):
        test_mask = frame["query_id"].astype(int).isin(test_ids)
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]

        prep = fit_preprocessor(
            train,
            features,
            clip_value=float(clip_value),
        )
        X_train = transform(train, prep)
        X_test = transform(test, prep)
        y_train = train[label_col].astype(bool).astype(float).to_numpy()

        beta = fit_logistic(
            X_train,
            y_train,
            l2=float(l2),
            max_iterations=int(max_iterations),
            tolerance=float(tolerance),
            balanced=bool(balanced),
        )
        scores.loc[test.index] = predict_logistic(X_test, beta)

        metadata.append({
            "fold": int(fold_index),
            "test_query_min": int(min(test_ids)),
            "test_query_max": int(max(test_ids)),
            "test_queries": int(len(test_ids)),
            "train_pairs": int(len(train)),
            "test_pairs": int(len(test)),
            "features_used": list(prep["features"]),
        })

    if scores.isna().any():
        raise RuntimeError("OOF scoring left missing rows")
    return scores.astype(float), metadata
