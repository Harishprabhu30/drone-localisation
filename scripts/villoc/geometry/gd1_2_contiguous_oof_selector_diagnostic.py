#!/usr/bin/env python3
"""GD1.2 contiguous out-of-fold multivariate geometry selector diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import yaml

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/top20_geometric_discrimination_gd1_2.yaml"


def resolve(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_preprocessor(
    frame: pd.DataFrame,
    features: list[str],
    clip_value: float,
) -> dict[str, Any]:
    specs = {}
    kept = []
    for feature in features:
        values = numeric_feature(frame[feature]).replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if finite.empty:
            continue
        median = float(finite.median())
        q1 = float(finite.quantile(0.25))
        q3 = float(finite.quantile(0.75))
        scale = float(q3 - q1)
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
        raise RuntimeError("No usable features in model family")
    return {"features": kept, "specs": specs}


def transform(frame: pd.DataFrame, prep: dict[str, Any]) -> np.ndarray:
    cols = []
    for feature in prep["features"]:
        spec = prep["specs"][feature]
        x = numeric_feature(frame[feature]).replace([np.inf, -np.inf], np.nan)
        x = x.fillna(spec["median"]).astype(float)
        z = (x.to_numpy() - spec["median"]) / spec["scale"]
        z = np.clip(z, -spec["clip"], spec["clip"])
        cols.append(z)
    return np.column_stack(cols).astype(float)


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
    pos = float(y.sum())
    neg = float(len(y) - pos)
    if pos <= 0 or neg <= 0:
        raise RuntimeError("Logistic training fold requires both classes")
    if balanced:
        sample_weight = np.where(
            y > 0.5,
            len(y) / (2.0 * pos),
            len(y) / (2.0 * neg),
        )
    else:
        sample_weight = np.ones(len(y), dtype=float)

    reg = np.eye(d + 1, dtype=float) * float(l2)
    reg[0, 0] = 0.0

    for _ in range(int(max_iterations)):
        p = sigmoid(design @ beta)
        grad = design.T @ (sample_weight * (p - y)) + reg @ beta
        curvature = sample_weight * p * (1.0 - p)
        H = design.T @ (curvature[:, None] * design) + reg
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]
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
    p = int(frame["label"].sum())
    n = int((~frame["label"]).sum())
    if p == 0 or n == 0:
        return None
    ranks = frame["score"].rank(method="average", ascending=True)
    s = float(ranks[frame["label"]].sum())
    return float((s - p * (p + 1) / 2.0) / (p * n))


def query_rank_hits(
    frame: pd.DataFrame,
    score_col: str,
    label: str,
    topks: list[int],
) -> dict[str, int]:
    hits = {f"top{k}": 0 for k in topks}
    for _, group in frame.groupby("query_id", sort=True):
        g = group.sort_values(
            [score_col, "pool_rank", "tile_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        for k in topks:
            hits[f"top{k}"] += int(g.head(int(k))[label].astype(bool).any())
    return hits


def contiguous_folds(query_ids: list[int], folds: int) -> list[set[int]]:
    arrays = np.array_split(np.array(sorted(query_ids), dtype=int), int(folds))
    return [set(int(v) for v in arr.tolist()) for arr in arrays]


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "GD1.2":
        raise RuntimeError("Expected GD1.2 config")
    if cfg["evaluation"].get("fold_scheme") != "contiguous_query_blocks":
        raise RuntimeError("Fold scheme changed")
    if cfg["evaluation"].get("no_hyperparameter_tuning") is not True:
        raise RuntimeError("Hyperparameter tuning must remain disabled")
    if float(cfg["evaluation"].get("logistic_l2")) != 1.0:
        raise RuntimeError("Fixed L2 contract changed")
    for key, value in cfg["scientific_boundary"].items():
        expected = True if key == "development_only" else False
        if value is not expected:
            raise RuntimeError(f"Scientific boundary mismatch: {key}")


def verify_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    gd11_report_path = resolve(cfg["gd1_1"]["report"])
    gd10_report_path = resolve(cfg["gd1_0"]["report"])
    labeled_path = resolve(cfg["gd1_1"]["labeled_evidence"])
    if not gd11_report_path.is_file() or not gd10_report_path.is_file() or not labeled_path.is_file():
        raise FileNotFoundError("Missing GD1.0/GD1.1 artifact")

    gd11 = json.loads(gd11_report_path.read_text(encoding="utf-8"))
    gd10 = json.loads(gd10_report_path.read_text(encoding="utf-8"))
    if gd11.get("status") != "PASS_GD1_1_POSTFREEZE_GEOMETRIC_DISCRIMINATION_AUDIT":
        raise RuntimeError("GD1.1 report not PASS")
    if gd10.get("status") != "PASS_GD1_0_REFERENCE_FREE_TOP20_GEOMETRIC_EVIDENCE_FREEZE":
        raise RuntimeError("GD1.0 report not PASS")
    if gd11.get("scientific_boundary", {}).get("blind_stress_loaded") is not False:
        raise RuntimeError("GD1.1 blind boundary violated")
    expected_diag = cfg["gd1_1"]["expected_feature_discrimination_sha256"]
    if gd11["outputs"]["feature_discrimination"]["sha256"] != expected_diag:
        raise RuntimeError("GD1.1 discrimination hash mismatch")
    expected_labeled = gd11["outputs"]["labeled_evidence"]["sha256"]
    if sha256_file(labeled_path) != expected_labeled:
        raise RuntimeError("GD1.1 labeled evidence bytes changed")

    frame = pd.read_csv(labeled_path).copy()
    if len(frame) != int(cfg["gd1_1"]["expected_pairs"]):
        raise RuntimeError("GD1.1 pair count mismatch")
    if frame["query_id"].nunique() != int(cfg["gd1_1"]["expected_queries"]):
        raise RuntimeError("GD1.1 query count mismatch")
    return frame, gd10


def family_features(gd10: dict[str, Any], frame: pd.DataFrame) -> dict[str, list[str]]:
    groups = gd10["evidence_groups"]
    out = {name: list(features) for name, features in groups.items()}
    all_geom = []
    for name in (
        "correspondence_strength",
        "spatial_support",
        "residual_geometry",
        "local_feature_consistency",
        "homography_shape",
        "projective_stability",
    ):
        for f in groups[name]:
            if f not in all_geom:
                all_geom.append(f)
    out["all_geometry"] = all_geom
    out["all_geometry_plus_retrieval_rank"] = all_geom + ["pool_rank"]
    out["inliers_only"] = ["inliers"]
    for name, features in out.items():
        missing = [f for f in features if f not in frame.columns]
        if missing:
            raise RuntimeError(f"{name} missing columns: {missing}")
    return out


def run_oof(
    frame: pd.DataFrame,
    features: list[str],
    label: str,
    cfg: dict[str, Any],
) -> tuple[pd.Series, list[dict[str, Any]]]:
    qids = sorted(frame["query_id"].astype(int).unique().tolist())
    fold_sets = contiguous_folds(qids, int(cfg["evaluation"]["folds"]))
    scores = pd.Series(index=frame.index, dtype=float)
    folds_out = []

    for fold_idx, test_ids in enumerate(fold_sets):
        test_mask = frame["query_id"].astype(int).isin(test_ids)
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]
        prep = fit_preprocessor(
            train,
            features,
            float(cfg["evaluation"]["preprocessing"]["standardized_clip"]),
        )
        Xtrain = transform(train, prep)
        Xtest = transform(test, prep)
        ytrain = train[label].astype(bool).astype(float).to_numpy()

        beta = fit_logistic(
            Xtrain, ytrain,
            l2=float(cfg["evaluation"]["logistic_l2"]),
            max_iterations=int(cfg["evaluation"]["logistic_max_iterations"]),
            tolerance=float(cfg["evaluation"]["logistic_tolerance"]),
            balanced=bool(cfg["evaluation"]["balanced_class_weights"]),
        )
        pred = predict_logistic(Xtest, beta)
        scores.loc[test.index] = pred
        folds_out.append({
            "fold": int(fold_idx + 1),
            "test_query_min": int(min(test_ids)),
            "test_query_max": int(max(test_ids)),
            "test_queries": int(len(test_ids)),
            "train_pairs": int(len(train)),
            "test_pairs": int(len(test)),
            "features_used": list(prep["features"]),
            "coefficients": {
                "intercept": float(beta[0]),
                **{
                    f: float(v)
                    for f, v in zip(prep["features"], beta[1:])
                },
            },
        })
    if scores.isna().any():
        raise RuntimeError("OOF scoring left missing rows")
    return scores.astype(float), folds_out


def retrieval_score(frame: pd.DataFrame) -> pd.Series:
    return -pd.to_numeric(frame["pool_rank"], errors="raise").astype(float)


def deterministic_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode("utf-8")


def deterministic_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_immutable(files: dict[Path, bytes]) -> None:
    bad = [str(p) for p, data in files.items() if p.exists() and p.read_bytes() != data]
    if bad:
        raise RuntimeError("Refusing to overwrite differing GD1.2 outputs: " + ", ".join(bad))
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GD1.2 contiguous out-of-fold multivariate geometry selector diagnostic"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    frame, gd10 = verify_inputs(cfg)
    families = family_features(gd10, frame)

    labels = list(cfg["labels"])
    topks = [int(k) for k in cfg["evaluation"]["topk"]]
    model_families = list(cfg["model_families"])

    score_rows = []
    report_models: dict[str, Any] = {}

    for label in labels:
        report_models[label] = {}
        for family in model_families:
            if family == "retrieval_rank_only":
                scores = retrieval_score(frame)
                folds_meta = []
            else:
                scores, folds_meta = run_oof(frame, families[family], label, cfg)

            tmp = frame[["query_id", "pool_rank", "tile_id", label]].copy()
            tmp["score"] = scores
            auc = binary_auc(tmp[label], tmp["score"])
            hits = query_rank_hits(tmp, "score", label, topks)
            report_models[label][family] = {
                "pair_auc_oof": auc,
                "query_hits": hits,
                "folds": folds_meta,
            }
            for idx, row in tmp.iterrows():
                score_rows.append({
                    "label": label,
                    "model_family": family,
                    "query_id": int(row["query_id"]),
                    "pool_rank": int(row["pool_rank"]),
                    "tile_id": str(row["tile_id"]),
                    "target": bool(row[label]),
                    "oof_score": float(row["score"]),
                })

    report = {
        "stage": "GD1.2",
        "status": "PASS_GD1_2_CONTIGUOUS_OOF_MULTIVARIATE_SELECTOR_DIAGNOSTIC",
        "scope": "development_only_diagnostic",
        "protocol": cfg["evaluation"],
        "model_families": model_families,
        "results": report_models,
        "scientific_boundary": {
            "blind_stress_loaded": False,
            "hyperparameter_tuning_executed": False,
            "frontend_changed": False,
            "candidate_pool_changed": False,
            "deployable_model_exported": False,
            "bootstrap_state_modified": False,
        },
        "interpretation": {
            "oof_reduces_same_query_training_leakage": True,
            "contiguous_folds_reduce_temporal_leakage_but_do_not_replace_external_validation": True,
            "no_selector_promoted": True,
            "no_final_model_fit": True,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "gd1_0_report_sha256": sha256_file(resolve(cfg["gd1_0"]["report"])),
            "gd1_1_report_sha256": sha256_file(resolve(cfg["gd1_1"]["report"])),
            "labeled_evidence_sha256": sha256_file(resolve(cfg["gd1_1"]["labeled_evidence"])),
        },
        "outputs": {},
    }

    scores_frame = pd.DataFrame(score_rows).sort_values(
        ["label", "model_family", "query_id", "pool_rank"],
        kind="mergesort",
    )
    out = resolve(cfg["output_root"])
    score_path = out / "gd1_2_oof_candidate_scores.csv"
    score_bytes = deterministic_csv(scores_frame)
    report["outputs"]["oof_scores"] = {
        "path": str(score_path.relative_to(ROOT)),
        "sha256": hashlib.sha256(score_bytes).hexdigest(),
        "rows": int(len(scores_frame)),
    }
    report_path = out / "gd1_2_oof_selector_report.json"
    report_bytes = deterministic_json(report)
    files = {score_path: score_bytes, report_path: report_bytes}
    write_immutable(files)

    print("Status:", report["status"])
    for label in labels:
        print(f"{label}:")
        for family in model_families:
            result = report_models[label][family]
            hits = result["query_hits"]
            auc = result["pair_auc_oof"]
            print(
                f"  {family}: auc={auc if auc is not None else 'NA'} "
                f"top1={hits['top1']}/403 top3={hits['top3']}/403 "
                f"top5={hits['top5']}/403 top10={hits['top10']}/403"
            )
    print("Blind stress loaded: False")
    print("Hyperparameter tuning executed: False")
    print("Deployable model exported: False")
    print("Report:", report_path)
    print("OOF score SHA256:", report["outputs"]["oof_scores"]["sha256"])


if __name__ == "__main__":
    main()
