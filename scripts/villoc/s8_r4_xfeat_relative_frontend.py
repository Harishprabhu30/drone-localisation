'''
run command for traj01 villoc dataset:

a. smoke test, running 100 frames:

ROOT=outputs/villoc/traj01_90deg_stable120m

python scripts/villoc/s8_r4_xfeat_relative_frontend.py \
  --manifest "$ROOT/metadata/s6a_relative_motion/s6a_sequence_manifest.csv" \
  --output-root "$ROOT" \
  --xfeat-repo third_party/accelerated_features \
  --orb-pairs "$ROOT/metadata/s6a_relative_motion/s6a1_orb_affine_pair_diagnostics.csv" \
  --orb-aligned "$ROOT/metadata/s6a_relative_motion/s6a2_orb_relative_trajectory_aligned_eval_only.csv" \
  --orb-summary "$ROOT/metadata/s6a_relative_motion/s6a2_orb_relative_trajectory_summary.csv" \
  --sequence traj01 \
  --device cpu \
  --resize-long 960 \
  --top-k 1200 \
  --detection-threshold 0.05 \
  --min-cossim 0.82 \
  --ransac-threshold 3.0 \
  --alignment-prefix-frames 50 \
  --thresholds-m 10,20,40,80,120 \
  --sustain-frames 5 \
  --max-frames 100 \
  2>&1 | tee "$ROOT/logs/s6a_relative_motion/s8r4_xfeat_smoke100.log"

  b. full run:

export PYTHONPATH=$PWD/src

ROOT=outputs/villoc/traj01_90deg_stable120m

python scripts/villoc/s8_r4_xfeat_relative_frontend.py \
  --manifest "$ROOT/metadata/s6a_relative_motion/s6a_sequence_manifest.csv" \
  --output-root "$ROOT" \
  --xfeat-repo third_party/accelerated_features \
  --orb-pairs "$ROOT/metadata/s6a_relative_motion/s6a1_orb_affine_pair_diagnostics.csv" \
  --orb-aligned "$ROOT/metadata/s6a_relative_motion/s6a2_orb_relative_trajectory_aligned_eval_only.csv" \
  --orb-summary "$ROOT/metadata/s6a_relative_motion/s6a2_orb_relative_trajectory_summary.csv" \
  --sequence traj01 \
  --device cpu \
  --resize-long 960 \
  --top-k 1200 \
  --detection-threshold 0.05 \
  --min-cossim 0.82 \
  --ransac-threshold 3.0 \
  --alignment-prefix-frames 50 \
  --thresholds-m 10,20,40,80,120 \
  --sustain-frames 5 \
  2>&1 | tee "$ROOT/logs/s6a_relative_motion/s8r4_xfeat_full403.log"

'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def import_s7_xfeat_module(repo_root: Path):
    path = repo_root / "scripts/satloc/s7/s7a_2_xfeat_full_trajectory_comparison.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("s7a2_xfeat_lib", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_image(repo_root: Path, value: Any) -> Path:
    raw = Path(str(value)).expanduser()
    for p in (raw, repo_root / raw):
        if p.exists():
            return p.resolve()
    return (repo_root / raw).resolve()


def load_manifest(repo_root: Path, manifest_path: Path, sequence: str, max_frames: int | None) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    if "sequence" in df.columns:
        df = df[df["sequence"].astype(str) == sequence].copy()

    # Core relative tracking requires only frame identity
    # and images. Reference ENU coordinates are optional
    # evaluation inputs and must never be required in blind mode.
    required = {"sequence_frame_id", "token0_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Manifest missing columns: {missing}")

    has_x_enu = "x_enu_m" in df.columns
    has_y_enu = "y_enu_m" in df.columns

    if has_x_enu != has_y_enu:
        raise RuntimeError(
            "Evaluation reference must provide both "
            "x_enu_m and y_enu_m, or neither."
        )

    image_col = None
    for c in ["image_path_resolved", "image_path", "resolved_image_path"]:
        if c in df.columns:
            image_col = c
            break
    if image_col is None:
        raise RuntimeError("Manifest has no usable image path column.")

    df["sequence_frame_id"] = pd.to_numeric(
        df["sequence_frame_id"],
        errors="raise",
    ).astype(int)

    df["token0_id"] = pd.to_numeric(
        df["token0_id"],
        errors="raise",
    ).astype(int)

    if has_x_enu:
        df["x_enu_m"] = pd.to_numeric(
            df["x_enu_m"],
            errors="raise",
        )
        df["y_enu_m"] = pd.to_numeric(
            df["y_enu_m"],
            errors="raise",
        )
    df = df.sort_values("sequence_frame_id", kind="mergesort").reset_index(drop=True)

    if max_frames is not None and max_frames > 0:
        df = df.head(max_frames).copy().reset_index(drop=True)

    expected = np.arange(len(df), dtype=int)
    if not np.array_equal(df["sequence_frame_id"].to_numpy(dtype=int), expected):
        raise RuntimeError("sequence_frame_id is not contiguous from 0 after filtering.")

    df["image_path_full_resolved"] = df[image_col].map(lambda v: str(resolve_image(repo_root, v)))
    missing_paths = [p for p in df["image_path_full_resolved"].map(Path) if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} images missing. First: {missing_paths[0]}")

    return df


def build_pair_manifest(seq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(seq) - 1):
        a = seq.iloc[i]
        b = seq.iloc[i + 1]
        rows.append({
            "comparison_pair_id": i,
            "pair_number": i,
            "frame_index_a": int(a["sequence_frame_id"]),
            "frame_index_b": int(b["sequence_frame_id"]),
            "token0_a": int(a["token0_id"]),
            "token0_b": int(b["token0_id"]),
            "in_stratified_diagnostic_subset": False,
            "primary_scene": "",
            "secondary_scene": "",
            "selection_role": "",
            "range_id": "",
        })
    return pd.DataFrame(rows)


def trajectory_path_length(points: np.ndarray) -> float:
    d = np.diff(np.asarray(points, dtype=float), axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))


def error_metrics(aligned: np.ndarray, reference: np.ndarray, cumulative_distance: np.ndarray, start_index: int) -> dict[str, float]:
    errors = np.linalg.norm(aligned - reference, axis=1)
    tail = errors[start_index:]
    distance = float(cumulative_distance[-1] - cumulative_distance[start_index])
    final_error = float(errors[-1])
    return {
        "rmse_m": float(math.sqrt(np.mean(tail * tail))),
        "mean_error_m": float(np.mean(tail)),
        "median_error_m": float(np.median(tail)),
        "p95_error_m": float(np.percentile(tail, 95)),
        "max_error_m": float(np.max(tail)),
        "final_error_m": final_error,
        "evaluation_distance_m": distance,
        "final_drift_per_100m": float(100.0 * final_error / distance) if distance > 1e-9 else float("nan"),
    }


def first_sustained_crossing(errors: np.ndarray, cumulative_distance: np.ndarray, threshold: float, start_index: int, sustain: int) -> dict[str, Any]:
    above = np.isfinite(errors) & (errors >= threshold)
    for idx in range(start_index, len(errors) - sustain + 1):
        if bool(np.all(above[idx:idx + sustain])):
            return {
                "threshold_m": float(threshold),
                "crossed": True,
                "frame_index": int(idx),
                "frames_after_prefix": int(idx - start_index),
                "distance_after_prefix_m": float(cumulative_distance[idx] - cumulative_distance[start_index]),
                "error_at_crossing_m": float(errors[idx]),
            }
    return {
        "threshold_m": float(threshold),
        "crossed": False,
        "frame_index": None,
        "frames_after_prefix": None,
        "distance_after_prefix_m": None,
        "error_at_crossing_m": None,
    }


def align_and_score(mod, seq: pd.DataFrame, pair_df: pd.DataFrame, feature_df: pd.DataFrame, prefix_frames: int, thresholds: list[float], sustain_frames: int):
    width = int(feature_df.iloc[0]["width"])
    height = int(feature_df.iloc[0]["height"])

    traj = mod.integrate_se2_scale_normalized(pair_df, width, height)
    merged = seq[["sequence_frame_id", "token0_id", "x_enu_m", "y_enu_m"]].merge(
        traj, on="sequence_frame_id", how="inner", validate="one_to_one"
    )

    reference = merged[["x_enu_m", "y_enu_m"]].to_numpy(dtype=float)
    reference = reference - reference[0]
    visual = merged[["visual_x_px", "visual_y_px"]].to_numpy(dtype=float)

    steps = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    cumulative_distance = np.concatenate([[0.0], np.cumsum(steps)])

    global_tf = mod.fit_similarity(visual, reference)
    global_aligned = mod.apply_similarity(visual, global_tf)

    prefix_count = min(max(prefix_frames, 3), len(merged))
    prefix_tf = mod.fit_similarity(visual[:prefix_count], reference[:prefix_count])
    prefix_aligned = mod.apply_similarity(visual, prefix_tf)

    global_err = np.linalg.norm(global_aligned - reference, axis=1)
    prefix_err = np.linalg.norm(prefix_aligned - reference, axis=1)

    aligned = merged.copy()
    aligned["method"] = "xfeat"
    aligned["reference_x_m"] = reference[:, 0]
    aligned["reference_y_m"] = reference[:, 1]
    aligned["reference_cumulative_distance_m"] = cumulative_distance
    aligned["global_aligned_x_m"] = global_aligned[:, 0]
    aligned["global_aligned_y_m"] = global_aligned[:, 1]
    aligned["global_alignment_error_m"] = global_err
    aligned["prefix_aligned_x_m"] = prefix_aligned[:, 0]
    aligned["prefix_aligned_y_m"] = prefix_aligned[:, 1]
    aligned["prefix_locked_error_m"] = prefix_err

    summary = {
        "reference_path_m": trajectory_path_length(reference),
        "global_alignment": {
            "scale_m_per_px": float(global_tf["scale"]),
            "rotation_deg": float(global_tf["rotation_deg"]),
            **error_metrics(global_aligned, reference, cumulative_distance, 0),
        },
        "prefix_locked_alignment": {
            "prefix_frames": int(prefix_count),
            "prefix_reference_distance_m": float(cumulative_distance[prefix_count - 1]),
            "scale_m_per_px": float(prefix_tf["scale"]),
            "rotation_deg": float(prefix_tf["rotation_deg"]),
            **error_metrics(prefix_aligned, reference, cumulative_distance, prefix_count - 1),
        },
        "threshold_crossings": [
            first_sustained_crossing(prefix_err, cumulative_distance, t, prefix_count - 1, sustain_frames)
            for t in thresholds
        ],
    }
    return aligned, summary


def finite_median(s):
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.median()) if len(v) else float("nan")


def finite_p05(s):
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(np.percentile(v, 5)) if len(v) else float("nan")


def save_plots(aligned: pd.DataFrame, pair_df: pd.DataFrame, orb_aligned_path: Path, orb_pairs_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    orb_aligned = pd.read_csv(orb_aligned_path)
    if "variant" in orb_aligned.columns:
        orb_aligned = orb_aligned[orb_aligned["variant"].astype(str) == "se2_scale_normalized"].copy()
    orb_aligned = orb_aligned.sort_values("sequence_frame_id")

    orb_pairs = pd.read_csv(orb_pairs_path)
    if "stride" in orb_pairs.columns:
        orb_pairs = orb_pairs[pd.to_numeric(orb_pairs["stride"], errors="coerce") == 1].copy()
    orb_pairs = orb_pairs[pd.to_numeric(orb_pairs["frame_index_b"], errors="coerce") < len(aligned)].copy()

    plt.figure(figsize=(9, 8))
    plt.plot(aligned["reference_x_m"], aligned["reference_y_m"], label="Reference — eval only")
    plt.plot(orb_aligned["prefix_aligned_x_m"], orb_aligned["prefix_aligned_y_m"], label="ORB")
    plt.plot(aligned["prefix_aligned_x_m"], aligned["prefix_aligned_y_m"], label="XFeat")
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Villoc ORB vs XFeat prefix-locked trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "s8r4_orb_xfeat_prefix_locked_xy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(orb_aligned["reference_cumulative_distance_m"], orb_aligned["prefix_locked_error_m"], label="ORB")
    plt.plot(aligned["reference_cumulative_distance_m"], aligned["prefix_locked_error_m"], label="XFeat")
    for t in [10, 20, 40, 80, 120]:
        plt.axhline(t, linestyle="--", linewidth=1)
    plt.xlabel("Reference cumulative distance [m] — eval only")
    plt.ylabel("Prefix-locked error [m]")
    plt.title("Villoc relative frontend error growth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "s8r4_orb_xfeat_error_vs_distance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(orb_pairs["frame_index_a"], orb_pairs["inlier_ratio"], label="ORB")
    plt.plot(pair_df["frame_index_a"], pair_df["inlier_ratio"], label="XFeat")
    plt.xlabel("Pair start frame")
    plt.ylabel("RANSAC inlier ratio")
    plt.ylim(0, 1.02)
    plt.title("Villoc ORB vs XFeat inlier ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "s8r4_orb_xfeat_inlier_ratio.png", dpi=180)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--xfeat-repo", type=Path, default=Path("third_party/accelerated_features"))
    p.add_argument("--orb-pairs", type=Path, default=None)
    p.add_argument("--orb-aligned", type=Path, default=None)
    p.add_argument("--orb-summary", type=Path, default=None)
    p.add_argument("--sequence", default="traj01")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--resize-long", type=int, default=960)
    p.add_argument("--top-k", type=int, default=1200)
    p.add_argument("--detection-threshold", type=float, default=0.05)
    p.add_argument("--min-cossim", type=float, default=0.82)
    p.add_argument("--ransac-threshold", type=float, default=3.0)
    p.add_argument("--alignment-prefix-frames", type=int, default=50)
    p.add_argument("--thresholds-m", default="10,20,40,80,120")
    p.add_argument("--sustain-frames", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--allow-unpinned-xfeat", action="store_true")
    p.add_argument(
        "--blind-only",
        action="store_true",
        help=(
            "Run image-only XFeat relative tracking and save "
            "the raw visual trajectory without reading or "
            "requiring reference ENU coordinates."
        ),
    )
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve() if args.output_root.is_absolute() else (repo_root / args.output_root).resolve()
    xfeat_repo = args.xfeat_repo.resolve() if args.xfeat_repo.is_absolute() else (repo_root / args.xfeat_repo).resolve()
    manifest_path = args.manifest.resolve() if args.manifest.is_absolute() else (repo_root / args.manifest).resolve()

    mod = import_s7_xfeat_module(repo_root)

    seq = load_manifest(
        repo_root,
        manifest_path,
        args.sequence,
        args.max_frames if args.max_frames > 0 else None,
    )
    pair_manifest = build_pair_manifest(seq)

    evaluation_reference_available = {
        "x_enu_m",
        "y_enu_m",
    }.issubset(seq.columns)

    if not args.blind_only:
        if not evaluation_reference_available:
            raise RuntimeError(
                "Reference ENU is unavailable. "
                "Use --blind-only for reference-free tracking."
            )

        missing_eval_inputs = [
            name
            for name, value in [
                ("--orb-pairs", args.orb_pairs),
                ("--orb-aligned", args.orb_aligned),
                ("--orb-summary", args.orb_summary),
            ]
            if value is None
        ]

        if missing_eval_inputs:
            raise RuntimeError(
                "Evaluation mode requires: "
                + ", ".join(missing_eval_inputs)
            )

    thresholds = [
        float(x.strip())
        for x in args.thresholds_m.split(",")
        if x.strip()
    ]

    torch, model, device, xfeat_commit = mod.load_official_xfeat(
        xfeat_repo,
        args.device,
        args.top_k,
        args.detection_threshold,
    )

    if xfeat_commit and xfeat_commit != mod.PINNED_XFEAT_COMMIT and not args.allow_unpinned_xfeat:
        raise RuntimeError(
            f"XFeat checkout is not pinned.\nExpected: {mod.PINNED_XFEAT_COMMIT}\nFound: {xfeat_commit}"
        )

    metadata_dir = output_root / "metadata/s8_xfeat_relative_frontend"
    reports_dir = output_root / "reports/s8_xfeat_relative_frontend"
    figures_dir = output_root / "figures/s8_xfeat_relative_frontend"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("S8.R4 XFeat Villoc relative frontend")
    print("-" * 72)
    print("frames:", len(seq))
    print("pairs:", len(pair_manifest))
    print("device:", device)
    print("xfeat repo:", xfeat_repo)
    print("xfeat commit:", xfeat_commit)

    wall_start = time.perf_counter()

    feature_rows = []
    pair_rows = []

    first = seq.iloc[0]
    prev_feat, prev_meta = mod.extract_one(
        torch, model, device, Path(first["image_path_full_resolved"]),
        args.resize_long, args.top_k, args.detection_threshold
    )
    feature_rows.append({
        "sequence_frame_id": int(first["sequence_frame_id"]),
        "token0_id": int(first["token0_id"]),
        **prev_meta,
    })

    for frame_index in range(1, len(seq)):
        cur = seq.iloc[frame_index]
        cur_feat, cur_meta = mod.extract_one(
            torch, model, device, Path(cur["image_path_full_resolved"]),
            args.resize_long, args.top_k, args.detection_threshold
        )
        feature_rows.append({
            "sequence_frame_id": int(cur["sequence_frame_id"]),
            "token0_id": int(cur["token0_id"]),
            **cur_meta,
        })

        pair = pair_manifest.iloc[frame_index - 1]
        result = mod.match_and_estimate(
            torch, model, device, prev_feat, cur_feat,
            args.min_cossim, args.ransac_threshold
        )

        pair_rows.append({
            "comparison_pair_id": int(pair["comparison_pair_id"]),
            "pair_number": int(pair["pair_number"]),
            "frame_index_a": int(pair["frame_index_a"]),
            "frame_index_b": int(pair["frame_index_b"]),
            "token0_a": int(pair["token0_a"]),
            "token0_b": int(pair["token0_b"]),
            **result,
        })

        prev_feat = cur_feat

        if frame_index % 50 == 0 or frame_index == len(seq) - 1:
            print(f"XFeat chain: frame {frame_index + 1}/{len(seq)}, pairs {frame_index}/{len(seq)-1}")

    wall_seconds = time.perf_counter() - wall_start

    feature_df = pd.DataFrame(feature_rows)
    pair_df = pd.DataFrame(pair_rows)

    feature_path = metadata_dir / "s8r4_xfeat_frame_features.csv"
    pair_path = metadata_dir / "s8r4_xfeat_pair_diagnostics.csv"
    feature_df.to_csv(feature_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    affine_success = bool(pair_df["affine_ok"].astype(bool).all())

    report = {
        "stage": "S8.R4",
        "status": "PASS_XFEAT_RELATIVE_FRONTEND" if affine_success else "REVIEW_XFEAT_INCOMPLETE_CHAIN",
        "frames": int(len(seq)),
        "pairs": int(len(pair_df)),
        "device": str(device),
        "xfeat_commit": xfeat_commit,
        "configuration": {
            "resize_long": args.resize_long,
            "top_k": args.top_k,
            "detection_threshold": args.detection_threshold,
            "min_cossim": args.min_cossim,
            "ransac_threshold": args.ransac_threshold,
            "alignment_prefix_frames": args.alignment_prefix_frames,
            "max_frames": args.max_frames,
        },
        "pair_summary": {
            "affine_success_rate": float(pair_df["affine_ok"].astype(bool).mean()),
            "good_quality_rate": float((pair_df["status"].astype(str) == "good").mean()),
            "matches_median": finite_median(pair_df["matches"]),
            "inliers_median": finite_median(pair_df["inliers"]),
            "inlier_ratio_median": finite_median(pair_df["inlier_ratio"]),
            "inlier_ratio_p05": finite_p05(pair_df["inlier_ratio"]),
            "feature_seconds": float(feature_df["feature_time_ms"].sum() / 1000.0),
            "matching_ransac_seconds": float(pair_df["matching_ransac_time_ms"].sum() / 1000.0),
            "wall_seconds": float(wall_seconds),
            "mean_pair_wall_s": float(wall_seconds / max(len(pair_df), 1)),
        },
        "outputs": {
            "features": str(feature_path),
            "pairs": str(pair_path),
        },
    }

    # -----------------------------------------------------
    # Blind-safe raw relative trajectory
    # -----------------------------------------------------

    if affine_success:
        width = int(
            feature_df.iloc[0]["width"]
        )
        height = int(
            feature_df.iloc[0]["height"]
        )

        raw_traj = (
            mod.integrate_se2_scale_normalized(
                pair_df,
                width,
                height,
            )
        )

        blind_raw = (
            seq[
                [
                    "sequence_frame_id",
                    "token0_id",
                ]
            ]
            .merge(
                raw_traj,
                on="sequence_frame_id",
                how="inner",
                validate="one_to_one",
            )
            .sort_values(
                "sequence_frame_id",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        forbidden_blind_columns = {
            "x_enu_m",
            "y_enu_m",
            "reference_x_m",
            "reference_y_m",
            "reference_cumulative_distance_m",
            "prefix_aligned_x_m",
            "prefix_aligned_y_m",
            "global_aligned_x_m",
            "global_aligned_y_m",
            "prefix_locked_error_m",
            "global_alignment_error_m",
            "lat",
            "lon",
            "latitude",
            "longitude",
        }

        leaked = sorted(
            forbidden_blind_columns
            & set(blind_raw.columns)
        )

        if leaked:
            raise RuntimeError(
                "Reference/evaluation leakage into "
                f"blind raw XFeat trajectory: {leaked}"
            )

        blind_raw["coordinate_contract"] = (
            "relative_visual_image_only"
        )
        blind_raw["reference_used"] = False

        blind_raw_path = (
            metadata_dir
            / "s8r4_xfeat_relative_trajectory_blind_raw.csv"
        )

        blind_raw.to_csv(
            blind_raw_path,
            index=False,
        )

        report["outputs"]["blind_raw"] = str(
            blind_raw_path
        )

        report["blind_contract"] = {
            "reference_used": False,
            "gps_used": False,
            "srt_used": False,
            "coordinate_unit": (
                "integrated_visual_pixel_scale"
            ),
            "metric_scale_available": False,
            "map_alignment_available": False,
        }

    # In blind mode we deliberately stop before any
    # reference alignment, error scoring, ORB comparison,
    # or evaluation plotting.
    if args.blind_only:
        if not affine_success:
            raise RuntimeError(
                "Cannot emit complete blind trajectory "
                "because XFeat affine chain is incomplete."
            )

        report["status"] = (
            "PASS_XFEAT_RELATIVE_FRONTEND_BLIND"
        )

        report_path = (
            reports_dir
            / "s8r4_xfeat_relative_frontend_report.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("Blind XFeat summary")
        print("-" * 72)
        print("status:", report["status"])
        print("frames:", report["frames"])
        print("pairs:", report["pairs"])
        print(
            "reference used:",
            report["blind_contract"]["reference_used"],
        )
        print(
            "metric scale available:",
            report["blind_contract"][
                "metric_scale_available"
            ],
        )
        print(
            "blind raw trajectory:",
            report["outputs"]["blind_raw"],
        )
        print("report:", report_path)

        return

    comparison_rows = []

    if affine_success:
        aligned, traj_summary = align_and_score(
            mod, seq, pair_df, feature_df,
            args.alignment_prefix_frames,
            thresholds,
            args.sustain_frames,
        )

        aligned_path = metadata_dir / "s8r4_xfeat_relative_trajectory_aligned_eval_only.csv"
        aligned.to_csv(aligned_path, index=False)

        prefix = traj_summary["prefix_locked_alignment"]
        global_s = traj_summary["global_alignment"]

        comparison_rows.append({
            "method": "xfeat",
            "frames": len(seq),
            "pairs": len(pair_df),
            "affine_success_rate": report["pair_summary"]["affine_success_rate"],
            "good_quality_rate": report["pair_summary"]["good_quality_rate"],
            "inlier_ratio_median": report["pair_summary"]["inlier_ratio_median"],
            "inlier_ratio_p05": report["pair_summary"]["inlier_ratio_p05"],
            "inliers_median": report["pair_summary"]["inliers_median"],
            "global_rmse_m": global_s["rmse_m"],
            "prefix_rmse_m": prefix["rmse_m"],
            "prefix_mean_m": prefix["mean_error_m"],
            "prefix_median_m": prefix["median_error_m"],
            "prefix_p95_m": prefix["p95_error_m"],
            "prefix_max_m": prefix["max_error_m"],
            "final_error_m": prefix["final_error_m"],
            "evaluation_distance_m": prefix["evaluation_distance_m"],
            "final_drift_per_100m": prefix["final_drift_per_100m"],
            "wall_seconds": wall_seconds,
            "mean_pair_wall_s": report["pair_summary"]["mean_pair_wall_s"],
        })

        # Add existing ORB summary rows for direct comparison.
        orb_sum = pd.read_csv(args.orb_summary)
        for _, row in orb_sum.iterrows():
            method = "orb_" + str(row.get("variant", "unknown"))
            comparison_rows.append({
                "method": method,
                "frames": int(row.get("frames", len(seq))),
                "pairs": len(seq) - 1,
                "affine_success_rate": float("nan"),
                "good_quality_rate": float("nan"),
                "inlier_ratio_median": float("nan"),
                "inlier_ratio_p05": float("nan"),
                "inliers_median": float("nan"),
                "global_rmse_m": row.get("global_alignment.rmse_m"),
                "prefix_rmse_m": row.get("prefix_locked_alignment.rmse_m"),
                "prefix_mean_m": row.get("prefix_locked_alignment.mean_error_m"),
                "prefix_median_m": row.get("prefix_locked_alignment.median_error_m"),
                "prefix_p95_m": row.get("prefix_locked_alignment.p95_error_m"),
                "prefix_max_m": row.get("prefix_locked_alignment.max_error_m"),
                "final_error_m": row.get("prefix_locked_alignment.final_error_m"),
                "evaluation_distance_m": row.get("prefix_locked_alignment.evaluation_distance_m"),
                "final_drift_per_100m": row.get("prefix_locked_alignment.final_drift_per_100m"),
                "wall_seconds": float("nan"),
                "mean_pair_wall_s": float("nan"),
            })

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = metadata_dir / "s8r4_xfeat_orb_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)

        crossing_path = metadata_dir / "s8r4_xfeat_drift_threshold_crossings.csv"
        pd.DataFrame(traj_summary["threshold_crossings"]).to_csv(crossing_path, index=False)

        save_plots(aligned, pair_df, args.orb_aligned, args.orb_pairs, figures_dir)

        report["trajectory_summary"] = traj_summary
        report["outputs"]["aligned"] = str(aligned_path)
        report["outputs"]["comparison"] = str(comparison_path)
        report["outputs"]["crossings"] = str(crossing_path)
        report["outputs"]["figures"] = str(figures_dir)

    report_path = reports_dir / "s8r4_xfeat_relative_frontend_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("XFeat summary")
    print("-" * 72)
    print("status:", report["status"])
    print("frames:", report["frames"])
    print("pairs:", report["pairs"])
    print("affine success:", report["pair_summary"]["affine_success_rate"])
    print("good quality:", report["pair_summary"]["good_quality_rate"])
    print("inlier ratio median:", report["pair_summary"]["inlier_ratio_median"])
    print("inliers median:", report["pair_summary"]["inliers_median"])
    print("wall seconds:", report["pair_summary"]["wall_seconds"])

    if affine_success:
        prefix = report["trajectory_summary"]["prefix_locked_alignment"]
        print()
        print("Prefix-locked trajectory")
        print("-" * 72)
        print("RMSE:", prefix["rmse_m"])
        print("p95:", prefix["p95_error_m"])
        print("max:", prefix["max_error_m"])
        print("final error:", prefix["final_error_m"])
        print("final drift per 100m:", prefix["final_drift_per_100m"])

    print()
    print("outputs:")
    for k, v in report["outputs"].items():
        print(k + ":", v)
    print("report:", report_path)


if __name__ == "__main__":
    main()
