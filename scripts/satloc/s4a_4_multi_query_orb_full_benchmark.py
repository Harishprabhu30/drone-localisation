from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uavloc.localization.orb_tile_retrieval import preprocess_bgr, detect_orb


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def contains_lonlat(row: pd.Series, lon: float, lat: float) -> bool:
    lon_min = min(float(row["lon_tl"]), float(row["lon_br"]))
    lon_max = max(float(row["lon_tl"]), float(row["lon_br"]))
    lat_min = min(float(row["lat_tl"]), float(row["lat_br"]))
    lat_max = max(float(row["lat_tl"]), float(row["lat_br"]))
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def center_error_m(query_row: pd.Series, tile_row: pd.Series) -> float:
    dx = float(query_row["utm_x_m"]) - float(tile_row["utm_x_m"])
    dy = float(query_row["utm_y_m"]) - float(tile_row["utm_y_m"])
    return float(np.sqrt(dx * dx + dy * dy))


def ratio_match_orb_fast(query_des, tile_des, matcher, ratio: float):
    if query_des is None or tile_des is None:
        return []

    if len(query_des) == 0 or len(tile_des) < 2:
        return []

    matches = matcher.knnMatch(query_des, tile_des, k=2)

    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    return good


def homography_from_cached_points(query_kp, tile_kp_arr, good_matches, ransac_thresh: float):
    if len(good_matches) < 4:
        return 0, 0.0, False

    src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([tile_kp_arr[m.trainIdx, 0:2] for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(ransac_thresh))

    if mask is None:
        return 0, 0.0, False

    inliers = int(mask.ravel().sum())
    inlier_ratio = float(inliers / max(len(good_matches), 1))
    success = bool(H is not None and inliers >= 4)

    return inliers, inlier_ratio, success


def select_query_rows(
    uav_df: pd.DataFrame,
    sequence: str,
    query_token0_ids: str | None,
    max_queries: int,
) -> pd.DataFrame:
    seq_df = uav_df[uav_df["sequence"] == sequence].copy()
    seq_df = seq_df.sort_values("frame_index_in_sequence").reset_index(drop=True)

    if len(seq_df) == 0:
        raise ValueError(f"No rows found for sequence={sequence}")

    if query_token0_ids:
        tokens = [int(x.strip()) for x in query_token0_ids.split(",") if x.strip()]
        selected = seq_df[seq_df["token0_id"].astype(int).isin(tokens)].copy()
        selected["token_order"] = selected["token0_id"].astype(int).map({t: i for i, t in enumerate(tokens)})
        selected = selected.sort_values("token_order").drop(columns=["token_order"])
        if len(selected) == 0:
            raise ValueError(f"No matching token0 IDs found: {tokens}")
        return selected.reset_index(drop=True)

    if max_queries >= len(seq_df):
        return seq_df.copy()

    indices = np.linspace(0, len(seq_df) - 1, max_queries, dtype=int)
    return seq_df.iloc[indices].copy().reset_index(drop=True)


def load_tile_feature_cache(sat_df: pd.DataFrame, cache_dir: Path):
    features = []

    print("\nLoading satellite feature cache into memory...")
    for i, (_, row) in enumerate(sat_df.iterrows(), start=1):
        tile_index = int(row["tile_index"])
        feature_path = cache_dir / f"tile_{tile_index:05d}.npz"

        if not feature_path.exists():
            features.append(
                {
                    "tile_index": tile_index,
                    "tile_row": row,
                    "feature_path": str(feature_path),
                    "exists": False,
                    "keypoints": np.zeros((0, 7), dtype=np.float32),
                    "descriptors": np.zeros((0, 32), dtype=np.uint8),
                }
            )
            continue

        with np.load(feature_path) as data:
            keypoints = data["keypoints"].copy()
            descriptors = data["descriptors"].copy()

        features.append(
            {
                "tile_index": tile_index,
                "tile_row": row,
                "feature_path": str(feature_path),
                "exists": True,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }
        )

        if i % 1000 == 0 or i == len(sat_df):
            print(f"  loaded {i}/{len(sat_df)} tile features")

    return features


def save_topk_panel(query_bgr, ranked_df: pd.DataFrame, out_path: Path, top_k: int, title: str):
    top_df = ranked_df.head(top_k).copy()

    ncols = 5
    n_items = len(top_df) + 1
    nrows = int(np.ceil(n_items / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(query_rgb)
    axes[0].set_title("UAV query")
    axes[0].axis("off")

    for ax_i, (_, row) in enumerate(top_df.iterrows(), start=1):
        tile_bgr = read_bgr(Path(str(row["tile_path"])))
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)

        ax_title = (
            f"Rank {int(row['rank'])} | tile {int(row['tile_index'])}\n"
            f"score={row['score']:.2f}, good={int(row['good_matches'])}, "
            f"inl={int(row['ransac_inliers'])}\n"
            f"GT={bool(row['contains_gt'])}, err={row['center_error_m']:.1f}m"
        )

        axes[ax_i].imshow(tile_rgb)
        axes[ax_i].set_title(ax_title, fontsize=9)
        axes[ax_i].axis("off")

    for j in range(n_items, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_summary_plots(summary_df: pd.DataFrame, aggregate: dict, out_fig_dir: Path, run_id: str):
    recall_path = out_fig_dir / f"{run_id}_recall_summary.png"
    rank_path = out_fig_dir / f"{run_id}_first_correct_rank.png"
    error_path = out_fig_dir / f"{run_id}_top1_error.png"
    runtime_path = out_fig_dir / f"{run_id}_runtime.png"

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["R@1", "R@5", "R@10"]
    values = [
        aggregate["recall_at_1"],
        aggregate["recall_at_5"],
        aggregate["recall_at_10"],
    ]

    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("ORB full-map retrieval recall")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value + 0.02, 0.03),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    if max(values) == 0:
        ax.text(
            1,
            0.5,
            "All recall values are 0.0\nNo GT tile found in top-10",
            ha="center",
            va="center",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(recall_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(summary_df["query_token0_id"], summary_df["first_correct_rank"], marker="o")
    ax.set_xlabel("Query token0_id")
    ax.set_ylabel("First correct rank")
    ax.set_title("First GT-containing tile rank per query")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(rank_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(summary_df["query_token0_id"], summary_df["top1_center_error_m"], marker="o")
    ax.set_xlabel("Query token0_id")
    ax.set_ylabel("Top-1 center error [m]")
    ax.set_title("Top-1 localization error per query")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(error_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(summary_df["query_token0_id"], summary_df["elapsed_s"], marker="o")
    ax.set_xlabel("Query token0_id")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Full-map retrieval runtime per query")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(runtime_path, dpi=160)
    plt.close(fig)

    return {
        "recall_plot": str(recall_path),
        "first_correct_rank_plot": str(rank_path),
        "top1_error_plot": str(error_path),
        "runtime_plot": str(runtime_path),
    }


def run_one_query(
    query_row: pd.Series,
    tile_features,
    query_bgr,
    variant: str,
    nfeatures: int,
    ratio: float,
    ransac_thresh: float,
    matcher,
    args,
):
    query_processed = preprocess_bgr(
        query_bgr,
        variant=variant,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_size=args.clahe_tile_size,
        alt_clahe_clip_limit=args.alt_clahe_clip_limit,
        alt_clahe_tile_size=args.alt_clahe_tile_size,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
    )
    query_kp, query_des = detect_orb(query_processed, nfeatures=nfeatures)

    gt_lon = float(query_row["lon"])
    gt_lat = float(query_row["lat"])

    rows = []

    for item in tile_features:
        tile_row = item["tile_row"]
        tile_index = int(item["tile_index"])

        if not item["exists"]:
            rows.append(
                {
                    "tile_index": tile_index,
                    "tile_filename": str(tile_row["filename"]),
                    "tile_path": str(tile_row["tile_path"]),
                    "feature_exists": False,
                    "contains_gt": contains_lonlat(tile_row, gt_lon, gt_lat),
                    "center_error_m": center_error_m(query_row, tile_row),
                    "tile_keypoints": 0,
                    "good_matches": 0,
                    "ransac_inliers": 0,
                    "inlier_ratio": 0.0,
                    "homography_success": False,
                    "score": 0.0,
                }
            )
            continue

        good = ratio_match_orb_fast(
            query_des=query_des,
            tile_des=item["descriptors"],
            matcher=matcher,
            ratio=ratio,
        )

        inliers, inlier_ratio, h_success = homography_from_cached_points(
            query_kp=query_kp,
            tile_kp_arr=item["keypoints"],
            good_matches=good,
            ransac_thresh=ransac_thresh,
        )

        score = float(inliers + 0.1 * len(good))

        rows.append(
            {
                "tile_index": tile_index,
                "tile_filename": str(tile_row["filename"]),
                "tile_path": str(tile_row["tile_path"]),
                "feature_path": str(item["feature_path"]),
                "feature_exists": True,
                "contains_gt": contains_lonlat(tile_row, gt_lon, gt_lat),
                "center_error_m": center_error_m(query_row, tile_row),
                "tile_keypoints": int(item["keypoints"].shape[0]),
                "good_matches": int(len(good)),
                "ransac_inliers": int(inliers),
                "inlier_ratio": float(inlier_ratio),
                "homography_success": bool(h_success),
                "score": score,
            }
        )

    ranked_df = pd.DataFrame(rows)
    ranked_df = ranked_df.sort_values(
        ["score", "ransac_inliers", "good_matches", "inlier_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranked_df.insert(0, "rank", np.arange(1, len(ranked_df) + 1))

    return ranked_df, len(query_kp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--query-token0-ids", default=None)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--variant", default="V2_clahe_luma")
    parser.add_argument("--nfeatures", type=int, default=1200)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-topk-panels", action="store_true")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-size", type=int, default=8)
    parser.add_argument("--alt-clahe-clip-limit", type=float, default=1.0)
    parser.add_argument("--alt-clahe-tile-size", type=int, default=8)
    parser.add_argument("--bilateral-d", type=int, default=13)
    parser.add_argument("--bilateral-sigma-color", type=float, default=30)
    parser.add_argument("--bilateral-sigma-space", type=float, default=55)
    args = parser.parse_args()

    cache_name = f"{args.variant}_nf{args.nfeatures}"
    cache_dir = Path("outputs/satloc/cache/s4a_orb_tile_features") / cache_name

    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing cache dir: {cache_dir}")

    if args.query_token0_ids:
        query_label = "tokens_" + args.query_token0_ids.replace(",", "_")
    else:
        query_label = f"q{args.max_queries}"

    run_id = (
        f"{args.sequence}_multiquery_fullcached_{args.variant}"
        f"_nf{args.nfeatures}_{query_label}"
    )

    out_meta_dir = Path("outputs/satloc/metadata/s4a_orb_tile_retrieval/multi_query_benchmark") / run_id
    out_report_dir = Path("outputs/satloc/reports/s4a_orb_tile_retrieval/multi_query_benchmark") / run_id
    out_fig_dir = Path("outputs/satloc/figures/s4a_orb_tile_retrieval/multi_query_benchmark") / run_id

    ensure_dir(out_meta_dir)
    ensure_dir(out_report_dir)
    ensure_dir(out_fig_dir)

    uav_df = pd.read_csv("outputs/satloc/metadata/uav_frames_index_enriched.csv")
    sat_df = pd.read_csv("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")
    sat_df = sat_df[sat_df["tile_exists"] == True].copy()
    sat_df = sat_df.sort_values("tile_index").reset_index(drop=True)

    if args.max_tiles is not None:
        sat_df = sat_df.head(args.max_tiles).copy()

    query_df = select_query_rows(
        uav_df=uav_df,
        sequence=args.sequence,
        query_token0_ids=args.query_token0_ids,
        max_queries=args.max_queries,
    )

    tile_features = load_tile_feature_cache(sat_df=sat_df, cache_dir=cache_dir)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    print("\nS4A.4 multi-query ORB full-map benchmark")
    print("----------------------------------------")
    print(f"Run ID: {run_id}")
    print(f"Variant: {args.variant}")
    print(f"nfeatures: {args.nfeatures}")
    print(f"Tiles ranked per query: {len(tile_features)}")
    print(f"Queries: {len(query_df)}")
    print(f"Save top-k panels: {args.save_topk_panels}")

    summary_rows = []
    start_all = perf_counter()

    for qi, (_, query_row) in enumerate(query_df.iterrows(), start=1):
        query_token = int(query_row["token0_id"])
        query_path = Path(str(query_row["image_path"]))
        query_bgr = read_bgr(query_path)

        print(f"\nQuery {qi}/{len(query_df)} | token={query_token} | {query_row['filename']}")

        start_q = perf_counter()

        ranked_df, query_kp_count = run_one_query(
            query_row=query_row,
            tile_features=tile_features,
            query_bgr=query_bgr,
            variant=args.variant,
            nfeatures=args.nfeatures,
            ratio=args.ratio,
            ransac_thresh=args.ransac_thresh,
            matcher=matcher,
            args=args,
        )

        elapsed_s = float(perf_counter() - start_q)

        ranked_df.insert(1, "sequence", args.sequence)
        ranked_df.insert(2, "query_token0_id", query_token)
        ranked_df.insert(3, "query_frame_index_in_sequence", int(query_row["frame_index_in_sequence"]))
        ranked_df.insert(4, "query_filename", str(query_row["filename"]))
        ranked_df.insert(5, "query_path", str(query_path))
        ranked_df.insert(6, "query_lon", float(query_row["lon"]))
        ranked_df.insert(7, "query_lat", float(query_row["lat"]))

        query_csv = out_meta_dir / f"query_token{query_token:04d}_ranked_tiles.csv"
        ranked_df.to_csv(query_csv, index=False)

        correct_df = ranked_df[ranked_df["contains_gt"] == True].copy()

        first_correct_rank = int(correct_df["rank"].min()) if len(correct_df) else None

        if len(correct_df):
            first_correct_row = correct_df.sort_values("rank").iloc[0]
            closest_correct_row = correct_df.sort_values("center_error_m").iloc[0]
            first_correct_score = float(first_correct_row["score"])
            first_correct_error = float(first_correct_row["center_error_m"])
            closest_correct_rank = int(closest_correct_row["rank"])
            closest_correct_error = float(closest_correct_row["center_error_m"])
        else:
            first_correct_score = None
            first_correct_error = None
            closest_correct_rank = None
            closest_correct_error = None

        top1 = ranked_df.iloc[0]
        top2 = ranked_df.iloc[1] if len(ranked_df) > 1 else ranked_df.iloc[0]

        summary_row = {
            "sequence": args.sequence,
            "query_token0_id": query_token,
            "query_frame_index_in_sequence": int(query_row["frame_index_in_sequence"]),
            "query_filename": str(query_row["filename"]),
            "query_path": str(query_path),
            "query_lon": float(query_row["lon"]),
            "query_lat": float(query_row["lat"]),
            "query_keypoints": int(query_kp_count),
            "tiles_ranked": int(len(ranked_df)),
            "first_correct_rank": first_correct_rank,
            "first_correct_score": first_correct_score,
            "first_correct_error_m": first_correct_error,
            "closest_correct_rank": closest_correct_rank,
            "closest_correct_error_m": closest_correct_error,
            "recall_at_1": bool((ranked_df.head(1)["contains_gt"] == True).any()),
            "recall_at_5": bool((ranked_df.head(5)["contains_gt"] == True).any()),
            "recall_at_10": bool((ranked_df.head(10)["contains_gt"] == True).any()),
            "top1_tile_index": int(top1["tile_index"]),
            "top1_score": float(top1["score"]),
            "top1_good_matches": int(top1["good_matches"]),
            "top1_ransac_inliers": int(top1["ransac_inliers"]),
            "top1_contains_gt": bool(top1["contains_gt"]),
            "top1_center_error_m": float(top1["center_error_m"]),
            "top2_score": float(top2["score"]),
            "score_margin_top1_top2": float(top1["score"] - top2["score"]),
            "score_margin_top1_first_correct": (
                float(top1["score"] - first_correct_score)
                if first_correct_score is not None
                else None
            ),
            "elapsed_s": elapsed_s,
            "ranked_csv": str(query_csv),
        }

        summary_rows.append(summary_row)

        if args.save_topk_panels:
            panel_path = out_fig_dir / f"query_token{query_token:04d}_top{args.top_k}_panel.png"
            save_topk_panel(
                query_bgr=query_bgr,
                ranked_df=ranked_df,
                out_path=panel_path,
                top_k=args.top_k,
                title=f"ORB full-map top-{args.top_k} | token {query_token}",
            )

        print(
            f"  top1_tile={summary_row['top1_tile_index']} "
            f"top1_err={summary_row['top1_center_error_m']:.1f}m "
            f"first_correct_rank={summary_row['first_correct_rank']} "
            f"R@1={summary_row['recall_at_1']} "
            f"R@5={summary_row['recall_at_5']} "
            f"R@10={summary_row['recall_at_10']} "
            f"time={elapsed_s:.1f}s"
        )

    summary_df = pd.DataFrame(summary_rows)

    aggregate = {
        "run_id": run_id,
        "sequence": args.sequence,
        "variant": args.variant,
        "nfeatures": int(args.nfeatures),
        "ratio": float(args.ratio),
        "ransac_thresh": float(args.ransac_thresh),
        "query_count": int(len(summary_df)),
        "tiles_ranked_per_query": int(len(tile_features)),
        "recall_at_1": float(summary_df["recall_at_1"].mean()),
        "recall_at_5": float(summary_df["recall_at_5"].mean()),
        "recall_at_10": float(summary_df["recall_at_10"].mean()),
        "first_correct_rank_median": float(summary_df["first_correct_rank"].dropna().median())
        if summary_df["first_correct_rank"].notna().any()
        else None,
        "first_correct_rank_mean": float(summary_df["first_correct_rank"].dropna().mean())
        if summary_df["first_correct_rank"].notna().any()
        else None,
        "top1_error_m_median": float(summary_df["top1_center_error_m"].median()),
        "top1_error_m_mean": float(summary_df["top1_center_error_m"].mean()),
        "closest_correct_error_m_median": float(summary_df["closest_correct_error_m"].dropna().median())
        if summary_df["closest_correct_error_m"].notna().any()
        else None,
        "elapsed_per_query_s_mean": float(summary_df["elapsed_s"].mean()),
        "elapsed_total_s": float(perf_counter() - start_all),
    }

    plot_paths = save_summary_plots(
        summary_df=summary_df,
        aggregate=aggregate,
        out_fig_dir=out_fig_dir,
        run_id=run_id,
    )

    summary_csv = out_meta_dir / "multi_query_summary.csv"
    aggregate_json = out_report_dir / "multi_query_aggregate_metrics.json"

    summary_df.to_csv(summary_csv, index=False)

    aggregate["summary_csv"] = str(summary_csv)
    aggregate["figures"] = plot_paths

    with aggregate_json.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print("\nAggregate result:")
    print(json.dumps(aggregate, indent=2))

    print("\nSaved:")
    print(f"  Summary CSV: {summary_csv}")
    print(f"  Aggregate JSON: {aggregate_json}")
    print(f"  Figures: {out_fig_dir}")


if __name__ == "__main__":
    main()