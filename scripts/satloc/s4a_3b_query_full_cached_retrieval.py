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

from uavloc.localization.orb_tile_retrieval import (
    preprocess_bgr,
    detect_orb,
    ratio_match_orb,
)


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


def find_query_row(
    uav_df: pd.DataFrame,
    sequence: str,
    token0_id: int | None,
    frame_index: int | None,
) -> pd.Series:
    seq_df = uav_df[uav_df["sequence"] == sequence].copy()

    if len(seq_df) == 0:
        raise ValueError(f"No UAV rows found for sequence={sequence}")

    if token0_id is not None:
        rows = seq_df[seq_df["token0_id"] == token0_id]
        if len(rows) == 0:
            raise ValueError(f"No row found for token0_id={token0_id}")
        return rows.iloc[0]

    if frame_index is not None:
        rows = seq_df[seq_df["frame_index_in_sequence"] == frame_index]
        if len(rows) == 0:
            raise ValueError(f"No row found for frame_index={frame_index}")
        return rows.iloc[0]

    return seq_df.iloc[0]


def homography_from_cached_points(
    query_kp,
    tile_kp_arr: np.ndarray,
    good_matches,
    ransac_thresh: float,
) -> tuple[int, float, bool]:
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


def save_topk_panel(
    query_bgr: np.ndarray,
    ranked_df: pd.DataFrame,
    out_path: Path,
    top_k: int,
    title: str,
) -> None:
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


def save_score_plot(ranked_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = ranked_df.head(min(len(ranked_df), 200)).copy()

    ax.plot(plot_df["rank"], plot_df["score"], linewidth=1.2)
    ax.scatter(plot_df["rank"], plot_df["score"], s=12)

    correct = plot_df[plot_df["contains_gt"] == True]
    if len(correct) > 0:
        ax.scatter(correct["rank"], correct["score"], s=60, marker="x", label="GT-containing tile")
        ax.legend()

    ax.set_xlabel("Rank")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--query-token0-id", type=int, default=100)
    parser.add_argument("--query-frame-index", type=int, default=None)
    parser.add_argument("--variant", default="V2_clahe_luma")
    parser.add_argument("--nfeatures", type=int, default=1200)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=10)
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
        raise FileNotFoundError(f"Missing feature cache: {cache_dir}")

    out_meta_dir = Path("outputs/satloc/metadata/s4a_orb_tile_retrieval/full_cached")
    out_report_dir = Path("outputs/satloc/reports/s4a_orb_tile_retrieval/full_cached")
    out_fig_dir = Path("outputs/satloc/figures/s4a_orb_tile_retrieval/full_cached")
    ensure_dir(out_meta_dir)
    ensure_dir(out_report_dir)
    ensure_dir(out_fig_dir)

    uav_df = pd.read_csv("outputs/satloc/metadata/uav_frames_index_enriched.csv")
    sat_df = pd.read_csv("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")
    sat_df = sat_df[sat_df["tile_exists"] == True].copy()
    sat_df = sat_df.sort_values("tile_index").reset_index(drop=True)

    if args.max_tiles is not None:
        sat_df = sat_df.head(args.max_tiles).copy()

    query_row = find_query_row(
        uav_df=uav_df,
        sequence=args.sequence,
        token0_id=args.query_token0_id,
        frame_index=args.query_frame_index,
    )

    query_path = Path(str(query_row["image_path"]))
    query_bgr = read_bgr(query_path)

    gt_lon = float(query_row["lon"])
    gt_lat = float(query_row["lat"])

    query_processed = preprocess_bgr(
        query_bgr,
        variant=args.variant,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_size=args.clahe_tile_size,
        alt_clahe_clip_limit=args.alt_clahe_clip_limit,
        alt_clahe_tile_size=args.alt_clahe_tile_size,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
    )
    query_kp, query_des = detect_orb(query_processed, nfeatures=args.nfeatures)

    run_id = (
        f"{args.sequence}_token{int(query_row['token0_id']):04d}"
        f"_fullcached_{args.variant}_nf{args.nfeatures}"
    )

    print("\nS4A.3B full cached retrieval")
    print("----------------------------")
    print(f"Query: {query_path}")
    print(f"Query lon/lat: {gt_lon:.8f}, {gt_lat:.8f}")
    print(f"Variant: {args.variant}")
    print(f"Cache dir: {cache_dir}")
    print(f"Tiles to rank: {len(sat_df)}")
    print(f"Query keypoints: {len(query_kp)}")

    rows = []
    start_all = perf_counter()

    for i, (_, tile_row) in enumerate(sat_df.iterrows(), start=1):
        tile_index = int(tile_row["tile_index"])
        feature_path = cache_dir / f"tile_{tile_index:05d}.npz"

        if not feature_path.exists():
            rows.append({
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
                "elapsed_ms": 0.0,
            })
            continue

        start_tile = perf_counter()
        data = np.load(feature_path)
        tile_kp_arr = data["keypoints"]
        tile_des = data["descriptors"]

        good = ratio_match_orb(query_des, tile_des, ratio=args.ratio)
        inliers, inlier_ratio, h_success = homography_from_cached_points(
            query_kp=query_kp,
            tile_kp_arr=tile_kp_arr,
            good_matches=good,
            ransac_thresh=args.ransac_thresh,
        )

        score = float(inliers + 0.1 * len(good))
        elapsed_ms = float((perf_counter() - start_tile) * 1000.0)

        rows.append({
            "sequence": args.sequence,
            "query_token0_id": int(query_row["token0_id"]),
            "query_frame_index_in_sequence": int(query_row["frame_index_in_sequence"]),
            "query_filename": str(query_row["filename"]),
            "query_path": str(query_path),
            "query_lon": gt_lon,
            "query_lat": gt_lat,
            "variant": args.variant,
            "tile_index": tile_index,
            "tile_filename": str(tile_row["filename"]),
            "tile_path": str(tile_row["tile_path"]),
            "feature_path": str(feature_path),
            "feature_exists": True,
            "contains_gt": contains_lonlat(tile_row, gt_lon, gt_lat),
            "center_error_m": center_error_m(query_row, tile_row),
            "tile_keypoints": int(tile_kp_arr.shape[0]),
            "good_matches": int(len(good)),
            "ransac_inliers": int(inliers),
            "inlier_ratio": float(inlier_ratio),
            "homography_success": bool(h_success),
            "score": score,
            "elapsed_ms": elapsed_ms,
        })

        if i % 500 == 0 or i == len(sat_df):
            print(f"  ranked {i}/{len(sat_df)} tiles")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["score", "ransac_inliers", "good_matches", "inlier_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    correct_rows = df[df["contains_gt"] == True].copy()
    first_correct_rank = int(correct_rows["rank"].min()) if len(correct_rows) else None

    metrics = {
        "run_id": run_id,
        "sequence": args.sequence,
        "query_token0_id": int(query_row["token0_id"]),
        "query_filename": str(query_row["filename"]),
        "query_path": str(query_path),
        "query_lon": gt_lon,
        "query_lat": gt_lat,
        "variant": args.variant,
        "nfeatures": int(args.nfeatures),
        "ratio": float(args.ratio),
        "ransac_thresh": float(args.ransac_thresh),
        "tiles_ranked": int(len(df)),
        "first_correct_rank": first_correct_rank,
        "recall_at_1": bool((df.head(1)["contains_gt"] == True).any()),
        "recall_at_5": bool((df.head(5)["contains_gt"] == True).any()),
        "recall_at_10": bool((df.head(10)["contains_gt"] == True).any()),
        "best_tile_index": int(df.iloc[0]["tile_index"]),
        "best_score": float(df.iloc[0]["score"]),
        "best_good_matches": int(df.iloc[0]["good_matches"]),
        "best_ransac_inliers": int(df.iloc[0]["ransac_inliers"]),
        "best_contains_gt": bool(df.iloc[0]["contains_gt"]),
        "best_center_error_m": float(df.iloc[0]["center_error_m"]),
        "elapsed_total_s": float(perf_counter() - start_all),
        "mean_tile_elapsed_ms": float(df["elapsed_ms"].mean()),
    }

    csv_path = out_meta_dir / f"{run_id}_ranked_tiles.csv"
    json_path = out_report_dir / f"{run_id}_summary.json"
    panel_path = out_fig_dir / f"{run_id}_top{args.top_k}_panel.png"
    score_plot_path = out_fig_dir / f"{run_id}_score_by_rank.png"

    df.to_csv(csv_path, index=False)

    save_topk_panel(
        query_bgr=query_bgr,
        ranked_df=df,
        out_path=panel_path,
        top_k=args.top_k,
        title=f"S4A.3B full cached top-{args.top_k} | {run_id}",
    )

    save_score_plot(
        ranked_df=df,
        out_path=score_plot_path,
        title=f"S4A.3B score by rank | {run_id}",
    )

    metrics["ranked_csv"] = str(csv_path)
    metrics["topk_panel"] = str(panel_path)
    metrics["score_plot"] = str(score_plot_path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nResult:")
    print(json.dumps(metrics, indent=2))
    print("\nSaved:")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Panel: {panel_path}")
    print(f"Plot: {score_plot_path}")


if __name__ == "__main__":
    main()