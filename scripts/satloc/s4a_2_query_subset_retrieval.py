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

from uavloc.localization.orb_tile_retrieval import match_orb_pair


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


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


def contains_lonlat(row: pd.Series, lon: float, lat: float) -> bool:
    lon_min = min(float(row["lon_tl"]), float(row["lon_br"]))
    lon_max = max(float(row["lon_tl"]), float(row["lon_br"]))
    lat_min = min(float(row["lat_tl"]), float(row["lat_br"]))
    lat_max = max(float(row["lat_tl"]), float(row["lat_br"]))
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def find_containing_tile(sat_df: pd.DataFrame, lon: float, lat: float) -> pd.Series:
    containing = sat_df[
        sat_df.apply(lambda r: contains_lonlat(r, lon=lon, lat=lat), axis=1)
    ].copy()

    if len(containing) == 0:
        sat_df = sat_df.copy()
        sat_df["center_lonlat_dist"] = np.sqrt(
            (sat_df["lon_center"].astype(float) - lon) ** 2
            + (sat_df["lat_center"].astype(float) - lat) ** 2
        )
        return sat_df.sort_values("center_lonlat_dist").iloc[0]

    containing["center_lonlat_dist"] = np.sqrt(
        (containing["lon_center"].astype(float) - lon) ** 2
        + (containing["lat_center"].astype(float) - lat) ** 2
    )
    return containing.sort_values("center_lonlat_dist").iloc[0]


def center_error_m(query_row: pd.Series, tile_row: pd.Series) -> float:
    dx = float(query_row["utm_x_m"]) - float(tile_row["utm_x_m"])
    dy = float(query_row["utm_y_m"]) - float(tile_row["utm_y_m"])
    return float(np.sqrt(dx * dx + dy * dy))


def build_candidate_subset(
    sat_df: pd.DataFrame,
    true_tile_index: int,
    radius: int,
) -> pd.DataFrame:
    low = max(0, int(true_tile_index) - int(radius))
    high = int(true_tile_index) + int(radius)
    subset = sat_df[
        (sat_df["tile_index"].astype(int) >= low)
        & (sat_df["tile_index"].astype(int) <= high)
    ].copy()
    return subset.sort_values("tile_index").reset_index(drop=True)


def save_topk_panel(
    query_bgr: np.ndarray,
    ranked_df: pd.DataFrame,
    query_title: str,
    out_path: Path,
    top_k: int = 10,
) -> None:
    top_df = ranked_df.head(top_k).copy()

    ncols = 5
    n_tiles = len(top_df)
    n_items = n_tiles + 1
    nrows = int(np.ceil(n_items / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(query_rgb)
    axes[0].set_title(query_title)
    axes[0].axis("off")

    for ax_i, (_, row) in enumerate(top_df.iterrows(), start=1):
        tile = read_bgr(Path(str(row["tile_path"])))
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        title = (
            f"Rank {int(row['rank'])} | tile {int(row['tile_index'])}\n"
            f"score={row['score']:.2f}, good={int(row['good_matches'])}, "
            f"inl={int(row['ransac_inliers'])}\n"
            f"containsGT={bool(row['contains_gt'])}, "
            f"err={row['center_error_m']:.1f}m"
        )

        axes[ax_i].imshow(tile_rgb)
        axes[ax_i].set_title(title, fontsize=9)
        axes[ax_i].axis("off")

    for j in range(n_items, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_score_plot(ranked_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ranked_df["rank"], ranked_df["score"], marker="o", linewidth=1)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_satloc.yaml")
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--query-token0-id", type=int, default=100)
    parser.add_argument("--query-frame-index", type=int, default=None)
    parser.add_argument("--candidate-radius", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--variants", default="V2_clahe_luma")
    parser.add_argument("--nfeatures", type=int, default=1200)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-size", type=int, default=8)
    parser.add_argument("--alt-clahe-clip-limit", type=float, default=1.0)
    parser.add_argument("--alt-clahe-tile-size", type=int, default=8)
    parser.add_argument("--bilateral-d", type=int, default=13)
    parser.add_argument("--bilateral-sigma-color", type=float, default=30)
    parser.add_argument("--bilateral-sigma-space", type=float, default=55)
    args = parser.parse_args()

    out_meta_dir = Path("outputs/satloc/metadata/s4a_orb_tile_retrieval/subset_debug")
    out_report_dir = Path("outputs/satloc/reports/s4a_orb_tile_retrieval/subset_debug")
    out_fig_dir = Path("outputs/satloc/figures/s4a_orb_tile_retrieval/subset_debug")
    ensure_dir(out_meta_dir)
    ensure_dir(out_report_dir)
    ensure_dir(out_fig_dir)

    uav_df = pd.read_csv("outputs/satloc/metadata/uav_frames_index_enriched.csv")
    sat_df = pd.read_csv("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")
    sat_df = sat_df[sat_df["tile_exists"] == True].copy()

    query_row = find_query_row(
        uav_df=uav_df,
        sequence=args.sequence,
        token0_id=args.query_token0_id,
        frame_index=args.query_frame_index,
    )

    gt_lon = float(query_row["lon"])
    gt_lat = float(query_row["lat"])

    true_tile_row = find_containing_tile(sat_df, lon=gt_lon, lat=gt_lat)
    true_tile_index = int(true_tile_row["tile_index"])

    candidate_df = build_candidate_subset(
        sat_df=sat_df,
        true_tile_index=true_tile_index,
        radius=args.candidate_radius,
    )

    query_path = Path(str(query_row["image_path"]))
    query_bgr = read_bgr(query_path)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run_id = (
        f"{args.sequence}_token{int(query_row['token0_id']):04d}"
        f"_subset_r{args.candidate_radius}"
    )

    print("\nS4A.2 query subset retrieval")
    print("----------------------------")
    print(f"Query: {query_path}")
    print(f"Query lon/lat: {gt_lon:.8f}, {gt_lat:.8f}")
    print(f"GT-containing/debug tile: {true_tile_index} | {true_tile_row['filename']}")
    print(f"Candidate subset size: {len(candidate_df)}")
    print(f"Variants: {variants}")

    all_variant_metrics = {}

    for variant in variants:
        print(f"\nRunning variant: {variant}")
        start_variant = perf_counter()

        rows = []
        for i, (_, tile_row) in enumerate(candidate_df.iterrows(), start=1):
            tile_path = Path(str(tile_row["tile_path"]))
            tile_bgr = read_bgr(tile_path)

            result = match_orb_pair(
                query_bgr=query_bgr,
                tile_bgr=tile_bgr,
                variant=variant,
                nfeatures=args.nfeatures,
                ratio=args.ratio,
                ransac_thresh=args.ransac_thresh,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_size=args.clahe_tile_size,
                alt_clahe_clip_limit=args.alt_clahe_clip_limit,
                alt_clahe_tile_size=args.alt_clahe_tile_size,
                bilateral_d=args.bilateral_d,
                bilateral_sigma_color=args.bilateral_sigma_color,
                bilateral_sigma_space=args.bilateral_sigma_space,
            )

            contains_gt = contains_lonlat(tile_row, lon=gt_lon, lat=gt_lat)
            err_m = center_error_m(query_row, tile_row)

            rows.append(
                {
                    "sequence": args.sequence,
                    "query_token0_id": int(query_row["token0_id"]),
                    "query_frame_index_in_sequence": int(query_row["frame_index_in_sequence"]),
                    "query_filename": str(query_row["filename"]),
                    "query_path": str(query_path),
                    "query_lon": gt_lon,
                    "query_lat": gt_lat,
                    "variant": variant,
                    "tile_index": int(tile_row["tile_index"]),
                    "tile_filename": str(tile_row["filename"]),
                    "tile_path": str(tile_path),
                    "tile_lon_center": float(tile_row["lon_center"]),
                    "tile_lat_center": float(tile_row["lat_center"]),
                    "contains_gt": bool(contains_gt),
                    "center_error_m": err_m,
                    "query_keypoints": result.query_keypoints,
                    "tile_keypoints": result.tile_keypoints,
                    "good_matches": result.good_matches,
                    "ransac_inliers": result.ransac_inliers,
                    "inlier_ratio": result.inlier_ratio,
                    "homography_success": result.homography_success,
                    "score": result.score,
                    "elapsed_ms": result.elapsed_ms,
                }
            )

            if i % 25 == 0 or i == len(candidate_df):
                print(f"  processed {i}/{len(candidate_df)} tiles")

        df = pd.DataFrame(rows)
        df = df.sort_values(
            ["score", "ransac_inliers", "good_matches", "inlier_ratio"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))

        correct_rows = df[df["contains_gt"] == True]
        first_correct_rank = int(correct_rows["rank"].min()) if len(correct_rows) else None

        recall_at_1 = bool((df.head(1)["contains_gt"] == True).any())
        recall_at_5 = bool((df.head(5)["contains_gt"] == True).any())
        recall_at_10 = bool((df.head(10)["contains_gt"] == True).any())

        elapsed_variant_s = float(perf_counter() - start_variant)

        variant_id = f"{run_id}_{variant}"
        csv_path = out_meta_dir / f"{variant_id}_ranked_tiles.csv"
        panel_path = out_fig_dir / f"{variant_id}_top{args.top_k}_panel.png"
        score_plot_path = out_fig_dir / f"{variant_id}_score_by_rank.png"

        df.to_csv(csv_path, index=False)

        save_topk_panel(
            query_bgr=query_bgr,
            ranked_df=df,
            query_title=f"Query token {int(query_row['token0_id'])}\n{variant}",
            out_path=panel_path,
            top_k=args.top_k,
        )

        save_score_plot(
            ranked_df=df,
            out_path=score_plot_path,
            title=f"{variant} score by rank | query token {int(query_row['token0_id'])}",
        )

        metrics = {
            "variant": variant,
            "candidate_count": int(len(candidate_df)),
            "first_correct_rank": first_correct_rank,
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "best_tile_index": int(df.iloc[0]["tile_index"]),
            "best_score": float(df.iloc[0]["score"]),
            "best_good_matches": int(df.iloc[0]["good_matches"]),
            "best_ransac_inliers": int(df.iloc[0]["ransac_inliers"]),
            "best_contains_gt": bool(df.iloc[0]["contains_gt"]),
            "best_center_error_m": float(df.iloc[0]["center_error_m"]),
            "elapsed_s": elapsed_variant_s,
            "mean_tile_elapsed_ms": float(df["elapsed_ms"].mean()),
            "ranked_csv": str(csv_path),
            "topk_panel": str(panel_path),
            "score_plot": str(score_plot_path),
        }

        all_variant_metrics[variant] = metrics

        print(
            f"best_tile={metrics['best_tile_index']} "
            f"score={metrics['best_score']:.2f} "
            f"good={metrics['best_good_matches']} "
            f"inliers={metrics['best_ransac_inliers']} "
            f"containsGT={metrics['best_contains_gt']} "
            f"first_correct_rank={metrics['first_correct_rank']} "
            f"R@1={metrics['recall_at_1']} "
            f"R@5={metrics['recall_at_5']} "
            f"R@10={metrics['recall_at_10']} "
            f"time={metrics['elapsed_s']:.1f}s"
        )

    summary = {
        "run_id": run_id,
        "query": {
            "sequence": args.sequence,
            "token0_id": int(query_row["token0_id"]),
            "frame_index_in_sequence": int(query_row["frame_index_in_sequence"]),
            "filename": str(query_row["filename"]),
            "path": str(query_path),
            "lon": gt_lon,
            "lat": gt_lat,
        },
        "debug_true_tile": {
            "tile_index": true_tile_index,
            "filename": str(true_tile_row["filename"]),
            "path": str(true_tile_row["tile_path"]),
        },
        "candidate_subset": {
            "mode": "debug_true_tile_index_window",
            "radius": int(args.candidate_radius),
            "count": int(len(candidate_df)),
            "tile_index_min": int(candidate_df["tile_index"].min()),
            "tile_index_max": int(candidate_df["tile_index"].max()),
        },
        "metrics_by_variant": all_variant_metrics,
    }

    json_path = out_report_dir / f"{run_id}_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved summary:")
    print(f"{json_path}")
    print(f"Figures:")
    print(f"{out_fig_dir}")


if __name__ == "__main__":
    main()