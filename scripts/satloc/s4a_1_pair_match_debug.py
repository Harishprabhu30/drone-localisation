from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def find_query_row(uav_df: pd.DataFrame, sequence: str, token0_id: int | None, frame_index: int | None) -> pd.Series:
    seq_df = uav_df[uav_df["sequence"] == sequence].copy()

    if len(seq_df) == 0:
        raise ValueError(f"No UAV rows found for sequence={sequence}")

    if token0_id is not None:
        rows = seq_df[seq_df["token0_id"] == token0_id]
        if len(rows) == 0:
            raise ValueError(f"No row found for sequence={sequence}, token0_id={token0_id}")
        return rows.iloc[0]

    if frame_index is not None:
        rows = seq_df[seq_df["frame_index_in_sequence"] == frame_index]
        if len(rows) == 0:
            raise ValueError(f"No row found for sequence={sequence}, frame_index_in_sequence={frame_index}")
        return rows.iloc[0]

    return seq_df.iloc[0]


def contains_lonlat(row: pd.Series, lon: float, lat: float) -> bool:
    lon_min = min(float(row["lon_tl"]), float(row["lon_br"]))
    lon_max = max(float(row["lon_tl"]), float(row["lon_br"]))
    lat_min = min(float(row["lat_tl"]), float(row["lat_br"]))
    lat_max = max(float(row["lat_tl"]), float(row["lat_br"]))
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def find_debug_true_tile(sat_df: pd.DataFrame, lon: float, lat: float) -> pd.Series:
    containing = sat_df[
        sat_df.apply(lambda r: contains_lonlat(r, lon=lon, lat=lat), axis=1)
    ].copy()

    if len(containing) == 0:
        # fallback: nearest tile center in lon/lat space
        sat_df = sat_df.copy()
        sat_df["lonlat_dist"] = np.sqrt(
            (sat_df["lon_center"].astype(float) - lon) ** 2
            + (sat_df["lat_center"].astype(float) - lat) ** 2
        )
        return sat_df.sort_values("lonlat_dist").iloc[0]

    # Use approximate lon/lat degree distance only for selecting among overlapping tiles.
    containing["center_lonlat_dist"] = np.sqrt(
        (containing["lon_center"].astype(float) - lon) ** 2
        + (containing["lat_center"].astype(float) - lat) ** 2
    )
    return containing.sort_values("center_lonlat_dist").iloc[0]


def meters_between_utm(row_a: pd.Series, row_b: pd.Series) -> float:
    dx = float(row_a["utm_x_m"]) - float(row_b["utm_x_m"])
    dy = float(row_a["utm_y_m"]) - float(row_b["utm_y_m"])
    return float(np.sqrt(dx * dx + dy * dy))


def save_preprocess_panel(query_gray: np.ndarray, tile_gray: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(query_gray, cmap="gray")
    axes[0].set_title("UAV query preprocessed")
    axes[0].axis("off")

    axes[1].imshow(tile_gray, cmap="gray")
    axes[1].set_title("Satellite tile preprocessed")
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_keypoint_panel(query_gray: np.ndarray, tile_gray: np.ndarray, query_kp, tile_kp, out_path: Path, title: str) -> None:
    query_vis = cv2.drawKeypoints(
        query_gray,
        query_kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    tile_vis = cv2.drawKeypoints(
        tile_gray,
        tile_kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    query_vis = cv2.cvtColor(query_vis, cv2.COLOR_BGR2RGB)
    tile_vis = cv2.cvtColor(tile_vis, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(query_vis)
    axes[0].set_title(f"UAV ORB keypoints: {len(query_kp)}")
    axes[0].axis("off")

    axes[1].imshow(tile_vis)
    axes[1].set_title(f"Satellite ORB keypoints: {len(tile_kp)}")
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_matches_image(
    query_gray: np.ndarray,
    tile_gray: np.ndarray,
    query_kp,
    tile_kp,
    matches,
    out_path: Path,
    title: str,
    max_matches: int = 80,
) -> None:
    matches_to_draw = matches[:max_matches]
    vis = cv2.drawMatches(
        query_gray,
        query_kp,
        tile_gray,
        tile_kp,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.imshow(vis)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_satloc.yaml")
    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--query-token0-id", type=int, default=100)
    parser.add_argument("--query-frame-index", type=int, default=None)
    parser.add_argument("--variants", default="V1_luma,V2_clahe_luma,V4_bilateral_alt_clahe")
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

    uav_index_path = Path("outputs/satloc/metadata/uav_frames_index_enriched.csv")
    sat_index_path = Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")

    out_meta_dir = Path("outputs/satloc/metadata/s4a_orb_tile_retrieval/pair_debug")
    out_report_dir = Path("outputs/satloc/reports/s4a_orb_tile_retrieval/pair_debug")
    out_fig_dir = Path("outputs/satloc/figures/s4a_orb_tile_retrieval/pair_debug")
    ensure_dir(out_meta_dir)
    ensure_dir(out_report_dir)
    ensure_dir(out_fig_dir)

    uav_df = pd.read_csv(uav_index_path)
    sat_df = pd.read_csv(sat_index_path)
    sat_df = sat_df[sat_df["tile_exists"] == True].copy()

    query_row = find_query_row(
        uav_df=uav_df,
        sequence=args.sequence,
        token0_id=args.query_token0_id,
        frame_index=args.query_frame_index,
    )
    true_tile_row = find_debug_true_tile(
        sat_df=sat_df,
        lon=float(query_row["lon"]),
        lat=float(query_row["lat"]),
    )

    query_path = Path(str(query_row["image_path"]))
    tile_path = Path(str(true_tile_row["tile_path"]))

    query_bgr = read_bgr(query_path)
    tile_bgr = read_bgr(tile_path)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    rows = []

    run_id = f"{args.sequence}_token{int(query_row['token0_id']):04d}_tile{int(true_tile_row['tile_index']):05d}"

    print("\nS4A.1 pair match debug")
    print("----------------------")
    print(f"Query: {query_path}")
    print(f"Query lon/lat: {float(query_row['lon']):.8f}, {float(query_row['lat']):.8f}")
    print(f"Debug true/containing tile: {tile_path}")
    print(f"Tile lon/lat center: {float(true_tile_row['lon_center']):.8f}, {float(true_tile_row['lat_center']):.8f}")
    print(f"Variants: {variants}")

    for variant in variants:
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

        prefix = f"{run_id}_{variant}"

        preprocess_path = out_fig_dir / f"{prefix}_preprocess_panel.png"
        keypoint_path = out_fig_dir / f"{prefix}_keypoints.png"
        matches_path = out_fig_dir / f"{prefix}_good_matches.png"
        inliers_path = out_fig_dir / f"{prefix}_ransac_inliers.png"

        save_preprocess_panel(
            result.query_processed,
            result.tile_processed,
            preprocess_path,
            title=f"{variant} preprocessing",
        )

        save_keypoint_panel(
            result.query_processed,
            result.tile_processed,
            result.query_kp,
            result.tile_kp,
            keypoint_path,
            title=f"{variant} ORB keypoints",
        )

        save_matches_image(
            result.query_processed,
            result.tile_processed,
            result.query_kp,
            result.tile_kp,
            result.good_match_objects,
            matches_path,
            title=f"{variant} good matches: {result.good_matches}",
            max_matches=80,
        )

        inlier_matches = []
        if result.inlier_mask is not None:
            inlier_matches = [
                m for m, keep in zip(result.good_match_objects, result.inlier_mask)
                if bool(keep)
            ]

        save_matches_image(
            result.query_processed,
            result.tile_processed,
            result.query_kp,
            result.tile_kp,
            inlier_matches,
            inliers_path,
            title=f"{variant} RANSAC inliers: {result.ransac_inliers}",
            max_matches=80,
        )

        row = {
            "sequence": args.sequence,
            "query_token0_id": int(query_row["token0_id"]),
            "query_frame_index_in_sequence": int(query_row["frame_index_in_sequence"]),
            "query_filename": str(query_row["filename"]),
            "query_path": str(query_path),
            "query_lon": float(query_row["lon"]),
            "query_lat": float(query_row["lat"]),
            "tile_index": int(true_tile_row["tile_index"]),
            "tile_filename": str(true_tile_row["filename"]),
            "tile_path": str(tile_path),
            "tile_lon_center": float(true_tile_row["lon_center"]),
            "tile_lat_center": float(true_tile_row["lat_center"]),
            "variant": variant,
            "query_keypoints": result.query_keypoints,
            "tile_keypoints": result.tile_keypoints,
            "raw_matches": result.raw_matches,
            "good_matches": result.good_matches,
            "ransac_inliers": result.ransac_inliers,
            "inlier_ratio": result.inlier_ratio,
            "homography_success": result.homography_success,
            "score": result.score,
            "elapsed_ms": result.elapsed_ms,
            "preprocess_panel": str(preprocess_path),
            "keypoints_panel": str(keypoint_path),
            "good_matches_figure": str(matches_path),
            "ransac_inliers_figure": str(inliers_path),
        }
        rows.append(row)

        print(
            f"{variant:26s} | "
            f"q_kp={result.query_keypoints:4d} "
            f"t_kp={result.tile_keypoints:4d} "
            f"good={result.good_matches:3d} "
            f"inliers={result.ransac_inliers:3d} "
            f"ratio={result.inlier_ratio:.3f} "
            f"H={result.homography_success} "
            f"score={result.score:.2f} "
            f"time={result.elapsed_ms:.1f}ms"
        )

    result_df = pd.DataFrame(rows)
    csv_path = out_meta_dir / f"{run_id}_pair_match_results.csv"
    json_path = out_report_dir / f"{run_id}_pair_match_summary.json"

    result_df.to_csv(csv_path, index=False)

    summary = {
        "run_id": run_id,
        "query": {
            "sequence": args.sequence,
            "token0_id": int(query_row["token0_id"]),
            "frame_index_in_sequence": int(query_row["frame_index_in_sequence"]),
            "filename": str(query_row["filename"]),
            "path": str(query_path),
            "lon": float(query_row["lon"]),
            "lat": float(query_row["lat"]),
        },
        "debug_true_tile": {
            "tile_index": int(true_tile_row["tile_index"]),
            "filename": str(true_tile_row["filename"]),
            "path": str(tile_path),
            "lon_center": float(true_tile_row["lon_center"]),
            "lat_center": float(true_tile_row["lat_center"]),
        },
        "results_csv": str(csv_path),
        "variants": rows,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Figs: {out_fig_dir}")


if __name__ == "__main__":
    main()