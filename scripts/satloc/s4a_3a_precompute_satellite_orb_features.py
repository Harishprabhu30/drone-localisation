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


def read_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def keypoints_to_array(kps) -> np.ndarray:
    if not kps:
        return np.zeros((0, 7), dtype=np.float32)

    rows = []
    for kp in kps:
        rows.append([
            kp.pt[0],
            kp.pt[1],
            kp.size,
            kp.angle,
            kp.response,
            kp.octave,
            kp.class_id,
        ])
    return np.asarray(rows, dtype=np.float32)


def save_keypoint_overlay(tile_bgr, kps, out_path: Path, title: str) -> None:
    vis = cv2.drawKeypoints(
        tile_bgr,
        kps,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(vis)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_histogram(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    values = df["keypoints"].astype(float)

    if values.nunique() == 1:
        v = float(values.iloc[0])
        ax.bar([v], [len(values)], width=20)
        ax.set_xlim(v - 100, v + 100)
        ax.text(
            v,
            len(values),
            f"all {len(values)} tiles = {int(v)} keypoints",
            ha="center",
            va="bottom",
        )
    else:
        ax.hist(values, bins=40)

    ax.set_xlabel("ORB keypoints per satellite tile")
    ax.set_ylabel("Tile count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="V2_clahe_luma")
    parser.add_argument("--nfeatures", type=int, default=1200)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-tiles", default="415,418,3545,3544,3554")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-size", type=int, default=8)
    parser.add_argument("--alt-clahe-clip-limit", type=float, default=1.0)
    parser.add_argument("--alt-clahe-tile-size", type=int, default=8)
    parser.add_argument("--bilateral-d", type=int, default=13)
    parser.add_argument("--bilateral-sigma-color", type=float, default=30)
    parser.add_argument("--bilateral-sigma-space", type=float, default=55)
    args = parser.parse_args()

    sat_index_path = Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")
    sat_df = pd.read_csv(sat_index_path)
    sat_df = sat_df[sat_df["tile_exists"] == True].copy()
    sat_df = sat_df.sort_values("tile_index").reset_index(drop=True)

    if args.max_tiles is not None:
        sat_df = sat_df.head(args.max_tiles).copy()

    cache_name = f"{args.variant}_nf{args.nfeatures}"
    cache_dir = Path("outputs/satloc/cache/s4a_orb_tile_features") / cache_name
    out_meta_dir = Path("outputs/satloc/metadata/s4a_orb_tile_retrieval/feature_cache")
    out_report_dir = Path("outputs/satloc/reports/s4a_orb_tile_retrieval/feature_cache")
    out_fig_dir = Path("outputs/satloc/figures/s4a_orb_tile_retrieval/feature_cache") / cache_name

    ensure_dir(cache_dir)
    ensure_dir(out_meta_dir)
    ensure_dir(out_report_dir)
    ensure_dir(out_fig_dir)

    sample_tiles = set()
    if args.sample_tiles.strip():
        sample_tiles = {int(x.strip()) for x in args.sample_tiles.split(",") if x.strip()}

    print("\nS4A.3A precompute satellite ORB features")
    print("----------------------------------------")
    print(f"Variant: {args.variant}")
    print(f"nfeatures: {args.nfeatures}")
    print(f"Tiles to process: {len(sat_df)}")
    print(f"Cache dir: {cache_dir}")

    rows = []
    start_all = perf_counter()

    for i, (_, row) in enumerate(sat_df.iterrows(), start=1):
        tile_index = int(row["tile_index"])
        tile_path = Path(str(row["tile_path"]))
        feature_path = cache_dir / f"tile_{tile_index:05d}.npz"

        if feature_path.exists() and not args.overwrite:
            try:
                data = np.load(feature_path)
                kp_count = int(data["keypoints"].shape[0])
                des_shape = tuple(data["descriptors"].shape) if "descriptors" in data else None
                rows.append({
                    "tile_index": tile_index,
                    "filename": str(row["filename"]),
                    "tile_path": str(tile_path),
                    "feature_path": str(feature_path),
                    "read_ok": True,
                    "computed": False,
                    "keypoints": kp_count,
                    "descriptor_rows": des_shape[0] if des_shape else 0,
                    "descriptor_cols": des_shape[1] if des_shape and len(des_shape) > 1 else 0,
                    "elapsed_ms": 0.0,
                })
                continue
            except Exception:
                pass

        start = perf_counter()
        img = read_bgr(tile_path)

        if img is None:
            rows.append({
                "tile_index": tile_index,
                "filename": str(row["filename"]),
                "tile_path": str(tile_path),
                "feature_path": str(feature_path),
                "read_ok": False,
                "computed": False,
                "keypoints": 0,
                "descriptor_rows": 0,
                "descriptor_cols": 0,
                "elapsed_ms": 0.0,
            })
            continue

        processed = preprocess_bgr(
            img,
            variant=args.variant,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_size=args.clahe_tile_size,
            alt_clahe_clip_limit=args.alt_clahe_clip_limit,
            alt_clahe_tile_size=args.alt_clahe_tile_size,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
        )

        kps, des = detect_orb(processed, nfeatures=args.nfeatures)
        kp_array = keypoints_to_array(kps)

        if des is None:
            des = np.zeros((0, 32), dtype=np.uint8)

        np.savez_compressed(
            feature_path,
            tile_index=np.asarray([tile_index], dtype=np.int32),
            keypoints=kp_array,
            descriptors=des,
        )

        elapsed_ms = float((perf_counter() - start) * 1000.0)

        if tile_index in sample_tiles:
            overlay_path = out_fig_dir / f"tile_{tile_index:05d}_orb_keypoints.png"
            save_keypoint_overlay(
                img,
                kps,
                overlay_path,
                title=f"Tile {tile_index} | {args.variant} | kp={len(kps)}",
            )

        rows.append({
            "tile_index": tile_index,
            "filename": str(row["filename"]),
            "tile_path": str(tile_path),
            "feature_path": str(feature_path),
            "read_ok": True,
            "computed": True,
            "keypoints": int(len(kps)),
            "descriptor_rows": int(des.shape[0]),
            "descriptor_cols": int(des.shape[1]) if len(des.shape) > 1 else 0,
            "elapsed_ms": elapsed_ms,
        })

        if i % 250 == 0 or i == len(sat_df):
            print(f"  processed {i}/{len(sat_df)} tiles")

    manifest_df = pd.DataFrame(rows)

    manifest_path = out_meta_dir / f"{cache_name}_manifest.csv"
    summary_path = out_report_dir / f"{cache_name}_summary.json"
    hist_path = out_fig_dir / f"{cache_name}_keypoint_histogram.png"

    manifest_df.to_csv(manifest_path, index=False)

    ok_df = manifest_df[manifest_df["read_ok"] == True].copy()
    save_histogram(
        ok_df,
        hist_path,
        title=f"Satellite ORB keypoint distribution | {cache_name}",
    )

    summary = {
        "variant": args.variant,
        "nfeatures": int(args.nfeatures),
        "tiles_requested": int(len(sat_df)),
        "tiles_read_ok": int(manifest_df["read_ok"].sum()),
        "tiles_computed_now": int(manifest_df["computed"].sum()),
        "keypoints_median": float(ok_df["keypoints"].median()) if len(ok_df) else None,
        "keypoints_mean": float(ok_df["keypoints"].mean()) if len(ok_df) else None,
        "keypoints_min": int(ok_df["keypoints"].min()) if len(ok_df) else None,
        "keypoints_max": int(ok_df["keypoints"].max()) if len(ok_df) else None,
        "elapsed_total_s": float(perf_counter() - start_all),
        "manifest": str(manifest_path),
        "cache_dir": str(cache_dir),
        "histogram": str(hist_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Histogram: {hist_path}")
    print(f"Cache: {cache_dir}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()