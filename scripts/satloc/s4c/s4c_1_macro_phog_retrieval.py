#!/usr/bin/env python3
"""
S4C.1 — Macro-contour PHOG full-map retrieval for SatLoc.

Pipeline:
  UAV image / satellite tile
  -> macro-contour extraction from S4C.0
  -> PHOG descriptor over final contour canvas
  -> cosine retrieval over full satellite tile database
  -> evaluate only after ranking

Important:
  UAV filename lon/lat is used only for evaluation/debug after ranking.
  It is not used for retrieval scoring.

code using v1:
export PYTHONPATH=$PWD/src

python scripts/satloc/s4c/s4c_1_macro_phog_retrieval.py \
  --sequence traj01 \
  --tokens 1,100,166 \
  --preprocess luma \
  --resize-size 512 \
  --edge-method sobel \
  --blur-ksize 3 \
  --threshold-mode percentile \
  --threshold-percentile 65
  --close-ksize 3 \
  --open-ksize 1 \
  --min-component-area 65 \
  --phog-levels 3 \
  --phog-bins 9 \
  --top-k 10 \
  --rebuild-cache \
  --save-topk-panels

Compact result
--------------
 token  top1_tile_id  top1_center_error_m  best_topk_center_error_m  first_rank_under_20m  first_rank_under_40m  first_rank_under_60m first_rank_contains_gt  query_runtime_s
     1          8237          2672.693485                 15.072437                   5.0                     5                     5                   None        26.434893
    40          2171            12.433470                 12.433470                   1.0                     1                     1                   None        29.599457
    50          1602          1960.460295                347.521599                 264.0                   160                   160                   None        26.357529
    58          1964          1516.572667                793.043228                2450.0                  2450                   584                   None        25.310894
    60          1964          1519.625767                 17.847211                   4.0                     4                     4                   None        25.177791
    67          1587          1448.174892                251.319680                2209.0                   101                    66                   None        24.823879
    74          1839          1542.254162               1431.137186                 894.0                   894                   798                   None        26.411413
    79          3064           958.955644                909.676344                 325.0                    23                    23                   None        25.289198
    90          7740          3211.489396                648.550079                3126.0                    14                    14                   None        29.350917
   100           540            20.692326                 20.692326                 216.0                     1                     1                   None        28.656506
   107          1194          1119.714447                998.983761                   NaN                    20                    20                   None        26.360426
   117          3000          1386.260555               1176.869387                  65.0                    65                    65                   None        31.657995
   129          1034           211.917830                 16.664889                   2.0                     2                     2                   None        27.355195
   139           527            19.472323                 19.472323                   1.0                     1                     1                   None        27.078628
   166           129           730.387070                 25.740995                   NaN                     4                     4                   None        31.042669
   259          3774            19.850081                 19.850081                   1.0                     1                     1                   None        29.560667
   269          4149             7.532041                  7.532041                   1.0                     1                     1                   None        31.336005
   276          2532           595.654983                  2.081395                   4.0                     4                     4                   None        29.528107
   288          7461          2272.063891                402.701150                 679.0                   679                   679                   None        25.984071
   300          6831          2020.897434                979.089597                1555.0                  1555                  1131                   None        23.431355
   310          6847          2555.297600                632.387387                  30.0                    30                    30                   None        22.708201
   326           659          1665.255416                159.901121                 825.0                   466                   369                   None        21.629959
   336          8235          2980.891697                752.725381                 581.0                   405                   300                   None        21.981369
   350          8235          2817.234996                832.458776                3155.0                   714                   714                   None        22.403850
   366          2500          1704.467569                987.811945                2410.0                  1014                   944                   None        23.618320
   387          6847          1889.025254               1007.583599                1097.0                   437                   437                   None        22.601507
   405          6846          1638.756170               1113.194318                1361.0                   838                   838                   None        21.913821
   421          2876          2143.358909               1563.578333                 355.0                    89                    89                   None        22.238111
   434          8236          1868.208221                879.790053                 221.0                    60                    60                   None        23.433779
   450          6804           352.336942                352.336942                 615.0                   615                   615                   None        22.725359
   474          2876          2439.226688               1607.012886                 983.0                   983                   761                   None        27.717791
   482          8332           909.027024                 13.126118                   2.0                     2                     2                   None        22.437562
   494          5316            28.829000                 28.829000                 392.0                     1                     1                   None        21.296901
   503          8360          1816.101708               1399.570607                 212.0                   212                   212                   None        28.821981
   516          4316            11.410698                 11.410698                   1.0                     1                     1                   None        23.744512
   533          6206           893.291087                 26.187880                1533.0                     2                     2                   None        24.340952
   546          3071           170.559536                170.559536                  45.0                    45                    45                   None        23.533366
   564          3362          1606.134561                105.492793                1111.0                   176                   176                   None        22.219598
   573          4380          2179.301484               2086.653268                8113.0                  7807                  7338                   None        24.478721
   577           368          1949.416349                905.186747                8349.0                  7548                  6906                   None        23.939949
   591          4556           505.197620                505.197620                6418.0                  5735                  3798                   None        24.344103
   614          8237          2389.238781               1017.644679                3312.0                  1618                  1011                   None        24.470645
   631          8234          2416.087474                948.391569                2396.0                  1275                   663                   None        22.706193
   653          5766          1415.817346               1262.799718                1027.0                   918                   356                   None        23.590078
   662          3125          1819.329938                542.759092                2754.0                  2289                   372                   None        23.024790
   679          8235          2266.738576               1362.674400                 385.0                   137                   137                   None        24.765376
   694          2672           725.418547                514.915503                  63.0                    48                    48                   None        24.960010
   710          8236          2253.846135                960.894771                2354.0                    50                    50                   None        25.324956
   731          5542            25.950245                 25.055578                2030.0                     1                     1                   None        24.221819
   746          5663            28.348883                 28.348883                   NaN                     1                     1                   None        23.672120
   760          4501          1163.469197                485.043430                  16.0                    16                    16                   None        24.241659
   768           530          1303.295147                152.301247                 216.0                   216                   216                   None        23.282032
   781          3001          1172.919615                641.148618                1579.0                   730                   730                   None        23.175802
   794          8112          3001.155104                622.958055                 720.0                   720                   720                   None        24.756701
   808          7357          2826.261070                517.423628                1848.0                   154                   154                   None        22.357224
   820          7971          2621.531889                450.292636                2896.0                   210                   210                   None        21.656064
   833          7232          2930.854928                511.820218                4571.0                  1860                  1852                   None        22.944869
   844          1318          1286.916178                430.938164                2676.0                  1877                  1670                   None        21.857272
   874          1410            31.147777                 18.228113                   6.0                     1                     1                   None        21.707204
   886           909           254.603822                 18.575541                   2.0                     2                     2                   None        23.274417
   905          2042            15.897900                 15.897900                   1.0                     1                     1                   None        23.424653
   914          2544            77.205585                 77.205585                  74.0                    11                    11                   None        24.685561
   927           381          1449.877483               1024.812740                 673.0                    35                    35                   None        21.390747
   937          3393           849.889017                218.033926                 158.0                   158                   158                   None        22.194430
   946          3793            22.599657                 22.599657                  80.0                     1                     1                   None        22.266940
   952          8235          2627.966138                453.277549                 113.0                   104                   104                   None        31.580567
   963          2627          1502.965608                763.015777                 265.0                   265                   265                   None        29.016529
   971          4016           952.833284                928.763785                   NaN                   138                   138                   None        28.306795
   990          3500          1638.154026                646.560281                   NaN                    13                    13                   None        27.008793
  1003          5171            24.362538                 24.362538                3267.0                     1                     1                   None        29.544205
  1015          8189          1184.921591                 23.228396                1255.0                     2                     2                   None        25.028759
  1024          7098          1985.394909                335.754813                2107.0                  1741                  1741                   None        23.275590
  1034          2751          1589.595600                844.140509                 108.0                   108                   108                   None        23.816644
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Import S4C.0 helpers dynamically from the same folder
# ------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
S4C0_PATH = THIS_DIR / "s4c_0_macrocontour_preflight.py"

if not S4C0_PATH.exists():
    raise FileNotFoundError(
        f"Missing S4C.0 helper script: {S4C0_PATH}\n"
        "Run/create scripts/satloc/s4c/s4c_0_macrocontour_preflight.py first."
    )

spec = importlib.util.spec_from_file_location("s4c0_helpers", S4C0_PATH)
s4c0 = importlib.util.module_from_spec(spec)
sys.modules["s4c0_helpers"] = s4c0
assert spec.loader is not None
spec.loader.exec_module(s4c0)


OUT_ROOT = Path("outputs/satloc")
OUT_FIG_DIR = OUT_ROOT / "figures/s4c_macrocontour_phog_chamfer/s4c1_phog_retrieval"
OUT_META_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/s4c1_phog_retrieval"
OUT_REPORT_DIR = OUT_ROOT / "reports/s4c_macrocontour_phog_chamfer/s4c1_phog_retrieval"
OUT_CACHE_DIR = OUT_ROOT / "metadata/s4c_macrocontour_phog_chamfer/cache"


# ------------------------------------------------------------
# General helpers
# ------------------------------------------------------------

def parse_tokens(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def safe_slug(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def settings_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "preprocess": args.preprocess,
        "resize_size": args.resize_size,
        "edge_method": args.edge_method,
        "blur_ksize": args.blur_ksize,
        "threshold_mode": args.threshold_mode,
        "threshold_percentile": args.threshold_percentile,
        "close_ksize": args.close_ksize,
        "open_ksize": args.open_ksize,
        "min_component_area": args.min_component_area,
        "phog_levels": args.phog_levels,
        "phog_bins": args.phog_bins,
        "phog_norm": "l2_global",
        "phog_angle": "unsigned_0_180",
    }


def settings_slug(args: argparse.Namespace) -> str:
    return (
        f"macro_{args.preprocess}_{args.edge_method}"
        f"_b{args.blur_ksize}"
        f"_c{args.close_ksize}"
        f"_o{args.open_ksize}"
        f"_area{args.min_component_area}"
        f"_r{args.resize_size}"
        f"_phogL{args.phog_levels}"
        f"B{args.phog_bins}"
    )


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec * 0.0
    return vec / norm


# ------------------------------------------------------------
# PHOG descriptor
# ------------------------------------------------------------

def phog_descriptor(
    contour_canvas: np.ndarray,
    levels: int = 3,
    bins: int = 9,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    PHOG over a binary/macro contour canvas.

    levels=3 means:
      level 0: 1x1
      level 1: 2x2
      level 2: 4x4
      level 3: 8x8

    For each cell:
      - compute Sobel orientation on contour canvas
      - histogram unsigned gradient orientation [0, pi)
      - weight by gradient magnitude
      - L1 normalize per cell
    Final descriptor:
      - concatenate all cells
      - L2 normalize globally
    """
    if contour_canvas.ndim != 2:
        raise ValueError("PHOG expects a single-channel contour canvas.")

    img = (contour_canvas.astype(np.float32) / 255.0)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    ang = np.arctan2(gy, gx)

    # Unsigned orientation: [0, pi)
    ang = np.mod(ang, np.pi)

    h, w = img.shape[:2]
    parts: list[np.ndarray] = []

    for level in range(levels + 1):
        grid = 2 ** level
        cell_h = h // grid
        cell_w = w // grid

        for gy_idx in range(grid):
            for gx_idx in range(grid):
                y0 = gy_idx * cell_h
                x0 = gx_idx * cell_w
                y1 = h if gy_idx == grid - 1 else (gy_idx + 1) * cell_h
                x1 = w if gx_idx == grid - 1 else (gx_idx + 1) * cell_w

                mag_cell = mag[y0:y1, x0:x1]
                ang_cell = ang[y0:y1, x0:x1]
                contour_cell = contour_canvas[y0:y1, x0:x1]

                # Use only active contour-gradient pixels.
                mask = (contour_cell > 0) & (mag_cell > eps)

                if not np.any(mask):
                    hist = np.zeros((bins,), dtype=np.float32)
                else:
                    hist, _ = np.histogram(
                        ang_cell[mask],
                        bins=bins,
                        range=(0.0, np.pi),
                        weights=mag_cell[mask],
                    )
                    hist = hist.astype(np.float32)
                    hist_sum = float(hist.sum())
                    if hist_sum > eps:
                        hist = hist / hist_sum

                parts.append(hist)

    desc = np.concatenate(parts).astype(np.float32)
    return l2_normalize(desc)


def compute_macro_and_phog(
    image_path: Path,
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray]:
    macro = s4c0.macro_contour_pipeline(
        image_path=image_path,
        resize_size=args.resize_size,
        preprocess=args.preprocess,
        edge_method=args.edge_method,
        blur_ksize=args.blur_ksize,
        threshold_mode=args.threshold_mode,
        threshold_percentile=args.threshold_percentile,
        close_ksize=args.close_ksize,
        open_ksize=args.open_ksize,
        min_component_area=args.min_component_area,
    )

    desc = phog_descriptor(
        macro.contour_canvas,
        levels=args.phog_levels,
        bins=args.phog_bins,
    )

    return macro, desc


# ------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------

def candidate_contains_gt(
    sat_row: pd.Series,
    sat_df: pd.DataFrame,
    uav_lon: float,
    uav_lat: float,
) -> bool:
    lon_min, lon_max, lat_min, lat_max = s4c0.find_bbox_cols(sat_df)
    if not all(c is not None for c in [lon_min, lon_max, lat_min, lat_max]):
        return False

    a = s4c0.safe_float(sat_row[lon_min])
    b = s4c0.safe_float(sat_row[lon_max])
    c = s4c0.safe_float(sat_row[lat_min])
    d = s4c0.safe_float(sat_row[lat_max])

    if a is None or b is None or c is None or d is None:
        return False

    lon_lo, lon_hi = min(a, b), max(a, b)
    lat_lo, lat_hi = min(c, d), max(c, d)

    return bool(lon_lo <= uav_lon <= lon_hi and lat_lo <= uav_lat <= lat_hi)


def first_rank_under(df: pd.DataFrame, threshold_m: float) -> int | None:
    valid = df[np.isfinite(df["center_error_m"]) & (df["center_error_m"] <= threshold_m)]
    if len(valid) == 0:
        return None
    return int(valid["rank"].min())


def first_true_rank(df: pd.DataFrame, col: str) -> int | None:
    if col not in df.columns:
        return None
    valid = df[df[col].astype(bool)]
    if len(valid) == 0:
        return None
    return int(valid["rank"].min())


# ------------------------------------------------------------
# Satellite PHOG cache
# ------------------------------------------------------------

def build_filename_index(sequence: str) -> tuple[dict[str, Path], list[Path], list[Path]]:
    fallback_uav_dirs = [
        Path("data/raw/satloc/part_1/UAV Data") / sequence,
        Path("data/raw/satloc/part_1/UAV Data"),
    ]

    fallback_sat_dirs = [
        Path("data/raw/satloc/part_1/Satellite Data/sat_image_ref"),
        Path("data/raw/satloc/part_1/Satellite Data"),
    ]

    filename_index = s4c0.build_filename_index(fallback_uav_dirs + fallback_sat_dirs)
    return filename_index, fallback_uav_dirs, fallback_sat_dirs


def cache_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    slug = settings_slug(args)
    cache_npz = OUT_CACHE_DIR / f"s4c1_satellite_phog_cache_{slug}.npz"
    cache_manifest = OUT_META_DIR / f"s4c1_satellite_phog_cache_manifest_{slug}.csv"
    return cache_npz, cache_manifest


def build_satellite_cache(
    sat_df: pd.DataFrame,
    filename_index: dict[str, Path],
    fallback_sat_dirs: list[Path],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    OUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)

    cache_npz, cache_manifest = cache_paths(args)

    descriptors: list[np.ndarray] = []
    row_positions: list[int] = []
    tile_ids: list[int] = []
    paths: list[str] = []
    manifest_rows: list[dict[str, Any]] = []

    t0 = time.perf_counter()

    for row_pos, (_, row) in enumerate(sat_df.iterrows()):
        image_path = s4c0.get_row_path(
            row,
            sat_df,
            filename_index,
            fallback_sat_dirs,
            kind="sat",
        )

        if image_path is None:
            continue

        tile_id = s4c0.get_tile_id(row, sat_df)
        if tile_id is None:
            tile_id = row_pos

        try:
            macro, desc = compute_macro_and_phog(image_path, args)
        except Exception as exc:
            print(f"[WARN] cache skip row_pos={row_pos} tile_id={tile_id}: {exc}")
            continue

        descriptors.append(desc)
        row_positions.append(row_pos)
        tile_ids.append(int(tile_id))
        paths.append(str(image_path))

        mrow = {
            "row_pos": row_pos,
            "tile_id": int(tile_id),
            "image_path": str(image_path),
        }
        mrow.update(macro.stats)
        manifest_rows.append(mrow)

        if len(descriptors) % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[cache] processed {len(descriptors)} satellite tiles in {elapsed:.1f}s")

    if len(descriptors) == 0:
        raise RuntimeError("No satellite PHOG descriptors were built. Check satellite image paths.")

    desc_arr = np.vstack(descriptors).astype(np.float32)
    row_pos_arr = np.array(row_positions, dtype=np.int32)
    tile_id_arr = np.array(tile_ids, dtype=np.int32)
    path_arr = np.array(paths)

    settings_json = json.dumps(settings_dict(args), sort_keys=True)

    np.savez_compressed(
        cache_npz,
        descriptors=desc_arr,
        row_positions=row_pos_arr,
        tile_ids=tile_id_arr,
        paths=path_arr,
        settings_json=np.array(settings_json),
    )

    pd.DataFrame(manifest_rows).to_csv(cache_manifest, index=False)

    print(f"[cache] saved: {cache_npz}")
    print(f"[cache] manifest: {cache_manifest}")

    return desc_arr, row_pos_arr, tile_id_arr, path_arr


def load_or_build_satellite_cache(
    sat_df: pd.DataFrame,
    filename_index: dict[str, Path],
    fallback_sat_dirs: list[Path],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_npz, _ = cache_paths(args)
    expected_settings = json.dumps(settings_dict(args), sort_keys=True)

    if cache_npz.exists() and not args.rebuild_cache:
        data = np.load(cache_npz, allow_pickle=False)

        stored_settings = str(data["settings_json"].item())
        if stored_settings == expected_settings:
            print(f"[cache] loaded: {cache_npz}")
            return (
                data["descriptors"].astype(np.float32),
                data["row_positions"].astype(np.int32),
                data["tile_ids"].astype(np.int32),
                data["paths"],
            )

        print("[cache] existing cache settings differ, rebuilding...")

    return build_satellite_cache(sat_df, filename_index, fallback_sat_dirs, args)


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def render_score_plot(
    ranked_df: pd.DataFrame,
    token: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    plot_df = ranked_df.head(min(len(ranked_df), 500)).copy()
    ax.plot(plot_df["rank"], plot_df["score_cosine"])

    ax.set_title(f"S4C.1 PHOG retrieval score by rank — token {token}")
    ax.set_xlabel("rank")
    ax.set_ylabel("cosine similarity")
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_topk_panel(
    token: int,
    q_macro: Any,
    gt_macro: Any,
    gt_label: str,
    ranked_top: pd.DataFrame,
    sat_df: pd.DataFrame,
    filename_index: dict[str, Path],
    fallback_sat_dirs: list[Path],
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    top_k = len(ranked_top)
    if top_k == 0:
        return

    fig, axes = plt.subplots(
        3,
        top_k,
        figsize=(3.0 * top_k, 9.0),
        squeeze=False,
    )

    # Top row: query and GT context.
    for c in range(top_k):
        axes[0, c].axis("off")

    axes[0, 0].imshow(q_macro.rgb)
    axes[0, 0].set_title("UAV query RGB", fontsize=9)
    axes[0, 0].axis("off")

    if top_k > 1:
        axes[0, 1].imshow(q_macro.contour_canvas, cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title("UAV contour", fontsize=9)
        axes[0, 1].axis("off")

    if top_k > 2:
        axes[0, 2].imshow(gt_macro.rgb)
        axes[0, 2].set_title("GT/nearest RGB", fontsize=9)
        axes[0, 2].axis("off")

    if top_k > 3:
        axes[0, 3].imshow(gt_macro.contour_canvas, cmap="gray", vmin=0, vmax=255)
        axes[0, 3].set_title(gt_label, fontsize=8)
        axes[0, 3].axis("off")

    # Candidate rows.
    for col, (_, r) in enumerate(ranked_top.iterrows()):
        sat_row = sat_df.iloc[int(r["row_pos"])]
        cand_path = s4c0.get_row_path(
            sat_row,
            sat_df,
            filename_index,
            fallback_sat_dirs,
            kind="sat",
        )

        if cand_path is None:
            axes[1, col].axis("off")
            axes[2, col].axis("off")
            continue

        try:
            cand_macro, _ = compute_macro_and_phog(cand_path, args)
        except Exception:
            axes[1, col].axis("off")
            axes[2, col].axis("off")
            continue

        rank = int(r["rank"])
        tile_id = int(r["tile_id"])
        score = float(r["score_cosine"])
        err = float(r["center_error_m"]) if np.isfinite(r["center_error_m"]) else float("nan")
        under40 = "≤40m" if np.isfinite(err) and err <= 40 else ""

        title = (
            f"R{rank} tile {tile_id}\n"
            f"s={score:.3f} err={err:.1f}m {under40}"
        )

        axes[1, col].imshow(cand_macro.rgb)
        axes[1, col].set_title(title, fontsize=8)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        axes[2, col].imshow(cand_macro.contour_canvas, cmap="gray", vmin=0, vmax=255)
        axes[2, col].set_title(
            f"macro={cand_macro.stats['cleaned_density']:.3f}",
            fontsize=8,
        )
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

    fig.suptitle(
        f"S4C.1 Macro-contour PHOG retrieval — token {token}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ------------------------------------------------------------
# Query retrieval
# ------------------------------------------------------------

def prepare_uav_df(uav_df: pd.DataFrame, sequence: str) -> pd.DataFrame:
    seq_col = s4c0.find_col(uav_df, ["sequence", "seq", "trajectory", "traj"], required=False)
    if seq_col is not None:
        uav_df = uav_df[uav_df[seq_col].astype(str) == sequence].copy()
    else:
        uav_df = uav_df.copy()

    tokens = []
    for _, row in uav_df.iterrows():
        tokens.append(s4c0.get_uav_token(row, uav_df))

    uav_df["_s4c_token"] = tokens
    return uav_df


def run_query(
    token: int,
    uav_df: pd.DataFrame,
    sat_df: pd.DataFrame,
    sat_desc: np.ndarray,
    sat_row_positions: np.ndarray,
    sat_tile_ids: np.ndarray,
    filename_index: dict[str, Path],
    fallback_uav_dirs: list[Path],
    fallback_sat_dirs: list[Path],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    match = uav_df[uav_df["_s4c_token"] == token]
    if len(match) == 0:
        print(f"[WARN] token {token}: UAV row not found")
        return None

    uav_row = match.iloc[0]

    uav_lon, uav_lat = s4c0.get_lon_lat(uav_row, uav_df)
    if uav_lon is None or uav_lat is None:
        print(f"[WARN] token {token}: could not parse UAV lon/lat for evaluation")
        return None

    uav_path = s4c0.get_row_path(
        uav_row,
        uav_df,
        filename_index,
        fallback_uav_dirs,
        kind="uav",
    )

    if uav_path is None:
        print(f"[WARN] token {token}: UAV image path not found")
        return None

    print(f"[query] token {token}: {uav_path}")

    q_t0 = time.perf_counter()
    q_macro, q_desc = compute_macro_and_phog(uav_path, args)

    # Cosine similarity because all descriptors are L2 normalized.
    scores = sat_desc @ q_desc
    l2_dist = np.linalg.norm(sat_desc - q_desc[None, :], axis=1)

    order = np.argsort(-scores)

    gt_row, gt_method, gt_error_m = s4c0.select_gt_or_nearest_tile(
        sat_df,
        float(uav_lon),
        float(uav_lat),
    )
    gt_tile_id = s4c0.get_tile_id(gt_row, sat_df)
    gt_path = s4c0.get_row_path(
        gt_row,
        sat_df,
        filename_index,
        fallback_sat_dirs,
        kind="sat",
    )

    gt_macro = None
    if gt_path is not None:
        gt_macro, _ = compute_macro_and_phog(gt_path, args)

    ranked_rows: list[dict[str, Any]] = []

    for rank_idx, cache_idx in enumerate(order, start=1):
        row_pos = int(sat_row_positions[cache_idx])
        sat_row = sat_df.iloc[row_pos]
        tile_id = int(sat_tile_ids[cache_idx])

        center_error_m = s4c0.tile_center_error(
            sat_row,
            sat_df,
            float(uav_lon),
            float(uav_lat),
        )

        contains_gt = candidate_contains_gt(
            sat_row,
            sat_df,
            float(uav_lon),
            float(uav_lat),
        )

        ranked_rows.append(
            {
                "sequence": args.sequence,
                "token": token,
                "rank": rank_idx,
                "row_pos": row_pos,
                "tile_id": tile_id,
                "score_cosine": float(scores[cache_idx]),
                "l2_distance": float(l2_dist[cache_idx]),
                "center_error_m": float(center_error_m),
                "contains_gt": bool(contains_gt),
                "is_gt_nearest_tile": bool(gt_tile_id is not None and tile_id == int(gt_tile_id)),
                "under_20m": bool(np.isfinite(center_error_m) and center_error_m <= 20.0),
                "under_40m": bool(np.isfinite(center_error_m) and center_error_m <= 40.0),
                "under_60m": bool(np.isfinite(center_error_m) and center_error_m <= 60.0),
                "gt_tile_id": gt_tile_id,
                "gt_selection_method": gt_method,
                "gt_center_error_m": float(gt_error_m),
                "uav_lon": float(uav_lon),
                "uav_lat": float(uav_lat),
            }
        )

    query_runtime_s = time.perf_counter() - q_t0
    ranked_df = pd.DataFrame(ranked_rows)

    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    slug = settings_slug(args)

    ranked_csv = OUT_META_DIR / f"s4c1_token{token:04d}_{slug}_ranked.csv"
    ranked_df.to_csv(ranked_csv, index=False)

    score_plot = OUT_FIG_DIR / f"s4c1_token{token:04d}_{slug}_score_by_rank.png"
    render_score_plot(ranked_df, token, score_plot)

    topk_panel = None
    if args.save_topk_panels and gt_macro is not None:
        topk_panel = OUT_FIG_DIR / f"s4c1_token{token:04d}_{slug}_top{args.top_k}_panel.png"
        gt_label = f"GT/nearest tile {gt_tile_id}\n{gt_method}, err={gt_error_m:.1f}m"
        render_topk_panel(
            token=token,
            q_macro=q_macro,
            gt_macro=gt_macro,
            gt_label=gt_label,
            ranked_top=ranked_df.head(args.top_k),
            sat_df=sat_df,
            filename_index=filename_index,
            fallback_sat_dirs=fallback_sat_dirs,
            args=args,
            out_path=topk_panel,
        )

    top1 = ranked_df.iloc[0]
    topk_df = ranked_df.head(args.top_k)

    summary = {
        "sequence": args.sequence,
        "token": token,
        "uav_image_path": str(uav_path),
        "ranked_csv": str(ranked_csv),
        "score_plot": str(score_plot),
        "topk_panel": str(topk_panel) if topk_panel is not None else None,
        "query_runtime_s": query_runtime_s,
        "top1_tile_id": int(top1["tile_id"]),
        "top1_score_cosine": float(top1["score_cosine"]),
        "top1_center_error_m": float(top1["center_error_m"]),
        "best_topk_center_error_m": float(topk_df["center_error_m"].min()),
        "first_rank_contains_gt": first_true_rank(ranked_df, "contains_gt"),
        "first_rank_gt_nearest_tile": first_true_rank(ranked_df, "is_gt_nearest_tile"),
        "first_rank_under_20m": first_rank_under(ranked_df, 20.0),
        "first_rank_under_40m": first_rank_under(ranked_df, 40.0),
        "first_rank_under_60m": first_rank_under(ranked_df, 60.0),
        "recall_at_1_contains_gt": bool(ranked_df.head(1)["contains_gt"].any()),
        "recall_at_5_contains_gt": bool(ranked_df.head(5)["contains_gt"].any()),
        "recall_at_10_contains_gt": bool(ranked_df.head(10)["contains_gt"].any()),
        "any_topk_under_20m": bool((topk_df["center_error_m"] <= 20.0).any()),
        "any_topk_under_40m": bool((topk_df["center_error_m"] <= 40.0).any()),
        "any_topk_under_60m": bool((topk_df["center_error_m"] <= 60.0).any()),
        "gt_tile_id": gt_tile_id,
        "gt_selection_method": gt_method,
        "gt_center_error_m": float(gt_error_m),
        "query_macro_cleaned_density": float(q_macro.stats["cleaned_density"]),
        "query_macro_contour_density": float(q_macro.stats["contour_density"]),
    }

    print(
        f"[OK] token {token}: "
        f"top1 tile={summary['top1_tile_id']} "
        f"err={summary['top1_center_error_m']:.1f}m "
        f"best@{args.top_k}={summary['best_topk_center_error_m']:.1f}m "
        f"first<=40m={summary['first_rank_under_40m']} "
        f"time={query_runtime_s:.2f}s"
    )

    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence", default="traj01")
    parser.add_argument("--tokens", default="1,100,166")
    parser.add_argument("--uav-index", default="outputs/satloc/metadata/uav_frames_index_enriched.csv")
    parser.add_argument("--sat-index", default="outputs/satloc/metadata/satellite_tiles_index_enriched.csv")

    # Macro settings: default = your chosen S4C.0 light macro-contour setting.
    parser.add_argument("--preprocess", default="luma", choices=["gray", "luma", "clahe_luma"])
    parser.add_argument("--resize-size", type=int, default=512)
    parser.add_argument("--edge-method", default="sobel", choices=["sobel", "canny"])
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--threshold-mode", default="otsu", choices=["otsu", "percentile"])
    parser.add_argument("--threshold-percentile", type=float, default=75.0)
    parser.add_argument("--close-ksize", type=int, default=3)
    parser.add_argument("--open-ksize", type=int, default=1)
    parser.add_argument("--min-component-area", type=int, default=65)

    # PHOG settings.
    parser.add_argument("--phog-levels", type=int, default=3)
    parser.add_argument("--phog-bins", type=int, default=9)

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--save-topk-panels", action="store_true")

    args = parser.parse_args()

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tokens = parse_tokens(args.tokens)

    uav_index = Path(args.uav_index)
    sat_index = Path(args.sat_index)

    if not uav_index.exists():
        raise FileNotFoundError(f"Missing UAV index: {uav_index}")
    if not sat_index.exists():
        raise FileNotFoundError(f"Missing satellite index: {sat_index}")

    uav_df = pd.read_csv(uav_index)
    sat_df = pd.read_csv(sat_index)

    uav_df = prepare_uav_df(uav_df, args.sequence)

    filename_index, fallback_uav_dirs, fallback_sat_dirs = build_filename_index(args.sequence)

    print("S4C.1 Macro-contour PHOG retrieval")
    print("-----------------------------------")
    print(f"Sequence:        {args.sequence}")
    print(f"Tokens:          {tokens}")
    print(f"Satellite tiles: {len(sat_df)}")
    print(f"Settings:        {settings_slug(args)}")
    print("")

    cache_t0 = time.perf_counter()
    sat_desc, sat_row_positions, sat_tile_ids, sat_paths = load_or_build_satellite_cache(
        sat_df=sat_df,
        filename_index=filename_index,
        fallback_sat_dirs=fallback_sat_dirs,
        args=args,
    )
    cache_runtime_s = time.perf_counter() - cache_t0

    print("")
    print(f"Satellite descriptor matrix: {sat_desc.shape}")
    print(f"Cache/load time: {cache_runtime_s:.2f}s")
    print("")

    query_summaries: list[dict[str, Any]] = []

    for token in tokens:
        result = run_query(
            token=token,
            uav_df=uav_df,
            sat_df=sat_df,
            sat_desc=sat_desc,
            sat_row_positions=sat_row_positions,
            sat_tile_ids=sat_tile_ids,
            filename_index=filename_index,
            fallback_uav_dirs=fallback_uav_dirs,
            fallback_sat_dirs=fallback_sat_dirs,
            args=args,
        )
        if result is not None:
            query_summaries.append(result)

    summary_df = pd.DataFrame(query_summaries)

    slug = settings_slug(args)
    summary_csv = OUT_META_DIR / f"s4c1_{args.sequence}_{slug}_query_summary.csv"
    summary_json = OUT_REPORT_DIR / f"s4c1_{args.sequence}_{slug}_summary.json"

    summary_df.to_csv(summary_csv, index=False)

    aggregate = {
        "stage": "S4C.1_macrocontour_PHOG_retrieval",
        "sequence": args.sequence,
        "tokens_requested": tokens,
        "tokens_processed": [int(x["token"]) for x in query_summaries],
        "num_queries": len(query_summaries),
        "settings": settings_dict(args),
        "settings_slug": slug,
        "satellite_descriptor_shape": list(sat_desc.shape),
        "cache_runtime_s": cache_runtime_s,
        "query_summary_csv": str(summary_csv),
        "query_summaries": query_summaries,
        "notes": [
            "PHOG retrieval only.",
            "No Chamfer reranking yet.",
            "UAV lon/lat used only after ranking for evaluation/debug.",
            "This block should be inspected visually before running a larger benchmark.",
        ],
    }

    if len(summary_df) > 0:
        aggregate.update(
            {
                "mean_top1_error_m": float(summary_df["top1_center_error_m"].mean()),
                "median_top1_error_m": float(summary_df["top1_center_error_m"].median()),
                "mean_best_topk_error_m": float(summary_df["best_topk_center_error_m"].mean()),
                "median_best_topk_error_m": float(summary_df["best_topk_center_error_m"].median()),
                "recall_at_1_contains_gt": float(summary_df["recall_at_1_contains_gt"].mean()),
                "recall_at_5_contains_gt": float(summary_df["recall_at_5_contains_gt"].mean()),
                "recall_at_10_contains_gt": float(summary_df["recall_at_10_contains_gt"].mean()),
                "topk_under_20m_rate": float(summary_df["any_topk_under_20m"].mean()),
                "topk_under_40m_rate": float(summary_df["any_topk_under_40m"].mean()),
                "topk_under_60m_rate": float(summary_df["any_topk_under_60m"].mean()),
            }
        )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print("")
    print("S4C.1 complete")
    print("--------------")
    print(f"Query summary CSV: {summary_csv}")
    print(f"Summary JSON:      {summary_json}")
    print(f"Figures:           {OUT_FIG_DIR}")

    if len(summary_df) > 0:
        print("")
        print("Compact result")
        print("--------------")
        cols = [
            "token",
            "top1_tile_id",
            "top1_center_error_m",
            "best_topk_center_error_m",
            "first_rank_under_20m",
            "first_rank_under_40m",
            "first_rank_under_60m",
            "first_rank_contains_gt",
            "query_runtime_s",
        ]
        print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
