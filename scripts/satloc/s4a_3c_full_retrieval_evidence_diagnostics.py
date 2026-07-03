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


def find_ranked_csv(
    sequence: str,
    query_token0_id: int,
    variant: str,
    nfeatures: int,
) -> Path:
    run_id = f"{sequence}_token{query_token0_id:04d}_fullcached_{variant}_nf{nfeatures}"
    csv_path = Path(
        "outputs/satloc/metadata/s4a_orb_tile_retrieval/full_cached"
    ) / f"{run_id}_ranked_tiles.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing ranked CSV: {csv_path}\n"
            "Run S4A.3B first for this query/variant/nfeatures."
        )

    return csv_path


def select_evidence_rows(ranked_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = []

    top_df = ranked_df.head(top_n).copy()
    for _, row in top_df.iterrows():
        r = row.copy()
        r["selection_label"] = f"top{int(row['rank'])}"
        selected.append(r)

    correct_df = ranked_df[ranked_df["contains_gt"] == True].copy()

    if len(correct_df) > 0:
        first_correct = correct_df.sort_values("rank").iloc[0].copy()
        first_correct["selection_label"] = "first_correct"
        selected.append(first_correct)

        best_correct = correct_df.sort_values(
            ["score", "ransac_inliers", "good_matches", "inlier_ratio"],
            ascending=[False, False, False, False],
        ).iloc[0].copy()
        best_correct["selection_label"] = "best_correct"
        selected.append(best_correct)

        closest_correct = correct_df.sort_values("center_error_m").iloc[0].copy()
        closest_correct["selection_label"] = "closest_correct"
        selected.append(closest_correct)

    selected_df = pd.DataFrame(selected)
    selected_df = selected_df.drop_duplicates(subset=["tile_index", "rank"]).reset_index(drop=True)
    return selected_df


def save_matches_image(
    query_gray: np.ndarray,
    tile_gray: np.ndarray,
    query_kp,
    tile_kp,
    matches,
    out_path: Path,
    title: str,
    max_matches: int = 100,
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

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(vis)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_selected_tiles_panel(
    query_bgr: np.ndarray,
    selected_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    ncols = 3
    n_items = len(selected_df) + 1
    nrows = int(np.ceil(n_items / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(query_rgb)
    axes[0].set_title("UAV query")
    axes[0].axis("off")

    for ax_i, (_, row) in enumerate(selected_df.iterrows(), start=1):
        tile_bgr = read_bgr(Path(str(row["tile_path"])))
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)

        label = (
            f"{row['selection_label']}\n"
            f"rank={int(row['rank'])}, tile={int(row['tile_index'])}\n"
            f"score={row['score']:.2f}, good={int(row['good_matches'])}, "
            f"inl={int(row['ransac_inliers'])}\n"
            f"GT={bool(row['contains_gt'])}, err={row['center_error_m']:.1f}m"
        )

        axes[ax_i].imshow(tile_rgb)
        axes[ax_i].set_title(label, fontsize=9)
        axes[ax_i].axis("off")

    for j in range(n_items, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_metric_bar_plot(selected_df: pd.DataFrame, out_path: Path, title: str) -> None:
    labels = [
        f"{row.selection_label}\nR{int(row.rank)} T{int(row.tile_index)}"
        for row in selected_df.itertuples()
    ]

    x = np.arange(len(selected_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, selected_df["score"], width, label="score")
    ax.bar(x, selected_df["good_matches"], width, label="good")
    ax.bar(x + width, selected_df["ransac_inliers"], width, label="inliers")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Count / score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_rank_context_plot(ranked_df: pd.DataFrame, selected_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    plot_df = ranked_df.head(min(len(ranked_df), 1000)).copy()
    ax.plot(plot_df["rank"], plot_df["score"], linewidth=1.0)
    ax.scatter(plot_df["rank"], plot_df["score"], s=8)

    for _, row in selected_df.iterrows():
        rank = int(row["rank"])
        if rank <= 1000:
            ax.scatter(rank, row["score"], s=80, marker="x")
            ax.text(
                rank,
                row["score"],
                f" {row['selection_label']}\n T{int(row['tile_index'])}",
                fontsize=8,
                va="bottom",
            )

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
    parser.add_argument("--query-token0-id", type=int, default=1)
    parser.add_argument("--variant", default="V2_clahe_luma")
    parser.add_argument("--nfeatures", type=int, default=1200)
    parser.add_argument("--top-n", type=int, default=3)
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

    ranked_csv = find_ranked_csv(
        sequence=args.sequence,
        query_token0_id=args.query_token0_id,
        variant=args.variant,
        nfeatures=args.nfeatures,
    )

    ranked_df = pd.read_csv(ranked_csv)
    selected_df = select_evidence_rows(ranked_df, top_n=args.top_n)

    if len(selected_df) == 0:
        raise RuntimeError("No evidence rows selected.")

    query_path = Path(str(ranked_df.iloc[0]["query_path"]))
    query_bgr = read_bgr(query_path)

    run_id = (
        f"{args.sequence}_token{args.query_token0_id:04d}"
        f"_fullcached_{args.variant}_nf{args.nfeatures}"
    )

    out_meta_dir = Path("outputs/satloc/metadata/s4a_orb_tile_retrieval/full_evidence_debug") / run_id
    out_report_dir = Path("outputs/satloc/reports/s4a_orb_tile_retrieval/full_evidence_debug") / run_id
    out_fig_dir = Path("outputs/satloc/figures/s4a_orb_tile_retrieval/full_evidence_debug") / run_id

    ensure_dir(out_meta_dir)
    ensure_dir(out_report_dir)
    ensure_dir(out_fig_dir)

    print("\nS4A.3C full-retrieval evidence diagnostics")
    print("------------------------------------------")
    print(f"Ranked CSV: {ranked_csv}")
    print(f"Query: {query_path}")
    print(f"Selected rows: {len(selected_df)}")

    evidence_rows = []

    for _, row in selected_df.iterrows():
        tile_path = Path(str(row["tile_path"]))
        tile_bgr = read_bgr(tile_path)

        result = match_orb_pair(
            query_bgr=query_bgr,
            tile_bgr=tile_bgr,
            variant=args.variant,
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

        inlier_matches = []
        if result.inlier_mask is not None:
            inlier_matches = [
                m for m, keep in zip(result.good_match_objects, result.inlier_mask)
                if bool(keep)
            ]

        label = str(row["selection_label"])
        rank = int(row["rank"])
        tile_index = int(row["tile_index"])

        good_path = out_fig_dir / f"{label}_rank{rank:04d}_tile{tile_index:05d}_good_matches.png"
        inlier_path = out_fig_dir / f"{label}_rank{rank:04d}_tile{tile_index:05d}_ransac_inliers.png"

        save_matches_image(
            result.query_processed,
            result.tile_processed,
            result.query_kp,
            result.tile_kp,
            result.good_match_objects,
            good_path,
            title=(
                f"{label} | rank {rank} | tile {tile_index} | "
                f"good={result.good_matches}, score={result.score:.2f}"
            ),
        )

        save_matches_image(
            result.query_processed,
            result.tile_processed,
            result.query_kp,
            result.tile_kp,
            inlier_matches,
            inlier_path,
            title=(
                f"{label} | rank {rank} | tile {tile_index} | "
                f"inliers={result.ransac_inliers}, H={result.homography_success}"
            ),
        )

        evidence_row = row.to_dict()
        evidence_row.update(
            {
                "rerun_query_keypoints": result.query_keypoints,
                "rerun_tile_keypoints": result.tile_keypoints,
                "rerun_good_matches": result.good_matches,
                "rerun_ransac_inliers": result.ransac_inliers,
                "rerun_inlier_ratio": result.inlier_ratio,
                "rerun_homography_success": result.homography_success,
                "rerun_score": result.score,
                "rerun_elapsed_ms": result.elapsed_ms,
                "good_matches_figure": str(good_path),
                "ransac_inliers_figure": str(inlier_path),
            }
        )
        evidence_rows.append(evidence_row)

        print(
            f"{label:16s} | rank={rank:4d} tile={tile_index:5d} "
            f"GT={bool(row['contains_gt'])} "
            f"err={float(row['center_error_m']):8.1f}m "
            f"score={result.score:5.2f} "
            f"good={result.good_matches:3d} "
            f"inliers={result.ransac_inliers:3d} "
            f"ratio={result.inlier_ratio:.3f} "
            f"H={result.homography_success}"
        )

    evidence_df = pd.DataFrame(evidence_rows)

    selected_panel_path = out_fig_dir / "selected_tiles_panel.png"
    metric_plot_path = out_fig_dir / "selected_tiles_metric_comparison.png"
    rank_context_path = out_fig_dir / "rank_context_score_plot.png"

    save_selected_tiles_panel(
        query_bgr=query_bgr,
        selected_df=selected_df,
        out_path=selected_panel_path,
        title=f"S4A.3C full retrieval evidence | {run_id}",
    )

    save_metric_bar_plot(
        selected_df=evidence_df,
        out_path=metric_plot_path,
        title=f"Score/good/inlier comparison | {run_id}",
    )

    save_rank_context_plot(
        ranked_df=ranked_df,
        selected_df=evidence_df,
        out_path=rank_context_path,
        title=f"Rank context score plot | {run_id}",
    )

    evidence_csv = out_meta_dir / "full_retrieval_evidence_tiles.csv"
    evidence_json = out_report_dir / "full_retrieval_evidence_summary.json"

    evidence_df.to_csv(evidence_csv, index=False)

    summary = {
        "run_id": run_id,
        "ranked_csv": str(ranked_csv),
        "query_path": str(query_path),
        "selected_tiles": evidence_rows,
        "selected_tiles_panel": str(selected_panel_path),
        "metric_plot": str(metric_plot_path),
        "rank_context_plot": str(rank_context_path),
        "evidence_csv": str(evidence_csv),
    }

    with evidence_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(f"CSV: {evidence_csv}")
    print(f"JSON: {evidence_json}")
    print(f"Figs: {out_fig_dir}")


if __name__ == "__main__":
    main()