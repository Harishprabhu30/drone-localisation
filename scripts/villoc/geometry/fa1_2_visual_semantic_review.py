#!/usr/bin/env python3
"""FA1.2 visual/semantic review material for residual SIFT failures.

Creates one montage per FA1.0 failure query plus a contact sheet and a structured
manual-review worksheet. It does not rerun any matcher or change any selector.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_fa1_2.yaml"


def resolve(v):
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def bool_value(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}


def resolve_existing(value: Any, *roots: Path) -> Path:
    p = Path(str(value)).expanduser()
    candidates = [p] if p.is_absolute() else [ROOT / p, *[r / p for r in roots]]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError("Could not resolve: " + str(value))


def verify_config(cfg):
    if cfg.get("stage") != "FA1.2":
        raise RuntimeError("Expected FA1.2 config")
    b = cfg["scientific_boundary"]
    if b["development_truth_metadata_visible"] is not True:
        raise RuntimeError("FA1.2 is explicitly a post-freeze development review")
    for key in (
        "matcher_rerun_allowed", "selector_change_allowed",
        "threshold_tuning_allowed", "confidence_rule_training_allowed",
        "automated_semantic_labeling_allowed", "blind_data_loaded",
    ):
        if b[key] is not False:
            raise RuntimeError(f"{key} must remain false")


def load_rgb(path: Path, max_side: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img


def add_panel(ax, image, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def finite_text(v, digits=1):
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "nonfinite"
    return f"{x:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)

    fa0_report = json.loads(resolve(cfg["fa1_0"]["report"]).read_text())
    if fa0_report.get("status") != cfg["fa1_0"]["expected_status"]:
        raise RuntimeError("FA1.0 report is not the frozen PASS")
    if int(fa0_report["counts"]["union_unique_queries"]) != int(cfg["fa1_0"]["expected_union_queries"]):
        raise RuntimeError("FA1.0 union count changed")

    fa11_report = json.loads(resolve(cfg["fa1_1"]["report"]).read_text())
    if fa11_report.get("status") != cfg["fa1_1"]["expected_status"]:
        raise RuntimeError("FA1.1 report is not the frozen PASS")
    if not fa11_report["parity"]["all_sift_pairs_match_frozen_evidence"]:
        raise RuntimeError("FA1.1 SIFT parity is not PASS")
    if not fa11_report["parity"]["all_scale_m2_pairs_match_frozen_evidence"]:
        raise RuntimeError("FA1.1 scale_m2 parity is not PASS")
    if int(fa11_report["counts"]["sift_diagnostic_pairs"]) != int(cfg["fa1_1"]["expected_sift_pairs"]):
        raise RuntimeError("FA1.1 SIFT pair count changed")
    if int(fa11_report["counts"]["oracle_rescuable_paired_cases"]) != int(cfg["fa1_1"]["expected_oracle_pairs"]):
        raise RuntimeError("FA1.1 oracle pair count changed")
    if int(fa11_report["counts"]["scale_m2_orb_severe_pairs"]) != int(cfg["fa1_1"]["expected_scale_m2_severe_pairs"]):
        raise RuntimeError("FA1.1 severe ORB pair count changed")

    cases = pd.read_csv(resolve(cfg["fa1_0"]["case_inventory"]))
    diag = pd.read_csv(resolve(cfg["fa1_1"]["sift_pair_diagnostics"]))
    overlay_root = resolve(cfg["fa1_1"]["overlay_root"])

    spec = yaml.safe_load(resolve(cfg["source"]["trajectory_spec"]).read_text())
    manifest = pd.read_csv(resolve(cfg["source"]["manifest"]))
    tile_index_path = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
    tiles = pd.read_csv(tile_index_path)

    cases["query_id"] = pd.to_numeric(cases["query_id"], errors="raise").astype(int)
    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    tiles["tile_id"] = tiles["tile_id"].astype(str)

    qlookup = manifest.set_index("query_id")
    tlookup = tiles.set_index("tile_id")

    if cases["query_id"].nunique() != int(cfg["fa1_0"]["expected_union_queries"]):
        raise RuntimeError("FA1.2 case count changed")

    max_side = int(cfg["visualization"]["image_max_side_px"])
    dpi = int(cfg["visualization"]["montage_dpi"])

    outroot = resolve(cfg["output_root"])
    montage_root = outroot / "montages"
    montage_root.mkdir(parents=True, exist_ok=True)

    worksheet_rows = []
    montage_paths = []

    for i, case in cases.sort_values("query_id").iterrows():
        qid = int(case["query_id"])
        selected_tile = str(case["sift_selected_tile_id"])
        oracle_tile = str(case["sift_oracle_tile_id"])
        orb_tile = str(case["scale_selected_tile_id"])

        qpath = resolve_existing(qlookup.loc[qid, "image_path"])
        selected_path = resolve_existing(
            tlookup.loc[selected_tile, "tile_path"], tile_index_path.parent
        )
        oracle_path = resolve_existing(
            tlookup.loc[oracle_tile, "tile_path"], tile_index_path.parent
        )
        orb_path = resolve_existing(
            tlookup.loc[orb_tile, "tile_path"], tile_index_path.parent
        )

        selected_overlay = overlay_root / f"q{qid:03d}_selected_{selected_tile}_sift_inliers.png"
        oracle_overlay = overlay_root / f"q{qid:03d}_oracle_{oracle_tile}_sift_inliers.png"
        if not selected_overlay.is_file() or not oracle_overlay.is_file():
            raise FileNotFoundError(
                f"Missing FA1.1 overlay for q{qid}: "
                f"{selected_overlay.name} / {oracle_overlay.name}"
            )

        query_img = load_rgb(qpath, max_side)
        selected_img = load_rgb(selected_path, max_side)
        oracle_img = load_rgb(oracle_path, max_side)
        orb_img = load_rgb(orb_path, max_side)
        selected_overlay_img = load_rgb(selected_overlay, max_side)
        oracle_overlay_img = load_rgb(oracle_overlay, max_side)

        selected_diag = diag[
            (pd.to_numeric(diag["query_id"], errors="coerce") == qid)
            & (diag["role"].astype(str) == "selected")
        ]
        oracle_diag = diag[
            (pd.to_numeric(diag["query_id"], errors="coerce") == qid)
            & (diag["role"].astype(str) == "oracle")
        ]
        if len(selected_diag) != 1 or len(oracle_diag) != 1:
            raise RuntimeError(f"FA1.1 diagnostics not unique for q{qid}")
        sd = selected_diag.iloc[0]
        od = oracle_diag.iloc[0]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        add_panel(
            axes[0, 0],
            query_img,
            f"UAV query q{qid}",
        )
        add_panel(
            axes[0, 1],
            selected_img,
            (
                f"SIFT SELECTED {selected_tile}\n"
                f"selector#1 | DINO#{int(case['sift_selected_pool_rank'])} | "
                f"err={finite_text(case['sift_selected_error_m'])} m | "
                f"inliers={int(case['sift_selected_inliers'])}"
            ),
        )
        add_panel(
            axes[0, 2],
            oracle_img,
            (
                f"SIFT ORACLE {oracle_tile}\n"
                f"selector#{int(case['sift_oracle_selector_rank'])} | "
                f"DINO#{int(case['sift_oracle_pool_rank'])} | "
                f"err={finite_text(case['sift_oracle_error_m'])} m | "
                f"inliers={int(case['sift_oracle_inliers'])}"
            ),
        )
        add_panel(
            axes[1, 0],
            orb_img,
            (
                f"scale_m2 ORB SELECTED {orb_tile}\n"
                f"err={finite_text(case['scale_selected_error_m'])} m"
            ),
        )
        add_panel(
            axes[1, 1],
            selected_overlay_img,
            (
                "SIFT selected inliers\n"
                f"ratio={finite_text(sd['inlier_ratio'], 3)} | "
                f"sym.err={finite_text(sd['symmetric_error_median_px'], 2)} px | "
                f"qHull={finite_text(sd['query_hull_coverage'], 3)}"
            ),
        )
        add_panel(
            axes[1, 2],
            oracle_overlay_img,
            (
                "SIFT oracle inliers\n"
                f"ratio={finite_text(od['inlier_ratio'], 3)} | "
                f"sym.err={finite_text(od['symmetric_error_median_px'], 2)} px | "
                f"qHull={finite_text(od['query_hull_coverage'], 3)}"
            ),
        )

        flags = []
        if bool_value(case["oracle_rescuable"]):
            flags.append("oracle-rescuable")
        if bool_value(case["catastrophic_damage"]):
            flags.append("catastrophic-damage")
        if bool_value(case["severe_orb_le40_to_sift_unsafe"]):
            flags.append("SEVERE ORB<=40->SIFT unsafe")

        fig.suptitle(
            f"FA1.2 q{qid} | " + " | ".join(flags),
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        montage_path = montage_root / f"q{qid:03d}_failure_review.png"
        fig.savefig(montage_path, dpi=dpi)
        plt.close(fig)
        montage_paths.append(montage_path)

        worksheet_rows.append({
            "query_id": qid,
            "montage_path": str(montage_path),
            "oracle_rescuable": bool_value(case["oracle_rescuable"]),
            "catastrophic_damage": bool_value(case["catastrophic_damage"]),
            "severe_orb_le40_to_sift_unsafe": bool_value(case["severe_orb_le40_to_sift_unsafe"]),
            "sift_selected_tile_id": selected_tile,
            "sift_oracle_tile_id": oracle_tile,
            "scale_m2_orb_tile_id": orb_tile,
            "sift_selected_error_m": case["sift_selected_error_m"],
            "sift_oracle_error_m": case["sift_oracle_error_m"],
            "scale_m2_orb_error_m": case["scale_selected_error_m"],
            "sift_selected_selector_rank": 1,
            "sift_oracle_selector_rank": int(case["sift_oracle_selector_rank"]),
            "sift_selected_pool_rank": int(case["sift_selected_pool_rank"]),
            "sift_oracle_pool_rank": int(case["sift_oracle_pool_rank"]),
            "sift_selected_inliers": int(case["sift_selected_inliers"]),
            "sift_oracle_inliers": int(case["sift_oracle_inliers"]),
            "sift_selected_inlier_ratio": sd["inlier_ratio"],
            "sift_oracle_inlier_ratio": od["inlier_ratio"],
            "sift_selected_symmetric_error_px": sd["symmetric_error_median_px"],
            "sift_oracle_symmetric_error_px": od["symmetric_error_median_px"],
            "sift_selected_query_hull_coverage": sd["query_hull_coverage"],
            "sift_oracle_query_hull_coverage": od["query_hull_coverage"],
            # Manual fields deliberately left blank.
            "scene_primary": "",
            "repetitive_buildings_roofs": "",
            "repetitive_roads_intersections": "",
            "vegetation_dominant": "",
            "parking_or_open_paved_repeat": "",
            "water_or_wetland": "",
            "strong_shadow_or_illumination_difference": "",
            "partial_overlap_or_edge_crop": "",
            "selected_match_spatially_concentrated": "",
            "oracle_match_spatially_concentrated": "",
            "selected_visual_match_plausible": "",
            "oracle_visual_match_plausible": "",
            "neighboring_tiles_same_locality": "",
            "likely_failure_mechanism": "",
            "reviewer_confidence_low_med_high": "",
            "notes": "",
        })

    worksheet = pd.DataFrame(worksheet_rows)
    worksheet_path = outroot / "fa1_2_manual_review_worksheet.csv"
    worksheet.to_csv(worksheet_path, index=False)

    # Contact sheet: thumbnails of all 20 case montages.
    cols = int(cfg["visualization"]["contact_sheet_columns"])
    rows = int(math.ceil(len(montage_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.6))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, path in zip(axes, montage_paths):
        img = load_rgb(path, int(cfg["visualization"]["contact_sheet_thumb_px"]))
        ax.imshow(img)
        ax.set_title(path.stem.split("_")[0], fontsize=10)
        ax.axis("off")
    fig.suptitle("FA1.2 — 20 residual SIFT failure cases", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    contact_path = outroot / "fa1_2_failure_contact_sheet.png"
    fig.savefig(contact_path, dpi=150)
    plt.close(fig)

    review_guide = outroot / "fa1_2_manual_review_guide.md"
    review_guide.write_text(
        """# FA1.2 manual review guide

For each case, inspect the UAV query, SIFT-selected tile, SIFT-oracle tile,
scale_m2 ORB tile, and selected/oracle inlier overlays.

Suggested manual labels:

- scene_primary:
  urban_dense | urban_sparse | road | industrial | vegetation | mixed | other
- boolean-like fields:
  yes | no | uncertain
- likely_failure_mechanism:
  repetitive_structure | repetitive_road_geometry | vegetation_aliasing |
  partial_overlap | selected_high_count_low_quality | homography_degeneracy |
  retrieval_deep_oracle | map_tile_boundary | illumination_or_shadow |
  scale_orientation_ambiguity | temporal_outlier | mixed | unclear
- reviewer_confidence_low_med_high:
  low | medium | high

Do not alter thresholds or selector rules based on this worksheet. FA1.2 is a
diagnostic review only.
"""
    )

    report_path = outroot / "fa1_2_report.json"
    report = {
        "stage": "FA1.2",
        "status": "PASS_FA1_2_VISUAL_SEMANTIC_REVIEW_PACKAGE",
        "counts": {
            "cases": int(len(worksheet)),
            "montages": int(len(montage_paths)),
            "contact_sheet": 1,
        },
        "inputs": {
            "fa1_0_status": fa0_report["status"],
            "fa1_1_status": fa11_report["status"],
            "fa1_1_sift_parity": bool(fa11_report["parity"]["all_sift_pairs_match_frozen_evidence"]),
            "fa1_1_scale_m2_parity": bool(fa11_report["parity"]["all_scale_m2_pairs_match_frozen_evidence"]),
        },
        "scientific_boundary": {
            "matcher_rerun": False,
            "selector_changed": False,
            "threshold_tuned": False,
            "confidence_rule_trained": False,
            "automated_semantic_labels": False,
            "manual_review_required": True,
            "blind_data_loaded": False,
        },
        "outputs": {
            "montage_root": str(montage_root),
            "contact_sheet": str(contact_path),
            "manual_review_worksheet": str(worksheet_path),
            "manual_review_guide": str(review_guide),
        },
    }
    report_path.write_text(json.dumps(report, indent=2))

    print("Status:", report["status"])
    print("Cases/montages:", f"{len(worksheet)}/{len(montage_paths)}")
    print("Contact sheet:", contact_path)
    print("Review worksheet:", worksheet_path)
    print("Review guide:", review_guide)
    print("Montage root:", montage_root)
    print("Matcher rerun / tuning / selector change: False")
    print("Manual semantic review required: True")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
