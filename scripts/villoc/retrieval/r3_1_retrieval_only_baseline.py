#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer


ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/absolute_frontend_decoupling_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/absolute_frontend_decoupling_v1/r3_1"
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_reference_xy(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"query_id", "eval_ref_lon", "eval_ref_lat"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Reference attachment missing {sorted(missing)}"
        )

    frame = frame.copy()
    frame["query_id"] = pd.to_numeric(
        frame["query_id"], errors="raise"
    ).astype(int)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3346",
        always_xy=True,
    )
    x, y = transformer.transform(
        pd.to_numeric(
            frame["eval_ref_lon"], errors="raise"
        ).to_numpy(float),
        pd.to_numeric(
            frame["eval_ref_lat"], errors="raise"
        ).to_numpy(float),
    )
    frame["gt_x"] = x
    frame["gt_y"] = y
    return frame.set_index("query_id")


def prepare_index(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    required = {
        "tile_id",
        "left_easting",
        "bottom_northing",
        "right_easting",
        "top_northing",
        "center_easting",
        "center_northing",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Tile index {path} missing {sorted(missing)}"
        )

    frame["tile_id"] = frame["tile_id"].astype(str)
    if frame["tile_id"].duplicated().any():
        raise RuntimeError(
            f"Tile index contains duplicate tile_id values: {path}"
        )
    return frame.set_index("tile_id")


def evaluate_selection(
    *,
    selection_name: str,
    selected: pd.DataFrame,
    tile_index: pd.DataFrame,
    reference_xy: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    selected = selected.copy()
    selected["query_id"] = pd.to_numeric(
        selected["query_id"], errors="raise"
    ).astype(int)
    selected["tile_id"] = selected["tile_id"].astype(str)

    if selected["query_id"].duplicated().any():
        raise RuntimeError(
            f"{selection_name}: more than one selected row per query"
        )

    for row in selected.itertuples(index=False):
        qid = int(row.query_id)
        tile_id = str(row.tile_id)

        if qid not in reference_xy.index:
            raise RuntimeError(
                f"{selection_name}: reference missing q{qid}"
            )
        if tile_id not in tile_index.index:
            raise RuntimeError(
                f"{selection_name}: tile index missing {tile_id}"
            )

        ref = reference_xy.loc[qid]
        tile = tile_index.loc[tile_id]

        x = float(ref["gt_x"])
        y = float(ref["gt_y"])
        error = float(
            math.hypot(
                float(tile["center_easting"]) - x,
                float(tile["center_northing"]) - y,
            )
        )
        contains = bool(
            float(tile["left_easting"]) <= x
            <= float(tile["right_easting"])
            and float(tile["bottom_northing"]) <= y
            <= float(tile["top_northing"])
        )

        rows.append(
            {
                "selection": selection_name,
                "query_id": qid,
                "tile_id": tile_id,
                "center_error_m": error,
                "contains_query": contains,
                "le40": bool(error <= 40.0),
                "le80": bool(error <= 80.0),
            }
        )

    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict:
    errors = frame["center_error_m"].to_numpy(float)
    return {
        "query_count": int(len(frame)),
        "contains_hits": int(frame["contains_query"].sum()),
        "contains_rate": float(frame["contains_query"].mean()),
        "le40_hits": int(frame["le40"].sum()),
        "le40_rate": float(frame["le40"].mean()),
        "le80_hits": int(frame["le80"].sum()),
        "le80_rate": float(frame["le80"].mean()),
        "center_error_mean_m": float(np.mean(errors)),
        "center_error_median_m": float(np.median(errors)),
        "center_error_p95_m": float(np.percentile(errors, 95)),
        "center_error_max_m": float(np.max(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    config_path = resolve(args.config)
    output_root = resolve(args.output_root)
    cfg = yaml.safe_load(config_path.read_text())

    artifacts = {
        name: resolve(value)
        for name, value in cfg["artifacts"].items()
    }

    required_names = (
        "stable_768_topk",
        "stable_768_tile_index",
        "stable_768_orb_eval_source",
        "triple_fused_topk",
        "triple_mixed_tile_index",
        "triple_orb_selected_eval",
        "reference_attachment",
    )
    for name in required_names:
        path = artifacts[name]
        if not path.exists():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    # ------------------------------
    # BLIND SELECTION PHASE
    # ------------------------------
    topk_768 = pd.read_csv(
        artifacts["stable_768_topk"]
    )
    fused = pd.read_csv(
        artifacts["triple_fused_topk"]
    )

    for label, frame in (
        ("768 Top-K", topk_768),
        ("triple fused Top-K", fused),
    ):
        required = {"query_id", "rank", "tile_id"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"{label} missing {sorted(missing)}"
            )

    selected_768 = topk_768[
        pd.to_numeric(
            topk_768["rank"], errors="coerce"
        ) == 1
    ][["query_id", "tile_id"]].copy()

    selected_triple = fused[
        pd.to_numeric(
            fused["rank"], errors="coerce"
        ) == 1
    ][["query_id", "tile_id"]].copy()

    if len(selected_768) != 403:
        raise RuntimeError(
            f"Expected 403 direct 768 selections, got {len(selected_768)}"
        )
    if len(selected_triple) != 403:
        raise RuntimeError(
            f"Expected 403 direct triple selections, got {len(selected_triple)}"
        )

    blind_freeze = pd.concat(
        [
            selected_768.assign(
                selection="768_dino_top1"
            ),
            selected_triple.assign(
                selection="triple_fused_rank1_representative"
            ),
        ],
        ignore_index=True,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    freeze_path = (
        output_root
        / "r3_1_blind_retrieval_only_selections.csv"
    )
    blind_freeze.to_csv(
        freeze_path,
        index=False,
    )
    freeze_sha = sha256_file(freeze_path)

    # ------------------------------
    # POST-SELECTION EVALUATION ONLY
    # ------------------------------
    reference_xy = load_reference_xy(
        artifacts["reference_attachment"]
    )
    index_768 = prepare_index(
        artifacts["stable_768_tile_index"]
    )
    mixed_index = prepare_index(
        artifacts["triple_mixed_tile_index"]
    )

    eval_768 = evaluate_selection(
        selection_name="768_dino_top1",
        selected=selected_768,
        tile_index=index_768,
        reference_xy=reference_xy,
    )
    eval_triple = evaluate_selection(
        selection_name="triple_fused_rank1_representative",
        selected=selected_triple,
        tile_index=mixed_index,
        reference_xy=reference_xy,
    )

    evaluated = pd.concat(
        [eval_768, eval_triple],
        ignore_index=True,
    )

    if sha256_file(freeze_path) != freeze_sha:
        raise RuntimeError(
            "Blind retrieval-only selection artifact changed during evaluation"
        )

    summary = {
        "768_dino_top1": summarize(eval_768),
        "triple_fused_rank1_representative": summarize(
            eval_triple
        ),
    }

    # Existing downstream comparators; not used to choose retrieval candidates.
    r14 = json.loads(
        artifacts["stable_768_orb_eval_source"].read_text()
    )
    stable_orb = r14["variants"]["768_s256"]["orb_selected"]

    triple_orb_eval = pd.read_csv(
        artifacts["triple_orb_selected_eval"]
    )
    triple_orb_summary = {
        "selected_contains_hits": int(
            triple_orb_eval[
                "selected_contains_query"
            ].astype(bool).sum()
        ),
        "selected_le40_hits": int(
            triple_orb_eval["selected_le40"].astype(bool).sum()
        ),
        "selected_le80_hits": int(
            triple_orb_eval["selected_le80"].astype(bool).sum()
        ),
    }

    named = {}
    for qid in cfg["r3_1"]["named_diagnostics"]:
        qid = int(qid)
        named[f"q{qid}"] = {}
        for name, frame in (
            ("768_dino_top1", eval_768),
            (
                "triple_fused_rank1_representative",
                eval_triple,
            ),
        ):
            row = frame[
                frame["query_id"] == qid
            ]
            named[f"q{qid}"][name] = (
                row.iloc[0].to_dict()
                if len(row) == 1
                else None
            )

    eval_path = (
        output_root
        / "r3_1_retrieval_only_evaluation.csv"
    )
    evaluated.to_csv(
        eval_path,
        index=False,
    )

    report = {
        "stage": "R3.1",
        "status": "PASS_R3_RETRIEVAL_ONLY_BASELINE",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Without ORB reranking/projection or state estimation, how good "
            "is direct 768 DINO Top-1 versus direct triple fused rank-1 "
            "representative selection?"
        ),
        "blind_selection_freeze": {
            "path": str(freeze_path),
            "sha256": freeze_sha,
            "reference_used_for_selection": False,
            "orb_used_for_selection": False,
            "state_used_for_selection": False,
        },
        "summary": summary,
        "existing_orb_comparators": {
            "stable_768_orb_selected": stable_orb,
            "triple_orb_selected": triple_orb_summary,
            "used_for_retrieval_selection": False,
        },
        "named_diagnostics": named,
        "input_hashes": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
        "outputs": {
            "evaluation_csv": str(eval_path),
        },
        "scope_guarantees": {
            "new_selector_introduced": False,
            "orb_reranking_run": False,
            "orb_projection_run": False,
            "bootstrap_run": False,
            "state_policy_changed": False,
            "reference_used_before_selection_freeze": False,
        },
        "next_decision": (
            "Use retrieval-only performance and named failures to design the "
            "first retrieval-native confidence/consensus selector. Do not "
            "reintroduce ORB/state until the selection question is isolated."
        ),
    }

    report_path = (
        output_root
        / "r3_1_retrieval_only_baseline_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("R3.1 — RETRIEVAL-ONLY ABSOLUTE SELECTION BASELINE")
    print("=" * 100)
    for name, item in summary.items():
        print(
            f"{name:38s} "
            f"contains={item['contains_hits']:3d}/403 "
            f"<=40={item['le40_hits']:3d}/403 "
            f"<=80={item['le80_hits']:3d}/403 "
            f"median={item['center_error_median_m']:.3f} "
            f"p95={item['center_error_p95_m']:.3f}"
        )

    print()
    print("Existing ORB-selected comparators (not used for selection)")
    print("-" * 100)
    print("768:", stable_orb)
    print("triple:", triple_orb_summary)

    for qid in cfg["r3_1"]["named_diagnostics"]:
        print()
        print(f"q{qid}:")
        print(
            json.dumps(
                named[f"q{int(qid)}"],
                indent=2,
                default=str,
            )
        )

    print()
    print("STATUS: PASS_R3_RETRIEVAL_ONLY_BASELINE")
    print("report:", report_path)


if __name__ == "__main__":
    main()
