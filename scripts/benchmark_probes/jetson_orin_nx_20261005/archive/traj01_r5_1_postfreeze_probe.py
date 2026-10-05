#!/usr/bin/env python3
"""Read-only post-freeze quality probe for a traj01 R5.1 v2 run.

This checks the blind freeze and its recorded inputs/outputs before opening
reference_attachment.csv. It reports the maturity-position error and a
diagnostic replay of each research policy's saved active transform. The replay
is not the 403-frame orchestrator submission or its official evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def verify(repo: Path, freeze: dict) -> None:
    assert freeze["stage"] == "R5.1_BLIND_IMPLEMENTATION_FREEZE"
    for name, value in freeze["blind_contract"].items():
        assert value is False, f"Blind contract {name}={value!r}"
    for section in ("inputs", "outputs"):
        if section == "inputs":
            records = [
                (name, freeze[section][name], freeze[section][hash_name])
                for name, hash_name in (
                    ("candidate_csv", "candidate_sha256"),
                    ("relative_csv", "relative_sha256"),
                    ("manifest_csv", "manifest_sha256"),
                    ("r4_11_source", "r4_11_source_sha256"),
                )
            ]
        else:
            records = [
                (name, info["path"], info["sha256"])
                for name, info in freeze[section].items()
            ]
        for name, value, expected in records:
            path = recorded_path(repo, value)
            assert path.is_file(), f"Missing frozen {section} {name}: {path}"
            actual = sha256(path)
            assert actual == expected, f"SHA256 mismatch for {section} {name}: {path}"
        print(f"Verified {len(records)} frozen {section} hashes")
    contract = repo / "configs/bootstrap/minimum_confident_v2/architecture_contract.json"
    assert sha256(contract) == freeze["architecture_contract_sha256"], "Architecture contract mismatch"
    print("Verified architecture contract hash")


def estimated_xy(x: float, y: float, model: dict) -> np.ndarray:
    a = complex(float(model["a_real"]), float(model["a_imag"]))
    b = complex(float(model["b_real"]), float(model["b_imag"]))
    estimate = a * complex(x, y) + b
    return np.array([estimate.real, estimate.imag])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--research-root", type=Path,
        default=Path("outputs/research_runs/minimum_confident_bootstrap/"
                     "traj01_blind_r5_1_v2_equivalence_001"),
        help="Frozen R5.1 output directory (existing 38-query run by default).",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    run = repo / "outputs/demo_runs/traj01_blind_regression_001"
    research = recorded_path(repo, str(args.research_root))
    freeze_path = research / "r5_1_blind_implementation_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    print("Blind freeze SHA256:", sha256(freeze_path))
    verify(repo, freeze)

    # Evaluation boundary: the first reference read happens below this line.
    reference_path = run / "evaluation/reference_attachment.csv"
    reference = pd.read_csv(reference_path)
    relative = pd.read_csv(recorded_path(repo, freeze["inputs"]["relative_csv"]))
    required_reference = {"query_id", "eval_ref_lon", "eval_ref_lat"}
    required_relative = {"token0_id", "visual_x_px", "visual_y_px"}
    assert required_reference <= set(reference), f"Reference columns: {list(reference)}"
    assert required_relative <= set(relative), f"Relative columns: {list(relative)}"
    reference["query_id"] = pd.to_numeric(reference["query_id"]).astype(int)
    relative["query_id"] = pd.to_numeric(relative["token0_id"]).astype(int)
    assert reference.query_id.is_unique and relative.query_id.is_unique
    east, north = Transformer.from_crs("EPSG:4326", "EPSG:3346", always_xy=True).transform(
        reference.eval_ref_lon.to_numpy(float), reference.eval_ref_lat.to_numpy(float)
    )
    reference = reference[["query_id"]].assign(gt_easting=east, gt_northing=north)
    joined = relative[["query_id", "visual_x_px", "visual_y_px"]].merge(
        reference, on="query_id", validate="one_to_one"
    ).set_index("query_id")
    max_q = int(freeze["configuration"]["max_query_id"] or freeze["counts"]["queries"])
    assert all(q in joined.index for q in range(1, max_q + 1)), "Reference/visual query IDs are incomplete"

    results = json.loads((research / "r5_1_blind_policy_results.json").read_text())
    assert results["blind_contract"]["reference_used"] is False
    timeline = pd.read_csv(research / "r5_1_blind_policy_timeline.csv")
    hypotheses = pd.read_csv(research / "r5_1_blind_subtile_hypotheses.csv")
    assert hypotheses.hypothesis_id.is_unique, "Ambiguous hypothesis IDs"
    hypotheses = hypotheses.set_index("hypothesis_id")

    print("Post-freeze reference joined to visual queries:", len(joined))
    print("Research prefix max query:", max_q)
    print("Policy replay uses saved active_hypothesis_id_after at each query;")
    print("it is a diagnostic, not the full orchestrator's fused trajectory.")
    for name, result in sorted(results["policy_results"].items()):
        maturity_q = result["matured_at_query_id"]
        if maturity_q is None:
            print(name, "NO_PROVISIONAL_LOCK")
            continue
        maturity_q = int(maturity_q)
        rows = timeline[(timeline.policy == name) & (timeline.update_query_id >= maturity_q)].copy()
        assert len(rows) == max_q - maturity_q + 1, f"Timeline gap for {name}"
        errors = []
        for _, row in rows.iterrows():
            q = int(row.update_query_id)
            visual = joined.loc[q]
            h = hypotheses.loc[int(row.active_hypothesis_id_after)]
            xy = estimated_xy(visual.visual_x_px, visual.visual_y_px, h)
            truth = visual[["gt_easting", "gt_northing"]].to_numpy(dtype=float)
            errors.append(float(np.linalg.norm(xy - truth)))
        errors = np.asarray(errors)
        maturity = result["maturity_map_state"]
        visual = joined.loc[maturity_q]
        maturity_xy = estimated_xy(visual.visual_x_px, visual.visual_y_px, maturity)
        maturity_truth = visual[["gt_easting", "gt_northing"]].to_numpy(dtype=float)
        print(name, json.dumps({
            "matured_at_q": maturity_q,
            "maturity_position_error_m": round(float(np.linalg.norm(maturity_xy - maturity_truth)), 3),
            "replayed_queries": len(errors),
            "replayed_rmse_m": round(float(np.sqrt(np.mean(errors ** 2))), 3),
            "replayed_median_m": round(float(np.median(errors)), 3),
            "replayed_final_q_error_m": round(float(errors[-1]), 3),
            "tracking_accepts": int((rows.action == "TRACKING_ACCEPT").sum()),
        }))


if __name__ == "__main__":
    main()
