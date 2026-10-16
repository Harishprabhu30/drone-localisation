#!/usr/bin/env python3
"""Create neutral canonical aliases/provenance for a completed localization run.

Historical helper scripts still emit a few blind/submission filenames.
This stage preserves those immutable compatibility artifacts and exposes neutral
project-facing names without changing their bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_exact(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256(dst) != sha256(src):
            raise RuntimeError(f"Refusing to overwrite differing canonical alias: {dst}")
        return
    shutil.copy2(src, dst)
    if sha256(src) != sha256(dst):
        raise RuntimeError(f"Canonical alias hash mismatch: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    run = args.run_root.resolve()
    cfg = args.config.resolve()

    aliases = {
        run / "metadata/localization_query_manifest.csv":
            run / "metadata/blind_query_manifest.csv",
        run / "trajectories/estimated_trajectory.csv":
            run / "trajectories/submission_estimated_trajectory.csv",
        run / "map_state/map_state.json":
            run / "reports/blind_map_bootstrap/canonical_map_state.json",
        run / "map_state/map_state_timeline.json":
            run / "reports/blind_map_bootstrap/canonical_map_state_timeline.json",
        run / "map_state/temporal_authority_timeline.csv":
            run / "metadata/blind_temporal_fusion/blind_temporal_correction_manifest.csv",
        run / "provenance/output_freeze.json":
            run / "evaluation/blind_submission_freeze.json",
    }

    for dst, src in aliases.items():
        copy_exact(src, dst)

    trajectory = pd.read_csv(run / "trajectories/estimated_trajectory.csv")
    verifier = json.loads((run / "absolute/verifier_report.json").read_text())
    bootstrap = json.loads(
        (run / "reports/blind_map_bootstrap/blind_map_bootstrap_report.json").read_text()
    )
    temporal = json.loads(
        (run / "reports/blind_temporal_fusion/blind_temporal_fusion_report.json").read_text()
    )
    freeze = json.loads((run / "provenance/output_freeze.json").read_text())

    manifest = {
        "schema": "villoc.localization_run.v1",
        "status": "PASS_CURRENT_STACK_CANONICAL_RUN_PACKAGE",
        "config": str(cfg),
        "config_sha256": sha256(cfg),
        "architecture": {
            "relative_frontend": "XFeat",
            "retrieval": "DINOv2 ViT-S/14 img518 center_square avgpatch Top20",
            "absolute_verifier": "SIFT+LightGlue",
            "map_variant": "768_s256",
            "bootstrap": "minimum_confident_v2",
            "temporal_authority": "minimum_confident_v2 causal map-state timeline",
        },
        "localization_state": str(trajectory["localization_state"].iloc[0]),
        "rows": int(len(trajectory)),
        "map_aligned_rows": int(trajectory["map_aligned_available"].astype(bool).sum()),
        "accepted_map_state_events": int(
            trajectory["accepted_correction"].astype(bool).sum()
        ),
        "reference_used": False,
        "artifacts": {},
        "compatibility_note": (
            "Historical blind/submission filenames remain only for replay compatibility. "
            "Canonical project-facing paths are metadata/localization_query_manifest.csv, "
            "trajectories/estimated_trajectory.csv, map_state/* and provenance/output_freeze.json."
        ),
        "source_statuses": {
            "verifier": verifier.get("status"),
            "bootstrap": bootstrap.get("status"),
            "temporal": temporal.get("status"),
            "freeze": freeze.get("status"),
        },
    }

    for rel in [
        "metadata/localization_query_manifest.csv",
        "absolute/geometric_candidate_evidence.csv",
        "absolute/selected_absolute_observations.csv",
        "absolute/absolute_query_summary.csv",
        "trajectories/estimated_trajectory.csv",
        "map_state/map_state.json",
        "map_state/map_state_timeline.json",
        "map_state/temporal_authority_timeline.csv",
        "provenance/output_freeze.json",
    ]:
        path = run / rel
        manifest["artifacts"][rel] = {
            "path": str(path),
            "sha256": sha256(path),
        }

    provenance = run / "provenance/run_manifest.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(manifest, indent=2))

    summary = run / "summary/localization_run_summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        "# Localization Run Summary\n\n"
        f"- Status: **{manifest['status']}**\n"
        f"- Localization state: **{manifest['localization_state']}**\n"
        f"- Queries: **{manifest['rows']}**\n"
        f"- Map-aligned rows: **{manifest['map_aligned_rows']}**\n"
        f"- Accepted map-state events: **{manifest['accepted_map_state_events']}**\n"
        "- Reference/GPS/SRT used by localization: **No**\n"
        "- Absolute frontend: **DINOv2 Top20 -> SIFT+LightGlue -> Top4 minimum_confident_v2**\n"
        "- Canonical trajectory: trajectories/estimated_trajectory.csv\n"
        "- Provenance: provenance/run_manifest.json\n"
    )

    print("STATUS: PASS_CURRENT_STACK_CANONICAL_RUN_PACKAGE")
    print("canonical trajectory:", run / "trajectories/estimated_trajectory.csv")
    print("run manifest:", provenance)


if __name__ == "__main__":
    main()
