"""Isolated 30-query R5.1 replay with exported Mac query ORB features.

Stage 06 and R5.1 recompute the same ORB features. This wrapper supplies the
exported query features to R5.1's R4.11 import; tile features stay Jetson
native. The frozen source files and original run directories are untouched.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()
BACKEND = ROOT / "scripts/villoc/blind_demo/bootstrap/minimum_confident_v2_backend.py"
R411_NAME = "r4_11_blind_subtile_projection_recompute.py"
ARCHITECTURE_SHA = "c47acc6313cd8d32a59b81d9457e29047192e7b9d6ef56524f39c4ce3208f93e"
EXPECTED = {
    BACKEND.name: "ee29deb6ba432dbe5d22a1ae8460c7bea4de8851d8f3671f1c1c1e1c576eaa83",
    R411_NAME: "393a93f30190e0e8e5a6ae7041e650478428ac4b79547ce7dec507c3c1426180",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-features", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--relative-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path(
        "outputs/demo_runs/jetson_native_cpu_baseline_20261002_002"))
    parser.add_argument("--max-query-id", type=int, default=30)
    args = parser.parse_args()
    assert args.max_query_id == 30, "This archived feature control covers queries 1..30 only"
    run = absolute(args.run_root).resolve()
    candidate_csv = absolute(args.candidate_csv).resolve()
    relative_csv = absolute(args.relative_csv).resolve()
    out = absolute(args.out_root).resolve()
    archive_path = absolute(args.mac_features).resolve()
    contract = ROOT / "configs/bootstrap/minimum_confident_v2/architecture_contract.json"
    r411_path = ROOT / "scripts/villoc/research/minimum_confident_bootstrap/diagnostics" / R411_NAME
    for path in (BACKEND, r411_path, contract, candidate_csv, relative_csv,
                 archive_path, run / "metadata/blind_query_manifest.csv"):
        assert path.is_file(), f"Missing input: {path}"
    for path in (BACKEND, r411_path):
        assert sha256(path) == EXPECTED[path.name], f"Frozen source changed: {path}"
    assert sha256(contract) == ARCHITECTURE_SHA, "Architecture contract changed"
    assert out != run and not out.is_relative_to(run), "Output must stay outside frozen run"
    assert not out.exists(), f"Choose a new isolated output directory: {out}"

    candidates = pd.read_csv(candidate_csv)
    ids = pd.to_numeric(candidates["query_id"], errors="raise").astype(int)
    assert len(candidates) == 600 and set(ids) == set(range(1, 31)), (
        "Expected 600 blind candidate rows for query IDs 1..30"
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"]))
        assert len(manifest) == 30
        features = {}
        for row in manifest:
            name = row["filename"]
            points = archive[f"{name}_points"]
            desc = archive[f"{name}_descriptors"]
            shape = tuple(int(x) for x in archive[f"{name}_shape"])
            assert len(points) == len(desc) == row["features"]
            keypoints = [cv2.KeyPoint(float(x), float(y), 31.0) for x, y in points]
            features[name] = (row["jpeg_sha256"], SimpleNamespace(
                ok=True, image_shape=shape, keypoints=keypoints,
                descriptors=desc, error=None,
            ))
    assert sorted(int(row["id"]) for row in manifest) == list(range(1, 31))
    for path_str in candidates["query_image_resolved"].drop_duplicates():
        path = absolute(path_str).resolve()
        assert path.name in features, path
        assert sha256(path) == features[path.name][0], f"JPEG mismatch: {path}"
    print("Validated 30 copied Mac JPEGs and query ORB feature sets", flush=True)

    spec = importlib.util.spec_from_file_location("frozen_bootstrap_control", BACKEND)
    assert spec and spec.loader
    backend = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = backend
    spec.loader.exec_module(backend)
    original_import = backend.dynamic_import
    used = set()

    def import_with_query_features(name, path):
        module = original_import(name, path)
        if Path(path).name == R411_NAME:
            native_compute = module.compute_features

            def controlled_compute(image_path, detector):
                image_path = Path(image_path)
                if image_path.name in features:
                    used.add(image_path.name)
                    return features[image_path.name][1]
                if image_path.name.startswith("blind_frame_"):
                    raise RuntimeError(f"Unexpected query outside 30-feature control: {image_path}")
                return native_compute(image_path, detector)

            module.compute_features = controlled_compute
        return module

    backend.dynamic_import = import_with_query_features
    sys.argv = [str(BACKEND),
                "--repo-root", str(ROOT),
                "--run-root", str(run),
                "--architecture-contract", str(contract),
                "--expected-contract-sha256", ARCHITECTURE_SHA,
                "--out-root", str(out),
                "--candidate-csv", str(candidate_csv),
                "--relative-csv", str(relative_csv),
                "--manifest-csv", str(run / "metadata/blind_query_manifest.csv"),
                "--max-query-id", "30"]
    backend.main()
    assert len(used) == 30, f"R5.1 used {len(used)} of 30 Mac query features"
    print("Mac query features used:", len(used), "| Jetson tile features native")
    print("R5.1 report:", out / "r5_1_minimum_confident_bootstrap_v2_report.json")
    timeline = pd.read_csv(out / "r5_1_blind_policy_timeline.csv")
    subset = timeline[(timeline["policy"] == "activate_half_track_half") &
                      (timeline["update_query_id"].isin([28, 29, 30]))]
    print(subset[["update_query_id", "minimum_innovation_m", "consistency_streak_after",
                  "mode_after", "matured_now"]].to_string(index=False))


if __name__ == "__main__":
    main()
