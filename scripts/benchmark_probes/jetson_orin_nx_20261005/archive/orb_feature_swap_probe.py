"""Export and cross-swap frozen Stage 06 ORB features for two image pairs."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PAIRS = ((1, "sat_000409"), (30, "sat_000046"))


def stage_module(root):
    source = root / "scripts/villoc/s8_12e1_top20_verifier_reranker.py"
    spec = importlib.util.spec_from_file_location("frozen_orb_stage", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source.read_bytes()).hexdigest()


def path_from(value, root):
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def feature_arrays(features, prefix):
    assert features.ok and features.descriptors is not None, features.error
    return {
        f"{prefix}_points": np.asarray([k.pt for k in features.keypoints], dtype=np.float32),
        f"{prefix}_descriptors": np.asarray(features.descriptors, dtype=np.uint8),
        f"{prefix}_shape": np.asarray(features.image_shape, dtype=np.int32),
    }


def feature_pack(stage, archive, prefix):
    points = archive[f"{prefix}_points"]
    descriptors = archive[f"{prefix}_descriptors"]
    shape = tuple(int(x) for x in archive[f"{prefix}_shape"])
    assert len(points) == len(descriptors)
    keypoints = [cv2.KeyPoint(float(x), float(y), 31.0) for x, y in points]
    return stage.FeaturePack(True, shape, keypoints, descriptors, None)


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)
export = sub.add_parser("export")
export.add_argument("--candidate-csv", type=Path, required=True)
export.add_argument("--out", type=Path, required=True)
compare = sub.add_parser("compare")
compare.add_argument("--mac-features", type=Path, required=True)
compare.add_argument("--jetson-features", type=Path, required=True)
args = parser.parse_args()
root = Path.cwd().resolve()
stage, source_sha = stage_module(root)

if args.command == "export":
    table = pd.read_csv(args.candidate_csv)
    arrays = {}
    for qid, tile in PAIRS:
        row = table.loc[
            (pd.to_numeric(table["query_id"]) == qid)
            & (table["tile_id"].astype(str) == tile)
        ].iloc[0]
        detector = stage.create_detector("orb", 1800)
        for suffix, col in (("q", "query_image_resolved"), ("t", "tile_image_resolved")):
            path = path_from(row[col], root)
            features = stage.compute_features(path, detector, "clahe_luma", 1024)
            arrays.update(feature_arrays(features, f"q{qid}_{suffix}"))
            print(qid, tile, suffix, path.name, len(features.keypoints))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print("source_sha256", source_sha, "saved", args.out)
else:
    with np.load(args.mac_features, allow_pickle=False) as mac, np.load(
        args.jetson_features, allow_pickle=False
    ) as jetson:
        for qid, tile in PAIRS:
            for q_name, q_archive in (("Mac", mac), ("Jetson", jetson)):
                for t_name, t_archive in (("Mac", mac), ("Jetson", jetson)):
                    q = feature_pack(stage, q_archive, f"q{qid}_q")
                    t = feature_pack(stage, t_archive, f"q{qid}_t")
                    result = stage.verify_pair(q, t, 0.80, 5.0)
                    print(json.dumps({
                        "query_id": qid,
                        "tile_id": tile,
                        "query_features": q_name,
                        "tile_features": t_name,
                        "good_matches": result.good_matches,
                        "inliers": result.inliers,
                        "query_coverage": result.query_inlier_coverage,
                        "tile_coverage": result.sat_inlier_coverage,
                        "verifier_score": result.verifier_score,
                    }, sort_keys=True))
