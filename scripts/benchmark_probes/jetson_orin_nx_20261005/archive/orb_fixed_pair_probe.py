"""Read-only ORB stage diagnostic on two Mac query / map tile pairs."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def digest(data):
    return hashlib.sha256(data).hexdigest()[:16]


def file_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else Path.cwd() / path


parser = argparse.ArgumentParser()
parser.add_argument("--candidate-csv", type=Path, required=True)
args = parser.parse_args()
source = Path.cwd() / "scripts/villoc/s8_12e1_top20_verifier_reranker.py"
spec = importlib.util.spec_from_file_location("frozen_orb_stage", source)
assert spec is not None and spec.loader is not None
stage = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stage
spec.loader.exec_module(stage)

candidates = pd.read_csv(args.candidate_csv)
print("opencv", cv2.__version__, "threads", cv2.getNumThreads(),
      "optimized", cv2.useOptimized())
for qid, tile in [(1, "sat_000409"), (30, "sat_000046")]:
    row = candidates.loc[
        (pd.to_numeric(candidates.query_id) == qid)
        & (candidates.tile_id.astype(str) == tile)
    ].iloc[0]
    qpath = file_path(row.query_image_resolved)
    tpath = file_path(row.tile_image_resolved)
    print("PAIR", qid, tile, "jpeg_sha", digest(qpath.read_bytes()), digest(tpath.read_bytes()))
    for seed in [None, 7]:
        for repeat in range(1, 4):
            if seed is not None:
                cv2.setRNGSeed(seed)
            detector = stage.create_detector("orb", 1800)
            qgray, qe = stage.read_image_for_verifier(qpath, "clahe_luma", 1024)
            tgray, te = stage.read_image_for_verifier(tpath, "clahe_luma", 1024)
            assert qgray is not None and tgray is not None, (qe, te)
            q = stage.compute_features(qpath, detector, "clahe_luma", 1024)
            t = stage.compute_features(tpath, detector, "clahe_luma", 1024)
            assert q.ok and t.ok, (q.error, t.error)
            if seed is not None:
                cv2.setRNGSeed(seed)
            match = stage.verify_pair(q, t, 0.80, 5.0)
            def feature_hash(f):
                points = np.asarray([k.pt for k in f.keypoints], dtype=np.float32)
                return digest(points.tobytes()), digest(f.descriptors.tobytes())
            print(json.dumps({
                "query_id": qid, "tile_id": tile,
                "seed": seed, "repeat": repeat,
                "gray_sha": [digest(qgray.tobytes()), digest(tgray.tobytes())],
                "keypoints": [len(q.keypoints), len(t.keypoints)],
                "feature_sha": [feature_hash(q), feature_hash(t)],
                "good_matches": match.good_matches, "inliers": match.inliers,
                "query_coverage": match.query_inlier_coverage,
                "tile_coverage": match.sat_inlier_coverage,
                "verifier_score": match.verifier_score,
            }, sort_keys=True))
