"""Compare OpenCV preprocessing and ORB at fixed image-pair boundaries."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def image_steps(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    stages = {"decoded_bgr": sha(image)}
    h, w = image.shape[:2]
    scale = 1024.0 / max(h, w)
    if scale < 1.0:
        image = cv2.resize(
            image, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    stages["resized_bgr"] = sha(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    stages["ycrcb"] = sha(ycrcb)
    stages["luma"] = sha(ycrcb[:, :, 0])
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ycrcb[:, :, 0])
    stages["clahe_luma"] = sha(gray)
    return stages


def feature_info(features):
    assert features.ok, features.error
    points = np.asarray([key.pt for key in features.keypoints], dtype=np.float32)
    return {
        "keypoints": len(features.keypoints),
        "points_sha256": sha(points),
        "descriptors_sha256": sha(features.descriptors),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--candidate-csv", type=Path, required=True)
args = parser.parse_args()
root = Path.cwd().resolve()
source = root / "scripts/villoc/s8_12e1_top20_verifier_reranker.py"
spec = importlib.util.spec_from_file_location("frozen_orb_stage", source)
assert spec is not None and spec.loader is not None
stage = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stage
spec.loader.exec_module(stage)

candidates = pd.read_csv(args.candidate_csv)
print(json.dumps({
    "opencv": cv2.__version__,
    "default_threads": cv2.getNumThreads(),
    "optimized": cv2.useOptimized(),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}))

for thread_mode in ("default", "one_thread"):
    if thread_mode == "one_thread":
        cv2.setNumThreads(1)
    for query_id, tile_id in ((1, "sat_000409"), (30, "sat_000046")):
        row = candidates.loc[
            (pd.to_numeric(candidates["query_id"]) == query_id)
            & (candidates["tile_id"].astype(str) == tile_id)
        ].iloc[0]
        paths = []
        for column in ("query_image_resolved", "tile_image_resolved"):
            p = Path(str(row[column]))
            paths.append(p if p.is_absolute() else root / p)
        query_path, tile_path = paths
        detector = stage.create_detector("orb", 1800)
        q_features = stage.compute_features(query_path, detector, "clahe_luma", 1024)
        t_features = stage.compute_features(tile_path, detector, "clahe_luma", 1024)
        match = stage.verify_pair(q_features, t_features, 0.80, 5.0)
        report = {
            "mode": thread_mode,
            "threads": cv2.getNumThreads(),
            "query_id": query_id,
            "tile_id": tile_id,
            "jpeg_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
            "query_steps": image_steps(query_path),
            "tile_steps": image_steps(tile_path),
            "query_orb": feature_info(q_features),
            "tile_orb": feature_info(t_features),
            "good_matches": match.good_matches,
            "inliers": match.inliers,
            "query_coverage": match.query_inlier_coverage,
            "tile_coverage": match.sat_inlier_coverage,
            "verifier_score": match.verifier_score,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
