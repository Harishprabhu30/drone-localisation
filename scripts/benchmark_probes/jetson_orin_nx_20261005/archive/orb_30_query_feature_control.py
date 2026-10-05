"""Blind, isolated Stage 06 rerank using Mac query ORB features on Jetson.

Export the 30 query features on Mac, then run the unmodified Stage 06 module
on Jetson with only query features replaced. Tile features remain Jetson native.
The JPEG SHA check prevents accidental cross-run image mixing.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_module(root):
    source = root / "scripts/villoc/s8_12e1_top20_verifier_reranker.py"
    spec = importlib.util.spec_from_file_location("frozen_stage06_control", source)
    assert spec and spec.loader, source
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert digest(source) == "c2ed7e289719de457831f52d54b1d58ebf9c79b4cbb14f27105313af9e88fb3b", (
        "Stage 06 source changed; review this probe before running"
    )
    return module


def resolve(value, root):
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def export(args, stage, root):
    rows = pd.read_csv(args.mac_verifier_csv)
    assert {"query_id", "query_image_resolved"}.issubset(rows.columns)
    rows["query_id"] = pd.to_numeric(rows["query_id"], errors="raise").astype(int)
    queries = rows.loc[rows["query_id"].between(1, 30),
                       ["query_id", "query_image_resolved"]].drop_duplicates("query_id")
    queries = queries.sort_values("query_id")
    assert queries.query_id.tolist() == list(range(1, 31)), (
        f"Expected Mac query IDs 1..30, got {queries.query_id.tolist()}"
    )
    detector = stage.create_detector("orb", 1800)
    arrays = {}
    manifest = []
    filenames = set()
    for _, row in queries.iterrows():
        path = resolve(row["query_image_resolved"], root)
        assert path.is_file(), path
        filename = path.name
        assert filename.startswith("blind_frame_") and filename not in filenames, filename
        filenames.add(filename)
        feat = stage.compute_features(path, detector, "clahe_luma", 1024)
        assert feat.ok and feat.descriptors is not None, (filename, feat.error)
        assert len(feat.keypoints) == len(feat.descriptors)
        arrays[f"{filename}_points"] = np.asarray([kp.pt for kp in feat.keypoints], dtype=np.float32)
        arrays[f"{filename}_descriptors"] = np.asarray(feat.descriptors, dtype=np.uint8)
        arrays[f"{filename}_shape"] = np.asarray(feat.image_shape, dtype=np.int32)
        manifest.append({"id": str(row["query_id"]), "filename": filename,
                         "jpeg_sha256": digest(path), "features": len(feat.keypoints)})
        print("exported", row["query_id"], filename, len(feat.keypoints), flush=True)
    arrays["manifest_json"] = np.asarray(json.dumps(manifest))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print("saved", args.out, "queries", len(manifest))


def compare_tables(mac_csv, jetson_csv):
    mac = pd.read_csv(mac_csv)
    jetson = pd.read_csv(jetson_csv)
    mac["query_id"] = pd.to_numeric(mac["query_id"], errors="raise").astype(int)
    jetson["query_id"] = pd.to_numeric(jetson["query_id"], errors="raise").astype(int)
    mac = mac.loc[mac.query_id.between(1, 30)].copy()
    assert len(mac) == len(jetson) == 600, (
        f"Expected 600 candidate rows for queries 1..30; Mac={len(mac)} Jetson={len(jetson)}"
    )
    keys = ["query_id", "tile_id"]
    columns = keys + ["good_matches", "inliers", "hybrid_rank", "hybrid_score"]
    joined = mac[columns].merge(jetson[columns], on=keys, how="outer",
                                suffixes=("_mac", "_control"), validate="one_to_one", indicator=True)
    assert len(joined) == 600 and (joined["_merge"] == "both").all(), "600 candidate pairs must match"
    print(json.dumps({
        "matching_pairs": len(joined),
        "different_good_matches": int((joined.good_matches_mac != joined.good_matches_control).sum()),
        "different_inliers": int((joined.inliers_mac != joined.inliers_control).sum()),
        "different_hybrid_ranks": int((joined.hybrid_rank_mac != joined.hybrid_rank_control).sum()),
        "max_hybrid_score_difference": float(np.max(np.abs(joined.hybrid_score_mac - joined.hybrid_score_control))),
    }, indent=2))
    for qid in (1, 22, 28, 29, 30):
        subset = joined.loc[joined.query_id == qid]
        for label in ("mac", "control"):
            top = subset.sort_values(f"hybrid_rank_{label}").head(4)
            print("query", qid, label, list(zip(top.tile_id.tolist(),
                  top[f"inliers_{label}"].tolist(), top[f"hybrid_rank_{label}"].tolist())))


def rerank(args, stage, root):
    with np.load(args.mac_features, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"]))
        assert len(manifest) == 30
        features = {}
        for entry in manifest:
            filename = entry["filename"]
            points = archive[f"{filename}_points"]
            descriptors = archive[f"{filename}_descriptors"]
            shape = tuple(int(x) for x in archive[f"{filename}_shape"])
            assert len(points) == len(descriptors) == entry["features"]
            keypoints = [cv2.KeyPoint(float(x), float(y), 31.0) for x, y in points]
            features[filename] = (entry["jpeg_sha256"],
                                  stage.FeaturePack(True, shape, keypoints, descriptors, None))
    assert len(features) == 30
    table = pd.read_csv(args.query_csv)
    assert "image_path" in table.columns and len(table) == 30
    for value in table.image_path:
        path = resolve(value, root)
        assert path.name in features, path
        assert digest(path) == features[path.name][0], f"Mac/Jetson JPEG mismatch: {path}"
    print("validated 30 Mac feature sets against Jetson JPEG hashes", flush=True)
    original = stage.compute_features
    used = set()

    def controlled_features(path, detector, preprocess, resize_long):
        if path.name.startswith("blind_frame_"):
            assert preprocess == "clahe_luma" and resize_long == 1024
            assert path.name in features, path
            used.add(path.name)
            return features[path.name][1]
        return original(path, detector, preprocess, resize_long)

    stage.compute_features = controlled_features
    options = [
        "--config", str(args.config), "--variant", "512_s256",
        "--tag", "dinov2_vits14_img518_center_square_avgpatch_cpu",
        "--repo-root", str(root), "--query-csv", str(args.query_csv),
        "--topk-csv", str(args.topk_csv), "--out-root", str(args.out_root),
        "--blind-only",
    ]
    stage.run(stage.parse_args(options))
    assert len(used) == 30, f"Stage 06 used {len(used)} of 30 Mac query features"
    print("used Mac query features for all", len(used), "queries; Jetson tile features native")
    outputs = list(args.out_root.rglob("s8_12e1_all_candidate_verifier_scores.csv"))
    assert len(outputs) == 1, outputs
    print("candidate_csv", outputs[0])
    if args.mac_verifier_csv:
        compare_tables(args.mac_verifier_csv, outputs[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    exp = sub.add_parser("export", help="Run on Mac with original Mac Stage 06 result")
    exp.add_argument("--mac-verifier-csv", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("rerank", help="Run isolated Stage 06 on Jetson")
    run.add_argument("--mac-features", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--query-csv", type=Path, required=True)
    run.add_argument("--topk-csv", type=Path, required=True)
    run.add_argument("--out-root", type=Path, required=True)
    run.add_argument("--mac-verifier-csv", type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    stage = stage_module(root)
    if args.operation == "export":
        export(args, stage, root)
    else:
        rerank(args, stage, root)


if __name__ == "__main__":
    main()
