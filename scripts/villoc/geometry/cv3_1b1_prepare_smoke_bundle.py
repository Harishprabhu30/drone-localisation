#!/usr/bin/env python3
"""Prepare the exact reference-free 9-pair CV3.1B1 asset bundle on the Mac."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import yaml
from PIL import Image, ImageOps

from uavloc.geometry.aerial_mast3r_smoke import sha256_file

ROOT = REPO_ROOT
DEFAULT_CONFIG = ROOT / "configs/research/crossview_scale_rotation_domain_cv3_1b1.yaml"


def resolve(v: str | Path) -> Path:
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


def resolve_existing(value: Any, *roots: Path) -> Path:
    p = Path(str(value)).expanduser()
    candidates = [p] if p.is_absolute() else [ROOT / p, *[r / p for r in roots]]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(str(value))


def canonical_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode()


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze_bytes(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"Refusing to overwrite differing bundle metadata: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def copy_immutable(src: Path, dst: Path) -> None:
    if dst.exists():
        if sha256_file(src) != sha256_file(dst):
            raise RuntimeError(f"Refusing to overwrite differing bundled asset: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return int(image.width), int(image.height)


def verify_config(cfg: dict[str, Any]) -> None:
    if cfg.get("stage") != "CV3.1B1":
        raise RuntimeError("Expected CV3.1B1 config")
    if int(cfg["sample"]["expected_pairs"]) != 9:
        raise RuntimeError("Smoke pair count changed")
    for key, value in cfg["scientific_boundary"].items():
        if value is not False:
            raise RuntimeError(f"{key} must remain false in CV3.1B1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CV3.1B1 smoke asset bundle")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    verify_config(cfg)

    top20_path = resolve(cfg["lineage"]["top20"])
    if sha256_file(top20_path) != cfg["lineage"]["expected_top20_sha256"]:
        raise RuntimeError("Frozen QV1.4 Top20 bytes changed")

    spec_path = resolve(cfg["lineage"]["spec"])
    manifest_path = resolve(cfg["lineage"]["manifest"])
    spec = yaml.safe_load(spec_path.read_text())
    manifest = pd.read_csv(manifest_path)
    pool = pd.read_csv(top20_path)
    tile_index_path = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
    tiles = pd.read_csv(tile_index_path)

    manifest["query_id"] = pd.to_numeric(manifest["query_id"], errors="raise").astype(int)
    pool["query_id"] = pd.to_numeric(pool["query_id"], errors="raise").astype(int)
    pool["pool_rank"] = pd.to_numeric(pool["pool_rank"], errors="raise").astype(int)
    pool["tile_id"] = pool["tile_id"].astype(str)
    tiles["tile_id"] = tiles["tile_id"].astype(str)

    refs = manifest["reference_available"].astype(str).str.lower().str.strip()
    if not refs.isin(["false", "0", "no", "n", "f"]).all():
        raise RuntimeError("Canonical blind manifest exposes reference")

    qids = [int(x) for x in cfg["sample"]["query_ids"]]
    ranks = [int(x) for x in cfg["sample"]["pool_ranks"]]
    sample = pool[pool["query_id"].isin(qids) & pool["pool_rank"].isin(ranks)].copy()
    sample = sample.sort_values(["query_id", "pool_rank"], kind="mergesort")
    if len(sample) != int(cfg["sample"]["expected_pairs"]):
        raise RuntimeError("CV3.1B1 smoke pair count changed")

    qlookup = manifest.set_index("query_id")
    tlookup = tiles.set_index("tile_id")
    bundle_root = resolve(cfg["bundle"]["root"])
    rows: list[dict[str, Any]] = []

    for row in sample.itertuples(index=False):
        qid, tile_id = int(row.query_id), str(row.tile_id)
        qsrc = resolve_existing(qlookup.loc[qid, "image_path"])
        tsrc = resolve_existing(tlookup.loc[tile_id, "tile_path"], tile_index_path.parent)

        qext = qsrc.suffix.lower() or ".jpg"
        text = tsrc.suffix.lower() or ".jpg"
        qrel = Path("queries") / f"q{qid:04d}{qext}"
        trel = Path("tiles") / f"{tile_id}{text}"
        qdst, tdst = bundle_root / qrel, bundle_root / trel
        copy_immutable(qsrc, qdst)
        copy_immutable(tsrc, tdst)

        qw, qh = image_size(qsrc)
        tw, th = image_size(tsrc)
        tile = tlookup.loc[tile_id]
        rows.append({
            "query_id": qid,
            "pool_rank": int(row.pool_rank),
            "tile_id": tile_id,
            "source_view": str(row.source_view),
            "source_rank": int(row.source_rank),
            "query_relpath": qrel.as_posix(),
            "tile_relpath": trel.as_posix(),
            "query_sha256": sha256_file(qdst),
            "tile_sha256": sha256_file(tdst),
            "query_width": qw,
            "query_height": qh,
            "tile_width": tw,
            "tile_height": th,
            "left_easting": float(tile["left_easting"]),
            "right_easting": float(tile["right_easting"]),
            "bottom_northing": float(tile["bottom_northing"]),
            "top_northing": float(tile["top_northing"]),
        })

    frame = pd.DataFrame(rows)
    forbidden = {"gt_x","gt_y","reference_x","reference_y","error_m"}
    if forbidden.intersection(frame.columns):
        raise RuntimeError("Reference fields leaked into CV3.1B1 bundle")

    manifest_bytes = canonical_csv(frame)
    bundle_manifest_path = bundle_root / cfg["bundle"]["manifest_filename"]
    freeze_bytes(bundle_manifest_path, manifest_bytes)

    report = {
        "stage": "CV3.1B1_BUNDLE",
        "status": "PASS_CV3_1B1_REFERENCE_FREE_SMOKE_BUNDLE",
        "sample": {
            "query_ids": qids,
            "pool_ranks": ranks,
            "pairs": int(len(frame)),
            "unique_queries": int(frame["query_id"].nunique()),
            "unique_tiles": int(frame["tile_id"].nunique()),
        },
        "scientific_boundary": {
            "reference_loaded": False,
            "oracle_labels_loaded": False,
            "reference_fields_in_bundle": False,
            "accuracy_claim": False,
        },
        "provenance": {
            "config_sha256": sha256_file(cfg_path),
            "top20_sha256": sha256_file(top20_path),
            "canonical_manifest_sha256": sha256_file(manifest_path),
            "tile_index_sha256": sha256_file(tile_index_path),
            "bundle_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
    }
    report_path = bundle_root / cfg["bundle"]["report_filename"]
    freeze_bytes(report_path, canonical_json(report))

    print("Status:", report["status"])
    print("Pairs:", len(frame))
    print("Unique queries:", frame["query_id"].nunique())
    print("Unique tiles:", frame["tile_id"].nunique())
    print("Top20 SHA256:", report["provenance"]["top20_sha256"])
    print("Bundle manifest SHA256:", report["provenance"]["bundle_manifest_sha256"])
    print("Reference loaded: False")
    print("Bundle root:", bundle_root)


if __name__ == "__main__":
    main()
