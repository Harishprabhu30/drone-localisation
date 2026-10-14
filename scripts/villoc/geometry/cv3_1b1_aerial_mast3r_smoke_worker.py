#!/usr/bin/env python3
"""Child worker for the CV3.1B1 nine-pair aerial-MASt3R smoke."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps

from uavloc.geometry.aerial_mast3r_smoke import (
    finite_pair,
    json_safe,
    mast3r_image_geometry,
    processed_to_canonical_xy,
    tile_pixel_to_map,
)


def memavailable_bytes() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return -1


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return int(image.width), int(image.height)


def append_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(json_safe(payload), sort_keys=True, allow_nan=False) + "\n"
        )
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    bundle_root = args.bundle_root.resolve()
    mast3r_root = (REPO_ROOT / cfg["upstream"]["mast3r_root"]).resolve()
    if str(mast3r_root) not in sys.path:
        sys.path.insert(0, str(mast3r_root))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from mast3r.fast_nn import extract_correspondences_nonsym
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("Torch default dtype changed")

    manifest = pd.read_csv(bundle_root / cfg["bundle"]["manifest_filename"])
    expected_pairs = int(cfg["sample"]["expected_pairs"])
    if len(manifest) != expected_pairs:
        raise RuntimeError("Bundle pair count changed")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = AsymmetricMASt3R.from_pretrained(str(args.snapshot_dir))
    model = model.to("cuda").eval()
    torch.cuda.synchronize()
    model_load_s = float(time.perf_counter() - load_start)
    model_parameter_count = int(sum(p.numel() for p in model.parameters()))
    model_parameter_bytes = int(sum(p.numel() * p.element_size() for p in model.parameters()))

    rows = []
    for pair_index, row in enumerate(manifest.itertuples(index=False), start=1):
        qpath = bundle_root / str(row.query_relpath)
        tpath = bundle_root / str(row.tile_relpath)
        qw, qh = image_size(qpath)
        tw, th = image_size(tpath)
        if (qw, qh) != (int(row.query_width), int(row.query_height)):
            raise RuntimeError("Bundled query dimensions changed")
        if (tw, th) != (int(row.tile_width), int(row.tile_height)):
            raise RuntimeError("Bundled tile dimensions changed")

        qgeom = mast3r_image_geometry(
            qw, qh,
            size=int(cfg["model"]["image_load_size"]),
            square_ok=bool(cfg["model"]["square_ok"]),
        )
        tgeom = mast3r_image_geometry(
            tw, th,
            size=int(cfg["model"]["image_load_size"]),
            square_ok=bool(cfg["model"]["square_ok"]),
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free_before, total_before = torch.cuda.mem_get_info()
        system_before = memavailable_bytes()
        total_start = time.perf_counter()

        images = load_images(
            [str(qpath), str(tpath)],
            size=int(cfg["model"]["image_load_size"]),
            square_ok=bool(cfg["model"]["square_ok"]),
            verbose=False,
        )
        qshape = tuple(map(int, images[0]["true_shape"][0]))
        tshape = tuple(map(int, images[1]["true_shape"][0]))
        if qshape != (qgeom.processed_height, qgeom.processed_width):
            raise RuntimeError(f"Query processed shape mismatch: {qshape}")
        if tshape != (tgeom.processed_height, tgeom.processed_width):
            raise RuntimeError(f"Tile processed shape mismatch: {tshape}")

        infer_start = time.perf_counter()
        output = inference(
            [tuple(images)],
            model,
            "cuda",
            batch_size=int(cfg["model"]["batch_size"]),
            verbose=False,
        )
        torch.cuda.synchronize()
        inference_s = float(time.perf_counter() - infer_start)
        peak_after_inference = int(torch.cuda.max_memory_allocated())

        pred1, pred2 = output["pred1"], output["pred2"]
        if "desc" not in pred1 or "desc_conf" not in pred1 or "desc" not in pred2 or "desc_conf" not in pred2:
            raise RuntimeError("Aerial-MASt3R predictions do not expose desc/desc_conf")

        match_start = time.perf_counter()
        corres = extract_correspondences_nonsym(
            pred1["desc"][0],
            pred2["desc"][0],
            pred1["desc_conf"][0],
            pred2["desc_conf"][0],
            device="cuda",
            subsample=int(cfg["matching"]["subsample"]),
            pixel_tol=int(cfg["matching"]["pixel_tol"]),
        )
        torch.cuda.synchronize()
        matching_s = float(time.perf_counter() - match_start)
        peak_after_matching = int(torch.cuda.max_memory_allocated())

        qproc = corres[0].detach().cpu().numpy().astype(np.float32)
        tproc = corres[1].detach().cpu().numpy().astype(np.float32)
        conf = corres[2].detach().cpu().numpy().astype(float)
        raw_correspondences = int(len(conf))

        conf_mask = conf >= float(cfg["matching"]["match_confidence"])
        border = float(cfg["matching"]["processed_border_px"])
        qvalid = (
            (qproc[:, 0] >= border) &
            (qproc[:, 0] < qgeom.processed_width - border) &
            (qproc[:, 1] >= border) &
            (qproc[:, 1] < qgeom.processed_height - border)
        )
        tvalid = (
            (tproc[:, 0] >= border) &
            (tproc[:, 0] < tgeom.processed_width - border) &
            (tproc[:, 1] >= border) &
            (tproc[:, 1] < tgeom.processed_height - border)
        )
        keep = conf_mask & qvalid & tvalid
        qproc = qproc[keep]
        tproc = tproc[keep]
        kept_conf = conf[keep]
        qpts = processed_to_canonical_xy(qproc, qgeom)
        tpts = processed_to_canonical_xy(tproc, tgeom)

        geometry_start = time.perf_counter()
        H = mask = None
        if len(qpts) >= 4:
            H, mask = cv2.findHomography(
                qpts,
                tpts,
                cv2.RANSAC,
                float(cfg["geometry"]["ransac_threshold_px"]),
            )
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(qpts), dtype=bool)
        inliers = int(inlier_mask.sum())
        homography_ok = bool(
            H is not None and inliers >= int(cfg["geometry"]["min_homography_inliers"])
        )

        u = v = e = n = math.nan
        finite_projection = False
        projected_inside_tile = False
        if homography_ok:
            center = np.float32([[[qw / 2.0, qh / 2.0]]])
            projected = cv2.perspectiveTransform(
                center,
                np.asarray(H, dtype=np.float64),
            ).reshape(2)
            u, v = float(projected[0]), float(projected[1])
            e, n = tile_pixel_to_map(
                u,
                v,
                (th, tw),
                (
                    float(row.left_easting),
                    float(row.right_easting),
                    float(row.bottom_northing),
                    float(row.top_northing),
                ),
            )
            finite_projection = finite_pair(e, n)
            projected_inside_tile = bool(
                finite_projection and 0 <= u < tw and 0 <= v < th
            )
        geometry_s = float(time.perf_counter() - geometry_start)

        total_s = float(time.perf_counter() - total_start)
        free_after, _ = torch.cuda.mem_get_info()
        system_after = memavailable_bytes()
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

        result = {
            "query_id": int(row.query_id),
            "pool_rank": int(row.pool_rank),
            "tile_id": str(row.tile_id),
            "source_view": str(row.source_view),
            "source_rank": int(row.source_rank),
            "raw_correspondences": raw_correspondences,
            "filtered_correspondences": int(len(qpts)),
            "inliers": inliers,
            "inlier_ratio": float(inliers / max(1, len(qpts))),
            "homography_ok": homography_ok,
            "finite_projection": finite_projection,
            "projected_inside_tile": projected_inside_tile,
            "projected_tile_u_px": float(u),
            "projected_tile_v_px": float(v),
            "projected_easting": float(e),
            "projected_northing": float(n),
            "confidence_sum": float(np.sum(kept_conf)) if len(kept_conf) else 0.0,
            "confidence_mean": float(np.mean(kept_conf)) if len(kept_conf) else 0.0,
            "confidence_median": float(np.median(kept_conf)) if len(kept_conf) else 0.0,
            "query_processed_width": qgeom.processed_width,
            "query_processed_height": qgeom.processed_height,
            "tile_processed_width": tgeom.processed_width,
            "tile_processed_height": tgeom.processed_height,
            "inference_ms": inference_s * 1000.0,
            "matching_ms": matching_s * 1000.0,
            "geometry_ms": geometry_s * 1000.0,
            "total_ms": total_s * 1000.0,
            "cuda_free_gib_before_pair": float(free_before / 2**30),
            "cuda_free_gib_after_pair_before_cleanup": float(free_after / 2**30),
            "system_available_gib_before_pair": float(system_before / 2**30),
            "system_available_gib_after_pair_before_cleanup": float(system_after / 2**30),
            "torch_peak_allocated_gib_pair": float(peak_allocated / 2**30),
            "torch_peak_reserved_gib_pair": float(peak_reserved / 2**30),
            "torch_peak_allocated_gib_after_inference": float(peak_after_inference / 2**30),
            "torch_peak_allocated_gib_after_matching": float(peak_after_matching / 2**30),
        }
        rows.append(result)
        append_trace(args.trace, {
            "event": "pair_complete",
            "pair_index": pair_index,
            **result,
        })

        del output, pred1, pred2, corres, images
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    payload = {
        "status": "PASS_WORKER_CV3_1B1_NINE_PAIR_SMOKE",
        "model_load_seconds": model_load_s,
        "model_parameter_count": model_parameter_count,
        "model_parameter_bytes": model_parameter_bytes,
        "rows": rows,
        "inference_executed": True,
        "reference_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Worker status:", payload["status"])
    print("Pairs completed:", len(rows), "/ 9")
    print("Reference loaded: False")


if __name__ == "__main__":
    main()
