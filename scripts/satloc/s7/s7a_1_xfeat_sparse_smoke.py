#!/usr/bin/env python3
"""S7A.1 — XFeat sparse smoke test on frozen S7A.0 diagnostic pairs.

Runs official VERLab XFeat on the 169 adjacent diagnostic pairs only.
Uses frozen settings: long side 960, top_k 1200, detection threshold 0.05,
MNN cosine 0.82, partial-affine RANSAC threshold 3 px.

No reference coordinates, trajectory errors, retrieval ranks, oracle labels, or
fusion outcomes are loaded. This is an image-only implementation gate.

Command used:

source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

mkdir -p outputs/satloc/reports/s7_relative_frontend
set -o pipefail

python scripts/satloc/s7/s7a_1_xfeat_sparse_smoke.py \
  2>&1 | tee \
  outputs/satloc/reports/s7_relative_frontend/s7a_1_xfeat_sparse_smoke.log
  
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


OFFICIAL_REPOSITORY = "https://github.com/verlab/accelerated_features.git"
PINNED_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
EXPECTED_FRAMES = 1034
EXPECTED_DIAGNOSTIC_PAIRS = 169


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def git_value(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return ""


def sync(torch: Any, device: Any) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def peak_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if platform.system() == "Darwin" else value / 1024.0


def resize_long_side(image: np.ndarray, target: int) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if target <= 0 or long_side == target:
        return image
    scale = target / float(long_side)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, new_size, interpolation=interpolation)


def resolve_image(repo_root: Path, value: Any) -> Path:
    raw = Path(str(value)).expanduser()
    for candidate in (raw, repo_root / raw):
        if candidate.exists():
            return candidate.resolve()
    return raw


def load_sequence(repo_root: Path, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sequence_frame_id", "token0_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Sequence manifest missing columns: {missing}")

    image_column = next(
        (c for c in ("image_path_resolved", "image_path", "resolved_image_path") if c in frame.columns),
        None,
    )
    if image_column is None:
        raise RuntimeError("Sequence manifest has no usable image-path column.")

    frame["sequence_frame_id"] = pd.to_numeric(
        frame["sequence_frame_id"], errors="raise"
    ).astype(int)
    frame["token0_id"] = pd.to_numeric(frame["token0_id"], errors="raise").astype(int)
    frame = frame.sort_values("sequence_frame_id", kind="mergesort").reset_index(drop=True)

    if len(frame) != EXPECTED_FRAMES:
        raise RuntimeError(f"Expected {EXPECTED_FRAMES} frames, found {len(frame)}.")
    if not np.array_equal(frame["sequence_frame_id"].to_numpy(), np.arange(1034)):
        raise RuntimeError("sequence_frame_id is not contiguous 0..1033.")
    if not np.array_equal(frame["token0_id"].to_numpy(), np.arange(1, 1035)):
        raise RuntimeError("token0_id is not the canonical range 1..1034.")

    frame["image_path_smoke"] = frame[image_column].map(
        lambda value: str(resolve_image(repo_root, value))
    )
    missing_paths = [Path(p) for p in frame["image_path_smoke"] if not Path(p).exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} UAV images are missing. First: {missing_paths[0]}"
        )
    return frame


def load_diagnostic_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "diagnostic_pair_order",
        "pair_number",
        "frame_index_a",
        "frame_index_b",
        "token0_a",
        "token0_b",
        "primary_scene",
        "secondary_scene",
        "selection_role",
        "range_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Diagnostic manifest missing columns: {missing}")

    numeric = [
        "diagnostic_pair_order",
        "pair_number",
        "frame_index_a",
        "frame_index_b",
        "token0_a",
        "token0_b",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame = frame.sort_values("diagnostic_pair_order", kind="mergesort").reset_index(drop=True)

    if len(frame) != EXPECTED_DIAGNOSTIC_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_DIAGNOSTIC_PAIRS} diagnostic pairs, found {len(frame)}."
        )
    if frame["pair_number"].duplicated().any():
        raise RuntimeError("Diagnostic pair numbers are duplicated.")
    if not (frame["frame_index_b"] == frame["frame_index_a"] + 1).all():
        raise RuntimeError("Diagnostic manifest contains non-adjacent pairs.")
    return frame


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def load_orb_subset(path: Path, diagnostic: pd.DataFrame) -> pd.DataFrame:
    orb = pd.read_csv(path)
    required = {
        "stride",
        "pair_number",
        "status",
        "affine_ok",
        "good_matches",
        "inliers",
        "inlier_ratio",
        "affine_scale",
        "affine_rotation_deg",
        "elapsed_ms",
    }
    missing = sorted(required - set(orb.columns))
    if missing:
        raise RuntimeError(f"ORB diagnostics missing columns: {missing}")

    orb["stride"] = pd.to_numeric(orb["stride"], errors="raise").astype(int)
    orb["pair_number"] = pd.to_numeric(orb["pair_number"], errors="raise").astype(int)
    orb = orb[orb["stride"] == 1].copy()
    orb["affine_ok"] = bool_series(orb["affine_ok"])

    labels = diagnostic[
        ["pair_number", "primary_scene", "secondary_scene", "selection_role", "range_id"]
    ]
    selected = labels.merge(orb, on="pair_number", how="left", validate="one_to_one")
    if selected["status"].isna().any():
        raise RuntimeError("Some diagnostic pairs are absent from frozen ORB outputs.")
    return selected


def load_xfeat(
    repo: Path,
    device_name: str,
    top_k: int,
    detection_threshold: float,
) -> tuple[Any, Any, Any, str]:
    module_path = repo / "modules" / "xfeat.py"
    weights_path = repo / "weights" / "xfeat.pt"
    if not module_path.exists():
        raise FileNotFoundError(f"Official XFeat module not found: {module_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Official XFeat weights not found: {weights_path}")

    sys.path.insert(0, str(repo))
    for name in list(sys.modules):
        if name == "modules" or name.startswith("modules."):
            del sys.modules[name]

    try:
        torch = importlib.import_module("torch")
        XFeat = getattr(importlib.import_module("modules.xfeat"), "XFeat")
    except Exception as exc:
        raise RuntimeError(f"Could not import official XFeat: {exc}") from exc

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = XFeat(
        weights=str(weights_path),
        top_k=top_k,
        detection_threshold=detection_threshold,
    )
    model.dev = device
    model.net = model.net.to(device).eval()
    model.eval()
    return torch, model, device, git_value(repo, "rev-parse", "HEAD")


def extract_cache(
    torch: Any,
    model: Any,
    device: Any,
    sequence: pd.DataFrame,
    frame_indices: list[int],
    resize_long: int,
    top_k: int,
    detection_threshold: float,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame, float]:
    cache: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()

    for progress, frame_index in enumerate(frame_indices, start=1):
        row = sequence.iloc[frame_index]
        image_path = Path(row["image_path_smoke"])
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"OpenCV could not read {image_path}")
        gray = resize_long_side(gray, resize_long)
        tensor = (
            torch.from_numpy(gray)
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            / 255.0
        )

        sync(torch, device)
        start = time.perf_counter()
        output = model.detectAndCompute(
            tensor, top_k=top_k, detection_threshold=detection_threshold
        )[0]
        sync(torch, device)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)

        cache[frame_index] = {
            "keypoints": output["keypoints"].detach(),
            "descriptors": output["descriptors"].detach(),
            "scores": output["scores"].detach(),
            "width": int(gray.shape[1]),
            "height": int(gray.shape[0]),
        }
        rows.append(
            {
                "sequence_frame_id": frame_index,
                "token0_id": int(row["token0_id"]),
                "keypoints": int(len(output["keypoints"])),
                "feature_time_ms": float(elapsed_ms),
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
                "read_ok": True,
            }
        )
        if progress % 25 == 0 or progress == len(frame_indices):
            print(f"XFeat cache: {progress}/{len(frame_indices)} frames")

    return cache, pd.DataFrame(rows), time.perf_counter() - wall_start


def match_pair(
    torch: Any,
    model: Any,
    device: Any,
    feature_a: dict[str, Any],
    feature_b: dict[str, Any],
    min_cossim: float,
    ransac_threshold: float,
) -> dict[str, Any]:
    desc_a = feature_a["descriptors"]
    desc_b = feature_b["descriptors"]
    key_a = feature_a["keypoints"]
    key_b = feature_b["keypoints"]

    if len(desc_a) == 0 or len(desc_b) == 0:
        return {
            "status": "no_descriptors",
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
            "matching_ransac_time_ms": 0.0,
        }

    sync(torch, device)
    start = time.perf_counter()
    idx_a, idx_b = model.match(desc_a, desc_b, min_cossim=min_cossim)
    sync(torch, device)
    points_a = key_a[idx_a].detach().cpu().numpy().astype(np.float32)
    points_b = key_b[idx_b].detach().cpu().numpy().astype(np.float32)

    if len(points_a) < 3:
        return {
            "status": "too_few_matches",
            "matches": int(len(points_a)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
            "matching_ransac_time_ms": 1000.0 * (time.perf_counter() - start),
        }

    affine, mask = cv2.estimateAffinePartial2D(
        points_a,
        points_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    if affine is None or mask is None:
        return {
            "status": "affine_failed",
            "matches": int(len(points_a)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "affine_ok": False,
            "matching_ransac_time_ms": float(elapsed_ms),
        }

    mask = np.asarray(mask).ravel().astype(bool)
    inliers = int(mask.sum())
    ratio = inliers / max(len(points_a), 1)
    a00, a01, tx = [float(v) for v in affine[0]]
    a10, a11, ty = [float(v) for v in affine[1]]
    scale = 0.5 * (math.hypot(a00, a10) + math.hypot(a01, a11))
    rotation_deg = math.degrees(math.atan2(a10, a00))
    good = (
        len(points_a) >= 30
        and inliers >= 20
        and ratio >= 0.35
        and 0.70 <= scale <= 1.40
    )
    return {
        "status": "good" if good else "weak",
        "matches": int(len(points_a)),
        "inliers": inliers,
        "inlier_ratio": float(ratio),
        "affine_ok": True,
        "affine_a00": a00,
        "affine_a01": a01,
        "affine_a10": a10,
        "affine_a11": a11,
        "affine_tx_px": tx,
        "affine_ty_px": ty,
        "affine_scale": float(scale),
        "affine_rotation_deg": float(rotation_deg),
        "matching_ransac_time_ms": float(elapsed_ms),
    }


def run_pairs(
    torch: Any,
    model: Any,
    device: Any,
    diagnostic: pd.DataFrame,
    cache: dict[int, dict[str, Any]],
    min_cossim: float,
    ransac_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for progress, pair in enumerate(diagnostic.itertuples(index=False), start=1):
        result = match_pair(
            torch,
            model,
            device,
            cache[int(pair.frame_index_a)],
            cache[int(pair.frame_index_b)],
            min_cossim,
            ransac_threshold,
        )
        rows.append(
            {
                "diagnostic_pair_order": int(pair.diagnostic_pair_order),
                "pair_number": int(pair.pair_number),
                "frame_index_a": int(pair.frame_index_a),
                "frame_index_b": int(pair.frame_index_b),
                "token0_a": int(pair.token0_a),
                "token0_b": int(pair.token0_b),
                "range_id": str(pair.range_id),
                "primary_scene": str(pair.primary_scene),
                "secondary_scene": str(pair.secondary_scene),
                "selection_role": str(pair.selection_role),
                **result,
            }
        )
        if progress % 25 == 0 or progress == len(diagnostic):
            print(f"XFeat pairs: {progress}/{len(diagnostic)}")
    return pd.DataFrame(rows)


def finite(values: pd.Series, fn: Any) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(fn(array)) if len(array) else float("nan")


def summarize(frame: pd.DataFrame, method: str, match_col: str, time_col: str) -> dict[str, Any]:
    ok = bool_series(frame["affine_ok"])
    successful = frame[ok]
    return {
        "method": method,
        "pairs": int(len(frame)),
        "affine_successes": int(ok.sum()),
        "affine_success_rate": float(ok.mean()),
        "good_quality_pairs": int((frame["status"].astype(str) == "good").sum()),
        "good_quality_rate": float((frame["status"].astype(str) == "good").mean()),
        "matches_median": finite(successful[match_col], np.median),
        "inliers_median": finite(successful["inliers"], np.median),
        "inlier_ratio_median": finite(successful["inlier_ratio"], np.median),
        "inlier_ratio_p05": finite(successful["inlier_ratio"], lambda x: np.percentile(x, 5)),
        "abs_rotation_deg_p95": finite(successful["affine_rotation_deg"].abs(), lambda x: np.percentile(x, 95)),
        "abs_scale_error_p95": finite((successful["affine_scale"] - 1.0).abs(), lambda x: np.percentile(x, 95)),
        "matching_ransac_time_ms_mean": finite(successful[time_col], np.mean),
    }


def scene_summary(xfeat: pd.DataFrame, orb: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scene, group in xfeat.groupby("primary_scene", sort=True):
        rows.append({"primary_scene": scene, **summarize(group, "xfeat", "matches", "matching_ransac_time_ms")})
    for scene, group in orb.groupby("primary_scene", sort=True):
        rows.append({"primary_scene": scene, **summarize(group, "orb", "good_matches", "elapsed_ms")})
    return pd.DataFrame(rows).sort_values(["primary_scene", "method"], kind="mergesort")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--xfeat-repo", type=Path, default=Path("third_party/accelerated_features"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-unpinned-xfeat", action="store_true")
    parser.add_argument("--resize-long", type=int, default=960)
    parser.add_argument("--top-k", type=int, default=1200)
    parser.add_argument("--detection-threshold", type=float, default=0.05)
    parser.add_argument("--min-cossim", type=float, default=0.82)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    args = parser.parse_args()

    if args.resize_long != 960 or args.top_k != 1200:
        raise RuntimeError("S7A.0 froze resize_long=960 and top_k=1200.")
    if abs(args.detection_threshold - 0.05) > 1e-12:
        raise RuntimeError("S7A.1 freezes official detection_threshold=0.05.")
    if abs(args.min_cossim - 0.82) > 1e-12:
        raise RuntimeError("S7A.1 freezes official min_cossim=0.82.")
    if abs(args.ransac_threshold - 3.0) > 1e-12:
        raise RuntimeError("S7A.0 froze RANSAC threshold=3.0 px.")

    root = args.repo_root.resolve()
    xfeat_repo = resolve(root, args.xfeat_repo)
    sequence_path = root / "outputs/satloc/metadata/s6a_relative_motion/s6a_sequence_manifest.csv"
    orb_path = root / "outputs/satloc/metadata/s6a_relative_motion/s6a1_orb_affine_pair_diagnostics.csv"
    diagnostic_path = root / "outputs/satloc/metadata/s7_relative_frontend/s7a_0_stratified_diagnostic_pairs.csv"
    protocol_path = root / "outputs/satloc/metadata/s7_relative_frontend/s7a_0_protocol_manifest.json"
    for path in (sequence_path, orb_path, diagnostic_path, protocol_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen input: {path}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PASS_FROZEN":
        raise RuntimeError("S7A.0 protocol is not PASS_FROZEN.")

    sequence = load_sequence(root, sequence_path)
    diagnostic = load_diagnostic_pairs(diagnostic_path)
    orb_subset = load_orb_subset(orb_path, diagnostic)
    torch, model, device, actual_commit = load_xfeat(
        xfeat_repo, args.device, args.top_k, args.detection_threshold
    )
    if actual_commit and actual_commit != PINNED_COMMIT and not args.allow_unpinned_xfeat:
        raise RuntimeError(
            "XFeat checkout is not at the frozen commit.\n"
            f"Expected: {PINNED_COMMIT}\nFound:    {actual_commit}\n"
            "Checkout the pinned commit or use --allow-unpinned-xfeat explicitly."
        )

    unique_frames = sorted(
        set(diagnostic["frame_index_a"].tolist())
        | set(diagnostic["frame_index_b"].tolist())
    )
    metadata_dir = ensure_dir(root / "outputs/satloc/metadata/s7_relative_frontend")
    reports_dir = ensure_dir(root / "outputs/satloc/reports/s7_relative_frontend")

    wall_start = time.perf_counter()
    cache, feature_df, feature_seconds = extract_cache(
        torch,
        model,
        device,
        sequence,
        unique_frames,
        args.resize_long,
        args.top_k,
        args.detection_threshold,
    )
    xfeat_pairs = run_pairs(
        torch,
        model,
        device,
        diagnostic,
        cache,
        args.min_cossim,
        args.ransac_threshold,
    )
    wall_seconds = time.perf_counter() - wall_start

    orb_summary = summarize(orb_subset, "orb", "good_matches", "elapsed_ms")
    xfeat_summary = summarize(xfeat_pairs, "xfeat", "matches", "matching_ransac_time_ms")
    xfeat_summary.update(
        {
            "unique_frames_extracted": int(len(feature_df)),
            "feature_time_ms_mean": finite(feature_df["feature_time_ms"], np.mean),
            "feature_time_ms_median": finite(feature_df["feature_time_ms"], np.median),
            "feature_cache_seconds": float(feature_seconds),
            "smoke_wall_clock_seconds": float(wall_seconds),
            "amortized_total_frontend_ms_per_pair": float(1000.0 * wall_seconds / len(xfeat_pairs)),
            "peak_process_memory_mb": float(peak_memory_mb()),
        }
    )

    gate = bool(
        xfeat_summary["affine_success_rate"] >= orb_summary["affine_success_rate"] - 0.05
        and xfeat_summary["good_quality_rate"] >= 0.80
        and xfeat_summary["inlier_ratio_p05"] >= 0.35
    )
    scenes = scene_summary(xfeat_pairs, orb_subset)

    feature_path = metadata_dir / "s7a_1_xfeat_frame_features.csv"
    pair_path = metadata_dir / "s7a_1_xfeat_pair_diagnostics.csv"
    summary_csv_path = metadata_dir / "s7a_1_smoke_method_summary.csv"
    scene_path = metadata_dir / "s7a_1_smoke_scene_summary.csv"
    json_path = reports_dir / "s7a_1_xfeat_sparse_smoke.json"
    report_path = reports_dir / "s7a_1_xfeat_sparse_smoke_report.md"

    feature_df.to_csv(feature_path, index=False)
    xfeat_pairs.to_csv(pair_path, index=False)
    pd.DataFrame([orb_summary, xfeat_summary]).to_csv(summary_csv_path, index=False)
    scenes.to_csv(scene_path, index=False)

    payload = {
        "generated_utc": utc_now(),
        "stage": "S7A.1",
        "status": "PASS" if gate else "FAIL_CLOSE_XFEAT",
        "official_xfeat": {
            "repository": OFFICIAL_REPOSITORY,
            "expected_commit": PINNED_COMMIT,
            "actual_commit": actual_commit,
            "checkout_path": str(xfeat_repo.relative_to(root)),
        },
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "device": str(device),
            "platform": platform.platform(),
        },
        "configuration": {
            "resize_long": args.resize_long,
            "top_k": args.top_k,
            "detection_threshold": args.detection_threshold,
            "min_cossim": args.min_cossim,
            "ransac_threshold": args.ransac_threshold,
            "diagnostic_pairs": int(len(diagnostic)),
            "unique_frames": int(len(unique_frames)),
        },
        "orb_same_pair_summary": orb_summary,
        "xfeat_summary": xfeat_summary,
        "comparison": {
            "affine_success_delta_xfeat_minus_orb": float(xfeat_summary["affine_success_rate"] - orb_summary["affine_success_rate"]),
            "good_quality_delta_xfeat_minus_orb": float(xfeat_summary["good_quality_rate"] - orb_summary["good_quality_rate"]),
            "inlier_ratio_median_delta_xfeat_minus_orb": float(xfeat_summary["inlier_ratio_median"] - orb_summary["inlier_ratio_median"]),
            "smoke_gate_pass": gate,
            "smoke_gate_rule": "affine success within 5 percentage points of ORB, good-quality rate >=0.80, and inlier-ratio p05 >=0.35",
        },
        "ground_truth_isolation": "No reference coordinates, trajectory errors, retrieval ranks, oracle labels, or fusion outcomes were loaded.",
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    scene_table = scenes[
        [
            "primary_scene",
            "method",
            "pairs",
            "affine_success_rate",
            "good_quality_rate",
            "inlier_ratio_median",
            "inlier_ratio_p05",
        ]
    ].to_markdown(index=False)
    report = f"""# S7A.1 — XFeat Sparse Smoke Test

Generated: `{payload['generated_utc']}`

## Status

```text
{payload['status']}
```

## Configuration

```text
XFeat commit:        {actual_commit or 'unknown'}
Device:              {device}
Diagnostic pairs:    {len(diagnostic)}
Unique frames:       {len(unique_frames)}
Long side:           {args.resize_long}
Top-k:               {args.top_k}
Detection threshold: {args.detection_threshold}
Minimum cosine:      {args.min_cossim}
RANSAC threshold:    {args.ransac_threshold}
```

## Same-pair result

```text
ORB affine success:        {orb_summary['affine_success_rate']:.4f}
XFeat affine success:      {xfeat_summary['affine_success_rate']:.4f}
ORB good-quality rate:     {orb_summary['good_quality_rate']:.4f}
XFeat good-quality rate:   {xfeat_summary['good_quality_rate']:.4f}
ORB median inlier ratio:   {orb_summary['inlier_ratio_median']:.4f}
XFeat median inlier ratio: {xfeat_summary['inlier_ratio_median']:.4f}
XFeat p05 inlier ratio:    {xfeat_summary['inlier_ratio_p05']:.4f}
XFeat amortized ms/pair:   {xfeat_summary['amortized_total_frontend_ms_per_pair']:.2f}
Smoke gate pass:           {gate}
```

## Scene diagnostics

{scene_table}

A pass authorizes one full traj01 XFeat comparison in S7A.2. A failure closes
XFeat without a threshold sweep.
"""
    report_path.write_text(report, encoding="utf-8")

    print("S7A.1 XFeat Sparse Smoke Test")
    print("-----------------------------")
    print(f"Status:                    {payload['status']}")
    print(f"XFeat commit:              {actual_commit or 'unknown'}")
    print(f"Device:                    {device}")
    print(f"Diagnostic pairs:          {len(diagnostic)}")
    print(f"Unique frames extracted:   {len(unique_frames)}")
    print(f"ORB affine success:        {orb_summary['affine_success_rate']:.4f}")
    print(f"XFeat affine success:      {xfeat_summary['affine_success_rate']:.4f}")
    print(f"ORB good-quality rate:     {orb_summary['good_quality_rate']:.4f}")
    print(f"XFeat good-quality rate:   {xfeat_summary['good_quality_rate']:.4f}")
    print(f"ORB inlier ratio median:   {orb_summary['inlier_ratio_median']:.4f}")
    print(f"XFeat inlier ratio median: {xfeat_summary['inlier_ratio_median']:.4f}")
    print(f"XFeat inlier ratio p05:    {xfeat_summary['inlier_ratio_p05']:.4f}")
    print(f"XFeat feature mean ms:     {xfeat_summary['feature_time_ms_mean']:.2f}")
    print(f"XFeat match+RANSAC ms:     {xfeat_summary['matching_ransac_time_ms_mean']:.2f}")
    print(f"Amortized frontend ms:     {xfeat_summary['amortized_total_frontend_ms_per_pair']:.2f}")
    print(f"Peak process memory MB:    {xfeat_summary['peak_process_memory_mb']:.1f}")
    print(f"Smoke gate pass:           {gate}")
    print(f"Pair diagnostics:          {pair_path.relative_to(root)}")
    print(f"Scene summary:             {scene_path.relative_to(root)}")
    print(f"JSON summary:              {json_path.relative_to(root)}")
    print(f"Report:                    {report_path.relative_to(root)}")
    return 0 if gate else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("S7A.1 XFeat Sparse Smoke Test", file=sys.stderr)
        print("-----------------------------", file=sys.stderr)
        print("Status: BLOCKED", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        raise
