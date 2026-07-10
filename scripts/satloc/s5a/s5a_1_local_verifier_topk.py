#!/usr/bin/env python3
"""
S5A.1 — Local verifier interface inside PHOG top-K

Purpose
-------
Runs a lightweight OpenCV local matcher/verifier inside the existing S4C PHOG top-K
candidate pool. This script is intentionally designed as the plumbing/interface block
before adding heavier learned matchers such as LightGlue, LoFTR, or RoMA.

Locked rule
-----------
Reference coordinates / error columns are evaluation-only. They are never used while
computing local verifier scores or ranking candidates.

Coode executed:
---------------
export PYTHONPATH=$PWD/src
python scripts/satloc/s5a/s5a_1_local_verifier_topk.py \
  --manifest outputs/satloc/metadata/s5a_learned_local_verifier/s5a0_failure_group_benchmark_manifest.csv \
  --sat-index outputs/satloc/metadata/satellite_tiles_index_enriched.csv \
  --top-k 50 \
  --backend akaze \
  --preprocess clahe_gray \
  --resize-long 768

Outputs
-------
outputs/satloc/metadata/s5a_learned_local_verifier/s5a1_local_verifier_candidate_scores.csv
outputs/satloc/metadata/s5a_learned_local_verifier/s5a1_local_verifier_query_summary.csv
outputs/satloc/reports/s5a_learned_local_verifier/s5a1_local_verifier_summary.json
outputs/satloc/figures/s5a_learned_local_verifier/s5a1_hit_rate_by_group.png
outputs/satloc/figures/s5a_learned_local_verifier/s5a1_median_error_by_group.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

PATH_COL_CANDIDATES = [
    "sat_image_path",
    "satellite_image_path",
    "tile_image_path",
    "tile_path",
    "candidate_image_path",
    "candidate_path",
    "image_path",
    "ref_image_path",
    "path",
    "filepath",
    "file_path",
]

FILENAME_COL_CANDIDATES = [
    "tile_filename",
    "sat_filename",
    "satellite_filename",
    "filename",
    "file",
    "image_filename",
    "name",
]

ID_COL_CANDIDATES = [
    "tile_id",
    "candidate_tile_id",
    "sat_tile_id",
    "satellite_tile_id",
    "tile_index",
    "candidate_tile_index",
    "sat_index",
    "satellite_index",
    "ref_index",
    "map_tile_id",
    "map_tile_index",
    "ranked_tile_id",
    "tile",
]

RANK_COL_CANDIDATES = ["rank", "phog_rank", "retrieval_rank", "candidate_rank", "orig_rank"]
SCORE_COL_CANDIDATES = ["score", "phog_score", "retrieval_score", "similarity", "sim", "final_score"]
ERROR_COL_CANDIDATES = [
    "center_error_m",
    "error_m",
    "top1_error_m",
    "distance_m",
    "gt_error_m",
    "eval_error_m",
    "candidate_error_m",
    "tile_center_error_m",
]
BOOLEAN_GT_COL_CANDIDATES = [
    "contains_gt",
    "contains_reference",
    "contains_ref",
    "gt_in_tile",
    "is_correct",
    "correct",
]


@dataclass
class LocalMatchResult:
    backend: str
    read_ok: bool
    query_keypoints: int = 0
    candidate_keypoints: int = 0
    raw_matches: int = 0
    good_matches: int = 0
    ransac_inliers: int = 0
    inlier_ratio: float = 0.0
    homography_success: bool = False
    local_score: float = 0.0
    elapsed_ms: float = 0.0
    error_message: str = ""


class SatellitePathResolver:
    """Best-effort satellite candidate path resolver.

    S4 scripts may use slightly different column names. This resolver first trusts
    explicit path columns in the ranked CSV. If unavailable, it uses the enriched
    satellite tile index and tries id/filename-based lookup.
    """

    def __init__(self, project_root: Path, sat_index_csv: Optional[Path]) -> None:
        self.project_root = project_root
        self.sat_index_csv = sat_index_csv
        self.sat_index: Optional[pd.DataFrame] = None
        self.lookup: Dict[str, Path] = {}
        self.basename_lookup: Dict[str, Path] = {}

        if sat_index_csv and sat_index_csv.exists():
            self.sat_index = pd.read_csv(sat_index_csv)
            self._build_lookup(self.sat_index)

    @staticmethod
    def _safe_key(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        # Normalize common integer-looking floats: 12.0 -> 12
        try:
            f = float(text)
            if f.is_integer():
                text = str(int(f))
        except Exception:
            pass
        return text

    def _resolve_existing_path(self, value: Any) -> Optional[Path]:
        key = self._safe_key(value)
        if not key:
            return None
        p = Path(key)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(self.project_root / p)
            candidates.append(p)
        for c in candidates:
            if c.exists() and c.suffix.lower() in IMAGE_EXTS:
                return c
        return None

    def _find_path_col(self, df: pd.DataFrame) -> Optional[str]:
        lower = {c.lower(): c for c in df.columns}
        for name in PATH_COL_CANDIDATES:
            if name in lower:
                return lower[name]
        # Fallback: any image-like path column.
        for c in df.columns:
            lc = c.lower()
            if "path" in lc or "file" in lc:
                series = df[c].dropna().astype(str).head(50)
                if any(Path(x).suffix.lower() in IMAGE_EXTS for x in series):
                    return c
        return None

    def _build_lookup(self, df: pd.DataFrame) -> None:
        path_col = self._find_path_col(df)
        if path_col is None:
            return

        # Add explicit path and basename keys.
        for _, row in df.iterrows():
            p = self._resolve_existing_path(row.get(path_col))
            if p is None:
                continue

            basename = p.name
            self.basename_lookup[basename] = p
            self.lookup[basename] = p
            self.lookup[p.stem] = p

            for col in df.columns:
                lc = col.lower()
                is_id_like = (
                    col in ID_COL_CANDIDATES
                    or lc in ID_COL_CANDIDATES
                    or "tile" in lc
                    or lc.endswith("id")
                    or lc.endswith("idx")
                    or "index" in lc
                    or "filename" in lc
                    or lc == "file"
                    or lc == "name"
                )
                if not is_id_like:
                    continue
                key = self._safe_key(row.get(col))
                if key:
                    self.lookup[key] = p
                    self.lookup[Path(key).name] = p
                    self.lookup[Path(key).stem] = p

    def resolve(self, row: pd.Series) -> Optional[Path]:
        # 1) Explicit path/filename columns in ranked CSV.
        for col in list(PATH_COL_CANDIDATES) + list(FILENAME_COL_CANDIDATES):
            if col in row.index:
                p = self._resolve_existing_path(row.get(col))
                if p is not None:
                    return p
                key = self._safe_key(row.get(col))
                if key:
                    if key in self.lookup:
                        return self.lookup[key]
                    if Path(key).name in self.lookup:
                        return self.lookup[Path(key).name]
                    if Path(key).stem in self.lookup:
                        return self.lookup[Path(key).stem]

        # 2) Any image-looking path value in row.
        for col in row.index:
            value = row.get(col)
            key = self._safe_key(value)
            if not key:
                continue
            suffix = Path(key).suffix.lower()
            if suffix in IMAGE_EXTS:
                p = self._resolve_existing_path(key)
                if p is not None:
                    return p
                if Path(key).name in self.lookup:
                    return self.lookup[Path(key).name]

        # 3) Any id-like column in row against satellite-index lookup.
        for col in row.index:
            lc = col.lower()
            if (
                col in ID_COL_CANDIDATES
                or lc in ID_COL_CANDIDATES
                or "tile" in lc
                or lc.endswith("id")
                or lc.endswith("idx")
                or "index" in lc
            ):
                key = self._safe_key(row.get(col))
                if key and key in self.lookup:
                    return self.lookup[key]

        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5A.1 local verifier inside PHOG top-K")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/satloc/metadata/s5a_learned_local_verifier/s5a0_failure_group_benchmark_manifest.csv"),
        help="S5A.0 benchmark manifest CSV.",
    )
    parser.add_argument(
        "--sat-index",
        type=Path,
        default=Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv"),
        help="Satellite tile enriched index CSV, used only to resolve tile image paths.",
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/satloc"))
    parser.add_argument("--sequence", type=str, default="traj01")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--backend", type=str, choices=["akaze", "orb", "sift"], default="akaze")
    parser.add_argument("--preprocess", type=str, choices=["gray", "luma", "clahe_gray"], default="clahe_gray")
    parser.add_argument("--resize-long", type=int, default=768, help="Resize longer side for matching. Use 0 to disable.")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=5.0)
    parser.add_argument("--threshold-m", type=float, default=40.0)
    parser.add_argument("--max-tokens", type=int, default=0, help="Optional debug limit. 0 means all tokens.")
    parser.add_argument(
        "--groups",
        type=str,
        default="",
        help="Optional comma-separated failure groups to run, e.g. selection_failure_correct_in_pool,lsd_destroyed_phog_success.",
    )
    parser.add_argument("--good-weight", type=float, default=0.05)
    parser.add_argument("--inlier-ratio-weight", type=float, default=2.0)
    parser.add_argument("--save-candidate-debug", action="store_true", help="Reserved flag for future visual panels.")
    return parser.parse_args()


def read_csv_checked(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def resolve_project_path(value: Any, project_root: Path) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    p = Path(text)
    candidates = [p] if p.is_absolute() else [project_root / p, p]
    for c in candidates:
        if c.exists():
            return c
    # Return the most likely path even if currently missing, useful for reporting.
    return candidates[0]


def find_first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def find_error_col(df: pd.DataFrame) -> Optional[str]:
    col = find_first_existing_col(df, ERROR_COL_CANDIDATES)
    if col is not None:
        return col
    # Conservative fallback: any numeric column with both "error" and metres hint.
    for c in df.columns:
        lc = c.lower()
        if "error" in lc and ("m" in lc or "meter" in lc or "metre" in lc):
            return c
    return None


def find_rank_col(df: pd.DataFrame) -> Optional[str]:
    return find_first_existing_col(df, RANK_COL_CANDIDATES)


def find_phog_score_col(df: pd.DataFrame) -> Optional[str]:
    return find_first_existing_col(df, SCORE_COL_CANDIDATES)


def find_bool_gt_col(df: pd.DataFrame) -> Optional[str]:
    return find_first_existing_col(df, BOOLEAN_GT_COL_CANDIDATES)


def parse_bool(value: Any) -> Optional[bool]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return None


def imread_preprocess(path: Path, preprocess: str, resize_long: int) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    if resize_long and resize_long > 0:
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side > 0 and long_side != resize_long:
            scale = resize_long / float(long_side)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    if preprocess in {"gray", "luma"}:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if preprocess == "clahe_gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    raise ValueError(f"Unknown preprocess mode: {preprocess}")


def create_detector_and_norm(backend: str):
    if backend == "orb":
        detector = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8)
        return detector, cv2.NORM_HAMMING, "binary"
    if backend == "akaze":
        detector = cv2.AKAZE_create()
        return detector, cv2.NORM_HAMMING, "binary"
    if backend == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("This OpenCV build does not provide cv2.SIFT_create(). Use --backend akaze or orb.")
        detector = cv2.SIFT_create(nfeatures=2500)
        return detector, cv2.NORM_L2, "float"
    raise ValueError(f"Unknown backend: {backend}")


def match_local(
    query_img: np.ndarray,
    cand_img: np.ndarray,
    backend: str,
    ratio: float,
    ransac_thresh: float,
    good_weight: float,
    inlier_ratio_weight: float,
) -> LocalMatchResult:
    t0 = time.perf_counter()
    result = LocalMatchResult(backend=backend, read_ok=True)
    try:
        detector, norm, _dtype = create_detector_and_norm(backend)
        q_kp, q_desc = detector.detectAndCompute(query_img, None)
        c_kp, c_desc = detector.detectAndCompute(cand_img, None)
        result.query_keypoints = 0 if q_kp is None else len(q_kp)
        result.candidate_keypoints = 0 if c_kp is None else len(c_kp)

        if q_desc is None or c_desc is None or result.query_keypoints < 4 or result.candidate_keypoints < 4:
            result.error_message = "insufficient_keypoints_or_descriptors"
            return result

        matcher = cv2.BFMatcher(norm, crossCheck=False)
        knn = matcher.knnMatch(q_desc, c_desc, k=2)
        result.raw_matches = len(knn)

        good = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio * n.distance:
                good.append(m)
        result.good_matches = len(good)

        if len(good) >= 4:
            src = np.float32([q_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([c_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
            if mask is not None:
                inliers = int(mask.ravel().sum())
                result.ransac_inliers = inliers
                result.inlier_ratio = float(inliers / max(1, len(good)))
                result.homography_success = bool(_H is not None and inliers >= 4)
        else:
            result.error_message = "fewer_than_4_good_matches"

        # Ranking score uses only local visual evidence, never reference error.
        result.local_score = (
            float(result.ransac_inliers)
            + good_weight * float(result.good_matches)
            + inlier_ratio_weight * float(result.inlier_ratio)
        )
        return result
    except Exception as exc:  # keep long runs robust
        result.error_message = f"exception:{type(exc).__name__}:{exc}"
        return result
    finally:
        result.elapsed_ms = 1000.0 * (time.perf_counter() - t0)


def choose_tile_identifier(row: pd.Series) -> str:
    for col in list(ID_COL_CANDIDATES) + list(FILENAME_COL_CANDIDATES):
        if col in row.index:
            value = row.get(col)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                text = str(value).strip()
                if text and text.lower() not in {"nan", "none", "null"}:
                    return text
    # Fallback: first column containing tile/id/index.
    for col in row.index:
        lc = col.lower()
        if "tile" in lc or "id" in lc or "index" in lc:
            value = row.get(col)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return str(value)
    return "unknown"


def get_numeric(row: pd.Series, col: Optional[str]) -> Optional[float]:
    if col is None or col not in row.index:
        return None
    try:
        value = row.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def ensure_out_dirs(out_dir: Path) -> Dict[str, Path]:
    paths = {
        "metadata": out_dir / "metadata" / "s5a_learned_local_verifier",
        "reports": out_dir / "reports" / "s5a_learned_local_verifier",
        "figures": out_dir / "figures" / "s5a_learned_local_verifier",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def sort_ranked_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rank_col = find_rank_col(df)
    if rank_col is not None:
        work = df.copy()
        work["__rank_tmp"] = pd.to_numeric(work[rank_col], errors="coerce")
        return work.sort_values("__rank_tmp", kind="mergesort").drop(columns=["__rank_tmp"])
    return df.copy()


def run_query(
    manifest_row: pd.Series,
    resolver: SatellitePathResolver,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    token = manifest_row.get("token", manifest_row.get("uav_token", "unknown"))
    failure_group = manifest_row.get("failure_group", "unknown")
    uav_col = "uav_image_path" if "uav_image_path" in manifest_row.index else None
    ranked_col = "phog_ranked_csv_path" if "phog_ranked_csv_path" in manifest_row.index else None

    if uav_col is None:
        for c in manifest_row.index:
            if "uav" in c.lower() and "path" in c.lower():
                uav_col = c
                break
    if ranked_col is None:
        for c in manifest_row.index:
            lc = c.lower()
            if ("rank" in lc or "phog" in lc) and "csv" in lc:
                ranked_col = c
                break

    uav_path = resolve_project_path(manifest_row.get(uav_col), args.project_root) if uav_col else None
    ranked_path = resolve_project_path(manifest_row.get(ranked_col), args.project_root) if ranked_col else None

    query_summary: Dict[str, Any] = {
        "sequence": args.sequence,
        "token": token,
        "failure_group": failure_group,
        "uav_image_path": str(uav_path) if uav_path else "",
        "phog_ranked_csv_path": str(ranked_path) if ranked_path else "",
        "top_k_requested": args.top_k,
        "backend": args.backend,
        "preprocess": args.preprocess,
        "status": "ok",
        "error_message": "",
    }

    if uav_path is None or not uav_path.exists():
        query_summary.update({"status": "failed", "error_message": "missing_uav_image"})
        return [], query_summary
    if ranked_path is None or not ranked_path.exists():
        query_summary.update({"status": "failed", "error_message": "missing_ranked_csv"})
        return [], query_summary

    query_img = imread_preprocess(uav_path, args.preprocess, args.resize_long)
    if query_img is None:
        query_summary.update({"status": "failed", "error_message": "uav_image_read_failed"})
        return [], query_summary

    ranked_df = pd.read_csv(ranked_path)
    if len(ranked_df) == 0:
        query_summary.update({"status": "failed", "error_message": "empty_ranked_csv"})
        return [], query_summary

    ranked_df = sort_ranked_candidates(ranked_df).head(args.top_k).reset_index(drop=True)
    rank_col = find_rank_col(ranked_df)
    phog_score_col = find_phog_score_col(ranked_df)
    error_col = find_error_col(ranked_df)
    gt_bool_col = find_bool_gt_col(ranked_df)

    candidate_rows: List[Dict[str, Any]] = []
    missing_candidate_paths = 0
    read_failures = 0
    t_query0 = time.perf_counter()

    for local_idx, cand_row in ranked_df.iterrows():
        cand_path = resolver.resolve(cand_row)
        phog_rank = get_numeric(cand_row, rank_col)
        if phog_rank is None:
            phog_rank = float(local_idx + 1)
        phog_score = get_numeric(cand_row, phog_score_col)
        eval_error_m = get_numeric(cand_row, error_col)
        contains_gt = parse_bool(cand_row.get(gt_bool_col)) if gt_bool_col else None
        tile_id = choose_tile_identifier(cand_row)

        base: Dict[str, Any] = {
            "sequence": args.sequence,
            "token": token,
            "failure_group": failure_group,
            "candidate_pool_rank": int(local_idx + 1),
            "phog_rank": phog_rank,
            "tile_id": tile_id,
            "candidate_image_path": str(cand_path) if cand_path else "",
            "phog_score": phog_score,
            "eval_error_m": eval_error_m,
            "contains_gt_eval_only": contains_gt,
        }

        if cand_path is None or not cand_path.exists():
            missing_candidate_paths += 1
            base.update(
                LocalMatchResult(
                    backend=args.backend,
                    read_ok=False,
                    error_message="missing_candidate_image",
                ).__dict__
            )
            candidate_rows.append(base)
            continue

        cand_img = imread_preprocess(cand_path, args.preprocess, args.resize_long)
        if cand_img is None:
            read_failures += 1
            base.update(
                LocalMatchResult(
                    backend=args.backend,
                    read_ok=False,
                    error_message="candidate_image_read_failed",
                ).__dict__
            )
            candidate_rows.append(base)
            continue

        match_res = match_local(
            query_img=query_img,
            cand_img=cand_img,
            backend=args.backend,
            ratio=args.ratio,
            ransac_thresh=args.ransac_thresh,
            good_weight=args.good_weight,
            inlier_ratio_weight=args.inlier_ratio_weight,
        )
        base.update(match_res.__dict__)
        candidate_rows.append(base)

    elapsed_query_s = time.perf_counter() - t_query0
    cand_df = pd.DataFrame(candidate_rows)

    # Ranking happens using local_score only. eval_error_m is used only after ranking.
    valid_score_df = cand_df.copy()
    valid_score_df["local_score_numeric"] = pd.to_numeric(valid_score_df.get("local_score", 0.0), errors="coerce").fillna(0.0)
    valid_score_df = valid_score_df.sort_values(
        ["local_score_numeric", "ransac_inliers", "good_matches"], ascending=[False, False, False], kind="mergesort"
    ).reset_index(drop=True)

    # Fill local ranks into the candidate rows.
    rank_map = {}
    for idx, r in valid_score_df.iterrows():
        # local unique key by candidate_pool_rank because tile id may repeat in rare cases.
        rank_map[int(r["candidate_pool_rank"])] = idx + 1
    for row in candidate_rows:
        row["local_verifier_rank"] = rank_map.get(int(row["candidate_pool_rank"]), None)
        row["hit_le_threshold_eval_only"] = (
            bool(row.get("eval_error_m") is not None and float(row["eval_error_m"]) <= args.threshold_m)
            if row.get("eval_error_m") is not None
            else None
        )

    # Query-level summary.
    phog_top1 = cand_df.sort_values("candidate_pool_rank", kind="mergesort").head(1)
    local_top1 = valid_score_df.head(1)

    def safe_first(df: pd.DataFrame, col: str) -> Any:
        if df is None or len(df) == 0 or col not in df.columns:
            return None
        value = df.iloc[0][col]
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    error_series = pd.to_numeric(cand_df.get("eval_error_m"), errors="coerce") if "eval_error_m" in cand_df else pd.Series(dtype=float)
    oracle_error = float(error_series.min()) if len(error_series.dropna()) else None

    phog_top1_error = safe_first(phog_top1, "eval_error_m")
    local_top1_error = safe_first(local_top1, "eval_error_m")
    phog_hit = bool(phog_top1_error is not None and float(phog_top1_error) <= args.threshold_m)
    local_hit = bool(local_top1_error is not None and float(local_top1_error) <= args.threshold_m)
    oracle_hit = bool(oracle_error is not None and oracle_error <= args.threshold_m)

    query_summary.update(
        {
            "ranked_candidates_read": int(len(ranked_df)),
            "candidates_scored": int(len(cand_df)),
            "missing_candidate_paths": int(missing_candidate_paths),
            "candidate_read_failures": int(read_failures),
            "elapsed_s": float(elapsed_query_s),
            "phog_top1_tile_id": safe_first(phog_top1, "tile_id"),
            "phog_top1_error_m": None if phog_top1_error is None else float(phog_top1_error),
            "phog_top1_hit_le_threshold": phog_hit,
            "local_top1_tile_id": safe_first(local_top1, "tile_id"),
            "local_top1_error_m": None if local_top1_error is None else float(local_top1_error),
            "local_top1_hit_le_threshold": local_hit,
            "local_top1_original_pool_rank": safe_first(local_top1, "candidate_pool_rank"),
            "local_top1_score": safe_first(local_top1, "local_score_numeric"),
            "local_top1_good_matches": safe_first(local_top1, "good_matches"),
            "local_top1_ransac_inliers": safe_first(local_top1, "ransac_inliers"),
            "local_top1_inlier_ratio": safe_first(local_top1, "inlier_ratio"),
            "oracle_best_topk_error_m": oracle_error,
            "oracle_hit_le_threshold": oracle_hit,
            "local_improved_vs_phog": (
                None
                if local_top1_error is None or phog_top1_error is None
                else bool(float(local_top1_error) < float(phog_top1_error))
            ),
            "local_destroyed_phog_success": bool(phog_hit and not local_hit),
            "local_rescued_phog_failure": bool((not phog_hit) and local_hit),
        }
    )
    return candidate_rows, query_summary


def build_summary(query_df: pd.DataFrame, candidate_df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "stage": "S5A.1_local_verifier_inside_PHOG_topK",
        "sequence": args.sequence,
        "backend": args.backend,
        "preprocess": args.preprocess,
        "top_k": args.top_k,
        "threshold_m": args.threshold_m,
#        "locked_rule": "reference coordinates/errors are evaluation-only and are not used for retrieval/ranking/scoring",
        "num_queries": int(len(query_df)),
        "num_candidates": int(len(candidate_df)),
        "status_counts": query_df["status"].value_counts(dropna=False).to_dict() if "status" in query_df else {},
    }

    ok = query_df[query_df["status"] == "ok"].copy() if "status" in query_df else query_df.copy()
    if len(ok) > 0:
        for col in [
            "phog_top1_hit_le_threshold",
            "local_top1_hit_le_threshold",
            "oracle_hit_le_threshold",
            "local_destroyed_phog_success",
            "local_rescued_phog_failure",
        ]:
            if col in ok.columns:
                summary[col + "_rate"] = float(ok[col].astype(bool).mean())

        for col in ["phog_top1_error_m", "local_top1_error_m", "oracle_best_topk_error_m", "elapsed_s"]:
            if col in ok.columns:
                vals = pd.to_numeric(ok[col], errors="coerce").dropna()
                if len(vals):
                    summary[col + "_median"] = float(vals.median())
                    summary[col + "_mean"] = float(vals.mean())

        group_summary = []
        for group, g in ok.groupby("failure_group", dropna=False):
            entry = {"failure_group": str(group), "count": int(len(g))}
            for col in ["phog_top1_hit_le_threshold", "local_top1_hit_le_threshold", "oracle_hit_le_threshold"]:
                if col in g.columns:
                    entry[col + "_rate"] = float(g[col].astype(bool).mean())
            for col in ["phog_top1_error_m", "local_top1_error_m", "oracle_best_topk_error_m"]:
                if col in g.columns:
                    vals = pd.to_numeric(g[col], errors="coerce").dropna()
                    entry[col + "_median"] = None if len(vals) == 0 else float(vals.median())
            group_summary.append(entry)
        summary["by_failure_group"] = sorted(group_summary, key=lambda x: x["failure_group"])

    return summary


def plot_group_hit_rates(query_df: pd.DataFrame, out_path: Path) -> None:
    ok = query_df[query_df["status"] == "ok"].copy()
    if len(ok) == 0:
        return
    rows = []
    for group, g in ok.groupby("failure_group", dropna=False):
        rows.append(
            {
                "failure_group": str(group),
                "PHOG top1": float(g["phog_top1_hit_le_threshold"].astype(bool).mean()),
                "Local verifier top1": float(g["local_top1_hit_le_threshold"].astype(bool).mean()),
                "Oracle topK": float(g["oracle_hit_le_threshold"].astype(bool).mean()),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("failure_group")
    x = np.arange(len(df))
    width = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - width, df["PHOG top1"], width, label="PHOG top1")
    ax.bar(x, df["Local verifier top1"], width, label="Local verifier top1")
    ax.bar(x + width, df["Oracle topK"], width, label="Oracle topK")
    ax.set_ylabel(f"Hit rate <= threshold")
    ax.set_ylim(0, 1.05)
    ax.set_title("S5A.1 hit rate by S4C.6C failure group")
    ax.set_xticks(x)
    ax.set_xticklabels(df["failure_group"].tolist(), rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_group_median_errors(query_df: pd.DataFrame, out_path: Path) -> None:
    ok = query_df[query_df["status"] == "ok"].copy()
    if len(ok) == 0:
        return
    rows = []
    for group, g in ok.groupby("failure_group", dropna=False):
        rows.append(
            {
                "failure_group": str(group),
                "PHOG top1": pd.to_numeric(g["phog_top1_error_m"], errors="coerce").median(),
                "Local verifier top1": pd.to_numeric(g["local_top1_error_m"], errors="coerce").median(),
                "Oracle topK": pd.to_numeric(g["oracle_best_topk_error_m"], errors="coerce").median(),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("failure_group")
    x = np.arange(len(df))
    width = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - width, df["PHOG top1"], width, label="PHOG top1")
    ax.bar(x, df["Local verifier top1"], width, label="Local verifier top1")
    ax.bar(x + width, df["Oracle topK"], width, label="Oracle topK")
    ax.set_ylabel("Median evaluation error [m]")
    ax.set_title("S5A.1 median error by S4C.6C failure group")
    ax.set_xticks(x)
    ax.set_xticklabels(df["failure_group"].tolist(), rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    out_paths = ensure_out_dirs(args.out_dir)

    manifest = read_csv_checked(args.manifest, "S5A.0 manifest")

    if args.groups.strip():
        allowed_groups = {g.strip() for g in args.groups.split(",") if g.strip()}
        manifest = manifest[manifest["failure_group"].isin(allowed_groups)].copy()

    if args.max_tokens and args.max_tokens > 0:
        manifest = manifest.head(args.max_tokens).copy()

    resolver = SatellitePathResolver(project_root=args.project_root, sat_index_csv=args.sat_index)

    all_candidate_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []

    print("S5A.1 local verifier inside PHOG top-K")
    print("----------------------------------------")
    print(f"Sequence:       {args.sequence}")
    print(f"Queries:        {len(manifest)}")
    print(f"Top-K:          {args.top_k}")
    print(f"Backend:        {args.backend}")
    print(f"Preprocess:     {args.preprocess}")
    print(f"Resize long:    {args.resize_long}")
    print(f"Threshold:      {args.threshold_m:.1f} m")
    print()

    t0 = time.perf_counter()
    for idx, row in manifest.reset_index(drop=True).iterrows():
        token = row.get("token", row.get("uav_token", idx))
        group = row.get("failure_group", "unknown")
        print(f"[{idx + 1:03d}/{len(manifest):03d}] token={token} group={group}", flush=True)
        cand_rows, q_summary = run_query(row, resolver, args)
        all_candidate_rows.extend(cand_rows)
        query_rows.append(q_summary)

    total_elapsed = time.perf_counter() - t0

    candidate_df = pd.DataFrame(all_candidate_rows)
    query_df = pd.DataFrame(query_rows)

    candidate_csv = out_paths["metadata"] / "s5a1_local_verifier_candidate_scores.csv"
    query_csv = out_paths["metadata"] / "s5a1_local_verifier_query_summary.csv"
    summary_json = out_paths["reports"] / "s5a1_local_verifier_summary.json"
    hit_fig = out_paths["figures"] / "s5a1_hit_rate_by_group.png"
    err_fig = out_paths["figures"] / "s5a1_median_error_by_group.png"

    candidate_df.to_csv(candidate_csv, index=False)
    query_df.to_csv(query_csv, index=False)

    summary = build_summary(query_df, candidate_df, args)
    summary["total_elapsed_s"] = float(total_elapsed)
    summary["candidate_scores_csv"] = str(candidate_csv)
    summary["query_summary_csv"] = str(query_csv)
    summary["hit_rate_figure"] = str(hit_fig)
    summary["median_error_figure"] = str(err_fig)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if len(query_df):
        plot_group_hit_rates(query_df, hit_fig)
        plot_group_median_errors(query_df, err_fig)

    ok = query_df[query_df["status"] == "ok"] if "status" in query_df else query_df
    print()
    print("S5A.1 complete")
    print("--------------")
    print(f"Queries processed:     {len(query_df)}")
    print(f"OK queries:            {len(ok)}")
    print(f"Candidate rows:        {len(candidate_df)}")
    if len(ok):
        print(f"PHOG hit <=thr:        {float(ok['phog_top1_hit_le_threshold'].astype(bool).mean()):.3f}")
        print(f"Local hit <=thr:       {float(ok['local_top1_hit_le_threshold'].astype(bool).mean()):.3f}")
        print(f"Oracle topK <=thr:     {float(ok['oracle_hit_le_threshold'].astype(bool).mean()):.3f}")
        phog_med = pd.to_numeric(ok["phog_top1_error_m"], errors="coerce").median()
        local_med = pd.to_numeric(ok["local_top1_error_m"], errors="coerce").median()
        print(f"PHOG median error:     {phog_med:.3f} m")
        print(f"Local median error:    {local_med:.3f} m")
    print(f"Candidate CSV:         {candidate_csv}")
    print(f"Query summary CSV:     {query_csv}")
    print(f"Summary JSON:          {summary_json}")
    print(f"Hit-rate figure:       {hit_fig}")
    print(f"Median-error figure:   {err_fig}")
    print()
#    print("Locked rule: reference coordinates/errors were used only after ranking for evaluation.")


if __name__ == "__main__":
    main()
