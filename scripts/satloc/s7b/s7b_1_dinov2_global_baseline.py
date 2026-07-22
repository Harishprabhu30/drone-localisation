#!/usr/bin/env python3
"""
S7B.1 — DINOv2 Global Retrieval Baseline

This patched version is based on the S7B.1-preflight result.

Important schema decision
-------------------------
Use the frozen S7 query manifest directly:

    outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_query_manifest.csv

Required query columns:
    token0_id             -> official S5C/S7 benchmark token
    image_path            -> existing UAV image path
    lon, lat              -> evaluation-only query reference location

Required satellite columns:
    tile_index            -> satellite tile id
    tile_path             -> existing satellite tile path
    lon_center, lat_center -> evaluation-only tile center location

No UAV-index join is needed for DINOv2 retrieval, because the query manifest
already has valid image_path/image_path_resolved fields.

Ground-truth rule
-----------------
DINOv2 descriptor extraction and cosine ranking do NOT use lon/lat, GT, GNSS,
or error labels. lon/lat are used only after ranking to compute offline
Recall@K and error metrics.

Smoke run
---------
cd /Users/harishprabhu/Documents/drone-localisation
source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python scripts/satloc/s7b/s7b_1_dinov2_global_baseline.py \
  --repo-root "$PWD" \
  --max-queries 10 \
  --max-sat-tiles 300 \
  --top-k 50 \
  --device auto \
  --batch-size 4 \
  --cache-tag smoke_q10_sat300 \
  2>&1 | tee outputs/satloc/reports/s7b_dinov2_global/s7b1_dinov2_smoke.log

Full run
--------
PYTHONUNBUFFERED=1 python -u scripts/satloc/s7b/s7b_1_dinov2_global_baseline.py \
  --repo-root "$PWD" \
  --top-k 100 \
  --image-size 518 \
  --device cpu \
  --batch-size 1 \
  --cache-tag full263_img518_cpu \
  2>&1 | tee outputs/satloc/reports/s7b_dinov2_global/s7b1_dinov2_full263_img518_cpu.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

# Force live progress when piping through tee.
from functools import partial
print = partial(print, flush=True)


DEFAULT_QUERY_MANIFEST = Path(
    "outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_query_manifest.csv"
)
DEFAULT_SAT_INDEX = Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")
DEFAULT_SCENE_LABELS = Path(
    "outputs/satloc/metadata/s7_retrieval_upgrade/s7_scene_labels_canonical_traj01.csv"
)

DEFAULT_METADATA_OUT = Path("outputs/satloc/metadata/s7b_dinov2_global")
DEFAULT_REPORT_OUT = Path("outputs/satloc/reports/s7b_dinov2_global")
DEFAULT_FIGURE_OUT = Path("outputs/satloc/figures/s7b_dinov2_global")


@dataclass
class DescriptorCache:
    descriptors: np.ndarray
    ids: list[str]
    paths: list[str]
    meta: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S7B.1 DINOv2 global retrieval baseline for SatLoc traj01."
    )

    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_MANIFEST)
    p.add_argument("--satellite-index", type=Path, default=DEFAULT_SAT_INDEX)
    p.add_argument("--scene-labels", type=Path, default=DEFAULT_SCENE_LABELS)
    p.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE_OUT)

    # Explicit schema from preflight.
    p.add_argument("--query-token-col", default="token0_id")
    p.add_argument("--query-path-col", default="image_path")
    p.add_argument("--query-lon-col", default="lon")
    p.add_argument("--query-lat-col", default="lat")
    p.add_argument("--sat-id-col", default="tile_index")
    p.add_argument("--sat-path-col", default="tile_path")
    p.add_argument("--sat-lon-col", default="lon_center")
    p.add_argument("--sat-lat-col", default="lat_center")
    p.add_argument("--scene-token-col", default="token")
    p.add_argument("--scene-col", default="primary_scene")

    p.add_argument(
        "--backend",
        choices=["auto", "torch_hub", "transformers"],
        default="auto",
    )
    p.add_argument("--model-name", default="dinov2_vits14")
    p.add_argument("--hf-model", default="facebook/dinov2-small")
    p.add_argument(
        "--pooling",
        choices=["cls", "avgpatch", "cls_avg_concat"],
        default="avgpatch",
    )
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument(
        "--crop-mode",
        choices=["center_square", "resize_square"],
        default="center_square",
    )

    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--threshold-m", type=float, default=40.0)
    p.add_argument("--recall-ks", default="1,5,10,20,50,100")

    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--max-sat-tiles", type=int, default=0)
    p.add_argument(
        "--sat-subset-mode",
        choices=["first", "uniform"],
        default="uniform",
    )

    p.add_argument("--cache-tag", default="")
    p.add_argument("--rebuild-sat-cache", action="store_true")
    p.add_argument("--rebuild-query-cache", action="store_true")
    p.add_argument("--allow-missing-images", action="store_true")
    p.add_argument("--skip-eval", action="store_true")

    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_csv(path: Path, name: str, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {name}: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}\nAvailable columns: {list(df.columns)}"
        )


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def safe_name(text: Any) -> str:
    import re

    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "run"


def resolve_path_value(root: Path, value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return ""

    p = Path(text)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(root / p)

    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def parse_recall_ks(text: str, top_k: int) -> list[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if 0 < k <= top_k:
            vals.append(k)
    vals = sorted(set(vals))
    if top_k not in vals:
        vals.append(top_k)
    return vals


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def haversine_m(lon1: Any, lat1: Any, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    lon1 = float(lon1)
    lat1 = float(lat1)
    lon2 = lon2.astype(float)
    lat2 = lat2.astype(float)

    r = 6371008.8
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def preprocess_pil(img: Image.Image, image_size: int, crop_mode: str) -> np.ndarray:
    img = img.convert("RGB")

    if crop_mode == "center_square":
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    img = img.resize((image_size, image_size), resample=resample)

    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))


def detect_device(requested: str) -> tuple[str, dict[str, Any]]:
    import torch

    info = {
        "requested_device": requested,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        selected = "cuda"
    elif requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but unavailable.")
        selected = "mps"
    elif requested == "cpu":
        selected = "cpu"
    else:
        if torch.cuda.is_available():
            selected = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected = "mps"
        else:
            selected = "cpu"

    info["selected_device"] = selected
    if selected == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return selected, info


class DinoExtractor:
    def __init__(
        self,
        backend: str,
        model_name: str,
        hf_model: str,
        pooling: str,
        image_size: int,
        crop_mode: str,
        device_request: str,
    ):
        self.backend_request = backend
        self.model_name = model_name
        self.hf_model = hf_model
        self.pooling = pooling
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.device, self.device_info = detect_device(device_request)
        self.model = None
        self.processor = None
        self.torch = None
        self.backend = ""
        self.load_error_history: list[str] = []
        self._load()

    def _load(self) -> None:
        import torch

        self.torch = torch
        backends = ["torch_hub", "transformers"] if self.backend_request == "auto" else [self.backend_request]

        for backend in backends:
            try:
                if backend == "torch_hub":
                    self.model = torch.hub.load(
                        "facebookresearch/dinov2",
                        self.model_name,
                        pretrained=True,
                    )
                    self.model.eval().to(self.device)
                    self.backend = "torch_hub"
                    return

                if backend == "transformers":
                    from transformers import AutoImageProcessor, AutoModel

                    self.processor = AutoImageProcessor.from_pretrained(self.hf_model)
                    self.model = AutoModel.from_pretrained(self.hf_model)
                    self.model.eval().to(self.device)
                    self.backend = "transformers"
                    return

            except Exception as exc:
                self.load_error_history.append(f"{backend}: {repr(exc)}")

        raise RuntimeError(
            "Could not load DINOv2 model. Tried:\n"
            + "\n".join(self.load_error_history)
            + "\nFirst run may need internet/model cache. Try backend transformers or pre-cache weights."
        )

    def meta(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_request": self.backend_request,
            "model_name": self.model_name,
            "hf_model": self.hf_model,
            "pooling": self.pooling,
            "image_size": self.image_size,
            "crop_mode": self.crop_mode,
            **self.device_info,
            "load_error_history": self.load_error_history,
        }

    def _pool_torch_hub(self, batch: Any) -> Any:
        torch = self.torch
        assert torch is not None

        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(batch)
        else:
            out = self.model(batch)

        if isinstance(out, dict):
            cls = out.get("x_norm_clstoken")
            patch = out.get("x_norm_patchtokens")

            if self.pooling == "cls":
                if cls is None:
                    raise RuntimeError("No CLS token found in DINOv2 output.")
                return cls
            if self.pooling == "avgpatch":
                if patch is not None:
                    return patch.mean(dim=1)
                if cls is not None:
                    return cls
                raise RuntimeError("No patch or CLS token found in DINOv2 output.")
            if cls is None or patch is None:
                raise RuntimeError("cls_avg_concat requires CLS and patch tokens.")
            return torch.cat([cls, patch.mean(dim=1)], dim=1)

        if torch.is_tensor(out):
            if out.ndim == 2:
                return out
            if out.ndim == 3:
                cls = out[:, 0]
                patch = out[:, 1:] if out.shape[1] > 1 else out
                if self.pooling == "cls":
                    return cls
                if self.pooling == "avgpatch":
                    return patch.mean(dim=1)
                return torch.cat([cls, patch.mean(dim=1)], dim=1)

        raise RuntimeError(f"Could not interpret DINOv2 output type={type(out)}")

    def _pool_transformers(self, images: list[Image.Image]) -> Any:
        torch = self.torch
        assert torch is not None
        assert self.processor is not None

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        hidden = out.last_hidden_state
        cls = hidden[:, 0]
        patch = hidden[:, 1:] if hidden.shape[1] > 1 else hidden

        if self.pooling == "cls":
            return cls
        if self.pooling == "avgpatch":
            return patch.mean(dim=1)
        return torch.cat([cls, patch.mean(dim=1)], dim=1)

    def encode_paths(self, paths: list[str]) -> np.ndarray:
        torch = self.torch
        assert torch is not None

        with torch.inference_mode():
            if self.backend == "transformers":
                images = [Image.open(p).convert("RGB") for p in paths]
                desc = self._pool_transformers(images)
            else:
                arrs = [
                    preprocess_pil(Image.open(p), self.image_size, self.crop_mode)
                    for p in paths
                ]
                batch = torch.from_numpy(np.stack(arrs, axis=0)).to(self.device)
                desc = self._pool_torch_hub(batch)

            return l2_normalize(desc.detach().float().cpu().numpy().astype(np.float32))


def df_hash(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.sha1()
    for col in cols:
        if col not in df.columns:
            continue
        h.update(col.encode())
        for v in df[col].astype(str).fillna("").tolist():
            h.update(v.encode())
            h.update(b"\0")
    return h.hexdigest()[:12]


def load_cache(path: Path, ids: list[str]) -> DescriptorCache | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    cached_ids = [str(x) for x in data["ids"].tolist()]
    if cached_ids != [str(x) for x in ids]:
        return None
    paths = [str(x) for x in data["paths"].tolist()]
    meta_json = str(data["meta_json"].item()) if "meta_json" in data else "{}"
    try:
        meta = json.loads(meta_json)
    except Exception:
        meta = {}
    return DescriptorCache(
        descriptors=l2_normalize(data["descriptors"].astype(np.float32)),
        ids=cached_ids,
        paths=paths,
        meta=meta,
    )


def save_cache(path: Path, descriptors: np.ndarray, ids: list[str], paths: list[str], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        descriptors=descriptors.astype(np.float32),
        ids=np.asarray(ids, dtype=str),
        paths=np.asarray(paths, dtype=str),
        meta_json=np.asarray(json.dumps(meta, default=json_safe)),
    )


def extract_or_load(
    name: str,
    df: pd.DataFrame,
    id_col: str,
    path_col: str,
    cache_path: Path,
    rebuild: bool,
    extractor: DinoExtractor,
    batch_size: int,
) -> DescriptorCache:
    ids = df[id_col].astype(str).tolist()
    paths = df[path_col].astype(str).tolist()

    if not rebuild:
        cached = load_cache(cache_path, ids)
        if cached is not None:
            print(f"[S7B.1] Using cached {name} descriptors: {cache_path}")
            return cached

    print(f"[S7B.1] Extracting {name} descriptors: rows={len(df)} batch_size={batch_size}")
    started = time.time()
    parts = []

    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        desc = extractor.encode_paths(paths[start:end])
        parts.append(desc)

        done = end
        if done == len(paths) or done % max(10 * batch_size, 1) == 0:
            elapsed = time.time() - started
            print(f"  {name}: {done}/{len(paths)} images, {done / max(elapsed, 1e-9):.2f} img/s")

    descriptors = l2_normalize(np.vstack(parts))
    meta = {
        "name": name,
        "rows": len(df),
        "descriptor_dim": int(descriptors.shape[1]),
        "runtime_s": float(time.time() - started),
        **extractor.meta(),
    }
    save_cache(cache_path, descriptors, ids, paths, meta)
    print(f"[S7B.1] Saved {name} descriptors: {cache_path}")
    return DescriptorCache(descriptors, ids, paths, meta)


def load_query_table(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    q = read_csv(resolve(root, args.query_manifest), "S7 query manifest")
    require_columns(
        q,
        [
            args.query_token_col,
            args.query_path_col,
            args.query_lon_col,
            args.query_lat_col,
        ],
        "S7 query manifest",
    )

    out = q.copy()
    out["token"] = pd.to_numeric(out[args.query_token_col], errors="raise").astype(int)
    out["query_image_path"] = out[args.query_path_col].apply(lambda x: resolve_path_value(root, x))
    out["query_lon_eval_only"] = pd.to_numeric(out[args.query_lon_col], errors="coerce")
    out["query_lat_eval_only"] = pd.to_numeric(out[args.query_lat_col], errors="coerce")

    scene_path = resolve(root, args.scene_labels)
    if scene_path.exists():
        scene = read_csv(scene_path, "scene labels", required=False)
        if not scene.empty and args.scene_token_col in scene.columns and args.scene_col in scene.columns:
            scene_small = scene[[args.scene_token_col, args.scene_col]].copy()
            scene_small["token"] = pd.to_numeric(scene_small[args.scene_token_col], errors="coerce").astype("Int64")
            scene_small = scene_small.dropna(subset=["token"]).copy()
            scene_small["token"] = scene_small["token"].astype(int)
            scene_small = scene_small[["token", args.scene_col]].rename(
                columns={args.scene_col: "primary_scene"}
            )
            scene_small = scene_small.drop_duplicates("token", keep="first")
            out = out.merge(scene_small, on="token", how="left", validate="many_to_one")

    if "primary_scene" not in out.columns:
        out["primary_scene"] = "unlabeled"
    out["primary_scene"] = (
        out["primary_scene"]
        .fillna("unlabeled")
        .astype(str)
        .str.strip()
        .replace({"": "unlabeled"})
    )

    out = out.drop_duplicates("token", keep="first").sort_values("token").reset_index(drop=True)
    if len(out) != 263:
        raise ValueError(f"Expected 263 official query tokens, found {len(out)}")
    return out


def load_sat_table(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    sat = read_csv(resolve(root, args.satellite_index), "satellite tile index")
    require_columns(
        sat,
        [
            args.sat_id_col,
            args.sat_path_col,
            args.sat_lon_col,
            args.sat_lat_col,
        ],
        "satellite tile index",
    )

    out = sat.copy()
    out["tile_id"] = out[args.sat_id_col].astype(str)
    out["sat_image_path"] = out[args.sat_path_col].apply(lambda x: resolve_path_value(root, x))
    out["sat_lon_eval_only"] = pd.to_numeric(out[args.sat_lon_col], errors="coerce")
    out["sat_lat_eval_only"] = pd.to_numeric(out[args.sat_lat_col], errors="coerce")
    out = out.drop_duplicates("tile_id", keep="first").reset_index(drop=True)
    return out


def filter_missing(df: pd.DataFrame, path_col: str, name: str, allow_missing: bool) -> pd.DataFrame:
    missing = df[path_col].astype(str).str.len().eq(0)
    if missing.any():
        examples = df.loc[missing].head(10)
        msg = f"{name}: {int(missing.sum())}/{len(df)} rows have missing paths.\n{examples}"
        if not allow_missing:
            raise FileNotFoundError(msg)
        print("[WARN]", msg)
        df = df.loc[~missing].copy()
    return df.reset_index(drop=True)


def subset_sat(sat: pd.DataFrame, max_sat: int, mode: str) -> pd.DataFrame:
    if max_sat <= 0 or max_sat >= len(sat):
        return sat.reset_index(drop=True)
    if mode == "first":
        return sat.head(max_sat).reset_index(drop=True)
    idx = np.linspace(0, len(sat) - 1, max_sat).round().astype(int)
    return sat.iloc[sorted(set(idx.tolist()))].reset_index(drop=True)


def retrieve_topk(q: np.ndarray, s: np.ndarray, top_k: int, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    q = l2_normalize(q)
    s = l2_normalize(s)
    k = min(top_k, s.shape[0])
    idx_parts = []
    sim_parts = []

    for start in range(0, q.shape[0], batch):
        end = min(start + batch, q.shape[0])
        sim = q[start:end] @ s.T
        if k == s.shape[0]:
            idx = np.argsort(-sim, axis=1)[:, :k]
        else:
            idx_part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sim, idx_part, axis=1), axis=1)
            idx = np.take_along_axis(idx_part, order, axis=1)
        sim_sorted = np.take_along_axis(sim, idx, axis=1)
        idx_parts.append(idx)
        sim_parts.append(sim_sorted)

    return np.vstack(idx_parts), np.vstack(sim_parts)


def build_outputs(
    qdf: pd.DataFrame,
    sdf: pd.DataFrame,
    top_idx: np.ndarray,
    top_sim: np.ndarray,
    threshold_m: float,
    recall_ks: list[int],
    skip_eval: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sat_ids = sdf["tile_id"].astype(str).to_numpy()
    sat_lon = sdf["sat_lon_eval_only"].to_numpy(float)
    sat_lat = sdf["sat_lat_eval_only"].to_numpy(float)

    cand_rows = []
    query_rows = []

    for qi, qrow in qdf.reset_index(drop=True).iterrows():
        inds = top_idx[qi]
        sims = top_sim[qi]

        errors = np.full(len(inds), np.nan, dtype=float)
        can_eval = (
            not skip_eval
            and pd.notna(qrow["query_lon_eval_only"])
            and pd.notna(qrow["query_lat_eval_only"])
        )
        if can_eval:
            errors = haversine_m(
                qrow["query_lon_eval_only"],
                qrow["query_lat_eval_only"],
                sat_lon[inds],
                sat_lat[inds],
            )

        token = int(qrow["token"])
        for rank, (sat_i, sim, err) in enumerate(zip(inds, sims, errors), start=1):
            cand_rows.append(
                {
                    "token": token,
                    "rank": int(rank),
                    "tile_id": str(sat_ids[sat_i]),
                    "dinov2_similarity": float(sim),
                    "eval_error_m": float(err) if math.isfinite(float(err)) else np.nan,
                    "hit_le_threshold_eval_only": bool(
                        math.isfinite(float(err)) and float(err) <= threshold_m
                    ),
                    "query_image_path": qrow["query_image_path"],
                    "sat_image_path": sdf.iloc[int(sat_i)]["sat_image_path"],
                    "primary_scene": qrow["primary_scene"],
                }
            )

        if np.isfinite(errors).any():
            oracle_pos = int(np.nanargmin(errors))
            oracle_error = float(errors[oracle_pos])
            oracle_rank = int(oracle_pos + 1)
            oracle_tile_id = str(sat_ids[inds[oracle_pos]])
            top1_error = float(errors[0])
        else:
            oracle_error = np.nan
            oracle_rank = np.nan
            oracle_tile_id = ""
            top1_error = np.nan

        row = {
            "token": token,
            "primary_scene": qrow["primary_scene"],
            "query_image_path": qrow["query_image_path"],
            "top1_tile_id": str(sat_ids[inds[0]]),
            "top1_similarity": float(sims[0]),
            "top1_error_m_eval_only": top1_error,
            "top1_hit_le_threshold_eval_only": bool(
                math.isfinite(top1_error) and top1_error <= threshold_m
            ),
            "oracle_tile_id_eval_only": oracle_tile_id,
            "oracle_error_m_eval_only": oracle_error,
            "oracle_rank_eval_only": oracle_rank,
        }
        for k in recall_ks:
            kk = min(k, len(errors))
            row[f"hit_at_{k}_eval_only"] = bool(
                kk > 0 and np.isfinite(errors[:kk]).any() and np.nanmin(errors[:kk]) <= threshold_m
            )
        query_rows.append(row)

    cand = pd.DataFrame(cand_rows)
    query = pd.DataFrame(query_rows)

    recall_rows = []
    for k in recall_ks:
        col = f"hit_at_{k}_eval_only"
        hits = int(query[col].sum()) if col in query.columns else 0
        total = int(len(query))
        recall_rows.append(
            {"k": int(k), "hits": hits, "queries": total, "recall": hits / total if total else 0.0}
        )
    recall = pd.DataFrame(recall_rows)

    scene_rows = []
    for scene, g in query.groupby("primary_scene", dropna=False):
        srow = {
            "primary_scene": scene,
            "queries": int(len(g)),
            "top1_hits": int(g["top1_hit_le_threshold_eval_only"].sum()),
            "top1_hit_rate": float(g["top1_hit_le_threshold_eval_only"].mean()),
            "median_top1_error_m": float(pd.to_numeric(g["top1_error_m_eval_only"], errors="coerce").median()),
            "median_oracle_error_m": float(pd.to_numeric(g["oracle_error_m_eval_only"], errors="coerce").median()),
            "median_oracle_rank": float(pd.to_numeric(g["oracle_rank_eval_only"], errors="coerce").median()),
        }
        for k in recall_ks:
            col = f"hit_at_{k}_eval_only"
            srow[f"recall_at_{k}"] = float(g[col].mean()) if col in g.columns else 0.0
        scene_rows.append(srow)
    scene = pd.DataFrame(scene_rows).sort_values(["queries", "primary_scene"], ascending=[False, True])

    return cand, query, recall, scene


def plot_recall(recall: pd.DataFrame, out: Path, threshold_m: float) -> None:
    if recall.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(recall["k"].astype(str), recall["recall"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("K")
    ax.set_ylabel(f"Recall@K <= {threshold_m:.0f} m")
    ax.set_title("S7B.1 DINOv2 global retrieval")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_hist(values: pd.Series, out: Path, title: str, xlabel: str) -> None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return
    clipped = vals.clip(upper=float(vals.quantile(0.98)))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = min(50, max(10, int(math.sqrt(len(clipped)))))
    ax.hist(clipped, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_env(path: Path, args: argparse.Namespace, extractor: DinoExtractor, extra: dict[str, Any]) -> None:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "args": {k: json_safe(v) for k, v in vars(args).items()},
        "extractor": extractor.meta(),
        **extra,
    }
    try:
        import torch
        env["torch_version"] = getattr(torch, "__version__", "unknown")
    except Exception as exc:
        env["torch_error"] = repr(exc)
    try:
        import transformers
        env["transformers_version"] = getattr(transformers, "__version__", "available_unknown")
    except Exception:
        env["transformers_version"] = "not_available"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2, default=json_safe), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()

    metadata_out = resolve(root, args.metadata_out)
    report_out = resolve(root, args.report_out)
    figure_out = resolve(root, args.figure_out)
    for d in [metadata_out, report_out, figure_out]:
        d.mkdir(parents=True, exist_ok=True)

    recall_ks = parse_recall_ks(args.recall_ks, args.top_k)

    run_bits = [
        args.backend,
        args.model_name if args.backend != "transformers" else safe_name(args.hf_model),
        args.pooling,
        f"img{args.image_size}",
        f"k{args.top_k}",
        f"q{args.max_queries if args.max_queries > 0 else 263}",
        f"sat{args.max_sat_tiles if args.max_sat_tiles > 0 else 8625}",
    ]
    if args.cache_tag:
        run_bits.append(args.cache_tag)
    run_name = safe_name("_".join(run_bits))

    print()
    print("S7B.1 — DINOv2 Global Retrieval Baseline")
    print("----------------------------------------")
    print(f"Repository root:        {root}")
    print(f"Run name:               {run_name}")
    print(f"Backend request:        {args.backend}")
    print(f"Model name:             {args.model_name}")
    print(f"HF model:               {args.hf_model}")
    print(f"Pooling:                {args.pooling}")
    print(f"Image size:             {args.image_size}")
    print(f"Top-K:                  {args.top_k}")
    print(f"Threshold m:            {args.threshold_m}")
    print()

    query_df = load_query_table(root, args)
    sat_df = load_sat_table(root, args)

    if args.max_queries > 0:
        query_df = query_df.head(args.max_queries).reset_index(drop=True)
    sat_df = subset_sat(sat_df, args.max_sat_tiles, args.sat_subset_mode)

    query_df = filter_missing(query_df, "query_image_path", "queries", args.allow_missing_images)
    sat_df = filter_missing(sat_df, "sat_image_path", "satellite tiles", args.allow_missing_images)

    print(f"Queries:                {len(query_df)}")
    print(f"Satellite tiles:        {len(sat_df)}")
    print("Scene distribution:")
    print(query_df["primary_scene"].value_counts(dropna=False).to_string())
    print()

    print("[S7B.1] Loading DINOv2 model/backend...")
    extractor = DinoExtractor(
        backend=args.backend,
        model_name=args.model_name,
        hf_model=args.hf_model,
        pooling=args.pooling,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        device_request=args.device,
    )

    cache_base = safe_name(
        f"{extractor.backend}_{args.model_name if extractor.backend == 'torch_hub' else args.hf_model}_"
        f"{args.pooling}_img{args.image_size}_{args.crop_mode}"
    )
    q_hash = df_hash(query_df, ["token", "query_image_path"])
    s_hash = df_hash(sat_df, ["tile_id", "sat_image_path"])

    # Cache names intentionally exclude run_name/top-K/query-count so descriptor caches
    # can be reused across q10 smoke, q263 full, and different Top-K evaluations.
    q_cache_path = metadata_out / f"s7b1_dinov2_query_descriptors_{cache_base}_{q_hash}.npz"
    s_cache_path = metadata_out / f"s7b1_dinov2_satellite_descriptors_{cache_base}_{s_hash}.npz"
    env_path = report_out / f"s7b1_dinov2_environment_{run_name}.json"
    print(f"[S7B.1] Query descriptor cache:     {q_cache_path}")
    print(f"[S7B.1] Satellite descriptor cache: {s_cache_path}")

    write_env(
        env_path,
        args,
        extractor,
        {
            "run_name": run_name,
            "queries": int(len(query_df)),
            "satellite_tiles": int(len(sat_df)),
            "query_hash": q_hash,
            "satellite_hash": s_hash,
            "schema_decision": {
                "query_token_col": args.query_token_col,
                "query_path_col": args.query_path_col,
                "sat_id_col": args.sat_id_col,
                "sat_path_col": args.sat_path_col,
            },
        },
    )

    sat_cache = extract_or_load(
        "satellite",
        sat_df,
        "tile_id",
        "sat_image_path",
        s_cache_path,
        args.rebuild_sat_cache,
        extractor,
        args.batch_size,
    )
    query_cache = extract_or_load(
        "query",
        query_df,
        "token",
        "query_image_path",
        q_cache_path,
        args.rebuild_query_cache,
        extractor,
        args.batch_size,
    )

    started = time.time()
    top_idx, top_sim = retrieve_topk(query_cache.descriptors, sat_cache.descriptors, args.top_k)
    retrieval_runtime_s = float(time.time() - started)

    cand, query, recall, scene = build_outputs(
        qdf=query_df,
        sdf=sat_df,
        top_idx=top_idx,
        top_sim=top_sim,
        threshold_m=args.threshold_m,
        recall_ks=recall_ks,
        skip_eval=args.skip_eval,
    )

    cand_csv = metadata_out / f"s7b1_dinov2_candidate_scores_{run_name}.csv"
    query_csv = metadata_out / f"s7b1_dinov2_query_summary_{run_name}.csv"
    recall_csv = metadata_out / f"s7b1_dinov2_recall_summary_{run_name}.csv"
    scene_csv = metadata_out / f"s7b1_dinov2_scene_summary_{run_name}.csv"
    summary_json = report_out / f"s7b1_dinov2_summary_{run_name}.json"

    recall_fig = figure_out / f"s7b1_dinov2_recall_at_k_{run_name}.png"
    top1_fig = figure_out / f"s7b1_dinov2_top1_error_hist_{run_name}.png"
    oracle_fig = figure_out / f"s7b1_dinov2_oracle_error_hist_{run_name}.png"

    cand.to_csv(cand_csv, index=False)
    query.to_csv(query_csv, index=False)
    recall.to_csv(recall_csv, index=False)
    scene.to_csv(scene_csv, index=False)

    plot_recall(recall, recall_fig, args.threshold_m)
    plot_hist(query["top1_error_m_eval_only"], top1_fig, "S7B.1 DINOv2 top-1 error", "Top-1 error [m], clipped p98")
    plot_hist(query["oracle_error_m_eval_only"], oracle_fig, "S7B.1 DINOv2 oracle error in Top-K", "Oracle error [m], clipped p98")

    top1_hits = int(query["top1_hit_le_threshold_eval_only"].sum())
    topk_col = f"hit_at_{max(recall_ks)}_eval_only"
    topk_hits = int(query[topk_col].sum()) if topk_col in query.columns else 0

    summary = {
        "stage": "S7B.1_dinov2_global_retrieval_baseline",
        "status": "COMPLETE",
        "run_name": run_name,
        "repo_root": str(root),
        "queries": int(len(query_df)),
        "satellite_tiles": int(len(sat_df)),
        "top_k": int(args.top_k),
        "threshold_m": float(args.threshold_m),
        "recall_ks": recall_ks,
        "backend": extractor.backend,
        "device": extractor.device,
        "model_name": args.model_name,
        "hf_model": args.hf_model,
        "pooling": args.pooling,
        "image_size": int(args.image_size),
        "descriptor_dim": int(query_cache.descriptors.shape[1]),
        "satellite_descriptor_runtime_s": sat_cache.meta.get("runtime_s"),
        "query_descriptor_runtime_s": query_cache.meta.get("runtime_s"),
        "retrieval_runtime_s": retrieval_runtime_s,
        "top1_hits_le_threshold": top1_hits,
        "top1_hit_rate": float(top1_hits / len(query)) if len(query) else 0.0,
        f"top{max(recall_ks)}_hits_le_threshold": topk_hits,
        f"top{max(recall_ks)}_recall": float(topk_hits / len(query)) if len(query) else 0.0,
        "median_top1_error_m_eval_only": float(pd.to_numeric(query["top1_error_m_eval_only"], errors="coerce").median()),
        "median_oracle_error_m_eval_only": float(pd.to_numeric(query["oracle_error_m_eval_only"], errors="coerce").median()),
        "median_oracle_rank_eval_only": float(pd.to_numeric(query["oracle_rank_eval_only"], errors="coerce").median()),
        "outputs": {
            "candidate_scores_csv": str(cand_csv),
            "query_summary_csv": str(query_csv),
            "recall_summary_csv": str(recall_csv),
            "scene_summary_csv": str(scene_csv),
            "summary_json": str(summary_json),
            "environment_json": str(env_path),
            "query_descriptor_cache": str(q_cache_path),
            "satellite_descriptor_cache": str(s_cache_path),
            "recall_figure": str(recall_fig),
            "top1_error_hist": str(top1_fig),
            "oracle_error_hist": str(oracle_fig),
        },
        "locked_rule": (
            "DINOv2 descriptors and rankings do not use GT/reference/GNSS. "
            "lon/lat/error metrics are computed only after ranking for offline evaluation."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2, default=json_safe), encoding="utf-8")

    print()
    print("S7B.1 — DINOv2 Global Retrieval Baseline")
    print("----------------------------------------")
    print("Status:                         COMPLETE")
    print(f"Run name:                       {run_name}")
    print(f"Backend selected:               {extractor.backend}")
    print(f"Device selected:                {extractor.device}")
    print(f"Queries:                        {len(query_df)}")
    print(f"Satellite tiles:                {len(sat_df)}")
    print(f"Descriptor dim:                 {query_cache.descriptors.shape[1]}")
    print(f"Top-K evaluated:                {args.top_k}")
    print(f"Threshold m:                    {args.threshold_m:.1f}")
    print(f"Top-1 hits:                     {top1_hits}/{len(query)}")
    print(f"Top-{max(recall_ks)} hits:                  {topk_hits}/{len(query)}")
    print(f"Median top-1 error m:           {summary['median_top1_error_m_eval_only']:.3f}")
    print(f"Median oracle error m:          {summary['median_oracle_error_m_eval_only']:.3f}")
    print(f"Median oracle rank:             {summary['median_oracle_rank_eval_only']:.3f}")
    print(f"Retrieval runtime s:            {retrieval_runtime_s:.3f}")
    print()
    print("Recall summary:")
    print(recall.to_string(index=False))
    print()
    print("Scene summary:")
    print(scene.to_string(index=False))
    print()
    print(f"Candidate scores CSV:           {cand_csv}")
    print(f"Query summary CSV:              {query_csv}")
    print(f"Recall summary CSV:             {recall_csv}")
    print(f"Scene summary CSV:              {scene_csv}")
    print(f"Summary JSON:                   {summary_json}")
    print(f"Environment JSON:               {env_path}")
    print(f"Recall figure:                  {recall_fig}")
    print(f"Top1 error hist:                {top1_fig}")
    print(f"Oracle error hist:              {oracle_fig}")
    print()
    print("Locked rule: GT/reference coordinates used only after ranking for offline evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
