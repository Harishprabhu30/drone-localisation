'''
Command Executed:

source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src
CFG=configs/dataset_villoc_45deg.yaml
LOGDIR=outputs/villoc/45_deg/logs/s8_12d_45deg_dino
mkdir -p "$LOGDIR"

python scripts/villoc/s8_11bc_build_dinov2_caches.py \
  --config "$CFG" \
  --reuse-map-caches \
  --map-cache-root outputs/villoc/90_deg/descriptors \
  --image-size 224 \
  --crop-mode center_square \
  --pooling avgpatch \
  2>&1 | tee "$LOGDIR/s8_12d8_build_dinov2_query_cache_45deg_reuse_map.log"

2. running traj01 villoc dataset:

mkdir -p outputs/villoc/traj01_90deg_stable120m/logs/s8_11bc_dinov2_caches

python scripts/villoc/s8_11bc_build_dinov2_caches.py \
  --config configs/dataset_villoc_traj01_90deg_stable120m.yaml \
  --reuse-map-caches \
  --map-cache-root outputs/villoc/90_deg/descriptors \
  --batch-size 1 \
  --image-size 224 \
  --crop-mode center_square \
  --pooling avgpatch \
  2>&1 | tee \
  outputs/villoc/traj01_90deg_stable120m/logs/s8_11bc_dinov2_caches/s8_11bc_build_query_cache_reuse_map.log

'''

from __future__ import annotations

import argparse
import yaml
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def root_join(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def infer_query_manifest(root: Path, cfg: dict) -> Path:
    dataset = cfg["dataset"]
    output_root = root_join(root, dataset["output_root"])

    candidates = [
        output_root / "metadata/s8_10b_canonical_uav_query_manifest.csv",
        output_root / "metadata/s8_5_uav_frames_index_v_1fps.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find query manifest. Tried:\n" +
        "\n".join(str(p) for p in candidates)
    )


def infer_tile_indexes(root: Path, cfg: dict) -> dict[str, Path]:
    map_cfg = cfg.get("map", {})
    variants = map_cfg.get("tile_variants", {})

    if not variants:
        raise KeyError("Missing map.tile_variants in dataset config")

    out = {}

    if isinstance(variants, dict):
        for name, v in variants.items():
            out[name] = root_join(root, v["index_csv"])
        return out

    if isinstance(variants, list):
        for v in variants:
            out[v["name"]] = root_join(root, v["index_csv"])
        return out

    raise TypeError("map.tile_variants must be dict or list")


def descriptor_dirs(root: Path, cfg: dict) -> tuple[Path, Path]:
    dataset = cfg["dataset"]
    output_root = root_join(root, dataset["output_root"])
    return (
        output_root / "descriptors",
        output_root / "reports/s8_11bc",
    )


ROOT = Path.cwd().resolve()

QUERY_CSV = ROOT / "outputs/villoc/90_deg/metadata/s8_10b_canonical_uav_query_manifest.csv"

TILE_INDEXES = {
    "512_s256": ROOT / "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_512_s256.csv",
    "1024_s512": ROOT / "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s512.csv",
    "1024_s256": ROOT / "outputs/villoc/90_deg/metadata/s8_9_satellite_tile_index_1024_s256.csv",
}

OUT_DESC_DIR = ROOT / "outputs/villoc/90_deg/descriptors"
OUT_REPORT_DIR = ROOT / "outputs/villoc/90_deg/reports/s8_11bc"

TORCH_HUB_REPO = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"


@dataclass(frozen=True)
class Protocol:
    stage: str = "S8.11B_C"
    model_name: str = "dinov2_vits14"
    backend: str = "torch_hub_local"
    checkpoint_path: str = str(CHECKPOINT)
    torch_hub_repo: str = str(TORCH_HUB_REPO)
    internet_allowed: bool = False
    device: str = "cpu"
    image_size: int = 224
    crop_mode: str = "center_square"
    pooling: str = "avgpatch"
    normalization: str = "imagenet"
    l2_normalize: bool = True
    descriptor_dtype: str = "float32"
    seed: int = 7
    batch_size: int = 1


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ordered_hash(values: Iterable[object]) -> str:
    joined = "\n".join(str(v) for v in values)
    return sha256_text(joined)


def resolve_repo_path(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def preprocess_image(path: Path, image_size: int, crop_mode: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    if crop_mode == "center_square":
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    elif crop_mode == "resize_square":
        pass
    else:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")

    img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    return np.transpose(arr, (2, 0, 1))


def validate_input_table(
    df: pd.DataFrame,
    id_col: str,
    path_col: str,
    table_name: str,
) -> tuple[np.ndarray, list[Path]]:
    if id_col not in df.columns:
        raise KeyError(f"{table_name}: missing id_col={id_col}")
    if path_col not in df.columns:
        raise KeyError(f"{table_name}: missing path_col={path_col}")

    if df[id_col].isna().any():
        raise ValueError(f"{table_name}: null IDs in {id_col}")
    if df[path_col].isna().any():
        raise ValueError(f"{table_name}: null paths in {path_col}")

    ids = df[id_col].astype(str).to_numpy()

    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{table_name}: duplicate IDs in {id_col}")

    paths = [resolve_repo_path(p) for p in df[path_col].astype(str).tolist()]
    missing = [str(p) for p in paths if not p.exists()]

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{table_name}: missing image files count={len(missing)}\n{preview}"
        )

    return ids, paths


def protocol_hash(protocol: Protocol, source_csv_sha256: str, ids_hash: str, paths_hash: str) -> str:
    payload = {
        "protocol": asdict(protocol),
        "source_csv_sha256": source_csv_sha256,
        "ordered_ids_hash": ids_hash,
        "ordered_paths_hash": paths_hash,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def cache_matches(cache_path: Path, expected_protocol_hash: str) -> bool:
    if not cache_path.exists():
        return False

    try:
        data = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(data["meta_json"]))
        return meta.get("protocol_hash") == expected_protocol_hash
    except Exception:
        return False


def load_model(protocol: Protocol):
    if not TORCH_HUB_REPO.exists():
        raise FileNotFoundError(f"Missing local torch hub repo: {TORCH_HUB_REPO}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing local checkpoint: {CHECKPOINT}")

    os.environ["TORCH_HOME"] = str(Path.home() / ".cache/torch")

    import torch

    if protocol.device != "cpu":
        raise ValueError(
            "This S8.11 frozen baseline is CPU-only because CUDA/MPS were unavailable in S8.11A.0."
        )

    model = torch.hub.load(
        str(TORCH_HUB_REPO),
        protocol.model_name,
        source="local",
        pretrained=True,
    )
    model.eval().to(protocol.device)

    return model, torch


def encode_paths(
    paths: list[Path],
    model,
    torch,
    protocol: Protocol,
    label: str,
) -> np.ndarray:
    descs: list[np.ndarray] = []
    total = len(paths)
    started = time.time()

    with torch.inference_mode():
        for start in range(0, total, protocol.batch_size):
            end = min(start + protocol.batch_size, total)
            batch_paths = paths[start:end]

            arr = np.stack(
                [
                    preprocess_image(
                        p,
                        image_size=protocol.image_size,
                        crop_mode=protocol.crop_mode,
                    )
                    for p in batch_paths
                ],
                axis=0,
            )

            tensor = torch.from_numpy(arr).to(protocol.device)

            out = model.forward_features(tensor)
            if protocol.pooling == "avgpatch":
                patch = out["x_norm_patchtokens"]
                batch_desc = patch.mean(dim=1).detach().float().cpu().numpy()
            elif protocol.pooling == "cls":
                cls = out["x_norm_clstoken"]
                batch_desc = cls.detach().float().cpu().numpy()
            else:
                raise ValueError(f"Unsupported pooling: {protocol.pooling}")

            descs.append(batch_desc.astype(np.float32))

            done = end
            if done == total or done % 25 == 0:
                elapsed = time.time() - started
                print(f"[{label}] encoded {done}/{total} images | elapsed_s={elapsed:.2f}", flush=True)

    descriptors = np.vstack(descs).astype(np.float32)

    if protocol.l2_normalize:
        descriptors = l2_normalize_np(descriptors).astype(np.float32)

    return descriptors


def save_cache(
    cache_path: Path,
    index_path: Path,
    descriptors: np.ndarray,
    ids: np.ndarray,
    paths: list[Path],
    source_csv: Path,
    input_columns: list[str],
    protocol: Protocol,
    cache_kind: str,
    variant: str | None,
    runtime_s: float,
) -> dict:
    source_csv_sha = sha256_file(source_csv)
    ids_hash = ordered_hash(ids)
    rel_paths = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in paths]
    paths_hash = ordered_hash(rel_paths)
    p_hash = protocol_hash(protocol, source_csv_sha, ids_hash, paths_hash)

    meta = {
        "stage": protocol.stage,
        "cache_kind": cache_kind,
        "variant": variant,
        "created_at_utc": now_utc(),
        "root": str(ROOT),
        "source_csv": str(source_csv),
        "source_csv_sha256": source_csv_sha,
        "input_columns_used_for_model": input_columns,
        "row_count": int(len(ids)),
        "descriptor_shape": list(descriptors.shape),
        "descriptor_dim": int(descriptors.shape[1]),
        "ids_hash": ids_hash,
        "paths_hash": paths_hash,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "protocol": asdict(protocol),
        "protocol_hash": p_hash,
        "runtime_s": float(runtime_s),
        "failed_image_count": 0,
        "failed_images": [],
        "leakage_rule": {
            "coordinates_used_for_descriptor": False,
            "oracle_used_for_descriptor": False,
            "coordinates_allowed_only_for": [
                "coverage audit",
                "oracle construction",
                "visualization",
                "evaluation",
            ],
        },
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_path,
        descriptors=descriptors.astype(np.float32),
        ids=ids.astype(str),
        paths=np.asarray(rel_paths, dtype=str),
        meta_json=np.asarray(json.dumps(meta, indent=2)),
    )

    index_df = pd.DataFrame(
        {
            "row": np.arange(len(ids), dtype=int),
            "id": ids.astype(str),
            "image_path": rel_paths,
        }
    )
    index_df.to_csv(index_path, index=False)

    return meta


def build_or_skip_cache(
    *,
    df: pd.DataFrame,
    id_col: str,
    path_col: str,
    source_csv: Path,
    cache_path: Path,
    index_path: Path,
    input_columns: list[str],
    protocol: Protocol,
    cache_kind: str,
    variant: str | None,
    model,
    torch,
    force: bool,
) -> dict:
    ids, paths = validate_input_table(df, id_col, path_col, cache_kind)

    source_csv_sha = sha256_file(source_csv)
    rel_paths = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in paths]
    ids_hash = ordered_hash(ids)
    paths_hash = ordered_hash(rel_paths)
    expected_protocol_hash = protocol_hash(protocol, source_csv_sha, ids_hash, paths_hash)

    if not force and cache_matches(cache_path, expected_protocol_hash):
        print(f"[SKIP] valid cache exists: {cache_path}")
        data = np.load(cache_path, allow_pickle=False)
        return json.loads(str(data["meta_json"]))

    print(f"[BUILD] {cache_kind} variant={variant} rows={len(ids)}")
    started = time.time()

    descriptors = encode_paths(
        paths=paths,
        model=model,
        torch=torch,
        protocol=protocol,
        label=f"{cache_kind}:{variant or 'shared'}",
    )

    runtime_s = time.time() - started

    meta = save_cache(
        cache_path=cache_path,
        index_path=index_path,
        descriptors=descriptors,
        ids=ids,
        paths=paths,
        source_csv=source_csv,
        input_columns=input_columns,
        protocol=protocol,
        cache_kind=cache_kind,
        variant=variant,
        runtime_s=runtime_s,
    )

    print(f"[WROTE] {cache_path}")
    print(f"[WROTE] {index_path}")
    print(f"[SHAPE] {descriptors.shape}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-mode", choices=["center_square", "resize_square"], default="center_square")
    parser.add_argument("--pooling", choices=["avgpatch", "cls"], default="avgpatch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reuse-map-caches", action="store_true", help="Build query cache for this dataset but skip rebuilding map caches if they already exist.",)
    parser.add_argument("--map-cache-root", default=None, help="Optional descriptor directory for reused map caches, e.g. outputs/villoc/90_deg/descriptors",)

    args = parser.parse_args()

    cfg = load_yaml(root_join(ROOT, args.config))

    QUERY_CSV = infer_query_manifest(ROOT, cfg)
    TILE_INDEXES = infer_tile_indexes(ROOT, cfg)

    OUT_DESC_DIR, OUT_REPORT_DIR = descriptor_dirs(ROOT, cfg)

    if args.map_cache_root:
        MAP_DESC_DIR = root_join(ROOT, args.map_cache_root)
    else:
        MAP_DESC_DIR = OUT_DESC_DIR

    OUT_DESC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DESC_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(7)
    np.random.seed(7)

    protocol = Protocol(
        batch_size=args.batch_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        pooling=args.pooling,
    )

    print("S8.11B/C — Build DINOv2 descriptor caches")
    print("----------------------------------------")
    print("ROOT:", ROOT)
    print("QUERY_CSV:", QUERY_CSV)
    print("TILE_INDEXES:", json.dumps({k: str(v) for k, v in TILE_INDEXES.items()}, indent=2))
    print("TORCH_HUB_REPO:", TORCH_HUB_REPO)
    print("CHECKPOINT:", CHECKPOINT)
    print("PROTOCOL:", json.dumps(asdict(protocol), indent=2))

    if not QUERY_CSV.exists():
        raise FileNotFoundError(QUERY_CSV)

    for variant, path in TILE_INDEXES.items():
        if not path.exists():
            raise FileNotFoundError(path)

    # Strictly read only model-safe columns for descriptor extraction.
    q_df = pd.read_csv(
        QUERY_CSV,
        usecols=["token0_id", "query_id", "image_path"],
    ).copy()

    tile_dfs = {
        variant: pd.read_csv(path, usecols=["tile_id", "tile_path"]).copy()
        for variant, path in TILE_INDEXES.items()
    }

    model, torch = load_model(protocol)

    cache_tag = (
        f"{protocol.model_name}"
        f"_img{protocol.image_size}"
        f"_{protocol.crop_mode}"
        f"_{protocol.pooling}"
        f"_cpu"
    )

    summary: dict = {
        "stage": "S8.11B_C",
        "status": "PASS",
        "created_at_utc": now_utc(),
        "protocol": asdict(protocol),
        "outputs": {},
    }

    # S8.11C: shared UAV query cache.
    q_cache = OUT_DESC_DIR / f"s8_11c_dinov2_queries_v_1fps_{cache_tag}.npz"
    q_index = OUT_DESC_DIR / f"s8_11c_dinov2_queries_v_1fps_{cache_tag}_index.csv"

    q_meta = build_or_skip_cache(
        df=q_df,
        id_col="query_id",
        path_col="image_path",
        source_csv=QUERY_CSV,
        cache_path=q_cache,
        index_path=q_index,
        input_columns=["token0_id", "query_id", "image_path"],
        protocol=protocol,
        cache_kind="query_cache",
        variant="v_1fps",
        model=model,
        torch=torch,
        force=args.force,
    )

    summary["outputs"]["query_cache"] = {
        "cache": str(q_cache),
        "index": str(q_index),
        "meta": q_meta,
    }

    # S8.11B: independent map caches for all tile variants.
    summary["outputs"]["map_caches"] = {}

    for variant, df in tile_dfs.items():
        # cache = OUT_DESC_DIR / f"s8_11b_dinov2_map_{variant}_{cache_tag}.npz"
        cache = MAP_DESC_DIR / f"s8_11b_dinov2_map_{variant}_{cache_tag}.npz"
        # index = OUT_DESC_DIR / f"s8_11b_dinov2_map_{variant}_{cache_tag}_index.csv"
        index = MAP_DESC_DIR / f"s8_11b_dinov2_map_{variant}_{cache_tag}_index.csv"

        if args.reuse_map_caches and cache.exists() and index.exists() and not args.force:
            print(f"[SKIP MAP CACHE - REUSE] {variant}: {cache}")
            summary["outputs"]["map_caches"][variant] = {
                "cache_npz": str(cache),
                "index_csv": str(index),
                "status": "resused_existing_map_cache",
            }
            continue

        meta = build_or_skip_cache(
            df=df,
            id_col="tile_id",
            path_col="tile_path",
            source_csv=TILE_INDEXES[variant],
            cache_path=cache,
            index_path=index,
            input_columns=["tile_id", "tile_path"],
            protocol=protocol,
            cache_kind="map_cache",
            variant=variant,
            model=model,
            torch=torch,
            force=args.force,
        )

        summary["outputs"]["map_caches"][variant] = {
            "cache": str(cache),
            "index": str(index),
            "meta": meta,
        }

    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUT_REPORT_DIR / f"s8_11bc_dinov2_cache_build_summary_{cache_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("----------------------------------------")
    print("S8.11B/C COMPLETE")
    print("STATUS: PASS_DESCRIPTOR_CACHES_BUILT")
    print("Summary:", summary_path)

    print("\nCache outputs:")
    print("Query:", q_cache)
    for variant in TILE_INDEXES:
        print(
            variant + ":",
            OUT_DESC_DIR / f"s8_11b_dinov2_map_{variant}_{cache_tag}.npz",
        )


if __name__ == "__main__":
    main()
