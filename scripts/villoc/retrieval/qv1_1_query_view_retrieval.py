#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps
from pyproj import Transformer


ROOT = Path.cwd().resolve()
TORCH_HUB_REPO = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"

DEFAULT_CONFIG = Path(
    "configs/research/query_view_candidate_generation_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/query_view_candidate_generation_v1/qv1_1"
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def load_npz_representation(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    required = {"descriptors", "ids"}
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(
            f"{path}: missing NPZ keys {sorted(missing)}"
        )

    descriptors = np.asarray(
        data["descriptors"],
        dtype=np.float32,
    )
    ids = np.asarray(
        data["ids"],
    ).astype(str)

    meta = {}
    if "meta_json" in data.files:
        meta = json.loads(
            str(data["meta_json"])
        )

    if descriptors.ndim != 2:
        raise RuntimeError(
            f"{path}: expected 2-D descriptors, got {descriptors.shape}"
        )
    if len(ids) != len(descriptors):
        raise RuntimeError(
            f"{path}: ids/descriptors length mismatch"
        )

    return descriptors, ids, meta


def preprocess_image(
    path: Path,
    image_size: int,
    mode: str,
) -> np.ndarray:
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(
            raw
        ).convert("RGB")

    w, h = img.size
    side = min(w, h)

    if mode == "left_square":
        img = img.crop(
            (0, 0, side, side)
        )
    elif mode == "center_square":
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        img = img.crop(
            (x0, y0, x0 + side, y0 + side)
        )
    elif mode == "right_square":
        x0 = w - side
        img = img.crop(
            (x0, 0, x0 + side, side)
        )
    elif mode == "resize_square":
        pass
    else:
        raise ValueError(
            f"Unsupported query view mode: {mode}"
        )

    img = img.resize(
        (image_size, image_size),
        Image.Resampling.BICUBIC,
    )

    arr = (
        np.asarray(img).astype(
            np.float32
        )
        / 255.0
    )
    mean = np.asarray(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )
    std = np.asarray(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    arr = (
        arr - mean
    ) / std

    return np.transpose(
        arr,
        (2, 0, 1),
    )


def load_model(device: str):
    if not TORCH_HUB_REPO.exists():
        raise FileNotFoundError(
            TORCH_HUB_REPO
        )
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            CHECKPOINT
        )

    os.environ["TORCH_HOME"] = str(
        Path.home() / ".cache/torch"
    )

    import torch

    if device not in {
        "cpu",
        "cuda",
    }:
        raise ValueError(
            f"Unsupported device: {device}"
        )

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    model = torch.hub.load(
        str(TORCH_HUB_REPO),
        "dinov2_vits14",
        source="local",
        pretrained=True,
    )
    model.eval().to(device)

    return model, torch


def encode_queries(
    *,
    paths: list[Path],
    mode: str,
    model,
    torch,
    device: str,
    batch_size: int,
    image_size: int,
) -> np.ndarray:
    chunks = []
    started = time.time()

    with torch.inference_mode():
        for start in range(
            0,
            len(paths),
            batch_size,
        ):
            end = min(
                len(paths),
                start + batch_size,
            )

            batch = np.stack(
                [
                    preprocess_image(
                        path,
                        image_size,
                        mode,
                    )
                    for path in paths[
                        start:end
                    ]
                ],
                axis=0,
            )

            tensor = torch.from_numpy(
                batch
            ).to(device)

            out = model.forward_features(
                tensor
            )
            patch = out[
                "x_norm_patchtokens"
            ]
            desc = (
                patch.mean(dim=1)
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            chunks.append(desc)

            if (
                end == len(paths)
                or end % 25 == 0
            ):
                print(
                    f"[{mode}] {end}/{len(paths)} "
                    f"elapsed_s={time.time() - started:.1f}",
                    flush=True,
                )

    return l2_normalize(
        np.vstack(chunks).astype(
            np.float32
        )
    ).astype(np.float32)


def rank_all(
    query_desc: np.ndarray,
    map_desc: np.ndarray,
    map_ids: np.ndarray,
    depth: int,
) -> pd.DataFrame:
    scores = (
        query_desc
        @ map_desc.T
    )

    rows = []

    for i in range(
        scores.shape[0]
    ):
        order = np.argsort(
            -scores[i],
            kind="mergesort",
        )[:depth]

        for rank, j in enumerate(
            order,
            start=1,
        ):
            rows.append(
                {
                    "query_row": i,
                    "rank": int(rank),
                    "tile_id": str(
                        map_ids[j]
                    ),
                    "score": float(
                        scores[i, j]
                    ),
                }
            )

    return pd.DataFrame(rows)


def rrf_fuse(
    rankings: dict[str, pd.DataFrame],
    query_ids: list[int],
    *,
    rrf_k: float,
    per_view_depth: int,
    final_top_k: int,
) -> pd.DataFrame:
    rows = []

    grouped = {
        name: {
            int(qid): frame.sort_values(
                "rank"
            ).head(
                per_view_depth
            )
            for qid, frame in ranking.groupby(
                "query_id"
            )
        }
        for name, ranking in rankings.items()
    }

    for qid in query_ids:
        acc: dict[str, dict[str, Any]] = {}

        for view_name, by_q in grouped.items():
            frame = by_q[int(qid)]

            for row in frame.itertuples(
                index=False
            ):
                tile_id = str(
                    row.tile_id
                )
                rank = int(
                    row.rank
                )

                item = acc.setdefault(
                    tile_id,
                    {
                        "rrf_score": 0.0,
                        "best_rank": rank,
                        "view_ranks": {},
                    },
                )
                item["rrf_score"] += (
                    1.0
                    / (
                        float(rrf_k)
                        + float(rank)
                    )
                )
                item["best_rank"] = min(
                    int(
                        item["best_rank"]
                    ),
                    rank,
                )
                item[
                    "view_ranks"
                ][view_name] = rank

        ordered = sorted(
            acc.items(),
            key=lambda item: (
                -float(
                    item[1][
                        "rrf_score"
                    ]
                ),
                int(
                    item[1][
                        "best_rank"
                    ]
                ),
                str(
                    item[0]
                ),
            ),
        )[:final_top_k]

        for rank, (
            tile_id,
            item,
        ) in enumerate(
            ordered,
            start=1,
        ):
            rows.append(
                {
                    "query_id": int(
                        qid
                    ),
                    "rank": int(
                        rank
                    ),
                    "tile_id": tile_id,
                    "rrf_score": float(
                        item[
                            "rrf_score"
                        ]
                    ),
                    "best_individual_rank": int(
                        item[
                            "best_rank"
                        ]
                    ),
                    "view_ranks_json": json.dumps(
                        item[
                            "view_ranks"
                        ],
                        sort_keys=True,
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def load_reference_xy(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path
    ).copy()

    required = {
        "query_id",
        "eval_ref_lon",
        "eval_ref_lat",
    }
    missing = required - set(
        frame.columns
    )
    if missing:
        raise RuntimeError(
            f"Reference attachment missing {sorted(missing)}"
        )

    frame["query_id"] = pd.to_numeric(
        frame["query_id"],
        errors="raise",
    ).astype(int)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3346",
        always_xy=True,
    )
    x, y = transformer.transform(
        pd.to_numeric(
            frame[
                "eval_ref_lon"
            ],
            errors="raise",
        ).to_numpy(float),
        pd.to_numeric(
            frame[
                "eval_ref_lat"
            ],
            errors="raise",
        ).to_numpy(float),
    )

    frame[
        "gt_x"
    ] = x
    frame[
        "gt_y"
    ] = y

    return frame.set_index(
        "query_id"
    )


def evaluate_ranking(
    ranking: pd.DataFrame,
    map_index: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    frame = ranking.copy()

    frame["query_id"] = pd.to_numeric(
        frame["query_id"],
        errors="raise",
    ).astype(int)
    frame["rank"] = pd.to_numeric(
        frame["rank"],
        errors="raise",
    ).astype(int)
    frame["tile_id"] = frame[
        "tile_id"
    ].astype(str)

    map_lookup = (
        map_index.assign(
            tile_id=map_index[
                "tile_id"
            ].astype(str)
        )
        .set_index(
            "tile_id"
        )
    )

    evaluated_rows = []

    for row in frame.itertuples(
        index=False
    ):
        qid = int(
            row.query_id
        )
        tile_id = str(
            row.tile_id
        )
        tile = map_lookup.loc[
            tile_id
        ]
        ref = reference.loc[
            qid
        ]

        x = float(
            ref["gt_x"]
        )
        y = float(
            ref["gt_y"]
        )

        error = float(
            math.hypot(
                float(
                    tile[
                        "center_easting"
                    ]
                ) - x,
                float(
                    tile[
                        "center_northing"
                    ]
                ) - y,
            )
        )

        contains = bool(
            float(
                tile[
                    "left_easting"
                ]
            ) <= x
            <= float(
                tile[
                    "right_easting"
                ]
            )
            and float(
                tile[
                    "bottom_northing"
                ]
            ) <= y
            <= float(
                tile[
                    "top_northing"
                ]
            )
        )

        base = row._asdict()
        base.update(
            {
                "center_error_m": error,
                "contains_query": contains,
                "le40": bool(
                    error <= 40.0
                ),
                "le80": bool(
                    error <= 80.0
                ),
            }
        )
        evaluated_rows.append(
            base
        )

    evaluated = pd.DataFrame(
        evaluated_rows
    )

    query_rows = []

    for qid, group in evaluated.groupby(
        "query_id",
        sort=True,
    ):
        g = group.sort_values(
            "rank"
        )

        top1 = g.iloc[0]
        top5 = g[
            g["rank"] <= 5
        ]
        top20 = g[
            g["rank"] <= 20
        ]

        containing = g[
            g[
                "contains_query"
            ]
        ]
        le40 = g[
            g["le40"]
        ]
        le80 = g[
            g["le80"]
        ]

        query_rows.append(
            {
                "query_id": int(
                    qid
                ),
                "top1_tile_id": str(
                    top1[
                        "tile_id"
                    ]
                ),
                "top1_center_error_m": float(
                    top1[
                        "center_error_m"
                    ]
                ),
                "top1_contains": bool(
                    top1[
                        "contains_query"
                    ]
                ),
                "top1_le40": bool(
                    top1[
                        "le40"
                    ]
                ),
                "top1_le80": bool(
                    top1[
                        "le80"
                    ]
                ),
                "contain_r1": bool(
                    top1[
                        "contains_query"
                    ]
                ),
                "contain_r5": bool(
                    top5[
                        "contains_query"
                    ].any()
                ),
                "contain_r20": bool(
                    top20[
                        "contains_query"
                    ].any()
                ),
                "le40_r20": bool(
                    top20[
                        "le40"
                    ].any()
                ),
                "le80_r20": bool(
                    top20[
                        "le80"
                    ].any()
                ),
                "first_contain_rank": (
                    int(
                        containing.iloc[
                            0
                        ][
                            "rank"
                        ]
                    )
                    if not containing.empty
                    else None
                ),
                "first_le40_rank": (
                    int(
                        le40.iloc[
                            0
                        ][
                            "rank"
                        ]
                    )
                    if not le40.empty
                    else None
                ),
                "first_le80_rank": (
                    int(
                        le80.iloc[
                            0
                        ][
                            "rank"
                        ]
                    )
                    if not le80.empty
                    else None
                ),
            }
        )

    query_eval = pd.DataFrame(
        query_rows
    )

    summary = {
        "query_count": int(
            len(query_eval)
        ),
        "top1_contains_hits": int(
            query_eval[
                "top1_contains"
            ].sum()
        ),
        "top1_le40_hits": int(
            query_eval[
                "top1_le40"
            ].sum()
        ),
        "top1_le80_hits": int(
            query_eval[
                "top1_le80"
            ].sum()
        ),
        "contain_r1_hits": int(
            query_eval[
                "contain_r1"
            ].sum()
        ),
        "contain_r5_hits": int(
            query_eval[
                "contain_r5"
            ].sum()
        ),
        "contain_r20_hits": int(
            query_eval[
                "contain_r20"
            ].sum()
        ),
        "le40_r20_hits": int(
            query_eval[
                "le40_r20"
            ].sum()
        ),
        "le80_r20_hits": int(
            query_eval[
                "le80_r20"
            ].sum()
        ),
        "top1_center_error_median_m": float(
            query_eval[
                "top1_center_error_m"
            ].median()
        ),
        "top1_center_error_p95_m": float(
            query_eval[
                "top1_center_error_m"
            ].quantile(
                0.95
            )
        ),
    }

    return query_eval, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
        ],
        default="cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(
        resolve(args.config).read_text()
    )
    control = cfg[
        "control"
    ]
    qv = cfg[
        "qv1_1"
    ]

    output_root = resolve(
        args.output_root
    )
    desc_dir = (
        output_root
        / "descriptors"
    )
    blind_dir = (
        output_root
        / "blind_rankings"
    )
    eval_dir = (
        output_root
        / "postfreeze_evaluation"
    )

    for path in (
        desc_dir,
        blind_dir,
        eval_dir,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest_path = resolve(
        control[
            "blind_manifest"
        ]
    )
    map_cache_path = resolve(
        control[
            "map_cache"
        ]
    )
    map_index_path = resolve(
        control[
            "map_index"
        ]
    )
    center_cache_path = resolve(
        control[
            "historical_query_cache"
        ]
    )
    reference_path = resolve(
        control[
            "reference_attachment"
        ]
    )

    for path in (
        manifest_path,
        map_cache_path,
        map_index_path,
        center_cache_path,
        reference_path,
        CHECKPOINT,
        TORCH_HUB_REPO,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    # --------------------------------------------------------------
    # PHASE 1 — BLIND INPUTS / DESCRIPTOR BUILD / RANK FREEZE.
    # --------------------------------------------------------------
    manifest = pd.read_csv(
        manifest_path,
        usecols=[
            "token0_id",
            "query_id",
            "image_path",
        ],
    ).copy()

    manifest["query_id"] = pd.to_numeric(
        manifest[
            "query_id"
        ],
        errors="raise",
    ).astype(int)

    query_ids = manifest[
        "query_id"
    ].tolist()
    query_id_strings = [
        str(qid)
        for qid in query_ids
    ]
    query_paths = [
        resolve(path)
        for path in manifest[
            "image_path"
        ].astype(str)
    ]

    map_desc, map_ids, map_meta = (
        load_npz_representation(
            map_cache_path
        )
    )
    center_desc, center_ids, center_meta = (
        load_npz_representation(
            center_cache_path
        )
    )

    if center_ids.tolist() != query_id_strings:
        raise RuntimeError(
            "Historical center cache query-ID order does not match blind manifest"
        )

    map_desc = l2_normalize(
        map_desc
    ).astype(np.float32)
    center_desc = l2_normalize(
        center_desc
    ).astype(np.float32)

    checkpoint_sha = sha256_file(
        CHECKPOINT
    )

    for label, meta in (
        ("map cache", map_meta),
        ("center query cache", center_meta),
    ):
        recorded = meta.get(
            "checkpoint_sha256"
        )
        if (
            recorded is not None
            and str(recorded)
            != checkpoint_sha
        ):
            raise RuntimeError(
                f"{label} checkpoint SHA differs from local DINO checkpoint"
            )

    image_size = int(
        control[
            "image_size"
        ]
    )

    new_modes = list(
        qv[
            "encoded_query_variants"
        ]
    )

    descriptor_sets = {
        "center_square": center_desc,
    }

    need_model = False

    for mode in new_modes:
        cache_path = (
            desc_dir
            / (
                f"qv1_1_queries_{mode}_"
                f"dinov2_vits14_img{image_size}_"
                f"avgpatch_{args.device}.npz"
            )
        )

        if (
            cache_path.exists()
            and not args.force
        ):
            desc, ids, meta = (
                load_npz_representation(
                    cache_path
                )
            )
            if ids.tolist() != query_id_strings:
                raise RuntimeError(
                    f"{mode}: cached query IDs do not match manifest"
                )
            descriptor_sets[
                mode
            ] = l2_normalize(
                desc
            ).astype(np.float32)
            print(
                f"[REUSE] {mode}: {cache_path}"
            )
            continue

        if not need_model:
            model, torch = load_model(
                args.device
            )
            need_model = True

        started = time.time()
        desc = encode_queries(
            paths=query_paths,
            mode=mode,
            model=model,
            torch=torch,
            device=args.device,
            batch_size=int(
                args.batch_size
            ),
            image_size=image_size,
        )
        runtime_s = float(
            time.time()
            - started
        )

        meta = {
            "stage": "QV1.1",
            "created_at_utc": now_utc(),
            "query_view": mode,
            "model_name": control[
                "backbone"
            ],
            "image_size": image_size,
            "pooling": control[
                "pooling"
            ],
            "normalization": "imagenet",
            "l2_normalized": True,
            "device": args.device,
            "query_count": len(
                query_ids
            ),
            "checkpoint_sha256": checkpoint_sha,
            "blind_manifest_sha256": sha256_file(
                manifest_path
            ),
            "coordinates_used": False,
            "reference_used": False,
            "runtime_s": runtime_s,
        }

        np.savez_compressed(
            cache_path,
            descriptors=desc,
            ids=np.asarray(
                query_id_strings,
                dtype=str,
            ),
            paths=np.asarray(
                [
                    str(path)
                    for path in query_paths
                ],
                dtype=str,
            ),
            meta_json=np.asarray(
                json.dumps(
                    meta,
                    indent=2,
                )
            ),
        )

        descriptor_sets[
            mode
        ] = desc

        print(
            f"[WROTE] {mode}: {cache_path}"
        )

    per_crop_depth = int(
        qv[
            "multicrop_fusion"
        ][
            "per_crop_depth"
        ]
    )

    rankings: dict[
        str,
        pd.DataFrame
    ] = {}

    for mode in (
        "center_square",
        "left_square",
        "right_square",
        "resize_square",
    ):
        ranked = rank_all(
            descriptor_sets[
                mode
            ],
            map_desc,
            map_ids,
            depth=per_crop_depth,
        )
        ranked[
            "query_id"
        ] = [
            query_ids[
                int(i)
            ]
            for i in ranked[
                "query_row"
            ].tolist()
        ]
        ranked = ranked.drop(
            columns=[
                "query_row",
            ]
        )
        rankings[
            mode
        ] = ranked

    fused = rrf_fuse(
        {
            "left_square": rankings[
                "left_square"
            ],
            "center_square": rankings[
                "center_square"
            ],
            "right_square": rankings[
                "right_square"
            ],
        },
        query_ids,
        rrf_k=float(
            qv[
                "multicrop_fusion"
            ][
                "rrf_k"
            ]
        ),
        per_view_depth=per_crop_depth,
        final_top_k=int(
            qv[
                "multicrop_fusion"
            ][
                "final_top_k"
            ]
        ),
    )

    rankings[
        "lcr_rrf"
    ] = fused

    blind_paths = {}

    for name, ranking in rankings.items():
        path = (
            blind_dir
            / f"qv1_1_{name}_ranking.csv"
        )
        ranking.to_csv(
            path,
            index=False,
        )
        blind_paths[
            name
        ] = path

    blind_hashes = {
        name: sha256_file(
            path
        )
        for name, path in blind_paths.items()
    }

    # --------------------------------------------------------------
    # PHASE 2 — POST-FREEZE EVALUATION ONLY.
    # --------------------------------------------------------------
    map_index = pd.read_csv(
        map_index_path
    )
    reference = load_reference_xy(
        reference_path
    )

    summaries = {}
    named = {}
    eval_paths = {}

    for name, ranking in rankings.items():
        query_eval, summary = (
            evaluate_ranking(
                ranking,
                map_index,
                reference,
            )
        )

        summaries[
            name
        ] = summary

        path = (
            eval_dir
            / f"qv1_1_{name}_query_eval.csv"
        )
        query_eval.to_csv(
            path,
            index=False,
        )
        eval_paths[
            name
        ] = path

        for qid in cfg[
            "evaluation"
        ][
            "named_diagnostics"
        ]:
            key = f"q{int(qid)}"
            named.setdefault(
                key,
                {},
            )
            row = query_eval[
                query_eval[
                    "query_id"
                ]
                == int(qid)
            ]
            named[
                key
            ][name] = (
                row.iloc[
                    0
                ].to_dict()
                if len(row) == 1
                else None
            )

    for name, path in blind_paths.items():
        if sha256_file(
            path
        ) != blind_hashes[
            name
        ]:
            raise RuntimeError(
                f"Blind ranking changed during evaluation: {name}"
            )

    expected = qv[
        "expected_center_parity"
    ]
    center_summary = summaries[
        "center_square"
    ]

    parity = {
        "query_count": int(
            center_summary[
                "query_count"
            ]
        )
        == int(
            expected[
                "query_count"
            ]
        ),
        "top1_contains_hits": int(
            center_summary[
                "top1_contains_hits"
            ]
        )
        == int(
            expected[
                "top1_contains_hits"
            ]
        ),
        "top1_le40_hits": int(
            center_summary[
                "top1_le40_hits"
            ]
        )
        == int(
            expected[
                "top1_le40_hits"
            ]
        ),
        "top1_le80_hits": int(
            center_summary[
                "top1_le80_hits"
            ]
        )
        == int(
            expected[
                "top1_le80_hits"
            ]
        ),
    }

    if not all(
        parity.values()
    ):
        raise RuntimeError(
            "Historical center-square parity failed: "
            + json.dumps(
                {
                    "expected": expected,
                    "measured": center_summary,
                    "checks": parity,
                },
                indent=2,
            )
        )

    report = {
        "stage": "QV1.1",
        "status": "PASS_QV1_QUERY_VIEW_RETRIEVAL_COMPARISON",
        "created_at_utc": now_utc(),
        "scientific_question": (
            "Does preserving more horizontal query context improve DINOv2 "
            "candidate generation against the frozen 768_s256 map?"
        ),
        "control": {
            "map_cache": str(
                map_cache_path
            ),
            "map_cache_sha256": sha256_file(
                map_cache_path
            ),
            "historical_center_query_cache": str(
                center_cache_path
            ),
            "historical_center_query_cache_sha256": sha256_file(
                center_cache_path
            ),
            "checkpoint_sha256": checkpoint_sha,
            "map_variant": control[
                "map_variant"
            ],
            "image_size": image_size,
            "pooling": control[
                "pooling"
            ],
        },
        "blind_ranking_freeze": {
            name: {
                "path": str(
                    path
                ),
                "sha256": blind_hashes[
                    name
                ],
            }
            for name, path in blind_paths.items()
        },
        "historical_center_parity": {
            "pass": True,
            "checks": parity,
            "expected": expected,
        },
        "summary": summaries,
        "named_diagnostics": named,
        "scope_guarantees": {
            "map_representation_changed": False,
            "descriptor_backbone_changed": False,
            "query_representation_changed": True,
            "orb_run": False,
            "bootstrap_run": False,
            "state_run": False,
            "reference_used_before_ranking_freeze": False,
            "gt_tuned_crop_weights_used": False,
        },
        "outputs": {
            "descriptor_dir": str(
                desc_dir
            ),
            "evaluation_csvs": {
                name: str(
                    path
                )
                for name, path in eval_paths.items()
            },
        },
        "next_decision": (
            "Promote a query-view representation only if it materially "
            "improves the 768 center-square candidate-generation control. "
            "Otherwise close query crop as a weak axis and move to the next "
            "representation/backend hypothesis."
        ),
    }

    report_path = (
        output_root
        / "qv1_1_query_view_retrieval_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    print("=" * 124)
    print(
        "QV1.1 — QUERY-VIEW CANDIDATE GENERATION COMPARISON"
    )
    print("=" * 124)

    order = (
        "center_square",
        "left_square",
        "right_square",
        "resize_square",
        "lcr_rrf",
    )

    for name in order:
        s = summaries[
            name
        ]
        print(
            f"{name:18s} "
            f"Top1 contain={s['top1_contains_hits']:3d}/403 "
            f"Top1<=40={s['top1_le40_hits']:3d}/403 "
            f"Top1<=80={s['top1_le80_hits']:3d}/403 "
            f"R5 contain={s['contain_r5_hits']:3d}/403 "
            f"R20 contain={s['contain_r20_hits']:3d}/403 "
            f"R20<=40={s['le40_r20_hits']:3d}/403 "
            f"R20<=80={s['le80_r20_hits']:3d}/403 "
            f"median={s['top1_center_error_median_m']:.2f}"
        )

    print()
    print(
        "historical center parity:",
        parity,
    )

    for qid in cfg[
        "evaluation"
    ][
        "named_diagnostics"
    ]:
        print()
        print(
            f"q{int(qid)}:"
        )
        print(
            json.dumps(
                named[
                    f"q{int(qid)}"
                ],
                indent=2,
                default=str,
            )
        )

    print()
    print(
        "STATUS: PASS_QV1_QUERY_VIEW_RETRIEVAL_COMPARISON"
    )
    print(
        "report:",
        report_path,
    )


if __name__ == "__main__":
    main()
