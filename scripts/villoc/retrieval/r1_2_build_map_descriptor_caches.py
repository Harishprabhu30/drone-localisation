#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import yaml

from uavloc.retrieval import (
    DinoV2CachedRetrievalBackend,
    RetrievalRepresentation,
)


ROOT = Path.cwd().resolve()

DEFAULT_PROTOCOL = Path(
    "configs/research/retrieval_candidate_pool_v2_r1.yaml"
)
DEFAULT_R11_REPORT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r1_map_pyramid/reports/"
    "r1_1_missing_levels_build_and_audit.json"
)
DEFAULT_REPORT = Path(
    "outputs/research_runs/retrieval_candidate_pool_v2/"
    "r1_map_pyramid/reports/"
    "r1_2_map_descriptor_cache_build_and_validation.json"
)

EXPECTED_R12_LEVELS = ("384_s256", "768_s256")
EXPECTED_DESCRIPTOR_DIM = 384


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_builder_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(path)

    module_name = "villoc_s8_11bc_builder"

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import descriptor builder from {path}"
        )

    module = importlib.util.module_from_spec(spec)

    # dataclasses resolves annotation/type metadata through
    # sys.modules[cls.__module__] while the class decorator runs.
    # A module created manually with module_from_spec() is not inserted into
    # sys.modules automatically, so register it before exec_module().
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise

    return module


def select_r12_levels(protocol_cfg: dict) -> list[dict]:
    levels = [
        level
        for level in protocol_cfg["experiment"]["levels"]
        if str(level["status"]) == "generate"
    ]

    names = tuple(
        str(level["name"])
        for level in levels
    )

    if set(names) != set(EXPECTED_R12_LEVELS):
        raise RuntimeError(
            "R1.2 is intentionally scoped to the two R1.1 research levels. "
            f"Expected {list(EXPECTED_R12_LEVELS)}, got {list(names)}."
        )

    by_name = {
        str(level["name"]): level
        for level in levels
    }

    return [
        by_name[name]
        for name in EXPECTED_R12_LEVELS
    ]


def cache_index_path(cache_path: Path) -> Path:
    return cache_path.with_name(
        cache_path.stem + "_index.csv"
    )


def inspect_cache_state(
    cache_path: Path,
    index_path: Path,
) -> str:
    cache_exists = cache_path.exists()
    index_exists = index_path.exists()

    if not cache_exists and not index_exists:
        return "absent"

    if cache_exists and index_exists:
        return "complete_candidate"

    return "partial_or_inconsistent"


def validate_builder_protocol(
    builder_protocol,
    research_cfg: dict,
) -> None:
    expected = research_cfg["retrieval_protocol"]
    actual = asdict(builder_protocol)

    required = {
        "model_name": expected["model_name"],
        "image_size": int(expected["image_size"]),
        "crop_mode": expected["crop_mode"],
        "pooling": expected["pooling"],
        "normalization": expected["normalization"],
        "l2_normalize": bool(expected["l2_normalize"]),
        "descriptor_dtype": expected["descriptor_dtype"],
    }

    mismatches = {
        key: {
            "actual": actual.get(key),
            "expected": value,
        }
        for key, value in required.items()
        if actual.get(key) != value
    }

    if mismatches:
        raise RuntimeError(
            "S8.11BC builder protocol does not match frozen R1 protocol: "
            + json.dumps(mismatches, indent=2)
        )


def validate_cache_against_index(
    *,
    representation: RetrievalRepresentation,
    index_df: pd.DataFrame,
    expected_variant: str,
) -> dict:
    required = {"tile_id", "tile_path"}
    missing = required - set(index_df.columns)

    if missing:
        raise RuntimeError(
            f"{expected_variant}: tile index missing columns {sorted(missing)}"
        )

    expected_ids = index_df["tile_id"].astype(str).to_numpy()

    if not np.array_equal(
        representation.ids.astype(str),
        expected_ids,
    ):
        mismatch = np.flatnonzero(
            representation.ids.astype(str)
            != expected_ids
        )
        preview = mismatch[:10].tolist()
        raise RuntimeError(
            f"{expected_variant}: cache IDs do not exactly match tile-index "
            f"row order. First mismatched rows: {preview}"
        )

    if representation.descriptors.shape != (
        len(index_df),
        EXPECTED_DESCRIPTOR_DIM,
    ):
        raise RuntimeError(
            f"{expected_variant}: unexpected descriptor shape "
            f"{representation.descriptors.shape}; expected "
            f"({len(index_df)}, {EXPECTED_DESCRIPTOR_DIM})"
        )

    metadata = representation.metadata
    if metadata.get("cache_kind") != "map_cache":
        raise RuntimeError(
            f"{expected_variant}: cache_kind is not map_cache."
        )

    if str(metadata.get("variant")) != expected_variant:
        raise RuntimeError(
            f"{expected_variant}: metadata variant mismatch: "
            f"{metadata.get('variant')}"
        )

    if int(metadata.get("row_count", -1)) != len(index_df):
        raise RuntimeError(
            f"{expected_variant}: metadata row_count mismatch."
        )

    if list(metadata.get("descriptor_shape", [])) != [
        len(index_df),
        EXPECTED_DESCRIPTOR_DIM,
    ]:
        raise RuntimeError(
            f"{expected_variant}: metadata descriptor_shape mismatch."
        )

    norms = np.linalg.norm(
        representation.descriptors,
        axis=1,
    )

    return {
        "row_count": int(len(index_df)),
        "descriptor_shape": list(
            representation.descriptors.shape
        ),
        "descriptor_dtype": str(
            representation.descriptors.dtype
        ),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "tile_ids_exact_index_order": True,
        "finite_descriptors": bool(
            np.isfinite(
                representation.descriptors
            ).all()
        ),
        "metadata_variant": str(
            metadata.get("variant")
        ),
        "protocol_hash": metadata.get(
            "protocol_hash"
        ),
        "checkpoint_sha256": metadata.get(
            "checkpoint_sha256"
        ),
        "source_csv_sha256": metadata.get(
            "source_csv_sha256"
        ),
        "ids_hash": metadata.get(
            "ids_hash"
        ),
        "paths_hash": metadata.get(
            "paths_hash"
        ),
        "runtime_s": metadata.get(
            "runtime_s"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R1.2 build DINOv2 map descriptor caches only for the new "
            "384_s256 and 768_s256 map-footprint levels, then validate "
            "them against the frozen canonical query representation."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--r1-1-report",
        type=Path,
        default=DEFAULT_R11_REPORT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Descriptor batch size. Default 1 preserves the frozen CPU "
            "cache-build protocol."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild the two research map caches even when valid-looking "
            "cache/index pairs already exist. Frozen control assets are never "
            "targeted."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    args = parser.parse_args()

    protocol_path = resolve(args.protocol)
    r11_report_path = resolve(args.r1_1_report)

    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    if not r11_report_path.exists():
        raise FileNotFoundError(r11_report_path)

    cfg = yaml.safe_load(
        protocol_path.read_text()
    )

    r11 = json.loads(
        r11_report_path.read_text()
    )

    if (
        r11.get("status")
        != "PASS_R1_MISSING_LEVELS_GENERATED_AND_AUDITED"
    ):
        raise RuntimeError(
            "R1.1 geometry gate has not passed. "
            "R1.2 descriptor encoding is blocked."
        )

    if args.batch_size != 1:
        raise RuntimeError(
            "R1.2 primary scientific cache build keeps batch_size=1 to "
            "match the frozen CPU descriptor protocol. Do not change it in "
            "this stage."
        )

    levels = select_r12_levels(cfg)

    canonical_query_cache = resolve(
        cfg["control"]["query_cache"]
    )

    if not canonical_query_cache.exists():
        raise FileNotFoundError(
            canonical_query_cache
        )

    query_sha_before = sha256_file(
        canonical_query_cache
    )

    backend = DinoV2CachedRetrievalBackend()
    query_representation = backend.load_representation(
        canonical_query_cache
    )

    expected_query_count = int(
        cfg["control"]["query_count_expected"]
    )

    if len(query_representation.ids) != expected_query_count:
        raise RuntimeError(
            "Canonical query-cache count mismatch: "
            f"{len(query_representation.ids)} != {expected_query_count}"
        )

    builder_path = (
        ROOT
        / "scripts/villoc/s8_11bc_build_dinov2_caches.py"
    )
    builder = load_builder_module(
        builder_path
    )

    builder_protocol = builder.Protocol(
        device="cpu",
        batch_size=args.batch_size,
        image_size=int(
            cfg["retrieval_protocol"]["image_size"]
        ),
        crop_mode=str(
            cfg["retrieval_protocol"]["crop_mode"]
        ),
        pooling=str(
            cfg["retrieval_protocol"]["pooling"]
        ),
    )

    validate_builder_protocol(
        builder_protocol,
        cfg,
    )

    protocol_signature = backend.protocol_signature(
        query_representation.metadata
    )
    expected_signature = {
        key: cfg["retrieval_protocol"][key]
        for key in (
            "model_name",
            "image_size",
            "crop_mode",
            "pooling",
            "normalization",
            "l2_normalize",
            "descriptor_dtype",
        )
    }

    if protocol_signature != expected_signature:
        raise RuntimeError(
            "Canonical query cache does not match the frozen R1 retrieval "
            "protocol. "
            f"query={protocol_signature}, expected={expected_signature}"
        )

    target_records = []

    for level in levels:
        name = str(level["name"])
        source_index = resolve(
            level["index_csv"]
        )
        cache_path = resolve(
            level["map_cache"]
        )
        index_path = cache_index_path(
            cache_path
        )

        if not source_index.exists():
            raise FileNotFoundError(
                source_index
            )

        state = inspect_cache_state(
            cache_path,
            index_path,
        )

        if (
            state == "partial_or_inconsistent"
            and not args.force
        ):
            raise RuntimeError(
                f"{name}: partial cache output detected. "
                "Inspect the research-only descriptor path or rerun R1.2 "
                "with --force to rebuild this research cache."
            )

        tile_df = pd.read_csv(
            source_index,
            usecols=[
                "tile_id",
                "tile_path",
            ],
        ).copy()

        target_records.append(
            {
                "name": name,
                "source_index": source_index,
                "source_index_sha256": sha256_file(
                    source_index
                ),
                "cache_path": cache_path,
                "index_path": index_path,
                "tile_df": tile_df,
                "preexisting_state": state,
            }
        )

    # Load the exact promoted S8.11BC DINO implementation once and use it for
    # both new map levels. No query descriptors are encoded in R1.2.
    model, torch = builder.load_model(
        builder_protocol
    )

    results = []

    for target in target_records:
        name = target["name"]
        cache_path = target["cache_path"]
        index_path = target["index_path"]
        tile_df = target["tile_df"]
        source_index = target["source_index"]

        print()
        print("=" * 88)
        print(
            f"R1.2 — BUILD/VALIDATE MAP CACHE {name}"
        )
        print("=" * 88)

        meta = builder.build_or_skip_cache(
            df=tile_df,
            id_col="tile_id",
            path_col="tile_path",
            source_csv=source_index,
            cache_path=cache_path,
            index_path=index_path,
            input_columns=[
                "tile_id",
                "tile_path",
            ],
            protocol=builder_protocol,
            cache_kind="map_cache",
            variant=name,
            model=model,
            torch=torch,
            force=args.force,
        )

        if not cache_path.exists():
            raise RuntimeError(
                f"{name}: expected map cache missing after build: "
                f"{cache_path}"
            )

        if not index_path.exists():
            raise RuntimeError(
                f"{name}: expected descriptor index missing after build: "
                f"{index_path}"
            )

        representation = backend.load_representation(
            cache_path
        )

        # This is a protocol/shape/norm compatibility gate only. It performs
        # no ranking and uses no coordinates/oracles.
        backend.validate_pair(
            query_representation,
            representation,
        )

        validation = validate_cache_against_index(
            representation=representation,
            index_df=tile_df,
            expected_variant=name,
        )

        checkpoint_sha = validation[
            "checkpoint_sha256"
        ]
        query_checkpoint_sha = (
            query_representation.metadata.get(
                "checkpoint_sha256"
            )
        )

        if checkpoint_sha != query_checkpoint_sha:
            raise RuntimeError(
                f"{name}: checkpoint SHA mismatch with canonical query cache. "
                f"map={checkpoint_sha}, query={query_checkpoint_sha}"
            )

        results.append(
            {
                "variant": name,
                "source_index": str(
                    source_index
                ),
                "source_index_sha256": target[
                    "source_index_sha256"
                ],
                "map_cache": str(
                    cache_path
                ),
                "map_cache_sha256": sha256_file(
                    cache_path
                ),
                "descriptor_index": str(
                    index_path
                ),
                "descriptor_index_sha256": sha256_file(
                    index_path
                ),
                "preexisting_state": target[
                    "preexisting_state"
                ],
                "validation": validation,
                "builder_meta_protocol_hash": meta.get(
                    "protocol_hash"
                ),
            }
        )

    query_sha_after = sha256_file(
        canonical_query_cache
    )

    if query_sha_after != query_sha_before:
        raise RuntimeError(
            "Canonical query cache changed during R1.2. "
            "This violates the experiment contract."
        )

    report = {
        "stage": "R1.2",
        "status": "PASS_R1_MAP_DESCRIPTOR_CACHES_BUILT_AND_VALIDATED",
        "created_at_utc": now_utc(),
        "protocol": str(
            protocol_path
        ),
        "protocol_sha256": sha256_file(
            protocol_path
        ),
        "r1_1_gate": {
            "report": str(
                r11_report_path
            ),
            "report_sha256": sha256_file(
                r11_report_path
            ),
            "status": r11["status"],
        },
        "canonical_query_cache": {
            "path": str(
                canonical_query_cache
            ),
            "sha256_before": query_sha_before,
            "sha256_after": query_sha_after,
            "unchanged": True,
            "query_count": int(
                len(
                    query_representation.ids
                )
            ),
            "descriptor_shape": list(
                query_representation.descriptors.shape
            ),
            "protocol_signature": protocol_signature,
            "checkpoint_sha256": (
                query_representation.metadata.get(
                    "checkpoint_sha256"
                )
            ),
        },
        "descriptor_builder": {
            "implementation": str(
                builder_path
            ),
            "implementation_sha256": sha256_file(
                builder_path
            ),
            "protocol": asdict(
                builder_protocol
            ),
            "network_download_allowed": False,
        },
        "map_caches": results,
        "scope_guarantees": {
            "query_cache_rebuilt": False,
            "query_cache_modified": False,
            "frozen_512_s256_map_cache_modified": False,
            "existing_1024_s256_map_cache_modified": False,
            "coordinates_used_for_descriptor": False,
            "oracle_used_for_descriptor": False,
            "retrieval_ranking_run": False,
            "retrieval_metrics_computed": False,
            "orb_run": False,
            "bootstrap_or_state_run": False,
        },
        "next_stage": {
            "stage": "R1.3",
            "action": (
                "Run independent retrieval for 384_s256, frozen 512_s256, "
                "768_s256 and existing 1024_s256 through the A1 backend, "
                "then evaluate candidate-pool quality without ORB/state replay."
            ),
        },
    }

    report_path = resolve(
        args.report
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("R1.2 — MAP DESCRIPTOR CACHE GATE")
    print("=" * 88)
    print("status:", report["status"])
    print(
        "canonical query cache unchanged:",
        True,
    )

    for result in results:
        validation = result["validation"]
        print(
            f"{result['variant']:10s} "
            f"shape={validation['descriptor_shape']} "
            f"norm=[{validation['norm_min']:.6f}, "
            f"{validation['norm_max']:.6f}] "
            f"sha256={result['map_cache_sha256'][:12]}..."
        )

    print()
    print("report:", report_path)


if __name__ == "__main__":
    main()
