'''
Command Executed:

python scripts/villoc/s8_12a_lightglue_preflight.py \
  2>&1 | tee outputs/villoc/90_deg/logs/s8_12a_lightglue_preflight.log

'''

from __future__ import annotations

import importlib.util
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path.cwd().resolve()

TAG = "dinov2_vits14_img224_center_square_avgpatch_cpu"
PRIMARY_VARIANT = "1024_s512"
PRIMARY_TOP_K = 20

CANDIDATE_CSV = (
    ROOT
    / "outputs/villoc/90_deg/retrieval/s8_11d"
    / f"s8_11d_topk_{PRIMARY_VARIANT}_{TAG}.csv"
)

QUERY_MANIFEST = (
    ROOT
    / "outputs/villoc/90_deg/metadata"
    / "s8_10b_canonical_uav_query_manifest.csv"
)

TILE_INDEX = (
    ROOT
    / "outputs/villoc/90_deg/metadata"
    / f"s8_9_satellite_tile_index_{PRIMARY_VARIANT}.csv"
)

POLICY_JSON = (
    ROOT
    / "outputs/villoc/90_deg/reports/s8_11e"
    / f"s8_11e_candidate_pool_policy_{TAG}.json"
)

REPORT_DIR = ROOT / "outputs/villoc/90_deg/reports/s8_12a"

REPORT_JSON = (
    REPORT_DIR
    / "s8_12a_lightglue_preflight.json"
)

PAIR_MANIFEST_CSV = (
    REPORT_DIR
    / f"s8_12a_primary_pairs_{PRIMARY_VARIANT}_top{PRIMARY_TOP_K}.csv"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_id(value) -> str:
    text = str(value).strip()

    if re.fullmatch(r"-?\d+\.0", text):
        return str(int(float(text)))

    return text


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    table_name: str,
) -> str:
    lower = {column.lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    raise KeyError(
        f"{table_name}: none of {candidates} found. "
        f"Available columns: {list(df.columns)}"
    )


def resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (ROOT / path).resolve()


def module_status(name: str) -> dict:
    spec = importlib.util.find_spec(name)

    return {
        "name": name,
        "available": spec is not None,
        "origin": str(spec.origin) if spec and spec.origin else None,
    }


def search_lightglue_sources() -> list[dict]:
    search_roots = [
        ROOT / "scripts",
        ROOT / "src",
    ]

    records = []

    patterns = [
        re.compile(r"\bLightGlue\b"),
        re.compile(r"\bSuperPoint\b"),
        re.compile(r"\bDISK\b"),
        re.compile(r"\bALIKED\b"),
        re.compile(r"\bSIFT\b"),
        re.compile(r"lightglue", re.IGNORECASE),
    ]

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for path in search_root.rglob("*.py"):
            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            matches = sorted(
                {
                    pattern.pattern
                    for pattern in patterns
                    if pattern.search(text)
                }
            )

            if not matches:
                continue

            imports = []

            for line in text.splitlines():
                stripped = line.strip()

                if (
                    "lightglue" in stripped.lower()
                    or "superpoint" in stripped.lower()
                    or "disk" in stripped.lower()
                    or "aliked" in stripped.lower()
                ):
                    imports.append(stripped)

            records.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "matched_patterns": matches,
                    "relevant_lines": imports[:30],
                }
            )

    return records


def find_candidate_checkpoints() -> list[dict]:
    roots = [
        ROOT,
        Path.home() / ".cache",
        Path.home() / ".cache/torch",
        Path.home() / ".cache/huggingface",
    ]

    filename_terms = [
        "lightglue",
        "superpoint",
        "disk",
        "aliked",
        "sift",
    ]

    suffixes = {
        ".pth",
        ".pt",
        ".ckpt",
        ".bin",
        ".safetensors",
    }

    discovered: dict[str, dict] = {}

    for base in roots:
        if not base.exists():
            continue

        try:
            iterator = base.rglob("*")
        except Exception:
            continue

        for path in iterator:
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue

            lower_name = path.name.lower()

            if path.suffix.lower() not in suffixes:
                continue

            if not any(term in lower_name for term in filename_terms):
                continue

            resolved = str(path.resolve())

            if resolved in discovered:
                continue

            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = None

            discovered[resolved] = {
                "path": resolved,
                "size_bytes": size_bytes,
            }

    return sorted(
        discovered.values(),
        key=lambda item: item["path"],
    )


def audit_inputs() -> dict:
    required = [
        CANDIDATE_CSV,
        QUERY_MANIFEST,
        TILE_INDEX,
        POLICY_JSON,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required S8.12A inputs:\n"
            + "\n".join(missing)
        )

    candidates = pd.read_csv(CANDIDATE_CSV)
    queries = pd.read_csv(QUERY_MANIFEST)
    tiles = pd.read_csv(TILE_INDEX)

    candidate_query_col = first_existing_column(
        candidates,
        ["query_id", "token0_id"],
        "candidate CSV",
    )
    candidate_tile_col = first_existing_column(
        candidates,
        ["tile_id", "sat_tile_id", "map_tile_id"],
        "candidate CSV",
    )
    rank_col = first_existing_column(
        candidates,
        ["rank", "candidate_rank"],
        "candidate CSV",
    )
    score_col = first_existing_column(
        candidates,
        ["score", "similarity", "cosine_similarity"],
        "candidate CSV",
    )

    query_id_col = first_existing_column(
        queries,
        ["query_id", "token0_id"],
        "query manifest",
    )
    query_path_col = first_existing_column(
        queries,
        ["image_path", "query_path"],
        "query manifest",
    )

    tile_id_col = first_existing_column(
        tiles,
        ["tile_id", "sat_tile_id", "map_tile_id"],
        "tile index",
    )
    tile_path_col = first_existing_column(
        tiles,
        ["tile_path", "image_path", "map_tile_path"],
        "tile index",
    )

    candidates = candidates.copy()
    queries = queries.copy()
    tiles = tiles.copy()

    candidates["query_id_norm"] = (
        candidates[candidate_query_col].map(normalize_id)
    )
    candidates["tile_id_norm"] = (
        candidates[candidate_tile_col].map(normalize_id)
    )
    candidates["rank_norm"] = (
        pd.to_numeric(candidates[rank_col], errors="raise")
        .astype(int)
    )

    queries["query_id_norm"] = (
        queries[query_id_col].map(normalize_id)
    )
    tiles["tile_id_norm"] = (
        tiles[tile_id_col].map(normalize_id)
    )

    if queries["query_id_norm"].duplicated().any():
        raise ValueError("Duplicate query IDs in query manifest")

    if tiles["tile_id_norm"].duplicated().any():
        raise ValueError("Duplicate tile IDs in tile index")

    primary = candidates[
        candidates["rank_norm"] <= PRIMARY_TOP_K
    ].copy()

    primary = primary.merge(
        queries[["query_id_norm", query_path_col]],
        on="query_id_norm",
        how="left",
        validate="many_to_one",
    )

    primary = primary.merge(
        tiles[["tile_id_norm", tile_path_col]],
        on="tile_id_norm",
        how="left",
        validate="many_to_one",
    )

    primary = primary.rename(
        columns={
            query_path_col: "query_image_path",
            tile_path_col: "tile_image_path",
            score_col: "retrieval_score",
        }
    )

    missing_query_join = int(
        primary["query_image_path"].isna().sum()
    )
    missing_tile_join = int(
        primary["tile_image_path"].isna().sum()
    )

    if missing_query_join:
        raise ValueError(
            f"{missing_query_join} candidate rows failed query join"
        )

    if missing_tile_join:
        raise ValueError(
            f"{missing_tile_join} candidate rows failed tile join"
        )

    primary["query_image_path"] = (
        primary["query_image_path"]
        .map(lambda value: str(resolve_path(value)))
    )
    primary["tile_image_path"] = (
        primary["tile_image_path"]
        .map(lambda value: str(resolve_path(value)))
    )

    primary["query_exists"] = (
        primary["query_image_path"]
        .map(lambda value: Path(value).is_file())
    )
    primary["tile_exists"] = (
        primary["tile_image_path"]
        .map(lambda value: Path(value).is_file())
    )

    primary = primary.sort_values(
        ["query_id_norm", "rank_norm"]
    ).reset_index(drop=True)

    output_columns = [
        "query_id_norm",
        "tile_id_norm",
        "rank_norm",
        "retrieval_score",
        "query_image_path",
        "tile_image_path",
        "query_exists",
        "tile_exists",
    ]

    optional_columns = [
        column
        for column in [
            "is_oracle",
            "center_error_m",
            "variant",
        ]
        if column in primary.columns
    ]

    pair_manifest = primary[
        output_columns + optional_columns
    ].rename(
        columns={
            "query_id_norm": "query_id",
            "tile_id_norm": "tile_id",
            "rank_norm": "retrieval_rank",
        }
    )

    pair_manifest.to_csv(
        PAIR_MANIFEST_CSV,
        index=False,
    )

    counts_per_query = (
        pair_manifest.groupby("query_id")
        .size()
    )

    rank_coverage = (
        pair_manifest.groupby("query_id")["retrieval_rank"]
        .apply(lambda values: sorted(values.tolist()))
    )

    malformed_rank_queries = [
        query_id
        for query_id, ranks in rank_coverage.items()
        if ranks != list(range(1, PRIMARY_TOP_K + 1))
    ]

    return {
        "candidate_csv_rows": int(len(candidates)),
        "primary_pair_rows": int(len(pair_manifest)),
        "unique_queries": int(
            pair_manifest["query_id"].nunique()
        ),
        "unique_tiles_in_primary_pool": int(
            pair_manifest["tile_id"].nunique()
        ),
        "minimum_pairs_per_query": int(counts_per_query.min()),
        "maximum_pairs_per_query": int(counts_per_query.max()),
        "malformed_rank_query_count": int(
            len(malformed_rank_queries)
        ),
        "malformed_rank_query_ids": malformed_rank_queries,
        "missing_query_file_count": int(
            (~pair_manifest["query_exists"]).sum()
        ),
        "missing_tile_file_count": int(
            (~pair_manifest["tile_exists"]).sum()
        ),
        "pair_manifest_csv": str(PAIR_MANIFEST_CSV),
        "candidate_columns": {
            "query_id": candidate_query_col,
            "tile_id": candidate_tile_col,
            "rank": rank_col,
            "score": score_col,
        },
        "query_columns": {
            "query_id": query_id_col,
            "image_path": query_path_col,
        },
        "tile_columns": {
            "tile_id": tile_id_col,
            "tile_path": tile_path_col,
        },
    }


def main() -> None:
    print("S8.12A — LightGlue Verifier Preflight")
    print("-------------------------------------")
    print("ROOT:", ROOT)
    print("Primary pool:", f"{PRIMARY_VARIANT}@{PRIMARY_TOP_K}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    package_status = [
        module_status("lightglue"),
        module_status("kornia"),
        module_status("cv2"),
        module_status("torchvision"),
    ]

    source_records = search_lightglue_sources()
    checkpoints = find_candidate_checkpoints()
    input_audit = audit_inputs()

    with POLICY_JSON.open("r", encoding="utf-8") as file:
        candidate_policy = json.load(file)

    package_available = {
        record["name"]: record["available"]
        for record in package_status
    }

    has_existing_lightglue_source = (
        len(source_records) > 0
    )

    all_paths_valid = (
        input_audit["missing_query_file_count"] == 0
        and input_audit["missing_tile_file_count"] == 0
    )

    pool_shape_valid = (
        input_audit["unique_queries"] == 115
        and input_audit["primary_pair_rows"]
        == 115 * PRIMARY_TOP_K
        and input_audit["minimum_pairs_per_query"]
        == PRIMARY_TOP_K
        and input_audit["maximum_pairs_per_query"]
        == PRIMARY_TOP_K
        and input_audit["malformed_rank_query_count"] == 0
    )

    report = {
        "stage": "S8.12A",
        "created_at_utc": utc_now(),
        "root": str(ROOT),
        "python": {
            "version": sys.version,
            "platform": platform.platform(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ),
            "selected_device_candidate": (
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    )
                    else "cpu"
                )
            ),
        },
        "numpy_version": np.__version__,
        "packages": package_status,
        "existing_lightglue_sources": source_records,
        "candidate_checkpoints": checkpoints,
        "input_audit": input_audit,
        "frozen_candidate_policy": candidate_policy,
        "proposed_verifier_protocol": {
            "candidate_variant": PRIMARY_VARIANT,
            "candidate_top_k": PRIMARY_TOP_K,
            "query_count": 115,
            "pair_count": 115 * PRIMARY_TOP_K,
            "retrieval_order_preserved": True,
            "gps_used_for_matching": False,
            "oracle_used_for_matching": False,
            "coordinates_used_for_matching": False,
            "evaluation_fields_retained_only_after_matching": [
                "is_oracle",
                "center_error_m",
            ],
            "initial_frontend_preference": (
                "reuse exact working SatLoc LightGlue frontend"
            ),
            "initial_run_policy": (
                "small smoke subset before full 2300-pair run"
            ),
        },
        "checks": {
            "candidate_pool_shape_valid": pool_shape_valid,
            "all_image_paths_valid": all_paths_valid,
            "lightglue_python_package_available": (
                package_available.get("lightglue", False)
            ),
            "existing_lightglue_source_found": (
                has_existing_lightglue_source
            ),
        },
    }

    mandatory_pass = (
        pool_shape_valid
        and all_paths_valid
        and (
            package_available.get("lightglue", False)
            or has_existing_lightglue_source
        )
    )

    report["status"] = (
        "PASS_LIGHTGLUE_PREFLIGHT"
        if mandatory_pass
        else "NEEDS_LIGHTGLUE_PROTOCOL_RESOLUTION"
    )

    REPORT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("Environment:")
    print("Python:", sys.version.split()[0])
    print("Torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print(
        "MPS:",
        bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ),
    )

    print()
    print("Package status:")
    for record in package_status:
        print(
            f"{record['name']}: "
            f"{'AVAILABLE' if record['available'] else 'NOT_FOUND'}"
        )

    print()
    print("Candidate-pool audit:")
    print(
        "Queries:",
        input_audit["unique_queries"],
    )
    print(
        "Pairs:",
        input_audit["primary_pair_rows"],
    )
    print(
        "Pairs/query:",
        input_audit["minimum_pairs_per_query"],
        "to",
        input_audit["maximum_pairs_per_query"],
    )
    print(
        "Missing query files:",
        input_audit["missing_query_file_count"],
    )
    print(
        "Missing tile files:",
        input_audit["missing_tile_file_count"],
    )
    print(
        "Malformed rank queries:",
        input_audit["malformed_rank_query_count"],
    )

    print()
    print(
        "Existing LightGlue-related source files:",
        len(source_records),
    )
    for record in source_records[:20]:
        print(" -", record["path"])

    print()
    print(
        "Candidate LightGlue-related checkpoints:",
        len(checkpoints),
    )
    for checkpoint in checkpoints[:20]:
        print(" -", checkpoint["path"])

    print()
    print("-------------------------------------")
    print("S8.12A COMPLETE")
    print("STATUS:", report["status"])
    print("Pair manifest:", PAIR_MANIFEST_CSV)
    print("Report:", REPORT_JSON)


if __name__ == "__main__":
    main()
