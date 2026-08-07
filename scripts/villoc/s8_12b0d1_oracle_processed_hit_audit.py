'''
Command Executed:

python scripts/villoc/s8_12b0d1_oracle_processed_hit_audit.py \
  2>&1 | tee \
  outputs/villoc/90_deg/logs/s8_12b0d1_oracle_processed_hit_audit.log

'''

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()

TAG = "dinov2_vits14_img224_center_square_avgpatch_cpu"
RUN_NAME = "s8_12b0_villoc_1024_s512_top20_smoke3"
THRESHOLD_M = 40.0

SELECTION_JSON = (
    ROOT
    / "outputs/villoc/90_deg/reports/s8_12b"
    / "s8_12b0_smoke_query_selection.json"
)

S811E_DIAGNOSTICS = (
    ROOT
    / "outputs/villoc/90_deg/reports/s8_11e"
    / f"s8_11e_query_level_scale_stride_diagnostics_{TAG}.csv"
)

PAIR_MANIFEST = (
    ROOT
    / "outputs/villoc/90_deg/reports/s8_12a"
    / "s8_12a_primary_pairs_1024_s512_top20.csv"
)

LG_CANDIDATES = (
    ROOT
    / "outputs/villoc/90_deg/metadata/s7d_lightglue"
    / f"s7d1_lightglue_candidate_scores_{RUN_NAME}.csv"
)

LG_RANKED = (
    ROOT
    / "outputs/villoc/90_deg/metadata/s7d_lightglue"
    / f"s7d1_lightglue_candidate_scores_ranked_{RUN_NAME}.csv"
)

LG_QUERY_SUMMARY = (
    ROOT
    / "outputs/villoc/90_deg/metadata/s7d_lightglue"
    / f"s7d1_lightglue_query_summary_{RUN_NAME}.csv"
)

ORACLE_AUDIT_CANDIDATES = [
    ROOT
    / "outputs/villoc/90_deg/metadata"
    / "s8_10b_uav_tile_oracle_1024_s512.csv",
]

OUT_CSV = (
    ROOT
    / "outputs/villoc/90_deg/reports/s8_12b"
    / "s8_12b0d1_oracle_processed_hit_reconciliation.csv"
)

OUT_JSON = (
    ROOT
    / "outputs/villoc/90_deg/reports/s8_12b"
    / "s8_12b0d1_oracle_processed_hit_reconciliation.json"
)


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if re.fullmatch(r"-?\d+\.0", text):
        return str(int(float(text)))

    return text


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    table_name: str,
    required: bool = True,
) -> str | None:
    lower = {str(column).lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if required:
        raise KeyError(
            f"{table_name}: none of {candidates} found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def parse_id_set(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()

    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {
            normalize_id(item)
            for item in value
            if normalize_id(item)
        }

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return set()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return {
                normalize_id(item)
                for item in parsed
                if normalize_id(item)
            }
    except Exception:
        pass

    cleaned = (
        text.replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("{", "")
        .replace("}", "")
        .replace('"', "")
        .replace("'", "")
    )

    parts = re.split(r"[,;|\s]+", cleaned)

    return {
        normalize_id(part)
        for part in parts
        if normalize_id(part)
    }


def locate_oracle_audit() -> Path | None:
    for path in ORACLE_AUDIT_CANDIDATES:
        if path.exists():
            return path

    search_roots = [
        ROOT / "outputs/villoc/90_deg/metadata",
        ROOT / "outputs/villoc/90_deg/reports",
    ]

    matches: list[Path] = []

    for search_root in search_roots:
        if search_root.exists():
            matches.extend(
                search_root.rglob("*s8_10b*oracle*.csv")
            )

    return sorted(matches)[0] if matches else None


def load_smoke_queries() -> tuple[list[str], dict[str, str]]:
    data = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))

    query_ids = []
    roles = {}

    for row in data["queries"]:
        query_id = normalize_id(row["query_id"])
        query_ids.append(query_id)
        roles[query_id] = str(row["role"])

    return query_ids, roles


def load_oracle_sets(
    smoke_query_ids: list[str],
) -> tuple[dict[str, set[str]], Path | None]:
    oracle_path = locate_oracle_audit()

    if oracle_path is None:
        return {}, None

    df = pd.read_csv(oracle_path)

    query_col = find_column(
        df,
        ["query_id", "token0_id", "sample_id"],
        "S8.10B oracle audit",
    )

    packed_col = find_column(
        df,
        ["oracle_tile_ids"],
        "S8.10B oracle audit",
        required=False,
    )

    single_col = find_column(
        df,
        [
            "oracle_tile_id",
            "best_oracle_tile_id",
            "nearest_tile_id",
        ],
        "S8.10B oracle audit",
        required=False,
    )

    has_oracle_col = find_column(
        df,
        ["has_oracle_tile", "has_oracle"],
        "S8.10B oracle audit",
        required=False,
    )

    df["_query_id"] = df[query_col].map(normalize_id)
    df = df[df["_query_id"].isin(smoke_query_ids)].copy()

    oracle_sets: dict[str, set[str]] = {
        query_id: set()
        for query_id in smoke_query_ids
    }

    for _, row in df.iterrows():
        query_id = row["_query_id"]

        if has_oracle_col is not None:
            has_oracle_value = row[has_oracle_col]

            if isinstance(has_oracle_value, str):
                has_oracle = (
                    has_oracle_value.strip().lower()
                    in {"true", "1", "yes", "y"}
                )
            else:
                has_oracle = bool(has_oracle_value)

            if not has_oracle:
                continue

        if packed_col is not None:
            oracle_sets[query_id].update(
                parse_id_set(row[packed_col])
            )

        if single_col is not None:
            oracle_id = normalize_id(row[single_col])

            if oracle_id:
                oracle_sets[query_id].add(oracle_id)

    return oracle_sets, oracle_path


def truthy(value: Any) -> bool:
    if pd.isna(value):
        return False

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return bool(value)


def main() -> None:
    print("S8.12B.0D.1 — Oracle Processed-Hit Audit")
    print("----------------------------------------")

    required = [
        SELECTION_JSON,
        S811E_DIAGNOSTICS,
        PAIR_MANIFEST,
        LG_CANDIDATES,
        LG_RANKED,
        LG_QUERY_SUMMARY,
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    smoke_query_ids, roles = load_smoke_queries()
    oracle_sets, oracle_path = load_oracle_sets(smoke_query_ids)

    diagnostics = pd.read_csv(S811E_DIAGNOSTICS)
    pairs = pd.read_csv(PAIR_MANIFEST)
    candidates = pd.read_csv(LG_CANDIDATES)
    ranked = pd.read_csv(LG_RANKED)
    query_summary = pd.read_csv(LG_QUERY_SUMMARY)

    diag_query_col = find_column(
        diagnostics,
        ["query_id", "token0_id"],
        "S8.11E diagnostics",
    )
    pair_query_col = find_column(
        pairs,
        ["query_id", "token0_id"],
        "S8.12A pair manifest",
    )
    pair_tile_col = find_column(
        pairs,
        ["tile_id", "sat_tile_id"],
        "S8.12A pair manifest",
    )
    pair_rank_col = find_column(
        pairs,
        ["retrieval_rank", "rank", "candidate_rank"],
        "S8.12A pair manifest",
    )

    candidate_query_col = find_column(
        candidates,
        ["query_id", "token0_id"],
        "LightGlue candidates",
    )
    candidate_tile_col = find_column(
        candidates,
        ["tile_id", "sat_tile_id"],
        "LightGlue candidates",
    )

    ranked_query_col = find_column(
        ranked,
        ["query_id", "token0_id"],
        "LightGlue ranked candidates",
    )
    ranked_tile_col = find_column(
        ranked,
        ["tile_id", "sat_tile_id"],
        "LightGlue ranked candidates",
    )

    lg_rank_col = find_column(
        ranked,
        [
            "lightglue_rank",
            "lg_rank",
            "policy_rank",
            "rank_lightglue_only",
        ],
        "LightGlue ranked candidates",
        required=False,
    )

    candidate_error_col = find_column(
        candidates,
        [
            "eval_error_m",
            "center_error_m",
            "tile_center_error_m",
            "candidate_error_m",
            "error_m",
            "chosen_error_m",
            "distance_m",
        ],
        "LightGlue candidates",
        required=False,
    )

    candidate_is_oracle_col = find_column(
        candidates,
        [
            "is_oracle",
            "oracle_candidate",
            "candidate_is_oracle",
        ],
        "LightGlue candidates",
        required=False,
    )

    summary_query_col = find_column(
        query_summary,
        ["query_id", "token0_id"],
        "LightGlue query summary",
    )

    summary_oracle_hit_col = find_column(
        query_summary,
        [
            "oracle_processed_hit_le_threshold",
            "oracle_processed_hit",
            "oracle_hit",
        ],
        "LightGlue query summary",
        required=False,
    )

    summary_oracle_error_col = find_column(
        query_summary,
        [
            "oracle_processed_error_m",
            "oracle_error_m",
            "best_processed_error_m",
        ],
        "LightGlue query summary",
        required=False,
    )

    for df, col in [
        (diagnostics, diag_query_col),
        (pairs, pair_query_col),
        (candidates, candidate_query_col),
        (ranked, ranked_query_col),
        (query_summary, summary_query_col),
    ]:
        df["_query_id"] = df[col].map(normalize_id)

    pairs["_tile_id"] = pairs[pair_tile_col].map(normalize_id)
    ranked["_tile_id"] = ranked[ranked_tile_col].map(normalize_id)
    candidates["_tile_id"] = (
        candidates[candidate_tile_col].map(normalize_id)
    )

    diag_rank_col = "1024_s512__first_oracle_rank"
    diag_error_col = "1024_s512__top1_center_error_m"

    if diag_rank_col not in diagnostics.columns:
        raise KeyError(f"Missing {diag_rank_col}")

    records = []

    for query_id in smoke_query_ids:
        role = roles[query_id]

        diag_row = diagnostics[
            diagnostics["_query_id"] == query_id
        ]

        query_pairs = pairs[
            pairs["_query_id"] == query_id
        ].copy()

        query_candidates = candidates[
            candidates["_query_id"] == query_id
        ].copy()

        query_ranked = ranked[
            ranked["_query_id"] == query_id
        ].copy()

        query_summary_row = query_summary[
            query_summary["_query_id"] == query_id
        ]

        if diag_row.empty:
            raise ValueError(
                f"Query {query_id}: missing from S8.11E diagnostics"
            )

        diag_row = diag_row.iloc[0]

        oracle_ids = oracle_sets.get(query_id, set())

        pair_tile_ids = set(query_pairs["_tile_id"])
        exact_oracle_ids_processed = sorted(
            oracle_ids.intersection(pair_tile_ids)
        )

        exact_oracle_present = len(exact_oracle_ids_processed) > 0

        exact_oracle_retrieval_ranks = sorted(
            query_pairs.loc[
                query_pairs["_tile_id"].isin(oracle_ids),
                pair_rank_col,
            ]
            .astype(int)
            .tolist()
        )

        lg_exact_oracle_ranks = []

        if lg_rank_col is not None and oracle_ids:
            lg_exact_oracle_ranks = sorted(
                pd.to_numeric(
                    query_ranked.loc[
                        query_ranked["_tile_id"].isin(oracle_ids),
                        lg_rank_col,
                    ],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )

        if candidate_error_col is not None:
            errors = pd.to_numeric(
                query_candidates[candidate_error_col],
                errors="coerce",
            )

            finite_errors = errors.dropna()

            minimum_processed_center_error_m = (
                float(finite_errors.min())
                if not finite_errors.empty
                else None
            )

            threshold_hit_recomputed = (
                minimum_processed_center_error_m is not None
                and minimum_processed_center_error_m <= THRESHOLD_M
            )
        else:
            minimum_processed_center_error_m = None
            threshold_hit_recomputed = None

        if (
            candidate_is_oracle_col is not None
            and not query_candidates.empty
        ):
            source_oracle_rows = int(
                query_candidates[
                    candidate_is_oracle_col
                ].map(truthy).sum()
            )
        else:
            source_oracle_rows = None

        if (
            summary_oracle_hit_col is not None
            and not query_summary_row.empty
        ):
            s7d_oracle_processed_hit = truthy(
                query_summary_row.iloc[0][
                    summary_oracle_hit_col
                ]
            )
        else:
            s7d_oracle_processed_hit = None

        if (
            summary_oracle_error_col is not None
            and not query_summary_row.empty
        ):
            value = pd.to_numeric(
                pd.Series([
                    query_summary_row.iloc[0][
                        summary_oracle_error_col
                    ]
                ]),
                errors="coerce",
            ).iloc[0]

            s7d_oracle_processed_error_m = (
                float(value)
                if pd.notna(value)
                else None
            )
        else:
            s7d_oracle_processed_error_m = None

        s811e_first_oracle_rank = int(
            float(diag_row[diag_rank_col])
        )

        classification = []

        if exact_oracle_present:
            classification.append(
                "EXACT_S8_10B_ORACLE_PRESENT_IN_TOP20"
            )
        else:
            classification.append(
                "EXACT_S8_10B_ORACLE_NOT_FOUND_IN_TOP20"
            )

        if threshold_hit_recomputed is True:
            classification.append(
                "HAS_PROCESSED_CANDIDATE_WITHIN_40M"
            )
        elif threshold_hit_recomputed is False:
            classification.append(
                "NO_PROCESSED_CANDIDATE_WITHIN_40M"
            )
        else:
            classification.append(
                "CENTER_ERROR_COLUMN_UNAVAILABLE"
            )

        if (
            exact_oracle_present
            and threshold_hit_recomputed is False
        ):
            classification.append(
                "ORACLE_SET_VS_CENTER_THRESHOLD_DEFINITION_MISMATCH"
            )

        if (
            s7d_oracle_processed_hit is not None
            and threshold_hit_recomputed is not None
            and s7d_oracle_processed_hit
            != threshold_hit_recomputed
        ):
            classification.append(
                "S7D_SUMMARY_VS_RECOMPUTED_THRESHOLD_MISMATCH"
            )

        records.append(
            {
                "role": role,
                "query_id": query_id,
                "s8_11e_first_oracle_rank": (
                    s811e_first_oracle_rank
                ),
                "s8_11e_top1_error_m": float(
                    diag_row[diag_error_col]
                ),
                "s8_10b_oracle_tile_ids": "|".join(
                    sorted(oracle_ids)
                ),
                "s8_10b_oracle_tile_count": len(oracle_ids),
                "exact_oracle_present_in_processed_top20": (
                    exact_oracle_present
                ),
                "processed_exact_oracle_tile_ids": "|".join(
                    exact_oracle_ids_processed
                ),
                "processed_exact_oracle_retrieval_ranks": (
                    "|".join(
                        str(rank)
                        for rank in exact_oracle_retrieval_ranks
                    )
                ),
                "processed_exact_oracle_lightglue_ranks": (
                    "|".join(
                        str(rank)
                        for rank in lg_exact_oracle_ranks
                    )
                ),
                "minimum_processed_center_error_m": (
                    minimum_processed_center_error_m
                ),
                "recomputed_processed_hit_le_40m": (
                    threshold_hit_recomputed
                ),
                "s7d_oracle_processed_hit": (
                    s7d_oracle_processed_hit
                ),
                "s7d_oracle_processed_error_m": (
                    s7d_oracle_processed_error_m
                ),
                "lightglue_source_is_oracle_rows": (
                    source_oracle_rows
                ),
                "classification": "|".join(classification),
            }
        )

    result = pd.DataFrame(records)

    exact_membership_hits = int(
        result[
            "exact_oracle_present_in_processed_top20"
        ].sum()
    )

    recomputed_threshold_hits = int(
        result["recomputed_processed_hit_le_40m"]
        .fillna(False)
        .sum()
    )

    if result["s7d_oracle_processed_hit"].notna().any():
        reported_threshold_hits = int(
            result["s7d_oracle_processed_hit"]
            .fillna(False)
            .sum()
        )
    else:
        reported_threshold_hits = None

    if exact_membership_hits == 3 and recomputed_threshold_hits == 2:
        conclusion = (
            "CONFIRMED_METRIC_DEFINITION_DIFFERENCE: "
            "all three queries contain an S8.10B oracle tile in Top-20, "
            "but only two queries contain a processed candidate whose "
            "tile-centre error is <=40 m."
        )
        status = "PASS_ORACLE_METRIC_RECONCILIATION"
    elif exact_membership_hits < 3:
        conclusion = (
            "ORACLE_MEMBERSHIP_INCONSISTENCY: at least one selected "
            "query does not expose an exact S8.10B oracle tile in the "
            "processed pair manifest. Audit tile-ID or manifest joins."
        )
        status = "NEEDS_ORACLE_MEMBERSHIP_FIX"
    elif (
        reported_threshold_hits is not None
        and reported_threshold_hits != recomputed_threshold_hits
    ):
        conclusion = (
            "S7D_THRESHOLD_RECOMPUTATION_MISMATCH: processed centre "
            "errors do not reproduce the query-summary oracle hit field."
        )
        status = "NEEDS_S7D_EVALUATION_AUDIT"
    else:
        conclusion = (
            "Oracle membership is consistent, but the exact source of "
            "the 2/3 count requires inspection of the emitted table."
        )
        status = "NEEDS_MANUAL_ORACLE_REVIEW"

    report = {
        "stage": "S8.12B.0D.1",
        "run_name": RUN_NAME,
        "threshold_m": THRESHOLD_M,
        "oracle_audit_source": (
            str(oracle_path)
            if oracle_path is not None
            else None
        ),
        "query_count": len(result),
        "exact_s8_10b_oracle_membership_hits": (
            exact_membership_hits
        ),
        "recomputed_processed_center_threshold_hits": (
            recomputed_threshold_hits
        ),
        "s7d_reported_oracle_processed_hits": (
            reported_threshold_hits
        ),
        "status": status,
        "conclusion": conclusion,
        "rows": result.to_dict(orient="records"),
    }

    result.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    display_columns = [
        "role",
        "query_id",
        "s8_11e_first_oracle_rank",
        "processed_exact_oracle_retrieval_ranks",
        "processed_exact_oracle_lightglue_ranks",
        "minimum_processed_center_error_m",
        "recomputed_processed_hit_le_40m",
        "s7d_oracle_processed_hit",
        "classification",
    ]

    print()
    print(result[display_columns].to_string(index=False))

    print()
    print("Reconciliation summary")
    print("----------------------")
    print(
        "Exact S8.10B oracle present:",
        f"{exact_membership_hits}/{len(result)}",
    )
    print(
        "Processed candidate <=40m:",
        f"{recomputed_threshold_hits}/{len(result)}",
    )
    print(
        "S7D reported oracle_processed_hits:",
        (
            f"{reported_threshold_hits}/{len(result)}"
            if reported_threshold_hits is not None
            else "COLUMN_NOT_FOUND"
        ),
    )
    print()
    print("Conclusion:", conclusion)

    print()
    print("----------------------------------------")
    print("S8.12B.0D.1 COMPLETE")
    print("STATUS:", status)
    print("CSV:", OUT_CSV)
    print("JSON:", OUT_JSON)


if __name__ == "__main__":
    main()
