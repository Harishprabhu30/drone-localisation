#!/usr/bin/env python3
"""RG2.1-A: freeze local tile sets from frozen QV anchors and RG2.0 topology.

This stage consumes no GT, SRT, reference attachment, image, descriptor, or ORB.
RG2.1-B may attach development reference *after* these tables are hashed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/local_map_neighborhood_rg2_1a.yaml"
GEOMETRY_COLUMNS = {
    "tile_id", "center_easting", "center_northing", "left_easting",
    "right_easting", "bottom_northing", "top_northing",
    "overlap_degree", "near_degree", "footprint_area_m2",
}
EDGE_COLUMNS = {
    "tile_id", "neighbor_tile_id", "overlaps", "nearby",
    "center_distance_m", "delta_easting_m", "delta_northing_m",
    "shared_area_m2", "overlap_iou",
}
POOL_COLUMNS = {
    "query_id", "pool_rank", "tile_id",
    "source_view", "source_rank", "selection_reason",
}
MANIFEST_COLUMNS = {"query_id", "reference_available"}
FORBIDDEN_TOKENS = (
    "ground_truth", "oracle", "gt_", "_gt", "reference_",
    "ref_lat", "ref_lon", "ref_east", "ref_north", "srt_",
    "gps_", "eval_ref", "error_m", "chosen_error",
    "projected_error", "tile_center_error",
)
ALLOWED_REFERENCE_FLAG = "reference_available"
ALLOWED_SOURCES = {
    "center_square", "left_square", "right_square", "resize_square",
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def guarded_csv(path: Path, required: set[str], *, blind: bool) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"Duplicate CSV fieldnames: {path}")
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")
        if blind:
            leaked = sorted(
                field for field in fields
                if field.strip().lower() != ALLOWED_REFERENCE_FLAG
                and any(t in field.strip().lower() for t in FORBIDDEN_TOKENS)
            )
            if leaked:
                raise ValueError(f"Reference/evaluation columns forbidden: {leaked}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    if any(None in row for row in rows):
        raise ValueError(f"Malformed CSV row in {path}")
    return rows


def parse_positive_int(value: str, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def validate_blind_inputs(
    manifest: list[dict], pool: list[dict], *,
    expected_queries: int, budget: int,
) -> tuple[list[int], dict[int, list[dict]]]:
    qids = [parse_positive_int(row["query_id"], "manifest query_id")
            for row in manifest]
    if len(qids) != expected_queries or len(set(qids)) != len(qids):
        raise ValueError("Manifest query count/uniqueness mismatch")
    if any(row["reference_available"].strip().lower()
           not in {"false", "0", "no", "n", "f"} for row in manifest):
        raise ValueError("RG2.1-A requires reference_available=false")
    qset = set(qids)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in pool:
        qid = parse_positive_int(row["query_id"], "pool query_id")
        if qid not in qset:
            raise ValueError(f"Unexpected pool query_id {qid}")
        rank = parse_positive_int(row["pool_rank"], "pool_rank")
        source_rank = parse_positive_int(row["source_rank"], "source_rank")
        if row["source_view"] not in ALLOWED_SOURCES:
            raise ValueError(f"Unexpected source view {row['source_view']}")
        tile_id = str(row["tile_id"]).strip()
        if not tile_id:
            raise ValueError("Empty tile_id in frozen pool")
        grouped[qid].append({
            "query_id": qid,
            "pool_rank": rank,
            "tile_id": tile_id,
            "source_view": row["source_view"],
            "source_rank": source_rank,
            "selection_reason": row["selection_reason"],
        })
    if set(grouped) != qset:
        raise ValueError("Manifest/pool query sets differ")
    for qid in qids:
        rows = sorted(grouped[qid], key=lambda row: row["pool_rank"])
        if len(rows) != budget:
            raise ValueError(f"Frozen Top{budget} budget mismatch at query {qid}")
        if [r["pool_rank"] for r in rows] != list(range(1, budget + 1)):
            raise ValueError(f"Noncontiguous or duplicated ranks at query {qid}")
        if len({r["tile_id"] for r in rows}) != budget:
            raise ValueError(f"Repeated tile ID within Top{budget} at query {qid}")
        if (rows[0]["source_view"] != "center_square"
                or rows[0]["source_rank"] != 1):
            raise ValueError(f"Frozen center-square Top1 violated at query {qid}")
        grouped[qid] = rows
    return sorted(qids), grouped


def bool_field(value: str, field: str) -> bool:
    if value not in ("true", "false"):
        raise ValueError(f"Noncanonical RG2.0 boolean {field}={value!r}")
    return value == "true"


def read_frozen_topology(tile_rows: list[dict], edges: list[dict]) -> tuple[
    dict[str, dict], dict[str, dict[str, dict]]
]:
    tiles = {}
    for row in tile_rows:
        tile_id = row["tile_id"]
        if tile_id in tiles:
            raise ValueError(f"Duplicate geometry tile ID {tile_id}")
        tiles[tile_id] = row
    adj: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in edges:
        src, dst = row["tile_id"], row["neighbor_tile_id"]
        if src not in tiles or dst not in tiles or src == dst:
            raise ValueError(f"Unknown/self topology edge {src}->{dst}")
        if dst in adj[src]:
            raise ValueError(f"Duplicate topology edge {src}->{dst}")
        edge = dict(row)
        edge["overlaps"] = bool_field(row["overlaps"], "overlaps")
        edge["nearby"] = bool_field(row["nearby"], "nearby")
        if not (edge["overlaps"] or edge["nearby"]):
            raise ValueError(f"Topology edge has no membership {src}->{dst}")
        for key in ("center_distance_m", "delta_easting_m",
                    "delta_northing_m", "shared_area_m2", "overlap_iou"):
            edge[key] = float(row[key])
            if not math.isfinite(edge[key]):
                raise ValueError(f"Nonfinite {key} on edge {src}->{dst}")
        if edge["overlaps"] and edge["shared_area_m2"] <= 0:
            raise ValueError("Positive overlap requires positive shared area")
        if edge["nearby"] and edge["center_distance_m"] <= 0:
            raise ValueError("Unexpected coincident near-neighbor tile centers")
        adj[src][dst] = edge
    for src, children in adj.items():
        for dst, edge in children.items():
            reverse = adj.get(dst, {}).get(src)
            if reverse is None:
                raise ValueError(f"Missing reciprocal edge {src}->{dst}")
            if edge["overlaps"] != reverse["overlaps"] or edge["nearby"] != reverse["nearby"]:
                raise ValueError(f"Non-reciprocal relationship {src}->{dst}")
            if abs(edge["shared_area_m2"] - reverse["shared_area_m2"]) > 0.00001:
                raise ValueError(f"Non-reciprocal shared area {src}->{dst}")
    for tile_id, tile in tiles.items():
        n_over = sum(e["overlaps"] for e in adj.get(tile_id, {}).values())
        n_near = sum(e["nearby"] for e in adj.get(tile_id, {}).values())
        if n_over != int(tile["overlap_degree"]) or n_near != int(tile["near_degree"]):
            raise ValueError(f"Degree mismatch for {tile_id}")
    return tiles, adj


def generate(
    *, trajectory_id: str, qids: list[int],
    grouped: dict[int, list[dict]],
    tiles: dict[str, dict],
    adjacent: dict[str, dict[str, dict]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Three fixed sets: anchor, anchor+near, anchor+positive-area overlap."""
    neighborhoods: list[dict] = []
    summaries: list[dict] = []
    snapshots: list[dict] = []
    for qid in qids:
        qv = grouped[qid]
        anchor = qv[0]["tile_id"]
        if any(r["tile_id"] not in tiles for r in qv):
            raise ValueError(f"QV pool references unindexed map tile at q{qid}")
        qv_ranks = {r["tile_id"]: r["pool_rank"] for r in qv}
        for r in qv:
            snapshots.append({
                "trajectory_id": trajectory_id,
                "query_id": qid,
                "pool_rank": r["pool_rank"],
                "tile_id": r["tile_id"],
                "source_view": r["source_view"],
                "source_rank": r["source_rank"],
            })
        links = adjacent.get(anchor, {})
        near_ids = {anchor} | {
            tile_id for tile_id, edge in links.items() if edge["nearby"]
        }
        overlap_ids = {anchor} | {
            tile_id for tile_id, edge in links.items() if edge["overlaps"]
        }
        union_ids = near_ids | overlap_ids

        def key(tile_id: str) -> tuple:
            return (0 if tile_id == anchor else 1,
                    0.0 if tile_id == anchor
                    else links[tile_id]["center_distance_m"], tile_id)

        near_sorted = sorted(near_ids, key=key)
        overlap_sorted = sorted(overlap_ids, key=key)
        union_sorted = sorted(union_ids, key=key)
        near_rank = {tile_id: i for i, tile_id in enumerate(near_sorted, 1)}
        overlap_rank = {tile_id: i for i, tile_id in enumerate(overlap_sorted, 1)}
        anchor_area = float(tiles[anchor]["footprint_area_m2"])
        for i, tile_id in enumerate(union_sorted, 1):
            is_anchor = tile_id == anchor
            e = links.get(tile_id) if not is_anchor else None
            neighborhoods.append({
                "trajectory_id": trajectory_id,
                "query_id": qid,
                "anchor_tile_id": anchor,
                "tile_id": tile_id,
                "union_rank": i,
                "immediate_rank": near_rank.get(tile_id, ""),
                "overlap_rank": overlap_rank.get(tile_id, ""),
                "is_anchor": is_anchor,
                "in_immediate": tile_id in near_ids,
                "in_full_overlap": tile_id in overlap_ids,
                "in_frozen_qv_top20": tile_id in qv_ranks,
                "frozen_qv_rank": qv_ranks.get(tile_id, ""),
                "center_distance_m": 0.0 if is_anchor else e["center_distance_m"],
                "delta_easting_m": 0.0 if is_anchor else e["delta_easting_m"],
                "delta_northing_m": 0.0 if is_anchor else e["delta_northing_m"],
                "shared_area_m2": anchor_area if is_anchor else e["shared_area_m2"],
                "overlap_iou": 1.0 if is_anchor else e["overlap_iou"],
            })
        if (len(near_ids) != 1 + int(tiles[anchor]["near_degree"])
                or len(overlap_ids) != 1 + int(tiles[anchor]["overlap_degree"])):
            raise ValueError(f"Neighborhood degree mismatch at q{qid}")
        summaries.append({
            "trajectory_id": trajectory_id,
            "query_id": qid,
            "anchor_tile_id": anchor,
            "anchor_count": 1,
            "immediate_count": len(near_ids),
            "full_overlap_count": len(overlap_ids),
            "union_count": len(union_ids),
            "top20_count": len(qv),
            "immediate_intersect_top20": len(near_ids & set(qv_ranks)),
            "overlap_intersect_top20": len(overlap_ids & set(qv_ranks)),
            "top20_outside_full_overlap": len(set(qv_ranks) - overlap_ids),
        })
    return neighborhoods, summaries, snapshots


def canonical_csv(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("Cannot freeze empty table")
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            k: ("true" if v else "false") if isinstance(v, bool)
            else f"{v:.6f}" if isinstance(v, float) else v
            for k, v in row.items()
        })
    return f.getvalue().encode("utf-8")


def canonical_json(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def freeze_immutable(files: dict[Path, bytes]) -> None:
    """Never overwrite an existing freeze with different bytes, including reports."""
    changed = [str(path) for path, data in files.items()
               if path.exists() and path.read_bytes() != data]
    if changed:
        raise RuntimeError("Frozen RG2.1-A files differ; refusing overwrite: "
                           + ", ".join(changed))
    for path, data in files.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use exclusive creation, so no pre-existing result can be replaced.
        with path.open("xb") as handle:
            handle.write(data)


def verified_sources(cfg_path: Path, cfg: dict) -> tuple[
    dict[str, dict], dict[str, dict[str, dict]], dict[str, str]
]:
    if cfg.get("stage") != "RG2.1-A" or cfg.get("map_variant") != "768_s256":
        raise ValueError("Wrong RG2.1-A stage/map")
    if cfg.get("map_crs") != "EPSG:3346":
        raise ValueError("Wrong RG2.1-A map CRS")
    for flag in ("reference_allowed", "reference_attachment_enabled",
                 "orb_enabled", "bootstrap_enabled", "state_enabled",
                 "parameter_tuning_enabled"):
        if cfg.get(flag) is not False:
            raise ValueError(f"{flag} must be false")
    if cfg.get("frozen_qv_policy") != "center_unique_allview_fill20":
        raise ValueError("QV policy mismatch")
    if cfg.get("expected_candidate_budget") != 20:
        raise ValueError("QV budget mismatch")
    if cfg.get("anchor") != {"pool_rank": 1, "source_view": "center_square",
                             "source_rank": 1}:
        raise ValueError("Frozen anchor contract mismatch")
    if cfg.get("neighborhoods") != {
        "immediate": "rg2_0_nearby", "full_overlap": "rg2_0_overlaps",
        "include_anchor": True, "expansion_budget": "none",
        "qv_top20_is_comparator_only": True,
    }:
        raise ValueError("Neighborhood rule changed without versioning")

    report_path = resolve(cfg["rg2_0_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("status") != "PASS_RG2_0_MAP_TOPOLOGY_PREFLIGHT"
            or report.get("map_variant") != "768_s256"
            or report.get("map_crs") != "EPSG:3346"
            or report.get("topology", {}).get("tile_count") != cfg["expected_map_tiles"]):
        raise ValueError("RG2.0 report contract mismatch")
    rg2_0_cfg = resolve(cfg["rg2_0_config"])
    if digest(rg2_0_cfg) != report["provenance"]["config_sha256"]:
        raise ValueError("RG2.0 config changed since topology pass")

    root = resolve(cfg["rg2_0_output_root"])
    tile_path = root / "rg2_0_tile_geometry.csv"
    edge_path = root / "rg2_0_tile_relationships.csv"
    if digest(tile_path) != report["provenance"]["tile_geometry_sha256"]:
        raise ValueError("RG2.0 tile geometry changed since preflight")
    edge_sha = digest(edge_path)
    if (edge_sha != report["provenance"]["relationships_sha256"]
            or edge_sha != cfg["expected_rg2_0_relationships_sha256"]):
        raise ValueError("RG2.0 topology SHA256 does not match frozen result")
    rg2_0_spec = resolve(yaml.safe_load(rg2_0_cfg.read_text())["trajectory_spec"])
    if digest(rg2_0_spec) != report["provenance"]["trajectory_spec_sha256"]:
        raise ValueError("RG2.0 trajectory spec changed")
    tile_index = resolve(
        yaml.safe_load(rg2_0_spec.read_text())["map"]["variants"]["768_s256"]["tile_index"]
    )
    if digest(tile_index) != report["provenance"]["tile_index_sha256"]:
        raise ValueError("RG2.0 map index changed")

    tile_rows = guarded_csv(tile_path, GEOMETRY_COLUMNS, blind=False)
    edges = guarded_csv(edge_path, EDGE_COLUMNS, blind=False)
    tiles, adj = read_frozen_topology(tile_rows, edges)
    if len(tiles) != cfg["expected_map_tiles"]:
        raise ValueError("RG2.0 tile count mismatch")
    provenance = {
        "rg2_1a_config_sha256": digest(cfg_path),
        "rg2_1a_script_sha256": digest(Path(__file__).resolve()),
        "rg2_0_report_sha256": digest(report_path),
        "rg2_0_relationships_sha256": edge_sha,
        "rg2_0_tile_geometry_sha256": digest(tile_path),
        "frozen_map_index_sha256": digest(tile_index),
    }
    return tiles, adj, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="RG2.1-A reference-free neighborhood freeze")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tiles, adj, provenance = verified_sources(cfg_path, cfg)
    out_root = resolve(cfg["output_root"])

    all_files: dict[Path, bytes] = {}
    trajectory_reports: dict[str, dict] = {}
    for key in ("development", "blind_stress"):
        tc = cfg["trajectories"][key]
        spec_path = resolve(tc["spec"])
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        role = spec["trajectory"]["role"]
        trajectory_id = spec["trajectory"]["id"]
        if role != key or role != tc["role"]:
            raise ValueError(f"Incorrect trajectory role: {key}")
        if (spec["map"]["crs"] != "EPSG:3346"
                or "768_s256" not in spec["map"]["variants"]):
            raise ValueError(f"Map variant/CRS inconsistent: {key}")
        spec_tile_index = resolve(spec["map"]["variants"]["768_s256"]["tile_index"])
        if digest(spec_tile_index) != provenance["frozen_map_index_sha256"]:
            raise ValueError(f"Map index differs between trajectories: {key}")

        manifest_path = resolve(tc["manifest"])
        pool_path = resolve(tc["frozen_pool"])
        manifest = guarded_csv(manifest_path, MANIFEST_COLUMNS, blind=True)
        pool = guarded_csv(pool_path, POOL_COLUMNS, blind=True)
        qids, groups = validate_blind_inputs(
            manifest, pool, expected_queries=tc["expected_queries"],
            budget=cfg["expected_candidate_budget"],
        )
        local, summaries, top20 = generate(
            trajectory_id=trajectory_id, qids=qids, grouped=groups,
            tiles=tiles, adjacent=adj,
        )
        area = out_root / key
        outputs = {
            "local_neighborhoods": (area / "rg2_1a_local_neighborhoods.csv",
                                    canonical_csv(local)),
            "query_summaries": (area / "rg2_1a_query_summaries.csv",
                                canonical_csv(summaries)),
            "frozen_qv_top20": (area / "rg2_1a_frozen_qv_top20.csv",
                                canonical_csv(top20)),
        }
        for _, (path, data) in outputs.items():
            all_files[path] = data
        trajectory_reports[key] = {
            "trajectory_id": trajectory_id,
            "role": role,
            "queries": len(qids),
            "frozen_qv_top20_rows": len(top20),
            "local_neighborhood_rows": len(local),
            "near_count_histogram": dict(sorted(Counter(
                str(row["immediate_count"]) for row in summaries
            ).items(), key=lambda kv: int(kv[0]))),
            "overlap_count_histogram": dict(sorted(Counter(
                str(row["full_overlap_count"]) for row in summaries
            ).items(), key=lambda kv: int(kv[0]))),
            "mean_near_windows_per_query": round(
                sum(r["immediate_count"] for r in summaries) / len(summaries), 6
            ),
            "mean_overlap_windows_per_query": round(
                sum(r["full_overlap_count"] for r in summaries) / len(summaries), 6
            ),
            "provenance": {
                "trajectory_spec_sha256": digest(spec_path),
                "blind_manifest_sha256": digest(manifest_path),
                "frozen_candidate_pool_sha256": digest(pool_path),
            },
            "outputs": {
                label: {"path": path.relative_to(out_root).as_posix(),
                        "sha256": sha_bytes(data), "rows": len(
                            local if label == "local_neighborhoods"
                            else summaries if label == "query_summaries" else top20)}
                for label, (path, data) in outputs.items()
            },
        }

    freeze_report = {
        "stage": "RG2.1-A",
        "status": "PASS_RG2_1A_REFERENCE_FREE_NEIGHBORHOOD_FREEZE",
        "map_variant": "768_s256",
        "map_crs": "EPSG:3346",
        "map_tiles": len(tiles),
        "provenance": provenance,
        "trajectories": trajectory_reports,
        "boundaries": {
            "reference_loaded": False,
            "srt_gps_loaded": False,
            "image_or_descriptor_loaded": False,
            "orb_executed": False,
            "candidate_generator_executed": False,
            "top20_reranked": False,
            "bootstrap_state_modified": False,
            "oracle_evaluation_executed": False,
        },
        "note": "Top20 is only the frozen comparator; local sets derive solely from anchor and RG2.0 geometry.",
    }
    report_path = out_root / "rg2_1a_freeze_report.json"
    all_files[report_path] = canonical_json(freeze_report)
    freeze_immutable(all_files)
    print(f"Status: {freeze_report['status']}")
    print(f"RG2.0 topology SHA256: {provenance['rg2_0_relationships_sha256']}")
    for key, tr in trajectory_reports.items():
        print(f"{key}: {tr['queries']} queries, {tr['local_neighborhood_rows']} local rows, "
              f"{tr['frozen_qv_top20_rows']} frozen Top20 rows")
        print(f"  near hist: {tr['near_count_histogram']}")
        print(f"  overlap hist: {tr['overlap_count_histogram']}")
        for label, entry in tr["outputs"].items():
            print(f"  {label} SHA256: {entry['sha256']}")
    print("Reference loaded: False")
    print("Reference/oracle evaluation executed: False")
    print(f"Freeze report: {report_path}")


if __name__ == "__main__":
    main()
