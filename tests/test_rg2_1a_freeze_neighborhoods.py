"""RG2.1-A reference-free neighborhood freeze tests (synthetic map and QV)."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.villoc.geometry.rg2_1a_freeze_neighborhoods import (
    canonical_csv, freeze_immutable, generate, guarded_csv,
    read_frozen_topology, validate_blind_inputs,
)


def fixtures(distance: float = 51.2):
    area = 153.6 ** 2
    common = {
        "overlap_degree": "1", "near_degree": "1",
        "footprint_area_m2": str(area),
    }
    tiles = [
        {"tile_id": "A", **common},
        {"tile_id": "B", **common},
        {"tile_id": "C", "overlap_degree": "0", "near_degree": "0",
         "footprint_area_m2": str(area)},
    ]
    width = 153.6 - distance
    share = width * 153.6
    iou = share / (2 * area - share)
    edges = [
        {"tile_id": s, "neighbor_tile_id": d,
         "overlaps": "true", "nearby": "true",
         "center_distance_m": str(distance),
         "delta_easting_m": str(direction * distance),
         "delta_northing_m": "0",
         "shared_area_m2": str(share), "overlap_iou": str(iou)}
        for s, d, direction in (("A", "B", 1), ("B", "A", -1))
    ]
    return read_frozen_topology(tiles, edges)


def mock_manifest(qids=(1, 2)):
    return [{"query_id": str(qid), "reference_available": "false"} for qid in qids]


def mock_pool(qids=(1, 2)):
    return [
        {"query_id": str(qid), "pool_rank": str(rank), "tile_id": tile_id,
         "source_view": "center_square" if rank == 1 else "right_square",
         "source_rank": "1",
         "selection_reason": "synthetic"}
        for qid in qids
        for rank, tile_id in ((1, "A"), (2, "C"))
    ]


class RG21ATests(unittest.TestCase):
    def test_expansion_stays_local_and_top20_is_comparator_only(self):
        tiles, adjacent = fixtures()
        qids, groups = validate_blind_inputs(
            mock_manifest(), mock_pool(), expected_queries=2, budget=2)
        local, summary, snapshot = generate(
            trajectory_id="synthetic", qids=qids,
            grouped=groups, tiles=tiles, adjacent=adjacent)
        self.assertEqual(len(local), 4)
        self.assertEqual(len(snapshot), 4)
        self.assertEqual(summary[0]["immediate_count"], 2)
        self.assertEqual(summary[0]["full_overlap_count"], 2)
        self.assertEqual(summary[0]["top20_outside_full_overlap"], 1)
        self.assertEqual(
            [r["tile_id"] for r in local if r["query_id"] == 1], ["A", "B"]
        )
        b = next(r for r in local if r["tile_id"] == "B")
        self.assertTrue(b["in_full_overlap"])
        self.assertTrue(b["in_immediate"])
        self.assertFalse(b["in_frozen_qv_top20"])
        self.assertEqual(b["overlap_rank"], 2)
        self.assertEqual(b["frozen_qv_rank"], "")

    def test_compressed_aoi_near_edge_is_not_deduplicated(self):
        tiles, adjacent = fixtures(distance=3.6)
        qids, groups = validate_blind_inputs(
            mock_manifest(qids=(1,)), mock_pool(qids=(1,)),
            expected_queries=1, budget=2)
        local, _, _ = generate(
            trajectory_id="synthetic", qids=qids,
            grouped=groups, tiles=tiles, adjacent=adjacent)
        self.assertEqual(len(local), 2)
        self.assertAlmostEqual(local[1]["center_distance_m"], 3.6)
        self.assertGreater(local[1]["overlap_iou"], 0.95)

    def test_pool_order_cannot_change_hash(self):
        tiles, adjacent = fixtures()
        qids, groups = validate_blind_inputs(
            mock_manifest(qids=(2, 1)), list(reversed(mock_pool())),
            expected_queries=2, budget=2)
        first = generate(trajectory_id="synthetic", qids=qids,
                         grouped=groups, tiles=tiles, adjacent=adjacent)
        qids2, groups2 = validate_blind_inputs(
            mock_manifest(), mock_pool(), expected_queries=2, budget=2)
        second = generate(trajectory_id="synthetic", qids=qids2,
                          grouped=groups2, tiles=tiles, adjacent=adjacent)
        self.assertEqual([canonical_csv(x) for x in first],
                         [canonical_csv(x) for x in second])

    def test_reject_reference_flag_true(self):
        manifest = mock_manifest(qids=(1,))
        manifest[0]["reference_available"] = "true"
        with self.assertRaisesRegex(ValueError, "reference_available"):
            validate_blind_inputs(manifest, mock_pool(qids=(1,)),
                                  expected_queries=1, budget=2)

    def test_reject_duplicate_pool_tile_or_ranks(self):
        manifest = mock_manifest(qids=(1,))
        pool = mock_pool(qids=(1,))
        pool[1]["tile_id"] = pool[0]["tile_id"]
        with self.assertRaisesRegex(ValueError, "Repeated tile"):
            validate_blind_inputs(manifest, pool, expected_queries=1, budget=2)
        pool = mock_pool(qids=(1,))
        pool[1]["pool_rank"] = "1"
        with self.assertRaisesRegex(ValueError, "ranks"):
            validate_blind_inputs(manifest, pool, expected_queries=1, budget=2)

    def test_reference_column_schema_rejected_before_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "query_id", "pool_rank", "tile_id", "source_view",
                    "source_rank", "selection_reason", "gt_easting"])
                writer.writeheader()
                writer.writerow({
                    "query_id": "1", "pool_rank": "1", "tile_id": "A",
                    "source_view": "center_square", "source_rank": "1",
                    "selection_reason": "a", "gt_easting": "123"})
            with self.assertRaisesRegex(ValueError, "Reference/evaluation"):
                guarded_csv(path, {
                    "query_id", "pool_rank", "tile_id", "source_view",
                    "source_rank", "selection_reason"}, blind=True)

    def test_sampling_alignment_error_ms_allowed(self):
        # Timestamp resampling metadata is not post-freeze position error.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "query_id", "reference_available",
                    "sampling_alignment_error_ms",
                ])
                writer.writeheader()
                writer.writerow({
                    "query_id": "1", "reference_available": "false",
                    "sampling_alignment_error_ms": "8.0",
                })
            rows = guarded_csv(path, {"query_id", "reference_available"}, blind=True)
            self.assertEqual(rows[0]["sampling_alignment_error_ms"], "8.0")

    def test_exact_error_m_still_forbidden(self):
        for prohibited in ("error_m", "ground_truth_error",
                           "reference_x_m", "oracle_tile_identity", "gps_lat"):
            with self.subTest(field=prohibited), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "manifest.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        "query_id", "reference_available", prohibited,
                    ])
                    writer.writeheader()
                    writer.writerow({
                        "query_id": "1", "reference_available": "false",
                        prohibited: "0.0",
                    })
                with self.assertRaisesRegex(ValueError, "Reference/evaluation"):
                    guarded_csv(path, {"query_id", "reference_available"}, blind=True)

    def test_freeze_never_overwrites_different_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "freeze.csv"
            freeze_immutable({path: b"frozen\n"})
            freeze_immutable({path: b"frozen\n"})
            with self.assertRaisesRegex(RuntimeError, "refusing overwrite"):
                freeze_immutable({path: b"changed\n"})
            self.assertEqual(path.read_bytes(), b"frozen\n")

    def test_topology_degree_mismatch_rejected(self):
        tiles, _ = fixtures()
        rows = [
            {"tile_id": key, **{"overlap_degree": "0", "near_degree": "0",
                                 "footprint_area_m2": "23592.96"}}
            for key in tiles
        ]
        edges = [{
            "tile_id": "A", "neighbor_tile_id": "B",
            "overlaps": "true", "nearby": "true",
            "center_distance_m": "51.2", "delta_easting_m": "51.2",
            "delta_northing_m": "0", "shared_area_m2": "100",
            "overlap_iou": "0.1",
        }, {
            "tile_id": "B", "neighbor_tile_id": "A",
            "overlaps": "true", "nearby": "true",
            "center_distance_m": "51.2", "delta_easting_m": "-51.2",
            "delta_northing_m": "0", "shared_area_m2": "100",
            "overlap_iou": "0.1",
        }]
        with self.assertRaisesRegex(ValueError, "Degree mismatch"):
            read_frozen_topology(rows, edges)


if __name__ == "__main__":
    unittest.main()
