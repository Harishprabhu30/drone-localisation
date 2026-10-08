#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageOps


ROOT = Path.cwd().resolve()
DEFAULT_CONFIG = Path(
    "configs/research/query_view_candidate_generation_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/research_runs/query_view_candidate_generation_v1/qv1_0"
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


def crop_boxes(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    side = min(width, height)

    if width < height:
        raise ValueError(
            f"QV1.0 expects landscape/square frames, got {width}x{height}"
        )

    slack = width - side

    return {
        "left_square": (0, 0, side, side),
        "center_square": (
            slack // 2,
            0,
            slack // 2 + side,
            side,
        ),
        "right_square": (
            slack,
            0,
            slack + side,
            side,
        ),
    }


def horizontal_union_fraction(
    boxes: dict[str, tuple[int, int, int, int]],
    width: int,
) -> float:
    intervals = sorted(
        (box[0], box[2])
        for box in boxes.values()
    )

    total = 0
    start, end = intervals[0]

    for x0, x1 in intervals[1:]:
        if x0 <= end:
            end = max(end, x1)
        else:
            total += end - start
            start, end = x0, x1

    total += end - start

    return float(total) / float(width)


def overlap_px(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    return max(
        0,
        min(a[2], b[2])
        - max(a[0], b[0]),
    )


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
    args = parser.parse_args()

    cfg = yaml.safe_load(
        resolve(args.config).read_text()
    )
    output_root = resolve(
        args.output_root
    )
    manifest_path = resolve(
        cfg["control"]["blind_manifest"]
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    manifest = pd.read_csv(
        manifest_path,
        usecols=[
            "token0_id",
            "query_id",
            "image_path",
        ],
    ).copy()

    expected = int(
        cfg["qv1_0"]["expected_query_count"]
    )

    if len(manifest) != expected:
        raise RuntimeError(
            f"Expected {expected} blind queries, got {len(manifest)}"
        )

    rows = []

    for row in manifest.itertuples(index=False):
        image_path = resolve(
            str(row.image_path)
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(
                raw
            )
            width, height = image.size

        boxes = crop_boxes(
            int(width),
            int(height),
        )
        side = min(
            int(width),
            int(height),
        )

        historical_left = (
            int(width) - side
        ) // 2
        historical_center = (
            historical_left,
            0,
            historical_left + side,
            side,
        )

        if (
            boxes["center_square"]
            != historical_center
        ):
            raise RuntimeError(
                f"Historical center-square mismatch q{row.query_id}"
            )

        union_fraction = (
            horizontal_union_fraction(
                boxes,
                int(width),
            )
        )

        if abs(
            union_fraction - 1.0
        ) > 1e-12:
            raise RuntimeError(
                f"L/C/R does not cover full width q{row.query_id}"
            )

        rows.append(
            {
                "token0_id": row.token0_id,
                "query_id": int(
                    row.query_id
                ),
                "image_path": str(
                    row.image_path
                ),
                "width_px": int(width),
                "height_px": int(height),
                "square_side_px": int(side),
                "horizontal_slack_px": int(
                    width - side
                ),
                "center_square_retained_horizontal_fraction": float(
                    side / width
                ),
                "center_square_discarded_horizontal_fraction": float(
                    1.0 - side / width
                ),
                "left_x0": boxes[
                    "left_square"
                ][0],
                "left_x1": boxes[
                    "left_square"
                ][2],
                "center_x0": boxes[
                    "center_square"
                ][0],
                "center_x1": boxes[
                    "center_square"
                ][2],
                "right_x0": boxes[
                    "right_square"
                ][0],
                "right_x1": boxes[
                    "right_square"
                ][2],
                "left_center_overlap_px": overlap_px(
                    boxes[
                        "left_square"
                    ],
                    boxes[
                        "center_square"
                    ],
                ),
                "center_right_overlap_px": overlap_px(
                    boxes[
                        "center_square"
                    ],
                    boxes[
                        "right_square"
                    ],
                ),
                "lcr_horizontal_union_fraction": union_fraction,
            }
        )

    audit = pd.DataFrame(
        rows
    )

    expected_w = int(
        cfg["qv1_0"]["expected_width_px"]
    )
    expected_h = int(
        cfg["qv1_0"]["expected_height_px"]
    )

    widths = sorted(
        audit["width_px"].unique().tolist()
    )
    heights = sorted(
        audit["height_px"].unique().tolist()
    )

    if widths != [expected_w]:
        raise RuntimeError(
            f"Unexpected widths {widths}; expected [{expected_w}]"
        )

    if heights != [expected_h]:
        raise RuntimeError(
            f"Unexpected heights {heights}; expected [{expected_h}]"
        )

    first = audit.iloc[0]

    geometry = {
        "width_px": int(
            first["width_px"]
        ),
        "height_px": int(
            first["height_px"]
        ),
        "side_px": int(
            first["square_side_px"]
        ),
        "horizontal_slack_px": int(
            first["horizontal_slack_px"]
        ),
        "center_square_retained_horizontal_fraction": float(
            first[
                "center_square_retained_horizontal_fraction"
            ]
        ),
        "center_square_discarded_horizontal_fraction": float(
            first[
                "center_square_discarded_horizontal_fraction"
            ]
        ),
        "left_square": [
            int(first["left_x0"]),
            0,
            int(first["left_x1"]),
            int(first["square_side_px"]),
        ],
        "center_square": [
            int(first["center_x0"]),
            0,
            int(first["center_x1"]),
            int(first["square_side_px"]),
        ],
        "right_square": [
            int(first["right_x0"]),
            0,
            int(first["right_x1"]),
            int(first["square_side_px"]),
        ],
        "left_center_overlap_px": int(
            first[
                "left_center_overlap_px"
            ]
        ),
        "center_right_overlap_px": int(
            first[
                "center_right_overlap_px"
            ]
        ),
        "lcr_horizontal_union_fraction": float(
            first[
                "lcr_horizontal_union_fraction"
            ]
        ),
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        output_root
        / "qv1_0_query_view_geometry_audit.csv"
    )
    audit.to_csv(
        audit_path,
        index=False,
    )

    report = {
        "stage": "QV1.0",
        "status": "PASS_QV1_QUERY_VIEW_PREFLIGHT",
        "created_at_utc": now_utc(),
        "query_count": int(
            len(audit)
        ),
        "manifest": {
            "path": str(
                manifest_path
            ),
            "sha256": sha256_file(
                manifest_path
            ),
            "columns_loaded": [
                "token0_id",
                "query_id",
                "image_path",
            ],
            "reference_columns_loaded": False,
        },
        "geometry": geometry,
        "historical_center_square_parity": True,
        "lcr_full_horizontal_coverage": True,
        "scope_guarantees": {
            "descriptor_inference_run": False,
            "coordinates_loaded": False,
            "reference_loaded": False,
            "orb_run": False,
            "bootstrap_run": False,
            "state_run": False,
        },
        "outputs": {
            "geometry_audit_csv": str(
                audit_path
            ),
        },
    }

    report_path = (
        output_root
        / "qv1_0_query_view_preflight_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print("=" * 100)
    print(
        "QV1.0 — QUERY-VIEW GEOMETRY / PROVENANCE PREFLIGHT"
    )
    print("=" * 100)
    print(
        "queries:",
        len(audit),
    )
    print(
        "dimensions:",
        f"{geometry['width_px']}x{geometry['height_px']}",
    )
    print(
        "center retained horizontal fraction:",
        f"{geometry['center_square_retained_horizontal_fraction']:.4f}",
    )
    print(
        "center discarded horizontal fraction:",
        f"{geometry['center_square_discarded_horizontal_fraction']:.4f}",
    )
    print(
        "left box:",
        geometry["left_square"],
    )
    print(
        "center box:",
        geometry["center_square"],
    )
    print(
        "right box:",
        geometry["right_square"],
    )
    print(
        "L/C/R union fraction:",
        f"{geometry['lcr_horizontal_union_fraction']:.4f}",
    )
    print()
    print(
        "STATUS: PASS_QV1_QUERY_VIEW_PREFLIGHT"
    )
    print(
        "report:",
        report_path,
    )


if __name__ == "__main__":
    main()
