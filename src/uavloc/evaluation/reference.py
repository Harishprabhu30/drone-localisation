"""Shared post-freeze reference-loading utilities.

Keep reusable evaluation/reference logic in the importable uavloc package.
Executable files under scripts/ should import this module rather than importing
other scripts.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from pyproj import Transformer


REQUIRED_REFERENCE_COLUMNS = {
    "query_id",
    "eval_ref_lon",
    "eval_ref_lat",
}


def load_reference_xy(
    path: str | Path,
    *,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:3346",
) -> pd.DataFrame:
    """Load a prepared reference attachment and add projected gt_x/gt_y.

    This preserves the VILLOC/QV evaluation convention used historically:
    eval_ref_lon/lat are transformed from EPSG:4326 into EPSG:3346.
    The returned frame is indexed uniquely by integer query_id.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path).copy()
    missing = sorted(REQUIRED_REFERENCE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Reference attachment missing columns: {missing}")

    frame["query_id"] = pd.to_numeric(
        frame["query_id"], errors="raise"
    ).astype(int)
    if frame["query_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["query_id"].duplicated(False), "query_id"]
            .astype(int)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Reference attachment has duplicate query IDs: {duplicates[:20]}"
        )

    lon = pd.to_numeric(frame["eval_ref_lon"], errors="raise").to_numpy(float)
    lat = pd.to_numeric(frame["eval_ref_lat"], errors="raise").to_numpy(float)
    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )
    x, y = transformer.transform(lon, lat)
    frame["gt_x"] = x
    frame["gt_y"] = y

    finite = (
        frame["gt_x"].map(lambda v: math.isfinite(float(v))).all()
        and frame["gt_y"].map(lambda v: math.isfinite(float(v))).all()
    )
    if not finite:
        raise ValueError("Reference transformation produced non-finite coordinates")

    return frame.set_index("query_id", drop=True).sort_index()
