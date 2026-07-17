'''

'''


from pathlib import Path

import cv2
import numpy as np
import pandas as pd


INDEX_PATH = Path(
    "outputs/satloc/metadata/uav_frames_index_enriched.csv"
)
SEQUENCE = "traj01"


def find_column(columns, candidates, description):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise RuntimeError(
        f"Could not find {description} column.\n"
        f"Available columns:\n{list(columns)}"
    )


if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Missing canonical SatLoc index: {INDEX_PATH}\n"
        "Run build_satloc_coordinate_index.py first."
    )

df = pd.read_csv(INDEX_PATH)

if "sequence" not in df.columns:
    raise RuntimeError(
        f"'sequence' column missing. Available columns: {list(df.columns)}"
    )

traj = df[df["sequence"].astype(str) == SEQUENCE].copy()

if traj.empty:
    raise RuntimeError(f"No rows found for sequence {SEQUENCE!r}.")

order_col = find_column(
    traj.columns,
    [
        "token1_order",
        "frame_order",
        "order",
        "token0_id",
        "frame_id",
        "query_token",
    ],
    "frame-order",
)

path_col = find_column(
    traj.columns,
    [
        "image_path",
        "uav_image_path",
        "file_path",
        "filepath",
        "path",
    ],
    "image-path",
)

traj["_order_numeric"] = pd.to_numeric(
    traj[order_col], errors="coerce"
)

invalid_order = int(traj["_order_numeric"].isna().sum())
traj = traj.dropna(subset=["_order_numeric"])
traj = traj.sort_values("_order_numeric").reset_index(drop=True)

orders = traj["_order_numeric"].to_numpy(dtype=float)
order_diffs = np.diff(orders)

duplicate_orders = int(np.sum(order_diffs == 0))
negative_steps = int(np.sum(order_diffs < 0))
nonunit_gaps = int(np.sum(order_diffs > 1))


def resolve_image_path(value):
    path = Path(str(value)).expanduser()

    candidates = [
        path,
        Path.cwd() / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return path


sample_indices = sorted(
    set([0, len(traj) // 4, len(traj) // 2,
         3 * len(traj) // 4, len(traj) - 1])
)

sample_results = []
for index in sample_indices:
    row = traj.iloc[index]
    image_path = resolve_image_path(row[path_col])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    sample_results.append(
        {
            "row_index": index,
            "order": row["_order_numeric"],
            "path": str(image_path),
            "exists": image_path.exists(),
            "read_ok": image is not None,
            "width": None if image is None else int(image.shape[1]),
            "height": None if image is None else int(image.shape[0]),
        }
    )

missing_count = 0
for value in traj[path_col]:
    if not resolve_image_path(value).exists():
        missing_count += 1

print("\nS6A.0 SatLoc relative-motion preflight")
print("--------------------------------------")
print(f"Index:                    {INDEX_PATH}")
print(f"Sequence:                 {SEQUENCE}")
print(f"Rows:                     {len(traj)}")
print(f"Order column:             {order_col}")
print(f"Image-path column:        {path_col}")
print(f"Invalid order values:     {invalid_order}")
print(f"Duplicate order values:   {duplicate_orders}")
print(f"Negative order steps:     {negative_steps}")
print(f"Non-unit order gaps:      {nonunit_gaps}")
print(f"Missing image files:      {missing_count}")
print(f"First order:              {orders[0]}")
print(f"Last order:               {orders[-1]}")

if {"x_enu_m", "y_enu_m"}.issubset(traj.columns):
    x = pd.to_numeric(traj["x_enu_m"], errors="coerce").to_numpy()
    y = pd.to_numeric(traj["y_enu_m"], errors="coerce").to_numpy()

    step_m = np.hypot(np.diff(x), np.diff(y))
    finite_step_m = step_m[np.isfinite(step_m)]

    if len(finite_step_m):
        print("\nEvaluation-only reference continuity")
        print("------------------------------------")
        print(f"Total reference path [m]: {finite_step_m.sum():.3f}")
        print(f"Median step [m]:          {np.median(finite_step_m):.4f}")
        print(f"95th percentile [m]:      {np.percentile(finite_step_m, 95):.4f}")
        print(f"Maximum step [m]:         {np.max(finite_step_m):.4f}")
        print(f"Zero-distance steps:      {np.sum(finite_step_m == 0)}")

print("\nSample image checks")
print("-------------------")
for result in sample_results:
    print(result)

print("\nAvailable columns")
print("-----------------")
print(list(traj.columns))

if missing_count == 0 and all(
    result["read_ok"] for result in sample_results
):
    print("\nPRELIMINARY RESULT: PASS")
    print(
        "traj01 is ready for S6A.1 consecutive-frame "
        "visual-motion diagnostics."
    )
else:
    print("\nPRELIMINARY RESULT: CHECK REQUIRED")
    print(
        "Resolve missing/unreadable image paths before "
        "running frame-pair estimation."
    )