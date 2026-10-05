"""Patch only the isolated 30-query ORB control in outputs/benchmark_probes."""

from pathlib import Path
import shutil

path = Path("outputs/benchmark_probes/orb_30_query_feature_control.py")
assert path.is_file(), f"Probe script missing: {path}"
source = path.read_text()

old_export = '''    queries = rows[["query_id", "query_image_resolved"]].drop_duplicates("query_id")
    assert len(queries) == 30, f"Expected 30 Mac queries, got {len(queries)}"
'''
new_export = '''    rows["query_id"] = pd.to_numeric(rows["query_id"], errors="raise").astype(int)
    queries = rows.loc[rows["query_id"].between(1, 30),
                       ["query_id", "query_image_resolved"]].drop_duplicates("query_id")
    queries = queries.sort_values("query_id")
    assert queries.query_id.tolist() == list(range(1, 31)), (
        f"Expected Mac query IDs 1..30, got {queries.query_id.tolist()}"
    )
'''
old_compare = '''    mac = pd.read_csv(mac_csv)
    jetson = pd.read_csv(jetson_csv)
    keys = ["query_id", "tile_id"]
'''
new_compare = '''    mac = pd.read_csv(mac_csv)
    jetson = pd.read_csv(jetson_csv)
    mac["query_id"] = pd.to_numeric(mac["query_id"], errors="raise").astype(int)
    jetson["query_id"] = pd.to_numeric(jetson["query_id"], errors="raise").astype(int)
    mac = mac.loc[mac.query_id.between(1, 30)].copy()
    assert len(mac) == len(jetson) == 600, (
        f"Expected 600 candidate rows for queries 1..30; Mac={len(mac)} Jetson={len(jetson)}"
    )
    keys = ["query_id", "tile_id"]
'''

for old, new, label in ((old_export, new_export, "export"),
                        (old_compare, new_compare, "comparison")):
    if new in source:
        print(label, "already fixed")
    elif source.count(old) == 1:
        source = source.replace(old, new)
        print(label, "fixed")
    else:
        raise SystemExit(f"Could not identify {label} block; probe file left untouched")

backup = path.with_suffix(path.suffix + ".before_q30_fix")
if not backup.exists():
    shutil.copy2(path, backup)
path.write_text(source)
assert old_export not in path.read_text() and new_export in path.read_text()
print("patched", path.resolve())
print("backup", backup.resolve())
