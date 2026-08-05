#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from exc


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_info(path: Path) -> Tuple[bool, Dict[str, Any], str | None]:
    try:
        df = pd.read_csv(path)
        info = {
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
        }
        return True, info, None
    except Exception as exc:
        return False, {}, str(exc)


def read_json_info(path: Path) -> Tuple[bool, Dict[str, Any], str | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            keys = list(obj.keys())
        else:
            keys = []
        info = {
            "json_type": type(obj).__name__,
            "top_level_keys": keys[:50],
        }
        return True, info, None
    except Exception as exc:
        return False, {}, str(exc)


def validate_entry(name: str, entry: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    rel_path = entry.get("path")
    required = bool(entry.get("required", False))
    kind = str(entry.get("kind", "file"))

    result: Dict[str, Any] = {
        "name": name,
        "path": rel_path,
        "required": required,
        "kind": kind,
        "exists": False,
        "ok": False,
        "warnings": [],
        "errors": [],
        "info": {},
        "description": entry.get("description", ""),
    }

    if not rel_path:
        msg = "No path specified."
        if required:
            result["errors"].append(msg)
        else:
            result["warnings"].append(msg)
        return result

    path = repo_root / rel_path
    result["abs_path"] = str(path)

    if not path.exists():
        msg = "File not found."
        if required:
            result["errors"].append(msg)
        else:
            result["warnings"].append(msg)
        return result

    result["exists"] = True

    if kind == "csv":
        ok, info, err = read_csv_info(path)
        if not ok:
            result["errors"].append(f"CSV read failed: {err}")
            return result

        result["info"].update(info)

        min_rows = entry.get("min_rows")
        if min_rows is not None and info["rows"] < int(min_rows):
            result["errors"].append(
                f"CSV has {info['rows']} rows, expected at least {min_rows}."
            )

        identity_any = entry.get("identity_columns_any") or []
        if identity_any:
            cols = set(info["columns"])
            found = [c for c in identity_any if c in cols]
            result["info"]["identity_columns_found"] = found
            if not found:
                result["warnings"].append(
                    f"No identity column found from candidates: {identity_any}"
                )

    elif kind == "json":
        ok, info, err = read_json_info(path)
        if not ok:
            result["errors"].append(f"JSON read failed: {err}")
            return result
        result["info"].update(info)

    else:
        try:
            result["info"]["size_bytes"] = int(path.stat().st_size)
        except Exception:
            pass

    result["ok"] = len(result["errors"]) == 0
    return result


def collect_input_entries(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    entries: List[Tuple[str, Dict[str, Any]]] = []

    for section_name in ["inputs"]:
        section = cfg.get(section_name, {})
        if isinstance(section, dict):
            for name, entry in section.items():
                if isinstance(entry, dict) and "path" in entry:
                    entries.append((name, entry))

    existing_figures = cfg.get("existing_figures", {})
    if isinstance(existing_figures, dict):
        for name, entry in existing_figures.items():
            if isinstance(entry, dict) and "path" in entry:
                item = dict(entry)
                item.setdefault("kind", "file")
                item.setdefault("required", False)
                item.setdefault("description", "Existing figure candidate.")
                entries.append((f"existing_figure.{name}", item))

    return entries


def create_output_dirs(cfg: Dict[str, Any], repo_root: Path) -> None:
    outputs = cfg.get("outputs", {})
    for key, rel in outputs.items():
        if not isinstance(rel, str):
            continue
        path = repo_root / rel
        if key.endswith("_dir") or key.endswith("root"):
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)


def write_reports(cfg: Dict[str, Any], results: List[Dict[str, Any]], repo_root: Path) -> None:
    outputs = cfg.get("outputs", {})

    json_path = repo_root / outputs.get(
        "resolved_manifest_json",
        "outputs/villoc/traj01_90deg_stable120m/reporting/manifests/report_inputs_resolved.json",
    )
    md_path = repo_root / outputs.get(
        "resolved_manifest_md",
        "outputs/villoc/traj01_90deg_stable120m/reporting/manifests/report_inputs_resolved.md",
    )

    ensure_parent(json_path)
    ensure_parent(md_path)

    summary = {
        "report_id": cfg.get("report", {}).get("id"),
        "dataset": cfg.get("project", {}).get("dataset_name"),
        "total_entries": len(results),
        "ok_count": sum(1 for r in results if r["ok"]),
        "error_count": sum(len(r["errors"]) for r in results),
        "warning_count": sum(len(r["warnings"]) for r in results),
        "results": results,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines: List[str] = []
    lines.append("# Report input validation\n")
    lines.append(f"- Report id: `{summary['report_id']}`")
    lines.append(f"- Dataset: `{summary['dataset']}`")
    lines.append(f"- Entries checked: {summary['total_entries']}")
    lines.append(f"- OK entries: {summary['ok_count']}")
    lines.append(f"- Errors: {summary['error_count']}")
    lines.append(f"- Warnings: {summary['warning_count']}")
    lines.append("\n## Results\n")
    lines.append("| Input | Required | Exists | OK | Rows | Notes |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for r in results:
        rows = r.get("info", {}).get("rows", "")
        notes = []
        if r["errors"]:
            notes.append("ERROR: " + "; ".join(r["errors"]))
        if r["warnings"]:
            notes.append("WARN: " + "; ".join(r["warnings"]))
        if not notes:
            notes.append("OK")
        lines.append(
            f"| `{r['name']}` | {r['required']} | {r['exists']} | {r['ok']} | {rows} | {' '.join(notes)} |"
        )

    lines.append("\n## Missing or problematic required inputs\n")
    required_bad = [r for r in results if r["required"] and not r["ok"]]
    if required_bad:
        for r in required_bad:
            lines.append(f"- `{r['name']}` → `{r.get('path')}`: {'; '.join(r['errors'])}")
    else:
        lines.append("- None.")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote JSON manifest: {json_path}")
    print(f"Wrote Markdown report: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/reporting/report_villoc_traj01_s8_figures.yaml",
        help="Reporting YAML config.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not exit non-zero if required inputs are missing.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    cfg_path = (repo_root / args.config).resolve()

    cfg = load_yaml(cfg_path)

    if cfg.get("validation", {}).get("create_output_directories", True):
        create_output_dirs(cfg, repo_root)

    entries = collect_input_entries(cfg)
    results = [validate_entry(name, entry, repo_root) for name, entry in entries]

    write_reports(cfg, results, repo_root)

    error_count = sum(len(r["errors"]) for r in results)
    required_bad = [r for r in results if r["required"] and not r["ok"]]

    print("\nValidation summary")
    print("------------------")
    print(f"Entries checked: {len(results)}")
    print(f"Errors: {error_count}")
    print(f"Required bad: {len(required_bad)}")

    if required_bad:
        print("\nRequired inputs needing attention:")
        for r in required_bad:
            print(f"- {r['name']}: {r.get('path')}")

    if required_bad and not args.no_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
