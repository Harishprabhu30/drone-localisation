'''
ROOT=outputs/villoc/traj01_90deg_stable120m
RUN="$ROOT/blind_demo_addons/eval_traj01_known_gt"
MEM="$RUN/metrics/stage_memory"

mkdir -p \
  "$MEM" \
  "$RUN/logs"

python scripts/villoc/blind_demo/measure_stage_memory.py \
  --stage correction_manifest_build \
  --output-json \
  "$MEM/correction_manifest_build.json" \
  --log \
  "$RUN/logs/memory_correction_manifest_build.log" \
  --sample-interval-s 0.10 \
  -- \
  python scripts/villoc/s8_fusion/s8_f1_build_absolute_correction_manifest.py \
    --root "$ROOT" \
    --intervals-m 50,100,200,400
'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError as exc:
    raise SystemExit(
        "measure_stage_memory.py requires psutil. "
        "It is intentionally used because Add-on 5 "
        "needs live peak-RSS measurement."
    ) from exc


MIB = 1024.0 ** 2
GIB = 1024.0 ** 3


def mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / MIB


def gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / GIB


def process_rss(process: psutil.Process) -> int | None:
    try:
        return int(
            process.memory_info().rss
        )
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        psutil.ZombieProcess,
    ):
        return None


def process_tree_snapshot(
    root: psutil.Process,
) -> dict[str, Any]:
    processes: dict[int, psutil.Process] = {}

    try:
        processes[root.pid] = root

        for child in root.children(
            recursive=True
        ):
            processes[child.pid] = child

    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        psutil.ZombieProcess,
    ):
        pass

    root_rss = process_rss(root)

    tree_rss = 0
    live_count = 0

    for proc in processes.values():
        rss = process_rss(proc)

        if rss is None:
            continue

        tree_rss += rss
        live_count += 1

    return {
        "root_rss_bytes": root_rss,
        "tree_rss_bytes": (
            tree_rss
            if live_count
            else None
        ),
        "live_process_count": live_count,
    }


def system_memory_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()

    return {
        "total_bytes": int(vm.total),
        "total_gib": gib(vm.total),
        "available_bytes": int(
            vm.available
        ),
        "available_gib": gib(
            vm.available
        ),
        "used_bytes": int(vm.used),
        "used_gib": gib(vm.used),
        "percent": float(vm.percent),
    }


def tee_output(
    stream,
    log_file,
) -> None:
    try:
        for line in iter(
            stream.readline,
            "",
        ):
            sys.stdout.write(line)
            sys.stdout.flush()

            log_file.write(line)
            log_file.flush()

    finally:
        try:
            stream.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pipeline stage while sampling "
            "its process-tree RSS with psutil."
        )
    )

    parser.add_argument(
        "--stage",
        required=True,
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--log",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    command = list(
        args.command
    )

    if (
        command
        and command[0] == "--"
    ):
        command = command[1:]

    if not command:
        raise RuntimeError(
            "No stage command supplied after --"
        )

    if args.sample_interval_s <= 0:
        raise ValueError(
            "--sample-interval-s must be > 0"
        )

    cwd = args.cwd.resolve()

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.log.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    monitor = psutil.Process(
        os.getpid()
    )

    monitor_rss_before = (
        process_rss(monitor)
    )

    system_before = (
        system_memory_snapshot()
    )

    env = os.environ.copy()

    # Makes Python child logs appear through the
    # monitor without long buffering delays.
    env.setdefault(
        "PYTHONUNBUFFERED",
        "1",
    )

    started = time.perf_counter()

    with args.log.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        print("=" * 78)
        print(
            "LIVE STAGE MEMORY MONITOR"
        )
        print("=" * 78)
        print(
            f"stage: {args.stage}"
        )
        print(
            "command:",
            " ".join(command),
        )
        print(
            f"sample interval: "
            f"{args.sample_interval_s:.3f} s"
        )
        print()

        child = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if child.stdout is None:
            raise RuntimeError(
                "Failed to capture child stdout."
            )

        reader = threading.Thread(
            target=tee_output,
            args=(
                child.stdout,
                log_file,
            ),
            daemon=True,
        )

        reader.start()

        root = psutil.Process(
            child.pid
        )

        sample_count = 0

        first_root_rss = None
        first_tree_rss = None

        last_root_rss = None
        last_tree_rss = None

        peak_root_rss = 0
        peak_tree_rss = 0

        peak_process_count = 0

        while child.poll() is None:
            snap = (
                process_tree_snapshot(
                    root
                )
            )

            root_rss = snap[
                "root_rss_bytes"
            ]

            tree_rss = snap[
                "tree_rss_bytes"
            ]

            proc_count = int(
                snap[
                    "live_process_count"
                ]
            )

            if root_rss is not None:
                if first_root_rss is None:
                    first_root_rss = (
                        root_rss
                    )

                last_root_rss = (
                    root_rss
                )

                peak_root_rss = max(
                    peak_root_rss,
                    root_rss,
                )

            if tree_rss is not None:
                if first_tree_rss is None:
                    first_tree_rss = (
                        tree_rss
                    )

                last_tree_rss = (
                    tree_rss
                )

                peak_tree_rss = max(
                    peak_tree_rss,
                    tree_rss,
                )

            peak_process_count = max(
                peak_process_count,
                proc_count,
            )

            sample_count += 1

            time.sleep(
                args.sample_interval_s
            )

        exit_code = child.wait()

        reader.join(
            timeout=5.0
        )

    finished = time.perf_counter()

    monitor_rss_after = (
        process_rss(monitor)
    )

    system_after = (
        system_memory_snapshot()
    )

    report = {
        "stage": args.stage,
        "status": (
            "PASS"
            if exit_code == 0
            else "FAIL"
        ),
        "measurement_method": (
            "psutil_process_tree_rss_sampling"
        ),
        "important_note": (
            "Tree RSS is the sum of RSS for the "
            "root process and live child processes. "
            "Shared pages may therefore be counted "
            "more than once if a stage uses multiple "
            "processes. The current Villoc stages are "
            "primarily single-process."
        ),
        "command": command,
        "cwd": str(cwd),
        "sample_interval_s": float(
            args.sample_interval_s
        ),
        "sample_count": int(
            sample_count
        ),
        "exit_code": int(
            exit_code
        ),
        "wall_time_s": float(
            finished - started
        ),
        "process_memory": {
            "rss_before_work_bytes": (
                first_tree_rss
            ),
            "rss_before_work_mib": (
                mib(first_tree_rss)
            ),
            "rss_after_work_last_live_bytes": (
                last_tree_rss
            ),
            "rss_after_work_last_live_mib": (
                mib(last_tree_rss)
            ),
            "rss_after_process_exit_mib": 0.0,
            "peak_root_rss_bytes": (
                peak_root_rss
            ),
            "peak_root_rss_mib": (
                mib(peak_root_rss)
            ),
            "peak_process_tree_rss_bytes": (
                peak_tree_rss
            ),
            "peak_process_tree_rss_mib": (
                mib(peak_tree_rss)
            ),
            "peak_live_process_count": (
                peak_process_count
            ),
        },
        "monitor_process": {
            "rss_before_bytes": (
                monitor_rss_before
            ),
            "rss_before_mib": (
                mib(monitor_rss_before)
            ),
            "rss_after_bytes": (
                monitor_rss_after
            ),
            "rss_after_mib": (
                mib(monitor_rss_after)
            ),
        },
        "system_memory_before": (
            system_before
        ),
        "system_memory_after": (
            system_after
        ),
        "log": str(
            args.log.resolve()
        ),
    }

    args.output_json.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "MEMORY MEASUREMENT SUMMARY"
    )
    print("=" * 78)

    print(
        f"stage:                  "
        f"{args.stage}"
    )

    print(
        f"exit code:              "
        f"{exit_code}"
    )

    print(
        f"wall time:              "
        f"{finished - started:.3f} s"
    )

    print(
        f"samples:                "
        f"{sample_count}"
    )

    print(
        f"RSS before work:        "
        f"{mib(first_tree_rss)} MiB"
    )

    print(
        f"RSS last live sample:   "
        f"{mib(last_tree_rss)} MiB"
    )

    print(
        f"peak root RSS:          "
        f"{mib(peak_root_rss):.3f} MiB"
    )

    print(
        f"peak process-tree RSS:  "
        f"{mib(peak_tree_rss):.3f} MiB"
    )

    print(
        f"peak process count:     "
        f"{peak_process_count}"
    )

    print()
    print(
        "saved:",
        args.output_json,
    )

    if exit_code != 0:
        raise SystemExit(
            exit_code
        )


if __name__ == "__main__":
    main()
