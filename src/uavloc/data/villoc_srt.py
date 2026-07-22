from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


_TIME_RE = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})"
)

_FIELD_RE = re.compile(r"\[([^\[\]]+?)\]")


def _srt_time_to_seconds(value: str) -> float:
    match = _TIME_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")

    h = int(match.group("h"))
    m = int(match.group("m"))
    s = int(match.group("s"))
    ms = int(match.group("ms"))

    return h * 3600.0 + m * 60.0 + s + ms / 1000.0


def _parse_subtitle_time(line: str) -> tuple[float, float]:
    left, right = line.split("-->")
    return _srt_time_to_seconds(left), _srt_time_to_seconds(right)


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_bracket_fields(text: str) -> Dict[str, str]:
    """Parse DJI/Anzu SRT bracket fields.

    Handles both simple fields:
        [focal_len: 24.00]

    and compound fields:
        [rel_alt: 70.020 abs_alt: 280.465]
        [gb_yaw: 178.7 gb_pitch: -89.9 gb_roll: 0.0]
    """
    fields: Dict[str, str] = {}

    # Work only inside bracketed metadata fields.
    for raw in _FIELD_RE.findall(text):
        clean = raw.strip()

        # Match key:value pairs until the next key:value or end of bracket.
        # This handles values like "1/933.94", "default", "-89.9", etc.
        for match in re.finditer(
            r"([A-Za-z_][A-Za-z0-9_ ]*?)\s*:\s*([^:]+?)(?=\s+[A-Za-z_][A-Za-z0-9_ ]*?\s*:|$)",
            clean,
        ):
            key = match.group(1).strip()
            value = match.group(2).strip().strip(",")

            # Normalize accidental spaces in keys such as "color_md ".
            key = re.sub(r"\s+", "_", key)

            fields[key] = value

    return fields


def parse_villoc_srt(
    srt_path: str | Path,
    *,
    video_id: str,
    modality: str,
    role: str,
) -> pd.DataFrame:
    srt_path = Path(srt_path)
    text = srt_path.read_text(encoding="utf-8", errors="replace")

    blocks = re.split(r"\n\s*\n", text.strip())
    rows: List[dict] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue

        try:
            srt_index = int(lines[0])
            subtitle_start_s, subtitle_end_s = _parse_subtitle_time(lines[1])
        except Exception:
            continue

        body = "\n".join(lines[2:])

        frame_match = re.search(r"FrameCnt:\s*(\d+)", body)
        diff_match = re.search(r"DiffTime:\s*(\d+)ms", body)
        timestamp_match = re.search(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)", body
        )

        fields = _parse_bracket_fields(body)

        row = {
            "srt_index": srt_index,
            "frame_cnt": int(frame_match.group(1)) if frame_match else None,
            "subtitle_start_s": subtitle_start_s,
            "subtitle_end_s": subtitle_end_s,
            "video_time_s": subtitle_start_s,
            "diff_time_ms": int(diff_match.group(1)) if diff_match else None,
            "timestamp_local": timestamp_match.group(1) if timestamp_match else None,
            "focal_len_mm": _safe_float(fields.get("focal_len")),
            "dzoom_ratio": _safe_float(fields.get("dzoom_ratio")),
            "iso": _safe_float(fields.get("iso")),
            "shutter": fields.get("shutter"),
            "fnum": _safe_float(fields.get("fnum")),
            "ev": _safe_float(fields.get("ev")),
            "color_md": fields.get("color_md"),
            "ae_meter_md": fields.get("ae_meter_md"),
            "lat": _safe_float(fields.get("latitude")),
            "lon": _safe_float(fields.get("longitude")),
            "rel_alt_m": _safe_float(fields.get("rel_alt")),
            "abs_alt_m": _safe_float(fields.get("abs_alt")),
            "gb_yaw_deg": _safe_float(fields.get("gb_yaw")),
            "gb_pitch_deg": _safe_float(fields.get("gb_pitch")),
            "gb_roll_deg": _safe_float(fields.get("gb_roll")),
            "source_srt": str(srt_path),
            "video_id": video_id,
            "modality": modality,
            "role": role,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df["timestamp_local"] = pd.to_datetime(
            df["timestamp_local"], errors="coerce"
        )

    return df
