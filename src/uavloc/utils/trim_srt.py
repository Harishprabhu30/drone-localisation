import re
from datetime import timedelta

INPUT = "data/raw/villoc/traj01/DJI_20260729104901_0002_V.srt"
OUTPUT = "data/raw/villoc/traj01/output.srt"

START = timedelta(seconds=55)
END = timedelta(minutes=3, seconds=49)


def parse_time(s):
    h, m, rest = s.split(":")
    sec, ms = rest.split(",")
    return timedelta(
        hours=int(h),
        minutes=int(m),
        seconds=int(sec),
        milliseconds=int(ms),
    )


def format_time(td):
    total_ms = int(td.total_seconds() * 1000)
    if total_ms < 0:
        total_ms = 0
    h = total_ms // 3600000
    total_ms %= 3600000
    m = total_ms // 60000
    total_ms %= 60000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


with open(INPUT, encoding="utf-8") as f:
    text = f.read().strip()

blocks = re.split(r"\n\s*\n", text)

out = []
idx = 1

for block in blocks:
    lines = block.splitlines()
    if len(lines) < 3:
        continue

    m = re.match(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
        lines[1],
    )
    if not m:
        continue

    t0 = parse_time(m.group(1))
    t1 = parse_time(m.group(2))

    if t1 <= START:
        continue
    if t0 >= END:
        continue

    new0 = max(t0 - START, timedelta())
    new1 = min(t1, END) - START

    out.append(
        "\n".join(
            [
                str(idx),
                f"{format_time(new0)} --> {format_time(new1)}",
                *lines[2:],
            ]
        )
    )
    idx += 1

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n\n".join(out))

print(f"Wrote {idx-1} subtitles.")
