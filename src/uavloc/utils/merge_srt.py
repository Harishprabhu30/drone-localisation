from datetime import timedelta
import re

srt_files = [
    ("data/raw/villoc/traj01_90deg_stable70m/DJI_20260729104901_0002_V.srt", 0),
    ("data/raw/villoc/traj01_90deg_stable70m/DJI_20260729105250_0003_V.srt", 173.729),
]

output = "data/raw/villoc/traj01_90deg_stable70m/merged.srt"


def parse_time(t):
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return timedelta(
        hours=int(h),
        minutes=int(m),
        seconds=int(s),
        milliseconds=int(ms)
    )


def format_time(td):
    ms = int(td.total_seconds() * 1000)
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


pattern = re.compile(
    r"(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)"
)

counter = 1
merged = []

for filename, offset_sec in srt_files:
    offset = timedelta(seconds=offset_sec)

    with open(filename, encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")

    for block in blocks:
        lines = block.splitlines()

        if len(lines) < 3:
            continue

        match = pattern.match(lines[1])
        if not match:
            continue

        start = parse_time(match.group(1)) + offset
        end = parse_time(match.group(2)) + offset

        merged.append(
            "\n".join([
                str(counter),
                f"{format_time(start)} --> {format_time(end)}",
                *lines[2:]
            ])
        )

        counter += 1


with open(output, "w", encoding="utf-8") as f:
    f.write("\n\n".join(merged))

print(f"Wrote {counter-1} subtitles")
