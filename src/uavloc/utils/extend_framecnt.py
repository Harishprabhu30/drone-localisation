import re

INPUT = "data/raw/villoc/traj01_90deg_stable70m/villoc_traj01_90deg_stable70m_V_merged.srt"
OUTPUT = "data/raw/villoc/traj01_90deg_stable70m/villoc_traj01_90deg_stable70m_V_merged_continuous_framecnt.srt"


frame_pattern = re.compile(r"FrameCnt:\s*(\d+)")


with open(INPUT, "r", encoding="utf-8") as f:
    text = f.read()


# Get all FrameCnt values
frames = [int(x) for x in frame_pattern.findall(text)]

if not frames:
    raise RuntimeError("No FrameCnt values found")


# Detect first video start and end
first_frame_start = frames[0]

previous = frames[0]
first_video_end_index = 0

for i, frame in enumerate(frames):
    if frame < previous:
        # DJI FrameCnt reset detected -> second video starts
        first_video_end_index = i
        break

    previous = frame
else:
    first_video_end_index = len(frames)


first_video_frames = frames[:first_video_end_index]
second_video_frames = frames[first_video_end_index:]


first_video_last = first_video_frames[-1]

print(f"First video original start FrameCnt: {first_frame_start}")
print(f"First video original end FrameCnt: {first_video_last}")
print(f"Second video starts at index: {first_video_end_index}")


# Offset to make first video start at zero
first_offset = first_frame_start


# Offset for second video continuation
second_offset = first_video_last - 1


print(f"First video offset: -{first_offset}")
print(f"Second video offset: +{second_offset}")


counter = 0
inside_second = False


def replace_frame(match):
    global counter, inside_second

    old_frame = int(match.group(1))

    # Detect reset
    if old_frame == second_video_frames[0] and counter > first_video_end_index:
        inside_second = True

    if counter < first_video_end_index:
        # Video 1 rebased to zero
        new_frame = old_frame - first_offset
    else:
        # Video 2 continues
        new_frame = old_frame + second_offset - first_offset

    counter += 1

    return f"FrameCnt: {new_frame}"


updated = frame_pattern.sub(replace_frame, text)


with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(updated)


print("Saved:", OUTPUT)
