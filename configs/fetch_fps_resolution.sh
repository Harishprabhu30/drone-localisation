source .drone_venv/bin/activate
export PYTHONPATH=$PWD/src

python - <<'PY'
from pathlib import Path
import cv2
import re

# raw_root = Path("data/raw/villoc/45_deg")
raw_root = Path("data/raw/villoc/traj01_90deg_stable70m")

streams = {
    "V": {
        "video": "villoc_traj01_90deg_stable70m_V_merged.MP4",
        "srt": "villoc_traj01_90deg_stable70m_V_merged_continuous_framecnt.SRT",
    },
    # "S": {
    #     "video": "CAM_20260413121601_0001_S.MP4",
    #     "srt": "CAM_20260413121601_0001_S.SRT",
    # },
    # "T": {
    #     "video": "CAM_20260413121601_0001_T.MP4",
    #     "srt": "CAM_20260413121601_0001_T.SRT",
    # },
}

print("\nVilloc 90° traj01_stable70m raw video/SRT preflight")
print("-" * 90)
print(f"{'Stream':<8} {'Video exists':<13} {'SRT exists':<11} {'FPS':<10} {'Resolution':<16} {'Frames':<10} {'Duration(s)':<12} {'SRT rows':<9}")
print("-" * 90)

for stream_id, files in streams.items():
    video_path = raw_root / files["video"]
    srt_path = raw_root / files["srt"]

    video_exists = video_path.exists()
    srt_exists = srt_path.exists()

    fps = None
    width = None
    height = None
    frame_count = None
    duration_s = None

    if video_exists:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = frame_count / fps if fps and fps > 0 else None
        cap.release()

    srt_rows = None
    if srt_exists:
        text = srt_path.read_text(errors="ignore")
        # Counts SRT subtitle blocks by timing lines.
        srt_rows = len(re.findall(r"\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}", text))

    print(
        f"{stream_id:<8} "
        f"{str(video_exists):<13} "
        f"{str(srt_exists):<11} "
        f"{fps if fps is not None else 'NA':<10.3f} "
        f"{f'{width}x{height}' if width else 'NA':<16} "
        f"{frame_count if frame_count is not None else 'NA':<10} "
        f"{duration_s if duration_s is not None else 'NA':<12.3f} "
        f"{srt_rows if srt_rows is not None else 'NA':<9}"
    )

print("-" * 90)
print("\nUse FPS and Resolution columns to update fps_expected and resolution_expected in configs/dataset_villoc_45deg.yaml")
PY
