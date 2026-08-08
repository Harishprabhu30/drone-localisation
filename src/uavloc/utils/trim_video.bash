#!/usr/bin/env bash

# Trim a video WITHOUT re-encoding (no quality loss).
#
# Usage:
#   ./trim.sh input.mp4 00:10:30 00:12:45
#
# Arguments:
#   input.mp4   Input video file
#   00:10:30    Start timestamp (HH:MM:SS)
#   00:12:45    End timestamp (HH:MM:SS)
#
# Options:
#   -h, --help  Show this help message
#
# Example:
#   ./trim.sh video.mp4 00:10:30 00:12:45
#
# Note:
#   -c copy means FFmpeg does NOT re-encode the video/audio.
#   This makes the operation very fast and preserves original quality.
#   Cuts may not be perfectly frame-accurate because FFmpeg is copying
#   existing encoded frames rather than decoding/re-encoding them.

set -e

show_help() {
    sed -n '3,25p' "$0"
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <input> <start> <end>"
    echo "Try '$0 --help' for more information."
    exit 1
fi

INPUT="$1"
START="$2"
END="$3"

# Remove the extension from the input filename and create an output name.
BASENAME="${INPUT%.*}"
EXT="${INPUT##*.}"
OUTPUT="${BASENAME}_trimmed.${EXT}"

# Trim without re-encoding.
ffmpeg -ss "$START" -i "$INPUT" -to "$END" -c copy "$OUTPUT"

echo "Done: $OUTPUT"
