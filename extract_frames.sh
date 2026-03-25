#!/bin/bash
set -e

VIDEO_PATH="$1"
NUM_FRAMES="${2:-100}"

if [ -z "$VIDEO_PATH" ]; then
    echo "Usage: $0 <video_path> [num_frames]"
    echo "  video_path: path to the mp4 file"
    echo "  num_frames: number of frames to extract (default: 100)"
    exit 1
fi

VIDEO_DIR="$(dirname "$VIDEO_PATH")"
VIDEO_NAME="$(basename "$VIDEO_PATH" .mp4)"
OUT_DIR="${VIDEO_DIR}/${VIDEO_NAME}/images"

TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$VIDEO_PATH")
echo "Video: $VIDEO_PATH"
echo "Total frames: $TOTAL_FRAMES"
echo "Extracting: $NUM_FRAMES frames"
echo "Output: $OUT_DIR"

STEP=$(echo "$TOTAL_FRAMES $NUM_FRAMES" | awk '{printf "%.4f", $1 / $2}')
SELECT_EXPR="select='lt(mod(n\\,${TOTAL_FRAMES})\\,0)+$(echo "$TOTAL_FRAMES $NUM_FRAMES" | awk '{for(i=0;i<'$NUM_FRAMES';i++) printf "eq(n\\," int(i * '$TOTAL_FRAMES' / '$NUM_FRAMES') ")+" ; print "0"}')'"

mkdir -p "$OUT_DIR"

TMP_DIR=$(mktemp -d)
ffmpeg -i "$VIDEO_PATH" \
    -vf "select='not(mod(n\,$(echo "$TOTAL_FRAMES $NUM_FRAMES" | awk '{print int($1/$2)}')))'" \
    -frames:v "$NUM_FRAMES" \
    -vsync vfr \
    "$TMP_DIR/%04d.png" -y 2>&1 | tail -3

i=0
for f in $(ls "$TMP_DIR"/*.png | sort); do
    mv "$f" "$OUT_DIR/$(printf '%03d' $i).png"
    i=$((i + 1))
done
rm -rf "$TMP_DIR"

ACTUAL=$(ls "$OUT_DIR"/*.png | wc -l)
echo "Done! Extracted $ACTUAL frames to $OUT_DIR"
