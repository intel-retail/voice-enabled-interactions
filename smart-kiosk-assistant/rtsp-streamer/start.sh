#!/bin/sh
set -eu

MEDIA_DIR=${MEDIA_DIR:-/media}
RTSP_PORT=${RTSP_PORT:-8554}
FFMPEG_BIN=${FFMPEG_BIN:-ffmpeg}
MEDIAMTX_BIN=${MEDIAMTX_BIN:-/opt/rtsp-streamer/mediamtx}

# Logging helper — prefixes every message with ISO timestamp and [rtsp-streamer]
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') [rtsp-streamer] $*"
}

# Loop configuration
# LOOP_COUNT: number of times to play the video (default -1 = infinite)
#   -1 = infinite loop
#    1 = play once
#    2 = play twice with optional blank frames between
LOOP_COUNT=${LOOP_COUNT:--1}

# BLANK_DURATION: seconds of black frames between loops (default 0 = no blanks)
# Only used when LOOP_COUNT > 1
BLANK_DURATION=${BLANK_DURATION:-0}

if [ ! -d "$MEDIA_DIR" ]; then
  log "ERROR Media directory $MEDIA_DIR does not exist" >&2
  exit 1
fi

if [ ! -x "$MEDIAMTX_BIN" ]; then
  log "ERROR mediamtx binary $MEDIAMTX_BIN not found or not executable" >&2
  exit 1
fi

# Input selection:
#   MEDIA_FILE  - stream only this single file (absolute path, or relative to
#                 MEDIA_DIR). When unset, every *.mp4 in MEDIA_DIR is published
#                 (original behavior). Other .mp4 files in MEDIA_DIR are kept on
#                 disk but ignored.
#   STREAM_NAME - override the published RTSP path. When unset, the path is the
#                 input filename without extension. Set this to keep a stable
#                 path for consumers when the underlying file changes.
if [ -n "${MEDIA_FILE:-}" ]; then
  case "$MEDIA_FILE" in
    /*) selected_file="$MEDIA_FILE" ;;
    *)  selected_file="$MEDIA_DIR/$MEDIA_FILE" ;;
  esac
  if [ ! -f "$selected_file" ]; then
    log "ERROR MEDIA_FILE '$selected_file' not found" >&2
    exit 1
  fi
  set -- "$selected_file"
else
  set -- "$MEDIA_DIR"/*.mp4
  if [ ! -e "$1" ]; then
    log "ERROR No .mp4 files found in $MEDIA_DIR" >&2
    exit 1
  fi
fi

log "Starting mediamtx RTSP server on port $RTSP_PORT ..."
"$MEDIAMTX_BIN" >/tmp/mediamtx.log 2>&1 &
mediamtx_pid=$!
pids="$mediamtx_pid"
log "mediamtx started (PID=$mediamtx_pid)"

# Wait for RTSP server to accept connections
log "Waiting for mediamtx to accept connections on port $RTSP_PORT ..."
retry=50
while ! nc -z 127.0.0.1 "$RTSP_PORT" >/dev/null 2>&1; do
  retry=$((retry - 1))
  if [ "$retry" -le 0 ]; then
    log "ERROR RTSP server failed to start on port $RTSP_PORT" >&2
    kill "$mediamtx_pid"
    wait "$mediamtx_pid" 2>/dev/null || true
    exit 1
  fi
  sleep 0.2
done
log "mediamtx is ready and accepting connections on port $RTSP_PORT"

# Function to generate black video with same properties as source
generate_black_video() {
  src_file=$1
  black_file=$2
  duration=$3
  
  # Get video properties from source
  resolution=$("$FFMPEG_BIN" -i "$src_file" 2>&1 | grep -oP '\d{3,4}x\d{3,4}' | head -1)
  fps=$("$FFMPEG_BIN" -i "$src_file" 2>&1 | grep -oP '\d+(\.\d+)? fps' | head -1 | grep -oP '[\d.]+')
  
  # Default to 1920x1080 @ 30fps if detection fails
  resolution=${resolution:-1920x1080}
  fps=${fps:-30}
  
  log "Generating ${duration}s black video at ${resolution} @ ${fps}fps"
  
  "$FFMPEG_BIN" -hide_banner -loglevel error \
    -f lavfi -i "color=c=black:s=${resolution}:r=${fps}:d=${duration}" \
    -c:v libx264 -preset ultrafast -tune stillimage \
    -pix_fmt yuv420p \
    -y "$black_file"
}

# Function to create concatenated video for limited loops with blanks
create_looped_video() {
  src_file=$1
  output_file=$2
  loop_count=$3
  blank_dur=$4
  
  tmpdir=$(mktemp -d)
  concat_list="$tmpdir/concat.txt"
  black_file="$tmpdir/black.mp4"
  
  # Generate black video if needed
  if [ "$blank_dur" -gt 0 ]; then
    generate_black_video "$src_file" "$black_file" "$blank_dur"
  fi
  
  # Build concat list
  i=1
  while [ "$i" -le "$loop_count" ]; do
    echo "file '$src_file'" >> "$concat_list"
    # Add black frames between loops (not after the last one)
    if [ "$blank_dur" -gt 0 ] && [ "$i" -lt "$loop_count" ]; then
      echo "file '$black_file'" >> "$concat_list"
    fi
    i=$((i + 1))
  done
  
  log "Concat list:"
  cat "$concat_list"
  
  # Create concatenated video
  "$FFMPEG_BIN" -hide_banner -loglevel info \
    -f concat -safe 0 -i "$concat_list" \
    -c copy \
    -y "$output_file"
  
  # Cleanup temp files (keep black_file path for later cleanup)
  rm -f "$concat_list"
  rm -f "$black_file"
  rmdir "$tmpdir" 2>/dev/null || true
}

# Map stream_name → station for logging
station_index=0

pids="$pids"
for file in "$@"; do
  [ -f "$file" ] || continue
  filename=$(basename "$file")
  stream_name=${STREAM_NAME:-${filename%.*}}
  station_index=$((station_index + 1))
  station_label="station_${station_index}"
  
  log "[${station_label}] Preparing RTSP stream '${stream_name}' from $file (loop_count=$LOOP_COUNT, blank_duration=${BLANK_DURATION}s)"
  
  # Determine streaming mode
  if [ "$LOOP_COUNT" -eq -1 ]; then
    # Infinite loop mode (original behavior)
    stream_loop_arg="-1"
    input_file="$file"
  elif [ "$LOOP_COUNT" -eq 1 ]; then
    # Single play, no loop
    stream_loop_arg="0"
    input_file="$file"
  else
    # Limited loops with optional blank frames
    if [ "$BLANK_DURATION" -gt 0 ]; then
      # Create concatenated video with blank frames
      looped_file="/tmp/${stream_name}_looped.mp4"
      log "[${station_label}] Creating looped video with ${BLANK_DURATION}s blank frames between ${LOOP_COUNT} loops..."
      create_looped_video "$file" "$looped_file" "$LOOP_COUNT" "$BLANK_DURATION"
      stream_loop_arg="0"  # Play the concatenated file once
      input_file="$looped_file"
    else
      # Simple loop without blanks
      stream_loop_arg="$((LOOP_COUNT - 1))"
      input_file="$file"
    fi
  fi
  
  # -re             throttles playback to real-time
  # -stream_loop    controls looping behavior
  # -an             drop audio (not needed for computer-vision pipelines)
  # -c:v libx264    re-encode to RTSP/RTP-friendly H.264 (MP4 -c copy is
  #                 rejected by MediaMTX with "400 Bad Request" because the
  #                 in-file parameter sets do not produce a valid SDP)
  # -profile baseline + -pix_fmt yuv420p  maximise decoder/RTP compatibility
  # -g 30           regular keyframes so consumers can join the stream quickly
  # -rtsp_transport tcp  publish over TCP
  # -f rtsp         publish to MediaMTX via RTSP protocol
  log "[${station_label}] Starting ffmpeg for stream '${stream_name}' → rtsp://127.0.0.1:${RTSP_PORT}/${stream_name} (PID will follow)"
  log "[${station_label}]   input_file=$input_file  stream_loop=$stream_loop_arg"
  
  "$FFMPEG_BIN" \
    -hide_banner \
    -loglevel info \
    -re \
    -stream_loop "$stream_loop_arg" \
    -i "$input_file" \
    -an \
    -c:v libx264 \
    -profile:v baseline \
    -pix_fmt yuv420p \
    -preset veryfast \
    -tune zerolatency \
    -g 30 \
    -rtsp_transport tcp \
    -f rtsp \
    "rtsp://127.0.0.1:${RTSP_PORT}/${stream_name}" &
  pid=$!
  pids="$pids $pid"
  log "[${station_label}] ffmpeg started for stream '${stream_name}' (PID=$pid, RTSP=rtsp://0.0.0.0:${RTSP_PORT}/${stream_name})"
  sleep 0.2
done

log "All streams started. Total streams: $station_index"

cleanup() {
  log "Shutdown signal received — stopping all RTSP streams ..."
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      log "Sending SIGTERM to PID=$pid"
      kill "$pid"
    fi
  done
  log "Cleanup complete"
}

trap 'cleanup' INT TERM

status=0
for pid in $pids; do
  if ! wait "$pid"; then
    status=$?
  fi
done

exit $status