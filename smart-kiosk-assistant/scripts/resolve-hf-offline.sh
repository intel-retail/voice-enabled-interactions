#!/usr/bin/env sh
#
# Resolve the HuggingFace offline mode to start the stack with, and refuse to
# start a run that is guaranteed to stall.
#
# Background (ITEP-97604)
# -----------------------
# setup_models.sh does NOT pre-fetch the ASR/TTS/RAG models; they are pulled
# in-container on the first online run. On a host with no egress and no proxy,
# audio-analyzer and text-to-speech therefore raise httpx.ConnectError during
# FastAPI lifespan, their healthcheck never passes, and every service gated on
# `depends_on: service_healthy` aborts after the full 600s timeout. `make up`
# previously started anyway and simply stalled.
#
# Decision table (when HF_HUB_OFFLINE is not already 1):
#
#   egress reachable                 -> 0  (online, unchanged)
#   no egress + all caches populated -> 1  (auto offline; the stack can run)
#   no egress + caches cold          -> exit 3 (fail fast with remediation)
#
# HF_HUB_OFFLINE=1 is always honoured untouched: forcing offline is a
# deliberate choice and this script never overrides it.
#
# Set HF_AUTO_OFFLINE=false to disable both the auto-switch and the fail-fast
# and restore the previous warn-and-continue behaviour.
#
# Writes the resolved value ("0" or "1") to stdout. All diagnostics go to
# stderr so the caller can safely do VALUE=$(resolve-hf-offline.sh).
#
# Set HF_RESOLVE_DRY_RUN=true to report the verdict without acting on it: the
# same probe runs, a human-readable summary goes to stderr, and the script
# always exits 0 (it never blocks). `check-env` uses this so there is exactly
# ONE probe implementation in the repo. An earlier inline copy of this logic
# in the Makefile drifted out of sync and contradicted this script -- see the
# comment on the `check-env` target.
#
# Exit codes: 0 resolved, 3 blocked (no egress, cold cache; never in dry run).

set -u

say() { printf '%s\n' "$*" >&2; }

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
HF_AUTO_OFFLINE="${HF_AUTO_OFFLINE:-true}"
COMPOSE_PROJECT="${COMPOSE_PROJECT:-smart-kiosk-assistant}"
HF_RESOLVE_DRY_RUN="${HF_RESOLVE_DRY_RUN:-false}"

dry_run() { [ "$HF_RESOLVE_DRY_RUN" = "true" ] || [ "$HF_RESOLVE_DRY_RUN" = "TRUE" ]; }

# Volumes that must hold real weights before an offline start can work.
# audio-analyzer is represented by its IR volume, not its HF source cache:
# cached source weights alone are not enough (see the long note below).
CACHE_VOLUMES="audio_analyzer_models text_to_speech_cache rag_service_hf_cache"
# Probe from inside a container, not the host: the host may have egress (or a
# proxy in the shell) that the containers do not.
#
# The probe tool matters more than the probe image. huggingface.co sits behind
# an edge that rejects busybox wget with "403 Forbidden" even when the network
# path is perfectly fine -- verified on a host where busybox wget got 403 and
# curl got 200 through the SAME proxy, with and without it. Treating that 403
# as "no egress" forces HF_HUB_OFFLINE=1 on a perfectly online host, and that
# breaks startup: `optimum-cli export openvino` has to reach the Hub API to
# infer a model's task, so a first-run IR export fails offline even when the
# source weights are cached.
#
# So: prefer an image that has curl, and never let a wget 403 alone mean
# "unreachable". The stack's own images carry curl and are the most faithful
# probe available -- same base, same TLS stack, same network as the real
# workload -- so they come first; alpine/busybox are a last resort.
#
# Pick by CAPABILITY, not by name: some service images ship neither curl nor
# wget (audio-analyzer is one), so probing with them silently fails with
# "wget: not found" and looks exactly like no egress. Ask each candidate what
# it actually has, prefer curl, and remember which tool was found.
PROBE_IMG=""
PROBE_TOOL=""
select_probe_image() {
    for I in "${REGISTRY_REPO:-intel}/text-to-speech:${RELEASE_TAG:-2026.3.0-rc1}" \
             "${REGISTRY_REPO:-intel}/rag-service:${RELEASE_TAG:-2026.3.0-rc1}" \
             "${REGISTRY_REPO:-intel}/kiosk-core:${RELEASE_TAG:-2026.3.0-rc1}" \
             "${REGISTRY_REPO:-intel}/audio-analyzer:${RELEASE_TAG:-2026.3.0-rc1}" \
             alpine:latest busybox:latest alpine busybox; do
        docker image inspect "$I" >/dev/null 2>&1 || continue
        TOOL=$(docker run --rm --entrypoint sh "$I" -c \
            'if command -v curl >/dev/null 2>&1; then echo curl
             elif command -v wget >/dev/null 2>&1; then echo wget
             else echo none; fi' 2>/dev/null | tr -d '\r\n')
        # Any image with a shell can still inspect the cache volumes.
        [ -n "$PROBE_IMG" ] || PROBE_IMG="$I"
        case "$TOOL" in
            curl) PROBE_IMG="$I"; PROBE_TOOL=curl; break ;;
            wget) [ -n "$PROBE_TOOL" ] || { PROBE_IMG="$I"; PROBE_TOOL=wget; } ;;
        esac
    done
}

# Count cache volumes that hold real, complete weight files.
#
# `find -L` follows the snapshot symlinks into blobs/, so an interrupted
# download (a dangling symlink, or a *.incomplete blob) is skipped rather than
# counted. The size floor rejects an empty models--*/ directory and truncated
# files: a real ASR/TTS/LLM weight file is tens of MB at minimum, so 1 MiB is
# far below any true positive while excluding configs and tokenizer JSON.
#
# The IR volumes are checked too, not just the HF source caches. Cached source
# weights are NOT sufficient for an offline start: audio-analyzer converts the
# model to OpenVINO IR on first run, and `optimum-cli export openvino` must
# reach the Hub API to infer the model's task. Offline, that export dies with
# "Hugging Face Hub is not reachable and we cannot infer the task from a
# cached model", the IR never appears, and the service crash-loops. Offline
# mode is only safe once the IR already exists.
cache_warm=0
cache_checked=0
count_warm_caches() {
    cache_warm=0
    cache_checked=0
    [ -n "$PROBE_IMG" ] || return 0
    for V in $CACHE_VOLUMES; do
        VOL="${COMPOSE_PROJECT}_${V}"
        docker volume inspect "$VOL" >/dev/null 2>&1 || continue
        cache_checked=$((cache_checked + 1))
        if docker run --rm -v "$VOL":/c:ro "$PROBE_IMG" \
            sh -c 'find -L /c -maxdepth 8 -type f \
                     \( -name "*.safetensors" -o -name "*.bin" -o -name "*.onnx" \
                        -o -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) \
                     ! -name "*.incomplete" -size +1024k -print -quit 2>/dev/null | grep -q .' \
            >/dev/null 2>&1; then
            cache_warm=$((cache_warm + 1))
        fi
    done
}

# An explicit offline request always wins -- forcing offline is a deliberate
# choice and this script never overrides it. In a dry run, still report
# whether the caches can actually satisfy that request.
if [ "$HF_HUB_OFFLINE" = "1" ]; then
    if dry_run; then
        select_probe_image
        count_warm_caches
        if [ "$cache_warm" -ge 3 ]; then
            say "  ✓ HF_HUB_OFFLINE=1 and all 3 model caches hold real weight files"
        else
            say "  ⚠ HF_HUB_OFFLINE=1 but only ${cache_warm}/3 model caches hold real weight files"
            say "    setup_models.sh does NOT pre-fetch the ASR/TTS/RAG models -- they are"
            say "    downloaded in-container on the first online run. Starting offline with"
            say "    a cold cache makes those services fail their healthcheck, and every"
            say "    service gated on them aborts with 'dependency failed to start'."
            say "    On a first run, set HF_HUB_OFFLINE=0 and TRANSFORMERS_OFFLINE=0 in .env."
        fi
    fi
    printf '1\n'
    exit 0
fi

if [ "$HF_AUTO_OFFLINE" = "false" ] || [ "$HF_AUTO_OFFLINE" = "FALSE" ]; then
    dry_run && say "  ⚠ HF_AUTO_OFFLINE=false — egress is not probed and startup is never blocked"
    printf '%s\n' "$HF_HUB_OFFLINE"
    exit 0
fi

select_probe_image

# Ask for the API endpoint huggingface_hub itself calls, not just the landing
# page, so a proxy that serves the site but blocks the API is caught.
HF_PROBE_URL="https://huggingface.co/api/models/openai/whisper-base"

# Exit 42 = "the tool was rejected, not the network" -> inconclusive.
PROBE_CMD='if command -v curl >/dev/null 2>&1; then
    curl -fsS --connect-timeout 5 --max-time 15 -o /dev/null "$1" 2>&1
else
    out=$(wget -q -T 10 --spider "$1" 2>&1) || {
        case "$out" in
            *403*|*Forbidden*) echo "$out"; exit 42 ;;
        esac
        echo "$out"; exit 1
    }
fi'

net_ok=""
probe_reason=""
probe_inconclusive=""
if [ -n "$PROBE_IMG" ] && [ -n "$PROBE_TOOL" ]; then
    # Pass the proxy through exactly as docker-compose.yml does, so the probe
    # tests the same path the services will take.
    probe_out=$(docker run --rm --entrypoint sh \
        ${HTTP_PROXY:+-e HTTP_PROXY="$HTTP_PROXY"} \
        ${http_proxy:+-e http_proxy="$http_proxy"} \
        ${HTTPS_PROXY:+-e HTTPS_PROXY="$HTTPS_PROXY"} \
        ${https_proxy:+-e https_proxy="$https_proxy"} \
        ${NO_PROXY:+-e NO_PROXY="$NO_PROXY"} \
        ${no_proxy:+-e no_proxy="$no_proxy"} \
        "$PROBE_IMG" -c "$PROBE_CMD" probe "$HF_PROBE_URL" 2>&1)
    probe_rc=$?
    probe_reason=$(printf '%s' "$probe_out" | tr '\n' ' ' | sed 's/  */ /g')

    if [ "$probe_rc" -eq 0 ]; then
        net_ok=1
    elif [ "$probe_rc" -eq 42 ]; then
        # Only wget was available and the Hub's edge refused it. That says
        # nothing about reachability, and guessing wrong here is expensive in
        # both directions, so defer rather than decide.
        probe_inconclusive=1
        probe_reason="only busybox wget was available to probe with and huggingface.co refused it (${probe_reason}); this does not mean the Hub is unreachable"
    else
        case "$probe_reason" in
            *"407"*|*"Proxy Authentication"*)
                probe_reason="the proxy demanded authentication (${probe_reason})" ;;
            "")
                probe_reason="no response (connection timed out or DNS failed)" ;;
        esac
    fi
elif curl -fsS --connect-timeout 3 --max-time 6 -o /dev/null "https://huggingface.co/api/models/openai/whisper-base" 2>/dev/null; then
    # --fail: without it curl exits 0 on an HTTP 403 from a filtering proxy,
    # which would select online mode for a Hub we cannot actually read.
    net_ok=1
    probe_inconclusive=1
else
    probe_reason="probed from the host, not a container (no local image to probe with)"
    probe_inconclusive=1
fi

if [ -n "$net_ok" ]; then
    if dry_run; then
        if [ -n "$probe_inconclusive" ]; then
            say "  ✓ huggingface.co is reachable (probed from the host, not a container —"
            say "    no local image to probe with, so containers may still differ)"
        else
            say "  ✓ huggingface.co is reachable from a container (HF_HUB_OFFLINE=0)"
        fi
    fi
    printf '0\n'
    exit 0
fi

# An inconclusive probe must not drive ANY decision -- neither blocking
# startup nor silently switching to offline. Forcing offline on a host that is
# actually online is not a safe default: `optimum-cli export openvino` needs
# the Hub API to infer a model's task, so the first-run IR export fails and
# the service crash-loops. Leave the caller's value alone and say why.
if [ -n "$probe_inconclusive" ] || [ -z "$PROBE_IMG" ] || [ -z "$PROBE_TOOL" ]; then
    if dry_run; then
        say "  ⚠ Could not determine whether huggingface.co is reachable"
        say "    ${probe_reason:-no container image with curl or wget to probe with}"
        say "    This is not a verdict and does not block startup. Pull a curl-capable"
        say "    image for an accurate probe, or set HF_HUB_OFFLINE explicitly in .env."
        exit 0
    fi
    say ""
    say "[hf] Could not reliably determine whether huggingface.co is reachable:"
    say "[hf]   ${probe_reason:-no container image with curl or wget to probe with}"
    say "[hf] Continuing with HF_HUB_OFFLINE=${HF_HUB_OFFLINE} rather than"
    say "[hf] guessing. Pull a curl-capable image for an accurate probe, or set"
    say "[hf] HF_HUB_OFFLINE explicitly in .env."
    say ""
    printf '%s\n' "$HF_HUB_OFFLINE"
    exit 0
fi

# No egress confirmed. Decide between auto-offline and fail-fast on cache
# contents -- but only if we were able to look properly.
count_warm_caches

if [ "$cache_warm" -ge 3 ]; then
    if dry_run; then
        say "  ⚠ huggingface.co is unreachable from a container, but all 3 model caches"
        say "    hold real weight files — 'make up' will start with HF_HUB_OFFLINE=1"
        say "    Probe result: ${probe_reason}"
        exit 0
    fi
    say ""
    say "[hf] huggingface.co is unreachable from a container:"
    say "[hf]   ${probe_reason}"
    say "[hf] The converted model caches hold real weight files, so starting with"
    say "[hf] HF_HUB_OFFLINE=1 -- the services load from cache instead of"
    say "[hf] stalling on a live Hub call."
    say "[hf] Set HF_AUTO_OFFLINE=false to disable this."
    say ""
    printf '1\n'
    exit 0
fi

if dry_run; then
    say "  ✗ huggingface.co is unreachable from a container and the model caches"
    if [ "$cache_checked" -eq 0 ]; then
        say "    do not exist yet — 'make up' will refuse to start"
    else
        say "    are cold (${cache_warm}/${cache_checked} hold real weights) — 'make up' will refuse to start"
    fi
    say "    Probe result: ${probe_reason}"
    say "    Set HTTP_PROXY/HTTPS_PROXY to an address routable from inside a"
    say "    container, or warm the caches on a connected host. 'make up' prints"
    say "    the full remediation. See docs/user-guide/troubleshooting.md."
    exit 0
fi

say ""
say "✗ Cannot start: huggingface.co is unreachable from a container and the"
if [ "$cache_checked" -eq 0 ]; then
    say "  model caches do not exist yet (nothing has been downloaded on this host)."
else
    say "  model caches are not usable: only ${cache_warm} of ${cache_checked} hold real weight files."
fi
say ""
say "  Probe result: ${probe_reason}"
say ""
say "  audio-analyzer and text-to-speech download their models in-container on"
say "  first run. Without egress they raise httpx.ConnectError during startup,"
say "  never become healthy, and every dependent service aborts after the full"
say "  600s timeout. Refusing to start rather than stall (ITEP-97604)."
say ""
say "  Fix one of these, then re-run:"
say ""
say "    1. Proxy -- export it (both cases are mirrored for you by make):"
say "         export HTTP_PROXY=http://proxy.example.com:3128"
say "         export HTTPS_PROXY=http://proxy.example.com:3128"
say "       or put HTTP_PROXY/HTTPS_PROXY in .env, or configure"
say "       proxies.default in ~/.docker/config.json."
say "       Use an address routable from inside a container -- not 127.0.0.1."
say ""
say "    2. Warm the cache on a connected host, copy the volumes over, then"
say "       set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 in .env."
say ""
say "    3. Override deliberately (it will likely stall):"
say "         make up HF_AUTO_OFFLINE=false"
say ""
exit 3
