#!/bin/bash
# setup_models.sh — Pre-download all models required by Smart Kiosk Assistant
#
# What this script downloads:
#   1. OVMS LLM  — OpenVINO/Qwen3-4B-int4-ov  (pre-converted INT4, ~2 GB)
#                  Served by ovms-llm for the ordering agent (tool calling).
#                  Uses OVMS pull mode — no local export/quantisation needed.
#
# Models downloaded inside their containers on first run (no setup needed):
#   • audio-analyzer  — Whisper (ASR) + Pyannote (speaker diarization)
#   • text-to-speech  — SpeechT5 / Qwen-TTS
#   • rag-service     — Qwen3-4B-Instruct-2507 (INT8, OpenVINO export) +
#                       BAAI/bge-large-en-v1.5 (embeddings) +
#                       BAAI/bge-reranker-base  (reranker)
#
# Usage:
#   ./setup_models.sh [OPTIONS]
#
# Options:
#   --device  CPU|GPU|NPU   Target inference device for OVMS  (default: GPU)
#   --skip-ovms             Skip the OVMS LLM model download
#   --hf-token  TOKEN       HuggingFace token (or export HF_TOKEN env var)
#
# Examples:
#   ./setup_models.sh                        # GPU (default)
#   ./setup_models.sh --device CPU           # CPU-only systems
#   ./setup_models.sh --hf-token hf_xxx...   # explicit HF token

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
VENV_DIR="${SCRIPT_DIR}/.setup-venv"
ENV_FILE="${SCRIPT_DIR}/.env"

# ── Model registry ────────────────────────────────────────────────────────────
# Best choice per https://docs.openvino.ai/2026/model-server/ovms_demos_continuous_batching_agent.html
# Pre-converted INT4 model with native tool calling support via hermes3 parser.
OVMS_SOURCE_MODEL="OpenVINO/Qwen3-4B-int4-ov"
OVMS_MODEL_NAME="OpenVINO/Qwen3-4B-int4-ov"

# ── Argument parsing ──────────────────────────────────────────────────────────
TARGET_DEVICE="GPU"
SKIP_OVMS=false
HF_TOKEN="${HF_TOKEN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            TARGET_DEVICE="${2^^}"   # uppercase
            shift 2
            ;;
        --skip-ovms)
            SKIP_OVMS=true
            shift
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --help|-h)
            sed -n '/^# Usage:/,/^$/p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

if [[ "${TARGET_DEVICE}" != "CPU" && "${TARGET_DEVICE}" != "GPU" && "${TARGET_DEVICE}" != "NPU" ]]; then
    echo "ERROR: --device must be CPU, GPU, or NPU (got: ${TARGET_DEVICE})"
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

print_section() {
    echo ""
    echo "------------------------------------------"
    echo "$1"
    echo "------------------------------------------"
}

# Check available RAM and warn if below threshold
check_memory() {
    local recommended_gb=16
    local total_gb=0
    if [ -f /proc/meminfo ]; then
        total_gb=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
    fi
    if [ "${total_gb}" -gt 0 ] && [ "${total_gb}" -lt "${recommended_gb}" ]; then
        echo ""
        echo "  ⚠  WARNING: System RAM is ${total_gb} GB (${recommended_gb} GB recommended)."
        echo ""
    fi
}

# Validate that a downloaded HuggingFace snapshot contains the expected files
check_ovms_model() {
    local model_path="$1"

    if [ ! -d "${model_path}" ]; then
        return 1
    fi

    # A valid OpenVINO INT4 model snapshot must contain:
    #   - openvino_model.xml  (IR graph definition)
    #   - openvino_model.bin  (IR weights)
    if ls "${model_path}"/openvino_model*.xml > /dev/null 2>&1 && \
       ls "${model_path}"/openvino_model*.bin > /dev/null 2>&1; then

        # Validate XML header is not corrupt
        local xml_file
        xml_file=$(ls "${model_path}"/openvino_model*.xml | head -1)
        local header
        header=$(head -c 5 "${xml_file}" 2>/dev/null || true)
        if [ "${header}" != "<?xml" ]; then
            echo "  ✗ openvino_model.xml appears corrupt — deleting and re-downloading"
            rm -rf "${model_path}"
            return 1
        fi

        echo "  ✓ Model found at ${model_path}"
        return 0
    fi

    echo "  ✗ Model not ready at ${model_path} (missing openvino_model.xml / .bin)"
    return 1
}

# ── Python venv for huggingface_hub download ─────────────────────────────────
_VENV_READY=0
ensure_venv() {
    if [ "${_VENV_READY}" -eq 1 ]; then return 0; fi

    print_section "Setting up Python environment"

    if [ ! -d "${VENV_DIR}" ] || [ ! -f "${VENV_DIR}/bin/pip" ]; then
        echo "  Creating virtual environment at ${VENV_DIR} ..."
        python3 -m venv "${VENV_DIR}" --clear
        echo "  ✓ Virtual environment created"
    else
        echo "  ✓ Virtual environment already exists"
    fi

    source "${VENV_DIR}/bin/activate"

    echo "  Installing huggingface_hub..."
    pip install -q --upgrade pip
    pip install -q "huggingface_hub>=0.23"
    echo "  ✓ huggingface_hub installed"

    _VENV_READY=1
}

# ── OVMS LLM model download ───────────────────────────────────────────────────
download_ovms_model() {
    print_section "OVMS LLM: ${OVMS_SOURCE_MODEL}"

    local target_path="${MODELS_DIR}/${OVMS_SOURCE_MODEL}"

    if check_ovms_model "${target_path}"; then
        echo "  ✓ Model already present — skipping download"
        return 0
    fi

    check_memory
    ensure_venv

    echo "  Downloading ${OVMS_SOURCE_MODEL} from HuggingFace..."
    echo "  Target: ${target_path}"
    echo "  (INT4 pre-converted OpenVINO model, ~2 GB)"
    echo ""

    local hf_token_arg=""
    if [ -n "${HF_TOKEN}" ]; then
        hf_token_arg="--token ${HF_TOKEN}"
    fi

    # Use huggingface_hub CLI for reliable snapshot download with progress
    "${VENV_DIR}/bin/huggingface-cli" download \
        "${OVMS_SOURCE_MODEL}" \
        --local-dir "${target_path}" \
        --local-dir-use-symlinks False \
        ${hf_token_arg}

    if check_ovms_model "${target_path}"; then
        echo ""
        echo "  ✓ ${OVMS_SOURCE_MODEL} downloaded successfully"
    else
        echo "  ✗ Download failed or model files are missing"
        echo "    Check ${target_path}"
        exit 1
    fi
}

# ── Update .env ───────────────────────────────────────────────────────────────
update_env() {
    print_section "Updating .env"

    # OVMS_MODEL_NAME
    if grep -q "^OVMS_MODEL_NAME=" "${ENV_FILE}" 2>/dev/null; then
        sed -i "s|^OVMS_MODEL_NAME=.*|OVMS_MODEL_NAME=${OVMS_MODEL_NAME}|" "${ENV_FILE}"
        echo "  ✓ OVMS_MODEL_NAME updated → ${OVMS_MODEL_NAME}"
    else
        echo "OVMS_MODEL_NAME=${OVMS_MODEL_NAME}" >> "${ENV_FILE}"
        echo "  ✓ OVMS_MODEL_NAME added  → ${OVMS_MODEL_NAME}"
    fi

    # TARGET_DEVICE (used by docker-compose for ovms-llm)
    if grep -q "^TARGET_DEVICE=" "${ENV_FILE}" 2>/dev/null; then
        sed -i "s|^TARGET_DEVICE=.*|TARGET_DEVICE=${TARGET_DEVICE}|" "${ENV_FILE}"
        echo "  ✓ TARGET_DEVICE updated  → ${TARGET_DEVICE}"
    else
        echo "TARGET_DEVICE=${TARGET_DEVICE}" >> "${ENV_FILE}"
        echo "  ✓ TARGET_DEVICE added    → ${TARGET_DEVICE}"
    fi

    # RENDER_GID — needed for GPU/NPU access inside containers
    if ! grep -q "^RENDER_GID=" "${ENV_FILE}" 2>/dev/null; then
        local render_gid
        render_gid=$(getent group render 2>/dev/null | cut -d: -f3 || echo "992")
        echo "RENDER_GID=${render_gid}" >> "${ENV_FILE}"
        echo "  ✓ RENDER_GID added       → ${render_gid}"
    else
        echo "  ✓ RENDER_GID already set"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
print_header "Smart Kiosk Assistant — Model Setup"
echo ""
echo "  Device  : ${TARGET_DEVICE}"
echo "  OVMS LLM: ${OVMS_SOURCE_MODEL}"
echo "  Models  : ${MODELS_DIR}"

mkdir -p "${MODELS_DIR}"

if [ "${SKIP_OVMS}" = false ]; then
    download_ovms_model
else
    echo ""
    echo "  ⚠  Skipping OVMS model download (--skip-ovms)"
fi

update_env

print_header "✓ Model Setup Complete!"
echo ""
echo "  Start the stack with:"
echo "    cd ${SCRIPT_DIR}"
echo "    docker compose up -d"
echo ""
echo "  Services:"
echo "    Gradio UI         → http://localhost:7860"
echo "    kiosk-core API    → http://localhost:8012"
echo "    rag-service API   → http://localhost:8020"
echo "    OVMS (LLM)        → http://localhost:8000"
echo "    metrics-collector → http://localhost:9000"
echo ""
