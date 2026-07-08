#!/bin/bash
# setup_models.sh — Pre-download all models required by Smart Kiosk Assistant
#
# What this script downloads:
#   1. OVMS LLM  — OpenVINO/Qwen3-4B-int8-ov  (pre-converted INT8, ~4 GB)
#                  Served by ovms-llm for the ordering agent (tool calling).
#                  INT8 chosen over INT4 for better instruction-following accuracy.
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
#   --int4                  Use INT4 model instead of INT8 (smaller, less accurate)
#   --skip-ovms             Skip the OVMS LLM model download
#   --identity              Also download identity-service models (face + voice)
#   --identity-only         Only download identity-service models (implies --skip-ovms)
#   --hf-token  TOKEN       HuggingFace token (or export HF_TOKEN env var)
#
# Examples:
#   ./setup_models.sh                        # GPU (default, INT8)
#   ./setup_models.sh --device CPU           # CPU-only systems
#   ./setup_models.sh --int4                 # INT4 (smaller, faster, less accurate)
#   ./setup_models.sh --identity             # LLM + identity (face detect/reid + ECAPA voice)
#   ./setup_models.sh --identity-only        # only identity-service models
#   ./setup_models.sh --hf-token hf_xxx...   # explicit HF token

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
VENV_DIR="${SCRIPT_DIR}/.setup-venv"
ENV_FILE="${SCRIPT_DIR}/.env"

# ── Model registry ────────────────────────────────────────────────────────────
# INT8 is the default: better instruction-following and tool-calling accuracy
# compared to INT4, at the cost of ~2x model size (~4 GB vs ~2 GB).
# Use --int4 flag to override.
OVMS_QUANT="int8"
OVMS_SOURCE_MODEL="OpenVINO/Qwen3-4B-int8-ov"
OVMS_MODEL_NAME="OpenVINO/Qwen3-4B-int8-ov"

# ── Identity-service models (face detection/reid + ECAPA voice) ───────────────
# Face IRs come straight from the Open Model Zoo storage; the ECAPA voice model
# is converted from SpeechBrain to an OpenVINO IR at setup time.
OMZ_BASE="https://storage.openvinotoolkit.org/repositories/open_model_zoo"
OMZ_VERSION="2023.0"
IDENTITY_PRECISION="FP16"
IDENTITY_FACE_MODELS=(
    "face-detection-retail-0005"
    "face-reidentification-retail-0095"
)
IDENTITY_VOICE_MODEL="ecapa-tdnn-voice"
IDENTITY_VOICE_SOURCE="speechbrain/spkrec-ecapa-voxceleb"

# ── Argument parsing ──────────────────────────────────────────────────────────
TARGET_DEVICE="GPU"
SKIP_OVMS=false
DO_IDENTITY=false
HF_TOKEN="${HF_TOKEN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            TARGET_DEVICE="${2^^}"   # uppercase
            shift 2
            ;;
        --int4)
            OVMS_QUANT="int4"
            OVMS_SOURCE_MODEL="OpenVINO/Qwen3-4B-int4-ov"
            OVMS_MODEL_NAME="OpenVINO/Qwen3-4B-int4-ov"
            shift
            ;;
        --skip-ovms)
            SKIP_OVMS=true
            shift
            ;;
        --identity)
            DO_IDENTITY=true
            shift
            ;;
        --identity-only)
            DO_IDENTITY=true
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
# AND the OVMS graph.pbtxt configuration required for serving.
#
# Why graph.pbtxt matters:
#   hf download fetches the OpenVINO model weights (openvino_model.xml/.bin) but
#   NOT graph.pbtxt. OVMS requires graph.pbtxt to serve the model; when the model
#   directory exists (non-empty) OVMS skips its own download/init and tries to read
#   graph.pbtxt directly — failing with "Unable to open file" if it's absent.
#   generate_graph_pbtxt() creates the correct graph.pbtxt based on TARGET_DEVICE.
check_ovms_model() {
    local model_path="$1"

    if [ ! -d "${model_path}" ]; then
        return 1
    fi

    # A valid OVMS-ready model directory must contain:
    #   - openvino_model.xml  (IR graph definition)
    #   - openvino_model.bin  (IR weights)
    #   - graph.pbtxt         (OVMS MediaPipe serving graph — NOT in the HF repo,
    #                          must be generated by generate_graph_pbtxt())
    if ls "${model_path}"/openvino_model*.xml > /dev/null 2>&1 && \
       ls "${model_path}"/openvino_model*.bin > /dev/null 2>&1 && \
       [ -f "${model_path}/graph.pbtxt" ]; then

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

    if ! ls "${model_path}"/openvino_model*.xml > /dev/null 2>&1 || \
       ! ls "${model_path}"/openvino_model*.bin > /dev/null 2>&1; then
        echo "  ✗ Model not ready at ${model_path} (missing openvino_model.xml / .bin)"
    else
        echo "  ✗ Model not ready at ${model_path} (missing graph.pbtxt)"
    fi
    return 1
}

# Generate the OVMS MediaPipe graph.pbtxt for the text_generation pipeline.
# OVMS requires this file in the model directory to serve the model.
# Content mirrors the official OVMS continuous_batching demo graphs:
#   CPU — demos/continuous_batching/graph.pbtxt
#   GPU — demos/continuous_batching/graph_gpu.pbtxt
generate_graph_pbtxt() {
    local model_path="$1"
    local device="${2:-CPU}"

    if [ -f "${model_path}/graph.pbtxt" ]; then
        echo "  ✓ graph.pbtxt already present — skipping generation"
        return 0
    fi

    echo "  Generating graph.pbtxt for device=${device}..."

    if [ "${device}" = "GPU" ]; then
        cat > "${model_path}/graph.pbtxt" << 'GRAPH_EOF'
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  node_options: {
      [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
          models_path: "./",
          plugin_config: '{}',
          dynamic_split_fuse: false,
          max_num_seqs: 256,
          max_num_batched_tokens:8192,
          cache_size: 0,
          device: "GPU"
      }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
}
GRAPH_EOF
    else
        # CPU (default) — uses u8 KV cache and dynamic quantization for memory efficiency
        cat > "${model_path}/graph.pbtxt" << 'GRAPH_EOF'
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  node_options: {
      [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
          models_path: "./",
          plugin_config: '{"KV_CACHE_PRECISION": "u8", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32"}',
          enable_prefix_caching: false
          cache_size: 0
      }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
}
GRAPH_EOF
    fi

    # Ensure OVMS (uid 5000) can read the graph config
    docker run --rm -v "${MODELS_DIR}:/models" alpine \
        sh -c "chmod 644 /models/${OVMS_SOURCE_MODEL}/graph.pbtxt" 2>/dev/null || \
        chmod 644 "${model_path}/graph.pbtxt" 2>/dev/null || true

    echo "  ✓ graph.pbtxt generated (device=${device})"
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

    echo "  Installing huggingface_hub (hf CLI)..."
    pip install -q --upgrade pip
    pip install -q "huggingface_hub[cli]>=0.23"
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

    # Weights may already be present (e.g. restored from CI cache) but
    # graph.pbtxt missing — avoid a full 2 GB re-download in that case.
    if ls "${target_path}"/openvino_model*.xml > /dev/null 2>&1 && \
       ls "${target_path}"/openvino_model*.bin > /dev/null 2>&1; then
        echo "  ⚠ Model weights present but graph.pbtxt missing — generating without re-download"
        generate_graph_pbtxt "${target_path}" "${TARGET_DEVICE}"
        if check_ovms_model "${target_path}"; then
            echo "  ✓ Model configuration restored"
            return 0
        fi
        echo "  ✗ graph.pbtxt generation failed — falling through to full download"
    fi

    check_memory
    ensure_venv

    # Ensure the target directory is writable by the current user.
    # The models/ tree may be owned by uid 5000 (OVMS container user) from
    # a previous run; use a minimal Docker container to fix permissions without sudo.
    if [ -d "${MODELS_DIR}" ]; then
        docker run --rm \
          -v "${MODELS_DIR}:/models" \
          alpine sh -c "chmod -R 777 /models" 2>/dev/null || true
    fi
    mkdir -p "${target_path}"

    echo "  Downloading ${OVMS_SOURCE_MODEL} from HuggingFace..."
    echo "  Target: ${target_path}"
    echo "  (INT${OVMS_QUANT^^} pre-converted OpenVINO model, ~$([ "$OVMS_QUANT" = "int4" ] && echo "2 GB" || echo "4 GB"))"
    echo ""

    local hf_token_arg=""
    if [ -n "${HF_TOKEN}" ]; then
        hf_token_arg="--token ${HF_TOKEN}"
    fi

    # Use hf CLI for reliable snapshot download with progress
    # (huggingface-cli is deprecated; hf is the modern replacement)
    "${VENV_DIR}/bin/hf" download \
        "${OVMS_SOURCE_MODEL}" \
        --local-dir "${target_path}" \
        ${hf_token_arg}

    # graph.pbtxt is not in the HuggingFace repo — generate it after download.
    generate_graph_pbtxt "${target_path}" "${TARGET_DEVICE}"

    if check_ovms_model "${target_path}"; then
        echo ""
        echo "  ✓ ${OVMS_SOURCE_MODEL} downloaded successfully"
        # Ensure OVMS (uid 5000) can read all model files
        docker run --rm -v "${MODELS_DIR}:/models" alpine \
            sh -c "chown -R 5000:5000 /models/OpenVINO && chmod -R 755 /models/OpenVINO" 2>/dev/null || true
    else
        echo "  ✗ Download failed or model files are missing"
        echo "    Check ${target_path}"
        exit 1
    fi
}

# ── Identity-service models ───────────────────────────────────────────────────
download_face_model() {
    local name="$1"
    local dest="${MODELS_DIR}/identity/${name}/${IDENTITY_PRECISION}"
    local url_base="${OMZ_BASE}/${OMZ_VERSION}/models_bin/1/${name}/${IDENTITY_PRECISION}"

    if [ -f "${dest}/${name}.xml" ] && [ -f "${dest}/${name}.bin" ]; then
        echo "  ✓ ${name} (${IDENTITY_PRECISION}) already present — skipping"
        return 0
    fi

    mkdir -p "${dest}"
    echo "  Downloading ${name} (${IDENTITY_PRECISION})..."
    curl -fSL "${url_base}/${name}.xml" -o "${dest}/${name}.xml"
    curl -fSL "${url_base}/${name}.bin" -o "${dest}/${name}.bin"

    if [ -f "${dest}/${name}.xml" ] && [ -f "${dest}/${name}.bin" ]; then
        echo "  ✓ ${name} downloaded → ${dest}"
    else
        echo "  ✗ Failed to download ${name}"
        exit 1
    fi
}

convert_voice_model() {
    local out_dir="${MODELS_DIR}/identity/${IDENTITY_VOICE_MODEL}/openvino"

    if [ -f "${out_dir}/${IDENTITY_VOICE_MODEL}.xml" ] && \
       [ -f "${out_dir}/${IDENTITY_VOICE_MODEL}.bin" ]; then
        echo "  ✓ Voice IR (${IDENTITY_VOICE_MODEL}) already present — skipping"
        return 0
    fi

    ensure_venv
    echo "  Installing conversion deps (torch, speechbrain, openvino)..."
    pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -q "speechbrain>=1.0" "openvino>=2024.0"

    mkdir -p "${out_dir}"
    echo "  Converting ${IDENTITY_VOICE_SOURCE} → OpenVINO IR..."
    "${VENV_DIR}/bin/python" \
        "${SCRIPT_DIR}/identity-service/tools/convert_ecapa_to_openvino.py" \
        --output-dir "${out_dir}" \
        --model-name "${IDENTITY_VOICE_MODEL}" \
        --source "${IDENTITY_VOICE_SOURCE}"

    if [ -f "${out_dir}/${IDENTITY_VOICE_MODEL}.xml" ]; then
        echo "  ✓ Voice IR saved → ${out_dir}"
    else
        echo "  ✗ Voice model conversion failed"
        echo "    (Identity-service will still run, but voice verification stays disabled.)"
    fi
}

download_identity_models() {
    print_section "Identity-service models → ${MODELS_DIR}/identity"
    for model in "${IDENTITY_FACE_MODELS[@]}"; do
        download_face_model "${model}"
    done
    convert_voice_model

    # Ensure the identity-service app user (uid 1000) can read the models.
    docker run --rm -v "${MODELS_DIR}:/models" alpine \
        sh -c "chmod -R a+rX /models/identity" 2>/dev/null || true
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

if [ "${DO_IDENTITY}" = true ]; then
    download_identity_models
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
