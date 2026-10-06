# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Smart Kiosk Assistant - Makefile
# =============================================================================

.PHONY: help init-env setup-dirs setup-models download-sample-video check-env show-config \
		build up down restart status logs \
		logs-ovms logs-rag logs-core logs-ui logs-audio logs-tts logs-metrics \
		test clean clean-metrics dev-build update-submodule update-submodules \
		benchmark-v2v list-profiles validate-profile _write-provenance \
		benchmark-matrix compare-profiles

# =============================================================================
# Load .env (silent if missing — check-env will warn)
# =============================================================================
-include .env
export

# =============================================================================
# Benchmark device profile (optional)
# =============================================================================
# A "profile" is a plain env file under configs/benchmark-profiles/ describing
# ONE device combination (e.g. ASR=NPU, TTS=CPU, LLM=GPU). It is the only thing
# a benchmarker has to edit. See that directory's README.md for the valid
# engine/device matrix and why the matrix pins non-default models.
#
#   make benchmark PROFILE=asr-npu_tts-cpu
#
# Layering is base .env -> profile -> make/CLI override. The profile is
# included AFTER .env above (so it wins over .env) but make command-line
# variables still beat both, which is what `make benchmark PROFILE=x QUEUE=true`
# relies on.
#
# Every value here is empty/neutral when PROFILE is unset, so an ordinary
# `make up` resolves byte-identically to the pre-profile behaviour.
PROFILE ?=
ifneq ($(PROFILE),)
_PROFILE_FILE := configs/benchmark-profiles/$(PROFILE).env
ifeq ($(wildcard $(_PROFILE_FILE)),)
$(error Unknown PROFILE '$(PROFILE)' — no $(_PROFILE_FILE). Run 'make list-profiles')
endif
include $(_PROFILE_FILE)
export
endif

# Per-profile output directories. With PROFILE unset these collapse to the
# historical "results"/"metrics", so nothing downstream changes.
#
# Why separate directories per profile rather than one shared results/:
# consolidate_multiple_run_of_metrics.py matches its KPI extractors by filename
# SUBSTRING, so two profiles' cpu_usage.log/qmassa*.json sitting side by side
# would be averaged into a single meaningless row. One leaf directory per
# profile keeps every run independently consolidatable.
RESULTS_ROOT ?= results
METRICS_ROOT ?= metrics
_PROFILE_SUB  = $(if $(PROFILE),/$(PROFILE),)
RESULTS_DIR   = $(RESULTS_ROOT)$(_PROFILE_SUB)
METRICS_DIR   = $(METRICS_ROOT)$(_PROFILE_SUB)

# Consumed by docker-compose.yml's bind mounts (kiosk-core /app/results and
# metrics-collector /tmp/results). Defaults there are ./results and ./metrics,
# so these only ever redirect when PROFILE is set.
export KIOSK_RESULTS_HOST_DIR := ./$(RESULTS_DIR)
export KIOSK_METRICS_HOST_DIR := ./$(METRICS_DIR)

# =============================================================================
# Configuration variables (defaults match .env)
# =============================================================================

COMPOSE_FILE     ?= docker-compose.yml

# REGISTRY=false  → build images locally from source
# REGISTRY=true   → pull pre-built images from the registry
#
# NOTE: `-include .env` above runs BEFORE this assignment, and the shipped
#       .env sets REGISTRY=intel. Because `?=` does not override an
#       already-set variable, the effective default when a .env is present is
#       PULL, not build. Pass REGISTRY=false explicitly to build locally:
#           make build REGISTRY=false
#
# NOTE: "false" and "true" are build-mode flags only. docker-compose always
#       receives the real registry name (REGISTRY=intel) so image tags are never
#       "false/<svc>:..." or "true/<svc>:...". Passing an explicit registry name
#       (e.g. REGISTRY=myrepo.example.com/team) also selects pull mode and is
#       forwarded to docker-compose verbatim — that registry must mirror every
#       image, including the upstream audio-analyzer, text-to-speech, and
#       hl-ai-metrics-collector, since `make up` resolves them the same way.
REGISTRY         ?= false

# _DOCKER_REGISTRY: the actual registry prefix passed to docker-compose.
# When the user passes REGISTRY=false or REGISTRY=true it is treated as a
# build-mode switch; the real registry name comes from .env (default: intel).
_ENV_REGISTRY    ?= intel
_DOCKER_REGISTRY  = $(if $(filter false FALSE true TRUE,$(REGISTRY)),$(_ENV_REGISTRY),$(REGISTRY))

RELEASE_TAG      ?= 2026.1.0
export TAG       := $(RELEASE_TAG)

TARGET_DEVICE    ?= GPU
OVMS_MODEL_NAME  ?= OpenVINO/Qwen3-4B-int8-ov

# Sample videos for the queue-service RTSP feed. Directory and source URLs are
# sourced from .env (see .env.example). Files are saved as sample_1.mp4 /
# sample_2.mp4 to match docker-compose (MEDIA_FILES).
SAMPLE_DATA_DIR    ?= ./Sample_data
SAMPLE_VIDEO_1_URL ?=
SAMPLE_VIDEO_2_URL ?=

# RTSP source mode and camera mount resolution.
# Compose cannot conditionally include devices, so we resolve one effective
# host device path here and pass it as RTSP_CAMERA_DEVICE:
#   sample -> /dev/null (safe no-op mapping)
#   live   -> CAMERA_DEVICE (must be set and valid)
# Strip whitespace from both to tolerate trailing spaces in .env.
_ENV_RTSP_SOURCE_MODE := $(RTSP_SOURCE_MODE)
RTSP_SOURCE_MODE      ?= $(_ENV_RTSP_SOURCE_MODE)
_RTSP_SOURCE_MODE_RAW  = $(strip $(RTSP_SOURCE_MODE))
_RTSP_SOURCE_MODE      = $(if $(_RTSP_SOURCE_MODE_RAW),$(_RTSP_SOURCE_MODE_RAW),sample)
_RTSP_LIVE_ON          = $(filter live,$(_RTSP_SOURCE_MODE))
_CAMERA_DEVICE_RAW     = $(strip $(CAMERA_DEVICE))
_RTSP_CAMERA_DEVICE    = $(if $(_RTSP_LIVE_ON),$(_CAMERA_DEVICE_RAW),/dev/null)
COMPOSE_BASE           = RTSP_SOURCE_MODE=$(_RTSP_SOURCE_MODE) RTSP_CAMERA_DEVICE=$(_RTSP_CAMERA_DEVICE) docker compose -f $(COMPOSE_FILE)

# Services with fully local build contexts (built from source in this repo).
# metrics-collector is NOT in this list: its source now lives in the
# edge-ai-suites repo (multi_modal_patient_monitoring/services/metrics-
# collector) and it is pulled as intel/hl-ai-metrics-collector, like the EAL
# images below.
LOCAL_ONLY_SERVICES := rag-service kiosk-core kiosk-ui rtsp-streamer queue-service

# Identity feature (profile-gated). Driven by KIOSK_CORE_IDENTITY_ENABLED in
# .env by default — set it to true there to build/start the identity-service
# container AND tell kiosk-core to call it. `IDENTITY=true|false` on the make
# command line still takes precedence over .env for one-off overrides.
# Default off so the standard stack is unchanged.
# NOTE: capture the .env value with ":=" (immediate expansion) BEFORE the
# "export KIOSK_CORE_IDENTITY_ENABLED = ..." line below rebinds that same
# name — otherwise IDENTITY and KIOSK_CORE_IDENTITY_ENABLED would reference
# each other recursively and make would abort with "Recursive variable
# 'IDENTITY' references itself".
_ENV_IDENTITY_ENABLED := $(KIOSK_CORE_IDENTITY_ENABLED)
IDENTITY          ?= $(_ENV_IDENTITY_ENABLED)
_IDENTITY_ON       = $(filter true TRUE,$(IDENTITY))
_IDENTITY_PROFILE  = $(if $(_IDENTITY_ON),--profile identity,)
_IDENTITY_SVC      = $(if $(_IDENTITY_ON),identity-service,)
export KIOSK_CORE_IDENTITY_ENABLED = $(if $(_IDENTITY_ON),true,false)

# Queue/RTSP feature (profile-gated, ON by default — unlike identity above).
# rtsp-streamer + queue-service (YOLOv8 person-counting) are CPU-hungry
# (measured ~650-750% + ~350-500% CPU continuously, whether or not a voice
# session is active) and share the host's CPU with the voice pipeline's
# only CPU-bound stage, TTS. QUEUE=false frees those cores for a run,
# useful for isolating/benchmarking voice-pipeline latency without them.
# Reuses KIOSK_CORE_QUEUE_SERVICE_ENABLED (.env) — the same flag kiosk-core
# already uses at the app layer to decide whether to call queue-service —
# so one flag now controls both "is the container even running" (this
# Makefile / compose profile) and "does kiosk-core call it" (config.py).
# `QUEUE=true|false` on the make command line overrides it for one-off runs.
_ENV_QUEUE_ENABLED := $(KIOSK_CORE_QUEUE_SERVICE_ENABLED)
QUEUE          ?= $(if $(_ENV_QUEUE_ENABLED),$(_ENV_QUEUE_ENABLED),true)
_QUEUE_ON       = $(filter true TRUE,$(QUEUE))
_QUEUE_PROFILE  = $(if $(_QUEUE_ON),--profile queue,)
export KIOSK_CORE_QUEUE_SERVICE_ENABLED = $(if $(_QUEUE_ON),true,false)

# audio-analyzer and text-to-speech are published to Docker Hub by the
# edge-ai-libraries CI pipeline and are pulled by default. With
# REGISTRY=false they are instead built locally from the sibling
# ../edge-ai-libraries/microservices/{audio-analyzer,text-to-speech}
# checkout (see their build.context in docker-compose.yml). In pull mode,
# text-to-speech is still always built locally because the registry image
# lacks Kokoro TTS support (see docker-compose.yml text-to-speech comment).

# Colors
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
NC     := \033[0m

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)Smart Kiosk Assistant — Available Commands$(NC)"
	@echo "==========================================="
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make init-env          Create .env from .env.example"
	@echo "  make setup-dirs        Create required runtime directories"
	@echo "  make setup-models      Download OVMS LLM model (Qwen3-4B) via setup_models.sh"
	@echo "  make download-sample-video  Download sample_1/sample_2 videos into SAMPLE_DATA_DIR"
	@echo "  make check-env         Validate .env and required variables"
	@echo "  make show-config       Print resolved configuration"
	@echo ""
	@echo "$(YELLOW)Core Commands:$(NC)"
	@echo "  make build             Pull images (default); REGISTRY=false builds locally"
	@echo "  make up                Start all services"
	@echo "  make down              Stop all services"
	@echo "  make restart           Stop then start all services"
	@echo "  make status            Show container status"
	@echo "  make test              Run health checks on all service endpoints"
	@echo ""
	@echo "$(YELLOW)Logs:$(NC)"
	@echo "  make logs              Tail logs for all services"
	@echo "  make logs-ovms         Tail ovms-llm logs"
	@echo "  make logs-rag          Tail rag-service logs"
	@echo "  make logs-core         Tail kiosk-core logs"
	@echo "  make logs-ui           Tail kiosk-ui logs"
	@echo "  make logs-audio        Tail audio-analyzer logs"
	@echo "  make logs-tts          Tail text-to-speech logs"
	@echo "  make logs-metrics      Tail metrics-collector logs"
	@echo ""
	@echo "$(YELLOW)Benchmark:$(NC)"
	@echo "  make benchmark            Run the voice-to-voice latency benchmark (brings stack up/down itself)"
	@echo "  make consolidate-metrics  Consolidate ./metrics + ./results into results/consolidated_metrics.csv"
	@echo "  make plot-metrics         Generate results/plot_metrics.png (CPU/GPU/NPU/memory usage)"
	@echo ""
	@echo "$(YELLOW)Device-combination benchmarking:$(NC)"
	@echo "  make list-profiles        List available device profiles (ASR/TTS/LLM combinations)"
	@echo "  make validate-profile     Pre-flight check a profile (PROFILE=<name>), no bring-up"
	@echo "  make benchmark-matrix     Run every profile sequentially, then compare"
	@echo "  make compare-profiles     Rebuild results/matrix_summary.csv from existing runs"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make dev-build         Build a single service: make dev-build SVC=rag-service"
	@echo "  make clean             Stop containers and remove named volumes"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make init-env                     # first-time setup"
	@echo "  make setup-models                 # download LLM model"
	@echo "  make build REGISTRY=false         # build all images locally"
	@echo "  make up                           # start the stack"
	@echo "  make build SVC=kiosk-ui           # rebuild a single service"
	@echo "  make setup-models TARGET_DEVICE=CPU  # CPU-only setup"
	@echo "  make benchmark && make consolidate-metrics && make plot-metrics  # full benchmark + report flow"
	@echo "  make benchmark PROFILE=asr-npu_tts-cpu  # benchmark one device combination"
	@echo "  make benchmark-matrix                   # benchmark every combination, then compare"
	@echo ""
	@echo "$(YELLOW)Current settings:$(NC)"
	@echo "  REGISTRY=$(REGISTRY)  TARGET_DEVICE=$(TARGET_DEVICE)  OVMS_MODEL_NAME=$(OVMS_MODEL_NAME)"
	@echo "  PROFILE=$(if $(PROFILE),$(PROFILE),(none))  RESULTS_DIR=$(RESULTS_DIR)  METRICS_DIR=$(METRICS_DIR)"
	@echo ""


# =============================================================================
# Setup
# =============================================================================

init-env: ## Create .env from .env.example (backs up existing .env)
	@if [ -f .env ]; then \
		echo "$(YELLOW).env already exists. Back up and overwrite? [y/N]$(NC)"; \
		read -r answer; \
		if [ "$$answer" = "y" ] || [ "$$answer" = "Y" ]; then \
			cp .env .env.backup; \
			echo "$(GREEN)  Backed up to .env.backup$(NC)"; \
			cp .env.example .env; \
			echo "$(GREEN)✓ Created .env from .env.example$(NC)"; \
		else \
			echo "$(YELLOW)  Skipped$(NC)"; \
		fi; \
	else \
		cp .env.example .env; \
		echo "$(GREEN)✓ Created .env from .env.example$(NC)"; \
	fi
	@echo "  Edit .env and set HF_TOKEN before running 'make setup-models' or 'make up'."

setup-dirs: ## Create required runtime directories
	@echo "$(BLUE)Creating required directories...$(NC)"
	@# results/ is bind-mounted into kiosk-core (CONTAINER_RESULTS_PATH, see
	@# docker-compose.yml) for kiosk_core.vlm_metrics_logger's output -- must
	@# exist with the host user's ownership before `up`, otherwise Docker
	@# auto-creates it as root and kiosk-core's user:"1000:1000" can't write.
	@mkdir -p models storage conversations $(METRICS_DIR) $(RESULTS_DIR)
	@# ...but ownership alone is not enough. The containers run as a fixed
	@# uid 1000, while the host user who cloned the repo may be any uid. When
	@# they differ the metrics writes fail silently and the benchmark reports
	@# zero transactions even though every turn succeeded. Widen the mode so
	@# uid 1000 can write regardless of who owns the directory.
	@for d in $(RESULTS_DIR) $(METRICS_DIR); do \
		owner=$$(stat -c '%u' "$$d" 2>/dev/null || echo 1000); \
		if [ "$$owner" != "1000" ]; then \
			chmod a+rwX "$$d" 2>/dev/null \
				&& echo "  $(YELLOW)$$d is owned by uid $$owner, containers run as 1000 — widened permissions$(NC)" \
				|| echo "  $(RED)✗ $$d is owned by uid $$owner and could not be made writable for uid 1000; benchmark metrics will be lost$(NC)"; \
		fi; \
	done
	@echo "$(GREEN)✓ Directories ready$(NC)"

setup-models: ## Download the OVMS LLM model (Qwen3-4B) via setup_models.sh
	@echo "$(BLUE)Downloading OVMS LLM model for TARGET_DEVICE=$(TARGET_DEVICE)...$(NC)"
	@if [ "$(TARGET_DEVICE)" = "CPU" ]; then \
		./setup_models.sh --device CPU; \
	elif [ "$(OVMS_MODEL_NAME)" = "OpenVINO/Qwen3-4B-int4-ov" ]; then \
		./setup_models.sh --device $(TARGET_DEVICE) --int4; \
	else \
		./setup_models.sh --device $(TARGET_DEVICE); \
	fi
	@echo "$(GREEN)✓ Model download complete$(NC)"

download-sample-video: setup-dirs ## Download sample videos (sample_1, sample_2) into SAMPLE_DATA_DIR (URLs from .env)
	@echo "$(BLUE)Downloading sample videos into $(SAMPLE_DATA_DIR)...$(NC)"
	@mkdir -p "$(SAMPLE_DATA_DIR)"
	@_download() { \
		name=$$1; url=$$2; dest=$$3; \
		if [ -z "$$url" ]; then \
			echo "  $(RED)✗ $$name URL is not set$(NC)  →  add it to .env"; \
			return 1; \
		fi; \
		if [ -f "$$dest" ]; then \
			echo "  $(GREEN)✓ $$name already exists: $$dest$(NC)"; \
			return 0; \
		fi; \
		echo "  URL : $$url"; \
		echo "  Dest: $$dest"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -L --progress-bar -o "$$dest" "$$url" \
				&& echo "  $(GREEN)✓ Download complete: $$dest$(NC)" \
				|| { echo "  $(RED)✗ Download failed for $$name. Check the URL in .env or add the video manually.$(NC)"; rm -f "$$dest"; return 1; }; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -q --show-progress -O "$$dest" "$$url" \
				&& echo "  $(GREEN)✓ Download complete: $$dest$(NC)" \
				|| { echo "  $(RED)✗ Download failed for $$name. Check the URL in .env or add the video manually.$(NC)"; rm -f "$$dest"; return 1; }; \
		else \
			echo "  $(RED)✗ Neither curl nor wget found. Install one or copy the video manually to $$dest$(NC)"; \
			return 1; \
		fi; \
	}; \
	_download "sample_1" "$(SAMPLE_VIDEO_1_URL)" "$(SAMPLE_DATA_DIR)/sample_1.mp4"; _rc1=$$?; \
	_download "sample_2" "$(SAMPLE_VIDEO_2_URL)" "$(SAMPLE_DATA_DIR)/sample_2.mp4"; _rc2=$$?; \
	if [ $$_rc1 -ne 0 ] || [ $$_rc2 -ne 0 ]; then \
		echo "$(RED)✗ One or more sample videos failed to download.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Sample videos ready in $(SAMPLE_DATA_DIR)$(NC)"

check-env: ## Validate .env and all required variables before running the stack
	@echo "$(BLUE)Validating environment...$(NC)"
	@ERRORS=0; \
	\
	if [ ! -f .env ]; then \
		echo "  $(RED)✗ .env not found$(NC)  →  run 'make init-env' first"; \
		ERRORS=$$((ERRORS+1)); \
	else \
		echo "  $(GREEN)✓ .env present$(NC)"; \
	fi; \
	\
	if [ -z "$(HF_TOKEN)" ]; then \
		if [ "$(KIOSK_CORE_DIARIZATION_ENABLED)" = "false" ]; then \
			echo "  $(YELLOW)⚠ HF_TOKEN is not set$(NC) (OK — diarization is disabled)"; \
		else \
			echo "  $(RED)✗ HF_TOKEN is not set$(NC)"; \
			echo "    Required for speaker diarization (pyannote/segmentation-3.0)."; \
			echo "    Set it in .env  →  https://huggingface.co/settings/tokens"; \
			echo "    Or add KIOSK_CORE_DIARIZATION_ENABLED=false to .env to skip diarization."; \
			ERRORS=$$((ERRORS+1)); \
		fi; \
	else \
		echo "  $(GREEN)✓ HF_TOKEN is set$(NC)"; \
	fi; \
	\
	if [ "$(TARGET_DEVICE)" != "CPU" ] && [ "$(TARGET_DEVICE)" != "GPU" ] && [ "$(TARGET_DEVICE)" != "NPU" ]; then \
		echo "  $(RED)✗ TARGET_DEVICE='$(TARGET_DEVICE)' is invalid$(NC)  (must be CPU | GPU | NPU)"; \
		ERRORS=$$((ERRORS+1)); \
	else \
		echo "  $(GREEN)✓ TARGET_DEVICE=$(TARGET_DEVICE)$(NC)"; \
	fi; \
	\
	if [ "$(_RTSP_SOURCE_MODE_RAW)" != "" ] && [ "$(_RTSP_SOURCE_MODE_RAW)" != "sample" ] && [ "$(_RTSP_SOURCE_MODE_RAW)" != "live" ]; then \
		echo "  $(RED)✗ RTSP_SOURCE_MODE='$(_RTSP_SOURCE_MODE_RAW)' is invalid$(NC)  (must be sample | live)"; \
		ERRORS=$$((ERRORS+1)); \
	else \
		echo "  $(GREEN)✓ RTSP_SOURCE_MODE=$(_RTSP_SOURCE_MODE)$(NC)"; \
	fi; \
	\
	if [ "$(_RTSP_SOURCE_MODE)" = "live" ]; then \
		if [ -z "$(_CAMERA_DEVICE_RAW)" ]; then \
			echo "  $(RED)✗ CAMERA_DEVICE is not set$(NC)  (required when RTSP_SOURCE_MODE=live)"; \
			ERRORS=$$((ERRORS+1)); \
		elif [ ! -e "$(_CAMERA_DEVICE_RAW)" ]; then \
			echo "  $(RED)✗ CAMERA_DEVICE='$(_CAMERA_DEVICE_RAW)' does not exist$(NC)"; \
			ERRORS=$$((ERRORS+1)); \
		elif [ ! -r "$(_CAMERA_DEVICE_RAW)" ]; then \
			echo "  $(RED)✗ CAMERA_DEVICE='$(_CAMERA_DEVICE_RAW)' is not readable$(NC)"; \
			ERRORS=$$((ERRORS+1)); \
		elif [ ! -c "$(_CAMERA_DEVICE_RAW)" ]; then \
			echo "  $(RED)✗ CAMERA_DEVICE='$(_CAMERA_DEVICE_RAW)' is not a character device$(NC)"; \
			ERRORS=$$((ERRORS+1)); \
		else \
			echo "  $(GREEN)✓ CAMERA_DEVICE=$(_CAMERA_DEVICE_RAW)$(NC)"; \
		fi; \
	else \
		echo "  $(GREEN)✓ CAMERA_DEVICE is optional in sample mode$(NC)"; \
	fi; \
	\
	if [ "$(TARGET_DEVICE)" = "GPU" ]; then \
		if [ -z "$(RENDER_GID)" ]; then \
			echo "  $(RED)✗ RENDER_GID is not set$(NC)  (required for GPU access inside containers)"; \
			echo "    Run: getent group render | cut -d: -f3  then add RENDER_GID=<value> to .env"; \
			ERRORS=$$((ERRORS+1)); \
		else \
			echo "  $(GREEN)✓ RENDER_GID=$(RENDER_GID)$(NC)"; \
		fi; \
		if [ -z "$$(ls /dev/dri/renderD* 2>/dev/null)" ]; then \
			echo "  $(YELLOW)⚠ TARGET_DEVICE=GPU but no /dev/dri/renderD* device found$(NC)"; \
			echo "    Install Intel GPU drivers or set TARGET_DEVICE=CPU in .env"; \
		else \
			echo "  $(GREEN)✓ GPU device node found$(NC) ($$(ls /dev/dri/renderD* 2>/dev/null | head -1))"; \
		fi; \
	fi; \
	\
	if [ ! -d models/$(OVMS_MODEL_NAME) ]; then \
		echo "  $(YELLOW)⚠ OVMS model not found at models/$(OVMS_MODEL_NAME)$(NC)"; \
		echo "    Run 'make setup-models' before 'make up' (first time only)"; \
	else \
		echo "  $(GREEN)✓ OVMS model present$(NC) (models/$(OVMS_MODEL_NAME))"; \
	fi; \
	\
	echo ""; \
	if [ $$ERRORS -gt 0 ]; then \
		echo "$(RED)✗ Environment check failed — fix the errors above before proceeding.$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)✓ Environment OK$(NC)"; \
	fi

show-config: ## Print resolved configuration
	@echo "$(BLUE)Current Configuration:$(NC)"
	@echo "======================"
	@echo ""
	@echo "$(YELLOW)Build:$(NC)"
	@echo "  REGISTRY          = $(REGISTRY)  (docker-compose prefix: $(_DOCKER_REGISTRY))"
	@echo "  RELEASE_TAG       = $(RELEASE_TAG)"
	@echo "  COMPOSE_FILE      = $(COMPOSE_FILE)"
	@echo ""
	@echo "$(YELLOW)Inference:$(NC)"
	@echo "  TARGET_DEVICE     = $(TARGET_DEVICE)"
	@echo "  OVMS_MODEL_NAME   = $(OVMS_MODEL_NAME)"
	@echo "  RENDER_GID        = $(RENDER_GID)"
	@echo "  ACCEL_MOUNT_PATH  = $$([ -n '$(ACCEL_MOUNT_PATH)' ] && echo '$(ACCEL_MOUNT_PATH)' || echo '(unset -> NPU disabled)')"
	@echo ""
	@echo "$(YELLOW)Benchmark profile:$(NC)"
	@echo "  PROFILE           = $(if $(PROFILE),$(PROFILE),(none))"
	@echo "  RESULTS_DIR       = $(RESULTS_DIR)"
	@echo "  METRICS_DIR       = $(METRICS_DIR)"
	@echo "  ASR_DEVICE        = $(if $(ASR_DEVICE),$(ASR_DEVICE),(config default))"
	@echo "  ASR_PREVIEW_DEVICE= $(if $(ASR_PREVIEW_DEVICE),$(ASR_PREVIEW_DEVICE),(= ASR_DEVICE))"
	@echo "  ASR_MODEL         = $(if $(ASR_MODEL),$(ASR_MODEL),(config default))"
	@echo "  TTS_DEVICE        = $(if $(TTS_DEVICE),$(TTS_DEVICE),(config default))"
	@echo "  TTS_MODEL/RUNTIME = $(if $(TTS_MODEL),$(TTS_MODEL),(config default))/$(if $(TTS_RUNTIME),$(TTS_RUNTIME),(config default))"
	@echo "  TTS_DTYPE         = $(if $(TTS_DTYPE),$(TTS_DTYPE),(config default))"
	@echo ""
	@echo "$(YELLOW)Authentication:$(NC)"
	@echo "  HF_TOKEN          = $$([ -n '$(HF_TOKEN)' ] && echo '***set***' || echo 'NOT SET')"
	@echo ""
	@echo "$(YELLOW)RTSP source:$(NC)"
	@echo "  RTSP_SOURCE_MODE  = $(_RTSP_SOURCE_MODE)"
	@echo "  CAMERA_DEVICE     = $$([ -n '$(_CAMERA_DEVICE_RAW)' ] && echo '$(_CAMERA_DEVICE_RAW)' || echo '(unset)')"
	@echo "  RTSP_CAMERA_DEVICE= $(_RTSP_CAMERA_DEVICE)"
	@echo ""
	@echo "$(YELLOW)Identity feature:$(NC)"
	@echo "  IDENTITY          = $(IDENTITY)  (from .env KIOSK_CORE_IDENTITY_ENABLED, override with IDENTITY=true|false)"
	@echo "  identity-service  = $$([ -n '$(_IDENTITY_SVC)' ] && echo 'enabled — will start with --profile identity' || echo 'disabled')"
	@echo ""
	@echo "$(YELLOW)Queue/RTSP feature:$(NC)"
	@echo "  QUEUE             = $(QUEUE)  (from .env KIOSK_CORE_QUEUE_SERVICE_ENABLED, override with QUEUE=true|false)"
	@echo "  rtsp-streamer/queue-service = $$([ -n '$(_QUEUE_PROFILE)' ] && echo 'enabled — will start with --profile queue' || echo 'disabled (frees ~10-12 CPU cores for voice-pipeline benchmarking)')"
	@echo ""

# =============================================================================
# Build
# =============================================================================

build: check-env ## Pull images (default with .env); REGISTRY=false builds locally
	@echo "$(BLUE)Preparing Docker images...$(NC)"
	@# set -e: without it this whole recipe is one `cmd; cmd; cmd` shell line
	@# whose exit status is that of the LAST command -- the tolerant
	@# `pull metrics-collector || echo ...` below, which always succeeds. A
	@# failed audio-analyzer or text-to-speech build was therefore swallowed
	@# and the recipe still printed "✓ Images ready" and exited 0.
	@set -e; \
	if [ "$(REGISTRY)" = "false" ] || [ "$(REGISTRY)" = "FALSE" ]; then \
		echo "$(YELLOW)  REGISTRY=false → Building local services from source$(NC)"; \
		REGISTRY=$(_DOCKER_REGISTRY) $(COMPOSE_BASE) $(_IDENTITY_PROFILE) $(_QUEUE_PROFILE) build $(LOCAL_ONLY_SERVICES) $(_IDENTITY_SVC); \
		echo "$(YELLOW)  Building audio-analyzer locally from ../edge-ai-libraries/microservices/audio-analyzer...$(NC)"; \
		REGISTRY=$(_DOCKER_REGISTRY) $(COMPOSE_BASE) build audio-analyzer; \
		echo "$(YELLOW)  Building text-to-speech locally from ../edge-ai-libraries/microservices/text-to-speech...$(NC)"; \
		REGISTRY=$(_DOCKER_REGISTRY) $(COMPOSE_BASE) build text-to-speech; \
	else \
		echo "$(YELLOW)  REGISTRY=$(REGISTRY) → Pulling images from registry '$(_DOCKER_REGISTRY)'$(NC)"; \
		REGISTRY=$(_DOCKER_REGISTRY) $(COMPOSE_BASE) pull $(LOCAL_ONLY_SERVICES); \
		if [ -n "$(_IDENTITY_SVC)" ]; then \
			echo "$(YELLOW)  Building identity-service locally (no registry image)...$(NC)"; \
			REGISTRY=$(_DOCKER_REGISTRY) $(COMPOSE_BASE) $(_IDENTITY_PROFILE) build identity-service; \
		fi; \
		echo "$(YELLOW)  Pulling audio-analyzer from registry...$(NC)"; \
		REGISTRY=$(_DOCKER_REGISTRY) $(COMPOSE_BASE) pull audio-analyzer; \
		echo "$(YELLOW)  Building text-to-speech locally (registry image lacks Kokoro TTS support - see docker-compose.yml text-to-speech comment)...$(NC)"; \
		REGISTRY=$(_DOCKER_REGISTRY) $(COMPOSE_BASE) build text-to-speech; \
	fi; \
	REGISTRY=$(_DOCKER_REGISTRY) $(COMPOSE_BASE) pull metrics-collector || \
		echo "$(YELLOW)  ⚠ Could not pull metrics-collector (intel/hl-ai-metrics-collector) from registry -- using locally cached image if present. Build it from edge-ai-suites/health-and-life-sciences-ai-suite/multi_modal_patient_monitoring/services/metrics-collector if missing.$(NC)"
	@echo "$(GREEN)✓ Images ready$(NC)"

# =============================================================================
# Up / Down
# =============================================================================

up: check-env setup-dirs clean-metrics ## Start all services
	@echo "$(BLUE)Starting Smart Kiosk Assistant...$(NC)"
	REGISTRY=$(_DOCKER_REGISTRY) $(COMPOSE_BASE) $(_IDENTITY_PROFILE) $(_QUEUE_PROFILE) up -d
	@echo ""
	@echo "$(GREEN)✓ Stack started$(NC)"
	@echo "  Kiosk UI        → http://localhost:7860"
	@echo "  kiosk-core API  → http://localhost:8012/docs"
	@echo "  rag-service API → http://localhost:8020/docs"
	@echo "  OVMS LLM        → http://localhost:8000/v3/models"
	@echo "  Metrics         → http://localhost:9000/metrics"
	@echo ""
	@echo "  Run 'make logs' to follow service output"
	@echo "  Run 'make test' to verify all endpoints are healthy"

down: ## Stop all services
	@echo "$(BLUE)Stopping Smart Kiosk Assistant...$(NC)"
	@# Always tear down with --profile identity/queue enabled, regardless of
	@# the IDENTITY/QUEUE/.env value at the time `down` is invoked. Compose
	@# scopes `down` to the same profile-filtered service set as `up`; if a
	@# profile was active when the stack was started and is no longer active
	@# at teardown time, compose never considers those containers part of
	@# the project and leaves them (and the network) running — this is what
	@# caused "Network ... Resource is still in use" and orphaned containers
	@# in the past. Including both profiles unconditionally on `down` is a
	@# no-op for any profile that was never started.
	$(COMPOSE_BASE) --profile identity --profile queue down
	@# Deliberately NOT clean-metrics here (only `up` clears it, as a
	@# prerequisite before the next run starts). `make benchmark`'s
	@# orchestrator calls `make down` internally right after the measured
	@# run, in a `finally:` block, before this Makefile's
	@# consolidate-metrics/plot-metrics targets ever run -- clearing
	@# ./metrics/ here deleted the just-collected qmassa/cpu/memory data
	@# (including the qmassa*-tool-generated.json file) before anyone could
	@# read it. Bind-mounted files surviving `docker compose down` is exactly
	@# how order-accuracy/take-away's flow works too: nothing removes
	@# collected metrics between stack teardown and the human/consolidate
	@# step, only between one run's teardown and the next run's startup.
	@echo "$(GREEN)✓ All services stopped$(NC)"

restart: down up ## Stop then restart all services

# =============================================================================
# Status & Monitoring
# =============================================================================

status: ## Show container status
	@echo "$(BLUE)Container Status:$(NC)"
	@$(COMPOSE_BASE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# =============================================================================
# Health Tests
# =============================================================================

test: ## Run health checks on all service endpoints
	@echo "$(BLUE)Running health checks...$(NC)"
	@_pass=0; _fail=0; \
	_check() { \
		name=$$1; url=$$2; \
		result=$$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" --max-time 5 "$$url" 2>/dev/null); \
		if [ "$$result" = "200" ]; then \
			echo "  $(GREEN)✓ $$name$(NC) ($$url)"; \
		else \
			echo "  $(RED)✗ $$name$(NC) ($$url) → HTTP $$result"; \
		fi; \
	}; \
	curl --noproxy '*' -s --max-time 5 http://127.0.0.1:8000/v3/models | grep -q "model" \
		&& echo "  $(GREEN)✓ ovms-llm$(NC) (http://127.0.0.1:8000/v3/models)" \
		|| echo "  $(RED)✗ ovms-llm$(NC) (http://127.0.0.1:8000/v3/models)"; \
	curl --noproxy '*' -s --max-time 5 http://127.0.0.1:8010/health | grep -q "ok" \
		&& echo "  $(GREEN)✓ audio-analyzer$(NC) (http://127.0.0.1:8010/health)" \
		|| echo "  $(RED)✗ audio-analyzer$(NC) (http://127.0.0.1:8010/health)"; \
	curl --noproxy '*' -s --max-time 5 http://127.0.0.1:8011/health | grep -q "ok" \
		&& echo "  $(GREEN)✓ text-to-speech$(NC) (http://127.0.0.1:8011/health)" \
		|| echo "  $(RED)✗ text-to-speech$(NC) (http://127.0.0.1:8011/health)"; \
	curl --noproxy '*' -s --max-time 5 http://127.0.0.1:8020/health | grep -q "ok" \
		&& echo "  $(GREEN)✓ rag-service$(NC) (http://127.0.0.1:8020/health)" \
		|| echo "  $(RED)✗ rag-service$(NC) (http://127.0.0.1:8020/health)"; \
	curl --noproxy '*' -s --max-time 5 http://127.0.0.1:8012/health | grep -q "ok" \
		&& echo "  $(GREEN)✓ kiosk-core$(NC) (http://127.0.0.1:8012/health)" \
		|| echo "  $(RED)✗ kiosk-core$(NC) (http://127.0.0.1:8012/health)"; \
	if [ -n "$(_IDENTITY_ON)" ]; then \
		curl --noproxy '*' -s --max-time 5 http://127.0.0.1:8013/health | grep -q "ok" \
			&& echo "  $(GREEN)✓ identity-service$(NC) (http://127.0.0.1:8013/health)" \
			|| echo "  $(RED)✗ identity-service$(NC) (http://127.0.0.1:8013/health)"; \
	fi; \
	curl --noproxy '*' -s --max-time 5 http://127.0.0.1:7860 | grep -q "html" \
		&& echo "  $(GREEN)✓ kiosk-ui$(NC) (http://127.0.0.1:7860)" \
		|| echo "  $(RED)✗ kiosk-ui$(NC) (http://127.0.0.1:7860)"

# =============================================================================
# Logs
# =============================================================================

logs: ## Tail logs for all services
	$(COMPOSE_BASE) logs -f

logs-ovms: ## Tail ovms-llm logs
	docker logs -f ovms-llm

logs-rag: ## Tail rag-service logs
	docker logs -f rag-service

logs-core: ## Tail kiosk-core logs
	docker logs -f kiosk-core

logs-ui: ## Tail kiosk-ui logs
	docker logs -f kiosk-ui

logs-audio: ## Tail audio-analyzer logs
	docker logs -f audio-analyzer

logs-tts: ## Tail text-to-speech logs
	docker logs -f text-to-speech

logs-metrics: ## Tail metrics-collector logs
	docker logs -f metrics-collector

logs-identity: ## Tail identity-service logs
	docker logs -f identity-service

# =============================================================================
# Development
# =============================================================================

dev-build: ## Rebuild and restart a single service: make dev-build SVC=rag-service
	@if [ -z "$(SVC)" ]; then \
		echo "$(RED)Error: specify a service — e.g. make dev-build SVC=rag-service$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Rebuilding $(SVC)...$(NC)"
	$(COMPOSE_BASE) build $(SVC)
	$(COMPOSE_BASE) up -d $(SVC)
	@echo "$(GREEN)✓ $(SVC) rebuilt and restarted$(NC)"

# =============================================================================
# Clean
# =============================================================================

clean: ## Stop containers and remove named volumes
	@echo "$(BLUE)Stopping services and removing volumes...$(NC)"
	$(COMPOSE_BASE) down -v
	@$(MAKE) --no-print-directory clean-metrics
	@echo "$(GREEN)✓ Containers stopped and volumes removed$(NC)"
	@echo "  Note: models/ directory is preserved. Delete it manually to free ~4 GB."

clean-metrics: ## Delete collected OS metrics dumps (metrics/)
	@# The OS-level collectors write into ./metrics (bind-mounted to
	@# /tmp/results). qmassa in particular accumulates a single JSON that grows
	@# without bound -- measured at ~596 MB/hour, 2.4 GB after four hours -- and
	@# nothing removed it between runs, so it accrued across the machine's whole
	@# lifetime rather than a single session.
	@#
	@# Docker creates the bind-mount directory root-owned, so a plain host-side
	@# `rm` gets EACCES. Deleting a file needs write permission on its *parent*
	@# directory, not on the file, so the fallback empties the directory from
	@# inside a root container and the host then recreates it owned by the
	@# invoking user -- which is what stops the permission problem recurring.
	@if [ -d $(METRICS_DIR) ]; then \
		sz=$$(du -sh $(METRICS_DIR) 2>/dev/null | cut -f1); \
		if ! rm -rf $(METRICS_DIR) 2>/dev/null; then \
			echo "  $(METRICS_DIR)/ is root-owned; clearing it via a container..."; \
			$(COMPOSE_BASE) run --rm --no-deps --user root --entrypoint sh \
				metrics-collector -c 'rm -rf /tmp/results/* /tmp/results/.[!.]*' \
				>/dev/null 2>&1 || true; \
			rm -rf $(METRICS_DIR) 2>/dev/null || true; \
		fi; \
		if [ -d $(METRICS_DIR) ]; then \
			echo "$(YELLOW)⚠ Could not fully remove $(METRICS_DIR)/ (needs root: sudo rm -rf $(METRICS_DIR))$(NC)"; \
		else \
			echo "$(GREEN)✓ Cleared $(METRICS_DIR)/ (freed $$sz)$(NC)"; \
		fi; \
	fi
	@mkdir -p $(METRICS_DIR)


update-submodule update-submodules: ## Check out the pinned performance-tools submodule commit
	@echo "$(BLUE)Cloning/updating performance tool repositories...$(NC)"
	@if ! git -C .. ls-files --error-unmatch performance-tools >/dev/null 2>&1; then \
		echo "$(RED)performance-tools is not registered as a submodule in the Git root.$(NC)"; \
		echo "  Register it once with: git -C .. submodule add --branch main https://github.com/intel-retail/performance-tools.git performance-tools"; \
		exit 1; \
	fi
	@# Deliberately NOT --remote. --remote ignores the gitlink recorded in
	@# this repository and fast-forwards the submodule to the tip of its
	@# configured branch (main), which does NOT contain
	@# benchmark_smart_kiosk_v2v.py -- so running this target used to be the
	@# very thing that made `make benchmark` fail with "orchestrator not
	@# found", which then told the user to run this target again. Without
	@# --remote the pinned commit recorded here is checked out, which is the
	@# only commit this stack is known to work against.
	@git -C .. submodule update --init --recursive performance-tools
	@echo "$(GREEN)Submodules updated to the pinned commit.$(NC)"
	@echo "  performance-tools @ $$(git -C ../performance-tools rev-parse --short HEAD)"
	@echo "  To intentionally move the pin, check out the new commit inside"
	@echo "  ../performance-tools and commit the updated gitlink here."

fetch-benchmark:
	@echo "$(BLUE)Fetching benchmark image from registry...$(NC)"
	docker pull $(REGISTRY_BENCHMARK)
	@echo "$(GREEN)Benchmark image ready$(NC)"

build-benchmark:
	@echo "$(BLUE)Building benchmark Docker image...$(NC)"
	@if [ "$(REGISTRY)" = "true" ]; then \
		$(MAKE) fetch-benchmark; \
	else \
		cd ../performance-tools && $(MAKE) build-benchmark-docker; \
	fi

# =============================================================================
# Voice-to-voice latency benchmark (performance-tools orchestrated)
# =============================================================================
# Runs the shared performance-tools orchestrator, which brings this stack up
# with the CPU-hungry vision services and diarization OFF, replays a scripted
# conversation through the real start-stream pipeline, and writes both the
# benchmark's own JSON report and vlm_application_metrics_*.txt files that the
# performance-tools collectors understand.
#
# The orchestrator owns the stack lifecycle (it calls `make up`/`make down`
# itself), so do NOT run this against an already-running stack unless you also
# pass V2V_ARGS="--skip_stack".
#
# V2V_SCRIPT defaults to "all": one pass over every scenario in
# tests/benchmarks/v2v_scripted_conversation_benchmark.py's SCRIPTS -- single
# item, multi-item with an addition, a hesitant self-correcting customer, and
# a price-question + upsell + confirm journey (11 turns). At V2V_RUNS=8 that
# is 88 measured turns, which is what makes the reported p95 mean something:
# the previous default (one 4-line conversation, V2V_RUNS=1) gave four turns,
# so "p95" was just the slowest of the four.
#
# Set V2V_SCRIPT= (empty) to fall back to Sample_data/conversation.jsonl --
# edit that one file to script a custom conversation without touching this
# Makefile or the submodule. Set V2V_SCRIPT=order-simple (or order-multi/
# order-drift/order-upsell) to run a single named scenario.
V2V_SCRIPT      ?= all
V2V_RUNS        ?= 8
V2V_WARMUP_RUNS ?= 1
V2V_LABEL       ?= v2v-perf-tools
V2V_ARGS        ?=
# Lives inside the performance-tools submodule (benchmark-scripts/), next to
# benchmark_order_accuracy.py, so it can bring the stack up the same way that
# script does and chain straight into consolidate-metrics/plot-metrics below
# without crossing repos. Now committed there (see performance-tools'
# benchmark_smart_kiosk_v2v.py history) and kept in sync via
# `make update-submodules`.
_PERF_TOOLS_DIR  = ../performance-tools/benchmark-scripts
_V2V_PERF_TOOLS  = $(_PERF_TOOLS_DIR)

benchmark: check-env validate-profile ## Run the voice-to-voice latency benchmark via performance-tools
	@if [ ! -f "$(_V2V_PERF_TOOLS)/benchmark_smart_kiosk_v2v.py" ]; then \
		echo "$(RED)benchmark orchestrator not found at $(_V2V_PERF_TOOLS)$(NC)"; \
		if [ ! -d "../performance-tools/.git" ] && [ ! -f "../performance-tools/.git" ]; then \
			echo "  The performance-tools submodule is not checked out."; \
			echo "  Clone with submodules:  git clone --recurse-submodules <repo>"; \
			echo "  Or, in an existing clone: make update-submodules"; \
		else \
			echo "  The submodule is checked out at $$(git -C ../performance-tools rev-parse --short HEAD),"; \
			echo "  which does not contain benchmark_smart_kiosk_v2v.py."; \
			echo "  Restore the commit pinned by this repository with: make update-submodules"; \
		fi; \
		exit 1; \
	fi
	@$(MAKE) --no-print-directory _write-provenance
	@echo "$(BLUE)Running voice-to-voice benchmark (profile=$(if $(PROFILE),$(PROFILE),none), script=$(if $(V2V_SCRIPT),$(V2V_SCRIPT),Sample_data/conversation.jsonl), runs=$(V2V_RUNS))...$(NC)"
	@# Warm-up runs are discarded. Changing an inference device forces an
	@# OpenVINO recompile/export on first use -- TTS on GPU measured 281s for
	@# the very first call -- so folding that cold start into the measured
	@# runs would swamp the steady-state number the profile exists to compare.
	@#
	@# The warm-up is delegated to the orchestrator's own --warmup_runs rather
	@# than run here. This Makefile used to do a separate warm-up pass first,
	@# which was wrong three ways: it brought the stack up and down a second
	@# time (so it warmed a process that was then discarded), `V2V_WARMUP_RUNS=0`
	@# did not disable the orchestrator's warm-up underneath it, and its output
	@# was sent to /dev/null with `|| true`, so a warm-up that failed outright
	@# looked identical to one that succeeded. Passing the value through means
	@# the run manifest's "warmup_runs" now records what actually ran.
	cd $(_V2V_PERF_TOOLS) && python3 benchmark_smart_kiosk_v2v.py \
		--app_dir $(CURDIR) \
		--results_dir $(CURDIR)/$(RESULTS_DIR) \
		$(if $(V2V_SCRIPT),--script $(V2V_SCRIPT),) \
		--runs $(V2V_RUNS) \
		--warmup_runs $(V2V_WARMUP_RUNS) \
		--label $(V2V_LABEL) \
		$(V2V_ARGS)
	@# Only reached when the orchestrator exited 0. A manifest left reading
	@# "started" therefore means the run died partway and the result files
	@# beside it belong to an earlier run.
	@python3 scripts/finish_run_manifest.py --results-dir $(RESULTS_DIR)
	@# The orchestrator's `make down` (called internally, in a `finally:`
	@# block) leaves the OS-level hardware counters sitting in ./metrics/
	@# (bind-mounted from the now-stopped metrics-collector container) --
	@# they are never copied into ./results/ on their own. Without this,
	@# `make benchmark` looks like it "doesn't generate" the qmassa/cpu/
	@# memory/npu files: they exist, just in the wrong directory. Chain the
	@# consolidation here so a single `make benchmark` always leaves
	@# ./results/ with the hardware counters *and* consolidated_metrics.csv.
	@$(MAKE) --no-print-directory consolidate-metrics

# =============================================================================
# Metrics processing (performance-tools orchestrated)
# =============================================================================
# Chains straight off `make benchmark`'s output. Two directories exist:
#   ./metrics/  -- continuous OS-level counters from the always-on
#                  metrics-collector container (cpu_usage.log, memory_usage.log,
#                  npu_usage.csv, qmassa<card>-<device>-<driver>-tool-generated.json)
#   ./results/  -- per-run application-level metrics from the v2v orchestrator
#                  (vlm_application_metrics_*.txt and its own JSON/CSV reports)
#
# consolidate_multiple_run_of_metrics.py matches KPI extractors by filename
# SUBSTRING across every file under --root_directory (e.g. its "camera"
# extractor matches any file merely named *camera*, like kiosk-ui's
# useCamera.ts) -- pointing it at the whole app repo silently misparses
# unrelated source files and crashes. So sync-metrics-into-results copies the
# known hardware-counter filenames from ./metrics/ into ./results/ first,
# and both targets below only ever point at ./results -- which, unlike the
# repo root, contains nothing but benchmark output.
#
# Both targets use a throwaway venv inside performance-tools (same pattern as
# `make benchmark`/order-accuracy's Makefile) since these scripts need
# pandas/natsort/matplotlib, which this app's own requirements.txt does not
# carry.
_PERF_TOOLS_VENV = . venv/bin/activate

sync-metrics-into-results: setup-dirs ## Copy OS-level hardware counters from ./metrics into ./results
	@# -p preserves modification times, which trim_metrics_to_window.py below
	@# uses to anchor each counter file onto the epoch clock -- none of these
	@# formats records an epoch timestamp, and two of them are written in the
	@# container's timezone. A plain `cp` would reset the mtimes to now and
	@# silently destroy that anchor.
	@cp -pf $(METRICS_DIR)/cpu_usage.log $(METRICS_DIR)/memory_usage.log $(METRICS_DIR)/npu_usage.csv \
		$(RESULTS_DIR)/ 2>/dev/null || true
	@cp -pf $(METRICS_DIR)/qmassa*-tool-generated.json $(RESULTS_DIR)/ 2>/dev/null || true
	@# The collector samples for the whole time the stack is up -- image
	@# pulls, the health wait, the discarded warmup, teardown -- so averaging
	@# the raw files describes a mostly-idle stack rather than the measured
	@# turns. Narrow them to the window the benchmark recorded. A no-op when
	@# results/measured_window.json is absent, i.e. outside a benchmark run.
	@python3 scripts/trim_metrics_to_window.py --results-dir $(RESULTS_DIR)
	@# qmassa rewrites its whole JSON document from scratch on every sample
	@# rather than appending; a container killed mid-write (or -- observed in
	@# practice -- CPU/IO contention while `docker compose down` tears down
	@# the whole stack in parallel, even with metrics-collector's own
	@# stop_grace_period) can leave this file truncated mid-object, which
	@# parse_qmassa_metrics_to_json.py then rejects outright. Repair any
	@# truncation in place before that parser ever sees the file -- see
	@# tests/benchmarks/repair_qmassa_json.py's docstring for the approach.
	@python3 tests/benchmarks/repair_qmassa_json.py --dir $(RESULTS_DIR) --keyword qmassa

parse-qmassa-metrics: sync-metrics-into-results ## Parse raw qmassa*-tool-generated.json into qmassa*-parsed.json
	@# consolidate-metrics and plot-metrics only match *parsed*.json, not the
	@# raw tool-generated file collect_gpu.sh writes -- this must run first.
	cd $(_PERF_TOOLS_DIR) && \
	( \
		python3 -m venv venv && \
		$(_PERF_TOOLS_VENV) && \
		pip3 install -q -r requirements.txt && \
		python3 parse_qmassa_metrics_to_json.py -d $(CURDIR)/$(RESULTS_DIR) -k qmassa; \
		deactivate \
	)

consolidate-metrics: parse-qmassa-metrics ## Consolidate hardware + benchmark metrics into one CSV
	@echo "$(BLUE)Consolidating metrics from ./results...$(NC)"
	cd $(_PERF_TOOLS_DIR) && \
	( \
		$(_PERF_TOOLS_VENV) && \
		python3 consolidate_multiple_run_of_metrics.py \
			--root_directory $(CURDIR)/$(RESULTS_DIR) \
			--output $(CURDIR)/$(RESULTS_DIR)/consolidated_metrics.csv; \
		deactivate \
	)
	@echo "$(GREEN)✓ Wrote results/consolidated_metrics.csv$(NC)"

plot-metrics: parse-qmassa-metrics ## Generate a CPU/GPU/NPU/memory usage graph from ./results
	@echo "$(BLUE)Generating metrics plot from ./results...$(NC)"
	cd $(_PERF_TOOLS_DIR) && \
	( \
		$(_PERF_TOOLS_VENV) && \
		timeout 60s python3 usage_graph_plot.py --dir $(CURDIR)/$(RESULTS_DIR) || true; \
		deactivate \
	)
	@echo "$(GREEN)✓ Wrote results/plot_metrics.png$(NC)"

# =============================================================================
# Device-combination benchmarking (profiles)
# =============================================================================
# A profile is an env file under configs/benchmark-profiles/ describing ONE
# device combination. See that directory's README.md for the engine/device
# compatibility matrix and the methodology behind the pinned models.
#
#   make list-profiles
#   make validate-profile PROFILE=asr-npu_tts-cpu
#   make benchmark        PROFILE=asr-npu_tts-cpu
#   make benchmark-matrix
#   make compare-profiles

_PROFILE_DIR  = configs/benchmark-profiles
_ALL_PROFILES = $(sort $(basename $(notdir $(wildcard $(_PROFILE_DIR)/*.env))))

# Profiles benchmark-matrix will run, in order. Override to run a subset:
#   make benchmark-matrix PROFILES="asr-cpu_tts-cpu asr-npu_tts-cpu"
PROFILES ?= $(_ALL_PROFILES)

list-profiles: ## List available benchmark device profiles
	@echo "$(BLUE)Available benchmark profiles ($(_PROFILE_DIR)/):$(NC)"
	@for p in $(_ALL_PROFILES); do \
		asr=$$(grep -E '^ASR_DEVICE=' $(_PROFILE_DIR)/$$p.env | cut -d= -f2); \
		tts=$$(grep -E '^TTS_DEVICE=' $(_PROFILE_DIR)/$$p.env | cut -d= -f2); \
		llm=$$(grep -E '^TARGET_DEVICE=' $(_PROFILE_DIR)/$$p.env | cut -d= -f2); \
		printf "  %-24s ASR=%-4s TTS=%-4s LLM=%-4s\n" "$$p" "$$asr" "$$tts" "$$llm"; \
	done
	@echo ""
	@echo "  Run one:  make benchmark PROFILE=<name>"
	@echo "  Run all:  make benchmark-matrix"

# Pre-flight gate for `benchmark`. Catches impossible combinations (Kokoro on
# GPU, distil-small.en on NPU) and -- more importantly -- silently mislabelled
# ones: with ACCEL_MOUNT_PATH unset or /dev/null, an "NPU" profile would
# quietly run on CPU and produce plausible but wrong numbers. Fails in seconds
# instead of 20+ minutes. No-op when PROFILE is unset.
validate-profile: ## Validate a device profile without starting anything
	@if [ -n "$(PROFILE)" ]; then \
		python3 scripts/validate_device_profile.py --profile $(PROFILE) || exit 1; \
	else \
		echo "$(YELLOW)No PROFILE set — validating current .env$(NC)"; \
		python3 scripts/validate_device_profile.py || exit 1; \
	fi

# Records exactly what produced a result set, so a number found months later
# can still be traced to its configuration. Written before the measured runs.
_write-provenance: setup-dirs
	@if [ -n "$(PROFILE)" ]; then \
		cp -f $(_PROFILE_FILE) $(RESULTS_DIR)/profile.env 2>/dev/null || true; \
	fi
	@python3 scripts/write_run_manifest.py \
		--results-dir $(RESULTS_DIR) \
		--metrics-dir $(METRICS_DIR) \
		--profile "$(PROFILE)" \
		--release-tag "$(RELEASE_TAG)" \
		--v2v-runs "$(V2V_RUNS)" \
		--v2v-warmup-runs "$(V2V_WARMUP_RUNS)" \
		--queue-enabled "$(QUEUE)"

benchmark-matrix: ## Run every profile sequentially, then build the comparison CSV
	@echo "$(BLUE)Validating all profiles before starting...$(NC)"
	@fail=0; for p in $(PROFILES); do \
		python3 scripts/validate_device_profile.py --profile $$p >/dev/null 2>&1 \
			|| { echo "$(RED)  ✗ $$p is invalid$(NC)"; \
			     python3 scripts/validate_device_profile.py --profile $$p 2>&1 | grep ERROR | head -2; \
			     fail=1; }; \
	done; \
	if [ $$fail -ne 0 ]; then \
		echo "$(RED)Aborting: fix the invalid profile(s) above, or narrow the run with PROFILES=\"...\".$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ All profiles valid$(NC)"
	@# Sequential by design -- profiles contend for the same CPU/GPU/NPU, so
	@# running them concurrently would measure contention, not the device.
	@# Continue on failure so one bad profile does not lose the whole matrix.
	@rm -f $(RESULTS_ROOT)/matrix_status.txt
	@mkdir -p $(RESULTS_ROOT)
	@for p in $(PROFILES); do \
		echo ""; \
		echo "$(BLUE)═══ $$p ═══$(NC)"; \
		if $(MAKE) --no-print-directory benchmark PROFILE=$$p; then \
			echo "$$p PASS" >> $(RESULTS_ROOT)/matrix_status.txt; \
		else \
			echo "$(RED)✗ $$p failed — continuing with the rest$(NC)"; \
			echo "$$p FAIL" >> $(RESULTS_ROOT)/matrix_status.txt; \
		fi; \
	done
	@echo ""
	@echo "$(BLUE)Matrix results:$(NC)"
	@while read -r name status; do \
		if [ "$$status" = "PASS" ]; then \
			printf "  $(GREEN)✓$(NC) %-24s %s\n" "$$name" "$$status"; \
		else \
			printf "  $(RED)✗$(NC) %-24s %s\n" "$$name" "$$status"; \
		fi; \
	done < $(RESULTS_ROOT)/matrix_status.txt
	@$(MAKE) --no-print-directory compare-profiles

compare-profiles: ## Build results/matrix_summary.csv from every profile's results
	@echo "$(BLUE)Comparing profiles under $(RESULTS_ROOT)/...$(NC)"
	@python3 tests/benchmarks/compare_profiles.py \
		--results-root $(RESULTS_ROOT) \
		--output $(RESULTS_ROOT)/matrix_summary.csv
