# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Smart Kiosk Assistant - Makefile
# =============================================================================

.PHONY: help init-env setup-dirs setup-models check-env show-config \
		build up down restart status logs \
		logs-ovms logs-rag logs-core logs-ui logs-audio logs-tts logs-metrics \
		test clean dev-build

# =============================================================================
# Load .env (silent if missing — check-env will warn)
# =============================================================================
-include .env
export

# =============================================================================
# Configuration variables (defaults match .env)
# =============================================================================

COMPOSE_FILE     ?= docker-compose.yml

# REGISTRY=false  → build images locally from source (default for dev)
# REGISTRY=true   → pull pre-built images from Docker Hub / use .env REGISTRY value
# NOTE: "false" is a build-mode flag only. docker-compose always receives the real
#       registry name (REGISTRY=intel) so image tags are never "false/<svc>:...".
REGISTRY         ?= false

# _DOCKER_REGISTRY: the actual registry prefix passed to docker-compose.
# When the user passes REGISTRY=false it is treated as a build-mode switch;
# the real registry name comes from .env (default: intel).
_ENV_REGISTRY    ?= intel
_DOCKER_REGISTRY  = $(if $(filter false FALSE,$(REGISTRY)),$(_ENV_REGISTRY),$(REGISTRY))

RELEASE_TAG      ?= 2026.1.0
export TAG       := $(RELEASE_TAG)

TARGET_DEVICE    ?= GPU
OVMS_MODEL_NAME  ?= OpenVINO/Qwen3-4B-int8-ov

# edge-ai-libraries path (audio-analyzer + text-to-speech build contexts live here)
EAL_PATH         := ../edge-ai-libraries/microservices

# Services with fully local build contexts (no edge-ai-libraries needed)
LOCAL_ONLY_SERVICES := metrics-collector rag-service kiosk-core kiosk-ui queue-service rtsp-streamer

# Identity feature (profile-gated). Pass IDENTITY=true to build/start the
# identity-service container AND tell kiosk-core to call it. Default off so the
# standard stack is unchanged.
IDENTITY          ?= false
_IDENTITY_ON       = $(filter true TRUE,$(IDENTITY))
_IDENTITY_PROFILE  = $(if $(_IDENTITY_ON),--profile identity,)
_IDENTITY_SVC      = $(if $(_IDENTITY_ON),identity-service,)
export KIOSK_CORE_IDENTITY_ENABLED = $(if $(_IDENTITY_ON),true,false)

# Services that require edge-ai-libraries checkout to build from source.
# When EAL_PATH is absent, pre-built images (intel/*:RELEASE_TAG) are used instead.
EAL_SERVICES     := audio-analyzer text-to-speech

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
	@echo "  make check-env         Validate .env and required variables"
	@echo "  make show-config       Print resolved configuration"
	@echo ""
	@echo "$(YELLOW)Core Commands:$(NC)"
	@echo "  make build             Build (REGISTRY=false) or pull (REGISTRY=true) images"
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
	@echo ""
	@echo "$(YELLOW)Current settings:$(NC)"
	@echo "  REGISTRY=$(REGISTRY)  TARGET_DEVICE=$(TARGET_DEVICE)  OVMS_MODEL_NAME=$(OVMS_MODEL_NAME)"
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
	@mkdir -p models storage
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
		echo "  $(RED)✗ HF_TOKEN is not set$(NC)"; \
		echo "    Set it in .env  →  https://huggingface.co/settings/tokens"; \
		ERRORS=$$((ERRORS+1)); \
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
	@echo "  EAL_PATH          = $(EAL_PATH)  ($$([ -d '$(EAL_PATH)' ] && echo 'found' || echo 'not present — EAL services use pre-built images'))"
	@echo ""
	@echo "$(YELLOW)Inference:$(NC)"
	@echo "  TARGET_DEVICE     = $(TARGET_DEVICE)"
	@echo "  OVMS_MODEL_NAME   = $(OVMS_MODEL_NAME)"
	@echo "  RENDER_GID        = $(RENDER_GID)"
	@echo ""
	@echo "$(YELLOW)Authentication:$(NC)"
	@echo "  HF_TOKEN          = $$([ -n '$(HF_TOKEN)' ] && echo '***set***' || echo 'NOT SET')"
	@echo ""

# =============================================================================
# Build
# =============================================================================

build: check-env ## Build (REGISTRY=false) or pull (REGISTRY=true) all images
	@echo "$(BLUE)Preparing Docker images...$(NC)"
	@if [ "$(REGISTRY)" = "false" ] || [ "$(REGISTRY)" = "FALSE" ]; then \
		echo "$(YELLOW)  REGISTRY=false → Building local services from source$(NC)"; \
		REGISTRY=$(_DOCKER_REGISTRY) docker compose -f $(COMPOSE_FILE) $(_IDENTITY_PROFILE) build $(LOCAL_ONLY_SERVICES) $(_IDENTITY_SVC); \
		if [ -d "$(EAL_PATH)/audio-analyzer" ] && [ -d "$(EAL_PATH)/text-to-speech" ]; then \
			echo "$(YELLOW)  Building EAL services (audio-analyzer, text-to-speech)...$(NC)"; \
			REGISTRY=$(_DOCKER_REGISTRY) docker compose -f $(COMPOSE_FILE) build audio-analyzer text-to-speech; \
		else \
			echo "$(YELLOW)  ⚠ edge-ai-libraries not found at $(EAL_PATH)$(NC)"; \
			echo "    audio-analyzer and text-to-speech use pre-built images ($(_DOCKER_REGISTRY)/*:$(RELEASE_TAG))"; \
			echo "    To build from source: see docs/user-guide/get-started.md (edge-ai-libraries sparse clone)"; \
		fi; \
	else \
		echo "$(YELLOW)  REGISTRY=$(REGISTRY) → Pulling images from registry$(NC)"; \
		REGISTRY=$(REGISTRY) docker compose -f $(COMPOSE_FILE) pull \
			$(LOCAL_ONLY_SERVICES) $(EAL_SERVICES); \
		if [ -n "$(_IDENTITY_SVC)" ]; then \
			echo "$(YELLOW)  Building identity-service locally (no registry image)...$(NC)"; \
			REGISTRY=$(_DOCKER_REGISTRY) docker compose -f $(COMPOSE_FILE) $(_IDENTITY_PROFILE) build identity-service; \
		fi; \
	fi
	@echo "$(GREEN)✓ Images ready$(NC)"

# =============================================================================
# Up / Down
# =============================================================================

up: check-env setup-dirs ## Start all services
	@echo "$(BLUE)Starting Smart Kiosk Assistant...$(NC)"
	REGISTRY=$(_DOCKER_REGISTRY) docker compose -f $(COMPOSE_FILE) $(_IDENTITY_PROFILE) up -d
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
	docker compose -f $(COMPOSE_FILE) down
	@echo "$(GREEN)✓ All services stopped$(NC)"

restart: down up ## Stop then restart all services

# =============================================================================
# Status & Monitoring
# =============================================================================

status: ## Show container status
	@echo "$(BLUE)Container Status:$(NC)"
	@docker compose -f $(COMPOSE_FILE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

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
	docker compose -f $(COMPOSE_FILE) logs -f

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
	docker compose -f $(COMPOSE_FILE) build $(SVC)
	docker compose -f $(COMPOSE_FILE) up -d $(SVC)
	@echo "$(GREEN)✓ $(SVC) rebuilt and restarted$(NC)"

# =============================================================================
# Clean
# =============================================================================

clean: ## Stop containers and remove named volumes
	@echo "$(BLUE)Stopping services and removing volumes...$(NC)"
	docker compose -f $(COMPOSE_FILE) down -v
	@echo "$(GREEN)✓ Containers stopped and volumes removed$(NC)"
	@echo "  Note: models/ directory is preserved. Delete it manually to free ~4 GB."
