# ISRIC SoilGrids v2.0 (250 m) -> Icechunk Zarr pipeline
#
# Store target: set ACCOUNT to commit directly into the Source Coop product, or
# leave it unset to use STORE (a local path / s3://bucket/prefix, default
# ./soilgrids_store_local) - so a bare `make materialize` never touches the
# published store.
#
#   make inspect-source
#   make init-store
#   make materialize BBOX=-104,37,-95,43 PROPERTIES=sand
#   make overviews PROPERTIES=sand
#   make status
#   make validate PROPERTIES=sand
#   make release ACCOUNT=chill

STORE        ?= ./soilgrids_store_local
ACCOUNT      ?=
WORK_DIR     ?= work
PROPERTIES   ?=
BBOX         ?=
CELLS        ?=
WORKERS      ?=
COMMIT_EVERY ?=
PROGRESS_EVERY ?=
RATE_LIMIT   ?=
SAMPLES      ?= 8
LISTING_SAMPLES ?=
CREDS_FILE   ?=
OVERWRITE    ?=
REBUILD      ?=
GC_HOURS     ?=
SUFFIX       ?=

CLI = uv run soilgrids

ifeq ($(ACCOUNT),)
  STORE_FLAGS = --store $(STORE)
else
  STORE_FLAGS = --source-coop-account $(ACCOUNT)
endif
# only pass a creds file when explicitly requested; otherwise the source-coop
# CLI's cached login is used (a stale creds.json must not shadow a fresh login)
CREDS_FLAG      = $(if $(CREDS_FILE),--credentials-file $(CREDS_FILE))
STORE_FLAGS    += $(CREDS_FLAG)
PROPERTIES_FLAG = $(if $(PROPERTIES),--properties $(PROPERTIES))
BBOX_FLAG       = $(if $(BBOX),--bbox $(BBOX))
CELLS_FLAG      = $(if $(CELLS),--cells $(CELLS))
WORKERS_FLAG    = $(if $(WORKERS),--workers $(WORKERS))
COMMIT_FLAG     = $(if $(COMMIT_EVERY),--commit-every $(COMMIT_EVERY))
PROGRESS_FLAG   = $(if $(PROGRESS_EVERY),--progress-every $(PROGRESS_EVERY))
RATE_FLAG       = $(if $(RATE_LIMIT),--rate-limit $(RATE_LIMIT))
LISTING_FLAG    = $(if $(LISTING_SAMPLES),--listing-samples $(LISTING_SAMPLES))
OVERWRITE_FLAG  = $(if $(OVERWRITE),--overwrite)
REBUILD_FLAG    = $(if $(REBUILD),--rebuild)
GC_HOURS_FLAG   = $(if $(GC_HOURS),--older-than-hours $(GC_HOURS))
SUFFIX_FLAG     = $(if $(SUFFIX),--suffix $(SUFFIX))

.DEFAULT_GOAL := help

.PHONY: help setup test lint inspect-source init-store materialize overviews status validate \
	release info garbage-collect publish-readme upload-audit show-config \
	clean-local-store clean-work clean-remote-store

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Variables: STORE=$(STORE)  ACCOUNT=$(ACCOUNT)  WORK_DIR=$(WORK_DIR)"
	@echo "             PROPERTIES=$(PROPERTIES)  BBOX=$(BBOX)  CELLS=$(CELLS)"
	@echo "             COMMIT_EVERY=$(COMMIT_EVERY)  PROGRESS_EVERY=$(PROGRESS_EVERY)"
	@echo "             WORKERS=$(WORKERS)  COMMIT_EVERY=$(COMMIT_EVERY)  SAMPLES=$(SAMPLES)"

setup: ## Install dependencies (uv sync)
	uv sync

test: ## Run the test suite (synthetic fixtures; no network needed)
	uv run pytest -q

lint: ## Fix lint issues and reformat code with ruff
	uv run ruff check --fix src tests
	uv run ruff format src tests

show-config: ## Print the frozen structural spec
	$(CLI) show-config

inspect-source: ## Phase 1: verify the source tree, write the tile manifest (LISTING_SAMPLES=0 for a real build)
	$(CLI) inspect-source --work-dir $(WORK_DIR) $(PROPERTIES_FLAG) $(WORKERS_FLAG) $(RATE_FLAG) $(LISTING_FLAG)

init-store: ## Phase 2: create (or additively extend) the store structure
	$(CLI) init-store $(STORE_FLAGS) $(PROPERTIES_FLAG)

materialize: ## Phase 3: fill native arrays; checkpointed and resumable
	$(CLI) materialize $(STORE_FLAGS) $(PROPERTIES_FLAG) --work-dir $(WORK_DIR) \
		$(BBOX_FLAG) $(CELLS_FLAG) $(WORKERS_FLAG) $(COMMIT_FLAG) $(RATE_FLAG) $(OVERWRITE_FLAG)

overviews: ## Phase 4: build the multiscale pyramid, one property at a time
	$(CLI) overviews $(STORE_FLAGS) $(PROPERTIES_FLAG) $(CELLS_FLAG) $(WORKERS_FLAG) \
		$(COMMIT_FLAG) $(PROGRESS_FLAG) $(REBUILD_FLAG)

status: ## Show the property x cell completion matrix
	$(CLI) status $(STORE_FLAGS) --work-dir $(WORK_DIR)

validate: ## Phase 6: verify structure and sampled contents
	$(CLI) validate $(STORE_FLAGS) $(PROPERTIES_FLAG) --work-dir $(WORK_DIR) --samples $(SAMPLES) $(WORKERS_FLAG)

release: ## Phase 7: tag the release (refuses while anything is incomplete)
	$(CLI) release $(STORE_FLAGS) $(SUFFIX_FLAG)

info: ## Show store structure, tags, and recent snapshots
	$(CLI) info $(STORE_FLAGS)

garbage-collect: ## Reclaim objects orphaned by checkpoint commits (never while writing)
	$(CLI) garbage-collect $(STORE_FLAGS) $(GC_HOURS_FLAG)

publish-readme: ## Upload product/README.md as the Source Coop landing page
	$(CLI) publish-readme --source-coop-account $(or $(ACCOUNT),chill) $(CREDS_FLAG)

upload-audit: ## Upload the source manifest and reports to audit/{version}/
	$(CLI) upload-audit --source-coop-account $(or $(ACCOUNT),chill) $(CREDS_FLAG) --work-dir $(WORK_DIR)

clean-local-store: ## Remove the local icechunk store ($(STORE))
	rm -rf $(STORE)

clean-work: ## Remove manifests and reports ($(WORK_DIR))
	rm -rf $(WORK_DIR)

clean-remote-store: ## DESTRUCTIVE: delete the published store; confirms twice
	$(CLI) clean-remote-store --source-coop-account $(or $(ACCOUNT),chill) $(CREDS_FLAG) $(WORKERS_FLAG)
