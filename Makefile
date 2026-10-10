# Makefile — diloco-measured
#
# STATUS: [CONFIRMED] — project closed 2026-08-18 (CLAUDE.md ADR-040). Targets mirror
# CLAUDE.md §9.1 (Journey A) and §14. `install`/`test`/`lint`/`typecheck`/`figures` are real
# and laptop-safe (no GPU or AWS). Infrastructure scripts can affect real AWS
# resources. Unified network-characterize/smoke/grid/converge wrappers are not
# implemented; use the documented campaign scripts in docs/REPRODUCTION.md.

.PHONY: help install install-measurement test test-unit test-integration test-measurement \
        lint typecheck cluster-up bootstrap network-characterize smoke grid converge \
        cluster-down figures cost-report clean

help:
	@echo "diloco-measured — see CLAUDE.md for full spec"
	@echo ""
	@echo "Offline (safe, \$$0, no GPU/torch needed):"
	@echo "  install              install the laptop-sized base package + dev deps (no torch)"
	@echo "  install-measurement  add torch/torchtitan/torchft on top — needed for cluster work"
	@echo "                       or the full test suite (make test-measurement)"
	@echo "  test                 unit + CPU integration tests, excluding torch-dependent ones"
	@echo "  test-measurement     the torch-dependent subset (needs install-measurement first)"
	@echo "  lint / typecheck     ruff / mypy"
	@echo "  figures              regenerate report figures from results/raw/ (FR-11, no GPU)"
	@echo ""
	@echo "Cluster (costs money — see CLAUDE.md §31.1):"
	@echo "  cluster-up           DRY-RUN by default (validates AWS calls, creates/costs nothing)."
	@echo "                       Real launch: LAUNCH=1 DILOCO_OPERATOR_IP=<your-ip> make cluster-up"
	@echo "                       (~\$$9.33/hr once actually launched, CLAUDE.md §5.2)"
	@echo "  bootstrap ROLE=gpu   install deps, lock clocks, sync dataset (ROLE=gpu|control)"
	@echo "  network-characterize unsupported wrapper; see docs/REPRODUCTION.md"
	@echo "  smoke                unsupported wrapper; see docs/REPRODUCTION.md"
	@echo "  grid PHASE=A         unsupported wrapper; see docs/REPRODUCTION.md"
	@echo "  converge PHASE=B     unsupported wrapper; see docs/REPRODUCTION.md"
	@echo "  cluster-down         DRY-RUN by default. Real teardown: make cluster-down TEARDOWN=1"
	@echo "  cost-report          cumulative cluster-hours and spend for what's running right now"

# ---- Offline mode (§31.1) --------------------------------------------------

install:
	uv sync --frozen --extra dev

install-measurement:
	uv sync --frozen --extra dev --extra measurement

test: test-unit test-integration

test-unit:
	uv run pytest -m "unit and not requires_torch" tests/unit -v

test-integration:
	uv run pytest -m "integration_cpu and not requires_torch" tests/integration_cpu -v

test-measurement:
	uv run pytest -m requires_torch tests/unit tests/integration_cpu -v

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

figures: install
	uv run diloco-measured figures

clean:
	rm -rf results/figures/*
	@echo "results/figures/ is generated and safe to delete (CLAUDE.md §14.1). results/raw/ is NOT touched."

# ---- Cluster mode (§31.1 — costs ~\$9.33/hr once actually launched; never develop here) ---

# cluster-up / cluster-down default to dry-run — the underlying scripts require an explicit
# flag to actually create or terminate anything (infra/launch_cluster.sh, infra/teardown.sh).
cluster-up:
ifeq ($(LAUNCH),1)
	bash infra/launch_cluster.sh --launch-for-real
else
	bash infra/launch_cluster.sh
	@echo ""
	@echo "[Makefile] That was a dry run. To actually launch (~\$$9.33/hr):"
	@echo "[Makefile]   DILOCO_OPERATOR_IP=<your-ip> make cluster-up LAUNCH=1"
endif

bootstrap:
	bash infra/setup_node.sh $(or $(ROLE),gpu)

network-characterize:
	uv run diloco-measured network characterize

smoke:
	uv run pytest -m e2e tests/e2e -v

grid:
	uv run diloco-measured grid --config configs/grids/phase_$(or $(PHASE),A).yaml --resume

converge:
	uv run diloco-measured converge --spec configs/grids/phase_$(or $(PHASE),B).yaml

cluster-down:
ifeq ($(TEARDOWN),1)
	bash infra/teardown.sh --terminate-for-real
else
	bash infra/teardown.sh
	@echo ""
	@echo "[Makefile] That was a dry run. To actually terminate: make cluster-down TEARDOWN=1"
endif

cost-report:
	bash infra/cost_report.sh

