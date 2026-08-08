# Makefile — diloco-measured
#
# STATUS: [PROPOSED] scaffold. Targets mirror CLAUDE.md §9.1 (Journey A) and §14.
# Most targets are stubs that fail loudly until the corresponding module (§14 repo
# structure) is implemented — per CLAUDE.md Architecture Principle #5 ("fail loud,
# fail early"), a stub that pretends to succeed is worse than one that doesn't exist.

.PHONY: help install test test-unit test-integration lint typecheck \
        cluster-up bootstrap network-characterize smoke grid converge \
        cluster-down figures cost-report clean

help:
	@echo "diloco-measured — see CLAUDE.md for full spec"
	@echo ""
	@echo "Offline (safe, \$$0):"
	@echo "  install              install package + dev deps"
	@echo "  test                 unit + CPU integration tests"
	@echo "  lint / typecheck     ruff / mypy"
	@echo "  figures              regenerate report figures from results/raw/ (FR-11, no GPU)"
	@echo ""
	@echo "Cluster (costs money — see CLAUDE.md §31.1):"
	@echo "  cluster-up           launch 4x GPU + 1 control node"
	@echo "  bootstrap            install deps, lock clocks, sync dataset"
	@echo "  network-characterize FR-01: iperf3 + NCCL BW curve + burst-decay probe"
	@echo "  smoke                E2E gate: 4 nodes, tiny model, 20 steps"
	@echo "  grid PHASE=A         run a campaign (FR-02..FR-05)"
	@echo "  converge PHASE=B     fixed-token-budget convergence runs (FR-06)"
	@echo "  cluster-down         idempotent teardown — leave nothing billable"
	@echo "  cost-report          cumulative cluster-hours and spend"

# ---- Offline mode (§31.1) --------------------------------------------------

install:
	uv sync --extra dev || pip install -e ".[dev]"

test: test-unit test-integration

test-unit:
	pytest -m unit tests/unit -v

test-integration:
	pytest -m integration_cpu tests/integration_cpu -v

lint:
	ruff check src tests

typecheck:
	mypy src

figures:
	diloco-measured figures

clean:
	rm -rf results/figures/*
	@echo "results/figures/ is generated and safe to delete (CLAUDE.md §14.1). results/raw/ is NOT touched."

# ---- Cluster mode (§31.1 — costs ~\$9.33/hr; never develop here) -----------

cluster-up:
	bash infra/launch_cluster.sh

bootstrap:
	bash infra/setup_node.sh

network-characterize:
	diloco-measured network characterize

smoke:
	pytest -m e2e tests/e2e -v

grid:
	diloco-measured grid --config configs/grids/phase_$(or $(PHASE),A).yaml --resume

converge:
	diloco-measured converge --spec configs/grids/phase_$(or $(PHASE),B).yaml

cluster-down:
	bash infra/teardown.sh

cost-report:
	bash infra/cost_report.sh
