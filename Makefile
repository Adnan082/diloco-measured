# Makefile — diloco-measured
#
# STATUS: [CONFIRMED] — project closed 2026-08-18 (CLAUDE.md ADR-040). Targets mirror
# CLAUDE.md §9.1 (Journey A) and §14. `install`/`test`/`lint`/`typecheck`/`figures` are real
# and laptop-safe (no GPU, no AWS, no torch). Cluster targets (`cluster-up`/`bootstrap`/
# `network-characterize`/`smoke`/`grid`/`converge`/`cluster-down`) drive real scripts against
# a real AWS account, real money — everything the project actually measured with them is
# already committed to `results/raw/`; re-running them starts a NEW real cluster.

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
	@echo "  network-characterize FR-01: iperf3 + NCCL BW curve + burst-decay probe"
	@echo "  smoke                E2E gate: 4 nodes, tiny model, 20 steps"
	@echo "  grid PHASE=A         run a campaign (FR-02..FR-05)"
	@echo "  converge PHASE=B     fixed-token-budget convergence runs (FR-06)"
	@echo "  cluster-down         DRY-RUN by default. Real teardown: make cluster-down TEARDOWN=1"
	@echo "  cost-report          cumulative cluster-hours and spend for what's running right now"

# ---- Offline mode (§31.1) --------------------------------------------------

install:
	uv sync --extra dev || pip install -e ".[dev]"

install-measurement:
	uv sync --extra dev --extra measurement || pip install -e ".[dev,measurement]"

test: test-unit test-integration

test-unit:
	pytest -m "unit and not requires_torch" tests/unit -v

test-integration:
	pytest -m "integration_cpu and not requires_torch" tests/integration_cpu -v

test-measurement:
	pytest -m requires_torch tests/unit tests/integration_cpu -v

lint:
	ruff check src tests

typecheck:
	mypy src

figures: install
	diloco-measured figures

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
	diloco-measured network characterize

smoke:
	pytest -m e2e tests/e2e -v

grid:
	diloco-measured grid --config configs/grids/phase_$(or $(PHASE),A).yaml --resume

converge:
	diloco-measured converge --spec configs/grids/phase_$(or $(PHASE),B).yaml

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
