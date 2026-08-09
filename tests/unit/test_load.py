"""Unit tests for analysis/load.py — schema-validated loading, refuses invalid records (FR-12)."""

from __future__ import annotations

import json

import pytest

from diloco_measured.analysis.load import (
    SchemaValidationError,
    load_network_profiles,
    load_run_results,
)


def _valid_run_result(run_id: str = "r0") -> dict:
    return {
        "run_id": run_id,
        "spec": {
            "spec_id": "s0", "phase": "cu_grid", "algorithm": "diloco",
            "implementation": "reference", "H": 32,
            "model_config": "configs/models/1b.toml",
            "bandwidth_requested_bps": 1_000_000_000,
            "world_size": 4, "micro_batch_size": 2, "seq_len": 1024, "grad_accum": 4,
            "budget_type": "steps", "budget_value": 200, "warmup_steps": 20,
            "compression": None, "seed": 0, "repeat_index": 0, "fault_schedule": None,
        },
        "fingerprint": {
            "harness_git_sha": "abc123", "harness_dirty": False, "harness_version": "v1",
            "pytorch_version": "2.13", "nccl_version": "2.20", "cuda_version": "12.4",
            "driver_version": "550.0", "instance_types": ["g6e.2xlarge"],
            "az": "us-east-1a", "gpu_clocks_locked": True, "kernel_version": "6.8.0",
            "dataset_shard_checksum": "deadbeef", "seed": 0,
        },
        "network_profile_id": "np0", "harness_version": "v1", "status": "completed",
        "started_at": "2026-08-09T00:00:00Z", "ended_at": "2026-08-09T00:05:00Z",
        "loader_bound_warning": False,
    }


def _valid_network_profile(profile_id: str = "np0") -> dict:
    return {
        "profile_id": profile_id,
        "captured_at": "2026-08-09T00:00:00Z",
        "cluster_id": "pg-0",
        "iperf_pairs": [
            {
                "from_node": "node0", "to_node": "node1", "direction": "fwd",
                "gbit_s": 4.9, "duration_s": 60.0,
            }
        ],
        "nccl_curve": [{"msg_bytes": 1048576, "achieved_bps": 4.5e9}],
        "shaping_fidelity": [
            {"requested_bps": 5_000_000_000, "measured_bps": 4.9e9, "error_pct": 2.0}
        ],
        "burst_decay_detected": False,
    }


@pytest.mark.unit
def test_load_run_results_reads_valid_records(tmp_path):
    (tmp_path / "run_a.json").write_text(json.dumps(_valid_run_result("run_a")))
    (tmp_path / "run_b.json").write_text(json.dumps(_valid_run_result("run_b")))

    records = load_run_results(tmp_path)
    assert {r["run_id"] for r in records} == {"run_a", "run_b"}


@pytest.mark.unit
def test_load_run_results_raises_on_schema_violation(tmp_path):
    bad = _valid_run_result("bad")
    del bad["status"]  # required field
    (tmp_path / "bad.json").write_text(json.dumps(bad))

    with pytest.raises(SchemaValidationError):
        load_run_results(tmp_path)


@pytest.mark.unit
def test_load_run_results_rejects_unknown_status_enum_value(tmp_path):
    bad = _valid_run_result("bad")
    bad["status"] = "totally_made_up_status"
    (tmp_path / "bad.json").write_text(json.dumps(bad))

    with pytest.raises(SchemaValidationError):
        load_run_results(tmp_path)


@pytest.mark.unit
def test_load_run_results_empty_dir_returns_empty_list(tmp_path):
    assert load_run_results(tmp_path) == []


@pytest.mark.unit
def test_load_network_profiles_reads_valid_records(tmp_path):
    (tmp_path / "profile_a.json").write_text(json.dumps(_valid_network_profile("profile_a")))
    records = load_network_profiles(tmp_path)
    assert len(records) == 1
    assert records[0]["profile_id"] == "profile_a"


@pytest.mark.unit
def test_load_network_profiles_raises_on_schema_violation(tmp_path):
    bad = _valid_network_profile("bad")
    del bad["nccl_curve"]  # required field
    (tmp_path / "bad.json").write_text(json.dumps(bad))

    with pytest.raises(SchemaValidationError):
        load_network_profiles(tmp_path)
