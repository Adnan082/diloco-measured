"""Unit tests for measurement/fingerprint.py — capture() and scrub() (FR-08, CLAUDE.md §23)."""

from __future__ import annotations

import json

import jsonschema
import pytest

from diloco_measured.measurement import fingerprint as fp_module
from diloco_measured.measurement.fingerprint import UNKNOWN, capture, scrub


@pytest.mark.unit
def test_scrub_redacts_account_id_in_string_value():
    result = scrub({"note": "role arn is for account 123456789012 in prod"})
    assert "123456789012" not in result["note"]
    assert "<redacted-account-id>" in result["note"]


@pytest.mark.unit
def test_scrub_redacts_arn():
    result = scrub({"note": "see arn:aws:iam::123456789012:role/MyRole for details"})
    assert "arn:aws:iam" not in result["note"]
    assert "<redacted-arn>" in result["note"]
    assert "123456789012" not in result["note"]  # the ARN's embedded account id too


@pytest.mark.unit
@pytest.mark.parametrize("ip", ["10.0.1.5", "172.16.0.1", "172.31.255.254", "192.168.1.1"])
def test_scrub_redacts_private_ips(ip):
    result = scrub({"note": f"node reachable at {ip} internally"})
    assert ip not in result["note"]
    assert "<redacted-ip>" in result["note"]


@pytest.mark.unit
def test_scrub_does_not_touch_public_looking_ips_or_normal_text():
    result = scrub({"note": "public endpoint 8.8.8.8, kernel 6.8.0-generic"})
    assert result["note"] == "public endpoint 8.8.8.8, kernel 6.8.0-generic"


@pytest.mark.unit
def test_scrub_recurses_into_lists_and_nested_dicts():
    result = scrub(
        {
            "nccl_env": {"NCCL_SOME_VAR": "leaked account 123456789012"},
            "instance_types": ["fine", "leaked ip 10.1.2.3"],
        }
    )
    assert "123456789012" not in result["nccl_env"]["NCCL_SOME_VAR"]
    assert "10.1.2.3" not in result["instance_types"][1]
    assert result["instance_types"][0] == "fine"


@pytest.mark.unit
def test_scrub_leaves_non_string_values_untouched():
    result = scrub({"seed": 42, "gpu_clocks_locked": True, "fault_schedule": None})
    assert result == {"seed": 42, "gpu_clocks_locked": True, "fault_schedule": None}


@pytest.mark.unit
def test_capture_passes_through_caller_supplied_fields():
    result = capture(seed=7, dataset_shard_checksum="deadbeef", gpu_clocks_locked=True)
    assert result["seed"] == 7
    assert result["dataset_shard_checksum"] == "deadbeef"
    assert result["gpu_clocks_locked"] is True


@pytest.mark.unit
def test_capture_defaults_harness_version_to_package_version():
    import diloco_measured

    result = capture(seed=0, dataset_shard_checksum="x", gpu_clocks_locked=False)
    assert result["harness_version"] == diloco_measured.__version__


@pytest.mark.unit
def test_capture_accepts_explicit_harness_version_override():
    result = capture(
        seed=0, dataset_shard_checksum="x", gpu_clocks_locked=False, harness_version="v7"
    )
    assert result["harness_version"] == "v7"


@pytest.mark.unit
def test_capture_falls_back_to_unknown_and_empty_list_when_probes_fail(monkeypatch):
    monkeypatch.setattr(fp_module, "_imds_get", lambda path: None)
    monkeypatch.setattr(fp_module, "_nvidia_driver_version", lambda: UNKNOWN)
    monkeypatch.setattr(fp_module, "_nvidia_smi_topo", lambda: None)
    monkeypatch.setattr(fp_module, "_git_sha_and_dirty", lambda repo_root: (UNKNOWN, False))

    result = capture(seed=0, dataset_shard_checksum="x", gpu_clocks_locked=False)

    assert result["instance_types"] == []
    assert result["az"] == UNKNOWN
    assert result["driver_version"] == UNKNOWN
    assert "nvidia_smi_topo" not in result  # optional field, omitted rather than set to None
    assert result["harness_git_sha"] == UNKNOWN
    assert result["harness_dirty"] is False


@pytest.mark.unit
def test_capture_uses_imds_values_when_available(monkeypatch):
    monkeypatch.setattr(
        fp_module,
        "_imds_get",
        lambda path: {
            "instance-type": "g6e.2xlarge",
            "placement/availability-zone": "us-east-1a",
            "placement/group-name": "pg-0",
        }.get(path),
    )

    result = capture(seed=0, dataset_shard_checksum="x", gpu_clocks_locked=False)

    assert result["instance_types"] == ["g6e.2xlarge"]
    assert result["az"] == "us-east-1a"
    assert result["placement_group_id"] == "pg-0"


@pytest.mark.unit
def test_capture_output_is_scrubbed(monkeypatch):
    monkeypatch.setattr(
        fp_module, "_nvidia_smi_topo", lambda: "topology near account 123456789012"
    )
    result = capture(seed=0, dataset_shard_checksum="x", gpu_clocks_locked=False)
    assert "123456789012" not in result.get("nvidia_smi_topo", "")


@pytest.mark.unit
def test_capture_result_validates_against_environment_fingerprint_schema(schemas_dir):
    """capture() must always be schema-shape-complete, even off-cluster (dev machine / CI)."""
    with open(schemas_dir / "run_result.v1.json") as f:
        run_result_schema = json.load(f)
    fp_schema = run_result_schema["$defs"]["EnvironmentFingerprint"]

    result = capture(seed=0, dataset_shard_checksum="deadbeef", gpu_clocks_locked=True)

    jsonschema.validate(instance=result, schema=fp_schema)
