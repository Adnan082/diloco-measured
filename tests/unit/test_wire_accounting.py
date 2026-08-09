"""Unit tests for measurement/wire.py::parse_proc_net_dev and ::account.

predict() is covered separately in tests/unit/test_wire_math.py.
"""

from __future__ import annotations

import pytest

from diloco_measured.measurement.wire import WireSnapshot, account, parse_proc_net_dev

# Trimmed to the columns that matter for parse_proc_net_dev (rx bytes = field 0, tx bytes =
# field 8); real /proc/net/dev has two more header lines and wider padding, which the parser
# ignores (it only reads whitespace-split fields after the colon).
SAMPLE_PROC_NET_DEV = """\
    lo:  123456     100    0 0 0 0 0 0   123456     100    0 0 0 0 0 0
  ens5: 9876543210 654321    0 0 0 0 0 12 1234567890  111222    0 0 0 0 0 0
"""


@pytest.mark.unit
def test_parse_proc_net_dev_extracts_rx_tx_for_named_iface():
    rx, tx = parse_proc_net_dev(SAMPLE_PROC_NET_DEV, "ens5")
    assert rx == 9876543210
    assert tx == 1234567890


@pytest.mark.unit
def test_parse_proc_net_dev_raises_for_missing_iface():
    with pytest.raises(KeyError):
        parse_proc_net_dev(SAMPLE_PROC_NET_DEV, "eth9")


@pytest.mark.unit
def test_parse_proc_net_dev_handles_loopback_too():
    rx, tx = parse_proc_net_dev(SAMPLE_PROC_NET_DEV, "lo")
    assert rx == 123456
    assert tx == 123456


@pytest.mark.unit
def test_account_computes_measured_bytes_and_overhead_ratio():
    before = WireSnapshot(per_node_bytes={"node0": 1000, "node1": 2000}, taken_at_s=0.0)
    after = WireSnapshot(per_node_bytes={"node0": 5000, "node1": 6000}, taken_at_s=10.0)
    # raw measured = (5000-1000) + (6000-2000) = 8000

    result = account(
        before, after, predicted_bytes_per_rank=1000, tokens_processed=100, idle_baseline_bytes=0
    )

    assert result["measured_bytes"] == 8000
    assert result["predicted_bytes"] == 2000  # 1000 per rank * 2 nodes
    assert result["overhead_ratio"] == pytest.approx(4.0)
    assert result["bytes_per_training_token_measured"] == pytest.approx(80.0)
    assert result["bytes_per_training_token_predicted"] == pytest.approx(20.0)
    assert result["idle_baseline_bytes"] == 0


@pytest.mark.unit
def test_account_subtracts_idle_baseline():
    before = WireSnapshot(per_node_bytes={"node0": 0}, taken_at_s=0.0)
    after = WireSnapshot(per_node_bytes={"node0": 1000}, taken_at_s=10.0)

    result = account(
        before, after, predicted_bytes_per_rank=100, tokens_processed=10, idle_baseline_bytes=200
    )
    assert result["measured_bytes"] == 800  # 1000 - 200 idle baseline


@pytest.mark.unit
def test_account_rejects_non_positive_predicted_or_tokens():
    before = WireSnapshot(per_node_bytes={"node0": 0}, taken_at_s=0.0)
    after = WireSnapshot(per_node_bytes={"node0": 1000}, taken_at_s=1.0)

    with pytest.raises(ValueError):
        account(before, after, predicted_bytes_per_rank=0, tokens_processed=10)
    with pytest.raises(ValueError):
        account(before, after, predicted_bytes_per_rank=100, tokens_processed=0)
