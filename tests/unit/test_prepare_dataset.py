"""Unit tests for infra/prepare_dataset.py's pure logic: packing, dtype selection,
checksumming, shard I/O round-trip, and manifest assembly. No tokenizer, no network -- see
tests/integration_cpu/test_prepare_dataset_tokenizer.py for the real-tokenizer path.
"""

from __future__ import annotations

import numpy as np
import pytest

from infra.prepare_dataset import (
    ManifestInputs,
    ShardInfo,
    build_manifest,
    dtype_for_vocab_size,
    pack_tokens,
    sha256_file,
    write_shard_npy,
)

# ---- dtype_for_vocab_size ---------------------------------------------------------------


@pytest.mark.unit
def test_small_vocab_gets_uint16():
    assert dtype_for_vocab_size(32_000) == np.dtype(np.uint16)


@pytest.mark.unit
def test_vocab_at_uint16_boundary_still_fits():
    assert dtype_for_vocab_size(2**16) == np.dtype(np.uint16)


@pytest.mark.unit
def test_large_vocab_gets_uint32():
    assert dtype_for_vocab_size(128_256) == np.dtype(np.uint32)  # tiktoken-style llama3 vocab


@pytest.mark.unit
def test_rejects_non_positive_vocab():
    with pytest.raises(ValueError, match="must be > 0"):
        dtype_for_vocab_size(0)


# ---- pack_tokens -------------------------------------------------------------------------


@pytest.mark.unit
def test_packs_exact_multiple_with_no_remainder():
    sequences = pack_tokens(range(10), seq_len=5)
    assert sequences == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


@pytest.mark.unit
def test_drops_incomplete_final_block():
    sequences = pack_tokens(range(12), seq_len=5)
    assert sequences == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]  # tokens 10, 11 dropped


@pytest.mark.unit
def test_stream_shorter_than_seq_len_yields_nothing():
    assert pack_tokens([1, 2, 3], seq_len=10) == []


@pytest.mark.unit
def test_rejects_non_positive_seq_len():
    with pytest.raises(ValueError, match="must be > 0"):
        pack_tokens([1, 2, 3], seq_len=0)


# ---- sha256_file / write_shard_npy --------------------------------------------------------


@pytest.mark.unit
def test_sha256_file_is_deterministic_and_content_sensitive(tmp_path):
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"hello world")
    p2.write_bytes(b"hello world!")
    assert sha256_file(p1) == sha256_file(p1)  # deterministic
    assert sha256_file(p1) != sha256_file(p2)  # content-sensitive


@pytest.mark.unit
def test_write_shard_npy_round_trips(tmp_path):
    sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    path = tmp_path / "shard_0000.npy"
    write_shard_npy(sequences, path, dtype=np.dtype(np.uint16))
    loaded = np.load(path)
    assert loaded.shape == (3, 3)
    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, np.array(sequences, dtype=np.uint16))


@pytest.mark.unit
def test_write_shard_npy_rejects_empty_sequences(tmp_path):
    with pytest.raises(ValueError, match="empty shard"):
        write_shard_npy([], tmp_path / "shard.npy", dtype=np.dtype(np.uint16))


@pytest.mark.unit
def test_write_shard_npy_rejects_ragged_sequences(tmp_path):
    with pytest.raises(ValueError, match="ragged"):
        write_shard_npy([[1, 2, 3], [4, 5]], tmp_path / "shard.npy", dtype=np.dtype(np.uint16))


# ---- build_manifest ------------------------------------------------------------------------


@pytest.mark.unit
def test_manifest_totals_sum_across_shards():
    inputs = ManifestInputs(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        split="train",
        tokenizer_name="gpt2",
        vocab_size=50_257,
        seq_len=1024,
        shards=[
            ShardInfo(filename="shard_0000.npy", n_sequences=10, n_tokens=10_240, sha256="aa"),
            ShardInfo(filename="shard_0001.npy", n_sequences=5, n_tokens=5_120, sha256="bb"),
        ],
    )
    manifest = build_manifest(inputs)
    assert manifest["total_tokens"] == 15_360
    assert manifest["total_sequences"] == 15
    assert len(manifest["shards"]) == 2
    assert manifest["shards"][0]["sha256"] == "aa"


@pytest.mark.unit
def test_manifest_carries_license_note_mentioning_no_redistribution():
    inputs = ManifestInputs(
        dataset_name="d", dataset_config=None, split="train",
        tokenizer_name="t", vocab_size=100, seq_len=8, shards=[],
    )
    manifest = build_manifest(inputs)
    assert "redistribute" in manifest["license_note"].lower()


@pytest.mark.unit
def test_manifest_created_at_is_iso8601_and_injectable():
    from datetime import UTC, datetime

    fixed = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)
    inputs = ManifestInputs(
        dataset_name="d", dataset_config=None, split="train",
        tokenizer_name="t", vocab_size=100, seq_len=8, shards=[],
    )
    manifest = build_manifest(inputs, created_at=fixed)
    assert manifest["created_at_utc"] == "2026-08-10T12:00:00+00:00"
