"""Exercises infra/prepare_dataset.py::prepare_shards() end to end against a REAL tokenizer
(gpt2, small and public) fed a handful of in-memory documents -- not `iter_dataset_documents`,
which streams from HuggingFace Hub and would pull real (large) FineWeb-Edu/C4 data; that path
is Day-0, on-cluster work, not something this dev environment's test suite should trigger.

Requires: the `dataprep` optional dependency group (`pip install -e .[dataprep]`) AND network
access to fetch gpt2's tokenizer files (~3 MB, cached by `transformers` after the first call).
Both are treated as "this test may not be runnable everywhere," not "this project doesn't
build the real thing" -- see test_prepare_dataset_fixture.py for the network-free path that
exercises the same shard/manifest contract against an already-committed real fixture.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers", reason="dataprep extra not installed")

from infra.prepare_dataset import prepare_shards  # noqa: E402


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:  # network unavailable, HF Hub unreachable, etc.
        pytest.skip(f"could not fetch gpt2 tokenizer (network required): {e}")


DOCUMENTS = [
    "Semi-synchronous training synchronizes every H local steps instead of every step. " * 5,
    "A token bucket filter shapes egress bandwidth on the Linux qdisc layer. " * 5,
]


@pytest.mark.integration_cpu
def test_prepare_shards_end_to_end_with_a_real_tokenizer(tmp_path, gpt2_tokenizer):
    manifest = prepare_shards(
        documents_per_shard=[lambda: DOCUMENTS],
        tokenizer=gpt2_tokenizer,
        seq_len=32,
        output_dir=tmp_path,
        dataset_name="test-only",
        split="test",
    )
    assert manifest["total_tokens"] > 0
    assert manifest["total_tokens"] % 32 == 0  # every sequence is exactly seq_len long
    assert (tmp_path / manifest["shards"][0]["filename"]).is_file()
    assert (tmp_path / "manifest.json").is_file()


@pytest.mark.integration_cpu
def test_two_shards_from_disjoint_document_sets_get_independent_checksums(tmp_path, gpt2_tokenizer):
    manifest = prepare_shards(
        documents_per_shard=[lambda: [DOCUMENTS[0]], lambda: [DOCUMENTS[1]]],
        tokenizer=gpt2_tokenizer,
        seq_len=16,
        output_dir=tmp_path,
        dataset_name="test-only",
        split="test",
    )
    assert len(manifest["shards"]) == 2
    checksums = {s["sha256"] for s in manifest["shards"]}
    assert len(checksums) == 2, "different content must not collide to the same checksum"


@pytest.mark.integration_cpu
def test_raises_when_documents_too_short_for_seq_len(tmp_path, gpt2_tokenizer):
    with pytest.raises(ValueError, match="0 full sequences"):
        prepare_shards(
            documents_per_shard=[lambda: ["short"]],
            tokenizer=gpt2_tokenizer,
            seq_len=10_000,
            output_dir=tmp_path,
            dataset_name="test-only",
            split="test",
        )
