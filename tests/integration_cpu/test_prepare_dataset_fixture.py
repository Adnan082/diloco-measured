"""Loads the COMMITTED fixture shard (tests/fixtures/dataset/, see GENERATED.md) -- exercises
the shard-file/manifest contract end to end with zero network access and zero dependency on
the `dataprep` extra being installed (only `numpy`, already a base dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from infra.prepare_dataset import sha256_file

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "dataset"


@pytest.fixture
def manifest() -> dict:
    return json.loads((FIXTURE_DIR / "manifest.json").read_text())


@pytest.mark.integration_cpu
def test_manifest_shard_matches_file_on_disk(manifest):
    assert len(manifest["shards"]) == 1
    entry = manifest["shards"][0]
    path = FIXTURE_DIR / entry["filename"]
    assert path.is_file()
    assert sha256_file(path) == entry["sha256"], "fixture shard must match its own manifest"


@pytest.mark.integration_cpu
def test_shard_shape_matches_manifest_counts(manifest):
    entry = manifest["shards"][0]
    array = np.load(FIXTURE_DIR / entry["filename"])
    assert array.shape == (entry["n_sequences"], manifest["seq_len"])
    assert array.size == entry["n_tokens"]


@pytest.mark.integration_cpu
def test_shard_dtype_fits_the_manifest_vocab_size(manifest):
    array = np.load(FIXTURE_DIR / manifest["shards"][0]["filename"])
    assert array.dtype == np.uint16  # gpt2 vocab_size=50257 fits uint16
    assert int(array.max()) < manifest["vocab_size"]


@pytest.mark.integration_cpu
def test_manifest_is_explicitly_not_a_real_dataset(manifest):
    """GENERATED.md documents this; the manifest itself must also make it unambiguous which
    real dataset (if any) this shard was drawn from -- it wasn't.
    """
    assert manifest["dataset_name"] == "fixture-only-not-a-real-dataset"
