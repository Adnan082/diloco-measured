# Fixture dataset shard

`fixture_shard_0000.npy` + `manifest.json` in this directory were produced by the REAL
`infra/prepare_dataset.py` pipeline (`prepare_shards()`, real `pack_tokens()`/
`write_shard_npy()`/`sha256_file()`), using the real, small, public `gpt2` tokenizer from
HuggingFace — not fabricated token ids.

**This is not a sample of FineWeb-Edu, C4, or any other real training corpus.** The four
source documents are sentences written for this repository (see the generator invocation
below), repeated to exceed one `seq_len=64` block. `manifest.json`'s `license_note` field is
the pipeline's standard boilerplate (it always cites FineWeb-Edu/C4, §40 Q7/ADR-019, since
that's the license note real Day-0 runs need) — it does not apply to this fixture, which has
no license concerns at all: nobody's corpus is being redistributed here, only four sentences
this project wrote for itself. `dataset_name: "fixture-only-not-a-real-dataset"` in the
manifest is the actual disambiguator.

**Purpose:** lets `tests/integration_cpu/` exercise the loading/shape/checksum path against a
real `.npy` shard and a real manifest with zero network access at test time, and zero
dependency on the `dataprep` extra (`transformers`/`datasets`) being installed.

**Regenerate with** (requires `pip install -e .[dataprep]`):
```python
from pathlib import Path
from transformers import AutoTokenizer
from infra.prepare_dataset import prepare_shards

DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog. " * 20,
    "DiLoCo trains inner steps locally before an outer synchronization. " * 20,
    "Bandwidth shaping with tc and tbf makes bytes on the wire a controlled variable. " * 20,
    "Compute utilization measures wall clock time spent computing versus blocked on sync. " * 20,
]
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prepare_shards(
    documents_per_shard=[lambda: DOCUMENTS],
    tokenizer=tokenizer,
    seq_len=64,
    output_dir=Path("tests/fixtures/dataset"),
    dataset_name="fixture-only-not-a-real-dataset",
    dataset_config=None,
    split="fixture",
    shard_name_prefix="fixture_shard",
)
```
