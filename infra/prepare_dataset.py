"""Dataset acquisition + tokenization (Day-0 plan, CLAUDE.md line ~2123: "Pre-tokenize
FineWeb-Edu on the control node -> S3"). `infra/setup_node.sh` owns syncing already-tokenized
shards FROM S3 to a GPU node's local NVMe (its `mount_nvme`/dataset-sync steps); this script
is what PRODUCES those shards in the first place, run once, on the CPU-only control node,
before any GPU node needs them.

STATUS: the actual tokenizer used to train is a Day-0 decision, not made here.
`vocab_size=32000` in `configs/models/{130m,500m,1b}.toml` is still `[PROPOSED]` pending it
(ADR-026) -- exactly like the `torchft` pin (SS40 Q2), this script does not silently make
that pin. `--tokenizer` accepts any HuggingFace `AutoTokenizer` name/path; nothing about the
packing, checksumming, or manifest logic below depends on which one is eventually chosen.

Licensing (SS40 Q7 / ADR-019): FineWeb-Edu and C4 are both ODC-BY v1.0, subject to Common
Crawl's ToU. This script streams directly from the original HuggingFace source under that
license and writes only local files + an S3 upload of the operator's own tokenized output --
it never redistributes raw dataset content, and neither does this repository (tokenized
shards are not committed; only their checksums and this script live in git).

Sequence packing convention: every document's token stream is followed by one EOS token, all
documents for a shard are concatenated into one flat stream, and the stream is chopped into
fixed-length `seq_len` blocks with the final, incomplete block DROPPED (not padded) -- the
same convention nanoGPT/torchtitan-style pretraining pipelines use, chosen so every training
sequence in a shard is fully "real" tokens with no padding mask to thread through the loader.

DEPENDENCY INJECTION NOTE (same pattern as measurement/train.py's apply_fn/verify_fn/
restore_fn): `prepare_shards()` takes a `documents` iterable/factory rather than calling
`datasets.load_dataset()` itself inline, so the packing/sharding/manifest orchestration is
unit-testable with a handful of in-memory strings and zero network access. The real
HF-Hub-streaming path lives in the separate, thin `iter_dataset_documents()`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

EOS_PLACEHOLDER_NAME = "eos_token_id"  # what we look for on the tokenizer object


def dtype_for_vocab_size(vocab_size: int) -> np.dtype:
    """Smallest unsigned int dtype that can hold every token id in `[0, vocab_size)`.

    `uint16` covers any vocab up to 65,536 (both the 32,000 SentencePiece-style and
    50,257-token GPT-2-style tokenizers used across this project fit); a tiktoken-style
    128k-vocab tokenizer needs `uint32`. Getting this wrong doesn't crash -- it silently
    truncates high token ids -- so it is derived from the tokenizer's actual vocab size,
    never assumed.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
    if vocab_size <= 2**16:
        return np.dtype(np.uint16)
    if vocab_size <= 2**32:
        return np.dtype(np.uint32)
    raise ValueError(f"vocab_size={vocab_size} exceeds uint32 range -- unexpected for an LLM")


def pack_tokens(token_stream: Iterable[int], seq_len: int) -> list[list[int]]:
    """Chop a flat token stream into fixed-length `seq_len` blocks. The final, incomplete
    block is dropped, not padded -- see the module docstring's packing convention.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    sequences: list[list[int]] = []
    current: list[int] = []
    for token_id in token_stream:
        current.append(token_id)
        if len(current) == seq_len:
            sequences.append(current)
            current = []
    return sequences


def iter_document_tokens(
    documents: Iterable[str],
    tokenizer,
    eos_token_id: int,
) -> Iterator[int]:
    """Tokenize each document in turn, yielding its token ids followed by one `eos_token_id`
    -- the flat stream `pack_tokens()` consumes. Deterministic given `tokenizer`, but not
    pure (the tokenizer is a stateful third-party object) -- kept separate from `pack_tokens`
    specifically so the packing logic can be tested with plain integers, no tokenizer needed.
    """
    for doc in documents:
        yield from tokenizer.encode(doc)
        yield eos_token_id


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes -- never loads the whole shard into memory to hash
    it, since shards are meant to be large.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_shard_npy(sequences: list[list[int]], path: Path, dtype: np.dtype) -> None:
    """Write `sequences` (each already exactly `seq_len` long) as one `(n_sequences, seq_len)`
    array -- memory-mappable at load time, which raw Parquet (row-oriented) is not, and this
    project already uses Parquet elsewhere (per-step telemetry) for a different reason.
    """
    if not sequences:
        raise ValueError("refusing to write an empty shard -- 0 sequences is a bug, not data")
    seq_len = len(sequences[0])
    if any(len(s) != seq_len for s in sequences):
        raise ValueError(f"ragged sequences -- expected every row to have length {seq_len}")
    array = np.asarray(sequences, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


@dataclass(frozen=True)
class ShardInfo:
    filename: str
    n_sequences: int
    n_tokens: int
    sha256: str


@dataclass(frozen=True)
class ManifestInputs:
    dataset_name: str
    dataset_config: str | None
    split: str
    tokenizer_name: str
    vocab_size: int
    seq_len: int
    shards: list[ShardInfo] = field(default_factory=list)


def build_manifest(inputs: ManifestInputs, created_at: datetime | None = None) -> dict:
    """Assemble the JSON manifest committed to `results/environment/` (or wherever the
    operator points `--output-dir`) alongside the (gitignored) shard files themselves.
    `run()`'s `dataset_shard_checksum` precondition (measurement/train.py,
    schemas/run_result.v1.json) is meant to be one entry's `sha256` from `shards` below --
    this manifest is the audit trail that number traces back to.
    """
    created_at = created_at or datetime.now(UTC)
    total_tokens = sum(s.n_tokens for s in inputs.shards)
    return {
        "dataset_name": inputs.dataset_name,
        "dataset_config": inputs.dataset_config,
        "split": inputs.split,
        "tokenizer_name": inputs.tokenizer_name,
        "vocab_size": inputs.vocab_size,
        "seq_len": inputs.seq_len,
        "created_at_utc": created_at.isoformat(),
        "total_tokens": total_tokens,
        "total_sequences": sum(s.n_sequences for s in inputs.shards),
        "shards": [
            {
                "filename": s.filename,
                "n_sequences": s.n_sequences,
                "n_tokens": s.n_tokens,
                "sha256": s.sha256,
            }
            for s in inputs.shards
        ],
        "license_note": (
            "Streamed directly from the original HuggingFace source under its own license "
            "(FineWeb-Edu/C4: ODC-BY v1.0, subject to Common Crawl's ToU -- SS40 Q7/ADR-019). "
            "No raw or tokenized dataset content is redistributed by this repository; this "
            "manifest and its checksums are the only committed artifacts."
        ),
    }


DocumentFactory = Callable[[], Iterable[str]]


def prepare_shards(
    documents_per_shard: list[DocumentFactory],
    tokenizer,
    seq_len: int,
    output_dir: Path,
    dataset_name: str,
    dataset_config: str | None = None,
    split: str = "train",
    shard_name_prefix: str = "shard",
) -> dict:
    """Tokenize, pack, and write one shard file per entry in `documents_per_shard`, then
    write and return `manifest.json`. Each entry is a zero-arg callable returning an iterable
    of documents for that shard (a factory, not a plain iterable, so a streaming source can be
    re-entered per shard without the caller having to manage cursors).

    This is the orchestration CLAUDE.md SS19.1 would call "decision logic" -- which shard gets
    which documents, when a shard is considered full is decided by the caller via how it
    partitions `documents_per_shard`, not here) but the actual network/HF-Hub call is not
    inline: see `iter_dataset_documents()` for that, injected by the CLI entrypoint.
    """
    output_dir = Path(output_dir)
    vocab_size = tokenizer.vocab_size
    eos_token_id = getattr(tokenizer, EOS_PLACEHOLDER_NAME, None)
    if eos_token_id is None:
        raise ValueError(
            f"tokenizer {tokenizer!r} has no {EOS_PLACEHOLDER_NAME} -- required to delimit "
            "documents in the packed stream"
        )
    dtype = dtype_for_vocab_size(vocab_size)

    shard_infos: list[ShardInfo] = []
    for i, doc_factory in enumerate(documents_per_shard):
        token_stream = iter_document_tokens(doc_factory(), tokenizer, eos_token_id)
        sequences = pack_tokens(token_stream, seq_len)
        if not sequences:
            raise ValueError(
                f"shard {i}: 0 full sequences produced -- source documents too short for "
                f"seq_len={seq_len}, or the factory yielded nothing"
            )
        filename = f"{shard_name_prefix}_{i:04d}.npy"
        path = output_dir / filename
        write_shard_npy(sequences, path, dtype)
        shard_infos.append(
            ShardInfo(
                filename=filename,
                n_sequences=len(sequences),
                n_tokens=len(sequences) * seq_len,
                sha256=sha256_file(path),
            )
        )

    manifest = build_manifest(
        ManifestInputs(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            tokenizer_name=getattr(tokenizer, "name_or_path", "unknown"),
            vocab_size=vocab_size,
            seq_len=seq_len,
            shards=shard_infos,
        )
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def iter_dataset_documents(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_field: str,
    n_documents: int,
    streaming: bool = True,
) -> Iterator[str]:
    """The one function in this module that actually touches HuggingFace Hub / the network.
    Streams `n_documents` from `dataset_name` without downloading the full corpus to disk
    first -- required for FineWeb-Edu/C4, which are far too large to fetch in whole.
    """
    import datasets  # imported lazily: every other function in this module is import-cheap

    ds = datasets.load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    for i, row in enumerate(ds):
        if i >= n_documents:
            break
        yield row[text_field]


def upload_manifest_and_shards_to_s3(output_dir: Path, s3_uri: str) -> None:
    """Shell out to the `aws` CLI (`aws s3 sync`), matching the rest of `infra/`'s existing
    pattern of driving the AWS CLI directly rather than adding a `boto3` dependency (same
    "fewest moving parts" reasoning as ADR-020's `torchft-nightly` pin).
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"s3_uri must start with s3://, got {s3_uri!r}")
    subprocess.run(
        ["aws", "s3", "sync", str(output_dir), s3_uri, "--exclude", "*.pyc"],
        check=True,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize a HuggingFace streaming dataset into fixed-length shards, checksum "
            "them, and write a manifest. Run once, on the control node, before Day 1."
        )
    )
    parser.add_argument("--dataset", required=True, help='e.g. "HuggingFaceFW/fineweb-edu"')
    parser.add_argument("--dataset-config", default=None, help='e.g. "sample-10BT"')
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--tokenizer", required=True,
        help="Any HuggingFace AutoTokenizer name/path. NOT pinned by this project yet -- "
        "SS40 Q2-style Day-0 decision.",
    )
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument(
        "--n-shards", type=int, required=True, help="One shard per replica is typical (4)."
    )
    parser.add_argument(
        "--documents-per-shard", type=int, required=True,
        help="Streamed document count per shard -- tune against --seq-len and the token "
        "budget; this script does not guess a default.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--s3-uri", default=None, help="If set, sync --output-dir there after.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def make_factory(shard_index: int) -> DocumentFactory:
        # Each shard streams its OWN slice of the dataset via HF `datasets`' `.skip()`, so
        # shards don't silently repeat the same leading documents.
        def factory() -> Iterable[str]:
            import datasets

            ds = datasets.load_dataset(
                args.dataset, args.dataset_config, split=args.split, streaming=True
            )
            ds = ds.skip(shard_index * args.documents_per_shard)
            for i, row in enumerate(ds):
                if i >= args.documents_per_shard:
                    break
                yield row[args.text_field]

        return factory

    manifest = prepare_shards(
        documents_per_shard=[make_factory(i) for i in range(args.n_shards)],
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
    )
    print(json.dumps({"total_tokens": manifest["total_tokens"], "shards": len(manifest["shards"])}))

    if args.s3_uri:
        upload_manifest_and_shards_to_s3(args.output_dir, args.s3_uri)


if __name__ == "__main__":
    main()
