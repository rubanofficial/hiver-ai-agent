"""
Build the FAISS embedding index of MicrosoftHelps historical evidence.

Reads the TWCS dataset, reconstructs MicrosoftHelps conversations with the
existing ``ConversationEvidenceBuilder``, embeds each conversation's customer
messages with the existing ``EmbeddingBuilder``, and writes an
``EmbeddingIndex`` JSON file that ``EvidenceRetriever.from_embedding_json()``
can load.

Usage::

    python -m evaluation.build_embedding_index
    python -m evaluation.build_embedding_index --max-records 200000 --seed 42
    python -m evaluation.build_embedding_index --head 2000 --output output/smoke.json

Only local CPU computation is involved; no AI/API calls are made.
"""

import argparse
import json
import random
import sys
import time

import pandas as pd

from src.evidence import ConversationEvidenceBuilder
from src.embeddings import EmbeddingBuilder


DATASET_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "in_response_to_tweet_id",
    "text",
    "created_at",
]
DEFAULT_OUTPUT = "evaluation/twcs_evidence_index.json"


def load_dataset(path: str) -> pd.DataFrame:
    """Load the TWCS dataset, reading only the needed columns."""
    return pd.read_csv(
        path,
        usecols=DATASET_COLUMNS,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "inbound": "bool",
        },
        low_memory=False,
    )


def load_golden_conversation_ids(path: str) -> set:
    """Read conversation_ids from a Golden Set JSON to hold out of the index."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = {int(rec["conversation_id"]) for rec in data.get("records", [])}
    return ids


def to_record_dicts(evidence) -> list:
    """Convert ConversationEvidence objects to embedding-index record dicts."""
    return [
        {
            "conv_id": ev.conv_id,
            "customer_messages": ev.customer_messages,
            "has_resolution_signal": ev.has_resolution_signal,
        }
        for ev in evidence
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/twcs.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-customer-len", type=int, default=5)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Deterministically cap the number of conversations embedded.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic --max-records sampling.",
    )
    parser.add_argument(
        "--exclude-golden-set",
        default=None,
        metavar="PATH",
        help="Hold out Golden Set conversations (by conversation_id) from the index.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Only embed the first N conversations (smoke test).",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"[index] Loading {args.dataset} ...", flush=True)
    df = load_dataset(args.dataset)
    print(f"[index] Loaded {len(df):,} rows in {time.time() - t0:.1f}s", flush=True)

    builder = ConversationEvidenceBuilder(df)
    evidence = builder.build(min_customer_len=args.min_customer_len)
    print(
        f"[index] {len(evidence):,} MicrosoftHelps conversations available",
        flush=True,
    )
    del df

    if args.exclude_golden_set:
        excluded = load_golden_conversation_ids(args.exclude_golden_set)
        before = len(evidence)
        evidence = [ev for ev in evidence if ev.conv_id not in excluded]
        print(
            f"[index] Excluded {before - len(evidence)} Golden Set conversations "
            f"({len(excluded)} conversation_ids hold out)",
            flush=True,
        )

    if args.head is not None:
        evidence = evidence[: args.head]
        print(f"[index] head limited to {len(evidence)} records", flush=True)

    if args.max_records is not None and len(evidence) > args.max_records:
        rng = random.Random(args.seed)
        evidence = sorted(rng.sample(evidence, args.max_records), key=lambda e: e.conv_id)
        print(
            f"[index] deterministically sampled {len(evidence)} records "
            f"(max-records {args.max_records}, seed {args.seed})",
            flush=True,
        )

    records = to_record_dicts(evidence)
    print(f"[index] Embedding {len(records)} conversations ...", flush=True)

    emb = EmbeddingBuilder()
    index = emb.build_index(records)
    emb.save_index(index, args.output)

    print(f"[index] Done in {time.time() - t0:.1f}s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()