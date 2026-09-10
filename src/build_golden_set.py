"""
Build the ~200-example Golden Evaluation Set from MicrosoftHelps history.

Usage::

    python -m src.build_golden_set
    python -m src.build_golden_set --target 200 --seed 42 --output evaluation

Pipeline:
    1. Load twcs.csv (once, with the same dtypes used elsewhere).
    2. Reuse ConversationEvidenceBuilder to extract all MicrosoftHelps
       conversations without scanning the dataset repeatedly.
    3. Deterministically sample ~``--target`` context-rich conversations.
    4. Write evaluation/golden_set.json, evaluation/golden_set.csv, and
       evaluation/labeling_guide.md.

No AI model is called. Intent/escalation labels are left empty for humans.
"""

import argparse
import sys
import time

import pandas as pd

from src.evidence import ConversationEvidenceBuilder
from src.golden_set import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_TARGET_SIZE,
    build_golden_set,
    build_labeling_guide_text,
    load_intent_taxonomy,
    save_golden_set,
    save_labeling_guide,
)


DATASET_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "in_response_to_tweet_id",
    "text",
    "created_at",
]


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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/twcs.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-customer-len", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    print(f"[golden] Loading {args.dataset} ...", flush=True)
    df = load_dataset(args.dataset)
    print(f"[golden] Loaded {len(df):,} rows in {time.time() - t0:.1f}s", flush=True)

    # Reuse the existing evidence builder (single pass over the conversations).
    builder = ConversationEvidenceBuilder(df)
    evidence = builder.build(min_customer_len=args.min_customer_len)
    print(f"[golden] {len(evidence):,} MicrosoftHelps conversations available", flush=True)

    records = build_golden_set(
        evidence,
        target_size=args.target,
        seed=args.seed,
    )
    print(
        f"[golden] Sampled {len(records)} conversations "
        f"(target {args.target}, seed {args.seed})",
        flush=True,
    )

    out_dir = save_golden_set(
        records,
        output_dir=args.output,
        source_label=args.dataset,
    )

    taxonomy = load_intent_taxonomy()
    guide = build_labeling_guide_text(taxonomy)
    save_labeling_guide(guide, output_dir=args.output)
    print(f"[golden] Written to {out_dir}", flush=True)
    print(f"[golden] Labeling guide: {out_dir / 'labeling_guide.md'}", flush=True)


if __name__ == "__main__":
    main()