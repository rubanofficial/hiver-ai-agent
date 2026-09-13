"""
Run the LLM-as-a-Judge evaluation for the AI support agent.

Judge input is built ONLY from customer_message, conversation_context,
predicted_intent, retrieved_evidence and generated_reply -- human Golden Set
labels are never read into the judge input (see evaluation/judge.py).

Cost/rate-limit safety:
* By default at most --limit (default 10) examples are judged on a single run.
  Pass --all or --limit 0 to judge every example explicitly.
* --golden-ids selects specific Golden IDs to judge.
* Every judged example is cached in evaluation/judge_results/cache/ so reruns
  do not call Gemini again for already-judged records.
* --dry-run lists what would be judged without calling Gemini.

Usage::

    export GEMINI_API_KEY=...
    export GEMINI_MODEL=gemini-2.0-flash

    python -m evaluation.run_judge --predictions evaluation/predictions.json --limit 10
    python -m evaluation.run_judge --predictions evaluation/predictions.json --golden-ids GOLDEN-0001 GOLDEN-0002
    python -m evaluation.run_judge --predictions evaluation/predictions.json --all
    python -m evaluation.run_judge --predictions evaluation/predictions.json --human-grounding evaluation/judge_human_review.csv

Human agreement:
* After judging, a human-review CSV (evaluation/judge_human_review.csv) is
  written with EMPTY human_grounding / human_notes.  Fill the binary label
  SUPPORTED / UNSUPPORTED for a subset of examples (independent human
  annotation) and pass --human-grounding to compute raw agreement and Cohen's
  kappa with the LLM grounding decision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

from evaluation.judge import (
    DEFAULT_MODEL,
    JudgeError,
    agreement_from_human_review,
    load_cache,
    load_judge_examples,
    input_fingerprint,
    run_judgments,
    write_human_review_csv,
)
from evaluation.evaluate import (
    DEFAULT_GOLDEN_SET_PATH, DEFAULT_INTENTS_YAML, EvaluationError,
)

DEFAULT_PREDICTIONS_PATH = Path(__file__).resolve().parent / "predictions.json"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "judge_results"
DEFAULT_CACHE_DIR = DEFAULT_RESULTS_DIR / "cache"
DEFAULT_HUMAN_REVIEW_PATH = (
    Path(__file__).resolve().parent / "judge_human_review.csv"
)
DEFAULT_LIMIT = 10
SCHEMA_VERSION = "1.0.0"

RATE_LIMIT_GUARD = (
    "The judge will NOT silently score the whole set. Use --limit N, "
    "--golden-ids or --all."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def select_examples(
    examples: List[Dict[str, Any]],
    limit: int,
    golden_ids: List[str],
    all_examples: bool,
) -> List[Dict[str, Any]]:
    """Choose the subset to judge (cost/rate-limit protection)."""
    if golden_ids:
        by_id = {e["golden_id"]: e for e in examples}
        missing = [gid for gid in golden_ids if gid not in by_id]
        if missing:
            raise JudgeError(
                f"Unknown golden_id(s) for judging: {missing}. "
                "They are not in the Golden Set."
            )
        selected = [by_id[gid] for gid in golden_ids]
    elif all_examples or limit <= 0:
        selected = list(examples)
    else:
        selected = examples[:limit]
    if not selected:
        raise JudgeError("No examples selected to judge.")
    return selected


def build_results_document(results: List[Dict[str, Any]], model: str,
                           from_cache: Dict[str, int]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "evaluation/run_judge.py",
        "generated_at": _now_iso(),
        "model": model,
        "count": len(results),
        "examples_fresh": from_cache["fresh"],
        "examples_from_cache": from_cache["cached"],
        "results": results,
    }


def write_results_document(results_dir: str, document: Dict[str, Any]) -> Path:
    path = Path(results_dir) / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    return path


def run_judge(
    predictions_path: str,
    golden_set_path: str = str(DEFAULT_GOLDEN_SET_PATH),
    labels_path: Optional[str] = None,
    intents_yaml: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    golden_ids: Optional[List[str]] = None,
    all_examples: bool = False,
    cache_dir: str = str(DEFAULT_CACHE_DIR),
    results_dir: str = str(DEFAULT_RESULTS_DIR),
    human_review_path: str = str(DEFAULT_HUMAN_REVIEW_PATH),
    human_grounding_path: Optional[str] = None,
    client: Any = None,
) -> Dict[str, Any]:
    """Judge a subset of examples and write results + human-review CSV.

    ``client`` is injectable for tests (a fake Gemini client).  When omitted a
    real ``GeminiJudgeClient`` is built from the environment.
    """
    examples = load_judge_examples(
        golden_set_path, predictions_path, labels_path=labels_path,
        intents_yaml=intents_yaml,
    )
    selected = select_examples(examples, limit, golden_ids or [], all_examples)
    model_name = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL

    if client is None:
        from evaluation.judge import GeminiJudgeClient
        client = GeminiJudgeClient(model_name=model_name)

    results, stats = run_judgments(
        client, selected, cache_dir=cache_dir, model_name=model_name
    )
    document = build_results_document(results, model_name, from_cache=stats)
    written = [write_results_document(results_dir, document)]
    written.append(write_human_review_csv(human_review_path, results))

    agreement = None
    if human_grounding_path:
        agreement = agreement_from_human_review(results, human_grounding_path)
        agreement_path = Path(results_dir) / "agreement.json"
        with open(agreement_path, "w", encoding="utf-8") as f:
            json.dump(agreement, f, indent=2, ensure_ascii=False)
        written.append(agreement_path)

    return {
        "model": model_name,
        "stats": stats,
        "written": [str(p) for p in written],
        "agreement": agreement,
        "results_dir": results_dir,
    }


def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation for the AI support agent."
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Agent predictions JSON (predicted_reply + retrieved_evidence).",
    )
    parser.add_argument("--golden-set", default=str(DEFAULT_GOLDEN_SET_PATH))
    parser.add_argument("--intents-yaml", default=str(DEFAULT_INTENTS_YAML))
    parser.add_argument("--labels-path", default=None, dest="labels_path")
    parser.add_argument(
        "--model", default=os.environ.get("GEMINI_MODEL"),
        help="Gemini model (default: GEMINI_MODEL or "
             f"{DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Max examples to judge per run (default {DEFAULT_LIMIT}; "
             "0 = all).",
    )
    parser.add_argument(
        "--golden-ids", nargs="*", default=None,
        help="Judge only these specific Golden IDs.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Judge every example (overrides --limit).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be judged without calling Gemini.",
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--human-review", default=str(DEFAULT_HUMAN_REVIEW_PATH),
        help="Where to write the human-review CSV.",
    )
    parser.add_argument(
        "--human-grounding", default=None,
        help="Path to a human-filled review CSV to compute agreement.",
    )
    args = parser.parse_args(argv)

    try:
        examples = load_judge_examples(
            args.golden_set, args.predictions,
            labels_path=args.labels_path, intents_yaml=args.intents_yaml,
        )
        selected = select_examples(
            examples, args.limit, args.golden_ids or [], args.all
        )

        if args.dry_run:
            cache = load_cache(args.cache_dir)
            print("[judge] DRY RUN - would call Gemini for these examples:")
            planned = 0
            for e in selected:
                from_cache_file = cache.get(str(e["golden_id"]))
                valid = from_cache_file is not None and (
                    from_cache_file.get("input_fingerprint")
                    == input_fingerprint(e)
                )
                tag = "cached (no call)" if valid else "NEW CALL"
                if not valid:
                    planned += 1
                print(f"  {e['golden_id']}: {tag}")
            print(f"[judge] planned Gemini calls: {planned}")
            print(RATE_LIMIT_GUARD)
            return 0
    except (JudgeError, EvaluationError) as exc:
        print(f"[judge] Cannot run judge.\n{exc}", file=sys.stderr)
        print(RATE_LIMIT_GUARD, file=sys.stderr)
        return 1

    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "[judge] GEMINI_API_KEY is not set. Set the GEMINI_API_KEY "
            "environment variable before running the judge.",
            file=sys.stderr,
        )
        return 1

    if not args.golden_ids and args.limit == DEFAULT_LIMIT and not args.all:
        print(
            f"[judge] Defaulting to a maximum of {DEFAULT_LIMIT} examples to "
            "limit cost. Pass --all (or --limit 0) to judge every example.",
        )

    try:
        outcome = run_judge(
            predictions_path=args.predictions,
            golden_set_path=args.golden_set,
            labels_path=args.labels_path,
            intents_yaml=args.intents_yaml,
            model=args.model,
            limit=args.limit,
            golden_ids=args.golden_ids,
            all_examples=args.all,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir,
            human_review_path=args.human_review,
            human_grounding_path=args.human_grounding,
        )
    except JudgeError as exc:
        print(f"[judge] Judge run failed.\n{exc}", file=sys.stderr)
        return 1

    stats = outcome["stats"]
    print(f"[judge] model                     : {outcome['model']}")
    print(f"[judge] examples judged           : {stats['examples']}")
    print(f"[judge] fresh Gemini calls        : {stats['fresh']}")
    print(f"[judge] served from cache         : {stats['cached']}")
    print("[judge] written:")
    for path in outcome["written"]:
        print(f"  - {path}")
    if outcome["agreement"]:
        a = outcome["agreement"]
        print(f"[judge] human-reviewed examples  : {a['human_reviewed_examples']}")
        print(f"[judge] raw agreement           : {a['raw_agreement']:.4f}")
        print(f"[judge] Cohen's kappa           : {a['cohen_kappa']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())