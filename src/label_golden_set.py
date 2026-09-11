"""
Interactive local labeler for the Golden Evaluation Set.

Fully offline and human-only:

* it never calls Gemini or any other AI model,
* it never generates or suggests labels.

Usage::

    python -m src.label_golden_set                  # resume at first unlabeled
    python -m src.label_golden_set --record GOLDEN-0010
    python -m src.label_golden_set --list           # just show progress
    python -m src.label_golden_set --export evaluation   # merge into a copy

Inside the labeler you can type, at the intent / escalation prompts:

  - a number to select that option (exactly one per record),
  - ``q`` (quit)  to save progress and exit,
  - ``b`` (back)  to go back to the previous record,
  - ``s`` (skip)  to leave the current record unlabeled and move on.

The optional notes prompt takes free-form text; leave it empty for no note.
Progress is saved to disk after every labeled record, so closing the program
never loses work.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import labeling as L

QUIT_COMMANDS = {"quit", "q", "exit"}
BACK_COMMANDS = {"back", "b"}
SKIP_COMMANDS = {"skip", "s"}


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _read_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def _prompt_option(
    name: str,
    options: List[str],
) -> Tuple[str, Optional[str]]:
    """Prompt until a valid option number or a navigation command is entered.

    Returns ``("ok", value)`` on a valid selection, or ``("quit" | "back" |
    "skip", None)`` for a navigation command.
    """
    commands = "q=quit b=back s=skip"
    while True:
        raw = _read_line(f"  [{name} (1-{len(options)} | {commands})] > ")
        low = raw.lower()
        if low in QUIT_COMMANDS:
            return ("quit", None)
        if low in BACK_COMMANDS:
            return ("back", None)
        if low in SKIP_COMMANDS:
            return ("skip", None)
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return ("ok", options[int(raw) - 1])
        print(
            f"  Please enter a number 1..{len(options)} (or "
            f"q/b/s). Got: {raw!r}"
        )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _indent(text: str, prefix: str = "  ") -> str:
    if not text:
        return prefix + "(none)"
    return "\n".join(
        prefix + (line if line else "") for line in str(text).splitlines()
    )


def _print_record(
    record: Dict[str, Any],
    index: int,
    total: int,
    store: Dict[str, Any],
) -> None:
    entry = store.get("labels", {}).get(record.get("golden_id"))
    status = "UNLABELED" if not L.is_complete(entry) else "PREVIOUSLY LABELED"
    bar = "-" * 60
    print()
    print(f"Record {index + 1}/{total}  {record['golden_id']}  "
          f"(conversation_id={record.get('conversation_id')})  [{status}]")
    print(bar)
    print("CUSTOMER MESSAGE")
    print(_indent(record.get("customer_message", "")))
    print()
    print("CONVERSATION CONTEXT")
    print(_indent(record.get("conversation_context", "")))
    print()
    responses = record.get("microsoft_responses", []) or []
    print("MICROSOFT HISTORICAL RESPONSES")
    if responses:
        for i, res in enumerate(responses, 1):
            print(f"  {i}) {res}")
    else:
        print("  (none)")
    print()
    if L.is_complete(entry):
        print("CURRENTLY SAVED")
        print(f"  intent_label      : {entry.get('intent_label', '')}")
        print(f"  escalation_label  : {entry.get('escalation_label', '')}")
        print(f"  notes             : {entry.get('notes', '') or '(none)'}")
    print(bar)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def _next_unlabeled_after(
    records: List[Dict[str, Any]],
    store: Dict[str, Any],
    index: int,
) -> int:
    """First unlabeled record strictly after ``index``, else ``len(records)``."""
    table = store.get("labels", {})
    for j in range(index + 1, len(records)):
        if not L.is_complete(table.get(records[j].get("golden_id"))):
            return j
    return len(records)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def run_interactive(
    payload: Dict[str, Any],
    store: Dict[str, Any],
    labels_path: str,
    start_index: int,
    intents: List[str],
) -> int:
    records = L.golden_records(payload)
    total = len(records)
    idx = start_index

    while True:
        if idx >= total:
            L.save_labels(store, labels_path)
            print()
            print(f"Labeling complete: all {total} records are labeled.")
            print(L.build_summary_text(payload, store))
            return 0

        record = records[idx]
        _print_record(record, idx, total, store)

        print("INTENT (choose exactly one):")
        for i, name in enumerate(intents, 1):
            print(f"  {i}) {name}")
        print("ESCALATION (choose exactly one):")
        for i, esc in enumerate(L.ESCALATION_VALUES, 1):
            print(f"  {i}) {esc}")

        action, intent_value = _prompt_option("intent", intents)
        if action == "quit":
            return _save_and_exit(store, labels_path, payload)
        if action == "back":
            idx = max(0, idx - 1)
            continue
        if action == "skip":
            idx = _next_unlabeled_after(records, store, idx)
            continue

        action, esc_value = _prompt_option("escalation", L.ESCALATION_VALUES)
        if action == "quit":
            return _save_and_exit(store, labels_path, payload)
        if action == "back":
            idx = max(0, idx - 1)
            continue
        if action == "skip":
            idx = _next_unlabeled_after(records, store, idx)
            continue

        notes = _read_line("  [notes (empty for none)] > ")
        if notes.lower() in QUIT_COMMANDS:
            return _save_and_exit(store, labels_path, payload)
        if notes.lower() in SKIP_COMMANDS:
            idx = _next_unlabeled_after(records, store, idx)
            continue

        try:
            L.set_label(
                store,
                record,
                intent_value,
                esc_value,
                notes=notes,
                valid_intents=intents,
            )
        except ValueError as exc:
            print(f"  Invalid label: {exc}")
            continue
        L.save_labels(store, labels_path)
        print(f"  [saved] {record['golden_id']}: "
              f"intent={intent_value}, escalation={esc_value} "
              f"({idx + 1}/{total})")
        idx = _next_unlabeled_after(records, store, idx)


def _save_and_exit(
    store: Dict[str, Any],
    labels_path: str,
    payload: Dict[str, Any],
) -> int:
    L.save_labels(store, labels_path)
    print()
    print(f"Progress saved to {labels_path}")
    print(L.build_summary_text(payload, store))
    return 0


# ---------------------------------------------------------------------------
# Non-interactive modes
# ---------------------------------------------------------------------------

def print_progress(payload: Dict[str, Any], store: Dict[str, Any]) -> None:
    records = L.golden_records(payload)
    total = len(records)
    resuming = L.first_unlabeled_index(records, store)
    print(L.build_summary_text(payload, store))
    if resuming < total:
        rec = records[resuming]
        print(f"Resume         : next unlabeled is record {resuming + 1}/{total} "
              f"({rec['golden_id']}, conversation_id={rec.get('conversation_id')})")
    else:
        print("Resume         : everything is already labeled.")


def export(payload: Dict[str, Any], store: Dict[str, Any], out_dir: str) -> int:
    dest = L.export_labeled_set(
        payload,
        store,
        output_path=str(Path(out_dir) / "golden_set.labeled.json"),
        labels_source=str(L.DEFAULT_LABELS_PATH),
    )
    labeled = L.count_labeled(store)
    print(f"Exported {labeled} labels into a copy at: {dest}")
    print("The original golden_set.json was not modified.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-path", default=str(L.DEFAULT_GOLDEN_SET_PATH),
        help="Path to the Golden Set JSON (default: evaluation/golden_set.json)",
    )
    parser.add_argument(
        "--labels-path", default=str(L.DEFAULT_LABELS_PATH),
        help="Where labels are stored (default: evaluation/golden_set.labels.json)",
    )
    parser.add_argument(
        "--record", default=None,
        help="Start labeling at this golden_id (e.g. GOLDEN-0010)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print progress and exit without labeling",
    )
    parser.add_argument(
        "--export", metavar="DIR", default=None,
        help="Merge labels into a labeled copy inside DIR and exit",
    )
    args = parser.parse_args(argv)

    payload = L.load_golden_set(args.golden_path)
    store = L.load_labels(args.labels_path)
    records = L.golden_records(payload)
    store["count_total"] = len(records)
    if not Path(args.labels_path).exists():
        store["source"] = args.golden_path
    intents = L.load_intent_names()

    if args.list:
        print_progress(payload, store)
        return 0

    if args.export:
        return export(payload, store, args.export)

    if args.record:
        matches = [i for i, r in enumerate(records)
                   if r.get("golden_id") == args.record]
        if not matches:
            print(f"Unknown golden_id: {args.record}", file=sys.stderr)
            return 1
        start = matches[0]
        print(f"Starting at {args.record} ({start + 1}/{len(records)})")
    else:
        start = L.first_unlabeled_index(records, store)
        if start >= len(records):
            print(f"Nothing to do: all {len(records)} records are already labeled.")
            return 0
        rec = records[start]
        print(f"Resuming at record {start + 1}/{len(records)} "
              f"({rec['golden_id']}), the first unlabeled record.")

    return run_interactive(payload, store, args.labels_path, start, intents)


if __name__ == "__main__":
    sys.exit(main())