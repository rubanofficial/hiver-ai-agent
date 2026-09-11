"""
Local human-labeling workflow for the Golden Evaluation Set.

This module provides the file I/O, validation, and resume helpers used by the
interactive labeling CLI (``src.label_golden_set``).  It is 100% offline:

* it never calls Gemini or any other AI model,
* it never generates or suggests a label,
* it only reads ``evaluation/golden_set.json`` and writes the labels file.

Labels are stored SEPARATELY from the source data (``golden_set.labels.json``)
so the original Golden Set is never modified and every label keeps full
conversation traceability (``golden_id`` + ``conversation_id``).  An export
helper merges the labels into a *copy* of the set when labeling is finished.
"""

import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .golden_set import load_intent_taxonomy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN_SET_PATH = PROJECT_ROOT / "evaluation" / "golden_set.json"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "evaluation" / "golden_set.labels.json"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "evaluation"

LABELS_SCHEMA_VERSION = "1.0.0"
HUMAN_LABELER = "human"

INTENT_FIELD = "intent_label"
ESCALATION_FIELD = "escalation_label"
NOTES_FIELD = "notes"
LABEL_FIELDS = [INTENT_FIELD, ESCALATION_FIELD, NOTES_FIELD]

# Valid escalation values; must match the runtime agent's enum exactly.
ESCALATION_VALUES = ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_golden_set(path: str = str(DEFAULT_GOLDEN_SET_PATH)) -> Dict[str, Any]:
    """Read the canonical Golden Set JSON and return its payload dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden set not found: {p}")
    # utf-8-sig tolerates a UTF-8 BOM (e.g. from editors on Windows).
    with open(p, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def golden_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of records from a loaded Golden Set payload."""
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Golden set payload has no 'records' list.")
    return records


def empty_labels_store(source: str = str(DEFAULT_GOLDEN_SET_PATH)) -> Dict[str, Any]:
    """A fresh, empty labels store (used when no labels file exists yet)."""
    return {
        "schema_version": LABELS_SCHEMA_VERSION,
        "source": source,
        "label_fields": list(LABEL_FIELDS),
        "count_total": 0,
        "count_labeled": 0,
        "labels": {},
    }


def load_labels(path: str = str(DEFAULT_LABELS_PATH)) -> Dict[str, Any]:
    """Load the labels store; returns an empty store when no file exists yet."""
    p = Path(path)
    if not p.exists():
        return empty_labels_store(source=str(p))
    with open(p, "r", encoding="utf-8-sig") as f:
        store = json.load(f)
    if not isinstance(store, dict) or not isinstance(store.get("labels"), dict):
        raise ValueError(f"Invalid labels store: missing 'labels' map in {p}")
    store.setdefault("schema_version", LABELS_SCHEMA_VERSION)
    store.setdefault("label_fields", list(LABEL_FIELDS))
    store.setdefault("labels", {})
    if not isinstance(store.get("label_fields"), list):
        store["label_fields"] = list(LABEL_FIELDS)
    store["count_labeled"] = _count_complete(store["labels"])
    return store


# ---------------------------------------------------------------------------
# Saving (incremental, atomic)
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically (temp file + os.replace).

    This guarantees a partially-written JSON can never be left behind, so an
    interrupted labeling session never corrupts the saved progress.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def save_labels(
    store: Dict[str, Any],
    path: str = str(DEFAULT_LABELS_PATH),
) -> Path:
    """Persist the labels store, recomputing ``count_labeled`` first."""
    p = Path(path)
    if not isinstance(store.get("labels"), dict):
        raise ValueError("Cannot save: labels store is missing the 'labels' map.")
    store["count_labeled"] = _count_complete(store["labels"])
    _atomic_write_json(p, store)
    return p


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_label(
    intent_label: str,
    escalation_label: str,
    notes: Optional[str] = None,
    valid_intents: Optional[List[str]] = None,
) -> List[str]:
    """Return a list of validation errors (empty list means valid).

    A label is valid when exactly one taxonomy intent and exactly one
    escalation value are provided.  ``notes`` is optional.
    """
    errors: List[str] = []
    if intent_label is None or str(intent_label).strip() == "":
        errors.append("intent_label is required (exactly one intent).")
    elif valid_intents and intent_label not in valid_intents:
        joined = ", ".join(valid_intents)
        errors.append(
            f"intent_label '{intent_label}' is not one of the "
            f"{len(valid_intents)} taxonomy intents ({joined})."
        )
    if escalation_label not in ESCALATION_VALUES:
        errors.append(
            f"escalation_label must be exactly one of {ESCALATION_VALUES}."
        )
    if notes is not None and not isinstance(notes, str):
        errors.append("notes must be a string.")
    return errors


def set_label(
    store: Dict[str, Any],
    record: Dict[str, Any],
    intent_label: str,
    escalation_label: str,
    notes: Optional[str] = None,
    valid_intents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Record one human label inside ``store`` (mutates and returns entry).

    Raises ``ValueError`` when the label is invalid; no AI and no automated
    suggestion is involved at any point.
    """
    golden_id = record.get("golden_id", "")
    errors = validate_label(intent_label, escalation_label, notes, valid_intents)
    if errors:
        raise ValueError("; ".join(errors))

    table = store.setdefault("labels", {})
    entry = table.get(golden_id, {})
    entry.update({
        "golden_id": golden_id,
        "conversation_id": record.get("conversation_id"),
        INTENT_FIELD: intent_label,
        ESCALATION_FIELD: escalation_label,
        NOTES_FIELD: "" if notes is None else str(notes),
        "labeled_by": HUMAN_LABELER,
        "labeled_at": entry.get("labeled_at", _now_iso()),
        "modified_at": _now_iso(),
    })
    table[golden_id] = entry
    store["count_labeled"] = _count_complete(table)
    return entry


# ---------------------------------------------------------------------------
# Completeness & progress
# ---------------------------------------------------------------------------

def is_complete(entry: Any) -> bool:
    """A label entry counts once BOTH intent and escalation are filled in."""
    if not isinstance(entry, dict):
        return False
    intent = str(entry.get(INTENT_FIELD, "") or "").strip()
    escalation = str(entry.get(ESCALATION_FIELD, "") or "").strip()
    return bool(intent) and escalation in ESCALATION_VALUES


def is_record_labeled(store: Dict[str, Any], golden_id: str) -> bool:
    return is_complete(store.get("labels", {}).get(golden_id))


def _count_complete(table: Dict[str, Any]) -> int:
    return sum(1 for entry in table.values() if is_complete(entry))


def count_labeled(store: Dict[str, Any]) -> int:
    """Number of fully labeled records in the store."""
    return _count_complete(store.get("labels", {}))


def first_unlabeled_index(
    records: List[Dict[str, Any]],
    store: Dict[str, Any],
) -> int:
    """Index of the first record (in Golden Set order) that still needs labels.

    Returns ``len(records)`` when every record is already labeled.  This is
    the resume point used by the CLI.
    """
    table = store.get("labels", {})
    for i, rec in enumerate(records):
        if not is_complete(table.get(rec.get("golden_id"))):
            return i
    return len(records)


def first_unlabeled_record(
    records: List[Dict[str, Any]],
    store: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    i = first_unlabeled_index(records, store)
    return records[i] if i < len(records) else None


# ---------------------------------------------------------------------------
# Merge / export (source data is never modified)
# ---------------------------------------------------------------------------

def merge_labels(
    payload: Dict[str, Any],
    store: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Deep-copy the Golden Set records with labels filled in.

    Unlabeled records keep empty label fields; every other source field
    (customer_message, conversation_context, microsoft_responses,
    source_tweets, metadata, ...) is preserved byte-for-byte.
    """
    records = deepcopy(golden_records(payload))
    table = store.get("labels", {})
    for rec in records:
        entry = table.get(rec.get("golden_id"))
        if is_complete(entry):
            rec[INTENT_FIELD] = entry.get(INTENT_FIELD, "")
            rec[ESCALATION_FIELD] = entry.get(ESCALATION_FIELD, "")
            rec[NOTES_FIELD] = entry.get(NOTES_FIELD, "")
        else:
            rec[INTENT_FIELD] = ""
            rec[ESCALATION_FIELD] = ""
            rec[NOTES_FIELD] = ""
    return records


def export_labeled_set(
    payload: Dict[str, Any],
    store: Dict[str, Any],
    output_path: Optional[str] = None,
    labels_source: Optional[str] = None,
) -> Path:
    """Write a labeled *copy* of the Golden Set to ``golden_set.labeled.json``.

    The original ``golden_set.json`` is left untouched.
    """
    records = merge_labels(payload, store)
    dest = Path(output_path) if output_path else (
        DEFAULT_EVALUATION_DIR / "golden_set.labeled.json"
    )
    copy_payload = {
        "schema_version": payload.get("schema_version", "1.0.0"),
        "generated_by": "src/labeling.py (human-label merge)",
        "source": payload.get("source", ""),
        "labels_source": labels_source or str(DEFAULT_LABELS_PATH),
        "count": len(records),
        "label_fields": list(LABEL_FIELDS),
        "records": records,
    }
    _atomic_write_json(dest, copy_payload)
    return dest


# ---------------------------------------------------------------------------
# Summary / intents
# ---------------------------------------------------------------------------

def load_intent_names(config_path: Optional[str] = None) -> List[str]:
    """Read-only view of the 10 intent names from config/intents.yaml."""
    return [entry.get("name") for entry in load_intent_taxonomy(config_path)]


def build_summary_text(
    payload: Dict[str, Any],
    store: Dict[str, Any],
) -> str:
    """Human-readable progress + label distribution report."""
    records = golden_records(payload)
    total = len(records)
    labeled = count_labeled(store)
    table = store.get("labels", {})

    lines = [
        f"Progress      : {labeled} of {total} labeled "
        f"({total - labeled} remaining)",
        "Intent distribution:",
    ]
    intent_counts: Dict[str, int] = {}
    esc_counts: Dict[str, int] = {}
    for entry in table.values():
        if is_complete(entry):
            intent_counts[entry.get(INTENT_FIELD, "")] = (
                intent_counts.get(entry.get(INTENT_FIELD, ""), 0) + 1
            )
            esc_counts[entry.get(ESCALATION_FIELD, "")] = (
                esc_counts.get(entry.get(ESCALATION_FIELD, ""), 0) + 1
            )
    if intent_counts:
        for name in sorted(intent_counts):
            lines.append(f"  - {name}: {intent_counts[name]}")
    else:
        lines.append("  - (none yet)")
    lines.append("Escalation distribution:")
    for value in ESCALATION_VALUES:
        lines.append(f"  - {value}: {esc_counts.get(value, 0)}")
    return "\n".join(lines)