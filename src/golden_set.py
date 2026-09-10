"""
Golden Evaluation Set builder for MicrosoftHelps AI Support Agent.

Creates the raw material for a human-labelled Golden Evaluation Set: a
deterministically sampled subset of MicrosoftHelps conversations, each with
enough surrounding context to understand the customer's issue.

This module performs NO AI calls of any kind.  It never assigns intent or
escalation labels itself - those fields are left EMPTY for a human to fill
in using the labeling guide (evaluation/labeling_guide.md).
"""

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .evidence import ConversationEvidence


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TARGET_SIZE = 200
DEFAULT_SEED = 42
RICH_RATIO = 0.7            # fraction of the sample drawn from "rich" convs
MIN_CURRENT_TURN_LEN = 15   # minimum chars in the customer's final message
REQUIRE_BRAND_RESPONSE = True  # only label conversations Microsoft replied to

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "intents.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation"

BRAND_DEFAULT = "MicrosoftHelps"

LABEL_FIELDS = [
    "intent_label",
    "escalation_label",
]

CSV_FIELDS = [
    "golden_id",
    "conversation_id",
    "customer_message",
    "conversation_context",
    "microsoft_responses",
    "source_tweet_ids",
    "intent_label",
    "escalation_label",
    "notes",
]


# ---------------------------------------------------------------------------
# Taxonomy loader (read-only use of config/intents.yaml)
# ---------------------------------------------------------------------------

def load_intent_taxonomy(config_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Return the intent names and definitions from config/intents.yaml.

    The taxonomy file is only READ here; it is never written or modified.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Intent taxonomy config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    entries = cfg.get("intents") or []
    return [
        {
            "name": e["name"],
            "definition": e.get("definition", ""),
        }
        for e in entries
        if isinstance(e, dict) and e.get("name")
    ]


# ---------------------------------------------------------------------------
# Evidence normalization
# ---------------------------------------------------------------------------

def _evidence_to_dict(rec: Any) -> Dict[str, Any]:
    """Normalize a ConversationEvidence object or plain dict to a dict."""
    if isinstance(rec, dict):
        return rec
    if isinstance(rec, ConversationEvidence):
        return {
            "conv_id": rec.conv_id,
            "tweets": [asdict(t) for t in rec.tweets],
            "customer_messages": list(rec.customer_messages),
            "brand_responses": list(rec.brand_responses),
            "num_turns": rec.num_turns,
            "has_resolution_signal": rec.has_resolution_signal,
            "metadata": dict(rec.metadata),
        }
    raise TypeError(
        f"Unsupported evidence record type: {type(rec).__name__}; "
        "expected ConversationEvidence or dict."
    )


# ---------------------------------------------------------------------------
# Eligibility & richness heuristics
# ---------------------------------------------------------------------------

def _customer_texts(rec: Dict[str, Any]) -> List[str]:
    return [m for m in rec.get("customer_messages", []) if m and m.strip()]


def _brand_texts(rec: Dict[str, Any]) -> List[str]:
    return [b for b in rec.get("brand_responses", []) if b and b.strip()]


def _num_turns(rec: Dict[str, Any]) -> int:
    return int(rec.get("num_turns", len(rec.get("tweets", []))) or 0)


def _richness_score(rec: Dict[str, Any]) -> int:
    """Higher = more conversation context. Used only to order candidates."""
    return (
        len(_customer_texts(rec)) * 10
        + len(_brand_texts(rec)) * 2
        + _num_turns(rec)
    )


def _is_eligible(rec: Dict[str, Any]) -> bool:
    """A conversation is sampleable when the issue is understandable."""
    cust_msgs = _customer_texts(rec)
    if not cust_msgs:
        return False
    current_turn = cust_msgs[-1]
    if len(current_turn.strip()) < MIN_CURRENT_TURN_LEN:
        return False
    if REQUIRE_BRAND_RESPONSE and not _brand_texts(rec):
        return False
    return True


def _is_rich(rec: Dict[str, Any]) -> bool:
    """'Rich' conversations have multiple turns / multiple customer messages."""
    return _num_turns(rec) >= 3 or len(_customer_texts(rec)) >= 2


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_evidence(
    evidence: List[Any],
    target_size: int = DEFAULT_TARGET_SIZE,
    seed: int = DEFAULT_SEED,
    rich_ratio: float = RICH_RATIO,
) -> List[Dict[str, Any]]:
    """Deterministically sample approximately ``target_size`` conversations.

    Sampling strategy (all deterministic, no AI involved):
        1. Keep only eligible conversations (a substantial final customer
           message and at least one MicrosoftHelps response).
        2. Order candidates by conversation richness (most context first),
           using conv_id as a stable tie-breaker.
        3. Split into a "rich" tier (>= 3 turns or >= 2 customer messages)
           and a "basic" tier.  Draw about ``rich_ratio`` of the sample from
           the rich tier and the rest from the basic tier, using a seeded
           RNG so the exact sample is reproducible.
        4. Backfill from the remaining candidates if a tier runs dry, then
           sort the final selection by conversation_id.

    Args:
        evidence: Iterable of ConversationEvidence objects or dicts.
        target_size: Approximate number of conversations to select.
        seed: Random seed for reproducible sampling.
        rich_ratio: Fraction of the sample taken from rich conversations.

    Returns:
        List of selected evidence dicts, sorted by conversation_id.
    """
    records = [_evidence_to_dict(r) for r in evidence]
    eligible = [r for r in records if _is_eligible(r)]
    if not eligible:
        return []

    ordered = sorted(
        eligible,
        key=lambda r: (-_richness_score(r), int(r["conv_id"])),
    )
    rich = [r for r in ordered if _is_rich(r)]
    basic = [r for r in ordered if not _is_rich(r)]

    rng = random.Random(seed)
    result: List[Dict[str, Any]] = []

    rich_target = round(target_size * rich_ratio)
    result.extend(rng.sample(rich, min(rich_target, len(rich))))
    result.extend(rng.sample(basic, min(target_size - len(result), len(basic))))

    # Backfill from whatever remains when a tier (or the pool) is small.
    if len(result) < target_size:
        picked = set(id(r) for r in result)
        remaining = [r for r in ordered if id(r) not in picked]
        result.extend(remaining[: target_size - len(result)])

    result.sort(key=lambda r: int(r["conv_id"]))
    return result[:target_size]


# ---------------------------------------------------------------------------
# Record formatting
# ---------------------------------------------------------------------------

def _turns_before_current(rec: Dict[str, Any]) -> str:
    """Build a readable transcript of the tweets BEFORE the customer's
    final message (the current turn).  Other-brand replies are labeled OTHER.
    """
    tweets = rec.get("tweets", [])
    if not tweets:
        return ""
    brand = rec.get("metadata", {}).get("brand", BRAND_DEFAULT)
    customer_tweets = [t for t in tweets if t.get("inbound")]
    if not customer_tweets:
        return ""
    current = customer_tweets[-1]

    lines: List[str] = []
    for t in tweets:
        if t is current:
            break
        if t.get("inbound"):
            role = "CUSTOMER"
        elif t.get("author_id") == brand:
            role = "MICROSOFT"
        else:
            role = "OTHER"
        lines.append(f"{role} ({t.get('tweet_id')}): {t.get('text', '').strip()}")
    return "\n".join(lines)


def format_golden_record(rec: Dict[str, Any], golden_index: int) -> Dict[str, Any]:
    """Wrap one sampled conversation into a GoldenEval record.

    Label fields (intent_label, escalation_label, notes) are always EMPTY:
    they are ground-truth fields reserved for a human labeler.
    """
    tweets = rec.get("tweets", [])
    customer_tweets = [t for t in tweets if t.get("inbound")]
    current_text = (
        customer_tweets[-1].get("text", "").strip()
        if customer_tweets
        else (_customer_texts(rec)[-1] if _customer_texts(rec) else "")
    )

    return {
        "golden_id": f"GOLDEN-{golden_index:04d}",
        "conversation_id": int(rec["conv_id"]),
        "customer_message": current_text,
        "conversation_context": _turns_before_current(rec),
        "microsoft_responses": _brand_texts(rec),
        "source_tweets": tweets,
        "metadata": rec.get("metadata", {}),
        "intent_label": "",
        "escalation_label": "",
        "notes": "",
    }


def build_golden_set(
    evidence: List[Any],
    target_size: int = DEFAULT_TARGET_SIZE,
    seed: int = DEFAULT_SEED,
    rich_ratio: float = RICH_RATIO,
) -> List[Dict[str, Any]]:
    """Sample context-rich conversations and format them as golden records."""
    sampled = sample_evidence(
        evidence,
        target_size=target_size,
        seed=seed,
        rich_ratio=rich_ratio,
    )
    return [
        format_golden_record(rec, i)
        for i, rec in enumerate(sampled, 1)
    ]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _flatten_to_csv_rows(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Flatten golden records into plain strings for CSV review."""
    rows: List[Dict[str, str]] = []
    for rec in records:
        rows.append({
            "golden_id": rec["golden_id"],
            "conversation_id": str(rec["conversation_id"]),
            "customer_message": rec["customer_message"],
            "conversation_context": rec["conversation_context"],
            "microsoft_responses": " || ".join(rec["microsoft_responses"]),
            "source_tweet_ids": ",".join(
                str(t.get("tweet_id")) for t in rec["source_tweets"]
            ),
            "intent_label": rec.get("intent_label", ""),
            "escalation_label": rec.get("escalation_label", ""),
            "notes": rec.get("notes", ""),
        })
    return rows


def save_golden_set(
    records: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    source_label: str = "dataset/twcs.csv",
) -> Path:
    """Write the golden set to ``golden_set.json`` and ``golden_set.csv``.

    JSON is the canonical, fully-structured form; CSV is a flattened copy
    for convenient manual review in a spreadsheet.
    """
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "golden_set.json"
    payload = {
        "schema_version": "1.0.0",
        "generated_by": "src/golden_set.py (human-label infrastructure)",
        "source": source_label,
        "count": len(records),
        "label_fields": LABEL_FIELDS,
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / "golden_set.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_flatten_to_csv_rows(records))

    return out_dir


# ---------------------------------------------------------------------------
# Labeling guide
# ---------------------------------------------------------------------------

def build_labeling_guide_text(taxonomy: List[Dict[str, str]]) -> str:
    """Generate the human labeling guide markdown from the intent taxonomy."""
    lines = [
        "# MicrosoftHelps Golden Evaluation Set - Labeling Guide",
        "",
        "This guide tells you how to fill in the empty label fields of each",
        "record in `golden_set.json` (or `golden_set.csv`). Labels are",
        "ground truth: they must be assigned by a human, never by the AI.",
        "",
        "---",
        "",
        "## Fields you label",
        "",
        "For every record, fill in these three fields:",
        "",
        "1. `intent_label`: exactly ONE intent from the 10-intent taxonomy.",
        "2. `escalation_label`: exactly ONE of `AUTO_HANDLE` or `ESCALATE_TO_HUMAN`.",
        "3. `notes` (optional): one line explaining any judgement call.",
        "",
        "---",
        "",
        "## Intent taxonomy (choose exactly one)",
        "",
        "Pick the intent that best matches the customer's *current message*,",
        "using the conversation context and Microsoft's responses to disambiguate.",
        "If two intents both fit, pick the closest and note the ambiguity in `notes`.",
        "",
    ]

    for i, entry in enumerate(taxonomy, 1):
        definition = entry.get("definition") or "No definition provided."
        lines.append(f"{i}. **{entry['name']}** - {definition}")

    lines += [
        "",
        "---",
        "",
        "## Escalation label",
        "",
        "- `AUTO_HANDLE`: the conversation shows a clear, relevant Microsoft",
        "  resolution. A reply can be safely grounded in the historical",
        "  evidence without guessing.",
        "- `ESCALATE_TO_HUMAN`: use this when the evidence is missing, weak, or",
        "  unrelated, the Microsoft reply would require unsupported facts, or",
        "  the issue involves a sensitive/high-risk intent such as:",
        "  - Complaint / Feedback",
        "  - Cancellation / Subscription",
        "  - Billing & Payments",
        "  - Account & Login (security-sensitive cases)",
        "",
        "Prefer `ESCALATE_TO_HUMAN` whenever you would not confidently send the",
        "reply to a real customer yourself.",
        "",
        "---",
        "",
        "_The 10-intent taxonomy is defined in `config/intents.yaml` and is",
        "shared with the runtime agent. Do not invent new intents._",
    ]
    return "\n".join(lines)


def save_labeling_guide(
    text: str,
    output_dir: Optional[str] = None,
) -> Path:
    """Write the labeling guide to ``labeling_guide.md``."""
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    guide_path = out_dir / "labeling_guide.md"
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_dir