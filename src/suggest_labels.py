"""
AI-assisted annotation suggestion engine for the Golden Evaluation Set.

This module generates ADVISORY label suggestions for MicrosoftHelps customer
conversations using Google Gemini.

Methodology Constraints:
1. Gemini receives ONLY:
   - customer_message
   - conversation_context
   - historical MicrosoftHelps responses
   - 10 intent definitions from config/intents.yaml
   - human-labeling independent escalation guidance
2. Existing human labels are NEVER sent to Gemini.
3. Suggestions are stored separately from human ground truth (golden_set.suggestions.json).
4. Every suggestion is cached so rerunning does not make duplicate API calls.
5. AI suggestions are NEVER automatically converted to final ground truth.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
import yaml

from .golden_set import load_intent_taxonomy

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN_SET_PATH = PROJECT_ROOT / "evaluation" / "golden_set.json"
DEFAULT_SUGGESTIONS_PATH = PROJECT_ROOT / "evaluation" / "golden_set.suggestions.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "intents.yaml"
DEFAULT_MODEL = "gemini-1.5-flash"

SUGGESTIONS_SCHEMA_VERSION = "1.0.0"

ESCALATION_VALUES = ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]


class SuggestionError(Exception):
    """Base error for suggestion generation."""


class ConfigurationError(SuggestionError):
    """Raised when GEMINI_API_KEY or required config is missing."""


class InvalidSuggestionResponseError(SuggestionError):
    """Raised when Gemini returns a malformed or invalid suggestion."""


@dataclass
class LabelSuggestion:
    golden_id: str
    conversation_id: Optional[int]
    suggested_intent: str
    suggested_escalation: str
    reason: str
    confidence: float
    suggested_at: str
    model: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_suggestions_store(
    source: str = str(DEFAULT_GOLDEN_SET_PATH),
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Create a fresh, empty suggestions store."""
    return {
        "schema_version": SUGGESTIONS_SCHEMA_VERSION,
        "generated_by": "src/suggest_labels.py (AI annotation assistant)",
        "source": str(source),
        "model": model,
        "count_total": 0,
        "count_suggested": 0,
        "suggestions": {},
    }


def load_suggestions_store(path: str = str(DEFAULT_SUGGESTIONS_PATH)) -> Dict[str, Any]:
    """Load suggestions from disk; return empty store if file does not exist."""
    p = Path(path)
    if not p.exists():
        return empty_suggestions_store(source=str(DEFAULT_GOLDEN_SET_PATH))
    with open(p, "r", encoding="utf-8-sig") as f:
        store = json.load(f)
    if not isinstance(store, dict) or not isinstance(store.get("suggestions"), dict):
        raise ValueError(f"Invalid suggestions store: missing 'suggestions' map in {p}")
    store.setdefault("schema_version", SUGGESTIONS_SCHEMA_VERSION)
    store.setdefault("suggestions", {})
    store["count_suggested"] = len(store["suggestions"])
    return store


def save_suggestions_store(
    store: Dict[str, Any],
    path: str = str(DEFAULT_SUGGESTIONS_PATH),
) -> Path:
    """Atomically save the suggestions store to disk."""
    import tempfile
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    store["count_suggested"] = len(store.get("suggestions", {}))
    
    fd, tmp = tempfile.mkstemp(
        dir=str(p.parent), prefix=p.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return p


class LabelSuggester:
    """
    Generates AI-assisted label suggestions using Gemini for human annotators.
    
    Adheres strictly to the methodology:
    - Receives ONLY customer message, context, historical evidence, taxonomy & human guide.
    - NEVER receives or sees human labels.
    - Caches all suggestions to prevent redundant API calls.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.config_path = config_path or str(DEFAULT_CONFIG_PATH)
        self.model_name = model_name or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self.client = client
        self.taxonomy = load_intent_taxonomy(self.config_path)
        self.valid_intents = [e["name"] for e in self.taxonomy]

    def build_prompt(
        self,
        customer_message: str,
        conversation_context: str,
        microsoft_responses: List[str],
    ) -> str:
        """
        Build the advisory suggestion prompt.
        
        Strict constraint: receives ONLY the 5 authorized pieces of information.
        No human labels are ever included.
        """
        taxonomy_lines = "\n".join(
            f"{i}. **{e['name']}** - {e.get('definition', '')}"
            for i, e in enumerate(self.taxonomy, 1)
        )

        responses_text = (
            "\n".join(f"  - {r.strip()}" for r in microsoft_responses)
            if microsoft_responses else "  (none)"
        )
        context_text = conversation_context.strip() if conversation_context.strip() else "(none)"

        return (
            "You are an expert AI annotation assistant for Microsoft customer support evaluation.\n"
            "Your task is to provide an ADVISORY label suggestion (intent + escalation) for a human reviewer.\n"
            "A human reviewer will inspect your suggestion and decide whether to accept or modify it.\n\n"
            "--- ALLOWED 10-INTENT TAXONOMY ---\n"
            f"{taxonomy_lines}\n\n"
            "--- ESCALATION JUDGMENT GUIDELINES ---\n"
            "- AUTO_HANDLE: Would an AI support agent be able to confidently send a grounded, accurate reply to this customer?\n"
            "- ESCALATE_TO_HUMAN: Should a human agent handle this due to missing/weak context, high risk, financial/account/security issues, or complex troubleshooting?\n\n"
            "--- INPUT CONVERSATION DATA ---\n"
            f"CUSTOMER MESSAGE:\n{customer_message.strip()}\n\n"
            f"CONVERSATION CONTEXT (preceding turns):\n{context_text}\n\n"
            f"HISTORICAL MICROSOFT RESPONSES:\n{responses_text}\n\n"
            "--- OUTPUT INSTRUCTIONS ---\n"
            "Return ONLY a single valid JSON object matching this exact schema:\n"
            "{\n"
            '  "suggested_intent": "<exactly one of the 10 intent names above>",\n'
            '  "suggested_escalation": "AUTO_HANDLE or ESCALATE_TO_HUMAN",\n'
            '  "reason": "<short 1-2 sentence explanation of your suggestion>",\n'
            '  "confidence": <float between 0.0 and 1.0>\n'
            "}"
        )

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences (```json ... ```) from model output."""
        import re
        stripped = text.strip()
        # Remove opening fence: ```json or ```
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped, flags=re.IGNORECASE)
        # Remove closing fence
        stripped = re.sub(r'```\s*$', '', stripped)
        return stripped.strip()

    @staticmethod
    def _parse_retry_delay(exc: Exception, default: float = 15.0) -> float:
        """Extract suggested retry delay (seconds) from a 429 error, or return default."""
        import re
        match = re.search(r'retry[_ ]delay[^0-9]*([0-9]+(?:\.[0-9]+)?)', str(exc), re.IGNORECASE)
        if match:
            return float(match.group(1)) + 2.0  # add 2s buffer
        # Also look for 'Please retry in X.Xs'
        match2 = re.search(r'retry in ([0-9]+(?:\.[0-9]+)?)s', str(exc), re.IGNORECASE)
        if match2:
            return float(match2.group(1)) + 2.0
        return default

    def _call_gemini(self, prompt: str, max_retries: int = 5) -> str:
        """Execute the Gemini API call with 429 rate-limit retry + backoff."""
        if self.client is not None:
            # Injected client for testing
            if hasattr(self.client, "generate_content"):
                resp = self.client.generate_content(prompt)
                return resp.text if hasattr(resp, "text") else str(resp)
            if callable(self.client):
                return self.client(prompt)
            raise SuggestionError(f"Unsupported injected client type: {type(self.client)}")

        if not self.api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Set GEMINI_API_KEY environment variable "
                "or pass api_key=... to generate suggestions."
            )

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        # gemini-2.5-* are thinking models: thinking tokens count against the
        # output budget, leaving almost nothing for the actual JSON.
        # Disable thinking (thinking_budget=0) for this structured JSON task.
        def _make_config(with_mime: bool, with_thinking: bool) -> dict:
            cfg: dict = {"temperature": 0.0, "max_output_tokens": 4096}
            if with_mime:
                cfg["response_mime_type"] = "application/json"
            if with_thinking:
                try:
                    cfg["thinking_config"] = genai.types.ThinkingConfig(thinking_budget=0)
                except Exception:
                    pass  # SDK version doesn't support ThinkingConfig — ignore
            return cfg

        # Attempt order: JSON mime + no-think → plain text + no-think → plain text + think
        for with_mime, with_thinking in [(True, True), (False, True), (False, False)]:
            attempt = 0
            while attempt <= max_retries:
                try:
                    cfg = _make_config(with_mime=with_mime, with_thinking=with_thinking)
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(**cfg),
                    )
                    text = response.text
                    if text and text.strip():
                        return text
                    break  # empty response — try next config
                except Exception as exc:
                    err_str = str(exc).lower()
                    # 429 rate limit — wait and retry same config
                    if "429" in str(exc) or "quota" in err_str or "rate" in err_str:
                        wait = self._parse_retry_delay(exc)
                        print(
                            f"[suggester] Rate limit hit (429). Waiting {wait:.0f}s before retry "
                            f"(attempt {attempt + 1}/{max_retries})...",
                            flush=True,
                        )
                        time.sleep(wait)
                        attempt += 1
                        continue
                    # Unsupported param — break inner loop and try next config
                    if any(k in err_str for k in ("mime", "unsupported", "invalid", "thinking")):
                        break
                    raise
            else:
                raise SuggestionError(
                    f"Gemini 429 rate limit: max retries ({max_retries}) exhausted. "
                    "Try --delay 15 or wait before re-running (cached records are skipped)."
                )
        raise SuggestionError("All Gemini call attempts failed; check model name and API key.")

    def parse_and_validate(
        self,
        raw_text: str,
        golden_id: str,
        conversation_id: Optional[int] = None,
    ) -> LabelSuggestion:
        """Parse and strictly validate Gemini's JSON suggestion."""
        if not raw_text or not raw_text.strip():
            raise InvalidSuggestionResponseError("Gemini returned an empty response.")

        # Strip markdown fences that some models add despite response_mime_type=json
        clean_text = self._strip_markdown_fences(raw_text)

        try:
            payload = json.loads(clean_text)
        except json.JSONDecodeError as exc:
            raise InvalidSuggestionResponseError(
                f"Gemini returned malformed JSON: {exc}\nRaw response: {raw_text[:300]!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise InvalidSuggestionResponseError(
                "Gemini response must be a single JSON object."
            )

        intent = payload.get("suggested_intent")
        if not isinstance(intent, str) or intent not in self.valid_intents:
            raise InvalidSuggestionResponseError(
                f"Invalid suggested_intent {intent!r}; must be one of {self.valid_intents}"
            )

        escalation = payload.get("suggested_escalation")
        if not isinstance(escalation, str) or escalation not in ESCALATION_VALUES:
            raise InvalidSuggestionResponseError(
                f"Invalid suggested_escalation {escalation!r}; must be one of {ESCALATION_VALUES}"
            )

        raw_conf = payload.get("confidence", 1.0)
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError) as exc:
            raise InvalidSuggestionResponseError(
                f"Non-numeric confidence score: {raw_conf!r}"
            ) from exc
        confidence = max(0.0, min(1.0, confidence))

        reason = str(payload.get("reason", "") or "").strip()

        return LabelSuggestion(
            golden_id=golden_id,
            conversation_id=conversation_id,
            suggested_intent=intent,
            suggested_escalation=escalation,
            reason=reason,
            confidence=confidence,
            suggested_at=_now_iso(),
            model=self.model_name,
        )

    def suggest_for_record(
        self,
        record: Dict[str, Any],
        suggestions_store: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> LabelSuggestion:
        """
        Generate or retrieve cached suggestion for one Golden Set record.
        
        Guarantees:
        - Reuses cached suggestion if present and force=False.
        - NEVER passes human labels to Gemini.
        """
        golden_id = record.get("golden_id", "")
        conv_id = record.get("conversation_id")

        if suggestions_store and not force:
            cached = suggestions_store.get("suggestions", {}).get(golden_id)
            if cached and isinstance(cached, dict):
                return LabelSuggestion(
                    golden_id=golden_id,
                    conversation_id=conv_id,
                    suggested_intent=cached["suggested_intent"],
                    suggested_escalation=cached["suggested_escalation"],
                    reason=cached.get("reason", ""),
                    confidence=float(cached.get("confidence", 1.0)),
                    suggested_at=cached.get("suggested_at", _now_iso()),
                    model=cached.get("model", self.model_name),
                )

        customer_message = record.get("customer_message", "")
        conversation_context = record.get("conversation_context", "")
        microsoft_responses = record.get("microsoft_responses", []) or []

        prompt = self.build_prompt(
            customer_message=customer_message,
            conversation_context=conversation_context,
            microsoft_responses=microsoft_responses,
        )

        raw_resp = self._call_gemini(prompt)
        suggestion = self.parse_and_validate(raw_resp, golden_id, conv_id)

        if suggestions_store is not None:
            suggestions_store.setdefault("suggestions", {})[golden_id] = suggestion.to_dict()
            suggestions_store["count_suggested"] = len(suggestions_store["suggestions"])

        return suggestion


def generate_all_suggestions(
    golden_path: str = str(DEFAULT_GOLDEN_SET_PATH),
    suggestions_path: str = str(DEFAULT_SUGGESTIONS_PATH),
    batch_size: int = 10,
    delay: float = 0.2,
    limit: Optional[int] = None,
    record_id: Optional[str] = None,
    force: bool = False,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate and cache suggestions for all Golden Set records in batches.
    
    Saves incrementally after each batch so progress is never lost.
    """
    with open(golden_path, "r", encoding="utf-8-sig") as f:
        golden_data = json.load(f)

    records = golden_data.get("records", [])
    if record_id:
        records = [r for r in records if r.get("golden_id") == record_id]
        if not records:
            raise ValueError(f"Record {record_id} not found in {golden_path}")

    if limit:
        records = records[:limit]

    store = load_suggestions_store(suggestions_path)
    store["source"] = golden_path
    store["count_total"] = len(golden_data.get("records", []))

    suggester = LabelSuggester(
        model_name=model_name,
        api_key=api_key,
        client=client,
    )
    store["model"] = suggester.model_name

    total = len(records)
    new_count = 0
    cached_count = 0

    print(f"[suggester] Processing {total} records using {suggester.model_name}...")
    for idx, rec in enumerate(records, 1):
        gid = rec.get("golden_id")
        is_cached = (not force) and (gid in store.get("suggestions", {}))

        if is_cached:
            cached_count += 1
        else:
            sug = suggester.suggest_for_record(rec, suggestions_store=store, force=force)
            new_count += 1
            if delay > 0 and idx < total:
                time.sleep(delay)

        if idx % batch_size == 0 or idx == total:
            save_suggestions_store(store, suggestions_path)
            print(
                f"[suggester] ({idx}/{total}) Saved suggestions store "
                f"({store['count_suggested']} total, {new_count} new, {cached_count} cached)"
            )

    save_suggestions_store(store, suggestions_path)
    print(
        f"[suggester] Complete: {store['count_suggested']} suggestions saved to {suggestions_path}"
    )
    return store


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Generate AI-assisted label suggestions for the Golden Evaluation Set."
    )
    parser.add_argument(
        "--golden-path", default=str(DEFAULT_GOLDEN_SET_PATH),
        help="Path to golden_set.json (default: evaluation/golden_set.json)",
    )
    parser.add_argument(
        "--suggestions-path", default=str(DEFAULT_SUGGESTIONS_PATH),
        help="Path to save suggestions (default: evaluation/golden_set.suggestions.json)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Google AI Studio Gemini API key (defaults to GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Gemini model name (default: GEMINI_MODEL env or gemini-1.5-flash)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Number of suggestions between incremental saves (default: 10)",
    )
    parser.add_argument(
        "--delay", type=float, default=13.0,
        help="Delay in seconds between Gemini calls (default: 13.0 — respects 5 RPM free-tier quota).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit suggestion generation to first N records",
    )
    parser.add_argument(
        "--record", default=None,
        help="Generate suggestion for a single golden_id (e.g. GOLDEN-0001)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate suggestions even if cached",
    )
    args = parser.parse_args(argv)

    try:
        generate_all_suggestions(
            golden_path=args.golden_path,
            suggestions_path=args.suggestions_path,
            batch_size=args.batch_size,
            delay=args.delay,
            limit=args.limit,
            record_id=args.record,
            force=args.force,
            model_name=args.model,
            api_key=args.api_key,
        )
        return 0
    except Exception as exc:
        print(f"[suggester] Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
