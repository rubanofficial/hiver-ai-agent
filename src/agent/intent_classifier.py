"""
Gemini-based Intent Classifier Module for MicrosoftHelps AI Support Agent.

Classifies an incoming customer message into exactly one of the 10
MicrosoftHelps support intents defined in config/intents.yaml, using the
official Google Gemini Python SDK. The intent taxonomy is loaded
dynamically from YAML so that no intent names are hard-coded here.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml
import google.generativeai as genai


class IntentClassifierError(Exception):
    """Base class for all intent-classifier errors."""


class ConfigurationError(IntentClassifierError):
    """Raised when required configuration (e.g. GEMINI_API_KEY) is missing."""


class InvalidIntentResponseError(IntentClassifierError):
    """Raised when Gemini returns a response that cannot be trusted."""


@dataclass
class IntentClassificationResult:
    """Represents the outcome of intent classification."""
    intent: str
    confidence: float = 1.0
    reason: str = ""
    probabilities: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentClassifier:
    """
    Classifies inbound customer support messages into domain-specific intents.

    Uses a Gemini model to pick exactly one intent from the configured
    taxonomy. The inferred intent is validated against the taxonomy before
    being returned.
    """

    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "intents.yaml"
    )
    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the classifier with intent taxonomy configurations.

        Args:
            config_path: Optional path to intents.yaml configuration.
            model_name: Gemini model identifier. Falls back to GEMINI_MODEL.
            api_key: Google AI Studio API key. Falls back to GEMINI_API_KEY.
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.model_name = model_name or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self.intents: List[str]
        self.intent_definitions: Dict[str, str]
        self.intents, self.intent_definitions = self._load_intents()

    def _load_intents(self) -> Tuple[List[str], Dict[str, str]]:
        """Load intent names and definitions from config/intents.yaml."""
        if not os.path.exists(self.config_path):
            raise ConfigurationError(
                f"Intent taxonomy config file not found: {self.config_path}"
            )
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            raise ConfigurationError(
                f"Failed to read intent taxonomy config {self.config_path}: {exc}"
            ) from exc

        entries = cfg.get("intents") or []
        if not entries:
            raise ConfigurationError(
                f"No intents defined in config {self.config_path}"
            )

        names: List[str] = []
        definitions: Dict[str, str] = {}
        for item in entries:
            if not isinstance(item, dict) or not item.get("name"):
                raise ConfigurationError(
                    f"Malformed intent entry in config {self.config_path}: {item!r}"
                )
            name = item["name"]
            names.append(name)
            definitions[name] = item.get("definition", "")
        return names, definitions

    def classify(self, customer_message: str) -> IntentClassificationResult:
        """
        Predict the customer support intent for the given message.

        Args:
            customer_message: Raw text from customer.

        Returns:
            IntentClassificationResult containing the classified intent,
            confidence, and a short reason.

        Raises:
            ConfigurationError: If GEMINI_API_KEY is not configured.
            InvalidIntentResponseError: If Gemini's response is malformed or
                contains an intent outside the configured taxonomy.
        """
        if not self.api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Set the GEMINI_API_KEY environment "
                "variable (or pass api_key=...) before classifying messages."
            )
        if not isinstance(customer_message, str) or not customer_message.strip():
            raise ValueError("customer_message must be a non-empty string.")

        raw_text = self._generate(customer_message)
        return self._parse_and_validate(raw_text)

    def _build_prompt(self, customer_message: str) -> str:
        """Build the few-shot taxonomy prompt passed to Gemini."""
        taxonomy_lines = "\n".join(
            f'- "{name}": {definition or "No definition provided."}'
            for name, definition in self.intent_definitions.items()
        )
        return (
            "You are the intent-classification engine of MicrosoftHelps, an AI "
            "support agent for Microsoft products and services.\n\n"
            "Classify the customer message below into EXACTLY ONE intent chosen "
            "from the allowed taxonomy.\n\n"
            "ALLOWED INTENTS (choose exactly one, using the exact name):\n"
            f"{taxonomy_lines}\n\n"
            "RULES:\n"
            "- Select exactly one intent. Never invent, modify, or rename an intent.\n"
            "- If the message is ambiguous, pick the closest allowed intent and "
            "reflect the uncertainty with a lower confidence score.\n"
            "- confidence must be a float between 0 and 1.\n"
            "- Return ONLY a single valid JSON object matching this schema:\n"
            '  {"intent": "<exact intent name>", '
            '"confidence": <float>, "reason": "<short reason>"}\n\n'
            f"CUSTOMER MESSAGE:\n{customer_message.strip()}"
        )

    def _generate(self, customer_message: str) -> str:
        """Call the Gemini model and return the raw response text."""
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            self._build_prompt(customer_message),
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
        return response.text

    def _parse_and_validate(self, raw_text: str) -> IntentClassificationResult:
        """Parse and strictly validate Gemini's structured JSON response."""
        if not raw_text or not raw_text.strip():
            raise InvalidIntentResponseError("Gemini returned an empty response.")

        try:
            payload = json.loads(raw_text.strip())
        except json.JSONDecodeError as exc:
            raise InvalidIntentResponseError(
                f"Gemini returned malformed JSON: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise InvalidIntentResponseError(
                "Gemini response must be a single JSON object."
            )

        intent = payload.get("intent")
        if not isinstance(intent, str) or intent not in self.intents:
            raise InvalidIntentResponseError(
                f"Gemini returned intent {intent!r}, which is not in the "
                "configured taxonomy."
            )

        raw_confidence = payload.get("confidence")
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError) as exc:
            raise InvalidIntentResponseError(
                "Gemini returned a non-numeric confidence score."
            ) from exc
        confidence = max(0.0, min(1.0, confidence))

        reason = payload.get("reason")
        reason_text = str(reason).strip() if reason is not None else ""

        probabilities = {name: 0.0 for name in self.intents}
        probabilities[intent] = confidence

        return IntentClassificationResult(
            intent=intent,
            confidence=confidence,
            reason=reason_text,
            probabilities=probabilities,
            metadata={"model": self.model_name, "source": "gemini"},
        )
