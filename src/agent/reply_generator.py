"""
Reply Generator Module for MicrosoftHelps AI Support Agent.

Generates brand-aligned, grounded replies based on classified intent
and retrieved historical evidence using the Google Gemini API.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from .retriever import Evidence


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ReplyGeneratorError(Exception):
    """Base class for all reply-generator errors."""


class ConfigurationError(ReplyGeneratorError):
    """Raised when required configuration (e.g. GEMINI_API_KEY) is missing."""


class InvalidReplyResponseError(ReplyGeneratorError):
    """Raised when Gemini returns a response that cannot be parsed."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DraftReply:
    """Represents a generated draft reply along with grounding metadata."""
    reply_text: str
    grounded: bool = True
    confidence: float = 1.0
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ReplyGenerator:
    """
    Synthesizes customer support replies conditioned on retrieved evidence.

    Uses a Gemini model to produce a concise, customer-facing reply that
    is grounded in the provided historical MicrosoftHelps evidence.
    """

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the reply generator.

        Args:
            model_name: Gemini model identifier. Falls back to GEMINI_MODEL env var.
            api_key: Google AI Studio API key. Falls back to GEMINI_API_KEY env var.
        """
        self.model_name = model_name or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")

    # -- prompt construction ------------------------------------------------

    def _format_evidence_block(self, evidence: List[Evidence]) -> str:
        """Format the evidence list into a numbered text block for the prompt."""
        if not evidence:
            return (
                "No historical evidence was retrieved for this customer message."
            )
        lines: List[str] = []
        for idx, ev in enumerate(evidence, start=1):
            lines.append(
                f"[{idx}] (ID: {ev.id}, score: {ev.score:.2f}) {ev.text}"
            )
        return "\n".join(lines)

    def _build_prompt(
        self,
        customer_message: str,
        intent: str,
        evidence: List[Evidence],
    ) -> str:
        """Build the prompt passed to Gemini for reply generation."""
        evidence_block = self._format_evidence_block(evidence)

        return (
            "You are a customer-support agent for MicrosoftHelps, an AI support "
            "service for Microsoft products and services.\n\n"
            "Your task is to write a concise, helpful reply to the customer "
            "message below. You MUST ground your reply in the RETRIEVED "
            "HISTORICAL EVIDENCE provided.\n\n"
            "RULES:\n"
            "- Use the historical evidence as your PRIMARY source of information.\n"
            "- Do NOT invent Microsoft policies, URLs, troubleshooting steps, "
            "prices, guarantees, or any other unsupported facts.\n"
            "- If the evidence is insufficient to safely answer, acknowledge "
            "that you cannot fully resolve the issue and avoid guessing. "
            "Suggest the customer contact Microsoft support directly.\n"
            "- Keep the reply professional, empathetic, and concise.\n"
            "- Return ONLY a single valid JSON object matching this schema:\n"
            '  {"reply_text": "<your customer-facing reply>", '
            '"confidence": <float between 0 and 1>, '
            '"grounded": <true or false>}\n\n'
            f"CUSTOMER MESSAGE:\n{customer_message.strip()}\n\n"
            f"CLASSIFIED INTENT:\n{intent}\n\n"
            f"RETRIEVED HISTORICAL EVIDENCE:\n{evidence_block}"
        )

    # -- Gemini call --------------------------------------------------------

    def _generate(self, customer_message: str, intent: str, evidence: List[Evidence]) -> str:
        """Call the Gemini model and return the raw response text."""
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            self._build_prompt(customer_message, intent, evidence),
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )
        return response.text

    # -- parsing & validation -----------------------------------------------

    def _parse_response(self, raw_text: str, evidence: List[Evidence]) -> DraftReply:
        """Parse and validate Gemini's structured JSON response into a DraftReply."""
        if not raw_text or not raw_text.strip():
            raise InvalidReplyResponseError("Gemini returned an empty response.")

        try:
            payload = json.loads(raw_text.strip())
        except json.JSONDecodeError as exc:
            raise InvalidReplyResponseError(
                f"Gemini returned malformed JSON: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise InvalidReplyResponseError(
                "Gemini response must be a single JSON object."
            )

        reply_text = payload.get("reply_text", "")
        if not isinstance(reply_text, str) or not reply_text.strip():
            raise InvalidReplyResponseError(
                "Gemini response missing or empty 'reply_text' field."
            )

        raw_confidence = payload.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        grounded = payload.get("grounded", bool(evidence))
        if not isinstance(grounded, bool):
            grounded = bool(evidence)

        citations = [ev.id for ev in evidence]

        return DraftReply(
            reply_text=reply_text.strip(),
            grounded=grounded,
            confidence=confidence,
            citations=citations,
            metadata={
                "model": self.model_name,
                "source": "gemini",
                "evidence_count": len(evidence),
            },
        )

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        customer_message: str,
        intent: str,
        evidence: List[Evidence],
    ) -> DraftReply:
        """
        Generate a candidate response grounded in evidence.

        Args:
            customer_message: Original customer message.
            intent: Identified intent category.
            evidence: Retrieved historical context or documentation.

        Returns:
            DraftReply containing response text and grounding metadata.

        Raises:
            ConfigurationError: If GEMINI_API_KEY is not configured.
            InvalidReplyResponseError: If Gemini's response is malformed.
        """
        if not self.api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Set the GEMINI_API_KEY environment "
                "variable (or pass api_key=...) before generating replies."
            )

        if not isinstance(customer_message, str) or not customer_message.strip():
            raise ValueError("customer_message must be a non-empty string.")

        raw_text = self._generate(customer_message, intent, evidence)
        return self._parse_response(raw_text, evidence)
