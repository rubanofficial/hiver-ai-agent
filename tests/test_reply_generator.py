"""
Focused tests for the Gemini-backed Reply Generator module.

All Gemini interactions are fully mocked — no real API calls are made.
"""

import json

import pytest

from src.agent.reply_generator import (
    ReplyGenerator,
    DraftReply,
    ReplyGeneratorError,
    ConfigurationError,
    InvalidReplyResponseError,
)
from src.agent.retriever import Evidence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_gemini(monkeypatch):
    """Provide a fake API key and stub Gemini responses for every test."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key")
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")


@pytest.fixture
def generator():
    """Return a ReplyGenerator wired to the fake API key."""
    return ReplyGenerator()


@pytest.fixture
def sample_evidence():
    """Return a small list of Evidence objects for testing."""
    return [
        Evidence(
            id="conv_001",
            text="Customer had a flickering screen issue resolved by updating display drivers.",
            score=0.92,
        ),
        Evidence(
            id="conv_002",
            text="Surface laptop screen flicker fixed by disabling adaptive brightness.",
            score=0.87,
        ),
    ]


def _fake_gemini_reply(
    reply_text="Thank you for reaching out. Based on similar cases, please try updating your display drivers.",
    confidence=0.9,
    grounded=True,
):
    """Helper to create a JSON string mimicking a valid Gemini response."""
    return json.dumps({
        "reply_text": reply_text,
        "confidence": confidence,
        "grounded": grounded,
    })


# ---------------------------------------------------------------------------
# Successful grounded reply generation
# ---------------------------------------------------------------------------

class TestSuccessfulGeneration:
    """Verify that a valid Gemini response produces a correct DraftReply."""

    def test_returns_draft_reply(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, msg, intent, ev: _fake_gemini_reply(),
        )
        result = generator.generate(
            customer_message="My Surface screen is flickering",
            intent="Technical Troubleshooting",
            evidence=sample_evidence,
        )
        assert isinstance(result, DraftReply)

    def test_reply_text_is_populated(self, generator, monkeypatch, sample_evidence):
        expected = "Please update your display drivers to fix the flickering."
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, msg, intent, ev: _fake_gemini_reply(reply_text=expected),
        )
        result = generator.generate(
            customer_message="Screen flickers constantly",
            intent="Technical Troubleshooting",
            evidence=sample_evidence,
        )
        assert result.reply_text == expected

    def test_confidence_from_gemini(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, msg, intent, ev: _fake_gemini_reply(confidence=0.75),
        )
        result = generator.generate(
            customer_message="Screen flickers",
            intent="Technical Troubleshooting",
            evidence=sample_evidence,
        )
        assert result.confidence == 0.75

    def test_grounded_flag_from_gemini(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, msg, intent, ev: _fake_gemini_reply(grounded=True),
        )
        result = generator.generate(
            customer_message="Screen flickers",
            intent="Technical Troubleshooting",
            evidence=sample_evidence,
        )
        assert result.grounded is True

    def test_citations_match_evidence_ids(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, msg, intent, ev: _fake_gemini_reply(),
        )
        result = generator.generate(
            customer_message="Screen flickers",
            intent="Technical Troubleshooting",
            evidence=sample_evidence,
        )
        assert result.citations == ["conv_001", "conv_002"]

    def test_metadata_contains_model_and_source(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, msg, intent, ev: _fake_gemini_reply(),
        )
        result = generator.generate(
            customer_message="Screen flickers",
            intent="Technical Troubleshooting",
            evidence=sample_evidence,
        )
        assert result.metadata["source"] == "gemini"
        assert result.metadata["model"] == "fake-model"
        assert result.metadata["evidence_count"] == 2


# ---------------------------------------------------------------------------
# Evidence inclusion in the Gemini prompt
# ---------------------------------------------------------------------------

class TestEvidenceInPrompt:
    """Verify that evidence content appears in the prompt sent to Gemini."""

    def test_evidence_text_appears_in_prompt(self, generator, monkeypatch, sample_evidence):
        captured_prompts = []

        def _capture_generate(self, customer_message, intent, evidence):
            prompt = generator._build_prompt(customer_message, intent, evidence)
            captured_prompts.append(prompt)
            return _fake_gemini_reply()

        monkeypatch.setattr(ReplyGenerator, "_generate", _capture_generate)

        generator.generate(
            customer_message="My Surface screen is flickering",
            intent="Technical Troubleshooting",
            evidence=sample_evidence,
        )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # Evidence text should be embedded in the prompt
        assert "flickering screen issue" in prompt
        assert "adaptive brightness" in prompt
        # Internal evidence IDs must NOT appear in the evidence block
        # (they are stripped to prevent leaking into customer replies)
        assert "conv_001" not in prompt.split("RETRIEVED HISTORICAL EVIDENCE")[-1]
        assert "conv_002" not in prompt.split("RETRIEVED HISTORICAL EVIDENCE")[-1]

    def test_customer_message_in_prompt(self, generator, monkeypatch, sample_evidence):
        captured_prompts = []

        def _capture_generate(self, customer_message, intent, evidence):
            prompt = generator._build_prompt(customer_message, intent, evidence)
            captured_prompts.append(prompt)
            return _fake_gemini_reply()

        monkeypatch.setattr(ReplyGenerator, "_generate", _capture_generate)

        generator.generate(
            customer_message="My Surface screen is flickering",
            intent="Technical Troubleshooting",
            evidence=sample_evidence,
        )

        assert "My Surface screen is flickering" in captured_prompts[0]

    def test_intent_in_prompt(self, generator, monkeypatch, sample_evidence):
        captured_prompts = []

        def _capture_generate(self, customer_message, intent, evidence):
            prompt = generator._build_prompt(customer_message, intent, evidence)
            captured_prompts.append(prompt)
            return _fake_gemini_reply()

        monkeypatch.setattr(ReplyGenerator, "_generate", _capture_generate)

        generator.generate(
            customer_message="Screen flickers",
            intent="Technical Troubleshooting",
            evidence=sample_evidence,
        )

        assert "Technical Troubleshooting" in captured_prompts[0]

    def test_grounding_instructions_in_prompt(self, generator, monkeypatch, sample_evidence):
        captured_prompts = []

        def _capture_generate(self, customer_message, intent, evidence):
            prompt = generator._build_prompt(customer_message, intent, evidence)
            captured_prompts.append(prompt)
            return _fake_gemini_reply()

        monkeypatch.setattr(ReplyGenerator, "_generate", _capture_generate)

        generator.generate(
            customer_message="Screen flickers",
            intent="Technical Troubleshooting",
            evidence=sample_evidence,
        )

        prompt = captured_prompts[0]
        assert "PRIMARY source" in prompt
        assert "Do NOT invent" in prompt
        assert "historical evidence" in prompt.lower()
        # New tiered prompt structure
        assert "UNIVERSAL RULES" in prompt
        assert "EVIDENCE QUALITY" in prompt


# ---------------------------------------------------------------------------
# Empty / no evidence handling
# ---------------------------------------------------------------------------

class TestEmptyEvidence:
    """Verify behaviour when no evidence is retrieved."""

    def test_empty_evidence_produces_valid_reply(self, generator, monkeypatch):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, msg, intent, ev: _fake_gemini_reply(
                reply_text="I don't have enough information to resolve this. Please contact Microsoft support.",
                confidence=0.3,
                grounded=False,
            ),
        )
        result = generator.generate(
            customer_message="Help me with my issue",
            intent="Technical Troubleshooting",
            evidence=[],
        )
        assert isinstance(result, DraftReply)
        assert result.citations == []
        assert result.metadata["evidence_count"] == 0

    def test_no_evidence_mention_in_prompt(self, generator, monkeypatch):
        captured_prompts = []

        def _capture_generate(self, customer_message, intent, evidence):
            prompt = generator._build_prompt(customer_message, intent, evidence)
            captured_prompts.append(prompt)
            return _fake_gemini_reply()

        monkeypatch.setattr(ReplyGenerator, "_generate", _capture_generate)

        generator.generate(
            customer_message="Help me",
            intent="Technical Troubleshooting",
            evidence=[],
        )

        assert "No historical evidence" in captured_prompts[0]

    def test_single_evidence_in_prompt(self, generator, monkeypatch):
        single_ev = [Evidence(id="conv_100", text="Reset your password via account settings.", score=0.80)]
        captured_prompts = []

        def _capture_generate(self, customer_message, intent, evidence):
            prompt = generator._build_prompt(customer_message, intent, evidence)
            captured_prompts.append(prompt)
            return _fake_gemini_reply()

        monkeypatch.setattr(ReplyGenerator, "_generate", _capture_generate)

        result = generator.generate(
            customer_message="I forgot my password",
            intent="Account & Login",
            evidence=single_ev,
        )

        # Evidence text must appear in the prompt
        assert "Reset your password" in captured_prompts[0]
        # Internal ID must NOT appear in the evidence block (stripped intentionally)
        evidence_section = captured_prompts[0].split("RETRIEVED HISTORICAL EVIDENCE")[-1]
        assert "conv_100" not in evidence_section
        # But the ID must still flow through to citations in the final reply
        assert "conv_100" in result.citations


# ---------------------------------------------------------------------------
# Invalid Gemini response handling
# ---------------------------------------------------------------------------

class TestInvalidGeminiResponse:
    """Verify that malformed or missing data from Gemini raises errors."""

    def test_empty_response_raises(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(ReplyGenerator, "_generate", lambda self, m, i, ev: "")
        with pytest.raises(InvalidReplyResponseError, match="empty response"):
            generator.generate("Hello", "Technical Troubleshooting", sample_evidence)

    def test_whitespace_only_response_raises(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(ReplyGenerator, "_generate", lambda self, m, i, ev: "   ")
        with pytest.raises(InvalidReplyResponseError, match="empty response"):
            generator.generate("Hello", "Technical Troubleshooting", sample_evidence)

    def test_malformed_json_raises(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(ReplyGenerator, "_generate", lambda self, m, i, ev: "not json {")
        with pytest.raises(InvalidReplyResponseError, match="malformed JSON"):
            generator.generate("Hello", "Technical Troubleshooting", sample_evidence)

    def test_array_response_raises(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(ReplyGenerator, "_generate", lambda self, m, i, ev: "[1, 2, 3]")
        with pytest.raises(InvalidReplyResponseError, match="single JSON object"):
            generator.generate("Hello", "Technical Troubleshooting", sample_evidence)

    def test_missing_reply_text_raises(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, m, i, ev: json.dumps({"confidence": 0.9, "grounded": True}),
        )
        with pytest.raises(InvalidReplyResponseError, match="reply_text"):
            generator.generate("Hello", "Technical Troubleshooting", sample_evidence)

    def test_empty_reply_text_raises(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, m, i, ev: json.dumps({"reply_text": "", "confidence": 0.9}),
        )
        with pytest.raises(InvalidReplyResponseError, match="reply_text"):
            generator.generate("Hello", "Technical Troubleshooting", sample_evidence)

    def test_non_numeric_confidence_defaults_to_zero(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, m, i, ev: json.dumps({
                "reply_text": "Here is your answer.",
                "confidence": "high",
                "grounded": True,
            }),
        )
        result = generator.generate("Hello", "Technical Troubleshooting", sample_evidence)
        assert result.confidence == 0.0

    def test_confidence_clamped_above_one(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, m, i, ev: json.dumps({
                "reply_text": "Here is your answer.",
                "confidence": 1.5,
                "grounded": True,
            }),
        )
        result = generator.generate("Hello", "Technical Troubleshooting", sample_evidence)
        assert result.confidence == 1.0

    def test_confidence_clamped_below_zero(self, generator, monkeypatch, sample_evidence):
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, m, i, ev: json.dumps({
                "reply_text": "Here is your answer.",
                "confidence": -0.5,
                "grounded": True,
            }),
        )
        result = generator.generate("Hello", "Technical Troubleshooting", sample_evidence)
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------

class TestMissingApiKey:
    """Verify that a missing API key raises ConfigurationError."""

    def test_missing_api_key_raises(self, monkeypatch, sample_evidence):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        gen = ReplyGenerator(api_key=None)
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
            gen.generate("Hello", "Technical Troubleshooting", sample_evidence)

    def test_empty_string_api_key_raises(self, sample_evidence):
        gen = ReplyGenerator(api_key="")
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
            gen.generate("Hello", "Technical Troubleshooting", sample_evidence)


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------

class TestConstructorDefaults:
    """Verify the generator initialises with sensible defaults."""

    def test_default_model_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "my-custom-model")
        gen = ReplyGenerator()
        assert gen.model_name == "my-custom-model"

    def test_explicit_model_overrides_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "env-model")
        gen = ReplyGenerator(model_name="explicit-model")
        assert gen.model_name == "explicit-model"

    def test_fallback_to_default_model(self, monkeypatch):
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        gen = ReplyGenerator()
        assert gen.model_name == "gemini-1.5-flash"

    def test_default_model_name_constant(self):
        assert ReplyGenerator.DEFAULT_MODEL == "gemini-1.5-flash"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Verify that invalid inputs are rejected before calling Gemini."""

    def test_empty_customer_message_raises(self, generator, monkeypatch):
        monkeypatch.setattr(ReplyGenerator, "_generate", lambda self, m, i, ev: "")
        with pytest.raises(ValueError, match="non-empty string"):
            generator.generate("", "Technical Troubleshooting", [])

    def test_whitespace_only_message_raises(self, generator, monkeypatch):
        monkeypatch.setattr(ReplyGenerator, "_generate", lambda self, m, i, ev: "")
        with pytest.raises(ValueError, match="non-empty string"):
            generator.generate("   ", "Technical Troubleshooting", [])


# ---------------------------------------------------------------------------
# Evidence quality classification
# ---------------------------------------------------------------------------

class TestEvidenceQuality:
    """Verify the _evidence_quality helper classifies evidence strength correctly."""

    def test_no_evidence_is_weak(self, generator):
        assert generator._evidence_quality([]) == "WEAK"

    def test_score_below_050_is_weak(self, generator):
        ev = [Evidence(id="e1", text="loosely related", score=0.49)]
        assert generator._evidence_quality(ev) == "WEAK"

    def test_score_at_050_is_relevant(self, generator):
        ev = [Evidence(id="e1", text="partially related", score=0.50)]
        assert generator._evidence_quality(ev) == "RELEVANT"

    def test_score_between_050_and_070_is_relevant(self, generator):
        ev = [Evidence(id="e1", text="partially related", score=0.62)]
        assert generator._evidence_quality(ev) == "RELEVANT"

    def test_score_at_070_is_strong(self, generator):
        ev = [Evidence(id="e1", text="closely related", score=0.70)]
        assert generator._evidence_quality(ev) == "STRONG"

    def test_score_above_070_is_strong(self, generator):
        ev = [Evidence(id="e1", text="closely related", score=0.91)]
        assert generator._evidence_quality(ev) == "STRONG"

    def test_uses_max_score_across_items(self, generator):
        """When multiple evidence items are present, the highest score decides."""
        ev = [
            Evidence(id="e1", text="loosely related", score=0.30),
            Evidence(id="e2", text="closely related", score=0.80),
        ]
        assert generator._evidence_quality(ev) == "STRONG"


# ---------------------------------------------------------------------------
# Prompt tiering — STRONG evidence
# ---------------------------------------------------------------------------

class TestStrongEvidencePrompt:
    """
    When evidence is STRONG (score >= 0.70), the prompt must instruct the
    model to summarise the concrete historical resolution and frame it as
    a similar-case suggestion.
    """

    @pytest.fixture
    def strong_evidence(self):
        return [
            Evidence(
                id="conv_win_01",
                text=(
                    "Customer reported Windows update stuck at 99%. "
                    "Agent suggested a forced reboot. "
                    "After the reboot the update completed successfully."
                ),
                score=0.778,
            ),
        ]

    def test_strong_quality_label_in_prompt(self, generator, strong_evidence):
        prompt = generator._build_prompt(
            customer_message="My Windows update is stuck at 99%.",
            intent="Technical Troubleshooting",
            evidence=strong_evidence,
        )
        assert "EVIDENCE QUALITY: STRONG" in prompt

    def test_strong_prompt_instructs_resolution_summary(self, generator, strong_evidence):
        prompt = generator._build_prompt(
            customer_message="My Windows update is stuck at 99%.",
            intent="Technical Troubleshooting",
            evidence=strong_evidence,
        )
        # Should tell the model to summarise the resolution from the evidence
        assert "summarise" in prompt or "summarize" in prompt.lower()
        assert "similar historical case" in prompt

    def test_strong_prompt_instructs_no_guarantee(self, generator, strong_evidence):
        prompt = generator._build_prompt(
            customer_message="My Windows update is stuck at 99%.",
            intent="Technical Troubleshooting",
            evidence=strong_evidence,
        )
        assert "guarantee" in prompt.lower()

    def test_strong_evidence_text_included_in_prompt(self, generator, strong_evidence):
        prompt = generator._build_prompt(
            customer_message="My Windows update is stuck at 99%.",
            intent="Technical Troubleshooting",
            evidence=strong_evidence,
        )
        assert "forced reboot" in prompt
        assert "completed successfully" in prompt

    def test_strong_evidence_id_not_in_evidence_block(self, generator, strong_evidence):
        """Internal conv IDs must be stripped from the evidence block."""
        prompt = generator._build_prompt(
            customer_message="My Windows update is stuck at 99%.",
            intent="Technical Troubleshooting",
            evidence=strong_evidence,
        )
        evidence_section = prompt.split("RETRIEVED HISTORICAL EVIDENCE")[-1]
        assert "conv_win_01" not in evidence_section

    def test_strong_evidence_produces_grounded_reply(self, generator, monkeypatch, strong_evidence):
        """A reply generated from STRONG evidence must be marked grounded=True."""
        expected_reply = (
            "Based on a similar case, a forced reboot resolved a Windows update "
            "stuck at 99%. Please try restarting your device — this is not "
            "guaranteed but has worked in similar situations."
        )
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, msg, intent, ev: json.dumps({
                "reply_text": expected_reply,
                "confidence": 0.82,
                "grounded": True,
            }),
        )
        result = generator.generate(
            customer_message="My Windows update is stuck at 99%.",
            intent="Technical Troubleshooting",
            evidence=strong_evidence,
        )
        assert result.grounded is True
        assert result.confidence >= 0.75
        assert result.reply_text == expected_reply


# ---------------------------------------------------------------------------
# Prompt tiering — RELEVANT evidence (no concrete resolution)
# ---------------------------------------------------------------------------

class TestRelevantEvidencePrompt:
    """
    When evidence is RELEVANT (0.50 <= score < 0.70) but does not contain a
    concrete resolution, the prompt must guide the model to explain WHY the
    case needs investigation rather than giving a generic redirect.
    """

    @pytest.fixture
    def relevant_evidence(self):
        return [
            Evidence(
                id="conv_store_01",
                text=(
                    "Customer purchased a game from the Microsoft Store and "
                    "reported missing content after download. "
                    "The issue was escalated to the licensing team for order review."
                ),
                score=0.627,
            ),
        ]

    def test_relevant_quality_label_in_prompt(self, generator, relevant_evidence):
        prompt = generator._build_prompt(
            customer_message="I bought the deluxe edition but got the standard version.",
            intent="Microsoft Store",
            evidence=relevant_evidence,
        )
        assert "EVIDENCE QUALITY: RELEVANT" in prompt

    def test_relevant_prompt_instructs_specific_reason(self, generator, relevant_evidence):
        """Model must be told to give a specific reason when recommending support."""
        prompt = generator._build_prompt(
            customer_message="I bought the deluxe edition but got the standard version.",
            intent="Microsoft Store",
            evidence=relevant_evidence,
        )
        assert "account entitlement" in prompt or "order lookup" in prompt or "investigation" in prompt

    def test_relevant_prompt_asks_for_missing_information(self, generator, relevant_evidence):
        prompt = generator._build_prompt(
            customer_message="I bought the deluxe edition but got the standard version.",
            intent="Microsoft Store",
            evidence=relevant_evidence,
        )
        assert "missing information" in prompt or "Ask the customer" in prompt

    def test_relevant_evidence_produces_useful_reply(self, generator, monkeypatch, relevant_evidence):
        """Reply for RELEVANT evidence must be grounded and carry useful context."""
        expected_reply = (
            "Thank you for reaching out. Based on a similar case, missing game "
            "content after a Microsoft Store purchase typically requires an "
            "order entitlement review by the Microsoft Store team. "
            "Could you share your order number so we can investigate further? "
            "You can also contact Microsoft Support directly for account and "
            "order lookups."
        )
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, msg, intent, ev: json.dumps({
                "reply_text": expected_reply,
                "confidence": 0.60,
                "grounded": True,
            }),
        )
        result = generator.generate(
            customer_message="I bought the deluxe edition but got the standard version.",
            intent="Microsoft Store",
            evidence=relevant_evidence,
        )
        assert result.grounded is True
        assert 0.40 <= result.confidence <= 0.75


# ---------------------------------------------------------------------------
# Prompt tiering — WEAK evidence
# ---------------------------------------------------------------------------

class TestWeakEvidencePrompt:
    """
    When evidence is WEAK (score < 0.50) the prompt must instruct the model
    to stay cautious and NOT attempt to answer from the evidence.
    """

    @pytest.fixture
    def weak_evidence(self):
        return [
            Evidence(
                id="conv_unrelated_01",
                text="Customer asked about a printer driver on an older system.",
                score=0.42,
            ),
        ]

    def test_weak_quality_label_in_prompt(self, generator, weak_evidence):
        prompt = generator._build_prompt(
            customer_message="My account is completely locked out.",
            intent="Account & Login",
            evidence=weak_evidence,
        )
        assert "EVIDENCE QUALITY: WEAK" in prompt

    def test_weak_prompt_instructs_no_answer_from_evidence(self, generator, weak_evidence):
        prompt = generator._build_prompt(
            customer_message="My account is completely locked out.",
            intent="Account & Login",
            evidence=weak_evidence,
        )
        assert "Do NOT attempt to answer" in prompt

    def test_no_evidence_also_produces_weak_quality(self, generator):
        prompt = generator._build_prompt(
            customer_message="Help me with my issue.",
            intent="Technical Troubleshooting",
            evidence=[],
        )
        assert "EVIDENCE QUALITY: WEAK" in prompt

    def test_weak_evidence_produces_cautious_ungrounded_reply(self, generator, monkeypatch, weak_evidence):
        """A weak-evidence reply must set grounded=False and low confidence."""
        cautious_reply = (
            "I don't have enough information to resolve this issue autonomously. "
            "Please contact Microsoft Support directly for assistance."
        )
        monkeypatch.setattr(
            ReplyGenerator,
            "_generate",
            lambda self, msg, intent, ev: json.dumps({
                "reply_text": cautious_reply,
                "confidence": 0.30,
                "grounded": False,
            }),
        )
        result = generator.generate(
            customer_message="My account is completely locked out.",
            intent="Account & Login",
            evidence=weak_evidence,
        )
        assert result.grounded is False
        assert result.confidence <= 0.45


# ---------------------------------------------------------------------------
# Prevention of unsupported claims
# ---------------------------------------------------------------------------

class TestUnsupportedClaimPrevention:
    """
    The prompt must always include explicit prohibitions against inventing
    Microsoft policies, URLs, internal IDs, similarity scores, and
    unsupported account/billing details.
    """

    def _prompt_for(self, generator, score):
        ev = [Evidence(id="ev1", text="A related Microsoft support case.", score=score)]
        return generator._build_prompt(
            customer_message="What is my account balance?",
            intent="Billing & Payments",
            evidence=ev,
        )

    def test_prohibits_invented_policies_in_strong_prompt(self, generator):
        prompt = self._prompt_for(generator, score=0.85)
        assert "Do NOT invent" in prompt

    def test_prohibits_invented_policies_in_relevant_prompt(self, generator):
        prompt = self._prompt_for(generator, score=0.55)
        assert "Do NOT invent" in prompt

    def test_prohibits_invented_policies_in_weak_prompt(self, generator):
        prompt = self._prompt_for(generator, score=0.30)
        assert "Do NOT invent" in prompt

    def test_prohibits_internal_ids_in_reply(self, generator):
        """All tiers must prohibit exposing internal IDs."""
        for score in (0.85, 0.55, 0.30):
            prompt = self._prompt_for(generator, score=score)
            assert "internal IDs" in prompt or "conversation IDs" in prompt

    def test_prohibits_similarity_scores_in_reply(self, generator):
        """All tiers must prohibit mentioning similarity scores."""
        for score in (0.85, 0.55, 0.30):
            prompt = self._prompt_for(generator, score=score)
            assert "similarity scores" in prompt

    def test_prohibits_unsupported_account_details(self, generator):
        """All tiers must prohibit fabricating account/billing/order details."""
        for score in (0.85, 0.55, 0.30):
            prompt = self._prompt_for(generator, score=score)
            assert "account details" in prompt or "billing" in prompt.lower()

    def test_internal_id_not_in_evidence_block_for_any_tier(self, generator):
        """IDs must be stripped from the evidence block in all tiers."""
        for score in (0.85, 0.55, 0.30):
            ev = [Evidence(id="secret_conv_id", text="A support case.", score=score)]
            prompt = generator._build_prompt(
                customer_message="Help me.",
                intent="Technical Troubleshooting",
                evidence=ev,
            )
            evidence_section = prompt.split("RETRIEVED HISTORICAL EVIDENCE")[-1]
            assert "secret_conv_id" not in evidence_section
