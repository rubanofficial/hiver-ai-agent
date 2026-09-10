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
        # Evidence IDs should appear
        assert "conv_001" in prompt
        assert "conv_002" in prompt

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

        generator.generate(
            customer_message="I forgot my password",
            intent="Account & Login",
            evidence=single_ev,
        )

        assert "Reset your password" in captured_prompts[0]
        assert "conv_100" in captured_prompts[0]


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
