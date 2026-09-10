"""
Basic tests for the MicrosoftHelps AI Support Agent pipeline.

Verifies that every module is importable and that the pipeline can be
constructed and executed with placeholder components. Gemini-backed
components are mocked so no real API calls are made during tests.
"""

import json

import pytest

from src.agent import (
    IntentClassifier,
    IntentClassificationResult,
    ConfigurationError,
    InvalidIntentResponseError,
    EvidenceRetriever,
    Evidence,
    ReplyGenerator,
    DraftReply,
    EscalationPolicy,
    EscalationDecision,
    EscalationResult,
    SupportPipeline,
    AgentResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_gemini(monkeypatch):
    """Provide a fake API key and stub Gemini responses for every test."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key")
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")

    def _fake_classify(self, customer_message):
        return json.dumps(
            {
                "intent": "Technical Troubleshooting",
                "confidence": 0.9,
                "reason": "Mocked Gemini response.",
            }
        )

    def _fake_reply(self, customer_message, intent, evidence):
        return json.dumps(
            {
                "reply_text": "Mocked reply based on retrieved evidence.",
                "confidence": 0.9,
                "grounded": bool(evidence),
            }
        )

    monkeypatch.setattr(IntentClassifier, "_generate", _fake_classify)
    monkeypatch.setattr(ReplyGenerator, "_generate", _fake_reply)


# ---------------------------------------------------------------------------
# Module-level import tests
# ---------------------------------------------------------------------------

class TestModuleImports:
    """Every public class should be importable from src.agent."""

    def test_import_classifier(self):
        assert IntentClassifier is not None

    def test_import_retriever(self):
        assert EvidenceRetriever is not None

    def test_import_generator(self):
        assert ReplyGenerator is not None

    def test_import_escalation(self):
        assert EscalationPolicy is not None
        assert EscalationDecision is not None

    def test_import_pipeline(self):
        assert SupportPipeline is not None
        assert AgentResult is not None


# ---------------------------------------------------------------------------
# Data-class / dataclass defaults
# ---------------------------------------------------------------------------

class TestAgentResultDefaults:
    """AgentResult should initialise with sensible empty defaults."""

    def test_default_construction(self):
        r = AgentResult(customer_message="hello")
        assert r.customer_message == "hello"
        assert r.intent is None
        assert r.retrieved_evidence == []
        assert r.draft_reply is None
        assert r.decision is None
        assert r.escalation_reason is None
        assert r.metadata == {}


class TestIntentClassificationResultDefaults:
    def test_defaults(self):
        r = IntentClassificationResult(intent="Test")
        assert r.intent == "Test"
        assert r.confidence == 1.0
        assert r.probabilities == {}


class TestEvidenceDefaults:
    def test_defaults(self):
        e = Evidence(id="1", text="sample")
        assert e.id == "1"
        assert e.source == "twcs_historical"


class TestDraftReplyDefaults:
    def test_defaults(self):
        d = DraftReply(reply_text="Hi there")
        assert d.reply_text == "Hi there"
        assert d.grounded is True


class TestEscalationDecision:
    def test_enum_values(self):
        assert EscalationDecision.AUTO_HANDLE.value == "AUTO_HANDLE"
        assert EscalationDecision.ESCALATE_TO_HUMAN.value == "ESCALATE_TO_HUMAN"


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------

class TestComponentConstruction:
    """Each component should be constructable with no arguments."""

    def test_classifier_default(self):
        clf = IntentClassifier()
        assert len(clf.intents) == 10

    def test_retriever_default(self):
        ret = EvidenceRetriever()
        assert ret.index_path is None

    def test_generator_default(self):
        gen = ReplyGenerator()
        assert gen.model_name == "fake-model"

    def test_escalation_default(self):
        pol = EscalationPolicy()
        assert "Billing & Payments" in pol.sensitive_intents


# ---------------------------------------------------------------------------
# Gemini intent classifier
# ---------------------------------------------------------------------------

EXPECTED_INTENTS = [
    "Technical Troubleshooting",
    "Product / Feature How-To",
    "Account & Login",
    "Billing & Payments",
    "Order / Delivery",
    "Warranty / Repair",
    "Network / Connectivity",
    "Microsoft Store",
    "Complaint / Feedback",
    "Cancellation / Subscription",
]


class TestGeminiIntentClassifier:
    """Gemini-based classifier behavior (Gemini fully mocked)."""

    def test_loads_all_10_intents_from_yaml(self):
        clf = IntentClassifier()
        assert len(clf.intents) == 10
        assert clf.intents == EXPECTED_INTENTS

    def test_model_comes_from_env(self):
        clf = IntentClassifier()
        assert clf.model_name == "fake-model"

    def test_valid_response_is_parsed(self, monkeypatch):
        clf = IntentClassifier()
        monkeypatch.setattr(
            clf,
            "_generate",
            lambda msg: json.dumps(
                {
                    "intent": "Technical Troubleshooting",
                    "confidence": 0.92,
                    "reason": "The customer reports a Windows update problem.",
                }
            ),
        )
        result = clf.classify("Windows update fails every time.")
        assert result.intent == "Technical Troubleshooting"
        assert result.confidence == 0.92
        assert "Windows update problem" in result.reason
        assert result.probabilities["Technical Troubleshooting"] == 0.92
        assert result.metadata["source"] == "gemini"

    def test_invalid_intent_is_rejected(self, monkeypatch):
        clf = IntentClassifier()
        monkeypatch.setattr(
            clf,
            "_generate",
            lambda msg: json.dumps(
                {
                    "intent": "Alien Invasion",
                    "confidence": 0.99,
                    "reason": "Not in taxonomy.",
                }
            ),
        )
        with pytest.raises(InvalidIntentResponseError):
            clf.classify("Hello!")

    def test_malformed_json_is_rejected(self, monkeypatch):
        clf = IntentClassifier()
        monkeypatch.setattr(clf, "_generate", lambda msg: "this is not json {")
        with pytest.raises(InvalidIntentResponseError):
            clf.classify("Hello!")

    def test_non_object_response_is_rejected(self, monkeypatch):
        clf = IntentClassifier()
        monkeypatch.setattr(clf, "_generate", lambda msg: "[1, 2, 3]")
        with pytest.raises(InvalidIntentResponseError):
            clf.classify("Hello!")

    def test_confidence_is_clamped_to_unit_range(self, monkeypatch):
        clf = IntentClassifier()
        monkeypatch.setattr(
            clf,
            "_generate",
            lambda msg: json.dumps(
                {"intent": "Billing & Payments", "confidence": 1.7, "reason": "x"}
            ),
        )
        assert clf.classify("x").confidence == 1.0
        monkeypatch.setattr(
            clf,
            "_generate",
            lambda msg: json.dumps(
                {"intent": "Billing & Payments", "confidence": -0.3, "reason": "x"}
            ),
        )
        assert clf.classify("x").confidence == 0.0

    def test_non_numeric_confidence_is_rejected(self, monkeypatch):
        clf = IntentClassifier()
        monkeypatch.setattr(
            clf,
            "_generate",
            lambda msg: json.dumps(
                {"intent": "Billing & Payments", "confidence": "high", "reason": "x"}
            ),
        )
        with pytest.raises(InvalidIntentResponseError):
            clf.classify("Hello!")

    def test_missing_api_key_raises_configuration_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        clf = IntentClassifier(api_key=None)
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
            clf.classify("Hello!")


# ---------------------------------------------------------------------------
# Pipeline construction & execution
# ---------------------------------------------------------------------------

class TestSupportPipeline:
    """The pipeline should construct with defaults and run without errors."""

    def test_default_construction(self):
        pipe = SupportPipeline()
        assert pipe.classifier is not None
        assert pipe.retriever is not None
        assert pipe.generator is not None
        assert pipe.escalation is not None

    def test_run_returns_agent_result(self):
        pipe = SupportPipeline()
        result = pipe.run("My Surface laptop screen is flickering.")
        assert isinstance(result, AgentResult)

    def test_run_populates_intent(self):
        pipe = SupportPipeline()
        result = pipe.run("How do I reset my password?")
        assert result.intent is not None
        assert result.intent_result is not None

    def test_run_populates_evidence(self):
        pipe = SupportPipeline()
        result = pipe.run("My order hasn't arrived.")
        assert isinstance(result.retrieved_evidence, list)

    def test_run_populates_draft_reply(self):
        pipe = SupportPipeline()
        result = pipe.run("I want a refund.")
        assert result.draft_reply is not None
        assert len(result.draft_reply.reply_text) > 0

    def test_run_populates_escalation(self):
        pipe = SupportPipeline()
        result = pipe.run("Your product is terrible!")
        assert result.decision is not None
        assert isinstance(result.decision, EscalationDecision)

    def test_full_pipeline_fields(self):
        pipe = SupportPipeline()
        result = pipe.run("I can't connect to Wi-Fi after the update.")
        # All top-level fields should be populated
        assert result.customer_message == "I can't connect to Wi-Fi after the update."
        assert result.intent is not None
        assert result.intent_result is not None
        assert isinstance(result.retrieved_evidence, list)
        assert result.draft_reply is not None
        assert result.decision is not None
        assert isinstance(result.escalation_result, EscalationResult)


# ---------------------------------------------------------------------------
# Escalation logic sanity
# ---------------------------------------------------------------------------

class TestEscalationBehaviour:
    """Verify placeholder escalation produces expected decisions."""

    def test_complaint_escalates(self):
        pol = EscalationPolicy()
        reply = DraftReply(reply_text="Sorry.")
        result = pol.evaluate(
            customer_message="This is unacceptable!",
            intent="Complaint & Feedback",
            draft_reply=reply,
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN
        assert result.escalation_reason is not None

    def test_ungrounded_escalates(self):
        pol = EscalationPolicy()
        reply = DraftReply(reply_text="Sorry.", grounded=False)
        result = pol.evaluate(
            customer_message="Help me.",
            intent="Technical Troubleshooting",
            draft_reply=reply,
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN

    def test_normal_auto_handles(self):
        pol = EscalationPolicy()
        reply = DraftReply(reply_text="Here is the fix.", grounded=True)
        evidence = [Evidence(id="e1", text="How to backup photos in Windows.", score=0.85)]
        result = pol.evaluate(
            customer_message="How do I backup photos?",
            intent="Product / Feature How-To",
            draft_reply=reply,
            evidence=evidence,
        )
        assert result.decision == EscalationDecision.AUTO_HANDLE
        assert result.escalation_reason is None
