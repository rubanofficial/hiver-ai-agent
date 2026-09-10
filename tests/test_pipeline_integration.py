"""
Focused integration tests for the complete support-agent pipeline wiring.

Verifies the end-to-end flow with real pipeline orchestration but fully
mocked AI calls:

    customer_message
        -> intent classification (stub classifier)
        -> evidence retrieval    (stub retriever)
        -> grounded reply        (real ReplyGenerator, Gemini mocked)
        -> escalation decision   (real EscalationPolicy)

No real Gemini API is called, no FAISS index is built, and no real
TWCS data is scanned. Evidence is injected as synthetic in-memory data.
"""

import json

import pytest

from src.agent import (
    IntentClassifier,
    IntentClassificationResult,
    InvalidIntentResponseError,
    Evidence,
    ReplyGenerator,
    EscalationDecision,
    SupportPipeline,
)
from src.agent.reply_generator import (
    ConfigurationError as ReplyGeneratorConfigError,
    InvalidReplyResponseError,
)


# ---------------------------------------------------------------------------
# Synthetic components
# ---------------------------------------------------------------------------

class StubClassifier:
    """A deterministic intent classifier for injection into the pipeline."""

    def __init__(self, intent="Product / Feature How-To", confidence=0.95):
        self.intent = intent
        self.confidence = confidence

    def classify(self, customer_message):
        return IntentClassificationResult(
            intent=self.intent,
            confidence=self.confidence,
            reason="Synthetic test classification.",
        )


class StubRetriever:
    """A deterministic retriever returning a fixed list of Evidence records."""

    def __init__(self, results=None):
        self.results = results or []
        self.last_query = None
        self.last_intent = None
        self.last_top_k = None

    def retrieve(self, query, intent=None, top_k=5):
        self.last_query = query
        self.last_intent = intent
        self.last_top_k = top_k
        return list(self.results)


USEFUL_EVIDENCE = [
    Evidence(id="c1", text="Flickering screen fixed by updating display drivers.", score=0.88),
    Evidence(id="c2", text="Display driver update resolved a similar issue.", score=0.81),
    Evidence(id="c3", text="Restarting after the driver update completed the fix.", score=0.72),
]

WEAK_EVIDENCE = [
    Evidence(id="w1", text="Only loosely related historical case.", score=0.31),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_gemini(monkeypatch):
    """Set a fake API key and stub the Gemini reply call for every test."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key")
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")

    def _fake_reply(self, customer_message, intent, evidence):
        return json.dumps(
            {
                "reply_text": "Here is a grounded reply based on historical evidence.",
                "confidence": 0.9,
                "grounded": bool(evidence),
            }
        )

    monkeypatch.setattr(ReplyGenerator, "_generate", _fake_reply)


def _build_pipeline(retriever=None, classifier=None, generator=None):
    """Build a pipeline with synthetic components and the real escalation stage."""
    return SupportPipeline(
        classifier=classifier or StubClassifier(),
        retriever=retriever or StubRetriever(),
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Complete pipeline flow / exposed results
# ---------------------------------------------------------------------------

class TestCompletePipeline:
    """The final AgentResult must expose every stage's output."""

    def test_all_stage_results_are_exposed(self):
        retriever = StubRetriever(USEFUL_EVIDENCE)
        classifier = StubClassifier(intent="Product / Feature How-To", confidence=0.95)
        result = _build_pipeline(retriever=retriever, classifier=classifier).run(
            "How do I stop my screen from flickering?",
        )

        # Classified intent + intent confidence
        assert result.intent == "Product / Feature How-To"
        assert result.intent_result.confidence == 0.95

        # Retrieved historical evidence
        assert len(result.retrieved_evidence) == 3

        # Generated reply
        assert result.draft_reply is not None
        assert len(result.draft_reply.reply_text) > 0

        # Escalation decision + reason
        assert result.decision == EscalationDecision.AUTO_HANDLE
        assert result.escalation_reason is None

    def test_retriever_receives_query_intent_and_top_k(self):
        retriever = StubRetriever(USEFUL_EVIDENCE)
        classifier = StubClassifier(intent="Network / Connectivity")
        _build_pipeline(retriever=retriever, classifier=classifier).run(
            "Wi-Fi drops after every Windows update.",
            top_k=3,
        )
        assert retriever.last_query == "Wi-Fi drops after every Windows update."
        assert retriever.last_intent == "Network / Connectivity"
        assert retriever.last_top_k == 3


# ---------------------------------------------------------------------------
# Normal customer question with useful evidence
# ---------------------------------------------------------------------------

class TestUsefulEvidence:
    """Strong historical evidence should flow through to an auto-handle."""

    def test_normal_question_auto_handles(self):
        result = _build_pipeline(retriever=StubRetriever(USEFUL_EVIDENCE)).run(
            "How do I back up my photos to OneDrive?",
        )
        assert result.decision == EscalationDecision.AUTO_HANDLE
        assert result.escalation_reason is None
        assert result.draft_reply.grounded is True
        assert result.escalation_result.risk_score == 0.0


# ---------------------------------------------------------------------------
# Customer question with no useful evidence
# ---------------------------------------------------------------------------

class TestNoUsefulEvidence:
    """Missing evidence should escalate through the real escalation rules."""

    def test_no_evidence_escalates(self):
        result = _build_pipeline(retriever=StubRetriever([])).run(
            "I have a unique problem nobody has reported.",
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN
        assert result.escalation_reason is not None
        assert len(result.escalation_reason.strip()) > 0
        assert result.draft_reply.grounded is False

    def test_weak_evidence_escalates(self):
        result = _build_pipeline(retriever=StubRetriever(WEAK_EVIDENCE)).run(
            "Something odd is happening with my device.",
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN
        assert "similarity" in result.escalation_reason


# ---------------------------------------------------------------------------
# Retrieval returning multiple evidence records
# ---------------------------------------------------------------------------

class TestMultipleEvidence:
    """Every retrieved record should flow through generate and escalate."""

    def test_multiple_records_flow_through_generate_and_escalate(self):
        retriever = StubRetriever(USEFUL_EVIDENCE)
        result = _build_pipeline(retriever=retriever).run(
            "My screen is flickering after the last update.",
        )

        assert len(result.retrieved_evidence) == 3
        # Reply generator received all records (citations come from evidence).
        assert result.draft_reply.citations == ["c1", "c2", "c3"]
        assert result.draft_reply.metadata["evidence_count"] == 3
        # Escalation stage received all records.
        assert result.escalation_result.metadata["evidence_count"] == 3


# ---------------------------------------------------------------------------
# Escalation cases
# ---------------------------------------------------------------------------

class TestEscalationCases:
    """High-risk intents escalate even with strong evidence."""

    def test_sensitive_intent_escalates(self):
        classifier = StubClassifier(intent="Complaint / Feedback")
        result = _build_pipeline(
            retriever=StubRetriever(USEFUL_EVIDENCE),
            classifier=classifier,
        ).run("I am very unhappy with the support I received.")

        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN
        assert "Complaint / Feedback" in result.escalation_reason


# ---------------------------------------------------------------------------
# Gemini failure / error paths
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    """Supported AI-stage errors should propagate through the pipeline."""

    def test_classifier_malformed_response_propagates(self, monkeypatch):
        clf = IntentClassifier()
        monkeypatch.setattr(clf, "_generate", lambda msg: "not json {")
        pipe = SupportPipeline(classifier=clf, retriever=StubRetriever(USEFUL_EVIDENCE))
        with pytest.raises(InvalidIntentResponseError):
            pipe.run("Hello there!")

    def test_reply_generator_invalid_response_propagates(self, monkeypatch):
        gen = ReplyGenerator()
        monkeypatch.setattr(gen, "_generate", lambda msg, intent, ev: "")
        pipe = SupportPipeline(
            classifier=StubClassifier(),
            retriever=StubRetriever(USEFUL_EVIDENCE),
            generator=gen,
        )
        with pytest.raises(InvalidReplyResponseError):
            pipe.run("How do I fix my screen?")

    def test_missing_api_key_propagates(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        gen = ReplyGenerator(api_key=None)
        pipe = SupportPipeline(
            classifier=StubClassifier(),
            retriever=StubRetriever(USEFUL_EVIDENCE),
            generator=gen,
        )
        with pytest.raises(ReplyGeneratorConfigError, match="GEMINI_API_KEY"):
            pipe.run("How do I update Windows?")