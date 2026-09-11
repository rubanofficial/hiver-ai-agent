"""
Tests for app.py (Streamlit UI interface).

Verifies UI structure and imports without invoking real external Gemini API calls.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.agent.pipeline import AgentResult
from src.agent.intent_classifier import IntentClassificationResult
from src.agent.retriever import Evidence
from src.agent.reply_generator import DraftReply
from src.agent.escalation import EscalationDecision, EscalationResult


def test_app_imports():
    """Verify that app module can be imported cleanly."""
    import app
    assert hasattr(app, "load_support_pipeline")
    assert hasattr(app, "main")


@patch("app.IntentClassifier")
@patch("app.EvidenceRetriever")
@patch("app.ReplyGenerator")
@patch("app.EscalationPolicy")
def test_load_support_pipeline(mock_esc, mock_gen, mock_ret, mock_clf):
    """Verify pipeline instantiation without network calls."""
    import app

    # Clear cached resource for testing
    app.load_support_pipeline.clear()

    pipeline, model_name, index_path = app.load_support_pipeline()
    assert pipeline is not None
    assert isinstance(model_name, str)
    assert isinstance(index_path, str)


def test_mock_pipeline_result_structure():
    """Verify AgentResult data contract matches what the UI expects."""
    intent_res = IntentClassificationResult(
        intent="Technical Troubleshooting",
        confidence=0.95,
        reason="Device failure reported",
    )
    evidence_items = [
        Evidence(
            id="12345",
            text="Have you tried restarting the Surface?",
            score=0.88,
        )
    ]
    draft = DraftReply(reply_text="Please try restarting your device.")
    esc_res = EscalationResult(
        decision=EscalationDecision.AUTO_HANDLE,
        escalation_reason=None,
    )

    result = AgentResult(
        customer_message="My surface won't turn on",
        intent=intent_res.intent,
        intent_result=intent_res,
        retrieved_evidence=evidence_items,
        draft_reply=draft,
        decision=esc_res.decision,
        escalation_reason=esc_res.escalation_reason,
        escalation_result=esc_res,
    )

    assert result.intent == "Technical Troubleshooting"
    assert result.intent_result.confidence == 0.95
    assert len(result.retrieved_evidence) == 1
    assert result.retrieved_evidence[0].id == "12345"
    assert result.draft_reply.reply_text == "Please try restarting your device."
    assert result.decision == EscalationDecision.AUTO_HANDLE
