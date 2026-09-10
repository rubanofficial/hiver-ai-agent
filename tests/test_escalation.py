"""
Tests for the rule-based escalation policy (src.agent.escalation).

The escalation stage is deterministic and makes no API calls of any kind.
All inputs (evidence, draft replies, intents) are constructed in-memory.
"""

import pytest

from src.agent.escalation import EscalationPolicy, EscalationDecision
from src.agent.retriever import Evidence
from src.agent.reply_generator import DraftReply


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def policy():
    """Return an EscalationPolicy with default thresholds."""
    return EscalationPolicy()


def _evidence(*scores, text="Customer message with resolved outcome.") -> list:
    """Build a list of Evidence objects from similarity scores."""
    return [
        Evidence(id=f"conv_{i}", text=text, score=score)
        for i, score in enumerate(scores)
    ]


def _grounded_reply(text="Here is the fix.", confidence=0.9) -> DraftReply:
    return DraftReply(reply_text=text, grounded=True, confidence=confidence)


# ---------------------------------------------------------------------------
# Strong historical evidence -> auto-handle
# ---------------------------------------------------------------------------

class TestAutoHandleStrongEvidence:
    """Sufficient evidence with a grounded reply should be auto-handled."""

    def test_strong_evidence_auto_handles(self, policy):
        result = policy.evaluate(
            customer_message="How do I back up my photos in Windows?",
            intent="Product / Feature How-To",
            draft_reply=_grounded_reply(),
            evidence=_evidence(0.85, 0.72, 0.68),
        )
        assert result.decision == EscalationDecision.AUTO_HANDLE
        assert result.escalation_reason is None
        assert result.risk_score == 0.0

    def test_medium_evidence_auto_handles(self, policy):
        result = policy.evaluate(
            customer_message="Wi-Fi keeps disconnecting.",
            intent="Network / Connectivity",
            draft_reply=_grounded_reply(confidence=0.7),
            evidence=_evidence(0.65, 0.55),
        )
        assert result.decision == EscalationDecision.AUTO_HANDLE

    def test_auto_handle_confidence_capped(self, policy):
        result = policy.evaluate(
            customer_message="How do I reset my password?",
            intent="Account & Login",
            draft_reply=_grounded_reply(),
            evidence=_evidence(0.99),
        )
        assert result.decision == EscalationDecision.AUTO_HANDLE
        assert result.confidence == pytest.approx(0.95)

    def test_auto_handle_reason_is_none(self, policy):
        result = policy.evaluate(
            customer_message="How do I uninstall an app?",
            intent="Product / Feature How-To",
            draft_reply=_grounded_reply(),
            evidence=_evidence(0.80),
        )
        assert result.escalation_reason is None


# ---------------------------------------------------------------------------
# Weak / no evidence -> escalate
# ---------------------------------------------------------------------------

class TestEscalateNoOrWeakEvidence:
    """Missing or low-similarity evidence should escalate to a human."""

    def test_escalate_no_evidence(self, policy):
        result = policy.evaluate(
            customer_message="I have a strange issue nobody has reported.",
            intent="Technical Troubleshooting",
            draft_reply=_grounded_reply(),
            evidence=[],
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN
        assert result.escalation_reason is not None

    def test_escalate_none_evidence(self, policy):
        result = policy.evaluate(
            customer_message="I have a strange issue nobody has reported.",
            intent="Technical Troubleshooting",
            draft_reply=_grounded_reply(),
            evidence=None,
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN

    def test_escalate_weak_top_score(self, policy):
        result = policy.evaluate(
            customer_message="Something odd is happening.",
            intent="Technical Troubleshooting",
            draft_reply=_grounded_reply(confidence=0.4),
            evidence=_evidence(0.35, 0.30),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN

    def test_escalate_no_confident_evidence(self, policy):
        # Both scores are above the weak threshold but below the confidence one.
        result = policy.evaluate(
            customer_message="Something vague is going on.",
            intent="Technical Troubleshooting",
            draft_reply=_grounded_reply(confidence=0.5),
            evidence=_evidence(0.55, 0.45),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN


# ---------------------------------------------------------------------------
# Borderline similarity -> deterministic outcomes
# ---------------------------------------------------------------------------

class TestBorderlineSimilarity:
    """Scores exactly at the configured thresholds behave deterministically."""

    def test_borderline_below_escapes(self, policy):
        # 0.49 is below min_top_score (0.50) -> escalate.
        result = policy.evaluate(
            customer_message="Borderline case below threshold.",
            intent="Technical Troubleshooting",
            draft_reply=_grounded_reply(confidence=0.5),
            evidence=_evidence(0.49, 0.40),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN

    def test_borderline_at_threshold_auto_handles(self, policy):
        # 0.50 meets min_top_score but not min_confident_score (0.60) -> escalate.
        result = policy.evaluate(
            customer_message="Borderline case exactly at threshold.",
            intent="Technical Troubleshooting",
            draft_reply=_grounded_reply(confidence=0.5),
            evidence=_evidence(0.50),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN

    def test_high_enough_score_auto_handles(self, policy):
        # 0.60 meets min_confident_score -> auto-handle.
        result = policy.evaluate(
            customer_message="Borderline case above confidence threshold.",
            intent="Technical Troubleshooting",
            draft_reply=_grounded_reply(confidence=0.6),
            evidence=_evidence(0.60),
        )
        assert result.decision == EscalationDecision.AUTO_HANDLE

    def test_custom_config_thresholds(self):
        pol = EscalationPolicy(config={"min_top_score": 0.30, "min_confident_score": 0.40})
        result = pol.evaluate(
            customer_message="Custom thresholds in effect.",
            intent="Technical Troubleshooting",
            draft_reply=_grounded_reply(confidence=0.5),
            evidence=_evidence(0.45),
        )
        assert result.decision == EscalationDecision.AUTO_HANDLE

    def test_custom_config_below_confident_threshold(self):
        pol = EscalationPolicy(config={"min_top_score": 0.30, "min_confident_score": 0.40})
        result = pol.evaluate(
            customer_message="Custom thresholds in effect but score is low.",
            intent="Technical Troubleshooting",
            draft_reply=_grounded_reply(confidence=0.5),
            evidence=_evidence(0.35),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN


# ---------------------------------------------------------------------------
# High-risk / sensitive intents -> escalate
# ---------------------------------------------------------------------------

class TestEscalateHighRiskIntents:
    """Sensitive intents escalate even when evidence is strong."""

    def test_escalate_complaint_intent(self, policy):
        result = policy.evaluate(
            customer_message="I am very unhappy with the support I received.",
            intent="Complaint / Feedback",
            draft_reply=_grounded_reply(),
            evidence=_evidence(0.90, 0.80),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN
        assert "Complaint / Feedback" in result.escalation_reason

    def test_escalate_complaint_backcompat_alias(self, policy):
        result = policy.evaluate(
            customer_message="This is unacceptable!",
            intent="Complaint & Feedback",
            draft_reply=_grounded_reply(),
            evidence=_evidence(0.90, 0.80),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN

    def test_escalate_cancellation_intent(self, policy):
        result = policy.evaluate(
            customer_message="Please cancel my subscription immediately.",
            intent="Cancellation / Subscription",
            draft_reply=_grounded_reply(),
            evidence=_evidence(0.90, 0.80),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN
        assert "Cancellation / Subscription" in result.escalation_reason

    def test_non_sensitive_intent_auto_handles(self, policy):
        result = policy.evaluate(
            customer_message="How do I use dark mode?",
            intent="Product / Feature How-To",
            draft_reply=_grounded_reply(),
            evidence=_evidence(0.80, 0.70),
        )
        assert result.decision == EscalationDecision.AUTO_HANDLE


# ---------------------------------------------------------------------------
# Draft reply problems -> escalate
# ---------------------------------------------------------------------------

class TestEscalateReplyProblems:
    """A missing or ungrounded draft reply always escalates."""

    def test_escalate_missing_reply(self, policy):
        result = policy.evaluate(
            customer_message="Please help.",
            intent="Technical Troubleshooting",
            draft_reply=None,
            evidence=_evidence(0.80),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN

    def test_escalate_ungrounded_reply(self, policy):
        reply = DraftReply(reply_text="I am not sure about this.", grounded=False)
        result = policy.evaluate(
            customer_message="Please help.",
            intent="Technical Troubleshooting",
            draft_reply=reply,
            evidence=_evidence(0.80, 0.70),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN


# ---------------------------------------------------------------------------
# Reason handling
# ---------------------------------------------------------------------------

class TestReasonHandling:
    """Every escalation includes a reason; auto-handles do not fabricate one."""

    def test_reason_always_present_on_escalation(self, policy):
        reasons = [
            policy.evaluate("x", "Technical Troubleshooting", _grounded_reply(), []),
            policy.evaluate("x", "Technical Troubleshooting", None, _evidence(0.8)),
            policy.evaluate("x", "Technical Troubleshooting", _grounded_reply(), _evidence(0.2)),
            policy.evaluate("x", "Complaint / Feedback", _grounded_reply(), _evidence(0.9)),
        ]
        for result in reasons:
            assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN
            assert result.escalation_reason
            assert len(result.escalation_reason.strip()) > 0

    def test_metadata_contains_rule_and_top_score(self, policy):
        result = policy.evaluate(
            customer_message="Please cancel now.",
            intent="Cancellation / Subscription",
            draft_reply=_grounded_reply(),
            evidence=_evidence(0.90),
        )
        assert result.metadata["rule"] == "high_risk_intent"
        assert result.metadata["top_score"] == pytest.approx(0.90)

    def test_auto_handle_metadata(self, policy):
        result = policy.evaluate(
            customer_message="How do I update Windows?",
            intent="Product / Feature How-To",
            draft_reply=_grounded_reply(),
            evidence=_evidence(0.80),
        )
        assert result.metadata["rule"] is None
        assert result.metadata["evidence_count"] == 1
        assert result.metadata["top_score"] == pytest.approx(0.80)

    def test_escalation_result_values(self, policy):
        result = policy.evaluate(
            customer_message="I have a vague problem.",
            intent="Technical Troubleshooting",
            draft_reply=_grounded_reply(confidence=0.5),
            evidence=_evidence(0.55, 0.45),
        )
        assert result.decision == EscalationDecision.ESCALATE_TO_HUMAN
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.risk_score <= 1.0
        assert result.metadata["rule"] == "no_confident_evidence"