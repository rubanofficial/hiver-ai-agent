"""
Escalation Policy Module for MicrosoftHelps AI Support Agent.

Determines whether the agent should AUTO_HANDLE the response or
ESCALATE_TO_HUMAN, providing an explicit reason when escalating.

The decision is rule-based and fully explainable. Rules are evaluated
in order; the first matching rule determines the outcome.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .retriever import Evidence
from .reply_generator import DraftReply


class EscalationDecision(str, Enum):
    """Allowed escalation decisions for the support agent."""
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


@dataclass
class EscalationResult:
    """Outcome of evaluating the escalation policy."""
    decision: EscalationDecision
    escalation_reason: Optional[str] = None
    confidence: float = 1.0
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EscalationPolicy:
    """
    Evaluates safety, confidence, intent sensitivity, and evidence
    sufficiency to decide between autonomous handling vs human escalation.

    All rules are explicit and ordered. Only the signals already available
    in the pipeline are used; no LLM is involved in the decision.

    Rules (first match wins):
        1. No draft reply -> ESCALATE
        2. Reply not grounded -> ESCALATE
        3. No evidence retrieved -> ESCALATE
        4. Best evidence similarity below min_top_score -> ESCALATE
        5. High-risk / sensitive intent -> ESCALATE
        6. No evidence matching the confidence threshold -> ESCALATE
        7. Otherwise -> AUTO_HANDLE
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "min_top_score": 0.50,
        "min_confident_score": 0.60,
        "high_risk_intents": {
            "Complaint / Feedback",
            "Cancellation / Subscription",
            "Billing & Payments",
            # Backward-compatible alias used by earlier tests/callers.
            "Complaint & Feedback",
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize policy rules and thresholds.

        Args:
            config: Optional dictionary overriding any DEFAULT_CONFIG setting.
        """
        merged = dict(self.DEFAULT_CONFIG)
        if config:
            merged.update(config)
        self.config: Dict[str, Any] = merged

        high_risk = self.config.get("high_risk_intents") or set()
        self.high_risk_intents = set(high_risk)
        self.min_top_score = float(self.config.get("min_top_score", 0.50))
        self.min_confident_score = float(self.config.get("min_confident_score", 0.60))

    @property
    def sensitive_intents(self) -> set:
        """Backward-compatible alias for the high-risk intent set."""
        return set(self.high_risk_intents)

    # -- public API ---------------------------------------------------------

    def evaluate(
        self,
        customer_message: str,
        intent: Optional[str],
        draft_reply: Optional[DraftReply],
        evidence: Optional[List[Evidence]] = None,
    ) -> EscalationResult:
        """
        Evaluate customer message, draft reply, and evidence to decide escalation.

        Args:
            customer_message: Incoming customer text.
            intent: Classified intent name.
            draft_reply: Generated candidate reply.
            evidence: Retrieved historical context.

        Returns:
            EscalationResult with decision (AUTO_HANDLE / ESCALATE_TO_HUMAN),
            an explanation, a confidence score, and a risk score.
        """
        evidence_list = list(evidence) if evidence else []
        top_score = max((e.score for e in evidence_list), default=0.0)

        rule = self._check_escalation_rules(customer_message, intent, draft_reply, evidence_list, top_score)
        if rule is not None:
            rule_name, reason, confidence, risk_score = rule
            return self._escalate(reason, confidence, risk_score, top_score, rule_name)

        confirmed_score = max(
            (e.score for e in evidence_list if e.score >= self.min_confident_score),
            default=top_score,
        )
        decision_confidence = max(0.0, min(confirmed_score, 0.95))

        return EscalationResult(
            decision=EscalationDecision.AUTO_HANDLE,
            escalation_reason=None,
            confidence=decision_confidence,
            risk_score=0.0,
            metadata={
                "top_score": top_score,
                "evidence_count": len(evidence_list),
                "rule": None,
            },
        )

    # -- internal helpers ---------------------------------------------------

    def _check_escalation_rules(
        self,
        customer_message: str,
        intent: Optional[str],
        draft_reply: Optional[DraftReply],
        evidence: List[Evidence],
        top_score: float,
    ) -> Optional[tuple]:
        """
        Return the first matching rule as (name, reason, confidence, risk_score),
        or None when every rule passes.
        """
        evidence_count = len(evidence)

        # Rule 1 — no draft reply was produced.
        if draft_reply is None:
            return (
                "no_draft_reply",
                "No draft reply was generated; human review is required.",
                0.9,
                0.9,
            )

        # Rule 2 — the reply could not be grounded in evidence.
        if draft_reply.grounded is False:
            return (
                "ungrounded_reply",
                "Generated reply could not be grounded in historical evidence; "
                "human review is required.",
                draft_reply.confidence or 0.5,
                0.8,
            )

        # Rule 3 — no evidence was retrieved at all.
        if evidence_count == 0:
            return (
                "no_evidence",
                "No historical evidence was retrieved for this customer "
                "message; human review is required.",
                0.6,
                0.7,
            )

        # Rule 4 — the best matching evidence is still too far away.
        if top_score < self.min_top_score:
            return (
                "low_similarity",
                f"Best evidence similarity ({top_score:.2f}) is below the "
                f"auto-handle threshold ({self.min_top_score:.2f}); human "
                "review is required.",
                0.5,
                0.7,
            )

        # Rule 5 — high-risk or sensitive intent handling.
        if intent in self.high_risk_intents:
            return (
                "high_risk_intent",
                f"Intent '{intent}' is high-risk and requires human handling.",
                0.9,
                0.8,
            )

        # Rule 6 — no evidence reached the confidence threshold.
        confirmed = [e for e in evidence if e.score >= self.min_confident_score]
        if not confirmed:
            return (
                "no_confident_evidence",
                "No evidence reached the confidence threshold for a safe "
                f"autonomous reply (minimum {self.min_confident_score:.2f}); "
                "human review is required.",
                0.5,
                0.7,
            )

        return None

    def _escalate(
        self,
        reason: str,
        confidence: float,
        risk_score: float,
        top_score: float,
        rule_name: str,
    ) -> EscalationResult:
        """Build an ESCALATE_TO_HUMAN result with shared metadata."""
        return EscalationResult(
            decision=EscalationDecision.ESCALATE_TO_HUMAN,
            escalation_reason=reason,
            confidence=confidence,
            risk_score=risk_score,
            metadata={
                "top_score": top_score,
                "rule": rule_name,
            },
        )