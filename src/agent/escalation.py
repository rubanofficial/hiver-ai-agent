"""
Escalation Policy Module for MicrosoftHelps AI Support Agent.

Eventually determines whether the agent should AUTO_HANDLE the response
or ESCALATE_TO_HUMAN, providing an explicit reason when escalating.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
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
    Evaluates safety, confidence, intent sensitivity, and evidence sufficiency
    to decide between autonomous handling vs human escalation.
    
    Current implementation is a skeleton / placeholder.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize policy rules and thresholds.
        
        Args:
            config: Optional dictionary with custom risk thresholds or intent rules.
        """
        self.config = config or {}
        # Sensitive intents that may require default escalation
        self.sensitive_intents = {"Billing & Payments", "Account & Login", "Complaint & Feedback"}

    def evaluate(
        self,
        customer_message: str,
        intent: Optional[str],
        draft_reply: Optional[DraftReply],
        evidence: Optional[List[Evidence]] = None
    ) -> EscalationResult:
        """
        Evaluate customer message, draft reply, and evidence to decide escalation.
        
        Args:
            customer_message: Incoming customer text.
            intent: Classified intent name.
            draft_reply: Generated candidate reply.
            evidence: Retrieved historical context.
            
        Returns:
            EscalationResult with decision (AUTO_HANDLE / ESCALATE_TO_HUMAN) and reason.
        """
        # Placeholder logic: escalate if intent is Complaint & Feedback, otherwise AUTO_HANDLE
        if intent == "Complaint & Feedback":
            return EscalationResult(
                decision=EscalationDecision.ESCALATE_TO_HUMAN,
                escalation_reason="Customer expressed severe dissatisfaction requiring human oversight.",
                confidence=0.9,
                risk_score=0.85
            )

        if not draft_reply or not draft_reply.grounded:
            return EscalationResult(
                decision=EscalationDecision.ESCALATE_TO_HUMAN,
                escalation_reason="Insufficient grounded evidence to generate reliable autonomous answer.",
                confidence=0.8,
                risk_score=0.7
            )

        return EscalationResult(
            decision=EscalationDecision.AUTO_HANDLE,
            escalation_reason=None,
            confidence=0.9,
            risk_score=0.1
        )
