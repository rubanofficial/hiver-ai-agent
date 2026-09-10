"""
Pipeline Orchestrator for MicrosoftHelps AI Support Agent.

Coordinates the end-to-end support workflow:
    customer_message
        -> intent classification
        -> evidence retrieval
        -> grounded reply generation
        -> escalation decision
        -> final agent output (AgentResult)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .intent_classifier import IntentClassifier, IntentClassificationResult
from .retriever import EvidenceRetriever, Evidence
from .reply_generator import ReplyGenerator, DraftReply
from .escalation import EscalationPolicy, EscalationDecision, EscalationResult


@dataclass
class AgentResult:
    """
    The complete output of the support-agent pipeline for a single customer message.

    Attributes:
        customer_message: The original inbound customer text.
        intent: Classified intent label (e.g. "Technical Troubleshooting").
        intent_result: Full intent classification output including confidence.
        retrieved_evidence: Evidence items retrieved for grounding.
        draft_reply: The generated draft reply.
        decision: Escalation decision (AUTO_HANDLE or ESCALATE_TO_HUMAN).
        escalation_reason: Human-readable reason when escalating.
        escalation_result: Full escalation output.
        metadata: Arbitrary pipeline metadata (timings, flags, etc.).
    """

    customer_message: str
    intent: Optional[str] = None
    intent_result: Optional[IntentClassificationResult] = None
    retrieved_evidence: List[Evidence] = field(default_factory=list)
    draft_reply: Optional[DraftReply] = None
    decision: Optional[EscalationDecision] = None
    escalation_reason: Optional[str] = None
    escalation_result: Optional[EscalationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SupportPipeline:
    """
    Orchestrates the full MicrosoftHelps support-agent workflow.

    Stages (in order):
        1. Intent Classification
        2. Evidence Retrieval
        3. Grounded Reply Generation
        4. Escalation Decision

    Each stage uses a pluggable component.  All components default to
    their skeleton implementations so the pipeline is importable and
    testable without any external services.
    """

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[EvidenceRetriever] = None,
        generator: Optional[ReplyGenerator] = None,
        escalation: Optional[EscalationPolicy] = None,
    ) -> None:
        """
        Initialise the pipeline with individual stage components.

        Args:
            classifier: Intent classification component.
            retriever: Evidence retrieval component.
            generator: Reply generation component.
            escalation: Escalation evaluation component.
        """
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or EvidenceRetriever()
        self.generator = generator or ReplyGenerator()
        self.escalation = escalation or EscalationPolicy()

    def run(
        self,
        customer_message: str,
        top_k: int = 3,
    ) -> AgentResult:
        """
        Execute the full support pipeline for a single customer message.

        Args:
            customer_message: Raw text from the customer.
            top_k: Number of evidence candidates to retrieve.

        Returns:
            AgentResult containing every stage's output.
        """
        result = AgentResult(customer_message=customer_message)

        # Stage 1 — Intent classification
        result.intent_result = self.classifier.classify(customer_message)
        result.intent = result.intent_result.intent

        # Stage 2 — Evidence retrieval
        result.retrieved_evidence = self.retriever.retrieve(
            query=customer_message,
            intent=result.intent,
            top_k=top_k,
        )

        # Stage 3 — Reply generation
        result.draft_reply = self.generator.generate(
            customer_message=customer_message,
            intent=result.intent,
            evidence=result.retrieved_evidence,
        )

        # Stage 4 — Escalation decision
        result.escalation_result = self.escalation.evaluate(
            customer_message=customer_message,
            intent=result.intent,
            draft_reply=result.draft_reply,
            evidence=result.retrieved_evidence,
        )
        result.decision = result.escalation_result.decision
        result.escalation_reason = result.escalation_result.escalation_reason

        return result
