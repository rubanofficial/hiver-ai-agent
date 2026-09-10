"""
MicrosoftHelps AI Support Agent Package.

Provides the core modules for intent classification, evidence retrieval,
grounded reply generation, and escalation policy evaluation.
"""

from .intent_classifier import (
    IntentClassifier,
    IntentClassificationResult,
    IntentClassifierError,
    ConfigurationError,
    InvalidIntentResponseError,
)
from .retriever import EvidenceRetriever, Evidence
from .reply_generator import ReplyGenerator, DraftReply
from .escalation import EscalationPolicy, EscalationDecision, EscalationResult
from .pipeline import SupportPipeline, AgentResult

__all__ = [
    "IntentClassifier",
    "IntentClassificationResult",
    "IntentClassifierError",
    "ConfigurationError",
    "InvalidIntentResponseError",
    "EvidenceRetriever",
    "Evidence",
    "ReplyGenerator",
    "DraftReply",
    "EscalationPolicy",
    "EscalationDecision",
    "EscalationResult",
    "SupportPipeline",
    "AgentResult",
]
