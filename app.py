"""
Microsoft Support AI - Customer Support Chatbot Interface.

A clean, professional customer-support chatbot frontend powered by the
production MicrosoftHelps AI support agent pipeline (IntentClassifier,
EvidenceRetriever, ReplyGenerator, EscalationPolicy).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import streamlit as st

from src.agent.pipeline import SupportPipeline, AgentResult
from src.agent.intent_classifier import IntentClassifier
from src.agent.retriever import EvidenceRetriever
from src.agent.reply_generator import ReplyGenerator
from src.agent.escalation import EscalationPolicy, EscalationDecision


@st.cache_resource
def load_support_pipeline() -> Tuple[SupportPipeline, str, str]:
    """
    Instantiate and cache the production pipeline.
    
    Loads configuration from environment variables (.env).
    Returns the pipeline instance, resolved model name, and embedding index path.
    """
    load_dotenv()
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    embedding_index = os.environ.get(
        "AGENT_EMBEDDING_INDEX", "evaluation/twcs_evidence_index.json"
    )

    classifier = IntentClassifier(model_name=model_name)
    if embedding_index and os.path.exists(embedding_index):
        retriever = EvidenceRetriever.from_embedding_json(embedding_index)
    else:
        retriever = EvidenceRetriever()
    generator = ReplyGenerator(model_name=model_name)
    escalation = EscalationPolicy()

    pipeline = SupportPipeline(
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        escalation=escalation,
    )
    return pipeline, model_name, embedding_index


def render_decision_details(details: Dict[str, Any]) -> None:
    """Render compact decision details inside an expander for evaluators/inspectors."""
    with st.expander("🔍 View AI decision details", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Predicted Intent:** `{details.get('intent', 'Unknown')}`")
            conf = details.get("confidence")
            if conf is not None:
                st.markdown(f"**Confidence:** `{conf * 100:.1f}%`")
            else:
                st.markdown("**Confidence:** `N/A`")
        with c2:
            st.markdown(f"**Escalation Decision:** `{details.get('decision', 'AUTO_HANDLE')}`")
            st.markdown(f"**Evidence Cases Found:** `{details.get('evidence_count', 0)}`")

        esc_reason = details.get("escalation_reason")
        if esc_reason:
            st.markdown(f"**Escalation Reason:** {esc_reason}")

        evidence_items = details.get("evidence_items", [])
        if evidence_items:
            st.markdown("**Retrieved Knowledge / Cases:**")
            for idx, item in enumerate(evidence_items, start=1):
                score = item.get("score", 0.0)
                text = item.get("text", "")
                st.caption(f"**Case #{idx}** (Relevance Score: `{score:.2f}`)")
                st.text(text[:300] + ("..." if len(text) > 300 else ""))


def main() -> None:
    st.set_page_config(
        page_title="Microsoft Support AI",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Custom styling for a polished, clean Microsoft support chat experience
    st.markdown(
        """
        <style>
        /* Header typography & spacing */
        .main-header {
            margin-bottom: 0.2rem;
            font-weight: 700;
            color: #0078D4;
        }
        .sub-header {
            color: #555555;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }
        /* Status indicator badges */
        .status-badge-auto {
            display: inline-block;
            color: #107C41;
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 0.35rem;
            margin-bottom: 0.5rem;
        }
        .status-badge-escalate {
            display: inline-block;
            color: #D83B01;
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 0.35rem;
            margin-bottom: 0.5rem;
        }
        /* Chat message container styling */
        .stChatMessage {
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 1. Header
    st.markdown("<h1 class='main-header'>Microsoft Support AI</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>AI-powered customer support assistant for Microsoft products & services</div>",
        unsafe_allow_html=True,
    )

    # Initialize conversation history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your Microsoft Support AI assistant. How can I help you today?",
                "status_badge": None,
                "decision_details": None,
            }
        ]

    # Sidebar controls (Clean & minimal)
    with st.sidebar:
        st.subheader("Support Session")
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I am your Microsoft Support AI assistant. How can I help you today?",
                    "status_badge": None,
                    "decision_details": None,
                }
            ]
            st.rerun()
        st.caption("Powered by MicrosoftHelps Support Agent")

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("status_badge"):
                st.markdown(msg["status_badge"], unsafe_allow_html=True)
            if msg.get("decision_details"):
                render_decision_details(msg["decision_details"])

    # Chat input at bottom
    user_input = st.chat_input("Describe your problem...")

    if user_input:
        clean_input = user_input.strip()
        if not clean_input:
            return

        # Append user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": clean_input,
                "status_badge": None,
                "decision_details": None,
            }
        )

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(clean_input)

        # Execute SupportPipeline
        with st.chat_message("assistant"):
            with st.spinner("Connecting with Microsoft Support knowledge base..."):
                try:
                    pipeline, _, _ = load_support_pipeline()
                    result: AgentResult = pipeline.run(customer_message=clean_input, top_k=3)

                    reply_text = (
                        result.draft_reply.reply_text
                        if result.draft_reply and result.draft_reply.reply_text
                        else "I understand your query. A Microsoft support representative will assist you further."
                    )

                    # Determine escalation decision
                    is_escalate = (
                        result.decision == EscalationDecision.ESCALATE_TO_HUMAN
                        or str(result.decision) == "ESCALATE_TO_HUMAN"
                        or (hasattr(result.decision, "value") and result.decision.value == "ESCALATE_TO_HUMAN")
                    )

                    if is_escalate:
                        status_badge = "<span class='status-badge-escalate'>⚠️ Human support recommended</span>"
                    else:
                        status_badge = "<span class='status-badge-auto'>✓ Automatically handled</span>"

                    # Prepare decision details for collapsible view
                    confidence = (
                        result.intent_result.confidence
                        if result.intent_result
                        else 1.0
                    )
                    evidence_list = []
                    if result.retrieved_evidence:
                        for ev in result.retrieved_evidence:
                            evidence_list.append({
                                "id": getattr(ev, "id", "N/A"),
                                "score": getattr(ev, "score", 0.0),
                                "text": getattr(ev, "text", ""),
                            })

                    decision_details = {
                        "intent": result.intent or "General Inquiry",
                        "confidence": confidence,
                        "decision": result.decision.value if hasattr(result.decision, "value") else str(result.decision),
                        "escalation_reason": result.escalation_reason,
                        "evidence_count": len(result.retrieved_evidence or []),
                        "evidence_items": evidence_list,
                    }

                    # Render response
                    st.markdown(reply_text)
                    st.markdown(status_badge, unsafe_allow_html=True)
                    render_decision_details(decision_details)

                    # Save to conversation history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply_text,
                            "status_badge": status_badge,
                            "decision_details": decision_details,
                        }
                    )

                except Exception:
                    fallback_text = (
                        "I apologize, but I am currently unable to process your request. "
                        "Please try again in a moment, or reach out to Microsoft Support directly."
                    )
                    st.markdown(fallback_text)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": fallback_text,
                            "status_badge": None,
                            "decision_details": None,
                        }
                    )


if __name__ == "__main__":
    main()
