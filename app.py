"""
MicrosoftHelps AI Support Agent - Interactive Demo Interface.

Provides a clean Streamlit interface to test the production AI support agent pipeline.
Uses the exact production pipeline (IntentClassifier, EvidenceRetriever,
ReplyGenerator, EscalationPolicy) without duplicating any agent logic.
"""

from __future__ import annotations

import os
from typing import Tuple

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


def main() -> None:
    st.set_page_config(
        page_title="MicrosoftHelps AI Support Agent",
        page_icon="🤖",
        layout="centered",
    )

    st.title("MicrosoftHelps AI Support Agent")
    st.markdown(
        "An AI-powered first-line customer support assistant for **@MicrosoftHelps**, "
        "providing automated intent triage, historical evidence retrieval, "
        "grounded draft replies, and safety escalation."
    )
    st.divider()

    # Inbound message input
    customer_message = st.text_area(
        "Customer message",
        height=130,
        placeholder="e.g. @MicrosoftHelps Is there anyway to update the shipping address on an existing Microsoft Store order? I just recently moved.",
    )

    run_clicked = st.button("Run Support Agent", type="primary", use_container_width=True)

    if run_clicked:
        clean_input = customer_message.strip() if customer_message else ""
        if not clean_input:
            st.warning("Please enter a customer message to run the support agent.")
            return

        with st.spinner("Executing production pipeline (classify -> retrieve -> draft -> evaluate)..."):
            try:
                pipeline, model_name, embedding_index = load_support_pipeline()
                result: AgentResult = pipeline.run(customer_message=clean_input, top_k=3)
            except Exception as exc:
                # Friendly error message without leaking sensitive credentials
                err_msg = str(exc)
                if "API_KEY" in err_msg.upper() or "API KEY" in err_msg.upper():
                    st.error("API configuration error: GEMINI_API_KEY is not set or invalid.")
                else:
                    st.error(f"Pipeline error: {type(exc).__name__}: {err_msg}")
                return

        # ------------------------------------------------------------------
        # Display Results
        # ------------------------------------------------------------------
        st.subheader("Results")

        # 1. Intent Section
        st.markdown("### Intent")
        intent_name = result.intent or "Unknown"
        confidence = (
            result.intent_result.confidence
            if result.intent_result
            else 1.0
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Predicted Intent:** `{intent_name}`")
        with col2:
            st.markdown(f"**Confidence:** `{confidence * 100:.1f}%`")

        if result.intent_result and result.intent_result.reason:
            st.caption(f"Reason: {result.intent_result.reason}")

        st.divider()

        # 2. Retrieved Evidence Section
        st.markdown("### Retrieved Evidence")
        evidence_items = result.retrieved_evidence or []
        if not evidence_items:
            st.info("No historical evidence retrieved for this query.")
        else:
            for idx, item in enumerate(evidence_items, start=1):
                score_pct = item.score * 100
                with st.expander(
                    f"Evidence #{idx} (Conversation ID: {item.id} | Relevance: {score_pct:.1f}%)",
                    expanded=(idx == 1),
                ):
                    st.markdown(f"**Similarity Score:** `{item.score:.4f}`")
                    st.markdown(f"**Conversation / Source ID:** `{item.id}`")
                    st.markdown("**Historical Context / Text:**")
                    st.text(item.text)

        st.divider()

        # 3. Draft Reply Section
        st.markdown("### Draft Reply")
        reply_text = (
            result.draft_reply.reply_text
            if result.draft_reply and result.draft_reply.reply_text
            else "No draft reply generated."
        )
        st.info(reply_text)

        st.divider()

        # 4. Escalation Decision Section
        st.markdown("### Escalation Decision")
        is_escalate = (
            result.decision == EscalationDecision.ESCALATE_TO_HUMAN
            or str(result.decision) == "ESCALATE_TO_HUMAN"
        )
        decision_label = (
            result.decision.value
            if hasattr(result.decision, "value")
            else str(result.decision)
        )

        if is_escalate:
            st.error(f"🚨 **Decision:** `{decision_label}`")
        else:
            st.success(f"✅ **Decision:** `{decision_label}`")

        if result.escalation_reason:
            st.markdown(f"**Reason:** {result.escalation_reason}")
        else:
            st.caption("Auto-handled: response is safely grounded with sufficient evidence confidence.")

        st.divider()

        # 5. Optional Technical Details (Collapsible)
        with st.expander("Technical Details", expanded=False):
            st.markdown(f"- **Underlying Model:** `{model_name}`")
            st.markdown(f"- **Evidence Index:** `{embedding_index}`")
            st.markdown(f"- **Retrieved Items Count:** `{len(evidence_items)}`")
            st.markdown(f"- **Retrieval Top-K Config:** `3`")
            if result.metadata:
                st.json(result.metadata)


if __name__ == "__main__":
    main()
