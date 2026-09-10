# MicrosoftHelps Intent Taxonomy — Draft Notes

## 1. Overview & Purpose
This document accompanies `analysis/microsoft_intents.csv` as the **First Draft Intent Taxonomy** for `MicrosoftHelps`. Having selected MicrosoftHelps as our target brand for the AI customer-support agent, this taxonomy establishes a structured categorization of customer problems observed across Twitter support interactions.

---

## 2. Why These 10 Intents Were Selected
The 10 intents were defined based on recurring operational patterns identified in our initial multi-turn conversation and keyword analysis:
1. **Technical Troubleshooting**: Windows OS update errors, blue screens, driver crashes, and software freezes constitute the largest bulk of inbound queries.
2. **Product / Feature How-To**: Functional guidance on Microsoft applications (Office, OneDrive, Windows settings, backward compatibility).
3. **Account & Login**: Account lockouts, MFA/recovery form hurdles, and hacked/deleted account recovery.
4. **Billing & Payments**: Unrecognized charges, payment method errors, and store transaction issues.
5. **Order / Delivery**: Physical shipments (Surface devices, accessories), delivery status, and tracking inquiries.
6. **Warranty / Repair**: Hardware defect diagnostics, warranty entitlements, and service tag/device replacements.
7. **Network / Connectivity**: Cloud sync disruptions, server errors, and peripheral connection issues (Bluetooth, Surface pen).
8. **Microsoft Store**: App download stalls, store updates, and app launching failures on Windows 10.
9. **Complaint / Feedback**: Support agent escalation complaints, service dissatisfaction, and product policy grievances.
10. **Cancellation / Subscription**: Pre-order cancellations, recurring Office 365 / Xbox subscriptions, and refund requests.

---

## 3. Known Limitation: The "Other / Uncategorized" Bucket
In our preliminary keyword-based screening (`analysis/intent_analysis.csv`), **51.4% (1,029 / 2,000)** of sampled customer messages fell into **"Other / Uncategorized"**.

### Reasons for this limitation:
- **Keyword Lexicon Boundaries**: The initial rule-based regex lexicon used strict keyword matching, which missed colloquial phrasing, multi-sentence contextual problems, or conversational fragments (e.g., *"Did clean install and it works now"*, *"I can't wait longer tonight"*).
- **Multi-Turn Context Dependency**: Customers frequently split problem descriptions across multiple consecutive tweets or reply with terse acknowledgments.
- **Noise & Dispersed Topics**: Many tweets contain short status checks or ambiguous expressions without explicit technical terminology.

---

## 4. Status: First Draft Taxonomy & Next Steps
- **Current Status**: This is an initial **FIRST DRAFT** taxonomy derived from empirical exploratory data analysis without LLM or machine learning intervention.
- **Refinement Strategy**: We will refine and validate this taxonomy using a **manually labelled Golden Evaluation Set**. By curating representative real conversation threads and reviewing boundary cases, we will eliminate ambiguity, improve coverage, and prepare robust ground-truth benchmarks for training and evaluating the AI customer-support agent.
