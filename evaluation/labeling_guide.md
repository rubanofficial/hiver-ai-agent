# MicrosoftHelps Golden Evaluation Set - Labeling Guide

This guide tells you how to fill in the empty label fields of each
record in `golden_set.json` (or `golden_set.csv`). Labels are
ground truth: they must be assigned by a human, never by the AI.

---

## Fields you label

For every record, fill in these three fields:

1. `intent_label`: exactly ONE intent from the 10-intent taxonomy.
2. `escalation_label`: exactly ONE of `AUTO_HANDLE` or `ESCALATE_TO_HUMAN`.
3. `notes` (optional): one line explaining any judgement call.

---

## Intent taxonomy (choose exactly one)

Pick the intent that best matches the customer's *current message*,
using the conversation context and Microsoft's responses to disambiguate.
If two intents both fit, pick the closest and note the ambiguity in `notes`.

1. **Technical Troubleshooting** - Diagnosing and resolving system errors, app crashes, update failures, or hardware malfunctions
2. **Product / Feature How-To** - Inquiries about using features, software functionality, settings, file compatibility, or configurations
3. **Account & Login** - Account recovery, password resets, multi-factor authentication, security info, or hacked account assistance
4. **Billing & Payments** - Questions or disputes concerning credit charges, payment methods, duplicate transactions, or invoice receipts
5. **Order / Delivery** - Inquiries about tracking physical hardware orders, shipping delays, missing shipments, or delivery status
6. **Warranty / Repair** - Hardware repair requests, warranty status verification, service tag checks, or physical device replacements
7. **Network / Connectivity** - Issues connecting to the internet, syncing cloud data, Bluetooth peripherals, or server outages
8. **Microsoft Store** - Problems downloading, installing, or updating apps and games from the Windows/Microsoft Store platform
9. **Complaint / Feedback** - Expressions of dissatisfaction with support quality, product changes, service policies, or staff responses
10. **Cancellation / Subscription** - Requests to cancel pre-orders, recurring subscriptions (Office 365, Xbox Live), or terminate services

---

## Escalation label (independent judgment)

This is YOUR decision as a reviewer, not a copy of any automated
rule. Base it on the evidence in this record alone. Ask yourself
two questions:

- `AUTO_HANDLE`: would I confidently allow the AI to send a
  grounded, accurate response to this customer right now?
- `ESCALATE_TO_HUMAN`: would I want a human involved because the
  evidence, the situation, or the risk makes autonomous handling
  inappropriate?

Choose `ESCALATE_TO_HUMAN` when any of the following holds (none of
them is a strict rule; use your judgment):

- the conversation context is missing, weak, or unrelated to the
  customer's issue, so an accurate reply cannot be grounded in it;
- answering correctly would require facts that are not present in
  the record;
- getting the answer wrong could cause real harm (financial, legal,
  privacy, safety, or security impact); or
- you personally would not feel comfortable sending the reply
  to a real customer.

The runtime agent runs its own automated escalation policy. Do NOT
use it (or any intent list) as your source of truth here: judge each
record independently. Your escalation label is the ground truth
that the agent's policy will later be measured against.

---

_The 10-intent taxonomy is defined in `config/intents.yaml` and is
shared with the runtime agent. Do not invent new intents._

---

---

## AI-Assisted Labeling & Provenance Tracking

To speed up annotation while guaranteeing 100% human-approved ground truth, you can pre-generate advisory AI suggestions using Gemini:

```bash
# Generate and cache advisory suggestions for all 200 records:
python -m src.suggest_labels
```

### Key Principles:
1. **Advisory Only**: AI suggestions are advisory hints displayed in the UI and CLI.
2. **Human in the Loop**: Every label must be explicitly approved or modified by a human.
3. **Strict Data Isolation**: Gemini receives ONLY customer message, conversation context, historical Microsoft responses, the 10 taxonomy definitions, and human guidance. It NEVER receives existing human labels.
4. **Caching**: Every AI suggestion is cached in `evaluation/golden_set.suggestions.json` so rerunning never makes redundant API calls.
5. **Provenance Auditing**: Every saved record records `label_source`:
   - `human_accepted_ai` — human reviewed and accepted the AI suggestion.
   - `human_modified_ai` — human reviewed the AI suggestion and modified intent/escalation.
   - `human_direct` — human entered labels directly without an AI suggestion.

---

## Running the interactive labeler (CLI)

Start (or resume) labeling in terminal:

```bash
python -m src.label_golden_set
```

For each record, it prints the customer message, conversation context, historical Microsoft responses, and the advisory AI suggestion (if available).

- Type `a` to **accept the advisory suggestion** (records with `human_accepted_ai`).
- Or enter numbers `1`..`10` for intent and `1`..`2` for escalation to select manually.
- Navigation keys: `b` (back), `s` (skip), `q` (save and quit).

Useful CLI commands:

```bash
python -m src.label_golden_set --list                 # show progress and provenance breakdown
python -m src.label_golden_set --record GOLDEN-0010   # jump directly to a record
python -m src.label_golden_set --export evaluation    # write labeled copy
```

---

## Web interface (dropdown-based, recommended)

Start the local server:

```bash
python -m src.labeling_server
```

This opens a browser at `http://127.0.0.1:8000`.

- **Sidebar**: Lists all 200 records with `todo` and `labeled` badges.
- **AI Suggestion Box**: Displays suggested intent, suggested escalation, confidence percentage, and reason.
- **Accept Suggestion Button**: One click accepts the suggestion and saves with `label_source: "human_accepted_ai"`.
- **Manual Select**: Use the dropdowns to choose or modify any intent/escalation.
- **Save Decision**: Persists progress to `evaluation/golden_set.labels.json`.
- **Export labeled copy**: Exports a merged copy to `evaluation/golden_set.labeled.json`. `golden_set.json` is never modified.