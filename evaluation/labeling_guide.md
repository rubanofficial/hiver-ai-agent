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

## Escalation label

- `AUTO_HANDLE`: the conversation shows a clear, relevant Microsoft
  resolution. A reply can be safely grounded in the historical
  evidence without guessing.
- `ESCALATE_TO_HUMAN`: use this when the evidence is missing, weak, or
  unrelated, the Microsoft reply would require unsupported facts, or
  the issue involves a sensitive/high-risk intent such as:
  - Complaint / Feedback
  - Cancellation / Subscription
  - Billing & Payments
  - Account & Login (security-sensitive cases)

Prefer `ESCALATE_TO_HUMAN` whenever you would not confidently send the
reply to a real customer yourself.

---

_The 10-intent taxonomy is defined in `config/intents.yaml` and is
shared with the runtime agent. Do not invent new intents._