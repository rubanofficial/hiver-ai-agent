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

## Running the interactive labeler (recommended)

You do not have to edit JSON by hand. A small interactive labeling tool is
provided. It is fully local: it never calls Gemini or any other AI model and
never suggests labels — every value below comes from you.

Start (or resume) labeling:

```bash
python -m src.label_golden_set
```

It opens at the first unlabeled record (`evaluation/golden_set.json` is read
but never modified). For each record it prints the golden id, the customer
message, the conversation context, and Microsoft's historical responses, then
asks you to:

- pick **exactly one** intent (number `1`..`10`), and
- pick **exactly one** escalation label (number `1` or `2`), and
- optionally type a note (or press Enter to skip).

At any intent/escalation prompt you can also type:

- `q` — save progress and quit
- `b` — go back to the previous record
- `s` — skip this record for now

Progress is saved to `evaluation/golden_set.labels.json` after every record,
so closing the program never loses work. Useful extras:

```bash
python -m src.label_golden_set --list                 # show progress only
python -m src.label_golden_set --record GOLDEN-0010   # jump to a record
python -m src.label_golden_set --export evaluation    # write labeled copy
```

`--export` writes a *copy* (`evaluation/golden_set.labeled.json`) with the
labels merged in. `golden_set.json` is never overwritten.

---

## Web interface (dropdown-based, recommended)

A browser-based labeling interface is also available.  It uses only the
Python standard library (no new dependencies) and provides native HTML
dropdowns for intent and escalation selection.

Start the local server:

```bash
python -m src.labeling_server
```

This opens a browser at `http://127.0.0.1:8000`.  The left sidebar lists all
200 records (labeled / unlabeled).  The right panel shows the customer
message, conversation context, and Microsoft responses, with two dropdowns:

- **Intent** — pick exactly one of the 10 taxonomy values.
- **Escalation** — pick `AUTO_HANDLE` or `ESCALATE_TO_HUMAN`.

There is also an optional notes textarea.  Click **Save** after each record;
progress is written to `evaluation/golden_set.labels.json` immediately, so
closing the browser or stopping the server never loses work.

On restart, the interface opens at the first unlabeled record automatically
(resume).

Other options:

```bash
python -m src.labeling_server --port 9000     # custom port
python -m src.labeling_server --no-open       # print URL, no browser
```

The **Export labeled copy** button writes a complete
`evaluation/golden_set.labeled.json` with human labels merged into every
record.  `golden_set.json` is never modified.