# Evaluation

Tooling for evaluating the AI support agent against the Golden Evaluation Set
(`golden_set.json`, 200 records).

| Artifact | Purpose |
| --- | --- |
| `evaluate.py` | Shared metric engine (accuracy, macro/weighted F1, per-intent precision/recall, confusion matrices, reports). |
| `run_agent.py` | Runs the **existing production agent** on the Golden Set and writes its predictions (inference only). |
| `baselines.py` / `run_baselines.py` | Majority and TF-IDF+LogisticRegression baselines, scored with `evaluate.py`. |
| `judge.py` / `run_judge.py` | LLM-as-a-Judge framework (reply quality + evidence grounding), human-review + agreement. |
| `predictions.json` / `predictions.csv` | Agent predictions produced by `run_agent.py`. |
| `predictions_failures.json` | Examples where the agent failed (recorded, never fabricated). |
| `agent_cache/` | Per-example cache that makes reruns resume without repeating Gemini calls. |
| `results/` | Agent evaluation artifacts. |
| `baseline_results/` | Baseline artifacts. |
| `judge_results/` | Judge artifacts (cached judgments, results.json, agreement.json). |
| `judge_human_review.csv` | Human-review sheet for grounding labels (never auto-filled). |
| `golden_set.labels.json` | Human labels store (if using the labeling workflow). |

---

## Agent inference (run the agent on the Golden Set)

This is the step where our **real** support agent answers each Golden Set
example. It is separate from evaluation on purpose:

```
golden_set.json
   └─▶ existing intent classifier   └─▶ existing reply generator  └─▶ AgentResult
        existing evidence retriever     existing escalation policy
   └─▶ predictions.json   (fed to evaluate.py and run_judge.py later)
```

### What this tool does and does NOT do

- **Does**: run the existing production pipeline on `customer_message` and save
  each record's prediction (`predicted_intent`, `predicted_intent_confidence`,
  `predicted_escalation`, `predicted_escalation_reason`, `predicted_reply`,
  `retrieved_evidence`).
- **Does NOT**: compute metrics, generate/modify human labels, or fabricate any
  result. The human `intent_label` / `escalation_label` / `notes` are ground
  truth and are **never** passed to the agent.
- If one example fails, the failure is recorded against its `golden_id` in
  `predictions_failures.json` and the other examples continue; failed examples
  are retried on the next run.

### 1. Configure Gemini (one time)

The API key is read from the environment — it is never stored in code:

```powershell
# PowerShell
$env:GEMINI_API_KEY = "your-real-key-here"     # required
$env:GEMINI_MODEL   = "gemini-2.0-flash"       # optional; default gemini-1.5-flash
$env:AGENT_EMBEDDING_INDEX = "..."             # optional; path to a FAISS embedding index
```

If `GEMINI_API_KEY` is missing, the runner refuses to start with a clear
message. It will never invent predictions or results without a real model call.

> **Optional: evidence retrieval.** The existing retriever searches a FAISS
> embedding index of historical MicrosoftHelps conversations. If you have one,
> point `--embedding-index` (or `AGENT_EMBEDDING_INDEX`) at it. Without an
> index the stock retriever returns no evidence, so the existing escalation
> policy will escalate each example (safe, but not very interesting).

> **Build the index (one time, local only, no AI calls):**
>
> ```powershell
> python -m evaluation.build_embedding_index --exclude-golden-set evaluation/golden_set.json
> ```
>
> Reconstructs MicrosoftHelps conversations from `dataset/twcs.csv`, embeds each
> conversation's customer messages with the same sentence-transformer model used
> at query time (`all-MiniLM-L6-v2`), and writes `evaluation/twcs_evidence_index.json`.
> The `--exclude-golden-set` hold-out removes the 200 Golden Set conversations so
> the agent can never retrieve an example's own resolution. `--max-records`,
> `--seed`, and `--head` are supported for smaller/deterministic builds. Point the
> runner at it with `--embedding-index evaluation/twcs_evidence_index.json`.

### 2. Test one example

```powershell
python -m evaluation.run_agent --golden-id GOLDEN-0001
```

Only `GOLDEN-0001` is processed; everything else is left untouched. If a valid
prediction for it already exists, it is reused — Gemini is not called again.

### 3. Run a small batch

```powershell
python -m evaluation.run_agent --limit 10
```

Processes at most **10 examples that do not already have a prediction** (in
Golden Set order). Great for checking cost before the full run.

### 4. Run all 200

```powershell
python -m evaluation.run_agent --all    # explicit: re-run every example, ignoring cache
```

If you have already run some examples, the normal command is just:

```powershell
python -m evaluation.run_agent          # resume: process everything still unprocessed
```

### 5. Dry run (no cost, no Gemini)

```powershell
python -m evaluation.run_agent --dry-run
python -m evaluation.run_agent --dry-run --limit 10
python -m evaluation.run_agent --dry-run --golden-id GOLDEN-0001
```

Prints exactly which examples would call Gemini (`NEW CALL`) versus which are
already cached (`CACHED`), and writes **nothing**. Dry-run artifacts are never
created, so there is no confusion with real predictions.

### How caching / resume works

- Every successful prediction is cached at `evaluation/agent_cache/<golden_id>.json`
  with a **fingerprint** (SHA-256 of the customer message + conversation context
  + model).
- On a rerun, a cached prediction is reused **only if its fingerprint still
  matches** the current Golden Set record — so if an example's input changes,
  it is re-run instead of serving a stale answer.
- `python -m evaluation.run_agent` therefore only spends Gemini calls on the
  examples that still need one, no matter how many times you stop and restart.

### Where predictions are stored

| File | Contents |
| --- | --- |
| `evaluation/predictions.json` | Canonical, evaluator-ready JSON list of valid predictions. |
| `evaluation/predictions.csv` | Flattened copy for quick spreadsheet review. |
| `evaluation/predictions_failures.json` | `golden_id` + error for every failed example. |
| `evaluation/agent_cache/` | Per-example cache that powers resume (not the metrics). |

Each prediction looks like:

```json
{
  "golden_id": "GOLDEN-0001",
  "predicted_intent": "Order / Delivery",
  "predicted_intent_confidence": 0.93,
  "predicted_escalation": "AUTO_HANDLE",
  "predicted_escalation_reason": null,
  "predicted_reply": "We can update the shipping address ...",
  "retrieved_evidence": [
    {
      "evidence_id": "12345",
      "source_tweet_id": 9687,
      "text": "You will need to cancel your order ...",
      "score": 0.82,
      "source": "twcs_historical",
      "metadata": {}
    }
  ]
}
```

### How predictions are passed to the evaluator

```powershell
python -m evaluation.evaluate --predictions evaluation/predictions.json
python -m evaluation.run_judge    --predictions evaluation/predictions.json --limit 10
```

`evaluate.py` matches each prediction to its human labels by `golden_id` and
computes the metrics there — the inference runner never does. Failed examples
are kept out of the predictions list (they live in `predictions_failures.json`),
so the evaluator only ever sees real predictions.

---

## LLM-as-a-Judge (reply quality + evidence grounding)

### Why LLM-as-a-judge?

Automatic metrics tell us *what* the agent got right but not *how good the
reply actually is*. Fine-grained human evaluation of every generated reply does
not scale. An LLM used **only as an evaluator** can apply a consistent rubric
to a generated reply and to the evidence the reply was supposed to rely on.
The judge is a scoring aid, **not ground truth** — it is always cross-checked
against human review on a subset.

### What the judge evaluates

For each example the judge receives **only**:

- `customer_message`
- `conversation_context`
- `predicted_intent`
- `retrieved_evidence`
- `generated_reply`

Human Golden Set labels (`intent_label`, `escalation_label`) are **never**
passed to Gemini — they cannot leak into the judgment. Even the `golden_id` is
kept out of the prompt.

Two dimensions are scored, in one strict JSON object per example:

**1. Reply quality** — 1–5 per dimension:

| Field | Question |
| --- | --- |
| `correctness` | Does the reply correctly address the customer's problem? |
| `helpfulness` | Does it provide useful, actionable support? |
| `relevance` | Does it stay focused on the customer's issue? |
| `clarity` | Is it understandable and professionally written? |
| `escalation_appropriateness` | Does it avoid unsafe/unjustified claims when human handling is needed? |
| `overall_reply_score` | Overall reply quality (1–5). |

**2. Evidence grounding** — 1–5 per dimension:

| Field | Question |
| --- | --- |
| `evidence_support` | Are the important claims in the reply supported by the retrieved evidence? |
| `unsupported_claims` | Does the reply introduce facts/policies/procedures/guarantees NOT supported by the evidence (5 = none)? |
| `evidence_relevance` | Is the retrieved evidence relevant to the customer's issue? |
| `grounding_score` | Overall degree to which the reply is grounded in the evidence (1–5). |

Scores are strict integers 1–5; each response must be valid JSON with all
required fields plus a short `reason`. A `grounding_score >= 3` maps to the
binary `SUPPORTED` decision (otherwise `UNSUPPORTED`).

### How to run the judge

```bash
export GEMINI_API_KEY=...          # never stored in source code
export GEMINI_MODEL=gemini-2.0-flash   # optional, configurable model

python -m evaluation.run_judge --predictions evaluation/predictions.json --limit 10
python -m evaluation.run_judge --predictions evaluation/predictions.json \
    --golden-ids GOLDEN-0001 GOLDEN-0002
python -m evaluation.run_judge --predictions evaluation/predictions.json --all
python -m evaluation.run_judge --predictions evaluation/predictions.json --dry-run
```

`--predictions` is the JSON produced by the agent (one object per `golden_id`
with `predicted_intent`, `predicted_reply`, `retrieved_evidence`). Outputs:

- `judge_results/results.json` — validated judgments (scores, reasons, input snapshot)
- `judge_results/cache/<golden_id>.json` — per-example cache
- `judge_human_review.csv` — human-review sheet (see below)

Temperature is fixed at 0 and structured (JSON) output is requested for
reproducibility. The model defaults to `GEMINI_MODEL` (or a built-in default)
and is always overridable.

### Cost / rate-limit protection

The judge never silently scores the whole set:

- Defaults to `--limit 10` on first run.
- `--golden-ids ...` selects specific records.
- `--all`/`--limit 0` are explicit opt-ins.
- `--dry-run` prints exactly which examples would call Gemini.

### Caching

Every judged example is cached under `judge_results/cache/` keyed by
`golden_id` plus an input fingerprint (SHA-256 of the judge input). Reruns
serve cached examples without calling Gemini. If judge input for a record
changes (e.g. a new reply), the fingerprint no longer matches and that example
is re-judged.

### Human review & judge–human agreement

After judging, `judge_human_review.csv` is written with **empty**
`human_grounding` / `human_notes` columns. For a subset of examples you:

1. Read the row (customer message, evidence, generated reply, LLM scores).
2. Independently assign a binary **human grounding judgment**:
   `SUPPORTED` or `UNSUPPORTED`.
3. Optionally add notes.

The human label is **never generated or suggested** by code. Agreement is then
computed between your independent annotation and the LLM grounding decision:

```bash
python -m evaluation.run_judge --predictions evaluation/predictions.json \
    --human-grounding evaluation/judge_human_review.csv
```

This writes `judge_results/agreement.json` with:

- raw agreement percentage
- Cohen's kappa (reported only when both annotators use both labels — it is
  undefined when a rater shows no variability)

Use raw agreement plus kappa together; kappa corrects for chance agreement and
assumes independent annotations.

### Limitations of LLM-as-a-judge

- The judge is not ground truth: it is a consistent, low-cost proxy.
- LLM judges can have blind spots (fluent but wrong replies), drift across
  versions, and are sensitive to the rubric wording.
- Scores are coarse (1–5) and should be compared, not treated as absolutes.
- Always validate with human review on a sample before trusting the numbers.
- For research purposes, judge scores and the human grounding labels are
  annotations — not authoritative outcomes.