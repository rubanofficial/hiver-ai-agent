"""
Tests for the Golden Evaluation Set infrastructure (src.golden_set).

All tests use small synthetic conversations / evidence records.  No real
data files (twcs.csv) are read and no AI calls are made.  The only external
file touched is config/intents.yaml, which is optional and only READ.
"""

import json

import pandas as pd
import pytest

from src.evidence import ConversationEvidence, ConversationEvidenceBuilder, TweetRecord
from src.golden_set import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_TARGET_SIZE,
    MIN_CURRENT_TURN_LEN,
    build_golden_set,
    build_labeling_guide_text,
    format_golden_record,
    load_intent_taxonomy,
    sample_evidence,
    save_golden_set,
    save_labeling_guide,
)


# ---------------------------------------------------------------------------
# Synthetic conversation helpers
# ---------------------------------------------------------------------------

_LONG_MSG = "My Surface laptop screen keeps flickering after the update."


def _tweet(tweet_id, author_id, inbound, text, in_response_to_tweet_id=""):
    return {
        "tweet_id": tweet_id,
        "author_id": author_id,
        "inbound": inbound,
        "text": text,
        "created_at": "Tue Jan 01 12:00:00 +0000 2018",
        "in_response_to_tweet_id": in_response_to_tweet_id,
    }


def _conv(
    conv_id,
    tweets,
    customer_messages=None,
    brand_responses=None,
    num_turns=None,
    has_resolution_signal=False,
):
    """Build a ConversationEvidence object from tweet dicts."""
    return ConversationEvidence(
        conv_id=conv_id,
        tweets=[TweetRecord(
            tweet_id=t["tweet_id"],
            author_id=t["author_id"],
            inbound=t["inbound"],
            text=t["text"],
            created_at=t["created_at"],
        ) for t in tweets],
        customer_messages=customer_messages
        if customer_messages is not None
        else [t["text"] for t in tweets if t["inbound"]],
        brand_responses=brand_responses
        if brand_responses is not None
        else [t["text"] for t in tweets if not t["inbound"] and t["author_id"] == "MicrosoftHelps"],
        num_turns=num_turns or len(tweets),
        has_resolution_signal=has_resolution_signal,
        metadata={"brand": "MicrosoftHelps"},
    )


def _simple_turn_conv(conv_id):
    """C -> B conversation with one substantial customer message."""
    return _conv(
        conv_id,
        tweets=[
            _tweet(conv_id * 10 + 1, "customer", True, _LONG_MSG),
            _tweet(
                conv_id * 10 + 2, "MicrosoftHelps", False,
                "Please try updating your display drivers.",
                in_response_to_tweet_id=str(conv_id * 10 + 1),
            ),
        ],
    )


def _cbcb_conv(conv_id):
    """C -> B -> C -> B conversation (rich context, with resolution signal)."""
    return _conv(
        conv_id,
        tweets=[
            _tweet(conv_id * 10 + 1, "customer", True, _LONG_MSG),
            _tweet(
                conv_id * 10 + 2, "MicrosoftHelps", False,
                "Please update your display drivers.",
                in_response_to_tweet_id=str(conv_id * 10 + 1),
            ),
            _tweet(
                conv_id * 10 + 3, "customer", True,
                "That worked, the screen is fine now. Thanks!",
                in_response_to_tweet_id=str(conv_id * 10 + 2),
            ),
            _tweet(
                conv_id * 10 + 4, "MicrosoftHelps", False,
                "Great to hear, glad we could help!",
                in_response_to_tweet_id=str(conv_id * 10 + 3),
            ),
        ],
        has_resolution_signal=True,
    )


def _no_brand_conv(conv_id):
    """Customer message but zero MicrosoftHelps replies - not sampleable."""
    return _conv(
        conv_id,
        tweets=[
            _tweet(conv_id * 10 + 1, "customer", True, _LONG_MSG),
        ],
        brand_responses=[],
    )


def _short_msg_conv(conv_id):
    """Customer final message too short to understand the issue."""
    return _conv(
        conv_id,
        tweets=[
            _tweet(conv_id * 10 + 1, "customer", True, "hi"),
            _tweet(conv_id * 10 + 2, "MicrosoftHelps", False, "Hello! How can we help?",
                   in_response_to_tweet_id=str(conv_id * 10 + 1)),
        ],
        customer_messages=["hi"],
    )


@pytest.fixture
def mixed_pool():
    """A varied pool of conversations covering eligible and ineligible cases."""
    return [
        _cbcb_conv(1),
        _simple_turn_conv(2),
        _cbcb_conv(3),
        _no_brand_conv(4),
        _short_msg_conv(5),
        _simple_turn_conv(6),
        _cbcb_conv(7),
        _simple_turn_conv(8),
    ]


# ---------------------------------------------------------------------------
# Sampling behaviour
# ---------------------------------------------------------------------------

class TestSampling:
    def test_samples_target_count(self, mixed_pool):
        sampled = sample_evidence(mixed_pool, target_size=4, seed=1)
        assert len(sampled) == 4

    def test_ineligible_conversations_are_excluded(self, mixed_pool):
        sampled = sample_evidence(mixed_pool, target_size=100, seed=1)
        conv_ids = {int(s["conv_id"]) for s in sampled}
        # conv 4 has no brand response; conv 5 has a too-short final message.
        assert 4 not in conv_ids
        assert 5 not in conv_ids

    def test_deterministic_given_seed(self, mixed_pool):
        a = sample_evidence(mixed_pool, target_size=4, seed=42)
        b = sample_evidence(mixed_pool, target_size=4, seed=42)
        assert [r["conv_id"] for r in a] == [r["conv_id"] for r in b]

    def test_target_smaller_than_rich_tier(self):
        pool = [_cbcb_conv(i) for i in range(1, 11)]
        sampled = sample_evidence(pool, target_size=5, seed=3, rich_ratio=1.0)
        assert len(sampled) == 5
        for r in sampled:
            # rich_ratio=1.0 + all rich pool -> every picked conv is rich
            assert int(r["num_turns"]) >= 3

    def test_target_larger_than_pool_returns_all(self, mixed_pool):
        sampled = sample_evidence(mixed_pool, target_size=200, seed=1)
        # 6 of the 8 convs are eligible (excludes 4: no brand reply,
        # and 5: too-short final customer message).
        assert len(sampled) == 6
        conv_ids = {int(s["conv_id"]) for s in sampled}
        assert conv_ids == {1, 2, 3, 6, 7, 8}

    def test_empty_pool_returns_empty(self):
        assert sample_evidence([], target_size=200, seed=1) == []

    def test_accepts_plain_dict_records(self):
        rec = _simple_turn_conv(1)
        d = {
            "conv_id": rec.conv_id,
            "tweets": rec.tweets,
            "customer_messages": rec.customer_messages,
            "brand_responses": rec.brand_responses,
            "num_turns": rec.num_turns,
            "has_resolution_signal": rec.has_resolution_signal,
            "metadata": rec.metadata,
        }
        sampled = sample_evidence([d], target_size=1, seed=1)
        assert len(sampled) == 1
        assert sampled[0]["conv_id"] == 1

    def test_result_sorted_by_conversation_id(self, mixed_pool):
        sampled = sample_evidence(mixed_pool, target_size=6, seed=1)
        ids = [int(r["conv_id"]) for r in sampled]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Record formatting
# ---------------------------------------------------------------------------

class TestRecordFormatting:
    def test_current_turn_is_last_customer_message(self):
        rec = _cbcb_conv(1)
        record = format_golden_record(sample_evidence([rec])[0], golden_index=1)
        assert record["customer_message"] == (
            "That worked, the screen is fine now. Thanks!"
        )

    def test_conversation_context_preserves_previous_turns(self):
        rec = _cbcb_conv(1)
        record = format_golden_record(sample_evidence([rec])[0], golden_index=1)
        context = record["conversation_context"]
        assert _LONG_MSG in context
        assert "Please update your display drivers." in context
        # The current customer turn must NOT appear in the context.
        assert "That worked, the screen is fine now" not in context

    def test_context_labels_roles(self):
        rec = _cbcb_conv(1)
        record = format_golden_record(sample_evidence([rec])[0], golden_index=1)
        assert "CUSTOMER (" in record["conversation_context"]
        assert "MICROSOFT (" in record["conversation_context"]

    def test_microsoft_responses_captured(self):
        rec = _cbcb_conv(1)
        record = format_golden_record(sample_evidence([rec])[0], golden_index=1)
        assert any("Please update your display drivers." in r for r in record["microsoft_responses"])
        assert any("glad we could help!" in r for r in record["microsoft_responses"])

    def test_source_tweets_preserved_for_traceability(self):
        rec = _cbcb_conv(1)
        record = format_golden_record(sample_evidence([rec])[0], golden_index=1)
        ids = {t["tweet_id"] for t in record["source_tweets"]}
        extra = {t.tweet_id for t in rec.tweets}
        assert ids == extra
        assert len(record["source_tweets"]) == 4

    def test_conversation_id_preserved(self):
        rec = _cbcb_conv(7)
        record = format_golden_record(sample_evidence([rec])[0], golden_index=1)
        assert record["conversation_id"] == 7

    def test_golden_id_sequential_and_unique(self):
        pool = [_cbcb_conv(i) for i in range(1, 11)] + [_simple_turn_conv(i) for i in range(11, 21)]
        records = build_golden_set(pool, target_size=10, seed=5)
        ids = [r["golden_id"] for r in records]
        assert len(ids) == len(set(ids))
        assert ids[0] == "GOLDEN-0001"
        assert ids[-1] == "GOLDEN-0010"

    def test_label_fields_are_empty_for_human(self):
        rec = _cbcb_conv(1)
        record = format_golden_record(sample_evidence([rec])[0], golden_index=1)
        assert record["intent_label"] == ""
        assert record["escalation_label"] == ""
        assert record["notes"] == ""


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def _records(self, n=5):
        pool = [_cbcb_conv(i) for i in range(1, 6)] + [_simple_turn_conv(i) for i in range(6, 11)]
        return build_golden_set(pool, target_size=n, seed=1)

    def test_save_json_round_trip(self, tmp_path):
        records = self._records()
        out = save_golden_set(records, output_dir=str(tmp_path))
        assert (out / "golden_set.json").exists()

        with open(out / "golden_set.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["count"] == len(records)
        assert payload["label_fields"] == ["intent_label", "escalation_label"]
        assert len(payload["records"]) == len(records)
        first = payload["records"][0]
        assert "customer_message" in first
        assert "conversation_context" in first
        assert first["intent_label"] == ""
        assert first["escalation_label"] == ""

    def test_save_csv_columns(self, tmp_path):
        records = self._records()
        out = save_golden_set(records, output_dir=str(tmp_path))
        csv_path = out / "golden_set.csv"
        assert csv_path.exists()

        import csv as _csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = _csv.reader(f)
            header = next(reader)
            rows = list(reader)
        assert "golden_id" in header
        assert "customer_message" in header
        assert "intent_label" in header
        assert "escalation_label" in header
        assert len(rows) == len(records)

    def test_interleaved_other_brand_in_context(self):
        tweets = [
            _tweet(101, "customer", True, "My device is broken and I need help."),
            _tweet(102, "random_bot", False, "Check this link: http://spam.example.com",
                   in_response_to_tweet_id="101"),
            _tweet(103, "MicrosoftHelps", False, "Please tell us the exact error code.",
                   in_response_to_tweet_id="102"),
            _tweet(104, "customer", True, _LONG_MSG, in_response_to_tweet_id="103"),
        ]
        rec = _conv(
            50,
            tweets,
            customer_messages=["My device is broken and I need help.", _LONG_MSG],
            brand_responses=["Please tell us the exact error code."],
        )
        record = format_golden_record(sample_evidence([rec])[0], golden_index=1)
        context = record["conversation_context"]
        assert "CUSTOMER (101):" in context
        assert "OTHER (102):" in context
        assert "MICROSOFT (103):" in context
        # Context ends right before the current turn (tweet 104).
        assert "MICROSOFT (103):" in record["conversation_context"].splitlines()[-1]
        assert len(context.splitlines()) == 3
        # The current turn text must not be duplicated into the context.
        assert _LONG_MSG not in context
        assert record["customer_message"] == _LONG_MSG


# ---------------------------------------------------------------------------
# Integration with the existing evidence builder (synthetic DataFrame)
# ---------------------------------------------------------------------------

class TestEvidenceBuilderIntegration:
    def test_full_flow_from_synthetic_dataframe(self):
        df = pd.DataFrame([
            _tweet(10, "customer", True, _LONG_MSG),
            _tweet(11, "MicrosoftHelps", False, "Please update display drivers.",
                   in_response_to_tweet_id="10"),
            _tweet(12, "customer", True, "Thanks, fixed now! Screen is stable.",
                   in_response_to_tweet_id="11"),
            _tweet(13, "MicrosoftHelps", False, "Great, have a nice day!",
                   in_response_to_tweet_id="12"),
        ])
        evidence = ConversationEvidenceBuilder(df).build(min_customer_len=5)
        assert len(evidence) == 1
        records = build_golden_set(evidence, target_size=200, seed=7)
        assert len(records) == 1
        rec = records[0]
        assert rec["conversation_id"] == 10
        assert rec["customer_message"] == "Thanks, fixed now! Screen is stable."
        assert "Please update display drivers." in rec["microsoft_responses"]

    def test_dict_records_match_builder_output_schema(self, mixed_pool):
        sampled = sample_evidence([_simple_turn_conv(2)], target_size=1, seed=1)
        rec = sampled[0]
        assert "conv_id" in rec
        assert "tweets" in rec
        assert "brand_responses" in rec


# ---------------------------------------------------------------------------
# Taxonomy & labeling guide
# ---------------------------------------------------------------------------

class TestTaxonomyAndGuide:
    EXPECTED_INTENTS = [
        "Technical Troubleshooting",
        "Product / Feature How-To",
        "Account & Login",
        "Billing & Payments",
        "Order / Delivery",
        "Warranty / Repair",
        "Network / Connectivity",
        "Microsoft Store",
        "Complaint / Feedback",
        "Cancellation / Subscription",
    ]

    def test_loads_10_intents_from_config(self):
        taxonomy = load_intent_taxonomy()
        names = [e["name"] for e in taxonomy]
        assert len(names) == 10
        assert names == self.EXPECTED_INTENTS

    def test_load_intent_taxonomy_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_intent_taxonomy(str(tmp_path / "missing.yaml"))

    def test_guide_covers_all_intents(self):
        taxonomy = load_intent_taxonomy()
        guide = build_labeling_guide_text(taxonomy)
        for entry in taxonomy:
            assert entry["name"] in guide

    def test_guide_explains_escalation_labels(self):
        guide = build_labeling_guide_text([{"name": "Test Intent", "definition": "x"}])
        assert "AUTO_HANDLE" in guide
        assert "ESCALATE_TO_HUMAN" in guide
        assert "must be assigned by a human" in guide

    def test_save_labeling_guide(self, tmp_path):
        out = save_labeling_guide("# Guide", output_dir=str(tmp_path))
        assert (out / "labeling_guide.md").exists()


# ---------------------------------------------------------------------------
# Module defaults sanity
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_defaults_are_200_examples_seeded(self):
        assert DEFAULT_TARGET_SIZE == 200
        assert DEFAULT_SEED == 42

    def test_default_output_dir_is_evaluation(self):
        assert DEFAULT_OUTPUT_DIR.name == "evaluation"

    def test_min_customer_turn_length_sanity(self):
        assert MIN_CURRENT_TURN_LEN >= 1


# This module must never import or reference the Gemini SDK.
class TestNoGeminiDependency:
    def test_golden_set_module_does_not_import_genai(self):
        import src.golden_set as gs
        source = open(gs.__file__, "r", encoding="utf-8").read()
        assert "import google.generativeai" not in source
        assert "genai" not in source
        assert "openai" not in source