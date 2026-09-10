"""
Tests for the historical evidence builder (src.evidence).

All tests use small synthetic DataFrames that mimic the TWCS schema.
No real dataset files are read.
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from src.evidence import (
    ConversationEvidence,
    ConversationEvidenceBuilder,
    TweetRecord,
    build_parent_map,
    find_root,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows):
    """Build a minimal TWCS-like DataFrame from a list of row dicts.

    Each row dict should have keys: tweet_id, author_id, inbound, text.
    ``in_response_to_tweet_id`` is optional; if missing it defaults to "".
    """
    for r in rows:
        r.setdefault("in_response_to_tweet_id", "")
        r.setdefault("created_at", "")
    return pd.DataFrame(rows)


# Synthetic conversation data used across multiple tests

_SIMPLE_CONV = _make_df([
    {"tweet_id": 100, "author_id": "alice", "inbound": True,
     "text": "Hey @MicrosoftHelps my Surface won't turn on."},
    {"tweet_id": 200, "author_id": "MicrosoftHelps", "inbound": False,
     "text": "Hi Alice, please try holding the power button for 15 seconds.",
     "in_response_to_tweet_id": "100"},
])

_CBCB_CONV = _make_df([
    {"tweet_id": 100, "author_id": "bob", "inbound": True,
     "text": "My Office 365 subscription keeps failing."},
    {"tweet_id": 200, "author_id": "MicrosoftHelps", "inbound": False,
     "text": "Bob, could you share your error code?",
     "in_response_to_tweet_id": "100"},
    {"tweet_id": 300, "author_id": "bob", "inbound": True,
     "text": "Error code 0x80070005 appears every time.",
     "in_response_to_tweet_id": "200"},
    {"tweet_id": 400, "author_id": "MicrosoftHelps", "inbound": False,
     "text": "Please try signing out and signing back in.",
     "in_response_to_tweet_id": "300"},
])

_RESOLUTION_CONV = _make_df([
    {"tweet_id": 100, "author_id": "carol", "inbound": True,
     "text": "Windows update keeps crashing."},
    {"tweet_id": 200, "author_id": "MicrosoftHelps", "inbound": False,
     "text": "Please try running the Windows Update Troubleshooter.",
     "in_response_to_tweet_id": "100"},
    {"tweet_id": 300, "author_id": "carol", "inbound": True,
     "text": "Thanks, that worked! It's fixed now.",
     "in_response_to_tweet_id": "200"},
])

_MULTI_BRAND_CONV = _make_df([
    {"tweet_id": 100, "author_id": "dave", "inbound": True,
     "text": "Xbox won't connect to live."},
    {"tweet_id": 200, "author_id": "MicrosoftHelps", "inbound": False,
     "text": "Please restart your console and router.",
     "in_response_to_tweet_id": "100"},
    {"tweet_id": 300, "author_id": "random_bot", "inbound": False,
     "text": "Check this link: http://spam.example.com",
     "in_response_to_tweet_id": "200"},
    {"tweet_id": 400, "author_id": "dave", "inbound": True,
     "text": "Still not working after restart.",
     "in_response_to_tweet_id": "300"},
    {"tweet_id": 500, "author_id": "MicrosoftHelps", "inbound": False,
     "text": "Let us know your gamertag so we can investigate further.",
     "in_response_to_tweet_id": "400"},
])

_NO_BRAND_CONV = _make_df([
    {"tweet_id": 100, "author_id": "eve", "inbound": True,
     "text": "Hello AppleSupport my iPhone is broken."},
    {"tweet_id": 200, "author_id": "AppleSupport", "inbound": False,
     "text": "Eve, please try force restarting the device.",
     "in_response_to_tweet_id": "100"},
])

_TWO_BRANDS_MIXED = pd.concat([_SIMPLE_CONV, _NO_BRAND_CONV], ignore_index=True)


# ---------------------------------------------------------------------------
# build_parent_map
# ---------------------------------------------------------------------------

class TestBuildParentMap:

    def test_simple_chain(self):
        df = _make_df([
            {"tweet_id": 1, "author_id": "a", "inbound": True,
             "text": "help"},
            {"tweet_id": 2, "author_id": "b", "inbound": False,
             "text": "ok", "in_response_to_tweet_id": "1"},
            {"tweet_id": 3, "author_id": "a", "inbound": True,
             "text": "thanks", "in_response_to_tweet_id": "2"},
        ])
        pm = build_parent_map(df)
        assert pm == {2: 1, 3: 2}

    def test_roots_excluded(self):
        """Tweets with no parent should not appear in the map."""
        df = _make_df([
            {"tweet_id": 10, "author_id": "x", "inbound": True,
             "text": "start"},
        ])
        pm = build_parent_map(df)
        assert pm == {}

    def test_branching(self):
        """Multiple children pointing to the same parent."""
        df = _make_df([
            {"tweet_id": 1, "author_id": "a", "inbound": True,
             "text": "q"},
            {"tweet_id": 2, "author_id": "b", "inbound": False,
             "text": "r1", "in_response_to_tweet_id": "1"},
            {"tweet_id": 3, "author_id": "c", "inbound": False,
             "text": "r2", "in_response_to_tweet_id": "1"},
        ])
        pm = build_parent_map(df)
        assert pm[2] == 1
        assert pm[3] == 1


# ---------------------------------------------------------------------------
# find_root
# ---------------------------------------------------------------------------

class TestFindRoot:

    def test_direct_child(self):
        pm = {2: 1}
        assert find_root(2, pm) == 1

    def test_deep_chain(self):
        pm = {2: 1, 3: 2, 4: 3, 5: 4}
        assert find_root(5, pm) == 1

    def test_standalone_tweet(self):
        """A tweet with no parent is its own root."""
        assert find_root(999, {}) == 999

    def test_path_compression(self):
        """After first call, intermediate nodes are cached."""
        pm = {2: 1, 3: 2, 4: 3}
        cache = {}
        assert find_root(4, pm, cache) == 1
        # Path compression should have cached nodes 2, 3, 4
        assert cache[2] == 1
        assert cache[3] == 1
        assert cache[4] == 1

    def test_loop_guard(self):
        """A circular parent map should not cause infinite loop."""
        pm = {1: 2, 2: 3, 3: 1}  # cycle
        result = find_root(1, pm)
        # Should return something (not hang), result may be arbitrary
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# ConversationEvidenceBuilder
# ---------------------------------------------------------------------------

class TestConversationEvidenceBuilder:

    def test_simple_cb_conversation(self):
        builder = ConversationEvidenceBuilder(_SIMPLE_CONV)
        evidence = builder.build()
        assert len(evidence) == 1
        ev = evidence[0]
        assert ev.num_turns == 2
        assert len(ev.customer_messages) == 1
        assert len(ev.brand_responses) == 1
        assert "Surface" in ev.customer_messages[0]
        assert "power button" in ev.brand_responses[0]

    def test_cbcb_conversation(self):
        builder = ConversationEvidenceBuilder(_CBCB_CONV)
        evidence = builder.build()
        assert len(evidence) == 1
        ev = evidence[0]
        assert ev.num_turns == 4
        assert len(ev.customer_messages) == 2
        assert len(ev.brand_responses) == 2
        assert ev.metadata["num_brand_turns"] == 2
        assert ev.metadata["num_customer_turns"] == 2

    def test_resolution_signal_detected(self):
        builder = ConversationEvidenceBuilder(_RESOLUTION_CONV)
        evidence = builder.build()
        assert len(evidence) == 1
        assert evidence[0].has_resolution_signal is True

    def test_no_resolution_in_other_conv(self):
        builder = ConversationEvidenceBuilder(_CBCB_CONV)
        evidence = builder.build()
        assert len(evidence) == 1
        assert evidence[0].has_resolution_signal is False

    def test_no_brand_response_excluded(self):
        """Conversations with zero MicrosoftHelps tweets are dropped."""
        builder = ConversationEvidenceBuilder(_NO_BRAND_CONV)
        evidence = builder.build()
        assert len(evidence) == 0

    def test_multi_brand_conv_only_microsoft(self):
        """In a mixed-brand conversation, only MicrosoftHelps tweets are
        counted as brand responses."""
        builder = ConversationEvidenceBuilder(_MULTI_BRAND_CONV)
        evidence = builder.build()
        assert len(evidence) == 1
        ev = evidence[0]
        # 2 MicrosoftHelps responses, not the random_bot one
        assert ev.metadata["num_brand_turns"] == 2
        assert len(ev.brand_responses) == 2
        # random_bot text should NOT appear in brand_responses
        for resp in ev.brand_responses:
            assert "spam.example.com" not in resp

    def test_two_independent_conversations(self):
        """Two separate brands produce separate conversation groups."""
        builder = ConversationEvidenceBuilder(_TWO_BRANDS_MIXED)
        evidence = builder.build()
        # Only MicrosoftHelps conversations survive
        assert len(evidence) == 1
        assert evidence[0].metadata["brand"] == "MicrosoftHelps"

    def test_min_customer_len_filter(self):
        short_conv = _make_df([
            {"tweet_id": 100, "author_id": "frank", "inbound": True,
             "text": "hi"},
            {"tweet_id": 200, "author_id": "MicrosoftHelps", "inbound": False,
             "text": "Hello! How can we help?",
             "in_response_to_tweet_id": "100"},
        ])
        builder = ConversationEvidenceBuilder(short_conv)
        evidence = builder.build(min_customer_len=5)
        assert len(evidence) == 1
        # "hi" is only 2 chars, should be filtered out
        assert evidence[0].customer_messages == []

    def test_preserves_tweet_ids(self):
        builder = ConversationEvidenceBuilder(_SIMPLE_CONV)
        evidence = builder.build()
        tweet_ids = [t.tweet_id for t in evidence[0].tweets]
        assert 100 in tweet_ids
        assert 200 in tweet_ids

    def test_brand_is_configurable(self):
        """Builder should work with any brand name."""
        conv = _make_df([
            {"tweet_id": 100, "author_id": "user1", "inbound": True,
             "text": "Help me with my Dell laptop."},
            {"tweet_id": 200, "author_id": "DellCares", "inbound": False,
             "text": "Please try restarting your Dell laptop.",
             "in_response_to_tweet_id": "100"},
        ])
        builder = ConversationEvidenceBuilder(conv, brand="DellCares")
        evidence = builder.build()
        assert len(evidence) == 1
        assert evidence[0].metadata["brand"] == "DellCares"


# ---------------------------------------------------------------------------
# JSON save / load round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:

    def test_save_and_load(self, tmp_path):
        builder = ConversationEvidenceBuilder(_CBCB_CONV)
        evidence = builder.build()
        out_file = str(tmp_path / "evidence.json")
        ConversationEvidenceBuilder.save_json(evidence, out_file)

        loaded = ConversationEvidenceBuilder.load_json(out_file)
        assert len(loaded) == 1
        rec = loaded[0]
        assert rec["conv_id"] == evidence[0].conv_id
        assert len(rec["tweets"]) == 4
        assert len(rec["customer_messages"]) == 2
        assert len(rec["brand_responses"]) == 2
        assert rec["num_turns"] == 4

    def test_loaded_tweets_have_required_fields(self, tmp_path):
        builder = ConversationEvidenceBuilder(_SIMPLE_CONV)
        evidence = builder.build()
        out_file = str(tmp_path / "ev.json")
        ConversationEvidenceBuilder.save_json(evidence, out_file)

        loaded = ConversationEvidenceBuilder.load_json(out_file)
        tweet = loaded[0]["tweets"][0]
        for key in ("tweet_id", "author_id", "inbound", "text", "created_at"):
            assert key in tweet

    def test_empty_evidence_saves_empty_list(self, tmp_path):
        out_file = str(tmp_path / "empty.json")
        ConversationEvidenceBuilder.save_json([], out_file)
        loaded = ConversationEvidenceBuilder.load_json(out_file)
        assert loaded == []


# ---------------------------------------------------------------------------
# ConversationEvidence data class
# ---------------------------------------------------------------------------

class TestConversationEvidenceDataclass:

    def test_defaults(self):
        ev = ConversationEvidence(conv_id=12345)
        assert ev.conv_id == 12345
        assert ev.tweets == []
        assert ev.customer_messages == []
        assert ev.brand_responses == []
        assert ev.num_turns == 0
        assert ev.has_resolution_signal is False
        assert ev.metadata == {}

    def test_tweet_record_defaults(self):
        tr = TweetRecord(tweet_id=1, author_id="test", inbound=True, text="hello")
        assert tr.created_at == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_dataframe(self):
        df = _make_df([])
        builder = ConversationEvidenceBuilder(df)
        evidence = builder.build()
        assert evidence == []

    def test_only_inbound_no_brand(self):
        df = _make_df([
            {"tweet_id": 1, "author_id": "user1", "inbound": True,
             "text": "Help me please."},
            {"tweet_id": 2, "author_id": "user2", "inbound": True,
             "text": "Same problem here."},
        ])
        builder = ConversationEvidenceBuilder(df)
        evidence = builder.build()
        assert evidence == []

    def test_brand_only_no_customer(self):
        """Brand tweet with no customer tweet in the conversation."""
        df = _make_df([
            {"tweet_id": 100, "author_id": "MicrosoftHelps", "inbound": False,
             "text": "We are experiencing a service outage."},
        ])
        builder = ConversationEvidenceBuilder(df)
        evidence = builder.build()
        assert len(evidence) == 1
        assert evidence[0].customer_messages == []
        assert len(evidence[0].brand_responses) == 1

    def test_nan_text_handled(self):
        df = pd.DataFrame([
            {"tweet_id": 100, "author_id": "user1", "inbound": True,
             "text": "Help me", "in_response_to_tweet_id": "", "created_at": ""},
            {"tweet_id": 200, "author_id": "MicrosoftHelps", "inbound": False,
             "text": float("nan"), "in_response_to_tweet_id": "100",
             "created_at": ""},
        ])
        builder = ConversationEvidenceBuilder(df)
        evidence = builder.build()
        assert len(evidence) == 1
        # NaN text should be converted to empty string
        assert evidence[0].brand_responses == [""]

    def test_filter_idempotent(self):
        """Calling filter_brand_conversations twice gives same result."""
        builder = ConversationEvidenceBuilder(_CBCB_CONV)
        first = builder.filter_brand_conversations()
        second = builder.filter_brand_conversations()
        assert len(first) == len(second)
        assert set(first["tweet_id"]) == set(second["tweet_id"])
