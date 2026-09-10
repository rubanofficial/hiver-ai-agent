"""
Tests for the embedding component (src.embeddings).

All tests mock SentenceTransformer so no model is downloaded and no
internet access is required.
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from src.embeddings import (
    DEFAULT_MODEL,
    EmbeddedRecord,
    EmbeddingIndex,
    EmbeddingBuilder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 4  # tiny dimension for tests


def _fake_encode(texts, show_progress_bar=False, normalize_embeddings=False):
    """Return deterministic pseudo-embeddings based on text length."""
    vectors = []
    for t in texts:
        vec = [float(len(t))] * DIM
        vectors.append(np.array(vec, dtype=np.float32))
    return np.array(vectors, dtype=np.float32)


class _FakeModel:
    """Minimal stand-in for SentenceTransformer."""

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=False):
        return _fake_encode(
            texts,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
        )

    def get_sentence_embedding_dimension(self):
        return DIM


@pytest.fixture(autouse=True)
def _mock_sentence_transformer(monkeypatch):
    """Patch SentenceTransformer so tests never touch the network."""
    import src.embeddings as emb_mod

    def _patched_load(self):
        if self._model is None:
            self._model = _FakeModel()
        return self._model

    monkeypatch.setattr(emb_mod.EmbeddingBuilder, "_load_model", _patched_load)


# ---------------------------------------------------------------------------
# Synthetic evidence records (dicts matching save_json / load_json format)
# ---------------------------------------------------------------------------

_SAMPLE_RECORDS = [
    {
        "conv_id": 100,
        "customer_messages": [
            "My Surface Pro won't turn on after the update.",
            "Still not working even after charging overnight.",
        ],
        "brand_responses": ["Please try holding the power button for 15 seconds."],
        "has_resolution_signal": False,
    },
    {
        "conv_id": 200,
        "customer_messages": ["Office 365 subscription keeps failing with error 0x80070005."],
        "brand_responses": ["Please sign out and sign back in."],
        "has_resolution_signal": False,
    },
    {
        "conv_id": 300,
        "customer_messages": [
            "Windows update crashes every time.",
            "Thanks, that worked!",
        ],
        "brand_responses": ["Run the Windows Update Troubleshooter."],
        "has_resolution_signal": True,
    },
]


# ---------------------------------------------------------------------------
# EmbeddedRecord / EmbeddingIndex data class defaults
# ---------------------------------------------------------------------------

class TestEmbeddedRecordDefaults:

    def test_defaults(self):
        r = EmbeddedRecord(conv_id=1, text="hello", embedding=[0.1, 0.2])
        assert r.conv_id == 1
        assert r.metadata == {}

    def test_with_metadata(self):
        r = EmbeddedRecord(
            conv_id=2, text="x", embedding=[0.0], metadata={"key": "val"}
        )
        assert r.metadata["key"] == "val"


class TestEmbeddingIndexDefaults:

    def test_defaults(self):
        idx = EmbeddingIndex(model_name="test", dimension=3)
        assert idx.records == []
        assert idx.dimension == 3


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

class TestModelConfiguration:

    def test_default_model_name(self):
        builder = EmbeddingBuilder()
        assert builder.model_name == DEFAULT_MODEL

    def test_explicit_model_name(self):
        builder = EmbeddingBuilder(model_name="custom-model")
        assert builder.model_name == "custom-model"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "env-model-name")
        builder = EmbeddingBuilder()
        assert builder.model_name == "env-model-name"

    def test_explicit_beats_env_var(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "env-model")
        builder = EmbeddingBuilder(model_name="explicit")
        assert builder.model_name == "explicit"

    def test_model_loaded_lazily(self):
        builder = EmbeddingBuilder()
        assert builder._model is None
        builder._load_model()
        assert builder._model is not None


# ---------------------------------------------------------------------------
# prepare_text
# ---------------------------------------------------------------------------

class TestPrepareText:

    def test_single_message(self):
        result = EmbeddingBuilder.prepare_text(["Hello there"])
        assert result == "Hello there"

    def test_multiple_messages_joined(self):
        result = EmbeddingBuilder.prepare_text(["Problem A", "Problem B"])
        assert result == "Problem A | Problem B"

    def test_empty_messages(self):
        result = EmbeddingBuilder.prepare_text([])
        assert result == ""

    def test_whitespace_only_filtered(self):
        result = EmbeddingBuilder.prepare_text(["   ", "  "])
        assert result == ""

    def test_strips_whitespace(self):
        result = EmbeddingBuilder.prepare_text(["  hello  ", "  world  "])
        assert result == "hello | world"


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------

class TestEmbedTexts:

    def test_returns_correct_count(self):
        builder = EmbeddingBuilder()
        vectors = builder.embed_texts(["hello", "world", "test"])
        assert len(vectors) == 3

    def test_vector_dimension(self):
        builder = EmbeddingBuilder()
        vectors = builder.embed_texts(["hello"])
        assert len(vectors[0]) == DIM

    def test_empty_text_gets_zero_vector(self):
        builder = EmbeddingBuilder()
        vectors = builder.embed_texts([""])
        assert vectors[0] == [0.0] * DIM

    def test_nonempty_text_gets_nonzero_vector(self):
        builder = EmbeddingBuilder()
        vectors = builder.embed_texts(["Surface won't turn on"])
        assert any(v != 0.0 for v in vectors[0])

    def test_mixed_empty_and_nonempty(self):
        builder = EmbeddingBuilder()
        vectors = builder.embed_texts(["hello", "", "world"])
        assert vectors[1] == [0.0] * DIM
        assert any(v != 0.0 for v in vectors[0])
        assert any(v != 0.0 for v in vectors[2])


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:

    def test_index_record_count(self):
        builder = EmbeddingBuilder()
        index = builder.build_index(_SAMPLE_RECORDS)
        assert len(index.records) == 3

    def test_index_metadata(self):
        builder = EmbeddingBuilder()
        index = builder.build_index(_SAMPLE_RECORDS)
        assert index.model_name == DEFAULT_MODEL
        assert index.dimension == DIM

    def test_record_conv_ids_match(self):
        builder = EmbeddingBuilder()
        index = builder.build_index(_SAMPLE_RECORDS)
        conv_ids = [r.conv_id for r in index.records]
        assert conv_ids == [100, 200, 300]

    def test_record_texts_combined(self):
        builder = EmbeddingBuilder()
        index = builder.build_index(_SAMPLE_RECORDS)
        # conv_id 100 has two customer messages joined
        rec100 = next(r for r in index.records if r.conv_id == 100)
        assert "Surface" in rec100.text
        assert " | " in rec100.text

    def test_record_embedding_is_list(self):
        builder = EmbeddingBuilder()
        index = builder.build_index(_SAMPLE_RECORDS)
        for rec in index.records:
            assert isinstance(rec.embedding, list)
            assert len(rec.embedding) == DIM

    def test_metadata_preserved(self):
        builder = EmbeddingBuilder()
        index = builder.build_index(_SAMPLE_RECORDS)
        rec300 = next(r for r in index.records if r.conv_id == 300)
        assert rec300.metadata["has_resolution_signal"] is True
        assert rec300.metadata["num_customer_messages"] == 2

    def test_empty_input(self):
        builder = EmbeddingBuilder()
        index = builder.build_index([])
        assert index.records == []
        assert index.dimension == DIM


# ---------------------------------------------------------------------------
# save_index / load_index round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:

    def test_save_and_load(self, tmp_path):
        builder = EmbeddingBuilder()
        index = builder.build_index(_SAMPLE_RECORDS)
        path = str(tmp_path / "embeddings.json")
        EmbeddingBuilder.save_index(index, path)

        loaded = EmbeddingBuilder.load_index(path)
        assert loaded.model_name == index.model_name
        assert loaded.dimension == index.dimension
        assert len(loaded.records) == len(index.records)

    def test_loaded_records_match(self, tmp_path):
        builder = EmbeddingBuilder()
        index = builder.build_index(_SAMPLE_RECORDS)
        path = str(tmp_path / "emb.json")
        EmbeddingBuilder.save_index(index, path)

        loaded = EmbeddingBuilder.load_index(path)
        for orig, load in zip(index.records, loaded.records):
            assert orig.conv_id == load.conv_id
            assert orig.text == load.text
            assert orig.embedding == load.embedding
            assert orig.metadata == load.metadata

    def test_empty_index_round_trip(self, tmp_path):
        builder = EmbeddingBuilder()
        index = builder.build_index([])
        path = str(tmp_path / "empty.json")
        EmbeddingBuilder.save_index(index, path)

        loaded = EmbeddingBuilder.load_index(path)
        assert loaded.records == []
        assert loaded.dimension == DIM


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_empty_customer_messages(self):
        records = [
            {"conv_id": 1, "customer_messages": ["", "  "],
             "brand_responses": ["Hi"], "has_resolution_signal": False},
        ]
        builder = EmbeddingBuilder()
        index = builder.build_index(records)
        assert len(index.records) == 1
        # Empty text produces zero vector
        assert index.records[0].embedding == [0.0] * DIM

    def test_missing_customer_messages_key(self):
        records = [{"conv_id": 1}]
        builder = EmbeddingBuilder()
        index = builder.build_index(records)
        assert len(index.records) == 1
        assert index.records[0].text == ""

    def test_single_record(self):
        records = [_SAMPLE_RECORDS[0]]
        builder = EmbeddingBuilder()
        index = builder.build_index(records)
        assert len(index.records) == 1
