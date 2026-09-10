"""
Tests for the FAISS-based evidence retriever (src.agent.retriever).

All tests use synthetic embeddings — no real TWCS dataset, no Gemini API,
and no internet access required.
"""

import json
import os
from typing import List

import numpy as np
import pytest

from src.embeddings import EmbeddedRecord, EmbeddingIndex, EmbeddingBuilder
from src.agent.retriever import Evidence, EvidenceRetriever


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

DIM = 8  # small dimension for hand-crafted test vectors


def _make_record(conv_id: int, embedding: List[float], text: str = "",
                 metadata: dict = None) -> EmbeddedRecord:
    return EmbeddedRecord(
        conv_id=conv_id,
        text=text or f"Customer message for conversation {conv_id}",
        embedding=embedding,
        metadata=metadata or {},
    )


def _make_index(records: List[EmbeddedRecord]) -> EmbeddingIndex:
    dim = len(records[0].embedding) if records else DIM
    return EmbeddingIndex(model_name="test-model", dimension=dim, records=records)


# Three semantic clusters: surface, billing, network
_SURFACE_RECORDS = [
    _make_record(100, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 "My Surface won't turn on",
                 {"category": "surface", "has_resolution_signal": False}),
    _make_record(101, [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 "Surface Pro screen is flickering",
                 {"category": "surface", "has_resolution_signal": False}),
    _make_record(102, [0.85, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 "Surface Book keyboard not responding",
                 {"category": "surface", "has_resolution_signal": True}),
]

_BILLING_RECORDS = [
    _make_record(200, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                 "I was charged twice for my subscription",
                 {"category": "billing", "has_resolution_signal": False}),
    _make_record(201, [0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0],
                 "Refund for incorrect billing amount",
                 {"category": "billing", "has_resolution_signal": False}),
]

_NETWORK_RECORDS = [
    _make_record(300, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 "Cannot connect to Wi-Fi after update",
                 {"category": "network", "has_resolution_signal": False}),
    _make_record(301, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
                 "Internet keeps dropping every few minutes",
                 {"category": "network", "has_resolution_signal": True}),
]

ALL_RECORDS = _SURFACE_RECORDS + _BILLING_RECORDS + _NETWORK_RECORDS
ALL_INDEX = _make_index(ALL_RECORDS)


# ---------------------------------------------------------------------------
# Fake embedder (avoids loading SentenceTransformer)
# ---------------------------------------------------------------------------

class _FakeEmbedder:
    """Returns deterministic vectors for known queries, zeros for unknown."""

    _QUERY_VECTORS = {
        "surface":   [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "billing":   [0.0, 0.0, 0.0, 0.0, 0.95, 0.05, 0.0, 0.0],
        "network":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.95],
    }

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        results = []
        for t in texts:
            lower = t.lower()
            matched = False
            for keyword, vec in self._QUERY_VECTORS.items():
                if keyword in lower:
                    results.append(vec)
                    matched = True
                    break
            if not matched:
                results.append([0.0] * DIM)
        return results


def _build_retriever(records=None) -> EvidenceRetriever:
    """Convenience: build a retriever with synthetic data."""
    idx = _make_index(records or ALL_RECORDS)
    return EvidenceRetriever(
        embedding_index=idx,
        embedder=_FakeEmbedder(),
    )


# ---------------------------------------------------------------------------
# Evidence dataclass
# ---------------------------------------------------------------------------

class TestEvidenceDataclass:

    def test_defaults(self):
        e = Evidence(id="42", text="hello")
        assert e.id == "42"
        assert e.score == 0.0
        assert e.source == "twcs_historical"
        assert e.metadata == {}

    def test_with_metadata(self):
        e = Evidence(id="1", text="x", score=0.8, metadata={"key": "val"})
        assert e.metadata["key"] == "val"
        assert e.score == 0.8


# ---------------------------------------------------------------------------
# Default / empty retriever
# ---------------------------------------------------------------------------

class TestDefaultRetriever:

    def test_no_index_returns_empty(self):
        ret = EvidenceRetriever()
        results = ret.retrieve("hello")
        assert results == []

    def test_num_records_zero(self):
        ret = EvidenceRetriever()
        assert ret.num_records == 0

    def test_dimension_zero(self):
        ret = EvidenceRetriever()
        assert ret.dimension == 0

    def test_index_path_none_by_default(self):
        ret = EvidenceRetriever()
        assert ret.index_path is None


# ---------------------------------------------------------------------------
# FAISS index building
# ---------------------------------------------------------------------------

class TestFaissIndexBuilding:

    def test_index_built_from_constructor(self):
        ret = _build_retriever()
        assert ret.num_records == len(ALL_RECORDS)

    def test_dimension_matches(self):
        ret = _build_retriever()
        assert ret.dimension == DIM

    def test_empty_records_no_faiss_index(self):
        idx = _make_index([])
        ret = EvidenceRetriever(embedding_index=idx, embedder=_FakeEmbedder())
        assert ret.num_records == 0
        assert ret._faiss_index is None

    def test_single_record(self):
        rec = [_make_record(1, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
        ret = _build_retriever(rec)
        assert ret.num_records == 1


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

class TestRetrieve:

    def test_returns_evidence_objects(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem")
        assert all(isinstance(r, Evidence) for r in results)

    def test_default_top_k_is_5(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem")
        assert len(results) <= 5

    def test_top_k_configurable(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem", top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_index(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem", top_k=100)
        assert len(results) == len(ALL_RECORDS)

    def test_top_k_1_returns_single(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem", top_k=1)
        assert len(results) == 1

    def test_scores_are_floats(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem")
        for r in results:
            assert isinstance(r.score, float)

    def test_scores_are_non_negative(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem")
        for r in results:
            assert r.score >= 0.0

    def test_scores_are_sorted_descending(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_preserves_conv_id(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem")
        conv_ids = {r.id for r in results}
        all_ids = {str(rec.conv_id) for rec in ALL_RECORDS}
        assert conv_ids.issubset(all_ids)

    def test_preserves_text(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem")
        for r in results:
            assert len(r.text) > 0

    def test_preserves_metadata(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem")
        for r in results:
            assert "category" in r.metadata
            assert "has_resolution_signal" in r.metadata


# ---------------------------------------------------------------------------
# Semantic similarity ordering
# ---------------------------------------------------------------------------

class TestSimilarityOrdering:

    def test_surface_query_returns_surface_first(self):
        ret = _build_retriever()
        results = ret.retrieve("My Surface device has an issue", top_k=3)
        # All top-3 should be surface-related (conv_ids 100, 101, 102)
        top_ids = {r.id for r in results}
        assert top_ids == {"100", "101", "102"}

    def test_billing_query_returns_billing_first(self):
        ret = _build_retriever()
        results = ret.retrieve("Billing and charges are wrong", top_k=2)
        top_ids = {r.id for r in results}
        assert top_ids == {"200", "201"}

    def test_network_query_returns_network_first(self):
        ret = _build_retriever()
        results = ret.retrieve("Wi-Fi network connection lost", top_k=2)
        top_ids = {r.id for r in results}
        assert top_ids == {"300", "301"}

    def test_surface_score_higher_than_billing(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem", top_k=len(ALL_RECORDS))
        surface_scores = [r.score for r in results if r.metadata.get("category") == "surface"]
        billing_scores = [r.score for r in results if r.metadata.get("category") == "billing"]
        assert min(surface_scores) > max(billing_scores)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_query_returns_empty(self):
        ret = _build_retriever()
        assert ret.retrieve("") == []
        assert ret.retrieve("   ") == []

    def test_empty_index_returns_empty(self):
        idx = _make_index([])
        ret = EvidenceRetriever(embedding_index=idx, embedder=_FakeEmbedder())
        assert ret.retrieve("hello") == []

    def test_intent_parameter_is_accepted(self):
        """The intent parameter should be accepted but not break retrieval."""
        ret = _build_retriever()
        results = ret.retrieve("Surface problem", intent="Technical Troubleshooting")
        assert len(results) > 0

    def test_zero_top_k_returns_empty(self):
        ret = _build_retriever()
        results = ret.retrieve("Surface problem", top_k=0)
        assert results == []


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestFactoryMethods:

    def test_from_embedding_index(self):
        ret = EvidenceRetriever.from_embedding_index(ALL_INDEX, _FakeEmbedder())
        assert ret.num_records == len(ALL_RECORDS)

    def test_from_embedding_json(self, tmp_path):
        # Save the index to a temp file
        path = str(tmp_path / "test_embeddings.json")
        data = {
            "model_name": "test-model",
            "dimension": DIM,
            "records": [
                {
                    "conv_id": r.conv_id,
                    "text": r.text,
                    "embedding": r.embedding,
                    "metadata": r.metadata,
                }
                for r in ALL_RECORDS
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)

        ret = EvidenceRetriever.from_embedding_json(path, _FakeEmbedder())
        assert ret.num_records == len(ALL_RECORDS)
        results = ret.retrieve("Surface problem", top_k=2)
        assert len(results) == 2

    def test_load_embedding_index_method(self, tmp_path):
        path = str(tmp_path / "idx.json")
        data = {
            "model_name": "test-model",
            "dimension": DIM,
            "records": [
                {
                    "conv_id": r.conv_id,
                    "text": r.text,
                    "embedding": r.embedding,
                    "metadata": r.metadata,
                }
                for r in ALL_RECORDS
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)

        ret = EvidenceRetriever()
        assert ret.num_records == 0
        ret.load_embedding_index(path)
        assert ret.num_records == len(ALL_RECORDS)


# ---------------------------------------------------------------------------
# Integration with EmbeddingBuilder save/load round-trip
# ---------------------------------------------------------------------------

class TestEmbeddingRoundTrip:
    """Verify retriever works end-to-end with EmbeddingBuilder's JSON format."""

    def test_round_trip_with_fake_model(self, tmp_path, monkeypatch):
        """Build index with fake model, save, load into retriever, search."""
        import src.embeddings as emb_mod

        # Patch SentenceTransformer to avoid network
        def _patched_load(self_inner):
            if self_inner._model is None:
                class _FakeST:
                    def encode(self, texts, show_progress_bar=False,
                               normalize_embeddings=False):
                        vecs = []
                        for t in texts:
                            vecs.append(np.array([float(len(t))] * DIM,
                                                 dtype=np.float32))
                        return np.array(vecs, dtype=np.float32)
                    def get_sentence_embedding_dimension(self):
                        return DIM
                self_inner._model = _FakeST()
            return self_inner._model

        monkeypatch.setattr(emb_mod.EmbeddingBuilder, "_load_model", _patched_load)

        # Build with the fake model
        builder = EmbeddingBuilder(model_name="fake-model")
        evidence_records = [
            {"conv_id": 1, "customer_messages": ["Surface won't turn on"],
             "brand_responses": ["Try holding power button."],
             "has_resolution_signal": False},
            {"conv_id": 2, "customer_messages": ["Billing is wrong"],
             "brand_responses": ["Contact support."],
             "has_resolution_signal": False},
        ]
        emb_index = builder.build_index(evidence_records)

        # Save to JSON
        path = str(tmp_path / "round_trip.json")
        EmbeddingBuilder.save_index(emb_index, path)

        # Load into retriever
        ret = EvidenceRetriever.from_embedding_json(path, builder)
        assert ret.num_records == 2

        # Query — builder's fake model uses text length as embedding,
        # so query "Surface" (len=7) will match record with text len closest to 7
        results = ret.retrieve("Surface problem here", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, Evidence) for r in results)
