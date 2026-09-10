"""
Embedding component for MicrosoftHelps historical evidence.

Converts customer messages from evidence records into semantic vector
embeddings using Sentence Transformers.  The resulting index can later
be searched to find similar historical conversations.

This module does NOT use FAISS, Gemini, or any external API calls.
It produces a local JSON index suitable for future similarity search.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Default model (small, fast, good quality for CPU)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedRecord:
    """A single evidence record paired with its embedding vector."""
    conv_id: int
    text: str                   # the text that was embedded
    embedding: List[float]      # the vector (list of floats)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingIndex:
    """Collection of embedded records ready for similarity search."""
    model_name: str
    dimension: int
    records: List[EmbeddedRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core component
# ---------------------------------------------------------------------------

class EmbeddingBuilder:
    """Turns evidence records into vector embeddings.

    Usage::

        builder = EmbeddingBuilder()                      # uses default model
        index   = builder.build_index(evidence_records)   # list of dicts
        builder.save_index(index, "output/embeddings.json")

    Parameters
    ----------
    model_name : str or None
        Name of the sentence-transformer model.  Falls back to the
        ``EMBEDDING_MODEL`` environment variable, then to the default
        ``all-MiniLM-L6-v2``.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", DEFAULT_MODEL
        )
        self._model = None          # lazily loaded

    # -- lazy model loading -------------------------------------------------

    def _load_model(self):
        """Load the SentenceTransformer model (only once)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[embeddings] Loading model '{self.model_name}' ...")
            self._model = SentenceTransformer(self.model_name)
            print(f"[embeddings] Model loaded  (dim={self._model.get_sentence_embedding_dimension()})")
        return self._model

    # -- text preparation ---------------------------------------------------

    @staticmethod
    def prepare_text(customer_messages: List[str]) -> str:
        """Combine a conversation's customer messages into one embeddable string.

        The messages are joined with " | " so the model sees them as a
        single coherent problem description.

        Parameters
        ----------
        customer_messages : list of str
            Customer messages from a ConversationEvidence record.

        Returns
        -------
        str  – a single text string ready for embedding.
        """
        filtered = [m.strip() for m in customer_messages if m.strip()]
        return " | ".join(filtered) if filtered else ""

    # -- public API ---------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into embedding vectors.

        Parameters
        ----------
        texts : list of str
            Texts to embed.  Empty strings are skipped and get zero vectors.

        Returns
        -------
        list of list of float – one vector per input text.
        """
        model = self._load_model()
        dim = model.get_sentence_embedding_dimension()

        # Separate empty from non-empty to avoid encoding blank strings
        non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_idx]

        # Allocate zero vectors for all
        vectors: List[List[float]] = [[0.0] * dim] * len(texts)

        if non_empty_texts:
            t0 = time.time()
            raw = model.encode(
                non_empty_texts,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            elapsed = time.time() - t0
            print(
                f"[embeddings] Encoded {len(non_empty_texts)} texts "
                f"in {elapsed:.2f}s (dim={dim})"
            )
            for j, idx in enumerate(non_empty_idx):
                vectors[idx] = raw[j].tolist()

        return vectors

    def build_index(
        self,
        evidence_records: List[Dict[str, Any]],
    ) -> EmbeddingIndex:
        """Build an embedding index from evidence records (dicts or dataclasses).

        Parameters
        ----------
        evidence_records : list
            Each record should have keys ``conv_id`` and
            ``customer_messages`` (list of str).  This matches the output
            of ``ConversationEvidenceBuilder.build()`` when serialised to
            dicts (e.g. via ``save_json`` / ``load_json``).

        Returns
        -------
        EmbeddingIndex containing one EmbeddedRecord per evidence record.
        """
        if not evidence_records:
            model = self._load_model()
            dim = model.get_sentence_embedding_dimension()
            return EmbeddingIndex(model_name=self.model_name, dimension=dim)

        texts = [
            self.prepare_text(r.get("customer_messages", []))
            for r in evidence_records
        ]
        vectors = self.embed_texts(texts)

        records: List[EmbeddedRecord] = []
        for rec, text, vec in zip(evidence_records, texts, vectors):
            records.append(EmbeddedRecord(
                conv_id=rec["conv_id"],
                text=text,
                embedding=vec,
                metadata={
                    "num_customer_messages": len(rec.get("customer_messages", [])),
                    "has_resolution_signal": rec.get("has_resolution_signal", False),
                },
            ))

        dim = len(records[0].embedding) if records else 0
        index = EmbeddingIndex(
            model_name=self.model_name,
            dimension=dim,
            records=records,
        )
        print(f"[embeddings] Index built: {len(records)} records, dim={dim}")
        return index

    # -- serialisation ------------------------------------------------------

    @staticmethod
    def save_index(index: EmbeddingIndex, path: str) -> None:
        """Save an embedding index to JSON.

        Parameters
        ----------
        index : EmbeddingIndex
            The index to save.
        path : str
            Destination file path.
        """
        data = {
            "model_name": index.model_name,
            "dimension": index.dimension,
            "records": [
                {
                    "conv_id": r.conv_id,
                    "text": r.text,
                    "embedding": r.embedding,
                    "metadata": r.metadata,
                }
                for r in index.records
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[embeddings] Saved index ({len(index.records)} records) to {path}")

    @staticmethod
    def load_index(path: str) -> EmbeddingIndex:
        """Load a previously saved embedding index from JSON.

        Returns an EmbeddingIndex ready for similarity search.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [
            EmbeddedRecord(
                conv_id=r["conv_id"],
                text=r["text"],
                embedding=r["embedding"],
                metadata=r.get("metadata", {}),
            )
            for r in data["records"]
        ]
        return EmbeddingIndex(
            model_name=data["model_name"],
            dimension=data["dimension"],
            records=records,
        )
