"""
Evidence Retriever Module for MicrosoftHelps AI Support Agent.

Retrieves relevant historical MicrosoftHelps conversations using FAISS
for fast dense vector similarity search over pre-computed embeddings.

Usage::

    from src.embeddings import EmbeddingBuilder, EmbeddingIndex

    # Load a previously built embedding index
    emb_index = EmbeddingBuilder.load_index("output/embeddings.json")

    # Build retriever from the embedding index
    retriever = EvidenceRetriever.from_embedding_index(emb_index)

    # Or load directly from a saved embedding index file
    retriever = EvidenceRetriever.from_embedding_json("output/embeddings.json")

    # Retrieve similar cases
    results = retriever.retrieve("My Surface won't turn on", top_k=5)
"""

import numpy as np
import faiss

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.embeddings import EmbeddingBuilder, EmbeddingIndex


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """Represents a retrieved piece of evidence/context from historical conversations.

    Attributes:
        id:      Unique identifier (the conversation ID).
        text:    The customer message text that was embedded.
        score:   Similarity score (0.0 to 1.0; higher = more similar).
        source:  Provenance tag for traceability.
        metadata: Arbitrary metadata carried over from the embedding index.
    """
    id: str
    text: str
    score: float = 0.0
    source: str = "twcs_historical"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FAISS-based retriever
# ---------------------------------------------------------------------------

class EvidenceRetriever:
    """Retrieves historical evidence using FAISS similarity search.

    FAISS (Facebook AI Similarity Search) builds a special index over all
    the embedding vectors so that finding the closest vectors to a query
    is nearly instant, even for millions of records.

    The retriever embeds a customer query with the *same* sentence-transformer
    model that was used to build the embedding index, then asks FAISS for
    the top-k nearest vectors.  Each match is returned as an ``Evidence``
    object with a similarity score and the original metadata.

    Parameters
    ----------
    embedding_index : EmbeddingIndex or None
        A pre-loaded embedding index.  If ``None``, you can call
        ``load_embedding_index()`` later or use the ``from_*`` factory
        methods.
    embedder : EmbeddingBuilder or None
        Used to embed the incoming query.  If ``None``, a fresh
        ``EmbeddingBuilder`` is created using the model name stored
        in the ``EmbeddingIndex``.
    """

    def __init__(
        self,
        embedding_index: Optional[EmbeddingIndex] = None,
        embedder: Optional[EmbeddingBuilder] = None,
    ) -> None:
        self._embedding_index: Optional[EmbeddingIndex] = embedding_index
        self._embedder = embedder
        self._faiss_index: Optional[faiss.Index] = None

        if self._embedding_index is not None:
            self._build_faiss_index()

    # -- factory methods ----------------------------------------------------

    @classmethod
    def from_embedding_index(
        cls,
        embedding_index: EmbeddingIndex,
        embedder: Optional[EmbeddingBuilder] = None,
    ) -> "EvidenceRetriever":
        """Create a retriever from a loaded EmbeddingIndex."""
        return cls(embedding_index=embedding_index, embedder=embedder)

    @classmethod
    def from_embedding_json(
        cls,
        path: str,
        embedder: Optional[EmbeddingBuilder] = None,
    ) -> "EvidenceRetriever":
        """Create a retriever by loading an EmbeddingIndex from a JSON file."""
        index = EmbeddingBuilder.load_index(path)
        return cls(embedding_index=index, embedder=embedder)

    @property
    def index_path(self) -> Optional[str]:
        """Return the file path if the index was loaded from disk, else None."""
        return getattr(self, "_index_path", None)

    @property
    def num_records(self) -> int:
        """Number of records in the FAISS index."""
        if self._faiss_index is None:
            return 0
        return self._faiss_index.ntotal

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        if self._embedding_index is None:
            return 0
        return self._embedding_index.dimension

    # -- internal helpers ---------------------------------------------------

    def _build_faiss_index(self) -> None:
        """Build a FAISS index from the loaded embedding records.

        Uses ``IndexFlatIP`` (inner product) which, for L2-normalized
        vectors, gives cosine similarity.  A flat (brute-force) index
        is used because it is simple and correct — a good fit for a
        codebase that values readability over raw speed.

        Raises
        ------
        ValueError
            If the embedding index has no records.
        """
        if not self._embedding_index.records:
            return

        dim = self._embedding_index.dimension
        vectors = np.array(
            [rec.embedding for rec in self._embedding_index.records],
            dtype=np.float32,
        )

        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(vectors)

    def _ensure_embedder(self) -> EmbeddingBuilder:
        """Return the embedder, creating one if needed."""
        if self._embedder is None:
            model_name = None
            if self._embedding_index is not None:
                model_name = self._embedding_index.model_name
            self._embedder = EmbeddingBuilder(model_name=model_name)
        return self._embedder

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into a float32 vector."""
        embedder = self._ensure_embedder()
        vectors = embedder.embed_texts([query])
        return np.array(vectors[0], dtype=np.float32).reshape(1, -1)

    # -- public API ---------------------------------------------------------

    def load_embedding_index(self, path: str) -> None:
        """Load an embedding index from a JSON file and build the FAISS index.

        Parameters
        ----------
        path : str
            Filesystem path to the JSON file produced by
            ``EmbeddingBuilder.save_index()``.
        """
        self._embedding_index = EmbeddingBuilder.load_index(path)
        self._build_faiss_index()

    def retrieve(
        self,
        query: str,
        intent: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Evidence]:
        """Retrieve the top-k most relevant evidence records for a query.

        Parameters
        ----------
        query : str
            Customer message text to search for.
        intent : str or None
            Optional intent label (currently unused; reserved for future
            intent-scoped retrieval).
        top_k : int
            Maximum number of results to return (default 5).

        Returns
        -------
        list of Evidence
            Each result includes a similarity ``score`` (0.0–1.0),
            the original customer message ``text``, the ``conv_id``,
            and any stored ``metadata``.

        Raises
        ------
        RuntimeError
            If no embedding index has been loaded.
        """
        if self._faiss_index is None or self._embedding_index is None:
            return []

        if not query or not query.strip():
            return []

        query_vec = self._embed_query(query)

        actual_k = min(top_k, self._faiss_index.ntotal)
        if actual_k == 0:
            return []

        distances, indices = self._faiss_index.search(query_vec, actual_k)

        results: List[Evidence] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            rec = self._embedding_index.records[idx]
            results.append(Evidence(
                id=str(rec.conv_id),
                text=rec.text,
                score=float(dist),
                metadata=rec.metadata,
            ))

        return results
