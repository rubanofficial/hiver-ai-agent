"""
Historical Evidence Builder for MicrosoftHelps conversations.

Extracts conversation-level evidence records from the TWCS dataset.
Each evidence record groups a full conversation thread (customer messages
and MicrosoftHelps responses) with metadata for traceability.

This module does NOT use embeddings, FAISS, or any LLM calls.
It produces a local JSON file suitable for later embedding/search.
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Resolution-signal regex (borrowed from resolution_analysis.py)
# ---------------------------------------------------------------------------

_RESOLVED_RE = re.compile(
    r"\b(thank\s+you|thanks\b|that\s+worked|it\s+worked|fixed\b|sorted\b|"
    r"works\s+now|appreciate\s+it|all\s+set\b|resolved\b|great\s+help|"
    r"solved\b|problem\s+solved|issue\s+(is\s+)?fixed|no\s+longer|"
    r"finally\s+working|back\s+up|back\s+online|got\s+it\s+working)",
    re.I,
)


# ---------------------------------------------------------------------------
# Parent-map helpers (reused from validate_microsoft / resolution_analysis)
# ---------------------------------------------------------------------------

def build_parent_map(df: pd.DataFrame) -> Dict[int, int]:
    """Build a child -> parent mapping from in_response_to_tweet_id.

    Parameters
    ----------
    df : DataFrame with columns ``tweet_id`` (int) and
         ``in_response_to_tweet_id`` (str or numeric).

    Returns
    -------
    dict mapping child tweet_id -> parent tweet_id.
    """
    p_num = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    valid = p_num.notna()
    return dict(zip(df.loc[valid, "tweet_id"].astype(int), p_num[valid].astype(int)))


def find_root(tid: int, parent_map: Dict[int, int],
              root_cache: Optional[Dict[int, int]] = None) -> int:
    """Find the root tweet_id for a given tweet, with path compression.

    Parameters
    ----------
    tid         : The tweet_id to resolve.
    parent_map  : child -> parent mapping.
    root_cache  : Mutable cache dict (created automatically if None).

    Returns
    -------
    The root tweet_id (an integer).
    """
    if root_cache is None:
        root_cache = {}
    curr = tid
    path: List[int] = []
    while curr in parent_map:
        if curr in root_cache:
            curr = root_cache[curr]
            break
        path.append(curr)
        curr = parent_map[curr]
        if len(path) > 60:          # loop guard
            break
    for node in path:
        root_cache[node] = curr
    root_cache[tid] = curr
    return curr


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TweetRecord:
    """One tweet inside a conversation evidence record."""
    tweet_id: int
    author_id: str
    inbound: bool
    text: str
    created_at: str = ""


@dataclass
class ConversationEvidence:
    """A complete conversation-level evidence record.

    This is what gets saved to JSON and later used for retrieval.
    It groups all tweets in a conversation thread together, separating
    customer messages from MicrosoftHelps responses, and carries enough
    metadata to trace back to the original dataset rows.
    """
    conv_id: int
    tweets: List[TweetRecord] = field(default_factory=list)
    customer_messages: List[str] = field(default_factory=list)
    brand_responses: List[str] = field(default_factory=list)
    num_turns: int = 0
    has_resolution_signal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

class ConversationEvidenceBuilder:
    """Extracts MicrosoftHelps conversation evidence from a TWCS DataFrame.

    Usage::

        builder = ConversationEvidenceBuilder(df)
        evidence = builder.build()
        builder.save_json(evidence, "output/evidence.json")

    Parameters
    ----------
    df : pd.DataFrame
        The full TWCS dataset (or a pre-filtered subset).  Must contain
        columns: tweet_id, author_id, inbound, text, in_response_to_tweet_id.
        ``created_at`` is optional; missing values become empty strings.
    brand : str
        The brand author_id to filter on (default ``"MicrosoftHelps"``).
    """

    def __init__(self, df: pd.DataFrame, brand: str = "MicrosoftHelps"):
        self.brand = brand
        self._df = df
        self._parent_map: Dict[int, int] = {}
        self._root_cache: Dict[int, int] = {}
        self._conv_ids: Optional[pd.Series] = None

    # -- internal helpers ---------------------------------------------------

    def _ensure_parent_map(self) -> None:
        if not self._parent_map:
            self._parent_map = build_parent_map(self._df)

    def _get_root(self, tid: int) -> int:
        self._ensure_parent_map()
        return find_root(tid, self._parent_map, self._root_cache)

    def _assign_conv_ids(self) -> pd.Series:
        """Return a Series of conv_id aligned with self._df rows."""
        if self._conv_ids is None:
            self._ensure_parent_map()
            self._conv_ids = self._df["tweet_id"].apply(self._get_root)
        return self._conv_ids

    # -- public API ---------------------------------------------------------

    def filter_brand_conversations(self) -> pd.DataFrame:
        """Return a DataFrame containing only tweets belonging to conversations
        where the brand posted at least one outbound tweet.

        This avoids scanning all 3M rows during evidence extraction.
        """
        if self._df.empty or "inbound" not in self._df.columns:
            return self._df.copy()
        brand_out = self._df[
            (~self._df["inbound"]) & (self._df["author_id"] == self.brand)
        ]
        conv_ids = set(brand_out["tweet_id"].apply(self._get_root))

        mask = self._df["tweet_id"].apply(self._get_root).isin(conv_ids)
        result = self._df[mask].copy()
        if result.empty:
            result["conv_id"] = pd.Series(dtype="int64")
        else:
            result["conv_id"] = result["tweet_id"].apply(self._get_root)
        return result

    def build(self, min_customer_len: int = 1) -> List[ConversationEvidence]:
        """Build evidence records from all MicrosoftHelps conversations.

        Parameters
        ----------
        min_customer_len : int
            Minimum character length for a customer message to be included
            in ``customer_messages`` (default 1 = keep everything).

        Returns
        -------
        list of ConversationEvidence, sorted by conv_id.
        """
        t0 = time.time()
        brand_df = self.filter_brand_conversations()
        if brand_df.empty or "conv_id" not in brand_df.columns:
            return []
        print(
            f"[evidence] Filtered to {len(brand_df):,} tweets across "
            f"MicrosoftHelps conversations in {time.time() - t0:.2f}s"
        )

        evidence_list: List[ConversationEvidence] = []

        for cid, group in brand_df.sort_values(["conv_id", "tweet_id"]).groupby("conv_id"):
            tweets: List[TweetRecord] = []
            for row in group.itertuples(index=False):
                text = str(row.text) if pd.notna(row.text) else ""
                created = str(getattr(row, "created_at", "")) or ""
                tweets.append(TweetRecord(
                    tweet_id=int(row.tweet_id),
                    author_id=str(row.author_id),
                    inbound=bool(row.inbound),
                    text=text,
                    created_at=created,
                ))

            # Require at least one MicrosoftHelps outbound tweet
            brand_turns = [t for t in tweets if not t.inbound and t.author_id == self.brand]
            cust_turns = [t for t in tweets if t.inbound]
            if not brand_turns:
                continue

            customer_texts = [t.text for t in cust_turns if len(t.text) >= min_customer_len]
            brand_texts = [t.text for t in brand_turns]

            # Resolution signal in last customer message
            has_resolution = False
            if cust_turns:
                has_resolution = bool(_RESOLVED_RE.search(cust_turns[-1].text))

            ev = ConversationEvidence(
                conv_id=int(cid),
                tweets=tweets,
                customer_messages=customer_texts,
                brand_responses=brand_texts,
                num_turns=len(tweets),
                has_resolution_signal=has_resolution,
                metadata={
                    "brand": self.brand,
                    "num_brand_turns": len(brand_turns),
                    "num_customer_turns": len(cust_turns),
                },
            )
            evidence_list.append(ev)

        evidence_list.sort(key=lambda e: e.conv_id)
        print(f"[evidence] Built {len(evidence_list)} conversation evidence records")
        return evidence_list

    @staticmethod
    def save_json(evidence: List[ConversationEvidence], path: str) -> None:
        """Save evidence records to a JSON file.

        Each record is serialised with full tweet details so the evidence
        can be traced back to individual dataset rows.
        """
        data = []
        for ev in evidence:
            data.append({
                "conv_id": ev.conv_id,
                "tweets": [asdict(t) for t in ev.tweets],
                "customer_messages": ev.customer_messages,
                "brand_responses": ev.brand_responses,
                "num_turns": ev.num_turns,
                "has_resolution_signal": ev.has_resolution_signal,
                "metadata": ev.metadata,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[evidence] Saved {len(data)} records to {path}")

    @staticmethod
    def load_json(path: str) -> List[Dict[str, Any]]:
        """Load previously saved evidence records from JSON.

        Returns a list of plain dicts (not ConversationEvidence objects)
        for easy inspection or downstream processing.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
