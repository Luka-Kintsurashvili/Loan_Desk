"""Retrieval over the lender's policy page: parse -> chunk by heading -> TF-IDF -> cosine search.

A one-page, six-section document doesn't need a vector database or dense embeddings;
TF-IDF plus a linear scan is the right-sized choice here, not a shortcut.
"""
from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def load_policy_chunks(path: Path, policy: dict) -> list[tuple[str, str]]:
    """Parse the policy markdown page into (section heading, section text) chunks, with
    policy.json's numeric limits filled into the page's {placeholders} so the two can't drift apart."""
    text = path.read_text(encoding="utf-8").format(
        max_pti_pct=policy["max_pti"] * 100, max_dti_pct=policy["max_dti_after"] * 100,
        min_fico=policy["min_fico"], max_ltv_pct=policy["max_ltv"] * 100)
    parts = re.split(r"(?m)^## (.+)$", text)  # parts[0] is the title/intro before the first heading
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


class PolicySearch:
    """A small TF-IDF index over the policy document's chunks."""

    def __init__(self, chunks: list[tuple[str, str]], source: str):
        self.chunks = chunks
        self.source = source
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._vectors = self._vectorizer.fit_transform([heading + " " + body for heading, body in chunks])

    def search(self, query: str, top_k: int = 3) -> dict:
        """Return up to top_k chunks most relevant to a free-text query, with section citations."""
        sims = cosine_similarity(self._vectorizer.transform([str(query)]), self._vectors)[0]
        order = sims.argsort()[::-1]
        top = [i for i in order[:top_k] if sims[i] > 0] or [int(order[0])]
        return {"results": [{"section": self.chunks[i][0], "text": self.chunks[i][1], "source": self.source} for i in top]}


_INDEX = PolicySearch(load_policy_chunks(config.POLICY_DOC_PATH, config.POLICY), "policy_docs/lender_policy.md")


def search_policy_docs(query: str) -> dict:
    """Tool entry point: search the lender's written policy for the section behind a rule."""
    return _INDEX.search(query)
