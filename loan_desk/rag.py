"""Retrieval over the lender's policy pages: parse -> chunk by heading -> TF-IDF -> cosine search.

One-page, few-section documents don't need a vector database or dense embeddings;
TF-IDF plus a linear scan is the right-sized choice here, not a shortcut.
"""
from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def load_policy_chunks(path: Path, format_kwargs: dict) -> list[tuple[str, str]]:
    """Parse a policy markdown page into (section heading, section text) chunks, with its
    policy.json's numeric limits filled into the page's {placeholders} so the two can't drift apart."""
    text = path.read_text(encoding="utf-8").format(**format_kwargs)
    parts = re.split(r"(?m)^## (.+)$", text)  # parts[0] is the title/intro before the first heading
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


class PolicySearch:
    """A small TF-IDF index combining one or more policy documents' chunks, each chunk
    keeping its own source document for an accurate citation regardless of which
    document the best match came from."""

    def __init__(self, chunk_sources: list[tuple[list[tuple[str, str]], str]]):
        self.chunks = [(heading, body, source) for chunks, source in chunk_sources for heading, body in chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._vectors = self._vectorizer.fit_transform([heading + " " + body for heading, body, _ in self.chunks])

    def search(self, query: str, top_k: int = 3) -> dict:
        """Return up to top_k chunks most relevant to a free-text query, with section citations."""
        sims = cosine_similarity(self._vectorizer.transform([str(query)]), self._vectors)[0]
        order = sims.argsort()[::-1]
        top = [i for i in order[:top_k] if sims[i] > 0] or [int(order[0])]
        return {"results": [{"section": h, "text": b, "source": s} for h, b, s in (self.chunks[i] for i in top)]}


_PERSONAL_KWARGS = {"max_pti_pct": config.POLICY["max_pti"] * 100, "max_dti_pct": config.POLICY["max_dti_after"] * 100,
                     "min_fico": config.POLICY["min_fico"], "max_ltv_pct": config.POLICY["max_ltv"] * 100}
_BUSINESS_KWARGS = {"min_dscr": config.BUSINESS_POLICY["min_dscr"], "max_leverage": config.BUSINESS_POLICY["max_leverage"],
                     "min_revenue": config.BUSINESS_POLICY["min_revenue"],
                     "min_years_in_business": config.BUSINESS_POLICY["min_years_in_business"]}
_INDEX = PolicySearch([
    (load_policy_chunks(config.POLICY_DOC_PATH, _PERSONAL_KWARGS), "policy_docs/lender_policy.md"),
    (load_policy_chunks(config.BUSINESS_POLICY_DOC_PATH, _BUSINESS_KWARGS), "policy_docs/business_policy.md"),
])


def search_policy_docs(query: str) -> dict:
    """Tool entry point: search the lender's written policy (personal or business) for the section behind a rule."""
    return _INDEX.search(query)
