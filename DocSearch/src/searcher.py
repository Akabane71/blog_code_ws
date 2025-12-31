from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from rapidfuzz import fuzz, process

from .indexer import IndexStore


class SearchEngine:
    def __init__(self, store: IndexStore, fuzzy_threshold: int = 82, fuzzy_limit: int = 3) -> None:
        self.store = store
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_limit = fuzzy_limit

    def search(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        self.store.load()
        tokens = self.store.tokenize(query)
        if not tokens or not self.store.index["docs"]:
            return []

        vocabulary = list(self.store.index["inverted_index"].keys())
        expanded_tokens = self._expand_tokens(tokens, vocabulary)
        scores: Dict[str, float] = defaultdict(float)

        doc_lengths = {doc_id: meta.get("length", 1) or 1 for doc_id, meta in self.store.index["docs"].items()}
        doc_count = max(len(doc_lengths), 1)

        for token, is_fuzzy in expanded_tokens:
            postings = self.store.index["inverted_index"].get(token)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((doc_count + 1) / (df + 1)) + 1
            weight = 0.8 if is_fuzzy else 1.0
            for doc_id, tf in postings.items():
                length = doc_lengths.get(doc_id, 1) or 1
                scores[doc_id] += weight * (tf / length) * idf

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    def _expand_tokens(
        self, tokens: Iterable[str], vocabulary: List[str]
    ) -> List[Tuple[str, bool]]:
        expanded: List[Tuple[str, bool]] = []
        vocab_set = set(vocabulary)
        for token in tokens:
            expanded.append((token, False))
            matches = process.extract(
                token,
                vocabulary,
                scorer=fuzz.WRatio,
                limit=self.fuzzy_limit,
                score_cutoff=self.fuzzy_threshold,
            )
            for match, _, _ in matches:
                if match in vocab_set and match != token:
                    expanded.append((match, True))
        return expanded


__all__ = ["SearchEngine"]
