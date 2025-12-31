from dataclasses import dataclass
from typing import List

from opensearchpy import OpenSearch

from .config import settings


class Searcher:
    def __init__(self, client: OpenSearch, index_name: str | None = None) -> None:
        self.client = client
        self.index_name = index_name or settings.index_name

    def search(self, query: str, size: int = 5) -> List["SearchResult"]:
        body = {
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content"],
                    "fuzziness": "AUTO",
                    "operator": "and",
                }
            },
        }
        response = self.client.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        results: List[SearchResult] = []
        for hit in hits:
            results.append(
                SearchResult(
                    path=hit["_source"]["path"],
                    score=float(hit.get("_score", 0.0)),
                )
            )
        return results


@dataclass
class SearchResult:
    path: str
    score: float
