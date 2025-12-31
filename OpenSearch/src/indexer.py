import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from opensearchpy import helpers

from .config import settings


class Indexer:
    def __init__(self, client, data_dir: Path | str | None = None, cache_path: Path | str | None = None, index_name: str | None = None) -> None:
        self.client = client
        self.data_dir = Path(data_dir or settings.data_dir)
        self.cache_path = Path(cache_path or settings.cache_path)
        self.index_name = index_name or settings.index_name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def build_index(self, full_reindex: bool = False) -> Dict[str, int]:
        """
        Build or update the index.
        - Incremental mode: only new/changed files are re-indexed; removed files are deleted.
        - Full rebuild: drop the index and re-index every file.
        """
        self._ensure_index(recreate=full_reindex)
        previous_cache = {} if full_reindex else self._load_cache()

        current_files = {self._doc_id(p): self._file_signature(p) for p in self._discover_files()}
        to_index = [doc_id for doc_id, sig in current_files.items() if previous_cache.get(doc_id) != sig]
        to_delete = [doc_id for doc_id in previous_cache.keys() if doc_id not in current_files]

        actions: List[dict] = []
        actions.extend(self._index_actions(to_index))
        actions.extend(self._delete_actions(to_delete))

        if actions:
            helpers.bulk(self.client, actions)

        self._save_cache(current_files)
        return {
            "indexed": len(to_index),
            "deleted": len(to_delete),
            "skipped": len(current_files) - len(to_index),
        }

    def _ensure_index(self, recreate: bool = False) -> None:
        exists = bool(self.client.indices.exists(index=self.index_name))
        if recreate and exists:
            self.client.indices.delete(index=self.index_name)
            exists = False
        if not exists:
            body = {
                "settings": {
                    "index": {"number_of_shards": 1, "number_of_replicas": 0},
                    "analysis": {
                        "analyzer": {
                            "text_analyzer": {
                                "tokenizer": "standard",
                                "filter": ["lowercase"],
                            }
                        }
                    },
                },
                "mappings": {
                    "properties": {
                        "path": {"type": "keyword"},
                        "modified_at": {"type": "date"},
                        "content": {
                            "type": "text",
                            "analyzer": "text_analyzer",
                            "search_analyzer": "text_analyzer",
                        },
                    }
                },
            }
            self.client.indices.create(index=self.index_name, body=body)

    def _discover_files(self) -> Iterable[Path]:
        for path in self.data_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
                yield path

    def _file_signature(self, path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_mtime_ns}-{stat.st_size}"

    def _doc_id(self, path: Path) -> str:
        return str(path.relative_to(self.data_dir))

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("files", {})
        except Exception:
            return {}

    def _save_cache(self, cache: Dict[str, str]) -> None:
        payload = {"files": cache}
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _index_actions(self, doc_ids: List[str]) -> Iterable[dict]:
        for doc_id in doc_ids:
            file_path = self.data_dir / doc_id
            content = file_path.read_text(encoding="utf-8")
            modified_at = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()
            yield {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": doc_id,
                "_source": {
                    "path": doc_id,
                    "modified_at": modified_at,
                    "content": content,
                },
            }

    def _delete_actions(self, doc_ids: List[str]) -> Iterable[dict]:
        for doc_id in doc_ids:
            yield {"_op_type": "delete", "_index": self.index_name, "_id": doc_id}
