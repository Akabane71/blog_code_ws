from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import jieba

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "tmp"
INDEX_FILE = CACHE_DIR / "index.json"


def _default_index() -> Dict:
    return {
        "version": 1,
        "updated_at": None,
        "docs": {},
        "inverted_index": {},
    }


class IndexStore:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        cache_dir: Path = CACHE_DIR,
        index_file: Path = INDEX_FILE,
    ) -> None:
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.index_file = index_file
        self.index: Dict = _default_index()
        self._loaded = False

    def load(self) -> None:
        if self._loaded and self.index:
            return
        if self.index_file.exists():
            try:
                self.index = json.loads(self.index_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.index = _default_index()
        else:
            self.index = _default_index()
        self._loaded = True

    def save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = self._index_for_storage()
        self.index_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text)
        raw_tokens = jieba.lcut(cleaned, cut_all=False)
        tokens: List[str] = []
        for token in raw_tokens:
            token = token.strip().lower()
            if not token or token.isspace():
                continue
            tokens.append(token)
        return tokens

    def build_index(self, full_rebuild: bool = False) -> Dict[str, int]:
        self.load()
        if full_rebuild:
            self.index = _default_index()

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

        current_files = {
            path.name: (path, path.stat().st_mtime)
            for path in sorted(self.data_dir.glob("*.txt"))
        }

        removed_docs = self._remove_deleted_docs(current_files)

        added = 0
        updated = 0

        for doc_id, (path, mtime) in current_files.items():
            doc_meta = self.index["docs"].get(doc_id)
            if doc_meta and abs(doc_meta.get("mtime", 0) - mtime) < 1e-6:
                continue
            if doc_meta:
                self._remove_doc(doc_id)
                updated += 1
            else:
                added += 1
            self._index_single_file(doc_id, path, mtime)

        self.index["updated_at"] = time.time()
        self.save()
        return {
            "added": added,
            "updated": updated,
            "removed": removed_docs,
            "total_docs": len(self.index["docs"]),
            "vocab_size": len(self.index["inverted_index"]),
        }

    def _index_single_file(self, doc_id: str, path: Path, mtime: float) -> None:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tokens = self.tokenize(text)
        freq = Counter(tokens)
        length = sum(freq.values())

        for token, count in freq.items():
            postings = self.index["inverted_index"].setdefault(token, {})
            postings[doc_id] = count

        self.index["docs"][doc_id] = {
            "path": str(path),
            "mtime": mtime,
            "length": length,
            "tokens": dict(freq),
        }

    def _remove_deleted_docs(self, current_files: Dict[str, Tuple[Path, float]]) -> int:
        to_remove = [doc_id for doc_id in self.index["docs"] if doc_id not in current_files]
        for doc_id in to_remove:
            self._remove_doc(doc_id)
        return len(to_remove)

    def _remove_doc(self, doc_id: str) -> None:
        doc_meta = self.index["docs"].get(doc_id)
        if not doc_meta:
            return
        tokens = doc_meta.get("tokens", {})
        for token, count in tokens.items():
            postings = self.index["inverted_index"].get(token)
            if not postings:
                continue
            postings.pop(doc_id, None)
            if not postings:
                self.index["inverted_index"].pop(token, None)
        self.index["docs"].pop(doc_id, None)

    def _index_for_storage(self) -> Dict:
        # Ensure integers stay integers for JSON serialization
        docs = {
            doc_id: {
                "path": meta["path"],
                "mtime": meta["mtime"],
                "length": int(meta["length"]),
                "tokens": {token: int(count) for token, count in meta.get("tokens", {}).items()},
            }
            for doc_id, meta in self.index["docs"].items()
        }
        inverted_index = {
            token: {doc_id: int(count) for doc_id, count in postings.items()}
            for token, postings in self.index["inverted_index"].items()
        }
        return {
            "version": self.index.get("version", 1),
            "updated_at": self.index.get("updated_at"),
            "docs": docs,
            "inverted_index": inverted_index,
        }


__all__ = ["IndexStore", "DATA_DIR", "CACHE_DIR", "INDEX_FILE"]
