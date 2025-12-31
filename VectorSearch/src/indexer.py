import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
METADATA_FILE = "index_metadata.json"
FAISS_INDEX_FILE = "index.faiss"


def chunk_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks to keep semantic continuity."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(normalized[start:end])
        if end >= length:
            break
        start = end - overlap
    return chunks


class EmbeddingClient:
    """Thin wrapper around a local, open-source embedding model."""

    def __init__(self, model_name: str | None = None) -> None:
        # 默认选用较小、CPU 友好的模型
        name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")
        self.model = SentenceTransformer(name)
        self.dim = self.model.get_sentence_embedding_dimension()

    @classmethod
    def from_env(cls) -> "EmbeddingClient":
        return cls()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]


@dataclass
class SearchResult:
    text: str
    source: str
    score: float


class DocumentIndexer:
    """Manage document embedding, caching, and vector search."""

    def __init__(
        self,
        data_dir: Path,
        tmp_dir: Path,
        embed_client: EmbeddingClient,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.data_dir = data_dir
        self.tmp_dir = tmp_dir
        self.meta_path = self.tmp_dir / METADATA_FILE
        self.index_path = self.tmp_dir / FAISS_INDEX_FILE
        self.embed_client = embed_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata: Dict = {"files": {}, "index_dim": None}
        self.index: faiss.IndexFlatL2 | None = None
        self.entries: List[Tuple[str, str]] = []  # (file_path, chunk_text)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata()
        if self.metadata["files"]:
            self._rebuild_index_from_metadata()

    def _load_metadata(self) -> None:
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def _save_metadata(self) -> None:
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _scan_files(self) -> Dict[str, float]:
        files: Dict[str, float] = {}
        for path in sorted(self.data_dir.glob("*.txt")):
            try:
                files[str(path)] = path.stat().st_mtime
            except OSError:
                continue
        return files

    def _rebuild_index_from_metadata(self) -> None:
        embeddings: List[List[float]] = []
        entries: List[Tuple[str, str]] = []
        for file_path in sorted(self.metadata["files"].keys()):
            file_info = self.metadata["files"][file_path]
            for chunk in file_info.get("chunks", []):
                embeddings.append(chunk["embedding"])
                entries.append((file_path, chunk["text"]))

        if not embeddings:
            self.index = None
            self.entries = []
            return

        dim = len(embeddings[0])
        self.metadata["index_dim"] = dim
        vectors = np.array(embeddings, dtype="float32")
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        self.entries = entries

        faiss.write_index(self.index, str(self.index_path))

    def _update_file_chunks(self, path: str, mtime: float) -> int:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            self.metadata["files"].pop(path, None)
            return 0

        embeddings = self.embed_client.embed_texts(chunks)
        if embeddings:
            self.metadata["index_dim"] = len(embeddings[0])

        stored_chunks = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            stored_chunks.append(
                {
                    "id": f"{path}#{idx}",
                    "text": chunk,
                    "embedding": [float(v) for v in embedding],
                }
            )

        self.metadata["files"][path] = {"mtime": mtime, "chunks": stored_chunks}
        return len(stored_chunks)

    def build(self, force_rebuild: bool = False) -> Dict[str, int]:
        if force_rebuild:
            self.metadata = {"files": {}, "index_dim": None}

        current_files = self._scan_files()
        deleted = [p for p in self.metadata["files"].keys() if p not in current_files]
        for path in deleted:
            self.metadata["files"].pop(path, None)

        changed_files = []
        for path, mtime in current_files.items():
            meta_entry = self.metadata["files"].get(path)
            if not meta_entry or meta_entry.get("mtime") != mtime:
                changed_files.append((path, mtime))

        chunks_added = 0
        for path, mtime in changed_files:
            chunks_added += self._update_file_chunks(path, mtime)

        if self.metadata["files"]:
            self._rebuild_index_from_metadata()
        else:
            self.index = None
            self.entries = []

        self._save_metadata()
        return {
            "files_processed": len(changed_files),
            "files_deleted": len(deleted),
            "chunks_added": chunks_added,
        }

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not self.index or self.index.ntotal == 0:
            raise RuntimeError("索引为空，请先构建或重建。")

        vector = self.embed_client.embed_texts([query])[0]
        dim = len(vector)
        if dim != self.metadata.get("index_dim"):
            raise RuntimeError("Embedding 维度不匹配，请重建索引。")

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(np.array([vector], dtype="float32"), k)
        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.entries):
                continue
            file_path, text = self.entries[idx]
            results.append(SearchResult(text=text, source=file_path, score=float(score)))
        return results
