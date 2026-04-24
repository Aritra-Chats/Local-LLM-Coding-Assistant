"""rag_engine.py — Sentinel repository RAG (Retrieval-Augmented Generation) engine.

Indexes source files by chunking them into overlapping text segments,
generating embeddings via the Ollama embedding API, persisting them as
numpy arrays, and performing cosine-similarity search at query time.

No external vector database required — the store is a lightweight
file-based index under the Sentinel workspace.
"""

import ast
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SENTINEL_HOME = Path(os.environ.get("SENTINEL_HOME", Path.home() / ".sentinel"))
_INDEX_DIR = _SENTINEL_HOME / "index"

_DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_EMBED_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/embeddings"

# Files to index
_INDEXABLE_EXTENSIONS = {".py", ".md", ".txt", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h"}

# Chunking settings
_CHUNK_SIZE = 400       # target chars per chunk
_CHUNK_OVERLAP = 80     # overlap between adjacent chunks


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single indexed text chunk.

    Attributes:
        chunk_id: Unique identifier (SHA-256 of content).
        file_path: Relative path of the source file.
        start_line: Line number where this chunk begins (1-based).
        end_line: Line number where this chunk ends (1-based).
        content: Raw text content of this chunk.
        language: File language hint (e.g. 'python', 'markdown').
    """

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str


@dataclass
class SearchResult:
    """A single RAG search result.

    Attributes:
        chunk: The matched Chunk.
        score: Cosine similarity score (0.0–1.0).
        rank: Result rank (1 = most relevant).
    """

    chunk: Chunk
    score: float
    rank: int


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------


class RAGEngine:
    """File-based RAG engine for repository indexing and semantic search.

    The engine maintains a per-project index directory containing:
        chunks.json   — chunk metadata list
        vectors.npy   — float32 embedding matrix (N × D)

    Attributes:
        project_root: Absolute path to the project being indexed.
        embedding_model: Ollama model tag used for embeddings.
        index_dir: Path to this project's index directory.
    """

    def __init__(
        self,
        project_root: str,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """Initialise the RAG engine for a project.

        Args:
            project_root: Absolute path to the project root directory.
            embedding_model: Ollama embedding model tag.
        """
        self.project_root = Path(project_root).resolve()
        self.embedding_model = embedding_model
        self.index_dir = _INDEX_DIR / self._project_slug()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: List[Chunk] = []
        self._vectors: Optional[np.ndarray] = None  # shape (N, D)
        self._load_index()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_project(self, force: bool = False) -> int:
        """Index all eligible files in the project root.

        Args:
            force: If True, re-index all files even if already present.

        Returns:
            Number of new chunks added.
        """
        existing_ids = {c.chunk_id for c in self._chunks}
        new_chunks: List[Chunk] = []

        for file_path in self._collect_files():
            rel = str(file_path.relative_to(self.project_root))
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for chunk in self._chunk_text(text, rel, self._detect_language(file_path)):
                if not force and chunk.chunk_id in existing_ids:
                    continue
                new_chunks.append(chunk)

        if not new_chunks:
            return 0

        embeddings = self._embed_chunks(new_chunks)
        self._append_to_index(new_chunks, embeddings)
        self._save_index()
        return len(new_chunks)

    def index_file(self, file_path: str) -> int:
        """Index or re-index a single file.

        Args:
            file_path: Absolute or project-relative path to the file.

        Returns:
            Number of chunks added or updated.
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return 0

        rel = str(path.relative_to(self.project_root))
        language = self._detect_language(path)
        new_chunks = self._chunk_text(text, rel, language)

        # Remove existing chunks for this file
        self._chunks = [c for c in self._chunks if c.file_path != rel]

        embeddings = self._embed_chunks(new_chunks)
        self._append_to_index(new_chunks, embeddings)
        self._save_index()
        return len(new_chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List["SearchResult"]:
        """Find the most relevant chunks for a query string.

        Results are cached in ``~/.sentinel/cache/rag/`` keyed on
        SHA-256(query + index_hash + top_k).  The index_hash is derived from
        the SHA-256 of the persisted vectors file so the cache is
        automatically invalidated when the project is re-indexed.

        Args:
            query: Natural-language or code query string.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of SearchResult instances.
        """
        if not self._chunks or self._vectors is None or len(self._vectors) == 0:
            return []

        # ── RAG cache read ────────────────────────────────────────────
        index_hash = self._index_hash()
        try:
            from context.context_cache import cache as _ctx_cache
            cached_hits = _ctx_cache.get_rag(query, index_hash, top_k)
            if cached_hits is not None:
                # Reconstruct SearchResult objects from cached dicts
                results: List[SearchResult] = []
                for hit in cached_hits:
                    chunk_data = hit.get("chunk", {})
                    if chunk_data:
                        results.append(SearchResult(
                            chunk=Chunk(**chunk_data),
                            score=float(hit.get("score", 0.0)),
                            rank=int(hit.get("rank", 0)),
                        ))
                if results:
                    return results
        except Exception:
            _ctx_cache = None

        # ── Embed query ───────────────────────────────────────────────
        query_vec = self._embed_text(query)
        if query_vec is None:
            return []

        # ── ANN / exact search ────────────────────────────────────────
        top_indices = self._ann_search(query_vec, top_k)

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            i = int(idx)
            if i < 0 or i >= len(self._chunks):
                continue
            # Score via exact cosine for accurate ranking of ANN candidates
            score = float(np.dot(
                query_vec / (np.linalg.norm(query_vec) + 1e-10),
                self._vectors[i] / (np.linalg.norm(self._vectors[i]) + 1e-10),
            ))
            results.append(
                SearchResult(
                    chunk=self._chunks[i],
                    score=score,
                    rank=rank,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        for rank, r in enumerate(results, start=1):
            r.rank = rank

        # ── RAG cache write ───────────────────────────────────────────
        try:
            from context.context_cache import cache as _ctx_cache2
            serialised = [
                {
                    "chunk": {
                        "chunk_id":   r.chunk.chunk_id,
                        "file_path":  r.chunk.file_path,
                        "start_line": r.chunk.start_line,
                        "end_line":   r.chunk.end_line,
                        "content":    r.chunk.content,
                        "language":   r.chunk.language,
                    },
                    "score": r.score,
                    "rank":  r.rank,
                }
                for r in results
            ]
            _ctx_cache2.set_rag(query, index_hash, top_k, serialised)
        except Exception:
            pass

        return results

    def _index_hash(self) -> str:
        """Return a short hash of the current vectors file for cache keying."""
        import hashlib as _hashlib
        try:
            vf = self._vectors_file()
            if vf.exists():
                data = vf.read_bytes()[:4096]  # first 4 KB is enough for a hash
                return _hashlib.sha256(data).hexdigest()[:16]
        except Exception:
            pass
        return "no-index"

    # ------------------------------------------------------------------
    # Index stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return statistics about the current index.

        Returns:
            Dict with chunk count, file count, and embedding dimensions.
        """
        files = {c.file_path for c in self._chunks}
        dims = int(self._vectors.shape[1]) if self._vectors is not None and len(self._vectors) else 0
        return {
            "chunks": len(self._chunks),
            "files": len(files),
            "embedding_dims": dims,
            "index_dir": str(self.index_dir),
        }

    def clear(self) -> None:
        """Clear the entire index for this project."""
        self._chunks = []
        self._vectors = None
        for f in (self._chunks_file(), self._vectors_file()):
            if Path(f).exists():
                Path(f).unlink()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Load index from disk if it exists."""
        chunks_file = self._chunks_file()
        vectors_file = self._vectors_file()

        if chunks_file.exists():
            raw = json.loads(chunks_file.read_text(encoding="utf-8"))
            self._chunks = [Chunk(**c) for c in raw]

        if vectors_file.exists() and self._chunks:
            self._vectors = np.load(str(vectors_file))

    def _save_index(self) -> None:
        """Persist index metadata and vectors to disk."""
        self._chunks_file().write_text(
            json.dumps([asdict(c) for c in self._chunks], indent=2),
            encoding="utf-8",
        )
        if self._vectors is not None and len(self._vectors):
            np.save(str(self._vectors_file()), self._vectors)

    def _append_to_index(
        self, new_chunks: List[Chunk], embeddings: np.ndarray
    ) -> None:
        """Append new chunks and their embeddings to the in-memory index.

        Args:
            new_chunks: List of new Chunk objects.
            embeddings: numpy array of shape (len(new_chunks), D).
        """
        self._chunks.extend(new_chunks)
        if self._vectors is None or len(self._vectors) == 0:
            self._vectors = embeddings
        else:
            self._vectors = np.vstack([self._vectors, embeddings])

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_text(
        self, text: str, file_path: str, language: str
    ) -> List[Chunk]:
        """Split a text into overlapping chunks.

        For Python files, attempts to split on top-level definitions first
        before falling back to character-based chunking.

        Args:
            text: Full file content.
            file_path: Relative file path (for metadata).
            language: Language identifier string.

        Returns:
            List of Chunk objects.
        """
        if language == "python":
            chunks = self._chunk_python(text, file_path)
            if chunks:
                return chunks

        return self._chunk_by_chars(text, file_path, language)

    def _chunk_python(self, text: str, file_path: str) -> List[Chunk]:
        """Chunk a Python file by top-level AST nodes.

        Args:
            text: Python source code.
            file_path: Relative path.

        Returns:
            List of Chunk objects, one per top-level definition/block.
            Returns empty list if parsing fails.
        """
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []

        lines = text.splitlines(keepends=True)
        chunks: List[Chunk] = []

        for node in ast.iter_child_nodes(tree):
            if not hasattr(node, "lineno"):
                continue
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1)
            segment = "".join(lines[start:end])
            if len(segment.strip()) < 10:
                continue
            chunks.append(
                Chunk(
                    chunk_id=self._hash(file_path + segment),
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=end,
                    content=segment,
                    language="python",
                )
            )

        return chunks

    def _chunk_by_chars(
        self, text: str, file_path: str, language: str
    ) -> List[Chunk]:
        """Split text into overlapping character-window chunks.

        Args:
            text: Full file content.
            file_path: Relative path.
            language: Language identifier.

        Returns:
            List of Chunk objects.
        """
        lines = text.splitlines(keepends=True)
        chunks: List[Chunk] = []
        current: List[str] = []
        current_len = 0
        start_line = 1

        for i, line in enumerate(lines, start=1):
            current.append(line)
            current_len += len(line)

            if current_len >= _CHUNK_SIZE:
                segment = "".join(current)
                chunks.append(
                    Chunk(
                        chunk_id=self._hash(file_path + segment),
                        file_path=file_path,
                        start_line=start_line,
                        end_line=i,
                        content=segment,
                        language=language,
                    )
                )
                # Overlap: keep last N chars worth of lines
                overlap_lines: List[str] = []
                overlap_len = 0
                for ol in reversed(current):
                    if overlap_len + len(ol) > _CHUNK_OVERLAP:
                        break
                    overlap_lines.insert(0, ol)
                    overlap_len += len(ol)
                current = overlap_lines
                current_len = overlap_len
                start_line = i - len(overlap_lines) + 1

        if current:
            segment = "".join(current)
            if len(segment.strip()) >= 10:
                chunks.append(
                    Chunk(
                        chunk_id=self._hash(file_path + segment),
                        file_path=file_path,
                        start_line=start_line,
                        end_line=len(lines),
                        content=segment,
                        language=language,
                    )
                )

        return chunks

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings for a list of chunks using batched inference.

        Uses EmbeddingClient.embed_batch() which fans out to sentence-transformers
        (single forward pass) or parallel Ollama calls, whichever is faster.

        Args:
            chunks: List of Chunk objects to embed.

        Returns:
            numpy float32 array of shape (len(chunks), D).
        """
        if not chunks:
            return np.empty((0,), dtype=np.float32)

        texts = [c.content for c in chunks]
        try:
            from models.embedding_client import EmbeddingClient
            ec = EmbeddingClient(
                model=self.embedding_model,
                base_url=_OLLAMA_BASE_URL,
            )
            matrix = ec.embed_batch_numpy(texts)
            if matrix.ndim == 2 and matrix.shape[0] == len(chunks):
                return matrix
        except Exception:
            pass

        # Fallback: sequential single-text embedding
        vecs: List[np.ndarray] = []
        for chunk in chunks:
            vec = self._embed_text(chunk.content)
            if vec is None:
                dim = vecs[0].shape[0] if vecs else 768
                vec = np.zeros(dim, dtype=np.float32)
            vecs.append(vec)

        return np.vstack(vecs).astype(np.float32) if vecs else np.empty((0,), dtype=np.float32)

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Call the Ollama embeddings API for a single text string.

        Results are cached in ``~/.sentinel/cache/embeddings/`` keyed on
        SHA-256(text) so that unchanged chunks are never re-embedded.

        Args:
            text: Text to embed (truncated to 2000 chars for safety).

        Returns:
            1D float32 numpy array, or None on failure.
        """
        truncated = text[:2000]

        # ── Cache read ────────────────────────────────────────────────
        try:
            from context.context_cache import cache as _ctx_cache
            cached_vec = _ctx_cache.get_embedding(truncated)
            if cached_vec is not None:
                return cached_vec
        except Exception:
            _ctx_cache = None  # cache unavailable — proceed without it
        else:
            _ctx_cache = _ctx_cache  # keep reference for write below

        # ── Embed via Ollama ──────────────────────────────────────────
        payload = {
            "model": self.embedding_model,
            "prompt": truncated,
        }
        try:
            response = requests.post(
                _EMBED_ENDPOINT,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            vec = np.array(data["embedding"], dtype=np.float32)
        except (requests.RequestException, KeyError, ValueError):
            return None

        # ── Cache write ───────────────────────────────────────────────
        try:
            from context.context_cache import cache as _ctx_cache2
            _ctx_cache2.set_embedding(truncated, vec)
        except Exception:
            pass

        return vec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_files(self) -> List[Path]:
        """Walk the project root and collect indexable files.

        Returns:
            List of Path objects for eligible files.
        """
        files: List[Path] = []
        skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".sentinel"}
        for path in self.project_root.rglob("*"):
            if any(part in skip_dirs for part in path.parts):
                continue
            if path.is_file() and path.suffix in _INDEXABLE_EXTENSIONS:
                files.append(path)
        return files

    @staticmethod
    def _detect_language(path: Path) -> str:
        """Map a file extension to a language identifier.

        Args:
            path: File path.

        Returns:
            Language string.
        """
        mapping = {
            ".py": "python", ".md": "markdown", ".txt": "text",
            ".js": "javascript", ".ts": "typescript", ".go": "go",
            ".rs": "rust", ".java": "java", ".cpp": "cpp",
            ".c": "c", ".h": "c",
        }
        return mapping.get(path.suffix.lower(), "text")

    # ------------------------------------------------------------------
    # ANN search (FAISS when available, exact cosine fallback)
    # ------------------------------------------------------------------

    def _ann_search(self, query: np.ndarray, top_k: int) -> np.ndarray:
        """Return indices of top_k nearest vectors using FAISS or exact cosine.

        Builds/rebuilds the FAISS index lazily when the vector count changes.

        Args:
            query:  1D float32 query embedding.
            top_k:  Number of results to return.

        Returns:
            1D integer array of indices into self._vectors.
        """
        if self._vectors is None or len(self._vectors) == 0:
            return np.array([], dtype=np.int64)

        # Lazy FAISS index build / rebuild
        n = len(self._vectors)
        if not hasattr(self, "_faiss_index") or self._faiss_index is None:
            self._faiss_index_size = 0

        if (
            not hasattr(self, "_faiss_index")
            or self._faiss_index is None
            or getattr(self, "_faiss_index_size", 0) != n
        ):
            try:
                from models.inference_engine import FAISSIndex
                self._faiss_index = FAISSIndex(dim=self._vectors.shape[1])
                self._faiss_index.build(self._vectors)
                self._faiss_index_size = n
            except Exception:
                self._faiss_index = None
                self._faiss_index_size = n

        if self._faiss_index is not None and self._faiss_index.is_built:
            try:
                return self._faiss_index.search(query, top_k)
            except Exception:
                pass

        # Exact cosine fallback
        scores = self._cosine_similarity(query, self._vectors)
        return np.argsort(scores)[::-1][:top_k]

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query vector and a matrix.

        Args:
            query: 1D array of shape (D,).
            matrix: 2D array of shape (N, D).

        Returns:
            1D array of shape (N,) with similarity scores.
        """
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        normed = matrix / norms
        return normed @ query_norm

    def _project_slug(self) -> str:
        """Return a filesystem-safe identifier for this project.

        Returns:
            Slug string derived from the project root path.
        """
        return hashlib.sha1(str(self.project_root).encode()).hexdigest()[:12]

    def _chunks_file(self) -> Path:
        return self.index_dir / "chunks.json"

    def _vectors_file(self) -> Path:
        return self.index_dir / "vectors.npy"

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
