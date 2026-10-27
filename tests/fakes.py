"""Hand-written stand-ins for the services' parts: each keeps what it was handed."""

import asyncio
import hashlib
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import override

import numpy
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import Chunker, Embedder, IndexStore, Vector, VectorMatrix
from semantic_code_mcp.domain.models import (
    Chunk,
    IndexCounts,
    IndexIdentity,
    IndexManifest,
    IndexUpdate,
    SearchCandidates,
    SearchQuery,
    SourceFile,
)
from semantic_code_mcp.storage.lancedb import LanceDBIndexStore

EMBEDDING_DIMENSION = 384
FAKE_MODEL_NAME = "fake-model"
FAKE_CHUNKER_VERSION = 1
FAKE_EMBEDDER_VERSION = 1
FAKE_SCHEMA_VERSION = 1
BYTE_RANGE = 255
# A digest byte scaled into [-1, 1]
VECTOR_SPAN = 2
VECTOR_OFFSET = 1


class FakeEmbedder(Embedder):
    """Vectors without a model: each text maps to a fixed vector derived from its hash.

    It starts loaded unless told otherwise; one that starts unloaded loads on `load()`
    or on its first text, as the real one does. It keeps the thread of every call, so a
    test can tell whether the model's work ran on the event loop.
    """

    def __init__(
        self,
        *,
        loaded: bool = True,
        model_name: str = FAKE_MODEL_NAME,
        revision: str | None = None,
        version: int = FAKE_EMBEDDER_VERSION,
        dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        self.embedded_texts: list[str] = []
        self.calling_threads: list[threading.Thread] = []
        self._loaded = loaded
        self._model_name = model_name
        self._revision = revision
        self._version = version
        self._dimension = dimension

    @property
    @override
    def model_name(self) -> str:
        return self._model_name

    @property
    @override
    def revision(self) -> str | None:
        return self._revision

    @property
    @override
    def version(self) -> int:
        return self._version

    @property
    @override
    def is_loaded(self) -> bool:
        return self._loaded

    @override
    def load(self) -> None:
        self.calling_threads.append(threading.current_thread())
        self._loaded = True

    @override
    def embed_query(self, text: str) -> Vector:
        self.calling_threads.append(threading.current_thread())
        self._loaded = True
        self.embedded_texts.append(text)
        return vector_of(text, self._dimension)

    @override
    def embed_documents(self, texts: list[str]) -> VectorMatrix:
        self.calling_threads.append(threading.current_thread())
        self._loaded = True
        self.embedded_texts.extend(texts)
        return numpy.stack([vector_of(text, self._dimension) for text in texts])


class FakeIndexStore(IndexStore):
    """A store that answers every search with the candidates it was given and keeps what
    is written: each chunk with its vector, each file's record. Once written, its manifest
    names the fake model; it has no full-text index to keep up."""

    def __init__(self, candidates: SearchCandidates) -> None:
        self.candidates = candidates
        self.searched: list[SearchQuery] = []
        self.chunks_by_file: dict[Path, list[Chunk]] = defaultdict(list)
        self.vectors_by_file: dict[Path, list[Vector]] = defaultdict(list)
        self.files_by_path: dict[Path, SourceFile] = {}
        self.written_manifest: IndexManifest | None = None

    @override
    def manifest(self) -> IndexManifest | None:
        return self.written_manifest

    @override
    def indexed_files(self) -> list[SourceFile]:
        return list(self.files_by_path.values())

    @override
    def counts(self) -> IndexCounts:
        return IndexCounts(
            files=len(self.files_by_path),
            chunks=sum(len(chunks) for chunks in self.chunks_by_file.values()),
        )

    @override
    def chunked_paths(self) -> set[Path]:
        return set(self.chunks_by_file)

    @override
    def stored_vectors(self, chunk_ids: list[str]) -> dict[str, Vector]:
        stored = {
            chunk.chunk_id: vector
            for path, chunks in self.chunks_by_file.items()
            for chunk, vector in zip(chunks, self.vectors_by_file[path], strict=True)
        }
        return {chunk_id: stored[chunk_id] for chunk_id in stored.keys() & set(chunk_ids)}

    @override
    def keep_up(self) -> None:
        return

    @override
    def write(self, update: IndexUpdate, vectors: VectorMatrix) -> None:
        replaced = [*(file.path for file in update.files), *update.removed_paths]
        for path in replaced:
            self.chunks_by_file.pop(path, None)
            self.vectors_by_file.pop(path, None)
            self.files_by_path.pop(path, None)
        for chunk, vector in zip(update.chunks, vectors, strict=True):
            self.chunks_by_file[chunk.path].append(chunk)
            self.vectors_by_file[chunk.path].append(vector)
        self.files_by_path.update({file.path: file for file in update.files})
        self.written_manifest = IndexManifest(
            identity=IndexIdentity(
                model_name=FAKE_MODEL_NAME,
                model_revision=None,
                embedder_version=FAKE_EMBEDDER_VERSION,
                chunker_version=FAKE_CHUNKER_VERSION,
            ),
            schema_version=FAKE_SCHEMA_VERSION,
            indexed_at=datetime.now(UTC),
        )

    @override
    def search_candidates(
        self, query_vector: Vector, query: SearchQuery, per_channel: int
    ) -> SearchCandidates:
        self.searched.append(query)
        return SearchCandidates(
            by_vector=self.candidates.by_vector[:per_channel],
            by_keyword=self.candidates.by_keyword[:per_channel],
        )


class VersionedChunker(Chunker):
    """A chunker that chunks as another does but says it is of another version."""

    def __init__(self, inner: Chunker, version: int) -> None:
        self._inner = inner
        self._version = version

    @property
    @override
    def supported_extensions(self) -> list[str]:
        return self._inner.supported_extensions

    @property
    @override
    def version(self) -> int:
        return self._version

    @override
    def chunk_file(self, project_root: Path, path: Path) -> list[Chunk]:
        return self._inner.chunk_file(project_root, path)


class ChangingChunker(VersionedChunker):
    """A chunker that changes the project each time it reads a file: an edit or a deletion
    made while an index run is under way, after the scan took the files' records."""

    def __init__(self, inner: Chunker, change: Callable[[], object]) -> None:
        super().__init__(inner, inner.version)
        self._change = change

    @override
    def chunk_file(self, project_root: Path, path: Path) -> list[Chunk]:
        self._change()
        return self._inner.chunk_file(project_root, path)


class StoreCancelledAfterItsWrite(LanceDBIndexStore):
    """The real store, whose run is cancelled as soon as a write of it lands: a client that
    hung up while the run was under way."""

    @override
    def write(self, update: IndexUpdate, vectors: VectorMatrix) -> None:
        super().write(update, vectors)
        raise asyncio.CancelledError


class ChunkCountRecorder:
    """A progress callback that notes, at each report, how many chunks a store holds."""

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        self.chunks_at_reports: list[int] = []

    async def __call__(self, _progress: float, _total: float, _message: str) -> None:
        self.chunks_at_reports.append(self._store.counts().chunks)


def vector_of(text: str, dimension: int) -> Vector:
    """A fixed vector for a text, its components in [-1, 1]."""
    digest = hashlib.sha256(text.encode()).digest()
    repeated = (digest * (dimension // len(digest) + 1))[:dimension]
    components = numpy.frombuffer(repeated, dtype=numpy.uint8).astype(numpy.float32)
    return components / BYTE_RANGE * VECTOR_SPAN - VECTOR_OFFSET


class ProgressSnapshot(BaseModel):
    """One progress report, and how far the embedder had got when it came."""

    model_config = ConfigDict(frozen=True)

    progress: float
    model_loaded: bool
    texts_embedded: int


class EmbedderWatchingRecorder:
    """A progress callback that notes, at each report, whether the embedder had loaded
    and how many texts it had embedded."""

    def __init__(self, embedder: FakeEmbedder) -> None:
        self._embedder = embedder
        self.snapshots: list[ProgressSnapshot] = []

    async def __call__(self, progress: float, _total: float, _message: str) -> None:
        self.snapshots.append(
            ProgressSnapshot(
                progress=progress,
                model_loaded=self._embedder.is_loaded,
                texts_embedded=len(self._embedder.embedded_texts),
            )
        )


class ProgressRecorder:
    """A progress callback that keeps every progress value and message it is handed."""

    def __init__(self) -> None:
        self.progress_values: list[float] = []
        self.messages: list[str] = []

    async def __call__(self, progress: float, _total: float, message: str) -> None:
        self.progress_values.append(progress)
        self.messages.append(message)
