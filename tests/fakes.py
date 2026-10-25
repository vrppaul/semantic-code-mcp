"""Hand-written stand-ins for the services' parts: each keeps what it was handed."""

import hashlib
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import override

import numpy
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import Chunker, Embedder, IndexStore, Vector, VectorMatrix
from semantic_code_mcp.domain.models import (
    Chunk,
    IndexCounts,
    IndexManifest,
    IndexUpdate,
    SearchResult,
    SourceFile,
    WriteScope,
)

EMBEDDING_DIMENSION = 384
FAKE_MODEL_NAME = "fake-model"
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
        dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        self.embedded_texts: list[str] = []
        self.calling_threads: list[threading.Thread] = []
        self._loaded = loaded
        self._model_name = model_name
        self._dimension = dimension

    @property
    @override
    def model_name(self) -> str:
        return self._model_name

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
    """A store that answers every search with the hits it was given and keeps what is
    written: each chunk with its vector, each file's record, the manifest."""

    def __init__(self, hits: list[SearchResult]) -> None:
        self.hits = hits
        self.chunks_by_file: dict[Path, list[Chunk]] = defaultdict(list)
        self.vectors_by_file: dict[Path, list[Vector]] = defaultdict(list)
        self.files_by_path: dict[Path, SourceFile] = {}
        self.written_manifest: IndexManifest | None = None

    @property
    @override
    def schema_version(self) -> int:
        return FAKE_SCHEMA_VERSION

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
    def write(self, update: IndexUpdate, vectors: VectorMatrix) -> None:
        replaced = {
            WriteScope.everything: [*self.chunks_by_file, *self.files_by_path],
            WriteScope.changed_files: [
                *(file.path for file in update.files),
                *update.removed_paths,
            ],
        }[update.scope]
        for path in replaced:
            self.chunks_by_file.pop(path, None)
            self.vectors_by_file.pop(path, None)
            self.files_by_path.pop(path, None)
        for chunk, vector in zip(update.chunks, vectors, strict=True):
            self.chunks_by_file[Path(chunk.file_path)].append(chunk)
            self.vectors_by_file[Path(chunk.file_path)].append(vector)
        self.files_by_path.update({file.path: file for file in update.files})
        self.written_manifest = update.manifest

    @override
    def search_hybrid(
        self,
        query_vector: Vector,
        query_text: str,
        limit: int,
        vector_weight: float,
    ) -> list[SearchResult]:
        return self.hits[:limit]


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
    def chunk_file(self, file_path: str) -> list[Chunk]:
        return self._inner.chunk_file(file_path)


class ChangingChunker(VersionedChunker):
    """A chunker that changes the project each time it reads a file: an edit or a deletion
    made while an index run is under way, after the scan took the files' records."""

    def __init__(self, inner: Chunker, change: Callable[[], object]) -> None:
        super().__init__(inner, inner.version)
        self._change = change

    @override
    def chunk_file(self, file_path: str) -> list[Chunk]:
        self._change()
        return self._inner.chunk_file(file_path)


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
