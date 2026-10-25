"""Hand-written stand-ins for the services' parts: each keeps what it was handed."""

import hashlib
import itertools
import threading
from collections import defaultdict
from typing import override

import numpy
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.domain.interfaces import Embedder, Vector, VectorMatrix, VectorStore
from semantic_code_mcp.domain.models import Chunk, SearchResult, StoreStatistics

EMBEDDING_DIMENSION = 384
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

    def __init__(self, *, loaded: bool = True) -> None:
        self.embedded_texts: list[str] = []
        self.calling_threads: list[threading.Thread] = []
        self._loaded = loaded

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
        return vector_of(text)

    @override
    def embed_documents(self, texts: list[str]) -> VectorMatrix:
        self.calling_threads.append(threading.current_thread())
        self._loaded = True
        self.embedded_texts.extend(texts)
        return numpy.stack([vector_of(text) for text in texts])


class FakeVectorStore(VectorStore):
    """A store that answers every search with the hits it was given and keeps what is written."""

    def __init__(self, hits: list[SearchResult]) -> None:
        self.hits = hits
        self.chunks_by_file: dict[str, list[Chunk]] = defaultdict(list)
        self.deleted_files: list[str] = []

    @override
    def add_chunks(self, chunks: list[Chunk], vectors: VectorMatrix) -> None:
        for chunk in chunks:
            self.chunks_by_file[chunk.file_path].append(chunk)

    @override
    def search_hybrid(
        self,
        query_vector: Vector,
        query_text: str,
        limit: int,
        vector_weight: float,
    ) -> list[SearchResult]:
        return self.hits[:limit]

    @override
    def delete_by_files(self, file_paths: list[str]) -> None:
        self.deleted_files.extend(file_paths)
        for file_path in file_paths:
            self.chunks_by_file.pop(file_path, None)

    @override
    def statistics(self) -> StoreStatistics:
        return StoreStatistics(
            indexed_files=sorted(self.chunks_by_file),
            chunks_count=sum(len(chunks) for chunks in self.chunks_by_file.values()),
        )

    @override
    def clear(self) -> None:
        self.chunks_by_file.clear()


class HeldWriteStore(FakeVectorStore):
    """A store whose first chunk write is held until a later delete, or until a wait ends.

    It stands in for a write still running in a worker thread after the request that
    started it was cancelled. The order is fixed without branches: each call takes the
    next event of its own sequence, and only the first write and the second delete
    take a live one.
    """

    def __init__(self, *, longest_hold_seconds: float) -> None:
        super().__init__(hits=[])
        self.first_write_started = threading.Event()
        self.first_write_landed = threading.Event()
        self._longest_hold_seconds = longest_hold_seconds
        first_write_released = threading.Event()
        always_open = threading.Event()
        always_open.set()
        unused = threading.Event()
        self._write_gates = itertools.chain([first_write_released], itertools.repeat(always_open))
        self._write_marks = itertools.chain([self.first_write_landed], itertools.repeat(unused))
        self._releases_on_delete = itertools.chain(
            [unused, first_write_released], itertools.repeat(unused)
        )

    @override
    def add_chunks(self, chunks: list[Chunk], vectors: VectorMatrix) -> None:
        gate = next(self._write_gates)
        landed = next(self._write_marks)
        self.first_write_started.set()
        gate.wait(self._longest_hold_seconds)
        super().add_chunks(chunks, vectors)
        landed.set()

    @override
    def delete_by_files(self, file_paths: list[str]) -> None:
        super().delete_by_files(file_paths)
        next(self._releases_on_delete).set()


def vector_of(text: str) -> Vector:
    """A fixed vector for a text, its components in [-1, 1]."""
    digest = hashlib.sha256(text.encode()).digest()
    repeated = (digest * (EMBEDDING_DIMENSION // len(digest) + 1))[:EMBEDDING_DIMENSION]
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
