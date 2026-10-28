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
    IndexedFiles,
    IndexIdentity,
    IndexManifest,
    IndexUpdate,
    SearchCandidates,
    SearchQuery,
    SearchScope,
    SourceFile,
    StoreKind,
    VectorIdentity,
)

EMBEDDING_DIMENSION = 384
FAKE_MODEL_NAME = "fake-model"
FAKE_CHUNKER_VERSION = 1
FAKE_EMBEDDER_VERSION = 1
FAKE_SCHEMA_VERSION = 1
BYTE_RANGE = 255
# A digest byte scaled into [-1, 1]
VECTOR_SPAN = 2
VECTOR_OFFSET = 1
# What makes the fake embedder's vectors
FAKE_MODEL = VectorIdentity(
    model_name=FAKE_MODEL_NAME, model_revision=None, embedder_version=FAKE_EMBEDDER_VERSION
)


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
    """A store that answers every search with the candidates it was given, by words with the
    chunks of its keyword candidates and by name with any candidate of that name, and keeps
    what is written: each file's chunks and record, and each model's vectors by chunk id.
    Once written, its manifest names the fake chunker; it has no full-text index to keep up."""

    def __init__(self, candidates: SearchCandidates) -> None:
        self.candidates = candidates
        self.searched: list[SearchQuery] = []
        self.chunks_by_file: dict[Path, list[Chunk]] = defaultdict(list)
        self.vectors_by_model: dict[VectorIdentity, dict[str, Vector]] = defaultdict(dict)
        self.files_by_path: dict[Path, SourceFile] = {}
        self.written_manifest: IndexManifest | None = None
        self.updates: list[IndexUpdate] = []

    @override
    def manifest(self) -> IndexManifest | None:
        return self.written_manifest

    @override
    def try_hold_for_run(self) -> bool:
        # One fake serves one session's runs, which its service's own lock keeps apart
        return True

    @override
    def release_run(self) -> None:
        pass

    @override
    def run_held(self) -> bool:
        # Its runs are its own service's, which no status of another session reads
        return False

    @override
    def indexed_files(self) -> IndexedFiles:
        # Its write records chunks and files together: no chunk is left without its record
        return IndexedFiles(records=list(self.files_by_path.values()), unrecorded_paths=[])

    @override
    def counts(self, model: VectorIdentity) -> IndexCounts:
        return IndexCounts(
            files=len(self.files_by_path),
            chunks=len(self.stored_chunks()),
            chunks_with_vectors=len(self.vectors_by_model[model]),
        )

    @override
    def write(self, update: IndexUpdate) -> None:
        self.updates.append(update)
        replaced = [*(file.path for file in update.files), *update.removed_paths]
        kept_ids = {chunk.chunk_id for chunk in update.chunks}
        for path in replaced:
            dropped_ids = {chunk.chunk_id for chunk in self.chunks_by_file.pop(path, [])} - kept_ids
            for vectors in self.vectors_by_model.values():
                for chunk_id in dropped_ids:
                    vectors.pop(chunk_id, None)
            self.files_by_path.pop(path, None)
        for chunk in update.chunks:
            self.chunks_by_file[chunk.path].append(chunk)
        self.files_by_path.update({file.path: file for file in update.files})
        self.written_manifest = IndexManifest(
            identity=IndexIdentity(chunker_version=FAKE_CHUNKER_VERSION),
            store_kind=StoreKind.sqlite,
            schema_version=FAKE_SCHEMA_VERSION,
            indexed_at=datetime.now(UTC),
        )

    @override
    def chunks_without_vectors(self, model: VectorIdentity) -> list[Chunk]:
        pending = {chunk.chunk_id: chunk for chunk in self.stored_chunks()}
        for chunk_id in self.vectors_by_model[model]:
            pending.pop(chunk_id, None)
        return list(pending.values())

    @override
    def add_vectors(
        self, model: VectorIdentity, chunk_ids: list[str], vectors: VectorMatrix
    ) -> None:
        given = dict(zip(chunk_ids, vectors, strict=True))
        held = {chunk.chunk_id for chunk in self.stored_chunks()}
        self.vectors_by_model[model].update(
            {chunk_id: given[chunk_id] for chunk_id in given.keys() & held}
        )

    @override
    def remove_vectors(self, model: VectorIdentity) -> None:
        self.vectors_by_model.pop(model, None)

    @override
    def search_words(self, query: SearchQuery, limit: int) -> list[Chunk]:
        self.searched.append(query)
        return [candidate.chunk for candidate in self.candidates.by_keyword[:limit]]

    @override
    def chunks_named(self, folded_name: str, scope: SearchScope) -> list[Chunk]:
        chunks_by_name: dict[str, dict[str, Chunk]] = defaultdict(dict)
        for candidate in [*self.candidates.by_vector, *self.candidates.by_keyword]:
            chunk = candidate.chunk
            chunks_by_name[chunk.name.casefold()][chunk.chunk_id] = chunk
        return list(chunks_by_name[folded_name].values())

    @override
    def search(
        self, model: VectorIdentity, query_vector: Vector, query: SearchQuery, per_channel: int
    ) -> SearchCandidates:
        self.searched.append(query)
        return SearchCandidates(
            by_vector=self.candidates.by_vector[:per_channel],
            by_keyword=self.candidates.by_keyword[:per_channel],
        )

    def stored_chunks(self) -> list[Chunk]:
        """Every chunk written, file by file in the order the files came."""
        return [chunk for chunks in self.chunks_by_file.values() for chunk in chunks]


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


class DelegatingStore(IndexStore):
    """Another store, every call handed on to it; a subclass changes what it means to."""

    def __init__(self, inner: IndexStore) -> None:
        self._inner = inner

    @override
    def manifest(self) -> IndexManifest | None:
        return self._inner.manifest()

    @override
    def try_hold_for_run(self) -> bool:
        return self._inner.try_hold_for_run()

    @override
    def release_run(self) -> None:
        self._inner.release_run()

    @override
    def run_held(self) -> bool:
        return self._inner.run_held()

    @override
    def indexed_files(self) -> IndexedFiles:
        return self._inner.indexed_files()

    @override
    def counts(self, model: VectorIdentity) -> IndexCounts:
        return self._inner.counts(model)

    @override
    def write(self, update: IndexUpdate) -> None:
        self._inner.write(update)

    @override
    def chunks_without_vectors(self, model: VectorIdentity) -> list[Chunk]:
        return self._inner.chunks_without_vectors(model)

    @override
    def add_vectors(
        self, model: VectorIdentity, chunk_ids: list[str], vectors: VectorMatrix
    ) -> None:
        self._inner.add_vectors(model, chunk_ids, vectors)

    @override
    def remove_vectors(self, model: VectorIdentity) -> None:
        self._inner.remove_vectors(model)

    @override
    def search_words(self, query: SearchQuery, limit: int) -> list[Chunk]:
        return self._inner.search_words(query, limit)

    @override
    def chunks_named(self, folded_name: str, scope: SearchScope) -> list[Chunk]:
        return self._inner.chunks_named(folded_name, scope)

    @override
    def search(
        self, model: VectorIdentity, query_vector: Vector, query: SearchQuery, per_channel: int
    ) -> SearchCandidates:
        return self._inner.search(model, query_vector, query, per_channel)


class StoreCancelledAfterItsWrite(DelegatingStore):
    """Another store, whose run is cancelled as soon as a write of it lands: a client that
    hung up while the run was under way."""

    @override
    def write(self, update: IndexUpdate) -> None:
        super().write(update)
        raise asyncio.CancelledError


class StoreHeldInItsWrite(DelegatingStore):
    """Another store, whose write waits to be released once it has begun, then lands in its
    thread: the test cancels the run in between, as a client that hangs up mid-write does."""

    def __init__(self, inner: IndexStore) -> None:
        super().__init__(inner)
        self.writing = threading.Event()
        self.release = threading.Event()
        self.written = threading.Event()

    @override
    def write(self, update: IndexUpdate) -> None:
        self.writing.set()
        self.release.wait()
        super().write(update)
        self.written.set()


class ChunkCountRecorder:
    """A progress callback that notes, at each report, how many chunks a store holds."""

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        self.chunks_at_reports: list[int] = []

    async def __call__(self, _progress: float, _total: float, _message: str) -> None:
        self.chunks_at_reports.append(self._store.counts(FAKE_MODEL).chunks)


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
