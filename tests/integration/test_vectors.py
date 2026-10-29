"""Tests of giving an index's chunks the model's vectors, as a search by meaning does before
it searches: what is embedded, when the model loads, and what an edit, a forced run or a
model of another width leaves to embed."""

import asyncio
import threading
from pathlib import Path
from typing import override

import numpy
import pytest

from semantic_code_mcp.app import build_chunker, files_opener
from semantic_code_mcp.config import Settings
from semantic_code_mcp.domain.interfaces import VectorMatrix
from semantic_code_mcp.domain.models import VectorIdentity
from semantic_code_mcp.services.indexing import IndexingService
from semantic_code_mcp.services.progress import ProgressReporter
from semantic_code_mcp.services.projects import vector_identity
from semantic_code_mcp.services.vectors import EMBEDDING_SLICE_SIZE
from tests.fakes import (
    EMBEDDING_DIMENSION,
    FAKE_MODEL,
    DelegatingStore,
    EmbedderWatchingRecorder,
    FakeEmbedder,
    FakeIndexStore,
    vector_of,
)
from tests.helpers import discovery_for, indexing_service, vector_service
from tests.indexing import ADDED_FUNCTION, IDENTITY, NO_PROGRESS, touch_later
from tests.stores import NO_CANDIDATES, StoreClass, open_store

# Functions in one file that fill two slices of chunks to embed, and a part of a third
TWO_AND_A_PART_SLICES = 2 * EMBEDDING_SLICE_SIZE + 10
# A vector width other than the fake model's
WIDE = 16
# A revision a model's weights are pinned at
PINNED_REVISION = "pinned-revision"


class StoreCancelledAfterItsVectors(DelegatingStore):
    """Another store, whose search by meaning is cancelled as soon as an addition of vectors
    lands: a client that hung up while the project embedded."""

    @override
    def add_vectors(
        self, model: VectorIdentity, chunk_ids: list[str], vectors: VectorMatrix
    ) -> None:
        super().add_vectors(model, chunk_ids, vectors)
        raise asyncio.CancelledError


def many_functions_project(root: Path, functions: int) -> Path:
    """A project of one file holding a number of small functions, one chunk each."""
    project = root / "many"
    project.mkdir()
    (project / "many.py").write_text(
        "".join(
            f"def function_{number}():\n    return {number}\n\n\n" for number in range(functions)
        )
    )
    return project


def add_a_function(path: Path) -> None:
    path.write_text(path.read_text() + ADDED_FUNCTION)
    touch_later(path)


async def test_every_chunk_is_embedded_once_with_its_own_vector_across_slices(
    test_settings: Settings, tmp_path: Path
):
    project = many_functions_project(tmp_path, TWO_AND_A_PART_SLICES)
    store = FakeIndexStore(NO_CANDIDATES)
    await indexing_service(store, test_settings, build_chunker()).index(
        project, force=False, progress=NO_PROGRESS
    )

    report = await vector_service(store, FakeEmbedder()).embed_missing(progress=NO_PROGRESS)

    chunks = store.stored_chunks()
    stored = numpy.stack([store.vectors_by_model[FAKE_MODEL][chunk.chunk_id] for chunk in chunks])
    expected = numpy.stack([vector_of(chunk.content, EMBEDDING_DIMENSION) for chunk in chunks])
    assert (report.chunks_embedded, numpy.array_equal(stored, expected)) == (len(chunks), True)


async def test_the_model_loads_and_embeds_off_the_event_loop(
    test_settings: Settings, sample_project: Path
):
    store = FakeIndexStore(NO_CANDIDATES)
    await indexing_service(store, test_settings, build_chunker()).index(
        sample_project, force=False, progress=NO_PROGRESS
    )
    embedder = FakeEmbedder(loaded=False)

    await vector_service(store, embedder).embed_missing(progress=NO_PROGRESS)

    assert embedder.calling_threads
    assert threading.main_thread() not in embedder.calling_threads


@pytest.mark.parametrize(
    ("functions", "embedded_at_reports"),
    [
        pytest.param(
            TWO_AND_A_PART_SLICES,
            [0, 0, EMBEDDING_SLICE_SIZE, 2 * EMBEDDING_SLICE_SIZE],
            id="the last slice in part",
        ),
        pytest.param(2 * EMBEDDING_SLICE_SIZE, [0, 0, EMBEDDING_SLICE_SIZE], id="whole slices"),
    ],
)
async def test_the_model_load_and_each_slice_are_reported_before_they_happen(
    test_settings: Settings, tmp_path: Path, functions: int, embedded_at_reports: list[int]
):
    """A report as the model loads, then one before each slice is embedded."""
    project = many_functions_project(tmp_path, functions)
    store = FakeIndexStore(NO_CANDIDATES)
    await indexing_service(store, test_settings, build_chunker()).index(
        project, force=False, progress=NO_PROGRESS
    )
    embedder = FakeEmbedder(loaded=False)
    recorder = EmbedderWatchingRecorder(embedder)

    await vector_service(store, embedder).embed_missing(progress=ProgressReporter(recorder))

    assert [
        [snapshot.texts_embedded for snapshot in recorder.snapshots],
        [snapshot.model_loaded for snapshot in recorder.snapshots[:2]],
    ] == [embedded_at_reports, [False, True]]


async def test_an_index_with_every_chunk_embedded_loads_no_model(
    test_settings: Settings, sample_project: Path
):
    store = FakeIndexStore(NO_CANDIDATES)
    await indexing_service(store, test_settings, build_chunker()).index(
        sample_project, force=False, progress=NO_PROGRESS
    )
    await vector_service(store, FakeEmbedder()).embed_missing(progress=NO_PROGRESS)
    unloaded = FakeEmbedder(loaded=False)

    report = await vector_service(store, unloaded).embed_missing(progress=NO_PROGRESS)

    assert (report.chunks_embedded, unloaded.is_loaded) == (0, False)


async def test_an_edited_file_embeds_only_its_chunks_whose_text_changed(
    test_settings: Settings, sample_project: Path, tmp_path: Path, store_class: StoreClass
):
    store = open_store(store_class, tmp_path / "index", IDENTITY)
    indexing = indexing_service(store, test_settings, build_chunker())
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    await vector_service(store, FakeEmbedder()).embed_missing(progress=NO_PROGRESS)
    add_a_function(sample_project / "main.py")
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    embedder = FakeEmbedder()

    await vector_service(store, embedder).embed_missing(progress=NO_PROGRESS)

    assert embedder.embedded_texts == [ADDED_FUNCTION.strip()]


async def test_a_forced_run_leaves_every_chunk_of_an_unpinned_model_to_embed_again(
    test_settings: Settings, sample_project: Path, tmp_path: Path, store_class: StoreClass
):
    """A model replaced under a name nothing pins gives other vectors that nothing else
    can see."""
    store = open_store(store_class, tmp_path / "index", IDENTITY)
    indexing = indexing_service(store, test_settings, build_chunker())
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    await vector_service(store, FakeEmbedder()).embed_missing(progress=NO_PROGRESS)
    await indexing.index(sample_project, force=True, progress=NO_PROGRESS)
    embedder = FakeEmbedder()

    await vector_service(store, embedder).embed_missing(progress=NO_PROGRESS)

    assert len(embedder.embedded_texts) == 6


async def test_a_forced_run_keeps_the_vectors_of_a_model_whose_weights_are_pinned(
    test_settings: Settings, sample_project: Path, tmp_path: Path, store_class: StoreClass
):
    """A pinned revision vouches for the vectors, which on a large repository took an hour
    to embed."""
    store = open_store(store_class, tmp_path / "index", IDENTITY)
    pinned = FakeEmbedder(revision=PINNED_REVISION)
    indexing = IndexingService(
        files=files_opener(test_settings, discovery_for(test_settings, build_chunker()))(),
        chunker=build_chunker(),
        store=store,
        model=vector_identity(pinned),
    )
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    await vector_service(store, pinned).embed_missing(progress=NO_PROGRESS)
    await indexing.index(sample_project, force=True, progress=NO_PROGRESS)
    embedder = FakeEmbedder(revision=PINNED_REVISION)

    await vector_service(store, embedder).embed_missing(progress=NO_PROGRESS)

    assert embedder.embedded_texts == []


async def test_after_a_model_of_another_width_took_its_name_every_chunk_is_embedded_anew(
    test_settings: Settings, sample_project: Path, tmp_path: Path, store_class: StoreClass
):
    """A name nothing pins, as a local path's: the edited chunk's wider vector takes the place
    of the model's old ones, so the next search by meaning embeds the rest."""
    store = open_store(store_class, tmp_path / "index", IDENTITY)
    indexing = indexing_service(store, test_settings, build_chunker())
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    await vector_service(store, FakeEmbedder()).embed_missing(progress=NO_PROGRESS)
    add_a_function(sample_project / "main.py")
    await indexing.index(sample_project, force=False, progress=NO_PROGRESS)
    wider = FakeEmbedder(dimension=WIDE)

    first = await vector_service(store, wider).embed_missing(progress=NO_PROGRESS)
    second = await vector_service(store, wider).embed_missing(progress=NO_PROGRESS)

    assert (
        first.chunks_embedded,
        second.chunks_embedded,
        store.counts(FAKE_MODEL).chunks_with_vectors,
    ) == (1, 6, 7)


async def test_a_search_by_meaning_cancelled_midway_keeps_the_slices_it_added(
    test_settings: Settings, tmp_path: Path, store_class: StoreClass
):
    """The next search by meaning embeds only the rest."""
    project = many_functions_project(tmp_path, TWO_AND_A_PART_SLICES)
    store = open_store(store_class, tmp_path / "index", IDENTITY)
    await indexing_service(store, test_settings, build_chunker()).index(
        project, force=False, progress=NO_PROGRESS
    )

    with pytest.raises(asyncio.CancelledError):
        await vector_service(StoreCancelledAfterItsVectors(store), FakeEmbedder()).embed_missing(
            progress=NO_PROGRESS
        )

    assert store.counts(FAKE_MODEL).chunks_with_vectors == EMBEDDING_SLICE_SIZE
