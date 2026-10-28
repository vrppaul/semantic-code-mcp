"""End-to-end tests: the MCP tools driven through a client, as an agent calls them."""

import asyncio
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest
from mcp import Client
from mcp.types import CallToolResult, ListRootsResult, Root

from semantic_code_mcp.app import create_app
from semantic_code_mcp.config import Settings
from semantic_code_mcp.domain.globs import REFUSED_CHARACTERS
from semantic_code_mcp.domain.models import SearchMode
from semantic_code_mcp.embedding import OnnxEmbedder
from semantic_code_mcp.server.responses import (
    IndexCodebaseResponse,
    IndexStatusResponse,
    SearchCodeResponse,
)
from tests.helpers import make_repository

GREETING_MODULE = '''def greet(name: str) -> str:
    """Greet a person by their name."""
    return f"Hello, {name}!"
'''

CONFIG_MODULE = '''import json


def load_config(path: str) -> dict:
    """Read the application configuration from a JSON file."""
    with open(path) as config_file:
        return json.load(config_file)
'''

CONFIGURATION_GUIDE = """# Reading the configuration

The application reads its configuration from a JSON file when it starts.
"""
# A question the function of config.py and the section of the guide both answer
CONFIGURATION_QUESTION = "read the configuration file"
GUIDE_SECTION = "Reading the configuration"

# An mtime well after any the test wrote, so the change is seen on any file system
LATER_MTIME_SECONDS = 4_000_000_000

RETRY_FUNCTION = '''

def retry_with_backoff(operation, attempts: int):
    """Call an operation again after a growing pause each time it fails."""
    return operation()
'''


# A root a client may name that is no directory on this machine
MISSING_ROOT = "file:///no/such/directory"
ROOT_ON_ANOTHER_MACHINE = "file://another-machine/project"


async def roots_answer(uris: list[str], _context: object) -> ListRootsResult:
    """A client's answer to `roots/list`: the workspace roots at the URIs given."""
    return ListRootsResult(roots=[Root(uri=uri) for uri in uris])


def client_with_roots(settings: Settings, embedder: OnnxEmbedder, uris: list[str]) -> Client:
    """A client of the app whose workspace roots are the URIs given."""
    # "auto" negotiates the newest protocol, as Claude Code does: the roots ride an
    # InputRequiredResult there, not a request back mid-call
    return Client(
        create_app(settings, embedder=embedder),
        mode="auto",
        list_roots_callback=partial(roots_answer, uris),
    )


def client_without_roots(settings: Settings, embedder: OnnxEmbedder) -> Client:
    """A client of the app that declares no workspace roots at all."""
    return Client(create_app(settings, embedder=embedder), mode="auto")


def structured(result: CallToolResult) -> object:
    assert not result.is_error, result.content
    return result.structured_content


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "greeting.py").write_text(GREETING_MODULE)
    (root / "config.py").write_text(CONFIG_MODULE)
    return root


@pytest.fixture
def client(test_settings: Settings, embedder: OnnxEmbedder) -> Client:
    # "legacy" goes through JSON-RPC framing, as a real client does
    return Client(create_app(test_settings, embedder=embedder), mode="legacy")


async def test_only_the_search_tool_asks_to_come_with_the_prompt(client: Client):
    """Claude Code defers a server's tools behind its tool search unless a tool asks
    otherwise; an agent that must look a tool up first spends a turn on it."""
    async with client:
        tools = (await client.list_tools()).tools

    assert {tool.name: tool.meta for tool in tools} == {
        "search_code": {"anthropic/alwaysLoad": True},
        "index_codebase": None,
        "index_status": None,
    }


async def test_search_returns_the_function_the_query_describes(project: Path, client: Client):
    async with client:
        result = await client.call_tool(
            "search_code", {"query": "say hello to someone", "project_path": str(project)}
        )

    first_hit = SearchCodeResponse.model_validate(structured(result)).results[0]
    assert (Path(first_hit.file_path).name, first_hit.name) == ("greeting.py", "greet")


@pytest.fixture
def documented_project(project: Path) -> Path:
    """The project, with a guide whose section answers what config.py's function does."""
    (project / "docs").mkdir()
    (project / "docs" / "configuration.md").write_text(CONFIGURATION_GUIDE)
    return project


@pytest.mark.parametrize(
    ("narrowing", "names"),
    [
        pytest.param({"chunk_kinds": ["section"]}, [GUIDE_SECTION], id="to sections"),
        pytest.param({"languages": ["python"]}, ["load_config"], id="to Python"),
        pytest.param({"path_glob": "docs"}, [GUIDE_SECTION], id="to a directory"),
        pytest.param({"exclude_glob": "docs"}, ["load_config"], id="leaving a directory out"),
        pytest.param(
            {"path_glob": ""}, ["load_config", GUIDE_SECTION], id="by a blank glob, not at all"
        ),
    ],
)
async def test_a_narrowed_search_keeps_to_what_it_names(
    documented_project: Path, client: Client, narrowing: dict[str, object], names: list[str]
):
    async with client:
        result = await client.call_tool(
            "search_code",
            {
                "query": CONFIGURATION_QUESTION,
                "project_path": str(documented_project),
                **narrowing,
            },
        )

    hits = SearchCodeResponse.model_validate(structured(result)).results
    assert sorted(hit.name for hit in hits) == sorted(names)


async def test_search_sends_debug_details_only_when_asked(project: Path, client: Client):
    async with client:
        plain = await client.call_tool(
            "search_code", {"query": "greeting", "project_path": str(project)}
        )
        detailed = await client.call_tool(
            "search_code",
            {"query": "greeting", "project_path": str(project), "include_debug": True},
        )

    assert SearchCodeResponse.model_validate(structured(plain)).debug is None
    assert SearchCodeResponse.model_validate(structured(detailed)).debug is not None


async def test_index_codebase_reports_what_it_indexed(project: Path, client: Client):
    async with client:
        result = await client.call_tool("index_codebase", {"project_path": str(project)})

    report = IndexCodebaseResponse.model_validate(structured(result))
    assert (report.files_indexed, report.chunks_indexed) == (2, 2)


async def test_index_status_counts_the_indexed_project_whose_index_embeds_nothing(
    project: Path, client: Client
):
    async with client:
        await client.call_tool("index_codebase", {"project_path": str(project)})
        result = await client.call_tool("index_status", {"project_path": str(project)})

    status = IndexStatusResponse.model_validate(structured(result))
    assert (
        status.is_indexed,
        status.files_count,
        status.chunks_count,
        status.embedded_chunks_count,
    ) == (True, 2, 2, 0)


@pytest.mark.parametrize(
    ("search_mode", "arguments", "embedded"),
    [
        pytest.param(SearchMode.words, {}, 0, id="by words, the default"),
        pytest.param(SearchMode.words, {"by_meaning": True}, 0, id="an agent asking for meaning"),
        pytest.param(SearchMode.words_and_meaning, {}, 2, id="set to search by meaning too"),
    ],
)
async def test_only_a_server_set_to_search_by_meaning_embeds_the_project_first(
    project: Path,
    tmp_path: Path,
    embedder: OnnxEmbedder,
    search_mode: SearchMode,
    arguments: dict[str, object],
    embedded: int,
):
    """The server's setting picks how a search ranks; an agent's call cannot (ADR 009, D2)."""
    settings = Settings(cache_dir=tmp_path / "cache", search_mode=search_mode)
    client = Client(create_app(settings, embedder=embedder), mode="legacy")

    async with client:
        await client.call_tool(
            "search_code", {"query": "greeting", "project_path": str(project), **arguments}
        )
        result = await client.call_tool("index_status", {"project_path": str(project)})

    assert IndexStatusResponse.model_validate(structured(result)).embedded_chunks_count == embedded


async def test_search_sees_a_function_added_after_indexing(project: Path, client: Client):
    async with client:
        await client.call_tool("index_codebase", {"project_path": str(project)})
        config_module = project / "config.py"
        config_module.write_text(config_module.read_text() + RETRY_FUNCTION)
        os.utime(config_module, (LATER_MTIME_SECONDS, LATER_MTIME_SECONDS))
        result = await client.call_tool(
            "search_code",
            {"query": "try again after waiting longer", "project_path": str(project)},
        )

    first_hit = SearchCodeResponse.model_validate(structured(result)).results[0]
    assert first_hit.name == "retry_with_backoff"


async def test_search_in_a_missing_project_is_an_error_naming_the_path(
    tmp_path: Path, client: Client
):
    missing = tmp_path / "missing"
    async with client:
        result = await client.call_tool(
            "search_code", {"query": "anything", "project_path": str(missing)}
        )

    assert result.is_error
    assert str(missing) in str(result.content)


async def test_a_glob_the_search_cannot_read_is_refused_before_the_project_is_looked_for(
    tmp_path: Path, client: Client
):
    """A refusal costs no call to git, and its reason is what the agent must change."""
    async with client:
        result = await client.call_tool(
            "search_code",
            {
                "query": "anything",
                "project_path": str(tmp_path / "missing"),
                "path_glob": "[ab].py",
            },
        )

    assert result.is_error
    assert REFUSED_CHARACTERS["["] in str(result.content)


async def test_a_limit_over_the_maximum_is_refused(project: Path, client: Client):
    async with client:
        result = await client.call_tool(
            "search_code", {"query": "anything", "project_path": str(project), "limit": 500}
        )

    assert result.is_error


async def test_a_failure_inside_a_tool_reaches_the_agent_with_its_cause(
    project: Path, tmp_path: Path, embedder: OnnxEmbedder
):
    """An index directory that cannot be made: the agent must hear why, not just that."""
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("")
    client = Client(create_app(Settings(cache_dir=blocking_file), embedder=embedder), mode="legacy")

    async with client:
        result = await client.call_tool("index_codebase", {"project_path": str(project)})

    assert result.is_error
    assert "not-a-directory" in str(result.content)


async def test_a_relative_project_path_is_refused(client: Client):
    async with client:
        result = await client.call_tool("index_status", {"project_path": "relative/project"})

    assert result.is_error
    assert "absolute" in str(result.content)


async def test_two_spellings_of_one_project_share_its_local_index(
    project: Path, tmp_path: Path, embedder: OnnxEmbedder
):
    settings = Settings(cache_dir=tmp_path / "cache", local_index=True)
    client = Client(create_app(settings, embedder=embedder), mode="legacy")
    roundabout = project / ".." / project.name

    async with client:
        await asyncio.gather(
            client.call_tool("index_codebase", {"project_path": str(roundabout)}),
            client.call_tool("index_codebase", {"project_path": str(project)}),
        )
        result = await client.call_tool("index_status", {"project_path": str(project)})

    # One lock for both spellings: the two runs do not store the chunks twice
    assert IndexStatusResponse.model_validate(structured(result)).chunks_count == 2


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A git repository with a package of one module and, outside it, a guide that
    answers the same question."""
    root = make_repository(tmp_path / "repository")
    (root / "greeting.py").write_text(GREETING_MODULE)
    (root / "docs").mkdir()
    (root / "docs" / "configuration.md").write_text(CONFIGURATION_GUIDE)
    (root / "package").mkdir()
    (root / "package" / "config.py").write_text(CONFIG_MODULE)
    return root


async def test_a_search_from_a_directory_of_a_repository_finds_only_what_lies_in_it(
    repository: Path, client: Client
):
    async with client:
        result = await client.call_tool(
            "search_code",
            {"query": CONFIGURATION_QUESTION, "project_path": str(repository / "package")},
        )

    hits = SearchCodeResponse.model_validate(structured(result)).results
    assert [(Path(hit.file_path), hit.name) for hit in hits] == [
        (repository / "package" / "config.py", "load_config")
    ]


@pytest.mark.parametrize(
    "searched",
    [
        pytest.param(Path("package"), id="a directory inside the repository"),
        pytest.param(Path(), id="the repository's root"),
    ],
)
async def test_a_search_names_the_repository_s_root_and_the_directory_it_kept_to(
    repository: Path, client: Client, searched: Path
):
    """A search kept to a directory says so, and names the root that would search it all."""
    async with client:
        result = await client.call_tool(
            "search_code",
            {"query": CONFIGURATION_QUESTION, "project_path": str(repository / searched)},
        )

    answer = SearchCodeResponse.model_validate(structured(result))
    assert (answer.project_root, answer.searched_directory) == (
        str(repository),
        str(repository / searched),
    )


async def test_a_directory_of_a_repository_is_indexed_as_the_whole_repository(
    repository: Path, client: Client
):
    async with client:
        await client.call_tool("index_codebase", {"project_path": str(repository / "package")})
        status = await client.call_tool("index_status", {"project_path": str(repository)})

    assert IndexStatusResponse.model_validate(structured(status)).files_count == 3


@pytest.mark.parametrize(
    ("tool", "response"),
    [
        pytest.param("index_codebase", IndexCodebaseResponse, id="index_codebase"),
        pytest.param("index_status", IndexStatusResponse, id="index_status"),
    ],
)
async def test_an_index_tool_asked_of_a_directory_of_a_repository_names_the_repository_s_root(
    repository: Path,
    client: Client,
    tool: str,
    response: type[IndexCodebaseResponse | IndexStatusResponse],
):
    """Its counts are the whole repository's, which the agent could not tell otherwise."""
    async with client:
        result = await client.call_tool(tool, {"project_path": str(repository / "package")})

    assert response.model_validate(structured(result)).project_root == str(repository)


@pytest.mark.parametrize(
    ("tool", "response"),
    [
        pytest.param("index_codebase", IndexCodebaseResponse, id="index_codebase"),
        pytest.param("index_status", IndexStatusResponse, id="index_status"),
    ],
)
async def test_an_index_tool_given_no_project_path_works_on_the_client_s_workspace_root(
    repository: Path,
    test_settings: Settings,
    embedder: OnnxEmbedder,
    tool: str,
    response: type[IndexCodebaseResponse | IndexStatusResponse],
):
    """As Claude Code names the directory it was started in."""
    client = client_with_roots(test_settings, embedder, [repository.as_uri()])

    async with client:
        result = await client.call_tool(tool, {})

    assert response.model_validate(structured(result)).project_root == str(repository)


@pytest.mark.parametrize(
    "blank",
    [pytest.param("", id="empty"), pytest.param("  ", id="blanks alone")],
)
async def test_a_blank_project_path_counts_as_left_out(
    repository: Path, test_settings: Settings, embedder: OnnxEmbedder, blank: str
):
    """As an agent fills an optional string it has nothing for."""
    client = client_with_roots(test_settings, embedder, [repository.as_uri()])

    async with client:
        result = await client.call_tool("index_status", {"project_path": blank})

    assert IndexStatusResponse.model_validate(structured(result)).project_root == str(repository)


async def test_a_search_given_no_project_path_searches_the_client_s_workspace_root(
    repository: Path, test_settings: Settings, embedder: OnnxEmbedder
):
    client = client_with_roots(test_settings, embedder, [repository.as_uri()])

    async with client:
        result = await client.call_tool("search_code", {"query": "say hello to someone"})

    first_hit = SearchCodeResponse.model_validate(structured(result)).results[0]
    assert Path(first_hit.file_path) == repository / "greeting.py"


@pytest.mark.parametrize(
    "client_of",
    [
        pytest.param(client_without_roots, id="a client that declares no roots"),
        pytest.param(partial(client_with_roots, uris=[]), id="a client that names none"),
        pytest.param(
            partial(client_with_roots, uris=[MISSING_ROOT]), id="a root that is no directory here"
        ),
        pytest.param(
            partial(client_with_roots, uris=[ROOT_ON_ANOTHER_MACHINE]),
            id="a root on another machine",
        ),
    ],
)
async def test_a_tool_given_no_project_path_and_no_one_workspace_root_asks_for_one(
    test_settings: Settings,
    embedder: OnnxEmbedder,
    client_of: Callable[[Settings, OnnxEmbedder], Client],
):
    client = client_of(test_settings, embedder)

    async with client:
        result = await client.call_tool("index_status", {})

    assert result.is_error
    assert "project_path" in str(result.content)


async def test_a_tool_given_no_project_path_and_two_workspace_roots_asks_for_one(
    repository: Path, tmp_path: Path, test_settings: Settings, embedder: OnnxEmbedder
):
    """Either could be meant: the server does not pick."""
    other = make_repository(tmp_path / "other")
    client = client_with_roots(test_settings, embedder, [repository.as_uri(), other.as_uri()])

    async with client:
        result = await client.call_tool("index_status", {})

    assert result.is_error
    assert "project_path" in str(result.content)


@pytest.mark.parametrize(
    "root",
    [
        pytest.param("home", id="a home under git for its dotfiles"),
        pytest.param("folder", id="a folder of repositories"),
    ],
)
async def test_a_tool_given_no_project_path_refuses_a_workspace_root_no_repository_names(
    tmp_path: Path,
    test_settings: Settings,
    embedder: OnnxEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    root: str,
):
    """A session started there would index every project below it as one."""
    make_repository(tmp_path / "home")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    make_repository(tmp_path / "folder" / "project")
    client = client_with_roots(test_settings, embedder, [(tmp_path / root).as_uri()])

    async with client:
        result = await client.call_tool("index_status", {})

    assert result.is_error
    assert "project_path" in str(result.content)


async def test_a_workspace_root_reached_through_a_link_is_the_project_it_links_to(
    repository: Path, tmp_path: Path, test_settings: Settings, embedder: OnnxEmbedder
):
    link = tmp_path / "link"
    link.symlink_to(repository)
    client = client_with_roots(test_settings, embedder, [link.as_uri()])

    async with client:
        result = await client.call_tool("index_status", {})

    assert IndexStatusResponse.model_validate(structured(result)).project_root == str(repository)
