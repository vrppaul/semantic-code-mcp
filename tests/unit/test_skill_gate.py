"""Tests for what the skill's gate reads of a text, and which paths of fields it knows."""

import importlib.util
from pathlib import Path

import pytest
from mcp.types import Tool

# The gate is a script of the repository, not a module of the package: loaded by its path
GATE_PATH = Path(__file__).parents[2] / "scripts" / "check_skill.py"
GATE_SPEC = importlib.util.spec_from_file_location("check_skill", GATE_PATH)
check_skill = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(check_skill)

# An answer with a nested object and a list of items its definitions describe
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "debug": {
            "anyOf": [
                {"type": "object", "properties": {"timings": {"$ref": "#/$defs/Timings"}}},
                {"type": "null"},
            ]
        },
        "results": {"type": "array", "items": {"$ref": "#/$defs/Hit"}},
    },
    "$defs": {
        "Timings": {"type": "object", "properties": {"total_ms": {"type": "number"}}},
        "Hit": {"type": "object", "properties": {"file_path": {"type": "string"}}},
    },
}
DESCRIPTION_WITH_ARGUMENTS = """Search code.

Args:
    query: What the code does.
    limit: How many results.
        Default: ten, which a continuation line says.

Returns:
    The results.
"""


@pytest.mark.parametrize(
    ("text", "path"),
    [
        pytest.param("`force: true`", ("force",), id="given a value after a colon"),
        pytest.param("`force = true`", ("force",), id="given a value after a spaced equals"),
        pytest.param("`index_codebase (force=true)`", ("index_codebase",), id="called"),
        pytest.param("`debug.timings.`", ("debug", "timings"), id="a path ending a sentence"),
        pytest.param("`results[0].file_path`", ("results", "file_path"), id="an item of a list"),
        pytest.param(
            "`debug.timings[1].total_ms`",
            ("debug", "timings", "total_ms"),
            id="an item of a list deeper in a path",
        ),
        pytest.param("`chunk_kinds2`", ("chunk_kinds2",), id="a name with a digit"),
    ],
)
def test_a_span_of_code_gives_the_path_of_names_it_writes(text: str, path: tuple[str, ...]):
    assert check_skill.code_paths(text) == {path}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("`uv run pytest`", id="a command"),
        pytest.param("`*.py`", id="a glob"),
        pytest.param("`Struct`", id="a capitalised name"),
        pytest.param("plain words with force", id="no code at all"),
    ],
)
def test_a_text_that_writes_no_name_in_code_gives_no_path(text: str):
    assert check_skill.code_paths(text) == set()


def test_an_args_section_gives_the_arguments_at_its_own_indent():
    """A deeper line that starts with a word and a colon continues an argument."""
    assert check_skill.argument_names(DESCRIPTION_WITH_ARGUMENTS) == ["query", "limit"]


def test_a_description_without_an_args_section_gives_no_arguments():
    assert check_skill.argument_names("Search code.\n\nReturns:\n    The results.\n") == []


def test_a_settings_variable_is_read_anywhere_in_a_text():
    text = "Set it:\n\n```bash\nexport SEMANTIC_CODE_MCP_USE_GITIGNOR=false\n```\n"

    assert check_skill.named_settings(text) == {"use_gitignor"}


def test_a_dotted_path_is_known_only_as_a_run_of_fields_an_answer_holds():
    """Each name of `debug.file_path` is a field, but no answer holds the one in the other."""
    tool = Tool(name="search", input_schema={"type": "object"}, output_schema=ANSWER_SCHEMA)
    surface = check_skill.ServerSurface(tools=(tool,), instructions="")

    runs = check_skill.surface_paths(surface)

    asked = [
        ("debug", "timings"),
        ("timings", "total_ms"),
        ("debug", "timings", "total_ms"),
        ("results", "file_path"),
        ("debug", "file_path"),
    ]
    assert [path in runs for path in asked] == [True, True, True, True, False]


# A server of one tool, and texts that agree with it: each check of the gate has a break
SEARCH_DESCRIPTION = (
    "Search python code.\n\nArgs:\n    query: What the code does.\n    languages: Only these.\n"
)
INSTRUCTIONS = "Use `search` for python code."
SKILL = (
    "Call `search` with `query` and `languages` (`python`); read `results` (`file_path`)"
    " and `debug.timings.total_ms`."
)


def surface_of(description: str, instructions: str) -> object:
    """The one-tool server's surface, with the tool's description and instructions given."""
    tool = Tool(
        name="search",
        description=description,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "languages": {"type": "array", "items": {"enum": ["python"]}},
            },
        },
        output_schema=ANSWER_SCHEMA,
    )
    return check_skill.ServerSurface(tools=(tool,), instructions=instructions)


def test_texts_that_agree_with_the_server_pass_the_gate():
    assert check_skill.disagreements(surface_of(SEARCH_DESCRIPTION, INSTRUCTIONS), SKILL) == []


@pytest.mark.parametrize(
    ("description", "instructions", "skill", "found"),
    [
        pytest.param(
            SEARCH_DESCRIPTION,
            INSTRUCTIONS,
            SKILL.replace("`query`", "`query` and `limt`"),
            ["limt, named in skills/semantic-search/SKILL.md, is none of the server's"],
            id="a name the server lacks",
        ),
        pytest.param(
            SEARCH_DESCRIPTION,
            INSTRUCTIONS,
            SKILL.replace("`file_path`", "`file_path`, `debug.file_path`"),
            [
                (
                    "debug.file_path, named in skills/semantic-search/SKILL.md,"
                    " is no path of the server's fields"
                )
            ],
            id="a path no answer holds",
        ),
        pytest.param(
            SEARCH_DESCRIPTION,
            INSTRUCTIONS,
            SKILL.replace("`query` and ", ""),
            ["skills/semantic-search/SKILL.md names no query"],
            id="a parameter the skill leaves out",
        ),
        pytest.param(
            SEARCH_DESCRIPTION.replace("    query:", "    question:"),
            INSTRUCTIONS,
            SKILL,
            [
                "search's description gives an argument question, which the tool lacks",
                "search's description gives no argument query",
            ],
            id="an argument the tool lacks",
        ),
        pytest.param(
            SEARCH_DESCRIPTION,
            INSTRUCTIONS.replace("python ", ""),
            SKILL,
            ["python is named nowhere in the server's instructions"],
            id="a language a text leaves out",
        ),
        pytest.param(
            SEARCH_DESCRIPTION,
            INSTRUCTIONS + " Set `SEMANTIC_CODE_MCP_USE_GITIGNOR`.",
            SKILL,
            [
                (
                    "SEMANTIC_CODE_MCP_USE_GITIGNOR, named in the server's instructions,"
                    " is no setting of the server's"
                )
            ],
            id="a setting the server lacks",
        ),
    ],
)
def test_each_disagreement_of_a_text_with_the_server_fails_the_gate(
    description: str, instructions: str, skill: str, found: list[str]
):
    assert check_skill.disagreements(surface_of(description, instructions), skill) == found
