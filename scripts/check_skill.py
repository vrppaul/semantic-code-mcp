"""Check that the skill and the server's own words agree with the server.

Ported from livereel's `scripts/check-skill.ts`. An agent calls the tools from what the
skill, the server's instructions and the tools' descriptions say, so a tool or a parameter
the skill leaves out is one no agent uses, and a name it gives that the server lacks is a
call that fails. A name counts as named where it stands in the skill's code: a span
between backticks, or a fenced block.

It builds the server as the app does, lists its tools through an in-memory client, and
holds:
- the skill's code to name every tool, every parameter of each, and every value a
  parameter takes from a fixed set (the chunk kinds, the languages);
- every name of the skill's code written in snake case to be one the server has: a tool,
  a parameter, a field of an answer, or a setting;
- the skill, the instructions and every tool's description to name each language the
  server indexes, the values of its `languages` parameter, by name ("Markdown"), and the
  instructions to name every tool. A language a text names that the server does not
  index goes unseen: no closed list of languages exists to look for one in;
- the skill to its length, as D16 of ADR 006 sets it.

Run in the repository's root, as pre-commit runs it: `uv run python scripts/check_skill.py`.
Prints each disagreement, a line each, and exits 1; prints nothing and exits 0 when the
texts and the server agree.
"""

import asyncio
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from mcp import Client
from mcp.types import Tool
from pydantic import BaseModel, ConfigDict

from semantic_code_mcp.app import create_app
from semantic_code_mcp.config import Settings
from semantic_code_mcp.embedding import OnnxEmbedder

SKILL = Path("skills/semantic-search/SKILL.md")
# The skill's length at most, in lines (ADR 006, D16)
LONGEST_SKILL = 200
# The model the embedder is built for; nothing here loads it
MODEL_NAME = "all-MiniLM-L6-v2"
DISAGREE = 1
AGREE = 0
# What the skill writes as code: a fenced block, or a span between backticks
CODE = re.compile(r"```.*?```|`[^`]+`", re.DOTALL)
# A name in snake case, as the tools, their parameters and their answers' fields are named
SNAKE_CASE_NAME = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
# The keys of a JSON schema this reads: an object's fields, a fixed set of values
PROPERTIES = "properties"
ENUM = "enum"
# A schema's named definitions, and a reference to one, as `#/$defs/Language`
DEFINITIONS = "$defs"
REFERENCE = "$ref"
DEFINITION_PREFIX = "#/$defs/"
LANGUAGES_PARAMETER = "languages"


class ServerSurface(BaseModel):
    """What the server shows an agent: its tools, and its instructions."""

    model_config = ConfigDict(frozen=True)

    tools: tuple[Tool, ...]
    instructions: str


async def server_surface() -> ServerSurface:
    """The tools and instructions of the server the app builds, as a client lists them."""
    server = create_app(Settings(), embedder=OnnxEmbedder(MODEL_NAME))
    async with Client(server) as client:
        listed = await client.list_tools()
        return ServerSurface(tools=tuple(listed.tools), instructions=client.instructions or "")


def schema_nodes(schema: object) -> Iterator[dict[str, object]]:
    """Every object of a JSON schema, its definitions and nested schemas among them."""
    if isinstance(schema, dict):
        yield schema
        for value in schema.values():
            yield from schema_nodes(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from schema_nodes(item)


def field_names(schema: object) -> set[str]:
    """The names of every field a schema holds, at any depth."""
    names: set[str] = set()
    for node in schema_nodes(schema):
        properties = node.get(PROPERTIES)
        if isinstance(properties, dict):
            names.update(str(name) for name in properties)
    return names


def fixed_values(schema: object) -> set[str]:
    """Every value a schema takes from a fixed set, at any depth."""
    values: set[str] = set()
    for node in schema_nodes(schema):
        enum = node.get(ENUM)
        if isinstance(enum, list):
            values.update(str(value) for value in enum)
    return values


def is_named(text: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def parameter_values(tool: Tool, parameter: str) -> set[str]:
    """The values a parameter of a tool takes from a fixed set, the definitions its schema
    refers to followed."""
    parameter_schema = tool.input_schema[PROPERTIES][parameter]
    definitions = tool.input_schema.get(DEFINITIONS, {})
    referred = []
    for node in schema_nodes(parameter_schema):
        reference = node.get(REFERENCE)
        if isinstance(reference, str):
            referred.append(definitions[reference.removeprefix(DEFINITION_PREFIX)])
    return fixed_values([parameter_schema, *referred])


def language_names(tools: tuple[Tool, ...]) -> list[str]:
    """Each language the server indexes, by name: the values of the `languages`
    parameter, capitalised as the texts write them."""
    values = {
        value
        for tool in tools
        if LANGUAGES_PARAMETER in tool.input_schema[PROPERTIES]
        for value in parameter_values(tool, LANGUAGES_PARAMETER)
    }
    return sorted(value.capitalize() for value in values)


def names_disagreements(surface: ServerSurface, skill: str) -> list[str]:
    """The tools, parameters and values the skill's code leaves out, the names in snake
    case it gives that the server lacks, and the tools the instructions leave out."""
    code = "\n".join(CODE.findall(skill))
    tools = {tool.name for tool in surface.tools}
    parameters = {name for tool in surface.tools for name in field_names(tool.input_schema)}
    values = {value for tool in surface.tools for value in fixed_values(tool.input_schema)}
    answers = {name for tool in surface.tools for name in field_names(tool.output_schema)}
    known = tools | parameters | answers | set(Settings.model_fields)
    left_out = sorted(name for name in tools | parameters | values if not is_named(code, name))
    not_there = sorted(set(SNAKE_CASE_NAME.findall(code)) - known)
    unnamed_tools = sorted(name for name in tools if not is_named(surface.instructions, name))
    return [
        *(f"{SKILL} does not name {name}" for name in left_out),
        *(f"{SKILL} names {name}, which the server does not have" for name in not_there),
        *(f"the server's instructions do not name {name}" for name in unnamed_tools),
    ]


def language_disagreements(surface: ServerSurface, skill: str) -> list[str]:
    """Each text that leaves out a language the server indexes; and the server itself,
    where it lists none, as then nothing here could be checked."""
    languages = language_names(surface.tools)
    if not languages:
        return [f"no tool of the server lists the values of `{LANGUAGES_PARAMETER}`"]
    texts = {str(SKILL): skill, "the server's instructions": surface.instructions} | {
        f"the description of {tool.name}": tool.description or "" for tool in surface.tools
    }
    return [
        f"{where} does not name {language}"
        for language in languages
        for where, text in texts.items()
        if not is_named(text, language)
    ]


def length_disagreements(skill: str) -> list[str]:
    lines = skill.count("\n")
    if lines > LONGEST_SKILL:
        return [f"{SKILL} runs to {lines} lines, more than {LONGEST_SKILL}"]
    return []


def disagreements(surface: ServerSurface, skill: str) -> list[str]:
    """Each place the skill or the server's own words disagree with the server."""
    return [
        *names_disagreements(surface, skill),
        *language_disagreements(surface, skill),
        *length_disagreements(skill),
    ]


def main() -> int:
    found = disagreements(asyncio.run(server_surface()), SKILL.read_text())
    for disagreement in found:
        sys.stdout.write(f"{disagreement}\n")
    return DISAGREE if found else AGREE


if __name__ == "__main__":
    sys.exit(main())
