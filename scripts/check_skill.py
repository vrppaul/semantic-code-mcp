"""Check that the skill and the server's own words agree with the server.

Ported from livereel's `scripts/check-skill.ts`. An agent calls the tools from what the
skill, the server's instructions and the tools' descriptions say, so a tool, a parameter
or a field of an answer the skill leaves out is one no agent uses, and a name a text gives
that the server lacks is a call that fails. A name counts as named where it stands in a
text's code: a span between backticks, or a fenced block.

It builds the server as the app does, its settings as they ship (the developer's
`SEMANTIC_CODE_MCP_*` variables left out), lists its tools through an in-memory client,
and holds:
- the skill's code to name every tool, every parameter of each, every value a parameter
  takes from a fixed set (the chunk kinds, the languages), and every field of every
  answer, nested ones among them;
- in the skill, the instructions, each tool's description and the descriptions its
  schemas give its parameters and the fields of its answer, every span of code that is a
  name or a dotted path of names (`debug.timings`, and one ending a sentence), alone,
  called (`index_codebase(...)`) or given a value (`force=true`, `force: true`), to be one
  the server has: a tool, a parameter, one of its fixed values, a field of an answer, or a
  setting, the examples' own project names aside; and every `SEMANTIC_CODE_MCP_*`
  variable anywhere in them, a fenced block among other code included, to name a setting;
- each tool's description to give, in its `Args:` section, every parameter of the tool
  and no other;
- the skill, the instructions and every tool's description to name each language the
  server indexes, the values of its `languages` parameter, in any case; and the
  instructions to name every tool;
- the skill to its length, as D16 of ADR 006 sets it.

What it cannot see: a name written in plain words outside `Args:` (with force); what a
sentence says of a name (a wrong default, cap or meaning, which the texts therefore leave
to the tools' schemas rather than repeating their numbers); the value after a name's `=`
or `:` or inside its call (`chunk_kinds=["struct"]`); a name written with a capital
(`Struct`), as the examples' own classes are; a name inside a fenced block among other
code; a language a text names that no chunker reads, as no closed list of languages
exists to look for one in.

Run in the repository's root, as pre-commit runs it: `uv run python scripts/check_skill.py`.
Prints each disagreement, a line each, and exits 1; prints nothing and exits 0 when the
texts and the server agree.
"""

import asyncio
import os
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
# The prefix of the settings a developer's environment may hold, which the gate leaves out
SETTINGS_PREFIX = "SEMANTIC_CODE_MCP_"
DISAGREE = 1
AGREE = 0
# What a text writes as code: a fenced block, or a span between backticks
CODE = re.compile(r"```.*?```|`[^`]+`", re.DOTALL)
CODE_QUOTE = "`"
# A span that is a name or a dotted path of names, alone, ending a sentence, called, or
# given a value: `chunk_kinds`, `debug.timings.`, `index_codebase(force=true)`,
# `force=true`, `force: true`
NAMED_SPAN = re.compile(r"(?P<path>[a-z][a-z_]*(?:\.[a-z][a-z_]*)*)(?:[(=:].*|\.)?", re.DOTALL)
PATH_GROUP = "path"
PATH_SEPARATOR = "."
# A settings variable of the server's, wherever a text writes one
SETTING_VARIABLE = re.compile(r"\bSEMANTIC_CODE_MCP_(?P<setting>[A-Z_]+)")
SETTING_GROUP = "setting"
# Names the texts give as a project's own, in their examples, which no server has
EXAMPLE_NAMES = frozenset({"greet", "src", "tests"})
# A docstring's section of arguments: its heading, then its indented lines, a blank line
# among them only where another indented one follows
ARGUMENTS_SECTION = re.compile(
    r"^Args:\n(?P<body>(?:[ \t]+\S.*\n?|[ \t]*\n(?=[ \t]+\S))*)", re.MULTILINE
)
BODY_GROUP = "body"
# A line of that section that starts with a name and a colon; those at the section's
# shallowest indent give its arguments, deeper ones continue a description
ARGUMENT_LINE = re.compile(r"(?P<indent>[ \t]+)(?P<name>[A-Za-z_]\w*):(?:\s.*)?")
INDENT_GROUP = "indent"
NAME_GROUP = "name"
# The keys of a JSON schema this reads: an object's fields, a fixed set of values, what
# a field is
PROPERTIES = "properties"
ENUM = "enum"
DESCRIPTION = "description"
# A schema's named definitions, and a reference to one, as `#/$defs/Language`
DEFINITIONS = "$defs"
REFERENCE = "$ref"
DEFINITION_PREFIX = "#/$defs/"
LANGUAGES_PARAMETER = "languages"
SKILL_TEXT = str(SKILL)
INSTRUCTIONS_TEXT = "the server's instructions"


class ServerSurface(BaseModel):
    """What the server shows an agent: its tools, and its instructions."""

    model_config = ConfigDict(frozen=True)

    tools: tuple[Tool, ...]
    instructions: str


async def server_surface() -> ServerSurface:
    """The tools and instructions of the server the app builds, with the settings it ships
    with, as a client lists them."""
    for name in [name for name in os.environ if name.startswith(SETTINGS_PREFIX)]:
        del os.environ[name]
    settings = Settings()
    server = create_app(settings, embedder=OnnxEmbedder(settings.embedding_model))
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


def schema_descriptions(schema: object) -> str:
    """What a schema says of its fields, every description at any depth, one after another."""
    descriptions = [node.get(DESCRIPTION) for node in schema_nodes(schema)]
    return "\n".join(description for description in descriptions if isinstance(description, str))


def fixed_values(schema: object) -> set[str]:
    """Every value a schema takes from a fixed set, at any depth."""
    values: set[str] = set()
    for node in schema_nodes(schema):
        enum = node.get(ENUM)
        if isinstance(enum, list):
            values.update(str(value) for value in enum)
    return values


def parameter_values(tool: Tool, parameter: str) -> set[str]:
    """The values a parameter of a tool takes from a fixed set, the definitions its schema
    refers to followed; a reference to no definition of the schema's own finds none."""
    parameter_schema = tool.input_schema[PROPERTIES][parameter]
    definitions = tool.input_schema.get(DEFINITIONS, {})
    referred = []
    for node in schema_nodes(parameter_schema):
        reference = node.get(REFERENCE)
        if isinstance(reference, str):
            referred.append(definitions.get(reference.removeprefix(DEFINITION_PREFIX)))
    return fixed_values([parameter_schema, *referred])


def language_values(tools: tuple[Tool, ...]) -> list[str]:
    """Each language the server indexes: the values of the `languages` parameter."""
    values = {
        value
        for tool in tools
        if LANGUAGES_PARAMETER in tool.input_schema[PROPERTIES]
        for value in parameter_values(tool, LANGUAGES_PARAMETER)
    }
    return sorted(values)


def is_named(text: str, name: str, flags: re.RegexFlag = re.NOFLAG) -> bool:
    """Whether a text holds the name as a word of its own."""
    return re.search(rf"\b{re.escape(name)}\b", text, flags) is not None


def code_of(text: str) -> str:
    """A text's code, its spans and fenced blocks, one after another."""
    return "\n".join(CODE.findall(text))


def code_names(text: str) -> set[str]:
    """The names a text's code gives: each name of a span that is a name or a dotted path
    of names, alone, called, or given a value."""
    names: set[str] = set()
    for span in CODE.findall(text):
        match = NAMED_SPAN.fullmatch(span.strip(CODE_QUOTE))
        if match is not None:
            names.update(match.group(PATH_GROUP).split(PATH_SEPARATOR))
    return names


def named_settings(text: str) -> set[str]:
    """The settings a text names by their variables, anywhere in it, as the settings'
    fields: the prefix is the server's own, so no other word is read as one."""
    return {match.group(SETTING_GROUP).lower() for match in SETTING_VARIABLE.finditer(text)}


def argument_names(description: str) -> list[str]:
    """The arguments a docstring's `Args:` section gives, by name, in its order."""
    section = ARGUMENTS_SECTION.search(description)
    if section is None:
        return []
    matches = [ARGUMENT_LINE.fullmatch(line) for line in section.group(BODY_GROUP).splitlines()]
    arguments = [match for match in matches if match is not None]
    if not arguments:
        return []
    shallowest = min(len(argument.group(INDENT_GROUP)) for argument in arguments)
    return [
        argument.group(NAME_GROUP)
        for argument in arguments
        if len(argument.group(INDENT_GROUP)) == shallowest
    ]


def surface_names(surface: ServerSurface) -> set[str]:
    """Every name an agent can use: the tools, their parameters and the values a parameter
    takes from a fixed set, and the fields of their answers."""
    tools = {tool.name for tool in surface.tools}
    parameters = {name for tool in surface.tools for name in field_names(tool.input_schema)}
    values = {value for tool in surface.tools for value in fixed_values(tool.input_schema)}
    answers = {name for tool in surface.tools for name in field_names(tool.output_schema)}
    return tools | parameters | values | answers


def texts_of(surface: ServerSurface, skill: str) -> dict[str, str]:
    """Every text an agent reads of the server, by where it stands: the skill, the
    instructions, and each tool's description and what its schemas say of their fields."""
    texts = {SKILL_TEXT: skill, INSTRUCTIONS_TEXT: surface.instructions}
    for tool in surface.tools:
        texts[f"{tool.name}'s description"] = tool.description or ""
        texts[f"{tool.name}'s schemas"] = "\n".join(
            [schema_descriptions(tool.input_schema), schema_descriptions(tool.output_schema)]
        )
    return texts


def left_out_disagreements(surface: ServerSurface, skill: str) -> list[str]:
    """What the skill's code leaves out of the server, and the tools the instructions leave
    out."""
    code = code_of(skill)
    left_out = sorted(name for name in surface_names(surface) if not is_named(code, name))
    unnamed_tools = sorted(
        tool.name for tool in surface.tools if not is_named(surface.instructions, tool.name)
    )
    return [
        *(f"{SKILL_TEXT} names no {name}" for name in left_out),
        *(f"{INSTRUCTIONS_TEXT} name no {name}" for name in unnamed_tools),
    ]


def unknown_name_disagreements(surface: ServerSurface, skill: str) -> list[str]:
    """Each name a text's code gives that the server lacks, and each settings variable
    that names no setting."""
    settings = set(Settings.model_fields)
    known = surface_names(surface) | settings | EXAMPLE_NAMES
    found: list[str] = []
    for where, text in texts_of(surface, skill).items():
        found.extend(
            f"{name}, named in {where}, is none of the server's"
            for name in sorted(code_names(text) - known)
        )
        found.extend(
            f"{SETTINGS_PREFIX}{name.upper()}, named in {where}, is no setting of the server's"
            for name in sorted(named_settings(text) - settings)
        )
    return found


def argument_disagreements(surface: ServerSurface) -> list[str]:
    """Each tool whose description's `Args:` section gives an argument the tool lacks, or
    leaves out one it has."""
    found: list[str] = []
    for tool in surface.tools:
        given = argument_names(tool.description or "")
        parameters = list(tool.input_schema[PROPERTIES])
        found.extend(
            f"{tool.name}'s description gives an argument {name}, which the tool lacks"
            for name in given
            if name not in parameters
        )
        found.extend(
            f"{tool.name}'s description gives no argument {name}"
            for name in parameters
            if name not in given
        )
    return found


def language_disagreements(surface: ServerSurface, skill: str) -> list[str]:
    """Each text that leaves out a language the server indexes; and the server itself,
    where it lists none, as then nothing here could be checked."""
    languages = language_values(surface.tools)
    if not languages:
        return [f"no tool of the server lists the values of `{LANGUAGES_PARAMETER}`"]
    texts = {SKILL_TEXT: skill, INSTRUCTIONS_TEXT: surface.instructions} | {
        f"the description of {tool.name}": tool.description or "" for tool in surface.tools
    }
    return [
        f"{language} is named nowhere in {where}"
        for language in languages
        for where, text in texts.items()
        if not is_named(text, language, re.IGNORECASE)
    ]


def length_disagreements(skill: str) -> list[str]:
    """The skill's length, where it runs past the most D16 allows."""
    lines = skill.count("\n")
    if lines > LONGEST_SKILL:
        return [f"{SKILL_TEXT} runs to {lines} lines, more than {LONGEST_SKILL}"]
    return []


def disagreements(surface: ServerSurface, skill: str) -> list[str]:
    """Each place the skill or the server's own words disagree with the server."""
    return [
        *left_out_disagreements(surface, skill),
        *unknown_name_disagreements(surface, skill),
        *argument_disagreements(surface),
        *language_disagreements(surface, skill),
        *length_disagreements(skill),
    ]


def main() -> int:
    """Print each disagreement; exit 1 when there is one, 0 when there is none."""
    found = disagreements(asyncio.run(server_surface()), SKILL.read_text())
    for disagreement in found:
        sys.stdout.write(f"{disagreement}\n")
    return DISAGREE if found else AGREE


if __name__ == "__main__":
    sys.exit(main())
