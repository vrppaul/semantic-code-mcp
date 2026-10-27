"""Break the text of a file a test holds, run the test, and put the text back.

`.claude/rules/testing.md` asks that a test written after its code, or rewritten, be seen
failing with the line it holds broken and passing with it restored, or the test may hold
nothing. This check runs both for each case and writes the file back as it was, whatever
the runs do.

Run in the repository's root, in the project's environment, with a file of cases:

    uv run python scripts/break_check.py CASES.toml

Each case is a `[[cases]]` table: `file`, its path from the root; `text`, as the file holds
it, exactly once; `broken`, what the text becomes while the test runs (empty deletes it);
`test`, the test's id as pytest takes it. TOML's literal strings ('''...''') keep a
backslash as it is written, where JSON would have its author double it. Every case is
checked against its file before any test runs, and one that cannot be placed refuses them
all. A test that loads the embedding model runs under whatever memory cap the check was
started in.

Prints a line per case as it finishes: `held`, or `NOT HELD` with how the two runs went. A
check killed during a case leaves that case's file broken, which `git diff` shows. Exits 0
when every case held, 1 when one did not, and 2 when the cases were refused.
"""

import argparse
import subprocess  # nosec B404
import sys
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

EVERY_CASE_HELD = 0
A_CASE_NOT_HELD = 1
# argparse exits with this code too, on arguments it cannot read
CASES_REFUSED = 2
# pytest's code for a run whose every test passed
PYTEST_PASSED = 0
# The test runs alone, stops at its first failure and leaves pytest's cache as it was, so
# no state of the broken code outlives its run
PYTEST_OPTIONS = ("-q", "-x", "--no-header", "-p", "no:cacheprovider")
# A broken line can make a test loop for ever: a run this long counts as failed
TEST_RUN_TIMEOUT_SECONDS = 600
# How much of a case's text its line of the report shows
REPORTED_TEXT_LENGTH = 60
# Where Python keeps the code it compiled of a module, beside it, one file per interpreter
COMPILED_DIRECTORY = "__pycache__"
COMPILED_FILE_PATTERN = "{module}.*.pyc"
HELD_VERDICT = "held"
NOT_HELD_VERDICT = "NOT HELD (failed broken: {failed_broken}, passed restored: {passed_restored})"
REPORT_LINE = "{verdict}: {test} <- {text!r}"


class BreakCase(BaseModel):
    """A text of a file to break, and the test that should fail while it is broken."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # From the repository's root
    file: Path
    # As the file holds it, exactly once
    text: str
    # What the text becomes while the test runs; empty deletes it
    broken: str
    # The test's id, as pytest takes it
    test: str


class BreakCases(BaseModel):
    """A file of cases: its `[[cases]]` tables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: list[BreakCase]


class CaseOutcome(BaseModel):
    """How a case's two runs of its test went."""

    model_config = ConfigDict(frozen=True)

    case: BreakCase
    failed_broken: bool
    passed_restored: bool

    @property
    def held(self) -> bool:
        return self.failed_broken and self.passed_restored

    def report(self) -> str:
        """The case's line of the report."""
        verdict = (
            HELD_VERDICT
            if self.held
            else NOT_HELD_VERDICT.format(
                failed_broken=self.failed_broken, passed_restored=self.passed_restored
            )
        )
        text = self.case.text.strip()[:REPORTED_TEXT_LENGTH]
        return REPORT_LINE.format(verdict=verdict, test=self.case.test, text=text)


class CasesRefusedError(ValueError):
    """A file of cases the check cannot read, or a case it cannot place in its file."""


def load_cases(path: Path) -> list[BreakCase]:
    """The cases of a file, each checked against the file it breaks; refused whole when the
    file of cases cannot be read or one case cannot be placed."""
    try:
        cases = BreakCases.model_validate(tomllib.loads(path.read_text())).cases
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        message = f"{path}: {error}"
        raise CasesRefusedError(message) from error
    refusals = [refusal for case in cases if (refusal := refusal_of(case)) is not None]
    if refusals:
        raise CasesRefusedError("\n".join(refusals))
    return cases


def refusal_of(case: BreakCase) -> str | None:
    """Why a case cannot be placed in its file, or None when it can."""
    try:
        source = case.file.read_text()
    except OSError as error:
        return f"{case.file}: {error}"
    occurrences = source.count(case.text)
    if occurrences != 1:
        return f"{case.file}: {case.text!r} occurs {occurrences} times, not once"
    if case.broken == case.text:
        return f"{case.file}: {case.text!r} is broken into itself"
    return None


def run_case(case: BreakCase) -> CaseOutcome:
    """Run a case's test with its text broken, write the file back as it was, and run the
    test again."""
    source = case.file.read_text()
    try:
        write_source(case.file, source.replace(case.text, case.broken, 1))
        failed_broken = not passes(case.test)
    finally:
        write_source(case.file, source)
    return CaseOutcome(case=case, failed_broken=failed_broken, passed_restored=passes(case.test))


def write_source(path: Path, source: str) -> None:
    """Write a file, and drop the code Python compiled of it: a write of the same size within
    the second of the last compile leaves that code looking current, and the next run would
    test the text before the write."""
    path.write_text(source)
    compiled_files = (path.parent / COMPILED_DIRECTORY).glob(
        COMPILED_FILE_PATTERN.format(module=path.stem)
    )
    for compiled_file in compiled_files:
        compiled_file.unlink(missing_ok=True)


def passes(test: str) -> bool:
    """Whether a test passes, run alone by this environment's pytest."""
    command = (sys.executable, "-m", "pytest", *PYTEST_OPTIONS, test)
    try:
        # This environment's own interpreter, given a test id from the cases' author
        completed = subprocess.run(  # nosec B603
            command, capture_output=True, check=False, timeout=TEST_RUN_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == PYTEST_PASSED


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Break what each test holds, run it, restore.")
    parser.add_argument("cases", type=Path, help="the TOML file of cases")
    cases_file: Path = parser.parse_args(arguments).cases
    try:
        cases = load_cases(cases_file)
    except CasesRefusedError as error:
        print(error, file=sys.stderr)
        return CASES_REFUSED
    outcomes: list[CaseOutcome] = []
    for case in cases:
        outcome = run_case(case)
        print(outcome.report(), flush=True)
        outcomes.append(outcome)
    return EVERY_CASE_HELD if all(outcome.held for outcome in outcomes) else A_CASE_NOT_HELD


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
