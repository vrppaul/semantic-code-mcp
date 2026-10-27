"""Tests for the break check: a line of code broken while its test runs, then put back."""

import importlib.util
from pathlib import Path

import pytest

# The check is a script of the repository, not a module of the package: loaded by its path
CHECK_PATH = Path(__file__).parents[2] / "scripts" / "break_check.py"
CHECK_SPEC = importlib.util.spec_from_file_location("break_check", CHECK_PATH)
break_check = importlib.util.module_from_spec(CHECK_SPEC)
CHECK_SPEC.loader.exec_module(break_check)

MODULE_FILE = "module.py"
TEST_FILE = "test_module.py"
TEST_ID = "test_module.py::test_it"
CASES_FILE = "cases.toml"
MODULE = "def double(number):\n    return number * 2\n"
DOUBLING = "number * 2"
# As long as the doubling, so a compiled module left from the broken run would pass for the
# restored one
TRIPLING = "number * 3"
# A test of the doubling, one the doubling cannot fail, and one that fails on the code as it is
TEST_OF_DOUBLING = "from module import double\n\n\ndef test_it():\n    assert double(2) == 4\n"
TEST_OF_ZERO = "from module import double\n\n\ndef test_it():\n    assert double(0) == 0\n"
FAILING_TEST = "from module import double\n\n\ndef test_it():\n    assert double(2) == 5\n"
HELD_REPORT = f"held: {TEST_ID} <- 'number * 2'\n"
# Long enough for pytest to start and pass, far shorter than the check's own allowance
SHORT_TIMEOUT_SECONDS = 5


def case_toml(file: str, text: str, broken: str) -> str:
    """One case of a cases file, its texts written as TOML's literal strings."""
    return (
        f"[[cases]]\nfile = '{file}'\ntext = '''{text}'''\n"
        f"broken = '''{broken}'''\ntest = '{TEST_ID}'\n"
    )


def write_project(root: Path, monkeypatch: pytest.MonkeyPatch, test: str, cases: str) -> None:
    """A project of one module and its test, with a cases file, as the check's directory."""
    (root / MODULE_FILE).write_text(MODULE)
    (root / TEST_FILE).write_text(test)
    (root / CASES_FILE).write_text(cases)
    monkeypatch.chdir(root)


@pytest.mark.parametrize("broken", [TRIPLING, ""], ids=["changed", "deleted"])
def test_a_case_whose_test_fails_while_its_line_is_broken_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, broken: str
):
    write_project(tmp_path, monkeypatch, TEST_OF_DOUBLING, case_toml(MODULE_FILE, DOUBLING, broken))

    exit_code = break_check.main([CASES_FILE])

    assert (exit_code, capsys.readouterr().out, (tmp_path / MODULE_FILE).read_text()) == (
        break_check.EVERY_CASE_HELD,
        HELD_REPORT,
        MODULE,
    )


@pytest.mark.parametrize(
    ("test", "verdict"),
    [
        (TEST_OF_ZERO, "NOT HELD (failed broken: False, passed restored: True)"),
        (FAILING_TEST, "NOT HELD (failed broken: True, passed restored: False)"),
    ],
    ids=["passes-broken", "fails-restored"],
)
def test_a_case_holds_only_when_its_test_fails_broken_and_passes_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    test: str,
    verdict: str,
):
    write_project(tmp_path, monkeypatch, test, case_toml(MODULE_FILE, DOUBLING, TRIPLING))

    exit_code = break_check.main([CASES_FILE])

    assert (exit_code, capsys.readouterr().out, (tmp_path / MODULE_FILE).read_text()) == (
        break_check.A_CASE_NOT_HELD,
        f"{verdict}: {TEST_ID} <- 'number * 2'\n",
        MODULE,
    )


def test_a_run_past_the_time_allowed_counts_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    looping = case_toml(MODULE_FILE, "return number * 2", "while True: pass")
    write_project(tmp_path, monkeypatch, TEST_OF_DOUBLING, looping)
    monkeypatch.setattr(break_check, "TEST_RUN_TIMEOUT_SECONDS", SHORT_TIMEOUT_SECONDS)

    exit_code = break_check.main([CASES_FILE])

    assert (exit_code, capsys.readouterr().out) == (
        break_check.EVERY_CASE_HELD,
        f"held: {TEST_ID} <- 'return number * 2'\n",
    )


@pytest.mark.parametrize(
    "unplaceable",
    [
        case_toml(MODULE_FILE, "number * 4", TRIPLING),
        case_toml(MODULE_FILE, "number", "count"),
        case_toml(MODULE_FILE, DOUBLING, DOUBLING),
        case_toml("missing.py", DOUBLING, TRIPLING),
        f"[[cases]]\nfile = '{MODULE_FILE}'\n",
        "not = toml = at all\n",
    ],
    ids=["text-absent", "text-twice", "breaks-nothing", "file-missing", "keys-missing", "not-toml"],
)
def test_cases_the_check_cannot_place_are_refused_before_any_test_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, unplaceable: str
):
    placeable = case_toml(MODULE_FILE, DOUBLING, TRIPLING)
    write_project(tmp_path, monkeypatch, TEST_OF_DOUBLING, placeable + unplaceable)

    exit_code = break_check.main([CASES_FILE])

    assert (exit_code, capsys.readouterr().out, (tmp_path / MODULE_FILE).read_text()) == (
        break_check.CASES_REFUSED,
        "",
        MODULE,
    )
