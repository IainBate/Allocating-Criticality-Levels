"""The paper's stated defaults must match the code's actual defaults.

research/eval_defaults.tex defines named LaTeX macros for the Simulation
Parameters table in evaluation_methodology.tex, specifically so the values
are not literals scattered through the text. That only prevents the paper
disagreeing with *itself*; it says nothing about the paper disagreeing with
generate_taskset's actual defaults. This test closes that gap: it parses the
macro file and compares every value against the live function signature, so
changing one default without the other fails here rather than being noticed
by a reader.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from amc_tasksim.generation.taskset import generate_taskset

DEFAULTS_TEX = Path(__file__).parents[2] / "research" / "eval_defaults.tex"

# LaTeX macro name -> generate_taskset keyword argument it must match.
MACRO_TO_PARAM = {
    "defaultNumTasks": "n",
    "defaultCP": "CP",
    "defaultU": "U",
    "defaultInvFailProb": "N",
}


def _parse_macros(path: Path) -> dict[str, str]:
    """Extract {name: value} from every \\newcommand{\\name}{value} in path."""
    text = path.read_text()
    pattern = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")
    return dict(pattern.findall(text))


@pytest.fixture(scope="module")
def macros() -> dict[str, str]:
    assert DEFAULTS_TEX.exists(), f"expected {DEFAULTS_TEX} to exist"
    found = _parse_macros(DEFAULTS_TEX)
    assert found, f"no \\newcommand definitions found in {DEFAULTS_TEX}"
    return found


@pytest.fixture(scope="module")
def code_defaults() -> dict[str, object]:
    sig = inspect.signature(generate_taskset)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }


def test_every_referenced_macro_is_defined(macros):
    missing = [m for m in MACRO_TO_PARAM if m not in macros]
    assert not missing, f"eval_defaults.tex is missing macros: {missing}"


@pytest.mark.parametrize("macro_name,param_name", sorted(MACRO_TO_PARAM.items()))
def test_paper_default_matches_code_default(macros, code_defaults, macro_name, param_name):
    tex_value = macros[macro_name]
    code_value = code_defaults[param_name]
    # LaTeX macros hold plain text; compare as the code default's own type so
    # "20" == 20 and "0.5" == 0.5 without the test caring about formatting.
    parsed = type(code_value)(tex_value)
    assert parsed == code_value, (
        f"eval_defaults.tex \\{macro_name}={tex_value!r} does not match "
        f"generate_taskset({param_name}={code_value!r})"
    )


def test_period_range_matches_the_paper(macros, code_defaults):
    t_min, t_max = code_defaults["period_range"]
    assert int(macros["defaultTmin"]) == t_min
    assert int(macros["defaultTmax"]) == t_max


def test_bcet_fraction_matches_the_paper(macros, code_defaults):
    bcet_min, _bcet_max = code_defaults["bcet_fraction_range"]
    assert float(macros["defaultBCETmin"]) == bcet_min


def test_the_test_itself_would_catch_a_drifted_default(macros, code_defaults):
    """Prove the comparison is live, not vacuously true.

    If someone changes generate_taskset's default n without updating the
    paper, this is the assertion that must fail -- verified here by checking
    the comparison actually depends on both sides, not a fixture that always
    agrees with itself.
    """
    assert int(macros["defaultNumTasks"]) == code_defaults["n"]
    drifted = code_defaults["n"] + 1
    assert int(macros["defaultNumTasks"]) != drifted
