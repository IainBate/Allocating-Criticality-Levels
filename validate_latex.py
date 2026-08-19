#!/usr/bin/env python3
r"""Validate LaTeX files by actually compiling them.

The previous version of this script only did regex checks (balanced braces,
matched begin/end, presence of \bibliography if \cite is used). Those catch
some typos but cannot catch what actually breaks a build: an undefined macro,
a missing \label for a \ref, a citation key absent from every .bib file, a
package error. All four of those were live bugs in this project's two papers
-- \round and \E used without definition, \ref{sec:parameters} with no
matching \label, and \bibliography pointing at a references.bib that did not
exist -- and every one of them passed the old regex checks silently.

This version compiles each file with tectonic (a self-contained LaTeX engine:
`brew install tectonic`, no TeX Live required) and parses its log for the
same class of error LaTeX itself would report, so "passes validation" means
"produces a PDF with no undefined citations, references, or macros" rather
than "has matching braces".

Usage:
    python validate_latex.py [file_or_directory...]

Exit status 0 if every file compiles cleanly, 1 otherwise.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Patterns that indicate a real problem, matched against tectonic's combined
# stdout+stderr. These are LaTeX's own diagnostic wording, not something this
# script invents.
_ERROR_PATTERNS = [
    (re.compile(r"^error:", re.MULTILINE), "compile error"),
    (re.compile(r"Undefined control sequence"), "undefined macro"),
    (re.compile(r"Citation `[^']*' .* undefined"), "undefined citation"),
    (re.compile(r"Reference `[^']*' .* undefined"), "undefined cross-reference"),
    (re.compile(r"! LaTeX Error"), "LaTeX error"),
    (re.compile(r"! Package \S+ Error"), "package error"),
    (re.compile(r"I couldn't open (style|database) file"), "missing bib/style file"),
    (re.compile(r"I found no \\\\?bibstyle command"), "missing bibliographystyle"),
]

# Cosmetic warnings that are not worth failing a commit over.
_IGNORE_PATTERNS = [
    re.compile(r"`h' float specifier changed to `ht'"),
    re.compile(r"Rerun to get "),
]


def find_tectonic() -> str | None:
    return shutil.which("tectonic")


def is_standalone_document(tex_path: Path) -> bool:
    """Whether this file is compilable on its own rather than an \\input fragment.

    A project can have shared files (macros.tex, a results table someone
    \\input's) that have no \\documentclass and are never compiled directly.
    Trying to compile one produces a failure that is not about the file's
    content, so those are skipped rather than reported as broken.
    """
    content = tex_path.read_text(errors="replace")
    return bool(re.search(r"\\documentclass", content))


def compile_check(tex_path: Path) -> list[str]:
    """Compile ``tex_path`` and return a list of problems found in the log.

    Runs in a scratch directory copying the whole parent directory of
    ``tex_path``, so sibling files it \\input's or \\bibliography's against
    (macros.tex, project_bib.bib, references.bib) are present -- compiling a
    single file in isolation would itself produce spurious "undefined" errors
    that have nothing to do with the file being checked.
    """
    tectonic = find_tectonic()
    if tectonic is None:
        return [
            "tectonic is not installed, so this is a syntax-only check "
            "(braces/environments), not a real compile. Install it with "
            "`brew install tectonic` for a check that catches undefined "
            "macros, missing citations, and broken cross-references."
        ] + _syntax_only_check(tex_path)

    source_dir = tex_path.parent
    with tempfile.TemporaryDirectory(prefix="texcheck_") as scratch:
        scratch_dir = Path(scratch)
        for item in source_dir.iterdir():
            if item.name.startswith(".") or item.suffix in {
                ".pdf", ".aux", ".log", ".bbl", ".blg", ".out",
            }:
                continue
            dest = scratch_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)

        proc = subprocess.run(
            [tectonic, "--keep-logs", tex_path.name],
            cwd=scratch_dir,
            capture_output=True,
            text=True,
            timeout=180,
        )
        combined = proc.stdout + "\n" + proc.stderr
        log_path = scratch_dir / tex_path.with_suffix(".log").name
        if log_path.exists():
            combined += "\n" + log_path.read_text(errors="replace")

        problems: list[str] = []
        for line in combined.splitlines():
            if any(p.search(line) for p in _IGNORE_PATTERNS):
                continue
            for pattern, label in _ERROR_PATTERNS:
                if pattern.search(line):
                    problems.append(f"{label}: {line.strip()}")
                    break

        if proc.returncode != 0 and not problems:
            problems.append(
                f"tectonic exited {proc.returncode} with no matched pattern; "
                f"tail of output:\n" + "\n".join(combined.splitlines()[-15:])
            )
        return problems


def _syntax_only_check(tex_path: Path) -> list[str]:
    """Fallback when tectonic is unavailable: the old regex-based checks."""
    content = tex_path.read_text(errors="replace")
    errors: list[str] = []

    depth = 0
    for i, char in enumerate(content, 1):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                errors.append(f"line {i}: unmatched closing brace")
                depth = 0
    if depth > 0:
        errors.append(f"missing {depth} closing brace(s)")

    begins = re.findall(r"\\begin\s*\{([^}]+)\}", content)
    ends = re.findall(r"\\end\s*\{([^}]+)\}", content)
    if sorted(begins) != sorted(ends):
        errors.append("mismatched \\begin/\\end environments")

    return errors


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        tex_files = sorted(Path(".").glob("**/*.tex"))
    else:
        tex_files = []
        for arg in argv[1:]:
            path = Path(arg)
            if path.is_file() and path.suffix == ".tex":
                tex_files.append(path)
            elif path.is_dir():
                tex_files.extend(sorted(path.glob("**/*.tex")))

    if not tex_files:
        print("No .tex files found")
        return 0

    all_problems: dict[str, list[str]] = {}
    checked = 0
    skipped = []
    for filepath in tex_files:
        if not is_standalone_document(filepath):
            skipped.append(str(filepath))
            continue
        checked += 1
        problems = compile_check(filepath)
        if problems:
            all_problems[str(filepath)] = problems

    if skipped:
        print(f"(skipped {len(skipped)} \\input fragment(s), no \\documentclass: "
              f"{', '.join(skipped)})")

    if not all_problems:
        print(f"\u2713 All {checked} standalone LaTeX document(s) compiled cleanly")
        return 0

    print(f"\n\u2717 {len(all_problems)} file(s) failed to compile cleanly:\n")
    for filepath, problems in sorted(all_problems.items()):
        print(f"{filepath}:")
        for problem in problems:
            print(f"  - {problem}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
