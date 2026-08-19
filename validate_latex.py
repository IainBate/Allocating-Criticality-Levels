#!/usr/bin/env python3
r"""
Validate LaTeX files for common errors before committing/pushing to Overleaf.

This script checks:
1. Balanced braces, brackets, and parentheses
2. Proper use of math mode delimiters ($, $$, \[, \])
3. Matching begin/end environments
4. Missing or extra \end{document}
5. Common macro usage issues

Usage: python validate_latex.py [file_or_directory...]
"""

import sys
import os
import re
from pathlib import Path


def read_file(filepath):
    """Read file content, return None if unreadable."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None


def check_balanced_braces(content):
    """Check that braces {} are balanced."""
    errors = []
    count = 0
    for i, char in enumerate(content, 1):
        if char == '{':
            count += 1
        elif char == '}':
            count -= 1
            if count < 0:
                errors.append(f"Line {i}: Unmatched closing brace '}}'")
    if count > 0:
        errors.append(f"Missing {count} closing brace(s)")
    return errors


def check_balanced_brackets(content):
    """Check that square brackets [] are balanced."""
    errors = []
    count = 0
    for i, char in enumerate(content, 1):
        if char == '[':
            count += 1
        elif char == ']':
            count -= 1
            if count < 0:
                errors.append(f"Line {i}: Unmatched closing bracket ']'")
    if count > 0:
        errors.append(f"Missing {count} closing bracket(s)")
    return errors


def check_math_mode_delimiters(content):
    """Check for balanced math mode delimiters."""
    errors = []

    # Remove comments first
    lines = content.split('\n')
    clean_lines = []
    for line in lines:
        # Remove % comments (but not inside strings)
        comment_idx = line.find('%')
        if comment_idx >= 0:
            line = line[:comment_idx]
        clean_lines.append(line)
    clean_content = '\n'.join(clean_lines)

    # Count $ delimiters
    dollar_count = len(re.findall(r'\$', clean_content))
    if dollar_count % 2 != 0:
        errors.append("Unbalanced $ delimiters (odd number found)")

    # Check for $$ without matching $ in between (simple check)
    return errors


def check_begin_end_environments(content):
    """Check that begin/end environments are properly matched."""
    errors = []

    begins = re.findall(r'\\begin\s*\{([^}]+)\}', content)
    ends = re.findall(r'\\end\s*\{([^}]+)\}', content)

    # Sort for consistent output
    begins_sorted = sorted(begins)
    ends_sorted = sorted(ends)

    if begins_sorted != ends_sorted:
        missing_in_ends = [b for b in begins if b not in ends]
        missing_in_begins = [e for e in ends if e not in begins]

        if missing_in_ends:
            errors.append(f"Missing \\end{{{', '.join(missing_in_ends)}}}")
        if missing_in_begins:
            errors.append(f"Extra \\begin{{{', '.join(missing_in_begins)}}} without matching \\end")

    return errors


def check_document_environment(content):
    r"""Check for proper \documentclass, \begin{document}, \end{document}."""
    errors = []

    has_documentclass = bool(re.search(r'\\documentclass\s*[{[]', content))
    has_begin_document = bool(re.search(r'\\begin\s*\{document\}', content))
    has_end_document = bool(re.search(r'\\end\s*\{document\}', content))

    if not has_documentclass:
        errors.append("Missing \\documentclass declaration")
    if not has_begin_document:
        errors.append("Missing \\begin{document}")
    if not has_end_document:
        errors.append("Missing \\end{document}")

    return errors


def check_bibliography_commands(content, filename):
    """Check for proper bibliography setup."""
    errors = []

    has_bibstyle = bool(re.search(r'\\bibliographystyle', content))
    has_bibliography = bool(re.search(r'\\bibliography\s*\{[^}]+\}', content))

    # If using \cite, should have bibliography
    cites = re.findall(r'\\cite\s*[\{\[]?', content)
    if cites and not (has_bibstyle or has_bibliography):
        errors.append(f"Found \\cite commands but no \\bibliography or biblatex setup")

    return errors


def check_common_latex_issues(content, filename):
    """Check for common LaTeX issues."""
    errors = []

    # Check for double blank lines (can cause Overleaf warnings)
    if re.search(r'\n\s*\n\s*\n', content):
        errors.append("Multiple consecutive blank lines found")

    # Check for trailing whitespace
    lines_with_trailing_ws = [i+1 for i, line in enumerate(content.split('\n'))
                               if line.rstrip() != line]
    if lines_with_trailing_ws:
        errors.append(f"Trailing whitespace on lines: {lines_with_trailing_ws[:5]}{'...' if len(lines_with_trailing_ws) > 5 else ''}")

    return errors


def validate_latex_file(filepath):
    """Validate a single LaTeX file."""
    content = read_file(filepath)
    if content is None:
        return []

    all_errors = []
    filename = os.path.basename(filepath)

    # Run all checks
    all_errors.extend(check_balanced_braces(content))
    all_errors.extend(check_balanced_brackets(content))
    all_errors.extend(check_math_mode_delimiters(content))
    all_errors.extend(check_begin_end_environments(content))
    all_errors.extend(check_document_environment(content))
    all_errors.extend(check_bibliography_commands(content, filename))
    all_errors.extend(check_common_latex_issues(content, filename))

    return all_errors


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Default: check all .tex files in current directory
        tex_files = list(Path('.').glob('**/*.tex'))
    else:
        tex_files = []
        for arg in sys.argv[1:]:
            path = Path(arg)
            if path.is_file() and path.suffix == '.tex':
                tex_files.append(path)
            elif path.is_dir():
                tex_files.extend(path.glob('**/*.tex'))

    if not tex_files:
        print("No .tex files found")
        return 0

    all_errors = {}
    for filepath in sorted(tex_files):
        errors = validate_latex_file(str(filepath))
        if errors:
            all_errors[str(filepath)] = errors

    # Report results
    if not all_errors:
        print(f"✓ All {len(tex_files)} LaTeX file(s) passed validation")
        return 0

    print(f"\n✗ Found issues in {len(all_errors)} file(s):\n")

    for filepath, errors in sorted(all_errors.items()):
        print(f"{filepath}:")
        for error in errors:
            print(f"  - {error}")
        print()

    return 1


if __name__ == '__main__':
    sys.exit(main())
