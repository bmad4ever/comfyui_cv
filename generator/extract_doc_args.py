# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Scan OpenCV .hpp headers and extract @param doc comments.

Reads core and contrib source trees, finds every ``CV_EXPORTS_W`` function
declaration preceded by a ``/** ... */`` Doxygen block, extracts the
``@param <name> <description>`` lines (including multi-line continuations),
and emits a flat CSV:

    func,param,tooltip

where *func* is the bare C++ name (e.g. ``GaussianBlur``) and *param* is the
parameter name exactly as spelled in the declaration.

Usage (from the repo root)::

    python generator/extract_doc_args.py \\
        --core   temp/opencv-5.x \\
        --contrib temp/opencv_contrib-5.x \\
        -o temp/task_smart_tooltips/doc_args.csv

The two source trees are symlinked/junctioned under ``temp/``; adjust paths if
they move.

Filter out unavailable modules and rename source-tree names to Python names::

    python generator/extract_doc_args.py \\
        --core   temp/opencv-5.x \\
        --contrib temp/opencv_contrib-5.x \\
        --exclude-modules cudaarithm,cudabgsegm,... \\
        --rename-modules fuzzy=ft \\
        -o temp/task_smart_tooltips/doc_args.csv

Output can then be filtered/curated and fed to ``apply_tooltips.py`` to
produce a ``PARAM_TOOLTIPS`` dict snippet for ``opencv_nodes/lowlevel.py``.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# A /** ... */ block.  Dot matches everything including newlines.
_DOC_COMMENT_RE = re.compile(r'/\*\*\s*(.*?)\s*\*/', re.DOTALL)

# The function declaration that follows the doc comment.
# We capture the function name and the full parameter list (parentheses included).
_FUNC_DECL_RE = re.compile(
    r'CV_EXPORTS_W\s+'       # exported-to-Python marker
    r'(?:[\w:*&<> ]+\s+)+'   # return type (may have templates, spaces, pointers)
    r'(\w+)\s*'              # function name
    r'\(([^)]*)\)',          # parameter list
    re.DOTALL,
)

# Doxygen commands to strip from the description text.
_DOXY_COMMANDS = re.compile(
    r'@brief\s*|@param\s+\w+\s*|@return\s*|@sa\s*|@note\s*|'
    r'@see\s*|@tparam\s+\w+\s*|@throws\s+\w+\s*|'
    r'@overload\s*|@deprecated\s*|@code.*?@endcode|'
    r'@cite\s+\w+|@ref\s+[\w:]+|@anchor\s+\w+|@defgroup\s+\w+|'
    r'@\{|\@\}',
    re.DOTALL,
)

# LaTeX block fragments: \f[...\f] → convert to $$...$$ for KaTeX rendering
_LATEX_BLOCK_RE = re.compile(r'\\f\[(.*?)\\f\]', re.DOTALL)

# LaTeX inline fragments: \f$...\f$ → convert to $...$ for KaTeX rendering
_LATEX_INLINE_RE = re.compile(r'\\f\$(.*?)\\f\$', re.DOTALL)

# Markdown emphasis: _text_
_MD_EMPHASIS_RE = re.compile(r'(?<!\w)_([^_]+)_(?!\w)')

# HTML tags and links
_HTML_RE = re.compile(r'<[^>]+>|https?://\S+')

# Trailing whitespace
_TRAILING_WS_RE = re.compile(r'\s+$', re.MULTILINE)

# Multiple spaces
_MULTI_SPACE_RE = re.compile(r'  +')

# Doxygen commands that start a new block (not a continuation of @param)
_BLOCK_START_RE = re.compile(
    r'^\s*@(param|return|sa|note|see|brief|overload|'
    r'deprecated|code|endcode|tparam|throws|cite|'
    r'ref|anchor|defgroup|\{|\})\b')


def _clean_description(text: str) -> str:
    """Strip Doxygen/LaTeX/HTML markup and collapse whitespace."""
    text = _LATEX_BLOCK_RE.sub(r'$$\1$$', text)
    text = _LATEX_INLINE_RE.sub(r'$\1$', text)
    text = _HTML_RE.sub('', text)
    text = _DOXY_COMMANDS.sub('', text)
    text = _MD_EMPHASIS_RE.sub(r'\1', text)
    text = text.replace('\\f$', '').replace('\\f[', '').replace('\\f]', '')
    text = text.replace('\\<', '<').replace('\\>', '>').replace('\\\\', '\\')
    text = text.replace('\n', ' ')
    text = _MULTI_SPACE_RE.sub(' ', text)
    text = text.strip()
    text = _TRAILING_WS_RE.sub('', text)
    return text


def _extract_params(doc_text: str) -> list[tuple[str, str]]:
    """Return [(param_name, cleaned_description), ...] from a doc comment block."""
    results = []
    lines = doc_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'@param\s+(\w+)\s*(.*)', line)
        if m:
            name = m.group(1)
            desc_parts = [m.group(2).strip()] if m.group(2).strip() else []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if _BLOCK_START_RE.match(next_line):
                    break
                stripped = next_line.strip()
                if stripped:
                    desc_parts.append(stripped)
                i += 1
            desc = ' '.join(desc_parts)
            desc = _clean_description(desc)
            if desc:
                results.append((name, desc))
        else:
            i += 1
    return results


def _extract_return(doc_text: str) -> str | None:
    """Return the cleaned @return description from a doc comment block, or None."""
    lines = doc_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'@return\s+(.*)', line)
        if m:
            desc_parts = [m.group(1).strip()] if m.group(1).strip() else []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if _BLOCK_START_RE.match(next_line):
                    break
                stripped = next_line.strip()
                if stripped:
                    desc_parts.append(stripped)
                i += 1
            desc = ' '.join(desc_parts)
            desc = _clean_description(desc)
            return desc if desc else None
        else:
            i += 1
    return None


def _extract_functions_from_hpp(filepath: str, module: str = '') -> tuple[list[dict], list[dict]]:
    """Parse one .hpp file and return (param_entries, return_entries)."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except OSError:
        return []

    param_entries = []
    return_entries = []
    for doc_match in _DOC_COMMENT_RE.finditer(content):
        doc_text = doc_match.group(1)
        doc_end = doc_match.end()

        remainder = content[doc_end:doc_end + 500]
        func_match = _FUNC_DECL_RE.search(remainder)
        if not func_match:
            continue

        func_name = func_match.group(1)
        if func_name.startswith('operator') or func_name.startswith('~'):
            continue

        qualified = f"{module}.{func_name}" if module else func_name

        params = _extract_params(doc_text)
        for pname, tooltip in params:
            param_entries.append({
                'func': qualified,
                'param': pname,
                'tooltip': tooltip,
            })

        ret = _extract_return(doc_text)
        if ret:
            return_entries.append({
                'func': qualified,
                'tooltip': ret,
            })

    return param_entries, return_entries


def _discover_hpp_files(source_root: str) -> list[str]:
    """Find all relevant .hpp files under source_root/modules/."""
    files = []
    modules_dir = os.path.join(source_root, 'modules')
    if not os.path.isdir(modules_dir):
        print(f"WARNING: {modules_dir} not found, skipping", file=sys.stderr)
        return files

    for mod_dir in os.listdir(modules_dir):
        mod_path = os.path.join(modules_dir, mod_dir)
        if not os.path.isdir(mod_path):
            continue
        inc_dir = os.path.join(mod_path, 'include', 'opencv2')
        if not os.path.isdir(inc_dir):
            continue
        for root, _dirs, filenames in os.walk(inc_dir):
            for fn in filenames:
                if fn.endswith('.hpp'):
                    files.append(os.path.join(root, fn))
    return files


def _module_prefix(func_name: str) -> str | None:
    """Return the dotted module prefix of a func name, or None if bare."""
    dot = func_name.find('.')
    return func_name[:dot] if dot != -1 else None


def _apply_renames(func_name: str, renames: dict[str, str]) -> str:
    """Apply module prefix renames (e.g. fuzzy -> ft) to a func name."""
    prefix = _module_prefix(func_name)
    if prefix and prefix in renames:
        return renames[prefix] + func_name[len(prefix):]
    return func_name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--core', required=True,
                    help='Path to the opencv-5.x source tree (contains modules/)')
    ap.add_argument('--contrib', required=True,
                    help='Path to the opencv_contrib-5.x source tree (contains modules/)')
    ap.add_argument('-o', '--output', required=True,
                    help='Output CSV path for param docs')
    ap.add_argument('--returns', required=True,
                    help='Output CSV path for return docs')
    ap.add_argument('--exclude-modules', default='',
                    help='Comma-separated module prefixes to exclude '
                         '(e.g. cudaarithm,cudafilters,sfm)')
    ap.add_argument('--auto-detect', action='store_true', default=True,
                    help='Auto-detect which contrib modules are available at '
                         'runtime and skip unavailable ones (default: enabled)')
    ap.add_argument('--no-auto-detect', action='store_false', dest='auto_detect',
                    help='Disable auto-detection; process all source-tree modules')
    ap.add_argument('--rename-modules', default='',
                    help='Comma-separated old=new module renames '
                         '(e.g. fuzzy=ft)')
    args = ap.parse_args()

    # Parse --exclude-modules
    excluded = set()
    if args.exclude_modules:
        excluded = {m.strip() for m in args.exclude_modules.split(',') if m.strip()}

    # Parse --rename-modules
    renames = {}
    if args.rename_modules:
        for pair in args.rename_modules.split(','):
            pair = pair.strip()
            if '=' in pair:
                old, new = pair.split('=', 1)
                renames[old.strip()] = new.strip()

    # Auto-detect available contrib modules at runtime
    available_modules = None
    if args.auto_detect:
        try:
            from .discover_modules import available_at_runtime
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(__file__))
            from discover_modules import available_at_runtime
        available_modules = available_at_runtime()
        if available_modules:
            print(f"Auto-detected {len(available_modules)} available contrib modules: "
                  f"{', '.join(sorted(available_modules))}", file=sys.stderr)
        else:
            print("WARNING: auto-detect found no available contrib modules "
                  "(cv2 not importable?) — processing all source-tree modules",
                  file=sys.stderr)

    all_params = []
    all_returns = []

    # --- Core modules ---
    core_files = _discover_hpp_files(args.core)
    print(f"Core: scanning {len(core_files)} .hpp files ...", file=sys.stderr)
    for fp in core_files:
        p, r = _extract_functions_from_hpp(fp, module='')
        all_params.extend(p)
        all_returns.extend(r)

    # --- Contrib modules ---
    contrib_modules_dir = os.path.join(args.contrib, 'modules')
    if os.path.isdir(contrib_modules_dir):
        for mod_dir in sorted(os.listdir(contrib_modules_dir)):
            mod_path = os.path.join(contrib_modules_dir, mod_dir)
            if not os.path.isdir(mod_path):
                continue
            # Skip excluded modules
            if mod_dir in excluded:
                print(f"  contrib/{mod_dir}: SKIPPED (excluded)", file=sys.stderr)
                continue
            # Skip modules not available at runtime
            if available_modules is not None and mod_dir not in available_modules:
                print(f"  contrib/{mod_dir}: SKIPPED (not available at runtime)",
                      file=sys.stderr)
                continue
            # Apply module rename for the prefix used in CSV output
            csv_module = renames.get(mod_dir, mod_dir)
            inc_dir = os.path.join(mod_path, 'include', 'opencv2')
            if not os.path.isdir(inc_dir):
                continue
            files = []
            for root, _dirs, filenames in os.walk(inc_dir):
                for fn in filenames:
                    if fn.endswith('.hpp'):
                        files.append(os.path.join(root, fn))
            if files:
                print(f"  contrib/{mod_dir}: {len(files)} .hpp files"
                      f"{f' -> {csv_module}' if csv_module != mod_dir else ''}",
                      file=sys.stderr)
                for fp in files:
                    p, r = _extract_functions_from_hpp(fp, module=csv_module)
                    all_params.extend(p)
                    all_returns.extend(r)

    # Deduplicate params (same func+param from different overloads -> keep first)
    seen = set()
    unique_params = []
    for e in all_params:
        key = (e['func'], e['param'])
        if key not in seen:
            seen.add(key)
            unique_params.append(e)

    # Deduplicate returns (same func -> keep first)
    seen_ret = set()
    unique_returns = []
    for e in all_returns:
        if e['func'] not in seen_ret:
            seen_ret.add(e['func'])
            unique_returns.append(e)

    # Write param CSV
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['func', 'param', 'tooltip'])
        w.writeheader()
        for e in sorted(unique_params, key=lambda x: (x['func'], x['param'])):
            w.writerow(e)

    # Write return CSV
    with open(args.returns, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['func', 'tooltip'])
        w.writeheader()
        for e in sorted(unique_returns, key=lambda x: x['func']):
            w.writerow(e)

    print(f"\nWrote {len(unique_params)} param entries to {args.output}",
          file=sys.stderr)
    print(f"Wrote {len(unique_returns)} return entries to {args.returns}",
          file=sys.stderr)


if __name__ == '__main__':
    main()
