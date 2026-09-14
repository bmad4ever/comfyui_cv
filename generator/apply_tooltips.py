# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Read CSVs of extracted OpenCV param/return docs and emit Python dict literals.

Reads a params CSV (``func,param,tooltip``) and optionally a returns CSV
(``func,tooltip``), both produced by ``extract_doc_args.py``, and emits Python
dict literals matching the format of ``PARAM_TOOLTIPS`` / ``RETURN_DOCS`` in
``opencv_nodes/lowlevel.py``.

Usage (from the repo root)::

    python generator/apply_tooltips.py \\
        temp/task_smart_tooltips/doc_args.csv \\
        --returns temp/task_smart_tooltips/return_docs.csv \\
        -o opencv_nodes/param_docs.py --var PARAM_DOCS \\
        --returns-out opencv_nodes/return_docs.py --returns-var RETURN_DOCS

Filter out unavailable modules and rename source-tree names to Python names::

    python generator/apply_tooltips.py \\
        temp/task_smart_tooltips/doc_args_new.csv \\
        --returns temp/task_smart_tooltips/return_docs_new.csv \\
        --exclude-modules cudaarithm,cudabgsegm,cudacodec,... \\
        --rename-modules fuzzy=ft \\
        -o opencv_nodes/param_docs.py --var PARAM_DOCS \\
        --returns-out opencv_nodes/return_docs.py --returns-var RETURN_DOCS
"""

import argparse
import csv
import os
import re
import sys


def _escape(s: str) -> str:
    """Escape a string for use inside a Python double-quoted string literal."""
    return s.replace('\\', '\\\\').replace('"', '\\"')


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


# Both emitted files hold prose lifted from OpenCV's own documentation, so the
# header credits that rather than claiming a blanket copyright over the strings.
# Kept in sync with tools/add_license_headers.py (DOC_DERIVED).
_HEADER = [
    "# SPDX-License-Identifier: GPL-3.0-only",
    "# Copyright (C) 2026 bmad4ever",
    "# Tooltip text extracted from the OpenCV documentation (Apache-2.0);",
    "# copyright of that text remains with the OpenCV contributors.",
    "# See LICENSE for the full GNU GPL v3 text.",
    "",
]


def _emit_dict(var_name: str, rows: list[tuple]) -> str:
    """Emit a Python dict literal from a list of (key..., value) tuples."""
    lines = list(_HEADER)
    lines.append(f"{var_name} = {{")
    if len(rows) > 0 and len(rows[0]) == 2:
        # Single-key dict: func -> tooltip
        for key, tooltip in rows:
            escaped = _escape(tooltip)
            lines.append(f'\t"{key}": "{escaped}",')
    else:
        # Two-key dict: (func, param) -> tooltip
        for func, param, tooltip in rows:
            escaped = _escape(tooltip)
            lines.append(f'\t("{func}", "{param}"): "{escaped}",')
    lines.append("}")
    lines.append("")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv_file', help='Input CSV (func,param,tooltip)')
    ap.add_argument('-o', '--output', help='Output .py file for param dict')
    ap.add_argument('--var', default='PARAM_DOCS',
                    help='Variable name for the param dict (default: PARAM_DOCS)')
    ap.add_argument('--returns', default=None,
                    help='Input CSV for return docs (func,tooltip)')
    ap.add_argument('--returns-out', default=None,
                    help='Output .py file for return dict')
    ap.add_argument('--returns-var', default='RETURN_DOCS',
                    help='Variable name for the return dict (default: RETURN_DOCS)')
    ap.add_argument('--only-new', action='store_true',
                    help='Exclude keys already present in lowlevel.py')
    ap.add_argument('--lowlevel', default=None,
                    help='Path to lowlevel.py (for --only-new dedup)')
    ap.add_argument('--exclude-modules', default='',
                    help='Comma-separated module prefixes to exclude '
                         '(e.g. cudaarithm,cudafilters,sfm)')
    ap.add_argument('--auto-detect', action='store_true', default=True,
                    help='Auto-detect which contrib modules are available at '
                         'runtime and filter out unavailable ones (default: enabled)')
    ap.add_argument('--no-auto-detect', action='store_false', dest='auto_detect',
                    help='Disable auto-detection; include all modules in CSV')
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
                  "(cv2 not importable?) — including all CSV entries",
                  file=sys.stderr)

    # Read params CSV
    rows = []
    with open(args.csv_file, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            func = row['func'].strip()
            param = row['param'].strip()
            tooltip = row['tooltip'].strip()
            if func and param and tooltip:
                rows.append((func, param, tooltip))

    # Apply module renames
    if renames:
        before = len(rows)
        rows = [(_apply_renames(f, renames), p, t) for f, p, t in rows]
        renamed = sum(1 for f, _p, _t in rows if _module_prefix(f) in renames.values())
        print(f"Renamed {renamed} entries via module renames", file=sys.stderr)

    # Exclude modules
    if excluded:
        before = len(rows)
        rows = [(f, p, t) for f, p, t in rows
                if _module_prefix(f) not in excluded]
        print(f"Excluded {before - len(rows)} entries for modules: "
              f"{', '.join(sorted(excluded))}", file=sys.stderr)

    # Auto-exclude modules not available at runtime
    if available_modules is not None:
        before = len(rows)
        rows = [(f, p, t) for f, p, t in rows
                if _module_prefix(f) is None or _module_prefix(f) in available_modules]
        print(f"Auto-excluded {before - len(rows)} entries for unavailable modules",
              file=sys.stderr)

    # Optionally filter out keys already in lowlevel.py
    existing = set()
    if args.only_new and args.lowlevel:
        with open(args.lowlevel, 'r', encoding='utf-8') as f:
            ll_content = f.read()
        for m in re.finditer(r'\("([^"]+)",\s*"([^"]+)"\)\s*:', ll_content):
            existing.add((m.group(1), m.group(2)))
        before = len(rows)
        rows = [(f, p, t) for f, p, t in rows if (f, p) not in existing]
        print(f"Filtered {before - len(rows)} param entries already in lowlevel.py",
              file=sys.stderr)

    # Emit param dict
    if args.output:
        output = _emit_dict(args.var, rows)
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
            f.write(output)
        print(f"Wrote {len(rows)} param entries to {args.output}", file=sys.stderr)
    else:
        print(_emit_dict(args.var, rows))

    # Read + emit return dict
    if args.returns and args.returns_out:
        ret_rows = []
        with open(args.returns, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                func = row['func'].strip()
                tooltip = row['tooltip'].strip()
                if func and tooltip:
                    ret_rows.append((func, tooltip))

        # Apply module renames
        if renames:
            ret_rows = [(_apply_renames(f, renames), t) for f, t in ret_rows]

        # Exclude modules
        if excluded:
            before = len(ret_rows)
            ret_rows = [(f, t) for f, t in ret_rows
                        if _module_prefix(f) not in excluded]
            print(f"Excluded {before - len(ret_rows)} return entries for modules: "
                  f"{', '.join(sorted(excluded))}", file=sys.stderr)

        # Auto-exclude modules not available at runtime
        if available_modules is not None:
            before = len(ret_rows)
            ret_rows = [(f, t) for f, t in ret_rows
                        if _module_prefix(f) is None or _module_prefix(f) in available_modules]
            print(f"Auto-excluded {before - len(ret_rows)} return entries for unavailable modules",
                  file=sys.stderr)

        # Filter existing
        if args.only_new and args.lowlevel:
            existing_ret = set()
            for m in re.finditer(r'"([^"]+)"\s*:\s*"', ll_content):
                existing_ret.add(m.group(1))
            before = len(ret_rows)
            ret_rows = [(f, t) for f, t in ret_rows if f not in existing_ret]
            print(f"Filtered {before - len(ret_rows)} return entries already in lowlevel.py",
                  file=sys.stderr)

        output = _emit_dict(args.returns_var, ret_rows)
        os.makedirs(os.path.dirname(args.returns_out) or '.', exist_ok=True)
        with open(args.returns_out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(output)
        print(f"Wrote {len(ret_rows)} return entries to {args.returns_out}",
              file=sys.stderr)


if __name__ == '__main__':
    main()
