# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Discover which OpenCV contrib submodules are available.

Two discovery modes:
  - ``discover_contrib_modules(cv2_dir)``: scans the cv2 package directory
    for subdirectories that carry a ``__init__.pyi`` stub (used by the
    registry generator which needs type information from stubs).
  - ``available_at_runtime()``: imports cv2 and probes ``hasattr(cv2, name)``
    for every known contrib module (used by the doc generators which only
    need to know whether a module is compiled into the binary).

Both functions return sorted lists/sets so callers get deterministic output.
"""

import os
import sys

# Subdirectories inside the cv2 package that are NOT contrib modules.
# The generator must never treat these as contrib submodules.
_NON_MODULE_DIRS = frozenset({
    "__pycache__",
    "cv2",           # nested cv2 package (the old layout)
    "cuda",          # opencv.cuda namespace, not a standalone module
    "gapi",          # Graph API, always present
    "infra",         # infrastructure helpers
    "ipp",           # Intel IPP integration
    "ml",            # machine-learning helpers, always bundled
    "ogl",           # OpenGL helpers
    "ovis",          # OGRE visualization (rare)
    "python-3.13",   # Python versioned subdir with .pyd
    "samples",       # sample data, not a module
    "typing",        # type annotations helper
    "utils",         # utility helpers
    "videoio_registry",
    # --- non-module directories that happen to live in the cv2 tree ---
    "tests",
})

# All contrib module names the project knows about.  Used for documentation
# and as a fallback when runtime discovery is not possible.  The actual
# inclusion is always driven by what the installed OpenCV exposes.
KNOWN_CONTRIB = [
    "alphamat",
    "aruco",
    "barcode",
    "bgsegm",
    "bioinspired",
    "ccm",
    "csrt",
    "datasets",
    "detail",
    "dnn_superres",
    "dpm",
    "face",
    "fisheye",
    "ft",                   # source-tree name: fuzzy
    "hfs",
    "img_hash",
    "intensity_transform",
    "line_descriptor",
    "linemod",
    "mcc",
    "motempl",
    "multicalib",
    "omnidir",
    "optflow",
    "phase_unwrapping",
    "plot",
    "ppf_match_3d",
    "quality",
    "rapid",
    "reg",
    "rgbd",
    "saliency",
    "sfm",
    "signal",
    "stereo",
    "structured_light",
    "text",
    "tracking",
    "videostab",
    "wechat_qrcode",
    "xfeatures2d",
    "ximgproc",
    "xobjdetect",
    "xphoto",
    "xstereo",
]


def discover_contrib_modules(cv2_dir: str) -> list[str]:
    """Return sorted list of contrib modules that have ``__init__.pyi`` stubs.

    This scans ``cv2_dir`` for subdirectories containing a ``__init__.pyi``
    file.  The stub is the type-information source the registry generator
    needs to build ``registry_contrib.py``.

    Missing stubs are not an error: contrib submodules only ship with
    ``opencv-contrib-python(-headless)``; a core-only install simply produces
    an empty list.
    """
    modules = []
    if not os.path.isdir(cv2_dir):
        return modules
    for name in sorted(os.listdir(cv2_dir)):
        if name in _NON_MODULE_DIRS:
            continue
        pyi = os.path.join(cv2_dir, name, "__init__.pyi")
        if os.path.isfile(pyi):
            modules.append(name)
    return modules


def available_at_runtime() -> set[str]:
    """Return set of contrib module names the running cv2 binary exposes.

    Imports ``cv2`` and probes ``hasattr(cv2, name)`` for every module in
    ``KNOWN_CONTRIB``.  Returns only modules that actually exist at runtime.

    This is used by the doc generators (``extract_doc_args.py``,
    ``apply_tooltips.py``) which don't need stubs — they only need to know
    whether a module is compiled into the binary so they can skip
    unavailable modules.
    """
    try:
        import cv2  # noqa: F811
    except ImportError:
        print("WARNING: cv2 not importable — cannot auto-detect modules",
              file=sys.stderr)
        return set()
    return {name for name in KNOWN_CONTRIB if hasattr(cv2, name)}
