# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# Adapted from geroldmeisinger/opencv-comfyui (GPL-3.0).
# See LICENSE for the full GNU GPL v3 text.

"""Generates opencv_nodes/registry.py from the cv2 type stubs (__init__.pyi).

The registry is a compact list of function descriptions. The actual ComfyUI V3
node classes are built at import time by opencv_nodes/lowlevel.py.

Run from the repository root:

    python generator/generator.py

Improvements over the old generator (which emitted full V1 classes):
* emits metadata instead of code (small, reviewable, version-controlled)
* deduplicates UMat overloads (same signature, different mat type)
* drops optional `dst` out-parameters entirely (ComfyUI has no in-place outputs)
* skips functions with unsupported parameter or return types instead of
  emitting broken classes
* blacklists, at GENERATION time and in ONE place (BLACKLIST = NONFREE |
  NONSAFE), everything that must never become a node

It also scans the cv2 SUBMODULES whose stubs ship next to __init__.pyi
(cv2/xphoto/__init__.pyi, cv2/ximgproc/__init__.pyi, ...). Those functions are
registered under their dotted name ("xphoto.oilPainting") so lowlevel.py can
resolve them with a nested getattr and build one node per function, exactly as
for the top-level ones. Contrib submodules are only present in
opencv-contrib-python(-headless) builds; lowlevel.py skips any entry the installed build
does not actually expose, so a core-only install simply gets fewer nodes.
"""

import ast
import os
from typing import List, Tuple, TypedDict

try:
	from .discover_modules import discover_contrib_modules
except ImportError:
	import sys as _sys
	_sys.path.insert(0, os.path.dirname(__file__))
	from discover_modules import discover_contrib_modules

class FuncInfo(TypedDict):
	name: str
	params: List[Tuple[str, str, bool]]  # (name, type, has_default)
	return_type: str

image_types = [
	"UMat",
	"cv2.typing.MatLike",
	"cv2.cuda.GpuMat",
	"numpy.ndarray",
]

int_types = [
	"int",
	"DrawMatchesFlags",
	"HandEyeCalibrationMethod",
	"RobotWorldHandEyeCalibrationMethod",
	"SolvePnPMethod",
	"AlgorithmHint",
]

# Composite cv2.typing.* types mapped to a distinct registry kind so the
# node layer can render each ergonomically (Point -> x/y ints, Size -> w/h,
# Scalar -> components, Rect -> x/y/w/h) instead of one free-text tuple widget.
# Every entry MUST map to a non-None kind: classify() returning None drops the
# whole function, so an unmapped composite would silently remove nodes.
# lowlevel.py falls back to the legacy literal-string rendering for any kind it
# does not (yet) split, so enriching these here is behavior-neutral on its own.
literal_types = {
	"cv2.typing.Point": "point2i",
	"cv2.typing.Point2d": "point2f",
	"cv2.typing.Point2f": "point2f",
	"cv2.typing.Size": "size",
	"cv2.typing.Scalar": "scalar",
	"cv2.typing.Rect": "rect",
	"cv2.typing.Rect2d": "rect",
	"cv2.typing.RotatedRect": "rotatedrect",
	"cv2.typing.TermCriteria": "termcriteria",
	"cv2.typing.Moments": "moments",
	"cv2.typing.Range": "range",
}

# ---------------------------------------------------------------------------
# What never becomes a node.
#
# ALL exclusion happens HERE, at generation time: a blacklisted function is
# simply absent from the registry, so no other file has to know about it.
# BLACKLIST is not written by hand - it is the union of exactly two small,
# separately-justified lists:
#
#   NONFREE : compiled out unless the build sets OPENCV_ENABLE_NONFREE, which
#             the official wheels do not. Every call raises "(-213) This
#             algorithm is patented and is excluded in this configuration".
#   NONSAFE : would hang, crash or over-reach a headless ComfyUI server - GUI
#             windows / trackbars / the event loop, or an arbitrary HOST
#             FILESYSTEM PATH (a downloaded workflow could then read any file
#             on the machine).
#
# A node that can only ever raise, or that reads the user's disk, is worse than
# no node at all - that is the whole membership test. Anything else stays.
#
# Names are QUALIFIED ("xphoto.bm3dDenoising"); for a top-level cv2 function the
# qualified name IS the bare name. The qualification is load-bearing: `inpaint`
# is both cv2.inpaint and cv2.xphoto.inpaint, two different algorithms whose
# masks mean OPPOSITE things, so a bare-name blacklist could not tell them apart.
#
# Compile-flag gating is NOT automatically a reason to blacklist - see
# EIGEN_GATED below, which deliberately stays exposed.
# The compile-flag survey these lists were built from is
# tools/compile_flag_gated_list.txt (source line references into the
# opencv_contrib 5.x tree).
# ---------------------------------------------------------------------------

# OPENCV_ENABLE_NONFREE gated. Line references are into opencv_contrib-5.x.
NONFREE = {
	# bm3d_image_denoising.cpp:305 and :341 (both overloads).
	"xphoto.bm3dDenoising",
	# tonemap.cpp:123. A create* factory returning Ptr<TonemapDurand>, so an
	# unsupported return type keeps it out of the registry anyway - listed so
	# the survey and this file agree.
	"xphoto.createTonemapDurand",
	# The rest of the survey is class-gated and out of the generator's reach
	# entirely (no free function to wrap): xfeatures2d SURF (surf.cpp:1026,
	# surf.cuda.cpp:51) and the rgbd volumetric trackers (kinfu.cpp:367,
	# colored_kinfu.cpp:400, large_kinfu.cpp:469, dynafu.cpp:528). rgbd is not
	# even in SUBMODULES.
}

# Unsafe on a headless server, in three flavours.
NONSAFE = {
	# --- windows, trackbars, the event loop: would block or crash the server.
	"imshow", "waitKey", "waitKeyEx", "pollKey",
	"namedWindow", "destroyWindow", "destroyAllWindows",
	"moveWindow", "resizeWindow", "setWindowTitle", "setWindowProperty",
	"getWindowProperty", "getWindowImageRect", "startWindowThread",
	"selectROI", "selectROIs", "setMouseCallback",
	"createButton", "createTrackbar", "getTrackbarPos", "setTrackbarPos",
	"setTrackbarMin", "setTrackbarMax", "displayOverlay", "displayStatusBar",
	# --- arbitrary host filesystem paths. In ComfyUI images arrive through
	# LoadImage, never through these, so the only thing they add is a
	# path-traversal surface a shared workflow could aim anywhere.
	# cv2.imdecode is deliberately NOT here: it decodes an in-memory buffer.
	"imread", "imreadmulti", "imcount",
	"haveImageReader", "haveImageWriter",
	"imwrite", "imwritemulti",
	"readOpticalFlow", "writeOpticalFlow",
	# cv2 5.0's 3D asset I/O (only loadPointCloud survives the type filter, the
	# rest are listed so the rule is stated once for the whole family).
	"loadPointCloud", "savePointCloud", "loadMesh", "saveMesh",
	# ground-truth disparity loader
	"ximgproc.readGT",
	# --- EMITS host filesystem paths rather than taking one. The build banner
	# is frozen at compile time and quotes the BUILD MACHINE's directories
	# (OPENCV_EXTRA_MODULES_PATH, the install prefix, every compiler and
	# third-party library it linked). For a PyPI wheel that machine is a
	# throwaway CI runner, but anyone who builds their own OpenCV - which is
	# exactly what the EIGEN_GATED nodes tell them to do - bakes their own home
	# directory, user name and project layout into it, and the whole point of
	# reading it is to paste it into a bug report. convert.BuildInfo
	# ("CV Build Information") is the replacement: same text, paths redacted.
	"getBuildInformation",
}

# The single table every code path consults.
BLACKLIST = NONFREE | NONSAFE

# Compile-flag gated like NONFREE, and deliberately NOT blacklisted.
#
# Eigen is a build-time dependency, not a licence: the SAME wheel API works the
# moment OpenCV is built with Eigen, so hiding these would make the pack lie
# about what OpenCV offers on a machine where they do run. They are generated as
# ordinary nodes; lowlevel.py reads this set (it is emitted into the registry
# files) and states the requirement in the node DESCRIPTION, checking
# cv2.getBuildInformation() so an Eigen-less build says so up front instead of
# failing with a bare -213 at execution time.
EIGEN_GATED = {
	# bimef.cpp:563 - one BIMEF_impl behind #ifdef HAVE_EIGEN; BIMEF2 is the
	# python name of the 6-argument overload of the same function.
	"intensity_transform.BIMEF",
	"intensity_transform.BIMEF2",
	# fbs_filter.cpp:637. Its class factory createFastBilateralSolverFilter
	# (fbs_filter.cpp:632) is Ptr<>-returning and never reaches the registry.
	"ximgproc.fastBilateralSolverFilter",
}

# In the cv2 stubs, DEFAULTED image-typed parameters are almost always in-place
# out-parameters (dst, edges, lines, circles, ...) whose result is also
# available through the return value. They have no equivalent in ComfyUI's
# dataflow model and only confused users, so they are dropped from the node
# inputs. The exception are mask parameters, which are genuine optional inputs.
#
# "Defaulted" is the test, NOT "nullable": the cv2 5 stubs annotate several
# REQUIRED array arguments as `MatLike | None` with no default (solvePnP's
# distCoeffs, projectPoints', undistort's, ...). You may pass None, but you must
# pass something - cv2 raises "Required argument 'distCoeffs' (pos 5) not
# found" otherwise. Reading those as optional dropped them from the node
# entirely, which is what a naive 4.12 -> 5.0 regeneration silently did.
def is_out_param(param_name: str) -> bool:
	return "mask" not in param_name.lower()


# A handful of cv2 functions return None and deliver their result ONLY by
# writing into a REQUIRED (non-optional) array argument - the whole
# cv2.intensity_transform API plus xphoto.dctDenoising / xphoto.inpaint.
# to_registry_entry() would drop them (return type "None"), leaving those
# modules completely unreachable. Instead they are registered with the
# out-parameter REMOVED from the inputs and added as an image RETURN, so the
# node looks like every other filter (src in, result out). lowlevel.py
# allocates the buffer (np.zeros_like of the reference input), passes it
# positionally and returns it.
#
# Value: (out_param_name, reference_param_name). The buffer is allocated with
# the shape and dtype of the reference input, which is what these functions
# require (verified against cv2 5.0.0: a (0, 0) placeholder raises).
INPLACE_OUT_PARAMS = {
	"intensity_transform.autoscaling": ("output", "input"),
	"intensity_transform.contrastStretching": ("output", "input"),
	"intensity_transform.gammaCorrection": ("output", "input"),
	"intensity_transform.logTransform": ("output", "input"),
	"xphoto.dctDenoising": ("dst", "src"),
	"xphoto.inpaint": ("dst", "src"),
}

# cv2 submodules to scan for free functions, in addition to the top-level
# namespace.  The list is discovered dynamically at generation time by
# scanning the cv2 package directory for subdirectories that carry an
# __init__.pyi stub (see discover_modules.discover_contrib_modules).
# Only modules whose functions make sense as image-processing nodes are
# included: GUI-only, dataset-loader and 3D-tracking modules are filtered
# out by the NON_MODULE_DIRS exclusion set in discover_modules.py, as are
# `utils` (binding self-tests) and `gapi` (a graph API that needs its own
# execution model).
#
# The variable is set to the dynamic list by parse_submodules() before
# generation.  For the static list of ALL known contrib modules (used for
# documentation and fallback), see discover_modules.KNOWN_CONTRIB.
SUBMODULES: list[str] = []

def strip_optional(t: str) -> tuple[str, bool]:
	if t.endswith(" | None"):
		return t[: -len(" | None")], True
	return t, False


def classify(t: str) -> str | None:
	"""Map a .pyi type string to a registry kind, or None if unsupported."""
	base, _ = strip_optional(t)
	# The cv2 5 stubs use unions for arguments that accept several equivalent
	# spellings of the same thing - inRange's `MatLike | Scalar`, which 4.12
	# spelled `MatLike`. Take the first alternative we support (the stubs list
	# the array form first, and an array is what the pack's NPARRAY sockets
	# carry anyway). Rejecting the whole union instead dropped inRange - a node
	# two shipped workflows use - out of the registry without a word.
	if " | " in base:
		for alt in base.split(" | "):
			kind = classify(alt)
			if kind is not None:
				return kind
		return None
	for it in image_types:
		if base == it:
			return "image"
	if base in int_types:
		return "int"
	if base == "float":
		return "float"
	if base == "bool":
		return "bool"
	if base == "str":
		return "str"
	if base == "_typing.Any":
		return "any"
	if base in literal_types:
		return literal_types[base]
	return None


def split_toplevel(s: str) -> List[str]:
	"""Split 'a, b[c, d], e' on top-level commas only."""
	parts, depth, cur = [], 0, ""
	for ch in s:
		if ch in "[(":
			depth += 1
		elif ch in "])":
			depth -= 1
		if ch == "," and depth == 0:
			parts.append(cur.strip())
			cur = ""
		else:
			cur += ch
	if cur.strip():
		parts.append(cur.strip())
	return parts


def parse_functions(content: str) -> List[FuncInfo]:
	tree = ast.parse(content)
	functions: List[FuncInfo] = []

	for node in ast.iter_child_nodes(tree):
		if not isinstance(node, ast.FunctionDef):
			continue

		args = node.args.args
		n_defaults = len(node.args.defaults)
		first_default_idx = len(args) - n_defaults

		params = []
		for i, arg in enumerate(args):
			param_type = ast.unparse(arg.annotation) if arg.annotation else "Any"
			params.append((arg.arg, param_type, i >= first_default_idx))

		return_type = ast.unparse(node.returns) if node.returns else "Any"
		functions.append(FuncInfo(name=node.name, params=params, return_type=return_type))

	return functions


def to_registry_entry(func_info: FuncInfo, module: str = ""):
	"""Convert one overload to (params, returns) in registry form, or None if
	unsupported. `module` is the submodule the function came from ("" for the
	top-level cv2 namespace) and only selects the INPLACE_OUT_PARAMS rule."""
	name = func_info["name"]
	if name.startswith("_"):
		return None
	# One table, keyed on the qualified name (a top-level function's qualified
	# name is its bare name), built from NONFREE + NONSAFE. See BLACKLIST.
	qualified = f"{module}.{name}" if module else name
	if qualified in BLACKLIST:
		return None
	inplace = INPLACE_OUT_PARAMS.get(qualified)

	params = []
	for param_name, param_type, has_default in func_info["params"]:
		kind = classify(param_type)
		if kind is None:
			return None
		# `T | None` with NO default is a REQUIRED argument that merely accepts
		# None - not an optional one. See is_out_param().
		optional = has_default
		if inplace is not None and param_name == inplace[0]:
			# result-only buffer: lowlevel.py allocates and returns it
			continue
		if kind == "image" and optional and is_out_param(param_name):
			continue  # drop out-parameter
		params.append((param_name, kind, 1 if optional else 0))

	rt = func_info["return_type"]
	if rt == "None" and inplace is not None:
		# the result travels through the out-parameter we just removed
		if not any(p[0] == inplace[1] for p in params):
			return None  # reference input vanished - cannot size the buffer
		return (params, ["image"])
	if rt in ("None", "Any"):
		return None
	if rt.replace(" ", "") == "tuple[float,float,float,float]":
		# A cv2 Scalar. The 4.12 stubs returned `cv2.typing.Scalar` here and the
		# registry kind was the composite 'scalar' (one output, repr'd as a
		# literal string); the 5.0 stubs inline the alias, and splitting the
		# tuple would turn mean/sumElems/trace into FOUR float outputs - a
		# silent re-slotting of every saved link into those nodes.
		return (params, ["scalar"])
	iter_types = split_toplevel(rt[len("tuple["):-1]) if rt.startswith("tuple[") else [rt]
	returns = []
	for it in iter_types:
		kind = classify(it)
		if kind is None:
			return None
		returns.append(kind)

	return (params, returns)


def normalized_signature(params, returns):
	"""Signature key that ignores parameter order-irrelevant details so that
	UMat/MatLike overload twins collapse into one entry."""
	return (
		tuple((n, k, o) for n, k, o in params),
		tuple(returns),
	)


def collect(func_infos: List[FuncInfo], module: str, by_name: dict) -> int:
	"""Fold one module's overloads into `by_name`, returning the skipped count.

	Functions are keyed by their QUALIFIED name ("xphoto.oilPainting"), which is
	also how lowlevel.py resolves them (nested getattr) and how it derives node
	ids, so a submodule function can never collide with a top-level one of the
	same name (xphoto.inpaint vs cv2.inpaint - two very different functions with
	OPPOSITE mask conventions)."""
	skipped = 0
	for fi in func_infos:
		entry = to_registry_entry(fi, module)
		if entry is None:
			skipped += 1
			continue
		params, returns = entry
		qualified = f"{module}.{fi['name']}" if module else fi["name"]
		if qualified in INPLACE_OUT_PARAMS:
			# remember WHERE cv2 wants the buffer: the out-parameter's index in
			# the original signature (1 for intensity_transform.*, 2 for
			# xphoto.inpaint). lowlevel.py inserts it back at that slot.
			out_name, ref_name = INPLACE_OUT_PARAMS[qualified][:2]
			index = next(i for i, p in enumerate(fi["params"]) if p[0] == out_name)
			INPLACE_OUT_PARAMS[qualified] = (out_name, ref_name, index)
		sigs = by_name.setdefault(qualified, [])
		key = normalized_signature(params, returns)
		if any(normalized_signature(p, r) == key for p, r in sigs):
			continue  # duplicate overload (e.g. UMat twin)
		sigs.append((params, returns))
	return skipped


def _eigen_subset(by_name: dict) -> list:
	"""The EIGEN_GATED names actually present in THIS registry file, sorted.

	Each file carries only its own, so a core-only regeneration cannot drop the
	contrib file's set (or vice versa) and lowlevel.py can simply union the two.
	Sorted, and emitted as set(<list>): a bare set literal's repr order depends
	on string hashing, i.e. on PYTHONHASHSEED, which would churn the file on
	every regeneration."""
	return [n for n in sorted(EIGEN_GATED) if n in by_name]


def _entry_lines(by_name: dict, var_name: str):
	lines = [f"{var_name} = ["]
	count = 0
	for name in sorted(by_name):
		variants = by_name[name]
		for i, (params, returns) in enumerate(variants):
			lines.append(f"\t({name!r}, {i}, {len(variants)}, {params!r}, {returns!r}),")
			count += 1
	lines.append("]")
	return lines, count


# Kept in sync with tools/add_license_headers.py, which stamps the same block on
# the hand-written sources (and on these registries, so a checkout without a
# regeneration still carries it). Change one, change the other.
_HEADER = [
	"# SPDX-License-Identifier: GPL-3.0-only",
	"# Copyright (C) 2026 bmad4ever",
	"# Adapted from geroldmeisinger/opencv-comfyui (GPL-3.0).",
	"# See LICENSE for the full GNU GPL v3 text.",
	"",
	"# Auto-generated by generator/generator.py -- do not edit by hand.",
]
_EIGEN_DOC = [
	"# Functions that OpenCV compiles out unless it was built WITH EIGEN. They",
	"# are exposed on purpose (a build that has Eigen runs them from this same",
	"# API); lowlevel.py adds the requirement to the node description, checking",
	"# cv2.getBuildInformation() so an Eigen-less build says so up front instead",
	"# of raising a bare (-213) at execution time. Generator source of truth:",
	"# generator/generator.py:EIGEN_GATED.",
]
_FORMAT_DOC = [
	"# Each entry: (func_name, variant_index, n_variants, params, returns)",
	"#   params : list of (param_name, kind, optional) with kind in",
	"#            {'image', 'int', 'float', 'bool', 'str', 'any', and the",
	"#            composite kinds 'point2i', 'point2f', 'size', 'scalar',",
	"#            'rect', 'rotatedrect', 'termcriteria', 'moments', 'range'}",
	"#   returns: list of kinds",
]


def write_registry(by_name: dict, out_path: str, cv2_version: str) -> int:
	lines = list(_HEADER)
	lines.append(f"# Generated from the type stubs of cv2 {cv2_version}.")
	lines.append("#")
	lines.extend(_FORMAT_DOC)
	lines.append("")
	entries, count = _entry_lines(by_name, "FUNCTIONS")
	lines.extend(entries)
	lines.append("")
	lines.extend(_EIGEN_DOC)
	lines.append(f"EIGEN_GATED = set({_eigen_subset(by_name)!r})")
	lines.append("")
	with open(out_path, "w", newline="\n") as f:
		f.write("\n".join(lines))
	return count


def write_contrib_registry(by_name: dict, out_path: str, cv2_version: str,
                           submodules: list[str] | None = None) -> int:
	"""Emit the SUBMODULE entries into their own file.

	Kept separate from registry.py on purpose. registry.py was generated from an
	older OpenCV (its top-level signatures are what every saved workflow's
	widget order was built against); re-emitting it against a newer cv2 silently
	adds/removes parameters (cv2 5 dropped `distCoeffs` from solvePnP/undistort
	and added `hint` to warpAffine/remap), which would misalign the stored
	widgets_values of existing workflows. Writing the submodules to a second
	file makes exposing them strictly ADDITIVE: no existing node changes shape.
	"""
	if submodules is None:
		submodules = SUBMODULES
	lines = list(_HEADER)
	lines.append(f"# Generated from the cv2 {cv2_version} SUBMODULE type stubs")
	lines.append(f"# ({', '.join(submodules)}).")
	lines.append("#")
	lines.append("# Same entry format as registry.py, except func_name is DOTTED")
	lines.append("# ('xphoto.oilPainting') and lowlevel.py resolves it with a nested")
	lines.append("# getattr on cv2. Most of these modules only exist in")
	lines.append("# opencv-contrib-python(-headless) builds; lowlevel.py skips any entry the")
	lines.append("# installed build does not expose, so a core-only install just gets")
	lines.append("# fewer nodes instead of failing to import.")
	lines.append("#")
	lines.extend(_FORMAT_DOC)
	lines.append("")
	entries, count = _entry_lines(by_name, "CONTRIB_FUNCTIONS")
	lines.extend(entries)
	lines.append("")
	lines.append("# Functions whose result is delivered by writing into an argument")
	lines.append("# instead of being returned (they return None). The out-parameter is")
	lines.append("# NOT an input above; lowlevel.py allocates a buffer shaped like the")
	lines.append("# reference input, passes it positionally in the out-parameter's")
	lines.append("# original slot, and returns it as the node's image output.")
	lines.append("#   {qualified_name: (out_param_name, reference_param_name)}")
	lines.append(f"INPLACE_OUT_PARAMS = {INPLACE_OUT_PARAMS!r}")
	lines.append("")
	lines.extend(_EIGEN_DOC)
	lines.append(f"EIGEN_GATED = set({_eigen_subset(by_name)!r})")
	lines.append("")
	with open(out_path, "w", newline="\n") as f:
		f.write("\n".join(lines))
	return count


def generate(func_infos: List[FuncInfo], out_path: str, cv2_version: str):
	"""Write registry.py from the TOP-LEVEL cv2 overloads."""
	by_name: dict[str, list] = {}
	skipped = collect(func_infos, "", by_name)
	count = write_registry(by_name, out_path, cv2_version)
	print(f"{len(func_infos)} overloads parsed, {skipped} skipped (unsupported/blacklisted)")
	print(f"{count} node entries for {len(by_name)} functions written to {out_path}")


def generate_contrib(submodule_infos: dict, out_path: str, cv2_version: str):
	"""Write registry_contrib.py from a {module: [FuncInfo, ...]} mapping."""
	by_name: dict[str, list] = {}
	skipped = 0
	total = 0
	for module, infos in submodule_infos.items():
		skipped += collect(infos, module, by_name)
		total += len(infos)
	count = write_contrib_registry(by_name, out_path, cv2_version,
	                               submodules=list(submodule_infos.keys()))
	print(f"{total} submodule overloads parsed, {skipped} skipped "
	      "(unsupported/blacklisted)")
	print(f"{count} node entries for {len(by_name)} submodule functions "
	      f"written to {out_path}")


def parse_submodules(cv2_dir: str) -> dict:
	"""{module: [FuncInfo, ...]} for every SUBMODULE whose stub is installed.

	Modules are discovered dynamically by scanning the cv2 package directory
	for subdirectories with ``__init__.pyi`` stubs.  A missing stub is not
	an error: contrib submodules only ship with
	opencv-contrib-python(-headless), and a core-only install should still
	regenerate."""
	global SUBMODULES
	SUBMODULES = discover_contrib_modules(cv2_dir)
	if not SUBMODULES:
		print("  (no contrib modules with .pyi stubs found)")
		return {}
	out = {}
	for module in SUBMODULES:
		pyi = os.path.join(cv2_dir, *module.split("."), "__init__.pyi")
		with open(pyi, "r", encoding="utf-8") as f:
			out[module] = parse_functions(f.read())
	return out


if __name__ == "__main__":
	import argparse
	import cv2

	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--target", choices=["core", "contrib", "both"],
	                default="contrib",
	                help="which registry to (re)generate. 'contrib' (the "
	                     "default) only rewrites registry_contrib.py, which is "
	                     "purely additive. 'core' rewrites registry.py against "
	                     "the INSTALLED OpenCV - only do that deliberately as "
	                     "part of an OpenCV upgrade, and re-check every saved "
	                     "workflow: cv2 changes top-level signatures between "
	                     "releases (5.0 dropped solvePnP/undistort's distCoeffs "
	                     "and added a 'hint' to warpAffine/remap), which shifts "
	                     "stored widgets_values.")
	args = ap.parse_args()

	cv2_dir = os.path.dirname(cv2.__file__)
	nodes_dir = os.path.join(os.path.dirname(__file__), "..", "opencv_nodes")
	os.makedirs(nodes_dir, exist_ok=True)

	if args.target in ("core", "both"):
		pyi_path = os.path.join(cv2_dir, "__init__.pyi")
		if not os.path.exists(pyi_path):
			raise FileNotFoundError(f'OpenCV .pyi file not found at "{pyi_path}"')
		with open(pyi_path, "r", encoding="utf-8") as f:
			content = f.read()
		generate(parse_functions(content),
		         os.path.join(nodes_dir, "registry.py"), cv2.__version__)

	if args.target in ("contrib", "both"):
		generate_contrib(parse_submodules(cv2_dir),
		                 os.path.join(nodes_dir, "registry_contrib.py"),
		                 cv2.__version__)
