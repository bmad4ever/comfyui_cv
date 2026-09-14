# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Named enum groups for OpenCV integer constants.

Instead of forcing users to type magic integers (e.g. `6` for COLOR_BGR2GRAY),
nodes expose dropdowns with the constant names. Options are resolved against
the installed cv2 at load time, so they always match the user's OpenCV version.

Option strings may combine flags with `|` (e.g. "INTER_LINEAR | WARP_INVERSE_MAP").
"""

import cv2


class EnumGroup:
	"""A set of selectable cv2 constants for a Combo input."""

	def __init__(self, options, default=None, extra_values=None, module=None):
		"""
		options      : list of option strings. Each is either a cv2 constant
		               name, a `|`-combination of constant names, or a key of
		               extra_values.
		default      : option preselected in the UI (first existing one if None).
		extra_values : dict mapping non-constant labels to explicit int values
		               (e.g. {"same as input": -1}).
		module       : cv2 SUBMODULE the constants live in ("omnidir"). Needed
		               when the names also exist at cv2 top level with DIFFERENT
		               values - cv2.CALIB_FIX_SKEW is 1<<25 while
		               cv2.omnidir.CALIB_FIX_SKEW is 2, and resolving the wrong
		               one silently sets unrelated bits.
		"""
		self.module = module
		self.extra_values = extra_values or {}
		self.options = [o for o in options if self._exists(o)]
		if not self.options:
			self.options = ["<none available in this cv2 build>"]
		if default is not None and self._exists(default) and default in self.options:
			self.default = default
		else:
			self.default = self.options[0]

	def _namespace(self):
		return getattr(cv2, self.module) if self.module else cv2

	def _exists(self, option: str) -> bool:
		if option in self.extra_values:
			return True
		ns = self._namespace()
		if ns is None:
			return False
		return all(hasattr(ns, part.strip()) for part in option.split("|"))

	def resolve(self, option) -> int:
		"""Convert a selected option string back to its integer value."""
		if isinstance(option, int) and not isinstance(option, bool):
			# legacy workflows saved while the param was still a raw int widget
			# (e.g. warpPolar's flags before it became a FlagGroup)
			return option
		if option in self.extra_values:
			return self.extra_values[option]
		if not str(option).strip():
			raise ValueError(f'Empty flags value - expected constant names like "{self.default}"')
		value = 0
		for part in str(option).split("|"):
			part = part.strip()
			if part in self.extra_values:
				# an extra-value label composed with real flags, e.g. the
				# "none (0)" base of a pure-flag FlagGroup: "none (0) | DFT_SCALE"
				value |= int(self.extra_values[part])
				continue
			ns = self._namespace()
			if ns is None or not hasattr(ns, part):
				where = f"cv2.{self.module}" if self.module else "cv2"
				raise ValueError(f'Unknown OpenCV constant "{part}" in {where} '
				                 f'(cv2 {cv2.__version__})')
			value |= int(getattr(ns, part))
		return value


class FlagGroup(EnumGroup):
	"""One base option (mutually exclusive, e.g. the INTER_* method) plus any
	subset of independently OR-able flag bits (e.g. WARP_RELATIVE_MAP).

	The value crossing the graph stays the pipe-joined constant string
	("INTER_LINEAR | WARP_RELATIVE_MAP"), so resolve() is inherited unchanged
	and workflows saved when the param was a curated combo keep working - but
	the node input renders as a STRING carrying a `cv_flags` spec that
	web/cv_flags.js turns into a base dropdown plus one toggle per flag,
	instead of a combo enumerating every combination.

	Past web/cvlib/flag_search.js's LIST_THRESHOLD flags the frontend swaps
	those toggles for ONE one-row button that opens a searchable checkbox
	POPOVER - a group that grows to 26 options makes the node taller than it is
	placeable, and under "Modern Node Design" a tall widget cannot be shrunk at
	all. The value and the spec are identical either way; only the rendering
	differs.
	"""

	def __init__(self, base, flags, default=None, extra_values=None, initial=None,
	             module=None):
		"""
		base         : mutually exclusive options (may be extra_values labels,
		               e.g. a "none (0)" base for a pure-flag parameter).
		flags        : independently OR-able cv2 constant names (toggles).
		initial      : the widget's preset value; unlike `default` it may carry
		               pre-set flag toggles ("none (0) | CALIB_CB_..." style) to
		               match cv2's own default. Falls back to `default`.
		"""
		super().__init__(base, default=default, extra_values=extra_values,
		                 module=module)
		ns = self._namespace()
		self.flags = [f for f in flags if ns is not None and hasattr(ns, f)]
		self.initial = self.default if initial is None else initial

	def spec(self) -> dict:
		"""Widget metadata published to the frontend via the input's extra_dict.
		tests/validate_workflows.py also reads it to check saved flag strings."""
		return {"base": self.options, "flags": self.flags, "default": self.default}


class NamedFlagGroup:
	"""A FlagGroup whose parts are names OF OUR OWN, not cv2 constants.

	Same widget contract as FlagGroup - web/cv_flags.js reads only
	`base`/`flags`/`default` and tests/validate_workflows.py only checks "at
	most one base plus a subset of the flags" - so the checkbox UI comes for
	free. What differs is the resolution: there is no `cv2.AREA` constant, so
	EnumGroup.resolve() would raise on every part, and OR-ing ints is not what
	a caller wants anyway. parse() hands back the SELECTED NAMES instead.

	It deliberately does NOT subclass FlagGroup: that class filters its flags
	by `hasattr(cv2, name)`, which would silently delete every option here.

	This is the group that outgrew the toggle rendering (26 flags), so it is
	the one the searchable popover renderer was written for - see FlagGroup's
	docstring and web/cvlib/flag_search.js.
	"""

	def __init__(self, base, flags, default=None, initial=None, tooltips=None):
		"""
		base     : mutually exclusive options (one, or a "none" spelling).
		flags    : independently selectable names, rendered as toggles.
		default  : the preselected base; falls back to the first option.
		initial  : the widget's preset value, which may carry pre-set toggles
		           ("no position | area"). Falls back to `default`.
		tooltips : {flag or base option: text} shown when the user hovers THAT
		           checkbox. The toggles are named "<input>.<flag>", which
		           matches no schema input, so without this they inherit
		           nothing - the per-option explanation has nowhere else to
		           live but the input's own tooltip, which then has to carry
		           every option at once.
		"""
		self.options = list(base)
		self.flags = list(flags)
		self.default = default if default in self.options else self.options[0]
		self.initial = self.default if initial is None else initial
		self.tooltips = dict(tooltips or {})

	def spec(self) -> dict:
		spec = {"base": self.options, "flags": self.flags, "default": self.default}
		if self.tooltips:
			# omitted entirely when empty, so a group that defines none keeps
			# the exact spec dict it published before
			spec["tooltips"] = dict(self.tooltips)
		return spec

	def parse(self, value):
		"""-> (base option, [selected flags]) for a pipe-joined widget value.

		The flags come back in DECLARED order, not in the order they appear in
		the string, so the column order a caller emits cannot depend on which
		toggle the user clicked first. An empty value means the default base
		with no flags (the "" back-compat sentinel of every other widget)."""
		parts = [p.strip() for p in str(value).split("|") if p.strip()]
		unknown = [p for p in parts
		           if p not in self.options and p not in self.flags]
		if unknown:
			raise ValueError(
				f"Unknown option(s) {unknown} - expected one of {self.options} "
				f"plus any of {self.flags}."
			)
		bases = [p for p in parts if p in self.options]
		if len(bases) > 1:
			raise ValueError(f"{bases} are mutually exclusive - pick one.")
		return (bases[0] if bases else self.default,
		        [f for f in self.flags if f in parts])


def _all_with_prefix(prefix: str, exclude_prefixes=()) -> list:
	names = []
	for name in dir(cv2):
		if not name.startswith(prefix):
			continue
		if any(name.startswith(ex) for ex in exclude_prefixes):
			continue
		if isinstance(getattr(cv2, name), int):
			names.append(name)
	return sorted(names)


# --- color conversion ------------------------------------------------------
_common_colors = [
	"COLOR_BGR2GRAY", "COLOR_GRAY2BGR", "COLOR_BGR2RGB", "COLOR_RGB2BGR",
	"COLOR_RGB2GRAY", "COLOR_GRAY2RGB", "COLOR_BGR2HSV", "COLOR_HSV2BGR",
	"COLOR_BGR2LAB", "COLOR_LAB2BGR", "COLOR_BGR2YCrCb", "COLOR_YCrCb2BGR",
	"COLOR_BGR2BGRA", "COLOR_BGRA2BGR", "COLOR_RGBA2RGB", "COLOR_RGB2RGBA",
]
_other_colors = [n for n in _all_with_prefix("COLOR_", exclude_prefixes=("COLOR_COLORCVT",)) if n not in _common_colors]
COLOR = EnumGroup(_common_colors + _other_colors, default="COLOR_BGR2GRAY")

# --- demosaicing (Bayer CFA -> RGB) ----------------------------------------
# cv2.demosaicing's `code` is a Bayer-specific slice of the COLOR_* space - the
# full COLOR dropdown (330 conversions) buries the ~12 codes that actually
# apply. This curated group is the four CFA layouts x three interpolation
# algorithms, all decoding to RGB (ComfyUI's channel order). OpenCV 4.x used the
# legacy 2-letter aliases (COLOR_BayerBG2BGR etc.) whose letters name the 2nd
# row/2nd col of the pattern - counter-intuitive; the 4-letter names added in
# recent builds spell the CFA's top-left 2x2 directly (RGGB = R at (0,0)), so we
# prefer those. 'CV Mosaic (RGB -> Bayer CFA)' with pattern P pairs with the
# COLOR_Bayer<P>2RGB* codes here (verified by PSNR round-trip in smoke_test).
_DEMOSAIC_PATTERNS = ["RGGB", "GRBG", "GBRG", "BGGR"]
_demosaic = (
	[f"COLOR_Bayer{p}2RGB" for p in _DEMOSAIC_PATTERNS]          # bilinear
	+ [f"COLOR_Bayer{p}2RGB_VNG" for p in _DEMOSAIC_PATTERNS]    # variable number of gradients
	+ [f"COLOR_Bayer{p}2RGB_EA" for p in _DEMOSAIC_PATTERNS]     # edge-aware
)
DEMOSAIC = EnumGroup(_demosaic, default="COLOR_BayerRGGB2RGB")

# --- chromatic aberration correction (Bayer CFA -> BGR) --------------------
# correctChromaticAberration's `bayer_pattern` demosaices a single-channel raw
# image before correction. It expects BGR output (not RGB) and uses the basic
# bilinear interpolator (no _VNG/_EA suffix).
BAYER_CA = EnumGroup(
	[f"COLOR_Bayer{p}2BGR" for p in _DEMOSAIC_PATTERNS],
	default="COLOR_BayerRGGB2BGR",
)

# --- geometry / sampling ---------------------------------------------------
INTER = EnumGroup(
	["INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC", "INTER_AREA",
	 "INTER_LANCZOS4", "INTER_LINEAR_EXACT", "INTER_NEAREST_EXACT"],
	default="INTER_LINEAR",
)
WARP_FLAGS = FlagGroup(  # warpAffine / warpPerspective
	base=["INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC", "INTER_LANCZOS4"],
	flags=["WARP_INVERSE_MAP", "WARP_FILL_OUTLIERS"],
	default="INTER_LINEAR",
)
WARP_POLAR_FLAGS = FlagGroup(  # warpPolar: interpolation + polar-mode bits
	base=["INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC", "INTER_LANCZOS4"],
	flags=["WARP_POLAR_LOG", "WARP_INVERSE_MAP", "WARP_FILL_OUTLIERS"],
	default="INTER_LINEAR",
)
WARP_ALL_FLAGS = FlagGroup(  # the 'CV Warp Flags' builder node: superset of the
	# remap/warpAffine/warpPerspective/warpPolar groups, so ONE authored value can
	# fan out to any of them (each cv2 function ignores the bits it doesn't use)
	base=["INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC", "INTER_LANCZOS4"],
	flags=["WARP_RELATIVE_MAP", "WARP_FILL_OUTLIERS", "WARP_INVERSE_MAP",
	       "WARP_POLAR_LOG"],
	default="INTER_LINEAR",
)
BORDER = EnumGroup(
	["BORDER_DEFAULT", "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
	 "BORDER_WRAP", "BORDER_REFLECT_101", "BORDER_TRANSPARENT", "BORDER_ISOLATED"],
	default="BORDER_DEFAULT",
)
ROTATE = EnumGroup(
	["ROTATE_90_CLOCKWISE", "ROTATE_180", "ROTATE_90_COUNTERCLOCKWISE"],
)
FLIP = EnumGroup(
	["horizontal (around y-axis)", "vertical (around x-axis)", "both"],
	default="horizontal (around y-axis)",
	extra_values={"horizontal (around y-axis)": 1, "vertical (around x-axis)": 0, "both": -1},
)

# --- iteration stop criteria (TermCriteria type flag) ----------------------
# The `type` field of a cv2 TermCriteria tuple. Friendly labels mirror
# the CV_TermCriteria builder so the split component reads the same way.
TERM_CRITERIA_TYPE = EnumGroup(
	["max count or epsilon (whichever first)", "max count only", "epsilon only"],
	default="max count or epsilon (whichever first)",
	extra_values={
		"max count or epsilon (whichever first)": cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
		"max count only": cv2.TERM_CRITERIA_COUNT,
		"epsilon only": cv2.TERM_CRITERIA_EPS,
	},
)

# --- thresholding ----------------------------------------------------------
# cv2.threshold's type is one base method OR-ed with independent flag bits
# (OTSU, TRIANGLE): a FlagGroup renders as a base dropdown + one toggle per
# flag via cv_flags.js instead of enumerating every combination.
THRESH = FlagGroup(
	base=["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO",
	       "THRESH_TOZERO_INV"],
	flags=["THRESH_OTSU", "THRESH_TRIANGLE"],
	default="THRESH_BINARY",
)
THRESH_BINARY_ONLY = EnumGroup(["THRESH_BINARY", "THRESH_BINARY_INV"])
ADAPTIVE_THRESH = EnumGroup(["ADAPTIVE_THRESH_GAUSSIAN_C", "ADAPTIVE_THRESH_MEAN_C"])

# --- morphology ------------------------------------------------------------
MORPH_OP = EnumGroup(
	["MORPH_ERODE", "MORPH_DILATE", "MORPH_OPEN", "MORPH_CLOSE",
	 "MORPH_GRADIENT", "MORPH_TOPHAT", "MORPH_BLACKHAT", "MORPH_HITMISS"],
	default="MORPH_OPEN",
)
MORPH_SHAPE = EnumGroup(["MORPH_ELLIPSE", "MORPH_RECT", "MORPH_CROSS"])

# --- norms / distances / depths -------------------------------------------
NORM = EnumGroup(
	["NORM_L2", "NORM_L1", "NORM_INF", "NORM_L2SQR", "NORM_HAMMING",
	 "NORM_HAMMING2", "NORM_MINMAX"],
	default="NORM_L2",
)
FLANN_INDEX = EnumGroup(
	["KDTREE (float32 descriptors)", "LSH (binary descriptors)"],
	default="KDTREE (float32 descriptors)",
	extra_values={
		"KDTREE (float32 descriptors)": 1,   # FLANN_INDEX_KDTREE
		"LSH (binary descriptors)": 6,        # FLANN_INDEX_LSH
		"KMEANS": 0,                          # FLANN_INDEX_KMEANS
		"HIERARCHICAL": 2,                    # FLANN_INDEX_HIERARCHICAL
		"AUTOTUNED": 3,                       # FLANN_INDEX_AUTOTUNED
	},
)
DIST = EnumGroup(
	["DIST_L2", "DIST_L1", "DIST_C", "DIST_L12", "DIST_FAIR", "DIST_WELSCH", "DIST_HUBER"],
	default="DIST_L2",
)
REDUCE = EnumGroup(  # cv2.reduce's rtype: the reduction OPERATION (not a depth)
	["REDUCE_SUM", "REDUCE_AVG", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_SUM2"],
	default="REDUCE_SUM",
)
DEPTH = EnumGroup(
	["same as input", "CV_8U", "CV_8S", "CV_16U", "CV_16S", "CV_32S", "CV_32F", "CV_64F"],
	default="same as input",
	extra_values={"same as input": -1},
)
QDFT = EnumGroup(  # cv2.ximgproc.qdft: "only DFT_INVERSE flags is supported"
	["forward (0)", "DFT_INVERSE"],
	default="forward (0)",
	extra_values={"forward (0)": 0},
)
DEPTH_FLOAT = EnumGroup(  # cv2.rescaleDepth: its doc allows only these two
	["CV_32F", "CV_64F"],
	default="CV_32F",
)
DECOMP = EnumGroup(
	["DECOMP_LU", "DECOMP_SVD", "DECOMP_EIG", "DECOMP_CHOLESKY", "DECOMP_QR", "DECOMP_NORMAL"],
)
GEMM_FLAGS = FlagGroup(  # cv2.gemm: transpose src1/src2/src3 before multiplying
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["GEMM_1_T", "GEMM_2_T", "GEMM_3_T"],
)
SVD_FLAGS = FlagGroup(  # cv2.SVDecomp
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["SVD_MODIFY_A", "SVD_NO_UV", "SVD_FULL_UV"],
)
CMP = EnumGroup(["CMP_EQ", "CMP_GT", "CMP_GE", "CMP_LT", "CMP_LE", "CMP_NE"])
HISTCMP = EnumGroup(  # cv2.compareHist
	["HISTCMP_CORREL", "HISTCMP_CHISQR", "HISTCMP_INTERSECT",
	 "HISTCMP_BHATTACHARYYA", "HISTCMP_CHISQR_ALT", "HISTCMP_KL_DIV"],
	default="HISTCMP_CORREL",
)
GRABCUT_MODE = EnumGroup(  # cv2.grabCut
	["GC_INIT_WITH_RECT", "GC_INIT_WITH_MASK", "GC_EVAL", "GC_EVAL_FREEZE_MODEL"],
	default="GC_INIT_WITH_RECT",
)
FLOODFILL = FlagGroup(  # cv2.floodFill's packed flags: connectivity lives in the
	# low byte, the value the MASK is filled with in bits 8-15 (0 -> 1), and the
	# FLOODFILL_* mode bits on top. The base picks connectivity + mask-fill value;
	# cv2's default is plain 4-connectivity.
	base=["4-connectivity (mask filled with 1)",
	      "8-connectivity (mask filled with 1)",
	      "4-connectivity, mask filled with 255",
	      "8-connectivity, mask filled with 255"],
	extra_values={"4-connectivity (mask filled with 1)": 4,
	              "8-connectivity (mask filled with 1)": 8,
	              "4-connectivity, mask filled with 255": 4 | (255 << 8),
	              "8-connectivity, mask filled with 255": 8 | (255 << 8)},
	flags=["FLOODFILL_FIXED_RANGE", "FLOODFILL_MASK_ONLY"],
)
# cv2.sort / cv2.sortIdx: pick a row/column axis AND a direction (combined)
SORT_FLAGS = EnumGroup(
	["SORT_EVERY_ROW | SORT_ASCENDING", "SORT_EVERY_ROW | SORT_DESCENDING",
	 "SORT_EVERY_COLUMN | SORT_ASCENDING", "SORT_EVERY_COLUMN | SORT_DESCENDING"],
)
CCL = EnumGroup(  # connected-components labeling algorithm (cv2.*WithAlgorithm)
	["CCL_DEFAULT", "CCL_WU", "CCL_GRANA", "CCL_BOLELLI", "CCL_SAUF",
	 "CCL_BBDT", "CCL_SPAGHETTI"],
	default="CCL_DEFAULT",
)
DIST_LABEL = EnumGroup(  # cv2.distanceTransformWithLabels labelType
	["DIST_LABEL_CCOMP", "DIST_LABEL_PIXEL"],
)
EDGE_FILTER = EnumGroup(["RECURS_FILTER", "NORMCONV_FILTER"])  # cv2.edgePreservingFilter
# optical-flow flags default to 0 ("none"); every bit is independently OR-able,
# so these are FlagGroups with a "none (0)" base rather than EnumGroups
OPTFLOW_PYRLK = FlagGroup(  # cv2.calcOpticalFlowPyrLK
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["OPTFLOW_USE_INITIAL_FLOW", "OPTFLOW_LK_GET_MIN_EIGENVALS"],
)
OPTFLOW_FARNEBACK = FlagGroup(  # cv2.calcOpticalFlowFarneback
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["OPTFLOW_USE_INITIAL_FLOW", "OPTFLOW_FARNEBACK_GAUSSIAN"],
)
OPTFLOW_ALL_FLAGS = FlagGroup(  # the 'CV Optical Flow Flags' builder node:
	# superset of the PyrLK + Farneback groups so one authored value can fan
	# out to either (each function ignores the other's bit)
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["OPTFLOW_USE_INITIAL_FLOW", "OPTFLOW_LK_GET_MIN_EIGENVALS",
	       "OPTFLOW_FARNEBACK_GAUSSIAN"],
)
CALIB_CB = FlagGroup(  # cv2.findChessboardCorners; initial = cv2's own default
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["CALIB_CB_ADAPTIVE_THRESH", "CALIB_CB_NORMALIZE_IMAGE",
	       "CALIB_CB_FAST_CHECK", "CALIB_CB_FILTER_QUADS"],
	initial="CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE",
)
CALIB_CB_SB = FlagGroup(  # cv2.findChessboardCornersSB* - a DIFFERENT flag set
	# than the classic detector (no adaptive-threshold/fast-check bits; instead
	# exhaustive search, subpixel accuracy, deltille/marker layouts)
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["CALIB_CB_NORMALIZE_IMAGE", "CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY",
	       "CALIB_CB_LARGER", "CALIB_CB_MARKER"],
)
CALIB_CB_ALL_FLAGS = FlagGroup(  # the 'CV Chessboard Flags' builder node:
	# superset of the classic + SB groups so ONE authored value can fan out to
	# either detector (the bit values are disjoint except NORMALIZE_IMAGE; each
	# function ignores the other's bits)
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["CALIB_CB_ADAPTIVE_THRESH", "CALIB_CB_NORMALIZE_IMAGE",
	       "CALIB_CB_FAST_CHECK", "CALIB_CB_FILTER_QUADS", "CALIB_CB_EXHAUSTIVE",
	       "CALIB_CB_ACCURACY", "CALIB_CB_LARGER", "CALIB_CB_MARKER"],
	initial="CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE",
)
CALIB_CIRCLES_GRID = FlagGroup(  # cv2.findCirclesGrid: layout plus clustering
	base=["CALIB_CB_SYMMETRIC_GRID", "CALIB_CB_ASYMMETRIC_GRID"],
	flags=["CALIB_CB_CLUSTERING"],
	default="CALIB_CB_SYMMETRIC_GRID",
)
OMNIDIR_CALIB = FlagGroup(  # cv2.omnidir.calibrate / stereoCalibrate
	# these constants live in cv2.omnidir and MOST of the names also exist at
	# cv2 top level with different values (CALIB_FIX_SKEW is 2 here, 1<<25
	# there), so the group is module-scoped
	base=["none (0)"], extra_values={"none (0)": 0}, module="omnidir",
	flags=["CALIB_USE_GUESS", "CALIB_FIX_SKEW", "CALIB_FIX_K1", "CALIB_FIX_K2",
	       "CALIB_FIX_P1", "CALIB_FIX_P2", "CALIB_FIX_XI", "CALIB_FIX_GAMMA",
	       "CALIB_FIX_CENTER"],
	initial="none (0) | CALIB_FIX_SKEW",
)
OMNIDIR_RECTIFY = EnumGroup(  # cv2.omnidir.undistortImage projection types
	["RECTIFY_PERSPECTIVE", "RECTIFY_CYLINDRICAL", "RECTIFY_STEREOGRAPHIC",
	 "RECTIFY_LONGLATI"],
	default="RECTIFY_PERSPECTIVE", module="omnidir",
)
OMNIDIR_STEREO_RECTIFY = EnumGroup(  # stereoReconstruct takes only these two
	["RECTIFY_PERSPECTIVE", "RECTIFY_LONGLATI"],
	default="RECTIFY_PERSPECTIVE", module="omnidir",
)
FISHEYE_CALIB = FlagGroup(  # cv2.fisheye.calibrate / stereoCalibrate
	# OpenCV 5 moved the fisheye flags into the shared CALIB_* namespace, so
	# they are cv2.CALIB_*, NOT cv2.fisheye.CALIB_* (that attribute no longer
	# exists) and their VALUES differ from the 4.x fisheye ones (RECOMPUTE was
	# 2, it is 1<<23 here). RECOMPUTE_EXTRINSIC is preset because without it the
	# same views calibrate to rms 99 px / fx 388 instead of 0.13 px / fx 320.
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["CALIB_RECOMPUTE_EXTRINSIC", "CALIB_FIX_SKEW", "CALIB_CHECK_COND",
	       "CALIB_FIX_PRINCIPAL_POINT", "CALIB_FIX_FOCAL_LENGTH",
	       "CALIB_FIX_K1", "CALIB_FIX_K2", "CALIB_FIX_K3", "CALIB_FIX_K4"],
	initial="none (0) | CALIB_RECOMPUTE_EXTRINSIC | CALIB_FIX_SKEW",
)
STEREO_RECTIFY = EnumGroup(  # cv2.stereoRectify's single flag bit; cv2's
	# default IS CALIB_ZERO_DISPARITY (same principal point in both views)
	["CALIB_ZERO_DISPARITY", "none (0)"],
	default="CALIB_ZERO_DISPARITY",
	extra_values={"none (0)": 0},
)

# --- Fourier / spectrum ------------------------------------------------------
# cv2.dft / cv2.idft flags default to 0; every DFT_* bit is independently
# OR-able, so this is a FlagGroup: a "none (0)" base plus one toggle per bit.
# DFT_COMPLEX_OUTPUT gives the full 2-channel (re, im) spectrum most pipelines
# want; idft usually wants SCALE + REAL_OUTPUT. Old workflows saved while this
# was a curated combo ("DFT_SCALE | DFT_REAL_OUTPUT") keep resolving.
DFT_FLAGS = FlagGroup(
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["DFT_COMPLEX_OUTPUT", "DFT_REAL_OUTPUT", "DFT_SCALE", "DFT_INVERSE",
	       "DFT_ROWS", "DFT_COMPLEX_INPUT"],
)
DCT_FLAGS = FlagGroup(  # cv2.dct / cv2.idct: the cosine transform has no
	# complex-output bits - only inverse and per-row operation apply
	base=["none (0)"], extra_values={"none (0)": 0},
	flags=["DCT_INVERSE", "DCT_ROWS"],
)
MUL_SPECTRUMS = EnumGroup(  # cv2.mulSpectrums' flags: 0 or DFT_ROWS
	["whole 2-D spectra", "each row is an independent 1-D spectrum (DFT_ROWS)"],
	default="whole 2-D spectra",
	extra_values={"whole 2-D spectra": 0,
	              "each row is an independent 1-D spectrum (DFT_ROWS)": cv2.DFT_ROWS},
)

# --- image codec ------------------------------------------------------------
IMREAD = EnumGroup(  # cv2.imdecode's color/depth loading flags
	["IMREAD_COLOR", "IMREAD_GRAYSCALE", "IMREAD_UNCHANGED",
	 "IMREAD_ANYDEPTH", "IMREAD_ANYCOLOR", "IMREAD_COLOR | IMREAD_ANYDEPTH",
	 "IMREAD_REDUCED_COLOR_2", "IMREAD_REDUCED_GRAYSCALE_2",
	 "IMREAD_REDUCED_COLOR_4", "IMREAD_REDUCED_GRAYSCALE_4",
	 "IMREAD_IGNORE_ORIENTATION"],
	default="IMREAD_COLOR",
)

# --- photo -----------------------------------------------------------------
CLONE = EnumGroup(["NORMAL_CLONE", "MIXED_CLONE", "MONOCHROME_TRANSFER"])
INPAINT = EnumGroup(["INPAINT_TELEA", "INPAINT_NS"])

# --- features / matching ---------------------------------------------------
TEMPLATE_MATCH = EnumGroup(
	["TM_CCOEFF_NORMED", "TM_CCOEFF", "TM_CCORR_NORMED", "TM_CCORR",
	 "TM_SQDIFF_NORMED", "TM_SQDIFF"],
	default="TM_CCOEFF_NORMED",
)
CONTOUR_RETR = EnumGroup(["RETR_EXTERNAL", "RETR_LIST", "RETR_CCOMP", "RETR_TREE", "RETR_FLOODFILL"])
CONTOUR_RETR_SIMPLE = EnumGroup(["RETR_EXTERNAL", "RETR_LIST"], default="RETR_EXTERNAL")
CONTOUR_APPROX = EnumGroup(["CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_NONE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS"])
HOUGH = EnumGroup(["HOUGH_GRADIENT", "HOUGH_GRADIENT_ALT", "HOUGH_STANDARD", "HOUGH_PROBABILISTIC", "HOUGH_MULTI_SCALE"])
CONTOURS_MATCH = EnumGroup(["CONTOURS_MATCH_I1", "CONTOURS_MATCH_I2", "CONTOURS_MATCH_I3"])

# --- robust model estimation / multi-view geometry ---------------------------
ROBUST_METHOD = EnumGroup(  # findHomography
	["RANSAC", "LMEDS", "RHO", "USAC_DEFAULT", "USAC_MAGSAC", "USAC_ACCURATE",
	 "least-squares (all points)"],
	default="RANSAC",
	extra_values={"least-squares (all points)": 0},
)
RANSAC_LMEDS = EnumGroup(  # findEssentialMat, estimateAffine2D & co
	["RANSAC", "LMEDS"],
	default="RANSAC",
)
FM_METHOD = EnumGroup(  # findFundamentalMat
	["FM_RANSAC", "FM_LMEDS", "FM_8POINT", "FM_7POINT"],
	default="FM_RANSAC",
)
EPILINE_IMAGE = EnumGroup(  # computeCorrespondEpilines' whichImage
	["points are from image 1 (lines for image 2)",
	 "points are from image 2 (lines for image 1)"],
	default="points are from image 1 (lines for image 2)",
	extra_values={"points are from image 1 (lines for image 2)": 1,
	              "points are from image 2 (lines for image 1)": 2},
)
SOLVEPNP = EnumGroup(
	["SOLVEPNP_ITERATIVE", "SOLVEPNP_EPNP", "SOLVEPNP_P3P", "SOLVEPNP_AP3P",
	 "SOLVEPNP_IPPE", "SOLVEPNP_IPPE_SQUARE", "SOLVEPNP_SQPNP"],
	default="SOLVEPNP_ITERATIVE",
)
MOTION = EnumGroup(  # findTransformECC
	["MOTION_AFFINE", "MOTION_TRANSLATION", "MOTION_EUCLIDEAN", "MOTION_HOMOGRAPHY"],
	default="MOTION_AFFINE",
)
KMEANS = EnumGroup(
	["KMEANS_PP_CENTERS", "KMEANS_RANDOM_CENTERS", "KMEANS_USE_INITIAL_LABELS"],
	default="KMEANS_PP_CENTERS",
)
ALGO_HINT = EnumGroup(  # the 'hint' parameter of GaussianBlur, cvtColor & co
	["ALGO_HINT_DEFAULT", "ALGO_HINT_ACCURATE", "ALGO_HINT_APPROX"],
	default="ALGO_HINT_DEFAULT",
)

# --- remap flags (includes WARP_RELATIVE_MAP, only meaningful for cv2.remap) ---
# WARP_RELATIVE_MAP changes the interpretation so map1/map2 are relative
# offsets (dx, dy) instead of absolute source coordinates:
#   dst(x,y) = src(x + map1(x,y), y + map2(x,y))
# Available since OpenCV 4.x, has value 32.
REMAP_FLAGS = FlagGroup(
    base=["INTER_LINEAR", "INTER_NEAREST", "INTER_CUBIC", "INTER_LANCZOS4"],
    flags=["WARP_RELATIVE_MAP", "WARP_FILL_OUTLIERS", "WARP_INVERSE_MAP"],
    default="INTER_LINEAR",
)

# --- map type for convertMaps ----------------------------------------------
# The dstmap1type parameter of cv2.convertMaps: CV_16SC2 packs (x,y) as
# int16 + a separate CV_16UC1 interpolation-coeff map for a ~30% faster
# fixed-point cv2.remap; CV_32FC1 is the standard float32 map.
CONVERT_MAP = EnumGroup(
    ["CV_16SC2 (fixed-point, faster remap)", "CV_32FC1 (float32, standard)"],
    default="CV_16SC2 (fixed-point, faster remap)",
    extra_values={"CV_16SC2 (fixed-point, faster remap)": 35,
                  "CV_32FC1 (float32, standard)": 5},
)

# --- drawing ---------------------------------------------------------------
LINE_TYPE = EnumGroup(["LINE_AA", "LINE_8", "LINE_4"])
FONT = EnumGroup(_all_with_prefix("FONT_HERSHEY"), default="FONT_HERSHEY_SIMPLEX")
MARKER = EnumGroup(_all_with_prefix("MARKER_"), default="MARKER_CROSS")
COLORMAP = EnumGroup(_all_with_prefix("COLORMAP_"), default="COLORMAP_VIRIDIS")

# --- color correction module (cv2.ccm) ------------------------------------
# Constants live under cv2.ccm, not cv2, so EnumGroup's default hasattr check
# would fail. Use extra_values with explicit integer lookups instead.
CCM_CCM_TYPE = EnumGroup(
    ["CCM_LINEAR", "CCM_AFFINE"],
    default="CCM_LINEAR",
    extra_values={
        "CCM_LINEAR": cv2.ccm.CCM_LINEAR,
        "CCM_AFFINE": cv2.ccm.CCM_AFFINE,
    },
)
CCM_COLOR_CHECKER = EnumGroup(
    ["COLORCHECKER_MACBETH", "COLORCHECKER_VINYL", "COLORCHECKER_DIGITAL_SG"],
    default="COLORCHECKER_MACBETH",
    extra_values={
        "COLORCHECKER_MACBETH": cv2.ccm.COLORCHECKER_MACBETH,
        "COLORCHECKER_VINYL": cv2.ccm.COLORCHECKER_VINYL,
        "COLORCHECKER_DIGITAL_SG": cv2.ccm.COLORCHECKER_DIGITAL_SG,
    },
)
CCM_COLOR_SPACE = EnumGroup(
    ["COLOR_SPACE_SRGB", "COLOR_SPACE_ADOBE_RGB", "COLOR_SPACE_WIDE_GAMUT_RGB",
     "COLOR_SPACE_PRO_PHOTO_RGB", "COLOR_SPACE_DCI_P3_RGB",
     "COLOR_SPACE_APPLE_RGB", "COLOR_SPACE_REC_709_RGB",
     "COLOR_SPACE_REC_2020_RGB"],
    default="COLOR_SPACE_SRGB",
    extra_values={
        "COLOR_SPACE_SRGB": cv2.ccm.COLOR_SPACE_SRGB,
        "COLOR_SPACE_ADOBE_RGB": cv2.ccm.COLOR_SPACE_ADOBE_RGB,
        "COLOR_SPACE_WIDE_GAMUT_RGB": cv2.ccm.COLOR_SPACE_WIDE_GAMUT_RGB,
        "COLOR_SPACE_PRO_PHOTO_RGB": cv2.ccm.COLOR_SPACE_PRO_PHOTO_RGB,
        "COLOR_SPACE_DCI_P3_RGB": cv2.ccm.COLOR_SPACE_DCI_P3_RGB,
        "COLOR_SPACE_APPLE_RGB": cv2.ccm.COLOR_SPACE_APPLE_RGB,
        "COLOR_SPACE_REC_709_RGB": cv2.ccm.COLOR_SPACE_REC_709_RGB,
        "COLOR_SPACE_REC_2020_RGB": cv2.ccm.COLOR_SPACE_REC_2020_RGB,
    },
)
CCM_DISTANCE = EnumGroup(
    ["DISTANCE_CIE2000", "DISTANCE_CIE76", "DISTANCE_CIE94_GRAPHIC_ARTS",
     "DISTANCE_CIE94_TEXTILES", "DISTANCE_CMC_1TO1", "DISTANCE_CMC_2TO1",
     "DISTANCE_RGB", "DISTANCE_RGBL"],
    default="DISTANCE_CIE2000",
    extra_values={
        "DISTANCE_CIE2000": cv2.ccm.DISTANCE_CIE2000,
        "DISTANCE_CIE76": cv2.ccm.DISTANCE_CIE76,
        "DISTANCE_CIE94_GRAPHIC_ARTS": cv2.ccm.DISTANCE_CIE94_GRAPHIC_ARTS,
        "DISTANCE_CIE94_TEXTILES": cv2.ccm.DISTANCE_CIE94_TEXTILES,
        "DISTANCE_CMC_1TO1": cv2.ccm.DISTANCE_CMC_1TO1,
        "DISTANCE_CMC_2TO1": cv2.ccm.DISTANCE_CMC_2TO1,
        "DISTANCE_RGB": cv2.ccm.DISTANCE_RGB,
        "DISTANCE_RGBL": cv2.ccm.DISTANCE_RGBL,
    },
)
CCM_LINEARIZATION = EnumGroup(
    ["LINEARIZATION_GAMMA", "LINEARIZATION_IDENTITY",
     "LINEARIZATION_COLORPOLYFIT", "LINEARIZATION_COLORLOGPOLYFIT",
     "LINEARIZATION_GRAYPOLYFIT", "LINEARIZATION_GRAYLOGPOLYFIT"],
    default="LINEARIZATION_GAMMA",
    extra_values={
        "LINEARIZATION_GAMMA": cv2.ccm.LINEARIZATION_GAMMA,
        "LINEARIZATION_IDENTITY": cv2.ccm.LINEARIZATION_IDENTITY,
        "LINEARIZATION_COLORPOLYFIT": cv2.ccm.LINEARIZATION_COLORPOLYFIT,
        "LINEARIZATION_COLORLOGPOLYFIT": cv2.ccm.LINEARIZATION_COLORLOGPOLYFIT,
	"LINEARIZATION_GRAYPOLYFIT": cv2.ccm.LINEARIZATION_GRAYPOLYFIT,
		"LINEARIZATION_GRAYLOGPOLYFIT": cv2.ccm.LINEARIZATION_GRAYLOGPOLYFIT,
	},
)

STITCHER_MODE = EnumGroup(
	["Stitcher_PANORAMA", "Stitcher_SCANS"],
	default="Stitcher_PANORAMA",
)

STITCHER_STATUS = EnumGroup(
	["Stitcher_OK", "Stitcher_ERR_NEED_MORE_IMGS",
	 "Stitcher_ERR_HOMOGRAPHY_EST_FAIL", "Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL"],
	default="Stitcher_OK",
)


# ---------------------------------------------------------------------------
# opencv-contrib submodules (xphoto / ximgproc / ft / img_hash)
#
# Their constants live under cv2.<module>, NOT cv2, so EnumGroup's hasattr(cv2,
# name) existence check would reject every one of them. They are therefore
# declared with explicit extra_values, resolved through _contrib_values() below,
# which ALSO makes the whole group degrade gracefully on a core-only build: a
# missing submodule yields an empty mapping, EnumGroup falls back to its
# "<none available in this cv2 build>" placeholder, and lowlevel.py has already
# skipped the corresponding nodes anyway.
# ---------------------------------------------------------------------------

def _contrib_values(module: str, names) -> dict:
	"""{name: int} for the constants a contrib submodule actually exposes."""
	mod = getattr(cv2, module, None)
	if mod is None:
		return {}
	return {n: int(getattr(mod, n)) for n in names if hasattr(mod, n)}


# --- xphoto ----------------------------------------------------------------
# cv2.xphoto.inpaint's algorithm. Deliberately NOT named INPAINT like the
# cv2.photo group: the two modules' inpainters take OPPOSITE masks (cv2.inpaint
# fills where the mask is non-zero, xphoto.inpaint treats non-zero as the KNOWN
# region), so keeping the names apart keeps the two dropdowns from looking
# interchangeable.
XPHOTO_INPAINT = EnumGroup(
	["INPAINT_FSR_FAST", "INPAINT_FSR_BEST", "INPAINT_SHIFTMAP"],
	default="INPAINT_FSR_FAST",
	extra_values=_contrib_values(
		"xphoto", ["INPAINT_SHIFTMAP", "INPAINT_FSR_BEST", "INPAINT_FSR_FAST"]),
)

# cv2.xphoto.oilPainting quantizes the image in this colour space before taking
# the per-bin mode; the space decides which differences count as "the same
# paint". Lab/HSV separate hue from lightness (painterly), GRAY ignores colour
# entirely (fastest, flattest). All eight verified working on cv2 5.0.0.
OIL_PAINTING_SPACE = EnumGroup(
	["COLOR_BGR2Lab", "COLOR_BGR2HSV", "COLOR_BGR2GRAY", "COLOR_BGR2YCrCb",
	 "COLOR_BGR2HLS", "COLOR_BGR2LUV", "COLOR_BGR2XYZ", "COLOR_BGR2YUV"],
	default="COLOR_BGR2Lab",
)

# --- ximgproc --------------------------------------------------------------
THINNING = EnumGroup(  # cv2.ximgproc.thinning
	["THINNING_ZHANGSUEN", "THINNING_GUOHALL"],
	default="THINNING_ZHANGSUEN",
	extra_values=_contrib_values(
		"ximgproc", ["THINNING_ZHANGSUEN", "THINNING_GUOHALL"]),
)
NIBLACK_BINARIZATION = EnumGroup(  # cv2.ximgproc.niBlackThreshold
	["BINARIZATION_NIBLACK", "BINARIZATION_SAUVOLA", "BINARIZATION_WOLF",
	 "BINARIZATION_NICK"],
	default="BINARIZATION_NIBLACK",
	extra_values=_contrib_values(
		"ximgproc", ["BINARIZATION_NIBLACK", "BINARIZATION_SAUVOLA",
		             "BINARIZATION_WOLF", "BINARIZATION_NICK"]),
)
DT_FILTER_MODE = EnumGroup(  # cv2.ximgproc.dtFilter / createDTFilter
	["DTF_NC", "DTF_IC", "DTF_RF"],
	default="DTF_NC",
	extra_values=_contrib_values("ximgproc", ["DTF_NC", "DTF_IC", "DTF_RF"]),
)
WMF_WEIGHT = EnumGroup(  # cv2.ximgproc.weightedMedianFilter
	["WMF_EXP", "WMF_IV1", "WMF_IV2", "WMF_COS", "WMF_JAC", "WMF_OFF"],
	default="WMF_EXP",
	extra_values=_contrib_values(
		"ximgproc", ["WMF_EXP", "WMF_IV1", "WMF_IV2", "WMF_COS", "WMF_JAC",
		             "WMF_OFF"]),
)
FHT_ANGLE_RANGE = EnumGroup(  # cv2.ximgproc.FastHoughTransform angleRange
	["ARO_315_135", "ARO_0_45", "ARO_45_90", "ARO_90_135", "ARO_315_0",
	 "ARO_315_45", "ARO_45_135", "ARO_CTR_HOR", "ARO_CTR_VER"],
	default="ARO_315_135",
	extra_values=_contrib_values(
		"ximgproc", ["ARO_0_45", "ARO_45_90", "ARO_90_135", "ARO_315_0",
		             "ARO_315_45", "ARO_45_135", "ARO_315_135", "ARO_CTR_HOR",
		             "ARO_CTR_VER"]),
)
FHT_OP = EnumGroup(  # cv2.ximgproc.FastHoughTransform accumulation operator
	["FHT_ADD", "FHT_AVE", "FHT_MIN", "FHT_MAX"],
	default="FHT_ADD",
	extra_values=_contrib_values(
		"ximgproc", ["FHT_MIN", "FHT_MAX", "FHT_ADD", "FHT_AVE"]),
)
FHT_DESKEW = EnumGroup(  # cv2.ximgproc.FastHoughTransform makeSkew
	["HDO_DESKEW", "HDO_RAW"],
	default="HDO_DESKEW",
	extra_values=_contrib_values("ximgproc", ["HDO_RAW", "HDO_DESKEW"]),
)

# --- ft (fuzzy transform) ---------------------------------------------------
# Only LINEAR is offered: cv2.ft.SINUS raises "!mv[0].empty()" in merge() for
# every radius/channel combination on this OpenCV 5.0.0 contrib build, so a
# SINUS option could only ever produce a broken node (smoke_test asserts it
# still fails, so a fixed build shows up as a failing test rather than silently
# staying hidden).
FT_KERNEL_FUNCTION = EnumGroup(
	["LINEAR"],
	default="LINEAR",
	extra_values=_contrib_values("ft", ["LINEAR", "SINUS"]),
)
FT_INPAINT_ALGORITHM = EnumGroup(  # cv2.ft.inpaint
	["ONE_STEP", "MULTI_STEP", "ITERATIVE"],
	# MULTI_STEP is the default because ONE_STEP is BROKEN on this OpenCV 5.0.0
	# build: it returns NaN at every mask-zero pixel regardless of the mask
	# convention or the radius (smoke_test pins this, so a fixed build fails
	# loudly instead of silently staying downgraded). ITERATIVE is equivalent
	# but slower; ONE_STEP stays selectable for old saved workflows.
	default="MULTI_STEP",
	extra_values=_contrib_values("ft", ["ONE_STEP", "MULTI_STEP", "ITERATIVE"]),
)

# --- img_hash ---------------------------------------------------------------
BLOCK_MEAN_HASH_MODE = EnumGroup(  # cv2.img_hash.blockMeanHash
	["BLOCK_MEAN_HASH_MODE_0", "BLOCK_MEAN_HASH_MODE_1"],
	default="BLOCK_MEAN_HASH_MODE_0",
	extra_values=_contrib_values(
		"img_hash", ["BLOCK_MEAN_HASH_MODE_0", "BLOCK_MEAN_HASH_MODE_1"]),
)

# --- our own (not cv2 constants) --------------------------------------------
# The column menu of 'CV Region Properties'. The position columns are the
# mutually exclusive part (a region has ONE place), everything else is an
# independent toggle - which is what makes "cluster by area alone"
# ("no position | area") expressible at all; the COMBO it replaces could only
# offer position or position+area.
REGION_PROPERTIES = NamedFlagGroup(
	base=["centroid (x, y)", "bbox center (x, y)", "no position"],
	flags=["area", "equivalent radius", "bbox width", "bbox height",
	       "aspect ratio", "extent", "perimeter", "circularity", "solidity",
	       "orientation (sin/cos)", "eccentricity",
	       "mean colour", "mean intensity",
	       # appended LAST on purpose: a selection's COLUMN ORDER follows the
	       # order declared here, so inserting a flag mid-list would re-slot
	       # every saved feature matrix
	       "hu moments (log magnitude)", "chirality",
	       "min-area rect long side", "min-area rect short side",
	       "min-area rect aspect (long/short)",
	       "ellipse major axis", "ellipse minor axis",
	       "min-area rect centroid clearance (long, short)",
	       "centroid offset in bbox (x, y)",
	       "canonical orientation (sin/cos)", "orientation confidence",
	       "hole count", "min-area rect extent",
	       "loose ends", "convexity defects (count, max depth)",
	       "corner count",
	       "median thickness", "thickness variation",
	       "inscribed circle radius", "roundness (inscribed/enclosing)",
	       "skeleton length", "radial distance variation"],
	default="centroid (x, y)",
	tooltips={
		"area": "Number of pixels in the region. Grows with the SQUARE of the "
			"size, so a shape twice as wide has four times the area - use "
			"'equivalent radius' if you want a column that scales like a length.",
		"equivalent radius": "sqrt(area) - the side of a square with the same "
			"area, i.e. a size in PIXELS rather than pixels squared. Not the "
			"radius of the equal-area circle (that would be sqrt(area/pi), "
			"smaller by 1.77); it is a length-scaled stand-in for 'area', which "
			"is what keeps it comparable with the other length columns after the "
			"'image-relative' rescale.",
		"perimeter": "Length of the region's outline in pixels "
			"(cv2.arcLength of its external contour). A ragged or spiky edge "
			"makes this much larger than a smooth one of the same area - which "
			"is exactly what 'circularity' measures.",
		"extent": "How much of its UPRIGHT bounding box the region fills: "
			"area / (bbox width * bbox height), between 0 and 1. A rectangle "
			"lying square-on reads 1.00, a circle 0.79 (pi/4), a diagonal bar or "
			"an L-shape much less. Because the box is upright it changes as the "
			"region turns - the rotation-free version of the same idea is "
			"'solidity'.",
		"circularity": "How close the outline is to a circle: "
			"4*pi*area / perimeter^2. Exactly 1 for a perfect circle, ~0.79 for "
			"a square, and it falls fast for anything ragged, spiky or "
			"elongated, because the perimeter is squared. Independent of size "
			"and rotation. Note a pixelated small blob measures below 1 even "
			"when it is meant to be round - the staircase edge lengthens the "
			"perimeter.",
		"solidity": "area / (area of the CONVEX HULL) - the hull being the "
			"rubber band stretched around the region. 1.00 means fully convex "
			"(no dents): a disc, a square, a triangle. Below 1 means the outline "
			"has bites taken out of it - a crescent, a star, a 'C'. It is the "
			"standard 'is this blob dented or does it have concave arms?' test, "
			"and unlike 'extent' it does not care which way the region is "
			"turned. Holes INSIDE the region do not lower it (the outline is "
			"traced RETR_EXTERNAL) - count those with 'hole count'.",
		"eccentricity": "How elongated the region is, from the equivalent "
			"ellipse (the same second-order moments as 'ellipse major/minor "
			"axis'): 0 is a circle, values approaching 1 are an ever thinner "
			"needle - it is sqrt(1 - (minor/major)^2), so it moves slowly at "
			"first (a 2:1 ellipse already reads 0.87). Size- and "
			"rotation-independent, and moment-based, so it follows the MASS "
			"rather than the extreme pixels the way "
			"'min-area rect aspect (long/short)' does.",
		"bbox width": "Width of the UPRIGHT bounding box - a property of the "
			"region's orientation as much as its size. 'min-area rect long "
			"side' is the rotation-invariant one.",
		"bbox height": "Height of the UPRIGHT bounding box - a property of the "
			"region's orientation as much as its size. 'min-area rect short "
			"side' is the rotation-invariant one.",
		"aspect ratio": "The UPRIGHT bounding box's width/height, so it turns "
			"with the region and passes through 1 at 45 degrees: one 120x30 "
			"rectangle reads 3.90 upright and 0.26 on its side. Below 1 for a "
			"tall region. For elongation that does NOT depend on orientation "
			"use 'min-area rect aspect (long/short)'.",
		"loose ends": "How many FREE ENDS the region's skeleton has - thin it to "
			"one pixel wide and count the tips. Rotation-, scale- and "
			"reflection-independent, and it separates shapes every geometric "
			"column scores the same. For LINE-LIKE regions (wires, cracks, "
			"roots, vessels, scratches, strokes) it is the literal end count: a "
			"bar or an open curve 2, a fork or a T-junction 3, a cross 4, an "
			"unbroken loop 0 - so it doubles as a break test, a broken ring "
			"reading 2 where an intact one reads 0. For SOLID blobs it counts "
			"the tips of the medial axis, i.e. how many directions the shape "
			"comes to a point in: a round blob 0, a rectangle or an ellipse 2, "
			"a triangle 3, a five-pointed star 5. Two caveats, both measured: "
			"short stubs do not survive thinning, so read it as a count of "
			"PRONOUNCED ends; and a nearly ROUND solid blob is degenerate - its "
			"medial axis is a single point, so a ragged boundary sprouts spokes "
			"and a disc can read 4 instead of 0. Smooth the mask (an open, or a "
			"median blur) before measuring if its edge is jagged. Spur pruning "
			"is NOT the answer and was tried: at any depth that removes those "
			"spokes it also deletes real ends (a bar 2 -> 0, a star 5 -> 3).",
		"convexity defects (count, max depth)": "TWO columns describing the "
			"CONCAVE BITES in the outline: how many are deeper than 5% of "
			"sqrt(area), and the deepest one as a fraction of sqrt(area). "
			"'solidity' measures the total dented AREA and cannot tell one deep "
			"notch from several shallow ones - these can: a gear reads many "
			"shallow defects, a crescent or a pair of touching blobs one deep "
			"one, and a convex shape (disc, box, ellipse) reads 0 and 0.0. Both "
			"are scale-free and rotation-independent. The count is the number of "
			"'arms' or 'lobes' for a star-like region; the depth says how "
			"pronounced they are.",
		"corner count": "How many vertices the outline has once simplified to "
			"2% of its own perimeter (cv2.approxPolyDP) - the standard "
			"'triangle vs quad vs pentagon vs blob' test, unchanged by size or "
			"rotation. A rounded or noisy outline gives a larger count rather "
			"than a wrong one, so use it as an ordering (3, 4, 5, ... many) "
			"rather than an exact vertex identification. The 2% is fixed - a "
			"toggle carries no parameter of its own - so for fine control over "
			"the tolerance use 'Approximate Contours' plus 'Filter Contours' "
			"instead.",
		"min-area rect extent": "RECTANGULARITY: area / (long side * short "
			"side) of cv2.minAreaRect, between 0 and 1 - what 'extent' would be "
			"if the box were allowed to TURN with the region. A rectangle reads "
			"1.00 at every angle (0.96+ once rasterized off-axis) where "
			"'extent' swings 1.00 to 0.45; a thin diagonal bar reads 1.00 "
			"against 0.20. A circle is 0.79 (pi/4), a triangle 0.50, a cross "
			"0.56. Use it for 'is this thing a box?' regardless of orientation; "
			"'solidity' answers the different question 'is it dented?', and the "
			"two disagree on purpose - a triangle and a disc are solid (0.99) "
			"but not rectangular, while a cross is rectangular-ish (0.56) and "
			"dented (0.60).",
		"min-area rect long side": "Longer side of cv2.minAreaRect, the "
			"smallest rectangle of ANY rotation that contains the region - so "
			"it measures the shape, not how it is lying.",
		"min-area rect short side": "Shorter side of cv2.minAreaRect. With the "
			"long side this is the region's true extent, e.g. the width of a "
			"diagonal bar, which the upright bbox cannot express.",
		"min-area rect aspect (long/short)": "Elongation from cv2.minAreaRect, "
			"always long/short so it is >= 1 and unchanged by rotation: the "
			"same 120x30 rectangle reads 3.86-4.04 at every angle where "
			"'aspect ratio' swings 3.90 to 0.26. 1.0 means a square or a "
			"round blob; large means a sliver.",
		"orientation (sin/cos)": "TWO columns, sin(2t) and cos(2t): a raw angle "
			"wraps at 180 degrees, which would tear one axis into two clusters.",
		"hu moments (log magnitude)": "SEVEN columns, hu1..hu7 - the shape "
			"invariants, unchanged by where a region sits, how big it is or "
			"which way up it is. Raw |h| spans some 30 orders of magnitude "
			"across the seven, so each is rescaled to its ORDER OF MAGNITUDE "
			"above a floor: the value rises with |h| and reads 0 for anything "
			"too small to be shape rather than numerical residue. Every column "
			"then gets its sign back except hu7, the only invariant whose sign "
			"flips when the region is mirrored - keeping it would split a shape "
			"and its reflection into different clusters. hu5 and hu6 also take "
			"either sign but are UNCHANGED by reflection, so their sign is real "
			"shape information and stays: two regions can match on magnitude "
			"alone and still be different shapes. Tick 'chirality' for hu7's "
			"sign on its own.",
		"chirality": "The handedness the Hu magnitudes drop, as "
			"sign(h7)*|h7|^0.25: negative and positive are mirror images, and "
			"it is exactly 0 for a mirror-symmetric region. Rescaled rather "
			"than raw because |h7| is ~1e-14 and would otherwise swamp every "
			"other column.",
		"mean colour": "THREE columns, mean B, G and R inside the region - "
			"needs the 'image' input wired.",
		"mean intensity": "Mean gray inside the region - needs the 'image' "
			"input wired.",
		"ellipse major axis": "Long axis (2 standard deviations) of the "
			"equivalent ellipse - the same second-order moments 'eccentricity' "
			"uses, kept as a LENGTH. Moment-based, so it measures where the mass "
			"is: the min-area rect is extremal and stretches to the outermost "
			"pixel, this does not.",
		"ellipse minor axis": "Short axis (2 standard deviations) of the "
			"equivalent ellipse. With the major axis this is the region's spread; "
			"on an 'F' it reads 29 against a 47 major, because the mass hangs off "
			"one side.",
		"min-area rect centroid clearance (long, short)": "TWO columns: the "
			"SHORTEST distance from the region's centroid to each opposing pair "
			"of min-area rect walls - first the pair the long side separates, "
			"then the short one, matching the 'min-area rect long/short side' "
			"ordering. It says how far off-centre the mass sits inside the "
			"region's own oriented box, which separates a 'C' from an 'O' at the "
			"same size and elongation. Ordered by SIDE LENGTH, not by cv2's own "
			"axes: those swap as the rectangle turns past its angle convention, "
			"so an axis-ordered pair would be noise. Unsigned - a sign would need "
			"a stable which-end-is-which, which a min-area rect does not have - "
			"so it maxes out at half the side and equals it exactly for any "
			"centrally symmetric region.",
		"centroid offset in bbox (x, y)": "TWO columns, the centroid as a "
			"fraction of the UPRIGHT bbox (0..1) - position-invariant, so the "
			"same shape reads the same anywhere in the frame. Being bbox-relative "
			"it turns with the region like 'aspect ratio' does; use 'min-area "
			"rect centroid clearance' for the rotation-invariant version.",
		"canonical orientation (sin/cos)": "TWO columns, sin(t) and cos(t) of the "
			"0-360 orientation - unlike 'orientation (sin/cos)', which is mod 180, "
			"this tells a shape from its own 180-degree rotation, because the "
			"THIRD-order moments pick the head from the tail. Only meaningful "
			"when the region actually HAS a head: for a near-symmetric one the "
			"half-turn is noise (an 'O' measures 0.0006 on 'orientation "
			"confidence', an 'F' 0.021), so tick that column too and ignore or "
			"gate on it.",
		"orientation confidence": "|nu30| in the canonical frame: how well "
			"defined 'canonical orientation' is, i.e. how asymmetric the region "
			"is along its own long axis. Measured ~0.02 for an 'F', 0.0006 for an "
			"'O', exactly 0 for a square. Test against ~0.005, never against 0.",
		"hole count": "Number of enclosed background regions - 0 for an 'F', 1 "
			"for a 'D', 2 for a 'B'. Not a moment (Euler number = 1 - holes for a "
			"single region), but it is topological, so rotation, scale and "
			"reflection cannot change it. Counted with the connectivity DUAL to "
			"the 'connectivity' widget, which is what stops a diagonal 1px gap "
			"from being a hole and a component at the same time.",
		"median thickness": "How THICK the region is, in pixels, everywhere "
			"along its middle: the distance transform (every pixel's distance to "
			"the nearest background pixel) read on the MEDIAL AXIS and doubled, "
			"then the median over that axis. For anything line-like - a wire, a "
			"crack, a vessel, a road, a scratch, a pen stroke - this is the one "
			"measurement that matters, and no other column can give it: "
			"'min-area rect short side' boxes the WHOLE region, so a coiled wire "
			"of 11 px stroke reads ~220 there and 10.8 here. For a solid blob it "
			"is the local inscribed diameter, so a disc reads its own diameter "
			"and a 200x140 box reads 142. Three caveats, all measured: it is "
			"exact for an even width and +1 px for an odd one (the transform "
			"measures centre to background CENTRE); it needs a clean mask, "
			"because a ragged edge sprouts medial-axis spokes whose thin ends "
			"drag the median down (a rotated disc read 161 against a true 180); "
			"and for a CONVEX SOLID it is nearly the same number as 'min-area "
			"rect short side' (142 vs 140 for that box) - the columns part "
			"company on anything curved, branching or hollow.",
		"thickness variation": "How UNIFORM that thickness is: the standard "
			"deviation over the medial axis divided by its mean, so it is "
			"scale-free and rotation-independent. 0.00 for a constant-width slab "
			"or an L of uniform arms, 0.05 for a ring drawn with one pen, but "
			"0.46 for a bar that steps from 9 px to 25 px, 0.53 for a tapering "
			"wedge and 0.61 for a five-pointed star. It is the 'does this "
			"stroke/wire/vessel keep the same gauge, or does it neck, taper and "
			"bulge?' test. Must be read on the medial axis, which is what this "
			"column does - taken over ALL the region's pixels instead, a "
			"PERFECTLY uniform slab measures 0.495, because the edge pixels' "
			"tiny distances dominate.",
		"inscribed circle radius": "Radius of the LARGEST CIRCLE that fits "
			"inside the region (the peak of the distance transform) - the "
			"region's widest point, where 'median thickness' is its typical one. "
			"A length in pixels, scaling exactly with size and steadier under "
			"rotation than any other shape column measured here (a disc 90.0 "
			"-> 89.6 at 37 degrees), because one interior maximum does not care "
			"about a ragged boundary. Use it for the 'core' size of a blob with "
			"arms or spikes: a five-pointed star reads 45 where its enclosing "
			"circle is 100. Also the classic label-anchor point - the same "
			"circle's CENTRE is the deepest point inside the shape.",
		"roundness (inscribed/enclosing)": "Inscribed circle radius / minimum "
			"ENCLOSING circle radius, 0 to 1: 1.00 for a disc, 0.58 for a "
			"rectangle, 0.50 for a triangle, 0.45 for a five-pointed star, 0.17 "
			"for an L and 0.06 for a bar or a ring. It answers the same question "
			"as 'circularity' - how compact is this thing - but from the two "
			"circles instead of from the perimeter, which makes it the "
			"NOISE-PROOF version: roughen a disc's edge and circularity collapses "
			"from 0.892 to 0.224 (the staircase inflates the squared perimeter) "
			"and solidity drops to 0.885, while roundness only moves 1.000 -> "
			"0.884. Rotation- and scale-invariant to ~1%. Reach for it whenever "
			"the masks come from a threshold, a segmentation net or anything "
			"else with a jagged outline.",
		"skeleton length": "Total length of the region's medial axis in pixels - "
			"the PATH length of a curvilinear region, which no bounding box can "
			"measure: a 100 px-radius arc reads 513 against a 211 px 'min-area "
			"rect long side', a coiled or meandering wire 412 against 329, while "
			"a STRAIGHT bar reads 199 against 210, i.e. the two agree exactly "
			"when the path is straight and diverge by however much it bends. "
			"With 'median thickness' it also factorises the region: length x "
			"thickness reproduces the area to within a few percent for anything "
			"line-like, and their ratio is a rotation-invariant elongation that "
			"follows the curve (16.6 for a bar, 61 for a ring). Degenerate for a "
			"round solid, whose whole axis is one point (a disc reads 1). "
			"Diagonal steps count as one pixel, so a 45-degree path measures "
			"~1/1.41 of its true length.",
		"radial distance variation": "Standard deviation / mean of the distance "
			"from the region's centroid to each point of its outline - the "
			"classic 'how far from a circle is this outline?' roughness measure, "
			"and the most stable column here under rotation and scale (it "
			"repeats to three decimals). 0.003 for a disc or a ring, 0.16 for a "
			"rectangle, 0.23 for a triangle or a star, 0.35 for a crescent, 0.42 "
			"for an L, 0.56 for a bar. It rises with ELONGATION as well as with "
			"raggedness and cannot tell a triangle from a star (0.226 vs 0.229), "
			"so read it as 'how much does the boundary wander', and pair it with "
			"'roundness' or 'corner count' when the two have to be told apart. "
			"Measured over EVERY outline pixel, not the polygon vertices, so it "
			"does not depend on how the contour happens to be stored.",
	},
)
