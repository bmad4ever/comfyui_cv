# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Feature detection / matching building blocks (NPARRAY-native).

Small composable nodes for the geometry exercises (homography, fundamental
matrix, panorama...): they cover the class-based cv2 APIs (ORB, SIFT,
BFMatcher, FLANN) that the auto-generated function wrappers in lowlevel.py
cannot reach, plus a few glue nodes (corners, size, polygon drawing) that
keep everything chainable as raw ndarrays - no tensor round trips between
processing steps.

Custom types:
* CV_KEYPOINTS - list of cv2.KeyPoint
* CV_MATCHES   - list of cv2.DMatch (sorted by distance)
"""

import logging
import math
import os
from ast import literal_eval

import cv2
import numpy as np
import torch

import comfy.utils
from comfy_api.latest import io

from . import enums, imageutils, polytype, scalars, tuples, warpfit, workerutils
from .convert import NPArray

CATEGORY = "image/CV/features"

KeyPoints = io.Custom("CV_KEYPOINTS")
Matches = io.Custom("CV_MATCHES")

def _feature2d(name):
	"""Resolve a Feature2D factory across the OpenCV 4/5 module split.

	OpenCV 5 moved BRISK/AKAZE/KAZE out of the cv2 root and into
	cv2.xfeatures2d, so `cv2.AKAZE_create` - which worked on the 4.x build the
	registry was generated against - raises AttributeError here. Look in both
	places and fail with a readable message rather than an AttributeError deep
	inside execute().
	"""
	factory = f"{name}_create"
	for holder in (cv2, getattr(cv2, "xfeatures2d", None)):
		fn = getattr(holder, factory, None)
		if fn is not None:
			return fn
	raise RuntimeError(
		f"This OpenCV build ({cv2.__version__}) has no {factory} in cv2 or "
		"cv2.xfeatures2d - pick another detector/descriptor. (OpenCV 5 moved "
		"BRISK/AKAZE/KAZE into the contrib xfeatures2d module; a plain "
		"opencv-python wheel installed over opencv-contrib-python(-headless) removes it - "
		"see tools/repair_opencv_contrib.py.)")


_DETECTORS = {
	"ORB": lambda n: cv2.ORB_create(nfeatures=n),
	"SIFT": lambda n: cv2.SIFT_create(nfeatures=n),
	"AKAZE": lambda n: _feature2d("AKAZE")(),
	"BRISK": lambda n: _feature2d("BRISK")(),
}


def _to_gray_u8(arr: np.ndarray) -> np.ndarray:
	arr = imageutils.ensure_uint8(np.asarray(arr))
	if arr.ndim == 3 and arr.shape[2] >= 3:
		return cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
	if arr.ndim == 3 and arr.shape[2] == 1:
		return arr[..., 0]
	if arr.ndim == 2:
		return arr
	raise ValueError(f"Cannot interpret array of shape {arr.shape} as an image.")


_to_bgr_u8 = imageutils._to_bgr_u8  # re-export for backward compat


def parse_color(color_str: str, n_channels: int) -> tuple:
	"""Parse a color string into an n_channels-length int tuple.

	Rules (scalars.parse, the one shared with 'CV Scalar' and every low-level
	cv2 Scalar widget), plus the two a DRAWING color needs on top: the result is
	rounded to ints, and it is fitted to the canvas' channel count.
	- Single value "255" -> broadcast to all channels: (255, 255, 255)
	- Tuple "(0, 0, 255)" -> use as-is if length == n_channels
	- Tuple longer than n_channels -> truncate, log warning
	- Tuple shorter than n_channels -> pad with zeros
	Float values in the tuple are accepted and cast to int (round if needed).
	"""
	# max_len=None: a color literal longer than a cv2 Scalar is truncated with a
	# warning here rather than refused - the canvas' channel count is what
	# decides, and it may be 1
	comps, broadcast = scalars.parse(color_str, "color", width=n_channels,
	                                 max_len=None)
	vals = tuple(int(round(c)) for c in comps)
	if broadcast:
		return vals

	if len(vals) > n_channels:
		logging.warning(
			"Color tuple %s has %d elements but image has %d channels; "
			"extra elements ignored.", vals, len(vals), n_channels
		)
		vals = vals[:n_channels]
	elif len(vals) < n_channels:
		vals = vals + (0,) * (n_channels - len(vals))

	return vals


def halo_color(bgr) -> tuple:
	"""The contrasting halo colour for text drawn in `bgr`.

	Black behind light text, white behind dark text. A fixed black halo is
	worse than none at all when the text itself is dark: it draws black on
	black and the caption disappears into the background it was meant to be
	lifted off (this is what black `color` + `outline` used to do).
	`bgr` is whatever `parse_color` returned, so 1..4 channels.
	"""
	if isinstance(bgr, (int, float)):
		return 0 if float(bgr) >= 128 else 255
	vals = tuple(bgr)
	if len(vals) >= 3:
		b, g, r = float(vals[0]), float(vals[1]), float(vals[2])
		lum = 0.0722 * b + 0.7152 * g + 0.2126 * r
	else:
		lum = sum(float(v) for v in vals[:2]) / max(1, len(vals[:2]))
	level = 0 if lum >= 128.0 else 255
	# an alpha channel stays opaque; the colour channels take the halo level
	if len(vals) == 4:
		return (level, level, level, vals[3])
	return tuple(level for _ in vals)


# ---------------------------------------------------------------------------
# redundant encoding: a second, colour-free channel for the drawing nodes
# ---------------------------------------------------------------------------
#
# Two overlaid point sets or line sets that differ only in COLOUR are
# unreadable for a colour-blind viewer and in greyscale. Every drawing node
# already offers exactly one non-colour channel (radius or thickness), which is
# not enough when both sets want to stay thin. Shape and line style are the
# missing ones - they are what a print figure uses, and they cost no palette.

# Dash patterns as (on, off, ...) run lengths in pixels, scaled by thickness so
# a fat dashed line still reads as dashed.
LINE_STYLES = ("solid", "dashed", "dotted", "dash-dot")
_DASH_PATTERNS = {
	"solid": (),
	"dashed": (11.0, 7.0),
	"dotted": (1.0, 5.0),
	"dash-dot": (13.0, 5.0, 1.0, 5.0),
}

LINE_STYLE_TOOLTIP = (
	"Stroke pattern. Colour alone cannot separate two overlaid line sets for a "
	"colour-blind viewer or in greyscale - give the second set a different "
	"style and the picture reads without the palette. 'solid' is cv2's own "
	"line."
)

# Marker shapes for point sets, mapped to the cv2.drawMarker types where one
# exists; the circle/square fills are drawn directly.
MARKER_SHAPES = ("filled circle", "open circle", "filled square", "open square",
                 "cross (+)", "x (tilted cross)", "star", "diamond",
                 "triangle up", "triangle down")
_CV_MARKERS = {
	"open square": cv2.MARKER_SQUARE,
	"cross (+)": cv2.MARKER_CROSS,
	"x (tilted cross)": cv2.MARKER_TILTED_CROSS,
	"star": cv2.MARKER_STAR,
	"diamond": cv2.MARKER_DIAMOND,
	"triangle up": cv2.MARKER_TRIANGLE_UP,
	"triangle down": cv2.MARKER_TRIANGLE_DOWN,
}

MARKER_SHAPE_TOOLTIP = (
	"Marker shape. Two point sets drawn on one image in two colours are the "
	"same picture to a colour-blind viewer - give the second set a different "
	"shape (or radius) and the distinction survives. 'filled circle' is the "
	"original marker."
)


def draw_marker(canvas, x, y, radius, color, shape="filled circle"):
	"""Stamp one point marker of the named shape."""
	center = (int(round(x)), int(round(y)))
	radius = max(1, int(radius))
	if shape == "filled circle":
		cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
	elif shape == "open circle":
		# at least 2 px: a 1 px anti-aliased ring never reaches its own colour
		# at any pixel, so it reads as a faint smudge over a photo - which
		# defeats the point of having a second, colour-free channel
		cv2.circle(canvas, center, radius, color, max(2, radius // 3), cv2.LINE_AA)
	elif shape == "filled square":
		cv2.rectangle(canvas, (center[0] - radius, center[1] - radius),
		              (center[0] + radius, center[1] + radius), color, -1)
	else:
		cv2.drawMarker(canvas, center, color,
		               _CV_MARKERS.get(shape, cv2.MARKER_CROSS),
		               radius * 2, max(1, radius // 2), cv2.LINE_AA)


def draw_styled_line(canvas, p0, p1, color, thickness, style="solid"):
	"""One segment in the named line style.

	cv2 has no stroke pattern, so a dashed/dotted line is walked by arc length
	and emitted as short solid pieces. Sub-pixel positions are kept in float
	until each piece is stamped, so a dash pattern does not drift along a long
	line.
	"""
	pattern = _DASH_PATTERNS.get(style, ())
	x0, y0 = float(p0[0]), float(p0[1])
	x1, y1 = float(p1[0]), float(p1[1])
	if not pattern:
		cv2.line(canvas, (int(round(x0)), int(round(y0))),
		         (int(round(x1)), int(round(y1))), color, thickness, cv2.LINE_AA)
		return
	length = math.hypot(x1 - x0, y1 - y0)
	if length <= 0.0:
		cv2.line(canvas, (int(round(x0)), int(round(y0))),
		         (int(round(x0)), int(round(y0))), color, thickness, cv2.LINE_AA)
		return
	ux, uy = (x1 - x0) / length, (y1 - y0) / length
	scale = max(1.0, float(thickness))
	runs = [max(1.0, run * scale) for run in pattern]
	pos, i = 0.0, 0
	while pos < length:
		run = runs[i % len(runs)]
		end = min(pos + run, length)
		if i % 2 == 0:   # even runs are "pen down"
			cv2.line(canvas,
			         (int(round(x0 + ux * pos)), int(round(y0 + uy * pos))),
			         (int(round(x0 + ux * end)), int(round(y0 + uy * end))),
			         color, thickness, cv2.LINE_AA)
		pos, i = end, i + 1


def draw_styled_polyline(canvas, pts, color, thickness, style="solid", closed=False):
	"""A polyline (optionally closed) in the named line style."""
	pts = np.asarray(pts).reshape(-1, 2)
	if len(pts) < 2:
		return
	pairs = list(zip(pts[:-1], pts[1:]))
	if closed:
		pairs.append((pts[-1], pts[0]))
	for a, b in pairs:
		draw_styled_line(canvas, a, b, color, thickness, style)


def _parse_size_literal(value, name="size") -> tuple[int, int]:
	if isinstance(value, str):
		try:
			value = literal_eval(value.strip())
		except Exception as e:
			raise ValueError(
				f'{name} must be a Size literal like "(4, 11)", got {value!r}.'
			) from e
	if not isinstance(value, (list, tuple)) or len(value) != 2:
		raise ValueError(f"{name} must be a 2-element Size, got {value!r}.")
	w, h = int(value[0]), int(value[1])
	if w <= 0 or h <= 0:
		raise ValueError(f"{name} values must be positive, got {(w, h)!r}.")
	return w, h


def _make_draw_canvas(image, tag):
	"""Build a drawable canvas from a resolved image-ish value.

	For a raw NPARRAY, MASK or LATENT the array's own channel layout is
	preserved (no BGR expansion - a grayscale NPARRAY stays single-channel,
	a 4-channel NPARRAY stays 4-channel). For a ComfyUI IMAGE (tag="image")
	it is converted to a uint8 BGR frame, since cv2 draws in BGR order.

	Returns (canvas, n_channels). n_channels is read from the *canvas* (after
	any conversion) so a color literal parsed with it always matches the
	actual drawable - see parse_color(), which already applies the channel
	rules (single value broadcasts; short tuples zero-pad; long tuples
	truncate).
	"""
	if tag == "image":
		canvas = _to_bgr_u8(image).copy()
	else:
		canvas = np.asarray(image).copy()
	n_channels = 1 if canvas.ndim == 2 else canvas.shape[2]
	return canvas, n_channels


def draw_text_outlined(canvas, text, org, font_scale, halo, fg, thickness,
                       font_face=cv2.FONT_HERSHEY_SIMPLEX) -> np.ndarray:
	"""Stamp `text` with a centred contrasting halo, correct at any thickness.

	`cv2.putText` with LINE_AA rasterizes a 1px stroke on a sub-pixel grid that
	is ~0.5-1px off-centre from strokes of thickness >= 2 (which centre on the
	integer grid). Drawing the halo and the text as two putText calls - the
	"thick halo, thin text" pattern - therefore shows the halo as an offset
	copy behind a 1px text. Instead the halo is a dilation of the text's own
	mask, so it is a symmetric ring around exactly the pixels the text will
	occupy (the dilation radius is `thickness`, matching the margin the old
	`thickness*3` stroke produced). Works for 1..4 channel canvases; `halo`
	and `fg` are whatever `halo_color`/`parse_color` returned (int or tuple).

	`canvas` is modified in place and returned; `org` is the putText baseline
	(bottom-left) origin. Pixels outside the canvas are clipped.
	"""
	if not text:
		return canvas
	(th_px, text_h), baseline = cv2.getTextSize(text, font_face, font_scale,
	                                            int(thickness))
	pad = max(3, int(thickness) + 2)
	mask = np.zeros((text_h + baseline + 2 * pad, th_px + 2 * pad), np.uint8)
	cv2.putText(mask, text, (pad, text_h + baseline + pad), font_face,
	            font_scale, 255, int(thickness), cv2.LINE_AA)
	dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
	                 iterations=max(1, int(thickness)))
	canvas_h, canvas_w = canvas.shape[:2]
	sx0, sy0 = max(0, org[0] - pad), max(0, org[1] - (text_h + baseline + pad))
	sx1, sy1 = (min(canvas_w, org[0] + th_px + pad),
	            min(canvas_h, org[1] - (text_h + baseline + pad) + mask.shape[0]))
	if sx1 <= sx0 or sy1 <= sy0:
		return canvas
	mx0, my0 = sx0 - (org[0] - pad), sy0 - (org[1] - (text_h + baseline + pad))
	sub = canvas[sy0:sy1, sx0:sx1]
	mask_px = mask[my0:my0 + sub.shape[0], mx0:mx0 + sub.shape[1]]
	dil_px = dil[my0:my0 + sub.shape[0], mx0:mx0 + sub.shape[1]]
	if sub.ndim == 3:
		halo_a = (dil_px.astype(np.float64) / 255.0)[..., None]
		text_a = (mask_px.astype(np.float64) / 255.0)[..., None]
		sub[...] = (sub * (1 - halo_a)
		            + np.asarray(halo, np.float64) * halo_a)
		sub[...] = (sub * (1 - text_a)
		            + np.asarray(fg, np.float64) * text_a)
	else:
		halo_val = float(np.asarray(halo).ravel()[0])
		fg_val = float(np.asarray(fg).ravel()[0])
		halo_a = dil_px.astype(np.float64) / 255.0
		text_a = mask_px.astype(np.float64) / 255.0
		sub[...] = (sub * (1 - halo_a) + halo_val * halo_a)
		sub[...] = (sub * (1 - text_a) + fg_val * text_a)
	return canvas


class DetectFeatures(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DetectFeatures",
			display_name="CV Detect Features",
			category=CATEGORY,
			description="Detects local features and computes their descriptors "
			            "(cv2.ORB/SIFT/AKAZE/BRISK detectAndCompute). Color input "
			            "is converted to grayscale internally; enable CLAHE to "
			            "equalize contrast first for low-contrast or unevenly lit "
			            "images. Chain into 'CV Match Features'. Finding zero "
			            "keypoints is a valid result (count = 0), not an error - "
			            "the downstream match/homography nodes handle it, so a "
			            "failed detection never halts the workflow.",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Image to detect features in (BGR or gray)."),
				io.Combo.Input("detector", options=list(_DETECTORS), default="ORB",
					tooltip="ORB is fast and free; SIFT is slower but more robust "
					        "to scale/rotation changes. AKAZE/BRISK ignore "
					        "max_features."),
				io.Int.Input("max_features", default=2000, min=0, max=100000,
					tooltip="Maximum keypoints to keep (ORB/SIFT only). "
					        "ORB: 0 means keep 0 features (empty output); "
					        "SIFT: 0 means unlimited (keep all detected features). "
					        "AKAZE/BRISK ignore this input."),
				io.Boolean.Input("clahe", default=False,
					tooltip="Apply CLAHE contrast equalization before detection - "
					        "helps on dim or low-contrast images."),
				polytype.poly_input("mask", types=polytype.MASK_ONLY, optional=True,
					tooltip="Optional mask: detect only where mask > 0."),
			],
			outputs=[
				KeyPoints.Output(display_name="keypoints"),
				NPArray.Output(display_name="descriptors"),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, image, detector, max_features, clahe, mask=None) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		mask, _ = polytype.resolve(mask)
		gray = _to_gray_u8(image)
		if clahe:
			gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
		if mask is not None:
			mask = imageutils.fit_mask_to(imageutils.ensure_uint8(np.asarray(mask)), gray.shape[:2])
		det = _DETECTORS[detector](max_features)
		keypoints, descriptors = det.detectAndCompute(gray, mask)
		if descriptors is None:  # nothing found: empty result with the right dtype
			dtype = np.uint8 if det.descriptorType() == cv2.CV_8U else np.float32
			descriptors = np.zeros((0, det.descriptorSize()), dtype)
		return io.NodeOutput(list(keypoints), descriptors, len(keypoints))


class FilterKeypoints(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FilterKeypoints",
			display_name="CV Filter Keypoints",
			category=CATEGORY,
			description="Keeps a robust subset of keypoints (and their paired "
			            "descriptors) by scale and strength. Use it on a "
			            "high-resolution reference object to drop the many tiny, "
			            "fine-texture keypoints a detector finds there: those "
			            "small-scale features have no counterpart once the object "
			            "appears smaller in the cluttered scene, so they only add "
			            "ambiguous matches and waste the descriptor budget. Prefer "
			            "this over guessing a 'max_features' count - filtering by "
			            "size/response keeps the structurally meaningful features "
			            "regardless of how many there are. Descriptors are filtered "
			            "in lockstep so they stay aligned with the kept keypoints. "
			            "Keeping zero keypoints is a valid result (count = 0), not "
			            "an error.",
			inputs=[
				KeyPoints.Input("keypoints",
					tooltip="Keypoints to filter (from 'CV Detect Features')."),
				NPArray.Input("descriptors",
					tooltip="Descriptors paired with keypoints (same node); kept "
					        "rows stay aligned with the kept keypoints."),
				io.Float.Input("min_size", default=0.0, min=0.0, max=1000.0, step=1.0,
					tooltip="Drop keypoints whose diameter (scale, in pixels of the "
					        "image they were detected in) is below this. 0 = keep "
					        "all. This is an ABSOLUTE pixel size: if the object "
					        "appears much smaller in the scene than in the "
					        "reference, keep this low (or only filter the reference)."),
				io.Float.Input("min_response", default=0.0, min=0.0, max=1.0, step=0.001,
					tooltip="Drop weak keypoints whose detector response (corner/"
					        "contrast strength) is below this. 0 = keep all."),
				io.Int.Input("top_k", default=0, min=0, max=100000,
					tooltip="After the size/response filters, keep only the K "
					        "strongest by response. 0 = keep all."),
			],
			outputs=[
				KeyPoints.Output(display_name="keypoints"),
				NPArray.Output(display_name="descriptors"),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, keypoints, descriptors, min_size, min_response, top_k) -> io.NodeOutput:
		kps = list(keypoints)
		desc = np.asarray(descriptors)
		idx = [i for i, k in enumerate(kps)
		       if k.size >= min_size and k.response >= min_response]
		if top_k and len(idx) > top_k:  # keep the strongest, then restore order
			idx = sorted(sorted(idx, key=lambda i: kps[i].response, reverse=True)[:top_k])
		kept_kps = [kps[i] for i in idx]
		# descriptors are row-aligned with keypoints out of detectAndCompute; keep
		# the matching rows (empty slice preserves dtype/columns when nothing passes)
		kept_desc = desc[idx] if idx else desc[:0]
		return io.NodeOutput(kept_kps, kept_desc, len(kept_kps))


class MatchFeatures(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MatchFeatures",
			display_name="CV Match Features",
			category=CATEGORY,
			description="Matches two descriptor sets with BFMatcher or FLANN. "
			            "Outputs matches sorted by distance and matched point "
			            "coordinates as Nx1x2 float32 arrays, ready for "
			            "'CV Find Homography (RANSAC)' / "
			            "cv2.findFundamentalMat / cv2.estimateAffine2D. Zero "
			            "matches is a valid result, not an error - check the "
			            "count (or the homography node's 'found' flag) downstream.",
			inputs=[
				KeyPoints.Input("keypoints_a",
					tooltip="Keypoints of image A (from 'CV Detect Features')."),
				NPArray.Input("descriptors_a",
					tooltip="Descriptors of image A, paired with keypoints_a. Must "
					        "use the same detector as image B."),
				KeyPoints.Input("keypoints_b",
					tooltip="Keypoints of image B (from 'CV Detect Features')."),
				NPArray.Input("descriptors_b",
					tooltip="Descriptors of image B, paired with keypoints_b."),
				io.Combo.Input("matcher",
					options=["BFMatcher", "FLANN"],
					default="BFMatcher",
					tooltip="BFMatcher: brute-force, exact, fast for small/mid "
					        "descriptor sets. FLANN: approximate nearest "
					        "neighbors, faster for large sets (10k+ descriptors)."),
				io.Combo.Input("norm_type",
					options=["auto"] + enums.NORM.options,
					default="auto",
					tooltip="Distance norm. 'auto' = HAMMING for uint8 (ORB/BRISK/AKAZE), "
					        "L2 for float32 (SIFT). Override only if you know "
					        "what you are doing."),
				io.Combo.Input("match_mode",
					options=["ratio test (Lowe)", "cross-check", "radius"],
					default="ratio test (Lowe)",
					tooltip="Ratio test: keep matches clearly better than the "
					        "runner-up (robust default). Cross-check: keep only "
					        "mutual best matches (BFMatcher only). Radius: keep "
					        "all matches within max distance (all matchers)."),
				io.Float.Input("ratio", default=0.75, min=0.1, max=1.0, step=0.05,
					tooltip="Lowe ratio threshold; lower = stricter. Ignored for "
					        "cross-check and radius modes."),
				io.Int.Input("k", default=2, min=2, max=32,
					tooltip="k for kNN matching. 2 = Lowe's ratio test (standard). "
					        "Higher k considers more candidates but is slower. "
					        "Ignored for cross-check mode.",
					optional=True, advanced=True),
				io.Float.Input("radius", default=0.0, min=0.0, step=0.1,
					tooltip="Max distance for radius matching. 0 = no limit "
					        "(returns all matches). Only used when mode = radius.",
					optional=True),
				io.Combo.Input("flann_algorithm",
					options=["auto"] + enums.FLANN_INDEX.options,
					default="auto",
					tooltip="FLANN index algorithm. 'auto' = KDTREE for float32 "
					        "descriptors, LSH for binary (uint8). Override to "
					        "force a specific algorithm.",
					optional=True, advanced=True),
				io.Int.Input("flann_trees", default=5, min=1, max=16,
					tooltip="FLANN KDTREE: number of randomized trees. More = "
					        "better accuracy, slower build.",
					optional=True, advanced=True),
				io.Int.Input("flann_checks", default=50, min=1, max=1000,
					tooltip="FLANN search: number of leaves to check. Higher = "
					        "more accurate but slower.",
					optional=True, advanced=True),
				io.Int.Input("flann_table_number", default=6, min=1, max=30,
					tooltip="FLANN LSH: number of hash tables. Only used for "
					        "binary descriptors.",
					optional=True, advanced=True),
				io.Int.Input("flann_key_size", default=12, min=4, max=32,
					tooltip="FLANN LSH: key bits per table. Only used for "
					        "binary descriptors.",
					optional=True, advanced=True),
				io.Int.Input("flann_multi_probe_level", default=1, min=0, max=5,
					tooltip="FLANN LSH: extra search probes per table. 0 = "
					        "standard LSH, higher = more accurate.",
					optional=True, advanced=True),
			],
			outputs=[
				Matches.Output(display_name="matches"),
				NPArray.Output(display_name="points_a",
					tooltip="Matched keypoint coordinates in image A, Nx1x2 float32."),
				NPArray.Output(display_name="points_b",
					tooltip="Matched keypoint coordinates in image B, Nx1x2 float32."),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, keypoints_a, descriptors_a, keypoints_b, descriptors_b,
	            matcher, norm_type, match_mode, ratio=0.75, k=2,
	            radius=0.0, flann_algorithm="auto", flann_trees=5,
	            flann_checks=50, flann_table_number=6, flann_key_size=12,
	            flann_multi_probe_level=1) -> io.NodeOutput:
		da, db = np.asarray(descriptors_a), np.asarray(descriptors_b)
		if da.dtype != db.dtype or da.shape[1:] != db.shape[1:]:
			raise ValueError(
				f"Descriptor sets are incompatible ({da.dtype} {da.shape} vs "
				f"{db.dtype} {db.shape}) - both images must use the same detector."
			)

		# --- resolve norm ---
		if str(norm_type) == "auto":
			norm = cv2.NORM_HAMMING if da.dtype == np.uint8 else cv2.NORM_L2
		else:
			norm = enums.NORM.resolve(norm_type)

		n_k = max(2, int(k))

		if len(da) == 0 or len(db) == 0:
			matches = []
		elif str(matcher) == "FLANN":
			matches = cls._match_flann(
				da, db, norm, match_mode, ratio, n_k, radius,
				flann_algorithm, flann_trees, flann_checks,
				flann_table_number, flann_key_size, flann_multi_probe_level)
		else:
			matches = cls._match_bf(da, db, norm, match_mode, ratio, n_k, radius)

		matches = sorted(matches, key=lambda m: m.distance)

		pts_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
		pts_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
		return io.NodeOutput(matches, pts_a, pts_b, len(matches))

	@staticmethod
	def _filter_ratio(pairs, ratio):
		"""Keep only matches whose distance is < ratio * runner-up distance."""
		result = []
		for group in pairs:
			if len(group) < 2:
				continue
			if group[0].distance < ratio * group[1].distance:
				result.append(group[0])
		return result

	@staticmethod
	def _match_bf(da, db, norm, match_mode, ratio, n_k, radius):
		"""BFMatcher: cross-check, ratio test, or radius matching."""
		if match_mode == "cross-check":
			return cv2.BFMatcher(norm, crossCheck=True).match(da, db)
		elif match_mode == "radius":
			raw = cv2.BFMatcher(norm).radiusMatch(da, db, maxDistance=radius)
			return [m for pairs in raw for m in pairs]
		else:
			pairs = cv2.BFMatcher(norm).knnMatch(da, db, k=n_k)
			return MatchFeatures._filter_ratio(pairs, ratio)

	@staticmethod
	def _match_flann(da, db, norm, match_mode, ratio, n_k, radius,
	                 algorithm, trees, checks, table_number, key_size,
	                 multi_probe_level):
		"""FLANN matcher with configurable index/search params."""
		is_float = (da.dtype == np.float32)

		# --- resolve algorithm ---
		if str(algorithm) == "auto":
			flann_algo = 1 if is_float else 6  # KDTREE / LSH
		else:
			flann_algo = enums.FLANN_INDEX.resolve(algorithm)

		# --- build index_params ---
		if flann_algo == 1:  # KDTREE
			index_params = dict(algorithm=1, trees=max(1, int(trees)))
		elif flann_algo == 6:  # LSH
			index_params = dict(algorithm=6, table_number=max(1, int(table_number)),
			                    key_size=max(4, int(key_size)),
			                    multi_probe_level=max(0, int(multi_probe_level)))
		else:
			index_params = dict(algorithm=flann_algo)

		search_params = dict(checks=max(1, int(checks)))
		flann = cv2.FlannBasedMatcher(index_params, search_params)

		if match_mode == "radius":
			raw = flann.radiusMatch(da, db, maxDistance=radius)
			return [m for pairs in raw for m in pairs]
		else:
			pairs = flann.knnMatch(da, db, k=n_k)
			return MatchFeatures._filter_ratio(pairs, ratio)


class FindHomographyRANSAC(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FindHomography",
			display_name="CV Find Homography (RANSAC)",
			category=CATEGORY,
			description="Estimates the homography between two matched point "
			            "sets (cv2.findHomography). Robust by design: with too "
			            "few points or a degenerate configuration it does NOT "
			            "fail the workflow - it returns found=false, an identity "
			            "homography (safe to feed into perspectiveTransform/"
			            "warpPerspective) and 0 inliers. 'found' is true only "
			            "when the estimate has at least min_inliers RANSAC "
			            "inliers - branch on it with a control-flow node (e.g. "
			            "Basic data handling's 'if/else') to bypass drawing or "
			            "compositing when nothing was found.",
			inputs=[
				NPArray.Input("points_a", tooltip="Source points, Nx1x2 (from 'CV Match Features')."),
				NPArray.Input("points_b", tooltip="Destination points, Nx1x2."),
				io.Combo.Input("method", options=enums.ROBUST_METHOD.options,
					default=enums.ROBUST_METHOD.default,
					tooltip="Robust estimator. RANSAC is the standard choice "
					        "for feature matches (which always contain outliers)."),
				io.Float.Input("reproj_threshold", default=3.0, min=0.1, max=100.0, step=0.5,
					tooltip="Maximum reprojection error in pixels for a match "
					        "to count as an inlier."),
				io.Int.Input("min_inliers", default=10, min=4, max=10000,
					tooltip="Minimum RANSAC inliers for 'found' to be true. "
					        "Raise it to reject false positives."),
			],
			outputs=[
				NPArray.Output(display_name="homography",
					tooltip="3x3 float64 matrix mapping points_a to points_b "
					        "(identity if not found)."),
				NPArray.Output(display_name="inlier_mask",
					tooltip="Nx1 uint8: 1 = inlier. Feed into 'CV Draw Matches'."),
				io.Int.Output(display_name="inlier_count"),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, points_a, points_b, method, reproj_threshold, min_inliers) -> io.NodeOutput:
		pts_a = np.asarray(points_a, np.float32).reshape(-1, 1, 2)
		pts_b = np.asarray(points_b, np.float32).reshape(-1, 1, 2)
		if len(pts_a) != len(pts_b):
			raise ValueError(
				f"points_a has {len(pts_a)} points but points_b has {len(pts_b)} - "
				"both must come from the same 'CV Match Features' node."
			)
		not_found = (np.eye(3, dtype=np.float64),
		             np.zeros((len(pts_a), 1), np.uint8), 0, False)
		if len(pts_a) < 4:
			return io.NodeOutput(*not_found)
		H, mask = cv2.findHomography(pts_a, pts_b, enums.ROBUST_METHOD.resolve(method),
		                             reproj_threshold)
		if H is None or not np.all(np.isfinite(H)):
			return io.NodeOutput(*not_found)
		mask = np.asarray(mask, np.uint8).reshape(-1, 1)
		inliers = int(mask.sum())
		if inliers < min_inliers:
			return io.NodeOutput(np.eye(3, dtype=np.float64), mask, inliers, False)
		return io.NodeOutput(H, mask, inliers, True)


class FindFundamentalMat(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FindFundamentalMat",
			display_name="CV Find Fundamental Matrix",
			category=CATEGORY,
			description="Estimates the fundamental matrix between two views of "
			            "a static scene (cv2.findFundamentalMat) from matched "
			            "point sets. Same failure-tolerant contract as 'OpenCV "
			            "Find Homography': with fewer than 8 points or a "
			            "degenerate configuration it returns found=false, an "
			            "all-zero matrix and 0 inliers instead of stopping the "
			            "workflow - branch on 'found' with a control-flow node. "
			            "Note: points from a single plane (poster, wall) are a "
			            "degenerate configuration for F - use views of a scene "
			            "with depth and a translated (not only rotated) camera.",
			inputs=[
				NPArray.Input("points_a", tooltip="Points in image 1, Nx1x2 (from 'CV Match Features')."),
				NPArray.Input("points_b", tooltip="Corresponding points in image 2, Nx1x2."),
				io.Combo.Input("method", options=enums.FM_METHOD.options,
					default=enums.FM_METHOD.default,
					tooltip="FM_RANSAC is the standard choice for feature "
					        "matches; FM_8POINT uses all points (only for "
					        "outlier-free input); FM_7POINT needs exactly 7."),
				io.Float.Input("reproj_threshold", default=3.0, min=0.1, max=100.0, step=0.5,
					tooltip="Maximum distance in pixels from a point to its "
					        "epipolar line for it to count as an inlier "
					        "(RANSAC only)."),
				io.Float.Input("confidence", default=0.99, min=0.5, max=1.0, step=0.01,
					tooltip="Desired probability that the estimate is correct "
					        "(RANSAC/LMedS only)."),
				io.Int.Input("min_inliers", default=15, min=8, max=10000,
					tooltip="Minimum inliers for 'found' to be true. Raise it "
					        "to reject accidental fits."),
			],
			outputs=[
				NPArray.Output(display_name="fundamental",
					tooltip="3x3 float64 fundamental matrix (all zeros if not "
					        "found - 'CV Draw Epipolar Lines' skips the "
					        "degenerate lines it produces)."),
				NPArray.Output(display_name="inlier_mask",
					tooltip="Nx1 uint8: 1 = inlier. Feed into 'CV Draw "
					        "Matches' / 'CV Draw Epipolar Lines'."),
				io.Int.Output(display_name="inlier_count"),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, points_a, points_b, method, reproj_threshold, confidence,
	            min_inliers) -> io.NodeOutput:
		pts_a = np.asarray(points_a, np.float32).reshape(-1, 1, 2)
		pts_b = np.asarray(points_b, np.float32).reshape(-1, 1, 2)
		if len(pts_a) != len(pts_b):
			raise ValueError(
				f"points_a has {len(pts_a)} points but points_b has {len(pts_b)} - "
				"both must come from the same 'CV Match Features' node."
			)
		not_found = (np.zeros((3, 3), np.float64),
		             np.zeros((len(pts_a), 1), np.uint8), 0, False)
		if len(pts_a) < 8:
			return io.NodeOutput(*not_found)
		try:
			F, mask = cv2.findFundamentalMat(pts_a, pts_b, enums.FM_METHOD.resolve(method),
			                                 reproj_threshold, confidence)
		except cv2.error:
			return io.NodeOutput(*not_found)
		if F is None or not np.all(np.isfinite(F)):
			return io.NodeOutput(*not_found)
		F = np.asarray(F, np.float64)[:3]  # FM_7POINT can stack up to 3 solutions
		if mask is None:  # non-robust methods may not produce a mask
			mask = np.ones((len(pts_a), 1), np.uint8)
		mask = np.asarray(mask, np.uint8).reshape(-1, 1)
		inliers = int(mask.sum())
		if inliers < min_inliers:
			return io.NodeOutput(np.zeros((3, 3), np.float64), mask, inliers, False)
		return io.NodeOutput(F, mask, inliers, True)


def _index_color(i: int, n_channels: int = 3):
	"""Deterministic distinct color per index, matching n_channels.

	The colours come from the Okabe-Ito colour-vision-deficiency-safe set
	(imageutils.categorical_bgr), so two classes stay apart for a dichromat
	reader instead of only for a trichromat one. A 1-channel canvas cannot
	carry hue at all, so it gets a spread of GREY LEVELS (categorical_gray)
	rather than the average of the colour - averaging a CVD-safe palette
	collapses every class to nearly the same grey.

	The index -> colour mapping is stable per channel count, so a point and its
	epipolar line still pair up when the two canvases differ.
	"""
	if n_channels == 1:
		return imageutils.categorical_gray(i)
	b, g, r = imageutils.categorical_bgr(i)
	if n_channels == 2:
		return (b, g)
	if n_channels == 4:
		return (b, g, r, 255)
	return (b, g, r)


class DrawEpilines(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawEpilines",
			display_name="CV Draw Epipolar Lines",
			category=CATEGORY,
			description="Debug helper: draws epipolar lines (ax + by + c = 0, "
			            "from cv2.computeCorrespondEpilines) across the image, "
			            "optionally marking this image's matched points in the "
			            "same per-index color - use two of these nodes with the "
			            "same inlier mask and a point in one view shares its "
			            "color with its epipolar line in the other. Degenerate "
			            "all-zero lines (e.g. from a not-found fundamental "
			            "matrix) are skipped instead of erroring.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw the epilines on (a copy is made)."),
				NPArray.Input("lines",
					tooltip="Nx1x3 line coefficients from cv2.computeCorrespondEpilines."),
				io.Int.Input("max_draw", default=40, min=0, max=10000,
					tooltip="Draw at most this many lines (after inlier "
					        "filtering). 0 = all - usually too cluttered."),
				io.Int.Input("thickness", default=1, min=1, max=16,
					tooltip="Line width in pixels (point dots scale with it)."),
				NPArray.Input("points", optional=True,
					tooltip="This image's matched points (Nx1x2): each is marked "
					        "with a dot in its line color."),
				NPArray.Input("inlier_mask", optional=True,
					tooltip="Nx1 uint8 mask from the fundamental-matrix node: "
					        "only entries flagged 1 are drawn."),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "image",
				tooltip="Same format as the image input.")],
		)

	@classmethod
	def execute(cls, source, lines, max_draw, thickness, points=None,
	            inlier_mask=None) -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		# cv2.computeCorrespondEpilines returns None for 0 points - a valid
		# "nothing matched" result that must flow through, not an error
		if lines is None or np.asarray(lines).size == 0:
			lines = np.zeros((0, 3), np.float64)
		lines = np.asarray(lines, np.float64).reshape(-1, 3)
		pts = None if points is None else np.asarray(points, np.float64).reshape(-1, 2)
		if pts is not None and len(pts) != len(lines):
			raise ValueError(
				f"got {len(lines)} lines but {len(pts)} points - both must come "
				"from the same matched point set."
			)
		if inlier_mask is not None:
			keep = np.asarray(inlier_mask).ravel().astype(bool)
			if len(keep) != len(lines):
				raise ValueError(
					f"inlier_mask has {len(keep)} entries but there are "
					f"{len(lines)} lines - it must come from the same points."
				)
			lines = lines[keep]
			pts = pts[keep] if pts is not None else None
		if max_draw and len(lines) > max_draw:
			lines = lines[:max_draw]
			pts = pts[:max_draw] if pts is not None else None

		canvas, n_channels = _make_draw_canvas(source, tag)
		h, w = canvas.shape[:2]

		def at(v):  # clamp before int conversion: lines can pass far outside
			return int(np.clip(round(v), -1_000_000, 1_000_000))

		for i, (a, b, c) in enumerate(lines):
			color = _index_color(i, n_channels)
			if max(abs(a), abs(b)) > 1e-12:  # skip degenerate (zero-F) lines
				if abs(b) >= abs(a):  # mostly horizontal: clip at left/right edge
					p0 = (0, at(-c / b))
					p1 = (w - 1, at(-(c + a * (w - 1)) / b))
				else:  # mostly vertical: clip at top/bottom edge
					p0 = (at(-c / a), 0)
					p1 = (at(-(c + b * (h - 1)) / a), h - 1)
				cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
			if pts is not None:
				cv2.circle(canvas, (at(pts[i][0]), at(pts[i][1])),
				           thickness + 3, color, -1, cv2.LINE_AA)
		return io.NodeOutput(polytype.restore(canvas, tag))


class DrawKeypoints(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawKeypoints",
			display_name="CV Draw Keypoints",
			category=CATEGORY,
			description="Debug helper: draws detected keypoints on the image "
			            "(cv2.drawKeypoints).",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw the keypoints on (a copy is made)."),
				KeyPoints.Input("keypoints",
					tooltip="Keypoints to draw (from 'CV Detect Features')."),
				io.Combo.Input("style", options=["plain", "rich (size + orientation)"],
					default="plain",
					tooltip="plain: small dots. rich: circles sized by scale with an "
					        "orientation tick."),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "image",
				tooltip="Same format as the image input.")],
		)

	@classmethod
	def execute(cls, source, keypoints, style) -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		flags = (cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
		         if style.startswith("rich") else cv2.DRAW_MATCHES_FLAGS_DEFAULT)
		return io.NodeOutput(polytype.restore(
			cv2.drawKeypoints(_to_bgr_u8(source), keypoints, None,
			                  color=(0, 255, 0), flags=flags), tag))


class DrawMatches(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGE_ONLY)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawMatches",
			display_name="CV Draw Matches",
			category=CATEGORY,
			description="Debug helper: draws the matched keypoint pairs side by "
			            "side (cv2.drawMatches). Feed the inlier mask from "
			            "cv2.findHomography / cv2.findFundamentalMat to show "
			            "only RANSAC inliers.",
			inputs=[
				polytype.match_input("image_a", cls._TEMPLATE,
					tooltip="Left image of the pair."),
				KeyPoints.Input("keypoints_a", tooltip="Keypoints of image A."),
				polytype.match_input("image_b", cls._TEMPLATE,
					tooltip="Right image of the pair (same format as image_a)."),
				KeyPoints.Input("keypoints_b", tooltip="Keypoints of image B."),
				Matches.Input("matches",
					tooltip="Matches from 'CV Match Features' to draw."),
				NPArray.Input("inlier_mask", optional=True,
					tooltip="Nx1 uint8 mask from findHomography & co: only "
					        "matches flagged 1 are drawn."),
				io.Int.Input("max_draw", default=0, min=0, max=10000,
					tooltip="Draw at most this many matches (best first). 0 = all."),
				io.String.Input("color", default="(0, 255, 0)", optional=True,
					tooltip="Color of the match lines and their keypoints, as a "
					        "single value or a BGR tuple. The green default "
					        "vanishes over foliage, grass or any green subject - "
					        "set something the pair of photos does not contain."),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "image",
				tooltip="Side-by-side canvas, same format as the image inputs.")],
		)

	@classmethod
	def execute(cls, image_a, keypoints_a, image_b, keypoints_b, matches,
	            inlier_mask=None, max_draw=0, color="(0, 255, 0)") -> io.NodeOutput:
		image_a, tag = polytype.resolve(image_a)
		image_b, _ = polytype.resolve(image_b)
		draw_mask = None
		if inlier_mask is not None:
			draw_mask = np.asarray(inlier_mask).ravel().astype(int).tolist()
			if len(draw_mask) != len(matches):
				raise ValueError(
					f"inlier_mask has {len(draw_mask)} entries but there are "
					f"{len(matches)} matches - it must come from the same "
					"points the matcher produced."
				)
		if max_draw and max_draw < len(matches):
			matches = matches[:max_draw]
			draw_mask = draw_mask[:max_draw] if draw_mask else None
		canvas = cv2.drawMatches(
			_to_bgr_u8(image_a), keypoints_a, _to_bgr_u8(image_b), keypoints_b,
			matches, None, matchColor=parse_color(color, 3),
			singlePointColor=(80, 80, 80),
			matchesMask=draw_mask, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
		return io.NodeOutput(polytype.restore(canvas, tag))


class CVSize(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CVSize",
			display_name="CV Array Size",
			category=CATEGORY,
			description="Reads width/height of an ndarray image. The 'size' "
			            "output is the whole (width, height) as ONE composite "
			            "value - wire it straight into a low-level cv2 node's "
			            "dsize / ksize / imageSize input; 'dsize' is the same "
			            "pair as a '(w, h)' literal string, for the composite "
			            "params that are still free-text. A LATENT input "
			            "reports its size in latent "
			            "CELLS (pixels / 8) - the units every cv2 node sees when "
			            "operating on a latent.",
			inputs=[polytype.poly_input("nparray",
				types=polytype.with_latent(polytype.IMAGEISH),
				tooltip="Image whose width/height to read (only the shape is "
				        "used). LATENT sizes count latent cells, not pixels.")],
			outputs=[
				io.Int.Output(display_name="width"),
				io.Int.Output(display_name="height"),
				io.String.Output(display_name="dsize",
					tooltip="'(width, height)' literal - for the composite cv2 "
					        "params that are still free-text."),
				tuples.CvTuple.Output(display_name="size",
					tooltip="(width, height) as ONE composite value - wire it "
					        "straight into a cv2 node's dsize / ksize / "
					        "imageSize input, in one link instead of two."),
			],
		)

	@classmethod
	def execute(cls, nparray) -> io.NodeOutput:
		nparray, _ = polytype.resolve(nparray)
		arr = np.asarray(nparray)
		if arr.ndim < 2:
			raise ValueError(f"Array of shape {arr.shape} has no image size.")
		h, w = arr.shape[:2]
		return io.NodeOutput(w, h, f"({w}, {h})", [int(w), int(h)])


class DetectCircleGrid(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DetectCircleGrid",
			display_name="CV Detect Circle Grid",
			category=CATEGORY,
			description="Detects a symmetric or asymmetric circle calibration "
			            "target (cv2.findCirclesGrid). Pattern size is authored "
			            "as a CV Size literal, so the same 'CV Size' node can feed "
			            "this and other calibration detectors. Failure-tolerant: "
			            "when the grid is not visible it returns found=false and "
			            "an empty Nx1x2 centers array instead of raising.",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Circle-grid photo or rendered target. It is converted "
					        "to grayscale internally."),
				io.String.Input("pattern_size", default="(4, 11)",
					tooltip="Number of circle centers per row/column as a Size "
					        "literal '(cols, rows)'. Use a 'CV Size' node to avoid "
					        "typing the tuple.",
					extra_dict={"multiline": False}),
				io.String.Input("flags",
					default=enums.CALIB_CIRCLES_GRID.initial,
					tooltip="Grid layout plus optional clustering. Symmetric and "
					        "asymmetric are mutually exclusive; clustering helps "
					        "with perspective distortion and clutter.",
					extra_dict={"cv_flags": enums.CALIB_CIRCLES_GRID.spec()}),
			],
			outputs=[
				io.Boolean.Output(display_name="found"),
				NPArray.Output(display_name="centers",
					tooltip="Detected circle centers as Nx1x2 float32, empty when "
					        "found=false."),
			],
		)

	@classmethod
	def execute(cls, image, pattern_size, flags) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		pattern = _parse_size_literal(pattern_size, "pattern_size")
		gray = _to_gray_u8(image)
		try:
			found, centers = cv2.findCirclesGrid(
				gray, pattern, flags=enums.CALIB_CIRCLES_GRID.resolve(flags))
		except cv2.error:
			found, centers = False, None
		if not found or centers is None:
			return io.NodeOutput(False, np.zeros((0, 1, 2), np.float32))
		return io.NodeOutput(True, np.asarray(centers, np.float32).reshape(-1, 1, 2))


def _circle_grid_object_points(pattern, grid_layout, square_size):
	"""Builds the planar object points for a circle calibration target.

	ASYMMETRIC layouts use the staggered convention (2*col + row%2, row):
	the offset between adjacent rows is what breaks the 180-degree rotational
	ambiguity that makes symmetric grids unusable for stereo correspondence.
	The returned order must match findCirclesGrid's center ordering.
	"""
	cols, rows = pattern
	objp = np.zeros((cols * rows, 3), np.float32)
	if grid_layout == "asymmetric (staggered)":
		k = 0
		for r in range(rows):
			for c in range(cols):
				objp[k] = (2 * c + (r % 2), r, 0)
				k += 1
	else:
		k = 0
		for r in range(rows):
			for c in range(cols):
				objp[k] = (c, r, 0)
				k += 1
	return objp * float(square_size)


def _find_circle_grid_centers(frame, pattern, grid_layout):
	"""findCirclesGrid on one BGR frame; returns (ok, centers or None)."""
	gray = _to_gray_u8(frame)
	flag = (cv2.CALIB_CB_ASYMMETRIC_GRID if grid_layout == "asymmetric (staggered)"
	        else cv2.CALIB_CB_SYMMETRIC_GRID)
	try:
		ok, centers = cv2.findCirclesGrid(gray, pattern, flags=flag | cv2.CALIB_CB_CLUSTERING)
	except cv2.error:
		return False, None
	if not ok or centers is None:
		return False, None
	return True, centers


class CalibrateCircleGrid(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CalibrateCircleGrid",
			display_name="CV Calibrate Camera (Circle Grid)",
			category=CATEGORY,
			description="Estimates a camera's intrinsics and lens distortion from "
			            "several circle-grid views (cv2.calibrateCamera). Calibration "
			            "needs per-view 2D/3D correspondences accumulated across "
			            "images, which the raw wrappers can't express, so this node "
			            "manages it exactly like the chessboard 'Calibrate Camera': "
			            "feed an IMAGE BATCH of circle-grid photos from DIFFERENT "
			            "angles; each is grayscaled, its circle centers are found "
			            "with cv2.findCirclesGrid and matched to planar object "
			            "points built from pattern size x square_size. Views where "
			            "the grid is not found are skipped. The 'asymmetric "
			            "(staggered)' layout is the usual choice - it is the "
			            "OpenCV-recommended calibration target for wide-angle and "
			            "stereo rigs, since the staggered rows break the 180-degree "
			            "ambiguity that plagues symmetric grids. Failure-tolerant: "
			            "with fewer than 3 usable views it returns found=false, an "
			            "identity matrix and zero distortion. Feed camera_matrix + "
			            "dist_coeffs into cv2.undistort, or into 'Stereo Calibrate "
			            "(Circle Grid)' as initial guesses; rms_error (px) reports "
			            "fit quality (under ~1 is good).",
			inputs=[
				io.Image.Input("images",
					tooltip="Batch of circle-grid views from different angles "
					        "(>= 3 usable; ~10-20 gives a good fit)."),
				io.Int.Input("pattern_cols", default=4, min=2, max=40,
					tooltip="Circle centers per row (width of the grid)."),
				io.Int.Input("pattern_rows", default=11, min=2, max=40,
					tooltip="Circle centers per column (height of the grid)."),
				io.Combo.Input("grid_layout",
					options=["asymmetric (staggered)", "symmetric"],
					default="asymmetric (staggered)",
					tooltip="Grid geometry. Asymmetric staggers each row by half a "
					        "spacing, which removes the 180-degree rotational "
					        "ambiguity of symmetric targets - the reason stereo and "
					        "wide-angle calibration prefer it."),
				io.Float.Input("square_size", default=1.0, min=1e-4, max=1e6,
					tooltip="Physical center-to-center spacing (e.g. mm); sets the "
					        "world scale. Leave 1.0 for relative calibration.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="camera_matrix",
					tooltip="3x3 intrinsic matrix K (fx, fy, cx, cy)."),
				NPArray.Output(display_name="dist_coeffs",
					tooltip="Distortion coefficients (k1, k2, p1, p2, k3)."),
				io.Float.Output(display_name="rms_error",
					tooltip="Mean reprojection error in pixels; lower is better (<1 good)."),
				io.Int.Output(display_name="views_used",
					tooltip="Number of input views with a detectable grid."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, images, pattern_cols, pattern_rows, grid_layout,
	            square_size=1.0) -> io.NodeOutput:
		pattern = (int(pattern_cols), int(pattern_rows))
		objp = _circle_grid_object_points(pattern, grid_layout, square_size)
		obj_points, img_points, size = [], [], None
		for frame in imageutils.image_batch_to_bgr(images):
			gray = _to_gray_u8(frame)
			size = (gray.shape[1], gray.shape[0])  # (w, h)
			ok, centers = _find_circle_grid_centers(frame, pattern, grid_layout)
			if not ok:
				continue
			obj_points.append(objp.copy())
			img_points.append(np.asarray(centers, np.float32).reshape(-1, 1, 2))
		if len(img_points) < 3 or size is None:
			return io.NodeOutput(np.eye(3, dtype=np.float64),
			                     np.zeros((1, 5), np.float64), 0.0, len(img_points), False)
		rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
		return io.NodeOutput(np.asarray(K, np.float64), np.asarray(dist, np.float64),
		                     float(rms), len(img_points), True)


class ImageCorners(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ImageCorners",
			display_name="CV Image Corners",
			category=CATEGORY,
			description="Outputs the 4 corner coordinates of the image as a "
			            "4x1x2 float32 array (top-left, top-right, bottom-right, "
			            "bottom-left). Send through cv2.perspectiveTransform "
			            "with a homography to project an object's outline into "
			            "another view.",
			inputs=[polytype.poly_input("nparray",
				tooltip="Image whose corners to emit (only the shape is used).")],
			outputs=[NPArray.Output(display_name="corners")],
		)

	@classmethod
	def execute(cls, nparray) -> io.NodeOutput:
		nparray, _ = polytype.resolve(nparray)
		arr = np.asarray(nparray)
		if arr.ndim < 2:
			raise ValueError(f"Array of shape {arr.shape} has no image size.")
		h, w = arr.shape[:2]
		corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
		return io.NodeOutput(corners)


class DrawPolygon(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawPolygon",
			display_name="CV Draw Polygon",
			category=CATEGORY,
			description="Draws a polygon/polyline through the given points "
			            "(cv2.polylines or cv2.fillPoly) - e.g. a projected "
			            "object outline or rotated text detection quads. "
			            "When several outlines share one image, vary "
			            "'line_style' as well as the color so they stay "
			            "tellable apart without relying on hue. "
			            "Accepts IMAGE or MASK input. When thickness is 0, "
			            "polygons are filled (useful for creating masks from "
			            "rotated quads). Multiple polygons: pass an (N, M, 2) "
			            "array to draw N separate M-point polygons (e.g. the "
			            "quads output from 'CV EAST Text Detect'). "
			            "To draw conditionally (e.g. only when 'CV Find "
			            "Homography' reports found=true), select between this "
			            "node's output and the original image with a control-"
			            "flow node such as Basic data handling's 'if/else' - "
			            "its lazy inputs skip the drawing branch entirely when "
			            "it is not taken.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made)."),
				NPArray.Input("points",
					tooltip="(N,2) for a single polyline/polygon, or "
					        "(N,M,2) for N separate M-point polygons "
					        "(e.g. rotated quads from EAST text detection)."),
				io.String.Input("color", default="(0, 0, 255)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(0, 0, 255)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated."),
				io.Int.Input("thickness", default=3, min=0, max=64,
					tooltip="Outline width in pixels. 0 = filled polygons "
					        "(cv2.fillPoly) - useful for creating masks."),
				io.Boolean.Input("closed", default=True,
					tooltip="Connect the last point back to the first (closed "
					        "polygon) vs leave it open (polyline). Ignored when "
					        "thickness=0 (always filled)."),
				io.Combo.Input("line_style", options=list(LINE_STYLES),
					default="solid", optional=True,
					tooltip=LINE_STYLE_TOOLTIP + " Ignored when the polygons "
					        "are filled (thickness=0, or a MASK canvas)."),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "image",
				tooltip="Same format as the image input.")],
		)

	@classmethod
	def execute(cls, source, points, color, thickness, closed,
	            line_style="solid") -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		arr = np.asarray(points)
		if len(arr) < 2:
			raise ValueError(f"Need at least 2 points to draw, got array of shape "
			                 f"{arr.shape}.")
		canvas, n_channels = _make_draw_canvas(source, tag)
		bgr = parse_color(color, n_channels)

		# 3-D (N, M, 2) with M >= 2 and N >= 2: N separate M-point polygons
		# (e.g. rotated quads from EAST text detection).
		# Everything else: flatten to (K, 2) and draw a single polyline.
		if arr.ndim == 3 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
			polys = [p.reshape(-1, 2).round().astype(np.int32) for p in arr
			         if len(p) >= 2]
		else:
			flat = arr.reshape(-1, 2)
			polys = [flat.round().astype(np.int32)] if len(flat) >= 2 else []

		if polys:
			if thickness == 0 or tag == "mask":
				fill_val = bgr[0] if canvas.ndim == 2 else bgr
				for poly in polys:
					cv2.fillPoly(canvas, [poly], fill_val)
			elif line_style == "solid":
				cv2.polylines(canvas, polys, closed, bgr, thickness, cv2.LINE_AA)
			else:
				for poly in polys:
					draw_styled_polyline(canvas, poly, bgr, thickness,
					                     line_style, bool(closed))

		return io.NodeOutput(polytype.restore(canvas, tag))


class DrawText(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawText",
			display_name="CV Draw Text",
			category=CATEGORY,
			description="Stamps a text label onto the image or mask (cv2.putText), "
			            "with an optional contrasting halo so it stays "
			            "readable on any background - print a pipeline's final "
			            "result (a reading, a count, a score) into its debug "
			            "image. Compose dynamic text with Basic data handling "
			            "string nodes ('to STRING', 'zfill', 'concat'). Empty "
			            "text passes the image through unchanged.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to stamp the text onto (a copy is made)."),
				io.String.Input("text", default="",
					tooltip="The label; convert to an input to feed a computed "
					        "string."),
				io.Float.Input("x", default=12.0, min=-1e6, max=1e6,
					tooltip="Left edge of the text baseline, in pixels."),
				io.Float.Input("y", default=42.0, min=-1e6, max=1e6,
					tooltip="Baseline height, in pixels (text sits above it)."),
				io.Float.Input("font_scale", default=1.2, min=0.1, max=64.0,
					tooltip="Font size multiplier (1.0 is the base Hershey size)."),
				io.String.Input("color", default="(0, 255, 0)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(0, 255, 0)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated."),
				io.Int.Input("thickness", default=2, min=1, max=64,
					tooltip="Stroke width of the glyphs in pixels."),
				io.Boolean.Input("outline", default=True,
					tooltip="Draw a contrasting halo behind the text: black "
					        "behind a light 'color', white behind a dark one. "
					        "It follows the colour, so black text on a black "
					        "background is lifted off it rather than "
					        "disappearing into a black halo."),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "image",
				tooltip="Same format as the image input.")],
		)

	@classmethod
	def execute(cls, source, text, x, y, font_scale, color, thickness, outline) -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		if not text:
			return io.NodeOutput(polytype.restore(canvas, tag))
		bgr = parse_color(color, n_channels)
		org = (int(round(x)), int(round(y)))
		if outline:
			draw_text_outlined(canvas, text, org, font_scale, halo_color(bgr),
			                   bgr, thickness)
		else:
			cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
			            bgr, thickness, cv2.LINE_AA)
		return io.NodeOutput(polytype.restore(canvas, tag))


class FlowToColor(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FlowToColor",
			display_name="CV Flow To Color",
			category=CATEGORY,
			description="Debug helper: renders a per-pixel ANGLE field as the "
			            "standard color wheel (angle = hue, magnitude = "
			            "brightness) - the proper way to look at gradient "
			            "directions, optical flow or any other vector field; "
			            "a plain grayscale render of angles lies at the "
			            "0/360 wrap-around. Feed it cv2_cartToPolar's two "
			            "outputs. Without a magnitude, every pixel renders "
			            "at full brightness.",
			inputs=[
				NPArray.Input("angle",
					tooltip="HxW per-pixel angles (cv2_cartToPolar output)."),
				NPArray.Input("magnitude", optional=True,
					tooltip="HxW per-pixel magnitudes; scales brightness so "
					        "weak vectors fade out. Must match the angle "
					        "shape."),
				io.Combo.Input("angle_unit",
					options=["radians (0-2*pi, cartToPolar default)",
					         "degrees (0-360)"],
					default="radians (0-2*pi, cartToPolar default)",
					tooltip="Set to degrees when cartToPolar ran with "
					        "angleInDegrees=true."),
				io.Float.Input("max_magnitude", default=0.0, min=0.0, max=1e9,
					tooltip="Magnitude mapped to full brightness; larger "
					        "values clip. 0 = auto (the field's own "
					        "maximum) - set it explicitly to compare "
					        "renders across images on the same scale."),
			],
			outputs=[NPArray.Output(display_name="nparray",
				tooltip="BGR uint8 rendering: HUE is the angle (0 deg / east "
				        "at hue 0, then round the wheel) and BRIGHTNESS is the "
				        "magnitude. Direction is carried by hue alone, which is "
				        "the axis a colour-blind reader loses - read this as a "
				        "picture of where the motion changes, and take the "
				        "angle itself off the numbers ('Inspect CV Data') "
				        "rather than off the colour.")],
		)

	@classmethod
	def execute(cls, angle, magnitude=None, angle_unit="radians (0-2*pi, cartToPolar default)",
	            max_magnitude=0.0) -> io.NodeOutput:
		ang = np.asarray(angle, np.float32)
		if ang.ndim == 3 and ang.shape[2] == 1:
			ang = ang[..., 0]
		if ang.ndim != 2 or ang.size == 0:
			raise ValueError(
				f"angle must be a HxW array, got shape {np.asarray(angle).shape}."
			)
		deg = ang if angle_unit.startswith("degrees") else np.degrees(ang)
		hue = ((deg % 360.0) / 2.0).astype(np.uint8)  # OpenCV hue range is 0-179
		if magnitude is None:
			value = np.full_like(hue, 255)
		else:
			mag = np.asarray(magnitude, np.float32)
			if mag.ndim == 3 and mag.shape[2] == 1:
				mag = mag[..., 0]
			if mag.shape != deg.shape:
				raise ValueError(
					f"magnitude shape {mag.shape} does not match angle shape "
					f"{deg.shape} - both must come from the same field."
				)
			scale = max_magnitude if max_magnitude > 0 else float(mag.max())
			if scale <= 0:
				scale = 1.0
			value = np.clip(mag / scale * 255.0, 0, 255).astype(np.uint8)
		hsv = np.dstack([hue, np.full_like(hue, 255), value])
		return io.NodeOutput(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))


class OpticalFlowFarneback(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_OpticalFlowFarneback",
			display_name="CV Optical Flow (Farneback)",
			category=CATEGORY,
			description="Dense optical flow between two frames "
			            "(cv2.calcOpticalFlowFarneback): the per-pixel motion "
			            "from frame_a to frame_b. The raw cv2 wrapper is "
			            "unusable here - its 'flow' in/out array is a required "
			            "input and the numeric parameters have no sensible "
			            "defaults - so this node passes flow=None and presets "
			            "the Farneback parameters. Both frames are converted to "
			            "grayscale and frame_b is resized to frame_a if their "
			            "sizes differ. Outputs the raw HxWx2 flow plus ready-to-"
			            "visualise magnitude and angle (RADIANS): feed angle + "
			            "magnitude straight into 'CV Flow To Color' (its "
			            "default radians mode), or the flow field into two "
			            "cv2.extractChannel nodes for dx/dy.",
			inputs=[
				polytype.poly_input("frame_a", types=polytype.IMAGE_ONLY,
					tooltip="First frame (earlier in time); any colour space, "
					        "converted to grayscale."),
				polytype.poly_input("frame_b", types=polytype.IMAGE_ONLY,
					tooltip="Second frame; resized to frame_a if the sizes differ."),
				io.Float.Input("pyr_scale", default=0.5, min=0.1, max=0.9, step=0.05,
					tooltip="Pyramid scale between levels (<1); 0.5 halves the "
					        "resolution at each level."),
				io.Int.Input("levels", default=3, min=1, max=10,
					tooltip="Number of pyramid levels (1 = no pyramid)."),
				io.Int.Input("winsize", default=15, min=3, max=101,
					tooltip="Averaging window size; larger = more robust to noise "
					        "but blurrier flow."),
				io.Int.Input("iterations", default=3, min=1, max=20,
					tooltip="Iterations at each pyramid level."),
				io.Int.Input("poly_n", default=5, min=3, max=11,
					tooltip="Neighborhood size for the polynomial expansion "
					        "(5 or 7 are typical)."),
				io.Float.Input("poly_sigma", default=1.2, min=0.3, max=3.0, step=0.1,
					tooltip="Gaussian sigma of the polynomial expansion (~1.1 for "
					        "poly_n=5, ~1.5 for poly_n=7)."),
				io.Combo.Input("window", options=["box (faster)", "gaussian (more accurate)"],
					default="box (faster)",
					tooltip="Averaging window type; gaussian is slower but yields "
					        "smoother flow."),
			],
			outputs=[
				NPArray.Output(display_name="flow",
					tooltip="HxWx2 float32 (dx, dy) displacement per pixel."),
				NPArray.Output(display_name="magnitude",
					tooltip="HxW float32 motion magnitude in pixels."),
				NPArray.Output(display_name="angle",
					tooltip="HxW float32 motion direction in RADIANS - feed "
					        "'CV Flow To Color' in its default radians mode."),
			],
		)

	@classmethod
	def execute(cls, frame_a, frame_b, pyr_scale, levels, winsize, iterations,
	            poly_n, poly_sigma, window) -> io.NodeOutput:
		frame_a, _ = polytype.resolve(frame_a)
		frame_b, _ = polytype.resolve(frame_b)
		prev = _to_gray_u8(frame_a)
		nxt = _to_gray_u8(frame_b)
		if nxt.shape != prev.shape:
			nxt = cv2.resize(nxt, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
		flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if window.startswith("gaussian") else 0
		flow = cv2.calcOpticalFlowFarneback(
			prev, nxt, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
		return io.NodeOutput(flow.astype(np.float32), mag.astype(np.float32), ang.astype(np.float32))


class DrawFlowGrid(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawFlowGrid",
			display_name="CV Draw Flow Grid",
			category=CATEGORY,
			description="Debug helper: renders a DENSE flow field (HxWx2, e.g. the "
			            "'flow' output of 'CV Optical Flow (Farneback)') as the "
			            "classic quiver plot - one arrow per regularly-spaced grid "
			            "cell. Each arrow is colored by its motion MAGNITUDE relative "
			            "to the field's maximum (auto, or a fixed max_magnitude to "
			            "compare frames on one scale), through a named colormap. "
			            "Complements 'CV Flow To Color' (per-pixel direction "
			            "wheel) and 'CV Draw Flow Vectors' (SPARSE point pairs). "
			            "An empty/zero field draws nothing and returns count=0.",
			inputs=[
				NPArray.Input("flow",
					tooltip="HxWx2 float (dx, dy) per pixel - the dense flow field "
					        "from 'CV Optical Flow (Farneback)'."),
				io.Int.Input("step", default=16, min=2, max=256,
					tooltip="Grid spacing in pixels: one arrow is sampled every "
					        "'step' pixels. Smaller = denser, busier."),
				io.Float.Input("scale", default=1.0, min=0.1, max=50.0, step=0.1,
					tooltip="Arrow-length multiplier for visibility only (the COLOR "
					        "still encodes the true, unscaled magnitude). Raise it "
					        "when sub-pixel flow makes the arrows too short to see."),
				io.Int.Input("thickness", default=1, min=1, max=32,
					tooltip="Arrow line width in pixels."),
				io.Combo.Input("colormap", options=enums.COLORMAP.options,
					default="COLORMAP_VIRIDIS",
					tooltip="Maps magnitude to color. The slowest grid arrow "
					        "takes the cold end, the fastest the hot end. The "
					        "default is perceptually uniform and rises "
					        "monotonically in brightness, so slow-vs-fast reads "
					        "correctly for a colour-blind viewer and in "
					        "greyscale; COLORMAP_JET is the familiar "
					        "blue->red but does neither."),
				polytype.poly_input("source", types=polytype.IMAGEISH, optional=True,
					tooltip="Optional canvas (IMAGE or MASK) to draw the arrows over "
					        "(resized to the flow if the sizes differ); a black "
					        "canvas is used if omitted."),
				io.Float.Input("tip_length", default=0.3, min=0.0, max=2.0,
					optional=True, advanced=True,
					tooltip="Arrowhead length as a fraction of the shaft "
					        "(cv2.arrowedLine tipLength)."),
				io.Float.Input("min_magnitude", default=0.0, min=0.0, max=1e6,
					optional=True, advanced=True,
					tooltip="Skip arrows shorter than this many pixels - hides the "
					        "static background. 0 draws every grid cell."),
				io.Float.Input("max_magnitude", default=0.0, min=0.0, max=1e6,
					optional=True, advanced=True,
					tooltip="Magnitude mapped to the hot end of the colormap; larger "
					        "values clip. 0 = auto (this field's fastest grid "
					        "arrow) - set it to compare frames on one scale."),
			],
			outputs=[
				NPArray.Output(display_name="nparray",
					tooltip="BGR uint8 quiver rendering."),
				io.Int.Output(display_name="count",
					tooltip="How many arrows were drawn (after the min_magnitude "
					        "filter) - branch on it with if/else for nothing-moved."),
			],
		)

	@classmethod
	def execute(cls, flow, step, scale, thickness, colormap,
	            source=None, tip_length=0.3, min_magnitude=0.0,
	            max_magnitude=0.0) -> io.NodeOutput:
		source, _ = polytype.resolve(source)
		flow = np.asarray(flow, np.float32)
		if flow.ndim != 3 or flow.shape[2] != 2:
			raise ValueError(
				f"flow must be a HxWx2 (dx, dy) field, got shape {flow.shape} - "
				"feed the 'flow' output of 'CV Optical Flow (Farneback)'."
			)
		h, w = flow.shape[:2]
		if source is not None:
			canvas = _to_bgr_u8(source).copy()
			if canvas.shape[:2] != (h, w):
				canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
		else:
			canvas = np.zeros((h, w, 3), np.uint8)
		if h == 0 or w == 0:
			return io.NodeOutput(canvas, 0)

		# JET (or chosen) magnitude ramp as a BGR lookup table.
		lut = cv2.applyColorMap(
			np.arange(256, dtype=np.uint8).reshape(1, 256),
			enums.COLORMAP.resolve(colormap))[0]

		step = max(2, int(step))
		ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1)
		fx, fy = flow[ys, xs, 0], flow[ys, xs, 1]
		mag = np.hypot(fx, fy)
		keep = mag >= float(min_magnitude)
		scale_ref = max_magnitude if max_magnitude > 0 else float(mag[keep].max() if keep.any() else 0.0)
		if scale_ref <= 0:
			scale_ref = 1.0
		colors = lut[np.clip(mag / scale_ref * 255.0, 0, 255).astype(np.int32)]

		count = 0
		for i in range(xs.size):
			if not keep[i]:
				continue
			x0, y0 = int(xs[i]), int(ys[i])
			x1 = int(round(x0 + fx[i] * scale))
			y1 = int(round(y0 + fy[i] * scale))
			cv2.arrowedLine(canvas, (x0, y0), (x1, y1),
				tuple(int(v) for v in colors[i]), int(thickness),
				cv2.LINE_AA, 0, float(tip_length))
			count += 1
		return io.NodeOutput(canvas, count)


class StereoDisparity(io.ComfyNode):
	_MODES = {
		"SGBM (balanced)": cv2.STEREO_SGBM_MODE_SGBM,
		"SGBM 3-way (faster)": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
		"HH (full-scale, slow)": cv2.STEREO_SGBM_MODE_HH,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_StereoDisparity",
			display_name="CV Stereo Disparity (SGBM)",
			category=CATEGORY,
			description="Per-pixel disparity from a RECTIFIED stereo pair "
			            "(cv2.StereoSGBM): how far each point shifts between the "
			            "left and right view - inversely proportional to depth. "
			            "StereoSGBM is a cv2 class the raw wrappers cannot expose, "
			            "so this node manages it: both frames are grayscaled, the "
			            "P1/P2 smoothness terms are derived from block_size "
			            "(OpenCV's documented heuristic) and the int16 result is "
			            "converted to float disparity in pixels. The pair MUST be "
			            "rectified (epipolar lines horizontal) - run "
			            "cv2.stereoRectify + initUndistortRectifyMap + cv2.remap "
			            "first. Preview the disparity with 'Preview CV Array' "
			            "(normalize / heatmap); larger disparity = closer.",
			inputs=[
				polytype.poly_input("left", types=polytype.IMAGE_ONLY,
					tooltip="Left rectified frame (grayscaled internally)."),
				polytype.poly_input("right", types=polytype.IMAGE_ONLY,
					tooltip="Right rectified frame; resized to left if sizes differ."),
				io.Int.Input("num_disparities", default=64, min=16, max=512, step=16,
					tooltip="Disparity search range (0..num_disparities), rounded up "
					        "to a multiple of 16. Larger covers nearer objects / "
					        "wider baselines but is slower."),
				io.Int.Input("block_size", default=7, min=1, max=21,
					tooltip="Matched block size in pixels (forced odd; 3-11 is "
					        "typical). Smaller = more detail but noisier."),
				io.Int.Input("min_disparity", default=0, min=-256, max=256,
					tooltip="Smallest disparity to search from (usually 0)."),
				io.Combo.Input("mode", options=list(cls._MODES), default="SGBM (balanced)",
					tooltip="SGBM is the standard; 3-way is faster at lower quality; "
					        "HH runs the full-scale two-pass (best, slowest)."),
				io.Int.Input("uniqueness_ratio", default=10, min=0, max=100,
					tooltip="Margin (%) by which the best match must beat the "
					        "runner-up to be accepted.", optional=True, advanced=True),
				io.Int.Input("speckle_window_size", default=100, min=0, max=1000,
					tooltip="Largest smooth disparity blob treated as speckle noise "
					        "and invalidated (0 = off).", optional=True, advanced=True),
				io.Int.Input("speckle_range", default=2, min=0, max=64,
					tooltip="Max disparity variation within a speckle component.",
					optional=True, advanced=True),
				io.Int.Input("disp12_max_diff", default=-1, min=-1, max=64,
					tooltip="Left-right consistency check, in pixels: a match whose "
					        "reverse match disagrees by more than this is invalidated. "
					        "-1 (cv2's default) turns it off and leaves occlusion junk "
					        "in the map; 1 is the usual setting when the disparity is "
					        "about to be interpolated or turned into 3D points.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="disparity",
					tooltip="HxW float32 disparity in pixels; invalid pixels are "
					        "below min_disparity (see 'valid')."),
				NPArray.Output(display_name="valid",
					tooltip="uint8 0/255 mask of pixels with a valid disparity."),
			],
		)

	@classmethod
	def execute(cls, left, right, num_disparities, block_size, min_disparity, mode,
	            uniqueness_ratio=10, speckle_window_size=100, speckle_range=2,
	            disp12_max_diff=-1) -> io.NodeOutput:
		left, _ = polytype.resolve(left)
		right, _ = polytype.resolve(right)
		lg = _to_gray_u8(left)
		rg = _to_gray_u8(right)
		if rg.shape != lg.shape:
			rg = cv2.resize(rg, (lg.shape[1], lg.shape[0]), interpolation=cv2.INTER_AREA)
		nd = max(16, int(round(num_disparities / 16)) * 16)
		bs = max(1, block_size | 1)  # SGBM needs an odd block size
		matcher = cv2.StereoSGBM_create(
			minDisparity=min_disparity, numDisparities=nd, blockSize=bs,
			P1=8 * bs * bs, P2=32 * bs * bs,
			uniquenessRatio=uniqueness_ratio,
			speckleWindowSize=speckle_window_size, speckleRange=speckle_range,
			disp12MaxDiff=int(disp12_max_diff),
			mode=cls._MODES[mode])
		# StereoSGBM returns int16 disparity scaled by 16; invalid = (min-1)*16
		disparity = matcher.compute(lg, rg).astype(np.float32) / 16.0
		valid = (disparity >= min_disparity).astype(np.uint8) * 255
		return io.NodeOutput(disparity, valid)


class StereoRectifyUncalibrated(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_StereoRectifyUncalibrated",
			display_name="CV Stereo Rectify (Uncalibrated)",
			category=CATEGORY,
			description="Computes the two rectifying homographies for an "
			            "UNCALIBRATED stereo pair (cv2.stereoRectifyUncalibrated) "
			            "from matched points and the fundamental matrix - no "
			            "intrinsics needed. Apply H1 to the left image and H2 to "
			            "the right (cv2.warpPerspective, borderMode=BORDER_CONSTANT) "
			            "and the two views become row-aligned: a scene point lands "
			            "on the SAME image row in both, so 'CV Stereo Disparity "
			            "(SGBM)' can match along horizontal scanlines. Feed the "
			            "RANSAC inliers (filter points_a/points_b with 'OpenCV "
			            "Filter Points By Mask' on the fundamental matrix's "
			            "inlier_mask) for a stable result. Failure-tolerant: with "
			            "fewer than 8 points, a degenerate configuration or a "
			            "not-found (all-zero) fundamental matrix it returns "
			            "found=false and two IDENTITY homographies (warping leaves "
			            "the images unchanged) instead of raising - branch on "
			            "'found' with a control-flow node.",
			inputs=[
				NPArray.Input("points_a",
					tooltip="Matched points in the left image, Nx1x2 (inliers from "
					        "'CV Filter Points By Mask')."),
				NPArray.Input("points_b",
					tooltip="Corresponding points in the right image, Nx1x2."),
				NPArray.Input("fundamental",
					tooltip="3x3 fundamental matrix from 'CV Find Fundamental "
					        "Matrix' (all zeros -> found=false here)."),
				polytype.poly_input("image",
					tooltip="Either rectified-pair frame; only its width/height are "
					        "read (the rectification image size)."),
				io.Float.Input("threshold", default=5.0, min=0.0, max=100.0, step=0.5,
					tooltip="Points farther than this (px) from their epipolar line "
					        "are dropped before solving. 0 keeps all points.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="H1",
					tooltip="3x3 float64 homography for the LEFT image (identity if "
					        "not found)."),
				NPArray.Output(display_name="H2",
					tooltip="3x3 float64 homography for the RIGHT image (identity if "
					        "not found)."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, points_a, points_b, fundamental, image, threshold=5.0) -> io.NodeOutput:
		eye = np.eye(3, dtype=np.float64)
		not_found = (eye.copy(), eye.copy(), False)
		if points_a is None or points_b is None or fundamental is None:
			return io.NodeOutput(*not_found)
		pts_a = np.asarray(points_a, np.float32).reshape(-1, 1, 2)
		pts_b = np.asarray(points_b, np.float32).reshape(-1, 1, 2)
		if len(pts_a) != len(pts_b):
			raise ValueError(
				f"points_a has {len(pts_a)} points but points_b has {len(pts_b)} - "
				"both must come from the same matched point set."
			)
		F = np.asarray(fundamental, np.float64)
		if len(pts_a) < 8 or F.shape != (3, 3) or not np.any(F):
			return io.NodeOutput(*not_found)
		image, _ = polytype.resolve(image)
		arr = np.asarray(image)
		if arr.ndim < 2:
			raise ValueError(f"image of shape {arr.shape} has no size to rectify to.")
		h, w = arr.shape[:2]
		try:
			ok, H1, H2 = cv2.stereoRectifyUncalibrated(
				pts_a, pts_b, F, (w, h), threshold=threshold)
		except cv2.error:
			return io.NodeOutput(*not_found)
		if not ok or H1 is None or H2 is None:
			return io.NodeOutput(*not_found)
		H1, H2 = np.asarray(H1, np.float64), np.asarray(H2, np.float64)
		if not (np.all(np.isfinite(H1)) and np.all(np.isfinite(H2))):
			return io.NodeOutput(*not_found)
		return io.NodeOutput(H1, H2, True)


class CalibrateCamera(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CalibrateCamera",
			display_name="CV Calibrate Camera (Chessboard)",
			category=CATEGORY,
			description="Estimates a camera's intrinsics (focal lengths, principal "
			            "point) and lens distortion from several chessboard views "
			            "(cv2.calibrateCamera). Calibration needs per-view 2D/3D "
			            "correspondences accumulated across images, which the raw "
			            "wrappers can't express, so this node manages it: feed an "
			            "IMAGE BATCH of board photos from DIFFERENT angles; each is "
			            "grayscaled, its inner corners are found and refined to "
			            "sub-pixel (cornerSubPix), and the matching planar object "
			            "points are built from pattern size x square_size. Views "
			            "where the board is not found are skipped. Failure-tolerant: "
			            "with fewer than 3 usable views it returns found=false, an "
			            "identity matrix and zero distortion. Feed camera_matrix + "
			            "dist_coeffs into cv2.undistort, or cv2.solvePnP for pose; "
			            "rms_error (px) reports fit quality (under ~1 is good).",
			inputs=[
				io.Image.Input("images",
					tooltip="Batch of chessboard views from different angles "
					        "(>= 3 usable; ~10-20 gives a good fit)."),
				io.Int.Input("pattern_cols", default=9, min=2, max=40,
					tooltip="Inner corners per row (squares per row minus 1)."),
				io.Int.Input("pattern_rows", default=6, min=2, max=40,
					tooltip="Inner corners per column (squares per column minus 1)."),
				io.Float.Input("square_size", default=1.0, min=1e-4, max=1e6,
					tooltip="Physical size of one square (e.g. mm); sets the world "
					        "scale. Leave 1.0 for relative calibration.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="camera_matrix",
					tooltip="3x3 intrinsic matrix K (fx, fy, cx, cy)."),
				NPArray.Output(display_name="dist_coeffs",
					tooltip="Distortion coefficients (k1, k2, p1, p2, k3)."),
				io.Float.Output(display_name="rms_error",
					tooltip="Mean reprojection error in pixels; lower is better (<1 good)."),
				io.Int.Output(display_name="views_used",
					tooltip="Number of input views with a detectable board."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, images, pattern_cols, pattern_rows, square_size=1.0) -> io.NodeOutput:
		pattern = (int(pattern_cols), int(pattern_rows))
		objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
		objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
		objp *= float(square_size)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
		obj_points, img_points, size = [], [], None
		for frame in imageutils.image_batch_to_bgr(images):
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			size = (gray.shape[1], gray.shape[0])  # (w, h)
			ok, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
			if not ok:
				continue
			corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
			obj_points.append(objp.copy())
			img_points.append(corners)
		if len(img_points) < 3 or size is None:
			return io.NodeOutput(np.eye(3, dtype=np.float64),
			                     np.zeros((1, 5), np.float64), 0.0, len(img_points), False)
		rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
		return io.NodeOutput(np.asarray(K, np.float64), np.asarray(dist, np.float64),
		                     float(rms), len(img_points), True)


class FisheyeCalibrate(io.ComfyNode):
	"""cv2.fisheye.calibrate over a batch of chessboard views.

	Three things cost real time here and are all handled inside execute():
	- cv2.fisheye wants ROW-vector point sets, (1, N, 3) / (1, N, 2). The
	  (N, 1, 2) shape findChessboardCorners returns makes it die with an
	  unrelated-looking "Sizes of input arguments do not match" from arithm.cpp.
	- CALIB_RECOMPUTE_EXTRINSIC is effectively mandatory: on the shipped
	  synthetic views, dropping it gives rms 99 px and fx 388 where the truth is
	  0.13 px and 320.
	- ill-conditioned view sets assert inside InitExtrinsics
	  ("fabs(norm_u1) > 0"), which is a detection outcome, not a wiring mistake.
	"""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FisheyeCalibrate",
			display_name="CV Fisheye Calibrate (Chessboard)",
			category=CATEGORY,
			description="Calibrates a WIDE-ANGLE lens with the equidistant "
			            "fisheye model (cv2.fisheye.calibrate) instead of the "
			            "Brown-Conrady model 'Calibrate Camera' uses. Same "
			            "input: a batch of chessboard views from different "
			            "angles, corner-detected and refined to sub-pixel here. "
			            "The output 'dist_coeffs' is FOUR coefficients "
			            "(k1..k4), not five, and only the fisheye nodes accept "
			            "it - feeding it to cv2.undistort is silently wrong. "
			            "Keep CALIB_RECOMPUTE_EXTRINSIC on: without it the "
			            "shipped example calibrates to 99 px RMS and a focal "
			            "length 20% off instead of 0.13 px. Failure tolerant: "
			            "fewer than 3 usable views, or a view set cv2 finds "
			            "ill-conditioned, returns found=false with an identity "
			            "matrix and zero distortion.",
			inputs=[
				io.Image.Input("images",
					tooltip="Batch of chessboard views from different angles "
					        "(>= 3 usable; ~8-20 gives a good fit). Fill the "
					        "frame, but note that boards pushed right into the "
					        "corners can make the solver's initialisation fail."),
				io.Int.Input("pattern_cols", default=9, min=2, max=40,
					tooltip="Inner corners per row (squares per row minus 1)."),
				io.Int.Input("pattern_rows", default=6, min=2, max=40,
					tooltip="Inner corners per column (squares per column minus 1)."),
				io.String.Input("flags",
					default=enums.FISHEYE_CALIB.initial,
					tooltip="Solver options, OR-ed. RECOMPUTE_EXTRINSIC "
					        "re-estimates each view's pose between intrinsic "
					        "steps and is what makes the fit converge; FIX_SKEW "
					        "pins alpha to 0 (what a real lens does); CHECK_COND "
					        "makes cv2 REJECT ill-conditioned views by throwing, "
					        "which this node reports as found=false; FIX_K1..K4 "
					        "hold single distortion terms.",
					extra_dict={"cv_flags": enums.FISHEYE_CALIB.spec()}),
				io.Float.Input("square_size", default=1.0, min=1e-4, max=1e6,
					tooltip="Physical size of one square (e.g. mm); sets the "
					        "world scale. Leave 1.0 for relative calibration.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="camera_matrix",
					tooltip="3x3 intrinsic matrix K (fx, fy, cx, cy)."),
				NPArray.Output(display_name="dist_coeffs",
					tooltip="FOUR fisheye coefficients (k1, k2, k3, k4) as 4x1. "
					        "Only 'Fisheye Undistort' and the other cv2.fisheye "
					        "entry points understand them."),
				io.Float.Output(display_name="rms_error",
					tooltip="Mean reprojection error in pixels; under ~1 is good."),
				io.Int.Output(display_name="views_used",
					tooltip="Number of input views with a detectable board."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, images, pattern_cols, pattern_rows, flags,
	            square_size=1.0) -> io.NodeOutput:
		pattern = (int(pattern_cols), int(pattern_rows))
		# (1, N, 3): cv2.fisheye rejects the (N, 1, 3) convention every other
		# calibration entry point takes
		objp = np.zeros((1, pattern[0] * pattern[1], 3), np.float64)
		objp[0, :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
		objp *= float(square_size)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		detect = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

		obj_points, img_points, size = [], [], None
		for frame in imageutils.image_batch_to_bgr(images):
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			size = (gray.shape[1], gray.shape[0])
			ok, corners = cv2.findChessboardCorners(gray, pattern, flags=detect)
			if not ok:
				continue
			corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
			obj_points.append(objp.copy())
			img_points.append(np.ascontiguousarray(
				np.asarray(corners, np.float64).reshape(1, -1, 2)))

		fail = (np.eye(3, dtype=np.float64), np.zeros((4, 1), np.float64),
		        0.0, len(img_points), False)
		if len(img_points) < 3 or size is None:
			return io.NodeOutput(*fail)
		try:
			rms, K, D, _, _ = cv2.fisheye.calibrate(
				obj_points, img_points, size, None, None,
				flags=enums.FISHEYE_CALIB.resolve(flags))
		except cv2.error as e:  # InitExtrinsics / CHECK_COND
			logging.warning("FisheyeCalibrate: %s", str(e).splitlines()[-1])
			return io.NodeOutput(*fail)
		K = np.asarray(K, np.float64)
		if not np.all(np.isfinite(K)):
			return io.NodeOutput(*fail)
		return io.NodeOutput(K, np.asarray(D, np.float64).reshape(4, 1),
		                     float(rms), len(img_points), True)


class FisheyeUndistort(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGE_ONLY)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FisheyeUndistort",
			display_name="CV Fisheye Undistort",
			category=CATEGORY,
			description="Straightens a wide-angle image with the equidistant "
			            "fisheye model (cv2.fisheye.initUndistortRectifyMap + "
			            "remap). Feed camera_matrix and the FOUR-coefficient "
			            "dist_coeffs from 'Fisheye Calibrate' - the five "
			            "Brown-Conrady coefficients of the pinhole calibrator "
			            "mean something different and produce a plausible but "
			            "wrong result. 'new_camera' is the framing knob: "
			            "'same K' keeps the original focal length and throws "
			            "away everything the straightening pushes off-frame, "
			            "while 'estimate new K' calls "
			            "estimateNewCameraMatrixForUndistortRectify, whose "
			            "'balance' scales the new focal length - 0 keeps the "
			            "longest focal cv2 considers valid, 1 the shortest "
			            "(measured on the shipped example: fx 166 against 50). "
			            "Which of the two leaves black corners depends on the "
			            "lens, so LOOK at both rather than assuming: on that "
			            "example balance 0.0 leaves 3.7% of the frame black and "
			            "1.0 leaves 0.3%. Also "
			            "outputs the K the result is expressed in, which is "
			            "what any later projection needs. NOTE: cv2's "
			            "estimator returns an all-NaN matrix when the "
			            "calibration left k3/k4 unconstrained (which a fit can "
			            "do while still reporting a fine RMS); this node "
			            "detects that, keeps the original K and logs a warning "
			            "- calibrate with CALIB_FIX_K3 | CALIB_FIX_K4 to get a "
			            "usable estimate.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="The distorted wide-angle image."),
				NPArray.Input("camera_matrix",
					tooltip="3x3 K from 'Fisheye Calibrate'."),
				NPArray.Input("dist_coeffs",
					tooltip="FOUR fisheye coefficients (k1..k4). A 5-element "
					        "pinhole vector is rejected rather than silently "
					        "truncated."),
				io.Combo.Input("new_camera",
					options=["same K (crops what straightening pushes off-frame)",
					         "estimate new K (balance controls the field of view)"],
					default="same K (crops what straightening pushes off-frame)",
					tooltip="Which intrinsics the undistorted image is expressed "
					        "in. 'same K' is cv2's own default behaviour."),
				io.Float.Input("balance", default=0.0, min=0.0, max=1.0, step=0.05,
					tooltip="Only read when new_camera estimates one. Blends "
					        "between the two focal lengths cv2 derives from the "
					        "undistorted image bounds: 0 = the longer one, 1 = "
					        "the shorter one (a wider view, everything smaller). "
					        "Check the result - which end leaves black corners "
					        "is lens-dependent."),
				io.Float.Input("fov_scale", default=1.0, min=0.1, max=5.0,
					step=0.05, optional=True, advanced=True,
					tooltip="Extra divisor on the estimated focal length: >1 "
					        "zooms out (more scene, smaller), <1 zooms in."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="The straightened image, same size as the input."),
				NPArray.Output(display_name="new_camera_matrix",
					tooltip="The 3x3 K the output image is expressed in - feed "
					        "this, not the original K, to anything that projects "
					        "into the undistorted image."),
			],
		)

	@classmethod
	def execute(cls, image, camera_matrix, dist_coeffs, new_camera, balance,
	            fov_scale=1.0) -> io.NodeOutput:
		src, tag = polytype.resolve(image)
		K = np.asarray(camera_matrix, np.float64).reshape(3, 3)
		D = np.asarray(dist_coeffs, np.float64).ravel()
		if D.size != 4:
			raise ValueError(
				f"fisheye dist_coeffs must be 4 coefficients (k1..k4), got "
				f"{D.size} - a 5-element vector comes from the PINHOLE "
				"'Calibrate Camera' node and does not describe this model."
			)
		D = D.reshape(4, 1)
		h, w = src.shape[:2]
		new_K = K.copy()
		if str(new_camera).startswith("estimate"):
			est = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
				K, D, (w, h), np.eye(3), balance=float(balance),
				fov_scale=float(fov_scale))
			est = np.asarray(est, np.float64)
			# MEASURED TRAP: the estimator returns an all-NaN matrix for a D
			# whose k3/k4 are unconstrained by the calibration views (the fit
			# still reports 0.13 px RMS). Emitting NaN would poison every
			# downstream op, so fall back to the original framing and say so -
			# the fix is to calibrate with CALIB_FIX_K3 | CALIB_FIX_K4.
			if np.all(np.isfinite(est)):
				new_K = est
			else:
				logging.warning(
					"FisheyeUndistort: estimateNewCameraMatrixForUndistortRectify "
					"returned NaN for D=%s - keeping the original K. Re-run the "
					"calibration with CALIB_FIX_K3 | CALIB_FIX_K4.", D.ravel())
		map1, map2 = cv2.fisheye.initUndistortRectifyMap(
			K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
		out = cv2.remap(src, map1, map2, cv2.INTER_LINEAR,
		                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
		return io.NodeOutput(polytype.restore(out, tag),
		                     np.asarray(new_K, np.float64))


def _require_module(name):
	"""The cv2 submodule, or a clear error naming the repair tool - contrib
	submodules import as EMPTY stubs when the plain opencv-python wheel was
	installed over opencv-contrib-python(-headless) (same rule as contrib._require)."""
	mod = getattr(cv2, name, None)
	if mod is not None and getattr(mod, "calibrate", None) is not None:
		return mod
	raise RuntimeError(
		f"cv2.{name} is missing from this OpenCV build (cv2 {cv2.__version__}). "
		"Installing opencv-python over opencv-contrib-python(-headless) overwrites the "
		"shared binary and leaves the contrib submodules as empty stubs - "
		"diagnose with 'python tools/repair_opencv_contrib.py --check' and "
		"repair with '--apply' (stop the ComfyUI server first).")


def _omnidir_corners(images, pattern, square_size, refine_window):
	"""Shared detection for the omnidir nodes: (1, N, k) point sets, and a
	SMALL cornerSubPix window - the usual 11x11 spans neighbouring squares on a
	compressed omnidir view and drags corners up to 7 px from where
	cv2.omnidir.projectPoints puts them (7x7 keeps every shipped view inside
	0.3 px). The calibration RMS does NOT reveal this: it is LOWER with the
	bigger window, because a smooth bias is absorbed by the fitted parameters."""
	objp = np.zeros((1, pattern[0] * pattern[1], 3), np.float64)
	objp[0, :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
	objp *= float(square_size)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	detect = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
	win = max(2, int(refine_window) // 2)
	obj_points, img_points, size = [], [], None
	for frame in imageutils.image_batch_to_bgr(images):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		size = (gray.shape[1], gray.shape[0])
		ok, corners = cv2.findChessboardCorners(gray, pattern, flags=detect)
		if not ok:
			continue
		corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), criteria)
		obj_points.append(objp.copy())
		img_points.append(np.ascontiguousarray(
			np.asarray(corners, np.float64).reshape(1, -1, 2)))
	return obj_points, img_points, size


class OmnidirCalibrate(io.ComfyNode):
	"""cv2.omnidir.calibrate - CMei's catadioptric model (K, xi, D).

	Same (1, N, k) point-set requirement as the fisheye calibrator, plus its
	own quirks: the flags live in cv2.omnidir and collide by NAME with the
	top-level CALIB_* ones (FIX_SKEW is 2 here, 1<<25 there), the call takes
	flags and criteria POSITIONALLY, and it returns an `idx` of the views it
	decided to keep - which can be fewer than it was given.
	"""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_OmnidirCalibrate",
			display_name="CV Omnidir Calibrate (Chessboard)",
			category=CATEGORY,
			description="Calibrates a catadioptric / >180-degree lens with "
			            "CMei's omnidirectional model (cv2.omnidir.calibrate). "
			            "It adds a mirror parameter 'xi' to the usual K and "
			            "four (k1, k2, p1, p2) coefficients, which is what "
			            "lets it describe a view wider than any pinhole or "
			            "fisheye model. Feed a batch of chessboard views; "
			            "corner detection uses a 7x7 refinement window by "
			            "default because the usual 11x11 drags corners several "
			            "pixels on a compressed omnidir view. WATCH THE "
			            "PARAMETERS, not just the RMS: focal length and xi are "
			            "strongly correlated, so a fit can reproject at 0.05 px "
			            "while reporting fx 327 / xi 0.96 for a camera whose "
			            "truth is 300 / 0.80 - unless the boards cover the "
			            "periphery, where the two stop being interchangeable. "
			            "Failure tolerant: fewer than 3 usable views, or a cv2 "
			            "error, returns found=false.",
			inputs=[
				io.Image.Input("images",
					tooltip="Batch of chessboard views from different angles "
					        "(>= 3 usable). Spread them across the frame: the "
					        "xi/focal degeneracy is what wide coverage breaks."),
				io.Int.Input("pattern_cols", default=9, min=2, max=40,
					tooltip="Inner corners per row (squares per row minus 1)."),
				io.Int.Input("pattern_rows", default=6, min=2, max=40,
					tooltip="Inner corners per column (squares per column minus 1)."),
				io.String.Input("flags",
					default=enums.OMNIDIR_CALIB.initial,
					tooltip="Solver options, OR-ed. These are the cv2.omnidir "
					        "constants, NOT the top-level CALIB_* ones of the "
					        "same name (different values). FIX_SKEW pins alpha "
					        "to 0; FIX_XI holds the mirror parameter; "
					        "FIX_CENTER holds the principal point.",
					extra_dict={"cv_flags": enums.OMNIDIR_CALIB.spec()}),
				io.Float.Input("square_size", default=1.0, min=1e-4, max=1e6,
					tooltip="Physical size of one square; sets the world scale.",
					optional=True, advanced=True),
				io.Int.Input("refine_window", default=7, min=3, max=31,
					tooltip="cornerSubPix search window (full width, odd). 7 by "
					        "default because at 11 the window spans neighbouring "
					        "squares where the image is compressed and pulls "
					        "corners up to 7 px away from where the model says "
					        "they are - and the calibration RMS goes DOWN when "
					        "that happens (0.057 against 0.073 px on the shipped "
					        "boards), because a smooth bias is absorbed by the "
					        "parameters. Do not tune this by RMS.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="camera_matrix",
					tooltip="3x3 intrinsic matrix K."),
				io.Float.Output(display_name="xi",
					tooltip="CMei's mirror parameter. 0 would be a pinhole; the "
					        "bigger it is, the further past 180 degrees the "
					        "model reaches."),
				NPArray.Output(display_name="dist_coeffs",
					tooltip="FOUR coefficients (k1, k2, p1, p2) as 4x1 - the "
					        "omnidir set, not the pinhole or fisheye one."),
				io.Float.Output(display_name="rms_error",
					tooltip="Mean reprojection error in pixels. A small value "
					        "does NOT mean K and xi are individually right."),
				io.Int.Output(display_name="views_used",
					tooltip="Views cv2 actually kept (it drops some itself)."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, images, pattern_cols, pattern_rows, flags,
	            square_size=1.0, refine_window=7) -> io.NodeOutput:
		_require_module("omnidir")
		obj_points, img_points, size = _omnidir_corners(
			images, (int(pattern_cols), int(pattern_rows)), square_size,
			refine_window)
		fail = (np.eye(3, dtype=np.float64), 0.0, np.zeros((4, 1), np.float64),
		        0.0, 0, False)
		if len(img_points) < 3 or size is None:
			return io.NodeOutput(*fail)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-9)
		try:
			# flags and criteria are POSITIONAL here, unlike every other
			# calibrate() in cv2
			rms, K, xi, D, _, _, idx = cv2.omnidir.calibrate(
				obj_points, img_points, size, None, None, None,
				enums.OMNIDIR_CALIB.resolve(flags), criteria)
		except cv2.error as e:
			logging.warning("OmnidirCalibrate: %s", str(e).splitlines()[-1])
			return io.NodeOutput(*fail)
		K = np.asarray(K, np.float64)
		xi = float(np.asarray(xi).ravel()[0])
		if not np.all(np.isfinite(K)) or not np.isfinite(xi):
			return io.NodeOutput(*fail)
		return io.NodeOutput(K, xi, np.asarray(D, np.float64).reshape(4, 1),
		                     float(rms), int(np.asarray(idx).size), True)


class OmnidirUndistort(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGE_ONLY)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_OmnidirUndistort",
			display_name="CV Omnidir Undistort",
			category=CATEGORY,
			description="Re-projects an omnidirectional image "
			            "(cv2.omnidir.undistortImage). Unlike the pinhole and "
			            "fisheye undistorters there is no single 'undistorted' "
			            "answer, because the source view is wider than a plane "
			            "can hold: pick the PROJECTION. 'perspective' gives a "
			            "normal photo of the middle and throws the rest away, "
			            "'cylindrical' and 'longitude-latitude' unroll the "
			            "whole field into a panorama-style strip, and "
			            "'stereographic' keeps angles locally. The output "
			            "camera matrix is derived from 'new_focal' for the "
			            "perspective/stereographic modes; the two unrolled "
			            "modes need the width/height-over-pi form instead, "
			            "which 'auto' builds for you (a perspective-style K "
			            "there gives a black or wildly cropped image, and cv2 "
			            "does not complain).",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="The omnidirectional image."),
				NPArray.Input("camera_matrix",
					tooltip="3x3 K from 'Omnidir Calibrate'."),
				NPArray.Input("dist_coeffs",
					tooltip="FOUR omnidir coefficients (k1, k2, p1, p2)."),
				io.Float.Input("xi", default=0.8, min=0.0, max=10.0, step=0.01,
					tooltip="CMei's mirror parameter from 'Omnidir Calibrate'."),
				io.Combo.Input("projection",
					options=enums.OMNIDIR_RECTIFY.options,
					default=enums.OMNIDIR_RECTIFY.default,
					tooltip="RECTIFY_PERSPECTIVE: a flat photo of the centre. "
					        "CYLINDRICAL / LONGLATI: the whole field unrolled. "
					        "STEREOGRAPHIC: conformal, keeps local shapes."),
				io.Float.Input("new_focal", default=200.0, min=0.0, max=10000.0,
					step=10.0,
					tooltip="Focal length of the output camera in pixels "
					        "(perspective / stereographic only): smaller = "
					        "wider view, more of the sphere squeezed in. 0 = "
					        "auto (a quarter of the output width). Ignored by "
					        "the two unrolled projections."),
				NPArray.Input("view_rotation", optional=True, advanced=True,
					tooltip="Where to AIM the output view: a 3x3 rotation (or a "
					        "3x1 Rodrigues vector) applied before the "
					        "re-projection. Identity looks along the optical "
					        "axis, which for a mirror rig is the middle of the "
					        "reflection - anything near the rim gets cropped "
					        "out of a perspective view, and this is the only "
					        "way to point at it."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="The re-projected image, same size as the input."),
				NPArray.Output(display_name="new_camera_matrix",
					tooltip="The 3x3 K the output is expressed in."),
			],
		)

	@classmethod
	def execute(cls, image, camera_matrix, dist_coeffs, xi, projection,
	            new_focal, view_rotation=None) -> io.NodeOutput:
		_require_module("omnidir")
		src, tag = polytype.resolve(image)
		K = np.asarray(camera_matrix, np.float64).reshape(3, 3)
		D = np.asarray(dist_coeffs, np.float64).ravel()
		if D.size != 4:
			raise ValueError(
				f"omnidir dist_coeffs must be 4 coefficients (k1, k2, p1, p2), "
				f"got {D.size} - a 5-element vector is the PINHOLE set.")
		D = D.reshape(1, 4)
		h, w = src.shape[:2]
		flag = enums.OMNIDIR_RECTIFY.resolve(projection)
		unrolled = flag in (cv2.omnidir.RECTIFY_LONGLATI,
		                    cv2.omnidir.RECTIFY_CYLINDRICAL)
		if unrolled:
			# cv2's own convention for the unrolled projections; a
			# perspective-style K here renders black without any error
			new_K = np.float64([[w / np.pi, 0, 0], [0, h / np.pi, 0], [0, 0, 1]])
		else:
			f = float(new_focal) if float(new_focal) > 0 else w / 4.0
			new_K = np.float64([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])
		R = np.eye(3)
		if view_rotation is not None and np.asarray(view_rotation).size:
			rot = np.asarray(view_rotation, np.float64)
			if rot.size == 3:
				R = cv2.Rodrigues(rot.reshape(3, 1))[0]
			elif rot.size == 9:
				R = rot.reshape(3, 3)
			else:
				raise ValueError(
					f"view_rotation must be a 3x3 matrix or a 3-element "
					f"Rodrigues vector, got {rot.size} values.")
		out = cv2.omnidir.undistortImage(
			src, K, D, np.float64([[float(xi)]]), flag, Knew=new_K,
			new_size=(w, h), R=R)
		return io.NodeOutput(polytype.restore(np.asarray(out), tag), new_K)


class OmnidirStereoReconstruct(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_OmnidirStereoReconstruct",
			display_name="CV Omnidir Stereo Reconstruct",
			category=CATEGORY,
			description="Rectifies an omnidirectional stereo pair, matches it "
			            "with SGBM and back-projects the result "
			            "(cv2.omnidir.stereoReconstruct) - the whole depth "
			            "pipeline in one call, because the rectification of a "
			            "catadioptric pair is not a homography and cannot be "
			            "wired from the pinhole nodes. Feed both cameras' K, "
			            "xi and 4-coefficient D plus the R and T between them. "
			            "The disparity is expressed in the RECTIFIED frame, so "
			            "compare it against 'left_rectified', not the input. "
			            "Invalid pixels come back as -1 from cv2; this node "
			            "writes 0 there and gives you a 'valid' mask instead. "
			            "Failure tolerant: a cv2 error returns empty outputs "
			            "with found=false.",
			inputs=[
				io.Image.Input("left", tooltip="Left omnidirectional image."),
				io.Image.Input("right", tooltip="Right omnidirectional image."),
				NPArray.Input("K_left", tooltip="3x3 K of the left camera."),
				NPArray.Input("dist_left",
					tooltip="Four omnidir coefficients of the left camera."),
				io.Float.Input("xi_left", default=0.8, min=0.0, max=10.0,
					step=0.01, tooltip="Left mirror parameter."),
				NPArray.Input("K_right", tooltip="3x3 K of the right camera."),
				NPArray.Input("dist_right",
					tooltip="Four omnidir coefficients of the right camera."),
				io.Float.Input("xi_right", default=0.8, min=0.0, max=10.0,
					step=0.01, tooltip="Right mirror parameter."),
				NPArray.Input("R",
					tooltip="3x3 rotation from the left to the right camera."),
				NPArray.Input("T",
					tooltip="3x1 translation from the left to the right camera. "
					        "Its length sets the world scale of the cloud."),
				io.Combo.Input("projection",
					options=enums.OMNIDIR_STEREO_RECTIFY.options,
					default=enums.OMNIDIR_STEREO_RECTIFY.default,
					tooltip="Rectification the matcher runs in. PERSPECTIVE "
					        "keeps the middle of the field at higher "
					        "resolution; LONGLATI unrolls the whole sphere and "
					        "matches more of it, at lower resolution per "
					        "pixel. cv2 accepts only these two here."),
				io.Int.Input("num_disparities", default=64, min=16, max=512,
					step=16,
					tooltip="SGBM search range, must be a multiple of 16. It "
					        "sets the NEAREST distance the pair can see."),
				io.Int.Input("block_size", default=7, min=3, max=51,
					tooltip="SGBM matching window (odd). Larger = smoother and "
					        "blinder to detail."),
			],
			outputs=[
				NPArray.Output(display_name="disparity",
					tooltip="Float32 disparity in the RECTIFIED frame; 0 where "
					        "nothing matched (cv2 writes -1 there)."),
				NPArray.Output(display_name="valid",
					tooltip="Uint8 mask, 255 where the disparity is real."),
				io.Image.Output(display_name="left_rectified",
					tooltip="The rectified left image - the frame the disparity "
					        "and the cloud are aligned with."),
				io.Image.Output(display_name="right_rectified",
					tooltip="The rectified right image."),
				NPArray.Output(display_name="point_cloud",
					tooltip="Nx3 float32 cloud in the left camera's frame, in "
					        "the units of T. Feed 'CV Filter Point Cloud' or "
					        "'CV Point Cloud Normals'."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, left, right, K_left, dist_left, xi_left, K_right,
	            dist_right, xi_right, R, T, projection, num_disparities,
	            block_size) -> io.NodeOutput:
		_require_module("omnidir")
		lf = imageutils.image_batch_to_bgr(left)[0]
		rf = imageutils.image_batch_to_bgr(right)[0]
		h, w = lf.shape[:2]

		def _coeffs(d, name):
			arr = np.asarray(d, np.float64).ravel()
			if arr.size != 4:
				raise ValueError(
					f"{name} must be 4 omnidir coefficients (k1, k2, p1, p2), "
					f"got {arr.size}.")
			return arr.reshape(1, 4)

		K1 = np.asarray(K_left, np.float64).reshape(3, 3)
		K2 = np.asarray(K_right, np.float64).reshape(3, 3)
		D1, D2 = _coeffs(dist_left, "dist_left"), _coeffs(dist_right, "dist_right")
		xi1 = np.float64([[float(xi_left)]])
		xi2 = np.float64([[float(xi_right)]])
		rot = np.asarray(R, np.float64).reshape(3, 3)
		tr = np.asarray(T, np.float64).reshape(3, 1)
		empty = (np.zeros((h, w), np.float32), np.zeros((h, w), np.uint8),
		         imageutils.bgr_to_image_batch([np.zeros_like(lf)]),
		         imageutils.bgr_to_image_batch([np.zeros_like(rf)]),
		         np.zeros((0, 3), np.float32), False)
		try:
			# new_size is NOT optional in practice: without it cv2 returns an
			# EMPTY object array for pointCloud and says nothing
			disparity, rec1, rec2, cloud = cv2.omnidir.stereoReconstruct(
				lf, rf, K1, D1, xi1, K2, D2, xi2, rot, tr,
				enums.OMNIDIR_STEREO_RECTIFY.resolve(projection),
				int(num_disparities), int(block_size),
				newSize=(w, h), pointType=cv2.omnidir.XYZ)
		except cv2.error as e:
			logging.warning("OmnidirStereoReconstruct: %s",
			                str(e).splitlines()[-1])
			return io.NodeOutput(*empty)

		disparity = np.asarray(disparity, np.float32)
		valid = (disparity > 0).astype(np.uint8) * 255
		disparity = np.where(disparity > 0, disparity, 0.0).astype(np.float32)
		# cv2 hands back a 0-D OBJECT array when it has no cloud to give (a
		# textureless pair, or new_size omitted) - and numpy reports THAT as
		# size 1, so the emptiness test has to be on the dimensionality
		cloud = np.asarray(cloud)
		cloud = (np.asarray(cloud, np.float32).reshape(-1, 3)
		         if cloud.ndim >= 2 and cloud.size else np.zeros((0, 3), np.float32))
		cloud = cloud[np.isfinite(cloud).all(1)]
		return io.NodeOutput(
			disparity, valid,
			imageutils.bgr_to_image_batch([np.asarray(rec1)]),
			imageutils.bgr_to_image_batch([np.asarray(rec2)]),
			np.ascontiguousarray(cloud), bool(len(cloud)))


class StereoCalibrate(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_StereoCalibrate",
			display_name="CV Stereo Calibrate (Chessboard)",
			category=CATEGORY,
			description="Estimates the relative pose (R, T) between two cameras - "
			            "and both sets of intrinsics unless they are pinned - from "
			            "matched chessboard "
			            "views (cv2.stereoCalibrate). Feed an IMAGE BATCH of stereo "
			            "pairs (left and right views of the SAME board poses, same "
			            "order); each pair is corner-detected independently and the "
			            "matching 3-D/2-D correspondences are accumulated. "
			            "Pre-calibrate each camera individually with 'Calibrate "
			            "Camera' and feed BOTH K matrices here to PIN the "
			            "intrinsics (CALIB_FIX_INTRINSIC - only R and T are then "
			            "estimated, which is the recommended two-stage workflow); "
			            "with one or neither K connected the intrinsics are "
			            "estimated/refined here and any K you did pass is only an "
			            "initial guess. "
			            "Failure-tolerant: with fewer than 3 usable views returns "
			            "found=false with identity R and zero T. Feed R, T, K, dist "
			            "into 'cv2.stereoRectify' for rectification, then "
			            "'Stereo Disparity (SGBM)' for depth.",
			inputs=[
				io.Image.Input("images_left",
					tooltip="Batch of LEFT chessboard views (>= 3 usable pairs; "
					        "~10-20 gives a good fit). Must be in the same order "
					        "as images_right."),
				io.Image.Input("images_right",
					tooltip="Batch of RIGHT chessboard views, same order/length "
					        "as images_left."),
				io.Int.Input("pattern_cols", default=9, min=2, max=40,
					tooltip="Inner corners per row (squares per row minus 1)."),
				io.Int.Input("pattern_rows", default=6, min=2, max=40,
					tooltip="Inner corners per column (squares per column minus 1)."),
				io.Float.Input("square_size", default=1.0, min=1e-4, max=1e6,
					tooltip="Physical size of one square (e.g. mm); sets the world "
					        "scale. Leave 1.0 for relative calibration.",
					optional=True, advanced=True),
				NPArray.Input("K_left", optional=True,
					tooltip="3x3 intrinsics for the left camera from 'Calibrate "
					        "Camera'. Connect BOTH K_left and K_right to hold the "
					        "intrinsics FIXED and solve only for R/T; on its own it "
					        "is just an initial guess and still gets refined. Leave "
					        "unconnected for a fresh estimate."),
				NPArray.Input("dist_left", optional=True,
					tooltip="Left distortion coefficients from 'Calibrate Camera'. "
					        "Also held fixed when both K matrices are connected."),
				NPArray.Input("K_right", optional=True,
					tooltip="3x3 intrinsics for the right camera. See K_left: both "
					        "connected = intrinsics pinned, one alone = initial "
					        "guess only."),
				NPArray.Input("dist_right", optional=True,
					tooltip="Right distortion coefficients."),
			],
			outputs=[
				NPArray.Output(display_name="K_left",
					tooltip="Refined 3x3 intrinsic matrix for the left camera."),
				NPArray.Output(display_name="dist_left",
					tooltip="Refined distortion coefficients for the left camera."),
				NPArray.Output(display_name="K_right",
					tooltip="Refined 3x3 intrinsic matrix for the right camera."),
				NPArray.Output(display_name="dist_right",
					tooltip="Refined distortion coefficients for the right camera."),
				NPArray.Output(display_name="R",
					tooltip="3x3 rotation matrix from left to right camera."),
				NPArray.Output(display_name="T",
					tooltip="3x1 translation vector from left to right camera."),
				io.Float.Output(display_name="rms_error",
					tooltip="Mean reprojection error in pixels; lower is better."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, images_left, images_right, pattern_cols, pattern_rows,
	            square_size=1.0, K_left=None, dist_left=None,
	            K_right=None, dist_right=None) -> io.NodeOutput:
		pattern = (int(pattern_cols), int(pattern_rows))
		objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
		objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
		objp *= float(square_size)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

		obj_points, img_points_l, img_points_r = [], None, None
		size = None
		left_frames = list(imageutils.image_batch_to_bgr(images_left))
		right_frames = list(imageutils.image_batch_to_bgr(images_right))
		n_pairs = min(len(left_frames), len(right_frames))

		for i in range(n_pairs):
			gray_l = cv2.cvtColor(left_frames[i], cv2.COLOR_BGR2GRAY)
			gray_r = cv2.cvtColor(right_frames[i], cv2.COLOR_BGR2GRAY)
			size = (gray_l.shape[1], gray_l.shape[0])

			ok_l, corners_l = cv2.findChessboardCorners(gray_l, pattern, flags=flags)
			ok_r, corners_r = cv2.findChessboardCorners(gray_r, pattern, flags=flags)
			if not (ok_l and ok_r):
				continue
			corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
			corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
			obj_points.append(objp.copy())
			if img_points_l is None:
				img_points_l = [corners_l]
				img_points_r = [corners_r]
			else:
				img_points_l.append(corners_l)
				img_points_r.append(corners_r)

		# Failure-tolerant: not enough views
		identity_R = np.eye(3, dtype=np.float64)
		zero_T = np.zeros((3, 1), dtype=np.float64)
		zero_dist = np.zeros((1, 5), dtype=np.float64)
		if len(obj_points) < 3 or size is None or img_points_l is None:
			K_fallback = np.eye(3, dtype=np.float64)
			return io.NodeOutput(K_fallback, zero_dist.copy(), K_fallback, zero_dist.copy(),
			                     identity_R, zero_T, 0.0, False)

		# Initial guesses from individual calibration if provided
		if K_left is not None:
			K1 = np.asarray(K_left, np.float64).copy()
		else:
			K1 = None
		if dist_left is not None:
			D1 = np.asarray(dist_left, np.float64).copy()
		else:
			D1 = None
		if K_right is not None:
			K2 = np.asarray(K_right, np.float64).copy()
		else:
			K2 = None
		if dist_right is not None:
			D2 = np.asarray(dist_right, np.float64).copy()
		else:
			D2 = None

		try:
			rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
				obj_points, img_points_l, img_points_r,
				K1, D1, K2, D2, size,
				criteria=criteria,
				flags=cv2.CALIB_FIX_INTRINSIC if (K_left is not None and K_right is not None) else 0)
		except cv2.error:
			return io.NodeOutput(np.eye(3, dtype=np.float64), zero_dist.copy(),
			                     np.eye(3, dtype=np.float64), zero_dist.copy(),
			                     identity_R, zero_T, 0.0, False)

		if not (np.all(np.isfinite(K1)) and np.all(np.isfinite(K2))
		        and np.all(np.isfinite(R)) and np.all(np.isfinite(T))):
			return io.NodeOutput(np.eye(3, dtype=np.float64), zero_dist.copy(),
			                     np.eye(3, dtype=np.float64), zero_dist.copy(),
			                     identity_R, zero_T, 0.0, False)

		return io.NodeOutput(
			np.asarray(K1, np.float64),
			np.asarray(D1, np.float64),
			np.asarray(K2, np.float64),
			np.asarray(D2, np.float64),
			np.asarray(R, np.float64),
			np.asarray(T, np.float64),
			float(rms),
			True)


class StereoCalibrateCircleGrid(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_StereoCalibrateCircleGrid",
			display_name="CV Stereo Calibrate (Circle Grid)",
			category=CATEGORY,
			description="Estimates the relative pose (R, T) between two cameras - "
			            "and both sets of intrinsics unless they are pinned - from "
			            "matched asymmetric circle-grid views "
			            "(cv2.stereoCalibrate). Feed "
			            "an IMAGE BATCH of stereo pairs (left and right views of "
			            "the SAME board poses, same order); each pair is "
			            "circle-grid detected independently and skipped unless "
			            "BOTH views find the full grid. The 'asymmetric "
			            "(staggered)' layout is the default and the point of this "
			            "node: a symmetric grid looks identical when rotated 180 "
			            "degrees, so a tilted board seen by two cameras cannot be "
			            "matched unambiguously - the staggered rows break that "
			            "ambiguity, which is why asymmetric circle grids are the "
			            "standard target for stereo rigs. Pre-calibrate each "
			            "camera with 'Calibrate Camera (Circle Grid)' and feed "
			            "BOTH K matrices here to PIN the intrinsics "
			            "(CALIB_FIX_INTRINSIC - only R and T are then estimated, "
			            "which is the recommended two-stage workflow); with one or "
			            "neither K connected the intrinsics are estimated/refined "
			            "here and any K you did pass is only an initial guess. "
			            "Failure-"
			            "tolerant: with fewer than 3 usable pairs returns "
			            "found=false with identity R and zero T. Feed R, T, K, "
			            "dist into 'cv2.stereoRectify' for rectification.",
			inputs=[
				io.Image.Input("images_left",
					tooltip="Batch of LEFT circle-grid views (>= 3 usable pairs; "
					        "~10-20 gives a good fit). Same order as images_right."),
				io.Image.Input("images_right",
					tooltip="Batch of RIGHT circle-grid views, same order/length "
					        "as images_left."),
				io.Int.Input("pattern_cols", default=4, min=2, max=40,
					tooltip="Circle centers per row (width of the grid)."),
				io.Int.Input("pattern_rows", default=11, min=2, max=40,
					tooltip="Circle centers per column (height of the grid)."),
				io.Combo.Input("grid_layout",
					options=["asymmetric (staggered)", "symmetric"],
					default="asymmetric (staggered)",
					tooltip="Grid geometry. Asymmetric staggers each row by half a "
					        "spacing, which removes the 180-degree rotational "
					        "ambiguity of symmetric targets - essential for matching "
					        "a tilted board across two views."),
				io.Float.Input("square_size", default=1.0, min=1e-4, max=1e6,
					tooltip="Physical center-to-center spacing (e.g. mm); sets the "
					        "world scale. Leave 1.0 for relative calibration.",
					optional=True, advanced=True),
				NPArray.Input("K_left", optional=True,
					tooltip="3x3 intrinsics for the left camera from 'Calibrate "
					        "Camera (Circle Grid)'. Connect BOTH K_left and K_right "
					        "to hold the intrinsics FIXED and solve only for R/T; "
					        "on its own it is just an initial guess and still gets "
					        "refined. Leave unconnected for a fresh estimate."),
				NPArray.Input("dist_left", optional=True,
					tooltip="Left distortion coefficients from 'Calibrate Camera "
					        "(Circle Grid)'. Also held fixed when both K matrices "
					        "are connected."),
				NPArray.Input("K_right", optional=True,
					tooltip="3x3 intrinsics for the right camera. See K_left: both "
					        "connected = intrinsics pinned, one alone = initial "
					        "guess only."),
				NPArray.Input("dist_right", optional=True,
					tooltip="Right distortion coefficients."),
			],
			outputs=[
				NPArray.Output(display_name="K_left",
					tooltip="Refined 3x3 intrinsic matrix for the left camera."),
				NPArray.Output(display_name="dist_left",
					tooltip="Refined distortion coefficients for the left camera."),
				NPArray.Output(display_name="K_right",
					tooltip="Refined 3x3 intrinsic matrix for the right camera."),
				NPArray.Output(display_name="dist_right",
					tooltip="Refined distortion coefficients for the right camera."),
				NPArray.Output(display_name="R",
					tooltip="3x3 rotation matrix from left to right camera."),
				NPArray.Output(display_name="T",
					tooltip="3x1 translation vector from left to right camera."),
				io.Float.Output(display_name="rms_error",
					tooltip="Mean reprojection error in pixels; lower is better."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, images_left, images_right, pattern_cols, pattern_rows,
	            grid_layout, square_size=1.0, K_left=None, dist_left=None,
	            K_right=None, dist_right=None) -> io.NodeOutput:
		pattern = (int(pattern_cols), int(pattern_rows))
		objp = _circle_grid_object_points(pattern, grid_layout, square_size)
		obj_points, img_points_l, img_points_r = [], None, None
		size = None
		left_frames = list(imageutils.image_batch_to_bgr(images_left))
		right_frames = list(imageutils.image_batch_to_bgr(images_right))
		n_pairs = min(len(left_frames), len(right_frames))

		for i in range(n_pairs):
			gray_l = _to_gray_u8(left_frames[i])
			gray_r = _to_gray_u8(right_frames[i])
			size = (gray_l.shape[1], gray_l.shape[0])
			ok_l, centers_l = _find_circle_grid_centers(left_frames[i], pattern, grid_layout)
			ok_r, centers_r = _find_circle_grid_centers(right_frames[i], pattern, grid_layout)
			if not (ok_l and ok_r):
				continue
			obj_points.append(objp.copy())
			if img_points_l is None:
				img_points_l = [np.asarray(centers_l, np.float32).reshape(-1, 1, 2)]
				img_points_r = [np.asarray(centers_r, np.float32).reshape(-1, 1, 2)]
			else:
				img_points_l.append(np.asarray(centers_l, np.float32).reshape(-1, 1, 2))
				img_points_r.append(np.asarray(centers_r, np.float32).reshape(-1, 1, 2))

		identity_R = np.eye(3, dtype=np.float64)
		zero_T = np.zeros((3, 1), dtype=np.float64)
		zero_dist = np.zeros((1, 5), dtype=np.float64)
		if len(obj_points) < 3 or size is None or img_points_l is None:
			K_fallback = np.eye(3, dtype=np.float64)
			return io.NodeOutput(K_fallback, zero_dist.copy(), K_fallback, zero_dist.copy(),
			                     identity_R, zero_T, 0.0, False)

		K1 = np.asarray(K_left, np.float64).copy() if K_left is not None else None
		D1 = np.asarray(dist_left, np.float64).copy() if dist_left is not None else None
		K2 = np.asarray(K_right, np.float64).copy() if K_right is not None else None
		D2 = np.asarray(dist_right, np.float64).copy() if dist_right is not None else None

		try:
			rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
				obj_points, img_points_l, img_points_r,
				K1, D1, K2, D2, size,
				criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
				flags=cv2.CALIB_FIX_INTRINSIC if (K_left is not None and K_right is not None) else 0)
		except cv2.error:
			return io.NodeOutput(np.eye(3, dtype=np.float64), zero_dist.copy(),
			                     np.eye(3, dtype=np.float64), zero_dist.copy(),
			                     identity_R, zero_T, 0.0, False)

		if not (np.all(np.isfinite(K1)) and np.all(np.isfinite(K2))
		        and np.all(np.isfinite(R)) and np.all(np.isfinite(T))):
			return io.NodeOutput(np.eye(3, dtype=np.float64), zero_dist.copy(),
			                     np.eye(3, dtype=np.float64), zero_dist.copy(),
			                     identity_R, zero_T, 0.0, False)

		return io.NodeOutput(
			np.asarray(K1, np.float64),
			np.asarray(D1, np.float64),
			np.asarray(K2, np.float64),
			np.asarray(D2, np.float64),
			np.asarray(R, np.float64),
			np.asarray(T, np.float64),
			float(rms),
			True)


class DecomposeProjectionMatrix(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DecomposeProjectionMatrix",
			display_name="CV Decompose Projection Matrix (RQ)",
			category=CATEGORY,
			description="Splits a camera/projection matrix with cv2.RQDecomp3x3. "
			            "This is the compact RQ decomposition used to recover a "
			            "K-like upper-triangular matrix and an orthogonal rotation "
			            "from M = R @ Q, while also exposing the per-axis rotation "
			            "matrices OpenCV computes. A full 3x4 projection matrix "
			            "P = K[R|t] is accepted directly: its left 3x3 block "
			            "(K @ R) is what gets decomposed, so R is the intrinsics "
			            "and Q the camera rotation. The translation is NOT "
			            "recovered - use 'Solve PnP' / 'Pose To Matrix' for that.",
			inputs=[
				NPArray.Input("matrix",
					tooltip="Matrix to decompose: either a 3x3 matrix, or a full "
					        "3x4 camera projection matrix P = K[R|t] (only its left "
					        "3x3 block is decomposed; the translation column is "
					        "ignored)."),
			],
			outputs=[
				NPArray.Output(display_name="matrix",
					tooltip="The 3x3 block that was actually decomposed, as float64 "
					        "(the input itself for a 3x3 matrix, its left 3x3 block "
					        "for a 3x4 one). R @ Q reconstructs exactly this - handy "
					        "for matrix comparisons and inspectors."),
				NPArray.Output(display_name="Q",
					tooltip="3x3 orthogonal matrix from the RQ decomposition."),
				NPArray.Output(display_name="R",
					tooltip="3x3 upper-triangular matrix from the RQ decomposition "
					        "(K-like for camera matrices)."),
				NPArray.Output(display_name="Qx",
					tooltip="OpenCV's x-axis rotation matrix component. NOTE the "
					        "composition order: OpenCV reduces M by RIGHT-multiplying "
					        "these, so Qx @ Qy @ Qz is Q TRANSPOSED - the identity "
					        "that holds is Q = Qz.T @ Qy.T @ Qx.T (exactly, and "
					        "M = R @ Qz.T @ Qy.T @ Qx.T)."),
				NPArray.Output(display_name="Qy",
					tooltip="OpenCV's y-axis rotation matrix component. Composes as "
					        "Q = Qz.T @ Qy.T @ Qx.T, not Qx @ Qy @ Qz (see Qx)."),
				NPArray.Output(display_name="Qz",
					tooltip="OpenCV's z-axis rotation matrix component. Composes as "
					        "Q = Qz.T @ Qy.T @ Qx.T, not Qx @ Qy @ Qz (see Qx)."),
			],
		)

	@classmethod
	def execute(cls, matrix) -> io.NodeOutput:
		M = np.asarray(matrix, np.float64)
		if M.shape == (3, 4):
			# P = K[R|t]: the left 3x3 block is K @ R, which is what RQ splits.
			# Verified identical to cv2.decomposeProjectionMatrix's K/R/Qx/Qy/Qz.
			M = M[:, :3]
		if M.shape != (3, 3):
			raise ValueError(
				f"matrix must be 3x3, or 3x4 for a projection matrix, got shape "
				f"{M.shape}.")
		if not np.all(np.isfinite(M)):
			raise ValueError("matrix contains NaN or Inf values.")
		_, R, Q, Qx, Qy, Qz = cv2.RQDecomp3x3(M)
		return io.NodeOutput(
			M.copy(),  # never hand back an upstream node's cached array
			np.asarray(Q, np.float64),
			np.asarray(R, np.float64),
			np.asarray(Qx, np.float64),
			np.asarray(Qy, np.float64),
			np.asarray(Qz, np.float64))


class DecomposeHomography(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DecomposeHomography",
			display_name="CV Decompose Homography",
			category=CATEGORY,
			description="Recovers the camera motion hidden in a PLANAR homography "
			            "(cv2.decomposeHomographyMat): given H and the camera "
			            "matrix K it returns the rotation, translation and plane "
			            "normal that would produce it. A homography between two "
			            "views of a plane is mathematically ambiguous, so OpenCV "
			            "returns up to FOUR candidate solutions - two of them "
			            "mirror pairs. Feed 'ref_points' (the source-image points "
			            "the homography was fitted on) to have "
			            "filterHomographyDecompByVisibleRefpoints drop the ones "
			            "that would put the plane BEHIND a camera; the 'visible' "
			            "mask marks the survivors. That normally removes the two "
			            "mirror solutions and leaves TWO - the remaining two-fold "
			            "ambiguity is real geometry, not a bug, and needs a third "
			            "view or a known plane normal to break. Pair with 'CV "
			            "Index Batch' (widget-convert its index and wire "
			            "best_index) to pull a candidate out. Failure tolerant: a "
			            "degenerate or non-finite H gives found=false with empty "
			            "stacks instead of raising.",
			inputs=[
				NPArray.Input("homography",
					tooltip="3x3 homography mapping the SOURCE view to the "
					        "DESTINATION view ('CV Find Homography', "
					        "cv2.getPerspectiveTransform, or a composed matrix)."),
				NPArray.Input("camera_matrix",
					tooltip="3x3 intrinsic matrix K of the camera that took BOTH "
					        "views ('CV Camera Matrix' / 'Calibrate Camera')."),
				NPArray.Input("ref_points", optional=True,
					tooltip="Optional Nx2 (or Nx1x2) PIXEL points in the SOURCE "
					        "image that lie on the plane - normally the very "
					        "correspondences H was fitted on. At least 4 are "
					        "needed. They are normalized with K and mapped through "
					        "H internally, then used to reject the candidates that "
					        "would put the plane behind a camera (filling 'visible' "
					        "and 'best_index')."),
			],
			outputs=[
				NPArray.Output(display_name="rotations",
					tooltip="(N, 3, 3) float64 stack of candidate rotation "
					        "matrices, N up to 4. Use 'CV Index Batch' to take one."),
				NPArray.Output(display_name="translations",
					tooltip="(N, 3) float64 stack of candidate translations, "
					        "SCALED BY THE PLANE DISTANCE: each row is t/d, not t. "
					        "Only its DIRECTION is metric unless you know d - "
					        "multiply by the real distance to the plane for units."),
				NPArray.Output(display_name="normals",
					tooltip="(N, 3) float64 stack of candidate plane normals, in "
					        "the SOURCE camera frame (unit length)."),
				NPArray.Output(display_name="visible",
					tooltip="(N, 1) uint8 mask, 255 for each candidate that keeps "
					        "ref_points in front of both cameras. All 255 when no "
					        "ref_points were given. Expect TWO survivors out of "
					        "four: the filter removes the mirror pair, not the "
					        "genuine two-fold planar ambiguity."),
				io.Int.Output(display_name="count",
					tooltip="How many candidate solutions were returned (0 when "
					        "not found, otherwise usually 4)."),
				io.Int.Output(display_name="best_index",
					tooltip="Index of the FIRST candidate that survives the "
					        "visibility filter (0 when no ref_points were given, "
					        "-1 when nothing was found or every candidate was "
					        "rejected). It is not necessarily the right one - "
					        "check 'visible' for the other survivor, and pick "
					        "between them with an outside cue (e.g. the dot "
					        "product of 'normals' with a known plane normal, then "
					        "'CV Pick Value (sorted)'). 'CV Index Batch' "
					        "clamps a -1 to 0, so branch on 'found' first."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, homography, camera_matrix, ref_points=None) -> io.NodeOutput:
		def _not_found():
			return io.NodeOutput(np.zeros((0, 3, 3), np.float64),
			                     np.zeros((0, 3), np.float64),
			                     np.zeros((0, 3), np.float64),
			                     np.zeros((0, 1), np.uint8), 0, -1, False)

		# an upstream estimator legitimately reports 'nothing found' as None or an
		# empty array - that must flow through, not crash
		if homography is None or camera_matrix is None:
			return _not_found()
		H = np.asarray(homography, np.float64)
		K = np.asarray(camera_matrix, np.float64)
		if H.size == 0 or K.size == 0:
			return _not_found()
		if H.shape[-2:] != (3, 3) or H.size != 9:
			raise ValueError(f"homography must be 3x3, got shape {H.shape}.")
		if K.shape[-2:] != (3, 3) or K.size != 9:
			raise ValueError(f"camera_matrix must be 3x3, got shape {K.shape}.")
		H = H.reshape(3, 3)
		K = K.reshape(3, 3)
		# a NaN H is what a failed estimate looks like, not a wiring mistake
		if not (np.all(np.isfinite(H)) and np.all(np.isfinite(K))):
			return _not_found()
		try:
			count, rots, trans, norms = cv2.decomposeHomographyMat(H, K)
		except cv2.error:
			return _not_found()
		if not count or not len(rots):
			return _not_found()
		R = np.stack([np.asarray(r, np.float64).reshape(3, 3) for r in rots])
		T = np.stack([np.asarray(t, np.float64).ravel() for t in trans])
		Nm = np.stack([np.asarray(n, np.float64).ravel() for n in norms])
		if not (np.all(np.isfinite(R)) and np.all(np.isfinite(T))
				and np.all(np.isfinite(Nm))):
			return _not_found()

		visible = np.full((len(R), 1), 255, np.uint8)
		best = 0
		ref = None if ref_points is None else np.asarray(ref_points)
		if ref is not None and ref.size:
			# the filter is bound to CV_32FC2 ONLY - a float64 Nx1x2 raises
			ref32 = np.ascontiguousarray(ref.reshape(-1, 1, 2), np.float32)
			if len(ref32) >= 4:
				try:
					# cv2 5.0's perspectiveTransform returns the array itself
					after = np.ascontiguousarray(
						cv2.perspectiveTransform(ref32, H), np.float32)
					# 'rectified' in OpenCV's docs means NORMALIZED camera
					# coordinates: the filter dots each point with the candidate
					# normal, so feeding PIXELS makes every test pass on the +z
					# side and it silently keeps one arbitrary solution.
					sol = cv2.filterHomographyDecompByVisibleRefpoints(
						rots, norms,
						np.ascontiguousarray(
							cv2.undistortPoints(ref32, K, None), np.float32),
						np.ascontiguousarray(
							cv2.undistortPoints(after, K, None), np.float32))
				except cv2.error:
					sol = None
				# 'nothing survives' comes back as an OBJECT array holding None,
				# not as an empty one - indexing with it would raise
				keep = np.zeros(0, np.intp)
				arr = np.zeros(0) if sol is None else np.asarray(sol).ravel()
				if arr.size and arr.dtype != object:
					keep = arr.astype(np.intp)
				visible = np.zeros((len(R), 1), np.uint8)
				if keep.size:
					visible[keep[(keep >= 0) & (keep < len(R))]] = 255
				best = int(np.argmax(visible.ravel())) if visible.any() else -1
		return io.NodeOutput(R, T, Nm, visible, int(len(R)), best, True)


class SolvePnPPoses(io.ComfyNode):
	# every branch returns candidates the same way, so the node can score them all
	# with one cv2.projectPoints pass (solveP3P returns no errors of its own)
	_SOLVERS = {
		"generic - every solution the method finds": None,
		"P3P - 3 or 4 points, up to 4 poses": "SOLVEPNP_P3P",
		"AP3P - 3 or 4 points, up to 4 poses": "SOLVEPNP_AP3P",
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SolvePnPPoses",
			display_name="CV Solve PnP (All Poses)",
			category=CATEGORY,
			description="The MULTI-SOLUTION pose solver: where 'CV Solve PnP "
			            "(Pose)' commits to one answer, this returns EVERY pose "
			            "consistent with the correspondences, each with its own "
			            "reprojection error (cv2.solveP3P / cv2.solvePnPGeneric). "
			            "That matters when the geometry is genuinely ambiguous - "
			            "three points (P3P) admit up to four poses, and a planar "
			            "target viewed near head-on has a well-known two-fold flip "
			            "(IPPE returns both). Use 'best_index' with 'CV Index "
			            "Batch' for the lowest-error candidate, or keep the whole "
			            "stack and disambiguate with extra evidence. Failure "
			            "tolerant: too few points, a count mismatch or a "
			            "degenerate configuration gives found=false with empty "
			            "stacks instead of raising.",
			inputs=[
				NPArray.Input("object_points",
					tooltip="3D points in object space, Nx3 or Nx1x3 ('CV Grid "
					        "Points' for a planar target, 'CV Points' otherwise)."),
				NPArray.Input("image_points",
					tooltip="The matching 2D image points, Nx2 or Nx1x2, in the "
					        "same order as object_points."),
				NPArray.Input("camera_matrix",
					tooltip="3x3 intrinsic matrix K ('CV Camera Matrix' or "
					        "'Calibrate Camera')."),
				io.Combo.Input("solver", options=list(cls._SOLVERS),
					default="generic - every solution the method finds",
					tooltip="'generic' runs the solver named below over ALL the "
					        "points (4 minimum). P3P/AP3P take EXACTLY 3 or 4 "
					        "points and return up to four poses - with 4 the "
					        "extra point only ranks them. AP3P is the newer, "
					        "numerically better-behaved formulation."),
				io.Combo.Input("method", options=enums.SOLVEPNP.options,
					default=enums.SOLVEPNP.default,
					tooltip="Which algorithm the 'generic' solver runs (ignored by "
					        "P3P/AP3P). ITERATIVE returns a single refined pose; "
					        "IPPE / IPPE_SQUARE are for PLANAR targets and are the "
					        "ones that expose the two-fold ambiguity; SQPNP and "
					        "EPNP are non-iterative alternatives."),
				NPArray.Input("dist_coeffs", optional=True,
					tooltip="Lens distortion coefficients (k1, k2, p1, p2[, k3...]); "
					        "leave unconnected for an ideal pinhole camera."),
			],
			outputs=[
				NPArray.Output(display_name="rvecs",
					tooltip="(N, 3) float64 stack of Rodrigues rotation vectors, "
					        "one row per candidate pose. Take one with 'CV Index "
					        "Batch' and feed 'CV Pose To Matrix' / "
					        "cv2.projectPoints / drawFrameAxes."),
				NPArray.Output(display_name="tvecs",
					tooltip="(N, 3) float64 stack of translation vectors, in the "
					        "same order (and units) as object_points."),
				NPArray.Output(display_name="reprojection_error",
					tooltip="(N, 1) float64 mean reprojection error in PIXELS for "
					        "each candidate, computed the same way for every "
					        "solver (cv2.projectPoints over all the points). Under "
					        "~1 px is a good fit; a large gap between the best and "
					        "the second best means the ambiguity is resolved."),
				io.Int.Output(display_name="count",
					tooltip="How many candidate poses were returned (0 when not "
					        "found)."),
				io.Int.Output(display_name="best_index",
					tooltip="Index of the candidate with the lowest reprojection "
					        "error, or -1 when not found."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, object_points, image_points, camera_matrix, solver, method,
	            dist_coeffs=None) -> io.NodeOutput:
		def _not_found():
			return io.NodeOutput(np.zeros((0, 3), np.float64),
			                     np.zeros((0, 3), np.float64),
			                     np.zeros((0, 1), np.float64), 0, -1, False)

		if object_points is None or image_points is None or camera_matrix is None:
			return _not_found()
		obj = np.asarray(object_points, np.float64)
		img = np.asarray(image_points, np.float64)
		if obj.size == 0 or img.size == 0:
			return _not_found()
		obj = obj.reshape(-1, 1, 3)
		img = img.reshape(-1, 1, 2)
		n = len(obj)
		if n != len(img):
			return _not_found()
		K = np.asarray(camera_matrix, np.float64)
		dist = None if dist_coeffs is None else np.asarray(dist_coeffs, np.float64)
		p3p_flag = cls._SOLVERS.get(str(solver))
		# the point-count rules are the solvers' own asserts, pre-checked so a
		# short match list is a soft miss rather than a halted workflow
		if p3p_flag is not None and n not in (3, 4):
			return _not_found()
		if p3p_flag is None and n < 4:
			return _not_found()
		try:
			if p3p_flag is not None:
				count, rvecs, tvecs = cv2.solveP3P(
					obj, img, K, dist, enums.SOLVEPNP.resolve(p3p_flag))
			else:
				count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
					obj, img, K, dist, flags=enums.SOLVEPNP.resolve(method))
		except cv2.error:
			return _not_found()
		if not count or rvecs is None or not len(rvecs):
			return _not_found()

		rv = np.stack([np.asarray(r, np.float64).ravel() for r in rvecs])
		tv = np.stack([np.asarray(t, np.float64).ravel() for t in tvecs])
		keep = np.isfinite(rv).all(1) & np.isfinite(tv).all(1)
		rv, tv = rv[keep], tv[keep]
		if not len(rv):
			return _not_found()
		# scored uniformly across solvers: solveP3P returns no errors at all, and
		# solvePnPGeneric's are only comparable within one call
		errs = []
		for r, t in zip(rv, tv):
			proj, _ = cv2.projectPoints(obj, r, t, K, dist)
			errs.append(float(np.linalg.norm(
				proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1).mean()))
		err = np.asarray(errs, np.float64).reshape(-1, 1)
		return io.NodeOutput(rv, tv, err, int(len(rv)), int(np.argmin(err)), True)


class SolvePnPPose(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SolvePnPPose",
			display_name="CV Solve PnP (Pose)",
			category=CATEGORY,
			description="Estimates the pose (rotation + translation) of a known set "
			            "of 3D object points seen at given 2D image points "
			            "(cv2.solvePnP) - e.g. a chessboard or planar marker for "
			            "augmented reality. Failure-tolerant like 'CV Find "
			            "Homography': fewer than 4 points, a count mismatch or a "
			            "degenerate configuration returns found=false with zero "
			            "rvec/tvec instead of raising. Feed object_points (a planar "
			            "grid from 'CV Grid Points', or any 'CV Points'), the "
			            "matching image_points (cv2.findChessboardCorners / Match "
			            "Features / goodFeaturesToTrack) and a camera matrix ('CV "
			            "Camera Matrix' or 'Calibrate Camera'); the rvec/tvec drive "
			            "cv2.projectPoints / drawFrameAxes to overlay a 3D object. "
			            "Branch on 'found' with a control-flow node to skip the "
			            "overlay when nothing was detected.",
			inputs=[
				NPArray.Input("object_points",
					tooltip="3D points in object space, Nx3 or Nx1x3 (e.g. 'CV Grid "
					        "Points' for a chessboard, or 'CV Points' for a cube)."),
				NPArray.Input("image_points",
					tooltip="The matching 2D image points, Nx2 or Nx1x2, in the same "
					        "order as object_points (e.g. findChessboardCorners)."),
				NPArray.Input("camera_matrix",
					tooltip="3x3 intrinsic matrix K ('CV Camera Matrix' or 'Calibrate "
					        "Camera')."),
				io.Combo.Input("method", options=enums.SOLVEPNP.options,
					default=enums.SOLVEPNP.default,
					tooltip="PnP solver. ITERATIVE (Levenberg-Marquardt) is the "
					        "general default; IPPE/IPPE_SQUARE are for planar targets."),
				NPArray.Input("dist_coeffs", optional=True,
					tooltip="Lens distortion coefficients (k1, k2, p1, p2[, k3...]); "
					        "leave unconnected for an ideal pinhole camera (no "
					        "distortion)."),
				# optional (a schema SUFFIX) rather than required on purpose: the
				# node shipped with 'method' as its only widget, so six saved
				# workflows store a single widget value. An optional widget keeps
				# those aligned and defaults to the old least-squares behavior.
				io.Combo.Input("estimation",
					options=["all points (least squares)", "RANSAC (reject outliers)"],
					default="all points (least squares)", optional=True,
					tooltip="'all points' fits every correspondence - correct for a "
					        "chessboard or marker, where they are all good. 'RANSAC' "
					        "tolerates wrong correspondences and is what feature "
					        "matches or stereo-derived 3D points need; it also fills "
					        "the 'inliers' output."),
				io.Float.Input("reprojection_error", default=3.0, min=0.1, max=100.0,
					step=0.5, optional=True, advanced=True,
					tooltip="RANSAC only: how many pixels a point may miss its "
					        "reprojection by and still count as an inlier."),
				io.Int.Input("ransac_iterations", default=1000, min=1, max=100000,
					optional=True, advanced=True,
					tooltip="RANSAC only: maximum sampling iterations."),
				io.Float.Input("confidence", default=0.999, min=0.0, max=1.0,
					step=0.001, optional=True, advanced=True,
					tooltip="RANSAC only: probability that the returned pose comes "
					        "from an all-inlier sample."),
			],
			outputs=[
				NPArray.Output(display_name="rvec",
					tooltip="3x1 Rodrigues rotation vector (zeros if not found). Pass "
					        "to cv2.projectPoints / drawFrameAxes / Rodrigues / "
					        "'CV Pose To Matrix'."),
				NPArray.Output(display_name="tvec",
					tooltip="3x1 translation vector (zeros if not found)."),
				io.Boolean.Output(display_name="found"),
				NPArray.Output(display_name="inliers",
					tooltip="Nx1 uint8 mask of the correspondences RANSAC kept (all "
					        "255 for the 'all points' estimation, all 0 if not "
					        "found). Filter the matches with 'CV Filter Points "
					        "By Mask' / 'CV Filter Points 3D By Mask'."),
				io.Int.Output(display_name="inlier_count"),
				io.Float.Output(display_name="reproj_error",
					tooltip="Mean reprojection error in pixels over the inliers - "
					        "the quality number for this pose (under ~1-3 px is a "
					        "good fit). 0 when not found."),
			],
		)

	@classmethod
	def execute(cls, object_points, image_points, camera_matrix, method,
	            estimation="all points (least squares)", dist_coeffs=None,
	            reprojection_error=3.0, ransac_iterations=1000,
	            confidence=0.999) -> io.NodeOutput:
		zero = np.zeros((3, 1), np.float64)

		def _not_found(n=0):
			return (zero, zero, False, np.zeros((n, 1), np.uint8), 0, 0.0)

		# cv2.findChessboardCorners & co return None when nothing was detected -
		# the 'no pose' result must flow through, not crash the reshape
		if object_points is None or image_points is None:
			return io.NodeOutput(*_not_found())
		obj = np.asarray(object_points, np.float64)
		img = np.asarray(image_points, np.float64)
		if obj.size == 0 or img.size == 0:
			return io.NodeOutput(*_not_found())
		obj = obj.reshape(-1, 1, 3)
		img = img.reshape(-1, 1, 2)
		n = len(obj)
		# <4 correspondences or a count mismatch (a wiring slip) cannot be solved -
		# fail soft instead of halting the workflow
		if n < 4 or n != len(img):
			return io.NodeOutput(*_not_found(n))
		K = np.asarray(camera_matrix, np.float64)
		dist = None if dist_coeffs is None else np.asarray(dist_coeffs, np.float64)
		flags = enums.SOLVEPNP.resolve(method)
		try:
			if str(estimation).startswith("RANSAC"):
				ok, rvec, tvec, inl = cv2.solvePnPRansac(
					obj, img, K, dist, flags=flags,
					reprojectionError=float(reprojection_error),
					iterationsCount=int(ransac_iterations),
					confidence=float(confidence))
				keep = np.zeros(n, bool)
				if inl is not None:
					keep[np.asarray(inl).ravel().astype(np.intp)] = True
			else:
				ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flags)
				keep = np.ones(n, bool)
		except cv2.error:
			return io.NodeOutput(*_not_found(n))
		if not ok or not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
			return io.NodeOutput(*_not_found(n))
		if not keep.any():  # RANSAC converged on nothing usable
			return io.NodeOutput(*_not_found(n))

		proj, _ = cv2.projectPoints(obj[keep], rvec, tvec, K, dist)
		err = float(np.linalg.norm(
			proj.reshape(-1, 2) - img[keep].reshape(-1, 2), axis=1).mean())
		return io.NodeOutput(np.asarray(rvec, np.float64),
		                     np.asarray(tvec, np.float64), True,
		                     (keep.astype(np.uint8) * 255).reshape(-1, 1),
		                     int(keep.sum()), err)


class PoseToMatrix(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PoseToMatrix",
			display_name="CV Pose To Matrix",
			category=CATEGORY,
			description="Assembles the rvec + tvec of 'CV Solve PnP' / "
			            "'CV ArUco Board Pose' into the transform matrix that "
			            "moves POINTS between the two frames (cv2.Rodrigues + a "
			            "concatenation cv2 has no function for). The 3x4 [R|t] "
			            "feeds 'CV Transform Points 3D': forward maps a point from "
			            "the pose's object/world frame into the camera frame, "
			            "inverse maps a camera-frame point (e.g. a stereo point "
			            "cloud) back into the object/world frame. Failure "
			            "tolerant: a missing or non-finite pose gives the identity "
			            "transform, which leaves points where they are.",
			inputs=[
				NPArray.Input("rvec",
					tooltip="3x1 Rodrigues rotation vector (or a 3x3 rotation "
					        "matrix, which is passed through)."),
				NPArray.Input("tvec",
					tooltip="3x1 translation vector."),
			],
			outputs=[
				NPArray.Output(display_name="matrix_3x4",
					tooltip="[R|t] as a 3x4 float64 matrix."),
				NPArray.Output(display_name="matrix_4x4",
					tooltip="The same transform as a 4x4 homogeneous matrix "
					        "(what 'CV ICP Register' and most 3D tools use)."),
				NPArray.Output(display_name="rotation",
					tooltip="The 3x3 rotation matrix R on its own."),
			],
		)

	@classmethod
	def execute(cls, rvec, tvec) -> io.NodeOutput:
		R = np.eye(3)
		t = np.zeros((3, 1))
		try:
			if rvec is not None and np.asarray(rvec).size:
				r = np.asarray(rvec, np.float64)
				R = r.reshape(3, 3) if r.size == 9 else cv2.Rodrigues(r.reshape(3, 1))[0]
			if tvec is not None and np.asarray(tvec).size:
				t = np.asarray(tvec, np.float64).reshape(3, 1)
		except (cv2.error, ValueError) as e:
			logging.warning("PoseToMatrix: unusable pose (%s) - returning identity", e)
			R, t = np.eye(3), np.zeros((3, 1))
		if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
			R, t = np.eye(3), np.zeros((3, 1))
		m34 = np.hstack([R, t])
		m44 = np.vstack([m34, [0.0, 0.0, 0.0, 1.0]])
		return io.NodeOutput(m34, m44, R)


class MatrixToPose(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MatrixToPose",
			display_name="CV Matrix To Pose",
			category=CATEGORY,
			description="Takes a 4x4 (or 3x4) rigid transform apart into the "
			            "rvec + tvec that cv2.projectPoints, 'CV Rasterize Mesh' "
			            "and 'CV Camera Pose To 3D View' ask for - the inverse of "
			            "'CV Pose To Matrix'. This is what turns the pose a "
			            "3D matcher produces ('CV PPF Pose Estimation', 'CV ICP "
			            "Register', 'CV Register Point Clouds (3D)') back into "
			            "something that can be RENDERED, so a recovered pose can "
			            "be checked against, or composited into, the picture it "
			            "came from. The rotation block is orthonormalized through "
			            "an SVD first, so a matrix that picked up drift from a "
			            "chain of multiplications still yields a valid rotation. "
			            "Failure tolerant: a missing, wrongly-shaped or "
			            "non-finite matrix gives a zero pose (identity rotation "
			            "at the origin) with found=false.",
			inputs=[
				NPArray.Input("matrix",
					tooltip="4x4 homogeneous or 3x4 [R|t] transform. A 3x3 "
					        "rotation on its own is accepted too and gives a zero "
					        "translation."),
			],
			outputs=[
				NPArray.Output(display_name="rvec",
					tooltip="3x1 Rodrigues rotation vector."),
				NPArray.Output(display_name="tvec",
					tooltip="3x1 translation vector, in the matrix's own units."),
				NPArray.Output(display_name="rotation",
					tooltip="The orthonormalized 3x3 rotation matrix R."),
				io.Float.Output(display_name="scale",
					tooltip="Mean singular value of the input's rotation block - "
					        "1.0 for a rigid transform. Anything else means the "
					        "matrix carries a scale (or drift) that rvec/tvec "
					        "CANNOT represent and that this node has divided out."),
				io.Boolean.Output(display_name="found",
					tooltip="False when the matrix was missing, the wrong shape, "
					        "or non-finite."),
			],
		)

	@classmethod
	def execute(cls, matrix) -> io.NodeOutput:
		zero = (np.zeros((3, 1)), np.zeros((3, 1)), np.eye(3), 0.0, False)
		if matrix is None:
			return io.NodeOutput(*zero)
		m = np.asarray(matrix, np.float64)
		if m.ndim > 2:
			m = m.reshape(m.shape[-2], m.shape[-1])
		if m.shape not in ((4, 4), (3, 4), (3, 3)) or not np.all(np.isfinite(m)):
			logging.warning("MatrixToPose: unusable matrix (shape %s) - "
			                "returning a zero pose", str(m.shape))
			return io.NodeOutput(*zero)

		R = m[:3, :3]
		t = m[:3, 3:4] if m.shape[1] == 4 else np.zeros((3, 1))
		# a pose that came out of a chain of matrix multiplications is only
		# approximately orthonormal, and cv2.Rodrigues does not mind - it just
		# returns a vector that no longer round-trips.  Nearest rotation by SVD.
		try:
			u, s, vt = np.linalg.svd(R)
		except np.linalg.LinAlgError as e:
			logging.warning("MatrixToPose: %s - returning a zero pose", e)
			return io.NodeOutput(*zero)
		Rn = u @ vt
		if np.linalg.det(Rn) < 0:  # a reflection is not a rotation
			u[:, -1] *= -1.0
			Rn = u @ vt
		rvec = cv2.Rodrigues(np.ascontiguousarray(Rn))[0]
		return io.NodeOutput(rvec, np.ascontiguousarray(t), Rn,
		                     float(np.mean(s)), True)


# --- ArUco fiducial markers ------------------------------------------------
# cv2.aruco constants live on the cv2.aruco submodule, NOT on cv2, so the
# EnumGroup machinery (which probes hasattr(cv2, name)) cannot see them - the
# dictionary is a plain named Combo whose option strings resolve via getattr on
# the submodule (still a readable dropdown, never a raw int).
def _aruco_dict_names() -> list:
	a = getattr(cv2, "aruco", None)
	if a is None:
		return ["<cv2.aruco not available in this build>"]
	# curated common families first, then any remaining DICT_* the build exposes
	common = [
		"DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
		"DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250",
		"DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
		"DICT_7X7_50", "DICT_7X7_250",
		"DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11",
	]
	names = [n for n in common if hasattr(a, n)]
	for n in sorted(dir(a)):
		if n.startswith("DICT_") and n not in names and isinstance(getattr(a, n), int):
			names.append(n)
	return names or ["<no ArUco dictionaries in this build>"]


_ARUCO_DICTS = _aruco_dict_names()
_ARUCO_DEFAULT_DICT = "DICT_6X6_250" if "DICT_6X6_250" in _ARUCO_DICTS else _ARUCO_DICTS[0]


def _aruco_detector(dictionary: str):
	a = cv2.aruco
	if not hasattr(a, dictionary):
		raise ValueError(f'Unknown ArUco dictionary "{dictionary}" (cv2 {cv2.__version__})')
	d = a.getPredefinedDictionary(getattr(a, dictionary))
	return a.ArucoDetector(d, a.DetectorParameters())


class ArucoDetectMarkers(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ArucoDetectMarkers",
			display_name="CV ArUco Detect Markers",
			category=CATEGORY,
			description="Detects ArUco fiducial markers (cv2.aruco.ArucoDetector). "
			            "Pick the dictionary the markers were printed from - it must "
			            "match, or nothing is found. Outputs the four image corners "
			            "of every marker and their integer ids. Finding no markers "
			            "is a valid result (count = 0, empty arrays with the right "
			            "dtype), never an error. Feed 'corners' "
			            "into 'CV ArUco Draw Markers' to visualize, or into "
			            "'CV ArUco Board Pose' to recover the board's 3D pose "
			            "for an augmented-reality overlay.",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Image containing the markers (BGR or gray)."),
				io.Combo.Input("dictionary", options=_ARUCO_DICTS,
					default=_ARUCO_DEFAULT_DICT,
					tooltip="ArUco dictionary the markers belong to. The NxN in the "
					        "name is the marker's internal bit grid; the trailing "
					        "number is the dictionary size. Must match the printed "
					        "markers."),
			],
			outputs=[
				NPArray.Output(display_name="corners",
					tooltip="Marker corners, (N, 4, 2) float32 in clockwise order "
					        "starting top-left. Empty (0, 4, 2) when none found."),
				NPArray.Output(display_name="ids",
					tooltip="Marker ids, (N,) int32. Empty (0,) when none found."),
				io.Int.Output(display_name="count",
					tooltip="Number of markers detected."),
			],
		)

	@classmethod
	def execute(cls, image, dictionary) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		gray = _to_gray_u8(image)
		corners, ids, _ = _aruco_detector(dictionary).detectMarkers(gray)
		if ids is None or len(corners) == 0:  # nothing found: empty, right dtype
			return io.NodeOutput(np.zeros((0, 4, 2), np.float32),
			                     np.zeros((0,), np.int32), 0)
		packed = np.stack([np.asarray(c, np.float32).reshape(4, 2) for c in corners])
		return io.NodeOutput(packed, np.asarray(ids, np.int32).reshape(-1), len(corners))


class ArucoDrawMarkers(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGE_ONLY)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ArucoDrawMarkers",
			display_name="CV ArUco Draw Markers",
			category=CATEGORY,
			description="Draws detected ArUco markers - green outlines, a red "
			            "top-left corner and (optionally) the id label "
			            "(cv2.aruco.drawDetectedMarkers). Visualization only: pair "
			            "it with 'CV ArUco Detect Markers'. An "
			            "empty 'corners' array passes the image through unchanged.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="Image to draw on (a copy is made)."),
				NPArray.Input("corners",
					tooltip="Marker corners from 'CV ArUco Detect Markers', "
					        "(N, 4, 2)."),
				NPArray.Input("ids", optional=True,
					tooltip="Marker ids, (N,); connect to print id labels."),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "image",
				tooltip="Same format as the image input.")],
		)

	@classmethod
	def execute(cls, image, corners, ids=None) -> io.NodeOutput:
		image, tag = polytype.resolve(image)
		canvas = _to_bgr_u8(image).copy()
		arr = np.asarray(corners, np.float32)
		if arr.size == 0:  # nothing to draw: pass through
			return io.NodeOutput(polytype.restore(canvas, tag))
		corner_list = [c.reshape(1, 4, 2) for c in arr.reshape(-1, 4, 2)]
		id_arg = None
		if ids is not None and np.asarray(ids).size:
			id_arg = np.asarray(ids, np.int32).reshape(-1, 1)
		cv2.aruco.drawDetectedMarkers(canvas, corner_list, id_arg)
		return io.NodeOutput(polytype.restore(canvas, tag))


class ArucoBoardPose(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ArucoBoardPose",
			display_name="CV ArUco Board Pose (Average)",
			category=CATEGORY,
			description="Estimates the pose (rotation + translation) of a planar "
			            "ArUco board by solving each marker's pose independently "
			            "(cv2.solvePnP, IPPE_SQUARE - exact for a coplanar square) "
			            "and AVERAGING them: the rotation matrices are meaned then "
			            "re-orthonormalized via SVD (a valid rotation), the "
			            "translations meaned to the board centroid. Because every "
			            "marker lies in the same plane its individual pose already "
			            "carries the board orientation; averaging cancels per-marker "
			            "detection noise. The two-fold planar-pose ambiguity is "
			            "resolved per marker by keeping the solution whose normal "
			            "faces the camera, so a stray flipped marker cannot corrupt "
			            "the mean. Feed the rvec/tvec into cv2.projectPoints + 'Draw "
			            "Polygon' to overlay a 3D object (see the ArUco cube "
			            "workflow). Zero markers returns found=false with a zero "
			            "pose instead of erroring - branch on 'found' with a "
			            "control-flow node.",
			inputs=[
				NPArray.Input("corners",
					tooltip="Marker corners from 'CV ArUco Detect Markers', "
					        "(N, 4, 2)."),
				NPArray.Input("camera_matrix",
					tooltip="3x3 intrinsic matrix K ('CV Camera Matrix' or "
					        "'Calibrate Camera')."),
				io.Float.Input("marker_length", default=1.0, min=1e-4, max=1e6,
					tooltip="Physical edge length of one marker; it sets the world "
					        "unit. The returned tvec and any object you project are "
					        "in these units (e.g. length 1 -> a cube of size 3 spans "
					        "three markers)."),
				NPArray.Input("dist_coeffs", optional=True,
					tooltip="Lens distortion coefficients; leave unconnected for an "
					        "ideal pinhole camera."),
			],
			outputs=[
				NPArray.Output(display_name="rvec",
					tooltip="3x1 Rodrigues rotation of the board (zeros if none "
					        "found). Pass to cv2.projectPoints / drawFrameAxes."),
				NPArray.Output(display_name="tvec",
					tooltip="3x1 translation to the board centroid (zeros if none "
					        "found)."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, corners, camera_matrix, marker_length, dist_coeffs=None) -> io.NodeOutput:
		not_found = (np.zeros((3, 1), np.float64), np.zeros((3, 1), np.float64), False)
		if corners is None:
			return io.NodeOutput(*not_found)
		arr = np.asarray(corners, np.float32)
		if arr.size == 0:
			return io.NodeOutput(*not_found)
		arr = arr.reshape(-1, 4, 2)
		s = float(marker_length)
		# canonical marker corners in its own frame (z=0), matching detectMarkers'
		# clockwise-from-top-left order; +z points out of the board toward the camera
		obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0],
		                [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], np.float32)
		K = np.asarray(camera_matrix, np.float64)
		dist = None if dist_coeffs is None else np.asarray(dist_coeffs, np.float64)
		rot_sum = np.zeros((3, 3))
		t_list = []
		n = 0
		for quad in arr:
			try:
				# IPPE_SQUARE returns BOTH planar-pose solutions plus their
				# reprojection errors; solvePnPGeneric orders them best-first.
				ret, rvecs, tvecs, errs = cv2.solvePnPGeneric(
					obj, quad, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
			except cv2.error:
				continue
			if not ret or len(rvecs) == 0:
				continue
			# resolve the two-fold ambiguity by the data: keep the solution that
			# reprojects the marker best, so a flipped fit can't skew the mean
			errs = (np.asarray(errs, np.float64).ravel() if errs is not None
			        else np.zeros(len(rvecs)))
			best = None
			for idx in np.argsort(errs):
				R, _ = cv2.Rodrigues(rvecs[idx])
				tvec = np.asarray(tvecs[idx], np.float64).reshape(3, 1)
				if np.all(np.isfinite(R)) and np.all(np.isfinite(tvec)):
					best = (R, tvec)
					break
			if best is None:
				continue
			rot_sum += best[0]
			t_list.append(best[1])
			n += 1
		if n == 0:
			return io.NodeOutput(*not_found)
		# nearest valid rotation to the mean (orthogonal Procrustes via SVD)
		U, _, Vt = np.linalg.svd(rot_sum / n)
		R_avg = U @ Vt
		if np.linalg.det(R_avg) < 0:  # reflection -> flip the least-significant axis
			U[:, -1] *= -1
			R_avg = U @ Vt
		rvec_avg, _ = cv2.Rodrigues(R_avg)
		tvec_avg = np.mean(t_list, axis=0)
		if not (np.all(np.isfinite(rvec_avg)) and np.all(np.isfinite(tvec_avg))):
			return io.NodeOutput(*not_found)
		return io.NodeOutput(np.asarray(rvec_avg, np.float64).reshape(3, 1),
		                     np.asarray(tvec_avg, np.float64).reshape(3, 1), True)


# --- ArUco boards: GridBoard / CharucoBoard --------------------------------
# cv2.aruco.GridBoard, CharucoBoard and CharucoDetector are classes, so
# generator.py cannot see them. One geometry convention is used across all
# three nodes below: cell_size is the board PITCH (one chessboard square for
# ChArUco, marker + separation for a GridBoard) and marker_ratio is the marker
# edge as a fraction of it - so the two board families are described by the
# same two numbers.

_BOARD_GRID = "GridBoard (markers only)"
_BOARD_CHARUCO = "ChArUco (chessboard + markers)"


def _aruco_dictionary(name):
	a = getattr(cv2, "aruco", None)
	if a is None or not hasattr(a, name):
		raise ValueError(
			f'Unknown ArUco dictionary "{name}" (cv2 {cv2.__version__}) - this '
			"build may lack the aruco module.")
	return a.getPredefinedDictionary(getattr(a, name))


def _make_board(board_type, dictionary, cols, rows, cell_size, marker_ratio,
                legacy_pattern=False):
	a = cv2.aruco
	d = _aruco_dictionary(dictionary)
	square = float(cell_size)
	# both board types need a strictly positive gap around each marker
	ratio = float(min(max(marker_ratio, 0.05), 0.95))
	marker = square * ratio
	size = (int(cols), int(rows))
	if board_type == _BOARD_GRID:
		return a.GridBoard(size, marker, square - marker, d), marker, square
	board = a.CharucoBoard(size, square, marker, d)
	if legacy_pattern and hasattr(board, "setLegacyPattern"):
		# boards printed with OpenCV < 4.6 start with a WHITE square; without
		# this the ids come out shifted and calibration silently degrades
		board.setLegacyPattern(True)
	return board, marker, square


class ArucoBoardImage(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ArucoBoardImage",
			display_name="CV ArUco Board Image",
			category=CATEGORY,
			description="Renders a printable ArUco GridBoard or ChArUco board "
			            "(cv2.aruco.GridBoard / CharucoBoard generateImage) - classes "
			            "the raw wrappers cannot reach. A GridBoard is a grid of "
			            "markers with gaps; a ChArUco board interleaves markers into "
			            "a chessboard, which is the best of both worlds for "
			            "calibration: the markers say WHICH corner is which (so a "
			            "partially visible or occluded board still works, unlike a "
			            "plain chessboard) while the chessboard corners themselves "
			            "localise to sub-pixel accuracy (better than marker corners). "
			            "Print this, photograph it from ~15 angles and feed the batch "
			            "to 'CV Calibrate Camera (ChArUco)'. The marker/square "
			            "lengths come out as numbers so the pose nodes downstream get "
			            "the world scale right.",
			inputs=[
				io.Combo.Input("board_type", options=[_BOARD_CHARUCO, _BOARD_GRID],
					default=_BOARD_CHARUCO,
					tooltip="ChArUco for calibration and robust pose (markers inside "
					        "a chessboard). GridBoard when you only want markers, "
					        "e.g. an AR target on a non-flat-lit surface."),
				io.Combo.Input("dictionary", options=_ARUCO_DICTS,
					default=_ARUCO_DEFAULT_DICT,
					tooltip="Marker family to draw from. It must match the "
					        "dictionary chosen in the detector node. A board needs "
					        "cols*rows/2 ids for ChArUco (cols*rows for a GridBoard), "
					        "so pick a dictionary big enough."),
				io.Int.Input("cols", default=5, min=2, max=50,
					tooltip="Number of cells across (chessboard squares for ChArUco, "
					        "markers for a GridBoard)."),
				io.Int.Input("rows", default=7, min=2, max=50,
					tooltip="Number of cells down. Use an ODD x EVEN board (like 5x7) "
					        "so it has no 180-degree rotational symmetry."),
				io.Int.Input("pixels_per_cell", default=100, min=20, max=1000,
					tooltip="Rendering resolution: pixels per cell. The image size "
					        "is derived from the board's own geometry, so the "
					        "aspect ratio is always exact (a mismatched size is what "
					        "makes cv2's generateImage throw)."),
				io.Float.Input("cell_size", default=0.04, min=1e-4, max=1e6, step=0.005,
					tooltip="Physical size of one cell in your world unit (e.g. "
					        "0.04 = 40 mm). It sets the scale of every pose measured "
					        "against this board; measure the PRINTED board and put "
					        "the real value here."),
				io.Float.Input("marker_ratio", default=0.7, min=0.05, max=0.95, step=0.05,
					tooltip="Marker edge as a fraction of the cell. For ChArUco the "
					        "rest is the white margin inside the black square (0.7 is "
					        "the usual); for a GridBoard it is the separation between "
					        "markers."),
				io.Int.Input("margin", default=20, min=0, max=500,
					tooltip="White quiet-zone border in pixels. Markers touching the "
					        "paper edge are often missed, so keep some.",
					optional=True, advanced=True),
				io.Boolean.Input("legacy_pattern", default=False,
					tooltip="ChArUco only: match boards generated by OpenCV BEFORE "
					        "4.6, whose top-left cell is WHITE. Turn it on only for "
					        "an old printed board - otherwise the corner ids shift.",
					optional=True, advanced=True),
			],
			outputs=[
				io.Image.Output(display_name="board",
					tooltip="The rendered board as an IMAGE - preview it, or save it "
					        "with the core 'Save Image' node and print it at a known "
					        "size."),
				io.Float.Output(display_name="marker_length",
					tooltip="Physical marker edge (cell_size * marker_ratio) - feed "
					        "'CV ArUco Board Pose'."),
				io.Float.Output(display_name="square_length",
					tooltip="Physical cell pitch - feed the ChArUco calibration node "
					        "so the intrinsics come out in your unit."),
				io.Int.Output(display_name="marker_count",
					tooltip="How many markers the board carries (ChArUco boards use "
					        "half the cells)."),
			],
		)

	@classmethod
	def execute(cls, board_type, dictionary, cols, rows, pixels_per_cell, cell_size,
	            marker_ratio, margin=20, legacy_pattern=False) -> io.NodeOutput:
		board, marker, square = _make_board(
			board_type, dictionary, cols, rows, cell_size, marker_ratio, legacy_pattern)
		# derive the pixel size from the board's OWN extent: generateImage
		# asserts if the requested size does not match the board aspect ratio
		w_m, h_m = board.getRightBottomCorner()[:2]
		scale = float(pixels_per_cell) / float(square)
		size = (max(1, int(round(w_m * scale))), max(1, int(round(h_m * scale))))
		img = board.generateImage(size, marginSize=int(margin))
		bgr = cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_GRAY2BGR)
		n_markers = len(np.asarray(board.getIds()).reshape(-1))
		return io.NodeOutput(imageutils.bgr_to_image_batch([bgr]),
		                     float(marker), float(square), int(n_markers))


class CharucoDetect(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CharucoDetect",
			display_name="CV ChArUco Detect",
			category=CATEGORY,
			description="Finds the chessboard corners of a ChArUco board and says "
			            "WHICH corner each one is (cv2.aruco.CharucoDetector) - the "
			            "class API generator.py cannot see. Compared with "
			            "cv2.findChessboardCorners this survives partial views, "
			            "occlusion and clipping at the image edge, because the "
			            "embedded markers identify the corners instead of relying on "
			            "the whole grid being visible. The 'object_points' output "
			            "pairs each detected corner with its 3-D position on the "
			            "board, which is exactly what cv2.solvePnP wants - wire "
			            "object_points + corners + a camera matrix into 'CV Solve "
			            "PnP Pose' for the board pose, or use 'CV Calibrate Camera "
			            "(ChArUco)' over a batch. The board parameters MUST match the "
			            "printed board. Zero corners is a valid result (found=false), "
			            "never an error.",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Photo of the ChArUco board (BGR or gray)."),
				io.Combo.Input("dictionary", options=_ARUCO_DICTS,
					default=_ARUCO_DEFAULT_DICT,
					tooltip="Dictionary the board's markers were drawn from - must "
					        "match 'CV ArUco Board Image' or nothing is found."),
				io.Int.Input("cols", default=5, min=2, max=50,
					tooltip="Chessboard squares across (the same number given to the "
					        "board image node, NOT the inner-corner count)."),
				io.Int.Input("rows", default=7, min=2, max=50,
					tooltip="Chessboard squares down."),
				io.Float.Input("cell_size", default=0.04, min=1e-4, max=1e6, step=0.005,
					tooltip="Physical square size of the PRINTED board; it sets the "
					        "unit of object_points and therefore of any pose."),
				io.Float.Input("marker_ratio", default=0.7, min=0.05, max=0.95, step=0.05,
					tooltip="Marker edge as a fraction of the square - the same value "
					        "used to generate the board."),
				io.Boolean.Input("legacy_pattern", default=False,
					tooltip="Set for boards generated by OpenCV before 4.6 (white "
					        "top-left cell). Wrong setting = shifted corner ids.",
					optional=True, advanced=True),
			],
			outputs=[
				io.Boolean.Output(display_name="found",
					tooltip="True when at least one ChArUco corner was identified - "
					        "branch on it with 'Basic data handling: IfElse'."),
				NPArray.Output(display_name="corners",
					tooltip="(N, 2) float32 sub-pixel image positions of the "
					        "identified chessboard corners - feed 'CV Draw "
					        "Points', or 'CV Solve PnP Pose' with object_points."),
				NPArray.Output(display_name="ids",
					tooltip="(N,) int32 which corner of the board each one is "
					        "(row-major). This identification is what a plain "
					        "chessboard cannot give you."),
				NPArray.Output(display_name="object_points",
					tooltip="(N, 3) float32 position of each detected corner ON THE "
					        "BOARD, in cell_size units (z = 0). Row-aligned with "
					        "'corners' - the 3-D half of the correspondence."),
				NPArray.Output(display_name="marker_corners",
					tooltip="(M, 4, 2) float32 corners of the ArUco markers that were "
					        "detected on the way - feed 'CV ArUco Draw Markers'."),
				NPArray.Output(display_name="marker_ids",
					tooltip="(M,) int32 ids of those markers."),
				io.Int.Output(display_name="count",
					tooltip="Number of identified chessboard corners; 0 is valid, not "
					        "an error."),
			],
		)

	@classmethod
	def _detect(cls, board, gray):
		"""(corners Nx2, ids N, marker_corners Mx4x2, marker_ids M) - all empty
		when the board is not visible. CharucoDetector returns None for the
		outputs it could not fill, which is data, not a failure."""
		detector = cv2.aruco.CharucoDetector(board)
		try:
			ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(gray)
		except cv2.error:
			ch_corners = ch_ids = m_corners = m_ids = None
		corners = (np.zeros((0, 2), np.float32) if ch_corners is None
		           else np.asarray(ch_corners, np.float32).reshape(-1, 2))
		ids = (np.zeros((0,), np.int32) if ch_ids is None
		       else np.asarray(ch_ids, np.int32).reshape(-1))
		mc = (np.zeros((0, 4, 2), np.float32) if m_corners is None or not len(m_corners)
		      else np.asarray(m_corners, np.float32).reshape(-1, 4, 2))
		mi = (np.zeros((0,), np.int32) if m_ids is None or not np.size(m_ids)
		      else np.asarray(m_ids, np.int32).reshape(-1))
		return corners, ids, mc, mi

	@classmethod
	def execute(cls, image, dictionary, cols, rows, cell_size, marker_ratio,
	            legacy_pattern=False) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		gray = _to_gray_u8(image)
		board, _, _ = _make_board(_BOARD_CHARUCO, dictionary, cols, rows, cell_size,
		                          marker_ratio, legacy_pattern)
		corners, ids, mc, mi = cls._detect(board, gray)
		if not len(corners):
			return io.NodeOutput(False, corners, ids, np.zeros((0, 3), np.float32),
			                     mc, mi, 0)
		obj, img_pts = board.matchImagePoints(corners.reshape(-1, 1, 2),
		                                      ids.reshape(-1, 1))
		return io.NodeOutput(
			True, np.asarray(img_pts, np.float32).reshape(-1, 2), ids,
			np.asarray(obj, np.float32).reshape(-1, 3), mc, mi, len(corners))


class CalibrateCameraCharuco(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CalibrateCameraCharuco",
			display_name="CV Calibrate Camera (ChArUco)",
			category=CATEGORY,
			description="Camera intrinsics + lens distortion from a batch of ChArUco "
			            "board photos - the robust alternative to 'CV Calibrate "
			            "Camera (Chessboard)'. A plain chessboard view is thrown away "
			            "unless EVERY inner corner is visible; a ChArUco view "
			            "contributes whatever corners it can identify, so photos "
			            "where the board runs off the frame or is partly occluded "
			            "still count. That means fewer, easier shots for the same "
			            "coverage - and coverage of the image CORNERS, which is "
			            "exactly where distortion is estimated, is what a full-board "
			            "requirement makes hardest. Views with too few corners are "
			            "skipped. Failure-tolerant: fewer than 3 usable views returns "
			            "found=false with an identity matrix and zero distortion.",
			inputs=[
				io.Image.Input("images",
					tooltip="Batch of board photos from DIFFERENT angles and "
					        "distances (>= 3 usable; ~15 gives a good fit). Tilt the "
					        "board - views that are all fronto-parallel cannot "
					        "separate focal length from distance."),
				io.Combo.Input("dictionary", options=_ARUCO_DICTS,
					default=_ARUCO_DEFAULT_DICT,
					tooltip="Dictionary the board's markers were drawn from."),
				io.Int.Input("cols", default=5, min=2, max=50,
					tooltip="Chessboard squares across (as given to the board image "
					        "node)."),
				io.Int.Input("rows", default=7, min=2, max=50,
					tooltip="Chessboard squares down."),
				io.Float.Input("cell_size", default=0.04, min=1e-4, max=1e6, step=0.005,
					tooltip="Physical square size of the PRINTED board; it sets the "
					        "world unit of the translation vectors."),
				io.Float.Input("marker_ratio", default=0.7, min=0.05, max=0.95, step=0.05,
					tooltip="Marker edge as a fraction of the square - the value used "
					        "to generate the board."),
				io.Int.Input("min_corners", default=6, min=4, max=1000,
					tooltip="Skip a view that identifies fewer corners than this. "
					        "Four is the theoretical minimum for a planar view; 6-10 "
					        "keeps the fit stable.", optional=True),
				io.Boolean.Input("legacy_pattern", default=False,
					tooltip="Set for boards generated by OpenCV before 4.6 (white "
					        "top-left cell).", optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="camera_matrix",
					tooltip="3x3 intrinsic matrix K (fx, fy, cx, cy)."),
				NPArray.Output(display_name="dist_coeffs",
					tooltip="Distortion coefficients (k1, k2, p1, p2, k3)."),
				io.Float.Output(display_name="rms_error",
					tooltip="Mean reprojection error in pixels; lower is better "
					        "(under ~1 is good)."),
				io.Int.Output(display_name="views_used",
					tooltip="How many input views contributed corners - compare it "
					        "with the batch size to see how many were skipped."),
				io.Int.Output(display_name="corners_used",
					tooltip="Total identified corners across all used views; this is "
					        "the number that really drives the fit."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, images, dictionary, cols, rows, cell_size, marker_ratio,
	            min_corners=6, legacy_pattern=False) -> io.NodeOutput:
		board, _, _ = _make_board(_BOARD_CHARUCO, dictionary, cols, rows, cell_size,
		                          marker_ratio, legacy_pattern)
		obj_points, img_points, size = [], [], None
		for frame in imageutils.image_batch_to_bgr(images):
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			size = (gray.shape[1], gray.shape[0])  # (w, h)
			corners, ids, _, _ = CharucoDetect._detect(board, gray)
			if len(corners) < int(min_corners):
				continue
			obj, img_pts = board.matchImagePoints(corners.reshape(-1, 1, 2),
			                                      ids.reshape(-1, 1))
			obj_points.append(np.asarray(obj, np.float32).reshape(-1, 1, 3))
			img_points.append(np.asarray(img_pts, np.float32).reshape(-1, 1, 2))
		total = int(sum(len(p) for p in img_points))
		if len(img_points) < 3 or size is None:
			return io.NodeOutput(np.eye(3, dtype=np.float64),
			                     np.zeros((1, 5), np.float64), 0.0,
			                     len(img_points), total, False)
		try:
			rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
		except cv2.error:
			# a degenerate set (e.g. every view fronto-parallel and identical)
			# is bad DATA, not a wiring mistake - report it as not-found
			return io.NodeOutput(np.eye(3, dtype=np.float64),
			                     np.zeros((1, 5), np.float64), 0.0,
			                     len(img_points), total, False)
		return io.NodeOutput(np.asarray(K, np.float64), np.asarray(dist, np.float64),
		                     float(rms), len(img_points), total, True)


class RecoverPose(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_RecoverPose",
			display_name="CV Recover Pose (Essential Matrix)",
			category=CATEGORY,
			description="Recovers the relative pose (rotation + translation "
			            "DIRECTION) between two CALIBRATED views of a static scene "
			            "from matched points and the camera matrix - the core step "
			            "of two-view structure-from-motion and monocular visual "
			            "odometry. It estimates the essential matrix "
			            "(cv2.findEssentialMat, RANSAC) and decomposes it with the "
			            "cheirality check (cv2.recoverPose), keeping the rotation/"
			            "translation for which the inliers lie in FRONT of both "
			            "cameras. Translation is a UNIT vector: monocular geometry "
			            "fixes direction, not absolute scale. Same failure-tolerant "
			            "contract as 'CV Find Fundamental Matrix': fewer than 5 "
			            "points, a degenerate configuration or too few inliers "
			            "returns found=false, an identity rotation and a zero "
			            "translation instead of halting - branch on 'found' with a "
			            "control-flow node. Note: a pure rotation (no camera "
			            "translation) or a single plane is degenerate; use views "
			            "with depth and a translated camera. Feed rotation + "
			            "translation into 'CV Triangulate Points (Two-View)'.",
			inputs=[
				NPArray.Input("points_a", tooltip="Points in view 1, Nx1x2 (from 'CV Match Features')."),
				NPArray.Input("points_b", tooltip="Corresponding points in view 2, Nx1x2."),
				NPArray.Input("camera_matrix",
					tooltip="3x3 intrinsic matrix K shared by both views ('CV Camera "
					        "Matrix' or 'Calibrate Camera')."),
				io.Combo.Input("method", options=enums.RANSAC_LMEDS.options,
					default=enums.RANSAC_LMEDS.default,
					tooltip="Robust estimator for the essential matrix. RANSAC is "
					        "the standard choice for feature matches."),
				io.Float.Input("threshold", default=1.0, min=0.1, max=100.0, step=0.1,
					tooltip="Max distance in pixels from a point to its epipolar "
					        "line to count as an inlier (RANSAC)."),
				io.Float.Input("confidence", default=0.999, min=0.5, max=1.0, step=0.001,
					tooltip="Desired probability that the estimate is correct "
					        "(RANSAC/LMedS)."),
				io.Int.Input("min_inliers", default=15, min=5, max=10000,
					tooltip="Minimum cheirality-consistent inliers for 'found' to "
					        "be true. Raise it to reject accidental fits."),
			],
			outputs=[
				NPArray.Output(display_name="rotation",
					tooltip="3x3 float64 rotation R mapping view-1 directions into "
					        "view 2 (identity if not found)."),
				NPArray.Output(display_name="translation",
					tooltip="3x1 float64 UNIT translation (camera-2 origin in view-1 "
					        "frame, up to scale; zero if not found)."),
				NPArray.Output(display_name="inlier_mask",
					tooltip="Nx1 uint8: 1 = point used in the recovered pose. Feed "
					        "into 'CV Filter Points By Mask' / 'Draw Matches'."),
				io.Int.Output(display_name="inlier_count"),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, points_a, points_b, camera_matrix, method, threshold,
	            confidence, min_inliers) -> io.NodeOutput:
		pts_a = np.asarray(points_a, np.float64).reshape(-1, 1, 2)
		pts_b = np.asarray(points_b, np.float64).reshape(-1, 1, 2)
		if len(pts_a) != len(pts_b):
			raise ValueError(
				f"points_a has {len(pts_a)} points but points_b has {len(pts_b)} - "
				"both must come from the same 'CV Match Features' node."
			)
		K = np.asarray(camera_matrix, np.float64).reshape(3, 3)
		return io.NodeOutput(*_recover_pose(
			pts_a, pts_b, K, enums.RANSAC_LMEDS.resolve(method),
			threshold, confidence, min_inliers))


class TriangulatePoints(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_TriangulatePoints",
			display_name="CV Triangulate Points (Two-View)",
			category=CATEGORY,
			description="Reconstructs sparse 3D points from two views given the "
			            "matched 2D points, the shared camera matrix and the "
			            "relative pose (cv2.triangulatePoints) - the sparse "
			            "structure half of two-view structure-from-motion. It "
			            "builds the projection matrices P1 = K[I|0] (view 1) and "
			            "P2 = K[R|t] (view 2, from 'CV Recover Pose'), "
			            "triangulates each correspondence and divides out the "
			            "homogeneous coordinate. Points are in VIEW-1 camera "
			            "coordinates and only up to the same global scale as the "
			            "unit translation. The 'valid' mask flags points that pass "
			            "the cheirality check (positive depth in BOTH cameras) and "
			            "are finite - outlier matches and points near infinity are "
			            "marked 0; preview points_3d with 'Inspect CV Data' and "
			            "check reproj_error (mean pixels over valid points; under "
			            "~1-2 px is a clean reconstruction). Failure-tolerant: 0 "
			            "points or a degenerate pose returns empty arrays, error 0 "
			            "and count 0 instead of raising.",
			inputs=[
				NPArray.Input("points_a", tooltip="Points in view 1, Nx1x2 (matched, same order as points_b)."),
				NPArray.Input("points_b", tooltip="Corresponding points in view 2, Nx1x2."),
				NPArray.Input("camera_matrix",
					tooltip="3x3 intrinsic matrix K shared by both views."),
				NPArray.Input("rotation",
					tooltip="3x3 relative rotation R (from 'CV Recover Pose')."),
				NPArray.Input("translation",
					tooltip="3x1 relative translation t (from 'CV Recover Pose')."),
			],
			outputs=[
				NPArray.Output(display_name="points_3d",
					tooltip="Nx3 float64 points in view-1 camera coordinates "
					        "(invalid points are zeroed - see 'valid')."),
				NPArray.Output(display_name="valid",
					tooltip="Nx1 uint8: 1 = finite and in front of both cameras."),
				io.Float.Output(display_name="reproj_error",
					tooltip="Mean reprojection error in pixels over valid points "
					        "(0 if none are valid)."),
				io.Int.Output(display_name="count",
					tooltip="Number of valid 3D points."),
			],
		)

	@classmethod
	def execute(cls, points_a, points_b, camera_matrix, rotation, translation) -> io.NodeOutput:
		empty = (np.zeros((0, 3), np.float64), np.zeros((0, 1), np.uint8), 0.0, 0)
		if points_a is None or points_b is None:
			return io.NodeOutput(*empty)
		pa = np.asarray(points_a, np.float64).reshape(-1, 2)
		pb = np.asarray(points_b, np.float64).reshape(-1, 2)
		if len(pa) != len(pb):
			raise ValueError(
				f"points_a has {len(pa)} points but points_b has {len(pb)} - "
				"both must come from the same matched point set."
			)
		n = len(pa)
		if n == 0:
			return io.NodeOutput(*empty)
		K = np.asarray(camera_matrix, np.float64).reshape(3, 3)
		R = np.asarray(rotation, np.float64).reshape(3, 3)
		t = np.asarray(translation, np.float64).reshape(3, 1)
		P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
		P2 = K @ np.hstack([R, t])
		try:
			pts4d = cv2.triangulatePoints(P1, P2, pa.T.copy(), pb.T.copy())
		except cv2.error:
			return io.NodeOutput(np.zeros((n, 3), np.float64),
			                     np.zeros((n, 1), np.uint8), 0.0, 0)
		w = pts4d[3]
		safe_w = np.where(np.abs(w) < 1e-12, 1e-12, w)
		X = (pts4d[:3] / safe_w).T  # Nx3 in view-1 camera coordinates
		finite = np.all(np.isfinite(X), axis=1)
		X[~finite] = 0.0
		z1 = X[:, 2]
		z2 = (X @ R.T + t.reshape(1, 3))[:, 2]  # depth in camera 2
		valid = finite & (z1 > 0) & (z2 > 0)

		def deproject(P):
			proj = (np.hstack([X, np.ones((n, 1))]) @ P.T)
			d = proj[:, 2:3]
			return proj[:, :2] / np.where(np.abs(d) < 1e-12, 1e-12, d)

		err = (np.linalg.norm(deproject(P1) - pa, axis=1)
		       + np.linalg.norm(deproject(P2) - pb, axis=1)) / 2.0
		mean_err = float(err[valid].mean()) if valid.any() else 0.0
		return io.NodeOutput(np.asarray(X, np.float64),
		                     valid.astype(np.uint8).reshape(-1, 1),
		                     mean_err, int(valid.sum()))


class RegisterPointClouds3D(io.ComfyNode):
	_METHODS = ["RANSAC + rigid refit", "RANSAC (affine)", "rigid (all points)"]

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_RegisterPointClouds3D",
			display_name="CV Register Point Clouds (3D)",
			category=CATEGORY,
			description="Finds the transform that maps one 3D point set onto "
			            "another given CORRESPONDING points "
			            "(cv2.estimateAffine3D) - the 3D counterpart of 'OpenCV "
			            "Find Homography (RANSAC)'. Typical use: two stereo pairs "
			            "of the same scene. Match features between the two left "
			            "views ('CV Match Features'), read the 3D position of "
			            "each match out of the two XYZ maps ('CV Sample Array At "
			            "Points' on cv2.reprojectImageTo3D's output), and this "
			            "node returns the 3x4 [R|t] that brings the second cloud "
			            "into the first one's frame - apply it with 'CV Transform "
			            "Points 3D' and merge with 'CV Merge Point Clouds'. "
			            "'RANSAC + rigid refit' (default) rejects the outliers "
			            "that noisy stereo depth always produces, then re-fits a "
			            "clean rigid transform on the inliers only. Failure "
			            "tolerant: fewer than 3 correspondences, a count mismatch "
			            "or a degenerate configuration returns the identity "
			            "transform with found=false instead of raising, so a "
			            "failed registration merges the clouds unaligned rather "
			            "than halting the workflow - branch on 'found' with a "
			            "'Basic data handling: IfElse' if you would rather skip "
			            "the second cloud entirely.",
			inputs=[
				NPArray.Input("source",
					tooltip="Nx3 points to be MOVED (the second view's cloud)."),
				NPArray.Input("target",
					tooltip="Nx3 points to move ONTO, one per source point in "
					        "the same order (the reference view's cloud)."),
				io.Combo.Input("method", options=cls._METHODS,
					default=cls._METHODS[0],
					tooltip="'RANSAC + rigid refit': robust outlier rejection, "
					        "then a rotation+translation(+uniform scale) fit on "
					        "the inliers - the right choice for stereo clouds. "
					        "'RANSAC (affine)': keeps the general affine fit, "
					        "which can absorb depth errors as shear. 'rigid (all "
					        "points)': no outlier rejection (only for clean, "
					        "already-filtered correspondences)."),
				io.Float.Input("ransac_threshold", default=0.3, min=0.0, max=1e6,
					step=0.05,
					tooltip="Inlier distance in the cloud's own unit (metres for "
					        "a calibrated stereo reconstruction). Stereo depth "
					        "noise grows with distance, so a few tenths of a "
					        "metre is realistic; too tight finds no inliers, too "
					        "loose accepts mismatches."),
				io.Float.Input("confidence", default=0.999, min=0.0, max=1.0,
					step=0.001,
					tooltip="RANSAC confidence: probability that the returned fit "
					        "comes from an all-inlier sample."),
			],
			outputs=[
				NPArray.Output(display_name="matrix",
					tooltip="3x4 [R|t] mapping source -> target (identity when "
					        "found=false). Feed 'CV Transform Points 3D'."),
				NPArray.Output(display_name="inliers",
					tooltip="Nx1 uint8 mask of the correspondences RANSAC kept "
					        "(all zero when found=false; all 255 for the "
					        "'rigid (all points)' method)."),
				io.Int.Output(display_name="inlier_count"),
				io.Float.Output(display_name="scale",
					tooltip="Uniform scale of the rigid fit (1.0 for a pure "
					        "rigid motion; far from 1 means the two clouds "
					        "disagree about depth, e.g. a wrong baseline)."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, source, target, method, ransac_threshold, confidence) -> io.NodeOutput:
		src = np.zeros((0, 3)) if source is None else np.asarray(source, np.float64).reshape(-1, 3)
		dst = np.zeros((0, 3)) if target is None else np.asarray(target, np.float64).reshape(-1, 3)
		identity = np.hstack([np.eye(3), np.zeros((3, 1))])
		n = len(src)
		fallback = (identity, np.zeros((n, 1), np.uint8), 0, 1.0, False)
		if len(src) != len(dst):
			raise ValueError(
				f"source has {len(src)} points but target has {len(dst)} - both "
				"must come from the same matched correspondences."
			)
		# too few correspondences is a normal outcome of a hard image pair:
		# report it instead of raising
		if n < 3:
			return io.NodeOutput(*fallback)

		scale = 1.0
		try:
			if str(method) == "rigid (all points)":
				M, scale = cv2.estimateAffine3D(src, dst, force_rotation=True)
				inl = np.ones(n, bool)
			else:
				retval, M, inliers = cv2.estimateAffine3D(
					src, dst, ransacThreshold=float(ransac_threshold),
					confidence=float(confidence))
				if not retval or M is None:
					return io.NodeOutput(*fallback)
				inl = np.asarray(inliers).ravel().astype(bool)
				if str(method) == "RANSAC + rigid refit" and int(inl.sum()) >= 3:
					# re-fit WITHOUT the outliers: the RANSAC model comes from a
					# minimal sample, the refit uses every inlier
					M, scale = cv2.estimateAffine3D(src[inl], dst[inl],
					                                force_rotation=True)
		except cv2.error as e:
			logging.warning("RegisterPointClouds3D: %s", e)
			return io.NodeOutput(*fallback)

		M = np.asarray(M, np.float64)
		if M.shape != (3, 4) or not np.all(np.isfinite(M)):
			return io.NodeOutput(*fallback)
		return io.NodeOutput(M, (inl.astype(np.uint8) * 255).reshape(-1, 1),
		                     int(inl.sum()), float(scale), True)


def _klt_track(gray_a, gray_b, p0, max_features, quality, min_distance,
               win_size, max_level, fb_threshold, mask=None):
	"""One KLT step: detect corners in gray_a (only when p0 is empty), track
	them into gray_b with pyramidal Lucas-Kanade, keep the pairs that pass the
	forward-backward gate. Returns (points_a, points_b), Nx1x2 float32.

	`p0` exists so a sequence can REUSE the previous step's surviving tracks
	instead of re-detecting every frame - that is what makes incremental visual
	odometry cheap, and it is the one thing the per-pair node cannot express.
	Finding nothing is a valid result (two empty arrays), never an exception.
	"""
	empty = np.zeros((0, 1, 2), np.float32)
	if p0 is None or len(p0) == 0:
		# goodFeaturesToTrack returns None when the frame has no corners
		p0 = cv2.goodFeaturesToTrack(gray_a, max_features, float(quality),
		                             float(min_distance), mask=mask)
		if p0 is None or len(p0) == 0:
			return empty, empty.copy()
	p0 = np.asarray(p0, np.float32).reshape(-1, 1, 2)
	win = (win_size | 1, win_size | 1)
	p1, st, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, p0, None,
	                                     winSize=win, maxLevel=max_level)
	if p1 is None or st is None:
		return empty, empty.copy()
	keep = st.ravel().astype(bool)
	if fb_threshold > 0 and keep.any():
		p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, p1, None,
		                                       winSize=win, maxLevel=max_level)
		fb_err = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
		keep = keep & stb.ravel().astype(bool) & (fb_err < fb_threshold)
	return (p0[keep].reshape(-1, 1, 2).astype(np.float32),
	        p1[keep].reshape(-1, 1, 2).astype(np.float32))


def _recover_pose(pts_a, pts_b, K, method, threshold, confidence, min_inliers):
	"""findEssentialMat + recoverPose, with the pack's failure-tolerant
	contract: (R, t, inlier_mask, inlier_count, found). A degenerate pair
	yields an identity rotation, a zero translation and found=False.
	`method` is an already-resolved cv2 flag, not a label."""
	eye = np.eye(3, dtype=np.float64)
	zero_t = np.zeros((3, 1), np.float64)
	not_found = (eye.copy(), zero_t.copy(),
	             np.zeros((len(pts_a), 1), np.uint8), 0, False)
	if len(pts_a) < 5:  # the 5-point algorithm needs at least 5 correspondences
		return not_found
	try:
		E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=method,
		                               prob=confidence, threshold=threshold)
	except cv2.error:
		return not_found
	if E is None or E.shape[0] < 3 or not np.all(np.isfinite(E)):
		return not_found
	E = np.asarray(E, np.float64)[:3]  # degenerate inputs can stack candidates
	try:
		_, R, t, mask = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)
	except cv2.error:
		return not_found
	if R is None or t is None or not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
		return not_found
	mask = (np.ones((len(pts_a), 1), np.uint8) if mask is None
	        else np.asarray(mask, np.uint8).reshape(-1, 1))
	inliers = int(mask.sum())
	if inliers < min_inliers:
		return eye.copy(), zero_t.copy(), mask, inliers, False
	return (np.asarray(R, np.float64), np.asarray(t, np.float64).reshape(3, 1),
	        mask, inliers, True)


def _compose_step(R_wc, centre, R_rel, t_rel, scale):
	"""Dead-reckon one relative pose onto the running camera-to-world pose.

	'Recover Pose' returns [R|t] mapping the earlier camera frame to the later
	one, so the later camera's centre sits at -R^T t in the earlier frame:
	R_wc <- R_wc R^T and centre <- centre - scale * R_wc R^T t.
	"""
	R_wc = np.asarray(R_wc, np.float64).reshape(3, 3)
	R_rel = np.asarray(R_rel, np.float64).reshape(3, 3)
	t_rel = np.asarray(t_rel, np.float64).reshape(3)
	R_new = R_wc @ R_rel.T
	return R_new, np.asarray(centre, np.float64).reshape(3) - float(scale) * (R_new @ t_rel)


class TrackFeaturesKLT(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_TrackFeaturesKLT",
			display_name="CV Track Features (KLT)",
			category=CATEGORY,
			description="Tracks corner features from frame_a into frame_b with "
			            "pyramidal Lucas-Kanade optical flow (cv2.goodFeaturesToTrack "
			            "+ cv2.calcOpticalFlowPyrLK) - the sparse tracker behind "
			            "monocular visual odometry. Both frames are grayscaled, "
			            "corners are detected in frame_a and tracked into frame_b, "
			            "and only confidently-tracked pairs are kept: the optical-flow "
			            "status flag must be set AND (forward-backward check) "
			            "re-tracking the point back into frame_a must land within "
			            "fb_threshold pixels of where it started. Outputs the matched "
			            "points in both frames (Nx1x2, same order), ready for 'OpenCV "
			            "Recover Pose'. Finding zero trackable corners is a valid "
			            "result (count = 0, empty arrays), not an error - so a VO "
			            "pipeline never halts on a textureless or blank frame.",
			inputs=[
				polytype.poly_input("frame_a", types=polytype.IMAGE_ONLY,
					tooltip="Earlier frame; corners are detected here."),
				polytype.poly_input("frame_b", types=polytype.IMAGE_ONLY,
					tooltip="Later frame; resized to frame_a if sizes differ."),
				io.Int.Input("max_features", default=500, min=0, max=10000,
					tooltip="Maximum corners to detect in frame_a (goodFeaturesToTrack); 0 = return all detected corners."),
				io.Float.Input("quality", default=0.01, min=1e-4, max=1.0, step=0.005,
					tooltip="Minimum corner quality relative to the strongest corner "
					        "(qualityLevel); lower keeps weaker corners."),
				io.Float.Input("min_distance", default=7.0, min=1.0, max=100.0,
					tooltip="Minimum spacing in pixels between detected corners."),
				io.Int.Input("win_size", default=21, min=3, max=101,
					tooltip="Lucas-Kanade search window (forced odd); larger tolerates "
					        "bigger motion but blurs fine detail."),
				io.Int.Input("max_level", default=3, min=0, max=8,
					tooltip="Pyramid levels (0 = no pyramid); more levels track larger "
					        "displacements."),
				io.Float.Input("fb_threshold", default=1.0, min=0.0, max=30.0, step=0.5,
					tooltip="Forward-backward consistency gate in pixels: a track is "
					        "dropped if re-tracking it back misses its origin by more "
					        "than this. 0 disables the check.",
					optional=True, advanced=True),
				polytype.poly_input("mask", types=polytype.MASK_ONLY, optional=True,
					tooltip="Optional mask: detect corners only where mask > 0."),
			],
			outputs=[
				NPArray.Output(display_name="points_a",
					tooltip="Tracked corners in frame_a, Nx1x2 float32."),
				NPArray.Output(display_name="points_b",
					tooltip="Their tracked positions in frame_b, Nx1x2 float32 (same order)."),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, frame_a, frame_b, max_features, quality, min_distance,
	            win_size, max_level, fb_threshold=1.0, mask=None) -> io.NodeOutput:
		frame_a, _ = polytype.resolve(frame_a)
		frame_b, _ = polytype.resolve(frame_b)
		mask, _ = polytype.resolve(mask)
		ga, gb = _to_gray_u8(frame_a), _to_gray_u8(frame_b)
		if gb.shape != ga.shape:
			gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
		m = None
		if mask is not None:
			m = imageutils.fit_mask_to(imageutils.ensure_uint8(np.asarray(mask)), ga.shape[:2])
		pa, pb = _klt_track(ga, gb, None, max_features, quality, min_distance,
		                    win_size, max_level, fb_threshold, m)
		return io.NodeOutput(pa, pb, len(pa))


class FindTransformECC(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FindTransformECC",
			display_name="CV Find Transform (ECC)",
			category=CATEGORY,
			description="Direct (intensity-based) image registration: estimates "
			            "the geometric transform that aligns 'image' onto "
			            "'template' by maximizing the ECC correlation "
			            "(cv2.findTransformECC) - no keypoints needed, ideal for "
			            "small shifts/rotations between frames of a burst or a "
			            "golden template. Color inputs are converted to grayscale "
			            "internally. Apply the result with cv2.warpAffine (or "
			            "warpPerspective for MOTION_HOMOGRAPHY), dsize = the "
			            "template's size, flags = 'INTER_LINEAR | WARP_INVERSE_MAP' "
			            "to pull the image into the template's frame. Robust by "
			            "design: raw cv2 RAISES when the iteration does not "
			            "converge (e.g. featureless or unrelated images) - this "
			            "node instead returns found=false with an identity matrix "
			            "that warps to a no-op, so the workflow keeps running; "
			            "branch on 'found' with a control-flow node (e.g. Basic "
			            "data handling's 'if/else').",
			inputs=[
				polytype.poly_input("template", types=polytype.IMAGE_ONLY,
					tooltip="Reference image the other one is aligned TO."),
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Image to align onto the template."),
				io.Combo.Input("motion_type", options=enums.MOTION.options,
					default="MOTION_EUCLIDEAN",
					tooltip="Transform model to estimate: TRANSLATION (2 dof), "
					        "EUCLIDEAN (shift + rotation, 3 dof), AFFINE (6 dof) "
					        "or HOMOGRAPHY (8 dof, 3x3). Fewer degrees of freedom "
					        "converge more reliably."),
				io.Int.Input("iterations", default=100, min=1, max=5000,
					tooltip="Maximum ECC refinement iterations."),
				io.Float.Input("epsilon", default=1e-5, min=0.0, max=1.0, step=1e-6,
					tooltip="Stop when the correlation improves by less than this "
					        "between iterations (smaller = more precise, slower)."),
				io.Int.Input("gauss_filt_size", default=5, min=1, max=101,
					tooltip="Odd size of the Gaussian blur applied internally "
					        "before matching - smooths noise so the optimization "
					        "does not chase it. 1 = no smoothing."),
				io.Float.Input("min_correlation", default=0.0, min=-1.0, max=1.0, step=0.01,
					tooltip="Minimum final ECC correlation for 'found' to be true "
					        "(1 = perfect). 0 accepts any converged estimate; "
					        "raise it to reject dubious alignments."),
				polytype.poly_input("mask", types=polytype.MASK_ONLY, optional=True,
					tooltip="Optional mask on the IMAGE being aligned: non-zero "
					        "pixels participate in the alignment."),
			],
			outputs=[
				NPArray.Output(display_name="warp_matrix",
					tooltip="2x3 float32 transform (3x3 for MOTION_HOMOGRAPHY) "
					        "mapping template coordinates into the image - use "
					        "WARP_INVERSE_MAP when warping the image back. "
					        "Identity if not found."),
				io.Float.Output(display_name="correlation",
					tooltip="Final ECC correlation coefficient (1 = perfect "
					        "alignment; 0.0 when not found)."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, template, image, motion_type, iterations, epsilon,
	            gauss_filt_size, min_correlation, mask=None) -> io.NodeOutput:
		template, _ = polytype.resolve(template)
		image, _ = polytype.resolve(image)
		mask, _ = polytype.resolve(mask)
		motion = enums.MOTION.resolve(motion_type)
		rows = 3 if motion == cv2.MOTION_HOMOGRAPHY else 2
		identity = np.eye(rows, 3, dtype=np.float32)
		tpl = _to_gray_u8(template).astype(np.float32)
		img = _to_gray_u8(image).astype(np.float32)
		m = None
		if mask is not None:
			m = imageutils.fit_mask_to(imageutils.ensure_uint8(np.asarray(mask)), img.shape[:2])
		criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
		            int(iterations), float(epsilon))
		try:
			correlation, warp = cv2.findTransformECC(
				tpl, img, identity.copy(), motion, criteria, m, int(gauss_filt_size) | 1)
		except cv2.error:
			# no convergence (featureless, unrelated or zero-variance input):
			# a valid outcome, not a wiring mistake
			return io.NodeOutput(identity, 0.0, False)
		if not np.all(np.isfinite(warp)) or not math.isfinite(correlation):
			return io.NodeOutput(identity, 0.0, False)
		if float(correlation) < float(min_correlation):
			return io.NodeOutput(identity, float(correlation), False)
		return io.NodeOutput(warp.astype(np.float32), float(correlation), True)


class PhaseCorrelateTranslation(io.ComfyNode):
	# cv2.phaseCorrelate IS in registry.py, but the raw wrapper is unusable in
	# practice: it needs a companion createHanningWindow at the matching size,
	# it asserts on anything that is not single-channel float32/float64, and its
	# (point2f, float) return lands in lowlevel's special-returns path. This
	# node is the whole recipe in one box.
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PhaseCorrelate",
			display_name="CV Phase Correlate (Translation)",
			category=CATEGORY,
			description="Sub-pixel GLOBAL translation between two images, measured "
			            "in the Fourier domain (cv2.phaseCorrelate) - the fast "
			            "standard for video stabilisation, burst alignment and "
			            "scan-line registration. It is neither feature-based nor "
			            "iterative: one FFT pair, no keypoints to fail on, no "
			            "convergence to babysit, and it is insensitive to uniform "
			            "brightness changes. Use it where 'Find Transform (ECC)' is "
			            "too slow and 'Detect Features' has nothing to lock onto "
			            "(sky, water, blur, plain scans); use those instead when "
			            "the motion is more than a shift. A Hann window is applied "
			            "internally (rectangular edges alone produce a strong false "
			            "peak). Failure-tolerant: mismatched or degenerate inputs "
			            "give dx=dy=0 with found=false, so branch on "
			            "'found' with a control-flow node (core 'If/Else Switch').",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Reference image. Converted to grayscale float32 "
					        "internally."),
				polytype.poly_input("shifted", types=polytype.IMAGE_ONLY,
					tooltip="The moved image. Must be the SAME size as 'image' - "
					        "phase correlation compares spectra, so it cannot "
					        "align frames of different sizes."),
				io.Float.Input("min_response", default=0.0, min=0.0, max=1.0,
					step=0.01,
					tooltip="Minimum peak response for 'found' to be true. The "
					        "response falls off as the overlap shrinks and as the "
					        "motion stops being a pure shift, so this is the "
					        "reject-the-nonsense knob. 0.0 accepts anything."),
				io.Boolean.Input("hann_window", default=True,
					optional=True, advanced=True,
					tooltip="Apply the Hann window before correlating. Leave it on: "
					        "without it the image borders act like a huge edge and "
					        "can dominate the real peak. Turn it off only for "
					        "already-windowed or periodic data."),
			],
			outputs=[
				io.Float.Output(display_name="dx",
					tooltip="Horizontal shift in pixels, sub-pixel accurate: how far "
					        "'shifted' moved RIGHT relative to 'image'. Warp it back "
					        "with a translation matrix (2x3) into cv2.warpAffine."),
				io.Float.Output(display_name="dy",
					tooltip="Vertical shift in pixels (positive = downward)."),
				io.Float.Output(display_name="response",
					tooltip="Peak sharpness in [0,1] - how confident the estimate "
					        "is. 0.0 when not found."),
				NPArray.Output(display_name="translation",
					tooltip="The same shift as a 2x3 float32 affine matrix "
					        "[[1,0,dx],[0,1,dy]], ready for cv2.warpAffine (identity "
					        "when not found, so warping is a no-op)."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, image, shifted, min_response=0.0,
			hann_window=True) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		shifted, _ = polytype.resolve(shifted)
		identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
		a = _to_gray_u8(image).astype(np.float32)
		b = _to_gray_u8(shifted).astype(np.float32)
		if a.shape != b.shape or a.size == 0 or min(a.shape[:2]) < 2:
			logging.warning("PhaseCorrelate: needs two same-size images, got %s "
			                "and %s; reporting found=false.", a.shape, b.shape)
			return io.NodeOutput(0.0, 0.0, 0.0, identity, False)
		window = None
		if hann_window:
			# createHanningWindow takes (width, height), the opposite order to
			# the array shape - swapping it silently windows the wrong axis
			window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
		try:
			(dx, dy), response = cv2.phaseCorrelate(a, b, window)
		except cv2.error as exc:
			logging.warning("PhaseCorrelate: cv2.phaseCorrelate failed (%s); reporting found=false.", exc)
			return io.NodeOutput(0.0, 0.0, 0.0, identity, False)
		if not all(math.isfinite(v) for v in (dx, dy, response)):
			return io.NodeOutput(0.0, 0.0, 0.0, identity, False)
		if float(response) < float(min_response):
			return io.NodeOutput(float(dx), float(dy), float(response), identity, False)
		matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], np.float32)
		return io.NodeOutput(float(dx), float(dy), float(response), matrix, True)


class TrackWindow(io.ComfyNode):
	_MEANSHIFT = "meanShift (fixed-size window)"
	_CAMSHIFT = "CamShift (adapts size and angle)"

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_TrackWindow",
			display_name="CV Track Window",
			category=CATEGORY,
			description="One tracking step of cv2.meanShift / cv2.CamShift: "
			            "slides the window uphill on a density map until it "
			            "sits on the local mass peak. The density is typically "
			            "'CV Back Project' output (histogram tracking), but "
			            "any single-channel likelihood map works (a threshold "
			            "mask, a heatmap...). Chain one node per frame: this "
			            "window output -> the next frame's window input; seed "
			            "the first frame with a 'CV Scalar' literal like "
			            "'(x, y, w, h)'. The raw cv2_meanShift wrapper cannot "
			            "chain (its rect comes out as a text literal) - this "
			            "node keeps the window as a wireable NPARRAY. "
			            "Failure-tolerant: a window off the image, a degenerate "
			            "size or a density with no mass under the result yields "
			            "found=false and echoes the INPUT window unchanged (the "
			            "track holds its last position instead of erroring) - "
			            "branch on 'found' with if/else. Visualize: bbox -> the "
			            "core 'Draw BBoxes' node; box_edges -> 'CV Draw "
			            "Segments' (shows CamShift's rotated box).",
			inputs=[
				polytype.poly_input("density", types=polytype.IMAGE_ONLY,
					tooltip="Density/likelihood map the window climbs on - "
					        "usually an 'CV Back Project' map. Color input "
					        "is converted to grayscale."),
				NPArray.Input("window",
					tooltip="Search window as 4 numbers (x, y, w, h): a 'CV "
					        "Scalar' literal for the first frame, the previous "
					        "'CV Track Window' window output afterwards."),
				io.Combo.Input("method", options=[cls._MEANSHIFT, cls._CAMSHIFT],
					default=cls._MEANSHIFT,
					tooltip="meanShift keeps the window size fixed (steady, "
					        "needs a roughly constant object size); CamShift "
					        "also fits size + orientation (box_edges shows the "
					        "rotated box) but can balloon on weak densities."),
				io.Int.Input("iterations", default=10, min=1, max=1000,
					tooltip="Maximum hill-climbing iterations per step; 10 is "
					        "the classic tracking value."),
				io.Float.Input("epsilon", default=1.0, min=0.0, max=100.0, step=0.1,
					tooltip="Stop when the window center moves less than this "
					        "many pixels between iterations."),
				io.Float.Input("min_density", default=0.0, min=0.0, max=255.0, step=1.0,
					tooltip="Minimum AVERAGE density inside the result window "
					        "for 'found' to be true. 0 accepts any non-zero "
					        "mass; raise it (e.g. 32 on a 0-255 back-projection) "
					        "so a few stray matching pixels do not count as a "
					        "locked track."),
			],
			outputs=[
				NPArray.Output(display_name="window",
					tooltip="Tracked window (x, y, w, h) as int32 - wire into "
					        "the next frame's window input. Echoes the input "
					        "window when found=false."),
				io.BoundingBox.Output(display_name="bbox",
					tooltip="The same window as core BOUNDING_BOX data "
					        "({x, y, width, height}) - chain into the core "
					        "'Draw BBoxes' node."),
				NPArray.Output(display_name="box_edges",
					tooltip="The box outline as 4 segments (4x4 endpoint "
					        "quadruples) for 'CV Draw Segments' - CamShift's "
					        "actual rotated box, meanShift's window rectangle."),
				NPArray.Output(display_name="center",
					tooltip="Box center as a 1x2 float array - collect centers "
					        "to draw the trajectory with 'CV Draw Points' "
					        "or 'CV Draw Segments'."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@staticmethod
	def _corners_to_edges(corners: np.ndarray) -> np.ndarray:
		nxt = np.roll(corners, -1, axis=0)
		return np.concatenate([corners, nxt], axis=1).astype(np.float32)

	@classmethod
	def _fallback(cls, win, found=False) -> io.NodeOutput:
		x, y, w, h = (int(round(v)) for v in win)
		corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
		return io.NodeOutput(
			np.array([x, y, w, h], np.int32),
			{"x": x, "y": y, "width": w, "height": h},
			cls._corners_to_edges(corners),
			np.array([[x + w / 2.0, y + h / 2.0]], np.float32),
			found,
		)

	@classmethod
	def execute(cls, density, window, method, iterations, epsilon,
	            min_density) -> io.NodeOutput:
		density, _ = polytype.resolve(density)
		dens = np.asarray(density)
		if dens.ndim == 3 and dens.shape[2] == 1:
			dens = dens[..., 0]
		elif dens.ndim == 3:
			dens = _to_gray_u8(dens)
		if dens.dtype not in (np.uint8, np.float32):
			dens = dens.astype(np.float32)

		win = np.asarray(window, np.float64).ravel()
		if win.size != 4:
			raise ValueError(
				f"window must hold 4 numbers (x, y, w, h), got {win.size} "
				f"value(s) of shape {np.asarray(window).shape}.")
		fh, fw = dens.shape[:2]
		x, y, w, h = (int(round(v)) for v in win)
		# clamp to the frame; a window fully outside or with no size cannot
		# track - that is data (an object that left the frame), not an error
		cx, cy = max(0, x), max(0, y)
		cw, ch = min(x + w, fw) - cx, min(y + h, fh) - cy
		if cw <= 0 or ch <= 0:
			return cls._fallback(win)

		criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
		            int(iterations), float(epsilon))
		try:
			if method == cls._CAMSHIFT:
				rot, (rx, ry, rw, rh) = cv2.CamShift(dens, (cx, cy, cw, ch), criteria)
			else:
				_, (rx, ry, rw, rh) = cv2.meanShift(dens, (cx, cy, cw, ch), criteria)
				rot = None
		except cv2.error:
			return cls._fallback(win)

		if rw <= 0 or rh <= 0:
			return cls._fallback(win)
		mean = float(dens[ry:ry + rh, rx:rx + rw].mean())
		if not mean > 0.0 or mean < float(min_density):
			return cls._fallback(win)  # lost: hold the last position
		if rot is not None:
			corners = cv2.boxPoints(rot).astype(np.float32)
			center = np.array([list(rot[0])], np.float32)
		else:
			corners = np.array([[rx, ry], [rx + rw, ry], [rx + rw, ry + rh],
			                    [rx, ry + rh]], np.float32)
			center = np.array([[rx + rw / 2.0, ry + rh / 2.0]], np.float32)
		return io.NodeOutput(
			np.array([rx, ry, rw, rh], np.int32),
			{"x": int(rx), "y": int(ry), "width": int(rw), "height": int(rh)},
			cls._corners_to_edges(corners),
			center,
			True,
		)


class ComposePose(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ComposePose",
			display_name="CV Compose Pose (Trajectory)",
			category=CATEGORY,
			description="Dead-reckons a camera trajectory by chaining one relative "
			            "pose onto the running absolute pose - the accumulation step "
			            "of monocular visual odometry. Given the running "
			            "camera-to-world rotation and the trajectory of camera "
			            "centres so far, plus the relative rotation + UNIT "
			            "translation from 'CV Recover Pose' and a step scale, it "
			            "updates the rotation and appends the new camera centre. "
			            "Because monocular geometry only fixes the translation "
			            "DIRECTION, the absolute step length must be supplied as "
			            "'scale' (from wheel odometry, a known baseline, or assumed "
			            "constant). Convention: 'CV Recover Pose' returns [R|t] "
			            "mapping the earlier camera frame to the later one, so the "
			            "world rotation updates as R_wc <- R_wc R^T and the centre as "
			            "centre <- centre - scale * R_wc R^T t. Chain one node per "
			            "frame pair (each node's rotation + trajectory outputs feed "
			            "the next), starting from an identity rotation ('CV Camera "
			            "Matrix' with fx=fy=1, cx=cy=0) and a single origin point "
			            "('CV Points'); gate it on 'found' with if/else so a failed "
			            "pose keeps the previous trajectory. Project the result with "
			            "'CV Project To Plane' and draw it with 'CV Draw "
			            "Polygon'. An empty/None trajectory starts at the origin.",
			inputs=[
				NPArray.Input("rotation",
					tooltip="Running 3x3 camera-to-world rotation (identity for the "
					        "first step)."),
				NPArray.Input("trajectory",
					tooltip="Camera centres so far, Nx3 (last row = current centre); "
					        "start from a single origin point."),
				NPArray.Input("relative_rotation",
					tooltip="3x3 relative rotation R from 'CV Recover Pose'."),
				NPArray.Input("relative_translation",
					tooltip="3x1 UNIT relative translation t from 'CV Recover Pose'."),
				io.Float.Input("scale", default=1.0, min=0.0, max=1e6,
					tooltip="Absolute length of this step (monocular VO cannot recover "
					        "it; use a known baseline or assume constant)."),
			],
			outputs=[
				NPArray.Output(display_name="rotation",
					tooltip="Updated 3x3 camera-to-world rotation."),
				NPArray.Output(display_name="trajectory",
					tooltip="Camera centres with the new one appended, (N+1)x3."),
				NPArray.Output(display_name="position",
					tooltip="The new camera centre, 3x1."),
			],
		)

	@classmethod
	def execute(cls, rotation, trajectory, relative_rotation, relative_translation, scale) -> io.NodeOutput:
		if trajectory is None or np.asarray(trajectory).size == 0:
			traj = np.zeros((1, 3), np.float64)
		else:
			traj = np.asarray(trajectory, np.float64).reshape(-1, 3)
		R_new, C_new = _compose_step(rotation, traj[-1], relative_rotation,
		                             relative_translation, scale)
		traj_out = np.vstack([traj, C_new.reshape(1, 3)])
		return io.NodeOutput(R_new, traj_out, C_new.reshape(3, 1))


class VisualOdometry(io.ComfyNode):
	_SCALE_SOURCES = ("constant",
	                  "per-step lengths (array)",
	                  "reference trajectory (step lengths)")
	_MOTION_GATES = ("none",
	                 "forward-dominant (reject sideways/backward steps)")

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_VisualOdometry",
			display_name="CV Visual Odometry (Sequence)",
			category=CATEGORY,
			description="Monocular visual odometry over a whole frame BATCH: the "
			            "camera's path through a static scene, dead-reckoned from "
			            "the video alone. Per consecutive pair it runs the same "
			            "three steps as 'CV Track Features (KLT)' -> 'OpenCV "
			            "Recover Pose (Essential Matrix)' -> 'CV Compose Pose "
			            "(Trajectory)' (it calls the very same code), carrying the "
			            "surviving tracks into the next frame and re-detecting "
			            "corners only when too few are left. Use the three "
			            "single-step nodes to SEE the pipeline on a couple of "
			            "frames; use this one to RUN it on a video, where a "
			            "per-frame graph loop cannot collect the trajectory (a "
			            "foreach fold keeps one accumulator, and the pose needs "
			            "two). Monocular geometry fixes the translation DIRECTION "
			            "only, never its length - supply the step length through "
			            "'scale_source' (a constant, wheel odometry, or a "
			            "reference track). Failure-tolerant throughout: a pair "
			            "with no recoverable motion (blank frame, no parallax, too "
			            "few inliers) HOLDS the previous pose and marks that frame "
			            "0 in 'found_mask' rather than halting, so the trajectory "
			            "always has exactly one row per input frame. Draw the "
			            "result with 'CV Project To Plane' -> 'CV Fit "
			            "Points To Box' -> 'CV Draw Path (ordered)', and score "
			            "it against a reference track with 'CV Trajectory "
			            "Error (ATE / RPE)'.",
			inputs=[
				io.MultiType.Input("frames", polytype.IMAGE_ONLY,
					tooltip="The video as an IMAGE batch, in order ('Load Video' -> "
					        "'Get Video Components'). Fewer than 2 frames is a valid "
					        "input: the result is a single origin and found=false."),
				NPArray.Input("camera_matrix",
					tooltip="3x3 intrinsic matrix K of the camera that shot the "
					        "sequence ('CV Camera Matrix' or 'Calibrate Camera'). "
					        "An uncalibrated guess bends the whole path - the "
					        "essential matrix cannot be recovered without it."),
				io.Int.Input("max_features", default=2000, min=0, max=20000,
					tooltip="Maximum corners to detect when a re-detection happens "
					        "(goodFeaturesToTrack); 0 = every corner found. Higher "
					        "means a slower but steadier pose."),
				io.Float.Input("quality", default=0.01, min=1e-4, max=1.0, step=0.005,
					tooltip="Minimum corner quality relative to the strongest corner "
					        "(qualityLevel); lower keeps weaker corners."),
				io.Float.Input("min_distance", default=7.0, min=1.0, max=100.0,
					tooltip="Minimum spacing in pixels between detected corners."),
				io.Int.Input("redetect_below", default=1000, min=5, max=20000,
					tooltip="Re-run corner detection once the surviving tracks drop "
					        "under this count. Tracks die as the camera advances, so "
					        "this is what keeps the pose fed; too low and the pose "
					        "degrades before the refill, too high and it re-detects "
					        "every frame (slower, and it throws away long tracks)."),
				io.Int.Input("win_size", default=21, min=3, max=101,
					tooltip="Lucas-Kanade search window (forced odd); larger tolerates "
					        "bigger motion but blurs fine detail."),
				io.Int.Input("max_level", default=3, min=0, max=8,
					tooltip="Pyramid levels (0 = no pyramid); more levels track larger "
					        "displacements - a fast-moving camera needs them."),
				io.Float.Input("fb_threshold", default=1.0, min=0.0, max=30.0, step=0.5,
					tooltip="Forward-backward consistency gate in pixels: a track is "
					        "dropped if re-tracking it back misses its origin by more "
					        "than this. 0 disables the check."),
				io.Combo.Input("method", options=enums.RANSAC_LMEDS.options,
					default=enums.RANSAC_LMEDS.default,
					tooltip="Robust estimator for the essential matrix. RANSAC is the "
					        "standard choice for tracked corners."),
				io.Float.Input("threshold", default=1.0, min=0.1, max=100.0, step=0.1,
					tooltip="Max distance in pixels from a point to its epipolar line "
					        "to count as an inlier (RANSAC)."),
				io.Float.Input("confidence", default=0.999, min=0.5, max=1.0, step=0.001,
					tooltip="Desired probability that the estimate is correct."),
				io.Int.Input("min_inliers", default=15, min=5, max=10000,
					tooltip="Minimum cheirality-consistent inliers for a step to "
					        "count. Below it the step is rejected and the pose held."),
				io.Combo.Input("scale_source", options=list(cls._SCALE_SOURCES),
					default=cls._SCALE_SOURCES[0],
					tooltip="Where each step's LENGTH comes from, since the video "
					        "cannot supply it. 'constant': every step is 'scale' long "
					        "- the path's shape is right, its size is arbitrary. "
					        "'per-step lengths (array)': one length per step in "
					        "'scale_reference'. 'reference trajectory (step lengths)': "
					        "'scale_reference' is an Nx3 track (GPS, wheel odometry, "
					        "ground truth) and its consecutive distances are used - "
					        "this is how a monocular result is made metric."),
				io.Float.Input("scale", default=1.0, min=0.0, max=1e6,
					tooltip="Step length for scale_source = 'constant'."),
				NPArray.Input("scale_reference", optional=True,
					tooltip="The step lengths, as either an (N-1,) / (N,) array of "
					        "lengths or an Nx3 reference trajectory. Shorter than the "
					        "batch: the last value repeats. Ignored when "
					        "scale_source = 'constant'."),
				io.Combo.Input("motion_gate", options=list(cls._MOTION_GATES),
					default=cls._MOTION_GATES[0], optional=True, advanced=True,
					tooltip="Optional sanity check on each recovered direction. "
					        "'forward-dominant' rejects a step whose translation is "
					        "mostly sideways or vertical rather than along the optical "
					        "axis - the classic guard for a forward-facing vehicle "
					        "camera, where such a step is nearly always a bad "
					        "essential-matrix fit. Leave at 'none' for a camera that "
					        "really can move sideways (a drone, a handheld orbit)."),
				io.Float.Input("min_step", default=0.0, min=0.0, max=1e6, step=0.01,
					optional=True, advanced=True,
					tooltip="Reject steps shorter than this (same unit as 'scale'). "
					        "At a standstill the translation direction is pure noise, "
					        "so integrating it only adds drift; 0.1 m is the usual "
					        "choice for a car at 10 fps. 0 accepts every step."),
			],
			outputs=[
				NPArray.Output(display_name="trajectory",
					tooltip="Camera centres, Nx3 float64 - one row per input frame, "
					        "row 0 at the origin. Feed 'CV Project To Plane'."),
				NPArray.Output(display_name="rotations",
					tooltip="Running camera-to-world rotations, Nx3x3 float64 "
					        "(rotations[0] is the identity)."),
				NPArray.Output(display_name="found_mask",
					tooltip="(N,) uint8 - 1 where that frame's pose came from a real "
					        "recovered step, 0 where it was HELD because the step "
					        "failed. Index 0 is 1 (the origin is known by "
					        "definition). Feed it to 'CV Draw Path (ordered)' as "
					        "'gap_mask' so the held stretches are visible."),
				NPArray.Output(display_name="inlier_counts",
					tooltip="(N,) int32 cheirality inliers per step (0 at index 0)."),
				NPArray.Output(display_name="track_counts",
					tooltip="(N,) int32 tracks that survived into each frame - the "
					        "health signal to watch; a collapse here precedes a "
					        "collapse in the pose."),
				io.Int.Output(display_name="redetections",
					tooltip="How many times corners had to be re-detected."),
				io.Boolean.Output(display_name="found",
					tooltip="True if at least one step was recovered. Gate the "
					        "downstream preview on it with 'if/else'."),
			],
		)

	@classmethod
	def _step_lengths(cls, source, scale, reference, steps):
		"""One length per step, whatever form the reference came in."""
		if steps <= 0:
			return np.zeros(0, np.float64)
		if str(source) == cls._SCALE_SOURCES[0] or reference is None:
			return np.full(steps, float(scale), np.float64)
		ref = np.asarray(reference, np.float64)
		if ref.size == 0:
			return np.full(steps, float(scale), np.float64)
		if str(source) == cls._SCALE_SOURCES[2]:
			pts = ref.reshape(-1, 3)
			lengths = (np.linalg.norm(np.diff(pts, axis=0), axis=1)
			           if len(pts) > 1 else np.zeros(0, np.float64))
		else:
			lengths = ref.reshape(-1)
		if len(lengths) == 0:
			return np.full(steps, float(scale), np.float64)
		if len(lengths) < steps:  # short reference: hold its last value
			lengths = np.concatenate(
				[lengths, np.full(steps - len(lengths), lengths[-1])])
		return np.abs(lengths[:steps].astype(np.float64))

	@classmethod
	def execute(cls, frames, camera_matrix, max_features, quality, min_distance,
	            redetect_below, win_size, max_level, fb_threshold, method,
	            threshold, confidence, min_inliers, scale_source, scale,
	            scale_reference=None, motion_gate="none", min_step=0.0) -> io.NodeOutput:
		grays = [_to_gray_u8(f) for f in imageutils.frames_from_input(frames)]
		n = len(grays)
		K = np.asarray(camera_matrix, np.float64).reshape(3, 3)
		traj = np.zeros((max(n, 1), 3), np.float64)
		rots = np.repeat(np.eye(3, dtype=np.float64)[None], max(n, 1), axis=0)
		found_mask = np.zeros(max(n, 1), np.uint8)
		inlier_counts = np.zeros(max(n, 1), np.int32)
		track_counts = np.zeros(max(n, 1), np.int32)
		found_mask[0] = 1  # the origin is known, it was not estimated
		if n < 2:
			return io.NodeOutput(traj, rots, found_mask, inlier_counts,
			                     track_counts, 0, False)

		lengths = cls._step_lengths(scale_source, scale, scale_reference, n - 1)
		flag = enums.RANSAC_LMEDS.resolve(method)
		forward_only = str(motion_gate) == cls._MOTION_GATES[1]
		shape = grays[0].shape[:2]
		tracked, redetections = None, 0
		pbar = comfy.utils.ProgressBar(n - 1)

		for i in range(1, n):
			if grays[i].shape[:2] != shape:
				grays[i] = cv2.resize(grays[i], (shape[1], shape[0]),
				                      interpolation=cv2.INTER_AREA)
			if tracked is None or len(tracked) < redetect_below:
				tracked, redetections = None, redetections + 1
			pa, pb = _klt_track(grays[i - 1], grays[i], tracked, max_features,
			                    quality, min_distance, win_size, max_level,
			                    fb_threshold)
			track_counts[i] = len(pa)
			R_rel, t_rel, _, inliers, ok = _recover_pose(
				np.asarray(pa, np.float64).reshape(-1, 1, 2),
				np.asarray(pb, np.float64).reshape(-1, 1, 2),
				K, flag, threshold, confidence, min_inliers)
			inlier_counts[i] = inliers
			step = float(lengths[i - 1])
			if ok and step <= float(min_step):
				ok = False          # a standstill: the direction is pure noise
			if ok and forward_only:
				tz = abs(float(t_rel[2, 0]))
				ok = tz > abs(float(t_rel[0, 0])) and tz > abs(float(t_rel[1, 0]))
			if ok:
				rots[i], traj[i] = _compose_step(rots[i - 1], traj[i - 1],
				                                 R_rel, t_rel, step)
				found_mask[i] = 1
			else:                   # hold the pose: a failed step is not a halt
				rots[i], traj[i] = rots[i - 1], traj[i - 1]
			tracked = pb if len(pb) else None
			pbar.update(1)

		return io.NodeOutput(traj, rots, found_mask, inlier_counts, track_counts,
		                     redetections, bool(found_mask[1:].any()))


class TrajectoryError(io.ComfyNode):
	_ALIGN = ("none",
	          "rigid (SE3)",
	          "similarity (Sim3, monocular scale-free)")

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_TrajectoryError",
			display_name="CV Trajectory Error (ATE / RPE)",
			category=CATEGORY,
			description="Scores one trajectory against another, index for index - "
			            "the standard way an odometry or SLAM result is compared "
			            "with ground truth. ATE (absolute trajectory error) is the "
			            "distance between corresponding points after the two tracks "
			            "have been brought into one frame; RPE (relative pose "
			            "error) is the disagreement between consecutive STEPS, "
			            "which sees local noise that a globally-fitted ATE hides. "
			            "Alignment matters: two odometry runs start in their own "
			            "coordinate frames, and a monocular run has no scale at "
			            "all, so comparing raw coordinates measures the arbitrary "
			            "choice of frame instead of the estimate. 'rigid (SE3)' "
			            "removes the rotation and offset; 'similarity (Sim3)' also "
			            "removes one global scale factor and is the honest setting "
			            "for a monocular result (report the fitted 'scale' with "
			            "it). Uses the Umeyama fit "
			            "(cv2.estimateAffine3D, force_rotation) - the same one "
			            "behind 'CV Register Point Clouds (3D)'. "
			            "Failure-tolerant: fewer than 3 usable pairs returns "
			            "found=false with zeroed metrics and the estimate passed "
			            "through unaligned.",
			inputs=[
				NPArray.Input("estimate",
					tooltip="The trajectory being scored, Nx3 (from 'CV Visual "
					        "Odometry (Sequence)' or 'CV Compose Pose')."),
				NPArray.Input("reference",
					tooltip="The trajectory to score AGAINST, Nx3, one row per "
					        "estimate row. The longer one is truncated."),
				io.Combo.Input("align", options=list(cls._ALIGN),
					default=cls._ALIGN[2],
					tooltip="How much of the difference to treat as a choice of "
					        "coordinate frame rather than error. 'none' compares raw "
					        "coordinates (only meaningful when both already share a "
					        "frame AND a scale). 'rigid (SE3)' fits rotation + "
					        "translation. 'similarity (Sim3)' also fits one global "
					        "scale - required for monocular odometry, which recovers "
					        "shape but not size."),
				io.Int.Input("rpe_step", default=1, min=1, max=10000,
					tooltip="Frame gap the relative-pose error is measured over. 1 "
					        "compares frame-to-frame motion; a larger gap reports "
					        "drift over that window instead."),
			],
			outputs=[
				NPArray.Output(display_name="aligned",
					tooltip="The estimate mapped into the reference's frame, Nx3 - "
					        "draw this on top of the reference, not the raw input."),
				NPArray.Output(display_name="errors",
					tooltip="(N,) float64 per-point ATE distance. Chart it against "
					        "the frame index to see WHERE the drift happened."),
				io.Float.Output(display_name="ate_rmse",
					tooltip="Root-mean-square ATE, in the reference's unit."),
				io.Float.Output(display_name="ate_mean"),
				io.Float.Output(display_name="ate_max",
					tooltip="Worst single-point error - the drift at its peak."),
				io.Float.Output(display_name="scale",
					tooltip="The global scale the alignment had to apply (1.0 for "
					        "'none' and 'rigid'). Far from 1 with Sim3 means the "
					        "estimate's size is off even though its shape may be "
					        "right."),
				io.Float.Output(display_name="rpe_rmse",
					tooltip="Root-mean-square relative-pose error over 'rpe_step' "
					        "frames, after alignment."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, estimate, reference, align, rpe_step) -> io.NodeOutput:
		est = (np.zeros((0, 3)) if estimate is None
		       else np.asarray(estimate, np.float64).reshape(-1, 3))
		ref = (np.zeros((0, 3)) if reference is None
		       else np.asarray(reference, np.float64).reshape(-1, 3))
		n = min(len(est), len(ref))
		est, ref = est[:n], ref[:n]
		nothing = (est.copy(), np.zeros(n, np.float64),
		           0.0, 0.0, 0.0, 1.0, 0.0, False)
		usable = (np.isfinite(est).all(axis=1) & np.isfinite(ref).all(axis=1)
		          if n else np.zeros(0, bool))
		if n < 3 or int(usable.sum()) < 3:
			return io.NodeOutput(*nothing)

		scale, aligned = 1.0, est.copy()
		if str(align) != cls._ALIGN[0]:
			try:
				# Umeyama: ref ~= scale * R @ est + t, with the scale returned
				# SEPARATELY - it is not baked into the matrix
				M, fitted = cv2.estimateAffine3D(est[usable], ref[usable],
				                                 force_rotation=True)
			except cv2.error as e:
				logging.warning("TrajectoryError: %s", e)
				return io.NodeOutput(*nothing)
			M = np.asarray(M, np.float64)
			if M.shape != (3, 4) or not np.all(np.isfinite(M)):
				return io.NodeOutput(*nothing)
			if str(align) == cls._ALIGN[2]:
				scale = float(fitted)
			aligned = scale * (est @ M[:, :3].T) + M[:, 3]

		errors = np.linalg.norm(aligned - ref, axis=1)
		errors[~usable] = np.nan
		good = errors[usable]
		step = min(int(rpe_step), max(n - 1, 1))
		d_est = aligned[step:] - aligned[:-step]
		d_ref = ref[step:] - ref[:-step]
		rel = np.linalg.norm(d_est - d_ref, axis=1)
		rel = rel[np.isfinite(rel)]
		rpe = float(np.sqrt(np.mean(rel ** 2))) if len(rel) else 0.0
		return io.NodeOutput(aligned, errors,
		                     float(np.sqrt(np.mean(good ** 2))),
		                     float(np.mean(good)), float(np.max(good)),
		                     scale, rpe, True)


_STITCHER_STATUS_LABELS = {
	cv2.Stitcher_OK: "Stitcher_OK",
	cv2.Stitcher_ERR_NEED_MORE_IMGS: "Stitcher_ERR_NEED_MORE_IMGS",
	cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Stitcher_ERR_HOMOGRAPHY_EST_FAIL",
	cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL",
}

def _capped_bounds(lo, hi, ref_lo, ref_hi, cap):
	"""Integer [lo, hi] containing [ref_lo, ref_hi], at most `cap` wide.

	When the projected extent exceeds the cap, the growth on each side of the
	reference interval is scaled down proportionally - the reference frame is
	always fully contained.
	"""
	lo, hi = math.floor(min(lo, ref_lo)), math.ceil(max(hi, ref_hi))
	budget = cap - (ref_hi - ref_lo)
	if budget <= 0:  # the reference alone is at the cap: keep just the reference
		return int(ref_lo), int(ref_hi)
	if hi - lo <= cap:
		return lo, hi
	grow_lo, grow_hi = ref_lo - lo, hi - ref_hi
	scale = budget / (grow_lo + grow_hi)
	return (math.ceil(ref_lo - grow_lo * scale),
	        math.floor(ref_hi + grow_hi * scale))


class StitchCanvas(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_StitchCanvas",
			display_name="CV Stitch Canvas",
			category=CATEGORY,
			description="Computes the output canvas for stitching two images: "
			            "given the homography that maps image_warp into "
			            "image_ref's frame (from 'CV Find Homography'), it "
			            "projects image_warp's corners, takes the union with "
			            "image_ref's rectangle, and outputs the canvas size plus "
			            "the matrices that place each image on that canvas: "
			            "warp_matrix (= translation x homography) for image_warp "
			            "and ref_matrix (a pure translation) for image_ref. Feed "
			            "both into cv2.warpPerspective with the dsize output and "
			            "borderMode=BORDER_CONSTANT (the default BORDER_DEFAULT "
			            "reflects the image into the empty canvas area, which "
			            "breaks the coverage masks), then combine with 'OpenCV "
			            "Feather Blend'. Robust: a "
			            "degenerate homography (non-finite projection) is treated "
			            "as identity, and the canvas is capped at max_size per "
			            "axis - the reference frame always stays fully contained, "
			            "so a bad estimate can never allocate a huge image.",
			inputs=[
				polytype.poly_input("image_warp",
					tooltip="Image that the homography maps into the reference frame "
					        "(only its size is used here)."),
				polytype.poly_input("image_ref",
					tooltip="Reference image; stays axis-aligned on the canvas "
					        "(only its size is used here)."),
				NPArray.Input("homography",
					tooltip="3x3 matrix mapping image_warp coordinates to "
					        "image_ref coordinates."),
				io.Int.Input("max_size", default=8192, min=256, max=32768,
					tooltip="Canvas width/height cap in pixels - protects against "
					        "degenerate homographies that would project corners "
					        "kilometers away."),
			],
			outputs=[
				NPArray.Output(display_name="warp_matrix",
					tooltip="3x3 matrix that places image_warp on the canvas "
					        "(translation x homography)."),
				NPArray.Output(display_name="ref_matrix",
					tooltip="3x3 translation that places image_ref on the canvas."),
				io.String.Output(display_name="dsize",
					tooltip="'(width, height)' canvas size literal, for the "
					        "composite params that are still free-text."),
				io.Int.Output(display_name="width"),
				io.Int.Output(display_name="height"),
				tuples.CvTuple.Output(display_name="size",
					tooltip="The canvas (width, height) as ONE composite value "
					        "- wire it into cv2.warpPerspective's dsize."),
			],
		)

	@classmethod
	def execute(cls, image_warp, image_ref, homography, max_size) -> io.NodeOutput:
		H = np.asarray(homography, np.float64).reshape(3, 3)
		wh, ww = np.asarray(polytype.resolve(image_warp)[0]).shape[:2]
		rh, rw = np.asarray(polytype.resolve(image_ref)[0]).shape[:2]
		corners = np.float64([[0, 0], [ww, 0], [ww, wh], [0, wh]]).reshape(-1, 1, 2)
		projected = cv2.perspectiveTransform(corners, H)
		if projected is None or not np.all(np.isfinite(projected)):
			projected = corners  # degenerate homography: behave like identity
		x0, x1 = _capped_bounds(projected[:, 0, 0].min(), projected[:, 0, 0].max(),
		                        0.0, float(rw), max_size)
		y0, y1 = _capped_bounds(projected[:, 0, 1].min(), projected[:, 0, 1].max(),
		                        0.0, float(rh), max_size)
		width, height = x1 - x0, y1 - y0
		T = np.float64([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
		return io.NodeOutput(T @ H, T, f"({width}, {height})", width, height,
		                     [int(width), int(height)])


_STITCH_SCALAR_INPUTS = (
	io.Combo.Input("mode",
		options=enums.STITCHER_MODE.options,
		default=enums.STITCHER_MODE.default,
		tooltip="Stitching mode: PANORAMA (default, general "
		        "panoramas) or SCANS (scans/linear scenes)."),
	io.Float.Input("registration_resol", default=-1.0, min=-1.0, max=100.0, step=0.01,
		tooltip="Resolution in megapixels for the registration "
		        "stage (feature detection + matching). -1 = opencv "
		        "default (auto-selects based on image count)."),
	io.Float.Input("seam_estimation_resol", default=-1.0, min=-1.0, max=100.0, step=0.01,
		tooltip="Resolution in megapixels for seam estimation. "
		        "-1 = opencv default (auto). Higher = more memory, "
		        "lower speed."),
	io.Float.Input("compositing_resol", default=-1.0, min=-1.0, max=100.0, step=0.01,
		tooltip="Resolution in megapixels for compositing/warping "
		        "phase. -1 = opencv default (auto). Higher = more "
		        "memory, sharper result."),
	io.Float.Input("pano_confidence_thresh", default=1.0, min=0.0, max=10.0, step=0.01,
		tooltip="Confidence threshold for seam estimation. Higher "
		        "values produce fewer, longer seams (simpler blends). "
		        "OpenCV default is 1.0."),
	io.Combo.Input("wave_correction",
		options=["true", "false"],
		default="true",
		tooltip="Enable wave correction (horizontal drift "
		        "compensation). Recommended for panoramas with "
		        "rotational motion."),
	io.Combo.Input("interpolation_flags",
		options=enums.INTER.options,
		default=enums.INTER.default,
		tooltip="Interpolation method used for the compositing "
		        "warp. INTER_LINEAR (default) is fast and "
		        "adequate for most panoramas; INTER_CUBIC or "
		        "INTER_LANCZOS4 are sharper but slower."),
)

_STITCH_OUTPUTS = (
	NPArray.Output(display_name="panorama",
		tooltip="Stitched panorama (uint8 BGR). On failure "
		        "returns a copy of the first input frame."),
	io.String.Output(display_name="status",
		tooltip="'OK' or an error label returned by "
		        "cv2.Stitcher.stitch()."),
	io.Boolean.Output(display_name="success",
		tooltip="True when stitching succeeded (status == OK). "
		        "Use with IfElse to branch on success/failure."),
)


def _build_stitch_config(mode, registration_resol, seam_estimation_resol,
                         compositing_resol, pano_confidence_thresh,
                         wave_correction, interpolation_flags):
	return {
		'mode': str(mode),
		'registrationResol': float(registration_resol),
		'seamEstimationResol': float(seam_estimation_resol),
		'compositingResol': float(compositing_resol),
		'panoConfidenceThresh': float(pano_confidence_thresh),
		'waveCorrection': wave_correction == "true",
		'interpolationFlags': enums.INTER.resolve(interpolation_flags),
	}


class CV_Stitch(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Stitch",
			display_name="CV Stitch",
			category=CATEGORY,
			description="Stitches a batch of images into a panorama "
			            "(cv2.Stitcher). Runs cv2.Stitcher in a separate "
			            "process so stitching can be interrupted mid-workflow. "
			            "Accepts IMAGE, LATENT, or NPARRAY "
			            "inputs via the polytype socket. For batch stitching "
			            "(2+ images), provide the frames as a batched NPARRAY "
			            "from 'Image Batch → CV Batch' (uint8 recommended so "
			            "cv2 receives the correct dtype). A single IMAGE or "
			            "LATENT frame produces no panorama (ERR_NEED_MORE_IMGS). "
			            "Returns the stitched panorama, a status label, and a "
			            "success flag for downstream branching - a failed stitch "
			            "never halts the workflow.",
			inputs=[
			polytype.poly_input("images",
				display_name="image batch",
				tooltip="Image batch to stitch. IMAGE/LATENT: frame 0 only. "
				        "NPARRAY batch [B,H,W,C]: all B frames are stitched "
				        "in order. Use 'Image Batch → CV Batch' to convert "
				        "a ComfyUI IMAGE batch into a batched NPARRAY."),
			polytype.poly_input("masks", types=polytype.MASK_ONLY,
				display_name="mask batch", optional=True,
				tooltip="Optional coverage mask batch (same B dim as the "
				        "image batch, or a single mask shared by all frames)."),
				*_STITCH_SCALAR_INPUTS,
			],
			outputs=list(_STITCH_OUTPUTS),
		)

	@classmethod
	def execute(cls, images, mode, registration_resol, seam_estimation_resol,
	            compositing_resol, pano_confidence_thresh, wave_correction,
	            interpolation_flags, masks=None) -> io.NodeOutput:
		if isinstance(images, torch.Tensor):
			_tag = "image" if images.ndim == 4 else "mask"
		elif isinstance(images, dict) and "samples" in images:
			_tag = "latent"
		else:
			_tag = "nparray"
		img_list = imageutils.frames_from_input(images)

		if len(img_list) < 2:
			logging.warning("CV_Stitch: need at least 2 images, got %d.",
			                len(img_list))
			fallback = (np.ascontiguousarray(img_list[0])
			            if img_list else np.zeros((1, 1, 3), np.uint8))
			return io.NodeOutput(
				polytype.restore(fallback, _tag),
				"Stitcher_ERR_NEED_MORE_IMGS", False)

		mask_list = None
		if masks is not None:
			mask_list = imageutils.masks_from_input(masks)
			if len(mask_list) == 1 and len(img_list) > 1:
				mask_list = mask_list * len(img_list)

		config = _build_stitch_config(
			mode, registration_resol, seam_estimation_resol,
			compositing_resol, pano_confidence_thresh,
			wave_correction, interpolation_flags)
		return _stitch_core(img_list, mask_list, config, _tag)


def _stitch_core(img_list, mask_list, config, tag):
	"""Spawn stitch worker subprocess, poll for result, return NodeOutput."""
	worker_script = workerutils.worker_script_path('stitch_worker.py')
	result_tag, *rest = workerutils.run_subprocess(
		worker_script,
		{'image_list': img_list, 'mask_list': mask_list, 'config': config})
	if result_tag == 'crash':
		logging.warning("CV_Stitch: %s", rest[0] if rest else "worker crashed.")
		return io.NodeOutput(
			polytype.restore(np.ascontiguousarray(img_list[0]), tag),
			"Stitcher_WORKER_CRASHED", False)
	if result_tag == 'error':
		err_msg = rest[0] if rest else "unknown"
		logging.warning("CV_Stitch: worker error: %s", err_msg)
		return io.NodeOutput(
			polytype.restore(np.ascontiguousarray(img_list[0]), tag),
			f"Stitcher_ERROR_{err_msg[:120]}", False)

	status, panorama = int(rest[0]), rest[1]

	if status != cv2.Stitcher_OK:
		logging.warning("CV_Stitch: stitch returned status %d, "
		                "returning first frame as fallback.", status)
		status_label = _STITCHER_STATUS_LABELS.get(
			status, f"unknown ({status})")
		return io.NodeOutput(
			polytype.restore(np.ascontiguousarray(img_list[0]), tag),
			status_label, False)

	return io.NodeOutput(
		polytype.restore(np.asarray(panorama, np.uint8), tag),
		"Stitcher_OK", True)


class CV_Stitch_List(io.ComfyNode):
	INPUT_IS_LIST = True

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Stitch_List",
			display_name="CV Stitch (List)",
			category=CATEGORY,
			description="Stitches a list of images into a panorama "
			            "(cv2.Stitcher). Accepts a list of IMAGE, LATENT, "
			            "or NPARRAY inputs via the polytype socket. "
			            "Only the first frame from each list element is "
			            "used. Runs cv2.Stitcher in a separate process so "
			            "stitching can be interrupted mid-workflow. "
			            "Returns the stitched panorama, a status label, "
			            "and a success flag for downstream branching.",
			inputs=[
			polytype.poly_input("images",
				display_name="image list",
				tooltip="List of images to stitch. Each element is an "
				        "IMAGE / LATENT / NPARRAY; only the first "
				        "frame of each is used."),
			polytype.poly_input("masks", types=polytype.MASK_ONLY,
				display_name="mask list", optional=True,
				tooltip="Optional list of coverage masks (one per "
				        "image in the image list)."),
				*_STITCH_SCALAR_INPUTS,
			],
			outputs=list(_STITCH_OUTPUTS),
		)

	@classmethod
	def execute(cls, images, mode, registration_resol, seam_estimation_resol,
	            compositing_resol, pano_confidence_thresh, wave_correction,
	            interpolation_flags, masks=None) -> io.NodeOutput:
		if not images:
			return io.NodeOutput(np.zeros((1, 1, 3), np.uint8),
				"Stitcher_ERR_NEED_MORE_IMGS", False)

		img_list = []
		for item in images:
			frames = imageutils.frames_from_input(item)
			if frames:
				img_list.append(frames[0])

		if len(img_list) < 2:
			logging.warning("CV_Stitch_List: need at least 2 images, "
			                "got %d.", len(img_list))
			fallback = (np.ascontiguousarray(img_list[0])
			            if img_list else np.zeros((1, 1, 3), np.uint8))
			return io.NodeOutput(fallback,
				"Stitcher_ERR_NEED_MORE_IMGS", False)

		mask_list_flat = None
		if masks is not None:
			mask_list_flat = []
			for item in masks:
				m_mask = imageutils.masks_from_input(item)
				if m_mask:
					mask_list_flat.append(m_mask[0])
			if len(mask_list_flat) == 1 and len(img_list) > 1:
				mask_list_flat = mask_list_flat * len(img_list)

		_unwrap = lambda v: v[0] if isinstance(v, list) else v
		return _stitch_core(img_list, mask_list_flat,
			_build_stitch_config(
				_unwrap(mode),
				_unwrap(registration_resol),
				_unwrap(seam_estimation_resol),
				_unwrap(compositing_resol),
				_unwrap(pano_confidence_thresh),
				_unwrap(wave_correction),
				_unwrap(interpolation_flags)),
			"nparray")


_STITCH_ADVANCED_OUTPUTS = (
	NPArray.Output(display_name="panorama",
		tooltip="Stitched panorama (uint8 BGR). On failure "
		        "returns a copy of the first input frame."),
	io.String.Output(display_name="status",
		tooltip="'OK' or an error label returned by "
		        "the detail pipeline."),
	io.Boolean.Output(display_name="success",
		tooltip="True when stitching succeeded (status == OK). "
		        "Use with IfElse to branch on success/failure."),
)

_STITCH_ADVANCED_FEATURES = ["orb", "sift"]
_STITCH_ADVANCED_MATCHERS = [
	"best_of_2_nearest", "best_of_2_nearest_range", "affine_best_of_2_nearest",
]
_STITCH_ADVANCED_ESTIMATORS = ["homography", "affine"]
_STITCH_ADVANCED_ADJUSTERS = ["ray", "reproj", "affine_partial", "no"]
_STITCH_ADVANCED_WARP_TYPES = [
	"spherical", "plane", "cylindrical", "fisheye", "stereographic",
	"mercator", "transverseMercator",
	"compressedPlaneA2B1", "compressedPlaneA1.5B1",
	"compressedPlanePortraitA2B1", "compressedPlanePortraitA1.5B1",
	"paniniA2B1", "paniniA1.5B1",
	"paniniPortraitA2B1", "paniniPortraitA1.5B1",
]
_STITCH_ADVANCED_WAVE_CORRECT = ["auto", "horiz", "vert", "no"]
_STITCH_ADVANCED_EXPOS_COMP = ["no", "gain", "gain_blocks", "channel", "channel_blocks"]
_STITCH_ADVANCED_SEAM_FINDERS = [
	"no", "voronoi", "gc_color", "gc_colorgrad", "dp_color", "dp_colorgrad",
]
_STITCH_ADVANCED_BLEND_TYPES = ["no", "feather", "multiband"]

_STITCH_ADVANCED_SCALAR_INPUTS = (
	io.Combo.Input("features",
		options=_STITCH_ADVANCED_FEATURES, default="orb",
		tooltip="Feature detector type. ORB is fast and "
		        "universally available. SIFT is more "
		        "distinctive but slower."),
	io.Combo.Input("matcher",
		options=_STITCH_ADVANCED_MATCHERS, default="best_of_2_nearest",
		tooltip="Feature matcher. best_of_2_nearest (default): "
		        "standard pairwise matching. "
		        "best_of_2_nearest_range: limits matching to "
		        "neighbors within range_width. "
		        "affine_best_of_2_nearest: for affine "
		        "transformations (SCANS mode)."),
	io.Float.Input("match_conf", default=-1.0, min=-1.0, max=1.0, step=0.01,
		tooltip="Feature match confidence threshold. -1 = auto "
		        "(0.3 for ORB, 0.65 for SIFT). Higher = "
		        "fewer, more reliable matches."),
	io.Int.Input("range_width", default=-1, min=-1, max=100,
		tooltip="Range width for best_of_2_nearest_range "
		        "matcher. -1 = match against all images. "
		        "Only used when matcher is "
		        "best_of_2_nearest_range."),
	io.Combo.Input("estimator",
		options=_STITCH_ADVANCED_ESTIMATORS, default="homography",
		tooltip="Camera estimator. homography (default): "
		        "perspective transforms between images "
		        "(PANORAMA mode). affine: affine transforms "
		        "(SCANS mode, planar scenes)."),
	io.Combo.Input("bundle_adjuster",
		options=_STITCH_ADVANCED_ADJUSTERS, default="ray",
		tooltip="Bundle adjustment cost function. ray (default): "
		        "ray-based error. reproj: reprojection error. "
		        "affine_partial: partial affine refinement. "
		        "no: skip bundle adjustment."),
	io.Float.Input("pano_confidence_thresh", default=1.0, min=0.0, max=10.0, step=0.01,
		tooltip="Confidence threshold for feature matching. "
		        "Images with confidence below this are "
		        "excluded. Also passed to the bundle "
		        "adjuster."),
	io.Combo.Input("warp_type",
		options=_STITCH_ADVANCED_WARP_TYPES, default="spherical",
		tooltip="Projection warp type. spherical (default): "
		        "good for wide panoramas. plane: no "
		        "warping (flat scenes). cylindrical: "
		        "vertical lines stay straight. "
		        "fisheye/stereographic/mercator: "
		        "specialized projections."),
	io.Combo.Input("wave_correction",
		options=_STITCH_ADVANCED_WAVE_CORRECT, default="auto",
		tooltip="Wave correction for horizontal/vertical "
		        "drift. auto (default): auto-detect "
		        "direction. horiz/vert: force direction. "
		        "no: disable."),
	io.Float.Input("registration_resol", default=-1.0, min=-1.0, max=100.0, step=0.01,
		tooltip="Resolution in megapixels for the "
		        "registration stage (feature detection "
		        "+ matching). -1 = full resolution."),
	io.Float.Input("seam_estimation_resol", default=-1.0, min=-1.0, max=100.0, step=0.01,
		tooltip="Resolution in megapixels for seam "
		        "estimation and exposure compensation. "
		        "-1 = full resolution."),
	io.Float.Input("compositing_resol", default=-1.0, min=-1.0, max=100.0, step=0.01,
		tooltip="Resolution in megapixels for "
		        "compositing/warping phase. "
		        "-1 = full resolution."),
	io.Combo.Input("interpolation_flags",
		options=enums.INTER.options,
		default=enums.INTER.default,
		tooltip="Interpolation method for warping. "
		        "INTER_LINEAR (default) is fast; "
		        "INTER_CUBIC or INTER_LANCZOS4 are "
		        "sharper."),
	io.Combo.Input("exposure_compensation",
		options=_STITCH_ADVANCED_EXPOS_COMP,
		default="gain_blocks",
		tooltip="Exposure compensation method. "
		        "gain_blocks (default): local block "
		        "gain. gain: global per-image gain. "
		        "channel: per-channel gain. "
		        "channel_blocks: local per-channel "
		        "gain. no: disable."),
	io.Int.Input("expos_comp_nr_feeds", default=1, min=1, max=10,
		tooltip="Number of exposure compensation feeds. "
		        "Only used for channel / "
		        "channel_blocks compensators."),
	io.Int.Input("expos_comp_nr_filtering", default=2, min=1, max=10,
		tooltip="Number of filtering iterations for "
		        "exposure compensation gains. Higher = "
		        "smoother gain maps. Default 2. "
		        "Only used for channel / "
		        "channel_blocks compensators."),
	io.Int.Input("expos_comp_block_size", default=32, min=1, max=256,
		tooltip="Block size (pixels) for block-based "
		        "exposure compensators (gain_blocks, "
		        "channel_blocks)."),
	io.Combo.Input("seam_finder",
		options=_STITCH_ADVANCED_SEAM_FINDERS,
		default="gc_colorgrad",
		tooltip="Seam finding method. "
		        "gc_colorgrad (default): GraphCut with "
		        "color + gradient cost. "
		        "gc_color: GraphCut, color only. "
		        "dp_color / dp_colorgrad: dynamic "
		        "programming. voronoi: Voronoi "
		        "diagram. no: skip."),
	io.Combo.Input("blend_type",
		options=_STITCH_ADVANCED_BLEND_TYPES,
		default="multiband",
		tooltip="Blending method. multiband (default): "
		        "Laplacian pyramid blend (best "
		        "quality). feather: linear distance "
		        "blend. no: direct cut."),
	io.Float.Input("blend_strength", default=5.0, min=0.0, max=100.0, step=0.5,
		tooltip="Blending strength [0-100]. Higher = "
		        "softer transition at seams. "
		        "Default 5.0."),
)


def _build_stitch_advanced_config(
	images, masks,
	features, matcher, match_conf, range_width,
	estimator, bundle_adjuster, pano_confidence_thresh,
	warp_type, wave_correction,
	registration_resol, seam_estimation_resol, compositing_resol,
	interpolation_flags,
	exposure_compensation, expos_comp_nr_feeds,
	expos_comp_nr_filtering, expos_comp_block_size,
	seam_finder,
	blend_type, blend_strength,
):
	has_masks = masks is not None
	return {
		"registrationResol": float(registration_resol),
		"seamEstimationResol": float(seam_estimation_resol),
		"compositingResol": float(compositing_resol),
		"panoConfidenceThresh": float(pano_confidence_thresh),
		"waveCorrection": str(wave_correction),
		"interpolationFlags": enums.INTER.resolve(interpolation_flags),
		"features": str(features),
		"matcher": str(matcher),
		"matchConf": float(match_conf),
		"rangeWidth": int(range_width),
		"estimator": str(estimator),
		"bundleAdjuster": str(bundle_adjuster),
		"warpType": str(warp_type),
		"exposureCompensation": str(exposure_compensation),
		"exposCompNrFeeds": int(expos_comp_nr_feeds),
		"exposCompNrFiltering": int(expos_comp_nr_filtering),
		"exposCompBlockSize": int(expos_comp_block_size),
		"seamFinder": str(seam_finder),
		"blendType": str(blend_type),
		"blendStrength": float(blend_strength),
		"hasMasks": has_masks,
	}


def _stitch_advanced_core(img_list, mask_list, config, tag):
	worker_script = workerutils.worker_script_path('stitch_advanced_worker.py')
	result_tag, *rest = workerutils.run_subprocess(
		worker_script,
		{'image_list': img_list, 'mask_list': mask_list, 'config': config})
	if result_tag == 'crash':
		logging.warning("CV_Stitch_Advanced: %s",
		                rest[0] if rest else "worker crashed.")
		return io.NodeOutput(
			polytype.restore(np.ascontiguousarray(img_list[0]), tag),
			"Stitcher_WORKER_CRASHED", False)
	if result_tag == 'error':
		err_msg = rest[0] if rest else "unknown"
		logging.warning("CV_Stitch_Advanced: worker error: %s", err_msg)
		return io.NodeOutput(
			polytype.restore(np.ascontiguousarray(img_list[0]), tag),
			f"Stitcher_ERROR_{err_msg[:120]}", False)

	status, panorama = int(rest[0]), rest[1]

	if status != cv2.Stitcher_OK:
		logging.warning("CV_Stitch_Advanced: stitch returned status %d, "
		                "returning first frame as fallback.", status)
		return io.NodeOutput(
			polytype.restore(np.ascontiguousarray(img_list[0]), tag),
			"Stitcher_ERR", False)

	return io.NodeOutput(
		polytype.restore(np.asarray(panorama, np.uint8), tag),
		"Stitcher_OK", True)


class CV_Stitch_Advanced(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Stitch_Advanced",
			display_name="CV Stitch (Advanced)",
			category=CATEGORY,
			description="Stitches a batch of images into a panorama using "
			            "the low-level cv2.detail pipeline. Provides full "
			            "control over feature detection, matching, camera "
			            "estimation, warp type, exposure compensation, seam "
			            "finding, and blending. Runs in a separate process so "
			            "stitching can be interrupted mid-workflow. "
			            "Accepts IMAGE, LATENT, or NPARRAY inputs via the "
			            "polytype socket. Returns the stitched panorama, a "
			            "status label, and a success flag for downstream "
			            "branching - a failed stitch never halts the workflow.",
			inputs=[
				polytype.poly_input("images",
					display_name="image batch",
					tooltip="Image batch to stitch. IMAGE/LATENT: frame 0 only. "
					        "NPARRAY batch [B,H,W,C]: all B frames are stitched "
					        "in order. Use 'Image Batch → CV Batch' to convert "
					        "a ComfyUI IMAGE batch into a batched NPARRAY."),
				*_STITCH_ADVANCED_SCALAR_INPUTS,
				polytype.poly_input("masks", types=polytype.MASK_ONLY,
					display_name="mask batch", optional=True,
					tooltip="Optional coverage mask batch (same B "
					        "dim as the image batch, or a single "
					        "mask shared by all frames)."),
			],
			outputs=list(_STITCH_ADVANCED_OUTPUTS),
		)

	@classmethod
	def execute(cls, images,
	            features, matcher, match_conf, range_width,
	            estimator, bundle_adjuster, pano_confidence_thresh,
	            warp_type, wave_correction,
	            registration_resol, seam_estimation_resol, compositing_resol,
	            interpolation_flags,
	            exposure_compensation, expos_comp_nr_feeds,
	            expos_comp_nr_filtering, expos_comp_block_size,
	            seam_finder,
	            blend_type, blend_strength,
	            masks=None) -> io.NodeOutput:
		if isinstance(images, torch.Tensor):
			_tag = "image" if images.ndim == 4 else "mask"
		elif isinstance(images, dict) and "samples" in images:
			_tag = "latent"
		else:
			_tag = "nparray"
		img_list = imageutils.frames_from_input(images)

		if len(img_list) < 2:
			logging.warning("CV_Stitch_Advanced: need at least 2 images, got %d.",
			                len(img_list))
			fallback = (np.ascontiguousarray(img_list[0])
			            if img_list else np.zeros((1, 1, 3), np.uint8))
			return io.NodeOutput(
				polytype.restore(fallback, _tag),
				"Stitcher_ERR_NEED_MORE_IMGS", False)

		mask_list = None
		if masks is not None:
			mask_list = imageutils.masks_from_input(masks)
			if len(mask_list) == 1 and len(img_list) > 1:
				mask_list = mask_list * len(img_list)

		config = _build_stitch_advanced_config(
			images, masks,
			features, matcher, match_conf, range_width,
			estimator, bundle_adjuster, pano_confidence_thresh,
			warp_type, wave_correction,
			registration_resol, seam_estimation_resol, compositing_resol,
			interpolation_flags,
			exposure_compensation, expos_comp_nr_feeds,
			expos_comp_nr_filtering, expos_comp_block_size,
			seam_finder,
			blend_type, blend_strength,
		)
		return _stitch_advanced_core(img_list, mask_list, config, _tag)


class CV_Stitch_Advanced_List(io.ComfyNode):
	INPUT_IS_LIST = True

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Stitch_Advanced_List",
			display_name="CV Stitch (Advanced List)",
			category=CATEGORY,
			description="Stitches a list of images into a panorama using "
			            "the low-level cv2.detail pipeline. Provides full "
			            "control over feature detection, matching, camera "
			            "estimation, warp type, exposure compensation, seam "
			            "finding, and blending. Runs in a separate process so "
			            "stitching can be interrupted mid-workflow. "
			            "Accepts a list of IMAGE, LATENT, or NPARRAY items "
			            "via the polytype socket. Only the first frame from "
			            "each list element is used. Returns the stitched "
			            "panorama, a status label, and a success flag.",
			inputs=[
				polytype.poly_input("images",
					display_name="image list",
					tooltip="List of images to stitch. Each element is an "
					        "IMAGE / LATENT / NPARRAY; only the first "
					        "frame of each is used."),
				*_STITCH_ADVANCED_SCALAR_INPUTS,
				polytype.poly_input("masks", types=polytype.MASK_ONLY,
					display_name="mask list", optional=True,
					tooltip="Optional list of coverage masks (one per "
					        "image in the image list)."),
			],
			outputs=list(_STITCH_ADVANCED_OUTPUTS),
		)

	@classmethod
	def execute(cls, images,
	            features, matcher, match_conf, range_width,
	            estimator, bundle_adjuster, pano_confidence_thresh,
	            warp_type, wave_correction,
	            registration_resol, seam_estimation_resol, compositing_resol,
	            interpolation_flags,
	            exposure_compensation, expos_comp_nr_feeds,
	            expos_comp_nr_filtering, expos_comp_block_size,
	            seam_finder,
	            blend_type, blend_strength,
	            masks=None) -> io.NodeOutput:
		if not images:
			return io.NodeOutput(np.zeros((1, 1, 3), np.uint8),
				"Stitcher_ERR_NEED_MORE_IMGS", False)

		img_list = []
		for item in images:
			frames = imageutils.frames_from_input(item)
			if frames:
				img_list.append(frames[0])

		if len(img_list) < 2:
			logging.warning("CV_Stitch_Advanced_List: need at least 2 images, "
			                "got %d.", len(img_list))
			fallback = (np.ascontiguousarray(img_list[0])
			            if img_list else np.zeros((1, 1, 3), np.uint8))
			return io.NodeOutput(fallback,
				"Stitcher_ERR_NEED_MORE_IMGS", False)

		mask_list_flat = None
		if masks is not None:
			mask_list_flat = []
			for item in masks:
				m_mask = imageutils.masks_from_input(item)
				if m_mask:
					mask_list_flat.append(m_mask[0])
			if len(mask_list_flat) == 1 and len(img_list) > 1:
				mask_list_flat = mask_list_flat * len(img_list)

		_unwrap = lambda v: v[0] if isinstance(v, list) else v
		config = _build_stitch_advanced_config(
			images, masks,
			_unwrap(features), _unwrap(matcher), _unwrap(match_conf), _unwrap(range_width),
			_unwrap(estimator), _unwrap(bundle_adjuster), _unwrap(pano_confidence_thresh),
			_unwrap(warp_type), _unwrap(wave_correction),
			_unwrap(registration_resol), _unwrap(seam_estimation_resol), _unwrap(compositing_resol),
			_unwrap(interpolation_flags),
			_unwrap(exposure_compensation), _unwrap(expos_comp_nr_feeds),
			_unwrap(expos_comp_nr_filtering), _unwrap(expos_comp_block_size),
			_unwrap(seam_finder),
			_unwrap(blend_type), _unwrap(blend_strength),
		)
		return _stitch_advanced_core(img_list, mask_list_flat, config, "nparray")


class ConstantLike(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ConstantLike",
			display_name="CV Constant Like",
			category=CATEGORY,
			description="Outputs a uint8 array filled with a constant, with the "
			            "same width/height as the input. Typical use: make an "
			            "all-white coverage mask, warp it with the exact same "
			            "matrix as its image (borderMode=BORDER_CONSTANT, or the "
			            "border reflection turns the whole canvas white), and "
			            "feed the warped masks to 'CV Feather Blend' so each "
			            "canvas pixel knows which sources actually cover it.",
			inputs=[
				polytype.poly_input("nparray", tooltip="Only its width/height are used."),
				io.Int.Input("value", default=255, min=0, max=255,
					tooltip="Fill value 0-255 (255 = white coverage mask)."),
				io.Combo.Input("channels",
					options=["grayscale (single channel)", "same as input"],
					default="grayscale (single channel)",
					tooltip="Masks are single-channel; 'same as input' replicates "
					        "the input's channel count for image-typed uses."),
			],
			outputs=[NPArray.Output(display_name="nparray")],
		)

	@classmethod
	def execute(cls, nparray, value, channels) -> io.NodeOutput:
		arr = np.asarray(polytype.resolve(nparray)[0])
		if arr.ndim < 2:
			raise ValueError(f"Array of shape {arr.shape} has no image size.")
		shape = arr.shape[:2]
		if channels == "same as input" and arr.ndim == 3:
			shape = (*shape, arr.shape[2])
		return io.NodeOutput(np.full(shape, value, np.uint8))


# Loud, repeated warning for the deliberately-fragile broadcast-view node below.
_ZEROSTRIDE_WARN = ("DANGER: this node emits a 0-STRIDE BROADCAST VIEW, "
	"not a real filled array. ")

_ZEROSTRIDE_DTYPES = {
	"uint8": np.uint8,
	"uint16": np.uint16,
	"int32": np.int32,
	"float32": np.float32,
	"float64": np.float64,
}


class ZeroStrideLike(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ZeroStrideLike",
			display_name="\u26a0\ufe0f CV ZeroStride Like",
			category=CATEGORY,
			description=_ZEROSTRIDE_WARN +
				"Outputs ONE backing element broadcast to the input's "
				"width/height (and channels), so it allocates almost nothing - the "
				"point is to skip allocating a full canvas that cv2 will immediately "
				"overwrite. It is ONLY safe as the `dst` of a cv2 node that writes "
				"its result into the array in place (cv2.randu, cv2.randn, "
				"cv2.accumulate, the drawing primitives, ...). NEVER use the output "
				"as a source image, mask, for previews, or convert it to a tensor: "
				"those read every element and will error or yield garbage (every "
				"row/column identical). The low-level wrapper copies it into a real "
				"contiguous array before cv2 runs, so cv2 always receives a "
				"legitimate array - the saving is only this node's own allocation.",
			inputs=[
				polytype.poly_input("nparray",
					tooltip="Only its width/height (and channels, if 'same as "
					        "input') and dtype (if dtype is 'same as input') are "
					        "used to size the broadcast view."),
				io.Float.Input("value", default=0.0, min=0.0,
					tooltip=_ZEROSTRIDE_WARN +
						"The single backing value broadcast to every pixel. cv2 "
						"overwrites it, so this only sets the view's dtype/shape, "
						"not the eventual content."),
				io.Combo.Input("channels",
					options=["grayscale (single channel)", "same as input"],
					default="grayscale (single channel)",
					tooltip="Mirrors CV Constant Like: single-channel, or "
					        "replicate the input's channel count."),
				io.Combo.Input("dtype",
					options=["same as input", "uint8", "uint16", "int32",
					         "float32", "float64"],
					default="uint8",
					tooltip="Element type of the view. 'same as input' takes the "
					        "input array's dtype; otherwise pick an explicit one "
					        "(uint8 matches CV Constant Like)."),
			],
			outputs=[NPArray.Output(display_name="nparray",
				tooltip=_ZEROSTRIDE_WARN +
					"0-stride broadcast VIEW. Safe only as a cv2 dst that is "
					"overwritten. Not a normal array - do not read it as one.")],
		)

	@classmethod
	def execute(cls, nparray, value, channels, dtype) -> io.NodeOutput:
		arr = np.asarray(polytype.resolve(nparray)[0])
		if arr.ndim < 2:
			raise ValueError(f"Array of shape {arr.shape} has no image size.")
		shape = arr.shape[:2]
		if channels == "same as input" and arr.ndim == 3:
			shape = (*shape, arr.shape[2])
		dt = arr.dtype if dtype == "same as input" else _ZEROSTRIDE_DTYPES[dtype]
		# one backing element broadcast to the full shape: a 0-stride view that
		# allocates almost nothing (the low-level wrapper copies it before cv2).
		return io.NodeOutput(np.broadcast_to(np.array([value], dt), shape))


class FeatherBlend(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGE_ONLY)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FeatherBlend",
			display_name="CV Feather Blend",
			category=CATEGORY,
			description="Blends two same-size images by their coverage: pixels "
			            "covered by only one image are copied untouched, and in "
			            "the overlap the weights ramp linearly with the distance "
			            "to each image's border (cv2.distanceTransform) over "
			            "'feather' pixels - a seamless transition for panorama "
			            "stitching. Coverage masks default to the non-black "
			            "pixels of each image; for warped images connect the "
			            "warped 'CV Constant Like' masks instead, which also "
			            "keeps true-black image content from being treated as "
			            "uncovered. The weights output (1 = pure image_a) "
			            "previews well as a heatmap.",
			inputs=[
				polytype.match_input("image_a", cls._TEMPLATE,
					tooltip="First image, already on the shared output canvas."),
				polytype.match_input("image_b", cls._TEMPLATE,
					tooltip="Second image, same size and format as image_a."),
				io.Float.Input("feather", default=32.0, min=0.0, max=1024.0, step=1.0,
					tooltip="Transition width in pixels inside the overlap. "
					        "0 = hard 50/50 average where both images are valid."),
				polytype.poly_input("mask_a", types=polytype.MASK_ONLY, optional=True,
					tooltip="Coverage of image_a (>0 = valid). Default: "
					        "non-black pixels of image_a."),
				polytype.poly_input("mask_b", types=polytype.MASK_ONLY, optional=True,
					tooltip="Coverage of image_b. Default: non-black pixels of image_b."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="Blended image in the inputs' format; black where "
					        "neither input is covered."),
				NPArray.Output(display_name="weights",
					tooltip="HxW float32 share of image_a (1 = pure A, 0 = pure "
					        "B) - debug it with 'Preview CV Array' in heatmap mode."),
			],
		)

	@classmethod
	def execute(cls, image_a, image_b, feather, mask_a=None, mask_b=None) -> io.NodeOutput:
		image_a, tag = polytype.resolve(image_a)
		image_b, _ = polytype.resolve(image_b)
		mask_a, _ = polytype.resolve(mask_a)
		mask_b, _ = polytype.resolve(mask_b)
		a, b = _to_bgr_u8(image_a), _to_bgr_u8(image_b)
		if a.shape[:2] != b.shape[:2]:
			raise ValueError(
				f"image_a is {a.shape[1]}x{a.shape[0]} but image_b is "
				f"{b.shape[1]}x{b.shape[0]} - both must be on the same size "
				"canvas (warp them with the same 'CV Stitch Canvas' dsize)."
			)

		def weight(img, mask):
			if mask is None:
				valid = (img.max(axis=2) > 0).astype(np.uint8)
			else:
				m = imageutils.ensure_uint8(np.asarray(mask))
				if m.ndim == 3:
					m = m.max(axis=2)
				valid = (imageutils.fit_mask_to(m, img.shape[:2]) > 0).astype(np.uint8)
			if feather <= 0:
				return valid.astype(np.float32)
			# pad so a mask covering the whole frame still gets finite distances
			dist = cv2.distanceTransform(np.pad(valid, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
			return np.minimum(dist / feather, 1.0, dtype=np.float32)

		wa, wb = weight(a, mask_a), weight(b, mask_b)
		total = np.maximum(wa + wb, 1e-6)
		share_a = np.where(wa + wb > 0, wa / total, 0.0).astype(np.float32)
		share_b = np.where(wa + wb > 0, wb / total, 0.0).astype(np.float32)
		out = a * share_a[..., None] + b * share_b[..., None]
		return io.NodeOutput(
			polytype.restore(np.clip(out.round(), 0, 255).astype(np.uint8), tag),
			share_a)


def _slot_key(name):
	"""Sort an autogrow slot name ("image0", "image1", "image10") numerically."""
	digits = "".join(ch for ch in str(name) if ch.isdigit())
	return int(digits) if digits else 0


class StackImages(io.ComfyNode):
	# IMAGEISH (NPARRAY + IMAGE + MASK): the node stacks masks too, and the
	# shared MatchType template makes every autogrow slot follow the type of
	# the first connected input (like core 'Create List').
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		template_matchtype = cls._TEMPLATE
		template_autogrow = io.Autogrow.TemplatePrefix(
			input=polytype.match_input("image", template=template_matchtype),
			prefix="image",
		)
		return io.Schema(
			node_id="CV_StackImages",
			display_name="CV Stack Images",
			category=CATEGORY,
			description="Concatenates any number of images/masks side by side "
			            "(or one above the other) on a black canvas, padding "
			            "each to the largest - before/after comparisons, "
			            "building strips tile by tile. All inputs share the "
			            "first input's type (IMAGE, MASK or NPARRAY) and stack "
			            "in the same direction. As the body of an Inspire "
			            "'Foreach List' loop (image0 = intermediate_output, "
			            "image1 = item) it folds a whole list into one contact "
			            "sheet.",
			inputs=[
				io.Autogrow.Input("images", template=template_autogrow),
				io.Combo.Input("direction",
					options=["horizontal (a then b)", "vertical (a above b)"],
					default="horizontal (a then b)",
					tooltip="Side by side, or one above the other."),
				io.Int.Input("gap", default=0, min=0, max=256,
					tooltip="Black separator between consecutive inputs, in pixels."),
			],
			outputs=[polytype.match_output(template_matchtype, "image",
				tooltip="Same format as the image inputs.")],
		)

	@classmethod
	def execute(cls, images, direction, gap) -> io.NodeOutput:
		items = [images[k] for k in sorted(images, key=_slot_key)]
		if not items:
			raise ValueError("CV Stack Images needs at least one input.")
		_, tag = polytype.resolve(items[0])
		if tag == "mask":
			frames = []
			for v in items:
				arr, _ = polytype.resolve(v)
				arr = imageutils.ensure_uint8(np.asarray(arr))
				if arr.ndim == 3 and arr.shape[2] == 1:
					arr = arr[..., 0]
				frames.append(np.ascontiguousarray(arr))
			chan = 1
		else:
			frames = [_to_bgr_u8(polytype.resolve(v)[0]) for v in items]
			chan = 3
		gap = max(0, int(gap))
		if direction.startswith("vertical"):
			w = max(f.shape[1] for f in frames)
			pad = lambda f: cv2.copyMakeBorder(  # noqa: E731
				f, 0, 0, 0, w - f.shape[1], cv2.BORDER_CONSTANT, value=0)
			spacer = np.zeros((gap, w, chan) if chan == 3 else (gap, w), np.uint8)
			parts = []
			for i, f in enumerate(frames):
				if i:
					parts.append(spacer)
				parts.append(pad(f))
			return io.NodeOutput(polytype.restore(np.vstack(parts), tag))
		h = max(f.shape[0] for f in frames)
		pad = lambda f: cv2.copyMakeBorder(  # noqa: E731
			f, 0, h - f.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
		spacer = np.zeros((h, gap, chan) if chan == 3 else (h, gap), np.uint8)
		parts = []
		for i, f in enumerate(frames):
			if i:
				parts.append(spacer)
			parts.append(pad(f))
		return io.NodeOutput(polytype.restore(np.hstack(parts), tag))


class StereoCalibrationFromCSV(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_StereoCalibrationFromCSV",
			display_name="CV Stereo Calibration From CSV",
			category=CATEGORY,
			description="Reads per-image stereo calibration from CSV files "
			            "(e.g. StereoGeo-CARLA). Computes camera intrinsics K "
			            "from vertical FOV (pinhole model, no distortion). "
			            "Returns K, dist (zeros), R, T ready for stereoRectify. "
			            "CSV columns — left: image_name, width, height, roll, "
			            "pitch, vfov. Right: same + R_00..R_22, t_0..t_2. "
			            "Found=false if the image name is not in both CSVs.",
			inputs=[
				io.String.Input("csv_path_left",
					tooltip="Left camera CSV. Bare filename resolves against "
					        "ComfyUI/input/; absolute paths pass through."),
				io.String.Input("csv_path_right",
					tooltip="Right camera CSV. Bare filename resolves against "
					        "ComfyUI/input/; absolute paths pass through."),
				io.String.Input("image_name",
					tooltip="Image filename to look up (e.g. '000000.png')."),
				io.Int.Input("image_width", default=640, min=1,
					tooltip="Image width in pixels (used for cx = width/2)."),
				io.Int.Input("image_height", default=480, min=1,
					tooltip="Image height in pixels (used for cy = height/2)."),
			],
			outputs=[
				NPArray.Output(display_name="K_left",
					tooltip="3x3 intrinsic matrix for the left camera."),
				NPArray.Output(display_name="dist_left",
					tooltip="1x5 distortion coefficients (zeros for pinhole)."),
				NPArray.Output(display_name="K_right",
					tooltip="3x3 intrinsic matrix for the right camera."),
				NPArray.Output(display_name="dist_right",
					tooltip="1x5 distortion coefficients (zeros for pinhole)."),
				NPArray.Output(display_name="R",
					tooltip="3x3 rotation matrix from left to right camera."),
				NPArray.Output(display_name="T",
					tooltip="3x1 translation vector from left to right camera."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, csv_path_left, csv_path_right, image_name,
	            image_width=640, image_height=480) -> io.NodeOutput:
		# Default: not found, identity, zeros
		I3 = np.eye(3, dtype=np.float64)
		dist = np.zeros((1, 5), np.float64)
		T = np.zeros((3, 1), np.float64)

		import folder_paths
		input_dir = folder_paths.get_input_directory()

		def _resolve(p):
			s = str(p).strip()
			if os.path.isabs(s):
				return s
			candidate = os.path.join(input_dir, s)
			if os.path.isfile(candidate):
				return candidate
			return s  # let open() fail naturally so caller sees the error

		def _read_csv(path):
			import csv
			with open(path, "r", newline="") as f:
				return list(csv.reader(f))

		try:
			left_rows = _read_csv(_resolve(csv_path_left))
			right_rows = _read_csv(_resolve(csv_path_right))
		except Exception as e:
			logging.warning("StereoCalibrationFromCSV: cannot read CSV: %s", e)
			return io.NodeOutput(I3, dist, I3, dist, I3, T, False)

		def _find_row(rows, name):
			for row in rows[1:]:  # skip header
				if row and row[0].strip() == name.strip():
					return row
			return None

		left_row = _find_row(left_rows, str(image_name))
		right_row = _find_row(right_rows, str(image_name))
		if left_row is None or right_row is None:
			logging.warning("StereoCalibrationFromCSV: image '%s' not in both CSVs",
			               image_name)
			return io.NodeOutput(I3, dist, I3, dist, I3, T, False)

		try:
			vfov_left = float(left_row[5])
			vfov_right = float(right_row[5])

			fy_l = (float(image_height) / 2.0) / math.tan(vfov_left / 2.0)
			fx_l = fy_l
			cx_l = float(image_width) / 2.0
			cy_l = float(image_height) / 2.0
			K_left = np.array([[fx_l, 0, cx_l],
			                   [0, fy_l, cy_l],
			                   [0, 0, 1]], dtype=np.float64)

			fy_r = (float(image_height) / 2.0) / math.tan(vfov_right / 2.0)
			fx_r = fy_r
			cx_r = float(image_width) / 2.0
			cy_r = float(image_height) / 2.0
			K_right = np.array([[fx_r, 0, cx_r],
			                    [0, fy_r, cy_r],
			                    [0, 0, 1]], dtype=np.float64)

			# R_00..R_22 = right_row[6:15], t_0..t_2 = right_row[15:18]
			R_vals = [float(right_row[i]) for i in range(6, 15)]
			R = np.array(R_vals, dtype=np.float64).reshape(3, 3)

			T_vals = [float(right_row[i]) for i in range(15, 18)]
			T = np.array(T_vals, dtype=np.float64).reshape(3, 1)

			return io.NodeOutput(K_left, dist, K_right, dist, R, T, True)
		except (IndexError, ValueError) as e:
			logging.warning("StereoCalibrationFromCSV: parse error: %s", e)
			return io.NodeOutput(I3, dist, I3, dist, I3, T, False)


class ScaleHomography(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ScaleHomography",
			display_name="CV Scale Homography",
			category=CATEGORY,
			description="Scales a 3x3 homography from one resolution to another "
			            "via numpy (T(scale_a) @ H @ T_inv(scale_b)). "
			            "If both images were downscaled by factor S for feature "
			            "detection, set scale_a=scale_b=1/S to recover the "
			            "full-resolution homography. "
			            "This is a numpy helper, not a cv2 wrapper.",
			inputs=[
				NPArray.Input("homography",
				    tooltip="3x3 homography matrix (float64)."),
				io.Float.Input("scale_a", default=1.0, min=1e-6, max=1e6,
				    tooltip="Scale factor from detection-space to full resolution "
				            "for image A (target/warp-ref image). 1.0 = no change."),
				io.Float.Input("scale_b", default=1.0, min=1e-6, max=1e6,
				    tooltip="Scale factor from detection-space to full resolution "
				            "for image B (source/warp image). 1.0 = no change."),
			],
			outputs=[
				NPArray.Output(display_name="homography"),
			],
		)

	@classmethod
	def execute(cls, homography, scale_a, scale_b) -> io.NodeOutput:
		if homography is None or homography.size == 0:
			return io.NodeOutput(np.eye(3, dtype=np.float64))
		H = np.asarray(homography, np.float64).reshape(3, 3)
		T_a = np.diag([scale_a, scale_a, 1.0])
		T_b_inv = np.diag([1.0 / scale_b, 1.0 / scale_b, 1.0])
		return io.NodeOutput(T_a @ H @ T_b_inv)


class Eye(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Eye",
			display_name="CV Eye",
			category=CATEGORY,
			description="Creates an NxN identity matrix via numpy (np.eye). "
			            "Useful for seeding homography chains or resetting to "
			            "a no-op transform. This is a numpy helper, not a cv2 wrapper.",
			inputs=[
				io.Int.Input("n", default=3, min=1, max=10,
				    tooltip="Size N of the NxN identity matrix. "
				            "Default 3 matches the homography format."),
			],
			outputs=[
				NPArray.Output(display_name="matrix",
				    tooltip="NxN float64 identity matrix."),
			],
		)

	@classmethod
	def execute(cls, n) -> io.NodeOutput:
		return io.NodeOutput(np.eye(int(n), dtype=np.float64))


class MatMul(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MatMul",
			display_name="CV Matrix Multiply",
			category=CATEGORY,
			description="Multiplies two matrices via numpy (left @ right). "
			            "Typically used to chain homographies: "
			            "H_cumul = H_previous @ H_pair so that the result "
			            "maps from the new image frame back to the canvas frame. "
			            "This is a numpy helper, not a cv2 wrapper.",
			inputs=[
				NPArray.Input("left",
				    tooltip="Left matrix (MxK)."),
				NPArray.Input("right",
				    tooltip="Right matrix (KxN)."),
			],
			outputs=[
				NPArray.Output(display_name="result",
				    tooltip="MxN float64 matrix = left @ right."),
			],
		)

	@classmethod
	def execute(cls, left, right) -> io.NodeOutput:
		if left is None or right is None:
			return io.NodeOutput(np.eye(3, dtype=np.float64))
		return io.NodeOutput(np.asarray(left, np.float64) @ np.asarray(right, np.float64))


# --- class-gated core APIs the generator cannot reach ----------------------
# cv2 exposes these through factory + object (StereoBM_create, KalmanFilter,
# createLineSegmentDetector, SimpleBlobDetector_create, ...). generator.py only
# parses top-level functions whose return type it can classify, so every one of
# them is missing from registry.py; the nodes below create the object, use it
# and drop it inside a single execute(), the same contract as StereoDisparity.


class StereoDisparityBM(io.ComfyNode):
	_PREFILTER = {
		"normalized response": cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE,
		"x-Sobel": cv2.STEREO_BM_PREFILTER_XSOBEL,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_StereoDisparityBM",
			display_name="CV Stereo Disparity (BM)",
			category=CATEGORY,
			description="Per-pixel disparity from a RECTIFIED stereo pair with the "
			            "classic BLOCK MATCHER (cv2.StereoBM) - the fast, local "
			            "cousin of 'CV Stereo Disparity (SGBM)'. It compares "
			            "raw SAD windows scanline by scanline with no global "
			            "smoothness term, so it runs several times faster but "
			            "leaves far more holes in low-texture areas; that is what "
			            "texture_threshold controls. Use it for a quick preview, a "
			            "speed/quality comparison, or when the scene is richly "
			            "textured. The pair MUST be rectified (epipolar lines "
			            "horizontal). Preview the disparity with 'Preview CV Array' "
			            "(normalize / heatmap); larger disparity = closer.",
			inputs=[
				polytype.poly_input("left", types=polytype.IMAGE_ONLY,
					tooltip="Left rectified frame (grayscaled internally)."),
				polytype.poly_input("right", types=polytype.IMAGE_ONLY,
					tooltip="Right rectified frame; resized to left if sizes differ."),
				io.Int.Input("num_disparities", default=64, min=16, max=512, step=16,
					tooltip="Disparity search range (0..num_disparities), rounded up "
					        "to a multiple of 16. Larger covers nearer objects / "
					        "wider baselines but is slower."),
				io.Int.Input("block_size", default=15, min=5, max=51,
					tooltip="SAD window size in pixels; forced ODD and clamped to "
					        "5..51 (StereoBM rejects anything else). Larger = "
					        "smoother but blurs depth edges; 15-21 is typical."),
				io.Int.Input("min_disparity", default=0, min=-256, max=256,
					tooltip="Smallest disparity to search from (usually 0)."),
				# 163863 = 51*51*63, i.e. block_size^2 * prefilter_cap at this
				# node's own widget maxima - the value above which the gate can
				# never pass a block. A lower cap (it used to be 1000) puts the
				# widget BELOW the parameter's useful range at every setting.
				io.Int.Input("texture_threshold", default=10, min=0, max=163863,
					tooltip="Reject a block whose texture (summed x-gradient) is "
					        "below this - the main reason BM leaves holes on flat "
					        "walls. 0 accepts every block (noisy). It is a SUM "
					        "over the whole SAD window, so it only bites in the "
					        "thousands: the ceiling above which every pixel is "
					        "rejected is block_size^2 * prefilter_cap (about "
					        "7000 at the defaults, 163863 at block_size 51 / "
					        "prefilter_cap 63)."),
				io.Int.Input("uniqueness_ratio", default=10, min=0, max=100,
					tooltip="Margin (%) by which the best match must beat the "
					        "runner-up to be accepted."),
				io.Combo.Input("prefilter", options=list(cls._PREFILTER),
					default="x-Sobel",
					tooltip="Pre-filter applied before matching to remove lighting "
					        "differences between the two cameras. x-Sobel (an x "
					        "gradient, clipped to prefilter_cap) is what OpenCV "
					        "itself defaults to and the usual choice. normalized "
					        "response is the older local brightness normalisation "
					        "kept from the legacy C API - it is more tolerant of a "
					        "smooth exposure difference between the cameras but "
					        "weaker on repetitive texture.",
					optional=True, advanced=True),
				io.Int.Input("prefilter_cap", default=31, min=1, max=63,
					tooltip="Pre-filtered values are clipped to +/- this.",
					optional=True, advanced=True),
				io.Int.Input("speckle_window_size", default=100, min=0, max=1000,
					tooltip="Largest smooth disparity blob treated as speckle noise "
					        "and invalidated (0 = off).", optional=True, advanced=True),
				io.Int.Input("speckle_range", default=2, min=0, max=64,
					tooltip="Max disparity variation within a speckle component.",
					optional=True, advanced=True),
				io.Int.Input("disp12_max_diff", default=-1, min=-1, max=256,
					tooltip="Max allowed left-right consistency error in pixels "
					        "(-1 = do not run the left-right check).",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="disparity",
					tooltip="HxW float32 disparity in pixels; rejected pixels are "
					        "below min_disparity (see 'valid')."),
				NPArray.Output(display_name="valid",
					tooltip="uint8 0/255 mask of pixels with a valid disparity - BM "
					        "rejects many more than SGBM, so always check it."),
				io.Float.Output(display_name="valid_fraction",
					tooltip="Fraction of pixels with a valid disparity (0..1) - the "
					        "headline number when comparing BM against SGBM."),
			],
		)

	@classmethod
	def execute(cls, left, right, num_disparities, block_size, min_disparity,
	            texture_threshold, uniqueness_ratio, prefilter="x-Sobel",
	            prefilter_cap=31, speckle_window_size=100, speckle_range=2,
	            disp12_max_diff=-1) -> io.NodeOutput:
		left, _ = polytype.resolve(left)
		right, _ = polytype.resolve(right)
		lg = _to_gray_u8(left)
		rg = _to_gray_u8(right)
		if rg.shape != lg.shape:
			rg = cv2.resize(rg, (lg.shape[1], lg.shape[0]), interpolation=cv2.INTER_AREA)
		nd = max(16, int(round(num_disparities / 16)) * 16)
		# StereoBM raises unless the SAD window is odd, within 5..255 and no
		# larger than the image - clamp instead of letting a widget value crash
		bs = max(5, min(51, int(block_size) | 1))
		bs = min(bs, (min(lg.shape[:2]) - 1) | 1)
		matcher = cv2.StereoBM_create(nd, bs)
		matcher.setMinDisparity(int(min_disparity))
		matcher.setTextureThreshold(int(texture_threshold))
		matcher.setUniquenessRatio(int(uniqueness_ratio))
		matcher.setPreFilterType(cls._PREFILTER[prefilter])
		matcher.setPreFilterCap(int(prefilter_cap))
		matcher.setSpeckleWindowSize(int(speckle_window_size))
		matcher.setSpeckleRange(int(speckle_range))
		matcher.setDisp12MaxDiff(int(disp12_max_diff))
		# StereoBM returns int16 disparity scaled by 16; rejected = (min-1)*16
		disparity = matcher.compute(lg, rg).astype(np.float32) / 16.0
		valid = (disparity >= min_disparity).astype(np.uint8) * 255
		return io.NodeOutput(disparity, valid, float(valid.mean()) / 255.0)


class QuasiDenseStereo(io.ComfyNode):
	# cv2 spells it 'textrureThreshold' - keep the typo isolated in here
	_PARAMS = (
		("max_seeds", "gftMaxNumFeatures", int),
		("seed_quality", "gftQualityThres", float),
		("seed_min_distance", "gftMinSeperationDist", int),
		("correlation_window", "corrWinSizeX", int),
		("correlation_window", "corrWinSizeY", int),
		("correlation_threshold", "correlationThreshold", float),
		("texture_threshold", "textrureThreshold", float),
		("disparity_gradient", "disparityGradient", int),
		("neighborhood_size", "neighborhoodSize", int),
		("border", "borderX", int),
		("border", "borderY", int),
	)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_QuasiDenseStereo",
			display_name="CV Quasi-Dense Stereo",
			category=CATEGORY,
			description="Stereo matching by SEED-AND-GROW "
			            "(cv2.stereo.QuasiDenseStereo), the third answer next to "
			            "'Stereo Disparity (SGBM)' and '(BM)'. Those two search a "
			            "disparity RANGE at every pixel independently; this one "
			            "tracks a few hundred good-features-to-track seeds with "
			            "Lucas-Kanade and then GROWS each match into its "
			            "neighbourhood as long as the correlation and the local "
			            "disparity gradient hold up. Consequences worth knowing: "
			            "there is no num_disparities to get wrong (a seed can sit "
			            "anywhere, so a wide baseline costs nothing), it answers "
			            "only where the image has texture to propagate through - "
			            "the output is quasi-dense, not dense - and it needs no "
			            "rectification to run, though the disparity output only "
			            "means anything on a RECTIFIED pair. The class is not "
			            "reachable from the raw wrappers. Failure-tolerant: a pair "
			            "with no trackable features (a blank frame) returns "
			            "found=false and empty arrays instead of the OpenCV "
			            "assertion the class raises.",
			inputs=[
				polytype.poly_input("left", types=polytype.IMAGE_ONLY,
					tooltip="Left frame (grayscaled internally)."),
				polytype.poly_input("right", types=polytype.IMAGE_ONLY,
					tooltip="Right frame; resized to left if the sizes differ - the "
					        "class asserts on mismatched frames."),
				io.Int.Input("max_seeds", default=500, min=10, max=5000,
					tooltip="How many goodFeaturesToTrack seeds to start from "
					        "(gftMaxNumFeatures). More seeds reach more regions - "
					        "propagation cannot cross a texture-less gap, so an "
					        "unseeded area stays empty however good the matching is."),
				io.Float.Input("seed_quality", default=0.01, min=0.0001, max=1.0,
					step=0.005,
					tooltip="Corner quality of a seed relative to the strongest "
					        "corner (gftQualityThres). Lower accepts weaker corners, "
					        "i.e. more seeds in flat regions."),
				io.Int.Input("seed_min_distance", default=10, min=1, max=200,
					tooltip="Minimum pixel distance between seeds "
					        "(gftMinSeperationDist) - spreads them over the frame "
					        "instead of clustering on one strong texture."),
				io.Int.Input("correlation_window", default=5, min=3, max=31,
					tooltip="Half-window of the correlation test used when growing a "
					        "match (corrWinSizeX/Y). Larger is more reliable and "
					        "blurs across depth discontinuities."),
				io.Float.Input("correlation_threshold", default=0.5, min=0.0, max=1.0,
					step=0.05,
					tooltip="Minimum normalized correlation for a grown match to be "
					        "kept. Raise it to trade coverage for correctness."),
				io.Float.Input("texture_threshold", default=200.0, min=0.0, max=10000.0,
					step=10.0,
					tooltip="Minimum local texture (sum of squared gradients) a pixel "
					        "must have before propagation continues into it. This is "
					        "what stops the growth from flooding a blank wall or the "
					        "sky with an invented disparity."),
				io.Int.Input("disparity_gradient", default=1, min=0, max=10,
					tooltip="How much the disparity may change between neighbouring "
					        "pixels. 1 keeps surfaces smooth; larger follows steep "
					        "slopes and lets the growth leak across object edges."),
				io.Int.Input("neighborhood_size", default=5, min=3, max=21,
					tooltip="Size of the neighbourhood a confirmed match propagates "
					        "into.", optional=True, advanced=True),
				io.Int.Input("border", default=15, min=0, max=100,
					tooltip="Image margin (borderX/borderY) excluded from matching, "
					        "so a correlation window never runs off the frame.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="disparity",
					tooltip="HxW float32 SIGNED horizontal disparity in pixels "
					        "(x_left - x_right), the same convention as 'Stereo "
					        "Disparity (SGBM)' / '(BM)' - so it feeds 'CV Disparity "
					        "Interpolate', cv2.reprojectImageTo3D and the WLS filter "
					        "directly. Built from the match set, NOT from cv2's own "
					        "getDisparity(), which returns the UNSIGNED euclidean "
					        "displacement (see 'displacement'). 0 where unmatched - "
					        "read 'valid', not zero."),
				NPArray.Output(display_name="valid",
					tooltip="uint8 0/255 mask of pixels that actually got a match "
					        "(the quasi-dense support)."),
				NPArray.Output(display_name="displacement",
					tooltip="HxW float32 of cv2's own getDisparity(): the EUCLIDEAN "
					        "length of each match's displacement, sqrt(dx^2 + dy^2), "
					        "unsigned and including vertical motion. On a rectified "
					        "pair it is |disparity|; on an unrectified one the two "
					        "differ by tens of pixels, which is why the disparity "
					        "output above is not this. NaN (cv2's 'no match') is "
					        "written as 0."),
				NPArray.Output(display_name="points_left",
					tooltip="Nx1x2 float32 left-image coordinates of every DENSE "
					        "match - wire straight into 'Triangulate Points', "
					        "'Find Homography (RANSAC)' or 'Draw Points'."),
				NPArray.Output(display_name="points_right",
					tooltip="Nx1x2 float32 right-image coordinates, row-aligned with "
					        "points_left."),
				NPArray.Output(display_name="seeds_left",
					tooltip="Nx1x2 float32 left-image coordinates of the SPARSE seed "
					        "matches the propagation started from - a few hundred "
					        "against the dense set's tens of thousands."),
				NPArray.Output(display_name="seeds_right",
					tooltip="Nx1x2 float32 right-image coordinates of the seeds."),
				io.Int.Output(display_name="dense_count",
					tooltip="Number of quasi-dense matches."),
				io.Int.Output(display_name="seed_count",
					tooltip="Number of seed matches."),
				io.Boolean.Output(display_name="found",
					tooltip="False when the pair produced no dense match at all "
					        "(blank or uncorrelated frames); the arrays are then "
					        "empty and the disparity is all zero."),
			],
		)

	@classmethod
	def execute(cls, left, right, max_seeds, seed_quality, seed_min_distance,
	            correlation_window, correlation_threshold, texture_threshold,
	            disparity_gradient, neighborhood_size=5, border=15) -> io.NodeOutput:
		left, _ = polytype.resolve(left)
		right, _ = polytype.resolve(right)
		lg = _to_gray_u8(left)
		rg = _to_gray_u8(right)
		if rg.shape != lg.shape:
			rg = cv2.resize(rg, (lg.shape[1], lg.shape[0]), interpolation=cv2.INTER_AREA)
		h, w = lg.shape[:2]

		def empty():
			pts = np.zeros((0, 1, 2), np.float32)
			return io.NodeOutput(np.zeros((h, w), np.float32), np.zeros((h, w), np.uint8),
			                     np.zeros((h, w), np.float32),
			                     pts, pts.copy(), pts.copy(), pts.copy(), 0, 0, False)

		# monoImgSize is what sizes the disparity map: a mismatch does NOT raise,
		# it silently returns a map of the WRONG size, so build from the frame
		qds = cv2.stereo.QuasiDenseStereo_create((w, h))
		values = {"max_seeds": int(max_seeds), "seed_quality": float(seed_quality),
		          "seed_min_distance": int(seed_min_distance),
		          "correlation_window": int(correlation_window),
		          "correlation_threshold": float(correlation_threshold),
		          "texture_threshold": float(texture_threshold),
		          "disparity_gradient": int(disparity_gradient),
		          "neighborhood_size": int(neighborhood_size), "border": int(border)}
		# Param is a property returning a COPY - mutate it, then assign it back
		params = qds.Param
		for name, attr, cast in cls._PARAMS:
			setattr(params, attr, cast(values[name]))
		qds.Param = params
		try:
			qds.process(lg, rg)
		except cv2.error as e:
			# a frame with no trackable corner reaches calcOpticalFlowPyrLK with
			# an EMPTY point set, which asserts inside lkpyramid.cpp
			logging.warning("QuasiDenseStereo: %s", str(e).splitlines()[-1])
			return empty()

		def pack(matches):
			if not matches:
				z = np.zeros((0, 1, 2), np.float32)
				return z, z.copy()
			a = np.float32([[m.p0[0], m.p0[1]] for m in matches]).reshape(-1, 1, 2)
			b = np.float32([[m.p1[0], m.p1[1]] for m in matches]).reshape(-1, 1, 2)
			return a, b

		dense_l, dense_r = pack(qds.getDenseMatches())
		seed_l, seed_r = pack(qds.getSparseMatches())

		# cv2's getDisparity() is the EUCLIDEAN length of the displacement
		# (sqrt(dx^2 + dy^2), unsigned), not the signed horizontal disparity
		# every other stereo node in this pack emits - measured 65 px apart on an
		# unrectified pair. Keep it as its own output and scatter the real
		# disparity from the match set, whose support is exactly the same pixels.
		displacement = np.asarray(qds.getDisparity(), np.float32)
		if displacement.shape[:2] != (h, w):  # defensive: see the monoImgSize note
			displacement = cv2.resize(displacement, (w, h), interpolation=cv2.INTER_NEAREST)
		displacement = np.where(np.isfinite(displacement), displacement, 0.0).astype(np.float32)

		disparity = np.zeros((h, w), np.float32)
		valid = np.zeros((h, w), np.uint8)
		if len(dense_l):
			xy = np.rint(dense_l.reshape(-1, 2)).astype(np.int32)
			inside = ((xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h))
			rows, cols = xy[inside, 1], xy[inside, 0]
			disparity[rows, cols] = (dense_l.reshape(-1, 2)[inside, 0]
			                         - dense_r.reshape(-1, 2)[inside, 0])
			valid[rows, cols] = 255
		return io.NodeOutput(disparity, valid, displacement,
		                     dense_l, dense_r, seed_l, seed_r,
		                     int(len(dense_l)), int(len(seed_l)), bool(len(dense_l)))


class KalmanStep(io.ComfyNode):
	"""One predict/correct step of cv2.KalmanFilter, state carried as NPARRAY."""

	# name -> (n_states, order): order 0 = position only, 1 = +velocity,
	# 2 = +acceleration. Measurement is always the 2-D position.
	_MODELS = {
		"constant position (x, y)": (2, 0),
		"constant velocity (x, y + vx, vy)": (4, 1),
		"constant acceleration (x, y + vx, vy + ax, ay)": (6, 2),
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_KalmanStep",
			display_name="CV Kalman Filter Step",
			category=CATEGORY,
			description="One predict (+ correct) step of cv2.KalmanFilter for a 2-D "
			            "point: smooths a noisy track and, crucially, KEEPS "
			            "PREDICTING while the measurement is missing. cv2.KalmanFilter "
			            "is a stateful class the raw wrappers cannot expose, so this "
			            "node packs the whole filter state (posterior state vector + "
			            "error covariance) into one NPARRAY: chain one node per frame, "
			            "this 'state' output -> the next frame's 'state' input, and "
			            "leave the first frame's state unconnected to initialise from "
			            "the first measurement. Leave 'measurement' unconnected (or "
			            "feed an empty array) on frames where the detector lost the "
			            "object - the node then predicts only and the track coasts "
			            "through the occlusion instead of jumping. Pair with any "
			            "detector output: 'CV Track Window' centers, a "
			            "'Select Component At Point' centroid, a YuNet face box...",
			inputs=[
				io.Combo.Input("model", options=list(cls._MODELS),
					default="constant velocity (x, y + vx, vy)",
					tooltip="Motion model. Constant position only smooths jitter; "
					        "constant velocity is the standard tracker (it coasts in "
					        "a straight line when the measurement drops out); "
					        "constant acceleration also follows curved motion but "
					        "needs a cleaner signal."),
				io.Float.Input("dt", default=1.0, min=0.001, max=1000.0, step=0.1,
					tooltip="Time between this step and the previous one, in "
					        "whatever unit the velocity should use. Leave at 1.0 to "
					        "work in 'pixels per frame'."),
				io.Float.Input("process_noise", default=0.01, min=1e-8, max=1e6, step=0.01,
					tooltip="How much the model is allowed to be wrong per step "
					        "(process noise covariance). RAISE it to follow abrupt "
					        "manoeuvres, LOWER it for a smoother, laggier track."),
				io.Float.Input("measurement_noise", default=1.0, min=1e-8, max=1e6, step=0.1,
					tooltip="How noisy the detector is, in squared pixels. Raise it "
					        "to trust the model more than the measurement (more "
					        "smoothing); lower it to snap onto every detection."),
				NPArray.Input("measurement", optional=True,
					tooltip="Measured position as 2 numbers (x, y) - a 1x2 array, a "
					        "'CV Scalar' literal or any Nx2 point set (the FIRST row "
					        "is used). Leave unconnected or pass an empty array for a "
					        "PREDICT-ONLY step (object occluded / detector failed)."),
				NPArray.Input("state", optional=True,
					tooltip="Packed filter state from the PREVIOUS frame's 'state' "
					        "output. Leave unconnected on the first frame: the filter "
					        "is then seeded at the measurement with zero velocity."),
				io.Float.Input("initial_uncertainty", default=1.0, min=1e-6, max=1e6, step=0.1,
					tooltip="Error covariance the filter starts with when no 'state' "
					        "is connected. Large = 'the seed position is a guess', so "
					        "the first few measurements dominate.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="state",
					tooltip="Packed filter state, float32 (n, n+1): column 0 is the "
					        "state vector, the rest is the error covariance. Wire it "
					        "into the next frame - it is not meant to be read "
					        "directly."),
				NPArray.Output(display_name="filtered",
					tooltip="1x2 filtered position AFTER the correction (the "
					        "smoothed track point). Equals 'predicted' on a "
					        "predict-only step. Feed 'CV Draw Points'."),
				NPArray.Output(display_name="predicted",
					tooltip="1x2 position the motion model expected BEFORE seeing "
					        "the measurement - the extrapolation that carries the "
					        "track through an occlusion."),
				NPArray.Output(display_name="velocity",
					tooltip="1x2 estimated velocity (units per dt); zeros for the "
					        "constant-position model. Draw it with 'CV Draw "
					        "Rays' to show where the track is heading."),
				io.Boolean.Output(display_name="corrected",
					tooltip="True when a measurement was supplied and applied, "
					        "false on a predict-only (coasting) step."),
			],
		)

	@classmethod
	def _build(cls, n, order, dt, process_noise, measurement_noise):
		kf = cv2.KalmanFilter(n, 2)
		A = np.eye(n, dtype=np.float32)
		if order >= 1:
			A[0, 2] = A[1, 3] = dt
		if order >= 2:
			A[2, 4] = A[3, 5] = dt
			A[0, 4] = A[1, 5] = 0.5 * dt * dt
		kf.transitionMatrix = A
		H = np.zeros((2, n), np.float32)
		H[0, 0] = H[1, 1] = 1.0
		kf.measurementMatrix = H
		kf.processNoiseCov = np.eye(n, dtype=np.float32) * float(process_noise)
		kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)
		return kf

	@classmethod
	def execute(cls, model, dt, process_noise, measurement_noise, measurement=None,
	            state=None, initial_uncertainty=1.0) -> io.NodeOutput:
		n, order = cls._MODELS[model]
		kf = cls._build(n, order, float(dt), process_noise, measurement_noise)

		meas = None
		if measurement is not None:
			m = np.asarray(measurement, np.float32).reshape(-1)
			if m.size >= 2:
				meas = m[:2].reshape(2, 1).copy()
			elif m.size:
				raise ValueError(
					f"measurement must hold 2 numbers (x, y), got {m.size}.")

		packed = None if state is None else np.asarray(state, np.float32)
		if packed is not None and packed.size:
			if packed.shape != (n, n + 1):
				raise ValueError(
					f"state has shape {tuple(packed.shape)} but the '{model}' model "
					f"needs {(n, n + 1)} - the model combo must match the node that "
					"produced the state.")
			kf.statePost = np.ascontiguousarray(packed[:, :1])
			kf.errorCovPost = np.ascontiguousarray(packed[:, 1:])
		else:
			# first frame: seed at the measurement (origin if there is none) with
			# zero velocity - a missing state is a valid start, never an error
			seed = np.zeros((n, 1), np.float32)
			if meas is not None:
				seed[0, 0], seed[1, 0] = float(meas[0, 0]), float(meas[1, 0])
			kf.statePost = seed
			kf.errorCovPost = np.eye(n, dtype=np.float32) * float(initial_uncertainty)

		predicted = kf.predict()
		if meas is not None:
			filtered = kf.correct(meas)
		else:
			# cv2 leaves statePost/errorCovPost stale when correct() is skipped;
			# promote the prior so the next step chains off the prediction
			kf.statePost = predicted.copy()
			kf.errorCovPost = kf.errorCovPre.copy()
			filtered = predicted

		out_state = np.concatenate(
			[kf.statePost.reshape(n, 1), kf.errorCovPost.reshape(n, n)], axis=1
		).astype(np.float32)
		vel = (np.asarray([[filtered[2, 0], filtered[3, 0]]], np.float32) if order >= 1
		       else np.zeros((1, 2), np.float32))
		return io.NodeOutput(
			out_state,
			np.asarray([[filtered[0, 0], filtered[1, 0]]], np.float32),
			np.asarray([[predicted[0, 0], predicted[1, 0]]], np.float32),
			vel,
			meas is not None,
		)


class DetectLineSegments(io.ComfyNode):
	_REFINE = {
		"none (fastest)": cv2.LSD_REFINE_NONE,
		"standard": cv2.LSD_REFINE_STD,
		"advanced (NFA validated)": cv2.LSD_REFINE_ADV,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DetectLineSegments",
			display_name="CV Detect Line Segments (LSD)",
			category=CATEGORY,
			description="Line Segment Detector (cv2.createLineSegmentDetector): "
			            "finds straight edge segments with SUBPIXEL endpoints and a "
			            "measured width, in one pass, with no threshold to tune. "
			            "Unlike cv2_HoughLinesP it works on the gradient field "
			            "directly - no Canny step, no accumulator resolution, and "
			            "each segment is a real, bounded piece of edge rather than a "
			            "vote peak, so it does not fuse two collinear edges into one "
			            "line. LSD is a cv2 class the raw wrappers cannot expose. "
			            "Output is the same Nx4 (x1, y1, x2, y2) layout HoughLinesP "
			            "produces, so 'CV Draw Segments' plugs straight in. "
			            "Finding nothing is a valid result (count = 0, empty arrays), "
			            "never an error.",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Image to search (converted to grayscale internally); "
					        "feed the ORIGINAL image, not an edge map."),
				io.Combo.Input("refine", options=list(cls._REFINE), default="standard",
					tooltip="Refinement level. 'standard' splits arcs and refines "
					        "endpoints; 'advanced' additionally validates each "
					        "segment with the NFA (number of false alarms) test and "
					        "fills the 'nfa' output; 'none' is the raw, fastest pass."),
				io.Float.Input("scale", default=0.8, min=0.1, max=2.0, step=0.05,
					tooltip="Image is downscaled by this factor before detection. "
					        "Below 1.0 suppresses noise and speeds things up; 1.0 "
					        "keeps full resolution and finds more short segments."),
				io.Float.Input("sigma_scale", default=0.6, min=0.1, max=2.0, step=0.05,
					tooltip="Gaussian blur sigma used for the downscale, expressed "
					        "as sigma = sigma_scale / scale."),
				io.Float.Input("quant", default=2.0, min=0.0, max=20.0, step=0.5,
					tooltip="Bound on the gradient quantisation error; raise it on "
					        "noisy or heavily compressed images."),
				io.Float.Input("angle_tolerance", default=22.5, min=1.0, max=90.0, step=0.5,
					tooltip="Gradient angle tolerance in DEGREES for pixels to join "
					        "the same line-support region."),
				io.Float.Input("density_threshold", default=0.7, min=0.0, max=1.0, step=0.05,
					tooltip="Minimum density of aligned pixels inside a segment's "
					        "enclosing rectangle; lower accepts sparser segments."),
				io.Float.Input("min_length", default=0.0, min=0.0, max=10000.0, step=1.0,
					tooltip="Post-filter: drop segments shorter than this many "
					        "pixels. 0 keeps every segment. LSD has no length knob of "
					        "its own, and short segments are what clutters a result.",
					optional=True),
				io.Float.Input("log_eps", default=0.0, min=-10.0, max=10.0, step=0.1,
					tooltip="Detection threshold for the 'advanced' NFA test: "
					        "-log10(NFA) > log_eps. Only used when refine is "
					        "'advanced'.", optional=True, advanced=True),
				io.Int.Input("bins", default=1024, min=1, max=8192,
					tooltip="Number of bins in the gradient-modulus pseudo-ordering.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="segments",
					tooltip="Nx4 float32 (x1, y1, x2, y2) subpixel endpoints - feed "
					        "'CV Draw Segments' directly."),
				NPArray.Output(display_name="widths",
					tooltip="(N,) float32 measured width of each segment in pixels."),
				NPArray.Output(display_name="lengths",
					tooltip="(N,) float32 segment length in pixels - use it with a "
					        "comparison node to keep only the long structural edges."),
				NPArray.Output(display_name="nfa",
					tooltip="(N,) float32 -log10(number of false alarms) per segment "
					        "(higher = more certain). All zeros unless refine is "
					        "'advanced', which is the only mode that computes it."),
				io.Int.Output(display_name="count",
					tooltip="How many segments survived - branch on it with if/else "
					        "for the nothing-found case."),
			],
		)

	@classmethod
	def execute(cls, image, refine, scale, sigma_scale, quant, angle_tolerance,
	            density_threshold, min_length=0.0, log_eps=0.0, bins=1024) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		gray = _to_gray_u8(image)
		lsd = cv2.createLineSegmentDetector(
			cls._REFINE[refine], float(scale), float(sigma_scale), float(quant),
			float(angle_tolerance), float(log_eps), float(density_threshold), int(bins))
		lines, widths, _prec, nfa = lsd.detect(gray)
		# LSD returns None for every output when it finds nothing, and nfa stays
		# None outside REFINE_ADV - both are data, not failures
		segs = (np.zeros((0, 4), np.float32) if lines is None
		        else np.asarray(lines, np.float32).reshape(-1, 4))
		w = (np.zeros((len(segs),), np.float32) if widths is None
		     else np.asarray(widths, np.float32).reshape(-1))
		n = (np.zeros((len(segs),), np.float32) if nfa is None
		     else np.asarray(nfa, np.float32).reshape(-1))
		lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]).astype(np.float32)
		if min_length > 0.0 and len(segs):
			keep = lengths >= float(min_length)
			segs, w, n, lengths = segs[keep], w[keep], n[keep], lengths[keep]
		return io.NodeOutput(segs, w, lengths, n, len(segs))


def _keypoints_output(kps):
	"""Shared (keypoints, points, sizes, responses, count) tuple for detectors."""
	kps = list(kps)
	pts = (np.asarray([k.pt for k in kps], np.float32) if kps
	       else np.zeros((0, 2), np.float32))
	sizes = (np.asarray([k.size for k in kps], np.float32) if kps
	         else np.zeros((0,), np.float32))
	resp = (np.asarray([k.response for k in kps], np.float32) if kps
	        else np.zeros((0,), np.float32))
	return io.NodeOutput(kps, pts, sizes, resp, len(kps))


_KEYPOINT_OUTPUTS = [
	KeyPoints.Output(display_name="keypoints",
		tooltip="cv2.KeyPoint list - draw with 'CV Draw Keypoints', or give "
		        "them descriptors with 'CV Compute Descriptors' to feed "
		        "'CV Match Features'."),
	NPArray.Output(display_name="points",
		tooltip="Nx2 float32 (x, y) centres - the point-set form for 'OpenCV "
		        "Draw Points', k-means, polar conversion, ..."),
	NPArray.Output(display_name="sizes",
		tooltip="(N,) float32 keypoint diameter in pixels."),
	NPArray.Output(display_name="responses",
		tooltip="(N,) float32 detector strength per keypoint - threshold it to "
		        "keep only the strong ones."),
	io.Int.Output(display_name="count",
		tooltip="How many were found; 0 is a valid result, not an error."),
]


class DetectBlobs(io.ComfyNode):
	_COLORS = {
		"dark blobs on a light background": 0,
		"light blobs on a dark background": 255,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DetectBlobs",
			display_name="CV Detect Blobs",
			category=CATEGORY,
			description="cv2.SimpleBlobDetector: finds compact bright-or-dark spots "
			            "by thresholding the image at many levels and keeping the "
			            "regions that stay stable across them, then filtering by "
			            "SHAPE. That shape filtering is what a threshold + "
			            "findContours chain does not give you: area, circularity "
			            "(how round), convexity (how dented) and inertia ratio (how "
			            "elongated) in one pass. Classic uses: counting cells, dots, "
			            "holes, bubbles or calibration targets. SimpleBlobDetector is "
			            "a cv2 class the raw wrappers cannot expose. It produces no "
			            "descriptors - run 'CV Compute Descriptors' if the blobs "
			            "must be matched. Finding nothing is a valid result "
			            "(count = 0).",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Image to search (converted to grayscale internally)."),
				io.Combo.Input("blob_color", options=list(cls._COLORS),
					default="dark blobs on a light background",
					tooltip="Polarity of the blobs. Pick the wrong one and the "
					        "detector returns nothing - this is the first thing to "
					        "flip when count is 0."),
				io.Float.Input("min_area", default=25.0, min=0.0, max=1e7, step=10.0,
					tooltip="Smallest blob area in PIXELS. 0 disables the lower "
					        "bound. Raise it to ignore speckle noise."),
				io.Float.Input("max_area", default=5000.0, min=0.0, max=1e7, step=100.0,
					tooltip="Largest blob area in pixels. 0 means NO upper limit "
					        "(the area filter is then one-sided)."),
				io.Float.Input("min_circularity", default=0.0, min=0.0, max=1.0, step=0.05,
					tooltip="Minimum 4*pi*area / perimeter^2: 1.0 is a perfect "
					        "circle, ~0.79 a square. 0 disables the filter."),
				io.Float.Input("min_convexity", default=0.0, min=0.0, max=1.0, step=0.05,
					tooltip="Minimum area / convex-hull area: 1.0 is fully convex, "
					        "lower values allow dents and crescents. 0 disables the "
					        "filter."),
				io.Float.Input("min_inertia", default=0.0, min=0.0, max=1.0, step=0.05,
					tooltip="Minimum inertia ratio (short axis / long axis): 1.0 is "
					        "a circle, near 0 is a thin streak. 0 disables the "
					        "filter - raise it to reject elongated smears."),
				io.Float.Input("min_threshold", default=50.0, min=0.0, max=255.0, step=5.0,
					tooltip="First grayscale threshold of the sweep.",
					optional=True, advanced=True),
				io.Float.Input("max_threshold", default=220.0, min=0.0, max=255.0, step=5.0,
					tooltip="Last grayscale threshold of the sweep.",
					optional=True, advanced=True),
				io.Float.Input("threshold_step", default=10.0, min=1.0, max=128.0, step=1.0,
					tooltip="Step between successive thresholds; smaller is more "
					        "thorough and slower.", optional=True, advanced=True),
				io.Int.Input("min_repeatability", default=2, min=1, max=100,
					tooltip="A blob must appear at this many consecutive thresholds "
					        "to be reported - the stability test.",
					optional=True, advanced=True),
				io.Float.Input("min_distance", default=10.0, min=0.0, max=10000.0, step=1.0,
					tooltip="Blobs closer together than this (pixels) are merged "
					        "into one.", optional=True, advanced=True),
			],
			outputs=_KEYPOINT_OUTPUTS,
		)

	@classmethod
	def execute(cls, image, blob_color, min_area, max_area, min_circularity,
	            min_convexity, min_inertia, min_threshold=50.0, max_threshold=220.0,
	            threshold_step=10.0, min_repeatability=2,
	            min_distance=10.0) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		gray = _to_gray_u8(image)
		p = cv2.SimpleBlobDetector_Params()
		p.minThreshold = float(min_threshold)
		p.maxThreshold = float(max_threshold)
		p.thresholdStep = float(threshold_step)
		p.minRepeatability = int(min_repeatability)
		p.minDistBetweenBlobs = float(min_distance)
		p.filterByColor = True
		p.blobColor = cls._COLORS[blob_color]
		# Each shape filter is driven by its own value instead of a separate
		# "enabled" boolean: 0 (or 0 max_area) simply means "do not filter".
		# SimpleBlobDetectorImpl::validateParameters rejects a 0 min* even when
		# the matching filterBy* is off, so the min/max are only written when
		# the filter is actually enabled - cv2's own defaults stand otherwise.
		big = float(np.finfo(np.float32).max)
		p.filterByArea = bool(min_area > 0.0 or max_area > 0.0)
		if p.filterByArea:
			p.minArea = float(min_area) if min_area > 0.0 else 1.0
			p.maxArea = float(max_area) if max_area > 0.0 else big
		p.filterByCircularity = bool(min_circularity > 0.0)
		if p.filterByCircularity:
			p.minCircularity, p.maxCircularity = float(min_circularity), big
		p.filterByConvexity = bool(min_convexity > 0.0)
		if p.filterByConvexity:
			p.minConvexity, p.maxConvexity = float(min_convexity), big
		p.filterByInertia = bool(min_inertia > 0.0)
		if p.filterByInertia:
			p.minInertiaRatio, p.maxInertiaRatio = float(min_inertia), big
		return _keypoints_output(cv2.SimpleBlobDetector_create(p).detect(gray))


class DetectCorners(io.ComfyNode):
	_FAST = "FAST (corner segment test)"
	_SHI = "GFTT (Shi-Tomasi)"
	_HARRIS = "GFTT (Harris)"
	_FAST_TYPES = {
		"9/16 (default)": cv2.FastFeatureDetector_TYPE_9_16,
		"7/12": cv2.FastFeatureDetector_TYPE_7_12,
		"5/8 (most permissive)": cv2.FastFeatureDetector_TYPE_5_8,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DetectCorners",
			display_name="CV Detect Corners",
			category=CATEGORY,
			description="Descriptor-free corner detectors: cv2.FastFeatureDetector "
			            "and cv2.GFTTDetector (Shi-Tomasi or Harris scoring). These "
			            "are cv2 CLASSES - the raw cv2_goodFeaturesToTrack wrapper "
			            "returns bare points, while these return proper KEYPOINTS "
			            "with a response, so they chain into 'CV Draw Keypoints', "
			            "'CV Compute Descriptors' and the matching nodes, and "
			            "FAST has no function form at all. Use FAST when you need "
			            "thousands of corners per millisecond (tracking, SLAM "
			            "front-ends) and GFTT when you want a bounded, well-spread "
			            "set (the classic KLT seed). Finding nothing is a valid "
			            "result (count = 0).",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Image to search (converted to grayscale internally)."),
				io.Combo.Input("detector", options=[cls._FAST, cls._SHI, cls._HARRIS],
					default=cls._SHI,
					tooltip="FAST: very fast, unbounded count, driven by "
					        "'threshold'. GFTT Shi-Tomasi: bounded, evenly spread, "
					        "driven by max_features/quality_level/min_distance. GFTT "
					        "Harris: same but scored with the Harris measure "
					        "(harris_k), which prefers sharper corners."),
				io.Int.Input("threshold", default=10, min=1, max=255,
					tooltip="FAST only: how much brighter/darker the arc of pixels "
					        "around a candidate must be. Lower = many more corners."),
				io.Boolean.Input("non_max_suppression", default=True,
					tooltip="FAST only: keep only the locally strongest corner "
					        "instead of every pixel that passes the test."),
				io.Int.Input("max_features", default=500, min=1, max=100000,
					tooltip="GFTT only: keep at most this many of the strongest "
					        "corners."),
				io.Float.Input("quality_level", default=0.01, min=0.0001, max=1.0, step=0.001,
					tooltip="GFTT only: minimum corner score as a FRACTION of the "
					        "best corner's score. 0.01 keeps everything within 1% of "
					        "the best; raise it to keep only strong corners."),
				io.Float.Input("min_distance", default=10.0, min=0.0, max=1000.0, step=1.0,
					tooltip="GFTT only: minimum spacing in pixels between kept "
					        "corners - this is what spreads them over the image."),
				io.Combo.Input("fast_type", options=list(cls._FAST_TYPES),
					default="9/16 (default)",
					tooltip="FAST only: how many of the 16 ring pixels must agree. "
					        "9/16 is the standard; shorter arcs fire more often.",
					optional=True, advanced=True),
				io.Int.Input("block_size", default=3, min=1, max=31,
					tooltip="GFTT only: neighbourhood size used to compute the "
					        "corner score (forced odd).", optional=True, advanced=True),
				io.Float.Input("harris_k", default=0.04, min=0.001, max=0.5, step=0.01,
					tooltip="GFTT Harris only: the free parameter k of the Harris "
					        "response det(M) - k*trace(M)^2.",
					optional=True, advanced=True),
			],
			outputs=_KEYPOINT_OUTPUTS,
		)

	@classmethod
	def execute(cls, image, detector, threshold, non_max_suppression, max_features,
	            quality_level, min_distance, fast_type="9/16 (default)", block_size=3,
	            harris_k=0.04) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		gray = _to_gray_u8(image)
		if detector == cls._FAST:
			det = cv2.FastFeatureDetector_create(
				int(threshold), bool(non_max_suppression), cls._FAST_TYPES[fast_type])
		else:
			det = cv2.GFTTDetector_create(
				maxCorners=int(max_features), qualityLevel=float(quality_level),
				minDistance=float(min_distance), blockSize=max(1, int(block_size) | 1),
				useHarrisDetector=(detector == cls._HARRIS), k=float(harris_k))
		return _keypoints_output(det.detect(gray))


class ComputeDescriptors(io.ComfyNode):
	_DESCRIPTORS = {
		"ORB (binary, fast)": lambda: cv2.ORB_create(),
		"SIFT (float, robust)": lambda: cv2.SIFT_create(),
		"BRISK (binary, scale-aware)": lambda: _feature2d("BRISK")(),
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ComputeDescriptors",
			display_name="CV Compute Descriptors",
			category=CATEGORY,
			description="Describes keypoints that were found by a DESCRIPTOR-FREE "
			            "detector ('CV Detect Corners', 'CV Detect Blobs', "
			            "annotated points...) so they can be matched. This is the "
			            "compute()-only half of the cv2 Feature2D class API, which no "
			            "raw wrapper exposes: detection and description are separate "
			            "steps in OpenCV, and splitting them lets you pair any "
			            "detector with any descriptor (FAST corners + SIFT "
			            "descriptors, blobs + ORB, ...). Wire the outputs into "
			            "'CV Match Features' exactly like 'CV Detect "
			            "Features'. NOTE the descriptor DROPS keypoints too close to "
			            "the border, so always use the 'keypoints' output here - not "
			            "the original list - or the descriptors and keypoints go out "
			            "of step.",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="The image the keypoints were detected in (converted to "
					        "grayscale internally)."),
				KeyPoints.Input("keypoints",
					tooltip="Keypoints to describe, from any detector node."),
				io.Combo.Input("descriptor", options=list(cls._DESCRIPTORS),
					default="ORB (binary, fast)",
					tooltip="ORB/BRISK give binary descriptors (match with Hamming), "
					        "SIFT gives 128-D float descriptors (match with L2). The "
					        "matcher node must use the norm that goes with the "
					        "descriptor."),
			],
			outputs=[
				KeyPoints.Output(display_name="keypoints",
					tooltip="The keypoints that COULD be described (border keypoints "
					        "are dropped) - row-aligned with the descriptors."),
				NPArray.Output(display_name="descriptors",
					tooltip="NxD descriptor matrix; uint8 for ORB/BRISK, float32 for "
					        "SIFT. Empty with the right dtype when nothing survives."),
				io.Int.Output(display_name="count",
					tooltip="How many keypoints kept a descriptor."),
			],
		)

	@classmethod
	def execute(cls, image, keypoints, descriptor) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		gray = _to_gray_u8(image)
		ext = cls._DESCRIPTORS[descriptor]()
		kps, desc = ext.compute(gray, list(keypoints))
		kps = list(kps or [])
		if desc is None:  # every keypoint was rejected: keep the right dtype
			dtype = np.uint8 if ext.descriptorType() == cv2.CV_8U else np.float32
			desc = np.zeros((0, ext.descriptorSize()), dtype)
		return io.NodeOutput(kps, np.asarray(desc), len(kps))


class GeneralizedHough(io.ComfyNode):
	_BALLARD = "Ballard (position only, fast)"
	_GUIL = "Guil (position + scale + rotation, slow)"

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_GeneralizedHough",
			display_name="CV Generalized Hough (Shape Template)",
			category=CATEGORY,
			description="Finds an arbitrary SHAPE given as a template image "
			            "(cv2.createGeneralizedHoughBallard / ...Guil). Where "
			            "cv2_HoughCircles/HoughLines only fit shapes with a formula, "
			            "the generalized transform builds an R-table from the "
			            "template's edge points and lets every scene edge vote for "
			            "where the shape's reference point must be - so it finds any "
			            "outline you can draw, and it does it from EDGES, which means "
			            "partial occlusion and texture/colour changes do not matter "
			            "(unlike cv2_matchTemplate, which correlates raw pixels). "
			            "Ballard finds translations only. Guil also searches scale and "
			            "rotation, which multiplies the accumulator by every scale x "
			            "angle step - keep the ranges TIGHT or it takes minutes. "
			            "These are cv2 classes the raw wrappers cannot expose. Zero "
			            "detections is a valid result (found=false).",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Scene to search (converted to grayscale internally; "
					        "edges are computed inside with the canny_* thresholds)."),
				polytype.poly_input("template", types=polytype.IMAGE_ONLY,
					tooltip="Image of the shape to find. Its edges become the "
					        "R-table, so a clean, tightly cropped outline works far "
					        "better than a busy crop."),
				io.Combo.Input("method", options=[cls._BALLARD, cls._GUIL],
					default=cls._BALLARD,
					tooltip="Ballard: translation only, one accumulator, fast. Guil: "
					        "also scale and rotation, orders of magnitude slower - "
					        "narrow min/max scale and angle before using it."),
				io.Int.Input("canny_low", default=50, min=1, max=255,
					tooltip="Lower Canny hysteresis threshold used for BOTH the "
					        "template and the scene edges. Lower it if the shape's "
					        "outline is faint."),
				io.Int.Input("canny_high", default=100, min=1, max=255,
					tooltip="Upper Canny hysteresis threshold; roughly 2-3x the lower "
					        "one."),
				io.Float.Input("min_distance", default=10.0, min=1.0, max=1000.0, step=1.0,
					tooltip="Minimum distance in pixels between two reported "
					        "detections - this is what stops one shape being reported "
					        "as a cluster of near-identical hits."),
				io.Int.Input("votes_threshold", default=30, min=1, max=100000,
					tooltip="Ballard only: how many edge points must vote for a "
					        "position before it counts. LOWER it if nothing is found, "
					        "raise it if the scene is full of false hits. It scales "
					        "with the template's edge-pixel count."),
				io.Float.Input("dp", default=1.0, min=0.5, max=10.0, step=0.1,
					tooltip="Accumulator resolution divisor: 1.0 = same resolution as "
					        "the image, 2.0 = half (coarser, faster, more tolerant).",
					optional=True, advanced=True),
				io.Int.Input("levels", default=360, min=10, max=3600,
					tooltip="Number of gradient-direction bins in the R-table.",
					optional=True, advanced=True),
				io.Float.Input("min_scale", default=0.8, min=0.05, max=10.0, step=0.05,
					tooltip="Guil only: smallest template scale to search.",
					optional=True, advanced=True),
				io.Float.Input("max_scale", default=1.2, min=0.05, max=10.0, step=0.05,
					tooltip="Guil only: largest template scale to search.",
					optional=True, advanced=True),
				io.Float.Input("scale_step", default=0.1, min=0.01, max=1.0, step=0.01,
					tooltip="Guil only: scale increment. Every extra step costs a "
					        "full pass.", optional=True, advanced=True),
				io.Float.Input("min_angle", default=0.0, min=0.0, max=360.0, step=1.0,
					tooltip="Guil only: smallest rotation in DEGREES to search.",
					optional=True, advanced=True),
				io.Float.Input("max_angle", default=360.0, min=0.0, max=360.0, step=1.0,
					tooltip="Guil only: largest rotation in degrees.",
					optional=True, advanced=True),
				io.Float.Input("angle_step", default=10.0, min=0.1, max=90.0, step=0.5,
					tooltip="Guil only: rotation increment in degrees. 1 degree over "
					        "the full circle is 360 passes - start coarse.",
					optional=True, advanced=True),
				io.Int.Input("position_votes", default=100, min=1, max=100000,
					tooltip="Guil only: vote threshold of the POSITION stage.",
					optional=True, advanced=True),
				io.Int.Input("scale_votes", default=100, min=1, max=100000,
					tooltip="Guil only: vote threshold of the SCALE stage.",
					optional=True, advanced=True),
				io.Int.Input("angle_votes", default=10000, min=1, max=1000000,
					tooltip="Guil only: vote threshold of the ANGLE stage.",
					optional=True, advanced=True),
			],
			outputs=[
				io.Boolean.Output(display_name="found",
					tooltip="True when at least one instance was detected - branch "
					        "on it with 'Basic data handling: IfElse'."),
				NPArray.Output(display_name="positions",
					tooltip="Nx2 float32 centres of the detected shapes - feed "
					        "'CV Draw Points'."),
				NPArray.Output(display_name="scales",
					tooltip="(N,) float32 template scale per detection (all 1.0 for "
					        "Ballard, which does not search scale)."),
				NPArray.Output(display_name="angles",
					tooltip="(N,) float32 rotation in DEGREES per detection (all 0 "
					        "for Ballard)."),
				NPArray.Output(display_name="votes",
					tooltip="Nx3 int32 accumulator votes (position, scale, angle) - "
					        "the detection confidence. Sort or threshold on column 0 "
					        "to keep the strongest hits."),
				io.BoundingBox.Output(display_name="bboxes",
					tooltip="One {x, y, width, height, score, label} dict per "
					        "detection: the template's size scaled by 'scales', "
					        "centred on the position - feed the core 'Draw BBoxes'."),
				io.Int.Output(display_name="count",
					tooltip="How many instances were detected; 0 is valid, not an "
					        "error."),
			],
		)

	@classmethod
	def execute(cls, image, template, method, canny_low, canny_high, min_distance,
	            votes_threshold, dp=1.0, levels=360, min_scale=0.8, max_scale=1.2,
	            scale_step=0.1, min_angle=0.0, max_angle=360.0, angle_step=10.0,
	            position_votes=100, scale_votes=100, angle_votes=10000) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		template, _ = polytype.resolve(template)
		scene = _to_gray_u8(image)
		tpl = _to_gray_u8(template)
		th, tw = tpl.shape[:2]

		if method == cls._GUIL:
			hough = cv2.createGeneralizedHoughGuil()
			hough.setMinScale(float(min_scale))
			hough.setMaxScale(float(max_scale))
			hough.setScaleStep(float(scale_step))
			hough.setMinAngle(float(min_angle))
			hough.setMaxAngle(float(max_angle))
			hough.setAngleStep(float(angle_step))
			hough.setPosThresh(int(position_votes))
			hough.setScaleThresh(int(scale_votes))
			hough.setAngleThresh(int(angle_votes))
		else:
			hough = cv2.createGeneralizedHoughBallard()
			hough.setVotesThreshold(int(votes_threshold))
		hough.setCannyLowThresh(int(canny_low))
		hough.setCannyHighThresh(int(canny_high))
		hough.setMinDist(float(min_distance))
		hough.setDp(float(dp))
		hough.setLevels(int(levels))
		hough.setTemplate(tpl)

		empty = (False, np.zeros((0, 2), np.float32), np.zeros((0,), np.float32),
		         np.zeros((0,), np.float32), np.zeros((0, 3), np.int32), [], 0)
		try:
			positions, votes = hough.detect(scene)
		except cv2.error:
			# an empty accumulator / degenerate template is "nothing found", and a
			# detector must never halt the workflow
			return io.NodeOutput(*empty)
		if positions is None or not np.size(positions):
			return io.NodeOutput(*empty)

		pos = np.asarray(positions, np.float32).reshape(-1, 4)
		vts = (np.asarray(votes, np.int32).reshape(-1, 3) if votes is not None
		       and np.size(votes) else np.zeros((len(pos), 3), np.int32))
		centres = pos[:, :2].copy()
		scales = pos[:, 2].copy()
		angles = pos[:, 3].copy()
		boxes = []
		for (cx, cy), s, v in zip(centres, scales, vts):
			w, h = tw * float(s), th * float(s)
			boxes.append({"x": int(round(cx - w / 2.0)), "y": int(round(cy - h / 2.0)),
			              "width": int(round(w)), "height": int(round(h)),
			              "score": float(v[0]), "label": "shape"})
		return io.NodeOutput(True, centres, scales, angles, vts, boxes, len(pos))


def _flow_outputs(flow):
	"""(flow, magnitude, angle) triple shared by the dense-flow nodes."""
	flow = np.asarray(flow, np.float32)
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	return io.NodeOutput(flow, mag.astype(np.float32), ang.astype(np.float32))


_FLOW_OUTPUT_SCHEMA = [
	NPArray.Output(display_name="flow",
		tooltip="HxWx2 float32 (dx, dy) displacement per pixel."),
	NPArray.Output(display_name="magnitude",
		tooltip="HxW float32 motion magnitude in pixels."),
	NPArray.Output(display_name="angle",
		tooltip="HxW float32 motion direction in RADIANS - feed 'CV Flow To "
		        "Color' in its default radians mode."),
]


class OpticalFlowDIS(io.ComfyNode):
	_PRESETS = {
		"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
		"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
		"medium (most accurate)": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_OpticalFlowDIS",
			display_name="CV Optical Flow (DIS)",
			category=CATEGORY,
			description="Dense Inverse Search optical flow (cv2.DISOpticalFlow) - "
			            "the modern replacement for 'CV Optical Flow "
			            "(Farneback)'. It matches small patches coarse-to-fine, "
			            "propagates the winners to their neighbours and then runs a "
			            "variational refinement, which gives noticeably sharper "
			            "motion boundaries AND runs faster than Farneback at "
			            "comparable quality. DISOpticalFlow is a cv2 class the raw "
			            "wrappers cannot expose. Start from a preset, then override "
			            "individual parameters if needed. Connect 'init_flow' (e.g. "
			            "the previous frame pair's flow) to warm-start a video "
			            "sequence - large motions converge far better that way.",
			inputs=[
				polytype.poly_input("frame_a", types=polytype.IMAGE_ONLY,
					tooltip="First frame (earlier in time); converted to grayscale."),
				polytype.poly_input("frame_b", types=polytype.IMAGE_ONLY,
					tooltip="Second frame; resized to frame_a if the sizes differ."),
				io.Combo.Input("preset", options=list(cls._PRESETS),
					default="medium (most accurate)",
					tooltip="Speed/quality preset; it presets finest_scale, patch "
					        "size/stride and the iteration counts. 'ultrafast' is "
					        "real-time-video fast, 'medium' is the best quality."),
				io.Int.Input("finest_scale", default=-1, min=-1, max=6,
					tooltip="Finest pyramid level actually computed. -1 keeps the "
					        "preset's value. 0 = full resolution (sharpest and "
					        "slowest); raising it blurs and speeds up."),
				io.Int.Input("gradient_iterations", default=-1, min=-1, max=100,
					tooltip="Gradient-descent iterations per patch. -1 keeps the "
					        "preset's value; more = more accurate, slower."),
				io.Int.Input("refinement_iterations", default=-1, min=-1, max=100,
					tooltip="Variational refinement iterations run after the patch "
					        "search. -1 keeps the preset's value; 0 turns the "
					        "refinement off entirely (rawer, blockier flow)."),
				NPArray.Input("init_flow", optional=True,
					tooltip="Optional HxWx2 flow to start from - pass the previous "
					        "frame pair's flow when processing a video. It is copied, "
					        "never written to, so the upstream node's cached array is "
					        "safe."),
				io.Int.Input("patch_size", default=-1, min=-1, max=64,
					tooltip="Side of the matched patch in pixels. -1 keeps the "
					        "preset's value (8 is the usual).",
					optional=True, advanced=True),
				io.Int.Input("patch_stride", default=-1, min=-1, max=32,
					tooltip="Spacing between patches. -1 keeps the preset's value; "
					        "smaller = denser and slower.",
					optional=True, advanced=True),
				io.Boolean.Input("spatial_propagation", default=True,
					tooltip="Seed each patch from its already-solved neighbour. "
					        "Almost always worth it - the main reason DIS beats a "
					        "plain patch search.", optional=True, advanced=True),
				io.Boolean.Input("mean_normalization", default=True,
					tooltip="Normalise patch means before matching, which makes the "
					        "match robust to brightness changes between frames.",
					optional=True, advanced=True),
			],
			outputs=_FLOW_OUTPUT_SCHEMA,
		)

	@classmethod
	def execute(cls, frame_a, frame_b, preset, finest_scale, gradient_iterations,
	            refinement_iterations, init_flow=None, patch_size=-1, patch_stride=-1,
	            spatial_propagation=True, mean_normalization=True) -> io.NodeOutput:
		frame_a, _ = polytype.resolve(frame_a)
		frame_b, _ = polytype.resolve(frame_b)
		prev = _to_gray_u8(frame_a)
		nxt = _to_gray_u8(frame_b)
		if nxt.shape != prev.shape:
			nxt = cv2.resize(nxt, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
		dis = cv2.DISOpticalFlow_create(cls._PRESETS[preset])
		# -1 means "leave whatever the preset chose" - never write a bogus value
		if finest_scale >= 0:
			dis.setFinestScale(int(finest_scale))
		if gradient_iterations >= 0:
			dis.setGradientDescentIterations(int(gradient_iterations))
		if refinement_iterations >= 0:
			dis.setVariationalRefinementIterations(int(refinement_iterations))
		if patch_size >= 0:
			dis.setPatchSize(int(patch_size))
		if patch_stride >= 0:
			dis.setPatchStride(int(patch_stride))
		dis.setUseSpatialPropagation(bool(spatial_propagation))
		dis.setUseMeanNormalization(bool(mean_normalization))

		init = None
		if init_flow is not None and np.asarray(init_flow).size:
			init = np.asarray(init_flow, np.float32)
			if init.shape[:2] != prev.shape[:2]:
				init = cv2.resize(init, (prev.shape[1], prev.shape[0]))
			# calc() writes into the flow argument - copy so an upstream node's
			# cached output is never corrupted (same rule as lowlevel.py)
			init = np.ascontiguousarray(init.copy())
		return _flow_outputs(dis.calc(prev, nxt, init))


class RefineFlowVariational(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_RefineFlowVariational",
			display_name="CV Refine Flow (Variational)",
			category=CATEGORY,
			description="Polishes an EXISTING dense flow field by minimising the "
			            "classic brightness-constancy + smoothness energy "
			            "(cv2.VariationalRefinement) - the same refinement DIS runs "
			            "internally, exposed as its own step so it can be applied to "
			            "any flow: a Farneback field, a DIS field with refinement "
			            "turned off, a flow you built yourself, or the same flow "
			            "twice. It sharpens motion boundaries and fills small holes "
			            "but cannot recover motion the input flow missed entirely - "
			            "it refines, it does not search. VariationalRefinement is a "
			            "cv2 class the raw wrappers cannot expose. The input flow is "
			            "copied before refining, so the upstream node's cached array "
			            "is never modified.",
			inputs=[
				polytype.poly_input("frame_a", types=polytype.IMAGE_ONLY,
					tooltip="First frame - the same frame_a the flow was computed "
					        "from; converted to grayscale."),
				polytype.poly_input("frame_b", types=polytype.IMAGE_ONLY,
					tooltip="Second frame; resized to frame_a if the sizes differ."),
				NPArray.Input("flow",
					tooltip="HxWx2 float32 flow to refine (from 'CV Optical Flow "
					        "(DIS)' or '(Farneback)'). Resized to the frame if the "
					        "sizes differ."),
				io.Int.Input("iterations", default=5, min=1, max=100,
					tooltip="Outer fixed-point iterations; more = smoother and "
					        "closer to the energy minimum, and slower."),
				io.Float.Input("alpha", default=20.0, min=0.0, max=1000.0, step=1.0,
					tooltip="Weight of the SMOOTHNESS term. Raise it for a smoother, "
					        "more filled-in field; lower it to keep sharp motion "
					        "discontinuities."),
				io.Float.Input("delta", default=5.0, min=0.0, max=1000.0, step=0.5,
					tooltip="Weight of the colour-constancy term (brightness must "
					        "match between frames)."),
				io.Float.Input("gamma", default=10.0, min=0.0, max=1000.0, step=0.5,
					tooltip="Weight of the gradient-constancy term - what keeps the "
					        "refinement working through illumination changes."),
				io.Int.Input("sor_iterations", default=5, min=1, max=100,
					tooltip="Inner successive-over-relaxation iterations per outer "
					        "step.", optional=True, advanced=True),
				io.Float.Input("omega", default=1.6, min=1.0, max=2.0, step=0.05,
					tooltip="Relaxation factor of the SOR solver; 1.6 is the "
					        "standard fast-converging value.",
					optional=True, advanced=True),
			],
			outputs=_FLOW_OUTPUT_SCHEMA,
		)

	@classmethod
	def execute(cls, frame_a, frame_b, flow, iterations, alpha, delta, gamma,
	            sor_iterations=5, omega=1.6) -> io.NodeOutput:
		frame_a, _ = polytype.resolve(frame_a)
		frame_b, _ = polytype.resolve(frame_b)
		prev = _to_gray_u8(frame_a)
		nxt = _to_gray_u8(frame_b)
		if nxt.shape != prev.shape:
			nxt = cv2.resize(nxt, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
		f = np.asarray(flow, np.float32)
		if f.ndim != 3 or f.shape[2] != 2:
			raise ValueError(
				f"flow must be HxWx2, got shape {tuple(f.shape)} - connect a dense "
				"optical-flow node's 'flow' output.")
		if f.shape[:2] != prev.shape[:2]:
			f = cv2.resize(f, (prev.shape[1], prev.shape[0]))
		vr = cv2.VariationalRefinement_create()
		vr.setFixedPointIterations(int(iterations))
		vr.setSorIterations(int(sor_iterations))
		vr.setAlpha(float(alpha))
		vr.setDelta(float(delta))
		vr.setGamma(float(gamma))
		vr.setOmega(float(omega))
		# calc() refines IN PLACE - never hand it the caller's array
		work = np.ascontiguousarray(f.copy())
		return _flow_outputs(vr.calc(prev, nxt, work))


class ThinPlateSplineWarp(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGE_ONLY)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ThinPlateSplineWarp",
			display_name="CV Thin Plate Spline Warp",
			category=CATEGORY,
			description="Non-rigid warp from point correspondences "
			            "(cv2.createThinPlateSplineShapeTransformer). Where a "
			            "homography can only move a plane rigidly (8 parameters, "
			            "straight lines stay straight), a thin plate spline BENDS the "
			            "image so that EVERY 'from' point lands exactly on its 'to' "
			            "point while minimising the bending energy in between - the "
			            "physical analogy is a thin metal sheet forced through the "
			            "control points. Uses: face and object morphing, warping one "
			            "shape onto another after 'CV Match Features', correcting "
			            "non-linear distortion no lens model covers. The shape module "
			            "is a class API no raw wrapper reaches. NOTE the classes must "
			            "come from the create* factory - constructing them directly "
			            "yields an abstract object whose every call throws.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="Image to bend; it is warped so that 'points_from' land "
					        "on 'points_to'."),
				NPArray.Input("points_from",
					tooltip="Nx2 control points in the INPUT image (matched "
					        "keypoints, landmarks, a grid...). At least 3."),
				NPArray.Input("points_to",
					tooltip="Nx2 target positions, row-aligned with points_from - "
					        "where each control point should end up."),
				io.Float.Input("regularization", default=0.0, min=0.0, max=1e6, step=1.0,
					tooltip="Smoothing of the spline. 0 interpolates the control "
					        "points EXACTLY (and follows noisy correspondences "
					        "faithfully); raising it trades exactness for a gentler, "
					        "more rigid warp - the knob to turn when the result "
					        "folds or ripples. IMAGE-SCALE DEPENDENT: cv2 adds this "
					        "to the diagonal of a kernel matrix whose other entries "
					        "are d^2*log(d^2) in PIXELS SQUARED, so useful values "
					        "are large - on a few-hundred-pixel image 100 buys 0.1 "
					        "px of give and ~1e4 is the first setting you can see. "
					        "Scale it with the SQUARE of the image size."),
				NPArray.Input("extra_points", optional=True,
					tooltip="Optional extra Nx2 points to push through the same "
					        "warp (e.g. a bounding box or contour that must follow "
					        "the image). Not used to fit the spline."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="The warped image, same format and size as the input."),
				NPArray.Output(display_name="warped_points",
					tooltip="Nx2 'extra_points' mapped through the spline (empty "
					        "when none were connected)."),
				NPArray.Output(display_name="residual",
					tooltip="Nx2 where 'points_from' actually landed minus where "
					        "they should have - all ~0 at regularization 0, growing "
					        "as you smooth. The honest measure of how much the "
					        "spline gave up."),
			],
		)

	@classmethod
	def execute(cls, image, points_from, points_to, regularization,
	            extra_points=None) -> io.NodeOutput:
		image, tag = polytype.resolve(image)
		src = np.asarray(points_from, np.float32).reshape(-1, 2)
		dst = np.asarray(points_to, np.float32).reshape(-1, 2)
		if len(src) != len(dst):
			raise ValueError(
				f"points_from has {len(src)} point(s) but points_to has {len(dst)} - "
				"they must be row-aligned correspondences.")
		if len(src) < 3:
			raise ValueError(
				f"a thin plate spline needs at least 3 correspondences, got {len(src)}.")

		# The spline's TWO-transformer rule (warpImage is a BACKWARD warp, so the
		# pixels need the to -> from fit while the points need from -> to) lives in
		# warpfit.fit() now, shared with CV_PasteThroughWarp.
		w = warpfit.fit("thin plate spline", src, dst, regularization)
		warped = w.warp_image(image)
		warped_pts = (w.forward(extra_points)
			      if extra_points is not None and np.asarray(extra_points).size
			      else np.zeros((0, 2), np.float32))
		residual = warpfit.residual(w, src, dst)
		return io.NodeOutput(polytype.restore(warped, tag), warped_pts, residual)


class AffineWarp(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGE_ONLY)
	# The descriptive labels rule 6 wants, mapped onto warpfit's model names.
	# warpfit.ALIASES carries the same pairs so a saved workflow's label still
	# resolves wherever a warpfit model string is expected.
	_MODELS = {
		"full affine - 6 dof (rotation, scale, shear, translation)": "affine",
		"similarity - 4 dof (rotation, uniform scale, translation)": "similarity",
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_AffineWarp",
			display_name="CV Affine Shape Warp",
			category=CATEGORY,
			description="RIGID warp fitted to point correspondences "
			            "(cv2.createAffineTransformer) - the counterpart of "
			            "'CV Thin Plate Spline Warp'. Where the spline BENDS "
			            "the image so every control point lands exactly on its "
			            "target, an affine fit has only 6 (or 4) parameters for the "
			            "whole image: it keeps straight lines straight and parallel "
			            "lines parallel, and solves the correspondences in the "
			            "least-squares sense. That is what makes it the right tool "
			            "when the correspondences are NOISY (a spline would chase "
			            "every wrong match) or when the true motion really is rigid, "
			            "and the wrong tool for a genuinely non-linear deformation - "
			            "'residual' is what tells the two cases apart. The shape "
			            "module is a class API no raw wrapper reaches, and the "
			            "classes must come from the create* factory. Unlike the "
			            "spline's, this warpImage is a FORWARD warp, so one fit "
			            "drives both the pixels and the points.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="Image to warp; it is transformed so that 'points_from' "
					        "move towards 'points_to'. Areas pulled in from outside "
					        "the frame come out black (BORDER_CONSTANT)."),
				NPArray.Input("points_from",
					tooltip="Nx2 control points in the INPUT image (matched "
					        "keypoints, landmarks, a grid...). At least 3."),
				NPArray.Input("points_to",
					tooltip="Nx2 target positions, row-aligned with points_from - "
					        "where each control point should end up. With more than "
					        "3 rows the fit is least-squares, so they need not be "
					        "reachable exactly."),
				io.Combo.Input("model", options=list(cls._MODELS),
					tooltip="How much freedom the fit gets. 'full affine' is the "
					        "usual 6-dof least-squares affine (it can shear and "
					        "scale the axes differently). 'similarity' drops shear "
					        "and non-uniform scale, keeping angles and shape - a "
					        "stiffer model that resists bad correspondences but "
					        "cannot represent a shear at all (its residual stays "
					        "large no matter how clean the points are)."),
				NPArray.Input("extra_points", optional=True,
					tooltip="Optional extra Nx2 points to push through the same "
					        "transform (e.g. a bounding box or contour that must "
					        "follow the image). Not used to fit it."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="The warped image, same format and size as the input."),
				NPArray.Output(display_name="warped_points",
					tooltip="Nx2 'extra_points' mapped through the transform (empty "
					        "when none were connected)."),
				NPArray.Output(display_name="residual",
					tooltip="Nx2 where 'points_from' actually landed minus where "
					        "they should have. ~0 only when the correspondences ARE "
					        "an affine (a similarity, for the 4-dof model); "
					        "otherwise it is the honest least-squares error - the "
					        "measure of how badly the rigid model fits."),
				NPArray.Output(display_name="matrix",
					tooltip="The fitted transform as a 2x3 float64 matrix, read back "
					        "by pushing the origin and the two unit vectors through "
					        "it - ready for 'cv2.warpAffine' / "
					        "'cv2.transform' or for comparing against a known pose."),
			],
		)

	@classmethod
	def execute(cls, image, points_from, points_to, model,
	            extra_points=None) -> io.NodeOutput:
		image, tag = polytype.resolve(image)
		src = np.asarray(points_from, np.float32).reshape(-1, 2)
		dst = np.asarray(points_to, np.float32).reshape(-1, 2)
		if len(src) != len(dst):
			raise ValueError(
				f"points_from has {len(src)} point(s) but points_to has {len(dst)} - "
				"they must be row-aligned correspondences.")
		if len(src) < 3:
			# cv2 dies inside cv::solve ("can not solve under-determined linear
			# systems") with 2 points, and asserts _matches.size() > 1 with 1
			raise ValueError(
				f"an affine fit needs at least 3 correspondences, got {len(src)}.")

		# ONE transformer, unlike ThinPlateSplineWarp - the rule, and the matrix
		# recovery the transformer does not expose, live in warpfit.
		w = warpfit.fit(cls._MODELS[model], src, dst)
		warped = w.warp_image(image)
		warped_pts = (w.forward(extra_points)
			      if extra_points is not None and np.asarray(extra_points).size
			      else np.zeros((0, 2), np.float32))
		residual = warpfit.residual(w, src, dst)
		matrix = warpfit.matrix_of(w)[:2]
		return io.NodeOutput(polytype.restore(warped, tag), warped_pts, residual, matrix)


class PasteThroughWarp(io.ComfyNode):
	_TEMPLATE = polytype.match_template()

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PasteThroughWarp",
			display_name="CV Paste Through Warp",
			category=CATEGORY,
			description="Warps an image through point correspondences and "
			            "composites it onto a background - the generic 'put THIS "
			            "picture THERE' step. The correspondences can come from "
			            "anywhere: 'CV Annotate Correspondences' (drag them by "
			            "hand, with the background loaded next to the image), "
			            "'CV Match Features', a detector, a grid. Pick the model "
			            "the geometry deserves: a thin plate spline BENDS the image "
			            "onto a curved or cloth-like surface, a homography handles a "
			            "flat one (poster, screen, document), and "
			            "affine/similarity/translation cover the rigid cases. The "
			            "composite is PREMULTIPLIED, so edges do not grow the dark "
			            "outline that masking-then-blending leaves on antialiased "
			            "boundaries. 'CV Quad Warp' in fill mode is the "
			            "4-corner, homography-only special case of this node. "
			            "Failure-tolerant: fewer correspondences than the model "
			            "needs (spline 3, homography 4, affine 3, similarity 2, "
			            "translation 1) or a degenerate fit passes the background "
			            "through untouched with found=false and an empty mask - "
			            "branch on 'found' with an if/else node.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="The picture to bend and paste. The output comes back "
					        "in THIS input's format."),
				NPArray.Input("points_from",
					tooltip="Nx2 control points in 'image' PIXEL coordinates - "
					        "where each point currently is."),
				NPArray.Input("points_to",
					tooltip="Nx2 target positions in 'background' PIXEL "
					        "coordinates, row-aligned with points_from - where the "
					        "same-numbered point must land. 'CV Annotate "
					        "Correspondences' with a background connected outputs "
					        "exactly this pair."),
				io.Combo.Input("model", options=list(warpfit.MODELS),
					default="thin plate spline", tooltip=warpfit.MODEL_TOOLTIP),
				io.Float.Input("regularization", default=0.0, min=0.0, max=1e6,
					step=1.0,
					tooltip="Smoothing of the SPLINE (ignored by the other "
					        "models). 0 interpolates the control points EXACTLY "
					        "and follows noisy correspondences faithfully; raising "
					        "it trades exactness for a gentler, more rigid warp - "
					        "the knob to turn when the result folds or ripples. "
					        "IMAGE-SCALE DEPENDENT: cv2 adds this to the diagonal "
					        "of a kernel matrix whose other entries are "
					        "d^2*log(d^2) in PIXELS SQUARED, so useful values are "
					        "large - on a few-hundred-pixel image 100 buys 0.1 px "
					        "of give and ~1e4 is the first setting you can see. "
					        "Scale it with the SQUARE of the image size."),
				polytype.poly_input("background", types=polytype.IMAGE_ONLY,
					optional=True,
					tooltip="The image pasted INTO: points_to lives in its "
					        "coordinates and the output is its size. Left "
					        "unconnected, the image is warped within its own canvas "
					        "over black, which is the plain warp."),
				polytype.poly_input("content_mask", types=polytype.MASK_ONLY,
					optional=True, advanced=True,
					tooltip="Which part of 'image' is real content: white is "
					        "pasted, black is not, and the mask travels through the "
					        "same warp so the edge stays exact. Unconnected means "
					        "all of it. A mask of a different size is resized, "
					        "never rejected."),
				polytype.poly_input("background_paintable", types=polytype.MASK_ONLY,
					optional=True, advanced=True,
					tooltip="Where on 'background' the paste is allowed to land - "
					        "white is paintable. This is how something in the "
					        "background occludes the pasted image. Unconnected "
					        "means everywhere."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="The composite, background-sized, in the 'image' "
					        "input's format. The background untouched when "
					        "found=false."),
				io.Mask.Output(display_name="mask",
					tooltip="Where the warped content actually landed, after both "
					        "mask gates - the alpha the composite used. All zeros "
					        "when found=false."),
				NPArray.Output(display_name="residual",
					tooltip="Nx2 where 'points_from' actually landed minus where it "
					        "should have. ~0 for the spline at regularization 0; "
					        "for the rigid models it is the honest least-squares "
					        "error, i.e. how badly the model fits. Empty when "
					        "found=false."),
				io.Boolean.Output(display_name="found",
					tooltip="False when there are fewer correspondences than the "
					        "model needs, or when the fit is degenerate."),
			],
		)

	@classmethod
	def execute(cls, image, points_from, points_to, model, regularization,
	            background=None, content_mask=None,
	            background_paintable=None) -> io.NodeOutput:
		image, tag = polytype.resolve(image)
		background, _ = polytype.resolve(background)
		content_mask, _ = polytype.resolve(content_mask)
		background_paintable, _ = polytype.resolve(background_paintable)
		src = warpfit.as_points(points_from)
		dst = warpfit.as_points(points_to)

		# No background is the plain warp: a black canvas the image's own size.
		# A premultiplied composite over black IS the warped image
		# (warped + 0 * (1 - coverage) == warped), so one code path covers both.
		canvas = np.asarray(image if background is None else background)
		if background is None:
			canvas = np.zeros_like(canvas)

		# Failure-tolerant like CV_QuadWarp, unlike CV_ThinPlateSplineWarp.
		# That node takes its points off a wire from another estimator, where
		# "too few" is a wiring mistake worth raising on; these usually arrive
		# from a user's clicks, where "too few" is simply not-annotated-yet.
		# Row misalignment still raises - that is a wiring mistake either way.
		w = warpfit.fit(model, src, dst, regularization)
		if w is None:
			empty = np.zeros(canvas.shape[:2], np.uint8)
			out = np.asarray(image if background is None else background).copy()
			return io.NodeOutput(polytype.restore(out, tag),
			                     imageutils.u8_to_mask_batch([empty]),
			                     np.zeros((0, 2), np.float32), False)

		composite, coverage = warpfit.paste_warped(
			image, canvas, w, content_mask=content_mask,
			paintable_mask=background_paintable)
		return io.NodeOutput(polytype.restore(composite, tag),
		                     imageutils.u8_to_mask_batch([coverage]),
		                     warpfit.residual(w, src, dst), True)


NODES = [
	DetectFeatures, FilterKeypoints, MatchFeatures, FindHomographyRANSAC,
	FindFundamentalMat,
	DrawEpilines, DrawKeypoints, DrawMatches, CVSize, ImageCorners, DrawPolygon,
	DrawText, FlowToColor, OpticalFlowFarneback, DrawFlowGrid, StereoDisparity,
	DetectCircleGrid, CalibrateCircleGrid, StereoRectifyUncalibrated, CalibrateCamera,
	FisheyeCalibrate, FisheyeUndistort,
	OmnidirCalibrate, OmnidirUndistort, OmnidirStereoReconstruct,
	StereoCalibrate, StereoCalibrateCircleGrid,
	DecomposeProjectionMatrix, DecomposeHomography, SolvePnPPose, SolvePnPPoses,
	PoseToMatrix,
	MatrixToPose,
	ArucoDetectMarkers, ArucoDrawMarkers, ArucoBoardPose,
	ArucoBoardImage, CharucoDetect, CalibrateCameraCharuco, RecoverPose,
	TriangulatePoints, RegisterPointClouds3D,
	TrackFeaturesKLT, FindTransformECC, PhaseCorrelateTranslation, TrackWindow,
	ComposePose, VisualOdometry, TrajectoryError,
	StackImages, StitchCanvas, CV_Stitch, CV_Stitch_List, CV_Stitch_Advanced,
	CV_Stitch_Advanced_List,
	ConstantLike, ZeroStrideLike, FeatherBlend,
	StereoCalibrationFromCSV, ScaleHomography, Eye, MatMul,
	StereoDisparityBM, QuasiDenseStereo, KalmanStep, DetectLineSegments, DetectBlobs, DetectCorners,
	ComputeDescriptors, GeneralizedHough, OpticalFlowDIS, RefineFlowVariational,
	ThinPlateSplineWarp, AffineWarp, PasteThroughWarp,
]
