# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Contour and shape-analysis building blocks (NPARRAY-native).

cv2.findContours returns a Python list of point arrays, which the
auto-generated function wrappers in lowlevel.py cannot represent - these
nodes cover it plus the shape-analysis glue used by the document
perspective-correction and clock-reading pipelines. Everything stays
chainable as raw ndarrays.

Custom type:
* CV_CONTOURS - list of contours (each an Nx1x2 int32 ndarray)
"""

import math
from ast import literal_eval

import cv2
import numpy as np

from comfy_api.latest import io

from . import enums, imageutils, polytype, shapepose, warpfit
from . import tuples
from .convert import NPArray
from .features import (LINE_STYLE_TOOLTIP, LINE_STYLES, _make_draw_canvas,
                       _to_bgr_u8, _to_gray_u8, draw_styled_polyline,
                       parse_color)

CATEGORY = "image/CV/contours"


def _thinning_available():
	"""Return cv2.ximgproc only if its thinning API is actually usable.

	The embedded opencv-contrib build is unreliable: cv2.ximgproc can exist
	while cv2.ximgproc.thinning (and the THINNING_* constants) have vanished
	mid-session. Checking `ximgproc is None` is not enough - we must probe the
	actual attributes used below. Returns the module on success, else None.
	"""
	ximgproc = getattr(cv2, "ximgproc", None)
	if ximgproc is None:
		return None
	if not all(hasattr(ximgproc, a) for a in
	           ("thinning", "THINNING_ZHANGSUEN", "THINNING_GUOHALL")):
		return None
	return ximgproc


def _thin_zhang_suen(binary):
	"""Zhang-Suen thinning in numpy -> a 1-pixel-wide 0/1 skeleton.

	Same algorithm as cv2.ximgproc.thinning(THINNING_ZHANGSUEN) and pinned
	against it by test_region_properties_loose_ends, but it is OURS on purpose:
	'loose ends' is a shipped column of 'CV Region Properties' and that
	contrib entry point has gone missing from this install mid-session before.
	The `Skeletonize` node's other fallback - open/erode morphology - is NOT an
	option here: measured on a closed ring it reports up to 30 endpoints where
	the truth is 0, because the morphological skeleton is not connected and
	every fragment contributes two.

	Vectorised over the whole array, so the cost is one pass per removed layer
	(~half the stroke width), not one per pixel.
	"""
	img = (np.asarray(binary) > 0).astype(np.uint8)
	while True:
		removed = False
		for step in (0, 1):
			p = np.pad(img, 1)
			# the 8 neighbours, clockwise from north (cv2/Zhang-Suen naming)
			p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
			p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
			p8, p9 = p[1:-1, :-2], p[:-2, :-2]
			ring = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
			# B = how many neighbours are set, A = how many 0->1 transitions
			# there are going round the ring (A == 1 means removing the pixel
			# cannot disconnect the skeleton)
			b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
			a = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8)
			        for i in range(8))
			corner = ((p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0) if step == 0
			          else (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0))
			drop = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & corner
			if drop.any():
				img[drop] = 0
				removed = True
		if not removed:
			return img


_thinning_warned = False

Contours = io.Custom("CV_CONTOURS")


class FindContours(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FindContours",
			display_name="CV Find Contours",
			category=CATEGORY,
			description="Extracts contours from a binary image "
			            "(cv2.findContours) - feed it edges (cv2.Canny) or a "
			            "threshold/mask image. Color input is converted to "
			            "grayscale; any non-zero pixel counts as foreground. "
			            "Contours come out sorted by area, largest first, with "
			            "specks below min_area dropped. Zero contours is a valid "
			            "result (count = 0), not an error.",
			inputs=[
				polytype.poly_input("image",
					tooltip="Binary image (edges or MASK); non-zero = "
					        "foreground."),
			io.Combo.Input("mode", options=enums.CONTOUR_RETR_SIMPLE.options,
				default=enums.CONTOUR_RETR_SIMPLE.default,
				tooltip="RETR_EXTERNAL: only outermost contours (usual "
				        "choice). RETR_LIST: all, ungrouped; uses the "
				        "TRUCO parallel engine (OpenCV 4.14+)."),
				io.Combo.Input("method", options=enums.CONTOUR_APPROX.options,
					default=enums.CONTOUR_APPROX.default,
					tooltip="CHAIN_APPROX_SIMPLE compresses straight segments "
					        "to their endpoints; CHAIN_APPROX_NONE keeps every "
					        "boundary pixel."),
				io.Float.Input("min_area", default=0.0, min=0.0, max=1e9, step=10.0,
					tooltip="Drop contours whose area (px²) is below this - "
					        "filters noise specks. 0 keeps everything."),
			],
			outputs=[
				Contours.Output(display_name="contours"),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, image, mode, method, min_area) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		gray = _to_gray_u8(image)
		try:
			found, _ = cv2.findContours(gray, enums.CONTOUR_RETR_SIMPLE.resolve(mode),
			                            enums.CONTOUR_APPROX.resolve(method))
		except cv2.error as e:
			raise RuntimeError(
				f"cv2.findContours failed on ndarray{gray.shape} {gray.dtype} "
				f"with {mode}: {e}"
			) from e
		contours = [c for c in found if cv2.contourArea(c) >= min_area]
		contours.sort(key=cv2.contourArea, reverse=True)
		return io.NodeOutput(contours, len(contours))


class DrawContours(io.ComfyNode):
	_TEMPLATE = polytype.match_template()

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawContours",
			display_name="CV Draw Contours",
			category=CATEGORY,
			description="Debug helper: draws contours on the image "
			            "(cv2.drawContours). Two contour sets over one image "
			            "should differ in 'line_style' as well as color, so the "
			            "distinction does not depend on hue. "
			            "Drawing nothing (0 contours) is "
			            "fine - the image passes through unchanged. The output "
			            "comes back in the same format as the image input "
			            "(IMAGE in, IMAGE out; MASK in, MASK out).",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made; the input "
					        "is not modified)."),
				Contours.Input("contours",
					tooltip="Contours to draw, e.g. from 'CV Find Contours'."),
				io.String.Input("color", default="(0, 255, 0)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(0, 255, 0)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated - so on a "
					        "single-channel MASK canvas a BGR tuple keeps only "
					        "its BLUE component ('(0, 255, 0)' becomes 0, i.e. "
					        "draws nothing visible). Use a single value like "
					        "'255' for a MASK."),
				io.Int.Input("thickness", default=2, min=1, max=64,
					tooltip="Outline width in pixels (ignored when 'fill' is on)."),
				io.Boolean.Input("fill", default=False,
					tooltip="Fill the contours instead of outlining them."),
				io.Int.Input("max_draw", default=0, min=0, max=10000,
					tooltip="Draw at most this many contours (largest first, "
					        "as sorted by 'CV Find Contours'). 0 = all."),
				io.Combo.Input("line_style", options=list(LINE_STYLES),
					default="solid", optional=True,
					tooltip=LINE_STYLE_TOOLTIP + " Ignored when 'fill' is on."),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "image",
				tooltip="Same format as the image input.")],
		)

	@classmethod
	def execute(cls, source, contours, color, thickness, fill, max_draw,
	            line_style="solid") -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		bgr = parse_color(color, n_channels)
		drawn = list(contours[:max_draw] if max_draw else contours)
		if drawn:
			if fill or line_style == "solid":
				cv2.drawContours(canvas, drawn, -1, bgr,
				                 cv2.FILLED if fill else thickness, cv2.LINE_AA)
			else:
				for contour in drawn:
					draw_styled_polyline(canvas, contour, bgr, thickness,
					                     line_style, closed=True)
		return io.NodeOutput(polytype.restore(canvas, tag))


class SelectContour(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SelectContour",
			display_name="CV Select Contour",
			category=CATEGORY,
			description="Picks a single contour by rank from a contour set - "
			            "'the largest one', 'the third largest'. 'CV Find "
			            "Contours' emits them sorted by area (largest first), so "
			            "index 0 is the biggest blob. The output is still a "
			            "CV_CONTOURS set (with one contour), ready for any other "
			            "contour node, e.g. as the reference of 'CV Filter "
			            "Contours By Shape'. Failure-tolerant: an index past the "
			            "end yields found=false and an empty set instead of an "
			            "error.",
			inputs=[
				Contours.Input("contours", tooltip="From 'CV Find Contours'."),
				io.Int.Input("index", default=0, min=0, max=100000,
					tooltip="Rank in the incoming order (Find Contours sorts by "
					        "area, largest first - 0 = biggest)."),
			],
			outputs=[
				Contours.Output(display_name="contour",
					tooltip="A 1-contour set (empty if not found)."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, contours, index) -> io.NodeOutput:
		if index >= len(contours):
			return io.NodeOutput([], False)
		return io.NodeOutput([contours[index]], True)


def _contour_property(contour: np.ndarray, prop: str) -> float:
	area = cv2.contourArea(contour)
	if prop.startswith("area"):
		return float(area)
	if prop.startswith("perimeter"):
		return float(cv2.arcLength(contour, True))
	if prop.startswith("circularity"):
		perimeter = cv2.arcLength(contour, True)
		return float(4.0 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
	if prop.startswith("aspect ratio"):
		_, _, w, h = cv2.boundingRect(contour)
		return float(w) / h if h > 0 else 0.0
	if prop.startswith("extent"):
		_, _, w, h = cv2.boundingRect(contour)
		return float(area / (w * h)) if w * h > 0 else 0.0
	if prop.startswith("solidity"):
		hull_area = cv2.contourArea(cv2.convexHull(contour))
		return float(area / hull_area) if hull_area > 0 else 0.0
	raise ValueError(f"Unknown contour property {prop!r}.")


class FilterContours(io.ComfyNode):
	_PROPERTIES = [
		"area (px^2)",
		"perimeter (px)",
		"circularity (4*pi*A/P^2, 1 = perfect circle)",
		"aspect ratio (bbox width / height)",
		"extent (area / bbox area)",
		"solidity (area / convex hull area)",
	]

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FilterContours",
			display_name="CV Filter Contours",
			category=CATEGORY,
			description="Keeps the contours whose measured property lies in "
			            "[min_value, max_value] - 'round things' (circularity "
			            "close to 1), 'wide things' (aspect ratio), 'solid blobs "
			            "vs. squiggles' (solidity)... Chain several to combine "
			            "criteria. The 'values' output holds the property of "
			            "each KEPT contour, aligned with the output set - to "
			            "tune the range, widen it and read the values with "
			            "'Inspect CV Data'. Zero matches is a valid result "
			            "(count = 0), not an error.",
			inputs=[
				Contours.Input("contours", tooltip="From 'CV Find Contours'."),
				io.Combo.Input("property", options=cls._PROPERTIES,
					default=cls._PROPERTIES[0],
					tooltip="What to measure on each contour. Circularity, "
					        "extent and solidity are scale-free ratios in 0-1; "
					        "area/perimeter are in pixels."),
				io.Float.Input("min_value", default=0.0, min=-1e9, max=1e9, step=0.05,
					tooltip="Smallest accepted property value (inclusive)."),
				io.Float.Input("max_value", default=1e9, min=-1e9, max=1e9, step=0.05,
					tooltip="Largest accepted property value (inclusive)."),
			],
			outputs=[
				Contours.Output(display_name="contours",
					tooltip="The kept contours, in the incoming order."),
				io.Int.Output(display_name="count"),
				NPArray.Output(display_name="values",
					tooltip="float32 property value per kept contour (aligned "
					        "with the contour output)."),
			],
		)

	@classmethod
	def execute(cls, contours, property, min_value, max_value) -> io.NodeOutput:
		kept, values = [], []
		for contour in contours:
			value = _contour_property(contour, property)
			if min_value <= value <= max_value:
				kept.append(contour)
				values.append(value)
		return io.NodeOutput(kept, len(kept), np.float32(values))


class FilterContoursByShape(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FilterContoursByShape",
			display_name="CV Filter Contours By Shape",
			category=CATEGORY,
			description="Keeps the contours that look like a reference shape "
			            "(cv2.matchShapes on Hu moments). The distance ignores "
			            "translation, scale and rotation by design, and "
			            "REFLECTION by accident: only the 7th Hu invariant's sign "
			            "changes under mirroring, and matchShapes skips any term "
			            "below its internal eps = 1e-5, which |h7| usually is. "
			            "Measured on this build: an 'F' glyph and its mirror are "
			            "indistinguishable (~0 on I1/I2/I3) - and the mirror can even "
			            "rank ABOVE a correctly-handed but rotated shape - while "
			            "a hook shape with a larger |h7| scores 0.50. So mirror "
			            "sensitivity here is a cliff, not a property, and is "
			            "never a usable handedness test. The scale/rotation/"
			            "mirror policies below are what put those dimensions "
			            "back; left at their defaults this node behaves exactly "
			            "as it always has. Identical shapes score 0; similar ones "
			            "stay below ~0.1-0.3 with CONTOURS_MATCH_I1. Make the "
			            "reference with 'CV Select Contour' (e.g. the largest "
			            "blob) or a second Find Contours over a template image; "
			            "when the reference set has several contours its first "
			            "one is used. Output is sorted most-similar first, with "
			            "every per-match output aligned to it. Failure-tolerant: "
			            "an empty reference or zero matches yields an empty set "
			            "(count = 0), not an error.",
			inputs=[
				Contours.Input("contours", tooltip="Candidates to test."),
				Contours.Input("reference",
					tooltip="The shape to compare against (first contour of the "
					        "set)."),
				io.Combo.Input("method", options=enums.CONTOURS_MATCH.options,
					default=enums.CONTOURS_MATCH.default,
					tooltip="How the Hu-moment signatures are compared (I1 is "
					        "the usual choice; I2/I3 are alternative norms)."),
				io.Float.Input("max_distance", default=0.3, min=0.0, max=1e6, step=0.01,
					tooltip="Largest accepted dissimilarity (0 = identical). "
					        "Raise it to 1e6 and inspect the distances output "
					        "to calibrate."),
				io.Combo.Input("scale_mode", options=shapepose.SCALE_MODES,
					default=shapepose.SCALE_MODES[0],
					tooltip=shapepose.SCALE_MODE_TOOLTIP,
					optional=True, advanced=True),
				io.Float.Input("scale_tolerance", default=0.10, min=0.0, max=100.0,
					step=0.01, tooltip=shapepose.SCALE_TOLERANCE_TOOLTIP,
					optional=True, advanced=True),
				io.Combo.Input("rotation_mode", options=shapepose.ROTATION_MODES,
					default=shapepose.ROTATION_MODES[0],
					tooltip=shapepose.ROTATION_MODE_TOOLTIP,
					optional=True, advanced=True),
				io.Float.Input("rotation_tolerance_deg", default=10.0, min=0.0, max=180.0,
					step=1.0, tooltip=shapepose.ROTATION_TOLERANCE_TOOLTIP,
					optional=True, advanced=True),
				io.Combo.Input("mirror_mode", options=shapepose.MIRROR_MODES,
					default=shapepose.MIRROR_MODES[0],
					tooltip=shapepose.MIRROR_MODE_TOOLTIP,
					optional=True, advanced=True),
				io.Float.Input("min_chirality", default=0.01, min=0.0, max=1.0,
					step=0.001, tooltip=shapepose.MIN_CHIRALITY_TOOLTIP,
					optional=True, advanced=True),
			],
			outputs=[
				Contours.Output(display_name="contours",
					tooltip="The matching contours, most similar first."),
				io.Int.Output(display_name="count"),
				NPArray.Output(display_name="distances",
					tooltip="float32 matchShapes distance per kept contour "
					        "(aligned with the contour output)."),
				NPArray.Output(display_name="scale_ratios",
					tooltip="(N,) float32 size of each kept contour relative to "
					        "the reference - the ratio of sqrt(m00), so 0.5 means "
					        "half the linear size. 0.0 when either shape has no "
					        "usable area."),
				NPArray.Output(display_name="rotations",
					tooltip="(N,) float32 rotation in degrees, 0-360, taking the "
					        "reference onto each kept contour (Y down, so it "
					        "turns clockwise on screen). For a MIRRORED match it "
					        "is the rotation applied AFTER flipping the reference "
					        "about its vertical axis. Noise for a shape with no "
					        "well-defined orientation - check 'pose_confidence' "
					        "on 'CV Shape Moments'."),
				NPArray.Output(display_name="mirrored",
					tooltip="(N,) float32 handedness verdict per kept contour: "
					        "+1 reflected, -1 same handedness, 0 undecidable "
					        "(one of the two shapes is mirror-symmetric, or is "
					        "less chiral than 'min_chirality')."),
				NPArray.Output(display_name="transforms",
					tooltip="(N,2,3) float32 similarity matrix per kept contour, "
					        "mapping REFERENCE coordinates onto that contour "
					        "(mirror, then scale, then rotate, then centroid to "
					        "centroid). Pull one row out with 'CV Slice Array' "
					        "(axis 0) and feed cv2_transform with the reference "
					        "points from 'CV Contour To Points' to draw the "
					        "reference over its match, or cv2_warpAffine to warp "
					        "a whole template image onto it."),
			],
		)

	@classmethod
	def execute(cls, contours, reference, method, max_distance,
	            scale_mode=shapepose.SCALE_MODES[0], scale_tolerance=0.10,
	            rotation_mode=shapepose.ROTATION_MODES[0], rotation_tolerance_deg=10.0,
	            mirror_mode=shapepose.MIRROR_MODES[0], min_chirality=0.01) -> io.NodeOutput:
		# These defaults MUST equal the schema defaults - run_workflows fills a
		# missing widget from the schema, so a divergence here is invisible to
		# every workflow test (test_filter_contours_by_shape_pose pins it).
		# An old graph carrying "" resolves to the invariant mode inside
		# shapepose._mode_index, so no sentinel handling is needed.
		if not len(contours) or not len(reference):
			return cls._empty()
		flag = enums.CONTOURS_MATCH.resolve(method)
		ref = reference[0]
		ref_m = cv2.moments(ref)
		ref_pose = shapepose.pose(ref_m, cv2.HuMoments(ref_m).ravel())
		scored = []
		for contour in contours:
			distance = cv2.matchShapes(ref, contour, flag, 0.0)
			if not np.isfinite(distance) or distance > max_distance:
				continue
			cand_m = cv2.moments(contour)
			cand_pose = shapepose.pose(cand_m, cv2.HuMoments(cand_m).ravel())
			rel = shapepose.relate(ref_pose, cand_pose, float(min_chirality))
			if not shapepose.accepts(rel, scale_mode, scale_tolerance,
			                         rotation_mode, rotation_tolerance_deg,
			                         mirror_mode):
				continue
			# everything rides on ONE tuple through ONE sort - deriving the pose
			# arrays separately is how they silently stop lining up
			scored.append((distance, contour, rel,
			               shapepose.similarity_matrix(ref_pose, cand_pose, rel)))
		if not scored:
			return cls._empty()
		scored.sort(key=lambda s: s[0])
		n = len(scored)
		return io.NodeOutput(
			[s[1] for s in scored], n,
			np.float32([s[0] for s in scored]),
			np.float32([s[2]["scale_ratio"] for s in scored]),
			np.float32([s[2]["rotation_deg"] for s in scored]),
			np.float32([float(s[2]["mirrored"]) for s in scored]),
			np.float32([s[3] for s in scored]).reshape(n, 2, 3),
		)

	@staticmethod
	def _empty():
		"""Zero matches is a valid result (rule 4). `transforms` must still be
		(0,2,3) or a downstream 'CV Slice Array' blows up on the found=false
		path."""
		return io.NodeOutput([], 0, np.zeros(0, np.float32),
		                     np.zeros(0, np.float32), np.zeros(0, np.float32),
		                     np.zeros(0, np.float32), np.zeros((0, 2, 3), np.float32))


class MergeContours(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MergeContours",
			display_name="CV Merge Contours",
			category=CATEGORY,
			description="Merges contours that lie within 'gap' pixels of each "
			            "other (boundary to boundary) into one contour per "
			            "group - reuniting an outline that broke into pieces, or "
			            "treating a cluster of blobs as one object. 'trace "
			            "merged outline' closes the gaps morphologically and "
			            "re-traces the union, keeping the original shapes; "
			            "'convex hull per group' spans each group with its hull "
			            "(simpler, convex). The labels output gives the group id "
			            "of every INPUT contour - inspect it to see what got "
			            "merged with what. Output is sorted by area, largest "
			            "first, like 'CV Find Contours'.",
			inputs=[
				polytype.poly_input("image",
					tooltip="Only its width/height are used (the canvas the "
					        "contours live on)."),
				Contours.Input("contours", tooltip="From 'CV Find Contours'."),
				io.Float.Input("gap", default=10.0, min=0.0, max=10000.0, step=1.0,
					tooltip="Contours closer than about this many pixels are "
					        "merged into one."),
				io.Combo.Input("mode",
					options=["trace merged outline (close gaps)",
					         "convex hull per group"],
					default="trace merged outline (close gaps)",
					tooltip="How each group becomes one contour: bridge the "
					        "gaps and re-trace (keeps the shapes), or take the "
					        "convex hull of all member points."),
			],
			outputs=[
				Contours.Output(display_name="contours",
					tooltip="One merged contour per group, sorted by area."),
				io.Int.Output(display_name="count"),
				NPArray.Output(display_name="labels",
					tooltip="int32 group id per INPUT contour (same order as "
					        "the input set) - which inputs were merged "
					        "together."),
			],
		)

	@classmethod
	def execute(cls, image, contours, gap, mode) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		h, w = np.asarray(image).shape[:2]
		if not len(contours):
			return io.NodeOutput([], 0, np.zeros(0, np.int32))
		# group: dilate the filled union by gap/2 - blobs within `gap` connect
		radius = max(1, int(round(gap / 2)))
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
		                                   (2 * radius + 1, 2 * radius + 1))
		filled = np.zeros((h, w), np.uint8)
		cv2.drawContours(filled, list(contours), -1, 255, cv2.FILLED)
		_, components = cv2.connectedComponents(cv2.dilate(filled, kernel),
		                                        connectivity=8)
		raw = []
		for contour in contours:
			x, y = contour[0][0]  # a boundary point lies inside its component
			raw.append(int(components[min(max(int(y), 0), h - 1),
			                          min(max(int(x), 0), w - 1)]))
		group_of = {v: i for i, v in enumerate(dict.fromkeys(raw))}
		labels = np.int32([group_of[v] for v in raw])

		merged = []
		for g in range(len(group_of)):
			members = [c for c, label in zip(contours, labels) if label == g]
			if mode.startswith("convex"):
				merged.append(cv2.convexHull(
					np.concatenate([m.reshape(-1, 1, 2) for m in members])))
				continue
			blank = np.zeros((h, w), np.uint8)
			cv2.drawContours(blank, members, -1, 255, cv2.FILLED)
			closed = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel)
			traced, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
			                             cv2.CHAIN_APPROX_SIMPLE)
			merged.extend(traced)
		merged.sort(key=cv2.contourArea, reverse=True)
		return io.NodeOutput(merged, len(merged), labels)


def _order_quad(pts: np.ndarray) -> np.ndarray:
	"""Order 4 points as top-left, top-right, bottom-right, bottom-left.

	Sorts by angle around the centroid (a simple, non-crossing cycle for
	any convex quad), then starts the cycle at the most top-left corner.
	The classic sum/diff heuristic is NOT used: on quads rotated near 45
	degrees it can pick the same point as both TL (min x+y) and TR
	(max x-y), and the duplicated corner degenerates the homography into
	a triangle.
	"""
	pts = pts.reshape(4, 2).astype(np.float32)
	center = pts.mean(axis=0)
	# increasing atan2 in image coordinates (y down) = clockwise on
	# screen, i.e. TL -> TR -> BR -> BL
	angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
	ordered = pts[np.argsort(angles)]
	start = int(np.argmin(ordered.sum(axis=1)))
	return np.roll(ordered, -start, axis=0).reshape(4, 1, 2)


def _collapse_to_quad(pts: np.ndarray):
	"""Reduce a convex N-gon (N > 4) to a quad by collapsing short edges.

	A page with a curled/dog-eared corner hulls into a pentagon: the fold
	cuts the corner off, and no approxPolyDP tolerance yields 4 points (5
	corners merge straight to 3). Repeatedly replace the shortest edge's
	two endpoints with the intersection of its neighboring edges - i.e.
	extend the real page edges until they meet where the corner would
	have been. Returns (4, 2) float32 or None (parallel neighbors, or
	the intersection runs away)."""
	pts = np.asarray(pts, np.float64).reshape(-1, 2)
	span = float(np.ptp(pts, axis=0).max())
	while len(pts) > 4:
		n = len(pts)
		edges = np.roll(pts, -1, axis=0) - pts
		lens = np.hypot(edges[:, 0], edges[:, 1])
		i = int(np.argmin(lens))
		if lens[i] > 0.6 * float(np.median(lens)):
			return None  # no clearly-short edge: a rounded blob (regular
			# N-gon), not a document with a corner cut off

		# neighbors of edge (i, i+1): edge (i-1, i) and edge (i+1, i+2)
		p0, d0 = pts[i - 1], edges[i - 1]
		p1, d1 = pts[(i + 1) % n], edges[(i + 1) % n]
		det = d0[0] * d1[1] - d0[1] * d1[0]
		if abs(det) < 1e-9:
			return None
		t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / det
		cross = p0 + t * d0
		if np.hypot(*(cross - pts[i])) > 0.5 * span:
			return None  # near-parallel neighbors: corner flies away
		pts = np.delete(pts, (i + 1) % n, axis=0)
		pts[i if (i + 1) % n > i else i - 1] = cross
	return pts.astype(np.float32)


class FindQuadrilateral(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FindQuadrilateral",
			display_name="CV Find Quadrilateral",
			category=CATEGORY,
			description="Finds the largest convex quadrilateral among the "
			            "contours - the classic document/whiteboard outline "
			            "detector. Each contour (largest first) is simplified "
			            "with cv2.approxPolyDP; the first one that simplifies "
			            "to 4 convex corners and covers at least "
			            "min_area_fraction of the image wins. When that "
			            "strict pass finds nothing, a second pass simplifies "
			            "each contour's CONVEX HULL instead, raising the "
			            "tolerance until 4 corners remain - recovering pages "
			            "with curled corners, shadow notches or texture-"
			            "broken edges. The corners come "
			            "out ordered top-left, top-right, bottom-right, "
			            "bottom-left, ready for 'CV Quad To Rectangle' + "
			            "cv2.getPerspectiveTransform. Failure-tolerant: when no "
			            "quad qualifies it returns found=false and the full "
			            "image corners (warping with them is a no-op), so the "
			            "workflow keeps running - branch on 'found' with an "
			            "if/else node.",
			inputs=[
				polytype.poly_input("image",
					tooltip="The photographed image - used for the area "
					        "reference and the not-found fallback corners."),
				Contours.Input("contours", tooltip="From 'CV Find Contours'."),
				io.Float.Input("epsilon_ratio", default=0.02, min=0.001, max=0.5, step=0.005,
					tooltip="approxPolyDP tolerance as a fraction of the "
					        "contour perimeter. Higher merges wobbly edges "
					        "into straight lines more aggressively."),
				io.Float.Input("min_area_fraction", default=0.05, min=0.0, max=1.0, step=0.01,
					tooltip="Smallest acceptable quad area as a fraction of "
					        "the image area - rejects small rectangles that "
					        "are not the document."),
			],
			outputs=[
				NPArray.Output(display_name="corners",
					tooltip="4x1x2 float32, ordered TL/TR/BR/BL (full image "
					        "corners if not found)."),
				io.Boolean.Output(display_name="found"),
				io.Float.Output(display_name="area_fraction",
					tooltip="Area of the detected quad relative to the image "
					        "(0.0 if not found)."),
			],
		)

	@classmethod
	def execute(cls, image, contours, epsilon_ratio, min_area_fraction) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		h, w = np.asarray(image).shape[:2]
		image_area = float(w * h)
		# pass 1: the contour as-is at the requested tolerance (exact
		# legacy behavior - anything it found before, it still finds)
		for contour in sorted(contours, key=cv2.contourArea, reverse=True):
			area = cv2.contourArea(contour)
			if area < min_area_fraction * image_area:
				break  # sorted: everything after this is smaller
			approx = cv2.approxPolyDP(contour, epsilon_ratio * cv2.arcLength(contour, True), True)
			if len(approx) == 4 and cv2.isContourConvex(approx):
				return io.NodeOutput(_order_quad(approx), True, area / image_area)
		# pass 2 (only when pass 1 found nothing): convex hull + epsilon
		# sweep. The hull absorbs the concavities that break pass 1 - a
		# curled page corner, a shadow notch, an edge broken into a zigzag
		# by texture - and raising epsilon merges wobbly hull facets until
		# exactly 4 corners remain. Sorting/thresholding must use the HULL
		# area: a broken page outline traces out and back, so its contour
		# area is a fraction of the region it visibly encloses.
		hulls = [cv2.convexHull(c) for c in contours if len(c) >= 3]
		for hull in sorted(hulls, key=cv2.contourArea, reverse=True):
			area = cv2.contourArea(hull)
			if area < min_area_fraction * image_area:
				break
			peri = cv2.arcLength(hull, True)
			eps = max(float(epsilon_ratio), 0.005)
			best = None  # smallest >4-gon the sweep saw
			while eps <= 0.12:
				approx = cv2.approxPolyDP(hull, eps * peri, True)
				if len(approx) < 4:
					break  # overshot: fewer corners than a quad has
				# the quad must FIT the hull, not merely have 4 corners: a
				# circle's hull also sweeps down to 4, but its inscribed
				# square covers only 2/pi (~0.64) of it
				if len(approx) == 4 and cv2.isContourConvex(approx) and \
						cv2.contourArea(approx) >= 0.8 * area:
					return io.NodeOutput(_order_quad(approx), True, area / image_area)
				if best is None or len(approx) < len(best):
					best = approx
				eps *= 1.4
			if best is not None and 5 <= len(best) <= 8:
				# a dog-eared page never sweeps to 4 (5 corners merge
				# straight to 3) - extend the true edges over the fold
				quad = _collapse_to_quad(best.reshape(-1, 2))
				if quad is not None and cv2.isContourConvex(
						quad.reshape(-1, 1, 2)) and \
						cv2.contourArea(quad) <= 1.4 * area:
					return io.NodeOutput(_order_quad(quad), True,
					                     area / image_area)
		fallback = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(4, 1, 2)
		return io.NodeOutput(fallback, False, 0.0)


class QuadToRectangle(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_QuadToRectangle",
			display_name="CV Quad To Rectangle",
			category=CATEGORY,
			description="Computes the axis-aligned target rectangle for a "
			            "TL/TR/BR/BL-ordered quadrilateral. By default the "
			            "output width/height are the longest of the opposing "
			            "edge pairs (projected edge lengths) - fine for mild "
			            "perspective, but a strongly foreshortened page (shot "
			            "from its bottom edge) comes out with the WRONG "
			            "orientation, e.g. a portrait document rectified into "
			            "a stretched landscape. Wire image_width/image_height "
			            "(e.g. from 'CV Array Shape' of the photo) to recover "
			            "the rectangle's TRUE aspect ratio from the "
			            "perspective itself (Zhang's whiteboard method: "
			            "square pixels + centered principal point). Feed "
			            "'corners' and 'rect_corners' into "
			            "cv2.getPerspectiveTransform (src/dst) and warp with "
			            "the dsize output.",
			inputs=[
				NPArray.Input("corners",
					tooltip="4x1x2 quad ordered TL/TR/BR/BL (from 'CV Find "
					        "Quadrilateral')."),
				io.Int.Input("max_size", default=8192, min=64, max=32768,
					tooltip="Cap on the output width/height; larger results "
					        "are scaled down proportionally."),
				io.Int.Input("image_width", default=0, min=0, max=32768,
					optional=True,
					tooltip="Width of the photo the corners live in. Both "
					        "dimensions > 0 enable perspective-corrected "
					        "aspect recovery; 0 keeps the projected-edge-"
					        "length behavior. Wire from 'CV Array Shape'."),
				io.Int.Input("image_height", default=0, min=0, max=32768,
					optional=True,
					tooltip="Height of the photo the corners live in. Both "
					        "dimensions > 0 enable perspective-corrected "
					        "aspect recovery; 0 keeps the projected-edge-"
					        "length behavior. Wire from 'CV Array Shape'."),
				io.Float.Input("focal_ratio", default=0.78, min=0.05, max=5.0,
					step=0.01, optional=True, advanced=True,
					tooltip="Assumed focal length as a fraction of "
					        "max(image_width, image_height), used ONLY when "
					        "the quad has a single vanishing point (opposing "
					        "edges parallel in the photo), where the focal "
					        "length cannot be recovered from the quad. 0.78 "
					        "~ a 28mm-equivalent phone lens; smaller = wider "
					        "lens = stronger foreshortening correction."),
			],
			outputs=[
				NPArray.Output(display_name="rect_corners",
					tooltip="4x1x2 float32 destination rectangle, same TL/TR/"
					        "BR/BL order."),
				io.String.Output(display_name="dsize",
					tooltip="'(width, height)' literal, for the composite cv2 "
					        "params that are still free-text."),
				io.Int.Output(display_name="width"),
				io.Int.Output(display_name="height"),
				tuples.CvTuple.Output(display_name="size",
					tooltip="The target (width, height) as ONE composite value "
					        "- wire it into cv2.warpPerspective's dsize."),
			],
		)

	@classmethod
	def execute(cls, corners, max_size, image_width=0, image_height=0,
	            focal_ratio=0.78) -> io.NodeOutput:
		pts = np.asarray(corners, np.float64).reshape(-1, 2)
		if len(pts) != 4:
			raise ValueError(
				f"corners must contain exactly 4 points (got {len(pts)}) - "
				"use 'CV Find Quadrilateral' to produce them."
			)
		image_wh = ((image_width, image_height)
		            if image_width > 0 and image_height > 0 else None)
		rect, w, h = _aspect_rect(pts, max_size, image_wh, focal_ratio)
		return io.NodeOutput(rect, f"({w}, {h})", w, h, [int(w), int(h)])


def _true_aspect(pts: np.ndarray, image_wh, focal_ratio: float):
	"""Real-world width/height ratio of the rectangle imaged as a quad.

	Zhang's whiteboard-scanning method: the TL/TR/BR/BL quad (4x2, pixels)
	is the perspective image of a rectangle; assuming square pixels and the
	principal point at the image center, the focal length falls out of the
	orthogonality of the rectangle's two edge directions, and with it the
	true aspect ratio. Projected edge lengths CANNOT give this: a portrait
	page photographed from its bottom edge foreshortens into a wide
	trapezoid and comes out landscape.

	When the quad has only ONE finite vanishing point (one-point
	perspective - opposing edges parallel in the image, the common case
	for a page shot straight-on from one side) the focal length is
	mathematically unobservable from the quad alone; ``focal_ratio *
	max(W, H)`` is assumed instead (0.78 ~ a 28mm-equivalent lens). The
	implied-focal validity window doubles as the noise guard: corner
	error in a near-degenerate quad produces wild or negative focal
	estimates, which fall back to the assumed focal too.

	Returns None when the quad is inconsistent with any camera (caller
	falls back to projected edge lengths).
	"""
	W, H = image_wh
	D = float(max(W, H))
	cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
	q = np.asarray(pts, np.float64).reshape(4, 2)
	tl, tr, br, bl = (np.array([x - cx, y - cy, 1.0]) for x, y in q)
	m1, m2, m3, m4 = tl, tr, bl, br  # m4 opposite m1 (Zhang's ordering)
	d2 = np.dot(np.cross(m2, m4), m3)
	e3 = np.dot(np.cross(m3, m4), m2)
	if abs(d2) < 1e-9 or abs(e3) < 1e-9:
		return None  # three corners collinear
	k2 = np.dot(np.cross(m1, m4), m3) / d2
	k3 = np.dot(np.cross(m1, m4), m2) / e3
	n2 = k2 * m2 - m1  # width direction (TL -> TR)
	n3 = k3 * m3 - m1  # height direction (TL -> BL)
	f2 = None
	zz = n2[2] * n3[2]
	if abs(zz) > 1e-12:
		cand = -(n2[0] * n3[0] + n2[1] * n3[1]) / zz
		if np.isfinite(cand) and (0.2 * D) ** 2 <= cand <= (5.0 * D) ** 2:
			f2 = cand
	if f2 is None:
		f2 = (max(focal_ratio, 0.05) * D) ** 2
	num = n2[0] ** 2 + n2[1] ** 2 + n2[2] ** 2 * f2
	den = n3[0] ** 2 + n3[1] ** 2 + n3[2] ** 2 * f2
	if not (np.isfinite(num) and np.isfinite(den)) or num <= 0 or den <= 0:
		return None
	ratio = float(np.sqrt(num / den))
	return ratio if 1.0 / 16.0 <= ratio <= 16.0 else None


def _aspect_rect(pts: np.ndarray, max_size: int, image_wh=None,
                 focal_ratio: float = 0.78):
	"""Aspect-preserving target rectangle for a TL/TR/BR/BL quad (4x2).

	Without image_wh the aspect comes from the projected edge lengths
	(legacy behavior - wrong under strong foreshortening). With image_wh
	the true rectangle aspect is recovered perspectively (_true_aspect)
	and the output keeps the pixel COUNT of the edge-length rectangle, so
	the resolution stays comparable to what was photographed.
	"""
	tl, tr, br, bl = pts
	# +1: corner coordinates are pixel centers, the output spans one more
	# pixel than the corner distance (a full-image quad maps to a no-op)
	width = max(np.hypot(*(tr - tl)), np.hypot(*(br - bl))) + 1
	height = max(np.hypot(*(bl - tl)), np.hypot(*(br - tr))) + 1
	if image_wh is not None and image_wh[0] > 0 and image_wh[1] > 0:
		ratio = _true_aspect(pts, image_wh, focal_ratio)
		if ratio is not None:
			area = width * height
			width = float(np.sqrt(area * ratio))
			height = float(np.sqrt(area / ratio))
	scale = min(1.0, max_size / max(width, height, 1.0))
	w = max(1, int(round(width * scale)))
	h = max(1, int(round(height * scale)))
	rect = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(4, 1, 2)
	return rect, w, h


class QuadWarp(io.ComfyNode):
	_TEMPLATE = polytype.match_template()

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_QuadWarp",
			display_name="CV Quad Warp",
			category=CATEGORY,
			description="Applies the homography defined by 4 annotated corner "
			            "points. 'extract' rectifies the quad into a rectangle "
			            "with the TRUE aspect ratio recovered from the "
			            "perspective (a manual document scan; Zhang's "
			            "whiteboard method with the principal point at the "
			            "image center); 'fill' "
			            "warps THIS image into the quad on a canvas (poster / "
			            "billboard / screen replacement). The 4 points may "
			            "arrive in any order - 'CV Annotate Points' click "
			            "order, detections, ... - they are sorted TL/TR/BR/BL "
			            "automatically. The matrix output is the 3x3 homography "
			            "actually used, ready for 'CV Homography Map' to "
			            "combine with other remap distortions. Failure-"
			            "tolerant: anything other than exactly 4 points passes "
			            "the image through with success=false and an identity "
			            "matrix - branch on success with an if/else node.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="extract: the photo containing the quad; fill: the "
					        "image to warp INTO the quad. The warped output "
					        "comes back in this input's format."),
				NPArray.Input("points",
					tooltip="4 corner points (Nx1x2 or Nx2), any order."),
				io.Combo.Input("mode",
					options=["extract: rectify the quad to a rectangle",
					         "fill: warp the image into the quad"],
					default="extract: rectify the quad to a rectangle",
					tooltip="extract: straighten the quad into a rectangle (document "
					        "scan). fill: warp this image into the quad on a canvas "
					        "(poster/screen replacement)."),
				io.Int.Input("max_size", default=8192, min=64, max=32768,
					tooltip="extract only: cap on the output width/height."),
				polytype.poly_input("canvas", types=polytype.IMAGE_ONLY, optional=True,
					tooltip="fill only: the background the warped image is "
					        "composited onto (the points live in ITS "
					        "coordinates). Black canvas around the quad when "
					        "left unconnected."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "warped",
					tooltip="Same format as the image input."),
				NPArray.Output(display_name="matrix",
					tooltip="3x3 homography used (identity when success=false)."),
				io.Boolean.Output(display_name="success"),
			],
		)

	@classmethod
	def execute(cls, image, points, mode, max_size, canvas=None) -> io.NodeOutput:
		image, tag = polytype.resolve(image)
		canvas, _ = polytype.resolve(canvas)
		img = np.asarray(image)
		pts = (np.zeros((0, 2), np.float64) if points is None
		       else np.asarray(points, np.float64).reshape(-1, 2))
		if len(pts) != 4:
			return io.NodeOutput(polytype.restore(img.copy(), tag), np.eye(3), False)
		quad = _order_quad(np.float32(pts)).reshape(4, 2)
		if mode.startswith("extract"):
			# the image IS the photo the quad lives in, so its center is the
			# principal point _true_aspect needs - perspective-corrected
			# aspect for free (falls back to edge lengths when inconsistent)
			rect, w, h = _aspect_rect(quad.astype(np.float64), max_size,
			                          (img.shape[1], img.shape[0]))
			matrix = cv2.getPerspectiveTransform(quad, rect.reshape(4, 2))
			warped = cv2.warpPerspective(img, matrix, (w, h),
			                             flags=cv2.INTER_LINEAR,
			                             borderMode=cv2.BORDER_CONSTANT)
			return io.NodeOutput(polytype.restore(warped, tag), matrix, True)
		# fill: this image's corners -> the quad, composited onto the canvas.
		# The homography through 4 implicit corner correspondences is the special
		# case CV_PasteThroughWarp generalises; both go through warpfit, so the
		# premultiplied composite that keeps the quad's edge free of a dark rim is
		# written once (warpfit.composite_over).
		src = _to_bgr_u8(img)
		sh, sw = src.shape[:2]
		if canvas is not None:
			out = _to_bgr_u8(canvas)
		else:
			cw = int(np.ceil(quad[:, 0].max())) + 1
			ch = int(np.ceil(quad[:, 1].max())) + 1
			out = np.zeros((ch, cw, 3), np.uint8)
		corners = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])
		w = warpfit.fit("homography", corners, quad)
		if w is None:   # four points that do not span a quad
			return io.NodeOutput(polytype.restore(img.copy(), tag), np.eye(3), False)
		composite, _ = warpfit.paste_warped(src, out, w)
		return io.NodeOutput(polytype.restore(composite, tag), w.matrix(), True)


class EnclosingCircle(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_EnclosingCircle",
			display_name="CV Enclosing Circle",
			category=CATEGORY,
			description="Smallest circle around a contour "
			            "(cv2.minEnclosingCircle) - e.g. a clock dial, a ball, "
			            "a coin. 'longest contour' (the default) selects by "
			            "perimeter, which stays robust when the shape's outline "
			            "breaks into an open arc: an open curve traces out and "
			            "back, so its AREA is near zero and area-based selection "
			            "would pick some small closed blob instead (this "
			            "happened with a clock rim that lost a few edge pixels "
			            "to decoder differences). 'largest contour' selects by "
			            "area; 'all contours merged' circles every contour "
			            "point at once. Failure-tolerant: with no contours it "
			            "returns found=false and zeros instead of stopping the "
			            "workflow (downstream nodes treat radius 0 as "
			            "not-found).",
			inputs=[
				Contours.Input("contours", tooltip="From 'CV Find Contours'."),
				io.Combo.Input("select",
					options=["longest contour (perimeter)",
					         "largest contour (area)",
					         "all contours merged"],
					default="longest contour (perimeter)",
					tooltip="Which contour to circle. Perimeter is robust to "
					        "broken outlines; merged spans everything (use it "
					        "when the whole edge map belongs to one object)."),
			],
			outputs=[
				io.Float.Output(display_name="center_x"),
				io.Float.Output(display_name="center_y"),
				io.Float.Output(display_name="radius"),
				io.Boolean.Output(display_name="found"),
				tuples.CvTuple.Output(display_name="center",
					tooltip="The centre as ONE composite (x, y) value - wire it "
					        "into a cv2 node's centre/point input in one link "
					        "instead of two. Subpixel: a point2i input rounds it."),
			],
		)

	@classmethod
	def execute(cls, contours, select) -> io.NodeOutput:
		if not len(contours):
			return io.NodeOutput(0.0, 0.0, 0.0, False, [0.0, 0.0])
		if select.startswith("all"):
			chosen = np.concatenate([c.reshape(-1, 2) for c in contours])
		elif select.startswith("largest"):
			chosen = max(contours, key=cv2.contourArea)
		else:
			chosen = max(contours, key=lambda c: cv2.arcLength(c, True))
		(cx, cy), r = cv2.minEnclosingCircle(chosen)
		return io.NodeOutput(float(cx), float(cy), float(r), True,
		                     [float(cx), float(cy)])


class ShapeMoments(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ShapeMoments",
			display_name="CV Shape Moments",
			category=CATEGORY,
			description="Measures every contour with cv2.moments/cv2.HuMoments "
			            "and surfaces the data the raw wrappers cannot: centroid, "
			            "area, orientation angle and eccentricity (both from the "
			            "second-order central moments), the major-axis segment, "
			            "and the 7 log-scaled Hu invariants (translation/scale/"
			            "rotation invariant shape signature - identical shapes "
			            "give near-identical vectors regardless of pose). Hu is "
			            "also blind to REFLECTION in everything but its 7th "
			            "component, so the last three outputs recover what the "
			            "invariance throws away: a canonical 0-360 orientation "
			            "(resolved from the third-order moments, unlike 'angles' "
			            "which is only defined mod 180), a signed handedness, and "
			            "how much either can be trusted. All outputs are aligned "
			            "with the input set, ready for 'CV Draw Points' "
			            "(centroids), 'CV Draw Segments' (axes) and 'Inspect "
			            "CV Data'. Zero contours is a valid result (empty arrays, "
			            "count = 0), not an error.",
			inputs=[
				Contours.Input("contours", tooltip="From 'CV Find Contours'."),
			],
			outputs=[
				NPArray.Output(display_name="centroids",
					tooltip="(N,1,2) float32 center of mass per contour - plugs "
					        "into 'CV Draw Points' / 'Draw Labels'."),
				NPArray.Output(display_name="areas",
					tooltip="(N,) float32 area in px^2 (the m00 moment)."),
				NPArray.Output(display_name="angles",
					tooltip="(N,) float32 orientation of the major axis in "
					        "degrees, [-90, 90), measured from the +X axis with Y "
					        "pointing DOWN (image coordinates). Meaningless for "
					        "near-circular shapes (eccentricity ~ 0)."),
				NPArray.Output(display_name="eccentricities",
					tooltip="(N,) float32 elongation in 0-1: 0 = perfect circle, "
					        "-> 1 = a line. From the central-moment eigenvalues."),
				NPArray.Output(display_name="axes",
					tooltip="(N,4) float32 major-axis segments (x1, y1, x2, y2) "
					        "through each centroid, 2 standard deviations long per "
					        "side - plugs into 'CV Draw Segments'."),
				NPArray.Output(display_name="hu_moments",
					tooltip="(N,7) float32 log-scaled Hu invariants: "
					        "-sign(h)*log10(|h|) per component (the raw values "
					        "span ~40 orders of magnitude). Compare shapes with "
					        "'CV Filter Contours By Shape' or a vector "
					        "distance."),
				io.Int.Output(display_name="count"),
				NPArray.Output(display_name="orientations",
					tooltip="(N,) float32 canonical orientation in degrees, "
					        "0-360, from the +X axis with Y pointing DOWN. "
					        "Unlike 'angles' (defined only mod 180) the head is "
					        "told from the tail using the third-order moments, so "
					        "a shape and its 180-degree rotation read "
					        "differently. Always equals 'angles' or 'angles' + "
					        "180. MEANINGLESS when pose_confidence is ~0."),
				NPArray.Output(display_name="chirality",
					tooltip="(N,) float32 signed handedness, from the 7th Hu "
					        "invariant - the ONLY one that is not "
					        "reflection-invariant (OpenCV: \"invariants to the "
					        "image scale, rotation, and reflection except the "
					        "seventh one, whose sign is changed by reflection\"; "
					        "Hu 1962). h7 is also the one routinely DISCARDED in "
					        "practice, because it is the smallest and noisiest of "
					        "the seven - raw values here are ~1e-6 - so this "
					        "output rescales it as sign(h7)*|h7|**0.25 to put it "
					        "on a thresholdable scale. The SIGN flips when the "
					        "shape is mirrored and survives any rotation or "
					        "scale; the MAGNITUDE says how far to trust it: an "
					        "'F' glyph reads 0.020-0.048, a scalene triangle "
					        "0.0069, and a mirror-symmetric shape (square, "
					        "circle, ellipse, isoceles triangle) reads below "
					        "~0.005, where the sign is only rasterization noise "
					        "(OpenCV again: the invariance assumes \"infinite "
					        "image resolution\"). It agrees in SIGN with "
					        "hu_moments[:,6] - that column negates the sign but "
					        "also takes log10 of a value far below 1, and the two "
					        "flips cancel."),
				NPArray.Output(display_name="pose_confidence",
					tooltip="(N,) float32 how well-defined 'orientations' is: "
					        "the normalized third moment along the major axis. "
					        "Measured 0.052 for a strongly asymmetric shape, "
					        "0.00057 for an ellipse and 0.0 for a square or "
					        "circle - so test against ~0.005, not against 0. "
					        "Below that the 0-360 orientation is a coin flip and "
					        "any rotation measured from it is noise."),
			],
		)

	@classmethod
	def execute(cls, contours) -> io.NodeOutput:
		centroids, areas, angles, eccs, axes, hu_all = [], [], [], [], [], []
		oris, chirs, confs = [], [], []
		for contour in contours:
			m = cv2.moments(contour)
			if m["m00"] <= 1e-9:
				# degenerate (a line/point traces zero area): centroid from the
				# raw points, no meaningful orientation/eccentricity
				pts = contour.reshape(-1, 2).astype(np.float64)
				cx, cy = (pts.mean(axis=0) if len(pts) else (0.0, 0.0))
				centroids.append((cx, cy))
				areas.append(0.0)
				angles.append(0.0)
				eccs.append(0.0)
				axes.append((cx, cy, cx, cy))
				hu_all.append(np.zeros(7, np.float64))
				oris.append(0.0)
				chirs.append(0.0)
				confs.append(0.0)
				continue
			cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
			angle, ecc, half = shapepose.second_order(m)
			dx, dy = half * math.cos(math.radians(angle)), half * math.sin(math.radians(angle))
			hu = cv2.HuMoments(m).ravel()
			# the canonical pose lives in shapepose so the raster moment nodes
			# and the matcher cannot drift from it; `angle`/`ecc`/`axes` above
			# stay as they were, bit for bit
			p = shapepose.pose(m, hu)
			centroids.append((cx, cy))
			areas.append(m["m00"])
			angles.append(angle)
			eccs.append(ecc)
			axes.append((cx - dx, cy - dy, cx + dx, cy + dy))
			hu_all.append(-np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-30)))
			oris.append(p["orientation"])
			chirs.append(p["chirality"])
			confs.append(p["confidence"])
		n = len(centroids)
		return io.NodeOutput(
			np.float32(centroids).reshape(n, 1, 2),
			np.float32(areas),
			np.float32(angles),
			np.float32(eccs),
			np.float32(axes).reshape(n, 4),
			np.float32(hu_all).reshape(n, 7),
			n,
			np.float32(oris),
			np.float32(chirs),
			np.float32(confs),
		)


class ContourToPoints(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ContourToPoints",
			display_name="CV Contour To Points",
			category=CATEGORY,
			description="Bridges the CV_CONTOURS type to a plain NPARRAY point "
			            "array so ONE contour can feed the raw cv2 geometry "
			            "wrappers (cv2.convexHull, cv2.convexityDefects, "
			            "cv2.intersectConvexConvex, cv2.pointPolygonTest, "
			            "cv2.fitEllipse, cv2.contourArea...) or the point-set "
			            "nodes. The inverse bridge is 'CV Points To Contour'. "
			            "Failure-tolerant: an index past the end yields "
			            "found=false and an empty (0,1,2) array instead of an "
			            "error.",
			inputs=[
				Contours.Input("contours", tooltip="From 'CV Find Contours'."),
				io.Int.Input("index", default=0, min=0, max=100000,
					tooltip="Which contour of the set to convert (Find Contours "
					        "sorts by area, largest first - 0 = biggest)."),
				io.Combo.Input("dtype",
					options=["int32 (native contour)", "float32"],
					default="int32 (native contour)",
					tooltip="Element type. Keep int32 for cv2.convexityDefects "
					        "(it requires the ORIGINAL integer contour); float32 "
					        "for subpixel geometry (cv2.intersectConvexConvex)."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="(N,1,2) point array - the layout every cv2 geometry "
					        "function and draw node expects."),
				io.Int.Output(display_name="count"),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, contours, index, dtype) -> io.NodeOutput:
		np_dtype = np.int32 if dtype.startswith("int32") else np.float32
		if index >= len(contours):
			return io.NodeOutput(np.zeros((0, 1, 2), np_dtype), 0, False)
		pts = np.asarray(contours[index]).reshape(-1, 1, 2).astype(np_dtype)
		return io.NodeOutput(pts, len(pts), True)


class PointsToContour(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PointsToContour",
			display_name="CV Points To Contour",
			category=CATEGORY,
			description="Bridges a plain NPARRAY point array back to the "
			            "CV_CONTOURS type, so a cv2-computed polygon (a "
			            "cv2.convexHull hull, a cv2.intersectConvexConvex "
			            "intersection, 'CV Points' corners...) can feed the "
			            "contour nodes - most usefully 'CV Draw Contours' to "
			            "render it. Coordinates are rounded to the int32 grid "
			            "contours live on. The inverse bridge is 'CV Contour "
			            "To Points'. None/empty input yields an empty set "
			            "(count = 0), not an error.",
			inputs=[
				NPArray.Input("points",
					tooltip="(N,2), (N,1,2) or (1,N,2) point array, any numeric "
					        "dtype (subpixel coordinates are rounded)."),
			],
			outputs=[
				Contours.Output(display_name="contours",
					tooltip="A 1-contour set (empty when the input is "
					        "None/empty)."),
				io.Int.Output(display_name="count",
					tooltip="How many points the contour holds."),
			],
		)

	@classmethod
	def execute(cls, points) -> io.NodeOutput:
		if points is None or np.asarray(points).size == 0:
			return io.NodeOutput([], 0)
		pts = np.asarray(points, np.float64).reshape(-1, 2)
		contour = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
		return io.NodeOutput([contour], len(contour))


class Skeletonize(io.ComfyNode):
	_TEMPLATE = polytype.match_template()

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Skeletonize",
			display_name="CV Skeletonize",
			category=CATEGORY,
			description="Reduces the foreground of a binary image to its "
			            "1-pixel-wide skeleton (centerline). Thinning "
			            "(cv2.ximgproc, opencv-contrib) gives clean connected "
			            "centerlines; 'morphological' is a coarser fallback "
			            "that works without the contrib build. Used by the "
			            "clock-reading pipeline to turn thick hands into "
			            "measurable rays.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="Binary image (or MASK); non-zero = foreground. The "
					        "skeleton comes back in this input's format."),
				io.Combo.Input("method",
					options=["thinning (Zhang-Suen)", "thinning (Guo-Hall)",
					         "morphological"],
					default="thinning (Zhang-Suen)",
					tooltip="Zhang-Suen / Guo-Hall need opencv-contrib-python(-headless) "
					        "(cv2.ximgproc) and give the cleanest centerlines; if "
					        "that build is unavailable they fall back to "
					        "'morphological' automatically. Morphological works "
					        "everywhere but is noisier."),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "skeleton",
				tooltip="Same format as the image input.")],
		)

	@classmethod
	def execute(cls, image, method) -> io.NodeOutput:
		global _thinning_warned
		image, tag = polytype.resolve(image)
		binary = ((_to_gray_u8(image) > 0) * np.uint8(255))
		if method.startswith("thinning"):
			ximgproc = _thinning_available()
			if ximgproc is not None:
				kind = (ximgproc.THINNING_GUOHALL if "Guo-Hall" in method
				        else ximgproc.THINNING_ZHANGSUEN)
				return io.NodeOutput(polytype.restore(
					ximgproc.thinning(binary, thinningType=kind), tag))
			# Failure tolerance: the contrib thinning API is
			# missing from this build, so degrade to the morphological skeleton
			# instead of halting the whole workflow with an AttributeError.
			if not _thinning_warned:
				print("[comfyui-cv] Skeletonize: cv2.ximgproc.thinning is "
				      "unavailable in this OpenCV build; falling back to the "
				      "'morphological' method. Install a working "
				      "opencv-contrib-python(-headless) for cleaner centerlines.")
				_thinning_warned = True
		skeleton = np.zeros_like(binary)
		work = binary.copy()
		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
		while work.any():
			opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
			skeleton |= cv2.subtract(work, opened)
			work = cv2.erode(work, kernel)
		return io.NodeOutput(polytype.restore(skeleton, tag))


class SelectComponentAtPoint(io.ComfyNode):
	_TEMPLATE = polytype.match_template()

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SelectComponentAtPoint",
			display_name="CV Select Component At Point",
			category=CATEGORY,
			description="Keeps only the connected component(s) of a binary "
			            "image that touch a given point (or a small disk "
			            "around it) - 'the blob at this location'. E.g. the "
			            "clock hands are the only dark blob connected to the "
			            "dial center; numerals, ticks and the rim are separate "
			            "components and disappear. Failure-tolerant: no "
			            "foreground at the point yields found=false and an "
			            "empty mask instead of an error.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="Binary image (or MASK); non-zero = foreground. The "
					        "mask comes back in this input's format."),
				io.Float.Input("x", default=0.0, min=-1e6, max=1e6,
					tooltip="X pixel coordinate of the probe point."),
				io.Float.Input("y", default=0.0, min=-1e6, max=1e6,
					tooltip="Y pixel coordinate of the probe point."),
				io.Float.Input("search_radius", default=0.0, min=0.0, max=1e6,
					tooltip="Also accept components that touch a disk of this "
					        "radius around (x, y). 0 = the exact pixel only."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "mask",
					tooltip="0/255 mask of the selected component(s), in the "
					        "image input's format."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, image, x, y, search_radius) -> io.NodeOutput:
		image, tag = polytype.resolve(image)
		binary = (_to_gray_u8(image) > 0).astype(np.uint8)
		h, w = binary.shape
		probe = np.zeros_like(binary)
		px, py = int(round(x)), int(round(y))
		if search_radius > 0:
			cv2.circle(probe, (px, py), max(1, int(round(search_radius))), 1, -1)
		elif 0 <= px < w and 0 <= py < h:
			probe[py, px] = 1
		_, labels = cv2.connectedComponents(binary, connectivity=8)
		wanted = np.unique(labels[(probe > 0) & (labels > 0)])
		if len(wanted) == 0:
			return io.NodeOutput(polytype.restore(np.zeros((h, w), np.uint8), tag), False)
		return io.NodeOutput(
			polytype.restore(np.isin(labels, wanted).astype(np.uint8) * 255, tag), True)


class KeepLargestComponent(io.ComfyNode):
	_TEMPLATE = polytype.match_template()

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_KeepLargestComponent",
			display_name="CV Keep Largest Component",
			category=CATEGORY,
			description="Keeps only the single largest connected component of a "
			            "binary mask (cv2.connectedComponentsWithStats), dropping "
			            "every smaller blob - the standard way to isolate one "
			            "object from scattered noise or secondary movers before "
			            "seeding GrabCut / watershed. Unlike 'Select Component At "
			            "Point' it needs no probe location: it just takes the "
			            "biggest blob by area. Failure-tolerant: an empty mask (no "
			            "foreground) yields found=false and an empty mask, not an "
			            "error.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="Binary image (or MASK); non-zero = foreground. The "
					        "mask comes back in this input's format."),
				io.Combo.Input("connectivity", options=["8", "4"], default="8",
					tooltip="8 also connects diagonal pixels; 4 only direct neighbours."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "mask",
					tooltip="0/255 mask of the largest component, in the image "
					        "input's format."),
				io.Int.Output(display_name="area",
					tooltip="Pixel area of the kept component (0 if none)."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, image, connectivity) -> io.NodeOutput:
		image, tag = polytype.resolve(image)
		binary = (_to_gray_u8(image) > 0).astype(np.uint8)
		h, w = binary.shape
		n, labels, stats, _ = cv2.connectedComponentsWithStats(
			binary, connectivity=int(connectivity))
		if n <= 1:  # label 0 is the background, so n == 1 means nothing found
			return io.NodeOutput(polytype.restore(np.zeros((h, w), np.uint8), tag), 0, False)
		biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
		mask = (labels == biggest).astype(np.uint8) * 255
		return io.NodeOutput(polytype.restore(mask, tag),
		                     int(stats[biggest, cv2.CC_STAT_AREA]), True)


class ComponentsTouchingBorder(io.ComfyNode):
	"""Keep (or drop) every blob that reaches a chosen image edge.

	The other two component nodes cannot express this. 'Keep Largest Component'
	takes the biggest blob, which on a washed-out driving frame is the ROAD
	(71 344 px) rather than the sky; 'Select Component At Point' takes ONE
	component, and a lamp post splits the sky of the shipped stereo frames into
	18 of them. Reaching the border is its own predicate: it is what separates
	unbounded background (sky, table surface, water) from bounded objects, and
	inverted it is the standard "drop detections clipped by the frame" filter.
	"""

	_TEMPLATE = polytype.match_template()
	_BORDERS = {
		"any edge": ("top", "bottom", "left", "right"),
		"top": ("top",),
		"bottom": ("bottom",),
		"left": ("left",),
		"right": ("right",),
		"top or bottom": ("top", "bottom"),
		"left or right": ("left", "right"),
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ComponentsTouchingBorder",
			display_name="CV Components Touching Border",
			category=CATEGORY,
			description="Keeps every connected component of a binary mask that "
			            "reaches a chosen image edge - or, with mode = 'remove', "
			            "everything that does NOT. Touching the frame edge is "
			            "what separates unbounded background (sky, a table "
			            "surface, water) from bounded objects, and the inverse "
			            "is the usual way to drop detections the frame clipped. "
			            "Neither 'Keep Largest Component' nor 'Select Component "
			            "At Point' can express it: the largest blob is often "
			            "something else entirely, and one probe point catches "
			            "one component, while a background region is routinely "
			            "cut into many by whatever crosses it. Failure-tolerant: "
			            "an empty mask (or one where nothing touches the chosen "
			            "edge) yields an empty mask and found=false, not an "
			            "error.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="Binary image (or MASK); non-zero = foreground. The "
					        "mask comes back in this input's format."),
				io.Combo.Input("border", options=list(cls._BORDERS),
					default="any edge",
					tooltip="Which image edge a component must reach to count."),
				io.Combo.Input("mode",
					options=["keep touching", "remove touching"],
					default="keep touching",
					tooltip="'keep touching' isolates the background that runs "
					        "off the frame; 'remove touching' drops blobs the "
					        "frame cut off and keeps the fully-visible ones."),
				io.Combo.Input("connectivity", options=["8", "4"], default="8",
					tooltip="8 also connects diagonal pixels; 4 only direct "
					        "neighbours.", optional=True, advanced=True),
				io.Int.Input("margin", default=0, min=0, max=64,
					optional=True, advanced=True,
					tooltip="How many pixels in from the edge still count as "
					        "'the border'. It has to cover whatever artefact the "
					        "frame edge already carries, or a region that "
					        "visually reaches the edge is missed entirely: "
					        "cv2.remap with BORDER_CONSTANT leaves a dark seam "
					        "one row deep, and any blur/derivative applied after "
					        "it smears that seam a few rows further, so a mask "
					        "derived from gradients may only start answering at "
					        "row 4. Morphology is NOT the problem - OpenCV pads "
					        "erosion with +inf, so an opening does not shrink a "
					        "region at the image border."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "mask",
					tooltip="0/255 mask of the kept components, in the image "
					        "input's format."),
				io.Int.Output(display_name="area",
					tooltip="Total pixel area kept."),
				io.Int.Output(display_name="count",
					tooltip="How many components were kept."),
				io.Boolean.Output(display_name="found",
					tooltip="False when nothing was kept (empty mask out)."),
			],
		)

	@classmethod
	def execute(cls, image, border, mode, connectivity="8",
	            margin=0) -> io.NodeOutput:
		image, tag = polytype.resolve(image)
		binary = (_to_gray_u8(image) > 0).astype(np.uint8)
		h, w = binary.shape
		empty = np.zeros((h, w), np.uint8)
		n, labels, stats, _ = cv2.connectedComponentsWithStats(
			binary, connectivity=int(connectivity))
		if n <= 1:  # label 0 is the background
			return io.NodeOutput(polytype.restore(empty, tag), 0, 0, False)

		m = int(np.clip(margin, 0, min(h, w) // 2 - 1 if min(h, w) > 2 else 0))
		bands = {
			"top": labels[:m + 1, :],
			"bottom": labels[h - 1 - m:, :],
			"left": labels[:, :m + 1],
			"right": labels[:, w - 1 - m:],
		}
		touching = set()
		for side in cls._BORDERS[border]:
			touching.update(int(v) for v in np.unique(bands[side]) if v)

		labels_of_interest = (touching if mode == "keep touching"
		                      else set(range(1, n)) - touching)
		if not labels_of_interest:
			return io.NodeOutput(polytype.restore(empty, tag), 0, 0, False)
		mask = np.isin(labels, sorted(labels_of_interest)).astype(np.uint8) * 255
		area = int(sum(stats[i, cv2.CC_STAT_AREA] for i in labels_of_interest))
		return io.NodeOutput(polytype.restore(mask, tag), area,
		                     len(labels_of_interest), True)


class FillHoles(io.ComfyNode):
	_TEMPLATE = polytype.match_template()

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FillHoles",
			display_name="CV Fill Holes",
			category=CATEGORY,
			description="Fills interior holes of a binary mask - background "
			            "pixels fully enclosed by foreground become foreground "
			            "(flood-fill from the border, then OR the unreachable "
			            "interior back in). Turns a ring into a solid disk, or a "
			            "motion mask with gaps over an object's low-texture "
			            "regions into a solid blob - essential before eroding it "
			            "into a GrabCut seed, so the seed does not punch holes in "
			            "the object. Objects touching the image edge are handled "
			            "(the mask is padded before the flood) and an empty mask "
			            "stays empty.",
			inputs=[
				polytype.match_input("image", cls._TEMPLATE,
					tooltip="Binary image (or MASK); non-zero = foreground. The "
					        "mask comes back in this input's format."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "mask",
					tooltip="0/255 mask with interior holes filled, in the "
					        "image input's format."),
			],
		)

	@classmethod
	def execute(cls, image) -> io.NodeOutput:
		image, tag = polytype.resolve(image)
		binary = (_to_gray_u8(image) > 0).astype(np.uint8) * 255
		h, w = binary.shape
		# pad a zero border so the flood seed at (0, 0) is always background,
		# even when the object touches an image edge; flooding marks every
		# background pixel reachable from outside, leaving only the holes unset
		padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
		flood = padded.copy()
		ff_mask = np.zeros((h + 4, w + 4), np.uint8)
		cv2.floodFill(flood, ff_mask, (0, 0), 255)
		holes = cv2.bitwise_not(flood)[1:-1, 1:-1]  # interior (enclosed) background
		return io.NodeOutput(polytype.restore(binary | holes, tag))


class DetectMSERRegions(io.ComfyNode):
	"""cv2.MSER - Maximally Stable Extremal Regions, a class the generator misses."""

	_BOTH = "both (dark and bright regions)"
	_BRIGHT = "bright regions only (single pass)"

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DetectMSERRegions",
			display_name="CV Detect MSER Regions",
			category=CATEGORY,
			description="Maximally Stable Extremal Regions (cv2.MSER): sweeps EVERY "
			            "grayscale threshold at once and keeps the blobs whose area "
			            "barely changes across a wide range of them. That is the "
			            "difference from 'CV Find Contours' on a thresholded "
			            "image - there is no threshold to pick, and regions at "
			            "different brightness levels are all found in the same pass, "
			            "so it survives uneven lighting. The classic use is text and "
			            "sign detection (letters are extremely stable regions) and "
			            "generic region proposals. MSER is a cv2 class the raw "
			            "wrappers cannot expose. Regions may NEST and overlap. "
			            "Finding nothing is a valid result (count = 0).",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Image to search (converted to grayscale internally)."),
				io.Int.Input("delta", default=5, min=1, max=100,
					tooltip="Threshold step over which a region's area must stay "
					        "stable. LARGER = fewer, more solidly stable regions; "
					        "smaller = many more, including near-duplicates."),
				io.Int.Input("min_area", default=60, min=1, max=10000000,
					tooltip="Smallest region area in pixels - raise it to drop "
					        "noise specks."),
				io.Int.Input("max_area", default=14400, min=1, max=10000000,
					tooltip="Largest region area in pixels; caps the whole-image "
					        "region that is otherwise always 'stable'."),
				io.Float.Input("max_variation", default=0.25, min=0.0, max=1.0, step=0.01,
					tooltip="Prune a region whose area varies more than this "
					        "relative amount over 'delta'. Lower = stricter."),
				io.Combo.Input("polarity", options=[cls._BOTH, cls._BRIGHT],
					default=cls._BOTH,
					tooltip="MSER normally runs twice - once on the image and once "
					        "inverted - so it finds dark-on-light AND light-on-dark "
					        "regions. Restrict it to the second pass only when you "
					        "know the polarity and want half the work."),
				io.Float.Input("min_diversity", default=0.2, min=0.0, max=1.0, step=0.01,
					tooltip="Prune a region that is too similar in area to its "
					        "parent - this is what suppresses nested near-duplicates.",
					optional=True, advanced=True),
			],
			outputs=[
				Contours.Output(display_name="regions",
					tooltip="One contour per region (its convex outline) - draw with "
					        "'CV Draw Contours' or measure with 'CV Shape "
					        "Moments'. Convex hulls, so concave detail is in 'mask'."),
				NPArray.Output(display_name="mask",
					tooltip="uint8 0/255 union of the EXACT region pixels (not the "
					        "hulls) - the lossless form of the result."),
				NPArray.Output(display_name="boxes",
					tooltip="Nx4 int32 (x, y, w, h) bounding boxes, one per region."),
				NPArray.Output(display_name="areas",
					tooltip="(N,) float32 exact pixel count of each region."),
				io.Int.Output(display_name="count",
					tooltip="How many regions were found; 0 is valid, not an error."),
			],
		)

	@classmethod
	def execute(cls, image, delta, min_area, max_area, max_variation, polarity,
	            min_diversity=0.2) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		gray = _to_gray_u8(image)
		mser = cv2.MSER_create()
		mser.setDelta(int(delta))
		mser.setMinArea(int(min_area))
		mser.setMaxArea(int(max_area))
		mser.setMaxVariation(float(max_variation))
		mser.setMinDiversity(float(min_diversity))
		mser.setPass2Only(polarity == cls._BRIGHT)
		regions, boxes = mser.detectRegions(gray)
		regions = list(regions or [])
		mask = np.zeros(gray.shape[:2], np.uint8)
		hulls, areas = [], []
		for r in regions:
			pts = np.asarray(r, np.int32).reshape(-1, 2)
			mask[pts[:, 1], pts[:, 0]] = 255
			hulls.append(cv2.convexHull(pts.reshape(-1, 1, 2)))
			areas.append(float(len(pts)))
		bx = (np.asarray(boxes, np.int32).reshape(-1, 4) if len(regions)
		      else np.zeros((0, 4), np.int32))
		return io.NodeOutput(hulls, mask, bx,
		                     np.asarray(areas, np.float32), len(hulls))


class ShapeDistance(io.ComfyNode):
	"""cv2 shape module: ShapeContext / Hausdorff distance between two contours."""

	_SHAPE_CONTEXT = "Shape Context (deformation-aware)"
	_HAUSDORFF = "Hausdorff (worst-case point distance)"
	_NORMS = {
		"L2 (euclidean)": cv2.NORM_L2,
		"L1 (manhattan)": cv2.NORM_L1,
		"C (chebyshev / max)": cv2.NORM_INF,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ShapeDistance",
			display_name="CV Shape Distance",
			category=CATEGORY,
			description="How different are two contours, as one number "
			            "(cv2.ShapeContextDistanceExtractor / "
			            "cv2.HausdorffDistanceExtractor). These live in OpenCV's "
			            "'shape' module as CLASSES, so no raw wrapper reaches them. "
			            "Shape Context builds a log-polar histogram around every "
			            "sample point and solves the correspondence between the two "
			            "shapes, so it tolerates bending and scale - it is "
			            "the strong matcher and much slower. It is NOT rotation "
			            "tolerant unless 'rotation_invariant' is on (measured: a "
			            "40-degree turn costs 1.93 off, 0.010 on). Neither method "
			            "is a mirror detector - use the 'mirror_mode' policy on "
			            "'CV Filter Contours By Shape' for that. Hausdorff is the "
			            "worst-case nearest-point distance: cheap, purely geometric, "
			            "and very sensitive to a single outlier point. Compare with "
			            "cv2_matchShapes (Hu moments), which is faster still but only "
			            "sees seven global numbers. SMALLER = more similar; feed the "
			            "distance into a comparison node to accept/reject a match.",
			inputs=[
				Contours.Input("contours_a",
					tooltip="First shape. A whole contour LIST is accepted; the "
					        "contour selected by 'index_a' is used."),
				Contours.Input("contours_b",
					tooltip="Second shape, compared against the first."),
				io.Combo.Input("method",
					options=[cls._SHAPE_CONTEXT, cls._HAUSDORFF],
					default=cls._SHAPE_CONTEXT,
					tooltip="Shape Context handles deformation and returns a "
					        "unitless matching cost; Hausdorff returns a distance in "
					        "PIXELS, so it only makes sense on shapes that are "
					        "already aligned and at the same scale."),
				io.Int.Input("sample_points", default=100, min=4, max=1000,
					tooltip="Both contours are resampled to this many points before "
					        "comparing, which is what makes contours of different "
					        "lengths comparable. More points = finer and slower "
					        "(Shape Context cost grows quickly)."),
				io.Int.Input("index_a", default=0, min=0, max=10000,
					tooltip="Which contour of 'contours_a' to use (0 = the largest, "
					        "since 'CV Find Contours' sorts by area).",
					optional=True),
				io.Int.Input("index_b", default=0, min=0, max=10000,
					tooltip="Which contour of 'contours_b' to use.", optional=True),
				io.Int.Input("angular_bins", default=12, min=4, max=64,
					tooltip="Shape Context only: angular bins of the log-polar "
					        "histogram.", optional=True, advanced=True),
				io.Int.Input("radial_bins", default=4, min=2, max=32,
					tooltip="Shape Context only: radial bins of the log-polar "
					        "histogram.", optional=True, advanced=True),
				io.Combo.Input("hausdorff_norm", options=list(cls._NORMS),
					default="L2 (euclidean)",
					tooltip="Hausdorff only: the point-to-point norm used.",
					optional=True, advanced=True),
				io.Float.Input("rank_proportion", default=0.6, min=0.0, max=1.0, step=0.05,
					tooltip="Hausdorff only: use this RANK instead of the true "
					        "maximum (0.6 = the 60th-percentile nearest-point "
					        "distance), which makes it robust to a few outliers. "
					        "1.0 is the classic worst-case Hausdorff distance.",
					optional=True, advanced=True),
				io.Boolean.Input("rotation_invariant", default=False,
					tooltip="Shape Context only: measure each log-polar histogram "
					        "against the local tangent instead of a fixed axis, "
					        "which is what actually makes it rotation tolerant "
					        "(measured on a 40-degree turn: 1.930 off, 0.010 on; "
					        "on a 20-degree turn 0.028 off, 0.009 on). It DEGRADES "
					        "past about 160 degrees, and it is not a mirror test - "
					        "a reflected shape lands among the large-rotation "
					        "scores. Off by default so saved graphs keep their "
					        "numbers.",
					optional=True, advanced=True),
			],
			outputs=[
				io.Float.Output(display_name="distance",
					tooltip="Shape dissimilarity - 0 means identical. Unitless for "
					        "Shape Context, PIXELS for Hausdorff."),
			],
		)

	@staticmethod
	def _sample(contours, index, n, which):
		items = list(contours or [])
		if not items:
			raise ValueError(
				f"'{which}' is empty - a shape distance needs two contours. Guard "
				"the branch with the 'count' output of the contour node.")
		if index >= len(items):
			raise ValueError(
				f"index_{which[-1]} = {index} but '{which}' holds {len(items)} "
				"contour(s).")
		pts = np.asarray(items[index], np.float32).reshape(-1, 2)
		if len(pts) < 2:
			raise ValueError(f"'{which}'[{index}] has {len(pts)} point(s).")
		# Resample to a fixed length so contours of different lengths compare -
		# both extractors require the two point sets to line up. It must be
		# uniform along the PERIMETER, not along the point INDEX: a shape
		# context histogram assumes roughly equal arc-length spacing, and
		# CHAIN_APPROX_SIMPLE (what 'CV Find Contours' emits by DEFAULT)
		# stores a polygon as a handful of corners, so index interpolation
		# crams the same number of samples onto a 3 px edge as onto a 90 px one.
		# Measured on a 13-point contour: spacing 0.17-13.33 px (77.8x) against
		# 1.4x here, which inflated distances up to 43x and made the answer
		# depend on the CHAIN_APPROX mode (up to 41x apart) when it must not.
		closed = np.vstack([pts, pts[:1]])
		step = np.linalg.norm(np.diff(closed, axis=0), axis=1)
		cum = np.concatenate([[0.0], np.cumsum(step)])
		total = float(cum[-1])
		if total <= 1e-9:                       # every point identical
			out = np.repeat(pts[:1], int(n), axis=0)
			return np.ascontiguousarray(out.reshape(1, -1, 2), np.float32)
		target = np.linspace(0.0, total, int(n), endpoint=False)
		seg = np.clip(np.searchsorted(cum, target, side="right") - 1,
		              0, len(step) - 1)
		frac = ((target - cum[seg]) / np.maximum(step[seg], 1e-9))[:, None]
		out = closed[seg] * (1.0 - frac) + closed[seg + 1] * frac
		return np.ascontiguousarray(out.reshape(1, -1, 2), np.float32)

	@classmethod
	def execute(cls, contours_a, contours_b, method, sample_points, index_a=0,
	            index_b=0, angular_bins=12, radial_bins=4,
	            hausdorff_norm="L2 (euclidean)", rank_proportion=0.6,
	            rotation_invariant=False) -> io.NodeOutput:
		a = cls._sample(contours_a, int(index_a), sample_points, "contours_a")
		b = cls._sample(contours_b, int(index_b), sample_points, "contours_b")
		# ALWAYS go through the create* factory: `cv2.ShapeContextDistanceExtractor()`
		# constructs the abstract base with no implementation behind it, and every
		# call on it then dies with "Unknown C++ exception from OpenCV code"
		if method == cls._HAUSDORFF:
			ext = cv2.createHausdorffDistanceExtractor()
			ext.setDistanceFlag(cls._NORMS[hausdorff_norm])
			ext.setRankProportion(float(rank_proportion))
		else:
			ext = cv2.createShapeContextDistanceExtractor()
			ext.setAngularBins(int(angular_bins))
			ext.setRadialBins(int(radial_bins))
			ext.setRotationInvariant(bool(rotation_invariant))
		return io.NodeOutput(float(ext.computeDistance(a, b)))


# The columns each 'CV Region Properties' selection emits, in the order
# enums.REGION_PROPERTIES declares them - the feature matrix must not depend on
# the order the user ticked the boxes in.
_POSITION_COLUMNS = {
	"centroid (x, y)": ["centroid_x", "centroid_y"],
	"bbox center (x, y)": ["bbox_center_x", "bbox_center_y"],
	"no position": [],
}
# the 8 neighbours of a pixel, for counting skeleton tips
_NEIGHBOUR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
# A convexity defect shallower than this fraction of sqrt(area) is rasterisation
# noise, not a bite out of the shape: every closed convex test shape measured
# under 0.02 and every genuinely notched one over 0.4.
DEFECT_MIN_DEPTH = 0.05
# cv2.approxPolyDP tolerance for 'corner count', as a fraction of the region's
# own perimeter - so it is scale-free. A flag carries no parameter of its own,
# hence a fixed value; 2% is the usual choice and it resolves a rounded blob
# into ~8 vertices while keeping a quadrilateral at 4.
CORNER_TOLERANCE = 0.02
# The Hu invariants are polynomials of rising DEGREE in the normalized central
# moments: h1 is degree 1, h2/h3/h4 degree 2, h6 degree 3, h5/h7 degree 4. Every
# eta is well under 1 for a region of any real extent, so each extra degree
# costs several orders of magnitude and |h| ends up spread over ~30 of them -
# hence the log.
#
# This is a log(0) GUARD, not a noise threshold, and deliberately so. It is
# tempting to put the floor where an invariant stops being shape and becomes
# rasterisation residue, and that boundary is real - but it is not a constant
# and not a function of region size either. Measured on shapes whose h5/h6/h7
# are ANALYTICALLY zero (so the reading IS the error): at a fixed radius of
# 40px, |h5| residue runs 1.18e-15 for a diamond, 2.68e-18 for a hexagon,
# 2.08e-21 for a 12-gon and exactly 0 for a disc or an axis-aligned square -
# ~47 orders of magnitude apart at the SAME size, set by the shape's symmetry
# order and how its edges land on the pixel grid. Any floor high enough to
# silence the diamond would erase real signal from the rest, and clipping is
# silent where an over-large value is at least visible. So the floor stays at
# the double-precision end and the transform keeps whatever cv2 computed.
HU_FLOOR = 1e-30


def _hu_signed_log(hu):
	"""-> the Hu invariants compressed to their order of magnitude, WITH sign.

	magnitude = log10(|h|) - log10(HU_FLOOR): at or below the floor it is 0 and
	it rises with |h|, so the sign can simply be multiplied back in afterwards.
	Two properties matter, and taking |log10|h|| instead loses both:

	- MONOTONE in |h|. |log10|h|| measures distance from |h| = 1 in either
	  direction, so a vanishing invariant becomes the LARGEST coordinate while a
	  strong one collapses toward 0 - backwards. It also folds: h1 exceeds 1 for
	  an elongated region, which then lands on top of a weak h1 below 1.
	- SIGN survives. h1..h4 are sums of squares, so their sign is positive by
	  construction and a negative value is numerical error - which the floor
	  already reads as zero magnitude, rather than as a large one. h5 and h6 are
	  genuinely sign-indefinite yet reflection-INVARIANT, so their sign is shape
	  information that no magnitude reproduces: two regions can agree on |h5| and
	  |h6| and still be different shapes. h7 alone changes sign under reflection.
	"""
	magnitude = np.log10(np.maximum(np.abs(hu), HU_FLOOR)) - np.log10(HU_FLOOR)
	return np.sign(hu) * magnitude


_REGION_COLUMNS = {
	"area": ["area"],
	"equivalent radius": ["equiv_radius"],
	"bbox width": ["bbox_width"],
	"bbox height": ["bbox_height"],
	"aspect ratio": ["aspect_ratio"],
	"extent": ["extent"],
	"perimeter": ["perimeter"],
	"circularity": ["circularity"],
	"solidity": ["solidity"],
	"orientation (sin/cos)": ["orient_sin", "orient_cos"],
	"eccentricity": ["eccentricity"],
	"mean colour": ["mean_b", "mean_g", "mean_r"],
	"mean intensity": ["mean_intensity"],
	"hu moments (log magnitude)": ["hu1", "hu2", "hu3", "hu4", "hu5", "hu6",
	                              "hu7"],
	"chirality": ["chirality"],
	"min-area rect long side": ["minrect_long"],
	"min-area rect short side": ["minrect_short"],
	"min-area rect aspect (long/short)": ["minrect_aspect"],
	"ellipse major axis": ["ellipse_major"],
	"ellipse minor axis": ["ellipse_minor"],
	"min-area rect centroid clearance (long, short)": ["minrect_clear_long",
	                                                   "minrect_clear_short"],
	"centroid offset in bbox (x, y)": ["centroid_off_x", "centroid_off_y"],
	"canonical orientation (sin/cos)": ["canon_sin", "canon_cos"],
	"orientation confidence": ["orient_confidence"],
	"hole count": ["holes"],
	"min-area rect extent": ["minrect_extent"],
	"loose ends": ["loose_ends"],
	"convexity defects (count, max depth)": ["defect_count", "defect_max_depth"],
	"corner count": ["corners"],
	"median thickness": ["thickness"],
	"thickness variation": ["thickness_variation"],
	"inscribed circle radius": ["inscribed_radius"],
	"roundness (inscribed/enclosing)": ["roundness"],
	"skeleton length": ["skeleton_length"],
	"radial distance variation": ["radial_variation"],
}


class RegionProperties(io.ComfyNode):
	"""regionprops as a node: regions in, a (N, D) feature matrix out.

	The gap this fills: every component node in the pack MEASURES its regions
	and then throws the numbers away - 'Connected Components' computes
	stats/centroids to sort by, `_contour_property` measures six scalars only
	to filter on one of them, and 'Shape Moments' is contour-only with no
	bbox or perimeter columns. Nothing emitted a MATRIX, so anything that
	wanted to cluster, classify or plot regions had to be its own node. That
	is what the deleted CV_KMeansMaskClusters was: this node plus
	cv2.kmeans, with the feature choice frozen into a two-option combo.

	The `labels` output is what makes the round trip work: row i-1 of
	`features` describes label i, so a per-region result (a cluster id, a
	score) goes back over the pixels with 'CV Take By Index'.
	"""

	_SCALES = ["raw (pixels)",
	           "image-relative (0-1)",
	           "standardized (z-score per column)"]

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_RegionProperties",
			display_name="CV Region Properties",
			category=CATEGORY,
			description="Measures every region of a mask and emits one ROW per "
			            "region: position, size, shape ratios, orientation and "
			            "(with an image wired in) mean colour, selected column "
			            "by column with the checkboxes. A single mask is split "
			            "into its connected blobs; a MASK batch (segmentation "
			            "output) is one region per frame. The 'labels' map "
			            "indexes the same regions - row i-1 describes label i - "
			            "so a per-region result can be painted back with 'CV "
			            "Take By Index'. Feed 'features' to cv2.kmeans to group "
			            "regions, to 'CV Train Classifier' to label them, or "
			            "to 'CV Chart Scatter' to look at them. An empty mask "
			            "gives 0 rows, not an error.",
			inputs=[
				io.Mask.Input("mask",
					tooltip="A single mask is split into its connected blobs; a "
					        "MASK batch clusters the masks themselves (one "
					        "region per frame, resized to frame 0's size)."),
				io.Combo.Input("connectivity", options=["8", "4"], default="8",
					tooltip="8 also connects diagonal pixels; 4 only direct "
					        "neighbours. Only used when splitting a single mask."),
				io.String.Input("properties",
					default=enums.REGION_PROPERTIES.initial,
					tooltip="Which columns to emit. The position choice is "
					        "exclusive (a region has one place); everything else "
					        "is an independent toggle, so 'no position | area' "
					        "measures size alone. Columns come out in the order "
					        "listed here, whatever order you tick them in; "
					        "'names' spells the final list out. Most toggles emit "
					        "ONE column, but the two orientation pairs, 'centroid "
					        "offset in bbox' and 'min-area rect centroid "
					        "clearance' emit two, 'mean colour' three and 'hu "
					        "moments (log magnitude)' SEVEN - that last one groups "
					        "regions by SHAPE alone. Each toggle carries its own "
					        "tooltip.",
					extra_dict={"cv_flags": enums.REGION_PROPERTIES.spec()}),
				io.Combo.Input("scale", options=cls._SCALES, default=cls._SCALES[0],
					tooltip="raw keeps pixel units - the honest measurement. "
					        "image-relative divides lengths by the frame size "
					        "and areas by its area, so the numbers survive a "
					        "resize. z-score centres each column and divides by "
					        "its standard deviation, which is what a distance-"
					        "based consumer (k-means, kNN) needs: raw, an area "
					        "in px^2 outweighs a centroid in px by thousands."),
				io.Int.Input("min_area", default=0, min=0, max=2**31 - 1,
					optional=True, advanced=True,
					tooltip="Regions smaller than this many pixels are dropped "
					        "(0 keeps every region). Specks are what make a "
					        "cluster count meaningless."),
				io.Combo.Input("on_empty",
					options=["empty table (0 rows)", "one all-zero row"],
					default="empty table (0 rows)", optional=True, advanced=True,
					tooltip="What to emit when the mask has no regions at all. "
					        "The empty table is the honest answer, but a "
					        "consumer that cannot take 0 rows - cv2.kmeans "
					        "asserts on an empty sample set - needs the padded "
					        "row instead. 'count' stays 0 either way, so the "
					        "row is never mistaken for a measurement."),
				io.Image.Input("image", optional=True, advanced=True,
					tooltip="Only needed for the 'mean colour' / 'mean "
					        "intensity' columns: the picture the mask belongs "
					        "to, at the same size. Frame 0 of a batch."),
			],
			outputs=[
				NPArray.Output(display_name="features",
					tooltip="(N, D) float32, one row per region in label order. "
					        "N = 0 for an empty mask."),
				NPArray.Output(display_name="labels",
					tooltip="[H,W] int32 label map: 0 = background, i = the "
					        "region described by row i-1. Feed it to 'CV Take By "
					        "Index' with a per-region table, or to 'OpenCV "
					        "Labels to Masks' to get the regions back."),
				io.String.Output(display_name="names",
					tooltip="The column names, comma separated - wire it into "
					        "'Preview as Text' to read the matrix."),
				io.Int.Output(display_name="count",
					tooltip="N, the number of regions measured."),
			],
		)

	@classmethod
	def _regions(cls, frames, connectivity, min_area, shape):
		"""-> ([region dict], [H,W] int32 label map). One dict per kept region:
		area/cx/cy/bbox plus the region's own uint8 crop for the shape work."""
		h, w = shape
		conn = int(connectivity)
		regions = []
		label_map = np.zeros((h, w), np.int32)
		if len(frames) == 1:
			binary = cv2.threshold(frames[0], 127, 255, cv2.THRESH_BINARY)[1]
			n, lab, stats, centroids = cv2.connectedComponentsWithStats(
				binary, connectivity=conn)
			remap = np.zeros(max(n, 1), np.int32)  # old label -> new (0 = dropped)
			for i in range(1, n):  # 0 is the background
				area = int(stats[i, cv2.CC_STAT_AREA])
				if area < min_area:
					continue
				x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
				bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
				crop = np.where(lab[y:y + bh, x:x + bw] == i, 255, 0).astype(np.uint8)
				regions.append({"area": area, "cx": float(centroids[i][0]),
				                "cy": float(centroids[i][1]), "x": x, "y": y,
				                "w": bw, "h": bh, "crop": crop, "conn": conn})
				remap[i] = len(regions)
			if n > 1:
				label_map = remap[lab]
		else:
			for frame in frames:
				binary = cv2.threshold(imageutils.fit_mask_to(frame, (h, w)),
				                       127, 255, cv2.THRESH_BINARY)[1]
				area = int(cv2.countNonZero(binary))
				if area == 0 or area < min_area:
					continue
				x, y, bw, bh = cv2.boundingRect(binary)
				m = cv2.moments(binary, binaryImage=True)
				regions.append({"area": area, "cx": float(m["m10"] / m["m00"]),
				                "cy": float(m["m01"] / m["m00"]), "x": x, "y": y,
				                "w": bw, "h": bh, "conn": conn,
				                "crop": binary[y:y + bh, x:x + bw]})
				# overlapping masks: the later region wins the shared pixels.
				# The MEASUREMENTS still see every mask whole - only the label
				# map has to pick one owner per pixel.
				label_map[binary > 0] = len(regions)
		return regions, label_map

	@classmethod
	def _contour(cls, region):
		"""Largest external contour of a region's crop, or None."""
		if "contour" not in region:
			found = cv2.findContours(region["crop"], cv2.RETR_EXTERNAL,
			                         cv2.CHAIN_APPROX_SIMPLE)[0]
			region["contour"] = max(found, key=cv2.contourArea) if found else None
		return region["contour"]

	@classmethod
	def _moments(cls, region):
		if "moments" not in region:
			region["moments"] = cv2.moments(region["crop"], binaryImage=True)
		return region["moments"]

	@classmethod
	def _rect(cls, region):
		"""Raw cv2.minAreaRect of the region: ((cx, cy), (w, h), angle) or None.

		Crop coordinates, like everything else derived from `region["crop"]`.
		"""
		if "rect" not in region:
			contour = cls._contour(region)
			region["rect"] = None if contour is None else cv2.minAreaRect(contour)
		return region["rect"]

	@classmethod
	def _min_rect(cls, region):
		"""-> (long side, short side) of the region's minimum-area rectangle.

		Sorted, never (w, h): cv2.minAreaRect's own w/h swap as the rectangle
		turns past its angle convention, so a raw w/h pair would flip between
		two rows of the SAME shape at different orientations - exactly what
		these columns exist to avoid.
		"""
		if "min_rect" not in region:
			rect = cls._rect(region)
			if rect is None:
				region["min_rect"] = (0.0, 0.0)
			else:
				(_, _), (w, h), _ = rect
				region["min_rect"] = (max(w, h), min(w, h))
		return region["min_rect"]

	@classmethod
	def _clearance(cls, region):
		"""-> (clearance across the LONG side, across the short one).

		The shortest distance from the region's centroid to each opposing pair
		of min-area rect walls, so it says how far off-centre the mass sits
		inside the region's own oriented box. Paired with the SIDE LENGTHS, not
		with cv2's (w, h) axes, for the same reason `_min_rect` sorts: those two
		swap as the rectangle turns past the angle convention, and an
		axis-ordered pair would be noise across rows of the same shape.

		Unsigned - which end of a min-area rect is "first" is not stable, so
		there is no direction to carry a sign - hence half the side is the
		maximum, reached exactly by a centrally symmetric region.
		"""
		if "clearance" not in region:
			rect, m = cls._rect(region), cls._moments(region)
			if rect is None or m["m00"] <= shapepose.DEGENERATE_M00:
				region["clearance"] = (0.0, 0.0)
			else:
				(rx, ry), (w, h), angle = rect
				dx = float(m["m10"] / m["m00"]) - rx
				dy = float(m["m01"] / m["m00"]) - ry
				rad = math.radians(angle)
				cos_a, sin_a = math.cos(rad), math.sin(rad)
				# the rect's own axes: w runs along u, h along v
				pairs = [(w, dx * cos_a + dy * sin_a),
				         (h, -dx * sin_a + dy * cos_a)]
				pairs.sort(key=lambda pair: -pair[0])
				region["clearance"] = tuple(
					max(0.0, side / 2.0 - abs(off)) for side, off in pairs)
		return region["clearance"]

	@classmethod
	def _pose(cls, region):
		"""-> shapepose.pose() for the region, or None when degenerate."""
		if "pose" not in region:
			hu = cls._hu(region)
			region["pose"] = (None if hu is None
			                  else shapepose.pose(cls._moments(region), hu))
		return region["pose"]

	@classmethod
	def _holes(cls, region):
		"""-> the number of background regions the crop fully encloses.

		Padded by one pixel so the outside is always one single component, and
		counted with the connectivity DUAL to the one the foreground used -
		8-connected foreground with 8-connected background would let a diagonal
		one-pixel gap be a wall and a doorway at the same time.
		"""
		if "holes" not in region:
			pad = cls._padded(region)
			dual = 4 if int(region.get("conn", 8)) == 8 else 8
			n = cv2.connectedComponents(255 - pad, connectivity=dual)[0]
			region["holes"] = max(0, n - 2)  # -1 for the label 0, -1 for outside
		return region["holes"]

	@classmethod
	def _padded(cls, region):
		"""The region's crop with a one-pixel background border.

		Shared by every skeleton/distance column so they index the SAME array,
		and it is the difference between right and nonsense rather than a
		nicety: the crop is the region's TIGHT bbox, so the shape touches all
		four edges. Thinning a stroke that the border cuts off leaves a spur at
		the cut (unpadded, a bar reads 6 tips instead of 2, a closed ring 8 and
		a disc 4), and the distance transform would read the frame edge as
		background-free and hand every edge pixel a distance it does not have.
		Same rule `_holes` follows, for the same reason.
		"""
		if "padded" not in region:
			region["padded"] = cv2.copyMakeBorder(region["crop"], 1, 1, 1, 1,
			                                      cv2.BORDER_CONSTANT, value=0)
		return region["padded"]

	@classmethod
	def _skeleton(cls, region):
		"""-> the region's medial axis as a padded 0/1 uint8 array.

		cv2.ximgproc.thinning is used when the build has it (18x faster) and
		`_thin_zhang_suen` otherwise; the two produce skeletons that differ
		pixel-wise but agree on the tip COUNT in every case measured, which
		`test_region_properties_loose_ends` pins so a build where they diverge
		fails loudly.
		"""
		if "skeleton" not in region:
			ximgproc = _thinning_available()
			padded = cls._padded(region)
			region["skeleton"] = (
				(ximgproc.thinning(padded,
				                   thinningType=ximgproc.THINNING_ZHANGSUEN) > 0
				 ).astype(np.uint8)
				if ximgproc is not None else _thin_zhang_suen(padded))
		return region["skeleton"]

	@classmethod
	def _distance(cls, region):
		"""-> the exact euclidean distance transform of the padded crop.

		DIST_MASK_PRECISE rather than the 3x3/5x5 approximations: it is the
		difference between an exact thickness and a chamfer estimate, and the
		crops are small.
		"""
		if "distance" not in region:
			region["distance"] = cv2.distanceTransform(
				cls._padded(region), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
		return region["distance"]

	@classmethod
	def _thickness(cls, region):
		"""-> (median local thickness in px, std/mean of it), 0.0 when empty.

		The distance transform is the radius of the largest circle centred on
		each pixel that stays inside the region, so on the MEDIAL AXIS twice it
		is the local thickness. Restricting it to the axis is the whole point:
		over ALL the region's pixels the edge ones dominate and a 9 px bar
		measures 6.5 while a perfectly uniform slab's "variation" reads 0.495.
		"""
		if "thickness" not in region:
			axis = cls._distance(region)[cls._skeleton(region) > 0]
			if axis.size == 0:
				region["thickness"] = (0.0, 0.0)
			else:
				mean = float(axis.mean())
				region["thickness"] = (
					2.0 * float(np.median(axis)),
					float(axis.std() / mean) if mean > 0 else 0.0)
		return region["thickness"]

	@classmethod
	def _loose_ends(cls, region):
		"""-> the number of free ends of the region's skeleton.

		A skeleton pixel with exactly ONE neighbour is a tip.
		"""
		if "loose_ends" not in region:
			skeleton = cls._skeleton(region)
			neighbours = cv2.filter2D(skeleton, cv2.CV_8U, _NEIGHBOUR_KERNEL,
			                          borderType=cv2.BORDER_CONSTANT) * skeleton
			region["loose_ends"] = int((neighbours == 1).sum())
		return region["loose_ends"]

	@classmethod
	def _enclosing(cls, region):
		"""-> radius of the minimum enclosing circle of the region's outline."""
		if "enclosing" not in region:
			contour = cls._contour(region)
			region["enclosing"] = (0.0 if contour is None else
			                       float(cv2.minEnclosingCircle(contour)[1]))
		return region["enclosing"]

	@classmethod
	def _radial_variation(cls, region):
		"""-> std/mean of the centroid-to-outline distance over EVERY outline
		pixel.

		CHAIN_APPROX_NONE on purpose, and it is not an optimisation to undo:
		`_contour` stores the SIMPLE approximation, whose vertices crowd where
		the outline bends, so the same statistic taken over them would be
		vertex-density weighted - the trap `shapepose` documents for the
		third-order moments.
		"""
		if "radial_variation" not in region:
			found = cv2.findContours(cls._padded(region), cv2.RETR_EXTERNAL,
			                         cv2.CHAIN_APPROX_NONE)[0]
			region["radial_variation"] = 0.0
			if found:
				points = max(found, key=len).reshape(-1, 2).astype(np.float64)
				m = cls._moments(region)
				# crop coordinates + 1 for the pad the outline was traced in
				centre = (np.array([m["m10"], m["m01"]]) / m["m00"] + 1.0
				          if m["m00"] > 0 else points.mean(axis=0))
				radius = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1])
				mean = float(radius.mean())
				if mean > 0:
					region["radial_variation"] = float(radius.std() / mean)
		return region["radial_variation"]

	@classmethod
	def _defects(cls, region):
		"""-> (how many convexity defects are deeper than DEFECT_MIN_DEPTH of
		sqrt(area), the deepest one as a fraction of sqrt(area)).

		The depth threshold and the normalisation are both relative to
		sqrt(area) so the pair is scale-free, and cv2 reports depths in 1/256
		of a pixel. Anything cv2 refuses to analyse (fewer than 4 points, a
		hull it cannot walk) is 0 defects rather than an exception - rule 4.
		"""
		if "defects" not in region:
			region["defects"] = (0, 0.0)
			contour = cls._contour(region)
			if contour is not None and len(contour) >= 4:
				try:
					hull = cv2.convexHull(contour, returnPoints=False)
					found = (cv2.convexityDefects(contour, hull)
					         if hull is not None and len(hull) >= 4 else None)
				except cv2.error:
					found = None
				if found is not None and len(found):
					depths = np.asarray(found).reshape(-1, 4)[:, 3] / 256.0
					scale = math.sqrt(max(float(cv2.contourArea(contour)), 1.0))
					relative = depths / scale
					region["defects"] = (int((relative > DEFECT_MIN_DEPTH).sum()),
					                     float(relative.max()))
		return region["defects"]

	@classmethod
	def _corners(cls, region):
		"""-> vertices left after simplifying the outline to CORNER_TOLERANCE of
		its own perimeter (cv2.approxPolyDP)."""
		if "corners" not in region:
			contour = cls._contour(region)
			region["corners"] = 0 if contour is None or len(contour) < 3 else len(
				cv2.approxPolyDP(
					contour, CORNER_TOLERANCE * cv2.arcLength(contour, True), True))
		return region["corners"]

	@classmethod
	def _hu(cls, region):
		"""-> the raw cv2.HuMoments vector of a region, or None if degenerate."""
		if "hu" not in region:
			m = cls._moments(region)
			region["hu"] = (None if m["m00"] <= shapepose.DEGENERATE_M00
			                else cv2.HuMoments(m).ravel())
		return region["hu"]

	@classmethod
	def _column(cls, name, regions, colours):
		"""-> (values array (N, C), scaling unit per column)."""
		n = len(regions)
		if name == "area":
			return (np.array([[r["area"]] for r in regions], np.float64), ["area"])
		if name == "equivalent radius":
			return (np.array([[math.sqrt(r["area"])] for r in regions], np.float64),
			        ["length"])
		if name == "bbox width":
			return (np.array([[r["w"]] for r in regions], np.float64), ["x"])
		if name == "bbox height":
			return (np.array([[r["h"]] for r in regions], np.float64), ["y"])
		if name == "aspect ratio":
			return (np.array([[r["w"] / r["h"] if r["h"] else 0.0]
			                  for r in regions], np.float64), ["ratio"])
		if name == "extent":
			# the region's PIXEL area over its bbox - the exact count, where
			# `_contour_property` has only the polygon area to work with
			return (np.array([[r["area"] / (r["w"] * r["h"]) if r["w"] * r["h"] else 0.0]
			                  for r in regions], np.float64), ["ratio"])
		if name in ("perimeter", "circularity", "solidity"):
			prop = "perimeter" if name == "perimeter" else name
			vals = []
			for r in regions:
				contour = cls._contour(r)
				vals.append([0.0 if contour is None
				             else _contour_property(contour, prop)])
			return (np.array(vals, np.float64),
			        ["length" if name == "perimeter" else "ratio"])
		if name == "orientation (sin/cos)":
			vals = []
			for r in regions:
				angle = shapepose.second_order(cls._moments(r))[0]
				# 2*angle: the principal axis is defined mod 180, so doubling it
				# makes the pair continuous all the way round
				vals.append([math.sin(math.radians(2.0 * angle)),
				             math.cos(math.radians(2.0 * angle))])
			return np.array(vals, np.float64), ["ratio", "ratio"]
		if name == "eccentricity":
			return (np.array([[shapepose.second_order(cls._moments(r))[1]]
			                  for r in regions], np.float64), ["ratio"])
		if name == "min-area rect long side":
			return (np.array([[cls._min_rect(r)[0]] for r in regions], np.float64),
			        ["length"])
		if name == "min-area rect short side":
			return (np.array([[cls._min_rect(r)[1]] for r in regions], np.float64),
			        ["length"])
		if name == "min-area rect aspect (long/short)":
			# long/short, so it is always >= 1 and cannot depend on which way
			# up the region happens to lie - unlike 'aspect ratio', which is
			# the UPRIGHT bbox's w/h and inverts as the shape passes 45 degrees
			vals = []
			for r in regions:
				lo, sh = cls._min_rect(r)
				vals.append([lo / sh if sh > 0 else 0.0])
			return np.array(vals, np.float64), ["ratio"]
		if name == "min-area rect extent":
			# 'extent' with a box that TURNS with the region: a rectangle reads
			# 1.0 at any angle, where the upright version drops to 0.5 on the
			# diagonal. Both terms come from the CONTOUR, deliberately - the
			# pixel count would be the wrong numerator here, because
			# cv2.minAreaRect spans contour points, i.e. pixel CENTRES, so a
			# 10x10 square measures 9x9 and pixels/rect gives 1.23 for a shape
			# that is exactly rectangular. contourArea uses the same convention
			# and cancels it (measured: 1.000 at every angle).
			vals = []
			for r in regions:
				lo, sh = cls._min_rect(r)
				contour = cls._contour(r)
				vals.append([float(cv2.contourArea(contour)) / (lo * sh)
				             if contour is not None and lo * sh > 0 else 0.0])
			return np.array(vals, np.float64), ["ratio"]
		if name in ("ellipse major axis", "ellipse minor axis"):
			# the equivalent ellipse of the second-order moments: `half` is
			# 2 sigma along the major axis and ecc^2 = 1 - l2/l1, so the minor
			# axis is half*sqrt(1 - ecc^2) = 2*sqrt(l2) - no second eigenvalue
			# to plumb out of shapepose, and the two stay consistent by
			# construction
			vals = []
			for r in regions:
				_, ecc, half = shapepose.second_order(cls._moments(r))
				vals.append([half if name.endswith("major axis")
				             else half * math.sqrt(max(0.0, 1.0 - ecc * ecc))])
			return np.array(vals, np.float64), ["length"]
		if name == "min-area rect centroid clearance (long, short)":
			return (np.array([list(cls._clearance(r)) for r in regions],
			                 np.float64), ["length", "length"])
		if name == "centroid offset in bbox (x, y)":
			return (np.array([[(r["cx"] - r["x"]) / r["w"] if r["w"] else 0.0,
			                   (r["cy"] - r["y"]) / r["h"] if r["h"] else 0.0]
			                  for r in regions], np.float64), ["ratio", "ratio"])
		if name == "canonical orientation (sin/cos)":
			# sin/cos of t, NOT of 2t: this angle is defined mod 360 (the
			# third-order moments pick the head from the tail), so unlike
			# 'orientation (sin/cos)' it must not be doubled - doubling would
			# throw away exactly the half-turn this column exists to keep
			vals = []
			for r in regions:
				p = cls._pose(r)
				angle = 0.0 if p is None else math.radians(p["orientation"])
				vals.append([math.sin(angle), math.cos(angle)])
			return np.array(vals, np.float64), ["ratio", "ratio"]
		if name == "orientation confidence":
			vals = [[0.0 if cls._pose(r) is None else cls._pose(r)["confidence"]]
			        for r in regions]
			return np.array(vals, np.float64), ["ratio"]
		if name == "hole count":
			return (np.array([[cls._holes(r)] for r in regions], np.float64),
			        ["ratio"])
		if name == "loose ends":
			return (np.array([[cls._loose_ends(r)] for r in regions], np.float64),
			        ["ratio"])
		if name == "convexity defects (count, max depth)":
			return (np.array([list(cls._defects(r)) for r in regions], np.float64),
			        ["ratio", "ratio"])
		if name == "corner count":
			return (np.array([[cls._corners(r)] for r in regions], np.float64),
			        ["ratio"])
		if name == "median thickness":
			return (np.array([[cls._thickness(r)[0]] for r in regions], np.float64),
			        ["length"])
		if name == "thickness variation":
			return (np.array([[cls._thickness(r)[1]] for r in regions], np.float64),
			        ["ratio"])
		if name == "inscribed circle radius":
			return (np.array([[float(cls._distance(r).max())] for r in regions],
			                 np.float64), ["length"])
		if name == "roundness (inscribed/enclosing)":
			vals = []
			for r in regions:
				enclosing = cls._enclosing(r)
				vals.append([float(cls._distance(r).max()) / enclosing
				             if enclosing > 0 else 0.0])
			return np.array(vals, np.float64), ["ratio"]
		if name == "skeleton length":
			return (np.array([[int(cls._skeleton(r).sum())] for r in regions],
			                 np.float64), ["length"])
		if name == "radial distance variation":
			return (np.array([[cls._radial_variation(r)] for r in regions],
			                 np.float64), ["ratio"])
		if name == "hu moments (log magnitude)":
			# raw |h| spans ~30 orders of magnitude, which no k-means column can
			# use, so _hu_signed_log rescales each to its order of magnitude and
			# restores the sign - see that helper for why the sign has to come
			# back rather than being folded away with abs().
			vals = []
			for r in regions:
				hu = cls._hu(r)
				if hu is None:
					vals.append([0.0] * 7)
					continue
				column = _hu_signed_log(hu)
				# h7 is the ONE invariant whose sign is handedness: it flips
				# when the region is mirrored, so keeping it would drive a shape
				# and its reflection into separate clusters. Only that column is
				# reduced to a magnitude; h5 and h6 are reflection-invariant and
				# keep theirs. Tick 'chirality' to get h7's sign back on its own.
				column[6] = abs(column[6])
				vals.append(list(column))
			return np.array(vals, np.float64), ["ratio"] * 7
		if name == "chirality":
			vals = []
			for r in regions:
				hu = cls._hu(r)
				vals.append([0.0 if hu is None else shapepose.chirality(hu)])
			return np.array(vals, np.float64), ["ratio"]
		if name == "mean colour":
			return colours[:, :3].astype(np.float64), ["colour"] * 3
		if name == "mean intensity":
			# BGR -> gray with the cvtColor weights, so it matches a grayscale
			# conversion of the same picture
			gray = (colours[:, 0] * 0.114 + colours[:, 1] * 0.587
			        + colours[:, 2] * 0.299)
			return gray.reshape(n, 1), ["colour"]
		raise ValueError(f"Unknown region property {name!r}.")

	@classmethod
	def _mean_colours(cls, image, label_map, count, shape):
		"""(N, 3) mean BGR per label, 0 where the label has no pixels."""
		h, w = shape
		if image is None:
			raise ValueError(
				"The 'mean colour' / 'mean intensity' columns need the 'image' "
				"input - wire the picture the mask belongs to."
			)
		bgr = imageutils.image_batch_to_bgr(image)[0]
		if bgr.shape[:2] != (h, w):
			raise ValueError(
				f"image is {bgr.shape[1]}x{bgr.shape[0]} but the mask is {w}x{h} "
				"- the colour columns sample the image THROUGH the mask, so the "
				"two must be the same size."
			)
		flat = label_map.reshape(-1)
		counts = np.bincount(flat, minlength=count + 1)[:count + 1]
		out = np.zeros((count, 3), np.float64)
		for c in range(3):
			sums = np.bincount(flat, weights=bgr[:, :, c].reshape(-1).astype(np.float64),
			                   minlength=count + 1)[:count + 1]
			filled = counts[1:] > 0
			out[filled, c] = sums[1:][filled] / counts[1:][filled]
		return out

	@classmethod
	def _rescale(cls, values, units, shape):
		h, w = shape
		divisors = {"x": float(w), "y": float(h), "length": math.sqrt(w * h),
		            "area": float(w * h), "ratio": 1.0, "colour": 255.0}
		return values / np.array([divisors[u] for u in units], np.float64)

	@classmethod
	def execute(cls, mask, connectivity="8", properties=None, scale=None,
	            min_area=0, on_empty="empty table (0 rows)",
	            image=None) -> io.NodeOutput:
		scale = scale or cls._SCALES[0]
		base, flags = enums.REGION_PROPERTIES.parse(
			enums.REGION_PROPERTIES.initial if properties in (None, "") else properties)
		names = list(_POSITION_COLUMNS[base])
		for flag in flags:
			names += _REGION_COLUMNS[flag]
		if not names:
			raise ValueError(
				"'properties' selects no columns at all - pick a position "
				"and/or tick at least one property."
			)

		frames = imageutils.mask_batch_to_u8(mask)
		h, w = frames[0].shape[:2]
		regions, label_map = cls._regions(frames, connectivity, int(min_area), (h, w))

		if not regions:
			# an empty mask is a real outcome (nothing detected upstream), not a
			# wiring mistake - emit a matrix of the right WIDTH, with as many
			# rows as `on_empty` asks for. The padded row exists because
			# cv2.kmeans asserts on an empty sample set: it keeps a clustering
			# graph running (every pixel stays background, so no mask is
			# produced) instead of stopping it with an OpenCV assertion.
			rows = 1 if on_empty.startswith("one") else 0
			return io.NodeOutput(np.zeros((rows, len(names)), np.float32), label_map,
			                     ", ".join(names), 0)

		columns, units = [], []
		if base == "centroid (x, y)":
			columns.append(np.array([[r["cx"], r["cy"]] for r in regions], np.float64))
			units += ["x", "y"]
		elif base == "bbox center (x, y)":
			columns.append(np.array([[r["x"] + r["w"] / 2.0, r["y"] + r["h"] / 2.0]
			                         for r in regions], np.float64))
			units += ["x", "y"]

		colours = None
		if any(f in ("mean colour", "mean intensity") for f in flags):
			colours = cls._mean_colours(image, label_map, len(regions), (h, w))
		for flag in flags:
			values, flag_units = cls._column(flag, regions, colours)
			columns.append(values)
			units += flag_units

		features = np.concatenate(columns, axis=1)
		if scale.startswith("image-relative"):
			features = cls._rescale(features, units, (h, w))
		elif scale.startswith("standardized"):
			std = features.std(axis=0)
			# a constant column (one region, or every region the same size) has
			# no spread to divide by - leave it centred at 0 rather than emit inf
			features = (features - features.mean(axis=0)) / np.where(std > 1e-12, std, 1.0)
		return io.NodeOutput(np.ascontiguousarray(features, np.float32), label_map,
		                     ", ".join(names), len(regions))


class RegionPropertyFlags(io.ComfyNode):
	"""The canvas half of 'CV Region Properties'' checkbox widget.

	A subgraph PROMOTES an input as a bare socket - the widget stays on the
	node inside the definition - so an instance of a blueprint that wraps
	Region Properties ('CV K-Means Mask Clusters') has nowhere to tick a box.
	Same problem the cv2 flag params have, same answer: author the value once
	in a node that owns the widget and wire the STRING out. It is also how one
	selection drives SEVERAL Region Properties nodes without drifting apart.
	"""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_RegionPropertyFlags",
			display_name="CV Region Property Flags",
			category=CATEGORY,
			description="Authors an 'CV Region Properties' column selection "
			            "once and fans it out: wire the 'properties' STRING "
			            "output into the properties input of any number of "
			            "Region Properties nodes - or into the promoted "
			            "'properties' input of a subgraph that wraps one ('CV "
			            "K-Means Mask Clusters'), which is a bare socket with no "
			            "widget of its own. The position choice is exclusive; "
			            "every other box is an independent toggle. 'names' and "
			            "'columns' describe the matrix the selection will "
			            "produce, without running anything.",
			inputs=[
				io.String.Input("properties",
					default=enums.REGION_PROPERTIES.initial,
					tooltip="Pipe-joined column names, e.g. \"no position | "
					        "area\" to cluster by size alone, or \"no position "
					        "| hu moments (log magnitude)\" to cluster by SHAPE "
					        "alone. In the UI this renders as the position "
					        "dropdown plus one toggle per property, each with "
					        "its own tooltip.",
					extra_dict={"cv_flags": enums.REGION_PROPERTIES.spec()}),
			],
			outputs=[
				io.String.Output(display_name="properties",
					tooltip="The pipe-joined selection - wire it into the "
					        "'properties' input of an 'CV Region "
					        "Properties' node or of a subgraph that promotes "
					        "one."),
				io.String.Output(display_name="names",
					tooltip="The column names the selection expands to, comma "
					        "separated and in emission order (the two orientation "
					        "pairs, 'centroid offset in bbox' and 'min-area rect "
					        "centroid clearance' are two columns each, 'mean "
					        "colour' three, 'hu moments (log magnitude)' seven - "
					        "hu1..hu7). "
					        "Wire it into 'Preview as Text'."),
				io.Int.Output(display_name="columns",
					tooltip="D, the width of the feature matrix - the number of "
					        "names, not the number of boxes ticked."),
			],
		)

	@classmethod
	def execute(cls, properties) -> io.NodeOutput:
		# parse() validates the names, so a typo is loud here rather than four
		# nodes downstream inside the subgraph
		base, flags = enums.REGION_PROPERTIES.parse(
			enums.REGION_PROPERTIES.initial if properties in (None, "") else properties)
		names = list(_POSITION_COLUMNS[base])
		for flag in flags:
			names += _REGION_COLUMNS[flag]
		return io.NodeOutput(properties, ", ".join(names), len(names))


NODES = [FindContours, DrawContours, SelectContour, FilterContours,
         FilterContoursByShape, MergeContours, FindQuadrilateral,
         QuadToRectangle, QuadWarp, EnclosingCircle, ShapeMoments,
         ContourToPoints, PointsToContour, Skeletonize,
         SelectComponentAtPoint, KeepLargestComponent,
         ComponentsTouchingBorder, FillHoles, RegionProperties,
         RegionPropertyFlags, DetectMSERRegions, ShapeDistance]
