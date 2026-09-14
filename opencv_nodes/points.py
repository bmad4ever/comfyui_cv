# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Point-set utilities (NPARRAY-native).

Generic building blocks for working with point arrays (Nx1x2 float32, the
format cv2.goodFeaturesToTrack / matchers / contours produce): filtering,
polar statistics, k-means clustering, per-cluster reduction, scalar picking
and a debug renderer. They exist so that geometric pipelines (e.g. the
clock-reading exercise) can be wired from small reusable nodes instead of
one-off task-specific ones.

Conventions: every node tolerates None or empty point sets (a valid "nothing
detected" result - cv2.goodFeaturesToTrack returns None) and outputs empty
arrays / zero counts instead of raising.
"""

import json
import logging
import math
from ast import literal_eval

import cv2
import numpy as np

from comfy_api.latest import io, ui

from . import enums, imageutils, polytype
from .convert import NPArray
from .features import (LINE_STYLE_TOOLTIP, LINE_STYLES, MARKER_SHAPE_TOOLTIP,
                       MARKER_SHAPES, _index_color, _make_draw_canvas,
                       _to_bgr_u8, draw_marker, draw_styled_line,
                       draw_text_outlined, halo_color, parse_color)

CATEGORY = "image/CV/points"

_EMPTY = np.zeros((0, 1, 2), np.float32)


def _drawable(x, y) -> bool:
	"""Is this point safe to hand cv2's integer drawing API?

	A NON-FINITE point means "no position for this item" - the same convention
	`scene_depth` uses for "no measurement here". The drawing nodes SKIP those
	instead of raising, so an overlay whose points are gated on something
	(a visibility test, a failed detection) can blank an entry out and keep the
	rest of the array index-aligned with its labels. int(NaN) raises ValueError,
	so without this guard a single hidden point takes down the whole overlay.
	"""
	return bool(np.isfinite(x)) and bool(np.isfinite(y))


def _as_points(points) -> np.ndarray:
	"""Nx2 float64 view of a point array; None/empty -> (0, 2)."""
	if points is None:
		return np.zeros((0, 2), np.float64)
	arr = np.asarray(points, np.float64)
	if arr.size == 0:
		return np.zeros((0, 2), np.float64)
	return arr.reshape(-1, 2)


def _to_points(arr: np.ndarray) -> np.ndarray:
	return np.float32(arr).reshape(-1, 1, 2)


Y_AXIS_MODES = ("y down (pixel rows)", "y up (world)")

Y_AXIS_TOOLTIP = (
	"Which way the second coordinate grows on the canvas. Pixel rows grow "
	"DOWNWARD, so mapping a world axis straight onto them draws it upside "
	"down - for a top-down (XZ) map that mirrors the picture and a left turn "
	"comes out looking like a right turn. Choose 'y up (world)' whenever the "
	"second coordinate is a world axis you want read as a map; leave it 'y "
	"down' when the points are already in pixel space."
)


def _fit_box_mapping(pts, width, height, margin, flip_y=False):
	"""Uniform world -> pixel mapping that centres `pts` in a width x height box.

	Returns (scale, offset_x, offset_y, sign_y) with
	    pixel = (x * scale + offset_x,  sign_y * y * scale + offset_y)
	so 'Fit Points To Box' and 'Draw Plot Frame' cannot disagree about where a
	world coordinate lands - including which way up it lands. A single point or
	a zero-extent set maps to the box centre at scale 1 instead of dividing by
	zero.
	"""
	sign = -1.0 if flip_y else 1.0
	if len(pts) == 0:
		return 1.0, width / 2.0, height / 2.0, sign
	flipped = pts * np.array([1.0, sign])
	lo, hi = flipped.min(axis=0), flipped.max(axis=0)
	span = float(np.max(hi - lo))
	usable = max(1.0, min(width, height) - 2.0 * margin)
	scale = usable / span if span > 1e-9 else 1.0
	mid = (lo + hi) / 2.0
	return (scale, width / 2.0 - mid[0] * scale,
	        height / 2.0 - mid[1] * scale, sign)


def _ramp_colors(n, colormap, n_channels):
	"""n colours walking a sequential colormap from start to end of a path.

	A sequential ramp (VIRIDIS by default, never JET) is what makes a drawn
	path readable as TIME rather than as a shape. On a 1- or 2-channel canvas
	there is no colour to spend, so the ramp becomes a luminance ramp - which
	is exactly the point: it still reads.
	"""
	steps = np.linspace(0, 255, max(n, 1)).astype(np.uint8).reshape(1, -1)
	if n_channels < 3:
		return [(int(v),) * n_channels for v in steps.ravel()]
	bgr = cv2.applyColorMap(steps, enums.COLORMAP.resolve(colormap)).reshape(-1, 3)
	pad = (255,) * (n_channels - 3)
	return [tuple(int(v) for v in c) + pad for c in bgr]


def _back_along(pts, i, distance):
	"""The point `distance` pixels back along the polyline from pts[i].

	Consecutive trajectory samples can be a fraction of a pixel apart, so an
	arrowhead drawn between two neighbours is invisible. Walking back a fixed
	arc length gives every arrow the same readable size whatever the speed."""
	x, y = pts[i]
	travelled = 0.0
	for j in range(i, 0, -1):
		seg = float(np.hypot(*(pts[j] - pts[j - 1])))
		if travelled + seg >= distance:
			f = (distance - travelled) / seg if seg > 1e-9 else 0.0
			return pts[j] + (pts[j - 1] - pts[j]) * f
		travelled += seg
	return pts[0] if len(pts) else np.array([x, y])


def _nice_step(span, target):
	"""A 1 / 2 / 5 x 10^k step giving roughly `target` intervals across span."""
	if span <= 0 or target < 1:
		return 1.0
	raw = span / float(target)
	mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
	for mult in (1.0, 2.0, 5.0, 10.0):
		if raw <= mult * mag:
			return mult * mag
	return 10.0 * mag


class _AnnotateImagePreview(ui.PreviewImage):
	"""Saves the preview exactly like ui.PreviewImage, but reports it under a
	custom ui key: the annotation overlay widget renders the image itself, so
	the frontend's built-in node preview must not also draw it. An optional
	second image (the correspondence editor's destination) travels under its
	own key - the browser can only ever show pixels the server previewed, which
	is why the background is a real (backend) input and not a frontend-only
	socket: a generated background never touches disk, so there is no URL for
	the JS to load until execute() saves one."""

	def __init__(self, image, cls=None, background=None):
		super().__init__(image, cls=cls)
		self.background_values = (ui.PreviewImage(background, cls=cls).values
		                          if background is not None else None)

	def as_dict(self):
		d = {"annotate_images": self.values}
		if self.background_values is not None:
			d["annotate_background"] = self.background_values
		return d


class AnnotatePoints(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_AnnotatePoints",
			display_name="CV Annotate Points",
			category=CATEGORY,
			description="Manual point input: run the workflow once to load the "
			            "image into the node, then LEFT-CLICK the preview to "
			            "add a point (drag to move it), SHIFT+CLICK to remove "
			            "the nearest one. Once the outline is closed (3 or more "
			            "points) a click is INSERTED into the nearest edge - the "
			            "highlighted one - so a shape can be refined where it is "
			            "wrong instead of in click order; ALT+CLICK appends at "
			            "the end instead. The points are stored normalized "
			            "(0-1 of the image size) in the 'points' field - "
			            "editable by hand too - and come out in pixel "
			            "coordinates, in ring order. The mask output fills "
			            "the polygon the points enclose (3 or more, any "
			            "count). Pair with 'CV Quad Warp' (4 corners), "
			            "'CV Select Component At Point' (a seed), 'OpenCV "
			            "Draw Points', ... Zero points is a valid output "
			            "(count = 0).",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="The image to annotate; it is shown inside the "
					        "node after the first run."),
				io.String.Input("points", default="[]",
					tooltip="Normalized [[x, y], ...] JSON, 0-1 of the image "
					        "size. Maintained by clicking the preview below; "
					        "hand-editing works as well."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="Nx1x2 float32 PIXEL coordinates, in click order."),
				io.Int.Output(display_name="count"),
				NPArray.Output(display_name="mask",
					tooltip="uint8 image-sized mask, 255 inside the polygon "
					        "the points enclose (click order, any count >= 3; "
					        "fewer points -> all zeros)."),
			],
			is_output_node=True,
		)

	@classmethod
	def execute(cls, image, points) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		frame = _to_bgr_u8(image)
		h, w = frame.shape[:2]
		pixels = _parse_norm_points(points) * [w - 1, h - 1]
		preview = _AnnotateImagePreview(imageutils.bgr_to_image_batch([frame]), cls=cls)
		return io.NodeOutput(_to_points(pixels), len(pixels),
		                     _points_mask(pixels, h, w), ui=preview)


def _parse_norm_points(points: str) -> np.ndarray:
	"""(N, 2) float64 normalized coordinates from the annotation JSON."""
	try:
		data = json.loads(points) if points.strip() else []
	except json.JSONDecodeError as e:
		raise ValueError(
			f'points must be JSON like [[0.25, 0.5], ...], got "{points}". ({e})'
		) from e
	if not isinstance(data, list) or not all(
			isinstance(p, (list, tuple)) and len(p) == 2
			and all(isinstance(v, (int, float)) for v in p) for p in data):
		raise ValueError(
			f'points must be a JSON list of [x, y] pairs, got "{points}".'
		)
	if not data:
		return np.zeros((0, 2), np.float64)
	return np.clip(np.float64(data).reshape(-1, 2), 0.0, 1.0)


def _points_mask(pixels: np.ndarray, height: int, width: int) -> np.ndarray:
	"""uint8 mask with the polygon of the points (click order) filled white.

	Generic in the point count: any N >= 3 encloses an area - exactly the
	outline the annotation UI draws, self-intersections fill even-odd -
	while fewer points cannot, giving an all-zeros mask (failure-tolerant,
	like every point node)."""
	mask = np.zeros((height, width), np.uint8)
	if len(pixels) >= 3:
		cv2.fillPoly(mask, [np.round(pixels).astype(np.int32)], 255,
		             cv2.LINE_AA)
	return mask


class AnnotateCorrespondences(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_AnnotateCorrespondences",
			display_name="CV Annotate Correspondences",
			category=CATEGORY,
			description="Manual point CORRESPONDENCES: run the workflow once to "
			            "load the image into the node, then LEFT-CLICK the "
			            "preview to add a pair (it starts pinned, from = to) and "
			            "DRAG either end - cyan is where a point IS, orange is "
			            "where it must LAND, and the arrow between them is the "
			            "displacement. SHIFT+CLICK removes a pair, ALT+DRAG "
			            "moves both ends together, DOUBLE-CLICK re-pins one. The "
			            "node draws a LIVE preview of the resulting warp so the "
			            "correspondences can be authored by eye. Pairs are stored "
			            "normalized (0-1 of the image size) in the 'pairs' field "
			            "- editable by hand too - and come out as two "
			            "row-aligned pixel point sets, which is exactly what "
			            "'CV Thin Plate Spline Warp', 'CV Find "
			            "Homography (RANSAC)', 'cv2.estimateAffine2D' and "
			            "'cv2.getPerspectiveTransform' consume. Because both "
			            "halves come from ONE widget they can never fall out of "
			            "alignment. Zero pairs is a valid output (count = 0). "
			            "With the optional 'background' connected the TO half of "
			            "every pair lives on THAT image instead (the editor shows "
			            "the two side by side), so the fitted warp maps image -> "
			            "background: the authoring surface for pasting the image "
			            "somewhere into the background.",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="The image the correspondences are placed on; it is "
					        "shown inside the node after the first run. The "
					        "outputs are PIXEL coordinates of THIS image, so a "
					        "consumer must be fed the same size."),
				io.String.Input("pairs", default="[]",
					tooltip="Normalized [[[from_x, from_y], [to_x, to_y]], ...] "
					        "JSON, 0-1 of the image size (of the background's "
					        "size for the 'to' half when a background is "
					        "connected). Maintained by dragging the preview "
					        "below; hand-editing works as well."),
				polytype.poly_input("background", types=polytype.IMAGE_ONLY,
					optional=True,
					tooltip="Optional DESTINATION image. When connected, the "
					        "editor shows it next to 'image', every pair's TO "
					        "end lives on it, and points_to comes out in ITS "
					        "pixel coordinates - feed points_from/points_to to a "
					        "homography/affine fit to warp 'image' into place. "
					        "Left unconnected, both halves live on 'image' as "
					        "before."),
			],
			outputs=[
				NPArray.Output(display_name="points_from",
					tooltip="Nx1x2 float32 PIXEL coordinates - where each control "
					        "point currently is, in creation order. Always in "
					        "'image' pixels."),
				NPArray.Output(display_name="points_to",
					tooltip="Nx1x2 float32 PIXEL coordinates - where the "
					        "same-numbered control point must land. Row-aligned "
					        "with points_from by construction. In 'background' "
					        "pixels when one is connected, 'image' pixels "
					        "otherwise."),
				io.Int.Output(display_name="count"),
			],
			is_output_node=True,
		)

	@classmethod
	def execute(cls, image, pairs, background=None) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		frame = _to_bgr_u8(image)
		h, w = frame.shape[:2]
		# normalized 1.0 -> pixel w - 1 (the AnnotatePoints convention; the JS
		# mesh divides by w - 1 to match). The 'to' half denormalizes against
		# the DESTINATION frame - the background when one is connected.
		bg_batch = None
		bh, bw = h, w
		if background is not None:
			bg, _ = polytype.resolve(background)
			bg_frame = _to_bgr_u8(bg)
			bh, bw = bg_frame.shape[:2]
			bg_batch = imageutils.bgr_to_image_batch([bg_frame])
		norm = _parse_norm_pairs(pairs)
		pts_from = norm[:, 0] * [w - 1, h - 1]
		pts_to = norm[:, 1] * [bw - 1, bh - 1]
		preview = _AnnotateImagePreview(imageutils.bgr_to_image_batch([frame]),
		                                cls=cls, background=bg_batch)
		return io.NodeOutput(_to_points(pts_from), _to_points(pts_to),
		                     len(norm), ui=preview)


def _parse_norm_pairs(pairs: str) -> np.ndarray:
	"""(N, 2, 2) float64 normalized [[from, to], ...] from the editor JSON."""
	shape = "[[[from_x, from_y], [to_x, to_y]], ...]"
	try:
		data = json.loads(pairs) if pairs.strip() else []
	except json.JSONDecodeError as e:
		raise ValueError(
			f'pairs must be JSON like {shape}, got "{pairs}". ({e})'
		) from e
	ok = isinstance(data, list) and all(
		isinstance(pair, (list, tuple)) and len(pair) == 2
		and all(isinstance(p, (list, tuple)) and len(p) == 2
		        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
		                for v in p)
		        for p in pair)
		for pair in data)
	if not ok:
		raise ValueError(
			f'pairs must be a JSON list of {shape} correspondences - a flat list '
			f'of [x, y] points (the "CV Annotate Points" format) is not one. '
			f'Got "{pairs}".')
	if not data:
		return np.zeros((0, 2, 2), np.float64)
	return np.clip(np.float64(data).reshape(-1, 2, 2), 0.0, 1.0)


class FilterPointsByDistance(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FilterPointsByDistance",
			display_name="CV Filter Points By Distance",
			category=CATEGORY,
			description="Keeps the points whose distance to an origin lies in "
			            "[min_dist, max_dist] - e.g. corners in an annular "
			            "band around a dial center, matches near an anchor. "
			            "None or zero input points are valid (count = 0).",
			inputs=[
				NPArray.Input("points", tooltip="Nx1x2 point array (or None)."),
				io.Float.Input("origin_x", default=0.0, min=-1e6, max=1e6,
					tooltip="X pixel coordinate of the reference origin (e.g. the "
					        "dial center, anchor, or ray start)."),
				io.Float.Input("origin_y", default=0.0, min=-1e6, max=1e6,
					tooltip="Y pixel coordinate of the reference origin (e.g. the "
					        "dial center, anchor, or ray start)."),
				io.Float.Input("min_dist", default=0.0, min=0.0, max=1e6,
					tooltip="Keep points at least this far (pixels) from the origin."),
				io.Float.Input("max_dist", default=1e6, min=0.0, max=1e6,
					tooltip="Keep points at most this far (pixels) from the origin."),
			],
			outputs=[
				NPArray.Output(display_name="points"),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, points, origin_x, origin_y, min_dist, max_dist) -> io.NodeOutput:
		pts = _as_points(points)
		d = np.hypot(pts[:, 0] - origin_x, pts[:, 1] - origin_y)
		kept = pts[(d >= min_dist) & (d <= max_dist)]
		return io.NodeOutput(_to_points(kept), len(kept))


class FilterPointsByMask(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FilterPointsByMask",
			display_name="CV Filter Points By Mask",
			category=CATEGORY,
			description="Keeps the points flagged 1 in a per-point mask - e.g. the "
			            "RANSAC inliers from 'CV Find Fundamental Matrix' / "
			            "'CV Find Homography (RANSAC)'. Run two of these with "
			            "the SAME mask to filter a matched pair in lockstep (order "
			            "is preserved), so 'CV Stereo Rectify (Uncalibrated)' "
			            "and cv2.triangulatePoints get clean correspondences. None "
			            "or zero input points are valid (count = 0); a None mask "
			            "passes every point through unchanged.",
			inputs=[
				NPArray.Input("points", tooltip="Nx1x2 point array (or None)."),
				NPArray.Input("mask", optional=True,
					tooltip="Nx1 uint8 mask: entries != 0 are kept. Must have one "
					        "entry per point. None = keep all points."),
			],
			outputs=[
				NPArray.Output(display_name="points"),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, points, mask=None) -> io.NodeOutput:
		pts = _as_points(points)
		if len(pts) == 0 or mask is None or np.asarray(mask).size == 0:
			return io.NodeOutput(_to_points(pts), len(pts))
		keep = np.asarray(mask).ravel().astype(bool)
		if len(keep) != len(pts):
			raise ValueError(
				f"mask has {len(keep)} entries but there are {len(pts)} points - "
				"the mask must come from the same matched point set."
			)
		kept = pts[keep]
		return io.NodeOutput(_to_points(kept), len(kept))


class PointsToPolar(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PointsToPolar",
			display_name="CV Points To Polar",
			category=CATEGORY,
			description="Polar coordinates of each point relative to an "
			            "origin: distance, angle in degrees [0, 360) and the "
			            "unit direction vector. The direction points are ideal "
			            "k-means features when grouping points by bearing "
			            "(points along one ray collapse to a single direction "
			            "regardless of how far out they sit) - cluster them "
			            "with the 'CV K-Means Points' subgraph blueprint while "
			            "keeping the original points for drawing.",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx1x2 point array to convert (or None -> 0 points)."),
				io.Float.Input("origin_x", default=0.0, min=-1e6, max=1e6,
					tooltip="X pixel coordinate of the reference origin (e.g. the "
					        "dial center, anchor, or ray start)."),
				io.Float.Input("origin_y", default=0.0, min=-1e6, max=1e6,
					tooltip="Y pixel coordinate of the reference origin (e.g. the "
					        "dial center, anchor, or ray start)."),
				io.Combo.Input("angle_convention",
					options=["clockwise from 12 o'clock (clock)",
					         "counter-clockwise from +x (math)"],
					default="clockwise from 12 o'clock (clock)",
					tooltip="Clock convention: 12h = 0, 3h = 90, 6h = 180. "
					        "Math convention: +x = 0, growing counter-"
					        "clockwise (y up)."),
			],
			outputs=[
				NPArray.Output(display_name="radius", tooltip="(N,) float32 distances."),
				NPArray.Output(display_name="angle", tooltip="(N,) float32 degrees [0, 360)."),
				NPArray.Output(display_name="directions",
					tooltip="Nx1x2 float32 unit vectors (0,0 for points on "
					        "the origin) - k-means features for grouping by "
					        "bearing."),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, points, origin_x, origin_y, angle_convention) -> io.NodeOutput:
		pts = _as_points(points)
		dx, dy = pts[:, 0] - origin_x, pts[:, 1] - origin_y
		radius = np.hypot(dx, dy)
		if angle_convention.startswith("clockwise"):
			angle = np.degrees(np.arctan2(dx, -dy)) % 360
		else:
			angle = np.degrees(np.arctan2(-dy, dx)) % 360
		angle[radius == 0] = 0.0  # atan2(+0, -0) would say 180
		safe = np.maximum(radius, 1e-9)
		directions = np.stack([dx / safe, dy / safe], axis=1)
		directions[radius == 0] = 0
		return io.NodeOutput(radius.astype(np.float32), angle.astype(np.float32),
		                     _to_points(directions), len(pts))


class PolarToPoints(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PolarToPoints",
			display_name="CV Polar To Points",
			category=CATEGORY,
			description="The inverse of 'CV Points To Polar': builds the "
			            "point at each (radius, angle) around an origin - e.g. "
			            "turn detected bearings back into image coordinates to "
			            "render them (the endpoint of each clock hand), or "
			            "place markers around a circle. A single-element "
			            "radius (or angle) array broadcasts against the other "
			            "array; empty inputs yield zero points.",
			inputs=[
				NPArray.Input("radius", tooltip="(N,) distances from the origin."),
				NPArray.Input("angle", tooltip="(N,) angles in degrees."),
				io.Float.Input("origin_x", default=0.0, min=-1e6, max=1e6,
					tooltip="X pixel coordinate of the reference origin (e.g. the "
					        "dial center, anchor, or ray start)."),
				io.Float.Input("origin_y", default=0.0, min=-1e6, max=1e6,
					tooltip="Y pixel coordinate of the reference origin (e.g. the "
					        "dial center, anchor, or ray start)."),
				io.Combo.Input("angle_convention",
					options=["clockwise from 12 o'clock (clock)",
					         "counter-clockwise from +x (math)"],
					default="clockwise from 12 o'clock (clock)",
					tooltip="Must match the convention the angles were "
					        "measured in."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="Nx1x2 float32 image coordinates."),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, radius, angle, origin_x, origin_y, angle_convention) -> io.NodeOutput:
		rad = (np.zeros((0,), np.float64) if radius is None
		       else np.asarray(radius, np.float64).ravel())
		ang = (np.zeros((0,), np.float64) if angle is None
		       else np.asarray(angle, np.float64).ravel())
		if len(rad) == 0 or len(ang) == 0:
			return io.NodeOutput(_EMPTY, 0)
		if len(rad) != len(ang):
			if len(rad) == 1:
				rad = np.full_like(ang, rad[0])
			elif len(ang) == 1:
				ang = np.full_like(rad, ang[0])
			else:
				raise ValueError(
					f"radius has {len(rad)} entries but angle has {len(ang)} - "
					"they must match (or one of them must be a single value)."
				)
		theta = np.radians(ang)
		if angle_convention.startswith("clockwise"):
			dx, dy = rad * np.sin(theta), -rad * np.cos(theta)
		else:
			dx, dy = rad * np.cos(theta), -rad * np.sin(theta)
		pts = np.stack([origin_x + dx, origin_y + dy], axis=1)
		return io.NodeOutput(_to_points(pts), len(pts))


class ReducePointsByLabel(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ReducePointsByLabel",
			display_name="CV Reduce Points By Label",
			category=CATEGORY,
			description="Collapses a labeled point set (from 'CV K-Means "
			            "Points') to one representative point per cluster: the "
			            "centroid, or the member farthest from / closest to an "
			            "origin (e.g. the tip of each clock hand, the "
			            "innermost match of each group). Rows follow the label "
			            "order (0, 1, ...), so outputs of several reductions "
			            "over the same labels stay aligned. The distances "
			            "output (representative-to-origin) doubles as a "
			            "sort key for 'CV Pick Value'.",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx1x2 labeled points (same order as 'labels')."),
				NPArray.Input("labels", tooltip="(N,) integer label per point."),
				io.Combo.Input("reduction",
					options=["centroid", "farthest from origin", "closest to origin"],
					default="centroid",
					tooltip="Representative per cluster: its centroid (mean), or the "
					        "member farthest from / closest to the origin."),
				io.Float.Input("origin_x", default=0.0, min=-1e6, max=1e6,
					tooltip="X pixel coordinate of the reference origin (e.g. the "
					        "dial center, anchor, or ray start)."),
				io.Float.Input("origin_y", default=0.0, min=-1e6, max=1e6,
					tooltip="Y pixel coordinate of the reference origin (e.g. the "
					        "dial center, anchor, or ray start)."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="Kx1x2 float32, one point per label."),
				NPArray.Output(display_name="distances",
					tooltip="(K,) float32 distance of each representative to "
					        "the origin."),
				io.Int.Output(display_name="count", tooltip="K, the number of labels."),
			],
		)

	@classmethod
	def execute(cls, points, labels, reduction, origin_x, origin_y) -> io.NodeOutput:
		pts = _as_points(points)
		lbl = (np.zeros((0,), np.int32) if labels is None
		       else np.asarray(labels).ravel().astype(np.int32))
		if len(pts) != len(lbl):
			raise ValueError(
				f"{len(pts)} points but {len(lbl)} labels - both must come "
				"from the same clustering run (e.g. the 'CV K-Means Points' "
				"blueprint)."
			)
		reduced = []
		for label in np.unique(lbl):
			members = pts[lbl == label]
			if reduction == "centroid":
				reduced.append(members.mean(axis=0))
			else:
				d = np.hypot(members[:, 0] - origin_x, members[:, 1] - origin_y)
				pick = np.argmax(d) if reduction.startswith("farthest") else np.argmin(d)
				reduced.append(members[pick])
		reduced = np.array(reduced) if reduced else np.zeros((0, 2))
		distances = np.hypot(reduced[:, 0] - origin_x, reduced[:, 1] - origin_y) \
			if len(reduced) else np.zeros((0,))
		return io.NodeOutput(_to_points(reduced), distances.astype(np.float32),
		                     len(reduced))


def _circular_mean(vals: np.ndarray, period: float) -> float:
	phase = vals * (2 * np.pi / period)
	mean = np.arctan2(np.sin(phase).mean(), np.cos(phase).mean())
	return float((mean * period / (2 * np.pi)) % period)


class ReduceValuesByLabel(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ReduceValuesByLabel",
			display_name="CV Reduce Values By Label",
			category=CATEGORY,
			description="Collapses a labeled value array (e.g. the per-point "
			            "angles of 'CV Points To Polar' + the labels of "
			            "the 'CV K-Means Points' blueprint) to one number per "
			            "cluster: "
			            "the mean, the median, or the mode - the center of the "
			            "densest window of the cluster's value distribution. "
			            "The mode is the robust choice when a minority of "
			            "contaminating members sits to one side (a numeral "
			            "fused to a clock hand shifts the mean AND the median, "
			            "but the hand's collinear pixels still form the densest "
			            "spike). Rows follow the label order (0, 1, ...), "
			            "aligned with 'CV Reduce Points By Label' over the "
			            "same labels. Empty input yields empty outputs. For an "
			            "IMAGE-scale reduction (multi-channel, a 2-D label map, "
			            "rows indexed by label value) use 'CV Reduce Array By "
			            "Label' instead - this one is a per-POINT reducer and "
			            "its mode path is quadratic in the cluster size.",
			inputs=[
				NPArray.Input("values", tooltip="(N,) numeric array, one value per point."),
				NPArray.Input("labels", tooltip="(N,) integer label per value."),
				io.Combo.Input("reduction",
					options=["mean", "median", "mode (densest value)"],
					default="mean",
					tooltip="Per-cluster summary. mode (densest window) is the robust "
					        "choice when a minority of contaminating values skews the "
					        "mean and median."),
				io.Float.Input("bandwidth", default=2.0, min=1e-6, max=1e6,
					tooltip="Mode only: half-width of the density window. The "
					        "value with the most neighbors within +/-bandwidth "
					        "wins; the result is the mean of that window."),
				io.Float.Input("circular_period", default=0.0, min=0.0, max=1e6,
					tooltip="Treat values as circular with this period (360 for "
					        "angles in degrees, so 359 and 1 are 2 apart); 0 = "
					        "plain numbers."),
			],
			outputs=[
				NPArray.Output(display_name="values",
					tooltip="(K,) float32, one value per label."),
				NPArray.Output(display_name="support",
					tooltip="(K,) int32: members inside the winning mode window "
					        "(= cluster size for mean/median). Low support "
					        "relative to the cluster size means no clear spike."),
				io.Int.Output(display_name="count", tooltip="K, the number of labels."),
			],
		)

	@classmethod
	def execute(cls, values, labels, reduction, bandwidth, circular_period) -> io.NodeOutput:
		vals = (np.zeros((0,), np.float64) if values is None
		        else np.asarray(values, np.float64).ravel())
		lbl = (np.zeros((0,), np.int32) if labels is None
		       else np.asarray(labels).ravel().astype(np.int32))
		if len(vals) != len(lbl):
			raise ValueError(
				f"{len(vals)} values but {len(lbl)} labels - both must "
				"describe the same point set."
			)
		period = circular_period
		reduced, support = [], []
		for label in np.unique(lbl):
			members = vals[lbl == label]
			if reduction == "mean":
				reduced.append(_circular_mean(members, period) if period > 0
				               else members.mean())
				support.append(len(members))
			elif reduction == "median":
				if period > 0:  # rotate the circular mean to mid-period first
					shift = period / 2 - _circular_mean(members, period)
					reduced.append((np.median((members + shift) % period) - shift) % period)
				else:
					reduced.append(np.median(members))
				support.append(len(members))
			else:  # mode: densest +/-bandwidth window (O(n^2), small clusters)
				diff = members[:, None] - members[None, :]
				if period > 0:
					diff = (diff + period / 2) % period - period / 2
				within = np.abs(diff) <= bandwidth
				pick = int(np.argmax(within.sum(axis=1)))
				window = members[within[pick]]
				reduced.append(_circular_mean(window, period) if period > 0
				               else window.mean())
				support.append(len(window))
		return io.NodeOutput(np.float32(reduced), np.int32(support), len(reduced))


class PickValue(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PickValue",
			display_name="CV Pick Value (sorted)",
			category=CATEGORY,
			description="Picks one scalar out of a small array: sorts the "
			            "values (or sorts by the optional key array, e.g. "
			            "'the angle of the hand with the largest radius') and "
			            "returns the rank-th one. The rank is clamped to the "
			            "array, and an empty input yields 0.0 / index -1 "
			            "instead of an error - gate on a found flag upstream.",
			inputs=[
				NPArray.Input("values", tooltip="(N,) numeric array."),
				io.Combo.Input("order", options=["largest first", "smallest first"],
					default="largest first",
					tooltip="Sort direction before applying rank (by values, or by "
					        "'key' if connected)."),
				io.Int.Input("rank", default=1, min=1, max=4096,
					tooltip="1 = first in the chosen order; clamped to N."),
				NPArray.Input("key", optional=True,
					tooltip="Sort by these values instead (same length); the "
					        "output still comes from 'values'."),
			],
			outputs=[
				io.Float.Output(display_name="value"),
				io.Int.Output(display_name="index",
					tooltip="Position in the input array (-1 when empty)."),
			],
		)

	@classmethod
	def execute(cls, values, order, rank, key=None) -> io.NodeOutput:
		vals = (np.zeros((0,), np.float64) if values is None
		        else np.asarray(values, np.float64).ravel())
		if len(vals) == 0:
			return io.NodeOutput(0.0, -1)
		sort_key = vals
		if key is not None:
			sort_key = np.asarray(key, np.float64).ravel()
			if len(sort_key) != len(vals):
				raise ValueError(
					f"values has {len(vals)} entries but key has {len(sort_key)} - "
					"both must describe the same items."
				)
		idx = np.argsort(sort_key)
		if order == "largest first":
			idx = idx[::-1]
		picked = int(idx[min(rank, len(vals)) - 1])
		return io.NodeOutput(float(vals[picked]), picked)


class DrawPoints(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawPoints",
			display_name="CV Draw Points",
			category=CATEGORY,
			description="Debug helper: marks each point with a circle, square, "
			            "cross, star, diamond or triangle. Connect the labels "
			            "from the 'CV K-Means Points' blueprint to "
			            "color the points by cluster (the same label always "
			            "gets the same color). When two point sets go onto the "
			            "SAME image, give them different SHAPES or radii, not "
			            "just different colors - shape is the half of the "
			            "distinction that survives colour blindness, greyscale "
			            "and a busy photo underneath. None/empty points pass "
			            "the image through unchanged.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made)."),
				NPArray.Input("points",
					tooltip="Nx1x2 points to mark (or None -> passthrough)."),
				io.Int.Input("marker_radius", default=5, min=1, max=128,
					tooltip="Radius of each filled circle marker, in pixels."),
				io.String.Input("color", default="(0, 255, 0)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(0, 255, 0)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated. "
					        "Ignored when labels are connected."),
				NPArray.Input("labels", optional=True,
					tooltip="(N,) integer labels: one color per cluster."),
				io.Combo.Input("marker_shape", options=list(MARKER_SHAPES),
					default="filled circle", optional=True,
					tooltip=MARKER_SHAPE_TOOLTIP),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "image",
				tooltip="Same format as the image input.")],
		)

	@classmethod
	def execute(cls, source, points, marker_radius, color, labels=None,
	            marker_shape="filled circle") -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		bgr = parse_color(color, n_channels)
		pts = _as_points(points)
		lbl = None
		if labels is not None and np.asarray(labels).size:
			lbl = np.asarray(labels).ravel().astype(np.int32)
			if len(lbl) != len(pts):
				raise ValueError(
					f"{len(pts)} points but {len(lbl)} labels - they must "
					"describe the same point set."
				)
		bgr = parse_color(color, n_channels)
		for i, (x, y) in enumerate(pts):
			if not _drawable(x, y):
				continue  # blanked-out point: skip it, keep the labels aligned
			c = _index_color(int(lbl[i]), n_channels) if lbl is not None else bgr
			draw_marker(canvas, x, y, marker_radius, c, marker_shape)
		return io.NodeOutput(polytype.restore(canvas, tag))


class DrawConnections(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawConnections",
			display_name="CV Draw Connections",
			category=CATEGORY,
			description="Debug helper: draws a line for every (i, j) EDGE between "
			            "points[i] and points[j] - a skeleton / graph over a point "
			            "set. Feed the 'landmarks' + 'connections' of 'OpenCV "
			            "MediaPipe Hand Pose' to draw the hand bones, or any Ex2 "
			            "index-pair array. Out-of-range indices are skipped; "
			            "None/empty points or edges pass the image through unchanged "
			            "(count=0). Connect labels to color each edge by cluster, "
			            "matching 'CV Draw Points'.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made)."),
				NPArray.Input("points",
					tooltip="Nx1x2 points the edges index into (or None)."),
				NPArray.Input("edges",
					tooltip="Ex2 int index pairs; each row (i, j) connects points[i] "
					        "to points[j]. None/empty -> passthrough."),
				io.Int.Input("thickness", default=2, min=1, max=64,
					tooltip="Line width in pixels."),
				io.String.Input("color", default="(255, 255, 255)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(255, 255, 255)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated. "
					        "Ignored when labels are connected."),
				NPArray.Input("labels", optional=True,
					tooltip="(E,) integer labels: one color per edge cluster."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="Same format as the image input."),
				io.Int.Output(display_name="count",
					tooltip="How many edges were actually drawn (valid indices)."),
			],
		)

	@classmethod
	def execute(cls, source, points, edges, thickness, color, labels=None) -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		pts = _as_points(points)
		eds = (np.zeros((0, 2), np.int64) if edges is None
		       else np.asarray(edges).reshape(-1, 2).astype(np.int64))
		lbl = None
		if labels is not None and np.asarray(labels).size:
			lbl = np.asarray(labels).ravel().astype(np.int32)
			if len(lbl) != len(eds):
				raise ValueError(
					f"{len(eds)} edges but {len(lbl)} labels - they must "
					"describe the same set."
				)
		bgr = parse_color(color, n_channels)
		count = 0
		for k, (i, j) in enumerate(eds):
			if not (0 <= i < len(pts) and 0 <= j < len(pts)):
				continue  # skip dangling indices instead of raising
			if not (_drawable(*pts[i]) and _drawable(*pts[j])):
				continue  # an edge to a blanked-out point has no position
			c = _index_color(int(lbl[k]), n_channels) if lbl is not None else bgr
			cv2.line(canvas, (int(round(pts[i, 0])), int(round(pts[i, 1]))),
			         (int(round(pts[j, 0])), int(round(pts[j, 1]))), c,
			         thickness, cv2.LINE_AA)
			count += 1
		return io.NodeOutput(polytype.restore(canvas, tag), count)


class DrawLabels(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawLabels",
			display_name="CV Draw Labels",
			category=CATEGORY,
			description="Debug helper: writes a text label next to each point. Feed "
			            "the 'landmarks' + 'keypoint_names' of 'CV MediaPipe Hand "
			            "Pose' to name every keypoint, or any per-point string/number "
			            "array. When 'labels' is left unconnected each point is "
			            "labelled with its own index (0, 1, 2, ...). None/empty points "
			            "pass the image through unchanged (count=0).",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made)."),
				NPArray.Input("points",
					tooltip="Nx1x2 points to label (or None -> passthrough)."),
				io.Float.Input("font_scale", default=0.4, min=0.1, max=8.0, step=0.1,
					tooltip="cv2 text scale; shrink it for dense point sets."),
				io.String.Input("color", default="(0, 255, 255)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(0, 255, 255)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated."),
				io.Int.Input("thickness", default=1, min=1, max=32,
					tooltip="Text stroke width in pixels."),
				NPArray.Input("labels", optional=True,
					tooltip="(N,) ARRAY of strings or numbers, one per point - "
					        "not a comma-separated string. Type them with 'CV "
					        "Text To Array' (one per line), or wire an array that "
					        "already exists: a QR/barcode detector's 'texts', "
					        "'CV Load Labels'.labels_array, a classifier's "
					        "'labels_out', or any numeric array via 'CV Numbers "
					        "To Array'. Leave it unwired to caption each point "
					        "with its INDEX, which is also what a point whose "
					        "position is NaN keeps - blanked points are skipped "
					        "without renumbering the rest."),
				io.Int.Input("offset_x", default=4, min=-4096, max=4096,
					optional=True, advanced=True,
					tooltip="Horizontal pixel offset of the text from its point."),
				io.Int.Input("offset_y", default=-4, min=-4096, max=4096,
					optional=True, advanced=True,
					tooltip="Vertical pixel offset of the text from its point."),
				io.Boolean.Input("outline", default=True, optional=True,
					tooltip="Draw a contrasting halo behind each label (black "
					        "behind a light 'color', white behind a dark one). "
					        "These labels land on whatever the image happens to "
					        "show, so a single flat colour at font_scale 0.4 is "
					        "often unreadable without it."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="Same format as the image input."),
				io.Int.Output(display_name="count",
					tooltip="How many labels were drawn."),
			],
		)

	@classmethod
	def execute(cls, source, points, font_scale, color, thickness,
	            labels=None, offset_x=4, offset_y=-4,
	            outline=True) -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		pts = _as_points(points)
		if labels is not None and np.asarray(labels).size:
			text = [str(v) for v in np.asarray(labels).ravel().tolist()]
			if len(text) != len(pts):
				raise ValueError(
					f"{len(pts)} points but {len(text)} labels - they must "
					"describe the same point set."
				)
		else:
			text = [str(i) for i in range(len(pts))]
		bgr = parse_color(color, n_channels)
		halo = halo_color(bgr)
		for (x, y), label in zip(pts, text):
			if not _drawable(x, y):
				continue  # blanked-out point: skip it, keep the labels aligned
			org = (int(round(x)) + int(offset_x), int(round(y)) + int(offset_y))
			if outline:
				draw_text_outlined(canvas, label, org, float(font_scale), halo,
				                   bgr, int(thickness))
			else:
				cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX,
				            float(font_scale), bgr, int(thickness), cv2.LINE_AA)
		return io.NodeOutput(polytype.restore(canvas, tag), len(pts))


class DrawRays(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawRays",
			display_name="CV Draw Rays",
			category=CATEGORY,
			description="Debug helper: draws a line from a shared origin to "
			            "each point - render the FINAL detected bearings on "
			            "top of the raw evidence (build the endpoints from "
			            "angles with 'CV Polar To Points'). Connect labels "
			            "to color each ray by cluster, matching 'CV Draw "
			            "Points'. None/empty points pass the image through "
			            "unchanged.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made)."),
				NPArray.Input("points", tooltip="Nx1x2 ray endpoints."),
				io.Float.Input("origin_x", default=0.0, min=-1e6, max=1e6,
					tooltip="X pixel coordinate of the reference origin (e.g. the "
					        "dial center, anchor, or ray start)."),
				io.Float.Input("origin_y", default=0.0, min=-1e6, max=1e6,
					tooltip="Y pixel coordinate of the reference origin (e.g. the "
					        "dial center, anchor, or ray start)."),
				io.Int.Input("thickness", default=2, min=1, max=64,
					tooltip="Ray line width in pixels."),
				io.String.Input("color", default="(0, 255, 255)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(0, 255, 255)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated. "
					        "Ignored when labels are connected."),
				NPArray.Input("labels", optional=True,
					tooltip="(N,) integer labels: one color per cluster."),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "image",
				tooltip="Same format as the image input.")],
		)

	@classmethod
	def execute(cls, source, points, origin_x, origin_y, thickness, color, labels=None) -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		pts = _as_points(points)
		lbl = None
		if labels is not None and np.asarray(labels).size:
			lbl = np.asarray(labels).ravel().astype(np.int32)
			if len(lbl) != len(pts):
				raise ValueError(
					f"{len(pts)} points but {len(lbl)} labels - they must "
					"describe the same point set."
				)
		bgr = parse_color(color, n_channels)
		origin = (int(round(origin_x)), int(round(origin_y)))
		for i, (x, y) in enumerate(pts):
			c = _index_color(int(lbl[i]), n_channels) if lbl is not None else bgr
			cv2.line(canvas, origin, (int(round(x)), int(round(y))), c,
			         thickness, cv2.LINE_AA)
		return io.NodeOutput(polytype.restore(canvas, tag))


class DrawSegments(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawSegments",
			display_name="CV Draw Segments",
			category=CATEGORY,
			description="Debug helper: draws line segments given as endpoint "
			            "quadruples (x1, y1, x2, y2) - exactly what "
			            "cv2_HoughLinesP outputs (Nx1x4). Connect labels to "
			            "color segments by cluster, matching 'CV Draw "
			            "Points'. Two segment sets on the SAME image should "
			            "differ in line STYLE as well as color - a dashed set "
			            "over a solid one reads for every viewer, a green set "
			            "over a red one does not. None/empty segments pass the "
			            "image through unchanged (HoughLinesP returns None when "
			            "nothing is found) and count=0.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made)."),
				NPArray.Input("segments",
					tooltip="Nx1x4 or Nx4 endpoint quadruples (x1, y1, x2, "
					        "y2); cv2_HoughLinesP plugs in directly."),
				io.Int.Input("thickness", default=2, min=1, max=64,
					tooltip="Line thickness in pixels."),
				io.String.Input("color", default="(0, 255, 0)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(0, 255, 0)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated. "
					        "Ignored when labels are connected."),
				NPArray.Input("labels", optional=True,
					tooltip="(N,) integer labels: one color per cluster."),
				io.Combo.Input("line_style", options=list(LINE_STYLES),
					default="solid", optional=True,
					tooltip=LINE_STYLE_TOOLTIP),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="Same format as the image input."),
				io.Int.Output(display_name="count",
					tooltip="How many segments were drawn - branch on it "
					        "with if/else for the nothing-found case."),
			],
		)

	@classmethod
	def execute(cls, source, segments, thickness, color, labels=None,
	            line_style="solid") -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		segs = (np.zeros((0, 4), np.float64) if segments is None
		        else np.asarray(segments, np.float64).reshape(-1, 4))
		lbl = None
		if labels is not None and np.asarray(labels).size:
			lbl = np.asarray(labels).ravel().astype(np.int32)
			if len(lbl) != len(segs):
				raise ValueError(
					f"{len(segs)} segments but {len(lbl)} labels - they must "
					"describe the same set."
				)
		bgr = parse_color(color, n_channels)
		for i, (x1, y1, x2, y2) in enumerate(segs):
			c = _index_color(int(lbl[i]), n_channels) if lbl is not None else bgr
			draw_styled_line(canvas, (x1, y1), (x2, y2), c, thickness, line_style)
		return io.NodeOutput(polytype.restore(canvas, tag), len(segs))


class DrawCircles(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawCircles",
			display_name="CV Draw Circles",
			category=CATEGORY,
			description="Debug helper: draws circles given as (x, y, radius) "
			            "triples - exactly what cv2_HoughCircles outputs "
			            "(1xNx3), with a center dot so concentric detections "
			            "stay readable. None/empty circles pass the image "
			            "through unchanged (HoughCircles returns None when "
			            "nothing is found) and count=0.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made)."),
				NPArray.Input("circles",
					tooltip="1xNx3 or Nx3 (x, y, radius) triples; "
					        "cv2_HoughCircles plugs in directly."),
				io.Int.Input("thickness", default=2, min=-1, max=64,
					tooltip="Outline thickness; -1 fills the circles."),
				io.String.Input("color", default="(0, 255, 0)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(0, 255, 0)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated."),
				io.Boolean.Input("draw_centers", default=True,
					tooltip="Also mark each center with a small dot."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="Same format as the image input."),
				io.Int.Output(display_name="count",
					tooltip="How many circles were drawn - branch on it "
					        "with if/else for the nothing-found case."),
			],
		)

	@classmethod
	def execute(cls, source, circles, thickness, color, draw_centers) -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		circ = (np.zeros((0, 3), np.float64) if circles is None
		        else np.asarray(circles, np.float64).reshape(-1, 3))
		bgr = parse_color(color, n_channels)
		for x, y, r in circ:
			center = (int(round(x)), int(round(y)))
			cv2.circle(canvas, center, int(round(r)), bgr, thickness, cv2.LINE_AA)
			if draw_centers:
				cv2.circle(canvas, center, 2, bgr, -1, cv2.LINE_AA)
		return io.NodeOutput(polytype.restore(canvas, tag), len(circ))


class DrawEllipses(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawEllipses",
			display_name="CV Draw Ellipses",
			category=CATEGORY,
			description="Debug helper: draws ellipses given as (cx, cy, "
			            "semi_axis_a, semi_axis_b, angle_deg) rows - the "
			            "layout 'CV Edge Drawing' emits (a circle is "
			            "simply a == b). A 4-column input is accepted as an "
			            "axis-aligned ellipse (angle 0). None/empty passes "
			            "the image through unchanged and count=0.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made)."),
				NPArray.Input("ellipses",
					tooltip="Nx5 (cx, cy, semi_axis_a, semi_axis_b, "
					        "angle_deg) or Nx4 (angle assumed 0); 'OpenCV "
					        "Edge Drawing' plugs in directly."),
				io.Int.Input("thickness", default=2, min=-1, max=64,
					tooltip="Outline thickness; -1 fills the ellipses."),
				io.String.Input("color", default="(0, 255, 0)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(0, 255, 0)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated. "
					        "Ignored when labels are connected."),
				io.Boolean.Input("draw_centers", default=True,
					tooltip="Also mark each center with a small dot."),
				NPArray.Input("labels", optional=True,
					tooltip="(N,) integer labels: one color per cluster."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="Same format as the image input."),
				io.Int.Output(display_name="count",
					tooltip="How many ellipses were drawn - branch on it "
					        "with if/else for the nothing-found case."),
			],
		)

	@classmethod
	def execute(cls, source, ellipses, thickness, color, draw_centers,
	            labels=None) -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		arr = np.zeros((0, 5), np.float64) if ellipses is None else np.asarray(
			ellipses, np.float64)
		arr = arr.reshape(-1, arr.shape[-1]) if arr.size else arr.reshape(0, 5)
		if arr.shape[1] == 4:  # axis-aligned: pad the missing angle column
			arr = np.concatenate([arr, np.zeros((len(arr), 1), np.float64)], 1)
		elif arr.shape[1] != 5:
			raise ValueError(
				f"ellipses must be Nx5 (cx, cy, a, b, angle) or Nx4, got "
				f"{arr.shape} - 'CV Edge Drawing' emits the Nx5 form."
			)
		lbl = None
		if labels is not None and np.asarray(labels).size:
			lbl = np.asarray(labels).ravel().astype(np.int32)
			if len(lbl) != len(arr):
				raise ValueError(
					f"{len(arr)} ellipses but {len(lbl)} labels - they must "
					"describe the same set."
				)
		bgr = parse_color(color, n_channels)
		for i, (cx, cy, a, b, angle) in enumerate(arr):
			c = _index_color(int(lbl[i]), n_channels) if lbl is not None else bgr
			center = (int(round(cx)), int(round(cy)))
			# cv2.ellipse takes SEMI-axes, which is what the ED layout carries
			cv2.ellipse(canvas, center, (int(round(a)), int(round(b))),
			            float(angle), 0.0, 360.0, c, thickness, cv2.LINE_AA)
			if draw_centers:
				cv2.circle(canvas, center, 2, c, -1, cv2.LINE_AA)
		return io.NodeOutput(polytype.restore(canvas, tag), len(arr))


class DrawFlowVectors(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)
	# 0-179 direction hue wheel, precomputed once as a BGR LUT.
	_HUE_LUT = cv2.cvtColor(
		np.dstack([np.arange(180, dtype=np.uint8).reshape(1, 180),
		           np.full((1, 180), 255, np.uint8),
		           np.full((1, 180), 255, np.uint8)]),
		cv2.COLOR_HSV2BGR)[0]
	_COLOR_MODES = ["fixed color", "direction (color wheel)", "magnitude (heatmap)"]
	_MAG_LUTS = {}

	@classmethod
	def _mag_lut(cls, colormap):
		"""BGR LUT for the magnitude ramp, built once per colormap."""
		cm = enums.COLORMAP.resolve(colormap)
		if cm not in cls._MAG_LUTS:
			cls._MAG_LUTS[cm] = cv2.applyColorMap(
				np.arange(256, dtype=np.uint8).reshape(1, 256), cm)[0]
		return cls._MAG_LUTS[cm]

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawFlowVectors",
			display_name="CV Draw Flow Vectors",
			category=CATEGORY,
			description="Debug helper: draws an arrow from each 'from' point to "
			            "its matching 'to' point - the proper way to look at "
			            "SPARSE optical flow / feature tracks (the equal-length, "
			            "same-order point pairs that 'CV Track Features (KLT)' "
			            "and 'CV Match Features' produce). Color the arrows by "
			            "motion direction (the standard flow color wheel) or "
			            "magnitude (a heatmap), or by cluster when labels are "
			            "connected (for motion segmentation). None/empty inputs "
			            "pass the image through unchanged and count=0; tiny "
			            "motions below min_magnitude are skipped.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made)."),
				NPArray.Input("points_from",
					tooltip="Nx1x2 arrow tails - the points in the FIRST frame "
					        "(e.g. 'Track Features (KLT)' points_a)."),
				NPArray.Input("points_to",
					tooltip="Nx1x2 arrow heads - the SAME points in the second "
					        "frame (points_b); must match points_from in count "
					        "and order."),
				io.Int.Input("thickness", default=1, min=1, max=64,
					tooltip="Arrow line width in pixels."),
				io.String.Input("color", default="(0, 255, 0)",
					tooltip="Color as a single value (broadcast to all "
					        "channels) or BGR tuple, e.g. '255' or "
					        "'(0, 255, 0)'. Shorter tuples are "
					        "zero-padded; longer tuples are truncated. "
					        "Used only in 'fixed color' mode (and "
					        "ignored when labels are connected)."),
				io.Combo.Input("color_by", options=list(cls._COLOR_MODES),
					default="direction (color wheel)",
					tooltip="How to color each arrow: a single fixed color; by "
					        "motion DIRECTION (hue = angle, the standard flow "
					        "wheel - hue alone, so it is a picture to look at, "
					        "not a key to read); or by MAGNITUDE, through "
					        "'magnitude_colormap' (dark = slow -> bright = fast, "
					        "readable as brightness alone). Connecting labels "
					        "overrides this with per-cluster colors."),
				io.Float.Input("tip_length", default=0.3, min=0.0, max=2.0,
					optional=True, advanced=True,
					tooltip="Arrowhead length as a fraction of the shaft "
					        "(cv2.arrowedLine tipLength)."),
				io.Float.Input("min_magnitude", default=0.0, min=0.0, max=1e6,
					optional=True, advanced=True,
					tooltip="Skip arrows shorter than this many pixels - hides "
					        "sub-pixel tracking jitter. 0 draws everything."),
				io.Float.Input("max_magnitude", default=0.0, min=0.0, max=1e6,
					optional=True, advanced=True,
					tooltip="Magnitude mapped to the hot end of the heatmap; "
					        "larger clip. 0 = auto (this frame's longest "
					        "vector) - set it to compare frames on one scale."),
				NPArray.Input("labels", optional=True, advanced=True,
					tooltip="(N,) integer labels: one color per cluster "
					        "(e.g. motion segmentation), overriding color_by."),
				io.Combo.Input("magnitude_colormap", options=enums.COLORMAP.options,
					default="COLORMAP_VIRIDIS", optional=True, advanced=True,
					tooltip="Ramp for the 'magnitude (heatmap)' mode. The "
					        "default is perceptually uniform and rises "
					        "monotonically in brightness, so slow-vs-fast "
					        "survives colour blindness and greyscale; "
					        "COLORMAP_JET is the familiar blue->red but does "
					        "neither."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="Same format as the image input."),
				io.Int.Output(display_name="count",
					tooltip="How many arrows were drawn (after the "
					        "min_magnitude filter) - branch on it with if/else "
					        "for the nothing-moved case."),
			],
		)

	@classmethod
	def execute(cls, source, points_from, points_to, thickness, color, color_by,
	            tip_length=0.3, min_magnitude=0.0, max_magnitude=0.0,
	            labels=None, magnitude_colormap="COLORMAP_VIRIDIS") -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		a = _as_points(points_from)
		b = _as_points(points_to)
		if len(a) != len(b):
			raise ValueError(
				f"points_from has {len(a)} points but points_to has {len(b)} - "
				"they must be the same matched set, in the same order."
			)
		if len(a) == 0:
			return io.NodeOutput(polytype.restore(canvas, tag), 0)
		lbl = None
		if labels is not None and np.asarray(labels).size:
			lbl = np.asarray(labels).ravel().astype(np.int32)
			if len(lbl) != len(a):
				raise ValueError(
					f"{len(a)} vectors but {len(lbl)} labels - they must "
					"describe the same set."
				)
		bgr = parse_color(color, n_channels)

		delta = b - a
		mag = np.hypot(delta[:, 0], delta[:, 1])
		keep = mag >= float(min_magnitude)
		# Direction/magnitude colors are precomputed for the kept vectors.
		mode = color_by if lbl is None else "fixed color"
		dir_colors = mag_colors = None
		if lbl is None and mode.startswith("direction"):
			hue = (np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 360.0) / 2.0
			dir_colors = cls._HUE_LUT[np.clip(hue, 0, 179).astype(np.int32)]
		elif lbl is None and mode.startswith("magnitude"):
			scale = max_magnitude if max_magnitude > 0 else float(mag[keep].max() if keep.any() else 0.0)
			if scale <= 0:
				scale = 1.0
			idx = np.clip(mag / scale * 255.0, 0, 255).astype(np.int32)
			mag_colors = cls._mag_lut(magnitude_colormap)[idx]

		count = 0
		for i in range(len(a)):
			if not keep[i]:
				continue
			if lbl is not None:
				c = _index_color(int(lbl[i]), n_channels)
			elif dir_colors is not None:
				c = tuple(int(v) for v in dir_colors[i])
			elif mag_colors is not None:
				c = tuple(int(v) for v in mag_colors[i])
			else:
				c = bgr
			cv2.arrowedLine(canvas, (int(round(a[i, 0])), int(round(a[i, 1]))),
			                (int(round(b[i, 0])), int(round(b[i, 1]))), c,
			                thickness, cv2.LINE_AA, tipLength=float(tip_length))
			count += 1
		return io.NodeOutput(polytype.restore(canvas, tag), count)


class DrawPath(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)
	_PROGRESS = ("solid", "colormap along path", "thicken along path")
	_LABEL_ENDS = ("none", "start / end", "first / last index")

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawPath",
			display_name="CV Draw Path (ordered)",
			category=CATEGORY,
			description="Draws a point sequence as a path whose DIRECTION and "
			            "ENDS are readable: a colour ramp (or a thickness ramp) "
			            "from the first point to the last, a distinct marker "
			            "shape at each end, and optional heading arrows along the "
			            "way. A plain polyline ('CV Draw Polygon') shows the "
			            "shape of a camera trajectory, a tracked point or a "
			            "contour but not which end it started from, which is the "
			            "one thing a reader needs. Use it for any ORDERED 2D "
			            "sequence; the three cues are redundant on purpose, so "
			            "two paths on one canvas stay apart in greyscale and for "
			            "a colour-blind viewer (give the second a different "
			            "line_style and end markers, not just another colour). "
			            "For a metric path, map it into pixels first with 'OpenCV "
			            "Project To Plane' -> 'CV Fit Points To Box' and put "
			            "'CV Draw Plot Frame' underneath for axes. "
			            "None/empty points pass the image through unchanged.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Image or mask to draw on (a copy is made)."),
				NPArray.Input("points",
					tooltip="Nx2 or Nx1x2 points IN ORDER (first = start). "
					        "Non-finite points are skipped, so a gated sequence "
					        "keeps its indexing. None/empty -> passthrough."),
				io.String.Input("color", default="(0, 255, 0)",
					tooltip="Colour as a single value (broadcast to all channels) "
					        "or a BGR tuple, e.g. '255' or '(0, 255, 0)'. Used for "
					        "the markers and labels always, and for the line itself "
					        "unless progress = 'colormap along path'."),
				io.Combo.Input("progress", options=list(cls._PROGRESS),
					default=cls._PROGRESS[1],
					tooltip="How the direction of travel is shown along the line. "
					        "'colormap along path' walks a sequential ramp from "
					        "start to end (the clearest, and it survives greyscale "
					        "because the ramp is monotone in luminance). 'thicken "
					        "along path' grows the stroke instead - use it when the "
					        "colour is already carrying something else. 'solid' "
					        "leaves the line flat and relies on the end markers."),
				io.Combo.Input("progress_colormap", options=enums.COLORMAP.options,
					default=enums.COLORMAP.default,
					tooltip="Ramp for 'colormap along path'. Keep a SEQUENTIAL map "
					        "(VIRIDIS, MAGMA, PLASMA): a rainbow like JET is not "
					        "monotone in brightness, so it invents features that are "
					        "not in the data and prints as mush."),
				io.Int.Input("thickness", default=2, min=1, max=64,
					tooltip="Line thickness in pixels (the maximum, for 'thicken "
					        "along path')."),
				io.Combo.Input("line_style", options=list(LINE_STYLES),
					default="solid",
					tooltip=LINE_STYLE_TOOLTIP + " Note that the pattern restarts "
					        "at every segment, so on a DENSELY sampled path (one "
					        "sample per video frame, a pixel apart) a dashed line "
					        "renders solid - there lean on 'progress' and the end "
					        "markers instead."),
				io.Combo.Input("start_marker", options=list(MARKER_SHAPES),
					default="filled square",
					tooltip="Marker at the FIRST point. " + MARKER_SHAPE_TOOLTIP),
				io.Combo.Input("end_marker", options=list(MARKER_SHAPES),
					default="triangle up",
					tooltip="Marker at the LAST point - make it a different shape "
					        "from start_marker, that is what tells the two apart."),
				io.Int.Input("marker_radius", default=6, min=1, max=128,
					tooltip="Radius of the two end markers, in pixels. 0-length "
					        "paths still get their single point marked."),
				io.Int.Input("arrow_every", default=0, min=0, max=100000,
					optional=True,
					tooltip="Draw a heading arrow every N points (0 = none). The "
					        "arrow spans a fixed arc length back along the path, so "
					        "it stays readable where samples are dense."),
				io.Combo.Input("label_ends", options=list(cls._LABEL_ENDS),
					default=cls._LABEL_ENDS[0], optional=True,
					tooltip="Text at the two ends, drawn with a contrast halo so it "
					        "survives any background. 'first / last index' stamps "
					        "the actual indices, which is what you want when the "
					        "path is a frame sequence."),
				NPArray.Input("gap_mask", optional=True, advanced=True,
					tooltip="(N,) 0/1 per point. A 0 means that point was HELD "
					        "rather than measured (e.g. 'found_mask' from 'OpenCV "
					        "Visual Odometry (Sequence)'). Those points get a small "
					        "x and their segment is drawn dotted, so a stretch the "
					        "estimator gave up on cannot pass for real data. The "
					        "marker is what carries it: a held point sits exactly "
					        "on its predecessor, so the segment has no length to "
					        "draw a style with."),
			],
			outputs=[
				polytype.match_output(cls._TEMPLATE, "image",
					tooltip="Same format as the image input."),
				io.Int.Output(display_name="count",
					tooltip="How many points were drawn."),
			],
		)

	@classmethod
	def execute(cls, source, points, color, progress, progress_colormap,
	            thickness, line_style, start_marker, end_marker, marker_radius,
	            arrow_every=0, label_ends="none", gap_mask=None) -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		pts = _as_points(points)
		if len(pts) == 0:
			return io.NodeOutput(polytype.restore(canvas, tag), 0)
		bgr = parse_color(color, n_channels)
		ramp = (_ramp_colors(len(pts), progress_colormap, n_channels)
		        if str(progress) == cls._PROGRESS[1] else None)
		held = None
		if gap_mask is not None and np.asarray(gap_mask).size:
			held = np.asarray(gap_mask).ravel().astype(bool)
			if len(held) != len(pts):
				raise ValueError(
					f"{len(pts)} points but {len(held)} gap_mask entries - they "
					"must describe the same sequence."
				)

		for i in range(1, len(pts)):
			a, b = pts[i - 1], pts[i]
			if not (_drawable(*a) and _drawable(*b)):
				continue   # a blanked-out sample breaks the line, not the node
			c = ramp[i] if ramp is not None else bgr
			t = thickness
			if str(progress) == cls._PROGRESS[2] and len(pts) > 1:
				t = max(1, int(round(1 + (thickness - 1) * i / (len(pts) - 1))))
			style = "dotted" if (held is not None and not held[i]) else line_style
			draw_styled_line(canvas, a, b, c, int(t), style)

		if held is not None:
			# a held point sits ON its predecessor, so there is no segment to
			# style - the marker is the only thing that can show the gap
			for i in np.flatnonzero(~held):
				if _drawable(*pts[i]):
					draw_marker(canvas, pts[i][0], pts[i][1],
					            max(2, marker_radius // 2),
					            ramp[i] if ramp is not None else bgr,
					            "x (tilted cross)")

		if arrow_every:
			head = max(8.0, 4.0 * thickness)
			for i in range(int(arrow_every), len(pts), int(arrow_every)):
				if not _drawable(*pts[i]):
					continue
				tail = _back_along(pts, i, head)
				if float(np.hypot(*(pts[i] - tail))) < 1.0:
					continue     # standing still: an arrow here would point nowhere
				c = ramp[i] if ramp is not None else bgr
				cv2.arrowedLine(canvas,
				                (int(round(tail[0])), int(round(tail[1]))),
				                (int(round(pts[i][0])), int(round(pts[i][1]))),
				                c, int(thickness), cv2.LINE_AA, tipLength=0.5)

		finite = [i for i in range(len(pts)) if _drawable(*pts[i])]
		if finite:
			first, last = finite[0], finite[-1]
			draw_marker(canvas, pts[first][0], pts[first][1], marker_radius,
			            bgr, start_marker)
			if last != first:
				draw_marker(canvas, pts[last][0], pts[last][1], marker_radius,
				            bgr, end_marker)
			if str(label_ends) != cls._LABEL_ENDS[0]:
				named = str(label_ends) == cls._LABEL_ENDS[1]
				pairs = [(first, "start" if named else str(first))]
				if last != first:
					pairs.append((last, "end" if named else str(last)))
				halo = halo_color(bgr)
				for idx, text in pairs:
					org = (int(round(pts[idx][0])) + marker_radius + 4,
					       int(round(pts[idx][1])) - marker_radius - 4)
					draw_text_outlined(canvas, text, org, 0.5, halo, bgr, 1)
		return io.NodeOutput(polytype.restore(canvas, tag), len(pts))


class DrawPlotFrame(io.ComfyNode):
	_TEMPLATE = polytype.match_template(types=polytype.IMAGEISH)
	_Y_AXIS = ("y down (pixel rows)", "y up (world)")
	_SCALE_BAR = ("none", "bottom-left", "bottom-right")

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DrawPlotFrame",
			display_name="CV Draw Plot Frame",
			category=CATEGORY,
			description="Puts a grid, ticks labelled in WORLD units and a scale "
			            "bar under a 2D plot - the axes that turn a drawn path or "
			            "point cloud into a readable figure. Without them a "
			            "trajectory is a squiggle with no size: you cannot tell a "
			            "3 metre wobble from a 300 metre one. Feed it the SAME "
			            "world points and margin you gave 'CV Fit Points To "
			            "Box' (via 'extent_from', or as explicit min/max) and it "
			            "reproduces that node's mapping exactly, so a tick at 50 m "
			            "lands where the point at 50 m lands. Draw this FIRST, "
			            "then the data on top. Nothing here reads the data itself, "
			            "so it also works as an empty graticule.",
			inputs=[
				polytype.match_input("source", cls._TEMPLATE,
					tooltip="Canvas to draw on (a copy is made) - e.g. 'OpenCV "
					        "Constant Like'. Its size sets the plot box."),
				NPArray.Input("extent_from", optional=True,
					tooltip="World points, Nx2, whose bounding box defines the "
					        "plotted range - pass the same array you gave 'Fit "
					        "Points To Box' (both tracks concatenated, if you are "
					        "overlaying two). When absent the explicit min/max "
					        "below are used."),
				io.Float.Input("x_min", default=0.0, min=-1e9, max=1e9,
					tooltip="World range, used only when 'extent_from' is empty."),
				io.Float.Input("x_max", default=100.0, min=-1e9, max=1e9),
				io.Float.Input("y_min", default=0.0, min=-1e9, max=1e9),
				io.Float.Input("y_max", default=100.0, min=-1e9, max=1e9),
				io.Int.Input("margin", default=20, min=0, max=4096,
					tooltip="Must MATCH the margin given to 'CV Fit Points To "
					        "Box', or the grid and the data disagree."),
				io.Int.Input("grid_x", default=6, min=0, max=64,
					tooltip="Roughly how many vertical grid lines to aim for; the "
					        "actual step is rounded to a 1 / 2 / 5 x 10^k value so "
					        "the labels stay round numbers. 0 = no grid."),
				io.Int.Input("grid_y", default=6, min=0, max=64,
					tooltip="Same for horizontal grid lines."),
				io.Combo.Input("y_axis", options=list(cls._Y_AXIS),
					default=cls._Y_AXIS[0],
					tooltip="Which way the second coordinate grows. Points that "
					        "came through 'Fit Points To Box' are already in PIXEL "
					        "rows (y grows downward) - that is the default. Choose "
					        "'y up (world)' only if you flipped them yourself."),
				io.String.Input("units", default="m",
					tooltip="Unit suffix on the tick labels and the scale bar "
					        "('m', 'px', 'mm'...). Empty for bare numbers."),
				io.String.Input("x_label", default="x",
					tooltip="Axis name drawn at the bottom right. Empty = none."),
				io.String.Input("y_label", default="z",
					tooltip="Axis name drawn at the top left. Empty = none. For a "
					        "top-down ('XZ') view of a trajectory this is z."),
				io.Combo.Input("scale_bar", options=list(cls._SCALE_BAR),
					default=cls._SCALE_BAR[1],
					tooltip="A labelled bar of known world length. It is the one "
					        "annotation that still says how big the picture is "
					        "after someone crops or rescales the image."),
				io.String.Input("color", default="(200, 200, 200)",
					tooltip="Colour of the ticks, labels, border and scale bar."),
				io.String.Input("grid_color", default="(70, 70, 70)",
					tooltip="Colour of the grid lines - keep it well below the "
					        "data's contrast so the grid stays behind it."),
			],
			outputs=[polytype.match_output(cls._TEMPLATE, "image",
				tooltip="Same format as the image input.")],
		)

	@classmethod
	def execute(cls, source, extent_from, x_min, x_max, y_min, y_max, margin,
	            grid_x, grid_y, y_axis, units, x_label, y_label, scale_bar,
	            color, grid_color) -> io.NodeOutput:
		source, tag = polytype.resolve(source)
		canvas, n_channels = _make_draw_canvas(source, tag)
		h, w = canvas.shape[:2]
		fg = parse_color(color, n_channels)
		grid = parse_color(grid_color, n_channels)
		pts = _as_points(extent_from)
		if len(pts) == 0:
			pts = np.array([[x_min, y_min], [x_max, y_max]], np.float64)
		scale, off_x, off_y, sign = _fit_box_mapping(
			pts, w, h, margin, str(y_axis) == cls._Y_AXIS[1])
		lo, hi = pts.min(axis=0), pts.max(axis=0)
		flip = sign            # the mapping decides it, so ticks cannot disagree
		suffix = f" {units}" if units else ""

		cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), fg, 1)
		font, fs, ft = imageutils.SCALE_FONT, imageutils.SCALE_FSCALE, imageutils.SCALE_FTHICK

		# the world range actually visible in the box, per axis
		view_lo = np.array([(margin - off_x) / scale,
		                    flip * (margin - off_y) / scale])
		view_hi = np.array([(w - margin - off_x) / scale,
		                    flip * (h - margin - off_y) / scale])
		for axis, count in ((0, grid_x), (1, grid_y)):
			if count <= 0:
				continue
			a, b = sorted((float(view_lo[axis]), float(view_hi[axis])))
			step = _nice_step(b - a, count)
			first = math.ceil(a / step) * step
			for k in range(int((b - first) / step) + 1):
				value = first + k * step
				if axis == 0:
					px = int(round(value * scale + off_x))
					if not 0 <= px < w:
						continue
					cv2.line(canvas, (px, 0), (px, h - 1), grid, 1)
					cv2.putText(canvas, imageutils._fmt_scale(value) + suffix,
					            (px + 3, h - 6), font, fs, fg, ft, cv2.LINE_AA)
				else:
					py = int(round(flip * value * scale + off_y))
					if not 0 <= py < h:
						continue
					cv2.line(canvas, (0, py), (w - 1, py), grid, 1)
					cv2.putText(canvas, imageutils._fmt_scale(value) + suffix,
					            (4, py - 4), font, fs, fg, ft, cv2.LINE_AA)

		if x_label:
			(tw, _), _ = cv2.getTextSize(x_label, font, fs, ft)
			cv2.putText(canvas, x_label, (w - tw - 6, h - 20), font, fs, fg,
			            ft, cv2.LINE_AA)
		if y_label:
			cv2.putText(canvas, y_label, (6, 16), font, fs, fg, ft, cv2.LINE_AA)

		if str(scale_bar) != cls._SCALE_BAR[0] and scale > 1e-12:
			length = _nice_step(float(np.max(hi - lo)), 5)
			px_len = int(round(length * scale))
			if 8 <= px_len <= w - 2 * margin:
				y = h - 30
				x0 = margin + 4 if str(scale_bar) == cls._SCALE_BAR[1] \
					else w - margin - 4 - px_len
				cv2.line(canvas, (x0, y), (x0 + px_len, y), fg, 2)
				for end in (x0, x0 + px_len):
					cv2.line(canvas, (end, y - 4), (end, y + 4), fg, 2)
				text = imageutils._fmt_scale(length) + suffix
				(tw, _), _ = cv2.getTextSize(text, font, fs, ft)
				cv2.putText(canvas, text, (x0 + (px_len - tw) // 2, y - 8),
				            font, fs, fg, ft, cv2.LINE_AA)
		return io.NodeOutput(polytype.restore(canvas, tag))


class ProjectToPlane(io.ComfyNode):
	_PLANES = {"XZ (top-down)": (0, 2), "XY (front)": (0, 1), "YZ (side)": (1, 2)}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ProjectToPlane",
			display_name="CV Project To Plane",
			category=CATEGORY,
			description="Drops one axis of 3D points to get a 2D ORTHOGRAPHIC view - "
			            "a top-down (XZ), front (XY) or side (YZ) projection of a 3D "
			            "point cloud or camera trajectory, ready for 'CV Draw "
			            "Points' / 'CV Draw Polygon'. (For a perspective render "
			            "use cv2.projectPoints with a camera matrix instead.) "
			            "None/empty input passes through as an empty array.",
			inputs=[
				NPArray.Input("points",
					tooltip="3D points, Nx3 or Nx1x3 (e.g. a triangulated cloud or a "
					        "'Compose Pose' trajectory). None/empty -> empty output."),
				io.Combo.Input("plane", options=list(cls._PLANES), default="XZ (top-down)",
					tooltip="Which two axes to keep: XZ is the bird's-eye view (drops "
					        "the height Y); XY is the front view; YZ the side view."),
			],
			outputs=[NPArray.Output(display_name="points",
				tooltip="2D points, Nx1x2 float32.")],
		)

	@classmethod
	def execute(cls, points, plane) -> io.NodeOutput:
		if points is None or np.asarray(points).size == 0:
			return io.NodeOutput(_EMPTY.copy())
		pts = np.asarray(points, np.float64).reshape(-1, 3)
		i, j = cls._PLANES[plane]
		return io.NodeOutput(np.float32(pts[:, [i, j]]).reshape(-1, 1, 2))


class FitPointsToBox(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FitPointsToBox",
			display_name="CV Fit Points To Box",
			category=CATEGORY,
			description="Scales and centres a set of 2D points so they fill a "
			            "width x height pixel box (with a margin), preserving the "
			            "aspect ratio (one uniform scale for both axes) - turns "
			            "metric coordinates like a triangulated cloud projected "
			            "top-down ('CV Project To Plane') or a 'Compose Pose' "
			            "trajectory into drawable pixels for 'CV Draw Points' / "
			            "'CV Draw Polygon'. Failure-tolerant: None/empty passes "
			            "through as an empty array and a single point (or a "
			            "zero-extent set) lands at the box centre, so it never "
			            "divides by zero or raises.",
			inputs=[
				NPArray.Input("points",
					tooltip="2D points, Nx2 or Nx1x2 (None/empty -> empty output)."),
				io.Int.Input("width", default=300, min=1, max=16384,
					tooltip="Target box width in pixels (match your canvas)."),
				io.Int.Input("height", default=300, min=1, max=16384,
					tooltip="Target box height in pixels."),
				io.Int.Input("margin", default=20, min=0, max=4096,
					tooltip="Padding kept clear inside the box, in pixels."),
				NPArray.Input("extent_from", optional=True,
					tooltip="Take the range to fit from THESE points instead of "
					        "'points'. Two tracks fitted separately each fill the "
					        "box, so overlaying them silently rescales one against "
					        "the other - pass both tracks concatenated here (and to "
					        "'CV Draw Plot Frame') and every set shares one "
					        "mapping. Empty = use 'points', as before."),
				io.Combo.Input("y_axis", options=list(Y_AXIS_MODES),
					default=Y_AXIS_MODES[0], optional=True,
					tooltip=Y_AXIS_TOOLTIP + " Pass the SAME value to 'OpenCV "
					        "Draw Plot Frame' so the grid and the data agree."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="Points scaled into the box, Nx1x2 float32."),
				io.Float.Output(display_name="scale",
					tooltip="Pixels per world unit, i.e. pixel = world * scale + "
					        "offset. Feed it to a scale bar or a measurement."),
				io.Float.Output(display_name="offset_x",
					tooltip="Pixel column that world x = 0 maps to."),
				io.Float.Output(display_name="offset_y",
					tooltip="Pixel row that world y = 0 maps to."),
			],
		)

	@classmethod
	def execute(cls, points, width, height, margin, extent_from=None,
	            y_axis=Y_AXIS_MODES[0]) -> io.NodeOutput:
		pts = _as_points(points)
		extent = _as_points(extent_from)
		if len(extent) == 0:
			extent = pts
		scale, off_x, off_y, sign = _fit_box_mapping(
			extent, width, height, margin, str(y_axis) == Y_AXIS_MODES[1])
		if len(pts) == 0:
			return io.NodeOutput(_EMPTY.copy(), scale, off_x, off_y)
		mapped = pts * np.array([scale, sign * scale]) + np.array([off_x, off_y])
		return io.NodeOutput(_to_points(mapped), scale, off_x, off_y)


class ScalePoints(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ScalePoints",
			display_name="CV Scale Points",
			category=CATEGORY,
			description="Multiplies point coordinates by a scale factor "
			            "via numpy (pts * scale). "
			            "Handles None or empty arrays (returns empty). "
			            "This is a numpy helper, not a cv2 wrapper.",
			inputs=[
				NPArray.Input("points",
				    tooltip="Nx1x2 or Nx2 point array (or None/empty)."),
				io.Float.Input("scale", default=1.0, min=-1e6, max=1e6,
				    tooltip="Scale factor applied to every coordinate. "
				            ">1 enlarges, <1 shrinks."),
			],
			outputs=[
				NPArray.Output(display_name="points"),
			],
		)

	@classmethod
	def execute(cls, points, scale) -> io.NodeOutput:
		pts = _as_points(points)
		return io.NodeOutput(_to_points(pts * scale))


class DelaunayTriangulation(io.ComfyNode):
	"""cv2.Subdiv2D - Delaunay triangulation + Voronoi diagram of a point set."""

	@classmethod
	def define_schema(cls):
		from .contours import Contours  # local: contours.py imports features, not points

		return io.Schema(
			node_id="CV_DelaunayTriangulation",
			display_name="CV Delaunay / Voronoi",
			category=CATEGORY,
			description="Connects a point set into a triangle mesh (Delaunay "
			            "triangulation) and its dual Voronoi diagram, via "
			            "cv2.Subdiv2D - a stateful class the raw wrappers cannot "
			            "expose. Delaunay is THE way to turn scattered points into a "
			            "mesh: it maximises the smallest angle, so the triangles are "
			            "as well-shaped as the points allow. Uses: face/image morphing "
			            "and piecewise-affine warping (triangulate landmarks, warp "
			            "each triangle), surface reconstruction from a sparse point "
			            "cloud, and nearest-neighbour region maps (the Voronoi cell of "
			            "a point is everything closer to it than to any other). Feed "
			            "any Nx2 point set - detected corners, blob centres, k-means "
			            "centroids, annotated points. Draw 'triangles'/'facets' with "
			            "'CV Draw Contours' or 'edges' with 'CV Draw "
			            "Segments'. Fewer than 3 points is a valid result (empty "
			            "outputs, count = 0).",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx2 (or Nx1x2) point set to triangulate. Duplicate "
					        "points collapse to one vertex."),
				io.Float.Input("margin", default=10.0, min=0.0, max=10000.0, step=1.0,
					tooltip="Padding around the points' bounding box when no "
					        "'bounds' image is connected. Subdiv2D needs an enclosing "
					        "rectangle and rejects points on its edge, so leave at "
					        "least a pixel or two."),
				polytype.poly_input("bounds", types=polytype.IMAGE_ONLY, optional=True,
					tooltip="Optional image whose SIZE defines the enclosing "
					        "rectangle - connect the image the points came from so "
					        "the Voronoi cells are clipped to it. Without it the "
					        "rectangle is the points' bounding box plus 'margin'."),
			],
			outputs=[
				Contours.Output(display_name="triangles",
					tooltip="One 3-point contour per Delaunay triangle. Triangles "
					        "touching Subdiv2D's virtual outer vertices are dropped, "
					        "so every one lies inside the rectangle."),
				NPArray.Output(display_name="edges",
					tooltip="Mx4 float32 (x1, y1, x2, y2) mesh edges - feed 'OpenCV "
					        "Draw Segments' for a clean wireframe (each edge drawn "
					        "once instead of three times per triangle)."),
				Contours.Output(display_name="facets",
					tooltip="One contour per VORONOI cell, in the same order as "
					        "'centers' - the region of the plane closest to that "
					        "point. Draw with 'CV Draw Contours'."),
				NPArray.Output(display_name="centers",
					tooltip="Kx2 float32 the facets belong to: the input points "
					        "after de-duplication and clamping into the rectangle."),
				io.Int.Output(display_name="count",
					tooltip="Number of triangles; 0 for fewer than 3 distinct points "
					        "- a valid result, not an error."),
			],
		)

	@classmethod
	def execute(cls, points, margin, bounds=None) -> io.NodeOutput:
		pts = _as_points(points)
		empty = ([], np.zeros((0, 4), np.float32), [], np.zeros((0, 2), np.float32), 0)
		if len(pts) < 3:
			return io.NodeOutput(*empty)

		if bounds is not None:
			h, w = polytype.spatial_size(bounds)
			x0, y0, x1, y1 = 0.0, 0.0, float(w), float(h)
		else:
			m = float(margin)
			x0, y0 = pts[:, 0].min() - m, pts[:, 1].min() - m
			x1, y1 = pts[:, 0].max() + m + 1.0, pts[:, 1].max() + m + 1.0
		rect = (int(np.floor(x0)), int(np.floor(y0)),
		        int(np.ceil(x1 - x0)) + 1, int(np.ceil(y1 - y0)) + 1)

		subdiv = cv2.Subdiv2D(rect)
		# Subdiv2D throws for a point on or outside the rectangle edge; a stray
		# point is data (a detector found something at the border), so clamp it
		# in rather than failing the workflow
		lo_x, lo_y = rect[0] + 1.0, rect[1] + 1.0
		hi_x, hi_y = rect[0] + rect[2] - 2.0, rect[1] + rect[3] - 2.0
		kept = []
		seen = set()
		for x, y in pts:
			p = (float(np.clip(x, lo_x, hi_x)), float(np.clip(y, lo_y, hi_y)))
			key = (round(p[0], 4), round(p[1], 4))
			if key in seen:
				continue
			seen.add(key)
			subdiv.insert(p)
			kept.append(p)
		if len(kept) < 3:
			return io.NodeOutput(*empty)

		def inside(x, y):
			return (rect[0] <= x <= rect[0] + rect[2]
			        and rect[1] <= y <= rect[1] + rect[3])

		triangles = []
		for t in np.asarray(subdiv.getTriangleList(), np.float64).reshape(-1, 6):
			tri = t.reshape(3, 2)
			# getTriangleList also returns triangles built on the 3 virtual
			# vertices Subdiv2D adds far outside the rect - drop those
			if all(inside(px, py) for px, py in tri):
				triangles.append(np.round(tri).astype(np.int32).reshape(-1, 1, 2))

		edges = []
		for e in np.asarray(subdiv.getEdgeList(), np.float64).reshape(-1, 4):
			if inside(e[0], e[1]) and inside(e[2], e[3]):
				edges.append(e)
		edge_arr = (np.asarray(edges, np.float32) if edges
		            else np.zeros((0, 4), np.float32))

		# Voronoi cells on the hull are unbounded, and Subdiv2D returns them
		# running far outside the rectangle - clip every (convex) facet against
		# it so they are drawable and their areas mean something
		facets, centers = subdiv.getVoronoiFacetList([])
		clip = np.asarray([[rect[0], rect[1]],
		                   [rect[0] + rect[2], rect[1]],
		                   [rect[0] + rect[2], rect[1] + rect[3]],
		                   [rect[0], rect[1] + rect[3]]], np.float32)
		facet_contours, facet_centers = [], []
		for f, c in zip(facets, np.asarray(centers, np.float32).reshape(-1, 2)):
			poly = np.asarray(f, np.float32).reshape(-1, 2)
			area, clipped = cv2.intersectConvexConvex(poly, clip)
			if area <= 0.0 or clipped is None or len(clipped) < 3:
				continue  # cell lies entirely outside the rectangle
			facet_contours.append(
				np.round(np.asarray(clipped, np.float64)).astype(np.int32).reshape(-1, 1, 2))
			facet_centers.append(c)
		center_arr = (np.asarray(facet_centers, np.float32).reshape(-1, 2) if facet_centers
		              else np.zeros((0, 2), np.float32))
		return io.NodeOutput(triangles, edge_arr, facet_contours, center_arr,
		                     len(triangles))


class SampleArrayAtPoints(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SampleArrayAtPoints",
			display_name="CV Sample Array At Points",
			category=CATEGORY,
			description="Reads the values of an HxW (xC) array at a set of 2D "
			            "point coordinates - the generic 'look up this map where "
			            "the features are' step. Feed the XYZ map of "
			            "cv2.reprojectImageTo3D and the matched points of 'OpenCV "
			            "Match Features' to get the 3D positions of the matches "
			            "(then cv2.estimateAffine3D registers two clouds); it "
			            "equally samples a depth map, an optical-flow field, a "
			            "confidence map or a colour image at keypoints. "
			            "Coordinates are in pixels (x, y), sub-pixel allowed. "
			            "Points outside the array are clamped to the edge and "
			            "flagged in 'valid' (0), so an out-of-frame keypoint "
			            "never raises - filter with 'CV Filter Points 3D By "
			            "Mask' / 'CV Filter Points By Mask'.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="HxW or HxWxC array to sample (XYZ map, depth, flow, "
					        "confidence, colour image...). Any dtype."),
				NPArray.Input("points",
					tooltip="Sample positions in pixels, Nx2 or Nx1x2 (x, y) - "
					        "e.g. 'CV Match Features' points_a/points_b. "
					        "None/empty gives an empty result, not an error."),
				io.Combo.Input("interpolation",
					options=["nearest", "bilinear"], default="nearest",
					tooltip="nearest: exact pixel value, never invents data - the "
					        "right choice for XYZ / label / disparity maps whose "
					        "neighbours may be invalid. bilinear: sub-pixel "
					        "weighted average, smoother but mixes in neighbours "
					        "(good for smooth fields, bad across depth edges)."),
				io.Combo.Input("validity",
					options=["in bounds and finite", "in bounds only"],
					default="in bounds and finite", optional=True, advanced=True,
					tooltip="What 'valid' means. 'in bounds and finite' also "
					        "clears the flag where the sampled value is NaN or Inf "
					        "(cv2.reprojectImageTo3D marks missing disparity that "
					        "way); 'in bounds only' flags every sample that landed "
					        "inside the array, whatever its value."),
				NPArray.Input("valid_in", optional=True,
					tooltip="Optional mask from an earlier sampler (one entry per "
					        "point): its zeros are carried into this node's 'valid' "
					        "output. Chain samplers through it to AND several "
					        "validity tests together - it stays correct for an "
					        "empty point set, which cv2.bitwise_and does not "
					        "(cv2 returns None for empty input)."),
			],
			outputs=[
				NPArray.Output(display_name="values",
					tooltip="NxC array of sampled values (C = 1 for a 2D input); "
					        "same dtype as the input for 'nearest', float for "
					        "'bilinear'."),
				NPArray.Output(display_name="valid",
					tooltip="Nx1 uint8 mask, 255 where the point was inside the "
					        "array (and finite, if require_finite)."),
				io.Int.Output(display_name="count",
					tooltip="Number of valid samples."),
			],
		)

	@classmethod
	def execute(cls, nparray, points, interpolation,
	            validity="in bounds and finite", valid_in=None) -> io.NodeOutput:
		src = np.asarray(nparray)
		if src.ndim == 2:
			src = src[:, :, None]
		h, w = src.shape[:2]
		chan = src.shape[2]
		pts = _as_points(points)
		if len(pts) == 0 or h == 0 or w == 0:
			empty = np.zeros((0, chan), src.dtype if str(interpolation) == "nearest"
			                 else np.float32)
			return io.NodeOutput(empty, np.zeros((0, 1), np.uint8), 0)

		x, y = pts[:, 0], pts[:, 1]
		inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)

		if str(interpolation) == "bilinear":
			# clamp first so the four taps always exist; 'inside' already records
			# which samples were extrapolated
			xc = np.clip(x, 0, w - 1)
			yc = np.clip(y, 0, h - 1)
			x0 = np.floor(xc).astype(np.intp)
			y0 = np.floor(yc).astype(np.intp)
			x1 = np.minimum(x0 + 1, w - 1)
			y1 = np.minimum(y0 + 1, h - 1)
			fx = (xc - x0)[:, None]
			fy = (yc - y0)[:, None]
			buf = src.astype(np.float64, copy=False)
			top = buf[y0, x0] * (1.0 - fx) + buf[y0, x1] * fx
			bottom = buf[y1, x0] * (1.0 - fx) + buf[y1, x1] * fx
			values = (top * (1.0 - fy) + bottom * fy)
			values = values.astype(np.float64 if src.dtype == np.float64 else np.float32)
		else:
			xi = np.clip(np.round(x), 0, w - 1).astype(np.intp)
			yi = np.clip(np.round(y), 0, h - 1).astype(np.intp)
			values = np.ascontiguousarray(src[yi, xi])

		valid = inside
		if str(validity) == "in bounds and finite" and np.issubdtype(values.dtype, np.floating):
			valid = valid & np.isfinite(values).all(axis=1)
		if valid_in is not None and np.asarray(valid_in).size:
			prev = np.asarray(valid_in).ravel().astype(bool)
			if len(prev) != len(valid):
				raise ValueError(
					f"valid_in has {len(prev)} entries but there are {len(valid)} "
					"points - it must come from a sampler fed the same point set."
				)
			valid = valid & prev
		mask = (valid.astype(np.uint8) * 255).reshape(-1, 1)
		return io.NodeOutput(values, mask, int(valid.sum()))


def _as_points3(points) -> np.ndarray:
	"""Nx3 float64 view of a 3D point array; None/empty -> (0, 3).

	An Nx6 (x,y,z,nx,ny,nz) cloud - what the whole surface-matching family
	passes around - is narrowed to its POSITIONS, not reshaped into 2N points.
	A bare `reshape(-1, 3)` read every normal as another point, which is
	silent: 'CV Transform Points 3D' returned twice as many points, half of
	them unit vectors sitting at the origin, and a nearest-distance readout
	computed over them looked plausible while measuring nothing."""
	if points is None:
		return np.zeros((0, 3), np.float64)
	arr = np.asarray(points, np.float64)
	if arr.size == 0:
		return np.zeros((0, 3), np.float64)
	if arr.ndim == 2 and arr.shape[-1] == 6:
		return np.ascontiguousarray(arr[:, :3])
	return arr.reshape(-1, 3)


def _as_colors(colors, n: int) -> np.ndarray:
	"""Nx3 uint8 colours padded/truncated to n rows; None -> white."""
	if colors is None or np.asarray(colors).size == 0:
		return np.full((n, 3), 255, np.uint8)
	col = np.asarray(colors)
	if col.dtype in (np.float32, np.float64):
		col = np.clip(col * 255.0 if float(col.max(initial=0.0)) <= 1.0 else col,
		              0, 255).astype(np.uint8)
	col = col.astype(np.uint8, copy=False).reshape(-1, col.shape[-1])
	if col.shape[1] == 1:
		col = np.repeat(col, 3, axis=1)
	col = col[:, :3]
	if len(col) < n:  # short colour list: pad with white rather than raising
		col = np.vstack([col, np.full((n - len(col), 3), 255, np.uint8)])
	return col[:n]


class TransformPoints3D(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_TransformPoints3D",
			display_name="CV Transform Points 3D",
			category=CATEGORY,
			description="Applies a 3x4 / 4x4 / 3x3 transform to an Nx3 point "
			            "cloud (cv2.transform) - the step that moves a second "
			            "stereo cloud into the first view's coordinate frame "
			            "before merging. Feed the matrix from "
			            "cv2.estimateAffine3D (3x4), 'CV ICP Register' (4x4) or "
			            "any rotation (3x3). 'direction' selects forward (the "
			            "matrix as given) or inverse: for a rigid matrix the "
			            "inverse is built exactly (R^T, -R^T t), otherwise the "
			            "homogeneous 4x4 is inverted numerically. Failure "
			            "tolerant: an empty/None cloud passes through empty and a "
			            "singular matrix returns the points unchanged.",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx3 (or Nx1x3 / HxWx3) point cloud. An Nx6 cloud "
					        "from 'CV Point Cloud Normals' / 'CV Mesh Vertex "
					        "Normals' is accepted and its POSITIONS come back "
					        "Nx3: rotating a cloud rotates its normals too, so "
					        "recompute them after moving it (which is exactly "
					        "the transform -> normals -> ICP chain workflow 81 "
					        "wires)."),
				NPArray.Input("matrix",
					tooltip="3x4 [R|t], 4x4 homogeneous, or 3x3 rotation."),
				io.Combo.Input("direction",
					options=["forward", "inverse"], default="forward",
					tooltip="forward: p' = R p + t (use for a matrix that already "
					        "maps THIS cloud into the target frame). inverse: "
					        "p' = R^-1 (p - t) (use when the matrix maps the "
					        "target frame into this one)."),
			],
			outputs=[NPArray.Output(display_name="points",
				tooltip="Transformed Nx3 float32 points, same order as the input.")],
		)

	@classmethod
	def execute(cls, points, matrix, direction) -> io.NodeOutput:
		pts = _as_points3(points)
		if len(pts) == 0:
			return io.NodeOutput(np.zeros((0, 3), np.float32))

		m = np.asarray(matrix, np.float64)
		if m.shape == (3, 3):
			M = np.hstack([m, np.zeros((3, 1))])
		elif m.shape == (4, 4):
			M = m[:3, :]
		elif m.shape == (3, 4):
			M = m
		else:
			raise ValueError(
				f"matrix must be 3x3, 3x4 or 4x4, got {m.shape} - check what "
				"feeds it (cv2.estimateAffine3D gives 3x4, ICP gives 4x4)."
			)

		if str(direction) == "inverse":
			R, t = M[:, :3], M[:, 3]
			if np.allclose(R @ R.T, np.eye(3), atol=1e-6):
				M = np.hstack([R.T, (-R.T @ t).reshape(3, 1)])  # exact rigid inverse
			else:
				H = np.vstack([M, [0.0, 0.0, 0.0, 1.0]])
				if abs(np.linalg.det(H)) < 1e-12:  # singular: nothing sensible to do
					return io.NodeOutput(np.float32(pts))
				M = np.linalg.inv(H)[:3, :]

		out = cv2.transform(np.float32(pts).reshape(-1, 1, 3), M)
		return io.NodeOutput(np.ascontiguousarray(out.reshape(-1, 3), np.float32))


class MergePointClouds(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MergePointClouds",
			display_name="CV Merge Point Clouds",
			category=CATEGORY,
			description="Concatenates two Nx3 point clouds and their per-vertex "
			            "colours in lockstep, so a multi-view reconstruction can "
			            "go to one 'CV Write PLY'. Both clouds must already be in "
			            "the SAME coordinate frame - register one onto the other "
			            "first ('CV Sample Array At Points' + cv2.estimateAffine3D "
			            "+ 'CV Transform Points 3D'). A missing colour input "
			            "becomes white for that cloud, and an empty/None cloud "
			            "simply contributes nothing (that is how a failed "
			            "registration degrades: you keep cloud A).",
			inputs=[
				NPArray.Input("points_a", tooltip="First Nx3 cloud."),
				NPArray.Input("points_b",
					tooltip="Second Nx3 cloud, already transformed into the same "
					        "frame as points_a. Leave it unconnected and the result "
					        "is cloud A alone, with count_b = 0 - which is also what "
					        "lets a SUBGRAPH expose an OPTIONAL cloud: a boundary "
					        "input may be left unfed only when every inner socket it "
					        "reaches is optional (prompt validation is static, so one "
					        "required socket downstream sinks the whole prompt).",
					optional=True),
				NPArray.Input("colors_a", optional=True,
					tooltip="Nx3 colours for cloud A (white if unconnected)."),
				NPArray.Input("colors_b", optional=True,
					tooltip="Nx3 colours for cloud B (white if unconnected)."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="(Na + Nb) x 3 float32 points, cloud A first."),
				NPArray.Output(display_name="colors",
					tooltip="(Na + Nb) x 3 uint8 colours in the same order."),
				io.Int.Output(display_name="count_a"),
				io.Int.Output(display_name="count_b"),
			],
		)

	@classmethod
	def execute(cls, points_a, points_b=None, colors_a=None, colors_b=None) -> io.NodeOutput:
		pa, pb = _as_points3(points_a), _as_points3(points_b)
		ca = _as_colors(colors_a, len(pa))
		cb = _as_colors(colors_b, len(pb))
		pts = np.vstack([pa, pb]).astype(np.float32)
		col = np.vstack([ca, cb]).astype(np.uint8)
		return io.NodeOutput(pts, col, len(pa), len(pb))


class FilterPoints3DByMask(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FilterPoints3DByMask",
			display_name="CV Filter Points 3D By Mask",
			category=CATEGORY,
			description="Keeps the rows of an Nx3 point cloud flagged non-zero "
			            "in a per-point mask, filtering the per-vertex colours in "
			            "lockstep - the 3D counterpart of 'CV Filter Points "
			            "By Mask' (which is 2D-only). Drive it with the inlier "
			            "mask of cv2.estimateAffine3D, the 'valid' output of 'CV "
			            "Sample Array At Points', or the 'near_mask' of 'CV Point "
			            "Cloud Nearest Distance' to keep only points a second "
			            "view confirms. A None mask passes everything through.",
			inputs=[
				NPArray.Input("points", tooltip="Nx3 (or Nx1x3) point cloud."),
				NPArray.Input("mask", optional=True,
					tooltip="Nx1 mask: rows != 0 are kept. One entry per point; "
					        "None keeps all."),
				NPArray.Input("colors", optional=True,
					tooltip="Nx3 per-vertex colours filtered with the same mask."),
			],
			outputs=[
				NPArray.Output(display_name="points"),
				NPArray.Output(display_name="colors"),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, points, mask=None, colors=None) -> io.NodeOutput:
		pts = _as_points3(points)
		col = _as_colors(colors, len(pts))
		if len(pts) == 0 or mask is None or np.asarray(mask).size == 0:
			return io.NodeOutput(np.float32(pts), col, len(pts))
		keep = np.asarray(mask).ravel().astype(bool)
		if len(keep) != len(pts):
			raise ValueError(
				f"mask has {len(keep)} entries but there are {len(pts)} points - "
				"the mask must come from the same point set."
			)
		return io.NodeOutput(np.float32(pts[keep]), col[keep], int(keep.sum()))


class PointCloudNearestDistance(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PointCloudNearestDistance",
			display_name="CV Point Cloud Nearest Distance",
			category=CATEGORY,
			description="For every point, the distance to the nearest point of a "
			            "reference cloud (cv2.flann k-d tree). Two uses: as a "
			            "cross-view AGREEMENT test - after registering a second "
			            "stereo cloud, points the first view confirms have a tiny "
			            "nearest distance while stereo blunders sit alone - and as "
			            "the honest QUALITY METRIC of a registration (median "
			            "distance drops as the alignment improves; an ICP "
			            "residual does not measure this). Outputs data only: "
			            "combine 'near_mask' with 'CV Filter Points 3D By Mask' to "
			            "keep confirmed points, or invert it to keep only the new "
			            "coverage. Both clouds must be in the same frame.",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx3 cloud to measure (the query points)."),
				NPArray.Input("reference",
					tooltip="Mx3 cloud to measure AGAINST. Empty reference -> "
					        "infinite distances and an all-zero mask, not an error."),
				io.Float.Input("threshold", default=0.25, min=0.0, step=0.01,
					tooltip="Distance (same unit as the cloud, metres for a "
					        "calibrated stereo reconstruction) below which a point "
					        "counts as confirmed by the reference cloud."),
				io.Int.Input("checks", default=32, min=1, max=4096,
					optional=True, advanced=True,
					tooltip="FLANN search effort: leaves visited per query. Higher "
					        "is more exact and slower."),
				io.Int.Input("trees", default=4, min=1, max=16,
					optional=True, advanced=True,
					tooltip="Randomized k-d trees in the index. More = more "
					        "accurate approximate search, slower build."),
			],
			outputs=[
				NPArray.Output(display_name="distance",
					tooltip="Nx1 float32 distance to the nearest reference point."),
				NPArray.Output(display_name="near_mask",
					tooltip="Nx1 uint8 mask, 255 where distance <= threshold."),
				io.Float.Output(display_name="median_distance",
					tooltip="Median nearest distance - the registration quality "
					        "number to watch (lower = better aligned)."),
				io.Float.Output(display_name="matched_fraction",
					tooltip="Fraction of points within the threshold (0..1)."),
			],
		)

	@classmethod
	def execute(cls, points, reference, threshold, checks=32, trees=4) -> io.NodeOutput:
		pts = np.float32(_as_points3(points))
		ref = np.float32(_as_points3(reference))
		if len(pts) == 0:
			return io.NodeOutput(np.zeros((0, 1), np.float32),
			                     np.zeros((0, 1), np.uint8), 0.0, 0.0)
		if len(ref) == 0:  # nothing to compare against is a valid result
			return io.NodeOutput(np.full((len(pts), 1), np.inf, np.float32),
			                     np.zeros((len(pts), 1), np.uint8), 0.0, 0.0)

		# cv2.flann exposes no FLANN_INDEX_* constants in this build - 1 is
		# FLANN_INDEX_KDTREE (same literal MatchFeatures' FLANN path uses)
		index = cv2.flann.Index(np.ascontiguousarray(ref),
		                        {"algorithm": 1, "trees": int(trees)})
		_, sq_dist = index.knnSearch(np.ascontiguousarray(pts), 1,
		                             params={"checks": int(checks)})
		dist = np.sqrt(np.asarray(sq_dist, np.float64).ravel()).astype(np.float32)
		near = dist <= float(threshold)
		return io.NodeOutput(dist.reshape(-1, 1),
		                     (near.astype(np.uint8) * 255).reshape(-1, 1),
		                     float(np.median(dist)), float(near.mean()))


class PointCloudNormals(io.ComfyNode):
	"""cv2.ppf_match_3d.computeNormalsPC3d - per-point surface normals.

	The gatekeeper for the surface-matching family: cv2's PPF detector and its
	ICP both want Nx6 (x,y,z,nx,ny,nz) clouds, and feeding them a bare Nx3
	array does NOT raise - `trainModel` took 341 s on a 2400-point cloud
	(against 3.5 s with normals, 31x on a smaller one) and `match` answers
	from a normal-less scene with a pose that is 85 degrees out on one cloud
	and all-NaN on another - silently, either way. So this
	node exists to be wired in front of them.
	"""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PointCloudNormals",
			display_name="CV Point Cloud Normals",
			category=CATEGORY,
			description="Estimates a surface normal per point by fitting a plane "
			            "to each point's k nearest neighbours "
			            "(cv2.ppf_match_3d.computeNormalsPC3d) and returns the "
			            "Nx6 (x,y,z,nx,ny,nz) cloud the surface-matching nodes "
			            "need. WIRE THIS IN FRONT OF 'CV PPF Pose Estimation' and "
			            "'CV ICP Register': given a bare Nx3 cloud those run "
			            "31x-100x slower and answer with a silently wrong pose "
			            "instead of complaining. The plane fit leaves the SIGN of each "
			            "normal arbitrary; 'flip toward viewpoint' orients them "
			            "all towards a camera position, which is what you want "
			            "for a cloud reconstructed from one view. Failure "
			            "tolerant: fewer than 3 points (or a cv2 error) gives "
			            "empty outputs with found=false.",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx3 point cloud (an Nx6 cloud is accepted - its "
					        "normals are recomputed)."),
				io.Int.Input("num_neighbors", default=12, min=3, max=200,
					tooltip="Neighbours per local plane fit. Higher = smoother "
					        "normals and more smearing across edges; lower = "
					        "noisier. 6-20 is the usual range."),
				io.Combo.Input("orientation",
					options=["as computed (sign arbitrary)",
					         "flip toward viewpoint"],
					default="as computed (sign arbitrary)",
					tooltip="A plane fit cannot tell inside from outside. "
					        "'flip toward viewpoint' points every normal at the "
					        "viewpoint below - correct for a cloud seen from one "
					        "camera, wrong for a closed object sampled all round."),
				io.Float.Input("viewpoint_x", default=0.0, step=0.1,
					optional=True, advanced=True,
					tooltip="Viewpoint the normals are flipped towards (only read "
					        "when orientation is 'flip toward viewpoint')."),
				io.Float.Input("viewpoint_y", default=0.0, step=0.1,
					optional=True, advanced=True,
					tooltip="Viewpoint Y. The camera origin is (0,0,0) for a "
					        "cloud straight out of 'CV Stereo Pair To Cloud'."),
				io.Float.Input("viewpoint_z", default=0.0, step=0.1,
					optional=True, advanced=True,
					tooltip="Viewpoint Z."),
			],
			outputs=[
				NPArray.Output(display_name="cloud",
					tooltip="Nx6 float32 (x,y,z,nx,ny,nz) - feed the surface "
					        "matching nodes with this, not with the Nx3 input."),
				NPArray.Output(display_name="normals",
					tooltip="Nx3 float32 unit normals only, in the input order "
					        "(for drawing or filtering)."),
				io.Boolean.Output(display_name="found",
					tooltip="False when the normals could not be computed "
					        "(fewer than 3 points, cv2 error)."),
			],
		)

	@classmethod
	def execute(cls, points, num_neighbors, orientation, viewpoint_x=0.0,
	            viewpoint_y=0.0, viewpoint_z=0.0) -> io.NodeOutput:
		# _as_points3 narrows an Nx6 cloud to its positions, which is what is
		# wanted here too - kept explicit because this node RECOMPUTES normals
		arr = np.asarray(points, np.float64) if points is not None else None
		if arr is not None and arr.ndim == 2 and arr.shape[-1] == 6:
			pts = arr[:, :3]
		else:
			pts = _as_points3(points)
		empty = (np.zeros((0, 6), np.float32), np.zeros((0, 3), np.float32))
		if len(pts) < 3:  # cv2 asserts on an empty cloud
			return io.NodeOutput(*empty, False)

		flip = str(orientation) == "flip toward viewpoint"
		viewpoint = np.array([[viewpoint_x, viewpoint_y, viewpoint_z]], np.float32)
		try:
			retval, cloud = cv2.ppf_match_3d.computeNormalsPC3d(
				np.ascontiguousarray(pts, np.float32),
				int(num_neighbors), bool(flip), viewpoint)
		except cv2.error as e:
			logging.warning("PointCloudNormals: %s", str(e).splitlines()[-1])
			return io.NodeOutput(*empty, False)
		# the docstring says "Returns 0 on success"; this build returns 1 - so the
		# only usable success test is the cloud itself
		cloud = np.ascontiguousarray(cloud, np.float32).reshape(-1, 6)
		if len(cloud) != len(pts) or not np.all(np.isfinite(cloud)):
			logging.warning("PointCloudNormals: cv2 returned %d rows for %d "
			                "points (retval %s)", len(cloud), len(pts), retval)
			return io.NodeOutput(*empty, False)
		return io.NodeOutput(cloud, np.ascontiguousarray(cloud[:, 3:]), True)


class DownsamplePointCloud(io.ComfyNode):
	"""cv2.ppf_match_3d.samplePCByQuantization - the pack's only point-cloud
	decimator.

	A cloud back-projected from a 640x480 depth map is ~38k points; PPF's
	training is O(sampled^2) and 'CV Point Cloud Normals' fits a plane per
	point, so both want the cloud cut down FIRST and by SPACING, not by stride
	- a depth cloud is in raster order, so taking every Nth row samples the
	image evenly and the surface unevenly.
	"""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DownsamplePointCloud",
			display_name="CV Downsample Point Cloud",
			category=CATEGORY,
			description="Thins a point cloud onto a GRID whose cell size is a "
			            "fraction of the bounding-box diagonal "
			            "(cv2.ppf_match_3d.samplePCByQuantization): one averaged "
			            "representative survives per occupied cell. That sets the "
			            "cloud's DENSITY, not a hard minimum distance - two points "
			            "either side of a cell boundary can still end up close. "
			            "Spacing-based, "
			            "not stride-based - which is the point, because a cloud "
			            "from 'CV Depth to 3D Points' arrives in raster order and "
			            "every Nth point would sample the IMAGE evenly and the "
			            "SURFACE unevenly. Put this in front of 'CV PPF Pose "
			            "Estimation' and 'CV Point Cloud Normals': PPF training "
			            "is quadratic in the sampled count, so a 38k-point depth "
			            "cloud is minutes of work and its 1.4k-point downsample "
			            "is seconds, with the same pose. Nx6 clouds keep their "
			            "normal columns. Failure tolerant: an empty cloud returns "
			            "empty with found=false.",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx3, or Nx6 (x,y,z,nx,ny,nz). The kept point is the "
					        "AVERAGE of its cell, normals included (re-normalized "
					        "here, since cv2 does not) - so a cell that spans two "
					        "opposing surfaces, the two faces of a thin wall or a "
					        "sign-arbitrary plane fit, averages to nothing and "
					        "that row comes back with a ZERO normal. Downsample "
					        "FIRST and compute normals after when the cloud is "
					        "thin or noisy; normals-first is fine (and cheaper) "
					        "for a mesh, where they are exact."),
				io.Float.Input("relative_sampling_step", default=0.05,
					min=0.005, max=0.5, step=0.005,
					tooltip="Grid cell size, as a fraction of the bounding-box "
					        "diagonal: 0.05 leaves points ~5% of the diagonal "
					        "apart on average. Match it to the "
					        "relative_sampling_step of 'CV PPF Pose Estimation' - "
					        "that node quantizes internally at the same scale, so "
					        "a finer cloud than its own step buys nothing and "
					        "costs the plane fits."),
				io.Combo.Input("weighting",
					options=["uniform", "weight by distance to centre"],
					default="uniform", optional=True, advanced=True,
					tooltip="How the representative of a cell is chosen. "
					        "'uniform' averages the cell's points; the weighted "
					        "form pulls it toward the points furthest from the "
					        "cloud's origin, which biases the sample toward the "
					        "silhouette. Leave it uniform unless you are "
					        "deliberately sampling a rim."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="The thinned cloud, float32, with the same number of "
					        "columns as the input."),
				io.Int.Output(display_name="count",
					tooltip="Points that survived. Watch it against the input "
					        "size - this is the number PPF's cost is quadratic "
					        "in."),
				io.Boolean.Output(display_name="found",
					tooltip="False when the cloud was empty or cv2 refused it."),
			],
		)

	@classmethod
	def execute(cls, points, relative_sampling_step, weighting="uniform") -> io.NodeOutput:
		arr = np.asarray(points) if points is not None else None
		if arr is None or arr.size == 0:
			return io.NodeOutput(np.zeros((0, 3), np.float32), 0, False)
		arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(-1, 3)
		# float64 raises "Unknown C++ exception" rather than converting
		pc = np.ascontiguousarray(arr, np.float32)
		if len(pc) == 0 or pc.shape[1] < 3:
			return io.NodeOutput(np.zeros((0, 3), np.float32), 0, False)

		def _span(col):
			return np.array([pc[:, col].min(), pc[:, col].max()], np.float32)

		weight = 1 if str(weighting).startswith("weight") else 0
		try:
			out = cv2.ppf_match_3d.samplePCByQuantization(
				pc, _span(0), _span(1), _span(2),
				float(relative_sampling_step), weight)
		except cv2.error as e:
			logging.warning("DownsamplePointCloud: %s", str(e).splitlines()[-1])
			return io.NodeOutput(np.zeros((0, pc.shape[1]), np.float32), 0, False)
		out = np.ascontiguousarray(out, np.float32).reshape(-1, pc.shape[1])
		degenerate = 0
		if out.shape[1] >= 6 and len(out):
			# cv2 averages the cell without re-normalizing, so the normal
			# columns come back short (and exactly zero where two opposing
			# surfaces fell in one cell) - PPF quantizes the ANGLE between
			# normals, which a non-unit vector distorts
			length = np.linalg.norm(out[:, 3:6], axis=1, keepdims=True)
			degenerate = int((length < 1e-6).sum())
			length[length < 1e-6] = 1.0
			out[:, 3:6] /= length
		logging.info("DownsamplePointCloud: %d -> %d points (step %.3f)%s",
		             len(pc), len(out), float(relative_sampling_step),
		             f", {degenerate} cancelled normals" if degenerate else "")
		return io.NodeOutput(out, len(out), len(out) > 0)


NODES = [AnnotatePoints, AnnotateCorrespondences,
         FilterPointsByDistance, FilterPointsByMask, DownsamplePointCloud,
         PointsToPolar, PolarToPoints, ReducePointsByLabel,
         ReduceValuesByLabel, PickValue, DrawPoints, DrawConnections, DrawLabels,
         DrawRays, DrawSegments, DrawCircles, DrawEllipses, DrawFlowVectors,
         DrawPath, DrawPlotFrame, ProjectToPlane,
         FitPointsToBox, ScalePoints, DelaunayTriangulation,
         SampleArrayAtPoints, TransformPoints3D, MergePointClouds,
         FilterPoints3DByMask, PointCloudNearestDistance, PointCloudNormals]
