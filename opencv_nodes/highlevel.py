# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""High-level OpenCV nodes that work directly on ComfyUI IMAGE/MASK tensors.

No manual Image<->ndarray conversion, no magic integers, batch-aware. Color
and dtype conversions happen internally; enum parameters are dropdowns.
"""

import logging
import math
import os

import cv2
import numpy as np
import torch

import comfy.utils
from comfy_api.latest import io, ui

from . import boxpolicy, css_colors, enums, imageutils, polytype
from .convert import NPArray

_log = logging.getLogger(__name__)
CATEGORY = "image/CV"


def _map_frames(image: torch.Tensor, fn) -> torch.Tensor:
	"""Apply fn(bgr_uint8) -> bgr/gray uint8 to every image in the batch."""
	return imageutils.bgr_to_image_batch([fn(f) for f in imageutils.image_batch_to_bgr(image)])


def _gray(frame_bgr: np.ndarray) -> np.ndarray:
	return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


class Transform(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Transform",
			display_name="CV Transform (Rotate/Scale/Shift)",
			category=CATEGORY,
			description="Rotates around the center, scales, shifts and/or flips the "
			            "image in one step (cv2.warpAffine under the hood).",
			inputs=[
				io.Image.Input("image",
					tooltip="Input image. A batch is processed frame by frame."),
				io.Float.Input("angle", default=0.0, min=-3600.0, max=3600.0, step=0.5,
					tooltip="Rotation in degrees, counter-clockwise."),
				io.Float.Input("zoom", default=1.0, min=0.01, max=16.0, step=0.05,
					tooltip="Uniform scale around the center. 1.0 = no change."),
				io.Float.Input("shift_x", default=0.0, min=-16384.0, max=16384.0,
					tooltip="Horizontal shift in pixels."),
				io.Float.Input("shift_y", default=0.0, min=-16384.0, max=16384.0,
					tooltip="Vertical shift in pixels."),
				io.Combo.Input("flip", options=["none"] + enums.FLIP.options, default="none",
					tooltip="Optional mirror applied before rotate/scale: horizontal, "
					        "vertical, or both."),
				io.Combo.Input("interpolation", options=enums.INTER.options, default=enums.INTER.default,
					tooltip="Resampling filter for the warp. NEAREST keeps hard edges; "
					        "LINEAR/CUBIC are smoother for photos."),
				io.Combo.Input("border", options=enums.BORDER.options, default="BORDER_CONSTANT",
					tooltip="What to fill the uncovered areas with (CONSTANT = black)."),
			],
			outputs=[io.Image.Output()],
		)

	@classmethod
	def execute(cls, image, angle, zoom, shift_x, shift_y, flip, interpolation, border) -> io.NodeOutput:
		inter = enums.INTER.resolve(interpolation)
		bt = enums.BORDER.resolve(border)

		def fn(f):
			if flip != "none":
				f = cv2.flip(f, enums.FLIP.resolve(flip))
			h, w = f.shape[:2]
			m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
			m[0, 2] += shift_x
			m[1, 2] += shift_y
			return cv2.warpAffine(f, m, (w, h), flags=inter, borderMode=bt)
		return io.NodeOutput(_map_frames(image, fn))


class Contrast(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Contrast",
			display_name="CV Contrast (CLAHE/Equalize)",
			category=CATEGORY,
			description="Improves contrast via CLAHE (adaptive, recommended) or "
			            "global histogram equalization.",
			inputs=[
				io.Image.Input("image",
					tooltip="Input image. A batch is processed frame by frame."),
				io.Combo.Input("method", options=["CLAHE (adaptive)", "histogram equalization"],
					default="CLAHE (adaptive)",
					tooltip="CLAHE equalizes contrast locally and limits noise "
					        "amplification (recommended); histogram equalization "
					        "stretches contrast globally (simpler, can blow out)."),
				io.Combo.Input("channels", options=["luminance only (LAB)", "each RGB channel"],
					default="luminance only (LAB)",
					tooltip="Luminance-only keeps colors natural; per-channel can "
					        "shift colors but is more dramatic."),
				io.Float.Input("clip_limit", default=2.0, min=0.1, max=40.0, step=0.1,
					tooltip="CLAHE only: contrast limiting; higher = stronger."),
				io.Int.Input("tile_size", default=8, min=1, max=64,
					tooltip="CLAHE only: grid size of local regions."),
			],
			outputs=[io.Image.Output()],
		)

	@classmethod
	def execute(cls, image, method, channels, clip_limit, tile_size) -> io.NodeOutput:
		if method.startswith("CLAHE"):
			clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
			enhance = clahe.apply
		else:
			enhance = cv2.equalizeHist

		def fn(f):
			if channels.startswith("luminance"):
				lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
				lab[..., 0] = enhance(lab[..., 0])
				return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
			return cv2.merge([enhance(c) for c in cv2.split(f)])
		return io.NodeOutput(_map_frames(image, fn))


class ColorMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ColorMap",
			display_name="CV Color Map",
			category=CATEGORY,
			description="Converts the image to grayscale and applies a false-color "
			            "map (viridis, jet, inferno, ...). Useful for visualizing "
			            "depth maps, heat maps and masks. Enable 'color_scale' to "
			            "draw a labelled colour bar (0..1 intensity) beside the "
			            "result so the colours can be read back as values.",
			inputs=[
				io.Image.Input("image",
					tooltip="Input image. A batch is processed frame by frame."),
				io.Combo.Input("colormap", options=enums.COLORMAP.options, default=enums.COLORMAP.default,
					tooltip="False-color palette. VIRIDIS/INFERNO are perceptually "
					        "uniform (best for data); JET is the classic rainbow."),
				io.Boolean.Input("color_scale", default=False, optional=True,
					tooltip="Append a vertical colour scale bar with 0..1 "
					        "intensity labels next to the colour-mapped image "
					        "(stays legible even for tiny / 1-pixel images)."),
			],
			outputs=[io.Image.Output()],
		)

	@classmethod
	def execute(cls, image, colormap, color_scale=False) -> io.NodeOutput:
		cm = enums.COLORMAP.resolve(colormap)

		def render(f):
			out = cv2.applyColorMap(_gray(f), cm)
			return imageutils.attach_color_scale(out, 0.0, 1.0, cm) if color_scale else out

		return io.NodeOutput(_map_frames(image, render))


class DetectLines(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DetectLines",
			display_name="CV Detect Lines (Hough)",
			category=CATEGORY,
			description="Detects straight line segments (Canny + probabilistic Hough "
			            "transform) and draws them.",
			inputs=[
				io.Image.Input("image",
					tooltip="Input image. A batch is processed frame by frame."),
				io.Float.Input("canny_low", default=50.0, min=0.0, max=5000.0,
					tooltip="Lower Canny hysteresis threshold for the edge pass that "
					        "feeds Hough; edges weaker than this are discarded."),
				io.Float.Input("canny_high", default=150.0, min=0.0, max=5000.0,
					tooltip="Upper Canny hysteresis threshold; edges stronger than "
					        "this are always kept."),
				io.Int.Input("threshold", default=50, min=1, max=10000,
					tooltip="Minimum number of votes - higher finds fewer, stronger lines."),
				io.Float.Input("min_length", default=50.0, min=1.0, max=16384.0,
					tooltip="Minimum segment length in pixels."),
				io.Float.Input("max_gap", default=10.0, min=0.0, max=1000.0,
					tooltip="Maximum gap between collinear segments to merge them."),
				io.Int.Input("thickness", default=2, min=1, max=64,
					tooltip="Line width in pixels used to draw the detected segments."),
			],
			outputs=[
				io.Image.Output(display_name="overlay"),
				io.Image.Output(display_name="lines_only"),
			],
		)

	@classmethod
	def execute(cls, image, canny_low, canny_high, threshold, min_length, max_gap, thickness) -> io.NodeOutput:
		overlays, lines_only = [], []
		for f in imageutils.image_batch_to_bgr(image):
			edges = cv2.Canny(_gray(f), canny_low, canny_high)
			lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
			                        minLineLength=min_length, maxLineGap=max_gap)
			overlay = f.copy()
			canvas = np.zeros_like(f)
			if lines is not None:
				# cv2 4.x returns (N, 1, 4), cv2 5.x returns (N, 4) - accept both
				for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
					cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), thickness, cv2.LINE_AA)
					cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), thickness, cv2.LINE_AA)
			overlays.append(overlay)
			lines_only.append(canvas)
		return io.NodeOutput(
			imageutils.bgr_to_image_batch(overlays),
			imageutils.bgr_to_image_batch(lines_only),
		)


def _css_lab_centers() -> np.ndarray:
	"""OpenCV L*a*b* (uint8 space, float32) of every embedded CSS color, cached."""
	if not hasattr(_css_lab_centers, "_cache"):
		bgr = np.zeros((len(css_colors.CSS_COLORS), 1, 3), np.uint8)
		for i, (_, hx) in enumerate(css_colors.CSS_COLORS):
			bgr[i, 0] = (int(hx[5:7], 16), int(hx[3:5], 16), int(hx[1:3], 16))
		_css_lab_centers._cache = cv2.cvtColor(
			bgr, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
	return _css_lab_centers._cache


def _min_lab_distances(pixels: np.ndarray, centers: np.ndarray, l1: bool) -> np.ndarray:
	"""Nearest-center Lab distance per center, chunked over pixels to bound memory."""
	best = np.full(len(centers), np.inf, np.float64)
	for i in range(0, len(pixels), 65536):
		diff = pixels[i:i + 65536][:, None, :] - centers[None, :, :]
		if l1:
			d = np.abs(diff).sum(axis=2)
		else:
			d = np.sqrt((diff * diff).sum(axis=2))
		best = np.minimum(best, d.min(axis=0))
	return best


class AbsentColor(io.ComfyNode):
	_METRICS = ("Euclidean (L2)", "Manhattan (L1)")

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_AbsentColor",
			display_name="CV Find Absent Color (Lab)",
			category=CATEGORY,
			description="Finds a CSS named color that does not appear in the image "
			            "and sits furthest (in L*a*b*) from the colors that do. "
			            "Converts the image to L*a*b*, measures each of the 148 CSS "
			            "named colors' distance to its nearest pixel (Euclidean L2 "
			            "or Manhattan L1), treats colors within 'threshold' as "
			            "present, and returns the absent color with the greatest "
			            "distance - its RGB scalar, CSS name and "
			            "#RRGGBB hex. If every color is present, 'found' is false "
			            "and the returned color is still the one furthest from the "
			            "image's existing colors.",
			inputs=[
				io.Image.Input("image",
					tooltip="Input image. A batch is accepted: a color counts as "
					        "present when it occurs in ANY frame."),
				io.Combo.Input("metric", options=list(cls._METRICS),
					default=cls._METRICS[0],
					tooltip="Distance in L*a*b* space: Euclidean (L2) is the "
					        "classic delta-E; Manhattan (L1) sums the channel "
					        "differences and is cheaper to reason about per "
					        "channel."),
				io.Float.Input("threshold", default=10.0, min=0.0, max=255.0,
					step=0.5,
					tooltip="A CSS color counts as present when any pixel is "
					        "within this Lab distance (OpenCV L*a*b* uint8 space: "
					        "L 0-255, a/b 0-255). 0 = only an exact match counts; "
					        "larger = more colors considered present."),
				io.Int.Input("max_pixels", default=200000, min=1000,
					max=100_000_000, optional=True, advanced=True,
					tooltip="Per-frame cap on pixels searched, applied as an even "
					        "deterministic stride so big images stay fast. A color "
					        "present in only a few pixels below this stride may "
					        "be missed - raise it for large canvases."),
			],
			outputs=[
				NPArray.Output(display_name="color_rgb",
					tooltip="CV Scalar (r, g, b) float64 of the chosen CSS "
					        "color (the image arrives in RGB, so the color is "
					        "returned in the same order)."),
				io.String.Output(display_name="color_name",
					tooltip="CSS name of the chosen color, lowercase (e.g. "
					        "'rebeccapurple')."),
				io.Color.Output(display_name="color_hex",
					tooltip="Hex as #RRGGBB uppercase, typed as a COLOR socket - "
					        "links straight into ComfyUI color-picker inputs "
					        "(e.g. 'CV Color From Hex')."),
				io.Float.Output(display_name="distance",
					tooltip="Lab distance of the chosen color to its nearest image "
					        "pixel (its distance to the existing colors)."),
				io.Boolean.Output(display_name="found",
					tooltip="True when an absent color exists (distance above the "
					        "threshold). False when every CSS color is present - "
					        "the color is then still the furthest of the 148 from "
					        "the existing colors."),
			],
		)

	@classmethod
	def execute(cls, image, metric="Euclidean (L2)", threshold=10.0,
	            max_pixels=200000) -> io.NodeOutput:
		centers = _css_lab_centers()
		l1 = metric.startswith("Manhattan")
		best = np.full(len(centers), np.inf, np.float64)
		for f in imageutils.image_batch_to_bgr(image):
			lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
			if len(lab) > max_pixels:
				step = int(np.ceil(len(lab) / max_pixels))
				lab = lab[::step]
			if len(lab) == 0:
				continue
			best = np.minimum(best, _min_lab_distances(lab, centers, l1))
		return cls._pick(best, threshold)

	@classmethod
	def _pick(cls, best: np.ndarray, threshold: float) -> io.NodeOutput:
		n = len(css_colors.CSS_COLORS)
		if not np.isfinite(best).any():  # empty image / empty batch
			name, hx, dist, found = "black", "#000000", 0.0, False
		else:
			absent = np.where(best > threshold)[0]
			if len(absent):
				i, found = int(absent[np.argmax(best[absent])]), True
			else:
				i, found = int(np.argmax(best)), False
			name, hx = css_colors.CSS_COLORS[i]
			dist = float(best[i])
		hx = hx.upper()
		scalar = np.asarray([int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)],
		                    np.float64)
		return io.NodeOutput(scalar, name, hx, dist, found)


def _is_min_method(flag: int) -> bool:
	return flag in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)


def _match_confidence(value: float, flag: int) -> float:
	"""Map the raw matchTemplate score to 'higher is better'."""
	if flag == cv2.TM_SQDIFF_NORMED:
		return 1.0 - value
	if flag == cv2.TM_SQDIFF:
		return -value
	return value


def _heatmap_u8(result: np.ndarray) -> np.ndarray:
	return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


_NORMED_FLAGS = (cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED)


def _match_template_nd(f, tmpl, flag, mask=None):
	"""cv2.matchTemplate for any channel count (cv2 caps it at 4 channels).

	More channels are matched in chunks of 4: the raw methods (SQDIFF/CCORR/
	CCOEFF) sum per-channel terms, so summing chunk maps is exact; the _NORMED
	methods average the per-chunk normalized maps instead (a slight
	approximation of a jointly-normalized score, still peaked at the match)."""
	c = 1 if f.ndim == 2 else f.shape[2]
	if c <= 4:
		return cv2.matchTemplate(f, tmpl, flag, mask=mask)
	maps = [cv2.matchTemplate(np.ascontiguousarray(f[:, :, i:i + 4]),
	                          np.ascontiguousarray(tmpl[:, :, i:i + 4]),
	                          flag, mask=mask)
	        for i in range(0, c, 4)]
	return np.mean(maps, axis=0) if flag in _NORMED_FLAGS else np.sum(maps, axis=0)


def _latent_frames(latent) -> list:
	samples = latent["samples"]
	if samples.ndim != 4:
		raise ValueError(
			f"Only 4-D [B,C,H,W] latents are supported, got shape "
			f"{tuple(samples.shape)}.")
	return [np.ascontiguousarray(s.movedim(0, -1).float().cpu().numpy())
	        for s in samples]


def _match_frames(image, template):
	"""Resolve the image+template pair to cv2 arrays for template matching.

	Both IMAGE -> uint8 BGR frames; both LATENT -> float32 [H,W,C] frames in
	latent space (searching a compressed image for a compressed patch).
	Mixing the two raises: they live in incompatible spaces."""
	img_lat, tmpl_lat = polytype.is_latent(image), polytype.is_latent(template)
	if img_lat != tmpl_lat:
		raise ValueError(
			"image and template must be the same kind - matching an IMAGE "
			"against a LATENT (or vice versa) compares incompatible spaces. "
			"Encode or decode one side first.")
	if img_lat:
		frames = _latent_frames(image)
		tmpl = _latent_frames(template)[0]
		if frames[0].shape[2] != tmpl.shape[2]:
			raise ValueError(
				f"image and template latents have different channel counts "
				f"({frames[0].shape[2]} vs {tmpl.shape[2]}) - they come from "
				"different model families.")
		return frames, tmpl, True
	return (list(imageutils.image_batch_to_bgr(image)),
	        imageutils.image_batch_to_bgr(template)[0], False)


def _latent_crop_batch(crops: list) -> dict:
	return {"samples": torch.from_numpy(np.stack(crops)).movedim(-1, 1).contiguous()}


class MatchTemplateMultiScale(io.ComfyNode):
	_TEMPLATE = io.MatchType.Template("image_type", [io.Image, io.Latent])

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MatchTemplateMultiScale",
			display_name="CV Match Template Multi-Scale",
			category=CATEGORY,
			description="Template matching when the size in the image is unknown: "
			            "tries a range of template scales and keeps the best "
			            "match - the usual workaround for matchTemplate not being "
			            "scale-invariant, packed into one node instead of a loop. "
			            "Restricted to normalized methods so scores are "
			            "comparable across scales. Chain bbox into the core "
			            "'Draw BBoxes' node to visualize the match. Also "
			            "accepts LATENT pairs (coordinates then count latent "
			            "cells).",
			inputs=[
				io.MatchType.Input("image", template=cls._TEMPLATE,
					tooltip="Image to search in. Accepts a LATENT too (then the "
					        "template must be a LATENT from the same model and "
					        "match_crop comes back as a LATENT)."),
				io.MultiType.Input("template", [io.Image, io.Latent],
					tooltip="Patch to search for. Same kind as 'image': IMAGE "
					        "with IMAGE, LATENT with LATENT."),
				io.Combo.Input("method",
					options=["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"],
					default="TM_CCOEFF_NORMED",
					tooltip="Normalized comparison method. TM_CCOEFF_NORMED is the most "
					        "robust general choice; all three give scores comparable "
					        "across scales."),
				io.Float.Input("scale_min", default=0.25, min=0.01, max=8.0, step=0.05,
					tooltip="Smallest template scale to try."),
				io.Float.Input("scale_max", default=2.0, min=0.01, max=8.0, step=0.05,
					tooltip="Largest template scale to try."),
				io.Int.Input("scale_steps", default=20, min=2, max=100,
					tooltip="Number of scales tried between scale_min and scale_max "
					        "(geometric spacing). More steps = more precise, slower."),
				io.Mask.Input("template_mask", optional=True,
					tooltip="Optional mask for non-rectangular templates (white = "
					        "compare). Resized along with the template at each scale. "
					        "Only supported by TM_CCORR_NORMED."),
			],
			outputs=[
				io.MatchType.Output(template=cls._TEMPLATE, display_name="match_crop",
					tooltip="The matched region, in the 'image' input's format "
					        "(IMAGE in -> IMAGE crop, LATENT in -> LATENT crop)."),
				io.Float.Output(display_name="score",
					tooltip="Best confidence across all scales (higher = better)."),
				io.Float.Output(display_name="best_scale",
					tooltip="Template scale that produced the best match."),
				io.BoundingBox.Output(display_name="bbox",
					tooltip=BBOX_TOOLTIP + " The match region; includes the score "
					"and scale as extra keys. A list of per-image boxes for batches."),
			],
		)

	@classmethod
	def execute(cls, image, template, method, scale_min, scale_max, scale_steps, template_mask=None) -> io.NodeOutput:
		flag = enums.TEMPLATE_MATCH.resolve(method)
		frames, tmpl, latent = _match_frames(image, template)
		th0, tw0 = tmpl.shape[:2]

		mask = None
		if template_mask is not None:
			if flag != cv2.TM_CCORR_NORMED:
				raise ValueError(
					"template_mask is only supported with TM_CCORR_NORMED (got "
					f"{method}). Disconnect the mask or switch the method."
				)
			mask = imageutils.fit_mask_to(imageutils.mask_batch_to_u8(template_mask)[0], (th0, tw0))

		if scale_min > scale_max:
			scale_min, scale_max = scale_max, scale_min
		scales = np.geomspace(scale_min, scale_max, scale_steps)

		crops, bboxes = [], []
		first_conf, first_scale = 0.0, 1.0
		pbar = comfy.utils.ProgressBar(len(frames) * len(scales))
		# a preview send skips the bar's own throttle - keep them to ~20 a run
		preview_every = max(1, len(frames) // 20)
		for fi, f in enumerate(frames):
			best = None  # (confidence, x, y, w, h, scale)
			for scale in scales:
				pbar.update(1)  # before the fit check: a skipped scale still counts
				w = max(4, int(round(tmpl.shape[1] * scale)))
				h = max(4, int(round(tmpl.shape[0] * scale)))
				if h > f.shape[0] or w > f.shape[1]:
					continue
				# INTER_AREA is capped at 4 channels - many-channel latents
				# downscale with INTER_LINEAR instead
				area_ok = tmpl.ndim == 2 or tmpl.shape[2] <= 4
				interp = cv2.INTER_AREA if scale < 1.0 and area_ok else cv2.INTER_LINEAR
				resized = cv2.resize(tmpl, (w, h), interpolation=interp)
				resized_mask = None
				if mask is not None:
					resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
				result = _match_template_nd(f, resized, flag, mask=resized_mask)
				result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
				value, (x, y) = (min_val, min_loc) if _is_min_method(flag) else (max_val, max_loc)
				conf = _match_confidence(value, flag)
				if best is None or conf > best[0]:
					best = (conf, x, y, w, h, float(scale))

			if best is None:
				raise ValueError(
					f"The template ({tmpl.shape[1]}x{tmpl.shape[0]}) does not fit "
					f"into the image ({f.shape[1]}x{f.shape[0]}) at any scale in "
					f"[{scale_min}, {scale_max}]. Lower scale_min."
				)
			conf, x, y, w, h, scale = best
			crop = f[y:y + h, x:x + w].copy()
			if crops and crop.shape != crops[0].shape:  # batch frames must stack
				crop = cv2.resize(crop, (crops[0].shape[1], crops[0].shape[0]))
			crops.append(crop)
			bboxes.append(_bbox_dict(x, y, w, h, score=round(conf, 4), scale=round(scale, 4)))
			if len(bboxes) == 1:
				first_conf, first_scale = conf, scale
			if fi % preview_every == 0 or fi == len(frames) - 1:
				# no count change - just attach the match to the tick already made
				pbar.update_absolute(pbar.current, None,
				                     imageutils.progress_preview(crop))

		return io.NodeOutput(
			_latent_crop_batch(crops) if latent else imageutils.bgr_to_image_batch(crops),
			float(first_conf),
			float(first_scale),
			[[b] for b in bboxes],  # one match per frame -> one inner list per frame
		)


BBOX_TOOLTIP = ("Core BOUNDING_BOX data: per-frame lists of {x, y, width, "
                "height} dicts - compatible with Draw BBoxes, Crop By "
                "Bounding Boxes, Image Crop, etc.")

# one implementation, in boxpolicy.py (shared with the crop/paste nodes)
_bbox_dict = boxpolicy.bbox_dict
_normalize_bboxes = boxpolicy.normalize
_nest_bboxes = boxpolicy.nest


def _cluster_color(index: int, total: int) -> np.ndarray:
	"""One BGR colour per cluster/mask index, from the CVD-safe categorical set.

	`total` is accepted (and ignored) so callers need not know the palette size:
	a fixed table beats a hue sweep scaled to the count, because a sweep puts
	green and yellow at nearly equal luminance whenever `total` is 4 or 5 - two
	masks a dichromat cannot separate, and nobody can separate in greyscale.
	"""
	return np.array(imageutils.categorical_bgr(index), dtype=np.uint8)


class ConnectedComponents(io.ComfyNode):
	_SORTS = {
		"largest first": lambda c: -c[2],
		"left to right": lambda c: c[1][0],
		"top to bottom": lambda c: c[1][1],
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ConnectedComponents",
			display_name="CV Connected Components (Split Mask)",
			category=CATEGORY,
			description="Splits a mask into its separate regions "
			            "(cv2.connectedComponentsWithStats): one MASK per blob as "
			            "a batch, plus bounding boxes. Chain into 'CV Crop by "
			            "Masks' to process each region individually (e.g. schedule "
			            "one inpainting per region); visualize with 'OpenCV "
			            "Overlay Masks' or the core 'Draw BBoxes' node.",
			inputs=[
				io.Mask.Input("mask", tooltip="Mask to split. A batch is merged "
					"(union) before splitting."),
				io.Combo.Input("connectivity", options=["8", "4"], default="8",
					tooltip="8 also connects diagonal pixels; 4 only direct neighbors."),
				io.Int.Input("min_area", default=16, min=1, max=2**31 - 1,
					tooltip="Regions smaller than this many pixels are dropped "
					        "(removes noise specks)."),
				io.Int.Input("max_regions", default=64, min=1, max=1024,
					tooltip="Keep at most this many regions (after sorting)."),
				io.Combo.Input("sort", options=list(cls._SORTS), default="largest first",
					tooltip="Order of the output region masks/boxes: by area "
					        "(largest first), left-to-right, or top-to-bottom."),
			],
			outputs=[
				io.Mask.Output(display_name="region_masks",
					tooltip="One full-size mask per region, as a MASK batch."),
				io.BoundingBox.Output(display_name="bboxes", tooltip=BBOX_TOOLTIP),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, mask, connectivity="8", min_area=16,
	            max_regions=64, sort="largest first") -> io.NodeOutput:
		frames = imageutils.mask_batch_to_u8(mask)
		union = frames[0]
		for m in frames[1:]:
			union = cv2.bitwise_or(union, imageutils.fit_mask_to(m, union.shape[:2]))
		binary = cv2.threshold(union, 127, 255, cv2.THRESH_BINARY)[1]
		h, w = binary.shape[:2]

		n, labels_img, stats, centroids = cv2.connectedComponentsWithStats(
			binary, connectivity=int(connectivity))
		comps = []  # (mask, (x, y, w, h), area, centroid)
		for i in range(1, n):  # 0 is the background
			area = int(stats[i, cv2.CC_STAT_AREA])
			if area < min_area:
				continue
			bbox = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
			        int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
			blob = np.where(labels_img == i, 255, 0).astype(np.uint8)
			comps.append((blob, bbox, area, [round(float(c), 1) for c in centroids[i]]))
		if not comps:
			# empty mask (or only specks below min_area): emit empty batches so the
			# rest of the workflow does nothing instead of raising.
			return io.NodeOutput(imageutils.u8_to_mask_batch([]), _nest_bboxes([]), 0)
		comps.sort(key=cls._SORTS[sort])
		comps = comps[:max_regions]

		return io.NodeOutput(
			imageutils.u8_to_mask_batch([c[0] for c in comps]),
			_nest_bboxes([_bbox_dict(*c[1]) for c in comps]),
			len(comps),
		)


class MasksToBBoxes(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MasksToBBoxes",
			display_name="CV Masks to BBoxes",
			category=CATEGORY,
			description="One tight bounding box per mask of a MASK batch - the "
			            "generic bridge from region masks (Connected Components, "
			            "Labels to Masks, Region Properties + the K-Means Mask "
			            "Clusters blueprint, Seg Masks, "
			            "GrabCut) to anything that consumes BOUNDING_BOX: 'OpenCV "
			            "Crop by BBoxes', 'CV Snap BBoxes', the core 'Draw "
			            "BBoxes' / 'Crop By Bounding Boxes'. Empty masks are "
			            "dropped by default; 'valid' says which input rows "
			            "survived, so the boxes can be realigned with the batch. "
			            "Empty input yields empty outputs and never raises.",
			inputs=[
				io.Mask.Input("masks",
					tooltip="One mask per region (MASK batch). A single mask "
					        "yields a single box around ALL of its white pixels - "
					        "split it with 'CV Connected Components' first if "
					        "you want one box per blob."),
				io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01,
					optional=True, advanced=True,
					tooltip="Mask values above this count as inside the region. "
					        "Raise it to ignore a soft/feathered halo."),
				io.Boolean.Input("drop_empty", default=True,
					optional=True, advanced=True,
					tooltip="Skip masks with no pixels above the threshold. Turn "
					        "off to keep the batch rows aligned one-to-one - empty "
					        "masks then emit a zero-size box at the origin."),
			],
			outputs=[
				io.BoundingBox.Output(display_name="bboxes", tooltip=BBOX_TOOLTIP),
				NPArray.Output(display_name="areas",
					tooltip="(B,) int32: the pixel count of each region, in the "
					        "same order as the boxes."),
				NPArray.Output(display_name="valid",
					tooltip="(N,) bool, one entry per INPUT mask: True where a box "
					        "was emitted. Use it to realign the boxes with the "
					        "original batch when drop_empty is on."),
				io.Int.Output(display_name="count",
					tooltip="Number of boxes emitted."),
			],
		)

	@classmethod
	def execute(cls, masks, threshold=0.5, drop_empty=True) -> io.NodeOutput:
		frames = imageutils.mask_batch_to_u8(masks)
		cutoff = int(round(float(threshold) * 255.0))
		bboxes, areas, valid = [], [], []
		for m in frames:
			binary = cv2.threshold(m, cutoff, 255, cv2.THRESH_BINARY)[1]
			area = int(cv2.countNonZero(binary))
			valid.append(area > 0)
			if area == 0:
				if drop_empty:
					continue
				bboxes.append(_bbox_dict(0, 0, 0, 0))
			else:
				bboxes.append(_bbox_dict(*cv2.boundingRect(binary)))
			areas.append(area)
		return io.NodeOutput(
			_nest_bboxes(bboxes),
			np.int32(areas) if areas else np.zeros((0,), np.int32),
			np.array(valid, dtype=bool),
			len(bboxes),
		)


class BBoxesToMasks(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_BBoxesToMasks",
			display_name="CV BBoxes to Masks",
			category=CATEGORY,
			description="Filled rectangle masks from bounding boxes - the inverse "
			            "of 'CV Masks to BBoxes'. Turns detections (YuNet, "
			            "YOLO, Match Template, SAM3) into masks you can feed to "
			            "inpainting, 'CV Crop by Masks', or any mask "
			            "arithmetic. Empty input yields an empty MASK batch (rule "
			            "4 - never raises).",
			inputs=[
				io.MultiType.Input("space", polytype.with_latent(polytype.IMAGEISH),
					tooltip="Reference the boxes were measured on - only its "
					        "height/width are used, so any IMAGE, MASK, NPARRAY or "
					        "LATENT works. For a LATENT that is the latent-cell "
					        "grid ('CV Scale BBoxes' converts between spaces)."),
				io.BoundingBox.Input("bboxes", force_input=True, tooltip=BBOX_TOOLTIP),
				io.Boolean.Input("merge", default=False, optional=True, advanced=True,
					tooltip="Union every box into ONE mask instead of emitting one "
					        "mask per box."),
			],
			outputs=[
				io.Mask.Output(display_name="masks",
					tooltip="One full-size mask per box (or a single merged mask), "
					        "white inside the rectangle."),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, space, bboxes, merge=False) -> io.NodeOutput:
		h, w = polytype.spatial_size(space)
		boxes = boxpolicy.normalize(bboxes)
		if not boxes:
			return io.NodeOutput(imageutils.u8_to_mask_batch([]), 0)
		frames = []
		canvas = np.zeros((h, w), np.uint8)
		for b in boxes:
			x, y = max(0, int(b["x"])), max(0, int(b["y"]))
			x1 = min(w, int(b["x"]) + int(b["width"]))
			y1 = min(h, int(b["y"]) + int(b["height"]))
			if not merge:
				canvas = np.zeros((h, w), np.uint8)
			if x1 > x and y1 > y:
				canvas[y:y1, x:x1] = 255
			if not merge:
				frames.append(canvas)
		if merge:
			frames = [canvas]
		return io.NodeOutput(imageutils.u8_to_mask_batch(frames), len(frames))


class SnapBBoxes(io.ComfyNode):
	"""Authors a crop sizing policy once and fans it out (the 'CV * Flags'
	idiom), and optionally applies it to a box list on the spot."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SnapBBoxes",
			display_name="CV Snap BBoxes (Crop Policy)",
			category=CATEGORY,
			description="Authors a crop sizing policy - padding, squareness, a "
			            "size that is a multiple of the VAE stride, one common "
			            "size so crops stack into a batch - and fans it out: wire "
			            "the 'policy' STRING output into the crop policy input of "
			            "any number of 'CV Crop by BBoxes' / 'CV Crop by "
			            "Masks' nodes and they all share one configuration (their "
			            "own policy widgets hide while linked). Wire 'bboxes' in "
			            "as well and it also returns the snapped boxes, e.g. for "
			            "the core 'Draw BBoxes' or 'Crop By Bounding Boxes'. "
			            "Order of operations: pad -> size mode -> square -> round "
			            "up to the size multiple -> cap at max size -> clamp "
			            "inside the reference space (re-centred, never resized "
			            "again). The two caps round DOWN to a multiple, so a "
			            "clamped crop still satisfies 'size multiple'.",
			inputs=[
				io.String.Input("policy",
					default=boxpolicy.compose(boxpolicy.MODE_ONE_BATCH),
					tooltip=boxpolicy.MODE_TOOLTIP + " " + boxpolicy.POLICY_TOOLTIP,
					extra_dict={"cv_policy": boxpolicy.spec(boxpolicy.MODE_ONE_BATCH)}),
				io.BoundingBox.Input("bboxes", force_input=True, optional=True,
					tooltip=BBOX_TOOLTIP + " Optional - without it the node is a "
					"pure policy source and 'bboxes' comes back empty."),
				io.MultiType.Input("space", polytype.with_latent(polytype.IMAGEISH),
					optional=True,
					tooltip="Optional reference the snapped boxes must stay "
					        "inside; only its height/width are read. Leave it "
					        "unconnected to let boxes grow past the image edges "
					        "(the crop nodes clamp against their own source "
					        "anyway)."),
			],
			outputs=[
				io.String.Output(display_name="policy",
					tooltip="The pipe-joined policy string - wire it into the "
					        "'policy' input of the crop nodes."),
				io.BoundingBox.Output(display_name="bboxes",
					tooltip=BBOX_TOOLTIP + " The snapped boxes (empty when no "
					"boxes were connected)."),
				io.Int.Output(display_name="width",
					tooltip="Resolved crop width (the largest one in 'keep per "
					        "box' mode). 0 when no boxes were connected."),
				io.Int.Output(display_name="height",
					tooltip="Resolved crop height (the largest one in 'keep per "
					        "box' mode). 0 when no boxes were connected."),
			],
		)

	@classmethod
	def execute(cls, policy, bboxes=None, space=None) -> io.NodeOutput:
		space_hw = None if space is None else polytype.spatial_size(space)
		defaults = {"mode": boxpolicy.MODE_ONE_BATCH}
		boxes, (w, h) = boxpolicy.apply(bboxes, policy, space_hw, defaults)
		# parse() again so a typo in a hand-edited string is loud even when no
		# boxes are connected (this node is usually used as a pure policy source)
		boxpolicy.parse(policy, defaults)
		return io.NodeOutput(str(policy), _nest_bboxes(boxes), int(w), int(h))


# --- list-aware crop/paste --------------------------------------------------
#
# These three nodes declare is_input_list, so the executor hands EVERY socket
# over as a list - widgets included (execution.py: input_data_all[x] =
# [input_data]).  That is what lets ONE node cover "N boxes on one image",
# "one box per frame of a batch" and "several box sets at once" without a
# separate *List node per case: the policy names the use case instead.
#
# Their crops output is is_output_list, so "emit a batch" is a ONE-element
# output list (downstream runs once) and "emit one item per region" is an
# N-element one (downstream runs N times) - merge_result_data extends rather
# than nests, so both shapes travel through the same socket.

def _unwrap(value):
	"""First element of an is_input_list socket - widgets and single sockets.

	Defensive on purpose: it must behave the same whether the caller wrapped
	the value or not, because a direct execute() call in a test does not wrap.
	"""
	if isinstance(value, list) and len(value) == 1:
		return value[0]
	return value


def _flatten_frames(sources) -> list:
	"""A flat sequence of single-frame values from an is_input_list socket.

	Each element of the incoming list contributes its OWN batch frames, in
	order - so a list of batches and a single batch flatten the same way.
	Frames keep their own type (IMAGE / MASK / NPARRAY / LATENT).
	"""
	frames = []
	for source in (sources if isinstance(sources, list) else [sources]):
		if source is None:
			continue
		for i in range(polytype.batch_size(source)):
			frames.append(polytype.select_frame(source, i))
	return frames


def _pair_groups(groups, frames, cfg, what="region group"):
	"""Which frame each region group reads, per the policy's mode.

	'one source' modes always read frame 0. 'per-frame' modes read frame g,
	reusing a single group across every frame (the executor's own repeat-the-
	last-element convention) and otherwise demanding one group per frame.
	"""
	if not boxpolicy.per_frame(cfg):
		return groups, [0] * len(groups)
	if len(groups) == 1 and len(frames) > 1:
		groups = groups * len(frames)          # same boxes on every frame
	if len(groups) != len(frames):
		raise ValueError(
			f"The policy pairs each {what} with its own frame, but there "
			f"are {len(frames)} frame(s) and {len(groups)} {what}(s). Use a "
			f"'one source, ...' policy mode to crop every {what} out of the "
			f"first frame instead.")
	return groups, list(range(len(groups)))


def _crop_groups(frames, groups, cfg, tag):
	"""Crop every box of every group, then shape the output per the policy.

	Sizing is resolved ONCE over all boxes (so 'uniform' means the largest
	across the whole set, not per group) against frame 0's size, then split
	back into groups.
	"""
	groups, frame_of = _pair_groups(groups, frames, cfg)
	flat = [b for group in groups for b in group]
	if not flat:
		return [], [], (0, 0)

	space = polytype.spatial_size(frames[0])
	sized, (w, h) = boxpolicy.apply(flat, None, space, cfg)

	crops, out_boxes, i = [], [], 0
	for g, group in enumerate(groups):
		frame = frames[frame_of[g]]
		for _ in group:
			box = sized[i]
			i += 1
			crops.append(polytype.spatial_crop(
				frame, box["x"], box["y"], box["width"], box["height"]))
			out_boxes.append(box)

	if boxpolicy.stacks(cfg):
		first = polytype.spatial_size(crops[0])
		for n, crop in enumerate(crops[1:], start=1):
			this = polytype.spatial_size(crop)
			if this != first:
				raise ValueError(
					f"Crop {n} is {this[1]}x{this[0]} but crop 0 is "
					f"{first[1]}x{first[0]}, so they cannot stack into one "
					f"batch. Either switch the policy mode to '-> list' (one "
					f"output item per region, any size) or set its 'size' "
					f"field to 'uniform'.")
		return ([polytype.concat(crops, tag)],
		        [_nest_bboxes(out_boxes)], (w, h))
	# one output item per region: downstream executes once per crop, so the
	# boxes have to fan out in lockstep for 'Paste by BBox' to pair them
	return crops, [_nest_bboxes([b]) for b in out_boxes], (w, h)


class CropByBBoxes(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		template = polytype.match_template(
			"crop_type", polytype.with_latent(polytype.IMAGEISH))
		return io.Schema(
			node_id="CV_CropByBBoxes",
			display_name="CV Crop by BBoxes",
			category=CATEGORY,
			description="Crops one region per bounding box, in the SOURCE'S OWN "
			            "TYPE - IMAGE, MASK, NPARRAY or LATENT all work, and the "
			            "slice is exact (no uint8 round trip), so crop -> process "
			            "-> 'CV Paste by BBox' is lossless. The policy MODE "
			            "picks the use case: '-> batch' stacks every crop at one "
			            "common size (downstream runs once - the inpainting "
			            "case), '-> list' emits one item per region at its own "
			            "tight size (downstream runs once per region). The "
			            "'per-frame' modes pair region group g with frame g of a "
			            "batch instead of reading frame 0. Several BOUNDING_BOX "
			            "values wired in are simply concatenated - the type "
			            "already nests one group per frame, so there is no "
			            "separate 'box list' input. Boxes are in the source's own "
			            "coordinate space (use 'CV Scale BBoxes' for a "
			            "LATENT). Empty input yields an empty batch and never "
			            "raises.",
			inputs=[
				io.MatchType.Input("source", template=template,
					tooltip="What to crop. IMAGE / MASK / NPARRAY / LATENT; the "
					        "crops come back in the same type. A batch - or "
					        "several inputs - flattens into one frame sequence."),
				io.BoundingBox.Input("bboxes", force_input=True,
					tooltip=BBOX_TOOLTIP + " Boxes in the source's coordinate "
					"space. The per-frame nesting IS the grouping; several "
					"values are concatenated."),
				io.String.Input("policy",
					default=boxpolicy.compose(boxpolicy.MODE_ONE_BATCH),
					optional=True, advanced=True,
					tooltip=boxpolicy.MODE_TOOLTIP + " " + boxpolicy.POLICY_TOOLTIP
					        + " Boxes are always clamped inside the source.",
					extra_dict={"cv_policy": boxpolicy.spec(boxpolicy.MODE_ONE_BATCH)}),
			],
			outputs=[
				polytype.match_output(template, "crops", is_output_list=True,
					tooltip="One stacked batch (batch modes) or one item per "
					        "region (list modes), in the source's own type."),
				io.BoundingBox.Output(display_name="bboxes", is_output_list=True,
					tooltip=BBOX_TOOLTIP + " The SNAPPED boxes actually cropped, "
					"fanned out in lockstep with 'crops' - feed these to 'OpenCV "
					"Paste by BBox', not the input boxes."),
				io.Int.Output(display_name="width",
					tooltip="Crop width (the largest one when sizes vary)."),
				io.Int.Output(display_name="height",
					tooltip="Crop height (the largest one when sizes vary)."),
				io.Int.Output(display_name="count",
					tooltip="Number of crops (regions), whatever the output shape."),
			],
			is_input_list=True,
		)

	@classmethod
	def execute(cls, source, bboxes, policy="") -> io.NodeOutput:
		frames = _flatten_frames(source)
		groups = boxpolicy.flatten_groups(bboxes)
		cfg = boxpolicy.parse(_unwrap(policy), {"mode": boxpolicy.MODE_ONE_BATCH})
		if not frames:
			raise ValueError("Crop by BBoxes got no source frames.")
		tag = polytype.type_tag(frames[0])
		crops, out_boxes, (w, h) = _crop_groups(frames, groups, cfg, tag)
		if not crops:
			# no detections is a normal outcome - pass an empty batch forward so
			# crop -> process -> paste becomes a no-op instead of raising
			return io.NodeOutput([polytype.empty_batch(tag)],
				[_nest_bboxes([])], 0, 0, 0)
		return io.NodeOutput(crops, out_boxes, int(w), int(h), len(crops)
			if not boxpolicy.stacks(cfg) else polytype.batch_size(crops[0]))


class CropByMasks(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		template = polytype.match_template(
			"crop_type", polytype.with_latent(polytype.IMAGEISH))
		return io.Schema(
			node_id="CV_CropByMasks",
			display_name="CV Crop by Masks",
			category=CATEGORY,
			description="Crops around each mask of a MASK batch (e.g. from "
			            "'Connected Components', 'Labels to Masks' or 'K-Means "
			            "Mask Clusters') - the mask-driven shorthand for 'OpenCV "
			            "Masks to BBoxes' -> 'CV Crop by BBoxes', and it "
			            "takes the SAME policy, so one 'Snap BBoxes' can drive "
			            "both in lockstep. The policy MODE picks the use case: "
			            "'-> batch' stacks every crop at one common size so they "
			            "paste back losslessly (downstream runs once), '-> list' "
			            "emits one item per region at its own tight size "
			            "(downstream runs once per region). The source can be an "
			            "IMAGE, MASK, NPARRAY or LATENT and the crops come back "
			            "in the same type; masks are fitted to its size, so a "
			            "LATENT source works with pixel-space masks. Also returns "
			            "the matching cropped masks (for inpainting) and the crop "
			            "regions as bounding boxes.",
			inputs=[
				io.MatchType.Input("image", template=template,
					tooltip="What to crop. IMAGE / MASK / NPARRAY / LATENT; the "
					        "crops come back in the same type."),
				io.Mask.Input("masks",
					tooltip="One mask per region to crop (MASK batch). Several "
					        "batches wired in are concatenated."),
				io.Int.Input("padding", default=16, min=0, max=4096,
					tooltip="Extra context around each region in pixels - "
					        "inpainting works better with surroundings included."),
				io.Boolean.Input("square", default=False,
					tooltip="Force square crops (useful for diffusion inpainting)."),
				io.String.Input("policy",
					default=boxpolicy.compose(boxpolicy.MODE_ONE_BATCH),
					optional=True, advanced=True,
					tooltip=boxpolicy.MODE_TOOLTIP + " " + boxpolicy.POLICY_TOOLTIP
					        + " The padding/square widgets above are the baseline; "
					          "a policy overrides only the fields it sets.",
					extra_dict={"cv_policy": boxpolicy.spec(
						boxpolicy.MODE_ONE_BATCH, exclude=("padding", "square"))}),
			],
			outputs=[
				polytype.match_output(template, "crops", is_output_list=True,
					tooltip="One stacked batch (batch modes) or one item per "
					        "region (list modes), in the source's own type."),
				io.Mask.Output(display_name="crop_masks", is_output_list=True,
					tooltip="Each region's mask cropped to its crop, fanned out "
					        "in lockstep with 'crops'."),
				io.BoundingBox.Output(display_name="bboxes", is_output_list=True,
					tooltip=BBOX_TOOLTIP + " Fanned out in lockstep with 'crops'."),
				NPArray.Output(display_name="region_index",
					tooltip="(B,) int32: which row of the input MASK batch each "
					        "crop came from. Empty masks produce no crop, so this "
					        "is what realigns the crops with the input batch."),
			],
			is_input_list=True,
		)

	@classmethod
	def execute(cls, image, masks, padding, square, policy="") -> io.NodeOutput:
		frames = _flatten_frames(image)
		if not frames:
			raise ValueError("Crop by Masks got no source frames.")
		tag = polytype.type_tag(frames[0])
		ih, iw = polytype.spatial_size(frames[0])
		cfg = boxpolicy.parse(_unwrap(policy), {
			"mode": boxpolicy.MODE_ONE_BATCH,
			"padding": _unwrap(padding),
			"square": bool(_unwrap(square)),
		})

		mask_frames = []
		for batch in (masks if isinstance(masks, list) else [masks]):
			if batch is not None:
				mask_frames.extend(imageutils.mask_batch_to_u8(batch))

		tight, binaries, region_index = [], [], []
		for i, m in enumerate(mask_frames):
			m = imageutils.fit_mask_to(m, (ih, iw))
			binary = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)[1]
			if cv2.countNonZero(binary) == 0:
				continue
			tight.append(_bbox_dict(*cv2.boundingRect(binary)))
			binaries.append(binary)
			region_index.append(i)

		# one mask = one region; per-frame modes pair region i with frame i, so
		# each region is its own group
		groups = [[b] for b in tight]
		crops, out_boxes, _ = _crop_groups(frames, groups, cfg, tag) if tight \
			else ([], [], (0, 0))
		if not crops:
			# no masks / all empty: pass empty batches forward so downstream
			# crop -> process -> paste becomes a no-op instead of raising.
			return io.NodeOutput(
				[polytype.empty_batch(tag)],
				[imageutils.u8_to_mask_batch([])],
				[_nest_bboxes([])],
				np.zeros((0,), np.int32),
			)

		flat_boxes = boxpolicy.normalize(out_boxes)
		crop_masks = [
			b_img[b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"]].copy()
			for b_img, b in zip(binaries, flat_boxes)
		]
		masks_out = ([imageutils.u8_to_mask_batch(crop_masks)]
		             if boxpolicy.stacks(cfg)
		             else [imageutils.u8_to_mask_batch([m]) for m in crop_masks])
		return io.NodeOutput(crops, masks_out, out_boxes, np.int32(region_index))


class PasteByBBox(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		template = polytype.match_template(
			"paste_type", polytype.with_latent(polytype.IMAGEISH))
		return io.Schema(
			node_id="CV_PasteByBBox",
			display_name="CV Paste by BBox",
			category=CATEGORY,
			description="Pastes crops back at the given bounding boxes - the "
			            "inverse of 'CV Crop by Masks' / 'CV Crop by "
			            "BBoxes'. It needs NO policy: it flattens whatever the "
			            "crop node emitted - one stacked batch or one item per "
			            "region - and pairs crop i with box i, so both output "
			            "shapes come home the same way. Works in the target's own "
			            "type (IMAGE, MASK, NPARRAY or LATENT) and writes the "
			            "values through untouched, so an unmodified crop "
			            "round-trips exactly. Connect the crop_masks to paste only "
			            "the masked region (with optional feathering) instead of "
			            "the full rectangle.",
			inputs=[
				io.MatchType.Input("image", template=template,
					tooltip="What to paste into. Frame 0 is used; the crops must "
					        "be the same type."),
				io.MatchType.Input("crops", template=template,
					tooltip="The crop node's 'crops' output, in either shape."),
				io.BoundingBox.Input("bboxes", force_input=True,
					tooltip=BBOX_TOOLTIP + " The crop node's 'bboxes' output (the "
					"SNAPPED boxes) - one per crop, in the same order."),
				io.Mask.Input("masks", optional=True,
					tooltip="Optional crop-space masks (white = paste); the full "
					        "rectangle is pasted if omitted."),
				io.Float.Input("feather", default=0.0, min=0.0, max=100.0, step=0.5,
					tooltip="Gaussian blur sigma applied to the paste masks for "
					        "soft seams."),
			],
			outputs=[
				polytype.match_output(template, "image",
					tooltip="The target with every crop pasted in, in its own type."),
			],
			is_input_list=True,
		)

	@classmethod
	def execute(cls, image, crops, bboxes, masks=None, feather=0.0) -> io.NodeOutput:
		targets = _flatten_frames(image)
		if not targets:
			raise ValueError("Paste by BBox got no target image.")
		# frame 0, cloned: writes below must never reach an upstream node's
		# cached output (same rule as lowlevel.py's copy-before-cv2)
		dst = polytype.clone(targets[0])
		ih, iw = polytype.spatial_size(dst)
		feather = float(_unwrap(feather) or 0.0)

		patches = _flatten_frames(crops)
		boxes = [b for group in boxpolicy.flatten_groups(bboxes) for b in group]
		mask_frames = []
		for batch in (masks if isinstance(masks, list) else [masks]):
			if batch is not None:
				mask_frames.extend(imageutils.mask_batch_to_u8(batch))

		if len(boxes) != len(patches):
			raise ValueError(
				f"Got {len(patches)} crops but {len(boxes)} bounding boxes - "
				"connect crops and bboxes from the same crop node.")

		for i, bbox in enumerate(boxes):
			bx, by = int(bbox["x"]), int(bbox["y"])
			bw, bh = int(bbox["width"]), int(bbox["height"])
			x, y = max(0, bx), max(0, by)
			w, h = min(bx + bw, iw) - x, min(by + bh, ih) - y
			if w <= 0 or h <= 0:
				raise ValueError(f"Bounding box {i} {bbox} lies outside the image ({iw}x{ih}).")

			patch = patches[i]
			ph, pw = polytype.spatial_size(patch)
			if (ph, pw) == (bh, bw) and (h, w) != (bh, bw):
				# the box was clipped at the image edge - slice the crop to match
				# instead of resizing it (a resize would shift every pixel)
				patch = polytype.spatial_crop(patch, x - bx, y - by, w, h)
			elif (ph, pw) != (h, w):
				patch = polytype.spatial_resize(patch, w, h)

			alpha = None
			if mask_frames:
				m = imageutils.fit_mask_to(mask_frames[min(i, len(mask_frames) - 1)], (bh, bw))
				m = m[y - by:y - by + h, x - bx:x - bx + w]
				alpha = m.astype(np.float32) / 255.0
				if feather > 0:
					alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
			polytype.paste_into(dst, patch, x, y, alpha)

		return io.NodeOutput(dst)


class ScaleBBoxes(io.ComfyNode):
	_SPACE_TYPES = polytype.with_latent(polytype.IMAGEISH)

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ScaleBBoxes",
			display_name="CV Scale BBoxes",
			category=CATEGORY,
			description="Rescales bounding boxes from one image space to another "
			            "by comparing the sizes of two references - e.g. a bbox "
			            "found by 'Match Template' on a LATENT (latent-cell "
			            "coordinates) mapped onto the original image (pixel "
			            "coordinates) for the core 'Draw BBoxes' node, or boxes "
			            "detected on a downscaled copy mapped back to full "
			            "resolution. Only the references' height/width are read, "
			            "so any IMAGE, MASK, NPARRAY or LATENT works on either side.",
			inputs=[
				io.BoundingBox.Input("bboxes", force_input=True,
					tooltip=BBOX_TOOLTIP + " Boxes in from_space coordinates: a "
					"single dict, a flat list, or per-frame nested lists. Extra "
					"keys (score, label, ...) are preserved."),
				io.MultiType.Input("from_space", cls._SPACE_TYPES,
					tooltip="Reference the boxes were measured on (e.g. the LATENT "
					        "that was searched). Only its height/width are used - "
					        "for a LATENT that is the latent-cell grid size."),
				io.MultiType.Input("to_space", cls._SPACE_TYPES,
					tooltip="Reference to map the boxes onto (e.g. the original "
					        "IMAGE, or its MASK - only height/width are used)."),
			],
			outputs=[
				io.BoundingBox.Output(display_name="bboxes",
					tooltip=BBOX_TOOLTIP + " Same nesting as the input, "
					"coordinates in to_space."),
				io.Float.Output(display_name="scale_x",
					tooltip="Applied horizontal factor: to_space width / "
					        "from_space width (8.0 for a stride-8 VAE latent)."),
				io.Float.Output(display_name="scale_y",
					tooltip="Applied vertical factor: to_space height / "
					        "from_space height."),
			],
		)

	@classmethod
	def execute(cls, bboxes, from_space, to_space) -> io.NodeOutput:
		fh, fw = polytype.spatial_size(from_space)
		th, tw = polytype.spatial_size(to_space)
		if fw <= 0 or fh <= 0:
			raise ValueError(f"from_space has a degenerate size ({fw}x{fh}).")
		sx, sy = tw / fw, th / fh

		def scale_box(b: dict) -> dict:
			# scale the edges, not width/height, so adjacent boxes stay adjacent
			x1, y1 = int(round(b["x"] * sx)), int(round(b["y"] * sy))
			x2 = int(round((b["x"] + b["width"]) * sx))
			y2 = int(round((b["y"] + b["height"]) * sy))
			return {**b, "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

		def walk(item):
			if isinstance(item, dict):
				return scale_box(item)
			if isinstance(item, (list, tuple)):
				if len(item) == 4 and all(isinstance(v, (int, float)) for v in item):
					return scale_box(_bbox_dict(*item))
				return [walk(v) for v in item]
			raise ValueError(f"Cannot interpret {item!r} as a bounding box.")

		return io.NodeOutput(walk(bboxes), float(sx), float(sy))


# --- BOUNDING_BOX <-> NPARRAY interchange ----------------------------------
#
# BOUNDING_BOX is a list of DICTS, so the numbers inside it (coordinates, and
# the score/label keys every detector attaches) are unreachable to the array
# world: they cannot be sorted, thresholded, measured or drawn through
# points.py.  These two nodes are the bridge, and they are what makes every
# FOREIGN detector in the install (core RT-DETR, SAM3, MediaPipe,
# PrimitiveBoundingBox) usable from the OpenCV side.
#
# The layouts are named, never a magic flag, and they are the two conventions
# in the wild: OpenCV's (x, y, w, h) and the DNN world's corner (x1, y1, x2,
# y2).  BOUNDING_BOX itself is always {x, y, width, height}.

_BOX_LAYOUTS = ["x, y, width, height", "x1, y1, x2, y2",
                "center x, center y, width, height"]

_BOX_LAYOUT_TOOLTIP = (
	"Column layout of the array. 'x, y, width, height' is OpenCV's own "
	"convention (and BOUNDING_BOX's); 'x1, y1, x2, y2' is the corner form most "
	"DNN heads and torchvision use; 'center x, center y, width, height' is the "
	"YOLO form. Only the ARRAY side changes - BOUNDING_BOX is always "
	"{x, y, width, height}.")


def _boxes_to_rows(boxes: list, layout: str) -> np.ndarray:
	"""(N,4) float32 rows in `layout` from a flat list of bbox dicts."""
	rows = np.zeros((len(boxes), 4), np.float32)
	for i, b in enumerate(boxes):
		x, y = float(b.get("x", 0.0)), float(b.get("y", 0.0))
		w, h = float(b.get("width", 0.0)), float(b.get("height", 0.0))
		if layout == _BOX_LAYOUTS[1]:
			rows[i] = (x, y, x + w, y + h)
		elif layout == _BOX_LAYOUTS[2]:
			rows[i] = (x + w / 2.0, y + h / 2.0, w, h)
		else:
			rows[i] = (x, y, w, h)
	return rows


def _rows_to_xywh(rows: np.ndarray, layout: str) -> np.ndarray:
	"""The inverse of :func:`_boxes_to_rows`, always as (x, y, w, h)."""
	rows = np.asarray(rows, np.float64).reshape(-1, rows.shape[-1])[:, :4]
	if layout == _BOX_LAYOUTS[1]:
		x, y = rows[:, 0], rows[:, 1]
		return np.stack([x, y, rows[:, 2] - x, rows[:, 3] - y], axis=1)
	if layout == _BOX_LAYOUTS[2]:
		w, h = rows[:, 2], rows[:, 3]
		return np.stack([rows[:, 0] - w / 2.0, rows[:, 1] - h / 2.0, w, h], axis=1)
	return rows.copy()


def _label_table(boxes: list):
	"""(label_ids int32, newline-joined unique names) for a flat box list.

	Labels are STRINGS in the wild (RT-DETR emits COCO class names), so the
	array side gets indices and the names travel beside them as one STRING -
	the same shape 'Load Labels' produces, so the two interchange.
	"""
	names, ids = [], []
	index = {}
	for b in boxes:
		raw = b.get("label", b.get("class", b.get("class_id")))
		key = "" if raw is None else str(raw)
		if key not in index:
			index[key] = len(names)
			names.append(key)
		ids.append(index[key])
	return (np.asarray(ids, np.int32) if ids else np.zeros((0,), np.int32),
	        "\n".join(names))


class BBoxesToArray(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_BBoxesToArray",
			display_name="CV BBoxes To Array",
			category=CATEGORY,
			description="Reads a BOUNDING_BOX value as plain arrays - the numeric "
			            "bridge OUT of the detection world. Every detector emits "
			            "dicts, so their coordinates and their score/label keys "
			            "cannot be sorted, thresholded, measured or plotted; this "
			            "node exposes them as NPARRAY, which the whole of points.py, "
			            "contours.py, the charts and the raw cv2 wrappers accept. "
			            "Works on any emitter: this pack's own nodes, core 'RT-DETR' "
			            "/ SAM3 / MediaPipe / 'Bounding Box', hand-authored boxes. "
			            "'CV Array To BBoxes' is the return leg. Empty input yields "
			            "empty arrays and never raises.",
			inputs=[
				io.BoundingBox.Input("bboxes", force_input=True,
					tooltip=BBOX_TOOLTIP + " A single dict, a flat list or "
					"per-frame nested lists are all accepted; the frame each box "
					"came from is reported in 'group_index'."),
				io.Combo.Input("layout", options=_BOX_LAYOUTS, default=_BOX_LAYOUTS[0],
					optional=True, advanced=True, tooltip=_BOX_LAYOUT_TOOLTIP),
			],
			outputs=[
				NPArray.Output(display_name="boxes",
					tooltip="(N,4) float32, one row per box in the chosen layout, "
					        "in the order the boxes appear (frame by frame). Cast "
					        "it with 'CV Cast Array' if a consumer wants int32."),
				NPArray.Output(display_name="scores",
					tooltip="(N,) float32 confidence, from each box's 'score' key. "
					        "Boxes with no score read as 1.0, so a threshold never "
					        "silently drops un-scored boxes. float32 like every "
					        "other score in this pack, so a round trip through "
					        "'CV Array To BBoxes' keeps ~7 digits, not all 17."),
				NPArray.Output(display_name="label_ids",
					tooltip="(N,) int32 index into 'labels'. Boxes with no label "
					        "all share index 0. Feed it to 'CV NMS Boxes' for "
					        "per-class suppression."),
				io.String.Output(display_name="labels",
					tooltip="The distinct label names, one per line, indexed by "
					        "'label_ids' - the same shape 'Load Labels' produces."),
				NPArray.Output(display_name="group_index",
					tooltip="(N,) int32: which per-frame GROUP each row came from. "
					        "Pass it back to 'CV Array To BBoxes' to restore the "
					        "original nesting."),
				io.Int.Output(display_name="count",
					tooltip="Total number of boxes (rows)."),
			],
		)

	@classmethod
	def execute(cls, bboxes, layout=None) -> io.NodeOutput:
		layout = layout or _BOX_LAYOUTS[0]
		groups = boxpolicy.groups(bboxes)
		flat, group_index = [], []
		for g, group in enumerate(groups):
			for b in group:
				flat.append(b)
				group_index.append(g)
		scores = np.asarray([float(b.get("score", 1.0)) for b in flat],
		                    np.float32) if flat else np.zeros((0,), np.float32)
		label_ids, labels = _label_table(flat)
		return io.NodeOutput(
			_boxes_to_rows(flat, layout),
			scores,
			label_ids,
			labels,
			np.asarray(group_index, np.int32) if group_index else np.zeros((0,), np.int32),
			len(flat),
		)


class ArrayToBBoxes(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ArrayToBBoxes",
			display_name="CV Array To BBoxes",
			category=CATEGORY,
			description="Builds a BOUNDING_BOX value from plain arrays - the return "
			            "leg of 'CV BBoxes To Array', and the way any array of "
			            "regions (a decoded DNN head, k-means cluster extents, "
			            "boxes computed with 'Math Expression' or the raw cv2 "
			            "wrappers) becomes croppable and drawable. Scores and labels "
			            "ride along as extra keys, which every bbox node in this "
			            "pack preserves. Empty input yields an empty BOUNDING_BOX "
			            "and never raises.",
			inputs=[
				NPArray.Input("boxes",
					tooltip="(N,4) or (N,>=4) array of box rows in the chosen "
					        "layout; extra columns are ignored. A single (4,) row "
					        "is accepted as one box. Floats are rounded."),
				io.Combo.Input("layout", options=_BOX_LAYOUTS, default=_BOX_LAYOUTS[0],
					optional=True, advanced=True, tooltip=_BOX_LAYOUT_TOOLTIP),
				NPArray.Input("scores", optional=True, advanced=True,
					tooltip="Optional (N,) confidences, written to each box's "
					        "'score' key. Omit to leave the boxes un-scored."),
				NPArray.Input("label_ids", optional=True, advanced=True,
					tooltip="Optional (N,) int indices into 'labels'. Without "
					        "'labels' the index itself is written as the label."),
				io.String.Input("labels", optional=True, advanced=True,
					multiline=True, default="",
					tooltip="Optional label names, one per line, indexed by "
					        "'label_ids' - takes 'Load Labels' output directly."),
				NPArray.Input("group_index", optional=True, advanced=True,
					tooltip="Optional (N,) int frame index per row (as emitted by "
					        "'CV BBoxes To Array'), restoring the per-frame "
					        "nesting. Omitted, every box lands in one group."),
			],
			outputs=[
				io.BoundingBox.Output(display_name="bboxes", tooltip=BBOX_TOOLTIP),
				io.Int.Output(display_name="count",
					tooltip="Number of boxes emitted."),
			],
		)

	@classmethod
	def execute(cls, boxes, layout=None, scores=None, label_ids=None,
			labels=None, group_index=None) -> io.NodeOutput:
		layout = layout or _BOX_LAYOUTS[0]
		rows = np.asarray(boxes)
		if rows.size == 0:
			return io.NodeOutput(_nest_bboxes([]), 0)
		if rows.ndim == 1:
			rows = rows.reshape(1, -1)
		rows = rows.reshape(rows.shape[0], -1)
		if rows.shape[1] < 4:
			raise ValueError(
				f"'boxes' needs at least 4 columns for layout '{layout}', got "
				f"shape {tuple(np.asarray(boxes).shape)}.")
		xywh = _rows_to_xywh(rows, layout)
		n = xywh.shape[0]

		def column(value):
			if value is None:
				return None
			flat = np.asarray(value).reshape(-1)
			return flat if flat.size else None

		score_col, label_col = column(scores), column(label_ids)
		names = [ln for ln in (labels or "").splitlines() if ln != ""] \
			if labels else []
		groups_col = column(group_index)

		built = []
		for i in range(n):
			extra = {}
			if score_col is not None and i < score_col.size:
				extra["score"] = float(score_col[i])
			if label_col is not None and i < label_col.size:
				idx = int(label_col[i])
				extra["label"] = names[idx] if 0 <= idx < len(names) else str(idx)
			x, y, w, h = xywh[i]
			built.append(_bbox_dict(round(x), round(y), round(w), round(h), **extra))

		if groups_col is None:
			return io.NodeOutput(_nest_bboxes(built), n)
		nested = {}
		for i, b in enumerate(built):
			nested.setdefault(int(groups_col[i]) if i < groups_col.size else 0,
			                  []).append(b)
		return io.NodeOutput([nested.get(g, []) for g in
		                      range(max(nested) + 1)], n)


class NMSBoxes(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_NMSBoxes",
			display_name="CV NMS Boxes",
			category=CATEGORY,
			description="Non-maximum suppression over any BOUNDING_BOX value: drops "
			            "boxes that overlap a higher-scoring box by more than the "
			            "IoU threshold. cv2's NMS is otherwise only reachable inside "
			            "'YOLO Detect Decode', so anything that produces overlapping "
			            "regions - two detectors merged, a lowered confidence "
			            "threshold, 'Masks to BBoxes' on a noisy segmentation, "
			            "'Match Template Multi-Scale' across scales - has no way to "
			            "de-duplicate. Suppression runs INSIDE each per-frame group, "
			            "never across frames. Empty input yields empty outputs "
			            "and never raises.",
			inputs=[
				io.BoundingBox.Input("bboxes", force_input=True,
					tooltip=BBOX_TOOLTIP + " Boxes are suppressed within each "
					"per-frame group independently."),
				io.Float.Input("nms_threshold", default=0.45, min=0.0, max=1.0,
					step=0.01,
					tooltip="Maximum IoU a kept box may have with a higher-scoring "
					        "one. Lower suppresses more aggressively; 1.0 keeps "
					        "everything. Measure the actual overlaps with 'CV Box "
					        "IoU Matrix' if you are unsure what to set."),
				io.Float.Input("score_threshold", default=0.0, min=0.0, max=1.0,
					step=0.01,
					tooltip="Boxes scoring below this are dropped before "
					        "suppression. 0.0 keeps them all."),
				NPArray.Input("scores", optional=True,
					tooltip="Optional (N,) confidences, in the flattened box order "
					        "('CV BBoxes To Array' emits exactly this). Omitted, "
					        "each box's own 'score' key is used, defaulting to 1.0 "
					        "- with no scores at all NMS keeps the first box of "
					        "each overlapping cluster."),
				NPArray.Input("class_ids", optional=True,
					tooltip="Optional (N,) class indices. WIRED, suppression runs "
					        "per class (boxes of different classes never suppress "
					        "each other) - that is the switch, there is no boolean "
					        "widget. 'CV BBoxes To Array' emits 'label_ids' for "
					        "this."),
				io.Int.Input("top_k", default=0, min=0, max=100000,
					optional=True, advanced=True,
					tooltip="Keep at most this many boxes per group after "
					        "suppression (0 = no limit), highest score first."),
				io.Float.Input("eta", default=1.0, min=0.01, max=10.0, step=0.01,
					optional=True, advanced=True,
					tooltip="Adaptive-threshold coefficient of cv2's NMS: the IoU "
					        "threshold is multiplied by this after each iteration. "
					        "1.0 (the default) keeps it constant."),
			],
			outputs=[
				io.BoundingBox.Output(display_name="bboxes",
					tooltip=BBOX_TOOLTIP + " The survivors, keeping the original "
					"per-frame nesting and every extra key."),
				NPArray.Output(display_name="kept_indices",
					tooltip="(K,) int32 indices of the survivors into the FLATTENED "
					        "input order - feed it to 'CV Take By Index' to keep any "
					        "parallel array (masks, embeddings, areas) in step."),
				io.Int.Output(display_name="count",
					tooltip="Number of boxes kept."),
				io.Int.Output(display_name="removed",
					tooltip="How many boxes were dropped (suppressed or below "
					        "'score_threshold'). 0 means NMS changed nothing."),
			],
		)

	@classmethod
	def execute(cls, bboxes, nms_threshold=0.45, score_threshold=0.0, scores=None,
			class_ids=None, top_k=0, eta=1.0) -> io.NodeOutput:
		groups = boxpolicy.groups(bboxes)
		score_col = None if scores is None else np.asarray(scores).reshape(-1)
		class_col = None if class_ids is None else np.asarray(class_ids).reshape(-1)
		if class_col is not None and class_col.size == 0:
			class_col = None

		out_groups, kept_indices, offset, dropped = [], [], 0, 0
		for group in groups:
			n = len(group)
			rects = [[float(b.get("x", 0.0)), float(b.get("y", 0.0)),
			          float(b.get("width", 0.0)), float(b.get("height", 0.0))]
			         for b in group]
			conf = []
			for i, b in enumerate(group):
				gi = offset + i
				if score_col is not None and gi < score_col.size:
					conf.append(float(score_col[gi]))
				else:
					conf.append(float(b.get("score", 1.0)))
			if n == 0:
				out_groups.append([])
				continue
			try:
				if class_col is not None:
					cls_ids = [int(class_col[offset + i])
					           if offset + i < class_col.size else 0
					           for i in range(n)]
					keep = cv2.dnn.NMSBoxesBatched(
						rects, conf, cls_ids, float(score_threshold),
						float(nms_threshold), float(eta),
						int(top_k) if top_k else 0)
				else:
					keep = cv2.dnn.NMSBoxes(
						rects, conf, float(score_threshold), float(nms_threshold),
						float(eta), int(top_k) if top_k else 0)
			except cv2.error as exc:  # a degenerate box must not halt a graph
				_log.warning("NMS failed on a group of %d boxes (%s); keeping them "
				             "all.", n, exc)
				keep = list(range(n))
			keep = [int(i) for i in np.asarray(keep).reshape(-1)] \
				if np.asarray(keep).size else []
			out_groups.append([group[i] for i in keep])
			kept_indices.extend(offset + i for i in keep)
			dropped += n - len(keep)
			offset += n

		return io.NodeOutput(
			out_groups if out_groups else _nest_bboxes([]),
			np.asarray(kept_indices, np.int32) if kept_indices
				else np.zeros((0,), np.int32),
			len(kept_indices),
			dropped,
		)


class BoxIoUMatrix(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_BoxIoUMatrix",
			display_name="CV Box IoU Matrix",
			category=CATEGORY,
			description="Intersection-over-union of every box in A against every box "
			            "in B, as an (N,M) array. The generic primitive behind "
			            "detection scoring against ground truth, frame-to-frame "
			            "track association, de-duplicating two detectors, and "
			            "'drop anything overlapping this keep-out region' - the "
			            "'CV Polygon IoU (convex)' subgraph only does a single pair. "
			            "Boxes are flattened across frames (use 'CV BBoxes To Array' "
			            "if you need to know which frame a row came from). Empty "
			            "input yields an empty matrix and never raises.",
			inputs=[
				io.BoundingBox.Input("bboxes_a", force_input=True,
					tooltip=BBOX_TOOLTIP + " Becomes the ROWS of the matrix."),
				io.BoundingBox.Input("bboxes_b", force_input=True,
					tooltip=BBOX_TOOLTIP + " Becomes the COLUMNS. Pass the same "
					"value on both sides to get the self-overlap matrix (its "
					"diagonal is 1.0)."),
			],
			outputs=[
				NPArray.Output(display_name="iou",
					tooltip="(N,M) float32 in [0,1]: intersection area / union area "
					        "for each pair. 0.0 where the boxes do not touch."),
				NPArray.Output(display_name="best_index",
					tooltip="(N,) int32: for each A box, the B box it overlaps most "
					        "(-1 when it overlaps nothing at all)."),
				NPArray.Output(display_name="best_iou",
					tooltip="(N,) float32: that best overlap. Threshold it to get "
					        "the classic matched/unmatched split."),
				io.Int.Output(display_name="count_a",
					tooltip="Number of A boxes (rows)."),
				io.Int.Output(display_name="count_b",
					tooltip="Number of B boxes (columns)."),
			],
		)

	@classmethod
	def execute(cls, bboxes_a, bboxes_b) -> io.NodeOutput:
		a = _boxes_to_rows(_normalize_bboxes(bboxes_a), _BOX_LAYOUTS[0])
		b = _boxes_to_rows(_normalize_bboxes(bboxes_b), _BOX_LAYOUTS[0])
		n, m = a.shape[0], b.shape[0]
		if n == 0 or m == 0:
			return io.NodeOutput(
				np.zeros((n, m), np.float32),
				np.full((n,), -1, np.int32),
				np.zeros((n,), np.float32), n, m)

		ax1, ay1 = a[:, 0:1].astype(np.float64), a[:, 1:2].astype(np.float64)
		ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
		bx1, by1 = b[:, 0].astype(np.float64), b[:, 1].astype(np.float64)
		bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]

		iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
		ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
		inter = iw * ih
		area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
		area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)
		union = area_a + area_b - inter
		# a zero-area box has no union with anything: report 0, never a NaN
		iou = np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)
		iou = iou.astype(np.float32)

		best = np.argmax(iou, axis=1).astype(np.int32)
		best_iou = iou[np.arange(n), best].astype(np.float32)
		best[best_iou <= 0.0] = -1
		return io.NodeOutput(iou, best, best_iou, n, m)


class OverlayMasks(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_OverlayMasks",
			display_name="CV Overlay Masks",
			category=CATEGORY,
			description="Visualizes a MASK batch by coloring each mask differently "
			            "over the image (or over black). Debug companion for "
			            "'Connected Components', the 'CV K-Means Mask Clusters' "
			            "blueprint, 'Crop by "
			            "Masks' etc.; use the core 'Draw BBoxes' node to "
			            "visualize bounding boxes.",
			inputs=[
				io.Mask.Input("masks", tooltip="MASK batch; each mask gets its own color."),
				io.Image.Input("image", optional=True,
					tooltip="Background to draw on; black if omitted."),
				io.Float.Input("opacity", default=0.6, min=0.0, max=1.0, step=0.05,
					tooltip="Color opacity over the background."),
			],
			outputs=[io.Image.Output(display_name="overlay")],
		)

	@classmethod
	def execute(cls, masks, image=None, opacity=0.6) -> io.NodeOutput:
		mask_frames = imageutils.mask_batch_to_u8(masks)
		if image is not None:
			bg = imageutils.image_batch_to_bgr(image)[0]
			h, w = bg.shape[:2]
		elif mask_frames:
			h, w = mask_frames[0].shape[:2]
			bg = np.zeros((h, w, 3), np.uint8)
		else:
			# no masks and no background image - nothing to draw
			return io.NodeOutput(imageutils.bgr_to_image_batch([]))

		overlay = bg.copy()
		for i, m in enumerate(mask_frames):
			color = _cluster_color(i, len(mask_frames))
			region = imageutils.fit_mask_to(m, (h, w)) > 127
			overlay[region] = ((1.0 - opacity) * bg[region] + opacity * color).astype(np.uint8)
		return io.NodeOutput(imageutils.bgr_to_image_batch([overlay]))


# ---------------------------------------------------------------------------
# Chromatic aberration correction
# ---------------------------------------------------------------------------

_CALIB_SUBDIR = "calib"
_CALIB_EXTENSIONS = {".yaml", ".xml"}


def _list_calib_files():
	""":yaml/.xml files from input/calib/ subfolder."""
	import folder_paths
	input_dir = folder_paths.get_input_directory()
	calib_dir = os.path.join(input_dir, _CALIB_SUBDIR)
	if not os.path.isdir(calib_dir):
		return []
	return sorted(
		f for f in os.listdir(calib_dir)
		if os.path.splitext(f)[1].lower() in _CALIB_EXTENSIONS
		and os.path.isfile(os.path.join(calib_dir, f))
	)


def _resolve_calib_file(name):
	"""Find a picked name in input/calib/."""
	import folder_paths
	input_dir = folder_paths.get_input_directory()
	path = os.path.join(input_dir, _CALIB_SUBDIR, name)
	if os.path.isfile(path):
		return path
	return None


def _build_ca_displacement_maps(coeff_mat, calib_size, degree, strength=1.0,
                                 img_size=None):
	"""Build per-channel remap grids from polynomial CA coefficients.

	Returns (map_x_r, map_y_r, map_x_b, map_y_b) — green is the reference
	channel (identity map).  The displacement is the *forward* (apply) direction,
	i.e. source pixel = output pixel + polynomial_displacement, which is the
	inverse of what ``correctChromaticAberration`` uses internally.

	*img_size* (w, h) overrides the grid dimensions when the image size
	differs from *calib_size*; the polynomial is still normalised to
	*calib_size*.
	"""
	cw, ch = calib_size
	if img_size is not None:
		w, h = img_size
	else:
		w, h = cw, ch
	grid_x, grid_y = np.meshgrid(
		np.arange(w, dtype=np.float32),
		np.arange(h, dtype=np.float32),
	)
	x_norm = (grid_x - cw * 0.5) / (cw * 0.5)
	y_norm = (grid_y - ch * 0.5) / (ch * 0.5)

	# Polynomial basis: for each total degree d, enumerate y^j * x^i
	# with j+i == d, j decreasing (y-first ordering matching OpenCV).
	basis = []
	for d in range(degree + 1):
		for j in range(d, -1, -1):
			i = d - j
			basis.append((y_norm ** j) * (x_norm ** i))
	n_terms = min(len(basis), coeff_mat.shape[1])

	def _eval(row_idx):
		out = np.zeros_like(grid_x)
		for k in range(n_terms):
			c = coeff_mat[row_idx, k]
			if c != 0.0:
				out = out + c * basis[k]
		return out * strength

	red_dx = _eval(2)
	red_dy = _eval(3)
	blue_dx = _eval(0)
	blue_dy = _eval(1)

	return (grid_x + red_dx, grid_y + red_dy,
	        grid_x + blue_dx, grid_y + blue_dy)


class ApplyChromaticAberration(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ApplyChromaticAberration",
			display_name="CV Apply Chromatic Aberration",
			category=CATEGORY,
			description="Simulates lateral chromatic aberration by displacing the "
			            "red and blue channels according to a polynomial model "
			            "loaded from a calibration file (the same file used by "
			            "'CV Chromatic Aberration Correction').  Use this "
			            "node upstream of the correction node to build a "
			            "round-trip demo.  Green stays fixed as the reference "
			            "channel.",
			inputs=[
				io.Image.Input("image",
					tooltip="Input image. A batch is processed frame by frame."),
				io.Combo.Input("calibration_file",
					options=[""] + _list_calib_files(),
					tooltip="Calibration file from input/calib/ (.yaml/.xml) "
					        "that defines the polynomial displacement. "
					        "Run 'Create CA Calibration' first, then pick "
					        "the same filename here."),
				io.Float.Input("strength", default=1.0, min=0.0, max=100.0,
					step=0.05,
					tooltip="Multiplier for all polynomial coefficients.  "
					        "1.0 = exact displacement from the calibration; "
					        "0.0 = identity (no CA); 2.0 = double strength."),
			],
			outputs=[io.Image.Output()],
		)

	@classmethod
	def execute(cls, image, calibration_file, strength=1.0) -> io.NodeOutput:
		if not calibration_file:
			raise ValueError("No calibration file selected.  Drop a .yaml/.xml "
			                 "file in input/calib/ and pick it from the "
			                 "dropdown.")
		path = _resolve_calib_file(calibration_file)
		if path is None:
			raise ValueError(f"Calibration file not found: {calibration_file!r} "
			                 f"(expected in input/{_CALIB_SUBDIR}/)")

		def fn(bgr):
			fs = cv2.FileStorage(path, cv2.FileStorage_READ)
			if not fs.isOpened():
				raise ValueError(f"Could not open calibration file: {path}")
			try:
				coeff_mat, calib_size, degree = \
					cv2.loadChromaticAberrationParams(fs.root())
			finally:
				fs.release()

			cw, ch = calib_size
			orig_h, orig_w = bgr.shape[:2]
			need_resize = (orig_w, orig_h) != (cw, ch)
			if need_resize:
				_log.warning("ApplyChromaticAberration: image %dx%d differs "
				             "from calibration %dx%d — resizing to calibration "
				             "size for consistent displacement, then back.  "
				             "Use a calibration matching your image size to "
				             "avoid this.",
				             orig_w, orig_h, cw, ch)
				bgr = cv2.resize(bgr, (cw, ch),
				                 interpolation=cv2.INTER_LINEAR)

			map_x_r, map_y_r, map_x_b, map_y_b = \
				_build_ca_displacement_maps(coeff_mat, calib_size, degree,
				                            strength,
				                            img_size=(bgr.shape[1],
				                                      bgr.shape[0]))
			result = bgr.copy()
			# BGR: ch 2 = red, ch 0 = blue
			result[:, :, 2] = cv2.remap(
				bgr[:, :, 2], map_x_r, map_y_r, cv2.INTER_LINEAR,
				borderMode=cv2.BORDER_REFLECT_101)
			result[:, :, 0] = cv2.remap(
				bgr[:, :, 0], map_x_b, map_y_b, cv2.INTER_LINEAR,
				borderMode=cv2.BORDER_REFLECT_101)

			if need_resize:
				result = cv2.resize(result, (orig_w, orig_h),
				                    interpolation=cv2.INTER_LINEAR)
			return result

		return io.NodeOutput(_map_frames(image, fn))


class CreateCACalibration(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CreateCACalibration",
			display_name="Create CA Calibration",
			category=CATEGORY,
			description="Generates a synthetic chromatic-aberration calibration "
			            "YAML file in input/calib/ with uniform (constant) "
			            "displacement for the red and blue channels.  The file "
			            "can be consumed by both 'Apply Chromatic Aberration' "
			            "(to simulate CA) and 'Chromatic Aberration Correction' "
			            "(to correct it).  The node writes to "
			            "input/calib/<filename> and shows a preview; pick the "
			            "same filename in the downstream node's calibration "
			            "dropdown.",
			inputs=[
				io.Float.Input("red_dx", default=0.0, min=-500.0, max=500.0,
					step=0.5,
					tooltip="Horizontal displacement of the red channel in "
					        "pixels (positive = red shifts right)."),
				io.Float.Input("red_dy", default=0.0, min=-500.0, max=500.0,
					step=0.5,
					tooltip="Vertical displacement of the red channel in "
					        "pixels (positive = red shifts down)."),
				io.Float.Input("blue_dx", default=0.0, min=-500.0, max=500.0,
					step=0.5,
					tooltip="Horizontal displacement of the blue channel in "
					        "pixels (positive = blue shifts right)."),
				io.Float.Input("blue_dy", default=0.0, min=-500.0, max=500.0,
					step=0.5,
					tooltip="Vertical displacement of the blue channel in "
					        "pixels (positive = blue shifts down)."),
				io.Int.Input("calib_width", default=512, min=1, max=100000,
					tooltip="Calibration image width in pixels (used to "
					        "normalise the polynomial coordinate system)."),
				io.Int.Input("calib_height", default=512, min=1, max=100000,
					tooltip="Calibration image height in pixels."),
				io.String.Input("filename", default="ca_synth.yaml",
					tooltip="Output filename. Saved to input/calib/. "
					        "Overwrites if it already exists."),
			],
			outputs=[],
			is_output_node=True,
		)

	@classmethod
	def execute(cls, red_dx, red_dy, blue_dx, blue_dy,
	            calib_width, calib_height,
	            filename) -> io.NodeOutput:
		import folder_paths

		name = os.path.basename(str(filename)).strip() or "ca_synth.yaml"
		input_dir = folder_paths.get_input_directory()
		calib_dir = os.path.join(input_dir, _CALIB_SUBDIR)
		os.makedirs(calib_dir, exist_ok=True)
		path = os.path.join(calib_dir, name)

		# degree 1 = 3 coefficients: [1, y, x]
		fs = cv2.FileStorage(path, cv2.FileStorage_WRITE)
		fs.write("image_width", int(calib_width))
		fs.write("image_height", int(calib_height))
		# YAML blue_channel -> coeff_mat rows 0-1 -> BLUE displacement
		fs.startWriteStruct("blue_channel", cv2.FILE_NODE_MAP)
		fs.startWriteStruct("coeffs_x", cv2.FILE_NODE_SEQ | cv2.FILE_NODE_EMPTY)
		for v in [float(blue_dx), 0.0, 0.0]:
			fs.write("", v)
		fs.endWriteStruct()
		fs.startWriteStruct("coeffs_y", cv2.FILE_NODE_SEQ | cv2.FILE_NODE_EMPTY)
		for v in [float(blue_dy), 0.0, 0.0]:
			fs.write("", v)
		fs.endWriteStruct()
		fs.write("rms", 0.0)
		fs.endWriteStruct()
		# YAML red_channel -> coeff_mat rows 2-3 -> RED displacement
		fs.startWriteStruct("red_channel", cv2.FILE_NODE_MAP)
		fs.startWriteStruct("coeffs_x", cv2.FILE_NODE_SEQ | cv2.FILE_NODE_EMPTY)
		for v in [float(red_dx), 0.0, 0.0]:
			fs.write("", v)
		fs.endWriteStruct()
		fs.startWriteStruct("coeffs_y", cv2.FILE_NODE_SEQ | cv2.FILE_NODE_EMPTY)
		for v in [float(red_dy), 0.0, 0.0]:
			fs.write("", v)
		fs.endWriteStruct()
		fs.write("rms", 0.0)
		fs.endWriteStruct()
		fs.release()

		preview = (f"Saved CA calibration -> {path}\n"
		           f"red: ({red_dx:+.1f}, {red_dy:+.1f})  "
		           f"blue: ({blue_dx:+.1f}, {blue_dy:+.1f})")
		return io.NodeOutput(
			ui=ui.PreviewText(preview),
		)


class ChromaticAberrationCorrection(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ChromaticAberrationCorrection",
			display_name="CV Chromatic Aberration Correction",
			category=CATEGORY,
			description="Corrects lateral chromatic aberration using a polynomial "
			            "distortion model loaded from a calibration file "
			            "(cv2.correctChromaticAberration). The calibration is "
			            "camera-specific: photograph a pattern of black discs on "
			            "white background, run apps/chromatic-aberration-calibration/"
			            "ca_calibration.py to produce a .yaml/.xml file, drop it in "
			            "input/calib/, and pick it here. For 3-channel images the "
			            "correction is applied directly; for single-channel raw "
			            "Bayer images, demosaicing is done first using the selected "
			            "Bayer pattern.",
			inputs=[
				io.Image.Input("image",
					tooltip="Input image to correct. A batch is processed "
					        "frame by frame."),
				io.Combo.Input("calibration_file",
					options=[""] + _list_calib_files(),
					tooltip="Calibration file from input/calib/ (.yaml/.xml). "
					        "Refresh node definitions (reload the page) to pick "
					        "up files added during the same session."),
				io.Combo.Input("bayer_pattern",
					options=enums.BAYER_CA.options,
					default=enums.BAYER_CA.default,
					tooltip="Bayer CFA layout for single-channel raw input. "
					        "Ignored when the image already has 3 channels.",
					optional=True, advanced=True),
			],
			outputs=[io.Image.Output()],
		)

	@classmethod
	def execute(cls, image, calibration_file, bayer_pattern="COLOR_BayerRGGB2BGR") -> io.NodeOutput:
		if not calibration_file:
			raise ValueError("No calibration file selected. Drop a .yaml/.xml "
			                 "file in input/calib/ and pick it from the dropdown.")

		path = _resolve_calib_file(calibration_file)
		if path is None:
			raise ValueError(f"Calibration file not found: {calibration_file!r} "
			                 f"(expected in input/{_CALIB_SUBDIR}/)")

		def fn(bgr):
			fs = cv2.FileStorage(path, cv2.FileStorage_READ)
			if not fs.isOpened():
				raise ValueError(f"Could not open calibration file: {path}")
			try:
				coeff_mat, calib_size, degree = cv2.loadChromaticAberrationParams(
					fs.root())
			finally:
				fs.release()

			# For single-channel raw input, let cv2.correctChromaticAberration
			# handle demosaicing internally via its bayer_pattern parameter.
			# For 3-channel input the parameter is ignored by cv2.
			bayer_code = getattr(cv2, bayer_pattern, cv2.COLOR_BayerRGGB2BGR) \
				if bayer_pattern else -1

			cw, ch = calib_size
			orig_h, orig_w = bgr.shape[:2]
			need_resize = (orig_w, orig_h) != (cw, ch)
			if need_resize:
				bgr = cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_LINEAR)

			corrected = cv2.correctChromaticAberration(
				bgr, coeff_mat, calib_size, degree, bayer_pattern=bayer_code)
			if corrected is None:
				raise ValueError("cv2.correctChromaticAberration returned None")

			if need_resize:
				corrected = cv2.resize(corrected, (orig_w, orig_h),
				                       interpolation=cv2.INTER_LINEAR)
			return corrected

		return io.NodeOutput(_map_frames(image, fn))


_LABEL_SORTS = {
	"by label value": lambda r: r[0],
	"by area (largest first)": lambda r: -r[2],
}


def _label_regions(labels):
	"""[(label, (x, y, w, h), area)] for every distinct label of a label map."""
	arr = np.asarray(labels)
	if arr.ndim == 3:  # batch of maps: use frame 0, like other high-level nodes
		arr = arr[0]
	regions = []
	for label in np.unique(arr):
		pts = np.argwhere(arr == label)
		x0, y0 = int(pts[:, 1].min()), int(pts[:, 0].min())
		regions.append((int(label),
		                (x0, y0, int(pts[:, 1].max()) - x0 + 1,
		                 int(pts[:, 0].max()) - y0 + 1),
		                int(len(pts))))
	return regions


TPL_PHOTOMETRIC = polytype.match_template("photometric_type")


def _trimmed_gain_bias(x, y, good, model, keep, iterations):
	"""Trimmed least-squares fit of y ~ gain * x + bias over `good` pixels.

	Returns (gain, bias, inlier_bool, residual) or None when the data cannot
	support a fit. The first inlier set is chosen by the RAW difference |y - x|
	rather than by a fit over everything: an exposure/white-balance drift is
	small and a subject is not, so the closest pixels are already the ones that
	show the same content. Starting from a plain all-pixel least squares
	instead lets a large translucent region (which is a competing linear model
	of its own) drag the fit away and the trimming then converges onto it.
	"""
	if good.sum() < 32:
		return None
	raw = np.abs(y - x)
	sub = good & (raw <= np.quantile(raw[good], keep))
	gain, bias = 1.0, 0.0
	for _ in range(int(iterations)):
		if sub.sum() < 32:
			return None
		xi, yi = x[sub], y[sub]
		if model == "bias only":
			gain, bias = 1.0, float(np.mean(yi - xi))
		elif model == "gain only":
			denom = float(np.dot(xi, xi))
			if denom <= 1e-12:
				return None
			gain, bias = float(np.dot(xi, yi) / denom), 0.0
		else:
			if float(np.var(xi)) <= 1e-9:
				return None
			a = np.stack([xi, np.ones_like(xi)], 1)
			sol, *_ = np.linalg.lstsq(a, yi, rcond=None)
			gain, bias = float(sol[0]), float(sol[1])
		if not (math.isfinite(gain) and math.isfinite(bias)) or not 0.01 <= gain <= 100.0:
			return None
		res = np.abs(y - (gain * x + bias))
		sub = good & (res <= np.quantile(res[good], keep))
	if sub.sum() < 32:
		return None
	residual = float(np.median(np.abs(y[sub] - (gain * x[sub] + bias))))
	return gain, bias, sub, residual


class PhotometricAlign(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PhotometricAlign",
			display_name="CV Photometric Align (Gain/Bias)",
			category=CATEGORY,
			description="Matches one image's exposure and white balance to "
			            "another's by fitting a per-channel gain and bias "
			            "(source * gain + bias ~ reference) - the photometric "
			            "counterpart of a geometric registration. Use it when "
			            "two shots of the same scene were taken minutes (or "
			            "one auto-exposure decision) apart: a clean background "
			            "plate against the take, a burst frame against its "
			            "neighbour, a flat field against the shot it corrects. "
			            "The fit is ROBUST: pixels whose content changed - a "
			            "subject that was not there in the plate, a moving "
			            "object, a specular flash - are outliers and are "
			            "trimmed away, so the gain describes the LIGHT, not "
			            "the difference. Saturated pixels are dropped too "
			            "(they carry no gain information). The pixels the fit "
			            "kept come back as 'inlier_mask', which is a ready "
			            "made 'what is unchanged here' mask. Fails soft (rule "
			            "of the pack): a featureless or degenerate pair "
			            "returns found=false with gain 1 / bias 0 and the "
			            "source untouched.",
			inputs=[
				polytype.match_input("source", TPL_PHOTOMETRIC,
					tooltip="Image to correct - its exposure is changed to "
					        "match the reference. Echoed back in the same "
					        "type (IMAGE/MASK/NPARRAY)."),
				polytype.poly_input("reference",
					tooltip="Image whose exposure/white balance is the "
					        "target. Same size and channel count as the "
					        "source; it is never modified."),
				io.Combo.Input("model",
					options=["gain + bias", "gain only", "bias only"],
					default="gain + bias",
					tooltip="What to fit. 'gain + bias' handles an exposure "
					        "change plus a black-level/flare shift (the usual "
					        "case). 'gain only' is the pure exposure/ISO "
					        "model. 'bias only' is an additive offset, e.g. "
					        "stray light added to one shot."),
				io.Combo.Input("scope",
					options=["per channel (also fixes white balance)",
					         "all channels together (exposure only)"],
					default="per channel (also fixes white balance)",
					tooltip="Per channel gives each of B, G, R its own gain "
					        "and bias, which also corrects a white-balance "
					        "drift. One shared fit is safer when the images "
					        "have few unchanged pixels, because all channels "
					        "then pool their evidence."),
				io.Float.Input("inlier_fraction", default=0.5, min=0.05, max=1.0, step=0.05,
					tooltip="Fraction of pixels kept as inliers at each "
					        "refit. Set it BELOW the fraction of the frame "
					        "that is genuinely unchanged (0.5 tolerates a "
					        "subject covering half the frame). 1.0 disables "
					        "the trimming and gives a plain least squares."),
				io.Int.Input("iterations", default=3, min=1, max=20,
					tooltip="Refits after the initial difference-based inlier "
					        "selection. 3 is enough for a photometric drift; "
					        "more only matters when the first selection was "
					        "poor."),
				polytype.poly_input("mask", types=polytype.MASK_ONLY, optional=True,
					tooltip="Optional region to fit ON (non-zero pixels), for "
					        "when you already know where the unchanged "
					        "background is. The trimming still runs inside "
					        "it."),
			],
			outputs=[
				polytype.match_output(TPL_PHOTOMETRIC, "aligned",
					tooltip="The source with gain/bias applied, in the "
					        "source's own type and dtype (integer types are "
					        "rounded and clipped). Identical to the input "
					        "when found=false."),
				NPArray.Output(display_name="gain",
					tooltip="float64 (C,) multiplier per channel, 1.0 where "
					        "nothing was fitted."),
				NPArray.Output(display_name="bias",
					tooltip="float64 (C,) offset per channel, in the input's "
					        "own units (0-255 for uint8), 0.0 where nothing "
					        "was fitted."),
				NPArray.Output(display_name="inlier_mask",
					tooltip="uint8 [H,W] 255 where the pixel was an inlier in "
					        "EVERY channel, i.e. where the two images show the "
					        "same thing under the fitted lighting. All zeros "
					        "when found=false."),
				io.Float.Output(display_name="residual",
					tooltip="Median absolute residual over the inliers after "
					        "the fit, in input units - how well the two images "
					        "agree once the light is accounted for."),
				io.Boolean.Output(display_name="found"),
			],
		)

	@classmethod
	def execute(cls, source, reference, model, scope, inlier_fraction, iterations,
	            mask=None) -> io.NodeOutput:
		src, tag = polytype.resolve(source)
		ref, _ = polytype.resolve(reference)
		src = np.asarray(src)
		ref = np.asarray(ref)
		if src.shape[:2] != ref.shape[:2]:
			raise ValueError(
				f"source is {src.shape[0]}x{src.shape[1]} but reference is "
				f"{ref.shape[0]}x{ref.shape[1]} - align the sizes first "
				f"(e.g. cv2.resize).")
		s3 = src if src.ndim == 3 else src[:, :, None]
		r3 = ref if ref.ndim == 3 else ref[:, :, None]
		if s3.shape[2] != r3.shape[2]:
			raise ValueError(
				f"source has {s3.shape[2]} channel(s) and reference "
				f"{r3.shape[2]} - convert one of them (e.g. cv2.cvtColor).")

		x_all = s3.astype(np.float64)
		y_all = r3.astype(np.float64)
		channels = s3.shape[2]
		good = np.ones(s3.shape[:2], bool)
		if mask is not None:
			m, _ = polytype.resolve(mask)
			good &= imageutils.fit_mask_to(
				imageutils.ensure_uint8(np.asarray(m)), s3.shape[:2]) > 0
		if np.issubdtype(src.dtype, np.integer) and np.issubdtype(ref.dtype, np.integer):
			# a clipped pixel carries no gain information: it only says "at
			# least this bright", and a gain fitted through it is pulled low
			lo_s, hi_s = np.iinfo(src.dtype).min, np.iinfo(src.dtype).max
			lo_r, hi_r = np.iinfo(ref.dtype).min, np.iinfo(ref.dtype).max
			unsat = ((x_all > lo_s) & (x_all < hi_s)
			         & (y_all > lo_r) & (y_all < hi_r)).all(axis=2)
			good &= unsat

		flat_good = good.ravel()
		gains = np.ones(channels, np.float64)
		biases = np.zeros(channels, np.float64)
		inliers = np.ones(s3.shape[:2], bool)
		residuals, ok = [], True
		if scope.startswith("all channels"):
			fit = _trimmed_gain_bias(
				x_all.reshape(-1, channels).ravel(),
				y_all.reshape(-1, channels).ravel(),
				np.repeat(flat_good, channels), model,
				float(inlier_fraction), iterations)
			if fit is None:
				ok = False
			else:
				g, b, sub, res = fit
				gains[:] = g
				biases[:] = b
				inliers = sub.reshape(s3.shape).all(axis=2)
				residuals.append(res)
		else:
			for c in range(channels):
				fit = _trimmed_gain_bias(
					x_all[:, :, c].ravel(), y_all[:, :, c].ravel(), flat_good,
					model, float(inlier_fraction), iterations)
				if fit is None:
					ok = False
					break
				gains[c], biases[c], sub, res = fit[0], fit[1], fit[2], fit[3]
				inliers &= sub.reshape(s3.shape[:2])
				residuals.append(res)

		if not ok:
			# degenerate input is a valid outcome, not a wiring mistake
			_log.info("Photometric Align: no usable fit (flat, empty or "
			          "fully-masked input) - passing the source through")
			return io.NodeOutput(polytype.restore(src, tag), np.ones(channels),
				np.zeros(channels), np.zeros(s3.shape[:2], np.uint8), 0.0, False)

		out = x_all * gains + biases
		if np.issubdtype(src.dtype, np.integer):
			info = np.iinfo(src.dtype)
			out = np.clip(np.rint(out), info.min, info.max)
		out = out.astype(src.dtype)
		if src.ndim == 2:
			out = out[:, :, 0]
		return io.NodeOutput(
			polytype.restore(np.ascontiguousarray(out), tag),
			gains, biases,
			(inliers.astype(np.uint8) * 255),
			float(np.mean(residuals)), True)


_LOCAL_WINDOWS = ["box (uniform)", "gaussian (soft)"]


def _local_mean(a, radius, window):
	"""Local average of `a` over a (2r+1)^2 window."""
	if window.startswith("gaussian"):
		return cv2.GaussianBlur(a, (0, 0), max(radius / 2.0, 0.1))
	k = 2 * int(radius) + 1
	return cv2.boxFilter(a, -1, (k, k))


class LocalLinearFit(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_LocalLinearFit",
			display_name="CV Local Linear Fit (per pixel)",
			category=CATEGORY,
			description="Fits 'target ~ slope * predictor + intercept' inside a "
			            "window around EVERY pixel and returns the fit as image "
			            "sized maps. This is the local counterpart of "
			            "'Photometric Align (Gain/Bias)': one global gain there, "
			            "a gain per pixel here. It is also the algebra behind "
			            "guided filtering, shading/flat-field estimation and "
			            "clean plate matting - a matte falls straight out of it, "
			            "because a shot over a known background obeys "
			            "C = (1-a) B + a F, so a regression of the shot on the "
			            "plate has slope 1-a and intercept a*F wherever alpha "
			            "and the foreground colour are roughly constant across "
			            "the window. Never raises on flat data: where the "
			            "predictor does not vary there is nothing to fit, and "
			            "the slope comes back 0 with the intercept holding the "
			            "local mean (check 'predictor_sigma' to find those "
			            "pixels).",
			inputs=[
				NPArray.Input("target",
					tooltip="The array being explained (the 'y' of the fit). "
					        "[H,W] or [H,W,C], any dtype; the maths runs in "
					        "float32."),
				NPArray.Input("predictor",
					tooltip="The array explaining it (the 'x'). Same size and "
					        "channel count as the target."),
				io.Int.Input("radius", default=5, min=1, max=200,
					tooltip="Half-size of the window, in pixels: the fit at "
					        "each pixel uses a (2*radius+1) square around it. "
					        "Bigger windows average out noise but assume the "
					        "relationship holds over a wider area - for matting "
					        "that means the alpha edge gets smeared, so keep it "
					        "just large enough to contain some predictor "
					        "texture."),
				io.Combo.Input("scope", options=["pooled (one slope for all "
				                                 "channels)", "per channel"],
					default="pooled (one slope for all channels)",
					tooltip="Pooled fits ONE slope per pixel using every "
					        "channel as evidence (the right choice when the "
					        "slope is a physical scalar, e.g. 1-alpha in "
					        "matting, and it is far more stable). Per channel "
					        "fits B, G and R independently, which also captures "
					        "a colour shift."),
				io.Combo.Input("window", options=_LOCAL_WINDOWS,
					default=_LOCAL_WINDOWS[0],
					tooltip="Shape of the averaging window. Box weights every "
					        "pixel in the square equally; gaussian (sigma = "
					        "radius/2) weights the centre more, which makes the "
					        "maps smoother and their edges less blocky."),
				io.Float.Input("ridge", default=0.0, min=0.0, max=1e6, step=1.0,
					optional=True, advanced=True,
					tooltip="Added to the predictor variance before dividing, "
					        "in SQUARED input units (so ~1.0 for the sensor "
					        "noise of an 8-bit image). It pulls the slope "
					        "toward 0 where the predictor barely varies instead "
					        "of letting noise decide it."),
			],
			outputs=[
				NPArray.Output(display_name="slope",
					tooltip="float32 [H,W] (pooled) or [H,W,C] (per channel). "
					        "0 where the predictor is flat."),
				NPArray.Output(display_name="intercept",
					tooltip="float32, same shape as the inputs: "
					        "local_mean(target) - slope * local_mean(predictor), "
					        "i.e. the target's value where the predictor would "
					        "be 0."),
				NPArray.Output(display_name="predictor_sigma",
					tooltip="float32 [H,W] (pooled: sqrt of the summed "
					        "per-channel variance) or [H,W,C]. How much the "
					        "predictor varies inside the window - the "
					        "confidence of the slope, in input units. A flat "
					        "region reads ~0 and its slope means nothing."),
				NPArray.Output(display_name="residual",
					tooltip="float32: local standard deviation of the target "
					        "AROUND the fitted line, in input units. Large "
					        "where a single line cannot describe the window "
					        "(e.g. an alpha edge crossing it)."),
			],
		)

	@classmethod
	def execute(cls, target, predictor, radius, scope, window,
	            ridge=0.0) -> io.NodeOutput:
		y = np.asarray(target)
		x = np.asarray(predictor)
		if y.shape[:2] != x.shape[:2]:
			raise ValueError(
				f"target is {y.shape[0]}x{y.shape[1]} but predictor is "
				f"{x.shape[0]}x{x.shape[1]} - the two must be pixel aligned "
				f"(warp/resize one of them first).")
		y3 = y if y.ndim == 3 else y[:, :, None]
		x3 = x if x.ndim == 3 else x[:, :, None]
		if y3.shape[2] != x3.shape[2]:
			raise ValueError(
				f"target has {y3.shape[2]} channel(s) and predictor "
				f"{x3.shape[2]} - convert one of them (e.g. cv2.cvtColor).")

		# centre on the global means first: the local moments then work on
		# numbers the size of the VARIATION, not of the intensities, which is
		# what keeps E[xy] - E[x]E[y] from cancelling away its own precision
		gx = x3.reshape(-1, x3.shape[2]).mean(0).astype(np.float32)
		gy = y3.reshape(-1, y3.shape[2]).mean(0).astype(np.float32)
		xf = x3.astype(np.float32) - gx
		yf = y3.astype(np.float32) - gy
		r = int(radius)
		mx = _local_mean(xf, r, window)
		my = _local_mean(yf, r, window)
		cov = _local_mean(xf * yf, r, window) - mx * my
		var_x = _local_mean(xf * xf, r, window) - mx * mx
		var_y = _local_mean(yf * yf, r, window) - my * my
		if mx.ndim == 2:                      # cv2 drops a length-1 channel axis
			mx, my = mx[:, :, None], my[:, :, None]
			cov, var_x, var_y = (cov[:, :, None], var_x[:, :, None],
			                     var_y[:, :, None])
		var_x = np.maximum(var_x, 0.0)
		var_y = np.maximum(var_y, 0.0)

		pooled = scope.startswith("pooled")
		if pooled:
			cov_f = cov.sum(2)
			var_xf = var_x.sum(2)
			var_yf = var_y.sum(2)
		else:
			cov_f, var_xf, var_yf = cov, var_x, var_y
		den = var_xf + float(ridge)
		slope = np.where(den > 1e-9, cov_f / np.maximum(den, 1e-9), 0.0)
		slope = slope.astype(np.float32)
		s3 = slope[:, :, None] if pooled else slope
		intercept = (my + gy) - s3 * (mx + gx)
		resid = np.sqrt(np.maximum(
			var_yf - 2.0 * slope * cov_f + slope * slope * var_xf, 0.0))
		sigma = np.sqrt(var_xf)

		def _shape(a):
			a = np.ascontiguousarray(a.astype(np.float32))
			return a[:, :, 0] if (a.ndim == 3 and y.ndim == 2) else a

		return io.NodeOutput(_shape(slope), _shape(intercept), _shape(sigma),
			_shape(resid))


class LabelsToMasks(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_LabelsToMasks",
			display_name="CV Labels to Masks (full size)",
			category=CATEGORY,
			description="Splits a label map (per-pixel region id: HFS, "
			            "superpixels, connectedComponents, k-means cluster "
			            "maps) into one full-size mask per distinct label. "
			            "Every present label becomes a region - including 0, "
			            "unless 'background' drops it. Labels need not "
			            "be contiguous; the batch rows align with "
			            "'label_values'. Empty input yields an empty MASK "
			            "batch and count 0 and never raises.",
			inputs=[
				NPArray.Input("labels", tooltip="(H, W) integer label map; "
					"each distinct id becomes one full-size mask. A batch of "
					"maps uses frame 0."),
				io.Combo.Input("sort", options=list(_LABEL_SORTS),
					default="by label value",
					tooltip="Order of the output masks: by label id, or by "
					        "area (largest first)."),
				io.Combo.Input("background",
					options=["keep every label", "drop label 0"],
					default="keep every label", optional=True, advanced=True,
					tooltip="'drop label 0' skips the background region, which "
					        "is what a connectedComponents-style map (or any "
					        "map built with 'CV Take By Index' over one) uses 0 "
					        "for - without it every such map emits an extra, "
					        "usually full-frame, background mask."),
			],
			outputs=[
				io.Mask.Output(display_name="masks",
					tooltip="One full-size mask per distinct label, as a "
					        "MASK batch (aligned with 'label_values')."),
				NPArray.Output(display_name="label_values",
					tooltip="(B,) int32: the label id of each mask."),
				io.Int.Output(display_name="count",
					tooltip="B, the number of distinct labels."),
			],
		)

	@classmethod
	def execute(cls, labels, sort="by label value",
	            background="keep every label") -> io.NodeOutput:
		regions = _label_regions(labels)
		if background == "drop label 0":
			regions = [r for r in regions if r[0] != 0]
		if not regions:
			return io.NodeOutput(imageutils.u8_to_mask_batch([]),
				np.zeros((0,), np.int32), 0)
		regions.sort(key=_LABEL_SORTS[sort])
		arr = np.asarray(labels)
		if arr.ndim == 3:
			arr = arr[0]
		mask_frames = [np.where(arr == r[0], 255, 0).astype(np.uint8)
		               for r in regions]
		return io.NodeOutput(
			imageutils.u8_to_mask_batch(mask_frames),
			np.int32([r[0] for r in regions]),
			len(regions),
		)


NODES = [
	Transform,
	Contrast, ColorMap, DetectLines,
	AbsentColor,
	MatchTemplateMultiScale,
	ConnectedComponents, MasksToBBoxes, BBoxesToMasks, SnapBBoxes,
	CropByBBoxes, CropByMasks, PasteByBBox, ScaleBBoxes,
	BBoxesToArray, ArrayToBBoxes, NMSBoxes, BoxIoUMatrix,
	OverlayMasks,
	PhotometricAlign, LocalLinearFit,
	LabelsToMasks,
	ApplyChromaticAberration, CreateCACalibration,
	ChromaticAberrationCorrection,
]
