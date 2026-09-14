# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Segmentation building blocks (NPARRAY-native).

GrabCut. The auto-generated cv2.grabCut wrapper in lowlevel.py is unusable in
practice - it expects caller-managed bgdModel/fgdModel state arrays, magic
GC_* integers and a hand-built label mask - so GrabCut gets a proper node
seeded from the things workflows actually have: annotated points (a bounding
rectangle), a probable-foreground mask (e.g. the polygon of 'CV Annotate
Points') and optional sure-FG/BG scribbles.

(A "what threshold should I use?" sweep is NOT a node here: emitting a numeric
range as a ComfyUI list is generic control flow, already covered by
'Basic data handling: DataListRange' - feed its output to cv2.threshold's
'thresh' input. Workflow 14 shows the pattern.)
"""

import cv2
import numpy as np

from comfy_api.latest import io

from . import imageutils, polytype
from .convert import NPArray
from .features import _to_bgr_u8, _to_gray_u8

CATEGORY = "image/CV/segmentation"


# ----------------------------------------------------------------------------
# Colour-range selection (complements the low-level cv2_inRange)
#
# A numpy array carries no colour-space tag, so the node states the 'format':
# which channel (if any) is a CIRCULAR hue and its period. Hue is channel 0 in
# OpenCV's HSV/HLS; cvtColor on 8-bit data packs 0..360deg into 0..179, the
# *_FULL variants into 0..255, and float conversions keep degrees (0..360).
# A circular channel wraps, so a band centred on 0 also accepts values near the
# top of the range - something cv2_inRange cannot express in a single call.
# ----------------------------------------------------------------------------

_FMT_LINEAR = "linear - no circular channel (BGR/RGB/Lab/Luv/YCrCb/gray)"
COLOR_FORMATS = [
	_FMT_LINEAR,
	"hue in channel 0, period 180 (OpenCV 8-bit HSV/HLS)",
	"hue in channel 0, period 256 (OpenCV 'full' HSV/HLS)",
	"hue in channel 0, period 360 (float HSV/HLS, degrees)",
]
_FORMAT_PERIOD = {
	_FMT_LINEAR: None,
	COLOR_FORMATS[1]: 180.0,
	COLOR_FORMATS[2]: 256.0,
	COLOR_FORMATS[3]: 360.0,
}


def _as_hwc(arr: np.ndarray) -> np.ndarray:
	a = np.asarray(arr)
	if a.ndim == 2:
		a = a[..., None]
	if a.ndim != 3:
		raise ValueError(f"Expected a colour array (H,W) or (H,W,C), got shape {a.shape}.")
	return a


def _channel_periods(fmt: str, n_channels: int) -> list:
	"""Per-channel circular period (or None) for the chosen colour format."""
	period = _FORMAT_PERIOD[fmt]
	periods = [None] * n_channels
	if period is not None and n_channels >= 1:
		periods[0] = period  # hue is channel 0 in HSV/HLS
	return periods


def _fit_vector(vec, c: int, name: str) -> np.ndarray:
	v = np.atleast_1d(np.asarray(vec, np.float64)).ravel()
	if v.size == 1:
		v = np.repeat(v, c)
	if v.size != c:
		raise ValueError(f"{name} has {v.size} value(s) but the array has {c} channel(s); "
		                 f"give one per channel or a single value to broadcast.")
	return v


def _range_mask(arr: np.ndarray, center, tolerance, periods: list) -> np.ndarray:
	"""uint8 0/255 mask of pixels within center +/- tolerance on every channel;
	channels with a period use the wrap-around (circular) distance."""
	a = _as_hwc(arr).astype(np.float64)
	h, w, c = a.shape
	center = _fit_vector(center, c, "center")
	tol = np.abs(_fit_vector(tolerance, c, "tolerance"))
	keep = np.ones((h, w), bool)
	for ch in range(c):
		diff = a[..., ch] - center[ch]
		p = periods[ch] if ch < len(periods) else None
		if p:
			diff = (diff + p / 2.0) % p - p / 2.0  # signed distance in (-p/2, p/2]
		keep &= np.abs(diff) <= tol[ch]
	return keep.astype(np.uint8) * 255


def _sample_color_stats(arr: np.ndarray, region, periods: list):
	"""Per-channel (center, spread) of the sampled pixels. Linear channels use
	mean / std; circular channels use the circular mean and circular standard
	deviation, so a hue spread straddling 0 is measured correctly."""
	a = _as_hwc(arr).astype(np.float64)
	h, w, c = a.shape
	if region is not None:
		m = _to_gray_u8(region)
		if m.shape[:2] != (h, w):
			m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
		pts = a[m > 0]
	else:
		pts = a.reshape(-1, c)
	if pts.shape[0] == 0:
		return None
	center = np.zeros(c, np.float64)
	spread = np.zeros(c, np.float64)
	for ch in range(c):
		vals = pts[:, ch]
		p = periods[ch] if ch < len(periods) else None
		if p:
			ang = vals * (2.0 * np.pi / p)
			cos_m, sin_m = float(np.cos(ang).mean()), float(np.sin(ang).mean())
			center[ch] = (np.arctan2(sin_m, cos_m) % (2.0 * np.pi)) * (p / (2.0 * np.pi))
			resultant = min(1.0, max(np.hypot(cos_m, sin_m), 1e-12))
			spread[ch] = np.sqrt(-2.0 * np.log(resultant)) * (p / (2.0 * np.pi))
		else:
			center[ch] = float(vals.mean())
			spread[ch] = float(vals.std())
	return center, spread


class GrabCut(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_GrabCut",
			display_name="CV GrabCut",
			category=CATEGORY,
			description="Interactive-style foreground extraction (cv2.grabCut): "
			            "give it a rough hint of where the object is and it "
			            "refines the boundary from color statistics. Hints, "
			            "combinable: 'points' (2 or more, e.g. from 'OpenCV "
			            "Annotate Points') mark the object's bounding rectangle "
			            "- everything outside is sure background; 'mask' marks "
			            "probable foreground (the annotate node's polygon "
			            "output fits directly); sure_foreground / "
			            "sure_background scribbles pin pixels that must keep "
			            "their side. Outputs the foreground as a 0/255 mask "
			            "plus the raw 4-value label map (sure/probable x fg/bg "
			            "- view with 'Preview CV Array' in heatmap mode). "
			            "Failure-tolerant: with no usable hints, or when "
			            "everything ends up background, success=false and an "
			            "empty mask come out instead of an error - branch on "
			            "success with an if/else node.",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Color image (BGR uint8); grabCut needs 3 channels "
					        "- grayscale is converted."),
				io.Int.Input("iterations", default=5, min=1, max=50,
					tooltip="GrabCut refinement iterations; 5 is usually "
					        "plenty."),
				NPArray.Input("points", optional=True,
					tooltip="2+ points whose bounding rectangle contains the "
					        "object (Nx1x2 or Nx2). Everything outside the "
					        "rectangle becomes sure background."),
				polytype.poly_input("mask", types=polytype.MASK_ONLY, optional=True,
					tooltip="Mask; non-zero = probable FOREGROUND (the "
					        "rest stays probable background unless 'points' "
					        "also marks a rectangle). The polygon mask of "
					        "'CV Annotate Points' plugs in directly."),
				polytype.poly_input("sure_foreground", types=polytype.MASK_ONLY, optional=True,
					tooltip="Scribble mask; non-zero pixels are forced "
					        "to stay foreground."),
				polytype.poly_input("sure_background", types=polytype.MASK_ONLY, optional=True,
					tooltip="Scribble mask; non-zero pixels are forced "
					        "to stay background."),
			],
			outputs=[
				NPArray.Output(display_name="mask",
					tooltip="uint8 0/255 foreground mask (sure + probable "
					        "foreground)."),
				NPArray.Output(display_name="labels",
					tooltip="Raw GC label map: 0 sure BG, 1 sure FG, 2 "
					        "probable BG, 3 probable FG - view with 'Preview "
					        "CV Array' (heatmap)."),
				io.Boolean.Output(display_name="success"),
			],
		)

	@classmethod
	def _fit(cls, hint, h, w) -> np.ndarray:
		mask = _to_gray_u8(hint)
		if mask.shape != (h, w):
			mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
		return mask > 0

	@classmethod
	def execute(cls, image, iterations, points=None, mask=None,
	            sure_foreground=None, sure_background=None) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		mask, _ = polytype.resolve(mask)
		sure_foreground, _ = polytype.resolve(sure_foreground)
		sure_background, _ = polytype.resolve(sure_background)
		img = _to_bgr_u8(image)
		h, w = img.shape[:2]
		labels = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
		seeded = False

		pts = (np.zeros((0, 2)) if points is None
		       else np.asarray(points, np.float64).reshape(-1, 2))
		if len(pts) >= 2:
			x, y, bw, bh = cv2.boundingRect(np.float32(pts).reshape(-1, 1, 2))
			x, y = max(0, x), max(0, y)
			bw, bh = min(bw, w - x), min(bh, h - y)
			if bw > 0 and bh > 0 and (bw < w or bh < h):
				labels[:] = cv2.GC_BGD  # outside the rectangle = sure background
				labels[y:y + bh, x:x + bw] = cv2.GC_PR_FGD
				seeded = True
		if mask is not None:
			probable = cls._fit(mask, h, w)
			labels[probable] = cv2.GC_PR_FGD
			seeded = seeded or bool(probable.any())
		if sure_background is not None:
			labels[cls._fit(sure_background, h, w)] = cv2.GC_BGD
		if sure_foreground is not None:
			sure = cls._fit(sure_foreground, h, w)
			labels[sure] = cv2.GC_FGD
			seeded = seeded or bool(sure.any())

		fg = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD))
		if not seeded or fg.all() or not fg.any():
			return io.NodeOutput(np.zeros((h, w), np.uint8), labels, False)

		bgd_model = np.zeros((1, 65), np.float64)
		fgd_model = np.zeros((1, 65), np.float64)
		try:
			cv2.grabCut(img, labels, None, bgd_model, fgd_model, iterations,
			            cv2.GC_INIT_WITH_MASK)
		except cv2.error as e:
			raise RuntimeError(
				f"cv2.grabCut failed on ndarray{img.shape} {img.dtype}: {e}"
			) from e
		out = (np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD))
		       .astype(np.uint8) * 255)
		return io.NodeOutput(out, labels, bool(out.any()))


class ColorRange(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ColorRange",
			display_name="CV Color Range",
			category=CATEGORY,
			description="Selects the pixels whose colour is within `tolerance` of "
			            "`center` on every channel - 'cv2.inRange around a value' "
			            "rather than lower/upper bounds. Unlike the raw inRange it "
			            "is CIRCULAR-aware: tell it via `format` which channel is a "
			            "hue and its period, and the band wraps, so a hue centred on "
			            "0 also keeps values near the top of the range (red across "
			            "the 179/0 seam) in ONE pass. Center/tolerance take one "
			            "value per channel (a 'CV Scalar' literal, or the outputs of "
			            "'CV Color Range From Sample'); a single value "
			            "broadcasts. Outputs a 0/255 mask - chain into morphology / "
			            "bitwise_and or 'CV Array -> Mask'. Convert to HSV/HLS first "
			            "(cv2_cvtColor) for intuitive hue/sat/val selection.",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Colour array in whatever space you want to threshold "
					        "(BGR, HSV, Lab, a single hue channel...). Set 'format' "
					        "to match. A ComfyUI IMAGE arrives as BGR uint8."),
				NPArray.Input("center",
					tooltip="Target colour, one value per channel, e.g. a 'CV "
					        "Scalar' (60, 200, 200) for green in 8-bit HSV. A single "
					        "value broadcasts to every channel."),
				NPArray.Input("tolerance",
					tooltip="Accepted half-width per channel, e.g. (15, 60, 60). A "
					        "pixel is kept when every channel is within center +/- "
					        "tolerance; for a circular channel a tolerance >= "
					        "period/2 accepts every hue."),
				io.Combo.Input("format", options=COLOR_FORMATS, default=_FMT_LINEAR,
					tooltip="Which channel (if any) is a circular hue, and its "
					        "period - numpy arrays carry no colour-space tag. All "
					        "other channels are treated linearly."),
			],
			outputs=[NPArray.Output(display_name="mask",
				tooltip="uint8 0/255 mask of the in-range pixels.")],
		)

	@classmethod
	def execute(cls, image, center, tolerance, format) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		periods = _channel_periods(format, _as_hwc(image).shape[2])
		return io.NodeOutput(_range_mask(image, center, tolerance, periods))


class ColorRangeFromSample(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ColorRangeFromSample",
			display_name="CV Color Range From Sample",
			category=CATEGORY,
			description="Learns a colour band from the image itself: it samples the "
			            "pixels (inside `region` if given, else the whole image), "
			            "takes their per-channel center and spread, and keeps "
			            "everything within `sigma` * spread of that center. Spread "
			            "is the standard deviation for linear channels and the "
			            "CIRCULAR standard deviation for a hue channel (set via "
			            "`format`), so a reddish sample straddling the 179/0 seam is "
			            "measured correctly. Seed `region` with a scribble or the "
			            "polygon mask of 'CV Annotate Points' over the colour you "
			            "want. Outputs the mask plus the derived `center` and "
			            "`tolerance` (feed them to 'CV Color Range' to reuse the "
			            "band on another image, or inspect them).",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Colour array to sample and threshold (set 'format' to "
					        "match its colour space). A ComfyUI IMAGE arrives as "
					        "BGR uint8."),
				io.Float.Input("sigma", default=2.5, min=0.0, max=20.0, step=0.1,
					tooltip="Band half-width in standard deviations: tolerance = "
					        "sigma * spread per channel. ~2-3 keeps most of a "
					        "normally-distributed colour; raise to be more "
					        "permissive."),
				io.Combo.Input("format", options=COLOR_FORMATS, default=_FMT_LINEAR,
					tooltip="Which channel (if any) is a circular hue, and its "
					        "period. The hue center/spread are then computed "
					        "circularly."),
				polytype.poly_input("region", types=polytype.MASK_ONLY, optional=True,
					tooltip="Mask marking which pixels to sample (non-zero = "
					        "sample). The polygon output of 'CV Annotate Points' "
					        "plugs in directly. Omit to sample the whole image."),
			],
			outputs=[
				NPArray.Output(display_name="mask",
					tooltip="uint8 0/255 mask of the in-range pixels."),
				NPArray.Output(display_name="center",
					tooltip="Per-channel sampled center (circular mean for the hue "
					        "channel) - float64, one element per channel."),
				NPArray.Output(display_name="tolerance",
					tooltip="Per-channel sigma * spread used for the band - feed it "
					        "and 'center' into 'CV Color Range'."),
			],
		)

	@classmethod
	def execute(cls, image, sigma, format, region=None) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		region, _ = polytype.resolve(region)
		c = _as_hwc(image).shape[2]
		periods = _channel_periods(format, c)
		stats = _sample_color_stats(image, region, periods)
		if stats is None:  # empty region -> nothing to learn, keep the workflow alive
			h, w = _as_hwc(image).shape[:2]
			zeros = np.zeros(c, np.float64)
			return io.NodeOutput(np.zeros((h, w), np.uint8), zeros, zeros)
		center, spread = stats
		tolerance = sigma * spread
		mask = _range_mask(image, center, tolerance, periods)
		return io.NodeOutput(mask, center, tolerance)


# ----------------------------------------------------------------------------
# Histogram + back-projection (cv2.calcHist / cv2.calcBackProject)
#
# Both cv2 functions take SEQUENCE parameters (lists of images, channels, bins
# and flattened ranges) that the .pyi-driven generator cannot represent, so the
# raw registry has no wrapper for them - a curated pair is the only way to get
# them at all. They share the channels/ranges vocabulary: compute a histogram
# over some channels of a reference region, then 'CV Back Project' scores every
# pixel of another image by how common its color was in that histogram - the
# classic soft segmentation behind meanShift/CamShift tracking.
# ----------------------------------------------------------------------------

_CHANNELS_TOOLTIP = (
	"Channel indices to histogram, comma-separated, e.g. '0' (hue of an HSV "
	"array) or '0, 1' (hue + saturation). Must exist in the image.")
_RANGES_TOOLTIP = (
	"Value range per channel as 'lo, hi' pairs, comma-separated and flattened, "
	"e.g. '0, 180' for 8-bit hue or '0, 180, 0, 256' for hue + saturation. A "
	"single pair broadcasts to every channel. 'hi' is EXCLUSIVE (8-bit full "
	"range = 0, 256).")

_NORM_NONE = "none (raw counts)"
_NORM_PEAK = "peak = 255 (back-projection ready)"
_NORM_SUM = "sum = 1 (probability)"


def _parse_numbers(text: str, name: str) -> list:
	parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
	try:
		values = [float(p) for p in parts]
	except ValueError as e:
		raise ValueError(f"'{name}' must be comma-separated numbers, got '{text}'. ({e})") from e
	if not values:
		raise ValueError(f"'{name}' is empty - give at least one number.")
	return values


def _hist_spec(image, channels, ranges, bins=None):
	"""Validated (img, channel_list, range_list, bin_list) for calcHist /
	calcBackProject. `bins=None` skips the bins (back-projection has none)."""
	img = _as_hwc(np.asarray(image))
	if img.dtype not in (np.uint8, np.uint16, np.float32):
		img = img.astype(np.float32)
	img = np.ascontiguousarray(img)
	chan = [int(v) for v in _parse_numbers(channels, "channels")]
	bad = [c for c in chan if not 0 <= c < img.shape[2]]
	if bad:
		raise ValueError(
			f"channels {bad} do not exist: the image has {img.shape[2]} "
			f"channel(s) (indices 0..{img.shape[2] - 1}).")
	rng = _parse_numbers(ranges, "ranges")
	if len(rng) == 2:
		rng = rng * len(chan)
	if len(rng) != 2 * len(chan):
		raise ValueError(
			f"ranges needs one 'lo, hi' pair per channel ({len(chan)} channel(s) "
			f"-> {2 * len(chan)} numbers, or one pair to broadcast), got {len(rng)}.")
	nbin = None
	if bins is not None:
		nbin = [int(v) for v in _parse_numbers(bins, "bins")]
		if len(nbin) == 1:
			nbin = nbin * len(chan)
		if len(nbin) != len(chan) or any(b < 1 for b in nbin):
			raise ValueError(
				f"bins needs one positive count per channel ({len(chan)} "
				f"channel(s), or one count to broadcast), got '{bins}'.")
	return img, chan, rng, nbin


class CalcHist(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CalcHist",
			display_name="CV Histogram",
			category=CATEGORY,
			description="Counts how often each color value occurs "
			            "(cv2.calcHist): picks one or more channels, splits "
			            "their range into bins and returns the bin counts as an "
			            "NPARRAY (1-D for one channel, 2-D for two...). Give it "
			            "a mask to histogram only a region - e.g. the polygon "
			            "of 'CV Annotate Points' over an object - and set "
			            "normalize to 'peak = 255' to feed the result straight "
			            "into 'CV Back Project' (the meanShift/CamShift "
			            "tracking recipe). Convert to HSV first (cv2_cvtColor) "
			            "to histogram hue rather than raw BGR. An empty mask "
			            "yields an all-zero histogram and pixels=0 instead of "
			            "an error. Preview 1-D/2-D histograms with 'Preview CV "
			            "Array' (normalize/heatmap).",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Array to histogram, in whatever color space you "
					        "want to count (convert with cv2_cvtColor first). "
					        "A ComfyUI IMAGE arrives as BGR uint8."),
				io.String.Input("channels", default="0", tooltip=_CHANNELS_TOOLTIP),
				io.String.Input("bins", default="32",
					tooltip="Bin count per channel, comma-separated (a single "
					        "count broadcasts), e.g. '180' for every 8-bit hue "
					        "or '30, 32' for a 2-D hue x saturation histogram. "
					        "Fewer bins = smoother, more tolerant matching."),
				io.String.Input("ranges", default="0, 256", tooltip=_RANGES_TOOLTIP),
				io.Combo.Input("normalize",
					options=[_NORM_NONE, _NORM_PEAK, _NORM_SUM],
					default=_NORM_NONE,
					tooltip="Post-scaling: raw counts; peak stretched to 255 "
					        "(what cv2.calcBackProject expects); or divided by "
					        "the total so the bins sum to 1. An all-zero "
					        "histogram stays zero in every mode."),
				polytype.poly_input("mask", types=polytype.MASK_ONLY, optional=True,
					tooltip="Optional region mask: non-zero pixels are counted, "
					        "the rest ignored. Resized to the image if needed. "
					        "Omit to histogram the whole image."),
			],
			outputs=[
				NPArray.Output(display_name="histogram",
					tooltip="float32 bin counts, one axis per channel (shape "
					        "(bins,) for one channel, (bins0, bins1) for two). "
					        "Feed to 'CV Back Project' with the SAME "
					        "channels + ranges."),
				io.Int.Output(display_name="pixels",
					tooltip="How many pixels were counted (inside the mask and "
					        "the ranges) - 0 means the histogram is empty."),
			],
		)

	@classmethod
	def execute(cls, image, channels, bins, ranges, normalize, mask=None) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		mask, _ = polytype.resolve(mask)
		img, chan, rng, nbin = _hist_spec(image, channels, ranges, bins)
		m = None
		if mask is not None:
			m = _to_gray_u8(mask)
			if m.shape[:2] != img.shape[:2]:
				m = cv2.resize(m, (img.shape[1], img.shape[0]),
				               interpolation=cv2.INTER_NEAREST)
			m = (m > 0).astype(np.uint8) * 255
		try:
			hist = cv2.calcHist([img], chan, m, nbin, rng)
		except cv2.error as e:
			raise RuntimeError(
				f"cv2.calcHist failed on ndarray{img.shape} {img.dtype} "
				f"(channels={chan}, bins={nbin}, ranges={rng}): {e}") from e
		hist = np.asarray(hist, np.float64)
		total = float(hist.sum())
		if total > 0.0:
			if normalize == _NORM_PEAK:
				hist = hist / float(hist.max()) * 255.0
			elif normalize == _NORM_SUM:
				hist = hist / total
		hist = hist.astype(np.float32)
		return io.NodeOutput(hist, int(round(total)))


class CalcBackProject(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CalcBackProject",
			display_name="CV Back Project",
			category=CATEGORY,
			description="Scores every pixel by how common its color was in a "
			            "reference histogram (cv2.calcBackProject): the output "
			            "map is bright where the pixel's value fell in a "
			            "well-filled bin and 0 elsewhere - a soft 'how much "
			            "does this look like the reference?' segmentation. "
			            "Compute the histogram with 'CV Histogram' "
			            "(normalize 'peak = 255') over the SAME color space, "
			            "and repeat its channels + ranges here. Chain the map "
			            "into 'CV Track Window' (meanShift/CamShift), "
			            "threshold it into a mask, or view it with 'Preview CV "
			            "Array' (heatmap). An all-zero histogram simply yields "
			            "an all-zero map.",
			inputs=[
				polytype.poly_input("image", types=polytype.IMAGE_ONLY,
					tooltip="Array to score, in the SAME color space the "
					        "histogram was computed in (convert with "
					        "cv2_cvtColor first)."),
				NPArray.Input("histogram",
					tooltip="Reference histogram from 'CV Histogram' - use "
					        "normalize 'peak = 255' there so the map spans the "
					        "full 0..255 range."),
				io.String.Input("channels", default="0", tooltip=_CHANNELS_TOOLTIP),
				io.String.Input("ranges", default="0, 256",
					tooltip=_RANGES_TOOLTIP + " Must repeat the values the "
					        "histogram was computed with."),
				io.Float.Input("scale", default=1.0, min=0.0, max=1e6, step=0.1,
					tooltip="Multiplier on the output scores (uint8 output "
					        "saturates at 255). 1 = use the histogram values "
					        "as-is."),
			],
			outputs=[NPArray.Output(display_name="backprojection",
				tooltip="Per-pixel likeness map, same height/width and dtype "
				        "as the image (uint8 in, uint8 out; single channel). "
				        "Bright = matches the reference colors.")],
		)

	@classmethod
	def execute(cls, image, histogram, channels, ranges, scale) -> io.NodeOutput:
		image, _ = polytype.resolve(image)
		img, chan, rng, _ = _hist_spec(image, channels, ranges)
		hist = np.ascontiguousarray(np.asarray(histogram, np.float32))
		if hist.ndim != len(chan):
			raise ValueError(
				f"histogram has {hist.ndim} axis/axes but {len(chan)} "
				f"channel(s) were requested - channels/ranges must repeat the "
				f"values 'CV Histogram' was run with.")
		try:
			back = cv2.calcBackProject([img], chan, hist, rng, float(scale))
		except cv2.error as e:
			raise RuntimeError(
				f"cv2.calcBackProject failed on ndarray{img.shape} {img.dtype} "
				f"(channels={chan}, hist{hist.shape}, ranges={rng}): {e}") from e
		return io.NodeOutput(back)


class TemporalReduce(io.ComfyNode):
	_STATS = {
		"median (robust static background)": np.median,
		"mean (smooth average)": np.mean,
		"min (darkest over time)": np.min,
		"max (brightest over time)": np.max,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_TemporalReduce",
			display_name="CV Temporal Reduce (Background Plate)",
			category=CATEGORY,
			description="Collapses a whole batch of frames to ONE image by reducing "
			            "each pixel OVER TIME (across the batch) - the standard way to "
			            "recover the empty 'background plate' of a fixed-camera video. "
			            "With the MEDIAN, anything that moves occupies each pixel only "
			            "a minority of the time and averages out, leaving the static "
			            "scene (pedestrians, cars vanish); mean is smoother but ghosts "
			            "moving objects, min/max pick the darkest/brightest value seen. "
			            "Feed a video IMAGE batch (LoadVideo -> Get Video Components); "
			            "the plate comes out as a BGR uint8 NPARRAY. A LATENT batch "
			            "({'samples': [B,C,H,W]}) also works - the same per-pixel "
			            "reduction runs over the batch axis in latent space and the "
			            "plate comes out as a FLOAT32 (H,W,C) NPARRAY (latent values "
			            "are never uint8-quantized), any channel count. Pair it with "
			            "'CV Background Model (Update)' (wire the plate into 'mean', "
			            "set learning_rate 0) to flag whatever differs from the "
			            "background in each frame. Also a generic temporal denoiser / "
			            "long-exposure. Best with frames spread across the clip so "
			            "moving objects sit in different places.",
			inputs=[
				io.MultiType.Input("images", [io.Image, io.Latent],
					tooltip="Video frames to reduce over time: an IMAGE batch (e.g. "
					        "'Get Video Components' images) or a LATENT batch "
					        "({'samples': [B,C,H,W]}). The reduction runs over the "
					        "batch dimension, one output pixel per input pixel. A "
					        "LATENT is reduced in float32 latent space (values "
					        "untouched, any channel count); an IMAGE reduces to BGR "
					        "uint8."),
				io.Combo.Input("statistic", options=list(cls._STATS),
					default="median (robust static background)",
					tooltip="Per-pixel reduction over time. Median = robust static "
					        "background (moving objects drop out); mean = smooth "
					        "average (ghosts movers); min/max = darkest/brightest."),
			],
			outputs=[
				NPArray.Output(display_name="plate",
					tooltip="The reduced background plate - BGR uint8 (H,W,3) for an "
					        "IMAGE batch, or FLOAT32 (H,W,C) for a LATENT batch. Wire "
					        "into 'CV Background Model (Update)' as 'mean', or "
					        "preview it with 'Preview CV Array'."),
			],
		)

	@classmethod
	def execute(cls, images, statistic) -> io.NodeOutput:
		reduce = cls._STATS[statistic]
		if polytype.is_latent(images):  # reduce in latent space, keep float32
			samples = images["samples"]
			if samples.ndim != 4:
				raise ValueError(
					f"Only 4-D [B,C,H,W] latents are supported, got shape "
					f"{tuple(samples.shape)}.")
			stack = samples.detach().float().cpu().numpy()  # [B,C,H,W]
			if stack.shape[0] == 0:  # empty batch -> keep the workflow alive
				return io.NodeOutput(np.zeros((1, 1, stack.shape[1]), np.float32))
			plate = reduce(stack.astype(np.float64), axis=0)  # [C,H,W]
			plate = np.moveaxis(plate, 0, -1).astype(np.float32)  # [H,W,C]
			return io.NodeOutput(np.ascontiguousarray(plate))
		frames = imageutils.image_batch_to_bgr(images)
		if not frames:  # empty batch -> keep the workflow alive
			return io.NodeOutput(np.zeros((1, 1, 3), np.uint8))
		stack = np.stack(frames, axis=0).astype(np.float64)
		plate = reduce(stack, axis=0)
		return io.NodeOutput(np.clip(np.rint(plate), 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------
# Temporal background model (background subtraction)
#
# cv2's BackgroundSubtractorMOG2 / KNN are STATEFUL C++ objects: .apply(frame)
# mutates an internal model in place and returns the foreground mask. That
# object cannot flow through a ComfyUI graph - it is not an ndarray, and because
# ComfyUI CACHES every node's output, a re-execution would call .apply() again on
# the same live object and double-update the model (the same in-place-mutation
# hazard lowlevel.py guards against by copying ndarray args). So the model is
# carried the way 'CV Compose Pose' carries a trajectory: as plain ndarrays
# (running mean + running variance) threaded frame-to-frame, each step producing
# a NEW pair. This is the single-Gaussian running-average background model (Wren
# 1997) - MOG2 without the mixture; generic, reusable and fully testable. Chain
# one node per frame; the first frame leaves 'mean'/'variance' unwired (cold
# start -> the model is seeded from the frame and nothing is foreground yet).
# ----------------------------------------------------------------------------


class BackgroundModelUpdate(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		# frame + running mean/variance are float per-pixel state; echo the input
		# format so the model threads as NPARRAY or LATENT. NOT IMAGE on purpose -
		# a uint8 round-trip every hop would destroy the running mean/variance.
		# The foreground is always a raw single-channel mask (a LATENT frame gives
		# a latent-resolution mask).
		model_t = io.MatchType.Template("bg_model_type", [NPArray, io.Latent])
		return io.Schema(
			node_id="CV_BackgroundModelUpdate",
			display_name="CV Background Model (Update)",
			category=CATEGORY,
			description="One step of temporal background subtraction with a "
			            "single-Gaussian per-pixel model (MOG2 without the "
			            "mixture). Given the running per-pixel 'mean' and "
			            "'variance' of the scene so far and the current 'frame', "
			            "it flags pixels that deviate too far from the model as "
			            "foreground, then updates the model toward the frame. "
			            "Because cv2's BackgroundSubtractorMOG2/KNN are stateful "
			            "objects that cannot travel through a graph (and would "
			            "double-update under ComfyUI's output caching), the model "
			            "IS the 'mean'/'variance' arrays - thread them node-to-node "
			            "like 'CV Compose Pose' threads a trajectory, one node "
			            "per frame. On the FIRST frame leave 'mean' and 'variance' "
			            "unwired: the model is seeded from the frame and the "
			            "foreground comes out empty (no history yet), so no "
			            "success gate is needed. A pixel is foreground when its "
			            "distance from the mean, in standard deviations averaged "
			            "over the colour channels, exceeds 'threshold'. For a "
			            "FIXED-camera clip you can skip the state chain entirely: "
			            "precompute a background plate once with 'CV Temporal "
			            "Reduce', wire it into 'mean', set learning_rate 0, and each "
			            "frame is scored independently against that plate (which is "
			            "what lets the detector run inside an Inspire foreach - the "
			            "loop then carries only ONE accumulator, the output frames). "
			            "Outputs "
			            "data only - view the foreground with 'Preview CV Array' "
			            "(image) or turn it into a MASK for 'CV Overlay Masks' "
			            "/ 'CV Array -> Mask'. Note: the model learns everywhere at "
			            "'learning_rate', so an object that stops moving is slowly "
			            "absorbed into the background (ghosting) - lower the rate "
			            "to remember longer. Works in LATENT space too: 'frame'/"
			            "'mean'/'variance' accept a LATENT and echo that format, so "
			            "the whole detector can run on VAE-encoded frames (the "
			            "foreground then comes out at latent resolution, H/8 x W/8). "
			            "BUT the variance/threshold defaults are uint8-intensity^2 "
			            "scaled; latents are ~unit scale, so for a LATENT frame "
			            "shrink 'init_variance' to ~1 and 'min_variance' to ~0.01 or "
			            "the model is so tolerant that NOTHING ever flags.",
			inputs=[
				io.MatchType.Input("frame", template=model_t,
					tooltip="Current frame: an ndarray (feed a ComfyUI IMAGE "
					        "through 'Image -> CV Array', BGR uint8) or a LATENT (a "
					        "single VAE-encoded frame). Compared against the running "
					        "model; the model threads in whatever format arrives."),
				io.Float.Input("learning_rate", default=0.05, min=0.0, max=1.0, step=0.01,
					tooltip="How fast the model follows the scene, per frame "
					        "(alpha): mean += alpha*(frame-mean). 0 freezes the "
					        "model after the first frame; ~0.01-0.1 adapts to "
					        "lighting while remembering a moving object for many "
					        "frames; 1 makes every frame the new background."),
				io.Float.Input("threshold", default=2.5, min=0.0, max=20.0, step=0.1,
					tooltip="Foreground cutoff in standard deviations: a pixel is "
					        "foreground when its per-channel-averaged distance from "
					        "the mean, divided by the model's standard deviation, "
					        "exceeds this. ~2-3 is typical; raise to suppress noise, "
					        "lower to catch faint motion. Scale-free (it is in std "
					        "devs), so it needs no change in latent space."),
				io.Float.Input("init_variance", default=100.0, min=1e-3, max=1e5,
					tooltip="Starting per-pixel variance (uint8 intensity^2 units; "
					        "100 = std 10) used to seed the model on the first frame "
					        "and whenever 'variance' is unwired. Larger = more "
					        "tolerant until the model settles. For a LATENT frame "
					        "this is far too large (latents are ~unit scale): drop it "
					        "to ~1 or the model starts hopelessly tolerant."),
				io.Float.Input("min_variance", default=16.0, min=1e-3, max=1e5,
					tooltip="Floor on the variance (uint8 intensity^2 units; 16 = "
					        "std 4) so a perfectly still pixel keeps a little noise "
					        "budget and does not flag on 1-2 grey-level jitter "
					        "(compressed / decoded video). For a LATENT frame lower "
					        "it to ~0.01 - the uint8 floor of 16 (std 4) dwarfs "
					        "latent deviations, so with the default NOTHING flags."),
				io.MatchType.Input("mean", template=model_t, optional=True,
					tooltip="Running per-pixel mean from the previous frame's node "
					        "(same format as 'frame'). Leave UNWIRED on the first "
					        "frame to seed the model from 'frame'."),
				io.MatchType.Input("variance", template=model_t, optional=True,
					tooltip="Running per-pixel variance from the previous frame's "
					        "node. Unwired -> reset to 'init_variance'."),
			],
			outputs=[
				io.MatchType.Output(template=model_t, display_name="mean",
					tooltip="Updated per-pixel mean (float32, same shape and format "
					        "as the frame) - wire into the next frame's node."),
				io.MatchType.Output(template=model_t, display_name="variance",
					tooltip="Updated per-pixel variance (float32, same format as the "
					        "frame) - wire into the next frame's node."),
				NPArray.Output(display_name="foreground",
					tooltip="uint8 0/255 foreground mask (moving / new pixels), one "
					        "channel. For a LATENT frame it is at LATENT resolution "
					        "(H/8, W/8) - upscale x8 (nearest) to overlay on pixels. "
					        "Empty on the first frame. Convert to a MASK for overlay "
					        "/ morphology."),
			],
		)

	@classmethod
	def execute(cls, frame, learning_rate, threshold, init_variance,
	            min_variance, mean=None, variance=None) -> io.NodeOutput:
		# resolve the frame to an ndarray, remembering its format so the running
		# mean/variance echo it (NPARRAY stays NPARRAY; LATENT stays LATENT -
		# frame 0, float32, values untouched). foreground is always a raw mask.
		frame_arr, tag = polytype.resolve(frame)
		f = _as_hwc(np.asarray(frame_arr)).astype(np.float64)
		h, w, c = f.shape
		min_var = float(min_variance)

		m_in = polytype.resolve(mean)[0] if mean is not None else None
		if m_in is None or np.asarray(m_in).size == 0:  # cold start: seed, no fg yet
			mean_out = f
			var_out = np.full_like(f, max(float(init_variance), min_var))
			fg = np.zeros((h, w), np.uint8)
			return io.NodeOutput(polytype.restore(mean_out.astype(np.float32), tag),
			                     polytype.restore(var_out.astype(np.float32), tag), fg)

		# copy the incoming state - never mutate an upstream node's cached output
		m = np.array(m_in, np.float64).reshape(f.shape)
		v_in = polytype.resolve(variance)[0] if variance is not None else None
		if v_in is not None and np.asarray(v_in).size > 0:
			v = np.array(v_in, np.float64).reshape(f.shape)
		else:
			v = np.full_like(f, float(init_variance))
		v = np.maximum(v, min_var)

		diff = f - m
		d2 = diff * diff
		dist = np.sqrt((d2 / v).mean(axis=2))  # std-devs, averaged over channels
		fg = (dist > float(threshold)).astype(np.uint8) * 255

		alpha = float(learning_rate)
		m_new = m + alpha * diff
		v_new = np.maximum((1.0 - alpha) * v + alpha * d2, min_var)
		return io.NodeOutput(polytype.restore(m_new.astype(np.float32), tag),
		                     polytype.restore(v_new.astype(np.float32), tag), fg)


NODES = [GrabCut, ColorRange, ColorRangeFromSample, CalcHist, CalcBackProject,
         TemporalReduce, BackgroundModelUpdate]
