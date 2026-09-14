# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Shared tensor <-> ndarray conversion helpers.

ComfyUI IMAGE:  torch.Tensor [B, H, W, C], C=3, RGB, float32, 0..1
ComfyUI MASK:   torch.Tensor [B, H, W] (sometimes [H, W]), float32, 0..1
OpenCV image:   np.ndarray [H, W] or [H, W, C], mostly uint8 BGR
"""

import cv2
import numpy as np
import torch


def image_batch_to_bgr(image: torch.Tensor) -> list:
	"""Convert a ComfyUI IMAGE batch to a list of uint8 BGR ndarrays."""
	if image.ndim == 3:
		image = image.unsqueeze(0)
	arr = (image.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
	return [np.ascontiguousarray(frame[..., ::-1]) for frame in arr]


def bgr_to_image_batch(frames: list) -> torch.Tensor:
	"""Convert a list of BGR/gray ndarrays back to a ComfyUI IMAGE batch."""
	if len(frames) == 0:  # empty batch (e.g. no regions found) - keep it flowing
		return torch.zeros((0, 1, 1, 3))
	out = []
	for frame in frames:
		frame = ensure_uint8(frame)
		if frame.ndim == 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
		elif frame.shape[2] == 1:
			frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
		elif frame.shape[2] == 3:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		elif frame.shape[2] == 4:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
		out.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
	return torch.stack(out)


def ensure_uint8(arr: np.ndarray) -> np.ndarray:
	"""Convert an arbitrary numeric ndarray to uint8 with sensible scaling."""
	if arr.dtype == np.uint8:
		return arr
	if arr.dtype == np.uint16:
		return (arr / 257).astype(np.uint8)
	if arr.dtype == bool:
		return arr.astype(np.uint8) * 255
	arr = arr.astype(np.float64)
	if arr.size == 0:
		return arr.astype(np.uint8)
	# non-finite values (NaN/Inf from distanceTransform / optical flow / score
	# maps) would poison min()/max() and collapse the result to near-black;
	# clamp them to the finite range so the rest scales correctly
	if not np.isfinite(arr).all():
		finite = arr[np.isfinite(arr)]
		lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
		arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
	amin, amax = float(arr.min()), float(arr.max())
	if 0.0 <= amin and amax <= 1.0:
		return (arr * 255.0).round().astype(np.uint8)
	if 0.0 <= amin and amax <= 255.0:
		return arr.round().astype(np.uint8)
	if amax > amin:  # fall back to min-max normalization
		return ((arr - amin) / (amax - amin) * 255.0).round().astype(np.uint8)
	return np.clip(arr, 0, 255).astype(np.uint8)


def _to_bgr_u8(arr: np.ndarray) -> np.ndarray:
	"""Convert a single ndarray (any dtype/channels) to uint8 BGR."""
	arr = ensure_uint8(np.asarray(arr))
	if arr.ndim == 2:
		return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
	if arr.ndim == 3 and arr.shape[2] == 1:
		return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
	if arr.ndim == 3 and arr.shape[2] == 4:
		return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
	if arr.ndim == 3 and arr.shape[2] == 3:
		return arr
	raise ValueError(f"Cannot interpret array of shape {arr.shape} as an image.")


def frames_from_input(image) -> list:
	"""List of uint8 BGR ndarrays from an IMAGE batch or NPARRAY (batched or single)."""
	if isinstance(image, torch.Tensor):
		return image_batch_to_bgr(image)
	arr = np.asarray(image)
	if arr.ndim == 4:
		return [_to_bgr_u8(arr[i]) for i in range(arr.shape[0])]
	return [_to_bgr_u8(arr)]


def first_bgr_frame(image):
	"""One BGR uint8 frame: the first frame of an IMAGE batch, or the NPARRAY
	itself as a single frame. Returns None for an empty (0-length) IMAGE batch."""
	if isinstance(image, torch.Tensor):
		frames = image_batch_to_bgr(image)
		return frames[0] if frames else None
	return _to_bgr_u8(image)


def mask_batch_to_u8(mask: torch.Tensor) -> list:
	"""Convert a ComfyUI MASK to a list of uint8 [H, W] ndarrays (0..255)."""
	if mask.ndim == 2:
		mask = mask.unsqueeze(0)
	if mask.ndim == 4:  # [B, 1, H, W] or [B, H, W, 1]
		mask = mask.squeeze(1) if mask.shape[1] == 1 else mask.squeeze(-1)
	arr = (mask.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
	return [np.ascontiguousarray(frame) for frame in arr]


def masks_from_input(mask) -> list:
	"""List of uint8 [H,W] ndarrays from a MASK batch or NPARRAY mask."""
	if isinstance(mask, torch.Tensor):
		return mask_batch_to_u8(mask)
	arr = np.asarray(mask)
	arr = ensure_uint8(arr)
	if arr.ndim == 2:
		return [np.ascontiguousarray(arr)]
	# Squeeze trailing single-channel dim for [B,H,W,1] or [H,W,1]
	if arr.ndim >= 3 and arr.shape[-1] == 1:
		arr = arr[..., 0]
		if arr.ndim == 2:
			return [np.ascontiguousarray(arr)]
	# [B, H, W]
	return [np.ascontiguousarray(arr[i]) for i in range(arr.shape[0])]


def u8_to_mask_batch(frames: list) -> torch.Tensor:
	"""Convert a list of single-channel ndarrays to a ComfyUI MASK batch."""
	if len(frames) == 0:  # empty batch (e.g. no regions found) - keep it flowing
		return torch.zeros((0, 1, 1))
	out = []
	for frame in frames:
		frame = ensure_uint8(frame)
		if frame.ndim == 3:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.shape[2] >= 3 else frame[..., 0]
		out.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
	return torch.stack(out)


def broadcast_batches(*tensors_as_lists):
	"""Zip per-image lists, repeating length-1 lists to match the longest."""
	n = max(len(lst) for lst in tensors_as_lists)
	for lst in tensors_as_lists:
		if len(lst) not in (1, n):
			raise ValueError(
				f"Batch sizes do not match and cannot be broadcast: "
				f"{[len(lst) for lst in tensors_as_lists]}"
			)
	return zip(*[lst * n if len(lst) == 1 else lst for lst in tensors_as_lists])


def fit_mask_to(mask_u8: np.ndarray, target_hw: tuple) -> np.ndarray:
	"""Resize a uint8 mask to the target (height, width) if needed."""
	h, w = target_hw
	if mask_u8.shape[:2] != (h, w):
		mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
	return mask_u8


def odd(ksize: int) -> int:
	"""Kernels for blur/threshold ops must be odd; round up instead of failing."""
	ksize = max(1, int(ksize))
	return ksize if ksize % 2 == 1 else ksize + 1


# ---------------------------------------------------------------------------
# categorical colours (one class/label -> one colour)
# ---------------------------------------------------------------------------

# Okabe-Ito, the standard colour-vision-deficiency-safe categorical set. Every
# pair stays separable under protanopia, deuteranopia and tritanopia, and the
# set spreads over LUMINANCE too, so it survives greyscale and a 1-channel
# canvas. One home for the table: `plots.py` offers it as a chart palette, the
# drawing nodes step through it via `categorical_bgr`.
OKABE_ITO_HEX = ("#56b4e9", "#e69f00", "#009e73", "#d55e00",
                 "#cc79a7", "#0072b2", "#f0e442", "#999999")


def hex_to_bgr(text: str) -> tuple:
	"""'#rrggbb' -> (b, g, r) ints."""
	text = text.lstrip("#")
	r, g, b = (int(text[i:i + 2], 16) for i in (0, 2, 4))
	return (b, g, r)


OKABE_ITO_BGR = tuple(hex_to_bgr(h) for h in OKABE_ITO_HEX)

# Past the 8 base colours the palette repeats, so indices 8..15 come back
# darkened and 16..23 washed towards white: 24 marks stay distinguishable
# instead of 3 exact repeats. Beyond 24 no categorical palette is readable
# anyway and the cycle simply restarts.
_TIER_SCALE = (1.0, 0.55, 1.0)
_TIER_WHITE = (0.0, 0.0, 0.55)
_TIERS = len(_TIER_SCALE)


def categorical_bgr(index: int) -> tuple:
	"""Deterministic, CVD-safe BGR colour for a class/label index."""
	index = int(index) % (len(OKABE_ITO_BGR) * _TIERS)
	base = OKABE_ITO_BGR[index % len(OKABE_ITO_BGR)]
	scale = _TIER_SCALE[index // len(OKABE_ITO_BGR)]
	white = _TIER_WHITE[index // len(OKABE_ITO_BGR)]
	out = []
	for c in base:
		v = c * scale
		out.append(int(round(min(255.0, max(0.0, v + (255.0 - v) * white)))))
	return tuple(out)


def categorical_gray(index: int) -> int:
	"""Deterministic grey level per index, for a 1-channel canvas.

	A hue palette carries no information on a single channel, so indices walk a
	van der Corput (bit-reversed base-2) sequence over 48..255 instead:
	consecutive indices land at opposite ends of the range and no two of the
	first 32 collide. 48 is the floor so index 0 is still visible on black.
	"""
	i, frac, denom = int(index) % 64 + 1, 0.0, 1.0
	while i:
		denom *= 2.0
		frac += (i % 2) / denom
		i //= 2
	return int(round(48 + frac * (255 - 48)))


def _fmt_scale(v: float) -> str:
	"""Compact tick label: plain integer when whole, else 3 significant figures."""
	v = float(v)
	if abs(v - round(v)) < 1e-6 and abs(v) < 1e6:
		return str(int(round(v)))
	return f"{v:.3g}"


SCALE_FONT, SCALE_FSCALE, SCALE_FTHICK = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1


def _scale_labels(vmin: float, vmax: float, ticks: int) -> list:
	"""Tick labels of a colour-scale bar, top (vmax) to bottom (vmin)."""
	return [_fmt_scale(vmax - (vmax - vmin) * (i / (ticks - 1) if ticks > 1 else 0))
	        for i in range(ticks)]


def color_scale_label_width(vmin: float, vmax: float, ticks: int = 5) -> int:
	"""Pixel width the tick labels of `attach_color_scale` would need.

	The only part of a colour-scale canvas whose width depends on the DATA - so
	a batch renders to one common width by taking the max of this over the
	frames and passing it back as `label_width`.
	"""
	labels = _scale_labels(vmin, vmax, ticks)
	return max(cv2.getTextSize(t, SCALE_FONT, SCALE_FSCALE, SCALE_FTHICK)[0][0]
	           for t in labels)


def attach_color_scale(frame: np.ndarray, vmin: float, vmax: float, colormap=None, *,
                       min_height: int = 160, bar_width: int = 26, gap: int = 10,
                       pad: int = 8, ticks: int = 5, label_width: int = 0) -> np.ndarray:
	"""Composite `frame` (uint8, gray or BGR) with a labelled vertical colour-scale
	bar to its right; returns a BGR uint8 image.

	The bar maps `vmin` (bottom) .. `vmax` (top) through `colormap` (an OpenCV
	COLORMAP_* int, or None for a grayscale ramp), matching how the caller
	rendered `frame`. The canvas height is max(frame height, `min_height`) so the
	bar and its tick labels stay legible even when `frame` is tiny (e.g. a single
	pixel) - the frame is then centred vertically beside a full-height bar.

	`label_width` reserves at least that many pixels for the tick labels (the
	surplus stays black on the right edge); pass the batch-wide maximum from
	`color_scale_label_width` to give every frame of a batch the same width.
	"""
	frame = ensure_uint8(frame)
	if frame.ndim == 2:
		frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
	elif frame.shape[2] == 1:
		frame = cv2.cvtColor(frame[..., 0], cv2.COLOR_GRAY2BGR)
	elif frame.shape[2] == 4:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
	fh, fw = frame.shape[:2]

	canvas_h = max(fh, min_height)
	bar_top = pad
	bar_h = max(1, canvas_h - 2 * pad)

	# vertical gradient, top = vmax (255) down to bottom = vmin (0)
	ramp = np.linspace(255, 0, bar_h).round().astype(np.uint8).reshape(-1, 1)
	bar = np.repeat(ramp, bar_width, axis=1)
	bar = (cv2.applyColorMap(bar, colormap) if colormap is not None
	       else cv2.cvtColor(bar, cv2.COLOR_GRAY2BGR))

	font, fscale, fthick = SCALE_FONT, SCALE_FSCALE, SCALE_FTHICK
	labels = _scale_labels(vmin, vmax, ticks)
	label_w = max(color_scale_label_width(vmin, vmax, ticks), int(label_width))
	tick_len, text_pad = 4, 4

	canvas_w = fw + gap + bar_width + tick_len + text_pad + label_w + pad
	canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
	y0 = (canvas_h - fh) // 2
	canvas[y0:y0 + fh, :fw] = frame

	bx = fw + gap
	canvas[bar_top:bar_top + bar_h, bx:bx + bar_width] = bar
	cv2.rectangle(canvas, (bx, bar_top), (bx + bar_width - 1, bar_top + bar_h - 1),
	              (255, 255, 255), 1)

	tx = bx + bar_width + tick_len + text_pad
	for i, text in enumerate(labels):
		frac = i / (ticks - 1) if ticks > 1 else 0
		ty = bar_top + int(round(frac * (bar_h - 1)))
		cv2.line(canvas, (bx + bar_width, ty), (bx + bar_width + tick_len, ty),
		         (255, 255, 255), 1)
		(_, th), _ = cv2.getTextSize(text, font, fscale, fthick)
		text_y = min(max(ty + th // 2, th), canvas_h - 1)
		cv2.putText(canvas, text, (tx, text_y), font, fscale, (255, 255, 255),
		            fthick, cv2.LINE_AA)
	return canvas


def progress_preview(frame, max_side: int = 384):
	"""("JPEG", PIL image, max_side) payload for ProgressBar.update_absolute().

	Downscales with cv2 BEFORE handing the frame over, because the server
	re-encodes whatever it receives - shrinking here is what keeps a preview of a
	4K frame cheap. Accepts uint8 or float BGR/gray; returns None for anything
	that cannot be rendered sensibly (empty, or a many-channel latent), and the
	caller then just ticks the bar without a preview.
	"""
	from PIL import Image  # ComfyUI core dependency, needed nowhere else here

	arr = np.asarray(frame)
	if arr.ndim not in (2, 3) or arr.size == 0:
		return None
	if arr.ndim == 3:
		c = arr.shape[2]
		if c == 1:
			arr = arr[:, :, 0]
		elif c == 3 or c == 4:
			arr = arr[:, :, :3]
		else:  # 2-channel, or a many-channel latent: no meaningful RGB reading
			return None

	if arr.dtype != np.uint8:
		arr = np.asarray(arr, np.float32)
		hi = float(arr.max()) if arr.size else 0.0
		arr = arr * 255.0 if hi <= 1.0 else arr
		arr = np.clip(arr, 0, 255).astype(np.uint8)

	h, w = arr.shape[:2]
	scale = max_side / float(max(h, w))
	if scale < 1.0:
		arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))),
		                 interpolation=cv2.INTER_AREA)

	if arr.ndim == 2:
		return ("JPEG", Image.fromarray(arr, "L"), max_side)
	rgb = np.ascontiguousarray(arr[:, :, ::-1])  # BGR -> RGB
	return ("JPEG", Image.fromarray(rgb, "RGB"), max_side)
