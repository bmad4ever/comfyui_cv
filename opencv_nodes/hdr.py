# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""HDR / exposure-bracketing and alignment nodes.

Multi-exposure imaging: shoot the SAME scene at several shutter speeds, stack
the frames into one IMAGE batch (core 'Batch Images'), then either

* align hand-held brackets with 'Align MTB' before fusion,
* fuse them straight into a display-ready picture with 'HDR Exposure Fusion
  (Mertens)' - no exposure times, no tonemapping, or
* recover a float RADIANCE map with 'HDR Merge to Radiance' (needs the shutter
  time of each frame) and compress it to a displayable image with 'HDR
  Tonemap'. The radiance map holds linear-light values well past 1.0, so it is
  NOT directly displayable - inspect it with 'Preview CV Array'
  (normalize/heatmap).

Every merge reads the batch frame-by-frame in the order it was stacked;
'Merge to Radiance' expects exposure_times in that SAME order. The frames must
already be registered: tripod brackets (like the bundled St. Louis set) are
aligned by construction; for hand-held shots, register them upstream (e.g. the
ECC image-registration exercise or 'Align MTB') before stacking.

These wrap cv2's stateful cv2.create* HDR factories (createMergeMertens /
createCalibrate* / createMerge* / createTonemap* / createAlignMTB), which the
generated low-level wrappers cannot expose.
"""

import re

import cv2
import numpy as np
import torch

import comfy.utils
from comfy_api.latest import io

from . import imageutils, polytype
from .convert import NPArray

CATEGORY = "image/CV/hdr"


def _exposures_bgr(image) -> list:
	"""ComfyUI IMAGE batch -> list of uint8 BGR frames (the LDR exposures)."""
	frames = imageutils.image_batch_to_bgr(image)
	if len(frames) < 2:
		raise ValueError(
			f"HDR needs a batch of at least 2 exposures, got {len(frames)}. "
			"Stack your bracketed shots into one batch with core 'Batch Images' "
			"first.")
	return frames


def _ldr_to_image(bgr_float: np.ndarray) -> torch.Tensor:
	"""float32 BGR (~0..1, operators can over/undershoot) -> IMAGE [1,H,W,3].

	Tonemap operators divide by local contrast, so a perfectly flat region
	(e.g. a clear sky) can come back NaN in the embedded cv2 build - sanitize
	before clipping so the node always emits a valid image.
	"""
	bgr_float = np.nan_to_num(np.asarray(bgr_float, np.float32), nan=0.0,
	                          posinf=1.0, neginf=0.0)
	rgb = np.clip(bgr_float, 0.0, 1.0)[..., ::-1]
	return torch.from_numpy(np.ascontiguousarray(rgb))[None]


def _parse_times(text, n) -> np.ndarray:
	parts = [p for p in re.split(r"[,\s]+", str(text).strip()) if p]
	try:
		times = [float(p) for p in parts]
	except ValueError:
		raise ValueError(
			f"exposure_times must be numbers, got {text!r}. Give one shutter "
			"time (seconds) per exposure, e.g. '0.033, 0.25, 2.5, 15'.")
	if len(times) != n:
		raise ValueError(
			f"exposure_times has {len(times)} value(s) but the batch holds {n} "
			"exposure(s) - give exactly one shutter time per frame, in the same "
			"order they were stacked.")
	if any(t <= 0 for t in times):
		raise ValueError("exposure_times must be positive (seconds).")
	return np.asarray(times, np.float32)


class HDRMergeMertens(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_HDRMergeMertens",
			display_name="CV HDR Exposure Fusion (Mertens)",
			category=CATEGORY,
			description="Blends a batch of exposures straight into one display-"
			            "ready image, keeping each region from whichever frame "
			            "exposed it best (well-exposedness x contrast x "
			            "saturation). Needs no exposure times and no separate "
			            "tonemapping - the simplest, most robust HDR-look path. "
			            "Output is an ordinary 0..1 IMAGE.",
			inputs=[
				io.Image.Input("image",
					tooltip="Batch of registered exposures of the same scene "
					        "(tripod, or pre-aligned)."),
				io.Float.Input("contrast_weight", default=1.0, min=0.0, max=5.0, step=0.1,
					tooltip="Emphasis on locally contrasty (in-focus, textured) "
					        "pixels. 0 ignores contrast."),
				io.Float.Input("saturation_weight", default=1.0, min=0.0, max=5.0, step=0.1,
					tooltip="Emphasis on saturated (vivid) pixels."),
				io.Float.Input("exposure_weight", default=1.0, min=0.0, max=5.0, step=0.1,
					tooltip="Emphasis on well-exposed (mid-tone) pixels - the "
					        "heart of the fusion; low values let blown highlights "
					        "and crushed shadows leak in."),
			],
			outputs=[io.Image.Output(display_name="fused",
				tooltip="Display-ready fused image (0..1 IMAGE).")],
		)

	@classmethod
	def execute(cls, image, contrast_weight, saturation_weight, exposure_weight) -> io.NodeOutput:
		frames = _exposures_bgr(image)
		merger = cv2.createMergeMertens(float(contrast_weight),
		                                float(saturation_weight),
		                                float(exposure_weight))
		fused = merger.process(frames)
		return io.NodeOutput(_ldr_to_image(fused))


class HDRMergeToRadiance(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_HDRMergeToRadiance",
			display_name="CV HDR Merge to Radiance",
			category=CATEGORY,
			description="Recovers the camera response curve from the bracketed "
			            "exposures and their shutter times, then fuses a single "
			            "float RADIANCE map (linear light, values well past "
			            "1.0). This is a raw CV array, NOT a displayable image - "
			            "view it with 'Preview CV Array' (normalize/heatmap) and "
			            "turn it into a picture with 'HDR Tonemap'.",
			inputs=[
				io.Image.Input("image",
					tooltip="Batch of registered exposures of the same scene. "
					        "Frame order MUST match exposure_times."),
				io.String.Input("exposure_times", default="0.033, 0.25, 2.5, 15.0",
					tooltip="Comma-separated shutter time (seconds) of each "
					        "frame, in the SAME order they were stacked. Only "
					        "the ratios matter, so relative values (e.g. 2^EV) "
					        "work too. Must be one value per exposure."),
				io.Combo.Input("method", options=["Debevec", "Robertson"], default="Debevec",
					tooltip="Response-calibration + merge algorithm. Debevec is "
					        "robust and the usual choice; Robertson can wring out "
					        "a little more range but is noisier and less stable."),
				io.Int.Input("samples", default=70, min=10, max=2000,
					tooltip="Debevec only: number of pixel locations sampled to "
					        "fit the response curve. More = steadier fit, slower.",
					optional=True, advanced=True),
			],
			outputs=[NPArray.Output(display_name="radiance",
				tooltip="HxWx3 float32 linear-radiance map (BGR). Feed it to "
				        "'HDR Tonemap'; preview it with 'Preview CV Array'.")],
		)

	@classmethod
	def execute(cls, image, exposure_times, method, samples=70) -> io.NodeOutput:
		frames = _exposures_bgr(image)
		times = _parse_times(exposure_times, len(frames))
		if method == "Robertson":
			crf = cv2.createCalibrateRobertson().process(frames, times)
			hdr = cv2.createMergeRobertson().process(frames, times, crf)
		else:
			crf = cv2.createCalibrateDebevec(int(samples)).process(frames, times)
			hdr = cv2.createMergeDebevec().process(frames, times, crf)
		# Robertson in particular can emit NaN/Inf on degenerate brackets; keep
		# the map finite so downstream tonemapping/preview never collapses.
		hdr = np.nan_to_num(np.asarray(hdr, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
		return io.NodeOutput(hdr)


class HDRTonemap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_HDRTonemap",
			display_name="CV HDR Tonemap",
			category=CATEGORY,
			description="Compresses a float radiance map (from 'HDR Merge to "
			            "Radiance') into a displayable 0..1 image. 'linear' only "
			            "gamma-corrects; Drago/Reinhard/Mantiuk are tone-mapping "
			            "operators that keep detail in both shadows and "
			            "highlights. The method-specific knobs are advanced "
			            "inputs - each is used only by its own method.",
			inputs=[
				NPArray.Input("radiance",
					tooltip="HxWx3 float32 radiance map from 'HDR Merge to "
					        "Radiance'."),
				io.Combo.Input("method",
					options=["linear", "Drago", "Reinhard", "Mantiuk"],
					default="Reinhard",
					tooltip="Tonemapping operator. linear = plain gamma; Reinhard "
					        "is a safe default; Drago suits very bright scenes; "
					        "Mantiuk is contrast-domain."),
				io.Float.Input("gamma", default=2.2, min=0.1, max=5.0, step=0.1,
					tooltip="Gamma applied by every operator. 1.0 = linear; 2.2 "
					        "matches a typical display; higher brightens midtones."),
				io.Float.Input("saturation", default=1.0, min=0.0, max=3.0, step=0.05,
					tooltip="Drago/Mantiuk only: color saturation of the result.",
					optional=True, advanced=True),
				io.Float.Input("bias", default=0.85, min=0.0, max=1.0, step=0.01,
					tooltip="Drago only: bias of the tone curve (0.7-0.9 typical).",
					optional=True, advanced=True),
				io.Float.Input("intensity", default=0.0, min=-8.0, max=8.0, step=0.1,
					tooltip="Reinhard only: overall brightness (- darker, + brighter).",
					optional=True, advanced=True),
				io.Float.Input("light_adapt", default=1.0, min=0.0, max=1.0, step=0.05,
					tooltip="Reinhard only: 1 = fully local adaptation, 0 = global.",
					optional=True, advanced=True),
				io.Float.Input("color_adapt", default=0.0, min=0.0, max=1.0, step=0.05,
					tooltip="Reinhard only: 1 = per-channel (auto white-balance) "
					        "adaptation, 0 = shared across channels.",
					optional=True, advanced=True),
				io.Float.Input("scale", default=0.7, min=0.0, max=2.0, step=0.05,
					tooltip="Mantiuk only: contrast scale factor (0.6-0.9 typical).",
					optional=True, advanced=True),
			],
			outputs=[io.Image.Output(display_name="image",
				tooltip="Tonemapped display image (0..1 IMAGE).")],
		)

	@classmethod
	def execute(cls, radiance, method, gamma, saturation=1.0, bias=0.85,
	            intensity=0.0, light_adapt=1.0, color_adapt=0.0, scale=0.7) -> io.NodeOutput:
		hdr = np.asarray(radiance, np.float32)
		if hdr.ndim != 3 or hdr.shape[2] != 3:
			raise ValueError(
				f"radiance must be an HxWx3 float map, got shape {hdr.shape}. "
				"It must come from 'HDR Merge to Radiance'.")
		hdr = np.ascontiguousarray(np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0))
		g = float(gamma)
		if method == "Drago":
			tm = cv2.createTonemapDrago(g, float(saturation), float(bias))
		elif method == "Mantiuk":
			tm = cv2.createTonemapMantiuk(g, float(scale), float(saturation))
		elif method == "Reinhard":
			tm = cv2.createTonemapReinhard(g, float(intensity), float(light_adapt),
			                               float(color_adapt))
		else:
			tm = cv2.createTonemap(g)
		return io.NodeOutput(_ldr_to_image(tm.process(hdr)))


# ---------------------------------------------------------------------------
# AlignMTB: align a batch of images via median threshold bitmaps
# ---------------------------------------------------------------------------

def _gray_for_shift(frame):
	"""Convert to grayscale if 3+-channel; pass through single-channel."""
	if frame.ndim == 3 and frame.shape[2] >= 3:
		return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	return frame


def _resolve_mtb_batch(value):
	"""Extract list of ndarrays from any image-ish batch input.

	Returns (frames, tag) where tag is 'image', 'mask', 'latent' or 'nparray'.
	"""
	if isinstance(value, torch.Tensor):
		if value.ndim == 4:
			return imageutils.image_batch_to_bgr(value), "image"
		if value.ndim in (2, 3):
			return imageutils.mask_batch_to_u8(value), "mask"
		raise ValueError(
			f"Cannot process tensor of shape {tuple(value.shape)} - expected "
			"4-D [B,H,W,C] (IMAGE) or 3-D [B,H,W] (MASK).")
	if polytype.is_latent(value):
		samples = value["samples"]
		if samples.ndim != 4:
			raise ValueError(
				f"Only 4-D [B,C,H,W] latents are supported, got "
				f"{tuple(samples.shape)}.")
		frames = [
			np.ascontiguousarray(
				samples[b].movedim(0, -1).float().cpu().numpy())
			for b in range(samples.shape[0])]
		return frames, "latent"
	if isinstance(value, (list, tuple)):
		return [np.asarray(v) for v in value], "nparray"
	arr = np.asarray(value)
	if arr.ndim == 4:
		return [np.ascontiguousarray(arr[i]) for i in range(arr.shape[0])], "nparray"
	return [arr], "nparray"


def _restore_mtb_batch(frames, tag):
	"""Convert a list of aligned ndarrays back to the original input type."""
	if tag == "image":
		return imageutils.bgr_to_image_batch([np.asarray(f) for f in frames])
	if tag == "mask":
		return imageutils.u8_to_mask_batch([np.asarray(f) for f in frames])
	if tag == "latent":
		tensors = []
		for f in frames:
			f = np.asarray(f)
			if f.ndim == 2:
				f = f[:, :, None]
			tensors.append(torch.from_numpy(np.ascontiguousarray(f)).movedim(-1, 0))
		return {"samples": torch.stack(tensors)}
	if len(frames) == 1:
		return frames[0]
	return list(frames)


def _mtb_ensure_2plus(n, name):
	if n < 2:
		raise ValueError(
			f"{name} needs at least 2 images to align, got {n}. "
			"Stack your bracketed shots into a batch with core "
			"'Batch Images' first.")


class AlignMTB(io.ComfyNode):
	"""Aligns a batch of images via cv2.AlignMTB.process().

	Wraps the OpenCV process function directly.  Accepts IMAGE /
	MASK / NPARRAY / LATENT batches and returns aligned images in
	the same format.
	"""

	@classmethod
	def define_schema(cls):
		template = polytype.match_template(types=polytype.with_latent(polytype.IMAGEISH))
		return io.Schema(
			node_id="CV_AlignMTB",
			display_name="CV Align MTB",
			category=CATEGORY,
			description="Aligns a batch of images using the Median Threshold "
			            "Bitmap algorithm (cv2.AlignMTB.process).  Each frame "
			            "is globally shifted to register with the others, "
			            "correcting small camera motions between bracketed "
			            "exposures before HDR fusion.  Connect a batch of 2+ "
			            "frames and get aligned images back in the same format "
			            "(IMAGE/MASK/NPARRAY/LATENT).",
			inputs=[
				polytype.match_input("image", template,
					tooltip="Batch of 2+ images to align.  Accepts IMAGE, "
					        "MASK, NPARRAY (list or 4-D array), or LATENT."),
				io.Int.Input("max_bits", default=6, min=1, max=32,
					tooltip="Maximum bit depth for the median bitmaps.  6 is "
					        "typical; higher captures finer intensity variation "
					        "but also more noise."),
				io.Int.Input("exclude_range", default=4, min=0, max=128,
					tooltip="Half-width of the pixel-value range excluded from "
					        "the median threshold bitmap.  Helps reject noise."),
				io.Boolean.Input("cut", default=True,
					tooltip="Exclude the darkest and brightest pixels from the "
					        "median threshold bitmap."),
			],
			outputs=[polytype.match_output(template, "image",
				tooltip="Aligned images in the same format as the input.")],
		)

	@classmethod
	def execute(cls, image, max_bits, exclude_range, cut) -> io.NodeOutput:
		frames, tag = _resolve_mtb_batch(image)
		_mtb_ensure_2plus(len(frames), "AlignMTB")
		aligner = cv2.createAlignMTB(int(max_bits), int(exclude_range), bool(cut))
		# process() requires 3-channel uint8; convert non-3ch frames
		was_converted = [f.ndim != 3 or f.shape[2] < 3 for f in frames]
		proc = []
		for f in frames:
			if f.ndim == 2:
				proc.append(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
			elif f.ndim == 3 and f.shape[2] >= 3 and f.dtype == np.uint8:
				proc.append(f)
			else:
				# float or nonstandard: quantize to uint8
				u8 = (np.clip(f, 0.0, 1.0) * 255).astype(np.uint8) if f.dtype != np.uint8 else f
				proc.append(np.ascontiguousarray(u8))
		dst = [np.empty_like(f) for f in proc]
		aligner.process(proc, dst)
		# In this OpenCV build, process() zeros the output slot of the
		# internal reference frame.  Detect and restore it.
		for i, (p, d) in enumerate(zip(proc, dst)):
			if d.min() == 0 and d.max() == 0:
				dst[i] = p.copy()
		# restore single-channel frames that were expanded to 3ch
		for i, wc in enumerate(was_converted):
			if wc:
				dst[i] = cv2.cvtColor(dst[i], cv2.COLOR_BGR2GRAY)
		return io.NodeOutput(_restore_mtb_batch(dst, tag))


class AlignMTBToReference(io.ComfyNode):
	"""Aligns each frame to a chosen reference using
	cv2.AlignMTB.calculateShift() + shiftMat().

	Unlike process(), which jointly optimises all frames, this
	node aligns every frame independently to one reference
	frame.  The reference passes through unmodified; every other
	frame gets a global shift so it best matches the reference's
	median threshold bitmap.
	"""

	@classmethod
	def define_schema(cls):
		template = polytype.match_template(types=polytype.with_latent(polytype.IMAGEISH))
		return io.Schema(
			node_id="CV_AlignMTBToReference",
			display_name="CV Align MTB To Reference",
			category=CATEGORY,
			description="Aligns a batch of images to a chosen reference frame "
			            "using cv2.AlignMTB.calculateShift() + shiftMat().  "
			            "The reference frame passes through unchanged; every "
			            "other frame is globally shifted to best match the "
			            "reference's median threshold bitmap.  The output "
			            "matches the input format "
			            "(IMAGE/MASK/NPARRAY/LATENT).",
			inputs=[
				polytype.match_input("image", template,
					tooltip="Batch of 2+ images to align.  Accepts IMAGE, "
					        "MASK, NPARRAY (list or 4-D array), or LATENT."),
				io.Int.Input("reference_index", default=0, min=0,
					tooltip="Index of the frame to use as the alignment "
					        "reference (0-based).  This frame is not shifted; "
					        "all others are shifted to match it."),
				io.Int.Input("max_bits", default=4, min=1, max=32,
					tooltip="Bit depth for the median bitmaps used by "
					        "calculateShift.  Values 2-4 give reliable "
					        "results in this OpenCV build; higher values "
					        "may return incorrect shifts on some inputs."),
				io.Int.Input("exclude_range", default=4, min=0, max=128,
					tooltip="Half-width of the pixel-value range excluded "
					        "from the median threshold bitmap."),
				io.Boolean.Input("cut", default=True,
					tooltip="Exclude the darkest and brightest pixels from "
					        "the median threshold bitmap."),
			],
			outputs=[polytype.match_output(template, "image",
				tooltip="Aligned images in the same format as the input.")],
		)

	@classmethod
	def execute(cls, image, reference_index, max_bits, exclude_range,
	            cut) -> io.NodeOutput:
		frames, tag = _resolve_mtb_batch(image)
		_mtb_ensure_2plus(len(frames), "AlignMTBToReference")
		if not 0 <= reference_index < len(frames):
			raise ValueError(
				f"reference_index {reference_index} is out of range "
				f"for a batch of {len(frames)} frames.")

		# calculateShift works reliably only at lower bit depths in this build,
		# and requires uint8 input (not float32).
		max_bits = max(1, min(int(max_bits), 4))
		aligner = cv2.createAlignMTB(max_bits, int(exclude_range), bool(cut))

		def _u8_gray(f):
			g = _gray_for_shift(f)
			if g.dtype != np.uint8:
				g = np.clip(g * 255.0, 0, 255).astype(np.uint8)
			return np.ascontiguousarray(g)

		ref_gray = _u8_gray(frames[reference_index])

		aligned = list(frames)
		pbar = comfy.utils.ProgressBar(len(frames))
		# a preview send skips the bar's own throttle - keep them to ~20 a run
		preview_every = max(1, len(frames) // 20)
		for i, f in enumerate(frames):
			# the reference does no work but still ticks, so the bar reaches 100%
			if i != reference_index:
				gray = _u8_gray(f)
				shift = aligner.calculateShift(ref_gray, gray)
				if shift != (0, 0):
					aligned[i] = aligner.shiftMat(f, shift)
			if i % preview_every == 0 or i == len(frames) - 1:
				pbar.update_absolute(i + 1, None,
				                     imageutils.progress_preview(aligned[i]))
			else:
				pbar.update(1)

		return io.NodeOutput(_restore_mtb_batch(aligned, tag))


NODES = [AlignMTB, AlignMTBToReference, HDRMergeMertens, HDRMergeToRadiance, HDRTonemap]
