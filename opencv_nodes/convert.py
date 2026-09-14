# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# Adapted from geroldmeisinger/opencv-comfyui (GPL-3.0).
# See LICENSE for the full GNU GPL v3 text.

"""Conversion nodes between ComfyUI tensors and raw OpenCV ndarrays (NPARRAY).

These are only needed for the low-level cv2.* nodes. The high-level nodes in
highlevel.py work on IMAGE/MASK directly.
"""

import json
import logging
import os
import shutil
from struct import error as StructError

import cv2
import numpy as np
import torch

from comfy_api.latest import io, ui

from . import buildinfo, enums, imageutils, render3d, scalars

NPArray = io.Custom("NPARRAY")

# imported after NPArray exists: polytype and tuples both pull NPArray from this
# module, so a top-of-file import would be circular
from . import polytype  # noqa: E402
from . import tuples  # noqa: E402

CATEGORY = "image/CV/low-level"
INT32_MAX = 2**31 - 1


class ImageToCV(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ImageToCV",
			display_name="Image → CV Array",
			category=CATEGORY,
			description="Converts a ComfyUI IMAGE to a raw OpenCV ndarray (NPARRAY) "
			            "for use with the low-level cv2.* nodes.",
			inputs=[
				io.Image.Input("image",
					tooltip="IMAGE to convert. Only one frame is taken (see "
					        "batch_index) since raw cv2 nodes work on single images."),
				io.Combo.Input(
					"color_format",
					options=["BGR (OpenCV default)", "RGB", "GRAY"],
					default="BGR (OpenCV default)",
					tooltip="Most cv2 functions expect BGR; functions like Canny or "
					        "threshold require a single-channel GRAY image.",
				),
				io.Combo.Input(
					"dtype",
					options=["uint8 (0-255)", "float32 (0-1)"],
					default="uint8 (0-255)",
					tooltip="Most cv2 functions expect uint8.",
				),
				io.Int.Input(
					"batch_index", default=0, min=0, max=4095,
					tooltip="Which image of the batch to convert (raw cv2 functions "
					        "operate on single images). Clamped to the batch size.",
				),
			],
			outputs=[NPArray.Output(display_name="nparray")],
		)

	@classmethod
	def execute(cls, image, color_format, dtype, batch_index) -> io.NodeOutput:
		frames = imageutils.image_batch_to_bgr(image)
		frame = frames[min(batch_index, len(frames) - 1)]
		if color_format == "RGB":
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		elif color_format == "GRAY":
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if dtype.startswith("float32"):
			frame = frame.astype(np.float32) / 255.0
		return io.NodeOutput(frame)


class ImageBatchToCV(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ImageBatchToCV",
			display_name="Image Batch \u2192 CV Batch",
			category=CATEGORY,
			description="Converts an entire ComfyUI IMAGE/MASK/LATENT batch to a "
			            "single batched NPARRAY (e.g. [B,H,W,C]) so every frame "
			            "can be processed in one cv2 call without per-frame "
			            "looping.  IMAGE becomes [B,H,W,C] BGR (float32 by "
			            "default to avoid uint8 quantisation and overflow "
			            "clamping); LATENT becomes [B,C,H,W] float32 with "
			            "values untouched; MASK becomes [B,H,W] float32 "
			            "0..1.  Pair with 'CV Batch \u2192 Image' (or "
			            "'Latent \u2192 CV Array') to convert back.  The "
			            "output is a RAW NPARRAY \u2014 no automatic type "
			            "preservation.",
			inputs=[
				io.MultiType.Input("input",
					[io.Image, io.Mask, io.Latent, NPArray],
					tooltip="IMAGE, MASK, LATENT or NPARRAY batch to convert.  "
					        "The entire batch is extracted (no batch_index)."),
				io.Combo.Input(
					"dtype",
					options=["float32 (0-1, no quantisation)",
					         "uint8 (0-255, standard cv2)"],
					default="float32 (0-1, no quantisation)",
					tooltip="For IMAGE/MASK: float32 keeps the original "
					        "precision (no [0,1]\u2192[0,255] quantisation "
					        "and no overflow clamping on add/multiply); "
					        "uint8 matches the standard cv2 path.  For "
					        "LATENT this is ignored (always float32).  "
					        "For NPARRAY the value passes through as-is."),
			],
			outputs=[NPArray.Output(display_name="batch",
				tooltip="Batched ndarray: [B,H,W,C] from IMAGE, [B,H,W] "
				        "from MASK, [B,C,H,W] from LATENT, or the NPARRAY "
				        "as-is.")],
		)

	@classmethod
	def execute(cls, input, dtype) -> io.NodeOutput:
		want_float32 = dtype.startswith("float32")
		if polytype.is_latent(input):
			samples = input["samples"]
			if samples.ndim != 4:
				raise ValueError(
					f"Only 4-D [B,C,H,W] latents are supported, got shape "
					f"{tuple(samples.shape)}.")
			return io.NodeOutput(samples.detach().float().cpu().numpy().copy())
		if isinstance(input, torch.Tensor):
			if input.ndim == 4:
				# IMAGE batch [B,H,W,C] -> [B,H,W,C] BGR
				frames = imageutils.image_batch_to_bgr(input)
				arr = np.stack(frames, axis=0)
				if want_float32:
					arr = arr.astype(np.float32) / 255.0
				return io.NodeOutput(arr)
			if input.ndim in (2, 3):
				# MASK [B,H,W] or [H,W] -> [B,H,W] float32 0..1 or uint8
				frames = imageutils.mask_batch_to_u8(input)
				arr = np.stack(frames, axis=0)
				if want_float32:
					arr = arr.astype(np.float32) / 255.0
				return io.NodeOutput(arr)
		# NPARRAY: pass through unchanged
		return io.NodeOutput(np.asarray(input))


class CVBatchToImage(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CVBatchToImage",
			display_name="CV Batch \u2192 Image Batch",
			category=CATEGORY,
			description="Converts a batched NPARRAY ([B,H,W,C] or [B,H,W]) "
			            "back to a ComfyUI IMAGE batch.  The inverse of "
			            "'Image Batch \u2192 CV Batch'.  For BGR uint8 input "
			            "(the standard cv2 path) the default settings work "
			            "directly.  'RGB' keeps the channel order for data "
			            "that is already RGB-ordered; both BGR and RGB "
			            "round-trip through uint8.  'RAW' converts nothing "
			            "and passes values as-is (for HDR or out-of-range "
			            "data).  4-channel BGRA input yields an IMAGE batch "
			            "plus an alpha MASK batch.  A single [H,W] mask is "
			            "wrapped to a one-frame batch; a 3-D [H,W,C] single "
			            "frame is read as a [B,H,W] gray batch \u2014 use "
			            "'CV Array \u2192 Image' for single color frames.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="Batched ndarray: [B,H,W,C] or [B,H,W].  "
					        "A single [H,W] mask is wrapped to a "
					        "one-frame batch; a 3-D [H,W,C] single frame "
					        "is read as a [B,H,W] gray batch \u2014 "
					        "use 'CV Array \u2192 Image' for single "
					        "color frames."),
				io.Combo.Input(
					"channel_order",
					options=["BGR (OpenCV default)", "RGB", "RAW"],
					default="BGR (OpenCV default)",
					tooltip="How to interpret 3/4-channel data.  BGR: "
					        "standard cv2\u2192IMAGE conversion (BGR\u2192RGB, "
					        "uint8\u2192float32 0..1).  RGB: skip channel "
					        "swap, clamp to 0..1 \u2014 for float32 BGR from "
					        "cv2 arithmetic where values stay in range.  "
					        "RAW: no conversion at all, values as-is \u2014 "
					        "for HDR or arithmetic results outside 0..1."),
			],
			outputs=[
				io.Image.Output(display_name="image",
					tooltip="IMAGE batch [B,H,W,C] float32 RGB 0..1 "
					        "(RAW mode may exceed 0..1)."),
				io.Mask.Output(display_name="alpha",
					tooltip="Alpha MASK batch from BGRA input "
					        "(all-255 opaque when input is RGB/BGR)."),
			],
		)

	@classmethod
	def execute(cls, nparray, channel_order) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.ndim == 2:
			# single mask [H,W] -> wrap as [1,H,W]
			arr = arr[np.newaxis]
		if arr.ndim not in (3, 4):
			raise ValueError(
				f"Expected [B,H,W,C] or [B,H,W], got "
				f"{arr.ndim}-D array of shape {arr.shape}.")
		if arr.ndim == 4 and arr.shape[3] < 3:
			raise ValueError(
				f"Got a 4-D array of shape {arr.shape} \u2014 this looks "
				"like a LATENT batch ([B,C,H,W]).  LATENT channels are "
				"model-defined (4 or 16) and cannot map to an IMAGE.  "
				"Use 'Latent \u2192 CV Array' per frame instead.")
		B = arr.shape[0]
		# --- RAW path: no conversion, just stack as torch -------------------
		if channel_order == "RAW":
			if arr.ndim == 3:
				arr = arr[..., None]  # [B,H,W] -> [B,H,W,1]
			img = torch.from_numpy(np.ascontiguousarray(
				arr.astype(np.float32)))
			alpha = torch.full((B, arr.shape[1], arr.shape[2]),
				255, dtype=torch.float32)
			return io.NodeOutput(img, alpha)
		# --- BGR / RGB path: per-frame ensure_uint8 + channel swap ----------
		frames = list(arr)  # split batch axis -> list of [H,W,C] or [H,W]
		alphas = []
		converted = []
		for frame in frames:
			alpha = None
			if frame.ndim == 2:
				# [H,W] mask: gray->RGB, opaque alpha
				frame = imageutils.ensure_uint8(frame)
				frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
				alpha = np.full(frame.shape[:2], 255, np.uint8)
			elif frame.shape[2] == 4:
				# BGRA: extract alpha, ensure BGR for pipeline
				alpha = frame[..., 3]
				frame = frame[..., :3]
				frame = imageutils.ensure_uint8(frame)
				if channel_order == "RGB":
					# input is RGB->BGR order was already swapped by
					# Image Batch->CV Batch, so treat as BGR
					pass
				# BGR->RGB happens below in the common path
			elif frame.shape[2] == 3:
				frame = imageutils.ensure_uint8(frame)
				if channel_order == "RGB":
					# input is RGB but ensure_uint8 expects raw data;
					# convert to BGR so the BGR2RGB below produces RGB
					frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			# common BGR->RGB conversion for all non-RAW paths
			if frame.ndim == 2:
				frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
			elif frame.shape[2] == 3:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			elif frame.shape[2] == 4:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
			converted.append(torch.from_numpy(
				frame.astype(np.float32) / 255.0))
			if alpha is not None:
				alphas.append(torch.from_numpy(
					imageutils.ensure_uint8(alpha).astype(np.float32)
					/ 255.0))
			else:
				alphas.append(torch.full(converted[-1].shape[:2],
					255.0, dtype=torch.float32))
		return io.NodeOutput(torch.stack(converted), torch.stack(alphas))


class CVToImage(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CVToImage",
			display_name="CV Array → Image",
			category=CATEGORY,
	description="Converts a raw OpenCV ndarray back to a ComfyUI IMAGE. "
		            "Handles gray/BGR/BGRA inputs and uint8/uint16/float dtypes "
		            "automatically. The alpha channel (if any) is also returned "
		            "as a MASK. 'RAW' mode preserves float32 values as-is with no "
		            "uint8 conversion or [0,1] clipping — for HDR radiance maps. "
		            "RAW mode does NOT convert channels: the input must already be "
		            "RGB-order (ComfyUI IMAGE format). Use a cv2.cvtColor node "
		            "upstream for BGR→RGB conversion.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="Gray [H,W], BGR/RGB [H,W,3] or BGRA [H,W,4] ndarray "
					        "(uint8/uint16/float). Use 'Inspect CV Data' if unsure."),
				io.Combo.Input(
					"channel_order",
					options=["BGR (OpenCV default)", "RGB", "RAW"],
					default="BGR (OpenCV default)",
					tooltip="How to interpret 3/4-channel input data. 'RAW' skips "
					        "uint8 conversion and does NOT convert channels — the "
					        "input NPARRAY must already be RGB-order (ComfyUI IMAGE "
					        "convention). Use a cv2.cvtColor upstream for BGR→RGB if "
					        "needed. Use for HDR radiance and other float data where "
					        "[0,1] would destroy information.",
				),
			],
			outputs=[
				io.Image.Output(display_name="image"),
				io.Mask.Output(display_name="alpha"),
			],
		)

	@classmethod
	def execute(cls, nparray, channel_order) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (1, 3, 4)) or arr.size == 0:
			raise ValueError(
				f"Cannot interpret array of shape {arr.shape} (dtype {arr.dtype}) as an image. "
				"Expected [H, W], [H, W, 1], [H, W, 3] or [H, W, 4]. "
				"Not every cv2 output is an image - use 'Inspect CV Data' to check."
			)

		if channel_order == "RAW":
			arr = np.asarray(arr, np.float32)
			if arr.ndim == 2:
				arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
			elif arr.shape[2] == 1:
				arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
			elif arr.shape[2] == 3:
				pass  # channel order is already correct for IMAGE (RGB)
			elif arr.shape[2] == 4:
				arr = arr[..., :3].copy()  # drop alpha, keep RGB as-is
			image = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
			mask = imageutils.u8_to_mask_batch([np.full(arr.shape[:2], 255, dtype=np.uint8)])
			return io.NodeOutput(image, mask)

		arr = imageutils.ensure_uint8(arr)

		alpha = None
		if arr.ndim == 3 and arr.shape[2] == 4:
			alpha = arr[..., 3]
			arr = arr[..., :3]
		if arr.ndim == 3 and arr.shape[2] == 3 and channel_order == "RGB":
			arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # helpers below expect BGR

		image = imageutils.bgr_to_image_batch([arr])
		if alpha is None:
			alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
		mask = imageutils.u8_to_mask_batch([alpha])
		return io.NodeOutput(image, mask)


class MaskToCV(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MaskToCV",
			display_name="Mask → CV Array",
			category=CATEGORY,
			description="Converts a ComfyUI MASK to a raw single-channel ndarray (NPARRAY). "
		            "Output can be uint8 (0-255) or float32 (0-1).",
	inputs=[
			io.Mask.Input("mask",
				tooltip="MASK to convert. One frame is taken (see batch_index)."),
			io.Combo.Input(
				"dtype",
				options=["uint8 (0-255)", "float32 (0-1)"],
				default="uint8 (0-255)",
				tooltip="uint8 for most cv2 functions; float32 (0-1) to preserve "
				        "fractional mask values for downstream float math.",
			),
			io.Int.Input("batch_index", default=0, min=0, max=4095,
				tooltip="Which mask of the batch to convert. Clamped to the "
				        "batch size."),
		],
		outputs=[NPArray.Output(display_name="nparray")],
	)

	@classmethod
	def execute(cls, mask, dtype, batch_index) -> io.NodeOutput:
		frames = imageutils.mask_batch_to_u8(mask)
		frame = frames[min(batch_index, len(frames) - 1)]
		if dtype.startswith("float32"):
			frame = frame.astype(np.float32) / 255.0
		return io.NodeOutput(frame)


class CVToMask(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CVToMask",
			display_name="CV Array → Mask",
			category=CATEGORY,
			description="Converts a single-channel ndarray to a ComfyUI MASK. "
			            "Multi-channel input is converted to grayscale first.",
			inputs=[NPArray.Input("nparray",
				tooltip="Single-channel [H,W] ndarray (multi-channel is converted "
				        "to grayscale first). Any dtype; scaled to a 0-1 mask.")],
			outputs=[io.Mask.Output(display_name="mask")],
		)

	@classmethod
	def execute(cls, nparray) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.ndim not in (2, 3) or arr.size == 0:
			raise ValueError(
				f"Cannot interpret array of shape {arr.shape} (dtype {arr.dtype}) as a mask."
			)
		return io.NodeOutput(imageutils.u8_to_mask_batch([arr]))


class CVToLatent(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CVToLatent",
			display_name="CV Array → Latent",
			category=CATEGORY,
			description="Wraps a raw float ndarray as a ComfyUI LATENT "
			            "({'samples': [1, C, H, W]}) so a cv2-produced array can "
			            "re-enter a latent pipeline - e.g. feed the plate from "
			            "'CV Temporal Reduce' (run on a LATENT batch) into "
			            "'VAE Decode'. The array's LAST axis becomes the latent "
			            "channel axis; values are kept as float32 and NEVER "
			            "quantized (latents are unbounded). This is the inverse of "
			            "feeding a LATENT into any OpenCV node, which unwraps frame "
			            "0 to [H,W,C]. The channel count must match what the "
			            "downstream VAE expects (16 for qwen_image_vae, 4 for "
			            "SD 1.5 / SDXL).",
			inputs=[NPArray.Input("nparray",
				tooltip="Float ndarray [H,W] or [H,W,C] (C = latent channels; a "
				        "2-D array becomes single-channel). Values are taken as-is "
				        "- no 0..1 or 0..255 scaling. Use 'Inspect CV Data' to "
				        "check the shape/dtype.")],
			outputs=[io.Latent.Output(display_name="latent")],
		)

	@classmethod
	def execute(cls, nparray) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.ndim not in (2, 3) or arr.size == 0:
			raise ValueError(
				f"Cannot interpret array of shape {arr.shape} (dtype {arr.dtype}) "
				"as a latent. Expected [H,W] or [H,W,C].")
		return io.NodeOutput(polytype.restore(arr, "latent"))


class LatentToCV(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_LatentToCV",
			display_name="Latent → CV Array",
			category=CATEGORY,
			description="Unwraps a ComfyUI LATENT "
			            "({'samples': [B,C,H,W]}) back to a float ndarray "
			            "[H,W,C] so it can be processed by OpenCV nodes. "
			            "Channels move to the last axis. Values are kept as "
			            "float32 - never quantized. This is the inverse of "
			            "'CV Array → Latent'.",
			inputs=[
				io.Latent.Input("latents",
					tooltip="A LATENT dict with 'samples' tensor of shape "
					        "[B,C,H,W]. The selected batch index is extracted "
					        "and channels are moved to the last axis, yielding "
					        "[H,W,C]."),
				io.Int.Input("batch_index", default=0, min=0, max=4095,
					tooltip="Which latent of the batch to convert. Clamped to "
					        "the batch size."),
			],
			outputs=[NPArray.Output(display_name="nparray",
				tooltip="Float32 ndarray [H,W,C] (C = latent channels; 16 for "
				        "qwen_image_vae, 4 for SD 1.5 / SDXL).")],
		)

	@classmethod
	def execute(cls, latents, batch_index) -> io.NodeOutput:
		if not polytype.is_latent(latents):
			raise ValueError(
				"Expected a LATENT dict with 'samples' key, "
				f"got {type(latents).__name__}.")
		samples = latents["samples"]
		if samples.ndim != 4:
			raise ValueError(
				f"Only 4-D [B,C,H,W] latents are supported, got shape "
				f"{tuple(samples.shape)}.")
		b = min(batch_index, samples.shape[0] - 1)
		arr = samples[b].movedim(0, -1).float().cpu().numpy()
		return io.NodeOutput(np.ascontiguousarray(arr))


class AnyToCV(io.ComfyNode):
	# one input socket that swallows IMAGE / MASK / LATENT / NPARRAY and always
	# emits a raw ndarray, so a subgraph boundary can accept an arbitrary type
	# and still process it with the low-level cv2.* nodes. Conversion follows the
	# polytype conventions (frame 0 of a batch): IMAGE -> uint8 BGR [H,W,3],
	# MASK -> uint8 [H,W], LATENT -> float32 [H,W,C] (values untouched), NPARRAY
	# passes through. The dedicated Image/Mask/Latent -> CV Array nodes stay the
	# right choice when you need their extra knobs (color_format, dtype,
	# batch_index); this one is the zero-config any-type bridge.
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_AnyToCV",
			display_name="* → CV Array",
			category=CATEGORY,
			description="Converts an IMAGE, MASK, LATENT or NPARRAY to a raw "
			            "OpenCV ndarray (NPARRAY) for the low-level cv2.* nodes, "
			            "auto-detecting the input type. Frame 0 of a batch is "
			            "taken. IMAGE becomes uint8 BGR [H,W,3]; MASK becomes "
			            "uint8 [H,W]; LATENT becomes float32 [H,W,C] with values "
			            "untouched; an NPARRAY passes through unchanged. Useful in "
			            "subgraphs whose boundary socket must accept an arbitrary "
			            "type but process it as a CV array. For extra control "
			            "(color_format / dtype / batch_index) use the dedicated "
			            "'Image → CV Array' / 'Mask → CV Array' / 'Latent → CV "
			            "Array' nodes instead.",
			inputs=[polytype.poly_input("input",
				types=polytype.with_latent(polytype.IMAGEISH),
				tooltip="IMAGE, MASK, LATENT or NPARRAY to convert to a raw "
				        "ndarray." + polytype.LATENT_NOTE)],
			outputs=[NPArray.Output(display_name="nparray",
				tooltip="Raw ndarray: uint8 BGR [H,W,3] from IMAGE, uint8 [H,W] "
				        "from MASK, float32 [H,W,C] from LATENT, or the NPARRAY "
				        "as-is.")],
		)

	@classmethod
	def execute(cls, input) -> io.NodeOutput:
		arr, _ = polytype.resolve(input)
		return io.NodeOutput(np.asarray(arr))


class ScalarArray(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ScalarArray",
			display_name="CV Scalar",
			category=CATEGORY,
			description="Types a constant vector for NPARRAY sockets that "
			            "expect one: cv2_inRange's lowerb/upperb bounds, "
			            "cv2_randn's mean/stddev, cv2_compare's src2, or "
			            "polynomial coefficients for 'CV Polynomial "
			            "Radial Map'. Write it as a literal like "
			            "'(35, 60, 60)' (or a single number, which becomes "
			            "'(N, N, N, N)'). Up to 32 elements for arbitrary "
			            "coefficient arrays. Same "
			            "spirit as 'CV Array Size' for Size-typed parameters.",
			inputs=[
				io.String.Input("values", default="(0, 0, 0)",
					tooltip="Number or tuple literal, e.g. '128' or "
					        "'(35, 60, 60)'. One element per channel for "
					        "bounds/means; a bare single number broadcasts "
					        "to four scalar slots. "
					        "For polynomial radial-map coefficients use "
					        "e.g. '(1.3, -0.6, 0.3)'."),
				io.Combo.Input("dtype",
					options=["float64 (cv2 Scalar default)", "float32",
					         "int32", "uint8"],
					default="float64 (cv2 Scalar default)",
					tooltip="Element type of the emitted array. float64 is "
					        "what cv2 expects for Scalar-like arguments."),
			],
			outputs=[NPArray.Output(display_name="nparray",
				tooltip="1-D constant array. A bare number N emits "
				        "(N, N, N, N); tuple literals emit one element per entry.")],
		)

	@classmethod
	def execute(cls, values, dtype) -> io.NodeOutput:
		# scalars.parse is the SHARED rule: a bare number broadcasts to the four
		# slots of a cv2 Scalar, a tuple is used as written. This node allows a
		# longer tuple than a Scalar does (max_len=32) because it also types
		# arbitrary coefficient vectors ('CV Polynomial Radial Map').
		comps, _ = scalars.parse(values, "values", width=4, max_len=32)
		arr = np.asarray(comps, np.float64)
		return io.NodeOutput(arr.astype(np.dtype(dtype.split(" ")[0])))


class TermCriteria(io.ComfyNode):
	_STOP = {
		"max count or epsilon (whichever first)": cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
		"max count only": cv2.TERM_CRITERIA_COUNT,
		"epsilon only": cv2.TERM_CRITERIA_EPS,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_TermCriteria",
			display_name="CV Term Criteria",
			category=CATEGORY,
			description="Builds the (type, max_count, epsilon) iteration-stop tuple "
			            "that cv2's iterative algorithms expect (cornerSubPix, "
			            "kmeans, calcOpticalFlowPyrLK, findTransformECC, meanShift, "
			            "solvePnPRefine*...). Wire its output into those nodes' "
			            "'criteria' input instead of typing the magic type integer "
			            "(3 = count|eps). Same spirit as 'CV Scalar' / 'CV Array "
			            "Size' for composite literals.",
			inputs=[
				io.Combo.Input("stop_on", options=list(cls._STOP),
					default="max count or epsilon (whichever first)",
					tooltip="When to stop iterating: after max_count iterations, when "
					        "the change drops below epsilon, or whichever comes first."),
				io.Int.Input("max_count", default=30, min=1, max=100000,
					tooltip="Maximum iterations (ignored when 'epsilon only')."),
				io.Float.Input("epsilon", default=0.001, min=0.0, max=1e6, step=0.0001,
					tooltip="Target accuracy / smallest change worth continuing for "
					        "(ignored when 'max count only')."),
			],
			outputs=[io.String.Output(display_name="criteria",
				tooltip="'(type, max_count, epsilon)' literal - feed it into a cv2 "
				        "node's criteria input (convert that widget to an input).")],
		)

	@classmethod
	def execute(cls, stop_on, max_count, epsilon) -> io.NodeOutput:
		return io.NodeOutput(f"({cls._STOP[stop_on]}, {int(max_count)}, {float(epsilon)})")


def _flag_builder(node_id, display_name, group, targets, note=""):
	"""Builds a pseudo-primitive node that authors one FlagGroup value (base
	dropdown + per-flag toggles, rendered by web/cv_flags.js) and fans it out:
	its 'flags' STRING output wires into the flags input of every node in
	`targets`, so they all share one configuration. The 'value' INT output is
	the resolved integer for raw int flag params or MathExpression."""
	example = group.initial + (f" | {group.flags[0]}" if group.flags else "")

	class FlagBuilder(io.ComfyNode):
		@classmethod
		def define_schema(cls):
			return io.Schema(
				node_id=node_id,
				display_name=display_name,
				category=CATEGORY,
				description=f"Authors a flags value once and fans it out: wire "
				            f"the 'flags' STRING output into the flags input of "
				            f"any number of {targets} nodes so they all share "
				            f"one configuration. Each toggle ORs one bit in. "
				            + (note + " " if note else "")
				            + "The 'value' INT output is the resolved integer "
				              "for raw int flag params or MathExpression.",
				inputs=[
					io.String.Input("flags", default=group.initial,
						tooltip=f"Pipe-joined cv2 constant names, e.g. "
						        f"\"{example}\". In the UI this renders as a "
						        f"base dropdown with one toggle per flag.",
						extra_dict={"cv_flags": group.spec()}),
				],
				outputs=[
					io.String.Output(display_name="flags",
						tooltip=f"The pipe-joined flags string - wire it into "
						        f"the flags input of {targets} nodes."),
					io.Int.Output(display_name="value",
						tooltip="The resolved integer (constants OR-ed together)."),
				],
			)

		@classmethod
		def execute(cls, flags) -> io.NodeOutput:
			# resolve() validates the names (loud error on a typo) and ORs the bits
			return io.NodeOutput(flags, group.resolve(flags))

	FlagBuilder.__name__ = FlagBuilder.__qualname__ = node_id
	return FlagBuilder


WarpFlags = _flag_builder(
	"CV_WarpFlags", "CV Warp Flags", enums.WARP_ALL_FLAGS,
	"cv2_remap / cv2_warpAffine / cv2_warpPerspective / cv2_warpPolar",
	note="The base dropdown picks the interpolation; cv2 ignores bits the "
	     "target function doesn't use - e.g. WARP_POLAR_LOG only matters "
	     "for warpPolar.")
DFTFlags = _flag_builder(
	"CV_DFTFlags", "CV DFT Flags", enums.DFT_FLAGS,
	"cv2_dft / cv2_idft",
	note="DFT_COMPLEX_OUTPUT gives the full 2-channel (re, im) spectrum the "
	     "spectrum tools expect; idft usually wants DFT_SCALE | DFT_REAL_OUTPUT.")
DCTFlags = _flag_builder(
	"CV_DCTFlags", "CV DCT Flags", enums.DCT_FLAGS,
	"cv2_dct / cv2_idct")
OptFlowFlags = _flag_builder(
	"CV_OptFlowFlags", "CV Optical Flow Flags", enums.OPTFLOW_ALL_FLAGS,
	"cv2_calcOpticalFlowPyrLK / cv2_calcOpticalFlowFarneback",
	note="Each function ignores the other's bit (LK_GET_MIN_EIGENVALS is "
	     "PyrLK-only, FARNEBACK_GAUSSIAN is Farneback-only).")
ChessboardFlags = _flag_builder(
	"CV_ChessboardFlags", "CV Chessboard Flags", enums.CALIB_CB_ALL_FLAGS,
	"cv2_findChessboardCorners / cv2_findChessboardCornersSB(WithMeta)",
	note="ADAPTIVE_THRESH/FAST_CHECK/FILTER_QUADS drive the classic detector, "
	     "EXHAUSTIVE/ACCURACY/LARGER/MARKER the SB one; NORMALIZE_IMAGE both.")
GemmFlags = _flag_builder(
	"CV_GemmFlags", "CV GEMM Flags", enums.GEMM_FLAGS,
	"cv2_gemm",
	note="GEMM_1_T/2_T/3_T transpose src1/src2/src3 before computing "
	     "alpha*src1*src2 + beta*src3.")
SVDFlags = _flag_builder(
	"CV_SVDFlags", "CV SVD Flags", enums.SVD_FLAGS,
	"cv2_SVDecomp")
FloodFillFlags = _flag_builder(
	"CV_FloodFillFlags", "CV FloodFill Flags", enums.FLOODFILL,
	"cv2_floodFill",
	note="The base picks connectivity + the mask fill value; FIXED_RANGE "
	     "compares against the seed pixel, MASK_ONLY leaves the image "
	     "untouched.")
ThreshFlags = _flag_builder(
	"CV_ThreshFlags", "CV Threshold Flags", enums.THRESH,
	"cv2_threshold / cv2_thresholdWithMask",
	note="The base picks the threshold method (BINARY, TRUNC, TOZERO, ...); "
	     "OTSU auto-computes the threshold from the histogram, TRIANGLE uses "
	     "the histogram peak-to-tail distance.")


class GridPoints(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_GridPoints",
			display_name="CV Grid Points",
			category=CATEGORY,
			description="Generates a regular planar grid of 3D points - the object "
			            "points cv2.solvePnP / cv2.calibrateCamera need for a "
			            "chessboard, which you otherwise cannot type (a 9x6 board is "
			            "54x3 values). Points run row by row to match "
			            "cv2.findChessboardCorners order: (0,0,z), (1,0,z), ... "
			            "scaled by 'spacing'. Pair the corners from "
			            "cv2.findChessboardCorners (image points) with this grid "
			            "(object points) and a camera matrix in cv2.solvePnP to "
			            "recover the board's pose.",
			inputs=[
				io.Int.Input("cols", default=9, min=1, max=1000,
					tooltip="Points per row (e.g. chessboard inner corners per row)."),
				io.Int.Input("rows", default=6, min=1, max=1000,
					tooltip="Points per column (e.g. inner corners per column)."),
				io.Float.Input("spacing", default=1.0, min=1e-6, max=1e6,
					tooltip="Distance between adjacent points (e.g. the square size "
					        "in mm). 1.0 = unit / relative coordinates."),
				io.Float.Input("z", default=0.0, min=-1e6, max=1e6,
					tooltip="Constant plane height for every point (0 = a flat board "
					        "on the z=0 plane).", optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="(cols*rows)x1x3 float32 grid points, row-major."),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, cols, rows, spacing, z=0.0) -> io.NodeOutput:
		n = int(cols) * int(rows)
		grid = np.zeros((n, 3), np.float32)
		grid[:, :2] = np.mgrid[0:int(cols), 0:int(rows)].T.reshape(-1, 2) * float(spacing)
		grid[:, 2] = float(z)
		return io.NodeOutput(grid.reshape(-1, 1, 3), n)


class PointsLiteral(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PointsLiteral",
			display_name="CV Points",
			category=CATEGORY,
			description="Authors an arbitrary point array from a Python literal - "
			            "the general counterpart to 'CV Scalar' (one vector) and "
			            "'CV Grid Points' (a regular grid). Write a list of points: "
			            "[[x, y], ...] -> Nx1x2 (image points or a polygon), or "
			            "[[x, y, z], ...] -> Nx1x3 (3D object points for cv2.solvePnP "
			            "/ projectPoints, e.g. the corners of a virtual cube). Emits "
			            "the same Nx1xD layout the detector / grid nodes produce.",
			inputs=[
				io.String.Input("points",
					default="[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]",
					tooltip="List of points as a literal: [[x, y], ...] for 2-D or "
					        "[[x, y, z], ...] for 3-D. A single [x, y[, z]] is also "
					        "accepted."),
				io.Combo.Input("dtype", options=["float32", "float64", "int32"],
					default="float32",
					tooltip="Element type. float32 is what cv2 geometry functions "
					        "(solvePnP / projectPoints) expect for points."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="Nx1x2 or Nx1x3 point array."),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, points, dtype) -> io.NodeOutput:
		from ast import literal_eval
		try:
			parsed = literal_eval(points.strip())
		except Exception as e:
			raise ValueError(
				f'points must be a list literal like "[[0, 0, 0], [1, 0, 0]]", '
				f'got "{points}". ({e})'
			) from e
		arr = np.atleast_2d(np.asarray(parsed, dtype=np.dtype(dtype)))
		if arr.ndim == 3 and arr.shape[1] == 1:  # already Nx1xD
			arr = arr.reshape(arr.shape[0], arr.shape[2])
		if arr.ndim != 2 or arr.shape[1] not in (2, 3):
			raise ValueError(
				f"points must hold 2-D or 3-D coordinates (shape Nx2 or Nx3), "
				f"got shape {arr.shape}."
			)
		return io.NodeOutput(arr.reshape(-1, 1, arr.shape[1]), int(len(arr)))


class SplitPoint(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SplitPoint",
			display_name="CV Split Point",
			category=CATEGORY,
			description="Splits a point (x, y) into its two coordinates. Takes "
			            "the CV_TUPLE a cv2 point output emits (min_loc/max_loc "
			            "from cv2.minMaxLoc, center from cv2.minEnclosingCircle) "
			            "as well as the \"(30, 12)\" literal string those "
			            "outputs used to be. Select 'int' for pixel indices "
			            "(truncates floats) or 'float' for subpixel coordinates. "
			            "'CV Split Tuple' does the same for any arity.",
			inputs=[
				tuples.TupleLiteralInput("point",
					placeholder="e.g. (30, 12)",
					tooltip="A point - wire it from a cv2 node's point output "
					        "(min_loc/max_loc, center, ...) or type the "
					        "\"(30, 12)\" literal."),
				io.Combo.Input("dtype", options=["int", "float"], default="int",
					tooltip="Output type: 'int' truncates to integer (for pixel "
					        "indices), 'float' preserves fractional values (for "
					        "subpixel coordinates)."),
			],
			outputs=[
				io.AnyType.Output(display_name="x",
					tooltip="X coordinate - int or float per dtype selection."),
				io.AnyType.Output(display_name="y",
					tooltip="Y coordinate - int or float per dtype selection."),
			],
		)

	@classmethod
	def execute(cls, point, dtype) -> io.NodeOutput:
		from ast import literal_eval
		if isinstance(point, (list, tuple)):
			parsed = point          # a CV_TUPLE link: already the two numbers
		else:
			try:
				parsed = literal_eval(str(point).strip())
			except Exception as e:
				raise ValueError(
					f"Expected a tuple literal like \"(30, 12)\", got "
					f"\"{point}\". ({e})"
				) from e
		if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
			raise ValueError(
				f"Expected a 2-element tuple/list, got {parsed!r}."
			)
		if dtype == "int":
			return io.NodeOutput(int(parsed[0]), int(parsed[1]))
		return io.NodeOutput(float(parsed[0]), float(parsed[1]))


class ParseMatrix(io.ComfyNode):
	"""Parses whitespace-delimited text into a 2-D numpy matrix."""

	_CV2_TO_NP = {
		cv2.CV_8U: np.uint8,
		cv2.CV_8S: np.int8,
		cv2.CV_16U: np.uint16,
		cv2.CV_16S: np.int16,
		cv2.CV_32S: np.int32,
		cv2.CV_32F: np.float32,
		cv2.CV_64F: np.float64,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ParseMatrix",
			display_name="Parse Matrix",
			category=CATEGORY,
			description="Parses a delimited text table into a 2-D numpy matrix. "
			            "Spaces, tabs, commas or semicolons separate columns, "
			            "newlines separate rows; consecutive separators are "
			            "collapsed. Reads back everything 'CV Array To Text' "
			            "writes (that is the round trip: format -> 'Basic data "
			            "handling: save STRING to file' -> load -> parse). "
			            "Rejects ragged rows (varying column counts).",
			inputs=[
				io.String.Input("text",
					default="1 2 3\n4 5 6",
					multiline=True,
					tooltip="Matrix data: spaces, tabs, commas or semicolons "
					        "separate columns, newlines separate rows. "
					        "Consecutive separators are collapsed."),
				io.Combo.Input("dtype",
					options=enums.DEPTH.options,
					default=enums.DEPTH.default,
					tooltip="Output array depth. 'same as input' defaults to CV_64F (float64).",
					optional=True, advanced=True),
				io.Boolean.Input("transpose",
					default=False,
					optional=True, advanced=True,
					tooltip="Swap rows and columns (transpose the matrix)."),
				io.Int.Input("skip_rows", default=0, min=0, max=1024,
					optional=True, advanced=True,
					tooltip="Drop this many leading lines before parsing - 1 skips "
					        "a CSV header row (a header is text, so parsing one as "
					        "numbers raises). Blank lines never count."),
			],
			outputs=[
				NPArray.Output(display_name="nparray",
					tooltip="Parsed 2-D numpy matrix."),
			],
		)

	@classmethod
	def execute(cls, text, dtype="same as input", transpose=False,
			skip_rows=0) -> io.NodeOutput:
		# commas/semicolons are separators too, so a CSV written by 'CV Array To
		# Text' (or exported by anything else) parses without a strip step
		lines = [line.replace(",", " ").replace(";", " ")
		         for line in text.strip().split("\n")]
		rows = [line.split() for line in lines if line.strip()]
		rows = rows[int(skip_rows):]
		if not rows:
			raise ValueError("No matrix data found in the input text.")
		ncols = len(rows[0])
		if ncols == 0:
			raise ValueError("First row has zero columns (only whitespace detected).")
		for i, row in enumerate(rows):
			if len(row) != ncols:
				raise ValueError(
					f"Inconsistent column count: row 0 has {ncols} columns, "
					f"row {i} has {len(row)}."
				)
		arr = np.asarray(rows, np.float64)
		if transpose:
			arr = arr.T
		depth = enums.DEPTH.resolve(dtype)
		out_dtype = cls._CV2_TO_NP.get(depth, np.float64)
		return io.NodeOutput(arr.astype(out_dtype))


class CameraMatrix(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CameraMatrix",
			display_name="CV Camera Matrix",
			category=CATEGORY,
			description="Builds the 3x3 pinhole intrinsic matrix "
			            "K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] that "
			            "cv2.solvePnP, projectPoints, undistort, stereoRectify, "
			            "initUndistortRectifyMap... require - a 3x3 you cannot type "
			            "with 'CV Scalar'. Use it for a known or approximate camera "
			            "(a common rough guess is fx = fy = image width and (cx, cy) "
			            "= the image center); for an accurate matrix calibrate with "
			            "'Calibrate Camera (Chessboard)' instead.",
			inputs=[
				io.Float.Input("fx", default=1000.0, min=1e-3, max=1e7,
					tooltip="Focal length along X in pixels."),
				io.Float.Input("fy", default=1000.0, min=1e-3, max=1e7,
					tooltip="Focal length along Y in pixels (usually = fx)."),
				io.Float.Input("cx", default=320.0, min=-1e6, max=1e6,
					tooltip="Principal point X in pixels (usually image width / 2)."),
				io.Float.Input("cy", default=240.0, min=-1e6, max=1e6,
					tooltip="Principal point Y in pixels (usually image height / 2)."),
			],
			outputs=[NPArray.Output(display_name="camera_matrix",
				tooltip="3x3 float64 intrinsic matrix K.")],
		)

	@classmethod
	def execute(cls, fx, fy, cx, cy) -> io.NodeOutput:
		K = np.array([[float(fx), 0.0, float(cx)],
		              [0.0, float(fy), float(cy)],
		              [0.0, 0.0, 1.0]], np.float64)
		return io.NodeOutput(K)


class CastArray(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CastArray",
			display_name="CV Cast Array",
			category=CATEGORY,
			description="Converts an ndarray to another element type, "
			            "saturating like cv2.convertTo: values are rounded "
			            "and clipped to the target range instead of "
			            "wrapping around. cv2 arithmetic (add, multiply, "
			            "compare...) refuses mixed types - e.g. int32 "
			            "connected-component labels cannot be added to a "
			            "uint8 mask without casting one side first.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="Array to convert; values survive, only the "
					        "element type changes."),
				io.Combo.Input("dtype",
					options=["uint8", "uint16", "int16", "int32",
					         "float32", "float64"],
					default="uint8",
					tooltip="Target element type. Integer targets round "
					        "and clip (saturate); float targets convert "
					        "values as-is."),
			],
			outputs=[NPArray.Output(display_name="nparray")],
		)

	@classmethod
	def execute(cls, nparray, dtype) -> io.NodeOutput:
		arr = np.asarray(nparray)
		target = np.dtype(dtype)
		if np.issubdtype(target, np.integer):
			info = np.iinfo(target)
			arr = np.clip(np.rint(arr.astype(np.float64)), info.min, info.max)
		return io.NodeOutput(arr.astype(target))


class SetRNGSeed(io.ComfyNode):
	"""cv2's global RNG seed as a node, on the wire so it cannot run too late.

	cv2.setRNGSeed is not in the registry (it returns void, so the generator
	drops it), which left every randomised cv2 node - kmeans above all -
	unrepeatable once its wrapper node was deleted in favour of a subgraph.
	A node with no data flowing through it would be worse than useless here:
	ComfyUI's execution order is driven by the LINKS, so a seed node that
	nothing consumed could run after the thing it is supposed to seed. Hence
	the passthrough - the value on the wire is the ordering constraint.

	That passthrough is a WILDCARD (io.AnyType, io_type "*"), not NPARRAY: the
	ordering constraint is the only thing the value is used for, so restricting
	it to one socket type would only limit where the node can be inserted.
	Seeding a random IMAGE fill, a RANSAC homography or an ml.py trainer means
	sitting on an IMAGE / CV_CONTOURS / MODEL / STRING wire, and "*" matches
	both ways in comfy_execution.validation.validate_node_input.
	"""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SetRNGSeed",
			display_name="CV Random Seed",
			category=CATEGORY,
			description="Seeds cv2's global random number generator, then passes "
			            "its input straight through unchanged. Put it on the wire "
			            "IMMEDIATELY before the node whose randomness you want to "
			            "pin - cv2.kmeans' initial centres, cv2.randu/randn, "
			            "RANSAC - and the same seed gives the same result every "
			            "run. It has to sit in the data path: execution order "
			            "follows the links, so a seed node feeding nothing may "
			            "run after the node it was meant to seed. The passthrough "
			            "accepts ANY type (image, mask, array, contours, model, "
			            "string, ...), so it can be inserted wherever the wire "
			            "into the randomised node happens to be.",
			inputs=[
				io.AnyType.Input("value",
					tooltip="Any value, passed through untouched (not even "
					        "copied) - it is here only to force this node to "
					        "run BEFORE the one you are seeding. Route a wire "
					        "that already feeds the randomised node through "
					        "here; the type does not matter."),
				io.Int.Input("seed", default=0, min=0, max=INT32_MAX,
					tooltip="Any value; the same seed reproduces the same "
					        "sequence. Note this is cv2's ONE global RNG, so the "
					        "result also depends on what other cv2 nodes drew "
					        "from it in between."),
			],
			outputs=[io.AnyType.Output(display_name="value",
				tooltip="The input value, unchanged.")],
		)

	@classmethod
	def execute(cls, value, seed) -> io.NodeOutput:
		cv2.setRNGSeed(int(seed))
		return io.NodeOutput(value)


class ReshapeArray(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ReshapeArray",
			display_name="CV Reshape Array",
			category=CATEGORY,
			description="Reinterprets an ndarray's shape without touching its "
			            "values (numpy reshape). The main use: fold a flat "
			            "per-sample vector back into a 2-D grid - e.g. the "
			            "query predictions of 'CV Train Classifier' over an "
			            "'CV Coordinate Grid' become a (rows, cols) label "
			            "image ready for a colormap. 0 infers that dimension "
			            "from the element count (at most one of rows/cols may "
			            "be 0). channels > 0 appends a third axis. Raises when "
			            "the element count does not fit the requested shape - "
			            "that is a wiring mistake, not data.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="Array to reshape; the element count must match "
					        "rows * cols (* channels)."),
				io.Int.Input("rows", default=0, min=0, max=INT32_MAX,
					tooltip="Target row count (height). 0 = infer from the "
					        "element count and the other dimensions."),
				io.Int.Input("cols", default=0, min=0, max=INT32_MAX,
					tooltip="Target column count (width). 0 = infer."),
				io.Int.Input("channels", default=0, min=0, max=512,
					optional=True, advanced=True,
					tooltip="Optional third axis size (e.g. 2 for point pairs, "
					        "3 for color). 0 = no channel axis."),
			],
			outputs=[NPArray.Output(display_name="nparray",
				tooltip="The same values in the new shape.")],
		)

	@classmethod
	def execute(cls, nparray, rows, cols, channels=0) -> io.NodeOutput:
		arr = np.asarray(nparray)
		shape = [int(rows) or -1, int(cols) or -1]
		if int(channels) > 0:
			shape.append(int(channels))
		if shape.count(-1) > 1:
			raise ValueError(
				"rows and cols cannot both be 0 - fix one of them so the other "
				"can be inferred."
			)
		try:
			return io.NodeOutput(np.ascontiguousarray(arr.reshape(shape)))
		except ValueError as e:
			raise ValueError(
				f"Cannot reshape {arr.size} elements (shape {arr.shape}) into "
				f"{tuple(shape)} - check the rows/cols against the producer "
				f"(e.g. the Coordinate Grid's rows/cols outputs). ({e})"
			) from e


class PermuteAxes(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PermuteAxes",
			display_name="CV Permute Axes",
			category=CATEGORY,
			description="Reorders an ndarray's axes (numpy moveaxis). The main use "
			            "is converting an (H, W, C) image array into (C, H, W) for "
			            "DNN blob input, or vice-versa. The 'source' axis is moved "
			            "to the 'dest' position; all other axes keep their relative "
			            "order.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="Array whose axes to permute."),
				io.Int.Input("source", default=-1, min=-4, max=4,
					tooltip="Axis to move FROM (-1 = last, e.g. the channel axis "
					        "in an HWC image)."),
				io.Int.Input("dest", default=0, min=-4, max=4,
					tooltip="Axis to move TO (0 = first, e.g. making channels "
					        "the leading dimension for NCHW)."),
			],
			outputs=[NPArray.Output(display_name="nparray",
				tooltip="The same data with axes reordered.")],
		)

	@classmethod
	def execute(cls, nparray, source, dest) -> io.NodeOutput:
		arr = np.asarray(nparray)
		try:
			return io.NodeOutput(np.ascontiguousarray(
				np.moveaxis(arr, int(source), int(dest))))
		except (ValueError, IndexError) as e:
			raise ValueError(
				f"Cannot move axis {source} to {dest} on an array of shape "
				f"{arr.shape}. ({e})") from e


class ToBlob(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ToBlob",
			display_name="CV To Blob",
			category=CATEGORY,
			description="Converts an HWC float32 image array into a 4-D NCHW "
			            "(1, C, H, W) DNN blob using cv2.dnn.blobFromImage. "
			            "Unlike the DNN Blob From Image node, this accepts "
			            "float32 arrays directly (no uint8 cast), so it works "
			            "on pre-normalized data (e.g. after ImageNet "
			            "normalization). Can also resize to a target size.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="HWC float32 array (H, W, C)."),
				io.Float.Input("scale", default=1.0,
					tooltip="Multiplier applied to each pixel. Use 1.0 "
					        "for already-scaled or normalized data."),
				io.Int.Input("target_width", default=0, min=0,
					tooltip="Target width in pixels. 0 = keep input size."),
				io.Int.Input("target_height", default=0, min=0,
					tooltip="Target height in pixels. 0 = keep input size."),
			],
			outputs=[NPArray.Output(display_name="blob",
				tooltip="4-D float32 blob shaped (1, C, H, W) ready "
				        "for cv2.dnn.Net.setInput().")],
		)

	@classmethod
	def execute(cls, nparray, scale, target_width, target_height) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.ndim == 2:
			arr = arr[:, :, np.newaxis]
		elif arr.ndim != 3:
			raise ValueError(
				f"Expected HWC (3-D) array, got {arr.ndim}-D shape "
				f"{arr.shape}.")
		dsize = (int(target_width), int(target_height)) if target_width and target_height else None
		blob = cv2.dnn.blobFromImage(
			arr.astype(np.float32), float(scale),
			dsize or (arr.shape[1], arr.shape[0]),
			(0, 0, 0))
		return io.NodeOutput(np.ascontiguousarray(blob))


class UnstackBatch(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_UnstackBatch",
			display_name="CV Unstack Batch",
			category=CATEGORY,
			description="Splits a batched NPARRAY along the first axis into a "
			            "list of individual arrays. E.g. (N, H, W, C) -> N x (H, W, C). "
			            "Use to process each frame of a DNN output batch through "
			            "per-frame nodes (Cast Array, CV Array -> Image, etc.).",
			inputs=[
				NPArray.Input("nparray",
					tooltip="Batched array, e.g. (N, H, W, C) from DNN Images "
					        "From Blob (Batch) or any N-D array with N>0 in axis 0."),
			],
			outputs=[NPArray.Output(display_name="frame", is_output_list=True,
				tooltip="List of (H, W, C) arrays, one per batch element.")],
		)

	@classmethod
	def execute(cls, nparray) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.ndim < 2:
			raise ValueError(
				f"Expected at least 2-D array, got {arr.ndim}-D shape {arr.shape}.")
		if arr.shape[0] == 0:
			return io.NodeOutput([np.zeros((0,), np.float32)])
		return io.NodeOutput([np.ascontiguousarray(arr[i]) for i in range(arr.shape[0])])


class StackBatch(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_StackBatch",
			display_name="CV Stack Batch",
			category=CATEGORY,
			description="Inverse of 'CV Unstack Batch': merges a list of "
			            "NPARRAYs into one batched array along the first axis. "
			            "E.g. N x (H, W, C) -> (N, H, W, C), where N is the "
			            "number of items in the list. All arrays must have "
			            "identical shapes (and the same number of dimensions) "
			            "so the stack stays regular.",
			inputs=[
				NPArray.Input("arrays",
					tooltip="List of NPARRAYs, one per batch element, e.g. the "
					        "'frame' list output of 'CV Unstack Batch'. All "
					        "arrays must share the same shape."),
			],
			outputs=[NPArray.Output(display_name="batch",
				tooltip="Batched array (N, ...): the first axis is the list "
				        "length, each element along it is one input array.")],
			is_input_list=True,
		)

	@classmethod
	def execute(cls, arrays) -> io.NodeOutput:
		frames = list(arrays) if isinstance(arrays, (list, tuple)) else [arrays]
		if not frames:
			return io.NodeOutput(np.zeros((0,), np.float32))
		first = np.asarray(frames[0])
		for i, f in enumerate(frames[1:], start=1):
			arr = np.asarray(f)
			if arr.shape != first.shape:
				raise ValueError(
					f"Cannot stack arrays of different shapes: item 0 is "
					f"{first.shape}, item {i} is {arr.shape}. All arrays in "
					f"the list must have identical shapes.")
		return io.NodeOutput(np.ascontiguousarray(np.stack(
			[np.asarray(f) for f in frames], axis=0)))


class ConcatArrays(io.ComfyNode):
	# The missing inverse of the split family. cv2's own hconcat / vconcat /
	# merge / split take a vector<Mat>, which the generator cannot express, so
	# none of them is in registry.py - and 'CV Stack Batch' only adds a NEW
	# leading axis. Without this node the shipped 'CV Split Channels (3)/(4)'
	# subgraphs are one-way, and two detection tables cannot be appended.
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ConcatArrays",
			display_name="CV Concat Arrays",
			category=CATEGORY,
			description="Joins arrays along an EXISTING axis - the inverse of "
			            "splitting one. axis 0 appends rows (two detection tables, "
			            "two point sets, two frames of samples), axis 1 puts them "
			            "side by side, axis -1 merges channels back together (the "
			            "return leg of 'CV Split Channels' / cv2.extractChannel). "
			            "Unlike 'CV Stack Batch' it adds no new axis and the inputs "
			            "need only agree on the OTHER axes. Mixed dtypes are "
			            "promoted the numpy way (uint8 + float32 -> float32); cast "
			            "first with 'CV Cast Array' if you need to control that.",
			inputs=[
				NPArray.Input("array1",
					tooltip="First array. Its number of dimensions and its sizes on "
					        "every axis EXCEPT the concat axis must match the others."),
				NPArray.Input("array2",
					tooltip="Second array, appended after the first."),
				io.Int.Input("axis", default=0, min=-8, max=8,
					tooltip="Which axis to join along. 0 = rows (append), 1 = "
					        "columns (side by side), -1 = the last axis (merge "
					        "arrays that already have a channel axis). Negative "
					        "counts from the end. ONE PAST the end - axis 2 for "
					        "(H,W) inputs - stacks a NEW last axis instead, which "
					        "is cv2.merge: three (H,W) channels become one (H,W,3) "
					        "image. Splitting with cv2.extractChannel drops that "
					        "axis, so that is the setting the round trip needs."),
				NPArray.Input("array3", optional=True, advanced=True,
					tooltip="Optional third array, appended after the second."),
				NPArray.Input("array4", optional=True, advanced=True,
					tooltip="Optional fourth array, appended after the third."),
			],
			outputs=[
				NPArray.Output(display_name="nparray",
					tooltip="The joined array. Its size on the concat axis is the "
					        "sum of the inputs'; every other axis is unchanged."),
				io.Int.Output(display_name="length",
					tooltip="Size of the result along the concat axis - handy for "
					        "splitting it back apart with 'CV Take By Index'."),
			],
		)

	@classmethod
	def execute(cls, array1, array2, axis=0, array3=None, array4=None) -> io.NodeOutput:
		parts, names = [], []
		for name, value in (("array1", array1), ("array2", array2),
				("array3", array3), ("array4", array4)):
			if value is None:
				continue
			arr = np.asarray(value)
			if arr.size == 0:
				continue  # an empty detection set contributes nothing
			parts.append(arr)
			names.append(name)
		if not parts:
			return io.NodeOutput(np.zeros((0,), np.float32), 0)

		ax = int(axis)
		ndim = parts[0].ndim
		if ax < 0:
			ax += ndim
		# one past the end = cv2.merge: stack a NEW last axis. cv2.extractChannel
		# returns (H,W), so this is what the split -> merge round trip needs and
		# a plain concatenate would silently glue the channels side by side.
		if ax == ndim:
			for name, arr in zip(names[1:], parts[1:]):
				if arr.shape != parts[0].shape:
					raise ValueError(
						f"Stacking a new last axis needs identical shapes: "
						f"{names[0]} is {tuple(parts[0].shape)}, {name} is "
						f"{tuple(arr.shape)}.")
			out = np.ascontiguousarray(np.stack(parts, axis=-1))
			return io.NodeOutput(out, int(out.shape[-1]))
		if not 0 <= ax < ndim:
			raise ValueError(
				f"axis {axis} is out of range for a {ndim}-D array "
				f"{tuple(parts[0].shape)} ({names[0]}).")
		for name, arr in zip(names[1:], parts[1:]):
			if arr.ndim != ndim:
				raise ValueError(
					f"Cannot concatenate a {arr.ndim}-D array ({name}, shape "
					f"{tuple(arr.shape)}) with a {ndim}-D one ({names[0]}, shape "
					f"{tuple(parts[0].shape)}). Reshape one of them first "
					f"('CV Reshape Array').")
			mismatch = [i for i in range(ndim)
			            if i != ax and arr.shape[i] != parts[0].shape[i]]
			if mismatch:
				raise ValueError(
					f"Cannot concatenate along axis {ax}: {names[0]} is "
					f"{tuple(parts[0].shape)} and {name} is {tuple(arr.shape)} - "
					f"they differ on axis {mismatch[0]}, which is not the concat "
					f"axis.")
		out = np.ascontiguousarray(np.concatenate(parts, axis=ax))
		return io.NodeOutput(out, int(out.shape[ax]))


class IndexBatch(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_IndexBatch",
			display_name="CV Index Batch",
			category=CATEGORY,
			description="Picks a single element from a batched NPARRAY by index. "
			            "E.g. (N, H, W, C) + index=2 -> (H, W, C). The index is "
			            "clamped to the valid range (no out-of-bounds error).",
			inputs=[
				NPArray.Input("nparray",
					tooltip="Batched array, e.g. (N, H, W, C) from 'CV Unstack "
					        "Batch' or 'DNN Images From Blob'."),
				io.Int.Input("index", default=0, min=0, max=4095,
					tooltip="Which element to extract (0-based). Clamped to the "
					        "batch size."),
			],
			outputs=[NPArray.Output(display_name="frame",
				tooltip="Single (H, W, C) array at the given index.")],
		)

	@classmethod
	def execute(cls, nparray, index) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.ndim < 2:
			raise ValueError(
				f"Expected at least 2-D array, got {arr.ndim}-D shape {arr.shape}.")
		if arr.shape[0] == 0:
			return io.NodeOutput(np.zeros((0,), np.float32))
		i = min(int(index), arr.shape[0] - 1)
		return io.NodeOutput(np.ascontiguousarray(arr[i]))


class MosaicBayer(io.ComfyNode):
	# The four CFA layouts as the (R, G, G, B) source-channel index at each of the
	# 2x2 positions (top-left, top-right, bottom-left, bottom-right). RGB input
	# convention: channel 0 = R, 1 = G, 2 = B. A pattern P here pairs with cv2's
	# COLOR_Bayer<P>2RGB* demosaicing code (see enums.DEMOSAIC).
	_PATTERNS = {
		"RGGB": (0, 1, 1, 2),
		"GRBG": (1, 0, 2, 1),
		"GBRG": (1, 2, 0, 1),
		"BGGR": (2, 1, 1, 0),
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MosaicBayer",
			display_name="CV Mosaic (RGB -> Bayer CFA)",
			category=CATEGORY,
			description="Mosaics a 3-channel RGB image down to a single-channel "
			            "Bayer Colour-Filter-Array (CFA): at each pixel it keeps "
			            "only the ONE channel that a real Bayer sensor would sample "
			            "there, following a 2x2 pattern. This is the inverse of "
			            "cv2.demosaicing - use it to build a synthetic CFA from a "
			            "ground-truth RGB (no licence-safe RAW Bayer sample exists), "
			            "then demosaic it back and score the reconstruction "
			            "(PSNR / absdiff) to see the zipper / moire artifacts each "
			            "algorithm leaves. The pattern names the top-left 2x2 layout "
			            "(RGGB = Red at (0,0)); feed the result into a cv2.demosaicing "
			            "node whose code is COLOR_Bayer<pattern>2RGB(_VNG/_EA).",
			inputs=[
				NPArray.Input("nparray",
					tooltip="RGB image [H,W,3] (channel 0 = R). Use 'Image -> CV "
					        "Array' in RGB mode to produce it. A single-channel "
					        "input is treated as already-mosaiced and passed "
					        "through unchanged."),
				io.Combo.Input("pattern", options=list(cls._PATTERNS),
					default="RGGB",
					tooltip="CFA layout of the top-left 2x2 block. RGGB = R at "
					        "(0,0), G at (0,1) and (1,0), B at (1,1) - the most "
					        "common sensor. Must match the demosaicing code "
					        "downstream (COLOR_Bayer<pattern>2RGB...)."),
			],
			outputs=[NPArray.Output(display_name="cfa",
				tooltip="Single-channel [H,W] CFA (same dtype as the input) - one "
				        "colour sample per pixel. Feed it into cv2.demosaicing.")],
		)

	@classmethod
	def execute(cls, nparray, pattern) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
			return io.NodeOutput(arr.reshape(arr.shape[:2]).copy())  # already a CFA
		if arr.ndim != 3 or arr.shape[2] < 3:
			raise ValueError(
				f"Expected an RGB image [H, W, 3], got shape {arr.shape}. Convert "
				"with 'Image -> CV Array' in RGB mode first."
			)
		r, g0, g1, b = cls._PATTERNS[pattern]
		cfa = np.empty(arr.shape[:2], arr.dtype)
		cfa[0::2, 0::2] = arr[0::2, 0::2, r]
		cfa[0::2, 1::2] = arr[0::2, 1::2, g0]
		cfa[1::2, 0::2] = arr[1::2, 0::2, g1]
		cfa[1::2, 1::2] = arr[1::2, 1::2, b]
		return io.NodeOutput(cfa)


class CVRoll(io.ComfyNode):
	# rolling moves pixels without changing nature or channel count, so the
	# output echoes the input's format (IMAGE/MASK/LATENT/NPARRAY). The MatchType
	# template is built lazily inside define_schema: convert.py is imported BY
	# polytype, so a class-body polytype call would hit the circular import.

	_MODES = [
		"fftshift (center the zero frequency)",
		"ifftshift (undo fftshift)",
		"custom (shift_x / shift_y)",
	]

	@classmethod
	def define_schema(cls):
		template = polytype.match_template(types=polytype.with_latent(polytype.IMAGEISH))
		return io.Schema(
			node_id="CV_CVRoll",
			display_name="CV Roll (FFT Shift)",
			category=CATEGORY,
			description="Cyclically shifts an array: pixels pushed off one edge "
			            "re-enter on the opposite edge (numpy.roll on the spatial "
			            "axes; channels are never mixed). The fftshift mode swaps "
			            "quadrants so cv2.dft's zero-frequency term moves from the "
			            "top-left corner to the center - the layout every spectrum "
			            "figure uses and the one to build centered frequency masks "
			            "in; ifftshift undoes it exactly (they differ on odd sizes). "
			            "Custom mode rolls by an explicit (shift_x, shift_y).",
			inputs=[
				polytype.match_input("array", template,
					tooltip="Array to roll (any dtype; a DFT spectrum, an image, a "
					        "mask...). The output echoes this input's format."
					        + polytype.LATENT_NOTE),
				io.Combo.Input("mode", options=cls._MODES, default=cls._MODES[0],
					tooltip="fftshift/ifftshift compute the half-size shifts from "
					        "the array itself; custom uses shift_x/shift_y."),
				io.Int.Input("shift_x", default=0, min=-INT32_MAX - 1, max=INT32_MAX,
					tooltip="Custom mode only: columns to roll right (negative = "
					        "left). Ignored by fftshift/ifftshift."),
				io.Int.Input("shift_y", default=0, min=-INT32_MAX - 1, max=INT32_MAX,
					tooltip="Custom mode only: rows to roll down (negative = up). "
					        "Ignored by fftshift/ifftshift."),
			],
			outputs=[polytype.match_output(template, "array",
				tooltip="The rolled array, in the same format the input arrived in.")],
		)

	@classmethod
	def execute(cls, array, mode, shift_x, shift_y) -> io.NodeOutput:
		arr, tag = polytype.resolve(array)
		arr = np.asarray(arr)
		if arr.ndim < 2:
			raise ValueError(f"Array of shape {arr.shape} has no spatial axes to roll.")
		h, w = arr.shape[:2]
		if mode.startswith("fftshift"):
			shift = (h // 2, w // 2)
		elif mode.startswith("ifftshift"):
			shift = (-(h // 2), -(w // 2))
		else:
			shift = (int(shift_y), int(shift_x))
		return io.NodeOutput(polytype.restore(np.roll(arr, shift, axis=(0, 1)), tag))


class ArrayStat(io.ComfyNode):
	# axis=0 reduces over every pixel, leaving one value per channel. nan-aware
	# so a stray NaN/Inf (distanceTransform / optical-flow / score maps) does not
	# poison the statistic into 'nan'.
	_STATS = {
		"median (robust dominant value)": np.nanmedian,
		"mean": np.nanmean,
		"min": np.nanmin,
		"max": np.nanmax,
		"std (spread)": np.nanstd,
	}

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ArrayStat",
			display_name="CV Array Statistic",
			category=CATEGORY,
			description="Reduces a whole array to ONE value per channel (median, "
			            "mean, min, max or std), returning a 1-D NPARRAY you can "
			            "broadcast straight into cv2 arithmetic (subtract, compare, "
			            "add...) - the same shape 'CV Scalar' emits. The MEDIAN is "
			            "the robust estimate of a field's DOMINANT value: e.g. the "
			            "background / camera motion of an optical-flow field, which "
			            "the median recovers even when a moving object contaminates a "
			            "large minority of pixels (the mean gets dragged toward it). "
			            "Subtract it from the field and threshold the residual to "
			            "segment whatever moves differently. NaN/Inf are ignored.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="2-D [H,W] or 3-D [H,W,C] array to reduce (e.g. an "
					        "optical-flow field HxWx2)."),
				io.Combo.Input("statistic", options=list(cls._STATS),
					default="median (robust dominant value)",
					tooltip="Reduction applied over every pixel, per channel. Median "
					        "= the dominant value (robust to outliers); mean = the "
					        "average (sensitive to them); std = the spread."),
			],
			outputs=[
				NPArray.Output(display_name="value",
					tooltip="1-D float64, one element per input channel - broadcastable "
					        "into cv2.subtract / compare / add like 'CV Scalar'."),
				io.String.Output(display_name="text",
					tooltip="The value(s) as a readable '(a, b, ...)' string - wire "
					        "into core 'Preview as Text' (PreviewAny) to display."),
			],
		)

	@classmethod
	def execute(cls, nparray, statistic) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.ndim not in (2, 3) or arr.size == 0:
			raise ValueError(
				f"Cannot reduce array of shape {arr.shape}; expected [H, W] or "
				"[H, W, C]. Use 'Inspect CV Data' to check the data first."
			)
		flat = arr.reshape(-1, 1) if arr.ndim == 2 else arr.reshape(-1, arr.shape[2])
		data = flat.astype(np.float64)
		# the nan* reducers ignore NaN but NOT Inf - map every non-finite value to
		# NaN first so they are genuinely excluded from the statistic
		data[~np.isfinite(data)] = np.nan
		with np.errstate(all="ignore"):  # an all-NaN channel warns; handled below
			val = cls._STATS[statistic](data, axis=0)
		# a channel that was entirely non-finite reduces to nan - report it as 0
		val = np.nan_to_num(np.atleast_1d(val), nan=0.0, posinf=0.0, neginf=0.0)
		text = "(" + ", ".join(f"{v:.4g}" for v in val) + ")"
		return io.NodeOutput(val.astype(np.float64), text)


class ReduceArrayByLabel(io.ComfyNode):
	"""Per-label reduction of an array, vectorized over the whole label map.

	This is the image-scale companion of 'CV Reduce Values By Label'
	(points.py), which reduces a 1-D per-POINT attribute and offers circular /
	densest-window statistics over a handful of clusters. This one takes the
	array whole (multi-channel, 2-D spatial), indexes its rows BY LABEL VALUE
	so the result can be fed straight back through 'CV Take By Index', and is
	pure numpy bincount - the only shape that survives 300k pixels x hundreds
	of regions.
	"""

	_REDUCTIONS = ["mean", "sum", "min", "max", "median"]

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ReduceArrayByLabel",
			display_name="CV Reduce Array By Label",
			category=CATEGORY,
			description="Collapses an array to ONE row per label: the mean, sum, "
			            "min, max or median of every element carrying that label. "
			            "The classic use is a region palette - feed an image plus "
			            "the label map of 'CV Superpixels' (or watershed, "
			            "connected components, HFS, k-means) and get the average "
			            "colour of each region, then paint it back with 'CV Take "
			            "By Index' for a flat, non-photorealistic render. Rows are "
			            "indexed BY LABEL VALUE (row i = label i), which is what "
			            "makes that round trip work; empty labels get "
			            "'empty_value'. For per-POINT attributes with circular or "
			            "densest-window statistics use 'CV Reduce Values By "
			            "Label' instead.",
			inputs=[
				NPArray.Input("values",
					tooltip="Array to reduce: (N,), (N,C), [H,W] or [H,W,C]. Its "
					        "leading axes must match 'labels' - the trailing axis "
					        "is the channel axis and is reduced independently."),
				NPArray.Input("labels",
					tooltip="Integer label per element: (N,) or [H,W], matching "
					        "the leading axes of 'values'. NEGATIVE labels are "
					        "excluded (watershed writes -1 on boundaries)."),
				io.Combo.Input("reduction", options=cls._REDUCTIONS, default="mean",
					tooltip="How the members of each label are collapsed. mean is "
					        "the region average (the palette case); median is the "
					        "robust version when a region straddles an edge; sum "
					        "with an all-ones input counts members (the 'counts' "
					        "output does that for you)."),
				io.Int.Input("num_labels", default=0, min=0, max=INT32_MAX,
					optional=True, advanced=True,
					tooltip="Number of rows K to emit. 0 = derive it from the "
					        "largest label present. Wire the 'count' output of "
					        "'CV Superpixels' here to pin the table height "
					        "even when a region ends up empty; labels >= K are "
					        "then ignored."),
				io.Float.Input("empty_value", default=0.0, min=-1e12, max=1e12,
					optional=True, advanced=True,
					tooltip="Value written for labels with no members (and for "
					        "every row when the input is empty). 0 paints such a "
					        "region black."),
				NPArray.Input("mask", optional=True, advanced=True,
					tooltip="Optional inclusion mask, same shape as 'labels': "
					        "nonzero elements take part in the reduction, zeros "
					        "are ignored (e.g. drop the sky before averaging)."),
			],
			outputs=[
				NPArray.Output(display_name="table",
					tooltip="(K,C) float32, row i = the reduction over label i. "
					        "Always 2-D, so a single-channel input gives (K,1). "
					        "Feed it to 'CV Take By Index' with the same label "
					        "map to paint the result back over the image."),
				NPArray.Output(display_name="counts",
					tooltip="(K,) int32: how many elements carried each label. 0 "
					        "marks a row that is 'empty_value', not a measurement."),
				io.Int.Output(display_name="count",
					tooltip="K, the number of rows in the table."),
			],
		)

	@classmethod
	def execute(cls, values, labels, reduction, num_labels=0, empty_value=0.0,
	            mask=None) -> io.NodeOutput:
		vals = np.zeros((0,), np.float64) if values is None else np.asarray(values)
		lab = np.zeros((0,), np.int64) if labels is None else np.asarray(labels)
		if lab.ndim == 0 or vals.ndim == 0:
			raise ValueError(
				f"values {vals.shape} / labels {lab.shape}: both must be arrays, "
				"not scalars."
			)
		if vals.shape[:lab.ndim] != lab.shape:
			raise ValueError(
				f"values of shape {vals.shape} do not match labels of shape "
				f"{lab.shape} - the leading axes must be identical (values may "
				"carry one extra channel axis). Use 'Inspect CV Data' to check."
			)
		channels = int(np.prod(vals.shape[lab.ndim:])) if vals.ndim > lab.ndim else 1
		flat = vals.reshape(-1, channels).astype(np.float64)
		idx = lab.reshape(-1).astype(np.int64)

		keep = idx >= 0
		if mask is not None:
			m = np.asarray(mask)
			if m.shape[:lab.ndim] != lab.shape:
				raise ValueError(
					f"mask of shape {m.shape} does not match labels of shape "
					f"{lab.shape}."
				)
			# a multi-channel mask counts as "include" where ANY channel is set
			keep &= (m.reshape(idx.size, -1) != 0).any(axis=1)

		size = int(num_labels)
		if size <= 0:
			size = int(idx[keep].max()) + 1 if keep.any() else 0
		keep &= idx < size
		idx, flat = idx[keep], flat[keep]

		counts = np.bincount(idx, minlength=size)[:size].astype(np.int32)
		table = np.full((size, channels), float(empty_value), np.float64)
		if size and idx.size:
			filled = counts > 0
			if reduction in ("mean", "sum"):
				sums = np.empty((size, channels), np.float64)
				for c in range(channels):
					sums[:, c] = np.bincount(idx, weights=flat[:, c], minlength=size)
				if reduction == "mean":
					sums[filled] /= counts[filled, None]
				table[filled] = sums[filled]
			elif reduction in ("min", "max"):
				op = np.minimum if reduction == "min" else np.maximum
				acc = np.full((size, channels), np.inf if reduction == "min"
				              else -np.inf, np.float64)
				op.at(acc, idx, flat)
				table[filled] = acc[filled]
			else:  # median: sort once by label, then slice each group
				order = np.argsort(idx, kind="stable")
				grouped = flat[order]
				starts = np.concatenate(([0], np.cumsum(counts)))
				for g in np.flatnonzero(filled):  # K iterations, not N
					table[g] = np.median(grouped[starts[g]:starts[g + 1]], axis=0)
		return io.NodeOutput(table.astype(np.float32), counts, int(size))


class TakeByIndex(io.ComfyNode):
	"""Gather: map an integer index array through a lookup table.

	numpy fancy indexing as a node. cv2.LUT and applyColorMap(userColor) stop
	at 256 entries and 8-bit input, which is exactly one entry short of every
	label map worth painting.
	"""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_TakeByIndex",
			display_name="CV Take By Index",
			category=CATEGORY,
			description="Looks every element of an integer array up in a table "
			            "and returns what it found: table row i replaces index "
			            "value i. With the (K,C) palette of 'CV Reduce Array By "
			            "Label' and the label map that produced it, this paints "
			            "each region with its own colour; it equally maps a "
			            "per-component area or score (from "
			            "'cv2.connectedComponentsWithStats') into a heat map. "
			            "Unlike cv2.LUT it has no 256-entry limit and takes any "
			            "integer dtype.",
			inputs=[
				NPArray.Input("table",
					tooltip="Lookup table, (K,) or (K,C...): row i is the value "
					        "that index i stands for. Its dtype is preserved."),
				NPArray.Input("index",
					tooltip="Integer array of any shape - typically an [H,W] "
					        "label map. Each element is replaced by the "
					        "corresponding table row."),
				io.Combo.Input("out_of_range", options=["clamp", "wrap", "fill"],
					default="clamp",
					tooltip="What to do with an index outside 0..K-1: clamp to "
					        "the nearest row, wrap around (modulo K, for a "
					        "repeating palette), or write 'fill_value'."),
				io.Float.Input("fill_value", default=0.0, min=-1e12, max=1e12,
					optional=True, advanced=True,
					tooltip="Value written for out-of-range indices when "
					        "out_of_range is 'fill' (cast to the table's dtype)."),
			],
			outputs=[
				NPArray.Output(display_name="values",
					tooltip="The gathered array: the shape of 'index' plus the "
					        "table's trailing axes, in the table's dtype. An "
					        "[H,W] index over a (K,3) table gives [H,W,3] - cast "
					        "it to uint8 with 'CV Cast Array' to preview it."),
			],
		)

	@classmethod
	def execute(cls, table, index, out_of_range, fill_value=0.0) -> io.NodeOutput:
		tab = np.zeros((0,), np.float32) if table is None else np.asarray(table)
		idx = np.zeros((0,), np.int64) if index is None else np.asarray(index)
		if tab.ndim == 0:
			raise ValueError("table must be an array of rows, not a scalar.")
		if not np.issubdtype(idx.dtype, np.integer):
			if not np.all(np.isfinite(idx)) or np.any(idx != np.floor(idx)):
				raise ValueError(
					f"index is {idx.dtype} with non-integer values - a lookup "
					"needs whole numbers. Use 'CV Cast Array' (int32) first."
				)
			idx = idx.astype(np.int64)
		rows = int(tab.shape[0])
		if rows == 0:
			# an empty table is a real outcome (0 detections, 0 regions), not a
			# wiring mistake: fill rather than raise
			if idx.size:
				logging.warning("CV Take By Index: empty table, filling %d "
				                "element(s) with %g.", idx.size, fill_value)
			return io.NodeOutput(np.full(idx.shape + tab.shape[1:],
			                             fill_value, tab.dtype))
		safe = idx % rows if out_of_range == "wrap" else np.clip(idx, 0, rows - 1)
		out = tab[safe]
		if out_of_range == "fill":
			bad = (idx < 0) | (idx >= rows)
			if bad.any():
				out[bad] = np.asarray(fill_value).astype(tab.dtype)
		return io.NodeOutput(out)


class CVPreview(io.ComfyNode):
	# The NaN/Inf marker must be a colour the ACTIVE palette cannot produce, or a
	# non-finite patch reads as a plausible value. A fixed constant cannot do that
	# once 'colormap' is the user's choice: pure red is outside viridis and outside
	# the grey ramp, but squarely inside JET, HOT and AUTUMN. So these are
	# CANDIDATES, best first, and `_nonfinite_bgr` keeps the first one the palette
	# stays clear of. Red leads, so viridis/grey still mark in red as they always
	# did. Documented in the node description - an undocumented marker is a
	# mystery, not a warning.
	_NONFINITE_CANDIDATES = (
		(0, 0, 255),      # red
		(255, 0, 255),    # magenta
		(255, 255, 0),    # cyan
		(255, 255, 255),  # white
		(0, 0, 0),        # black
	)
	_NONFINITE_BGR = _NONFINITE_CANDIDATES[0]
	# "far enough from every colour in the map" - half of one channel's full
	# range, in BGR euclidean distance. Red clears it against viridis (234) and
	# against the grey ramp (208), and fails against JET (0), which is the whole
	# point of measuring instead of asserting.
	_NONFINITE_MIN_DIST = 128.0

	# The channel collapse the retired 'Preview CV Array (advanced)' subgraph
	# wrapped this node in. The REDUCE_* spellings are cv2.reduce's own rtypes, so
	# a value saved by that subgraph carries over verbatim; MAGNITUDE_L2 is the
	# one cv2.reduce cannot do and the only correct collapse for a SIGNED field
	# (optical flow HxWx2, gradients), where sum and mean cancel to zero.
	REDUCE_KEEP = "KEEP_ALL"
	REDUCE_MAGNITUDE = "MAGNITUDE_L2"
	REDUCE_OPTIONS = [REDUCE_KEEP, *enums.REDUCE.options, REDUCE_MAGNITUDE]
	_REDUCERS = {
		"REDUCE_SUM": lambda a: a.sum(axis=2),
		"REDUCE_AVG": lambda a: a.mean(axis=2),
		"REDUCE_MAX": lambda a: a.max(axis=2),
		"REDUCE_MIN": lambda a: a.min(axis=2),
		"REDUCE_SUM2": lambda a: (a * a).sum(axis=2),
		REDUCE_MAGNITUDE: lambda a: np.sqrt((a * a).sum(axis=2)),
	}

	RANGE_PER_FRAME = "per frame (auto)"
	RANGE_BATCH = "batch (shared auto)"
	RANGE_MANUAL = "manual"
	RANGE_OPTIONS = [RANGE_PER_FRAME, RANGE_BATCH, RANGE_MANUAL]

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CVPreview",
			display_name="Preview CV Array",
			category=CATEGORY,
			description="Previews a raw ndarray directly - no manual conversion "
			            "needed. 'image' renders uint8/float data as-is, "
			            "'normalize' min-max stretches it (gradients, disparity, "
			            "score maps), 'heatmap' additionally applies a colormap. "
			            "Multi-channel normalize/heatmap previews draw channels "
			            "as a 2x2 quadrant view with one shared value scale, or "
			            "set 'reduce_channels' to collapse them into a single map "
			            "instead (the only way to preview more than 4 channels, "
			            "e.g. a LATENT). Enable 'color_scale' to draw a labelled "
			            "value bar beside the preview (the bar keeps a legible "
			            "height even for tiny or 1-pixel arrays). A batched input "
			            "previews EVERY frame; by default each frame stretches on "
			            "its OWN value range, which makes frames incomparable - "
			            "'range_mode' switches to one shared range or a fixed "
			            "vmin..vmax. The label column is sized to the widest "
			            "labels in the batch so all frames come out the same "
			            "width. NON-FINITE pixels (NaN/Inf, common in "
			            "distanceTransform, optical-flow and score maps) are "
			            "excluded from the value range and painted in a marker "
			            "colour the active colormap cannot produce, so they "
			            "cannot be mistaken for data: PURE RED (0, 0, 255) for "
			            "viridis and for the grey ramp, magenta or cyan for maps "
			            "that contain red (JET, HOT, AUTUMN). 'Inspect CV Data' "
			            "reports how many there were. Also returns the rendering "
			            "as IMAGE for further chaining.",
			inputs=[
				# not polytype.poly_input: its shared note says "frame 0 of a
				# batch", and this node renders the WHOLE batch
				io.MultiType.Input("nparray", polytype.IMAGEISH,
					tooltip="2-D or 3-D ndarray to render. image mode supports "
					        "1/3/4 channels; normalize/heatmap support 1/2/3/4 "
					        "channels and draw multi-channel inputs as quadrants. "
					        "More channels than that need 'reduce_channels'. "
					        "A batched IMAGE/MASK/LATENT or a 4-D [B,H,W,C] array "
					        "renders every frame. For other shapes use 'Inspect CV "
					        "Data' instead."),
				io.Combo.Input("render", options=["image", "normalize", "heatmap"],
					default="image",
					tooltip="image: show uint8/float values as-is. normalize: min-max "
					        "stretch to 0-255 (for gradients/disparity/score maps). "
					        "heatmap: normalize then apply the 'colormap' palette."),
				io.Boolean.Input("color_scale", default=False, optional=True, advanced=True,
					tooltip="Draw a vertical colour/value scale bar with min..max "
					        "labels next to the preview. The labels are the "
					        "array's value range ('normalize'/'heatmap' map that "
					        "range to the bar exactly). On a batch each frame gets "
					        "its own range unless 'range_mode' says otherwise, and "
					        "the label column is widened to the batch's widest "
					        "labels so the frames stay the same width (the surplus "
					        "is black)."),
				io.Combo.Input("reduce_channels", options=cls.REDUCE_OPTIONS,
					default=cls.REDUCE_KEEP, optional=True, advanced=True,
					tooltip="Collapse a multi-channel array to ONE 2-D map before "
					        "rendering, instead of drawing the channels as "
					        "quadrants. KEEP_ALL leaves the array alone. "
					        "REDUCE_SUM/AVG/MAX/MIN/SUM2 are cv2.reduce's "
					        "reductions across the channel axis. MAGNITUDE_L2 is "
					        "sqrt(sum of squares) - use it for SIGNED fields "
					        "(optical flow, gradients), where AVG and SUM cancel "
					        "to near zero. This is also the only way to preview an "
					        "array with more than 4 channels, such as a LATENT. "
					        "The reduction runs in float32, so SUM cannot wrap."),
				io.Combo.Input("colormap", options=enums.COLORMAP.options,
					default=enums.COLORMAP.default, optional=True, advanced=True,
					tooltip="Palette for 'heatmap' (ignored by the other render "
					        "modes). VIRIDIS/INFERNO are perceptually uniform and "
					        "the right default for data; JET is the classic "
					        "rainbow and misreads as structure. The colour bar and "
					        "the NaN marker both follow this choice."),
				io.Combo.Input("range_mode", options=cls.RANGE_OPTIONS,
					default=cls.RANGE_PER_FRAME, optional=True, advanced=True,
					tooltip="Which value range 'normalize'/'heatmap' stretch to "
					        "0-255 (ignored by 'image'). per frame: each frame "
					        "uses its own min..max - the default, but it makes the "
					        "frames of a batch incomparable and a near-static "
					        "sequence flicker. batch: one min..max shared by every "
					        "frame. manual: the fixed 'vmin'..'vmax' below, with "
					        "out-of-range values clamped."),
				io.Float.Input("vmin", default=0.0, optional=True, advanced=True,
					tooltip="Bottom of the value range, used only when "
					        "'range_mode' is manual. If vmin >= vmax the node "
					        "falls back to the automatic range rather than "
					        "failing."),
				io.Float.Input("vmax", default=1.0, optional=True, advanced=True,
					tooltip="Top of the value range, used only when 'range_mode' "
					        "is manual."),
				io.Float.Input("clip_percent", default=0.0, min=0.0, max=49.0, step=0.1,
					optional=True, advanced=True,
					tooltip="Robust stretch for the automatic range modes: ignore "
					        "this percentage of values at EACH end when picking "
					        "min..max, so a single outlier (a hot pixel, a huge "
					        "distanceTransform corner) cannot flatten the rest of "
					        "the map to black. 2.0 is a good starting point. "
					        "0 = plain min..max. Ignored when 'range_mode' is "
					        "manual."),
			],
			outputs=[io.Image.Output(display_name="image")],
			is_output_node=True,
		)

	@classmethod
	def _finite_range(cls, data) -> tuple[float, float]:
		data = np.asarray(data)
		if not (np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_)):
			return 0.0, 0.0
		finite = data[np.isfinite(data)]
		if finite.size == 0:
			return 0.0, 0.0
		return float(finite.min()), float(finite.max())

	@classmethod
	def _nonfinite_pixel_mask(cls, data) -> np.ndarray:
		data = np.asarray(data)
		if not (np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_)):
			return np.zeros(data.shape[:2], dtype=bool)
		ok = np.isfinite(data)
		return ~np.all(ok, axis=2) if data.ndim == 3 else ~ok

	@classmethod
	def _nonfinite_bgr(cls, colormap=None) -> tuple:
		"""Marker colour for NaN/Inf: the first candidate `colormap` cannot make.

		'image'/'normalize' render through a grey ramp (colormap None), 'heatmap'
		through an OpenCV COLORMAP_*. Either way the marker has to sit outside
		that palette, which is measured against the map's own 256-entry LUT
		rather than assumed - see `_NONFINITE_CANDIDATES`. If nothing clears the
		threshold (a map that covers the cube), the farthest candidate wins.
		"""
		ramp = np.arange(256, dtype=np.uint8).reshape(-1, 1)
		lut = (cv2.applyColorMap(ramp, colormap) if colormap is not None
		       else cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR))
		lut = lut.reshape(-1, 3).astype(np.int32)
		best, best_dist = cls._NONFINITE_CANDIDATES[0], -1.0
		for cand in cls._NONFINITE_CANDIDATES:
			dist = float(np.sqrt(((lut - np.array(cand, np.int32)) ** 2).sum(axis=1)).min())
			if dist >= cls._NONFINITE_MIN_DIST:
				return cand
			if dist > best_dist:
				best, best_dist = cand, dist
		return best

	@classmethod
	def _reduce_channels(cls, arr, mode) -> np.ndarray:
		"""Collapse the channel axis to one 2-D map (the retired 'advanced' subgraph).

		KEEP_ALL and 2-D input pass straight through, so the 2x2 quadrant view is
		unchanged. Everything else runs in float32: REDUCE_SUM on uint8 would
		wrap, and the point of collapsing is to look at the values.
		"""
		arr = np.asarray(arr)
		fn = cls._REDUCERS.get(mode)
		if fn is None or arr.ndim != 3 or arr.size == 0:
			return arr
		return fn(arr.astype(np.float32))

	@classmethod
	def _value_span(cls, arr, range_mode, vmin, vmax, clip_percent):
		"""Explicit (lo, hi) to stretch between, or None for 'let cv2.normalize do it'.

		None is the untouched default path - per-frame auto range, no clipping.
		Every other mode needs the span up front, and 'batch' needs one even when
		it is just the frame's own range, so the caller can take the min/max
		across the batch. A span that is not a real interval (manual vmin >= vmax,
		a constant frame) returns None: rule 4, a nonsense knob falls back, it
		does not halt a graph.
		"""
		if range_mode == cls.RANGE_MANUAL:
			if float(vmax) > float(vmin):
				return float(vmin), float(vmax)
			return None
		if not (clip_percent > 0 or range_mode == cls.RANGE_BATCH):
			return None
		data = np.asarray(arr)
		if not (np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_)):
			return None
		finite = data[np.isfinite(data)]
		if finite.size == 0:
			return None
		if clip_percent > 0:
			lo, hi = np.percentile(finite.astype(np.float64),
			                       [float(clip_percent), 100.0 - float(clip_percent)])
		else:
			lo, hi = finite.min(), finite.max()
		return (float(lo), float(hi)) if float(hi) > float(lo) else None

	@classmethod
	def _mark_nonfinite(cls, frame: np.ndarray, mask: np.ndarray, marker=None) -> np.ndarray:
		if not mask.any():
			return frame
		frame = imageutils.ensure_uint8(frame)
		if frame.ndim == 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		elif frame.shape[2] == 1:
			frame = cv2.cvtColor(frame[..., 0], cv2.COLOR_GRAY2BGR)
		elif frame.shape[2] == 4:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
		frame = frame.copy()
		frame[mask] = cls._NONFINITE_BGR if marker is None else marker
		return frame

	@classmethod
	def _channel_quadrants(cls, data, fill_value):
		data = np.asarray(data)
		if data.ndim == 2:
			return data
		if data.shape[2] == 1:
			return data[..., 0]
		h, w, c = data.shape
		tiles = []
		for i in range(4):
			if i < c:
				tiles.append(data[..., i])
			else:
				tiles.append(np.full((h, w), fill_value, dtype=data.dtype))
		top = np.concatenate((tiles[0], tiles[1]), axis=1)
		bottom = np.concatenate((tiles[2], tiles[3]), axis=1)
		return np.concatenate((top, bottom), axis=0)

	@classmethod
	def _input_frames(cls, value) -> list:
		"""Per-frame arrays of an image-ish input: every frame of a batch, else one.

		A batched IMAGE ([B,H,W,C]), MASK ([B,H,W]), LATENT ({samples: [B,C,H,W]})
		or 4-D NPARRAY previews EVERY frame; anything else goes through
		polytype.resolve (frame 0 / the array itself), as before.
		"""
		if isinstance(value, torch.Tensor) and value.ndim == 4 and value.shape[0] > 1:
			return imageutils.image_batch_to_bgr(value)
		if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] > 1:
			return imageutils.mask_batch_to_u8(value)
		if polytype.is_latent(value):
			samples = value["samples"]
			if samples.ndim == 4 and samples.shape[0] > 1:
				return [np.ascontiguousarray(s.movedim(0, -1).float().cpu().numpy())
				        for s in samples]
		arr = np.asarray(polytype.resolve(value)[0])
		if arr.ndim == 4:  # [B, H, W, C] NPARRAY batch
			return [arr[i] for i in range(arr.shape[0])]
		return [arr]

	@classmethod
	def _pad_to_common(cls, frames: list) -> list:
		"""Black-pad right/bottom so every frame of a batch stacks into one IMAGE.

		With `color_scale` the label column is already sized batch-wide, so this
		is normally a no-op; it keeps the invariant if a frame differs anyway.
		"""
		if len(frames) < 2:
			return frames
		h = max(f.shape[0] for f in frames)
		w = max(f.shape[1] for f in frames)
		out = []
		for f in frames:
			dh, dw = h - f.shape[0], w - f.shape[1]
			if dh or dw:
				f = np.pad(f, [(0, dh), (0, dw)] + [(0, 0)] * (f.ndim - 2))
			out.append(f)
		return out

	@classmethod
	def execute(cls, nparray, render, color_scale=False,
	            reduce_channels=REDUCE_KEEP, colormap=enums.COLORMAP.default,
	            range_mode=RANGE_PER_FRAME, vmin=0.0, vmax=1.0,
	            clip_percent=0.0) -> io.NodeOutput:
		frames = cls._input_frames(nparray)
		if not frames:
			raise ValueError(
				"Cannot render an empty batch. Use 'Inspect CV Data' to check the "
				"data first.")
		frames = [cls._reduce_channels(arr, reduce_channels) for arr in frames]
		cmap = enums.COLORMAP.resolve(colormap) if render == "heatmap" else None
		spans = [cls._value_span(arr, range_mode, vmin, vmax, clip_percent)
		         for arr in frames]
		if range_mode == cls.RANGE_BATCH:
			# one range for the whole batch, so the same colour means the same
			# value in every frame - frames whose own range is degenerate (a
			# constant map) contribute nothing and inherit the shared one
			real = [s for s in spans if s is not None]
			shared = (min(s[0] for s in real), max(s[1] for s in real)) if real else None
			spans = [shared] * len(frames)
		rendered = [cls._render_frame(np.asarray(arr), render, cmap, span)
		            for arr, span in zip(frames, spans)]
		if color_scale:
			# one label column for the whole batch: the tick labels are the only
			# data-dependent part of the canvas width, so sizing them to the
			# widest frame's labels makes every frame come out the same width
			label_w = max(imageutils.color_scale_label_width(mn, mx)
			              for _, mn, mx, _ in rendered)
			out = [imageutils.attach_color_scale(f, mn, mx, cmap, label_width=label_w)
			       for f, mn, mx, cmap in rendered]
		else:
			out = [f for f, _, _, _ in rendered]
		image = imageutils.bgr_to_image_batch(cls._pad_to_common(out))
		return io.NodeOutput(image, ui=ui.PreviewImage(image, cls=cls))

	@classmethod
	def _render_frame(cls, arr, render, cmap=None, span=None):
		"""One frame -> (uint8 image, value min, value max, colormap or None).

		`cmap` is the resolved COLORMAP_* for 'heatmap' (None = viridis, the
		default the caller resolves); `span` is an explicit (lo, hi) to stretch
		between, or None to min-max the frame as before.
		"""
		channels = arr.shape[2] if arr.ndim == 3 else 1
		bad_channels = channels not in (1, 2, 3, 4) or (render == "image" and channels == 2)
		if arr.size == 0 and arr.ndim in (2, 3):
			# an EMPTY array is a normal outcome (0 detections, 0 regions, an
			# empty IoU matrix), not a wiring mistake: a preview that raised
			# would halt the whole graph on a frame that found nothing
			return np.zeros((16, 16, 3), np.uint8), 0.0, 0.0, None
		if arr.ndim not in (2, 3) or bad_channels or arr.size == 0:
			raise ValueError(
				f"Cannot render array of shape {arr.shape} (dtype {arr.dtype}) as an image. "
				"Set 'reduce_channels' to collapse the channel axis, or use "
				"'Inspect CV Data' for non-image data."
			)
		if render == "image":
			scale_min, scale_max = cls._finite_range(arr)
			nonfinite_mask = cls._nonfinite_pixel_mask(arr)
			frame = imageutils.ensure_uint8(arr)
			frame = cls._mark_nonfinite(frame, nonfinite_mask, cls._nonfinite_bgr(None))
			return frame, scale_min, scale_max, None
		if render == "heatmap" and cmap is None:
			cmap = cv2.COLORMAP_VIRIDIS
		data = arr.astype(np.float32)
		scale_min, scale_max = cls._finite_range(data)
		if span is not None:
			# an explicit range (manual, batch-shared or percentile-clipped) is
			# what the colour bar must label too, so it replaces the frame's own
			scale_min, scale_max = span
		if data.ndim == 3 and data.shape[2] > 1:
			nonfinite_mask = cls._channel_quadrants(~np.isfinite(data), False)
		else:
			nonfinite_mask = cls._nonfinite_pixel_mask(data)
		# non-finite values (NaN/Inf from distanceTransform / optical flow /
		# score maps) would poison cv2.normalize into an all-black frame;
		# clamp them to the finite range so the real structure still shows
		if nonfinite_mask.any():
			data = np.nan_to_num(data, nan=scale_min, posinf=scale_max, neginf=scale_min)
		data = cls._channel_quadrants(data, scale_min)
		if span is None:
			frame = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		else:
			lo, hi = span
			frame = np.round(np.clip((data - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
		if render == "heatmap":
			frame = cv2.applyColorMap(frame, cmap)
		frame = cls._mark_nonfinite(frame, nonfinite_mask,
		                            cls._nonfinite_bgr(cmap if render == "heatmap" else None))
		return frame, scale_min, scale_max, cmap if render == "heatmap" else None


class CVInspect(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CVInspect",
			display_name="Inspect CV Data",
			category=CATEGORY,
			description="Debug helper: shows shape, dtype and value statistics of "
			            "any value (ndarrays, tuples, scalars...). Connect any "
			            "output to understand what a cv2 function returned.",
			inputs=[io.AnyType.Input("value",
				tooltip="Any value to inspect - ndarray, tensor, tuple/list, scalar. "
				        "Wire the 'summary' output into core 'Preview as Text' "
				        "(PreviewAny) to actually display it.")],
			outputs=[io.String.Output(display_name="summary")],
			is_output_node=True,
		)

	@classmethod
	def _summarize(cls, value, indent="") -> str:
		if isinstance(value, np.ndarray):
			lines = [f"{indent}ndarray shape={value.shape} dtype={value.dtype}"]
			if value.size > 0 and np.issubdtype(value.dtype, np.bool_):
				true = int(value.sum())
				lines.append(f"{indent}  True: {true}/{value.size} ({true / value.size:.1%})")
			elif value.size > 0 and np.issubdtype(value.dtype, np.number):
				# compute over finite values so a stray NaN/Inf (common in
				# distanceTransform / optical-flow / score-map outputs) does not
				# poison the stats into 'nan'; report how many were dropped
				finite, nonfinite = value, 0
				if np.issubdtype(value.dtype, np.floating):
					ok = np.isfinite(value)
					nonfinite = int(value.size - int(ok.sum()))
					finite = value[ok]
				if finite.size:
					lines.append(
						f"{indent}  min={finite.min():.6g} max={finite.max():.6g} "
						f"mean={finite.mean():.6g}"
					)
				if nonfinite:
					lines.append(f"{indent}  non-finite (NaN/Inf): {nonfinite}/{value.size}")
			return "\n".join(lines)
		if isinstance(value, torch.Tensor):
			return (f"{indent}torch.Tensor shape={tuple(value.shape)} dtype={value.dtype} "
			        f"device={value.device}")
		if isinstance(value, (tuple, list)):
			lines = [f"{indent}{type(value).__name__} of {len(value)} items:"]
			for i, item in enumerate(value[:8]):
				lines.append(f"{indent}[{i}]: {cls._summarize(item, indent + '  ').lstrip()}")
			if len(value) > 8:
				lines.append(f"{indent}... {len(value) - 8} more")
			return "\n".join(lines)
		text = repr(value)
		return f"{indent}{type(value).__name__}: {text[:500]}"

	@classmethod
	def execute(cls, value) -> io.NodeOutput:
		summary = cls._summarize(value)
		return io.NodeOutput(summary, ui=ui.PreviewText(summary))


def _camera_param_dirs():
	"""Output dir first (where Save writes), then input dir."""
	import folder_paths
	return [folder_paths.get_output_directory(), folder_paths.get_input_directory()]


def _list_camera_param_files():
	"""`.json` files visible to Load, output dir before input dir, de-duped."""
	seen, files = set(), []
	for d in _camera_param_dirs():
		try:
			names = os.listdir(d)
		except OSError:
			continue
		for f in sorted(names):
			if f.lower().endswith(".json") and f not in seen \
					and os.path.isfile(os.path.join(d, f)):
				seen.add(f)
				files.append(f)
	return files


def _resolve_camera_param_file(name):
	"""Find a picked name in output dir, then input dir."""
	for d in _camera_param_dirs():
		path = os.path.join(d, name)
		if os.path.isfile(path):
			return path
	return None


# --- array <-> the rest of the graph ---------------------------------------
#
# Numbers computed in an array are trapped there: they cannot be written to a
# file, pasted into a report, sorted with basic_data_handling's list nodes or
# iterated with Inspire's foreach.  These three nodes are the way out and back.
# The INBOUND text half already exists ('Parse Matrix' reads a whitespace/comma
# table), so only the formatter is needed here - pair it with
# 'Basic data handling: save STRING to file' rather than adding a file picker.

_TEXT_FORMATS = ["CSV (comma)", "TSV (tab)", "plain (space)", "JSON"]


class ArrayToText(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ArrayToText",
			display_name="CV Array To Text",
			category=CATEGORY,
			description="Formats an array as a text table - the way measurements "
			            "LEAVE the graph. Wire it into core 'Preview as Text' to "
			            "read the numbers, or into 'Basic data handling: save "
			            "STRING to file' to write a .csv (this node deliberately "
			            "has no file picker - the helper pack already does disk "
			            "IO). 'Parse Matrix' is the return leg, so a table can be "
			            "written, re-read and plotted. Unlike 'Inspect CV Data', "
			            "which SUMMARISES, this emits every value.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="Array to format. 1-D becomes one row, 2-D one row per "
					        "row; anything higher is flattened to (-1, last axis) "
					        "for the table formats and kept fully nested for JSON."),
				io.Combo.Input("format", options=_TEXT_FORMATS,
					tooltip="CSV/TSV/plain emit one line per row with the chosen "
					        "separator - 'Parse Matrix' reads all three back. JSON "
					        "emits a nested list, which core 'Extract Text from "
					        "JSON' and the chart nodes can consume."),
				io.Int.Input("decimals", default=4, min=-1, max=17,
					tooltip="Digits after the decimal point for float arrays. -1 "
					        "prints full repeatable precision (17 digits) - use it "
					        "when the text is a round trip, not a report. Integer "
					        "arrays always print exactly."),
				io.String.Input("header", default="", optional=True, advanced=True,
					tooltip="Optional column names, comma-separated, written as the "
					        "first line (table formats only). Keep it ASCII - "
					        "workflow JSON is read with the Windows codepage."),
				io.Int.Input("max_elements", default=100000, min=1, max=10000000,
					optional=True, advanced=True,
					tooltip="Safety cap: stop after this many values and report it "
					        "on 'truncated'. Formatting a whole image would produce "
					        "a multi-megabyte string that the UI has to carry."),
			],
			outputs=[
				io.String.Output(display_name="text",
					tooltip="The formatted table. Newline-separated; no trailing "
					        "newline, so 'Parse Matrix' reads it back unchanged."),
				io.Int.Output(display_name="rows",
					tooltip="Number of data rows written (excluding the header)."),
				io.Boolean.Output(display_name="truncated",
					tooltip="True when 'max_elements' cut the table short - the "
					        "text is then incomplete and must not be round-tripped."),
			],
		)

	@classmethod
	def execute(cls, nparray, format, decimals=4, header="",
			max_elements=100000) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.dtype == object:
			raise ValueError(
				"Cannot format an object array as text - it holds python objects, "
				"not numbers. Inspect it with 'Inspect CV Data' instead.")
		cap = int(max_elements)
		truncated = bool(arr.size > cap)

		is_int = np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)
		if is_int:
			fmt = lambda v: str(int(v))
		elif int(decimals) < 0:
			fmt = lambda v: repr(float(v))
		else:
			spec = f"%.{int(decimals)}f"
			fmt = lambda v: spec % float(v)

		if format == _TEXT_FORMATS[3]:
			flat = arr.reshape(-1)[:cap] if truncated else arr
			# json.dumps would print full float repr and reject NaN/Inf, which
			# every score map is full of; format each leaf the same way the
			# table formats do so the two agree.
			def dump(a):
				a = np.asarray(a)
				if a.ndim == 0:
					return fmt(a)
				return "[" + ", ".join(dump(v) for v in a) + "]"
			text = dump(flat)
			rows = int(flat.shape[0]) if flat.ndim else 1
		else:
			sep = {"CSV (comma)": ", ", "TSV (tab)": "\t"}.get(format, " ")
			table = arr.reshape(1, -1) if arr.ndim <= 1 else \
				arr.reshape(-1, arr.shape[-1])
			if arr.size == 0:
				table = table.reshape(0, table.shape[-1] if table.ndim > 1 else 0)
			width = table.shape[1] if table.ndim > 1 else 0
			limit = max(1, cap // width) if width else 0
			if truncated and width:
				table = table[:limit]
			lines = []
			if header:
				lines.append(sep.join(
					p.strip() for p in str(header).split(",")))
			for row in table:
				lines.append(sep.join(fmt(v) for v in np.asarray(row).reshape(-1)))
			text = "\n".join(lines)
			rows = int(table.shape[0]) if table.ndim > 1 else 0

		return io.NodeOutput(text, rows, truncated)


class ArrayToNumbers(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ArrayToNumbers",
			display_name="CV Array To Numbers",
			category=CATEGORY,
			description="Emits an array's values as a ComfyUI LIST of FLOATs - the "
			            "bridge into the helper packs. Downstream executes once per "
			            "value, which is how an Inspire foreach iterates detections, "
			            "and 'Basic data handling: convert Data List to LIST' turns "
			            "it into that pack's whole sort/filter/min/max surface. For "
			            "arrays, not images: keep the count small (each value is a "
			            "separate execution). 'CV Numbers To Array' is the return "
			            "leg; to iterate whole ROWS as arrays use 'CV Unstack "
			            "Batch' instead.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="Array to read. A 2-D array is read row by row unless "
					        "'column' picks one column out of it."),
				io.Int.Input("column", default=-1, min=-1, max=4095,
					tooltip="For a 2-D array: which column to emit (-1 = every "
					        "value, row by row). Column 4 of a detection table is "
					        "its score, so this is how a score column reaches the "
					        "list nodes."),
				io.Int.Input("max_items", default=1024, min=1, max=100000,
					optional=True, advanced=True,
					tooltip="Safety cap on the number of values emitted. Each one "
					        "is a separate downstream execution, so a whole image "
					        "here would run the rest of the graph a million times."),
			],
			outputs=[
				io.Float.Output(display_name="value", is_output_list=True,
					tooltip="One FLOAT per value, in order. Cast with 'Basic data "
					        "handling: to INT' if a consumer needs an integer - INT "
					        "and FLOAT sockets do not auto-coerce."),
				io.Int.Output(display_name="count",
					tooltip="How many values were emitted (after the cap)."),
			],
		)

	@classmethod
	def execute(cls, nparray, column=-1, max_items=1024) -> io.NodeOutput:
		arr = np.asarray(nparray)
		if arr.dtype == object:
			raise ValueError("Cannot read an object array as numbers.")
		if int(column) >= 0:
			flat = arr.reshape(1, -1) if arr.ndim <= 1 else arr.reshape(-1, arr.shape[-1])
			if int(column) >= flat.shape[-1]:
				raise ValueError(
					f"column {int(column)} is out of range: the array has "
					f"{flat.shape[-1]} column(s) (shape {tuple(arr.shape)}).")
			values = flat[:, int(column)].reshape(-1)
		else:
			values = arr.reshape(-1)
		values = values[:int(max_items)]
		out = [float(v) for v in values]
		# an output list must never be empty or the executor has nothing to run:
		# emit a single 0.0 and say so with count=0
		return io.NodeOutput(out if out else [0.0], len(out))


class NumbersToArray(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_NumbersToArray",
			display_name="CV Numbers To Array",
			category=CATEGORY,
			description="Collects a ComfyUI LIST of FLOATs back into one NPARRAY - "
			            "the return leg of 'CV Array To Numbers' and the way results "
			            "computed in the helper packs (a foreach body, 'Math "
			            "Expression', a sorted list) re-enter the array world. "
			            "Reshapes into rows on the way in, so a flat stream of "
			            "numbers becomes a point set or a matrix.",
			inputs=[
				io.Float.Input("values", force_input=True,
					tooltip="A ComfyUI list of FLOATs (any node that emits a list, "
					        "e.g. 'CV Array To Numbers', 'Basic data handling: "
					        "convert LIST to Data List', an Inspire foreach body). "
					        "A single value yields a 1-element array."),
				io.Int.Input("columns", default=0, min=0, max=4096,
					tooltip="Row width. 0 keeps the result 1-D; 2 makes an (N,2) "
					        "point set, 3 an (N,3) cloud. The value count must "
					        "divide by it."),
				io.Combo.Input("dtype",
					options=["float32", "float64", "int32", "uint8"],
					default="float32",
					tooltip="Element type of the result. float32 is what most cv2 "
					        "geometry functions want; int32 for indices/labels."),
			],
			outputs=[
				NPArray.Output(display_name="nparray",
					tooltip="The collected array, (N,) or (N, columns)."),
				io.Int.Output(display_name="count",
					tooltip="Number of rows in the result."),
			],
			is_input_list=True,
		)

	@classmethod
	def execute(cls, values, columns=0, dtype="float32") -> io.NodeOutput:
		# is_input_list is a WHOLE-NODE flag: every input arrives wrapped,
		# widgets included - and a direct execute() call in a test does not
		# wrap. Unwrap the widgets defensively, exactly like CV_Stitch_List.
		_unwrap = lambda v: v[0] if isinstance(v, list) else v
		cols = int(_unwrap(columns) or 0)
		kind = str(_unwrap(dtype) or "float32")

		items = values if isinstance(values, (list, tuple)) else [values]
		flat = []
		for item in items:
			if isinstance(item, (list, tuple, np.ndarray)):
				flat.extend(float(v) for v in np.asarray(item).reshape(-1))
			elif item is not None:
				flat.append(float(item))
		arr = np.asarray(flat, np.float64)
		if cols > 0:
			if arr.size % cols:
				raise ValueError(
					f"{arr.size} value(s) do not divide into rows of {cols}.")
			arr = arr.reshape(-1, cols)
		out = arr.astype(np.dtype(kind))
		return io.NodeOutput(np.ascontiguousarray(out), int(out.shape[0]))


class SaveCameraParams(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SaveCameraParams",
			display_name="CV Save Camera Params (JSON)",
			category=CATEGORY,
			description="Writes a camera matrix + distortion coefficients to a "
			            "human-readable JSON file in ComfyUI's output folder, so a "
			            "calibration done once ('Calibrate Camera (Chessboard)') can "
			            "be reloaded later with 'CV Load Camera Params (JSON)' instead "
			            "of recalibrating. Arrays are stored as nested lists (shapes "
			            "preserved); image size and RMS error are optional metadata "
			            "(leave at 0 to omit). This is an output node: it has no "
			            "outputs at all and still runs - the file name it wrote is "
			            "shown on the node.",
			inputs=[
				NPArray.Input("camera_matrix",
					tooltip="3x3 intrinsic matrix K (e.g. from Calibrate Camera "
					        "or CV Camera Matrix)."),
				NPArray.Input("dist_coeffs",
					tooltip="Distortion coefficients (k1, k2, p1, p2, k3...)."),
				io.String.Input("filename", default="camera_params",
					tooltip="Output file name; '.json' is appended if missing. "
					        "Path components are stripped - the file always lands "
					        "directly in the output folder."),
				io.Int.Input("image_width", default=0, min=0, max=100000,
					tooltip="Calibration image width in px, recorded as metadata. "
					        "0 = omit.", optional=True, advanced=True),
				io.Int.Input("image_height", default=0, min=0, max=100000,
					tooltip="Calibration image height in px, recorded as metadata. "
					        "0 = omit.", optional=True, advanced=True),
				io.Float.Input("rms_error", default=0.0, min=0.0, max=1e9,
					tooltip="Reprojection RMS (px) from Calibrate Camera, recorded "
					        "as metadata. 0 = omit.", optional=True, advanced=True),
			],
			outputs=[],
			is_output_node=True,
		)

	@classmethod
	def execute(cls, camera_matrix, dist_coeffs, filename,
	            image_width=0, image_height=0, rms_error=0.0) -> io.NodeOutput:
		import folder_paths
		name = os.path.basename(str(filename)).strip()
		if not name:
			name = "camera_params"
		if not name.lower().endswith(".json"):
			name += ".json"
		data = {
			"camera_matrix": np.asarray(camera_matrix, np.float64).tolist(),
			"dist_coeffs": np.asarray(dist_coeffs, np.float64).tolist(),
		}
		if int(image_width) > 0 and int(image_height) > 0:
			data["image_width"] = int(image_width)
			data["image_height"] = int(image_height)
		if float(rms_error) > 0:
			data["rms_error"] = float(rms_error)
		path = os.path.join(folder_paths.get_output_directory(), name)
		with open(path, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=2)
		# Only the folder-relative name is shown: an absolute path in a preview
		# is the leak 'CV Build Information' / 'CV Install Example Inputs' redact
		# (drive layout, install location, user name) once a screenshot is shared.
		return io.NodeOutput(ui=ui.PreviewText(f"Saved camera params -> output/{name}"))


class LoadCameraParams(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_LoadCameraParams",
			display_name="CV Load Camera Params (JSON)",
			category=CATEGORY,
			description="Reads a camera matrix + distortion coefficients back from a "
			            "JSON file written by 'CV Save Camera Params (JSON)'. The "
			            "dropdown lists '.json' files in the output folder (where Save "
			            "writes) and then the input folder. Outputs feed cv2.undistort, "
			            "cv2.solvePnP, projectPoints, stereoRectify... image_width / "
			            "image_height / rms_error are 0 when the file did not record "
			            "them. Refresh the node definitions (reload the page) to pick "
			            "up files saved during the same session.",
			inputs=[io.Combo.Input("file", options=_list_camera_param_files(),
				tooltip="Saved camera-params JSON to load (output folder first, "
				        "then input folder).")],
			outputs=[
				NPArray.Output(display_name="camera_matrix",
					tooltip="3x3 float64 intrinsic matrix K."),
				NPArray.Output(display_name="dist_coeffs",
					tooltip="Distortion coefficients as stored (float64)."),
				io.Int.Output(display_name="image_width",
					tooltip="Recorded image width in px, or 0 if absent."),
				io.Int.Output(display_name="image_height",
					tooltip="Recorded image height in px, or 0 if absent."),
				io.Float.Output(display_name="rms_error",
					tooltip="Recorded reprojection RMS in px, or 0.0 if absent."),
			],
		)

	@classmethod
	def execute(cls, file) -> io.NodeOutput:
		path = _resolve_camera_param_file(file)
		if path is None:
			raise ValueError(f"Camera params file not found: {file!r}")
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		K = np.asarray(data["camera_matrix"], np.float64)
		dist = np.asarray(data["dist_coeffs"], np.float64)
		return io.NodeOutput(K, dist, int(data.get("image_width", 0)),
		                     int(data.get("image_height", 0)),
		                     float(data.get("rms_error", 0.0)))

	@classmethod
	def fingerprint_inputs(cls, file):
		import hashlib
		path = _resolve_camera_param_file(file)
		if path is None:
			return file
		m = hashlib.sha256()
		with open(path, "rb") as f:
			m.update(f.read())
		return m.digest().hex()

	@classmethod
	def validate_inputs(cls, file) -> bool | str:
		if _resolve_camera_param_file(file) is None:
			return f"Invalid camera params file: {file}"
		return True


def _rotation_to_quaternion(R):
	"""3x3 rotation matrix -> (x, y, z, w) unit quaternion (Shepperd's method)."""
	m00, m01, m02 = R[0]
	m10, m11, m12 = R[1]
	m20, m21, m22 = R[2]
	trace = m00 + m11 + m22
	if trace > 0:
		s = 2.0 * np.sqrt(trace + 1.0)
		w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
	elif m00 > m11 and m00 > m22:
		s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
		w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
	elif m11 > m22:
		s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
		w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
	else:
		s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
		w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
	return float(x), float(y), float(z), float(w)


class CameraPoseTo3DView(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CameraPoseTo3DView",
			display_name="CV Camera Pose To 3D View",
			category=CATEGORY,
			description="Adapts an OpenCV camera model to the LOAD3D_CAMERA dict the "
			            "native 3D preview nodes take ('Preview 3D (Advanced)' / "
			            "'Preview Splat' / 'Preview Point Cloud' camera_info input), so "
			            "a calibrated real camera drives the Three.js viewer camera. "
			            "Feed the intrinsic matrix from 'Calibrate Camera (Chessboard)' "
			            "or 'CV Load Camera Params (JSON)' and the extrinsics "
			            "(rvec/tvec) from 'CV Solve PnP (Pose)': the viewer camera "
			            "is placed at the physical camera's position relative to the "
			            "board/object, looking down its optical axis. Coordinates are "
			            "converted from OpenCV (right-handed, Y down, camera looks +Z) "
			            "to Three.js (right-handed, Y up, camera looks -Z) by negating "
			            "Y and Z. The orbit target is the point on the optical axis "
			            "nearest orbit_anchor, so OrbitControls' lookAt reproduces the "
			            "pose's viewing direction exactly (camera roll is lost - "
			            "OrbitControls keeps +Y up). fov/aspect/quaternion are filled "
			            "from the intrinsics per the LOAD3D_CAMERA spec, but the "
			            "current frontend applies only position/target/zoom from a "
			            "camera_info input - the viewer keeps its own FOV. "
			            "Failure-tolerant: without rvec/tvec (or with the zero vectors "
			            "Solve PnP emits on found=false) the camera falls back to a "
			            "front view of orbit_anchor at 10*scale distance.",
			inputs=[
				NPArray.Input("camera_matrix",
					tooltip="3x3 intrinsic matrix K ('Calibrate Camera', 'CV Load "
					        "Camera Params' or 'CV Camera Matrix'). Sets the reported "
					        "vertical FOV: fov_y = 2*atan(image_height / (2*fy))."),
				NPArray.Input("rvec", optional=True,
					tooltip="3x1 Rodrigues rotation vector from 'CV Solve PnP "
					        "(Pose)' (a 3x3 rotation matrix is also accepted). Leave "
					        "unconnected for a frontal default view."),
				NPArray.Input("tvec", optional=True,
					tooltip="3x1 translation vector from 'CV Solve PnP (Pose)'. "
					        "Leave unconnected for a frontal default view."),
				NPArray.Input("orbit_anchor", optional=True,
					tooltip="3D point (in the SAME object/world coordinates as the "
					        "solve-PnP object points, e.g. the board centre from 'CV "
					        "Points' [[4, 2.5, 0]]) that the orbit pivot should sit "
					        "nearest to. The target is its projection onto the optical "
					        "axis, so orbiting rotates around the observed object. "
					        "Default: the object-space origin."),
				NPArray.Input("dist_coeffs", optional=True,
					tooltip="Lens distortion coefficients (k1, k2, p1, p2[, k3]) from "
					        "'Calibrate Camera' / 'CV Load Camera Params'. Rides along "
					        "in the dict (_dist_coeffs) for viewers that can render "
					        "distortion ('CV Preview 3D (Calibrated Camera)'); the core "
					        "preview nodes ignore it."),
				io.Int.Input("image_width", default=0, min=0, max=100000,
					tooltip="Calibration image width in px, for the aspect ratio. "
					        "0 = estimate as 2*cx from the camera matrix.",
					optional=True, advanced=True),
				io.Int.Input("image_height", default=0, min=0, max=100000,
					tooltip="Calibration image height in px, for the vertical FOV. "
					        "0 = estimate as 2*cy from the camera matrix.",
					optional=True, advanced=True),
				io.Float.Input("scale", default=1.0, min=1e-6, max=1e6,
					tooltip="Multiplier from object units (e.g. chessboard squares or "
					        "mm) to 3D scene units, applied to the camera position and "
					        "orbit target. Use it to size the camera distance to the "
					        "previewed model.",
					optional=True, advanced=True),
			],
			outputs=[
				io.Load3DCamera.Output(display_name="camera_info",
					tooltip="LOAD3D_CAMERA dict (position/target/zoom/cameraType + "
					        "quaternion/fov/aspect/near/far) - wire into the "
					        "camera_info input of 'Preview 3D (Advanced)', 'Preview "
					        "Splat' or 'Preview Point Cloud'."),
				io.Float.Output(display_name="fov_y_deg",
					tooltip="Vertical field of view implied by the intrinsics, in "
					        "degrees (2*atan(h/(2*fy)))."),
				io.Float.Output(display_name="distance",
					tooltip="Camera-to-target distance in scene units (after scale)."),
			],
		)

	@classmethod
	def execute(cls, camera_matrix, rvec=None, tvec=None, orbit_anchor=None,
	            dist_coeffs=None, image_width=0, image_height=0,
	            scale=1.0) -> io.NodeOutput:
		scale = float(scale)
		K = None if camera_matrix is None else np.asarray(camera_matrix, np.float64)
		fov = 50.0
		aspect = 1.0
		if K is not None and K.shape == (3, 3) and np.all(np.isfinite(K)):
			fy, cx, cy = K[1, 1], K[0, 2], K[1, 2]
			w = float(image_width) if int(image_width) > 0 else 2.0 * cx
			h = float(image_height) if int(image_height) > 0 else 2.0 * cy
			if fy > 0 and h > 0:
				fov = float(np.degrees(2.0 * np.arctan(h / (2.0 * fy))))
			if w > 0 and h > 0:
				aspect = float(w / h)

		def _vec3(a):
			if a is None:
				return None
			v = np.asarray(a, np.float64).reshape(-1)
			return v[:3] if v.size >= 3 and np.all(np.isfinite(v[:3])) else None

		R = np.eye(3)
		rv = None if rvec is None else np.asarray(rvec, np.float64)
		if rv is not None and rv.size == 9:
			if np.all(np.isfinite(rv)):
				R = rv.reshape(3, 3)
		elif _vec3(rv) is not None:
			R, _ = cv2.Rodrigues(_vec3(rv).reshape(3, 1))
		t = _vec3(tvec)
		t = np.zeros(3) if t is None else t
		anchor = _vec3(orbit_anchor)
		anchor = np.zeros(3) if anchor is None else anchor

		flip = np.array([1.0, -1.0, -1.0])  # OpenCV world -> Three.js world (Y/Z negate)
		position = flip * (-R.T @ t) * scale
		forward = flip * R[2, :]  # optical axis (+Z in camera space) in world coords
		n = np.linalg.norm(forward)
		forward = forward / n if n > 1e-12 else np.array([0.0, 0.0, -1.0])
		anchor3 = flip * anchor * scale
		dist = float(np.dot(anchor3 - position, forward))
		if not np.isfinite(dist) or dist < 1e-9:
			# no usable pose (Solve PnP found=false emits zero vectors) or the
			# anchor is behind the camera: back out to a default framing of the
			# anchor along the current viewing direction
			dist = max(float(np.linalg.norm(anchor3 - position)), 10.0 * scale)
			position = anchor3 - forward * dist
		target = position + forward * dist
		# camera->world rotation in Three.js conventions: world flip on the left,
		# camera-space axis flip (Y down->up, looks +Z->-Z) on the right
		qx, qy, qz, qw = _rotation_to_quaternion((flip[:, None] * R.T) * flip[None, :])
		near = max(dist / 100.0, 1e-4)
		camera_info = {
			"position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
			"target": {"x": float(target[0]), "y": float(target[1]), "z": float(target[2])},
			"zoom": 1.0,
			"cameraType": "perspective",
			"quaternion": {"x": qx, "y": qy, "z": qz, "w": qw},
			"fov": fov,
			"aspect": aspect,
			"near": near,
			"far": max(dist * 100.0, near * 10.0),
		}
		# Underscore extras (not part of the LOAD3D_CAMERA TypedDict, ignored by
		# the core widgets): the raw OpenCV camera model, for sinks that can use
		# more than position/target/zoom. _extrinsics/_intrinsics are exactly what
		# the frontend's computeCameraFromMatrices (Preview3D result[3]/[4], see
		# the PreviewUI3D shim in opencv_nodes/__init__.py) takes to apply the
		# exact pose + the intrinsics' vertical FOV; _dist_coeffs/_image_size feed
		# 'CV Preview 3D (Calibrated Camera)'. Only embedded when the intrinsics
		# are usable (fy, cy > 0) - computeCameraFromMatrices derives the FOV as
		# 2*atan(cy/fy), which is meaningless for the identity-K fallback.
		if K is not None and K.shape == (3, 3) and np.all(np.isfinite(K)) \
				and K[1, 1] > 0 and K[1, 2] > 0:
			# re-derive t from the FINAL position so the fallback framing and the
			# scale multiplier stay consistent with the matrices path
			t_final = -R @ (flip * position)
			camera_info["_extrinsics"] = [
				[float(R[i, 0]), float(R[i, 1]), float(R[i, 2]), float(t_final[i])]
				for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
			camera_info["_intrinsics"] = [[float(v) for v in row] for row in K]
			d = None if dist_coeffs is None else np.asarray(dist_coeffs, np.float64).reshape(-1)
			if d is not None and d.size > 0 and np.all(np.isfinite(d)):
				camera_info["_dist_coeffs"] = [float(v) for v in d]
			w = float(image_width) if int(image_width) > 0 else 2.0 * float(K[0, 2])
			h = float(image_height) if int(image_height) > 0 else 2.0 * float(K[1, 2])
			camera_info["_image_size"] = [w, h]
		return io.NodeOutput(camera_info, fov, dist)


class Preview3DCalibrated(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Preview3DCalibrated",
			display_name="CV Preview 3D (Calibrated Camera)",
			category=CATEGORY,
			is_experimental=True,
			is_output_node=True,
			description="A self-contained Three.js viewer (web/cv_preview3d.js) "
			            "that renders a 3D model through the FULL OpenCV camera "
			            "model from 'CV Camera Pose To 3D View': the projection "
			            "matrix is built directly from fx/fy/cx/cy (off-axis "
			            "principal point and anamorphic pixels included), the pose "
			            "keeps the camera roll the core OrbitControls viewers drop, "
			            "and lens distortion (_dist_coeffs: k1, k2, p1, p2, k3) is "
			            "replicated with an overscanned render + per-pixel "
			            "Brown-Conrady warp shader - something no 4x4 projection "
			            "matrix can express. Wire bg_image with the calibration "
			            "photo and the render composites over it in exact register "
			            "(the AR overlay check). Toolbar: reset to the calibrated "
			            "pose, toggle distortion / background / the board-plane "
			            "grid; drag to orbit (orbiting drops roll until reset). "
			            "GLB/GLTF only (Three.js GLTFLoader); the canvas is "
			            "letterboxed to the calibration image's aspect so the "
			            "overlay lines up. Without _intrinsics in camera_info (the "
			            "bridge's degenerate-K fallback) it degrades to a plain "
			            "fov/aspect perspective view, distortion off. The 'render' "
			            "IMAGE output is the same composite produced server-side "
			            "(software rasterizer + cv2.undistortPoints lens warp, no "
			            "grid), so the calibrated view is also available as data "
			            "for further processing.",
			inputs=[
				io.MultiType.Input(
					"model_3d",
					types=[io.File3DGLB, io.File3DGLTF, io.File3DAny],
					tooltip="3D model from a 3D loader node (e.g. 'Load 3D "
					        "(Advanced)'). GLB is the reliable format: a .gltf with "
					        "external buffers/textures will not resolve them from "
					        "the temp folder."),
				io.Load3DCamera.Input("camera_info",
					tooltip="Camera dict from 'CV Camera Pose To 3D View'. The "
					        "underscore extras (_intrinsics/_dist_coeffs/"
					        "_image_size) unlock the faithful projection and the "
					        "distortion pass; a plain LOAD3D_CAMERA dict still "
					        "works as a simple perspective view."),
				io.Image.Input("bg_image", optional=True,
					tooltip="Backdrop composited behind the render - use the very "
					        "photo the pose was solved from to verify the virtual "
					        "camera matches the real one (the model should sit on "
					        "the photographed board)."),
				io.Combo.Input("shading",
					options=["lambert (headlight)", "unlit (raw colors)"],
					default="lambert (headlight)",
					tooltip="Lighting of the server-side render. 'lambert "
					        "(headlight)' shades faces by their angle to a "
					        "camera-side light (shows the 3D form); 'unlit (raw "
					        "colors)' outputs the texture/base colors untouched - "
					        "no lighting baked into the pixels, best when the "
					        "render is data for further compositing. The mask "
					        "output is unaffected.", optional=True),
				io.Float.Input("resolution_scale", default=1.0, min=0.25, max=4.0,
					tooltip="Multiplies the OUTPUT resolution of render + mask "
					        "(camera matrix scales with it, so the framing is "
					        "identical - 2.0 renders the same view at twice the "
					        "calibration image's width/height).", optional=True),
				io.Combo.Input("antialias",
					options=["off", "2x supersample", "4x supersample"],
					default="2x supersample",
					tooltip="Anti-aliasing of the server-side render: the scene is "
					        "rasterized and lens-warped at 2x/4x the output size, "
					        "then box-filtered down - smooths polygon and "
					        "silhouette edges (the mask edge gets fractional "
					        "values). 'off' renders 1:1 (fastest, hard edges).",
					optional=True),
				io.String.Input("object_transform", default="",
					tooltip="Model transform in Three.js world space, as JSON: "
					        '{"position": [x, y, z], "quaternion": [x, y, z, w], '
					        '"scale": [x, y, z]} (scale may be a single number; '
					        "missing keys default to identity; empty = identity). "
					        "The viewer's move/rotate/scale GIZMO writes this "
					        "widget as you drag, so the next run renders the model "
					        "where you left it - and the viewer's 'reset' button puts "
					        "the model back on whatever this widget says, so a mis-drag "
					        "costs one click once the text is right. Convert to an input to drive the "
					        "placement from upstream instead (the gizmo then only "
					        "previews): 'CV 3D Object Transform' writes this "
					        "string from readable position/rotation/scale widgets "
					        "- including a 'model_up' dropdown that stands a "
					        "Y-up model on the target plane instead of you "
					        "finding the quaternion - and emits the matching 4x4, "
					        "so the mesh lane moves with the render. 'CV Parse "
					        "Object Transform' reads a gizmo-authored string back "
					        "into the graph.",
					optional=True),
				NPArray.Input("scene_depth", optional=True,
					tooltip="Optional HxW float32 depth map of the PHOTOGRAPHED "
					        "scene - metric camera-space Z, in the SAME units as "
					        "the pose (metres if the calibration is). Wherever it "
					        "is nearer than the model, the model is cut away, so "
					        "a real car in front of the virtual one hides it "
					        "instead of the overlay floating on top. Build it "
					        "from stereo with cv2.reprojectImageTo3D + "
					        "'cv2 extractChannel' (coi=2), or from a metric "
					        "monocular depth net. Non-finite or <= 0 means 'no "
					        "measurement here' and never occludes, so the holes "
					        "a matcher leaves are safe. Resolution is free - it "
					        "is nearest-resampled to the render."),
				io.Float.Input("scene_depth_bias", default=0.0, min=-1000.0,
					max=1000.0, step=0.05, optional=True, advanced=True,
					tooltip="Added to scene_depth before the comparison, in "
					        "scene units. Stereo depth is noisy and the surface "
					        "the model stands ON is the one most likely to eat "
					        "it: a small positive bias (0.1-0.5 m on a driving "
					        "scene) pushes the scene back so the model wins ties. "
					        "Negative biases it the other way."),
			],
			outputs=[
				io.Image.Output(display_name="render",
					tooltip="Server-side software render (opencv_nodes/render3d.py) "
					        "of the model over bg_image, through the SAME camera "
					        "model as the live widget: overscanned pinhole "
					        "rasterization warped into the lens with "
					        "cv2.undistortPoints. No grid/axes - model and photo "
					        "only, ready for compositing or diffing against the "
					        "original photo. GLB triangles only."),
				io.Mask.Output(display_name="silhouette",
					tooltip="The rendered model's coverage as a MASK (1 = model, "
					        "0 = background), same resolution as the render, lens "
					        "distortion applied. With antialiasing on, edge pixels "
					        "carry fractional coverage. Feeds compositing, "
					        "Overlay Masks, bbox extraction..."),
				io.Mask.Output(display_name="depth",
					tooltip="OBJECT-relative depth as a MASK: 1 (white) at the "
					        "model's point nearest the camera, 0 (black) at its "
					        "farthest point - normalized over the model's own "
					        "camera-space extent (ALL vertices, occluded ones "
					        "included), never an arbitrary near/far range, so the "
					        "visible minimum need not reach 0 when the far side is "
					        "hidden. Background is 0; same resolution/distortion "
					        "as the render. Feeds depth ControlNets, DoF blurs, "
					        "fog compositing..."),
				io.Mask.Output(display_name="occlusion",
					tooltip="Where scene_depth HID the model: 1 on the pixels "
					        "the model would have covered but the photographed "
					        "scene is in front of, 0 everywhere else. All zeros "
					        "when scene_depth is not wired. Use it to check the "
					        "cut is landing on the right object, or to feather "
					        "the contact edge."),
				NPArray.Output(display_name="depth_metric",
					tooltip="The rasterizer's own HxW float32 z-buffer: camera-"
					        "space Z in SCENE UNITS - the same quantity 'CV "
					        "Rasterize Mesh' calls depth, and NOT the 'depth' "
					        "output above (which is normalized to [0, 1] for "
					        "viewing). This is what the scene_depth input of 'CV "
					        "Project Points (Sequence)' / the 'CV Annotate Model' "
					        "blueprint wants, so anchors on the far side of the "
					        "model get hidden without rasterizing it a second "
					        "time - and unlike 'CV Rasterize Mesh' it carries the "
					        "lens distortion, so it stays registered with the "
					        "render. +inf where nothing was drawn, which reads as "
					        "'no surface': a point projecting off the silhouette "
					        "stays visible. With scene_depth wired this is the "
					        "COMPOSITE surface (the photographed scene wherever "
					        "it won the depth test), so an anchor a real object "
					        "hides is hidden too."),
			],
		)

	@classmethod
	def execute(cls, model_3d, camera_info, bg_image=None,
	            shading="lambert (headlight)", resolution_scale=1.0,
	            antialias="2x supersample", object_transform="",
	            scene_depth=None, scene_depth_bias=0.0) -> io.NodeOutput:
		import uuid
		import folder_paths
		temp_dir = folder_paths.get_temp_directory()
		os.makedirs(temp_dir, exist_ok=True)
		fmt = getattr(model_3d, "format", None) or "glb"
		model_name = f"cv_preview3d_{uuid.uuid4().hex}.{fmt}"
		model_path = os.path.join(temp_dir, model_name)
		model_3d.save_to(model_path)
		payload = {"model": model_name, "camera": camera_info}
		transform = None
		raw = str(object_transform).strip()
		if raw:
			try:
				spec = json.loads(raw)
				if not isinstance(spec, dict):
					raise TypeError("not a JSON object")
			except (json.JSONDecodeError, TypeError) as e:
				raise ValueError(
					'object_transform must be JSON like {"position": [x, y, z], '
					f'"quaternion": [x, y, z, w], "scale": [x, y, z]}}, got {raw!r} ({e})'
				) from e
			transform = render3d.compose_transform(
				spec.get("position"), spec.get("quaternion"), spec.get("scale"))
			payload["transform"] = spec
		bgr = None
		if bg_image is not None:
			bg_name = f"cv_preview3d_bg_{uuid.uuid4().hex}.png"
			bgr = imageutils.image_batch_to_bgr(bg_image)[0]
			cv2.imwrite(os.path.join(temp_dir, bg_name), bgr)
			payload["bg"] = bg_name
		zmap = None
		if scene_depth is not None:
			zmap = np.asarray(polytype.resolve(scene_depth)[0], np.float32)
			if zmap.ndim == 3:
				zmap = zmap[:, :, 0]
			if zmap.ndim != 2:
				raise ValueError(
					"Preview3DCalibrated: scene_depth must be a single-channel "
					f"HxW depth map, got shape {np.asarray(scene_depth).shape}")
			# the widget occludes in the shader, so it needs the same map; a
			# packed RGB PNG is the only lossless way through a browser canvas
			rgb, z_max = render3d.encode_depth_png(zmap)
			depth_name = f"cv_preview3d_depth_{uuid.uuid4().hex}.png"
			cv2.imwrite(os.path.join(temp_dir, depth_name), rgb[:, :, ::-1])
			payload["depth"] = depth_name
			payload["depth_max"] = z_max
			payload["depth_bias"] = float(scene_depth_bias)
		ss = {"off": 1, "2x supersample": 2, "4x supersample": 4}.get(str(antialias), 2)
		render, silhouette, depth, occlusion, depth_metric = \
			render3d.render_calibrated(
				model_path, camera_info, bgr,
				shaded=not str(shading).startswith("unlit"),
				resolution_scale=float(resolution_scale), antialias=ss,
				model_transform=transform, scene_depth=zmap,
				depth_bias=float(scene_depth_bias))
		return io.NodeOutput(imageutils.bgr_to_image_batch([render]),
		                     torch.from_numpy(np.ascontiguousarray(silhouette))[None],
		                     torch.from_numpy(np.ascontiguousarray(depth))[None],
		                     torch.from_numpy(np.ascontiguousarray(occlusion))[None],
		                     np.ascontiguousarray(depth_metric, np.float32),
		                     ui={"cv3d": [payload]})


_OBJECT_SPACES = ["OpenCV (Y down, Z into the scene)", "glTF / Three.js (Y up)"]
_THREE_FLIP4 = np.diag([1.0, -1.0, -1.0, 1.0])

# Both are rotations; the difference is PASSIVE vs ACTIVE. 'space' declares
# which convention the widgets are read in - the two differ by Rx(180), so the
# same placement typed in either one emits the same string. 'model_up' turns
# the model inside whichever convention was selected. Different problems, hence
# two widgets - see the tooltips, which say so to the user as well.
_MODEL_UPS = ["leave as the file has it",
              "+Y is up (glTF / most GLB exports)",
              "+Z is up (Blender / CAD / OBJ)"]


def _model_up_rotation(model_up):
	"""3x3 that stands a model on the target plane, in the FILE's own axes
	(= Three.js axes: GLTFLoader applies no rotation of its own).

	'Up' in this pack is the calibration target's normal, OpenCV +Z, which is
	Three.js -Z - NOT Three.js +Y. So a stock +Y-up glTF needs +Y -> -Z, i.e.
	Rx(-90): exactly the 'rotate_x_deg = -90' every AR workflow here types by
	hand. A +Z-up export needs +Z -> -Z, Rx(180) (the choice of which
	horizontal axis that flips is arbitrary; rotate_z_deg settles the facing).
	Both are X rotations, so conjugating them into OpenCV space is a no-op -
	the caller conjugates anyway, so a future +X option stays correct."""
	name = str(model_up)
	if name.startswith("+Y"):
		return cv2.Rodrigues(np.array([-np.pi / 2.0, 0.0, 0.0]))[0]
	if name.startswith("+Z"):
		return cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]
	return np.eye(3)


def _to_three_space(m):
	"""4x4 between OpenCV world and Three.js world. The two differ by
	F = diag(1, -1, -1), so a TRANSFORM conjugates: M_three = F.M_cv.F (F is
	its own inverse, hence one helper serves both directions)."""
	return _THREE_FLIP4 @ np.asarray(m, np.float64) @ _THREE_FLIP4


def _decompose(m):
	"""4x4 -> (position, quaternion xyzw, scale xyz). Per-COLUMN norms, so a
	non-uniform gizmo scale survives the round trip."""
	m = np.asarray(m, np.float64)
	scale = np.linalg.norm(m[:3, :3], axis=0)
	safe = np.where(scale > 1e-12, scale, 1.0)
	q = _rotation_to_quaternion(m[:3, :3] / safe[None, :])
	return m[:3, 3].copy(), np.array(q, np.float64), scale


def _compose_srt(position, rotate_deg, scale, pre=None):
	"""4x4 = T . Rz . Ry . Rx . S about the ORIGIN - the same order and axis
	convention as the 'CV Transform Points (Scale, Rotate, Translate)'
	blueprint, so the two agree to float precision.

	`pre` is an extra 3x3 applied BEFORE all three, in the same space: the
	up-axis fix. Keeping it innermost is what lets rotate_* stay a delta the
	user nudges on top of a model that is already standing."""
	rad = np.asarray(rotate_deg, np.float64) * (np.pi / 180.0)
	rot = np.eye(3)
	for axis, angle in zip(np.eye(3), rad):
		rot = cv2.Rodrigues(axis * angle)[0] @ rot
	if pre is not None:
		rot = rot @ np.asarray(pre, np.float64)
	m = np.eye(4)
	m[:3, :3] = rot * float(scale)
	m[:3, 3] = np.asarray(position, np.float64)
	return m


def _transform_json(position, quaternion, scale, decimals=9):
	"""The object_transform widget's dict. Rounded rather than repr'd - this
	string is read by a HUMAN in a widget as often as by the renderer."""
	def r(v):
		return [round(float(x), decimals) + 0.0 for x in np.asarray(v).reshape(-1)]
	return json.dumps({"position": r(position), "quaternion": r(quaternion),
					   "scale": r(scale)})


class ObjectTransform3D(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ObjectTransform3D",
			display_name="CV 3D Object Transform",
			category=CATEGORY,
			description="Places a 3D model, from ONE set of numbers, in BOTH "
						"lanes that can carry a placement: the 'object_transform' "
						"string 'CV Preview 3D (Calibrated Camera)' renders with, "
						"and the 4x4 'CV Transform Points 3D' applies to the mesh "
						"'CV Mesh From 3D Model' pulled out of the same file. "
						"Those two live in DIFFERENT coordinate systems - the "
						"viewer is Three.js (Y up), OpenCV is Y down - so driving "
						"them from two hand-typed values is how a render and a "
						"tracker end up disagreeing about where the object is. Set "
						"'space' to whatever 'CV Mesh From 3D Model'."
						"axis_convention is set to, type the numbers in that "
						"space, and the conversion is done for you. 'model_up' "
						"then stands the model on the target plane, so the "
						"'rotate_x_deg = -90' the AR workflows type by hand is a "
						"dropdown. 'CV Parse Object Transform' is the return leg, "
						"for a placement dragged out with the viewer's gizmo.",
			inputs=[
				io.Combo.Input("space", options=_OBJECT_SPACES,
					default=_OBJECT_SPACES[0],
					tooltip="Which convention the numbers below are written in, "
							"and the space 'matrix' comes back in - match it to "
							"'CV Mesh From 3D Model'.axis_convention. The two are "
							"the same scene under a 180 degree roll about X (both "
							"right-handed), so this says how to READ the numbers "
							"below rather than what to do to the model: type y/z "
							"(and y/z rotations) with the opposite sign in the other "
							"space and the result is identical. Flipping this ALONE "
							"therefore does move the model - it is not a free switch, "
							"it is a promise about what your numbers mean. To turn "
							"the model on purpose, use 'model_up' and rotate_*. The 'object_transform' string is always "
							"Three.js whatever this says, because that is what "
							"the viewer reads."),
				io.Combo.Input("model_up", options=_MODEL_UPS,
					default=_MODEL_UPS[0],
					tooltip="Which axis points UP inside the model FILE - a "
							"one-off turn that stands it on the calibration "
							"target, applied before your rotate_* values so those "
							"stay a free nudge on top. Note that UP here is the "
							"target's own normal (OpenCV +Z, since the markers "
							"lie in its XY plane), NOT +/-Y: a stock +Y-up GLB "
							"arrives lying flat across the board. Most GLB is +Y "
							"up (the manual equivalent is rotate_x_deg = -90); "
							"Blender/CAD/OBJ geometry is usually +Z up. Leave it "
							"alone for a model already authored to stand. Not the "
							"same job as 'space': that only renames a placement, "
							"this moves the model."),
				io.Float.Input("position_x", default=0.0, min=-1e6, max=1e6,
					step=0.01,
					tooltip="Translation along X, in the SAME units as the mesh "
							"and the camera pose (metres, if the calibration is). "
							"Applied last, after the rotation."),
				io.Float.Input("position_y", default=0.0, min=-1e6, max=1e6,
					step=0.01,
					tooltip="Translation along Y. This axis flips with 'space' - "
							"+Y is UP in Three.js, DOWN in OpenCV - so the same "
							"number moves the model opposite ways."),
				io.Float.Input("position_z", default=0.0, min=-1e6, max=1e6,
					step=0.01,
					tooltip="Translation along Z: OFF the target plane, since the "
							"board normal is the up direction here. +Z runs into "
							"the scene in OpenCV, toward the viewer in Three.js."),
				io.Float.Input("rotate_x_deg", default=0.0, min=-360.0, max=360.0,
					step=1.0,
					tooltip="Rotation about X, in degrees, about the mesh ORIGIN "
							"(use 'CV Mesh From 3D Model'.recenter to spin about "
							"the object's own centre). Applied FIRST of the "
							"three, and after 'model_up' - tilt, once the model "
							"already stands."),
				io.Float.Input("rotate_y_deg", default=0.0, min=-360.0, max=360.0,
					step=1.0,
					tooltip="Rotation about Y, in degrees. Applied second - the "
							"full order is Rz . Ry . Rx, matching the 'CV "
							"Transform Points (Scale, Rotate, Translate)' "
							"blueprint."),
				io.Float.Input("rotate_z_deg", default=0.0, min=-360.0, max=360.0,
					step=1.0,
					tooltip="Rotation about Z, in degrees. Applied LAST of the "
							"three - with 'model_up' set, this is the one that "
							"aims the standing model (its yaw about the board "
							"normal)."),
				io.Float.Input("scale", default=1.0, min=1e-6, max=1e6, step=0.01,
					tooltip="Uniform scale, applied before rotation and "
							"translation. Reconciles authoring units with the "
							"calibration (centimetres against a pose in metres "
							"needs 0.01) - a scale error reads as an object at "
							"the right bearing and the wrong depth."),
			],
			outputs=[
				io.String.Output(display_name="object_transform",
					tooltip="JSON for the 'object_transform' input of 'CV Preview "
							"3D (Calibrated Camera)': position, quaternion "
							"[x, y, z, w] and scale in Three.js world space. "
							"Convert that widget to an input and wire this into "
							"it; the viewer's gizmo then only previews, because "
							"this value wins on the next run."),
				NPArray.Output(display_name="matrix",
					tooltip="The same placement as a 4x4 float64 model matrix, in "
							"the selected 'space'. Feed it to 'CV Transform "
							"Points 3D' to move the mesh (or annotation anchors, "
							"or a point cloud) exactly the way the render moved "
							"the model."),
			],
		)

	@classmethod
	def execute(cls, space, model_up, position_x, position_y, position_z,
				rotate_x_deg, rotate_y_deg, rotate_z_deg, scale) -> io.NodeOutput:
		pre = _model_up_rotation(model_up)
		if not str(space).startswith("glTF"):
			# the fix is named in the FILE's axes; express it in the space the
			# rest of the numbers live in (conjugate by the Rx(180) between them)
			flip = _THREE_FLIP4[:3, :3]
			pre = flip @ pre @ flip
		matrix = _compose_srt((position_x, position_y, position_z),
							  (rotate_x_deg, rotate_y_deg, rotate_z_deg), scale,
							  pre)
		three = matrix if str(space).startswith("glTF") else _to_three_space(matrix)
		pos, quat, scl = _decompose(three)
		return io.NodeOutput(_transform_json(pos, quat, scl), matrix)


class ParseObjectTransform(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ParseObjectTransform",
			display_name="CV Parse Object Transform",
			category=CATEGORY,
			description="Reads an 'object_transform' JSON string back into the "
						"graph as a 4x4 - the return leg of 'CV 3D Object "
						"Transform'. The viewer's move/rotate/scale GIZMO writes "
						"that widget as you drag, so this is how a placement found "
						"BY HAND becomes data: wire (or paste) the string here and "
						"the mesh, the annotation anchors and the silhouette all "
						"follow the model you parked. Never raises - an unreadable "
						"string reports valid=false and returns identity, so a "
						"half-typed widget cannot halt a run. What comes back is "
						"the whole placement as one rotation: a string written "
						"by 'CV 3D Object Transform' with 'model_up' set parses "
						"to the COMBINED turn, not to that dropdown plus the "
						"rotate_* values separately.",
			inputs=[
				io.String.Input("object_transform", default="", multiline=True,
					tooltip="The JSON from the viewer's object_transform widget: "
							"position, quaternion [x, y, z, w] and scale, in "
							"Three.js world space. Scale may be a single number "
							"and missing keys default to identity, exactly as the "
							"renderer reads it. Empty is identity and counts as "
							"VALID - that is what an untouched widget holds."),
				io.Combo.Input("space", options=_OBJECT_SPACES,
					default=_OBJECT_SPACES[0],
					tooltip="Coordinate system for ALL the outputs. Match it to "
							"'CV Mesh From 3D Model'.axis_convention so the "
							"matrix applies to that mesh directly; the input "
							"string itself is always Three.js."),
			],
			outputs=[
				NPArray.Output(display_name="matrix",
					tooltip="4x4 float64 model matrix in the selected space, "
							"ready for 'CV Transform Points 3D'."),
				NPArray.Output(display_name="position",
					tooltip="1-D [x, y, z] translation, in the selected space."),
				NPArray.Output(display_name="quaternion",
					tooltip="1-D unit quaternion [x, y, z, w] of the rotation, in "
							"the selected space (conjugated by the axis flip when "
							"that is OpenCV, so it recomposes 'matrix' together "
							"with 'position' and 'scale')."),
				NPArray.Output(display_name="scale",
					tooltip="1-D per-axis [x, y, z] scale. The gizmo can scale "
							"non-uniformly, so this is a vector even though 'CV "
							"3D Object Transform' only emits uniform ones."),
				io.Boolean.Output(display_name="valid",
					tooltip="False when the string could not be read as an "
							"object_transform - the other outputs are then "
							"identity. Wire it to an 'If/Else Switch' to fall "
							"back to a preset placement instead of rendering the "
							"model at the origin."),
			],
		)

	@classmethod
	def execute(cls, object_transform, space) -> io.NodeOutput:
		raw = str(object_transform).strip()
		valid, spec = True, {}
		if raw:
			try:
				spec = json.loads(raw)
				if not isinstance(spec, dict):
					raise TypeError("not a JSON object")
			except (json.JSONDecodeError, TypeError, ValueError):
				valid, spec = False, {}
		try:
			three = render3d.compose_transform(
				spec.get("position"), spec.get("quaternion"), spec.get("scale"))
		except (TypeError, ValueError, IndexError):
			valid, three = False, np.eye(4)
		if not np.all(np.isfinite(three)):
			valid, three = False, np.eye(4)
		matrix = three if str(space).startswith("glTF") else _to_three_space(three)
		pos, quat, scl = _decompose(matrix)
		return io.NodeOutput(matrix, pos, quat, scl, valid)


class CVArrayShape(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CVArrayShape",
			display_name="CV Array Shape",
			category=CATEGORY,
			description="Reads the spatial dimensions and channel count of an "
			            "IMAGE/MASK/LATENT/NPARRAY. For a [H,W] array or MASK "
			            "channels=1; for [H,W,C] arrays channels=C. LATENT "
			            "sizes count latent cells (pixels / VAE stride). Only "
			            "the shape is inspected - no data is read or modified.",
			inputs=[polytype.poly_input("nparray",
				types=polytype.with_latent(polytype.IMAGEISH),
				tooltip="Array whose shape to read. Only the shape is "
				        "inspected regardless of dtype. LATENT sizes count "
				        "latent cells.")],
			outputs=[
				io.Int.Output(display_name="height",
					tooltip="Height (first spatial dimension). For LATENT "
					        "inputs this counts latent cells (pixels / stride)."),
				io.Int.Output(display_name="width",
					tooltip="Width (second spatial dimension). For LATENT "
					        "inputs this counts latent cells."),
				io.Int.Output(display_name="channels",
					tooltip="Number of channels: 1 for 2-D arrays / MASK, "
					        "C for [H,W,C] arrays / IMAGE / LATENT."),
				io.String.Output(display_name="dsize",
					tooltip="'(width, height)' literal, for the composite cv2 "
					        "params that are still free-text. Prefer the 'size' "
					        "output for a Size-typed input."),
				tuples.CvTuple.Output(display_name="size",
					tooltip="(width, height) as ONE composite value - wire it "
					        "straight into a cv2 node's dsize / ksize / winSize "
					        "/ imageSize input. Both components travel together, "
					        "so the size can never arrive half-connected."),
			],
		)

	@classmethod
	def execute(cls, nparray) -> io.NodeOutput:
		arr, _ = polytype.resolve(nparray)
		arr = np.asarray(arr)
		if arr.ndim < 2:
			raise ValueError(
				f"Array of shape {arr.shape} has no image dimensions. "
				"Expected at least 2-D [H,W] or 3-D [H,W,C]."
			)
		h, w = arr.shape[:2]
		channels = 1 if arr.ndim == 2 else arr.shape[2]
		return io.NodeOutput(h, w, channels, f"({w}, {h})", [int(w), int(h)])


class CVDType(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_CVDType",
            display_name="CV DType",
            category=CATEGORY,
            description="Returns the dtype name of an ndarray as a string, "
                        "suitable for connecting to CV Cast Array's dtype input.",
            inputs=[
                NPArray.Input("nparray",
                    tooltip="Any ndarray whose dtype to inspect."),
            ],
            outputs=[io.String.Output(display_name="dtype",
                tooltip="The numpy dtype name (e.g. 'float32', 'uint8').")],
        )

    @classmethod
    def execute(cls, nparray) -> io.NodeOutput:
        arr = nparray if isinstance(nparray, np.ndarray) else np.asarray(nparray)
        return io.NodeOutput(arr.dtype.name)


class ColorFromHex(io.ComfyNode):
    _codes_cache = None

    @classmethod
    def _available_codes(cls):
        if cls._codes_cache is None:
            valid = [n for n in dir(cv2) if n.startswith("COLOR_RGB2")
                     and isinstance(getattr(cv2, n), int)]
            valid.sort()
            cls._codes_cache = valid + ["none (RGB)"]
        return cls._codes_cache

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_ColorFromHex",
            display_name="CV Color From Hex",
            category=CATEGORY,
            description="Converts a ComfyUI native COLOR hex string (#RRGGBB) to a tuple string "
                        "literal for OpenCV draw nodes (e.g. '(0, 0, 255)' for blue in BGR). "
                        "The hex string is parsed as sRGB. Select a cvtColor code to convert to "
                        "the desired color space, or 'none (RGB)' to keep the RGB values. "
                        "'scale_to_range' normalises each channel to [0, 1] using the target "
                        "format's per-channel maximum (OpenCV's uint8 ranges: H=179 in HSV/HLS, "
                        "H=255 in HLS_FULL, all other channels 255).",
            inputs=[
                io.Color.Input("color", default="#00ff00",
                               tooltip="Input color in #RRGGBB hex format - renders as a native "
                                        "color picker in the UI."),
                io.Combo.Input("code", options=cls._available_codes(),
                               default="COLOR_RGB2BGR",
                               tooltip="cv2 cvtColor conversion code. 'none (RGB)' passes the "
                                        "sRGB value through unchanged."),
                io.Boolean.Input("scale_to_range", default=False,
                              tooltip="When enabled, each channel is divided by its uint8 range "
                                       "max (179 for H in HSV/HLS, 255 else) so the tuple lands "
                                       "in [0, 1]. Disabled = raw uint8 values."),
            ],
            outputs=[
                io.String.Output(display_name="tuple_string",
                                 tooltip="Tuple literal, e.g. '(0, 0, 255)' - feed into draw "
                                          "nodes' colour input."),
            ],
        )

    @staticmethod
    def _range_maxes(code: str, n_channels: int):
        """Return per-channel max uint8 values for scale_to_range."""
        if "HSV" in code or (code.endswith("_HLS") and "_FULL" not in code):
            return (179.0,) + (255.0,) * (n_channels - 1) if n_channels > 1 else (255.0,)
        return tuple([255.0] * n_channels)

    @classmethod
    def execute(cls, color, code, scale_to_range) -> io.NodeOutput:
        h = color.lstrip("#")
        if len(h) != 6:
            h = "00ff00"  # fallback green
        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        except ValueError:
            r, g, b = 0, 255, 0

        if code == "none (RGB)":
            values = (r, g, b)
        else:
            pixel = np.array([[[r, g, b]]], dtype=np.uint8)
            code_int = int(getattr(cv2, code))
            cvt = cv2.cvtColor(pixel, code_int)
            values = tuple(int(x) for x in cvt.ravel())

        if scale_to_range:
            maxes = cls._range_maxes(code, len(values))
            scaled = tuple(v / m for v, m in zip(values, maxes))
        else:
            # strip trailing .0 from whole floats
            scaled = tuple(int(v) if abs(v - round(v)) < 1e-9 else v for v in values)

        if len(values) == 1:
            out = f"({scaled[0]},)"
        else:
            out = "(" + ", ".join(str(v) for v in scaled) + ")"

        return io.NodeOutput(out)


class DepthTo3D(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DepthTo3D",
			display_name="CV Depth to 3D Points",
			category=CATEGORY,
			description="Back-projects a depth map to a colored 3-D point cloud "
			            "using camera intrinsics. For each pixel (u, v) with depth "
			            "Z, computes X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy. "
			            "Unlike cv2.reprojectImageTo3D (which is for stereo "
			            "disparity), this handles monocular depth directly. "
			            "Feed the depth as a float32 NPARRAY (in any unit — "
			            "the depth_scale converts to your desired output unit). "
			            "Output is Nx3 points + Nx3 colors for 'Write PLY'.",
			inputs=[
				NPArray.Input("depth",
					tooltip="HxW float32 depth map. Values can be in any unit "
					        "(mm, meters, raw model output) — use depth_scale "
					        "to convert. Single-channel; if loaded via "
					        "Image->CV (GRAY float32), the channel is already "
					        "extracted."),
				NPArray.Input("camera_matrix",
					tooltip="3x3 camera intrinsic matrix K from "
					        "Calibrate Camera (Chessboard) or CV Camera "
					        "Matrix. fx=K[0,0], fy=K[1,1], cx=K[0,2], "
					        "cy=K[1,2]."),
				NPArray.Input("colors", optional=True,
					tooltip="HxWx3 or HxW uint8/float RGB color image "
					        "(same dimensions as depth). If disconnected, "
					        "all points are white."),
			io.Float.Input("depth_scale", default=1.0, min=0.0, step=0.001,
				tooltip="Multiplier to convert depth values to meters. "
				        "E.g. 0.001 if depth is in mm, 1.0 if already "
				        "in meters, 65535/1000=65.535 for NYU Depth V2 "
				        "(uint16 mm loaded as float32 0-1)."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="Nx3 float32 array of 3-D positions."),
				NPArray.Output(display_name="colors",
					tooltip="Nx3 uint8 array of per-vertex RGB colors."),
			],
		)

	@classmethod
	def execute(cls, depth, camera_matrix, colors=None, depth_scale=1.0) -> io.NodeOutput:
		d = np.asarray(depth, np.float32)
		if d.ndim == 3:
			d = d[:, :, 0]
		h, w = d.shape

		K = np.asarray(camera_matrix, np.float64)
		fx, fy = float(K[0, 0]), float(K[1, 1])
		cx, cy = float(K[0, 2]), float(K[1, 2])

		Z = d * float(depth_scale)

		u, v = np.meshgrid(
			np.arange(w, dtype=np.float32),
			np.arange(h, dtype=np.float32),
		)
		# a rasterized depth map is +inf where nothing was drawn, and at the
		# principal point (u - cx) is exactly 0, so the product is 0 * inf =
		# nan. Those rows are dropped by the finite mask below; the warning
		# they raise on the way there is noise, not a signal.
		with np.errstate(invalid="ignore"):
			X = (u - cx) * Z / fx
			Y = -(v - cy) * Z / fy

		pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)

		finite_mask = np.isfinite(pts).all(axis=1)
		n_invalid = int((~finite_mask).sum())
		if n_invalid:
			pct = 100.0 * n_invalid / pts.shape[0]
			logging.info("DepthTo3D: total=%d invalid=%d (%.1f%%)",
			             pts.shape[0], n_invalid, pct)
			pts = pts[finite_mask]

		if colors is not None:
			c = np.asarray(colors)
			if c.ndim == 3:
				if c.shape[2] == 4:
					c = c[:, :, :3]
				if c.dtype in (np.float32, np.float64):
					c = np.clip(c * 255.0, 0, 255).astype(np.uint8)
				else:
					c = c.astype(np.uint8)
			elif c.ndim == 2:
				c = np.stack([c, c, c], axis=-1).astype(np.uint8)
			c = c.reshape(-1, 3)
			if n_invalid:
				c = c[finite_mask]
		else:
			c = np.full((pts.shape[0], 3), 255, np.uint8)

		return io.NodeOutput(pts, c)


class WritePly(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_WritePly",
			display_name="CV Write PLY (Point Cloud)",
			category=CATEGORY,
			description="Converts numpy point arrays to a binary PLY file for "
			            "ComfyUI's 'Preview Point Cloud' node. Feed an Nx3 float "
			            "array of 3-D positions and optionally an Nx3 array of RGB "
			            "colors (0-255 uint8 or 0-1 float). The PLY is written to "
			            "a temp file and wrapped as a File3D object that plugs "
			            "straight into PreviewPointCloud or SaveGLB.",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx3 or Nx1x3 float array of 3-D vertex positions "
					        "(e.g. from cv2.reprojectImageTo3D, Triangulate "
					        "Points, or the depth-to-points subgraph)."),
				NPArray.Input("colors", optional=True,
					tooltip="Nx3 or Nx1x3 array of per-vertex RGB colors. "
					        "uint8 0-255 or float 0-1 (auto-scaled). If "
					        "disconnected, vertices are white."),
			],
			outputs=[io.File3DPLY.Output(display_name="ply",
				tooltip="Binary PLY point cloud. Feed into 'Preview Point Cloud' "
				        "or 'Save 3D Model'.")],
		)

	@classmethod
	def execute(cls, points, colors=None) -> io.NodeOutput:
		from io import BytesIO

		pts = np.asarray(points, np.float32).reshape(-1, 3)
		n_raw = pts.shape[0]
		if n_raw == 0:
			raise ValueError("WritePly: points array is empty.")

		# Filter out Inf/NaN vertices (e.g. from reprojectImageTo3D on
		# invalid disparity).  PLY viewers choke on non-finite bounds.
		finite_mask = np.isfinite(pts).all(axis=1)
		n_invalid = int((~finite_mask).sum())
		if n_invalid:
			pct = 100.0 * n_invalid / pts.shape[0]
			logging.info("WritePly: total=%d invalid=%d (%.1f%%)",
			             pts.shape[0], n_invalid, pct)
		pts = pts[finite_mask]
		n = pts.shape[0]
		if n == 0:
			raise ValueError("WritePly: all points are non-finite.")

		# Build colors
		if colors is not None:
			col = np.asarray(colors)
			if col.dtype in (np.float32, np.float64):
				col = np.clip(col * 255.0, 0, 255).astype(np.uint8)
			else:
				col = np.clip(col, 0, 255).astype(np.uint8)
			col = col.reshape(-1, 3)
			if n_invalid and col.shape[0] == n_raw:
				col = col[finite_mask]
			if col.shape[0] != n:
				if col.shape[0] == 1:
					col = np.broadcast_to(col, (n, 3)).copy()
				else:
					raise ValueError(
						f"WritePly: colors has {col.shape[0]} rows but points "
						f"has {n} - must match or be a single color.")
			has_color = True
		else:
			col = np.full((n, 3), 255, np.uint8)
			has_color = False

		# Build binary PLY
		buf = BytesIO()
		# Header
		header_lines = [
			"ply",
			"format binary_little_endian 1.0",
			f"element vertex {n}",
			"property float x",
			"property float y",
			"property float z",
			"property uchar red",
			"property uchar green",
			"property uchar blue",
			"end_header",
		]
		buf.write(("\n".join(header_lines) + "\n").encode("ascii"))
		# Body: x,y,z as float32 + r,g,b as uint8 per vertex
		dtype = np.dtype([
			("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
			("red", "u1"), ("green", "u1"), ("blue", "u1"),
		])
		vertices = np.empty(n, dtype=dtype)
		vertices["x"] = pts[:, 0]
		vertices["y"] = pts[:, 1]
		vertices["z"] = pts[:, 2]
		vertices["red"] = col[:, 0]
		vertices["green"] = col[:, 1]
		vertices["blue"] = col[:, 2]
		buf.write(vertices.tobytes())

		data = buf.getvalue()
		from comfy_api.latest import Types
		return io.NodeOutput(Types.File3D(BytesIO(data), file_format="ply"))


class FilterPointCloud(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FilterPointCloud",
			display_name="CV Filter Point Cloud",
			category=CATEGORY,
			description="Removes outliers from an Nx3 point cloud by distance from "
			            "the origin.  Useful for pruning extreme-depth points from "
			            "stereo reprojection or depth estimation that would blow up "
			            "the bounding box and hide the scene.",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx3 or Nx1x3 float array of 3-D positions."),
				NPArray.Input("colors", optional=True,
					tooltip="Nx3 per-vertex colors to filter in sync.  If "
					        "disconnected, all surviving points are white."),
				io.Float.Input("min_distance", default=0.0, min=0.0, step=0.1,
					tooltip="Minimum Euclidean distance from the origin.  Points "
					        "closer than this are removed (0 = no lower bound)."),
				io.Float.Input("max_distance", default=100.0, min=0.0, step=0.1,
					tooltip="Maximum Euclidean distance from the origin.  Points "
					        "farther than this are removed (0 = no upper bound)."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="Filtered Nx3 points."),
				NPArray.Output(display_name="colors",
					tooltip="Filtered Nx3 colors (same order as points)."),
			],
		)

	@classmethod
	def execute(cls, points, colors=None, min_distance=0.0, max_distance=100.0) -> io.NodeOutput:
		pts = np.asarray(points, np.float32).reshape(-1, 3)
		n_raw = pts.shape[0]

		dist = np.linalg.norm(pts, axis=1)
		mask = np.ones(n_raw, dtype=bool)
		if min_distance > 0:
			mask &= dist >= min_distance
		if max_distance > 0:
			mask &= dist <= max_distance

		n_filtered = int((~mask).sum())
		if n_filtered:
			pct = 100.0 * n_filtered / n_raw
			logging.info("FilterPointCloud: total=%d filtered=%d (%.1f%%) "
			             "[%.1f .. %.1f]",
			             n_raw, n_filtered, pct, min_distance, max_distance)

		pts = pts[mask]
		col_out = None
		if colors is not None:
			col = np.asarray(colors).reshape(-1, 3)
			col_out = col[mask]

		return io.NodeOutput(pts, col_out)


class FlipAxis(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FlipAxis",
			display_name="CV Flip Axis (3D)",
			category=CATEGORY,
			description="Negates one axis of an Nx3 point array. Useful for "
			            "converting between coordinate conventions "
			            "(e.g. OpenCV Y-down -> Three.js Y-up).",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx3 or Nx1x3 float array of 3-D positions."),
				io.Combo.Input("axis", options=["Y", "X", "Z"],
					tooltip="Which axis to negate. Y-up (Three.js) vs "
					        "Y-down (OpenCV) is the most common need."),
			],
			outputs=[NPArray.Output(display_name="points",
				tooltip="Nx3 points with the selected axis negated.")],
		)

	@classmethod
	def execute(cls, points, axis="Y") -> io.NodeOutput:
		pts = np.asarray(points, np.float32).reshape(-1, 3).copy()
		col = {"X": 0, "Y": 1, "Z": 2}[str(axis).strip().upper()]
		pts[:, col] *= -1.0
		return io.NodeOutput(pts)


class ConvertAxisConvention(io.ComfyNode):
	# Every OTHER thing that crosses between the two conventions already has a
	# node that names them: a mesh at load ('CV Mesh From 3D Model'), its
	# annotation anchors ('CV Annotate Points On Model'), a placement ('CV 3D
	# Object Transform' / 'CV Parse Object Transform') and a camera ('CV Camera
	# Pose To 3D View'). A CLOUD already in the graph had nothing, so it was
	# spelled out as a pair of 'CV Flip Axis (3D)' nodes - and a reader then has
	# to RECOGNIZE a convention instead of reading its name. Workflow 81 shipped
	# for a day with only one of the two flips: a reflection, which every check
	# passes and which the viewer draws happily, showing the far side of the
	# truck. This node exists so that mistake cannot be spelled.
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ConvertAxisConvention",
			display_name="CV Convert Axis Convention (3D)",
			category=CATEGORY,
			description="Moves a point cloud between the two coordinate systems "
			            "this pack lives in: OpenCV (Y down, Z into the scene - "
			            "what every solver, pose and depth map here uses) and "
			            "glTF / Three.js (Y up, Z toward the viewer - what 'Load "
			            "3D', 'Preview Point Cloud' and the viewer gizmo use). "
			            "The conversion negates Y AND Z, which is a 180-degree "
			            "turn about X; negating just ONE of them is a REFLECTION, "
			            "and a mirrored cloud is the failure nothing catches - "
			            "right size, right place, aligned with its own scene, "
			            "object facing the wrong way. Use this in front of 'CV "
			            "Write PLY' and behind 'CV Depth to 3D Points' rather "
			            "than counting axis flips by hand. An Nx6 cloud keeps its "
			            "normals, turned with the points. Failure tolerant: an "
			            "empty cloud comes back empty.",
			inputs=[
				NPArray.Input("points",
					tooltip="Nx3 (or Nx1x3 / HxWx3) point cloud, or the Nx6 "
					        "(x,y,z,nx,ny,nz) form the surface-matching nodes "
					        "pass around - the normals are directions, so they "
					        "get the same flip as the positions."),
				io.Combo.Input("from_space", options=_OBJECT_SPACES,
					default=_OBJECT_SPACES[0],
					tooltip="Which convention the incoming cloud is written in. "
					        "A cloud from 'CV Depth to 3D Points', 'CV Write PLY' "
					        "or a 3D loader is in the glTF/Three.js one; anything "
					        "that has been through a pose, a camera matrix or a "
					        "solver is in the OpenCV one."),
				io.Combo.Input("to_space", options=_OBJECT_SPACES,
					default=_OBJECT_SPACES[1],
					tooltip="Which convention to leave it in. The flip is its own "
					        "inverse, so only whether these two DIFFER changes "
					        "the numbers - setting both is what makes the graph "
					        "say which way it is going instead of leaving the "
					        "reader to count minus signs."),
			],
			outputs=[NPArray.Output(display_name="points",
				tooltip="The cloud in 'to_space', float32, same shape as the "
				        "input (Nx3 or Nx6). Unchanged when the two spaces are "
				        "the same.")],
		)

	@classmethod
	def execute(cls, points, from_space, to_space) -> io.NodeOutput:
		arr = np.asarray(points) if points is not None else None
		if arr is None or arr.size == 0:
			return io.NodeOutput(np.zeros((0, 3), np.float32))
		arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(-1, 3)
		out = np.array(arr, np.float32)
		if str(from_space) != str(to_space):
			sign = np.diag(_THREE_FLIP4)[:3].astype(np.float32)  # (1, -1, -1)
			out[:, :3] *= sign
			if out.shape[1] >= 6:  # normals are directions: same turn
				out[:, 3:6] *= sign
		return io.NodeOutput(np.ascontiguousarray(out))


class SliceArray(io.ComfyNode):
	# The generic slice the pack kept needing and working around: a column out of
	# an (N,4) bbox table (the Draw Labels chain does it with Permute -> Take By
	# Index -> Permute), a score column, and - the one that finally paid for it -
	# cropping a DNN result back to the source size after padding the input up to
	# the multiple of 32 the network demands. numpy semantics, one axis per node.
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SliceArray",
			display_name="CV Slice Array",
			category=CATEGORY,
			description="Takes a range out of ONE axis of an array (numpy "
			            "slicing). Crop a padded result back to its original "
			            "size (axis 0 = rows, axis 1 = columns - one node each), "
			            "pull a single column out of a detection table (start=2, "
			            "stop=3 keeps the axis; see 'keep_axis'), subsample every "
			            "n-th row with step, or reverse an axis with step=-1. "
			            "Negative start/stop count from the end. Chain one node "
			            "per axis. Out-of-range bounds CLAMP the numpy way rather "
			            "than raising - an empty result is a valid outcome, not a "
			            "wiring mistake.",
			inputs=[
				NPArray.Input("nparray",
					tooltip="Array to slice. Any shape and dtype; values are not "
					        "touched, only selected."),
				io.Int.Input("axis", default=0, min=-8, max=8,
					tooltip="Which axis to slice. 0 = rows (height of an image, "
					        "detections of a table), 1 = columns (width), -1 = "
					        "the last axis (channels). Negative counts from the "
					        "end."),
				io.Int.Input("start", default=0, min=-INT32_MAX, max=INT32_MAX,
					tooltip="First index to keep, counting from 0. Negative "
					        "counts from the end (-2 = the second-to-last)."),
				io.Int.Input("stop", default=0, min=-INT32_MAX, max=INT32_MAX,
					tooltip="One PAST the last index to keep. 0 means 'to the "
					        "end of the axis' - the common case when cropping a "
					        "padded array back to a computed size, where the "
					        "size arrives as a link. Negative counts from the "
					        "end (-1 = drop the last element)."),
				io.Int.Input("step", default=1, min=-1024, max=1024,
					optional=True, advanced=True,
					tooltip="Stride between kept indices. 2 = every other one. A "
					        "NEGATIVE step walks backwards, and then 'start' is "
					        "where it starts FROM: reversing a whole axis is "
					        "start=-1, stop=0, step=-1 - with the default start=0 "
					        "a backwards walk stops immediately and returns a "
					        "single element. 0 is rejected (numpy does not "
					        "define it)."),
				io.Boolean.Input("keep_axis", default=True,
					optional=True, advanced=True,
					tooltip="When the slice keeps exactly ONE index, whether the "
					        "axis survives with length 1 (True, e.g. (N,4) -> "
					        "(N,1)) or is dropped (False, (N,4) -> (N,)). Off is "
					        "what a downstream node wanting a flat vector of "
					        "scores expects; on is what keeps a table a table."),
			],
			outputs=[
				NPArray.Output(display_name="nparray",
					tooltip="The selected part of the input, same dtype. A view "
					        "is never returned - the result is contiguous, so a "
					        "later cv2 node cannot write into the original."),
			],
		)

	@classmethod
	def execute(cls, nparray, axis, start, stop, step=1,
	            keep_axis=True) -> io.NodeOutput:
		arr = np.asarray(nparray)
		ax, st = int(axis), int(step)
		if st == 0:
			raise ValueError("step must not be 0 - use 1 to keep every element.")
		if arr.ndim == 0:
			raise ValueError("Cannot slice a 0-D array (a scalar has no axes).")
		if not -arr.ndim <= ax < arr.ndim:
			raise ValueError(
				f"axis {ax} is out of range for a {arr.ndim}-D array of shape "
				f"{arr.shape} - valid axes are {-arr.ndim}..{arr.ndim - 1}.")
		# stop == 0 means "to the end": a computed length arrives as a link and
		# 0 is the only value that cannot mean a real slice (it would be empty)
		hi = None if int(stop) == 0 else int(stop)
		sel = [slice(None)] * arr.ndim
		sel[ax] = slice(int(start), hi, st)
		out = arr[tuple(sel)]
		if not keep_axis and out.shape[ax] == 1:
			out = np.squeeze(out, axis=ax)
		# copy=True, not ascontiguousarray: a slice along the FIRST axis of a
		# C-contiguous array is itself contiguous, so ascontiguousarray hands
		# back the VIEW - which would alias the producer's cached output and let
		# a later in-place cv2 node corrupt it (the reason lowlevel.py copies
		# every ndarray arg). test_slice_array pins this.
		return io.NodeOutput(np.array(out, copy=True, order="C"))


_REDACTION = ["hide directories, keep file names",
              "hide paths entirely"]


class BuildInfo(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_BuildInfo",
			display_name="CV Build Information",
			category=CATEGORY,
			description="Reports how THIS OpenCV was compiled - the answer to "
			            "'does my build have Eigen / non-free / OpenCL / CUDA', "
			            "and the text to paste into a bug report. It replaces "
			            "the raw cv2.getBuildInformation wrapper, which is "
			            "deliberately not exposed: the banner is frozen at "
			            "compile time and quotes the BUILD MACHINE's "
			            "directories, so on a self-built OpenCV it carries your "
			            "home directory, user name and project layout. This "
			            "node redacts every path and reports how many it hid. "
			            "Wire 'info' into core 'Preview as Text' to read it.",
			inputs=[
				io.Combo.Input("redaction", options=_REDACTION,
					tooltip="How much of each path to remove. Keeping the file "
					        "name leaves the banner informative (which compiler, "
					        "which python, which libraries - only WHERE they live "
					        "is gone); hiding paths entirely replaces each one "
					        "with '<path>'. Nothing else in the text is touched."),
			],
			outputs=[
				io.String.Output(display_name="info",
					tooltip="The build banner with paths redacted. Multi-line; "
					        "read it with core 'Preview as Text'."),
				io.Boolean.Output(display_name="has_eigen",
					tooltip="True when the build reports 'Eigen: YES'. The "
					        "Eigen-gated nodes (BIMEF, BIMEF2, "
					        "ximgproc.fastBilateralSolverFilter) raise OpenCV "
					        "error (-213) on every call when this is false - "
					        "gate that branch with 'if/else'."),
				io.Boolean.Output(display_name="has_nonfree",
					tooltip="True when the build reports 'Non-free algorithms: "
					        "YES'. The pack never exposes a non-free function "
					        "(they are blacklisted at generation time), so this "
					        "is informational."),
			],
		)

	@classmethod
	def execute(cls, redaction) -> io.NodeOutput:
		banner = cv2.getBuildInformation()
		# redact_paths also returns how many it hid; that count is deliberately
		# NOT an output. The '<path>' markers are already visible in the text,
		# and nothing downstream could branch on the number - unlike the two
		# flags, which gate an Eigen-only branch through 'if/else'.
		info, _hidden = buildinfo.redact_paths(
			banner, keep_names=(redaction != "hide paths entirely"))
		return io.NodeOutput(info,
		                     buildinfo.banner_flag(banner, "Eigen"),
		                     buildinfo.banner_flag(banner, "Non-free algorithms"))


class MeshFrom3DModel(io.ComfyNode):
	# The pack could already render a GLB ('CV Preview 3D (Calibrated Camera)')
	# and write a PLY, but nothing could read a mesh INTO the graph: every point
	# cloud had to come from stereo. This is the missing source for cv2.rapid,
	# ICP and PPF.
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MeshFrom3DModel",
			display_name="CV Mesh From 3D Model",
			category=CATEGORY,
			description="Reads the GEOMETRY of a 3D model file into plain arrays: "
			            "vertices (Nx3) and triangle indices (Mx3), ready for "
			            "'cv2 rapid rapid', 'CV ICP Register', 'CV PPF Pose "
			            "Estimation' or 'CV Point Cloud Normals'. Every triangle "
			            "primitive of the default scene is merged into ONE indexed "
			            "mesh with the node transforms baked in - the same parser "
			            "'CV Preview 3D (Calibrated Camera)' renders with, so what "
			            "you see and what you measure cannot drift apart. "
			            "Materials, UVs and normals are dropped; positions and "
			            "connectivity only. GLB is the reliable format (a .gltf "
			            "with external buffers will not resolve them). Failure "
			            "tolerant: an unreadable or non-triangle model returns "
			            "empty arrays with loaded=false rather than halting.",
			inputs=[
				io.MultiType.Input("model_3d",
					types=[io.File3DGLB, io.File3DGLTF, io.File3DAny],
					tooltip="Model from a 3D loader node ('Load 3D', 'Load 3D "
					        "(Advanced)'). GLB is the format that actually works: "
					        "a .gltf with external buffers will not resolve them "
					        "from the temp folder."),
				io.Combo.Input("axis_convention",
					options=["OpenCV (Y down, Z into the scene)",
					         "glTF / Three.js (Y up)"],
					default="OpenCV (Y down, Z into the scene)",
					tooltip="glTF stores Y up and Z toward the viewer; OpenCV's "
					        "camera looks down +Z with Y down. Leave this on the "
					        "OpenCV setting whenever the mesh feeds solvePnP, "
					        "cv2.rapid or projectPoints alongside a real camera "
					        "pose - it negates Y and Z, exactly the flip 'CV "
					        "Camera Pose To 3D View' applies in the other "
					        "direction. Pick the glTF setting only to compare "
					        "against the viewer's own coordinates."),
				io.Combo.Input("recenter",
					options=["keep model origin", "bounding-box centre", "centroid"],
					default="keep model origin", optional=True,
					tooltip="Where the returned points put the origin. The model "
					        "origin is what a pose solved against this mesh will "
					        "refer to, so keep it unless you specifically want the "
					        "object's own centre (a pose then means 'centre of "
					        "the object', which is usually what an annotation "
					        "anchor wants). 'centroid' averages the vertices, so "
					        "dense regions pull it."),
				io.Float.Input("scale", default=1.0, min=1e-6, max=1e6,
					step=0.01, optional=True,
					tooltip="Multiplies every coordinate. Use it to put the mesh "
					        "in the SAME units as the camera pose - a model "
					        "authored in centimetres needs 0.01 against a "
					        "calibration in metres, and a scale mismatch shows up "
					        "as a pose that is right in direction and wrong in "
					        "depth."),
			],
			outputs=[
				NPArray.Output(display_name="pts3d",
					tooltip="Nx3 float32 vertices. This is the 'pts3d' input of "
					        "the cv2.rapid nodes and a valid Nx3 point cloud for "
					        "'CV Point Cloud Normals' / ICP / PPF."),
				NPArray.Output(display_name="tris",
					tooltip="Mx3 int32 triangle indices into pts3d - the 'tris' "
					        "input of the cv2.rapid nodes. Winding follows the "
					        "file; cv2.rapid's backface culling depends on it, so "
					        "a model exported with flipped normals silhouettes "
					        "inside out."),
				NPArray.Output(display_name="center",
					tooltip="1x3 bounding-box centre of the RETURNED points (after "
					        "recentre and scale). Wire it to 'CV Transform Points "
					        "3D' or use it to place an annotation anchor."),
				io.Float.Output(display_name="diameter",
					tooltip="Bounding-box diagonal of the returned mesh, in the "
					        "same units. The natural scale reference: an edge "
					        "budget for 'CV Mesh Split Long Edges' is a fraction "
					        "of this, and PPF's sampling steps are fractions of it "
					        "too."),
				io.Int.Output(display_name="vertex_count",
					tooltip="Number of vertices (0 when loaded=false)."),
				io.Int.Output(display_name="triangle_count",
					tooltip="Number of triangles (0 when loaded=false)."),
				io.Boolean.Output(display_name="loaded",
					tooltip="False when the file could not be read or holds no "
					        "triangles. Gate the rest of the graph on it with an "
					        "'If/Else Switch' instead of letting empty arrays "
					        "reach a solver."),
			],
		)

	@classmethod
	def execute(cls, model_3d, axis_convention="OpenCV (Y down, Z into the scene)",
	            recenter="keep model origin", scale=1.0) -> io.NodeOutput:
		import uuid

		import folder_paths
		fail = io.NodeOutput(np.zeros((0, 3), np.float32),
		                     np.zeros((0, 3), np.int32),
		                     np.zeros((1, 3), np.float32), 0.0, 0, 0, False)
		path = model_3d
		if not isinstance(model_3d, str):
			# a File3D from a loader node: materialize it where the parser can
			# reach it, the same way Preview3DCalibrated does
			temp_dir = folder_paths.get_temp_directory()
			os.makedirs(temp_dir, exist_ok=True)
			fmt = getattr(model_3d, "format", None) or "glb"
			path = os.path.join(temp_dir, f"cv_mesh_{uuid.uuid4().hex}.{fmt}")
			model_3d.save_to(path)
		elif path and not os.path.isfile(path):
			resolved = folder_paths.get_annotated_filepath(path)
			path = resolved if os.path.isfile(resolved) else path
		if not path or not os.path.isfile(path):
			logging.warning("MeshFrom3DModel: no readable model file at %r", path)
			return fail
		try:
			pts, tris = render3d.load_mesh(path)
		except (ValueError, KeyError, OSError, StructError) as e:
			logging.warning("MeshFrom3DModel: cannot read %r (%s)", path, e)
			return fail
		if len(pts) == 0 or len(tris) == 0:
			return fail
		pts = pts.astype(np.float64)
		if str(axis_convention).startswith("OpenCV"):
			pts = pts * render3d._FLIP
		lo, hi = pts.min(axis=0), pts.max(axis=0)
		if str(recenter).startswith("bounding"):
			pts = pts - 0.5 * (lo + hi)
		elif str(recenter).startswith("centroid"):
			pts = pts - pts.mean(axis=0)
		pts = pts * float(scale)
		lo, hi = pts.min(axis=0), pts.max(axis=0)
		center = (0.5 * (lo + hi)).astype(np.float32).reshape(1, 3)
		return io.NodeOutput(np.ascontiguousarray(pts, np.float32), tris,
		                     center, float(np.linalg.norm(hi - lo)),
		                     len(pts), len(tris), True)


class MeshSplitLongEdges(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MeshSplitLongEdges",
			display_name="CV Mesh Split Long Edges",
			category=CATEGORY,
			description="Refines a mesh until no triangle has an edge longer than "
			            "max_edge, by repeatedly bisecting the longest edge (one "
			            "triangle becomes two). Only coarse regions grow, so it "
			            "costs about half the triangles a uniform 1->4 subdivision "
			            "needs for the same result. "
			            "This exists for cv2.rapid: extractControlPoints "
			            "interpolates its 3D control points LINEARLY between "
			            "consecutive silhouette VERTICES, using a blend factor "
			            "taken from screen-space distance - so a control point "
			            "between two far-apart vertices is wrong by roughly "
			            "L_px^2/(16*fx) in depth, and its 2D position by the chord "
			            "sagitta L_px^2/(8*R_px) on a curved silhouette, where "
			            "L_px is the ON-SCREEN spacing of silhouette vertices. "
			            "Neither error shrinks if you render at a higher "
			            "resolution - they are geometric, not raster. Measured on "
			            "a 3624-triangle vehicle at fx=900, 8 m away: the "
			            "untouched mesh puts control points up to 23 px off their "
			            "own 3D points and tracking DIVERGES; at L_px<=80 (5960 "
			            "triangles, +14% tracking time) that error is 0.9 px and "
			            "the same sequence tracks to 3 px. "
			            "Rule of thumb: max_edge = 80 * distance / fx. Below "
			            "L_px~20 the raster quantization floor takes over and "
			            "further splitting buys nothing.",
			inputs=[
				NPArray.Input("pts3d",
					tooltip="Nx3 vertices, from 'CV Mesh From 3D Model'."),
				NPArray.Input("tris",
					tooltip="Mx3 int triangle indices. Winding is preserved."),
				io.Float.Input("max_edge", default=0.7, min=0.0, max=1e9,
					step=0.01,
					tooltip="Longest edge allowed, IN THE MESH'S OWN UNITS. "
					        "Convert from the on-screen budget with "
					        "max_edge = L_px * distance / fx (L_px = 80 is the "
					        "knee) - a 'Math Expression' node keeps that visible "
					        "in the graph. 0 disables splitting and passes the "
					        "mesh through unchanged, which is how you A/B the "
					        "effect."),
				io.Int.Input("max_rounds", default=24, min=1, max=64,
					optional=True, advanced=True,
					tooltip="Safety stop. Each round splits every triangle still "
					        "over budget, so the longest edge roughly halves per "
					        "round and 24 rounds is far more than any sane budget "
					        "needs. Lower it to cap the triangle count on a "
					        "pathological model."),
			],
			outputs=[
				NPArray.Output(display_name="pts3d",
					tooltip="Nx3 float32 vertices: the originals unchanged, plus "
					        "one new vertex per split edge. Existing indices keep "
					        "their meaning, so anything indexed against the input "
					        "mesh stays valid."),
				NPArray.Output(display_name="tris",
					tooltip="Mx3 int32 triangles of the refined mesh. Bisection "
					        "leaves T-junctions where a neighbour was not split - "
					        "harmless for silhouette extraction, since the "
					        "neighbour's rasterized edge still passes over the new "
					        "vertex."),
				io.Int.Output(display_name="triangle_count",
					tooltip="Triangles after splitting. Watch it against the input "
					        "count: this is the cost you are paying, and past "
					        "L_px~20 it grows fast for no accuracy."),
			],
		)

	@classmethod
	def execute(cls, pts3d, tris, max_edge, max_rounds=24) -> io.NodeOutput:
		pts, tri = render3d.split_long_edges(pts3d, tris, float(max_edge),
		                                     int(max_rounds))
		return io.NodeOutput(pts, tri, len(tri))


class MeshVertexNormals(io.ComfyNode):
	# The surface-matching family wants Nx6 clouds, and 'CV Point Cloud Normals'
	# is the only way the pack had to make one - but a plane fit through k
	# neighbours cannot tell inside from outside, and its 'flip toward viewpoint'
	# option is wrong for a CLOSED model sampled all round.  Measured on the
	# shipped truck against a single-view scene cloud: plane-fit model normals
	# put the PPF pose 177-179 degrees out at every sampling step tried, while
	# the winding-derived normals below land it 9.2 degrees out - the difference
	# between a pose ICP polishes to 1.5 degrees and a pose that is backwards.
	# A mesh carries the answer in its triangle winding; this node reads it.
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MeshVertexNormals",
			display_name="CV Mesh Vertex Normals",
			category=CATEGORY,
			description="Turns a MESH (vertices + triangles) into the Nx6 "
			            "(x,y,z,nx,ny,nz) cloud 'CV PPF Pose Estimation' and 'CV "
			            "ICP Register' want, with the normals taken from the "
			            "TRIANGLE WINDING - so they point consistently outward "
			            "instead of being sign-arbitrary. Use this, not 'CV Point "
			            "Cloud Normals', whenever the cloud comes from a mesh: a "
			            "plane fit through k neighbours cannot tell inside from "
			            "outside, and on a closed model matched against a "
			            "single-view scene that ambiguity puts the PPF pose ~180 "
			            "degrees out (measured 177-179 on the shipped example, "
			            "against 9.2 with these normals). Each vertex gets the "
			            "AREA-WEIGHTED sum of its incident face normals, so large "
			            "triangles dominate the average as they should. Wire it "
			            "AFTER 'CV Mesh Split Long Edges' - that node adds "
			            "vertices, and normals must follow the final topology. "
			            "Failure tolerant: an empty mesh or an index out of range "
			            "returns empty arrays with found=false.",
			inputs=[
				NPArray.Input("pts3d",
					tooltip="Nx3 vertices, from 'CV Mesh From 3D Model' or 'CV "
					        "Mesh Split Long Edges'."),
				NPArray.Input("tris",
					tooltip="Mx3 int triangle indices into pts3d. The WINDING is "
					        "the whole input here: a model exported with flipped "
					        "faces gives inward normals, which is what the "
					        "orientation combo is for."),
				io.Combo.Input("orientation",
					options=["outward (follow triangle winding)",
					         "inward (flip the winding)"],
					default="outward (follow triangle winding)",
					tooltip="glTF and OBJ both wind counter-clockwise seen from "
					        "outside, so the default is right for anything a 3D "
					        "loader hands you. Flip it when a match comes back "
					        "consistently 180 degrees out AND the scene normals "
					        "are already oriented - that is the signature of a "
					        "model whose faces are inside out."),
			],
			outputs=[
				NPArray.Output(display_name="cloud",
					tooltip="Nx6 float32 (x,y,z,nx,ny,nz) - feed this to 'CV PPF "
					        "Pose Estimation', 'CV ICP Register' or 'CV Downsample "
					        "Point Cloud'."),
				NPArray.Output(display_name="normals",
					tooltip="Nx3 float32 unit normals only, in vertex order (for "
					        "drawing, filtering, or a back-face test)."),
				io.Boolean.Output(display_name="found",
					tooltip="False when the mesh is empty or an index points past "
					        "the end of pts3d."),
			],
		)

	@classmethod
	def execute(cls, pts3d, tris, orientation) -> io.NodeOutput:
		empty = (np.zeros((0, 6), np.float32), np.zeros((0, 3), np.float32))
		pts = np.asarray(pts3d, np.float64).reshape(-1, 3)
		tri = np.asarray(tris, np.int64).reshape(-1, 3)
		if len(pts) == 0 or len(tri) == 0 or tri.max(initial=-1) >= len(pts) 				or tri.min(initial=0) < 0:
			logging.warning("MeshVertexNormals: empty mesh or index out of range "
			                "(%d verts, %d tris)", len(pts), len(tri))
			return io.NodeOutput(*empty, False)

		a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
		# |cross| is twice the triangle area, so accumulating the UNnormalized
		# face normal is the area weighting - no separate weight term needed
		face = np.cross(b - a, c - a)
		acc = np.zeros_like(pts)
		for k in range(3):
			np.add.at(acc, tri[:, k], face)
		if str(orientation).startswith("inward"):
			acc = -acc

		length = np.linalg.norm(acc, axis=1, keepdims=True)
		# a vertex touched only by degenerate (zero-area) triangles has no
		# direction to give - leave it at zero rather than dividing by zero
		length[length == 0] = 1.0
		normals = np.ascontiguousarray(acc / length, np.float32)
		cloud = np.ascontiguousarray(
			np.hstack([pts.astype(np.float32), normals]), np.float32)
		return io.NodeOutput(cloud, normals, True)


class RasterizeMesh(io.ComfyNode):
	# 'CV Preview 3D (Calibrated Camera)' could only render a GLB FILE, so a mesh
	# the graph had built - split, transformed, tracked - could not be drawn at
	# all, and nothing in the pack produced METRIC depth (that node's `depth` is
	# object-relative and normalized). This closes both gaps with the same
	# rasterizer, so its mask and the viewer's picture cannot disagree.
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_RasterizeMesh",
			display_name="CV Rasterize Mesh",
			category=CATEGORY,
			description="Renders a mesh that already lives in the graph "
			            "(pts3d + tris, not a model file) through an OpenCV "
			            "camera, and returns what the rest of the pack actually "
			            "needs from a render: the SILHOUETTE as a MASK and "
			            "METRIC camera-space depth. Shares its z-buffer with 'CV "
			            "Preview 3D (Calibrated Camera)' - on the shipped truck "
			            "the two silhouettes agree at IoU 1.0000 - so the mask "
			            "you composite with and the picture you look at can never "
			            "drift apart.\n\n"
			            "rvec/tvec accept a single pose OR the Bx3x1 stacks 'CV "
			            "Rapid Track (Sequence)' emits, so one node turns a "
			            "tracked clip into a per-frame matte and a per-frame "
			            "depth map. The depth output is the metric map "
			            "'CV Preview 3D (Calibrated Camera)'.scene_depth wants, "
			            "which is how a tracked real object occludes a virtual "
			            "one.\n\n"
			            "Pinhole only: lens distortion is NOT applied (undistort "
			            "the frames first, or use the viewer node when you need "
			            "the lens). Triangles only, no texture, no shading - this "
			            "node emits geometry, not a picture.",
			inputs=[
				NPArray.Input("pts3d",
					tooltip="Nx3 vertices in OPENCV object space, from 'CV Mesh "
					        "From 3D Model' (optionally through 'CV Mesh Split "
					        "Long Edges' or 'CV Transform Points 3D')."),
				NPArray.Input("tris",
					tooltip="Mx3 int triangle indices into pts3d. Indices past "
					        "the end of pts3d yield an empty render rather than "
					        "reading out of bounds the way "
					        "cv2.rapid.drawWireframe does."),
				NPArray.Input("K",
					tooltip="3x3 camera matrix of the frames being matched. Must "
					        "be the SAME K the pose was solved with, or the "
					        "silhouette lands in the wrong place."),
				NPArray.Input("rvec",
					tooltip="Rotation, Rodrigues 3x1 - or a Bx3x1 STACK to render "
					        "a whole tracked sequence in one execution."),
				NPArray.Input("tvec",
					tooltip="Translation 3x1, or the matching Bx3x1 stack. Its "
					        "units are the units the depth output is in."),
				io.Int.Input("width", default=800, min=8, max=8192,
					tooltip="Output width in pixels. Use the frame's own size so "
					        "the mask lines up with the footage; K must match it "
					        "(halving the render means halving fx/fy/cx/cy too)."),
				io.Int.Input("height", default=600, min=8, max=8192,
					tooltip="Output height in pixels, matching the frames."),
			],
			outputs=[
				io.Mask.Output(display_name="silhouette",
					tooltip="One 0/1 MASK per pose (a MASK batch): 1 where the "
					        "mesh covers the frame. Hard-edged - there is no "
					        "supersampling here; blur it if you need a soft "
					        "matte. This is the per-frame roto of a tracked "
					        "object, ready for compositing, inpainting or a "
					        "ControlNet."),
				NPArray.Output(display_name="depth",
					tooltip="METRIC camera-space Z in the mesh's own units, "
					        "float32, and `inf` where the mesh does not cover the "
					        "pixel - which is exactly what "
					        "'CV Preview 3D (Calibrated Camera)'.scene_depth "
					        "reads as 'no measurement here, never occlude'. "
					        "Shape is HxW for a single pose and BxHxW for a pose "
					        "stack; take one frame out with 'CV Index Batch'. "
					        "Sample it at projected points ('CV Sample Array At "
					        "Points') to test whether an annotation anchor is "
					        "facing the camera."),
				io.Float.Output(display_name="coverage",
					tooltip="Mean fraction of the frame the mesh covers, over all "
					        "poses. A health signal: 0 means the model missed the "
					        "frame entirely (wrong K, wrong units, pose behind "
					        "the camera), not that the render failed."),
			],
		)

	@classmethod
	def execute(cls, pts3d, tris, K, rvec, tvec, width, height) -> io.NodeOutput:
		pts = np.asarray(polytype.resolve(pts3d)[0], np.float64).reshape(-1, 3)
		tri = np.asarray(polytype.resolve(tris)[0], np.int64).reshape(-1, 3)
		cam = np.asarray(polytype.resolve(K)[0], np.float64).reshape(3, 3)
		rvecs = np.asarray(polytype.resolve(rvec)[0], np.float64).reshape(-1, 3, 1)
		tvecs = np.asarray(polytype.resolve(tvec)[0], np.float64).reshape(-1, 3, 1)
		if len(rvecs) != len(tvecs):
			raise ValueError(
				f"rvec and tvec must hold the same number of poses, got "
				f"{len(rvecs)} and {len(tvecs)}. Wire both from the same "
				"tracker output.")
		masks, depths = [], []
		for rv, tv in zip(rvecs, tvecs):
			alpha, zbuf = render3d.rasterize_mesh(pts, tri, rv, tv, cam,
			                                      (int(width), int(height)))
			masks.append(alpha)
			depths.append(zbuf)
		mask = torch.from_numpy(np.ascontiguousarray(np.stack(masks)))
		depth = np.ascontiguousarray(np.stack(depths))
		return io.NodeOutput(mask, depth[0] if len(depth) == 1 else depth,
		                     float(np.mean(masks)))


class ProjectPointsSequence(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ProjectPointsSequence",
			display_name="CV Project Points (Sequence)",
			category=CATEGORY,
			description="Projects one set of 3D object points through a whole "
			            "STACK of poses - the Bx3x1 rvecs/tvecs 'CV Rapid Track "
			            "(Sequence)' emits - so a tracked clip drives the drawing "
			            "nodes without one 'cv2 projectPoints' per frame.\n\n"
			            "Wire a depth map from 'CV Rasterize Mesh' into "
			            "scene_depth and each point also gets a VISIBLE flag: a "
			            "point on the far side of the model is behind the "
			            "surface the camera sees, so an annotation anchored to it "
			            "should be hidden rather than drawn floating over the "
			            "object. That is the hidden-line removal an overlay needs "
			            "to look attached instead of pasted on.",
			inputs=[
				NPArray.Input("points_3d",
					tooltip="Nx3 points in the mesh's object space - annotation "
					        "anchors, measurement ends, a bounding box's corners. "
					        "Pick them off the model with the 'CV Pick Point On "
					        "Mesh' blueprint rather than typing coordinates."),
				NPArray.Input("rvecs",
					tooltip="Bx3x1 rotations (a single 3x1 works too)."),
				NPArray.Input("tvecs",
					tooltip="Bx3x1 translations, same length as rvecs."),
				NPArray.Input("K",
					tooltip="3x3 camera matrix of the frames."),
				NPArray.Input("dist_coeffs", optional=True,
					tooltip="Optional lens distortion (k1, k2, p1, p2[, k3...]). "
					        "Leave unwired for a pinhole camera. Note that 'CV "
					        "Rasterize Mesh' is pinhole-only, so a distorted "
					        "projection will NOT line up with its depth map - "
					        "undistort the frames instead of mixing the two."),
				NPArray.Input("scene_depth", optional=True,
					tooltip="Optional BxHxW (or HxW) metric depth from 'CV "
					        "Rasterize Mesh'. When wired, each projected point is "
					        "compared with the depth actually rendered at its "
					        "pixel and reported in 'visible'."),
				io.Combo.Input("hidden_points",
					options=["keep coordinates", "blank out (NaN)"],
					default="keep coordinates", optional=True,
					tooltip="What points_2d holds where 'visible' is 0. 'keep "
					        "coordinates' returns where the point WOULD be, which "
					        "is what you want for measuring or for your own "
					        "gating. 'blank out (NaN)' replaces them with NaN - "
					        "the drawing nodes skip non-finite points, so an "
					        "overlay gets hidden-line removal for free and the "
					        "rows stay index-aligned with their labels (filtering "
					        "them out instead renumbers everything). Only ever "
					        "differs when scene_depth is wired."),
				io.Float.Input("depth_tolerance", default=0.02, min=0.0,
					max=1000.0, step=0.005, optional=True, advanced=True,
					tooltip="How far BEHIND the rendered surface a point may sit "
					        "and still count as visible, in scene units. An "
					        "anchor picked ON the surface lands within a "
					        "rasterization pixel of it, so a small tolerance "
					        "(~1% of the object) absorbs that without letting "
					        "far-side points through."),
			],
			outputs=[
				NPArray.Output(display_name="points_2d",
					tooltip="BxNx2 float32 pixel positions, one row per pose. "
					        "Slice a frame out with 'CV Index Batch' and feed "
					        "'CV Draw Points' / 'Draw Labels' / 'Draw "
					        "Connections'."),
				NPArray.Output(display_name="visible",
					tooltip="BxN uint8: 1 where the point faces the camera, by "
					        "comparing its own Z with scene_depth at its pixel. "
					        "All ones when scene_depth is not wired (nothing to "
					        "test against) and 0 for points that fall outside the "
					        "frame. Gate the drawing on it with 'CV Filter "
					        "Points By Mask'."),
				NPArray.Output(display_name="depths",
					tooltip="BxN float32 camera-space Z of each point - how far "
					        "it is from the camera, in scene units. Useful to "
					        "scale a label with distance, or to sort overlapping "
					        "callouts back to front."),
			],
		)

	@classmethod
	def execute(cls, points_3d, rvecs, tvecs, K, dist_coeffs=None,
	            scene_depth=None, hidden_points="keep coordinates",
	            depth_tolerance=0.02) -> io.NodeOutput:
		pts = np.asarray(polytype.resolve(points_3d)[0], np.float64).reshape(-1, 3)
		rv = np.asarray(polytype.resolve(rvecs)[0], np.float64).reshape(-1, 3, 1)
		tv = np.asarray(polytype.resolve(tvecs)[0], np.float64).reshape(-1, 3, 1)
		cam = np.asarray(polytype.resolve(K)[0], np.float64).reshape(3, 3)
		if len(rv) != len(tv):
			raise ValueError(
				f"rvecs and tvecs must hold the same number of poses, got "
				f"{len(rv)} and {len(tv)}. Wire both from the same tracker.")
		dist = None
		if dist_coeffs is not None:
			dist = np.asarray(polytype.resolve(dist_coeffs)[0], np.float64).reshape(-1)
			if dist.size == 0:
				dist = None
		zmap = None
		if scene_depth is not None:
			zmap = np.asarray(polytype.resolve(scene_depth)[0], np.float32)
			if zmap.ndim == 2:
				zmap = zmap[None]
		if len(pts) == 0:
			empty = np.zeros((len(rv), 0, 2), np.float32)
			return io.NodeOutput(empty, np.zeros((len(rv), 0), np.uint8),
			                     np.zeros((len(rv), 0), np.float32))

		out2d, vis, zs = [], [], []
		for i, (r, t) in enumerate(zip(rv, tv)):
			p2 = cv2.projectPoints(pts, r, t, cam, dist)[0].reshape(-1, 2)
			z = (pts @ cv2.Rodrigues(r)[0].T + t.reshape(3))[:, 2]
			ok = np.isfinite(p2).all(axis=1) & (z > 0)
			if zmap is not None:
				plane = zmap[min(i, len(zmap) - 1)]
				h, w = plane.shape[:2]
				x = np.round(p2[:, 0]).astype(np.int64)
				y = np.round(p2[:, 1]).astype(np.int64)
				inside = ok & (x >= 0) & (x < w) & (y >= 0) & (y < h)
				surface = np.full(len(pts), np.inf, np.float32)
				surface[inside] = plane[y[inside], x[inside]]
				# a point BEHIND the rendered surface is on the far side
				ok = inside & (z <= surface + float(depth_tolerance))
			p2 = p2.astype(np.float32)
			if str(hidden_points).startswith("blank"):
				p2 = np.where(ok[:, None], p2, np.float32(np.nan))
			out2d.append(p2)
			vis.append(ok.astype(np.uint8))
			zs.append(z.astype(np.float32))
		return io.NodeOutput(np.ascontiguousarray(np.stack(out2d)),
		                     np.ascontiguousarray(np.stack(vis)),
		                     np.ascontiguousarray(np.stack(zs)))


class UnprojectPoints(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_UnprojectPoints",
			display_name="CV Unproject Points",
			category=CATEGORY,
			description="Turns pixel positions plus a known depth into 3D "
			            "camera-space points - the inverse of 'cv2 projectPoints' "
			            "when you already know how far away each point is. "
			            "'CV Depth to 3D Points' does this for a WHOLE depth "
			            "image; this does it for the handful of points you "
			            "actually care about (a click, a detection, a contour "
			            "vertex), which is what turns a click on a rendered "
			            "object into a 3D anchor on it.\n\n"
			            "Depth is camera-space Z (distance along the optical "
			            "axis), not distance from the camera centre - the same "
			            "convention as 'CV Rasterize Mesh'. Points whose depth is "
			            "non-finite or <= 0 come back as NaN rather than raising, "
			            "so a click that missed the object stays a valid row you "
			            "can filter out.",
			inputs=[
				NPArray.Input("points_2d",
					tooltip="Nx2 (or Nx1x2) pixel positions. 'CV Annotate "
					        "Points' produces these from clicks on the image."),
				NPArray.Input("depths",
					tooltip="N camera-space Z values, one per point - sample them "
					        "out of a depth map with 'CV Sample Array At Points'. "
					        "A single value broadcasts to every point, which is "
					        "how you unproject onto a plane at a known distance."),
				NPArray.Input("K",
					tooltip="3x3 camera matrix the points were seen through."),
				NPArray.Input("dist_coeffs", optional=True,
					tooltip="Optional lens distortion of that camera. When wired, "
					        "the points are undistorted first "
					        "(cv2.undistortPoints), so pixels measured on a real "
					        "photo unproject onto the ideal ray. Leave unwired "
					        "for a pinhole render."),
			],
			outputs=[
				NPArray.Output(display_name="points_3d",
					tooltip="Nx3 float32 points in CAMERA space (X right, Y down, "
					        "Z into the scene). To get them into the model's own "
					        "frame, invert the pose that rendered them - 'OpenCV "
					        "Pose To Matrix' -> 'cv2 invert' -> 'CV Transform "
					        "Points 3D' - which is what the 'CV Pick Point On "
					        "Mesh' blueprint wires up."),
				NPArray.Output(display_name="valid",
					tooltip="N uint8: 0 where the depth was missing (non-finite "
					        "or <= 0) and the point came back NaN. A click that "
					        "landed on the background reads 0 here."),
				io.Int.Output(display_name="count",
					tooltip="How many points unprojected successfully."),
			],
		)

	@classmethod
	def execute(cls, points_2d, depths, K, dist_coeffs=None) -> io.NodeOutput:
		pts = np.asarray(polytype.resolve(points_2d)[0], np.float64).reshape(-1, 2)
		z = np.asarray(polytype.resolve(depths)[0], np.float64).reshape(-1)
		cam = np.asarray(polytype.resolve(K)[0], np.float64).reshape(3, 3)
		if len(pts) == 0:
			return io.NodeOutput(np.zeros((0, 3), np.float32),
			                     np.zeros(0, np.uint8), 0)
		if z.size == 1:
			z = np.full(len(pts), float(z[0]))
		if len(z) != len(pts):
			raise ValueError(
				f"depths must hold one value per point (or a single value to "
				f"broadcast), got {len(z)} for {len(pts)} points.")
		dist = None
		if dist_coeffs is not None:
			dist = np.asarray(polytype.resolve(dist_coeffs)[0], np.float64).reshape(-1)
			if dist.size == 0:
				dist = None
		if dist is None:
			# normalized ray = K^-1 [u, v, 1]
			xy = (pts - cam[:2, 2]) / np.array([cam[0, 0], cam[1, 1]])
		else:
			xy = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64),
			                         cam, dist).reshape(-1, 2)
		valid = np.isfinite(z) & (z > 0) & np.isfinite(xy).all(axis=1)
		zc = np.where(valid, z, np.nan)
		out = np.stack([xy[:, 0] * zc, xy[:, 1] * zc, zc], axis=1)
		return io.NodeOutput(np.ascontiguousarray(out, np.float32),
		                     valid.astype(np.uint8), int(valid.sum()))


class TextToArray(io.ComfyNode):
	# The counterpart of 'CV Array To Text', and the missing half of 'CV Numbers
	# To Array': the string-array convention the pack already uses for captions
	# (a QR detector's `texts`, 'CV Load Labels'.labels_array) had NO node
	# that lets you simply TYPE a handful of them.
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_TextToArray",
			display_name="CV Text To Array",
			category=CATEGORY,
			description="Splits typed text into an (N,) array of STRINGS - the "
			            "form the pack's caption inputs take ('CV Draw "
			            "Labels'.labels), and the same thing a QR/barcode "
			            "detector's `texts` output or 'CV Load Labels'."
			            "labels_array already produces. Use it to name a handful "
			            "of things by hand: annotation anchors, class names for a "
			            "small classifier, legend entries.\n\n"
			            "'CV Numbers To Array' is the numeric counterpart (it "
			            "parses the same text into floats/ints instead) and 'CV "
			            "Array To Text' is the inverse. Entries keep their order "
			            "and their count, so an array of N labels lines up "
			            "one-for-one with N points - which is what every consumer "
			            "assumes.",
			inputs=[
				io.String.Input("text", multiline=True, default="",
					tooltip="The entries to split, one per line by default. "
					        "Blank lines are dropped, so a trailing newline is "
					        "harmless; surrounding whitespace on each entry is "
					        "trimmed. Empty text gives an empty array, not an "
					        "error."),
				io.Combo.Input("separator",
					options=["one per line", "comma", "semicolon", "whitespace"],
					default="one per line",
					tooltip="How to cut the text into entries. 'one per line' is "
					        "the safest for captions - it is the only option that "
					        "lets an entry contain spaces AND commas. Pick "
					        "'comma' for a list pasted from a CSV header, "
					        "'whitespace' for single-word tokens."),
				io.Boolean.Input("keep_empty", default=False, optional=True,
					advanced=True,
					tooltip="Keep entries that are empty after trimming. Off "
					        "drops them, which is what you want when the text was "
					        "typed by hand. Turn it ON when the entries must stay "
					        "aligned with a fixed-length list and a blank line "
					        "means 'no caption for this one'."),
			],
			outputs=[
				NPArray.Output(display_name="nparray",
					tooltip="(N,) numpy array of strings (dtype '<U'). Wire it "
					        "into 'CV Draw Labels'.labels, or anywhere a "
					        "detector's `texts` output would go."),
				io.Int.Output(display_name="count",
					tooltip="How many entries were produced. Check it against the "
					        "number of points you are labelling - the drawing "
					        "nodes reject a mismatch rather than silently "
					        "labelling the wrong ones."),
			],
		)

	@classmethod
	def execute(cls, text, separator, keep_empty=False) -> io.NodeOutput:
		raw = str(text)
		if separator == "one per line":
			parts = raw.splitlines()
		elif separator == "comma":
			parts = raw.split(",")
		elif separator == "semicolon":
			parts = raw.split(";")
		else:
			parts = raw.split()
		parts = [p.strip() for p in parts]
		if not keep_empty:
			parts = [p for p in parts if p]
		# dtype '<U' - the same thing np.array(list_of_str) gives the detectors,
		# so consumers cannot tell where the labels came from
		arr = np.array(parts, dtype=np.str_) if parts else np.empty(0, dtype=np.str_)
		return io.NodeOutput(arr, len(parts))


class AnnotateModelPoints(io.ComfyNode):
	# The authoring surface the anchor workflows were missing. 'CV Pick Point On
	# Mesh' turns a click on a PHOTO into an anchor, which only reaches surfaces
	# the camera can currently see; this one annotates the MODEL itself, so an
	# anchor can sit on the far side, and it needs no pose, no frame and no
	# calibration to exist.
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_AnnotateModelPoints",
			display_name="CV Annotate Points On Model",
			category=CATEGORY,
			is_experimental=True,
			is_output_node=True,
			description="Manual 3D point input: run the workflow once to load the "
			            "model into the node, then ORBIT to the side you want and "
			            "LEFT-CLICK the surface to drop an anchor. SHIFT+click an "
			            "existing marker removes it; drag orbits instead of "
			            "adding, so the view is free. The points come out in the "
			            "MODEL's own frame, which is what makes them ride the "
			            "object once a tracker moves it - feed them to 'CV Project "
			            "Points (Sequence)' or the 'CV Annotate Model' "
			            "blueprint.\n\n"
			            "A plain orbit viewer on purpose: no calibrated camera, no "
			            "lens distortion, no pose. Those matter when you are "
			            "matching footage; here you are only pointing at geometry, "
			            "and a free camera is the fastest way to reach the far "
			            "side of an object.\n\n"
			            "The picks live in the 'points' widget as JSON "
			            "([[x, y, z], ...]), so they survive a save/reload, can be "
			            "typed or pasted by hand, and can be diffed. GLB only "
			            "(Three.js GLTFLoader), same as the calibrated viewer.",
			inputs=[
				io.MultiType.Input("model_3d",
					types=[io.File3DGLB, io.File3DGLTF, io.File3DAny],
					tooltip="Model to annotate, from a 3D loader node. Use the "
					        "SAME file you feed 'CV Mesh From 3D Model', or the "
					        "anchors will refer to different geometry."),
				io.String.Input("points", default="[]", multiline=True,
					tooltip="The picked anchors as JSON, [[x, y, z], ...], in the "
					        "coordinates the FILE stores (glTF's Y-up). The viewer "
					        "writes this as you click; edit it by hand for exact "
					        "values. Empty or '[]' means no anchors yet - a valid "
					        "state, not an error."),
				io.Combo.Input("axis_convention",
					options=["OpenCV (Y down, Z into the scene)",
					         "glTF / Three.js (Y up)"],
					default="OpenCV (Y down, Z into the scene)",
					tooltip="Which convention the OUTPUT is in. Leave it on "
					        "OpenCV to match 'CV Mesh From 3D Model's default, so "
					        "the anchors land in the same frame as the mesh a pose "
					        "was solved against. Mismatch it and the anchors "
					        "mirror through the object."),
			],
			outputs=[
				NPArray.Output(display_name="points_3d",
					tooltip="Nx3 float32 anchors in the model's own frame, in the "
					        "order they were clicked - so the Nth anchor keeps the "
					        "Nth caption from 'CV Text To Array'."),
				io.Int.Output(display_name="count",
					tooltip="How many anchors are set. 0 is valid: the node emits "
					        "an empty (0, 3) array rather than failing, so a graph "
					        "still executes before you have clicked anything."),
			],
		)

	@classmethod
	def execute(cls, model_3d, points="[]", axis_convention=
	            "OpenCV (Y down, Z into the scene)") -> io.NodeOutput:
		import uuid

		import folder_paths
		raw = str(points).strip() or "[]"
		try:
			data = json.loads(raw)
		except json.JSONDecodeError as e:
			raise ValueError(
				f'points must be JSON like [[x, y, z], ...], got "{points}". ({e})'
			) from e
		if not isinstance(data, list) or not all(
				isinstance(p, (list, tuple)) and len(p) == 3
				and all(isinstance(v, (int, float)) and not isinstance(v, bool)
				        for v in p) for p in data):
			raise ValueError(
				"points must be a JSON list of [x, y, z] triples (the viewer "
				f'writes it for you), got "{points}".')
		pts = np.asarray(data, np.float64).reshape(-1, 3)
		if str(axis_convention).startswith("OpenCV"):
			pts = pts * render3d._FLIP

		# stage the model where the browser widget can fetch it, exactly as
		# Preview3DCalibrated does - a fresh name per run so the viewer reloads
		payload = {}
		if model_3d is not None and not isinstance(model_3d, str):
			temp_dir = folder_paths.get_temp_directory()
			os.makedirs(temp_dir, exist_ok=True)
			fmt = getattr(model_3d, "format", None) or "glb"
			name = f"cv_annot3d_{uuid.uuid4().hex}.{fmt}"
			model_3d.save_to(os.path.join(temp_dir, name))
			payload["model"] = name
		return io.NodeOutput(np.ascontiguousarray(pts, np.float32), len(pts),
		                     ui={"cv3d_annotate": [payload]})


# The pack ships its own sample media in example_inputs/, but a LoadImage /
# LoadVideo / Load3D combo can only offer what is already inside
# ComfyUI/input - so every example workflow used to open with a red node and a
# README instruction to copy files by hand. This node does the copy from
# inside the graph, which is the one place the user is guaranteed to be when
# they hit the missing file.
_EXAMPLE_INPUTS_DIR = os.path.join(
	os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_inputs")

# Core's Load3D nodes list `input/3d` only (comfy_extras/nodes_load_3d.py), so a
# model dropped in the input ROOT is invisible to them. Everything else - images,
# video, the calibration JSON/YAML/CSV/TXT sidecars - is listed from the root.
_MODEL_3D_EXTS = (".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply")

_INSTALL_MODES = ["copy the files",
                  "list what would be copied (dry run)"]
_INSTALL_EXISTING = ["keep the file already there",
                     "overwrite it with the packaged copy"]


class InstallExampleInputs(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_InstallExampleInputs",
			display_name="CV Install Example Inputs",
			category="image/CV",
			description="Copies the sample media this pack ships in its "
			            "'example_inputs' folder into ComfyUI's 'input' folder, so "
			            "the LoadImage / LoadVideo / Load3D dropdowns in the example "
			            "workflows actually find their files. Run it once after "
			            "installing the pack, then reload the page (the combos are "
			            "built when the node definitions are fetched). 3D models go "
			            "to 'input/3d' because that is the only place core's Load3D "
			            "nodes look; everything else lands in 'input' itself. "
			            "Nothing outside 'input' is touched and nothing is deleted. "
			            "Failure tolerant: a missing source folder or an unreadable "
			            "file is reported with ok=false instead of halting the run.",
			inputs=[
				io.Combo.Input("mode", options=_INSTALL_MODES,
					tooltip="'copy the files' writes them. The dry run touches "
					        "nothing and reports exactly what a real run would do - "
					        "use it first if the input folder is shared with other "
					        "packs or holds your own work."),
				io.Combo.Input("existing_files", options=_INSTALL_EXISTING,
					tooltip="What to do when a file of the same name is already in "
					        "the input folder. Keeping it is safe (your own file "
					        "wins, and a second run of this node is then a no-op); "
					        "overwrite is for repairing a sample you edited in "
					        "place. Only names that collide with a packaged file "
					        "are ever considered."),
			],
			outputs=[
				io.String.Output(display_name="report",
					tooltip="Multi-line summary: the counts and one line per file "
					        "that failed. Wire it into core 'Preview as Text' to "
					        "read it. Destinations are named RELATIVE to the "
					        "ComfyUI folder and any error text is passed through "
					        "the same path redaction as 'CV Build Information', so "
					        "the report can go straight into a bug report."),
				io.Int.Output(display_name="copied",
					tooltip="How many files were written (always 0 in dry-run mode, "
					        "where 'would_copy' carries the number instead)."),
				io.Int.Output(display_name="would_copy",
					tooltip="How many files a real run would write with these "
					        "settings - the dry run's answer, and equal to 'copied' "
					        "after a successful real run."),
				io.Int.Output(display_name="skipped",
					tooltip="Files already present that were left alone. Everything "
					        "is skipped on the second run of an unchanged install, "
					        "which is how you tell the copy took."),
				io.Boolean.Output(display_name="ok",
					tooltip="False when the source folder is missing or any single "
					        "file failed to copy - the run still finishes and the "
					        "report names what went wrong. Gate a follow-up branch "
					        "on it with 'If/Else Switch'."),
			],
			is_output_node=True,
		)

	@classmethod
	def execute(cls, mode, existing_files) -> io.NodeOutput:
		import folder_paths
		dry = str(mode).startswith("list")
		overwrite = str(existing_files).startswith("overwrite")
		input_dir = folder_paths.get_input_directory()
		# Destinations are named RELATIVE, and the input folder never leaves the
		# node as a value. An absolute path here would be the same leak
		# 'CV Build Information' was written to close (buildinfo.redact_paths):
		# a report is pasted into bug threads and a workflow JSON is shared with
		# the widget values in it, and either one would then carry the user's
		# drive layout, install location and user name.
		lines = ["example_inputs -> ComfyUI/input "
		         "(3D models -> ComfyUI/input/3d)"]

		if not os.path.isdir(_EXAMPLE_INPUTS_DIR):
			lines.append("MISSING: this pack has no example_inputs folder - "
			             "re-download the pack, or copy the files by hand.")
			report = "\n".join(lines)
			return io.NodeOutput(report, 0, 0, 0, False,
			                     ui=ui.PreviewText(report))

		names = sorted(n for n in os.listdir(_EXAMPLE_INPUTS_DIR)
		               if os.path.isfile(os.path.join(_EXAMPLE_INPUTS_DIR, n)))
		copied = would = skipped = 0
		failed = []
		for name in names:
			src = os.path.join(_EXAMPLE_INPUTS_DIR, name)
			sub = "3d" if name.lower().endswith(_MODEL_3D_EXTS) else ""
			dst_dir = os.path.join(input_dir, sub) if sub else input_dir
			dst = os.path.join(dst_dir, name)
			if os.path.exists(dst) and not overwrite:
				skipped += 1
				continue
			would += 1
			if dry:
				continue
			try:
				os.makedirs(dst_dir, exist_ok=True)
				# copy2, not copy: the modification time is what tells a user
				# whether the sample in input/ is the packaged one or their edit
				shutil.copy2(src, dst)
				copied += 1
			except OSError as e:
				# str(OSError) ends in the absolute path it failed on
				# ("[Errno 13] Permission denied: 'C:/.../input/boat1.jpg'"),
				# so it goes through the Build Information redactor before it
				# is allowed into an output the user will paste somewhere.
				reason, _hidden = buildinfo.redact_paths(str(e), keep_names=False)
				failed.append(f"FAILED {name}: {reason}")

		verb = "would copy" if dry else "copied"
		lines.append(f"{len(names)} packaged files: {verb} {would}, "
		             f"already present {skipped}, failed {len(failed)}")
		if dry:
			lines.append("DRY RUN - nothing was written.")
		lines.extend(failed)
		if not dry and not failed:
			lines.append("Reload the ComfyUI page so the Load* dropdowns list them.")
		report = "\n".join(lines)
		return io.NodeOutput(report, copied, would, skipped, not failed,
		                     ui=ui.PreviewText(report))

	@classmethod
	def fingerprint_inputs(cls, mode, existing_files):
		# The input folder is not an input, so ComfyUI would serve the first
		# run's cached result forever - including to a user who ran the dry run,
		# read it, and switched to 'copy the files' after deleting something.
		# NaN never equals itself, so the node re-runs on every queue; the work
		# is a 33 MB copy that skips everything already there.
		return float("NaN")


NODES = [ImageToCV, ImageBatchToCV, CVBatchToImage, CVToImage, MaskToCV, CVToMask, CVToLatent, LatentToCV, AnyToCV, ScalarArray, TermCriteria,
         WarpFlags, DFTFlags, DCTFlags, OptFlowFlags, ChessboardFlags, GemmFlags, SVDFlags,
         FloodFillFlags, ThreshFlags, GridPoints, PointsLiteral, SplitPoint, ParseMatrix, CameraMatrix, CastArray, SetRNGSeed,
          ReshapeArray, PermuteAxes, SliceArray, ToBlob, UnstackBatch, StackBatch, ConcatArrays, IndexBatch, MosaicBayer, CVRoll, ArrayStat, ReduceArrayByLabel, TakeByIndex, CVPreview, CVInspect, ArrayToText, ArrayToNumbers, NumbersToArray, SaveCameraParams,
         LoadCameraParams, CameraPoseTo3DView, Preview3DCalibrated, CVArrayShape, CVDType,
         ColorFromHex, WritePly, DepthTo3D, FilterPointCloud, FlipAxis,
         ConvertAxisConvention,
         ObjectTransform3D, ParseObjectTransform,
         MeshFrom3DModel, MeshSplitLongEdges, MeshVertexNormals, RasterizeMesh,
         ProjectPointsSequence, UnprojectPoints, TextToArray,
         AnnotateModelPoints, BuildInfo, InstallExampleInputs]
