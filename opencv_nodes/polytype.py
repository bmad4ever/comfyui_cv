# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Polymorphic IMAGE/MASK/NPARRAY socket helpers.

Mirrors the core "Resize Image/Mask" node pattern (comfy_extras/
nodes_post_processing.py): one socket accepts several types via
io.MultiType / io.MatchType, the concrete type is detected at runtime,
and MatchType-linked outputs come back in the same format as the input.

Conventions:
* IMAGE (torch [B,H,W,C]) converts to a single uint8 BGR ndarray (frame 0).
* MASK  (torch [B,H,W])   converts to a single uint8 [H,W] ndarray (frame 0).
* LATENT ({"samples": [B,C,H,W]}) converts to a float32 [H,W,C] ndarray
  (frame 0), values untouched - latents are unbounded floats, never uint8.
* NPARRAY passes through untouched.
* restore() converts a result ndarray back to the tagged input format, so
  type-preserving nodes echo the caller's type like the core Resize node.
"""

import cv2
import numpy as np
import torch

from comfy_api.latest import io

from . import imageutils
from .convert import NPArray

# NPARRAY first: existing workflows link NPARRAY, keep it the primary type
IMAGEISH = [NPArray, io.Image, io.Mask]
IMAGE_ONLY = [NPArray, io.Image]
MASK_ONLY = [NPArray, io.Mask]

BATCH_NOTE = ("Accepts a ComfyUI IMAGE/MASK directly (frame 0 of a batch)"
	" or an NPARRAY.  Arithmetic ops (add, multiply, etc.) process the"
	" full IMAGE batch when both inputs have the same batch size.")
LATENT_NOTE = (" A LATENT link is processed in latent space: frame 0 becomes a "
	"float32 [H,W,C] array (any channel count), values untouched."
	" Arithmetic ops (add, multiply, etc.) also accept a full LATENT batch"
	" ({samples: [B,C,H,W]}) — the whole batch flows through when both"
	" inputs have the same batch size.")


def with_latent(types):
	"""Extend a socket type list with LATENT (channel-agnostic float ops only)."""
	return list(types) + [io.Latent]


def resolve(value):
	"""Coerce an image-ish socket value to an ndarray, remembering its type.

	Returns (ndarray_or_None, tag) with tag in {"nparray", "image", "mask",
	"latent"}. Torch tensors follow the core convention: 4-D [B,H,W,C] is an
	IMAGE (-> uint8 BGR frame 0), anything else is a MASK (-> uint8 [H,W]
	frame 0). A LATENT dict yields float32 [H,W,C] (frame 0, values as-is).
	ndarrays and None pass through with tag "nparray".
	"""
	if isinstance(value, torch.Tensor):
		if value.ndim == 4:
			return imageutils.image_batch_to_bgr(value)[0], "image"
		return imageutils.mask_batch_to_u8(value)[0], "mask"
	if is_latent(value):
		samples = value["samples"]
		if samples.ndim != 4:
			raise ValueError(
				f"Only 4-D [B,C,H,W] latents are supported, got shape "
				f"{tuple(samples.shape)} (video/3-D latents cannot map to a "
				"cv2 image).")
		arr = samples[0].movedim(0, -1).float().cpu().numpy()
		return np.ascontiguousarray(arr), "latent"
	return value, "nparray"


def restore(arr, tag):
	"""Convert a result ndarray back to the format `resolve` tagged."""
	if tag == "image":
		return imageutils.bgr_to_image_batch([np.asarray(arr)])
	if tag == "mask":
		return imageutils.u8_to_mask_batch([np.asarray(arr)])
	if tag == "latent":
		arr = np.asarray(arr, dtype=np.float32)
		if arr.ndim == 2:
			arr = arr[:, :, None]
		samples = torch.from_numpy(np.ascontiguousarray(arr)).movedim(-1, 0)
		# noise_mask/batch_index are dropped on purpose: a spatial op has
		# invalidated their geometry anyway
		return {"samples": samples.unsqueeze(0)}
	return arr


def is_latent(value):
	return isinstance(value, dict) and "samples" in value


def resolve_batch(value):
	"""Extract a full LATENT batch as a [B,C,H,W] float32 numpy array.

	For LATENT_BATCH_SAFE cv2 ops (arithmetic) that accept 4-D arrays
	directly.  Returns (ndarray, "latent") for LATENT inputs, falls back
	to resolve() for everything else.
	"""
	if not is_latent(value):
		return resolve(value)
	samples = value["samples"]
	if samples.ndim != 4:
		raise ValueError(
			f"Only 4-D [B,C,H,W] latents are supported, got shape "
			f"{tuple(samples.shape)} (video/3-D latents cannot map to a "
			"cv2 image).")
	return samples.detach().float().cpu().numpy().copy(), "latent"


def restore_batch(arr):
	"""Wrap a [B,C,H,W] float32 numpy array back as a LATENT dict."""
	arr = np.asarray(arr, dtype=np.float32)
	return {"samples": torch.from_numpy(np.ascontiguousarray(arr))}


def spatial_size(value):
	"""(height, width) of any image-ish socket value, without converting it.

	Works on IMAGE ([B,H,W,C]), MASK ([B,H,W] or [H,W]), LATENT (samples
	[B,C,H,W] - size in latent cells) and NPARRAY ([H,W,...]) values.
	"""
	if isinstance(value, torch.Tensor):
		if value.ndim == 4:
			return int(value.shape[1]), int(value.shape[2])
		return int(value.shape[-2]), int(value.shape[-1])
	if is_latent(value):
		samples = value["samples"]
		return int(samples.shape[-2]), int(samples.shape[-1])
	arr = np.asarray(value)
	if arr.ndim < 2:
		raise ValueError(
			f"Cannot read a height/width off a {arr.ndim}-D array "
			f"(shape {tuple(arr.shape)}).")
	return int(arr.shape[0]), int(arr.shape[1])


def type_tag(value):
	"""The `resolve` tag of a value, without converting anything."""
	if isinstance(value, torch.Tensor):
		return "image" if value.ndim == 4 else "mask"
	if is_latent(value):
		return "latent"
	return "nparray"


def batch_size(value):
	"""Number of frames an image-ish socket value carries.

	An NPARRAY is [H,W,...] - ONE frame - by this pack's convention, with one
	exception: a 4-D array is the stack `concat` builds, i.e. [N,H,W,C]. That
	is what lets a batch of NPARRAY crops flow back into 'Paste by BBox'.
	"""
	if isinstance(value, torch.Tensor):
		return int(value.shape[0]) if value.ndim >= 3 else 1
	if is_latent(value):
		return int(value["samples"].shape[0])
	arr = np.asarray(value)
	return int(arr.shape[0]) if arr.ndim == 4 else 1


def select_frame(value, index):
	"""One frame of an image-ish value, still in its own type (batch of 1).

	Cropping/pasting must not round-trip through uint8 BGR the way
	`resolve` does - a crop is a lossless operation and IMAGE/LATENT are
	float. So these helpers slice the native container instead.
	"""
	n = batch_size(value)
	index = int(index) % n if n else 0
	if isinstance(value, torch.Tensor):
		return value[index:index + 1] if value.ndim >= 3 else value.unsqueeze(0)
	if is_latent(value):
		out = {k: v for k, v in value.items() if k not in ("noise_mask", "batch_index")}
		out["samples"] = value["samples"][index:index + 1]
		return out
	arr = np.asarray(value)
	return arr[index] if arr.ndim == 4 else arr


def spatial_crop(value, x, y, w, h):
	"""Slice an image-ish value spatially, preserving type, dtype and batch."""
	x, y, w, h = int(x), int(y), int(w), int(h)
	if isinstance(value, torch.Tensor):
		if value.ndim == 4:                       # IMAGE [B,H,W,C]
			return value[:, y:y + h, x:x + w, :].contiguous()
		if value.ndim == 3:                       # MASK  [B,H,W]
			return value[:, y:y + h, x:x + w].contiguous()
		return value[y:y + h, x:x + w].contiguous()
	if is_latent(value):
		out = {k: v for k, v in value.items() if k not in ("noise_mask", "batch_index")}
		out["samples"] = value["samples"][:, :, y:y + h, x:x + w].contiguous()
		return out
	arr = np.asarray(value)
	if arr.ndim < 2:
		raise ValueError(f"Cannot crop a {arr.ndim}-D array (shape {tuple(arr.shape)}).")
	return np.ascontiguousarray(arr[y:y + h, x:x + w])


def concat(items, tag):
	"""Stack same-shaped, same-typed crops into one batch."""
	if not items:
		return empty_batch(tag)
	if tag == "latent":
		out = {k: v for k, v in items[0].items() if k != "samples"}
		out["samples"] = torch.cat([it["samples"] for it in items], dim=0)
		return out
	if tag in ("image", "mask"):
		return torch.cat([it if it.ndim >= 3 else it.unsqueeze(0) for it in items], dim=0)
	return np.stack([np.asarray(it) for it in items])


def empty_batch(tag, channels=3):
	"""A zero-row batch of `tag`'s type - the "nothing found" output."""
	if tag == "image":
		return torch.zeros((0, 1, 1, 3))
	if tag == "mask":
		return torch.zeros((0, 1, 1))
	if tag == "latent":
		return {"samples": torch.zeros((0, max(1, channels), 1, 1))}
	return np.zeros((0, 1, 1), dtype=np.float32)


def clone(value):
	"""A writable copy of an image-ish value (paste must never write into an
	upstream node's cached output - same rule as lowlevel.py's copy-before-cv2)."""
	if isinstance(value, torch.Tensor):
		return value.clone()
	if is_latent(value):
		out = {k: v for k, v in value.items()}
		out["samples"] = value["samples"].clone()
		return out
	return np.array(value, copy=True)


def spatial_resize(value, width, height):
	"""Resample an image-ish value to (width, height), preserving type/dtype."""
	import torch.nn.functional as F

	width, height = max(1, int(width)), max(1, int(height))

	def _resize_bchw(t):
		mode = "area" if (t.shape[-2] > height and t.shape[-1] > width) else "bilinear"
		kw = {} if mode == "area" else {"align_corners": False}
		return F.interpolate(t.float(), size=(height, width), mode=mode, **kw)

	if isinstance(value, torch.Tensor):
		if value.ndim == 4:                        # IMAGE [B,H,W,C]
			out = _resize_bchw(value.movedim(-1, 1)).movedim(1, -1)
			return out.to(value.dtype).contiguous()
		if value.ndim == 3:                        # MASK [B,H,W]
			out = _resize_bchw(value.unsqueeze(1)).squeeze(1)
			return out.to(value.dtype).contiguous()
		out = _resize_bchw(value[None, None])[0, 0]
		return out.to(value.dtype).contiguous()
	if is_latent(value):
		out = {k: v for k, v in value.items() if k not in ("noise_mask", "batch_index")}
		out["samples"] = _resize_bchw(value["samples"]).to(
			value["samples"].dtype).contiguous()
		return out
	arr = np.asarray(value)
	interp = cv2.INTER_AREA if (arr.shape[0] > height and arr.shape[1] > width) \
		else cv2.INTER_LINEAR
	# cv2.resize is capped at 4 channels - resize many-channel arrays in slices
	if arr.ndim == 3 and arr.shape[2] > 4:
		planes = [cv2.resize(arr[..., c], (width, height), interpolation=interp)
		          for c in range(arr.shape[2])]
		return np.ascontiguousarray(np.stack(planes, axis=-1))
	return np.ascontiguousarray(cv2.resize(arr, (width, height), interpolation=interp))


def paste_into(dst, src, x, y, alpha=None, premultiplied=False):
	"""Blend `src` into `dst` at (x, y). Both must share the same tag.

	`alpha` is an optional float32 [h, w] ndarray in 0..1 (crop space); it is
	broadcast over channels. `dst` is modified in place and returned.

	`premultiplied` says `src` has ALREADY been multiplied by `alpha` - which is
	what a warp onto a BORDER_CONSTANT canvas produces - so the blend is
	`src + dst * (1 - alpha)` instead of the usual lerp. Blending such a source
	a second time with the same alpha squares it and leaves a dark rim on every
	antialiased boundary pixel. The sum can overshoot, so an integer result is
	clipped to its dtype range.
	"""
	x, y = int(x), int(y)
	if is_latent(dst):
		d, s = dst["samples"], src["samples"]
		h, w = s.shape[-2], s.shape[-1]
		region = d[:, :, y:y + h, x:x + w]
		if alpha is None:
			region.copy_(s[:region.shape[0]])
		else:
			a = torch.from_numpy(alpha).to(region.dtype).to(region.device)
			patch = s[:region.shape[0]]
			region.mul_(1.0 - a).add_(patch if premultiplied else patch * a)
		return dst
	if isinstance(dst, torch.Tensor):
		if dst.ndim == 4:                          # IMAGE [B,H,W,C]
			h, w = src.shape[1], src.shape[2]
			region = dst[:, y:y + h, x:x + w, :]
			a = None if alpha is None else torch.from_numpy(alpha[..., None]).to(region.dtype)
		else:                                      # MASK [B,H,W]
			h, w = src.shape[-2], src.shape[-1]
			region = dst[:, y:y + h, x:x + w]
			a = None if alpha is None else torch.from_numpy(alpha).to(region.dtype)
		patch = src if src.ndim == dst.ndim else src.unsqueeze(0)
		patch = patch[:region.shape[0]].to(region.dtype).to(region.device)
		if a is None:
			region.copy_(patch)
		else:
			a = a.to(region.device)
			region.mul_(1.0 - a).add_(patch if premultiplied else patch * a)
		return dst
	arr = np.asarray(dst)
	patch = np.asarray(src)
	h, w = patch.shape[0], patch.shape[1]
	region = arr[y:y + h, x:x + w]
	if alpha is None:
		region[...] = patch.astype(region.dtype)
	else:
		a = alpha if region.ndim == 2 else alpha[..., None]
		p = patch.astype(np.float32)
		blended = (p if premultiplied else p * a) + region.astype(np.float32) * (1.0 - a)
		if np.issubdtype(region.dtype, np.integer):
			info = np.iinfo(region.dtype)
			blended = np.clip(blended, info.min, info.max)
		region[...] = blended.astype(region.dtype)
	return arr


def _note(tooltip):
	return ((tooltip + " ") if tooltip else "") + BATCH_NOTE


def poly_input(id, types=None, tooltip=None, **kw):
	"""MultiType input: accepts NPARRAY plus IMAGE and/or MASK, no output link."""
	return io.MultiType.Input(id, types or IMAGEISH, tooltip=_note(tooltip), **kw)


def match_template(template_id="image_type", types=None):
	"""MatchType template shared by an input and the outputs that echo its type."""
	return io.MatchType.Template(template_id, types or IMAGEISH)


def match_input(id, template, tooltip=None, **kw):
	return io.MatchType.Input(id, template=template, tooltip=_note(tooltip), **kw)


def match_output(template, display_name, tooltip=None, **kw):
	return io.MatchType.Output(template=template, display_name=display_name,
		tooltip=tooltip, **kw)
