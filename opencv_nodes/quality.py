# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Full coverage of `cv2.quality`, the objective image-quality module.

One module owns all of cv2.quality. 'CV Quality Compare' is the STATIC
one-shot call, `Quality*_compute(ref, cmp)` for a single pair of images (it
lived in `contrib.py` until the class-API nodes arrived; its node_id is frozen
as `CV_ImageQuality` because saved workflows key on it). The other two
halves of the module are only reachable through the CLASS API, and they are the
halves that matter for real comparisons:

* `Quality*_create(ref)` keeps the reference's PRECOMPUTED state (SSIM caches
  the reference's mean/variance maps, GMSD its gradient magnitudes) and scores
  any number of candidates against it. Comparing eight denoiser settings with
  the static call re-derives the reference eight times; 'CV Quality Compare
  Batch' derives it once and scores the whole IMAGE batch, which is what a
  parameter sweep actually produces.

* `QualityBRISQUE` is NO-REFERENCE: it scores an image with nothing to compare
  it to, by fitting an SVM over 36 natural-scene-statistics features. That is
  the only metric here that can grade a single image, and its feature vector
  (`computeFeatures`) is a generic 36-d descriptor - feed it to `ml.py`'s
  'Train Classifier' to learn a quality/defect classifier of your own.

Node ids keep the module visible (`CV_Quality_*`, display 'CV Quality
...') so they read as cv2.quality at a glance.

As in `contrib.py`, every cv2 object is created, used and dropped inside one
`execute()`: a `QualityBase` is not an ndarray and ComfyUI's output caching
would keep mutating a live one across re-executions.

Gotcha, pinned by `test_quality_compare_batch`: `QualityMSE`'s INSTANCE
`compute()` takes an `InputArrayOfArrays` and its python binding rejects every
form of a numpy image on this build ("Unsupported input type in
quality_utils::extract_mat" for an ndarray, a list, a 4-D stack or a UMat;
a UMat even resolves to the 2-argument STATIC overload instead). SSIM, PSNR and
GMSD take a plain mat and work. So MSE alone goes through the static
`QualityMSE_compute(ref, cmp)` per frame - correct, just without the cached
reference state. Also note GMSD's quality map comes back at HALF resolution
(the metric works on a downsampled gradient map); don't assume a map has the
input's size.

BRISQUE needs two data files (an SVM model and its feature ranges) which
OpenCV does not ship with the wheel. They live in `ComfyUI/models/quality`;
get them from opencv_contrib (`modules/quality/samples/brisque_model_live.yml`
and `brisque_range_live.yml`).
"""

import os

import cv2
import numpy as np
import torch

import comfy.utils
from comfy_api.latest import io

from . import imageutils, polytype
from .convert import NPArray

CATEGORY = "image/CV/quality"

_QUALITY_DIRNAME = "quality"

# (factory, higher-is-better) - the direction is per metric and there is no way
# to read it off the object, so it travels as an output BOOLEAN (see below).
METRICS = {
	"SSIM (structural similarity)": ("QualitySSIM", True),
	"PSNR (dB)": ("QualityPSNR", True),
	"MSE (mean squared error)": ("QualityMSE", False),
	"GMSD (gradient magnitude similarity)": ("QualityGMSD", False),
}


def _quality(attr: str = None):
	"""cv2.quality, or a clear error naming the repair tool (see contrib.py)."""
	mod = getattr(cv2, "quality", None)
	if mod is not None and (attr is None or hasattr(mod, attr)):
		return mod
	raise RuntimeError(
		f"cv2.quality{'.' + attr if attr else ''} is not available in this "
		f"OpenCV build ({cv2.__version__}). The opencv-contrib modules are "
		"missing - most likely plain opencv-python was installed over "
		"opencv-contrib-python(-headless) (they share one site-packages/cv2). Stop the "
		"ComfyUI server and run: python tools/repair_opencv_contrib.py --apply")


def _register_quality_folder():
	"""Register models/quality with folder_paths (idempotent).

	Same pattern as ml._register_cascade_folder: runs at import AND inside
	define_schema so the combo is populated on a cold server. The extension set
	is {".yml", ".yaml", ".xml"} because the BRISQUE data is a cv2 FileStorage
	dump, which cv2 writes in any of those.
	"""
	import folder_paths

	if _QUALITY_DIRNAME not in folder_paths.folder_names_and_paths:
		path = os.path.join(folder_paths.models_dir, _QUALITY_DIRNAME)
		os.makedirs(path, exist_ok=True)
		folder_paths.folder_names_and_paths[_QUALITY_DIRNAME] = (
			[path], {".yml", ".yaml", ".xml"})


def _quality_files() -> list:
	import folder_paths

	_register_quality_folder()
	return [f.replace(os.sep, "/")
	        for f in folder_paths.get_filename_list(_QUALITY_DIRNAME)]


def _quality_path(name: str) -> str:
	"""Resolve a combo choice to a real path, contained to models/quality.

	Combo options only constrain the frontend - the /prompt API takes any
	string - so this is where a path outside the registered roots is refused.
	"""
	import folder_paths

	_register_quality_folder()
	name = (name or "").strip()
	if not name:
		raise ValueError(
			"no BRISQUE data file selected. Put brisque_model_live.yml and "
			"brisque_range_live.yml (opencv_contrib, "
			"modules/quality/samples/) in ComfyUI/models/quality.")
	path = folder_paths.get_full_path(_QUALITY_DIRNAME, name)
	if path is None:
		path = folder_paths.get_full_path(
			_QUALITY_DIRNAME, name.replace("/", os.sep))
	if path is None:
		available = _quality_files() or (
			"(none - download brisque_model_live.yml / brisque_range_live.yml "
			"from opencv_contrib, modules/quality/samples)")
		raise ValueError(
			f"'{name}' is not in ComfyUI/models/quality. Available: {available}")
	roots = [os.path.realpath(r)
	         for r in folder_paths.folder_names_and_paths[_QUALITY_DIRNAME][0]]
	real = os.path.realpath(path)
	if not any(real == r or real.startswith(r + os.sep) for r in roots):
		raise ValueError(f"'{name}' resolves outside ComfyUI/models/quality")
	return real


_register_quality_folder()


def _bgr_frames(image) -> list:
	"""Any image-ish input -> list of uint8 BGR frames (whole batch)."""
	if isinstance(image, torch.Tensor):
		if image.ndim == 4:
			frames = imageutils.image_batch_to_bgr(image)
		else:
			frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
			          for f in imageutils.mask_batch_to_u8(image)]
	else:
		arr = np.asarray(polytype.resolve(image)[0])
		if arr.ndim == 2:
			frames = [cv2.cvtColor(imageutils.ensure_uint8(arr),
			                       cv2.COLOR_GRAY2BGR)]
		elif arr.ndim == 4:
			frames = [f for f in arr]
		else:
			frames = [arr]
	out = []
	for f in frames:
		f = np.asarray(f)
		if f.dtype != np.uint8:
			f = np.clip(f * 255.0 if f.max() <= 1.0 else f, 0, 255).astype(np.uint8)
		out.append(np.ascontiguousarray(f))
	if not out:
		raise ValueError("empty image input")
	return out


def _channel_mean(score) -> float:
	"""cv2 returns a 4-element Scalar; average the meaningful channels.

	PSNR is +inf for identical images - report a large finite number so
	downstream arithmetic/comparison still works instead of being poisoned.
	"""
	vals = [float(v) for v in np.asarray(score).ravel()[:3]]
	vals = [1e9 if np.isinf(v) else (0.0 if np.isnan(v) else v) for v in vals]
	return float(np.mean(vals))


class QualityCompareBatch(io.ComfyNode):
	"""`Quality*_create(ref)` + `.compute(frame)` over a whole IMAGE batch."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Quality_CompareBatch",
			display_name="CV Quality Compare Batch (cv2.quality)",
			category=CATEGORY,
			description="Scores EVERY frame of an image batch against one "
			            "reference (cv2.quality class API). This is the node "
			            "for a parameter sweep: render N candidates into a "
			            "batch, get N scores out, and pick the winner with the "
			            "'best_index' output. The reference's internal state is "
			            "computed ONCE and reused for all N frames, which the "
			            "one-shot 'CV Image Quality (compare)' cannot do. "
			            "Read the direction: SSIM/PSNR higher is better, "
			            "MSE/GMSD lower is better - 'higher_is_better' says "
			            "which, and 'best_index' already accounts for it. All "
			            "frames must match the reference's size.",
			inputs=[
				polytype.poly_input("reference", types=polytype.IMAGEISH,
					tooltip="The ground-truth / original image (frame 0 if a "
					        "batch). Its quality state is computed once."),
				polytype.poly_input("images", types=polytype.IMAGEISH,
					tooltip="Batch of processed candidates to score. Every "
					        "frame must have the reference's size and channel "
					        "count."),
				io.Combo.Input("metric",
					options=list(METRICS),
					default="SSIM (structural similarity)",
					tooltip="SSIM tracks perceived structural change (0-1, 1 = "
					        "identical). PSNR is a log-scale error in dB (>40 "
					        "is typically indistinguishable). MSE is the raw "
					        "squared error. GMSD compares gradient structure "
					        "and correlates well with perceived sharpness loss."),
				io.Float.Input("max_pixel_value", default=255.0,
					min=1.0, max=65535.0, step=1.0,
					optional=True, advanced=True,
					tooltip="PSNR only: the maximum per-channel pixel value "
					        "used as the numerator (255 for 8-bit images). "
					        "Ignored by the other metrics."),
			],
			outputs=[
				io.Float.Output(display_name="best_score",
					tooltip="Score of the best frame, averaged over the colour "
					        "channels."),
				io.Int.Output(display_name="best_index",
					tooltip="Index of the best frame in the batch - argmax for "
					        "SSIM/PSNR, argmin for MSE/GMSD. Feed it to a list "
					        "picker to select the winning candidate."),
				io.Boolean.Output(display_name="higher_is_better",
					tooltip="True for SSIM/PSNR, False for MSE/GMSD. Wire it "
					        "into a comparison so a workflow can rank without "
					        "hardcoding the metric's direction."),
				NPArray.Output(display_name="scores",
					tooltip="float32 array of shape (N,) - one score per frame, "
					        "in batch order. Chart it with 'CV Chart Series'."),
				NPArray.Output(display_name="quality_maps",
					tooltip="Per-pixel quality/error maps, float32 (N,H,W,C) - "
					        "preview with 'Preview CV Array' "
					        "(normalize/heatmap) to see WHERE frames differ. "
					        "GMSD's map is HALF the input's resolution (the "
					        "metric works on a downsampled gradient map)."),
			],
		)

	@classmethod
	def execute(cls, reference, images, metric,
	            max_pixel_value=255.0) -> io.NodeOutput:
		quality = _quality("QualitySSIM_create")
		name, higher_better = METRICS[metric]
		ref = _bgr_frames(reference)[0]
		frames = _bgr_frames(images)
		for i, f in enumerate(frames):
			if f.shape != ref.shape:
				raise ValueError(
					f"frame {i} {f.shape} does not match the reference "
					f"{ref.shape} - resize before comparing.")
		scores, maps = [], []
		pbar = comfy.utils.ProgressBar(len(frames))
		if name == "QualityMSE":
			# instance compute() is an InputArrayOfArrays its binding cannot
			# take here (see module docstring) - the static call is equivalent
			for f in frames:
				score, qmap = quality.QualityMSE_compute(ref, f)
				scores.append(_channel_mean(score))
				maps.append(np.asarray(qmap, np.float32))
				pbar.update(1)
		else:
			factory = getattr(quality, name + "_create")
			obj = (factory(ref, float(max_pixel_value)) if name == "QualityPSNR"
			       else factory(ref))
			for f in frames:
				scores.append(_channel_mean(obj.compute(f)))
				maps.append(np.asarray(obj.getQualityMap(), np.float32))
				pbar.update(1)
			obj.clear()
		scores = np.asarray(scores, np.float32)
		best = int(np.argmax(scores) if higher_better else np.argmin(scores))
		# maps of a grayscale reference come back 2-D; keep the stack regular
		maps = [m if m.ndim == 3 else m[..., None] for m in maps]
		return io.NodeOutput(float(scores[best]), best, bool(higher_better),
		                     scores, np.stack(maps).astype(np.float32))


class QualityBRISQUE(io.ComfyNode):
	"""`QualityBRISQUE_compute`: no-reference quality score, per frame."""

	@classmethod
	def define_schema(cls):
		_register_quality_folder()
		files = _quality_files()
		return io.Schema(
			node_id="CV_Quality_BRISQUE",
			display_name="CV Quality BRISQUE (no-reference)",
			category=CATEGORY,
			description="Grades image quality with NO reference image "
			            "(cv2.quality.QualityBRISQUE): an SVM over 36 "
			            "natural-scene-statistics features, trained on the LIVE "
			            "database. Score runs 0 (best) to 100 (worst) - use it "
			            "to reject blurry/noisy/over-compressed frames when "
			            "there is no ground truth to compare against, which is "
			            "every real capture. Needs two data files in "
			            "ComfyUI/models/quality (brisque_model_live.yml and "
			            "brisque_range_live.yml from opencv_contrib, "
			            "modules/quality/samples). Scores the whole batch.",
			inputs=[
				polytype.poly_input("images", types=polytype.IMAGEISH,
					tooltip="Image(s) to grade. Every frame of a batch is "
					        "scored; frames may differ in size."),
				io.Combo.Input("model_file",
					options=files,
					default=("brisque_model_live.yml"
					         if "brisque_model_live.yml" in files
					         else (files[0] if files else "")),
					tooltip="The trained SVM (brisque_model_live.yml) from "
					        "ComfyUI/models/quality."),
				io.Combo.Input("range_file",
					options=files,
					default=("brisque_range_live.yml"
					         if "brisque_range_live.yml" in files
					         else (files[-1] if files else "")),
					tooltip="The feature-range file (brisque_range_live.yml) "
					        "that normalizes the features before the SVM. It "
					        "MUST be the pair of the model file above."),
			],
			outputs=[
				io.Float.Output(display_name="worst_score",
					tooltip="The highest (= WORST) BRISQUE score in the batch. "
					        "Compare it against a threshold to gate a "
					        "pipeline: LIVE-trained BRISQUE puts clean photos "
					        "roughly under 30 and heavy degradation over 60."),
				io.Int.Output(display_name="worst_index",
					tooltip="Index of the worst-scoring frame in the batch."),
				NPArray.Output(display_name="scores",
					tooltip="float32 array of shape (N,) - one score per frame, "
					        "in batch order. Chart it with 'CV Chart Series'."),
			],
		)

	@classmethod
	def execute(cls, images, model_file, range_file) -> io.NodeOutput:
		quality = _quality("QualityBRISQUE_compute")
		model_path = _quality_path(model_file)
		range_path = _quality_path(range_file)
		obj = quality.QualityBRISQUE_create(model_path, range_path)
		scores = np.asarray(
			[_channel_mean(np.asarray(obj.compute(f)).ravel()[:1])
			 for f in _bgr_frames(images)], np.float32)
		obj.clear()
		worst = int(np.argmax(scores))
		return io.NodeOutput(float(scores[worst]), worst, scores)


class QualityBRISQUEFeatures(io.ComfyNode):
	"""`QualityBRISQUE_computeFeatures`: the 36-d NSS descriptor, per frame."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Quality_BRISQUEFeatures",
			display_name="CV Quality BRISQUE Features",
			category=CATEGORY,
			description="The 36 natural-scene-statistics features BRISQUE "
			            "scores with (cv2.quality.QualityBRISQUE"
			            "_computeFeatures) - mean/variance/shape of locally "
			            "normalized luminance at two scales, plus its pairwise "
			            "products in four directions. Needs NO model file. As a "
			            "generic descriptor of how 'natural' an image's "
			            "statistics are, feed it to 'CV Train Classifier' "
			            "to learn your OWN quality or defect classifier instead "
			            "of accepting the LIVE-trained SVM's notion of quality.",
			inputs=[
				polytype.poly_input("images", types=polytype.IMAGEISH,
					tooltip="Image(s) to describe. One feature row per frame."),
			],
			outputs=[
				NPArray.Output(display_name="features",
					tooltip="float32 array of shape (N, 36) - one row per "
					        "frame, ready as a sample matrix for "
					        "'CV Train Classifier' / "
					        "'CV Stack Feature Classes'."),
				io.Int.Output(display_name="count",
					tooltip="Number of feature rows (= batch size)."),
			],
		)

	@classmethod
	def execute(cls, images) -> io.NodeOutput:
		quality = _quality("QualityBRISQUE_computeFeatures")
		rows = [np.asarray(quality.QualityBRISQUE_computeFeatures(f),
		                   np.float32).reshape(-1)
		        for f in _bgr_frames(images)]
		features = np.stack(rows).astype(np.float32)
		return io.NodeOutput(features, int(features.shape[0]))


class ImageQuality(io.ComfyNode):
	"""`Quality*_compute(ref, cmp)`: the static one-shot call, one pair.

	Moved here from `contrib.py` so one module owns all of cv2.quality. The
	node_id is FROZEN as `CV_ImageQuality`: it is the key saved in every
	workflow JSON (12, 56, 57, 60 ship with it), so renaming it would turn
	those into missing-node boxes. Only `display_name` and `category` changed -
	neither is serialized - which is what puts it in the same menu, and the
	same name family, as the class-API nodes above.

	Each cv2.quality metric is a separate class with an identical
	`Quality*_compute(ref, cmp) -> (score, map)` shape, so one node with a
	metric dropdown is both smaller and easier to compare across metrics.
	"""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ImageQuality",
			display_name="CV Quality Compare (cv2.quality)",
			category=CATEGORY,
			description="Scores how close ONE image is to a reference "
			            "(cv2.quality) - the objective way to compare two "
			            "settings of a filter, an upscaler or a denoiser "
			            "instead of eyeballing them. Outputs the overall score "
			            "and a per-pixel quality map showing WHERE the images "
			            "disagree. Read the direction carefully: for SSIM and "
			            "PSNR higher is better, for MSE and GMSD LOWER is "
			            "better ('higher_is_better' says which). Both images "
			            "must be the same size. To score a whole BATCH of "
			            "candidates against one reference, use 'CV Quality "
			            "Compare Batch' instead - it computes the reference's "
			            "state once and returns the winner's index.",
			inputs=[
				polytype.poly_input("reference", types=polytype.IMAGEISH,
					tooltip="The ground-truth / original image."),
				polytype.poly_input("image", types=polytype.IMAGEISH,
					tooltip="The processed image to score against the "
					        "reference. Must be the same size."),
				io.Combo.Input("metric",
					options=list(METRICS),
					default="SSIM (structural similarity)",
					tooltip="SSIM tracks perceived structural change (0-1, 1 = "
					        "identical) and is the usual choice. PSNR is a "
					        "log-scale error in dB (>40 is typically "
					        "indistinguishable). MSE is the raw squared error. "
					        "GMSD compares gradient structure and correlates "
					        "well with perceived sharpness loss."),
			],
			outputs=[
				io.Float.Output(display_name="score",
					tooltip="Overall score, averaged over the colour channels."),
				io.Boolean.Output(display_name="higher_is_better",
					tooltip="True for SSIM/PSNR, False for MSE/GMSD. Wire it "
					        "into a comparison node so a workflow can pick the "
					        "better of two candidates without hardcoding the "
					        "metric's direction."),
				NPArray.Output(display_name="quality_map",
					tooltip="Per-pixel quality/error map - preview it with "
					        "'Preview CV Array' (normalize/heatmap) to see "
					        "WHERE the two images differ. GMSD's map is HALF "
					        "the input's resolution."),
			],
		)

	@classmethod
	def execute(cls, reference, image, metric) -> io.NodeOutput:
		quality = _quality("QualitySSIM_compute")
		name, higher_better = METRICS[metric]
		ref = _bgr_frames(reference)[0]
		cmp = _bgr_frames(image)[0]
		if ref.shape != cmp.shape:
			raise ValueError(
				f"reference {ref.shape} and image {cmp.shape} must have the "
				"same size and channel count - resize one of them first.")
		score, qmap = getattr(quality, name + "_compute")(ref, cmp)
		return io.NodeOutput(_channel_mean(score), bool(higher_better),
		                     np.asarray(qmap, np.float32))


# 'CV Quality Compare' first: it is the one-pair entry point, and the batch
# node reads as its generalization.
NODES = [ImageQuality, QualityCompareBatch, QualityBRISQUE,
         QualityBRISQUEFeatures]
