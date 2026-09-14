# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Classical machine learning nodes (cv2.ml + feature extractors).

The pack's other DNN nodes only do INFERENCE; this module adds TRAINING.
The key design constraint (see workflow_proposals_v2.txt item 0): cv2.ml
models are stateful C++ handles that must never cross a graph edge, or
ComfyUI's output caching would double-train them. So 'CV Train
Classifier' is a PURE FUNCTION - train + evaluate + predict all happen
inside one execute() (features + labels + hyperparameters -> predictions +
accuracy + optional serialized model text), which makes it cache-safe by
construction.

Feature extractors feed it:
* 'CV HOG Features'  - hand-crafted gradient histograms (cv2.HOGDescriptor,
  stateless per call), one descriptor row per input frame.
* 'CV Deep Features' - a frozen ONNX backbone as a fixed feature extractor
  (cv2.dnn forward to an interior layer + global average pool). This is the
  OpenCV-idiomatic "freeze all layers, train a new head" transfer learning:
  cv2.dnn has NO backward pass, so the head is a cv2.ml model instead of a
  softmax. CRITICAL build gotcha baked in: OpenCV 5's default fused engine
  (ENGINE_NEW) breaks interior-layer taps - net.forward(name) either crashes
  or SILENTLY returns the final logits. The node pins ENGINE_CLASSIC, where
  taps work but ONNX layer names gain an 'onnx_node!' prefix (added
  automatically when the bare name is not found).

Plus small reusable data utilities: 'CV Random Points' (synthetic
Gaussian/uniform point clouds), 'CV Coordinate Grid' (a dense meshgrid of
pixel coordinates - predict over it to render decision REGIONS), and 'OpenCV
Stack Feature Classes' (concatenate per-class feature arrays and emit the
matching label vector).

Conventions: data only - visualize with 'Preview CV Array' /
'CV Draw Points' / the reshape+colormap chain; failure tolerant -
degenerate training data (empty, single class, an algorithm that cannot fit)
yields success=false and safe empty/zero outputs, never a raised exception;
only genuine wiring mistakes (mismatched lengths, invalid HOG geometry) raise.
"""

import gzip
import os

import cv2
import numpy as np
import torch

import comfy.utils
from comfy_api.latest import io, ui

from . import imageutils, polytype
from .convert import NPArray
from .dnn import _model_path, _onnx_models

CATEGORY = "image/CV/ml"

_EMPTY_POINTS = np.zeros((0, 1, 2), np.float32)


def _rows(value, name):
	"""(N, D) float32 view of a feature array; None/empty -> (0, 0)."""
	if value is None:
		return np.zeros((0, 0), np.float32)
	arr = np.asarray(value, np.float32)
	if arr.size == 0:
		return np.zeros((len(arr) if arr.ndim else 0, 0), np.float32)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 1)
	return np.ascontiguousarray(arr.reshape(len(arr), -1))


class RandomPoints(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_RandomPoints",
			display_name="CV Random Points",
			category=CATEGORY,
			description="Synthesizes a random 2-D point cloud around a center - "
			            "a Gaussian blob (or a uniform box) of count points. "
			            "Deterministic per seed. Use several of these as "
			            "synthetic classes for 'CV Stack Feature Classes' + "
			            "'CV Train Classifier' (the classic decision-boundary "
			            "playground), as k-means test data, or to scatter markers "
			            "for 'CV Draw Points'. count = 0 is a valid output "
			            "(empty array).",
			inputs=[
				io.Int.Input("count", default=60, min=0, max=100000,
					tooltip="How many points to draw. 0 is valid and yields an "
					        "empty point array (for testing the degenerate path)."),
				io.Combo.Input("distribution",
					options=["gaussian (normal)", "uniform (center +/- spread)"],
					default="gaussian (normal)",
					tooltip="gaussian: normal distribution, spread = standard "
					        "deviation per axis. uniform: even fill of the box "
					        "center +/- spread."),
				io.Float.Input("center_x", default=100.0, min=-1e6, max=1e6,
					tooltip="X coordinate of the cloud center (pixels)."),
				io.Float.Input("center_y", default=100.0, min=-1e6, max=1e6,
					tooltip="Y coordinate of the cloud center (pixels)."),
				io.Float.Input("spread_x", default=12.0, min=0.0, max=1e6,
					tooltip="Horizontal spread: std for gaussian, half-extent "
					        "for uniform (pixels)."),
				io.Float.Input("spread_y", default=12.0, min=0.0, max=1e6,
					tooltip="Vertical spread: std for gaussian, half-extent for "
					        "uniform (pixels)."),
				io.Int.Input("seed", default=0, min=0, max=2**31 - 1,
					tooltip="Random seed - the same seed always yields the same "
					        "points. Use different seeds for different classes."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="Nx1x2 float32 (x, y) points - feeds 'CV Draw "
					        "Points', 'CV Stack Feature Classes' and every "
					        "other point node."),
				io.Int.Output(display_name="count"),
			],
		)

	@classmethod
	def execute(cls, count, distribution, center_x, center_y, spread_x, spread_y,
	            seed) -> io.NodeOutput:
		n = int(count)
		if n == 0:
			return io.NodeOutput(_EMPTY_POINTS.copy(), 0)
		rng = np.random.default_rng(int(seed))
		if distribution.startswith("gaussian"):
			pts = rng.normal([center_x, center_y], [spread_x, spread_y], (n, 2))
		else:
			pts = rng.uniform([center_x - spread_x, center_y - spread_y],
			                  [center_x + spread_x, center_y + spread_y], (n, 2))
		return io.NodeOutput(np.float32(pts).reshape(-1, 1, 2), n)


class CoordinateGrid(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CoordinateGrid",
			display_name="CV Coordinate Grid",
			category=CATEGORY,
			description="A dense meshgrid of pixel coordinates: every (x, y) on "
			            "a step-spaced grid over a width x height canvas, row-"
			            "major (y outer, x inner). Feed it into 'CV Train "
			            "Classifier' query_features to classify EVERY grid "
			            "location, then 'CV Reshape Array' the predictions back "
			            "to (rows, cols) and colormap them - that renders the "
			            "decision REGIONS of the classifier. Also a generic "
			            "sample-position generator for remap/lookup experiments.",
			inputs=[
				io.Int.Input("width", default=224, min=1, max=8192,
					tooltip="Canvas width in pixels the grid covers (x goes "
					        "0, step, 2*step, ... below width)."),
				io.Int.Input("height", default=224, min=1, max=8192,
					tooltip="Canvas height in pixels the grid covers (y goes "
					        "0, step, 2*step, ... below height)."),
				io.Int.Input("step", default=2, min=1, max=1024,
					tooltip="Grid spacing in pixels. 1 = every pixel (slow to "
					        "classify); 2-4 is plenty for a preview."),
			],
			outputs=[
				NPArray.Output(display_name="points",
					tooltip="Nx1x2 float32 (x, y) grid coordinates, row-major: "
					        "N = rows * cols. Feeds query_features of 'OpenCV "
					        "Train Classifier'."),
				io.Int.Output(display_name="cols",
					tooltip="Grid columns = number of x positions. Use as the "
					        "cols of 'CV Reshape Array' on the predictions."),
				io.Int.Output(display_name="rows",
					tooltip="Grid rows = number of y positions. Use as the rows "
					        "of 'CV Reshape Array' on the predictions."),
				io.Int.Output(display_name="count", tooltip="rows * cols."),
			],
		)

	@classmethod
	def execute(cls, width, height, step) -> io.NodeOutput:
		xs = np.arange(0, int(width), int(step), dtype=np.float32)
		ys = np.arange(0, int(height), int(step), dtype=np.float32)
		gx, gy = np.meshgrid(xs, ys)
		pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
		return io.NodeOutput(pts.reshape(-1, 1, 2), len(xs), len(ys), len(pts))


class StackFeatureClasses(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_StackFeatureClasses",
			display_name="CV Stack Feature Classes",
			category=CATEGORY,
			description="Builds a labeled training set from per-class feature "
			            "arrays: the rows of class_a get label 0, class_b label "
			            "1, and so on. Each input is flattened to one row per "
			            "sample (points, HOG rows and deep embeddings all work); "
			            "every connected input must yield the same feature "
			            "length. Feeds 'CV Train Classifier' directly; the "
			            "labels also color 'CV Draw Points'. Empty classes "
			            "are valid (they contribute no rows); leave class_c/d "
			            "unconnected when you have fewer classes.",
			inputs=[
				NPArray.Input("class_a",
					tooltip="Feature array of class 0: N samples, any shape - "
					        "flattened to one row per sample."),
				NPArray.Input("class_b",
					tooltip="Feature array of class 1 (same feature length as "
					        "class_a)."),
				NPArray.Input("class_c", optional=True,
					tooltip="Feature array of class 2 (optional)."),
				NPArray.Input("class_d", optional=True,
					tooltip="Feature array of class 3 (optional)."),
			],
			outputs=[
				NPArray.Output(display_name="features",
					tooltip="(N, D) float32: all class rows concatenated in "
					        "class order."),
				NPArray.Output(display_name="labels",
					tooltip="(N,) int32: 0 for class_a rows, 1 for class_b, ... "
					        "aligned with features."),
				io.Int.Output(display_name="count", tooltip="Total sample count N."),
				io.Int.Output(display_name="class_count",
					tooltip="How many class inputs are connected."),
			],
		)

	@classmethod
	def execute(cls, class_a, class_b, class_c=None, class_d=None) -> io.NodeOutput:
		connected = [(k, v) for k, v in (("class_a", class_a), ("class_b", class_b),
		                                 ("class_c", class_c), ("class_d", class_d))
		             if v is not None]
		rows, labels, dim = [], [], None
		for idx, (name, value) in enumerate(connected):
			r = _rows(value, name)
			if len(r) == 0:
				continue
			if dim is None:
				dim = r.shape[1]
			elif r.shape[1] != dim:
				raise ValueError(
					f"{name} has feature length {r.shape[1]} but earlier classes "
					f"have {dim} - every class must use the same features."
				)
			rows.append(r)
			labels.append(np.full(len(r), idx, np.int32))
		if not rows:
			return io.NodeOutput(np.zeros((0, 0), np.float32),
			                     np.zeros((0,), np.int32), 0, len(connected))
		return io.NodeOutput(np.concatenate(rows, axis=0),
		                     np.concatenate(labels), int(sum(map(len, rows))),
		                     len(connected))


class HOGFeatures(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_HOGFeatures",
			display_name="CV HOG Features",
			category=CATEGORY,
			description="Histogram-of-Oriented-Gradients descriptor per frame "
			            "(cv2.HOGDescriptor): each input frame is grayscaled, "
			            "resized to win_size x win_size and reduced to one "
			            "fixed-length gradient-orientation feature row - the "
			            "classic hand-crafted descriptor for training a shape/"
			            "digit classifier ('CV Train Classifier'). The "
			            "defaults (win 20, cell 10, block 10, stride 5, 9 bins) "
			            "give the 81-dim descriptor of OpenCV's digits.py "
			            "tutorial. Geometry rules (raised as errors, they are "
			            "configuration mistakes): win_size and block_size must "
			            "be multiples of cell_size, and (win_size - block_size) "
			            "must be a multiple of block_stride. An empty IMAGE "
			            "batch yields an empty (0, D) array.",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Frames to describe: an IMAGE batch gives one "
					        "feature row per frame; an NPARRAY (gray/BGR/BGRA) "
					        "is a single frame."),
				io.Int.Input("win_size", default=20, min=8, max=512,
					tooltip="Square window size in px - every frame is resized "
					        "to this before computing. Bigger keeps more detail "
					        "but grows the descriptor."),
				io.Int.Input("cell_size", default=10, min=2, max=256,
					tooltip="Histogram cell size in px. win_size and block_size "
					        "must be multiples of it."),
				io.Int.Input("block_size", default=10, min=2, max=256,
					tooltip="Normalization block size in px (a multiple of "
					        "cell_size, at most win_size)."),
				io.Int.Input("block_stride", default=5, min=1, max=256,
					tooltip="Block step in px; (win_size - block_size) must be "
					        "a multiple of it."),
				io.Int.Input("nbins", default=9, min=2, max=32,
					tooltip="Orientation bins per histogram (9 is the standard)."),
			],
			outputs=[
				NPArray.Output(display_name="features",
					tooltip="(N, D) float32, one HOG row per frame - feeds "
					        "'CV Train Classifier' or 'CV Stack Feature "
					        "Classes'."),
				io.Int.Output(display_name="dims",
					tooltip="Descriptor length D (81 with the defaults)."),
				io.Int.Output(display_name="count", tooltip="Frames described."),
			],
		)

	@classmethod
	def execute(cls, image, win_size, cell_size, block_size, block_stride,
	            nbins) -> io.NodeOutput:
		win, cell, block, stride = (int(win_size), int(cell_size),
		                            int(block_size), int(block_stride))
		if block % cell or win % cell:
			raise ValueError(
				f"win_size ({win}) and block_size ({block}) must be multiples "
				f"of cell_size ({cell})."
			)
		if block > win or (win - block) % stride:
			raise ValueError(
				f"(win_size - block_size) = {win - block} must be a non-negative "
				f"multiple of block_stride ({stride})."
			)
		hog = cv2.HOGDescriptor((win, win), (block, block), (stride, stride),
		                        (cell, cell), int(nbins))
		dims = int(hog.getDescriptorSize())
		rows = []
		for frame in imageutils.frames_from_input(image):
			gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			        if frame.ndim == 3 else frame)
			gray = cv2.resize(gray, (win, win), interpolation=cv2.INTER_AREA)
			rows.append(hog.compute(gray).ravel())
		features = (np.stack(rows).astype(np.float32) if rows
		            else np.zeros((0, dims), np.float32))
		return io.NodeOutput(features, dims, len(rows))


# --- frozen-backbone deep features ------------------------------------------------

_deep_cache: dict = {}


def _classic_net(path):
	"""readNetFromONNX pinned to ENGINE_CLASSIC where the build still has it.

	Returns (net, pinned). OpenCV 5's fused graph engine breaks interior-layer
	taps: net.forward(inner_name) either hits a bufidx assertion or SILENTLY
	returns the final output instead of the requested layer. ENGINE_CLASSIC
	taps work, at the cost of layer names gaining an 'onnx_node!' prefix.

	OpenCV 5.1 REMOVED ENGINE_CLASSIC - the surviving ENGINE_OPENCV is that same
	graph engine - so on such a build `pinned` is False and there is no engine
	left that can tap an interior layer. Falling back silently would hand back
	classifier logits dressed as a bottleneck embedding, so the caller verifies
	the tap instead (see _tap_is_real)."""
	hit = _deep_cache.get(path)
	if hit is None:
		engine = getattr(cv2.dnn, "ENGINE_CLASSIC", None)
		net = (cv2.dnn.readNetFromONNX(path, engine) if engine is not None
		       else cv2.dnn.readNetFromONNX(path))
		hit = (net, engine is not None)
		_deep_cache[path] = hit
	return hit


def _tap_is_real(net, blob, name):
	"""True when forward(name) differs from the net's final output.

	The graph engine answers a named interior tap with the FINAL output and
	raises nothing, so only comparing the VALUES can tell a working tap from an
	ignored one. Shape is not enough and is actively misleading: the bare
	forward() returns squeezenet's logits as (1, 1000) while forward(name) hands
	back the very same numbers as (1, 1000, 1, 1). Compared flattened for that
	reason. Run once per execute, not per frame."""
	net.setInput(np.ascontiguousarray(blob))
	tapped = np.asarray(net.forward(name), np.float32).ravel()
	net.setInput(np.ascontiguousarray(blob))
	final = np.asarray(net.forward(), np.float32).ravel()
	return tapped.size != final.size or not np.array_equal(tapped, final)


class DeepFeatures(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		models = _onnx_models()
		return io.Schema(
			node_id="CV_DeepFeatures",
			display_name="CV Deep Features",
			category=CATEGORY,
			description="A frozen ONNX backbone as a fixed feature extractor: "
			            "each frame is forwarded to layer_name (an interior "
			            "bottleneck, NOT the classifier logits) and the "
			            "activation map is global-average-pooled into one "
			            "embedding row. This is transfer learning the OpenCV "
			            "way - cv2.dnn cannot backpropagate, so the backbone "
			            "stays frozen and a cv2.ml head ('CV Train "
			            "Classifier') is trained on the embeddings instead. "
			            "E.g. SqueezeNet 1.1: layer squeezenet0_concat7 -> "
			            "512-dim embeddings. The net is pinned to OpenCV's "
			            "CLASSIC engine where the build still has one (the fused "
			            "graph engine mis-handles interior taps); classic layer "
			            "names carry an 'onnx_node!' prefix which is added "
			            "automatically when the bare name is not found. A wrong "
			            "layer name or an empty batch yields success=false and an "
			            "empty (0, 0) array, never an error - but a build with NO "
			            "classic engine (OpenCV 5.1 removed it) cannot tap an "
			            "interior layer at all, and that RAISES rather than "
			            "quietly returning the classifier logits.",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Frames to embed: an IMAGE batch gives one embedding "
					        "row per frame; an NPARRAY (gray/BGR/BGRA) is a "
					        "single frame."),
				io.Combo.Input("model", options=models,
					tooltip="Backbone .onnx from ComfyUI/models/onnx (e.g. "
					        "squeezenet1.1-7.onnx)."),
				io.String.Input("layer_name", default="",
					tooltip="Layer to tap, e.g. squeezenet0_concat7 (the "
					        "'onnx_node!' classic-engine prefix is added "
					        "automatically if needed). Blank = the model's final "
					        "output - usually class LOGITS, which transfer worse "
					        "than an interior bottleneck."),
				io.Int.Input("input_size", default=224, min=16, max=2048,
					tooltip="Square net input in px (224 for the ImageNet "
					        "backbones); frames are resized by blobFromImage."),
				io.Float.Input("scale", default=0.00392156862745098, min=0.0,
					max=1e6, step=0.0001,
					tooltip="Pixel scale factor applied after mean subtraction; "
					        "1/255 = 0.0039215... maps 0-255 to the 0-1 range "
					        "ImageNet models expect."),
				io.Boolean.Input("swap_rb", default=True,
					tooltip="Swap R and B: OpenCV frames are BGR, most ONNX "
					        "backbones want RGB - leave on."),
				io.Combo.Input("pooling",
					options=["global average pool (embedding)",
					         "flatten (raw activations)"],
					default="global average pool (embedding)",
					tooltip="How a (C, H, W) activation map becomes one row: "
					        "average each channel over space (C dims, position-"
					        "invariant - the standard embedding), or flatten "
					        "everything (C*H*W dims, keeps layout)."),
				io.Float.Input("mean_r", default=123.675, min=-255.0, max=255.0,
					step=0.001, optional=True, advanced=True,
					tooltip="RED channel mean subtracted before scaling. Default "
					        "= ImageNet 0.485 * 255. Set 0 to disable."),
				io.Float.Input("mean_g", default=116.28, min=-255.0, max=255.0,
					step=0.001, optional=True, advanced=True,
					tooltip="GREEN channel mean (ImageNet 0.456 * 255)."),
				io.Float.Input("mean_b", default=103.53, min=-255.0, max=255.0,
					step=0.001, optional=True, advanced=True,
					tooltip="BLUE channel mean (ImageNet 0.406 * 255)."),
				io.Float.Input("std_r", default=0.229, min=1e-6, max=255.0,
					step=0.001, optional=True, advanced=True,
					tooltip="RED channel std the scaled pixels are divided by "
					        "(ImageNet 0.229). Set 1 to disable."),
				io.Float.Input("std_g", default=0.224, min=1e-6, max=255.0,
					step=0.001, optional=True, advanced=True,
					tooltip="GREEN channel std (ImageNet 0.224). 1 disables."),
				io.Float.Input("std_b", default=0.225, min=1e-6, max=255.0,
					step=0.001, optional=True, advanced=True,
					tooltip="BLUE channel std (ImageNet 0.225). 1 disables."),
			],
			outputs=[
				NPArray.Output(display_name="features",
					tooltip="(N, D) float32 embeddings, one row per frame - "
					        "feeds 'CV Train Classifier' / 'CV Stack "
					        "Feature Classes'. Empty (0, 0) on failure."),
				io.Int.Output(display_name="dims",
					tooltip="Embedding length D (512 for SqueezeNet's "
					        "squeezenet0_concat7)."),
				io.Int.Output(display_name="count", tooltip="Frames embedded."),
				io.Boolean.Output(display_name="success",
					tooltip="False when the layer name does not exist in the "
					        "model or no frame could be embedded - gate "
					        "downstream training with if/else."),
			],
		)

	@classmethod
	def execute(cls, image, model, layer_name, input_size, scale, swap_rb,
	            pooling, mean_r=123.675, mean_g=116.28, mean_b=103.53,
	            std_r=0.229, std_g=0.224, std_b=0.225) -> io.NodeOutput:
		empty = np.zeros((0, 0), np.float32)
		path = _model_path(model, "Deep-features backbone")
		net, pinned = _classic_net(path)
		name = str(layer_name).strip()
		if name:
			layers = set(net.getLayerNames())
			if name not in layers:
				prefixed = "onnx_node!" + name
				if prefixed in layers:
					name = prefixed
				else:
					return io.NodeOutput(empty, 0, 0, False)
		size = int(input_size)
		std = np.array([std_r, std_g, std_b], np.float32).reshape(1, 3, 1, 1)
		if not swap_rb:
			std = std[:, ::-1]
		rows = []
		checked = False
		frames = imageutils.frames_from_input(image)
		pbar = comfy.utils.ProgressBar(len(frames))
		try:
			for frame in frames:
				if frame.ndim == 2:
					frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
				blob = cv2.dnn.blobFromImage(
					frame, float(scale), (size, size),
					(float(mean_r), float(mean_g), float(mean_b)),
					swapRB=bool(swap_rb), crop=False)
				if blob.shape[1] == 3:
					blob = blob / std
				if name and not pinned and not checked:
					checked = True
					if not _tap_is_real(net, blob, name):
						raise RuntimeError(
							f"'CV Deep Features': layer {name!r} of "
							f"{os.path.basename(path)} cannot be tapped on this "
							f"OpenCV build (cv2 {cv2.__version__}). The engine "
							f"returned the model's FINAL output - the classifier "
							f"logits - instead of the requested activation, "
							f"without raising. OpenCV 5.1 removed ENGINE_CLASSIC, "
							f"the only engine whose forward(layer) honours an "
							f"interior name, so a frozen-backbone embedding is "
							f"not available here. Leave layer_name blank to use "
							f"the final output deliberately, or run a build that "
							f"still has the classic engine.")
				net.setInput(np.ascontiguousarray(blob))
				out = np.asarray(net.forward(name) if name else net.forward(),
				                 np.float32)
				if out.ndim >= 3 and pooling.startswith("global average"):
					# (1, C, H, W) or (1, C, ...) -> mean over the spatial axes
					out = out.reshape(out.shape[0], out.shape[1], -1).mean(axis=2)
				rows.append(out.reshape(-1))
				pbar.update(1)
		except cv2.error:
			return io.NodeOutput(empty, 0, 0, False)
		if not rows:
			return io.NodeOutput(empty, 0, 0, False)
		features = np.stack(rows).astype(np.float32)
		return io.NodeOutput(features, int(features.shape[1]), len(rows), True)


# --- the trainer -------------------------------------------------------------------

# cv2.ml algorithms as NAMED dropdown options (no magic ints). Only the
# factories present in the loaded build are offered.
_ALGO_SPEC = [
	("SVM (support vector machine)", "svm", "SVM_create"),
	("KNearest (k-nearest neighbors)", "knn", "KNearest_create"),
	("RTrees (random forest)", "rtrees", "RTrees_create"),
	("DTrees (single decision tree)", "dtrees", "DTrees_create"),
	("Boost (boosted trees, 2 classes only)", "boost", "Boost_create"),
	("NormalBayes (gaussian bayes)", "bayes", "NormalBayesClassifier_create"),
	("LogisticRegression", "logreg", "LogisticRegression_create"),
	("ANN_MLP (small neural net, backprop)", "mlp", "ANN_MLP_create"),
]
_ALGORITHMS = {label: (kind, factory) for label, kind, factory in _ALGO_SPEC
               if hasattr(cv2.ml, factory)}

# SVM kernels as named options (cv2.ml.SVM_* ints stay hidden).
_SVM_KERNELS = {label: getattr(cv2.ml, const) for label, const in [
	("RBF (gaussian)", "SVM_RBF"),
	("linear", "SVM_LINEAR"),
	("polynomial (degree 3)", "SVM_POLY"),
	("sigmoid", "SVM_SIGMOID"),
	("chi-square", "SVM_CHI2"),
	("histogram intersection", "SVM_INTER"),
] if hasattr(cv2.ml, const)}

_SVM_MANUAL = "manual (use C and gamma)"
_SVM_AUTO = "trainAuto (cross-validated grid search)"


class TrainClassifier(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_TrainClassifier",
			display_name="CV Train Classifier",
			category=CATEGORY,
			description="Trains a cv2.ml classifier and evaluates it, all inside "
			            "one node: the features are split into a train and a "
			            "held-out test set (stratified per class, deterministic "
			            "per seed), the picked algorithm is fitted on the train "
			            "split and scored on the test split (accuracy + "
			            "confusion matrix). The model never leaves the node - "
			            "train, evaluate and the optional query prediction all "
			            "happen in one call, so caching stays correct. Connect "
			            "query_features (e.g. an 'CV Coordinate Grid') to "
			            "also classify arbitrary extra samples with the freshly "
			            "trained model - reshape + colormap those predictions to "
			            "render decision regions. Degenerate data (no samples, a "
			            "single class, an algorithm that cannot fit - e.g. Boost "
			            "with 3 classes) yields success=false with safe empty "
			            "outputs and query predictions of the first class, never "
			            "an error. model_yaml carries the serialized model for "
			            "reuse outside ComfyUI (cv2.ml.*_load).",
			inputs=[
				NPArray.Input("features",
					tooltip="(N, D) training samples, one row per sample (any "
					        "shape flattens to rows): stacked points, HOG rows, "
					        "deep embeddings..."),
				NPArray.Input("labels",
					tooltip="(N,) integer class label per row - e.g. the labels "
					        "of 'CV Stack Feature Classes'."),
				NPArray.Input("query_features", optional=True,
					tooltip="Optional extra samples to classify with the trained "
					        "model (same feature length D), e.g. an 'OpenCV "
					        "Coordinate Grid' to paint decision regions."),
				io.Combo.Input("algorithm", options=list(_ALGORITHMS),
					default=next(iter(_ALGORITHMS)) if _ALGORITHMS else "",
					tooltip="cv2.ml algorithm to train. SVM is the strong "
					        "default; KNearest is instance-based (no real "
					        "training); RTrees/DTrees/Boost are tree-based "
					        "(Boost handles exactly 2 classes); NormalBayes "
					        "fits gaussians; LogisticRegression is linear; "
					        "ANN_MLP is a small fully-connected net trained by "
					        "real backpropagation (from scratch - cv2 cannot "
					        "finetune pretrained deep nets)."),
				io.Float.Input("test_fraction", default=0.3, min=0.0, max=0.9,
					step=0.05,
					tooltip="Held-out fraction per class for the accuracy / "
					        "confusion outputs. 0 evaluates on the training set "
					        "itself (optimistic - fine for decision-region "
					        "demos)."),
				io.Combo.Input("svm_kernel", options=list(_SVM_KERNELS),
					default=next(iter(_SVM_KERNELS)) if _SVM_KERNELS else "",
					optional=True, advanced=True,
					tooltip="SVM only: the kernel. RBF handles curved "
					        "boundaries; linear is fastest and best when D is "
					        "large (e.g. HOG/deep features)."),
				io.Combo.Input("svm_tuning", options=[_SVM_MANUAL, _SVM_AUTO],
					default=_SVM_MANUAL, optional=True, advanced=True,
					tooltip="SVM only: manual uses the C / gamma below; "
					        "trainAuto cross-validates a grid over them "
					        "(slower, robust default for real features)."),
				io.Float.Input("svm_c", default=1.0, min=1e-6, max=1e6,
					optional=True, advanced=True,
					tooltip="SVM only (manual): soft-margin penalty C. Higher "
					        "fits the training set tighter (risk of "
					        "overfitting)."),
				io.Float.Input("svm_gamma", default=1.0, min=1e-9, max=1e6,
					optional=True, advanced=True,
					tooltip="SVM only (manual, RBF/poly/sigmoid/chi2): kernel "
					        "width. Higher = wigglier boundary. For pixel "
					        "coordinates try 0.001-0.01; for normalized "
					        "features 0.1-10."),
				io.Int.Input("knn_k", default=5, min=1, max=256,
					optional=True, advanced=True,
					tooltip="KNearest only: how many neighbors vote."),
				io.Int.Input("tree_depth", default=8, min=1, max=64,
					optional=True, advanced=True,
					tooltip="RTrees/DTrees/Boost only: maximum tree depth."),
				io.Int.Input("forest_size", default=64, min=1, max=2048,
					optional=True, advanced=True,
					tooltip="RTrees: number of trees; Boost: number of weak "
					        "learners."),
				io.Int.Input("mlp_hidden", default=32, min=1, max=4096,
					optional=True, advanced=True,
					tooltip="ANN_MLP only: neurons in the single hidden layer."),
				io.Int.Input("mlp_iterations", default=300, min=1, max=100000,
					optional=True, advanced=True,
					tooltip="ANN_MLP: RPROP epochs; LogisticRegression: batch "
					        "gradient iterations."),
				# 'seed' must be the LAST widget: the frontend auto-appends a
				# control_after_generate companion right after any widget named
				# seed - anywhere else it would displace every later saved
				# widget value (same convention as the K-Means nodes)
				io.Int.Input("seed", default=0, min=0, max=2**31 - 1,
					optional=True, advanced=True,
					tooltip="Seed for the stratified shuffle split (and cv2's "
					        "RNG) - same seed, same split, same model."),
			],
			outputs=[
				NPArray.Output(display_name="predictions",
					tooltip="(M,) int32 predicted label of every EVAL sample "
					        "(the held-out split, or the train set when "
					        "test_fraction = 0). Aligned with eval_labels."),
				NPArray.Output(display_name="eval_labels",
					tooltip="(M,) int32 ground-truth label of every eval sample "
					        "- compare against predictions."),
				io.Float.Output(display_name="accuracy",
					tooltip="Fraction of eval samples classified correctly "
					        "(chance = 1 / class count)."),
				io.Float.Output(display_name="train_accuracy",
					tooltip="Accuracy on the training split itself - much "
					        "higher than 'accuracy' means overfitting."),
				NPArray.Output(display_name="confusion",
					tooltip="(K, K) int32 confusion matrix over the eval set: "
					        "row = true class, column = predicted. Upscale with "
					        "cv2.resize (INTER_NEAREST) + 'Preview CV Array' "
					        "(heatmap) to see it."),
				io.Boolean.Output(display_name="success",
					tooltip="False when training was impossible (no samples, "
					        "one class, algorithm/data mismatch) - gate "
					        "downstream consumers with if/else."),
				NPArray.Output(display_name="query_predictions",
					tooltip="(Q,) int32 predicted label per query_features row "
					        "(empty when unconnected; all first-class when "
					        "success=false, so reshapes stay valid)."),
				io.String.Output(display_name="model_yaml",
					tooltip="The trained model serialized as OpenCV YAML "
					        "(cv2.ml.SVM_load & co. read it back outside "
					        "ComfyUI). Empty when unavailable."),
			],
		)

	# --- helpers -----------------------------------------------------------------

	@staticmethod
	def _split(y_idx, test_fraction, rng):
		"""Stratified train/eval indices; every class keeps >= 1 train sample."""
		train, test = [], []
		for c in range(int(y_idx.max()) + 1 if len(y_idx) else 0):
			members = np.flatnonzero(y_idx == c)
			rng.shuffle(members)
			n_test = min(int(round(test_fraction * len(members))), len(members) - 1)
			n_test = max(n_test, 0)
			test.extend(members[:n_test])
			train.extend(members[n_test:])
		return np.array(sorted(train), np.int64), np.array(sorted(test), np.int64)

	@classmethod
	def _build(cls, kind, n_dims, n_classes, p):
		if kind == "svm":
			m = cv2.ml.SVM_create()
			m.setType(cv2.ml.SVM_C_SVC)
			m.setKernel(int(_SVM_KERNELS[p["svm_kernel"]]))
			m.setC(float(p["svm_c"]))
			m.setGamma(float(p["svm_gamma"]))
			m.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
			                   1000, 1e-6))
		elif kind == "knn":
			m = cv2.ml.KNearest_create()
			m.setIsClassifier(True)
			m.setDefaultK(int(p["knn_k"]))
		elif kind == "rtrees":
			m = cv2.ml.RTrees_create()
			m.setMaxDepth(int(p["tree_depth"]))
			m.setTermCriteria((cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS,
			                   int(p["forest_size"]), 1e-3))
		elif kind == "dtrees":
			m = cv2.ml.DTrees_create()
			m.setMaxDepth(int(p["tree_depth"]))
			m.setCVFolds(0)  # pruning folds > 0 are broken in the python bindings
			m.setMinSampleCount(2)
		elif kind == "boost":
			m = cv2.ml.Boost_create()
			m.setMaxDepth(int(p["tree_depth"]))
			m.setWeakCount(int(p["forest_size"]))
		elif kind == "bayes":
			m = cv2.ml.NormalBayesClassifier_create()
		elif kind == "logreg":
			m = cv2.ml.LogisticRegression_create()
			m.setLearningRate(0.05)
			m.setIterations(max(100, int(p["mlp_iterations"])))
			m.setRegularization(cv2.ml.LOGISTIC_REGRESSION_REG_L2)
			m.setTrainMethod(cv2.ml.LOGISTIC_REGRESSION_BATCH)
		else:  # mlp
			m = cv2.ml.ANN_MLP_create()
			m.setLayerSizes(np.int32([n_dims, int(p["mlp_hidden"]), n_classes]))
			m.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
			m.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
			m.setTermCriteria((cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS,
			                   int(p["mlp_iterations"]), 1e-4))
		return m

	@staticmethod
	def _train(model, kind, X, y_idx, n_classes, p):
		if kind == "logreg":
			# LogisticRegression insists on a COLUMN matrix of float labels
			model.train(X, cv2.ml.ROW_SAMPLE,
			            y_idx.astype(np.float32).reshape(-1, 1))
		elif kind == "mlp":
			onehot = np.full((len(y_idx), n_classes), -1.0, np.float32)
			onehot[np.arange(len(y_idx)), y_idx] = 1.0
			model.train(X, cv2.ml.ROW_SAMPLE, onehot)
		elif kind == "svm" and p["svm_tuning"] == _SVM_AUTO:
			counts = np.bincount(y_idx, minlength=n_classes)
			k_fold = int(np.clip(counts[counts > 0].min(), 2, 10))
			model.trainAuto(X, cv2.ml.ROW_SAMPLE, y_idx.astype(np.int32),
			                kFold=k_fold)
		else:
			model.train(X, cv2.ml.ROW_SAMPLE, y_idx.astype(np.int32))

	@staticmethod
	def _predict_idx(model, kind, X, n_classes, p):
		if len(X) == 0:
			return np.zeros((0,), np.int32)
		if kind == "knn":
			results = model.findNearest(X, int(p["knn_k"]))[1]
		elif kind == "mlp":
			return model.predict(X)[1].argmax(axis=1).astype(np.int32)
		else:
			results = model.predict(X)[1]
		idx = np.rint(np.asarray(results).ravel()).astype(np.int32)
		return np.clip(idx, 0, n_classes - 1)

	@staticmethod
	def _serialize(model):
		# Wrap the model in its default-name top node (opencv_ml_svm, ...):
		# that is the layout StatModel::save produces, so the text loads back
		# with cv2.ml.SVM_load & co. and with _load_model below. A bare
		# model.write(fs) would drop the wrapper and the loaders would fail.
		try:
			fs = cv2.FileStorage(
				"model.yml", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
			fs.startWriteStruct(model.getDefaultName(), cv2.FILE_NODE_MAP)
			model.write(fs)
			fs.endWriteStruct()
			return fs.releaseAndGetString()
		except cv2.error:
			return ""

	@classmethod
	def execute(cls, features, labels, algorithm, test_fraction,
	            query_features=None, svm_kernel=None, svm_tuning=_SVM_MANUAL,
	            svm_c=1.0, svm_gamma=1.0, knn_k=5, tree_depth=8, forest_size=64,
	            mlp_hidden=32, mlp_iterations=300, seed=0) -> io.NodeOutput:
		X = _rows(features, "features")
		y = (np.zeros((0,), np.int64) if labels is None
		     else np.asarray(labels).ravel().astype(np.int64))
		if len(X) != len(y):
			raise ValueError(
				f"features has {len(X)} rows but labels has {len(y)} entries - "
				"both must describe the same samples."
			)
		q = _rows(query_features, "query_features")
		classes = np.unique(y)
		n_classes = len(classes)
		fallback_class = int(classes[0]) if n_classes else 0
		empty_i = np.zeros((0,), np.int32)

		def fallback():
			return io.NodeOutput(
				empty_i.copy(), empty_i.copy(), 0.0, 0.0,
				np.zeros((0, 0), np.int32), False,
				np.full((len(q),), fallback_class, np.int32), "")

		if len(X) == 0 or n_classes < 2 or X.shape[1] == 0:
			return fallback()
		if q.size and q.shape[1] != X.shape[1]:
			raise ValueError(
				f"query_features has feature length {q.shape[1]} but the "
				f"training features have {X.shape[1]} - they must match."
			)

		y_idx = np.searchsorted(classes, y).astype(np.int32)
		rng = np.random.default_rng(int(seed))
		cv2.setRNGSeed(int(seed))
		train_i, test_i = cls._split(y_idx, float(test_fraction), rng)
		eval_i = test_i if len(test_i) else train_i
		params = {"svm_kernel": svm_kernel or next(iter(_SVM_KERNELS)),
		          "svm_tuning": svm_tuning, "svm_c": svm_c, "svm_gamma": svm_gamma,
		          "knn_k": knn_k, "tree_depth": tree_depth,
		          "forest_size": forest_size, "mlp_hidden": mlp_hidden,
		          "mlp_iterations": mlp_iterations}
		kind = _ALGORITHMS[algorithm][0]
		if len(np.unique(y_idx[train_i])) < 2:
			return fallback()
		if kind == "boost" and n_classes > 2:
			# cv2 Boost is a two-class learner; on more classes it silently
			# degrades into nonsense instead of raising - fail deterministically
			return fallback()
		try:
			model = cls._build(kind, X.shape[1], n_classes, params)
			cls._train(model, kind, X[train_i], y_idx[train_i], n_classes, params)
			pred_eval = cls._predict_idx(model, kind, X[eval_i], n_classes, params)
			pred_train = cls._predict_idx(model, kind, X[train_i], n_classes, params)
			pred_query = (cls._predict_idx(model, kind, q, n_classes, params)
			              if q.size else empty_i.copy())
		except cv2.error:
			# the algorithm cannot fit this data (e.g. Boost with > 2 classes)
			return fallback()

		accuracy = float((pred_eval == y_idx[eval_i]).mean()) if len(eval_i) else 0.0
		train_accuracy = (float((pred_train == y_idx[train_i]).mean())
		                  if len(train_i) else 0.0)
		confusion = np.zeros((n_classes, n_classes), np.int64)
		np.add.at(confusion, (y_idx[eval_i], pred_eval), 1)
		return io.NodeOutput(
			classes[pred_eval].astype(np.int32),
			classes[y_idx[eval_i]].astype(np.int32),
			accuracy, train_accuracy, confusion.astype(np.int32), True,
			classes[pred_query].astype(np.int32) if len(pred_query)
			else empty_i.copy(),
			cls._serialize(model))


# --- save / reuse ------------------------------------------------------------------
# The trained model is TEXT (the loadable OpenCV YAML above), so persisting it is
# just writing a file - and RELOADING it never hands a stateful handle across a
# graph edge: 'CV Predict Classifier' loads + predicts inside one execute()
# (a pure function of yaml text + features), the same cache-safety argument as
# the trainer itself. Models live in ComfyUI/models/cvml (named for cv2.ml - the
# trainer emits eight algorithm families, not just SVMs) as gzip-compressed
# .yml.gz (KNN/SVM serializations embed their training data; gzip ~10x).

_CVML_DIRNAME = "cvml"


def _register_cvml_folder():
	"""Register models/cvml with folder_paths (idempotent), like dnn.py does
	for models/onnx. Runs at import and inside define_schema so the combo is
	populated even on a cold server."""
	import folder_paths

	if _CVML_DIRNAME not in folder_paths.folder_names_and_paths:
		path = os.path.join(folder_paths.models_dir, _CVML_DIRNAME)
		os.makedirs(path, exist_ok=True)
		# .gz: gzip-compressed YAML (what Save Classifier writes - KNN/SVM
		# models embed their training data and compress ~10x); plain
		# .yml/.yaml files dropped in by hand load too
		folder_paths.folder_names_and_paths[_CVML_DIRNAME] = (
			[path], {".yml", ".yaml", ".gz"})


def _cvml_models() -> list:
	import folder_paths

	_register_cvml_folder()
	return folder_paths.get_filename_list(_CVML_DIRNAME)


_register_cvml_folder()

# top-node name of the serialized YAML (opencv_ml_svm, opencv_ml_rtrees, ...)
# -> (our algorithm kind, factory) - built from the live cv2 so the names can
# never drift from what getDefaultName() actually returns in this build
_MODEL_READERS = {}
for _label, (_kind, _factory) in _ALGORITHMS.items():
	_m = getattr(cv2.ml, _factory)()
	_MODEL_READERS[_m.getDefaultName()] = (_kind, _factory)
if '_m' in locals():
	del _m


def _load_model(yaml_text):
	"""(model, kind) from loadable cv2.ml YAML text; (None, None) on anything
	that is not a readable serialized model (failure-tolerant by design)."""
	try:
		fs = cv2.FileStorage(
			yaml_text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
		node = fs.getFirstTopLevelNode()
		reader = _MODEL_READERS.get(node.name())
		if reader is None:
			return None, None
		kind, factory = reader
		model = getattr(cv2.ml, factory)()
		model.read(node)
		return model, kind
	except cv2.error:
		return None, None


class SaveClassifier(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SaveClassifier",
			display_name="CV Save Classifier",
			category=CATEGORY,
			description="Writes a trained classifier to ComfyUI/models/cvml as "
			            "a gzip-compressed '.yml.gz' file, so a model trained "
			            "once ('CV Train Classifier' model_yaml output) can "
			            "be reloaded in a LATER workflow by 'CV Predict "
			            "Classifier'. The content is the standard OpenCV "
			            "StatModel YAML (KNN/SVM models embed their training "
			            "data - gzip shrinks them ~10x); to use it outside "
			            "ComfyUI, gunzip first, then cv2.ml.SVM_load & co. read "
			            "it. An existing file of the same name is overwritten "
			            "(deterministic re-runs). An EMPTY model_yaml (a trainer "
			            "that reported success=false) writes nothing and flags "
			            "saved=false instead of raising. This is an output node: "
			            "it runs even with nothing wired downstream.",
			inputs=[
				io.String.Input("model_yaml", force_input=True,
					tooltip="The model_yaml output of 'CV Train Classifier' "
					        "(loadable OpenCV YAML text)."),
				io.String.Input("filename", default="classifier",
					tooltip="File name inside models/cvml; '.yml.gz' is "
					        "appended if missing, path components are stripped."),
			],
			outputs=[
				io.String.Output(display_name="filename",
					tooltip="Name of the file written inside models/cvml - the "
					        "same string the 'model' combo of 'CV Predict "
					        "Classifier' lists after a page reload ('' when "
					        "nothing was saved). The folder it lives in is "
					        "deliberately NOT part of it: an absolute path "
					        "leaving the node would end up in shared workflow "
					        "JSON and pasted previews, carrying the drive "
					        "layout, install location and user name with it."),
				io.Boolean.Output(display_name="saved",
					tooltip="False when model_yaml was empty (the trainer "
					        "failed upstream) - nothing was written."),
			],
			is_output_node=True,
		)

	@classmethod
	def execute(cls, model_yaml, filename) -> io.NodeOutput:
		import folder_paths

		_register_cvml_folder()
		text = str(model_yaml or "")
		if not text.strip():
			return io.NodeOutput("", False, ui=ui.PreviewText(
				"Nothing to save: model_yaml is empty (trainer failed?)"))
		name = os.path.basename(str(filename)).strip() or "classifier"
		if name.lower().endswith((".yml", ".yaml")):
			name += ".gz"
		elif not name.lower().endswith(".gz"):
			name += ".yml.gz"
		path = os.path.join(folder_paths.folder_names_and_paths
		                    [_CVML_DIRNAME][0][0], name)
		# mtime=0 keeps the gzip bytes identical across re-runs of the same model
		with gzip.GzipFile(path, "wb", mtime=0) as f:
			f.write(text.encode("utf-8"))
		# The path stays inside execute(): only the bare name leaves the node,
		# and the preview names the folder relative to ComfyUI (same reason as
		# buildinfo.redact_paths / 'CV Install Example Inputs').
		return io.NodeOutput(name, True, ui=ui.PreviewText(
			f"Saved classifier -> models/{_CVML_DIRNAME}/{name}"))


class PredictClassifier(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		models = _cvml_models()
		return io.Schema(
			node_id="CV_PredictClassifier",
			display_name="CV Predict Classifier",
			category=CATEGORY,
			description="Classifies feature rows with a PREVIOUSLY TRAINED "
			            "cv2.ml model: pick a file from ComfyUI/models/cvml - "
			            "the gzip '.yml.gz' files 'CV Save Classifier' "
			            "writes, or a plain '.yml'/'.yaml' dropped in by hand - "
			            "or connect a model_yaml STRING to skip the file "
			            "entirely (it takes precedence when connected). "
			            "Loading and predicting "
			            "happen inside one call - the model never crosses a "
			            "graph edge, so caching stays correct (the same "
			            "pure-function argument as 'CV Train Classifier'). "
			            "The features must have the SAME length D the model was "
			            "trained on (e.g. the same 'CV HOG Features' "
			            "settings). Predictions are the label values the model "
			            "was trained on - 'CV Train Classifier' trains on "
			            "class INDICES 0..K-1 in 'CV Stack Feature Classes' "
			            "input order (class_a = 0, ...). Failure-tolerant: a "
			            "missing/corrupt file, a non-model YAML, a feature-"
			            "length mismatch or empty features yield success=false "
			            "and empty predictions, never an error - gate "
			            "downstream consumers with if/else.",
			inputs=[
				NPArray.Input("features",
					tooltip="(N, D) rows to classify (any shape flattens to one "
					        "row per sample) - same D as at training time."),
				io.Combo.Input("model", options=models,
					tooltip="Saved classifier from ComfyUI/models/cvml (written "
					        "by 'CV Save Classifier'). Reload the page to "
					        "list files saved during the same session."),
				io.String.Input("model_yaml", force_input=True, optional=True,
					tooltip="Optional: a model as YAML text (e.g. straight from "
					        "'CV Train Classifier'). When connected and "
					        "non-empty it overrides the model file above."),
			],
			outputs=[
				NPArray.Output(display_name="predictions",
					tooltip="(N,) int32 predicted label per feature row (class "
					        "indices 0..K-1 for models from 'CV Train "
					        "Classifier'). Empty when success=false."),
				io.Int.Output(display_name="count",
					tooltip="How many rows were classified."),
				io.Boolean.Output(display_name="success",
					tooltip="False when the model could not be loaded or the "
					        "features do not fit it - gate with if/else."),
			],
		)

	@classmethod
	def execute(cls, features, model="", model_yaml=None) -> io.NodeOutput:
		import folder_paths

		empty = np.zeros((0,), np.int32)
		X = _rows(features, "features")
		text = ""
		if model_yaml is not None and str(model_yaml).strip():
			text = str(model_yaml)
		else:
			_register_cvml_folder()
			path = folder_paths.get_full_path(_CVML_DIRNAME, str(model))
			if path is not None:
				try:
					opener = (gzip.open if path.lower().endswith(".gz")
					          else open)
					with opener(path, "rt", encoding="utf-8") as f:
						text = f.read()
				except OSError:  # gzip.BadGzipFile subclasses OSError
					text = ""
		clf, kind = _load_model(text) if text.strip() else (None, None)
		if clf is None or len(X) == 0 or X.shape[1] == 0:
			return io.NodeOutput(empty, 0, False)
		try:
			if kind == "knn":
				results = clf.findNearest(X, max(1, int(clf.getDefaultK())))[1]
				preds = np.rint(np.asarray(results).ravel()).astype(np.int32)
			elif kind == "mlp":
				preds = clf.predict(X)[1].argmax(axis=1).astype(np.int32)
			else:
				preds = np.rint(clf.predict(X)[1].ravel()).astype(np.int32)
		except cv2.error:
			# e.g. a feature-length mismatch with the trained var_count
			return io.NodeOutput(empty, 0, False)
		return io.NodeOutput(preds, len(preds), True)


# --- classic boosted cascade detection --------------------------------------
# cv2.CascadeClassifier is the pre-deep-learning object detector: a cascade of
# boosted Haar/LBP stages, serialized as an XML file. OpenCV 5 DELETED the
# data/ folder that used to ship those XMLs (cv2.data.haarcascades is an empty
# directory in this build), so the cascades live in ComfyUI/models/cascades and
# are downloaded from the opencv 4.x branch - see README. The class API is
# invisible to generator.py, exactly like cv2.ml itself.

_CASCADE_DIRNAME = "cascades"


def _register_cascade_folder():
	"""Register models/cascades with folder_paths (idempotent), like
	_register_cvml_folder above. Runs at import and inside define_schema so the
	combo is populated even on a cold server."""
	import folder_paths

	if _CASCADE_DIRNAME not in folder_paths.folder_names_and_paths:
		path = os.path.join(folder_paths.models_dir, _CASCADE_DIRNAME)
		os.makedirs(path, exist_ok=True)
		folder_paths.folder_names_and_paths[_CASCADE_DIRNAME] = ([path], {".xml"})


def _cascade_models() -> list:
	import folder_paths

	_register_cascade_folder()
	return folder_paths.get_filename_list(_CASCADE_DIRNAME)


_register_cascade_folder()


class CascadeDetect(io.ComfyNode):
	"""cv2.CascadeClassifier: Viola-Jones / LBP cascade object detection."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CascadeDetect",
			display_name="CV Cascade Detect",
			category=CATEGORY,
			description="The classic Viola-Jones detector (cv2.CascadeClassifier): a "
			            "cascade of boosted Haar or LBP stages that rejects "
			            "non-objects in a few operations and only spends real work "
			            "on promising windows. It needs no GPU and no ONNX runtime, "
			            "which is why it still matters - and it is the historical "
			            "counterpart to the pack's neural detectors ('CV YuNet "
			            "Face Detect' is far more accurate on faces; compare them "
			            "side by side). Trained cascades are XML files in "
			            "ComfyUI/models/cascades; OpenCV 5 no longer ships any, so "
			            "download them from the opencv 4.x branch (data/haarcascades, "
			            "data/lbpcascades) - they still load here. Data only: feed "
			            "'bboxes' to the core 'Draw BBoxes'. Zero detections is a "
			            "valid result (found=false), never an error.",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Image to search. An IMAGE batch uses its first frame; "
					        "converted to grayscale internally."),
				io.Combo.Input("cascade", options=_cascade_models(),
					tooltip="Cascade XML from ComfyUI/models/cascades "
					        "(haarcascade_frontalface_default.xml, "
					        "haarcascade_eye.xml, lbpcascade_* ...). Each file "
					        "detects exactly ONE object class."),
				io.Float.Input("scale_factor", default=1.1, min=1.01, max=2.0, step=0.01,
					tooltip="How much the search window grows between pyramid "
					        "levels. 1.1 = 10% per step: smaller finds more objects "
					        "at more sizes and is much slower; 1.3+ is fast and "
					        "misses objects between steps."),
				io.Int.Input("min_neighbors", default=5, min=0, max=100,
					tooltip="How many overlapping detections a window must collect "
					        "to be kept. This is the precision knob: raise it to kill "
					        "false positives, lower it if real objects are missed."),
				io.Int.Input("min_size", default=30, min=0, max=10000,
					tooltip="Ignore objects smaller than this many pixels on a side. "
					        "0 = no lower limit (much slower on big images)."),
				io.Int.Input("max_size", default=0, min=0, max=10000,
					tooltip="Ignore objects larger than this many pixels on a side. "
					        "0 = no upper limit."),
				io.Boolean.Input("equalize_hist", default=True,
					tooltip="Run cv2.equalizeHist first. Haar cascades were trained "
					        "on equalized crops, so this is the standard "
					        "pre-processing and usually helps on dim or "
					        "unevenly-lit photos.", optional=True, advanced=True),
				io.Float.Input("min_confidence", default=0.0, min=0.0, max=1000.0, step=0.1,
					tooltip="Drop detections whose stage weight (the cascade's own "
					        "confidence, from detectMultiScale3) is below this. 0 "
					        "keeps everything - raise it to rank and trim without "
					        "touching min_neighbors.", optional=True, advanced=True),
			],
			outputs=[
				io.Boolean.Output(display_name="found",
					tooltip="True when at least one object was detected - branch on "
					        "it with 'Basic data handling: IfElse'."),
				io.BoundingBox.Output(display_name="bboxes",
					tooltip="One {x, y, width, height, score, label} dict per "
					        "detection (score = the cascade's stage weight) - feed "
					        "the core 'Draw BBoxes' node."),
				NPArray.Output(display_name="boxes",
					tooltip="Nx4 int32 (x, y, w, h) - the same detections as a raw "
					        "array, e.g. to seed 'CV Track Window'."),
				NPArray.Output(display_name="centers",
					tooltip="Nx2 float32 detection centres - feed 'CV Draw "
					        "Points' or an 'CV Kalman Filter Step'."),
				NPArray.Output(display_name="scores",
					tooltip="(N,) float32 stage weights: how strongly the cascade "
					        "voted for each detection. Higher = more confident."),
				io.Int.Output(display_name="count",
					tooltip="How many objects were detected; 0 is valid, not an "
					        "error."),
			],
		)

	@classmethod
	def execute(cls, image, cascade, scale_factor, min_neighbors, min_size, max_size,
	            equalize_hist=True, min_confidence=0.0) -> io.NodeOutput:
		import folder_paths

		empty = (False, [], np.zeros((0, 4), np.int32), np.zeros((0, 2), np.float32),
		         np.zeros((0,), np.float32), 0)
		frame = imageutils.first_bgr_frame(image)
		if frame is None:
			return io.NodeOutput(*empty)

		_register_cascade_folder()
		path = folder_paths.get_full_path(_CASCADE_DIRNAME, cascade)
		if path is None:
			raise ValueError(
				f"cascade '{cascade}' not found in ComfyUI/models/{_CASCADE_DIRNAME}. "
				"Download the XML from the opencv 4.x branch (data/haarcascades or "
				"data/lbpcascades) and drop it there.")
		clf = cv2.CascadeClassifier(path)
		if clf.empty():
			# a truncated download or a 5.x-branch 404 page saved as .xml - a
			# config mistake, so say exactly that instead of returning 0 objects
			raise ValueError(
				f"'{cascade}' loaded as an EMPTY cascade - the file is not a valid "
				"OpenCV cascade XML (a truncated or HTML-error download is the "
				"usual cause).")

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if equalize_hist:
			gray = cv2.equalizeHist(gray)
		kw = {"scaleFactor": float(scale_factor), "minNeighbors": int(min_neighbors)}
		if min_size > 0:
			kw["minSize"] = (int(min_size), int(min_size))
		if max_size > 0:
			kw["maxSize"] = (int(max_size), int(max_size))
		# detectMultiScale3 is the only variant that reports the stage weights
		rects, _levels, weights = clf.detectMultiScale3(
			gray, outputRejectLevels=True, **kw)
		rects = np.asarray(rects, np.int32).reshape(-1, 4)
		if not len(rects):
			return io.NodeOutput(*empty)
		scores = np.asarray(weights, np.float32).reshape(-1)
		if len(scores) != len(rects):  # older builds can return a scalar list
			scores = np.zeros((len(rects),), np.float32)
		if min_confidence > 0.0:
			keep = scores >= float(min_confidence)
			rects, scores = rects[keep], scores[keep]
			if not len(rects):
				return io.NodeOutput(*empty)

		label = os.path.splitext(os.path.basename(cascade))[0]
		boxes = [{"x": int(x), "y": int(y), "width": int(w), "height": int(h),
		          "score": round(float(s), 4), "label": label}
		         for (x, y, w, h), s in zip(rects, scores)]
		centers = np.stack([rects[:, 0] + rects[:, 2] / 2.0,
		                    rects[:, 1] + rects[:, 3] / 2.0], axis=1).astype(np.float32)
		return io.NodeOutput(True, boxes, rects, centers, scores, len(rects))


NODES = [RandomPoints, CoordinateGrid, StackFeatureClasses, HOGFeatures,
         DeepFeatures, TrainClassifier, SaveClassifier, PredictClassifier,
         CascadeDetect]
