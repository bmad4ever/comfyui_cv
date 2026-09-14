# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""ONE spelling for "fit a warp to point correspondences, then apply it".

Five models behind a single interface, plus the pad -> warp -> crop ->
premultiplied-composite chain that pastes a warped image onto a background.

Why this module exists: the composite was written twice before it was written
once. `CV_QuadWarp`'s fill branch had its own white-plate/alpha/clip block,
and the 36-node "Annotate Thin Plate Spline & Paste" blueprint rebuilt the same
thing out of `cv2_bitwise_and` / `cv2_bitwise_not` / `cv2_multiply` / `cv2_add`.
`QuadWarp` fill is the 4-implicit-corner, homography-only special case of what
`CV_PasteThroughWarp` does with N correspondences and any model.

A leaf module on purpose: cv2 + numpy + the two local helper modules, no
`comfy_api`, no node classes. `points.py` already imports `features.py`, so
anything shared by both has to sit below both.

MODELS and MIN_PAIRS mirror `web/cvlib/warp_math.js` byte for byte - the
annotate editor draws a live WebGL preview of the SAME map, and "what you drag
is what the server computes" only holds while the two lists agree.
`tests/smoke_test.py::test_warpfit_models_match_js` pins them.
"""

import cv2
import numpy as np

from . import imageutils, polytype

MODELS = ("thin plate spline", "homography", "affine", "similarity",
          "translation")

# Below these a fit is not merely inaccurate, it is undetermined. Mirrors
# MIN_PAIRS in web/cvlib/warp_math.js.
MIN_PAIRS = {
	"thin plate spline": 3,
	"homography": 4,
	"affine": 3,
	"similarity": 2,
	"translation": 1,
}

# Descriptive combo labels that other nodes already ship (rule 6) mapped onto
# the canonical short names, so option strings saved in existing workflows keep
# resolving.
ALIASES = {
	"full affine - 6 dof (rotation, scale, shear, translation)": "affine",
	"similarity - 4 dof (rotation, uniform scale, translation)": "similarity",
}

MODEL_TOOLTIP = (
	"Which map is fitted through the correspondences. 'thin plate spline' BENDS "
	"the image so every 'from' point lands exactly on its 'to' point (3+ pairs, "
	"non-rigid - faces, cloth, anything a matrix cannot express). 'homography' "
	"is the 8-dof planar perspective map (4+ pairs; straight lines stay "
	"straight - posters, screens, documents). 'affine' is 6-dof (3+ pairs: "
	"rotation, scale, shear, translation, parallel lines stay parallel), "
	"'similarity' 4-dof (2+ pairs: rotation, uniform scale, translation, shape "
	"preserved), 'translation' a pure shift (1+ pair). Fewer pairs than the "
	"model needs is not an error - nothing is warped and 'found' comes out "
	"false.")


def canonical(model):
	"""The short model name, accepting the descriptive labels other nodes use."""
	name = ALIASES.get(model, model)
	if name not in MIN_PAIRS:
		raise ValueError(
			f"unknown warp model {model!r}; expected one of {list(MODELS)}.")
	return name


def min_pairs(model):
	"""How many correspondences this model needs before it is determined."""
	return MIN_PAIRS[canonical(model)]


def as_points(value):
	"""(N, 2) float32, treating an unconnected socket as no points."""
	if value is None:
		return np.zeros((0, 2), np.float32)
	return np.asarray(value, np.float32).reshape(-1, 2)


class Warp:
	"""A fitted correspondence warp. ONE interface for all five models:

	    forward(points)          -> (N, 2) where a SOURCE point goes
	    backward(points)         -> (N, 2) where a DESTINATION point came from
	    warp_image(src, dsize)   -> ndarray, the image carried along with it
	    matrix()                 -> 3x3 float64, or None for the spline

	`resizes` says whether warp_image can write into a canvas of a DIFFERENT
	size than its input. The matrix models can (cv2.warpPerspective takes a
	dsize); the cv2 shape transformers cannot - `warpImage` only ever fills the
	input array's own canvas, which is why `warp_onto` pads first.
	"""

	def __init__(self, model, *, fw=None, bw=None, mat=None):
		self.model = model
		self._fw = fw          # cv2 ShapeTransformer, from -> to
		self._bw = bw          # cv2 ShapeTransformer, to -> from
		self._mat = mat        # 3x3 float64, the models warped BY a matrix
		self._affine_mat = None  # the affine family's matrix, read back in fit()
		self.resizes = mat is not None

	# -- points ------------------------------------------------------------
	def forward(self, points):
		pts = as_points(points)
		if not len(pts):
			return np.zeros((0, 2), np.float32)
		if self._mat is not None:
			return self._apply_matrix(self._mat, pts)
		_, mapped = self._fw.applyTransformation(pts.reshape(1, -1, 2))
		return np.asarray(mapped, np.float32).reshape(-1, 2)

	def backward(self, points):
		pts = as_points(points)
		if not len(pts):
			return np.zeros((0, 2), np.float32)
		if self._bw is not None:      # the spline's to -> from fit
			_, mapped = self._bw.applyTransformation(pts.reshape(1, -1, 2))
			return np.asarray(mapped, np.float32).reshape(-1, 2)
		# every other model is a matrix, so its inverse IS the backward map
		return self._apply_matrix(np.linalg.inv(self._mat if self._mat is not None
		                                        else self._affine_mat), pts)

	@staticmethod
	def _apply_matrix(mat, pts):
		out = cv2.perspectiveTransform(pts.reshape(1, -1, 2).astype(np.float64),
		                               mat)
		return np.asarray(out, np.float32).reshape(-1, 2)

	# -- pixels ------------------------------------------------------------
	def warp_image(self, src, dsize=None):
		"""Carry `src`'s pixels along the fitted map.

		`dsize` is (width, height) and only the matrix models accept one that
		differs from the source - see `resizes`.
		"""
		canvas = np.asarray(src)
		h, w = canvas.shape[:2]
		if dsize is None:
			dsize = (w, h)
		if self._mat is not None:
			warped = cv2.warpPerspective(canvas, self._mat, dsize,
			                             flags=cv2.INTER_LINEAR,
			                             borderMode=cv2.BORDER_CONSTANT)
			return np.ascontiguousarray(warped)
		if tuple(dsize) != (w, h):
			raise ValueError(
				f"the {self.model!r} transformer can only warp within its "
				f"input's own canvas ({w}x{h}), not into {dsize[0]}x{dsize[1]} "
				"- pad first (warpfit.warp_onto does).")
		# ONE transformer for the affine family, TWO for the spline - see fit().
		warped = (self._bw or self._fw).warpImage(canvas)
		if warped.shape != canvas.shape:   # warpImage can drop a trailing 1-channel
			warped = warped.reshape(canvas.shape)
		return warped

	def matrix(self):
		"""The map as a 3x3, or None for the spline (which has no matrix)."""
		return None if self._mat is None else self._mat.copy()


def _shape_transformer(model, src, dst, regularization):
	matches = [cv2.DMatch(i, i, 0) for i in range(len(src))]
	if model == "thin plate spline":
		t = cv2.createThinPlateSplineShapeTransformer()
		t.setRegularizationParameter(float(regularization))
	else:
		# cv2.createAffineTransformer(fullAffine): True = the 6-dof
		# least-squares affine, False = the 4-dof partial (similarity) fit
		t = cv2.createAffineTransformer(model == "affine")
	t.estimateTransformation(src.reshape(1, -1, 2), dst.reshape(1, -1, 2), matches)
	return t


def _affine_matrix(t):
	"""Read a transformer's matrix back by pushing a basis through it.

	The transformer never exposes it; transforming the origin and the two unit
	vectors is exact for an affine map.
	"""
	basis = np.float32([[0, 0], [1, 0], [0, 1]]).reshape(1, -1, 2)
	_, b = t.applyTransformation(basis)
	b = np.asarray(b, np.float64).reshape(-1, 2)
	return np.float64([
		[b[1, 0] - b[0, 0], b[2, 0] - b[0, 0], b[0, 0]],
		[b[1, 1] - b[0, 1], b[2, 1] - b[0, 1], b[0, 1]],
		[0.0, 0.0, 1.0],
	])


def fit(model, points_from, points_to, regularization=0.0):
	"""Fit `model` through the correspondences. None when it is undetermined.

	Raises ONLY on row misalignment, which is always a caller bug. Too few
	points is NOT an error here: the caller decides what it means. On a wire
	from another estimator "too few" is a wiring mistake and the node should
	raise its own message (ThinPlateSplineWarp, AffineWarp); coming from a
	user's clicks it is the normal not-yet-annotated state and the node should
	report found=False (PasteThroughWarp, QuadWarp).
	"""
	name = canonical(model)
	src = as_points(points_from)
	dst = as_points(points_to)
	if len(src) != len(dst):
		raise ValueError(
			f"points_from has {len(src)} point(s) but points_to has {len(dst)} - "
			"they must be row-aligned correspondences.")
	if len(src) < MIN_PAIRS[name]:
		return None

	if name == "translation":
		d = (dst - src).mean(axis=0)
		return Warp(name, mat=np.float64([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1]]))

	if name == "homography":
		if len(src) == 4:
			mat = cv2.getPerspectiveTransform(src, dst)
		else:
			# method 0 = plain least squares. RANSAC belongs to
			# CV_FindHomographyRANSAC, which is an ESTIMATOR with its own
			# inlier mask and thresholds; this module only applies what it is
			# handed.
			mat, _ = cv2.findHomography(src, dst, 0)
		if mat is None or not np.isfinite(mat).all() or abs(np.linalg.det(mat)) < 1e-12:
			return None
		return Warp(name, mat=np.float64(mat))

	if name == "thin plate spline":
		# TWO transformers on purpose. applyTransformation() applies the fitted
		# map directly, so points need the from -> to fit; warpImage() is a
		# BACKWARD warp (it pushes the fitted map through the destination grid
		# and samples), so feeding it the same fit moves the pixels the WRONG
		# WAY - the image needs the to -> from fit for its content to follow the
		# points. Verified on this build: with one shared fit the control point
		# and the pixels under it travel in opposite directions.
		return Warp(name,
		            fw=_shape_transformer(name, src, dst, regularization),
		            bw=_shape_transformer(name, dst, src, regularization))

	# affine / similarity: ONE transformer, unlike the spline. warpImage() here
	# is cv2.warpAffine with the fitted matrix and no WARP_INVERSE_MAP, i.e. a
	# FORWARD warp: the same from -> to fit that maps the points also carries
	# the pixels under them the same way (verified on this build - a pure +50 px
	# fit moves the content +50 px). It is kept as the transformer rather than
	# reduced to its matrix so warp_image() stays byte-identical to what
	# CV_AffineWarp has always produced.
	t = _shape_transformer(name, src, dst, regularization)
	w = Warp(name, fw=t)
	w._affine_mat = _affine_matrix(t)
	return w


def matrix_of(w):
	"""The 3x3 of any model that has one, including the affine family.

	`Warp.matrix()` is None for the transformer-backed models because their
	pixels go through cv2's own warpImage; the affine family still HAS a matrix
	and CV_AffineWarp outputs it.
	"""
	if w is None:
		return None
	if w.matrix() is not None:
		return w.matrix()
	return getattr(w, "_affine_mat", None)


def residual(w, points_from, points_to):
	"""(N, 2) where `points_from` actually landed minus where it should have.

	~0 for an exact interpolator (the spline at regularization 0); for the rigid
	models it is the honest least-squares error - the measure of how badly the
	model fits.
	"""
	dst = as_points(points_to)
	if w is None or not len(dst):
		return np.zeros((0, 2), np.float32)
	return w.forward(points_from) - dst


# ---------------------------------------------------------------------------
# the pad -> warp -> crop -> composite chain
# ---------------------------------------------------------------------------

def coverage_plate(shape_hw):
	"""uint8 all-255 [H, W]: "every pixel of the source is real content".

	Pushed through the SAME warp as the picture, it comes back as the alpha of
	where that content landed - including the soft edge interpolation leaves.
	"""
	return np.full(tuple(shape_hw[:2]), 255, np.uint8)


def gate(arr, mask=None):
	"""`arr` restricted to `mask`; mask=None means "all of it".

	The unconnected-optional-MASK case and "no restriction" are the same thing,
	which is why every caller can pass its input straight through. A mask that
	does not match is FITTED, not rejected (rule 4).
	"""
	if mask is None:
		return arr
	arr = np.asarray(arr)
	m = imageutils.fit_mask_to(np.asarray(mask, np.uint8), arr.shape[:2])
	if arr.dtype == np.uint8:
		return cv2.bitwise_and(arr, arr, mask=m)
	# bitwise_and is integer-only; the NPARRAY path carries floats too
	keep = m != 0
	return np.where(keep[..., None] if arr.ndim == 3 else keep, arr, 0)


def warp_onto(w, src, canvas_hw, *, border_value=0):
	"""Warp `src` so that it lands in a (H, W) DESTINATION canvas.

	The matrix models take a dsize and write straight into the canvas. The cv2
	shape transformers cannot: `warpImage` only ever fills the input array's own
	canvas, so `src` is first padded bottom/right to max(src, canvas) with
	BORDER_CONSTANT, warped, then sliced back to exactly `canvas_hw`.

	BORDER_CONSTANT and not BORDER_DEFAULT: REFLECT_101 would mirror the content
	- and, worse, an all-white coverage plate - into the padding, so the paste
	would grow a mirrored fringe wherever the image is smaller than the
	background.

	TOP-LEFT anchored on purpose: padding only at the bottom and right leaves
	every `points_from` pixel coordinate valid in the padded canvas, so the
	correspondences drive the fit unchanged.
	"""
	src = np.asarray(src)
	h, w_ = src.shape[:2]
	height, width = int(canvas_hw[0]), int(canvas_hw[1])
	if w.resizes:
		return w.warp_image(src, (width, height))
	pad_b, pad_r = max(height - h, 0), max(width - w_, 0)
	if pad_b or pad_r:
		src = cv2.copyMakeBorder(src, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT,
		                         value=border_value)
	warped = w.warp_image(src)
	return np.ascontiguousarray(warped[:height, :width])


def composite_over(background, warped, coverage):
	"""PREMULTIPLIED composite: background * (1 - coverage) + warped.

	NOT a lerp. The constant black border makes `warped` equal
	source-times-coverage along the boundary already, so blending it a second
	time with the same alpha keeps black-blended fringe pixels - a visible dark
	outline around everything pasted. `background` is not modified.
	"""
	alpha = np.asarray(coverage, np.float32) / 255.0
	return polytype.paste_into(np.asarray(background).copy(), warped, 0, 0,
	                           alpha=alpha, premultiplied=True)


def paste_warped(image, background, w, *, content_mask=None,
                 paintable_mask=None):
	"""Warp `image` through `w` and composite it onto `background`.

	Returns (composite, coverage) - the pasted result and the uint8 alpha of
	where the content actually landed, both background-sized. `content_mask`
	says which part of `image` is real content; `paintable_mask` says where on
	`background` the paste is allowed to land. Either unconnected means "no
	restriction".
	"""
	src = np.asarray(image)
	canvas_hw = np.asarray(background).shape[:2]

	plate = coverage_plate(src.shape)
	src = gate(src, content_mask)
	plate = gate(plate, content_mask)

	warped = warp_onto(w, src, canvas_hw)
	coverage = warp_onto(w, plate, canvas_hw)

	warped = gate(warped, paintable_mask)
	coverage = gate(coverage, paintable_mask)

	return composite_over(background, warped, coverage), coverage
