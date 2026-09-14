# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Image moments over the INTENSITY function - Hu's original formulation.

`cv2.moments` takes either a point set or a raster, and the two are different
objects. The contour nodes in `contours.py` only ever reach the point-set case
(Green's theorem over a polygon, i.e. a uniform density inside an outline).
This module reaches the raster case, which is the general one:

    f = pixel value           binaryImage=False, cv2's DEFAULT - Hu (1962)
                              defines m_pq = sum x^p y^q f(x,y) over exactly
                              this density, so a shaded region is a different
                              shape from a flat one
    f = 1 where non-zero      binaryImage=True - the silhouette
    f = 1 inside a polygon    contour input; binaryImage is IGNORED there

Note the argument name is a lie worth knowing about: `binaryImage` does not
assert "this image is already binary", it INSTRUCTS cv2 to binarize. Measured on
an image holding 9 px of value 1 and 4 px of value 200: m00 = 809 (the sum of
the values) with False and 13 (the count of non-zero pixels) with True.

Why this is worth a node at all: two shapes with IDENTICAL outlines are
indistinguishable to `matchShapes` on contours (measured: exactly 0.000000 for a
flat disk against a disk shaded from the left) and separate cleanly on intensity
(0.025743), while the canonical orientation reads 0 vs 270 degrees. The pose
recovery is if anything better here - a chiral intensity pattern inside a
perfectly CIRCULAR silhouette, which the contour path cannot see at all,
recovered rotation to <=0.01 degrees and scale to 3 decimal places.

Four measured traps, all silent, all reflected in the tooltips:

* NOT INVARIANT TO EXPOSURE. `nu -> k^(1-gamma) * nu` under a gain k, so a
  uniform brightness change alone moves every Hu component (gain 0.5 cost I1
  0.0394 raw). You cannot be invariant to both spatial scale and gain from
  moments alone, so this is a choice, not something to quietly "fix" - hence the
  `weighting` combo. Peak normalization is the DEFAULT because it is also what
  makes a 0..1 IMAGE and a 0..255 NPARRAY of the same picture comparable at all;
  raw values are one option, not the only one.
* A NON-ZERO BACKGROUND IS CATASTROPHIC. A +12 offset over a 400x400 frame moved
  the orientation by 18.97 degrees, because every background pixel now carries
  mass. `binary` weighting makes it WORSE, not better: every pixel is non-zero,
  so the whole frame becomes the shape.
* SHADING TILTS THE AXIS. A left-to-right gradient over a static shape moved the
  centroid 0.85 px and the principal axis 0.8 degrees. That is the feature, but
  it means an intensity match is a match on lighting as much as on form.
* NEGATIVE VALUES GIVE A NEGATIVE m00 (measured -9.0 for -1.0 over 9 px), which
  would break sqrt() and the nu normalization. `shapepose.pose` returns None
  there and the nodes report it as "no pose", never as an error (rule 4).

Also: `cv2.moments` REJECTS a 3-channel array outright (moments.cpp:623), so
everything is reduced to one channel on the way in.

The invariance policy (scale/rotation/mirror) is `shapepose`'s, shared verbatim
with 'CV Filter Contours By Shape' so the two matchers cannot drift apart.
"""

import cv2
import numpy as np
import torch

from comfy_api.latest import io

from . import enums, polytype, shapepose
from .convert import NPArray

CATEGORY = "image/CV/moments"

_RAW = "intensity, raw values (cv2 default)"
_PEAK = "intensity, peak-normalized"
_BINARY = "binary (any non-zero counts as 1)"
WEIGHTINGS = [_PEAK, _RAW, _BINARY]

WEIGHTING_TOOLTIP = (
	"Which density function the moments integrate. 'intensity' is Hu's original "
	"formulation and cv2's default (mass = pixel value), so a shaded region is a "
	"different shape from a flat one. Hu moments are NOT invariant to a "
	"brightness gain, so 'peak-normalized' (divide by the frame's own maximum) "
	"is the default: it more than halved the error of a x0.5 exposure change in "
	"testing (I1 0.0394 raw -> 0.0146). It does NOT fully remove the effect - "
	"dimming also shrinks the effective support of a soft-edged region, and no "
	"normalization can fix that. 'binary' ignores the values and uses "
	"the silhouette, which is the closest thing to what the contour nodes "
	"measure - but beware, it treats EVERY non-zero pixel as part of the shape, "
	"so any background haze swallows the whole frame."
)

# An intensity field's normalized third moments are ~1000x smaller than a
# hard-edged glyph's (measured 1e-5 against 0.020-0.048), so the contour node's
# 0.01 gate would call every intensity match undecidable.
_MIN_CHIRALITY_DEFAULT = 1e-6


def _frames_1ch(value, which):
	"""Any image-ish input -> list of single-channel float32 frames.

	IMAGE/MASK tensors are lifted onto the 0..255 scale a uint8 NPARRAY already
	uses. That lift is what makes the same picture measure the same however it
	reached the node - verified: Hu components 0-5 are bit-identical between a
	0..1 IMAGE and a 0..255 NPARRAY of one image, on every weighting. (Component
	6 is not, but it is ~0 for a symmetric shape, where the log scale amplifies
	pure float noise to +-30; `chirality` reports such a shape as undecidable,
	which is the honest answer.) An NPARRAY keeps its own values - a float score
	map is a legitimate density.
	"""
	if isinstance(value, torch.Tensor):
		# a MASK is [B,H,W] (or [H,W]), an IMAGE is [B,H,W,C]
		arr = value.detach().cpu().numpy().astype(np.float32) * 255.0
		frames = [arr] if arr.ndim == 2 else [arr[i] for i in range(arr.shape[0])]
	else:
		arr = np.asarray(value).astype(np.float32, copy=False)
		# the pack's NPARRAY convention: [H,W,...] is ONE frame, and only a 4-D
		# array is the batch polytype.concat builds
		if arr.ndim in (2, 3):
			frames = [arr]
		elif arr.ndim == 4:
			frames = [arr[i] for i in range(arr.shape[0])]
		else:
			raise ValueError(
				f"'{which}' has shape {arr.shape} - expected an image, a mask or "
				"an NPARRAY of 2, 3 or 4 dimensions.")
	out = []
	for f in frames:
		if f.ndim == 3:
			if f.shape[-1] == 1:
				f = f[..., 0]
			elif f.shape[-1] in (3, 4) and f.size:
				# cv2.moments REJECTS a multi-channel array; use the luminance so
				# a colour picture reduces the way every other gray path does
				f = cv2.cvtColor(np.ascontiguousarray(f[..., :3], np.float32),
				                 cv2.COLOR_BGR2GRAY)
			else:
				f = f.mean(axis=-1) if f.size else f.reshape(f.shape[:2])
		if not f.size:
			# a zero-size frame still owes the caller a row; 1x1 of zero mass
			# lands in the "no pose" branch instead of asserting inside cv2
			f = np.zeros((1, 1), np.float32)
		out.append(np.ascontiguousarray(f, np.float32))
	return out


def _weighted(frame, weighting):
	"""Apply the density choice. Never raises on a degenerate frame."""
	if weighting == _BINARY:
		return (frame != 0.0).astype(np.float32)
	if weighting == _RAW:
		return frame
	peak = float(np.max(np.abs(frame))) if frame.size else 0.0
	return frame / peak if peak > 0.0 else frame


def _measure(frame, weighting):
	"""-> (moments dict, raw Hu vector) for one prepared frame."""
	prepared = _weighted(frame, weighting)
	m = cv2.moments(prepared, binaryImage=False)
	return m, cv2.HuMoments(m).ravel()


class ImageMoments(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ImageMoments",
			display_name="CV Image Moments",
			category=CATEGORY,
			description="Measures every frame of a batch with cv2.moments over "
			            "the IMAGE INTENSITY rather than an outline - Hu's "
			            "original density formulation, and the case the contour "
			            "nodes cannot reach. Two regions with identical outlines "
			            "but different internal shading are the SAME shape to "
			            "'CV Shape Moments' and different shapes here. "
			            "Outputs mirror 'CV Shape Moments' one-for-one so the "
			            "two are interchangeable downstream. Feed it crops from "
			            "'Crop By Masks' / 'Crop By BBoxes' to measure regions of "
			            "one scene. Beware: intensity moments are NOT invariant "
			            "to exposure and a non-zero background is fatal - see the "
			            "weighting tooltip. Zero frames is a valid result, not an "
			            "error.",
			inputs=[
				polytype.poly_input("images", types=polytype.IMAGEISH,
					tooltip="Frames to measure, one row of output per frame. "
					        "Reduced to a single channel (luminance for a colour "
					        "picture) because cv2.moments rejects a 3-channel "
					        "array."),
				io.Combo.Input("weighting", options=WEIGHTINGS, default=_PEAK,
					tooltip=WEIGHTING_TOOLTIP),
			],
			outputs=[
				NPArray.Output(display_name="centroids",
					tooltip="(N,1,2) float32 centre of mass per frame - plugs "
					        "into 'CV Draw Points' / 'Draw Labels'. This is "
					        "the INTENSITY centroid, so shading moves it."),
				NPArray.Output(display_name="mass",
					tooltip="(N,) float32 the m00 moment: the SUM OF THE PIXEL "
					        "VALUES, not an area (with 'binary' weighting it is "
					        "the count of non-zero pixels, which is an area)."),
				NPArray.Output(display_name="angles",
					tooltip="(N,) float32 orientation of the major axis in "
					        "degrees, [-90, 90), from the +X axis with Y pointing "
					        "DOWN. Meaningless for a radially symmetric "
					        "distribution (eccentricity ~ 0)."),
				NPArray.Output(display_name="eccentricities",
					tooltip="(N,) float32 elongation in 0-1: 0 = radially "
					        "symmetric, -> 1 = a line. From the central-moment "
					        "eigenvalues."),
				NPArray.Output(display_name="axes",
					tooltip="(N,4) float32 major-axis segments (x1, y1, x2, y2) "
					        "through each centroid, 2 standard deviations long "
					        "per side - plugs into 'CV Draw Segments'."),
				NPArray.Output(display_name="hu_moments",
					tooltip="(N,7) float32 log-scaled Hu invariants, "
					        "-sign(h)*log10(|h|) per component, matching 'OpenCV "
					        "Shape Moments'. Invariant to translation, scale and "
					        "rotation of the intensity field - but NOT to a "
					        "brightness change."),
				io.Int.Output(display_name="count"),
				NPArray.Output(display_name="orientations",
					tooltip="(N,) float32 canonical orientation, 0-360, with the "
					        "head told from the tail by the third-order moments "
					        "(unlike 'angles', which is only defined mod 180). "
					        "MEANINGLESS when pose_confidence is ~0."),
				NPArray.Output(display_name="chirality",
					tooltip="(N,) float32 signed handedness from the 7th Hu "
					        "invariant, the only one whose sign changes under "
					        "reflection (CV HuMoments docs; Hu 1962) and the "
					        "one usually discarded as too small and too noisy to "
					        "trust - hence the sign(h7)*|h7|**0.25 rescaling. The "
					        "sign flips under reflection and survives rotation "
					        "and scale. NOTE the magnitudes here are far smaller "
					        "than for a hard-edged silhouette - a smooth "
					        "intensity field measures around 1e-5 where an 'F' "
					        "glyph measures 0.02-0.05 - so the matcher's "
					        "'min_chirality' gate needs a much lower value on "
					        "this path."),
				NPArray.Output(display_name="pose_confidence",
					tooltip="(N,) float32 how well-defined 'orientations' is (the "
					        "normalized third moment along the major axis). Judge "
					        "it relative to the other frames in the batch, not "
					        "against the silhouette thresholds quoted on 'OpenCV "
					        "Shape Moments'."),
			],
		)

	@classmethod
	def execute(cls, images, weighting=_PEAK) -> io.NodeOutput:
		frames = _frames_1ch(images, "images")
		cents, mass, angles, eccs, axes = [], [], [], [], []
		hu_all, oris, chirs, confs = [], [], [], []
		for frame in frames:
			m, hu = _measure(frame, weighting)
			p = shapepose.pose(m, hu)
			if p is None:
				# empty, or a signed-mass float field: no pose exists (rule 4)
				cents.append((0.0, 0.0))
				mass.append(float(m["m00"]))
				angles.append(0.0)
				eccs.append(0.0)
				axes.append((0.0, 0.0, 0.0, 0.0))
				hu_all.append(np.zeros(7, np.float64))
				oris.append(0.0)
				chirs.append(0.0)
				confs.append(0.0)
				continue
			angle, ecc, half = shapepose.second_order(m)
			dx = half * np.cos(np.radians(angle))
			dy = half * np.sin(np.radians(angle))
			cents.append((p["cx"], p["cy"]))
			mass.append(p["m00"])
			angles.append(angle)
			eccs.append(ecc)
			axes.append((p["cx"] - dx, p["cy"] - dy, p["cx"] + dx, p["cy"] + dy))
			hu_all.append(-np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-30)))
			oris.append(p["orientation"])
			chirs.append(p["chirality"])
			confs.append(p["confidence"])
		n = len(cents)
		return io.NodeOutput(
			np.float32(cents).reshape(n, 1, 2),
			np.float32(mass),
			np.float32(angles),
			np.float32(eccs),
			np.float32(axes).reshape(n, 4),
			np.float32(hu_all).reshape(n, 7),
			n,
			np.float32(oris),
			np.float32(chirs),
			np.float32(confs),
		)


class MatchImageMoments(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_MatchImageMoments",
			display_name="CV Match Image Moments",
			category=CATEGORY,
			description="Scores every frame of a batch against one reference by "
			            "Hu moments of the IMAGE INTENSITY (cv2.matchShapes on "
			            "raster moments), with the same scale/rotation/mirror "
			            "policy as 'CV Filter Contours By Shape'. Use it when "
			            "the thing that distinguishes two candidates is INSIDE "
			            "the outline: two disks shaded from opposite sides are "
			            "identical to a contour matcher (exactly 0.000000) and "
			            "separate here. Emits a keep-mask rather than a filtered "
			            "batch, so a zero-match result stays a valid batch - "
			            "combine 'matched' with 'CV Index Batch' to select. "
			            "Failure-tolerant: an empty batch yields found=false and "
			            "empty arrays, and a frame cv2 refuses to compare (one "
			            "side carrying no mass at all) comes back as an infinite "
			            "distance rather than an error.",
			inputs=[
				polytype.poly_input("reference", types=polytype.IMAGEISH,
					tooltip="The template to match against (frame 0 if a batch). "
					        "It does NOT have to be the same size as the "
					        "candidates - moments are scale invariant."),
				polytype.poly_input("images", types=polytype.IMAGEISH,
					tooltip="Candidate frames, one score per frame."),
				io.Combo.Input("method", options=enums.CONTOURS_MATCH.options,
					default=enums.CONTOURS_MATCH.default,
					tooltip="How the Hu signatures are compared (I1 is the usual "
					        "choice; I2/I3 are alternative norms)."),
				io.Float.Input("max_distance", default=0.3, min=0.0, max=1e6, step=0.01,
					tooltip="Largest accepted dissimilarity (0 = identical). "
					        "Raise it to 1e6 and read the distances output to "
					        "calibrate - intensity distances do not have the same "
					        "scale as contour ones."),
				io.Combo.Input("weighting", options=WEIGHTINGS, default=_PEAK,
					tooltip=WEIGHTING_TOOLTIP),
				io.Combo.Input("scale_mode", options=shapepose.SCALE_MODES,
					default=shapepose.SCALE_MODES[0],
					tooltip=shapepose.SCALE_MODE_TOOLTIP + " NOTE: on this path "
					        "'size' is sqrt(total intensity), so a brighter copy "
					        "reads as a bigger one unless weighting is 'binary'.",
					optional=True, advanced=True),
				io.Float.Input("scale_tolerance", default=0.10, min=0.0, max=100.0,
					step=0.01, tooltip=shapepose.SCALE_TOLERANCE_TOOLTIP,
					optional=True, advanced=True),
				io.Combo.Input("rotation_mode", options=shapepose.ROTATION_MODES,
					default=shapepose.ROTATION_MODES[0],
					tooltip=shapepose.ROTATION_MODE_TOOLTIP,
					optional=True, advanced=True),
				io.Float.Input("rotation_tolerance_deg", default=10.0, min=0.0, max=180.0,
					step=1.0, tooltip=shapepose.ROTATION_TOLERANCE_TOOLTIP,
					optional=True, advanced=True),
				io.Combo.Input("mirror_mode", options=shapepose.MIRROR_MODES,
					default=shapepose.MIRROR_MODES[0],
					tooltip=shapepose.MIRROR_MODE_TOOLTIP,
					optional=True, advanced=True),
				io.Float.Input("min_chirality", default=_MIN_CHIRALITY_DEFAULT,
					min=0.0, max=1.0, step=1e-6,
					tooltip=shapepose.MIN_CHIRALITY_TOOLTIP + " Those numbers are "
					        "for hard-edged silhouettes; a smooth intensity field "
					        "measures ~1e-5, which is why the default here is far "
					        "lower than on the contour matcher. Read the "
					        "'chirality' output of 'CV Image Moments' on your "
					        "own data before raising it.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="distances",
					tooltip="(N,) float32 matchShapes distance per INPUT frame, "
					        "in batch order (not sorted - the batch order is what "
					        "'CV Index Batch' needs). Rejected frames keep their "
					        "real distance; use 'matched' to tell them apart."),
				NPArray.Output(display_name="matched",
					tooltip="(N,) float32 keep-mask, 1 where the frame passed "
					        "BOTH the distance threshold and the invariance "
					        "policy, else 0."),
				io.Int.Output(display_name="best_index",
					tooltip="Index of the closest ACCEPTED frame, or -1 when "
					        "nothing matched. Feed it to 'CV Index Batch'."),
				io.Boolean.Output(display_name="found",
					tooltip="False when no frame passed - wire it into an "
					        "'if/else' rather than testing best_index by hand."),
				NPArray.Output(display_name="scale_ratios",
					tooltip="(N,) float32 sqrt(m00) of each frame over the "
					        "reference's. 0.0 where no pose exists."),
				NPArray.Output(display_name="rotations",
					tooltip="(N,) float32 rotation in degrees, 0-360, taking the "
					        "reference onto that frame (Y down, clockwise on "
					        "screen). For a MIRRORED match it is the rotation "
					        "applied AFTER flipping the reference about its "
					        "vertical axis."),
				NPArray.Output(display_name="mirrored",
					tooltip="(N,) float32 handedness verdict: +1 reflected, -1 "
					        "same handedness, 0 undecidable (symmetric, or below "
					        "'min_chirality')."),
				NPArray.Output(display_name="transforms",
					tooltip="(N,2,3) float32 similarity matrix per frame mapping "
					        "REFERENCE coordinates onto that frame (mirror, "
					        "scale, rotate, centroid to centroid). Feed a row to "
					        "cv2_warpAffine to overlay the template on its match."),
			],
		)

	@classmethod
	def execute(cls, reference, images, method, max_distance, weighting=_PEAK,
	            scale_mode=shapepose.SCALE_MODES[0], scale_tolerance=0.10,
	            rotation_mode=shapepose.ROTATION_MODES[0], rotation_tolerance_deg=10.0,
	            mirror_mode=shapepose.MIRROR_MODES[0],
	            min_chirality=_MIN_CHIRALITY_DEFAULT) -> io.NodeOutput:
		# these defaults MUST equal the schema defaults - a workflow test cannot
		# see a divergence, because the runner fills a missing widget from the
		# schema (test_match_image_moments pins it)
		frames = _frames_1ch(images, "images")
		ref_frames = _frames_1ch(reference, "reference")
		if not frames or not ref_frames:
			return cls._empty(len(frames))
		flag = enums.CONTOURS_MATCH.resolve(method)
		ref_prepared = _weighted(ref_frames[0], weighting)
		ref_m, ref_hu = _measure(ref_frames[0], weighting)
		ref_pose = shapepose.pose(ref_m, ref_hu)
		dists, keep, ratios, rots, mirs, mats = [], [], [], [], [], []
		for frame in frames:
			prepared = _weighted(frame, weighting)
			try:
				distance = float(cv2.matchShapes(ref_prepared, prepared, flag, 0.0))
			except cv2.error:
				distance = float("inf")
			# cv2 signals "these cannot be compared" (one side has no mass) by
			# returning DBL_MAX, not by raising and not with an inf: np.isfinite
			# says True, and casting it to float32 overflows to inf with a
			# RuntimeWarning. Normalize it here so the output array is honest.
			if distance > 3.0e38:
				distance = float("inf")
			m, hu = _measure(frame, weighting)
			cand_pose = shapepose.pose(m, hu)
			rel = shapepose.relate(ref_pose, cand_pose, float(min_chirality))
			ok = (np.isfinite(distance) and distance <= max_distance
			      and shapepose.accepts(rel, scale_mode, scale_tolerance,
			                            rotation_mode, rotation_tolerance_deg,
			                            mirror_mode))
			dists.append(distance)
			keep.append(1.0 if ok else 0.0)
			ratios.append(rel["scale_ratio"])
			rots.append(rel["rotation_deg"])
			mirs.append(float(rel["mirrored"]))
			mats.append(shapepose.similarity_matrix(ref_pose, cand_pose, rel))
		dists = np.float32(dists)
		keep = np.float32(keep)
		accepted = np.flatnonzero(keep > 0.0)
		best = int(accepted[int(np.argmin(dists[accepted]))]) if accepted.size else -1
		return io.NodeOutput(
			dists, keep, best, best >= 0,
			np.float32(ratios), np.float32(rots), np.float32(mirs),
			np.float32(mats).reshape(len(mats), 2, 3),
		)

	@staticmethod
	def _empty(n=0):
		z = np.zeros(n, np.float32)
		return io.NodeOutput(z, np.zeros(n, np.float32), -1, False,
		                     np.zeros(n, np.float32), np.zeros(n, np.float32),
		                     np.zeros(n, np.float32),
		                     np.zeros((n, 2, 3), np.float32))


NODES = [ImageMoments, MatchImageMoments]
