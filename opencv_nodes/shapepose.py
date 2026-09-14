# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""ONE spelling for "how are these two shapes related?" across the whole pack.

Hu moments are invariant to translation, scale AND rotation, which is usually
sold as the point of them. It is also the problem: `cv2.matchShapes` cannot be
asked for LESS invariance, so "find this shape at this size, standing this way
up" is not expressible.

Reflection is a subtler story, and worth stating precisely because the obvious
summary is wrong. Six of the seven invariants are reflection-invariant too; only
h7 carries handedness. OpenCV's own HuMoments docs put it exactly:

    "These values are proved to be invariants to the image scale, rotation, and
     reflection except the seventh one, whose sign is changed by reflection.
     This invariance is proved with the assumption of infinite image resolution.
     In case of raster images, the computed Hu invariants for the original and
     transformed images are a bit different."
    -- docs.opencv.org, imgproc "Structural Analysis and Shape Descriptors";
       the invariants are Hu, M.-K., "Visual Pattern Recognition by Moment
       Invariants", IRE Trans. Information Theory, IT-8, 179-187, 1962.

`matchShapes` DOES use that sign - and then usually throws it away by accident.
`modules/imgproc/src/matchcontours.cpp` guards every term with
`if( ama > eps && amb > eps )` where `double eps = 1.e-5;`, and |h7| is
typically 1e-6 to 1e-8, so the one term carrying handedness is silently skipped.
Measured on this build: an "F" glyph (|h7| = 6.5e-06) scores 0.000000 against
its mirror on I1, I2 AND I3 - indistinguishable from an identical copy, to
within float noise - while a hook-shaped polygon
(|h7| = 9.4e-05, above eps) scores **0.496 / 8.06 / 2.00**. So mirror
sensitivity is a CLIFF at |h7| ~ 1e-5, not a property: absent for most shapes,
abruptly enormous for a few, and never a usable handedness test either way.
That is what makes an explicit mirror policy necessary rather than a nicety.

This module recovers the pose that `matchShapes` throws away, so a matcher can
put the dimensions back one at a time:

    scale     sqrt(m00) ratio                        (measured: 0.7 -> 0.698-0.705)
    rotation  canonical 0-360 orientation difference (measured: exact to <0.4 deg)
    mirror    the sign of the 7th Hu invariant       (correct in all 20 poses tried)
    translation                                      never relevant, always ignored

Three things here are load-bearing and were each measured rather than assumed:

* The 0-360 orientation needs the THIRD-order moments. The second-order axis is
  only defined mod 180, so a shape and its upside-down copy are identical to it.
  Rotating the third-order central moments into the principal frame and taking
  the sign of nu30 resolves the head from the tail. Use the ANALYTIC rotation
  (below), not a sum over contour vertices - the latter is vertex-density
  sensitive and silently wrong for a polygon with uneven sampling.

* `chirality` is `sign(h7) * |h7|**0.25`, not a bare sign. h5 + i*h7 is a
  third-order complex invariant, so |h7|**0.25 is a NORMALIZED THIRD MOMENT -
  same units and same order of magnitude as `confidence`. One float then carries
  both the verdict (its sign) and how much to trust it (its magnitude), and it
  needs no division by a noisy denominator. Measured magnitudes: an "F" glyph
  0.020-0.048, a scalene triangle 0.0069, an isoceles triangle 0.0003-0.0054, a
  square or circle exactly 0.0. A genuinely chiral shape is size-invariant while
  a symmetric shape's noise decays with resolution, so THE TWO FAMILIES OVERLAP
  at low resolution - hence `min_chirality`, and hence a triangle is the wrong
  shape to demonstrate handedness with.

* `mirrored` is TRI-STATE, not a boolean. A mirror-symmetric shape IS its own
  mirror, so "is this one reflected?" genuinely has no answer for a square; +1
  means proven reflected, -1 proven same-handed, 0 undecidable.

The mirror convention is a CHOICE and everything downstream depends on it: a
mirror is a flip about the VERTICAL axis, applied BEFORE the rotation. Flipping
about the horizontal axis instead would report a different rotation for the same
match. `similarity_matrix` is what pins it, and the transform-residual test is
what stops a refactor changing it quietly.

Pure Python: no cv2, no numpy, no ComfyUI - unit-testable on synthetic moment
dicts, which is how the circular-wrap and tri-state rules stay honest.
"""

import math

# `cv2.moments` returns m00 = 0 for a zero-area contour and a NEGATIVE m00 for a
# reversed-winding one (measured: -9.0), which would break sqrt() and the nu
# normalization silently. Both are "no pose available" (contours.ShapeMoments
# already uses this threshold for its own degenerate branch).
DEGENERATE_M00 = 1e-9

SCALE_MODES = ["any scale (invariant)",
               "same scale as reference"]
ROTATION_MODES = ["any rotation (invariant)",
                  "same orientation as reference",
                  "same orientation or 180 deg flipped"]
MIRROR_MODES = ["either handedness (invariant)",
                "same handedness only (reject mirrored)",
                "mirrored only"]

# Shared by the contour matcher and the raster matcher so the two interfaces
# cannot drift apart. Numbers quoted are measured on this build.
SCALE_MODE_TOOLTIP = (
	"Hu moments ignore size. 'same scale as reference' additionally requires the "
	"candidate to be about as big as the reference - the ratio of sqrt(m00), so "
	"0.5 means half the linear size."
)
SCALE_TOLERANCE_TOOLTIP = (
	"Accepted size band as a fraction: 0.10 accepts 0.91x to 1.10x of the "
	"reference (symmetric in the ratio, so growing and shrinking are treated "
	"alike). 0 demands an exact match. Ignored unless scale is 'same scale as "
	"reference'."
)
ROTATION_MODE_TOOLTIP = (
	"'same orientation as reference' keeps only candidates standing the same way "
	"up. 'same orientation or 180 deg flipped' also accepts the upside-down copy "
	"- USE IT for 2-fold symmetric shapes (ellipse, rectangle), whose 0-360 "
	"orientation flips arbitrarily. Both are meaningless for a shape with no "
	"well-defined orientation: check 'pose_confidence' first (a rotating square "
	"reads 0, 140, 145, 155, 135 degrees - pure noise)."
)
ROTATION_TOLERANCE_TOOLTIP = (
	"Half-width of the accepted rotation window in degrees. Compared CIRCULARLY, "
	"so 359 degrees counts as 1 degree away from 0."
)
MIRROR_MODE_TOOLTIP = (
	"The matchShapes distance is a poor handedness test, so decide it here "
	"instead. Only the 7th Hu invariant's SIGN changes under reflection (OpenCV: "
	"\"invariants to the image scale, rotation, and reflection except the "
	"seventh one, whose sign is changed by reflection\"), and matchShapes skips "
	"any term below its internal eps = 1e-5 - which |h7| usually is. Measured: "
	"an 'F' and its mirror are indistinguishable (~0), while a hook with a bigger "
	"|h7| scores 0.50. 'same handedness only' drops reflected copies; 'mirrored "
	"only' keeps just the reflected ones. A mirror-symmetric shape (square, "
	"circle, isoceles triangle) is its own mirror, so it counts as UNDECIDED: "
	"kept by 'same handedness only', dropped by 'mirrored only'."
)
MIN_CHIRALITY_TOOLTIP = (
	"How chiral a shape must be before mirroring can be decided at all - the "
	"magnitude of the 'chirality' value. This knob exists because the 7th Hu "
	"invariant is the one routinely discarded in practice: it is the smallest "
	"and the most fragile, and OpenCV's own docs note the invariance is proved "
	"\"with the assumption of infinite image resolution\", so \"in case of "
	"raster images, the computed Hu invariants for the original and transformed "
	"images are a bit different\". Rasterizing a symmetric shape therefore "
	"produces a small NON-zero h7 whose sign is meaningless. Measured "
	"magnitudes: an 'F' glyph 0.020-0.048, a scalene triangle 0.0069, a "
	"rasterized symmetric shape up to ~0.005 of pure noise, a square exactly 0. "
	"Below this on EITHER side the match is reported as mirrored = 0 "
	"(undecidable) rather than guessed. Set 0 to trust the sign always."
)


def _mode_index(value, options):
	"""Position of `value` in `options`, or 0 (the invariant/permissive mode).

	Falling back to 0 rather than raising keeps a workflow saved against an older
	label list loading with today's behaviour, which is the same back-compat rule
	the low-level wrappers apply to their "" sentinel (rule 4: a config mistake
	must not halt a graph).
	"""
	try:
		return list(options).index(value)
	except ValueError:
		return 0


def chirality(hu):
	"""Signed handedness from the 7 raw Hu invariants.

	The SIGN flips under reflection and survives any rotation or scale; the
	MAGNITUDE says how chiral the shape is (see the module docstring for the
	measured bands). `hu` is the raw `cv2.HuMoments()` vector, not the
	log-scaled `hu_moments` output of 'CV Shape Moments' - though those two
	do agree in SIGN, because that column negates the sign and then takes
	log10 of a magnitude far below 1, and the two flips cancel.
	"""
	h7 = float(hu[6])
	return math.copysign(abs(h7) ** 0.25, h7)


def pose(m, hu):
	"""-> pose dict, or None when the shape has no usable moments.

	`m` is the dict `cv2.moments` returns, `hu` the raw `cv2.HuMoments` vector.
	None means zero-area, or a negative m00 (a reversed contour, or a float
	image carrying negative values) - callers treat it as "no pose", never as an
	error.
	"""
	m00 = float(m["m00"])
	if m00 <= DEGENERATE_M00:
		return None
	mu20, mu02 = float(m["mu20"]) / m00, float(m["mu02"]) / m00
	mu11 = float(m["mu11"]) / m00
	# principal axis of the second-order central moments - defined only mod 180
	theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
	cos_t, sin_t = math.cos(theta), math.sin(theta)
	# ...so rotate the THIRD-order moments into that frame and let the skew pick
	# the head from the tail. nu* are already normalized by m00**2.5, which is
	# why this needs no explicit power of m00 (and cannot overflow on a big blob).
	n30 = (float(m["nu30"]) * cos_t ** 3
	       + 3.0 * float(m["nu21"]) * cos_t * cos_t * sin_t
	       + 3.0 * float(m["nu12"]) * cos_t * sin_t * sin_t
	       + float(m["nu03"]) * sin_t ** 3)
	if n30 < 0.0:
		theta += math.pi
		n30 = -n30
	return {
		"cx": float(m["m10"]) / m00,
		"cy": float(m["m01"]) / m00,
		"m00": m00,
		"size": math.sqrt(m00),
		"orientation": math.degrees(theta) % 360.0,
		# |nu30| in the canonical frame: how well-defined `orientation` is.
		# Measured: 0.052 for a chiral "F", 0.00057 for an ellipse, 0.0 for a
		# square - note an ellipse is 2e-5, not 0, so a "> 0" test accepts noise.
		"confidence": abs(n30),
		"chirality": chirality(hu),
	}


def second_order(m):
	"""-> (angle_deg, eccentricity, half_major_axis) from the second-order
	central moments.

	The eigenvalues of the covariance matrix are the variances along the
	principal axes; their ratio gives the eccentricity and the atan2 the
	major-axis direction (y down, so positive angles turn clockwise on screen).
	`angle` is defined only mod 180 - `pose()['orientation']` is the 0-360
	version. Moved here VERBATIM from contours.ShapeMoments so the contour and
	raster moment nodes cannot drift; the expressions are unchanged so its
	`angles` output stays bit-identical.
	"""
	m00 = float(m["m00"])
	mu20, mu02 = float(m["mu20"]) / m00, float(m["mu02"]) / m00
	mu11 = float(m["mu11"]) / m00
	common = math.sqrt(4.0 * mu11 ** 2 + (mu20 - mu02) ** 2)
	l1 = (mu20 + mu02 + common) / 2.0
	l2 = (mu20 + mu02 - common) / 2.0
	angle = 0.5 * math.degrees(math.atan2(2.0 * mu11, mu20 - mu02))
	ecc = math.sqrt(max(0.0, 1.0 - l2 / l1)) if l1 > 1e-12 else 0.0
	half = 2.0 * math.sqrt(max(0.0, l1))  # 2 sigma along the major axis
	return angle, ecc, half


def angle_delta(a, b):
	"""Signed a - b wrapped into (-180, 180].

	Every angle comparison in the pack goes through this. The naive abs(a - b)
	calls 359.9 and 0.1 degrees 359.8 apart instead of 0.2, which would make a
	rotation-aware match reject a PERFECT hit - one measured case really does
	return 360.00.
	"""
	return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def relate(ref, cand, min_chirality=0.0):
	"""How to get from the reference shape to the candidate.

	`ref`/`cand` are `pose()` results (either may be None). Returns
	`{scale_ratio, rotation_deg, mirrored}` with `mirrored` tri-state:
	+1 reflected, -1 proven same handedness, 0 undecidable.
	"""
	if ref is None or cand is None:
		# no pose on one side: report "nothing known" rather than a plausible
		# lie. `accepts` rejects ratio 0 in scale-aware mode, so a zero-area
		# reference yields a visibly EMPTY result instead of a wrong one.
		return {"scale_ratio": 0.0, "rotation_deg": 0.0, "mirrored": 0}
	decidable = (abs(ref["chirality"]) >= min_chirality
	             and abs(cand["chirality"]) >= min_chirality)
	if not decidable:
		mirrored = 0
	elif ref["chirality"] * cand["chirality"] < 0.0:
		mirrored = 1
	else:
		mirrored = -1
	if mirrored > 0:
		# a flip about the vertical axis maps an orientation t to 180 - t, so the
		# rotation applied AFTER the flip is c - (180 - r). Verified to <0.4 deg
		# against known poses; see the module docstring on the convention.
		rotation = (cand["orientation"] + ref["orientation"] - 180.0) % 360.0
	else:
		rotation = (cand["orientation"] - ref["orientation"]) % 360.0
	return {
		"scale_ratio": cand["size"] / ref["size"],
		"rotation_deg": rotation,
		"mirrored": mirrored,
	}


def similarity_matrix(ref, cand, rel):
	"""2x3 similarity mapping REFERENCE coordinates onto the candidate.

	Mirror, then scale, then rotate, then translate centroid to centroid -
	`A = s * R(phi) * M`. Measured residual when warping the reference outline
	onto its match: 0.00-0.65 px.

	Degenerate input (either pose None) yields the identity plus a zero shift:
	there is no pose to express, and the caller has already been told so by
	`relate`'s zero scale_ratio.
	"""
	if ref is None or cand is None:
		return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
	s = rel["scale_ratio"]
	phi = math.radians(rel["rotation_deg"])
	cos_p, sin_p = math.cos(phi), math.sin(phi)
	if rel["mirrored"] > 0:
		# R @ diag(-1, 1)
		a11, a12 = -s * cos_p, -s * sin_p
		a21, a22 = -s * sin_p, s * cos_p
	else:
		a11, a12 = s * cos_p, -s * sin_p
		a21, a22 = s * sin_p, s * cos_p
	tx = cand["cx"] - (a11 * ref["cx"] + a12 * ref["cy"])
	ty = cand["cy"] - (a21 * ref["cx"] + a22 * ref["cy"])
	return ((a11, a12, tx), (a21, a22, ty))


def accepts(rel, scale_mode, scale_tolerance, rotation_mode,
            rotation_tolerance_deg, mirror_mode):
	"""Does this relation pass the requested invariance policy?

	Every mode's first option is the fully-invariant one, so the defaults
	reproduce plain `cv2.matchShapes` behaviour exactly.
	"""
	if _mode_index(scale_mode, SCALE_MODES) == 1:
		ratio = rel["scale_ratio"]
		if ratio <= 0.0:
			return False
		# symmetric in the ratio: 1.1x and 1/1.1x are the same distance away
		if abs(math.log(ratio)) > math.log(1.0 + max(float(scale_tolerance), 0.0)):
			return False
	rot_mode = _mode_index(rotation_mode, ROTATION_MODES)
	if rot_mode:
		tol = max(float(rotation_tolerance_deg), 0.0)
		off = abs(angle_delta(rel["rotation_deg"], 0.0))
		if rot_mode == 2:
			off = min(off, abs(angle_delta(rel["rotation_deg"], 180.0)))
		if off > tol:
			return False
	mir_mode = _mode_index(mirror_mode, MIRROR_MODES)
	if mir_mode == 1 and rel["mirrored"] > 0:
		return False
	# "mirrored only" demands a PROVEN reflection: an undecidable (symmetric)
	# shape is not a mirrored variant of anything.
	if mir_mode == 2 and rel["mirrored"] <= 0:
		return False
	return True
