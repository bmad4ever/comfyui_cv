# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Remap-map generators (NPARRAY-native).

cv2.remap renders dst(p) = src(map_x(p), map_y(p)) from two float32
coordinate arrays. These nodes BUILD such maps: 'CV Identity Map' emits
the do-nothing grid, and every distortion node REWRITES the coordinates fed
into it - so chaining distortion nodes composes their effects with no extra
plumbing. Finish with the low-level cv2_remap node to sample the image
(use borderMode=BORDER_CONSTANT unless you want edge reflections).

All transforms are closed-form; coordinates pushed outside the source frame
are valid (cv2.remap fills them with the border).
"""

import cv2
import numpy as np

from comfy_api.latest import io

from . import polytype
from .convert import NPArray

CATEGORY = "image/CV/remap"


def _as_map(name, arr) -> np.ndarray:
	if arr is None:
		raise ValueError(f"{name} is required - start from 'CV Identity Map'.")
	a = np.asarray(arr, np.float32)
	if a.ndim != 2 or a.size == 0:
		raise ValueError(f"{name} must be a non-empty HxW float array, got shape "
		                 f"{np.asarray(arr).shape}.")
	return a


def _map_pair(map_x, map_y):
	mx, my = _as_map("map_x", map_x), _as_map("map_y", map_y)
	if mx.shape != my.shape:
		raise ValueError(f"map_x {mx.shape} and map_y {my.shape} must have the "
		                 "same shape - both must come from the same map chain.")
	return mx, my


def _center(shape, center_x, center_y):
	h, w = shape
	return center_x * (w - 1), center_y * (h - 1)


class IdentityMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_IdentityMap",
			display_name="CV Identity Map",
			category=CATEGORY,
			description="The starting point of every remap chain: the do-"
			            "nothing coordinate grid sized like the given image "
			            "(map_x[y][x] = x, map_y[y][x] = y). Feed it through "
			            "distortion-map nodes (radial lens, cylinder, pinch, "
			            "wave, displacement, homography) - each rewrites the "
			            "coordinates, so chaining composes distortions - and "
			            "apply the final pair with the low-level cv2_remap.",
			inputs=[
				polytype.poly_input("image", tooltip="Only its width/height are used."),
			],
			outputs=[
				NPArray.Output(display_name="map_x",
					tooltip="HxW float32 source-x coordinate per pixel."),
				NPArray.Output(display_name="map_y",
					tooltip="HxW float32 source-y coordinate per pixel."),
			],
		)

	@classmethod
	def execute(cls, image) -> io.NodeOutput:
		arr = np.asarray(polytype.resolve(image)[0])
		if arr.ndim < 2 or arr.size == 0:
			raise ValueError(f"image must be a non-empty HxW(xC) array, got shape {arr.shape}.")
		h, w = arr.shape[:2]
		map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32),
		                           np.arange(h, dtype=np.float32))
		return io.NodeOutput(map_x, map_y)


class RadialLensMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_RadialLensMap",
			display_name="CV Radial Lens Map",
			category=CATEGORY,
			description="Polynomial radial (lens) distortion, like ImageMagick's "
			            "barrel distort: scales each coordinate's distance to "
			            "the center by a*r^3 + b*r^2 + c*r + d (r normalized so "
			            "r = 1 at the nearest edge). Positive b bulges the "
			            "center outward (barrel/fisheye look), negative b "
			            "pinches it (pincushion); mixing signs of a and b gives "
			            "mustache distortion. The inverse form undoes the "
			            "polynomial with the SAME coefficients (it solves "
			            "r' * P(r') = r per pixel with a few Newton steps, the "
			            "technique cv2.undistortPoints uses) - chain both to "
			            "round-trip a distortion to sub-pixel accuracy. Where "
			            "the polynomial is not monotonic (extreme coefficients "
			            "fold the image onto itself) the inverse clamps to the "
			            "last invertible radius instead of blowing up.",
			inputs=[
				NPArray.Input("map_x",
					tooltip="Incoming source-x coordinate map to distort. Start the "
					        "chain with 'CV Identity Map'; further remap nodes "
					        "compose onto it."),
				NPArray.Input("map_y",
					tooltip="Incoming source-y coordinate map (must share map_x's "
					        "shape - both come from the same chain)."),
				io.Float.Input("a", default=0.0, min=-8.0, max=8.0, step=0.01,
					tooltip="r^3 coefficient."),
				io.Float.Input("b", default=0.15, min=-8.0, max=8.0, step=0.01,
					tooltip="r^2 coefficient: + barrel, - pincushion."),
				io.Float.Input("c", default=0.0, min=-8.0, max=8.0, step=0.01,
					tooltip="r coefficient."),
				io.Combo.Input("formula",
					options=["polynomial: r * (a r^3 + b r^2 + c r + d)",
					         "inverse: undoes the polynomial (same coefficients)"],
					default="polynomial: r * (a r^3 + b r^2 + c r + d)",
					tooltip="polynomial applies the distortion; inverse undoes it "
					        "with the same coefficients (chain both to round-trip)."),
				io.Float.Input("d", default=0.0, min=-8.0, max=8.0, step=0.01,
					tooltip="Constant term; 0 = auto (1 - a - b - c), which "
					        "keeps the r = 1 circle fixed.", optional=True,
					advanced=True),
				io.Float.Input("center_x", default=0.5, min=0.0, max=1.0, step=0.01,
					tooltip="Distortion center as a fraction of the width.",
					optional=True, advanced=True),
				io.Float.Input("center_y", default=0.5, min=0.0, max=1.0, step=0.01,
					tooltip="Distortion center as a fraction of the height.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="map_x"),
				NPArray.Output(display_name="map_y"),
			],
		)

	@classmethod
	def execute(cls, map_x, map_y, a, b, c, formula,
	            d=0.0, center_x=0.5, center_y=0.5) -> io.NodeOutput:
		mx, my = _map_pair(map_x, map_y)
		h, w = mx.shape
		cx, cy = _center(mx.shape, center_x, center_y)
		norm = max(min(cx, (w - 1) - cx, cy, (h - 1) - cy), 1.0)
		if d == 0.0:
			d = 1.0 - (a + b + c)

		def poly_at(s):
			return ((a * s + b) * s + c) * s + d

		dx, dy = mx - cx, my - cy
		r = np.sqrt(dx * dx + dy * dy) / norm
		if formula.startswith("polynomial"):
			factor = poly_at(r)
		else:
			# inverse: solve s * P(s) = r for the source radius s of each
			# pixel (vectorized Newton; one division is only the first-order
			# inverse and drifts as a*r^3 grows). Iterates are clamped, so
			# radii past a fold of a non-monotonic P degrade gracefully.
			cap = float(r.max()) * 4.0 + 1.0
			s = np.clip(r / np.maximum(np.abs(poly_at(r)), 1e-6), 0.0, cap)
			for _ in range(10):
				f = s * poly_at(s) - r
				df = poly_at(s) + s * ((3.0 * a * s + 2.0 * b) * s + c)
				df = np.where(np.abs(df) < 1e-6, np.where(df < 0, -1e-6, 1e-6), df)
				s = np.clip(s - f / df, 0.0, cap)
			factor = np.where(r > 1e-9, s / np.maximum(r, 1e-9),
			                  1.0 / max(abs(d), 1e-6))
		return io.NodeOutput((cx + dx * factor).astype(np.float32),
		                     (cy + dy * factor).astype(np.float32))


class CylinderMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_CylinderMap",
			display_name="CV Cylinder Map",
			category=CATEGORY,
			description="Cylindrical projection: 'wrap' renders the image as if "
			            "glued onto a cylinder seen from its axis - the classic "
			            "panorama pre-warp that compresses the side edges so "
			            "rotated shots line up for stitching; 'unwrap' is its "
			            "inverse (stretches the sides, flattens an already-"
			            "cylindrical image). field_of_view sets the focal "
			            "length: f = (extent / 2) / tan(fov / 2).",
			inputs=[
				NPArray.Input("map_x",
					tooltip="Incoming source-x coordinate map to distort. Start the "
					        "chain with 'CV Identity Map'; further remap nodes "
					        "compose onto it."),
				NPArray.Input("map_y",
					tooltip="Incoming source-y coordinate map (must share map_x's "
					        "shape - both come from the same chain)."),
				io.Float.Input("field_of_view", default=90.0, min=10.0, max=160.0,
					step=1.0, tooltip="Horizontal (or vertical, see axis) field "
					                  "of view in degrees; wider = stronger "
					                  "curvature."),
				io.Combo.Input("direction",
					options=["wrap around the cylinder (pano pre-warp)",
					         "unwrap the cylinder (inverse)"],
					default="wrap around the cylinder (pano pre-warp)",
					tooltip="wrap curves a flat image onto a cylinder (panorama "
					        "pre-warp); unwrap is the inverse (flattens it back)."),
				io.Combo.Input("axis",
					options=["vertical axis (curves the left/right edges)",
					         "horizontal axis (curves the top/bottom edges)"],
					default="vertical axis (curves the left/right edges)",
					tooltip="Cylinder orientation: vertical curves the left/right "
					        "edges, horizontal curves the top/bottom edges.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="map_x"),
				NPArray.Output(display_name="map_y"),
			],
		)

	@classmethod
	def execute(cls, map_x, map_y, field_of_view, direction,
	            axis="vertical axis (curves the left/right edges)") -> io.NodeOutput:
		mx, my = _map_pair(map_x, map_y)
		h, w = mx.shape
		swap = axis.startswith("horizontal")
		if swap:
			mx, my = my, mx
			w, h = h, w
		cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
		f = cx / np.tan(np.radians(field_of_view) / 2.0)
		u = mx - cx
		if direction.startswith("wrap"):
			# the output pixel column is the cylinder angle; clip so tan stays finite
			theta = np.clip(u / f, -1.55, 1.55)
			new_x = cx + f * np.tan(theta)
			new_y = cy + (my - cy) / np.cos(theta)
		else:
			new_x = cx + f * np.arctan(u / f)
			new_y = cy + (my - cy) * f / np.sqrt(f * f + u * u)
		if swap:
			new_x, new_y = new_y, new_x
		return io.NodeOutput(new_x.astype(np.float32), new_y.astype(np.float32))


class PinchStretchMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PinchStretchMap",
			display_name="CV Pinch/Stretch Map",
			category=CATEGORY,
			description="Per-axis pinch or stretch anchored at the frame "
			            "borders (the edges never move): coordinates are "
			            "remapped by s' = 1 - (1 - |s|)^power on each side of "
			            "the center. power > 1 compresses the image content "
			            "toward the center (pinch), power < 1 spreads it "
			            "toward the edges (stretch), 1 = no change. The two "
			            "axes are independent - e.g. squeeze only x. An off-"
			            "center pivot makes the effect asymmetric.",
			inputs=[
				NPArray.Input("map_x",
					tooltip="Incoming source-x coordinate map to distort. Start the "
					        "chain with 'CV Identity Map'; further remap nodes "
					        "compose onto it."),
				NPArray.Input("map_y",
					tooltip="Incoming source-y coordinate map (must share map_x's "
					        "shape - both come from the same chain)."),
				io.Float.Input("power_x", default=1.5, min=0.05, max=20.0, step=0.05,
					tooltip="> 1 pinch, < 1 stretch, 1 = identity."),
				io.Float.Input("power_y", default=1.5, min=0.05, max=20.0, step=0.05,
					tooltip="> 1 pinch, < 1 stretch, 1 = identity."),
				io.Float.Input("center_x", default=0.5, min=0.0, max=1.0, step=0.01,
					tooltip="Pivot as a fraction of the width.",
					optional=True, advanced=True),
				io.Float.Input("center_y", default=0.5, min=0.0, max=1.0, step=0.01,
					tooltip="Pivot as a fraction of the height.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="map_x"),
				NPArray.Output(display_name="map_y"),
			],
		)

	@staticmethod
	def _pinch_axis(m, extent, center, power):
		"""Border-anchored power curve; coordinates outside [0, extent] are
		left untouched (they are already sampling the border fill)."""
		neg = max(center, 1e-6)
		pos = max(1.0 - center, 1e-6)
		n = m / max(extent, 1.0) - center
		n = np.where(n < 0, n / neg, n / pos)  # each side spans [-1, 1]
		inside = np.abs(n) <= 1.0
		bent = np.sign(n) * (1.0 - np.power(1.0 - np.minimum(np.abs(n), 1.0), power))
		n = np.where(inside, bent, n)
		n = np.where(n < 0, n * neg, n * pos) + center
		return (n * max(extent, 1.0)).astype(np.float32)

	@classmethod
	def execute(cls, map_x, map_y, power_x, power_y,
	            center_x=0.5, center_y=0.5) -> io.NodeOutput:
		mx, my = _map_pair(map_x, map_y)
		h, w = mx.shape
		return io.NodeOutput(cls._pinch_axis(mx, w - 1, center_x, power_x),
		                     cls._pinch_axis(my, h - 1, center_y, power_y))


class WaveMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_WaveMap",
			display_name="CV Wave Map",
			category=CATEGORY,
			description="Sinusoidal displacement: each row is shifted "
			            "horizontally by amplitude_x * sin(2*pi * y / "
			            "wavelength_x + phase) and each column vertically by "
			            "the same construction on x - the classic flag/water "
			            "ripple. Set an amplitude to 0 to disable that axis.",
			inputs=[
				NPArray.Input("map_x",
					tooltip="Incoming source-x coordinate map to distort. Start the "
					        "chain with 'CV Identity Map'; further remap nodes "
					        "compose onto it."),
				NPArray.Input("map_y",
					tooltip="Incoming source-y coordinate map (must share map_x's "
					        "shape - both come from the same chain)."),
				io.Float.Input("amplitude_x", default=8.0, min=0.0, max=10000.0,
					step=0.5, tooltip="Horizontal shift in pixels (driven by y)."),
				io.Float.Input("wavelength_x", default=64.0, min=0.1, max=100000.0,
					step=1.0, tooltip="Vertical period in pixels of the "
					                  "horizontal wave."),
				io.Float.Input("amplitude_y", default=0.0, min=0.0, max=10000.0,
					step=0.5, tooltip="Vertical shift in pixels (driven by x)."),
				io.Float.Input("wavelength_y", default=64.0, min=0.1, max=100000.0,
					step=1.0, tooltip="Horizontal period in pixels of the "
					                  "vertical wave."),
				io.Float.Input("phase", default=0.0, min=-360.0, max=360.0,
					step=1.0, tooltip="Phase offset in degrees - animate it to "
					                  "make the wave travel.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="map_x"),
				NPArray.Output(display_name="map_y"),
			],
		)

	@classmethod
	def execute(cls, map_x, map_y, amplitude_x, wavelength_x,
	            amplitude_y, wavelength_y, phase=0.0) -> io.NodeOutput:
		mx, my = _map_pair(map_x, map_y)
		ph = np.radians(phase)
		new_x = mx + amplitude_x * np.sin(2.0 * np.pi * my / wavelength_x + ph)
		new_y = my + amplitude_y * np.sin(2.0 * np.pi * mx / wavelength_y + ph)
		return io.NodeOutput(new_x.astype(np.float32), new_y.astype(np.float32))


class DisplacementMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DisplacementMap",
			display_name="CV Displacement Map",
			category=CATEGORY,
			description="Shifts each coordinate by the brightness of a "
			            "displacement image: offset = (gray - 0.5) * 2 * "
			            "strength, so mid-gray moves nothing, white pulls from "
			            "+strength pixels away and black from -strength. Feed "
			            "a blurred mask, noise, a depth map or the image's own "
			            "luminance (glass/heat-haze refraction). The "
			            "displacement image is converted to grayscale and "
			            "resized to the map size automatically.",
			inputs=[
				NPArray.Input("map_x",
					tooltip="Incoming source-x coordinate map to distort. Start the "
					        "chain with 'CV Identity Map'; further remap nodes "
					        "compose onto it."),
				NPArray.Input("map_y",
					tooltip="Incoming source-y coordinate map (must share map_x's "
					        "shape - both come from the same chain)."),
				polytype.poly_input("displacement",
					tooltip="Gray or color image; uint8/uint16 are scaled to "
					        "0-1, float is used as-is."),
				io.Float.Input("strength_x", default=16.0, min=-10000.0, max=10000.0,
					step=0.5, tooltip="Maximum horizontal shift in pixels "
					                  "(negative inverts the direction)."),
				io.Float.Input("strength_y", default=16.0, min=-10000.0, max=10000.0,
					step=0.5, tooltip="Maximum vertical shift in pixels "
					                  "(negative inverts the direction)."),
			],
			outputs=[
				NPArray.Output(display_name="map_x"),
				NPArray.Output(display_name="map_y"),
			],
		)

	@classmethod
	def execute(cls, map_x, map_y, displacement, strength_x, strength_y) -> io.NodeOutput:
		mx, my = _map_pair(map_x, map_y)
		displacement, _ = polytype.resolve(displacement)
		disp = np.asarray(displacement)
		if disp.ndim == 3:
			code = cv2.COLOR_BGRA2GRAY if disp.shape[2] == 4 else cv2.COLOR_BGR2GRAY
			disp = cv2.cvtColor(disp, code)
		if disp.ndim != 2 or disp.size == 0:
			raise ValueError(f"displacement must be a non-empty image, got shape "
			                 f"{np.asarray(displacement).shape}.")
		if disp.dtype == np.uint8:
			disp = disp.astype(np.float32) / 255.0
		elif disp.dtype == np.uint16:
			disp = disp.astype(np.float32) / 65535.0
		else:
			disp = disp.astype(np.float32)
		if disp.shape != mx.shape:
			disp = cv2.resize(disp, (mx.shape[1], mx.shape[0]), interpolation=cv2.INTER_LINEAR)
		offset = (disp - 0.5) * 2.0
		return io.NodeOutput((mx + offset * strength_x).astype(np.float32),
		                     (my + offset * strength_y).astype(np.float32))


class FlowMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_FlowMap",
			display_name="CV Flow Map",
			category=CATEGORY,
			description="Adds a dense displacement field (e.g. optical flow) to "
			            "the remap coordinates: map_x += scale * flow_dx, "
			            "map_y += scale * flow_dy. Feed it the HxWx2 (dx, dy) "
			            "field from 'CV Optical Flow (Farneback)' (or any "
			            "vector field) to warp an image ALONG that motion with "
			            "the low-level cv2_remap. 'scale' applies only a fraction "
			            "of the motion - the key to optical-flow frame "
			            "interpolation: because remap samples BACKWARD, scale = "
			            "-t pulls a frame a fraction t of the way along its flow "
			            "toward the next frame (so the moving content lands at "
			            "the in-between position SHARP, not ghosted like a raw "
			            "cross-fade). scale = 0 is the identity, 1 applies the "
			            "full motion, values past 1 or negative extrapolate. "
			            "Like every remap node it rewrites an incoming map, so "
			            "it composes with lens/wave/homography distortions in one "
			            "resampling pass.",
			inputs=[
				NPArray.Input("map_x",
					tooltip="Incoming source-x coordinate map to displace. Start the "
					        "chain with 'CV Identity Map'; further remap nodes "
					        "compose onto it."),
				NPArray.Input("map_y",
					tooltip="Incoming source-y coordinate map (must share map_x's "
					        "shape - both come from the same chain)."),
				NPArray.Input("flow",
					tooltip="HxWx2 float (dx, dy) displacement per pixel, e.g. the "
					        "'flow' output of 'CV Optical Flow (Farneback)'. "
					        "Resized to the map if the sizes differ."),
				io.Float.Input("scale", default=1.0, min=-100.0, max=100.0, step=0.05,
					tooltip="Multiplies the flow before adding it. 1 = full motion, "
					        "0 = identity, -t = sample a fraction t of the way back "
					        "along the motion (frame interpolation at time t), "
					        "negative = reverse the motion."),
			],
			outputs=[
				NPArray.Output(display_name="map_x"),
				NPArray.Output(display_name="map_y"),
			],
		)

	@classmethod
	def execute(cls, map_x, map_y, flow, scale) -> io.NodeOutput:
		mx, my = _map_pair(map_x, map_y)
		f = np.asarray(flow, np.float32)
		if f.ndim != 3 or f.shape[2] != 2 or f.size == 0:
			raise ValueError(f"flow must be a non-empty HxWx2 (dx, dy) field, got "
			                 f"shape {np.asarray(flow).shape}.")
		if f.shape[:2] != mx.shape:
			f = cv2.resize(f, (mx.shape[1], mx.shape[0]), interpolation=cv2.INTER_LINEAR)
		return io.NodeOutput((mx + scale * f[..., 0]).astype(np.float32),
		                     (my + scale * f[..., 1]).astype(np.float32))


class HomographyMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_HomographyMap",
			display_name="CV Homography Map",
			category=CATEGORY,
			description="Applies a 3x3 projective transform to the map "
			            "coordinates - cv2.warpPerspective expressed as a "
			            "remap, so a perspective change can be CHAINED with "
			            "lens/wave/pinch distortions in one resampling pass. "
			            "With invert enabled (default) feed the usual source-"
			            "to-destination homography ('CV Find Homography', "
			            "getPerspectiveTransform); remap needs the opposite "
			            "direction and the node inverts it for you.",
			inputs=[
				NPArray.Input("map_x",
					tooltip="Incoming source-x coordinate map to distort. Start the "
					        "chain with 'CV Identity Map'; further remap nodes "
					        "compose onto it."),
				NPArray.Input("map_y",
					tooltip="Incoming source-y coordinate map (must share map_x's "
					        "shape - both come from the same chain)."),
				NPArray.Input("matrix", tooltip="3x3 homography."),
				io.Boolean.Input("invert", default=True,
					tooltip="Invert the matrix (on for src->dst homographies; "
					        "off if you already have the dst->src mapping)."),
			],
			outputs=[
				NPArray.Output(display_name="map_x"),
				NPArray.Output(display_name="map_y"),
			],
		)

	@classmethod
	def execute(cls, map_x, map_y, matrix, invert) -> io.NodeOutput:
		mx, my = _map_pair(map_x, map_y)
		hm = np.asarray(matrix, np.float64)
		if hm.shape != (3, 3):
			raise ValueError(f"matrix must be 3x3, got shape {hm.shape}.")
		if invert:
			try:
				hm = np.linalg.inv(hm)
			except np.linalg.LinAlgError as e:
				raise ValueError("The homography is not invertible (degenerate "
				                 "matrix) - did detection fail upstream?") from e
		xw = hm[0, 0] * mx + hm[0, 1] * my + hm[0, 2]
		yw = hm[1, 0] * mx + hm[1, 1] * my + hm[1, 2]
		ww = hm[2, 0] * mx + hm[2, 1] * my + hm[2, 2]
		ww = np.where(np.abs(ww) < 1e-9, 1e-9, ww)
		return io.NodeOutput((xw / ww).astype(np.float32),
		                     (yw / ww).astype(np.float32))


class PolynomialRadialMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_PolynomialRadialMap",
			display_name="CV Polynomial Radial Map",
			category=CATEGORY,
			description="Radial polynomial warp: the distance r of each pixel "
			            "from the center is scaled by P(r) = c0 + c1*r + c2*r² + "
			            "... + cn*r^n (coefficients are an NPARRAY input). "
			            "r' = r * P(r) gives the sampling radius for that pixel. "
			            "A single coefficient [1.0] is the identity; the three-"
			            "coefficient set [1+s, -2s, s] reproduces the radial-"
			            "extrapolate formula r' = r * (1 + s*(1-r)²), where s>0 "
			            "pushes outer content outward (magnifies the center) and "
			            "s<0 pinches toward it. Coefficients are a data input, so "
			            "the SAME array can feed multiple nodes and branches. "
			            "Chains with other remap generators (Identity Map -> this "
			            "-> Wave Map -> ... -> cv2_remap) for one-pass resampling.",
			inputs=[
				NPArray.Input("map_x",
					tooltip="Incoming source-x coordinate map. Start the chain "
					        "with 'CV Identity Map'; further remap nodes "
					        "compose onto it."),
				NPArray.Input("map_y",
					tooltip="Incoming source-y coordinate map (same shape as "
					        "map_x - both come from the same chain)."),
				NPArray.Input("coeffs",
					tooltip="1D polynomial coefficients [c0, c1, c2, ..., cn]. "
					        "P(r) = c0 + c1*r + c2*r² + ... + cn*r^n, and "
					        "r' = r * P(r). For the radial-extrapolate formula "
					        "use [1+s, -2s, s] (s = strength). "
					        "Create the array with 'CV Scalar', e.g. "
					        "'(1.3, -0.6, 0.3)' for s=0.3."),
				io.Float.Input("center_x", default=0.5, min=0.0, max=1.0,
					step=0.01,
					tooltip="Center of the radial effect as a fraction of "
					        "the image width.",
					optional=True, advanced=True),
				io.Float.Input("center_y", default=0.5, min=0.0, max=1.0,
					step=0.01,
					tooltip="Center of the radial effect as a fraction of "
					        "the image height.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="map_x"),
				NPArray.Output(display_name="map_y"),
			],
		)

	@classmethod
	def execute(cls, map_x, map_y, coeffs,
	            center_x=0.5, center_y=0.5) -> io.NodeOutput:
		mx, my = _map_pair(map_x, map_y)
		h, w = mx.shape
		c = np.asarray(coeffs, np.float32).ravel()
		if c.ndim != 1 or c.size == 0:
			raise ValueError(
				f"coeffs must be a non-empty 1D array, got shape "
				f"{np.asarray(coeffs).shape}."
			)
		cx, cy = _center(mx.shape, center_x, center_y)
		norm = max(min(cx, (w - 1) - cx, cy, (h - 1) - cy), 1.0)
		dx, dy = mx - cx, my - cy
		r = np.sqrt(dx * dx + dy * dy) / norm
		P = np.zeros_like(r, dtype=np.float32)
		for i, ci in enumerate(c):
			P += ci * (r ** i)
		P = np.maximum(P, 1e-6)
		scale = np.where(r > 1e-9, P, c[0] if c[0] > 0 else 1.0)
		return io.NodeOutput((cx + dx * scale).astype(np.float32),
		                     (cy + dy * scale).astype(np.float32))


class RelativeIdentityMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_RelativeIdentityMap",
			display_name="CV Relative Identity Map",
			category=CATEGORY,
			description="Like 'CV Identity Map' but emits (0, 0) instead of "
			            "(x, y) - the starting point for a map chain that will be "
			            "sampled with WARP_RELATIVE_MAP (where cv2.remap interprets "
			            "the maps as relative offsets dx, dy instead of absolute "
			            "source coordinates). Feed it through the same distortion "
			            "nodes (wave, displacement, flow...); each adds offsets. "
			            "Finish with cv2_remap(flags=INTER_LINEAR | "
			            "WARP_RELATIVE_MAP).",
			inputs=[
				polytype.poly_input("image", tooltip="Only its width/height are used."),
			],
			outputs=[
				NPArray.Output(display_name="map_x",
					tooltip="HxW float32 zero-filled map (relative dx)."),
				NPArray.Output(display_name="map_y",
					tooltip="HxW float32 zero-filled map (relative dy)."),
			],
		)

	@classmethod
	def execute(cls, image) -> io.NodeOutput:
		arr = np.asarray(polytype.resolve(image)[0])
		if arr.ndim < 2 or arr.size == 0:
			raise ValueError(f"image must be a non-empty HxW(xC) array, got shape {arr.shape}.")
		h, w = arr.shape[:2]
		return io.NodeOutput(np.zeros((h, w), dtype=np.float32),
		                     np.zeros((h, w), dtype=np.float32))


class VectorDisplacementMap(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_VectorDisplacementMap",
			display_name="CV Vector Displacement Map",
			category=CATEGORY,
			description="Adds a per-pixel 2-D (dx, dy) displacement from a "
			            "multi-channel source to the coordinate maps - the key "
			            "difference from the scalar 'DisplacementMap' (which "
			            "drives both axes from the same grayscale). Connect a "
			            "normal map (R=dx, G=dy), Sobel gradients, optical flow, "
			            "or any multi-channel image to 'vector_field'; or wire "
			            "individual dx/dy arrays directly. uint8 inputs are "
			            "mapped to [-1, 1] so mid-gray = no displacement; "
			            "float inputs are used as raw pixel-offset values. "
			            "Designed for use with WARP_RELATIVE_MAP.",
			inputs=[
				NPArray.Input("map_x",
					tooltip="Incoming source-x coordinate map. Start the chain with "
					        "'CV Identity Map' or 'CV Relative Identity Map'."),
				NPArray.Input("map_y",
					tooltip="Incoming source-y coordinate map (same shape as map_x)."),
				polytype.poly_input("vector_field",
					tooltip="Multi-channel displacement source: uses ch0 as dx, "
					        "ch1 as dy. HxWx2 (direct vector field), HxWx3/4 "
					        "(normal map: R->dx, G->dy), or HxW (same for both "
					        "axes). uint8 mapped to [-1,1]; float used as-is.",
					optional=True),
				NPArray.Input("dx",
					tooltip="Per-pixel X displacement (overrides vector_field "
					        "when connected). Single-channel HxW.",
					optional=True),
				NPArray.Input("dy",
					tooltip="Per-pixel Y displacement (overrides vector_field "
					        "when connected). Single-channel HxW.",
					optional=True),
				io.Float.Input("strength_x", default=10.0, min=-10000.0, max=10000.0,
					step=0.5, tooltip="Multiplier for the X displacement in pixels."),
				io.Float.Input("strength_y", default=10.0, min=-10000.0, max=10000.0,
					step=0.5, tooltip="Multiplier for the Y displacement in pixels."),
			],
			outputs=[
				NPArray.Output(display_name="map_x",
					tooltip="Displaced source-x coordinate map."),
				NPArray.Output(display_name="map_y",
					tooltip="Displaced source-y coordinate map."),
			],
		)

	@classmethod
	def _normalize(cls, arr: np.ndarray) -> np.ndarray:
		"""Map uint8 [0,255] → [-1,1]; leave float as-is."""
		if arr.dtype == np.uint8:
			return (arr.astype(np.float32) / 127.5) - 1.0
		return arr.astype(np.float32)

	@classmethod
	def execute(cls, map_x, map_y, strength_x, strength_y,
	            vector_field=None, dx=None, dy=None) -> io.NodeOutput:
		mx, my = _map_pair(map_x, map_y)
		if dx is not None and dy is not None:
			dx_arr = np.asarray(dx)
			dy_arr = np.asarray(dy)
		elif vector_field is not None:
			vf = np.asarray(polytype.resolve(vector_field)[0])
			if vf.ndim not in (2, 3) or vf.size == 0:
				raise ValueError(
					f"vector_field must be HxW or HxWx(2+), got shape {vf.shape}."
				)
			if vf.ndim == 2:
				dx_arr = dy_arr = vf
			elif vf.shape[2] >= 2:
				dx_arr = vf[..., 0]
				dy_arr = vf[..., 1]
			else:
				raise ValueError(
					f"vector_field needs at least 2 channels, got {vf.shape[2]}."
				)
		else:
			raise ValueError(
				"Either connect 'vector_field' or both 'dx' and 'dy'."
			)
		dx_arr = cls._normalize(dx_arr)
		dy_arr = cls._normalize(dy_arr)
		if dx_arr.shape != mx.shape or dy_arr.shape != mx.shape:
			dx_arr = cv2.resize(dx_arr, (mx.shape[1], mx.shape[0]),
			                    interpolation=cv2.INTER_LINEAR)
			dy_arr = cv2.resize(dy_arr, (mx.shape[1], mx.shape[0]),
			                    interpolation=cv2.INTER_LINEAR)
		return io.NodeOutput((mx + dx_arr * strength_x).astype(np.float32),
		                     (my + dy_arr * strength_y).astype(np.float32))


NODES = [IdentityMap, RelativeIdentityMap, RadialLensMap, CylinderMap,
         PinchStretchMap, WaveMap, DisplacementMap, FlowMap, HomographyMap,
         PolynomialRadialMap, VectorDisplacementMap]
