// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Brown-Conrady inversion for 'CV Preview 3D (Calibrated Camera)' - PURE MATH,
// no DOM, no ComfyUI imports, for the same reason web/cvlib/warp_math.js and
// depth_codec.js are: tests/smoke_test.py::test_distort_math_js_matches_cv2 runs
// THIS FILE under node and compares it against cv2.undistortPoints.
//
// This directory (web/cvlib/) is not auto-loaded by ComfyUI - only top-level
// web/*.js are registered as extensions - so a module here is inert until
// something imports it.
//
// WHY IT EXISTS (the bug it fixes): the viewer renders the scene through an
// ideal pinhole camera into an OVERSCANNED target and then warps that render
// per-pixel in a fragment shader. Three.js knows nothing about the warp, so
// TransformControls raycasts the pointer through the PINHOLE frustum while the
// user is looking at the DISTORTED picture - the gizmo handle is drawn at one
// place and picked at another, the gap growing with radius. Feeding the
// raycaster a pointer pushed through the same map the shader uses closes it.
//
// The important property is not accuracy but AGREEMENT: the shader's inverse is
// a truncated fixed-point iteration, so it is not exactly cv2's answer at large
// |k|. Because the pointer runs the SAME iteration count over the SAME
// coefficients, the picked ray is the ray that produced the pixel under the
// cursor even where both are a poor inverse - which is why this works past the
// "low distortion" regime. The remaining limit is genuine: outside the
// overscanned render there is no pixel to pick (`inside` says so), and where the
// forward map stops being injective no inverse exists at all.

// Shared by the JS and the GLSL below so the two cannot drift apart. This is
// the cv2.undistortPoints iteration; cv2's own default is 5, the extra rounds
// are cheap here and tighten the agreement with cv2 at large distortion.
export const UNDISTORT_ITERATIONS = 8;

// How far the answer may re-project from the pixel it came from before that
// pixel counts as belonging to no ray at all, in IMAGE PIXELS (the caller scales
// the normalized residual by the focal length). Half a pixel is under what any
// consumer can see and orders of magnitude under the blow-up this guards.
export const UNDISTORT_PIXEL_TOL = 0.5;

// (xd, yd) are DISTORTED normalized camera coordinates: ((u, v) - (cx, cy)) / (fx, fy).
// dist is [k1, k2, p1, p2, k3]. Returns the ideal (pinhole) normalized coords
// plus `residual` = |forward(answer) - (xd, yd)| in those same normalized units.
//
// The residual is not diagnostics, it is the only thing that says the answer
// means anything. A calibration with a strong k3 makes r_d(r_u) turn over, and
// past that turning point the forward map covers NO output pixel: the fixed
// point has nothing to converge to and runs away - sometimes back INTO the
// frame, which is the dangerous case, because the runaway coordinate then
// indexes real geometry and paints a ghost. cv2 hides the same failure behind
// its 5-iteration truncation and its icdist < 0 bail; here the caller multiplies
// `residual` by the focal length and drops the pixel past UNDISTORT_PIXEL_TOL.
export function undistortNormalized(xd, yd, dist, iterations = UNDISTORT_ITERATIONS) {
	const [k1, k2, p1, p2, k3] = dist;
	let x = xd, y = yd;
	for (let i = 0; i < iterations; i++) {
		const r2 = x * x + y * y;
		const radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
		const tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
		const ty = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
		x = (xd - tx) / radial;
		y = (yd - ty) / radial;
	}
	// a diverged iterate carries Infinity/NaN into the residual, where every
	// comparison against the tolerance is false - it can never be accepted
	const back = distortNormalized(x, y, dist);
	return { x, y, residual: Math.hypot(back.x - xd, back.y - yd) };
}

// Forward Brown-Conrady - the model the iteration above inverts. Only used by
// the tests (as the round-trip reference) and available to any caller that needs
// to project a known 3D point into the distorted frame.
export function distortNormalized(x, y, dist) {
	const [k1, k2, p1, p2, k3] = dist;
	const r2 = x * x + y * y;
	const radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
	return {
		x: x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
		y: y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
	};
}

// The GLSL half, so the post-process shader and the pointer share one
// expression. Same trick as DECODE_DEPTH_GLSL in depth_codec.js.
export const UNDISTORT_GLSL = `
// .xy is the ideal (pinhole) normalized point, .z the residual of the forward
// model - the JS twin above returns exactly these two, and for the same reason.
vec3 undistortNormalized(vec2 xd, vec4 d1, float k3) {
	vec2 xu = xd;
	for (int i = 0; i < ${UNDISTORT_ITERATIONS}; i++) {
		float r2 = dot(xu, xu);
		float radial = 1.0 + r2 * (d1.x + r2 * (d1.y + r2 * k3));
		vec2 tang = vec2(
			2.0 * d1.z * xu.x * xu.y + d1.w * (r2 + 2.0 * xu.x * xu.x),
			d1.z * (r2 + 2.0 * xu.y * xu.y) + 2.0 * d1.w * xu.x * xu.y);
		xu = (xd - tang) / radial;
	}
	float r2 = dot(xu, xu);
	float radial = 1.0 + r2 * (d1.x + r2 * (d1.y + r2 * k3));
	vec2 back = xu * radial + vec2(
		2.0 * d1.z * xu.x * xu.y + d1.w * (r2 + 2.0 * xu.x * xu.x),
		d1.z * (r2 + 2.0 * xu.y * xu.y) + 2.0 * d1.w * xu.x * xu.y);
	return vec3(xu, length(back - xd));
}
`;

// Map a pointer from the DISPLAYED (distorted) canvas into the NDC of the
// OVERSCANNED pinhole render - i.e. the NDC the viewer's projection matrix
// produces, which is what THREE.Raycaster.setFromCamera expects.
//
// This is the fragment shader's own chain, in the same order and the same
// image-pixel units: display NDC -> output image px -> distorted normalized ->
// undistort -> overscan image px -> overscan NDC. `margin` is the viewer's
// overscan fraction per side (0 when distortion is off, and then the whole
// thing collapses to the identity - one code path, no special case).
//
// Returns { x, y, inside }; `inside` is false when the ray leaves the rendered
// target, where the shader writes transparent and there is nothing to pick - and
// equally when the inverse did not converge onto its own pixel (UNDISTORT_PIXEL_TOL),
// which is the same pixel the shader now discards.
export function displayToRenderNDC(ndcX, ndcY, opts) {
	const { K, width, height, margin = 0, dist = null,
		iterations = UNDISTORT_ITERATIONS } = opts;
	const px = (ndcX + 1) * 0.5 * width;
	const py = (1 - ndcY) * 0.5 * height;
	let ux = (px - K.cx) / K.fx;
	let uy = (py - K.cy) / K.fy;
	let solved = true;
	if (dist) {
		const u = undistortNormalized(ux, uy, dist, iterations);
		// negated so a NaN residual (an overflowed iterate) also reads as "no ray"
		solved = !!(u.residual * Math.min(K.fx, K.fy) <= UNDISTORT_PIXEL_TOL);
		ux = u.x; uy = u.y;
	}
	const wp = width * (1 + 2 * margin), hp = height * (1 + 2 * margin);
	const sx = K.fx * ux + (K.cx + width * margin);
	const sy = K.fy * uy + (K.cy + height * margin);
	return {
		x: 2 * sx / wp - 1,
		y: 1 - 2 * sy / hp,
		inside: solved && sx >= 0 && sx <= wp && sy >= 0 && sy <= hp,
	};
}
