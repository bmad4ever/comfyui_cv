// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Scene-depth packing for 'CV Preview 3D (Calibrated Camera)' - PURE MATH, no
// DOM, no ComfyUI imports, for the same reason web/cvlib/warp_math.js is:
// tests/smoke_test.py::test_depth_codec_js_matches_python runs THIS FILE under
// node and compares it against opencv_nodes/render3d.py, so the depth the
// shader occludes with is provably the depth the server occluded with.
//
// This directory (web/cvlib/) is not auto-loaded by ComfyUI - only top-level
// web/*.js are registered as extensions - so a module here is inert until
// something imports it.
//
// Why a packed RGB image at all: the node has to hand the browser a metric
// depth map, and the transports that look easier are all lossy. A 16-bit gray
// PNG is silently truncated to 8 bits by a canvas/WebGL upload (that would be
// 12 cm steps at 30 m - the occlusion edge would crawl), and an 8-bit gray one
// is worse. So render3d.encode_depth_png writes 24-bit fixed point over
// [0, zMax], big-endian across R/G/B, with code 0 reserved for "no
// measurement". That reservation is load-bearing: a stereo disparity map has
// holes, and a hole decoded as z=0 would erase the model wherever the matcher
// failed, so unknown decodes to +Infinity (infinitely far, never occludes).

export const DEPTH_MAX_CODE = 0xffffff;

// r/g/b are BYTES (0-255), matching how the PNG was written. Returns metric Z
// in the same units the encoder was given, or +Infinity for "unknown".
export function decodeDepth(r, g, b, zMax) {
	const code = (r << 16) | (g << 8) | b;
	if (code === 0) return Infinity;
	return ((code - 1) / (DEPTH_MAX_CODE - 1)) * zMax;
}

// The same expression as GLSL, for the post-process shader. `texel` is a vec4
// sampled from the packed texture (components in 0..1, so each is scaled back
// to a byte); the result is +Infinity-equivalent (a huge float) for unknown,
// which keeps the comparison in the shader branch-free.
export const DECODE_DEPTH_GLSL = `
float decodeDepth(vec4 texel, float zMax) {
	vec3 bytes = floor(texel.rgb * 255.0 + 0.5);
	float code = bytes.r * 65536.0 + bytes.g * 256.0 + bytes.b;
	if (code < 0.5) return 1.0e20;          // 0 = no measurement = infinitely far
	return ((code - 1.0) / ${DEPTH_MAX_CODE - 1}.0) * zMax;
}
`;

// Standard perspective depth-buffer linearization: a WebGL depth texture stores
// the window-space z (0..1) of the projection matrix, which is NOT linear in
// camera space. Returns positive camera-space distance. 1.0 (the cleared value)
// comes back as +Infinity so empty pixels never occlude anything.
export function linearizeDepth(d, near, far) {
	if (d >= 1.0) return Infinity;
	const ndc = 2.0 * d - 1.0;
	return (2.0 * near * far) / (far + near - ndc * (far - near));
}

export const LINEARIZE_DEPTH_GLSL = `
float linearizeDepth(float d, float near, float far) {
	if (d >= 1.0) return 1.0e20;
	float ndc = 2.0 * d - 1.0;
	return (2.0 * near * far) / (far + near - ndc * (far - near));
}
`;
