// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Arc-length curve resampling for the correspondence editor's curve-pair tool -
// PURE MATH, no DOM, no ComfyUI imports. Deliberate, like warp_math.js:
// tests/smoke_test.py::test_curve_resample_js_matches_numpy runs THIS FILE
// under node.exe against a numpy reference, because uneven spacing here would
// be silent - the pairs it emits LOOK plausible while quietly weighting the
// fitted warp toward wherever the stroke was drawn slowly (pointermove events
// cluster where the hand hesitates).
//
// resampleCurve(pts, k): k points spaced UNIFORMLY ALONG THE ARC of the
// polyline pts ([[x, y], ...]), endpoints preserved exactly. Degenerate cases
// are all valid inputs (rule-4 spirit): fewer than 2 points or a zero-length
// stroke return k copies of the first point; duplicate consecutive points
// (zero-length segments) are stepped over.
//
// Call it in PIXEL coordinates, not normalized ones: normalized space is
// aspect-squashed, so "uniform" there would stretch the spacing along the
// image's long side.

export function resampleCurve(pts, k) {
	const n = Math.max(2, Math.round(k));
	if (!pts.length) return [];
	if (pts.length === 1) {
		return Array.from({ length: n }, () => [pts[0][0], pts[0][1]]);
	}
	const cum = [0];
	for (let i = 1; i < pts.length; i++) {
		const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
		cum.push(cum[i - 1] + d);
	}
	const total = cum[cum.length - 1];
	if (!(total > 0)) {
		return Array.from({ length: n }, () => [pts[0][0], pts[0][1]]);
	}
	const out = [];
	let seg = 0;
	for (let j = 0; j < n; j++) {
		const t = (j / (n - 1)) * total;
		// walk to the segment containing t; skip zero-length segments (the
		// strict < keeps the LAST segment reaching t, so a = 1 lands exactly on
		// the vertex and the final sample is the final input point, bit-exact)
		while (seg < pts.length - 2 && cum[seg + 1] < t) seg++;
		const span = cum[seg + 1] - cum[seg];
		const a = span > 0 ? (t - cum[seg]) / span : 0;
		out.push([pts[seg][0] + (pts[seg + 1][0] - pts[seg][0]) * a,
		          pts[seg][1] + (pts[seg + 1][1] - pts[seg][1]) * a]);
	}
	return out;
}
