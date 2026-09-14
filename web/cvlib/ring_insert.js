// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Where a newly clicked vertex goes in a point ring - PURE MATH, no DOM, no
// ComfyUI imports, for the same reason web/cvlib/warp_math.js and
// depth_codec.js are: tests/smoke_test.py::test_ring_insert_js runs THIS FILE
// under node against a numpy reference, so the one part of the annotation
// surface that can be silently wrong (the INDEX a point lands at - the picture
// looks right either way, but the point ORDER is what every consumer of the
// 'points' output reads) is pinned.
//
// This directory (web/cvlib/) is not auto-loaded by ComfyUI - only top-level
// web/*.js are registered as extensions - so a module here is inert until
// something imports it.
//
// The rule: while the outline is still open (fewer than 3 points) a click
// APPENDS, because the click order IS the order the user means. Once it is
// closed, the outline is a RING and "the end of the list" is an arbitrary
// place on it - so the click goes into the edge it landed next to. That is
// what makes refining a shape local: without it, fixing one bad corner means
// deleting and re-clicking every point after it just to keep the ring's
// winding intact.

// Distance from p to the SEGMENT ab - not to the infinite line through a and
// b: a click past the end of an edge belongs to the edge it is actually
// beside, not to whichever line happens to point at it. Points are [x, y] in
// any single consistent space (the caller uses element-local CSS pixels, so
// the answer matches what the user SEES rather than the image's aspect).
export function segmentDistance(p, a, b) {
	const vx = b[0] - a[0], vy = b[1] - a[1];
	const len2 = vx * vx + vy * vy;
	// a degenerate edge (duplicated vertex) collapses to the distance to a
	const t = len2 > 0
		? Math.min(1, Math.max(0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / len2))
		: 0;
	return Math.hypot(p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy));
}

// Index of the CLOSED ring's edge nearest to p: the edge running from vertex j
// to vertex (j + 1) % n. Returns -1 when there is no ring yet (< 3 points),
// where appending is already the right order. Ties go to the lower index.
export function nearestEdge(pts, p) {
	if (!Array.isArray(pts) || pts.length < 3) return -1;
	let best = -1, bestD = Infinity;
	for (let j = 0; j < pts.length; j++) {
		const d = segmentDistance(p, pts[j], pts[(j + 1) % pts.length]);
		if (d < bestD) { best = j; bestD = d; }
	}
	return best;
}

// Where to splice a new vertex for a click at p. Note the closing edge
// (n-1 -> 0) yields n, i.e. a plain append - so a click just outside the last
// edge still behaves exactly the way it did before this rule existed.
// `append` (the ALT modifier upstream) forces that old behaviour everywhere.
export function insertionIndex(pts, p, append = false) {
	const n = Array.isArray(pts) ? pts.length : 0;
	if (append) return n;
	const edge = nearestEdge(pts, p);
	return edge >= 0 ? edge + 1 : n;
}
