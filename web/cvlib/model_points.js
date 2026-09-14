// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Parsing for the "CV Annotate Points On Model" widget value.
//
// The `points` STRING widget is the source of truth for the 3D annotation
// viewer, and it is edited from three directions: the viewer writes it on every
// click, ComfyUI restores it from a saved workflow, and the user types into it
// by hand. Only the first of those produces well-formed JSON at all times - a
// half-typed value has to be tolerated, not thrown away - so the parse is
// separated from the viewer here, where it can be tested under node.
//
// Contract: never raise, always report what happened. `status` is null when the
// value is clean, and a short human string otherwise; `points` is whatever
// survived, so the viewer can show a partially-typed list instead of blanking.

/** Is this one anchor - a triple of finite numbers? */
export function isAnchor(p) {
	return Array.isArray(p) && p.length === 3
		&& p.every((v) => typeof v === "number" && Number.isFinite(v));
}

/**
 * @param {string} raw the widget text
 * @returns {{points: number[][], status: string|null, parsed: boolean}}
 *   `parsed` is false only when the text is not JSON yet (mid-edit).
 */
export function parseAnchors(raw) {
	const text = (raw ?? "").trim();
	if (!text) return { points: [], status: null, parsed: true };
	let data;
	try {
		data = JSON.parse(text);
	} catch {
		// mid-edit text is expected, not a failure: keep the old markers up
		return { points: [], status: "points: not valid JSON yet", parsed: false };
	}
	if (!Array.isArray(data))
		return { points: [], status: "points: expected a list", parsed: true };
	const points = data.filter(isAnchor).map((p) => [p[0], p[1], p[2]]);
	const status = points.length === data.length
		? null
		: `points: ${data.length - points.length} of ${data.length} entries are `
		  + "not [x, y, z] numbers";
	return { points, status, parsed: true };
}
