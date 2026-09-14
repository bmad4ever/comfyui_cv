// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Which cv_flags widget gets the searchable list, and how that list RANKS -
// PURE, no DOM and no ComfyUI imports, for the same reason ring_insert.js and
// warp_math.js are: tests/smoke_test.py::test_flag_search_js runs THIS FILE
// under node against the real flag vocabulary, so the half that can be
// silently wrong (which option a query surfaces, and in what order) is pinned
// while the DOM half stays a real-UI check.
//
// This directory (web/cvlib/) is not auto-loaded by ComfyUI - only top-level
// web/*.js are registered as extensions - so a module here is inert until
// something imports it. web/cv_flags.js is the importer.

// Past this many flags, one toggle widget per flag makes the node taller than
// it is placeable ('CV Region Properties' has 25). The eight cv2 flag
// builders top out at 8 (CALIB_CB_ALL_FLAGS), so they keep their toggles and
// nothing about them changes.
export const LIST_THRESHOLD = 10;

export function useList(spec) {
	const flags = spec && Array.isArray(spec.flags) ? spec.flags : [];
	return flags.length > LIST_THRESHOLD;
}

// One query token against one label. A token matches as a SUBSEQUENCE, so
// "mnrct" finds every "min-area rect ..." row - which a substring filter
// cannot - and the bonuses are what keep the ranking sane:
//   +3 the matched char continues the previous match (contiguous run),
//   +2 it starts a word (position 0, or the char before it is not a letter or
//      a digit - the labels are full of spaces, hyphens and parentheses),
//   +1 otherwise.
// A small penalty on where the token first lands breaks ties toward a hit near
// the front of the label. Returns null when the token is not a subsequence.
function tokenScore(token, text) {
	let score = 0, at = 0, first = -1, prev = -2;
	for (const ch of token) {
		const hit = text.indexOf(ch, at);
		if (hit < 0) return null;
		if (first < 0) first = hit;
		if (hit === prev + 1) score += 3;
		else if (hit === 0 || !/[a-z0-9]/.test(text[hit - 1])) score += 2;
		else score += 1;
		prev = hit;
		at = hit + 1;
	}
	return score - 0.05 * Math.min(first, 20);
}

// Whole query against one label. Whitespace splits it into tokens and EVERY
// token must match ("rect clear" -> the centroid-clearance row), which is what
// makes a two-word memory of an option enough to find it. An empty query
// matches everything at score 0. Case-insensitive.
export function fuzzyScore(query, text) {
	const tokens = String(query ?? "").toLowerCase().trim().split(/\s+/)
		.filter(Boolean);
	if (!tokens.length) return 0;
	const hay = String(text ?? "").toLowerCase();
	let total = 0;
	for (const token of tokens) {
		const s = tokenScore(token, hay);
		if (s === null) return null;
		total += s;
	}
	return total;
}

// -> [{ item, index, score }] best first. The sort is STABLE on the declared
// index, so equally-good rows keep the order the Python group declares (which
// is the feature-matrix COLUMN order) and the list never reshuffles under the
// cursor.
export function fuzzyFilter(query, items) {
	const out = [];
	(items || []).forEach((item, index) => {
		const score = fuzzyScore(query, item);
		if (score !== null) out.push({ item, index, score });
	});
	out.sort((a, b) => (b.score - a.score) || (a.index - b.index));
	return out;
}
