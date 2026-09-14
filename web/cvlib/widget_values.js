// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// widgets_values round-trip repair for nodes that carry NON-PERSISTED widgets.
//
// A widget is kept out of the saved workflow by `widget.serialize = false`
// (persistence) - a different flag from `widget.options.serialize = false`,
// which only keeps it out of the API prompt. Both halves of the frontend agree
// on WHICH widgets to skip and disagree on HOW to index what is left
// (ComfyUI_frontend 1.48.7, src/lib/litegraph/src/LGraphNode.ts):
//
//   serialize()   o.widgets_values[i] = w.value    // i = index in the FULL list
//   configure()   w.value = info.widgets_values[i++]  // i counts KEPT widgets
//
// So the writer leaves a HOLE at every skipped widget (JSON.stringify renders a
// hole as null) while the reader expects those entries to be absent. One
// non-persisted widget in the middle of the list therefore shifts every value
// below it by one on the next load, undo or paste:
//
//   cv2_GaussianBlur, widgets  ksize | ksize.w | ksize.h | sigmaX | sigmaY | borderType | hint
//   saved                     "(9,9)"    null      null      0        0    "BORDER_DEFAULT" "ALGO_HINT_DEFAULT"
//   read back                 "(9,9)"     -         -       null     null      0                0
//
// This module is the repair, and it is deliberately format-restoring rather
// than clever: the CANONICAL array is one entry per PERSISTED widget, in widget
// order, which is exactly what a node without sub-widgets writes, what
// tests/validate_workflows.py checks against the schema, and what
// tests/run_workflows.py maps onto the schema positionally. Composite inputs
// (web/cv_tuple.js) therefore keep occupying ONE entry, spelled as the Python
// literal, as they did before their components were split into sub-widgets.
//
// Pure functions, no frontend imports: web/cv_widget_values.js does the
// patching, tests/smoke_test.py runs THIS file under node against the two
// LGraphNode algorithms above.

/** Widgets flagged `serialize = false` are not persisted in the workflow. */
function skipped(widget) {
	return !widget || widget.serialize === false;
}

function keptCount(widgets) {
	return widgets.reduce((n, w) => n + (skipped(w) ? 0 : 1), 0);
}

/**
 * Does `values` (as written by LGraphNode.serialize) contain holes?
 *
 * The writer stops at the last persisted widget, so a skipped widget below that
 * point - and only there - left a hole behind. False for an already-compact
 * array, which is what makes the patch a no-op once the frontend is fixed.
 */
export function hasWidgetValueHoles(widgets, values) {
	if (!Array.isArray(widgets) || !Array.isArray(values)) return false;
	return widgets.some((w, i) => skipped(w) && i < values.length);
}

/**
 * FULL-list-indexed values -> one entry per persisted widget.
 *
 * A widget beyond the end of `values` contributes NOTHING (no padding): a
 * trailing optional widget that was never saved has to stay at its schema
 * default, exactly as the frontend materializes it.
 */
export function compactWidgetValues(widgets, values) {
	const out = [];
	widgets.forEach((w, i) => {
		if (skipped(w) || i >= values.length) return;
		out.push(i in values ? values[i] : null);
	});
	return out;
}

/**
 * Is `values` a legacy FULL-list array (one entry per widget, persisted or not)
 * rather than the canonical compact one?
 *
 * Two shapes reach this from workflows already on disk: holes written as null
 * by the serializer above, and the empty string a DOM widget serialized as
 * while it was persisted by accident (`options.serialize` set, `serialize` not).
 * Both are recognised by requiring every slot owned by a skipped widget to be
 * empty - so an array that is merely longer than the current schema (a widget
 * the node has since lost) is left alone instead of being re-indexed on a
 * guess.
 *
 * The LENGTH test is what makes this safe, and it is also the deliberate limit:
 * "" is this pack's "use the OpenCV default" sentinel, so a canonical array can
 * legitimately hold empty entries at exactly the slots a full-list array leaves
 * empty. Only "more entries than there are persisted widgets" separates the two
 * with certainty, so a full-list array that stops early (the node gained widgets
 * after it was saved) is NOT recognised - reading it the old way is the
 * pre-existing shift, whereas re-indexing a canonical array would be new data
 * loss. Pinned by test_widget_values_js_round_trip.
 */
export function isFullListWidgetValues(widgets, values) {
	if (!Array.isArray(widgets) || !Array.isArray(values)) return false;
	if (!widgets.some((w) => skipped(w))) return false;
	if (values.length <= keptCount(widgets)) return false;
	if (values.length > widgets.length) return false;
	return widgets.every((w, i) => {
		if (!skipped(w) || i >= values.length) return true;
		const v = values[i];
		return v === null || v === undefined || v === "";
	});
}

/** `values` in canonical form: compacted when it is a legacy full-list array. */
export function normalizeWidgetValues(widgets, values) {
	return isFullListWidgetValues(widgets, values)
		? compactWidgetValues(widgets, values)
		: values;
}
