// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Keeps widgets_values ROUND-TRIPPING on this pack's nodes.
//
// Why any of this is needed - and what the canonical array looks like - is in
// web/cvlib/widget_values.js. In short: the frontend writes the array indexed by
// the FULL widget list and reads it back counting only the PERSISTED widgets, so
// a `serialize:false` widget in the MIDDLE of the list (every composite CV_TUPLE
// input has two to four of them, right behind the schema widget - web/cv_tuple.js)
// shifts every value below it by one on load, undo, paste and node clone.
//
// The two hooks are per node instance, one is the exact inverse of the other:
//   serialize()  compacts the freshly written array (holes dropped), so what
//                lands in the .json is one entry per persisted widget - exactly
//                the shape the node had before its components were split, and
//                the shape tests/validate_workflows.py checks;
//   configure()  accepts a FULL-list array from a workflow saved earlier (or by
//                an unpatched frontend) and compacts it before litegraph reads
//                it, so no saved workflow has to be migrated by hand.
// Both are no-ops when the array is already canonical, so they stay inert if the
// frontend's own indexing is fixed upstream.
//
// Widgets this affects are ALWAYS pack-internal (composite components, flag
// toggles, editor surfaces), so the patch is limited to this pack's node classes
// rather than installed graph-wide.

import { app } from "../../scripts/app.js";
import { compactWidgetValues, hasWidgetValueHoles, normalizeWidgetValues }
	from "./cvlib/widget_values.js";

const PACK_NODE = /^(cv2_|CV_)/;

app.registerExtension({
	name: "opencv_comfyui.widget_values",

	nodeCreated(node) {
		// nodeCreated fires once per instance, but extensions can create nodes
		// during configure as well - guard so the wrappers never stack
		if (node._cvWidgetValuesPatched) return;
		if (!PACK_NODE.test(String(node.comfyClass ?? node.type ?? ""))) return;
		node._cvWidgetValuesPatched = true;

		// The widget list is inspected INSIDE the hooks, never captured here:
		// other extensions (flag lists, chart/editor surfaces, arity widgets)
		// add or hide widgets after this runs.
		const serialize = node.serialize;
		node.serialize = function () {
			const info = serialize.apply(this, arguments);
			const widgets = this.widgets ?? [];
			const values = info?.widgets_values;
			if (Array.isArray(values) && hasWidgetValueHoles(widgets, values))
				info.widgets_values = compactWidgetValues(widgets, values);
			return info;
		};

		const configure = node.configure;
		node.configure = function (info) {
			const values = info?.widgets_values;
			if (Array.isArray(values)) {
				const fixed = normalizeWidgetValues(this.widgets ?? [], values);
				// a COPY when it changed: `info` is the caller's serialized
				// state (the undo stack keeps reusing it), so it must not be
				// rewritten in place
				if (fixed !== values) info = { ...info, widgets_values: fixed };
			}
			return configure.call(this, info);
		};
	},
});
