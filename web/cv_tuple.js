// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Composite widget for CV_TUPLE - one cv2 Size / Point / Rect / TermCriteria
// as ONE input.
//
// The Python side (opencv_nodes/tuples.py CvTuple.Input) publishes the input
// with an options dict carrying a `cv_tuple` spec:
//   { components: [ { name, default, min, max, is_float, tooltip, options? } ] }
// CV_TUPLE is not a core widget type, so without this file the input would
// render as a bare socket with no way to type a value at all - `getCustomWidgets`
// is what registers it (widgetStore.registerCustomWidgets, keyed by io_type),
// and litegraphService.addInputWidget then pairs it with a socket because the
// widget's options never set `socketless`.
//
// Rendering follows web/cv_flags.js, and the value contract is the same trick:
// the SCHEMA widget is a plain `string` holding the whole composite as its
// Python literal ("(640, 480)"), and the per-component number/combo widgets are
// appended right behind it with serialize:false. That buys three things at once:
// - widgets_values stays ONE entry per composite param, and its spelling is
//   byte-identical to the pre-split literal, so a workflow saved before the
//   components were split still loads. NOTE: this only holds because
//   web/cv_widget_values.js repairs the array - the frontend writes
//   widgets_values by FULL-widget-list index and reads it back counting only the
//   persisted widgets, so a serialize:false widget anywhere but the TAIL (which
//   is what these components are) leaves a hole that shifts every value below
//   the composite. See web/cvlib/widget_values.js;
// - the string widget is a CORE widget type, so it renders in the classic
//   canvas AND under Modern Node Design - unlike core's own BoundingBoxWidget,
//   whose drawWidget is drawVueOnlyWarning;
// - a VISIBLE schema widget stays at the input's own index, which is what keeps
//   the input-slot row on the right row (the frontend lays slot rows out from
//   visible widgets while slot ownership follows the schema widget - see the
//   comment at the top of cv_flags.js).
//
// While the input is linked the upstream node decides the value, so the
// component widgets are DISABLED - never hidden, because a hidden widget drops
// out of getLayoutWidgets() and shifts every slot row below it.

import { app } from "../../scripts/app.js";

// A component value -> its Python-literal spelling. Dropdown components carry a
// NAME (TermCriteria's type flag), which has to be quoted or the literal is not
// parseable; JSON's double quotes are valid Python.
function literalPart(value, comp) {
	if (comp.options) return JSON.stringify(String(value ?? ""));
	const n = Number(value);
	if (!Number.isFinite(n)) return "0";
	return comp.is_float ? String(n) : String(Math.round(n));
}

function compose(components, values) {
	return "(" + components.map((c, i) => literalPart(values[i], c)).join(", ") + ")";
}

// "(640, 480)" / "[640, 480]" / "5" -> one value per component. A bare number
// broadcasts (a 5x5 kernel), matching tuples.parse() on the Python side; a
// value that cannot be read at all leaves the components untouched, so a
// hand-edited or legacy string is never silently rewritten.
function decompose(text, components) {
	const raw = String(text ?? "").trim().replace(/^[([]|[)\]]$/g, "");
	if (!raw) return null;
	const parts = [];
	let depth = 0, cur = "", quote = null;
	for (const ch of raw) {
		if (quote) {
			cur += ch;
			if (ch === quote) quote = null;
			continue;
		}
		if (ch === '"' || ch === "'") { quote = ch; cur += ch; continue; }
		if (ch === "(" || ch === "[") depth++;
		if (ch === ")" || ch === "]") depth--;
		if (ch === "," && depth === 0) { parts.push(cur); cur = ""; continue; }
		cur += ch;
	}
	parts.push(cur);
	let vals = parts.map((p) => p.trim()).filter((p) => p.length);
	if (!vals.length) return null;
	if (vals.length === 1 && components.length > 1) {
		vals = new Array(components.length).fill(vals[0]);
	}
	if (vals.length !== components.length) return null;
	return vals.map((v, i) => {
		const comp = components[i];
		if (comp.options) return v.replace(/^['"]|['"]$/g, "");
		const n = Number(v);
		return Number.isFinite(n) ? (comp.is_float ? n : Math.round(n)) : comp.default;
	});
}

// 'CV Tuple' emits 2, 3 or 4 components, but its schema has to declare all four
// (a node's widget list is static). Hide the ones the current `count` does not
// use, so the node shows exactly the value it produces.
//
// Two rules make this safe. HIDING SHIFTS SLOT ROWS: the frontend lays widget
// input-slot rows out from VISIBLE widgets while slot ownership follows the
// schema widget (see the comment at the top of web/cv_flags.js), so only the
// TAIL widgets may be hidden - `c` and `d` are the last two, and nothing below
// them can be displaced. And a CONNECTED component is never hidden: its link
// would otherwise be drawn to a stale row, and silently dropping a wire the
// user made is worse than one extra row.
const TUPLE_NODE = "CV_Tuple";
const ARITY_WIDGETS = { c: 3, d: 4 };

function setupArity(node) {
	const byName = (n) => node.widgets?.find((w) => w.name === n);
	const count = byName("count");
	if (!count) return;
	const linked = (name) =>
		!!node.inputs?.some((i) => i.name === name && i.link != null);

	const apply = (resize) => {
		const n = Number(count.value) || 2;
		let changed = false;
		for (const [name, needs] of Object.entries(ARITY_WIDGETS)) {
			const w = byName(name);
			if (!w) continue;
			const hidden = n < needs && !linked(name);
			if (!!w.hidden !== hidden) changed = true;
			w.hidden = hidden;
			w.options.hidden = hidden;
		}
		// only shrink/grow on a real change: configure() applies the SAVED size
		// after this runs on load, and resizing there would discard it
		if (changed && resize && node.computeSize) {
			node.setSize([node.size[0], node.computeSize()[1]]);
		}
		node.setDirtyCanvas?.(true, true);
	};

	const cb = count.callback;
	count.callback = function () {
		const r = cb?.apply(this, arguments);
		apply(true);
		return r;
	};
	apply(false);
	(node._cvTupleSync ??= []).push(() => apply(false));
}

// 'CV Tuple' also has a `dtype` combo, and until the value is executed it is the
// ONLY thing that says whether the components are pixels or world units. The
// component widgets are created once from the FLOAT,INT spec (fractional step,
// 4 decimals), so on an int tuple the arrows crawled in hundredths and the field
// showed "3.0000" for a kernel size that cv2 will only ever see as 3.
//
// So mirror the combo onto the widgets: int -> step 1, no decimals, and the
// current values ROUNDED (execute() casts anyway, so rounding here only makes
// the node show what it already emits); float -> the spec's own step and
// precision back. `round` is what makes litegraph snap dragged/typed values, and
// the callback wrapper re-rounds after a direct text entry.
const COMPONENT_WIDGETS = ["a", "b", "c", "d"];

function setupDtype(node) {
	const byName = (n) => node.widgets?.find((w) => w.name === n);
	const dtype = byName("dtype");
	if (!dtype) return;

	const isInt = () => String(dtype.value ?? "int") !== "float";

	const apply = () => {
		const int = isInt();
		for (const name of COMPONENT_WIDGETS) {
			const w = byName(name);
			if (!w || w.type !== "number") continue;
			w.options ??= {};
			// the float spec, captured before the first int switch overwrites it
			w.options._cvFloatSpec ??= {
				step: w.options.step,
				step2: w.options.step2,
				precision: w.options.precision,
				round: w.options.round,
			};
			const spec = w.options._cvFloatSpec;
			// step is the coarse (shift-drag) increment, step2 the real one
			w.options.step = int ? 10 : spec.step;
			w.options.step2 = int ? 1 : spec.step2;
			w.options.precision = int ? 0 : spec.precision;
			w.options.round = int ? 1 : spec.round;
			if (int) w.value = Math.round(Number(w.value) || 0);

			if (!w._cvDtypeWrapped) {
				w._cvDtypeWrapped = true;
				const cb = w.callback;
				w.callback = function (v, ...rest) {
					if (isInt()) this.value = Math.round(Number(v) || 0);
					return cb?.call(this, this.value, ...rest);
				};
			}
		}
		node.setDirtyCanvas?.(true, true);
	};

	const cb = dtype.callback;
	dtype.callback = function () {
		const r = cb?.apply(this, arguments);
		apply();
		return r;
	};
	apply();
	(node._cvTupleSync ??= []).push(apply);
}

function specOf(inputData) {
	const opts = inputData?.[1];
	const spec = opts && typeof opts === "object" ? opts.cv_tuple : null;
	const components = Array.isArray(spec?.components) ? spec.components : null;
	return components && components.length ? components : null;
}

app.registerExtension({
	name: "opencv_comfyui.cv_tuple",

	getCustomWidgets() {
		return {
			// 'CV Tuple' is a JOIN node, so its component sockets have to take
			// an INT link (an image width, a kernel size) as readily as a FLOAT
			// one - and INT/FLOAT do NOT auto-coerce. A comma-joined io_type is
			// matched permutation-wise by isValidConnection, but it is not a
			// core widget type, so without this entry the input would render as
			// a bare socket with no way to type a value. Plain number widget.
			"FLOAT,INT"(node, inputName, inputData) {
				const opts = inputData?.[1] ?? {};
				const widget = node.addWidget("number", inputName,
					Number(opts.default) || 0, () => {}, {
						min: opts.min ?? -1e38,
						max: opts.max ?? 1e38,
						step: 0.1,
						step2: opts.step ?? 0.01,
						precision: 4,
					});
				widget.tooltip = opts.tooltip ?? widget.tooltip;
				return { widget };
			},

			CV_TUPLE(node, inputName, inputData) {
				const components = specOf(inputData) ?? [
					{ name: "a", default: 0, is_float: true, min: -1e38, max: 1e38 },
					{ name: "b", default: 0, is_float: true, min: -1e38, max: 1e38 },
				];
				const opts = inputData?.[1] ?? {};
				const initial = Array.isArray(opts.default)
					? compose(components, opts.default)
					: (typeof opts.default === "string" && opts.default.trim()
						? opts.default
						: compose(components, components.map((c) => c.default)));

				// the SCHEMA widget: a core string widget at this input's own
				// index, carrying the whole composite
				const main = node.addWidget("string", inputName, initial, () => {
					readBack();
					node.setDirtyCanvas?.(true, true);
				}, {});
				main.tooltip = opts.tooltip ?? main.tooltip;

				const values = components.map((c) => c.default);
				const sub = [];
				const write = () => {
					main.value = compose(components, values);
					node.setDirtyCanvas?.(true, true);
				};

				components.forEach((comp, i) => {
					const w = comp.options
						? node.addWidget("combo", `${inputName}.${comp.name}`,
							String(comp.default ?? comp.options[0]),
							(v) => { values[i] = v; write(); },
							{ values: [...comp.options] })
						: node.addWidget("number", `${inputName}.${comp.name}`,
							Number(comp.default) || 0,
							(v) => {
								values[i] = comp.is_float ? Number(v) : Math.round(Number(v));
								write();
							},
							{
								min: comp.min ?? -1e38,
								max: comp.max ?? 1e38,
								step: comp.is_float ? 0.1 : 10,
								step2: comp.is_float ? 0.01 : 1,
								precision: comp.is_float ? 4 : 0,
							});
					// BOTH flags: the serializer checks the widget property, the
					// layout reads options (same as web/cv_flags.js)
					w.serialize = false;
					w.options.serialize = false;
					w.label = comp.name;
					w.tooltip = comp.tooltip ?? "";
					if (opts.advanced) {
						w.advanced = true;
						w.options.advanced = true;
					}
					sub.push(w);
				});

				// main string -> component widgets (fresh default, workflow load,
				// undo/redo, and a hand-typed literal)
				const readBack = () => {
					const parsed = decompose(main.value, components);
					if (!parsed) return;
					parsed.forEach((v, i) => { values[i] = v; sub[i].value = v; });
				};

				// linked -> the upstream node owns the value; disable the editors
				// (disable, NEVER hide - a hidden widget shifts every slot row
				// below it)
				const updateLinkState = () => {
					const linked = !!node.inputs?.some(
						(i) => i.name === inputName && i.link != null);
					main.disabled = linked;
					for (const w of sub) w.disabled = linked;
				};

				readBack();
				updateLinkState();
				(node._cvTupleSync ??= []).push(() => { readBack(); updateLinkState(); });

				return { widget: main };
			},
		};
	},

	nodeCreated(node) {
		if (node.comfyClass === TUPLE_NODE || node.type === TUPLE_NODE) {
			setupArity(node);
			setupDtype(node);
		}
		if (!node._cvTupleSync) return;
		const onConnectionsChange = node.onConnectionsChange;
		node.onConnectionsChange = function () {
			const r = onConnectionsChange?.apply(this, arguments);
			this._cvTupleSync?.forEach((sync) => sync());
			return r;
		};
	},

	// configure() has applied widgets_values by now - re-read the loaded literal
	// into the component widgets and refresh the link state
	loadedGraphNode(node) {
		node._cvTupleSync?.forEach((sync) => sync());
	},
});
