// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Composite widget for the crop sizing policy (opencv_nodes/boxpolicy.py).
//
// Sibling of cv_flags.js and deliberately a copy rather than a shared
// generalisation: cv_flags backs eight shipped builder nodes and every saved
// FlagGroup widget value, and this widget needs TYPED sub-widgets (int, float,
// bool) instead of bare OR-able names. Same contract otherwise - the Python
// side publishes a STRING input whose options dict carries a `cv_policy` spec
// { base: [...size modes], default: "...", fields: [{name,type,default,min,
// max,step,label,tooltip}] } and the value crossing the graph stays the
// pipe-joined string ("uniform (largest) | padding=24 | size_multiple=8").
//
// Rendering follows cv_flags.js exactly, and for the same two reasons:
// the raw string widget is REPLACED IN PLACE by a combo listing the size
// modes whose VALUE holds the whole composed string (widgets_values is
// written by array index, and the frontend lays input-slot rows out from
// VISIBLE widgets while slot ownership follows the schema widget). The typed
// field widgets are appended at the tail with serialize:false.
//
// Only NON-DEFAULT fields are written into the string. That is what lets an
// upstream 'CV Snap BBoxes' override exactly the fields it sets and leave
// the consumer node's own widgets in charge of the rest.
//
// The field widgets inherit the schema widget's `advanced` flag, so on the
// crop nodes (where the policy input is optional+advanced) they live inside
// the node's "advanced inputs" section and stay hidden until it is expanded,
// while on 'CV Snap BBoxes' - where the policy IS the interface - they
// are always visible.
//
// While the input is linked the upstream node decides the value: the field
// widgets hide and the combo is disabled; both come back on disconnect.

import { app } from "../../scripts/app.js";

function policySpecs(nodeData) {
	const found = {};
	for (const group of ["required", "optional"]) {
		const inputs = nodeData?.input?.[group] ?? {};
		for (const name of Object.keys(inputs)) {
			const opts = inputs[name]?.[1];
			if (opts && typeof opts === "object" && opts.cv_policy) {
				found[name] = { spec: opts.cv_policy, advanced: !!opts.advanced };
			}
		}
	}
	return found;
}

// "16" -> 16, "1.5" -> 1.5, "on"/"1"/"true" -> true
function coerce(type, raw, fallback) {
	if (type === "bool") {
		if (typeof raw === "boolean") return raw;
		return ["1", "true", "on", "yes"].includes(String(raw).trim().toLowerCase());
	}
	if (type === "choice") return String(raw).trim() || fallback;
	const n = Number(String(raw).trim());
	if (!Number.isFinite(n)) return fallback;
	return type === "int" ? Math.round(n) : n;
}

function sameValue(type, a, b) {
	if (type === "bool") return !!a === !!b;
	if (type === "choice") return String(a) === String(b);
	return Number(a) === Number(b);
}

app.registerExtension({
	name: "opencv_comfyui.cv_policy",

	beforeRegisterNodeDef(nodeType, nodeData) {
		const specs = policySpecs(nodeData);
		if (Object.keys(specs).length) nodeType.prototype._cvPolicySpecs = specs;
	},

	nodeCreated(node) {
		const specs = node._cvPolicySpecs;
		if (!specs) return;
		node._cvPolicySync = [];
		for (const [name, { spec, advanced }] of Object.entries(specs)) {
			const idx = node.widgets?.findIndex((w) => w.name === name) ?? -1;
			if (idx < 0) continue;
			const base = Array.isArray(spec.base) && spec.base.length ? spec.base : null;
			const fields = Array.isArray(spec.fields) ? spec.fields : [];
			const legacy = (spec.legacy && typeof spec.legacy === "object") ? spec.legacy : {};
			if (!base) continue;
			const fallback = base.includes(spec.default) ? spec.default : base[0];
			const initial = node.widgets[idx].value ?? spec.default ?? fallback;

			let curBase = fallback;
			const sub = [];
			const compose = () => {
				combo.value = [
					curBase,
					...sub
						.filter((w) => !sameValue(w.cvType, w.value, w.cvDefault))
						.map((w) => `${w.cvName}=${w.cvType === "bool"
							? (w.value ? 1 : 0)
							: w.value}`),
				].join(" | ");
				node.setDirtyCanvas?.(true, true);
			};

			// dropdown offers the size modes, value carries the whole composed
			// string (a combo happily displays a value outside its list)
			const combo = node.addWidget("combo", name, initial, (v) => {
				const parts = String(v ?? "").split("|").map((s) => s.trim()).filter(Boolean);
				curBase = parts.find((p) => base.includes(p)) ?? curBase;
				compose();
				queueMicrotask?.(compose);
			}, { values: [...base] });
			combo.tooltip = node.widgets[idx].tooltip ?? combo.tooltip;
			node.widgets.pop();          // take it off the tail...
			node.widgets[idx] = combo;   // ...and swap it in for the string widget

			for (const f of fields) {
				const type = f.type === "bool" ? "toggle"
					: f.type === "choice" ? "combo" : "number";
				const opts = f.type === "bool"
					? { on: "on", off: "off" }
					: f.type === "choice"
					? { values: [...(f.choices ?? [])] }
					: {
						min: f.min ?? 0,
						max: f.max ?? 8192,
						step: (f.step ?? 1) * 10,  // litegraph "step" is 1/10 of a drag unit
						precision: f.type === "int" ? 0 : 2,
						round: f.type === "int" ? 1 : false,
					};
				const w = node.addWidget(type, `${name}.${f.name}`, f.default, compose, opts);
				w.label = f.label ?? f.name;
				w.tooltip = f.tooltip;
				w.cvName = f.name;
				w.cvType = f.type;
				w.cvDefault = f.default;
				// widget-level property is what the serializer checks; options
				// for good measure (core sets both)
				w.serialize = false;
				w.options.serialize = false;
				// inherit the schema input's advanced flag so the fields land in
				// the node's collapsible "advanced inputs" section with it
				w.advanced = advanced;
				w.options.advanced = advanced;
				sub.push(w);
			}

			// combo string -> base + fields (fresh node default AND workflow
			// load); a field the string does not name is at its default, which
			// is exactly what compose() left out
			const parse = () => {
				const parts = String(combo.value ?? "").split("|")
					.map((s) => s.trim()).filter(Boolean);
				const seen = {};
				for (const p of parts) {
					const eq = p.indexOf("=");
					if (eq > 0) seen[p.slice(0, eq).trim()] = p.slice(eq + 1).trim();
				}
				// a base the spec still lists wins; otherwise map a LEGACY base
				// (a workflow saved before the vocabulary changed) exactly the
				// way boxpolicy.parse does, so an old value keeps its meaning
				// instead of silently reverting to the default
				curBase = parts.find((p) => base.includes(p)) ?? null;
				if (curBase === null) {
					const old = parts.find((p) => legacy[p]);
					if (old) {
						const [mode, size] = legacy[old];
						curBase = mode;
						if (size && !(("size") in seen)) seen.size = size;
					} else {
						curBase = fallback;
					}
				}
				for (const w of sub) {
					w.value = w.cvName in seen
						? coerce(w.cvType, seen[w.cvName], w.cvDefault)
						: w.cvDefault;
				}
			};

			// linked -> upstream decides the value: disable the combo and hide
			// the field widgets; unlinked -> the composite widget is the interface
			const updateLinkState = () => {
				const linked = !!node.inputs?.some(
					(i) => i.name === name && i.link != null);
				combo.disabled = linked;
				for (const w of sub) {
					w.hidden = linked;
					w.options.hidden = linked;
				}
				node.setDirtyCanvas?.(true, true);
			};

			parse();
			updateLinkState();
			node._cvPolicySync.push(() => { parse(); updateLinkState(); });
		}

		const onConnectionsChange = node.onConnectionsChange;
		node.onConnectionsChange = function () {
			const r = onConnectionsChange?.apply(this, arguments);
			this._cvPolicySync?.forEach((sync) => sync());
			return r;
		};
	},

	// configure() has applied widgets_values by the time this fires - re-read
	// the loaded string into the field widgets and refresh the link state
	loadedGraphNode(node) {
		node._cvPolicySync?.forEach((sync) => sync());
	},
});
