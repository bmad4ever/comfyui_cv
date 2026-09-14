// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Composite widget for OR-able cv2 flag parameters.
//
// The Python side (opencv_nodes/enums.py FlagGroup via lowlevel.py, and the
// 'CV Warp Flags' builder node) publishes such parameters as a STRING input
// whose options dict carries a `cv_flags` spec: { base: [...mutually
// exclusive names], flags: [...OR-able names], default: "..." }. The
// serialized/executed value stays the pipe-joined constant string
// ("INTER_LINEAR | WARP_RELATIVE_MAP") - identical to what the old curated
// combos saved, so old workflows and headless/API callers need no migration.
//
// Rendering: the raw string widget is REPLACED IN PLACE (same name, same
// node.widgets index) by a combo whose dropdown lists the base options but
// whose VALUE always holds the full composed string. Keeping a visible schema
// widget at the original index matters twice over:
// - widgets_values is written by array index, so the composed string lands
//   exactly where the schema expects it (saved JSON indistinguishable from a
//   plain STRING widget; the flag UI is built in nodeCreated, i.e. at the TAIL
//   past every persisted widget, so skipping it leaves no hole. A
//   serialize:false widget anywhere ELSE does leave one and shifts the values
//   below it - see web/cvlib/widget_values.js);
// - the frontend lays input-slot rows out from VISIBLE widgets while slot
//   ownership follows the schema widget - a hidden interpolation widget
//   shifted every slot below it one row up, so grabbing "borderMode" actually
//   grabbed "interpolation". A visible combo keeps the interpolation slot on
//   the interpolation row, so wiring a STRING (e.g. from 'CV Warp Flags')
//   works by dragging from/to that slot directly.
//
// The flags themselves get one of two renderers, chosen by flag COUNT
// (cvlib/flag_search.js::useList - the threshold and the ranking live there
// because they are the testable half):
// - up to LIST_THRESHOLD flags: one serialize:false toggle widget per flag,
//   appended at the tail. Every cv2 flag group is this shape (8 at most).
// - more: ONE serialize:false DOM widget at the tail, exactly one row tall,
//   holding a button that opens a POPOVER with a search box and the checkbox
//   list. 'CV Region Properties' has 35 columns, which as toggles made the
//   node taller than it was placeable.
// Both drive the same compose()/parse() pair, so the value contract - and
// therefore widgets_values - is identical either way.
//
// While the input is linked the upstream node decides the value: the flag UI
// hides and the combo is disabled; both come back on disconnect.

import { app } from "../../scripts/app.js";
import { useList, fuzzyFilter } from "./cvlib/flag_search.js";

// The button that lives ON the node is ONE widget row tall and never changes
// height. Everything else - search box, checkbox list - lives in a popover
// anchored to it and parented to document.body, i.e. OUTSIDE the node.
//
// That placement is the whole design, and it is what the two previous attempts
// got wrong: under "Modern Node Design" (Nodes 2.0) a DOM widget's element is
// mounted into a bare `<div class="flex flex-col *:flex-1">` (WidgetDOM.vue)
// with no height imposed, the node is laid out `min-h-(--node-height)`
// (LGraphNode.vue) so DOM CONTENT ALWAYS WINS over the dragged size, and
// useNodeResize.ts clamps every drag to the node's measured content height.
// A list inside the node therefore sets the node's floor no matter what
// computeSize / computeLayoutSize say, and any attempt to track the node's
// height from inside is a fight with the layout. A popover has no height
// inside the node at all, so there is nothing to fight - and it behaves the
// same in the classic renderer.
const BUTTON_H = 26;
const PANEL_MAX_H = 340;
const PANEL_MIN_W = 260;

function flagSpecs(nodeData) {
	const found = {};
	for (const group of ["required", "optional"]) {
		const inputs = nodeData?.input?.[group] ?? {};
		for (const name of Object.keys(inputs)) {
			const opts = inputs[name]?.[1];
			if (opts && typeof opts === "object" && opts.cv_flags) {
				found[name] = opts.cv_flags;
			}
		}
	}
	return found;
}

/**
 * The searchable list renderer: a one-row button on the node, and a popover
 * (parented to document.body) holding the search box and the checkbox list.
 * Returns the same interface the toggle renderer exposes to the caller:
 * { selected(), setSelected(names), setHidden(v) }.
 *
 * `tips` are the per-option explanations from the Python spec; a toggle widget
 * carries them as widget.tooltip, a row carries them as the native title.
 */
function listRenderer(node, name, flags, tips, onChange) {
	const INPUT_CSS = "background:var(--comfy-input-bg,#222);"
		+ "color:var(--input-text,#ddd);"
		+ "border:1px solid var(--border-color,#444);border-radius:4px;";

	// --- the part that lives ON the node: one button, fixed height ----------
	const container = document.createElement("div");
	container.style.cssText = `display:flex;align-items:center;height:${BUTTON_H}px;`
		+ "width:100%;font:12px sans-serif;";
	const button = document.createElement("button");
	button.style.cssText = "flex:1 1 auto;height:100%;padding:0 8px;cursor:pointer;"
		+ "display:flex;align-items:center;justify-content:space-between;gap:6px;"
		+ "font:inherit;text-align:left;overflow:hidden;" + INPUT_CSS;
	const buttonLabel = document.createElement("span");
	buttonLabel.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
	const caret = document.createElement("span");
	caret.textContent = "▾";
	caret.style.cssText = "flex:0 0 auto;opacity:0.7;";
	button.append(buttonLabel, caret);
	container.append(button);

	// --- the popover: search + count + only + clear + the scrolling list -----
	const panel = document.createElement("div");
	panel.style.cssText = "position:fixed;z-index:10000;display:none;"
		+ "flex-direction:column;gap:4px;padding:6px;"
		+ "font:12px sans-serif;color:var(--input-text,#ddd);"
		+ "background:var(--comfy-menu-bg,#353535);"
		+ "border:1px solid var(--border-color,#444);border-radius:6px;"
		+ "box-shadow:0 6px 20px rgba(0,0,0,0.45);";

	const header = document.createElement("div");
	header.style.cssText = "display:flex;gap:4px;align-items:center;flex:0 0 auto;";
	const search = document.createElement("input");
	search.type = "text";
	search.placeholder = "filter...";
	search.style.cssText = "flex:1 1 auto;min-width:40px;padding:3px 5px;font:inherit;"
		+ INPUT_CSS;
	const count = document.createElement("span");
	count.style.cssText = "flex:0 0 auto;opacity:0.7;white-space:nowrap;";
	const btn = (text, title) => {
		const b = document.createElement("button");
		b.textContent = text;
		b.title = title;
		b.style.cssText = "flex:0 0 auto;padding:3px 7px;cursor:pointer;font:inherit;"
			+ INPUT_CSS;
		return b;
	};
	const onlyBtn = btn("only", "show only the ticked columns");
	const clearBtn = btn("clear", "untick every column");
	header.append(search, count, onlyBtn, clearBtn);

	// --- body: one row per flag, in DECLARED order --------------------------
	// Declared order is the feature-matrix COLUMN order, so the list reads the
	// same way the matrix does. Filtering hides rows, it never reorders them
	// and never touches a value: a ticked row hidden by a query stays ticked.
	const body = document.createElement("div");
	body.style.cssText = `flex:1 1 auto;min-height:0;max-height:${PANEL_MAX_H}px;`
		+ "overflow-y:auto;overflow-x:hidden;" + INPUT_CSS;

	const rows = flags.map((flag) => {
		const row = document.createElement("label");
		row.style.cssText = "display:flex;gap:6px;align-items:center;padding:2px 5px;"
			+ "cursor:pointer;white-space:nowrap;";
		const box = document.createElement("input");
		box.type = "checkbox";
		box.style.cssText = "flex:0 0 auto;margin:0;cursor:pointer;";
		const text = document.createElement("span");
		text.textContent = flag;
		text.style.cssText = "overflow:hidden;text-overflow:ellipsis;";
		if (tips[flag]) row.title = tips[flag];
		row.append(box, text);
		box.addEventListener("change", () => { paint(); onChange(); });
		body.append(row);
		return { flag, row, box, text };
	});

	panel.append(header, body);

	let onlySelected = false;
	const paint = () => {
		const picked = rows.filter((r) => r.box.checked);
		count.textContent = `${picked.length}/${rows.length}`;
		buttonLabel.textContent = picked.length
			? `${picked.length} of ${rows.length}: ${picked.map((r) => r.flag).join(", ")}`
			: `none of ${rows.length} columns`;
		// the button is one line wide, so the full selection goes in the tooltip
		button.title = picked.length
			? picked.map((r) => r.flag).join("\n")
			: "no columns ticked - click to choose";
		onlyBtn.style.opacity = onlySelected ? "1" : "0.55";
		for (const r of rows) {
			r.text.style.fontWeight = r.box.checked ? "600" : "400";
			r.text.style.opacity = r.box.checked ? "1" : "0.75";
		}
	};
	const refilter = () => {
		const keep = new Set(
			fuzzyFilter(search.value, rows.map((r) => r.flag)).map((m) => m.index));
		rows.forEach((r, i) => {
			const show = keep.has(i) && (!onlySelected || r.box.checked);
			r.row.style.display = show ? "flex" : "none";
		});
	};
	search.addEventListener("input", refilter);
	onlyBtn.addEventListener("click", (e) => {
		e.preventDefault();
		onlySelected = !onlySelected;
		paint();
		refilter();
	});
	clearBtn.addEventListener("click", (e) => {
		e.preventDefault();
		for (const r of rows) r.box.checked = false;
		paint();
		refilter();
		onChange();
	});

	// --- open / close -------------------------------------------------------
	let open = false;
	const place = () => {
		const r = button.getBoundingClientRect();
		panel.style.width = `${Math.max(r.width, PANEL_MIN_W)}px`;
		panel.style.left = `${Math.max(4, Math.min(r.left,
			window.innerWidth - Math.max(r.width, PANEL_MIN_W) - 4))}px`;
		// below the button, or above it when there is no room down there
		const h = panel.offsetHeight || PANEL_MAX_H;
		panel.style.top = (r.bottom + 4 + h <= window.innerHeight)
			? `${r.bottom + 4}px`
			: `${Math.max(4, r.top - 4 - h)}px`;
	};
	const closePanel = () => {
		if (!open) return;
		open = false;
		panel.style.display = "none";
		panel.remove();
		document.removeEventListener("pointerdown", onDocPointer, true);
		document.removeEventListener("wheel", onDocWheel, true);
		window.removeEventListener("resize", place);
	};
	const onDocPointer = (e) => {
		if (!panel.contains(e.target) && !container.contains(e.target)) closePanel();
	};
	// the canvas zooms/pans on wheel, which would leave the popover stranded
	const onDocWheel = (e) => { if (!panel.contains(e.target)) closePanel(); };
	const openPanel = () => {
		if (open) return;
		open = true;
		document.body.append(panel);
		panel.style.display = "flex";
		place();
		place();   // once the panel has a height, re-decide above/below
		search.value = "";
		refilter();
		search.focus();
		document.addEventListener("pointerdown", onDocPointer, true);
		document.addEventListener("wheel", onDocWheel, true);
		window.addEventListener("resize", place);
	};
	button.addEventListener("click", (e) => {
		e.preventDefault();
		e.stopPropagation();
		if (open) closePanel(); else openPanel();
	});

	// The popover is a sibling of the canvas, so its keys would otherwise reach
	// the graph's keybindings (Delete would eat the selected node while the
	// user is typing a filter). Escape closes it.
	for (const type of ["pointerdown", "pointerup", "wheel", "keyup", "keypress",
	                    "dblclick", "contextmenu"]) {
		panel.addEventListener(type, (e) => e.stopPropagation());
		container.addEventListener(type, (e) => e.stopPropagation());
	}
	for (const el of [panel, container]) {
		el.addEventListener("keydown", (e) => {
			e.stopPropagation();
			if (e.key === "Escape") { closePanel(); button.focus(); }
		});
	}

	// A neutral widget type so the Vue node renderer treats it as a plain DOM
	// widget instead of routing it through its own draw-canvas path (same
	// reasoning as annotate_points.js / cv_charts.js). It is exactly one row
	// tall in both renderers, so the node behaves like any other widget node.
	const widget = node.addDOMWidget(`${name}.__list`, "opencv_flaglist",
		container, {
			serialize: false,
			hideOnZoom: false,
			getMinHeight: () => BUTTON_H,
			getMaxHeight: () => BUTTON_H,
		});
	// BOTH flags, same as the toggle renderer below: `options.serialize` only
	// keeps the widget out of the API prompt, workflow persistence is the
	// widget-level property. Without it the checkbox list saved its empty value
	// as one extra widgets_values entry (see web/cvlib/widget_values.js); the
	// selected flags live in the composed string on the combo.
	widget.serialize = false;
	widget.computeLayoutSize = () => ({
		minHeight: BUTTON_H, maxHeight: BUTTON_H, minWidth: 40, maxWidth: 1e6,
	});

	// the popover lives outside the node's DOM, so it has to be cleaned up by
	// hand when the node goes away (or it is left floating over the canvas)
	const onRemoved = node.onRemoved;
	node.onRemoved = function () {
		closePanel();
		return onRemoved?.apply(this, arguments);
	};

	paint();
	refilter();

	return {
		selected: () => rows.filter((r) => r.box.checked).map((r) => r.flag),
		setSelected: (names) => {
			const on = new Set(names);
			for (const r of rows) r.box.checked = on.has(r.flag);
			paint();
			refilter();
		},
		setHidden: (hidden) => {
			widget.hidden = hidden;
			if (widget.options) widget.options.hidden = hidden;
			container.style.display = hidden ? "none" : "flex";
			if (hidden) closePanel();
		},
	};
}

/** One serialize:false toggle widget per flag - the original renderer. */
function toggleRenderer(node, name, flags, tips, onChange) {
	const sub = [];
	for (const flag of flags) {
		const t = node.addWidget("toggle", `${name}.${flag}`, false, onChange,
			{ on: "on", off: "off" });
		t.label = flag;
		t.cvFlagName = flag;
		// per-option help. The frontend shows `widget.tooltip` for the widget
		// under the cursor and otherwise falls back to the nodeDef input
		// tooltip BY WIDGET NAME - and a toggle is named "<input>.<flag>",
		// which matches no input, so without this the checkboxes have no
		// tooltip at all and everything worth saying about an individual
		// option has to be crammed into the input's.
		if (tips[flag]) t.tooltip = tips[flag];
		// widget-level property is what the serializer checks; options for
		// good measure (core sets both)
		t.serialize = false;
		t.options.serialize = false;
		sub.push(t);
	}
	return {
		selected: () => sub.filter((w) => w.value).map((w) => w.cvFlagName),
		setSelected: (names) => {
			const on = new Set(names);
			for (const w of sub) w.value = on.has(w.cvFlagName);
		},
		setHidden: (hidden) => {
			for (const w of sub) {
				w.hidden = hidden;
				w.options.hidden = hidden;
			}
		},
	};
}

app.registerExtension({
	name: "opencv_comfyui.cv_flags",

	beforeRegisterNodeDef(nodeType, nodeData) {
		const specs = flagSpecs(nodeData);
		if (Object.keys(specs).length) nodeType.prototype._cvFlagSpecs = specs;
	},

	nodeCreated(node) {
		const specs = node._cvFlagSpecs;
		if (!specs) return;
		node._cvFlagSync = [];
		for (const [name, spec] of Object.entries(specs)) {
			const idx = node.widgets?.findIndex((w) => w.name === name) ?? -1;
			if (idx < 0) continue;
			const base = Array.isArray(spec.base) && spec.base.length ? spec.base : null;
			const flags = Array.isArray(spec.flags) ? spec.flags : [];
			const tips = (spec.tooltips && typeof spec.tooltips === "object")
				? spec.tooltips : {};
			if (!base) continue;
			const fallback = base.includes(spec.default) ? spec.default : base[0];
			const initial = node.widgets[idx].value ?? spec.default ?? fallback;

			// the currently selected base option; the flag states live in the
			// renderer's own checkboxes/toggles
			let curBase = fallback;
			let ui = null;
			const compose = () => {
				combo.value = [curBase, ...(ui ? ui.selected() : [])].join(" | ");
				node.setDirtyCanvas?.(true, true);
			};

			// dropdown offers the bases, value carries the whole composed string
			// (a combo happily displays a value outside its list). The callback
			// sees the picked base (arrows/menu) - or a programmatic full string -
			// takes its base part and immediately re-composes the full value.
			const combo = node.addWidget("combo", name, initial, (v) => {
				const parts = String(v ?? "").split("|")
					.map((s) => s.trim()).filter(Boolean);
				curBase = parts.find((p) => base.includes(p)) ?? curBase;
				compose();
				// belt and braces: should any widget path assign the picked entry
				// AFTER the callback, re-assert the composed value right behind it
				queueMicrotask?.(compose);
			}, { values: [...base] });
			node.widgets.pop();          // take it off the tail...
			node.widgets[idx] = combo;   // ...and swap it in for the string widget

			ui = (useList(spec) && typeof node.addDOMWidget === "function")
				? listRenderer(node, name, flags, tips, compose)
				: toggleRenderer(node, name, flags, tips, compose);

			// combo string -> base + flag states (fresh node default AND workflow
			// load); unknown parts (hand-edited strings, legacy bare ints from
			// before the param was a FlagGroup) leave the flags at their
			// fallbacks and are only overwritten once the user touches one
			const parse = () => {
				const parts = String(combo.value ?? "").split("|")
					.map((s) => s.trim()).filter(Boolean);
				curBase = parts.find((p) => base.includes(p)) ?? fallback;
				ui.setSelected(parts);
			};

			// linked -> upstream decides the value: disable the combo and hide
			// the flag UI; unlinked -> the composite widget is the interface
			const updateLinkState = () => {
				const linked = !!node.inputs?.some(
					(i) => i.name === name && i.link != null);
				combo.disabled = linked;
				ui.setHidden(linked);
				node.setDirtyCanvas?.(true, true);
			};

			parse();
			updateLinkState();
			node._cvFlagSync.push(() => { parse(); updateLinkState(); });
		}

		const onConnectionsChange = node.onConnectionsChange;
		node.onConnectionsChange = function () {
			const r = onConnectionsChange?.apply(this, arguments);
			this._cvFlagSync?.forEach((sync) => sync());
			return r;
		};
	},

	// configure() has applied widgets_values by the time this fires - re-read
	// the loaded string into the flag UI and refresh the link state (also
	// covers undo/redo, which reloads the graph through the same path)
	loadedGraphNode(node) {
		node._cvFlagSync?.forEach((sync) => sync());
	},
});
