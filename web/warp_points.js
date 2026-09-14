// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Editing surface for the "CV Annotate Correspondences" node.
//
// The Python node previews its input image under the ui key "annotate_images"
// (shared with CV Annotate Points). This extension draws that image, the
// control-point pairs, and a LIVE preview of the warp they describe, on a
// <canvas> added as a DOM widget - the only surface that renders in both the
// classic LiteGraph rendering and "Modern Node Design (Nodes 2.0)", where node
// bodies are DOM and onDrawForeground / onMouseDown are never called.
//
// Interaction: left-click empty space adds a PINNED pair (from == to) and
// starts dragging it, left-drag moves an endpoint, SHIFT+click removes a whole
// pair, ALT+DRAG moves both ends together, double-click re-pins a pair. Which
// end a plain drag grabs is set by the "drag edits" widget - it never hides
// anything, both ends are always drawn.
//
// POWER TOOLS (all frontend-only - the executed value stays the "pairs" JSON):
//   * SELECTION: CTRL+click toggles an endpoint in/out of the selection,
//     CTRL+drag on empty space rubber-bands endpoints into it. Dragging any
//     selected endpoint moves the WHOLE selection; a round handle floating
//     above the selection rotates/scales it about its centroid (per pane, in
//     pixel space, so a rotation is a true rotation on a non-square image);
//     SHIFT+click on a selected endpoint deletes every selected pair.
//     Click empty space to clear.
//   * SHAPE SEEDS: the "add shape" combo + button drop a pinned scaffold
//     (grids, ellipses, lines) to deform from - the multi-point warps that were
//     cumbersome to click out one pair at a time.
//   * CURVE PAIR: the "draw curve pair" toggle arms a two-stroke mode - draw
//     along a feature (a flag edge, a cloth fold), OR simply CLICK one of the
//     image's four borders to select the whole border as the source curve
//     (the common flag/cloth case: map the picture's own edges onto drawn
//     curves, one border at a time), then draw where it must land; both
//     curves are resampled to "curve points" points UNIFORMLY BY ARC LENGTH
//     (cvlib/curve_ops.js, pinned against numpy - pointermove events cluster
//     where the hand slows, so a naive subsample would weight the warp toward
//     the slow parts of the stroke) and appended as pairs in order. Borders
//     run left->right (top/bottom) and top->bottom (left/right); draw the
//     target stroke in the same direction, or backwards for a mirror.
//
// BACKGROUND MODE: when the node's optional `background` input is connected the
// server previews it under "annotate_background", and the surface splits into
// two panes - the source image on the left (cyan FROM ends), the background on
// the right (orange TO ends). Every pair's `to` half is then normalized to the
// BACKGROUND's size (matching the Python side), the fitted warp maps image ->
// background, and the live preview composites the warped image OVER the
// background - the authoring surface for "paste this picture somewhere in
// there". A click on either pane adds a pair pinned at the same normalized
// spot of the other pane. The background can only ever arrive through the
// server preview: the browser has no access to the pixels of a linked IMAGE,
// which is why the input is a real (backend) socket and not a frontend-only
// affordance. The curve tool draws stroke 1 on the image pane and stroke 2 on
// the background pane.
//
// Pairs are stored normalized (0-1 of the image size; of the background's size
// for the `to` half in background mode) as JSON in the node's "pairs" STRING
// widget - the value the Python side parses - so they survive saves/reloads and
// can be edited by hand. The view widgets are serialize:false and their state
// lives in node.properties instead, because it is not part of the executed
// value. The selection, the rubber band and a half-finished curve pair are
// TRANSIENT by design - they are gestures, not state worth saving.
//
// The preview is the same math the server runs (cvlib/warp_math.js is pinned
// against cv2 by tests/smoke_test.py) evaluated on a grid and drawn as a
// textured mesh - see cvlib/warp_gl.js for why it is a mesh and not a
// per-pixel shader.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { makeModel, meshFromModel, MODELS } from "./cvlib/warp_math.js";
import { resampleCurve } from "./cvlib/curve_ops.js";
import * as warpGL from "./cvlib/warp_gl.js";

const NODE_CLASS = "CV_AnnotateCorrespondences";
const HIT_RADIUS = 12;    // element-space (CSS) units
const MIN_AREA_H = 220;   // minimum canvas height in px
const CELL_PX = 10;       // target mesh cell size, in canvas pixels
const GRID_LINES = 12;    // deformation-grid overlay resolution
const PANE_GAP = 8;       // px between the two panes in background mode
const HANDLE_OFFSET = 36; // px from the selection centroid to its handle
const BORDER_HIT = 14;    // px around an image border that counts as clicking it

// the four borders as normalized endpoint pairs - the DRAW ORDER is part of
// the contract (stated in the banner): top/bottom run left->right, the sides
// top->bottom, so the user knows which way to draw the target stroke
const BORDER_ENDS = {
	top: [[0, 0], [1, 0]], bottom: [[0, 1], [1, 1]],
	left: [[0, 0], [0, 1]], right: [[1, 0], [1, 1]],
};

const FROM_COLOR = "rgba(80, 210, 255, 0.95)";
const TO_COLOR = "rgba(255, 165, 40, 0.95)";
const CURVE_COLOR = "rgba(255, 235, 60, 0.9)";

const DRAG_MODES = ["both (nearest)", "from (original)", "to (destination)"];
const SHAPES = ["grid 3x3", "grid 4x4", "ellipse (8 points)",
                "ellipse (12 points)", "horizontal line (5)",
                "vertical line (5)"];

const DEFAULT_VIEW = {
	drag: "both (nearest)",
	preview: true,
	opacity: 1.0,
	model: MODELS[0],
	mesh: false,
	shape: SHAPES[0],
	curveN: 8,
};

function viewURL(d) {
	return api.apiURL(
		`/view?filename=${encodeURIComponent(d.filename)}` +
		`&type=${encodeURIComponent(d.type ?? "temp")}` +
		`&subfolder=${encodeURIComponent(d.subfolder ?? "")}` +
		`&t=${Date.now()}`
	);
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));

// Normalized [0,1] point sets for the "add shape" seeds - pinned pairs the
// user deforms afterwards. Inset from the border so every point is grabbable.
function shapePoints(name) {
	const lo = 0.12, hi = 0.88, mid = 0.5;
	const pts = [];
	if (name.startsWith("grid")) {
		const n = name.includes("4x4") ? 4 : 3;
		for (let r = 0; r < n; r++) {
			for (let c = 0; c < n; c++) {
				pts.push([lo + (c / (n - 1)) * (hi - lo),
				          lo + (r / (n - 1)) * (hi - lo)]);
			}
		}
	} else if (name.startsWith("ellipse")) {
		const k = name.includes("12") ? 12 : 8;
		for (let i = 0; i < k; i++) {
			const a = (i / k) * Math.PI * 2;
			pts.push([mid + 0.38 * Math.cos(a), mid + 0.38 * Math.sin(a)]);
		}
	} else if (name.startsWith("horizontal")) {
		for (let i = 0; i < 5; i++) pts.push([lo + (i / 4) * (hi - lo), mid]);
	} else if (name.startsWith("vertical")) {
		for (let i = 0; i < 5; i++) pts.push([mid, lo + (i / 4) * (hi - lo)]);
	}
	return pts;
}

app.registerExtension({
	name: "opencv_comfyui.warp_points",

	async nodeCreated(node) {
		if (node.comfyClass !== NODE_CLASS) return;
		const pairsWidget = node.widgets?.find((w) => w.name === "pairs");
		if (!pairsWidget) return;

		const img = new Image();
		const bgImg = new Image();
		let drag = null;    // { index, end: 0|1|"both", homeRect?, startN?, start? }
		let hover = null;   // { index, end }
		let lastClick = 0;
		// power-tool state, all transient (gestures, not saved state)
		const selection = new Set();  // "index:end" keys
		let selDrag = null; // { kind:"move"|"xform", startPos|handleStart, items, centroids }
		let band = null;    // { x0, y0, x1, y1 }
		let curve = null;   // { phase: 0|1, stroke: [..]|null, ptsA: [..] (px) }

		// --- view state (NOT part of the executed value -> node.properties) ---
		node.properties = node.properties || {};
		const view = Object.assign({}, DEFAULT_VIEW, node.properties.cv_warp_view || {});
		const saveView = () => {
			node.properties.cv_warp_view = Object.assign({}, view);
		};

		// --- the DOM-widget canvas -------------------------------------------
		// The container is the positioning context; the canvas is ABSOLUTELY
		// positioned inside it so it contributes nothing to the container's flow
		// height. In Nodes 2.0 the node height is measured from the widget
		// element's content, and an in-flow canvas whose attribute size we set
		// each redraw feeds that measurement back - the node then grows without
		// bound. (Same reason as web/annotate_points.js.)
		const container = document.createElement("div");
		container.style.position = "relative";
		container.style.width = "100%";
		container.style.height = "100%";
		container.style.overflow = "hidden";

		const canvas = document.createElement("canvas");
		canvas.style.position = "absolute";
		canvas.style.inset = "0";
		canvas.style.width = "100%";
		canvas.style.height = "100%";
		canvas.style.display = "block";
		canvas.style.borderRadius = "4px";
		canvas.style.cursor = "crosshair";
		canvas.style.touchAction = "none";
		container.appendChild(canvas);

		// A neutral type (not "canvas") so the Vue node renderer treats it as a
		// plain DOM widget instead of routing it through its own draw path.
		const surface = node.addDOMWidget("correspondence_surface", "opencv_warp",
			container, { serialize: false, hideOnZoom: false });
		// BOTH flags: `options.serialize` only keeps the widget out of the API
		// prompt - workflow persistence is the widget-level property, and
		// without it this editing surface saved its empty value as one extra
		// widgets_values entry (see web/cvlib/widget_values.js). The pairs
		// themselves live in the `pairs` string widget.
		surface.serialize = false;
		surface.computeLayoutSize = () => ({
			minHeight: MIN_AREA_H, maxHeight: 1e6, minWidth: 20, maxWidth: 1e6,
		});
		// legacy LiteGraph asks for a node-space size instead
		surface.computeSize = (width) =>
			[width, Math.max(MIN_AREA_H, Math.round((width || 300) * 0.75))];

		// --- model -----------------------------------------------------------
		// Pairs are cached so a drag does not re-parse (and re-serialize) the
		// widget on every pointermove; the widget is written on release. The
		// cache is invalidated whenever the widget value changed behind our back
		// (hand edit, undo, workflow load).
		let cache = null;
		let cacheRaw = null;
		const getPairs = () => {
			const raw = pairsWidget.value ?? "[]";
			if (raw !== cacheRaw) {
				cacheRaw = raw;
				try {
					const v = JSON.parse(raw || "[]");
					cache = Array.isArray(v) ? v.filter((p) =>
						Array.isArray(p) && p.length === 2 &&
						Array.isArray(p[0]) && p[0].length === 2 &&
						Array.isArray(p[1]) && p[1].length === 2) : [];
				} catch (e) {
					cache = [];
				}
			}
			return cache;
		};
		const round4 = (v) => Math.round(v * 1e4) / 1e4;
		const setPairs = (pairs, commit) => {
			cache = pairs;
			if (commit) {
				cacheRaw = JSON.stringify(pairs.map(([f, t]) =>
					[[round4(f[0]), round4(f[1])], [round4(t[0]), round4(t[1])]]));
				pairsWidget.value = cacheRaw;
				pairsWidget.callback?.(cacheRaw);
			}
			redraw();
		};
		const deepPairs = () => getPairs().map((p) => [p[0].slice(), p[1].slice()]);

		// --- background mode --------------------------------------------------
		const hasBg = () => !!(bgImg.complete && bgImg.naturalWidth);
		// the frame the `to` half is normalized against - the background when
		// one is loaded, the image itself otherwise (matching the Python node)
		const dstDims = () => (hasBg()
			? { w: bgImg.naturalWidth, h: bgImg.naturalHeight }
			: { w: img.naturalWidth || 1, h: img.naturalHeight || 1 });
		const endDims = (end) => (end === 1 ? dstDims()
			: { w: img.naturalWidth || 1, h: img.naturalHeight || 1 });

		// --- warp model (refit only when the inputs actually change) ----------
		let fitKey = null;
		let fitted = null;
		const currentModel = () => {
			const pairs = getPairs();
			const iw = img.naturalWidth || 1;
			const ih = img.naturalHeight || 1;
			const d = dstDims();
			const key = `${view.model}|${iw}x${ih}|${d.w}x${d.h}|${JSON.stringify(pairs)}`;
			if (key === fitKey) return fitted;
			fitKey = key;
			// image PIXEL coordinates, denormalized exactly like the Python node
			// (normalized 1.0 -> pixel w - 1) so the preview and the server
			// agree; the `to` half denormalizes against the DESTINATION frame
			const sx = iw > 1 ? iw - 1 : 1;
			const sy = ih > 1 ? ih - 1 : 1;
			const tx = d.w > 1 ? d.w - 1 : 1;
			const ty = d.h > 1 ? d.h - 1 : 1;
			const from = pairs.map(([f]) => [f[0] * sx, f[1] * sy]);
			const to = pairs.map(([, t]) => [t[0] * tx, t[1] * ty]);
			// regularization 0: the editor authors correspondences, smoothing is
			// the consumer node's knob (and it is image-scale dependent)
			fitted = makeModel(view.model, from, to, 0);
			return fitted;
		};

		// --- geometry (element-local CSS pixels) ------------------------------
		const area = () => ({
			w: Math.max(1, canvas.clientWidth || canvas.width),
			h: Math.max(1, canvas.clientHeight || canvas.height),
		});
		const fitRect = (iw, ih, bx, by, bw, bh) => {
			const s = Math.min(bw / iw, bh / ih);
			const w = iw * s, h = ih * s;
			return { x: bx + (bw - w) / 2, y: by + (bh - h) / 2, w, h };
		};
		// One rect in the single-image mode; two side-by-side panes in
		// background mode. `src` hosts the FROM ends, `dst` the TO ends - in
		// single-image mode they are the SAME rect, which is what keeps every
		// consumer below mode-agnostic.
		const layout = () => {
			const a = area();
			const iw = img.naturalWidth || 1, ih = img.naturalHeight || 1;
			if (!hasBg()) {
				const r = fitRect(iw, ih, 0, 0, a.w, a.h);
				return { split: false, src: r, dst: r };
			}
			const half = (a.w - PANE_GAP) / 2;
			return {
				split: true,
				src: fitRect(iw, ih, 0, 0, half, a.h),
				dst: fitRect(bgImg.naturalWidth, bgImg.naturalHeight,
				             half + PANE_GAP, 0, half, a.h),
			};
		};
		const rectFor = (L, end) => (end === 0 ? L.src : L.dst);
		// the two end-spaces share one pane in single-image mode - group
		// selection geometry by PANE, not by end, so a mixed selection there
		// rotates about one centroid instead of two coincident ones
		const paneOf = (L, end) => (L.split ? end : 0);
		const inRect = (pos, r) => pos[0] >= r.x && pos[0] <= r.x + r.w &&
		                           pos[1] >= r.y && pos[1] <= r.y + r.h;
		// which of the image's four borders a position is close to (null = none)
		const borderNear = (pos, r) => {
			if (pos[0] < r.x - BORDER_HIT || pos[0] > r.x + r.w + BORDER_HIT ||
					pos[1] < r.y - BORDER_HIT || pos[1] > r.y + r.h + BORDER_HIT) {
				return null;
			}
			const d = {
				top: Math.abs(pos[1] - r.y),
				bottom: Math.abs(pos[1] - (r.y + r.h)),
				left: Math.abs(pos[0] - r.x),
				right: Math.abs(pos[0] - (r.x + r.w)),
			};
			let best = null;
			for (const k of Object.keys(d)) {
				if (d[k] <= BORDER_HIT && (best === null || d[k] < d[best])) best = k;
			}
			return best;
		};
		const toNormRaw = (pos, r) => [(pos[0] - r.x) / r.w, (pos[1] - r.y) / r.h];
		const toNorm = (pos, r) => toNormRaw(pos, r).map(clamp01);
		const toPx = (n, r) => [r.x + n[0] * r.w, r.y + n[1] * r.h];

		// which endpoints a plain drag may grab
		const grabbable = () => (view.drag === "from (original)" ? [0]
			: view.drag === "to (destination)" ? [1] : [0, 1]);

		const nearest = (pos, L, ends) => {
			let best = null, bestD = HIT_RADIUS;
			getPairs().forEach((pair, i) => {
				for (const e of ends) {
					const [x, y] = toPx(pair[e], rectFor(L, e));
					const d = Math.hypot(pos[0] - x, pos[1] - y);
					// <= so that on a pinned pair (from == to) the LAST end
					// checked wins; ends is ordered [0, 1], so that is "to"
					if (d <= bestD) { best = { index: i, end: e }; bestD = d; }
				}
			});
			return best;
		};

		// --- selection --------------------------------------------------------
		const selKey = (i, e) => `${i}:${e}`;
		// stale keys (a pair was removed or the widget was hand-edited) are
		// filtered at read time instead of tracked at write time
		const selectedItems = () => {
			const n = getPairs().length;
			const items = [];
			for (const k of selection) {
				const [i, e] = k.split(":").map(Number);
				if (i < n && (e === 0 || e === 1)) items.push({ index: i, end: e });
			}
			return items;
		};

		// centroid (display px) of the selected endpoints in each pane, plus the
		// rotate/scale handle floating above the busiest pane
		const selectionGeometry = (L) => {
			const items = selectedItems();
			if (items.length < 2) return null;
			const sums = {}, counts = {};
			for (const it of items) {
				const p = toPx(getPairs()[it.index][it.end], rectFor(L, it.end));
				const pane = paneOf(L, it.end);
				sums[pane] = sums[pane] || [0, 0];
				sums[pane][0] += p[0]; sums[pane][1] += p[1];
				counts[pane] = (counts[pane] || 0) + 1;
			}
			const centroids = {};
			for (const pane of Object.keys(sums)) {
				centroids[pane] = [sums[pane][0] / counts[pane],
				                   sums[pane][1] / counts[pane]];
			}
			const handlePane = (counts[1] || 0) >= (counts[0] || 0) && 1 in centroids ? 1 : 0;
			const c = centroids[handlePane];
			return { items, centroids, handlePane,
			         handle: [c[0], Math.max(12, c[1] - HANDLE_OFFSET)] };
		};

		const startSelDrag = (L, kind, pos, geo) => {
			selDrag = {
				kind,
				startPos: pos.slice(),
				handleStart: pos.slice(),
				handlePane: geo.handlePane,
				centroids: geo.centroids,
				items: geo.items.map((it) => ({
					...it,
					startPx: toPx(getPairs()[it.index][it.end], rectFor(L, it.end)),
				})),
			};
		};

		const applySelDrag = (L, pos) => {
			const pairs = deepPairs();
			if (selDrag.kind === "move") {
				const dx = pos[0] - selDrag.startPos[0];
				const dy = pos[1] - selDrag.startPos[1];
				for (const it of selDrag.items) {
					const p = [it.startPx[0] + dx, it.startPx[1] + dy];
					pairs[it.index][it.end] = toNorm(p, rectFor(L, it.end));
				}
			} else {
				// similarity about each pane's own centroid, driven by the
				// handle pane's angle/scale. Display px scale uniformly onto
				// image px (fitRect keeps the aspect), so this is a TRUE
				// rotation on the picture, not a normalized-space shear.
				const c = selDrag.centroids[selDrag.handlePane];
				const v0 = [selDrag.handleStart[0] - c[0], selDrag.handleStart[1] - c[1]];
				const v1 = [pos[0] - c[0], pos[1] - c[1]];
				const l0 = Math.max(1e-6, Math.hypot(v0[0], v0[1]));
				const s = Math.hypot(v1[0], v1[1]) / l0;
				const da = Math.atan2(v1[1], v1[0]) - Math.atan2(v0[1], v0[0]);
				const cos = Math.cos(da) * s, sin = Math.sin(da) * s;
				for (const it of selDrag.items) {
					const cc = selDrag.centroids[paneOf(L, it.end)];
					const rx = it.startPx[0] - cc[0], ry = it.startPx[1] - cc[1];
					const p = [cc[0] + rx * cos - ry * sin, cc[1] + rx * sin + ry * cos];
					pairs[it.index][it.end] = toNorm(p, rectFor(L, it.end));
				}
			}
			setPairs(pairs, false);
		};

		const finishBand = (L) => {
			const r = { x: Math.min(band.x0, band.x1), y: Math.min(band.y0, band.y1),
			            w: Math.abs(band.x1 - band.x0), h: Math.abs(band.y1 - band.y0) };
			band = null;
			const ends = grabbable();
			getPairs().forEach((pair, i) => {
				for (const e of ends) {
					const p = toPx(pair[e], rectFor(L, e));
					if (p[0] >= r.x && p[0] <= r.x + r.w &&
							p[1] >= r.y && p[1] <= r.y + r.h) {
						selection.add(selKey(i, e));
					}
				}
			});
			redraw();
		};

		// --- curve pair -------------------------------------------------------
		let curveWidget = null;
		const cancelCurve = () => {
			curve = null;
			if (curveWidget) curveWidget.value = false;
			redraw();
		};
		const finishStroke = (L) => {
			const end = curve.phase === 0 ? 0 : 1;
			const r = rectFor(L, end);
			const stroke = curve.stroke || [];
			curve.stroke = null;
			if (stroke.length < 2) {
				// a CLICK, not a stroke - in phase 0 a click near one of the
				// image's borders selects that whole border as the source
				// curve (the flag/cloth case: map the picture's own edge onto
				// a drawn curve)
				const b = stroke.length && curve.phase === 0
					? borderNear(stroke[0], r) : null;
				if (b) {
					const d = endDims(0);
					const ax = d.w > 1 ? d.w - 1 : 1, ay = d.h > 1 ? d.h - 1 : 1;
					curve.ptsA = BORDER_ENDS[b].map((n) => [n[0] * ax, n[1] * ay]);
					curve.border = b;
					curve.phase = 1;
				}
				redraw();
				return;
			}
			// display -> PIXEL coordinates of this end's frame: arc-length
			// uniformity must be measured in pixels (normalized space is
			// aspect-squashed and would stretch spacing along the long side)
			const d = endDims(end);
			const sx = d.w > 1 ? d.w - 1 : 1, sy = d.h > 1 ? d.h - 1 : 1;
			const px = stroke.map((p) => {
				const n = toNorm(p, r);
				return [n[0] * sx, n[1] * sy];
			});
			if (curve.phase === 0) {
				curve.ptsA = px;
				curve.phase = 1;
				redraw();
				return;
			}
			const k = Math.max(2, Math.min(64, Math.round(view.curveN) || 8));
			const dA = endDims(0);
			const ax = dA.w > 1 ? dA.w - 1 : 1, ay = dA.h > 1 ? dA.h - 1 : 1;
			const A = resampleCurve(curve.ptsA, k);
			const B = resampleCurve(px, k);
			const pairs = getPairs().slice();
			for (let j = 0; j < k; j++) {
				pairs.push([[clamp01(A[j][0] / ax), clamp01(A[j][1] / ay)],
				            [clamp01(B[j][0] / sx), clamp01(B[j][1] / sy)]]);
			}
			curve = null;
			if (curveWidget) curveWidget.value = false;
			setPairs(pairs, true);
		};

		// --- drawing ----------------------------------------------------------
		const drawArrow = (ctx, x0, y0, x1, y1, alpha) => {
			const len = Math.hypot(x1 - x0, y1 - y0);
			if (len < 1.5) return;
			ctx.globalAlpha = alpha;
			ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			ctx.moveTo(x0, y0);
			ctx.lineTo(x1, y1);
			ctx.stroke();
			const ang = Math.atan2(y1 - y0, x1 - x0);
			const head = Math.min(9, len * 0.4);
			ctx.beginPath();
			ctx.moveTo(x1, y1);
			ctx.lineTo(x1 - head * Math.cos(ang - 0.4), y1 - head * Math.sin(ang - 0.4));
			ctx.lineTo(x1 - head * Math.cos(ang + 0.4), y1 - head * Math.sin(ang + 0.4));
			ctx.closePath();
			ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
			ctx.fill();
			ctx.globalAlpha = 1;
		};

		// A regular grid in the SOURCE pushed forward into the DESTINATION pane -
		// the classic picture of what the warp does to the image.
		const drawDeformationGrid = (ctx, L, model) => {
			const iw = img.naturalWidth || 1, ih = img.naturalHeight || 1;
			const sx = iw > 1 ? iw - 1 : 1, sy = ih > 1 ? ih - 1 : 1;
			const d = dstDims();
			const dx = d.w > 1 ? d.w - 1 : 1, dy = d.h > 1 ? d.h - 1 : 1;
			const r = L.dst;
			ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
			ctx.lineWidth = 1;
			ctx.beginPath();
			for (let i = 0; i <= GRID_LINES; i++) {
				for (const swap of [false, true]) {
					for (let j = 0; j <= GRID_LINES * 3; j++) {
						const a = i / GRID_LINES, b = j / (GRID_LINES * 3);
						const px = (swap ? b : a) * sx, py = (swap ? a : b) * sy;
						const [ox, oy] = model.forward(px, py);
						const x = r.x + (ox / dx) * r.w, y = r.y + (oy / dy) * r.h;
						if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
					}
				}
			}
			ctx.stroke();
		};

		let banner = "";
		const drawWarp = (ctx, L, model, dpr) => {
			banner = "";
			if (!model) return false;
			if (!model.ok) { banner = model.reason; return false; }
			if (!view.preview) return false;
			const iw = img.naturalWidth || 1, ih = img.naturalHeight || 1;
			if (!warpGL.isAvailable() || !warpGL.uploadTexture(node, img)) {
				// no WebGL: show the ORIGINAL plus the deformation grid rather
				// than pixels we cannot produce
				banner = "WebGL unavailable - showing the deformation grid only";
				drawDeformationGrid(ctx, L, model);
				return false;
			}
			const r = L.dst;
			const d = dstDims();
			const cols = Math.max(8, Math.min(64, Math.round(r.w / CELL_PX)));
			const rows = Math.max(8, Math.min(64, Math.round(r.h / CELL_PX)));
			const mesh = meshFromModel(model.backward, iw, ih, cols, rows, d.w, d.h);
			const dw = Math.max(1, Math.round(r.w * dpr));
			const dh = Math.max(1, Math.round(r.h * dpr));
			// background mode: transparent outside the warped image, so the
			// un-warped background shows through - the whole point of the paste
			const glCanvas = warpGL.render(node, mesh, dw, dh, !L.split);
			if (!glCanvas) return false;
			const src = warpGL.renderRect(dw, dh);
			ctx.globalAlpha = Math.max(0, Math.min(1, view.opacity));
			ctx.drawImage(glCanvas, src.x, src.y, src.w, src.h, r.x, r.y, r.w, r.h);
			ctx.globalAlpha = 1;
			banner = "live preview (regularization 0) - the workflow's own warp "
			         + "output is the ground truth";
			return true;
		};

		const paneLabel = (ctx, r, text) => {
			ctx.font = "10px sans-serif";
			ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
			const tw = ctx.measureText(text).width;
			ctx.fillRect(r.x + 3, r.y + r.h - 17, tw + 8, 14);
			ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
			ctx.fillText(text, r.x + 7, r.y + r.h - 6);
		};

		const drawPolyline = (ctx, pts) => {
			if (pts.length < 2) return;
			ctx.strokeStyle = CURVE_COLOR;
			ctx.lineWidth = 2;
			ctx.beginPath();
			ctx.moveTo(pts[0][0], pts[0][1]);
			for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
			ctx.stroke();
		};

		const redraw = () => {
			const a = area();
			const dpr = window.devicePixelRatio || 1;
			const bw = Math.round(a.w * dpr), bh = Math.round(a.h * dpr);
			if (canvas.width !== bw || canvas.height !== bh) {
				canvas.width = bw;
				canvas.height = bh;
			}
			const ctx = canvas.getContext("2d");
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.clearRect(0, 0, a.w, a.h);
			ctx.fillStyle = "#181818";
			ctx.fillRect(0, 0, a.w, a.h);

			if (!(img.complete && img.naturalWidth)) {
				ctx.fillStyle = "#888";
				ctx.font = "12px sans-serif";
				ctx.textAlign = "center";
				const cx = a.w / 2, cy = a.h / 2;
				ctx.fillText("Run the workflow once to load the image.", cx, cy - 18);
				ctx.fillText("Left-click: add a pinned pair / drag an end.", cx, cy);
				ctx.fillText("Shift+click: remove. Alt+drag: move both.", cx, cy + 18);
				ctx.fillText("Ctrl+click/drag: select many. Shapes & curves below.", cx, cy + 36);
				ctx.textAlign = "left";
				return;
			}

			const L = layout();
			ctx.drawImage(img, L.src.x, L.src.y, L.src.w, L.src.h);
			if (L.split) {
				ctx.drawImage(bgImg, L.dst.x, L.dst.y, L.dst.w, L.dst.h);
			}
			const model = currentModel();
			drawWarp(ctx, L, model, dpr);
			if (view.mesh && model?.ok) drawDeformationGrid(ctx, L, model);
			if (L.split) {
				paneLabel(ctx, L.src, "image (from)");
				paneLabel(ctx, L.dst, "background (to)");
			}

			const pairs = getPairs();
			if (!pairs.length) {
				banner = banner || (L.split
					? "click either pane to add a control point pair"
					: "click the image to add a control point");
			} else if (model && !model.ok && !banner) {
				banner = model.reason;
			}

			const ends = grabbable();
			const dim = (end) => (ends.includes(end) ? 1 : 0.4);
			ctx.font = "11px sans-serif";
			pairs.forEach((pair, i) => {
				const [fx, fy] = toPx(pair[0], L.src);
				const [tx, ty] = toPx(pair[1], L.dst);
				drawArrow(ctx, fx, fy, tx, ty, Math.max(dim(0), dim(1)));
				for (const [end, x, y, color] of
						[[0, fx, fy, FROM_COLOR], [1, tx, ty, TO_COLOR]]) {
					ctx.globalAlpha = dim(end);
					ctx.beginPath();
					ctx.arc(x, y, 5, 0, Math.PI * 2);
					ctx.fillStyle = color;
					ctx.fill();
					ctx.strokeStyle = "#000";
					ctx.lineWidth = 1;
					ctx.stroke();
					if (selection.has(selKey(i, end))) {
						ctx.setLineDash([3, 3]);
						ctx.beginPath();
						ctx.arc(x, y, 8.5, 0, Math.PI * 2);
						ctx.strokeStyle = "#fff";
						ctx.lineWidth = 1.5;
						ctx.stroke();
						ctx.setLineDash([]);
					}
					if (hover && hover.index === i && hover.end === end) {
						ctx.beginPath();
						ctx.arc(x, y, 9, 0, Math.PI * 2);
						ctx.strokeStyle = "#fff";
						ctx.lineWidth = 1.5;
						ctx.stroke();
					}
					ctx.globalAlpha = 1;
				}
				ctx.fillStyle = "#fff";
				ctx.fillText(String(i + 1), tx + 8, ty - 8);
				// in split mode the arrow crosses the pane gap, so the pairing
				// needs the number on BOTH ends to stay legible
				if (L.split) ctx.fillText(String(i + 1), fx + 8, fy - 8);
			});

			// selection centroid + rotate/scale handle
			const geo = selectionGeometry(L);
			if (geo) {
				const c = geo.centroids[geo.handlePane];
				ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
				ctx.lineWidth = 1;
				ctx.beginPath();
				ctx.moveTo(c[0] - 5, c[1]); ctx.lineTo(c[0] + 5, c[1]);
				ctx.moveTo(c[0], c[1] - 5); ctx.lineTo(c[0], c[1] + 5);
				ctx.stroke();
				ctx.beginPath();
				ctx.moveTo(c[0], c[1]);
				ctx.lineTo(geo.handle[0], geo.handle[1]);
				ctx.stroke();
				ctx.beginPath();
				ctx.arc(geo.handle[0], geo.handle[1], 6, 0, Math.PI * 2);
				ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
				ctx.fill();
				ctx.strokeStyle = "#000";
				ctx.stroke();
				banner = `${geo.items.length} selected - drag a selected point to `
				         + "move all, drag the round handle to rotate/scale, "
				         + "shift+click one to delete all, click empty to clear";
			}

			// rubber band
			if (band) {
				ctx.fillStyle = "rgba(120, 180, 255, 0.15)";
				ctx.strokeStyle = "rgba(120, 180, 255, 0.8)";
				ctx.lineWidth = 1;
				const rx = Math.min(band.x0, band.x1), ry = Math.min(band.y0, band.y1);
				const rw = Math.abs(band.x1 - band.x0), rh = Math.abs(band.y1 - band.y0);
				ctx.fillRect(rx, ry, rw, rh);
				ctx.strokeRect(rx, ry, rw, rh);
			}

			// curve strokes: the stored first curve (converted back to display)
			// while the second is awaited, the live stroke being drawn, and in
			// phase 0 the border a click would select
			if (curve) {
				if (curve.hover && curve.phase === 0 && !curve.stroke) {
					drawPolyline(ctx, BORDER_ENDS[curve.hover].map((n) =>
						toPx(n, rectFor(L, 0))));
				}
				if (curve.ptsA) {
					const d = endDims(0);
					const ax = d.w > 1 ? d.w - 1 : 1, ay = d.h > 1 ? d.h - 1 : 1;
					drawPolyline(ctx, curve.ptsA.map((p) =>
						toPx([p[0] / ax, p[1] / ay], rectFor(L, 0))));
				}
				if (curve.stroke) drawPolyline(ctx, curve.stroke);
				const dirs = "top/bottom run left->right, sides top->bottom";
				banner = curve.phase === 0
					? (curve.hover
						? `click to map the image's ${curve.hover} border (${dirs})`
						: (L.split
							? "curve 1/2: draw along the IMAGE pane - or click one of its borders"
							: "curve 1/2: draw along a feature - or click an image border"))
					: (curve.border
						? `curve 2/2: draw where the ${curve.border} border lands (${dirs})`
						: (L.split
							? "curve 2/2: draw where it lands ON THE BACKGROUND pane "
							  + "(same direction)"
							: "curve 2/2: draw where it must land (same direction)"));
			}

			if (banner) {
				ctx.font = "11px sans-serif";
				ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
				const tw = ctx.measureText(banner).width;
				ctx.fillRect(4, 4, tw + 10, 17);
				ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
				ctx.fillText(banner, 9, 16);
			}
		};

		// repaint on container resize (node resize / zoom relayout), coalesced
		// into one rAF and skipped on an unchanged size so the observer can
		// never drive a resize feedback loop
		let rafPending = false;
		let lastW = 0, lastH = 0;
		if (typeof ResizeObserver !== "undefined") {
			new ResizeObserver(() => {
				const a = area();
				if (a.w === lastW && a.h === lastH) return;
				lastW = a.w; lastH = a.h;
				if (rafPending) return;
				rafPending = true;
				requestAnimationFrame(() => { rafPending = false; redraw(); });
			}).observe(container);
		}

		// --- view widgets ------------------------------------------------------
		// LiteGraph widgets, NOT a DOM toolbar: a DOM <select> click inside the
		// widget reaches the LiteGraph canvas's own processMouseDown in legacy
		// mode and starts a node-resize drag.
		const addView = (type, name, value, callback, options) => {
			const w = node.addWidget(type, name, value, callback, options);
			// BOTH flags: the serializer checks the widget property, the layout
			// reads options (same as web/cv_flags.js and web/cv_policy.js).
			w.serialize = false;
			w.options = w.options || {};
			w.options.serialize = false;
			return w;
		};
		addView("combo", "drag edits", view.drag,
			(v) => { view.drag = v; saveView(); redraw(); }, { values: DRAG_MODES });
		addView("toggle", "preview warp", view.preview,
			(v) => { view.preview = !!v; saveView(); redraw(); });
		addView("number", "warp opacity", view.opacity,
			(v) => { view.opacity = clamp01(v); saveView(); redraw(); },
			// litegraph "step" is 1/10 of a drag unit -> 0.5 means 0.05
			{ min: 0, max: 1, step: 0.5, precision: 2 });
		addView("combo", "preview model", view.model,
			(v) => { view.model = v; fitKey = null; saveView(); redraw(); },
			{ values: MODELS.slice() });
		addView("toggle", "show mesh", view.mesh,
			(v) => { view.mesh = !!v; saveView(); redraw(); });
		addView("button", "pin image corners", null, () => {
			// from == to on every corner: in single-image mode that pins the
			// frame; in background mode it maps the image's corners onto the
			// background's (a full-frame paste to start deforming from)
			const pairs = getPairs().slice();
			for (const c of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
				if (!pairs.some(([f]) => f[0] === c[0] && f[1] === c[1])) {
					pairs.push([[c[0], c[1]], [c[0], c[1]]]);
				}
			}
			setPairs(pairs, true);
		});
		addView("combo", "add shape", view.shape,
			(v) => { view.shape = v; saveView(); }, { values: SHAPES.slice() });
		addView("button", "add shape points", null, () => {
			// a pinned scaffold to deform (drag targets one by one, or select
			// them all and use the group move / rotate-scale handle)
			const pairs = getPairs().slice();
			for (const c of shapePoints(view.shape)) {
				const p = [round4(c[0]), round4(c[1])];
				if (!pairs.some(([f]) => f[0] === p[0] && f[1] === p[1])) {
					pairs.push([p.slice(), p.slice()]);
				}
			}
			setPairs(pairs, true);
		});
		addView("number", "curve points", view.curveN,
			(v) => {
				view.curveN = Math.max(2, Math.min(64, Math.round(v) || 8));
				saveView();
			},
			{ min: 2, max: 64, step: 10, precision: 0 });
		curveWidget = addView("toggle", "draw curve pair", false, (v) => {
			if (v && img.naturalWidth) {
				curve = { phase: 0, stroke: null, ptsA: null,
				          border: null, hover: null };
			} else {
				curve = null;
				if (curveWidget) curveWidget.value = false;
			}
			redraw();
		});

		// --- interaction -------------------------------------------------------
		const eventPos = (e) => [e.offsetX, e.offsetY];

		canvas.addEventListener("pointerdown", (e) => {
			if (e.button !== 0) return;
			if (!img.naturalWidth) return;
			e.stopPropagation();
			e.preventDefault();
			canvas.setPointerCapture?.(e.pointerId);
			const L = layout();
			const pos = eventPos(e);
			const pairs = getPairs().slice();

			// --- curve capture takes the whole surface while armed -----------
			if (curve) {
				const end = curve.phase === 0 ? 0 : 1;
				const r = rectFor(L, end);
				// border clicks may land just OUTSIDE the letterboxed rect
				if (L.split && !inRect(pos, r) &&
						!(curve.phase === 0 && borderNear(pos, r))) {
					return; // wrong pane
				}
				curve.stroke = [pos];
				redraw();
				return;
			}

			if (e.shiftKey) {
				// removing half a correspondence is meaningless - drop the pair.
				// On a SELECTED endpoint the delete applies to the WHOLE
				// selection (a pair is selected when either of its ends is),
				// which is the bulk-delete for scaffolds and curve strokes.
				const hit = nearest(pos, L, [0, 1]);
				if (hit) {
					if (selection.has(selKey(hit.index, 0)) ||
							selection.has(selKey(hit.index, 1))) {
						const doomed = new Set(
							selectedItems().map((it) => it.index));
						for (const i of [...doomed].sort((a, b) => b - a)) {
							pairs.splice(i, 1);
						}
					} else {
						pairs.splice(hit.index, 1);
					}
					selection.clear();   // indices shifted
					setPairs(pairs, true);
				}
				drag = null;
				return;
			}

			// --- selection: ctrl+click toggles, ctrl+drag rubber-bands -------
			if (e.ctrlKey || e.metaKey) {
				const hit = nearest(pos, L, grabbable());
				if (hit) {
					const k = selKey(hit.index, hit.end);
					selection.has(k) ? selection.delete(k) : selection.add(k);
				} else {
					band = { x0: pos[0], y0: pos[1], x1: pos[0], y1: pos[1] };
				}
				drag = null;
				redraw();
				return;
			}

			// --- the rotate/scale handle -------------------------------------
			const geo = selectionGeometry(L);
			if (geo && Math.hypot(pos[0] - geo.handle[0],
			                      pos[1] - geo.handle[1]) <= HIT_RADIUS) {
				startSelDrag(L, "xform", pos, geo);
				// the handle's lever arm is measured from the centroid to where
				// the DRAG started, not to the handle's resting spot, so there
				// is no jump on grab
				return;
			}

			const now = Date.now();
			const dbl = now - lastClick < 350;
			lastClick = now;
			const hit = nearest(pos, L, e.altKey ? [0, 1] : grabbable());

			// --- dragging any selected endpoint moves the whole selection ----
			if (hit && !dbl && geo && selection.has(selKey(hit.index, hit.end))) {
				startSelDrag(L, "move", pos, geo);
				return;
			}

			if (hit && dbl) {
				// re-pin: zero this pair's displacement (in background mode:
				// back to the same NORMALIZED spot of both frames)
				pairs[hit.index] = [pairs[hit.index][0].slice(),
				                    pairs[hit.index][0].slice()];
				setPairs(pairs, true);
				drag = null;
				return;
			}
			if (hit) {
				selection.clear();  // a plain grab of an unselected point
				drag = { index: hit.index, end: e.altKey ? "both" : hit.end };
				if (drag.end === "both") {
					// both ends follow the pointer by the same NORMALIZED delta,
					// measured in the pane the grab happened in - each end then
					// applies it inside its own frame
					drag.homeRect = rectFor(L, hit.end);
					drag.startN = toNormRaw(pos, drag.homeRect);
					drag.start = pairs[hit.index].map((p) => p.slice());
				}
				redraw();
				return;
			}
			// empty space with an active selection: the click CLEARS it (adding
			// a stray pair under a selection gesture would be worse)
			if (selection.size) {
				selection.clear();
				drag = null;
				redraw();
				return;
			}
			// empty space: a new PINNED pair (from == to, same normalized spot
			// of each frame), then drag the end that lives in the clicked pane -
			// one press-drag-release authors a whole correspondence. In
			// single-image mode the pane is ambiguous, so the "drag edits" mode
			// decides; clicks in the dead gap between panes do nothing.
			let end;
			if (L.split) {
				end = inRect(pos, L.src) ? 0 : inRect(pos, L.dst) ? 1 : null;
				if (end === null) { drag = null; return; }
			} else {
				end = view.drag === "from (original)" ? 0 : 1;
			}
			const n = toNorm(pos, rectFor(L, end));
			pairs.push([n.slice(), n.slice()]);
			drag = { index: pairs.length - 1, end };
			setPairs(pairs, false);
		});

		canvas.addEventListener("pointermove", (e) => {
			const L = layout();
			const pos = eventPos(e);
			if (curve && curve.stroke) {
				if ((e.buttons & 1) === 0) { finishStroke(L); return; }
				e.stopPropagation();
				e.preventDefault();
				const last = curve.stroke[curve.stroke.length - 1];
				if (Math.hypot(pos[0] - last[0], pos[1] - last[1]) > 2) {
					curve.stroke.push(pos);
					redraw();
				}
				return;
			}
			if (curve && curve.phase === 0) {
				// hover feedback: which border a click would select
				const hb = borderNear(pos, rectFor(L, 0));
				if (hb !== curve.hover) { curve.hover = hb; redraw(); }
				return;
			}
			if (band) {
				if ((e.buttons & 1) === 0) { finishBand(L); return; }
				e.stopPropagation();
				e.preventDefault();
				band.x1 = pos[0];
				band.y1 = pos[1];
				redraw();
				return;
			}
			if (selDrag) {
				if ((e.buttons & 1) === 0) { selDrag = null; return; }
				e.stopPropagation();
				e.preventDefault();
				applySelDrag(L, pos);
				return;
			}
			if (!drag) {
				// hover feedback removes the last doubt about which end a drag
				// would grab
				const hit = img.naturalWidth && !curve
					? nearest(pos, L, grabbable()) : null;
				const changed = (hit?.index !== hover?.index) || (hit?.end !== hover?.end);
				hover = hit;
				canvas.style.cursor = hit ? "move" : "crosshair";
				if (changed) redraw();
				return;
			}
			// the left button came up without us seeing the release - drop it
			if ((e.buttons & 1) === 0) { drag = null; return; }
			e.stopPropagation();
			e.preventDefault();
			const pairs = getPairs().slice();
			if (drag.index >= pairs.length) { drag = null; return; }
			const pair = [pairs[drag.index][0].slice(), pairs[drag.index][1].slice()];
			if (drag.end === "both") {
				const n = toNormRaw(pos, drag.homeRect);
				const dx = n[0] - drag.startN[0], dy = n[1] - drag.startN[1];
				for (const end of [0, 1]) {
					pair[end] = [clamp01(drag.start[end][0] + dx),
					             clamp01(drag.start[end][1] + dy)];
				}
			} else {
				pair[drag.end] = toNorm(pos, rectFor(L, drag.end));
			}
			pairs[drag.index] = pair;
			setPairs(pairs, false); // committed on release, not per move
		});

		const endDrag = (e) => {
			canvas.releasePointerCapture?.(e.pointerId);
			const L = layout();
			if (curve && curve.stroke) { finishStroke(L); return; }
			if (band) { finishBand(L); return; }
			if (selDrag) {
				selDrag = null;
				setPairs(getPairs(), true);
				return;
			}
			if (!drag) return;
			drag = null;
			setPairs(getPairs(), true);
		};
		canvas.addEventListener("pointerup", endDrag);
		canvas.addEventListener("pointercancel", endDrag);
		canvas.addEventListener("pointerleave", () => {
			if (!drag && !selDrag && !band && hover) { hover = null; redraw(); }
		});

		// --- image sources -----------------------------------------------------
		for (const im of [img, bgImg]) {
			im.addEventListener("load", () => { fitKey = null; redraw(); });
			im.addEventListener("error", () => {
				// e.g. a temp image gone after a server restart
				im.removeAttribute("src");
				fitKey = null;
				redraw();
			});
		}
		const setImage = (im, d) => {
			if (!d?.filename) return;
			im.src = viewURL(d);
		};
		const onExecuted = node.onExecuted;
		node.onExecuted = function (message) {
			const r = onExecuted?.apply(this, arguments);
			const d = message?.annotate_images?.[0];
			if (d) {
				node.properties = node.properties || {};
				node.properties.annotate_image = d; // survives save/reload
				setImage(img, d);
			}
			const b = message?.annotate_background?.[0];
			if (b) {
				node.properties = node.properties || {};
				node.properties.annotate_background = b;
				setImage(bgImg, b);
			} else if (d) {
				// this run had NO background (input disconnected): fall back to
				// the single-image mode instead of keeping a stale pane around
				delete node.properties?.annotate_background;
				bgImg.removeAttribute("src");
				fitKey = null;
				redraw();
			}
			return r;
		};
		const onConfigure = node.onConfigure;
		node.onConfigure = function (info) {
			const r = onConfigure?.apply(this, arguments);
			Object.assign(view, DEFAULT_VIEW, info?.properties?.cv_warp_view || {});
			for (const w of node.widgets || []) {
				if (w.name === "drag edits") w.value = view.drag;
				else if (w.name === "preview warp") w.value = view.preview;
				else if (w.name === "warp opacity") w.value = view.opacity;
				else if (w.name === "preview model") w.value = view.model;
				else if (w.name === "show mesh") w.value = view.mesh;
				else if (w.name === "add shape") w.value = view.shape;
				else if (w.name === "curve points") w.value = view.curveN;
				else if (w.name === "draw curve pair") w.value = false;
			}
			selection.clear();
			selDrag = null;
			band = null;
			curve = null;
			fitKey = null;
			setImage(img, info?.properties?.annotate_image);
			if (info?.properties?.annotate_background) {
				setImage(bgImg, info.properties.annotate_background);
			} else {
				bgImg.removeAttribute("src");
			}
			redraw();
			return r;
		};
		const onRemoved = node.onRemoved;
		node.onRemoved = function () {
			warpGL.release(node);
			return onRemoved?.apply(this, arguments);
		};
		if (node.properties?.annotate_image) setImage(img, node.properties.annotate_image);
		if (node.properties?.annotate_background) {
			setImage(bgImg, node.properties.annotate_background);
		}
		redraw();
	},
});
