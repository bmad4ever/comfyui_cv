// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Annotation surface for the "CV Annotate Points" node.
//
// The Python node previews its input image under the custom ui key
// "annotate_images". This extension draws that image - plus the editable
// points - on a real <canvas> added as a DOM widget (node.addDOMWidget).
// ComfyUI positions and scales DOM widgets inside the node body in BOTH the
// classic LiteGraph rendering AND "Modern Node Design (Nodes 2.0)", where node
// bodies are DOM and the old per-node onDrawForeground / onMouseDown canvas
// hooks are never exercised - so a DOM-widget canvas is the only surface that
// renders in both modes. Pointer events are read in element-local coordinates
// (e.offsetX/offsetY), so there is no screen/layout coordinate split.
//
// Interaction: left-click adds a point, dragging moves the nearest one,
// SHIFT+click removes it. Points are stored normalized (0-1 of the image
// size) as JSON in the node's "points" STRING widget - the value the
// Python side parses - so annotations survive saves/reloads and can be
// edited by hand.
//
// Once the outline is CLOSED (3+ points) a new point is inserted into the
// NEAREST EDGE rather than appended at the end: refining a shape is then a
// matter of clicking where the outline is wrong, instead of deleting and
// re-clicking every point after it just to keep the ring in order. The edge
// that would receive the point is highlighted on hover. ALT+click still
// appends at the end (the old behaviour), which is the only way to grow an
// ordered/open sequence once the third point exists.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { nearestEdge, insertionIndex } from "./cvlib/ring_insert.js";

const EXEC_CLASS = "CV_AnnotatePoints"; // image arrives via execution
const HIT_RADIUS = 12; // element-space (CSS) units
const MIN_AREA_H = 160; // minimum canvas height in px

function viewURL(d) {
	return api.apiURL(
		`/view?filename=${encodeURIComponent(d.filename)}` +
		`&type=${encodeURIComponent(d.type ?? "temp")}` +
		`&subfolder=${encodeURIComponent(d.subfolder ?? "")}` +
		`&t=${Date.now()}`
	);
}

app.registerExtension({
	name: "opencv_comfyui.annotate_points",

	async nodeCreated(node) {
		if (node.comfyClass !== EXEC_CLASS) return;
		const pointsWidget = node.widgets?.find((w) => w.name === "points");
		if (!pointsWidget) return;

		const img = new Image();
		let dragIndex = -1;
		let hoverEdge = -1; // edge a click would split, highlighted while hovering

		// --- the DOM-widget canvas ------------------------------------------
		// The widget element is a fixed-position-context container; the canvas is
		// ABSOLUTELY positioned inside it so it contributes nothing to the
		// container's flow height. In Modern Node Design (Nodes 2.0) the node
		// height is measured from the widget element's content - an in-flow canvas
		// whose intrinsic/attribute size we set each redraw fed that measurement
		// back and the node grew without bound. Out-of-flow canvas + overflow
		// hidden breaks that loop; the frontend alone controls the container size.
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
		canvas.style.touchAction = "none"; // we handle pointer gestures ourselves
		container.appendChild(canvas);

		// A neutral type (not "canvas") so the Vue node renderer treats it as a
		// plain DOM widget and does not route it through its own draw-canvas path.
		const widget = node.addDOMWidget("annotate_surface", "opencv_annotate", container, {
			serialize: false,
			hideOnZoom: false,
		});
		// BOTH flags: `options.serialize` only keeps the widget out of the API
		// prompt - workflow persistence is the widget-level property, and
		// without it this editing surface saved its empty value as one extra
		// widgets_values entry (see web/cvlib/widget_values.js). The points
		// themselves live in the `points` string widget.
		widget.serialize = false;
		// Let the surface flex: the frontend distributes the node's free vertical
		// space to DOM widgets between minHeight and maxHeight. We only pin a
		// minimum so the node stays freely resizable; the image is letterboxed
		// inside whatever box the user gives it.
		widget.computeLayoutSize = () => ({
			minHeight: MIN_AREA_H,
			maxHeight: 1e6,
			minWidth: 20,
			maxWidth: 1e6,
		});

		// --- model ----------------------------------------------------------
		const getPoints = () => {
			try {
				const v = JSON.parse(pointsWidget.value || "[]");
				return Array.isArray(v)
					? v.filter((p) => Array.isArray(p) && p.length === 2)
					: [];
			} catch (e) {
				return [];
			}
		};
		const setPoints = (pts) => {
			// any edit renumbers the ring, so a remembered edge index would now
			// highlight a different edge - it is re-derived on the next move
			hoverEdge = -1;
			pointsWidget.value = JSON.stringify(
				pts.map(([x, y]) => [Math.round(x * 1e4) / 1e4, Math.round(y * 1e4) / 1e4])
			);
			pointsWidget.callback?.(pointsWidget.value);
			redraw();
		};

		// --- geometry (all in element-local CSS pixels) ----------------------
		// the canvas element rect (its own coordinate space)
		const area = () => ({
			x: 0,
			y: 0,
			w: Math.max(1, canvas.clientWidth || canvas.width),
			h: Math.max(1, canvas.clientHeight || canvas.height),
		});
		// the letterboxed image rect inside the area
		const imageRect = (a) => {
			const iw = img.naturalWidth || 1;
			const ih = img.naturalHeight || 1;
			const s = Math.min(a.w / iw, a.h / ih);
			const w = iw * s, h = ih * s;
			return { x: a.x + (a.w - w) / 2, y: a.y + (a.h - h) / 2, w, h };
		};
		const toNorm = (pos, r) => [
			Math.min(1, Math.max(0, (pos[0] - r.x) / r.w)),
			Math.min(1, Math.max(0, (pos[1] - r.y) / r.h)),
		];
		const nearestIndex = (pos, r) => {
			let best = -1, bestD = HIT_RADIUS;
			getPoints().forEach(([nx, ny], i) => {
				const d = Math.hypot(pos[0] - (r.x + nx * r.w), pos[1] - (r.y + ny * r.h));
				if (d < bestD) { best = i; bestD = d; }
			});
			return best;
		};
		// element-space vertices of the current ring. The edge search runs in
		// THIS space, not in normalized coordinates, so "nearest" means nearest
		// on screen even when the image is letterboxed into a non-square box.
		const screenPoints = (r) =>
			getPoints().map(([nx, ny]) => [r.x + nx * r.w, r.y + ny * r.h]);

		// --- drawing ----------------------------------------------------------
		const redraw = () => {
			const a = area();
			// size the backing store to the element * devicePixelRatio for crisp
			// output, then draw in CSS pixels
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

			if (img.complete && img.naturalWidth) {
				const r = imageRect(a);
				ctx.drawImage(img, r.x, r.y, r.w, r.h);
				const pts = screenPoints(r);
				if (pts.length >= 2) {
					ctx.beginPath();
					ctx.moveTo(pts[0][0], pts[0][1]);
					for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
					if (pts.length >= 3) ctx.closePath();
					ctx.strokeStyle = "rgba(255, 220, 0, 0.9)";
					ctx.lineWidth = 1.5;
					ctx.stroke();
				}
				// the edge a click would split, so the insertion is visible
				// BEFORE it happens (the whole point of the feature is that the
				// new vertex does not land at the end of the list)
				if (hoverEdge >= 0 && hoverEdge < pts.length && pts.length >= 3) {
					const a0 = pts[hoverEdge];
					const b0 = pts[(hoverEdge + 1) % pts.length];
					ctx.beginPath();
					ctx.moveTo(a0[0], a0[1]);
					ctx.lineTo(b0[0], b0[1]);
					ctx.strokeStyle = "rgba(80, 200, 255, 0.95)";
					ctx.lineWidth = 3;
					ctx.stroke();
				}
				ctx.font = "11px sans-serif";
				pts.forEach(([x, y], i) => {
					ctx.beginPath();
					ctx.arc(x, y, 5, 0, Math.PI * 2);
					ctx.fillStyle = "rgba(255, 220, 0, 0.95)";
					ctx.fill();
					ctx.strokeStyle = "#000";
					ctx.lineWidth = 1;
					ctx.stroke();
					ctx.fillStyle = "#fff";
					ctx.fillText(String(i + 1), x + 7, y - 7);
				});
			} else {
				ctx.fillStyle = "#888";
				ctx.font = "12px sans-serif";
				ctx.textAlign = "center";
				const cx = a.x + a.w / 2;
				const cy = a.y + a.h / 2;
				ctx.fillText("Run the workflow once to load the image.", cx, cy - 9);
				ctx.fillText("Left-click: add / drag. Shift+click: remove.", cx, cy + 9);
				ctx.fillText("Closed shape: the click goes into the nearest edge (Alt: at the end).",
				             cx, cy + 27);
				ctx.textAlign = "left";
			}
		};

		// repaint when the container is resized (node resize / zoom relayout),
		// coalesced into one rAF and skipped when the size is unchanged so the
		// observer can never drive a resize feedback loop
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

		// --- interaction (pointer events on the canvas element) ---------------
		const eventPos = (e) => [e.offsetX, e.offsetY];
		canvas.addEventListener("pointerdown", (e) => {
			if (e.button !== 0) return;
			if (!img.naturalWidth) return;
			e.stopPropagation();
			e.preventDefault();
			canvas.setPointerCapture?.(e.pointerId);
			const r = imageRect(area());
			const pts = getPoints();
			const idx = nearestIndex(eventPos(e), r);
			if (e.shiftKey) {
				if (idx >= 0) {
					pts.splice(idx, 1);
					setPoints(pts);
				}
				dragIndex = -1;
				return;
			}
			dragIndex = idx;
			if (dragIndex < 0) {
				// closed outline -> the new vertex belongs to the edge the user
				// clicked next to, so the ring stays in order and refining a
				// shape never means re-clicking everything after the fix.
				// ALT forces the old append-at-the-end behaviour.
				const at = insertionIndex(screenPoints(r), eventPos(e), e.altKey);
				pts.splice(at, 0, toNorm(eventPos(e), r));
				dragIndex = at; // keep dragging the point that was just inserted
				setPoints(pts);
			}
		});
		canvas.addEventListener("pointermove", (e) => {
			if (dragIndex < 0) {
				// not dragging: preview which edge would take the next point
				if (!img.naturalWidth) return;
				const r = imageRect(area());
				const pos = eventPos(e);
				// hovering an existing vertex means "drag it", not "split an
				// edge" - do not promise an insertion that will not happen
				const edge = (e.altKey || nearestIndex(pos, r) >= 0)
					? -1
					: nearestEdge(screenPoints(r), pos);
				if (edge !== hoverEdge) { hoverEdge = edge; redraw(); }
				return;
			}
			// the left button came up without us seeing the release - drop it
			if ((e.buttons & 1) === 0) { dragIndex = -1; return; }
			e.stopPropagation();
			e.preventDefault();
			const pts = getPoints();
			if (dragIndex < pts.length) {
				pts[dragIndex] = toNorm(eventPos(e), imageRect(area()));
				setPoints(pts);
			}
		});
		const endDrag = (e) => {
			if (dragIndex < 0) return;
			dragIndex = -1;
			canvas.releasePointerCapture?.(e.pointerId);
		};
		canvas.addEventListener("pointerup", endDrag);
		canvas.addEventListener("pointercancel", endDrag);
		canvas.addEventListener("pointerleave", () => {
			if (hoverEdge < 0) return;
			hoverEdge = -1;
			redraw();
		});

		// --- image source --------------------------------------------------------
		img.addEventListener("load", () => {
			redraw();
		});
		img.addEventListener("error", () => {
			// e.g. a temp image gone after a server restart
			img.removeAttribute("src");
			redraw();
		});
		const setImage = (d) => {
			if (!d?.filename) return;
			img.src = viewURL(d);
		};
		const onExecuted = node.onExecuted;
		node.onExecuted = function (message) {
			const r = onExecuted?.apply(this, arguments);
			const d = message?.annotate_images?.[0];
			if (d) {
				node.properties = node.properties || {};
				node.properties.annotate_image = d; // survives save/reload
				setImage(d);
			}
			return r;
		};
		const onConfigure = node.onConfigure;
		node.onConfigure = function (info) {
			const r = onConfigure?.apply(this, arguments);
			setImage(info?.properties?.annotate_image);
			return r;
		};
		if (node.properties?.annotate_image) setImage(node.properties.annotate_image);
		redraw();
	},
});
