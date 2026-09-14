// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Annotation surface for the "CV Annotate Points On Model" node.
//
// A plain orbit viewer over the node's GLB, with click-to-drop anchors on the
// model surface. Deliberately NOT the calibrated viewer (web/cv_preview3d.js):
// there is no pose, no intrinsics and no lens distortion here, because you are
// pointing at geometry rather than matching footage - and a free camera is the
// fastest way to reach the far side of an object.
//
// Interaction: LEFT-CLICK the surface adds an anchor, SHIFT+click (or ALT) on
// an existing marker removes it, dragging orbits. A click is distinguished from
// an orbit by pointer travel, so the camera stays free.
//
// Picks are stored in the node's "points" STRING widget as JSON [[x, y, z], ..]
// in the coordinates the FILE stores (glTF Y-up); the Python node applies the
// axis_convention flip on the way out. Storing the file's own numbers keeps the
// widget diffable against anything a DCC would report.
//
// Three.js r160 is vendored under ./lib (imports rewritten to relative paths);
// see lib/LICENSE.three.txt. Headless tests can only verify the served file +
// API execution - interactive behaviour needs the real UI.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import * as THREE from "./lib/three.module.min.js";
import { GLTFLoader } from "./lib/GLTFLoader.js";
import { OrbitControls } from "./lib/OrbitControls.js";
import { parseAnchors } from "./cvlib/model_points.js";

const NODE_CLASS = "CV_AnnotateModelPoints";
const CLICK_SLOP = 4;      // px of pointer travel still counted as a click
const HIT_FRACTION = 0.03; // marker pick radius, as a fraction of the model size

function tempURL(name) {
	return api.apiURL(
		`/view?filename=${encodeURIComponent(name)}&type=temp&subfolder=`);
}

class AnnotateViewer {
	constructor(viewport, canvasHolder, statusEl) {
		this.viewport = viewport;
		this.statusEl = statusEl;
		this.points = [];        // THREE.Vector3, model space
		this.onChange = null;
		this.modelSize = 1;

		this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
		this.renderer.setClearColor(0x181818, 1);
		canvasHolder.appendChild(this.renderer.domElement);
		this.renderer.domElement.style.display = "block";

		this.scene = new THREE.Scene();
		this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.5));
		const sun = new THREE.DirectionalLight(0xffffff, 1.8);
		sun.position.set(5, 10, 7.5);
		this.scene.add(sun);

		this.modelRoot = new THREE.Group();   // identity: world space IS model space
		this.scene.add(this.modelRoot);
		this.markerRoot = new THREE.Group();
		this.scene.add(this.markerRoot);
		this.grid = new THREE.GridHelper(10, 10, 0x777777, 0x3a3a3a);
		this.scene.add(this.grid);
		this.axes = new THREE.AxesHelper(1);
		this.scene.add(this.axes);

		this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
		this.camera.position.set(3, 3, 5);
		this.controls = new OrbitControls(this.camera, this.renderer.domElement);
		this.controls.enableDamping = false;
		this.controls.addEventListener("change", () => this.render());

		this.loader = new GLTFLoader();
		this.raycaster = new THREE.Raycaster();
		this._bindPointer();

		// The `points` widget is the source of truth, and it changes behind our
		// back in ways there is no single event for: ComfyUI restores serialized
		// widget values AFTER onNodeCreated (so reading it once at build time
		// sees the default), a multiline textarea does not reliably fire its
		// callback per keystroke, and undo/redo and API edits fire nothing at
		// all. Watching the string on the animation frame covers every one of
		// them for the cost of a string compare.
		this._watch = null;
		const tick = () => { this._raf = requestAnimationFrame(tick); this._watch?.(); };
		tick();
	}

	dispose() {
		if (this._raf) cancelAnimationFrame(this._raf);
		this._raf = null;
		this._watch = null;
		this.renderer?.dispose?.();
	}

	setStatus(t) { if (this.statusEl) this.statusEl.textContent = t || ""; }

	// --- picking ----------------------------------------------------------
	_bindPointer() {
		const el = this.renderer.domElement;
		let down = null;
		el.addEventListener("pointerdown", (e) => {
			if (e.button !== 0) return;
			down = { x: e.clientX, y: e.clientY };
		});
		el.addEventListener("pointerup", (e) => {
			if (e.button !== 0 || !down) return;
			const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
			down = null;
			if (moved > CLICK_SLOP) return;   // that was an orbit, not a click
			this._click(e);
		});
		// keep the canvas from starting a node-resize drag in legacy LiteGraph
		for (const ev of ["pointerdown", "mousedown", "contextmenu"])
			el.addEventListener(ev, (evt) => evt.stopPropagation());
	}

	_ndc(e) {
		const r = this.renderer.domElement.getBoundingClientRect();
		return new THREE.Vector2(
			((e.clientX - r.left) / Math.max(r.width, 1)) * 2 - 1,
			-((e.clientY - r.top) / Math.max(r.height, 1)) * 2 + 1);
	}

	_click(e) {
		if (!this.modelRoot.children.length) return;
		this.raycaster.setFromCamera(this._ndc(e), this.camera);

		// SHIFT/ALT removes the marker under the cursor
		if (e.shiftKey || e.altKey) {
			const hit = this.raycaster.intersectObjects(this.markerRoot.children,
			                                            false)[0];
			if (hit) {
				const i = this.markerRoot.children.indexOf(hit.object);
				if (i >= 0) {
					this.points.splice(i, 1);
					this._rebuildMarkers();
					this._emit();
				}
				return;
			}
			// nothing under the cursor: fall back to the nearest marker to the ray
			let best = -1, bestD = this.modelSize * HIT_FRACTION * 2;
			this.points.forEach((p, i) => {
				const d = this.raycaster.ray.distanceToPoint(p);
				if (d < bestD) { bestD = d; best = i; }
			});
			if (best >= 0) {
				this.points.splice(best, 1);
				this._rebuildMarkers();
				this._emit();
			}
			return;
		}

		const hit = this.raycaster.intersectObject(this.modelRoot, true)[0];
		if (!hit) { this.setStatus("click ON the model to drop an anchor"); return; }
		this.setStatus("");
		this.points.push(hit.point.clone());
		this._rebuildMarkers();
		this._emit();
	}

	_emit() {
		this.onChange?.(this.points.map(
			(p) => [+p.x.toFixed(5), +p.y.toFixed(5), +p.z.toFixed(5)]));
	}

	// --- markers ----------------------------------------------------------
	/** Extent the markers are sized against: the model when there is one, the
	 *  points themselves before the node has ever run. Without the fallback a
	 *  pre-filled list draws at the default modelSize=1 and is invisible next
	 *  to a metres-wide point spread. */
	_sceneScale() {
		if (this.modelRoot.children.length) return this.modelSize;
		if (this.points.length > 1) {
			const b = new THREE.Box3().setFromPoints(this.points);
			return Math.max(b.getSize(new THREE.Vector3()).length(), 1e-3);
		}
		return this.modelSize;
	}

	_rebuildMarkers() {
		for (const m of [...this.markerRoot.children]) {
			this.markerRoot.remove(m);
			m.geometry?.dispose();
			m.material?.dispose();
		}
		const r = Math.max(this._sceneScale() * 0.012, 1e-4);
		const geo = new THREE.SphereGeometry(r, 16, 12);
		this.points.forEach((p, i) => {
			const mat = new THREE.MeshBasicMaterial({
				color: i === 0 ? 0x00ff88 : 0x00c8ff, depthTest: false,
			});
			const s = new THREE.Mesh(geo.clone(), mat);
			s.position.copy(p);
			s.renderOrder = 10;
			this.markerRoot.add(s);
		});
		geo.dispose();
		this.render();
	}

	setPoints(list) {
		this.points = (list || [])
			.filter((p) => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite))
			.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
		this._rebuildMarkers();
		// Nothing has framed the view yet (the node has not been run, so there
		// is no model): aim at the points so a pre-filled list is not sitting
		// off-screen behind the "run the workflow" message.
		if (!this._framed && !this.modelRoot.children.length && this.points.length) {
			const box = new THREE.Box3().setFromPoints(this.points);
			if (box.getSize(new THREE.Vector3()).length() < 1e-6)
				box.expandByScalar(Math.max(this.modelSize * 0.5, 0.5));
			this._frame(box, false);
		}
	}

	clear() {
		this.points = [];
		this._rebuildMarkers();
		this._emit();
	}

	// --- model ------------------------------------------------------------
	load(payload) {
		if (!payload?.model) return;
		this.setStatus("loading model...");
		this.modelRoot.clear();
		this.loader.load(tempURL(payload.model), (gltf) => {
			this.modelRoot.add(gltf.scene);
			const box = new THREE.Box3().setFromObject(gltf.scene);
			const size = box.getSize(new THREE.Vector3());
			this.modelSize = Math.max(size.length(), 1e-3);
			this._frame(box);
			this._rebuildMarkers();     // marker radius follows the model size
			this.setStatus("");
			this._resize();
		}, undefined, (err) => {
			console.error("annotate_model_points: model load failed", err);
			this.setStatus("model load failed (GLB expected) - see console");
		});
		this._resize();
	}

	_frame(box, sticky = true) {
		if (sticky) this._framed = true;   // a real model has framed the view
		const c = box.getCenter(new THREE.Vector3());
		const d = Math.max(box.getSize(new THREE.Vector3()).length(), 1e-3);
		this.controls.target.copy(c);
		this.camera.position.copy(c).add(new THREE.Vector3(0.8, 0.6, 1.0)
			.normalize().multiplyScalar(d * 1.4));
		this.camera.near = d / 500;
		this.camera.far = d * 100;
		this.camera.updateProjectionMatrix();
		this.grid.position.set(c.x, box.min.y, c.z);
		this.grid.scale.setScalar(d / 6);
		this.axes.scale.setScalar(d / 6);
		this.controls.update();
		this.render();
	}

	resetCamera() {
		if (!this.modelRoot.children.length) return;
		this._frame(new THREE.Box3().setFromObject(this.modelRoot));
	}

	// --- plumbing ---------------------------------------------------------
	_resize() {
		const w = Math.max(this.viewport.clientWidth | 0, 1);
		const h = Math.max(this.viewport.clientHeight | 0, 1);
		this.renderer.setSize(w, h, false);
		this.renderer.domElement.style.width = `${w}px`;
		this.renderer.domElement.style.height = `${h}px`;
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
		this.render();
	}

	render() { this.renderer.render(this.scene, this.camera); }
}

function buildWidget(node) {
	const pointsWidget = node.widgets?.find((w) => w.name === "points");

	const container = document.createElement("div");
	for (const ev of ["pointerdown", "mousedown", "contextmenu", "touchstart"])
		container.addEventListener(ev, (e) => e.stopPropagation());
	Object.assign(container.style, {
		display: "flex", flexDirection: "column", position: "relative",
		width: "100%", minHeight: "260px",
		background: "#181818", borderRadius: "4px", overflow: "hidden",
	});

	const toolbar = document.createElement("div");
	Object.assign(toolbar.style, {
		display: "flex", gap: "10px", alignItems: "center", padding: "4px 8px",
		fontSize: "11px", color: "#ccc", background: "#242424",
		flex: "0 0 auto", userSelect: "none", zIndex: "2",
	});
	container.appendChild(toolbar);

	const countEl = document.createElement("span");
	countEl.textContent = "0 anchors";
	toolbar.appendChild(countEl);

	const mkButton = (label, cb) => {
		const b = document.createElement("button");
		b.textContent = label;
		Object.assign(b.style, { fontSize: "11px", cursor: "pointer" });
		b.addEventListener("click", (e) => { e.stopPropagation(); cb(); });
		toolbar.appendChild(b);
		return b;
	};

	const hint = document.createElement("span");
	hint.textContent = "click = add · shift+click = remove · drag = orbit";
	Object.assign(hint.style, { marginLeft: "auto", color: "#888" });

	const viewport = document.createElement("div");
	Object.assign(viewport.style, { position: "relative", flex: "1 1 auto" });
	container.appendChild(viewport);

	const canvasHolder = document.createElement("div");
	Object.assign(canvasHolder.style, { position: "absolute", inset: "0" });
	viewport.appendChild(canvasHolder);

	const statusEl = document.createElement("div");
	Object.assign(statusEl.style, {
		position: "absolute", left: "0", right: "0", top: "45%",
		textAlign: "center", color: "#888", fontSize: "12px", pointerEvents: "none",
	});
	statusEl.textContent = "run the workflow to load the model";
	viewport.appendChild(statusEl);

	const viewer = new AnnotateViewer(viewport, canvasHolder, statusEl);
	mkButton("reset view", () => viewer.resetCamera());
	mkButton("clear", () => viewer.clear());
	toolbar.appendChild(hint);

	const refresh = () => { countEl.textContent = `${viewer.points.length} anchors`; };

	// --- the widget is the source of truth --------------------------------
	// Markers are rebuilt FROM the string, never kept in sync with it by hand,
	// so a value typed by the user, restored from a saved workflow, or set by
	// undo all land on screen the same way a click does.
	let lastSeen = null;
	const applyFromWidget = () => {
		const raw = pointsWidget?.value ?? "[]";
		lastSeen = raw;
		const { points, status, parsed } = parseAnchors(raw);
		if (status) viewer.setStatus(status);
		else if ((viewer.statusEl?.textContent || "").startsWith("points:"))
			viewer.setStatus("");
		if (!parsed) return;     // half-typed text: leave the markers alone
		viewer.setPoints(points);
		refresh();
	};

	viewer.onChange = (list) => {
		if (pointsWidget) {
			pointsWidget.value = JSON.stringify(list);
			lastSeen = pointsWidget.value;   // our own write, no need to re-read
			pointsWidget.callback?.(pointsWidget.value);
		}
		refresh();
	};

	viewer._watch = () => {
		if (pointsWidget && pointsWidget.value !== lastSeen) applyFromWidget();
	};
	applyFromWidget();   // and once now, for a node built with a value already set

	// Both flags: options.serialize only keeps the widget out of the API prompt;
	// workflow persistence is the widget-level property, and without it this
	// editing surface saves an extra widgets_values entry (see
	// web/cvlib/widget_values.js). The picks live in the `points` string widget.
	const widget = node.addDOMWidget("annotate_model_surface",
	                                 "opencv_annotate_model", container,
	                                 { serialize: false });
	widget.serialize = false;
	widget.computeLayoutSize = () => ({ minHeight: 260, minWidth: 260 });

	node._cvAnnot3d = viewer;
}

app.registerExtension({
	name: "opencv_comfyui.annotate_model_points",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_CLASS) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			onNodeCreated?.apply(this, arguments);
			buildWidget(this);
			const [w, h] = this.size;
			this.setSize([Math.max(w, 380), Math.max(h, 440)]);

			// Same node-sizing guard as web/cv_preview3d.js - see the long
			// comment there. The properties panel re-fits a selected node via
			// setSize(computeSize()), which would shrink this node below the
			// size the user dragged and lock the 3D surface; but computeSize()
			// is ALSO the minimum height during a real resize drag, so it can
			// only be overridden when the canvas is not actively resizing us.
			const node = this;
			const origComputeSize = node.computeSize.bind(node);
			node.computeSize = function () {
				const r = origComputeSize();
				const resizing = typeof app !== "undefined" && app.canvas
					&& app.canvas.resizing_node === node;
				return resizing ? r : [node.size[0], node.size[1]];
			};
			const origSetSize = node.setSize.bind(node);
			let userSize = [node.size[0], node.size[1]];
			node.setSize = function (s) {
				const resizing = typeof app !== "undefined" && app.canvas
					&& app.canvas.resizing_node === node;
				if (resizing) {
					if (Array.isArray(s)) userSize = [s[0], s[1]];
					return origSetSize(s);
				}
				if (Array.isArray(s)
						&& (s[1] < userSize[1] - 1 || s[0] < userSize[0] - 1))
					return origSetSize([userSize[0], userSize[1]]);
				return origSetSize(s);
			};
			this.onResize = () => this._cvAnnot3d?._resize();
			this._cvAnnot3d?._resize();
		};

		// the viewer runs an animation frame to watch the points widget; without
		// this it keeps running (and holds its WebGL context) after the node is
		// deleted
		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onRemoved = function () {
			this._cvAnnot3d?.dispose();
			this._cvAnnot3d = null;
			return onRemoved?.apply(this, arguments);
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const payload = message?.cv3d_annotate?.[0];
			if (payload && this._cvAnnot3d) this._cvAnnot3d.load(payload);
		};
	},
});
