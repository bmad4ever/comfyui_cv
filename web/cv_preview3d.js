// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Calibrated-camera 3D viewer for CV_Preview3DCalibrated.
//
// Why this exists: the core Load3D/Preview3D widgets drive their camera through
// OrbitControls state (position/target/zoom) and, at best, a vertical FOV - a
// calibrated OpenCV camera cannot be represented faithfully that way. This
// widget owns its Three.js camera outright, so it can:
//   * build the projection matrix directly from K (fx, fy, cx, cy): off-axis
//     principal point and fx != fy render exactly, not just a centred fov;
//   * apply the solve-PnP pose INCLUDING roll (OrbitControls forces +Y up);
//   * replicate lens distortion (k1, k2, p1, p2, k3), which is nonlinear and
//     impossible in any 4x4 projection: the scene is rendered to an OVERSCANNED
//     pinhole target, then a full-screen pass warps it per-pixel with the
//     iterative inverse Brown-Conrady model (the cv2.undistortPoints fixed-point
//     iteration, run in the fragment shader);
//   * composite over the calibration photo (bg_image) in exact register - if
//     the camera model is right, the rendered model sits on the photographed
//     board.
//
// Data contract (Python side, convert.py Preview3DCalibrated.execute):
//   message.cv3d[0] = { model: "<temp glb>", bg?: "<temp png>", camera: {...},
//                       depth?: "<temp png>", depth_max?: <float>,
//                       depth_bias?: <float> }
// where camera is the LOAD3D_CAMERA dict from 'CV Camera Pose To 3D View',
// whose underscore extras (_intrinsics 3x3, _dist_coeffs, _image_size [W, H],
// _extrinsics) carry the OpenCV camera model. Without _intrinsics the viewer
// degrades to a plain fov/aspect perspective camera, distortion off.
//
// Three.js r160 is vendored under ./lib (imports rewritten to relative paths);
// see lib/LICENSE.three.txt. Headless tests can only verify the served file +
// API execution - interactive behavior needs the real UI.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import * as THREE from "./lib/three.module.min.js";
import { GLTFLoader } from "./lib/GLTFLoader.js";
import { OrbitControls } from "./lib/OrbitControls.js";
import { TransformControls } from "./lib/TransformControls.js";
import { DECODE_DEPTH_GLSL, LINEARIZE_DEPTH_GLSL } from "./cvlib/depth_codec.js";
import { UNDISTORT_GLSL, UNDISTORT_PIXEL_TOL, displayToRenderNDC }
	from "./cvlib/distort_math.js";

const NODE_CLASS = "CV_Preview3DCalibrated";
const OVERSCAN = 0.25; // rendered margin per side (fraction of the image size)

const DISTORT_FRAG = `
precision highp float;
varying vec2 vUv;
uniform sampler2D tScene;
uniform sampler2D tDepth;      // the render target's own depth buffer
uniform sampler2D tSceneDepth; // packed metric depth of the PHOTOGRAPHED scene
uniform vec4 uK;      // output fx, fy, cx, cy (image px, y down)
uniform vec2 uSize;   // output W, H (image px)
uniform vec4 uKov;    // overscan-render fx, fy, cx', cy' (image px)
uniform vec2 uOvSize; // overscan-render W', H' (image px)
uniform vec4 uD1;     // k1, k2, p1, p2
uniform float uK3;    // k3
uniform float uDistOn;
uniform float uOcclOn;
uniform float uZMax;  // scene depth encoding range
uniform float uZBias;
uniform float uNear;
uniform float uFar;
${DECODE_DEPTH_GLSL}
${LINEARIZE_DEPTH_GLSL}
${UNDISTORT_GLSL}
void main() {
	// this output pixel, in image coordinates (y down), as seen by the REAL
	// (distorted) camera
	vec2 pix = vec2(vUv.x * uSize.x, (1.0 - vUv.y) * uSize.y);
	vec2 xd = (pix - uK.zw) / uK.xy; // distorted normalised camera coords
	// invert Brown-Conrady by fixed-point iteration (cv2.undistortPoints). The
	// iteration lives in cvlib/distort_math.js because the GIZMO POINTER runs
	// the identical one: picking must land on the pixel the cursor is over, not
	// on the pinhole ray Three.js would otherwise raycast.
	vec2 xu = xd;
	if (uDistOn > 0.5) {
		vec3 u = undistortNormalized(xd, uD1, uK3);
		// A calibration whose r_d(r_u) turns over (a strong negative k3, which a
		// chessboard-only fit produces easily) describes NO ray for the outer
		// pixels - 22% of the frame on workflow 57's own calibration. The inverse
		// cannot converge there and runs away, sometimes back INSIDE the overscan
		// render, where it samples real geometry and paints a ghost of the model
		// in the corner. A pixel whose answer does not re-project onto itself is
		// not a pixel of this camera: drop it, exactly as render3d._distort does
		// server-side. Negated so a NaN residual fails the test too.
		if (!(u.z * min(uK.x, uK.y) <= ${UNDISTORT_PIXEL_TOL.toFixed(3)})) {
			gl_FragColor = vec4(0.0);
			return;
		}
		xu = u.xy;
	}
	// sample the ideal pinhole render of the same ray
	vec2 src = uKov.xy * xu + uKov.zw;
	vec2 uv = vec2(src.x / uOvSize.x, 1.0 - src.y / uOvSize.y);
	if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
		gl_FragColor = vec4(0.0);
		return;
	}
	if (uOcclOn > 0.5) {
		// the model's camera-space distance comes from the OVERSCAN render (uv),
		// the photographed scene's from the OUTPUT pixel (vUv) - the depth map
		// belongs to the real, distorted frame, exactly like the background
		// photo does. Same test as render3d._occlude.
		float zModel = linearizeDepth(texture2D(tDepth, uv).x, uNear, uFar);
		float zScene = decodeDepth(texture2D(tSceneDepth, vUv), uZMax);
		if (zModel > zScene + uZBias) {
			gl_FragColor = vec4(0.0);
			return;
		}
	}
	gl_FragColor = texture2D(tScene, uv);
}
`;

const DISTORT_VERT = `
varying vec2 vUv;
void main() {
	vUv = uv;
	gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

function tempURL(name) {
	return api.apiURL(
		`/view?filename=${encodeURIComponent(name)}&type=temp&subfolder=`);
}

function vec3Of(d, fallback) {
	if (!d || !Number.isFinite(d.x + d.y + d.z)) return fallback.clone();
	return new THREE.Vector3(d.x, d.y, d.z);
}

class CalibratedViewer {
	constructor(container, viewport, canvasHolder, statusEl) {
		this.container = container;
		this.viewport = viewport;
		this.statusEl = statusEl;

		this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
		this.renderer.setClearColor(0x000000, 0);
		// every clear in render() is explicit: the helper overlay pass has to
		// blend onto the screen without wiping the occluded model pass, and it
		// has to keep the depth buffer pass 1 left behind
		this.renderer.autoClear = false;
		canvasHolder.appendChild(this.renderer.domElement);
		this.renderer.domElement.style.display = "block";

		this.bgImg = document.createElement("img");
		Object.assign(this.bgImg.style, {
			position: "absolute", left: "0", top: "0",
			width: "100%", height: "100%", display: "none",
		});
		canvasHolder.insertBefore(this.bgImg, this.renderer.domElement);
		this.renderer.domElement.style.position = "relative";

		this.scene = new THREE.Scene();
		this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.5));
		const sun = new THREE.DirectionalLight(0xffffff, 2.0);
		sun.position.set(5, 10, 7.5);
		this.scene.add(sun);
		this.modelRoot = new THREE.Group();
		this.scene.add(this.modelRoot);
		// helpers on the BOARD plane: OpenCV's z=0 object plane maps to the
		// Three.js z=0 plane (the world flip negates y and z), so the grid is
		// rotated out of the default ground orientation into XY
		this.grid = new THREE.GridHelper(12, 12, 0x888888, 0x444444);
		this.grid.rotation.x = Math.PI / 2;
		this.scene.add(this.grid);
		this.axes = new THREE.AxesHelper(2); // object-space origin, like drawFrameAxes
		this.scene.add(this.axes);

		this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
		const applyProjection = () => this._applyProjection();
		this.camera.updateProjectionMatrix = applyProjection;

		this.controls = new OrbitControls(this.camera, this.renderer.domElement);
		this.controls.enableDamping = false;
		this.controls.addEventListener("change", () => this.render());

		// move/rotate/scale gizmo on the model group; its state round-trips
		// through the node's object_transform widget so the server-side render
		// picks the placement up on the next run
		this.onTransformChange = null;
		this.cameraLocked = true;
		this.setCameraLock(this.cameraLocked);
		this.gizmo = new TransformControls(this.camera, this.renderer.domElement);
		this._patchGizmoPointer();
		this.gizmo.addEventListener("dragging-changed", (e) => {
			this.controls.enabled = !e.value && !this.cameraLocked;
		});
		this.gizmo.addEventListener("objectChange", () => {
			this._emitTransform();
			this.render();
		});
		this.gizmo.enabled = false;
		this.gizmo.visible = false;
		this.scene.add(this.gizmo);

		// post pass: full-screen quad warping the overscanned pinhole render.
		// The depth attachment is what lets the same pass cut the model where
		// the photographed scene is nearer - NOTE that MSAA (samples > 0) and a
		// readable depth texture are mutually exclusive in WebGL2, so the
		// target is rebuilt without multisampling as soon as a depth map
		// arrives (_setOcclusionTarget); the shader's own edge quality is
		// unaffected, only the polygon edges get slightly harder.
		this.target = this._makeTarget(4, 4, false);
		this.postScene = new THREE.Scene();
		this.postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
		this.postMaterial = new THREE.ShaderMaterial({
			vertexShader: DISTORT_VERT,
			fragmentShader: DISTORT_FRAG,
			transparent: true,
			uniforms: {
				tScene: { value: this.target.texture },
				tDepth: { value: null },
				tSceneDepth: { value: null },
				uK: { value: new THREE.Vector4(1, 1, 0.5, 0.5) },
				uSize: { value: new THREE.Vector2(1, 1) },
				uKov: { value: new THREE.Vector4(1, 1, 0.5, 0.5) },
				uOvSize: { value: new THREE.Vector2(1, 1) },
				uD1: { value: new THREE.Vector4(0, 0, 0, 0) },
				uK3: { value: 0 },
				uDistOn: { value: 0 },
				uOcclOn: { value: 0 },
				uZMax: { value: 1 },
				uZBias: { value: 0 },
				uNear: { value: 0.01 },
				uFar: { value: 1000 },
			},
		});
		this.postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.postMaterial));

		this.loader = new GLTFLoader();
		this.payload = null;
		this.K = null;          // {fx, fy, cx, cy}
		this.imageSize = null;  // [W, H]
		this.dist = null;       // [k1, k2, p1, p2, k3]
		this.distortionOn = true;
		this.bgOn = true;
		this.occlusionOn = true;
		this.sceneDepthTex = null;
		this.showGrid = true;
		this.cssW = 0;
		this.cssH = 0;

		this._resize = this._resize.bind(this);
		this._toolbarEl = null;
		this._pollLastW = -1;
		this._pollLastH = -1;
		this._pollLastZ = -1;
		this._lastReserved = 0;
		this._resizePending = false;
		// The DOM-widget box can change without a layout-triggered ResizeObserver
		// fire: legacy (Nodes 2.0 OFF) canvas rendering positions/sizes the
		// element via transforms or defers it to a later frame, so a plain
		// ResizeObserver on the element alone misses it. We therefore also poll
		// the true on-screen rect every animation frame and re-fit whenever it
		// changes. Use clientWidth/clientHeight (not getBoundingClientRect) to
		// stay immune to the modern renderer's CSS transform zoom chain on
		// nodes — getBoundingClientRect reports zoom-scaled dimensions, which
		// would make the 3D scene grow/shrink with canvas zoom.
		this._roCallback = () => this._scheduleResize();
		new ResizeObserver(this._roCallback).observe(container);
		this._startPolling();
	}

	_makeTarget(w, h, withDepth) {
		// WebGL2 cannot resolve a multisampled depth buffer into a sampleable
		// texture, so occlusion trades MSAA for a readable depth attachment.
		const opts = { format: THREE.RGBAFormat };
		if (withDepth) {
			opts.depthTexture = new THREE.DepthTexture(w, h);
			opts.depthTexture.type = THREE.UnsignedIntType; // 24-bit, not 16
			opts.depthTexture.minFilter = THREE.NearestFilter;
			opts.depthTexture.magFilter = THREE.NearestFilter;
		} else {
			opts.samples = 4;
		}
		return new THREE.WebGLRenderTarget(Math.max(4, w), Math.max(4, h), opts);
	}

	_setOcclusionTarget(withDepth) {
		if (!!this.target.depthTexture === !!withDepth) return;
		const { width, height } = this.target;
		this.target.dispose();
		this.target = this._makeTarget(width, height, withDepth);
		this.postMaterial.uniforms.tScene.value = this.target.texture;
		this.postMaterial.uniforms.tDepth.value = this.target.depthTexture || null;
	}

	_loadSceneDepth(payload) {
		this.sceneDepthTex?.dispose();
		this.sceneDepthTex = null;
		if (!payload.depth) {
			this._setOcclusionTarget(false);
			this._applyOcclusionUniforms();
			return;
		}
		const tex = new THREE.TextureLoader().load(tempURL(payload.depth), () => {
			this._applyOcclusionUniforms();
			this.render();
		});
		// the packed bytes are DATA, not colour: an sRGB decode or any filtering
		// between texels would mix two 24-bit codes into a third, unrelated
		// depth (a mid-grey between "3 m" and "40 m" is not 20 m)
		tex.colorSpace = THREE.NoColorSpace;
		tex.minFilter = THREE.NearestFilter;
		tex.magFilter = THREE.NearestFilter;
		tex.generateMipmaps = false;
		this.sceneDepthTex = tex;
		this._setOcclusionTarget(true);
		this._applyOcclusionUniforms();
	}

	_applyOcclusionUniforms() {
		const u = this.postMaterial.uniforms;
		const on = !!(this.sceneDepthTex && this.occlusionOn && this.target.depthTexture);
		u.tSceneDepth.value = this.sceneDepthTex;
		u.tDepth.value = this.target.depthTexture || null;
		u.uOcclOn.value = on ? 1 : 0;
		u.uZMax.value = this.payload?.depth_max ?? 1;
		u.uZBias.value = this.payload?.depth_bias ?? 0;
		u.uNear.value = this.camera.near;
		u.uFar.value = this.camera.far;
	}

	setOcclusion(on) {
		this.occlusionOn = on;
		this._applyOcclusionUniforms();
		this.render();
	}

	_scheduleResize() {
		if (this._resizePending) return;
		this._resizePending = true;
		requestAnimationFrame(() => {
			this._resizePending = false;
			this._resize();
		});
	}

	_startPolling() {
		if (this._polling) return;
		this._polling = true;
		const tick = () => {
			if (!this._polling) return;
			if (!this.container.isConnected) { this._polling = false; return; }
			// watch the node geometry + canvas zoom (authoritative), not the
			// measured rect, so horizontal resizes / zoom always re-fit even when
			// the legacy renderer leaves the DOM element's box stale
			const node = this._node;
			const resizing = typeof app !== "undefined" && app.canvas
				&& app.canvas.resizing_node === node;
			// The properties panel (focusing the node -> INPUTS shown on the
			// right) re-fits the node, assigning a fixed size directly to
			// node.size (not via setSize), which collapses the 3D surface. When
			// the node is NOT being actively resized we continuously restore the
			// user's size, so any such re-fit is undone and the surface keeps
			// tracking. During an active resize drag we adopt the new size.
			if (node && this._userSize) {
				if (resizing) {
					this._userSize[0] = node.size[0];
					this._userSize[1] = node.size[1];
				} else if (node.size[0] !== this._userSize[0]
						|| node.size[1] !== this._userSize[1]) {
					node.size[0] = this._userSize[0];
					node.size[1] = this._userSize[1];
				}
			}
			const wKey = node ? Math.round(node.size[0]) : -1;
			const hKey = node ? Math.round(node.size[1]) : -1;
			const ds = (typeof app !== "undefined" && app.canvas && app.canvas.ds);
			const zKey = ds ? Math.round((ds.scale || 1) * 1000) : -1;
			if (wKey !== this._pollLastW || hKey !== this._pollLastH || zKey !== this._pollLastZ) {
				this._pollLastW = wKey;
				this._pollLastH = hKey;
				this._pollLastZ = zKey;
				this._resize();
			}
			requestAnimationFrame(tick);
		};
		requestAnimationFrame(tick);
	}

	get overscan() {
		return this.dist && this.distortionOn ? OVERSCAN : 0;
	}

	setStatus(text) {
		this.statusEl.textContent = text;
		this.statusEl.style.display = text ? "block" : "none";
	}

	load(payload) {
		this.payload = payload;
		const cam = payload.camera || {};
		const Km = cam._intrinsics;
		this.K = Km ? { fx: Km[0][0], fy: Km[1][1], cx: Km[0][2], cy: Km[1][2] } : null;
		this.imageSize = cam._image_size || null;
		const d = cam._dist_coeffs;
		this.dist = d && d.some((v) => v !== 0)
			? [d[0] || 0, d[1] || 0, d[2] || 0, d[3] || 0, d[4] || 0]
			: null;

		if (payload.bg) {
			this.bgImg.src = tempURL(payload.bg);
		} else {
			this.bgImg.removeAttribute("src");
		}
		this._updateBgVisibility();
		this._loadSceneDepth(payload);

		// the transform the server rendered with (from the object_transform
		// widget or its upstream link) is the viewer's starting placement
		this.applyTransformSpec(payload.transform);

		this.resetCamera();
		this.setStatus("loading model...");
		this.modelRoot.clear();
		this.loader.load(
			tempURL(payload.model),
			(gltf) => {
				// no recentring/rescaling: the pose is relative to the solve-PnP
				// object-space origin, so the model must keep its native origin
				this.modelRoot.add(gltf.scene);
				this.setStatus("");
				this._resize();
			},
			undefined,
			(err) => {
				console.error("cv_preview3d: model load failed", err);
				this.setStatus("model load failed (GLB expected) - see console");
			});
		this._resize();
	}

	// {position, quaternion, scale} in Three.js world space, the shape the
	// object_transform widget carries (missing keys = identity, scale may be one
	// number). Shared by load() and the reset button so a hand-edited widget and
	// a rendered payload are applied by the SAME code.
	applyTransformSpec(spec) {
		const tr = spec || {};
		const p = tr.position || [0, 0, 0];
		const q = tr.quaternion || [0, 0, 0, 1];
		let s = tr.scale ?? [1, 1, 1];
		if (typeof s === "number") s = [s, s, s];
		this.modelRoot.position.set(p[0], p[1], p[2]);
		this.modelRoot.quaternion.set(q[0], q[1], q[2], q[3]);
		this.modelRoot.scale.set(s[0], s[1], s[2]);
	}

	// The object_transform widget as it stands RIGHT NOW, or null when upstream
	// owns it (converted to a linked input): the string only exists server-side
	// then, so the last rendered payload is the best the viewer can know.
	_transformWidgetValue() {
		const node = this._node;
		if (!node) return null;
		const linked = node.inputs?.some(
			(i) => i.name === "object_transform" && i.link != null);
		if (linked) return null;
		const w = node.widgets?.find((w) => w.name === "object_transform");
		return w ? String(w.value ?? "") : null;
	}

	// Puts the model back where object_transform says it is - NOT at the origin.
	// A gizmo drag overwrites that widget, so this undoes a mis-drag only if the
	// widget was fixed first (or was never dragged); that is deliberate, the
	// widget is the single source of truth the next render will use.
	resetObject() {
		const raw = this._transformWidgetValue();
		let spec = this.payload?.transform || {};
		if (raw !== null) {
			const text = raw.trim();
			if (!text) {
				spec = {};       // empty widget = identity, exactly as execute() reads it
			} else {
				try {
					const parsed = JSON.parse(text);
					if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
						throw new TypeError("not a JSON object");
					spec = parsed;
				} catch (e) {
					// the next run would fail on this string too (execute raises), so
					// say so here rather than silently snapping to some other pose
					this._flashStatus("object_transform is not valid JSON - kept the "
					                  + "placement of the last render");
					console.warn("cv_preview3d: object_transform is not valid JSON", e);
				}
			}
		}
		this.applyTransformSpec(spec);
		this.render();
	}

	_flashStatus(text, ms = 3000) {
		this.setStatus(text);
		setTimeout(() => {
			if (this.statusEl.textContent === text) this.setStatus("");
		}, ms);
	}

	resetCamera() {
		const cam = this.payload?.camera || {};
		const pos = vec3Of(cam.position, new THREE.Vector3(0, 0, 10));
		const tgt = vec3Of(cam.target, new THREE.Vector3(0, 0, 0));
		this.camera.position.copy(pos);
		const q = cam.quaternion;
		if (q && Number.isFinite(q.x + q.y + q.z + q.w)) {
			// exact orientation incl. roll - deliberately NOT controls.update(),
			// which would re-derive a roll-free lookAt; controls take over (and
			// drop roll) only when the user actually drags
			this.camera.quaternion.set(q.x, q.y, q.z, q.w);
		} else {
			this.camera.lookAt(tgt);
		}
		this.controls.target.copy(tgt);
		this.camera.near = Math.max((cam.near ?? 0.01), 1e-4);
		this.camera.far = Math.max((cam.far ?? 1000), this.camera.near * 10);
		if (!this.K) {
			this.camera.fov = cam.fov || 50;
		}
		// occlusion reads this depth buffer, so its precision matters: a 24-bit
		// attachment (UnsignedIntType, forced in _makeTarget) at near=0.01 still
		// resolves ~5 mm at 30 m, while the 16-bit default would be over a
		// metre there and the cut would visibly crawl
		this._applyOcclusionUniforms();
		this._applyProjection();
		this.render();
	}

	_applyProjection() {
		const near = this.camera.near, far = this.camera.far;
		if (this.K && this.imageSize) {
			const [W, H] = this.imageSize;
			const m = this.overscan;
			// overscanned pinhole: same focal lengths, principal point shifted by
			// the margin - the shader samples this with the SAME image-unit maths
			const Wp = W * (1 + 2 * m), Hp = H * (1 + 2 * m);
			const cxp = this.K.cx + W * m, cyp = this.K.cy + H * m;
			// OpenCV -> WebGL clip: x_img = fx X/Z + cx (y down) with the Three.js
			// camera looking down -Z, y up
			this.camera.projectionMatrix.set(
				2 * this.K.fx / Wp, 0, 1 - 2 * cxp / Wp, 0,
				0, 2 * this.K.fy / Hp, 2 * cyp / Hp - 1, 0,
				0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
				0, 0, -1, 0);
			this.camera.projectionMatrixInverse
				.copy(this.camera.projectionMatrix).invert();
			this.postMaterial.uniforms.uK.value.set(this.K.fx, this.K.fy, this.K.cx, this.K.cy);
			this.postMaterial.uniforms.uSize.value.set(W, H);
			this.postMaterial.uniforms.uKov.value.set(this.K.fx, this.K.fy, cxp, cyp);
			this.postMaterial.uniforms.uOvSize.value.set(Wp, Hp);
		} else {
			THREE.PerspectiveCamera.prototype.updateProjectionMatrix.call(this.camera);
		}
		const d = this.dist || [0, 0, 0, 0, 0];
		this.postMaterial.uniforms.uD1.value.set(d[0], d[1], d[2], d[3]);
		this.postMaterial.uniforms.uK3.value = d[4];
		this.postMaterial.uniforms.uDistOn.value = this.dist && this.distortionOn ? 1 : 0;
	}

	_aspect() {
		return this.imageSize ? this.imageSize[0] / this.imageSize[1]
			: (this.payload?.camera?.aspect || 4 / 3);
	}

	_resize() {
		const legacy = !!(window.LiteGraph && window.LiteGraph.vueNodesMode === false);
		// A widget interaction (e.g. toggling a checkbox) makes the legacy
		// renderer re-sync and pin the DOM element's width to a fixed pixel
		// value, so CSS 100% no longer tracks the node and horizontal resizes
		// stop propagating. Re-assert the width every resize so it keeps filling
		// the node's widget slot.
		this.container.style.width = "100%";
		const toolbarH = this._toolbarEl ? this._toolbarEl.offsetHeight : 0;
		let availW = this.container.clientWidth;
		let availH;
		if (legacy) {
			// Legacy mode: the DOM widget container has CSS
			// transform: scale(zoom) applied by the renderer, which makes
			// getBoundingClientRect() report zoom-scaled dimensions. Use
			// clientWidth/Height instead so the 3D scene stays fixed at the
			// node's layout size; the CSS transform handles visual scaling.
			// The container height is set explicitly below from the node's
			// widget geometry (graph-space), so derive the available body
			// height from clientHeight after setting it.
			const node = this._node;
			// Fill the node body below this widget. widget.y is the top of our
			// slot in graph units, so node.size[1] - y is the available body
			// height. The height is therefore INDEPENDENT of the node width.
			// A widget interaction (checkbox toggle) or a resize drag can
			// transiently invalidate widget.y (0 / undefined); reuse the last
			// good body height then so the surface never jumps to the full node
			// or to a width-derived height.
			let reservedGraph = this._lastReserved;
			if (!(reservedGraph > 0))
				reservedGraph = node ? Math.round(node.size[1] * 0.8) : 300;
			if (node && this._widget) {
				const y = this._widget.y;
				const ok = typeof y === "number" && isFinite(y)
					&& y > 0 && y < node.size[1];
				if (ok) {
					const body = node.size[1] - y;
					if (body > 0) {
						reservedGraph = Math.round(body);
						this._lastReserved = reservedGraph;
					}
				}
			}
			if (this.container.style.height !== `${reservedGraph}px`)
				this.container.style.height = `${reservedGraph}px`;
			availH = this.container.clientHeight - toolbarH;
		} else {
			// Modern (Nodes 2.0) lays the box out in normal flow - trust
			// clientWidth/Height which are immune to CSS transform zoom.
			availH = this.container.clientHeight - toolbarH;
		}
		if (!availW || !availH) return;
		this.viewport.style.top = `${toolbarH}px`;
		this.viewport.style.bottom = "0px";
		// letterbox to the calibration image's aspect so the bg photo overlays
		// in register
		const a = this._aspect();
		let w = availW, h = Math.round(availW / a);
		if (h > availH) { h = availH; w = Math.round(availH * a); }
		this.cssW = w; this.cssH = h;
		const holder = this.renderer.domElement.parentElement;
		Object.assign(holder.style, {
			width: `${w}px`, height: `${h}px`,
			left: `${Math.floor((availW - w) / 2)}px`,
			top: `${Math.floor((availH - h) / 2)}px`,
		});
		const dpr = window.devicePixelRatio || 1;
		this.renderer.setPixelRatio(dpr);
		this.renderer.setSize(w, h);
		const m = this.overscan;
		this.target.setSize(
			Math.max(4, Math.round(w * dpr * (1 + 2 * m))),
			Math.max(4, Math.round(h * dpr * (1 + 2 * m))));
		if (!this.K) this.camera.aspect = w / h;
		this._applyProjection();
		this.render();
	}

	setDistortion(on) {
		this.distortionOn = on;
		this._resize(); // overscan margin changes the target size + projection
	}

	setBackground(on) {
		this.bgOn = on;
		this._updateBgVisibility();
	}

	_updateBgVisibility() {
		const show = (this.bgOn ?? true) && this.payload?.bg;
		this.bgImg.style.display = show ? "block" : "none";
	}

	setGrid(on) {
		this.showGrid = on;
		this.render();
	}

	setCameraLock(on) {
		// freeze OrbitControls so gizmo edits can't nudge the calibrated view;
		// the gizmo and the reset button keep working
		this.cameraLocked = on;
		this.controls.enabled = !on;
	}

	// Pointer -> the NDC of the OVERSCANNED PINHOLE render, which is the space
	// the camera's projection matrix (and therefore THREE.Raycaster) lives in.
	// Returns null when there is nothing to correct (no calibration, distortion
	// off) so the caller can pass the raw pointer straight through.
	_pointerToRenderNDC(x, y) {
		if (!this.K || !this.imageSize) return null;
		const dist = this.distortionOn ? this.dist : null;
		if (!dist) return null; // overscan is 0 too - the map is the identity
		const [W, H] = this.imageSize;
		return displayToRenderNDC(x, y, {
			K: this.K, width: W, height: H, margin: this.overscan, dist,
		});
	}

	_patchGizmoPointer() {
		// THE GIZMO IS DRAWN THROUGH THE DISTORTION AND PICKED WITHOUT IT.
		// TransformControls renders as ordinary scene content, so the post pass
		// warps its handles like everything else - but its hit test raycasts the
		// raw canvas NDC through the ideal pinhole frustum, which is a DIFFERENT
		// ray. The handle you see and the handle you grab drift apart with
		// radius, which is exactly where a lens bends most.
		//
		// _getPointer is an instance property (TransformControls.js:169), so it
		// can be wrapped here instead of patching the vendored file. Everything
		// downstream - hover highlight, axis pick, plane drag - goes through it,
		// so one wrap corrects all three, and the drag stays consistent because
		// the drag plane is intersected with the corrected ray as well.
		const raw = this.gizmo._getPointer;
		this.gizmo._getPointer = (event) => {
			const p = raw(event);
			// pointer lock reports movement deltas, not a position (that branch
			// returns 0,0) - there is nothing to unwarp
			if (this.renderer.domElement.ownerDocument.pointerLockElement) return p;
			const c = this._pointerToRenderNDC(p.x, p.y);
			if (!c) return p;
			// `inside` false means no ray of this camera reaches that pixel - it
			// left the render, or the lens model cannot be inverted there and the
			// shader leaves it empty (see cvlib/distort_math.js). Pick NOTHING
			// rather than pick along a ray that produced no pixel: (2, 2) is
			// outside every frustum, so the raycast simply misses.
			if (!c.inside) return { x: 2, y: 2, button: p.button };
			return { x: c.x, y: c.y, button: p.button };
		};
	}

	setGizmoMode(mode) {
		// "off" | "translate" | "rotate" | "scale"
		if (mode === "off") {
			this.gizmo.detach();
			this.gizmo.enabled = false;
			this.gizmo.visible = false;
		} else {
			this.gizmo.attach(this.modelRoot);
			this.gizmo.setMode(mode);
			this.gizmo.enabled = true;
			this.gizmo.visible = true;
		}
		this.render();
	}

	_emitTransform() {
		if (!this.onTransformChange) return;
		const r = (v) => Math.round(v * 1e6) / 1e6;
		const p = this.modelRoot.position, q = this.modelRoot.quaternion,
			s = this.modelRoot.scale;
		this.onTransformChange({
			position: [r(p.x), r(p.y), r(p.z)],
			quaternion: [r(q.x), r(q.y), r(q.z), r(q.w)],
			scale: [r(s.x), r(s.y), r(s.z)],
		});
	}

	render() {
		if (!this.cssW) return;
		this.grid.visible = this.showGrid;
		this.axes.visible = this.showGrid;
		const u = this.postMaterial.uniforms;
		this.renderer.setRenderTarget(this.target);
		this.renderer.clear();
		this.renderer.render(this.scene, this.camera);
		this.renderer.setRenderTarget(null);
		this.renderer.clear();
		this.renderer.render(this.postScene, this.postCamera);
		if (u.uOcclOn.value <= 0.0) return;

		// The grid, the object axes and the transform gizmo are UI, not scene
		// content: the photographed depth must never cut them, or you cannot
		// see the handle you are dragging the model behind. The gizmo cannot
		// survive the cut by accident either - TransformControls' materials are
		// depthTest:false / depthWrite:false, so its pixels carry the CLEARED
		// depth (= infinitely far) and the occlusion test erases them wherever
		// the photo has a measurement, i.e. almost everywhere.
		//
		// So they get a second pass with uOcclOn off, blended over the first
		// (autoClear is off; the shader writes alpha 0 outside the geometry).
		// The DEPTH buffer is deliberately NOT cleared: pass 1 left the model's
		// depth in it, so the grid is still hidden where the model is genuinely
		// in front of it - only the scene-depth cut is skipped, not the model's
		// own occlusion. Cost is one extra pass over three helper objects.
		this.modelRoot.visible = false;
		this.renderer.setRenderTarget(this.target);
		this.renderer.clear(true, false, false); // colour only - keep pass 1 depth
		this.renderer.render(this.scene, this.camera);
		this.renderer.setRenderTarget(null);
		u.uOcclOn.value = 0;
		this.renderer.render(this.postScene, this.postCamera);
		u.uOcclOn.value = 1;
		this.modelRoot.visible = true;
	}
}

function buildWidget(node) {
	// Toolbar (gizmo dropdown / checkboxes / reset) is always shown.
	// Was temporarily disabled (SHOW_TOOLBAR=false) to confirm it is NOT the
	// cause of the selection + properties-panel resize bug.
	const SHOW_TOOLBAR = true;
	const container = document.createElement("div");
	// Stop pointer/mouse events from bubbling to the canvas. In legacy mode
	// (Nodes 2.0 OFF) the canvas's processMouseDown listens on the canvas
	// element; our DOM widget lives inside it, so a checkbox/toolbar click
	// reaches that handler and starts a node-resize drag. The resize handler
	// then clamps the node height to computeSize()[1] (= width*0.8), shrinking
	// the node below the size the user dragged and "locking" the 3D surface.
	// Bubble-phase stop lets the checkbox + OrbitControls receive the event
	// first (so they still work) while keeping it away from the canvas.
	for (const ev of ["pointerdown", "mousedown", "contextmenu", "touchstart"])
		container.addEventListener(ev, (e) => e.stopPropagation());
	Object.assign(container.style, {
		display: "flex", flexDirection: "column",
		position: "relative",
		width: "100%", minHeight: "260px",
		background: "#181818", borderRadius: "4px", overflow: "hidden",
	});

	const toolbar = document.createElement("div");
	Object.assign(toolbar.style, {
		display: "flex", gap: "10px", alignItems: "center",
		padding: "4px 8px", fontSize: "11px", color: "#ccc",
		background: "#242424", flex: "0 0 auto", userSelect: "none",
	});
	if (SHOW_TOOLBAR) container.appendChild(toolbar);

	const viewport = document.createElement("div");
	Object.assign(viewport.style, {
		position: "absolute", left: "0", top: "0", right: "0", bottom: "0",
	});
	container.appendChild(viewport);

	const canvasHolder = document.createElement("div");
	Object.assign(canvasHolder.style, { position: "absolute" });
	viewport.appendChild(canvasHolder);

	const statusEl = document.createElement("div");
	Object.assign(statusEl.style, {
		position: "absolute", left: "0", right: "0", top: "45%",
		textAlign: "center", color: "#888", fontSize: "12px",
		pointerEvents: "none",
	});
	statusEl.textContent = "run the workflow to render";
	viewport.appendChild(statusEl);

	const viewer = new CalibratedViewer(container, viewport, canvasHolder, statusEl);
	viewer._toolbarEl = SHOW_TOOLBAR ? toolbar : null;

	const gizmoSel = document.createElement("select");
	Object.assign(gizmoSel.style, { fontSize: "11px" });
	for (const [value, label] of [["off", "gizmo: off"], ["translate", "gizmo: move"],
	                              ["rotate", "gizmo: rotate"], ["scale", "gizmo: scale"]]) {
		const opt = document.createElement("option");
		opt.value = value;
		opt.textContent = label;
		gizmoSel.appendChild(opt);
	}
	gizmoSel.addEventListener("change", (e) => { e.stopPropagation(); viewer.setGizmoMode(gizmoSel.value); });
	toolbar.appendChild(gizmoSel);

	const toggle = (label, initial, cb) => {
		const wrap = document.createElement("label");
		Object.assign(wrap.style, { display: "flex", gap: "3px", alignItems: "center", cursor: "pointer" });
		const box = document.createElement("input");
		box.type = "checkbox";
		box.checked = initial;
		// Stop the event reaching the Vue node component: a checkbox toggle
		// bubbling up made the node re-render/re-size and pinned the 3D surface
		// to cover the header.
		box.addEventListener("change", (e) => { e.stopPropagation(); cb(box.checked); });
		wrap.appendChild(box);
		wrap.appendChild(document.createTextNode(label));
		toolbar.appendChild(wrap);
		return box;
	};
	toggle("distortion", true, (on) => viewer.setDistortion(on));
	toggle("background", true, (on) => viewer.setBackground(on));
	// no-op until the node's scene_depth input is wired
	toggle("occlusion", true, (on) => viewer.setOcclusion(on));
	toggle("grid", true, (on) => viewer.setGrid(on));
	toggle("lock camera", true, (on) => viewer.setCameraLock(on));

	const resetBtn = document.createElement("button");
	resetBtn.textContent = "reset";
	resetBtn.title = "Put the view back on the calibrated camera AND the model "
		+ "back where object_transform says it is - the placement the next run "
		+ "will render. Orbiting and dragging the gizmo are both undone.";
	Object.assign(resetBtn.style, { fontSize: "11px", padding: "1px 8px" });
	resetBtn.addEventListener("click", (e) => {
		e.stopPropagation();
		viewer.resetObject();   // first: resetCamera() renders, and the frame it
		viewer.resetCamera();   // draws should already show the restored placement
	});
	toolbar.appendChild(resetBtn);

	// gizmo edits land in the object_transform widget, so the placement
	// serializes with the workflow and the next run renders it server-side;
	// while the widget is converted to a linked input, upstream owns the value
	viewer.onTransformChange = (spec) => {
		const linked = node.inputs?.some(
			(i) => i.name === "object_transform" && i.link != null);
		if (linked) return;
		const w = node.widgets?.find((w) => w.name === "object_transform");
		if (w) {
			w.value = JSON.stringify(spec);
			node.graph?.setDirtyCanvas(true, false);
		}
	};

	const widget = node.addDOMWidget("cv3d_viewer", "cv3d", container, {
		serialize: false,
		hideOnZoom: false,
	});
	// BOTH flags: `options.serialize` only keeps the widget out of the API
	// prompt - workflow persistence is the widget-level property, and without it
	// the viewport saved its empty value as one extra widgets_values entry (see
	// web/cvlib/widget_values.js). The placement lives in `object_transform`.
	widget.serialize = false;
	// Nodes 2.0 (modern renderer) sizes DOM widgets from computeLayoutSize;
	// legacy LiteGraph falls back to computeSize below. Both must exist so the
	// surface gets a real box in either renderer.
	widget.computeLayoutSize = () => ({
		minHeight: 260, maxHeight: 1e6, minWidth: 20, maxWidth: 1e6,
	});
	widget.computeSize = (width) => [width, Math.max(260, Math.round(width * 0.8))];
	viewer._widget = widget;
	viewer._node = node;
	node._cvViewer = viewer;
}

app.registerExtension({
	name: "opencv.Preview3DCalibrated",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_CLASS) return;
		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			onNodeCreated?.apply(this, arguments);
			buildWidget(this);
			const [w, h] = this.size;
			this.setSize([Math.max(w, 380), Math.max(h, 460)]);
			// The "Workflow Overview" properties panel re-fits the selected node
			// via setSize(node.computeSize()), which (with our natural height of
			// width*0.8) shrinks the node below the size the user dragged and
			// locks the 3D surface. We can't just return the current size from
			// computeSize() unconditionally: the canvas resize clamp uses
			// computeSize() as the node's MINIMUM height during a drag, so that
			// would also block manual shrinking. The distinguishing signal is the
			// canvas's resizing_node - it equals THIS node ONLY while the user is
			// actively dragging a resize handle. So report the natural size while
			// resizing (clamp still allows shrinking) and the current size
			// otherwise (the selection re-fit then becomes a no-op).
			const node = this;
			const origComputeSize = node.computeSize.bind(node);
			node.computeSize = function () {
				const r = origComputeSize();
				const resizing = typeof app !== "undefined" && app.canvas
					&& app.canvas.resizing_node === node;
				return resizing ? r : [node.size[0], node.size[1]];
			};
			// The properties-panel re-fit may also set an EXPLICIT size that
			// bypasses computeSize() entirely, so guard setSize too. While an
			// active resize drag is in progress (canvas.resizing_node === node)
			// we obey every size change (and remember it as the user's size);
			// any shrink that arrives when we are NOT actively resizing is the
			// panel's re-fit - drop it and keep the user's size. Gating on
			// resizing_node (not node.selected) is essential: selecting a node
			// also happens at the start of a manual resize drag, so gating on
			// selected would have blocked manual shrinking too.
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
			// legacy LiteGraph doesn't drive DOM-widget relayout on node resize,
			// so re-fit the 3D surface whenever the node box changes
			this.onResize = () => this._cvViewer?._resize();
			this._cvViewer._userSize = [this.size[0], this.size[1]];
			this._cvViewer?._resize();
		};
		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const payload = message?.cv3d?.[0];
			if (payload && this._cvViewer) this._cvViewer.load(payload);
		};
	},
});
