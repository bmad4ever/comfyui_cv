// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// One shared WebGL1 context for every correspondence editor on the graph.
//
// Each node renders its warped image HERE, into a module-level offscreen
// canvas, then blits the result into its own ordinary 2D canvas with
// ctx.drawImage. Two reasons:
//   - browsers cap live WebGL contexts (~16); a per-node context means the
//     17th editor silently goes black. This caps them at ONE.
//   - the node's visible surface stays a 2D canvas, so markers, arrows, index
//     labels, the opacity cross-fade and the status banner are plain ctx2d
//     calls instead of GL line/text work.
//
// The mesh comes from cvlib/warp_math.js: a regular grid over the DESTINATION
// with uvs pointing back into the source image.

let gl = null;
let canvas = null;
let program = null;
let attribs = null;
let buffers = null;
let dead = false;
let initialized = false;

const textures = new Map(); // key -> { tex, w, h }

const VERT = `
attribute vec2 aPos;
attribute vec2 aUv;
varying vec2 vUv;
void main() {
	vUv = aUv;
	// aPos is [0,1] over the destination; flip y so row 0 is the top row
	gl_Position = vec4(aPos.x * 2.0 - 1.0, 1.0 - aPos.y * 2.0, 0.0, 1.0);
}`;

const FRAG = `
precision mediump float;
uniform sampler2D uTex;
uniform float uBorderAlpha;
varying vec2 vUv;
void main() {
	// outside the source image there is nothing to sample. In the single-image
	// editor that is painted OPAQUE black (BORDER_CONSTANT, not a mirror):
	// a transparent hole would let the un-warped original show through and read
	// as warped pixels that happen not to have moved. In background mode the
	// opposite is the point - the un-warped BACKGROUND is exactly what belongs
	// outside the paste - so uBorderAlpha selects between the two (1 = opaque
	// black, 0 = fully transparent). Either way it carries the blit's
	// globalAlpha like the rest of the warp, so the opacity slider still
	// cross-fades the whole result.
	if (vUv.x < 0.0 || vUv.x > 1.0 || vUv.y < 0.0 || vUv.y > 1.0) {
		gl_FragColor = vec4(0.0, 0.0, 0.0, uBorderAlpha);
		return;
	}
	gl_FragColor = texture2D(uTex, vUv);
}`;

function compile(type, src) {
	const s = gl.createShader(type);
	gl.shaderSource(s, src);
	gl.compileShader(s);
	if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
		console.error("[opencv-comfyui] warp shader:", gl.getShaderInfoLog(s));
		return null;
	}
	return s;
}

function build() {
	const vs = compile(gl.VERTEX_SHADER, VERT);
	const fs = compile(gl.FRAGMENT_SHADER, FRAG);
	if (!vs || !fs) return false;
	program = gl.createProgram();
	gl.attachShader(program, vs);
	gl.attachShader(program, fs);
	gl.linkProgram(program);
	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		console.error("[opencv-comfyui] warp link:", gl.getProgramInfoLog(program));
		return false;
	}
	attribs = {
		pos: gl.getAttribLocation(program, "aPos"),
		uv: gl.getAttribLocation(program, "aUv"),
		tex: gl.getUniformLocation(program, "uTex"),
		borderAlpha: gl.getUniformLocation(program, "uBorderAlpha"),
	};
	buffers = {
		pos: gl.createBuffer(), uv: gl.createBuffer(), idx: gl.createBuffer(),
	};
	return true;
}

function init() {
	if (initialized) return !dead;
	initialized = true;
	try {
		canvas = document.createElement("canvas");
		canvas.width = canvas.height = 16;
		// preserveDrawingBuffer: we drawImage() the canvas after rendering, and
		// without it the buffer is only guaranteed valid inside the same task.
		gl = canvas.getContext("webgl", {
			preserveDrawingBuffer: true, alpha: true, antialias: true,
			premultipliedAlpha: false,
		}) || canvas.getContext("experimental-webgl", {
			preserveDrawingBuffer: true, alpha: true, antialias: true,
		});
		if (!gl) throw new Error("no webgl context");
		canvas.addEventListener("webglcontextlost", (e) => {
			e.preventDefault(); // without this the context is never restored
			dead = true;
			textures.clear();
			program = null;
		});
		canvas.addEventListener("webglcontextrestored", () => {
			// every GL object is gone; rebuild the program and let the callers
			// re-upload their textures lazily
			textures.clear();
			dead = !build();
		});
		dead = !build();
	} catch (err) {
		console.warn("[opencv-comfyui] WebGL unavailable, warp preview disabled:", err);
		dead = true;
	}
	return !dead;
}

export function isAvailable() {
	return init();
}

// Upload (once) the source image for `key`. Returns false when WebGL is dead.
export function uploadTexture(key, img) {
	if (!init()) return false;
	const have = textures.get(key);
	if (have && have.src === img.src) return true;
	if (have) gl.deleteTexture(have.tex);
	const tex = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, tex);
	gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); // the uvs are already y-down
	gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
	// NPOT trap: a non-power-of-two texture renders BLACK unless wrapping is
	// CLAMP_TO_EDGE and the min filter takes no mipmaps.
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
	try {
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
	} catch (err) {
		console.warn("[opencv-comfyui] texture upload failed:", err);
		gl.deleteTexture(tex);
		return false;
	}
	textures.set(key, { tex, src: img.src });
	return true;
}

export function release(key) {
	const have = textures.get(key);
	if (have && gl) gl.deleteTexture(have.tex);
	textures.delete(key);
}

// Render `mesh` (from warp_math.meshFromModel) at w x h device pixels.
// Returns the offscreen canvas to blit, or null when unavailable.
// opaqueBorder: what to paint where the mesh samples outside the source image -
// true = opaque black (single-image editor), false = transparent (background
// mode, where the un-warped background must show through).
export function render(key, mesh, w, h, opaqueBorder = true) {
	if (!init()) return null;
	const entry = textures.get(key);
	if (!entry || !program) return null;
	// grow-only: resizing a drawing buffer every frame is expensive, and the
	// widget's size wobbles by a pixel constantly
	if (canvas.width < w || canvas.height < h) {
		canvas.width = Math.max(canvas.width, w);
		canvas.height = Math.max(canvas.height, h);
	}
	gl.viewport(0, canvas.height - h, w, h); // GL origin is bottom-left
	gl.disable(gl.DEPTH_TEST);
	gl.enable(gl.SCISSOR_TEST);
	gl.scissor(0, canvas.height - h, w, h);
	gl.clearColor(0, 0, 0, 0);
	gl.clear(gl.COLOR_BUFFER_BIT);
	gl.useProgram(program);

	gl.bindBuffer(gl.ARRAY_BUFFER, buffers.pos);
	gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.DYNAMIC_DRAW);
	gl.enableVertexAttribArray(attribs.pos);
	gl.vertexAttribPointer(attribs.pos, 2, gl.FLOAT, false, 0, 0);

	gl.bindBuffer(gl.ARRAY_BUFFER, buffers.uv);
	gl.bufferData(gl.ARRAY_BUFFER, mesh.uvs, gl.DYNAMIC_DRAW);
	gl.enableVertexAttribArray(attribs.uv);
	gl.vertexAttribPointer(attribs.uv, 2, gl.FLOAT, false, 0, 0);

	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.idx);
	gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.DYNAMIC_DRAW);

	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, entry.tex);
	gl.uniform1i(attribs.tex, 0);
	gl.uniform1f(attribs.borderAlpha, opaqueBorder ? 1.0 : 0.0);
	gl.drawElements(gl.TRIANGLES, mesh.indices.length, gl.UNSIGNED_SHORT, 0);
	gl.disable(gl.SCISSOR_TEST);
	return canvas;
}

// where in the offscreen canvas render() put the result (it draws at the TOP,
// because the viewport is placed against the top edge)
export function renderRect(w, h) {
	return { x: 0, y: 0, w, h };
}
