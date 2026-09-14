// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Warp models for the correspondence editor - PURE MATH, no DOM, no ComfyUI
// imports. That is deliberate: tests/smoke_test.py::test_warp_math_js_matches_cv2
// runs THIS FILE under node.exe and compares it against cv2, so the warp the user
// sees while dragging is provably the warp the server will compute.
//
// This directory (web/cvlib/) is not auto-loaded by ComfyUI - only top-level
// web/*.js are registered as extensions - so a module here is inert until
// something imports it.
//
// The thin plate spline reproduces cv2.createThinPlateSplineShapeTransformer
// exactly:
//     U(d2) = d2 * log(d2 + FLT_EPSILON)      (OpenCV's own kernel, px^2 units)
//     K[i][i] = regularization                 (NOT zero - this is where the
//                                               regularization parameter lands)
//     L = [[K, P], [P^T, 0]]  with P = [1, x, y]
//     solve L * W = [dst; 0]
// Measured against this build's cv2 over 50 query points at regularization
// 0 / 0.5 / 5 / 20: max abs error 8e-4 px. NOTE the px^2 units - the regularizer
// is image-SCALE dependent, which is why cv2's own knob does nothing below ~1e3
// on a few-hundred-pixel image.

const FLT_EPSILON = 1.1920928955078125e-7;

// Gauss-Jordan with partial pivoting. A is n x n (row arrays, modified in
// place), B is n x m. Returns the solution or null when the system is singular -
// collinear-only or duplicated control points. Callers MUST handle null: a NaN
// mesh renders as nothing at all, which is worse than no preview.
export function solveInPlace(A, B) {
	const n = A.length;
	const m = B[0].length;
	for (let col = 0; col < n; col++) {
		let piv = col;
		for (let r = col + 1; r < n; r++) {
			if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
		}
		if (!(Math.abs(A[piv][col]) > 1e-9)) return null;
		if (piv !== col) {
			const ta = A[piv]; A[piv] = A[col]; A[col] = ta;
			const tb = B[piv]; B[piv] = B[col]; B[col] = tb;
		}
		const d = A[col][col];
		for (let c = col; c < n; c++) A[col][c] /= d;
		for (let c = 0; c < m; c++) B[col][c] /= d;
		for (let r = 0; r < n; r++) {
			if (r === col) continue;
			const f = A[r][col];
			if (f === 0) continue;
			for (let c = col; c < n; c++) A[r][c] -= f * A[col][c];
			for (let c = 0; c < m; c++) B[r][c] -= f * B[col][c];
		}
	}
	return B;
}

const kernel = (d2) => d2 * Math.log(d2 + FLT_EPSILON);

// from/to: arrays of [x, y] in IMAGE PIXELS. Returns the (n + 3) x 2 weight
// matrix, or null when degenerate.
export function fitTPS(from, to, reg) {
	const n = from.length;
	if (n < 3) return null;
	const A = [];
	for (let i = 0; i < n + 3; i++) A.push(new Array(n + 3).fill(0));
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) {
			if (i === j) { A[i][j] = reg; continue; }
			const dx = from[i][0] - from[j][0];
			const dy = from[i][1] - from[j][1];
			A[i][j] = kernel(dx * dx + dy * dy);
		}
		A[i][n] = 1; A[i][n + 1] = from[i][0]; A[i][n + 2] = from[i][1];
		A[n][i] = 1; A[n + 1][i] = from[i][0]; A[n + 2][i] = from[i][1];
	}
	const B = [];
	for (let i = 0; i < n; i++) B.push([to[i][0], to[i][1]]);
	for (let i = 0; i < 3; i++) B.push([0, 0]);
	return solveInPlace(A, B);
}

export function evalTPS(W, ctl, x, y) {
	const n = ctl.length;
	let ox = W[n][0] + W[n + 1][0] * x + W[n + 2][0] * y;
	let oy = W[n][1] + W[n + 1][1] * x + W[n + 2][1] * y;
	for (let i = 0; i < n; i++) {
		const dx = x - ctl[i][0];
		const dy = y - ctl[i][1];
		const u = kernel(dx * dx + dy * dy);
		ox += u * W[i][0];
		oy += u * W[i][1];
	}
	return [ox, oy];
}

// --- parametric models -------------------------------------------------------
// All return a 3x3 row-major matrix (affine families keep [0, 0, 1] as the last
// row) so one inverse and one evaluator serve every one of them.

function normalizer(pts) {
	let cx = 0, cy = 0;
	for (const p of pts) { cx += p[0]; cy += p[1]; }
	cx /= pts.length; cy /= pts.length;
	let d = 0;
	for (const p of pts) d += Math.hypot(p[0] - cx, p[1] - cy);
	d /= pts.length;
	const s = d > 1e-9 ? Math.SQRT2 / d : 1;
	return { T: [[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]], s, cx, cy };
}

export function matInverse(M) {
	const [a, b, c] = M[0], [d, e, f] = M[1], [g, h, i] = M[2];
	const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
	if (Math.abs(det) < 1e-12) return null;
	return [
		[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
		[(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
		[(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
	];
}

const matMul = (A, B) => A.map((row, r) =>
	[0, 1, 2].map((c) => row[0] * B[0][c] + row[1] * B[1][c] + row[2] * B[2][c]));

export function evalMatrix(M, x, y) {
	const w = M[2][0] * x + M[2][1] * y + M[2][2];
	const iw = Math.abs(w) > 1e-12 ? 1 / w : 0;
	return [(M[0][0] * x + M[0][1] * y + M[0][2]) * iw,
	        (M[1][0] * x + M[1][1] * y + M[1][2]) * iw];
}

// Normalized DLT: conditioning matters here, an un-normalized 8x8 on pixel
// coordinates loses several digits.
export function fitHomography(from, to) {
	if (from.length < 4) return null;
	const nf = normalizer(from), nt = normalizer(to);
	const A = [], B = [];
	for (let i = 0; i < from.length; i++) {
		const [x, y] = evalMatrix(nf.T, from[i][0], from[i][1]);
		const [u, v] = evalMatrix(nt.T, to[i][0], to[i][1]);
		A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.push([u]);
		A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); B.push([v]);
	}
	// least squares via the normal equations (8x8) - plenty for <= 64 points
	const N = [], R = [];
	for (let r = 0; r < 8; r++) {
		N.push(new Array(8).fill(0));
		R.push([0]);
	}
	for (let k = 0; k < A.length; k++) {
		for (let r = 0; r < 8; r++) {
			R[r][0] += A[k][r] * B[k][0];
			for (let c = 0; c < 8; c++) N[r][c] += A[k][r] * A[k][c];
		}
	}
	const h = solveInPlace(N, R);
	if (!h) return null;
	const Hn = [[h[0][0], h[1][0], h[2][0]],
	            [h[3][0], h[4][0], h[5][0]],
	            [h[6][0], h[7][0], 1]];
	const inv = matInverse(nt.T);
	return inv ? matMul(inv, matMul(Hn, nf.T)) : null;
}

export function fitAffine(from, to) {
	if (from.length < 3) return null;
	const N = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
	const R = [[0, 0], [0, 0], [0, 0]];
	for (let i = 0; i < from.length; i++) {
		const row = [from[i][0], from[i][1], 1];
		for (let r = 0; r < 3; r++) {
			R[r][0] += row[r] * to[i][0];
			R[r][1] += row[r] * to[i][1];
			for (let c = 0; c < 3; c++) N[r][c] += row[r] * row[c];
		}
	}
	const w = solveInPlace(N, R);
	return w ? [[w[0][0], w[1][0], w[2][0]],
	            [w[0][1], w[1][1], w[2][1]],
	            [0, 0, 1]] : null;
}

// Rotation + uniform scale + translation, closed form (Umeyama without the
// reflection guard - a mirrored fit is a legitimate thing to preview).
export function fitSimilarity(from, to) {
	const n = from.length;
	if (n < 2) return null;
	let fx = 0, fy = 0, tx = 0, ty = 0;
	for (let i = 0; i < n; i++) {
		fx += from[i][0]; fy += from[i][1]; tx += to[i][0]; ty += to[i][1];
	}
	fx /= n; fy /= n; tx /= n; ty /= n;
	let sa = 0, sb = 0, sf = 0;
	for (let i = 0; i < n; i++) {
		const ax = from[i][0] - fx, ay = from[i][1] - fy;
		const bx = to[i][0] - tx, by = to[i][1] - ty;
		sa += ax * bx + ay * by;
		sb += ax * by - ay * bx;
		sf += ax * ax + ay * ay;
	}
	if (sf < 1e-12) return null;
	const a = sa / sf, b = sb / sf;
	return [[a, -b, tx - a * fx + b * fy],
	        [b, a, ty - b * fx - a * fy],
	        [0, 0, 1]];
}

export function fitTranslation(from, to) {
	if (!from.length) return null;
	let dx = 0, dy = 0;
	for (let i = 0; i < from.length; i++) {
		dx += to[i][0] - from[i][0];
		dy += to[i][1] - from[i][1];
	}
	return [[1, 0, dx / from.length], [0, 1, dy / from.length], [0, 0, 1]];
}

export const MODELS = ["thin plate spline", "homography", "affine",
                       "similarity", "translation"];

const MIN_PAIRS = {
	"thin plate spline": 3, homography: 4, affine: 3, similarity: 2,
	translation: 1,
};

// The single interface the renderer and the overlays consume.
//   forward(x, y)  - where a source pixel goes (what applyTransformation does)
//   backward(x, y) - where a DESTINATION pixel comes from (what warpImage needs)
// For the TPS, backward is a SECOND fit (to -> from), exactly like
// features.ThinPlateSplineWarp does - warpImage is a backward warp, so sharing
// one fit moves the pixels the wrong way.
export function makeModel(kind, from, to, reg = 0) {
	const need = MIN_PAIRS[kind] ?? 3;
	if (from.length < need) {
		return { ok: false, reason: `${kind} needs ${need} pairs (have ${from.length})` };
	}
	if (kind === "thin plate spline") {
		const fw = fitTPS(from, to, reg);
		const bw = fitTPS(to, from, reg);
		if (!fw || !bw) return { ok: false, reason: "degenerate control points" };
		return {
			ok: true,
			forward: (x, y) => evalTPS(fw, from, x, y),
			backward: (x, y) => evalTPS(bw, to, x, y),
		};
	}
	const fit = { homography: fitHomography, affine: fitAffine,
	              similarity: fitSimilarity, translation: fitTranslation }[kind];
	const M = fit ? fit(from, to) : null;
	const Mi = M ? matInverse(M) : null;
	if (!M || !Mi) return { ok: false, reason: "degenerate control points" };
	return {
		ok: true,
		forward: (x, y) => evalMatrix(M, x, y),
		backward: (x, y) => evalMatrix(Mi, x, y),
	};
}

// A regular grid over the DESTINATION, textured with where each vertex came
// from. Piecewise-linear between vertices: at h ~ 10 px the interpolation error
// of a smooth warp is ~0.03 px, so this is visually exact and costs cols*rows
// model evaluations instead of one per pixel.
//   positions - destination grid in [0, 1] of the image
//   uvs       - backward(dst) / (w - 1, h - 1), matching the Python node's
//               normalized-1.0-is-pixel-w-1 convention
const indexCache = new Map();

export function gridIndices(cols, rows) {
	const key = `${cols}x${rows}`;
	let idx = indexCache.get(key);
	if (idx) return idx;
	idx = new Uint16Array(cols * rows * 6);
	let k = 0;
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			const a = r * (cols + 1) + c, b = a + 1;
			const d = a + cols + 1, e = d + 1;
			idx[k++] = a; idx[k++] = b; idx[k++] = d;
			idx[k++] = b; idx[k++] = e; idx[k++] = d;
		}
	}
	indexCache.set(key, idx);
	return idx;
}

// dstW/dstH: the DESTINATION frame's pixel size, when it is not the source
// image's (the correspondence editor's background mode fits image -> background,
// so the grid spans the background while the uvs index the source texture).
// Defaulting them to the source size keeps the classic single-image call sites.
export function meshFromModel(backward, imgW, imgH, cols, rows,
                              dstW = imgW, dstH = imgH) {
	const nv = (cols + 1) * (rows + 1);
	const positions = new Float32Array(nv * 2);
	const uvs = new Float32Array(nv * 2);
	const sx = imgW > 1 ? imgW - 1 : 1;
	const sy = imgH > 1 ? imgH - 1 : 1;
	const dx = dstW > 1 ? dstW - 1 : 1;
	const dy = dstH > 1 ? dstH - 1 : 1;
	let k = 0;
	for (let r = 0; r <= rows; r++) {
		for (let c = 0; c <= cols; c++) {
			const u = c / cols, v = r / rows;
			const [srcX, srcY] = backward(u * dx, v * dy);
			positions[k] = u; uvs[k] = srcX / sx; k++;
			positions[k] = v; uvs[k] = srcY / sy; k++;
		}
	}
	return { positions, uvs, indices: gridIndices(cols, rows), cols, rows };
}
