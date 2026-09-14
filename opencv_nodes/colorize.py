# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Colorization nodes using FLANN + superpixel matching (NPARRAY-native).

FLANN k-d tree matching over superpixel means, with all helpers inlined.
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import comfy.utils
from comfy_api.latest import io

from . import imageutils

CATEGORY = "image/CV/colorize"

PATCH = 5
PCA_DIM = 12
W_T, W_L, W_S = 2.0, 1.0, 1.0
STD_NORM = 32.0


def _stride_patches(L, step):
	"""Raw 5x5 patches on a stride-step grid.

	Returns (P, mL, sd) where P (h,w,25) in [0,1],
	mL (h,w) patch mean luminance [0,255],
	sd (h,w) patch std [0,255].
	"""
	pad = PATCH // 2
	Lp = np.pad(L, pad, mode="reflect")
	win = sliding_window_view(Lp, (PATCH, PATCH))
	win = win[::step, ::step]
	P = win.reshape(win.shape[0], win.shape[1], -1).astype(np.float32) / 255.0
	mL = P.mean(axis=2) * 255.0
	sd = P.std(axis=2) * 255.0
	return P, mL, sd


class _FeatureSpace:
	"""PCA-reduced raw-patch feature space shared by both images."""

	def __init__(self, P_c, mL_c, sd_c, mask_c, n_train=60000, seed=0):
		h, w, Dp = P_c.shape
		ys, xs = np.nonzero(mask_c)
		rng = np.random.default_rng(seed)
		idx = rng.choice(len(ys), min(n_train, len(ys)), replace=False)
		desc = P_c[ys[idx], xs[idx]]
		self.mean_d = desc.mean(axis=0)
		U, S, Vt = np.linalg.svd(desc - self.mean_d, full_matrices=False)
		self.pc = Vt[:PCA_DIM]
		T = (desc - self.mean_d) @ self.pc.T
		self.tscale = T.std(axis=0) + 1e-6

	def features(self, P, mL, sd):
		"""F: (h*w, D) flat feature array."""
		h, w, Dp = P.shape
		T = (P.reshape(-1, Dp) - self.mean_d) @ self.pc.T
		F = np.concatenate([
			W_T * T / self.tscale,
			(W_L * mL.ravel() / 255.0)[:, None],
			(W_S * sd.ravel() / STD_NORM)[:, None],
		], axis=1).astype(np.float32)
		return F, (h, w)


def _slic(L01, n_segments=300, iters=10, compactness=10.0):
	"""Simple SLIC in numpy. L01: float in [0,1]. Returns (labels, M)."""
	H, W = L01.shape
	step = max(2, int(np.sqrt(H * W / n_segments)))
	ys = np.arange(step // 2, H, step)
	xs = np.arange(step // 2, W, step)
	CY, CX = np.meshgrid(ys, xs, indexing="ij")
	M = CY.size
	cy = CY.ravel().astype(np.float64)
	cx = CX.ravel().astype(np.float64)
	cl = L01[cy.astype(int), cx.astype(int)]
	labels = -np.ones((H, W), np.int32)
	dist = np.full((H, W), np.inf)
	m_over_s = (compactness / step) ** 2
	yy_full, xx_full = np.mgrid[0:H, 0:W]
	S = step
	for _ in range(iters):
		dist[:] = np.inf
		for i in range(M):
			y0 = max(0, int(cy[i] - S))
			y1 = min(H, int(cy[i] + S + 1))
			x0 = max(0, int(cx[i] - S))
			x1 = min(W, int(cx[i] + S + 1))
			yy = yy_full[y0:y1, x0:x1]
			xx = xx_full[y0:y1, x0:x1]
			d = ((L01[y0:y1, x0:x1] - cl[i]) ** 2
				 + m_over_s * ((yy - cy[i]) ** 2 + (xx - cx[i]) ** 2))
			sl = (slice(y0, y1), slice(x0, x1))
			upd = d < dist[sl]
			dist[sl][upd] = d[upd]
			labels[sl][upd] = i
		lab = labels.ravel()
		cnt = np.bincount(lab, minlength=M).astype(np.float64)
		sy = np.bincount(lab, weights=yy_full.ravel(), minlength=M)
		sx = np.bincount(lab, weights=xx_full.ravel(), minlength=M)
		slum = np.bincount(lab, weights=L01.ravel(), minlength=M)
		ok = cnt > 0
		cy[ok] = sy[ok] / cnt[ok]
		cx[ok] = sx[ok] / cnt[ok]
		cl[ok] = slum[ok] / cnt[ok]
	return labels, M


def _segment_means(F, labels, M):
	"""Mean of flat features F (N,D) per segment label. Empty → global mean."""
	lab = labels.ravel()
	cnt = np.bincount(lab, minlength=M).astype(np.float64)
	cnt_safe = np.maximum(cnt, 1)
	D = F.shape[1]
	out = np.empty((M, D), np.float64)
	for j in range(D):
		out[:, j] = np.bincount(lab, weights=F[:, j], minlength=M) / cnt_safe
	glob = F.mean(axis=0)
	out[cnt == 0] = glob
	return out.astype(np.float32), cnt


def _guided_filter(guide, p, r, eps):
	"""Guided filter via boxFilter. guide, p float32, returns float32."""
	I = guide.astype(np.float32)
	P = p.astype(np.float32)
	ksize = (2 * r + 1, 2 * r + 1)
	mean_I = cv2.boxFilter(I, -1, ksize)
	mean_P = cv2.boxFilter(P, -1, ksize)
	var_I = cv2.boxFilter(I * I, -1, ksize) - mean_I * mean_I
	cov_IP = cv2.boxFilter(I * P, -1, ksize) - mean_I * mean_P
	a = cov_IP / (var_I + eps)
	b = mean_P - a * mean_I
	return cv2.boxFilter(a, -1, ksize) * I + cv2.boxFilter(b, -1, ksize)


def _or_default(val, default):
	"""Return val if it is a usable value, else default.

	Optional widget inputs may arrive as empty string when disconnected.
	"""
	if isinstance(val, str) and val.strip() == "":
		return default
	if val is None:
		return default
	return val


class ColorizeSuperpixelsFlann(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ColorizeSuperpixelsFlann",
			display_name="Colorize Superpixels (FLANN)",
			category=CATEGORY,
			description="Colorizes a grayscale image using a color reference image "
			            "via FLANN-based superpixel matching. Internally extracts "
			            "stride-2 patch features, builds a PCA feature space from "
			            "the reference, segments both images into superpixels (SLIC), "
			            "matches gray superpixels to color superpixels via FLANN "
			            "KDTree kNN, and transfers a/b chrominance through "
			            "distance-weighted kernel regression with guided filter "
			            "cleanup.",
			inputs=[
				io.Image.Input("target",
					tooltip="Grayscale image to colorize (converted to grayscale "
					        "internally, so a color image also works)."),
				io.Image.Input("reference",
					tooltip="Color reference whose a/b palette is transferred. "
					        "The target's own luminance structure is preserved."),
				io.Int.Input("n_seg", default=300, min=10, max=5000,
					tooltip="Number of SLIC superpixels. More segments = finer "
					        "color detail but slower."),
				io.Int.Input("k", default=8, min=1, max=128,
					tooltip="kNN neighbors. Higher = smoother, more blending."),
				io.Int.Input("flann_checks", default=16, min=1, max=1024,
					optional=True, advanced=True,
					tooltip="FLANN KDTree search checks. Higher = more accurate, "
					        "slower."),
				io.Int.Input("flann_trees", default=5, min=1, max=50,
					optional=True, advanced=True,
					tooltip="FLANN KDTree tree count. More trees = faster queries, "
					        "more memory."),
				io.Int.Input("guided_r", default=12, min=0, max=100,
					optional=True, advanced=True,
					tooltip="Guided filter radius (0 = disabled). Smooths "
					        "chrominance while respecting luminance edges."),
				io.Float.Input("guided_eps", default=0.04, min=0.0, max=10.0,
					step=0.01, optional=True, advanced=True,
					tooltip="Guided filter epsilon squared. Higher = smoother, "
					        "less edge-aware."),
			],
			outputs=[
				io.Image.Output(display_name="image",
					tooltip="Colorized result: target luminance + reference "
					        "chrominance transferred via FLANN superpixel matching."),
				io.Boolean.Output(display_name="success",
					tooltip="False when the pipeline could not produce a result "
					        "(empty input, no valid segments, FLANN found no matches)."),
			],
		)

	@classmethod
	def execute(cls, target, reference, n_seg=300, k=8,
	            flann_checks=16, flann_trees=5,
	            guided_r=12, guided_eps=0.04) -> io.NodeOutput:
		# Resolve optional defaults
		flann_checks = _or_default(flann_checks, 16)
		flann_trees = _or_default(flann_trees, 5)
		guided_r = _or_default(guided_r, 12)
		guided_eps = _or_default(guided_eps, 0.04)

		# --- Prepare images ---
		target_frames = imageutils.image_batch_to_bgr(target)
		ref_frames = imageutils.image_batch_to_bgr(reference)
		if not target_frames or not ref_frames:
			h = w = 64
			fake = imageutils.bgr_to_image_batch(
				[np.zeros((h, w, 3), np.uint8)])
			return io.NodeOutput(fake, False)

		# no per-frame loop exists here - this node is single-image, so the bar
		# tracks the seven heavy stages of the pipeline instead of a frame count
		pbar = comfy.utils.ProgressBar(7)

		target_bgr = target_frames[0]
		ref_bgr = ref_frames[0]

		gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)  # (H, W) uint8
		H, W = gray.shape

		lab_full = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
		L_col_full = lab_full[:, :, 0]  # (H, W) float32 0..255
		ab_full = lab_full[:, :, 1:]    # (H, W, 2) float32

		# --- Stride-2 feature grids ---
		STEP = 2
		gray_f32 = gray.astype(np.float32)
		P_g, mL_g, sd_g = _stride_patches(gray_f32, STEP)
		P_c, mL_c, sd_c = _stride_patches(L_col_full, STEP)
		pbar.update(1)
		h_g, w_g = P_g.shape[:2]
		h_c, w_c = P_c.shape[:2]

		# ab chrominance at stride grid, flat (Nc, 2)
		ab_c = np.stack([
			ab_full[::STEP, ::STEP, 0].ravel(),
			ab_full[::STEP, ::STEP, 1].ravel(),
		], axis=1)

		# Valid mask: all pixels are valid since we take the whole image
		mask_c = np.ones((h_c, w_c), bool)

		# --- FeatureSpace (PCA) ---
		try:
			fs = _FeatureSpace(P_c, mL_c, sd_c, mask_c)
			Fg, shape_g = fs.features(P_g, mL_g, sd_g)
			Fc, shape_c = fs.features(P_c, mL_c, sd_c)
			pbar.update(1)
		except Exception:
			h = w = 64
			fake = imageutils.bgr_to_image_batch(
				[np.zeros((h, w, 3), np.uint8)])
			# finish the bar: this is a clean failure reported through `success`,
			# and a bar frozen part-way reads as a hang instead
			pbar.update_absolute(7)
			return io.NodeOutput(fake, False)

		# --- Luminance at stride grid, [0, 1] ---
		L_GRAY01 = (gray_f32[::STEP, ::STEP] / 255.0).astype(np.float32)
		L_COL01 = (L_col_full[::STEP, ::STEP] / 255.0).astype(np.float32)

		# --- FLANN colorize pipeline (adapted from colorize_superpixels_flann) ---
		try:
			lab_g, M_g = _slic(L_GRAY01, n_segments=n_seg)
			lab_c, M_c = _slic(L_COL01, n_segments=n_seg)
			pbar.update(1)
		except Exception:
			h = w = 64
			fake = imageutils.bgr_to_image_batch(
				[np.zeros((h, w, 3), np.uint8)])
			pbar.update_absolute(7)
			return io.NodeOutput(fake, False)

		if M_g == 0 or M_c == 0:
			fake = imageutils.bgr_to_image_batch(
				[np.zeros((H, W, 3), np.uint8)])
			pbar.update_absolute(7)
			return io.NodeOutput(fake, False)

		Fg_seg, _ = _segment_means(Fg, lab_g, M_g)
		Fc_seg, cnt_c = _segment_means(Fc, lab_c, M_c)

		ab_seg = np.empty((M_c, 2), np.float64)
		lab_flat = lab_c.ravel()
		cnt_safe = np.maximum(cnt_c, 1)
		ab_seg[:, 0] = np.bincount(lab_flat, weights=ab_c[:, 0],
		                           minlength=M_c) / cnt_safe
		ab_seg[:, 1] = np.bincount(lab_flat, weights=ab_c[:, 1],
		                           minlength=M_c) / cnt_safe
		ab_seg[cnt_c == 0] = 128.0
		pbar.update(1)

		index_params = dict(algorithm=1, trees=flann_trees)
		search_params = dict(checks=flann_checks)
		try:
			matcher = cv2.FlannBasedMatcher(index_params, search_params)
			matcher.add([Fc_seg.astype(np.float32)])
			matcher.train()
			matches = matcher.knnMatch(Fg_seg.astype(np.float32),
			                           k=min(k, M_c))
			pbar.update(1)
		except cv2.error:
			fake = imageutils.bgr_to_image_batch(
				[np.zeros((H, W, 3), np.uint8)])
			pbar.update_absolute(7)
			return io.NodeOutput(fake, False)

		if not matches or all(len(row) == 0 for row in matches):
			fake = imageutils.bgr_to_image_batch(
				[np.zeros((H, W, 3), np.uint8)])
			pbar.update_absolute(7)
			return io.NodeOutput(fake, False)

		d = np.array([[m.distance for m in row] for row in matches],
		             dtype=np.float64)
		ii = np.array([[m.trainIdx for m in row] for row in matches],
		              dtype=np.int64)
		if d.ndim == 1:
			d = d[:, None]
			ii = ii[:, None]

		hband = d[:, -1:] + 1e-6
		wgt = np.exp(-0.5 * (d / hband) ** 2)
		wgt /= wgt.sum(axis=1, keepdims=True)
		seg_ab = (ab_seg[ii] * wgt[..., None]).sum(axis=1)

		a_map = seg_ab[:, 0][lab_g]
		b_map = seg_ab[:, 1][lab_g]

		a = cv2.resize(a_map.astype(np.float32), (W, H),
		               interpolation=cv2.INTER_LINEAR)
		b = cv2.resize(b_map.astype(np.float32), (W, H),
		               interpolation=cv2.INTER_LINEAR)
		pbar.update(1)

		if guided_r > 0:
			g01 = gray.astype(np.float32) / 255.0
			a = _guided_filter(g01, a, guided_r, guided_eps ** 2)
			b = _guided_filter(g01, b, guided_r, guided_eps ** 2)
		pbar.update(1)  # unconditional: the bar reaches 7 even with guided_r == 0

		# --- Compose LAB → BGR ---
		lab_result = np.stack([
			gray.astype(np.float32),
			a.astype(np.float32),
			b.astype(np.float32),
		], axis=-1).clip(0, 255).astype(np.uint8)
		result_bgr = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

		return io.NodeOutput(
			imageutils.bgr_to_image_batch([result_bgr]),
			True,
		)


NODES = [ColorizeSuperpixelsFlann]
