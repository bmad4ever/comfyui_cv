# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Headless software renderer for 'CV Preview 3D (Calibrated Camera)'.

Renders a GLB through the FULL OpenCV camera model, server-side, so the node
can emit an IMAGE output (the browser widget renders live but cannot feed a
Python output mid-run - a canvas capture would always be one run stale).

Pipeline, mirroring web/cv_preview3d.js:
  * parse the GLB (JSON + BIN chunks, no external resources, no Draco),
	walk the node hierarchy, collect world-space triangles + per-primitive
	base-color texture/factor;
  * map Three.js world -> OpenCV world (negate y/z - the same diag(1,-1,-1)
	the bridge node uses), apply the [R|t] extrinsics, project through an
	OVERSCANNED pinhole K' and rasterize with a z-buffer (perspective-correct
	UVs, flat Lambert shading, double-sided);
  * bend the pinhole render into the real lens with cv2.undistortPoints: for
	every output (distorted) pixel the iterative inverse Brown-Conrady model
	finds the ideal ray, which indexes the overscan render via cv2.remap -
	per-pixel exact, unlike distorting only the vertices;
  * alpha-composite over the background photo.

Deliberately minimal: triangles only, no skinning/animation/Draco/sRGB curves,
nearest-texel sampling. Plenty for "does the calibrated camera line up with
the photo", which is what the IMAGE output is for.
"""

import functools
import json
import struct

import cv2
import numpy as np

_COMPONENT_DTYPES = {
	5120: np.int8, 5121: np.uint8, 5122: np.int16,
	5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
_TYPE_COMPONENTS = {
	"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16,
}
_FLIP = np.array([1.0, -1.0, -1.0])  # Three.js world <-> OpenCV world
DEPTH_MAX_CODE = 0xFFFFFF  # 24-bit depth packing, see encode_depth_png


def _parse_glb(path):
	with open(path, "rb") as f:
		blob = f.read()
	if len(blob) < 20 or blob[:4] != b"glTF":
		raise ValueError(f"{path!r} is not a binary glTF (.glb) file.")
	_, _, total = struct.unpack("<4sII", blob[:12])
	offset, gltf, binary = 12, None, b""
	while offset + 8 <= min(total, len(blob)):
		clen, ctype = struct.unpack("<I4s", blob[offset:offset + 8])
		chunk = blob[offset + 8:offset + 8 + clen]
		if ctype == b"JSON":
			gltf = json.loads(chunk)
		elif ctype == b"BIN\x00":
			binary = chunk
		offset += 8 + clen
	if gltf is None:
		raise ValueError(f"{path!r} has no JSON chunk - corrupt GLB.")
	for buf in gltf.get("buffers", []):
		if "uri" in buf:
			raise ValueError("GLB with external buffer URIs is not supported - "
			                 "re-export as a self-contained .glb.")
	return gltf, binary


def _accessor(gltf, binary, idx):
	acc = gltf["accessors"][idx]
	if "sparse" in acc:
		raise ValueError("sparse glTF accessors are not supported.")
	comp = _COMPONENT_DTYPES[acc["componentType"]]
	ncomp = _TYPE_COMPONENTS[acc["type"]]
	itemsize = np.dtype(comp).itemsize
	count = acc["count"]
	if "bufferView" not in acc:
		return np.zeros((count, ncomp), comp)
	bv = gltf["bufferViews"][acc["bufferView"]]
	start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
	stride = bv.get("byteStride") or ncomp * itemsize
	arr = np.ndarray((count, ncomp), comp, buffer=binary, offset=start,
	                 strides=(stride, itemsize))
	return np.ascontiguousarray(arr)


def _node_matrix(node):
	if "matrix" in node:
		return np.asarray(node["matrix"], np.float64).reshape(4, 4, order="F")
	m = np.eye(4)
	if "scale" in node:
		m[:3, :3] *= np.asarray(node["scale"], np.float64)
	if "rotation" in node:  # glTF quaternion is (x, y, z, w)
		x, y, z, w = node["rotation"]
		rot = np.array([
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		])
		m[:3, :] = rot @ m[:3, :]
	if "translation" in node:
		m[:3, 3] = node["translation"]
	return m


def _texture_bgr(gltf, binary, tex_index, cache):
	if tex_index in cache:
		return cache[tex_index]
	img = None
	source = gltf["textures"][tex_index].get("source")
	if source is not None:
		info = gltf["images"][source]
		if "bufferView" in info:
			bv = gltf["bufferViews"][info["bufferView"]]
			start = bv.get("byteOffset", 0)
			raw = np.frombuffer(binary, np.uint8, bv["byteLength"], start)
			decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
			if decoded is not None:
				img = decoded.astype(np.float32) / 255.0
	cache[tex_index] = img
	return img


def _collect_primitives(gltf, binary):
	"""Yields (world_vertices Nx3 (three.js space), faces Mx3, uvs Nx2 or None,
	texture BGR float or None, base_color BGR array) for every triangle
	primitive reachable from the default scene."""
	scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
	stack = [(idx, np.eye(4)) for idx in scene.get("nodes", [])]
	tex_cache = {}
	prims = []
	while stack:
		idx, parent = stack.pop()
		node = gltf["nodes"][idx]
		world = parent @ _node_matrix(node)
		for child in node.get("children", []):
			stack.append((child, world))
		if "mesh" not in node:
			continue
		for prim in gltf["meshes"][node["mesh"]]["primitives"]:
			if prim.get("mode", 4) != 4 or "POSITION" not in prim.get("attributes", {}):
				continue
			verts = _accessor(gltf, binary, prim["attributes"]["POSITION"]).astype(np.float64)
			verts = verts @ world[:3, :3].T + world[:3, 3]
			if "indices" in prim:
				faces = _accessor(gltf, binary, prim["indices"]).reshape(-1)
			else:
				faces = np.arange(len(verts), dtype=np.uint32)
			faces = faces.astype(np.int64).reshape(-1, 3)
			uvs = None
			if "TEXCOORD_0" in prim["attributes"]:
				uvs = _accessor(gltf, binary, prim["attributes"]["TEXCOORD_0"]).astype(np.float64)
				if uvs.shape[1] != 2:
					uvs = None
			texture, base = None, np.array([1.0, 1.0, 1.0])
			mat_idx = prim.get("material")
			if mat_idx is not None:
				pbr = gltf["materials"][mat_idx].get("pbrMetallicRoughness", {})
				factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
				base = np.array(factor[2::-1], np.float64)  # RGB -> BGR
				tex_info = pbr.get("baseColorTexture")
				if tex_info is not None:
					texture = _texture_bgr(gltf, binary, tex_info["index"], tex_cache)
			prims.append((verts, faces, uvs, texture, base))
	return prims


def compose_transform(position=None, quaternion=None, scale=None):
	"""4x4 model matrix (Three.js world space, T * R * S) from the gizmo's
	position [x, y, z], quaternion [x, y, z, w] and scale (scalar or
	[x, y, z]) - the dict 'CV Preview 3D (Calibrated Camera)' stores in its
	object_transform widget. Missing parts default to identity."""
	m = np.eye(4)
	if scale is not None:
		s = np.asarray(scale, np.float64).reshape(-1)
		s = np.full(3, s[0]) if s.size == 1 else s[:3]
		m[:3, :3] *= s
	if quaternion is not None:
		x, y, z, w = np.asarray(quaternion, np.float64).reshape(-1)[:4]
		n = np.sqrt(x * x + y * y + z * z + w * w)
		if n > 1e-12:
			x, y, z, w = x / n, y / n, z / n, w / n
			rot = np.array([
				[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
				[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
				[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
			])
			m[:3, :] = rot @ m[:3, :]
	if position is not None:
		m[:3, 3] = np.asarray(position, np.float64).reshape(-1)[:3]
	return m


def _camera_from_info(camera_info):
	"""(R, t, K, dist, (W, H)) in OpenCV conventions. Falls back to a camera
	synthesized from position/quaternion/fov when the underscore extras are
	absent, so a plain LOAD3D_CAMERA dict still renders."""
	info = camera_info or {}
	ext = info.get("_extrinsics")
	if ext is not None:
		ext = np.asarray(ext, np.float64)
		R, t = ext[:3, :3], ext[:3, 3]
	else:
		q = info.get("quaternion") or {}
		x, y, z, w = (q.get(k, 0.0) for k in ("x", "y", "z", "w"))
		if not np.isfinite(x + y + z + w) or abs(x) + abs(y) + abs(z) + abs(w) < 1e-9:
			w = 1.0
		c2w = np.array([
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		])
		# invert the bridge's mapping: R_cv = F @ R_three_c2w^T @ M
		R = (_FLIP[:, None] * c2w.T) * _FLIP[None, :]
		p = info.get("position") or {}
		pos = np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 10.0)])
		t = -R @ (_FLIP * pos)
	K = info.get("_intrinsics")
	size = info.get("_image_size")
	if K is not None and size is not None:
		K = np.asarray(K, np.float64)
		W, H = int(round(size[0])), int(round(size[1]))
	else:
		H = 600
		aspect = float(info.get("aspect") or 4.0 / 3.0)
		W = int(round(H * aspect))
		fov = float(info.get("fov") or 50.0)
		fy = H / (2.0 * np.tan(np.radians(fov) / 2.0))
		K = np.array([[fy, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]])
	dist = np.asarray(info.get("_dist_coeffs") or [], np.float64).reshape(-1)
	if dist.size and not dist.any():
		dist = dist[:0]
	W, H = max(W, 8), max(H, 8)
	return R, t, K, dist, (W, H)


def _rasterize(prims, R, t, K, size_px, margin_px, shaded=True):
	"""Renders through the overscanned pinhole (fx, fy, cx+mx, cy+my) at
	(W+2mx, H+2my). Returns float32 BGR + alpha + camera-space z-buffer (inf
	where empty), all at overscan size. shaded=False disables the Lambert term
	(unlit texture/base colors)."""
	W, H = size_px
	mx, my = margin_px
	Wp, Hp = W + 2 * mx, H + 2 * my
	fx, fy = K[0, 0], K[1, 1]
	cxp, cyp = K[0, 2] + mx, K[1, 2] + my
	color = np.zeros((Hp, Wp, 3), np.float32)
	alpha = np.zeros((Hp, Wp), np.float32)
	zbuf = np.full((Hp, Wp), np.inf, np.float32)
	light = np.array([0.25, -0.35, 0.9])
	light = light / np.linalg.norm(light)

	for verts_three, faces, uvs, texture, base in prims:
		cam = (_FLIP * verts_three) @ R.T + t
		z = cam[:, 2]
		valid = z > 1e-6
		u = np.where(valid, fx * cam[:, 0] / np.where(valid, z, 1.0) + cxp, 0.0)
		v = np.where(valid, fy * cam[:, 1] / np.where(valid, z, 1.0) + cyp, 0.0)
		zinv = np.where(valid, 1.0 / np.where(valid, z, 1.0), 0.0)
		for f in faces:
			if not valid[f].all():
				continue  # a robust near-plane clip is overkill for a preview
			ux, vy = u[f], v[f]
			x0 = max(int(np.floor(ux.min())), 0)
			x1 = min(int(np.ceil(ux.max())) + 1, Wp)
			y0 = max(int(np.floor(vy.min())), 0)
			y1 = min(int(np.ceil(vy.max())) + 1, Hp)
			if x0 >= x1 or y0 >= y1:
				continue
			area = ((ux[1] - ux[0]) * (vy[2] - vy[0])
			        - (ux[2] - ux[0]) * (vy[1] - vy[0]))
			if abs(area) < 1e-9:
				continue
			# OpenCV convention throughout: integer pixel indices ARE the sample
			# coordinates (centers), matching K and the remap in _distort
			gx, gy = np.meshgrid(np.arange(x0, x1, dtype=np.float64),
			                     np.arange(y0, y1, dtype=np.float64))
			b0 = ((ux[1] - gx) * (vy[2] - gy) - (ux[2] - gx) * (vy[1] - gy)) / area
			b1 = ((ux[2] - gx) * (vy[0] - gy) - (ux[0] - gx) * (vy[2] - gy)) / area
			b2 = 1.0 - b0 - b1
			inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
			if not inside.any():
				continue
			zi = b0 * zinv[f[0]] + b1 * zinv[f[1]] + b2 * zinv[f[2]]
			depth = np.where(zi > 1e-12, 1.0 / np.maximum(zi, 1e-12), np.inf).astype(np.float32)
			patch = zbuf[y0:y1, x0:x1]
			win = inside & (depth < patch)
			if not win.any():
				continue
			if shaded:
				# flat Lambert on the face normal in camera space, double-sided
				n = np.cross(cam[f[1]] - cam[f[0]], cam[f[2]] - cam[f[0]])
				norm = np.linalg.norm(n)
				shade = 0.3 + 0.7 * abs(float(n @ light) / norm) if norm > 0 else 1.0
			else:
				shade = 1.0
			if texture is not None and uvs is not None:
				# perspective-correct UV: interpolate uv/z, divide by 1/z
				uw = (b0 * uvs[f[0], 0] * zinv[f[0]] + b1 * uvs[f[1], 0] * zinv[f[1]]
				      + b2 * uvs[f[2], 0] * zinv[f[2]]) / np.maximum(zi, 1e-12)
				vw = (b0 * uvs[f[0], 1] * zinv[f[0]] + b1 * uvs[f[1], 1] * zinv[f[1]]
				      + b2 * uvs[f[2], 1] * zinv[f[2]]) / np.maximum(zi, 1e-12)
				th, tw = texture.shape[:2]
				tx = np.clip((np.mod(uw, 1.0) * tw).astype(np.int32), 0, tw - 1)
				ty = np.clip((np.mod(vw, 1.0) * th).astype(np.int32), 0, th - 1)
				texel = texture[ty, tx] * base
			else:
				texel = np.broadcast_to(base, (y1 - y0, x1 - x0, 3))
			color[y0:y1, x0:x1][win] = (texel * shade)[win]
			alpha[y0:y1, x0:x1][win] = 1.0
			patch[win] = depth[win]
	return color, alpha, zbuf


# Half an output pixel, and 8 rounds of the fixed point: BOTH are shared with
# web/cvlib/distort_math.js (UNDISTORT_PIXEL_TOL, UNDISTORT_ITERATIONS). The
# renderer deliberately does NOT call cv2.undistortPoints here - see
# _undistort_map.
_UNDISTORT_PIXEL_TOL = 0.5
_UNDISTORT_ITERATIONS = 8


def _undistort_normalized(xd, yd, d, iterations=_UNDISTORT_ITERATIONS):
	"""Inverse Brown-Conrady by the SAME truncated fixed point the viewer's
	fragment shader runs, in the same float32, so the two produce one picture.

	Returns (x, y) and the residual of the FORWARD model in normalized units.
	That residual is not diagnostics: it is the only thing that says the answer
	means anything (see _undistort_map)."""
	k1, k2, p1, p2, k3 = np.asarray(d, np.float64).reshape(-1)[:5].astype(np.float32)
	xd = np.ascontiguousarray(xd, np.float32)
	yd = np.ascontiguousarray(yd, np.float32)
	x, y = xd.copy(), yd.copy()
	# p1 = p2 = 0 is the common case and halves the work per round
	tangential = bool(p1) or bool(p2)
	for _ in range(iterations):
		r2 = x * x + y * y
		radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
		if tangential:
			tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
			ty = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
			x, y = (xd - tx) / radial, (yd - ty) / radial
		else:
			x, y = xd / radial, yd / radial
	r2 = x * x + y * y
	radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
	bx, by = x * radial, y * radial
	if tangential:
		bx = bx + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
		by = by + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
	return x, y, np.hypot(bx - xd, by - yd)


@functools.lru_cache(maxsize=2)
def _undistort_map_cached(K_bytes, dist_bytes, size_px, margin_px):
	K = np.frombuffer(K_bytes, np.float64).reshape(3, 3)
	dist = np.frombuffer(dist_bytes, np.float64)
	maps = _undistort_map_uncached(K, dist, size_px, margin_px)
	for m in maps:
		m.flags.writeable = False   # a cached array is shared, never a scratch buffer
	return maps


def _undistort_map(K, dist, size_px, margin_px):
	"""Cached front door for _undistort_map_uncached: a batch, a video or a
	re-render with the same camera rebuilds the same 2 x 7.7 MB pair otherwise,
	and it is the most expensive part of a distorted render. Keyed on the BYTES
	of K and dist, so two equal cameras hit the same entry."""
	K = np.ascontiguousarray(K, np.float64)
	dist = np.ascontiguousarray(np.asarray(dist, np.float64).reshape(-1))
	map_x, map_y = _undistort_map_cached(K.tobytes(), dist.tobytes(),
	                                     tuple(size_px), tuple(margin_px))
	return map_x, map_y


def _undistort_map_uncached(K, dist, size_px, margin_px):
	"""cv2.remap maps from the DISTORTED output frame into the overscanned
	pinhole render, with the pixels the lens model cannot explain sent off the
	edge of that render (-1, so BORDER_CONSTANT leaves them empty).

	The guard is not defensive plumbing. A distortion fit with a strong negative
	k3 - what a chessboard-only calibration produces routinely, including
	workflow 57's own - makes r_d(r_u) turn over inside the image: past that
	turning point NO ray projects to the pixel, so the inverse has nothing to
	converge to. Nothing in cv2 says so; it truncates its iteration and returns
	whatever it had reached, which for workflow 57 is meaningless over the outer
	22% of the frame and can point back INTO the render, painting a smeared ghost
	of the model in the corners. Re-projecting the answer through the forward
	model and demanding it land back on its own pixel is the honest test.

	WHY NOT cv2.undistortPoints. The viewer cannot call it, so the renderer does
	not either: measured against a COLMAP-style damped Newton run to convergence
	(100 iterations, trust region) over workflow 57's frame, cv2's default
	5 undamped rounds leave 76.0% of the frame usable where the shader's 8 leave
	77.7% and the true invertible disc is 78.2% - so calling cv2 here put a
	6-to-20-pixel-wide ring of DISAGREEMENT between the IMAGE output and the
	widget, which is the one thing this renderer exists to avoid. One inverse,
	one tolerance, both sides.

	float32 throughout, because float32 is what the shader runs: the two then
	agree on the borderline pixels rather than merely near them."""
	W, H = size_px
	mx, my = margin_px
	gx = np.arange(W, dtype=np.float32)[None, :]
	gy = np.arange(H, dtype=np.float32)[:, None]
	xd = (gx - np.float32(K[0, 2])) / np.float32(K[0, 0])
	yd = (gy - np.float32(K[1, 2])) / np.float32(K[1, 1])
	xd, yd = np.broadcast_arrays(xd, yd)
	# a runaway iterate overflows float32 on its way back through the forward
	# model, and that overflow IS the answer (the comparison below rejects it),
	# so it must not print a warning on every frame
	with np.errstate(over="ignore", invalid="ignore"):
		x, y, resid = _undistort_normalized(xd, yd, dist)
		resid_px = resid * np.float32(min(K[0, 0], K[1, 1]))
		# negated, so a non-finite residual also reads as "no ray here"
		bad = ~(resid_px <= _UNDISTORT_PIXEL_TOL)
		map_x = (np.float32(K[0, 0]) * x + np.float32(K[0, 2] + mx))
		map_y = (np.float32(K[1, 1]) * y + np.float32(K[1, 2] + my))
	map_x, map_y = np.ascontiguousarray(map_x), np.ascontiguousarray(map_y)
	map_x[bad] = -1.0
	map_y[bad] = -1.0
	return map_x, map_y


def _distort(color, alpha, depth, zbuf, K, dist, size_px, margin_px):
	"""Warps the overscanned pinhole render into the real (distorted) lens:
	each output pixel is undistorted (iterative inverse Brown-Conrady, via
	cv2.undistortPoints) and the resulting ideal ray indexes the render.

	The camera-space z-buffer rides along so scene occlusion can be tested in
	the same (distorted) frame the photo and its depth map live in. It is warped
	as INVERSE depth: cv2.remap on a buffer full of `inf` would spread NaN over
	every silhouette, while 1/z is 0 exactly where the model is absent and
	interpolates the way a depth buffer should.

	Pixels the lens model cannot account for are left EMPTY rather than sampled
	from wherever the failed inverse happened to point - see _undistort_map."""
	map_x, map_y = _undistort_map(K, dist, size_px, margin_px)
	color = cv2.remap(color, map_x, map_y, cv2.INTER_LINEAR,
	                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	alpha = cv2.remap(alpha, map_x, map_y, cv2.INTER_LINEAR,
	                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	depth = cv2.remap(depth, map_x, map_y, cv2.INTER_LINEAR,
	                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	zinv = np.where(np.isfinite(zbuf), 1.0 / np.maximum(zbuf, 1e-12), 0.0)
	zinv = cv2.remap(zinv.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
	                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	zbuf = np.where(zinv > 0.0, 1.0 / np.maximum(zinv, 1e-12),
	                np.inf).astype(np.float32)
	return color, alpha, depth, zbuf


def _occlude(color, alpha, depth, zbuf, scene_depth, depth_bias):
	"""Cuts the model where the photographed scene is nearer to the camera.

	`scene_depth` is metric camera-space Z in the SAME units as the pose, at any
	resolution (nearest-resampled to the render). Non-finite or <= 0 means "no
	measurement here", which has to read as INFINITELY FAR: a stereo disparity
	map has holes, and treating a hole as z=0 would erase the model wherever the
	matcher failed. Returns the occlusion coverage in [0, 1] and the z-buffer of
	the COMPOSITE (min of model and scene where the scene is known), which is
	the surface the camera actually sees once the cut has been made."""
	h, w = alpha.shape
	scene = np.asarray(scene_depth, np.float32)
	if scene.ndim == 3:
		scene = scene[:, :, 0]
	if scene.shape != (h, w):
		scene = cv2.resize(scene, (w, h), interpolation=cv2.INTER_NEAREST)
	known = np.isfinite(scene) & (scene > 0.0)
	hidden = known & (zbuf > scene + float(depth_bias))
	occlusion = (hidden & (alpha > 0.0)).astype(np.float32)
	alpha = np.where(hidden, 0.0, alpha).astype(np.float32)
	color = np.where(hidden[..., None], 0.0, color).astype(np.float32)
	depth = np.where(hidden, 0.0, depth).astype(np.float32)
	# the metric buffer follows what is now VISIBLE: the scene wherever it is
	# nearer (it is the surface an annotation anchor there sits behind), the
	# model elsewhere, +inf where neither was measured
	zbuf = np.where(known, np.minimum(zbuf, scene), zbuf).astype(np.float32)
	return color, alpha, depth, occlusion, zbuf


def encode_depth_png(scene_depth):
	"""Packs a metric depth map into an 8-bit RGB image for the browser widget.

	24-bit fixed point over [0, z_max], big-endian across R/G/B, with 0 reserved
	for "unknown" (the +1 offset below), decoded by web/cvlib/depth_codec.js.
	NOT a 16-bit gray PNG: a browser canvas silently truncates those to 8 bits,
	which at 30 m would quantise depth to 12 cm steps and make the occlusion
	edge crawl. Returns (uint8 HxWx3 RGB image, z_max)."""
	z = np.asarray(scene_depth, np.float32)
	if z.ndim == 3:
		z = z[:, :, 0]
	known = np.isfinite(z) & (z > 0.0)
	z_max = float(z[known].max()) if known.any() else 1.0
	if not np.isfinite(z_max) or z_max <= 0.0:
		z_max = 1.0
	# 1..MAX_CODE, leaving 0 free to mean "no measurement"
	code = np.zeros(z.shape, np.uint32)
	code[known] = 1 + np.round(np.clip(z[known] / z_max, 0.0, 1.0)
	                           * (DEPTH_MAX_CODE - 1)).astype(np.uint32)
	rgb = np.stack([(code >> 16) & 0xFF, (code >> 8) & 0xFF, code & 0xFF], -1)
	return rgb.astype(np.uint8), z_max


def decode_depth_rgb(rgb, z_max):
	"""Inverse of encode_depth_png, mirroring web/cvlib/depth_codec.js exactly.
	0 (unknown) decodes to +inf, which is what _occlude treats as 'no scene'."""
	c = np.asarray(rgb, np.uint32)
	code = (c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]
	z = (code.astype(np.float64) - 1.0) / (DEPTH_MAX_CODE - 1) * float(z_max)
	return np.where(code == 0, np.inf, z).astype(np.float32)


def render_calibrated(glb_path, camera_info, bg_bgr=None, overscan=0.25,
                      shaded=True, resolution_scale=1.0, antialias=1,
                      model_transform=None, scene_depth=None, depth_bias=0.0):
	"""Renders the GLB through the OpenCV camera in camera_info and composites
	over bg_bgr (uint8 BGR, resized to the output size). Returns
	(uint8 BGR image, float32 alpha silhouette in [0, 1], float32 depth in
	[0, 1], float32 occlusion coverage in [0, 1], float32 METRIC camera-space
	Z), all at _image_size * resolution_scale. No grid/axes - model and photo
	only.

	The fifth value is the rasterizer's own z-buffer, in scene units - the same
	quantity 'CV Rasterize Mesh' returns as `depth`, and what
	cv2.projectPoints-based hidden-line removal needs. It is +inf where nothing
	was rendered: that reads as "no surface" to every consumer, so a point
	projecting off the silhouette stays visible instead of being culled by a
	background depth of 0.

	`scene_depth` (optional) is a metric camera-space Z map of the PHOTOGRAPHED
	scene, in the same units as the pose - the photo's own z-buffer, typically
	channel 2 of cv2.reprojectImageTo3D on a stereo disparity. Where it is
	nearer than the model, the model is cut away, which is what makes the
	composite an AR overlay rather than a sticker. `depth_bias` is added to the
	scene depth first: stereo depth is noisy, and pushing the scene a little
	further away keeps a surface the model is standing ON from eating it.

	The depth map is OBJECT-relative, not an arbitrary near/far: 1 at the
	model's point nearest the camera, 0 at its farthest point - limits taken
	over ALL (transformed) vertices, occluded ones included, so the visible
	minimum need not reach 0. Background is 0.

	resolution_scale scales the OUTPUT resolution (K scales with it, so the
	framing is identical); antialias (1/2/4) supersamples internally - the
	scene is rasterized AND lens-warped at the higher resolution, then
	box-filtered down, which also antialiases the silhouette/depth edges."""
	gltf, binary = _parse_glb(glb_path)
	prims = _collect_primitives(gltf, binary)
	if model_transform is not None:
		# the gizmo's matrix, applied in Three.js world space exactly like the
		# viewer's modelRoot group (compose_transform builds it)
		A = np.asarray(model_transform, np.float64)
		prims = [(v @ A[:3, :3].T + A[:3, 3], f, uv, tex, base)
		         for (v, f, uv, tex, base) in prims]
	R, t, K, dist, (W, H) = _camera_from_info(camera_info)
	out_w = max(8, int(round(W * float(resolution_scale))))
	out_h = max(8, int(round(H * float(resolution_scale))))
	ss = max(1, int(antialias))
	rw, rh = out_w * ss, out_h * ss
	Ks = K.copy()
	Ks[0, :] *= rw / float(W)
	Ks[1, :] *= rh / float(H)
	margin = (int(round(rw * overscan)), int(round(rh * overscan))) if dist.size else (0, 0)
	color, alpha, zbuf = _rasterize(prims, R, t, Ks, (rw, rh), margin, shaded)
	# object-relative depth normalization: limits from ALL vertices in front of
	# the camera (occluded ones included), NOT from the visible z-buffer - the
	# nearest surface point maps to 1, the farthest (possibly hidden) to 0
	z_lims = [z[z > 1e-6] for z in
	          (((_FLIP * v) @ R.T + t)[:, 2] for v, _f, _uv, _tex, _base in prims)]
	z_lims = [z for z in z_lims if z.size]
	if z_lims:
		z_near = min(float(z.min()) for z in z_lims)
		z_far = max(float(z.max()) for z in z_lims)
	else:
		z_near = z_far = 0.0
	span = z_far - z_near
	if span > 1e-12:
		depth = np.clip((z_far - zbuf) / span, 0.0, 1.0).astype(np.float32)
	else:
		depth = alpha.copy()  # flat (or empty) object: every covered pixel is nearest
	if dist.size:
		color, alpha, depth, zbuf = _distort(color, alpha, depth, zbuf, Ks, dist,
		                                     (rw, rh), margin)
	else:
		# margin is (0, 0) already
		color, alpha, depth, zbuf = (color[:rh, :rw], alpha[:rh, :rw],
		                             depth[:rh, :rw], zbuf[:rh, :rw])
	# occlude BEFORE the antialias downsample, so the cut edge picks up the same
	# fractional coverage as the silhouette edge instead of a hard staircase
	occlusion = np.zeros_like(alpha)
	if scene_depth is not None:
		color, alpha, depth, occlusion, zbuf = _occlude(color, alpha, depth,
		                                                zbuf, scene_depth,
		                                                depth_bias)
	if ss > 1:
		color = cv2.resize(color, (out_w, out_h), interpolation=cv2.INTER_AREA)
		alpha = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_AREA)
		depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_AREA)
		occlusion = cv2.resize(occlusion, (out_w, out_h),
		                       interpolation=cv2.INTER_AREA)
		# NEAREST, never AREA: box-filtering a z-buffer averages across the
		# silhouette into a surface that is not there, and any window touching
		# the +inf background would poison the whole output pixel
		zbuf = cv2.resize(zbuf, (out_w, out_h),
		                  interpolation=cv2.INTER_NEAREST)
	if bg_bgr is not None:
		bg = cv2.resize(bg_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
		bg = bg.astype(np.float32) / 255.0
	else:
		bg = np.full((out_h, out_w, 3), 0.13, np.float32)
	a = alpha[..., None]
	out = bg * (1.0 - a) + color * a
	return (np.clip(out * 255.0, 0, 255).astype(np.uint8), alpha, depth,
	        occlusion, np.ascontiguousarray(zbuf, np.float32))


def load_mesh(path):
	"""(vertices Nx3 float32, triangles Mx3 int32) of a GLB/GLTF, in Three.js
	world space - every triangle primitive of the default scene merged into one
	indexed mesh with the node transforms baked in.

	Shares the parser with render_calibrated, so what the viewer draws and what
	the geometry nodes measure can never drift apart. Materials, UVs and normals
	are dropped: the consumers (cv2.rapid, ppf_match_3d, ICP) want positions and
	connectivity only.
	"""
	gltf, binary = _parse_glb(path)
	verts, faces, offset = [], [], 0
	for v, f, _uv, _tex, _base in _collect_primitives(gltf, binary):
		verts.append(v)
		faces.append(f + offset)
		offset += len(v)
	if not verts:
		return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
	return (np.ascontiguousarray(np.concatenate(verts), np.float32),
	        np.ascontiguousarray(np.concatenate(faces), np.int32))


def split_long_edges(pts3d, tris, max_edge, max_rounds=24):
	"""Longest-edge bisection: repeatedly split the longest edge of every
	triangle that still exceeds `max_edge`, 1 triangle -> 2.

	Why this and not uniform 1->4 subdivision: cv2.rapid's extractControlPoints
	interpolates its 3D control points LINEARLY between consecutive silhouette
	VERTICES, with the blend factor taken from screen-space arc length. The
	resulting control point is therefore wrong by roughly L_px^2 / (16*f)
	(the perspective/affine mismatch) and its 2D position by the chord sagitta
	L_px^2 / (8*R_px) - both driven by L_px, the on-screen spacing of silhouette
	vertices, and neither reduced by rendering at a higher resolution. Bisection
	reaches a given L_px for about half the triangles uniform subdivision needs,
	and only where the mesh is actually coarse.

	Splitting the longest edge alone leaves T-junctions; that is harmless here
	(the neighbour's rasterized edge still passes over the new vertex, so it is
	still picked up as a silhouette vertex) and later rounds split the neighbour
	anyway once its own longest edge is the one over budget.
	"""
	P = np.asarray(pts3d, np.float64).reshape(-1, 3)
	T = np.asarray(tris, np.int64).reshape(-1, 3)
	if len(P) == 0 or len(T) == 0 or not np.isfinite(max_edge) or max_edge <= 0:
		return (np.ascontiguousarray(P, np.float32),
		        np.ascontiguousarray(T, np.int32))
	for _ in range(int(max_rounds)):
		lengths = np.stack(
			[np.linalg.norm(P[T[:, i]] - P[T[:, (i + 1) % 3]], axis=1)
			 for i in range(3)], axis=1)
		which = lengths.argmax(axis=1)
		longest = lengths[np.arange(len(T)), which]
		split = longest > max_edge
		if not split.any():
			break
		hot = T[split]
		w = which[split]
		rows = np.arange(len(hot))
		a = hot[rows, w]                    # longest edge start
		b = hot[rows, (w + 1) % 3]          # longest edge end
		c = hot[rows, (w + 2) % 3]          # opposite corner
		# one new vertex per distinct edge, shared by both adjacent triangles
		key = np.sort(np.stack([a, b], axis=1), axis=1)
		uniq, inv = np.unique(key, axis=0, return_inverse=True)
		mid = len(P) + inv.reshape(-1)
		P = np.concatenate([P, 0.5 * (P[uniq[:, 0]] + P[uniq[:, 1]])])
		T = np.concatenate([T[~split],
		                    np.stack([a, mid, c], axis=1),
		                    np.stack([mid, b, c], axis=1)])
	return (np.ascontiguousarray(P, np.float32),
	        np.ascontiguousarray(T, np.int32))


def rasterize_mesh(pts3d, tris, rvec, tvec, K, size_px):
	"""(silhouette float32 HxW in 0/1, depth float32 HxW) for a mesh already in
	OPENCV object space - the arrays a graph carries, not a GLB.

	Shares `_rasterize` with the calibrated viewer, so the mask this returns and
	the picture that node draws come from the same z-buffer. `_rasterize` takes
	Three.js-space vertices and applies `_FLIP` itself, so the flip is undone on
	the way in (it is its own inverse).

	depth is METRIC camera-space Z in the mesh's own units, `inf` where nothing
	was hit - which is exactly what `Preview3DCalibrated.scene_depth` treats as
	"no measurement here". Pinhole only: lens distortion is not applied.
	"""
	import cv2 as _cv2
	W, H = int(size_px[0]), int(size_px[1])
	pts = np.asarray(pts3d, np.float64).reshape(-1, 3)
	faces = np.asarray(tris, np.int64).reshape(-1, 3)
	if len(pts) == 0 or len(faces) == 0 or faces.max(initial=-1) >= len(pts):
		return (np.zeros((H, W), np.float32), np.full((H, W), np.inf, np.float32))
	R = _cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]
	t = np.asarray(tvec, np.float64).reshape(3)
	prims = [(pts * _FLIP, faces, None, None, np.ones(3))]
	_color, alpha, zbuf = _rasterize(prims, R, t, np.asarray(K, np.float64),
	                                 (W, H), (0, 0), shaded=False)
	return alpha, zbuf
