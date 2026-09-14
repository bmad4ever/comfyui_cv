# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Pure cv2+numpy task cores for the interruptible DNN worker.

Every function here is the compute half of a DNN node in opencv_nodes/dnn.py:
it takes pre-resolved ABSOLUTE model paths and BGR ndarrays (never ComfyUI
tensors, never folder_paths lookups) and returns a PICKLABLE result tuple.
The node's execute() does the schema-side work (path resolution, tensor ->
ndarray conversion, empty-input early returns) and hands the rest to
workers/dnn_worker.py through workerutils.run_subprocess(), so a long forward
pass / generation loop runs in a process ComfyUI can terminate mid-call.

Standalone - no opencv_nodes / ComfyUI imports. cv2.KeyPoint and cv2.DMatch
are NOT picklable in this build, so the feature/matcher tasks serialize them
to (N, 6) / (M, 4) row arrays and the parent rebuilds the objects (the node
outputs still carry real cv2.KeyPoint / cv2.DMatch objects).

THIRD-PARTY: the _MPPalmDet / _MPHandPose classes below are ported from the
OpenCV Zoo "handpose_estimation_mediapipe" reference implementation
(https://github.com/opencv/opencv_zoo, Apache-2.0; upstream models by Google
MediaPipe). They are NOT covered by this file's GPL-3.0 header - see the
License and provenance section of README.md.
"""

import os
import re

import cv2
import numpy as np


def _xyxy_to_box(x1, y1, x2, y2, score, label):
	x1, y1, x2, y2 = (int(round(float(v))) for v in (x1, y1, x2, y2))
	return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
	        "score": round(float(score), 4), "label": label}


def _keypoint_rows(kps):
	"""cv2.KeyPoint list -> (N, 6) rows [x, y, size, angle, response, octave]."""
	return np.array(
		[[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave]
		 for kp in kps], np.float64)


def _dmatch_rows(dms):
	"""cv2.DMatch list -> (M, 4) rows [queryIdx, trainIdx, imgIdx, distance]."""
	return np.array(
		[[dm.queryIdx, dm.trainIdx, dm.imgIdx, dm.distance] for dm in dms],
		np.float64)


# --- MediaPipe hand-pose pipeline --------------------------------------------
# Ported verbatim from opencv_nodes/dnn.py (which mirrors the OpenCV Zoo
# handpose_estimation_mediapipe reference). Only the model caches live here in
# the worker now; the math is unchanged.


def _mp_palm_anchors() -> np.ndarray:
	"""The 2016 SSD anchors of the MediaPipe palm detector, generated instead of
	hard-coded: a 24x24 grid with 2 anchors/cell (1152) then a 12x12 grid with 6
	anchors/cell (864). Cell centres are (col+0.5)/grid, (row+0.5)/grid, repeated
	per anchor. Verified bit-for-bit against the reference _load_anchors() array."""
	anchors = []
	for grid, rep in ((24, 2), (12, 6)):
		for row in range(grid):
			cy = (row + 0.5) / grid
			for col in range(grid):
				cx = (col + 0.5) / grid
				for _ in range(rep):
					anchors.append((cx, cy))
	return np.array(anchors, np.float32)


_PALM_ANCHORS = _mp_palm_anchors()


# --- OpenCV Zoo port: Apache-2.0, https://github.com/opencv/opencv_zoo -------
# _MPPalmDet + _MPHandPose follow the Zoo's handpose_estimation_mediapipe
# mp_palmdet.py / mp_handpose.py closely (method names kept for diffability).
class _MPPalmDet:
	"""cv2.dnn palm detector (192x192 SSD). infer() returns an (N, 19) array,
	each row [x1, y1, x2, y2, 7*(lx, ly), score] - the exact layout the hand-pose
	stage consumes."""

	input_size = np.array([192, 192])  # wh

	def __init__(self, path):
		self.net = cv2.dnn.readNet(path)

	def _preprocess(self, image):
		pad_bias = np.array([0., 0.])  # left, top
		ratio = min(self.input_size / image.shape[:2])
		if image.shape[0] != self.input_size[0] or image.shape[1] != self.input_size[1]:
			ratio_size = (np.array(image.shape[:2]) * ratio).astype(np.int32)
			image = cv2.resize(image, (ratio_size[1], ratio_size[0]))
			pad_h = self.input_size[0] - ratio_size[0]
			pad_w = self.input_size[1] - ratio_size[1]
			pad_bias[0] = left = pad_w // 2
			pad_bias[1] = top = pad_h // 2
			right = pad_w - left
			bottom = pad_h - top
			image = cv2.copyMakeBorder(image, top, bottom, left, right,
			                           cv2.BORDER_CONSTANT, None, (0, 0, 0))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype(np.float32) / 255.0
		pad_bias = (pad_bias / ratio).astype(np.int32)
		return image[np.newaxis, :, :, :], pad_bias

	def infer(self, image, score_threshold, nms_threshold, top_k):
		h, w, _ = image.shape
		input_blob, pad_bias = self._preprocess(image)
		self.net.setInput(input_blob)
		output_blob = self.net.forward(self.net.getUnconnectedOutLayersNames())
		return self._postprocess(output_blob, np.array([w, h]), pad_bias,
		                         score_threshold, nms_threshold, top_k)

	def _postprocess(self, output_blob, original_shape, pad_bias,
	                 score_threshold, nms_threshold, top_k):
		score = output_blob[1][0, :, 0]
		box_delta = output_blob[0][0, :, 0:4]
		landmark_delta = output_blob[0][0, :, 4:]
		scale = max(original_shape)

		score = score.astype(np.float64)
		score = 1 / (1 + np.exp(-score))

		cxy_delta = box_delta[:, :2] / self.input_size
		wh_delta = box_delta[:, 2:] / self.input_size
		xy1 = (cxy_delta - wh_delta / 2 + _PALM_ANCHORS) * scale
		xy2 = (cxy_delta + wh_delta / 2 + _PALM_ANCHORS) * scale
		boxes = np.concatenate([xy1, xy2], axis=1)
		boxes -= [pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]]

		keep_idx = cv2.dnn.NMSBoxes(boxes, score, score_threshold, nms_threshold, top_k=top_k)
		if len(keep_idx) == 0:
			return np.empty(shape=(0, 19), dtype=np.float32)
		selected_score = score[keep_idx]
		selected_box = boxes[keep_idx]

		selected_landmarks = landmark_delta[keep_idx].reshape(-1, 7, 2)
		selected_landmarks = selected_landmarks / self.input_size
		selected_anchors = _PALM_ANCHORS[keep_idx]
		for idx, landmark in enumerate(selected_landmarks):
			landmark += selected_anchors[idx]
		selected_landmarks *= scale
		selected_landmarks -= pad_bias

		return np.c_[selected_box.reshape(-1, 4),
		             selected_landmarks.reshape(-1, 14),
		             selected_score.reshape(-1, 1)].astype(np.float32)


class _MPHandPose:
	"""cv2.dnn hand-pose regressor (224x224). infer(image, palm) returns a
	(132,) result [hand_box(4), screen_lms(63), world_lms(63), handedness, conf]
	for one palm, or None when the model's confidence is below threshold."""

	input_size = np.array([224, 224])  # wh
	PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0
	PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2
	PALM_BOX_PRE_SHIFT_VECTOR = [0, 0]
	PALM_BOX_PRE_ENLARGE_FACTOR = 4
	PALM_BOX_SHIFT_VECTOR = [0, -0.4]
	PALM_BOX_ENLARGE_FACTOR = 3
	HAND_BOX_SHIFT_VECTOR = [0, -0.1]
	HAND_BOX_ENLARGE_FACTOR = 1.65

	def __init__(self, path):
		self.net = cv2.dnn.readNet(path)

	def _cropAndPadFromPalm(self, image, palm_bbox, for_rotation=False):
		wh_palm_bbox = palm_bbox[1] - palm_bbox[0]
		shift_vector = (self.PALM_BOX_PRE_SHIFT_VECTOR if for_rotation
		                else self.PALM_BOX_SHIFT_VECTOR)
		shift_vector = shift_vector * wh_palm_bbox
		palm_bbox = palm_bbox + shift_vector
		center_palm_bbox = np.sum(palm_bbox, axis=0) / 2
		wh_palm_bbox = palm_bbox[1] - palm_bbox[0]
		enlarge_scale = (self.PALM_BOX_PRE_ENLARGE_FACTOR if for_rotation
		                 else self.PALM_BOX_ENLARGE_FACTOR)
		new_half_size = wh_palm_bbox * enlarge_scale / 2
		palm_bbox = np.array([center_palm_bbox - new_half_size,
		                      center_palm_bbox + new_half_size])
		palm_bbox = palm_bbox.astype(np.int32)
		palm_bbox[:, 0] = np.clip(palm_bbox[:, 0], 0, image.shape[1])
		palm_bbox[:, 1] = np.clip(palm_bbox[:, 1], 0, image.shape[0])
		image = image[palm_bbox[0][1]:palm_bbox[1][1], palm_bbox[0][0]:palm_bbox[1][0], :]
		side_len = (np.linalg.norm(image.shape[:2]) if for_rotation
		            else max(image.shape[:2]))
		side_len = int(side_len)
		pad_h = side_len - image.shape[0]
		pad_w = side_len - image.shape[1]
		left = pad_w // 2
		top = pad_h // 2
		right = pad_w - left
		bottom = pad_h - top
		image = cv2.copyMakeBorder(image, top, bottom, left, right,
		                           cv2.BORDER_CONSTANT, None, (0, 0, 0))
		bias = palm_bbox[0] - [left, top]
		return image, palm_bbox, bias

	def _preprocess(self, image, palm):
		pad_bias = np.array([0, 0], dtype=np.int32)  # left, top
		palm_bbox = palm[0:4].reshape(2, 2)
		image, palm_bbox, bias = self._cropAndPadFromPalm(image, palm_bbox, True)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		pad_bias += bias

		palm_bbox -= pad_bias
		palm_landmarks = palm[4:18].reshape(7, 2) - pad_bias
		p1 = palm_landmarks[self.PALM_LANDMARKS_INDEX_OF_PALM_BASE]
		p2 = palm_landmarks[self.PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE]
		radians = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
		radians = radians - 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
		angle = np.rad2deg(radians)
		center_palm_bbox = np.sum(palm_bbox, axis=0) / 2
		rotation_matrix = cv2.getRotationMatrix2D(center_palm_bbox, angle, 1.0)
		rotated_image = cv2.warpAffine(image, rotation_matrix,
		                               (image.shape[1], image.shape[0]))
		homogeneous_coord = np.c_[palm_landmarks, np.ones(palm_landmarks.shape[0])]
		rotated_palm_landmarks = np.array([
			np.dot(homogeneous_coord, rotation_matrix[0]),
			np.dot(homogeneous_coord, rotation_matrix[1])])
		rotated_palm_bbox = np.array([
			np.amin(rotated_palm_landmarks, axis=1),
			np.amax(rotated_palm_landmarks, axis=1)])

		crop, rotated_palm_bbox, _ = self._cropAndPadFromPalm(rotated_image, rotated_palm_bbox)
		blob = cv2.resize(crop, dsize=self.input_size,
		                  interpolation=cv2.INTER_AREA).astype(np.float32)
		blob = blob / 255.
		return blob[np.newaxis, :, :, :], rotated_palm_bbox, angle, rotation_matrix, pad_bias

	def infer(self, image, palm, conf_threshold):
		input_blob, rotated_palm_bbox, angle, rotation_matrix, pad_bias = \
			self._preprocess(image, palm)
		self.net.setInput(input_blob)
		output_blob = self.net.forward(self.net.getUnconnectedOutLayersNames())
		return self._postprocess(output_blob, rotated_palm_bbox, angle,
		                         rotation_matrix, pad_bias, conf_threshold)

	def _postprocess(self, blob, rotated_palm_bbox, angle, rotation_matrix,
	                 pad_bias, conf_threshold):
		landmarks, conf, handedness, landmarks_word = blob
		conf = conf[0][0]
		if conf < conf_threshold:
			return None

		landmarks = landmarks[0].reshape(-1, 3)          # (1, 63) -> (21, 3)
		landmarks_word = landmarks_word[0].reshape(-1, 3)

		wh_rotated_palm_bbox = rotated_palm_bbox[1] - rotated_palm_bbox[0]
		scale_factor = wh_rotated_palm_bbox / self.input_size
		landmarks[:, :2] = (landmarks[:, :2] - self.input_size / 2) * max(scale_factor)
		landmarks[:, 2] = landmarks[:, 2] * max(scale_factor)
		coords_rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
		rotated_landmarks = np.dot(landmarks[:, :2], coords_rotation_matrix[:, :2])
		rotated_landmarks = np.c_[rotated_landmarks, landmarks[:, 2]]
		rotated_landmarks_world = np.dot(landmarks_word[:, :2], coords_rotation_matrix[:, :2])
		rotated_landmarks_world = np.c_[rotated_landmarks_world, landmarks_word[:, 2]]

		rotation_component = np.array([
			[rotation_matrix[0][0], rotation_matrix[1][0]],
			[rotation_matrix[0][1], rotation_matrix[1][1]]])
		translation_component = np.array([rotation_matrix[0][2], rotation_matrix[1][2]])
		inverted_translation = np.array([
			-np.dot(rotation_component[0], translation_component),
			-np.dot(rotation_component[1], translation_component)])
		inverse_rotation_matrix = np.c_[rotation_component, inverted_translation]
		center = np.append(np.sum(rotated_palm_bbox, axis=0) / 2, 1)
		original_center = np.array([
			np.dot(center, inverse_rotation_matrix[0]),
			np.dot(center, inverse_rotation_matrix[1])])
		landmarks[:, :2] = rotated_landmarks[:, :2] + original_center + pad_bias

		bbox = np.array([np.amin(landmarks[:, :2], axis=0),
		                 np.amax(landmarks[:, :2], axis=0)])
		wh_bbox = bbox[1] - bbox[0]
		shift_vector = self.HAND_BOX_SHIFT_VECTOR * wh_bbox
		bbox = bbox + shift_vector
		center_bbox = np.sum(bbox, axis=0) / 2
		wh_bbox = bbox[1] - bbox[0]
		new_half_size = wh_bbox * self.HAND_BOX_ENLARGE_FACTOR / 2
		bbox = np.array([center_bbox - new_half_size, center_bbox + new_half_size])

		return np.r_[bbox.reshape(-1), landmarks.reshape(-1),
		             rotated_landmarks_world.reshape(-1), handedness[0][0], conf]


# The 21 MediaPipe hand keypoints, in model output order (see the TF.js
# hand-pose-detection README). Emitted per hand so the landmark rows are
# self-describing and can drive a text overlay ('OpenCV Draw Labels').
MEDIAPIPE_HAND_NAMES = [
	"wrist",
	"thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
	"index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
	"middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
	"ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
	"pinky_finger_mcp", "pinky_finger_pip", "pinky_finger_dip", "pinky_finger_tip",
]

# The official MediaPipe HAND_CONNECTIONS: the 21 edges of the hand skeleton
# (each finger chained from its base, the palm closed wrist->pinky_mcp and along
# the knuckles). Index pairs into the 21 keypoints; the node offsets them per
# hand so they index the concatenated landmark array directly.
MEDIAPIPE_HAND_CONNECTIONS = np.array([
	(0, 1), (1, 2), (2, 3), (3, 4),           # thumb
	(0, 5), (5, 6), (6, 7), (7, 8),           # index
	(5, 9), (9, 10), (10, 11), (11, 12),      # middle
	(9, 13), (13, 14), (14, 15), (15, 16),    # ring
	(13, 17), (17, 18), (18, 19), (19, 20),   # pinky
	(0, 17),                                  # palm base -> pinky mcp
], np.int32)


# --- shared cv2.dnn plumbing -------------------------------------------------


def _apply_backend_target(net, backend_val, target_val):
	"""Set the net's preferable backend/target from RESOLVED values (None =
	skip). An unsupported combo must not halt the task - falls back to the
	default, exactly like the in-process path did."""
	try:
		if backend_val is not None:
			net.setPreferableBackend(int(backend_val))
		if target_val is not None:
			net.setPreferableTarget(int(target_val))
	except cv2.error:
		pass


def _read_net_dispatch(path, engine):
	""".tflite goes to readNetFromTFLite (no engine arg), everything else to
	readNetFromONNX with the pre-resolved engine value (None = default)."""
	if str(path).lower().endswith(".tflite"):
		return cv2.dnn.readNetFromTFLite(path)
	if engine is None:
		return cv2.dnn.readNetFromONNX(path)
	return cv2.dnn.readNetFromONNX(path, engine=engine)


# --- LightGlue-family preprocessing -----------------------------------------
# The extractors in the LightGlue family (DISK, SuperPoint) are NOT
# interchangeable in how they want their input, and cv2.dnn will not tell you
# when you get it wrong: measured on OpenCV 5.0, feeding DISK's 3-channel input
# a 1-channel blob runs without error and returns plausible-looking keypoints
# with useless descriptors. So the preprocessing is read off the MODEL, and the
# numbers come from lightglue's own `preprocess_conf` / `DISK.forward`
# (github.com/cvg/LightGlue): scale to 0..1, RGB or gray per the model, pad up
# to a multiple of 16, resize the long side to 1024 by default.

_EXTRACTOR_PAD_MULTIPLE = 16      # DISK's U-Net downsamples 4x (kornia's
                                  # pad_if_not_divisible=True); SuperPoint
                                  # needs 8, which a multiple of 16 satisfies.
_LIGHTGLUE_RESIZE = 1024          # lightglue ImagePreprocessor/preprocess_conf

_RE_NET_INPUT = re.compile(r"<Input>\s+\S+\s+\[([^\]]+)\]")
_net_channels_cache: dict = {}


def _net_input_channels(net, key):
	"""Channel count the net's first input DECLARES, or None if it is symbolic.

	net.dump() is the only route that survives the 5.x graph engine: Layer.blobs
	is empty (weights became Const layers) and Net.setInputShape /
	Net.getLayerShapes have no usable Python binding here ("Expected
	cv::MatShape for argument 'shape'"). The dump names it outright -
	`"image" // <Input> CV_32FC1 [1 x 3 x width x num_keypoints]`."""
	if key in _net_channels_cache:
		return _net_channels_cache[key]
	channels = None
	try:
		match = _RE_NET_INPUT.search(net.dump())
	except cv2.error:
		match = None
	if match:
		dims = [d.strip() for d in match.group(1).split(" x ")]
		if len(dims) == 4 and dims[1].isdigit():
			channels = int(dims[1])
	_net_channels_cache[key] = channels
	return channels


def _preprocess_extractor_frame(frame, channels, resize_long_side):
	"""(blob, scale) for a LightGlue-family extractor.

	`scale` is the factor the frame was resized by, so keypoints can be divided
	back into the INPUT frame's own pixels. The padding needs no such undo: it
	only ever extends right/bottom, leaving the origin - and therefore every
	keypoint coordinate - where it was.

	`channels` None (a model whose input channel count is symbolic) is treated
	as colour, the LightGlue family's majority case."""
	scale = 1.0
	if resize_long_side > 0:
		long_side = max(frame.shape[:2])
		if long_side and long_side != resize_long_side:
			scale = float(resize_long_side) / float(long_side)
			frame = cv2.resize(
				frame,
				(max(1, int(round(frame.shape[1] * scale))),
				 max(1, int(round(frame.shape[0] * scale)))),
				interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

	h, w = frame.shape[:2]
	mult = _EXTRACTOR_PAD_MULTIPLE
	pad_h = (h + mult - 1) // mult * mult
	pad_w = (w + mult - 1) // mult * mult
	if (pad_h, pad_w) != (h, w):
		frame = cv2.copyMakeBorder(frame, 0, pad_h - h, 0, pad_w - w,
		                           cv2.BORDER_CONSTANT, value=0)

	if channels == 1:
		src = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3
		       else frame)
		swap = False
	else:
		src = (frame if frame.ndim == 3
		       else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
		swap = True                    # the frame is BGR, the model wants RGB
	blob = cv2.dnn.blobFromImage(src, 1.0 / 255.0, (pad_w, pad_h), None,
	                             swapRB=swap, crop=False)
	return blob, scale


def _normalize_keypoints(kpts, size):
	"""LightGlue's own `normalize_keypoints` (lightglue/lightglue.py).

	`size` is the (width, height) of the image the keypoints came from, or None
	to fall back to their own extent - which is upstream's own fallback for a
	missing `image_size`, and what the matcher node uses when its optional
	`space_*` inputs are unconnected.

	This has to happen HERE because the exported ONNX does not do it: measured
	on box.png/box_in_scene.png with DISK, raw pixel keypoints yield 6 matches
	where normalized ones yield 266."""
	kpts = np.asarray(kpts, np.float32).reshape(-1, 2)
	if not len(kpts):
		return kpts
	if size is None:
		size = 1.0 + kpts.max(axis=0) - kpts.min(axis=0)
	size = np.asarray(size, np.float32).ravel()[:2]
	span = float(size.max())
	if span <= 0:
		return kpts
	return (kpts - size / 2.0) / (span / 2.0)


def _parse_extractor_outputs(outs, max_kp):
	"""Parse ONNX feature extractor outputs into (keypoints, descriptors).

	Supports three formats:
	  3-output: (1,N,2) kpts + (1,N) scores + (1,N,D) descs  (DISK, SuperPoint)
	  2-output flat:  (1,N,4) kpts [x,y,angle,response] + (1,N,D) descs
	  2-output dense: (1,C,H,W) score map + (1,C,H,W) descriptor map

	`max_kp` keeps the STRONGEST keypoints, not the first ones the model
	happened to emit: DISK's ONNX export returns them in detection order, not
	by score (measured on box.png - scores run 9.2, 10.9, 29.4, 19.8, ...), so
	truncating a raw prefix threw away the best features at random.
	"""
	kps = []
	descs = np.empty((0, 0), np.float32)

	if len(outs) == 3:
		kp_raw = np.asarray(outs[0])[0]
		scores_raw = np.asarray(outs[1])[0].ravel()
		descs_raw = np.asarray(outs[2])[0].astype(np.float32)
		n = min(len(kp_raw), len(scores_raw), len(descs_raw))
		keep = np.arange(n)
		if n > max_kp:
			keep = np.argsort(scores_raw[:n])[::-1][:max_kp]
			keep.sort()                # detection order within the kept set
		for i in keep:
			kps.append(cv2.KeyPoint(
				float(kp_raw[i, 0]), float(kp_raw[i, 1]),
				1.0, 0.0, float(scores_raw[i]), 0))
		descs = descs_raw[keep].astype(np.float32)
	elif len(outs) == 2:
		o0 = np.asarray(outs[0], np.float32)
		o1 = np.asarray(outs[1], np.float32)
		if o0.ndim == 3 and o1.ndim == 3:
			# Flat: (1,N,4) + (1,N,D)
			kp_raw = o0[0]
			descs = o1[0].astype(np.float32)
			n = min(len(kp_raw), len(descs))
			keep = np.arange(n)
			if n > max_kp:
				response = (kp_raw[:n, 3] if kp_raw.shape[1] > 3
				            else np.zeros(n, np.float32))
				keep = np.argsort(response)[::-1][:max_kp]
				keep.sort()
			for i in keep:
				kps.append(cv2.KeyPoint(
					float(kp_raw[i, 0]), float(kp_raw[i, 1]),
					1.0,
					float(kp_raw[i, 2]) if kp_raw.shape[1] > 2 else 0.0,
					float(kp_raw[i, 3]) if kp_raw.shape[1] > 3 else 1.0,
					0))
			descs = descs[keep]
		elif o0.ndim == 4 and o1.ndim == 4:
			# Dense: score map (1,C,H,W) + desc map (1,C,H,W)
			score_map = o0[0]
			desc_map = o1[0]
			if score_map.shape[0] > 1:
				score_map = score_map.max(axis=0)
			else:
				score_map = score_map[0]
			C, H, W = score_map.shape if score_map.ndim == 3 else (1, *score_map.shape)
			flat = score_map.ravel()
			k = min(max_kp, len(flat))
			top_idx = np.argpartition(flat, -k)[-k:]
			top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
			pts_descs = []
			for idx in top_idx:
				v = float(flat[idx])
				if v <= 0:
					continue
				sy = int(idx) // W
				sx = int(idx) % W
				kps.append(cv2.KeyPoint(float(sx), float(sy), 1.0, 0.0, v, 0))
				feat_xy = desc_map[:, sy, sx] if desc_map.ndim == 3 else desc_map[sy, sx]
				pts_descs.append(feat_xy.astype(np.float32))
			if pts_descs:
				descs = np.stack(pts_descs)
	return kps, descs


# --- LLM helpers (ported verbatim; only the loaders take pre-resolved paths) --


def _last_token(logits):
	"""Greedy next-token pick from a (1, seq, vocab) (or (1, vocab)) logits."""
	arr = np.asarray(logits)
	row = arr[:, -1, :] if arr.ndim == 3 else arr.reshape(arr.shape[0], -1)
	return int(np.argmax(row.reshape(-1)))


def _set_named(net, arr, name):
	"""net.setInput(arr, name) that silently skips inputs the graph does not
	have (exports differ: with/without position_ids etc.)."""
	try:
		net.setInput(np.ascontiguousarray(arr), name)
	except cv2.error:
		pass


def _greedy_decoder_only(net, prompt_ids, max_new, stops, use_kv):
	"""Autoregressive greedy loop over a decoder-only LM (input_ids +
	attention_mask [+ position_ids]). KV-cache mode needs a causal-lm-with-past
	(merged) export: OpenCV routes present.* -> past_key_values.* internally.
	Returns the generated ids (stop ids included, as emitted)."""
	ids = list(prompt_ids)
	generated = []

	def full_pass(ids2d):
		_set_named(net, ids2d, "input_ids")
		_set_named(net, np.ones((1, ids2d.shape[1]), np.int64), "attention_mask")
		_set_named(net, np.arange(ids2d.shape[1], dtype=np.int64).reshape(1, -1),
		           "position_ids")
		return net.forward()

	if use_kv:
		net.enableKVCache()
		net.resetKVCache()
		# Merged transformers.js exports branch on a scalar bool input
		# ('use_cache_branch'). It MUST be fed something or the If layer
		# asserts inp->total() == 1u. Per the export's contract it is False
		# for the prompt pass (there is no past yet) and True from the first
		# single-token step on, once OpenCV's KV routing has filled the
		# past_key_values.* inputs.
		# UNVERIFIED on this build: every merged decoder we have returns
		# all-NaN logits here whatever this flag is set to (see
		# opencv5_model_compatibility.md), so this follows the spec rather
		# than a passing test.
		_set_named(net, np.array([False]), "use_cache_branch")
		input_ids = np.array([ids], np.int64)
		new_id = _last_token(full_pass(input_ids))
		generated.append(new_id)
		for _ in range(int(max_new) - 1):
			if new_id in stops:
				break
			cur = input_ids.shape[1] + len(generated)
			_set_named(net, np.array([True]), "use_cache_branch")
			_set_named(net, np.array([[new_id]], np.int64), "input_ids")
			_set_named(net, np.ones((1, cur), np.int64), "attention_mask")
			_set_named(net, np.array([[cur - 1]], np.int64), "position_ids")
			new_id = _last_token(net.forward())
			generated.append(new_id)
	else:
		net.disableKVCache()
		# Full-sequence mode re-feeds the whole prompt every step, so a merged
		# export must stay on the no-cache branch. Without this the If layer
		# aborts with 'inp->total() == 1u' - full sequence mode was simply
		# unusable with any merged decoder.
		_set_named(net, np.array([False]), "use_cache_branch")
		input_ids = np.array([ids], np.int64)
		for _ in range(int(max_new)):
			new_id = _last_token(full_pass(input_ids))
			if new_id in stops:
				break
			generated.append(new_id)
			input_ids = np.concatenate(
				[input_ids, np.array([[new_id]], np.int64)], axis=1)
	return generated


_llm_net_cache: dict = {}
_tokenizer_cache: dict = {}


def _load_gen_net(path, engine):
	"""readNetFromONNX for a pre-resolved model path, cached per (path, engine)
	WITHIN one worker run."""
	key = (path, engine)
	net = _llm_net_cache.get(key)
	if net is None:
		try:
			net = (cv2.dnn.readNetFromONNX(path) if engine is None
			       else cv2.dnn.readNetFromONNX(path, engine=engine))
		except cv2.error as exc:
			raise ValueError(f"cv2.dnn could not load '{os.path.basename(path)}': {exc}")
		_llm_net_cache[key] = net
	return net


def _load_tokenizer(tokenizer_dir):
	"""cv2.dnn.Tokenizer.load for a model folder (config.json + tokenizer.json
	inside it), cached per directory within one worker run. Accepts the folder
	or the config.json path directly."""
	d = os.path.normpath(str(tokenizer_dir).strip())
	cfg = d if d.lower().endswith(".json") else os.path.join(d, "config.json")
	if not os.path.isfile(cfg):
		raise ValueError(f"Tokenizer folder '{tokenizer_dir}' has no config.json.")
	tok = _tokenizer_cache.get(cfg)
	if tok is None:
		try:
			tok = cv2.dnn.Tokenizer_load(cfg)
		except cv2.error as exc:
			raise ValueError(f"cv2.dnn.Tokenizer could not load '{cfg}': {exc}")
		_tokenizer_cache[cfg] = tok
	return tok


# --- task cores ---------------------------------------------------------------
# Each returns a PICKLABLE tuple. Keys are the worker dispatch ids used by
# opencv_nodes/dnn.py.

_yunet_cache: dict = {}
_sface_cache: dict = {}
_palm_cache: dict = {}
_hand_cache: dict = {}
_wechat_cache: dict = {}
_colorize_cache: dict = {}
_east_cache: dict = {}
_forward_cache: dict = {}
_forward_all_cache: dict = {}
_feature_extractor_cache: dict = {}
_match_model_cache: dict = {}


def _task_yunet_detect(model_path, frames, conf, nms, top_k):
	"""(bboxes, landmarks, scores, face_count) - bboxes flat for a single
	frame, nested per frame for a batch."""
	key = (model_path, round(conf, 6), round(nms, 6), int(top_k))
	det = _yunet_cache.get(key)
	if det is None:
		det = cv2.FaceDetectorYN.create(
			model=model_path, config="", input_size=(320, 320),
			score_threshold=float(conf), nms_threshold=float(nms), top_k=int(top_k))
		_yunet_cache[key] = det

	per_frame_boxes = []
	landmarks_all = []
	scores_all = []
	for frame in frames:
		h, w = frame.shape[:2]
		det.setInputSize((w, h))
		faces = det.detect(frame)[1]
		boxes = []
		if faces is not None and len(faces):
			faces = np.asarray(faces, np.float32)
			for f in faces:
				x, y, bw, bh = f[0:4]
				boxes.append({
					"x": int(round(x)), "y": int(round(y)),
					"width": int(round(bw)), "height": int(round(bh)),
					"score": round(float(f[14]), 4), "label": "face",
				})
			landmarks_all.append(faces[:, 4:14].reshape(-1, 2))
			scores_all.append(faces[:, 14])
		per_frame_boxes.append(boxes)

	landmarks = (np.concatenate(landmarks_all, axis=0).astype(np.float32)
	             if landmarks_all else np.zeros((0, 2), np.float32))
	scores = (np.concatenate(scores_all, axis=0).astype(np.float32)
	          if scores_all else np.zeros((0,), np.float32))
	face_count = int(sum(len(b) for b in per_frame_boxes))
	bboxes = per_frame_boxes[0] if len(per_frame_boxes) == 1 else per_frame_boxes
	return bboxes, landmarks, scores, face_count


def _task_sface_embed(model_path, frame, landmarks):
	"""((N, 128) embeddings, [aligned 112x112 BGR crops]) for SFace.

	cv2's alignCrop only reads columns 4..13 of a YuNet face row (the five
	landmarks) - the bbox and score are ignored - so any 5-point landmark set
	drives it, which is what keeps the node generic.
	"""
	rec = _sface_cache.get(model_path)
	if rec is None:
		rec = cv2.FaceRecognizerSF.create(model_path, "")
		_sface_cache[model_path] = rec

	pts = np.asarray(landmarks, np.float32).reshape(-1, 5, 2)
	feats, crops = [], []
	for p in pts:
		row = np.zeros((15,), np.float32)
		row[4:14] = p.reshape(-1)
		crop = np.ascontiguousarray(rec.alignCrop(frame, row))
		crops.append(crop)
		feats.append(np.asarray(rec.feature(crop), np.float32).reshape(-1))
	emb = (np.stack(feats, axis=0).astype(np.float32) if feats
	       else np.zeros((0, 128), np.float32))
	return emb, crops


def _task_mp_palm_detect(model_path, frame, score_threshold, nms_threshold, top_k):
	"""((N, 19) palms,) - the parent rebuilds boxes/points/scores from it."""
	det = _palm_cache.get(model_path)
	if det is None:
		det = _MPPalmDet(model_path)
		_palm_cache[model_path] = det
	palms = det.infer(frame, float(score_threshold), float(nms_threshold), int(top_k))
	return np.asarray(palms, np.float32).reshape(-1, 19),


def _task_mp_hand_pose(model_path, frame, palms, conf_threshold):
	"""(bboxes, landmarks, world, handedness, scores, count, connections, names)."""
	empty = (np.zeros((0, 2), np.float32), np.zeros((0, 3), np.float32),
	         np.zeros((0,), np.float32), np.zeros((0,), np.float32), 0,
	         np.zeros((0, 2), np.int32), np.array([], dtype="<U20"))
	det = _hand_cache.get(model_path)
	if det is None:
		det = _MPHandPose(model_path)
		_hand_cache[model_path] = det
	boxes, lms, world, handed, scores = [], [], [], [], []
	for palm in palms:
		# infer() mutates its image crop internally; the frame itself is untouched
		result = det.infer(frame, palm.copy(), float(conf_threshold))
		if result is None:
			continue
		boxes.append(_xyxy_to_box(result[0], result[1], result[2], result[3],
		                          result[131], "hand"))
		lms.append(result[4:67].reshape(21, 3)[:, :2])
		world.append(result[67:130].reshape(21, 3))
		handed.append(result[130])
		scores.append(result[131])

	if not boxes:
		return [], *empty
	landmarks = np.concatenate(lms, axis=0).astype(np.float32)
	world_lms = np.concatenate(world, axis=0).astype(np.float32)
	handedness = np.asarray(handed, np.float32)
	conf = np.asarray(scores, np.float32)
	connections = np.concatenate(
		[MEDIAPIPE_HAND_CONNECTIONS + 21 * h for h in range(len(boxes))],
		axis=0).astype(np.int32)
	names = np.array(MEDIAPIPE_HAND_NAMES * len(boxes), dtype="<U20")
	return boxes, landmarks, world_lms, handedness, conf, len(boxes), connections, names


def _task_wechat_qr(detect_path, sr_path, frame):
	"""(bboxes, corners, connections, texts, labels, count)."""
	empty_str = np.array([], dtype="<U1")
	key = (detect_path, sr_path)
	det = _wechat_cache.get(key)
	if det is None:
		if not hasattr(cv2, "wechat_qrcode") or not hasattr(
				cv2.wechat_qrcode, "WeChatQRCode"):
			raise RuntimeError(
				"cv2.wechat_qrcode.WeChatQRCode is unavailable - the loaded "
				"OpenCV build lacks the wechat_qrcode contrib module. Stop the "
				"ComfyUI server and reinstall opencv-contrib-python(-headless) so it owns "
				"the shared cv2.pyd (see CLAUDE.md).")
		det = cv2.wechat_qrcode.WeChatQRCode(detect_path, sr_path)
		_wechat_cache[key] = det

	texts, points = det.detectAndDecode(frame)
	if not len(texts):
		return [], np.zeros((0, 2), np.float32), np.zeros((0, 2), np.int32), \
		       empty_str, empty_str, 0

	corner_arrays, boxes, labels, conns = [], [], [], []
	offset = 0
	for text, quad in zip(texts, points):
		quad = np.asarray(quad, np.float32).reshape(-1, 2)
		n = len(quad)
		corner_arrays.append(quad)
		x1, y1 = quad.min(axis=0)
		x2, y2 = quad.max(axis=0)
		boxes.append(_xyxy_to_box(x1, y1, x2, y2, 1.0, text or "qr"))
		# closed-quad edges, offset into the concatenated corner rows
		edges = [(k, (k + 1) % n) for k in range(n)]
		conns.append(np.array(edges, np.int32) + offset)
		# decoded text on the first corner only, blanks on the rest
		labels.extend([text] + [""] * (n - 1))
		offset += n

	corners = np.concatenate(corner_arrays, axis=0).astype(np.float32)
	connections = np.concatenate(conns, axis=0).astype(np.int32)
	return boxes, corners, connections, np.array(list(texts)), np.array(labels), \
	       len(corner_arrays)


def _colorize(net, frame, strength):
	"""Colorize one BGR uint8 frame -> BGR uint8 frame (same resolution)."""
	h, w = frame.shape[:2]
	# Convert to grayscale (colour images are converted first)
	if frame.ndim == 3 and frame.shape[2] >= 3:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	else:
		gray = frame[..., 0] if frame.ndim == 3 else frame
	# Map 0-255 -> 0-100 (Lab L range)
	img_l = gray.astype(np.float32) / 255.0 * 100.0

	# Resize to 224x224 and mean-center (model training convention)
	img_l_rs = cv2.resize(img_l, (224, 224))
	img_l_rs -= 50.0

	# Build blob and run forward pass
	blob = cv2.dnn.blobFromImage(img_l_rs)  # (1, 1, 224, 224)
	net.setInput(blob)
	ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))  # (224, 224, 2)

	# Scale ab channels by strength
	if strength != 1.0:
		ab_dec *= float(strength)

	# Upsample ab to original resolution
	ab_dec_us = cv2.resize(ab_dec, (w, h))

	# Recombine L + predicted ab -> Lab -> BGR
	img_lab_out = np.concatenate(
		(img_l[:, :, np.newaxis], ab_dec_us), axis=2)
	img_bgr_out = np.clip(
		cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)
	return (img_bgr_out * 255).astype(np.uint8)


def _task_colorize(model_path, frames, strength):
	"""(colored_frames,) - list of BGR uint8, one per input frame."""
	net = _colorize_cache.get(model_path)
	if net is None:
		net = cv2.dnn.readNetFromONNX(model_path)
		_colorize_cache[model_path] = net
	return [np.ascontiguousarray(_colorize(net, frame, float(strength)))
	        for frame in frames],


def _task_east_detect(model_path, frames, conf, nms, size):
	"""(bboxes, quads, scores, det_count) - bboxes flat for a single frame,
	nested per frame for a batch."""
	key = (model_path, float(conf), float(nms), int(size))
	net = _east_cache.get(key)
	if net is None:
		net = cv2.dnn.TextDetectionModel_EAST(model_path)
		net.setConfidenceThreshold(float(conf)).setNMSThreshold(float(nms))
		net.setInputParams(1.0, (int(size), int(size)),
		                   (123.68, 116.78, 103.94), True, False)
		_east_cache[key] = net

	all_bboxes, all_quads, all_scores = [], [], []
	for frame in frames:
		h, w = frame.shape[:2]
		boxes, confidences = net.detectTextRectangles(frame)
		bboxes, quads, scores = [], [], []
		if boxes:
			for box, confv in zip(boxes, confidences):
				corners = cv2.boxPoints(box).astype(np.float32)
				xs, ys = corners[:, 0], corners[:, 1]
				x1, y1 = max(0, int(xs.min())), max(0, int(ys.min()))
				x2, y2 = min(w, int(xs.max())), min(h, int(ys.max()))
				if x2 <= x1 or y2 <= y1:
					continue
				bboxes.append(_xyxy_to_box(x1, y1, x2, y2, confv, "text"))
				quads.append(corners)
				scores.append(float(confv))
		scores_arr = (np.asarray(scores, np.float32) if scores
		              else np.empty(0, np.float32))
		quads_arr = (np.stack(quads, axis=0) if quads
		             else np.empty((0, 4, 2), np.float32))
		all_bboxes.append(bboxes)
		all_quads.append(quads_arr)
		all_scores.append(scores_arr)

	bboxes = all_bboxes[0] if len(all_bboxes) == 1 else all_bboxes
	scores = (np.concatenate(all_scores) if all_scores
	          else np.empty(0, np.float32))
	quads = (np.concatenate(all_quads, axis=0) if all_quads
	         else np.empty((0, 4, 2), np.float32))
	return bboxes, quads, scores, int(sum(len(b) for b in all_bboxes))


def _task_dnn_forward(model_path, engine, backend_val, target_val, blob, name):
	"""(output,) - one forward pass over the (N, C, H, W) blob."""
	key = (model_path, engine)
	net = _forward_cache.get(key)
	if net is None:
		net = _read_net_dispatch(model_path, engine)
		_forward_cache[key] = net
	_apply_backend_target(net, backend_val, target_val)

	arr = np.ascontiguousarray(np.asarray(blob, np.float32))
	if arr.size == 0 or (arr.ndim >= 1 and arr.shape[0] == 0):
		return arr,
	net.setInput(arr)
	out = net.forward(name) if name else net.forward()
	return np.asarray(out),


def _task_dnn_forward_all(model_path, engine, backend_val, target_val, blob,
                          output_layers, input_names=()):
	"""(outputs, output_names, count) - every fetched unconnected layer.

	input_names drives a MULTI-INPUT net: the blob's batch dimension is split
	into one equal group per name and each group goes to its own setInput().
	Measured on opencv-zoo RAFT: splitting one blobFromImages([a, b]) this way is
	bit-identical to two separate blobFromImage + setInput calls."""
	key = (model_path, engine)
	net = _forward_all_cache.get(key)
	if net is None:
		net = _read_net_dispatch(model_path, engine)
		_forward_all_cache[key] = net
	_apply_backend_target(net, backend_val, target_val)

	arr = np.ascontiguousarray(np.asarray(blob, np.float32))
	if arr.size == 0 or (arr.ndim >= 1 and arr.shape[0] == 0):
		return [], "", 0
	names = [s.strip() for s in str(output_layers).split(",") if s.strip()]
	if not names:
		names = list(net.getUnconnectedOutLayersNames())
	in_names = [str(s) for s in (input_names or [])]
	if len(in_names) > 1:
		for group, iname in zip(np.split(arr, len(in_names), axis=0), in_names):
			net.setInput(np.ascontiguousarray(group), iname)
	elif in_names:
		net.setInput(arr, in_names[0])
	else:
		net.setInput(arr)
	outs = net.forward(names)
	outs = [np.asarray(o) for o in outs]
	return outs, "\n".join(names), len(outs)


def _task_feature_extract(model_path, engine, frame, max_keypoints,
                          resize_long_side=_LIGHTGLUE_RESIZE):
	"""(kp_rows, descriptors, count) - kp_rows is a (N, 6) serialization of the
	cv2.KeyPoint list; the parent rebuilds the objects.

	Keypoints come back in the INPUT frame's pixels whatever resolution the
	model was actually fed, so everything downstream (matching, homography,
	Draw Matches) stays in the coordinates the user wired in."""
	key = (model_path, engine)
	net = _feature_extractor_cache.get(key)
	if net is None:
		net = _read_net_dispatch(model_path, engine)
		_feature_extractor_cache[key] = net

	blob, scale = _preprocess_extractor_frame(
		frame, _net_input_channels(net, key), int(resize_long_side))
	net.setInput(blob)
	outs = net.forward(net.getUnconnectedOutLayersNames())
	kps, descs = _parse_extractor_outputs(outs, int(max_keypoints))
	rows = _keypoint_rows(kps)
	if len(rows) and scale != 1.0:
		rows[:, :2] /= scale
	return rows, descs, len(kps)


def _task_match_features(model_path, engine, kpts_a_rows, kpts_b_rows, da, db,
                         size_a=None, size_b=None):
	"""(dmatch_rows, points_a, points_b, scores, match_count) - dmatch_rows is a
	(M, 4) serialization of the cv2.DMatch list; the parent rebuilds it.

	`size_a`/`size_b` are the (width, height) of the images the two keypoint
	sets were detected in, for LightGlue's keypoint normalization; None falls
	back to the keypoints' own extent (upstream's own fallback)."""
	key = (model_path, engine)
	net = _match_model_cache.get(key)
	if net is None:
		net = _read_net_dispatch(model_path, engine)
		_match_model_cache[key] = net

	kps_a = [cv2.KeyPoint(float(r[0]), float(r[1]), float(r[2]),
	                      float(r[3]), float(r[4]), int(r[5])) for r in kpts_a_rows]
	kps_b = [cv2.KeyPoint(float(r[0]), float(r[1]), float(r[2]),
	                      float(r[3]), float(r[4]), int(r[5])) for r in kpts_b_rows]
	pts_a = np.array([[kp.pt[0], kp.pt[1]] for kp in kps_a], np.float32)
	pts_b = np.array([[kp.pt[0], kp.pt[1]] for kp in kps_b], np.float32)
	kpts1 = _normalize_keypoints(pts_a, size_a).reshape(1, -1, 2)
	kpts2 = _normalize_keypoints(pts_b, size_b).reshape(1, -1, 2)
	d1 = da.reshape(1, -1, da.shape[-1]).astype(np.float32)
	d2 = db.reshape(1, -1, db.shape[-1]).astype(np.float32)

	def _forward(n):
		names = n.getUnconnectedOutLayersNames()
		n.setInput(kpts1, "kpts0")
		n.setInput(kpts2, "kpts1")
		n.setInput(d1, "desc0")
		n.setInput(d2, "desc1")
		return names, n.forward(names)

	try:
		names, outs = _forward(net)
	except cv2.error:
		# cv2 5.0 freezes a net's input shapes on its first forward: a second
		# pair with a different keypoint count dies in einsum_layer.cpp
		# ("Einsum operands can not be broadcasted"), and setInputShape has no
		# usable Python binding to reshape it. Reload and retry once (~8 s for
		# disk_lightglue.onnx), paid only when the count actually changes.
		# dnn_worker.py spawns a fresh process per node call, so the shipped
		# path forwards once per net and never reaches this; it is here for any
		# in-process caller that reuses the cache across pairs.
		net = _read_net_dispatch(model_path, engine)
		_match_model_cache[key] = net
		names, outs = _forward(net)

	# By NAME, not position: getUnconnectedOutLayersNames() makes no ordering
	# promise, and matches0/mscores0 are two of four similar-looking outputs.
	by_name = dict(zip(names, outs))
	m0 = np.asarray(by_name["matches0"] if "matches0" in by_name else outs[0],
	                np.int64).ravel()
	s0 = np.asarray(by_name["mscores0"] if "mscores0" in by_name
	                else outs[min(2, len(outs) - 1)], np.float32).ravel()
	m0 = m0[:len(kps_a)]

	if len(s0) < len(m0):              # a score per A-keypoint, or nothing
		s0 = np.zeros(len(m0), np.float32)
	valid = (m0 != -1) & (m0 < len(kps_b))
	indices = np.where(valid)[0]
	dmatches = [
		cv2.DMatch(int(i), int(m0[i]), 1.0 - float(s0[i]))
		for i in indices
	]
	scores = s0[indices].astype(np.float32) if len(indices) else np.zeros(0, np.float32)

	n_matches = len(dmatches)
	if n_matches > 0:
		order = np.argsort(scores)[::-1]
		dmatches = [dmatches[i] for i in order]
		scores = scores[order]

	points_a = np.zeros((0, 1, 2), np.float32)
	points_b = np.zeros((0, 1, 2), np.float32)
	if n_matches > 0:
		points_a = np.array(
			[[kps_a[dm.queryIdx].pt] for dm in dmatches], np.float32)
		points_b = np.array(
			[[kps_b[dm.trainIdx].pt] for dm in dmatches], np.float32)

	return (_dmatch_rows(dmatches), points_a, points_b,
	        scores.astype(np.float32), n_matches)


def _task_llm_generate(model_path, engine, tokenizer_dir, text, stops,
                       max_new_tokens, use_kv, bos):
	"""(response, full_text, new_tokens) - tokenization, generation and
	detokenization all happen here (the tokenizer is not picklable)."""
	net = _load_gen_net(model_path, engine)
	tok = _load_tokenizer(tokenizer_dir)
	ids = list(tok.encode(text))
	if bos is not None:
		ids = [int(bos)] + ids
	if not ids:
		raise ValueError("The prompt encoded to zero tokens.")
	generated = _greedy_decoder_only(net, ids, int(max_new_tokens), stops, use_kv)
	clean = [t for t in generated if t not in stops]
	return tok.decode(clean), tok.decode(ids + clean), len(generated)


def _task_vlm_generate(vision_path, embed_path, lm_path, engine, tokenizer_dir,
                       frame, prompt, max_new_tokens, stops, image_size,
                       pixel_mean, pixel_std):
	"""(response, new_tokens)."""
	tok = _load_tokenizer(tokenizer_dir)
	vis_net = _load_gen_net(vision_path, engine)
	emb_net = _load_gen_net(embed_path, engine)
	lm_net = _load_gen_net(lm_path, engine)

	size = int(image_size)
	img = cv2.resize(frame, (size, size))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
	img = (img - np.asarray(pixel_mean, np.float32)) / np.asarray(pixel_std, np.float32)
	pixel = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])

	_set_named(vis_net, pixel, "pixel_values")
	image_feats = np.asarray(vis_net.forward())        # (1, T, D)

	ids = list(tok.encode(str(prompt)))
	if not ids:
		raise ValueError("The prompt encoded to zero tokens.")
	_set_named(emb_net, np.array([ids], np.int64), "input_ids")
	text_embeds = np.asarray(emb_net.forward())        # (1, P, D)
	if image_feats.shape[-1] != text_embeds.shape[-1]:
		raise ValueError(
			f"vision feature width {image_feats.shape[-1]} != embedding "
			f"width {text_embeds.shape[-1]} - mismatched model parts.")

	inputs_embeds = np.concatenate([image_feats, text_embeds], axis=1)
	generated = []
	_set_named(lm_net, inputs_embeds, "inputs_embeds")
	new_id = _last_token(lm_net.forward())
	generated.append(new_id)
	for _ in range(int(max_new_tokens) - 1):
		if new_id in stops:
			break
		_set_named(emb_net, np.array([[new_id]], np.int64), "input_ids")
		inputs_embeds = np.concatenate(
			[inputs_embeds, np.asarray(emb_net.forward())], axis=1)
		_set_named(lm_net, inputs_embeds, "inputs_embeds")
		new_id = _last_token(lm_net.forward())
		generated.append(new_id)
	clean = [t for t in generated if t not in stops]
	return tok.decode(clean), len(generated)


def _task_seq2seq_generate(encoder_path, decoder_path, vision_path, embed_path,
                           engine, tokenizer_dir, frame, prompt, start_ids,
                           max_new_tokens, stops, image_size, pixel_mean,
                           pixel_std, prompt_token_ids, prompt_suffix_ids):
	"""(response, new_tokens)."""
	tok = _load_tokenizer(tokenizer_dir)
	enc_net = _load_gen_net(encoder_path, engine)
	dec_net = _load_gen_net(decoder_path, engine)
	vis_net = (_load_gen_net(vision_path, engine) if vision_path else None)
	emb_net = (_load_gen_net(embed_path, engine) if embed_path else None)

	size = int(image_size)
	img = cv2.resize(frame, (size, size))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
	img = (img - np.asarray(pixel_mean, np.float32)) / np.asarray(pixel_std, np.float32)
	pixel = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])

	# --- encoder ----------------------------------------------------------
	if vis_net is None:
		# shape (a): the image IS the encoder input (ViT-GPT2 style)
		_set_named(enc_net, pixel, "pixel_values")
	else:
		# shape (b): vision features + embedded prompt as inputs_embeds
		_set_named(vis_net, pixel, "pixel_values")
		feats = np.asarray(vis_net.forward())
		if str(prompt_token_ids).strip():
			prompt_ids = list(_parse_id_list(prompt_token_ids))
		else:
			prompt_ids = (list(tok.encode(str(prompt))) +
			              list(_parse_id_list(prompt_suffix_ids)))
		if not prompt_ids:
			raise ValueError("No encoder prompt: empty prompt and "
			                 "prompt_token_ids.")
		if emb_net is None:
			raise ValueError("a vision_model without an embed_model "
			                 "cannot build the encoder's inputs_embeds.")
		_set_named(emb_net, np.array([prompt_ids], np.int64), "input_ids")
		text_embeds = np.asarray(emb_net.forward())
		if feats.shape[-1] != text_embeds.shape[-1]:
			raise ValueError(
				f"vision feature width {feats.shape[-1]} != embedding "
				f"width {text_embeds.shape[-1]} - mismatched model parts.")
		enc_in = np.concatenate([feats, text_embeds], axis=1)
		_set_named(enc_net, enc_in, "inputs_embeds")
		_set_named(enc_net, np.ones((1, enc_in.shape[1]), np.int64),
		           "attention_mask")
	hidden = np.ascontiguousarray(enc_net.forward())

	# --- decoder ----------------------------------------------------------
	ids = list(start_ids)
	dec_net.enableKVCache()
	dec_net.resetKVCache()
	# merged transformers.js exports branch on a scalar bool input
	# ('use_cache_branch'): False for the first pass, True once the KV cache
	# holds a past (see _greedy_decoder_only for the same contract).
	_set_named(dec_net, np.array([False]), "use_cache_branch")
	generated = []

	def dec_step(ids2d):
		if emb_net is not None:
			_set_named(emb_net, ids2d, "input_ids")
			_set_named(dec_net, np.asarray(emb_net.forward()),
			           "inputs_embeds")
		_set_named(dec_net, ids2d, "input_ids")
		_set_named(dec_net, hidden, "encoder_hidden_states")
		_set_named(dec_net, np.ones((1, hidden.shape[1]), np.int64),
		           "encoder_attention_mask")
		return dec_net.forward()

	new_id = _last_token(dec_step(np.array([ids], np.int64)))
	generated.append(new_id)
	for _ in range(int(max_new_tokens) - 1):
		if new_id in stops:
			break
		_set_named(dec_net, np.array([True]), "use_cache_branch")
		new_id = _last_token(dec_step(np.array([[new_id]], np.int64)))
		generated.append(new_id)
	clean = [t for t in generated if t not in stops]
	return tok.decode(clean), len(generated)


def _parse_id_list(text, fallback=()):
	"""'151645, 151643' -> (151645, 151643); blank -> fallback."""
	s = str(text).strip()
	if not s:
		return tuple(fallback)
	try:
		return tuple(int(t) for t in s.replace(";", ",").split(",") if t.strip())
	except ValueError:
		raise ValueError(f"stop/start token ids '{text}' is not a "
		                 "comma-separated list of integers.")


TASK_FUNCS = {
	"yunet_detect": _task_yunet_detect,
	"sface_embed": _task_sface_embed,
	"mp_palm_detect": _task_mp_palm_detect,
	"mp_hand_pose": _task_mp_hand_pose,
	"wechat_qr": _task_wechat_qr,
	"colorize": _task_colorize,
	"east_detect": _task_east_detect,
	"dnn_forward": _task_dnn_forward,
	"dnn_forward_all": _task_dnn_forward_all,
	"feature_extract": _task_feature_extract,
	"match_features": _task_match_features,
	"llm_generate": _task_llm_generate,
	"vlm_generate": _task_vlm_generate,
	"seq2seq_generate": _task_seq2seq_generate,
}
