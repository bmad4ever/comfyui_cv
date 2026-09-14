# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""DNN-model-based OpenCV nodes.

Nodes that wrap OpenCV's cv2.dnn / model zoo detectors. These load an .onnx
model from ComfyUI's models/onnx folder and run inference on an IMAGE.

Design notes:
* Data only - no visualization is baked in. The detector outputs bounding
  boxes (core BOUNDING_BOX), landmark points (NPARRAY) and scores; chain into
  the core 'Draw BBoxes' node and 'CV Draw Points' to see them.
* Failure tolerant - zero faces is a valid result: empty arrays / zero count,
  never a raised exception. Only genuine wiring mistakes raise.

The MediaPipe hand-pose pipeline is two-stage (mirrors the OpenCV Zoo
handpose_estimation_mediapipe reference): 'CV MediaPipe Palm Detect' finds
each palm (box + 7 palm landmarks), and 'CV MediaPipe Hand Pose' takes those
palm rows and regresses the 21 hand keypoints per hand. The palm rows are passed
between the two nodes as a raw (N, 19) NPARRAY, so the hand-pose node re-crops,
rotates and aligns each hand exactly the way the reference does. Gesture naming
is deliberately NOT baked in (that is task semantics) - build it as a
subgraph over the emitted landmarks if you want it.

'CV WeChat QR Detect' wraps cv2.wechat_qrcode.WeChatQRCode (mirrors the
weChatQRcode reference demo): it detects AND decodes every QR code in one shot
using two .onnx models (a detect model + a super-resolution model). It needs the
wechat_qrcode CONTRIB module in the loaded OpenCV build - the stock
opencv-python wheel omits it, so opencv-contrib-python(-headless) must own the shared
cv2.pyd (tools/repair_opencv_contrib.py diagnoses and repairs an install where
it does not). Data only: the decoded strings, the 4 corner points per code and
closed-quad edges come out for the core 'Draw BBoxes' node + 'CV Draw
Points/Connections/Labels'.

'CV Colorize Model' colourizes a grayscale image using the colorization_deploy_v2
ONNX model (Zhang et al., ECCV 2016) through cv2.dnn. It takes the L channel of the
input in Lab space, runs it through the network to predict the ab colour channels, and
merges them back to produce a full-colour BGR output. The 'strength' slider scales the
predicted ab channels (0 = grayscale passthrough, 1 = original model output).

Finally there is a set of GENERIC cv2.dnn primitives - 'CV DNN Blob From
Image', 'CV DNN Forward' and 'CV DNN Images From Blob' - that decompose
a forward pass into reusable steps (generic building blocks rather than
bespoke wrappers). Chained with the plain cv2 nodes cv2.convertScaleAbs
(scale + clip) and 'CV Array -> Image', they handle any ONNX image model
(denoise, deblur, super-resolution, style transfer).

The same generic philosophy covers letterboxed detection/segmentation models
(the YOLO26-seg pipeline, workflow 51): 'OpenCV
DNN Letterbox' (aspect-preserving resize + pad, emitting ratio/pads), 'CV DNN
Forward All' (every unconnected output at once; the loader dispatches on the file
extension so .tflite models are listed and load via readNetFromTFLite), 'OpenCV
DNN Pick Output' (select outputs by index, layer name, or shape), 'CV YOLO
Detect Decode' (unwrap a detection head into bboxes/scores/class ids - four
named layouts: NMS-free rows, yolov5/v8 class-scores rows/columns, yolov4
separate boxes+confs), 'CV YOLO Seg Masks' (optional second stage: one
binary MASK per detection from the kept rows' mask coefficients + the proto
map) and 'CV Class Scores Decode' (classification heads, top-k). Class
names come from a plain .txt file via 'CV Load Labels' instead of a
hard-coded list, so the chain adapts across the YOLO family and similar models
by swapping the model file, layout dropdown and labels.

The OpenCV 5 LLM/VLM support (built-in BPE/SentencePiece Tokenizer, KV-cache
routing in the new graph engine) is exposed the same generic way: 'CV DNN
Tokenizer' (encode/decode around a model folder with config.json +
tokenizer.json), 'CV LLM Generate' (decoder-only text generation: prompt
template dropdown - chatml/gemma3/raw -, greedy loop, KV-cache or
full-sequence decode modes, covering Qwen2.5/Gemma3/GPT-2 ONNX exports),
'CV VLM Generate' (decoder-only VLMs of the PaliGemma shape: vision
encoder + token embedding + language model, image features prefixed to the
text embeddings) and 'CV Seq2Seq Generate' (encoder-decoder models like
ViT-GPT2 image captioning: image -> encoder hidden states -> autoregressive
decoder). Models live in models/llm (the LLM category); every model file and
the chat template are dropdown choices, no model-specific monolith.
"""

import logging
import os

import cv2
import numpy as np
import torch

from comfy_api.latest import io

from . import imageutils, polytype, workerutils
from .convert import NPArray
from .features import KeyPoints, Matches, _to_bgr_u8

CATEGORY = "image/CV/dnn"


def _register_onnx_folder():
	"""Register models/onnx with folder_paths so the model combo can list it.

	ComfyUI does not ship an 'onnx' folder mapping by default; add one (idempotent)
	pointing at models/onnx. Runs at import and again inside define_schema so the
	combo is populated even on a cold server."""
	import folder_paths

	if "onnx" not in folder_paths.folder_names_and_paths:
		path = os.path.join(folder_paths.models_dir, "onnx")
		os.makedirs(path, exist_ok=True)
		folder_paths.folder_names_and_paths["onnx"] = ([path], {".onnx", ".tflite"})
	elif folder_paths.folder_names_and_paths["onnx"][1]:
		# An extra_model_paths.yaml registration leaves an EMPTY extension set,
		# which means "list every file" - adding .tflite to it would RESTRICT
		# the listing to .tflite only (the missing-.onnx bug). Only extend an
		# explicitly-filtered set, and never mutate it in place (the set may be
		# shared with other folders) - replace the entry with a merged copy.
		paths, exts = folder_paths.folder_names_and_paths["onnx"]
		wanted = {".onnx", ".tflite"}
		if not wanted <= set(exts):
			folder_paths.folder_names_and_paths["onnx"] = (paths, set(exts) | wanted)


def _onnx_models() -> list:
	"""Every model under the onnx roots, '/'-separated relative names, so saved
	workflows stay portable across machines (same reason as _llm_listing)."""
	import folder_paths

	_register_onnx_folder()
	return [f.replace(os.sep, "/") for f in folder_paths.get_filename_list("onnx")]


_register_onnx_folder()


class YuNetFaceDetect(io.ComfyNode):
	"""cv2.FaceDetectorYN (YuNet) face detector.

	Outputs one bbox per detected face plus the 5 facial landmarks
	(right eye, left eye, nose tip, right/left mouth corner) as a point array.
	"""

	# cache the created detector across calls: (path, conf, nms, top_k) -> model
	_cache: dict = {}

	@classmethod
	def define_schema(cls):
		models = _onnx_models()
		return io.Schema(
			node_id="CV_YuNetFaceDetect",
			display_name="CV YuNet Face Detect",
			category=CATEGORY,
			description="Detects faces with the YuNet CNN (cv2.FaceDetectorYN). "
			            "Place the .onnx model in ComfyUI/models/onnx (e.g. "
			            "face_detection_yunet_2023mar.onnx). Outputs bounding "
			            "boxes, the 5 facial landmarks per face as points, the "
			            "per-face confidence and the face count. Data only - "
			            "visualize the bboxes with the core 'Draw BBoxes' node "
			            "and the landmarks with 'CV Draw Points'. Zero faces "
			            "is a valid result (empty outputs).",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Input image. An IMAGE batch is processed frame by frame; "
					        "an NPARRAY is treated as a single frame (gray/BGR/BGRA, "
					        "any dtype). The landmark and score outputs are "
					        "concatenated across frames."),
				io.Combo.Input("model", options=models,
					tooltip="YuNet .onnx model file from ComfyUI/models/onnx."),
				io.Float.Input("conf_threshold", default=0.9, min=0.0, max=1.0, step=0.01,
					tooltip="Minimum confidence to keep a face. Lower detects more "
					        "(and more false positives)."),
				io.Float.Input("nms_threshold", default=0.3, min=0.0, max=1.0, step=0.01,
					tooltip="Non-maximum suppression IoU: boxes overlapping by more "
					        "than this are merged."),
				io.Int.Input("top_k", default=5000, min=1, max=20000,
					tooltip="Keep at most this many boxes before NMS."),
			],
			outputs=[
				io.BoundingBox.Output(display_name="bboxes",
					tooltip="One {x, y, width, height, score} dict per face - feed the "
					        "core 'Draw BBoxes' node or 'Image Crop'. A batch nests "
					        "one list per frame."),
				NPArray.Output(display_name="landmarks",
					tooltip="(N, 2) float32 point array: the 5 landmarks of every face, "
					        "in order right eye, left eye, nose, right/left mouth corner. "
					        "Feed 'CV Draw Points'. Empty (0, 2) when no faces."),
				NPArray.Output(display_name="scores",
					tooltip="(N,) float32 confidence per face, same order as bboxes."),
				io.Int.Output(display_name="face_count",
					tooltip="Total number of faces detected across the batch."),
			],
		)

	@classmethod
	def _detector(cls, model_name, conf, nms, top_k):
		# the detector itself is built inside the worker (per call); only the
		# resolved model path + thresholds cross the subprocess boundary
		return _model_path(model_name, "YuNet"), float(conf), float(nms), int(top_k)

	@classmethod
	def execute(cls, image, model, conf_threshold, nms_threshold, top_k) -> io.NodeOutput:
		frames = imageutils.frames_from_input(image)
		if not frames:
			return io.NodeOutput([], np.zeros((0, 2), np.float32),
			                     np.zeros((0,), np.float32), 0)
		path, conf, nms, top = cls._detector(model, conf_threshold, nms_threshold, top_k)
		bboxes, landmarks, scores, face_count = _run_dnn_worker(
			"YuNet Face Detect", "yunet_detect", {
				"model_path": path,
				"frames": [np.ascontiguousarray(f) for f in frames],
				"conf": conf, "nms": nms, "top_k": top})
		return io.NodeOutput(bboxes, landmarks, scores, face_count)


# --- MediaPipe hand-pose pipeline ---------------------------------------------
# The two-stage MediaPipe pipeline (palm detect + hand pose) lives in
# workers/dnn_tasks.py, ported from the OpenCV Zoo
# handpose_estimation_mediapipe reference (Apache-2.0; see that file's
# header and the README provenance section). This module only ships the runnable
# schema wrappers; every stage runs in the interruptible DNN worker.


def _recreate_exception(exc_type, message):
	"""Rebuild the worker's exception in the parent for the common cases, so
	downstream code that caught the in-process exception type (e.g. `except
	cv2.error` around a DNN call) keeps working unchanged."""
	# the worker serializes type(exc).__name__; cv2.error's __name__ is "error",
	# not "cv2.error" - accept both so `except cv2.error` keeps working
	exc_cls = {"cv2.error": cv2.error, "error": cv2.error,
	           "ValueError": ValueError, "RuntimeError": RuntimeError,
	           "TypeError": TypeError,
	           "KeyError": KeyError}.get(str(exc_type), RuntimeError)
	return exc_cls(str(message))


def _run_dnn_worker(node_label, task, args):
	"""Run a DNN task core in the interruptible worker subprocess.

	The node's execute() resolves model names to absolute paths and converts
	tensors to ndarrays, then this hands the rest to the worker so a long
	forward pass / generation loop runs in a process ComfyUI can terminate
	mid-call. DNN work has NO safe fallback value (unlike stitching), so any
	worker failure or crash raises; the original exception type is preserved
	for the common cases (cv2.error, ValueError, ...) so the public behavior
	matches the old in-process path. Nets are reloaded per call (~1-30 s); the
	schemas and widget orders are untouched.
	"""
	tag, result = workerutils.run_subprocess(
		workerutils.worker_script_path('dnn_worker.py'),
		{'task': task, 'args': args})
	if tag == 'ok':
		return result
	if tag == 'crash':
		msg = result if isinstance(result, str) and result else "unknown error"
		raise RuntimeError(f"{node_label} worker crashed: {msg}")
	if (isinstance(result, (tuple, list)) and len(result) >= 2
	        and isinstance(result[0], str)):
		raise _recreate_exception(result[0], result[1])
	raise RuntimeError(f"{node_label} worker failed: {result}")


def _keypoint_rows(kps):
	"""cv2.KeyPoint list -> (N, 6) rows [x, y, size, angle, response, octave].
	cv2.KeyPoint is NOT picklable in this build, so the worker serializes them."""
	return np.array(
		[[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave]
		 for kp in kps], np.float64)


def _keypoints_from_rows(rows):
	"""The inverse of _keypoint_rows: (N, 6) -> cv2.KeyPoint list."""
	return [cv2.KeyPoint(float(r[0]), float(r[1]), float(r[2]),
	                     float(r[3]), float(r[4]), int(r[5])) for r in rows]


def _dmatches_from_rows(rows):
	"""(M, 4) [queryIdx, trainIdx, imgIdx, distance] -> cv2.DMatch list
	(cv2.DMatch is not picklable either)."""
	return [cv2.DMatch(int(r[0]), int(r[1]), int(r[2]), float(r[3]))
	        for r in rows]


def _model_path(model_name: str, kind: str) -> str:
	import folder_paths

	path = folder_paths.get_full_path("onnx", model_name)
	if path is None and os.sep != "/":
		path = folder_paths.get_full_path("onnx", model_name.replace("/", os.sep))
	if path is None:
		raise ValueError(
			f"{kind} model '{model_name}' not found in models/onnx. "
			"Place the .onnx file there.")
	return path


def _xyxy_to_box(x1, y1, x2, y2, score, label):
	x1, y1, x2, y2 = (int(round(float(v))) for v in (x1, y1, x2, y2))
	return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
	        "score": round(float(score), 4), "label": label}


class MPPalmDetect(io.ComfyNode):
	"""MediaPipe palm detector (cv2.dnn). Stage 1 of hand-pose estimation."""

	@classmethod
	def define_schema(cls):
		models = _onnx_models()
		return io.Schema(
			node_id="CV_MPPalmDetect",
			display_name="CV MediaPipe Palm Detect",
			category=CATEGORY,
			description="Detects palms with the MediaPipe palm SSD (cv2.dnn). "
			            "Place palm_detection_mediapipe_2023feb.onnx in "
			            "ComfyUI/models/onnx and pick it. Stage 1 of hand-pose "
			            "estimation: feed the 'palms' output into 'CV MediaPipe "
			            "Hand Pose'. Outputs are DATA only - visualize the boxes with "
			            "the core 'Draw BBoxes' node and the 7 palm points with "
			            "'CV Draw Points'. Zero palms is a valid result (empty "
			            "outputs). Processes a single image (the first frame of a batch).",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Input image. An IMAGE batch uses its first frame; an "
					        "NPARRAY (gray/BGR/BGRA, any dtype) is treated as one frame."),
				io.Combo.Input("model", options=models,
					tooltip="MediaPipe palm .onnx model from ComfyUI/models/onnx "
					        "(palm_detection_mediapipe_2023feb.onnx)."),
				io.Float.Input("score_threshold", default=0.6, min=0.0, max=1.0, step=0.01,
					tooltip="Minimum sigmoid confidence to keep a palm. Lower detects "
					        "more (and more false positives)."),
				io.Float.Input("nms_threshold", default=0.3, min=0.0, max=1.0, step=0.01,
					tooltip="Non-maximum suppression IoU: palms overlapping by more "
					        "than this are merged."),
				io.Int.Input("top_k", default=5000, min=1, max=20000,
					tooltip="Keep at most this many candidate boxes before NMS."),
			],
			outputs=[
				io.BoundingBox.Output(display_name="bboxes",
					tooltip="One {x, y, width, height, score} dict per palm - feed the "
					        "core 'Draw BBoxes' node."),
				NPArray.Output(display_name="palms",
					tooltip="(N, 19) float32: each row [x1, y1, x2, y2, 7*(lx, ly), "
					        "score]. Feed this straight into 'CV MediaPipe Hand "
					        "Pose'. Empty (0, 19) when no palms."),
				NPArray.Output(display_name="palm_points",
					tooltip="(N*7, 2) float32: the 7 palm landmarks of every palm "
					        "(wrist + finger bases), for 'CV Draw Points'. Empty "
					        "(0, 2) when no palms."),
				NPArray.Output(display_name="scores",
					tooltip="(N,) float32 confidence per palm, same order as bboxes."),
				io.Int.Output(display_name="palm_count",
					tooltip="Number of palms detected."),
			],
		)

	@classmethod
	def execute(cls, image, model, score_threshold, nms_threshold, top_k) -> io.NodeOutput:
		frame = imageutils.first_bgr_frame(image)
		if frame is None:
			return io.NodeOutput([], np.zeros((0, 19), np.float32),
			                     np.zeros((0, 2), np.float32), np.zeros((0,), np.float32), 0)
		path = _model_path(model, "MediaPipe palm")
		palms, = _run_dnn_worker("MediaPipe Palm Detect", "mp_palm_detect", {
			"model_path": path,
			"frame": np.ascontiguousarray(frame),
			"score_threshold": float(score_threshold),
			"nms_threshold": float(nms_threshold),
			"top_k": int(top_k)})

		boxes = [_xyxy_to_box(p[0], p[1], p[2], p[3], p[18], "palm") for p in palms]
		points = (palms[:, 4:18].reshape(-1, 2).astype(np.float32)
		          if len(palms) else np.zeros((0, 2), np.float32))
		scores = (palms[:, 18].astype(np.float32)
		          if len(palms) else np.zeros((0,), np.float32))
		return io.NodeOutput(boxes, palms, points, scores, int(len(palms)))


class MPHandPose(io.ComfyNode):
	"""MediaPipe hand-pose regressor (cv2.dnn). Stage 2: 21 keypoints per hand."""

	@classmethod
	def define_schema(cls):
		models = _onnx_models()
		return io.Schema(
			node_id="CV_MPHandPose",
			display_name="CV MediaPipe Hand Pose",
			category=CATEGORY,
			description="Regresses the 21 hand keypoints per palm with the MediaPipe "
			            "hand-pose model (cv2.dnn). Stage 2 of hand-pose estimation: "
			            "feed the 'palms' output of 'CV MediaPipe Palm Detect' plus "
			            "the same image. Place handpose_estimation_mediapipe_2023feb_"
			            "int8bq.onnx in ComfyUI/models/onnx and pick it. Outputs are "
			            "DATA only - draw the hand boxes with the core 'Draw BBoxes' "
			            "node and the 21 keypoints with 'CV Draw Points'. A palm "
			            "whose confidence is below conf_threshold is dropped; zero "
			            "hands is a valid result (empty outputs). Processes a single "
			            "image (the first frame of a batch).",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="The SAME image the palms were detected on. An IMAGE batch "
					        "uses its first frame; an NPARRAY is treated as one frame."),
				NPArray.Input("palms",
					tooltip="The (N, 19) 'palms' array from 'CV MediaPipe Palm "
					        "Detect'. Each row seeds one hand's crop/rotation."),
				io.Combo.Input("model", options=models,
					tooltip="MediaPipe hand-pose .onnx model from ComfyUI/models/onnx "
					        "(handpose_estimation_mediapipe_2023feb_int8bq.onnx)."),
				io.Float.Input("conf_threshold", default=0.9, min=0.0, max=1.0, step=0.01,
					tooltip="Minimum model confidence to keep a hand. Palms below this "
					        "are dropped (not an error)."),
			],
			outputs=[
				io.BoundingBox.Output(display_name="bboxes",
					tooltip="One {x, y, width, height, score} dict per hand (the tight "
					        "hand box) - feed the core 'Draw BBoxes' node."),
				NPArray.Output(display_name="landmarks",
					tooltip="(N*21, 2) float32 screen keypoints of every hand, in "
					        "MediaPipe order (wrist, thumb..pinky). Feed 'CV Draw "
					        "Points'. Empty (0, 2) when no hands."),
				NPArray.Output(display_name="world_landmarks",
					tooltip="(N*21, 3) float32 metric 3D keypoints (x, y, z in meters, "
					        "origin at the hand centre). For inspection / 3D use."),
				NPArray.Output(display_name="handedness",
					tooltip="(N,) float32 in [0, 1]: <=0.5 is a left hand, >0.5 right, "
					        "same order as bboxes."),
				NPArray.Output(display_name="scores",
					tooltip="(N,) float32 confidence per hand, same order as bboxes."),
				io.Int.Output(display_name="hand_count",
					tooltip="Number of hands kept (palms that passed conf_threshold)."),
				NPArray.Output(display_name="connections",
					tooltip="(N*21, 2) int32 edge index-pairs (the MediaPipe hand "
					        "skeleton), already offset per hand to index the 'landmarks' "
					        "output directly. Feed 'landmarks' + this into 'CV Draw "
					        "Connections' to draw the bones. Empty (0, 2) when no hands."),
				NPArray.Output(display_name="keypoint_names",
					tooltip="(N*21,) string array naming every keypoint (wrist, "
					        "thumb_tip, ...), aligned row-for-row with 'landmarks'. Feed "
					        "'landmarks' + this into 'CV Draw Labels' to annotate "
					        "them. Empty (0,) when no hands."),
			],
		)

	@classmethod
	def execute(cls, image, palms, model, conf_threshold) -> io.NodeOutput:
		empty = (np.zeros((0, 2), np.float32), np.zeros((0, 3), np.float32),
		         np.zeros((0,), np.float32), np.zeros((0,), np.float32), 0,
		         np.zeros((0, 2), np.int32), np.array([], dtype="<U20"))
		frame = imageutils.first_bgr_frame(image)
		palms = np.asarray(palms, np.float32).reshape(-1, 19)
		if frame is None or len(palms) == 0:
			return io.NodeOutput([], *empty)
		path = _model_path(model, "MediaPipe hand-pose")
		return io.NodeOutput(*_run_dnn_worker("MediaPipe Hand Pose", "mp_hand_pose", {
			"model_path": path,
			"frame": np.ascontiguousarray(frame),
			"palms": palms,
			"conf_threshold": float(conf_threshold)}))


class WeChatQRDetect(io.ComfyNode):
	"""cv2.wechat_qrcode.WeChatQRCode detector: finds AND decodes QR codes."""

	@classmethod
	def define_schema(cls):
		models = _onnx_models()
		return io.Schema(
			node_id="CV_WeChatQRDetect",
			display_name="CV WeChat QR Detect",
			category=CATEGORY,
			description="Detects and decodes every QR code in an image with "
			            "cv2.wechat_qrcode.WeChatQRCode (one shot - it both locates "
			            "and reads them). Pick the two .onnx models from "
			            "ComfyUI/models/onnx: a detect model (detect_2026april.onnx) "
			            "and a super-resolution model (sr_2026april.onnx) that "
			            "upscales small codes before decoding. Outputs are DATA only "
			            "- draw the boxes with the core 'Draw BBoxes' node, the 4 "
			            "corners with 'CV Draw Points', the code outline by "
			            "feeding 'corners' + 'connections' to 'CV Draw "
			            "Connections', and the decoded text onto the image by feeding "
			            "'corners' + 'labels' to 'CV Draw Labels'. Zero codes is a "
			            "valid result (empty outputs). Processes a single image (the "
			            "first frame of a batch). Needs the wechat_qrcode CONTRIB "
			            "module in the loaded OpenCV build (opencv-contrib-python(-headless)).",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Input image. An IMAGE batch uses its first frame; an "
					        "NPARRAY (gray/BGR/BGRA, any dtype) is treated as one frame."),
				io.Combo.Input("detect_model", options=models,
					tooltip="WeChat QR DETECT .onnx model from ComfyUI/models/onnx "
					        "(detect_2026april.onnx)."),
				io.Combo.Input("sr_model", options=models,
					tooltip="WeChat QR SUPER-RESOLUTION .onnx model from "
					        "ComfyUI/models/onnx (sr_2026april.onnx) - upsamples small "
					        "codes so they decode."),
			],
			outputs=[
				io.BoundingBox.Output(display_name="bboxes",
					tooltip="One {x, y, width, height, score} dict per QR code (the "
					        "axis-aligned box around its 4 corners, label = decoded "
					        "text) - feed the core 'Draw BBoxes' node."),
				NPArray.Output(display_name="corners",
					tooltip="(N*4, 2) float32: the 4 corner points of every code, in "
					        "order. Feed 'CV Draw Points', or 'CV Draw "
					        "Connections'/'Draw Labels' with the matching outputs "
					        "below. Empty (0, 2) when no codes."),
				NPArray.Output(display_name="connections",
					tooltip="(N*4, 2) int32 edge index-pairs closing each code's quad, "
					        "already offset per code to index 'corners' directly. Feed "
					        "'corners' + this into 'CV Draw Connections' to outline "
					        "the codes. Empty (0, 2) when no codes."),
				NPArray.Output(display_name="texts",
					tooltip="(N,) string array: the decoded text of each code (one "
					        "entry per code). Feed 'Preview as Text'/'Inspect CV Data'. "
					        "An empty string means the code was located but not decoded."),
				NPArray.Output(display_name="labels",
					tooltip="(N*4,) string array aligned row-for-row with 'corners': "
					        "each code's decoded text on its FIRST corner, blank on the "
					        "other three. Feed 'corners' + this into 'CV Draw "
					        "Labels' to write the text once per code. Empty (0,) when "
					        "no codes."),
				io.Int.Output(display_name="qr_count",
					tooltip="Number of QR codes detected."),
			],
		)

	@classmethod
	def execute(cls, image, detect_model, sr_model) -> io.NodeOutput:
		empty_str = np.array([], dtype="<U1")
		empty = ([], np.zeros((0, 2), np.float32), np.zeros((0, 2), np.int32),
		         empty_str, empty_str, 0)
		frame = imageutils.first_bgr_frame(image)
		if frame is None:
			return io.NodeOutput(*empty)

		dp = _model_path(detect_model, "WeChat QR detect")
		sp = _model_path(sr_model, "WeChat QR super-resolution")
		return io.NodeOutput(*_run_dnn_worker("WeChat QR Detect", "wechat_qr", {
			"detect_path": dp, "sr_path": sp,
			"frame": np.ascontiguousarray(frame)}))


class ColorizeModel(io.ComfyNode):
	"""Grayscale image colorization (cv2.dnn). Output is a colorized IMAGE."""

	@classmethod
	def define_schema(cls):
		models = _onnx_models()
		return io.Schema(
			node_id="CV_ColorizeModel",
			display_name="CV Colorize Model",
			category=CATEGORY,
			description="Automatic colorization of a grayscale image with the "
			            "colorization_deploy_v2 ONNX model (Zhang et al., ECCV 2016 "
			            "via cv2.dnn). Place colorization_deploy_v2_2026april.onnx "
			            "in ComfyUI/models/onnx and pick it. The model predicts the "
			            "Lab ab colour channels from the L channel, which are merged "
			            "back with the original L to produce a full-colour output. "
			            "Feed a grayscale image; colour images are converted to "
			            "grayscale first. The 'strength' slider scales the predicted "
			            "ab channels (0 = grayscale passthrough, 1 = original model "
			            "output). An IMAGE batch is processed frame by frame.",
			inputs=[
				io.Image.Input("image",
					tooltip="Grayscale input image. A colour IMAGE is converted "
					        "to grayscale before colourization. An IMAGE batch is "
					        "processed frame by frame; the output resolution matches "
					        "the input."),
				io.Combo.Input("model", options=models,
					tooltip="Colorization .onnx model from ComfyUI/models/onnx "
					        "(colorization_deploy_v2_2026april.onnx)."),
				io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05,
					tooltip="Scales the predicted ab colour channels before merging "
					        "with L. 0.0 = grayscale passthrough, 1.0 = original model "
					        "output, >1.0 = oversaturated colours.",
					optional=True),
			],
			outputs=[
				io.Image.Output(display_name="image",
					tooltip="The colourized image, same resolution as the input. "
					        "Feed a Preview Image to see it."),
			],
		)

	@classmethod
	def execute(cls, image, model, strength=1.0) -> io.NodeOutput:
		path = _model_path(model, "Colorize")
		frames = imageutils.frames_from_input(image)
		if not frames:
			return io.NodeOutput(imageutils.bgr_to_image_batch([]))
		colored, = _run_dnn_worker("Colorize Model", "colorize", {
			"model_path": path,
			"frames": [np.ascontiguousarray(f) for f in frames],
			"strength": float(strength)})
		return io.NodeOutput(imageutils.bgr_to_image_batch(colored))


# --- Generic cv2.dnn primitives -------------------------------------------------
# Reusable building blocks that decompose any ONNX forward pass, so a model
# pipeline can be assembled from generic nodes instead of a bespoke wrapper:
# 'DNN Blob From Image' (preprocess) -> 'DNN Forward' (load +
# infer) -> 'DNN Images From Blob' (postprocess, undoes the CHW->HWC transpose).
# The scale/clip and colour steps are then plain cv2 nodes (cv2.convertScaleAbs
# with alpha=255 to scale+clip to 0-255, then 'CV Array -> Image' for RGB), which
# is exactly how workflow 32 reproduces workflow 31's NAFNet deblur without the
# CV_NafnetDeblur node. The forward passes run in the interruptible DNN
# worker (workers/dnn_tasks.py 'dnn_forward' / 'dnn_forward_all').


# cv2.dnn compute backend / target as NAMED dropdowns (never a raw int).
# Only the constants present in the loaded OpenCV build are offered, each mapped
# from a readable label. The first entry maps to None = "leave the net on its
# default", which SKIPS the setPreferable* call entirely.
#
# WHAT THIS BUILD ACTUALLY SUPPORTS (measured across every engine x backend x
# target combination, not read off the constant list):
#
#   * The graph engine IGNORES both selectors - it warns "Back-ends are not
#     supported by the new graph engine for now" / "Targets are not supported by
#     the new graph engine for now" and runs on the default CPU path regardless.
#     Proof it is not merely tolerant: DNN_BACKEND_CUDA and DNN_BACKEND_VKCOM
#     RAISE on the classic engine and run fine (identically timed) on the graph
#     one, i.e. the setting never reached anything.
#   * The CLASSIC engine DID honour them, and OpenCL was real on the 5.0 build:
#     that one was compiled with OpenCL (NVD3D11) and reached the discrete GPU.
#   * CUDA can never work here - the wheel is built without it
#     (getCudaEnabledDeviceCount() == 0, getAvailableTargets(DNN_BACKEND_CUDA)
#     is empty), so DNN_BACKEND_CUDA raises and DNN_TARGET_CUDA silently falls
#     back to CPU (identical timings). Same for OpenVINO/Inference Engine.
#
# ON THE CURRENT BUILD BOTH WIDGETS ARE INERT (measured 2026-09-07, OpenCV
# 5.1.0-dev). 5.1 removed ENGINE_CLASSIC - the surviving ENGINE_OPENCV IS the
# graph engine, which is what the warnings above say - so there is no longer an
# engine that honours a backend or a target, and this build also reports
# cv2.ocl.haveOpenCL() == False. squeezenet1.1-7 times identically (0.0037-
# 0.0042 s) across default / OPENCV+CPU / OPENCV+OPENCL / CUDA / VKCOM. The
# selectors are kept because they cost nothing and a build with the classic
# engine (or a future non-CPU graph back-end) makes them live again.
#
# HISTORICAL, on the 5.0 classic engine: OpenCL was NOT a free win but a
# per-model lottery and mostly a LOSS. Median of 8-20 forwards after warm-up:
#   squeezenet1.1-7  0.0036 -> 0.0010 s  (3.6x FASTER)
#   yolo26n-seg      0.222  -> 0.302  s  (0.73x)
#   EAST text        0.059  -> 0.086  s  (0.68x)
#   RAFT             1.685  -> 6.594  s  (0.26x)
#   YuNet            0.0020 -> 0.0276 s  (0.07x - 14x SLOWER)
# Part of the cause was visible in the log: that build's OpenCL kernel source for
# `dnn/activations` failed to compile ("use of undeclared identifier 'std'" -
# a std::pow left in device code), so affected layers fell back per-layer with a
# host<->device copy each way. Always measure the model you actually run.
#
# "auto" stays the default: it skips the setPreferable* calls entirely, which is
# the fastest path for 4 of the 5 models above and keeps the log warning-free.
def _dnn_backends() -> dict:
	"""{label: cv2.dnn.DNN_BACKEND_* or None}, filtered to what this build has."""
	spec = [
		("auto (default backend)", None),
		("OpenCV", "DNN_BACKEND_OPENCV"),
		("CUDA", "DNN_BACKEND_CUDA"),
		("Inference Engine (OpenVINO)", "DNN_BACKEND_INFERENCE_ENGINE"),
		("Vulkan (VKCOM)", "DNN_BACKEND_VKCOM"),
		("WebNN", "DNN_BACKEND_WEBNN"),
		("TIM-VX", "DNN_BACKEND_TIMVX"),
		("CANN", "DNN_BACKEND_CANN"),
	]
	return {label: (None if attr is None else getattr(cv2.dnn, attr))
	        for label, attr in spec if attr is None or hasattr(cv2.dnn, attr)}


def _dnn_targets() -> dict:
	"""{label: cv2.dnn.DNN_TARGET_* or None}, filtered to what this build has."""
	spec = [
		("auto (default target)", None),
		("CPU", "DNN_TARGET_CPU"),
		("CPU FP16", "DNN_TARGET_CPU_FP16"),
		("CUDA", "DNN_TARGET_CUDA"),
		("CUDA FP16", "DNN_TARGET_CUDA_FP16"),
		("OpenCL", "DNN_TARGET_OPENCL"),
		("OpenCL FP16", "DNN_TARGET_OPENCL_FP16"),
		("Vulkan", "DNN_TARGET_VULKAN"),
		("NPU", "DNN_TARGET_NPU"),
	]
	return {label: (None if attr is None else getattr(cv2.dnn, attr))
	        for label, attr in spec if attr is None or hasattr(cv2.dnn, attr)}


# The engine constants are BUILD-SPECIFIC and upstream renamed them mid-flight.
# OpenCV 5.0 shipped ENGINE_CLASSIC (the 4.x per-layer engine) alongside
# ENGINE_NEW (the fused graph engine); 5.1 collapsed the pair into a single
# ENGINE_OPENCV, which IS the graph engine - `//!< Does not support non-CPU
# back-ends for now` in dnn.hpp is ENGINE_NEW's caveat, and classic has no
# successor at all. Every spelling stays in the spec so either build renders a
# correct dropdown; _ENGINE_LEGACY maps a workflow saved on the older build onto
# a label this one has.
def _dnn_engines() -> dict:
	"""{label: cv2.dnn.ENGINE_* or None}, filtered to what this build has."""
	spec = [
		("auto (default engine)", None),
		("classic", "ENGINE_CLASSIC"),
		("new graph", "ENGINE_NEW"),
		("OpenCV (built-in graph)", "ENGINE_OPENCV"),
		("ONNX Runtime", "ENGINE_ORT"),
	]
	return {label: (None if attr is None else getattr(cv2.dnn, attr))
	        for label, attr in spec if attr is None or hasattr(cv2.dnn, attr)}


# {stored label: (label to use instead, the substitution is exact)} for labels a
# build no longer offers. "new graph" -> "auto" is exact on a 5.1 build (auto
# resolves to ENGINE_OPENCV, which is that same graph engine); "classic" is a
# real behaviour change and warns, because the engine it asked for is gone.
_ENGINE_LEGACY = {
	"classic": ("auto (default engine)", False),
	"new graph": ("auto (default engine)", True),
	"OpenCV (built-in graph)": ("new graph", True),
}

_ENGINE_WARNED = set()


def _engine_graph_default() -> str:
	"""Label of the graph engine on this build, for models that only parse there.

	On 5.0 that is the explicit "new graph"; on 5.1 the split is gone and "auto"
	resolves to ENGINE_OPENCV, the same engine. Computed rather than hard-coded so
	the Combo default is always a member of its own options list."""
	engines = _dnn_engines()
	for label in ("new graph", "auto (default engine)"):
		if label in engines:
			return label
	return next(iter(engines))


def _engine_val(label):
	"""Label -> cv2.dnn.ENGINE_* (None for 'auto'), never silently.

	A stored label this build does not know used to fall through `dict.get` to
	None, which is indistinguishable from "auto" - that is how the RAFT blueprint
	spent a whole rebuild running on the very engine its `classic` pin existed to
	avoid. A known-legacy label is remapped and warns once; anything else is a
	wiring error and raises."""
	engines = _dnn_engines()
	if label in engines:
		return engines[label]
	alt, exact = _ENGINE_LEGACY.get(label, (None, True))
	if alt in engines:
		if not exact and label not in _ENGINE_WARNED:
			_ENGINE_WARNED.add(label)
			logging.warning(
				"DNN engine %r is not available in this OpenCV build (%s) - "
				"running on %r instead. This is NOT the same engine: results "
				"that depended on the pin may differ.",
				label, cv2.__version__, alt)
		return engines[alt]
	raise ValueError(
		f"engine={label!r} is not a DNN engine this OpenCV build "
		f"({cv2.__version__}) offers. Pick one of: {', '.join(engines)}.")


class DnnBlobFromImage(io.ComfyNode):
	"""cv2.dnn.blobFromImage/blobFromImages: image(s) -> 4-D NCHW blob."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DnnBlobFromImage",
			display_name="CV DNN Blob From Image",
			category=CATEGORY,
			description="Builds a cv2.dnn input blob from an IMAGE batch or a "
			            "single image: an optional resize, per-channel mean "
			            "subtraction, a scale factor and a BGR<->RGB swap, packed "
			            "into the (N, C, H, W) NCHW float32 tensor a network's "
			            "setInput() expects. Encodes every frame in the batch "
			            "(single image gives N=1). Generic PREPROCESS step - feed "
			            "the blob into 'CV DNN Forward'.",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="IMAGE batch to encode. Every frame is resized and "
					        "encoded; an NPARRAY (gray/BGR/BGRA, any dtype) is "
					        "treated as a single BGR frame."),
				io.Float.Input("scale", default=1.0, min=0.0, max=1e6, step=0.001,
					tooltip="Multiplies every pixel (after mean subtraction). Use "
					        "1/255 = 0.003921 to map 0-255 pixels into the 0-1 range "
					        "most ONNX models expect; 1.0 leaves them at 0-255."),
				io.Int.Input("width", default=0, min=0, max=8192,
					tooltip="Resize width the network wants, in px. 0 = keep the "
					        "image's own width (for fully-convolutional models)."),
				io.Int.Input("height", default=0, min=0, max=8192,
					tooltip="Resize height the network wants, in px. 0 = keep the "
					        "image's own height."),
				io.Boolean.Input("swap_rb", default=True,
					tooltip="Swap the R and B channels. OpenCV frames are BGR but "
					        "most ONNX models are trained on RGB, so leave this on."),
				io.Float.Input("mean_r", default=0.0, min=-255.0, max=255.0, step=0.1,
					tooltip="Value subtracted from the RED channel before scaling "
					        "(0 = none). Set the model's training mean if needed.",
					optional=True, advanced=True),
				io.Float.Input("mean_g", default=0.0, min=-255.0, max=255.0, step=0.1,
					tooltip="Value subtracted from the GREEN channel before scaling "
					        "(0 = none).", optional=True, advanced=True),
				io.Float.Input("mean_b", default=0.0, min=-255.0, max=255.0, step=0.1,
					tooltip="Value subtracted from the BLUE channel before scaling "
					        "(0 = none).", optional=True, advanced=True),
				io.Boolean.Input("crop", default=False,
					tooltip="On: resize so the shorter side fits then centre-crop to "
					        "(width, height). Off: resize the whole image (may change "
					        "aspect ratio).",
					optional=True, advanced=True),
			],
			outputs=[NPArray.Output(display_name="blob",
				tooltip="(N, C, H, W) float32 NCHW blob for the whole batch. "
				        "Feed 'CV DNN Forward'. Empty (0, 3, 1, 1) for an "
				        "empty IMAGE batch.")],
		)

	@classmethod
	def execute(cls, image, scale, width, height, swap_rb,
	            mean_r=0.0, mean_g=0.0, mean_b=0.0, crop=False) -> io.NodeOutput:
		mean = (float(mean_r), float(mean_g), float(mean_b))
		if isinstance(image, torch.Tensor):
			frames = imageutils.image_batch_to_bgr(image)
			if not frames:
				return io.NodeOutput(np.zeros((0, 3, 1, 1), np.float32))
			h0, w0 = frames[0].shape[:2]
			size = (int(width) or w0, int(height) or h0)
			blob = cv2.dnn.blobFromImages(
				frames, float(scale), size, mean,
				swapRB=bool(swap_rb), crop=bool(crop))
			return io.NodeOutput(blob)
		# NPARRAY path: treat as single frame
		frame = _to_bgr_u8(image)
		if frame is None:
			return io.NodeOutput(np.zeros((0, 3, 1, 1), np.float32))
		h0, w0 = frame.shape[:2]
		size = (int(width) or w0, int(height) or h0)
		blob = cv2.dnn.blobFromImage(
			frame, float(scale), size, mean,
			swapRB=bool(swap_rb), crop=bool(crop))
		return io.NodeOutput(blob)


class DnnForward(io.ComfyNode):
	"""cv2.dnn: load an ONNX model and run a forward pass on an input blob."""

	@classmethod
	def define_schema(cls):
		models = _onnx_models()
		return io.Schema(
			node_id="CV_DnnForward",
			display_name="CV DNN Forward",
			category=CATEGORY,
			description="Loads an ONNX model with cv2.dnn.readNetFromONNX and runs "
			            "a forward pass on the input blob. Generic INFERENCE step for "
			            "any ONNX model - pair 'CV DNN Blob From Image' before it "
			            "and 'CV DNN Images From Blob' after it to rebuild "
			            "images. The forward pass runs in the interruptible DNN "
			            "worker. " 
			            "Leave output_layer blank for the model's default output, "
			            "or name a specific layer. The compute backend/target can be "
			            "picked (advanced) for builds that support acceleration.",
			inputs=[
				NPArray.Input("blob",
					tooltip="(N, C, H, W) NCHW input blob from 'CV DNN Blob "
					        "From Image' (or any NCHW float array)."),
				io.Combo.Input("model", options=models,
					tooltip="ONNX model file from ComfyUI/models/onnx to run."),
				io.String.Input("output_layer", default="",
					tooltip="Name of the output layer to fetch. Blank = the model's "
					        "single / default output (right for most models).",
					optional=True, advanced=True),
				io.Combo.Input("backend", options=list(_dnn_backends()),
					default="auto (default backend)",
					tooltip="Compute backend (cv2.dnn.setPreferableBackend). 'auto' "
					        "leaves the net on its default and is right for almost "
					        "every model. Only backends present in the loaded build "
					        "are listed, but PRESENT IS NOT USABLE: this wheel is "
					        "built without CUDA and without OpenVINO, so 'CUDA' and "
					        "'Inference Engine' raise on the classic engine. The "
					        "GRAPH engine ignores this setting entirely (it warns "
					        "'Back-ends are not supported by the new graph engine "
					        "for now'), and only the CLASSIC engine honoured it - "
					        "OpenCV 5.1 removed that engine, so on this build the "
					        "widget has NO effect on any model.",
					optional=True, advanced=True),
				io.Combo.Input("target", options=list(_dnn_targets()),
					default="auto (default target)",
					tooltip="Compute target / device (cv2.dnn.setPreferableTarget). "
					        "'auto' leaves the net on its default (CPU). Honoured on "
					        "the CLASSIC engine only - OpenCV 5.1 removed that "
					        "engine, and this build also reports haveOpenCL() == "
					        "False, so the widget currently does NOTHING (squeezenet "
					        "times identically across every target).\n"
					        "HISTORICAL, on the 5.0 classic engine: OpenCL was a "
					        "per-model lottery and usually a LOSS - squeezenet 3.6x "
					        "faster, but yolo26n-seg 0.73x, EAST 0.68x, RAFT 0.26x "
					        "and YuNet 0.07x (14x SLOWER), partly because that "
					        "build's OpenCL kernel for 'dnn/activations' failed to "
					        "compile and those layers fell back per-layer with a "
					        "device round trip each way.",
					optional=True, advanced=True),
				io.Combo.Input("engine", options=list(_dnn_engines()),
					default="auto (default engine)",
					tooltip="DNN engine passed to cv2.dnn.readNetFromONNX, chosen "
					        "once at model load time. WHICH OPTIONS EXIST DEPENDS ON "
					        "THE BUILD: OpenCV 5.0 offered 'classic' (the 4.x "
					        "per-layer engine) and 'new graph'; 5.1 merged them into "
					        "a single 'OpenCV (built-in graph)' - the graph engine - "
					        "so there 'auto' is the only meaningful choice. 'ONNX "
					        "Runtime' needs a build with WITH_ONNXRUNTIME=ON and "
					        "otherwise falls back to the built-in engine with a "
					        "warning. A workflow saved with an engine this build "
					        "lacks is remapped and logs a warning - it does not "
					        "silently become 'auto'.",
					optional=True, advanced=True),
			],
			outputs=[NPArray.Output(display_name="output",
				tooltip="The raw output blob from the network (NCHW for image "
				        "models). Feed 'CV DNN Images From Blob' to turn an "
				        "image output back into a picture, or 'Inspect CV Data' to "
				        "see its shape.")],
		)

	@classmethod
	def execute(cls, blob, model, output_layer="",
	            backend="auto (default backend)", target="auto (default target)",
	            engine="auto (default engine)") -> io.NodeOutput:
		model_name = model[0] if isinstance(model, list) else model
		out_layer = output_layer[0] if isinstance(output_layer, list) else output_layer
		be = backend[0] if isinstance(backend, list) else backend
		tg = target[0] if isinstance(target, list) else target
		eng = engine[0] if isinstance(engine, list) else engine

		path = _model_path(model_name, "ONNX")
		result, = _run_dnn_worker("DNN Forward", "dnn_forward", {
			"model_path": path,
			"engine": _engine_val(eng),
			"backend_val": _dnn_backends().get(be),
			"target_val": _dnn_targets().get(tg),
			"blob": np.ascontiguousarray(np.asarray(blob, np.float32)),
			"name": str(out_layer).strip()})
		return io.NodeOutput(result)


class DnnImagesFromBlob(io.ComfyNode):
	"""cv2.dnn.imagesFromBlob: NCHW blob -> (N, H, W, C) batch."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DnnImagesFromBlob",
			display_name="CV DNN Images From Blob",
			category=CATEGORY,
			description="Unpacks a (N, C, H, W) NCHW blob back into an "
			            "(N, H, W, C) batch with cv2.dnn.imagesFromBlob - the "
			            "generic POSTPROCESS that undoes the CHW->HWC transpose "
			            "blobFromImage did. Values are returned untouched (still "
			            "the network's output range, e.g. 0-1, and in the blob's "
			            "channel order). For a 3-channel image the output channels "
			            "match the blob (RGB if you swapped R/B going in). Use "
			            "'CV Unstack Batch' to split into individual frames, or "
			            "'CV Index Batch' to pick one by index.",
			inputs=[
				NPArray.Input("blob",
					tooltip="(N, C, H, W) NCHW blob, e.g. an image model's "
					        "output from 'CV DNN Forward'. Any channel count: "
					        "C=1 gray, C=3 colour, C=2 a dense flow field "
					        "((H, W, 2) dx/dy, which is what the flow nodes take)."),
			],
			outputs=[NPArray.Output(display_name="images",
				tooltip="(N, H, W, C) float32 batch in the network's output "
				        "range. Use 'CV Unstack Batch' or 'CV Index Batch' to "
				        "split into individual frames.")],
		)

	@classmethod
	def execute(cls, blob) -> io.NodeOutput:
		arr = np.asarray(blob, np.float32)
		if arr.ndim != 4:
			raise ValueError(
				f"Expected a 4-D (N, C, H, W) blob, got shape {arr.shape}. Use "
				"'Inspect CV Data' to check - only an image model's NCHW output "
				"can be unpacked into images.")
		if arr.shape[0] == 0:
			return io.NodeOutput(np.zeros((0, 0, 0, arr.shape[1]), np.float32))
		imgs = cv2.dnn.imagesFromBlob(np.ascontiguousarray(arr))
		return io.NodeOutput(np.stack([np.asarray(im, np.float32) for im in imgs], axis=0))


class DnnMaskOutput(io.ComfyNode):
	"""Postprocess a segmentation/depth model's raw blob into a ComfyUI MASK."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DnnMaskOutput",
			display_name="CV DNN Mask Output",
			category=CATEGORY,
			description="Converts a segmentation model's raw NCHW output blob into "
			            "a ComfyUI MASK. Squeezes the batch/channel dims, optionally "
			            "applies sigmoid (for models that output logits), resizes to the "
			            "original image size and normalises to 0-1. Pair with 'OpenCV "
			            "DNN Forward' (engine=classic for Swin-Transformer models like "
			            "BiRefNet).",
			inputs=[
				NPArray.Input("blob",
					tooltip="Raw output from 'CV DNN Forward'. Typically "
					        "(1, 1, H, W) for a single-channel segmentation model."),
				io.Combo.Input("sigmoid", options=["auto", "yes", "no"],
					default="auto",
					tooltip="'auto' applies sigmoid only when values lie outside "
					        "[0, 1] (i.e. raw logits). 'yes' always applies it. 'no' "
					        "skips it (model already outputs probabilities)."),
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Original input image, used to read the target resize "
					        "dimensions. If disconnected, 'width' and 'height' widgets "
					        "are used instead (0 = keep the model's native resolution).",
					optional=True),
				io.Int.Input("width", default=0, min=0, max=8192,
					tooltip="Target mask width in px. 0 = keep the model's output "
					        "resolution (or use the connected image's width).",
					optional=True),
				io.Int.Input("height", default=0, min=0, max=8192,
					tooltip="Target mask height in px. 0 = keep the model's output "
					        "resolution (or use the connected image's height).",
					optional=True),
			],
			outputs=[io.Mask.Output(display_name="mask",
				tooltip="Float32 MASK in [0, 1] at the target resolution. "
				        "Feed into 'Overlay Masks' or any mask consumer.")],
		)

	@classmethod
	def execute(cls, blob, image=None, width=0, height=0,
	            sigmoid="auto") -> io.NodeOutput:
		arr = np.asarray(blob, np.float32)
		if arr.size == 0:
			return io.NodeOutput(torch.zeros(0, 1, 1))

		# Squeeze all size-1 dims from the front: (1,1,H,W)->(H,W), (1,H,W)->(H,W).
		while arr.ndim > 2 and arr.shape[0] == 1:
			arr = arr.squeeze(axis=0)
		# Single-channel 3-D: (1,H,W) -> (H,W).
		if arr.ndim == 3 and arr.shape[0] == 1:
			arr = arr.squeeze(axis=0)
		# Multi-channel 3-D (N,H,W) or (H,W,C): take first channel.
		if arr.ndim == 3:
			arr = arr[0] if arr.shape[0] <= arr.shape[-1] else arr[..., 0]

		# Sigmoid activation
		sigmoid = str(sigmoid) if sigmoid else "auto"
		if sigmoid == "yes":
			arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -500, 500)))
		elif sigmoid == "auto":
			if arr.min() < -1e-3 or arr.max() > 1 + 1e-3:
				arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -500, 500)))

		# Determine target size
		if image is not None and isinstance(image, torch.Tensor) and image.numel() > 0:
			# ComfyUI IMAGE: (B, H, W, C)
			tgt_h, tgt_w = int(image.shape[1]), int(image.shape[2])
		else:
			tgt_w = int(width) if width else 0
			tgt_h = int(height) if height else 0

		# Resize if a target was specified and differs from the current size
		if tgt_w > 0 and tgt_h > 0 and (arr.shape[1] != tgt_w or arr.shape[0] != tgt_h):
			arr = cv2.resize(arr, (tgt_w, tgt_h),
				interpolation=cv2.INTER_LINEAR)

		# Clip to [0, 1] and convert to MASK tensor [1, H, W]
		arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
		mask = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
		return io.NodeOutput(mask)


# --- EAST scene-text detection --------------------------------------------------
# Wraps cv2.dnn.TextDetectionModel_EAST (mirrors the east_text_detection_demo
# reference). Unlike the generic DNN pipeline, EAST's high-level API handles
# blobFromImage, forward, decode, and NMS internally. It returns rotated
# rectangles (center, size, angle) and confidences. The node outputs both
# axis-aligned bboxes (for the core 'Draw BBoxes' node) and rotated quads
# (Nx4x2 corner points for 'CV Draw Polygon'). Runs in the interruptible
# DNN worker (workers/dnn_tasks.py 'east_detect').


class EastTextDetect(io.ComfyNode):
	"""EAST scene-text detector (cv2.dnn.TextDetectionModel_EAST)."""

	@classmethod
	def define_schema(cls):
		models = _onnx_models()
		return io.Schema(
			node_id="CV_EastTextDetect",
			display_name="CV EAST Text Detect",
			category=CATEGORY,
		description="Detects scene text with the EAST CNN (cv2.dnn). "
		            "Place the .onnx model in ComfyUI/models/onnx (e.g. "
		            "east_text_detection_2026jul.onnx). Outputs axis-aligned "
		            "bboxes (for the core 'Draw BBoxes' node), rotated quads "
		            "(Nx4x2 corner points for 'CV Draw Polygon'), and "
		            "scores. Zero detections is a valid result (empty outputs).",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Input image. An IMAGE batch is processed frame by frame; "
					        "an NPARRAY (BGR/RGB/gray, any dtype) is treated as one "
					        "frame."),
				io.Combo.Input("model", options=models,
					tooltip="EAST .onnx model from ComfyUI/models/onnx."),
				io.Float.Input("conf_threshold", default=0.5, min=0.0, max=1.0, step=0.01,
					tooltip="Minimum confidence to keep a detection. Lower detects more "
					        "(and more false positives)."),
				io.Float.Input("nms_threshold", default=0.4, min=0.0, max=1.0, step=0.01,
					tooltip="Non-maximum suppression IoU threshold for merging "
					        "overlapping text boxes."),
				io.Int.Input("input_size", default=320, min=320, max=1024, step=32,
					tooltip="Square input size for the network. Larger is more "
					        "accurate but slower. 320 matches the reference demo."),
			],
			outputs=[
				io.BoundingBox.Output(display_name="bboxes",
					tooltip="Axis-aligned {x, y, width, height, score, label} dicts "
					        "- feed the core 'Draw BBoxes' node for a quick preview."),
				NPArray.Output(display_name="quads",
					tooltip="(N, 4, 2) float32 rotated corner points per detection. "
					        "Feed into 'CV Draw Polygon' for precise rotated "
					        "outlines (set thickness=0 for filled polygons or a mask)."),
				NPArray.Output(display_name="scores",
					tooltip="(N,) float32 confidence per detection, same order as "
					        "bboxes. Empty (0,) when nothing is detected."),
				io.Int.Output(display_name="det_count",
					tooltip="Total number of text regions detected."),
			],
		)

	@classmethod
	def execute(cls, image, model, conf_threshold, nms_threshold,
	            input_size) -> io.NodeOutput:
		model_name = model[0] if isinstance(model, list) else model
		path = _model_path(model_name, "EAST text")
		frames = imageutils.frames_from_input(image)
		if not frames:
			return io.NodeOutput([], np.empty((0, 4, 2), np.float32),
			                     np.empty(0, np.float32), 0)

		return io.NodeOutput(*_run_dnn_worker("EAST Text Detect", "east_detect", {
			"model_path": path,
			"frames": [np.ascontiguousarray(f) for f in frames],
			"conf": float(conf_threshold), "nms": float(nms_threshold),
			"size": int(input_size)}))


# ---------------------------------------------------------------------------
# FeatureExtractModel / MatchFeaturesModel
#
# Two separate nodes for model-based feature extraction and matching.
# FeatureExtractModel runs any ONNX feature extractor (DISK, SuperPoint, etc.)
# and outputs CV_KEYPOINTS + descriptors.
# MatchFeaturesModel runs any ONNX matcher model (LightGlue family) and
# outputs CV_MATCHES + point arrays.
#
# Both nodes auto-discover ALL .onnx files from models/onnx (including
# subdirectories) — no path convention enforced, user picks the right model.
# Both run in the interruptible DNN worker (workers/dnn_tasks.py
# 'feature_extract' / 'match_features'); cv2.KeyPoint / cv2.DMatch are not
# picklable, so the worker returns row arrays and this module rebuilds the
# objects before emitting the outputs.
# ---------------------------------------------------------------------------


class FeatureExtractModel(io.ComfyNode):
	"""Run any ONNX feature extractor on an image."""

	@classmethod
	def define_schema(cls):
		all_models = _onnx_models()
		return io.Schema(
			node_id="CV_FeatureExtractModel",
			display_name="CV Feature Extract (Model)",
			category=CATEGORY,
			description="Runs an ONNX feature extractor model (DISK, SuperPoint, "
			            "etc.) on an image and outputs keypoints + descriptors in "
			            "the standard format for 'CV Match Features (Model)' "
			            "and 'CV Draw Matches'.\n\n"
			            "All .onnx files in models/onnx (including subdirectories) "
			            "are listed — pick the right extractor for your matcher. "
			            "Output parsing supports 3-output models "
			            "(e.g. DISK: keypoints (1,N,2) + scores (1,N) + "
			            "descriptors (1,N,D)), flat 2-output models "
			            "((1,N,4) + (1,N,D)), and dense score/descriptor maps "
			            "((1,C,H,W) + (1,C,H,W)).\n\n"
			            "Preprocessing follows the LightGlue reference "
			            "implementation and is read off the MODEL, not guessed: "
			            "pixels are scaled to 0..1, fed as RGB or gray according "
			            "to the channel count the model declares, and padded up "
			            "to a multiple of 16 (DISK's U-Net needs it). Keypoints "
			            "always come back in the INPUT image's pixels.\n\n"
			            "Zero features is a valid result (empty outputs, not an error).",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Image to extract features from (BGR or gray)."),
				io.Combo.Input("model", options=all_models,
					default=all_models[0] if all_models else "",
					tooltip="ONNX feature extractor model from models/onnx "
					        "(e.g. lightglue/disk.onnx, lightglue/superpoint.onnx)."),
				io.Int.Input("max_keypoints", default=2048, min=64, max=16384,
					tooltip="Maximum keypoints to return, keeping the "
					        "highest-scoring ones (the models emit keypoints in "
					        "detection order, not by score)."),
				io.Int.Input("resize_long_side", default=1024, min=0, max=8192,
					tooltip="Resize the image so its LONG side is this many "
					        "pixels before extraction, the way the LightGlue "
					        "reference implementation does (its own default is "
					        "1024). It UPSCALES small images too, which is where "
					        "most of the keypoints on a small photo come from. "
					        "0 keeps the native resolution. Keypoints are mapped "
					        "back to the input image either way."),
				io.Combo.Input("engine", options=list(_dnn_engines()),
					default="auto (default engine)",
					tooltip="DNN engine for inference.",
					optional=True, advanced=True),
			],
			outputs=[
				KeyPoints.Output(display_name="keypoints",
					tooltip="cv2.KeyPoint list."),
				NPArray.Output(display_name="descriptors",
					tooltip="(N, D) float32 descriptors."),
				io.Int.Output(display_name="count",
					tooltip="Number of detected features."),
			],
		)

	@classmethod
	def execute(cls, image, model, max_keypoints, resize_long_side=1024,
	            engine="auto (default engine)") -> io.NodeOutput:
		frame = imageutils.first_bgr_frame(image)
		if frame is None:
			return io.NodeOutput([], np.empty((0, 0), np.float32), 0)

		model_name = str(model).strip() if model else ""
		if not model_name:
			return io.NodeOutput([], np.empty((0, 0), np.float32), 0)

		path = _model_path(model_name, "Feature extractor")
		eng = engine[0] if isinstance(engine, list) else engine
		kp_rows, descs, count = _run_dnn_worker("Feature Extract (Model)",
		                                        "feature_extract", {
			"model_path": path,
			"engine": _engine_val(eng),
			"frame": np.ascontiguousarray(frame),
			"max_keypoints": int(max_keypoints),
			"resize_long_side": int(resize_long_side)})
		kps = _keypoints_from_rows(kp_rows) if len(kp_rows) else []
		return io.NodeOutput(kps, descs, int(count))


class MatchFeaturesModel(io.ComfyNode):
	"""Run an ONNX matcher model (LightGlue family) on two keypoint sets."""

	@classmethod
	def define_schema(cls):
		all_models = _onnx_models()
		return io.Schema(
			node_id="CV_MatchFeaturesModel",
			display_name="CV Match Features (Model)",
			category=CATEGORY,
			description="Matches two sets of keypoints+descriptors using an ONNX "
			            "matcher model (LightGlue family) via cv2.dnn.\n\n"
			            "All .onnx files in models/onnx (including subdirectories) "
			            "are listed — pick the matcher that matches your extractor "
			            "(e.g. lightglue/disk_lightglue.onnx for DISK features).\n\n"
			            "Outputs follow the standard feature infrastructure: "
			            "cv2.DMatch list for 'CV Draw Matches', Nx1x2 point "
			            "arrays for 'CV Find Homography (RANSAC)'.\n\n"
			            "LightGlue reads keypoints in a normalized frame centred "
			            "on the image, and the exported ONNX does NOT do that "
			            "conversion itself - this node does it, from the sizes on "
			            "'space_a'/'space_b' when they are wired, or from the "
			            "keypoints' own extent when they are not (the reference "
			            "implementation's own fallback).\n\n"
			            "Zero matches is a valid result (empty outputs, not an error).",
			inputs=[
				KeyPoints.Input("keypoints_a",
					tooltip="Keypoints of image A (from 'CV Feature Extract (Model)')."),
				NPArray.Input("descriptors_a",
					tooltip="Descriptors of image A, paired with keypoints_a."),
				KeyPoints.Input("keypoints_b",
					tooltip="Keypoints of image B (from 'CV Feature Extract (Model)')."),
				NPArray.Input("descriptors_b",
					tooltip="Descriptors of image B, paired with keypoints_b."),
				io.Combo.Input("model", options=all_models,
					default=all_models[0] if all_models else "",
					tooltip="ONNX matcher model from models/onnx "
					        "(e.g. lightglue/disk_lightglue.onnx)."),
				io.MultiType.Input("space_a",
					polytype.with_latent(polytype.IMAGEISH), optional=True,
					tooltip="Image A itself - only its height/width are read, to "
					        "normalize keypoints_a the way LightGlue expects. "
					        "Leave unconnected to infer the size from the "
					        "keypoints' own extent instead."),
				io.MultiType.Input("space_b",
					polytype.with_latent(polytype.IMAGEISH), optional=True,
					tooltip="Image B itself - only its height/width are read, to "
					        "normalize keypoints_b. Leave unconnected to infer the "
					        "size from the keypoints' own extent instead."),
				io.Combo.Input("engine", options=list(_dnn_engines()),
					default="auto (default engine)",
					tooltip="DNN engine for inference.",
					optional=True, advanced=True),
			],
			outputs=[
				Matches.Output(display_name="matches",
					tooltip="cv2.DMatch list sorted best-first."),
				NPArray.Output(display_name="points_a",
					tooltip="(M, 1, 2) float32 matched point coordinates in A."),
				NPArray.Output(display_name="points_b",
					tooltip="(M, 1, 2) float32 matched point coordinates in B."),
				NPArray.Output(display_name="scores",
					tooltip="(M,) float32 match confidence scores."),
				io.Int.Output(display_name="match_count",
					tooltip="Number of matches found."),
			],
		)

	@classmethod
	def execute(cls, keypoints_a, descriptors_a, keypoints_b, descriptors_b,
	            model, space_a=None, space_b=None,
	            engine="auto (default engine)") -> io.NodeOutput:
		empty = ([], np.zeros((0, 1, 2), np.float32),
		         np.zeros((0, 1, 2), np.float32), np.zeros(0, np.float32), 0)

		kps_a = list(keypoints_a) if keypoints_a else []
		kps_b = list(keypoints_b) if keypoints_b else []
		da = np.asarray(descriptors_a, np.float32) if len(descriptors_a) else np.empty((0, 0), np.float32)
		db = np.asarray(descriptors_b, np.float32) if len(descriptors_b) else np.empty((0, 0), np.float32)

		if not kps_a or not kps_b or da.size == 0 or db.size == 0:
			return io.NodeOutput(*empty)

		model_name = str(model).strip() if model else ""
		if not model_name:
			return io.NodeOutput(*empty)

		path = _model_path(model_name, "Matcher")
		eng = engine[0] if isinstance(engine, list) else engine

		def _wh(space):
			"""(width, height), or None to let the matcher infer it."""
			if space is None:
				return None
			h, w = polytype.spatial_size(space)
			return (int(w), int(h)) if w and h else None

		rows, points_a, points_b, scores, count = _run_dnn_worker(
			"Match Features (Model)", "match_features", {
				"model_path": path,
				"engine": _engine_val(eng),
				"kpts_a_rows": _keypoint_rows(kps_a),
				"kpts_b_rows": _keypoint_rows(kps_b),
				"da": da, "db": db,
				"size_a": _wh(space_a), "size_b": _wh(space_b)})
		dmatches = _dmatches_from_rows(rows) if len(rows) else []
		return io.NodeOutput(dmatches, points_a, points_b, scores, int(count))


# --- Generic DNN detection/segmentation pipeline --------------------------------
# Decomposes a letterboxed detection/segmentation forward pass (the YOLO-seg
# pipeline) into composable steps
# 'DNN Letterbox' (aspect-preserving resize + pad) -> 'DNN
# Blob From Image' (above) -> 'DNN Forward All' (every unconnected output at
# once, extension-dispatched loader so .tflite works too) -> 'DNN Pick Output'
# (select outputs by index, NAME, or shape - model output ORDER stops mattering)
# -> 'YOLO Detect Decode' (unwrap the detection head: four named layouts -
# NMS-free rows, yolov5/v8 class-scores rows/columns, yolov4 separate
# boxes+confs) -> optionally 'YOLO Seg Masks' (instance masks from the kept
# det_rows' coefficients + the proto map; detection-only models simply omit
# it). 'Class Scores Decode' covers classification heads with the same labels
# flow. Class names come from a plain .txt file via 'Load Labels' instead of a
# hard-coded list, so the same chain adapts across the YOLO family and similar
# models - only the model file, layout dropdown and labels change (see workflow
# 70_yolo_seg.json).


class DnnLetterbox(io.ComfyNode):
	"""Aspect-preserving resize + centered pad (Ultralytics LetterBox style)."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DnnLetterbox",
			display_name="CV DNN Letterbox",
			category=CATEGORY,
			description="Resizes an image to fit a square network input while "
			            "preserving aspect ratio, then pads the remainder with a "
			            "constant colour (Ultralytics LetterBox convention, pad "
			            "114). Outputs the padded image plus the resize ratio and "
			            "left/top padding so a downstream decoder (e.g. 'OpenCV "
			            "'CV YOLO Detect Decode' / 'CV YOLO Seg Masks') "
			            "coordinates. Generic PREPROCESS step for letterboxed "
			            "detection/segmentation models - feed the padded image "
			            "into 'CV DNN Blob From Image'. All frames of an IMAGE "
			            "batch share the same size, so one ratio/pad pair applies "
			            "to the whole batch.",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Input image. An IMAGE batch is letterboxed frame by "
					        "frame (all frames share one size, hence one "
					        "ratio/pad); an NPARRAY is treated as a single frame."),
				io.Int.Input("size", default=640, min=32, max=4096, step=32,
					tooltip="Square network input size in pixels (the model's "
					        "imgsz, e.g. 640 for YOLO seg models)."),
				io.Int.Input("pad_value", default=114, min=0, max=255,
					tooltip="Grey value used for the letterbox padding. 114 "
					        "matches the Ultralytics default.",
					optional=True, advanced=True),
			],
			outputs=[
				io.Image.Output(display_name="image",
					tooltip="The letterboxed (size, size) image. Feed 'CV DNN "
					        "Blob From Image'."),
				io.Float.Output(display_name="ratio",
					tooltip="Resize ratio applied (padded_px = orig_px * ratio). "
					        "Feed the decoder's 'ratio' input."),
				io.Int.Output(display_name="pad_left",
					tooltip="Left padding in letterboxed pixels. Feed the "
					        "decoder's 'pad_left' input."),
				io.Int.Output(display_name="pad_top",
					tooltip="Top padding in letterboxed pixels. Feed the "
					        "decoder's 'pad_top' input."),
			],
		)

	@staticmethod
	def _letterbox(frame, size, pad):
		"""(padded, ratio, pad_left, pad_top) - Ultralytics LetterBox rounding."""
		h0, w0 = frame.shape[:2]
		ratio = min(size / h0, size / w0)
		unpad_w, unpad_h = round(w0 * ratio), round(h0 * ratio)
		dw, dh = (size - unpad_w) / 2.0, (size - unpad_h) / 2.0
		top, bottom = round(dh - 0.1), round(dh + 0.1)
		left, right = round(dw - 0.1), round(dw + 0.1)
		if (unpad_w, unpad_h) != (w0, h0):
			frame = cv2.resize(frame, (unpad_w, unpad_h),
			                   interpolation=cv2.INTER_LINEAR)
		padded = cv2.copyMakeBorder(frame, top, bottom, left, right,
		                            cv2.BORDER_CONSTANT, value=(pad, pad, pad))
		return padded, ratio, left, top

	@classmethod
	def execute(cls, image, size, pad_value=114) -> io.NodeOutput:
		frames = imageutils.frames_from_input(image)
		if not frames:
			return io.NodeOutput(torch.zeros(0, 1, 1, 3), 1.0, 0, 0)
		size, pad = int(size), int(pad_value)
		padded, ratio, left, top = [], 1.0, 0, 0
		for i, frame in enumerate(frames):
			p, r, l, t = cls._letterbox(frame, size, pad)
			if i == 0:
				ratio, left, top = r, l, t
			padded.append(p)
		return io.NodeOutput(imageutils.bgr_to_image_batch(padded),
		                     float(ratio), int(left), int(top))


class DnnForwardAll(io.ComfyNode):
	"""cv2.dnn forward pass returning EVERY unconnected output layer."""

	@classmethod
	def define_schema(cls):
		models = _onnx_models()
		return io.Schema(
			node_id="CV_DnnForwardAll",
			display_name="CV DNN Forward All",
			category=CATEGORY,
			description="Loads a model (.onnx via readNetFromONNX, .tflite via "
			            "readNetFromTFLite - picked from the file extension) and "
			            "runs a forward pass returning ALL unconnected output "
			            "layers at once (multi-output models like YOLO seg heads: "
			            "a detection tensor + a mask-prototype map). Generic "
			            "INFERENCE step - pair 'CV DNN Blob From Image' before "
			            "it and 'CV DNN Pick Output' + a decoder after it. "
			            "The forward pass runs in the interruptible DNN worker. "
			            "Leave output_layers blank "
			            "for every unconnected output, or give a comma-separated "
			            "list of layer names. MULTI-INPUT models (an optical-flow "
			            "or change-detection net that takes two frames) are driven "
			            "through input_names: the blob's batch dimension is split "
			            "into one group per name.",
			inputs=[
				NPArray.Input("blob",
					tooltip="(N, C, H, W) NCHW input blob from 'CV DNN Blob "
					        "From Image'."),
				io.Combo.Input("model", options=models,
					tooltip="Model file from ComfyUI/models/onnx (.onnx or "
					        ".tflite; the loader is picked from the extension)."),
				io.String.Input("output_layers", default="",
					tooltip="Comma-separated output layer names to fetch. Blank = "
					        "every unconnected output layer (right for most "
					        "models).",
					optional=True, advanced=True),
				io.Combo.Input("backend", options=list(_dnn_backends()),
					default="auto (default backend)",
					tooltip="Compute backend (cv2.dnn.setPreferableBackend). 'auto' "
					        "leaves the net on its default and is right for almost "
					        "every model. Listed is not usable: this wheel has no "
					        "CUDA and no OpenVINO, so those two raise on the classic "
					        "engine. The GRAPH engine ignores the setting entirely, "
					        "and OpenCV 5.1 removed the classic engine that honoured "
					        "it - so on this build the widget has no effect.",
					optional=True, advanced=True),
				io.Combo.Input("target", options=list(_dnn_targets()),
					default="auto (default target)",
					tooltip="Compute target / device (cv2.dnn.setPreferableTarget). "
					        "'auto' leaves the net on its default (CPU). Honoured on "
					        "the CLASSIC engine only, which OpenCV 5.1 removed, and "
					        "this build reports haveOpenCL() == False - so the "
					        "widget currently does nothing.\n"
					        "HISTORICAL (5.0 classic engine): OpenCL was usually "
					        "SLOWER - squeezenet 3.6x faster, but yolo26n-seg 0.73x, "
					        "EAST 0.68x, RAFT 0.26x, YuNet 0.07x.",
					optional=True, advanced=True),
				io.Combo.Input("engine", options=list(_dnn_engines()),
					default="auto (default engine)",
					tooltip="DNN engine for ONNX models (cv2.dnn.readNetFromONNX). "
					        "Ignored for .tflite files. 'auto' lets OpenCV pick. "
					        "The options depend on the build: OpenCV 5.1 merged "
					        "'classic' and 'new graph' into one engine, so 'auto' is "
					        "the only meaningful choice there, and 'ONNX Runtime' "
					        "falls back to it unless the build has "
					        "WITH_ONNXRUNTIME=ON. NOTE for two-frame models: the "
					        "opencv-zoo RAFT export returns an all-NaN field for "
					        "some input pairs on the graph engine, whatever this "
					        "widget says - 'CV Dense Flow (DNN Model)' recovers most "
					        "of them by re-running with input_names reversed and "
					        "negating the flow.",
					optional=True, advanced=True),
				io.String.Input("input_names", default="",
					tooltip="Comma-separated NETWORK INPUT names for a multi-input "
					        "model. Blank = one unnamed setInput (right for the "
					        "usual single-input model). With K names the blob's "
					        "batch dimension N is split into K equal groups, one "
					        "fed to each named input - so a 2-frame IMAGE batch "
					        "through 'DNN Blob From Image' drives a two-frame "
					        "model with input_names = '0,1' (opencv-zoo RAFT), or "
					        "'image0,image1' / 'left,right' for other exports. N "
					        "must be a multiple of K.\n"
					        "The groups are CONTIGUOUS, so a batch of pairs is "
					        "laid out [all first frames][all second frames], NOT "
					        "interleaved. Whether a batch works at all is the "
					        "MODEL's business: many two-input exports (opencv-zoo "
					        "RAFT among them) freeze batch 1 into their "
					        "constants, and a 2-pair blob then dies in a Concat "
					        "layer with 'Inconsistent shape' - feed one pair per "
					        "execution and loop for more.\n"
					        "All groups share the one blob's preprocessing, so "
					        "this fits models whose inputs are images of the SAME "
					        "size and scaling; a model mixing images with a "
					        "differently-shaped input is out of reach here.",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="outputs",
					tooltip="List of raw output arrays, one per fetched layer, in "
					        "the same order as output_names. Feed 'CV DNN "
					        "Pick Output' to select one, or 'Inspect CV Data'."),
				io.String.Output(display_name="output_names",
					tooltip="Newline-separated names of the fetched layers, "
					        "aligned with 'outputs'. Feed 'Preview as Text'."),
				io.Int.Output(display_name="count",
					tooltip="Number of output layers fetched."),
			],
		)

	@classmethod
	def execute(cls, blob, model, output_layers="",
	            backend="auto (default backend)", target="auto (default target)",
	            engine="auto (default engine)", input_names="") -> io.NodeOutput:
		eng = engine[0] if isinstance(engine, list) else engine
		be = backend[0] if isinstance(backend, list) else backend
		tg = target[0] if isinstance(target, list) else target
		arr = np.ascontiguousarray(np.asarray(blob, np.float32))
		if arr.size == 0 or (arr.ndim >= 1 and arr.shape[0] == 0):
			return io.NodeOutput([], "", 0)

		# A multi-input net is fed by SPLITTING the blob's batch dimension, so the
		# whole pipeline still needs only one preprocess node. Checked here rather
		# than in the worker so the message names the widget the user must fix.
		in_names = [s.strip() for s in str(input_names).split(",") if s.strip()]
		if len(in_names) > 1 and arr.shape[0] % len(in_names):
			raise ValueError(
				f"'DNN Forward All' got input_names={in_names} ({len(in_names)} "
				f"network inputs) but a blob with batch {arr.shape[0]}, which is "
				f"not a multiple of {len(in_names)}. Feed 'DNN Blob From Image' "
				f"an IMAGE batch of {len(in_names)} frames (or a multiple).")

		path = _model_path(model[0] if isinstance(model, list) else model, "DNN")
		outs, names, count = _run_dnn_worker("DNN Forward All", "dnn_forward_all", {
			"model_path": path,
			"engine": _engine_val(eng),
			"backend_val": _dnn_backends().get(be),
			"target_val": _dnn_targets().get(tg),
			"blob": arr,
			"output_layers": str(output_layers),
			"input_names": in_names})
		return io.NodeOutput(outs, names, int(count))


class DnnPickOutput(io.ComfyNode):
	"""Select one array out of a 'DNN Forward All' outputs list."""

	_MODES = ["by index", "by name", "3-D detection head", "4-D prototype map",
	          "4-D map (highest resolution)", "first", "last"]

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DnnPickOutput",
			display_name="CV DNN Pick Output",
			category=CATEGORY,
			description="Selects one output array from the list produced by "
			            "'CV DNN Forward All' - by position ('by index', "
			            "'first', 'last'), by layer NAME ('by name', matched "
			            "against Forward All's output_names, so output ORDER "
			            "stops mattering), or by shape heuristic ('3-D "
			            "detection head' picks the first (1, N, C) tensor, '4-D "
			            "prototype map' the first (1, C, H, W) tensor that is "
			            "not a (..., 1, 4) box tensor). Generic plumbing node: "
			            "chain one per output you need into the matching "
			            "decoder input, or into 'Inspect CV Data'.",
			inputs=[
				NPArray.Input("outputs",
					tooltip="The 'outputs' list from 'CV DNN Forward All'."),
				io.Combo.Input("mode", options=cls._MODES, default="by index",
					tooltip="How to pick: 'by index' uses the index widget; "
					        "'by name' matches the name widget against the "
					        "connected output_names; the shape heuristics pick "
					        "the first output with that number of dimensions, "
					        "except '4-D map (highest resolution)', which picks "
					        "the 4-D output with the largest H*W (the full-res "
					        "flow of a multi-scale model, whatever order the "
					        "engine lists the layers in); 'first'/'last' pick "
					        "the ends."),
				io.Int.Input("index", default=0, min=0, max=63,
					tooltip="Position to pick when mode = 'by index' (0-based, "
					        "matching the output_names order)."),
				io.String.Input("output_names", force_input=True, optional=True,
					tooltip="The 'output_names' STRING from 'CV DNN Forward "
					        "All' (needed for mode = 'by name')."),
				io.String.Input("name", default="",
					tooltip="Layer name to pick when mode = 'by name'. Exact "
					        "match first, then substring match.",
					optional=True),
			],
			outputs=[
				NPArray.Output(display_name="output",
					tooltip="The picked output array."),
				io.Int.Output(display_name="index",
					tooltip="The position the array was picked from (useful when "
					        "a name or shape heuristic did the picking)."),
			],
		)

	@classmethod
	def execute(cls, outputs, mode, index, output_names="", name="") -> io.NodeOutput:
		outs = list(outputs) if outputs is not None else []
		if not outs:
			raise ValueError(
				"'DNN Pick Output' got an empty outputs list - check that 'DNN "
				"Forward All' ran on a non-empty blob.")
		mode = str(mode)
		if mode == "first":
			idx = 0
		elif mode == "last":
			idx = len(outs) - 1
		elif mode == "3-D detection head":
			idx = next((i for i, o in enumerate(outs) if np.ndim(o) == 3), -1)
		elif mode == "4-D prototype map":
			# reject (..., 1, 4) tensors - those are box outputs, not proto maps
			idx = next((i for i, o in enumerate(outs)
			            if np.ndim(o) == 4 and np.shape(o)[-1] != 4), -1)
		elif mode == "4-D map (highest resolution)":
			# Order-independent pick for a multi-SCALE model (RAFT returns the flow
			# at full resolution AND at 1/8): largest H*W of the 4-D outputs, first
			# one on a tie. getUnconnectedOutLayersNames() order is NOT stable -
			# measured: RAFT lists ['12007', '12006'] on the default/new engine and
			# ['12006', '12007'] on classic - so 'by index'/'last' silently pick a
			# different tensor per engine.
			cand = [(np.prod(np.shape(o)[-2:]), -i, i) for i, o in enumerate(outs)
			        if np.ndim(o) == 4]
			idx = max(cand)[2] if cand else -1
		elif mode == "by name":
			if isinstance(output_names, np.ndarray):
				names = [str(s) for s in output_names.tolist()]
			else:
				names = [ln.strip() for ln in str(output_names or "").splitlines()
				         if ln.strip()]
			if not names:
				raise ValueError(
					"'DNN Pick Output' (by name) needs the output_names input "
					"wired from 'DNN Forward All'.")
			target = str(name).strip()
			idx = next((i for i, nm in enumerate(names) if nm == target), -1)
			if idx < 0:
				idx = next((i for i, nm in enumerate(names)
				            if target and target in nm), -1)
			if idx < 0:
				raise ValueError(
					f"'DNN Pick Output' (by name) found no layer '{target}' "
					f"among {names}.")
		else:  # by index
			idx = int(index)
		if idx < 0 or idx >= len(outs):
			raise ValueError(
				f"'DNN Pick Output' ({mode}) found no matching output among "
				f"{len(outs)} outputs (shapes: "
				f"{[tuple(np.shape(o)) for o in outs]}).")
		return io.NodeOutput(np.asarray(outs[idx]), int(idx))


def _register_labels_folder():
	"""Register models/labels with folder_paths so the labels combo can list it.

	ComfyUI does not ship a 'labels' folder mapping by default; add one
	(idempotent) pointing at models/labels.  Runs at import and again inside
	define_schema so the combo is populated even on a cold server."""
	import folder_paths

	if "labels" not in folder_paths.folder_names_and_paths:
		path = os.path.join(folder_paths.models_dir, "labels")
		os.makedirs(path, exist_ok=True)
		folder_paths.folder_names_and_paths["labels"] = ([path], {".txt"})
	elif folder_paths.folder_names_and_paths["labels"][1]:
		paths, exts = folder_paths.folder_names_and_paths["labels"]
		wanted = {".txt"}
		if not wanted <= set(exts):
			folder_paths.folder_names_and_paths["labels"] = (paths, set(exts) | wanted)


def _label_files() -> list:
	import folder_paths

	_register_labels_folder()
	return folder_paths.get_filename_list("labels")


_register_labels_folder()


class LoadLabels(io.ComfyNode):
	"""Read a class-labels .txt file (one label per line)."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_LoadLabels",
			display_name="CV Load Labels",
			category=CATEGORY,
			description="Reads a class-labels text file (one label per line, "
			            "e.g. models/labels/coco-labels-2014_2017.txt) and "
			            "outputs the labels as a newline-joined STRING plus a "
			            "string array. Feed either into a decoder's 'labels' "
			            "input (e.g. 'CV YOLO Detect Decode') so detections are "
			            "named by class instead of by id.",
			inputs=[
				io.Combo.Input("file", options=_label_files(),
					tooltip="Label .txt file from ComfyUI/models/labels. "
					        "Line N (0-based) is the name of class id N."),
			],
			outputs=[
				io.String.Output(display_name="labels",
					tooltip="The labels, newline-joined. Feed a decoder's "
					        "'labels' input (also a multiline widget you can "
					        "paste into directly)."),
				NPArray.Output(display_name="labels_array",
					tooltip="The labels as a (C,) string array, one row per "
					        "class id."),
				io.Int.Output(display_name="count",
					tooltip="Number of labels read."),
			],
		)

	@classmethod
	def execute(cls, file) -> io.NodeOutput:
		import folder_paths

		path = folder_paths.get_full_path("labels", str(file))
		if path is None or not os.path.isfile(path):
			raise ValueError(f"Label file '{file}' not found in models/labels.")
		with open(path, encoding="utf-8") as fh:
			lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
		return io.NodeOutput("\n".join(lines),
		                     np.array(lines, dtype="<U64"), len(lines))


class YoloDetectDecode(io.ComfyNode):
	"""Decode a YOLO-family detection head into bboxes/scores/class ids."""

	_LAYOUTS = [
		"rows: xyxy + conf + class id (NMS-free)",
		"rows: xywh center + class scores (NMS)",
		"columns: xywh center + class scores (NMS)",
		"separate boxes + confs (NMS)",
	]

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_YoloDetectDecode",
			display_name="CV YOLO Detect Decode",
			category=CATEGORY,
			description="Unwraps a YOLO-family detection head into bounding "
			            "boxes, scores, class ids and labels - WITHOUT any mask "
			            "handling, so detection-only models work too (chain "
			            "'CV YOLO Seg Masks' for instance segmentation). "
			            "The 'layout' dropdown covers the common head formats: "
			            "'rows: xyxy + conf + class id' for NMS-free end-to-end "
			            "heads (yolo26/yolov11, rows [x1, y1, x2, y2, conf, "
			            "class_id, ...] in letterboxed pixels, any extra columns "
			            "pass through to det_rows for the mask stage); 'rows/'"
			            "'columns: xywh center + class scores' for yolov5/v8 "
			            "heads (per-class score columns, cxcywh boxes, NMS "
			            "applied); 'separate boxes + confs' for yolov4-style "
			            "ONNX exports (normalized xyxy boxes (1, N, 1, 4) in "
			            "'det' + per-class confidences (1, N, C) wired to "
			            "'confs', NMS applied). Boxes are un-letterboxed back "
			            "to the original image. DATA only - draw with the core "
			            "'Draw BBoxes' node. Zero detections is a valid result "
			            "(found=false, empty outputs). Processes a single image "
			            "(the first frame of a batch).",
			inputs=[
				NPArray.Input("det",
					tooltip="Detection head output. Shape depends on 'layout': "
					        "(1, N, 6+C)/(N, 6+C) rows for the NMS-free layout, "
					        "(N, 4+C) or (1, 4+C, N) for the class-scores "
					        "layouts, (1, N, 1, 4) normalized xyxy for the "
					        "separate-boxes layout."),
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="The ORIGINAL image (before letterboxing) - its size "
					        "is the bbox target size. An IMAGE batch uses its "
					        "first frame."),
				io.Float.Input("ratio", default=1.0, min=1e-6, max=100.0, step=0.001,
					tooltip="Letterbox resize ratio from 'CV DNN Letterbox' "
					        "(1.0 if the image was not letterboxed)."),
				io.Int.Input("pad_left", default=0, min=0, max=4096,
					tooltip="Letterbox left padding from 'CV DNN Letterbox'."),
				io.Int.Input("pad_top", default=0, min=0, max=4096,
					tooltip="Letterbox top padding from 'CV DNN Letterbox'."),
				io.Int.Input("net_size", default=640, min=32, max=4096, step=32,
					tooltip="Square letterboxed input size the model ran at. "
					        "Only used by the 'separate boxes + confs' layout, "
					        "whose boxes are normalized to [0, 1] of it."),
				io.Float.Input("conf_threshold", default=0.5, min=0.0, max=1.0,
					step=0.01,
					tooltip="Minimum confidence to keep a detection. Lower "
					        "detects more (and more false positives)."),
				io.Combo.Input("layout", options=cls._LAYOUTS,
					default=cls._LAYOUTS[0],
					tooltip="The detection head's row/column format (see the "
					        "node description)."),
				NPArray.Input("confs",
					tooltip="Per-class confidence tensor (1, N, C) - REQUIRED "
					        "for the 'separate boxes + confs' layout, ignored "
					        "otherwise.",
					optional=True),
				io.Float.Input("nms_threshold", default=0.45, min=0.0, max=1.0,
					step=0.01,
					tooltip="Non-maximum suppression IoU, used by the NMS "
					        "layouts: boxes overlapping by more than this are "
					        "merged.",
					optional=True, advanced=True),
				io.String.Input("labels", default="", multiline=True,
					tooltip="Class names, one per line (line N = class id N), "
					        "e.g. from 'CV Load Labels'. Blank = detections "
					        "are labelled by their numeric class id.",
					optional=True),
			],
			outputs=[
				io.BoundingBox.Output(display_name="bboxes",
					tooltip="One {x, y, width, height, score, label} dict per "
					        "detection in ORIGINAL-image pixels - feed the core "
					        "'Draw BBoxes' node."),
				NPArray.Output(display_name="scores",
					tooltip="(N,) float32 confidence per detection, same order "
					        "as bboxes."),
				NPArray.Output(display_name="class_ids",
					tooltip="(N,) int32 class id per detection, same order as "
					        "bboxes."),
				NPArray.Output(display_name="labels_out",
					tooltip="(N,) string array of the class label per detection "
					        "(the name from 'labels', or the numeric id). Feed "
					        "'Preview as Text'."),
				NPArray.Output(display_name="det_rows",
					tooltip="(N, 6+K) float32 kept rows in LETTERBOXED pixels "
					        "[x1, y1, x2, y2, conf, class_id, ...extra columns "
					        "from the head - the NMS-free seg layout's mask "
					        "coefficients ride along here]. Feed 'CV YOLO "
					        "Seg Masks'. Empty (0, 6) when nothing is detected."),
				io.Int.Output(display_name="det_count",
					tooltip="Number of detections kept."),
				io.Boolean.Output(display_name="found",
					tooltip="True when at least one detection passed the "
					        "confidence threshold. Feed an 'if/else' node to "
					        "skip downstream visualization when empty."),
			],
		)

	@classmethod
	def execute(cls, det, image, ratio, pad_left, pad_top, net_size,
	            conf_threshold, layout, confs=None, nms_threshold=0.45,
	            labels="") -> io.NodeOutput:
		frame = imageutils.first_bgr_frame(image)
		empty = ([], np.zeros(0, np.float32), np.zeros(0, np.int32),
		         np.array([], dtype="<U16"), np.zeros((0, 6), np.float32),
		         0, False)
		if frame is None:
			return io.NodeOutput(*empty)
		orig_h, orig_w = frame.shape[:2]
		d = np.asarray(det, np.float32)
		layout = str(layout)
		conf = float(conf_threshold)

		if layout == cls._LAYOUTS[0]:
			# NMS-free: rows [x1, y1, x2, y2, conf, class_id, ...extras...]
			rows = d.reshape(-1, d.shape[-1])
			if rows.shape[1] < 6:
				raise ValueError(
					f"'det' rows have {rows.shape[1]} columns - need at least "
					"6 ([x1, y1, x2, y2, conf, class_id]). Not an NMS-free "
					"detection head; check the 'layout' dropdown.")
			keep = rows[:, 4] >= conf
			kept = rows[keep]
			det_rows = kept.astype(np.float32)
			boxes_lb, scores_k, cids_k = (det_rows[:, 0:4], det_rows[:, 4],
			                              det_rows[:, 5].astype(np.int32))
		elif layout in (cls._LAYOUTS[1], cls._LAYOUTS[2]):
			# yolov5/v8: rows or columns [cx, cy, w, h, class scores...]
			arr = d.squeeze()
			if arr.ndim != 2:
				raise ValueError(
					f"'det' must be (N, 4+C) or (1, 4+C, N) for layout "
					f"'{layout}', got shape {d.shape}.")
			rows = arr if layout == cls._LAYOUTS[1] else arr.T
			if rows.shape[1] < 5:
				raise ValueError(
					f"'det' rows have {rows.shape[1]} columns - need at least "
					"5 ([cx, cy, w, h, class scores...]).")
			class_scores = rows[:, 4:]
			cids_all = class_scores.argmax(axis=1)
			scores_all = class_scores[np.arange(len(rows)), cids_all]
			keep = scores_all >= conf
			rows_k = rows[keep]
			scores_k, cids_k = scores_all[keep], cids_all[keep].astype(np.int32)
			cx, cy, bw, bh = (rows_k[:, 0], rows_k[:, 1],
			                  rows_k[:, 2], rows_k[:, 3])
			boxes_lb = np.column_stack([cx - bw / 2, cy - bh / 2,
			                            cx + bw / 2, cy + bh / 2])
			keep_idx = cls._nms(boxes_lb, scores_k, conf, nms_threshold)
			boxes_lb, scores_k, cids_k = (boxes_lb[keep_idx], scores_k[keep_idx],
			                              cids_k[keep_idx])
			det_rows = np.column_stack([boxes_lb, scores_k, cids_k]).astype(np.float32)
		elif layout == cls._LAYOUTS[3]:
			# yolov4-style: normalized xyxy boxes + separate per-class confs
			if confs is None or np.asarray(confs).size == 0:
				raise ValueError(
					"Layout 'separate boxes + confs' needs the 'confs' input "
					"wired (the (1, N, C) confidence output).")
			boxes_n = d.reshape(-1, 4)
			c = np.asarray(confs, np.float32).reshape(len(boxes_n), -1)
			boxes_lb = boxes_n * float(net_size)      # [0,1] -> letterboxed px
			cids_all = c.argmax(axis=1)
			scores_all = c[np.arange(len(c)), cids_all]
			keep = scores_all >= conf
			boxes_lb, scores_k = boxes_lb[keep], scores_all[keep]
			cids_k = cids_all[keep].astype(np.int32)
			keep_idx = cls._nms(boxes_lb, scores_k, conf, nms_threshold)
			boxes_lb, scores_k, cids_k = (boxes_lb[keep_idx], scores_k[keep_idx],
			                              cids_k[keep_idx])
			det_rows = np.column_stack([boxes_lb, scores_k, cids_k]).astype(np.float32)
		else:
			raise ValueError(f"Unknown layout '{layout}'.")

		if len(det_rows) == 0:
			return io.NodeOutput(*empty)

		ratio = float(ratio)
		pad_x, pad_y = int(pad_left), int(pad_top)
		names = [ln.strip() for ln in str(labels or "").splitlines() if ln.strip()]

		def _label(cid):
			return names[cid] if 0 <= cid < len(names) else str(cid)

		boxes, scores, class_ids, label_strs = [], [], [], []
		for i in range(len(det_rows)):
			r = det_rows[i]
			x1 = (r[0] - pad_x) / ratio
			y1 = (r[1] - pad_y) / ratio
			x2 = (r[2] - pad_x) / ratio
			y2 = (r[3] - pad_y) / ratio
			cid = int(r[5])
			name = _label(cid)
			boxes.append(_xyxy_to_box(x1, y1, x2, y2, r[4], name))
			scores.append(float(r[4]))
			class_ids.append(cid)
			label_strs.append(name)

		return io.NodeOutput(boxes,
		                     np.asarray(scores, np.float32),
		                     np.asarray(class_ids, np.int32),
		                     np.array(label_strs, dtype="<U64"),
		                     det_rows.astype(np.float32),
		                     len(boxes), True)

	@staticmethod
	def _nms(boxes_xyxy, scores, conf, nms_threshold):
		"""Indices kept by cv2.dnn.NMSBoxes (all kept when empty)."""
		if len(boxes_xyxy) == 0:
			return np.zeros(0, np.int64)
		xywh = np.column_stack([boxes_xyxy[:, 0], boxes_xyxy[:, 1],
		                        boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
		                        boxes_xyxy[:, 3] - boxes_xyxy[:, 1]])
		idx = cv2.dnn.NMSBoxes(xywh.tolist(), np.asarray(scores).tolist(),
		                       float(conf), float(nms_threshold))
		if len(idx) == 0:
			return np.zeros(0, np.int64)
		return np.asarray(idx).flatten()


class YoloSegMasks(io.ComfyNode):
	"""Build instance masks from kept detection rows + mask prototypes."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_YoloSegMasks",
			display_name="CV YOLO Seg Masks",
			category=CATEGORY,
			description="Builds one binary instance MASK per detection from the "
			            "'det_rows' of 'CV YOLO Detect Decode' (whose extra "
			            "columns carry the mask coefficients) plus the model's "
			            "(1, C, mh, mw) mask-prototype map. Mirrors ultralytics "
			            "ops.process_mask: logits = coeffs @ proto, crop to the "
			            "box in proto space, bilinear-upsample to the "
			            "letterboxed input, threshold at 0 (sigmoid > 0.5), "
			            "then un-letterbox back to the original image. Only the "
			            "NMS-free seg layout carries coefficients - a "
			            "detection-only model or layout raises a clear wiring "
			            "error here. Zero detections is a valid result (empty "
			            "mask batch). DATA only - visualize with 'Overlay "
			            "Masks'. Processes a single image (the first frame of "
			            "a batch).",
			inputs=[
				NPArray.Input("det_rows",
					tooltip="The (N, 6+C) 'det_rows' from 'CV YOLO Detect "
					        "Decode' (NMS-free seg layout): [x1, y1, x2, y2, "
					        "conf, class_id, C mask coefficients] in "
					        "LETTERBOXED pixels."),
				NPArray.Input("proto",
					tooltip="Mask-prototype map, (1, C, mh, mw) or (C, mh, mw). "
					        "C must equal the number of mask coefficients per "
					        "det row."),
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="The ORIGINAL image (before letterboxing) - its "
					        "size is the mask target size. An IMAGE batch uses "
					        "its first frame."),
				io.Float.Input("ratio", default=1.0, min=1e-6, max=100.0, step=0.001,
					tooltip="Letterbox resize ratio from 'CV DNN Letterbox' "
					        "(1.0 if the image was not letterboxed)."),
				io.Int.Input("pad_left", default=0, min=0, max=4096,
					tooltip="Letterbox left padding from 'CV DNN Letterbox'."),
				io.Int.Input("pad_top", default=0, min=0, max=4096,
					tooltip="Letterbox top padding from 'CV DNN Letterbox'."),
				io.Int.Input("net_size", default=640, min=32, max=4096, step=32,
					tooltip="Square letterboxed input size the model ran at "
					        "(the 'size' used in 'CV DNN Letterbox')."),
			],
			outputs=[
				io.Mask.Output(display_name="masks",
					tooltip="(N, H, W) binary instance masks at the original "
					        "image size, aligned row-for-row with the Detect "
					        "Decode bboxes. Feed 'Overlay Masks' or any mask "
					        "consumer."),
				io.Int.Output(display_name="mask_count",
					tooltip="Number of masks built (= the detection count)."),
			],
		)

	@classmethod
	def execute(cls, det_rows, proto, image, ratio, pad_left, pad_top,
	            net_size) -> io.NodeOutput:
		frame = imageutils.first_bgr_frame(image)
		if frame is None:
			return io.NodeOutput(torch.zeros(0, 1, 1), 0)
		orig_h, orig_w = frame.shape[:2]

		rows = np.asarray(det_rows, np.float32).reshape(
			-1, np.asarray(det_rows).shape[-1] if np.asarray(det_rows).size else 6)
		if len(rows) == 0:
			return io.NodeOutput(torch.zeros(0, orig_h, orig_w), 0)

		proto = np.asarray(proto, np.float32).squeeze()
		if proto.ndim != 3:
			raise ValueError(
				f"'proto' must be (1, C, mh, mw) or (C, mh, mw), got shape "
				f"{np.asarray(proto).shape}. Check the 'DNN Pick Output' mode.")
		c_proto, mh, mw = proto.shape
		n_coeff = rows.shape[1] - 6
		if n_coeff <= 0:
			raise ValueError(
				"'det_rows' carries no mask coefficients (rows have "
				f"{rows.shape[1]} columns, need 6 + C). Only the NMS-free seg "
				"layout passes coefficients through - a detection-only model/"
				"layout cannot feed 'YOLO Seg Masks'.")
		if n_coeff != c_proto:
			raise ValueError(
				f"'det_rows' carries {n_coeff} mask coefficients per row but "
				f"'proto' has {c_proto} channels - mismatched model outputs. "
				"Check the 'DNN Pick Output' modes.")

		ratio = float(ratio)
		pad_x, pad_y = int(pad_left), int(pad_top)
		net_w = net_h = int(net_size)

		# instance masks (mirrors ultralytics ops.process_mask + crop_mask)
		coeffs = rows[:, 6:].astype(np.float32)
		boxes_lb = rows[:, 0:4]
		logits = coeffs @ proto.reshape(c_proto, mh * mw)   # N x (mh*mw)
		wr, hr = mw / net_w, mh / net_h
		content_w = min(net_w, max(0, round(orig_w * ratio)))
		content_h = min(net_h, max(0, round(orig_h * ratio)))
		cols = np.arange(mw)[None, :]
		rows_idx = np.arange(mh)[:, None]
		masks = []
		for i in range(logits.shape[0]):
			m = logits[i].reshape(mh, mw).copy()
			bx1, by1 = boxes_lb[i][0] * wr, boxes_lb[i][1] * hr
			bx2, by2 = boxes_lb[i][2] * wr, boxes_lb[i][3] * hr
			crop = ((cols >= bx1) & (cols < bx2)
			        & (rows_idx >= by1) & (rows_idx < by2))
			m[~crop] = 0.0
			up = cv2.resize(m, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
			binm = (up > 0.0).astype(np.uint8)
			content = binm[pad_y:pad_y + content_h, pad_x:pad_x + content_w]
			full = cv2.resize(content, (orig_w, orig_h),
			                  interpolation=cv2.INTER_NEAREST)
			masks.append(full.astype(np.float32))

		mask_t = torch.from_numpy(np.stack(masks, axis=0))
		return io.NodeOutput(mask_t, len(masks))


class ClassScoresDecode(io.ComfyNode):
	"""Decode a classification score vector into top-k labels + scores."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ClassScoresDecode",
			display_name="CV Class Scores Decode",
			category=CATEGORY,
			description="Unwraps a classification model's score vector ((1, C) "
			            "or (C,) from 'CV DNN Pick Output') into the top-k "
			            "class labels, scores and ids. Class names come from "
			            "the same labels text as the detectors ('CV Load "
			            "Labels', line N = class id N). DATA only - feed "
			            "'Preview as Text' to see the ranking.",
			inputs=[
				NPArray.Input("scores",
					tooltip="(1, C) or (C,) per-class scores/logits from the "
					        "model (pick it out of 'DNN Forward All' with 'DNN "
					        "Pick Output')."),
				io.Int.Input("top_k", default=5, min=1, max=1000,
					tooltip="How many of the highest-scoring classes to emit."),
				io.Combo.Input("activation",
					options=["auto", "softmax", "sigmoid", "none"],
					default="auto",
					tooltip="'auto' applies softmax only when the values lie "
					        "outside [0, 1] (i.e. raw logits). 'softmax'/'sigmoid' "
					        "always apply, 'none' never (model already outputs "
					        "probabilities).",
					optional=True, advanced=True),
				io.String.Input("labels", default="", multiline=True,
					tooltip="Class names, one per line (line N = class id N), "
					        "e.g. from 'CV Load Labels'. Blank = classes "
					        "are labelled by their numeric id.",
					optional=True),
			],
			outputs=[
				NPArray.Output(display_name="labels_out",
					tooltip="(K,) string array of the top-k class labels, best "
					        "first. Feed 'Preview as Text'."),
				NPArray.Output(display_name="scores_top",
					tooltip="(K,) float32 scores of the top-k classes (after "
					        "the chosen activation), best first."),
				NPArray.Output(display_name="class_ids",
					tooltip="(K,) int32 class ids of the top-k classes."),
				io.String.Output(display_name="top_label",
					tooltip="The single best class label (top-1)."),
			],
		)

	@classmethod
	def execute(cls, scores, top_k, activation="auto", labels="") -> io.NodeOutput:
		arr = np.asarray(scores, np.float32).squeeze().reshape(-1)
		if arr.size == 0:
			return io.NodeOutput(np.array([], dtype="<U16"),
			                     np.zeros(0, np.float32),
			                     np.zeros(0, np.int32), "")
		activation = str(activation or "auto")
		if activation == "softmax" or (activation == "auto"
		                               and (arr.min() < -1e-6 or arr.max() > 1 + 1e-6)):
			e = np.exp(arr - arr.max())
			probs = e / e.sum()
		elif activation == "sigmoid":
			probs = 1.0 / (1.0 + np.exp(-np.clip(arr, -500, 500)))
		else:  # 'none', or 'auto' with values already in [0, 1]
			probs = arr

		k = max(1, min(int(top_k), arr.size))
		idx = np.argsort(probs)[::-1][:k]
		names = [ln.strip() for ln in str(labels or "").splitlines() if ln.strip()]
		label_list = [names[i] if 0 <= i < len(names) else str(i) for i in idx]
		return io.NodeOutput(np.array(label_list, dtype="<U64"),
		                     probs[idx].astype(np.float32),
		                     idx.astype(np.int32),
		                     label_list[0])


# ---------------------------------------------------------------------------
# OpenCV 5 LLM / VLM support: dnn.Tokenizer + the new graph engine's KV-cache
# routing (net.enableKVCache() feeds the graph's present.* outputs back into
# its past_key_values.* inputs). Generation is a stateful loop, so it runs in
# the interruptible DNN worker (workers/dnn_tasks.py 'llm_generate' /
# 'vlm_generate' / 'seq2seq_generate' - the tokenizer is not picklable, so
# tokenization/detokenization happen inside the worker); model file, template
# and decode mode are
# dropdown choices so the same nodes cover the whole supported family.
# ---------------------------------------------------------------------------


def _register_llm_folder():
	"""Register models/llm with folder_paths under the 'LLM' category (the
	name extra_model_paths.yaml uses). Idempotent; only adds a fallback when
	nothing registered the category.

	The extension set is left EMPTY on purpose - folder_paths then lists every
	file, which is exactly what lets _llm_listing() see each model folder's
	config.json / tokenizer.json for free (see _llm_bundles). Anywhere a model
	FILE is wanted we filter to .onnx ourselves, so a GGUF or a tokenizer.json
	never becomes a model choice. This also matches how the yaml registration
	behaves, so the fallback and the real install agree."""
	import folder_paths

	if "LLM" not in folder_paths.folder_names_and_paths:
		path = os.path.join(folder_paths.models_dir, "llm")
		os.makedirs(path, exist_ok=True)
		folder_paths.folder_names_and_paths["LLM"] = ([path], set())


_register_llm_folder()

# Combo sentinels. _AUTO derives a part from the chosen model folder; _NONE is
# both "this optional part is unused" and the empty-install placeholder.
_AUTO = "auto (from model folder)"
_NONE = "[none]"


def _combo(options, *lead) -> list:
	"""Option list for a model/tokenizer dropdown, never empty (ComfyUI cannot
	render a combo with no options, and a fresh install has no models)."""
	opts = [*lead, *options]
	return opts if opts else [_NONE]


def _llm_listing() -> list:
	"""Every file under the LLM roots, '/'-separated relative names.

	ONE folder_paths.get_filename_list call, which ComfyUI already caches per
	folder mtime - _llm_models / _tokenizer_dirs / _llm_bundles all derive from
	it, so building these schemas costs no extra disk I/O."""
	import folder_paths

	_register_llm_folder()
	return [f.replace(os.sep, "/") for f in folder_paths.get_filename_list("LLM")]


def _llm_models() -> list:
	"""All .onnx files under the LLM roots ('/'-separated relative names, so
	saved workflows stay portable across machines)."""
	return sorted(f for f in _llm_listing() if f.lower().endswith(".onnx"))


def _tokenizer_dirs() -> list:
	"""Folders holding BOTH an OpenCV-format config.json and a tokenizer.json,
	i.e. every folder cv2.dnn.Tokenizer_load can be pointed at."""
	seen: dict = {}
	for f in _llm_listing():
		d, _, base = f.rpartition("/")
		if base.lower() in ("config.json", "tokenizer.json"):
			seen.setdefault(d or ".", set()).add(base.lower())
	return sorted(d for d, got in seen.items() if len(got) == 2)


def _llm_bundles(files=None, toks=None) -> list:
	"""The model folders offered in the 'model' dropdown: every directory that
	directly holds at least one .onnx, collapsed onto its nearest ancestor that
	is a tokenizer folder.

	That collapse is what turns a multi-part export into ONE choice instead of
	one per file - paligemma2-3b-pt-224/{siglip,embedding,gemma}/*.onnx all fold
	into paligemma2-3b-pt-224, which is also where its tokenizer pair lives. It
	falls out of the layout rather than special-casing any model.

	'files'/'toks' are injectable so the collapse can be tested without models
	installed."""
	toks = set(_tokenizer_dirs() if toks is None else toks)
	bundles = set()
	for f in (_llm_models() if files is None else files):
		d, _, _base = f.rpartition("/")
		d = d or "."
		parts = d.split("/") if d != "." else []
		for i in range(len(parts) - 1, 0, -1):   # nearest tokenizer ancestor
			anc = "/".join(parts[:i])
			if anc in toks:
				d = anc
				break
		bundles.add(d)
	return sorted(bundles)


# Ordered filename patterns per role, matched as case-folded substrings of each
# .onnx's path RELATIVE TO THE MODEL FOLDER - so a part sitting in a subfolder
# can match on the folder name too ('siglip/vision_model.onnx' -> vision).
_ROLE_PATTERNS = {
	"lm":      ("decoder_model_merged", "decoder_model", "gemma", "model"),
	"encoder": ("encoder_model", "encoder"),
	"decoder": ("decoder_model_merged", "decoder_model", "decoder"),
	"vision":  ("vision_model", "vision_encoder", "vision", "siglip", "vit"),
	"embed":   ("embed_tokens", "embedding", "embed"),
}


def _bundle_files(bundle, files=None) -> list:
	"""The .onnx files inside a model folder, as full '/'-relative names."""
	if files is None:
		files = _llm_models()
	if bundle == ".":
		return [f for f in files if "/" not in f]
	pre = bundle + "/"
	return [f for f in files if f.startswith(pre)]


def _resolve_role(bundle, role, files=None) -> str:
	"""The .onnx inside 'bundle' that plays 'role'.

	A folder with a single .onnx uses it whatever the role; otherwise the first
	_ROLE_PATTERNS entry matching EXACTLY one file wins. No match - or an
	ambiguous one - is a wiring/config mistake rather than a detection failure,
	so it raises (with the candidates listed); the per-part override widget is
	the escape hatch for a layout these patterns do not cover."""
	inside = _bundle_files(bundle, files)
	if not inside:
		raise ValueError(f"Model folder '{bundle}' holds no .onnx file.")
	cut = 0 if bundle == "." else len(bundle) + 1
	if len(inside) == 1:
		return inside[0]
	for pat in _ROLE_PATTERNS.get(role, ()):
		hits = [f for f in inside if pat in f[cut:].lower()]
		if len(hits) == 1:
			return hits[0]
	raise ValueError(
		f"Cannot tell which .onnx in '{bundle}' is the {role} model - set the "
		f"'{role}_file' override. Candidates: "
		+ ", ".join(f[cut:] for f in inside))


def _bundle_tokenizer(bundle, toks=None):
	"""The tokenizer folder belonging to a model folder: the folder itself when
	it holds the config.json + tokenizer.json pair, else its nearest ancestor
	that does. None when the family ships no tokenizer at all (ViT-GPT2 borrows
	gpt2's - that is what the 'tokenizer' override is for)."""
	if toks is None:
		toks = set(_tokenizer_dirs())
	parts = [] if bundle == "." else bundle.split("/")
	for i in range(len(parts), 0, -1):
		cand = "/".join(parts[:i])
		if cand in toks:
			return cand
	return "." if "." in toks else None


def _pick(value):
	"""Combo values sometimes arrive wrapped in a list; unwrap and strip."""
	return str(value[0] if isinstance(value, list) else value).strip()


def _llm_roots() -> list:
	import folder_paths

	_register_llm_folder()
	return [os.path.realpath(p) for p in folder_paths.get_folder_paths("LLM")]


def _within(path, roots) -> bool:
	return any(path == r or path.startswith(r + os.sep) for r in roots)


def _llm_dir_path(rel, what="tokenizer") -> str:
	"""Absolute path of a '/'-relative FOLDER name inside the LLM roots.

	Combo options only constrain the frontend - the /prompt API takes whatever
	string it is handed - so this is where the value is actually contained: the
	resolved realpath must sit inside a registered LLM root. That is what stops
	a shared workflow from pointing cv2.dnn.Tokenizer_load at an arbitrary
	config.json elsewhere on the machine."""
	name = _pick(rel)
	if not name or name in (_NONE, _AUTO):
		raise ValueError(f"No {what} folder selected.")
	roots = _llm_roots()
	for root in roots:
		cand = os.path.realpath(os.path.join(root, name.replace("/", os.sep)))
		if os.path.isdir(cand) and _within(cand, roots):
			return cand
	raise ValueError(
		f"{what} folder '{name}' is not inside models/llm. Pick one of the "
		"listed folders - paths outside the registered model roots are "
		"rejected on purpose.")


def _llm_model_path(name) -> str:
	"""Absolute path of a '/'-relative .onnx name, contained like the above."""
	import folder_paths

	n = _pick(name)
	path = folder_paths.get_full_path("LLM", n)
	if path is None and os.sep != "/":
		path = folder_paths.get_full_path("LLM", n.replace("/", os.sep))
	if path is None or not os.path.isfile(path):
		raise ValueError(f"Model '{n}' not found in models/llm.")
	real = os.path.realpath(path)
	if not _within(real, _llm_roots()):
		raise ValueError(f"Model '{n}' resolves outside models/llm.")
	return real


def _bundle_name(bundle) -> str:
	b = _pick(bundle)
	if not b or b == _NONE:
		raise ValueError("No model folder selected - put an ONNX export under "
		                 "models/llm/<family>/ and reload the node.")
	return b


def _part_path(bundle, role, override, files=None):
	"""Absolute path of one model part: the override when the user set one,
	else the file auto-resolved from the model folder. None when an optional
	part is left at [none]."""
	ov = _pick(override)
	if ov == _NONE:
		return None
	if ov and ov != _AUTO:
		return _llm_model_path(ov)
	return _llm_model_path(_resolve_role(bundle, role, files))


def _tokenizer_dir_for(bundle, tokenizer) -> str:
	"""Absolute tokenizer folder for a model folder, override taking priority."""
	ov = _pick(tokenizer)
	if ov and ov not in (_AUTO, _NONE):
		return _llm_dir_path(ov)
	rel = _bundle_tokenizer(bundle)
	if rel is None:
		raise ValueError(
			f"Model folder '{bundle}' has no config.json + tokenizer.json - set "
			"the 'tokenizer' override to a folder that does (ViT-GPT2 borrows "
			"the gpt2 one).")
	return _llm_dir_path(rel)


_tokenizer_cache: dict = {}


def _load_tokenizer(tokenizer_dir):
	"""cv2.dnn.Tokenizer.load for a model folder (config.json + tokenizer.json
	inside it), cached per directory. Takes an ALREADY-RESOLVED absolute path
	from _tokenizer_dir_for; the loader itself wants the config.json path and
	derives the folder from it, so pass that through when handed one."""
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


def _parse_float_list(text, channels=3):
	"""'0.5' -> (0.5, 0.5, 0.5); '0.485, 0.456, 0.406' -> per-channel tuple."""
	parts = [float(t) for t in str(text).replace(";", ",").split(",") if t.strip()]
	if len(parts) == 1:
		return tuple(parts * channels)
	if len(parts) != channels:
		raise ValueError(f"'{text}' must be one value or {channels} "
		                 "comma-separated values.")
	return tuple(parts)


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


class DnnTokenizer(io.ComfyNode):
	"""cv2.dnn.Tokenizer: text <-> token ids around an LLM model folder."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_DnnTokenizer",
			display_name="CV DNN Tokenizer",
			category=CATEGORY,
			description="Encodes text to token ids or decodes ids back to text "
			            "with OpenCV 5's built-in tokenizer (cv2.dnn.Tokenizer: "
			            "BPE for the gpt2/gpt4/qwen2.5 families, SentencePiece "
			            "for Gemma). Lists every folder under models/llm holding "
			            "an OpenCV-format config.json plus the model family's "
			            "tokenizer.json (e.g. opencv_extra testdata/dnn/llm/"
			            "<family>/). The 'CV LLM/VLM/Seq2Seq Generate' nodes "
			            "tokenize internally - use this node to inspect tokens or "
			            "to build prompts token by token.",
			inputs=[
				io.Combo.Input("mode", options=["encode (text -> ids)",
				                                "decode (ids -> text)"],
					tooltip="encode: 'text' -> token_ids. decode: the wired "
					        "'token_ids' -> text."),
				io.Combo.Input("tokenizer", options=_combo(_tokenizer_dirs()),
					tooltip="Model folder under models/llm holding the family's "
					        "config.json + tokenizer.json (e.g. opencv/qwen2.5-"
					        "0.5b-instruct). Relative to the model root, so the "
					        "workflow stays portable."),
				io.String.Input("text", default="What is OpenCV?", multiline=True,
					tooltip="Text to encode (encode mode).",
					optional=True, advanced=True),
				NPArray.Input("token_ids",
					tooltip="(N,) or (1, N) integer token ids to decode "
					        "(decode mode).",
					optional=True, advanced=True),
			],
			outputs=[
				NPArray.Output(display_name="token_ids",
					tooltip="(1, N) int32 token ids (encode mode; empty (1, 0) "
					        "in decode mode)."),
				io.String.Output(display_name="text",
					tooltip="Decoded text (decode mode; the input text echoed "
					        "in encode mode)."),
				io.Int.Output(display_name="count",
					tooltip="Number of tokens encoded / decoded."),
			],
		)

	@classmethod
	def execute(cls, mode, tokenizer, text="What is OpenCV?",
	            token_ids=None) -> io.NodeOutput:
		tok = _load_tokenizer(_llm_dir_path(tokenizer))
		if mode.startswith("encode"):
			ids = list(tok.encode(str(text)))
			arr = np.array([ids], np.int32) if ids else np.zeros((1, 0), np.int32)
			return io.NodeOutput(arr, str(text), len(ids))
		if token_ids is None:
			raise ValueError("decode mode needs token_ids wired in.")
		ids = np.asarray(token_ids).reshape(-1).astype(np.int64).tolist()
		return io.NodeOutput(np.zeros((1, 0), np.int32),
		                     tok.decode(ids), len(ids))


# Prompt templates for decoder-only LLMs: wrap, default stop ids, default BOS.
_LLM_TEMPLATES = {
	"none (raw prompt)": (
		lambda p: p, (), None),
	"chatml (Qwen2.5)": (
		lambda p: "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n",
		(151645, 151643), None),          # <|im_end|>, <|endoftext|>
	"gemma3 (Gemma 3)": (
		lambda p: "<start_of_turn>user\n" + p + "<end_of_turn>\n"
		          "<start_of_turn>model\n",
		(1, 106), 2),                     # <eos>, <end_of_turn>; <bos>
}


class LLMGenerate(io.ComfyNode):
	"""Decoder-only LLM text generation (Qwen2.5 / Gemma3 / GPT-2 ONNX)."""

	@classmethod
	def define_schema(cls):
		models = _llm_models()
		return io.Schema(
			node_id="CV_LLMGenerate",
			display_name="CV LLM Generate",
			category=CATEGORY,
			description="Generates text with a decoder-only LLM loaded by "
			            "OpenCV 5's new DNN engine (cv2.dnn.Tokenizer + "
			            "KV-cache): Qwen2.5/Gemma3 instruct exports or GPT-2, "
			            "as a model folder under models/llm (e.g. optimum-cli "
			            "export onnx --task causal-lm-with-past, or the merged "
			            "decoder of a transformers.js export). The chat "
			            "template, stop ids and decode mode are dropdown "
			            "choices - nothing is baked to one model. Greedy "
			            "decoding (deterministic). Mirror of the opencv samples "
			            "qwen/gemma3/gpt2_inference.py. Generation runs in the "
			            "interruptible DNN worker (tokenization + decoding "
			            "included).",
			inputs=[
				io.Combo.Input("model", options=_combo(_llm_bundles()),
					tooltip="Model FOLDER under models/llm (e.g. opencv/qwen2.5-"
					        "0.5b-instruct). Its .onnx and its tokenizer are "
					        "found inside it; override either below if the "
					        "layout is unusual. KV-cache mode needs a with-past/"
					        "merged export; a plain causal-lm export only works "
					        "in 'full sequence' mode."),
				io.Combo.Input("template", options=list(_LLM_TEMPLATES),
					default="chatml (Qwen2.5)",
					tooltip="Prompt wrapper. chatml: Qwen2.5-Instruct "
					        "(<|im_start|>user ...). gemma3: Gemma 3 instruct "
					        "(<start_of_turn>user ..., prepends BOS). "
					        "'none': raw completion (GPT-2). The template also "
					        "sets the default stop/BOS ids."),
				io.String.Input("prompt", default="What is OpenCV?",
					multiline=True,
					tooltip="User prompt, wrapped by 'template'."),
				io.Int.Input("max_new_tokens", default=32, min=1, max=2048,
					tooltip="Maximum tokens to generate. CPU inference is "
					        "roughly a few tokens/second for a 0.5B model - "
					        "keep this small."),
				io.Combo.Input("decode_mode",
					options=["kv-cache (with-past export)",
					         "full sequence (no-past export)"],
					tooltip="kv-cache: fast, one token per step through the "
					        "graph, needs a causal-lm-with-past / merged export. "
					        "full sequence: re-feeds the whole growing sequence "
					        "every step (slow, but works with plain exports)."),
				io.Combo.Input("lm_file", options=_combo(models, _AUTO),
					default=_AUTO,
					tooltip="Override which .onnx in the model folder is the "
					        "language model. 'auto' picks the merged/only "
					        "decoder export - set this only when the folder "
					        "holds several and auto cannot tell.",
					optional=True, advanced=True),
				io.Combo.Input("tokenizer", options=_combo(_tokenizer_dirs(), _AUTO),
					default=_AUTO,
					tooltip="Override the tokenizer folder. 'auto' uses the "
					        "model folder's own config.json + tokenizer.json "
					        "(or the nearest parent holding them); set this for "
					        "an export that ships no tokenizer of its own.",
					optional=True, advanced=True),
				io.String.Input("stop_token_ids", default="",
					tooltip="Comma-separated ids that end generation. Blank = "
					        "the template default (Qwen2.5: 151645, 151643; "
					        "Gemma3: 1, 106; GPT-2: 50256 suggested).",
					optional=True, advanced=True),
				io.Int.Input("bos_token_id", default=-1, min=-1, max=1000000,
					tooltip="Token prepended to the encoded prompt. -1 = the "
					        "template default (Gemma3 prepends 2; the others "
					        "none).",
					optional=True, advanced=True),
				io.Combo.Input("engine", options=list(_dnn_engines()),
					default=_engine_graph_default(),
					tooltip="DNN engine for cv2.dnn.readNetFromONNX. LLM ONNX "
					        "graphs need the OpenCV 5 GRAPH engine; the "
					        "classic 4.x engine cannot parse them. OpenCV 5.1 "
					        "merged the two into one engine, so 'auto' is the "
					        "graph engine there and this default follows the "
					        "build.",
					optional=True, advanced=True),
			],
			outputs=[
				io.String.Output(display_name="response",
					tooltip="The generated text only (prompt stripped, stop "
					        "tokens removed)."),
				io.String.Output(display_name="full_text",
					tooltip="Prompt + response, detokenized."),
				io.Int.Output(display_name="new_tokens",
					tooltip="Number of tokens generated (before stop-id "
					        "removal)."),
			],
		)

	@classmethod
	def execute(cls, model, template, prompt, max_new_tokens, decode_mode,
	            lm_file=_AUTO, tokenizer=_AUTO, stop_token_ids="",
	            bos_token_id=-1, engine="new graph") -> io.NodeOutput:
		wrap, def_stops, def_bos = _LLM_TEMPLATES[_pick(template)]
		stops = _parse_id_list(stop_token_ids, def_stops)
		bundle = _bundle_name(model)
		bos = def_bos if int(bos_token_id) < 0 else int(bos_token_id)
		response, full_text, new_tokens = _run_dnn_worker(
			"LLM Generate", "llm_generate", {
				"model_path": _part_path(bundle, "lm", lm_file),
				"engine": _engine_val(_pick(engine)),
				"tokenizer_dir": _tokenizer_dir_for(bundle, tokenizer),
				"text": wrap(str(prompt)),
				"stops": stops,
				"max_new_tokens": int(max_new_tokens),
				"use_kv": _pick(decode_mode).startswith("kv-cache"),
				"bos": bos})
		return io.NodeOutput(response, full_text, int(new_tokens))


class VLMGenerate(io.ComfyNode):
	"""Decoder-only VLM (PaliGemma shape: vision encoder + embedding + LM)."""

	@classmethod
	def define_schema(cls):
		models = _llm_models()
		return io.Schema(
			node_id="CV_VLMGenerate",
			display_name="CV VLM Generate",
			category=CATEGORY,
			description="Answers a text prompt about an image with a "
			            "decoder-only VLM of the PaliGemma shape: a model "
			            "folder under models/llm holding three .onnx parts - a "
			            "vision encoder (image -> image-feature tokens), a "
			            "token embedding (prompt ids -> text embeddings) and a "
			            "language model ([image features | text embeddings] -> "
			            "logits), each found automatically inside the folder. "
			            "Mirrors the opencv vlm_inference.py sample "
			            "(paligemma2-3b-pt-224, prompt 'cap en\\n' to caption). "
			            "Greedy decoding; runs in the interruptible DNN worker. "
			            "The vision "
			            "encoder output is CONCATENATED before the prompt "
			            "embeddings, PaliGemma-style - VLMs with a different "
			            "fusion need the seq2seq node or their own subgraph.",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="The image to ask about (first frame of a batch). "
					        "Resized to image_size x image_size."),
				io.Combo.Input("model", options=_combo(_llm_bundles()),
					tooltip="Model FOLDER under models/llm (e.g. opencv/"
					        "paligemma2-3b-pt-224). Its vision / embedding / "
					        "language parts and its tokenizer are found inside "
					        "it - override any of them below if the layout is "
					        "unusual."),
				io.String.Input("prompt", default="cap en\n", multiline=True,
					tooltip="Task prompt ('cap en\\n' = caption in English for "
					        "PaliGemma; a question for instruct VLMs)."),
				io.Int.Input("max_new_tokens", default=32, min=1, max=1024,
					tooltip="Maximum tokens to generate. NOTE: 'full "
					        "sequence' mode re-runs the whole prefix every "
					        "step - a 3B fp32 LM takes ~a minute per token on "
					        "CPU, keep this small."),
				io.Combo.Input("vision_file", options=_combo(models, _AUTO),
					default=_AUTO,
					tooltip="Override the vision encoder .onnx (input "
					        "'pixel_values', e.g. PaliGemma2's SigLIP "
					        "vision_model.onnx -> (1, 256, 2304) image-feature "
					        "tokens). 'auto' finds it in the model folder.",
					optional=True, advanced=True),
				io.Combo.Input("embed_file", options=_combo(models, _AUTO),
					default=_AUTO,
					tooltip="Override the token embedding .onnx (input "
					        "'input_ids' -> embedding rows, e.g. "
					        "embedding.onnx).",
					optional=True, advanced=True),
				io.Combo.Input("lm_file", options=_combo(models, _AUTO),
					default=_AUTO,
					tooltip="Override the language model .onnx (input "
					        "'inputs_embeds' -> logits, e.g. gemma2_3b.onnx).",
					optional=True, advanced=True),
				io.Combo.Input("tokenizer", options=_combo(_tokenizer_dirs(), _AUTO),
					default=_AUTO,
					tooltip="Override the tokenizer folder. 'auto' uses the "
					        "model folder's own config.json + tokenizer.json "
					        "(PaliGemma2: the gemma2 SentencePiece pair).",
					optional=True, advanced=True),
				io.String.Input("stop_token_ids", default="1",
					tooltip="Comma-separated ids that end generation. "
					        "PaliGemma/Gemma: 1 (<eos>).",
					optional=True, advanced=True),
				io.Int.Input("image_size", default=224, min=32, max=2048,
					tooltip="Square size the image is resized to for the "
					        "vision encoder.",
					optional=True, advanced=True),
				io.String.Input("pixel_mean", default="0.5",
					tooltip="Pixel normalization mean: one value or 'r, g, b' "
					        "(SigLIP: 0.5).",
					optional=True, advanced=True),
				io.String.Input("pixel_std", default="0.5",
					tooltip="Pixel normalization std: one value or 'r, g, b' "
					        "(SigLIP: 0.5).",
					optional=True, advanced=True),
				io.Combo.Input("engine", options=list(_dnn_engines()),
					default=_engine_graph_default(),
					tooltip="DNN engine for cv2.dnn.readNetFromONNX. These "
					        "graphs need the OpenCV 5 GRAPH engine. OpenCV 5.1 "
					        "merged its two engines into one, so 'auto' is the "
					        "graph engine there and this default follows the "
					        "build.",
					optional=True, advanced=True),
			],
			outputs=[
				io.String.Output(display_name="response",
					tooltip="The generated answer (stop tokens removed)."),
				io.Int.Output(display_name="new_tokens",
					tooltip="Number of tokens generated."),
			],
		)

	@classmethod
	def execute(cls, image, model, prompt, max_new_tokens, vision_file=_AUTO,
	            embed_file=_AUTO, lm_file=_AUTO, tokenizer=_AUTO,
	            stop_token_ids="1", image_size=224, pixel_mean="0.5",
	            pixel_std="0.5", engine="new graph") -> io.NodeOutput:
		stops = _parse_id_list(stop_token_ids, (1,))
		bundle = _bundle_name(model)
		files = _llm_models()

		frames = (imageutils.image_batch_to_bgr(image)
		          if isinstance(image, torch.Tensor) else [_to_bgr_u8(image)])
		if not frames or frames[0] is None:
			raise ValueError("VLM Generate needs a non-empty image.")

		response, new_tokens = _run_dnn_worker("VLM Generate", "vlm_generate", {
			"vision_path": _part_path(bundle, "vision", vision_file, files),
			"embed_path": _part_path(bundle, "embed", embed_file, files),
			"lm_path": _part_path(bundle, "lm", lm_file, files),
			"engine": _engine_val(_pick(engine)),
			"tokenizer_dir": _tokenizer_dir_for(bundle, tokenizer),
			"frame": np.ascontiguousarray(frames[0]),
			"prompt": str(prompt),
			"max_new_tokens": int(max_new_tokens),
			"stops": stops,
			"image_size": int(image_size),
			"pixel_mean": _parse_float_list(pixel_mean),
			"pixel_std": _parse_float_list(pixel_std)})
		return io.NodeOutput(response, int(new_tokens))


class Seq2SeqGenerate(io.ComfyNode):
	"""Encoder-decoder generation (ViT-GPT2 captioning / Florence-2 shapes)."""

	@classmethod
	def define_schema(cls):
		models = _llm_models()
		opt_models = _combo(models, _NONE, _AUTO)
		return io.Schema(
			node_id="CV_Seq2SeqGenerate",
			display_name="CV Seq2Seq Generate",
			category=CATEGORY,
			description="Generates text from an encoder-decoder model folder in "
			            "models/llm. Two shapes: (a) image-captioning style - "
			            "the encoder takes the image directly (input "
			            "'pixel_values' -> hidden states) and the decoder "
			            "generates token ids against them ('input_ids' + "
			            "'encoder_hidden_states'); (b) Florence-2 style - a "
			            "separate vision encoder produces image features that "
			            "are concatenated with the embedded text prompt as the "
			            "text encoder's 'inputs_embeds', and the decoder takes "
			            "embedded start tokens ('inputs_embeds'). Pick a merged "
			            "with-past decoder export for KV-cache. Greedy "
			            "decoding; runs in the interruptible DNN worker.",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="The image to describe (first frame of a batch). "
					        "Resized to image_size x image_size."),
				io.Combo.Input("model", options=_combo(_llm_bundles()),
					tooltip="Model FOLDER under models/llm (e.g. opencv/vit-"
					        "gpt2-image-captioning). Its encoder and decoder "
					        ".onnx are found inside it - override them below if "
					        "the layout is unusual. Note ViT-GPT2 ships no "
					        "tokenizer of its own: set 'tokenizer' to the gpt2 "
					        "folder."),
				io.String.Input("prompt", default="", multiline=True,
					tooltip="Text fed to the ENCODER (task instruction, e.g. "
					        "Florence-2's '<CAPTION>' - BPE-tokenized as plain "
					        "text exactly like HF does, add suffix '2' for "
					        "the BART </s>). Ignored in shape (a)."),
				io.String.Input("start_token_ids", default="50256",
					tooltip="Comma-separated DECODER start ids (ViT-GPT2: "
					        "50256 = <|endoftext|> BOS; Florence-2: 2)."),
				io.Int.Input("max_new_tokens", default=32, min=1, max=1024,
					tooltip="Maximum tokens to generate."),
				io.Combo.Input("encoder_file", options=_combo(models, _AUTO),
					default=_AUTO,
					tooltip="Override the text/main encoder .onnx. Shape (a): "
					        "input 'pixel_values' (ViT-GPT2's "
					        "encoder_model.onnx). Shape (b): inputs "
					        "'inputs_embeds' + 'attention_mask' (Florence-2's "
					        "encoder_model.onnx).",
					optional=True, advanced=True),
				io.Combo.Input("decoder_file", options=_combo(models, _AUTO),
					default=_AUTO,
					tooltip="Override the decoder .onnx (-> logits; inputs "
					        "'input_ids' or 'inputs_embeds' + "
					        "'encoder_hidden_states'). 'auto' prefers the "
					        "merged with-past export, which is what KV-cache "
					        "needs.",
					optional=True, advanced=True),
				io.Combo.Input("vision_file", options=opt_models, default=_NONE,
					tooltip="[none] = shape (a): the image goes straight into "
					        "the encoder. Otherwise a separate vision encoder "
					        ".onnx ('pixel_values' -> image features, e.g. "
					        "Florence-2's vision_encoder.onnx) whose output is "
					        "prefixed to the embedded prompt; 'auto' finds it "
					        "in the model folder.",
					optional=True, advanced=True),
				io.Combo.Input("embed_file", options=opt_models, default=_NONE,
					tooltip="[none] = the decoder takes raw 'input_ids'. "
					        "Otherwise a token-embedding .onnx ('input_ids' "
					        "-> embedding rows, e.g. Florence-2's "
					        "embed_tokens.onnx) used for the encoder prompt "
					        "and the decoder steps.",
					optional=True, advanced=True),
				io.Combo.Input("tokenizer", options=_combo(_tokenizer_dirs(), _AUTO),
					default=_AUTO,
					tooltip="Override the tokenizer folder. 'auto' uses the "
					        "model folder's own config.json + tokenizer.json - "
					        "ViT-GPT2 ships none, so point this at the gpt2 "
					        "folder.",
					optional=True, advanced=True),
				io.String.Input("prompt_token_ids", default="",
					tooltip="Comma-separated ids fed to the encoder INSTEAD "
					        "of encoding 'prompt' - for special/task tokens "
					        "the OpenCV tokenizer cannot produce. Blank = "
					        "encode 'prompt' as plain text.",
					optional=True, advanced=True),
				io.String.Input("prompt_suffix_ids", default="",
					tooltip="Comma-separated ids APPENDED to the encoded "
					        "prompt (BART-style tokenizers end the encoder "
					        "input with </s>: use '2' for Florence-2).",
					optional=True, advanced=True),
				io.String.Input("stop_token_ids", default="50256",
					tooltip="Comma-separated ids that end generation "
					        "(ViT-GPT2: 50256; Florence-2/BART-style: 2).",
					optional=True, advanced=True),
				io.Int.Input("image_size", default=224, min=32, max=2048,
					tooltip="Square size the image is resized to (ViT-GPT2: "
					        "224; Florence-2: 768).",
					optional=True, advanced=True),
				io.String.Input("pixel_mean", default="0.5",
					tooltip="Pixel normalization mean: one value or 'r, g, "
					        "b' (ViT-GPT2: 0.5; Florence-2: 0.485, 0.456, "
					        "0.406).",
					optional=True, advanced=True),
				io.String.Input("pixel_std", default="0.5",
					tooltip="Pixel normalization std: one value or 'r, g, b' "
					        "(ViT-GPT2: 0.5; Florence-2: 0.229, 0.224, "
					        "0.225).",
					optional=True, advanced=True),
				io.Combo.Input("engine", options=list(_dnn_engines()),
					default=_engine_graph_default(),
					tooltip="DNN engine for cv2.dnn.readNetFromONNX. These "
					        "graphs need the OpenCV 5 GRAPH engine. OpenCV 5.1 "
					        "merged its two engines into one, so 'auto' is the "
					        "graph engine there and this default follows the "
					        "build.",
					optional=True, advanced=True),
			],
			outputs=[
				io.String.Output(display_name="response",
					tooltip="The generated text (stop tokens removed)."),
				io.Int.Output(display_name="new_tokens",
					tooltip="Number of tokens generated."),
			],
		)

	@classmethod
	def execute(cls, image, model, prompt, start_token_ids, max_new_tokens,
	            encoder_file=_AUTO, decoder_file=_AUTO, vision_file=_NONE,
	            embed_file=_NONE, tokenizer=_AUTO, prompt_token_ids="",
	            prompt_suffix_ids="", stop_token_ids="50256", image_size=224,
	            pixel_mean="0.5", pixel_std="0.5",
	            engine="new graph") -> io.NodeOutput:
		stops = _parse_id_list(stop_token_ids, (50256,))
		start_ids = _parse_id_list(start_token_ids, (50256,))
		bundle = _bundle_name(model)
		files = _llm_models()

		frames = (imageutils.image_batch_to_bgr(image)
		          if isinstance(image, torch.Tensor) else [_to_bgr_u8(image)])
		if not frames or frames[0] is None:
			raise ValueError("Seq2Seq Generate needs a non-empty image.")

		response, new_tokens = _run_dnn_worker("Seq2Seq Generate",
		                                       "seq2seq_generate", {
			"encoder_path": _part_path(bundle, "encoder", encoder_file, files),
			"decoder_path": _part_path(bundle, "decoder", decoder_file, files),
			"vision_path": _part_path(bundle, "vision", vision_file, files),
			"embed_path": _part_path(bundle, "embed", embed_file, files),
			"engine": _engine_val(_pick(engine)),
			"tokenizer_dir": _tokenizer_dir_for(bundle, tokenizer),
			"frame": np.ascontiguousarray(frames[0]),
			"prompt": str(prompt),
			"start_ids": start_ids,
			"max_new_tokens": int(max_new_tokens),
			"stops": stops,
			"image_size": int(image_size),
			"pixel_mean": _parse_float_list(pixel_mean),
			"pixel_std": _parse_float_list(pixel_std),
			"prompt_token_ids": str(prompt_token_ids),
			"prompt_suffix_ids": str(prompt_suffix_ids)})
		return io.NodeOutput(response, int(new_tokens))


class SFaceEmbeddings(io.ComfyNode):
	"""cv2.FaceRecognizerSF: 128-D identity embedding per aligned face."""

	@classmethod
	def define_schema(cls):
		models = _onnx_models()
		return io.Schema(
			node_id="CV_SFaceEmbeddings",
			display_name="CV SFace Embeddings",
			category=CATEGORY,
			description="Turns each detected face into a 128-D identity vector "
			            "(cv2.FaceRecognizerSF, the SFace model). This is the "
			            "recognition half that 'CV YuNet Face Detect' does not "
			            "do: YuNet answers WHERE a face is, SFace answers WHOSE it "
			            "is. FaceRecognizerSF is a cv2 class the auto-generated "
			            "wrappers cannot expose. Place "
			            "face_recognition_sface_2021dec.onnx in ComfyUI/models/onnx. "
			            "Wire YuNet's 'landmarks' output straight into 'landmarks' "
			            "here - cv2 aligns each face onto a canonical 112x112 crop "
			            "using ONLY those five points (eyes, nose, mouth corners), "
			            "which is what makes the embedding pose-invariant. Compare "
			            "the vectors with 'CV Embedding Match'. Data only: the "
			            "aligned crops come out as an IMAGE batch so you can see "
			            "exactly what the model read. Zero faces is a valid result "
			            "(empty embeddings, count = 0).",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Image the faces were detected in. An IMAGE batch uses "
					        "its first frame."),
				io.Combo.Input("model", options=models,
					tooltip="SFace .onnx model from ComfyUI/models/onnx "
					        "(face_recognition_sface_2021dec.onnx)."),
				NPArray.Input("landmarks",
					tooltip="(N*5, 2) float32 landmarks, five per face in YuNet's "
					        "order (right eye, left eye, nose, right/left mouth "
					        "corner) - the 'landmarks' output of 'CV YuNet Face "
					        "Detect'. The row count must be a multiple of 5."),
			],
			outputs=[
				NPArray.Output(display_name="embeddings",
					tooltip="(N, 128) float32, one identity vector per face, in the "
					        "same order as the landmarks. Feed 'CV Embedding "
					        "Match'. Empty (0, 128) when there are no faces."),
				io.Image.Output(display_name="aligned",
					tooltip="IMAGE batch of the aligned 112x112 crops the model "
					        "actually read - preview it when a match looks wrong; a "
					        "bad alignment is the usual cause."),
				io.Int.Output(display_name="count",
					tooltip="How many faces were embedded."),
			],
		)

	@classmethod
	def execute(cls, image, model, landmarks) -> io.NodeOutput:
		frame = imageutils.first_bgr_frame(image)
		pts = (np.zeros((0, 2), np.float32) if landmarks is None
		       else np.asarray(landmarks, np.float32).reshape(-1, 2))
		if frame is None or not len(pts):
			return io.NodeOutput(np.zeros((0, 128), np.float32),
			                     imageutils.bgr_to_image_batch([]), 0)
		if len(pts) % 5:
			raise ValueError(
				f"landmarks has {len(pts)} rows, which is not a multiple of 5 - "
				"SFace aligns each face from exactly five points. Connect the "
				"'landmarks' output of 'CV YuNet Face Detect'.")
		emb, crops = _run_dnn_worker(
			"SFace Embeddings", "sface_embed", {
				"model_path": _model_path(model, "SFace"),
				"frame": np.ascontiguousarray(frame),
				"landmarks": np.ascontiguousarray(pts)})
		return io.NodeOutput(np.asarray(emb, np.float32),
		                     imageutils.bgr_to_image_batch(list(crops)),
		                     int(len(emb)))


class EmbeddingMatch(io.ComfyNode):
	"""Compare two sets of embedding vectors (cosine / normalized L2)."""

	_COSINE = "cosine similarity (higher = same)"
	_L2 = "normalized L2 distance (lower = same)"

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_EmbeddingMatch",
			display_name="CV Embedding Match",
			category=CATEGORY,
			description="Matches every vector in one embedding set against every "
			            "vector in another, with the two metrics "
			            "cv2.FaceRecognizerSF.match implements: cosine similarity "
			            "and the L2 distance between L2-normalized vectors. Written "
			            "as plain array maths on purpose - it needs no model file and "
			            "therefore works on ANY embeddings, not just SFace: the "
			            "'CV Deep Features' backbone vectors, DNN class scores, "
			            "your own descriptors. Typical use: 'query' = the faces in "
			            "the scene, 'gallery' = the known people; then 'best_index' "
			            "says who each query face is and 'is_match' says whether to "
			            "believe it. Empty inputs give empty outputs, never an error.",
			inputs=[
				NPArray.Input("query",
					tooltip="(A, D) embeddings to identify - e.g. the faces found in "
					        "the scene. A single (D,) vector is accepted as one row."),
				NPArray.Input("gallery",
					tooltip="(B, D) known reference embeddings, same dimension D. "
					        "Each query row is compared against all of them."),
				io.Combo.Input("metric", options=[cls._COSINE, cls._L2],
					default=cls._COSINE,
					tooltip="The two metrics cv2.FaceRecognizerSF.match implements. "
					        "Cosine is a SIMILARITY (bigger is more alike, best match = "
					        "maximum); normalized L2 is a DISTANCE (smaller is more "
					        "alike, best match = minimum). This only chooses HOW scores "
					        "are computed and compared - where the accept/reject line "
					        "sits is entirely the 'threshold' input."),
				io.Float.Input("threshold", default=0.363, min=-1.0, max=100.0, step=0.001,
					tooltip="Decision threshold for 'is_match', always used: with cosine "
					        "a match needs score >= threshold, with normalized L2 it "
					        "needs score <= threshold. The right value depends on YOUR "
					        "embeddings; as a reference, SFace's official face-identity "
					        "operating points are 0.363 (cosine) and 1.128 (normalized "
					        "L2), which is where the default comes from. Remember to "
					        "change it when you switch metric."),
			],
			outputs=[
				NPArray.Output(display_name="scores",
					tooltip="(A, B) float32 score matrix, query rows x gallery "
					        "columns - preview it with 'Preview CV Array' (heatmap) "
					        "to see the whole comparison at once."),
				NPArray.Output(display_name="best_index",
					tooltip="(A,) int32 index of the best gallery entry for each "
					        "query row (-1 when the gallery is empty)."),
				NPArray.Output(display_name="best_score",
					tooltip="(A,) float32 score of that best entry."),
				NPArray.Output(display_name="is_match",
					tooltip="(A,) uint8 0/255 - whether the best score passes the "
					        "threshold. Use it as labels for 'CV Draw Points' or "
					        "to filter the query boxes."),
				io.Boolean.Output(display_name="any_match",
					tooltip="True when at least one query row matched - branch on it "
					        "with 'Basic data handling: IfElse'."),
			],
		)

	@classmethod
	def execute(cls, query, gallery, metric, threshold) -> io.NodeOutput:
		def rows(a, name):
			if a is None:
				return np.zeros((0, 0), np.float32)
			arr = np.asarray(a, np.float32)
			if arr.ndim == 1:
				arr = arr.reshape(1, -1)
			if arr.ndim != 2:
				raise ValueError(
					f"{name} must be a (N, D) embedding matrix, got shape "
					f"{tuple(arr.shape)}.")
			return arr

		q, g = rows(query, "query"), rows(gallery, "gallery")
		if not q.size or not g.size:
			return io.NodeOutput(
				np.zeros((len(q), len(g)), np.float32),
				np.full((len(q),), -1, np.int32),
				np.zeros((len(q),), np.float32),
				np.zeros((len(q),), np.uint8), False)
		if q.shape[1] != g.shape[1]:
			raise ValueError(
				f"query vectors are {q.shape[1]}-D but gallery vectors are "
				f"{g.shape[1]}-D - they must come from the same model.")

		# L2-normalize both sides: cv2's FR_COSINE is the dot product of the
		# normalized vectors and FR_NORM_L2 is the distance between them
		qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
		gn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
		# startswith, not ==: the labels used to quote the SFace thresholds, so
		# workflows saved before that was dropped carry the longer string
		if str(metric).startswith("normalized L2"):
			scores = np.linalg.norm(qn[:, None, :] - gn[None, :, :], axis=2)
			best = np.argmin(scores, axis=1)
			hit = scores[np.arange(len(q)), best] <= float(threshold)
		else:
			scores = qn @ gn.T
			best = np.argmax(scores, axis=1)
			hit = scores[np.arange(len(q)), best] >= float(threshold)
		best_score = scores[np.arange(len(q)), best].astype(np.float32)
		return io.NodeOutput(scores.astype(np.float32), best.astype(np.int32),
		                     best_score, (hit.astype(np.uint8) * 255), bool(hit.any()))


NODES = [YuNetFaceDetect, SFaceEmbeddings, EmbeddingMatch,
         MPPalmDetect, MPHandPose, WeChatQRDetect,
         ColorizeModel, EastTextDetect, DnnBlobFromImage, DnnForward,
         DnnImagesFromBlob, DnnMaskOutput, DnnLetterbox, DnnForwardAll,
         DnnPickOutput, LoadLabels, YoloDetectDecode, YoloSegMasks,
         ClassScoresDecode,
         DnnTokenizer, LLMGenerate, VLMGenerate, Seq2SeqGenerate,
         FeatureExtractModel, MatchFeaturesModel]
