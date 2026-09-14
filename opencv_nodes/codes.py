# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""QR code ENCODING and DECODING with OpenCV's own objdetect implementation.

The generator in `lowlevel.py` only reaches cv2 FUNCTIONS, and OpenCV's QR
support is exposed as CLASSES (`cv2.QRCodeEncoder`, `cv2.QRCodeDetector`,
`cv2.QRCodeDetectorAruco`), so none of it was reachable before this module -
the pack could only READ codes through the WeChat model in `dnn.py`
(`CV WeChat QR Detect`), which needs two .onnx files and cannot write a
code at all.

Same rule as `contrib.py`: a stateful cv2 object must never become graph state
(ComfyUI caches node outputs, so a cached handle would be mutated by the next
run), therefore each node creates, configures, uses and drops its detector /
encoder INSIDE one `execute()`.

Three generic nodes, no task semantics baked in:

* 'CV QR Encode'    - text -> QR bitmap (IMAGE). Version, error-correction
                          level, encoding mode and Structured-Append splitting
                          are named dropdowns/widgets; the payload of a
                          Structured Append set comes out as an IMAGE BATCH.
* 'CV QR Detect'    - detect AND decode every code in an image, with the
                          classic scanline detector or the ArUco-based one.
* 'CV QR Decode At Points' - decode a quad someone ELSE found (contour
                          quad, homography, annotation, another detector), so
                          detection and decoding stay composable.

Data only: corner points, quads, bboxes, decoded strings and the
rectified/binarized "straight" codes come out as plain data for 'Draw BBoxes',
'CV Draw Points/Connections/Labels', 'Preview Image' and 'Preview as Text'.
Zero codes is a valid result: the nodes emit empty arrays plus
`found=false` instead of raising.
"""

import cv2
import numpy as np

from comfy_api.latest import io

from . import imageutils, polytype
from .convert import NPArray

CATEGORY = "image/CV/codes"


# --- shared helpers ---------------------------------------------------------

# error-correction level: how much of the code may be damaged and still decode
ECC_LEVELS = {
	"L (~7% recovery)": cv2.QRCODE_ENCODER_CORRECT_LEVEL_L,
	"M (~15% recovery)": cv2.QRCODE_ENCODER_CORRECT_LEVEL_M,
	"Q (~25% recovery)": cv2.QRCODE_ENCODER_CORRECT_LEVEL_Q,
	"H (~30% recovery)": cv2.QRCODE_ENCODER_CORRECT_LEVEL_H,
}

# payload encoding mode - 'auto' lets OpenCV pick the densest one that fits
ENCODE_MODES = {
	"auto (pick the densest that fits)": cv2.QRCODE_ENCODER_MODE_AUTO,
	"numeric (0-9)": cv2.QRCODE_ENCODER_MODE_NUMERIC,
	"alphanumeric (0-9 A-Z $%*+-./: space)": cv2.QRCODE_ENCODER_MODE_ALPHANUMERIC,
	"byte (Latin-1 / raw bytes)": cv2.QRCODE_ENCODER_MODE_BYTE,
	"ECI (UTF-8)": cv2.QRCODE_ENCODER_MODE_ECI,
	"kanji (Shift-JIS)": cv2.QRCODE_ENCODER_MODE_KANJI,
}

DETECTORS = [
	"classic (QRCodeDetector)",
	"ArUco-based (QRCodeDetectorAruco)",
]

EMPTY_STR = np.array([], dtype="<U1")


def _version_of(side: int) -> int:
	"""QR version from a rendered side length in modules.

	cv2's encoder frames the symbol with a TWO-module quiet zone per side (the
	spec asks for four), so a version-v code (21 + 4*(v-1) modules) comes out
	as side = 25 + 4*(v-1) modules."""
	v = (side - 21) // 4
	return int(v) if 1 <= v <= 40 else 0


def _pad_to(img: np.ndarray, h: int, w: int, value: int = 255) -> np.ndarray:
	"""Pad a gray image out to h x w with `value` (white = extra quiet zone)."""
	if img.shape[0] == h and img.shape[1] == w:
		return img
	return cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, w - img.shape[1],
	                          cv2.BORDER_CONSTANT, value=value)


def _stack_gray(frames: list):
	"""Gray frames of differing sizes -> one IMAGE batch (white-padded).

	Straight/rectified codes are sized by their own version, so a multi-code
	image yields differently-sized crops that torch.stack would reject."""
	frames = [np.asarray(f) for f in frames if f is not None and np.size(f)]
	if not frames:
		return imageutils.bgr_to_image_batch([])
	h = max(f.shape[0] for f in frames)
	w = max(f.shape[1] for f in frames)
	return imageutils.bgr_to_image_batch([_pad_to(imageutils.ensure_uint8(f), h, w)
	                                      for f in frames])


def _quads_to_outputs(quads, texts):
	"""(bboxes, corners, connections, labels) for a list of 4x2 quads."""
	corner_arrays, boxes, labels, conns = [], [], [], []
	offset = 0
	for quad, text in zip(quads, texts):
		quad = np.asarray(quad, np.float32).reshape(-1, 2)
		n = len(quad)
		if n == 0:
			continue
		corner_arrays.append(quad)
		x1, y1 = quad.min(axis=0)
		x2, y2 = quad.max(axis=0)
		boxes.append({"x": int(round(float(x1))), "y": int(round(float(y1))),
		              "width": int(round(float(x2 - x1))),
		              "height": int(round(float(y2 - y1))),
		              "score": 1.0, "label": text or "qr"})
		# closed-quad edges, indexed into the concatenated corner rows
		conns.append(np.array([(k, (k + 1) % n) for k in range(n)], np.int32) + offset)
		# decoded text on the first corner only, blanks on the rest
		labels.extend([text] + [""] * (n - 1))
		offset += n
	if not corner_arrays:
		return [], np.zeros((0, 2), np.float32), np.zeros((0, 2), np.int32), EMPTY_STR
	return (boxes,
	        np.concatenate(corner_arrays, axis=0).astype(np.float32),
	        np.concatenate(conns, axis=0).astype(np.int32),
	        np.array(labels))


def _points_to_quads(points) -> np.ndarray:
	"""Any accepted point layout -> (N, 4, 2) float32 quads."""
	arr = np.asarray(polytype.resolve(points)[0], dtype=np.float32)
	arr = arr.reshape(-1, 2)
	if len(arr) < 4:
		return np.zeros((0, 4, 2), np.float32)
	return np.ascontiguousarray(arr[:len(arr) // 4 * 4].reshape(-1, 4, 2))


def _make_detector(kind: str, eps_x=None, eps_y=None, use_alignment_markers=None):
	"""A freshly configured QR detector (never cached - see module docstring)."""
	if kind.startswith("ArUco"):
		# QRCodeDetectorAruco tunes itself through QRCodeDetectorAruco_Params
		# and has none of the classic scanline knobs below.
		return cv2.QRCodeDetectorAruco()
	det = cv2.QRCodeDetector()
	if eps_x is not None:
		det.setEpsX(float(eps_x))
	if eps_y is not None:
		det.setEpsY(float(eps_y))
	if use_alignment_markers is not None:
		det.setUseAlignmentMarkers(bool(use_alignment_markers))
	return det


# --- nodes ------------------------------------------------------------------

class QREncode(io.ComfyNode):
	"""cv2.QRCodeEncoder: text -> QR code bitmap."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_QREncode",
			display_name="CV QR Encode",
			category=CATEGORY,
			description="Generates a QR code from text with cv2.QRCodeEncoder "
			            "(OpenCV's own encoder - no model files, no extra "
			            "dependency). The output is a black-and-white IMAGE "
			            "whose modules are 'module_size' pixels wide, upscaled "
			            "with nearest-neighbour so the code stays crisp and "
			            "scannable. OpenCV frames the symbol with only TWO "
			            "quiet-zone modules where the spec asks for four, so "
			            "'quiet_zone' adds the missing white margin (default 2 "
			            "= spec-compliant). 'version' fixes the symbol size (1 = 21x21 "
			            "modules ... 40 = 177x177) - leave it at 0 to let "
			            "OpenCV pick the smallest version that fits the payload "
			            "at the chosen error-correction level. Set "
			            "'structure_number' above 1 to split a long message "
			            "over that many codes in Structured Append mode: the "
			            "output is then an IMAGE BATCH of the parts (white-"
			            "padded to a common size) which 'CV QR Detect' "
			            "reassembles into the full message when they are all "
			            "visible in one image.",
			inputs=[
				io.String.Input("text", default="https://opencv.org", multiline=True,
					tooltip="Payload to encode. Must be representable in the "
					        "chosen 'mode' (numeric = digits only, alphanumeric "
					        "= 0-9 A-Z and $%*+-./: and space) and must fit the "
					        "chosen version - otherwise the node reports the "
					        "OpenCV error instead of writing a corrupt code."),
				io.Combo.Input("error_correction", options=list(ECC_LEVELS),
					default="M (~15% recovery)",
					tooltip="Reed-Solomon error-correction level: how much of "
					        "the code may be damaged/occluded and still decode. "
					        "Higher levels leave less room for data (the symbol "
					        "grows) but survive print wear, glare and logos "
					        "pasted over the centre."),
				io.Combo.Input("mode", options=list(ENCODE_MODES),
					default="auto (pick the densest that fits)",
					tooltip="Payload encoding. 'auto' lets OpenCV choose the "
					        "densest mode the text fits. numeric/alphanumeric "
					        "pack more characters per module but restrict the "
					        "alphabet; byte carries arbitrary bytes; ECI (UTF-8) "
					        "tags the payload as UTF-8 for non-Latin text; kanji "
					        "packs Shift-JIS."),
				io.Int.Input("version", default=0, min=0, max=40,
					tooltip="Symbol version = size in modules (v = 21 + 4*(v-1) "
					        "modules square). 0 = auto: the smallest version "
					        "that fits. A fixed version keeps a set of codes the "
					        "same physical size; too small for the payload is an "
					        "error."),
				io.Int.Input("module_size", default=8, min=1, max=64,
					tooltip="Pixels per QR module in the output image "
					        "(nearest-neighbour upscale). 1 = the raw bitmap "
					        "(one pixel per module) - too small for most "
					        "scanners; 6-12 is a good print/screen size."),
				io.Int.Input("quiet_zone", default=2, min=0, max=16,
					tooltip="EXTRA white border in modules, added around the "
					        "TWO quiet-zone modules OpenCV emits. The QR spec "
					        "asks for four, so 2 (the default) is the "
					        "spec-compliant margin; raise it when the code is "
					        "composited onto a busy background."),
				io.Int.Input("structure_number", default=1, min=1, max=16,
					tooltip="Structured Append: number of codes the message is "
					        "split over (1 = a single code). The parts decode "
					        "into one message only when a decoder sees all of "
					        "them; 'CV QR Detect' writes the joined message "
					        "on the first part."),
			],
			outputs=[
				io.Image.Output(display_name="qr_image",
					tooltip="The QR code as an IMAGE (white background, black "
					        "modules, 3 identical channels). A batch of "
					        "'structure_number' frames in Structured Append "
					        "mode."),
				io.Int.Output(display_name="modules",
					tooltip="Side length of the encoder's bitmap in modules, "
					        "its 2-module quiet zone included and the extra "
					        "'quiet_zone' excluded (version 1 = 25)."),
				io.Int.Output(display_name="version",
					tooltip="QR version actually used (1-40), i.e. what 'auto' "
					        "resolved to. 0 if the size is not a valid version."),
			],
		)

	@classmethod
	def execute(cls, text, error_correction, mode, version, module_size,
	            quiet_zone, structure_number) -> io.NodeOutput:
		params = cv2.QRCodeEncoder_Params()
		params.correction_level = ECC_LEVELS[error_correction]
		params.mode = ENCODE_MODES[mode]
		params.version = int(version)
		params.structure_number = max(1, int(structure_number))
		encoder = cv2.QRCodeEncoder_create(params)
		try:
			if params.structure_number > 1:
				codes = list(encoder.encodeStructuredAppend(text))
			else:
				codes = [encoder.encode(text)]
		except cv2.error as exc:
			# a payload that does not fit / does not match the mode is a real
			# input mistake: say which knob to change instead of a raw cv2 dump
			raise RuntimeError(
				f"cv2.QRCodeEncoder could not encode this payload "
				f"({len(text)} chars, mode='{mode}', version="
				f"{'auto' if not version else version}, "
				f"error_correction='{error_correction}'). Lower the "
				f"error-correction level, raise/auto the version, shorten the "
				f"text, or pick a mode its characters fit. OpenCV said: "
				f"{exc}") from exc

		codes = [np.asarray(c, np.uint8) for c in codes if c is not None and np.size(c)]
		if not codes:
			raise RuntimeError("cv2.QRCodeEncoder returned an empty code for "
			                   "this payload.")

		side = int(codes[0].shape[0])
		frames = []
		for code in codes:
			if quiet_zone:
				code = cv2.copyMakeBorder(code, quiet_zone, quiet_zone, quiet_zone,
				                          quiet_zone, cv2.BORDER_CONSTANT, value=255)
			if module_size > 1:
				code = cv2.resize(code, None, fx=module_size, fy=module_size,
				                  interpolation=cv2.INTER_NEAREST)
			frames.append(code)
		h = max(f.shape[0] for f in frames)
		w = max(f.shape[1] for f in frames)
		frames = [_pad_to(f, h, w) for f in frames]
		return io.NodeOutput(imageutils.bgr_to_image_batch(frames), side,
		                     _version_of(side))


class QRDetect(io.ComfyNode):
	"""cv2.QRCodeDetector / QRCodeDetectorAruco: find AND read every QR code."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_QRDetect",
			display_name="CV QR Detect",
			category=CATEGORY,
			description="Detects and decodes every QR code in an image with "
			            "OpenCV's built-in detector - no model files, unlike "
			            "'CV WeChat QR Detect' (cv2.dnn). Two detectors: "
			            "the classic scanline one (finder-pattern scan, tunable "
			            "eps_x/eps_y, alignment-marker refinement) and the "
			            "ArUco-based one, which is usually better on small, "
			            "rotated or perspective-distorted codes. 'curved' mode "
			            "runs detectAndDecodeCurved for a SINGLE code wrapped "
			            "around a cylinder/bottle (classic detector only). "
			            "Outputs are DATA only: feed 'bboxes' to the core 'Draw "
			            "BBoxes' node, 'corners' + 'connections' to 'OpenCV "
			            "Draw Connections' for the exact quad, 'corners' + "
			            "'labels' to 'CV Draw Labels' for the text, 'text' "
			            "to 'Preview as Text', and 'straight_codes' to 'Preview "
			            "Image' to see the rectified binarized symbols. Zero "
			            "codes is a valid result (empty outputs, found=false). "
			            "A code that is located but unreadable comes out with a "
			            "quad and an EMPTY string. Codes written in Structured "
			            "Append mode decode into one joined message on the "
			            "first part, the rest blank. Processes a single image "
			            "(the first frame of a batch).",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Image to scan. An IMAGE batch uses its first frame; "
					        "an NPARRAY (gray/BGR, any dtype) is treated as one "
					        "frame."),
				io.Combo.Input("detector", options=DETECTORS,
					tooltip="classic: cv2.QRCodeDetector, the scanline "
					        "finder-pattern detector (honours eps_x/eps_y and "
					        "use_alignment_markers). ArUco-based: "
					        "cv2.QRCodeDetectorAruco, which reuses the ArUco "
					        "marker machinery - more robust on small, rotated "
					        "or warped codes, and ignores the classic knobs."),
				io.Combo.Input("mode", options=["multi (all codes, flat)",
				                                "curved (single code)"],
					tooltip="multi: detectAndDecodeMulti - every code in the "
					        "image (the normal choice). curved: "
					        "detectAndDecodeCurved - ONE code on a curved "
					        "surface, rectified before decoding; classic "
					        "detector only, and it fails on some flat codes, so "
					        "only use it for genuinely curved ones."),
				io.Float.Input("eps_x", default=0.2, min=0.0, max=1.0, step=0.01,
					tooltip="Classic detector only: tolerance of the HORIZONTAL "
					        "scan for the 1:1:3:1:1 finder-pattern ratio. Raise "
					        "it for blurry/low-contrast codes, lower it to "
					        "reject false finders.",
					optional=True, advanced=True),
				io.Float.Input("eps_y", default=0.1, min=0.0, max=1.0, step=0.01,
					tooltip="Classic detector only: the same tolerance for the "
					        "VERTICAL scan of the finder pattern.",
					optional=True, advanced=True),
				io.Boolean.Input("use_alignment_markers", default=True,
					tooltip="Classic detector only: refine the corner positions "
					        "with the code's alignment markers (OpenCV's "
					        "default). Turning it off is faster and can help on "
					        "damaged codes whose markers are unreliable.",
					optional=True, advanced=True),
			],
			outputs=[
				io.Boolean.Output(display_name="found",
					tooltip="True when at least one code was located. Feed a "
					        "'Basic data handling: IfElse' to branch instead of "
					        "gating a node with a boolean widget."),
				io.Int.Output(display_name="qr_count",
					tooltip="Number of QR codes located."),
				io.String.Output(display_name="text",
					tooltip="Every decoded string, one per line - wire into the "
					        "core 'Preview as Text' (PreviewAny) node. Empty "
					        "when nothing decoded."),
				io.BoundingBox.Output(display_name="bboxes",
					tooltip="One {x, y, width, height, score, label} dict per "
					        "code (axis-aligned box around its 4 corners, label "
					        "= decoded text) - feed the core 'Draw BBoxes' node."),
				NPArray.Output(display_name="corners",
					tooltip="(N*4, 2) float32: the 4 corner points of every "
					        "code, in order. Feed 'CV Draw Points', or "
					        "'CV Draw Connections'/'Draw Labels' together "
					        "with the matching outputs. Empty (0, 2) when no "
					        "codes."),
				NPArray.Output(display_name="connections",
					tooltip="(N*4, 2) int32 edge index-pairs closing each code's "
					        "quad, indexed into 'corners' - feed 'CV Draw "
					        "Connections' to outline the codes exactly (unlike "
					        "the axis-aligned bboxes)."),
				NPArray.Output(display_name="texts",
					tooltip="(N,) string array of the decoded payloads, one per "
					        "code - inspect with 'Inspect CV Data' + 'Preview as "
					        "Text'. An empty entry means the code was located "
					        "but not decoded."),
				NPArray.Output(display_name="labels",
					tooltip="(N*4,) string array aligned row-for-row with "
					        "'corners': each code's text on its FIRST corner, "
					        "blank on the other three. Feed 'corners' + this to "
					        "'CV Draw Labels' to write the text once per "
					        "code."),
				io.Image.Output(display_name="straight_codes",
					tooltip="IMAGE batch of the rectified, binarized symbols "
					        "(one per decoded code, perspective removed) - the "
					        "picture the decoder actually read. Frames of "
					        "differing versions are white-padded to a common "
					        "size. Empty batch when nothing decoded."),
			],
		)

	@classmethod
	def execute(cls, image, detector, mode, eps_x=0.2, eps_y=0.1,
	            use_alignment_markers=True) -> io.NodeOutput:
		empty = (False, 0, "", [], np.zeros((0, 2), np.float32),
		         np.zeros((0, 2), np.int32), EMPTY_STR, EMPTY_STR,
		         imageutils.bgr_to_image_batch([]))
		frame = imageutils.first_bgr_frame(image)
		if frame is None:
			return io.NodeOutput(*empty)

		det = _make_detector(detector, eps_x, eps_y, use_alignment_markers)
		curved = mode.startswith("curved")
		if curved:
			if detector.startswith("ArUco"):
				# QRCodeDetectorAruco has no curved path in OpenCV 5
				det = _make_detector(DETECTORS[0], eps_x, eps_y,
				                     use_alignment_markers)
			text, points, straight = det.detectAndDecodeCurved(frame)
			texts = [text or ""]
			quads = [] if points is None else [np.asarray(points, np.float32)]
			straights = [] if straight is None or not np.size(straight) else [straight]
		else:
			ok, infos, points, straights = det.detectAndDecodeMulti(frame)
			if not ok or points is None or not len(points):
				return io.NodeOutput(*empty)
			quads = [np.asarray(p, np.float32) for p in points]
			texts = list(infos) + [""] * (len(quads) - len(infos))
			straights = list(straights) if straights is not None else []

		# drop degenerate quads WITH their string so the two stay aligned
		kept = [(q, t) for q, t in zip(quads, texts) if np.size(q)]
		if not kept:
			return io.NodeOutput(*empty)
		quads = [q for q, _ in kept]
		texts = [t for _, t in kept]

		boxes, corners, connections, labels = _quads_to_outputs(quads, texts)
		joined = "\n".join(t for t in texts if t)
		return io.NodeOutput(True, len(quads), joined, boxes, corners,
		                     connections, np.array(texts), labels,
		                     _stack_gray(straights))


class QRDecodeAtPoints(io.ComfyNode):
	"""Decode QR codes whose quads were found by something else."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_QRDecodeAtPoints",
			display_name="CV QR Decode At Points",
			category=CATEGORY,
			description="Decodes the QR code(s) inside quads that were located "
			            "by ANY other means - a contour quadrilateral, a "
			            "homography, hand-annotated points, a DNN detector - "
			            "using cv2's decodeMulti/decodeCurved. This keeps "
			            "detection and decoding composable: when OpenCV's own "
			            "detector misses a code but you can outline it (e.g. "
			            "'CV Find Quadrilateral' on a thresholded region, "
			            "or 'CV Annotate Points'), the payload is still "
			            "readable. Points are consumed 4 at a time in corner "
			            "order (any winding); trailing points that do not "
			            "complete a quad are ignored. Give it the corners of the "
			            "SYMBOL, not of the white quiet zone around it - cv2 "
			            "rectifies exactly the quad it is handed, so a quad that "
			            "includes the margin decodes to nothing. Undecodable "
			            "quads come out as empty strings, never an error.",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Image containing the code(s). An IMAGE batch uses "
					        "its first frame."),
				NPArray.Input("points",
					tooltip="Corner points, (N*4, 2) or (N, 4, 2), in per-code "
					        "corner order - e.g. the 'corners' output of 'OpenCV "
					        "QR Detect' / 'CV Find Quadrilateral', or "
					        "annotated points. cv2 expects the corners of the "
					        "SYMBOL ITSELF (the outer finder-pattern corners), "
					        "not of the white quiet zone around it - a quad that "
					        "includes the margin decodes to an empty string."),
				io.Combo.Input("detector", options=DETECTORS,
					tooltip="Which cv2 decoder implementation reads the "
					        "rectified patch. Both accept externally supplied "
					        "corners; the ArUco-based one can be more tolerant "
					        "of small/warped symbols."),
				io.Combo.Input("mode", options=["straight (flat surface)",
				                                "curved (cylindrical surface)"],
					tooltip="straight: decodeMulti - perspective-rectify the "
					        "quad and read it. curved: decodeCurved per quad, "
					        "for codes wrapped around a curved surface (classic "
					        "detector only)."),
			],
			outputs=[
				io.Boolean.Output(display_name="found",
					tooltip="True when at least one quad decoded to a non-empty "
					        "string."),
				io.Int.Output(display_name="decoded_count",
					tooltip="How many of the supplied quads decoded."),
				io.String.Output(display_name="text",
					tooltip="Decoded strings, one per line - wire into 'Preview "
					        "as Text'."),
				NPArray.Output(display_name="texts",
					tooltip="(N,) string array aligned with the supplied quads; "
					        "an empty entry marks a quad that did not decode."),
				io.Image.Output(display_name="straight_codes",
					tooltip="IMAGE batch of the rectified binarized symbols, one "
					        "per decoded quad (white-padded to a common size)."),
			],
		)

	@classmethod
	def execute(cls, image, points, detector, mode) -> io.NodeOutput:
		empty = (False, 0, "", EMPTY_STR, imageutils.bgr_to_image_batch([]))
		frame = imageutils.first_bgr_frame(image)
		quads = _points_to_quads(points)
		if frame is None or not len(quads):
			return io.NodeOutput(*empty)

		curved = mode.startswith("curved")
		det = _make_detector(DETECTORS[0] if curved else detector)
		texts, straights = [], []
		if curved:
			for quad in quads:
				try:
					text, straight = det.decodeCurved(frame, quad)
				except cv2.error:  # a degenerate quad must not halt the graph
					text, straight = "", None
				texts.append(text or "")
				if straight is not None and np.size(straight):
					straights.append(straight)
		else:
			try:
				ok, infos, straight_list = det.decodeMulti(frame, quads)
			except cv2.error:
				ok, infos, straight_list = False, (), ()
			if not ok:
				infos, straight_list = (), ()
			texts = list(infos) + [""] * (len(quads) - len(infos))
			straights = [s for s in (straight_list or []) if s is not None and np.size(s)]

		texts = texts[:len(quads)]
		decoded = [t for t in texts if t]
		return io.NodeOutput(bool(decoded), len(decoded), "\n".join(decoded),
		                     np.array(texts) if texts else EMPTY_STR,
		                     _stack_gray(straights))


class BarcodeDetect(io.ComfyNode):
	"""cv2.barcode.BarcodeDetector: find AND read 1-D product barcodes."""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_BarcodeDetect",
			display_name="CV Barcode Detect",
			category=CATEGORY,
			description="Detects and decodes 1-D product barcodes (EAN-8, EAN-13, "
			            "UPC-A, UPC-E) with cv2.barcode.BarcodeDetector - the "
			            "linear-barcode counterpart of 'CV QR Detect', and "
			            "another cv2 CLASS the auto-generated wrappers cannot reach. "
			            "It locates the bar region by its dense parallel gradients, "
			            "rectifies it and reads the bar widths, so it copes with "
			            "rotation, perspective and the curve of a bottle label. "
			            "Outputs mirror the QR node - DATA only: 'bboxes' to the core "
			            "'Draw BBoxes', 'corners' + 'connections' to 'CV Draw "
			            "Connections' for the exact quad, 'corners' + 'labels' to "
			            "'CV Draw Labels', 'text' to 'Preview as Text'. Zero "
			            "barcodes is a valid result (empty outputs, found=false), "
			            "never an error. Processes a single image (the first frame of "
			            "a batch). NOTE: unlike QR there is no encoder in OpenCV - "
			            "this node only reads.",
			inputs=[
				polytype.poly_input("image", polytype.IMAGE_ONLY,
					tooltip="Image to scan. An IMAGE batch uses its first frame; an "
					        "NPARRAY (gray/BGR, any dtype) is treated as one frame."),
				io.Float.Input("gradient_threshold", default=64.0, min=0.0, max=255.0, step=1.0,
					tooltip="Minimum gradient magnitude for a pixel to count towards "
					        "a bar region. LOWER it for faint, low-contrast or "
					        "out-of-focus barcodes; raise it to reject busy "
					        "background texture that looks like bars."),
				io.Float.Input("downsampling_threshold", default=512.0, min=64.0, max=8192.0, step=32.0,
					tooltip="Images larger than this (in pixels, on the shorter "
					        "side) are downsampled before the SEARCH - decoding still "
					        "happens at full resolution. Raise it when a small "
					        "barcode in a big photo is missed.",
					optional=True, advanced=True),
				io.String.Input("detector_scales", default="0.01, 0.03, 0.06, 0.08",
					tooltip="Comma-separated relative scales the detector searches "
					        "at, as a fraction of the image size. Add a larger value "
					        "(e.g. 0.2) when the barcode fills much of the frame, a "
					        "smaller one when it is tiny. Blank = OpenCV's defaults.",
					optional=True, advanced=True),
			],
			outputs=[
				io.Boolean.Output(display_name="found",
					tooltip="True when at least one barcode was located. Feed a "
					        "'Basic data handling: IfElse' to branch instead of "
					        "gating a node with a boolean widget."),
				io.Int.Output(display_name="barcode_count",
					tooltip="Number of barcodes located."),
				io.String.Output(display_name="text",
					tooltip="Every decoded payload, one per line - wire into the "
					        "core 'Preview as Text' (PreviewAny) node. Empty when "
					        "nothing decoded."),
				io.BoundingBox.Output(display_name="bboxes",
					tooltip="One {x, y, width, height, score, label} dict per barcode "
					        "(axis-aligned box around its 4 corners, label = decoded "
					        "digits) - feed the core 'Draw BBoxes' node."),
				NPArray.Output(display_name="corners",
					tooltip="(N*4, 2) float32: the 4 corner points of every barcode, "
					        "in order. Feed 'CV Draw Points', or 'CV Draw "
					        "Connections'/'Draw Labels' with the matching outputs. "
					        "Empty (0, 2) when nothing is found."),
				NPArray.Output(display_name="connections",
					tooltip="(N*4, 2) int32 edge index-pairs closing each barcode's "
					        "quad, indexed into 'corners' - feed 'CV Draw "
					        "Connections' to outline the rotated region exactly."),
				NPArray.Output(display_name="texts",
					tooltip="(N,) string array of the decoded payloads, one per "
					        "barcode. An empty entry means the barcode was located "
					        "but could not be read."),
				NPArray.Output(display_name="formats",
					tooltip="(N,) string array of the symbology per barcode "
					        "('EAN_13', 'EAN_8', 'UPC_A', 'UPC_E', or 'NONE' when "
					        "only located)."),
				NPArray.Output(display_name="labels",
					tooltip="(N*4,) string array aligned row-for-row with 'corners': "
					        "each barcode's text on its FIRST corner, blank on the "
					        "other three. Feed 'corners' + this to 'CV Draw "
					        "Labels' to write the digits once per barcode."),
			],
		)

	@classmethod
	def execute(cls, image, gradient_threshold, downsampling_threshold=512.0,
	            detector_scales="0.01, 0.03, 0.06, 0.08") -> io.NodeOutput:
		empty = (False, 0, "", [], np.zeros((0, 2), np.float32),
		         np.zeros((0, 2), np.int32), EMPTY_STR, EMPTY_STR, EMPTY_STR)
		frame = imageutils.first_bgr_frame(image)
		if frame is None:
			return io.NodeOutput(*empty)

		det = cv2.barcode.BarcodeDetector()
		det.setGradientThreshold(float(gradient_threshold))
		det.setDownsamplingThreshold(float(downsampling_threshold))
		scales = [float(s) for s in str(detector_scales).replace(",", " ").split() if s]
		if scales:
			det.setDetectorScales(np.asarray(scales, np.float32))

		ok, texts, types, points = det.detectAndDecodeWithType(frame)
		if not ok or points is None or not len(points):
			return io.NodeOutput(*empty)

		quads = [np.asarray(p, np.float32) for p in points]
		texts = list(texts) + [""] * (len(quads) - len(texts))
		types = list(types) + ["NONE"] * (len(quads) - len(types))
		# drop degenerate quads WITH their string/type so the three stay aligned
		kept = [(q, t, ty) for q, t, ty in zip(quads, texts, types) if np.size(q)]
		if not kept:
			return io.NodeOutput(*empty)
		quads = [q for q, _, _ in kept]
		texts = [t for _, t, _ in kept]
		types = [ty for _, _, ty in kept]

		boxes, corners, connections, labels = _quads_to_outputs(quads, texts)
		joined = "\n".join(t for t in texts if t)
		return io.NodeOutput(True, len(quads), joined, boxes, corners, connections,
		                     np.array(texts), np.array(types), labels)


NODES = [QREncode, QRDetect, QRDecodeAtPoints, BarcodeDetect]
