# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""OpenCV integration for ComfyUI (V3 API).

Layers:
* highlevel - curated, IMAGE/MASK-native nodes with dropdowns and batch support
* features  - feature detection/matching building blocks (NPARRAY-native)
* contours  - contour extraction/quadrilateral building blocks (NPARRAY-native)
* points    - point-set utilities: filter/polar/k-means/reduce/pick/draw
* segmentation - GrabCut and threshold-sweep exploration
* remap     - composable cv2.remap map generators (identity/lens/cylinder/...)
* hdr       - HDR / exposure-bracketing (align/fuse/merge-radiance/tonemap)
* convert   - IMAGE/MASK <-> raw ndarray (NPARRAY) bridges and a debug inspector
* tuples    - the CV_TUPLE composite-value socket type (one cv2 Size/Point/...
              as ONE wireable input) plus its builder / splitter nodes
* moments   - cv2.moments over the image INTENSITY (Hu's density function), the
              raster counterpart of contours' Shape Moments / Filter By Shape
* plots     - data visualization: cv2.plot rasters + interactive ECharts charts
* contrib   - opencv-contrib class APIs the generator cannot reach
              (white balance, superpixels, saliency, quality, bgsegm)
* quality   - cv2.quality class API: batch full-reference metrics + BRISQUE
* dnn       - DNN-model-based detectors (YuNet face detection)
* ml        - trainable cv2.ml classifiers + HOG / frozen-backbone features
* lowlevel  - auto-generated wrappers of the raw cv2.* functions
"""

from comfy_api.latest import ComfyExtension, io, ui as _ui

from . import codes, colorize, contours, contrib, convert, ccm, dnn, features, hdr, highlevel, lowlevel, ml, moments, plots, points, quality, remap, segmentation, tuples


# --- Preview3D camera-matrices shim ------------------------------------------
# The frontend's Comfy.Preview3D extension already reads result[3]/result[4] as
# (4x4 [R|t] extrinsics, 3x3 K intrinsics) and applies them via
# setCameraFromMatrices - exact pose plus the intrinsics' vertical FOV
# (2*atan(cy/fy)), which the plain camera_info path cannot set. Core
# UI.PreviewUI3D never fills those slots, so 'CV Camera Pose To 3D View' smuggles
# the matrices inside the camera_info dict (_extrinsics/_intrinsics, ignored by
# every core widget) and this wrapper surfaces them when that dict flows through
# the core Preview3D node. No-op for every other camera_info (extra slots stay
# absent), so stock behavior is unchanged.
if not getattr(_ui.PreviewUI3D, "_opencv_matrices_shim", False):
	_orig_preview3d_as_dict = _ui.PreviewUI3D.as_dict

	def _preview3d_as_dict_with_matrices(self):
		out = _orig_preview3d_as_dict(self)
		info = getattr(self, "camera_info", None)
		result = out.get("result")
		if (isinstance(info, dict) and isinstance(result, list) and len(result) == 3
				and "_extrinsics" in info and "_intrinsics" in info):
			out["result"] = result + [info["_extrinsics"], info["_intrinsics"]]
		return out

	_ui.PreviewUI3D.as_dict = _preview3d_as_dict_with_matrices
	_ui.PreviewUI3D._opencv_matrices_shim = True


class OpenCVExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [*highlevel.NODES, *features.NODES, *contours.NODES,
                *points.NODES, *segmentation.NODES, *remap.NODES,
                *hdr.NODES, *convert.NODES, *dnn.NODES, *ml.NODES,
                *ccm.NODES, *colorize.NODES, *codes.NODES, *contrib.NODES,
                *moments.NODES, *plots.NODES, *quality.NODES, *tuples.NODES,
                *lowlevel.NODES]


async def comfy_entrypoint() -> OpenCVExtension:
	return OpenCVExtension()
