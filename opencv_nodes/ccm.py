# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Color Correction Module (CCM) nodes.

Wraps cv2.ccm for color calibration via ColorChecker charts.

Custom types:
* CCM_MODEL - fitted cv2.ccm.ColorCorrectionModel ready for inference
"""

import cv2
import numpy as np
import torch

from comfy_api.latest import io

from . import enums, imageutils, polytype
from .convert import NPArray

CATEGORY = "image/CV/ccm"

CCMModel = io.Custom("CCM_MODEL")


def _map_frames(image: torch.Tensor, fn) -> torch.Tensor:
    """Apply fn(bgr_uint8) -> bgr/gray uint8 to every image in the batch."""
    return imageutils.bgr_to_image_batch([fn(f) for f in imageutils.image_batch_to_bgr(image)])


# ---------------------------------------------------------------------------
# Phase 1: detect color checker patches
# ---------------------------------------------------------------------------

class DetectColorChecker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_DetectColorChecker",
            display_name="CV Detect Color Checker",
            category=CATEGORY,
            description="Detects a ColorChecker chart (Macbeth 24, Vinyl 18, "
                        "or DigitalSG 140) in the image and extracts the "
                        "measured patch colors. Outputs the color patches as "
                        "an NPARRAY for use with 'Create CCM Model'.",
            inputs=[
                io.Image.Input("image",
                    tooltip="Input image containing a ColorChecker chart."),
                io.Combo.Input("chart_type",
                    options=["MCC24 (Macbeth)", "VINYL18", "SG140"],
                    default="MCC24 (Macbeth)",
                    tooltip="Type of color checker chart to detect."),
            ],
            outputs=[
                NPArray.Output(display_name="color_patches",
                    tooltip="Detected patch colors as [N,1,3] float64 RGB in "
                            "[0,1]. Feed into 'Create CCM Model'."),
                io.Boolean.Output(display_name="found",
                    tooltip="True if a chart was detected."),
            ],
        )

    @classmethod
    def execute(cls, image, chart_type) -> io.NodeOutput:
        frames = imageutils.image_batch_to_bgr(image)
        frame = frames[0]  # use first frame

        chart_map = {
            "MCC24 (Macbeth)": cv2.mcc.MCC24,
            "VINYL18": cv2.mcc.VINYL18,
            "SG140": cv2.mcc.SG140,
        }
        chart = chart_map.get(chart_type, cv2.mcc.MCC24)

        detector = cv2.mcc.CCheckerDetector_create()
        detector.setColorChartType(chart)
        success = detector.process(frame, nc=1)

        if not success:
            empty = np.zeros((0, 1, 3), dtype=np.float64)
            return io.NodeOutput(empty, False)

        checkers = detector.getListColorChecker()
        checker = checkers[0]
        charts_rgb = checker.getChartsRGB()
        width = charts_rgb.shape[0]
        src = charts_rgb[:, 1].copy().reshape(int(width / 3), 1, 3) / 255.0
        src = src.astype(np.float64)

        return io.NodeOutput(src, True)


# ---------------------------------------------------------------------------
# Phase 2: create and fit a CCM model
# ---------------------------------------------------------------------------

class CreateCCMModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_CreateCCMModel",
            display_name="CV Create CCM Model",
            category=CATEGORY,
            description="Creates and fits a Color Correction Model from "
                        "detected color patches. The fitted model can be "
                        "applied to images with 'Apply Color Correction'.",
            inputs=[
                NPArray.Input("color_patches",
                    tooltip="Detected patch colors from 'Detect Color Checker' "
                            "[N,1,3] float64 in [0,1]."),
                io.Combo.Input("ccm_type",
                    options=enums.CCM_CCM_TYPE.options,
                    default=enums.CCM_CCM_TYPE.default,
                    tooltip="CCM_LINEAR: 3x3 matrix (no offset). "
                            "CCM_AFFINE: 4x3 matrix (with offset)."),
                io.Combo.Input("color_space",
                    options=enums.CCM_COLOR_SPACE.options,
                    default=enums.CCM_COLOR_SPACE.default,
                    tooltip="Working color space for the correction."),
                io.Combo.Input("distance",
                    options=enums.CCM_DISTANCE.options,
                    default=enums.CCM_DISTANCE.default,
                    tooltip="Color distance metric used in optimization. "
                            "CIE2000 is most perceptually accurate."),
                io.Combo.Input("linearization",
                    options=enums.CCM_LINEARIZATION.options,
                    default=enums.CCM_LINEARIZATION.default,
                    tooltip="Linearization method before applying CCM. "
                            "GAMMA is the standard choice."),
                io.Float.Input("linearization_gamma", default=2.2, min=0.1, max=5.0, step=0.1,
                    tooltip="Gamma value for LINEARIZATION_GAMMA."),
                io.Int.Input("linearization_degree", default=3, min=1, max=10,
                    tooltip="Polynomial degree for POLYFIT/LOGPOLYFIT methods."),
                io.Combo.Input("color_checker",
                    options=enums.CCM_COLOR_CHECKER.options,
                    default=enums.CCM_COLOR_CHECKER.default,
                    tooltip="Reference color checker type (must match chart)."),
                io.Float.Input("saturated_lower", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Lower saturation threshold for filtering patches."),
                io.Float.Input("saturated_upper", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Upper saturation threshold (0 = disabled)."),
            ],
            outputs=[
                CCMModel.Output(display_name="ccm_model",
                    tooltip="Fitted color correction model."),
                NPArray.Output(display_name="ccm_matrix",
                    tooltip="The computed 3x3 or 4x3 color correction matrix."),
                io.Float.Output(display_name="loss",
                    tooltip="Fit quality metric (lower is better)."),
            ],
        )

    @classmethod
    def execute(cls, color_patches, ccm_type, color_space, distance,
                linearization, linearization_gamma, linearization_degree,
                color_checker, saturated_lower, saturated_upper) -> io.NodeOutput:
        patches = np.asarray(color_patches, dtype=np.float64)
        if patches.ndim == 2:
            patches = patches.reshape(-1, 1, 3)

        cc_map = {
            "COLORCHECKER_MACBETH": cv2.ccm.COLORCHECKER_MACBETH,
            "COLORCHECKER_VINYL": cv2.ccm.COLORCHECKER_VINYL,
            "COLORCHECKER_DIGITAL_SG": cv2.ccm.COLORCHECKER_DIGITAL_SG,
        }
        const_color = cc_map.get(color_checker, cv2.ccm.COLORCHECKER_MACBETH)

        model = cv2.ccm.ColorCorrectionModel(patches, const_color)
        model.setCcmType(enums.CCM_CCM_TYPE.resolve(ccm_type))
        model.setColorSpace(enums.CCM_COLOR_SPACE.resolve(color_space))
        model.setDistance(enums.CCM_DISTANCE.resolve(distance))
        model.setLinearization(enums.CCM_LINEARIZATION.resolve(linearization))
        model.setLinearizationGamma(float(linearization_gamma))
        model.setLinearizationDegree(int(linearization_degree))

        if saturated_lower != 0.0 or saturated_upper != 0.0:
            model.setSaturatedThreshold(float(saturated_lower), float(saturated_upper))

        model.compute()

        ccm_matrix = model.getColorCorrectionMatrix()
        loss = model.getLoss()

        return io.NodeOutput(model, np.asarray(ccm_matrix), float(loss))


# ---------------------------------------------------------------------------
# Phase 2: apply fitted CCM to an image
# ---------------------------------------------------------------------------

class ApplyColorCorrection(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_ApplyColorCorrection",
            display_name="CV Apply Color Correction",
            category=CATEGORY,
            description="Applies a fitted Color Correction Model to an image. "
                        "Connect the 'ccm_model' output from 'Create CCM "
                        "Model' here.",
            inputs=[
                io.Image.Input("image",
                    tooltip="Input image to correct. A batch is processed "
                            "frame by frame."),
                CCMModel.Input("ccm_model",
                    tooltip="Fitted model from 'Create CCM Model'."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, ccm_model) -> io.NodeOutput:
        def fn(f):
            result = ccm_model.correctImage(f)
            return np.clip(result, 0, 255).astype(np.uint8) if f.dtype == np.uint8 else np.clip(result, 0, 1).astype(np.float32)
        return io.NodeOutput(_map_frames(image, fn))


# ---------------------------------------------------------------------------
# Phase 2: inspect CCM results
# ---------------------------------------------------------------------------

class GetCCMMatrix(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_GetCCMMatrix",
            display_name="CV Get CCM Matrix",
            category=CATEGORY,
            description="Extracts the computed Color Correction Matrix from a "
                        "fitted model. The matrix is 3x3 (CCM_LINEAR) or "
                        "4x3 (CCM_AFFINE).",
            inputs=[
                CCMModel.Input("ccm_model",
                    tooltip="Fitted model from 'Create CCM Model'."),
            ],
            outputs=[NPArray.Output(display_name="ccm_matrix")],
        )

    @classmethod
    def execute(cls, ccm_model) -> io.NodeOutput:
        matrix = ccm_model.getColorCorrectionMatrix()
        return io.NodeOutput(np.asarray(matrix))


class GetCCMLoss(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_GetCCMLoss",
            display_name="CV Get CCM Loss",
            category=CATEGORY,
            description="Returns the loss (fit quality) of a fitted Color "
                        "Correction Model. Lower values indicate better "
                        "color accuracy.",
            inputs=[
                CCMModel.Input("ccm_model",
                    tooltip="Fitted model from 'Create CCM Model'."),
            ],
            outputs=[io.Float.Output(display_name="loss")],
        )

    @classmethod
    def execute(cls, ccm_model) -> io.NodeOutput:
        return io.NodeOutput(float(ccm_model.getLoss()))


NODES = [DetectColorChecker, CreateCCMModel,
         ApplyColorCorrection, GetCCMMatrix, GetCCMLoss]
