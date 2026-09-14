# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# Adapted from geroldmeisinger/opencv-comfyui (GPL-3.0).
# See LICENSE for the full GNU GPL v3 text.

from .opencv_nodes import comfy_entrypoint

WEB_DIRECTORY = "./web"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
