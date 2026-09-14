# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Curated nodes for the opencv-contrib modules whose APIs cannot be generated.

The generated wrappers in `lowlevel.py` cover every contrib FUNCTION (see
`registry_contrib.py`). What they cannot cover is the other half of these
modules, which is exposed as stateful C++ CLASSES built by a `create*` factory:

* `cv2.xphoto.createSimpleWB()` / `createGrayworldWB()` / `createLearningBasedWB()`
* `cv2.ximgproc.createSuperpixelSLIC()` / `createSuperpixelLSC()` /
  `createSuperpixelSEEDS()` / `createScanSegment()`
* `cv2.ximgproc.createFastLineDetector()` / `createEdgeDrawing()`
* `cv2.saliency.StaticSaliency*_create()`
* `cv2.bgsegm.createBackgroundSubtractor*()` (+ the core MOG2 / KNN)
* `cv2.img_hash.*_create()` (for its `compare`, which the free functions lack)

`cv2.quality` used to live here too; it now has its own module (`quality.py`),
which covers the static `Quality*_compute` AND the class API this file could
not reach.

Those objects cannot travel through a ComfyUI graph: they are not ndarrays, and
because ComfyUI CACHES node outputs, a re-execution would keep mutating the same
live object (the hazard `segmentation.py` documents for MOG2/KNN). So each node
here creates the object, configures it, uses it and drops it INSIDE one
`execute()` - the object never becomes graph state.

That is also why several of these nodes deliberately aggregate more than one cv2
call: a lone `create*` node would output an unusable handle, and a lone `.apply()`
node would need one. A background subtractor in particular is only meaningful
over a SEQUENCE, so 'CV Background Subtract (Batch)' consumes the whole IMAGE
batch in one execution instead of asking the user to thread a model frame to
frame. Each node stays generic and reusable and emits DATA only -
labels, masks, scores - leaving visualization to the existing preview/overlay
nodes.
"""

import logging

import cv2
import numpy as np
import torch

import comfy.utils
from comfy_api.latest import io

from . import imageutils, polytype
from .contours import Contours
from .convert import NPArray
from .features import _FLOW_OUTPUT_SCHEMA, _flow_outputs, _to_gray_u8

CATEGORY = "image/CV/contrib"


def _require(module: str, attr: str = None):
    """The cv2 submodule, or a clear error naming the repair tool.

    The contrib stub PACKAGES ship with every opencv wheel, so `cv2.xphoto`
    imports fine even on a core-only build and only turns out to be empty at
    call time. Installing plain opencv-python over opencv-contrib-python(-headless)
    overwrites the shared cv2 binary, which is exactly how this install lost
    its contrib modules - so say so instead of raising a bare AttributeError.
    """
    mod = getattr(cv2, module, None)
    if mod is not None and (attr is None or hasattr(mod, attr)):
        return mod
    raise RuntimeError(
        f"cv2.{module}{'.' + attr if attr else ''} is not available in this "
        f"OpenCV build ({cv2.__version__}). The opencv-contrib modules are "
        "missing - most likely plain opencv-python was installed over "
        "opencv-contrib-python(-headless) (they share one site-packages/cv2). Stop the "
        "ComfyUI server and run: python tools/repair_opencv_contrib.py --apply")


def _bgr_frames(image):
    """Any image-ish input -> list of uint8 BGR frames."""
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            return imageutils.image_batch_to_bgr(image)
        return [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                for f in imageutils.mask_batch_to_u8(image)]
    arr = np.asarray(polytype.resolve(image)[0])
    if arr.ndim == 2:
        return [cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)]
    if arr.ndim == 4:
        return [np.ascontiguousarray(f) for f in arr]
    return [np.ascontiguousarray(arr)]


def _one_bgr(image):
    """Single uint8 BGR frame (frame 0 of a batch)."""
    frames = _bgr_frames(image)
    if not frames:
        raise ValueError("empty image input")
    f = frames[0]
    if f.dtype != np.uint8:
        f = np.clip(f * 255.0 if f.max() <= 1.0 else f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(f)


# ---------------------------------------------------------------------------
# White balance (cv2.xphoto)
# ---------------------------------------------------------------------------

class WhiteBalance(io.ComfyNode):
    """The three cv2.xphoto white balancers behind one method dropdown.

    They share an interface (`WhiteBalancer.balanceWhite`) and a purpose, and
    the right one is a per-image judgement call, so three near-identical nodes
    would only make the choice harder to compare.
    """

    @classmethod
    def define_schema(cls):
        template = polytype.match_template(types=polytype.IMAGE_ONLY)
        return io.Schema(
            node_id="CV_XPhotoWhiteBalance",
            display_name="CV White Balance (xphoto)",
            category=CATEGORY,
            description="Removes a colour cast automatically (cv2.xphoto). "
                        "'simple' clips a percentage of the extreme pixels per "
                        "channel and stretches the rest - fast and predictable. "
                        "'grayworld' assumes the scene averages to gray, which "
                        "suits mixed scenes but over-corrects a legitimately "
                        "one-coloured image (a sunset, a forest). "
                        "'learning-based' uses a small built-in regression "
                        "model and is the most robust on real photographs. "
                        "Colour-cast removal is what makes a reference photo "
                        "usable as a style/colour source, so this is usually a "
                        "PRE-processing step. Needs a 3-channel image.",
            inputs=[
                polytype.match_input("image", template,
                    tooltip="Image with an unwanted colour cast. Must be "
                            "3-channel; the output echoes this input's format."),
                io.Combo.Input("method",
                    options=["simple", "grayworld", "learning-based"],
                    default="simple",
                    tooltip="Which balancer to use. Try 'simple' first; switch "
                            "to 'learning-based' when the scene is genuinely "
                            "dominated by one colour and 'grayworld' washes it "
                            "out."),
                io.Float.Input("clip_percent", default=2.0, min=0.0, max=49.0,
                    step=0.5, optional=True, advanced=True,
                    tooltip="'simple' only: percentage of the darkest and "
                            "brightest pixels ignored per channel before "
                            "stretching. 0 uses the absolute extremes, so a "
                            "single blown pixel can ruin the balance; 1-5 is "
                            "the useful range."),
                io.Float.Input("saturation_threshold", default=0.9, min=0.0,
                    max=1.0, step=0.01, optional=True, advanced=True,
                    tooltip="'grayworld' / 'learning-based' only: pixels whose "
                            "saturation exceeds this are treated as already "
                            "clipped and excluded from the estimate. Lower it "
                            "to ignore more of a strongly coloured subject."),
            ],
            outputs=[
                polytype.match_output(template, "balanced",
                    tooltip="White-balanced image, in the same format as the "
                            "input."),
            ],
        )

    @classmethod
    def execute(cls, image, method, clip_percent=2.0,
                saturation_threshold=0.9) -> io.NodeOutput:
        xphoto = _require("xphoto", "createSimpleWB")
        bgr = _one_bgr(image)
        if method == "grayworld":
            wb = xphoto.createGrayworldWB()
            wb.setSaturationThreshold(float(saturation_threshold))
        elif method == "learning-based":
            wb = xphoto.createLearningBasedWB()
            wb.setSaturationThreshold(float(saturation_threshold))
        else:
            wb = xphoto.createSimpleWB()
            # cv2 expects a FRACTION of pixels to clip at each end
            wb.setP(float(clip_percent))
        out = wb.balanceWhite(bgr)
        _, tag = polytype.resolve(image)
        return io.NodeOutput(polytype.restore(np.asarray(out), tag))


# ---------------------------------------------------------------------------
# Superpixels (cv2.ximgproc)
# ---------------------------------------------------------------------------

class Superpixels(io.ComfyNode):
    """All four ximgproc superpixel algorithms behind one dropdown.

    Every one of them is `create* -> iterate -> getLabels`, so exposing the
    factories separately would produce un-wireable handles. Outputs are DATA
    (a label map + a contour mask + the count); draw them with the existing
    preview / overlay nodes.
    """

    _SLIC = {"SLIC": "SLIC", "SLICO": "SLICO", "MSLIC": "MSLIC"}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_Superpixels",
            display_name="CV Superpixels",
            category=CATEGORY,
            description="Splits the image into perceptually uniform regions "
                        "that respect edges (cv2.ximgproc). Superpixels are a "
                        "much better unit than a pixel grid for region-wise "
                        "recolouring, palette work, selective masking or "
                        "cartoon flattening: each region is already colour- "
                        "and edge-coherent. Outputs the label map (one integer "
                        "per pixel) plus a contour mask of the region "
                        "boundaries. Feed the image and the labels to 'CV "
                        "Reduce Array By Label' (mean) for one colour per "
                        "region, then paint it back with 'CV Take By Index' "
                        "(the 'CV Superpixel Paint' subgraph blueprint does "
                        "exactly that); or preview the labels with 'Preview CV "
                        "Array' (normalize).",
            inputs=[
                polytype.poly_input("image", types=polytype.IMAGE_ONLY,
                    tooltip="Image to segment (3-channel). Frame 0 of a batch."),
                io.Combo.Input("algorithm",
                    options=["SLIC", "SLICO", "MSLIC", "LSC", "SEEDS",
                             "ScanSegment"],
                    default="SLICO",
                    tooltip="SLIC needs an explicit compactness; SLICO (the "
                            "default) tunes it per region; MSLIC adds "
                            "manifold-aware sizing. LSC preserves fine "
                            "structure best. SEEDS and ScanSegment take a "
                            "TARGET COUNT instead of a region size and are the "
                            "fastest."),
                io.Int.Input("region_size", default=20, min=4, max=512,
                    tooltip="SLIC/SLICO/MSLIC/LSC: average superpixel width in "
                            "pixels - the region COUNT follows from the image "
                            "size. Ignored by SEEDS/ScanSegment, which use "
                            "num_superpixels instead."),
                io.Int.Input("num_superpixels", default=200, min=2, max=10000,
                    tooltip="SEEDS/ScanSegment: how many regions to aim for "
                            "(approximate). Ignored by the others."),
                io.Int.Input("iterations", default=10, min=1, max=100,
                    tooltip="Refinement passes. More = better-fitting "
                            "boundaries, linearly slower; 4-10 is usually "
                            "enough."),
                io.Float.Input("compactness", default=10.0, min=0.1, max=100.0,
                    step=0.1, optional=True, advanced=True,
                    tooltip="SLIC/MSLIC: how strongly regions are pulled "
                            "towards a compact shape. High = tidy squarish "
                            "blobs that cut across edges; low = ragged regions "
                            "that hug edges. For LSC this is the 'ratio' "
                            "(0-1); SLICO ignores it entirely."),
                io.Boolean.Input("enforce_connectivity", default=True,
                    optional=True, advanced=True,
                    tooltip="Merge stray disconnected fragments into their "
                            "neighbours so every label is one connected "
                            "region. Turn off only if you want the raw "
                            "clustering."),
            ],
            outputs=[
                NPArray.Output(display_name="labels",
                    tooltip="int32 (H,W) map: the region index each pixel "
                            "belongs to, from 0 to count-1."),
                NPArray.Output(display_name="contours",
                    tooltip="uint8 (H,W) mask, 255 on region boundaries - "
                            "overlay it to see the segmentation."),
                io.Int.Output(display_name="count",
                    tooltip="Number of superpixels actually produced (may "
                            "differ from the requested count)."),
            ],
        )

    @classmethod
    def execute(cls, image, algorithm, region_size, num_superpixels,
                iterations, compactness=10.0,
                enforce_connectivity=True) -> io.NodeOutput:
        ximgproc = _require("ximgproc", "createSuperpixelSLIC")
        bgr = _one_bgr(image)
        h, w = bgr.shape[:2]

        if algorithm in cls._SLIC:
            algo = getattr(ximgproc, cls._SLIC[algorithm])
            sp = ximgproc.createSuperpixelSLIC(bgr, algo, int(region_size),
                                               float(compactness))
            sp.iterate(int(iterations))
        elif algorithm == "LSC":
            # LSC's 'ratio' is 0-1, while the shared compactness widget is 0-100
            sp = ximgproc.createSuperpixelLSC(bgr, int(region_size),
                                              float(compactness) / 100.0)
            sp.iterate(int(iterations))
        elif algorithm == "SEEDS":
            sp = ximgproc.createSuperpixelSEEDS(w, h, bgr.shape[2],
                                                int(num_superpixels), 4)
            sp.iterate(bgr, int(iterations))
        else:  # ScanSegment
            sp = ximgproc.createScanSegment(w, h, int(num_superpixels))
            sp.iterate(bgr)

        if enforce_connectivity and hasattr(sp, "enforceLabelConnectivity"):
            sp.enforceLabelConnectivity()
        labels = np.ascontiguousarray(np.asarray(sp.getLabels(), np.int32))
        contours = np.ascontiguousarray(np.asarray(sp.getLabelContourMask(),
                                                   np.uint8))
        return io.NodeOutput(labels, contours, int(sp.getNumberOfSuperpixels()))


# ---------------------------------------------------------------------------
# Saliency (cv2.saliency)
# ---------------------------------------------------------------------------

class Saliency(io.ComfyNode):
    """Both static saliency detectors, plus their adaptive binarization.

    `computeSaliency` and `computeBinaryMap` are two calls on the same handle,
    so splitting them would force the handle into the graph. Returning both maps
    keeps the node data-only while staying usable in one drop.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_Saliency",
            display_name="CV Saliency",
            category=CATEGORY,
            description="Estimates which parts of the image draw the eye "
                        "(cv2.saliency) - no model file needed. Useful for "
                        "attention-weighted crops, auto-framing a subject, or "
                        "steering a mask towards what matters. "
                        "'spectral-residual' is a fast global frequency-domain "
                        "estimate; 'fine-grained' works on local centre-surround "
                        "contrast and gives sharper object outlines. Outputs a "
                        "0-1 saliency MASK plus an auto-thresholded binary "
                        "mask.",
            inputs=[
                polytype.poly_input("image", types=polytype.IMAGE_ONLY,
                    tooltip="Image to analyse (3-channel). Frame 0 of a batch."),
                io.Combo.Input("method",
                    options=["fine-grained", "spectral-residual"],
                    default="fine-grained",
                    tooltip="'fine-grained' follows object boundaries more "
                            "closely; 'spectral-residual' is faster and "
                            "coarser, highlighting whatever is unusual in the "
                            "frequency spectrum."),
            ],
            outputs=[
                io.Mask.Output(display_name="saliency",
                    tooltip="0-1 MASK: how salient each pixel is. Feed it to "
                            "'CV Color Map' for a heatmap, or use it "
                            "directly as a soft weight."),
                io.Mask.Output(display_name="binary",
                    tooltip="Auto-thresholded 0/1 MASK of the salient region "
                            "(Otsu-style, computed by OpenCV). Empty is a "
                            "valid result on a flat image."),
            ],
        )

    @classmethod
    def execute(cls, image, method) -> io.NodeOutput:
        saliency = _require("saliency", "StaticSaliencyFineGrained_create")
        bgr = _one_bgr(image)
        if method == "spectral-residual":
            det = saliency.StaticSaliencySpectralResidual_create()
        else:
            det = saliency.StaticSaliencyFineGrained_create()
        ok, smap = det.computeSaliency(bgr)
        if not ok or smap is None:
            # a detector must never halt the workflow
            smap = np.zeros(bgr.shape[:2], np.float32)
        smap = np.nan_to_num(np.asarray(smap, np.float32), nan=0.0,
                             posinf=1.0, neginf=0.0)

        binary = np.zeros(bgr.shape[:2], np.uint8)
        try:
            ok_b, b = det.computeBinaryMap(smap)
            if ok_b and b is not None:
                binary = np.asarray(b, np.uint8)
        except cv2.error:
            # spectral-residual's binarization is not implemented in every
            # build; fall back to Otsu on the saliency map itself
            u8 = np.clip(smap * 255.0, 0, 255).astype(np.uint8)
            _, binary = cv2.threshold(u8, 0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        smap = np.clip(smap, 0.0, 1.0)
        return io.NodeOutput(
            torch.from_numpy(np.ascontiguousarray(smap))[None],
            torch.from_numpy(
                np.ascontiguousarray((binary > 0).astype(np.float32)))[None])


# ---------------------------------------------------------------------------
# Background subtraction over a batch (cv2.bgsegm + core MOG2/KNN)
# ---------------------------------------------------------------------------

class BackgroundSubtractBatch(io.ComfyNode):
    """Runs a stateful cv2 background subtractor over a whole IMAGE batch.

    A background subtractor is only meaningful over a SEQUENCE: `apply()`
    mutates an internal model frame by frame. That model cannot cross a ComfyUI
    graph (not an ndarray, and ComfyUI's output caching would re-apply it on
    every re-execution - the hazard `segmentation.py` documents), so the whole
    sequence is consumed inside ONE execute() and only ndarrays come out. This
    complements 'CV Background Model (Update)', which threads a hand-rolled
    single-Gaussian model frame-to-frame for per-frame graphs.
    """

    _ALGOS = {
        "MOG2 (core)": ("", "createBackgroundSubtractorMOG2"),
        "KNN (core)": ("", "createBackgroundSubtractorKNN"),
        "MOG (bgsegm)": ("bgsegm", "createBackgroundSubtractorMOG"),
        "GMG (bgsegm)": ("bgsegm", "createBackgroundSubtractorGMG"),
        "CNT (bgsegm)": ("bgsegm", "createBackgroundSubtractorCNT"),
        "GSOC (bgsegm)": ("bgsegm", "createBackgroundSubtractorGSOC"),
        "LSBP (bgsegm)": ("bgsegm", "createBackgroundSubtractorLSBP"),
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_BackgroundSubtractBatch",
            display_name="CV Background Subtract (Batch)",
            category=CATEGORY,
            description="Learns the background of a CLIP and returns a "
                        "foreground mask per frame - the practical way to pull "
                        "moving subjects out of static-camera footage without "
                        "a segmentation model. Feed the whole batch (core "
                        "'Batch Images' or a video loader) in frame order; the "
                        "subtractor is created, run over the sequence and "
                        "dropped inside this one node, because a stateful cv2 "
                        "model cannot travel through a cached graph. "
                        "'warmup_frames' lets the model settle before the "
                        "masks are kept, so the first frames are not all "
                        "foreground. START WITH MOG2, and give the model "
                        "FRAMES: measured on a 200-frame fixed-camera clip, "
                        "against a temporal-median reference calling 5.0% of "
                        "the frame moving, MOG2 reads 4.6% and CNT/GSOC/LSBP "
                        "read 7.9/9.3/13.3% - one range, a fair comparison. "
                        "Cut the SAME clip to 50 frames and those three jump "
                        "to 16.0/12.5/16.0% while MOG2 barely moves: they are "
                        "not less accurate, they are unconverged.",
            inputs=[
                io.Image.Input("images",
                    tooltip="Batch of frames IN ORDER from a fixed camera. At "
                            "least 2; the more the model sees, the cleaner the "
                            "background estimate."),
                io.Combo.Input("algorithm", options=list(cls._ALGOS),
                    default="MOG2 (core)",
                    tooltip="MOG2/KNN (core OpenCV) are fast, adaptive, give a "
                            "background plate, and converge in a few dozen "
                            "frames - the right default for clip-length "
                            "footage. MOG is the classic mixture model and is "
                            "the most conservative. CNT is extremely fast. "
                            "GSOC and LSBP are the most accurate ON LONG "
                            "SEQUENCES; MEASURED, all three of CNT/GSOC/LSBP "
                            "read 3-4x the reference on a 50-frame clip and "
                            "only settle after several hundred frames, so "
                            "prefer MOG2 unless the clip is long. "
                            "GMG needs many initialization frames and "
                            "provides no plate."),
                io.Int.Input("warmup_frames", default=0, min=0, max=1000,
                    tooltip="Frames used ONLY to train the model before masks "
                            "start being collected. The output then has this "
                            "many fewer frames. 0 keeps every frame, including "
                            "the noisy first ones. ~12 is plenty for MOG2/KNN. "
                            "It cannot rescue CNT/GSOC/LSBP on a short clip: "
                            "those need hundreds of frames of actual "
                            "sequence, and spending them on warmup just "
                            "leaves fewer to keep - check "
                            "'foreground_fraction' rather than assuming."),
                io.Float.Input("learning_rate", default=-1.0, min=-1.0, max=1.0,
                    step=0.01, optional=True, advanced=True,
                    tooltip="How fast the background adapts, per frame. -1 "
                            "(the default) lets the algorithm choose; 0 freezes "
                            "the model after the warmup, which is what you want "
                            "when a subject stops moving and would otherwise be "
                            "absorbed into the background; 1 makes every frame "
                            "the new background."),
                io.Boolean.Input("detect_shadows", default=True, optional=True,
                    advanced=True,
                    tooltip="MOG2/KNN only: mark shadows separately (they come "
                            "out as grey 127 rather than white 255) - see "
                            "'shadows_as_foreground'."),
                io.Boolean.Input("shadows_as_foreground", default=False,
                    optional=True, advanced=True,
                    tooltip="Count detected shadows as part of the subject. "
                            "Off (the default) keeps the mask to the object "
                            "itself, which is usually what a matte needs."),
            ],
            outputs=[
                io.Mask.Output(display_name="foreground",
                    tooltip="One 0/1 MASK per frame (a MASK batch): the moving "
                            "/ new pixels. Clean it up with morphology and "
                            "'CV Keep Largest Component'."),
                NPArray.Output(display_name="background",
                    tooltip="The learned background plate, always BGR uint8. "
                            "MOG2/KNN/GSOC/LSBP provide one directly; CNT's "
                            "is single-channel and is promoted to BGR; MOG "
                            "and GMG provide none, so the temporal median of "
                            "the clip is returned instead. Preview it with "
                            "'Preview CV Array'."),
                io.Float.Output(display_name="foreground_fraction",
                    tooltip="Average fraction of pixels flagged as foreground "
                            "(0-1) across the kept frames - the number to "
                            "sanity-check every run against. On static-camera "
                            "footage a good result is a small minority, "
                            "roughly 0.05-0.2. Above ~0.3 the model has not "
                            "settled: use a longer clip or switch to MOG2 "
                            "rather than raising warmup_frames. Near 0 means "
                            "nothing moved."),
            ],
        )

    @classmethod
    def execute(cls, images, algorithm, warmup_frames, learning_rate=-1.0,
                detect_shadows=True, shadows_as_foreground=False) -> io.NodeOutput:
        module, factory = cls._ALGOS[algorithm]
        if module:
            src = _require(module, factory)
        else:
            src = cv2
        frames = _bgr_frames(images)
        if len(frames) < 2:
            raise ValueError(
                f"Background subtraction needs at least 2 frames, got "
                f"{len(frames)}. Stack a clip into one batch (core 'Batch "
                "Images' or a video loader) first.")
        if warmup_frames >= len(frames):
            raise ValueError(
                f"warmup_frames ({warmup_frames}) must be smaller than the "
                f"batch size ({len(frames)}) or no frames would be kept.")

        if not module:  # core MOG2 / KNN take detectShadows
            sub = getattr(src, factory)(detectShadows=bool(detect_shadows))
        else:
            sub = getattr(src, factory)()

        rate = float(learning_rate)
        masks = []
        pbar = comfy.utils.ProgressBar(len(frames))
        # a preview send skips the bar's own throttle - keep them to ~20 a run
        preview_every = max(1, len(frames) // 20)
        for i, f in enumerate(frames):
            # copy: cv2 subtractors may write into the frame, which is another
            # node's cached output (same rule lowlevel.py follows)
            fg = sub.apply(np.ascontiguousarray(f.copy()), learningRate=rate)
            if i >= int(warmup_frames):
                fg = np.asarray(fg, np.uint8)
                if shadows_as_foreground:
                    masks.append((fg > 0).astype(np.float32))
                else:
                    # 127 = shadow in MOG2/KNN; anything below 255 is not a
                    # confident foreground pixel
                    masks.append((fg >= 255).astype(np.float32))
            if i % preview_every == 0 or i == len(frames) - 1:
                pbar.update_absolute(i + 1, None,
                                     imageutils.progress_preview(fg))
            else:
                pbar.update(1)

        try:
            bg = sub.getBackgroundImage()
            bg = np.asarray(bg, np.uint8) if bg is not None else None
            if bg is not None and bg.size == 0:
                bg = None       # GMG returns an empty Mat
            # CNT hands back a SINGLE-CHANNEL plate while every other algorithm
            # returns BGR. Promote it so the output type does not depend on
            # which algorithm the user picked.
            if bg is not None and bg.ndim == 2:
                bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        except cv2.error:
            bg = None  # MOG / GMG do not implement it
        if bg is None:
            # a median over the sequence is the natural stand-in for the plate
            bg = np.median(np.stack(frames, 0).astype(np.float32), 0)
            bg = np.clip(np.rint(bg), 0, 255).astype(np.uint8)

        fraction = float(np.mean([m.mean() for m in masks])) if masks else 0.0
        return io.NodeOutput(
            torch.from_numpy(np.ascontiguousarray(np.stack(masks, 0))),
            np.ascontiguousarray(bg), fraction)


# ---------------------------------------------------------------------------
# Perceptual hashing (cv2.img_hash)
# ---------------------------------------------------------------------------

class ImageHashCompare(io.ComfyNode):
    """Hashes two images AND compares them in one node.

    The free `img_hash.*` functions (already exposed as low-level wrappers)
    return a raw byte array, but the `compare` that makes it MEAN anything only
    exists on the class. Hash-and-compare is also the only shape that is
    directly usable in a workflow: a lone hash is not a decision.
    """

    _ALGOS = {
        "pHash (DCT, best all-round)": "PHash",
        "average (fastest)": "AverageHash",
        "block mean": "BlockMeanHash",
        "colour moment (rotation-tolerant)": "ColorMomentHash",
        "Marr-Hildreth (edge structure)": "MarrHildrethHash",
        "radial variance (rotation-tolerant)": "RadialVarianceHash",
    }

    # cv2's ImgHashBase.compare() is NOT consistent across the family: most
    # return a DISTANCE (0 = identical, bigger = more different), but
    # RadialVarianceHash returns a peak-correlation SIMILARITY where 1.0 means
    # identical and smaller means less alike. Measured on cv2 5.0.0
    # (temp/_probe_hash.py): identical images give 0.0 for every algorithm
    # except RadialVarianceHash, which gives 1.0. Left as-is, the node's
    # "distance <= threshold" flag would be exactly INVERTED for that one
    # algorithm - it would call every unrelated pair similar. So its score is
    # converted to a distance here and the node keeps ONE contract: 0 =
    # identical, larger = more different, for every algorithm.
    _SIMILARITY_ALGOS = {"RadialVarianceHash"}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_ImageHashCompare",
            display_name="CV Image Hash Compare",
            category=CATEGORY,
            description="Decides whether two images are the SAME PICTURE "
                        "(cv2.img_hash), tolerating rescaling, compression and "
                        "mild edits - unlike a pixel diff. Use it to skip "
                        "duplicate outputs, detect that a generation barely "
                        "changed, or find which reference a result resembles. "
                        "Outputs the distance, a similarity flag against your "
                        "threshold, and both hashes. The distance always reads "
                        "the SAME WAY - 0 = identical, larger = more different "
                        "- for every algorithm (cv2 itself returns a similarity "
                        "for 'radial variance'; this node converts it). The "
                        "SCALE still differs: the bit-based hashes report a "
                        "Hamming distance in BITS (~2 = a rescaled copy, ~30 = "
                        "unrelated), while 'colour moment' and 'radial "
                        "variance' report small floating-point distances - so "
                        "tune 'threshold' per algorithm.",
            inputs=[
                polytype.poly_input("image_a", types=polytype.IMAGE_ONLY,
                    tooltip="First image (3-channel). Frame 0 of a batch."),
                polytype.poly_input("image_b", types=polytype.IMAGE_ONLY,
                    tooltip="Second image. It may be a different SIZE - that is "
                            "the point of a perceptual hash."),
                io.Combo.Input("algorithm", options=list(cls._ALGOS),
                    default="pHash (DCT, best all-round)",
                    tooltip="pHash is the robust default. 'average' is fastest "
                            "but brightness-sensitive. 'colour moment' and "
                            "'radial variance' survive ROTATION. "
                            "'Marr-Hildreth' is the most sensitive to "
                            "structural edits."),
                io.Float.Input("threshold", default=10.0, min=0.0, max=1000.0,
                    step=0.5,
                    tooltip="Distance at or below which the images count as "
                            "similar. For the bit-based hashes this is in bits "
                            "(~5 = near-identical, ~10 = same scene, >20 = "
                            "unrelated); for 'colour moment' use a much smaller "
                            "value (~1)."),
            ],
            outputs=[
                io.Float.Output(display_name="distance",
                    tooltip="How far apart the two hashes are, ALWAYS in the "
                            "same direction: 0 = identical, larger = more "
                            "different. The useful scale depends on the "
                            "algorithm (see the node description)."),
                io.Boolean.Output(display_name="similar",
                    tooltip="True when distance <= threshold. Feed it to a "
                            "'Basic data handling: IfElse' to pick between two "
                            "branches of the graph."),
                NPArray.Output(display_name="hash_a",
                    tooltip="Raw hash of image_a - keep it to compare against "
                            "many candidates without re-hashing."),
                NPArray.Output(display_name="hash_b",
                    tooltip="Raw hash of image_b."),
            ],
        )

    @classmethod
    def execute(cls, image_a, image_b, algorithm, threshold) -> io.NodeOutput:
        img_hash = _require("img_hash", "PHash_create")
        name = cls._ALGOS[algorithm]
        hasher = getattr(img_hash, name + "_create")()
        ha = np.asarray(hasher.compute(_one_bgr(image_a)))
        hb = np.asarray(hasher.compute(_one_bgr(image_b)))
        score = float(hasher.compare(ha, hb))
        if name in cls._SIMILARITY_ALGOS:
            # 1.0 = identical -> 0.0 = identical, so every algorithm reports a
            # distance and 'distance <= threshold' means the same thing
            score = 1.0 - score
        distance = float(np.clip(score, 0.0, None))
        return io.NodeOutput(distance, bool(distance <= float(threshold)),
                             ha, hb)


class StereoDisparityWLS(io.ComfyNode):
    """SGBM + its right matcher + the WLS filter, all owned by one execute().

    `ximgproc.createDisparityWLSFilter(matcher)` is only meaningful together
    with `createRightMatcher(matcher)` of the SAME instance: the filter reads
    both disparity maps to run the left-right consistency check that produces
    its confidence map. Three separate nodes would have to pass live cv2
    objects through the graph, which is exactly what this module exists to
    avoid, so the SGBM widgets are repeated here (they mirror
    'CV Stereo Disparity (SGBM)') and the node owns all three objects.
    """

    _MODES = {
        "SGBM (balanced)": cv2.STEREO_SGBM_MODE_SGBM,
        "SGBM 3-way (faster)": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        "HH (full-scale, slow)": cv2.STEREO_SGBM_MODE_HH,
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_StereoDisparityWLS",
            display_name="CV Stereo Disparity (WLS filtered)",
            category=CATEGORY,
            description="Disparity from a RECTIFIED stereo pair, cleaned with "
                        "the ximgproc weighted-least-squares filter: SGBM is "
                        "run left-to-right AND right-to-left, the two maps are "
                        "consistency-checked, and the result is smoothed while "
                        "sticking to the left image's edges. It fills the "
                        "holes SGBM leaves in low-texture areas WITHOUT "
                        "inventing structure the way inpainting does - on the "
                        "example driving pair the valid fraction goes from "
                        "~65% to ~95% - and it also outputs a per-pixel "
                        "confidence map, which is what lets downstream steps "
                        "keep only trustworthy points ('CV Sample Array At "
                        "Points' + 'CV Filter Points 3D By Mask'). The pair "
                        "MUST be rectified first (cv2.stereoRectify + "
                        "cv2.initUndistortRectifyMap + cv2.remap). Preview "
                        "with 'Preview CV Array' (normalize / heatmap).",
            inputs=[
                polytype.poly_input("left", types=polytype.IMAGE_ONLY,
                    tooltip="Left rectified frame (grayscaled internally)."),
                polytype.poly_input("right", types=polytype.IMAGE_ONLY,
                    tooltip="Right rectified frame; resized to left if the "
                            "sizes differ."),
                io.Int.Input("num_disparities", default=64, min=16, max=512,
                    step=16,
                    tooltip="Disparity search range (0..num_disparities), "
                            "rounded up to a multiple of 16. Larger covers "
                            "nearer objects / wider baselines but is slower."),
                io.Int.Input("block_size", default=7, min=1, max=21,
                    tooltip="Matched block size in pixels (forced odd; 3-11 is "
                            "typical). Smaller = more detail but noisier."),
                io.Int.Input("min_disparity", default=0, min=-256, max=256,
                    tooltip="Smallest disparity to search from (usually 0)."),
                io.Combo.Input("mode", options=list(cls._MODES),
                    default="SGBM (balanced)",
                    tooltip="SGBM is the standard; 3-way is faster at lower "
                            "quality; HH runs the full-scale two-pass (best, "
                            "slowest)."),
                io.Float.Input("wls_lambda", default=8000.0, min=0.0, max=100000.0,
                    step=100.0,
                    tooltip="Regularization strength: how strongly the filtered "
                            "disparity is smoothed towards the edges of the "
                            "left image. 8000 is the OpenCV-recommended "
                            "default; larger = smoother, more hole filling, "
                            "more bleeding across depth edges."),
                io.Float.Input("wls_sigma_color", default=1.5, min=0.0, max=10.0,
                    step=0.1,
                    tooltip="How sensitive the filter is to left-image edges. "
                            "0.8-2.0 is the useful range: too low leaks "
                            "disparity across object boundaries, too high "
                            "makes it follow image noise."),
                io.Int.Input("uniqueness_ratio", default=10, min=0, max=100,
                    optional=True, advanced=True,
                    tooltip="Margin (%) by which the best match must beat the "
                            "runner-up to be accepted."),
                io.Int.Input("speckle_window_size", default=100, min=0, max=1000,
                    optional=True, advanced=True,
                    tooltip="Largest smooth disparity blob treated as speckle "
                            "noise and invalidated (0 = off)."),
                io.Int.Input("speckle_range", default=2, min=0, max=64,
                    optional=True, advanced=True,
                    tooltip="Max disparity variation within a speckle component."),
                io.Float.Input("lrc_threshold", default=24.0, min=0.0, max=256.0,
                    step=1.0, optional=True, advanced=True,
                    tooltip="Left-right consistency tolerance in 1/16 pixel "
                            "units: disagreements above it are marked "
                            "unconfident and re-filled by the filter."),
                io.Int.Input("depth_discontinuity_radius", default=0, min=0, max=32,
                    optional=True, advanced=True,
                    tooltip="Radius (px) around depth discontinuities used when "
                            "computing confidence. 0 = the filter's own "
                            "heuristic (block_size / 2)."),
                io.Int.Input("disp12_max_diff", default=-1, min=-1, max=64,
                    optional=True, advanced=True,
                    tooltip="SGBM's OWN left-right check, in pixels: matches "
                            "that disagree by more than this are invalidated "
                            "before the WLS filter ever sees them. -1 (cv2's "
                            "default) turns it off and leaves occlusion junk "
                            "in the map; 1 is the usual setting when the "
                            "disparity feeds an interpolation or a point "
                            "cloud."),
            ],
            outputs=[
                NPArray.Output(display_name="disparity",
                    tooltip="HxW float32 WLS-filtered disparity in pixels."),
                NPArray.Output(display_name="confidence",
                    tooltip="HxW float32 confidence map, 0-255; low where the "
                            "two matching directions disagreed (occlusions, "
                            "texture-less areas). Sample it at your points to "
                            "reject unreliable 3D."),
                NPArray.Output(display_name="valid",
                    tooltip="uint8 0/255 mask of pixels with a valid filtered "
                            "disparity."),
                NPArray.Output(display_name="disparity_raw",
                    tooltip="The unfiltered left-to-right SGBM disparity "
                            "(float32 pixels) - keep it to show what the "
                            "filter changed."),
            ],
        )

    @classmethod
    def execute(cls, left, right, num_disparities, block_size, min_disparity,
                mode, wls_lambda, wls_sigma_color, uniqueness_ratio=10,
                speckle_window_size=100, speckle_range=2, lrc_threshold=24.0,
                depth_discontinuity_radius=0, disp12_max_diff=-1) -> io.NodeOutput:
        ximgproc = _require("ximgproc", "createDisparityWLSFilter")
        lg = cv2.cvtColor(_one_bgr(left), cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(_one_bgr(right), cv2.COLOR_BGR2GRAY)
        if rg.shape != lg.shape:
            rg = cv2.resize(rg, (lg.shape[1], lg.shape[0]),
                            interpolation=cv2.INTER_AREA)

        nd = max(16, int(round(num_disparities / 16)) * 16)
        bs = max(1, block_size | 1)  # SGBM needs an odd block size
        matcher = cv2.StereoSGBM_create(
            minDisparity=min_disparity, numDisparities=nd, blockSize=bs,
            P1=8 * bs * bs, P2=32 * bs * bs,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size, speckleRange=speckle_range,
            disp12MaxDiff=int(disp12_max_diff),
            mode=cls._MODES[mode])
        right_matcher = ximgproc.createRightMatcher(matcher)

        # both matchers return int16 disparity scaled by 16; the filter wants
        # those RAW maps, not the float ones
        disp_left = matcher.compute(lg, rg)
        disp_right = right_matcher.compute(rg, lg)

        wls = ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(float(wls_lambda))
        wls.setSigmaColor(float(wls_sigma_color))
        wls.setLRCthresh(int(round(lrc_threshold)))
        if int(depth_discontinuity_radius) > 0:
            wls.setDepthDiscontinuityRadius(int(depth_discontinuity_radius))
        filtered = wls.filter(disp_left, lg, None, disp_right)
        confidence = np.asarray(wls.getConfidenceMap(), np.float32)

        disparity = np.asarray(filtered, np.float32) / 16.0
        valid = (disparity >= min_disparity).astype(np.uint8) * 255
        return io.NodeOutput(disparity, confidence, valid,
                             np.asarray(disp_left, np.float32) / 16.0)


class DisparityFilterWLS(io.ComfyNode):
    """The WLS filter applied to a disparity map from ANY source.

    'CV Stereo Disparity (WLS filtered)' owns its own SGBM, which is the right
    shape for the common case but leaves the filter unreachable for a map that
    came from StereoBM, quasi-dense stereo, a depth network or an earlier
    refinement pass. `createDisparityWLSFilterGeneric` is the entry point for
    those, and it is class-gated so the generator cannot reach it either.

    The trap it hides is the ROI. The matcher-bound filter derives one from the
    matcher's search range; the generic one defaults to the FULL FRAME, which
    feeds the invalid leftmost `num_disparities` columns into the solve. On the
    example gallery pair the two answers differ by 2109 px at the worst pixel -
    a silent, spectacular failure. `ximgproc.computeROI` is not bound in Python,
    so the rect has to come from somewhere else:

    * with a right-view map, a THROWAWAY StereoSGBM carrying the declared search
      range is enough - `createDisparityWLSFilter(dummy)` then computes OpenCV's
      own ROI and filters whatever maps it is handed, whichever matcher made
      them. This path also produces the confidence map.
    * without one, the bound factory ASSERTS (disparity_filters.cpp:302 wants a
      non-empty right map), so it is Generic(False) plus the rect measured off
      `getROI()`: x = min_disparity + num_disparities, y = 0,
      x + w = width + min(min_disparity, 0), h = height. Verified on five
      parameter sets; block_size does NOT enter it on this build.

    `block_size` is here for a second reason, and it is the one that actually
    cost time: the ROI is NOT the only thing the bound filter reads off the
    matcher. It also sets `depth_discontinuity_radius = ceil(block_size / 2)`
    (measured for block sizes 1..21; a StereoBM matcher gets the generic default
    5 instead), and being ONE off there moves the filtered map by 2111 px at the
    worst pixel - the same order as getting the ROI wrong, from a parameter that
    reads like a confidence-map detail. Both paths therefore derive it from the
    declared block size unless the widget overrides it.

    Confidence needs both directions, so it comes back as zeros on the
    no-right-map path rather than as a fake all-255 map.
    """

    _ROI_MODES = ["auto (skip the invalid left band)", "full frame (cv2 default)"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_DisparityFilterWLS",
            display_name="CV Disparity Filter (WLS)",
            category=CATEGORY,
            description="Runs the ximgproc weighted-least-squares filter over a "
                        "disparity map you already have, from ANY matcher: "
                        "StereoBM, quasi-dense stereo, a depth network, or an "
                        "earlier refinement pass. It fills holes and smooths "
                        "the map while sticking to the edges of the guide "
                        "image - on the example gallery pair a StereoBM map "
                        "goes from 22% to 54% valid. Wire disparity_right (the "
                        "right-to-left map, negative) to also get the "
                        "confidence map and the left-right consistency check. "
                        "Set num_disparities / min_disparity to the search "
                        "range the map was computed with: they only pick the "
                        "region to filter, but getting them wrong lets the "
                        "invalid left band poison the whole result. Use 'CV "
                        "Stereo Disparity (WLS filtered)' instead when you want "
                        "SGBM and the filter in one node. NOTE the filter's "
                        "prior is fronto-parallel, so a textureless SLANTED "
                        "surface (a road) is bent towards a staircase - 'CV "
                        "Disparity Interpolate (Edge-Aware)' fits a local plane "
                        "instead and is the better choice there.",
            inputs=[
                NPArray.Input("disparity",
                    tooltip="HxW float32 disparity in PIXELS (what every "
                            "disparity node in this pack outputs), not the "
                            "raw 1/16 int16 map cv2 returns. Invalid pixels "
                            "are the ones below min_disparity."),
                polytype.poly_input("guide", types=polytype.IMAGE_ONLY,
                    tooltip="The left rectified frame the disparity belongs "
                            "to; its edges are what the filter follows. "
                            "Grayscaled internally."),
                io.Float.Input("wls_lambda", default=8000.0, min=0.0,
                    max=100000.0, step=100.0,
                    tooltip="Regularization strength: how strongly the "
                            "filtered disparity is smoothed towards the edges "
                            "of the guide image. 8000 is the OpenCV-recommended "
                            "default; larger = smoother, more hole filling, "
                            "more bleeding across depth edges."),
                io.Float.Input("wls_sigma_color", default=1.5, min=0.0, max=10.0,
                    step=0.1,
                    tooltip="How sensitive the filter is to guide-image edges. "
                            "0.8-2.0 is the useful range: too low leaks "
                            "disparity across object boundaries, too high "
                            "makes it follow image noise."),
                NPArray.Input("disparity_right", optional=True,
                    tooltip="Optional HxW float32 right-to-left disparity in "
                            "pixels, NEGATIVE as ximgproc.createRightMatcher "
                            "produces it. With it the filter runs its "
                            "left-right consistency check and the confidence "
                            "output is real; without it confidence comes back "
                            "as zeros."),
                io.Int.Input("num_disparities", default=64, min=16, max=512,
                    step=16, optional=True, advanced=True,
                    tooltip="The search range the disparity map was computed "
                            "with. Used ONLY to place the region to filter - "
                            "the leftmost min_disparity + num_disparities "
                            "columns can never have been matched, and letting "
                            "them into the solve corrupts the whole map."),
                io.Int.Input("min_disparity", default=0, min=-256, max=256,
                    optional=True, advanced=True,
                    tooltip="The smallest disparity the map was searched from. "
                            "Also the threshold for the 'valid' output."),
                io.Int.Input("block_size", default=7, min=1, max=21,
                    optional=True, advanced=True,
                    tooltip="The block size the map was matched with (forced "
                            "odd). It sets depth_discontinuity_radius to "
                            "ceil(block_size / 2), exactly as the "
                            "matcher-bound filter does - and that is not a "
                            "cosmetic detail: one step off there changes the "
                            "filtered disparity by 2000+ px at the worst "
                            "pixel. Leave it matching the matcher."),
                io.Float.Input("lrc_threshold", default=24.0, min=0.0, max=256.0,
                    step=1.0, optional=True, advanced=True,
                    tooltip="Left-right consistency tolerance in 1/16 pixel "
                            "units: disagreements above it are marked "
                            "unconfident and re-filled by the filter. Only "
                            "does anything when disparity_right is wired."),
                io.Int.Input("depth_discontinuity_radius", default=0, min=0,
                    max=32, optional=True, advanced=True,
                    tooltip="Radius (px) around depth discontinuities used "
                            "when computing confidence. 0 = ceil(block_size / "
                            "2), which is what the matcher-bound filter picks."),
                io.Combo.Input("roi", options=list(cls._ROI_MODES),
                    default=cls._ROI_MODES[0], optional=True, advanced=True,
                    tooltip="'auto' filters everything except the invalid left "
                            "band implied by num_disparities/min_disparity, "
                            "which is what the matcher-bound filter does. "
                            "'full frame' reproduces cv2's own generic default "
                            "- keep it only to demonstrate the difference, it "
                            "corrupts a real map."),
            ],
            outputs=[
                NPArray.Output(display_name="disparity",
                    tooltip="HxW float32 WLS-filtered disparity in pixels."),
                NPArray.Output(display_name="confidence",
                    tooltip="HxW float32 confidence map, 0-255; low where the "
                            "two matching directions disagreed. ALL ZEROS when "
                            "disparity_right is not wired - the filter has "
                            "nothing to check against."),
                NPArray.Output(display_name="valid",
                    tooltip="uint8 0/255 mask of pixels whose filtered "
                            "disparity is at or above min_disparity."),
            ],
        )

    @classmethod
    def _as_map(cls, disparity, name):
        d = np.asarray(polytype.resolve(disparity)[0], np.float32)
        if d.ndim == 3 and d.shape[2] == 1:
            d = d[:, :, 0]
        if d.ndim != 2:
            raise ValueError(
                f"DisparityFilterWLS: '{name}' must be a single-channel HxW "
                f"map, got shape {np.asarray(disparity).shape}")
        return d

    @classmethod
    def execute(cls, disparity, guide, wls_lambda, wls_sigma_color,
                disparity_right=None, num_disparities=64, min_disparity=0,
                block_size=7, lrc_threshold=24.0, depth_discontinuity_radius=0,
                roi="auto (skip the invalid left band)") -> io.NodeOutput:
        ximgproc = _require("ximgproc", "createDisparityWLSFilterGeneric")
        disp = cls._as_map(disparity, "disparity")
        h, w = disp.shape
        gray = cv2.cvtColor(_one_bgr(guide), cv2.COLOR_BGR2GRAY)
        if gray.shape != disp.shape:
            # rescaling a disparity would rescale its VALUES too, so the guide
            # is the one that moves
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

        # the filter works on the int16 x16 maps the matchers hand out
        left16 = np.round(np.nan_to_num(disp, nan=-1.0) * 16.0).astype(np.int16)
        right16 = None
        if disparity_right is not None:
            dr = cls._as_map(disparity_right, "disparity_right")
            if dr.shape != disp.shape:
                raise ValueError(
                    f"DisparityFilterWLS: disparity_right is {dr.shape[1]}x"
                    f"{dr.shape[0]} but disparity is {w}x{h}")
            right16 = np.round(np.nan_to_num(dr, nan=-1.0) * 16.0).astype(np.int16)

        nd = max(16, int(round(num_disparities / 16)) * 16)
        md = int(min_disparity)
        bs = max(1, int(block_size) | 1)
        if roi == cls._ROI_MODES[1]:
            rect = (0, 0, w, h)
        else:
            x = max(0, min(md + nd, w - 1))
            rect = (x, 0, max(1, w + min(md, 0) - x), h)

        try:
            if right16 is not None:
                # the bound factory computes OpenCV's own ROI from a matcher's
                # search range - it never touches the matcher again, so a
                # throwaway one carrying the declared range is enough, whatever
                # produced the maps
                dummy = cv2.StereoSGBM_create(minDisparity=md, numDisparities=nd,
                                              blockSize=bs)
                wls = ximgproc.createDisparityWLSFilter(dummy)
            else:
                wls = ximgproc.createDisparityWLSFilterGeneric(False)
            wls.setLambda(float(wls_lambda))
            wls.setSigmaColor(float(wls_sigma_color))
            wls.setLRCthresh(int(round(lrc_threshold)))
            # ceil(bs / 2), the value the bound filter derives from an SGBM
            # matcher - the generic one would otherwise sit on its own default
            # of 5 and answer thousands of pixels differently
            wls.setDepthDiscontinuityRadius(int(depth_discontinuity_radius)
                                            if int(depth_discontinuity_radius) > 0
                                            else (bs + 1) // 2)
            if right16 is not None and roi == cls._ROI_MODES[0]:
                filtered = wls.filter(left16, gray, None, right16)
            else:
                filtered = wls.filter(left16, gray, None, right16, rect)
            raw_conf = wls.getConfidenceMap()
        except cv2.error as e:
            logging.warning("DisparityFilterWLS: cv2 error, passing the input "
                            "disparity through: %s", e)
            return io.NodeOutput(disp.copy(), np.zeros((h, w), np.float32),
                                 (disp >= md).astype(np.uint8) * 255)

        out = np.asarray(filtered, np.float32) / 16.0
        # no right map -> no left-right check -> getConfidenceMap() is None
        conf = (np.zeros((h, w), np.float32) if raw_conf is None
                else np.asarray(raw_conf, np.float32))
        return io.NodeOutput(out, conf, (out >= md).astype(np.uint8) * 255)


class DisparityInterpolate(io.ComfyNode):
    """ximgproc's sparse-match interpolators used as a disparity hole filler.

    Both the alternatives already in the pack fill holes with a locally
    CONSTANT model and get one half of a driving scene wrong:

    * `cv2.inpaint` (Telea) diffuses from the hole rim and is image-blind, so
      the ground comes out fine and every object silhouette bleeds - measured
      on the shipped StereoGeo-CARLA pair, only 15% of its strong disparity
      gradients land within 2 px of an image edge.
    * `DisparityWLSFilter` sticks to the left image's edges (91% by the same
      measure) but its smoothness prior is fronto-parallel, so the textureless
      slanted road is bent into a ridge: a plane fit through the road band goes
      from 0.06 m RMS / 98% inliers (raw SGBM) to 0.36 m / 33%. It corrupts
      pixels SGBM already had right, not just the ones it fills (0.34 m RMS
      restricted to raw-valid pixels), and lambda only trades that against
      coverage (lambda=100 -> 0.09 m but 70% valid).

    `EdgeAwareInterpolator` (EpicFlow) and `RICInterpolator` fit a local AFFINE
    model per neighbourhood / superpixel instead, which is exactly a slanted
    plane, so the road survives AND the silhouettes do: RIC scores 0.06 m RMS /
    97% inliers with 85-90% edge alignment at ~100% coverage, in ~0.2 s for
    640x480.

    They are optical-flow interpolators, so the disparity is fed in as the
    horizontal component of a sparse match set ((x, y) -> (x - d, y)) and read
    back out of channel 0 of the dense flow, negated.

    Class API (a `create*` factory + setters + one call), hence this module
    rather than a generated wrapper; the object is built, used and dropped
    inside one execute() like every other node here.
    """

    _METHODS = {
        "RIC (superpixel plane fit)": "ric",
        "EPIC (edge-aware, faster)": "epic",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_DisparityInterpolate",
            display_name="CV Disparity Interpolate (Edge-Aware)",
            category=CATEGORY,
            description="Turns a HOLEY disparity map into a dense one by "
                        "interpolating its trustworthy pixels with a local "
                        "PLANE model (cv2.ximgproc RICInterpolator / "
                        "EdgeAwareInterpolator), guided by the rectified left "
                        "image. Use it instead of 'cv2.inpaint' (image-blind: "
                        "smears disparity across object silhouettes) and "
                        "instead of relying on the WLS filter to fill: WLS "
                        "assumes locally CONSTANT disparity, so it bends a "
                        "textureless slanted surface like a road into a ridge "
                        "- measured on the shipped driving pair, a plane fit "
                        "through the road band degrades from 0.06 m RMS to "
                        "0.36 m, while this node keeps 0.06 m at ~100% "
                        "coverage. Feed it the RAW matcher output (plus the "
                        "WLS 'confidence' map if you have one, to drop "
                        "unreliable seeds). The pair must be RECTIFIED. "
                        "Failure-tolerant: with too few seeds it returns the "
                        "input disparity unchanged and found=false.",
            inputs=[
                polytype.poly_input("left", types=polytype.IMAGE_ONLY,
                    tooltip="Rectified LEFT frame - the guide whose edges and "
                            "superpixels the interpolation follows."),
                polytype.poly_input("right", types=polytype.IMAGE_ONLY,
                    tooltip="Rectified RIGHT frame. Same size as 'left'; the "
                            "interpolators take both views of the match set."),
                NPArray.Input("disparity",
                    tooltip="HxW float disparity in PIXELS with holes "
                            "(negative / zero where the matcher failed), e.g. "
                            "'CV Stereo Disparity (SGBM)' or the "
                            "'disparity_raw' output of the WLS node."),
                io.Combo.Input("method", options=list(cls._METHODS),
                    default="RIC (superpixel plane fit)",
                    tooltip="RIC fits a plane per SLIC superpixel and keeps "
                            "silhouettes best (~0.2 s at 640x480); EPIC "
                            "(EpicFlow) is ~2x faster and slightly smoother "
                            "on flat ground but blurrier at depth edges. RIC "
                            "randomizes its model fitting, so its output "
                            "varies slightly run to run; EPIC is "
                            "deterministic."),
                io.Float.Input("min_disparity", default=0.05, min=-256.0,
                    max=256.0, step=0.05,
                    tooltip="Disparity above which an input pixel counts as a "
                            "seed. The SGBM nodes mark invalid pixels below "
                            "min_disparity, so the default keeps everything "
                            "positive."),
                NPArray.Input("confidence", optional=True,
                    tooltip="Optional HxW confidence map (0-255), e.g. the "
                            "'confidence' output of 'CV Stereo Disparity (WLS "
                            "filtered)'. Seeds below the threshold are "
                            "dropped, which is how occlusions and failed "
                            "left-right checks stay out of the fit."),
                io.Float.Input("confidence_threshold", default=128.0, min=0.0,
                    max=255.0, step=1.0, optional=True, advanced=True,
                    tooltip="Minimum confidence for a seed (ignored when no "
                            "confidence map is connected)."),
                io.Int.Input("k", default=32, min=4, max=256,
                    optional=True, advanced=True,
                    tooltip="Neighbouring seeds used to fit each local model. "
                            "Larger = smoother and slower."),
                io.Int.Input("superpixel_size", default=15, min=4, max=64,
                    optional=True, advanced=True,
                    tooltip="RIC only: average SLIC superpixel side in pixels. "
                            "Smaller follows finer structure at more cost."),
                io.Combo.Input("post_smoothing",
                    options=["global smoother (recommended)", "none"],
                    default="global smoother (recommended)",
                    optional=True, advanced=True,
                    tooltip="Run the edge-aware global smoother over the "
                            "interpolated result. 'none' leaves the raw "
                            "per-superpixel planes visible as blocky steps "
                            "(measured: edge alignment 89% -> 53%)."),
                io.Int.Input("max_seeds", default=30000, min=64, max=32000,
                    optional=True, advanced=True,
                    tooltip="Seeds are subsampled to at most this many. "
                            "cv2 ASSERTS on 32767 or more (match_num < "
                            "SHRT_MAX), and a full 640x480 map has ~180k "
                            "valid pixels, so this cap is not optional."),
            ],
            outputs=[
                NPArray.Output(display_name="disparity",
                    tooltip="HxW float32 dense disparity in pixels."),
                NPArray.Output(display_name="valid",
                    tooltip="uint8 0/255 mask of pixels with a disparity above "
                            "min_disparity (near-complete after a successful "
                            "interpolation)."),
                NPArray.Output(display_name="seeds",
                    tooltip="uint8 0/255 mask of the input pixels actually "
                            "used as seeds - preview it to see what the fit "
                            "was given."),
                io.Boolean.Output(display_name="found",
                    tooltip="False when there were too few seeds to "
                            "interpolate; the input disparity is passed "
                            "through unchanged."),
            ],
        )

    @classmethod
    def execute(cls, left, right, disparity, method, min_disparity=0.05,
                confidence=None, confidence_threshold=128.0, k=32,
                superpixel_size=15, post_smoothing="global smoother (recommended)",
                max_seeds=30000) -> io.NodeOutput:
        ximgproc = _require("ximgproc", "createRICInterpolator")
        lb, rb = _one_bgr(left), _one_bgr(right)
        if rb.shape[:2] != lb.shape[:2]:
            rb = cv2.resize(rb, (lb.shape[1], lb.shape[0]),
                            interpolation=cv2.INTER_AREA)

        disp = np.asarray(polytype.resolve(disparity)[0], np.float32)
        if disp.ndim == 3 and disp.shape[2] == 1:
            disp = disp[:, :, 0]
        if disp.ndim != 2:
            raise ValueError(
                "DisparityInterpolate: 'disparity' must be a single-channel "
                f"HxW map, got shape {np.asarray(disparity).shape}")
        h, w = lb.shape[:2]
        if disp.shape != (h, w):
            # rescaling a disparity would rescale its VALUES too, so refuse
            raise ValueError(
                f"DisparityInterpolate: disparity is {disp.shape[1]}x"
                f"{disp.shape[0]} but the images are {w}x{h} - interpolate the "
                "disparity of THESE rectified frames")

        thr = float(min_disparity)
        seeds = np.isfinite(disp) & (disp > thr)
        if confidence is not None:
            conf = np.asarray(polytype.resolve(confidence)[0], np.float32)
            if conf.ndim == 3 and conf.shape[2] == 1:
                conf = conf[:, :, 0]
            if conf.shape[:2] != (h, w):
                raise ValueError(
                    "DisparityInterpolate: confidence map is "
                    f"{conf.shape[1]}x{conf.shape[0]}, expected {w}x{h}")
            seeds &= conf >= float(confidence_threshold)
        ys, xs = np.nonzero(seeds)
        # a match must land inside the right image, or cv2 drops the whole set
        keep = (xs - disp[ys, xs]) >= 0
        ys, xs = ys[keep], xs[keep]

        seed_mask = np.zeros((h, w), np.uint8)
        if len(xs) < 16:
            logging.warning("DisparityInterpolate: only %d seeds, passing the "
                            "input disparity through", len(xs))
            return io.NodeOutput(disp.copy(),
                                 (disp > thr).astype(np.uint8) * 255,
                                 seed_mask, False)

        limit = min(int(max_seeds), 32000)  # cv2 asserts match_num < SHRT_MAX
        if len(xs) > limit:
            pick = np.linspace(0, len(xs) - 1, limit).astype(np.int64)
            ys, xs = ys[pick], xs[pick]
        seed_mask[ys, xs] = 255

        src = np.stack([xs, ys], 1).astype(np.float32)
        dst = np.stack([xs - disp[ys, xs], ys], 1).astype(np.float32)
        smooth = post_smoothing != "none"

        def _interpolate(kind, smoothing):
            if kind == "ric":
                it = ximgproc.createRICInterpolator()
                it.setK(int(k))
                it.setSuperpixelSize(int(superpixel_size))
                it.setUseGlobalSmootherFilter(smoothing)
                it.setRefineModels(True)
                # variational refinement is a 2D optical-flow polish; on a
                # disparity field it would introduce a vertical component
                it.setUseVariationalRefinement(False)
            else:
                it = ximgproc.createEdgeAwareInterpolator()
                it.setK(int(k))
                it.setUsePostProcessing(smoothing)
            flow = it.interpolate(lb, src, rb, dst)
            return np.clip(-np.asarray(flow, np.float32)[:, :, 0], 0.0, None)

        # These interpolators intermittently hand back an all-but-empty field
        # from a perfectly good match set - not reproducible on a given input,
        # and it comes and goes with what else the process has been doing. The
        # trailing global smoother is the fragile part (turning it off is what
        # recovers, and it is also why coverage is bimodal: ~0.995 when the
        # smoother runs, ~0.86 when it does not). An interpolation SPREADS its
        # seeds, so it can never legitimately return fewer valid pixels than it
        # was given: that is a post-condition this node can check, and it beats
        # emitting an empty point cloud.
        kind = cls._METHODS[method]
        other = "epic" if kind == "ric" else "ric"
        plan = [(kind, smooth), (kind, smooth), (kind, False), (other, smooth)]
        dense, ok = None, False
        for attempt, (this_kind, this_smooth) in enumerate(plan):
            try:
                dense = _interpolate(this_kind, this_smooth)
            except cv2.error as e:
                logging.warning("DisparityInterpolate: cv2 error, passing the "
                                "input disparity through: %s", e)
                return io.NodeOutput(disp.copy(),
                                     (disp > thr).astype(np.uint8) * 255,
                                     seed_mask, False)
            ok = int((dense > thr).sum()) >= len(xs)
            if ok:
                if attempt:
                    logging.warning("DisparityInterpolate: recovered on "
                                    "attempt %d (%s, smoothing=%s)",
                                    attempt + 1, this_kind, this_smooth)
                break
            logging.warning("DisparityInterpolate: interpolating %d seeds with "
                            "%s returned only %d valid pixels (attempt %d)",
                            len(xs), this_kind, int((dense > thr).sum()),
                            attempt + 1)
        if not ok:
            return io.NodeOutput(disp.copy(),
                                 (disp > thr).astype(np.uint8) * 255,
                                 seed_mask, False)
        return io.NodeOutput(dense, (dense > thr).astype(np.uint8) * 255,
                             seed_mask, True)


class ICPRegister(io.ComfyNode):
    """cv2.ppf_match_3d.ICP - iterative closest point between two clouds.

    Failure-tolerant: a raise, a non-finite pose or an empty cloud
    returns the identity with found=false instead of halting the workflow.

    The returned `residual` is NOT a quality signal. Measured on the example
    two-pair driving scene it reported 5e-5 while the alignment it produced was
    WORSE than its RANSAC initialization (nearest-neighbour median 0.34 m ->
    0.69 m, plus 7.6 degrees of spurious rotation). Judge the result with
    'CV Point Cloud Nearest Distance' instead.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_ICPRegister",
            display_name="CV ICP Register (Point Clouds)",
            category=CATEGORY,
            description="Refines the alignment of two 3D point clouds with "
                        "iterative closest point (cv2.ppf_match_3d.ICP) and "
                        "returns the 4x4 pose that maps 'model' onto 'scene' - "
                        "feed it to 'CV Transform Points 3D'. ICP only POLISHES: "
                        "it needs a good initial alignment (register the clouds "
                        "with cv2.estimateAffine3D first) and it converges to a "
                        "wrong answer from a bad start. IMPORTANT: the "
                        "'residual' output is an internal number, not a quality "
                        "measure - on the shipped stereo example it reported "
                        "5e-5 for a pose that made the alignment worse. Always "
                        "verify with 'CV Point Cloud Nearest Distance' (median "
                        "distance must go DOWN) and branch on it with a "
                        "'Basic data handling: IfElse'. Downsample first: ICP "
                        "cost grows with the cloud size.",
            inputs=[
                NPArray.Input("model",
                    tooltip="Nx3 (or Nx6 with normals) cloud to be moved."),
                NPArray.Input("scene",
                    tooltip="Mx3 (or Mx6) reference cloud to align onto."),
                io.Int.Input("iterations", default=100, min=1, max=2000,
                    tooltip="Max ICP iterations per pyramid level."),
                io.Float.Input("tolerance", default=0.05, min=0.0, max=1.0,
                    step=0.005,
                    tooltip="Relative improvement below which iteration stops."),
                io.Float.Input("rejection_scale", default=2.5, min=0.0, max=10.0,
                    step=0.1,
                    tooltip="Outlier rejection: correspondences farther than "
                            "this many standard deviations are dropped each "
                            "iteration. Lower = stricter."),
                io.Int.Input("num_levels", default=6, min=1, max=10,
                    tooltip="Coarse-to-fine pyramid levels; more levels tolerate "
                            "a rougher initial alignment - up to the point where "
                            "the coarsest level has nothing left to fit, and then "
                            "ICP DIVERGES rather than degrading. A PARTIAL scene "
                            "(a model matched against the one face a depth camera "
                            "can see) hits that first: measured on a 360-point "
                            "model against a 176-point view, 4 levels converge "
                            "and 5 diverge; on a 1363-point model against a "
                            "1335-point view, 6 converge and 8 diverge. Lower it "
                            "when 'found' comes back false with a 1e10 residual."),
            ],
            outputs=[
                NPArray.Output(display_name="pose",
                    tooltip="4x4 rigid transform mapping model -> scene "
                            "(identity when found=false)."),
                io.Float.Output(display_name="residual",
                    tooltip="ICP's own residual. NOT a quality measure - see the "
                            "node description; use 'CV Point Cloud Nearest "
                            "Distance' to judge the alignment. The ONE thing it "
                            "does say: 9999999999 is cv2's divergence sentinel, "
                            "and this node turns that into found=false."),
                io.Boolean.Output(display_name="found",
                    tooltip="False when ICP could not run (empty cloud, cv2 "
                            "error, non-finite pose) or DIVERGED. cv2 signals "
                            "divergence with a 1e10 residual while still "
                            "returning 0 from registerModelToScene, so the "
                            "return code alone would report success on a pose "
                            "hundreds of units out - too many pyramid levels for "
                            "the point count is the usual cause. Gate the "
                            "transform on this with a 'Basic data handling: "
                            "IfElse'."),
            ],
        )

    @classmethod
    def execute(cls, model, scene, iterations, tolerance, rejection_scale,
                num_levels) -> io.NodeOutput:
        eye = np.eye(4, dtype=np.float64)
        _require("ppf_match_3d", "ICP")

        def _cloud(arr):
            a = np.asarray(arr, np.float32)
            if a.size == 0:
                return np.zeros((0, 3), np.float32)
            a = a.reshape(-1, a.shape[-1] if a.ndim > 1 else 3)
            return np.ascontiguousarray(a[:, :6] if a.shape[1] >= 6 else a[:, :3])

        m, s = _cloud(model), _cloud(scene)
        if len(m) < 3 or len(s) < 3:
            return io.NodeOutput(eye, 0.0, False)

        icp = cv2.ppf_match_3d_ICP(int(iterations), float(tolerance),
                                   float(rejection_scale), int(num_levels))
        try:
            retval, residual, pose = icp.registerModelToScene(m, s)
        except cv2.error as e:
            logging.warning("ICPRegister: cv2 error, returning identity: %s", e)
            return io.NodeOutput(eye, 0.0, False)
        pose = np.asarray(pose, np.float64).reshape(4, 4)
        # registerModelToScene returns 0 whether it converged or not: on
        # divergence it hands back a garbage pose and residual 9999999999
        # (measured with num_levels=8 on a 1363-point model, which lands
        # 117 degrees and 37 units out while retval stays 0)
        diverged = float(residual) >= 1e9
        if diverged:
            logging.warning("ICPRegister: diverged (residual %g, num_levels %d) "
                            "- returning identity", float(residual),
                            int(num_levels))
        if retval != 0 or diverged or not np.all(np.isfinite(pose)):
            return io.NodeOutput(eye, float(residual), False)
        return io.NodeOutput(pose, float(residual), True)


class PPFPoseEstimation(io.ComfyNode):
    """cv2.ppf_match_3d.PPF3DDetector - GLOBAL pose of a model in a scene.

    The complement of ICPRegister: point-pair features vote for a pose from
    scratch (no initial alignment) but the answer is COARSE - measured on the
    workflow-81 clouds the winning pose is 13-15 degrees and ~20 units off the
    truth, which ICP then polishes to ~0.05 degrees. Ship both.

    Traps this node covers, all measured on cv2 5.0.0:
    - a MODEL without normals does not raise: `trainModel` on an Nx3 cloud took
      341 s (against 3.5 s for the same cloud Nx6) and a scene without normals
      answers 85 degrees out (all-NaN on another cloud). The node computes
      the normals itself rather than
      hand cv2 an Nx3 array.
    - float64 clouds raise, an empty cloud and a PLANAR scene raise "Unknown
      C++ exception" - caught, found=false.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_PPFPoseEstimation",
            display_name="CV PPF Pose Estimation",
            category=CATEGORY,
            description="Finds a known 3D model inside a larger 3D scene with "
                        "no initial guess (cv2.ppf_match_3d.PPF3DDetector): it "
                        "hashes oriented point PAIRS of the model, then lets "
                        "the scene's pairs vote for the 4x4 pose that maps "
                        "model -> scene. Unlike 'CV ICP Register' it does not "
                        "need the clouds to be roughly aligned already - but "
                        "its answer is COARSE (on the shipped example ~14 "
                        "degrees out), so the standard pipeline is PPF then "
                        "ICP: 'CV Transform Points 3D' the model by this pose, "
                        "'CV Point Cloud Normals' it again, 'CV ICP Register' "
                        "onto the scene, and compose the two with 'CV Matrix "
                        "Multiply'. Cost grows steeply as the sampling steps "
                        "shrink (training is O(sampled^2)); downsample big "
                        "clouds first. Failure tolerant: a degenerate cloud (a "
                        "plane, too few points) returns the identity with "
                        "found=false.",
            inputs=[
                NPArray.Input("model",
                    tooltip="The object to look for, Nx6 (x,y,z,nx,ny,nz) from "
                            "'CV Point Cloud Normals'. An Nx3 cloud is accepted "
                            "and its normals are computed here - never hand cv2 "
                            "one directly, it silently runs 31x-100x slower "
                            "and answers wrong."),
                NPArray.Input("scene",
                    tooltip="The cloud to search in, Nx6 (or Nx3, as above). "
                            "May contain clutter and other objects - that is "
                            "what PPF is for."),
                io.Float.Input("relative_sampling_step", default=0.05,
                    min=0.005, max=0.5, step=0.005,
                    tooltip="Model downsampling, as a fraction of the model's "
                            "diameter: 0.05 keeps points ~5% of the diameter "
                            "apart. THE cost knob - halving it roughly "
                            "quadruples training time (0.025 -> 26 s where 0.05 "
                            "-> 3.5 s on a 2400-point model)."),
                io.Float.Input("relative_distance_step", default=0.05,
                    min=0.005, max=0.5, step=0.005,
                    tooltip="Quantization of the point-pair distance in the "
                            "hash table, as a fraction of the diameter. Larger "
                            "tolerates more noise and blurs the vote."),
                io.Int.Input("num_angles", default=30, min=6, max=120,
                    tooltip="Angle bins the pair features are quantized into. "
                            "More = finer rotation resolution, bigger table."),
                io.Float.Input("relative_scene_sample_step", default=0.2,
                    min=0.01, max=1.0, step=0.01,
                    tooltip="Fraction of scene points used as reference points "
                            "(0.2 = every 5th). Lower = faster, less thorough."),
                io.Float.Input("relative_scene_distance", default=0.05,
                    min=0.005, max=0.5, step=0.005,
                    tooltip="Scene downsampling distance, as a fraction of the "
                            "model diameter (the scene equivalent of "
                            "relative_sampling_step)."),
                io.Int.Input("max_poses", default=5, min=1, max=100,
                    tooltip="How many of the highest-voted pose clusters to "
                            "return in 'poses'. 'pose' is always the first."),
                io.Int.Input("normal_neighbors", default=12, min=3, max=200,
                    optional=True, advanced=True,
                    tooltip="Neighbours per plane fit, used ONLY for an input "
                            "that arrives without normals. Prefer wiring "
                            "'CV Point Cloud Normals' explicitly."),
            ],
            outputs=[
                NPArray.Output(display_name="pose",
                    tooltip="4x4 rigid transform mapping model -> scene, the "
                            "highest-voted cluster (identity when "
                            "found=false). Feed 'CV Transform Points 3D'."),
                NPArray.Output(display_name="poses",
                    tooltip="Kx4x4 stack of the top 'max_poses' candidates, "
                            "best first. PPF often gets the right pose at rank "
                            "2-3, so refine several with ICP and keep the one "
                            "with the lowest 'CV Point Cloud Nearest Distance'."),
                NPArray.Output(display_name="votes",
                    tooltip="Kx1 float32 vote count per candidate - a RELATIVE "
                            "confidence only (it scales with the cloud size and "
                            "the sampling steps), not a quality measure."),
                io.Int.Output(display_name="pose_count",
                    tooltip="Number of candidates returned (<= max_poses)."),
                io.Boolean.Output(display_name="found",
                    tooltip="False when the match could not run (empty or "
                            "degenerate cloud, cv2 error). Gate the transform "
                            "on it with a 'Basic data handling: IfElse'."),
            ],
        )

    @classmethod
    def execute(cls, model, scene, relative_sampling_step,
                relative_distance_step, num_angles, relative_scene_sample_step,
                relative_scene_distance, max_poses,
                normal_neighbors=12) -> io.NodeOutput:
        eye = np.eye(4, dtype=np.float64)
        empty = (eye, np.zeros((0, 4, 4), np.float64),
                 np.zeros((0, 1), np.float32), 0, False)
        ppf = _require("ppf_match_3d", "PPF3DDetector")

        def _cloud(arr):
            """Nx6 float32 with normals - cv2 needs float32 and does NOT check
            that the normals are there."""
            a = np.asarray(arr, np.float32) if arr is not None else None
            if a is None or a.size == 0:
                return np.zeros((0, 6), np.float32)
            a = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(-1, 3)
            if a.shape[1] >= 6:
                return np.ascontiguousarray(a[:, :6])
            a = np.ascontiguousarray(a[:, :3])
            if len(a) < 3:
                return np.zeros((0, 6), np.float32)
            retval, withn = ppf.computeNormalsPC3d(
                a, int(normal_neighbors), False, np.zeros((1, 3), np.float32))
            return np.ascontiguousarray(withn, np.float32).reshape(-1, 6)

        try:
            m, s = _cloud(model), _cloud(scene)
        except cv2.error as e:
            logging.warning("PPFPoseEstimation: normals: %s",
                            str(e).splitlines()[-1])
            return io.NodeOutput(*empty)
        if len(m) < 3 or len(s) < 3:
            return io.NodeOutput(*empty)

        detector = ppf.PPF3DDetector(float(relative_sampling_step),
                                     float(relative_distance_step),
                                     float(num_angles))
        try:
            detector.trainModel(m)
            results = detector.match(s, float(relative_scene_sample_step),
                                     float(relative_scene_distance))
        except cv2.error as e:  # planar/degenerate clouds throw
            logging.warning("PPFPoseEstimation: %s", str(e).splitlines()[-1])
            return io.NodeOutput(*empty)
        if not results:
            return io.NodeOutput(*empty)

        poses, votes = [], []
        for r in results[:int(max_poses)]:
            p = np.asarray(r.pose, np.float64).reshape(4, 4)
            if not np.all(np.isfinite(p)):  # an Nx3 scene reaches cv2 as NaN
                continue
            poses.append(p)
            votes.append(float(r.numVotes))
        if not poses:
            return io.NodeOutput(*empty)
        return io.NodeOutput(poses[0], np.stack(poses),
                             np.float32(votes).reshape(-1, 1),
                             len(poses), True)


# ---------------------------------------------------------------------------
# Hierarchical Feature Selection segmentation (cv2.hfs)
# ---------------------------------------------------------------------------
#
# Another class API: `cv2.hfs.HfsSegment_create` builds a stateful object whose
# create() already carries the params, so the whole configure+run+drop happens
# inside one execute(). The CPU entry point (`performSegmentCpu`) is what this
# build can run - OpenCV labels it "reference only" (it resamples with
# cv2.resize + INTER_LINEAR instead of the GPU's bespoke kernels) but it is THE
# portable path on a CPU-only install. The 7 params below are the class's own
# defaults (read back from a freshly-created object), so an untouched node
# behaves exactly like stock HfsSegment.


class HfsSegment(io.ComfyNode):
    """cv2.hfs hierarchical feature selection: merges SLIC superpixels by
    texton + colour similarity into regions.

    HFS is the (undocumented) segmentation core that used to back several
    "one-click matting" demos. It first clusters the image into superpixels
    (SLIC), then merges them greedily along a gradient-descending criterion in
    two stages. The pipeline is fully input-driven: create() fixes the height
    and width, so the node builds a fresh object per call with the input's own
    H/W and never carries state across executions.
    """

    @classmethod
    def define_schema(cls):
        template = polytype.match_template(types=polytype.IMAGE_ONLY)
        return io.Schema(
            node_id="CV_HfsSegment",
            display_name="CV HFS Segmentation (hierarchical feature selection)",
            category=CATEGORY,
            description="Segments an image into regions via hierarchical "
                        "feature selection (cv2.hfs): SLIC superpixels are "
                        "merged by texton + colour similarity over two "
                        "gradient-threshold stages. Outputs a per-pixel region "
                        "index map (uint16, one region label per pixel), a "
                        "region-coloured version of the image (each region is "
                        "repainted with its average colour), and the region "
                        "count. Works without a GPU (CPU path). The merge "
                        "thresholds trade region purity against region count: "
                        "raise them to merge more aggressively, and lower "
                        "'spatial weight' to let colour/texture dominate over "
                        "proximity.",
            inputs=[
                polytype.match_input("image", template,
                    tooltip="Full-colour image to segment (3 channels). The "
                            "region index map has one label per pixel and the "
                            "same height/width."),
                io.Float.Input("egb_threshold_i", default=0.08, min=0.0,
                    max=1.0, step=0.01, optional=True, advanced=True,
                    tooltip="Stage-I merge threshold: gradient below this and "
                            "two regions merge. Higher = fewer, larger regions "
                            "(and less detail kept)."),
                io.Int.Input("min_region_size_i", default=100, min=1,
                    max=100000, optional=True, advanced=True,
                    tooltip="Stage-I minimum region size in pixels; regions "
                            "smaller than this get absorbed into a neighbour."),
                io.Float.Input("egb_threshold_ii", default=0.28, min=0.0,
                    max=1.0, step=0.01, optional=True, advanced=True,
                    tooltip="Stage-II threshold, applied to the coarser "
                            "regions. Usually comfortably larger than "
                            "egb_threshold_i, hence the second coarser pass."),
                io.Int.Input("min_region_size_ii", default=200, min=1,
                    max=100000, optional=True, advanced=True,
                    tooltip="Stage-II minimum region size in pixels; the final "
                            "merge removes regions smaller than this."),
                io.Float.Input("spatial_weight", default=0.6, min=0.0,
                    max=1.0, step=0.05, optional=True, advanced=True,
                    tooltip="Weight of spatial distance inside the SLIC "
                            "clustering (0 = colour/texture only, 1 = pure "
                            "proximity). The default already leans on colour."),
                io.Int.Input("superpixel_size", default=8, min=2, max=256,
                    optional=True, advanced=True,
                    tooltip="SLIC superpixel size in pixels. Smaller = more "
                            "seeds = finer detail captured; larger runs "
                            "faster and over-merges small features."),
                io.Int.Input("slic_iterations", default=5, min=1, max=50,
                    optional=True, advanced=True,
                    tooltip="SLIC clustering iterations. 5 is plenty for "
                            "convergence; raise to 10 on very textured "
                            "content."),
            ],
            outputs=[
                polytype.match_output(template, "segmented",
                    tooltip="The input with every region repainted by its "
                            "average colour - a quick visual of the region "
                            "map."),
                NPArray.Output(display_name="labels",
                    tooltip="uint16 per-pixel region index (0..num_regions-1), "
                            "the same height/width as the input. Feed this "
                            "into mask/crop/Draw Nodes."),
                io.Int.Output(display_name="num_regions",
                    tooltip="Number of distinct regions in the label map."),
            ],
        )

    @classmethod
    def execute(cls, image, egb_threshold_i=0.08, min_region_size_i=100,
                egb_threshold_ii=0.28, min_region_size_ii=200,
                spatial_weight=0.6, superpixel_size=8, slic_iterations=5,
                ) -> io.NodeOutput:
        hfs = _require("hfs", "HfsSegment_create")
        bgr = _one_bgr(image)
        h, w = bgr.shape[:2]
        if bgr.size == 0:
            raise ValueError("HFS needs a non-empty image.")
        seg = hfs.HfsSegment_create(int(h), int(w), float(egb_threshold_i),
                                    int(min_region_size_i),
                                    float(egb_threshold_ii),
                                    int(min_region_size_ii),
                                    float(spatial_weight),
                                    int(superpixel_size),
                                    int(slic_iterations))
        try:
            labels = seg.performSegmentCpu(bgr, False)
        except cv2.error as e:
            logging.warning("HfsSegment: cv2 error, returning empty result: %s", e)
            labels = np.zeros((h, w), np.uint16)
        labels = np.ascontiguousarray(labels.reshape(h, w))
        # labels can skip indices (the merger frees ids), so count DISTINCT
        # regions instead of trusting max()+1
        n_ids = int(labels.max()) + 1 if labels.size else 0

        # Repaint each region with its average colour (the exact semantics of
        # performSegment's ifDraw=True) without running the segmentation twice
        # and without double-converting pixel space.
        flat = bgr.reshape(-1, 3).astype(np.float32)
        lab = labels.ravel()
        counts = np.bincount(lab, minlength=n_ids)
        sums = np.zeros((n_ids, 3), np.float64)
        for c in range(3):
            sums[:, c] = np.bincount(lab, weights=flat[:, c], minlength=n_ids)
        means = np.where(counts[:, None] > 0,
                         sums / np.maximum(counts[:, None], 1), 0.0)
        colored = means[lab].reshape(h, w, 3).astype(np.uint8)
        num_regions = int((counts > 0).sum())
        _, tag = polytype.resolve(image)
        return io.NodeOutput(polytype.restore(colored, tag), labels, num_regions)


# ---------------------------------------------------------------------------
# Optical flow (cv2.optflow)
# ---------------------------------------------------------------------------
#
# registry_contrib.py holds exactly TWO optflow entries - calcOpticalFlowSF and
# calcOpticalFlowSparseToDense - because generator.classify() maps any type it
# cannot turn into a widget to None and drops the overload. That skips the
# module's own algorithms twice over:
#
#   * every create*/factory returns a class Ptr (DeepFlow, PCAFlow, DualTVL1,
#     Dense/SparseRLOF), and
#   * calcOpticalFlowDenseRLOF / calcOpticalFlowSparseRLOF ARE free functions,
#     but both take an `RLOFOpticalFlowParameter` object, so RLOF is gated by a
#     class even though its entry point is not one.
#
# class_gated_core.md only ever ticked DISOpticalFlow / VariationalRefinement
# (both core, both in features.py) and dismissed the SparsePyrLK / Farneback
# CLASSES as duplicates of exposed functions - it never covered cv2.optflow.
#
# Measured on this build (OpenCV 5.0.0), three of the factories are NOT worth a
# node and are deliberately absent: createOptFlow_SimpleFlow().calc returns an
# ALL-NaN field, createOptFlow_SparseToDense().calc raises "Unknown C++
# exception" (its free function works and is already generated), and
# createOptFlow_Farneback duplicates 'CV Optical Flow (Farneback)'. The GPC
# (Global Patch Collider) flow has no Python path at all: GPCTree,
# GPCTrainingSamples and GPCDetails are bound but GPCForest - the only class
# with a flow method - is not.

_FLOW_ALGOS = {
    "DeepFlow (large motion)": "deepflow",
    "PCAFlow (fast, learned prior)": "pcaflow",
}

_RLOF_INTERP = {
    "EPIC (edge-preserving, default)": "INTERP_EPIC",
    "RIC (superpixel plane fit)": "INTERP_RIC",
    "GEO (geodesic, fastest)": "INTERP_GEO",
}

_RLOF_SOLVER = {
    "bilinear (accurate, default)": "ST_BILINEAR",
    "standard (faster)": "ST_STANDART",
}

_RLOF_SUPPORT = {
    "cross-based segmentation (default)": "SR_CROSS",
    "fixed window": "SR_FIXED",
}

_RIC_SLIC = {
    "SLIC": "SLIC",
    "SLICO": "SLICO",
    "MSLIC": "MSLIC",
}


def _optflow_const(name):
    """A cv2.optflow constant. They are NOT on the cv2 root, so enums.EnumGroup
    (which resolves with hasattr(cv2, ...)) cannot carry them - hence the plain
    label->name maps above, the same shape as DisparityInterpolate._METHODS."""
    return int(getattr(_require("optflow"), name))


def _flow_frame_pair(frame_a, frame_b):
    """Two grayscale uint8 frames of identical size, for the dense flow nodes."""
    frame_a, _ = polytype.resolve(frame_a)
    frame_b, _ = polytype.resolve(frame_b)
    prev, nxt = _to_gray_u8(frame_a), _to_gray_u8(frame_b)
    if nxt.shape != prev.shape:
        nxt = cv2.resize(nxt, (prev.shape[1], prev.shape[0]),
                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(prev), np.ascontiguousarray(nxt)


def _rlof_frame_pair(frame_a, frame_b):
    """Two 8-bit 3-channel BGR frames - RLOF REQUIRES colour (CV_8UC3): its
    cross-based support region segments on colour, and cv2 asserts on a
    single-channel input."""
    a, b = _one_bgr(frame_a), _one_bgr(frame_b)
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def _init_flow_for(init_flow, target_hw):
    """An optional HxWx2 warm-start field, sized to the frame - or None.

    calc() WRITES into the flow argument, so this always returns a fresh copy
    and never the upstream node's cached array (same rule as lowlevel.py).
    """
    if init_flow is None:
        return None
    init = np.asarray(init_flow, np.float32)
    if init.size == 0:
        return None
    if init.shape[:2] != tuple(target_hw):
        init = cv2.resize(init, (target_hw[1], target_hw[0]))
    return np.ascontiguousarray(init.copy())


def _rlof_params(**kwargs):
    """Build the RLOFOpticalFlowParameter both RLOF entry points need.

    Takes the advanced widgets by NAME and rejects anything unexpected: both
    nodes collect these through **rlof, and a silently-swallowed key would mean
    a renamed widget quietly reverting to its default.

    The object is created, filled and handed straight to cv2 inside one
    execute() - it never becomes graph state (see this module's docstring).
    """
    unknown = set(kwargs) - set(_RLOF_ADVANCED_DEFAULTS)
    if unknown:
        raise ValueError(f"unexpected RLOF parameter(s): {sorted(unknown)}")
    v = {**_RLOF_ADVANCED_DEFAULTS, **kwargs}

    optflow = _require("optflow", "RLOFOpticalFlowParameter_create")
    p = optflow.RLOFOpticalFlowParameter_create()
    p.setSolverType(_optflow_const(_RLOF_SOLVER[v["solver_type"]]))
    p.setSupportRegionType(_optflow_const(_RLOF_SUPPORT[v["support_region"]]))
    p.setSmallWinSize(int(v["small_win_size"]))
    p.setLargeWinSize(int(v["large_win_size"]))
    p.setMaxLevel(int(v["max_level"]))
    p.setMaxIteration(int(v["max_iterations"]))
    # The default parameter object has normSigma0/1 = FLT_MAX, which is how RLOF
    # spells "plain L2, no M-estimator"; setUseMEstimator(True) swaps in the
    # Lorentzian sigmas. There is no getUseMEstimator to read it back.
    p.setUseMEstimator(bool(v["use_m_estimator"]))
    p.setUseIlluminationModel(bool(v["use_illumination_model"]))
    p.setUseGlobalMotionPrior(bool(v["use_global_motion_prior"]))
    p.setCrossSegmentationThreshold(int(v["cross_segmentation_threshold"]))
    p.setMinEigenValue(float(v["min_eigen_value"]))
    p.setGlobalMotionRansacThreshold(
        float(v["global_motion_ransac_threshold"]))
    return p


_RLOF_ADVANCED_INPUTS = [
    io.Combo.Input("solver_type", options=list(_RLOF_SOLVER),
        default="bilinear (accurate, default)", optional=True, advanced=True,
        tooltip="Iteration scheme. 'bilinear' interpolates sub-pixel and is "
                "more accurate; 'standard' samples on the pixel grid and is "
                "faster."),
    io.Combo.Input("support_region", options=list(_RLOF_SUPPORT),
        default="cross-based segmentation (default)", optional=True,
        advanced=True,
        tooltip="Shape of the region matched around each point. 'cross-based' "
                "grows the window along colour-similar pixels, which is what "
                "keeps RLOF sharp at motion boundaries; 'fixed window' is the "
                "classic square block."),
    io.Int.Input("small_win_size", default=9, min=3, max=51, optional=True,
        advanced=True,
        tooltip="Window used at the finest level / inside a support region."),
    io.Int.Input("large_win_size", default=21, min=3, max=101, optional=True,
        advanced=True,
        tooltip="Largest matching window; bigger tolerates bigger motion and "
                "blurs fine detail."),
    io.Int.Input("max_level", default=4, min=0, max=8, optional=True,
        advanced=True,
        tooltip="Pyramid levels (0 = no pyramid). More levels track larger "
                "displacements."),
    io.Int.Input("max_iterations", default=30, min=1, max=200, optional=True,
        advanced=True,
        tooltip="Iterations per level before giving up on a point."),
    io.Boolean.Input("use_m_estimator", default=False, optional=True,
        advanced=True,
        tooltip="Use the robust Lorentzian norm instead of plain L2. cv2's own "
                "default is OFF (the parameter object ships with the norm "
                "sigmas at FLT_MAX). Turning it on costs time and helps when "
                "the two frames contain outlier motion."),
    io.Boolean.Input("use_illumination_model", default=True, optional=True,
        advanced=True,
        tooltip="Solve for a per-region brightness/contrast change as well as "
                "the motion - the 'robust' in RLOF. Turn it off only if the "
                "exposure is provably constant and you need the speed."),
    io.Boolean.Input("use_global_motion_prior", default=True, optional=True,
        advanced=True,
        tooltip="Estimate a global (camera) motion first and start every point "
                "from it. Helps a panning camera, hurts nothing much."),
    io.Int.Input("cross_segmentation_threshold", default=25, min=0, max=255,
        optional=True, advanced=True,
        tooltip="Colour difference at which the cross-based support region "
                "stops growing (ignored for a fixed window)."),
    io.Float.Input("min_eigen_value", default=0.0001, min=0.0, max=1.0,
        step=0.0001, optional=True, advanced=True,
        tooltip="Points whose structure tensor is flatter than this are "
                "declared untrackable."),
    io.Float.Input("global_motion_ransac_threshold", default=10.0, min=0.0,
        max=100.0, step=0.5, optional=True, advanced=True,
        tooltip="RANSAC inlier threshold used when fitting the global motion "
                "prior (ignored when that prior is off)."),
]

_RLOF_ADVANCED_DEFAULTS = dict(
    solver_type="bilinear (accurate, default)",
    support_region="cross-based segmentation (default)",
    small_win_size=9, large_win_size=21, max_level=4, max_iterations=30,
    use_m_estimator=False, use_illumination_model=True,
    use_global_motion_prior=True, cross_segmentation_threshold=25,
    min_eigen_value=0.0001, global_motion_ransac_threshold=10.0,
)


class OpticalFlowTVL1(io.ComfyNode):
    """cv2.optflow.DualTVL1OpticalFlow - the classic variational dense flow.

    Built with `DualTVL1OpticalFlow_create()`. `createOptFlow_DualTVL1()` also
    works on this build (it returns the same concrete `DualTVL1OpticalFlow`
    despite a docstring that says "Creates instance of cv::DenseOpticalFlow"),
    but the explicit factory is the one that cannot regress into the abstract
    base - which is exactly what the other three optflow factories DO return,
    see OpticalFlowClassic. `test_optflow_tvl1` pins that both are concrete
    here, so a build where that stops being true fails loudly.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_OpticalFlowTVL1",
            display_name="CV Optical Flow (TV-L1)",
            category=CATEGORY,
            description="Dense optical flow by TV-L1 energy minimization "
                        "(cv2.optflow.DualTVL1OpticalFlow) - the accuracy "
                        "reference of the classic (non-learned) methods. Its "
                        "L1 data term tolerates illumination change and "
                        "occlusion far better than Farneback's least squares, "
                        "and its total-variation prior keeps motion "
                        "discontinuities SHARP instead of smearing them. The "
                        "price is speed: expect seconds per megapixel, roughly "
                        "an order of magnitude slower than 'CV Optical "
                        "Flow (DIS)'. Use it for offline quality work and DIS "
                        "for anything interactive. Connect 'init_flow' (the "
                        "previous pair's flow) to warm-start a video sequence.",
            inputs=[
                polytype.poly_input("frame_a", types=polytype.IMAGE_ONLY,
                    tooltip="First frame (earlier in time); converted to "
                            "grayscale."),
                polytype.poly_input("frame_b", types=polytype.IMAGE_ONLY,
                    tooltip="Second frame; resized to frame_a if the sizes "
                            "differ."),
                io.Float.Input("lambda_", default=0.15, min=0.001, max=2.0,
                    step=0.01,
                    tooltip="Weight of the data term against the smoothness "
                            "term. LOWER = smoother flow (the prior wins); "
                            "higher follows the pixels and picks up noise."),
                io.Int.Input("scales", default=5, min=1, max=10,
                    tooltip="Pyramid levels. More levels capture larger "
                            "motion; each costs time."),
                io.Int.Input("warps", default=5, min=1, max=20,
                    tooltip="Warping steps per pyramid level - the outer "
                            "linearization loop. More is more accurate and "
                            "linearly slower; this is the main speed knob."),
                io.Float.Input("epsilon", default=0.01, min=1e-4, max=1.0,
                    step=0.001,
                    tooltip="Stopping threshold. Tighter (smaller) is more "
                            "accurate and much slower."),
                NPArray.Input("init_flow", optional=True,
                    tooltip="Optional HxWx2 float32 flow to start from - pass "
                            "the previous frame pair's flow when processing "
                            "video. It is copied, never written to, so the "
                            "upstream node's cached array stays safe."),
                io.Float.Input("tau", default=0.25, min=0.01, max=1.0,
                    step=0.01, optional=True, advanced=True,
                    tooltip="Time step of the dual formulation. The theory "
                            "needs tau < 0.125 for guaranteed convergence; "
                            "cv2 ships 0.25 because it works in practice."),
                io.Float.Input("theta", default=0.3, min=0.01, max=2.0,
                    step=0.01, optional=True, advanced=True,
                    tooltip="Tightness of the coupling between the two "
                            "variables the dual scheme splits the problem "
                            "into. Small values slow convergence."),
                io.Float.Input("gamma", default=0.0, min=0.0, max=1.0,
                    step=0.01, optional=True, advanced=True,
                    tooltip="Weight of the gradient-constancy term. Above 0 it "
                            "adds robustness to brightness changes at some "
                            "cost; cv2's default is 0 (off)."),
                io.Float.Input("scale_step", default=0.8, min=0.1, max=0.95,
                    step=0.05, optional=True, advanced=True,
                    tooltip="Size ratio between consecutive pyramid levels."),
                io.Int.Input("inner_iterations", default=30, min=1, max=200,
                    optional=True, advanced=True,
                    tooltip="Inner iterations used to solve the linearized "
                            "problem at each warp."),
                io.Int.Input("outer_iterations", default=10, min=1, max=100,
                    optional=True, advanced=True,
                    tooltip="Outer iterations of the whole scheme."),
                io.Int.Input("median_filtering", default=5, min=1, max=9,
                    optional=True, advanced=True,
                    tooltip="Aperture of the median filter applied between "
                            "iterations to reject outliers. 1 disables it."),
            ],
            outputs=_FLOW_OUTPUT_SCHEMA,
        )

    @classmethod
    def execute(cls, frame_a, frame_b, lambda_, scales, warps, epsilon,
                init_flow=None, tau=0.25, theta=0.3, gamma=0.0, scale_step=0.8,
                inner_iterations=30, outer_iterations=10,
                median_filtering=5) -> io.NodeOutput:
        optflow = _require("optflow", "DualTVL1OpticalFlow_create")
        prev, nxt = _flow_frame_pair(frame_a, frame_b)
        tvl1 = optflow.DualTVL1OpticalFlow_create()
        tvl1.setLambda(float(lambda_))
        tvl1.setScalesNumber(int(scales))
        tvl1.setWarpingsNumber(int(warps))
        tvl1.setEpsilon(float(epsilon))
        tvl1.setTau(float(tau))
        tvl1.setTheta(float(theta))
        tvl1.setGamma(float(gamma))
        tvl1.setScaleStep(float(scale_step))
        tvl1.setInnerIterations(int(inner_iterations))
        tvl1.setOuterIterations(int(outer_iterations))
        tvl1.setMedianFiltering(int(median_filtering))

        init = _init_flow_for(init_flow, prev.shape[:2])
        tvl1.setUseInitialFlow(init is not None)
        return _flow_outputs(tvl1.calc(prev, nxt, init))


class OpticalFlowRLOF(io.ComfyNode):
    """cv2.optflow.calcOpticalFlowDenseRLOF - sparse RLOF + interpolation.

    The free function is not generated because of its `rlofParam` argument, and
    the CLASS (createOptFlow_DenseRLOF) is worse than useless here: it comes
    back typed as the abstract `DenseOpticalFlow`, so none of the 14 knobs
    below - all of which are arguments of the free function - can be reached
    through it. Hence: build the parameter object, call the function.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_OpticalFlowRLOF",
            display_name="CV Optical Flow (RLOF, Dense)",
            category=CATEGORY,
            description="Dense optical flow by Robust Local Optical Flow "
                        "(cv2.optflow.calcOpticalFlowDenseRLOF): it tracks a "
                        "GRID of points with an illumination-robust local "
                        "solver, then interpolates them into a dense field "
                        "with the same edge-aware interpolators as 'CV "
                        "Disparity Interpolate'. That makes it the "
                        "quality/speed middle ground between DIS and TV-L1, "
                        "and the best of the three when the two frames differ "
                        "in EXPOSURE, because the illumination model is part "
                        "of the per-point solve. Unlike every other flow node "
                        "here it needs COLOUR: the cross-based support region "
                        "segments on colour, so a grayscale input is expanded "
                        "to BGR and the method loses its main advantage. "
                        "Raise 'grid_step' for speed, lower it for detail.",
            inputs=[
                polytype.poly_input("frame_a", types=polytype.IMAGE_ONLY,
                    tooltip="First frame. Kept in COLOUR (8-bit BGR) - RLOF "
                            "uses the colour to grow its support regions."),
                polytype.poly_input("frame_b", types=polytype.IMAGE_ONLY,
                    tooltip="Second frame, same size as frame_a."),
                io.Int.Input("grid_step", default=6, min=1, max=64,
                    tooltip="Spacing in pixels of the point grid that is "
                            "actually tracked; everything between is "
                            "interpolated. The dominant speed/detail knob "
                            "(cv2 default 6)."),
                io.Combo.Input("interpolation", options=list(_RLOF_INTERP),
                    default="EPIC (edge-preserving, default)",
                    tooltip="How the tracked grid becomes a dense field. EPIC "
                            "fits a local affine model along image edges, RIC "
                            "fits one plane per SLIC superpixel (sharpest at "
                            "motion boundaries, slowest, and it RANDOMIZES its "
                            "model fitting so its output varies slightly run "
                            "to run), GEO is a plain geodesic weighting and is "
                            "the fastest."),
                io.Float.Input("forward_backward_threshold", default=1.0,
                    min=0.0, max=30.0, step=0.5,
                    tooltip="Drop a grid point when re-tracking it back misses "
                            "its origin by more than this many pixels. 0 "
                            "disables the check and keeps every point, "
                            "including the ones that ran into an occlusion."),
                io.Boolean.Input("use_variational_refinement", default=False,
                    tooltip="Run a variational polish over the interpolated "
                            "field (the same refinement DIS ends with). "
                            "Sharpens boundaries at a noticeable cost; cv2's "
                            "default is off."),
                io.Boolean.Input("use_post_proc", default=True, optional=True,
                    advanced=True,
                    tooltip="Run the fast global smoother after "
                            "interpolation. Recommended - without it the "
                            "per-region models stay visible as blocky steps."),
                io.Int.Input("epic_k", default=128, min=1, max=512,
                    optional=True, advanced=True,
                    tooltip="EPIC/GEO: neighbouring tracked points used to fit "
                            "each local model. Larger = smoother, slower."),
                io.Float.Input("epic_sigma", default=0.05, min=0.001, max=1.0,
                    step=0.005, optional=True, advanced=True,
                    tooltip="EPIC/GEO: falloff of the geodesic distance "
                            "weighting."),
                io.Float.Input("epic_lambda", default=999.0, min=0.0,
                    max=5000.0, step=1.0, optional=True, advanced=True,
                    tooltip="EPIC/GEO: regularization of the local affine "
                            "fit."),
                io.Int.Input("ric_superpixel_size", default=15, min=4, max=64,
                    optional=True, advanced=True,
                    tooltip="RIC only: average SLIC superpixel side in pixels. "
                            "Smaller follows finer structure at more cost."),
                io.Combo.Input("ric_slic_type", options=list(_RIC_SLIC),
                    default="SLIC", optional=True, advanced=True,
                    tooltip="RIC only: superpixel algorithm. SLIC is the "
                            "baseline, SLICO adapts its compactness "
                            "automatically, MSLIC is the manifold variant."),
                io.Float.Input("fgs_lambda", default=500.0, min=1.0,
                    max=10000.0, step=10.0, optional=True, advanced=True,
                    tooltip="Post-processing smoother strength (ignored when "
                            "use_post_proc is off)."),
                io.Float.Input("fgs_sigma", default=1.5, min=0.01, max=100.0,
                    step=0.1, optional=True, advanced=True,
                    tooltip="Post-processing edge sensitivity in colour units "
                            "(ignored when use_post_proc is off)."),
                *_RLOF_ADVANCED_INPUTS,
            ],
            outputs=_FLOW_OUTPUT_SCHEMA,
        )

    @classmethod
    def execute(cls, frame_a, frame_b, grid_step, interpolation,
                forward_backward_threshold, use_variational_refinement,
                use_post_proc=True, epic_k=128, epic_sigma=0.05,
                epic_lambda=999.0, ric_superpixel_size=15, ric_slic_type="SLIC",
                fgs_lambda=500.0, fgs_sigma=1.5, **rlof) -> io.NodeOutput:
        optflow = _require("optflow", "calcOpticalFlowDenseRLOF")
        a, b = _rlof_frame_pair(frame_a, frame_b)
        common = dict(
            rlofParam=_rlof_params(**rlof),
            forwardBackwardThreshold=float(forward_backward_threshold),
            gridStep=(int(grid_step), int(grid_step)),
            epicSigma=float(epic_sigma), epicLambda=float(epic_lambda),
            ricSPSize=int(ric_superpixel_size),
            ricSLICType=int(getattr(_require("ximgproc"),
                                    _RIC_SLIC[ric_slic_type])),
            use_post_proc=bool(use_post_proc),
            fgsLambda=float(fgs_lambda), fgsSigma=float(fgs_sigma),
            use_variational_refinement=bool(use_variational_refinement))

        # The interpolator's neighbour count cannot exceed what the tracked
        # grid supplies, and cv2 expresses that as a RAISE from deep inside
        # ximgproc ("m_validSize >= m_size ... decreasig k parameter" in
        # MinHeap::Push). The ceiling is not a clean formula - measured on a
        # 120x160 pair it is k<=96 at grid_step 6 but k<=19 at grid_step 20, and
        # a 240x320 pair takes the full 256 - so halve k until it fits rather
        # than pretending to know the bound. A flow node must not halt a
        # workflow, so the last resort is EPIC, which needs far fewer
        # seeds than RIC.
        interp = _RLOF_INTERP[interpolation]
        k = max(1, int(epic_k))
        while True:
            try:
                flow = optflow.calcOpticalFlowDenseRLOF(
                    a, b, None, interp_type=_optflow_const(interp),
                    epicK=k, **common)
                return _flow_outputs(flow)
            except cv2.error as e:
                if k > 4:
                    k //= 2
                    logging.warning("OpticalFlowRLOF: %s interpolation rejected "
                                    "epic_k, retrying with %d", interp, k)
                    continue
                if interp != "INTERP_EPIC":
                    logging.warning("OpticalFlowRLOF: %s interpolation failed "
                                    "(%s), falling back to EPIC", interp, e)
                    interp, k = "INTERP_EPIC", max(1, int(epic_k))
                    continue
                raise


# A sparse RLOF node ('CV Track Features (RLOF)', an illumination-robust
# drop-in for 'CV Track Features (KLT)') was written and then DELETED,
# because on this OpenCV 5.0.0 build `calcOpticalFlowSparseRLOF` never writes
# its tracked positions back through the Python binding: `nextPts` comes out
# byte-identical to whatever went in. Seed it with the input points and the
# "tracked" points are the input points (zero displacement on a pair with a
# known 6 px shift); seed it with `points + 7` and the output is exactly
# `points + 7`; pass None and it returns UNINITIALISED memory (values ~1e-11
# and ~8e-43). The `SparseRLOFOpticalFlow_create().calc()` class path is the
# same code and behaves identically.
#
# The tracker itself runs - `status` and `err` are real (196/200 points on a
# matching pair against 12/200 on unrelated frames, mean err 0.08) - so a node
# built on it would look healthy, report a plausible count, and silently hand
# every downstream consumer (Recover Pose, Find Homography) a zero-motion
# correspondence set. That is a silently wrong result, so it is pinned by
# `test_sparse_rlof_never_writes_nextpts` instead of shipped: when a future
# build fixes the binding, that test fails loudly and the node can be restored
# from git history. Dense RLOF below is unaffected -
# it runs its own C++ path and tracks correctly.


class OpticalFlowClassic(io.ComfyNode):
    """DeepFlow and PCAFlow, the two parameterless cv2.optflow factories.

    Both come back typed as the abstract `DenseOpticalFlow` - the Python
    binding exposes `calc()` and nothing else, so DeepFlow's documented fields
    (alpha, delta, gamma, sigma) and PCAFlow's prior are simply not reachable.
    With no parameters to separate them, two nodes would only be two ways to
    call the same one-line body, so they share an algorithm dropdown.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_OpticalFlowClassic",
            display_name="CV Optical Flow (DeepFlow / PCAFlow)",
            category=CATEGORY,
            description="Two more dense optical-flow algorithms from "
                        "cv2.optflow, both class-gated and neither tunable "
                        "(their Python bindings expose no setters). DeepFlow "
                        "matches a descriptor pyramid before the variational "
                        "solve, which is what lets it follow LARGE "
                        "displacements that pyramidal methods lose. PCAFlow "
                        "projects the field onto a learned low-dimensional "
                        "basis: it is fast and immune to noise, but it can "
                        "only produce motion its basis can express, so fine "
                        "structure is smoothed away. Start with 'OpenCV "
                        "Optical Flow (DIS)' and come here when the motion is "
                        "too big (DeepFlow) or the input too noisy (PCAFlow).",
            inputs=[
                polytype.poly_input("frame_a", types=polytype.IMAGE_ONLY,
                    tooltip="First frame (earlier in time); converted to "
                            "grayscale."),
                polytype.poly_input("frame_b", types=polytype.IMAGE_ONLY,
                    tooltip="Second frame; resized to frame_a if the sizes "
                            "differ."),
                io.Combo.Input("algorithm", options=list(_FLOW_ALGOS),
                    default="DeepFlow (large motion)",
                    tooltip="DeepFlow: descriptor matching + variational "
                            "refinement, best on large motion, slowest here. "
                            "PCAFlow: learned-basis projection, fast and "
                            "noise-tolerant, blurs fine structure."),
            ],
            outputs=_FLOW_OUTPUT_SCHEMA,
        )

    @classmethod
    def execute(cls, frame_a, frame_b, algorithm) -> io.NodeOutput:
        optflow = _require("optflow", "createOptFlow_DeepFlow")
        prev, nxt = _flow_frame_pair(frame_a, frame_b)
        factory = {"deepflow": optflow.createOptFlow_DeepFlow,
                   "pcaflow": optflow.createOptFlow_PCAFlow}[
            _FLOW_ALGOS[algorithm]]
        return _flow_outputs(factory().calc(prev, nxt, None))


# ---------------------------------------------------------------------------
# Line segments and edge chains (cv2.ximgproc)
# ---------------------------------------------------------------------------

class FastLineDetect(io.ComfyNode):
    """cv2.ximgproc.createFastLineDetector - Canny + segment growing.

    The third line-segment family the pack exposes, and the only one that can
    take an EDGE MAP as its input (canny_aperture_size = 0), which is what makes
    it composable with the pack's own edge nodes instead of owning its
    thresholds. Output layout matches HoughLinesP / LSD so 'CV Draw
    Segments' plugs straight in.
    """

    _APERTURE = {
        "3 (default)": 3,
        "5": 5,
        "7": 7,
        "0 - the input IS an edge map": 0,
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_FastLineDetect",
            display_name="CV Detect Line Segments (FLD)",
            category=CATEGORY,
            description="Fast Line Detector (cv2.ximgproc.createFastLineDetector"
                        "): runs Canny, chains the edge pixels and splits each "
                        "chain where it stops being straight. It is a cv2 CLASS, "
                        "so the generated wrappers cannot reach it.\n\n"
                        "Against the other two: cv2_HoughLinesP votes for "
                        "INFINITE lines and cuts them afterwards (it fuses "
                        "collinear edges and needs three thresholds); 'OpenCV "
                        "Detect Line Segments (LSD)' works on the gradient field "
                        "with no edge step and measures a width and an NFA per "
                        "segment; FLD is the fastest of the three, has an "
                        "explicit length threshold, and can MERGE collinear "
                        "neighbours back together (do_merge) - the one thing LSD "
                        "will not do. Set canny_aperture_size to 0 to feed your "
                        "OWN edge map (cv2.Canny, Structured Edge Detection, a "
                        "thresholded gradient...) instead of its internal Canny. "
                        "Finding nothing is a valid result (count = 0, empty "
                        "arrays), never an error.",
            inputs=[
                polytype.poly_input("image", types=polytype.IMAGE_ONLY,
                    tooltip="Image to search (converted to grayscale "
                            "internally). Feed the ORIGINAL image - unless "
                            "canny_aperture_size is 0, in which case feed a "
                            "BINARY edge map."),
                io.Int.Input("length_threshold", default=10, min=3, max=10000,
                    tooltip="Segments shorter than this many pixels are "
                            "discarded by the detector itself (unlike LSD, "
                            "which has no length knob and needs a post-filter). "
                            "Raise it to keep only structural edges."),
                io.Float.Input("distance_threshold", default=1.414214, min=0.0,
                    max=100.0, step=0.1,
                    tooltip="A pixel farther than this from the hypothesised "
                            "line is an outlier, which ends the segment. Small "
                            "= many short straight pieces; large = fewer, "
                            "looser segments that bend."),
                io.Float.Input("canny_th1", default=50.0, min=0.0, max=1000.0,
                    step=1.0,
                    tooltip="Lower hysteresis threshold of the INTERNAL Canny. "
                            "Ignored when canny_aperture_size is 0."),
                io.Float.Input("canny_th2", default=50.0, min=0.0, max=1000.0,
                    step=1.0,
                    tooltip="Upper hysteresis threshold of the internal Canny. "
                            "Ignored when canny_aperture_size is 0."),
                io.Combo.Input("canny_aperture_size",
                    options=list(cls._APERTURE), default="3 (default)",
                    tooltip="Sobel aperture of the internal Canny. '0' skips "
                            "Canny entirely and treats the input as an edge "
                            "map, so you can drive FLD from any edge node in "
                            "the pack."),
                io.Boolean.Input("do_merge", default=False,
                    tooltip="Incrementally merge segments that are collinear "
                            "and adjacent, so one long edge broken by noise "
                            "comes back as one segment instead of a chain of "
                            "pieces."),
                io.Float.Input("min_length", default=0.0, min=0.0, max=10000.0,
                    step=1.0, optional=True,
                    tooltip="Post-filter on the measured length, applied after "
                            "any merging - length_threshold gates the detector, "
                            "this gates the RESULT (they differ once do_merge "
                            "is on). 0 keeps every segment."),
            ],
            outputs=[
                NPArray.Output(display_name="segments",
                    tooltip="Nx4 float32 (x1, y1, x2, y2) endpoints - the same "
                            "layout HoughLinesP and LSD produce, so 'OpenCV "
                            "Draw Segments' plugs straight in."),
                NPArray.Output(display_name="lengths",
                    tooltip="(N,) float32 segment length in pixels."),
                io.Int.Output(display_name="count",
                    tooltip="How many segments survived - branch on it with "
                            "if/else for the nothing-found case."),
            ],
        )

    @classmethod
    def execute(cls, image, length_threshold, distance_threshold, canny_th1,
                canny_th2, canny_aperture_size, do_merge,
                min_length=0.0) -> io.NodeOutput:
        ximgproc = _require("ximgproc", "createFastLineDetector")
        img, _ = polytype.resolve(image)
        # copy: the detector gets a buffer of our own, never an upstream node's
        # cached output (the lowlevel.py in-place rule)
        gray = np.ascontiguousarray(_to_gray_u8(img)).copy()
        fld = ximgproc.createFastLineDetector(
            int(length_threshold), float(distance_threshold), float(canny_th1),
            float(canny_th2), cls._APERTURE[canny_aperture_size],
            bool(do_merge))
        lines = fld.detect(gray)
        # FLD returns None when it finds nothing - data, not a failure
        segs = (np.zeros((0, 4), np.float32) if lines is None
                else np.asarray(lines, np.float32).reshape(-1, 4))
        lengths = np.hypot(segs[:, 2] - segs[:, 0],
                           segs[:, 3] - segs[:, 1]).astype(np.float32)
        if min_length > 0.0 and len(segs):
            keep = lengths >= float(min_length)
            segs, lengths = segs[keep], lengths[keep]
        return io.NodeOutput(segs, lengths, len(segs))


class EdgeDrawingDetect(io.ComfyNode):
    """cv2.ximgproc.createEdgeDrawing - edge CHAINS, then lines, then ellipses.

    One node for the whole object because the three results are three calls on
    the same handle over one `detectEdges` pass: `detectLines`/`detectEllipses`
    are cheap next to it (measured: 15 ms of edges against 4 ms + 10 ms on a
    1500x1050 photo), and splitting them would push the live C++ object into the
    graph - the hazard this whole module exists to avoid.

    The segments come out as CV_CONTOURS, which is what makes this node
    composable: an ED edge chain is an ordered pixel run, exactly what the
    contour nodes already consume (arc length, approx-poly, draw, ...).
    """

    _OPERATORS = {
        "Prewitt": 0,
        "Sobel": 1,
        "Scharr": 2,
        "LSD": 3,
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_EdgeDrawing",
            display_name="CV Edge Drawing",
            category=CATEGORY,
            description="Edge Drawing (cv2.ximgproc.createEdgeDrawing): a cv2 "
                        "CLASS the generated wrappers cannot reach. Instead of "
                        "thresholding a gradient like Canny, it picks ANCHOR "
                        "pixels at gradient maxima and walks the ridge between "
                        "them, so the edge map is one-pixel-wide and already "
                        "CHAINED - you get the edge SEGMENTS as ordered point "
                        "runs (CV_CONTOURS), not a pile of lit pixels you have "
                        "to trace yourself.\n\n"
                        "On top of those chains it fits straight LINES (Nx4, "
                        "same layout as HoughLinesP / LSD / FLD) and CIRCLES + "
                        "ELLIPSES (EDCircles), which is the one thing the other "
                        "line detectors here cannot do. Nothing found is a valid "
                        "result: every output comes back empty, never None.",
            inputs=[
                polytype.poly_input("image", types=polytype.IMAGE_ONLY,
                    tooltip="Image to trace (converted to grayscale "
                            "internally). Feed the ORIGINAL image, not an edge "
                            "map - ED computes its own gradient."),
                io.Combo.Input("operator", options=list(cls._OPERATORS),
                    default="Prewitt",
                    tooltip="Gradient operator. Prewitt (cv2's default) and "
                            "Sobel are 3x3; Scharr is the more rotation-accurate "
                            "3x3 kernel; 'LSD' switches the whole line stage to "
                            "the LSD-style detector."),
                io.Int.Input("gradient_threshold", default=20, min=1, max=255,
                    tooltip="Pixels with a gradient below this can never be part "
                            "of an edge. This is the ONE knob that decides how "
                            "much low-contrast detail is traced; lower = more "
                            "edges and more noise."),
                io.Int.Input("anchor_threshold", default=0, min=0, max=255,
                    tooltip="How much a pixel must exceed its neighbours along "
                            "the gradient to become an ANCHOR (a seed the "
                            "tracing starts from). 0 anchors every local "
                            "maximum, which is cv2's default; raising it thins "
                            "the result to the strongest edges."),
                io.Int.Input("scan_interval", default=1, min=1, max=32,
                    tooltip="Anchors are only looked for every Nth row/column. "
                            "1 scans everything; higher is faster and misses "
                            "short edges."),
                io.Int.Input("min_path_length", default=10, min=0, max=10000,
                    tooltip="Edge chains shorter than this many pixels are "
                            "dropped - the declutter knob for the SEGMENTS "
                            "output."),
                io.Float.Input("sigma", default=1.0, min=0.0, max=10.0,
                    step=0.1,
                    tooltip="Gaussian blur applied before the gradient. Raise it "
                            "on noisy images; 0 disables the smoothing."),
                io.Int.Input("min_line_length", default=-1, min=-1, max=10000,
                    optional=True,
                    tooltip="Shortest line the line stage will fit, in pixels. "
                            "-1 (cv2's default) derives it from the image size."),
                io.Boolean.Input("pf_mode", default=False, optional=True,
                    advanced=True,
                    tooltip="Parameter-Free mode: ED validates every edge with "
                            "the Helmholtz principle instead of the gradient "
                            "threshold, so the result stops depending on the "
                            "knobs above. Slower, and the usual choice when you "
                            "are after ELLIPSES."),
                io.Boolean.Input("nfa_validation", default=True, optional=True,
                    advanced=True,
                    tooltip="Validate fitted lines and ellipses with the "
                            "number-of-false-alarms test. Off returns more, "
                            "less trustworthy fits."),
                io.Boolean.Input("sum_flag", default=True, optional=True,
                    advanced=True,
                    tooltip="Gradient magnitude as |gx| + |gy| (on, cv2's "
                            "default) instead of the exact hypot."),
                io.Float.Input("line_fit_error", default=1.0, min=0.0, max=10.0,
                    step=0.1, optional=True, advanced=True,
                    tooltip="Maximum RMS error, in pixels, for a piece of chain "
                            "to be accepted as one straight line."),
                io.Float.Input("max_distance_between_two_lines", default=6.0,
                    min=0.0, max=100.0, step=0.5, optional=True, advanced=True,
                    tooltip="Two collinear line pieces farther apart than this "
                            "are not merged (used by the ellipse fitting too)."),
                io.Float.Input("max_error_threshold", default=1.3, min=0.0,
                    max=10.0, step=0.1, optional=True, advanced=True,
                    tooltip="Maximum fit error accepted when circular arcs are "
                            "joined into a circle or an ellipse."),
            ],
            outputs=[
                NPArray.Output(display_name="edge_map",
                    tooltip="uint8 (H,W) mask, 255 on the traced edges - "
                            "one-pixel-wide by construction. Feed it to 'CV "
                            "Array -> Mask' or to any node that takes a Canny "
                            "result."),
                NPArray.Output(display_name="gradient",
                    tooltip="uint16 (H,W) gradient magnitude ED computed - "
                            "preview it with 'Preview CV Array' (normalize)."),
                Contours.Output(display_name="segments",
                    tooltip="The edge CHAINS as CV_CONTOURS (one ordered Nx1x2 "
                            "int32 point run each) - draw them with 'OpenCV "
                            "Draw Contours' or measure them with the contour "
                            "nodes. These are open curves, so area is "
                            "meaningless; length is not."),
                NPArray.Output(display_name="lines",
                    tooltip="Nx4 float32 (x1, y1, x2, y2) straight fits - same "
                            "layout as HoughLinesP / LSD / FLD, so 'CV Draw "
                            "Segments' plugs straight in."),
                NPArray.Output(display_name="ellipses",
                    tooltip="Nx5 float32 (cx, cy, semi_axis_a, semi_axis_b, "
                            "angle_deg) - circles come back with a == b and "
                            "angle 0. Draw them with 'CV Draw Ellipses'."),
                io.Int.Output(display_name="segment_count",
                    tooltip="How many edge chains were traced."),
                io.Int.Output(display_name="line_count",
                    tooltip="How many straight lines were fitted."),
                io.Int.Output(display_name="ellipse_count",
                    tooltip="How many circles + ellipses were fitted - branch "
                            "on it with if/else, 0 is a normal outcome on a "
                            "scene with no round objects."),
            ],
        )

    @classmethod
    def execute(cls, image, operator, gradient_threshold, anchor_threshold,
                scan_interval, min_path_length, sigma, min_line_length=-1,
                pf_mode=False, nfa_validation=True, sum_flag=True,
                line_fit_error=1.0, max_distance_between_two_lines=6.0,
                max_error_threshold=1.3) -> io.NodeOutput:
        ximgproc = _require("ximgproc", "createEdgeDrawing")
        img, _ = polytype.resolve(image)
        gray = np.ascontiguousarray(_to_gray_u8(img)).copy()

        ed = ximgproc.createEdgeDrawing()
        params = ximgproc.EdgeDrawing.Params()
        params.EdgeDetectionOperator = cls._OPERATORS[operator]
        params.GradientThresholdValue = int(gradient_threshold)
        params.AnchorThresholdValue = int(anchor_threshold)
        params.ScanInterval = max(1, int(scan_interval))
        params.MinPathLength = int(min_path_length)
        params.Sigma = float(sigma)
        params.MinLineLength = int(min_line_length)
        params.PFmode = bool(pf_mode)
        params.NFAValidation = bool(nfa_validation)
        params.SumFlag = bool(sum_flag)
        params.LineFitErrorThreshold = float(line_fit_error)
        params.MaxDistanceBetweenTwoLines = float(max_distance_between_two_lines)
        params.MaxErrorThreshold = float(max_error_threshold)
        ed.setParams(params)

        ed.detectEdges(gray)
        edge_map = np.ascontiguousarray(np.asarray(ed.getEdgeImage(), np.uint8))
        gradient = np.ascontiguousarray(np.asarray(ed.getGradientImage(),
                                                   np.uint16))
        # getSegments() gives (K,2) int32 runs; CV_CONTOURS is Nx1x2
        segments = [np.ascontiguousarray(np.asarray(s, np.int32).reshape(-1, 1, 2))
                    for s in ed.getSegments()]

        raw_lines = ed.detectLines()
        lines = (np.zeros((0, 4), np.float32) if raw_lines is None
                 else np.asarray(raw_lines, np.float32).reshape(-1, 4))

        # detectEllipses packs BOTH shapes into one 6-column row: a circle is
        # (cx, cy, radius, 0, 0, 0) and an ellipse (cx, cy, 0, a, b, theta), so
        # the drawable semi-axes are always col2 + col3 and col2 + col4
        raw_ell = ed.detectEllipses()
        if raw_ell is None:
            ellipses = np.zeros((0, 5), np.float32)
        else:
            e = np.asarray(raw_ell, np.float64).reshape(-1, 6)
            ellipses = np.stack([e[:, 0], e[:, 1], e[:, 2] + e[:, 3],
                                 e[:, 2] + e[:, 4], e[:, 5]],
                                axis=1).astype(np.float32)
        return io.NodeOutput(edge_map, gradient, segments, lines, ellipses,
                             len(segments), len(lines), len(ellipses))


class RapidTrackSequence(io.ComfyNode):
    """Runs cv2.rapid over a whole IMAGE batch, threading the pose frame to frame.

    Like the background subtractors above, this is only meaningful over a
    SEQUENCE: each frame starts from the pose the previous one ended at. The
    per-frame pose cannot be threaded through the graph by hand (it would need a
    loop with feedback), so the whole batch is consumed in one execute().
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CV_RapidTrackSequence",
            display_name="CV Rapid Track (Sequence)",
            category=CATEGORY,
            description="Tracks a known 3D mesh through an IMAGE batch with "
                        "cv2.rapid (Harris & Stennett 1990): for each frame it "
                        "samples control points along the model's projected "
                        "silhouette, searches perpendicular to the contour for "
                        "the strongest image gradient, and refines the pose by "
                        "PnP. Every frame starts from the previous frame's "
                        "result, so it is a TRACKER, not a detector - it needs "
                        "an initial pose that is already close (tens of pixels) "
                        "and small motion between frames.\n\n"
                        "Feed it a mesh from 'CV Mesh From 3D Model' via "
                        "'CV Mesh Split Long Edges': a coarse mesh makes "
                        "cv2.rapid's control points wrong by tens of pixels and "
                        "the track diverges (see that node's description).\n\n"
                        "Failure tolerant: if the model leaves the frame or its "
                        "silhouette collapses, cv2 raises internally; the node "
                        "stops advancing, reports tracked=false and repeats the "
                        "last good pose for the remaining frames instead of "
                        "halting the workflow.",
            inputs=[
                io.Image.Input("frames",
                    tooltip="The clip to track through, as one IMAGE batch "
                            "(a video loader, or core 'Batch Images'). Frames "
                            "are processed in order and the pose carries over, "
                            "so the batch must BE the sequence."),
                NPArray.Input("pts3d",
                    tooltip="Nx3 model vertices, in the same units as tvec."),
                NPArray.Input("tris",
                    tooltip="Mx3 int triangle indices. Used for backface "
                            "culling and the silhouette, so the winding "
                            "matters."),
                NPArray.Input("K",
                    tooltip="3x3 camera matrix of the FRAMES as given "
                            "('CV Camera Matrix' or a calibration). Lens "
                            "distortion is not modelled by cv2.rapid - undistort "
                            "the frames first if the lens is wide."),
                NPArray.Input("rvec",
                    tooltip="Initial rotation (Rodrigues 3x1) for the FIRST "
                            "frame. Typically from solvePnP on a marker, or the "
                            "pose the object was placed at."),
                NPArray.Input("tvec",
                    tooltip="Initial translation (3x1) for the first frame, in "
                            "the mesh's units."),
                io.Int.Input("num_control_points", default=128, min=8, max=2048,
                    tooltip="How many points to sample along the silhouette per "
                            "iteration. More points average out bad matches but "
                            "cost linearly. 128 is a good default; below ~32 a "
                            "few wrong correspondences can swing the pose."),
                io.Int.Input("search_length", default=12, min=1, max=256,
                    tooltip="Half-length of the search line, in pixels of the "
                            "SCALED frame - the tracker looks this far either "
                            "side of the predicted contour. It sets the capture "
                            "range, but NOT for free: cv2.rapid takes the "
                            "strongest gradient anywhere on the line, so a long "
                            "line lets texture and background edges outvote the "
                            "silhouette. Measured on a textured vehicle over "
                            "clutter, raising it from 12 to 96 pinned the error "
                            "at ~46 px no matter how good the start was. Prefer "
                            "raising capture range with 'scale' instead."),
                io.Int.Input("iterations", default=2, min=1, max=20,
                    tooltip="Refinement passes per frame. 2 is usually the "
                            "sweet spot; 1 lags behind the motion, and many "
                            "passes on a static frame drift rather than settle "
                            "(cv2.rapid's control points are re-extracted each "
                            "pass, so it is not a converging solver)."),
                io.Float.Input("scale", default=1.0, min=0.05, max=1.0,
                    step=0.05, optional=True,
                    tooltip="Downscale the frames before tracking (the camera "
                            "matrix is scaled to match, so the pose stays in the "
                            "original units). This is the CAPTURE-RANGE knob, "
                            "and the one to reach for before search_length: the "
                            "range it buys is search_length/scale in original "
                            "pixels, it costs LESS than the equivalent "
                            "search_length (the mask rasterization shrinks "
                            "quadratically), and the INTER_AREA decimation "
                            "suppresses the texture edges that mislead a long "
                            "search line rather than adding more of them. "
                            "It is NOT free accuracy: on a clip that starts on "
                            "the object and only needs precision, dropping to "
                            "0.5 took the measured error from 2.8 px to 18 px. "
                            "Leave it at 1.0 unless the pose you start from, or "
                            "the motion between frames, is genuinely outside "
                            "search_length."),
            ],
            outputs=[
                NPArray.Output(display_name="rvecs",
                    tooltip="Bx3x1 float64 stack: the rotation for each frame of "
                            "the batch, in order. Slice one out with 'CV Index "
                            "Batch' to project or render at that frame's pose."),
                NPArray.Output(display_name="tvecs",
                    tooltip="Bx3x1 float64 stack of translations, aligned with "
                            "rvecs."),
                NPArray.Output(display_name="ratios",
                    tooltip="Bx1 float32: per frame, the fraction of search "
                            "lines that produced a usable correspondence "
                            "(cv2.rapid's return value). A RELATIVE health "
                            "signal - a sudden drop means the model came "
                            "unstuck. Not an accuracy measure."),
                NPArray.Output(display_name="rmsds",
                    tooltip="Bx1 float32: per frame, cv2.rapid's own 2D "
                            "reprojection difference. Low rmsd with a wrong pose "
                            "is possible (the tracker can fit a self-consistent "
                            "but shifted silhouette), so read it together with "
                            "ratios rather than alone."),
                NPArray.Output(display_name="rvec",
                    tooltip="Rotation after the LAST tracked frame - the "
                            "starting pose for a following clip."),
                NPArray.Output(display_name="tvec",
                    tooltip="Translation after the last tracked frame."),
                io.Boolean.Output(display_name="tracked",
                    tooltip="False if cv2.rapid failed on any frame (the model "
                            "left the view, or its silhouette produced no usable "
                            "vertices). The pose outputs are still valid up to "
                            "that frame and repeat afterwards. Gate downstream "
                            "compositing on it with an 'If/Else Switch'."),
            ],
        )

    @classmethod
    def execute(cls, frames, pts3d, tris, K, rvec, tvec, num_control_points,
                search_length, iterations, scale=1.0) -> io.NodeOutput:
        _require("rapid", "rapid")
        images = _bgr_frames(frames)
        pts = np.ascontiguousarray(np.asarray(polytype.resolve(pts3d)[0],
                                              np.float32).reshape(-1, 3))
        tri = np.ascontiguousarray(np.asarray(polytype.resolve(tris)[0],
                                              np.int32).reshape(-1, 3))
        cam = np.asarray(polytype.resolve(K)[0], np.float64).reshape(3, 3)
        rv = np.asarray(polytype.resolve(rvec)[0], np.float64).reshape(3, 1).copy()
        tv = np.asarray(polytype.resolve(tvec)[0], np.float64).reshape(3, 1).copy()
        s = float(np.clip(scale, 0.05, 1.0))
        cam_s = np.diag([s, s, 1.0]) @ cam

        rvecs, tvecs, ratios, rmsds = [], [], [], []
        tracked = len(images) > 0 and len(pts) >= 3 and len(tri) >= 1
        pbar = comfy.utils.ProgressBar(len(images))
        for frame in images:
            if tracked:
                img = frame if s == 1.0 else cv2.resize(
                    frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                ratio = rmsd = 0.0
                try:
                    for _ in range(int(iterations)):
                        ratio, rv, tv, rmsd = cv2.rapid.rapid(
                            img, int(num_control_points), int(search_length),
                            pts, tri, cam_s, rv, tv)
                except cv2.error as e:
                    # the model left the frame, or its silhouette gave no usable
                    # vertices - keep the last good pose and stop advancing
                    logging.warning("RapidTrackSequence: lost the track at frame "
                                    "%d (%s)", len(rvecs), e)
                    tracked = False
                    ratio = rmsd = 0.0
                rv = np.asarray(rv, np.float64).reshape(3, 1)
                tv = np.asarray(tv, np.float64).reshape(3, 1)
                ratios.append(float(ratio))
                rmsds.append(float(rmsd))
            else:
                ratios.append(0.0)
                rmsds.append(0.0)
            rvecs.append(rv.copy())
            tvecs.append(tv.copy())
            # the outer loop only: after a lost track the inner iterations are
            # skipped, so a len(images) * iterations total would over-count
            pbar.update(1)

        if not rvecs:  # empty batch: still emit usable shapes
            return io.NodeOutput(np.zeros((0, 3, 1)), np.zeros((0, 3, 1)),
                                 np.zeros((0, 1), np.float32),
                                 np.zeros((0, 1), np.float32), rv, tv, False)
        return io.NodeOutput(
            np.ascontiguousarray(np.stack(rvecs)),
            np.ascontiguousarray(np.stack(tvecs)),
            np.asarray(ratios, np.float32).reshape(-1, 1),
            np.asarray(rmsds, np.float32).reshape(-1, 1),
            rv, tv, bool(tracked))


NODES = [WhiteBalance, Superpixels, Saliency,
         BackgroundSubtractBatch, ImageHashCompare,
         StereoDisparityWLS, DisparityFilterWLS, DisparityInterpolate,
         ICPRegister,
         PPFPoseEstimation,
         RapidTrackSequence,
         HfsSegment, OpticalFlowTVL1, OpticalFlowRLOF, OpticalFlowClassic,
         FastLineDetect, EdgeDrawingDetect]
