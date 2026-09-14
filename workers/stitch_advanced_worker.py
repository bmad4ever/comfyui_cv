# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Advanced cv2 stitch worker using the detail pipeline.

Standalone subprocess — no opencv_nodes / ComfyUI imports at module level.
Provides full exposure compensation, seam finding, warp type, blend control.

The stage order and parameter set follow OpenCV's own
samples/python/stitching_detailed.py (https://github.com/opencv/opencv,
Apache-2.0)."""

import math
import os
import pickle
import sys


def _compute_scale(megapix, h, w):
    if megapix <= 0:
        return 1.0
    return min(1.0, math.sqrt(megapix * 1e6 / (h * w)))


def run(image_list, mask_list, config, result_queue):
    dn = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = dn, dn
    try:
        import cv2
        import numpy as np

        def _make_compensator(expos_type, nr_feeds, nr_filtering, block_size):
            if expos_type == "gain":
                comp = cv2.detail.GainCompensator()
            elif expos_type == "gain_blocks":
                comp = cv2.detail.BlocksGainCompensator()
            elif expos_type == "channel":
                comp = cv2.detail.ChannelsCompensator()
            elif expos_type == "channel_blocks":
                comp = cv2.detail.BlocksChannelsCompensator()
            else:
                return cv2.detail.NoExposureCompensator()
            if hasattr(comp, 'setNrFeeds') and nr_feeds > 0:
                comp.setNrFeeds(nr_feeds)
            if hasattr(comp, 'setNrGainsFilteringIterations') and nr_filtering > 0:
                comp.setNrGainsFilteringIterations(nr_filtering)
            if hasattr(comp, 'setBlockSize') and block_size > 0:
                comp.setBlockSize(block_size, block_size)
            return comp

        def _make_seam_finder(seam_type):
            if seam_type == "no":
                return cv2.detail.SeamFinder_createDefault(
                    cv2.detail.SeamFinder_NO
                )
            elif seam_type == "voronoi":
                return cv2.detail.SeamFinder_createDefault(
                    cv2.detail.SeamFinder_VORONOI_SEAM
                )
            elif seam_type == "gc_color":
                return cv2.detail.GraphCutSeamFinder("COST_COLOR")
            elif seam_type == "gc_colorgrad":
                return cv2.detail.GraphCutSeamFinder("COST_COLOR_GRAD")
            elif seam_type == "dp_color":
                return cv2.detail.DpSeamFinder("COLOR")
            elif seam_type == "dp_colorgrad":
                return cv2.detail.DpSeamFinder("COLOR_GRAD")
            return cv2.detail.SeamFinder_createDefault(
                cv2.detail.SeamFinder_VORONOI_SEAM
            )

        def _make_blender(blend_type, blend_strength, dst_corners, dst_sizes):
            dst_sz = cv2.detail.resultRoi(
                corners=dst_corners, sizes=dst_sizes
            )
            blend_width = (
                math.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            )
            if blend_type == "multiband" and blend_width >= 1:
                blender = cv2.detail_MultiBandBlender()
                nb = max(1, int(round(math.log(blend_width) / math.log(2.0) - 1.0)))
                blender.setNumBands(nb)
            elif blend_type == "feather" and blend_width >= 1:
                blender = cv2.detail_FeatherBlender()
                blender.setSharpness(1.0 / blend_width)
            else:
                blender = cv2.detail.Blender_createDefault(
                    cv2.detail.Blender_NO
                )
            blender.prepare(dst_sz)
            return blender

        n = len(image_list)
        if n < 2:
            result_queue.put(("error", "need at least 2 images"))
            return

        cfg = config
        work_megapix = cfg["registrationResol"]
        seam_megapix = cfg["seamEstimationResol"]
        compose_megapix = cfg["compositingResol"]
        conf_thresh = cfg["panoConfidenceThresh"]
        wave_type = cfg["waveCorrection"]
        interp = cfg["interpolationFlags"]
        features_type = cfg.get("features", "orb")
        matcher_type = cfg.get("matcher", "best_of_2_nearest")
        match_conf = cfg.get("matchConf", -1.0)
        range_width = cfg.get("rangeWidth", -1)
        estimator_type = cfg.get("estimator", "homography")
        ba_type = cfg.get("bundleAdjuster", "ray")
        warp_type = cfg.get("warpType", "spherical")
        expos_type = cfg.get("exposureCompensation", "gain")
        expos_nr_feeds = int(cfg.get("exposCompNrFeeds", 1))
        expos_nr_filtering = int(cfg.get("exposCompNrFiltering", 2))
        expos_block_size = int(cfg.get("exposCompBlockSize", 32))
        seam_type = cfg.get("seamFinder", "gc_colorgrad")
        blend_type = cfg.get("blendType", "multiband")
        blend_strength = cfg.get("blendStrength", 5.0)

        full_sizes = [(img.shape[1], img.shape[0]) for img in image_list]

        h0, w0 = image_list[0].shape[:2]
        work_scale = _compute_scale(work_megapix, h0, w0)

        imgs_work = []
        for img in image_list:
            if work_scale < 1.0:
                ws = int(round(w0 * work_scale))
                hs = int(round(h0 * work_scale))
                imgs_work.append(
                    cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)
                )
            else:
                imgs_work.append(img)

        if features_type == "sift":
            finder = cv2.SIFT_create()
        else:
            finder = cv2.ORB_create()

        features_list = []
        for img in imgs_work:
            feat = cv2.detail.computeImageFeatures2(finder, img)
            features_list.append(feat)

        if match_conf <= 0:
            match_conf = 0.3 if features_type == "orb" else 0.65

        if matcher_type == "best_of_2_nearest_range":
            matcher = cv2.detail_BestOf2NearestRangeMatcher(
                range_width, False, match_conf
            )
        elif matcher_type == "affine_best_of_2_nearest":
            matcher = cv2.detail_AffineBestOf2NearestMatcher(False, match_conf)
        else:
            matcher = cv2.detail_BestOf2NearestMatcher(False, match_conf)

        p = matcher.apply2(features_list)
        matcher.collectGarbage()

        indices = cv2.detail.leaveBiggestComponent(
            features_list, p, conf_thresh
        )
        if len(indices) < 2:
            result_queue.put(
                ("error", "leaveBiggestComponent: fewer than 2 images match")
            )
            return

        imgs_work = [imgs_work[i] for i in indices]
        full_imgs = [image_list[i] for i in indices]
        full_sizes = [full_sizes[i] for i in indices]
        if mask_list is not None:
            mask_subset = [mask_list[i] for i in indices]
        else:
            mask_subset = None
        n = len(full_imgs)

        if estimator_type == "affine":
            estimator = cv2.detail_AffineBasedEstimator()
        else:
            estimator = cv2.detail_HomographyBasedEstimator()
        b, cameras = estimator.apply(features_list, p, None)
        if not b:
            result_queue.put(("error", "camera estimation failed"))
            return

        for cam in cameras:
            cam.R = np.array(cam.R, dtype=np.float32)

        if ba_type == "reproj":
            adjuster = cv2.detail_BundleAdjusterReproj()
        elif ba_type == "affine_partial":
            adjuster = cv2.detail_BundleAdjusterAffinePartial()
        elif ba_type == "no":
            adjuster = cv2.detail_NoBundleAdjuster()
        else:
            adjuster = cv2.detail_BundleAdjusterRay()
        adjuster.setConfThresh(conf_thresh)
        b, cameras = adjuster.apply(features_list, p, cameras)
        if not b:
            result_queue.put(("error", "bundle adjustment failed"))
            return

        focals = [cam.focal for cam in cameras]
        focals.sort()
        if len(focals) % 2 == 1:
            warped_image_scale = focals[len(focals) // 2]
        else:
            warped_image_scale = (
                focals[len(focals) // 2] + focals[len(focals) // 2 - 1]
            ) / 2.0

        if wave_type != "no":
            if wave_type == "horiz":
                wc = cv2.detail.WAVE_CORRECT_HORIZ
            elif wave_type == "vert":
                wc = cv2.detail.WAVE_CORRECT_VERT
            else:
                wc = cv2.detail.WAVE_CORRECT_AUTO
            rmats = [np.copy(cam.R) for cam in cameras]
            rmats = cv2.detail.waveCorrect(rmats, wc)
            for idx, cam in enumerate(cameras):
                cam.R = rmats[idx]

        seam_scale = _compute_scale(seam_megapix, h0, w0)
        seam_work_aspect = seam_scale / work_scale if work_scale > 0 else 1.0

        imgs_seam = []
        for img in full_imgs:
            if seam_scale < 1.0:
                ws = int(round(img.shape[1] * seam_scale))
                hs = int(round(img.shape[0] * seam_scale))
                imgs_seam.append(
                    cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)
                )
            else:
                imgs_seam.append(img)

        masks_seam = []
        for i in range(n):
            ih, iw = imgs_seam[i].shape[:2]
            if mask_subset is not None and i < len(mask_subset):
                m = mask_subset[i]
                if m.shape[:2] != (ih, iw):
                    m = cv2.resize(
                        m, (iw, ih), interpolation=cv2.INTER_NEAREST
                    )
                masks_seam.append(m)
            else:
                masks_seam.append(np.full((ih, iw), 255, dtype=np.uint8))

        warper_seam = cv2.PyRotationWarper(
            warp_type, warped_image_scale * seam_work_aspect
        )
        corners_seam = []
        sizes_seam = []
        images_warped_seam = []
        masks_warped_seam = []
        for i in range(n):
            K = cameras[i].K().astype(np.float32)
            K[0, 0] *= seam_work_aspect
            K[0, 2] *= seam_work_aspect
            K[1, 1] *= seam_work_aspect
            K[1, 2] *= seam_work_aspect
            corner, img_wp = warper_seam.warp(
                imgs_seam[i], K, cameras[i].R, interp, cv2.BORDER_REFLECT
            )
            corners_seam.append(corner)
            sizes_seam.append((img_wp.shape[1], img_wp.shape[0]))
            images_warped_seam.append(img_wp)
            _, mask_wp = warper_seam.warp(
                masks_seam[i],
                K,
                cameras[i].R,
                cv2.INTER_NEAREST,
                cv2.BORDER_CONSTANT,
            )
            masks_warped_seam.append(mask_wp)

        compensator = _make_compensator(
            expos_type, expos_nr_feeds, expos_nr_filtering, expos_block_size
        )
        compensator.feed(
            corners=corners_seam,
            images=images_warped_seam,
            masks=masks_warped_seam,
        )

        images_warped_f = [img.astype(np.float32) for img in images_warped_seam]
        seam_finder = _make_seam_finder(seam_type)
        masks_warped_seam = seam_finder.find(
            images_warped_f, corners_seam, masks_warped_seam
        )

        compose_scale = _compute_scale(compose_megapix, h0, w0)
        compose_work_aspect = (
            compose_scale / work_scale if work_scale > 0 else 1.0
        )

        for cam in cameras:
            cam.focal *= compose_work_aspect
            cam.ppx *= compose_work_aspect
            cam.ppy *= compose_work_aspect

        warped_image_scale_comp = warped_image_scale * compose_work_aspect
        warper_comp = cv2.PyRotationWarper(warp_type, warped_image_scale_comp)

        corners_comp = []
        sizes_comp = []
        for i in range(n):
            sz = full_sizes[i]
            comp_sz = (
                int(round(sz[0] * compose_scale)),
                int(round(sz[1] * compose_scale)),
            )
            K = cameras[i].K().astype(np.float32)
            roi = warper_comp.warpRoi(comp_sz, K, cameras[i].R)
            corners_comp.append(roi[0:2])
            sizes_comp.append(roi[2:4])

        blender = _make_blender(
            blend_type, blend_strength, corners_comp, sizes_comp
        )

        for i in range(n):
            img = full_imgs[i]
            if compose_scale < 1.0:
                ws = int(round(img.shape[1] * compose_scale))
                hs = int(round(img.shape[0] * compose_scale))
                img = cv2.resize(
                    img, (ws, hs), interpolation=cv2.INTER_LINEAR
                )
            K = cameras[i].K().astype(np.float32)
            _, img_wp = warper_comp.warp(
                img, K, cameras[i].R, interp, cv2.BORDER_REFLECT
            )

            mask = np.full(img.shape[:2], 255, dtype=np.uint8)
            _, mask_wp = warper_comp.warp(
                mask, K, cameras[i].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT
            )

            compensator.apply(i, corners_comp[i], img_wp, mask_wp)
            img_wp_s = img_wp.astype(np.int16)

            if i < len(masks_warped_seam):
                dilated = cv2.dilate(masks_warped_seam[i], None)
                seam_resized = cv2.resize(
                    dilated,
                    (mask_wp.shape[1], mask_wp.shape[0]),
                    0,
                    0,
                    cv2.INTER_LINEAR_EXACT,
                )
                mask_wp = cv2.bitwise_and(seam_resized, mask_wp)

            blender.feed(cv2.UMat(img_wp_s), mask_wp, corners_comp[i])

        result, _ = blender.blend(None, None)
        if result is None:
            result_queue.put(("error", "blending returned None"))
            return

        result_u8 = np.clip(result, 0, 255).astype(np.uint8)
        result_queue.put(("ok", int(cv2.Stitcher_OK), result_u8))

    except BaseException:
        import traceback

        result_queue.put(("error", traceback.format_exc()))
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        dn.close()


if __name__ == "__main__":
    input_path = sys.argv[1]
    result_path = sys.argv[2]
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    import multiprocessing as mp

    q = mp.Queue()
    run(data["image_list"], data["mask_list"], data["config"], q)
    result = q.get()
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
