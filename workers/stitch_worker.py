# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Minimal cv2 stitch worker for interruptible subprocess.
Standalone — no opencv_nodes / ComfyUI imports at module level."""

import os
import pickle
import sys


def run(image_list, mask_list, config, result_queue):
    # Redirect stdout/stderr to suppress any cv2/numpy startup noise
    dn = open(os.devnull, 'w')
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = dn, dn
    try:
        import cv2

        st = cv2.Stitcher.create(getattr(cv2, config['mode']))
        st.setRegistrationResol(config['registrationResol'])
        st.setSeamEstimationResol(config['seamEstimationResol'])
        st.setCompositingResol(config['compositingResol'])
        st.setPanoConfidenceThresh(config['panoConfidenceThresh'])
        st.setWaveCorrection(config['waveCorrection'])

        st.setInterpolationFlags(config['interpolationFlags'])

        status, panorama = st.stitch(image_list, mask_list)

        if panorama is not None:
            import numpy as np
            panorama = np.asarray(panorama, dtype=np.uint8)
        result_queue.put(('ok', int(status), panorama))
    except BaseException:
        import traceback
        result_queue.put(('error', traceback.format_exc()))
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        dn.close()


if __name__ == '__main__':
    input_path = sys.argv[1]
    result_path = sys.argv[2]
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    import multiprocessing as mp
    q = mp.Queue()
    run(data['image_list'], data['mask_list'], data['config'], q)
    result = q.get()
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
