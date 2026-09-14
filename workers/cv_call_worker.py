# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Generic interruptible cv2 call worker.
Standalone — no opencv_nodes / ComfyUI imports at module level.

The main process offloads slow cv2 calls (lowlevel.OFFLOAD_WHEN mode
triggers and the lowlevel.OFFLOAD_ALWAYS set - e.g. xphoto.inpaint's
FSR_BEST mode, dctDenoising, pyrMeanShiftFiltering) into this subprocess so
the server can terminate the work mid-call on cancel.

Payload (pickled): {'func_name': 'xphoto.inpaint', 'jobs': [...]}
  func_name - dotted cv2 attribute path, resolved via nested getattr
  jobs      - list of {'args': [...], 'kwargs': {...}, 'ret_slot': int | None}
              args are FINAL (any pre-allocated output buffer is already
              spliced in at its positional slot by the caller, mirroring
              lowlevel._returning's wrap). When ret_slot is not None the
              caller wants the buffer at args[ret_slot] back instead of the
              function's raw return value (xphoto.inpaint returns None and
              delivers its result by writing into its dst argument).

Result (pickled): ('ok', [per-job results]) or ('error', traceback)."""

import os
import pickle
import sys


def run(func_name, jobs, result_queue):
    # Redirect stdout/stderr to suppress any cv2/numpy startup noise
    dn = open(os.devnull, 'w')
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = dn, dn
    try:
        import cv2

        fn = cv2
        for part in func_name.split('.'):
            fn = getattr(fn, part)

        results = []
        for job in jobs:
            result = fn(*job['args'], **job['kwargs'])
            ret_slot = job.get('ret_slot')
            if ret_slot is not None:
                result = job['args'][ret_slot]
            results.append(result)
        result_queue.put(('ok', results))
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
    run(data['func_name'], data['jobs'], q)
    result = q.get()
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
