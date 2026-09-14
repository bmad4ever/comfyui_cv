# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Interruptible DNN inference worker.

Runs one of the task cores from dnn_tasks.py (face/palm/hand/QR/EAST/colorize
detections, generic ONNX/TFLite forward passes, feature extract/matcher models,
and the LLM/VLM/Seq2Seq generation loops) in a subprocess so ComfyUI can
terminate it mid-call. The parent (opencv_nodes/dnn.py) resolves model names to
absolute paths, converts tensors to BGR ndarrays and passes those in the
pickled payload; the worker reloads the models fresh every run (nets are cached
per (path, engine) WITHIN one run so multi-frame nodes reuse the loaded net).

Payload (pickled): {'task': <TASK_FUNCS key>, 'args': {**kwargs}}
Result (pickled):
  ('ok', (<task return values...>))                     - success
  ('error', (<exc type name>, <message>, <traceback>))  - worker-reported
    failure; the parent re-raises the original exception type for the common
    cases (cv2.error, ValueError) so downstream behavior matches the old
    in-process path.

Standalone - no opencv_nodes / ComfyUI imports at module level.
"""

import os
import pickle
import sys


def run(task, args):
	# Redirect stdout/stderr to suppress any cv2/numpy startup noise.
	dn = open(os.devnull, 'w')
	old_out, old_err = sys.stdout, sys.stderr
	sys.stdout, sys.stderr = dn, dn
	try:
		import numpy as np

		from dnn_tasks import TASK_FUNCS

		fn = TASK_FUNCS[task]
		return ('ok', fn(**args))
	except BaseException:
		import traceback

		return ('error', (type(sys.exc_info()[1]).__name__,
		                  str(sys.exc_info()[1]), traceback.format_exc()))
	finally:
		sys.stdout, sys.stderr = old_out, old_err
		dn.close()


if __name__ == '__main__':
	input_path = sys.argv[1]
	result_path = sys.argv[2]
	with open(input_path, 'rb') as f:
		data = pickle.load(f)
	sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
	result = run(data['task'], data['args'])
	with open(result_path, 'wb') as f:
		pickle.dump(result, f)
