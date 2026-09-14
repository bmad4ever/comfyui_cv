# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Shared interruptible-subprocess plumbing for slow cv2 work.

All slow cv2 work that can run for minutes (lowlevel OFFLOAD_WHEN /
OFFLOAD_ALWAYS calls, stitcher runs, DNN inference, whole-execute curated
nodes) is delegated to a worker script so ComfyUI can cancel mid-call: a
blocking cv2 call cannot be stopped cooperatively, and a long one wedges the
whole run. Every worker follows the same contract:

* argv[1] = path to a pickled input payload
* argv[2] = path where the result is pickled
* result  = ('ok', payload...) or ('error', message) - FIRST element is the
  tag, the rest the payload.

run_subprocess() is the spawn + poll + terminate dance shared by every worker.
On ComfyUI cancel the worker is terminated (killed if it does not stop within
10s) and InterruptProcessingException is RE-RAISED so the whole run aborts -
a filter node must not quietly continue with a half-computed result. A worker
that crashed without writing a readable result comes back as
('crash', message); a worker that reported its own failure comes back as
('error', message). Callers decide how fatal each is (stitch falls back to
the input image; lowlevel/dnn raise).
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time

from comfy.model_management import (
	InterruptProcessingException,
	throw_exception_if_processing_interrupted,
)


def worker_script_path(name):
	"""Absolute path of a script under the pack's workers/ directory."""
	return os.path.join(
		os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
		'workers', name)


def run_subprocess(worker_script, payload, poll_interval=0.5, kill_timeout=10.0):
	"""Run `worker_script` with a pickled `payload`, polling for cancel.

	Returns the worker's ('tag', payload...) tuple. On ComfyUI cancel the
	worker is terminated and InterruptProcessingException is re-raised. A
	worker that crashes without a readable result comes back as
	('crash', message); a worker that reported its own failure as
	('error', message) - the caller decides whether that is fatal.
	"""
	input_fd, input_path = tempfile.mkstemp(suffix='.pkl')
	os.close(input_fd)
	result_fd, result_path = tempfile.mkstemp(suffix='.pkl')
	os.close(result_fd)
	try:
		with open(input_path, 'wb') as f:
			pickle.dump(payload, f)
		proc = subprocess.Popen(
			[sys.executable, worker_script, input_path, result_path],
			stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		try:
			while proc.poll() is None:
				throw_exception_if_processing_interrupted()
				time.sleep(poll_interval)
		except InterruptProcessingException:
			proc.terminate()
			try:
				proc.wait(timeout=kill_timeout)
			except subprocess.TimeoutExpired:
				proc.kill()
				proc.wait()
			raise
		if proc.returncode != 0:
			return ('crash', f"worker exited with code {proc.returncode}.")
		try:
			with open(result_path, 'rb') as f:
				result = pickle.load(f)
		except Exception as exc:
			return ('crash', f"failed to read worker result: {exc}")
		if not isinstance(result, tuple) or not result or not result[0]:
			return ('crash', "worker wrote a malformed result.")
		return result
	finally:
		for _p in (input_path, result_path):
			try:
				os.unlink(_p)
			except OSError:
				pass
