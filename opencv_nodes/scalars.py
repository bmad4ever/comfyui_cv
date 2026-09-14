# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""ONE spelling for a cv2 Scalar across the whole pack.

A cv2 Scalar is four doubles. Three places in this pack type one, and they used
to disagree: the curated drawing nodes take a literal string ("(0, 255, 0)"),
'CV Scalar' takes the same literal, and the generated low-level wrappers used to
take four separate FLOAT widgets. This module holds the single parser all three
share, so a value that reads correctly on one node reads identically on the
others.

The rules are 'CV Scalar''s (convert.ScalarArray):

    "128"            -> (128, 128, 128, 128)   a bare number broadcasts
    "(0, 255, 0)"    -> (0, 255, 0)            a tuple is used as written
    "(0, 255, 0, 64)" -> (0, 255, 0, 64)
    128              -> (128, 128, 128, 128)   already-numeric input (a
                                               widget-converted INT/FLOAT link)
                                               broadcasts the same way

cv2 ignores Scalar components past the target's channel count, so a shorter
tuple is legal and a longer one is not (`width`, 4 by default). No rounding
happens here - a Scalar is doubles; 'parse_color' is what rounds, because a
drawing colour also has to match the canvas channel count.

Pure Python: no cv2, no numpy, no ComfyUI - unit-testable on its own.
"""

from ast import literal_eval

MAX_COMPONENTS = 4

# Shown on every low-level scalar widget and quoted by the curated nodes.
SCALAR_TOOLTIP = (
	"cv2 Scalar as a literal, e.g. \"(0, 255, 0)\" (BGR) or \"(0, 255, 0, 64)\" "
	"(BGRA). A bare number broadcasts to every component, so \"255\" means "
	"(255, 255, 255, 255). Components past the target's channel count are "
	"ignored by OpenCV."
)


def parse(value, name="value", width=MAX_COMPONENTS, max_len=-1):
	"""-> (components, broadcast).

	`components` is a tuple of floats; `broadcast` is True when the caller wrote
	a bare number (so it was replicated `width` times) rather than a tuple.
	Callers that need to fit a different channel count - parse_color - branch on
	that flag, because replicating 255 four times must not then look like "the
	user gave me too many components".

	`max_len` bounds a written-out tuple (default: `width`; None = unbounded, for
	parse_color, which truncates an over-long colour with a warning rather than
	refusing it).
	"""
	if max_len == -1:
		max_len = width
	if isinstance(value, bool):
		raise ValueError(f'"{name}" must be a number or tuple literal, got {value!r}.')
	if isinstance(value, (int, float)):
		parsed = value
	elif isinstance(value, (tuple, list)):
		parsed = tuple(value)
	else:
		text = str(value).strip()
		if not text:
			raise ValueError(
				f'"{name}" is empty. {SCALAR_TOOLTIP}')
		try:
			parsed = literal_eval(text)
		except (ValueError, SyntaxError) as e:
			raise ValueError(
				f'"{name}" must be a number or tuple literal like "(0, 255, 0)", '
				f'got "{value}". ({e})') from e

	if isinstance(parsed, bool):
		raise ValueError(f'"{name}" must be a number or tuple literal, got {parsed!r}.')
	if isinstance(parsed, (int, float)):
		return (float(parsed),) * width, True
	if not isinstance(parsed, (tuple, list)) or not parsed:
		raise ValueError(
			f'"{name}" must be a number or a non-empty tuple literal like '
			f'"(0, 255, 0)", got {parsed!r}.')
	try:
		comps = tuple(float(c) for c in parsed)
	except (TypeError, ValueError) as e:
		raise ValueError(
			f'"{name}" must hold numbers only, got {parsed!r}. ({e})') from e
	if max_len is not None and len(comps) > max_len:
		raise ValueError(
			f'"{name}" has {len(comps)} components; a cv2 Scalar holds at most '
			f'{max_len}. Write "(B, G, R)" or "(B, G, R, A)".')
	return comps, False


def parse_scalar(value, name="value", width=MAX_COMPONENTS):
	"""The tuple to hand cv2 for a Scalar-typed argument."""
	return parse(value, name, width)[0]


def format_scalar(components):
	"""Inverse of `parse` for display/migration: (0.0, 255.0) -> "(0, 255)".

	Whole numbers render without a decimal point so a colour reads the way a
	user would type it."""
	out = []
	for c in components:
		f = float(c)
		out.append(str(int(f)) if f == int(f) else repr(f))
	return "(" + ", ".join(out) + ")"
