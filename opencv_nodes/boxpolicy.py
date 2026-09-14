# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Bounding-box sizing policy: one composite widget, one implementation.

Cropping a region is two decisions - WHERE (the tight bounds of a mask /
detection) and HOW BIG (padding, squareness, a size that a VAE accepts, a
common size so the crops stack into one batch).  The second decision is pure
geometry, it is identical for every crop node, and it is the part a user
actually wants to experiment with, so it lives here instead of being baked
into each node.

The policy crosses the graph as ONE pipe-joined string, exactly like the
`FlagGroup` values in enums.py::

    "one source, regions -> batch | padding=24 | size_multiple=8"

The first token is the MODE (the "base", mutually exclusive); the rest are
``key=value`` overrides.  Only NON-DEFAULT fields are written, so the string
stays short, hand-readable, and - crucially - "unset" is distinguishable from
"set to the default".  That is what makes an upstream 'CV Snap BBoxes' node
compose with a consumer node's own widgets: the policy overrides exactly the
fields it names and nothing else.

In the UI the string renders as a base dropdown plus one typed sub-widget per
field (web/cv_policy.js), the same trick web/cv_flags.js plays for OR-able
cv2 flags.  While the input is linked the sub-widgets hide and the upstream
node decides the value.

The mode names a USE CASE rather than exposing orthogonal axes, because the
axes multiply out to combinations nobody wants (and some that cannot exist -
per-region sizes cannot stack into a batch).  It answers exactly two questions:

* which frame does each region GROUP read - always frame 0, or frame g?
* does the node emit ONE stacked batch, or one item per region?

The crop nodes are ``is_input_list`` + ``is_output_list``, so "emit a batch" is
a one-element output list and "emit per region" is an N-element one; downstream
executes once or N times accordingly.  See MODE_TOOLTIP and the node
descriptions.  There is deliberately no separate "bboxes list" concept:
BOUNDING_BOX already nests per frame, and a ComfyUI list of them is simply
concatenated (see :func:`flatten_groups`).

Order of operations in :func:`apply` (fixed, and documented on every node that
exposes it):

1. pad     - ``padding`` px + ``padding_percent`` of the box's longer side,
             outward on all four sides
2. size    - ``tight`` / ``uniform`` / ``fixed`` (``auto`` = uniform when the
             mode stacks a batch, tight when it emits one item per region)
3. square  - grow the shorter side to the longer one
4. multiple- round each side UP to a multiple of ``size_multiple``
5. max_size- cap each side, rounding back DOWN to a multiple
6. space   - cap each side at the reference height/width (again rounding down
             to a multiple), then re-centre on the original box centre and
             shift the box inside the reference

Steps 5 and 6 round DOWN so a clamped box still satisfies ``size_multiple``:
a 100x100 image with ``size_multiple=8`` yields 96x96, never 100x100.
"""

import numpy as np

# --- the schema ------------------------------------------------------------

# base modes, worded neutrally ("regions") so ONE policy string drives both the
# bbox-driven and the mask-driven crop node - that is what lets a single
# 'Snap BBoxes' fan out to both in lockstep.
MODE_ONE_BATCH = "one source, regions -> batch"
MODE_ONE_LIST = "one source, regions -> list"
MODE_PER_BATCH = "per-frame regions -> batch"
MODE_PER_LIST = "per-frame regions -> list"

MODES = [MODE_ONE_BATCH, MODE_ONE_LIST, MODE_PER_BATCH, MODE_PER_LIST]

# modes whose output is a single stacked batch (every crop must share one size)
_STACKING = {MODE_ONE_BATCH, MODE_PER_BATCH}
# modes that read frame g for region group g instead of always frame 0
_PER_FRAME = {MODE_PER_BATCH, MODE_PER_LIST}

# workflows saved before the vocabulary changed carry the old size-mode names.
# They stay valid for ever: this is the same back-compat contract the FlagGroup
# values have (enums.py), and workflow 16 ships one.
LEGACY_MODES = {
	"uniform (largest)": (MODE_ONE_BATCH, None),
	"keep per box": (MODE_ONE_LIST, None),
	"fixed": (MODE_ONE_BATCH, "fixed"),
}

SIZE_CHOICES = ["auto", "tight", "uniform", "fixed"]

# name, type, default, min, max, step, label, tooltip
_FIELDS = (
	("padding", "int", 0, 0, 8192, 1, "padding (px)",
		"Grow every box outward by this many pixels before sizing - "
		"inpainting and detection re-runs work better with context."),
	("padding_percent", "float", 0.0, 0.0, 500.0, 1.0, "padding (%)",
		"Extra outward padding as a percentage of the box's LONGER side, "
		"added on top of 'padding (px)'. Scale-invariant, so small and "
		"large regions get proportional context."),
	("square", "bool", False, None, None, None, "square",
		"Grow the shorter side to match the longer one (diffusion "
		"inpainting usually wants square crops)."),
	("size_multiple", "int", 1, 1, 512, 1, "size multiple",
		"Round each side UP to a multiple of this. 8 for a stock VAE, 64 "
		"for SDXL - a crop that is not a multiple of the VAE stride is "
		"silently resized (or rejected) downstream. 1 disables it."),
	("fixed_width", "int", 0, 0, 8192, 8, "fixed width",
		"Crop width when the size mode is 'fixed' (0 = fall back to the "
		"largest box's width). Ignored in the other modes."),
	("fixed_height", "int", 0, 0, 8192, 8, "fixed height",
		"Crop height when the size mode is 'fixed' (0 = fall back to the "
		"largest box's height). Ignored in the other modes."),
	("max_size", "int", 0, 0, 8192, 8, "max size",
		"Cap both sides at this many pixels (0 = no cap), rounded back "
		"down to a multiple of 'size multiple'. Keeps one huge region "
		"from setting a uniform crop size the GPU cannot hold."),
	("size", "choice", "auto", None, None, None, "size",
		"How each crop is sized. 'auto' follows the mode: one common size "
		"when it stacks a batch, each region's own tight size when it emits "
		"one item per region. 'uniform' forces the common size even in list "
		"mode; 'tight' keeps per-region sizes (they cannot stack, so a batch "
		"mode rejects it); 'fixed' uses the fixed width/height fields.",
		SIZE_CHOICES),
)

FIELD_NAMES = tuple(f[0] for f in _FIELDS)
DEFAULTS = {f[0]: f[2] for f in _FIELDS}
_TYPES = {f[0]: f[1] for f in _FIELDS}
_CHOICES = {f[0]: f[8] for f in _FIELDS if len(f) > 8}

MODE_TOOLTIP = (
	"Which use case this crop serves. 'one source, regions -> batch' crops "
	"every region out of frame 0 at one common size and stacks them (the "
	"inpainting case - downstream runs once on the batch). '-> list' instead "
	"emits one item per region at its own tight size, so downstream runs once "
	"per region. The 'per-frame' modes pair region group g with frame g of a "
	"batch instead of always reading frame 0."
)

POLICY_TOOLTIP = (
	"Crop policy. In the UI this is a mode dropdown plus one widget per "
	"option; wire an 'CV Snap BBoxes' node in to drive several crop "
	"nodes from one place (the widgets then hide). A linked policy overrides "
	"only the fields it actually sets."
)


def spec(default_mode=None, exclude=()) -> dict:
	"""Widget metadata published to the frontend via the input's extra_dict.

	`exclude` drops fields the consuming node already exposes as its own
	widgets (CV_CropByMasks keeps its historical padding/square widgets
	so saved workflows keep loading); tests/validate_workflows.py reads the
	same spec to check saved policy strings.
	"""
	default_mode = default_mode or MODES[0]
	if default_mode not in MODES:
		raise ValueError(f"{default_mode!r} is not one of {MODES}")
	return {
		"base": list(MODES),
		"default": default_mode,
		# old size-mode bases -> [mode, size or null], so web/cv_policy.js
		# normalizes a legacy saved value the same way parse() does instead of
		# silently dropping it back to the default base
		"legacy": {old: [mode, size] for old, (mode, size) in LEGACY_MODES.items()},
		"fields": [
			dict({"name": f[0], "type": f[1], "default": f[2], "min": f[3],
			      "max": f[4], "step": f[5], "label": f[6], "tooltip": f[7]},
			     **({"choices": f[8]} if len(f) > 8 else {}))
			for f in _FIELDS if f[0] not in exclude
		],
	}


# --- string <-> dict -------------------------------------------------------

def _coerce(name, raw):
	kind = _TYPES[name]
	text = str(raw).strip()
	if kind == "choice":
		if text not in _CHOICES[name]:
			raise ValueError(
				f"Crop policy field {name}={raw!r} is not one of "
				f"{_CHOICES[name]}.")
		return text
	try:
		if kind == "int":
			return int(round(float(text)))
		if kind == "float":
			return float(text)
	except ValueError:
		raise ValueError(
			f"Crop policy field {name}={raw!r} is not a {kind}.") from None
	# bool: accept the widget's on/off as well as python-ish spellings
	if text.lower() in ("1", "true", "on", "yes"):
		return True
	if text.lower() in ("0", "false", "off", "no", ""):
		return False
	raise ValueError(f"Crop policy field {name}={raw!r} is not a boolean.")


def parse(policy, defaults=None) -> dict:
	"""Resolve a policy string to {mode, <field>: value, ...}.

	`defaults` (e.g. a node's own padding/square widgets) supplies the
	baseline; the string overrides only the keys it names. An empty/None
	policy is "everything from the baseline" - the back-compat sentinel used
	throughout this pack. Unknown keys raise (a typo must be loud).

	Legacy size-mode bases ("uniform (largest)", "keep per box", "fixed") from
	workflows saved before the vocabulary changed keep resolving, mapping onto
	the equivalent mode (+ size override).
	"""
	def _set_mode(value):
		if value in MODES:
			out["mode"] = value
			return
		if value in LEGACY_MODES:
			mode, size = LEGACY_MODES[value]
			out["mode"] = mode
			if size is not None:
				out["size"] = size
			return
		raise ValueError(
			f"Crop policy: {value!r} is not a mode {MODES} and is not a "
			f"'field=value' override.")

	out = dict(DEFAULTS)
	out["mode"] = MODES[0]
	for key, value in (defaults or {}).items():
		if key == "mode":
			_set_mode(value)
		elif key in DEFAULTS:
			out[key] = _coerce(key, value)
		else:
			raise ValueError(f"Unknown crop policy field {key!r}.")

	for part in str(policy or "").split("|"):
		part = part.strip()
		if not part:
			continue
		if "=" not in part:
			_set_mode(part)
			continue
		key, _, value = part.partition("=")
		key = key.strip()
		if key not in DEFAULTS:
			raise ValueError(
				f"Crop policy: unknown field {key!r} "
				f"(known: {', '.join(FIELD_NAMES)}).")
		out[key] = _coerce(key, value)
	return out


def stacks(cfg) -> bool:
	"""True when the resolved policy emits ONE batch instead of one item each."""
	return cfg["mode"] in _STACKING


def per_frame(cfg) -> bool:
	"""True when region group g reads frame g instead of always frame 0."""
	return cfg["mode"] in _PER_FRAME


def compose(mode=None, **fields) -> str:
	"""Build a policy string carrying the mode plus the NON-default fields."""
	mode = mode or MODES[0]
	if mode not in MODES:
		raise ValueError(f"{mode!r} is not one of {MODES}")
	parts = [mode]
	for name in FIELD_NAMES:
		if name not in fields:
			continue
		value = _coerce(name, fields[name])
		if value == DEFAULTS[name]:
			continue
		kind = _TYPES[name]
		if kind == "float":
			text = value
		elif kind == "choice":
			text = value
		else:  # int, and bool as 1/0 (what the toggle widget serializes)
			text = int(value)
		parts.append(f"{name}={text}")
	return " | ".join(parts)


# --- geometry --------------------------------------------------------------

def bbox_dict(x, y, w, h, **extra) -> dict:
	return {"x": int(x), "y": int(y), "width": int(w), "height": int(h), **extra}


def normalize(bboxes) -> list:
	"""Accept a single dict, a flat list, per-frame nested lists, or (x,y,w,h)
	sequences, and return a flat list of bbox dicts."""
	if bboxes is None:
		return []
	if isinstance(bboxes, dict):
		return [bboxes]
	if isinstance(bboxes, (list, tuple)) and len(bboxes) == 4 and all(
			isinstance(v, (int, float, np.integer, np.floating)) for v in bboxes):
		return [bbox_dict(*bboxes)]
	flat = []
	for item in bboxes:
		if isinstance(item, dict):
			flat.append(item)
		elif isinstance(item, (list, tuple)):
			flat.extend(normalize(item))  # per-frame nesting, recursively
		else:
			raise ValueError(f"Cannot interpret {item!r} as a bounding box.")
	return flat


def groups(bboxes) -> list:
	"""The per-frame GROUPS of a BOUNDING_BOX value, as a list of box lists.

	BOUNDING_BOX is natively nested one inner list per frame, so this is the
	grouping - there is deliberately no separate "bboxes list" concept. A bare
	dict or a flat list of dicts is one group; anything already nested keeps
	its groups.
	"""
	if bboxes is None:
		return []
	if isinstance(bboxes, dict):
		return [[bboxes]]
	if not isinstance(bboxes, (list, tuple)):
		raise ValueError(f"Cannot interpret {bboxes!r} as bounding boxes.")
	if not bboxes:
		return []
	# a flat [x, y, w, h] or a flat list of dicts is a single group
	if all(isinstance(b, dict) for b in bboxes):
		return [list(bboxes)]
	if len(bboxes) == 4 and all(
			isinstance(v, (int, float, np.integer, np.floating)) for v in bboxes):
		return [[bbox_dict(*bboxes)]]
	return [normalize(item) for item in bboxes]


def flatten_groups(values) -> list:
	"""Concatenate the groups of several BOUNDING_BOX values into one list.

	This is what an `is_input_list` crop node does with the ComfyUI list
	dimension: a list of BOUNDING_BOX values simply contributes its groups in
	order, so "several box sets" needs no new socket or node type.
	"""
	if values is None:
		return []
	if not isinstance(values, list) or (
			values and all(isinstance(v, dict) for v in values)):
		return groups(values)
	out = []
	for value in values:
		out.extend(groups(value))
	return out


def nest(bboxes) -> list:
	"""Wrap a flat bbox list as the per-frame nesting core nodes expect.

	The core BOUNDING_BOX convention is one inner list per image of the batch
	(comfy_extras/nodes_sdpose.py). 'Draw BBoxes' tolerates a flat list;
	'Crop By Bounding Boxes' indexes it per frame and dies on a dict, so
	everything in this pack emits the nested form.
	"""
	return [list(bboxes)]


def _round_up(value, multiple):
	if multiple <= 1:
		return int(value)
	return int(-(-int(value) // multiple) * multiple)


def _round_down(value, multiple):
	if multiple <= 1:
		return int(value)
	return int(int(value) // multiple * multiple)


def resolve_size(cfg) -> str:
	"""The effective sizing rule: 'tight', 'uniform' or 'fixed'.

	'auto' follows the mode - a batch has to share one size, a per-region list
	is free to be ragged - so the common cases need no extra widget at all.
	"""
	size = cfg.get("size", "auto")
	if size != "auto":
		return size
	return "uniform" if stacks(cfg) else "tight"


def apply(bboxes, policy=None, space_hw=None, defaults=None):
	"""Apply a sizing policy to a list of boxes.

	Returns (boxes, (width, height)) with the boxes in the same order and
	all extra keys (score, label, ...) preserved; the second element is the
	resolved uniform/fixed crop size, or the largest per-box size when the
	sizing is 'tight'. `space_hw` is the (height, width) the boxes must stay
	inside - pass the source's size and the boxes come back clamped and
	re-centred, never out of bounds. Empty input yields ([], (0, 0)).
	"""
	boxes = normalize(bboxes)
	cfg = parse(policy, defaults)
	if not boxes:
		return [], (0, 0)

	sh, sw = (None, None) if space_hw is None else (int(space_hw[0]), int(space_hw[1]))
	mult = max(1, int(cfg["size_multiple"]))

	# 1. pad outward (padded rectangles keep their true centre)
	padded = []
	for b in boxes:
		x, y = float(b["x"]), float(b["y"])
		w, h = float(b["width"]), float(b["height"])
		pad = float(cfg["padding"]) + max(w, h) * float(cfg["padding_percent"]) / 100.0
		padded.append((x - pad, y - pad, w + 2 * pad, h + 2 * pad))

	# 2. sizing
	max_w = max(p[2] for p in padded)
	max_h = max(p[3] for p in padded)
	size = resolve_size(cfg)
	if size == "uniform":
		sizes = [(max_w, max_h)] * len(padded)
	elif size == "fixed":
		fw = float(cfg["fixed_width"]) or max_w
		fh = float(cfg["fixed_height"]) or max_h
		sizes = [(fw, fh)] * len(padded)
	else:  # tight
		sizes = [(p[2], p[3]) for p in padded]

	# 3. square
	if cfg["square"]:
		sizes = [(max(w, h), max(w, h)) for w, h in sizes]

	def _cap(value, limit):
		"""Shrink to `limit`, staying a multiple of `mult` (>= mult, or 1)."""
		if value <= limit:
			return int(value)
		return _round_down(limit, mult) or max(1, int(limit))

	out, resolved_w, resolved_h = [], 0, 0
	for source, (px, py, pw, ph), (tw, th) in zip(boxes, padded, sizes):
		# 4. multiple (up), 5. max_size, 6. reference space - both cap DOWN to
		# a multiple so a clamped crop still satisfies size_multiple
		tw, th = _round_up(np.ceil(tw), mult), _round_up(np.ceil(th), mult)
		cap = int(cfg["max_size"])
		if cap > 0:
			tw, th = _cap(tw, cap), _cap(th, cap)
		if sw is not None:
			tw, th = _cap(tw, sw), _cap(th, sh)
		tw, th = max(1, int(tw)), max(1, int(th))
		resolved_w, resolved_h = max(resolved_w, tw), max(resolved_h, th)

		# re-centre the resized rectangle on the padded box's centre, then
		# shift it inside the reference space (never resize it again)
		cx, cy = px + pw / 2.0, py + ph / 2.0
		left = int(round(cx - tw / 2.0))
		top = int(round(cy - th / 2.0))
		if sw is not None:
			left = min(max(left, 0), sw - tw)
			top = min(max(top, 0), sh - th)
		extras = {k: v for k, v in source.items()
		          if k not in ("x", "y", "width", "height")}
		out.append({**extras, "x": left, "y": top, "width": tw, "height": th})

	return out, (resolved_w, resolved_h)
