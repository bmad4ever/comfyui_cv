# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""THE composite-value socket type (CV_TUPLE) and its builder / splitter nodes.

A cv2 Size, Point, Point2f, Rect or TermCriteria is ONE value with a fixed
arity, and this module is the only place that says what that means for a
ComfyUI socket.

Why a type and not N widgets: the ergonomics rework split every composite param
into per-component widgets (dsize_w/dsize_h, center_x/center_y, ...), which made
the components wireable but also made HALF A VALUE a legal graph - link dsize_w,
leave dsize_h on its widget default, and nothing complains. The frontend cannot
help: a link has exactly one target slot and an input slot owns exactly one
widget (litegraph NodeInputSlot / LLink / widgetValuePropagation.ts), so "one
wire into a split pair" does not exist. One typed, atomic input is the only
shape that delivers it.

The shape mirrors core's ImageCropV2 (comfy_extras/nodes_images.py:59), which
replaced the deprecated ImageCrop's four INT inputs with a single
BOUNDING_BOX-typed input: ONE schema widget carrying the whole value, plus
per-component sub-widgets added by the frontend (serialize:false), plus one
builder node emitting the composite. Here:

* the value on the wire is a plain LIST of numbers ([640, 480]) - arity-generic,
  JSON-serializable, one widgets_values entry. The component LABELS come from
  the schema (published as `cv_tuple` in the input's options dict, the way
  `cv_flags` and `cv_policy` already publish theirs), so one type serves
  (w, h), (x, y) and (type, max_count, epsilon) without four io_types;
* `CvTuple.Input` subclasses io.WidgetInput, so it IS a widget as far as
  validate_workflows' is_widget() is concerned - the socket lists of every saved
  workflow stay byte-identical and the value takes exactly one widgets_values
  slot;
* `parse()` accepts every spelling that could reach the socket - a list, a
  tuple, the legacy "(w, h)" literal string that old workflows and every
  existing STRING emitter still speak, a 1-D ndarray, or a core BoundingBox
  dict - so nothing that used to resolve stops resolving.

`Number` is the second type here and it exists only for the BUILDER: a join
node's components have to accept an INT link (an image width, a kernel size) as
well as a FLOAT one, and they have to stay WIDGETS so a promoted subgraph input
an instance leaves unconnected still resolves. Its io_type is the comma-joined
"FLOAT,INT" that `isValidConnection` matches permutation-wise.

cv2's Rect is deliberately NOT handled here: it is exactly (x, y, w, h), i.e.
core's BOUNDING_BOX, which already has a widget and a builder
(PrimitiveBoundingBox). `bbox_to_rect` is the lossless bridge, and it has to
cope with this pack's OWN BOUNDING_BOX convention, which is a list nested one
group per FRAME (boxpolicy.nest) rather than core's single dict.

No cv2 import here on purpose: the parsing rules are unit-testable on plain
Python values, like scalars.py and shapepose.py.
"""

from ast import literal_eval

import numpy as np

from comfy_api.latest import io

from .convert import NPArray

CATEGORY = "image/CV/low-level"
INT32_MAX = 2**31 - 1


# --- the type ----------------------------------------------------------------

def component(name, default=0, minimum=-INT32_MAX - 1, maximum=INT32_MAX,
              is_float=False, tooltip="", options=None):
	"""One component of a composite value, as the frontend and the reassembler
	both need it. `options` (a list of names) makes it a dropdown component -
	TermCriteria's type flag - whose value stays a STRING for the caller to
	resolve; every other component is coerced to int or float here."""
	return {
		"name": name,
		"default": default,
		"min": minimum,
		"max": maximum,
		"is_float": bool(is_float),
		"tooltip": tooltip,
		"options": list(options) if options else None,
	}


def defaults(components):
	return [c["default"] for c in components]


@io.comfytype(io_type="FLOAT,INT")
class Number(io.ComfyTypeIO):
	"""A number widget whose socket accepts an INT link as well as a FLOAT one.

	This exists because 'CV Tuple' is a JOIN node: its components are fed from
	whatever computed the value, and in this pack that is usually an INT (an
	image width, a kernel size, a Math Expression's int slot) while a Point can
	be fractional. INT and FLOAT do NOT auto-coerce in the UI
	(`LiteGraphGlobal.isValidConnection` compares the two type strings), but a
	COMMA-JOINED type is matched permutation-wise - the same mechanism
	`io.MultiType` uses.

	`io.MultiType.Input` itself is unusable here: it is not a `WidgetInput`, so
	the value could not be typed in place and every harness that splits a schema
	into widgets and sockets would class it a socket. A comfytype of our own
	keeps the widget, and `web/cv_tuple.js` registers the io_type so the frontend
	renders one (an unregistered type renders as a bare socket).

	The widget fallback is the load-bearing part: a promoted subgraph input that
	an instance leaves unconnected must reach a WIDGET, or the inner node waits
	forever for a value nobody feeds. Putting a cast node in front of the join
	would break exactly that.
	"""

	Type = float

	class Input(io.WidgetInput):
		def __init__(self, id: str, default=0.0, minimum=-1e38, maximum=1e38,
		             step=0.00001, display_name: str = None, optional=False,
		             tooltip: str = None, advanced: bool = None):
			super().__init__(id, display_name, optional, tooltip, None,
			                 float(default), None, None, None,
			                 {"min": minimum, "max": maximum, "step": step},
			                 None, advanced)


class LiteralOrTyped(io.WidgetInput):
	"""A widget input whose SOCKET also accepts one typed composite link.

	The widget stays a core STRING one (`widget_type`), so the value can still
	be typed in place, takes exactly one `widgets_values` slot and keeps the
	spelling every saved workflow already carries. The SOCKET is the
	comma-joined "<TYPE>,STRING", which `LiteGraph.isValidConnection` matches
	permutation-wise - so the matching typed OUTPUT wires straight in.

	Two ComfyUI mechanics make this work, and neither is obvious:
	`litegraphService` picks the widget with `widgetType ?? type` but types the
	socket with `type`, and `io.WidgetInput.get_io_type()` returns `widget_type`
	when it is set - which would publish the socket as a plain STRING, hence the
	override. `io.MultiType.Input` reaches the same shape in core but is not a
	`WidgetInput`, so every harness that splits a schema into widgets and
	sockets (validate_workflows' `is_widget`) would class it a socket.
	"""

	# set on a standalone subclass; a nested `Input` inherits the enclosing
	# comfytype's io_type instead (only Input/Output get a `Parent`)
	COMPOSITE_TYPE = None

	def __init__(self, id: str, default="", display_name: str = None,
	             optional=False, tooltip: str = None, placeholder: str = None,
	             advanced: bool = None, force_input: bool = None):
		super().__init__(id, display_name, optional, tooltip, None, default,
		                 None, "STRING", force_input,
		                 {"placeholder": placeholder} if placeholder else None,
		                 None, advanced)

	def get_io_type(self):
		return f"{self.COMPOSITE_TYPE or self.io_type},STRING"


@io.comfytype(io_type="CV_ROTATED_RECT")
class RotatedRect(io.ComfyTypeIO):
	"""cv2's RotatedRect: ((center_x, center_y), (width, height), angle).

	Its own type because the value is NESTED and cv2 enforces the nesting -
	`cv2.boxPoints((30., 20., 10., 20., 15.))` raises "Can't parse 'box' as
	RotatedRect. Expected sequence length 3, got 5". So the flat CV_TUPLE
	spelling cannot carry one, and a type that keeps the three groups is the
	only shape that survives a round trip back into cv2.
	"""

	Type = tuple

	class Input(LiteralOrTyped):
		pass


@io.comfytype(io_type="CV_MOMENTS")
class Moments(io.ComfyTypeIO):
	"""cv2.moments' result: a dict of 24 NAMED doubles (m00 ... nu03).

	Its own type because the names are load-bearing: the Python binding reads
	the mapping BY KEY, so handing `cv2.HuMoments` the same 24 numbers as a
	list returns all zeros instead of raising - a silent wrong answer.
	"""

	Type = dict

	class Input(LiteralOrTyped):
		pass


@io.comfytype(io_type="CV_TUPLE")
class CvTuple(io.ComfyTypeIO):
	Type = list

	class Input(io.WidgetInput):
		"""A composite cv2 value as ONE input.

		Renders as a widget (web/cv_tuple.js builds one numeric/dropdown
		sub-widget per component) and, because every widget input also gets a
		socket, accepts a link from 'CV Tuple' or any node emitting CV_TUPLE.
		"""

		def __init__(self, id: str, components: list, display_name: str = None,
		             optional=False, tooltip: str = None, default=None,
		             advanced: bool = None, force_input: bool = None,
		             extra_types=None):
			if default is None:
				default = defaults(components)
			super().__init__(id, display_name, optional, tooltip, None, default,
			                 None, None, force_input,
			                 {"cv_tuple": {"components": components}}, None,
			                 advanced)
			self.components = components
			# widen the SOCKET without widening every composite input in the
			# pack: only the splitter takes a BOUNDING_BOX, because only there
			# is "4 numbers, any provenance" the whole point. Wiring a box into
			# a 2-component dsize would just be an arity error one node later.
			self.extra_types = list(extra_types or [])

		def get_io_type(self):
			return ",".join([self.io_type, *self.extra_types])


class TupleLiteralInput(LiteralOrTyped):
	"""A CV_TUPLE link OR the "(30, 12)" literal, on a plain STRING widget.

	For a node that already HAD a literal widget and only needs the socket
	widened ('CV Split Point'): keeping the core string widget keeps the saved
	value, the widget count and the free-text entry exactly as they were.
	"""

	COMPOSITE_TYPE = "CV_TUPLE"


# --- parsing -----------------------------------------------------------------

# core's BoundingBox dict keys, and the component names this pack uses for the
# same quantities (a cv2 Rect is (x, y, w, h))
_BBOX_ALIASES = {"w": "width", "h": "height", "width": "w", "height": "h"}


def _sequence(value, pname):
	"""Anything that could reach a CV_TUPLE input -> a flat list of parts."""
	if value is None:
		return None
	if isinstance(value, np.ndarray):
		return list(value.reshape(-1))
	if isinstance(value, dict):
		return value
	if isinstance(value, str):
		text = value.strip()
		if not text:
			return None
		try:
			parsed = literal_eval(text)
		except (ValueError, SyntaxError) as e:
			raise ValueError(
				f'Parameter "{pname}" is not a valid composite literal: '
				f'"{value}". Examples: (3, 3), [640, 480], 5. ({e})'
			) from e
		return _sequence(parsed, pname)
	if isinstance(value, (list, tuple)):
		return list(value)
	return [value]  # a bare number broadcasts (see parse)


def parse(value, components, pname="value"):
	"""Resolve any accepted spelling of a composite into a list of components.

	Accepted: a list/tuple, a legacy "(w, h)" literal string, a 1-D ndarray, a
	core BoundingBox-style dict, a bare number (broadcast over every component -
	`5` is a 5x5 kernel), or None / "" (every component at its schema default).

	Dropdown components come back as their STRING name; the caller resolves them
	(lowlevel.py hands them to the EnumGroup). Every other component is coerced
	to int or float per the spec, so a float typed into a Size cannot reach cv2.
	"""
	parts = _sequence(value, pname)
	if parts is None:
		parts = defaults(components)
	elif isinstance(parts, dict):
		picked = []
		for c in components:
			name = c["name"]
			if name in parts:
				picked.append(parts[name])
			elif _BBOX_ALIASES.get(name) in parts:
				picked.append(parts[_BBOX_ALIASES[name]])
			else:
				raise ValueError(
					f'Parameter "{pname}" got a dict without a "{name}" key: '
					f"{sorted(parts)}. Expected keys "
					f"{[c['name'] for c in components]} (or a bounding box's "
					f"x/y/width/height)."
				)
		parts = picked
	elif len(parts) == 1 and len(components) > 1:
		parts = list(parts) * len(components)   # bare number broadcasts
	if len(parts) != len(components):
		raise ValueError(
			f'Parameter "{pname}" needs {len(components)} components '
			f"({', '.join(c['name'] for c in components)}), got {len(parts)}: "
			f"{parts!r}."
		)
	out = []
	for c, v in zip(components, parts):
		if c["options"] is not None:
			out.append(v)
		elif c["is_float"]:
			out.append(float(v))
		else:
			out.append(int(v))
	return out


def format_literal(values):
	"""The '(a, b)' spelling the STRING-typed composite params still take."""
	return "(" + ", ".join(repr(v) for v in values) + ")"


# --- RotatedRect and Moments -------------------------------------------------

def parse_rotated_rect(value, pname="box"):
	"""Any accepted spelling of a cv2 RotatedRect -> ((cx, cy), (w, h), angle).

	Accepts the nested form cv2 itself returns, a flat 5-sequence (what a
	'CV Tuple'-style splitter or a hand-typed literal is likely to produce), a
	literal string of either, or a 1-D/2-D ndarray. Always returns the NESTED
	tuple, because that is the only spelling cv2 takes back: a flat 5-sequence
	raises "Can't parse 'box' as RotatedRect. Expected sequence length 3, got 5".
	"""
	if isinstance(value, str):
		text = value.strip()
		if not text:
			raise ValueError(
				f'Parameter "{pname}" is required: a rotated rectangle like '
				f'"((30, 20), (10, 4), 15)" (center, size, angle in degrees).')
		try:
			value = literal_eval(text)
		except (ValueError, SyntaxError) as e:
			raise ValueError(
				f'Parameter "{pname}" is not a valid rotated-rectangle literal: '
				f'"{text}". Expected ((cx, cy), (w, h), angle). ({e})') from e
	if isinstance(value, np.ndarray):
		value = value.reshape(-1).tolist()
	if isinstance(value, dict):  # a BOUNDING_BOX is an AXIS-ALIGNED rect
		x, y, w, h = bbox_to_rect(value, pname)
		return ((x + w / 2.0, y + h / 2.0), (float(w), float(h)), 0.0)
	if not isinstance(value, (list, tuple)):
		raise ValueError(
			f'Parameter "{pname}" needs a rotated rectangle '
			f"((cx, cy), (w, h), angle), got {value!r}.")
	parts = list(value)
	if len(parts) == 3 and all(isinstance(p, (list, tuple, np.ndarray))
	                           for p in parts[:2]):
		(cx, cy), (w, h), angle = parts[0], parts[1], parts[2]
	elif len(parts) == 5:
		cx, cy, w, h, angle = parts
	else:
		raise ValueError(
			f'Parameter "{pname}" needs ((cx, cy), (w, h), angle) or the flat '
			f"(cx, cy, w, h, angle), got {len(parts)} component(s): {parts!r}.")
	return ((float(cx), float(cy)), (float(w), float(h)), float(angle))


# cv2.moments' 24 keys, in the order the binding builds them
MOMENT_KEYS = ("m00 m10 m01 m20 m11 m02 m30 m21 m12 m03 "
               "mu20 mu11 mu02 mu30 mu21 mu12 mu03 "
               "nu20 nu11 nu02 nu30 nu21 nu12 nu03").split()


def parse_moments(value, pname="m"):
	"""Any accepted spelling of cv2.moments' result -> the 24-key dict.

	A LIST is rejected on purpose. `cv2.HuMoments` reads the mapping by KEY, so
	the same 24 numbers as a sequence come back as seven zeros - a wrong answer
	with no error, which is exactly what a "just flatten it" bridge would give.
	"""
	if isinstance(value, str):
		text = value.strip()
		if not text:
			raise ValueError(
				f'Parameter "{pname}" is required: the moments dict from '
				f"'cv2.moments' (wire it in, or paste its "
				f"{{'m00': ..., 'm10': ...}} literal).")
		try:
			value = literal_eval(text)
		except (ValueError, SyntaxError) as e:
			raise ValueError(
				f'Parameter "{pname}" is not a valid moments literal: "{text}". '
				f"Expected {{'m00': ..., 'm10': ...}}. ({e})") from e
	if not isinstance(value, dict):
		raise ValueError(
			f'Parameter "{pname}" needs the NAMED moments dict from '
			f"'cv2.moments', not {type(value).__name__}. cv2 reads the moments "
			f"by key (m00, m10, ...), and a plain sequence of the same numbers "
			f"silently yields zeros.")
	missing = [k for k in MOMENT_KEYS if k not in value]
	if missing:
		raise ValueError(
			f'Parameter "{pname}" is missing {len(missing)} moment key(s) '
			f"(first: {missing[0]}). Expected all of {', '.join(MOMENT_KEYS)}.")
	return {k: float(value[k]) for k in MOMENT_KEYS}


# --- the core BOUNDING_BOX bridge (cv2 Rect) ---------------------------------

def bbox_to_rect(value, pname="rect"):
	"""Core BOUNDING_BOX -> a cv2 Rect tuple (x, y, w, h).

	`io.BoundingBox` is ONE {x, y, width, height} dict, but this pack's own
	BOUNDING_BOX outputs carry the core per-frame convention: a list with one
	GROUP of boxes per frame (boxpolicy.nest). Both land on the same io_type, so
	a detector's output really can arrive here - flatten it and insist on
	exactly one box rather than indexing blindly.
	"""
	if isinstance(value, (str, bytes)) or value is None:
		return tuple(parse(value, RECT_COMPONENTS, pname))
	boxes = value
	while isinstance(boxes, (list, tuple)) and len(boxes) and \
			isinstance(boxes[0], (list, tuple)):
		boxes = [b for group in boxes for b in group]
	if isinstance(boxes, (list, tuple)):
		if len(boxes) != 1:
			raise ValueError(
				f'Parameter "{pname}" takes exactly ONE bounding box, got '
				f"{len(boxes)}. Select one (e.g. 'CV Index Batch' / a filter) "
				f"before wiring it into a cv2 rect."
			)
		boxes = boxes[0]
	return tuple(parse(boxes, RECT_COMPONENTS, pname))


def rect_to_bbox(rect):
	"""A cv2 Rect (x, y, w, h) -> the core BOUNDING_BOX dict."""
	x, y, w, h = (int(v) for v in rect)
	return {"x": x, "y": y, "width": w, "height": h}


def _is_bbox(value):
	"""Is this a core BOUNDING_BOX value (a box dict, or groups of them)?"""
	while isinstance(value, (list, tuple)) and value:
		value = value[0]
	return isinstance(value, dict)


RECT_COMPONENTS = [
	component("x", 0, -INT32_MAX - 1, INT32_MAX, False,
		"Rectangle top-left corner X in pixels."),
	component("y", 0, -INT32_MAX - 1, INT32_MAX, False,
		"Rectangle top-left corner Y in pixels."),
	component("w", 0, 0, INT32_MAX, False, "Rectangle width in pixels (>= 0)."),
	component("h", 0, 0, INT32_MAX, False, "Rectangle height in pixels (>= 0)."),
]


# --- nodes -------------------------------------------------------------------

_GENERIC = [
	component("a", 0.0, -1e38, 1e38, True, "First component (width / x / ...)."),
	component("b", 0.0, -1e38, 1e38, True, "Second component (height / y / ...)."),
	component("c", 0.0, -1e38, 1e38, True, "Third component (z / width / ...)."),
	component("d", 0.0, -1e38, 1e38, True, "Fourth component (height / ...)."),
]


class Tuple(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Tuple",
			display_name="CV Tuple",
			category=CATEGORY,
			description="Authors one composite cv2 value - a Size (width, "
			            "height), a Point (x, y), a 3-D point, a Rect - from "
			            "typed number widgets, and emits it as ONE wire. Wire "
			            "'tuple' into any cv2 node's composite input (dsize, "
			            "ksize, center, patternSize, winSize, ...): both "
			            "components travel together, so a half-connected size "
			            "is impossible. One instance can drive many nodes. "
			            "'array' is the same numbers as an NPARRAY vector, and "
			            "'literal' is the '(a, b)' string the few params that "
			            "are still free-text literals take (and what a subgraph "
			            "boundary can carry). This is also the way to set a "
			            "composite input on a SUBGRAPH instance: a promoted "
			            "input is a bare socket with no widget behind it.",
			inputs=[
				io.Int.Input("count", default=2, min=2, max=4,
					tooltip="How many components to emit: 2 for a Size / 2-D "
					        "point, 3 for a 3-D point or an iteration-stop "
					        "tuple, 4 for a Rect / Scalar."),
				io.Combo.Input("dtype", options=["int", "float"], default="int",
					tooltip="Component type. 'int' for pixel sizes, kernel "
					        "sizes and pixel coordinates (cv2 rejects a float "
					        "Size); 'float' for subpixel coordinates and world "
					        "units."),
				Number.Input("a",
					tooltip="First component - width / x. The socket takes an INT "
					        "or a FLOAT link, so an image width and a subpixel "
					        "coordinate both wire in directly."),
				Number.Input("b",
					tooltip="Second component - height / y. Takes an INT or a "
					        "FLOAT link."),
				Number.Input("c",
					tooltip="Third component (used when count >= 3) - z / width.",
					optional=True, advanced=True),
				Number.Input("d",
					tooltip="Fourth component (used when count = 4) - height.",
					optional=True, advanced=True),
			],
			outputs=[
				CvTuple.Output(display_name="tuple",
					tooltip="The whole composite as one value - wire it into a "
					        "cv2 node's composite input."),
				NPArray.Output(display_name="array",
					tooltip="The same components as a 1-D NPARRAY vector, for "
					        "the sockets that take an array (cv2.inRange "
					        "bounds, arithmetic operands)."),
				io.String.Output(display_name="literal",
					tooltip="The '(a, b)' literal string, for the composite "
					        "params that are still free-text (Scalar, "
					        "RotatedRect, Moments) and for STRING subgraph "
					        "boundaries."),
			],
		)

	@classmethod
	def execute(cls, count, dtype, a, b, c=0.0, d=0.0) -> io.NodeOutput:
		cast = float if dtype == "float" else int
		values = [cast(v) for v in (a, b, c, d)[:int(count)]]
		arr = np.asarray(values, dtype=np.float32 if dtype == "float" else np.int32)
		return io.NodeOutput(values, arr, format_literal(values))


class SplitTuple(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_SplitTuple",
			display_name="CV Split Tuple",
			category=CATEGORY,
			description="The reverse of 'CV Tuple': takes one composite value "
			            "and emits its components separately, so a size or a "
			            "point can drive plain INT/FLOAT inputs (a Math "
			            "Expression, a crop, a loop bound). Components past the "
			            "value's own length come out as 0.",
			inputs=[
				# force_input: a SOCKET, never a widget. Every CV_TUPLE widget
				# renders one field per component (web/cv_tuple.js), and on a
				# SPLITTER those fields are just 'CV Tuple' again - you would be
				# typing a value in order to take it apart. forceInput makes
				# litegraphService.addInputSocket skip the widget path entirely,
				# so the node is exactly what it claims to be: one composite in,
				# its components out.
				CvTuple.Input("tuple", _GENERIC, force_input=True,
					extra_types=["BOUNDING_BOX"],
					tooltip="Composite value to split - wire it from 'CV Tuple', "
					        "from any cv2 node's point / size output, or from a "
					        "BOUNDING_BOX (a cv2 Rect: x, y, width, height)."),
				io.Combo.Input("dtype", options=["int", "float"], default="int",
					tooltip="Output type: 'int' truncates (pixel indices), "
					        "'float' preserves fractions (subpixel)."),
			],
			outputs=[
				io.AnyType.Output(display_name="a", tooltip="First component."),
				io.AnyType.Output(display_name="b", tooltip="Second component."),
				io.AnyType.Output(display_name="c",
					tooltip="Third component (0 if the value has only two)."),
				io.AnyType.Output(display_name="d",
					tooltip="Fourth component (0 if the value is shorter)."),
				io.Int.Output(display_name="count",
					tooltip="How many components the value actually carried."),
			],
		)

	@classmethod
	def execute(cls, tuple, dtype) -> io.NodeOutput:  # noqa: A002 (socket name)
		if _is_bbox(tuple):
			# a BOUNDING_BOX arrives nested one GROUP per frame (boxpolicy.nest),
			# so it is unwrapped by the same bridge every cv2 rect param uses
			parts = list(bbox_to_rect(tuple, "tuple"))
		else:
			parts = _sequence(tuple, "tuple")
		if parts is None:
			parts = []
		elif isinstance(parts, dict):
			parts = [parts.get(k, 0) for k in ("x", "y", "width", "height")]
		cast = float if dtype == "float" else int
		values = [cast(v) for v in parts]
		n = len(values)
		values = (values + [cast(0)] * 4)[:4]
		return io.NodeOutput(*values, n)


NODES = [Tuple, SplitTuple]
