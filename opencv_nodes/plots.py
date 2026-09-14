# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Data-visualization nodes: raster cv2.plot renders and interactive web charts.

`Preview CV Array` answers "what does this array look like as pixels". That is
the wrong question for COLLECTED DATA - a similarity score per detected face, a
histogram, a per-class metric - where the array is a handful of numbers whose
meaning lives in their VALUES and in what each index refers to. Rendered as
pixels a 5x1 score matrix is a five-pixel image; `color_scale` puts a legend
next to it but cannot label an index, cannot show the value, and min-max
normalization actively destroys the absolute scale that made the numbers mean
something (workflow 70: cosine similarities against SFace's 0.363 threshold).

Two layers, both generic and both DATA-IN / PICTURE-OUT:

* `CV Plot 2D (cv2.plot)` wraps `cv2.plot.Plot2d`, the one plotting API OpenCV
  itself ships. It renders a single series to a BGR image, so the plot is an
  IMAGE like any other - chain it into Save Image, composite it, diff it.
  Plot2d is deliberately thin (one series, no labels, no legend), which is
  exactly why the second layer exists.

* `CV Chart Series` / `CV Chart Scatter` / `CV Chart Matrix` build an Apache
  ECharts option IN PYTHON and hand it to `web/cv_charts.js`, which only
  instantiates the chart. Everything that decides what the picture says -
  labels, axis names, palettes, reference lines, value formatting - is
  server-side and comes back out as the `spec` STRING, so it is inspectable
  (wire it into 'Preview as Text') and testable headlessly. The JS half holds
  no chart logic to get out of sync.

Labels come in as ONE free-form string (JSON list, or ';' / newline / ','
separated), which is deliberately the same shape workflow 12 already uses to
label a sheet of images: a `PrimitiveStringMultiline` of ';'-separated names
feeds `RegexSplitDataList` -> `CV Draw Text` for the tiles AND, whole, the
chart's label input. One source of names, so bar i and tile i provably carry
the same label.

ECharts 5.6.0 (Apache-2.0) is vendored at `web/lib/echarts.esm.min.js`, imported
relative like the three.js viewer does; see `web/lib/LICENSE.echarts.txt`.
"""

import json
import math

import cv2
import numpy as np

from comfy_api.latest import io

from . import imageutils
from .convert import NPArray
from .features import parse_color

CATEGORY = "image/CV/plots"

# A chart is a picture of a few hundred numbers. Past these counts the request
# is a misunderstanding (charting an image), not a big dataset - say so instead
# of freezing a browser tab with a 1-megapixel option object.
MAX_SERIES_VALUES = 200_000
MAX_MATRIX_CELLS = 40_000

# Continuous ramps, as the stops the ECharts visualMap interpolates between.
# 'viridis' first so charts match `Preview CV Array`'s heatmap render.
CONTINUOUS_PALETTES = {
	"viridis": ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
	"magma": ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"],
	"plasma": ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"],
	"inferno": ["#000004", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"],
	"turbo": ["#30123b", "#4145ab", "#4675ed", "#39a2fc", "#1bcfd4", "#62fbb2",
	          "#a4fc3b", "#d1e834", "#f9ba38", "#fb7e21", "#e4460a", "#7a0403"],
	"grayscale": ["#000000", "#ffffff"],
	"coolwarm (diverging)": ["#3b4cc0", "#7396f5", "#b4c8f0", "#dcdcdc",
	                         "#f2c0a6", "#ee8468", "#b40426"],
}

# Discrete series/class colors.
CATEGORICAL_PALETTES = {
	"bright": ["#4e9bf5", "#f2994a", "#2ecc71", "#e74c3c", "#9b59b6",
	           "#16a085", "#f1c40f", "#e056a0", "#95a5a6", "#5d7ce0"],
	# same table the drawing nodes step through (imageutils.categorical_bgr)
	"colorblind-safe (Okabe-Ito)": list(imageutils.OKABE_ITO_HEX),
	"viridis (sampled)": None,   # sampled from the continuous ramp on demand
	"grayscale": None,
}

THEMES = {
	"dark": {"bg": "#1e1e21", "text": "#c8c8cc", "axis": "#6b6b73",
	         "split": "#37373d", "label": "#e6e6ea"},
	"light": {"bg": "#ffffff", "text": "#303036", "axis": "#9a9aa2",
	          "split": "#e2e2e8", "label": "#1a1a1e"},
}


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(text):
	text = text.lstrip("#")
	return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
	return "#%02x%02x%02x" % tuple(int(round(max(0, min(255, c)))) for c in rgb)


def sample_palette(name, count):
	"""`count` hex colors from a named palette, cycling or interpolating.

	Categorical palettes repeat once exhausted (10 series of one color each is
	still readable when the legend names them); the ramp-derived ones are
	sampled evenly so N classes spread across the whole ramp.
	"""
	count = max(int(count), 1)
	fixed = CATEGORICAL_PALETTES.get(name)
	if fixed:
		return [fixed[i % len(fixed)] for i in range(count)]
	stops = CONTINUOUS_PALETTES.get(
		"grayscale" if name == "grayscale" else
		name.replace(" (sampled)", ""), CONTINUOUS_PALETTES["viridis"])
	if count == 1:
		return [_interp_stops(stops, 0.5)]
	return [_interp_stops(stops, i / (count - 1)) for i in range(count)]


def _interp_stops(stops, t):
	"""Color at position `t` (0-1) along a list of hex stops."""
	t = max(0.0, min(1.0, float(t)))
	pos = t * (len(stops) - 1)
	low = int(math.floor(pos))
	high = min(low + 1, len(stops) - 1)
	frac = pos - low
	a, b = _hex_to_rgb(stops[low]), _hex_to_rgb(stops[high])
	return _rgb_to_hex([a[i] + (b[i] - a[i]) * frac for i in range(3)])


def split_labels(text):
	"""Free-form label list -> list[str].

	Accepts a JSON array, or ';' / newline / ',' separated text - in that
	order of preference, so a ';'-separated `PrimitiveStringMultiline` (the
	workflow 12 label-sheet pattern) works whether or not it also wraps lines.
	Empty text gives an empty list; blank entries are kept so index i of the
	label list always lines up with index i of the data.
	"""
	text = "" if text is None else str(text)
	if not text.strip():
		return []
	stripped = text.strip()
	if stripped.startswith("["):
		try:
			data = json.loads(stripped)
			if isinstance(data, list):
				return [str(v) for v in data]
		except json.JSONDecodeError:
			pass   # not JSON after all - fall through to the separators
	for sep in (";", "\n", ","):
		if sep in text:
			return [part.strip() for part in text.split(sep)]
	return [stripped]


def labels_for(text, count, prefix="#"):
	"""Exactly `count` labels: the given ones, padded/truncated by index.

	Never raises on a count mismatch - a chart whose label list is one short is
	still a readable chart, and halting a workflow over a label is exactly the
	kind of failure intolerance this pack does not accept.
	"""
	labels = split_labels(text)
	out = []
	for i in range(int(count)):
		name = labels[i].strip() if i < len(labels) and labels[i].strip() else f"{prefix}{i}"
		out.append(name)
	return out


def parse_reference_lines(text):
	"""'0.363=SFace threshold; 0.5' -> [(0.363, 'SFace threshold'), (0.5, '0.5')].

	Entries that are not a number are skipped rather than raising: a reference
	line is annotation, and a typo must not cost the whole chart.
	"""
	lines = []
	for entry in split_labels(text):
		if not entry.strip():
			continue
		value, _, label = entry.partition("=")
		try:
			number = float(value.strip())
		except ValueError:
			continue
		if not math.isfinite(number):
			continue
		lines.append((number, label.strip() or _fmt_number(number)))
	return lines


def _fmt_number(value, decimals=3):
	if not math.isfinite(value):
		return "n/a"
	if float(value).is_integer() and abs(value) < 1e15:
		return str(int(value))
	return f"{value:.{decimals}g}"


def jsonable(value):
	"""float -> JSON-safe (NaN/Inf become null, which ECharts draws as a gap)."""
	number = float(value)
	return number if math.isfinite(number) else None


def as_columns(array, name="values"):
	"""Any array-ish input -> (N, K) float64: N samples, K series (columns).

	A 1-D array, an (N, 1) column and a (1, N) row all mean "one series of N
	values" - the three shapes cv2 nodes hand out for the same thing
	(`calcHist` gives (bins, 1), a score row gives (1, N)). Extra size-1 axes
	are squeezed first so an (N, 1, 2) point block or a (1, N, 1) histogram
	need no reshape node in front.
	"""
	data = np.asarray(array)
	if data.dtype == object:
		raise ValueError(f"{name} must be numeric, got an object array.")
	data = np.asarray(data, dtype=np.float64)
	if data.ndim > 2:
		data = data.reshape([d for d in data.shape if d != 1] or [1])
	if data.ndim > 2:
		raise ValueError(
			f"Cannot chart {name} of shape {np.asarray(array).shape}: expected 1-D "
			"values, or 2-D (samples x series). Use 'Preview CV Array' for "
			"image-shaped data.")
	if data.ndim == 0:
		data = data.reshape(1, 1)
	elif data.ndim == 1:
		data = data.reshape(-1, 1)
	elif data.shape[0] == 1 and data.shape[1] > 1:
		data = data.T   # a row vector is one series, not N series of 1 point
	if data.size > MAX_SERIES_VALUES:
		raise ValueError(
			f"{name} has {data.size} values, more than the {MAX_SERIES_VALUES} a "
			"chart can carry. Chart a summary (histogram, reduction) or preview "
			"the raw array with 'Preview CV Array'.")
	return data


def theme_colors(theme):
	return THEMES.get(str(theme), THEMES["dark"])


# `containLabel: True` makes ECharts grow the plot area INWARD until the tick
# labels fit, so a grid margin is pure outside padding, never label room: a big
# `left` does not "make space for the labels", it only pushes the whole chart
# right and leaves a blank column that is as wide as the guess was wrong.
# ECharts unions the axisLabel rects only, so the axis NAME is the one thing
# containLabel does NOT cover - a named vertical axis gets its own band here,
# and the matching nameGap (which depends on the rendered label width) is
# measured on the client in web/cv_charts.js.
GRID_MARGIN = 12
AXIS_NAME_BAND = 26


def base_option(title, theme, *, legend=False, grid_left=GRID_MARGIN):
	"""The frame every chart shares: title, theme colors, toolbox, grid."""
	colors = theme_colors(theme)
	option = {
		"backgroundColor": colors["bg"],
		"animation": False,
		"textStyle": {"color": colors["text"], "fontSize": 11},
		"grid": {"left": grid_left, "right": 24, "top": 48 if title else 28,
		         "bottom": 44, "containLabel": True},
		"toolbox": {
			"right": 12, "top": 6, "itemSize": 13,
			"iconStyle": {"borderColor": colors["text"]},
			"feature": {
				"dataZoom": {"yAxisIndex": "none", "title": {
					"zoom": "box zoom", "back": "undo zoom"}},
				"restore": {"title": "reset"},
				"saveAsImage": {"title": "save PNG", "name": "cv_chart",
				                "backgroundColor": colors["bg"], "pixelRatio": 2},
			},
		},
	}
	if title:
		option["title"] = {"text": title, "left": "center", "top": 6,
		                   "textStyle": {"color": colors["label"], "fontSize": 13,
		                                 "fontWeight": "normal"}}
	if legend:
		option["legend"] = {"bottom": 4, "textStyle": {"color": colors["text"]},
		                    "itemHeight": 8, "itemWidth": 14}
		option["grid"]["bottom"] = 58
	return option


def axis_style(theme, name="", *, category=False, data=None, invert=False):
	colors = theme_colors(theme)
	axis = {
		"type": "category" if category else "value",
		"name": name, "nameLocation": "middle", "nameGap": 26,
		"nameTextStyle": {"color": colors["text"]},
		"axisLine": {"lineStyle": {"color": colors["axis"]}},
		"axisTick": {"lineStyle": {"color": colors["axis"]}},
		"axisLabel": {"color": colors["text"], "hideOverlap": True},
		"splitLine": {"lineStyle": {"color": colors["split"], "type": "dashed"}},
	}
	if category:
		axis["data"] = list(data or [])
		axis["splitLine"] = {"show": False}
	if invert:
		axis["inverse"] = True
	return axis


def apply_range(axis, mode, low, high):
	"""'auto' / 'start at zero' / 'manual' -> axis min/max, in place."""
	if mode == "start at zero":
		axis["min"] = 0
	elif mode == "manual":
		axis["min"] = float(low)
		axis["max"] = float(high)
	return axis


def mark_line(references, theme, horizontal=False):
	"""markLine spec for reference values (thresholds, targets, means).

	A markLine is anchored to the axis it is a value ON, not to a direction:
	`{"yAxis": v}` always draws a HORIZONTAL line. With horizontal bars the
	value axis is the x axis, so the same key would draw the threshold across
	the categories instead of across the values - it must become `{"xAxis": v}`.
	"""
	if not references:
		return None
	colors = theme_colors(theme)
	axis_key = "xAxis" if horizontal else "yAxis"
	return {
		"silent": True, "symbol": "none",
		"lineStyle": {"color": colors["label"], "type": "dashed", "width": 1},
		"label": {"color": colors["label"],
		          "position": "insideEndTop" if horizontal else "insideStartTop",
		          "formatter": "{b}", "fontSize": 10},
		"data": [{"name": label, axis_key: value} for value, label in references],
	}


def reserve_axis_name_band(option):
	"""Widen the left margin only when the vertical axis actually has a name.

	Labels need no margin (`containLabel`), the rotated name does - it is drawn
	OUTSIDE the label band, so without this it lands under the toolbox/edge and
	is clipped. Charts with no y name keep the bare margin.
	"""
	grid = option.get("grid")
	if not grid or not isinstance(grid.get("left"), (int, float)):
		return option
	if str(option.get("yAxis", {}).get("name") or "").strip():
		grid["left"] = max(grid["left"], GRID_MARGIN + AXIS_NAME_BAND)
	return option


def chart_output(option, height, spec_name, decimals=3):
	"""NodeOutput for an interactive chart: the ui payload + the spec STRING.

	`decimals` is metadata, not part of the ECharts option: value formatters
	are JS functions and cannot survive JSON, so the frontend builds them from
	this precision hint. Keeping it OUT of `option` leaves the spec a valid,
	directly-loadable ECharts option.
	"""
	reserve_axis_name_band(option)
	payload = {"option": option, "height": int(height), "name": spec_name,
	           "decimals": int(decimals)}
	spec = json.dumps(payload, allow_nan=False, sort_keys=False)
	return io.NodeOutput(spec, ui={"cv_chart": [payload]})


# ---------------------------------------------------------------------------
# cv2.plot - the raster plot OpenCV ships
# ---------------------------------------------------------------------------

class Plot2D(io.ComfyNode):
	"""cv2.plot.Plot2d, owned inside one execute().

	Same containment rule as contrib.py: the Plot2d object is created,
	configured, rendered and dropped here, so nothing stateful travels through
	the graph (and a cached re-execution cannot mutate a live C++ object).
	"""

	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_Plot2D",
			display_name="CV Plot 2D (cv2.plot)",
			category=CATEGORY,
			description="Renders one series of numbers as a line/point plot "
			            "IMAGE using cv2.plot.Plot2d - OpenCV's own plotting "
			            "module. Because the result is an ordinary IMAGE it "
			            "saves, composites and diffs like any other: this is "
			            "the node for putting a curve INTO a picture (a report "
			            "sheet, a video frame, a saved PNG). Feed a 1-D array "
			            "(plotted against its index) or an (N, 2) array of "
			            "x, y pairs; x_values overrides the x axis. Plot2d "
			            "draws ONE series with no labels or legend - for "
			            "several series, category names, tooltips or a "
			            "threshold line use 'CV Chart Series' instead. Empty "
			            "input renders an empty plot rather than failing.",
			inputs=[
				NPArray.Input("values",
					tooltip="The numbers to plot: 1-D (or (N, 1) / (1, N)) is "
					        "plotted against the sample index; an (N, 2) array "
					        "is read as x, y pairs. Converted to float64, which "
					        "is the only dtype Plot2d accepts. Larger values are "
					        "drawn UPWARDS (see y_down)."),
				io.Int.Input("width", default=640, min=400, max=4096,
					tooltip="Plot width in pixels. cv2.plot CLAMPS anything "
					        "below 400 up to 400 - that is the module, not this "
					        "node."),
				io.Int.Input("height", default=400, min=300, max=4096,
					tooltip="Plot height in pixels; clamped up to 300 minimum "
					        "by cv2.plot itself."),
				io.Combo.Input("draw", options=["line", "points only"],
					default="line",
					tooltip="'line' connects consecutive samples; 'points only' "
					        "marks them without the connecting segments - the "
					        "honest choice when the samples are independent "
					        "measurements rather than a signal."),
				NPArray.Input("x_values", optional=True,
					tooltip="Optional x coordinate per sample (same length as "
					        "values). Without it samples are plotted against "
					        "their index. Ignored when values is (N, 2)."),
				io.Int.Input("line_width", default=1, min=1, max=16,
					optional=True, advanced=True,
					tooltip="Stroke width of the plotted line in pixels."),
				io.Boolean.Input("show_grid", default=True, optional=True,
					advanced=True,
					tooltip="Draw the background grid."),
				io.Int.Input("grid_lines", default=10, min=1, max=100,
					optional=True, advanced=True,
					tooltip="Number of grid divisions when the grid is shown."),
				io.Boolean.Input("show_text", default=True, optional=True,
					advanced=True,
					tooltip="Draw the axis value texts (the min/max readouts "
					        "cv2.plot prints along the axes)."),
				io.Combo.Input("value_range",
					options=["auto (data range)", "manual"],
					default="auto (data range)", optional=True, advanced=True,
					tooltip="'auto' lets Plot2d fit the axes to the data - "
					        "which hides how big the values are in absolute "
					        "terms. 'manual' pins the axes so two plots of "
					        "different runs are comparable."),
				io.Float.Input("min_x", default=0.0, min=-1e9, max=1e9,
					optional=True, advanced=True,
					tooltip="Left edge of the x axis ('manual' range only)."),
				io.Float.Input("max_x", default=1.0, min=-1e9, max=1e9,
					optional=True, advanced=True,
					tooltip="Right edge of the x axis ('manual' range only)."),
				io.Float.Input("min_y", default=0.0, min=-1e9, max=1e9,
					optional=True, advanced=True,
					tooltip="Bottom of the y axis ('manual' range only)."),
				io.Float.Input("max_y", default=1.0, min=-1e9, max=1e9,
					optional=True, advanced=True,
					tooltip="Top of the y axis ('manual' range only)."),
				io.String.Input("line_color", default="(0, 255, 0)",
					optional=True, advanced=True,
					tooltip="Series color as a BGR tuple, e.g. '(0, 255, 0)'."),
				io.String.Input("background_color", default="(30, 30, 30)",
					optional=True, advanced=True,
					tooltip="Plot background color, BGR."),
				io.String.Input("axis_color", default="(200, 200, 200)",
					optional=True, advanced=True,
					tooltip="Axis line color, BGR."),
				io.String.Input("grid_color", default="(70, 70, 70)",
					optional=True, advanced=True,
					tooltip="Grid line color, BGR."),
				io.String.Input("text_color", default="(230, 230, 230)",
					optional=True, advanced=True,
					tooltip="Axis text color, BGR."),
				io.Int.Input("print_point_index", default=-1, min=-1,
					max=1_000_000, optional=True, advanced=True,
					tooltip="Prints the value of ONE sample onto the plot: the "
					        "index to annotate, or -1 for none. Useful to call "
					        "out the winner of a comparison."),
				io.Boolean.Input("y_down", default=False,
					optional=True, advanced=True,
					tooltip="Flips the y axis so larger values go DOWN, matching "
					        "image coordinates (or a cost/error where lower is "
					        "better). Off - the default - draws larger values "
					        "UP, like 'CV Chart Series' and every other plot. "
					        "Note this is the OPPOSITE of raw cv2.plot, whose "
					        "own default draws downward; the node corrects it."),
			],
			outputs=[
				io.Image.Output(display_name="plot",
					tooltip="The rendered plot as a normal IMAGE (BGR converted "
					        "to ComfyUI RGB), ready for Save Image, "
					        "compositing or stacking next to the data it "
					        "describes."),
			],
		)

	@classmethod
	def execute(cls, values, width, height, draw, x_values=None, line_width=1,
	            show_grid=True, grid_lines=10, show_text=True,
	            value_range="auto (data range)", min_x=0.0, max_x=1.0,
	            min_y=0.0, max_y=1.0, line_color="(0, 255, 0)",
	            background_color="(30, 30, 30)", axis_color="(200, 200, 200)",
	            grid_color="(70, 70, 70)", text_color="(230, 230, 230)",
	            print_point_index=-1, y_down=False) -> io.NodeOutput:
		if not hasattr(cv2, "plot") or not hasattr(cv2.plot, "Plot2d_create"):
			raise RuntimeError(
				f"cv2.plot is not available in this OpenCV build ({cv2.__version__}). "
				"It ships with opencv-contrib-python(-headless); see "
				"tools/repair_opencv_contrib.py.")
		data = np.asarray(values)
		xs = None
		if data.ndim == 2 and 2 in data.shape and x_values is None and data.size > 2:
			# an (N, 2) / (2, N) block is x, y pairs
			pairs = data if data.shape[1] == 2 else data.T
			xs = np.ascontiguousarray(pairs[:, 0], dtype=np.float64).reshape(-1, 1)
			ys = np.ascontiguousarray(pairs[:, 1], dtype=np.float64).reshape(-1, 1)
		else:
			ys = as_columns(data, "values")[:, :1]
			ys = np.ascontiguousarray(ys, dtype=np.float64)
			if x_values is not None:
				xs = as_columns(x_values, "x_values")[:, :1]
				xs = np.ascontiguousarray(xs, dtype=np.float64)
				if len(xs) != len(ys):
					raise ValueError(
						f"x_values has {len(xs)} samples but values has {len(ys)}. "
						"cv2.plot pairs them by index, so the lengths must match.")
		bg = parse_color(background_color, 3)
		if ys.size == 0:
			# Plot2d throws on an empty Mat; an empty plot is the failure-tolerant
			# answer - the workflow keeps running and shows a blank frame.
			frame = np.full((max(int(height), 300), max(int(width), 400), 3),
			                bg, dtype=np.uint8)
			return io.NodeOutput(imageutils.bgr_to_image_batch([frame]))
		# cv2.plot cannot draw NaN/Inf and silently produces garbage axes; carry
		# them at the series mean so the finite samples still read correctly.
		finite = ys[np.isfinite(ys)]
		ys = np.nan_to_num(ys, nan=float(finite.mean()) if finite.size else 0.0,
		                   posinf=float(finite.max()) if finite.size else 0.0,
		                   neginf=float(finite.min()) if finite.size else 0.0)
		plot = (cv2.plot.Plot2d_create(xs, ys) if xs is not None
		        else cv2.plot.Plot2d_create(ys))
		plot.setPlotSize(int(width), int(height))
		plot.setNeedPlotLine(draw == "line")
		plot.setPlotLineWidth(int(line_width))
		plot.setShowGrid(bool(show_grid))
		plot.setGridLinesNumber(int(grid_lines))
		plot.setShowText(bool(show_text))
		# cv2.plot's UNINVERTED orientation puts larger values at the BOTTOM -
		# the opposite of every other plot, this pack's chart nodes included.
		# So the node's y-up default is cv2's "inverted" one; the flag is
		# negated here, not passed through.
		plot.setInvertOrientation(not bool(y_down))
		plot.setPlotLineColor(parse_color(line_color, 3))
		plot.setPlotBackgroundColor(bg)
		plot.setPlotAxisColor(parse_color(axis_color, 3))
		plot.setPlotGridColor(parse_color(grid_color, 3))
		plot.setPlotTextColor(parse_color(text_color, 3))
		if value_range == "manual":
			plot.setMinX(float(min_x))
			plot.setMaxX(float(max_x))
			plot.setMinY(float(min_y))
			plot.setMaxY(float(max_y))
		if int(print_point_index) >= 0:
			plot.setPointIdxToPrint(int(print_point_index))
		frame = np.ascontiguousarray(plot.render())
		if frame.ndim == 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		return io.NodeOutput(imageutils.bgr_to_image_batch([frame]))


# ---------------------------------------------------------------------------
# interactive web charts
# ---------------------------------------------------------------------------

class ChartSeries(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ChartSeries",
			display_name="CV Chart Series",
			category=CATEGORY,
			description="Interactive bar/line/area/step chart of collected "
			            "numbers, drawn in the node (Apache ECharts): hover a "
			            "mark to read its exact value, box-zoom, save a PNG. "
			            "Use it whenever an array is a MEASUREMENT PER THING "
			            "rather than a picture - one similarity score per "
			            "detected face, a histogram, a metric per method - "
			            "where 'Preview CV Array' would render a few pixels "
			            "and normalize the absolute scale away. Feed a 1-D "
			            "array for one series or (samples x series) for "
			            "several. x_labels names each sample: give it the SAME "
			            "';'-separated string that labels the matching image "
			            "tiles (Draw Text / RegexSplitDataList) and bar i "
			            "provably refers to tile i. reference_lines draws the "
			            "decision threshold the numbers are judged against.",
			inputs=[
				NPArray.Input("values",
					tooltip="1-D array (one series), or 2-D (samples x series) "
					        "with one COLUMN per series. (N, 1) and (1, N) both "
					        "count as one series of N samples."),
				io.Combo.Input("chart",
					options=["bar", "line", "line (smooth)", "area", "step",
					         "scatter"],
					default="bar",
					tooltip="Mark type. 'bar' for independent measurements "
					        "(one per detected object/method); 'line'/'area'/"
					        "'step' when consecutive samples form a signal or "
					        "a curve (histogram, score vs parameter); "
					        "'scatter' to show samples without implying "
					        "continuity."),
				io.String.Input("title", default="",
					tooltip="Heading drawn above the chart. Empty draws none."),
				io.String.Input("x_labels", default="", optional=True,
					tooltip="Name per sample, as a JSON list or ';' / newline / "
					        "',' separated text (e.g. 'face 0;face 1;face 2'). "
					        "Missing names fall back to '#0', '#1', ... - a "
					        "short list never breaks the chart. Feed the same "
					        "string that labels the corresponding images."),
				io.String.Input("series_names", default="", optional=True,
					tooltip="Name per series (per COLUMN of values), same "
					        "formats as x_labels. Shown in the legend and the "
					        "tooltip; defaults to 'series 0', 'series 1', ..."),
				io.String.Input("reference_lines", default="", optional=True,
					tooltip="Horizontal marker lines: 'value' or 'value=label', "
					        "separated like the label lists - e.g. "
					        "'0.363=SFace threshold'. This is how a decision "
					        "threshold becomes visible in the picture instead "
					        "of living only in a widget."),
				io.Combo.Input("color_by",
					options=["series color", "value (gradient)",
					         "above/below first reference line"],
					default="series color", optional=True,
					tooltip="'series color' gives each series one color. "
					        "'value (gradient)' colors every mark by its own "
					        "value and adds a labelled scale. 'above/below "
					        "first reference line' splits the marks into two "
					        "colors at the threshold - the pass/fail read, "
					        "with the legend saying which side is which."),
				io.Combo.Input("orientation",
					options=["vertical (bars go up)",
					         "horizontal (bars go right)"],
					default="vertical (bars go up)", optional=True,
					tooltip="Horizontal keeps long sample names readable - the "
					        "right choice when the labels are file or person "
					        "names rather than indices."),
				io.Boolean.Input("show_values", default=False, optional=True,
					tooltip="Print each value next to its mark, so the chart "
					        "reads without hovering (and survives being saved "
					        "as a PNG)."),
				io.Combo.Input("value_range",
					options=["auto (data range)", "start at zero", "manual"],
					default="start at zero", optional=True,
					tooltip="'auto' fits the values, which EXAGGERATES small "
					        "differences; 'start at zero' keeps bar lengths "
					        "proportional to the values; 'manual' pins the "
					        "axis so separate runs are comparable."),
				io.Float.Input("min_value", default=0.0, min=-1e9, max=1e9,
					optional=True, advanced=True,
					tooltip="Value-axis minimum ('manual' range only)."),
				io.Float.Input("max_value", default=1.0, min=-1e9, max=1e9,
					optional=True, advanced=True,
					tooltip="Value-axis maximum ('manual' range only)."),
				io.String.Input("x_axis_name", default="", optional=True,
					advanced=True,
					tooltip="Caption under the sample axis (what the samples "
					        "ARE: 'detected face', 'bin', 'method')."),
				io.String.Input("y_axis_name", default="", optional=True,
					advanced=True,
					tooltip="Caption beside the value axis, with units "
					        "('cosine similarity', 'pixels', 'dB')."),
				io.Combo.Input("palette",
					options=list(CATEGORICAL_PALETTES),
					default="colorblind-safe (Okabe-Ito)",
					optional=True, advanced=True,
					tooltip="Series colors. The default stays readable for a "
					        "colour-blind reader and in greyscale; 'bright' is "
					        "punchier but puts a red and a green next to each "
					        "other; the sampled ramps suit ORDERED series "
					        "(sweep steps)."),
				io.Combo.Input("gradient_palette",
					options=list(CONTINUOUS_PALETTES), default="viridis",
					optional=True, advanced=True,
					tooltip="Ramp used by the 'value (gradient)' coloring; "
					        "'viridis' matches 'Preview CV Array' heatmaps."),
				io.Combo.Input("theme", options=list(THEMES), default="dark",
					optional=True, advanced=True,
					tooltip="Chart colors. 'light' is for a PNG that will be "
					        "pasted into a document."),
				io.Int.Input("decimals", default=3, min=0, max=8,
					optional=True, advanced=True,
					tooltip="Decimals shown in the tooltips and the printed "
					        "values. Trailing zeros are dropped."),
				io.Int.Input("chart_height", default=320, min=160, max=2048,
					optional=True, advanced=True,
					tooltip="Height of the chart area inside the node, in "
					        "pixels."),
			],
			outputs=[
				io.String.Output(display_name="spec",
					tooltip="The chart definition as JSON - the exact option "
					        "the frontend renders. Wire into 'Preview as Text' "
					        "to read the values that were charted."),
			],
			is_output_node=True,
		)

	@classmethod
	def execute(cls, values, chart, title, x_labels="", series_names="",
	            reference_lines="", color_by="series color",
	            orientation="vertical (bars go up)", show_values=False,
	            value_range="start at zero", min_value=0.0, max_value=1.0,
	            x_axis_name="", y_axis_name="",
	            palette="colorblind-safe (Okabe-Ito)",
	            gradient_palette="viridis", theme="dark", decimals=3,
	            chart_height=320) -> io.NodeOutput:
		data = as_columns(values, "values")
		samples, count = data.shape
		names = labels_for(series_names, count, "series ")
		categories = labels_for(x_labels, samples, "#")
		references = parse_reference_lines(reference_lines)
		colors = sample_palette(palette, count)
		horizontal = orientation.startswith("horizontal")
		option = base_option(title, theme, legend=count > 1)
		option["color"] = colors
		option["tooltip"] = {"trigger": "axis" if chart != "scatter" else "item",
		                     "axisPointer": {"type": "shadow"}}
		category_axis = axis_style(theme, x_axis_name, category=True,
		                           data=categories, invert=horizontal)
		value_axis = apply_range(axis_style(theme, y_axis_name), value_range,
		                         min_value, max_value)
		option["xAxis"] = value_axis if horizontal else category_axis
		option["yAxis"] = category_axis if horizontal else value_axis
		kind = {"bar": "bar", "scatter": "scatter"}.get(chart, "line")
		marks = mark_line(references, theme, horizontal)
		colors_holder = theme_colors(theme)
		option["series"] = []
		for column in range(count):
			series = {
				"name": names[column], "type": kind,
				"data": [jsonable(v) for v in data[:, column]],
			}
			if chart == "line (smooth)":
				series["smooth"] = True
			elif chart == "step":
				series["step"] = "middle"
			elif chart == "area":
				series["areaStyle"] = {"opacity": 0.25}
			if kind == "line":
				series["symbolSize"] = 6
				series["sampling"] = "lttb"
			if kind == "bar":
				series["barMaxWidth"] = 48
			if show_values:
				series["label"] = {
					"show": True, "color": colors_holder["label"], "fontSize": 10,
					"position": "right" if horizontal else "top",
					"formatter": "{c}"}
			if marks and column == 0:
				series["markLine"] = marks
			option["series"].append(series)
		cls._apply_color_by(option, color_by, data, references, gradient_palette,
		                    theme, horizontal)
		return chart_output(option, chart_height, "series", decimals)

	@staticmethod
	def _apply_color_by(option, color_by, data, references, ramp, theme, horizontal):
		"""visualMap for the value-driven colorings; no-op for 'series color'."""
		finite = data[np.isfinite(data)]
		if color_by == "series color" or finite.size == 0:
			return
		stops = CONTINUOUS_PALETTES.get(ramp, CONTINUOUS_PALETTES["viridis"])
		colors = theme_colors(theme)
		common = {"dimension": 0 if horizontal else 1, "orient": "horizontal",
		          "left": "center", "bottom": 4, "itemWidth": 12, "itemHeight": 10,
		          "textStyle": {"color": colors["text"], "fontSize": 10}}
		if color_by == "value (gradient)":
			low, high = float(finite.min()), float(finite.max())
			if high <= low:
				high = low + 1.0
			option["visualMap"] = dict(common, type="continuous", min=low,
			                           max=high, calculable=True,
			                           inRange={"color": stops})
		elif references:
			threshold, label = references[0]
			option["visualMap"] = dict(
				common, type="piecewise", showLabel=True,
				pieces=[{"lt": threshold, "color": stops[0],
				         "label": f"below {label}"},
				        {"gte": threshold, "color": stops[-1],
				         "label": f"at or above {label}"}])
		option["grid"]["bottom"] = 74


class ChartScatter(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ChartScatter",
			display_name="CV Chart Scatter",
			category=CATEGORY,
			description="Interactive x/y scatter of a point set (Apache "
			            "ECharts): hover a point to read its coordinates and "
			            "name, box-zoom into a cluster, save a PNG. This is "
			            "the plot for point DATA - k-means centers, feature "
			            "vectors, training samples, a keypoint spread - as "
			            "opposed to 'CV Draw Points', which stamps points "
			            "onto the image they came from. Optional classes color "
			            "the points by cluster/label with a legend. Turn on "
			            "'y grows downwards' when the coordinates are pixel "
			            "positions, so the plot matches the image.",
			inputs=[
				NPArray.Input("points",
					tooltip="(N, 2) x/y coordinates; (N, 1, 2) - the shape the "
					        "point nodes emit - is accepted as-is. Empty is "
					        "valid and draws an empty plot."),
				io.String.Input("title", default="",
					tooltip="Heading drawn above the chart. Empty draws none."),
				NPArray.Input("classes", optional=True,
					tooltip="Optional integer label per point (k-means labels, "
					        "classifier predictions): points are grouped into "
					        "one colored, legend-named series per distinct "
					        "value. Length must match the point count."),
				io.String.Input("class_names", default="", optional=True,
					tooltip="Names for the class values, in ascending value "
					        "order, as a JSON list or ';' / newline / ',' "
					        "separated text. Defaults to 'class 0', ..."),
				io.String.Input("point_names", default="", optional=True,
					tooltip="Name per point (same order as points), shown in "
					        "the tooltip - use the same label list that names "
					        "the matching image tiles."),
				io.Boolean.Input("y_down", default=False, optional=True,
					tooltip="Flip the y axis so it grows downwards, matching "
					        "image/pixel coordinates. Leave off for ordinary "
					        "feature-space data."),
				io.Int.Input("point_size", default=10, min=2, max=60,
					optional=True,
					tooltip="Marker diameter in pixels."),
				io.String.Input("x_axis_name", default="", optional=True,
					advanced=True,
					tooltip="Caption under the x axis (what x measures)."),
				io.String.Input("y_axis_name", default="", optional=True,
					advanced=True,
					tooltip="Caption beside the y axis (what y measures)."),
				io.Combo.Input("palette",
					options=list(CATEGORICAL_PALETTES),
					default="colorblind-safe (Okabe-Ito)", optional=True,
					advanced=True,
					tooltip="Class colors. The sampled ramps suit ORDERED "
					        "classes (a sweep, a rank); the fixed palettes suit "
					        "unordered ones."),
				io.Combo.Input("theme", options=list(THEMES), default="dark",
					optional=True, advanced=True,
					tooltip="Chart colors. 'light' is for a PNG that will be "
					        "pasted into a document."),
				io.Int.Input("decimals", default=3, min=0, max=8,
					optional=True, advanced=True,
					tooltip="Decimals shown for the coordinates in the "
					        "tooltip. Trailing zeros are dropped."),
				io.Int.Input("chart_height", default=340, min=160, max=2048,
					optional=True, advanced=True,
					tooltip="Height of the chart area inside the node, in "
					        "pixels."),
			],
			outputs=[
				io.String.Output(display_name="spec",
					tooltip="The chart definition as JSON - the exact option "
					        "the frontend renders. Wire into 'Preview as Text' "
					        "to read the plotted coordinates."),
			],
			is_output_node=True,
		)

	@classmethod
	def execute(cls, points, title, classes=None, class_names="",
	            point_names="", y_down=False, point_size=10, x_axis_name="",
	            y_axis_name="", palette="colorblind-safe (Okabe-Ito)",
	            theme="dark", decimals=3, chart_height=340) -> io.NodeOutput:
		data = np.asarray(points)
		data = data.reshape(-1, data.shape[-1]) if data.ndim >= 2 else data.reshape(-1, 1)
		if data.size and data.shape[1] != 2:
			raise ValueError(
				f"points must be (N, 2) x/y pairs, got shape {np.asarray(points).shape}. "
				"For per-sample measurements use 'CV Chart Series'.")
		data = np.asarray(data, dtype=np.float64).reshape(-1, 2)
		if data.size > MAX_SERIES_VALUES:
			raise ValueError(
				f"points has {len(data)} points, more than the "
				f"{MAX_SERIES_VALUES // 2} a chart can carry.")
		names = labels_for(point_names, len(data), "#") if str(point_names).strip() else None
		groups = cls._group(data, classes, class_names, names)
		colors = sample_palette(palette, len(groups))
		option = base_option(title, theme, legend=len(groups) > 1)
		option["color"] = colors
		option["tooltip"] = {"trigger": "item"}
		option["xAxis"] = axis_style(theme, x_axis_name)
		option["yAxis"] = axis_style(theme, y_axis_name, invert=bool(y_down))
		option["series"] = [
			{"name": name, "type": "scatter", "symbolSize": int(point_size),
			 "data": rows}
			for name, rows in groups
		]
		return chart_output(option, chart_height, "scatter", decimals)

	@staticmethod
	def _group(data, classes, class_names, names):
		"""[(series name, [[x, y, point name], ...]), ...] split by class."""
		def rows(indices):
			out = []
			for i in indices:
				row = [jsonable(data[i, 0]), jsonable(data[i, 1])]
				if names:
					row.append(names[i])
				out.append(row)
			return out

		if classes is None or np.asarray(classes).size == 0:
			return [("points", rows(range(len(data))))]
		labels = np.asarray(classes).reshape(-1)
		if len(labels) != len(data):
			raise ValueError(
				f"classes has {len(labels)} entries but there are {len(data)} "
				"points; they are paired by index.")
		values = sorted({int(v) for v in labels})
		captions = labels_for(class_names, max(values) + 1 if values else 0, "class ")
		groups = []
		for value in values:
			indices = [i for i, v in enumerate(labels) if int(v) == value]
			caption = captions[value] if 0 <= value < len(captions) else f"class {value}"
			groups.append((caption, rows(indices)))
		return groups


class ChartMatrix(io.ComfyNode):
	@classmethod
	def define_schema(cls):
		return io.Schema(
			node_id="CV_ChartMatrix",
			display_name="CV Chart Matrix",
			category=CATEGORY,
			description="Interactive labelled heatmap of a 2-D array (Apache "
			            "ECharts): every cell carries its row and column NAME "
			            "and its exact value on hover, with a labelled color "
			            "scale in the array's own units. Made for small "
			            "tabular arrays where each cell means something - a "
			            "confusion matrix, a per-pair score matrix, a method x "
			            "metric table - which 'Preview CV Array' can only "
			            "render as one anonymous pixel per cell. Prefer 'CV "
			            "Chart Series' when one dimension is 1: a single "
			            "column of scores reads better as labelled bars.",
			inputs=[
				NPArray.Input("matrix",
					tooltip="2-D array (rows x columns). Size-1 axes are "
					        "squeezed away, so an (N, 1, M) block charts as "
					        "N x M."),
				io.String.Input("title", default="",
					tooltip="Heading drawn above the chart. Empty draws none."),
				io.String.Input("row_labels", default="", optional=True,
					tooltip="Name per row, as a JSON list or ';' / newline / "
					        "',' separated text. Missing names fall back to "
					        "'row 0', 'row 1', ..."),
				io.String.Input("col_labels", default="", optional=True,
					tooltip="Name per column, same formats. Missing names fall "
					        "back to 'col 0', 'col 1', ..."),
				io.Boolean.Input("show_values", default=True, optional=True,
					tooltip="Print each cell's value inside it, so the table "
					        "reads without hovering. Turn off for large "
					        "matrices where the text would not fit."),
				io.Int.Input("decimals", default=3, min=0, max=8,
					optional=True,
					tooltip="Significant decimals for the printed cell values "
					        "and the tooltip."),
				io.Combo.Input("value_range",
					options=["auto (data range)", "symmetric around zero",
					         "manual"],
					default="auto (data range)", optional=True,
					tooltip="What the color scale spans. 'auto' uses the "
					        "array's own min/max - which makes tiny "
					        "differences look dramatic; 'manual' pins it to the "
					        "meaningful range (0-1 for similarities) so the "
					        "colors mean the same thing across runs; "
					        "'symmetric around zero' centers a diverging ramp "
					        "on 0 for signed data (differences, correlations)."),
				io.Float.Input("min_value", default=0.0, min=-1e9, max=1e9,
					optional=True, advanced=True,
					tooltip="Color scale minimum ('manual' range only)."),
				io.Float.Input("max_value", default=1.0, min=-1e9, max=1e9,
					optional=True, advanced=True,
					tooltip="Color scale maximum ('manual' range only)."),
				io.Combo.Input("palette", options=list(CONTINUOUS_PALETTES),
					default="viridis", optional=True, advanced=True,
					tooltip="Color ramp. 'viridis' matches 'Preview CV Array'"
					        "'s heatmap render; 'coolwarm (diverging)' belongs "
					        "with 'symmetric around zero'."),
				io.Combo.Input("theme", options=list(THEMES), default="dark",
					optional=True, advanced=True,
					tooltip="Chart colors. 'light' is for a PNG that will be "
					        "pasted into a document."),
				io.Int.Input("chart_height", default=340, min=160, max=2048,
					optional=True, advanced=True,
					tooltip="Height of the chart area inside the node, in "
					        "pixels."),
			],
			outputs=[
				io.String.Output(display_name="spec",
					tooltip="The chart definition as JSON - the exact option "
					        "the frontend renders. Wire into 'Preview as Text' "
					        "to read the charted cells."),
			],
			is_output_node=True,
		)

	@classmethod
	def execute(cls, matrix, title, row_labels="", col_labels="",
	            show_values=True, decimals=3, value_range="auto (data range)",
	            min_value=0.0, max_value=1.0, palette="viridis", theme="dark",
	            chart_height=340) -> io.NodeOutput:
		data = np.asarray(matrix)
		if data.ndim > 2:
			data = data.reshape([d for d in data.shape if d != 1] or [1])
		data = np.atleast_2d(np.asarray(data, dtype=np.float64))
		if data.ndim != 2:
			raise ValueError(
				f"matrix must be 2-D, got shape {np.asarray(matrix).shape}.")
		if data.size > MAX_MATRIX_CELLS:
			raise ValueError(
				f"matrix has {data.size} cells, more than the {MAX_MATRIX_CELLS} a "
				"chart can carry. Preview image-shaped data with 'Preview CV "
				"Array' instead.")
		rows, cols = data.shape
		row_names = labels_for(row_labels, rows, "row ")
		col_names = labels_for(col_labels, cols, "col ")
		low, high = cls._range(data, value_range, min_value, max_value)
		colors = theme_colors(theme)
		option = base_option(title, theme)
		option["grid"]["bottom"] = 72
		option["tooltip"] = {"trigger": "item"}
		option["xAxis"] = axis_style(theme, "", category=True, data=col_names)
		option["yAxis"] = axis_style(theme, "", category=True, data=row_names,
		                             invert=True)
		option["visualMap"] = {
			"type": "continuous", "min": low, "max": high, "calculable": True,
			"orient": "horizontal", "left": "center", "bottom": 4,
			"itemWidth": 12, "itemHeight": 120,
			"textStyle": {"color": colors["text"], "fontSize": 10},
			"inRange": {"color": CONTINUOUS_PALETTES.get(
				palette, CONTINUOUS_PALETTES["viridis"])},
		}
		cells = [[c, r, jsonable(data[r, c])]
		         for r in range(rows) for c in range(cols)]
		option["series"] = [{
			"type": "heatmap", "data": cells,
			"itemStyle": {"borderColor": colors["bg"], "borderWidth": 1},
			"label": {"show": bool(show_values), "color": colors["label"],
			          "fontSize": 10},
			"emphasis": {"itemStyle": {"borderColor": colors["label"],
			                           "borderWidth": 2}},
		}]
		return chart_output(option, chart_height, "matrix", decimals)

	@staticmethod
	def _range(data, mode, low, high):
		finite = data[np.isfinite(data)]
		if mode == "manual":
			return float(low), float(high)
		if finite.size == 0:
			return 0.0, 1.0
		if mode == "symmetric around zero":
			extent = float(np.abs(finite).max()) or 1.0
			return -extent, extent
		data_low, data_high = float(finite.min()), float(finite.max())
		return (data_low, data_high) if data_high > data_low else (data_low, data_low + 1.0)


NODES = [Plot2D, ChartSeries, ChartScatter, ChartMatrix]
