// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Interactive chart surface for the "CV Chart *" nodes (opencv_nodes/plots.py).
//
// Deliberately thin: the Python node builds the COMPLETE ECharts option and
// ships it under the custom ui key "cv_chart" (same mechanism as cv_preview3d's
// "cv3d"), so everything that decides what the picture says lives server-side,
// comes back out of the node as the `spec` STRING and is testable headlessly.
// This file only owns what JSON cannot carry:
//
//   * the DOM widget the chart lives in (node.addDOMWidget - the only surface
//     that renders in BOTH classic LiteGraph and "Modern Node Design"),
//   * value formatter FUNCTIONS (labels/tooltips) rebuilt from the numeric
//     `decimals` hint in the payload,
//   * resize/dispose plumbing.
//
// Payload contract: message.cv_chart[0] = { option, height, name, decimals }.
//
// ECharts 5.6.0 (Apache-2.0) is vendored under ./lib (the ESM build is fully
// self-contained - no bare imports); see lib/LICENSE.echarts.txt.

import { app } from "../../scripts/app.js";
import * as echarts from "./lib/echarts.esm.min.js";

const NODE_CLASSES = [
	"CV_ChartSeries",
	"CV_ChartScatter",
	"CV_ChartMatrix",
];
const MIN_AREA_H = 160;

/** Number -> short string: `decimals` places, trailing zeros dropped. */
function fmt(value, decimals) {
	if (value === null || value === undefined || typeof value !== "number") return "-";
	if (!Number.isFinite(value)) return "n/a";
	return String(Number(value.toFixed(decimals)));
}

/**
 * Install the formatter functions the JSON option could not carry.
 * Everything else in `option` is used exactly as Python built it.
 */
function withFormatters(option, decimals, kind) {
	const opt = option;
	if (kind === "matrix") {
		const cols = opt.xAxis?.data ?? [];
		const rows = opt.yAxis?.data ?? [];
		opt.tooltip = {
			...(opt.tooltip ?? {}),
			formatter: (p) => {
				const [c, r, v] = p.value;
				return `${rows[r] ?? r} &times; ${cols[c] ?? c}<br/><b>${fmt(v, decimals)}</b>`;
			},
		};
		for (const series of opt.series ?? []) {
			if (series.label?.show) {
				series.label = { ...series.label, formatter: (p) => fmt(p.value[2], decimals) };
			}
		}
		return opt;
	}
	if (kind === "scatter") {
		opt.tooltip = {
			...(opt.tooltip ?? {}),
			formatter: (p) => {
				const [x, y, name] = p.value;
				const head = name ? `${name}<br/>` : "";
				return `${head}${p.marker} ${p.seriesName}<br/>x ${fmt(x, decimals)}, y ${fmt(y, decimals)}`;
			},
		};
		return opt;
	}
	// series charts: axis-triggered tooltip listing every series at that sample
	opt.tooltip = {
		...(opt.tooltip ?? {}),
		formatter: (params) => {
			const list = Array.isArray(params) ? params : [params];
			if (!list.length) return "";
			const head = `${list[0].axisValueLabel ?? list[0].name}`;
			const body = list
				.map((p) => `${p.marker} ${p.seriesName}: <b>${fmt(p.value, decimals)}</b>`)
				.join("<br/>");
			return `${head}<br/>${body}`;
		},
	};
	for (const series of opt.series ?? []) {
		if (series.label?.show) {
			series.label = { ...series.label, formatter: (p) => fmt(p.value, decimals) };
		}
	}
	return opt;
}

/** Pixel gap left between the tick labels and the axis name. */
const NAME_PAD = 10;

/** The plot rectangle ECharts actually laid out (labels already subtracted). */
function gridRect(instance) {
	const grid = instance.getModel?.()?.getComponent?.("grid");
	const rect = grid?.coordinateSystem?.getRect?.();
	return rect && Number.isFinite(rect.x) ? rect : null;
}

/**
 * Push the axis names clear of the tick labels.
 *
 * `grid.containLabel` grows the plot area inward until the tick LABELS fit, but
 * it unions the axisLabel rects only - the axis NAME is not part of that, and
 * `nameGap` is measured from the axis line outwards, so any fixed gap chosen in
 * the JSON collides with the labels as soon as they are wider than the guess.
 * The label band is a font-metric fact that only exists after a render, which
 * is exactly the kind of thing the option JSON cannot carry: measure it from
 * the laid-out grid rect and set the gap from it. Only `nameGap` changes, and
 * names take no part in the layout, so this cannot feed back into the rect.
 */
function fitAxisNames(instance, option) {
	const grid = option?.grid;
	const rect = gridRect(instance);
	if (!grid || !rect || instance.isDisposed?.()) return;
	const patch = {};
	const yAxis = option.yAxis, xAxis = option.xAxis;
	if (yAxis?.name && typeof grid.left === "number") {
		const labels = rect.x - grid.left;          // rendered y label width
		patch.yAxis = { nameGap: Math.max(Math.round(labels) + NAME_PAD, NAME_PAD) };
	}
	if (xAxis?.name && typeof grid.bottom === "number") {
		const labels = instance.getHeight() - grid.bottom - (rect.y + rect.height);
		patch.xAxis = { nameGap: Math.max(Math.round(labels) + NAME_PAD, NAME_PAD) };
	}
	if (patch.xAxis || patch.yAxis) instance.setOption(patch);
}


/** The DOM widget + its ECharts instance, created once per node. */
function buildWidget(node) {
	if (node._cvChart) return node._cvChart;

	const container = document.createElement("div");
	container.style.position = "relative";
	container.style.width = "100%";
	container.style.height = "100%";
	container.style.overflow = "hidden";
	container.style.borderRadius = "4px";

	// Absolutely positioned so the canvas contributes nothing to the
	// container's flow height (same reasoning as annotate_points.js).
	const surface = document.createElement("div");
	surface.style.position = "absolute";
	surface.style.inset = "0";
	container.appendChild(surface);

	const placeholder = document.createElement("div");
	placeholder.textContent = "run to draw the chart";
	placeholder.style.position = "absolute";
	placeholder.style.inset = "0";
	placeholder.style.display = "flex";
	placeholder.style.alignItems = "center";
	placeholder.style.justifyContent = "center";
	placeholder.style.font = "12px sans-serif";
	placeholder.style.opacity = "0.45";
	placeholder.style.pointerEvents = "none";
	container.appendChild(placeholder);

	// A neutral widget type so the Vue node renderer treats it as a plain DOM
	// widget instead of routing it through its own draw-canvas path.
	const widget = node.addDOMWidget("cv_chart_surface", "opencv_chart", container, {
		serialize: false,
		hideOnZoom: false,
	});
	// BOTH flags: `options.serialize` only keeps the widget out of the API
	// prompt - workflow persistence is the widget-level property, and without it
	// this display surface saved its empty value as one extra widgets_values
	// entry (see web/cvlib/widget_values.js)
	widget.serialize = false;
	let wanted = MIN_AREA_H;
	widget.computeLayoutSize = () => ({
		minHeight: wanted, maxHeight: 1e6, minWidth: 20, maxWidth: 1e6,
	});
	// legacy LiteGraph rendering asks for a node size instead
	widget.computeSize = () => [node.size?.[0] ?? 300, wanted];

	let chart = null;
	const ensure = () => {
		if (!chart || chart.isDisposed?.()) {
			chart = echarts.init(surface, null, { renderer: "canvas" });
		}
		return chart;
	};

	// the last option as PYTHON built it - fitAxisNames measures against its
	// grid margins, so it must not be read back from the live (patched) chart
	let lastOption = null;
	const observer = new ResizeObserver(() => {
		if (chart && !chart.isDisposed?.() && surface.clientWidth > 0) {
			chart.resize();
			// label widths change with the tick set, so the gap is re-measured
			fitAxisNames(chart, lastOption);
		}
	});
	observer.observe(container);

	const onRemoved = node.onRemoved;
	node.onRemoved = function () {
		observer.disconnect();
		chart?.dispose?.();
		chart = null;
		node._cvChart = null;
		return onRemoved?.apply(this, arguments);
	};

	node._cvChart = {
		load(payload) {
			const height = Math.max(Number(payload?.height) || MIN_AREA_H, MIN_AREA_H);
			if (height !== wanted) {
				wanted = height;
				// grow the node body so the requested chart height actually fits
				const [w, h] = node.size ?? [320, 240];
				if (h < wanted + 80) node.setSize([w, wanted + 80]);
				node.graph?.setDirtyCanvas(true, true);
			}
			placeholder.style.display = "none";
			const instance = ensure();
			const option = withFormatters(
				payload.option, Number(payload.decimals ?? 3), payload.name);
			// notMerge: a rerun with fewer series must not leave the old ones behind
			instance.setOption(option, { notMerge: true });
			lastOption = option;
			fitAxisNames(instance, option);
			requestAnimationFrame(() => {
				if (instance.isDisposed?.() || surface.clientWidth <= 0) return;
				instance.resize();
				fitAxisNames(instance, option);
			});
		},
	};
	return node._cvChart;
}

app.registerExtension({
	name: "opencv_comfyui.cv_charts",

	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (!NODE_CLASSES.includes(nodeData.name)) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			onNodeCreated?.apply(this, arguments);
			buildWidget(this);
			if (this.size?.[1] < 320) this.setSize([Math.max(this.size[0], 380), 420]);
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const payload = message?.cv_chart?.[0];
			if (!payload) return;
			try {
				buildWidget(this).load(payload);
			} catch (err) {
				console.error("[opencv_comfyui] chart render failed", err);
			}
		};
	},
});
