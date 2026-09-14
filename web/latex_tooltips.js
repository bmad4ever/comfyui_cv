// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 bmad4ever
// See LICENSE for the full GNU GPL v3 text.

// Renders LaTeX inside node tooltips. ComfyUI draws a tooltip as a plain Vue
// text node, so the math that OpenCV's doxygen docs carry (hundreds of the
// param_docs.py / return_docs.py strings use $...$ and $$...$$) would otherwise
// show up as raw markup. This extension swaps the text content of every
// .node-tooltip for KaTeX-rendered math whenever it contains LaTeX delimiters.

import { app } from "../../scripts/app.js";
import katex from "./lib/katex.mjs";

if (!document.querySelector('link[data-cv-latex]')) {
	const link = document.createElement("link");
	link.rel = "stylesheet";
	// Resolved relative to THIS file, so the stylesheet is found whatever the
	// checkout directory is named (the served path is /extensions/<folder>/).
	link.href = new URL("./lib/katex.min.css", import.meta.url).href;
	link.dataset.cvLatex = "1";
	document.head.appendChild(link);
}

const DELIM = /(\$\$[\s\S]+?\$\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]|\$[^$\n]+?\$)/;

function renderLatexFrag(text) {
	const parts = text.split(DELIM);
	const frag = document.createDocumentFragment();
	for (const part of parts) {
		if (!part) continue;
		let m;
		if ((m = part.match(/^\$\$([\s\S]+?)\$\$$/))) {
			const span = document.createElement("span");
			span.style.display = "block";
			try {
				span.innerHTML = katex.renderToString(m[1], { displayMode: true });
			} catch (e) {
				span.textContent = m[0];
			}
			frag.appendChild(span);
		} else if ((m = part.match(/^\\\(([\s\S]+?)\\\)$/))) {
			const span = document.createElement("span");
			try {
				span.innerHTML = katex.renderToString(m[1], { displayMode: false });
			} catch (e) {
				span.textContent = m[0];
			}
			frag.appendChild(span);
		} else if ((m = part.match(/^\\\[(([\s\S]+?))\\\]$/))) {
			const span = document.createElement("span");
			span.style.display = "block";
			try {
				span.innerHTML = katex.renderToString(m[1], { displayMode: true });
			} catch (e) {
				span.textContent = m[0];
			}
			frag.appendChild(span);
		} else if ((m = part.match(/^\$([^$\n]+?)\$$/))) {
			const span = document.createElement("span");
			try {
				span.innerHTML = katex.renderToString(m[1], { displayMode: false });
			} catch (e) {
				span.textContent = m[0];
			}
			frag.appendChild(span);
		} else {
			frag.appendChild(document.createTextNode(part));
		}
	}
	return frag;
}

function processTextEl(el) {
	const text = el.textContent || "";
	if (!text || el.dataset.myTestLatex === text) return;
	if (!DELIM.test(text)) return;
	el.dataset.myTestLatex = text;
	try {
		el.textContent = "";
		el.appendChild(renderLatexFrag(text));
	} catch (e) {
		el.textContent = text;
	}
}

function scan() {
	document.querySelectorAll(".p-tooltip-text").forEach(processTextEl);
}

function installObserver() {
	const mo = new MutationObserver((records) => {
		for (const r of records) {
			for (const node of r.addedNodes) {
				if (node.nodeType !== 1) continue;
				if (node.classList?.contains("p-tooltip-text")) {
					processTextEl(node);
				} else if (node.querySelector) {
					node.querySelectorAll(".p-tooltip-text").forEach(processTextEl);
				}
			}
		}
	});
	mo.observe(document.body, { childList: true, subtree: true });
	setInterval(scan, 100);
}

app.registerExtension({
	name: "opencv_comfyui.latex_tooltips",
	setup() {
		requestAnimationFrame(installObserver);
	},
});

requestAnimationFrame(installObserver);