# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 bmad4ever
# See LICENSE for the full GNU GPL v3 text.

"""Reading cv2's build banner without leaking the build machine's filesystem.

`cv2.getBuildInformation()` is the answer to "does this OpenCV have Eigen /
non-free / OpenCL / CUDA", which is exactly what a bug report needs. It is also
frozen at COMPILE time and quotes the build machine's directories: the extra
modules path, the install prefix, every compiler, python and third-party library
it linked. For a PyPI wheel that machine is a throwaway CI runner (paths like
`D:/a/opencv-python/opencv-python/...` that never existed on this computer), but
anyone who builds their own OpenCV - which is what the Eigen-gated nodes tell
them to do - bakes their own home directory, user name and project layout into
it instead. So the raw wrapper is blacklisted (generator.NONSAFE) and this
module backs the `CV Build Information` node, which serves the same text with
the paths redacted.

No cv2 nodes here, and no cv2 import beyond the banner: this is pure text
handling so it can be unit-tested against hand-written banners from platforms
this machine is not (`test_build_info_redaction`), and so `lowlevel.py` can
reuse `banner_flag()` for its Eigen check without importing a node module.
"""

import re

# Everything that may appear INSIDE one path segment. Excludes whitespace, the
# separators themselves, and the punctuation OpenCV uses to put several paths on
# one line or to append a "(ver 1.2)" note.
_SEG = r"[^\s\\/;,\"'<>|]"

# A space is part of the path only when a later separator proves the segment
# continues ("C:/Program Files/Microsoft Visual Studio/2022/..." is ONE path,
# "cl.exe  (ver 19.44)" is not) - and never when the next token STARTS a path of
# its own, or two paths on one line merge into a single match and only the last
# is redacted. That happens on the JNI line (three absolute include dirs) and on
# a compiler command line (-I/opt/eigen3/include -L/usr/local/lib). The four
# "starts a path" spellings are the four _PATH_RES entries; the dash is what
# keeps a flag prefix from matching an ordinary interior segment ("Files/").
_NEW_PATH = (r"(?![A-Za-z]:[\\/])(?!/)(?!~[\\/])"
             r"(?!-{1,2}[A-Za-z][\w-]*[\\/])")
_SP = r" " + _NEW_PATH + r"(?=" + _SEG + r"*(?: " + _SEG + r"+)*[\\/])"
_TAIL = r"(?:[\\/]|" + _SEG + r"|" + _SP + r")*"

# A POSIX path starts where a token starts: at whitespace, at the beginning, or
# after '=' or an opening quote/bracket. Without that, the RELATIVE path OpenCV
# prints for the python module ("install path: python/cv2/python-3") is caught
# mid-token and comes out as "python<path>/python-3" - a relative path names
# nothing about the machine, so redacting it is pure damage. The optional
# `flag` group is what still catches an include directory glued to a compiler
# switch (-I/opt/eigen3/include); it is put back untouched.
_LEAD = r"(?<![^\s=,(\"'])"
_FLAG = r"(?P<flag>-{1,2}[A-Za-z][\w-]*)?"

_PATH_RES = (
	re.compile(r"\\\\" + _SEG + _TAIL),            # UNC   \\server\share\...
	re.compile(r"[A-Za-z]:[\\/]" + _TAIL),         # drive C:/... or C:\...
	re.compile(_LEAD + r"~[\\/]" + _TAIL),         # home  ~/src/opencv
	# POSIX absolute. The lookahead demands a SECOND separator, which is what
	# keeps MSVC compiler flags (/DWIN32 /O2), "Video I/O" and "n/a" out of it.
	re.compile(_LEAD + _FLAG + r"/(?=" + _SEG + r"+[\\/])" + _SEG + _TAIL),
)

# Stashed before path matching so a documentation URL survives intact.
_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)

PLACEHOLDER = "<path>"


def redact_paths(text, keep_names=True):
	"""Replace filesystem paths in `text`. Returns (redacted, count).

	With `keep_names` the final component survives (`<path>/cl.exe`), which is
	what makes the banner still worth reading - the compiler, the python and the
	libraries are named, only their location is gone. Without it every path
	collapses to `<path>`.
	"""
	stash = []

	def _hide(m):
		stash.append(m.group(0))
		return f"\x00U{len(stash) - 1}\x00"

	out = _URL_RE.sub(_hide, text)
	count = 0

	def _sub(m):
		nonlocal count
		count += 1
		flag = m.group("flag") if "flag" in m.re.groupindex else None
		flag = flag or ""
		path = m.group(0)[len(flag):]
		if not keep_names:
			return flag + PLACEHOLDER
		base = re.split(r"[\\/]", path.rstrip("\\/"))[-1]
		return flag + (f"{PLACEHOLDER}/{base}" if base else PLACEHOLDER)

	for pat in _PATH_RES:
		out = pat.sub(_sub, out)
	for i, url in enumerate(stash):
		out = out.replace(f"\x00U{i}\x00", url)
	return out, count


def banner_flag(banner, label):
	"""True when the banner line `label:` answers YES.

	OpenCV writes "  Eigen:  YES (ver 3.4.0)" / "  Eigen:  NO", so the test is
	on the first word after the colon, not on a substring of the whole line.
	"""
	prefix = label + ":"
	for line in banner.splitlines():
		stripped = line.strip()
		if stripped.startswith(prefix):
			return stripped[len(prefix):].strip().upper().startswith("YES")
	return False
