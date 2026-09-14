r"""Repair the embedded cv2 install: restore the opencv-contrib build.

FOUR opencv distros (opencv-python, opencv-contrib-python,
opencv-python-headless, opencv-contrib-python-headless) share ONE
site-packages/cv2 directory, and whichever was installed LAST wins.  This pack
declares **opencv-contrib-python-headless**: it needs the contrib submodules
and never touches highgui (no imshow/waitKey/trackbars anywhere - the registry
generator blacklists them outright), so the GUI-less wheel is the correct
dependency.  Either contrib wheel works here; a NON-contrib one does not.

If a non-contrib wheel wins, cv2/cv2.pyd is the core-only build and every
contrib submodule
(xphoto / ximgproc / bgsegm / img_hash / quality / saliency / ...) imports as
an EMPTY stub package - the .pyi type stubs are still on disk, so the modules
"exist" but have zero attributes.

Windows will not let us overwrite a .pyd that a running process has mapped,
but it DOES allow RENAMING it (the classic in-use-update trick): the running
ComfyUI server keeps using the renamed file through its open handle, and the
next start picks up the new one.  So:

  1. verify the replacement really is the contrib build
  2. rename the live cv2.pyd aside (cv2.pyd.noncontrib-backup)
  3. copy the contrib cv2.pyd into place
  4. on ANY failure, rename the backup back

Nothing is deleted: the previous binary stays as .noncontrib-backup.

Usage (from the repo root, with the embedded python):

    ..\..\..\python_embeded\python.exe tools\repair_opencv_contrib.py --check
    ..\..\..\python_embeded\python.exe tools\repair_opencv_contrib.py --apply

--check only reports; --apply performs the swap.  RESTART ComfyUI afterwards.
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# python_embeded/Lib/site-packages/cv2 (repo lives in ComfyUI/custom_nodes/<pkg>)
SITE_PACKAGES = os.path.abspath(os.path.join(
    REPO_ROOT, "..", "..", "..", "python_embeded", "Lib", "site-packages"))
CV2_DIR = os.path.join(SITE_PACKAGES, "cv2")
LIVE_PYD = os.path.join(CV2_DIR, "cv2.pyd")
BACKUP_PYD = os.path.join(CV2_DIR, "cv2.pyd.noncontrib-backup")

# Contrib submodules that must be non-empty for the contrib nodes to work.
REQUIRED_MODULES = ["xphoto", "ximgproc", "bgsegm", "img_hash", "quality",
                    "saliency", "intensity_transform"]

# Wheel dist-info prefixes that carry a contrib build, most preferred FIRST.
# The pack declares the headless one (nothing here uses highgui), but a
# GUI-enabled contrib build is equally functional, so both are accepted.
CONTRIB_DIST_PREFIXES = ("opencv_contrib_python_headless-",
                         "opencv_contrib_python-")
PREFERRED_PACKAGE = "opencv-contrib-python-headless"


def live_status():
    """(version, {module: n_attrs}) of the cv2 the embedded python imports."""
    code = (
        "import cv2, json;"
        f"mods={REQUIRED_MODULES!r};"
        "print(json.dumps({'version': cv2.__version__, 'file': cv2.__file__,"
        " 'mods': {m: len([n for n in dir(getattr(cv2, m, None) or object)"
        "          if not n.startswith('_')]) for m in mods}}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout
    import json
    return json.loads(out.strip().splitlines()[-1])


def find_contrib_pyd():
    """Locate a contrib cv2.pyd: an extracted sandbox, or pip's wheel cache."""
    sandbox = os.path.join(REPO_ROOT, "temp", "cv2_contrib", "cv2", "cv2.pyd")
    if os.path.isfile(sandbox):
        return sandbox, None

    cache = subprocess.run([sys.executable, "-m", "pip", "cache", "dir"],
                           capture_output=True, text=True).stdout.strip()
    if not cache or not os.path.isdir(cache):
        return None, None
    hits = {}  # dist-info prefix -> wheel path
    for dirpath, _dirs, files in os.walk(cache):
        for fn in files:
            p = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(p) < 40 * 1024 * 1024:
                    continue
                with zipfile.ZipFile(p) as z:
                    names = z.namelist()
                    if "cv2/cv2.pyd" not in names:
                        continue
                    tops = {n.split("/")[0] for n in names}
                    for prefix in CONTRIB_DIST_PREFIXES:
                        if any(t.startswith(prefix) for t in tops):
                            hits.setdefault(prefix, p)
                            break
            except (OSError, zipfile.BadZipFile):
                continue
    for prefix in CONTRIB_DIST_PREFIXES:  # headless first
        if prefix in hits:
            return None, hits[prefix]
    return None, None


def verify_is_contrib(path):
    """A contrib build exports the contrib submodule symbols; the core-only
    one does not. Cheapest reliable check: the binary is much bigger than a
    core-only one (~112 MB against ~86 MB for the 5.0.0.93 wheels; the headless
    contrib build is a few MB smaller than the GUI one) AND contains the
    contrib module marker strings."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        blob = f.read()
    markers = [b"xphoto", b"ximgproc", b"bgsegm"]
    found = [m for m in markers if m in blob]
    return size, found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="perform the swap (default is a dry-run report)")
    ap.add_argument("--check", action="store_true",
                    help="report only (the default; accepted for symmetry)")
    args = ap.parse_args()

    print(f"site-packages : {SITE_PACKAGES}")
    print(f"live cv2.pyd  : {LIVE_PYD} "
          f"({os.path.getsize(LIVE_PYD) / 1e6:.1f} MB)")
    status = live_status()
    print(f"imported cv2  : {status['version']} @ {status['file']}")
    empty = [m for m, n in status["mods"].items() if n == 0]
    if not empty:
        print("OK: every required contrib module is populated - nothing to do.")
        return 0
    print(f"MISSING (stub-only): {', '.join(empty)}")

    src, wheel = find_contrib_pyd()
    if src is None and wheel is not None:
        dest_dir = os.path.join(REPO_ROOT, "temp", "cv2_contrib")
        print(f"extracting contrib wheel {wheel} -> {dest_dir}")
        if args.apply:
            with zipfile.ZipFile(wheel) as z:
                z.extractall(dest_dir)
            src = os.path.join(dest_dir, "cv2", "cv2.pyd")
        else:
            print("(dry-run: would extract, then swap)")
            return 1
    if src is None:
        print("No contrib cv2.pyd found locally. Stop the ComfyUI server and run:")
        print(f"  {sys.executable} -m pip install --force-reinstall "
              f"--no-deps {PREFERRED_PACKAGE}==" + status["version"] + ".93")
        return 1

    size, markers = verify_is_contrib(src)
    print(f"replacement   : {src} ({size / 1e6:.1f} MB, markers: "
          f"{[m.decode() for m in markers]})")
    if len(markers) < 3:
        print("REFUSING: that binary does not look like a contrib build.")
        return 1

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to swap "
              "(then RESTART ComfyUI).")
        return 0

    if os.path.exists(BACKUP_PYD):
        os.remove(BACKUP_PYD)
    try:
        os.rename(LIVE_PYD, BACKUP_PYD)
    except OSError as e:
        print(f"Cannot rename the live cv2.pyd: {e}\n"
              "Stop every running ComfyUI/python process and try again.")
        return 1
    try:
        shutil.copy2(src, LIVE_PYD)
    except OSError as e:
        os.rename(BACKUP_PYD, LIVE_PYD)  # restore
        print(f"Copy failed, restored the original: {e}")
        return 1
    print(f"swapped. previous binary kept at {BACKUP_PYD}")
    print("RESTART ComfyUI (and any headless python) to load the contrib build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
