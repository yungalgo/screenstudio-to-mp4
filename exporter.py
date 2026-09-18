#!/usr/bin/env python3
"""Unified export engine for screenstudio-to-mp4.

Runs prepare → cursor → render → audio → mux with progress callbacks.
Works both from source and from a frozen macOS .app (PyInstaller).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Optional

ProgressCb = Callable[[str, float], None]


def app_root() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.abspath(__file__))


ROOT = app_root()
SCRIPTS = os.path.join(ROOT, "scripts")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from render_lib import clean_path, parse_ffmpeg_progress  # noqa: E402


def _bundled(name: str) -> Optional[str]:
    cand = os.path.join(ROOT, name)
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    # PyInstaller sometimes puts binaries in a nested folder
    for sub in ("", "bin"):
        cand = os.path.join(ROOT, sub, name) if sub else os.path.join(ROOT, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def find_ffmpeg() -> str:
    bundled = _bundled("ffmpeg")
    if bundled:
        return bundled
    found = shutil.which("ffmpeg")
    if found:
        return found
    for loc in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.isfile(loc) and os.access(loc, os.X_OK):
            return loc
    return "ffmpeg"


def find_ffprobe() -> str:
    bundled = _bundled("ffprobe")
    if bundled:
        return bundled
    found = shutil.which("ffprobe")
    if found:
        return found
    for loc in ("/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"):
        if os.path.isfile(loc) and os.access(loc, os.X_OK):
            return loc
    return "ffprobe"


def check_dependencies() -> list[str]:
    """Return missing dependency descriptions (empty if OK)."""
    missing = []
    if find_ffmpeg() == "ffmpeg" and not shutil.which("ffmpeg"):
        missing.append("ffmpeg (install with: brew install ffmpeg)")
    if find_ffprobe() == "ffprobe" and not shutil.which("ffprobe"):
        missing.append("ffprobe (comes with ffmpeg)")
    try:
        import PIL  # noqa: F401
    except ImportError:
        missing.append("Pillow (install with: pip3 install pillow)")
    return missing


def _call_with_argv(main_fn, argv: list[str], **kwargs) -> None:
    old = sys.argv[:]
    try:
        sys.argv = argv
        main_fn(**kwargs)
    except SystemExit as exc:
        code = exc.code
        if code not in (0, None):
            raise RuntimeError(f"{argv[0]} exited with {code}") from exc
    finally:
        sys.argv = old


class RenderExporter:
    """Drive the full Screen Studio → MP4 pipeline."""

    def __init__(
        self,
        bundle_path: str,
        output_path: str,
        work_dir: Optional[str] = None,
        options: Optional[dict] = None,
    ):
        self.bundle_path = clean_path(bundle_path)
        self.output_path = clean_path(output_path)
        self.work_dir = (
            os.path.abspath(os.path.expanduser(work_dir))
            if work_dir
            else tempfile.mkdtemp(prefix="screenstudio_work_")
        )
        self.options = options or {}
        self.ffmpeg_path = find_ffmpeg()
        self.ffprobe_path = find_ffprobe()

    def run_pipeline(self, progress_callback: Optional[ProgressCb] = None) -> bool:
        def update(msg: str, pct: float) -> None:
            if progress_callback:
                progress_callback(msg, pct)

        missing = check_dependencies()
        if missing:
            raise RuntimeError("Missing dependencies:\n- " + "\n- ".join(missing))

        if not os.path.isdir(self.bundle_path):
            raise FileNotFoundError(
                f"Bundle not found or not a folder:\n{self.bundle_path}\n\n"
                "Tip: .screenstudio files are packages — use the full path, "
                "or pick one from the detected list."
            )
        if not os.path.isfile(os.path.join(self.bundle_path, "project.json")):
            raise FileNotFoundError(
                f"Not a valid Screen Studio project:\n{self.bundle_path}\n"
                "(missing project.json)"
            )

        os.makedirs(self.work_dir, exist_ok=True)
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        env = os.environ.copy()
        bin_dirs = []
        for p in (self.ffmpeg_path, self.ffprobe_path):
            d = os.path.dirname(p)
            if d and d not in bin_dirs:
                bin_dirs.append(d)
        if bin_dirs:
            env["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + env.get("PATH", "")

        try:
            import inspect_bundle
            import prepare_render
            import cursor_layer

            update("Reading your Screen Studio project…", 1.0)
            try:
                inspect_bundle.main(self.bundle_path)
            except SystemExit:
                pass
            except Exception:
                pass  # inspection is best-effort diagnostics

            update("Preparing video layout, zooms, and effects…", 2.0)
            prep = [
                "prepare_render.py",
                "--bundle",
                self.bundle_path,
                "--work",
                self.work_dir,
                "--output",
                self.output_path,
            ]
            opts = self.options
            if opts.get("frame"):
                prep.extend(["--frame", os.path.abspath(os.path.expanduser(opts["frame"]))])
            if "frame_blur" in opts:
                prep.extend(["--frame-blur", str(opts["frame_blur"])])
            if opts.get("screen_frac") is not None:
                prep.extend(["--screen-frac", str(opts["screen_frac"])])
            if opts.get("out_width") is not None:
                prep.extend(["--out-width", str(opts["out_width"])])
            if opts.get("webcam"):
                prep.extend(["--webcam", str(opts["webcam"])])
            if opts.get("zooms"):
                prep.extend(["--zooms", str(opts["zooms"])])
            if opts.get("cursor"):
                prep.extend(["--cursor", str(opts["cursor"])])
            if opts.get("audio"):
                prep.extend(["--audio", str(opts["audio"])])
            if opts.get("audio_cleanup"):
                prep.extend(["--audio-cleanup", str(opts["audio_cleanup"])])
            if "crf" in opts:
                prep.extend(["--crf", str(opts["crf"])])
            if opts.get("preset"):
                prep.extend(["--preset", str(opts["preset"])])
            _call_with_argv(prepare_render.main, prep)

            cursor_mode = opts.get("cursor", "auto")
            has_cursor = cursor_mode != "off"
            if has_cursor:
                update("Drawing the mouse cursor and click ripples…", 3.0)

                def _cursor_prog(cur_f: int, tot_f: int) -> None:
                    if tot_f > 0:
                        frac = min(1.0, max(0.0, cur_f / tot_f))
                        update(
                            f"Drawing the mouse cursor… {int(frac * 100)}%",
                            3.0 + frac * 7.0,
                        )

                _call_with_argv(
                    cursor_layer.main,
                    [
                        "cursor_layer.py",
                        "--bundle",
                        self.bundle_path,
                        "--work",
                        self.work_dir,
                    ],
                    progress_callback=_cursor_prog,
                )

            plan_path = os.path.join(self.work_dir, "plan.json")
            out_dur = 0.0
            try:
                import json
                plan = json.load(open(plan_path, encoding="utf-8"))
                out_dur = float(plan.get("out_dur_s") or 0)
            except Exception:
                pass

            vid_start = 10.0 if has_cursor else 3.0
            vid_end = 92.0
            update("Rendering video (this is the longest step)…", vid_start)
            self._run_script(
                os.path.join(self.work_dir, "render_full.sh"),
                env=env,
                progress=update,
                label="Rendering video",
                pct_start=vid_start,
                pct_end=vid_end,
                duration=out_dur,
            )

            update("Building the audio track…", 92.0)
            self._run_script(
                os.path.join(self.work_dir, "audio_build.sh"),
                env=env,
                progress=update,
                label="Building audio",
                pct_start=92.0,
                pct_end=97.0,
                duration=out_dur,
            )

            update("Combining video and audio into your MP4…", 97.0)
            self._run_script(os.path.join(self.work_dir, "mux.sh"), env=env)

            if not os.path.isfile(self.output_path):
                raise RuntimeError(f"Export finished but file not found:\n{self.output_path}")

            update("Done! Your MP4 is ready.", 100.0)
            return True
        except Exception as exc:
            update(f"Export failed: {exc}", -1.0)
            raise

    def _run_script(
        self,
        script_path: str,
        env: dict,
        progress: Optional[ProgressCb] = None,
        label: str = "",
        pct_start: float = 0.0,
        pct_end: float = 0.0,
        duration: float = 0.0,
    ) -> None:
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Missing generated script:\n{script_path}")

        content = open(script_path, encoding="utf-8").read()
        patched = content
        if self.ffmpeg_path != "ffmpeg":
            patched = patched.replace("ffmpeg ", f'"{self.ffmpeg_path}" ')
        if self.ffprobe_path != "ffprobe":
            patched = patched.replace("ffprobe ", f'"{self.ffprobe_path}" ')
        if patched != content:
            open(script_path, "w", encoding="utf-8").write(patched)

        proc = subprocess.Popen(
            ["zsh", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=self.work_dir,
            bufsize=1,
        )
        logs: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            logs.append(line)
            t = parse_ffmpeg_progress(line)
            if t is not None and duration > 0 and progress:
                frac = min(1.0, max(0.0, t / duration))
                pct = pct_start + (pct_end - pct_start) * frac
                mins, secs = divmod(int(t), 60)
                total_m, total_s = divmod(int(duration), 60)
                progress(
                    f"{label}… {mins}:{secs:02d} / {total_m}:{total_s:02d}",
                    pct,
                )
            print(line, end="", flush=True)
        rc = proc.wait()
        if rc != 0:
            err = "".join(logs).strip()
            raise RuntimeError(f"{os.path.basename(script_path)} failed:\n{err}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 exporter.py <bundle.screenstudio> <output.mp4>")
        sys.exit(1)

    exporter = RenderExporter(sys.argv[1], sys.argv[2])
    exporter.run_pipeline(lambda msg, pct: print(f"[{pct:5.1f}%] {msg}"))
