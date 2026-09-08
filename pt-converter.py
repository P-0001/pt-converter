# pt/pt-converter.py
#
# Standalone CLI that wraps Ultralytics' YOLO export to produce ONNX files from
# PyTorch (.pt) checkpoints. Intended to be packaged with PyInstaller into
# pt-converter.exe and invoked as a subprocess by the Go backend.
#
# SECURITY NOTE
# -------------
# A .pt file is a Python pickle. Loading an untrusted checkpoint with
# Ultralytics/PyTorch can execute arbitrary code with the current user's
# permissions. This script is a CLI tool, not a library, and it does not import
# or execute arbitrary code from the model file beyond what Ultralytics itself
# does when loading a .pt checkpoint (which is inherent to the .pt format).
# The Go backend must:
#   - show a warning that the model must come from a trusted source;
#   - never auto-convert a downloaded file;
#   - run this converter as a child process without elevation, never through
#     cmd.exe or PowerShell, passing each argument separately; and
#   - remember that a subprocess is not a sandbox.
# Stronger isolation requires a separate Windows sandbox design and is out of
# scope for this tool.
#
# MACHINE-READABLE CONTRACT
# -------------------------
# Command:  pt-converter.exe MODEL.pt --output MODEL.onnx --opset 12
#
# ALL output is JSON-parseable. Every line on stdout/stderr is prefixed with
# one of two constants:
#
#   PT_CONVERTER_LOG    — log/progress lines (library output + stage messages)
#   PT_CONVERTER_RESULT — exactly ONE final result line
#
# Log line JSON:    {"level":"info|warn|error","msg":"<text>","stage":"..."}
#   The "stage" field is optional; present on stage messages emitted by this
#   script (starting, complete, failed).
#
# Success result JSON: {"ok":true,"output":"<absolute path>","bytes":<int>}
# Failure result JSON: {"ok":false,"code":"<stable_error_code>"}
#
# Library output (Ultralytics, PyTorch) is captured by wrapping sys.stdout
# and sys.stderr with JSONLogStream, which re-emits each line as an
# PT_CONVERTER_LOG JSON object. Lines already carrying an PT_CONVERTER_
# prefix pass through unchanged to avoid double-wrapping.
#
# Exit code is 0 only after export completes and the output file exists.
# On failure the exit code is nonzero and the result JSON carries a stable
# error code.

import argparse
import json

# PyInstaller + Windows multiprocessing: without this the frozen exe
# deadlocks on spawn. Must run before anything else imports torch, which
# may start worker processes internally.
import multiprocessing
import os
import re
import shutil
import sys
from pathlib import Path

VERSION = "1.0.1"

multiprocessing.freeze_support()

_frozen = getattr(sys, "frozen", False)

# In frozen mode, torch's JIT cache (TORCH_HOME) defaults to a path inside
# the read-only extraction directory. Redirect it to a writable temp dir
# so torch doesn't crash trying to write cache files.
if _frozen:
    import tempfile

    _torch_cache = os.path.join(tempfile.gettempdir(), "pt-converter-torch")
    os.makedirs(_torch_cache, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", _torch_cache)

# ultralytics pip-installs anything it thinks is missing, at inference/export
# time, into whatever interpreter is running. In a frozen exe there is no pip
# and no write target — a silent multi-hundred-MB download attempt or a
# confusing crash. We install everything up front in the build; ultralytics
# must not second-guess it.
os.environ.setdefault("YOLO_AUTOINSTALL", "0")

PREFIX = "PT_CONVERTER_RESULT "
LOG_PREFIX = "PT_CONVERTER_LOG "

# Strip ANSI CSI escape sequences (color codes, cursor moves) from captured
# library/argparse output so the JSON "msg" values stay plain text. Covers
# SGR color codes (\x1b[1;34m) and other CSI sequences in one pattern.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Supported configuration: FP32 detection and pose models only.
SUPPORTED_TASKS = ("detect", "pose")
SUPPORTED_OPSETS = (12,)


class ConverterError(Exception):
    """Raised with a stable error code as the first arg."""

    def __init__(self, code, detail=None):
        super().__init__(code)
        self.code = code
        self.detail = detail


class JSONLogStream:
    """Wraps each line written to a stream as PT_CONVERTER_LOG JSON.

    Lines already starting with an PT_CONVERTER_ prefix pass through
    unchanged to avoid double-wrapping. All other text is wrapped as
    {"level":"<level>","msg":"<line>"}.
    """

    _PASS_PREFIXES = (PREFIX, LOG_PREFIX)

    def __init__(self, target, level):
        self.target = target
        self.level = level
        self._buf = ""

    def write(self, text):
        if not text:
            return
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)

    def _emit(self, line):
        stripped = _ANSI_RE.sub("", line).strip()
        if not stripped:
            return
        if stripped.startswith(self._PASS_PREFIXES):
            self.target.write(stripped + "\n")
        else:
            payload = {"level": self.level, "msg": stripped}
            self.target.write(
                LOG_PREFIX + json.dumps(payload, separators=(",", ":")) + "\n"
            )
        self.target.flush()

    def flush(self):
        if self._buf.strip():
            self._emit(self._buf)
            self._buf = ""
        self.target.flush()

    def __getattr__(self, name):
        return getattr(self.target, name)


def emit_result(payload):
    """Print exactly one result line: PREFIX + compact JSON."""
    print(PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def emit_log(level, msg, **extra):
    """Print a JSON log line: LOG_PREFIX + compact JSON.

    Optional keyword args are merged into the payload (e.g. stage="starting").
    """
    payload = {"level": level, "msg": msg}
    payload.update(extra)
    print(LOG_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def _install_json_streams():
    """Replace sys.stdout/stderr with JSONLogStream wrappers.

    Returns a restore function that puts the original streams back.
    """
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    sys.stdout = JSONLogStream(real_stdout, "info")
    sys.stderr = JSONLogStream(real_stderr, "warn")

    def restore():
        sys.stdout = real_stdout
        sys.stderr = real_stderr

    return restore


def _cleanup_partial(partial):
    """Best-effort removal of a .partial file. Raises ConverterError on failure."""
    if partial is None:
        return
    try:
        if partial.exists():
            partial.unlink()
    except Exception as exc:
        raise ConverterError("cleanup_failed", str(exc)) from exc


def _atomic_move(src, dst):
    """Copy src to dst.partial, fsync, os.replace to dst, then unlink src.

    Returns the final destination Path. Raises ConverterError on failure and
    attempts to clean up the .partial file before re-raising.
    """
    partial = dst.with_suffix(dst.suffix + ".partial")
    try:
        shutil.copy2(src, partial)
        with partial.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, dst)
    except Exception as exc:
        try:
            if partial.exists():
                partial.unlink()
        except Exception:  # noqa: BLE001, S110 - swallow secondary cleanup error; the primary failure is reported
            pass
        raise ConverterError("conversion_failed", f"atomic move failed: {exc}") from exc
    finally:
        # If the source differs from the destination, remove the automatic output.
        try:
            if src != dst and src.exists():
                src.unlink()
        except Exception:  # noqa: BLE001, S110 - failure to delete the automatic output is non-fatal; the requested output is already in place
            pass
    return dst


def run_conversion(model, output, opset, yolo_factory=None):
    """Core conversion logic, separated from argv parsing for testability.

    Args:
        model: resolved Path to the .pt model (must exist, must be .pt).
        output: resolved Path to the desired .onnx output (must not exist).
        opset: int opset version (must be in SUPPORTED_OPSETS).
        yolo_factory: callable(str) -> YOLO-like object with .task and .export.
            Defaults to importing ultralytics.YOLO. Injected in tests.

    Returns:
        The resolved output Path on success.

    Raises:
        ConverterError with a stable code on any failure.
    """
    partial = output.with_suffix(output.suffix + ".partial")

    try:
        # Validate opset first (cheap, no IO).
        if opset not in SUPPORTED_OPSETS:
            raise ConverterError("unsupported_opset", f"opset {opset} not supported")

        # Validate input extension.
        if model.suffix.lower() != ".pt":
            raise ConverterError("input_not_pt", f"input suffix {model.suffix!r}")

        # Validate input existence.
        if not model.exists():
            raise ConverterError("model_not_found", str(model))

        # Validate output does not already exist.
        if output.exists():
            raise ConverterError("output_exists", str(output))

        # Ensure the output parent directory exists.
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise ConverterError("conversion_failed", f"mkdir failed: {exc}") from exc

        # Load the model via Ultralytics (or an injected factory in tests).
        if yolo_factory is None:
            from ultralytics import YOLO as yolo_factory  # type: ignore

        try:
            yolo = yolo_factory(str(model))
        except ConverterError:
            raise
        except Exception as exc:
            raise ConverterError(
                "conversion_failed", f"model load failed: {exc}"
            ) from exc

        # Check the task type.
        task = getattr(yolo, "task", None)
        if task not in SUPPORTED_TASKS:
            raise ConverterError("unsupported_task", f"task {task!r} not supported")

        # Export to ONNX. Ultralytics writes next to the model with the same stem.
        try:
            exported_path = yolo.export(format="onnx", opset=opset, simplify=False)
        except Exception as exc:
            raise ConverterError("conversion_failed", f"export failed: {exc}") from exc

        exported = Path(str(exported_path)).resolve()

        # Atomically move the automatic output to the requested output.
        _atomic_move(exported, output)

        # Final existence + size check.
        if not output.exists():
            raise ConverterError("conversion_failed", "output missing after move")

        return output
    except ConverterError:
        # Clean up any .partial file on every failure path.
        _cleanup_partial(partial)
        raise
    except Exception as exc:
        _cleanup_partial(partial)
        raise ConverterError("conversion_failed", str(exc)) from exc


def build_parser():
    """Construct the argparse parser. Exposed for tests."""
    parser = argparse.ArgumentParser(
        prog="pt-converter",
        description="Convert a YOLO .pt checkpoint to ONNX.",
    )
    parser.add_argument("model", type=Path, help="Path to the input .pt model.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the desired output .onnx file.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        choices=[12],
        help="ONNX opset version (only 12 is supported).",
    )
    return parser


def main(argv=None, yolo_factory=None):
    """Entry point. Returns an exit code (0 on success, 1 on failure).

    Always emits exactly one PT_CONVERTER_RESULT line. All other output
    (library logs, stage messages) is emitted as PT_CONVERTER_LOG lines.
    """
    restore = _install_json_streams()
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        try:
            model = args.model.resolve()
            output = args.output.resolve()
        except Exception:  # noqa: BLE001 - path resolution failures
            emit_result({"ok": False, "code": "conversion_failed"})
            return 1

        emit_log(
            "info",
            f"Starting conversion V:{VERSION}",
            stage="starting",
            model=str(model),
            output=str(output),
            opset=args.opset,
        )

        try:
            final_output = run_conversion(model, output, args.opset, yolo_factory)
            size = final_output.stat().st_size
            emit_log(
                "info",
                "Conversion complete",
                stage="complete",
                output=str(final_output),
                bytes=int(size),
            )
            emit_result({"ok": True, "output": str(final_output), "bytes": int(size)})
            return 0
        except ConverterError as exc:
            emit_log(
                "error", f"Conversion failed: {exc.code}", stage="failed", code=exc.code
            )
            emit_result({"ok": False, "code": exc.code})
            return 1
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001 - unexpected failure still emits a line
            emit_log(
                "error", "Unexpected failure", stage="failed", code="conversion_failed"
            )
            emit_result({"ok": False, "code": "conversion_failed"})
            return 1
    finally:
        restore()


if __name__ == "__main__":
    try:
        _code = main()
        raise SystemExit(_code)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - last-resort guard
        emit_result({"ok": False, "code": "conversion_failed"})
        raise SystemExit(1)
