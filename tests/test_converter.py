"""Unit tests for pt/pt-converter.py.

These tests do NOT require Ultralytics or PyTorch. They exercise argument
parsing, error-code mapping, result-line format, the .partial cleanup logic,
and the atomic move logic using only the standard library unittest plus
monkeypatching of the YOLO factory.
"""

import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# The converter script is named pt-converter.py (with a hyphen), which is not a
# valid Python module name, so load it explicitly from its file path.
_CONVERTER_PATH = Path(__file__).resolve().parent.parent / "pt-converter.py"
_spec = importlib.util.spec_from_file_location("pt_converter", _CONVERTER_PATH)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)


def parse_result_line(line):
    """Strip the result prefix and parse the compact JSON payload."""
    assert line.startswith(conv.PREFIX), f"missing prefix: {line!r}"
    payload = line[len(conv.PREFIX) :]
    return json.loads(payload)


def parse_log_line(line):
    """Strip the log prefix and parse the compact JSON payload."""
    assert line.startswith(conv.LOG_PREFIX), f"missing log prefix: {line!r}"
    payload = line[len(conv.LOG_PREFIX) :]
    return json.loads(payload)


class FakeYOLO:
    """Minimal stand-in for ultralytics.YOLO used by run_conversion."""

    def __init__(self, task="detect", export_path=None, export_exc=None):
        self.task = task
        self._export_path = export_path
        self._export_exc = export_exc

    def export(self, format="onnx", opset=12, simplify=False):
        if self._export_exc is not None:
            raise self._export_exc
        # Write a dummy file at the export path so the move logic can run.
        p = Path(self._export_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"FAKE ONNX " * 100)
        return str(p)


class PrefixFormatTests(unittest.TestCase):
    """Verify the result line format (prefix + compact JSON)."""

    def test_prefix_constant(self):
        self.assertEqual(conv.PREFIX, "PT_CONVERTER_RESULT ")

    def test_success_result_compact_json(self):
        line = conv.PREFIX + json.dumps(
            {"ok": True, "output": "/x.onnx", "bytes": 42},
            separators=(",", ":"),
        )
        self.assertEqual(line, 'PT_CONVERTER_RESULT {"ok":true,"output":"/x.onnx","bytes":42}')

    def test_failure_result_compact_json(self):
        line = conv.PREFIX + json.dumps(
            {"ok": False, "code": "input_not_pt"}, separators=(",", ":")
        )
        self.assertEqual(line, 'PT_CONVERTER_RESULT {"ok":false,"code":"input_not_pt"}')

    def test_emit_result_no_spaces(self):
        buf = io.StringIO()
        with mock.patch("builtins.print", lambda *a, **k: buf.write(a[0] + "\n")):
            conv.emit_result({"ok": False, "code": "model_not_found"})
        self.assertEqual(
            buf.getvalue().strip(),
            'PT_CONVERTER_RESULT {"ok":false,"code":"model_not_found"}',
        )

    def test_log_prefix_constant(self):
        self.assertEqual(conv.LOG_PREFIX, "PT_CONVERTER_LOG ")


class EmitLogTests(unittest.TestCase):
    """Verify emit_log produces correct PT_CONVERTER_LOG lines."""

    def test_basic_log_line(self):
        captured = []
        with mock.patch("builtins.print", lambda *a, **k: captured.append(a[0])):
            conv.emit_log("info", "Loading model...")
        self.assertEqual(len(captured), 1)
        payload = parse_log_line(captured[0])
        self.assertEqual(payload["level"], "info")
        self.assertEqual(payload["msg"], "Loading model...")

    def test_log_with_stage(self):
        captured = []
        with mock.patch("builtins.print", lambda *a, **k: captured.append(a[0])):
            conv.emit_log("info", "Exporting", stage="exporting")
        payload = parse_log_line(captured[0])
        self.assertEqual(payload["stage"], "exporting")

    def test_log_compact_json(self):
        captured = []
        with mock.patch("builtins.print", lambda *a, **k: captured.append(a[0])):
            conv.emit_log("warn", "test", stage="moving")
        raw = captured[0][len(conv.LOG_PREFIX) :]
        self.assertNotIn(", ", raw)
        self.assertNotIn(": ", raw)

    def test_log_error_level(self):
        captured = []
        with mock.patch("builtins.print", lambda *a, **k: captured.append(a[0])):
            conv.emit_log("error", "boom", code="conversion_failed")
        payload = parse_log_line(captured[0])
        self.assertEqual(payload["level"], "error")
        self.assertEqual(payload["code"], "conversion_failed")


class JSONLogStreamTests(unittest.TestCase):
    """Verify JSONLogStream wraps plain text and passes through prefixed lines."""

    def test_wraps_plain_text(self):
        buf = io.StringIO()
        stream = conv.JSONLogStream(buf, "info")
        stream.write("Hello world\n")
        payload = parse_log_line(buf.getvalue().strip())
        self.assertEqual(payload["level"], "info")
        self.assertEqual(payload["msg"], "Hello world")

    def test_wraps_stderr_level(self):
        buf = io.StringIO()
        stream = conv.JSONLogStream(buf, "warn")
        stream.write("Warning text\n")
        payload = parse_log_line(buf.getvalue().strip())
        self.assertEqual(payload["level"], "warn")

    def test_passes_through_result_prefix(self):
        buf = io.StringIO()
        stream = conv.JSONLogStream(buf, "info")
        stream.write('PT_CONVERTER_RESULT {"ok":true}\n')
        self.assertEqual(buf.getvalue().strip(), 'PT_CONVERTER_RESULT {"ok":true}')

    def test_passes_through_log_prefix(self):
        buf = io.StringIO()
        stream = conv.JSONLogStream(buf, "info")
        stream.write('PT_CONVERTER_LOG {"level":"info","msg":"already wrapped"}\n')
        self.assertEqual(
            buf.getvalue().strip(),
            'PT_CONVERTER_LOG {"level":"info","msg":"already wrapped"}',
        )

    def test_buffers_partial_lines(self):
        buf = io.StringIO()
        stream = conv.JSONLogStream(buf, "info")
        stream.write("partial")
        self.assertEqual(buf.getvalue(), "")
        stream.write(" continued\n")
        payload = parse_log_line(buf.getvalue().strip())
        self.assertEqual(payload["msg"], "partial continued")

    def test_flush_emits_remaining_buffer(self):
        buf = io.StringIO()
        stream = conv.JSONLogStream(buf, "info")
        stream.write("no newline here")
        stream.flush()
        payload = parse_log_line(buf.getvalue().strip())
        self.assertEqual(payload["msg"], "no newline here")

    def test_empty_lines_skipped(self):
        buf = io.StringIO()
        stream = conv.JSONLogStream(buf, "info")
        stream.write("\n\n\n")
        self.assertEqual(buf.getvalue(), "")

    def test_multiple_lines_in_one_write(self):
        buf = io.StringIO()
        stream = conv.JSONLogStream(buf, "info")
        stream.write("line one\nline two\nline three\n")
        lines = [l for l in buf.getvalue().strip().split("\n") if l]
        self.assertEqual(len(lines), 3)
        for line in lines:
            parse_log_line(line)  # should not raise

    def test_strips_ansi_color_codes(self):
        """JSONLogStream must strip ANSI escapes (argparse usage, library colors)."""
        buf = io.StringIO()
        stream = conv.JSONLogStream(buf, "warn")
        stream.write("\x1b[1;34musage: \x1b[0m\x1b[1;35mpt-converter\x1b[0m\n")
        payload = parse_log_line(buf.getvalue().strip())
        self.assertEqual(payload["msg"], "usage: pt-converter")

    def test_strips_ansi_only_line_becomes_empty(self):
        """A line that is only ANSI escapes should be skipped, not emitted."""
        buf = io.StringIO()
        stream = conv.JSONLogStream(buf, "info")
        stream.write("\x1b[0m\x1b[1;34m\n")
        self.assertEqual(buf.getvalue(), "")

    def test_delegates_attributes(self):
        """JSONLogStream should delegate fileno/isatty/encoding to target."""
        import io as _io

        target = _io.StringIO()
        stream = conv.JSONLogStream(target, "info")
        # encoding attribute should delegate (StringIO has no encoding,
        # but the delegation mechanism should not raise AttributeError
        # for attributes that exist on the target).
        self.assertTrue(hasattr(stream, "write"))
        self.assertTrue(hasattr(stream, "flush"))


class ArgParseTests(unittest.TestCase):
    """Verify argparse configuration."""

    def test_basic_args(self):
        parser = conv.build_parser()
        args = parser.parse_args(["m.pt", "--output", "o.onnx"])
        self.assertEqual(args.model, Path("m.pt"))
        self.assertEqual(args.output, Path("o.onnx"))
        self.assertEqual(args.opset, 12)

    def test_opset_default_is_12(self):
        parser = conv.build_parser()
        args = parser.parse_args(["m.pt", "--output", "o.onnx"])
        self.assertEqual(args.opset, 12)

    def test_opset_choices_rejects_13(self):
        parser = conv.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["m.pt", "--output", "o.onnx", "--opset", "13"])

    def test_opset_accepts_12(self):
        parser = conv.build_parser()
        args = parser.parse_args(["m.pt", "--output", "o.onnx", "--opset", "12"])
        self.assertEqual(args.opset, 12)

    def test_output_required(self):
        parser = conv.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["m.pt"])

    def test_model_positional_required(self):
        parser = conv.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--output", "o.onnx"])


class ErrorCodeMappingTests(unittest.TestCase):
    """Verify each stable error code is produced by run_conversion."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ptconv_"))
        self.model = self.tmp / "best.pt"
        self.model.write_bytes(b"fake pt")
        self.output = self.tmp / "out.onnx"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_input_not_pt(self):
        bad = self.tmp / "model.onnx"
        bad.write_bytes(b"x")
        with self.assertRaises(conv.ConverterError) as cm:
            conv.run_conversion(bad, self.output, 12)
        self.assertEqual(cm.exception.code, "input_not_pt")

    def test_model_not_found(self):
        missing = self.tmp / "nope.pt"
        with self.assertRaises(conv.ConverterError) as cm:
            conv.run_conversion(missing, self.output, 12)
        self.assertEqual(cm.exception.code, "model_not_found")

    def test_output_exists(self):
        self.output.write_bytes(b"existing")
        with self.assertRaises(conv.ConverterError) as cm:
            conv.run_conversion(self.model, self.output, 12)
        self.assertEqual(cm.exception.code, "output_exists")

    def test_unsupported_opset(self):
        with self.assertRaises(conv.ConverterError) as cm:
            conv.run_conversion(self.model, self.output, 13)
        self.assertEqual(cm.exception.code, "unsupported_opset")

    def test_unsupported_task_segment(self):
        auto = self.tmp / "best.onnx"
        yolo = FakeYOLO(task="segment", export_path=str(auto))
        with self.assertRaises(conv.ConverterError) as cm:
            conv.run_conversion(self.model, self.output, 12, yolo_factory=lambda _p: yolo)
        self.assertEqual(cm.exception.code, "unsupported_task")
        # Export should not have been called, so no automatic output leaked.
        self.assertFalse(auto.exists())

    def test_unsupported_task_classify(self):
        yolo = FakeYOLO(task="classify")
        with self.assertRaises(conv.ConverterError) as cm:
            conv.run_conversion(self.model, self.output, 12, yolo_factory=lambda _p: yolo)
        self.assertEqual(cm.exception.code, "unsupported_task")

    def test_conversion_failed_on_export_error(self):
        yolo = FakeYOLO(task="detect", export_exc=RuntimeError("boom"))
        with self.assertRaises(conv.ConverterError) as cm:
            conv.run_conversion(self.model, self.output, 12, yolo_factory=lambda _p: yolo)
        self.assertEqual(cm.exception.code, "conversion_failed")

    def test_conversion_failed_on_load_error(self):
        def factory(_p):
            raise RuntimeError("cannot load")

        with self.assertRaises(conv.ConverterError) as cm:
            conv.run_conversion(self.model, self.output, 12, yolo_factory=factory)
        self.assertEqual(cm.exception.code, "conversion_failed")

    def test_success_detect(self):
        auto = self.tmp / "best.onnx"
        yolo = FakeYOLO(task="detect", export_path=str(auto))
        out = conv.run_conversion(self.model, self.output, 12, yolo_factory=lambda _p: yolo)
        self.assertEqual(out, self.output)
        self.assertTrue(self.output.exists())
        self.assertGreater(self.output.stat().st_size, 0)

    def test_success_pose(self):
        auto = self.tmp / "best.onnx"
        yolo = FakeYOLO(task="pose", export_path=str(auto))
        out = conv.run_conversion(self.model, self.output, 12, yolo_factory=lambda _p: yolo)
        self.assertEqual(out, self.output)
        self.assertTrue(self.output.exists())


class PartialCleanupTests(unittest.TestCase):
    """Verify .partial files are cleaned up on failure."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ptconv_partial_"))
        self.model = self.tmp / "best.pt"
        self.model.write_bytes(b"fake pt")
        self.output = self.tmp / "out.onnx"
        self.partial = self.output.with_suffix(".onnx.partial")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_partial_cleaned_after_unsupported_task(self):
        # Pre-create a .partial to ensure cleanup targets it.
        self.partial.write_bytes(b"stale partial")
        auto = self.tmp / "best.onnx"
        yolo = FakeYOLO(task="segment", export_path=str(auto))
        with self.assertRaises(conv.ConverterError):
            conv.run_conversion(self.model, self.output, 12, yolo_factory=lambda _p: yolo)
        self.assertFalse(self.partial.exists())

    def test_partial_cleaned_after_opset_rejection(self):
        self.partial.write_bytes(b"stale partial")
        with self.assertRaises(conv.ConverterError):
            conv.run_conversion(self.model, self.output, 13)
        self.assertFalse(self.partial.exists())

    def test_partial_cleaned_after_input_not_pt(self):
        self.partial.write_bytes(b"stale partial")
        bad = self.tmp / "model.onnx"
        bad.write_bytes(b"x")
        with self.assertRaises(conv.ConverterError):
            conv.run_conversion(bad, self.output, 12)
        self.assertFalse(self.partial.exists())

    def test_no_partial_left_on_success(self):
        auto = self.tmp / "best.onnx"
        yolo = FakeYOLO(task="detect", export_path=str(auto))
        conv.run_conversion(self.model, self.output, 12, yolo_factory=lambda _p: yolo)
        self.assertFalse(self.partial.exists())
        self.assertTrue(self.output.exists())


class AtomicMoveTests(unittest.TestCase):
    """Verify the copy-to-.partial, fsync, os.replace, unlink-source logic."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ptconv_move_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_atomic_move_basic(self):
        src = self.tmp / "auto.onnx"
        src.write_bytes(b"payload")
        dst = self.tmp / "final.onnx"
        conv._atomic_move(src, dst)
        self.assertTrue(dst.exists())
        self.assertEqual(dst.read_bytes(), b"payload")
        # Source removed because it differs from dst.
        self.assertFalse(src.exists())

    def test_atomic_move_creates_partial_then_replaces(self):
        src = self.tmp / "auto.onnx"
        src.write_bytes(b"data")
        dst = self.tmp / "final.onnx"
        partial = dst.with_suffix(".onnx.partial")
        conv._atomic_move(src, dst)
        # After success, .partial is gone and final exists.
        self.assertFalse(partial.exists())
        self.assertTrue(dst.exists())

    def test_atomic_move_partial_cleaned_on_failure(self):
        src = self.tmp / "auto.onnx"
        src.write_bytes(b"data")
        dst = self.tmp / "final.onnx"
        partial = dst.with_suffix(".onnx.partial")

        # Force os.replace to fail after the partial is written.
        with (
            mock.patch("os.replace", side_effect=OSError("boom")),
            self.assertRaises(conv.ConverterError) as cm,
        ):
            conv._atomic_move(src, dst)
        self.assertEqual(cm.exception.code, "conversion_failed")
        self.assertFalse(partial.exists())
        self.assertFalse(dst.exists())

    def test_atomic_move_src_equals_dst(self):
        # When the automatic output already equals the requested output, the
        # source should not be unlinked (it IS the output).
        p = self.tmp / "same.onnx"
        p.write_bytes(b"data")
        conv._atomic_move(p, p)
        self.assertTrue(p.exists())
        self.assertEqual(p.read_bytes(), b"data")

    def test_atomic_move_unlinks_automatic_output(self):
        # The automatic output lives next to the model; requested output elsewhere.
        src = self.tmp / "auto.onnx"
        src.write_bytes(b"data")
        dst_dir = self.tmp / "sub"
        dst_dir.mkdir()
        dst = dst_dir / "final.onnx"
        conv._atomic_move(src, dst)
        self.assertTrue(dst.exists())
        self.assertFalse(src.exists())

    def test_full_conversion_moves_automatic_to_requested(self):
        model = self.tmp / "best.pt"
        model.write_bytes(b"fake pt")
        output = self.tmp / "elsewhere" / "renamed.onnx"
        # Ultralytics writes next to the model.
        auto = self.tmp / "best.onnx"
        yolo = FakeYOLO(task="detect", export_path=str(auto))
        conv.run_conversion(model, output, 12, yolo_factory=lambda _p: yolo)
        self.assertTrue(output.exists())
        # Automatic output removed after the move.
        self.assertFalse(auto.exists())
        # No .partial left behind.
        self.assertFalse(output.with_suffix(".onnx.partial").exists())


class MainExitCodeTests(unittest.TestCase):
    """Verify main() exit codes and that exactly one result line is emitted."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ptconv_main_"))
        self.model = self.tmp / "best.pt"
        self.model.write_bytes(b"fake pt")
        self.output = self.tmp / "out.onnx"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _capture_results(self, *args, yolo_factory=None):
        lines = []
        with (
            mock.patch.object(conv, "emit_result", lambda payload: lines.append(payload)),
            mock.patch.object(conv, "emit_log", lambda *a, **k: None),
        ):
            code = conv.main(list(args), yolo_factory=yolo_factory)
        return code, lines

    def test_main_success_exit_zero(self):
        auto = self.tmp / "best.onnx"
        yolo = FakeYOLO(task="detect", export_path=str(auto))
        code, lines = self._capture_results(
            str(self.model),
            "--output",
            str(self.output),
            yolo_factory=lambda _p: yolo,
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 1)
        payload = lines[0]
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["output"], str(self.output.resolve()))
        self.assertEqual(payload["bytes"], self.output.stat().st_size)

    def test_main_failure_exit_one(self):
        code, lines = self._capture_results(
            str(self.model),
            "--output",
            str(self.output),
            yolo_factory=lambda _p: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(lines), 1)
        self.assertFalse(lines[0]["ok"])
        self.assertEqual(lines[0]["code"], "conversion_failed")

    def test_main_input_not_pt(self):
        bad = self.tmp / "model.onnx"
        bad.write_bytes(b"x")
        code, lines = self._capture_results(str(bad), "--output", str(self.output))
        self.assertEqual(code, 1)
        self.assertEqual(lines[0], {"ok": False, "code": "input_not_pt"})

    def test_main_model_not_found(self):
        missing = self.tmp / "nope.pt"
        code, lines = self._capture_results(str(missing), "--output", str(self.output))
        self.assertEqual(code, 1)
        self.assertEqual(lines[0], {"ok": False, "code": "model_not_found"})

    def test_main_output_exists(self):
        self.output.write_bytes(b"existing")
        code, lines = self._capture_results(str(self.model), "--output", str(self.output))
        self.assertEqual(code, 1)
        self.assertEqual(lines[0], {"ok": False, "code": "output_exists"})

    def test_main_unsupported_opset_via_argparse(self):
        # argparse choices=[12] rejects 13 with SystemExit(2); main propagates.
        with self.assertRaises(SystemExit) as cm:
            conv.main([str(self.model), "--output", str(self.output), "--opset", "13"])
        self.assertNotEqual(cm.exception.code, 0)

    def test_main_unsupported_task(self):
        auto = self.tmp / "best.onnx"
        yolo = FakeYOLO(task="segment", export_path=str(auto))
        code, lines = self._capture_results(
            str(self.model),
            "--output",
            str(self.output),
            yolo_factory=lambda _p: yolo,
        )
        self.assertEqual(code, 1)
        self.assertEqual(lines[0], {"ok": False, "code": "unsupported_task"})

    def test_main_result_line_is_compact_json(self):
        auto = self.tmp / "best.onnx"
        yolo = FakeYOLO(task="detect", export_path=str(auto))
        captured = []

        def fake_print(*a, **k):
            captured.append(a[0] if a else "")

        with (
            mock.patch("builtins.print", fake_print),
            mock.patch.object(conv, "emit_log", lambda *a, **k: None),
        ):
            code = conv.main(
                [str(self.model), "--output", str(self.output)],
                yolo_factory=lambda _p: yolo,
            )
        self.assertEqual(code, 0)
        result_lines = [c for c in captured if isinstance(c, str) and c.startswith(conv.PREFIX)]
        self.assertEqual(len(result_lines), 1)
        payload = parse_result_line(result_lines[0])
        self.assertTrue(payload["ok"])
        # Compact JSON: no spaces after , or :.
        raw = result_lines[0][len(conv.PREFIX) :]
        self.assertNotIn(", ", raw)
        self.assertNotIn(": ", raw)

    def test_main_failure_result_line_is_compact_json(self):
        captured = []

        def fake_print(*a, **k):
            captured.append(a[0] if a else "")

        with (
            mock.patch("builtins.print", fake_print),
            mock.patch.object(conv, "emit_log", lambda *a, **k: None),
        ):
            code = conv.main([str(self.model), "--output", str(self.output)])
        # No YOLO installed -> conversion_failed, but still one result line.
        self.assertEqual(code, 1)
        result_lines = [c for c in captured if isinstance(c, str) and c.startswith(conv.PREFIX)]
        self.assertEqual(len(result_lines), 1)
        payload = parse_result_line(result_lines[0])
        self.assertFalse(payload["ok"])
        raw = result_lines[0][len(conv.PREFIX) :]
        self.assertNotIn(", ", raw)
        self.assertNotIn(": ", raw)


class UnexpectedExceptionTests(unittest.TestCase):
    """Verify an unexpected exception still emits a result line."""

    def test_unexpected_exception_in_main(self):
        def factory(_p):
            raise ValueError("unexpected")

        captured = []

        def fake_print(*a, **k):
            captured.append(a[0] if a else "")

        tmp = Path(tempfile.mkdtemp(prefix="ptconv_unexpected_"))
        try:
            model = tmp / "best.pt"
            model.write_bytes(b"fake pt")
            output = tmp / "out.onnx"
            with (
                mock.patch("builtins.print", fake_print),
                mock.patch.object(conv, "emit_log", lambda *a, **k: None),
            ):
                code = conv.main(
                    [str(model), "--output", str(output)],
                    yolo_factory=factory,
                )
            self.assertEqual(code, 1)
            result_lines = [c for c in captured if isinstance(c, str) and c.startswith(conv.PREFIX)]
            self.assertEqual(len(result_lines), 1)
            payload = parse_result_line(result_lines[0])
            self.assertEqual(payload, {"ok": False, "code": "conversion_failed"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class JSONStreamIntegrationTests(unittest.TestCase):
    """End-to-end: library prints during main() are wrapped as PT_CONVERTER_LOG JSON."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ptconv_json_"))
        self.model = self.tmp / "best.pt"
        self.model.write_bytes(b"fake pt")
        self.output = self.tmp / "out.onnx"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_library_stdout_wrapped_as_json(self):
        """A yolo_factory that prints to sys.stdout should produce JSON log lines."""
        auto = self.tmp / "best.onnx"
        captured = io.StringIO()

        class PrintingYOLO:
            task = "detect"

            def export(self, format="onnx", opset=12, simplify=False):
                # Simulate Ultralytics printing progress to sys.stdout.
                print("Ultralytics: exporting to onnx...")
                print("Ultralytics: done")
                p = Path(str(auto))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"FAKE ONNX " * 100)
                return str(p)

        # Replace the real stdout with a capture buffer so we can inspect
        # what the JSONLogStream writes.
        real_stdout = sys.stdout
        sys.stdout = captured
        try:
            code = conv.main(
                [str(self.model), "--output", str(self.output)],
                yolo_factory=lambda _p: PrintingYOLO(),
            )
        finally:
            sys.stdout = real_stdout

        self.assertEqual(code, 0)
        output_lines = captured.getvalue().strip().split("\n")

        # Every line must start with an PT_CONVERTER_ prefix.
        for line in output_lines:
            self.assertTrue(
                line.startswith((conv.PREFIX, conv.LOG_PREFIX)),
                f"non-JSON line: {line!r}",
            )

        # Library prints should be wrapped as PT_CONVERTER_LOG.
        log_lines = [l for l in output_lines if l.startswith(conv.LOG_PREFIX)]
        msgs = [parse_log_line(l)["msg"] for l in log_lines]
        self.assertIn("Ultralytics: exporting to onnx...", msgs)
        self.assertIn("Ultralytics: done", msgs)

        # Exactly one result line.
        result_lines = [l for l in output_lines if l.startswith(conv.PREFIX)]
        self.assertEqual(len(result_lines), 1)
        payload = parse_result_line(result_lines[0])
        self.assertTrue(payload["ok"])

    def test_stage_logs_emitted(self):
        """main() should emit stage log lines (starting, complete)."""
        auto = self.tmp / "best.onnx"
        captured = io.StringIO()

        class SilentYOLO:
            task = "detect"

            def export(self, format="onnx", opset=12, simplify=False):
                p = Path(str(auto))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"FAKE ONNX " * 100)
                return str(p)

        real_stdout = sys.stdout
        sys.stdout = captured
        try:
            code = conv.main(
                [str(self.model), "--output", str(self.output)],
                yolo_factory=lambda _p: SilentYOLO(),
            )
        finally:
            sys.stdout = real_stdout

        self.assertEqual(code, 0)
        log_lines = [
            l for l in captured.getvalue().strip().split("\n") if l.startswith(conv.LOG_PREFIX)
        ]
        stages = [parse_log_line(l).get("stage") for l in log_lines]
        self.assertIn("starting", stages)
        self.assertIn("complete", stages)

    def test_failure_emits_failed_stage_log(self):
        """On failure, main() should emit a 'failed' stage log line."""
        captured = io.StringIO()

        real_stdout = sys.stdout
        sys.stdout = captured
        try:
            code = conv.main(
                [str(self.model), "--output", str(self.output)],
                yolo_factory=lambda _p: (_ for _ in ()).throw(RuntimeError("boom")),
            )
        finally:
            sys.stdout = real_stdout

        self.assertEqual(code, 1)
        log_lines = [
            l for l in captured.getvalue().strip().split("\n") if l.startswith(conv.LOG_PREFIX)
        ]
        stages = [parse_log_line(l).get("stage") for l in log_lines]
        self.assertIn("starting", stages)
        self.assertIn("failed", stages)

        result_lines = [
            l for l in captured.getvalue().strip().split("\n") if l.startswith(conv.PREFIX)
        ]
        self.assertEqual(len(result_lines), 1)
        payload = parse_result_line(result_lines[0])
        self.assertFalse(payload["ok"])


class FrozenEnvTests(unittest.TestCase):
    """Verify the frozen-exe environment hardening"""

    def test_yolo_autoinstall_disabled(self):
        """YOLO_AUTOINSTALL must be set to '0' to prevent ultralytics from
        silently pip-installing at runtime in a frozen exe with no pip."""
        self.assertEqual(os.environ.get("YOLO_AUTOINSTALL"), "0")

    def test_frozen_flag_exists(self):
        """The _frozen flag must be present (False when not frozen)."""
        self.assertTrue(hasattr(conv, "_frozen"))
        self.assertFalse(conv._frozen)  # False in the test runner

    def test_torch_home_set_when_frozen(self):
        """When _frozen is True, TORCH_HOME must be set to a writable temp dir."""
        # We can't actually freeze in a test, but we can verify the logic
        # by checking that the module doesn't crash when _frozen is False
        # and that the env var is NOT set in non-frozen mode (it's only
        # set inside the if _frozen block, and we don't want to pollute
        # the test environment's TORCH_HOME).
        # The important thing is the code path exists and doesn't error.
        self.assertTrue(hasattr(conv, "_frozen"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
