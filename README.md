# pt-converter

Convert an Ultralytics YOLO detection or pose checkpoint (`.pt`) to ONNX on Windows x64. The single-file executable bundles Python and CPU PyTorch, so you can run it without installing Python or configuring a GPU.

## Key features

- Export detection and pose models to ONNX opset 12 with graph simplification disabled.
- Choose the output directory and filename. The converter creates missing parent directories and rejects a destination that already exists.
- Integrate with a parent process using prefixed JSON progress messages, a final result, and process exit codes.

## Usage

Download `pt-converter.exe` from [GitHub Releases](https://github.com/P-0001/pt-converter/releases). Each release build includes a SHA-256 checksum file.

**Only load trusted checkpoints.** Loading a pickled `.pt` model can execute code with the permissions of the user running the converter.

From PowerShell, with the executable and a trusted `model.pt` in the current directory, run

```powershell
.\pt-converter.exe .\model.pt --output .\exports\model.onnx
```

The positional model path and `--output` are required. Both accept relative or absolute paths. The optional `--opset 12` selects the only supported opset and is also the default. Choose a writable input directory because Ultralytics initially exports beside the checkpoint. Keep that same-stem `.onnx` path free even when requesting a different destination.

### Calling from another process

Read stdout and stderr line by line. Split each converter message at the first space and parse the remainder as JSON.

| Prefix | Payload |
| --- | --- |
| `PT_CONVERTER_LOG` | `level` and `msg`. Conversion stage messages also carry `stage` set to `starting`, `complete`, or `failed`. |
| `PT_CONVERTER_RESULT` | One result on stdout. Success includes the absolute output path and file size in bytes. Failure includes an error code. |

Example result lines from separate invocations

```text
PT_CONVERTER_RESULT {"ok":true,"output":"C:\\models\\exports\\model.onnx","bytes":12345678}
PT_CONVERTER_RESULT {"ok":false,"code":"model_not_found"}
```

Successful conversion exits with status 0. Conversion errors exit with status 1 and report `input_not_pt`, `model_not_found`, `output_exists`, `unsupported_task`, `conversion_failed`, or `cleanup_failed`. Invalid CLI arguments, including an unsupported opset, exit with status 2 and report `invalid_args`. The `--help` path prints usage and exits with status 0 but also emits an `invalid_args` result.

## Project structure

```text
pt-converter.py           CLI, validation, model export, file handling, and JSON output
pt-converter.spec         PyInstaller configuration for the single-file console executable
assets/                   Application icon and logo
pyproject.toml            Runtime and development dependencies, CPU wheel index, Ruff settings
uv.lock                   Locked dependency resolution
.python-version           Python version selected by uv
scripts/build.ps1         Windows build script
tests/test_converter.py   Converter unit tests using a stand-in for Ultralytics YOLO
.github/workflows/ci.yml  Windows checks, tagged builds, and release publication
```

After validation, the CLI loads the checkpoint with Ultralytics and checks its task. It copies the exported ONNX file to a `.partial` file beside the requested destination, flushes it to disk, and uses an atomic rename to publish the result. When the export and destination paths differ, it attempts to remove the intermediate export.

## Prerequisites and local build

Build on Windows x64 with [uv](https://docs.astral.sh/uv/) and PowerShell 7+ (`pwsh`) on your PATH. The project selects Python 3.14. From the repository root, run

```powershell
pwsh -ExecutionPolicy Bypass -File scripts\build.ps1
```

The script runs `uv sync` to install runtime and development dependencies, runs the unit tests, then packages `dist\pt-converter.exe` with PyInstaller. The project configures the CPU wheel index for Torch and TorchVision.

To run from source in the project environment, use

```powershell
uv sync
uv run python pt-converter.py .\model.pt --output .\exports\model.onnx --opset 12
```

## Configuration

Both environment variables are optional. The converter preserves values already set in the environment.

| Variable | Converter default | Required | Purpose |
| --- | --- | --- | --- |
| `YOLO_AUTOINSTALL` | `0` | No | Disables automatic dependency installation by Ultralytics. |
| `TORCH_HOME` | Packaged executable uses `pt-converter-torch` under the system temporary directory. Source runs leave it unset unless supplied. | No | Selects a writable Torch cache directory. |

## Development and testing

Run the same checks as Windows CI from the repository root.

```powershell
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest tests -v
```

The unit tests use a fake YOLO exporter to exercise validation, output handling, cleanup, and the CLI protocol without a model download. CI runs checks on pushes and pull requests targeting `master` and on pushed tags matching `v*` or `[0-9]*`. After checks pass on a matching tag, CI builds the executable and publishes it with `pt-converter.exe.sha256` as release assets.
