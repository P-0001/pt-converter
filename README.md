# pt-converter — Standalone `.pt` → `.onnx` Converter

A self-contained Windows x64 executable that converts a YOLO `.pt`
checkpoint to an ONNX model using Ultralytics + PyTorch (CPU). It is built
with PyInstaller so the end user needs **no Python installed**.

## Prerequisites

- **Windows x64**
- **uv** on PATH — manages the Python toolchain, venv, and pinned
  dependencies (declared in `pyproject.toml`). Only needed on the *build*
  machine, not on machines that run the finished `pt-converter.exe`.
- PowerShell 7+ (`pwsh`) on PATH.

## Build

From the repository root:

```powershell
pwsh -ExecutionPolicy Bypass -File scripts\build.ps1
```

`build.ps1` runs `uv sync` to create/update the project venv and install
the pinned runtime dependencies plus the `dev` dependency-group
(pyinstaller, pytest). The PyTorch CPU wheels are resolved via the
`pytorch-cpu` index declared in `pyproject.toml` (`[[tool.uv.index]]` +
`[tool.uv.sources]`), so no extra flags are needed. It then runs the unit
tests and invokes PyInstaller with `pt-converter.spec`.

To run only the tests (no build):

```powershell
uv sync
uv run pytest tests -v
```

## CI and releases

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `master`,
every PR, and every `v*` tag:

- **Test job** — `uv sync` + lint, format checks, and tests on
  `windows-latest`. Runs on all events.
- **Build job** — runs `scripts\build.ps1` to produce `pt-converter.exe`
  and its SHA-256 file. Gated to `master` pushes and tags only because the
  PyInstaller bundle takes 10+ minutes and pulls ~500 MB of wheels.
- **GitHub Release** — a `v*` tag creates a release with generated notes and
  attaches `pt-converter.exe` plus `pt-converter.exe.sha256`. Re-running the
  tag workflow replaces the release assets instead of creating duplicates.

To publish a release, create and push a version tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

## Output

- **Location:** `dist/pt-converter.exe`
- **Expected size:** ~500 MB, possibly >1 GB depending on the pinned Torch
  version and packaging. Publish the measured size in the release notes.
- **Cold start:** PyInstaller one-file mode extracts the bundle to a temp
  directory on every launch, so the first invocation takes several seconds
  (often 5–15s) before any conversion work begins. This is a documented
  trade-off of the single-file artifact; if it proves worse than the
  convenience, the fallback is a signed one-directory build shipped as a ZIP
  (the Go orchestration contract is unchanged).
- After building, compute the SHA-256 for the Go release manifest:

  ```powershell
  (Get-FileHash 'dist/pt-converter.exe' -Algorithm SHA256).Hash.ToLower()
  ```

## CLI contract

```text
pt-converter.exe MODEL.pt --output MODEL.onnx --opset 12
```

- Accepts absolute paths for `MODEL.pt` and `--output`.
- Rejects a missing model, non-`.pt` input, unsupported opset, or an
  existing output (Go chooses a non-colliding destination first).
- Exports FP32 detection and pose models only.
- **ALL output is JSON-parseable.** Every line on stdout/stderr is prefixed
  with one of:
  - `PT_CONVERTER_LOG ` — log/progress lines (library output + stage messages)
  - `PT_CONVERTER_RESULT ` — exactly one final result line
- Library output (Ultralytics, PyTorch) is automatically wrapped as
  `PT_CONVERTER_LOG {"level":"info|warn","msg":"<text>"}`. ANSI color
  escapes (e.g. argparse usage, library color output) are stripped so
  `msg` values are always plain text.
- Stage messages include a `"stage"` field: `starting`, `complete`, `failed`.
- Exit code `0` only after export completes and the output exists; nonzero
  on failure with a stable error code in the result JSON.

Log line format:

```text
PT_CONVERTER_LOG {"level":"info","msg":"Ultralytics: exporting to onnx..."}
PT_CONVERTER_LOG {"level":"info","msg":"Starting conversion","stage":"starting"}
```

Result line format (success):

```text
PT_CONVERTER_RESULT {"ok":true,"output":"C:\\\\path\\\\to\\\\MODEL.onnx","bytes":12345678}
```

Result line format (failure):

```text
PT_CONVERTER_RESULT {"ok":false,"code":"input_not_pt"}
```

## Security warning

A `.pt` file is a **Python pickle**. Loading an untrusted checkpoint can
execute arbitrary code with the current user's permissions. Compiling the
converter into an `.exe` does **not** remove that risk — it only removes the
Python installation requirement.

- Only convert models from a trusted source.
- The app must show a warning before conversion and must never auto-convert
  a downloaded file.
- The Go backend runs the converter as a child process **without elevation**,
  never through `cmd.exe` or PowerShell, and passes each argument separately
  with `exec.CommandContext`. A subprocess is not a sandbox; stronger
  isolation is out of scope for this tool.
