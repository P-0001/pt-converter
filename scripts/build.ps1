# build.ps1 - builds pt-converter.exe from pt-converter.py on Windows x64.
# Lives in scripts/ but operates on the project root (its parent dir).
#
# Process:
#   1. Verify uv is available.
#   2. `uv sync` - create/update the project venv and install the pinned
#      runtime deps plus the `dev` dependency-group (pyinstaller, pytest).
#      The PyTorch CPU wheels are resolved via [[tool.uv.index]] and
#      [tool.uv.sources] in pyproject.toml, so no extra flags are needed.
#   3. Run the converter unit tests under the project env.
#   4. Run PyInstaller with pt-converter.spec to produce a single
#      pt-converter.exe (one-file mode).
#   5. Verify the exe exists and report its size.
#
# Run from anywhere with:
#   pwsh -ExecutionPolicy Bypass -File scripts\build.ps1
#
# Requires PowerShell 7+ (pwsh) and uv on PATH.

# Fail fast on any error.
$ErrorActionPreference = "Stop"

# Project root is the parent of this script's directory (scripts/).
$Root     = Split-Path -Parent $PSScriptRoot
$SpecFile = Join-Path $Root "pt-converter.spec"
$DistDir  = Join-Path $Root "dist"
$ExePath  = Join-Path $DistDir "pt-converter.exe"

Write-Host "=== pt-converter build ===" -ForegroundColor Cyan
Write-Host "root: $Root"

# 1. Check uv is available.
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    throw "uv was not found on PATH. Install it from https://docs.astral.sh/uv/ and re-run build.ps1."
}
Write-Host "Found uv: $(& uv --version)"

# 2. Sync the project env (runtime deps + dev group). Must run from the
#    project root so uv finds pyproject.toml / uv.lock.
Write-Host "Syncing project environment (uv sync) ..." -ForegroundColor Yellow
Push-Location $Root
try {
    & uv sync
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed." }
} finally {
    Pop-Location
}

# 3. Run the converter unit tests.
$TestsDir = Join-Path $Root "tests"
if (Test-Path $TestsDir) {
    Write-Host "Running unit tests ..." -ForegroundColor Yellow
    & uv run pytest $TestsDir -v
    if ($LASTEXITCODE -ne 0) { throw "Unit tests failed." }
} else {
    Write-Host "No tests directory found at $TestsDir ; skipping tests." -ForegroundColor DarkYellow
}

# 4. Run PyInstaller with the spec file.
Write-Host "Building pt-converter.exe with PyInstaller ..." -ForegroundColor Yellow
& uv run pyinstaller --noconfirm --distpath $DistDir --workpath (Join-Path $Root "build") $SpecFile
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

# 5. Verify the output exe exists.
if (-not (Test-Path $ExePath)) {
    throw "Build finished but $ExePath was not produced."
}

# 6. Report the output file size.
$exeItem = Get-Item $ExePath
$sizeBytes = $exeItem.Length
$sizeMB = [math]::Round($sizeBytes / 1MB, 1)
Write-Host ""
Write-Host "=== BUILD SUCCEEDED ===" -ForegroundColor Green
Write-Host "Output: $ExePath"
Write-Host "Size:   $sizeMB MB ($sizeBytes bytes)"
Write-Host ""
Write-Host "Next: compute its SHA-256 for the Go release manifest:" -ForegroundColor Cyan
Write-Host "  (Get-FileHash '$ExePath' -Algorithm SHA256).Hash.ToLower()"
