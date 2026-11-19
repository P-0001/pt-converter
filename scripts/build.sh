#!/usr/bin/env bash
# build.sh - builds the pt-converter binary from pt-converter.py on Linux x64
# or macOS arm64. Lives in scripts/ but operates on the project root (its
# parent dir). POSIX counterpart of build.ps1.
#
# Process:
#   1. Verify uv is available.
#   2. `uv sync` - create/update the project venv and install the pinned
#      runtime deps plus the `dev` dependency-group (pyinstaller, pytest).
#   3. Run the converter unit tests under the project env.
#   4. Run PyInstaller with pt-converter.spec to produce a single
#      pt-converter binary (one-file mode).
#   5. Verify the binary exists and is executable, and report its size.
#
# Run from anywhere with:
#   bash scripts/build.sh
#
# Requires bash and uv on PATH.

# Fail fast on any error.
set -euo pipefail

# Project root is the parent of this script's directory (scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC_FILE="$ROOT/pt-converter.spec"
DIST_DIR="$ROOT/dist"
BIN_PATH="$DIST_DIR/pt-converter"

echo "=== pt-converter build ==="
echo "root: $ROOT"

# 1. Check uv is available.
if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found on PATH. Install it from https://docs.astral.sh/uv/ and re-run build.sh." >&2
    exit 1
fi
echo "Found uv: $(uv --version)"

# 2. Sync the project env (runtime deps + dev group). Must run from the
#    project root so uv finds pyproject.toml / uv.lock.
echo "Syncing project environment (uv sync) ..."
cd "$ROOT"
uv sync

# 3. Run the converter unit tests.
TESTS_DIR="$ROOT/tests"
if [ -d "$TESTS_DIR" ]; then
    echo "Running unit tests ..."
    uv run pytest tests -v
else
    echo "No tests directory found at $TESTS_DIR ; skipping tests."
fi

# 4. Run PyInstaller with the spec file.
echo "Building pt-converter with PyInstaller ..."
uv run pyinstaller --noconfirm --distpath "$DIST_DIR" --workpath "$ROOT/build" "$SPEC_FILE"

# 5. Verify the output binary exists and is executable.
if [ ! -f "$BIN_PATH" ]; then
    echo "Build finished but $BIN_PATH was not produced." >&2
    exit 1
fi
if [ ! -x "$BIN_PATH" ]; then
    echo "Build produced $BIN_PATH but it is not executable." >&2
    exit 1
fi

# 6. Report the output file size.
SIZE_BYTES="$(wc -c < "$BIN_PATH" | tr -d '[:space:]')"
SIZE_MB="$(awk -v b="$SIZE_BYTES" 'BEGIN { printf "%.1f", b / 1048576 }')"
echo
echo "=== BUILD SUCCEEDED ==="
echo "Output: $BIN_PATH"
echo "Size:   $SIZE_MB MB ($SIZE_BYTES bytes)"
echo
echo "Next: compute its SHA-256 for the Go release manifest:"
if command -v sha256sum >/dev/null 2>&1; then
    echo "  sha256sum '$BIN_PATH'"
else
    echo "  shasum -a 256 '$BIN_PATH'"
fi
