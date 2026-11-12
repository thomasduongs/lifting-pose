#!/usr/bin/env bash
# Build the native biomech viewer in release mode and run it with the supplied arguments.
# Default (no arguments): opens the recorded-motion replay. Use --simulation for the live arm
# simulator, --snapshot PATH for a headless PNG. See docs/rust-viewer.md for all options.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CRATE="$ROOT/crates/biomech-viewer"
BIN="$CRATE/target/release/biomech-viewer"

if ! command -v cargo >/dev/null 2>&1; then
  echo "view_biomech: cargo not found; install the Rust toolchain (https://rustup.rs)" >&2
  exit 1
fi

cargo build --release --manifest-path "$CRATE/Cargo.toml"

# Run from the project root so relative asset paths resolve the same way every time.
cd "$ROOT"
exec "$BIN" "$@"
