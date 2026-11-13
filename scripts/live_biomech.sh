#!/usr/bin/env bash
# Build the root lifting-pose binary in release mode and open the live split view:
# anatomical model on the left, camera on the right. Nothing is recorded or saved.
# Usage:
#   scripts/live_biomech.sh                       # default webcam (--camera 0)
#   scripts/live_biomech.sh --camera 1            # another camera index
#   scripts/live_biomech.sh --input clip.mp4      # file input for testing (no webcam opened)
#   scripts/live_biomech.sh --input clip.mp4 --live-snapshot out.png   # headless single frame
# Any other flags are forwarded unchanged. See docs/live-biomech.md for controls and privacy notes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$ROOT/target/release/lifting-pose"

if ! command -v cargo >/dev/null 2>&1; then
  echo "live_biomech: cargo not found; install the Rust toolchain (https://rustup.rs)" >&2
  exit 1
fi

cargo build --release --manifest-path "$ROOT/Cargo.toml" --bin lifting-pose

# Default to the webcam only when no input file or camera index was given.
has_input=0
has_camera=0
for arg in "$@"; do
  case "$arg" in
    --input|--input=*) has_input=1 ;;
    --camera|--camera=*) has_camera=1 ;;
  esac
done
args=("$@")
if [ "$has_input" -eq 0 ] && [ "$has_camera" -eq 0 ]; then
  args+=(--camera 0)
fi

# Run from the project root so relative model and asset paths resolve the same way every time.
cd "$ROOT"
exec "$BIN" --live-biomech "${args[@]}"
