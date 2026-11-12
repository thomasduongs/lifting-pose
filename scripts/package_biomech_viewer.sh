#!/usr/bin/env bash
# Build the native viewer in release mode and package it as a self-contained macOS app:
#   output/Biomech Viewer.app/Contents/MacOS/biomech-viewer
#   output/Biomech Viewer.app/Contents/Resources/assets/biomech-lite/{model,visual,motion}.json, NOTICE.md, provenance/
# No install, no global settings, no code signing. An existing bundle is moved aside, never overwritten.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CRATE="$ROOT/crates/biomech-viewer"
ASSETS="$ROOT/assets/biomech-lite"
OUT="$ROOT/output"
APP="$OUT/Biomech Viewer.app"
VERSION="0.1.0"

if ! command -v cargo >/dev/null 2>&1; then
  echo "package_biomech_viewer: cargo not found; install the Rust toolchain (https://rustup.rs)" >&2
  exit 1
fi

cargo build --release --manifest-path "$CRATE/Cargo.toml"
BIN="$CRATE/target/release/biomech-viewer"
[ -x "$BIN" ] || { echo "package_biomech_viewer: built binary missing: $BIN" >&2; exit 1; }

for f in model.json visual.json motion.json NOTICE.md; do
  [ -f "$ASSETS/$f" ] || { echo "package_biomech_viewer: missing $ASSETS/$f" >&2; exit 1; }
done
[ -d "$ASSETS/provenance" ] || { echo "package_biomech_viewer: missing $ASSETS/provenance" >&2; exit 1; }

mkdir -p "$OUT"
if [ -e "$APP" ]; then
  STAMP="$(date +%Y%m%d-%H%M%S)"
  mv "$APP" "$APP.previous-$STAMP"
  echo "moved existing bundle to: $APP.previous-$STAMP"
fi

# Stage in a temporary directory, then move into place, so a failed copy never leaves a half bundle.
STAGE="$(mktemp -d "$OUT/.biomech-viewer-stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
STAGED_APP="$STAGE/Biomech Viewer.app"
mkdir -p "$STAGED_APP/Contents/MacOS" "$STAGED_APP/Contents/Resources/assets/biomech-lite"

cp "$BIN" "$STAGED_APP/Contents/MacOS/biomech-viewer"
chmod 755 "$STAGED_APP/Contents/MacOS/biomech-viewer"
cp "$ASSETS/model.json" "$ASSETS/visual.json" "$ASSETS/motion.json" "$ASSETS/NOTICE.md" \
   "$STAGED_APP/Contents/Resources/assets/biomech-lite/"
cp -R "$ASSETS/provenance" "$STAGED_APP/Contents/Resources/assets/biomech-lite/provenance"

cat > "$STAGED_APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>local.liftingpose.biomech-viewer</string>
  <key>CFBundleName</key><string>Biomech Viewer</string>
  <key>CFBundleDisplayName</key><string>Biomech Viewer</string>
  <key>CFBundleExecutable</key><string>biomech-viewer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

mv "$STAGED_APP" "$APP"
echo "packaged: $APP"
