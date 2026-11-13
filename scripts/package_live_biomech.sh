#!/usr/bin/env bash
# Build the root binary in release mode and package the live split view as a local macOS app:
#   output/Live Biomech.app/Contents/MacOS/live-biomech        (native copy of lifting-pose; auto-enables live mode)
#   output/Live Biomech.app/Contents/Resources/models/{mediapipe-pose,yolox-s,rtmpose-m,handpose,palmdet,motionagformer-xs}.onnx
#   output/Live Biomech.app/Contents/Resources/assets/biomech-lite/{model,visual}.json, NOTICE.md, provenance/
# The executable is a copy of the real binary, not a shell wrapper, so macOS camera permission
# is granted to this app's bundle identity. No recording, no install, no global settings.
# Dynamic libraries referenced via @rpath (for example ONNX Runtime) are copied into
# Contents/Frameworks and re-pointed. Homebrew OpenCV is NOT bundled: it stays a dependency of
# this Mac. An existing bundle is moved aside, never overwritten.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ASSETS="$ROOT/assets/biomech-lite"
OUT="$ROOT/output"
APP="$OUT/Live Biomech.app"
VERSION="0.1.0"
BIN="$ROOT/target/release/lifting-pose"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "package_live_biomech: macOS only (camera permission and .app bundle)" >&2
  exit 1
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "package_live_biomech: cargo not found; install the Rust toolchain (https://rustup.rs)" >&2
  exit 1
fi

cargo build --release --manifest-path "$ROOT/Cargo.toml" --bin lifting-pose
[ -x "$BIN" ] || { echo "package_live_biomech: built binary missing: $BIN" >&2; exit 1; }

for f in "$ASSETS/model.json" "$ASSETS/visual.json" "$ASSETS/NOTICE.md" \
         "$ROOT/models/mediapipe-pose.onnx" "$ROOT/models/yolox-s.onnx" \
         "$ROOT/models/rtmpose-m.onnx" "$ROOT/models/handpose.onnx" \
         "$ROOT/models/palmdet.onnx" "$ROOT/models/motionagformer-xs.onnx"; do
  [ -f "$f" ] || { echo "package_live_biomech: missing $f" >&2; exit 1; }
done
[ -d "$ASSETS/provenance" ] || { echo "package_live_biomech: missing $ASSETS/provenance" >&2; exit 1; }

mkdir -p "$OUT"
if [ -e "$APP" ]; then
  STAMP="$(date +%Y%m%d-%H%M%S)"
  mv "$APP" "$APP.previous-$STAMP"
  echo "moved existing bundle to: $APP.previous-$STAMP"
fi

# Stage in a temporary directory, then move into place, so a failed copy never leaves a half bundle.
STAGE="$(mktemp -d "$OUT/.live-biomech-stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
STAGED_APP="$STAGE/Live Biomech.app"
MACOS="$STAGED_APP/Contents/MacOS"
RES="$STAGED_APP/Contents/Resources"
FRAMEWORKS="$STAGED_APP/Contents/Frameworks"
mkdir -p "$MACOS" "$RES/models" "$RES/assets/biomech-lite" "$FRAMEWORKS"

EXE="$MACOS/live-biomech"
cp "$BIN" "$EXE"
chmod 755 "$EXE"

cp "$ROOT/models/"{mediapipe-pose,yolox-s,rtmpose-m,handpose,palmdet,motionagformer-xs}.onnx "$RES/models/"
cp "$ASSETS/model.json" "$ASSETS/visual.json" "$ASSETS/NOTICE.md" "$RES/assets/biomech-lite/"
cp -R "$ASSETS/provenance" "$RES/assets/biomech-lite/provenance"

# Copy @rpath dylibs (ONNX Runtime via ort copy-dylibs lands next to the release binary) and
# point the executable at the copies. Absolute non-system paths (Homebrew OpenCV) are reported.
ort_search_dirs=("$ROOT/target/release" "$ROOT/target/release/deps")
changed_exe=0
while IFS= read -r dep; do
  dep="${dep%% *}"
  case "$dep" in
    @rpath/*)
      name="${dep#@rpath/}"
      source=""
      for dir in "${ort_search_dirs[@]}"; do
        if [ -f "$dir/$name" ]; then source="$dir/$name"; break; fi
      done
      if [ -z "$source" ]; then
        echo "package_live_biomech: cannot locate @rpath dependency $name" >&2
        exit 1
      fi
      cp "$source" "$FRAMEWORKS/$name"
      chmod 644 "$FRAMEWORKS/$name"
      install_name_tool -id "@executable_path/../Frameworks/$name" "$FRAMEWORKS/$name"
      install_name_tool -change "$dep" "@executable_path/../Frameworks/$name" "$EXE"
      changed_exe=1
      ;;
    /usr/lib/*|/System/*) ;;
    *) echo "note: $EXE depends on $dep (kept as a local machine dependency; not bundled)" ;;
  esac
done < <(otool -L "$EXE" | tail -n +2 | sed -E 's/^[[:space:]]+//; s/ \(compatibility.*$//')

if [ "$changed_exe" -eq 1 ]; then
  # install_name_tool invalidates signatures on arm64; re-sign ad hoc so the copies load.
  for lib in "$FRAMEWORKS"/*.dylib; do
    [ -f "$lib" ] && codesign --force --sign - "$lib" >/dev/null
  done
  codesign --force --sign - "$EXE" >/dev/null
fi

cat > "$STAGED_APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>local.liftingpose.live-biomech</string>
  <key>CFBundleName</key><string>Live Biomech</string>
  <key>CFBundleDisplayName</key><string>Live Biomech</string>
  <key>CFBundleExecutable</key><string>live-biomech</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSCameraUsageDescription</key><string>Live Biomech uses the camera to estimate your pose locally on this Mac for the live anatomical view. Frames are processed in memory and are not recorded, saved or sent anywhere.</string>
</dict>
</plist>
PLIST

# Seal the whole bundle last (Info.plist + resources), always, with the stable identifier,
# so NSCameraUsageDescription is bound to the signature macOS checks for camera access.
codesign --force --deep --sign - --identifier local.liftingpose.live-biomech "$STAGED_APP" >/dev/null
codesign --verify --deep --strict "$STAGED_APP"

mv "$STAGED_APP" "$APP"
echo "packaged: $APP"
echo "run: open \"$APP\"  (camera prompt appears on first use; no recording or uploads)"
