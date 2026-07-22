#!/usr/bin/env bash
# One-shot build: regenerate the Xcode project, build Debug, then re-sign with the
# stable local identity so ⌘B keeps its Accessibility grant across rebuilds.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

command -v xcodegen >/dev/null || { echo "[err] xcodegen not installed (brew install xcodegen)"; exit 1; }

echo "[1/3] xcodegen generate"
xcodegen generate

echo "[2/3] xcodebuild (Debug)"
xcodebuild -project CorticalPersuasionDecoder.xcodeproj \
  -target CorticalPersuasionDecoder -configuration Debug \
  CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO build

echo "[3/3] re-sign with stable identity"
"$REPO/scripts/sign_app.sh"

echo "[done] build/Debug/CorticalPersuasionDecoder.app"
