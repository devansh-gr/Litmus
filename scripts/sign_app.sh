#!/usr/bin/env bash
# Re-sign the built .app with the stable "CPD Local Signing" identity so the
# Accessibility grant survives rebuilds. Falls back to ad-hoc (with a warning) if
# the identity is missing, so a fresh checkout still builds a runnable app.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$REPO/build/Debug/CorticalPersuasionDecoder.app}"
ENTITLEMENTS="$REPO/CorticalPersuasionDecoder.entitlements"
IDENTITY_CN="CPD Local Signing"

[ -d "$APP" ] || { echo "[err] app not found: $APP" >&2; exit 1; }

if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENTITY_CN"; then
  SIGN="$IDENTITY_CN"
  echo "[*] signing with stable identity '$IDENTITY_CN'"
else
  SIGN="-"   # ad-hoc
  echo "[warn] '$IDENTITY_CN' not found -- signing AD-HOC. The ⌘B Accessibility"
  echo "       grant will die on the next rebuild. Run scripts/make_signing_identity.sh once."
fi

# Flat bundle (no embedded frameworks), so a single pass is enough. Hardened
# runtime + entitlements kept consistent with the Xcode build.
codesign --force --options runtime --timestamp=none \
  --entitlements "$ENTITLEMENTS" --sign "$SIGN" "$APP"

echo "[ok] signed. verifying…"
codesign --verify --verbose=2 "$APP" && \
  codesign -dv "$APP" 2>&1 | grep -E "Authority|Signature|TeamIdentifier|Identifier=" || true
