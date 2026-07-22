#!/usr/bin/env bash
# Create a persistent, self-signed CODE-SIGNING identity in the login keychain.
#
# WHY: the app was ad-hoc signed, so its "designated requirement" is just its
# code hash -- which changes on EVERY rebuild. macOS ties the Accessibility (TCC)
# grant that ⌘B needs to that requirement, so every rebuild silently revoked the
# grant and the hotkey went dead (the #1 recurring annoyance). Signing with a
# STABLE identity makes the requirement "signed by THIS cert", which survives
# rebuilds, so the grant sticks.
#
# No Apple ID / Developer Program needed -- this is a local self-signed cert.
# Run ONCE. Idempotent: re-running is a no-op if the identity already exists.
set -euo pipefail

IDENTITY_CN="CPD Local Signing"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENTITY_CN"; then
  echo "[ok] identity '$IDENTITY_CN' already exists -- nothing to do."
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[*] generating self-signed code-signing certificate (10y)…"
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$TMP/key.pem" -out "$TMP/cert.pem" -days 3650 \
  -subj "/CN=$IDENTITY_CN" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=critical,codeSigning" >/dev/null 2>&1

# -legacy + SHA1 MAC: OpenSSL 3.x defaults to AES-256/SHA-256 PKCS12, which
# macOS's Security framework cannot import ("MAC verification failed"). The legacy
# 3DES/SHA1 encoding is what `security import` understands.
openssl pkcs12 -export -legacy -macalg sha1 -out "$TMP/identity.p12" \
  -inkey "$TMP/key.pem" -in "$TMP/cert.pem" \
  -name "$IDENTITY_CN" -passout pass:cpd >/dev/null 2>&1

echo "[*] importing into login keychain (pre-authorising codesign)…"
# -A pre-authorises all tools to use the key (no per-sign ACL dialog); -T names
# codesign explicitly as a belt-and-braces. Import into the already-unlocked
# login keychain does not prompt.
security import "$TMP/identity.p12" -k "$KEYCHAIN" -P cpd -A -T /usr/bin/codesign

echo "[ok] created identity '$IDENTITY_CN':"
security find-identity -v -p codesigning | grep "$IDENTITY_CN" || true
