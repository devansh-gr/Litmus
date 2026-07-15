#!/bin/bash
# Detached verifier: waits for the baseline, then runs the full hybrid and writes
# results to verify_result.txt with a DONE marker. Survives turn cycling because
# it is launched via nohup (unlike harness-tracked background bash jobs).
cd "/Users/devanshgaur/Documents/Projects/CorticalPersuasionDecoder/server" || exit 1
OUT=verify_result.txt
: > "$OUT"

until [ -f baseline.npz ] && ! pgrep -f build_baseline.py >/dev/null; do sleep 15; done
echo "baseline.npz ready: $(ls -la baseline.npz | awk '{print $5}') bytes" >> "$OUT"

FEAR='{"text":"Act now before it is too late - this deadly outbreak will devastate your family and the government is hiding it from you."}'

echo "=== /classify ===" >> "$OUT"
curl -s -m 120 -X POST http://127.0.0.1:8765/classify -H 'Content-Type: application/json' -d "$FEAR" >> "$OUT" 2>&1
echo >> "$OUT"
echo "=== /brainmap (baseline-corrected) ===" >> "$OUT"
curl -s -m 400 -X POST http://127.0.0.1:8765/brainmap -H 'Content-Type: application/json' -d "$FEAR" >> "$OUT" 2>&1
echo >> "$OUT"
echo "VERIFY_DONE" >> "$OUT"
