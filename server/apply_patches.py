"""Re-apply the local patches TRIBE v2 needs to run without CUDA (Apple Silicon).

Both patched files live in git-ignored dirs (.venv/ and vendor/), so this script
is the source of truth. Idempotent — safe to re-run.

    .venv/bin/python apply_patches.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent

PATCHES = [
    # 1. WhisperX (word timings) hard-codes float16, which ctranslate2 cannot do
    #    on CPU / Apple Silicon.
    (
        HERE / "vendor/tribev2/tribev2/eventstransforms.py",
        '        compute_type = "float16"\n',
        '        # ctranslate2 has no efficient float16 path on CPU / Apple Silicon.\n'
        '        compute_type = "float16" if device == "cuda" else "int8"\n',
    ),
    # 2. The checkpoint config bakes device="accelerate" (device_map=auto +
    #    float16) for the Llama text encoder, which hard-fails without CUDA.
    (
        HERE / ".venv/lib/python3.11/site-packages/neuralset/extractors/base.py",
        '        if self.layers != "all":\n',
        '        # Checkpoint configs bake in "accelerate"/"cuda"; on machines without\n'
        '        # CUDA (e.g. Apple Silicon) those paths hard-fail. Coerce to CPU.\n'
        '        if self.device in ("accelerate", "cuda") and not torch.cuda.is_available():\n'
        '            self.device = "cpu"\n'
        '        if self.layers != "all":\n',
    ),
]


def main() -> int:
    failed = False
    for path, old, new in PATCHES:
        if not path.exists():
            print(f"[MISS] {path} does not exist — run the install first")
            failed = True
            continue
        text = path.read_text()
        if new in text:
            print(f"[skip] already patched: {path.name}")
        elif old in text:
            path.write_text(text.replace(old, new, 1))
            print(f"[ok]   patched: {path.name}")
        else:
            print(f"[FAIL] anchor not found in {path.name} — upstream changed?")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
