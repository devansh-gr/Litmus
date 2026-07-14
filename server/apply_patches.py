"""Re-apply the local patches TRIBE v2 needs to run without CUDA (Apple Silicon).

The patched files live in git-ignored dirs (.venv/, vendor/), so this script is
the source of truth. Idempotent — safe to re-run.

    .venv/bin/python apply_patches.py

Why each patch exists:
  1. WhisperX (word timings) hard-codes float16; ctranslate2 has no efficient
     float16 path on CPU / Apple Silicon.
  2. neuralset needs `os` imported for the device-routing patch (4).
  3. "mps" is not in the device Literal, so pydantic rejects it.
  4. Checkpoint configs bake in device="accelerate"/"cuda", which hard-fails
     without CUDA. Route per-extractor:
       - HuggingFaceText (Llama): MPS has an mps_matmul shape bug that kills the
         process ("Failed to infer result type"). Keep on CPU (~10s anyway).
       - Audio/other encoders: MPS is a huge win (4m16s on CPU -> 3s on MPS).
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
TRIBE = HERE / "vendor/tribev2/tribev2"
NEURALSET = HERE / ".venv/lib/python3.11/site-packages/neuralset"

PATCHES = [
    (
        TRIBE / "eventstransforms.py",
        '        compute_type = "float16"\n',
        '        # ctranslate2 has no efficient float16 path on CPU / Apple Silicon.\n'
        '        compute_type = "float16" if device == "cuda" else "int8"\n',
    ),
    (
        NEURALSET / "extractors/base.py",
        "import logging\nimport typing as tp\n",
        "import logging\nimport os\nimport typing as tp\n",
    ),
    (
        NEURALSET / "extractors/base.py",
        '    device: tp.Literal["auto", "cpu", "cuda", "accelerate"] = "auto"\n',
        '    device: tp.Literal["auto", "cpu", "cuda", "mps", "accelerate"] = "auto"\n',
    ),
    (
        NEURALSET / "extractors/base.py",
        '        if self.layers != "all":\n',
        '        # Route per-extractor on CUDA-less machines (see module docstring).\n'
        '        if self.device in ("accelerate", "cuda", "cpu") and not torch.cuda.is_available():\n'
        '            mps_ok = (\n'
        '                torch.backends.mps.is_available()\n'
        '                and os.environ.get("CPD_NO_MPS") != "1"\n'
        '                and name not in ("HuggingFaceText",)\n'
        '            )\n'
        '            if mps_ok:\n'
        '                self.device = "mps"  # type: ignore[assignment]\n'
        '            elif self.device != "cpu":\n'
        '                self.device = "cpu"\n'
        '        if self.layers != "all":\n',
    ),
]


def main() -> int:
    failed = False
    for path, old, new in PATCHES:
        if not path.exists():
            print(f"[MISS] {path} — run the install first")
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
