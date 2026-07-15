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
    # num_workers=None -> ~20 forked DataLoader workers that DEADLOCK on macOS
    # (froze the baseline build for 4h at 0% CPU). Force single-process loading.
    (
        TRIBE / "main.py",
        "    batch_size: int = 64\n    num_workers: int | None = None\n",
        "    batch_size: int = 64\n"
        "    # Was None -> ~20 forked DataLoader workers, which DEADLOCKS on macOS (froze\n"
        "    # the baseline build for 4h at 0% CPU). Single-process loading can't deadlock\n"
        "    # and is faster for our tiny single-sentence inputs.\n"
        "    num_workers: int | None = 0\n",
    ),
    # PRIVACY: upstream synthesises speech with gTTS, which uploads the text to
    # Google. This tool reads whatever the user highlights, so that is an
    # unacceptable leak. Use macOS `say` (offline) instead.
    (
        TRIBE / "demo_utils.py",
        "        from gtts import gTTS\n"
        "        from langdetect import detect\n"
        "\n"
        '        audio_path = Path(self.infra.uid_folder(create=True)) / "audio.mp3"\n'
        "        lang = detect(self.text)\n"
        "        tts = gTTS(self.text, lang=lang)\n"
        "        tts.save(str(audio_path))\n"
        '        logger.info(f"Wrote TTS audio to {audio_path}")\n',
        "        # PRIVACY: upstream uses gTTS, which uploads the text to Google's servers.\n"
        "        # This tool analyses whatever the user highlights on screen, so that is an\n"
        "        # unacceptable leak. Synthesise locally instead (macOS `say`), and only\n"
        "        # fall back to gTTS with a loud warning if no offline voice exists.\n"
        "        import shutil\n"
        "        import subprocess\n"
        "\n"
        "        folder = Path(self.infra.uid_folder(create=True))\n"
        "\n"
        '        if shutil.which("say") and shutil.which("ffmpeg"):\n'
        '            aiff_path = folder / "audio.aiff"\n'
        '            audio_path = folder / "audio.wav"\n'
        '            subprocess.run(["say", "-o", str(aiff_path), self.text], check=True)\n'
        "            subprocess.run(\n"
        '                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff_path),\n'
        '                 "-ar", "16000", "-ac", "1", str(audio_path)],\n'
        "                check=True,\n"
        "            )\n"
        "            aiff_path.unlink(missing_ok=True)\n"
        '            logger.info(f"Wrote OFFLINE TTS audio to {audio_path}")\n'
        "        else:\n"
        "            from gtts import gTTS\n"
        "            from langdetect import detect\n"
        "\n"
        "            logger.warning(\n"
        '                "No offline TTS available - FALLING BACK TO gTTS, which SENDS THE "\n'
        '                "TEXT TO GOOGLE. Install ffmpeg / run on macOS to avoid this."\n'
        "            )\n"
        '            audio_path = folder / "audio.mp3"\n'
        "            tts = gTTS(self.text, lang=detect(self.text))\n"
        "            tts.save(str(audio_path))\n"
        '            logger.info(f"Wrote TTS audio to {audio_path}")\n',
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
