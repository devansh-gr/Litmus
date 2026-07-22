"""Shared TRIBE event construction for the brain-map path.

Two modes, selected by CPD_BRAINMAP_MODE:

  "text"  (default) -- inject synthetic Word events directly (NO audio). Skips
          TTS + WhisperX transcription entirely. Experiment I1 proved this
          preserves the semantic signal (d=0.99 vs the audio path's 0.95) while
          dropping every audio dependency (say/ffmpeg/WhisperX) and the biggest
          chunk of latency (the large-v3 transcription).

  "audio" -- the original path: text -> offline TTS (say+ffmpeg) -> WhisperX
          word timings -> events. Kept for provenance / comparison.

IMPORTANT: baseline.npz MUST be built with the SAME mode used at query time.
The two paths produce different absolute activations, so a z-score baseline from
one path is invalid for the other. build_baseline.py reads this same env var, and
the audio-path baseline is preserved as baseline_audio.npz for reference.
"""

import os

import pandas as pd

BRAINMAP_MODE = os.environ.get("CPD_BRAINMAP_MODE", "text")
WORD_DT = 0.35  # synthetic per-word onset spacing / duration (s); matches the I1 probe


def text_only_events(text: str) -> pd.DataFrame:
    """Build TRIBE events from text with NO audio.

    Synthetic Word events (0.35 s/word) are fed through the text pipeline
    (AddText / AddSentenceToWords / AddContextToWords). ExtractWordsFromAudio
    early-returns because Word events already exist, so transcription is skipped
    -- that early-return is the whole trick (see I1).
    """
    from tribev2.demo_utils import get_audio_and_text_events

    words = text.replace(",", "").replace(".", "").split()
    if not words:
        raise ValueError("no words to build events from")
    rows = [
        {"type": "Word", "text": w, "start": i * WORD_DT, "duration": WORD_DT,
         "timeline": "default", "subject": "default", "filepath": ""}
        for i, w in enumerate(words)
    ]
    return get_audio_and_text_events(pd.DataFrame(rows))


def harden_tribe(model):
    """Force single-process data loading. from_pretrained restores num_workers to
    the checkpoint's N_CPUS (~20); those forked DataLoader workers DEADLOCK on
    macOS -- the "silent 4-hour freeze" in the gotchas (and it froze this very
    baseline build at 2/30). Single-process loading can't deadlock and is faster
    for our one-sentence inputs. MUST run after every from_pretrained."""
    try:
        model.data.num_workers = 0
    except AttributeError:
        pass
    return model


def build_events(model, text: str, text_path: str) -> pd.DataFrame:
    """Mode-dispatched event construction.

    `model` and `text_path` are only used by the audio path (TTS + WhisperX via
    get_events_dataframe). The text path ignores them and needs no files on disk.
    """
    if BRAINMAP_MODE == "text":
        return text_only_events(text)
    return model.get_events_dataframe(text_path=text_path)
