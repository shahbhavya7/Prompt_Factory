"""Spectral-gate denoising. Removes background noise, not background voices.

WHAT THIS LAYER IS FOR, honestly stated: fans, hum, traffic, keyboard, room
tone. It does NOT reject other people talking — to a spectral gate a colleague's
voice is clean speech with the same statistics as the caller's, and it passes
straight through. That job belongs to speaker.py, and this module exists partly
to make the next layer's job easier: a cleaner signal produces a more stable
speaker embedding, and it improves STT accuracy on its own.

WHY noisereduce RATHER THAN DeepFilterNet. DeepFilterNet is the better denoiser
by output quality and would have been the obvious pick. It was tested and
rejected on two hard constraints, both verified rather than assumed:

  * it imports torch at module load (~2GB), despite shipping a Rust backend
    that makes the wheel look torch-free;
  * installing it downgrades numpy 2.4.6 -> 1.26.4, which risks the rest of
    this project's stack.

noisereduce is pure numpy/scipy, adds ~7ms per utterance, and leaves the
environment alone. Less aggressive on hard noise; the right trade here.
"""

from __future__ import annotations

import os

import numpy as np

# Proportion of the noise-gate to apply. 1.0 is full spectral subtraction, which
# is audibly aggressive and can hollow out quiet consonants — and a mangled
# consonant costs STT accuracy, which is the thing this is meant to help.
# 0.8 removes the great majority of steady noise while leaving speech intact.
PROP_DECREASE = float(os.environ.get("DENOISE_PROP_DECREASE", "0.8"))

# Below this many samples there is not enough signal for the STFT to estimate a
# noise profile from, and the result is worse than the input. Short blips are
# passed through untouched and left for the speaker gate to judge.
_MIN_SAMPLES = 512


class Denoiser:
    """Stateless spectral-gate denoiser over float32 mono audio.

    Deliberately tolerant: an optimisation that raises mid-call would take the
    call down with it, so every failure path returns the ORIGINAL audio. Worse
    audio is recoverable; a dropped call is not.
    """

    def __init__(self, enabled: bool = True, prop_decrease: float = PROP_DECREASE):
        self.prop_decrease = prop_decrease
        self._nr = None
        self.enabled = False
        if not enabled:
            return
        try:
            import noisereduce

            self._nr = noisereduce
            self.enabled = True
        except ImportError:
            # Optional by design — the speaker gate is the layer that matters,
            # and it works fine on un-denoised audio.
            print("[audio] noisereduce not installed; denoising disabled "
                  "(pip install noisereduce)")

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size < _MIN_SAMPLES:
            return audio
        try:
            out = self._nr.reduce_noise(
                y=audio, sr=sample_rate,
                prop_decrease=self.prop_decrease,
                stationary=True,   # steady background, estimated over the clip
            )
            return out.astype(np.float32, copy=False)
        except Exception as exc:  # pragma: no cover - optimisation path
            print(f"[audio] denoise failed, passing audio through: "
                  f"{type(exc).__name__}: {exc}")
            return audio
