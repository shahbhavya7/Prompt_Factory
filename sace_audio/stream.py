"""Utterance-level gating over a continuous frame stream.

THE PROBLEM THIS SOLVES. The speaker gate judges an UTTERANCE — it needs a
second or so of voice to produce a stable embedding. But audio arrives as a
continuous stream of ~10-20ms frames, and a single frame carries nowhere near
enough signal to identify anyone. Something has to group frames into utterances,
decide once per utterance, and forward or drop the whole group.

HOW IT DECIDES, and why the decision is deferred. Speech is detected by simple
frame energy (see _SILENCE_FRAMES). Frames are buffered while someone is
talking; when the utterance ends, it is denoised, scored, and either released
downstream or dropped entirely.

That means audio is HELD until the utterance completes — it is not forwarded
frame by frame. There is no way around this: the identity of a speaker is not
knowable from the first 20ms, so a gate that forwarded frames immediately would
already have leaked most of the utterance by the time it could judge it. The
cost is that STT starts on an utterance slightly later than it otherwise would.
Measured against turns of 7,000-14,000ms, that lands inside STT's own variance.

WHY ENERGY AND NOT SILERO. Silero VAD is already loaded and is better at this,
but the framework owns that instance and consumes its own copy of the stream —
reaching into it from here would couple this module to LiveKit's internals,
which is exactly what this package exists to avoid. Frame energy is crude but
sufficient: it only has to find utterance BOUNDARIES, and the speaker gate makes
the actual judgement.
"""

from __future__ import annotations

import os

import numpy as np

# RMS above which a frame counts as speech. Deliberately low — this is only
# finding boundaries, and clipping the start of an utterance would cost the
# speaker gate the voiced audio it needs.
SPEECH_RMS = float(os.environ.get("AUDIO_SPEECH_RMS", "0.01"))

# Consecutive quiet frames that end an utterance. At ~20ms a frame, 25 frames is
# ~500ms — long enough to survive the pauses inside a sentence, short enough not
# to glue a caller's answer onto whatever follows it.
_SILENCE_FRAMES = int(os.environ.get("AUDIO_SILENCE_FRAMES", "25"))

# Hard ceiling on a buffered utterance. Someone who talks continuously must not
# be held indefinitely: at this point the buffer is judged and released, and the
# next frames start a new utterance.
_MAX_UTTERANCE_SEC = float(os.environ.get("AUDIO_MAX_UTTERANCE", "15.0"))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


class UtteranceFilter:
    """Groups frames into utterances and drops the ones that are not the caller.

    Framework-agnostic: `feed` takes a float32 array plus its sample rate and
    returns a list of arrays to forward (empty if the audio is being buffered or
    was rejected). The LiveKit-specific conversion lives in the caller.
    """

    def __init__(self, denoiser=None, gate=None,
                 denoise_enabled: bool = True, gate_enabled: bool = True):
        self.denoiser = denoiser
        self.gate = gate
        # Runtime switches, flipped by the dashboard mid-call. Kept separate
        # from the objects themselves so a toggle is free: the ONNX session and
        # the enrolled reference stay loaded, and turning the gate back on does
        # not pay the ~100ms model load again. Also means a toggle can be
        # honoured between one utterance and the next rather than only between
        # calls.
        self.denoise_enabled = denoise_enabled
        self.gate_enabled = gate_enabled
        self._buf: list[np.ndarray] = []
        self._quiet = 0
        self._in_speech = False
        self._rate = 16000

    def _flush(self) -> list[np.ndarray]:
        """Judge the buffered utterance. Returns it, or nothing if rejected."""
        if not self._buf:
            return []
        audio = np.concatenate(self._buf)
        self._buf, self._quiet, self._in_speech = [], 0, False

        if self.denoiser is not None and self.denoise_enabled:
            audio = self.denoiser(audio, self._rate)

        if self.gate is not None and self.gate_enabled:
            ok, sim = self.gate.accepts(audio, self._rate)
            if not ok:
                dur = len(audio) / self._rate
                print(f"  [audio] dropped {dur:.1f}s — not the enrolled caller "
                      f"(cosine {sim:.3f} < {self.gate.threshold})")
                return []
        return [audio]

    def feed(self, frame: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        self._rate = sample_rate
        speech = _rms(frame) >= SPEECH_RMS

        if speech:
            self._in_speech = True
            self._quiet = 0
            self._buf.append(frame)
            held = sum(len(b) for b in self._buf) / sample_rate
            if held >= _MAX_UTTERANCE_SEC:
                return self._flush()
            return []

        if self._in_speech:
            # Quiet frames inside an utterance are kept: trailing audio carries
            # the end of the final word, and STT needs it.
            self._buf.append(frame)
            self._quiet += 1
            if self._quiet >= _SILENCE_FRAMES:
                return self._flush()
            return []

        # Silence outside an utterance. Forwarded rather than swallowed, so
        # whatever consumes this still sees a continuous stream and its own
        # endpointing keeps working.
        return [frame]

    def drain(self) -> list[np.ndarray]:
        """Whatever is still buffered, judged. For end of stream."""
        return self._flush()

    @property
    def active(self) -> bool:
        """Whether this filter would change the audio at all right now.

        False means feed() still buffers utterances but releases every one of
        them untouched — the caller can use this to skip the whole path.
        """
        return bool(
            (self.denoiser is not None and self.denoise_enabled)
            or (self.gate is not None and self.gate_enabled)
        )
