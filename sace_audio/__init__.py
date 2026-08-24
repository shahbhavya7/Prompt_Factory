"""Caller-voice isolation: denoise, then reject anyone who is not the caller.

WHY THIS EXISTS, AND WHY IT IS NOT VAD.

Silero VAD answers "is anyone speaking?". It cannot tell voices apart, so a
colleague talking nearby, a TV, or someone across the room clears its threshold,
reaches Deepgram, and is transcribed as if the caller had said it. Raising the
VAD threshold only filters by LOUDNESS — it drops distant speech but passes a
clear voice next to the mic, which is exactly the case that matters most.

Two layers, in this order:

    mic -> denoise -> speaker gate -> STT

  1. DENOISE (noisereduce, spectral gating). Removes steady background — fans,
     hum, traffic, room tone. This is not the layer that rejects other people:
     to a denoiser a colleague's voice is clean speech and passes untouched. It
     is here because it measurably improves both STT accuracy and the stability
     of the embedding the next layer computes.

  2. SPEAKER GATE (wespeaker ResNet34, ONNX). The layer that actually answers
     "is this the CALLER?". One enrolled reference embedding, cosine-compared
     against every utterance; below threshold the audio is dropped before it
     ever reaches STT.

DELIBERATELY FRAMEWORK-AGNOSTIC. Everything here takes and returns plain numpy
float32 at a given sample rate. It does not import livekit, and it must not:
LiveKit is paid and is expected to be replaced, and this is the piece that has
to survive that. The LiveKit-specific glue lives in voice_agent.stt_node and is
a dozen lines.

This replaces LiveKit's BVC, which was removed. BVC ran on LiveKit's servers, so
it did nothing in console mode; this runs locally and works in both.

MEASURED on this machine, per utterance (not per frame):

    segment    denoise    fbank     onnx     TOTAL
      400ms      7.0ms     0.5ms    3.8ms    ~11ms
     1500ms      7.0ms     0.8ms    9.6ms    ~17ms

Against turns that run 7,000-14,000ms end to end, that is ~0.1% — inside the
run-to-run variance of STT alone.
"""

from sace_audio.denoise import Denoiser
from sace_audio.speaker import SpeakerGate, enroll_from_file, load_enrollment
from sace_audio.stream import UtteranceFilter

__all__ = ["Denoiser", "SpeakerGate", "UtteranceFilter",
           "enroll_from_file", "load_enrollment"]
