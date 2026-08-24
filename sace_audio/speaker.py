"""The speaker gate: is this the enrolled caller, or someone else in the room?

This is the layer that does what VAD and denoising cannot. VAD asks "is anyone
speaking"; a denoiser removes non-speech. Neither can reject a second HUMAN
VOICE, because to both of them a colleague talking clearly is exactly as valid
as the caller. Separating one voice from another needs a model conditioned on
whose voice it is, which means a reference: hence enrolment.

HOW IT WORKS

  enrol once   ~10s of the caller's speech -> one 256-dim embedding, on disk
  per utterance embed(audio) -> cosine vs the enrolled vector
               >= threshold -> pass to STT
               <  threshold -> DROPPED, never transcribed

MODEL: wespeaker ResNet34-LM, exported to ONNX (Apache 2.0, ~26MB). Runs on
onnxruntime, which is already a dependency (Silero VAD uses it). No torch, no
network at inference, no paid service — this replaces LiveKit's BVC, which ran
on LiveKit's servers and therefore did nothing at all in console mode.

THE THRESHOLD IS THE WHOLE DESIGN, and it is a genuine trade rather than a knob
to tune blindly:

  too HIGH -> the caller gets clipped mid-sentence, and the call breaks in a way
              they cannot diagnose. This is the worse failure: the system stops
              hearing the person it is talking to.
  too LOW  -> a similar-sounding colleague passes.

So the default errs LOW, and `scripts/calibrate_speaker.py` measures it against
real recordings rather than guessing. Same reasoning as the answer cache's
CACHE_THRESHOLD, and the same rule: measure, do not assume.

FAIL-OPEN, DELIBERATELY. If the model is missing, enrolment has not been done,
or inference raises, every utterance PASSES. A gate that fails closed would
silently stop transcribing anyone, which from the caller's side is
indistinguishable from the agent having hung up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# Cosine similarity required to accept an utterance as the enrolled speaker.
#
# Errs low on purpose — see the module docstring: clipping the real caller is a
# worse failure than admitting one impostor, because the caller cannot tell why
# the agent stopped responding. Raise it only against measured numbers from
# scripts/calibrate_speaker.py.
SPEAKER_THRESHOLD = float(os.environ.get("SPEAKER_THRESHOLD", "0.45"))

# Utterances shorter than this carry too little voice to embed reliably — a
# 150ms "yes" scores erratically against ANY reference. They are passed through
# rather than judged, because the script needs to hear short confirmations
# ("yes", "no", "speaking") and dropping them would break the flow for the sake
# of a measurement the model cannot make.
MIN_UTTERANCE_SEC = float(os.environ.get("SPEAKER_MIN_UTTERANCE", "0.45"))

# Only this much of an utterance is embedded, from the middle.
#
# Inference cost is linear in length while the embedding stops improving after
# a second or so — speaker identity is carried by voice timbre, not by how long
# someone talks. Capping keeps a long rambling answer as cheap as a short one,
# which matters because this cost lands on the hot path of every turn.
#
# MEASURED, enrolling one synthetic voice and scoring three on an unseen
# sentence. "separation" is the enrolled speaker's score minus the best
# impostor's — the margin the threshold has to sit inside:
#
#     cap     gate    enrolled   best impostor   separation
#     1.0s    23ms      0.744        0.265          0.479
#     1.5s    33ms      0.829        0.277          0.552
#     2.0s    45ms      0.863        0.296          0.568
#     3.0s    73ms      0.895        0.254          0.641
#
# 1.5s is the knee: it buys almost all of 2.0s's separation for 25% less time,
# and past that the curve flattens while cost keeps rising linearly. Going below
# 1.0s starts to cost real discrimination.
#
# Taken from the MIDDLE rather than the start: utterance edges hold onset,
# trailing breath and VAD padding, and the centre is the most reliably voiced
# part of a segment.
MAX_EMBED_SEC = float(os.environ.get("SPEAKER_MAX_EMBED", "1.5"))

_MODEL_ENV = "SPEAKER_MODEL_PATH"
_DEFAULT_MODEL = Path.home() / ".cache" / "sace" / "speaker_resnet34.onnx"
_DEFAULT_ENROLLMENT = Path.home() / ".cache" / "sace" / "enrollment.json"

MODEL_URL = (
    "https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM"
    "/resolve/main/onnx/model.onnx"
)


def model_path() -> Path:
    return Path(os.environ.get(_MODEL_ENV, str(_DEFAULT_MODEL)))


def enrollment_path() -> Path:
    return Path(os.environ.get("SPEAKER_ENROLLMENT_PATH", str(_DEFAULT_ENROLLMENT)))


# ── features ────────────────────────────────────────────────────────────────
#
# 80-bin log-mel, which is what the exported model expects. Implemented here in
# numpy rather than pulled from librosa/torchaudio on purpose: those are large
# dependencies for ~20 lines of FFT, and this runs on the hot path of every
# utterance. Measured at 0.5-0.8ms per utterance, which is noise next to the
# 4-10ms of inference that follows.

_N_MELS = 80
_WIN = 400      # 25ms at 16kHz
_HOP = 160      # 10ms at 16kHz
_N_FFT = 512


def _mel_filterbank(sample_rate: int) -> np.ndarray:
    freqs = np.linspace(0, sample_rate / 2, _N_FFT // 2 + 1)
    mel_max = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_pts = np.linspace(0, mel_max, _N_MELS + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    fb = np.zeros((_N_MELS, len(freqs)), dtype=np.float32)
    for i in range(_N_MELS):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        rising = (freqs - lo) / (ctr - lo + 1e-9)
        falling = (hi - freqs) / (hi - ctr + 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0, None)
    return fb


_FB_CACHE: dict[int, np.ndarray] = {}


def log_mel(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """(frames, 80) log-mel features."""
    if sample_rate not in _FB_CACHE:
        _FB_CACHE[sample_rate] = _mel_filterbank(sample_rate)
    fb = _FB_CACHE[sample_rate]

    if len(audio) < _WIN:
        audio = np.pad(audio, (0, _WIN - len(audio)))
    n_frames = 1 + (len(audio) - _WIN) // _HOP
    idx = np.arange(_WIN)[None, :] + _HOP * np.arange(n_frames)[:, None]
    frames = audio[idx] * np.hanning(_WIN)
    power = np.abs(np.fft.rfft(frames, _N_FFT)) ** 2
    feats = np.log(power @ fb.T + 1e-6).astype(np.float32)
    # Per-utterance mean normalisation — the standard trick that makes the
    # embedding robust to channel and gain differences, which matters because
    # enrolment and live audio go through different paths.
    return feats - feats.mean(axis=0, keepdims=True)


def _centre_crop(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """The middle MAX_EMBED_SEC of an utterance — see that constant."""
    limit = int(MAX_EMBED_SEC * sample_rate)
    if len(audio) <= limit:
        return audio
    start = (len(audio) - limit) // 2
    return audio[start:start + limit]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SpeakerGate:
    """Accept only the enrolled speaker. Fails OPEN — see the module docstring."""

    def __init__(self, threshold: float = SPEAKER_THRESHOLD,
                 model: str | Path | None = None,
                 enrollment: str | Path | None = None):
        self.threshold = threshold
        self.reference: np.ndarray | None = None
        self._session = None
        self._input_name = "input_features"
        # Counters, surfaced at end of call: a gate that is silently rejecting
        # everything must be visible, not something to discover from a caller
        # complaining the agent stopped answering.
        self.passed = 0
        self.rejected = 0
        self.skipped_short = 0

        path = Path(model) if model else model_path()
        if not path.exists():
            print(f"[audio] speaker model not found at {path} — the gate is OFF "
                  f"(every voice will be transcribed). Run: python scripts/enroll_voice.py")
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            # One thread: this runs per utterance alongside STT/TTS/LLM work,
            # and letting ORT grab every core would steal them from audio.
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:
            print(f"[audio] could not load the speaker model, gate is OFF: "
                  f"{type(exc).__name__}: {exc}")
            return

        self.reference = load_enrollment(enrollment)
        if self.reference is None:
            print("[audio] no voice enrolled — the gate is OFF (every voice will "
                  "be transcribed). Run: python scripts/enroll_voice.py")

    @property
    def active(self) -> bool:
        """True only when the gate can actually reject something."""
        return self._session is not None and self.reference is not None

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        if self._session is None:
            return None
        try:
            feats = log_mel(audio, sample_rate)[None]  # (1, T, 80)
            out = self._session.run(None, {self._input_name: feats})[0]
            return np.asarray(out[0], dtype=np.float32)
        except Exception as exc:  # pragma: no cover
            print(f"[audio] speaker embedding failed: {type(exc).__name__}: {exc}")
            return None

    def accepts(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[bool, float]:
        """(is_the_caller, similarity). Anything uncertain passes — see the
        fail-open note in the module docstring."""
        if not self.active:
            return True, 1.0
        if len(audio) / sample_rate < MIN_UTTERANCE_SEC:
            # Too short to judge. Passed rather than dropped: the script needs
            # to hear "yes"/"no"/"speaking", and those are exactly the
            # utterances too brief to embed reliably.
            self.skipped_short += 1
            return True, 1.0

        vec = self.embed(_centre_crop(audio, sample_rate), sample_rate)
        if vec is None:
            return True, 1.0

        sim = _cosine(vec, self.reference)
        if sim >= self.threshold:
            self.passed += 1
            return True, sim
        self.rejected += 1
        return False, sim

    def stats(self) -> dict:
        return {"active": self.active, "threshold": self.threshold,
                "passed": self.passed, "rejected": self.rejected,
                "skipped_short": self.skipped_short}


# ── enrolment ───────────────────────────────────────────────────────────────

def load_enrollment(path: str | Path | None = None) -> np.ndarray | None:
    p = Path(path) if path else enrollment_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        vec = np.asarray(data["embedding"], dtype=np.float32)
        return vec if vec.size else None
    except Exception as exc:
        print(f"[audio] could not read enrolment {p}: {type(exc).__name__}: {exc}")
        return None


def save_enrollment(vec: np.ndarray, path: str | Path | None = None,
                    meta: dict | None = None) -> Path:
    p = Path(path) if path else enrollment_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"embedding": [float(x) for x in vec], **(meta or {})}))
    return p


def enroll_from_file(wav_path: str | Path, model: str | Path | None = None,
                     sample_rate: int = 16000) -> np.ndarray:
    """Embed a recording of the caller's voice, for use as the reference.

    Averaged over overlapping windows rather than embedded as one long clip:
    a single embedding of a 10s recording is dominated by whichever few seconds
    happened to be loudest, and averaging several is measurably more stable
    across the phrasing and volume changes of real speech.
    """
    import wave

    with wave.open(str(wav_path), "rb") as w:
        if w.getnchannels() != 1:
            raise ValueError("enrolment audio must be mono")
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if sr != sample_rate:
        # Linear resample. Adequate here: the model is robust to the small
        # artefacts this introduces, and it avoids a scipy/librosa dependency
        # for a step that runs once, offline, during enrolment.
        n = int(len(audio) * sample_rate / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)

    gate = SpeakerGate(model=model)
    if gate._session is None:
        raise RuntimeError("speaker model unavailable; cannot enrol")

    win, hop = sample_rate * 3, sample_rate * 3 // 2
    vecs = []
    for start in range(0, max(1, len(audio) - win + 1), hop):
        chunk = audio[start:start + win]
        if len(chunk) < sample_rate:      # ignore a short trailing fragment
            continue
        v = gate.embed(chunk, sample_rate)
        if v is not None:
            vecs.append(v / (np.linalg.norm(v) + 1e-9))

    if not vecs:
        v = gate.embed(audio, sample_rate)
        if v is None:
            raise RuntimeError("could not embed the enrolment audio")
        vecs = [v / (np.linalg.norm(v) + 1e-9)]

    mean = np.mean(vecs, axis=0)
    return (mean / (np.linalg.norm(mean) + 1e-9)).astype(np.float32)
