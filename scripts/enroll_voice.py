"""Record the caller's voice once, so the speaker gate knows who to listen for.

    python scripts/enroll_voice.py              # record 12s from the mic
    python scripts/enroll_voice.py --file a.wav # or use an existing recording
    python scripts/enroll_voice.py --check      # what is enrolled right now

Downloads the ~26MB ONNX model on first run and caches it in ~/.cache/sace.

RE-ENROL WHEN THE MICROPHONE CHANGES. The embedding captures the whole capture
chain, not just the voice — headset, laptop mic and phone produce measurably
different vectors for the same person, enough to push a genuine caller under the
threshold. If the gate starts rejecting you, this is the first thing to suspect.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sace_audio.speaker import (
    MODEL_URL,
    SpeakerGate,
    enroll_from_file,
    enrollment_path,
    model_path,
    save_enrollment,
)

RATE = 16000
SECONDS = 12

PROMPT = """
Read this aloud, at the pace and volume you would use on a real call:

    "Hi, this is a test of the voice enrolment system. I am recording a short
     sample so the agent can tell my voice apart from other people talking
     nearby. Today is a normal working day and the weather outside is fine."
"""


def ensure_model() -> Path:
    path = model_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading the speaker model (~26MB) to {path} …")
    import urllib.request

    # To a temp name first, then renamed: an interrupted download that lands on
    # the real path would be a corrupt file that fails confusingly on every
    # later run rather than re-downloading.
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(path)
    print("done.")
    return path


def record(seconds: int = SECONDS) -> Path:
    try:
        import sounddevice as sd
    except ImportError:
        print("Recording needs sounddevice:  pip install sounddevice\n"
              "Or record a WAV yourself and pass --file.", file=sys.stderr)
        raise SystemExit(2)

    print(PROMPT)
    input(f"Press Enter, then speak for {seconds} seconds… ")
    print("recording…", flush=True)
    audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16")
    sd.wait()
    print("done.")

    out = Path.home() / ".cache" / "sace" / "enrollment.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.tobytes())
    return out


def check() -> int:
    gate = SpeakerGate()
    st = gate.stats()
    print(f"model      : {model_path()}  {'✓' if model_path().exists() else '— MISSING'}")
    print(f"enrolment  : {enrollment_path()}  {'✓' if gate.reference is not None else '— NONE'}")
    print(f"threshold  : {st['threshold']}")
    print(f"gate active: {st['active']}")
    if not st["active"]:
        print("\nThe gate is OFF — every voice will be transcribed. "
              "Run this script with no arguments to enrol.")
    return 0 if st["active"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="use an existing mono WAV instead of recording")
    ap.add_argument("--seconds", type=int, default=SECONDS)
    ap.add_argument("--check", action="store_true", help="show what is enrolled")
    args = ap.parse_args()

    if args.check:
        return check()

    ensure_model()
    wav = Path(args.file) if args.file else record(args.seconds)
    if not wav.exists():
        print(f"no such file: {wav}", file=sys.stderr)
        return 2

    print("computing the voice embedding…")
    vec = enroll_from_file(wav)
    path = save_enrollment(vec, meta={"source": str(wav), "sample_rate": RATE})
    print(f"enrolled ✓  {path}")

    # Immediately score the enrolment against ITSELF. This is a sanity check on
    # the whole path — features, model, cosine — not a measure of accuracy, and
    # it is worth the two seconds: a silent microphone produces a confident
    # embedding of nothing, and without this the failure would only surface
    # mid-call as the gate rejecting the very person it was enrolled on.
    gate = SpeakerGate()
    if gate.active:
        import wave as _w

        with _w.open(str(wav), "rb") as w:
            raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        ok, sim = gate.accepts(audio, RATE)
        print(f"self-check : cosine {sim:.3f} vs threshold {gate.threshold} "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("  The enrolment does not match itself. The recording is "
                  "probably silent or clipped — check the microphone and redo it.")
            return 1
        print("\nThe gate is live. Other voices will be dropped before STT.")
        print("If it ever rejects you, re-run this script — a different "
              "microphone changes the embedding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
