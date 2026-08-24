"""Measure SPEAKER_THRESHOLD against real recordings instead of guessing it.

    python scripts/calibrate_speaker.py me1.wav me2.wav --others them1.wav them2.wav

Scores each recording against the current enrolment and prints the two bands —
you, and everyone else. A threshold is only trustworthy if those bands do not
overlap, and this shows whether they do.

WHY THIS EXISTS. The default (0.45) is set from synthetic voices on one machine.
Real numbers depend on your voice, your microphone and your room, and the cost
of getting it wrong is asymmetric:

  too HIGH -> the real caller is clipped mid-call, and cannot tell why
  too LOW  -> a similar-sounding colleague gets through

So the recommendation below sits in the GAP between the bands, biased toward the
low end — same reasoning as the answer cache's CACHE_THRESHOLD, measured the
same way.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sace_audio.denoise import Denoiser
from sace_audio.speaker import SpeakerGate, _centre_crop, _cosine, load_enrollment


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def score(paths: list[str], gate: SpeakerGate, ref: np.ndarray,
          denoiser: Denoiser) -> list[tuple[str, float]]:
    out = []
    for p in paths:
        try:
            audio, sr = read_wav(p)
        except Exception as exc:
            print(f"  ! {p}: {type(exc).__name__}: {exc}")
            continue
        audio = denoiser(audio, sr)
        vec = gate.embed(_centre_crop(audio, sr), sr)
        if vec is None:
            print(f"  ! {p}: could not embed")
            continue
        out.append((Path(p).name, _cosine(vec, ref)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mine", nargs="+", help="recordings of the enrolled speaker")
    ap.add_argument("--others", nargs="*", default=[],
                    help="recordings of OTHER people")
    args = ap.parse_args()

    ref = load_enrollment()
    if ref is None:
        print("No enrolment found. Run scripts/enroll_voice.py first.", file=sys.stderr)
        return 2
    gate = SpeakerGate()
    if gate._session is None:
        return 2
    den = Denoiser()

    print("\nYOU (should score HIGH):")
    mine = score(args.mine, gate, ref, den)
    for name, s in mine:
        print(f"  {s:6.3f}  {name}")

    others = []
    if args.others:
        print("\nOTHERS (should score LOW):")
        others = score(args.others, gate, ref, den)
        for name, s in others:
            print(f"  {s:6.3f}  {name}")

    if not mine:
        print("\nNothing scored.", file=sys.stderr)
        return 1

    lo_mine = min(s for _, s in mine)
    print(f"\ncurrent threshold : {gate.threshold}")
    print(f"your weakest      : {lo_mine:.3f}")

    if not others:
        print("\nNo --others given, so this only shows you are recognised — it "
              "cannot show whether anyone else would be. Record a colleague and "
              "re-run before trusting a raised threshold.")
        return 0

    hi_other = max(s for _, s in others)
    print(f"strongest impostor: {hi_other:.3f}")

    if hi_other >= lo_mine:
        print("\nTHE BANDS OVERLAP — no threshold separates these recordings. "
              "Usually the enrolment is poor (too short, noisy, or a different "
              "microphone). Re-enrol with a clean 12s sample before tuning.")
        return 1

    # Sit in the gap, biased low: clipping the real caller is the worse failure.
    suggested = round(hi_other + (lo_mine - hi_other) * 0.35, 2)
    print(f"gap               : {lo_mine - hi_other:.3f}")
    print(f"\nSUGGESTED         : SPEAKER_THRESHOLD={suggested}")
    print("  (placed in the gap, biased toward the low end — a false accept "
          "costs one bad transcript, a false reject breaks the call)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
