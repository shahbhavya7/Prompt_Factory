/** The two audio-processing switches, flipped live on the running agent.
 *
 * They are separate because they do genuinely different jobs and fail
 * differently — collapsing them into one "noise cancellation" switch would hide
 * that a denoiser cannot reject a person:
 *
 *   Denoise    strips steady background (fans, hum, traffic). Passes other
 *              people's SPEECH straight through — to a spectral gate a
 *              colleague's voice is clean audio.
 *   Voice gate drops any utterance that is not the enrolled caller. This is the
 *              one that rejects other people, and it needs an enrolled voice.
 *
 * `available` is distinct from `enabled` on purpose. A layer with no enrolment
 * (or no library installed) CANNOT be turned on, so it renders disabled with
 * the reason attached rather than as an ordinary off switch that does nothing
 * when clicked.
 */
export default function AudioToggles({ state, onChange, disabled }) {
  if (!state || !state.filter_present) return null;

  const rows = [
    {
      key: "denoise",
      label: "Denoise",
      available: state.denoise_available,
      enabled: state.denoise_enabled,
      why: "noisereduce is not installed",
      hint: "Removes steady background noise. Does not reject other voices.",
    },
    {
      key: "gate",
      label: "Caller-voice only",
      available: state.gate_available,
      enabled: state.gate_enabled,
      why: "no voice enrolled — run scripts/enroll_voice.py",
      hint: "Drops anyone who is not the enrolled caller, before transcription.",
    },
  ];

  return (
    <div className="audio-toggles">
      {rows.map((r) => (
        <label
          key={r.key}
          className={`audio-toggle ${r.available ? "" : "audio-toggle--off"}`}
          title={r.available ? r.hint : r.why}
        >
          <input
            type="checkbox"
            checked={r.enabled}
            disabled={disabled || !r.available}
            onChange={(e) => onChange({ [r.key]: e.target.checked })}
          />
          <span>{r.label}</span>
          {!r.available && <span className="audio-toggle__note">unavailable</span>}
        </label>
      ))}
    </div>
  );
}
