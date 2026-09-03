import { useCallback, useEffect, useRef, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";
import { joinCall } from "../api/voice";

/** Talk to Maya from THIS device — the browser becomes the caller.
 *
 * The rest of the dashboard is a spectator: it watches what the agent decided
 * and can ask it to start a call, but it carries no audio. That is fine while
 * the browser and the agent are the same machine, because `./run.sh voice`
 * binds that machine's microphone. It stops being fine the moment the
 * dashboard is opened on a phone through a tunnel — the server's microphone is
 * in another room, and "start call" would listen to the wrong one.
 *
 * So this component joins the LiveKit room as a real participant and publishes
 * the phone's own mic. The agent is dispatched into the same room by
 * POST /voice/join (see api/main.py — the worker uses explicit dispatch, so
 * without that call nobody would be there). The desktop dashboard keeps
 * watching the same session over the spectator websocket, unchanged.
 *
 * The audio path is phone -> LiveKit Cloud -> agent. It does NOT go through the
 * tunnel: LiveKit Cloud is already publicly reachable, and the tunnel exists
 * only so the phone can load this page.
 */
export default function PhoneCall() {
  const [status, setStatus] = useState("idle"); // idle | joining | live | error
  const [error, setError] = useState(null);
  const [room, setRoom] = useState(null);
  const [muted, setMuted] = useState(false);
  const roomRef = useRef(null);
  // Where the agent's voice is played. Held in a ref and attached on track
  // subscribe rather than rendered per-track: there is exactly one remote audio
  // track (Maya), and a stable element avoids the autoplay prompt that a
  // freshly-created one triggers on iOS.
  const audioRef = useRef(null);

  const leave = useCallback(async () => {
    const r = roomRef.current;
    roomRef.current = null;
    setRoom(null);
    setStatus("idle");
    if (r) {
      try {
        await r.disconnect();
      } catch {
        /* already gone — leaving is best-effort */
      }
    }
  }, []);

  const join = useCallback(async () => {
    setError(null);
    setStatus("joining");
    try {
      // The token is minted per join and scoped to one room, so a stale tab
      // cannot rejoin a finished call.
      const { url, token } = await joinCall();

      const r = new Room({
        // The caller is a person on a phone in a room with background noise.
        adaptiveStream: true,
        dynacast: true,
        audioCaptureDefaults: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      r.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio && audioRef.current) {
          track.attach(audioRef.current);
        }
      });
      r.on(RoomEvent.Disconnected, () => {
        roomRef.current = null;
        setRoom(null);
        setStatus("idle");
      });

      await r.connect(url, token);
      // Publishing the mic is the whole point, and it is what triggers the
      // browser's permission prompt. It must happen after connect so the
      // failure surfaces as "no mic" rather than "no room".
      await r.localParticipant.setMicrophoneEnabled(true);

      roomRef.current = r;
      setRoom(r);
      setMuted(false);
      setStatus("live");
    } catch (e) {
      // getUserMedia fails with a NotAllowedError on an insecure origin, which
      // is the single most likely thing to go wrong here: a phone loading this
      // over plain http:// gets no microphone at all. Say so, rather than
      // showing a bare DOM exception.
      const insecure =
        typeof window !== "undefined" &&
        !window.isSecureContext &&
        e?.name === "NotAllowedError";
      setError(
        insecure
          ? "The browser blocked the microphone because this page is not on a " +
            "secure origin. Open the https:// tunnel URL rather than the " +
            "http:// LAN address."
          : e?.message || String(e),
      );
      setStatus("error");
      await leave();
    }
  }, [leave]);

  const toggleMute = useCallback(async () => {
    const r = roomRef.current;
    if (!r) return;
    const next = !muted;
    await r.localParticipant.setMicrophoneEnabled(!next);
    setMuted(next);
  }, [muted]);

  // Leaving the page mid-call should hang up, not leave a ghost participant
  // publishing an open microphone into the room.
  useEffect(() => {
    return () => {
      const r = roomRef.current;
      roomRef.current = null;
      if (r) r.disconnect().catch(() => {});
    };
  }, []);

  return (
    <section className="phone-call">
      <header>
        <h2>Call from this device</h2>
        <p className="hint">
          Joins the call with this device&apos;s microphone. Open this page on
          your phone through the tunnel URL to call from the phone.
        </p>
      </header>

      {status !== "live" ? (
        <button type="button" onClick={join} disabled={status === "joining"}>
          {status === "joining" ? "Connecting…" : "Call Maya"}
        </button>
      ) : (
        <div className="phone-call-live">
          <span className="live-dot" aria-hidden="true" />
          <span>Connected{room?.name ? ` · ${room.name}` : ""}</span>
          <button type="button" onClick={toggleMute}>
            {muted ? "Unmute" : "Mute"}
          </button>
          <button type="button" onClick={leave}>
            Hang up
          </button>
        </div>
      )}

      {error ? <p className="error">{error}</p> : null}

      {/* autoPlay + playsInline: iOS Safari will not start audio otherwise, and
          the element is always mounted so the first remote track has somewhere
          to attach the moment it arrives. */}
      <audio ref={audioRef} autoPlay playsInline />
    </section>
  );
}
