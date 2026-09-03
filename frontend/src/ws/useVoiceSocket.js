import { useEffect, useRef, useState } from "react";

// Derived from the page's own origin, not hardcoded to localhost. Two things
// break otherwise once the dashboard is reachable from a phone: "localhost" is
// the phone rather than the machine running the agent, and a ws:// socket on an
// https:// page is blocked as mixed content. Taking the scheme from the page
// makes it wss:// exactly when it has to be, and vite.config.js proxies /ws to
// the agent's broadcast on :8765.
const WS_URL =
  import.meta.env.VITE_VOICE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws`;
const RECONNECT_DELAY_MS = 2000;

/** Connection to voice_agent.py's spectator broadcast.
 *
 * Mostly watch-only: it receives retrieval / turn / learned / learning_done /
 * call_started / call_ended / start_refused / audio_state events as the real
 * agent produces them. It can send back {"type": "start_call"},
 * {"type": "end_call"} and {"type": "set_audio"} — everything else shown by the
 * dashboard is still just what the agent already decided and did.
 * Reconnects on drop, since the agent process may not be up yet when the
 * dashboard is opened, or may restart between calls.
 */
export function useVoiceSocket(onEvent) {
  const [connected, setConnected] = useState(false);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;
  const socketRef = useRef(null);

  useEffect(() => {
    let socket;
    let reconnectTimer;
    let cancelled = false;

    function connect() {
      socket = new WebSocket(WS_URL);
      socketRef.current = socket;

      socket.onopen = () => setConnected(true);

      socket.onmessage = (msg) => {
        try {
          const event = JSON.parse(msg.data);
          onEventRef.current(event);
        } catch {
          // malformed frame from a mismatched server version; ignore rather
          // than crash the dashboard over one bad event
        }
      };

      socket.onclose = () => {
        setConnected(false);
        if (!cancelled) {
          reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };

      socket.onerror = () => socket.close();
    }

    connect();

    return () => {
      cancelled = true;
      clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  function send(type) {
    const sock = socketRef.current;
    if (sock && sock.readyState === WebSocket.OPEN) {
      sock.send(JSON.stringify({ type }));
      return true;
    }
    return false;
  }

  const endCall = () => send("end_call");
  const startCall = () => send("start_call");

  /** Flip an audio layer mid-call. Either field may be omitted, meaning "leave
   *  that one alone". The agent echoes an `audio_state` back rather than the UI
   *  assuming success — a layer that is unavailable cannot be switched on, and
   *  the toggle must end up showing what is ACTUALLY running. */
  const setAudio = (opts) => {
    const sock = socketRef.current;
    if (sock && sock.readyState === WebSocket.OPEN) {
      sock.send(JSON.stringify({ type: "set_audio", ...opts }));
      return true;
    }
    return false;
  };

  return { connected, endCall, startCall, setAudio };
}
