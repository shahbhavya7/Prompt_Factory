import { useEffect, useRef, useState } from "react";

const WS_URL = import.meta.env.VITE_VOICE_WS_URL || "ws://localhost:8765";
const RECONNECT_DELAY_MS = 2000;

/** Connection to voice_agent.py's spectator broadcast.
 *
 * Mostly watch-only: it receives retrieval / turn / learned / learning_done /
 * call_started / call_ended / start_refused events as the real agent produces
 * them. The two things it can send back are {"type": "start_call"} and
 * {"type": "end_call"} — everything else shown by the dashboard is still just
 * what the agent already decided and did.
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

  return { connected, endCall, startCall };
}
