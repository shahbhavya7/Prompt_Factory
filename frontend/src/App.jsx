import { useReducer } from "react";
import { useVoiceSocket } from "./ws/useVoiceSocket";
import CallStatusBar from "./components/CallStatusBar";
import TranscriptView from "./components/TranscriptView";
import MemoryLookupCard from "./components/MemoryLookupCard";
import LearningFeed from "./components/LearningFeed";
import EndCallButton from "./components/EndCallButton";

const initialState = {
  sessionId: null,
  callActive: false,
  turns: [], // completed {retrieval + turn} pairs, in order
  pendingRetrieval: null, // a retrieval event with no matching turn yet
  learning: [],
  learningDone: false,
};

function reducer(state, event) {
  switch (event.type) {
    case "call_started":
      return { ...initialState, sessionId: event.session_id, callActive: true };
    case "call_ended":
      return { ...state, callActive: false };
    case "retrieval":
      // A retrieval landing at all is itself proof a call is in progress —
      // covers the dashboard-opened-mid-call case where "call_started" was
      // never seen (see hasSeenACall in App.jsx for the matching reasoning).
      return { ...state, pendingRetrieval: event, callActive: true };
    case "turn": {
      const retrieval =
        state.pendingRetrieval?.turn_index === event.turn_index
          ? state.pendingRetrieval
          : null;
      return {
        ...state,
        pendingRetrieval: null,
        turns: [...state.turns, { retrieval, turn: event }],
      };
    }
    case "learned":
      return { ...state, learning: [...state.learning, event] };
    case "learning_done":
      return { ...state, learningDone: true };
    default:
      return state;
  }
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { connected, endCall } = useVoiceSocket(dispatch);

  const transcriptTurns = state.turns.map(({ turn }) => turn);
  // A dashboard opened mid-call can miss the "call_started" frame entirely
  // (it only reaches clients already connected at that instant) — so "a call
  // happened" is inferred from turns actually having arrived, not from
  // sessionId, which would otherwise stay null for the rest of that call.
  const hasSeenACall = state.turns.length > 0 || !!state.sessionId;
  const ended = !state.callActive && hasSeenACall;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1 className="app-title">Maya — live</h1>
          <p className="app-subtitle">Watching voice_agent.py in real time</p>
        </div>
        <EndCallButton disabled={!connected || !state.callActive} onClick={endCall} />
      </header>

      <CallStatusBar
        connected={connected}
        callActive={state.callActive}
        sessionId={state.sessionId}
      />

      <div className="chat-view">
        <TranscriptView turns={transcriptTurns} />
      </div>

      <section>
        <h2 className="app-title" style={{ fontSize: "1.15rem", marginBottom: "var(--space-2)" }}>
          Memory lookups, per turn
        </h2>
        {state.turns.length === 0 && !state.pendingRetrieval && (
          <p className="status-line">Nothing retrieved yet.</p>
        )}
        {state.turns.map(({ retrieval, turn }) =>
          retrieval ? (
            <MemoryLookupCard key={turn.turn_index} retrieval={retrieval} turn={turn} />
          ) : null
        )}
        {state.pendingRetrieval && (
          <MemoryLookupCard retrieval={state.pendingRetrieval} turn={null} />
        )}
      </section>

      <LearningFeed ended={ended} done={state.learningDone} entries={state.learning} />
    </div>
  );
}
