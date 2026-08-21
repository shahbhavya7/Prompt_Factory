import { useCallback, useEffect, useReducer, useState } from "react";
import { useVoiceSocket } from "./ws/useVoiceSocket";
import { fetchPending } from "./api/review";
import CallStatusBar from "./components/CallStatusBar";
import TranscriptView from "./components/TranscriptView";
import MemoryLookupCard from "./components/MemoryLookupCard";
import LearningFeed from "./components/LearningFeed";
import EndCallButton from "./components/EndCallButton";
import ReviewQueue from "./components/ReviewQueue";

const initialState = {
  sessionId: null,
  callActive: false,
  turns: [], // completed {retrieval + turn} pairs, in order
  pendingRetrieval: null, // a retrieval event with no matching turn yet
  learning: [],
  learningDone: false,
  // Broadcast with learning_done: how many rules are now waiting on a human,
  // including leftovers from earlier calls. Lets the tab badge update the
  // moment a call ends without polling the HTTP API mid-call.
  pendingCount: null,
};

function reducer(state, event) {
  switch (event.type) {
    case "call_started":
      return {
        ...initialState,
        // Survives the reset: it describes the queue, not this call.
        pendingCount: state.pendingCount,
        sessionId: event.session_id,
        callActive: true,
      };
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
      return {
        ...state,
        learningDone: true,
        pendingCount:
          typeof event.pending_count === "number" ? event.pending_count : state.pendingCount,
      };
    default:
      return state;
  }
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { connected, endCall } = useVoiceSocket(dispatch);
  const [tab, setTab] = useState("live");
  const [pendingCount, setPendingCount] = useState(null);
  // Bumped to make ReviewQueue refetch — on tab entry, and whenever a call's
  // learning run just queued something new.
  const [refreshToken, setRefreshToken] = useState(0);

  // The count the socket reports (fresh after a call) wins over the last HTTP
  // read, so the badge is right whether or not the review tab has been opened.
  useEffect(() => {
    if (typeof state.pendingCount === "number") {
      setPendingCount(state.pendingCount);
      setRefreshToken((n) => n + 1);
    }
  }, [state.pendingCount]);

  // One read at mount so the badge is populated before any call happens —
  // the queue usually has a backlog from previous sessions.
  useEffect(() => {
    let cancelled = false;
    fetchPending()
      .then((d) => {
        if (!cancelled) setPendingCount(d.count ?? 0);
      })
      .catch(() => {
        /* API down — the review tab surfaces the error properly when opened */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleCountChange = useCallback((n) => setPendingCount(n), []);

  const openReview = useCallback(() => {
    setTab("review");
    setRefreshToken((n) => n + 1);
  }, []);

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
        {tab === "live" && (
          <EndCallButton disabled={!connected || !state.callActive} onClick={endCall} />
        )}
      </header>

      <nav className="tabs" role="tablist">
        <button
          role="tab"
          aria-selected={tab === "live"}
          className={`tab ${tab === "live" ? "tab--active" : ""}`}
          onClick={() => setTab("live")}
        >
          Live call
        </button>
        <button
          role="tab"
          aria-selected={tab === "review"}
          className={`tab ${tab === "review" ? "tab--active" : ""}`}
          onClick={openReview}
        >
          Pending review
          {pendingCount > 0 && <span className="tab__badge">{pendingCount}</span>}
        </button>
      </nav>

      {tab === "live" ? (
        <>
          <CallStatusBar
            connected={connected}
            callActive={state.callActive}
            sessionId={state.sessionId}
          />

          <div className="chat-view">
            <TranscriptView turns={transcriptTurns} />
          </div>

          <section>
            <h2
              className="app-title"
              style={{ fontSize: "1.15rem", marginBottom: "var(--space-2)" }}
            >
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

          <LearningFeed
            ended={ended}
            done={state.learningDone}
            entries={state.learning}
            onReview={openReview}
          />
        </>
      ) : (
        <ReviewQueue onCountChange={handleCountChange} refreshToken={refreshToken} />
      )}
    </div>
  );
}
