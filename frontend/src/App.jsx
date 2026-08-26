import { useCallback, useEffect, useReducer, useState } from "react";
import { useVoiceSocket } from "./ws/useVoiceSocket";
import { fetchPending } from "./api/review";
import { startAgent, stopAgent, voiceStatus } from "./api/voice";
import CallStatusBar from "./components/CallStatusBar";
import TranscriptView from "./components/TranscriptView";
import MemoryLookupCard from "./components/MemoryLookupCard";
import LearningFeed from "./components/LearningFeed";
import CallControls from "./components/CallControls";
import AudioToggles from "./components/AudioToggles";
import ReviewQueue from "./components/ReviewQueue";

const initialState = {
  sessionId: null,
  // Which CampaignConfig this call is running (see sace_chat.campaign) —
  // null on a build that predates campaigns, which the header treats as
  // "coverage".
  campaign: null,
  callActive: false,
  turns: [], // completed {retrieval + turn} pairs, in order
  pendingRetrieval: null, // a retrieval event with no matching turn yet
  learning: [],
  learningDone: false,
  // Broadcast with learning_done: how many rules are now waiting on a human,
  // including leftovers from earlier calls. Lets the tab badge update the
  // moment a call ends without polling the HTTP API mid-call.
  pendingCount: null,
  // A start has been asked for but "call_started" has not come back yet. The
  // agent has to build an AgentSession, connect Deepgram STT/TTS and load VAD,
  // which is a couple of seconds — long enough that an unacknowledged click
  // reads as a dead button.
  starting: false,
  // Why the agent refused to start, if it did. Shown until the next attempt.
  startError: null,
  // Pushed by the agent on call start and after every toggle. Null until then,
  // which is what hides the switches when there is nothing to switch.
  audio: null,
};

function reducer(state, event) {
  switch (event.type) {
    // Dispatched locally by the Start button, not sent by the agent — the
    // pending state has to appear on click rather than a round-trip later.
    case "start_requested":
      return { ...state, starting: true, startError: null };
    case "start_refused":
      return { ...state, starting: false, startError: event.reason || "unknown reason" };
    case "call_started":
      return {
        ...initialState,
        // Survives the reset: it describes the queue, not this call.
        pendingCount: state.pendingCount,
        sessionId: event.session_id,
        campaign: event.campaign || null,
        callActive: true,
      };
    case "call_ended":
      // audio is cleared with the call: the toggles act on a live agent, and
      // leaving them on screen after it exits would invite clicks that go
      // nowhere.
      return { ...state, callActive: false, starting: false, audio: null };
    case "audio_state":
      return { ...state, audio: event };
    case "retrieval":
      // A retrieval landing at all is itself proof a call is in progress —
      // covers the dashboard-opened-mid-call case where "call_started" was
      // never seen (see hasSeenACall in App.jsx for the matching reasoning).
      return { ...state, pendingRetrieval: event, callActive: true, starting: false };
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
  const { connected, endCall, startCall, setAudio } = useVoiceSocket(dispatch);
  const [tab, setTab] = useState("live");
  const [pendingCount, setPendingCount] = useState(null);
  // Bumped to make ReviewQueue refetch — on tab entry, and whenever a call's
  // learning run just queued something new.
  const [refreshToken, setRefreshToken] = useState(0);
  // Spawning voice_agent.py over HTTP — separate from the reducer's `starting`,
  // which tracks a CALL being started on an agent that is already up.
  const [agentBooting, setAgentBooting] = useState(false);
  const [agentStopping, setAgentStopping] = useState(false);
  const [agentError, setAgentError] = useState(null);
  // Is a voice_agent PROCESS alive, whether or not its websocket is up? The two
  // differ exactly when it matters: a crashed or stranded agent still holds
  // port 8765 while the dashboard reads "not connected", and that is when the
  // kill button needs to be offered.
  const [processAlive, setProcessAlive] = useState(false);

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

  // Polled rather than pushed, because the signal is needed precisely when the
  // websocket — the only push channel — is down. Slow (4s): it drives a
  // secondary button, and this is a local demo API.
  useEffect(() => {
    let cancelled = false;
    const read = () =>
      voiceStatus()
        .then((s) => {
          if (!cancelled) setProcessAlive(!!s.running);
        })
        .catch(() => {
          // API down: it cannot report a stray agent, and it could not kill one
          // either, so claiming none is the honest state for the button.
          if (!cancelled) setProcessAlive(false);
        });
    read();
    const id = setInterval(read, 4000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [connected, agentBooting, agentStopping]);

  const handleCountChange = useCallback((n) => setPendingCount(n), []);

  // The optimistic "starting" flag is set here rather than in the socket hook
  // so it clears through the same reducer that handles call_started /
  // start_refused — one place decides what the button says.
  const handleStart = useCallback(() => {
    if (startCall()) dispatch({ type: "start_requested" });
  }, [startCall]);

  // Bring voice_agent.py itself up. Once it boots it opens the spectator
  // websocket, useVoiceSocket's reconnect loop picks it up within a couple of
  // seconds, and the button becomes "Start call" on its own — so there is
  // nothing to do on success but wait for `connected` to flip.
  // Fire-and-forget: the agent echoes an audio_state back, and THAT is what
  // updates the switches. No optimistic flip — a request to enable a layer that
  // is unavailable is ignored, and the UI must show what is really running.
  const handleAudioChange = useCallback((opts) => setAudio(opts), [setAudio]);

  const handleStartAgent = useCallback(() => {
    setAgentBooting(true);
    setAgentError(null);
    startAgent()
      .catch((err) => setAgentError(err.message || String(err)))
      // Cleared regardless: on success the websocket takes over the button's
      // state, and on failure the error is what the user needs to see.
      .finally(() => setAgentBooting(false));
  }, []);

  // Shut the agent down and free the websocket port. Distinct from End call,
  // which ends the CALL and leaves the agent up for the next one — this is for
  // when the session is over. Takes a few seconds, because the agent runs its
  // learning loop on SIGINT rather than being killed outright.
  const handleStopAgent = useCallback(() => {
    setAgentStopping(true);
    setAgentError(null);
    stopAgent()
      .then((r) => {
        if (!r.stopped && r.reason) setAgentError(r.reason);
      })
      .catch((err) => setAgentError(err.message || String(err)))
      .finally(() => setAgentStopping(false));
  }, []);

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
          <h1 className="app-title">Maya live </h1>
          <p className="app-subtitle">Watching voice_agent.py in real time</p>
        </div>
        {tab === "live" && (
          <CallControls
            connected={connected}
            callActive={state.callActive}
            starting={state.starting}
            agentBooting={agentBooting}
            agentStopping={agentStopping}
            processAlive={processAlive}
            onStartAgent={handleStartAgent}
            onStopAgent={handleStopAgent}
            onStart={handleStart}
            onEnd={endCall}
          />
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
            campaign={state.campaign}
            starting={state.starting}
            agentBooting={agentBooting}
            startError={state.startError || agentError}
          />

          <AudioToggles
            state={state.audio}
            onChange={handleAudioChange}
            disabled={!connected}
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
