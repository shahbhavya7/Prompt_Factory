#!/usr/bin/env bash
#
# One command to manually test the renewal campaign by phone call: brings up
# the isolated sace_kb stack, loads the KB if needed, starts the LiveKit
# worker and the live dashboard, and prints everything a tester needs.
#
# Deliberately NOT the repo-root run.sh (which is the shared/coverage stack's
# CLI, and api/main.py spawns `./run.sh voice` as a subprocess — overwriting
# it would break the dashboard's "Start agent & call" button for that stack).
# This script is the renewal-only counterpart to scripts/reset_kb_db.sh.
#
#   ./scripts/run_renewal.sh
#
# Ctrl-C stops the worker and the dashboard cleanly. It does NOT tear down
# the docker stack — that's scripts/reset_kb_db.sh's job, a separate and
# deliberate action, not something a Ctrl-C should trigger as a side effect.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE=(docker compose -f docker-compose.kb.yml)
LOG_DIR="$ROOT/logs"
VOICE_LOG="$LOG_DIR/voice_agent.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
DOCKER_LOG="$LOG_DIR/docker-up.log"
VOICE_WS_PORT="${VOICE_WS_PORT:-8765}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
YLW=$'\033[33m'; RST=$'\033[0m'

info() { printf '%s==>%s %s\n' "$BOLD" "$RST" "$*"; }
ok()   { printf '  %s✓%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '  %s!%s %s\n' "$YLW" "$RST" "$*"; }
die()  { printf '  %s✗%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

mkdir -p "$LOG_DIR"

# ─────────────────────────────────── env ────────────────────────────────────
# .env first (LiveKit/Deepgram/LLM keys — shared across stacks), THEN .env.kb
# (overrides DATABASE_URL/SACE_EXPECTED_DB/EMBEDDING_* to the isolated stack),
# THEN SACE_CAMPAIGN — exported here so every child process gets them without
# depending on the caller's shell already having them set.
[[ -f .env ]] || die ".env missing — copy .env.example and fill it in"
[[ -f .env.kb ]] || die ".env.kb missing — copy .env.kb.example and fill it in"
set -a
# shellcheck disable=SC1091
source .env
# shellcheck disable=SC1091
source .env.kb
set +a
export SACE_CAMPAIGN=renewal

case "${DATABASE_URL:-}" in
  *:5433/sace_kb) : ;;
  *) die "DATABASE_URL is not the isolated sace_kb stack (got: ${DATABASE_URL:-<unset>}) — check .env.kb" ;;
esac

[[ -n "${DEEPGRAM_API_KEY:-}" ]] || die "DEEPGRAM_API_KEY unset — voice needs Deepgram for STT/TTS"
case "${LIVEKIT_URL:-}" in
  ""|*your-project*) die "LIVEKIT_URL is unset or still the placeholder in .env" ;;
esac
[[ -n "${LIVEKIT_API_KEY:-}" && -n "${LIVEKIT_API_SECRET:-}" ]] || die "LIVEKIT_API_KEY / LIVEKIT_API_SECRET unset"

# ────────────────────────────────── docker ───────────────────────────────────
info "isolated sace_kb stack"
docker info >/dev/null 2>&1 || die "Docker isn't running — start Docker Desktop first"

stack_healthy() {
  "${COMPOSE[@]}" ps --format '{{.Health}}' db 2>/dev/null | grep -qi healthy
}

if stack_healthy; then
  ok "sace-kb-db already healthy — not restarting it"
else
  if ! "${COMPOSE[@]}" up -d >"$DOCKER_LOG" 2>&1; then
    if grep -qi "port is already allocated\|address already in use\|bind.*5433" "$DOCKER_LOG"; then
      die "port 5433 is already in use by something else (not our sace-kb-db) — free it, or see: lsof -i :5433"
    fi
    die "docker compose up failed — see $DOCKER_LOG"
  fi
  printf '  waiting for postgres'
  up=0
  for _ in $(seq 1 45); do
    if stack_healthy; then up=1; break; fi
    printf '.'; sleep 1
  done
  printf '\n'
  [[ "$up" == "1" ]] || die "sace-kb-db did not become healthy in 45s — ${COMPOSE[*]} logs db"
  ok "sace-kb-db healthy on :5433"
fi

# Startup guard, for real: import sace_chat.db under this exact env and let
# its own SACE_EXPECTED_DB check (sace_chat/db.py) fire if anything resolved
# to the wrong database. A clear message here, not a stack trace.
if ! python3 -c "import sace_chat.db" 2>"$LOG_DIR/guard-check.log"; then
  cat "$LOG_DIR/guard-check.log" >&2
  die "startup guard refused to continue — DATABASE_URL did not resolve to sace_kb (see above)"
fi
ok "startup guard: DATABASE_URL resolves to sace_kb"

# ────────────────────────────────── the KB ───────────────────────────────────
info "renewal knowledge base"
python3 -c "from sace_chat.db import init_db; init_db()" || die "schema init failed"

kb_counts() {
  python3 - <<'PY' 2>/dev/null
from sqlalchemy import text
from sace_chat.db import engine
with engine.connect() as c:
    chunks = c.execute(text("SELECT count(*) FROM chunks_renewal")).scalar()
    cache = c.execute(text("SELECT count(*) FROM answer_cache_renewal")).scalar()
print(f"{chunks} {cache}")
PY
}

read -r CHUNK_COUNT CACHE_COUNT <<< "$(kb_counts)"
if [[ -z "${CHUNK_COUNT:-}" ]]; then
  die "cannot read chunks_renewal — is the isolated database really up?"
elif [[ "$CHUNK_COUNT" == "0" ]]; then
  warn "chunks_renewal is empty — loading (scripts/load_kb_renewal.py)"
  python3 scripts/load_kb_renewal.py || die "load_kb_renewal.py failed"
  read -r CHUNK_COUNT CACHE_COUNT <<< "$(kb_counts)"
  ok "loaded $CHUNK_COUNT rules, $CACHE_COUNT cache entries"
else
  # Idempotent on purpose: reloading on every run defeats the point of the
  # loader being idempotent, and is wasted embedding calls besides.
  ok "chunks_renewal already has $CHUNK_COUNT rules, $CACHE_COUNT cache entries — not reloading"
fi

# ─────────────────────────── worker + dashboard ──────────────────────────────
info "voice worker + dashboard"
set -m  # background jobs get their own process group, so cleanup can kill it whole

port_listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

VOICE_PID=""
if port_listening "$VOICE_WS_PORT"; then
  warn "something is already listening on :$VOICE_WS_PORT — assuming a worker from a "
  warn "previous run of this script and not starting a second one"
else
  nohup python3 voice_agent.py dev >"$VOICE_LOG" 2>&1 &
  VOICE_PID=$!
fi

FRONTEND_PID=""
if port_listening "$FRONTEND_PORT"; then
  warn "something is already listening on :$FRONTEND_PORT — assuming the dashboard "
  warn "from a previous run and not starting a second one"
else
  (cd frontend && nohup npm run dev -- --port "$FRONTEND_PORT" >"$FRONTEND_LOG" 2>&1) &
  FRONTEND_PID=$!
fi

cleanup() {
  echo
  info "stopping (docker stack left running — scripts/reset_kb_db.sh tears that down)"
  for pid in "$VOICE_PID" "$FRONTEND_PID"; do
    [[ -n "$pid" ]] || continue
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  done
  wait 2>/dev/null
  ok "stopped"
  exit 0
}
trap cleanup INT TERM

# Poll each background process for a real sign of life rather than guessing
# with a fixed sleep — a dead process fails this immediately with a useful
# log tail instead of a 45s wait for nothing.
wait_for_worker() {
  for _ in $(seq 1 45); do
    if [[ -n "$VOICE_PID" ]] && ! kill -0 "$VOICE_PID" 2>/dev/null; then
      warn "voice_agent.py exited early — last lines of $VOICE_LOG:"
      tail -20 "$VOICE_LOG" | sed 's/^/    /'
      return 1
    fi
    grep -q '\[boot\] engine ready' "$VOICE_LOG" 2>/dev/null && return 0
    port_listening "$VOICE_WS_PORT" && return 0
    sleep 1
  done
  warn "worker did not confirm startup in 45s — last lines of $VOICE_LOG:"
  tail -20 "$VOICE_LOG" | sed 's/^/    /'
  return 1
}

wait_for_frontend() {
  for _ in $(seq 1 45); do
    if [[ -n "$FRONTEND_PID" ]] && ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
      warn "the dashboard dev server exited early — last lines of $FRONTEND_LOG:"
      tail -20 "$FRONTEND_LOG" | sed 's/^/    /'
      return 1
    fi
    port_listening "$FRONTEND_PORT" && return 0
    sleep 1
  done
  warn "dashboard did not confirm startup in 45s — last lines of $FRONTEND_LOG:"
  tail -20 "$FRONTEND_LOG" | sed 's/^/    /'
  return 1
}

wait_for_worker   || die "voice worker failed to start — see $VOICE_LOG"
wait_for_frontend || die "dashboard failed to start — see $FRONTEND_LOG"

# livekit-agents' own worker-registration log line isn't something this repo
# controls the wording of, so this looks for it best-effort rather than
# asserting on exact text; [boot] engine ready (our own line, already
# confirmed above) is the reliable half of "the worker is really up."
if grep -qiE "registered worker|worker.*(ready|running)|connected to" "$VOICE_LOG" 2>/dev/null; then
  LIVEKIT_STATUS="connected (see $VOICE_LOG for the exact line)"
else
  LIVEKIT_STATUS="process is up; check $VOICE_LOG for LiveKit's own connection confirmation"
fi

echo
info "ready for a call"
printf '  dashboard   http://localhost:%s\n' "$FRONTEND_PORT"
printf '  worker      %s\n' "$LIVEKIT_STATUS"
printf '  renewal kb  %s rules, %s cache entries\n' "$CHUNK_COUNT" "$CACHE_COUNT"
printf '  logs        %s\n' "$LOG_DIR"
printf '\n  Ctrl-C to stop the worker and dashboard (the docker stack stays up)\n\n'

wait
