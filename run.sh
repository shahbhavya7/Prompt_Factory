#!/usr/bin/env bash
#
# sace-chat runner.
#
# The Postgres/pgvector container, the Streamlit engineering app, an (older)
# FastAPI + React text-chat demo, and the React app in frontend/ — now a
# LIVE SPECTATOR DASHBOARD for voice_agent.py, not a chat client. Watching a
# call needs no separate backend: voice_agent.py itself opens a small
# websocket (VOICE_WS_PORT, default 8765) and pushes retrieval/turn/learned
# events straight to any browser tab running `cd frontend && npm run dev` —
# purely additive, nothing flows back into the call. streamlit_app.py and
# api/main.py (the older text-chat demo's backend) are untouched by this and
# still import the engine in-process the same way they always did.
#
#   ./run.sh start        db + app in the BACKGROUND (logs go to a file)
#   ./run.sh dev          db + app in the FOREGROUND, logs live in this terminal
#   ./run.sh demo         db + the (older) FastAPI text-chat demo backend
#   ./run.sh voice        db + voice agent in console mode — talk to it with your
#                         mic; per-turn logs stream live, AND to the live dashboard
#                         if `cd frontend && npm run dev` is running. No LiveKit creds needed.
#   ./run.sh voice-dev    voice agent as a LiveKit worker (needs a real
#                         LIVEKIT_URL + key/secret), logs live
#   ./run.sh all          db + dashboard + local audible voice agent
#   ./run.sh all-livekit  db + dashboard + LiveKit worker waiting for rooms
#   ./run.sh stop         stop the app and the db container
#   ./run.sh restart
#   ./run.sh status       what's up, on which ports, with row counts
#   ./run.sh logs [app|db]   follow the background app's log
#   ./run.sh reload-kb    re-embed the seed rules; keeps learned rules
#   ./run.sh reset-kb     drop and rebuild the table; LOSES learned rules
#   ./run.sh repair       audit and re-embed any null/zero/wrong-dim vectors
#   ./run.sh shell        psql into the database
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CONDA_ENV="sace-chat"
APP_PORT="${APP_PORT:-8501}"
API_PORT="${API_PORT:-8000}"
DB_PORT="5434"          # off 5432 (host Postgres) and off 5433, which on this
                        # machine is already held by an unrelated project's
                        # container (atlas-db-1) — picked 5434 to avoid both.
APP_LOG="/tmp/sace-chat-app.log"
APP_PIDFILE="/tmp/sace-chat-app.pid"

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
YLW=$'\033[33m'; RST=$'\033[0m'

info() { printf '%s==>%s %s\n' "$BOLD" "$RST" "$*"; }
ok()   { printf '  %s✓%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '  %s!%s %s\n' "$YLW" "$RST" "$*"; }
die()  { printf '  %s✗%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

# ---------------------------------------------------------------- environment

activate_env() {
  local base
  base="$(conda info --base 2>/dev/null)" || die "conda not found on PATH"
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || die "conda env '$CONDA_ENV' not found (conda create -n $CONDA_ENV python=3.11)"
}

load_dotenv() {
  [[ -f .env ]] || die ".env missing — needs DATABASE_URL, SACE_LLM_KEY, EMBEDDING_MODE"
  set -a; # shellcheck disable=SC1091
  source .env; set +a
}

check_env_sanity() {
  # The vector column's dimension is fixed at table-creation time, so a mode
  # change without a matching dim is the classic 'different vector dimensions
  # 1536 and 384' failure. Catch it here instead of at query time.
  local mode="${EMBEDDING_MODE:-mock}" dim="${EMBEDDING_DIM:-384}"
  case "$mode" in
    openai) [[ "$dim" == "1536" ]] || warn "EMBEDDING_MODE=openai but EMBEDDING_DIM=$dim (expected 1536) — reload-kb after fixing" ;;
    mock)   [[ "$dim" == "384"  ]] || warn "EMBEDDING_MODE=mock but EMBEDDING_DIM=$dim (expected 384) — reload-kb after fixing" ;;
  esac
  [[ -n "${SACE_LLM_KEY:-}" ]] || warn "SACE_LLM_KEY unset — the app will fall back to MockLLM"
}

# ------------------------------------------------------------------------ db

db_running()  { [[ -n "$(docker compose ps -q db 2>/dev/null)" ]]; }
db_healthy()  { docker compose ps --format '{{.Status}}' 2>/dev/null | head -1 | grep -qi healthy; }

start_db() {
  info "database"
  docker info >/dev/null 2>&1 || die "Docker isn't running — start Docker Desktop first"

  if db_healthy; then
    ok "already healthy on :$DB_PORT"
  else
    docker compose up -d >/dev/null 2>&1 || die "docker compose up failed (see: docker compose logs db)"
    printf '  waiting for health'
    for _ in $(seq 1 45); do
      if db_healthy; then printf '\n'; ok "healthy on :$DB_PORT"; return 0; fi
      printf '.'; sleep 1
    done
    printf '\n'
    die "database did not become healthy in 45s — docker compose logs db"
  fi
}

chunk_count() {
  python - <<'PY' 2>/dev/null
from sqlalchemy import text
from sace_chat.db import engine
try:
    with engine.connect() as c:
        print(c.execute(text("select count(*) from chunks")).scalar())
except Exception:
    print(-1)
PY
}

ensure_kb() {
  info "knowledge base"
  python -c "from sace_chat.db import init_db; init_db()" 2>/dev/null || die "schema init failed — is the db up?"
  local n; n="$(chunk_count)"
  if [[ "$n" == "-1" ]]; then
    die "cannot read the chunks table"
  elif [[ "$n" == "0" ]]; then
    warn "chunks table empty — loading kb.py (this embeds every rule)"
    python load_kb.py || die "load_kb.py failed"
    ok "loaded $(chunk_count) chunks"
  else
    ok "$n chunks present"
    local learned
    learned="$(python - <<'PY' 2>/dev/null
from sqlalchemy import text
from sace_chat.db import engine
with engine.connect() as c:
    print(c.execute(text("select count(*) from chunks where learned_kind is not null")).scalar())
PY
)"
    [[ "${learned:-0}" != "0" ]] && ok "$learned learned rule(s) from the consolidator"
    local review
    review="$(python - <<'PY' 2>/dev/null
from sqlalchemy import text
from sace_chat.db import engine
with engine.connect() as c:
    print(c.execute(text("select count(*) from needs_review")).scalar())
PY
)"
    [[ "${review:-0}" != "0" ]] && warn "$review row(s) in needs_review awaiting a human"
  fi
}

reload_kb() {
  activate_env; load_dotenv; check_env_sanity; start_db
  info "reloading knowledge base"
  # load_kb.py replaces the seed rules and leaves learned ones alone, so this
  # deliberately does NOT drop the table — dropping it would throw away
  # everything the consolidator has learned.
  warn "re-embeds every seed rule; learned rules are kept"
  python -c "from sace_chat.db import init_db; init_db()" || die "schema init failed"
  python load_kb.py || die "load_kb.py failed"
  ok "reloaded to $(chunk_count) rules at EMBEDDING_MODE=${EMBEDDING_MODE:-mock}"
}

# Wipes learned rules too — only for an embedding-model change, where every
# stored vector is the wrong dimension and nothing in the table is salvageable.
reset_kb() {
  activate_env; load_dotenv; check_env_sanity; start_db
  info "resetting knowledge base"
  warn "this DROPS the chunks table — learned rules will be lost"
  python -c "
from sqlalchemy import text
from sace_chat.db import engine
with engine.begin() as c:
    c.execute(text('DROP TABLE IF EXISTS chunks'))
print('  dropped chunks table')
" || die "drop failed"
  python load_kb.py || die "load_kb.py failed"
  ok "reset to $(chunk_count) rules at EMBEDDING_MODE=${EMBEDDING_MODE:-mock}"
}

# ----------------------------------------------------------------------- app

app_pid() {
  # Trust the pidfile only if that pid is still our streamlit process.
  if [[ -f "$APP_PIDFILE" ]]; then
    local p; p="$(cat "$APP_PIDFILE" 2>/dev/null)"
    if [[ -n "$p" ]] && ps -p "$p" -o command= 2>/dev/null | grep -q streamlit; then
      echo "$p"; return 0
    fi
  fi
  pgrep -f "streamlit run streamlit_app.py" | head -1
}

start_app() {
  info "app"
  local existing; existing="$(app_pid)"
  if [[ -n "$existing" ]]; then
    warn "already running (pid $existing) — restarting so it picks up .env and code changes"
    stop_app
  fi

  nohup streamlit run streamlit_app.py \
    --server.port "$APP_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    > "$APP_LOG" 2>&1 &
  echo $! > "$APP_PIDFILE"

  for _ in $(seq 1 45); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://localhost:$APP_PORT/" 2>/dev/null)" == "200" ]]; then
      ok "http://localhost:$APP_PORT"
      return 0
    fi
    sleep 1
  done
  warn "app did not answer on :$APP_PORT in 45s — last lines of $APP_LOG:"
  tail -15 "$APP_LOG" | sed 's/^/    /'
  return 1
}

stop_app() {
  local p; p="$(app_pid)"
  if [[ -n "$p" ]]; then
    kill "$p" 2>/dev/null
    for _ in $(seq 1 10); do ps -p "$p" >/dev/null 2>&1 || break; sleep 1; done
    ps -p "$p" >/dev/null 2>&1 && kill -9 "$p" 2>/dev/null
    ok "app stopped (pid $p)"
  else
    ok "app not running"
  fi
  rm -f "$APP_PIDFILE"
}

# ------------------------------------------------------------------ commands

cmd_start() {
  activate_env; load_dotenv; check_env_sanity
  start_db
  ensure_kb
  start_app || exit 1
  echo
  info "ready"
  printf '  app     http://localhost:%s\n' "$APP_PORT"
  printf '  db      postgres://localhost:%s\n' "$DB_PORT"
  printf '  model   %s\n' "${SACE_LLM_MODEL:-<mock>}"
  printf '  logs    ./run.sh logs app\n'
}

# Foreground: the process owns this terminal and its logs stream here. Ctrl-C
# stops it. Use this rather than `start` when you want to watch what happens.
cmd_dev() {
  activate_env; load_dotenv; check_env_sanity
  start_db
  ensure_kb
  stop_app                      # never leave a background copy on the same port
  echo
  info "app in the foreground — logs below, Ctrl-C to stop"
  printf '  http://localhost:%s\n\n' "$APP_PORT"
  exec streamlit run streamlit_app.py \
    --server.port "$APP_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
}

# The stakeholder demo's backend only — same Engine, wrapped as a REST API for
# the React app in frontend/. Run `cd frontend && npm install && npm run dev`
# in a second terminal for the UI itself; that dev server is a separate
# process this script does not manage.
cmd_demo() {
  activate_env; load_dotenv; check_env_sanity
  start_db
  ensure_kb
  echo
  info "demo API in the foreground — logs below, Ctrl-C to stop"
  printf '  api     http://localhost:%s\n' "$API_PORT"
  printf '  then, in another terminal: cd frontend && npm install && npm run dev\n\n'
  exec uvicorn api.main:app --reload --port "$API_PORT"
}

# Console mode talks to your microphone directly and needs NO LiveKit
# credentials — it runs a local mock job. This is the way to place a test call.
cmd_voice() {
  activate_env; load_dotenv; check_env_sanity
  [[ -n "${DEEPGRAM_API_KEY:-}" ]] || die "DEEPGRAM_API_KEY unset — voice needs Deepgram for STT and TTS"
  start_db
  ensure_kb
  echo
  info "voice agent · console mode · speak into your mic"
  printf '  one line per turn: intent, governing rule, tokens, grounding cosine, latency\n'
  printf '  turns are also written to the `turns` table\n'
  printf '  live dashboard: cd frontend && npm run dev, then open it — same events,\n'
  printf '  streamed over ws://localhost:%s as the call happens\n' "${VOICE_WS_PORT:-8765}"
  printf '  Ctrl-C to hang up (the learning loop runs on shutdown)\n\n'
  exec python voice_agent.py console
}

# Worker mode: registers with LiveKit and waits for rooms. Needs real creds.
cmd_voice_dev() {
  activate_env; load_dotenv; check_env_sanity
  [[ -n "${DEEPGRAM_API_KEY:-}" ]] || die "DEEPGRAM_API_KEY unset"
  case "${LIVEKIT_URL:-}" in
    ""|*your-project*) die "LIVEKIT_URL is unset or still the placeholder — worker mode needs a real project URL (use './run.sh voice' for a local mic call instead)" ;;
  esac
  [[ -n "${LIVEKIT_API_KEY:-}" && -n "${LIVEKIT_API_SECRET:-}" ]] || die "LIVEKIT_API_KEY / LIVEKIT_API_SECRET unset"
  start_db
  ensure_kb
  echo
  info "voice agent · LiveKit worker · ${LIVEKIT_URL}"
  printf '  live dashboard: cd frontend && npm run dev, then open it\n\n'
  exec python voice_agent.py dev
}

cmd_all() {
  activate_env; load_dotenv; check_env_sanity
  [[ -n "${DEEPGRAM_API_KEY:-}" ]] || die "DEEPGRAM_API_KEY unset"
  start_db
  ensure_kb
  start_app || exit 1
  echo
  info "dashboard + local voice agent"
  printf '  dashboard http://localhost:%s\n' "$APP_PORT"
  printf '  audio     local microphone + speakers\n\n'
  exec python voice_agent.py console
}

cmd_all_livekit() {
  activate_env; load_dotenv; check_env_sanity
  [[ -n "${DEEPGRAM_API_KEY:-}" ]] || die "DEEPGRAM_API_KEY unset"
  case "${LIVEKIT_URL:-}" in
    ""|*your-project*) die "LIVEKIT_URL is unset or still the placeholder — all-livekit mode needs a real LiveKit project URL" ;;
  esac
  [[ -n "${LIVEKIT_API_KEY:-}" && -n "${LIVEKIT_API_SECRET:-}" ]] || die "LIVEKIT_API_KEY / LIVEKIT_API_SECRET unset"
  start_db
  ensure_kb
  start_app || exit 1
  echo
  info "dashboard + LiveKit voice worker"
  printf '  dashboard http://localhost:%s\n' "$APP_PORT"
  printf '  worker    %s\n\n' "$LIVEKIT_URL"
  exec python voice_agent.py dev
}

cmd_stop() {
  info "stopping"
  stop_app
  if db_running; then
    docker compose down >/dev/null 2>&1 && ok "db stopped (volume kept — data survives)"
  else
    ok "db not running"
  fi
  warn "the host's own Postgres on :5432 was left alone"
}

cmd_status() {
  info "status"
  if db_healthy;   then ok "db   healthy on :$DB_PORT"
  elif db_running; then warn "db   running but not healthy"
  else                  warn "db   stopped"; fi

  local p; p="$(app_pid)"
  if [[ -n "$p" ]]; then
    local code; code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://localhost:$APP_PORT/" 2>/dev/null)"
    if [[ "$code" == "200" ]]; then ok "app  http://localhost:$APP_PORT (pid $p)"
    else warn "app  pid $p alive but HTTP $code on :$APP_PORT"; fi
  else
    warn "app  stopped"
  fi

  if db_healthy; then
    activate_env; load_dotenv
    local n; n="$(chunk_count)"
    [[ "$n" != "-1" ]] && ok "kb   $n chunks" || warn "kb   chunks table unreadable"
    printf '  %sembeddings=%s dim=%s model=%s%s\n' \
      "$DIM" "${EMBEDDING_MODE:-mock}" "${EMBEDDING_DIM:-384}" "${SACE_LLM_MODEL:-<mock>}" "$RST"
  fi
}

cmd_logs() {
  case "${1:-app}" in
    app) [[ -f "$APP_LOG" ]] || die "no app log yet"; tail -f "$APP_LOG" ;;
    db)  docker compose logs -f db ;;
    *)   die "logs takes 'app' or 'db'" ;;
  esac
}

cmd_shell() {
  load_dotenv
  db_healthy || die "db isn't up — ./run.sh start"
  docker compose exec db psql -U sace -d sace_chat
}

case "${1:-start}" in
  start)     cmd_start ;;
  all)       cmd_all ;;
  all-livekit) cmd_all_livekit ;;
  dev)       cmd_dev ;;
  demo)      cmd_demo ;;
  voice)     cmd_voice ;;
  voice-dev) cmd_voice_dev ;;
  stop)      cmd_stop ;;
  restart)   cmd_stop; echo; cmd_start ;;
  status)    cmd_status ;;
  logs)      cmd_logs "${2:-app}" ;;
  reload-kb) reload_kb ;;
  reset-kb)  reset_kb ;;
  repair)    activate_env; load_dotenv; python scripts/repair_embeddings.py "${2:---fix}" ;;
  shell)     cmd_shell ;;
  -h|--help|help)
    # Print the leading comment block and stop at the first line that is not a
    # comment, so the help text cannot drift out of sync with a line number.
    awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}" ;;
  *) die "unknown command '${1}' — try: start all all-livekit dev demo voice voice-dev stop restart status logs reload-kb reset-kb repair shell" ;;
esac
