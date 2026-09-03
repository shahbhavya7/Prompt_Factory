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
API_PORT="${API_PORT:-8000}"
NGROK_WEB_PORT="${NGROK_WEB_PORT:-4141}"
API_LOG="$LOG_DIR/api.log"
NGROK_LOG="$LOG_DIR/ngrok.log"

# The interpreter, resolved once. `python3` is not good enough on its own: the
# voice path needs livekit-agents and its plugins, and the machine's default
# python3 routinely does not have them (this repo's run.sh expects a conda env
# named sace-chat, which may not exist). A repo-local .venv is preferred when
# present so every child process below agrees on which interpreter is running,
# rather than the worker failing on `import livekit` twenty lines later.
PY_BIN="${PYTHON:-}"
if [[ -z "$PY_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY_BIN="$ROOT/.venv/bin/python"
  else
    PY_BIN="python3"
  fi
fi

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

# The port is whatever SACE_KB_PORT says (see docker-compose.kb.yml), NOT a
# literal 5433: that number was hardcoded here, so overriding the port to dodge
# a collision made this guard reject the very configuration it was told to use.
# What actually matters is the two things the guard was written to protect —
# the isolated DATABASE NAME, and that it is the port this run publishes.
case "${DATABASE_URL:-}" in
  *:"${SACE_KB_PORT:-5433}"/sace_kb) : ;;
  *) die "DATABASE_URL is not the isolated sace_kb stack on port ${SACE_KB_PORT:-5433} (got: ${DATABASE_URL:-<unset>}) — check .env.kb" ;;
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
if ! "$PY_BIN" -c "import sace_chat.db" 2>"$LOG_DIR/guard-check.log"; then
  cat "$LOG_DIR/guard-check.log" >&2
  die "startup guard refused to continue — DATABASE_URL did not resolve to sace_kb (see above)"
fi
ok "startup guard: DATABASE_URL resolves to sace_kb"

# Fail here, with the fix, rather than in a nohup'd worker log nobody reads.
if ! "$PY_BIN" -c "import livekit.agents, livekit.api" 2>/dev/null; then
  die "livekit-agents is not installed for $PY_BIN — run:
       python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
     (or set PYTHON=/path/to/an/interpreter that has it)"
fi
ok "interpreter: $PY_BIN (livekit-agents present)"

# macOS ships a Python whose ssl module has no CA bundle it can find, and
# livekit's client is aiohttp-based, so every connection to LIVEKIT_URL fails
# with SSLCertVerificationError: unable to get local issuer certificate. It
# looks like a LiveKit outage and is not one — measured here against a live
# project, the same call succeeds the moment SSL_CERT_FILE points at certifi's
# bundle. Exported (not just set) because the worker and the API are separate
# child processes and both need it.
if [[ -z "${SSL_CERT_FILE:-}" ]]; then
  CERT_BUNDLE="$("$PY_BIN" -c "import certifi; print(certifi.where())" 2>/dev/null || true)"
  if [[ -n "$CERT_BUNDLE" && -f "$CERT_BUNDLE" ]]; then
    export SSL_CERT_FILE="$CERT_BUNDLE"
    ok "SSL_CERT_FILE -> $CERT_BUNDLE"
  else
    warn "certifi not found for $PY_BIN — if LiveKit connections fail with"
    warn "SSLCertVerificationError, set SSL_CERT_FILE to a CA bundle"
  fi
fi

# ────────────────────────────────── the KB ───────────────────────────────────
info "renewal knowledge base"
"$PY_BIN" -c "from sace_chat.db import init_db; init_db()" || die "schema init failed"

kb_counts() {
  "$PY_BIN" - <<'PY' 2>/dev/null
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
  "$PY_BIN" scripts/load_kb_renewal.py || die "load_kb_renewal.py failed"
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

NGROK_PID=""
VOICE_PID=""
if port_listening "$VOICE_WS_PORT"; then
  warn "something is already listening on :$VOICE_WS_PORT — assuming a worker from a "
  warn "previous run of this script and not starting a second one"
else
  nohup "$PY_BIN" voice_agent.py dev >"$VOICE_LOG" 2>&1 &
  VOICE_PID=$!
fi

# The FastAPI. Not previously started by this script, because the renewal flow
# drives calls over the spectator websocket rather than HTTP — but the
# call-from-this-device panel needs POST /voice/join to mint a LiveKit token
# and dispatch the agent, and that lives here.
API_PID=""
if port_listening "$API_PORT"; then
  warn "something is already listening on :$API_PORT — assuming the API from a "
  warn "previous run and not starting a second one"
else
  nohup "$PY_BIN" -m uvicorn api.main:app --host 127.0.0.1 --port "$API_PORT" \
    >"$API_LOG" 2>&1 &
  API_PID=$!
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
  for pid in "$VOICE_PID" "$FRONTEND_PID" "$API_PID" "$NGROK_PID"; do
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
    # "registered worker" is THE readiness signal for this worker, and the
    # only one available at boot. The two below cannot fire yet and waiting on
    # them is what made a perfectly healthy worker look dead for 45s:
    # voice_agent.py sets `agent_name` in WorkerOptions, which selects EXPLICIT
    # dispatch, so no job runs until one is dispatched — and `[boot] engine
    # ready` (build_engine) and the :8765 spectator server (entrypoint) both
    # only happen inside a job. They are kept as additional accept conditions
    # because a worker that has already taken a call satisfies them too.
    grep -q 'registered worker' "$VOICE_LOG" 2>/dev/null && return 0
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

wait_for_api() {
  for _ in $(seq 1 45); do
    if [[ -n "$API_PID" ]] && ! kill -0 "$API_PID" 2>/dev/null; then
      warn "the API exited early — last lines of $API_LOG:"
      tail -20 "$API_LOG" | sed 's/^/    /'
      return 1
    fi
    port_listening "$API_PORT" && return 0
    sleep 1
  done
  warn "API did not confirm startup in 45s — last lines of $API_LOG:"
  tail -20 "$API_LOG" | sed 's/^/    /'
  return 1
}

wait_for_api      || die "API failed to start — see $API_LOG"
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

# ───────────────────────────── the tunnel (optional) ─────────────────────────
#
# WHY A TUNNEL IS NEEDED AT ALL, and what it is NOT for.
#
# It does not carry the call. Audio goes phone -> LiveKit Cloud -> agent, and
# LiveKit Cloud is already publicly reachable; the tunnel never sees a packet
# of it. What the tunnel carries is the DASHBOARD, so the phone can load the
# page in the first place.
#
# It is still effectively required for a phone, for a reason that has nothing
# to do with reachability: getUserMedia only works in a SECURE CONTEXT. A phone
# on the same wifi could reach http://192.168.x.x:5173 directly and the browser
# would STILL refuse it the microphone, because plain http on a LAN address is
# not a secure origin. The tunnel is https, so the mic is allowed.
#
# One tunnel is enough because vite.config.js proxies the API and the spectator
# websocket under the same origin — see the comment there.
#
# Skipped, with a note, when ngrok is absent: everything else still runs and
# the dashboard still works locally.
PUBLIC_URL=""

tunnel_url() {
  "$PY_BIN" - "$NGROK_WEB_PORT" <<'NGROKPY' 2>/dev/null
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://localhost:{sys.argv[1]}/api/tunnels", timeout=1) as r:
        for t in json.load(r).get("tunnels", []):
            if t.get("proto") == "https":
                print(t["public_url"])
                break
except Exception:
    pass
NGROKPY
}

start_tunnel() {
  if ! command -v ngrok >/dev/null 2>&1; then
    warn "ngrok not on PATH — skipping the tunnel (dashboard stays local-only)."
    warn "install it (brew install ngrok) and re-run to call from a phone."
    return 1
  fi
  if [[ -z "${NGROK_AUTHTOKEN:-}" ]] \
     && [[ ! -f "$HOME/Library/Application Support/ngrok/ngrok.yml" ]] \
     && [[ ! -f "$HOME/.config/ngrok/ngrok.yml" ]]; then
    warn "ngrok has no authtoken configured — skipping the tunnel."
    warn "run: ngrok config add-authtoken <token>   (or set NGROK_AUTHTOKEN)"
    return 1
  fi

  # ngrok's inspector defaults to :4040, which is routinely already taken — it
  # is on this machine, by an unrelated project's ngrok container. Pinning it
  # to our own port keeps the two from fighting, and gives us a local API to
  # read the public URL back out of.
  #
  # THE OVERLAY IS LAYERED, NOT SUBSTITUTED, and that distinction is the whole
  # of this block. `--config` REPLACES ngrok's default config set rather than
  # adding to it, so passing only our file would hide the user's real config —
  # including the authtoken they set with `ngrok config add-authtoken`, leaving
  # a confusing auth failure. `--config` may be repeated and is merged in
  # order, so the user's config goes first and ours second, overriding nothing
  # but web_addr.
  #
  # The version key has to MATCH the user's file: ngrok refuses to merge
  # configs that declare different schema versions, and a v2 config is still
  # the common case (that is what `ngrok config add-authtoken` writes today).
  # So it is read off their file rather than assumed.
  local default_cfg=""
  for candidate in "$HOME/Library/Application Support/ngrok/ngrok.yml" \
                   "$HOME/.config/ngrok/ngrok.yml"; do
    [[ -f "$candidate" ]] && { default_cfg="$candidate"; break; }
  done

  local cfg_version="2"
  if [[ -n "$default_cfg" ]]; then
    local found
    found="$(sed -n 's/^version:[[:space:]]*["'"'"']\{0,1\}\([0-9]\{1,\}\)["'"'"']\{0,1\}[[:space:]]*$/\1/p' \
             "$default_cfg" | head -1)"
    [[ -n "$found" ]] && cfg_version="$found"
  fi

  local cfg="$LOG_DIR/ngrok.yml"
  {
    echo "version: \"$cfg_version\""
    echo "web_addr: localhost:$NGROK_WEB_PORT"
    # Only when supplied via the environment. A token already in the user's own
    # config arrives through the merge above and must not be duplicated here,
    # where it would land in a log directory.
    [[ -n "${NGROK_AUTHTOKEN:-}" ]] && echo "authtoken: $NGROK_AUTHTOKEN"
  } >"$cfg"

  local -a cfg_args=()
  [[ -n "$default_cfg" ]] && cfg_args+=(--config "$default_cfg")
  cfg_args+=(--config "$cfg")

  # NGROK_EXTRA_ARGS exists so the one-endpoint conflict below can be resolved
  # without editing this file — see the ERR_NGROK_334 branch.
  local -a extra=()
  # shellcheck disable=SC2206
  [[ -n "${NGROK_EXTRA_ARGS:-}" ]] && extra=(${NGROK_EXTRA_ARGS})

  nohup ngrok http "$FRONTEND_PORT" "${cfg_args[@]}" ${extra[@]+"${extra[@]}"} \
    >"$NGROK_LOG" 2>&1 &
  NGROK_PID=$!

  for _ in $(seq 1 30); do
    if ! kill -0 "$NGROK_PID" 2>/dev/null; then
      NGROK_PID=""
      # ERR_NGROK_334 is worth naming, because the generic "ngrok exited" is
      # actively misleading here: nothing is wrong with this project's setup.
      # A free ngrok account may have exactly ONE endpoint online, and the
      # usual cause is another of your own projects already holding it — a
      # long-running `ngrok` container is easy to forget about. Measured on
      # this machine: an unrelated compose stack had one up for a day.
      if grep -q "ERR_NGROK_334" "$NGROK_LOG" 2>/dev/null; then
        local busy
        busy="$(grep -o "https://[a-z0-9.-]*ngrok[a-z0-9.-]*" "$NGROK_LOG" | head -1)"
        warn "ngrok refused: your account already has an endpoint online${busy:+ ($busy)}."
        warn "A free plan allows one at a time, so the other one has to go:"
        warn "  docker ps | grep -i ngrok     # find it"
        warn "  docker stop <container>       # then re-run this script"
        warn "(ngrok suggests --pooling-enabled, but that is NOT a fix here: it"
        warn " load-balances ONE hostname across both tunnels, so roughly half"
        warn " the requests would land on the other project.)"
        warn "The dashboard still works at http://localhost:$FRONTEND_PORT; only the"
        warn "phone path needs the tunnel."
      else
        warn "ngrok exited early — last lines of $NGROK_LOG:"
        tail -15 "$NGROK_LOG" | sed 's/^/    /'
      fi
      return 1
    fi
    PUBLIC_URL="$(tunnel_url)"
    [[ -n "$PUBLIC_URL" ]] && return 0
    sleep 1
  done
  warn "ngrok started but no https tunnel appeared in 30s — see $NGROK_LOG"
  return 1
}

info "public tunnel"
if start_tunnel; then
  ok "tunnel up: $PUBLIC_URL"
fi

echo
info "ready for a call"
printf '  dashboard   http://localhost:%s\n' "$FRONTEND_PORT"
printf '  worker      %s\n' "$LIVEKIT_STATUS"
printf '  renewal kb  %s rules, %s cache entries\n' "$CHUNK_COUNT" "$CACHE_COUNT"
printf '  api         http://localhost:%s\n' "$API_PORT"
if [[ -n "$PUBLIC_URL" ]]; then
  printf '  on a phone  %s\n' "$PUBLIC_URL"
  printf '              open that on the handset, then press "Call Maya" to talk\n'
  printf '              to her from the phone. The call still streams into the\n'
  printf '              desktop dashboard live, exactly as it does now.\n'
else
  printf '  on a phone  (no tunnel — see the warnings above)\n'
fi
printf '  note        the dashboard shows "disconnected" until the first call:\n'
printf '              the spectator socket on :%s is opened by the agent\n' "$VOICE_WS_PORT"
printf '              inside a job, and this worker uses explicit dispatch, so\n'
printf '              no job exists until you press Call Maya. It connects by\n'
printf '              itself a second later.\n'
printf '  logs        %s\n' "$LOG_DIR"
printf '\n  Ctrl-C to stop the worker, API, dashboard and tunnel (the docker stack stays up)\n\n'

wait
