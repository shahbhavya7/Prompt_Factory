#!/usr/bin/env bash
# Nuke and reload the isolated renewal-KB database (docker-compose.kb.yml).
#
# This is most of the value of isolating sace_kb in the first place: you can
# run this freely while tuning the KB, with zero risk to the shared sace_chat
# database other branches use.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE=(docker compose -f docker-compose.kb.yml)

echo "==> tearing down sace-kb-db (drops the volume: sace-kb-pgdata)"
"${COMPOSE[@]}" down -v

echo "==> starting a fresh sace-kb-db"
"${COMPOSE[@]}" up -d

echo "==> waiting for it to become healthy"
for _ in $(seq 1 30); do
  status="$("${COMPOSE[@]}" ps --format '{{.Status}}' db 2>/dev/null || true)"
  if echo "$status" | grep -qi healthy; then
    break
  fi
  sleep 1
done
if ! echo "$status" | grep -qi healthy; then
  echo "sace-kb-db did not become healthy in 30s — ${COMPOSE[*]} logs db" >&2
  exit 1
fi

echo "==> initializing schema"
python3 - "$ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from dotenv import load_dotenv
load_dotenv(f"{sys.argv[1]}/.env.kb", override=True)
from sace_chat.db import init_db, DATABASE_URL
init_db()
print(f"schema initialized on {DATABASE_URL}")
PY

echo "==> done — sace_kb is a clean, schema-initialized database"
