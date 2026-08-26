"""Every test on this branch runs against the isolated renewal-KB database
(docker-compose.kb.yml, sace-kb-db on port 5433) — never the shared sace_chat
database other branches use.

Loaded by pytest before any test module is imported, so DATABASE_URL and
SACE_EXPECTED_DB are already pinned by the time a test module's own
load_dotenv() call runs (that call does not override already-set vars) and by
the time sace_chat.db builds its engine and enforces the guard.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env.kb", override=True)
