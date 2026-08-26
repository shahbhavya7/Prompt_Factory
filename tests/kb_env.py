"""Shared env pin for every test on this branch.

Every test here runs against the isolated renewal-KB database
(docker-compose.kb.yml, sace-kb-db on port 5433) — never the shared sace_chat
database other branches use. Call pin() before importing anything from
sace_chat, and before the file's own load_dotenv() call: pin() sets
DATABASE_URL / SACE_EXPECTED_DB from .env.kb with override=True, and
load_dotenv()'s default behaviour is to never clobber an already-set var, so
those two values survive whatever the file's own load_dotenv() pulls from the
shared .env afterward.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent


def pin() -> None:
    load_dotenv(_ROOT / ".env.kb", override=True)
