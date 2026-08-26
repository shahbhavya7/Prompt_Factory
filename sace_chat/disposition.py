"""Code-enforced consent gate and verification-disposition state machine for
the renewal campaign.

Nothing in Maya's own turn loop (engine.py, retrieve.py, the flow rules'
`sets`) ever calls anything in this module. `consent_recorded` and
disposition transitions are HUMAN-AGENT actions — recorded by the calling
code after a human counsellor actually did the thing (read the consent
script and got agreement, filed the form, confirmed retention at a
verification checkpoint), never inferred from anything the model extracts.
This is the same "enforce in code, not in a prompt" pattern check_never_say
already uses for a different invariant.
"""

from __future__ import annotations

OPEN = "OPEN"
SELF_FILING = "SELF_FILING"
SUBMITTED = "SUBMITTED"
VERIFIED_RETAINED_D90 = "VERIFIED_RETAINED_D90"
CLOSED_SUCCESS = "CLOSED_SUCCESS"

# Checkpoints SELF_FILING/SUBMITTED schedule. Self-report is not evidence a
# renewal actually went through — every one of these fires without closing
# the case; only a day-90 checkpoint that confirms retention does.
VERIFICATION_SCHEDULE_DAYS = (14, 30, 60, 90)


class ConsentRequiredError(PermissionError):
    """Raised by every side-effecting renewal action below when
    state.consent_recorded is not True. The check runs BEFORE any side
    effect — nothing below it executes."""


def record_human_consent(state) -> None:
    """The ONLY place consent_recorded may become True. Call this from the
    human counsellor's own code path, after they have read the consent
    script aloud and the caller agreed — never from Maya's turn loop."""
    state.consent_recorded = True


def send_upload_link(state) -> None:
    """Side-effecting: sends the document-upload link. Refuses outright,
    before anything else runs, if consent has not been recorded."""
    if not state.consent_recorded:
        raise ConsentRequiredError(
            "cannot send the upload link: consent_recorded is False"
        )
    state.upload_link_sent = True


def mark_filed(state, disposition: str) -> None:
    """Side-effecting: marks the renewal as filed via the human-agent path
    (either they filed it on the caller's behalf, or confirmed the caller's
    self-filed submission). Refuses outright if consent has not been
    recorded."""
    if not state.consent_recorded:
        raise ConsentRequiredError(
            "cannot mark filed: consent_recorded is False"
        )
    if disposition not in (SELF_FILING, SUBMITTED):
        raise ValueError(f"mark_filed: unexpected disposition {disposition!r}")
    state.disposition = disposition


def record_address_update(state) -> None:
    """The county address record was actually updated by a human — Maya's
    own address_capture rule can record what the caller SAID the new
    address is (state.collected_fields), but never that the county's own
    record was updated; only this call may set that."""
    state.address_updated_by_human = True


def start_self_filing(state, expected_file_date: str | None = None) -> None:
    """Path C: the caller is filing it themselves. No consent gate — self-
    filing does not involve anyone acting on the caller's behalf. Schedules
    verification; never closes the case by itself."""
    state.disposition = SELF_FILING
    state.expected_file_date = expected_file_date


def record_verification(state, day: int, retained: bool) -> str:
    """A scheduled checkpoint (D+14/30/60/90) fired. Returns the outcome
    name. Only a day-90 checkpoint that confirms retention may close the
    case — every earlier checkpoint, and a day-90 checkpoint that does NOT
    confirm retention, leaves `disposition` exactly as the caller's own
    self-report already set it (SELF_FILING/SUBMITTED): a self-report is
    not proof of anything on its own.
    """
    if day not in VERIFICATION_SCHEDULE_DAYS:
        raise ValueError(f"record_verification: {day} is not a scheduled checkpoint")
    if day == 90 and retained:
        state.disposition = CLOSED_SUCCESS
        return VERIFIED_RETAINED_D90
    return state.disposition
