"""The structured hand-off packet warm_transfer emits.

"Maya's filled me in" is the premise: the human counsellor picking up must
never make the caller repeat anything already established this call. Built
purely from CallState — no I/O, no side effect — so it is cheap to build and
easy to test in isolation from the websocket/persistence layer that carries
it (see voice_agent.py's finish_turn, which attaches this as extra fields on
the existing `turn` event rather than a new event type).
"""

from __future__ import annotations


def build_transfer_packet(state) -> dict:
    fields = getattr(state, "collected_fields", None) or {}
    return {
        "identity": {
            "verified": bool(fields.get("identity_verified")),
            "name": fields.get("name"),
            "date_of_birth": fields.get("date_of_birth"),
        },
        "address": {
            "new_address": fields.get("new_address"),
            "needs_county_update": bool(fields.get("new_address"))
                                    and not state.address_updated_by_human,
        },
        "packet_received": state.packet_received,
        "already_submitted": state.already_submitted,
        "willingness": state.willingness,
        "pre_transfer_answers": {
            "available_now": state.available_now,
            "has_camera_phone": state.has_camera_phone,
            "helper_at_home": state.helper_at_home,
        },
        "consent_prebriefed": state.consent_prebriefed,
        # Every diversion answered this call, with its rule id — so the
        # counsellor (or a future audit) can see exactly what was already
        # covered without re-asking or re-answering it.
        "kb_answers_given": list(getattr(state, "kb_answers_given", None) or []),
    }
