"""Generate sace_chat/kb_renewal.py from the renewal campaign's source files.

sace_chat/kb_renewal.py is a GENERATED, committed artifact — never hand-edit
it. Re-run this script after editing the CSV, or after changing FLOW_RULES or
RENEWAL_STABLE_CORE below, and commit the regenerated file.

Two kinds of content go into the pool, from two different sources and by two
different methods:

  A) CSV-driven Q&A rules (T1-T4). Mechanically parsed from
     data/renewal/Maya_knowledge_base_1_10_Knowledge_base_.csv — every field
     maps straight across (see _csv_rows / _rule_from_csv_row below). No
     judgment calls: the mapping is the same for all 126 rows.

  B) Flow rules. These CANNOT be parsed mechanically: Script_simplified.pdf
     is a worked example transcript (patient "Maria Reyes", a specific
     address, specific dollar figures), not a templated script the way
     coverage's kb.py is, and Coverage_Renewal_Call_Flow_1.html's mermaid
     graph describes step ORDER and BRANCHING as abstract boxes with no
     dialogue at all. FLOW_RULES below is the one place a human read both
     and placed the result: verbatim campaign-invariant lines from the PDF,
     with the illustrative caller-identity/campaign-constant values
     (patient name, clinic name, callback number) normalised to the same
     {placeholder} tokens coverage's kb.py already uses — a caller-specific
     value like an address or a dollar figure spoken back to THIS caller is
     never templated as a placeholder (there is no fixed value to substitute);
     it is described as an instruction to read back whatever the caller just
     gave, exactly the same pattern kb.py's own confirm_phone_number rule
     already uses. `requires`/`sets`/`step_order` encode the mermaid graph's
     branch points as prerequisites — see retrieve.py's _fetch_general.

Known gaps, flagged rather than filled (see scripts/load_renewal_sources.py
for the mechanical ones found in the CSV/PDF themselves):

  - no_answer_retry and wrong_person have no verbatim Maya line anywhere in
    Script_simplified.pdf — it is a "happy path" script and never shows
    either branch. The mermaid graph names the BEHAVIOUR (retry cadence;
    flag if incorrect) but gives no spoken line. Rather than invent one,
    these two reuse coverage's own already-shipped generic lines for the
    identical situation (kb.py's patient_unavailable/retry_line and
    wrong_person_close) — flagged here, not silently passed off as
    renewal-sourced content. The script owner should replace these with a
    renewal-specific line if one exists.
  - already_submitted_close is in the same position: mermaid's D4 "Yes"
    branch ("Call ends - nothing further needed") has no line in the PDF
    either, since the transcript's patient never says they already
    submitted. Reuses kb.py's counselor_ack_close-style plain close pattern
    for the same reason.
  - Cross-call state (Day+2's reminder_upload_check needs to know a PRIOR
    call already reached consent_prebrief) is not modelled — CallState is
    one call's memory, and nothing here persists flow position across calls.
    Given a permissive `requires={}` so it is reachable by semantic
    similarity alone on whatever call it is relevant to; a real
    implementation needs a case-record-backed "which step did we reach last
    time" field, which is out of scope here.
  - already_submitted_check's `requires` does not force address_capture to
    have completed first when packet_received=False (only that packet_check
    itself ran) — expressing "B OR C must be true" is outside the flat
    exact-match/any-set `requires` grammar this phase built. In practice the
    caller's own utterance about the address is what routes to
    address_capture by similarity; this is a known looseness, not a silent
    gap.

Run:  python scripts/build_kb_renewal.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CSV_PATH = REPO_ROOT / "data" / "renewal" / "Maya_knowledge_base_1_10_Knowledge_base_.csv"
OUT_PATH = REPO_ROOT / "sace_chat" / "kb_renewal.py"
CSV_ENCODING = "cp1252"  # confirmed by inspection — see load_renewal_sources.py

REQUIRES_ANY_SET = "__any__"
REQUIRES_NOT_SET = "__not_set__"

# ─────────────────────────── A) CSV -> T1-T4 rules ───────────────────────────

# Tier -> (priority, transfer). Verified 1:1 against the CSV's own "Answer
# source" column (T1<->Static, T2<->Case record, T3/T4<->Transfer) — tier
# alone determines this for every one of the 126 rows, no exceptions.
_TIER_BEHAVIOUR = {
    "T1": {"priority": "normal", "transfer": False},
    "T2": {"priority": "normal", "transfer": False},
    "T3": {"priority": "normal", "transfer": True},
    "T4": {"priority": "critical", "transfer": True},
}

# The three T2 rows' bracketed instructions, named precisely (only 3 rows —
# hardcoded per-row rather than parsed generically, since turning free English
# into a field-name slug is a judgment call, not a mechanical transform).
_T2_CASE_FIELDS = {
    "KB-LTR-08": ["worker_name", "county_phone"],
    "KB-DUE-01": ["due_date"],
    "KB-DUE-06": ["outstanding_county_request"],
}


def _snake(topic: str) -> str:
    return topic.strip().lower().replace(" ", "_").replace("-", "_")


def _csv_rows() -> list[dict]:
    """Every Q&A row (ID starting 'KB-'), keyed by the CSV's own header.
    Section-banner rows (title text as the ID, e.g. "THE LETTER AND WHAT IT
    IS...") are skipped — they carry no Tier and no answer of their own."""
    with open(CSV_PATH, newline="", encoding=CSV_ENCODING) as f:
        rows = list(csv.reader(f))
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "ID")
    header = rows[header_idx]
    data_rows = [r for r in rows[header_idx + 1:] if r and r[0].strip()]
    return [
        dict(zip(header, r))
        for r in data_rows
        if r[0].startswith("KB-")
    ]


def _rule_from_csv_row(row: dict) -> dict:
    """One Chunk-shaped dict from one CSV row. Pure mapping — see the module
    docstring's table; no content is invented here."""
    rule_id = row["ID"].lower().replace("-", "_")
    tier = row["Tier"].strip()
    behaviour = _TIER_BEHAVIOUR[tier]
    topic_intent = _snake(row["Topic"])
    variants = [v.strip() for v in row["How patients actually say it"].split("·") if v.strip()]

    return {
        "id": rule_id,
        "title": row["Canonical question"].strip(),
        "text": row["What Maya says"].strip(),
        "cue": " · ".join(variants),
        "intent": topic_intent,
        "priority": behaviour["priority"],
        "terminal": False,
        "exclusive": False,
        "tier": tier,
        "transfer": behaviour["transfer"],
        "tags": {
            "citation": row.get("Citation", "").strip(),
            "design_note": row.get("Design note", "").strip(),
            "caller_phrasings": variants,
            **({"case_fields": _T2_CASE_FIELDS[row["ID"]]} if row["ID"] in _T2_CASE_FIELDS else {}),
        },
    }


# ───────────────────────── safety labels, retained verbatim ─────────────────
# Reused unchanged from sace_chat.kb — these are cross-campaign safety
# invariants (do-not-call, abuse, a medical emergency, garbled audio,
# frustration), not renewal-script content, and neither the CSV nor the PDF
# defines any of them. Importing the exact rule objects (not re-authoring
# equivalents) is what keeps this an extraction rather than new content.
_SAFETY_RULE_IDS = (
    "special_dnc", "special_abuse", "medical_emergency",
    "special_garbled_audio", "special_frustration",
)


def _safety_rules_and_exemplars():
    from sace_chat.kb import INTENT_EXEMPLARS as COVERAGE_EXEMPLARS
    from sace_chat.kb import RULES as COVERAGE_RULES

    by_id = {r.id: r for r in COVERAGE_RULES}
    rules = []
    exemplars = {}
    for rid in _SAFETY_RULE_IDS:
        chunk = by_id[rid]
        # medical_emergency is a general-pool (intent=None) rule in coverage;
        # the user's brief asked to "retain the safety labels" as ROUTABLE
        # intents here, so it is given its own intent label rather than
        # kept general — the other four already carry a real intent.
        intent = chunk.intent or "medical_emergency"
        rules.append({
            "id": chunk.id, "title": chunk.title, "text": chunk.text, "cue": chunk.cue,
            "intent": intent, "priority": chunk.priority, "terminal": chunk.terminal,
            "exclusive": chunk.exclusive, "tier": None, "transfer": False,
            "tags": {"source": "reused verbatim from sace_chat.kb (safety invariant)"},
        })
        exemplars[intent] = list(COVERAGE_EXEMPLARS.get(chunk.intent or "medical_emergency", []))
    # medical_emergency has no INTENT_EXEMPLARS entry in coverage (it's a
    # general-pool rule there, routed by plain similarity, not by intent) —
    # give it a short exemplar set of its own so IntentRouter can route it.
    exemplars.setdefault("medical_emergency", [
        "I think I'm having a heart attack",
        "I can't breathe",
        "someone has collapsed, I need an ambulance",
        "she's bleeding badly",
    ])
    return rules, exemplars


# ───────────────────────────── B) flow rules ─────────────────────────────
# See the module docstring for how these were extracted and why they cannot
# be generated mechanically. `text` is verbatim Maya dialogue from
# Script_simplified.pdf with {patient_first_name}/{patient_last_name}/
# {business_entity}/{callback_number} substituted for the transcript's
# illustrative "Maria Reyes"/"Santa Rosa Community Health"/"(707) 555-0142" —
# every other word is unchanged. Where the PDF has Maya read a caller-specific
# value back (the new address), the text is a plain instruction, matching
# kb.py's own confirm_phone_number convention, not invented dialogue.
FLOW_RULES = [
    {
        "id": "open_identify",
        "title": "Open the call — confirm this is the patient",
        "text": 'AT THE VERY START of the call, before anything else has been said, say: '
                '"Hello, is this {patient_first_name} {patient_last_name}?" Then wait.',
        "cue": "hello?, hi, yeah?, who's calling, hello anyone there -- the call has just "
               "connected and nobody has said anything yet.",
        "requires": {"name_confirmed": REQUIRES_NOT_SET}, "sets": {}, "step_order": 1,
    },
    {
        "id": "disclose_ai_and_recording",
        "title": "Disclose the AI assistant, the recording, and the reason for the call",
        "text": 'AFTER asking whether this is {patient_first_name} {patient_last_name}, WHEN '
                'they confirm — "yes", "speaking", "this is her", "yeah that\'s me" — add '
                'name_confirmed to extracted_fields, set to true, then say in full: "Hi '
                "{patient_first_name}. My name is Maya and I'm an automated assistant calling "
                "for {business_entity}, where you get your care. This call is recorded. I'm "
                "calling about your Medi-Cal renewal, and I can put you through to a person "
                'any time you\'d like. Is now an okay time for a few minutes?" Then wait.',
        "cue": "yes speaking, this is her, yeah that's me, who's this, speaking -- the caller "
               "has confirmed they are the patient.",
        "requires": {"name_confirmed": REQUIRES_NOT_SET}, "sets": {"name_confirmed": True},
        "step_order": 2,
    },
    {
        "id": "verify_dob",
        "title": "Confirm identity with date of birth",
        "text": 'AFTER asking whether now is an okay time, WHEN the caller agrees — "yeah", '
                '"sure", "go ahead", "I got a text about this yesterday" — say: "Just so I\'m '
                "speaking with the right person, can you tell me your date of birth?\" Then "
                "wait.",
        "cue": "yeah I've got a minute, sure go ahead, okay that's fine, now works for me "
               "-- the caller has agreed that now is a good time to talk.",
        "requires": {"name_confirmed": True, "disclosed": REQUIRES_NOT_SET},
        "sets": {"disclosed": True}, "step_order": 3,
    },
    {
        "id": "packet_check",
        "title": "Ask whether the renewal packet arrived",
        "text": 'AFTER asking for the date of birth, WHEN the caller gives one, read it back '
                'to confirm, add identity_verified to extracted_fields set to true, then say: '
                '"{business_entity} sent you a Medi-Cal renewal packet, to your address on '
                "file. A lot of these get lost in the post or go to an old address, so I "
                'want to check — did that reach you?" Naming the post or an old address '
                "BEFORE asking is deliberate — it gives the caller permission to say no "
                "without it sounding like their fault.",
        "cue": "March 8th 1983, that's my birthday, here's my date of birth -- the caller has "
               "given their date of birth.",
        "requires": {"disclosed": True, "identity_verified": REQUIRES_NOT_SET},
        "sets": {"identity_verified": True}, "step_order": 4,
    },
    {
        "id": "address_capture",
        "title": "Packet never arrived — capture and confirm the new address, then continue",
        "text": 'AFTER asking whether the renewal packet arrived, WHEN it did NOT — "no", '
                '"we moved", "that\'s an old address" — add packet_received to '
                'extracted_fields set to false, then say: "That happens constantly, and '
                "it's the most common reason people lose their Medi-Cal — so I'm glad we "
                'caught it. What\'s the address now?" Once they give an address, read it '
                'back in full and confirm it — "Let me read that back — [the address they '
                'gave]. Is that right?" Once they confirm, add new_address to '
                'extracted_fields, say plainly: "Thank you. I\'ve noted that, and a person '
                "will need to update it with the county — I can't do that part myself. One "
                'more quick thing before I let you go —" and CONTINUE. Never end the call '
                "here.",
        "cue": "no we moved, that's an old address, I don't live there anymore, in May I "
               "think, here's the new address, apartment, is that right yes -- the caller "
               "says the renewal packet went to the wrong or an old address, or is giving or "
               "confirming a new one.",
        "requires": {
            "identity_verified": True, "packet_received": False,
            "already_submitted": REQUIRES_NOT_SET,
        },
        "sets": {"packet_received": False, "address_updated": True}, "step_order": 5,
    },
    {
        "id": "already_submitted_check",
        "title": "Ask whether the renewal has already been sent in",
        "text": 'AFTER asking whether the packet arrived (and, if it did not, after the '
                'address has been corrected and confirmed), say: "Have you already sent the '
                'renewal back to the county?" This is the natural next question whether the '
                "packet arrived or the address just needed fixing — either way nothing else "
                "is asked before it.",
        "cue": "yes it arrived, yes I got it, okay, alright, one more quick thing sure, go "
               "ahead ask away -- the packet-arrival question has just been answered yes, or "
               "the address correction has just been wrapped up.",
        "requires": {"identity_verified": True, "already_submitted": REQUIRES_NOT_SET},
        "sets": {"packet_received": True}, "step_order": 6,
    },
    {
        "id": "already_submitted_close",
        "title": "Already submitted — nothing further needed",
        # No verbatim line exists in Script_simplified.pdf for this branch —
        # see the module docstring's "Known gaps". Reuses kb.py's plain,
        # already-shipped close pattern for the identical situation
        # (counselor_ack_close) rather than inventing renewal-specific words.
        "text": 'WHEN the patient confirms they already sent the renewal in, say: "Perfect, '
                'thank you for confirming — there\'s nothing further needed from you right '
                'now. Take care!" Then stop.',
        "cue": "yes I already sent it, already submitted, mailed it back weeks ago -- the "
               "patient has confirmed the renewal is already in.",
        "requires": {"identity_verified": True, "already_submitted": REQUIRES_NOT_SET},
        "sets": {"already_submitted": True}, "step_order": 7,
        "terminal": True, "exclusive": True,
    },
    {
        "id": "willingness_ask",
        "title": "Ask whether they want help, or will file it themselves",
        "text": 'AFTER asking whether the renewal has already been sent in, WHEN it has NOT — '
                '"no", "not yet", "I never got it" — say: "Understood. There are counsellors '
                "at {business_entity} who do these renewals every day, in Spanish, free of "
                "charge. They can go through the form with you and send it in for you, or "
                "if you'd rather do it yourself I can just answer questions. Which would you "
                'prefer?" Then wait.',
        "cue": "no I never got it, not yet, haven't sent it in, no not submitted -- the "
               "patient has said the renewal has not been submitted yet.",
        "requires": {"identity_verified": True, "already_submitted": REQUIRES_NOT_SET},
        "sets": {"already_submitted": False, "willingness_asked": True}, "step_order": 8,
    },
    {
        "id": "m1_availability",
        "title": "Wants help — check availability for about twenty minutes",
        "text": 'AFTER asking whether they want help or will self-file, WHEN they choose help '
                '— "they can do it", "that\'s easier", "yes please help" — add willingness to '
                'extracted_fields set to "help", then say: "Before I pass you over, three '
                "quick things so we don't waste your time. First — do you have about twenty "
                'minutes now, or would later be better?" Then wait.',
        "cue": "they can do it, I don't have to fill it out, that's better, okay let's do "
               "that, yes please help -- the patient has chosen to have a counsellor help "
               "rather than self-file.",
        "requires": {"willingness_asked": True, "willingness": REQUIRES_NOT_SET},
        "sets": {"willingness": "help"}, "step_order": 9,
    },
    {
        "id": "m2_camera_phone",
        "title": "Second check — a phone that takes photos",
        "text": 'AFTER asking about availability, WHEN they answer, add available_now to '
                'extracted_fields, then say: "Second — are you on a mobile right now that '
                "takes photos? At some point we'll need a picture of a document, and I want "
                'to know which way is easiest for you." Then wait.',
        "cue": "now is okay, I'm home, yes I've got time, later would be better -- the "
               "patient has answered whether now is a good time.",
        "requires": {"willingness": "help", "available_now": REQUIRES_NOT_SET},
        "sets": {"available_now": True}, "step_order": 10,
    },
    {
        "id": "m3_helper_at_home",
        "title": "Third check — anyone at home who helps with the phone or paperwork",
        "text": 'AFTER asking about a camera phone, WHEN they answer, add has_camera_phone to '
                'extracted_fields, then say: "And third — is there anyone at home who '
                'usually helps you with your phone or with paperwork?" Then wait.',
        "cue": "yeah I'm on my phone, I have a smartphone, no I don't have a phone with a "
               "camera -- the patient has answered whether they have a camera phone.",
        "requires": {
            "willingness": "help", "available_now": REQUIRES_ANY_SET,
            "has_camera_phone": REQUIRES_NOT_SET,
        },
        "sets": {"has_camera_phone": True}, "step_order": 11,
    },
    {
        "id": "consent_prebrief",
        "title": "Pre-brief the counsellor's consent step before transferring",
        "text": 'AFTER asking who at home helps with the phone or paperwork, WHEN they '
                'answer, add helper_at_home to extracted_fields, then say: "One more thing '
                "so it's not a surprise. For a counsellor to send the form in on your "
                "behalf, you'll need to give permission on the recording — they'll read out "
                "exactly what you're agreeing to, and it takes a couple of minutes. It's "
                "completely your choice, and you can change your mind at any time. Does "
                'that sound okay?" Maya never takes this consent herself — it is always the '
                "counsellor's own step, read aloud on the call, after the transfer. Add "
                "consent_prebriefed to extracted_fields, set to true, since the pre-brief "
                "has now been given regardless of how they answer.",
        "cue": "my daughter helps sometimes, my husband does that, nobody really, she's at "
               "school -- the patient has answered who at home helps with their phone or "
               "paperwork.",
        "requires": {
            "willingness": "help",
            "has_camera_phone": REQUIRES_ANY_SET,
            "helper_at_home": REQUIRES_NOT_SET,
        },
        "sets": {"helper_at_home": True, "consent_prebriefed": True}, "step_order": 12,
    },
    {
        "id": "warm_transfer",
        "title": "Transfer to the human counsellor",
        "text": 'AFTER the consent pre-brief, WHEN they agree — "okay", "like a signature", '
                '"sounds okay" — say: "Let me put you through to a counsellor now. They\'ll '
                "have everything we've just talked about, so you won't need to start again.\" "
                "Then hand off, carrying forward everything captured this call: name, date "
                "of birth, address, willingness, and the three pre-transfer answers.",
        "cue": "like a signature, exactly like that, okay that's fine, sounds okay, yes "
               "understood -- the patient has agreed to the consent pre-brief.",
        "requires": {"consent_prebriefed": True},
        "sets": {"transferred": True}, "step_order": 13,
        "terminal": True, "exclusive": True, "transfer": True,
    },
    {
        "id": "selffile_expected_date",
        "title": "Self-filing — ask when they plan to send it",
        "text": 'AFTER asking whether they want help or will self-file, WHEN they choose to '
                'file it themselves — "no I\'ll do it", "I did it last year, it\'s fine" — '
                'add willingness to extracted_fields set to "self_file", then say: "That\'s '
                'completely fine. When are you planning to send it?" Then wait.',
        "cue": "no I'll do it, I did it last year it's fine, I'll file it myself -- the "
               "patient has chosen to file the renewal themselves rather than have a "
               "counsellor help.",
        "requires": {"willingness_asked": True, "willingness": REQUIRES_NOT_SET},
        "sets": {"willingness": "self_file"}, "step_order": 14,
    },
    {
        "id": "selffile_qa_open",
        "title": "Self-filing — open the floor for questions",
        "text": 'AFTER asking when they plan to send it, WHEN they answer, add '
                'expected_submission_date to extracted_fields, then say: "I\'ll set a '
                'reminder for around then. Do you have any questions about it while I\'m '
                'here?" If they ask something concrete, answer it from the matching topic '
                "rule rather than here — this rule only opens the floor. Add qa_opened to "
                "extracted_fields, set to true, since the floor is now open regardless of "
                "how they answer.",
        "cue": "this weekend, next week sometime, by Friday, in a few days -- the patient "
               "has given an expected date for sending the renewal in themselves.",
        "requires": {"willingness": "self_file", "expected_submission_date": REQUIRES_NOT_SET},
        "sets": {"expected_submission_date": True, "qa_opened": True}, "step_order": 15,
    },
    {
        "id": "selffile_teachback",
        "title": "Self-filing — teach-back before closing",
        "text": 'AFTER opening the floor for questions, WHEN they have no more — "no I think '
                'that\'s it", "no other questions" — say: "Before you go — so I know I '
                "explained it right, which number are you putting down for your pay?\" "
                "Confirm their answer matches what was discussed.",
        "cue": "no I think that's it, no other questions, that's everything, nothing else "
               "-- the patient has no more questions after the open floor.",
        "requires": {
            "willingness": "self_file", "qa_opened": True,
            "teachback_done": REQUIRES_NOT_SET,
        },
        "sets": {}, "step_order": 16,
    },
    {
        "id": "selffile_close",
        "title": "Self-filing — close",
        "text": 'AFTER the teach-back question, WHEN they answer correctly, add '
                'teachback_done to extracted_fields set to true, then say: "That\'s the one. '
                'Take care, {patient_first_name}." Then stop.',
        "cue": "the big one, before taxes, the gross amount, yes that's right -- the patient "
               "has answered the teach-back question.",
        "requires": {
            "willingness": "self_file", "qa_opened": True,
            "teachback_done": REQUIRES_NOT_SET,
        },
        "sets": {"teachback_done": True},
        "step_order": 17, "terminal": True, "exclusive": True,
    },
    {
        "id": "reminder_upload_check",
        "title": "Follow-up call — the document upload link has not been used yet",
        "text": 'Say: "Hi {patient_first_name}, it\'s Maya from {business_entity} again. We '
                "sent you a link to send a photo of your documents and it hasn't come through "
                "yet. Nothing's gone wrong — I just want to check whether the link worked for "
                'you, or whether there\'s an easier way for you to get it to us." Frame this '
                "as the LINK failing, never as the patient failing — that framing is why a "
                "forgetful patient answers honestly instead of getting defensive. Offer the "
                "same link again, or replying to the text with a photo directly.",
        "cue": "the link about my documents never came through, I didn't get any upload "
               "link, my link isn't working -- a prior call ended with a document-upload "
               "link sent and it has not been used yet.",
        # No durable cross-call state exists yet to gate this on "did a prior
        # call reach this point" — a real implementation needs one, since
        # this rule is meant to open a FRESH call where nothing else in this
        # table's fields has been set yet, which rules out gating it on any
        # of this campaign's own flow flags without making it unreachable.
        # The cue is deliberately specific to the upload link itself (not
        # generic apology phrasing) so it does not compete with unrelated
        # short affirmatives elsewhere in a call — see the module docstring's
        # known gaps.
        "requires": {}, "sets": {"upload_reminder_sent": True}, "step_order": 18,
    },
    {
        "id": "no_answer_retry",
        "title": "Nobody answered, or the call cannot continue right now",
        # No verbatim line in the PDF for this branch — see "Known gaps".
        # Reuses kb.py's retry_line verbatim (a generic, already-shipped
        # closing for the identical situation), flagged rather than invented.
        "text": 'WHEN the call cannot usefully continue — nobody is available, or an earlier '
                'attempt already failed — end it without pressing: "No problem — we\'ll try '
                'again soon. Take care!" Then stop.',
        "cue": "can't talk right now, nobody's home, this isn't a good time, call back later "
               "-- the call cannot usefully continue right now.",
        "requires": {"name_confirmed": REQUIRES_NOT_SET}, "sets": {}, "step_order": 19,
        "terminal": True, "exclusive": True,
    },
    {
        "id": "wrong_person",
        "title": "Wrong person or wrong number entirely",
        # No verbatim line in the PDF for this branch either — reuses kb.py's
        # wrong_person_close verbatim; see "Known gaps".
        "text": 'WHEN it turns out this is the wrong person entirely: "So sorry — thank you, '
                'take care!" Then stop.',
        "cue": "wrong number, nobody by that name here, you've got the wrong person -- the "
               "call has reached someone who is not the patient at all.",
        "requires": {"name_confirmed": REQUIRES_NOT_SET}, "sets": {}, "step_order": 20,
        "terminal": True, "exclusive": True,
    },
]


def _render(rules: list[dict], stable_core: str, intent_exemplars: dict, valid_intents: list) -> str:
    lines = [
        '"""GENERATED FILE — do not hand-edit.',
        "",
        "Produced by scripts/build_kb_renewal.py from:",
        "  data/renewal/Maya_knowledge_base_1_10_Knowledge_base_.csv  (T1-T4 Q&A rules)",
        "  data/renewal/Script_simplified.pdf                         (flow rule dialogue)",
        "  data/renewal/Coverage_Renewal_Call_Flow_1.html             (flow order/branching)",
        "",
        "Re-run the build script and commit the result — never edit this file directly.",
        '"""',
        "",
        "from sace_chat.models import Chunk",
        "",
        f"RENEWAL_STABLE_CORE = {stable_core!r}",
        "",
        f"RENEWAL_INTENT_EXEMPLARS = {intent_exemplars!r}",
        "",
        f"RENEWAL_VALID_INTENTS = frozenset({valid_intents!r})",
        "",
        "RULES = [",
    ]
    for r in rules:
        lines.append("    Chunk(")
        for key in ("id", "title", "text", "cue", "intent", "priority", "terminal",
                    "exclusive", "tier", "transfer", "requires", "sets", "step_order", "tags"):
            if key not in r:
                continue
            lines.append(f"        {key}={r[key]!r},")
        lines.append("    ),")
    lines.append("]")
    lines.append("")
    lines.append("CHUNKS = RULES")
    lines.append("")
    return "\n".join(lines)


RENEWAL_STABLE_CORE = """\
# ROLE
You are Maya, an automated assistant calling for {business_entity}, where {patient_first_name} \
gets their care. You are calling about their Medi-Cal renewal packet. This call is recorded.

# THE GOAL OF THIS CALL
Confirm identity, check whether the renewal packet reached them and correct the address if not, \
find out whether they already submitted it, then either hand them to a human counsellor (free, \
available in Spanish, any time) or help them file it themselves with plain factual answers. You \
never take consent on the counsellor's behalf — that is always their own step, read aloud after \
the transfer.

# PERSONALITY
Warm, plain, unhurried. One question per turn. Never claim to have fixed anything yourself that \
only a person can — say so plainly and keep going, never end the call over it.

# HOW YOU DECIDE WHAT TO SAY
The GOVERNING RULE below is the only thing that determines this turn's reply. If it gives a \
scripted line in quotes, that is your reply, matched closely or lightly rephrased for a natural \
read. You do not add content from REFERENCE or from your own sense of how such calls usually go.

# NEVER SAY — read this before every reply
Never state, in your own words, whether {patient_first_name} qualifies for Medi-Cal, any income \
limit, any dollar amount, or any deadline — UNLESS that exact value was handed to you verbatim in \
this call's own case record. If you do not have the real number, say so and offer to find out or \
transfer, never estimate or guess one. This is enforced in code as well as here: a reply that \
violates it is rewritten or replaced before it ever reaches the caller.

# SAFETY
No medical advice. If the caller describes a medical emergency, say exactly: "If this is a \
medical emergency, please hang up and call 911." Never collect a Social Security Number or ID \
number.

# LANGUAGE
Renewal help is available in Spanish at no charge through the human counsellors — offer this \
plainly if asked, and never attempt to continue the call in a language other than English \
yourself.
"""


def _disambiguate_duplicate_ids(rules: list[dict]) -> None:
    """KB-LTR-04 appears twice in the CSV with genuinely different content
    (never received the form vs. received-then-lost it) — see
    scripts/load_renewal_sources.py's KNOWN GAP note. Silently letting two
    rules share one id would collide on insert and drop one of them without
    anyone noticing; instead every id seen more than once gets a stable
    _a/_b/... suffix (in CSV row order) and a loud warning, so both survive
    and the collision stays visible until the script owner picks one."""
    seen: dict[str, int] = {}
    for r in rules:
        seen[r["id"]] = seen.get(r["id"], 0) + 1
    dupes = {rid for rid, n in seen.items() if n > 1}
    if not dupes:
        return
    counters: dict[str, int] = {}
    for r in rules:
        if r["id"] not in dupes:
            continue
        n = counters.get(r["id"], 0)
        counters[r["id"]] = n + 1
        suffix = chr(ord("a") + n)
        original = r["id"]
        r["id"] = f"{original}_{suffix}"
        print(f"  WARNING: duplicate CSV id {original!r} -> disambiguated as {r['id']!r} "
              f"({r['title']!r}) — pick which row is authoritative before Phase 5")


def main() -> int:
    csv_rules = [_rule_from_csv_row(r) for r in _csv_rows()]
    _disambiguate_duplicate_ids(csv_rules)
    safety_rules, safety_exemplars = _safety_rules_and_exemplars()

    all_rules = FLOW_RULES + csv_rules + safety_rules

    intent_exemplars = dict(safety_exemplars)
    for r in csv_rules:
        intent_exemplars.setdefault(r["intent"], [])
        # Capped at 12 per label for phrasing diversity without one enormous
        # topic (Income, 14 rows) drowning out the others in embedding cost.
        remaining = 12 - len(intent_exemplars[r["intent"]])
        if remaining > 0:
            intent_exemplars[r["intent"]].extend(r["tags"]["caller_phrasings"][:remaining])

    valid_intents = sorted(intent_exemplars) + ["none"]

    tiers = {}
    for r in csv_rules:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1

    OUT_PATH.write_text(_render(all_rules, RENEWAL_STABLE_CORE, intent_exemplars, valid_intents))

    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  {len(csv_rules)} CSV-derived rules, tiers={tiers}")
    print(f"  {len(FLOW_RULES)} flow rules")
    print(f"  {len(safety_rules)} safety rules reused from sace_chat.kb")
    print(f"  {len(all_rules)} total rules, {len(intent_exemplars)} intent labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
