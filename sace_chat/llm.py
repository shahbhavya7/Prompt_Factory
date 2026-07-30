"""LLM access: one structured decision per turn.

The turn contract is a single JSON object covering the intent and Maya's reply
together. Requested with response_format=json_object where the provider supports
it, which removes the "model forgot to emit the control JSON" failure mode that
dogged the earlier free-text contract.
"""

import json
import os
import re

TURN_SCHEMA_KEYS = ("intent", "reply", "call_should_end", "extracted_fields")


def build_messages(system: str, user: str) -> list[dict]:
    """The exact message list a turn decision is sent as.

    Single source of truth on purpose: the engine captures this to show what was
    sent, and OpenAICompatibleLLM.chat_json sends this same function's output. If
    the shape ever changes, both change together, so the transparency viewer
    cannot drift into showing a reconstruction of a payload that differs from the
    real one.
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_messages(messages: list[dict]) -> str:
    """The messages as one verbatim string, for display.

    Every character of every message content appears unmodified; the only added
    text is the `=== ROLE ===` separators, which exist because the API takes a
    list of messages and a text box can only show one string.
    """
    return "\n\n".join(
        f"=== {m['role'].upper()} ===\n{m['content']}" for m in messages
    )

# The mock routes off whichever rule the prompt says is governing — the same
# thing the real model is told to follow.
_GOVERNING_RE = re.compile(r"GOVERNING RULE.*?\[([a-z0-9_]+)\]", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?|```")

# Canned replies, keyed by governing rule id. Crude by design — this exists so
# the pipeline and UI are exercisable with no API key, not to model behaviour.
_MOCK_REPLIES = {
    "open_greeting": (
        "Hi, I'm Maya. This is an important call from your primary care provider. I'm calling "
        "about Bhavya's Medi-Cal coverage status. This call will be recorded for training "
        "purposes. Do you have a couple of minutes?", False),
    "verify_identity": (
        "Great, thanks. And just so I've got the right person — is this Bhavya Shah?", False),
    "benefits_check": (
        "Perfect, thank you. So — we noticed you're not showing as assigned to us for your "
        "Medi-Cal anymore. Nothing to worry about — I just wanted to check in. Do you still "
        "have your Medi-Cal benefits?", False),
    "still_has_benefits_close": (
        "Oh, that's great to hear — then there's nothing you need to do at all. Thanks for "
        "your time — take care!", True),
    "counselor_assist": (
        "Okay, no worries at all. We've got health coverage counselors who can help sort out "
        "whether Medi-Cal or Covered California is the right fit for you. You can text the "
        "word KEEP, or call 1-800-555-0100.", False),
    "counselor_ack_close": (
        "You can call or text KEEP, our counselors will help you! We'll also send these "
        "details by text, so they're handy. Thanks for your time — take care!", True),
    "send_details_by_text": (
        "Of course — we'll send these details by text to this number, so they're handy. "
        "Thanks for your time — take care!", True),
    "retry_line": ("No problem — we'll try again soon. Take care!", True),
    "unclear_response": ("Sorry, could you say that once more for me?", False),
    "special_dnc": (
        "Of course — I'm making a note of that right now so we don't call again. Thanks for "
        "your time. Take care!", True),
    "special_abuse": (
        "I'm sorry for any frustration — we'll end the call here. Take care.", True),
    "special_busy": ("No problem — we'll try again soon. Take care!", True),
    "special_elsewhere": (
        "Got it, thank you for letting us know. If you ever change your mind, you can call "
        "1-800-555-0100 or text the word KEEP. Take care!", True),
    "special_pricing_q": (
        "That's a great question for our coverage counselors — they can walk you through all "
        "of that. You can text the word KEEP, or call 1-800-555-0100.", False),
    "special_recorded_q": ("Yes, this call is recorded for training purposes.", False),
    "special_ai_question": (
        "I am, yes — I'm Maya, a virtual assistant calling on behalf of Community Medical "
        "Center.", False),
}


class MockLLM:
    """Offline stand-in so the loop runs with no API key."""

    name = "MockLLM"

    def chat_json(self, system: str, user: str) -> str:
        match = _GOVERNING_RE.search(system)
        rule_id = match.group(1) if match else ""
        reply, ends = _MOCK_REPLIES.get(
            rule_id, ("Sorry, could you say a bit more about that?", False)
        )
        intent = rule_id[len("special_"):] if rule_id.startswith("special_") else "none"
        return json.dumps({
            "intent": intent,
            "reply": reply,
            "call_should_end": ends,
            "extracted_fields": {},
        })

    # Kept so the consolidator's LLM extraction has a usable offline path.
    def chat(self, system: str, messages: list[dict]) -> str:
        return json.dumps({"candidates": []})


class OpenAICompatibleLLM:
    def __init__(self, api_key=None, base_url=None, model=None):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or os.environ["SACE_LLM_KEY"],
            base_url=base_url or os.environ.get("SACE_LLM_BASE", "https://api.openai.com/v1"),
        )
        self._model = model or os.environ.get("SACE_LLM_MODEL", "gpt-4o-mini")
        self.name = self._model

    def _kwargs(self, messages, json_mode):
        kwargs = {"model": self._model, "messages": messages}
        # Reasoning models (gpt-5 family, o-series) reject an explicit
        # temperature with a 400.
        if not re.match(r"^(gpt-5|o\d)", self._model):
            kwargs["temperature"] = 0.2
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def chat_json(self, system: str, user: str) -> str:
        messages = build_messages(system, user)
        try:
            resp = self._client.chat.completions.create(**self._kwargs(messages, json_mode=True))
        except Exception:
            # Providers without response_format still work; the prompt asks for
            # JSON only and parse_json_object is tolerant.
            resp = self._client.chat.completions.create(**self._kwargs(messages, json_mode=False))
        return resp.choices[0].message.content

    def chat(self, system: str, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(**self._kwargs(
            [{"role": "system", "content": system}] + messages, json_mode=False
        ))
        return resp.choices[0].message.content


def get_llm():
    if os.environ.get("SACE_LLM_KEY"):
        return OpenAICompatibleLLM()
    return MockLLM()


def parse_json_object(raw: str):
    """Best-effort parse of a single JSON object from a model response.

    Returns (obj, error). Tries the whole string, then a fence-stripped version,
    then the outermost {...} span. `error` is a short reason when nothing parsed,
    so the caller can log why and fall back.
    """
    if not raw or not raw.strip():
        return None, "empty response"

    candidates = [raw.strip(), _FENCE_RE.sub("", raw).strip()]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError:
            continue

    return None, "no parseable JSON object"
