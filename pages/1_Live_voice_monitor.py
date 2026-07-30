"""Live voice monitor — the verification surface for the voice agent.

Reads the `turns` table, which voice_agent.py writes one row to per turn. The
point of this page is that `prompt_sent` is the EXACT string the speaking agent
handed to the LLM, captured inside llm_node at call time. Nothing here rebuilds
a prompt; it only displays stored bytes.

Added as a separate page rather than a tab so streamlit_app.py is untouched.
"""

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text as sql_text

load_dotenv()

from sace_chat.db import engine as db_engine
from sace_chat.engine import MONOLITH_TOKENS

st.set_page_config(page_title="sace-chat · voice monitor", page_icon="🎙️", layout="wide")

OUTCOME_COLOR = {
    "grounded": "#6ee7b7",
    "ungrounded": "#ffb454",
    "spliced": "#ff7b72",
    "no-rule": "#8b93a3",
    "unscored": "#8b93a3",
}

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .gov { background: linear-gradient(90deg, rgba(94,196,255,.10), #171b24);
             border: 1px solid #2b4a63; border-left: 3px solid #5ec4ff;
             border-radius: 10px; padding: 12px 14px; margin-bottom: 9px; }
      .ref { background: #14171e; border: 1px dashed #262c38; border-radius: 10px;
             padding: 9px 12px; margin-bottom: 8px; opacity: .62; }
      .rule-id { font-family: ui-monospace, Menlo, monospace; font-size: 13px;
                 font-weight: 700; color: #5ec4ff; }
      .role { display:block; font-size:9px; text-transform:uppercase;
              letter-spacing:.12em; color:#6b7383; margin-bottom:3px; }
      .badge { display:inline-block; font-size:9.5px; font-weight:700;
               text-transform:uppercase; letter-spacing:.06em; padding:2px 6px;
               border-radius:4px; margin-left:6px; vertical-align:middle; }
      .meta { font-family: ui-monospace, Menlo, monospace; font-size:10.5px;
              color:#6b7383; margin-top:4px; }
      .lat { font-family: ui-monospace, Menlo, monospace; font-size:11px; color:#9aa3b4; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=1.0, show_spinner=False)
def load_turns(limit: int, session_filter: str | None):
    """Recent voice turns. Cached for 1s so the auto-refresh does not hammer
    Postgres while still feeling live."""
    where = "WHERE source = 'voice'"
    params = {"limit": limit}
    if session_filter and session_filter != "(all)":
        where += " AND session_id = :sid"
        params["sid"] = session_filter
    with db_engine.connect() as conn:
        rows = conn.execute(sql_text(
            f"SELECT * FROM turns {where} ORDER BY created_at DESC, turn_index DESC "
            "LIMIT :limit"
        ), params).mappings().all()
        sessions = [r[0] for r in conn.execute(sql_text(
            "SELECT DISTINCT session_id FROM turns WHERE source='voice' "
            "ORDER BY session_id DESC LIMIT 50"
        )).all()]
    return [dict(r) for r in rows], sessions


@st.cache_data(ttl=5.0, show_spinner=False)
def load_rule_meta():
    """id -> (text, source, learned_kind, terminal) so a stored rule id can be
    shown with the rule's own text and a LEARNED badge."""
    with db_engine.connect() as conn:
        rows = conn.execute(sql_text(
            "SELECT id, title, text, source, learned_kind, terminal, intent FROM chunks"
        )).mappings().all()
    return {r["id"]: dict(r) for r in rows}


@st.cache_data(ttl=5.0, show_spinner=False)
def load_transcripts(limit: int = 10):
    with db_engine.connect() as conn:
        rows = conn.execute(sql_text(
            "SELECT session_id, turn_count, learning_results, transcript, created_at "
            "FROM call_transcripts WHERE source='voice' "
            "ORDER BY created_at DESC LIMIT :limit"
        ), {"limit": limit}).mappings().all()
    return [dict(r) for r in rows]


def rule_card(rule_id, meta, governing: bool, cosine=None):
    info = meta.get(rule_id)
    if info is None:
        st.markdown(
            f"<div class='{'gov' if governing else 'ref'}'><span class='rule-id'>{rule_id}</span>"
            f"<div class='meta'>rule no longer in the pool</div></div>",
            unsafe_allow_html=True,
        )
        return
    badges = ""
    if info["intent"]:
        badges += ("<span class='badge' style='color:#ffb454;background:#ffb45422;"
                   f"border:1px solid #ffb45444'>intent · {info['intent']}</span>")
    if info["terminal"]:
        badges += ("<span class='badge' style='color:#ff7b72;background:#ff7b7222;"
                   "border:1px solid #ff7b7244'>terminal</span>")
    if info["source"] == "learned":
        badges += ("<span class='badge' style='color:#6ee7b7;background:#6ee7b722;"
                   f"border:1px solid #6ee7b744'>learned · {info['learned_kind'] or '?'}</span>")
    body = info["text"] if governing else info["text"][:200] + "…"
    cos = "" if cosine is None else f" · reply cos {cosine:.3f}"
    st.markdown(
        f"<div class='{'gov' if governing else 'ref'}'>"
        f"<span class='role'>{'governing rule — the only source of this reply' if governing else 'reference — background only'}</span>"
        f"<span class='rule-id'>{rule_id}</span>{badges}"
        f"<div class='meta'>{len(info['text'])} chars{cos}</div>"
        f"<div class='meta' style='color:#9aa3b4;line-height:1.5'>{body}</div></div>",
        unsafe_allow_html=True,
    )


st.markdown("#### Live voice monitor")
st.caption(
    "One row per spoken turn, written by `voice_agent.py`. `prompt_sent` is captured "
    "inside `llm_node` at call time — this is the prompt the speaking agent used."
)

col_a, col_b, col_c = st.columns([1, 1, 2])
limit = col_a.selectbox("turns to show", [10, 25, 50, 100], index=1)
auto = col_b.toggle("auto-refresh (1s)", value=True)

_rows, _sessions = load_turns(limit, None)
session_filter = col_c.selectbox("session", ["(all)"] + _sessions)


@st.fragment(run_every=1.0 if auto else None)
def monitor():
    rows, _ = load_turns(limit, session_filter)
    meta = load_rule_meta()

    if not rows:
        st.info(
            "No voice turns recorded yet. Start the agent with "
            "`python voice_agent.py dev` (or `console`) and speak.",
            icon="🎙️",
        )
        return

    latest = rows[0]
    m = st.columns(5)
    m[0].metric("turns recorded", len(rows))
    m[1].metric("last assembled", f"{latest['assembled_tokens'] or 0:,} tok")
    saved = (1 - (latest["assembled_tokens"] or 0) / MONOLITH_TOKENS) * 100
    m[2].metric("saved vs 5,782", f"{saved:.1f}%")
    m[3].metric("last grounding cos", f"{latest['grounding_cosine'] or 0:.3f}")
    m[4].metric("last end-to-end", f"{latest['latency_ms'] or 0:.0f} ms")

    lat = [r["latency_ms"] for r in rows if r["latency_ms"]]
    if lat:
        st.caption(
            f"latency over {len(lat)} turn(s) · median **{sorted(lat)[len(lat)//2]:.0f} ms** · "
            f"max **{max(lat):.0f} ms** · budget 1,500 ms"
        )

    st.divider()

    for row in rows:
        colour = OUTCOME_COLOR.get(row["validation_outcome"], "#8b93a3")
        head = (
            f"turn {row['turn_index']} · {row['governing_rule_id'] or 'no rule'} · "
            f"{row['validation_outcome']} cos {row['grounding_cosine'] or 0:.3f} · "
            f"{row['latency_ms'] or 0:.0f} ms"
        )
        with st.expander(head, expanded=(row is rows[0])):
            st.markdown(f"**Caller:** {row['user_text']}")
            st.markdown(f"**Maya:** {row['reply_text']}")
            st.markdown(
                f"<div class='lat'>stt {row['stt_ms'] or 0:.0f} · ctx {row['context_ms'] or 0:.0f}"
                f" · llm ttft {row['llm_ttft_ms'] or 0:.0f} · tts ttfb {row['tts_ttfb_ms'] or 0:.0f}"
                f" · total {row['latency_ms'] or 0:.0f} ms &nbsp;|&nbsp; intent "
                f"{row['intent'] or 'none'} @ {row['intent_cosine'] or 0:.3f}"
                f" &nbsp;|&nbsp; <span style='color:{colour}'>{row['validation_outcome']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            if row["governing_rule_id"]:
                rule_card(row["governing_rule_id"], meta, True, row["grounding_cosine"])
            for rid in (row["reference_rule_ids"] or "").split(","):
                if rid.strip():
                    rule_card(rid.strip(), meta, False)

            tokens = row["assembled_tokens"] or 0
            st.progress(min(1.0, max(0.0, (1 - tokens / MONOLITH_TOKENS))))
            st.caption(
                f"assembled **{tokens:,}** vs monolith **{MONOLITH_TOKENS:,}** — "
                f"saved **{(1 - tokens / MONOLITH_TOKENS) * 100:.1f}%**"
            )

            prompt = row["prompt_sent"] or ""
            contains_user = row["user_text"] in prompt
            gov_meta = meta.get(row["governing_rule_id"] or "")
            st.caption(
                ("✅" if contains_user else "❌")
                + " prompt contains this turn's transcribed speech verbatim"
            )
            st.code(prompt, language="markdown")
            if gov_meta:
                st.caption(
                    f"{len(prompt):,} chars sent · governing rule id "
                    f"`{row['governing_rule_id']}` appears in it: "
                    f"{'yes' if row['governing_rule_id'] in prompt else 'no'}"
                )

    st.divider()
    st.markdown("**Finished calls & learning**")
    for t in load_transcripts():
        results = t["learning_results"] or []
        with st.expander(
            f"{t['session_id']} · {t['turn_count']} turns · {len(results)} candidate(s)"
        ):
            if not results:
                st.caption("No candidate rules extracted.")
            for r in results:
                icon = {"inserted": "✅", "duplicate-skipped": "♻️",
                        "conflict-needs-review": "⚠️", "ungrounded-rejected": "🚫"}.get(
                    r.get("outcome"), "•")
                st.markdown(
                    f"{icon} **{r.get('outcome')}** · `intent={r.get('intent')}` "
                    f"· `{r.get('learned_kind')}`"
                )
                st.caption(r.get("text", ""))
                if r.get("detail"):
                    st.caption(f"↳ {r['detail']}")
            st.caption("transcript")
            st.code(t.get("transcript", ""), language="text")


monitor()
