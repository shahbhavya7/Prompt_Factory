"""sace-chat proof UI.

Left: the chat. Right: what retrieval actually did — which single rule governed
the turn, what was merely in scope beside it, whether the reply is traceable to
the governing rule, and the prompt size against the 5,782-token monolith.

Run:  streamlit run streamlit_app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from sace_chat import manager
from sace_chat.consolidator import run_learning_loop
from sace_chat.db import engine as db_engine, init_db
from sace_chat.embeddings import get_embedder
from sace_chat.engine import MONOLITH_TOKENS, Engine
from sace_chat.kb import RULES, STABLE_CORE
from sace_chat.llm import get_llm
from sace_chat.retrieve import CallState

st.set_page_config(page_title="sace-chat", page_icon="🎧", layout="wide")

OUTCOME_STYLE = {
    "grounded": ("#6ee7b7", "grounded in the governing rule"),
    "ungrounded": ("#ffb454", "not traceable to the governing rule"),
    "spliced": ("#ff7b72", "spliced content from a reference rule"),
    "no-rule": ("#8b93a3", "no rule retrieved"),
    "unscored": ("#8b93a3", "not scored"),
}

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
      .gov {
        background: linear-gradient(90deg, rgba(94,196,255,.10), #171b24);
        border: 1px solid #2b4a63; border-left: 3px solid #5ec4ff;
        border-radius: 10px; padding: 13px 15px; margin-bottom: 10px;
      }
      .ref {
        background: #14171e; border: 1px dashed #262c38; border-radius: 10px;
        padding: 10px 13px; margin-bottom: 8px; opacity: .62;
      }
      .rule-id { font-family: ui-monospace, Menlo, monospace; font-size: 13px; font-weight: 700; color: #5ec4ff; }
      .ref .rule-id { font-size: 11.5px; color: #8b93a3; }
      .role {
        display: block; font-size: 9px; text-transform: uppercase; letter-spacing: .12em;
        color: #6b7383; margin-bottom: 3px;
      }
      .badge {
        display: inline-block; font-size: 9.5px; font-weight: 700; text-transform: uppercase;
        letter-spacing: .06em; padding: 2px 6px; border-radius: 4px; margin-left: 6px;
        vertical-align: middle;
      }
      .meta { font-family: ui-monospace, Menlo, monospace; font-size: 10.5px; color: #6b7383; margin-top: 5px; }
      .snip { font-family: ui-monospace, Menlo, monospace; font-size: 10.5px; color: #9aa3b4; margin-top: 6px; line-height: 1.5; }
      .ref .snip { font-size: 10px; }
      .proof { border-radius: 10px; padding: 10px 12px; margin-bottom: 12px; font-size: 13px; }
      .proof .lbl { font-size: 9.5px; text-transform: uppercase; letter-spacing: .09em; color: #6b7383; display: block; }
      .proof .val { font-family: ui-monospace, Menlo, monospace; font-weight: 650; }
      .statebox {
        background: #1d222c; border: 1px solid #1e242e; border-radius: 8px;
        padding: 7px 10px; text-align: center;
      }
      .statebox .k { font-size: 9px; text-transform: uppercase; letter-spacing: .08em; color: #6b7383; display: block; }
      .statebox .v { font-family: ui-monospace, Menlo, monospace; font-size: 12.5px; font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def boot():
    init_db()
    embedder = get_embedder()
    llm = get_llm()
    eng = Engine(
        stable_core=STABLE_CORE,
        rules=RULES,
        embedder=embedder,
        manager=manager,
        llm=llm,
    )
    # Embeds the intent exemplars once, here, rather than on the first caller
    # turn where it would show up as latency.
    eng.router.warm()
    return eng, embedder, llm


engine, embedder, llm = boot()


def new_call():
    st.session_state.state = CallState()
    st.session_state.history = []
    st.session_state.messages = []
    st.session_state.turns = []
    st.session_state.learning = None


if "state" not in st.session_state:
    new_call()

# ─────────────────────────── sidebar ───────────────────────────
with st.sidebar:
    st.subheader("Call controls")

    if st.button("Reset call", use_container_width=True):
        new_call()
        st.rerun()

    if st.button("End call & learn", use_container_width=True, type="primary"):
        transcript = "\n".join(st.session_state.history)
        if not transcript.strip():
            st.session_state.learning = {"error": "Nothing to learn from — the call is empty."}
        else:
            with st.spinner("Extracting and gating candidate rules…"):
                try:
                    with db_engine.connect() as conn:
                        results = run_learning_loop(transcript, embedder, conn, llm=llm)
                    st.session_state.learning = {
                        "results": [
                            {
                                "text": r.candidate.text,
                                "intent": r.candidate.intent or "(general)",
                                "kind": r.candidate.learned_kind,
                                "outcome": r.outcome,
                                "detail": r.detail,
                            }
                            for r in results
                        ]
                    }
                except Exception as exc:
                    st.session_state.learning = {"error": f"{type(exc).__name__}: {exc}"}
        st.session_state.state.ended = True
        st.rerun()

    st.divider()
    st.caption(f"model · `{getattr(llm, 'name', type(llm).__name__)}`")
    st.caption(f"embeddings · `{os.environ.get('EMBEDDING_MODE', 'mock')}`")
    st.caption(f"monolith baseline · `{MONOLITH_TOKENS} tok`")
    st.caption(f"rules in memory · `{len(RULES)} seed`")
    if st.session_state.state.ended:
        st.warning("Call ended.", icon="🛑")

# ─────────────────────────── layout ───────────────────────────
st.markdown("#### sace-chat — memory-only retrieval")
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    for msg in st.session_state.messages:
        with st.chat_message("user" if msg["role"] == "caller" else "assistant"):
            st.markdown(msg["text"])

    if prompt := st.chat_input("Type the caller's turn…", disabled=st.session_state.state.ended):
        st.session_state.messages.append({"role": "caller", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("…"):
                try:
                    reply, _, debug = engine.step(
                        st.session_state.state, st.session_state.history, prompt
                    )
                except Exception as exc:
                    reply, debug = f"[engine error: {type(exc).__name__}: {exc}]", None
            st.markdown(reply)
        st.session_state.messages.append({"role": "maya", "text": reply})
        if debug:
            st.session_state.turns.append(debug)
        st.rerun()


def rule_card(rule, governing: bool):
    badges = ""
    if rule["intent"]:
        badges += (
            "<span class='badge' style='color:#ffb454;background:#ffb45422;"
            f"border:1px solid #ffb45444'>intent · {rule['intent']}</span>"
        )
    if rule["terminal"]:
        badges += (
            "<span class='badge' style='color:#ff7b72;background:#ff7b7222;"
            "border:1px solid #ff7b7244'>terminal</span>"
        )
    if rule["exclusive"]:
        badges += (
            "<span class='badge' style='color:#c9a0ff;background:#c9a0ff22;"
            "border:1px solid #c9a0ff44'>exclusive</span>"
        )
    if rule["source"] == "learned":
        badges += (
            "<span class='badge' style='color:#6ee7b7;background:#6ee7b722;"
            f"border:1px solid #6ee7b744'>learned · {rule['learned_kind'] or '?'}</span>"
        )
    cos = "—" if rule["cosine"] is None else f"{rule['cosine']:.3f}"
    body = rule["text"] if governing else rule["snippet"] + "…"
    st.markdown(
        f"<div class='{'gov' if governing else 'ref'}'>"
        f"<span class='role'>{'governing rule — the only source of this reply' if governing else 'reference — background only'}</span>"
        f"<span class='rule-id'>{rule['id']}</span>{badges}"
        f"<div class='meta'>retrieval sim {rule['similarity']:.3f} · reply cos {cos} · {rule['char_len']} chars</div>"
        f"<div class='snip'>{body}</div></div>",
        unsafe_allow_html=True,
    )


with right:
    turn = st.session_state.turns[-1] if st.session_state.turns else None

    st.markdown("**Call state**")
    snap = turn["state_snapshot"] if turn else {
        "intent": st.session_state.state.intent,
        "opt_out": st.session_state.state.opt_out,
        "ended": st.session_state.state.ended,
        "asked_questions": [],
    }
    cols = st.columns(4)
    for col, (k, v) in zip(cols, [
        ("intent", snap["intent"]),
        ("opt-out", "yes" if snap["opt_out"] else "no"),
        ("ended", "yes" if snap["ended"] else "no"),
        ("asked", str(len(snap["asked_questions"]))),
    ]):
        color = "#ffb454" if v not in ("none", "no", "0") else "#6b7383"
        col.markdown(
            f"<div class='statebox'><span class='k'>{k}</span>"
            f"<span class='v' style='color:{color}'>{v}</span></div>",
            unsafe_allow_html=True,
        )

    if turn is None:
        st.info("Send a caller turn to populate the proof panel.", icon="💬")
    else:
        st.markdown("")
        color, label = OUTCOME_STYLE.get(turn["outcome"], ("#8b93a3", turn["outcome"]))
        extra = " · ⟳ regenerated once" if turn["regenerated"] else ""
        st.markdown(
            f"<div class='proof' style='background:{color}1a;border:1px solid {color}4d'>"
            f"<span class='lbl'>validation</span>"
            f"<span class='val' style='color:{color}'>{label} · governing cos "
            f"{turn['governing_cosine']:.3f} (threshold {turn['grounding_threshold']}){extra}</span></div>",
            unsafe_allow_html=True,
        )

        if turn["intent"]:
            st.caption(
                f"intent **{turn['intent']}** matched at cos **{turn['intent_similarity']:.3f}** — "
                "routed straight to that rule"
            )
        else:
            runner = turn["intent_ranked"][0] if turn["intent_ranked"] else ("—", 0.0)
            st.caption(
                f"no intent matched (closest **{runner[0]}** at {runner[1]:.3f}) — searched the general pool"
            )

        if turn["governing"]:
            rule_card(turn["governing"], governing=True)
        for ref in turn["reference"]:
            rule_card(ref, governing=False)
        if not turn["reference"]:
            st.caption("nothing else in scope this turn")

        st.markdown("**Prompt size**")
        m1, m2, m3 = st.columns(3)
        m1.metric("assembled", f"{turn['assembled_prompt_tokens']:,}")
        m2.metric("monolith", f"{turn['monolith_tokens']:,}")
        m3.metric("saved", f"{turn['saved_pct']:.1f}%")
        st.progress(min(1.0, max(0.0, turn["saved_pct"] / 100)))

        avg = sum(t["saved_pct"] for t in st.session_state.turns) / len(st.session_state.turns)
        st.caption(
            f"session average saved **{avg:.1f}%** across {len(st.session_state.turns)} turn(s) · "
            f"last turn **{turn['elapsed_ms']:.0f} ms**"
        )

        for n in turn.get("notes") or []:
            st.caption(f"⚠︎ {n}")

        with st.expander("View assembled prompt"):
            st.caption(f"{len(turn['assembled_prompt']):,} chars — exactly what was sent")
            st.code(turn["assembled_prompt"], language="markdown")

        with st.expander("Turn JSON"):
            st.json(turn["turn_json"])
            st.caption("retrieval query (Maya's last line + the caller's message)")
            st.code(turn["query_text"])
            st.caption("intent ranking")
            st.json(turn["intent_ranked"])
            st.caption("raw model output")
            st.code(turn["raw_llm_output"] or "(empty)", language="json")

        with st.expander("Learning log"):
            learning = st.session_state.learning
            if not learning:
                st.caption("Run **End call & learn** in the sidebar after a call.")
            elif "error" in learning:
                st.error(learning["error"])
            elif not learning["results"]:
                st.caption("No candidate rules extracted — nothing new in this call.")
            else:
                icon = {
                    "inserted": "✅",
                    "duplicate-skipped": "♻️",
                    "conflict-needs-review": "⚠️",
                    "ungrounded-rejected": "🚫",
                }
                for row in learning["results"]:
                    st.markdown(
                        f"{icon.get(row['outcome'], '•')} **{row['outcome']}** "
                        f"· `intent={row['intent']}` · `{row['kind']}`"
                    )
                    st.caption(row["text"])
                    if row["detail"]:
                        st.caption(f"↳ {row['detail']}")
