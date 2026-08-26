import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kb_env
kb_env.pin()
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from sace_chat import answer_cache as ac
from sace_chat.db import engine as db_engine, SessionLocal
from sace_chat.embeddings import get_embedder
from sqlalchemy import text as sql

emb = get_embedder()
ac.clear()
fp = ac.question_fingerprint
Q = "where is the clinic located"
PEND_A = fp("do you still have Medi-Cal benefits?")
PEND_B = fp("would you like me to repeat that information?")
v = emb.embed(Q)

a = ac.store(question=Q, question_vec=v, reply="Counselors can help. For now, do you still have Medi-Cal benefits?",
             intent="clinic_location", governing_rule_id="special_clinic_location", pending=PEND_A)
b = ac.store(question=Q, question_vec=v, reply="Counselors can help. Would you like me to repeat that information?",
             intent="clinic_location", governing_rule_id="special_clinic_location", pending=PEND_B)
c = ac.store(question=Q, question_vec=v, reply="Counselors can help with that.",
             intent="clinic_location", governing_rule_id="special_clinic_location", pending="")
print(f"3 distinct rows, not collapsed: {len({a,b,c})==3}   ids={a},{b},{c}")

with db_engine.connect() as conn:
    for label, pend in [("pending=A", PEND_A), ("pending=B", PEND_B),
                        ("pending=UNRELATED", fp("what is your date of birth"))]:
        h = ac.lookup(conn, v, "clinic_location", pending=pend)
        print(f"  {label:20} -> {(h['id'] if h else 'None'):16} {h['reply'][:54] if h else '-'}")

    hb = ac.lookup(conn, v, "clinic_location", pending=PEND_B)
    print(f"\n  pinned-to-A NEVER served under pending B : {hb is None or hb['id'] != a}")
    hu = ac.lookup(conn, v, "clinic_location", pending=fp("what is your date of birth"))
    print(f"  unrelated pending -> only the unpinned row: {hu is not None and hu['id'] == c}")
    print(f"  intent=None (flow turn) -> always None    : {ac.lookup(conn, v, None, pending=PEND_A) is None}")
    print(f"  never-cache intent -> always None         : {ac.lookup(conn, v, 'dnc', pending='') is None}")

d = ac.store(question="where is the clinic located?", question_vec=v, reply="Updated.",
             intent="clinic_location", governing_rule_id="special_clinic_location", pending=PEND_A)
with SessionLocal() as s:
    n = s.execute(sql("SELECT count(*) FROM answer_cache")).scalar()
print(f"\n  same vector + same pending updates in place: {d == a} (rows still {n})")

blocked = ac.store(question=Q, question_vec=v, reply="x", intent=None,
                   governing_rule_id="open_greeting", pending="")
blocked2 = ac.store(question=Q, question_vec=v, reply="x", intent="dnc",
                    governing_rule_id="special_dnc", pending="")
print(f"  store() refuses intent=None                : {blocked is None}")
print(f"  store() refuses a never-cache rule         : {blocked2 is None}")
ac.clear()
print("\ncleaned up")
