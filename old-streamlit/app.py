import base64
import json
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from loan_math import summarize

# .env sits next to this file (falls back to the folder above)
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
MODEL = os.getenv("LOAN_BOT_MODEL", "gemini-flash-lite-latest")
client = OpenAI()

EXTRACT_PROMPT = """You read personal loan documents (offers, contracts, schedules) in any language, including Georgian.
Return ONLY a JSON object with these keys. Use null if the document does not say. Never guess.
lender, currency, loan_amount, annual_interest_rate_pct, term_months, monthly_payment,
upfront_fees (total one-time fees as a number), early_repayment_penalty (short text),
late_payment_penalty (short text), collateral_or_guarantor (short text), notes (short text, anything unusual).
Numbers must be plain numbers with no symbols."""

CHAT_RULES = """You help a person understand their personal loan.
The facts below come from the document (reading errors are possible) and from exact calculations.
Use only these facts. If something is missing, say you don't know. Point out risks plainly
(high all-in rate, fees, penalties, payment mismatch). Reply in the user's language, short and clear.
This is information, not financial advice.
FACTS: """


def num(x):
    try:
        return float(re.sub(r"[^\d.\-]", "", str(x)))
    except ValueError:
        return 0.0


def extract(files):
    content = [{"type": "text", "text": EXTRACT_PROMPT}]
    for f in files:
        b64 = base64.b64encode(f.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{f.type};base64,{b64}"}})
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": content}])
    text = r.choices[0].message.content
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


st.title("Loan document checker")
st.sidebar.caption(f"Model: {MODEL}")

files = st.file_uploader("Upload a loan document (photo or screenshot)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
if files and st.button("Read document"):
    try:
        with st.spinner("Reading..."):
            st.session_state.loan = extract(files)
            st.session_state.chat = []
    except Exception as e:
        st.error(f"Could not read the document: {e}")

if "loan" in st.session_state:
    d = st.session_state.loan
    st.subheader("What I read (fix anything wrong)")
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Loan amount", value=num(d.get("loan_amount")), min_value=0.0, step=100.0)
    rate = c2.number_input("Annual rate %", value=num(d.get("annual_interest_rate_pct")), min_value=0.0, step=0.5)
    months = c3.number_input("Term (months)", value=int(num(d.get("term_months"))), min_value=0, step=1)
    c4, c5 = st.columns(2)
    fees = c4.number_input("Upfront fees", value=num(d.get("upfront_fees")), min_value=0.0, step=10.0)
    stated = c5.number_input("Monthly payment in document", value=num(d.get("monthly_payment")), min_value=0.0, step=10.0)

    facts = {
        "lender": d.get("lender"), "currency": d.get("currency"),
        "loan_amount": amount, "annual_rate_pct": rate, "term_months": months, "upfront_fees": fees,
        "early_repayment_penalty": d.get("early_repayment_penalty"),
        "late_payment_penalty": d.get("late_payment_penalty"),
        "collateral_or_guarantor": d.get("collateral_or_guarantor"), "notes": d.get("notes"),
    }
    if amount > 0 and months > 0:
        s = summarize(amount, rate, int(months), fees, stated)
        facts["calculated"] = s
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monthly payment", f"{s['monthly_payment_used']:,.2f}")
        m2.metric("Total repaid", f"{s['total_repaid']:,.2f}")
        m3.metric("Total cost", f"{s['total_cost_of_loan']:,.2f}")
        m4.metric("All-in rate", f"{s['all_in_annual_rate_pct']}%")
        if s["payment_mismatch"]:
            st.warning(f"Payment in the document ({stated:,.2f}) differs from the calculated one ({s['monthly_payment_calculated']:,.2f}). Check the rate, term and fees.")
    for label, key in [("Early repayment", "early_repayment_penalty"), ("Late payment", "late_payment_penalty"), ("Collateral / guarantor", "collateral_or_guarantor"), ("Notes", "notes")]:
        if d.get(key):
            st.write(f"**{label}:** {d[key]}")

    st.divider()
    history = st.session_state.setdefault("chat", [])
    for m in history:
        st.chat_message(m["role"]).write(m["content"])
    prompt = st.chat_input("Ask about this loan...")
    if st.button("Explain this loan and flag risks"):
        prompt = "Explain this loan in simple words and flag any risks."
    if prompt:
        history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
            msgs = [{"role": "system", "content": CHAT_RULES + json.dumps(facts, ensure_ascii=False)}] + history
            reply = client.chat.completions.create(model=MODEL, messages=msgs).choices[0].message.content
        except Exception as e:
            reply = f"Error: {e}"
        history.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)
