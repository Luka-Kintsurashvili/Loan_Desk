import json
import os
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib

import credit_model as cm

HERE = Path(__file__).parent
env = HERE / ".env"
for line in env.read_text().splitlines() if env.exists() else []:
    m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*?)\s*$", line)
    if m:
        os.environ.setdefault(m[1], m[2])
KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
BASE = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
MODEL = os.environ.get("MODEL", "gemini-flash-lite-latest")
if not KEY:
    raise SystemExit("Put LLM_API_KEY=... in the .env file")
if not cm.PATH.exists():
    raise SystemExit("Train the model first: python credit_model.py data\\LC_loans_granting_model_dataset.csv")
B = joblib.load(cm.PATH)

# EDIT ME: example limits only, not regulations. Set your lender's real policy in policy.json.
POLICY = {"max_pd": 0.15, "max_pti": 0.20, "max_dti_after": 0.43, "max_ltv": 0.80,
          "min_income": 24000, "min_fico": 600, "default_rate_pct": 12.0, "currency": "$", "usd_per_unit": 1.0}
pf = HERE / "policy.json"
if pf.exists():
    POLICY.update(json.loads(pf.read_text()))
else:
    pf.write_text(json.dumps(POLICY, indent=2))

F = set(B["features"])
pick = lambda *names: next((n for n in names if n in F), None)
M_AMT, M_INC = pick("loan_amnt", "amount_requested", "loan_amount"), pick("annual_inc", "revenue", "income", "annual_income")
M_DTI, M_FICO, M_TERM = pick("dti", "dti_n"), pick("fico_range_low", "fico_n", "fico", "risk_score"), pick("term")
AMT, TERM, INC, DTI, FICO = "loan_amount", "term_months", "annual_income", "dti_pct", "credit_score"
ROLE = {f: k for f, k in ((M_AMT, AMT), (M_INC, INC), (M_DTI, DTI), (M_FICO, FICO), (M_TERM, TERM)) if f}
print("Model inputs matched to chat fields:", {k: (f"{f} (typical {B['baseline'][f]:.4g})" if f in B["num"] else f) for f, k in ROLE.items()})
print("Other model inputs:", [f for f in B["features"] if f not in ROLE])
EXTRA = {"rate_pct": "interest rate the lender offers, percent", "collateral_value": "market value of collateral",
         "monthly_income": "income per month, if the user gives it monthly",
         "monthly_debt_payments": "existing recurring loan and credit card payments per month; the server turns it into DTI", AMT: "loan amount",
         TERM: "loan term in months, e.g. 24", INC: "income per year",
         DTI: "existing monthly debt payments as % of income, 18 means 18%", FICO: "credit score (FICO), only if known"}
ALLOWED = (F - set(ROLE)) | set(EXTRA)


HINTS = {"experience_c": "1 if the applicant has any credit history, 0 if none (not years)"}


def describe(f):
    if f in HINTS:
        return f"{f} ({HINTS[f]})"
    if f in B["num"]:
        return f"{f} (number, typical {B['baseline'][f]:.4g})"
    return f"{f} (one of: {', '.join(map(str, B['cat_values'][f][:12]))})"


FIELDS = ("ALLOWED KEYS:\n" + "\n".join("- " + describe(f) for f in B["features"] if f in ALLOWED) + "\n"
          + "\n".join(f"- {k} ({v})" for k, v in EXTRA.items()) + "\n")
PARSE = ("Extract loan application facts from the user message and any attached documents. Return ONLY a JSON object "
         "{key: value} using only the allowed keys. Amounts are plain numbers in the lender's currency, no symbols. If income is "
         "given per month return monthly_income, if per year return annual_income. term_months is in months, so convert years (7 years = 84). dti and rates are percent "
         "numbers (dti 18 means 18%). From a pay stub or bank statement use monthly_income (regular salary per month) and monthly_debt_payments (recurring loan and credit card payments per month, not rent or utilities). If the document says collateral is none, return collateral_value 0. Omit unknown keys. Never guess.\n")
COMPOSE = ("You assist a loan officer. Write plain text only: no markdown, no asterisks, no bullet symbols. Use short lines, and "
           "numbered lines like '1) ...' for lists. Reply in the user's language. Use ONLY the CASE and DECISION below and never invent "
           "numbers or failed checks. Do not mention model metrics, policy limits or disclaimers unless asked; the page shows them.\n"
           "If the user asks a question, answer it first, briefly.\n"
           "If DECISION.ask or DECISION.ask_more is not empty, ask for everything at once in one message (after the verdict if there is one): "
           "first the items in ask (needed for the policy checks), then the items in ask_more (they make the default estimate more "
           "accurate, most useful first). Use one numbered list, a few plain words each. Say they can answer in any order and skip "
           "what they don't know. Never ask again for anything already in CASE. If ask_more_total is larger than the list, say more "
           "details exist.\n"
           "If DECISION.request exists, start with one verdict line (meets_policy = fits policy; conditions = does not fit as "
           "requested but fits with changes; refer = does not fit policy, refer or decline), then the chance of default as a "
           "percentage (request.pd), the monthly payment, and each failed check with its value and limit. Add one or two short lines "
           "on the biggest risk reasons (reason values are converted to USD). Then give the options (amount, term, payment) and the "
           "collateral value needed (collateral_needed) only if the user asks about collateral or the loan/collateral check failed. If rate_assumed is true, say the rate is assumed.\n"
           "The default chance does not depend on term or interest rate (the model has no such input), so options that differ only by term share the same default chance: say so briefly when you compare them. If JUST READ is not empty, first confirm in one short line what you read so the user can correct it. If out_of_range is true, "
           "say the model only saw personal loans up to about $40,000, so the default chance is unreliable here (a business loan needs a different model).\n")
def num(v):
    m = re.search(r"(-?\d+\.?\d*)\s*(k)?(?![a-zA-Z])", str(v).replace(",", ""), re.I)
    return float(m[1]) * (1000 if m[2] else 1) if m else 0.0


LABELS = {"loan_amnt": "loan amount", "emp_length": "years at current job", "home_ownership": "renting, owning or mortgage", "home_ownership_n": "renting, owning or mortgage",
          "purpose": "purpose of the loan", "fico_range_low": "credit score (FICO)", "fico_n": "credit score (FICO)", "revenue": "annual income",
          "dti_n": "debt-to-income", "experience_c": "has any credit history (yes or no)", "verification_status": "is the income verified",
          "delinq_2yrs": "late payments (30+ days) in the last 2 years", "inq_last_6mths": "credit inquiries in the last 6 months",
          "open_acc": "number of open credit accounts", "pub_rec": "public records (bankruptcies, judgments)", "revol_bal": "credit card balance",
          "revol_util": "credit card utilization %", "total_acc": "total credit accounts ever opened", "mort_acc": "number of mortgage accounts",
          "application_type": "individual or joint application", "pub_rec_bankruptcies": "bankruptcies"}
CORE = [(AMT, "loan amount"), (TERM, "term in months"), (INC, "income (monthly or yearly)"), ("rate_pct", "interest rate offered"),
        (DTI, "existing monthly debt payments as % of income"), ("collateral_value", "collateral value, if the loan is secured")]


def ask(case):
    core = [l for k, l in CORE if case.get(k) in (None, "")]
    imp = B.get("importance", {})
    rest = sorted((f for f in B["features"] if f != "fico_range_high" and ROLE.get(f, f) not in (AMT, TERM, INC, DTI)
                   and case.get(ROLE.get(f, f)) in (None, "")), key=lambda f: -imp.get(f, 0))
    return core, [LABELS.get(f, f.replace("_", " ")) for f in rest]


def to_model(case, amt, inc):
    sc = {k: v for k, v in case.items() if k in F and k not in ROLE}
    for f, v in ((M_AMT, amt), (M_INC, inc), (M_DTI, case.get(DTI)), (M_FICO, case.get(FICO)), (M_TERM, case.get(TERM))):
        if f and v not in (None, ""):
            sc[f] = v
    if M_FICO == "fico_range_low" and "fico_range_high" in F and sc.get(M_FICO):
        sc["fico_range_high"] = sc[M_FICO] + 4
    return sc


def var(sc, a, t):
    v = dict(sc)
    if M_AMT:
        v[M_AMT] = a
    if M_TERM:
        v[M_TERM] = t
    return v


REASON = {"home_ownership_n": "housing status", "experience_c": "credit history", "revenue": "annual income"}


def named(rs):
    return {k: [{**r, "feature": REASON.get(r["feature"]) or LABELS.get(r["feature"], r["feature"].replace("_", " "))} for r in v] for k, v in rs.items()}


def pmt(p, a, n):
    r = a / 1200
    return p / n if r == 0 else p * r / (1 - (1 + r) ** -n)


def evaluate(case):
    amt, n, inc = num(case.get(AMT)), num(case.get(TERM)), num(case.get(INC))
    core, more = ask(case)
    d = {"currency": POLICY["currency"], "ask": core, "ask_more": more[:15], "ask_more_total": len(more), "model": B["metrics"], "policy": POLICY,
         "missing": [k for k, v in ((AMT, amt), (TERM, n), (INC, inc)) if not v]}
    if d["missing"]:
        d["verdict"] = "need_info"
        return d
    rate = num(case.get("rate_pct"))
    d["rate_assumed"] = not rate
    d["out_of_range"] = amt * POLICY["usd_per_unit"] > 40000
    rate = rate or POLICY["default_rate_pct"]
    dti, coll = num(case.get(DTI)), num(case.get("collateral_value"))
    fico = num(case.get(FICO)) if FICO else 0

    def row(a, t, p):
        pay, mi = pmt(a, rate, t), inc / 12
        c = [("Default probability", p, POLICY["max_pd"], p <= POLICY["max_pd"]),
             ("Payment / income", pay / mi, POLICY["max_pti"], pay / mi <= POLICY["max_pti"])]
        if case.get(DTI) not in (None, ""):
            x = dti / 100 + pay / mi
            c.append(("DTI with this loan", x, POLICY["max_dti_after"], x <= POLICY["max_dti_after"]))
        if coll:
            x = a / coll
            c.append(("Loan / collateral value", x, POLICY["max_ltv"], x <= POLICY["max_ltv"]))
        c.append(("Annual income", inc, POLICY["min_income"], inc >= POLICY["min_income"]))
        if fico:
            c.append(("FICO score", fico, POLICY["min_fico"], fico >= POLICY["min_fico"]))
        return {"amount": a, "term": int(t), "payment": round(pay, 2), "pd": round(p, 4), "ok": all(z[3] for z in c),
                "checks": [dict(zip(("name", "value", "limit", "ok"), z)) for z in c]}

    fx = POLICY["usd_per_unit"]
    sc = to_model(case, amt * fx, inc * fx)  # the model was trained on USD
    main = row(amt, n, float(cm.score(B, [sc])[0]))
    cands = [(round(amt * f, -2), t) for f in (1, .9, .8, .7, .6, .5) for t in sorted({int(n), 36, 60})
             if not (f == 1 and t == int(n))]
    pds = cm.score(B, [var(sc, a * fx, t) for a, t in cands])
    opts = sorted((r for r in (row(a, t, float(p)) for (a, t), p in zip(cands, pds)) if r["ok"]),
                  key=lambda r: (-r["amount"], r["pd"]))
    d.update(request=main, options=opts[:3], reasons=named(cm.reasons(B, sc)),
             collateral_needed=round(amt / POLICY["max_ltv"]),
             verdict="meets_policy" if main["ok"] else ("conditions" if opts else "refer"),
             completeness=round(sum(case.get(ROLE.get(f, f)) not in (None, "") for f in B["features"]) / len(B["features"]), 2))
    return d


def gemini(messages):
    req = urllib.request.Request(
        BASE + "/chat/completions",
        json.dumps({"model": MODEL, "messages": messages}).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM {e.code}: {e.read().decode()[:300]}")


def merge(case, ups):
    case = dict(case)
    ups = dict(ups or {})
    if ups.get("monthly_income") not in (None, ""):
        ups[INC] = num(ups["monthly_income"]) * 12
    ups.pop("monthly_income", None)
    for k, v in ups.items():
        if k in ALLOWED and v not in (None, ""):
            if k in HINTS:
                case[k] = 1 if str(v).strip().lower() in ("yes", "true", "y") or num(v) > 0 else 0
            else:
                case[k] = num(v) if (k in B["num"] or k in EXTRA) else v
    if case.get("monthly_debt_payments") not in (None, "") and num(case.get(INC)) and case.get(DTI) in (None, ""):
        case[DTI] = round(100 * num(case["monthly_debt_payments"]) / (num(case[INC]) / 12), 1)
    if case.get("fico_range_low") and "fico_range_high" in F and case.get("fico_range_high") in (None, ""):
        case["fico_range_high"] = case["fico_range_low"] + 4
    return case


def chat(body):
    case, msgs, images = body.get("case", {}), body.get("messages", []), body.get("images", [])
    text = msgs[-1]["content"]
    read, raw = {}, ""
    if images or re.search(r"\d", text):  # only spend a model call on parsing when there may be facts
        content = [{"type": "text", "text": PARSE + FIELDS + "CURRENT CASE: " + json.dumps(case) + "\nUSER MESSAGE: " + text}]
        content += [{"type": "image_url", "image_url": {"url": u}} for u in images[:3]]
        try:
            raw = gemini([{"role": "user", "content": content}])
            ups = json.loads(re.search(r"\{[\s\S]*\}", raw)[0])
            read = {k: v for k, v in ups.items() if k in ALLOWED and v not in (None, "")}
            case = merge(case, ups)
        except (ValueError, TypeError):
            pass
    if images:
        print(f"[documents] {len(images)} image(s), {sum(len(i) for i in images) // 1024} KB, model reply: {raw[:300]!r}")
    d = evaluate(case)
    system = COMPOSE + f"Amounts are in {POLICY['currency']}; never ask about the currency. Unknown inputs are filled with typical values, so the estimate still runs: never refuse to give it when DECISION.request exists, just say which inputs were assumed.\n" + "JUST READ THIS TURN: " + json.dumps(read) + (" (a file was attached but nothing readable was found)" if images and not read else "") + "\n" + "CASE: " + json.dumps(case) + "\nDECISION: " + json.dumps(d)
    return {"reply": gemini([{"role": "system", "content": system}] + msgs[-10:]), "case": case, "decision": d, "read": read, "debug": {"images": len(images), "model_reply": raw[:300]} if images and not read else None}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, {"error": "not found"})
        try:
            self._send(200, chat(json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Open http://localhost:8000  (chat model: {MODEL}, PD model: {B['metrics']['model']}, test AUC {B['metrics']['auc']})")
    ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
