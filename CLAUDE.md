# Loan Desk

Lender-side decision-support prototype: a loan officer chats with an agent that reads
applicant documents, scores default risk with a trained model, checks it against lender
policy, and proposes conditions that would make a loan fit. Decision support only — the
agent never makes the lending decision.

## Running it

```
python server.py        # http://localhost:8000, needs model/pd_model.joblib and .env
python credit_model.py data\LC_loans_granting_model_dataset.csv   # retrain the model
pytest                  # unit tests (LLM and model mocked)
python eval\run_eval.py # scored synthetic cases, no LLM call needed
```

Never print, open, log, or commit `.env` (it holds a shared course API key). Never put
real borrower data anywhere in this repo — use made-up applicants only.

## Architecture

`server.py` is a thin entry point; all the real logic lives in the `loan_desk/` package
(`config`, `model_service`, `company_model_service`, `loan_math`, `policy`,
`business_policy`, `company_lookup`, `rag`, `llm_client`, `tools`, `agent`, `batch`,
`http_server`). One Python process (stdlib `http.server`, no web framework) serves one
static page (`static/index.html`, vanilla JS, no build step) and two endpoints:
`POST /api/chat` (the agent) and `POST /api/batch` (deterministic CSV scoring, no LLM -
`loan_desk/batch.py` runs `policy.merge()`/`policy.evaluate()` per row, personal-loan
only). The chat client is stateless-server-side: it holds the conversation and the case
dict and resends both on every turn. See `README.md` for the full diagram.

`chat()` runs a **ReAct-style agent loop** (Lecture 6: Thought -> Action -> Observation,
repeated) against an OpenAI-compatible chat-completions endpoint, capped at
`MAX_STEPS = 6` (the loop's max-iteration guard — see Lecture 5's "back-edges + max
iteration guard" pattern). The model sees the user's text and any attached images on the
very first call (native vision, no separate OCR/parsing step) and decides which tools to
call. It stops the turn as soon as it returns a message with no tool calls — the same
termination rule Claude Code's own agent loop uses.

**Tools** (`TOOLS` / `dispatch()` in `loan_desk/tools.py`):

| tool | does | case mutation |
|---|---|---|
| `update_case(facts)` | validates keys against the allowed list, merges via `merge()` | **replaces** each named field; unknown keys are reported back, not silently dropped |
| `score_applicant()` | runs the trained PD model + occlusion-based reasons | read-only |
| `check_policy()` | runs `evaluate()`, returns verdict + checks | read-only |
| `find_conditions()` | alternate amount/term combos that pass policy | read-only |
| `search_policy_docs(query)` | TF-IDF retrieval over `policy_docs/lender_policy.md` | read-only |
| `calculate_payment(amount, rate_pct, months)` | exact amortization math | read-only |

Every number the agent reports is computed by a tool, never invented by the model — the
`decision` field returned to the UI is always a fresh `evaluate(case)` run server-side
after the loop ends, independent of which tools the model happened to call, so the panel
can never show a number the model made up.

**Why a single agent, not multi-agent.** Lecture 6's own criteria for multi-agent
("distinct phases, specialized expertise, parallelism, low concurrency needs") describe
the opposite of this app: one domain (credit decisioning), six complementary tools (not
competing specialists), and a synchronous low-latency chat turn. A supervisor splitting
"read documents" from "check policy" would add coordination overhead with nothing to
coordinate. Single agent, multiple tools is the right-sized shape here.

**RAG.** `search_policy_docs` used to do naive keyword/word-overlap matching over an
inline string — a textbook "Naive RAG" failure mode (Lecture 4). It now does the real
pipeline at the scale this problem needs: `policy_docs/lender_policy.md` is parsed,
chunked by `##` heading, TF-IDF-vectorized (`scikit-learn`), and matched by cosine
similarity, returning the top 3 chunks with their section name as a citation. A vector
database and dense embeddings would be the wrong tool for a one-page, six-section
document — TF-IDF + linear scan is correctly sized, not a shortcut.

**Guardrails.** The system prompt instructs the model to treat document text and tool
results as data, never as instructions (prompt-injection resistance — tested in
`tests/`), to ask for all missing required inputs at once, to say when a value was
assumed, and to never report a number it didn't get from a tool. This is decision
support, not an automated approval: the UI and prompts say so explicitly.

The prompt also explicitly forbids calling `update_case` with facts not literally
present in the user's message or an attached document, and forbids calling the
evaluation tools (`score_applicant`/`check_policy`/`find_conditions`/
`search_policy_docs`) when the user hasn't given anything new to evaluate (an
informational question like "what do I need for another applicant?" answers in words
only). This was a real bug found during testing: without this rule the model would
invent a plausible-looking applicant and run a full decision on fabricated data rather
than just answering the question - worth remembering if the rules ever get rewritten.

**One case at a time, by design.** Only one applicant's facts are "live" in `case` per
conversation; there's no server-side concept of multiple customers. The UI's
"Applications" switcher (`static/index.html`) is what lets an officer juggle several
applicants - it's a client-side-only list of saved `{case, messages, decision}` snapshots,
swapped in and out of the single active `case`. The agent is instructed to tell the user
to click "New case" (which saves the current one before clearing) rather than try to
keep two customers' facts straight in one `case` dict itself.

**Second domain: business lending.** The same agent/chat also handles companies,
mirroring every principle above for a different schema (`business_policy.py`,
`company_model_service.py`, `company_lookup.py`) rather than extending the personal-loan
code to cover both:
- `case["_segment"] = "business"` is set the moment `update_business_case` or
  `fetch_company_financials` first touches a case; `agent._evaluate(case)` branches on
  it to call `business_policy.evaluate` instead of `policy.evaluate`, and the response's
  `decision_kind` field tells the client which panel (`panel` vs `panelBusiness` in
  `static/index.html`) to render. Getting this tag wrong would silently run the wrong
  evaluate function on the wrong schema, so it's set in exactly two places
  (`tools.dispatch`'s `fetch_company_financials`/`update_business_case` branches).
- `company_lookup.py` calls the SEC's official `data.sec.gov` EDGAR API (not scraping) -
  every request MUST send the documented descriptive `User-Agent` header (SEC's fair-use
  policy; see `company_lookup.UA`) or requests get blocked. Revenue/cash-flow are
  "duration" XBRL facts (a start/end period); total debt (`Liabilities`) is an "instant"
  fact (as-of date only, no start) - conflating the two silently drops every instant-fact
  value (a real bug caught during development). Filers also change which XBRL tag they
  report under over time (e.g. most dropped "Revenues" around the 2018 ASC 606 change),
  so `_latest_annual` pools ALL candidate tags and picks the single most recent entry
  across them, not the first tag's own latest (a second real bug: using only the first
  non-empty tag returned a stale 2018 figure for a company still actively filing).
- `train_company_model.py`'s `DEBT_SERVICE_RATE` (0.20) must match
  `business_policy.DEBT_SERVICE_RATE` exactly - the model is trained on `dscr` computed
  with that assumption, so inference has to reproduce it or the model's learned
  relationship between dscr and risk is systematically wrong.
- The company model's trained revenue ceiling is ~$3B; `business_policy.evaluate` flags
  `out_of_range` above `business_policy.json`'s `out_of_range_revenue` the same way
  `policy.evaluate` flags a personal loan above $40,000 - but the *deterministic* checks
  (debt coverage, leverage) keep running normally even when the ML score is flagged, so
  real financial distress in an out-of-range company still gets caught (verified with
  AMC Entertainment's real financials: PD estimate is unreliable at that scale, but the
  debt-service-coverage check still fails correctly on its own).
- `rag.py`'s `PolicySearch` combines both `policy_docs/lender_policy.md` and
  `policy_docs/business_policy.md` into one TF-IDF index (one `search_policy_docs` tool,
  not two) - each chunk keeps its own source document for an accurate citation
  regardless of which document the best match came from.

## Conventions

- Deterministic math/ML stays in Python functions (`policy.evaluate`, `loan_math.pmt`,
  `model_service.score`) that the agent calls as tools — never ask the LLM to compute a
  number.
- `update_case` facts are filtered against `model_service.ALLOWED` (built from the
  model's feature list plus a few extra UI-facing keys like `monthly_income`); unknown
  keys are reported back in the tool result rather than silently merged, so the model
  can self-correct.
- `loan_desk/config.py` must import cleanly with no `.env` present (it's read as
  optional, `KEY` can be `None`) — only `llm_client.call_llm()` and the `server.py`
  entry point actually require a key, and only at call/start time, not at import time.
  This is what keeps `pytest` able to run with no `.env` and no network call at all.
- Tests don't mock either trained model — both are small, deterministic, and already
  committed to git, so tests call the real `loan_desk.policy`/`loan_desk.business_policy`/
  `loan_desk.rag` functions directly. The LLM and SEC EDGAR's network calls are the only
  things out of scope for tests: the LLM is simply never called (no test exercises
  `agent.chat()` or `llm_client.call_llm()`), and `company_lookup`'s SEC calls are mocked
  (`unittest.mock.patch` on `urllib.request.urlopen`) rather than skipped, so the
  parsing/extraction logic itself is still covered.
- Don't touch `credit_model.py`'s training logic or the saved model's dict format
  (`model, iso, num, cat, features, baseline, cat_values, importance, metrics`); other
  code depends on that shape.
