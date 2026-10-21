const ORIGIN = Deno.env.get("ALLOWED_ORIGIN") ?? "*";
const MODEL = Deno.env.get("MODEL") ?? "gemini-flash-lite-latest";
const cors = {
  "Access-Control-Allow-Origin": ORIGIN,
  "Access-Control-Allow-Headers": "content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

const EXTRACT_PROMPT = `You read personal loan documents (offers, contracts, schedules) in any language, including Georgian.
Return ONLY a JSON object with these keys. Use null if the document does not say. Never guess.
lender, currency, loan_amount, annual_interest_rate_pct, term_months, monthly_payment,
upfront_fees (total one-time fees as a number), early_repayment_penalty (short text),
late_payment_penalty (short text), collateral_or_guarantor (short text), notes (short text, anything unusual).
Numbers must be plain numbers with no symbols.`;

const CHAT_RULES = `You help a person understand their personal loan.
The facts below come from the document (reading errors are possible) and from exact calculations.
Use only these facts. If something is missing, say you don't know. Point out risks plainly
(high all-in rate, fees, penalties, payment mismatch). Reply in the user's language, short and clear.
This is information, not financial advice.
FACTS: `;

async function gemini(messages: unknown[]): Promise<string> {
  const r = await fetch("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${Deno.env.get("GEMINI_API_KEY")}` },
    body: JSON.stringify({ model: MODEL, messages }),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j?.[0]?.error?.message ?? j?.error?.message ?? "Gemini error " + r.status);
  return j.choices[0].message.content as string;
}

Deno.serve(async (req) => {
  const json = (b: unknown, s = 200) =>
    new Response(JSON.stringify(b), { status: s, headers: { ...cors, "Content-Type": "application/json" } });
  const origin = req.headers.get("origin");
  if (ORIGIN !== "*" && origin && origin !== ORIGIN) return json({ error: "forbidden" }, 403);
  if (req.method === "OPTIONS") return new Response("ok", { headers: cors });
  try {
    const { mode, images, facts, messages } = await req.json();
    if (mode === "extract") {
      if (!Array.isArray(images) || images.length === 0 || images.length > 3) return json({ error: "send 1 to 3 images" }, 400);
      const content = [{ type: "text", text: EXTRACT_PROMPT }, ...images.map((url: string) => ({ type: "image_url", image_url: { url } }))];
      const text = await gemini([{ role: "user", content }]);
      return json({ data: JSON.parse(text.match(/\{[\s\S]*\}/)![0]) });
    }
    if (mode === "chat") {
      return json({ reply: await gemini([{ role: "system", content: CHAT_RULES + JSON.stringify(facts) }, ...messages]) });
    }
    return json({ error: "bad mode" }, 400);
  } catch (e) {
    return json({ error: String((e as Error).message ?? e) }, 500);
  }
});
