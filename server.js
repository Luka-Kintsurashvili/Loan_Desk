const http = require("http"), fs = require("fs"), path = require("path");

try {
  for (const l of fs.readFileSync(path.join(__dirname, ".env"), "utf8").split(/\r?\n/)) {
    const m = l.match(/^\s*([A-Z_]+)\s*=\s*(.*?)\s*$/);
    if (m && !(m[1] in process.env)) process.env[m[1]] = m[2];
  }
} catch { /* no .env file */ }
const KEY = process.env.GEMINI_API_KEY, MODEL = process.env.MODEL || "gemini-flash-lite-latest";
if (!KEY) { console.error("Put your key in .env like this: GEMINI_API_KEY=..."); process.exit(1); }

// prompts live in one place: the Supabase function file
const src = fs.readFileSync(path.join(__dirname, "supabase/functions/loan-ai/index.ts"), "utf8");
const prompt = n => src.match(new RegExp(n + " = `([^`]*)`"))[1];
const EXTRACT_PROMPT = prompt("EXTRACT_PROMPT"), CHAT_RULES = prompt("CHAT_RULES");

async function gemini(messages) {
  const r = await fetch("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer " + KEY },
    body: JSON.stringify({ model: MODEL, messages }),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j?.[0]?.error?.message ?? j?.error?.message ?? "Gemini error " + r.status);
  return j.choices[0].message.content;
}

http.createServer(async (req, res) => {
  if (req.method === "GET" && (req.url === "/" || req.url === "/index.html")) {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    return res.end(fs.readFileSync(path.join(__dirname, "index.html")));
  }
  if (req.method === "POST" && req.url === "/api") {
    let body = ""; for await (const c of req) body += c;
    const send = (s, o) => { res.writeHead(s, { "Content-Type": "application/json" }); res.end(JSON.stringify(o)); };
    try {
      const { mode, images, facts, messages } = JSON.parse(body);
      if (mode === "extract") {
        if (!Array.isArray(images) || !images.length || images.length > 3) return send(400, { error: "send 1 to 3 images" });
        const content = [{ type: "text", text: EXTRACT_PROMPT }, ...images.map(url => ({ type: "image_url", image_url: { url } }))];
        const text = await gemini([{ role: "user", content }]);
        return send(200, { data: JSON.parse(text.match(/\{[\s\S]*\}/)[0]) });
      }
      if (mode === "chat") return send(200, { reply: await gemini([{ role: "system", content: CHAT_RULES + JSON.stringify(facts) }, ...messages]) });
      return send(400, { error: "bad mode" });
    } catch (e) { return send(500, { error: String(e.message || e) }); }
  }
  res.writeHead(404); res.end("not found");
}).listen(8000, "127.0.0.1", () => console.log("Open http://localhost:8000  (model: " + MODEL + ")"));
