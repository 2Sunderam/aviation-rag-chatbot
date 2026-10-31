// FAA Handbooks RAG — chat UI with retrieval visualization (no framework).
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let sessionId = localStorage.getItem("rag_session") || crypto.randomUUID();
localStorage.setItem("rag_session", sessionId);
let busy = false;

// ---------- pipeline stats (ingestion visualization) ----------
async function loadStats() {
  const s = await (await fetch("/api/stats")).json();
  const sum = (k) => s.books.reduce((a, b) => a + b[k], 0);
  $("#models").textContent = `Embeddings: ${s.embedding_model} (local) · LLM: ${s.llm} (Inception, Groq backup) · Qdrant: ${s.collection}`;
  $("#pipeline").textContent = `${s.books.length} PDFs → ${sum("pages").toLocaleString()} pages → ${sum("chunks").toLocaleString()} chunks → ${(s.vectors.text + s.vectors.figure).toLocaleString()} vectors`;
  const steps = [
    ["PDFs", s.books.length], ["Pages", sum("pages")], ["OCR pages (RapidOCR)", sum("ocr_pages")],
    ["Chunks (800 tok, 15% overlap)", sum("chunks")], ["Figures extracted", sum("figures")],
    ["Text vectors (dense+BM25)", s.vectors.text], ["Figure vectors", s.vectors.figure],
  ];
  $("#flow").innerHTML = steps.map(([k, v], i) =>
    `${i ? '<span class="arrow">→</span>' : ""}<div class="step"><b>${Number(v).toLocaleString()}</b>${esc(k)}</div>`).join("");
  $("#books tbody").innerHTML = s.books.map((b) =>
    `<tr><td>${esc(b.title)}</td><td>${b.pages}</td><td>${b.ocr_pages}</td><td>${b.chunks}</td><td>${b.figures}</td></tr>`).join("");
}
$("#pipelineToggle").onclick = () => { $("#pipelinePanel").hidden = !$("#pipelinePanel").hidden; };

// ---------- answer rendering ----------
function renderAnswer(text, valid = null) {
  const lines = esc(text).split("\n");
  let html = "", inList = false;
  for (const raw of lines) {
    const line = raw.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    const li = line.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)/);
    if (li) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${li[1]}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    if (line.trim()) html += `<p>${line}</p>`;
  }
  if (inList) html += "</ul>";
  return html.replace(/\[(F?\d+)\]/g, (_, ref) =>
    `<button class="cite${valid && !valid.has(ref) ? " bad" : ""}" data-ref="${ref}">${ref}</button>`);
}

document.addEventListener("click", (e) => {
  const c = e.target.closest(".cite");
  if (!c) return;
  const el = document.querySelector(`[data-src="${c.dataset.ref}"]`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("cited"); setTimeout(() => el.classList.remove("cited"), 1600);
});

// ---------- retrieval visualization ----------
function kind(h) {
  if (h.dense_rank && h.bm25_rank) return ["both", "found by both"];
  if (h.dense_rank) return ["dense", "semantic only"];
  return ["bm25", "keyword only"];
}

function renderRetrieval(ev) {
  $("#retrievalEmpty").hidden = true;
  $("#query").innerHTML = `Planner → ${ev.queries.length} search quer${ev.queries.length > 1 ? "ies" : "y"}: ` +
    ev.queries.map((q, i) => `<i>Q${i + 1}: ${esc(q)}</i>`).join(" · ");
  const maxRrf = Math.max(...ev.chunks.map((h) => h.rrf_score), 1e-9);
  $("#chunkHead").hidden = !ev.chunks.length;
  $("#chunks").innerHTML = ev.chunks.map((h, i) => {
    const [cls, label] = kind(h), p = h.payload;
    return `<div class="card ${cls}" data-src="${i + 1}">
      <div class="top"><span class="cit">[${i + 1}] ${esc(h.citation)}</span><span>RRF ${h.rrf_score.toFixed(3)}</span></div>
      <div class="sec">${esc(p.section || "")}</div>
      <div class="bar"><span style="width:${(100 * h.rrf_score / maxRrf).toFixed(0)}%"></span></div>
      <div class="badges"><span class="badge">${label}</span>${h.query_idx != null && ev.queries.length > 1 ? `<span class="badge">Q${h.query_idx + 1}</span>` : ""}
        <span class="badge">dense ${h.dense_rank ? "#" + h.dense_rank + " · cos " + h.dense_score.toFixed(3) : "—"}</span>
        <span class="badge">BM25 ${h.bm25_rank ? "#" + h.bm25_rank + " · " + h.bm25_score.toFixed(2) : "—"}</span>
        ${p.ocr ? '<span class="badge">OCR</span>' : ""}</div>
      <div class="text">${esc(p.text)}</div>
      <button class="more" onclick="this.closest('.card').classList.toggle('open')">show more / less</button>
    </div>`;
  }).join("");
  $("#figHead").hidden = !ev.figures.length;
  $("#figures").innerHTML = ev.figures.map((h, i) => {
    const p = h.payload;
    return `<div class="fig" data-src="F${i + 1}" data-url="${esc(p.url)}" data-cap="${esc(p.caption || "")} — ${esc(h.citation)}">
      <img loading="lazy" src="${esc(p.url)}" alt="${esc(p.caption || "figure")}">
      <div class="cap"><b>[F${i + 1}]</b> ${esc(p.caption || "No caption")}</div>
      <div class="cit">${esc(h.citation)} · ${kind(h)[1]}</div></div>`;
  }).join("");
}

$("#figures").addEventListener("click", (e) => {
  const f = e.target.closest(".fig");
  if (!f) return;
  const d = $("#lightbox");
  d.querySelector("img").src = f.dataset.url;
  d.querySelector("p").textContent = f.dataset.cap;
  d.showModal();
});

function renderTimings(t) {
  const gen = Math.max((t.llm_total || 0) - (t.llm_first_token || 0), 0);
  const parts = [["t-embed", "embed", t.embed], ["t-search", "search", (t.search_text || 0) + (t.search_figures || 0)],
    ["t-rewrite", "planner", t.plan], ["t-ttft", "LLM 1st token", t.llm_first_token], ["t-gen", "generation", gen]]
    .filter((p) => p[2]);
  const total = parts.reduce((a, p) => a + p[2], 0) || 1;
  $("#timings").innerHTML = parts.map(([c, l, v]) =>
    `<div class="${c}" style="width:${(100 * v / total).toFixed(1)}%" title="${l}: ${v} ms">${v / total > 0.12 ? l : ""}</div>`).join("");
  $("#timings").insertAdjacentHTML("afterend", "");
  document.querySelector(".timing-text")?.remove();
  $("#timings").insertAdjacentHTML("afterend",
    `<div class="timing-text">${parts.map(([, l, v]) => `${l} ${v} ms`).join(" · ")} · <b>end-to-end ${(t.end_to_end / 1000).toFixed(2)} s</b></div>`);
}

// ---------- chat ----------
function addMsg(cls, html) {
  document.querySelector(".empty")?.remove();
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.innerHTML = html;
  $("#messages").appendChild(div);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return div;
}

async function send(text) {
  if (busy || !text.trim()) return;
  busy = true; $("#send").disabled = true;
  addMsg("user", esc(text));
  const bot = addMsg("bot typing", "");
  let answer = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
        if (!line.startsWith("data:")) continue;
        const ev = JSON.parse(line.slice(5));
        if (ev.type === "retrieval") renderRetrieval(ev);
        else if (ev.type === "token") { answer += ev.text; bot.innerHTML = renderAnswer(answer); }
        else if (ev.type === "done") {
          bot.classList.remove("typing");
          bot.innerHTML = renderAnswer(answer, new Set(ev.sources.map((s) => s.ref)));
          if (ev.sources.length) bot.insertAdjacentHTML("beforeend",
            `<div class="sources"><b>Sources:</b> ${ev.sources.map((s) => `[${s.ref}] ${esc(s.citation)}`).join(" · ")}</div>`);
          bot.insertAdjacentHTML("beforeend", `<div class="meta">${(ev.timings_ms.end_to_end / 1000).toFixed(2)} s end-to-end · ${esc(ev.model || "")}</div>`);
          renderTimings(ev.timings_ms);
        } else if (ev.type === "error") { bot.classList.remove("typing"); bot.textContent = "Error: " + ev.message; }
        $("#messages").scrollTop = $("#messages").scrollHeight;
      }
    }
  } catch (err) {
    bot.classList.remove("typing"); bot.textContent = "Request failed: " + err;
  } finally {
    busy = false; $("#send").disabled = false; $("#input").focus();
  }
}

$("#form").onsubmit = (e) => { e.preventDefault(); const t = $("#input").value; $("#input").value = ""; send(t); };
$("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#form").requestSubmit(); } });
$("#messages").addEventListener("click", (e) => { const b = e.target.closest(".examples button"); if (b) send(b.textContent); });
$("#reset").onclick = async () => {
  await fetch(`/api/reset/${sessionId}`, { method: "POST" });
  sessionId = crypto.randomUUID(); localStorage.setItem("rag_session", sessionId);
  $("#messages").innerHTML = ""; addMsg("bot", "<p>New conversation started. Memory cleared.</p>");
};

loadStats().catch(() => { $("#models").textContent = "stats unavailable"; });
