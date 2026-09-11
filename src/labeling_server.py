"""
Minimal local web labeler for the Golden Evaluation Set (stdlib only).

Zero new dependencies — runs on Python's built-in ``http.server`` and serves
a small single-page HTML+JS interface with native <select> dropdowns for
the intent and escalation labels.  Labels are saved incrementally to
``evaluation/golden_set.labels.json``; the original Golden Set JSON is
**never** modified.

Usage::

    python -m src.labeling_server            # open http://127.0.0.1:8000
    python -m src.labeling_server --port 9000
    python -m src.labeling_server --no-open   # print URL only

The server exposes a tiny JSON API (``/api/index``, ``/api/record/<id>``,
``/api/intents``, ``POST /api/export``) plus a single ``GET /`` page that
consumes it via plain ``fetch``.  No AI model is called at any point.
"""

import argparse
import json
import sys
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import labeling as L

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class LabelerApp:
    """Thin wrapper around ``src.labeling`` that holds the in-memory state."""

    def __init__(self, golden_path: str, labels_path: str,
                 config_path: str = None) -> None:
        self.golden_path = Path(golden_path)
        self.labels_path = Path(labels_path)
        self.payload = L.load_golden_set(golden_path)
        self.records = L.golden_records(self.payload)
        self.store = L.load_labels(labels_path)
        self.store["count_total"] = len(self.records)
        if not self.labels_path.exists():
            self.store["source"] = str(self.golden_path)
        self.intents = L.load_intent_names(config_path)
        self.lock = threading.Lock()

    # ---- read helpers ----------------------------------------------------

    def index(self) -> dict:
        """Return a lightweight index (no full records) for the sidebar."""
        with self.lock:
            table = self.store.get("labels", {})
            records = [
                {
                    "golden_id": r["golden_id"],
                    "conversation_id": r.get("conversation_id"),
                    "labeled": L.is_complete(table.get(r["golden_id"])),
                }
                for r in self.records
            ]
        fu_idx = L.first_unlabeled_index(self.records, self.store)
        return {
            "count_total": len(self.records),
            "count_labeled": L.count_labeled(self.store),
            "first_unlabeled": (
                self.records[fu_idx]["golden_id"] if fu_idx < len(self.records)
                else None
            ),
            "records": records,
        }

    def get_record(self, golden_id: str) -> dict | None:
        """Return full record content plus the current label (if any)."""
        with self.lock:
            rec = next(
                (r for r in self.records if r["golden_id"] == golden_id), None
            )
            if rec is None:
                return None
            entry = self.store.get("labels", {}).get(golden_id) or {}
            return {
                "golden_id": rec["golden_id"],
                "conversation_id": rec.get("conversation_id"),
                "customer_message": rec.get("customer_message", ""),
                "conversation_context": rec.get("conversation_context", ""),
                "microsoft_responses": rec.get("microsoft_responses", []) or [],
                "intent_label": entry.get(L.INTENT_FIELD, ""),
                "escalation_label": entry.get(L.ESCALATION_FIELD, ""),
                "notes": entry.get(L.NOTES_FIELD, ""),
            }

    # ---- write helpers ---------------------------------------------------

    def save(self, golden_id: str, body: dict) -> dict:
        """Validate + persist one label; raise ``ValueError`` on bad input."""
        rec = next(
            (r for r in self.records if r["golden_id"] == golden_id), None
        )
        if rec is None:
            raise ValueError(f"Unknown golden_id: {golden_id}")

        intent = (body.get("intent_label") or "").strip()
        escalation = (body.get("escalation_label") or "").strip()
        notes = (body.get("notes") or "").strip()

        errors = L.validate_label(intent, escalation, notes, self.intents)
        if errors:
            raise ValueError("; ".join(errors))

        with self.lock:
            entry = L.set_label(
                self.store, rec, intent, escalation, notes,
                valid_intents=self.intents,
            )
            L.save_labels(self.store, str(self.labels_path))
        fu_idx = L.first_unlabeled_index(self.records, self.store)
        return {
            "saved": entry,
            "count_labeled": L.count_labeled(self.store),
            "count_total": len(self.records),
            "first_unlabeled": (
                self.records[fu_idx]["golden_id"]
                if fu_idx < len(self.records) else None
            ),
        }

    def export(self, output_path: str = None) -> str:
        """Merge labels into a labeled copy next to the Golden Set JSON."""
        if output_path is None:
            output_path = str(self.golden_path.parent / "golden_set.labeled.json")
        with self.lock:
            dest = L.export_labeled_set(
                self.payload, self.store,
                output_path=output_path,
                labels_source=str(self.labels_path),
            )
        return str(dest)


# ---------------------------------------------------------------------------
# HTTP handler (thin shim — all logic lives in LabelerApp)
# ---------------------------------------------------------------------------

class LabelingHandler(BaseHTTPRequestHandler):
    """Serves the static page + JSON API.  ``app`` is injected via partial."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args, app=None, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    # No request-level logging to keep the console clean.
    def log_message(self, fmt, *args):  # noqa: D401
        pass

    # ---- helpers ---------------------------------------------------------

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, code: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> str:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length).decode("utf-8") if length else ""

    def _path(self) -> str:
        return urlparse(self.path).path

    # ---- GET -------------------------------------------------------------

    def do_GET(self):  # noqa: N802
        path = self._path()
        if path in ("/", "/index.html"):
            self._html(200, PAGE_HTML)
        elif path == "/api/index":
            self._json(200, self.app.index())
        elif path == "/api/intents":
            self._json(200, {"intents": self.app.intents})
        elif path.startswith("/api/record/"):
            gid = path[len("/api/record/"):]
            rec = self.app.get_record(gid)
            if rec is None:
                self._json(404, {"error": f"Unknown golden_id: {gid}"})
            else:
                self._json(200, rec)
        else:
            self._json(404, {"error": "Not found"})

    # ---- PUT -------------------------------------------------------------

    def do_PUT(self):  # noqa: N802
        path = self._path()
        if not path.startswith("/api/record/"):
            self._json(404, {"error": "Not found"})
            return
        gid = path[len("/api/record/"):]
        try:
            body = json.loads(self._read_body() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "Request body is not valid JSON."})
            return
        try:
            result = self.app.save(gid, body)
            self._json(200, result)
        except (ValueError, KeyError) as exc:
            self._json(400, {"error": str(exc), "golden_id": gid})

    # ---- POST -----------------------------------------------------------

    def do_POST(self):  # noqa: N802
        path = self._path()
        if path == "/api/export":
            try:
                dest = self.app.export()
                self._json(200, {"exported_to": dest})
            except Exception as exc:
                self._json(500, {"error": str(exc)})
        else:
            self._json(404, {"error": "Not found"})


# ---------------------------------------------------------------------------
# Embedded HTML + JS  (single page, no frameworks, no build step)
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Golden Set Labeler</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,-apple-system,sans-serif;background:#f5f6f8;
       color:#1a1a1a;display:flex;height:100vh}
  #sidebar{width:280px;min-width:220px;background:#fff;border-right:1px solid #d0d5dd;
           display:flex;flex-direction:column;overflow:hidden}
  #sidebar-header{padding:14px 16px;border-bottom:1px solid #d0d5dd;
                  font-size:14px;font-weight:600;background:#fafbfc}
  #progress{margin-top:4px;font-weight:400;color:#555;font-size:13px}
  #record-list{flex:1;overflow-y:auto;padding:4px 0}
  .record-item{padding:8px 16px;cursor:pointer;font-size:13px;
               display:flex;gap:6px;align-items:baseline}
  .record-item:hover{background:#eef2ff}
  .record-item.active{background:#dde5ff;font-weight:600}
  .record-item .id{min-width:100px}
  .record-item .badge{font-size:11px;padding:1px 5px;border-radius:3px;
                      white-space:nowrap}
  .badge.done{background:#d1fae5;color:#065f46}
  .badge.todo{background:#fee2e2;color:#991b1b}
  #main{flex:1;display:flex;flex-direction:column;overflow:hidden}
  #main-header{padding:14px 20px;border-bottom:1px solid #d0d5dd;background:#fafbfc;
               display:flex;justify-content:space-between;align-items:center}
  #main-header h2{font-size:15px;font-weight:600}
  #main-body{flex:1;overflow-y:auto;padding:20px 24px;display:flex;gap:24px}
  #record-panel{flex:2;min-width:0}
  #label-panel{flex:1;min-width:240px;max-width:340px}
  .section{margin-bottom:16px}
  .section-label{font-size:12px;font-weight:600;text-transform:uppercase;
                 color:#555;margin-bottom:4px;letter-spacing:.3px}
  pre{background:#fff;border:1px solid #d0d5dd;border-radius:6px;padding:10px 12px;
      white-space:pre-wrap;word-wrap:break-word;font-size:13px;
      line-height:1.5;max-height:240px;overflow-y:auto;margin:0}
  ol{padding-left:20px;font-size:13px;line-height:1.6}
  ol li{margin-bottom:4px}
  label{display:block;font-size:13px;font-weight:600;margin-bottom:4px}
  select,textarea{width:100%;padding:8px 10px;font-size:13px;
                  border:1px solid #aab0ba;border-radius:4px;background:#fff}
  select:focus,textarea:focus{outline:none;border-color:#4f6df5;
                              box-shadow:0 0 0 2px rgba(79,109,245,.25)}
  textarea{min-height:60px;resize:vertical}
  .btn{padding:8px 14px;font-size:13px;font-weight:600;border:none;
       border-radius:4px;cursor:pointer}
  .btn-primary{background:#4f6df5;color:#fff}
  .btn-primary:disabled{background:#9aa7d3;cursor:default}
  .btn-secondary{background:#e5e7eb;color:#1a1a1a}
  .btn-export{background:#f1f5f9;border:1px solid #aab0ba;font-weight:400}
  #controls{display:flex;gap:8px;margin-top:12px;align-items:center}
  #status-msg{font-size:13px;margin-top:10px;min-height:20px}
  .msg-ok{color:#065f46}
  .msg-err{color:#991b1b}
  #toolbar{display:flex;gap:6px;align-items:center}
  .hint{font-size:11px;color:#888;margin-top:12px}
  #all-done-banner{padding:10px 16px;background:#d1fae5;color:#065f46;
                   font-weight:600;font-size:13px;display:none;text-align:center}
</style>
</head>
<body>

<aside id="sidebar">
  <div id="sidebar-header">
    Golden Set Labeler
    <div id="progress"></div>
  </div>
  <div id="all-done-banner">All 200 records labeled. Export when ready.</div>
  <div id="record-list"></div>
</aside>

<div id="main">
  <div id="main-header">
    <h2 id="title">Select a record</h2>
    <div id="toolbar">
      <button class="btn btn-secondary" id="btn-prev">Prev</button>
      <button class="btn btn-secondary" id="btn-next">Next</button>
      <button class="btn btn-secondary" id="btn-next-unlabeled">Next unlabeled</button>
      <button class="btn btn-export" id="btn-export">Export labeled copy</button>
    </div>
  </div>
  <div id="main-body">
    <div id="record-panel">
      <div class="section">
        <div class="section-label">Customer message</div>
        <pre id="customer-message">(none)</pre>
      </div>
      <div class="section">
        <div class="section-label">Conversation context</div>
        <pre id="conversation-context">(none)</pre>
      </div>
      <div class="section">
        <div class="section-label">Microsoft historical responses</div>
        <ol id="microsoft-responses"></ol>
      </div>
    </div>
    <div id="label-panel">
      <div class="section">
        <label for="intent">Intent</label>
        <select id="intent"><option value="">-- select --</option></select>
      </div>
      <div class="section">
        <label for="escalation">Escalation</label>
        <select id="escalation">
          <option value="">-- select --</option>
          <option value="AUTO_HANDLE">AUTO_HANDLE</option>
          <option value="ESCALATE_TO_HUMAN">ESCALATE_TO_HUMAN</option>
        </select>
      </div>
      <div class="section">
        <label for="notes">Notes (optional)</label>
        <textarea id="notes" placeholder="Optional judgment note ..."></textarea>
      </div>
      <div id="controls">
        <button class="btn btn-primary" id="btn-save">Save</button>
      </div>
      <div id="status-msg"></div>
      <div class="hint">
        No AI model is invoked.<br>
        All labels are entered by you (human ground truth).
      </div>
    </div>
  </div>
</div>

<script>
"use strict";

const $ = id => document.getElementById(id);

const state = { records: [], idx: -1 };

async function api(method, url, body){
  const opts = { method, headers: {"Content-Type":"application/json"} };
  if(body != null) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  const data = await res.json().catch(()=>({}));
  if(!res.ok){
    const err = new Error(data.error || ("HTTP " + res.status));
    err.data = data;
    throw err;
  }
  return data;
}

function renderProgress(n, total){
  $("progress").textContent = `${n} / ${total} labeled`;
  $("all-done-banner").style.display = (n >= total && total > 0) ? "block" : "none";
}

function renderList(){
  const el = $("record-list");
  el.innerHTML = "";
  state.records.forEach((r, i) => {
    const div = document.createElement("div");
    div.className = "record-item" + (i === state.idx ? " active" : "");
    const badgeClass = r.labeled ? "done" : "todo";
    const badgeText  = r.labeled ? "labeled" : "unlabeled";
    div.innerHTML =
      `<span class="id">${r.golden_id}</span>` +
      `<span class="badge ${badgeClass}">${badgeText}</span>`;
    div.addEventListener("click", () => showRecord(i));
    el.appendChild(div);
  });
}

async function showRecord(idx){
  if(idx < 0 || idx >= state.records.length) return;
  state.idx = idx;
  const r = state.records[idx];
  $("title").textContent = `${r.golden_id}  (conversation ${r.conversation_id})`;
  renderList();

  const rec = await api("GET", "/api/record/" + encodeURIComponent(r.golden_id));
  $("customer-message").textContent = rec.customer_message || "(none)";
  $("conversation-context").textContent = rec.conversation_context || "(none)";

  const ul = $("microsoft-responses");
  ul.innerHTML = "";
  (rec.microsoft_responses || []).forEach(t => {
    const li = document.createElement("li");
    li.textContent = t;
    ul.appendChild(li);
  });
  if(!rec.microsoft_responses || !rec.microsoft_responses.length){
    const li = document.createElement("li");
    li.textContent = "(none)";
    ul.appendChild(li);
  }

  $("intent").value     = rec.intent_label     || "";
  $("escalation").value = rec.escalation_label || "";
  $("notes").value      = rec.notes            || "";
  $("status-msg").textContent = "";
  $("btn-save").disabled = false;
}

function nextUnlabeled(from){
  for(let i = from; i < state.records.length; i++)
    if(!state.records[i].labeled) return i;
  return -1;
}

function setMsg(text, ok){
  const el = $("status-msg");
  el.textContent = text;
  el.className = ok ? "msg-ok" : "msg-err";
}

$("btn-save").addEventListener("click", async () => {
  const intent     = $("intent").value;
  const escalation = $("escalation").value;
  const notes      = $("notes").value.trim();
  if(!intent){ setMsg("Please select an intent.", false); $("intent").focus(); return; }
  if(!escalation){ setMsg("Please select an escalation label.", false); $("escalation").focus(); return; }

  const gid = state.records[state.idx].golden_id;
  try {
    $("btn-save").disabled = true;
    const res = await api("PUT",
      "/api/record/" + encodeURIComponent(gid),
      { intent_label: intent, escalation_label: escalation, notes }
    );
    state.records[state.idx].labeled = true;
    renderProgress(res.count_labeled, res.count_total);
    renderList();
    setMsg(`Saved ${gid}  (${res.count_labeled} / ${res.count_total})`, true);

    const next = nextUnlabeled(state.idx + 1);
    if(next >= 0) showRecord(next);
  } catch(e){
    setMsg("Error: " + e.message, false);
    $("btn-save").disabled = false;
  }
});

$("btn-prev").addEventListener("click", () => {
  if(state.idx > 0) showRecord(state.idx - 1);
});
$("btn-next").addEventListener("click", () => {
  if(state.idx < state.records.length - 1) showRecord(state.idx + 1);
});
$("btn-next-unlabeled").addEventListener("click", () => {
  const n = nextUnlabeled(state.idx + 1);
  if(n >= 0) showRecord(n);
});

$("btn-export").addEventListener("click", async () => {
  try {
    const res = await api("POST", "/api/export");
    setMsg("Exported labeled copy to: " + res.exported_to, true);
  } catch(e){ setMsg("Export failed: " + e.message, false); }
});

(async function init(){
  const intents = await api("GET", "/api/intents");
  const sel = $("intent");
  (intents.intents || []).forEach(name => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  });

  const idx = await api("GET", "/api/index");
  state.records = idx.records;
  renderProgress(idx.count_labeled, idx.count_total);
  renderList();

  const target = idx.first_unlabeled
    || (state.records[0] && state.records[0].golden_id);
  if(target){
    const start = state.records.findIndex(r => r.golden_id === target);
    if(start >= 0) showRecord(start);
  }
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _find_free_port(host: str, port: int) -> int:
    """Return *port* if free, else the next available port (up to +50)."""
    for offset in range(51):
        candidate = port + offset
        try:
            with HTTPServer((host, candidate), lambda *a, **kw: None) as probe:
                pass
            return candidate
        except OSError:
            continue
    raise RuntimeError(f"No free port found near {port}")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--golden-path", default=str(L.DEFAULT_GOLDEN_SET_PATH))
    parser.add_argument("--labels-path", default=str(L.DEFAULT_LABELS_PATH))
    parser.add_argument("--no-open", action="store_true",
                        help="Do not auto-open a browser")
    args = parser.parse_args(argv)

    app = LabelerApp(args.golden_path, args.labels_path)
    free_port = _find_free_port(args.host, args.port)

    handler_factory = partial(LabelingHandler, app=app)
    server = HTTPServer((args.host, free_port), handler_factory)

    url = f"http://{args.host}:{free_port}"
    fu = L.first_unlabeled_index(app.records, app.store)
    print(f"Golden Set Labeler listening on {url}")
    print(
        f"Progress: {L.count_labeled(app.store)} / "
        f"{len(app.records)} labeled"
    )
    if fu < len(app.records):
        rec = app.records[fu]
        print(f"Resume : {rec['golden_id']} (record {fu + 1})")
    else:
        print("All records are already labeled.")
    print("Press Ctrl+C to stop.\n")

    if not args.no_open:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
