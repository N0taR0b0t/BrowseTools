from flask import Flask, jsonify, render_template_string, request
import json, os
import logging

app = Flask(__name__)
LOG_FILE = "llm_master_log.json"

# Suppress /latest route log spam in terminal
log = logging.getLogger('werkzeug')
class FilterOutLatest(logging.Filter):
    def filter(self, record):
        return "/latest" not in record.getMessage()
log.addFilter(FilterOutLatest())

# ------------------ Utility ------------------ #
def latest_entry():
    """Return the last well-formed JSON line in LOG_FILE or None."""
    if not os.path.exists(LOG_FILE):
        return None
    with open(LOG_FILE, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos, buf = f.tell() - 1, b""
        while pos >= 0:
            f.seek(pos)
            char = f.read(1)
            if char == b"\n" and buf:
                break
            buf = char + buf
            pos -= 1
    try:
        return json.loads(buf.decode())
    except json.JSONDecodeError:
        return None

# ------------------ Routes ------------------ #
@app.route("/latest")
def latest_json():
    """Serve latest log entry as JSON (used by frontend polling)."""
    entry = latest_entry()
    return jsonify(entry or {})

@app.route("/")
def index():
    """Serve main HTML page with auto-refresh and pretty formatting."""
    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LLM Latest Log</title>
<style>
    body       { font-family: Inter, Arial, sans-serif; margin: 2rem; color:#222; }
    h2, h3     { margin: .2rem 0; }
    .meta      { margin-bottom: 1rem; }
    .role      { padding: .4rem .6rem; border-radius: .4rem; margin-bottom:.6rem; white-space:pre-wrap; }
    .system    { background:#e7f0ff; color:#0940c6; }
    .user      { background:#e8ffe7; color:#157812; }
    .assistant { background:#fff5e5; color:#b85d00; }
    .error     { color:#c00; }
</style>
</head>
<body>
    <h2>Latest LLM Log Entry</h2>
    <div id="content"><em>Loading…</em></div>

<script>
async function fetchAndRender(){
    try{
        const res = await fetch("/latest", {cache:"no-store"});
        const data = await res.json();
        document.getElementById("content").innerHTML = render(data);
    }catch(e){
        document.getElementById("content").innerHTML = "<p class='error'>Failed to load log.</p>";
    }
}
function render(log){
    if(!log || !log.timestamp) return "<p>No entry found.</p>";
    let html = `
      <div class="meta">
        <strong>Timestamp:</strong> ${log.timestamp}<br>
        <strong>Model:</strong> ${log.request.model}<br>
        <strong>Status:</strong> ${log.status}
      </div>
      <h3>Messages</h3>`;
    for(const m of log.request.messages){
        html += `<div class="role ${m.role}"><strong>${m.role}:</strong> ${m.content}</div>`;
    }
    html += `<h3>Assistant Action</h3>
             <div class="role assistant">${log.assistant}</div>`;
    return html;
}
setInterval(fetchAndRender, 2000);
fetchAndRender();
</script>
</body>
</html>"""
    return render_template_string(html)

# ------------------ Main ------------------ #
if __name__ == "__main__":
    app.run(debug=True, port=5000)