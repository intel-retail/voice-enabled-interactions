from __future__ import annotations

import io
import os
import time
import wave
from typing import Any, Generator

import gradio as gr
import httpx
import numpy as np

from kiosk_core import config as kiosk_config

# ── Config ────────────────────────────────────────────────────────────────────
KIOSK_CORE_URL          = os.getenv("KIOSK_CORE_UI_BASE_URL",           "http://127.0.0.1:8012")
RAG_URL                 = os.getenv("KIOSK_CORE_UI_RAG_URL",            "http://127.0.0.1:8020/api/v1/query")
TTS_URL                 = os.getenv("KIOSK_CORE_UI_TTS_URL",            "http://127.0.0.1:8011/v1/audio/speech")
ANALYZER_URL            = os.getenv("KIOSK_CORE_UI_ANALYZER_URL",       "http://127.0.0.1:8010/v1/audio/transcriptions")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_UI_TIMEOUT_SECONDS",       "120.0"))
POLL_INTERVAL_SECONDS   = float(os.getenv("KIOSK_CORE_UI_POLL_INTERVAL_SECONDS", "0.35"))
_CHUNK_SECONDS          = kiosk_config.DEFAULT_CHUNK_SECONDS

# ── CSS (only targets our own divs + minimal Gradio overrides) ────────────────
STYLE = """
/*
 * Intel Light Theme
 *   bg-base  #FFFFFF   page — clean white
 *   bg-1     #F4F7FB   chat pane — very light blue-grey
 *   bg-2     #EBF2FA   assistant bubble — soft Intel blue tint
 *   border   #C8D8EA   Intel blue-grey border
 *   user-bub #0068B5   Intel Blue
 *   text-hi  #1A1A1A   near-black body text
 *   text-md  #4A6070   Intel blue-grey secondary
 *   text-lo  #8FA0AE   muted
 *   accent   #0068B5   Intel Blue
 */

/* Page */
.gradio-container { background: #FFFFFF !important; }
footer { display: none !important; }

/* Layout */
.kiosk-row   { width: 100% !important; align-items: flex-start !important; gap: 16px !important; }
.kiosk-left  { min-width: 0 !important; }
.kiosk-right { width: 320px !important; min-width: 260px !important; max-width: 340px !important; flex-shrink: 0 !important; }

/* Chat pane */
.chat-pane {
    height: 420px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 14px 10px;
    background: #F4F7FB;
    border-radius: 12px;
    border: 1px solid #C8D8EA;
    scroll-behavior: smooth;
}
.chat-pane::-webkit-scrollbar { width: 4px; }
.chat-pane::-webkit-scrollbar-thumb { background: #C8D8EA; border-radius: 4px; }
.chat-empty { margin: auto; color: #8FA0AE; font-size: 0.85rem; font-style: italic; }

/* Bubbles */
.msg-row { display: flex; align-items: flex-end; gap: 8px; }
.msg-row.user { flex-direction: row-reverse; }

.bubble {
    max-width: 75%;
    padding: 10px 15px;
    border-radius: 18px;
    font-size: 0.93rem;
    line-height: 1.6;
    word-break: break-word;
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
}
.msg-row.user .bubble {
    background: #0068B5;
    color: #FFFFFF;
    border-bottom-right-radius: 4px;
}
.msg-row.asst .bubble {
    background: #EBF2FA;
    color: #1A1A1A;
    border: 1px solid #C8D8EA;
    border-bottom-left-radius: 4px;
}
.bubble.partial { opacity: 0.65; }
.cursor {
    display: inline-block;
    color: #0068B5;
    animation: blink 0.9s step-end infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* Status */
.status-line {
    text-align: center;
    color: #4A6070;
    font-size: 0.78rem;
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
    min-height: 20px;
}

/* Headings */
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container .prose h1, .gradio-container .prose h2 {
    color: #1A1A1A !important;
}

/* Hide Gradio labels on the mic audio widget */
#kiosk-mic > .block > label { display: none !important; }

/* Gradio component shells — white with Intel blue-grey border */
.gradio-container .block,
.gradio-container .wrap,
.gradio-container .form,
.gradio-container fieldset,
.gradio-container .panel {
    background: #FFFFFF !important;
    border-color: #C8D8EA !important;
}
.gradio-container .waveform-container,
.gradio-container .recording-container,
.gradio-container .controls {
    background: #F4F7FB !important;
    color: #1A1A1A !important;
}
.gradio-container .waveform-container button,
.gradio-container .controls button,
.gradio-container .recording-container button {
    background: #EBF2FA !important;
    color: #0068B5 !important;
    border-color: #C8D8EA !important;
}
.gradio-container details,
.gradio-container details > summary {
    background: #FFFFFF !important;
    border-color: #C8D8EA !important;
    color: #1A1A1A !important;
}
.gradio-container details[open] > div {
    background: #F4F7FB !important;
    border-color: #C8D8EA !important;
}
.gradio-container label span,
.gradio-container .label-wrap span {
    color: #4A6070 !important;
}

/* ── KPI panel ── */
.kpi-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 4px 0;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #C8D8EA;
    border-left: 3px solid #0068B5;
    border-radius: 10px;
    padding: 12px 14px;
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
}
.kpi-card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0068B5;
    margin-bottom: 8px;
}
.kpi-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 3px 0;
    border-bottom: 1px solid #EBF2FA;
}
.kpi-row:last-child { border-bottom: none; }
.kpi-key {
    font-size: 0.75rem;
    color: #4A6070;
    white-space: nowrap;
    margin-right: 8px;
}
.kpi-val {
    font-size: 0.78rem;
    color: #1A1A1A;
    font-weight: 500;
    text-align: right;
    word-break: break-all;
}
.kpi-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 1px 7px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-green  { background: #D4F5E5; color: #0A6640; }
.badge-blue   { background: #D0E8F8; color: #004E8C; }
.badge-purple { background: #E8D8F8; color: #5E2D9E; }
"""

# ── Chat HTML helpers ─────────────────────────────────────────────────────────
def _esc(t: str) -> str:
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")

def _render_chat(history: list[dict], partial_user: str = "", partial_asst: str = "") -> str:
    rows: list[str] = []
    for msg in history:
        cls = "user" if msg["role"] == "user" else "asst"
        rows.append(f'<div class="msg-row {cls}"><div class="bubble">{_esc(msg["text"])}</div></div>')
    if partial_user:
        rows.append(f'<div class="msg-row user"><div class="bubble partial">{_esc(partial_user)}</div></div>')
    if partial_asst:
        rows.append(
            f'<div class="msg-row asst"><div class="bubble partial">'
            f'{_esc(partial_asst)}<span class="cursor">▌</span></div></div>'
        )
    inner = "\n".join(rows) if rows else '<div class="chat-empty">Tap 🎤 and ask a question</div>'
    scroll_js = "<script>setTimeout(()=>{var p=document.querySelector('.chat-pane');if(p)p.scrollTop=p.scrollHeight;},40);</script>"
    return f'<div class="chat-pane">{inner}</div>{scroll_js}'

# ── API helpers ───────────────────────────────────────────────────────────────
def _numpy_to_wav(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()

def _open_session(sr: int) -> dict[str, Any]:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as c:
        r = c.post(f"{KIOSK_CORE_URL}/api/v1/sessions/start-stream", json={
            "sample_rate": sr,
            "chunk_seconds": kiosk_config.DEFAULT_CHUNK_SECONDS,  # 5.0s
            "silence_timeout_seconds": 2.0,
            "max_session_seconds": 60.0,
            "silence_threshold": 900,
            "language": "en", "temperature": 0.0,
            "analyzer_url": ANALYZER_URL, "rag_url": RAG_URL, "tts_url": TTS_URL,
            "tts_model": "qwen-tts", "tts_language": "English",
        })
    r.raise_for_status(); return r.json()

def _push(sid: str, wav: bytes) -> None:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as c:
        c.post(f"{KIOSK_CORE_URL}/api/v1/sessions/{sid}/audio",
               content=wav, headers={"Content-Type": "audio/wav"}).raise_for_status()

def _eos(sid: str) -> None:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as c:
        c.post(f"{KIOSK_CORE_URL}/api/v1/sessions/{sid}/audio/end").raise_for_status()

def _poll(sid: str) -> dict[str, Any]:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as c:
        r = c.get(f"{KIOSK_CORE_URL}/api/v1/sessions/{sid}")
    r.raise_for_status(); return r.json()

def _latest_audio(session: dict, prev: int) -> tuple:
    segs = session.get("tts_audio_segments") or []
    if len(segs) > prev:
        return gr.update(value=segs[-1]["audio_file"], autoplay=True), len(segs)
    return gr.skip(), prev

# ── State ─────────────────────────────────────────────────────────────────────
_INIT: dict = {"session_id": None, "buffer": [], "sample_rate": 16000, "history": []}

# ── Handlers ──────────────────────────────────────────────────────────────────
def on_start(state: dict):
    s = dict(state); s["session_id"] = None; s["buffer"] = []
    return s, _render_chat(s["history"], partial_user="🎤  Listening…"), gr.skip(), "🎙  Listening — speak now"

def on_chunk(state: dict, chunk):
    if chunk is None:
        return state, gr.skip(), gr.skip(), gr.skip()
    sr, data = chunk
    if data is None or len(data) == 0:
        return state, gr.skip(), gr.skip(), gr.skip()
    if data.ndim > 1: data = data[:, 0]
    data = data.astype(np.int16)

    s = dict(state); s["sample_rate"] = sr
    s["buffer"] = list(s.get("buffer", [])) + [data]

    if s["session_id"] is None:
        try:
            s["session_id"] = _open_session(sr)["session_id"]
        except Exception as e:
            return s, gr.skip(), gr.skip(), f"❌ {e}"

    total = sum(len(b) for b in s["buffer"])
    if total >= int(sr * _CHUNK_SECONDS):
        audio = np.concatenate(s["buffer"])
        s["buffer"] = []
        try: _push(s["session_id"], _numpy_to_wav(audio, sr))
        except Exception: pass

    transcript = ""
    try:
        transcript = str(_poll(s["session_id"]).get("transcript", "")).strip()
    except Exception: pass

    partial = transcript or "🎤  Listening…"
    return s, _render_chat(s["history"], partial_user=partial), gr.skip(), "🎙  Listening — speak now"

def on_stop(state: dict) -> Generator:
    s = dict(state)
    sid = s.get("session_id"); sr = s.get("sample_rate", 16000)
    history = list(s.get("history", []))

    if not sid:
        yield s, _render_chat(history), gr.update(value=None), "No audio — try again"
        return

    remaining = s.get("buffer", [])
    if remaining:
        try: _push(sid, _numpy_to_wav(np.concatenate(remaining), sr))
        except Exception: pass

    try: _eos(sid)
    except Exception as e:
        yield s, _render_chat(history), gr.update(value=None), f"❌ {e}"; return

    yield s, _render_chat(history, partial_user="⏳  Processing…"), gr.update(value=None), "⏳  Processing…"

    prev_audio = 0
    while True:
        try: session = _poll(sid)
        except Exception as e:
            yield s, _render_chat(history), gr.update(value=None), f"❌ {e}"; return

        transcript    = str(session.get("transcript","")).strip()
        response_text = str(session.get("response","")).strip()
        audio_upd, prev_audio = _latest_audio(session, prev_audio)
        running = session.get("status","") in {"running","stopping"}

        n = len(session.get("tts_audio_segments") or [])
        if n:             st = f"🔊  Speaking… ({n} clip{'s' if n>1 else ''})"
        elif response_text: st = "💬  Generating response…"
        elif transcript:    st = "📝  Querying knowledge base…"
        else:               st = "⏳  Processing speech…"

        yield s, _render_chat(history, partial_user=transcript, partial_asst=response_text), audio_upd, st

        if not running:
            if transcript:    history.append({"role":"user",      "text": transcript})
            if response_text: history.append({"role":"assistant", "text": response_text})
            s["history"] = history; s["session_id"] = None; s["buffer"] = []
            yield s, _render_chat(history), gr.skip(), "✓  Done — tap 🎤 for another question"
            break
        time.sleep(POLL_INTERVAL_SECONDS)

# ── KPI panel HTML (static placeholders — swap values when wiring live data) ──
_KPI_HTML = """
<div class="kpi-panel">

  <div class="kpi-card">
    <div class="kpi-card-title">🎤 ASR — Speech Recognition</div>
    <div class="kpi-row">
      <span class="kpi-key">Model</span>
      <span class="kpi-val">whisper-base</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Backend</span>
      <span class="kpi-val">OpenVINO</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Precision</span>
      <span class="kpi-val"><span class="kpi-badge badge-green">INT8</span></span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Device</span>
      <span class="kpi-val">CPU</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Language</span>
      <span class="kpi-val">English</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Latency (est.)</span>
      <span class="kpi-val">— ms</span>
    </div>
  </div>

  <div class="kpi-card">
    <div class="kpi-card-title">🔍 RAG — Retrieval</div>
    <div class="kpi-row">
      <span class="kpi-key">Embeddings</span>
      <span class="kpi-val">all-MiniLM-L6-v2</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Vector DB</span>
      <span class="kpi-val">ChromaDB</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">LLM</span>
      <span class="kpi-val">— (placeholder)</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Top-K</span>
      <span class="kpi-val">—</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Latency (est.)</span>
      <span class="kpi-val">— ms</span>
    </div>
  </div>

  <div class="kpi-card">
    <div class="kpi-card-title">🔊 TTS — Speech Synthesis</div>
    <div class="kpi-row">
      <span class="kpi-key">Model</span>
      <span class="kpi-val">Qwen-TTS</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Backend</span>
      <span class="kpi-val">OpenVINO</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Precision</span>
      <span class="kpi-val"><span class="kpi-badge badge-blue">FP16</span></span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Device</span>
      <span class="kpi-val">CPU</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Language</span>
      <span class="kpi-val">English</span>
    </div>
    <div class="kpi-row">
      <span class="kpi-key">Latency (est.)</span>
      <span class="kpi-val">— ms</span>
    </div>
  </div>

</div>
"""

# ── App ───────────────────────────────────────────────────────────────────────
def create_app() -> gr.Blocks:
    with gr.Blocks(title="Kiosk Voice Assistant") as app:
        state = gr.State(dict(_INIT))

        gr.Markdown("## 🎙 Kiosk Voice Assistant")

        with gr.Row(elem_classes=["kiosk-row"]):

            # ── Left: chat + mic ──────────────────────────────────────────────
            with gr.Column(elem_classes=["kiosk-left"]):
                chat   = gr.HTML(value=_render_chat([]))
                status = gr.HTML(value='<div class="status-line">Tap the mic and ask a question</div>')
                mic    = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="Microphone",
                    elem_id="kiosk-mic",
                )
                tts = gr.Audio(label="Assistant", interactive=False, autoplay=True)

            # ── Right: collapsible model KPI panel ────────────────────────────
            with gr.Column(elem_classes=["kiosk-right"]):
                with gr.Accordion(label="📊 Model KPIs", open=False):
                    gr.HTML(value=_KPI_HTML)

        outs = [state, chat, tts, status]

        mic.start_recording(fn=on_start, inputs=[state],         outputs=outs)
        mic.stream(         fn=on_chunk, inputs=[state, mic],    outputs=outs, stream_every=0.5)
        mic.stop_recording( fn=on_stop,  inputs=[state],         outputs=outs)

    return app

def launch_app() -> Any:
    # Allow Gradio to serve TTS audio files generated by kiosk-core
    _generated_audio = os.path.join(
        os.path.dirname(__file__), "generated_audio"
    )
    os.makedirs(_generated_audio, exist_ok=True)
    return create_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=STYLE,
        allowed_paths=[_generated_audio],
    )

if __name__ == "__main__":
    launch_app()
