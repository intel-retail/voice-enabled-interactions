from __future__ import annotations

import io
import os
import time
import wave
from typing import Any, Generator
from urllib.parse import urlparse

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

# ── Derived base URLs for KPI endpoints ──────────────────────────────────────
def _svc_base(full_url: str) -> str:
    p = urlparse(full_url)
    return f"{p.scheme}://{p.netloc}"

_ANALYZER_BASE = _svc_base(ANALYZER_URL)
_TTS_BASE      = _svc_base(TTS_URL)
_RAG_BASE      = _svc_base(RAG_URL)

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

/* ═══════════════════════════════════════════════════════════
   AUDIO CONTROL STRIP  —  one unified card
   Left half : mic selector + record button
   Right half : speaker icon → waveform player when audio plays
   ═══════════════════════════════════════════════════════════ */

/* The row itself becomes the single card */
.audio-pair {
    gap: 0 !important;
    align-items: stretch !important;
    margin-top: 6px !important;
    flex-wrap: nowrap !important;
    background: linear-gradient(160deg, #EEF5FF 0%, #E6F0FA 100%) !important;
    border: 1.5px solid #C8D8EA !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 8px rgba(0,104,181,0.07) !important;
    overflow: hidden !important;
}
.audio-pair > div {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    align-self: stretch !important;
}
/* Inside each column, force the gr.Audio root and its block to fill the
   column's full height so vertical centering actually has room to work */
.audio-pair > div > #kiosk-mic,
.audio-pair > div > #kiosk-tts,
.audio-pair > div > .form,
.audio-pair > div > .form > #kiosk-mic,
.audio-pair > div > .form > #kiosk-tts {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}

/* Strip individual card styling from both blocks — they live inside the row card */
#kiosk-mic,
#kiosk-tts {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
#kiosk-mic > .block,
#kiosk-tts > .block {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 12px 14px !important;
    height: 100% !important;
    box-sizing: border-box !important;
}
#kiosk-mic > .block {
    display: block !important;
    position: relative !important;
    min-height: 140px !important;
}
/* Absolutely center the Speak button (and any mic controls) inside the
   mic half so it stays in the middle even when the TTS side stretches
   the row taller */
#kiosk-mic .controls,
#kiosk-mic .recording-container,
#kiosk-mic .minimal-audio-player {
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    margin: 0 !important;
    width: auto !important;
    height: auto !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* Vertical divider between mic and TTS halves */
#kiosk-tts > .block {
    border-left: 1px solid #C8D8EA !important;
}

/* Hide labels on both audio widgets */
#kiosk-mic label, #kiosk-mic .label-wrap, #kiosk-mic [data-testid="label"],
#kiosk-tts label, #kiosk-tts .label-wrap, #kiosk-tts [data-testid="label"] {
    display: none !important;
}

/* ── Mic half ── */
#kiosk-mic .minimal-audio-player,
#kiosk-mic .wrapper,
#kiosk-mic .waveform-wrapper {
    width: 100% !important;
    min-height: 76px !important;
    height: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#kiosk-mic .canvases,
#kiosk-mic .progress,
#kiosk-mic .scroll,
#kiosk-mic .cursor,
#kiosk-mic .timestamp {
    align-self: center !important;
}
#kiosk-mic .waveform-container,
#kiosk-mic .recording-container {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
    min-height: 76px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#kiosk-mic .controls,
#kiosk-mic .waveform-container .controls {
    background: transparent !important;
    gap: 8px !important;
    padding: 4px 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-wrap: wrap !important;
    width: 100% !important;
    margin: 0 auto !important;
}
#kiosk-mic .controls button,
#kiosk-mic .waveform-container button,
#kiosk-mic .recording-container button {
    background: #0068B5 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 999px !important;
    width: min(100%, 220px) !important;
    margin: 0 auto !important;
    height: 40px !important;
    min-width: 40px !important;
    padding: 0 16px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    white-space: nowrap !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    box-shadow: 0 3px 12px rgba(0,104,181,0.32) !important;
    transition: background 0.15s ease, transform 0.12s ease, box-shadow 0.15s ease !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
}
#kiosk-mic .controls button:hover,
#kiosk-mic .waveform-container button:hover,
#kiosk-mic .recording-container button:hover {
    background: #005A9E !important;
    transform: scale(1.06) !important;
    box-shadow: 0 5px 18px rgba(0,104,181,0.45) !important;
}
#kiosk-mic .controls button svg,
#kiosk-mic .waveform-container button svg,
#kiosk-mic .recording-container button svg {
    stroke: #FFFFFF !important;
    fill: #FFFFFF !important;
    width: 24px !important; height: 24px !important;
}
#kiosk-mic select {
    background: rgba(255,255,255,0.7) !important;
    border: 1px solid #C8D8EA !important;
    border-radius: 8px !important;
    color: #1A1A1A !important;
    font-size: 0.76rem !important;
    padding: 4px 10px !important;
    box-shadow: none !important;
    outline: none !important;
    cursor: pointer !important;
    width: 100% !important;
    margin-bottom: 6px !important;
}
#kiosk-mic select:focus {
    border-color: #0068B5 !important;
    box-shadow: 0 0 0 2px rgba(0,104,181,0.15) !important;
}
/* Hide the device selector from the left card (it lives in the Mic Device accordion) */
#kiosk-mic select {
    display: none !important;
}
/* Style the select once it's moved into the right-panel accordion */
/* Ensure the whole ancestor chain is full-width so 100% resolves correctly */
#mic-device-panel {
    width: 100% !important;
    box-sizing: border-box !important;
    display: block !important;
}
/* The gr.HTML block Gradio wraps around our div */
#mic-device-panel > *,
#mic-device-panel + * {
    width: 100% !important;
    box-sizing: border-box !important;
}
#mic-device-panel select {
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    margin-top: 8px !important;
    background: #FFFFFF !important;
    border: 1px solid #C8D8EA !important;
    border-radius: 8px !important;
    color: #1A1A1A !important;
    font-size: 0.82rem !important;
    padding: 7px 10px !important;
    box-shadow: none !important;
    outline: none !important;
    cursor: pointer !important;
    min-width: 0 !important;
}
#mic-device-panel select:focus {
    border-color: #0068B5 !important;
    box-shadow: 0 0 0 2px rgba(0,104,181,0.15) !important;
}
/* Reset/clear button — small ghost, turns red on hover to signal destructive
   Uses more specific selector (.controls [aria-label]) to override width:100% */
#kiosk-mic .controls [aria-label="Reset audio"],
#kiosk-mic .waveform-container [aria-label="Reset audio"],
#kiosk-mic .recording-container [aria-label="Reset audio"] {
    background: rgba(255,255,255,0.55) !important;
    color: #8FA0AE !important;
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    border: 1px solid #C8D8EA !important;
    box-shadow: none !important;
}
#kiosk-mic .controls [aria-label="Reset audio"]:hover,
#kiosk-mic .waveform-container [aria-label="Reset audio"]:hover,
#kiosk-mic .recording-container [aria-label="Reset audio"]:hover {
    background: rgba(211,47,47,0.08) !important;
    color: #D32F2F !important;
    border-color: rgba(211,47,47,0.35) !important;
    transform: scale(1.05) !important;
    box-shadow: none !important;
}
#kiosk-mic .controls [aria-label="Reset audio"] svg,
#kiosk-mic .waveform-container [aria-label="Reset audio"] svg,
#kiosk-mic .recording-container [aria-label="Reset audio"] svg {
    stroke: currentColor !important;
    fill: none !important;
    width: 13px !important;
    height: 13px !important;
}

/* ── Assistant / TTS half ── */
#kiosk-tts .waveform-container {
    background: transparent !important;
    border: none !important;
    padding: 2px 0 !important;
    min-height: 76px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
/* Replace music-note SVG with a compact assistant placeholder */
#kiosk-tts [aria-label="Empty value"] .icon svg {
    display: none !important;
}
#kiosk-tts [aria-label="Empty value"] {
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
}
#kiosk-tts [aria-label="Empty value"] .icon {
    width: auto;
    height: auto;
    margin: 0 auto;
    padding: 10px 14px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.78);
    border: 1px solid #D8E8F5;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
}
#kiosk-tts [aria-label="Empty value"] .icon::after {
    content: "🎙 Assistant voice";
    font-size: 1.08rem;
    font-weight: 500;
    display: block;
    line-height: 1;
    color: #6E879B;
    letter-spacing: 0.01em;
}
#kiosk-tts .controls {
    background: transparent !important;
    padding: 4px 2px !important;
    gap: 6px !important;
    display: flex !important;
    align-items: center !important;
}
#kiosk-tts .controls button,
#kiosk-tts .play-pause-button,
#kiosk-tts button.icon {
    background: rgba(255,255,255,0.8) !important;
    color: #0068B5 !important;
    border: 1px solid #C8D8EA !important;
    border-radius: 50% !important;
    width: 38px !important; height: 38px !important; min-width: 38px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 1px 5px rgba(0,104,181,0.12) !important;
    transition: background 0.15s ease, border-color 0.15s ease, transform 0.12s ease !important;
    cursor: pointer !important;
}
#kiosk-tts .controls button:hover,
#kiosk-tts .play-pause-button:hover,
#kiosk-tts button.icon:hover {
    background: #D0E8F8 !important;
    border-color: #0068B5 !important;
    transform: scale(1.07) !important;
}
#kiosk-tts .controls button svg,
#kiosk-tts .play-pause-button svg,
#kiosk-tts button.icon svg {
    stroke: #0068B5 !important;
    fill: #0068B5 !important;
    width: 17px !important;
    height: 17px !important;
}
/* Volume slider */
#kiosk-tts input[type=range] {
    accent-color: #0068B5 !important;
}
/* Timestamps */
#kiosk-tts .timestamps,
#kiosk-tts .timestamp {
    color: #4A6070 !important;
    font-size: 0.72rem !important;
    font-family: Inter, monospace !important;
}

/* Remaining Gradio shell overrides */
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

/* ── Knowledge-base ingest panel ── */
.ingest-note {
    font-size: 0.74rem;
    color: #4A6070;
    line-height: 1.6;
    padding: 10px 12px;
    background: #F4F7FB;
    border-radius: 8px;
    border-left: 3px solid #0068B5;
    margin-bottom: 10px;
}
.ingest-note strong { color: #1A1A1A; }
.ingest-status {
    font-size: 0.78rem;
    padding: 8px 12px;
    border-radius: 8px;
    margin-top: 8px;
    min-height: 0;
}
.ingest-status.loading { background: #EBF2FA; color: #004E8C; border: 1px solid #C8D8EA; }
.ingest-status.success { background: #D4F5E5; color: #0A6640; border: 1px solid #A8E6C8; }
.ingest-status.error   { background: #FDECEA; color: #B71C1C; border: 1px solid #F5C6CB; }
.ingest-status.warn    { background: #FFF8E1; color: #795500; border: 1px solid #FFD966; }
"""

# ── Chat HTML helpers ─────────────────────────────────────────────────────────
def _esc(t: str) -> str:
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")


_INGEST_NOTE_HTML = """
<div class="ingest-note">
  Upload a <strong>.txt</strong> or <strong>.md</strong> file to update the assistant&#39;s knowledge base.
  <strong>This replaces the existing knowledge base.</strong>
</div>
"""


def _ingest_doc(file) -> Generator:
    """Clear the current knowledge base and ingest the uploaded document."""
    if file is None:
        yield (
            '<div class="ingest-status warn">⚠️ Please select a file first.</div>',
            gr.update(),
            gr.update(),
        )
        return

    # Immediately lock the mic and button while work is in progress
    yield (
        '<div class="ingest-status loading">'
        '⏳ Refreshing knowledge base &#8212; the assistant will be back shortly&#8230;'
        '</div>',
        gr.update(interactive=False),
        gr.update(interactive=False, value="Ingesting…"),
    )

    filename = os.path.basename(file) if isinstance(file, str) else os.path.basename(file.name)
    filepath = file if isinstance(file, str) else file.name

    try:
        # 1. Wipe the existing knowledge base
        with httpx.Client(timeout=15.0, trust_env=False) as c:
            c.delete(f"{_RAG_BASE}/api/v1/context")

        # 2. Ingest the new document
        with open(filepath, "rb") as fh:
            content = fh.read()

        with httpx.Client(timeout=300.0, trust_env=False) as c:
            resp = c.post(
                f"{_RAG_BASE}/api/v1/context/file",
                files={"file": (filename, content, "text/plain")},
            )
            resp.raise_for_status()
            result = resp.json()

        chunks = result.get("chunks_added", "?")
        src    = result.get("source", filename)
        yield (
            f'<div class="ingest-status success">'
            f'✅ Knowledge base updated &#8212; {chunks} chunks ingested from'
            f' &#34;{_esc(src)}&#34;. The assistant is ready.'
            f'</div>',
            gr.update(interactive=True),
            gr.update(interactive=True, value="Upload & Ingest"),
        )

    except Exception as exc:  # noqa: BLE001
        yield (
            f'<div class="ingest-status error">'
            f'⚠️ Ingestion failed: {_esc(str(exc))}. '
            f'The previous knowledge base remains active.'
            f'</div>',
            gr.update(interactive=True),
            gr.update(interactive=True, value="Upload & Ingest"),
        )

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
            "tts_model": "speecht5", "tts_language": "English",
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

# ── KPI helpers ───────────────────────────────────────────────────────────────
_PROVIDER_LABELS = {
    "openai": "OpenAI Whisper", "openvino": "OpenVINO",
    "whispercpp": "Whisper.cpp", "pytorch": "PyTorch",
}
_DTYPE_COLORS = {"int4": "purple", "int8": "green", "fp16": "blue", "fp32": "blue"}


def _fmt_ms(val: Any) -> str:
    return f"{val:,.0f} ms" if val is not None else "—"


def _badge(val: str | None) -> str:
    if not val:
        return "—"
    color = _DTYPE_COLORS.get(val.lower(), "blue")
    return f'<span class="kpi-badge badge-{color}">{_esc(val.upper())}</span>'


def _kpi_row(key: str, val: str) -> str:
    return (
        f'<div class="kpi-row">'
        f'<span class="kpi-key">{key}</span>'
        f'<span class="kpi-val">{val}</span>'
        f'</div>'
    )


def _kpi_card(title: str, rows: list[tuple[str, str]]) -> str:
    inner = "\n".join(_kpi_row(k, v) for k, v in rows)
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-card-title">{title}</div>'
        f'{inner}'
        f'</div>'
    )


def _get_kpi(url: str) -> dict:
    try:
        with httpx.Client(timeout=4.0, trust_env=False) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.json()
    except Exception:
        return {}


def _fetch_kpis() -> tuple[dict, dict, dict]:
    """Return (asr_data, rag_data, tts_data) with merged model-info + latency."""
    asr_info  = _get_kpi(f"{_ANALYZER_BASE}/v1/model-info")
    asr_perf  = _get_kpi(f"{_ANALYZER_BASE}/v1/performance")
    tts_info  = _get_kpi(f"{_TTS_BASE}/v1/model-info")
    tts_perf  = _get_kpi(f"{_TTS_BASE}/v1/performance")
    rag_info  = _get_kpi(f"{_RAG_BASE}/api/v1/model-info")
    rag_perf  = _get_kpi(f"{_RAG_BASE}/api/v1/performance")
    return (
        {**asr_info, "perf": asr_perf.get("latency", {})},
        {**rag_info, "perf": rag_perf.get("latency", {})},
        {**tts_info, "perf": tts_perf.get("latency", {})},
    )


def _render_kpi_html(asr: dict, rag: dict, tts: dict) -> str:
    ap, rp, tp = asr.get("perf", {}), rag.get("perf", {}), tts.get("perf", {})

    asr_provider = _PROVIDER_LABELS.get(str(asr.get("provider", "")).lower(),
                                        asr.get("provider") or "—")
    asr_card = _kpi_card("🎤 ASR — Speech Recognition", [
        ("Model",         _esc(str(asr.get("model") or "—"))),
        ("Backend",       _esc(asr_provider)),
        ("Precision",     _badge(str(asr.get("weight_format") or "") or None)),
        ("Device",        _esc(str(asr.get("device") or "—").upper())),
        ("Last latency",  _fmt_ms(ap.get("last_ms"))),
    ])

    llm_id  = str(rag.get("llm_model") or "—")
    emb_id  = str(rag.get("embedding_model") or "—")
    rag_card = _kpi_card("🔍 RAG — Retrieval + Generation", [
        ("LLM",           _esc(llm_id.split("/")[-1])),
        ("LLM Device",    _esc(str(rag.get("llm_device") or "—"))),
        ("Precision",     _badge(str(rag.get("llm_weight_format") or "") or None)),
        ("Embeddings",    _esc(emb_id.split("/")[-1])),
        ("Emb Device",    _esc(str(rag.get("embedding_device") or "—"))),
        ("Docs indexed",  _esc(str(rag.get("document_count") if rag.get("document_count") is not None else "—"))),
        ("Top-K",         _esc(str(rag.get("top_k") or "—"))),
        ("Last latency",  _fmt_ms(rp.get("last_ms"))),
    ])

    tts_runtime = _PROVIDER_LABELS.get(str(tts.get("runtime", "")).lower(),
                                       tts.get("runtime") or "—")
    tts_model   = str(tts.get("model") or "—").split("/")[-1]
    tts_card = _kpi_card("🔊 TTS — Speech Synthesis", [
        ("Model",         _esc(tts_model)),
        ("Backend",       _esc(tts_runtime)),
        ("Precision",     _badge(str(tts.get("dtype") or "") or None)),
        ("Device",        _esc(str(tts.get("device") or "—").upper())),
        ("Language",      _esc(str(tts.get("default_language") or "—"))),
        ("Last latency",  _fmt_ms(tp.get("last_ms"))),
    ])

    return f'<div class="kpi-panel">{asr_card}{rag_card}{tts_card}</div>'


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
                with gr.Row(elem_classes=["audio-pair"]):
                    with gr.Column(scale=1, min_width=160):
                        mic = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            streaming=True,
                            label="🎤 Your Voice",
                            elem_id="kiosk-mic",
                        )
                    with gr.Column(scale=1, min_width=160):
                        tts = gr.Audio(
                            label="�️ Assistant",
                            interactive=False,
                            autoplay=True,
                            elem_id="kiosk-tts",
                        )

            # ── Right: collapsible panels ────────────────────────────────────
            with gr.Column(elem_classes=["kiosk-right"]):
                with gr.Accordion(label="🎤 Device Settings", open=False):
                    gr.HTML(value='<div id="mic-device-panel"><p class="ingest-note">Select the microphone to use for recording.</p></div>')

                with gr.Accordion(label="📚 Update Knowledge Base", open=False):
                    gr.HTML(value=_INGEST_NOTE_HTML)
                    doc_file = gr.File(
                        file_types=[".txt", ".md"],
                        label="Select document (.txt or .md)",
                        file_count="single",
                    )
                    ingest_btn = gr.Button("Upload & Ingest", variant="primary", size="sm")
                    ingest_status = gr.HTML(value="")

                with gr.Accordion(label="📊 Model KPIs", open=True):
                    kpi_panel = gr.HTML(value=_render_kpi_html({}, {}, {}))
                    refresh_btn = gr.Button("🔄 Refresh", size="sm", variant="secondary")

        outs = [state, chat, tts, status]

        mic.start_recording(fn=on_start, inputs=[state],         outputs=outs)
        mic.stream(         fn=on_chunk, inputs=[state, mic],    outputs=outs, stream_every=0.5)
        mic.stop_recording( fn=on_stop,  inputs=[state],         outputs=outs)

        refresh_btn.click(
            fn=lambda: _render_kpi_html(*_fetch_kpis()),
            outputs=[kpi_panel],
        )
        ingest_btn.click(
            fn=_ingest_doc,
            inputs=[doc_file],
            outputs=[ingest_status, mic, ingest_btn],
        )
        app.load(
            fn=lambda: _render_kpi_html(*_fetch_kpis()),
            outputs=[kpi_panel],
        )
        app.load(
            fn=None,
            js="""() => {
                // Rename "Record" button to "Speak"
                const renameRecord = () => {
                    document.querySelectorAll('#kiosk-mic button').forEach(btn => {
                        btn.childNodes.forEach(node => {
                            if (node.nodeType === Node.TEXT_NODE &&
                                    node.textContent.trim().toLowerCase() === 'record') {
                                node.textContent = node.textContent.replace(/record/i, 'Speak');
                            }
                        });
                        const span = btn.querySelector('span');
                        if (span && span.textContent.trim().toLowerCase() === 'record') {
                            span.textContent = 'Speak';
                        }
                    });
                };
                const moveSelect = () => {
                    const sel = document.querySelector('#kiosk-mic select');
                    const tgt = document.querySelector('#mic-device-panel');
                    if (sel && tgt && !tgt.contains(sel)) {
                        // Strip any inline width/size styles so CSS 100% takes over
                        sel.removeAttribute('style');
                        sel.style.width = '100%';
                        sel.style.boxSizing = 'border-box';
                        tgt.appendChild(sel);
                        return true;
                    }
                    return !!(sel && tgt);
                };
                // Retry every 400 ms for up to 15 s (Svelte renders lazily).
                let tries = 0;
                const poll = setInterval(() => {
                    renameRecord();
                    if (moveSelect() || ++tries > 37) clearInterval(poll);
                }, 400);
                // Re-run whenever DOM changes (accordion open, Gradio re-render)
                const obs = new MutationObserver(() => { moveSelect(); renameRecord(); });
                obs.observe(document.body, { childList: true, subtree: true });
                setTimeout(() => obs.disconnect(), 20000);
            }""",
        )

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
