"""Knowledge-base ingestion + question-answering accuracy tests.

This module verifies the full RAG loop end to end against a *real* running
stack:

  1. Ingest ``knowledge-base-samples/QuickBite-M.md`` through the public
     ingestion API (``POST /api/v1/context/file``).
  2. Ask a battery of questions whose answers appear **only** in that
     document, over both retrieval paths the kiosk actually uses:

       * ``POST /api/v1/query``       — direct RAG (SSE token stream), the
         path used by the voice pipeline for pure information questions.
       * ``POST /api/v1/agent/chat``  — the agentic path, which must decide
         to call the ``knowledge_lookup`` tool rather than answering from
         parametric memory or reaching for an ordering tool.

  3. Score each answer by expected-keyword presence and assert an overall
     pass-rate threshold.

Why keyword scoring
-------------------
The LLM phrases answers freely, so exact-match assertions are brittle.  Each
question therefore declares ``expected`` — a list of *alternative groups*.  An
answer passes when **every** group has at least one of its variants present
(case-insensitive).  This tolerates paraphrasing while still catching a model
that invents a price or silently fails to retrieve.

Running
-------
As a pytest tier3 test (requires the full stack to be up)::

    pytest tests/functional/test_knowledge_base_qa.py -m tier3

Standalone, with a human-readable report::

    python3 tests/functional/test_knowledge_base_qa.py
    python3 tests/functional/test_knowledge_base_qa.py --clear --agent-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover
    # The host interpreter has no pytest (tests normally run in CI/containers).
    # Provide a minimal shim so this module stays importable and usable as a
    # standalone script.
    class _MarkStub:
        def __getattr__(self, _name):
            def _decorator(func):
                return func

            return _decorator

    class _PytestStub:
        mark = _MarkStub()

        @staticmethod
        def fixture(*args, **kwargs):
            def _decorator(func):
                return func

            return _decorator(args[0]) if args and callable(args[0]) else _decorator

        @staticmethod
        def skip(reason: str = ""):
            raise RuntimeError(f"skipped: {reason}")

    pytest = _PytestStub()  # type: ignore[assignment]

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


_KIOSK_ROOT = Path(__file__).resolve().parents[2]

RAG_BASE = "http://localhost:8020"
KB_FILE = _KIOSK_ROOT / "knowledge-base-samples" / "QuickBite-M.md"

# Ingestion has to embed ~470 lines of markdown; the LLM-backed semantic
# chunker makes this slow on a cold embedding model.
INGEST_TIMEOUT_SECONDS = 600
# A single RAG answer includes retrieval + rerank + generation on the iGPU.
QUERY_TIMEOUT_SECONDS = 300

# Fraction of questions that must pass for the suite to be considered healthy.
# Retrieval is stochastic (reranker ties, chunk boundaries), so we assert a
# threshold rather than perfection — but a real regression drops well below it.
MIN_PASS_RATE = 0.80


# ---------------------------------------------------------------------------
# Question bank — every answer is grounded in QuickBite-M.md
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KBQuestion:
    """A knowledge-base question and the evidence its answer must contain.

    Attributes:
        qid: Short stable identifier used in reports.
        question: Natural-language question as a customer would ask it.
        expected: Alternative groups. The answer passes only when every group
            has at least one variant present (case-insensitive substring).
        section: KB section the fact comes from, for failure triage.
    """

    qid: str
    question: str
    expected: list[list[str]]
    section: str


KB_QUESTIONS: list[KBQuestion] = [
    KBQuestion(
        qid="hours-sunday",
        question="What time do you open on Sunday?",
        expected=[["9 am", "9am", "9:00", "nine"]],
        section="Header / Hours",
    ),
    KBQuestion(
        qid="wifi-ssid",
        question="What is the guest Wi-Fi network name?",
        expected=[["quickbite_guest", "quickbite guest"]],
        section="Header / Wi-Fi",
    ),
    KBQuestion(
        qid="loyalty-name",
        question="What is your loyalty program called?",
        expected=[["qbrewards", "qb rewards"]],
        section="Header / Loyalty",
    ),
    KBQuestion(
        qid="delivery-radius",
        question="What is your delivery radius?",
        expected=[["4 km", "4km", "four km"]],
        section="Header / Delivery",
    ),
    KBQuestion(
        qid="seating",
        question="How many indoor seats does the outlet have?",
        expected=[["64"]],
        section="Header / Seating",
    ),
    KBQuestion(
        qid="catering-notice",
        question="How much advance notice do you need for bulk catering orders?",
        expected=[["24-hour", "24 hour", "24 hours", "24-hours"]],
        section="Header / Catering",
    ),
    KBQuestion(
        qid="last-order",
        question="How long before closing can I place my last order?",
        expected=[["30 min", "30-min", "thirty min", "half an hour"]],
        section="Header / Last Order",
    ),
    KBQuestion(
        qid="price-classic-chicken",
        question="How much does the Classic Chicken Burger cost?",
        expected=[["169"]],
        section="MENU 1 / Chicken Burgers",
    ),
    KBQuestion(
        qid="price-jain-burger",
        question="What is the price of the Jain Burger?",
        expected=[["139"]],
        section="MENU 1 / Vegetarian Burgers",
    ),
    KBQuestion(
        qid="vegan-burger",
        question="Do you have a vegan burger, and what is it called?",
        expected=[["vegan soya", "soya burger"]],
        section="MENU 1 / Vegetarian Burgers",
    ),
    KBQuestion(
        qid="cheese-addon",
        question="How much extra does it cost to add a cheese slice to a burger?",
        expected=[["25"]],
        section="MENU 1 / Customisation",
    ),
    KBQuestion(
        qid="wheat-bun",
        question="Is there a surcharge for a whole wheat bun?",
        expected=[["15"]],
        section="MENU 1 / Bread Options",
    ),
    KBQuestion(
        qid="classic-combo",
        question="What is included in the Classic Combo and how much is it?",
        expected=[["199"], ["fries"]],
        section="MENU 10 / Burger Combos",
    ),
    KBQuestion(
        qid="dietary-tag-ve",
        question="In your menu what does the VE dietary tag mean?",
        expected=[["vegan"]],
        section="Dietary Tag System",
    ),
    KBQuestion(
        qid="breakfast-hours",
        question="Until what time do you serve breakfast?",
        expected=[["11 am", "11am", "11:00", "eleven"]],
        section="Header / Breakfast Hours",
    ),
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _require_requests() -> None:
    if requests is None:  # pragma: no cover
        pytest.skip("`requests` is not installed on the host interpreter")


def _service_healthy(url: str, timeout: int = 10) -> bool:
    """Single-shot health probe that never raises."""
    try:
        return requests.get(url, timeout=timeout).status_code < 400
    except Exception:  # noqa: BLE001
        return False


def _skip_if_rag_down() -> None:
    _require_requests()
    if not _service_healthy(f"{RAG_BASE}/health"):
        pytest.skip(
            f"rag-service is not healthy ({RAG_BASE}/health) — run `make up` first."
        )


def clear_context() -> None:
    """Drop every document from the shared vector collection."""
    resp = requests.delete(f"{RAG_BASE}/api/v1/context", timeout=120)
    resp.raise_for_status()


def context_stats() -> dict[str, Any]:
    """Return the vector-store statistics reported by rag-service."""
    resp = requests.get(f"{RAG_BASE}/api/v1/context/stats", timeout=60)
    resp.raise_for_status()
    return resp.json()


def ingest_knowledge_base(path: Path = KB_FILE) -> dict[str, Any]:
    """Ingest a markdown knowledge base through the public file API.

    Handles both response shapes the service may return:
      * ``BatchIngestResponse`` — ``{total_chunks_added, files_*, results}``
      * ``IngestResponse``      — ``{chunks_added, source}`` (older builds)

    Args:
        path: Path to the ``.md`` document to ingest.

    Returns:
        A normalised dict with ``total_chunks_added``, ``files_processed``,
        ``files_failed``, ``results`` and the original body under ``raw``.

    Raises:
        FileNotFoundError: If the knowledge-base file is missing.
        AssertionError: If the service reports a failure.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge-base file not found: {path}")

    with path.open("rb") as handle:
        resp = requests.post(
            f"{RAG_BASE}/api/v1/context/file",
            files={"file": (path.name, handle, "text/markdown")},
            timeout=INGEST_TIMEOUT_SECONDS,
        )
    assert resp.status_code == 200, (
        f"Ingestion failed: HTTP {resp.status_code} — {resp.text[:500]}"
    )
    body = resp.json()

    if "total_chunks_added" in body:
        normalised = {
            "total_chunks_added": body["total_chunks_added"],
            "files_processed": body.get("files_processed", 1),
            "files_failed": body.get("files_failed", 0),
            "results": body.get("results", []),
        }
    else:
        normalised = {
            "total_chunks_added": body.get("chunks_added", 0),
            "files_processed": 1,
            "files_failed": 0,
            "results": [
                {"source": body.get("source", path.name), "status": "ok",
                 "chunks_added": body.get("chunks_added", 0)}
            ],
        }

    failed = [r for r in normalised["results"] if r.get("status") != "ok"]
    assert not failed, f"Ingestion reported failures: {failed}"
    normalised["raw"] = body
    return normalised


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 ``<think>`` chain-of-thought from a generated answer.

    The RAG path does not disable thinking, so answers arrive as
    ``<think>…</think>`` followed by the real reply.  An unterminated block
    (truncated generation) is dropped entirely rather than scored.
    """
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_OPEN.sub("", text)
    return text.strip()


def query_rag(question: str, top_k: int | None = None) -> tuple[str, str]:
    """Ask the direct RAG endpoint.

    ``/api/v1/query`` responds with an SSE stream of ``{"token": "..."}``
    events terminated by an optional metrics event.

    Returns:
        ``(answer, raw)`` where ``answer`` has any ``<think>`` block removed
        and ``raw`` is the untouched generation (needed to tell a genuinely
        wrong answer apart from one truncated mid-thinking).
    """
    payload: dict[str, Any] = {"transcription": question}
    if top_k is not None:
        payload["top_k"] = top_k

    tokens: list[str] = []
    with requests.post(
        f"{RAG_BASE}/api/v1/query",
        json=payload,
        stream=True,
        timeout=QUERY_TIMEOUT_SECONDS,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            try:
                event = json.loads(raw_line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            if "token" in event:
                tokens.append(event["token"])
            elif "error" in event:
                raise RuntimeError(f"rag-service streamed an error: {event['error']}")
    raw = "".join(tokens)
    return strip_thinking(raw), raw


def is_thinking_truncated(raw: str) -> bool:
    """True when generation opened a ``<think>`` block but never closed it.

    Qwen3 chain-of-thought is not disabled on the RAG path, so a long enough
    reasoning trace can consume the whole token budget and leave no answer.
    """
    return "<think>" in raw.lower() and "</think>" not in raw.lower()


def query_agent(question: str, session_id: str) -> tuple[str, list[str]]:
    """Ask the agentic endpoint and return ``(reply, tool_calls)``."""
    resp = requests.post(
        f"{RAG_BASE}/api/v1/agent/chat",
        json={
            "session_id": session_id,
            "user_id": "kb-test",
            "transcription": question,
        },
        timeout=QUERY_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    body = resp.json()
    return strip_thinking(body.get("reply") or ""), list(body.get("tool_calls") or [])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class AnswerResult:
    """Outcome of scoring one question against one retrieval path."""

    qid: str
    question: str
    section: str
    answer: str
    passed: bool
    missing: list[list[str]] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    truncated_thinking: bool = False


def score_answer(answer: str, expected: Iterable[Iterable[str]]) -> list[list[str]]:
    """Return the expected groups that are absent from ``answer``.

    An empty return value means the answer satisfied every group.
    """
    haystack = answer.lower()
    return [
        list(group)
        for group in expected
        if not any(variant.lower() in haystack for variant in group)
    ]


def _normalise_digits(text: str) -> str:
    """Strip thousands separators so "1,299" matches an expected "1299"."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def evaluate(
    questions: list[KBQuestion],
    *,
    use_agent: bool,
    session_prefix: str,
) -> list[AnswerResult]:
    """Run every question through one retrieval path and score the answers."""
    results: list[AnswerResult] = []
    for q in questions:
        started = time.perf_counter()
        tool_calls: list[str] = []
        truncated = False
        try:
            if use_agent:
                # A fresh session per question keeps history from leaking
                # answers between questions.
                answer, tool_calls = query_agent(q.question, f"{session_prefix}-{q.qid}")
            else:
                answer, raw = query_rag(q.question)
                truncated = is_thinking_truncated(raw)
        except Exception as exc:  # noqa: BLE001
            answer = f"<request failed: {exc}>"
        latency_ms = (time.perf_counter() - started) * 1000

        missing = score_answer(_normalise_digits(answer), q.expected)
        results.append(
            AnswerResult(
                qid=q.qid,
                question=q.question,
                section=q.section,
                answer=answer,
                passed=not missing,
                missing=missing,
                tool_calls=tool_calls,
                latency_ms=latency_ms,
                truncated_thinking=truncated,
            )
        )
    return results


def pass_rate(results: list[AnswerResult]) -> float:
    """Fraction of results that passed, or 0.0 for an empty list."""
    return (sum(r.passed for r in results) / len(results)) if results else 0.0


def format_report(title: str, results: list[AnswerResult]) -> str:
    """Render a human-readable pass/fail table."""
    lines = [
        "=" * 78,
        title,
        "=" * 78,
    ]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"[{mark}] {r.qid:<24} {r.latency_ms:7.0f} ms  {r.question}")
        if r.tool_calls:
            lines.append(f"         tools : {r.tool_calls}")
        snippet = " ".join(r.answer.split())[:150]
        lines.append(f"         answer: {snippet}")
        if r.truncated_thinking:
            lines.append(
                "         NOTE  : generation ran out of tokens inside <think> — "
                "no answer was produced (disable thinking on this path)"
            )
        if not r.passed:
            lines.append(f"         MISSING evidence: {r.missing}  (KB: {r.section})")
    passed = sum(r.passed for r in results)
    truncated = sum(r.truncated_thinking for r in results)
    lines.append("-" * 78)
    lines.append(
        f"{passed}/{len(results)} passed  ({pass_rate(results) * 100:.0f}%)"
    )
    if truncated:
        lines.append(
            f"{truncated} answer(s) lost to unterminated chain-of-thought"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pytest tests (tier3 — full stack required)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ingested_kb() -> dict[str, Any]:
    """Ingest the QuickBite knowledge base once for the whole module."""
    _skip_if_rag_down()
    before = context_stats().get("document_count", 0)
    body = ingest_knowledge_base()
    after = context_stats().get("document_count", 0)
    return {"response": body, "documents_before": before, "documents_after": after}


class TestKnowledgeBaseIngestion:
    """The knowledge base must be ingestible through the public API."""

    @pytest.mark.tier3
    def test_kb_sample_file_exists(self):
        """The QuickBite knowledge-base sample must ship with the repo."""
        assert KB_FILE.is_file(), f"Missing knowledge-base sample: {KB_FILE}"
        assert KB_FILE.stat().st_size > 1000, "Knowledge-base sample looks truncated"

    @pytest.mark.tier3
    def test_ingest_adds_chunks(self, ingested_kb):
        """Ingestion must report at least one chunk added and succeed for the file."""
        body = ingested_kb["response"]
        assert body["files_processed"] == 1, body
        assert body["files_failed"] == 0, body
        assert body["total_chunks_added"] > 0, (
            f"Ingestion added no chunks — the document was not indexed: {body}"
        )

    @pytest.mark.tier3
    def test_vector_store_grew(self, ingested_kb):
        """The vector store's document count must increase after ingestion."""
        assert ingested_kb["documents_after"] > ingested_kb["documents_before"], (
            "document_count did not increase after ingestion "
            f"({ingested_kb['documents_before']} → {ingested_kb['documents_after']})"
        )


class TestKnowledgeBaseQA:
    """Questions answerable only from the ingested document must be answered."""

    @pytest.mark.tier3
    def test_direct_rag_answers_questions(self, ingested_kb):
        """`POST /api/v1/query` must answer KB questions with grounded facts."""
        results = evaluate(KB_QUESTIONS, use_agent=False, session_prefix="kb-rag")
        report = format_report("Direct RAG — /api/v1/query", results)
        print("\n" + report)
        assert pass_rate(results) >= MIN_PASS_RATE, (
            f"Direct RAG answered only {pass_rate(results) * 100:.0f}% correctly "
            f"(need {MIN_PASS_RATE * 100:.0f}%).\n{report}"
        )

    @pytest.mark.tier3
    def test_agent_answers_questions_via_knowledge_lookup(self, ingested_kb):
        """The agent must route info questions to `knowledge_lookup` and answer them."""
        results = evaluate(KB_QUESTIONS, use_agent=True, session_prefix="kb-agent")
        report = format_report("Agentic RAG — /api/v1/agent/chat", results)
        print("\n" + report)
        assert pass_rate(results) >= MIN_PASS_RATE, (
            f"Agent answered only {pass_rate(results) * 100:.0f}% correctly "
            f"(need {MIN_PASS_RATE * 100:.0f}%).\n{report}"
        )

    @pytest.mark.tier3
    def test_agent_uses_knowledge_lookup_tool(self, ingested_kb):
        """An information question must trigger the knowledge_lookup tool.

        Guards against the agent answering menu/policy questions from
        parametric memory (ungrounded) or misrouting them to an ordering tool.
        """
        _, tool_calls = query_agent(
            "What is the guest Wi-Fi network name?", "kb-agent-tool-check"
        )
        assert "knowledge_lookup" in tool_calls, (
            f"Expected knowledge_lookup to be called, got tool_calls={tool_calls}"
        )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Run ingestion + Q&A as a script and print a report."""
    parser = argparse.ArgumentParser(
        description="Ingest the QuickBite knowledge base and test RAG question answering."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing context before ingesting (clean-room run).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Query only; assume the knowledge base is already ingested.",
    )
    parser.add_argument(
        "--rag-only", action="store_true", help="Only test the direct /api/v1/query path."
    )
    parser.add_argument(
        "--agent-only", action="store_true", help="Only test the /api/v1/agent/chat path."
    )
    parser.add_argument(
        "--kb-file", type=Path, default=KB_FILE, help="Knowledge-base markdown file."
    )
    args = parser.parse_args(argv)

    if requests is None:
        print("ERROR: the `requests` package is required.", file=sys.stderr)
        return 2

    if not _service_healthy(f"{RAG_BASE}/health"):
        print(f"ERROR: rag-service is not healthy at {RAG_BASE}/health.", file=sys.stderr)
        print("Start the stack with `make up` first.", file=sys.stderr)
        return 2

    if args.clear:
        print("[kb] clearing existing context …")
        clear_context()

    if not args.skip_ingest:
        before = context_stats().get("document_count", 0)
        print(f"[kb] ingesting {args.kb_file.name} (documents before: {before}) …")
        started = time.perf_counter()
        body = ingest_knowledge_base(args.kb_file)
        elapsed = time.perf_counter() - started
        after = context_stats().get("document_count", 0)
        print(
            f"[kb] ingested {body['total_chunks_added']} chunk(s) in {elapsed:.1f}s "
            f"— documents {before} → {after}"
        )

    exit_code = 0
    if not args.agent_only:
        results = evaluate(KB_QUESTIONS, use_agent=False, session_prefix="kb-rag")
        print("\n" + format_report("Direct RAG — /api/v1/query", results))
        if pass_rate(results) < MIN_PASS_RATE:
            exit_code = 1

    if not args.rag_only:
        results = evaluate(KB_QUESTIONS, use_agent=True, session_prefix="kb-agent")
        print("\n" + format_report("Agentic RAG — /api/v1/agent/chat", results))
        if pass_rate(results) < MIN_PASS_RATE:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
