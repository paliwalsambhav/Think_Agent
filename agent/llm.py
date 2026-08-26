"""
LLM wrapper around Groq with:
  - Model routing  : cheap fast model for tool-heavy research steps,
                     larger model for planning / reflection / synthesis
  - Retry / backoff: up to MAX_RETRIES attempts with exponential back-off
                     that respects the Retry-After header from Groq
  - Token budgets  : kept small to stay within free-tier TPM limits
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

from groq import APIStatusError, Groq, RateLimitError

log = logging.getLogger(__name__)

# ---------- token usage tracking (for AgentOps tracing) ----------
# Accumulates across every _call() made during one node's execution.
# Nodes call reset_usage() at the start and get_usage() at the end to
# report real input/output token counts to the tracer.
_usage_accumulator = {"input_tokens": 0, "output_tokens": 0}


def reset_usage() -> None:
    _usage_accumulator["input_tokens"] = 0
    _usage_accumulator["output_tokens"] = 0


def get_usage() -> dict:
    return dict(_usage_accumulator)


# ---------- model aliases ----------
FAST_MODEL   = os.environ.get("GROQ_FAST_MODEL", "openai/gpt-oss-20b")   # high TPM limit on free tier ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â use for research loops
REASON_MODEL = os.environ.get("GROQ_REASON_MODEL", "openai/gpt-oss-120b")  # smarter ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â use for plan / reflect / synthesise
# NOTE: previously hardcoded to "llama3-70b-8192", which Groq has decommissioned.
# Every call using REASON_MODEL was silently failing and falling into fallback
# paths in graph.py (planner_node, reflect_node, synthesize_node).
# If you want a genuinely larger model again, check Groq's current model list
# and set GROQ_REASON_MODEL in your .env ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â do not hardcode it here.

MAX_RETRIES    = 6
BASE_DELAY_SEC = 1.5  # minimum wait between retries
MAX_DELAY_SEC  = 60

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(
            api_key=os.environ["GROQ_API_KEY"],
            max_retries=0,  # we handle retries ourselves for proper back-off
        )
    return _client


def _call(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float = 0.3,
) -> str:
    """Core retry loop around Groq chat completions."""
    client = _get_client()
    delay = BASE_DELAY_SEC

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if resp.usage:
                _usage_accumulator["input_tokens"] += resp.usage.prompt_tokens
                _usage_accumulator["output_tokens"] += resp.usage.completion_tokens
            return resp.choices[0].message.content.strip()

        except RateLimitError as exc:
            retry_after = None
            try:
                retry_after = float(exc.response.headers.get("retry-after", 0))
            except Exception:
                pass
            wait = max(retry_after or delay, delay)
            log.warning(
                "Rate-limited (attempt %d/%d). Waiting %.1fs ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦",
                attempt, MAX_RETRIES, wait,
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)
            delay = min(delay * 2, MAX_DELAY_SEC)

        except APIStatusError as exc:
            if exc.status_code in (500, 502, 503, 504) and attempt < MAX_RETRIES:
                log.warning("Groq %d error, retrying in %.1fs ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦", exc.status_code, delay)
                time.sleep(delay)
                delay = min(delay * 2, MAX_DELAY_SEC)
            else:
                raise

    raise RuntimeError("Unreachable")


def _strip_fences(text: str) -> str:
    """Remove accidental markdown code fences from JSON responses."""
    return re.sub(r"```[a-z]*", "", text).strip().strip("`").strip()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def complete(system: str, user: str, json_mode: bool = False) -> str:
    """
    General-purpose LLM call. Uses REASON_MODEL.
    Set json_mode=True when you expect a JSON response ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â strips fences automatically.
    """
    raw = _call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        model=REASON_MODEL,
        max_tokens=600,
    )
    return _strip_fences(raw) if json_mode else raw


def plan(topic: str, prior_notes: list[str]) -> list[str]:
    """Return 3ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ4 sub-questions to research."""
    prior = "\n".join(prior_notes) if prior_notes else "None"
    content = _call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research planner. Given a topic and any prior notes, "
                    "return ONLY a JSON array of 3-4 focused sub-questions (strings). "
                    "No markdown, no preamble ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â raw JSON array only."
                ),
            },
            {"role": "user", "content": f"Topic: {topic}\n\nPrior notes:\n{prior}"},
        ],
        model=REASON_MODEL,
        max_tokens=300,
    )
    return json.loads(_strip_fences(content))


_SEARCH_KEYWORDS = (
    "current", "latest", "today", "now", "recent", "2024", "2025", "2026",
    "2027", "price", "pricing", "cost", "rate limit", "release", "version",
    "news", "update", "announcement",
)


def should_search(question: str) -> bool:
    """Decide whether a question requires a live web search.

    Defaults to searching whenever there's real doubt: an LLM saying "NO" on
    a time-sensitive question (pricing, versions, "current/latest") silently
    routes to answer_from_knowledge(), which just guesses from the model's
    training cutoff. A keyword check catches the obvious cases before we
    even ask the LLM, since those are exactly the questions where a wrong
    NO is most costly and most likely.
    """
    if any(kw in question.lower() for kw in _SEARCH_KEYWORDS):
        return True  # keyword already confirms it -- skip the extra Groq call

    answer = _call(
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer with exactly one word: YES or NO.\n"
                    "Say YES if the question could involve any current, "
                    "recent, time-sensitive, or frequently-changing "
                    "information (prices, versions, dates, news, statuses, "
                    "rankings, availability, etc.) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â when in doubt, say YES.\n"
                    "Say NO only for timeless facts, definitions, or "
                    "concepts that don't change over time."
                ),
            },
            {"role": "user", "content": question},
        ],
        model=FAST_MODEL,
        max_tokens=5,
    )
    return not answer.strip().upper().startswith("N")


def answer_from_context(question: str, search_results: str) -> str:
    """Synthesise a concise answer from raw search snippets."""
    return _call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Using ONLY the provided search results, "
                    "write a concise 2-4 sentence answer. Cite no sources by URL ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â just facts."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nSearch results:\n{search_results}",
            },
        ],
        model=FAST_MODEL,
        max_tokens=350,
    )


def answer_from_knowledge(question: str) -> str:
    """Answer a question from parametric knowledge (no search needed)."""
    return _call(
        messages=[
            {
                "role": "system",
                "content": "Answer the question concisely in 2-4 sentences.",
            },
            {"role": "user", "content": question},
        ],
        model=FAST_MODEL,
        max_tokens=300,
    )


def reflect(topic: str, notes: list[str]) -> tuple[bool, list[str]]:
    """
    Reflect on the notes collected so far.
    Returns (needs_more_research: bool, additional_questions: list[str]).
    """
    notes_text = "\n".join(f"- {n}" for n in notes)
    raw = _call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a critical research editor. "
                    "Given a topic and research notes, decide if the notes are sufficient "
                    "for a comprehensive report. "
                    'Reply with a JSON object: {"sufficient": true/false, "additional_questions": ["..."]}. '
                    "additional_questions should be at most 2 specific follow-up questions, "
                    "only populated when sufficient is false. "
                    "No markdown, raw JSON only."
                ),
            },
            {"role": "user", "content": f"Topic: {topic}\n\nNotes:\n{notes_text}"},
        ],
        model=REASON_MODEL,
        max_tokens=250,
    )
    data = json.loads(_strip_fences(raw))
    sufficient = bool(data.get("sufficient", True))
    additional = [q.strip() for q in data.get("additional_questions", []) if q.strip()][:2]
    return (not sufficient), additional


def synthesise(topic: str, notes: list[str]) -> str:
    """Write a structured Markdown research report from notes."""
    notes_text = "\n".join(f"- {n}" for n in notes)
    return _call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior research analyst. "
                    "Write a well-structured research report in Markdown with: "
                    "## Overview, ## Key Findings (bullet points), ## Analysis, ## Conclusion. "
                    "Base it ONLY on the provided notes. Be clear and professional. "
                    "When writing prices, spell out the currency code (e.g. 'USD 0.05') "
                    "instead of using a dollar sign, since $ is reserved for markdown math."
                ),
            },
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nResearch notes:\n{notes_text}",
            },
        ],
        model=REASON_MODEL,
        max_tokens=2200,
        temperature=0.4,
    )