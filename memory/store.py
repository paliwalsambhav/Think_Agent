"""
Long-term memory for the agent.

Short-term memory (the notes gathered *during* one research run) lives in
the LangGraph state and disappears when the run ends. This module is the
agent's long-term memory: every finished run is written to SQLite, and the
`recall` step at the start of a new run checks whether the agent already
looked into something similar before — the same way you'd glance at old
notes before re-researching a topic.

Deliberately dependency-free (stdlib `sqlite3`) so there's nothing extra to
provision when deploying.
"""

import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "agentloop.db"


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                notes_json TEXT NOT NULL,
                report TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def save_session(session_id: str, topic: str, plan: List[str], notes: List[dict], report: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, topic, plan_json, notes_json, report, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, topic, json.dumps(plan), json.dumps(notes), report, time.time()),
        )


def list_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, topic, created_at FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["plan"] = json.loads(d.pop("plan_json"))
        d["notes"] = json.loads(d.pop("notes_json"))
        return d


def delete_session(session_id: str) -> None:
    """Delete a single past run by id."""
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def delete_all_sessions() -> None:
    """
    Wipe all long-term memory. Used by the 'Clear history' control in the
    UI — e.g. before recording a clean demo video/screenshot, or if a user
    just wants a fresh start.
    """
    with _connect() as conn:
        conn.execute("DELETE FROM sessions")


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def find_similar_session(topic: str, threshold: float = 0.45) -> Optional[Dict[str, Any]]:
    """
    Keyword-overlap (Jaccard) match against past topics.

    This is intentionally simple rather than embedding/vector based: it
    keeps the project dependency-free, and it's transparent to explain in
    an interview ("the agent checks if it has researched something with
    enough word overlap before, and if so, gives itself that prior summary
    as a head start"). Swapping this for a vector store is a natural
    "next step" talking point.
    """
    target = _words(topic)
    if not target:
        return None

    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, topic, report, created_at FROM sessions ORDER BY created_at DESC LIMIT 200"
        ).fetchall()

    best, best_score = None, 0.0
    for row in rows:
        candidate = _words(row["topic"])
        if not candidate:
            continue
        overlap = len(target & candidate) / len(target | candidate)
        if overlap > best_score:
            best, best_score = dict(row), overlap

    if best and best_score >= threshold:
        return best
    return None