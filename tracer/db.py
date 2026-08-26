"""
Postgres (Supabase) storage layer for AgentOps.

Gracefully disables itself when SUPABASE_DB_* env vars aren't set (e.g. on
Render, where this is optional observability, not core functionality). All
public functions become safe no-ops in that case instead of raising --
tracing is a nice-to-have; a missing Supabase connection should never take
down an actual research run.
"""

import logging
import os
import time
import uuid
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

_REQUIRED_VARS = ("SUPABASE_DB_HOST", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD")


def _tracing_enabled() -> bool:
    return all(os.environ.get(v) for v in _REQUIRED_VARS)


def _connect():
    return psycopg2.connect(
        host=os.environ["SUPABASE_DB_HOST"],
        port=os.environ.get("SUPABASE_DB_PORT", "5432"),
        dbname=os.environ.get("SUPABASE_DB_NAME", "postgres"),
        user=os.environ["SUPABASE_DB_USER"],
        password=os.environ["SUPABASE_DB_PASSWORD"],
        sslmode="require",
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def init_db():
    if not _tracing_enabled():
        log.info("Supabase env vars not set -- AgentOps tracing disabled for this run.")
        return
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS traces (
                        id UUID PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        step_name TEXT NOT NULL,
                        started_at DOUBLE PRECISION NOT NULL,
                        duration_ms DOUBLE PRECISION NOT NULL,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        estimated_cost_usd DOUBLE PRECISION,
                        status TEXT NOT NULL,
                        error TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS eval_runs (
                        id UUID PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        score DOUBLE PRECISION NOT NULL,
                        created_at DOUBLE PRECISION NOT NULL,
                        notes TEXT
                    )
                """)
            conn.commit()
    except Exception as e:
        log.warning("AgentOps tracing DB init failed (%s) -- continuing without tracing.", e)


def new_run_id() -> str:
    return str(uuid.uuid4())[:8]


def insert_trace(
    run_id, agent_name, step_name, started_at, duration_ms,
    input_tokens, output_tokens, estimated_cost_usd, status, error=None,
):
    if not _tracing_enabled():
        return
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO traces
                       (id, run_id, agent_name, step_name, started_at, duration_ms,
                        input_tokens, output_tokens, estimated_cost_usd, status, error)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (str(uuid.uuid4()), run_id, agent_name, step_name, started_at,
                     duration_ms, input_tokens, output_tokens, estimated_cost_usd,
                     status, error),
                )
            conn.commit()
    except Exception as e:
        log.warning("AgentOps insert_trace failed (%s) -- continuing.", e)


def log_eval_score(run_id, agent_name, score, notes=""):
    if not _tracing_enabled():
        return
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO eval_runs (id, run_id, agent_name, score, created_at, notes)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (str(uuid.uuid4()), run_id, agent_name, score, time.time(), notes),
                )
            conn.commit()
    except Exception as e:
        log.warning("AgentOps log_eval_score failed (%s) -- continuing.", e)


def fetch_traces(limit=500):
    if not _tracing_enabled():
        return []
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM traces ORDER BY started_at DESC LIMIT %s", (limit,))
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.warning("AgentOps fetch_traces failed (%s).", e)
        return []


def fetch_eval_runs(agent_name=None, limit=500):
    if not _tracing_enabled():
        return []
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                if agent_name:
                    cur.execute(
                        "SELECT * FROM eval_runs WHERE agent_name = %s ORDER BY created_at DESC LIMIT %s",
                        (agent_name, limit),
                    )
                else:
                    cur.execute("SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT %s", (limit,))
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.warning("AgentOps fetch_eval_runs failed (%s).", e)
        return []


@contextmanager
def ensure_db():
    init_db()
    yield