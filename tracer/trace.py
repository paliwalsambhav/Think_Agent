"""
The @traced decorator — wrap any agent function/LLM call to log latency,
token usage, and estimated cost to the AgentOps Postgres db.
"""

import functools
import time
import contextvars

from tracer.db import init_db, insert_trace, new_run_id

COST_PER_1M_TOKENS = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "default": {"input": 0.50, "output": 0.70},
}

_run_id_var = contextvars.ContextVar("run_id", default=None)
_agent_name_var = contextvars.ContextVar("agent_name", default="unnamed_agent")


def set_run_context(agent_name, run_id=None):
    init_db()
    rid = run_id or new_run_id()
    _run_id_var.set(rid)
    _agent_name_var.set(agent_name)
    return rid


def estimate_cost(model, input_tokens, output_tokens):
    rates = COST_PER_1M_TOKENS.get(model, COST_PER_1M_TOKENS["default"])
    return (input_tokens / 1_000_000) * rates["input"] + (
        output_tokens / 1_000_000
    ) * rates["output"]


def traced(step_name, model="default"):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            run_id = _run_id_var.get() or set_run_context(_agent_name_var.get())
            agent_name = _agent_name_var.get()
            started_at = time.time()
            status = "success"
            error_msg = None
            input_tokens = output_tokens = None
            cost = None

            try:
                result = fn(*args, **kwargs)
                if isinstance(result, dict):
                    input_tokens = result.get("input_tokens")
                    output_tokens = result.get("output_tokens")
                    if input_tokens is not None and output_tokens is not None:
                        cost = estimate_cost(model, input_tokens, output_tokens)
                return result
            except Exception as e:
                status = "error"
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.time() - started_at) * 1000
                insert_trace(
                    run_id=run_id, agent_name=agent_name, step_name=step_name,
                    started_at=started_at, duration_ms=duration_ms,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    estimated_cost_usd=cost, status=status, error=error_msg,
                )

        return wrapper

    return decorator
