"""
Think_Agent -- FastAPI deployment (replaces the Streamlit frontend)

Reuses agent/graph.py, agent/llm.py, agent/tools.py, agent/state.py, and
memory/store.py UNCHANGED. This file only replaces streamlit_app.py.

Run locally:
    uvicorn main:app --reload

Deploy:
    Render only. Set GROQ_API_KEY, TAVILY_API_KEY, and optionally
    GROQ_FAST_MODEL, GROQ_REASON_MODEL, MAX_SUBQUESTIONS, SUPABASE_DB_*
    directly in the Render service's Environment tab -- agent/llm.py,
    agent/tools.py, and tracer/db.py all read these straight from
    os.environ.
"""

import json
import logging

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.graph import run as agent_run
from memory import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s")
log = logging.getLogger(__name__)

store.init_db()

app = FastAPI(title="Think_Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    topic: str


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/run")
async def run_research(req: RunRequest):
    topic = (req.topic or "").strip()

    async def stream():
        if not topic:
            yield _sse("error", {"message": "Topic cannot be empty."})
            return
        try:
            for node_name, snapshot in agent_run(topic):
                payload = {"node": node_name}
                if node_name == "planner":
                    payload["plan"] = snapshot.get("plan", [])
                elif node_name == "research":
                    trace = snapshot.get("trace", [])
                    if trace:
                        payload["message"] = trace[-1]["message"]
                elif node_name == "reflect":
                    payload["decision"] = snapshot.get("decision", "")
                elif node_name == "synthesize":
                    payload["report"] = snapshot.get("final_report", "")
                    payload["sources"] = snapshot.get("all_sources", [])
                yield _sse("node", payload)
            yield _sse("done", {})
        except Exception as exc:
            log.error("Agent run failed: %s", exc, exc_info=True)
            yield _sse("error", {"message": "An error occurred during research. Please try again in a moment."})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/sessions")
async def list_sessions():
    try:
        return store.list_sessions(limit=20)
    except Exception as e:
        log.error("Could not load history: %s", e)
        return []


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    full = store.get_session(session_id)
    if not full:
        return {"error": "not found"}
    return full


@app.delete("/api/sessions")
async def clear_sessions():
    store.delete_all_sessions()
    return {"ok": True}


app.mount("/", StaticFiles(directory="static", html=True), name="static")