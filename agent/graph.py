"""
The agent's control flow, built as a LangGraph StateGraph.

    START
      |
    recall          -> check long-term memory for a similar past run
      |
    planner         -> LLM breaks the topic into sub-questions
      |
    research  <----+  -> for each sub-question: decides whether to search
      |            |     the web, calls Tavily if needed, writes a short answer
      | (loop while plan has items)
      v
    reflect   ------+  -> LLM checks notes for gaps; either adds more
      |                  sub-questions (loops back to research) or moves on
      v
    synthesize      -> LLM writes the final structured report
      |
    persist         -> save the run to SQLite (long-term memory)
      |
     END
"""

import gc
import json
import logging
import os
import uuid
from typing import Dict, List

from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent import llm, tools
from memory import store
from tracer.trace import traced, set_run_context
from tracer.db import log_eval_score

log = logging.getLogger(__name__)

# Cap on sub-questions per run â€” bounds worst-case memory regardless of topic
# complexity. Overridable via env var, same as MAX_SUBQUESTIONS in planner.
MAX_REFLECT_ADDITIONS = int(os.environ.get("MAX_REFLECT_ADDITIONS", "3"))


def _trace(state: AgentState, node: str, message: str) -> List[dict]:
    t = list(state.get("trace", []))
    t.append({"node": node, "message": message})
    return t


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def recall_node(state: AgentState) -> dict:
    topic = state["topic"]
    past = store.find_similar_session(topic)
    if past:
        context = (
            f"On {past['created_at']:.0f}, a similar topic (\"{past['topic']}\") "
            f"was researched. Prior summary:\n{past['report'][:1200]}"
        )
        msg = f"Found related prior research: \"{past['topic']}\" â€” using it as background context."
    else:
        context = None
        msg = "No related prior research found in long-term memory. Starting fresh."

    return {
        "recalled_context": context,
        "trace": _trace(state, "recall", msg),
    }


@traced(step_name="planner", model="llama-3.1-8b-instant")
def planner_node(state: AgentState) -> dict:
    llm.reset_usage()
    max_q = int(os.environ.get("MAX_SUBQUESTIONS", "4"))
    topic = state["topic"]

    prior_notes: List[str] = []
    if state.get("recalled_context"):
        prior_notes.append(state["recalled_context"])

    questions: List[str] = []
    try:
        questions = llm.plan(topic, prior_notes)
        questions = [q.strip() for q in questions if q.strip()][:max_q]
        if not questions:
            raise ValueError("LLM returned empty question list")
    except Exception as e:
        log.error("Planner LLM failed (%s) â€” using fallback questions", e)
        questions = [
            f"What is {topic} and why does it matter?",
            f"What are the key challenges and opportunities in {topic}?",
            f"What are the latest developments and future trends in {topic}?",
        ]

    msg = "Planned " + str(len(questions)) + " sub-questions: " + "; ".join(questions)
    return {
        "plan": questions,
        "notes": [],
        "completed_questions": [],
        "reflect_iterations": 0,
        "trace": _trace(state, "planner", msg),
        **llm.get_usage(),
    }


@traced(step_name="research", model="llama-3.1-8b-instant")
def research_node(state: AgentState) -> dict:
    llm.reset_usage()
    plan = list(state.get("plan", []))
    if not plan:
        return {}

    question = plan.pop(0)
    completed = list(state.get("completed_questions", [])) + [question]

    sources: List[str] = []

    try:
        if llm.should_search(question):
            results = tools.web_search(question)
            sources = [r["url"] for r in results if r.get("url")][:5]
            search_results_text = "\n\n".join(
                f"Title: {r.get('title', '')}\n{r.get('content', r.get('snippet', ''))}"
                for r in results
            )
            answer = llm.answer_from_context(question, search_results_text)
            trace_msg = f"Researching (web): {question}"
        else:
            answer = llm.answer_from_knowledge(question)
            trace_msg = f"Researching (knowledge): {question}"
    except Exception as e:
        log.error("Research node failed for question %r: %s", question, e)
        answer = "Could not retrieve an answer for this question."
        trace_msg = f"Researching (failed): {question}"

    note = {"question": question, "answer": answer.strip(), "sources": sources}
    notes = list(state.get("notes", [])) + [note]

    return {
        "plan": plan,
        "notes": notes,
        "completed_questions": completed,
        "trace": _trace(state, "research", trace_msg),
        **llm.get_usage(),
    }


def route_after_research(state: AgentState) -> str:
    return "research" if state.get("plan") else "reflect"


@traced(step_name="reflect", model="llama-3.1-8b-instant")
def reflect_node(state: AgentState) -> dict:
    llm.reset_usage()
    iterations = state.get("reflect_iterations", 0)
    max_iterations = state.get("max_reflect_iterations", 2)

    if iterations >= max_iterations:
        return {
            "decision": "done",
            "trace": _trace(state, "reflect", "Reached reflection cap â€” moving to synthesis."),
            **llm.get_usage(),
        }

    notes = state.get("notes", [])
    notes_list = [f"Q: {n['question']} A: {n['answer']}" for n in notes]

    try:
        needs_more, additional = llm.reflect(state["topic"], notes_list)
        # llm.reflect() returns (bool, list[str])
        additional = [q.strip() for q in additional if q.strip()][:MAX_REFLECT_ADDITIONS]
    except Exception as e:
        log.error("Reflect node failed: %s", e)
        needs_more, additional = False, []

    if not needs_more or not additional:
        return {
            "decision": "done",
            "trace": _trace(state, "reflect", "Notes judged sufficient â€” moving to synthesis."),
            **llm.get_usage(),
        }

    msg = "Found gaps â€” adding follow-up questions: " + "; ".join(additional)
    return {
        "decision": "more_research",
        "plan": additional,
        "reflect_iterations": iterations + 1,
        "trace": _trace(state, "reflect", msg),
        **llm.get_usage(),
    }


def route_after_reflect(state: AgentState) -> str:
    return "research" if state.get("decision") == "more_research" else "synthesize"


@traced(step_name="synthesize", model="llama-3.1-8b-instant")
def synthesize_node(state: AgentState) -> dict:
    llm.reset_usage()
    notes = state.get("notes", [])
    notes_list = [
        f"Q: {n['question']}\nA: {n['answer']}\nSources: {', '.join(n['sources']) or 'none'}"
        for n in notes
    ]

    if state.get("recalled_context"):
        notes_list.append(f"Background context:\n{state['recalled_context']}")

    try:
        report = llm.synthesise(state["topic"], notes_list)
    except Exception as e:
        log.error("Synthesize node failed: %s", e)
        report = f"# {state['topic']}\n\n" + "\n\n".join(
            f"**{n['question']}**\n{n['answer']}" for n in notes
        )

    all_sources = []
    seen = set()
    for n in notes:
        for url in n.get("sources", []):
            if url not in seen:
                seen.add(url)
                all_sources.append(url)

    return {
        "final_report": report,
        "all_sources": all_sources,
        "trace": _trace(state, "synthesize", "Final report written."),
        **llm.get_usage(),
    }


def persist_node(state: AgentState) -> dict:
    try:
        store.save_session(
            session_id=state["session_id"],
            topic=state["topic"],
            plan=state.get("completed_questions", []),
            notes=state.get("notes", []),
            report=state.get("final_report", ""),
        )
    except Exception as e:
        log.error("Persist node failed: %s", e)
    return {"trace": _trace(state, "persist", "Saved this run to long-term memory.")}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("recall",     recall_node)
    g.add_node("planner",    planner_node)
    g.add_node("research",   research_node)
    g.add_node("reflect",    reflect_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("persist",    persist_node)

    g.add_edge(START, "recall")
    g.add_edge("recall", "planner")
    g.add_edge("planner", "research")
    g.add_conditional_edges("research", route_after_research, {"research": "research", "reflect": "reflect"})
    g.add_conditional_edges("reflect",  route_after_reflect,  {"research": "research", "synthesize": "synthesize"})
    g.add_edge("synthesize", "persist")
    g.add_edge("persist", END)

    return g.compile()


graph = build_graph()


def run(topic: str):
    """Generator that yields (node_name, snapshot) as each node finishes."""
    run_id = set_run_context(agent_name="Think_Agent")

    init: AgentState = {
        "session_id":             str(uuid.uuid4()),
        "topic":                  topic,
        "recalled_context":       None,
        "plan":                   [],
        "completed_questions":    [],
        "notes":                  [],
        "reflect_iterations":     0,
        "max_reflect_iterations": 2,
        "decision":               "",
        "final_report":           "",
        "trace":                  [],
    }
    try:
        for event in graph.stream(init):
            for node_name, snapshot in event.items():
                yield node_name, snapshot
    finally:
        # Release this run's notes/sources/report before the worker picks up
        # the next request â€” matters on Render's free 512MB instances.
        gc.collect()

    # Placeholder eval score for now â€” swap in a real metric (e.g. report
    # length/quality check, LLM-as-judge, keyword coverage) once you have one.
    log_eval_score(run_id=run_id, agent_name="Think_Agent", score=0.8)