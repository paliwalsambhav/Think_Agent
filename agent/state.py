"""
Shared state that flows through every node of the research agent graph.

This is the agent's *working memory* for a single run: it accumulates as the
graph moves from planning -> researching -> reflecting -> writing, and every
node reads/writes a slice of it. LangGraph persists this dict between node
calls, which is what lets the agent "remember" earlier steps without us
wiring up our own plumbing.
"""

from typing import TypedDict, List, Dict, Optional, Literal


class Note(TypedDict):
    question: str
    answer: str
    sources: List[str]


class TraceEvent(TypedDict):
    node: str
    message: str


class AgentState(TypedDict, total=False):
    # ---- input ----
    topic: str
    session_id: str

    # ---- long-term memory recall (from SQLite, across past runs) ----
    recalled_context: Optional[str]

    # ---- planning ----
    plan: List[str]              # queue of sub-questions still to research
    completed_questions: List[str]

    # ---- short-term memory (this run only) ----
    notes: List[Note]

    # ---- control flow ----
    reflect_iterations: int
    max_reflect_iterations: int
    decision: Literal["more_research", "done"]

    # ---- output ----
    final_report: str

    # ---- observability: lets the API stream "what is the agent doing" ----
    trace: List[TraceEvent]
