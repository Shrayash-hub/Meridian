"""
pipeline/graph.py — Graph construction and compilation.

Wires all eight pipeline nodes into a complete, compilable LangGraph
StateGraph. At this stage all nodes are stubs that return empty dicts;
the topology is correct and the graph runs end-to-end without errors.

Stages implemented here (stubs):
  1. Requirements Agent    — sequential
  2. Architecture Agent    — sequential; planning node
  3. Coding Agents         — parallel fan-out via Send API (one per Task)
  4. Test-Fix Loop         — iterative loop with conditional exit
  5. Review Agents         — parallel fan-out via Send API (three reviewers)
  6. Review Synthesizer    — sequential; convergence point
  7. Human-in-the-Loop     — interrupt() gate (checkpointer added later)
  8. PR Author             — sequential; terminal node

Pending (future steps):
  - Checkpointer for HITL persistence (required for interrupt() to work).
  - Real node logic replacing each stub.
  - Memory update post-pipeline (runs outside the graph).
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from pipeline.state import PipelineState, CodeChange
from pipeline.nodes.requirements import requirements_node
from pipeline.nodes.architecture import architecture_node
from pipeline.nodes.coding import coding_node, CodingAgentState
from pipeline.nodes.testing import testing_node
from pipeline.nodes.review import (
    security_review_node,
    performance_review_node,
    quality_review_node,
)
from pipeline.nodes.synthesizer import synthesizer_node
from pipeline.nodes.hitl import hitl_node
from pipeline.nodes.pr_author import pr_author_node
from pipeline.nodes.memory_update import memory_update_node
from memory.store import store as long_term_store
import time


# ---------------------------------------------------------------------------
# Node name constants — defined once, referenced everywhere.
# Avoids hardcoded strings in edges and routing functions.
# ---------------------------------------------------------------------------

REQUIREMENTS       = "requirements"
ARCHITECTURE       = "architecture"
CODING             = "coding"
TESTING            = "testing"
SECURITY_REVIEW    = "security_review"
PERFORMANCE_REVIEW = "performance_review"
QUALITY_REVIEW     = "quality_review"
SYNTHESIZER        = "synthesizer"
HITL               = "hitl"
PR_AUTHOR          = "pr_author"
MEMORY_UPDATE      = "memory_update"


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_to_coding_agents(state: PipelineState) -> list[Send]:
    """
    Fan-out: dispatch one coding agent per Task produced by the Architecture
    Agent. Each Send carries a CodingAgentState slice containing exactly the
    task, the requirements spec, and an empty code_changes accumulator.

    Each agent's output is merged back into PipelineState["code_changes"]
    by the operator.add reducer defined on that field in state.py.
    """
    tasks = state.get("tasks") or []

    # ponytail: guard against an empty task list — if the Architecture Agent
    # produced no tasks (e.g. the issue required no file changes), skip
    # coding and proceed directly to testing. Ceiling: this bypasses the
    # coding stage entirely. When architecture_node is implemented it should
    # never produce an empty task list for a real issue; this guard is only
    # a safety net for stub runs and degenerate inputs.
    if not tasks:
        return [Send(TESTING, state)]

    return [
        Send(
            CODING,
            CodingAgentState(
                task=task,
                requirements_spec=state.get("requirements_spec"),
                code_changes=[],
            ),
        )
        for task in tasks
    ]


def route_to_review_agents(state: PipelineState) -> list[Send]:
    """
    Fan-out: dispatch all three review agents simultaneously. Each Send
    carries a ReviewAgentState slice with its reviewer identity, the full
    set of code changes, requirements context, and an empty findings
    accumulator.

    Each agent's findings are merged into PipelineState["review_findings_raw"]
    by the operator.add reducer defined on that field in state.py.
    """
    code_changes: list[CodeChange] = state.get("code_changes") or []
    requirements_spec = state.get("requirements_spec")

    # NOTE: time.sleep() here does NOT stagger parallel dispatch.
    # LangGraph receives all Send()s at once and fires them simultaneously.
    # To rate-limit reviewer calls, the sleep belongs inside each review
    # node function, not here. See review.py preamble.
    sends = []
    sends.append(
        Send(
            SECURITY_REVIEW,
            {
                "reviewer_type": "security",
                "code_changes": code_changes,
                "requirements_spec": requirements_spec,
                "review_findings_raw": [],
            },
        )
    )
    sends.append(
        Send(
            PERFORMANCE_REVIEW,
            {
                "reviewer_type": "performance",
                "code_changes": code_changes,
                "requirements_spec": requirements_spec,
                "review_findings_raw": [],
            },
        )
    )
    sends.append(
        Send(
            QUALITY_REVIEW,
            {
                "reviewer_type": "quality",
                "code_changes": code_changes,
                "requirements_spec": requirements_spec,
                "review_findings_raw": [],
            },
        )
    )
    return sends



def route_after_testing(state: PipelineState) -> str | list[Send]:
    """
    Controls the test-fix loop and the exit to the review fan-out.

    Returns a string (TESTING) to loop back when tests are still failing
    and the retry budget has not been exhausted.

    Returns a list[Send] to fan out to all three review agents once the
    loop exits — either because tests passed or the budget is exhausted
    (partial results are still reviewed so the engineer sees the full picture).

    LangGraph supports returning either a string or a list of Send objects
    from the same conditional edge routing function.
    """
    test_results = state.get("test_results")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 5)

    # Guard: if no code changes exist, exit loop immediately.
    # Looping with no code produces nothing and spins forever.
    # ponytail: this occurs when upstream agents fail (e.g. quota
    # exhaustion). Pipeline proceeds to review with empty changes.
    if not state.get("code_changes"):
        return route_to_review_agents(state)

    # Loop back: tests ran, failed, and we still have budget for a fix.
    # Pydantic object after live run vs plain dict after checkpoint resume.
    # The 'Deserializing unregistered type TestReport' warning means
    # test_results arrives as a dict after HITL resume — .final_status
    # would raise AttributeError and silently exit the loop.
    _final_status = (
        getattr(test_results, "final_status", None)
        or (test_results.get("final_status") if isinstance(test_results, dict) else None)
    )
    if (
        test_results is not None
        and _final_status != "passed"
        and revision_count < max_revisions
    ):
        return TESTING

    # Exit: either tests passed, or the budget is exhausted.
    # Fan out to all three reviewers simultaneously.
    return route_to_review_agents(state)


def route_after_hitl(state: PipelineState) -> str:
    """
    Routes execution after the human engineer makes a decision.

    "approve"         → proceed to PR Author (Stage 8).
    "request_changes" → loop back to Coding Agents (Stage 3) for another
                        revision cycle. revision_count is incremented inside
                        hitl_node when fully implemented.
    "reject"          → terminate the pipeline.
    anything else     → terminate the pipeline.

    ponytail: the fallthrough-to-END default is intentional. An unrecognised
    decision value should not leave the graph in an infinite wait state.
    Ceiling: when hitl_node is implemented, it should validate the decision
    value before writing it to state.
    """
    decision = state.get("human_decision", "")

    if decision == "approve":
        return PR_AUTHOR
    if decision == "request_changes":
        return CODING
    # "reject" or any unrecognised value — terminate safely.
    return END


def route_after_pr(state: PipelineState) -> str:
    """Route to memory update on success, else END."""
    status = state.get("pipeline_status", "")
    if status == "pr_opened":
        return MEMORY_UPDATE
    # ponytail: on pr_failed or pr_author_failed, skip memory.
    # Memory update is best-effort — pipeline result is preserved.
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

builder = StateGraph(PipelineState)

# Register all nodes.
builder.add_node(REQUIREMENTS,       requirements_node)
builder.add_node(ARCHITECTURE,       architecture_node)
builder.add_node(CODING,             coding_node)
builder.add_node(TESTING,            testing_node)
builder.add_node(SECURITY_REVIEW,    security_review_node)
builder.add_node(PERFORMANCE_REVIEW, performance_review_node)
builder.add_node(QUALITY_REVIEW,     quality_review_node)
builder.add_node(SYNTHESIZER,        synthesizer_node)
builder.add_node(HITL,               hitl_node)
builder.add_node(PR_AUTHOR,          pr_author_node)
builder.add_node(MEMORY_UPDATE,      memory_update_node)

# Entry point.
builder.set_entry_point(REQUIREMENTS)

# Stage 1 → Stage 2: sequential.
builder.add_edge(REQUIREMENTS, ARCHITECTURE)

# Stage 2 → Stage 3: Send API fan-out — one coding agent per Task.
builder.add_conditional_edges(ARCHITECTURE, route_to_coding_agents)

# Stage 3 → Stage 4: all coding agents converge on the testing node.
# Every parallel coding branch feeds into this single edge, so testing
# only starts once all coding agents have completed (LangGraph guarantees
# this via the fan-in on CODING → TESTING).
builder.add_edge(CODING, TESTING)

# Stage 4 loop + Stage 4 → Stage 5 fan-out:
# route_after_testing returns TESTING (string) to loop, or a list[Send]
# to fan out to all three reviewers simultaneously.
builder.add_conditional_edges(TESTING, route_after_testing)

# Stage 5 → Stage 6: all three reviewers converge on the synthesizer.
builder.add_edge(SECURITY_REVIEW,    SYNTHESIZER)
builder.add_edge(PERFORMANCE_REVIEW, SYNTHESIZER)
builder.add_edge(QUALITY_REVIEW,     SYNTHESIZER)

# Stage 6 → Stage 7: sequential.
builder.add_edge(SYNTHESIZER, HITL)

# Stage 7 → conditional: approve → PR_AUTHOR, request_changes → CODING, reject → END.
builder.add_conditional_edges(
    HITL,
    route_after_hitl,
    {
        PR_AUTHOR: PR_AUTHOR,
        CODING:    CODING,
        END:       END,
    },
)

# Stage 8 → conditional: pr_opened → memory_update → END, else → END.
builder.add_conditional_edges(PR_AUTHOR, route_after_pr)
builder.add_edge(MEMORY_UPDATE, END)

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

# MemorySaver persists graph state across interrupt()/resume cycles.
# Required for HITL gate (Stage 7) to function.
# ponytail: switch to SqliteSaver or PostgresCheckpointer in
# production for state that survives process restarts.
checkpointer = MemorySaver()
pipeline = builder.compile(
    checkpointer=checkpointer,
    store=long_term_store,
)
