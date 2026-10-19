"""
pipeline/nodes/coding.py — Stage 3: Coding Agent (worker node)

This is the worker in the orchestrator-worker pattern. The Architecture Agent
(orchestrator) produces a list of Tasks; the LangGraph Send API spawns one
instance of this node per Task, all running in parallel.

Each invocation:
  1. Receives exactly one Task via CodingAgentState (not the full PipelineState)
  2. Calls the LLM to produce complete file content for that task
  3. Returns a CodeChange wrapped in a list so the operator.add reducer can
     accumulate results from all parallel instances into PipelineState["code_changes"]

Writes complete file content (not diffs). One instance owns exactly one file.
"""

import operator
import logging
import time
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.state import (
    RequirementsSpec,
    Task,
    CodeChange,
)
from pipeline.models import coding_model
import time
logger = logging.getLogger(__name__)


class CodingAgentState(TypedDict):
    """
    State slice passed to each coding agent instance via Send API.
    Each instance receives exactly one task and operates independently.
    The code_changes field uses operator.add so results from all
    parallel instances accumulate into PipelineState["code_changes"].
    """
    task: Task
    requirements_spec: RequirementsSpec
    # Reducer merges each agent's [CodeChange] into the parent list on fan-in.
    code_changes: Annotated[list[CodeChange], operator.add]


CODING_SYSTEM_PROMPT = """You are a senior software engineer implementing a specific, well-defined coding task.

Your job is to write complete, production-quality file content for exactly one file.

Rules you must follow:

1. Write COMPLETE file content — every import, every function, every class, every line.
   If the file already exists and you are modifying it, include the unchanged parts too.
   Do not write partial implementations, pseudocode, TODOs, or placeholder comments.
   The content must be immediately usable as-is.

2. Write ONLY the content for the file specified in "File to Implement".
   Do not generate content for any other file.

3. Follow the constraints and patterns described in the requirements context.
   Do not introduce new dependencies, patterns, or abstractions beyond what the task requires.

4. The explanation field must describe in plain English:
   - What was changed and why
   - Any assumptions made about the existing codebase
   - Any risks or edge cases the reviewer should check

The response must be a JSON object with EXACTLY these four field names — no others:
- "task_id": string — copy exactly from the input "Task ID" field
- "file_path": string — copy exactly from the input "File to Implement" field
- "content": string — the complete file content
- "explanation": string — plain English description of what was changed and why

Do NOT use any other field names (not "code", not "file", not "description", not "changes").
Respond with a valid JSON object that strictly follows the provided schema.

FastAPI HTTP 400 vs 422 — mandatory rule:
When an issue or task requires HTTP 400 for invalid input:
  - Do NOT use Pydantic field validators, EmailStr, or any Pydantic type
    that validates format. These cause FastAPI to return HTTP 422, not 400.
  - Instead, declare the field as a plain `str` in the Pydantic request model,
    then validate it manually in the route handler using re.match, and raise
    HTTPException(status_code=400, detail="...") directly.
  - In tests, assert response.status_code == 400, not 422.
Example (correct):
    class RegisterRequest(BaseModel):
        email: str  # plain str, NOT EmailStr
    @app.post("/api/register", status_code=201)
    def register(body: RegisterRequest):
        if not re.match(r"^[^@]+@[^@]+\\.[^@]+$", body.email):
            raise HTTPException(status_code=400, detail="Invalid email format")

CRITICAL RULE — HTTP 400 vs HTTP 422:
When an issue requires returning HTTP 400 for validation errors, do NOT
use Pydantic field validators, EmailStr, or @field_validator. These cause
FastAPI to return HTTP 422 Unprocessable Entity, not HTTP 400.

Instead, keep the field as a plain `str` and validate manually in the
route handler:

    import re
    from fastapi import HTTPException

    @router.post("/endpoint")
    def handler(request: RequestModel):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", request.email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        ...

This pattern returns HTTP 400 exactly as required. Use it whenever the
issue acceptance criteria says "return HTTP 400" for a validation error.
"""


def coding_node(state: CodingAgentState) -> dict:
    # Guard against empty state slice from empty task list
    if not state.get("task"):
        logger.warning("coding_node called with no task, skipping")
        return {"code_changes": []}
    time.sleep(15)

    # Brief pause to avoid Groq free-tier rate limits when pipeline
    # stages fire in rapid succession.
    # ponytail: remove this when switching to a paid tier or
    # a provider without per-minute token limits.
    # time.sleep(5)
    """
    Stage 3: Coding Agent (Worker Node).

    Receives one Task via Send API and generates complete file content
    for that task. Runs in parallel with other coding_node instances —
    one instance per task produced by the Architecture Agent.

    Writes complete file content, not diffs. Each instance owns exactly
    one file. Results accumulate into PipelineState["code_changes"]
    via the operator.add reducer.

    Reads:  state["task"], state["requirements_spec"]
    Writes: state["code_changes"] (appends one CodeChange)
    """
    task = state["task"]
    spec = state.get("requirements_spec")

    # Guard: if requirements_spec is absent (upstream failure), nothing to do.
    if spec is None:
        logger.error(
            "Coding agent called with no requirements_spec",
            extra={"task_id": task.task_id},
        )
        return {"code_changes": []}

    # Build numbered acceptance criteria list
    criteria_lines = [f"{i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)]
    criteria_text = "\n".join(criteria_lines) if criteria_lines else "None specified."

    # Build numbered constraints list
    if spec.constraints:
        constraints_lines = [f"{i}. {c}" for i, c in enumerate(spec.constraints, 1)]
        constraints_text = "\n".join(constraints_lines)
    else:
        constraints_text = "None specified."

    human_message_content = (
        f"Task ID: {task.task_id}\n"
        f"Task Title: {task.title}\n"
        f"File to Implement: {task.file_path}\n"
        f"Risk Level: {task.risk_level}\n"
        f"\nImplementation Instructions:\n"
        f"{task.instructions}\n"
        f"\nRequirements Context:\n"
        f"Summary: {spec.summary}\n"
        f"\nAcceptance Criteria:\n"
        f"{criteria_text}\n"
        f"\nTechnical Constraints:\n"
        f"{constraints_text}"
    )

    # json_mode required for reliable structured output across providers (Decision 007)
    structured_model = coding_model.with_structured_output(CodeChange, method="json_mode")

    try:
        change = structured_model.invoke([
            SystemMessage(content=CODING_SYSTEM_PROMPT),
            HumanMessage(content=human_message_content),
        ])

        logger.info(
            "Code change generated",
            extra={
                "task_id": task.task_id,
                "file_path": task.file_path,
                "content_length": len(change.content),
            },
        )

        # Wrap in a list — operator.add reducer expects list[CodeChange],
        # not a single CodeChange. Each parallel agent contributes one item.
        return {"code_changes": [change]}

    except Exception as e:
        logger.error(
            f"Coding agent failed: {e}",
            exc_info=True,
            extra={
                "task_id": task.task_id,
                "file_path": task.file_path,
                "error": str(e),
            },
        )
        # Return empty list so the reducer has something to merge.
        # Pipeline continues — the testing stage works with whatever
        # changes were successfully produced by other parallel agents.
        # Worker nodes do not set pipeline_status; that is the
        # orchestrator's responsibility.
        return {"code_changes": []}
