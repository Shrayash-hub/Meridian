"""
pipeline/nodes/architecture.py — Stage 2: Architecture Agent (Planning Node)

Reads the RequirementsSpec produced by Stage 1 and decomposes the implementation
into a list of independent Task objects. Task decomposition is separated from
requirements analysis so that specification remains focused purely on 'what'
needs to be built and verified, while architectural planning determines 'how'
the changes map to discrete codebase files and parallel execution units.

The tasks list produced here directly drives the Send API fan-out in Stage 3:
one Coding Agent is dispatched per task, with all coding agents running in parallel.
"""

import logging
import time
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.state import (
    PipelineState,
    ImplementationPlan,
    Task,
)
from pipeline.models import flash_model
from pipeline.tools.code_search import search_codebase

logger = logging.getLogger(__name__)

# Module-level constant: Defining the system prompt at module scope decouples
# prompt maintenance and versioning from function execution logic, making it
# easily inspectable, reusable, and testable across test suites.
ARCHITECTURE_SYSTEM_PROMPT = """You are a senior software architect decomposing a software requirements specification into an actionable, file-based implementation plan.

Carefully produce an implementation plan with these strict rules for each field:

For the tasks list:
- Decompose the implementation into independent subtasks.
- Each task corresponds to exactly one file that needs to be created or modified.
- Tasks must be genuinely independent — no task should depend on the output of another task being completed first.
- Maintain file-based granularity: one task per file with meaningful change. Do not split one file's changes across multiple tasks. Do not combine changes to different files into one task.
- Each task's instructions field must be a detailed, self-contained description of exactly what to implement in that file — specific enough that an autonomous coding agent can implement it without asking questions.
- task_id must be a short slug: "task-001", "task-002", etc.
- risk_level: "low" for adding new code/tests without altering existing logic; "medium" for modifying existing business logic; "high" for changes that could break existing functionality, public APIs, or security controls.
- dependencies: always an empty list ([]). All tasks run in parallel via Send API fan-out in the current pipeline; dependency ordering is not used.

For total_files_affected:
- The exact count of unique files across all tasks (must equal len(tasks)).

For estimated_complexity:
- "low": 1-2 files, straightforward additions or fixes.
- "medium": 3-5 files, or modifications to existing business logic.
- "high": 6+ files, cross-cutting concerns, or architectural refactoring.

For risk_summary:
- Write one clear paragraph summarizing the highest-risk aspects of this implementation, sensitive interfaces or files, and what coding agents must be careful not to regress.

Each task object must contain EXACTLY these field names — no substitutions, no additions:
- "task_id": string slug like "task-001"
- "title": string, short descriptive title
- "file_path": string, single file path — NOT a list, NOT "files", NOT "file"
- "instructions": string, detailed implementation instructions — NOT "description", NOT "desc"
- "dependencies": always an empty list []
- "risk_level": "low", "medium", or "high"

Using any other field name (e.g. "description", "files", "file", "details") will cause a fatal parsing error.
Respond with a valid JSON object that strictly follows the provided schema.
"""


def architecture_node(state: PipelineState) -> dict:
    time.sleep(15)
    """
    Stage 2: Architecture Agent (Planning Node).

    Reads the RequirementsSpec and decomposes the implementation
    into independent Task objects. The tasks list produced here
    drives the Send API fan-out in Stage 3 — one Coding Agent
    is spawned per task, all running in parallel.

    Reads:  state["requirements_spec"]
    Writes: state["implementation_plan"],
            state["tasks"],
            state["pipeline_status"]
    """
    spec = state.get("requirements_spec")

    if spec is None:
        logger.error("Architecture agent called with no requirements_spec")
        return {"pipeline_status": "architecture_failed: no requirements_spec"}

    # Format acceptance criteria as a numbered list
    criteria_lines = [f"{i}. {c}" for i, c in enumerate(spec.acceptance_criteria, 1)]
    criteria_text = "\n".join(criteria_lines) if criteria_lines else "None specified"

    # Format technical constraints as a numbered list
    if spec.constraints:
        constraints_lines = [f"{i}. {c}" for i, c in enumerate(spec.constraints, 1)]
        constraints_text = "\n".join(constraints_lines)
    else:
        constraints_text = "None specified"

    # Format files likely affected as a bulleted list
    if spec.files_likely_affected:
        files_lines = [f"- {f}" for f in spec.files_likely_affected]
        files_text = "\n".join(files_lines)
    else:
        files_text = "Not specified"

    human_message_content = (
        f"Requirements Summary:\n"
        f"{spec.summary}\n\n"
        f"Acceptance Criteria:\n"
        f"{criteria_text}\n\n"
        f"Technical Constraints:\n"
        f"{constraints_text}\n\n"
        f"Files Likely Affected:\n"
        f"{files_text}\n\n"
        f"Confidence Level: {spec.confidence}"
    )

    # Inject relevant existing code from the codebase index (RAG).
    # search_codebase returns "" when no index exists — graceful degradation.
    rag_context = search_codebase(query=spec.summary, k=3)
    if rag_context:
        human_message_content = human_message_content + "\n\n" + rag_context

    messages = [
        SystemMessage(content=ARCHITECTURE_SYSTEM_PROMPT),
        HumanMessage(content=human_message_content),
    ]

    structured_model = flash_model.with_structured_output(ImplementationPlan, method="json_mode")

    try:
        plan = structured_model.invoke(messages)

        logger.info(
            "Implementation plan created",
            extra={
                "task_count": len(plan.tasks),
                "complexity": plan.estimated_complexity,
                "files_affected": plan.total_files_affected,
            },
        )

        # Write both implementation_plan and top-level tasks for route_to_coding_agents
        return {
            "implementation_plan": plan,
            "tasks": plan.tasks,
            "pipeline_status": "architecture_complete",
        }

    except Exception as e:
        logger.error(
            "Architecture agent failed",
            extra={"error": str(e)},
        )
        return {
            "pipeline_status": f"architecture_failed: {str(e)}",
        }
