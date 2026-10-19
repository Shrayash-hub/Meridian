"""
pipeline/state.py — Data contracts for the entire SDLC pipeline.

This file is the single source of truth for all data that flows through
the pipeline. It defines:

  1. Pydantic schemas — structured output types used by each agent.
     The LLM is instructed to produce JSON that validates against these
     schemas. Field descriptions are written as instructions to the model.

  2. PipelineState — the LangGraph TypedDict that accumulates results
     across all eight pipeline stages. Fields that multiple parallel
     agents write to use the `operator.add` reducer so that concurrent
     writes are merged rather than overwritten.

Nothing in this file imports from anywhere else in the project.
All other modules import FROM this file.
"""

import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage 1 — Requirements Agent output
# ---------------------------------------------------------------------------

class RequirementsSpec(BaseModel):
    """
    Structured interpretation of a GitHub issue produced by the Requirements
    Agent. This spec drives every downstream stage. Every field must be
    complete enough for the Architecture Agent to plan an implementation
    without reading the original issue.
    """

    summary: str = Field(
        description=(
            "Write a single paragraph that restates what needs to be built in "
            "precise engineering terms. Do not copy the issue verbatim — "
            "translate it into an unambiguous implementation objective."
        )
    )
    acceptance_criteria: list[str] = Field(
        description=(
            "List specific, testable conditions that the implementation must "
            "satisfy for the issue to be considered resolved. Each item must "
            "be independently verifiable by running a test or inspecting "
            "observable behaviour. Phrase each item as 'Given X, when Y, "
            "then Z'."
        )
    )
    constraints: list[str] = Field(
        description=(
            "List explicit constraints the implementation must respect. "
            "Include things like: must reuse an existing pattern, must not "
            "break a specific interface, must stay within a performance "
            "budget, must remain backwards-compatible. Each constraint must "
            "name the specific thing being constrained."
        )
    )
    files_likely_affected: list[str] = Field(
        description=(
            "List the file paths (relative to the project root) that are "
            "likely to require changes based on codebase RAG retrieval. "
            "Include files that export interfaces the change depends on. "
            "Do not include files you are merely guessing at."
        )
    )
    ambiguities: list[str] = Field(
        description=(
            "List any points in the issue that are unclear, contradictory, "
            "or underspecified. If the issue is fully clear, return an empty "
            "list. Each item should name the specific ambiguity and explain "
            "why it matters for implementation."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "Rate your confidence that this spec is complete enough to "
            "proceed to implementation without human clarification. "
            "'high' means no ambiguities and all constraints are clear. "
            "'medium' means minor ambiguities exist but implementation can "
            "proceed with reasonable assumptions. "
            "'low' means the issue is too underspecified to implement "
            "correctly — a HITL interrupt should be triggered."
        )
    )


# ---------------------------------------------------------------------------
# Stage 2 — Architecture Agent output
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """
    A single, atomic implementation subtask produced by the Architecture
    Agent. Each Task is executed by exactly one Coding Agent. Tasks are
    designed to be independent so they can run in parallel.
    """

    task_id: str = Field(
        description=(
            "A unique identifier for this task. Use a short, slug-style "
            "string, e.g. 'add-validation-to-user-model'. Must be unique "
            "within the implementation plan."
        )
    )
    title: str = Field(
        description=(
            "A short, imperative-mood description of what this task "
            "accomplishes, e.g. 'Add input validation to UserModel'. "
            "Maximum 80 characters."
        )
    )
    file_path: str = Field(
        description=(
            "The path (relative to project root) of the single file this "
            "task creates or modifies. Each task operates on exactly one "
            "file. If multiple files must change, create multiple tasks."
        )
    )
    instructions: str = Field(
        description=(
            "Detailed, self-contained instructions for the Coding Agent "
            "assigned to this task. Describe exactly what change to make, "
            "what interfaces to call, what patterns to follow, and what "
            "the change must not break. The Coding Agent will only see this "
            "field and the RequirementsSpec — do not assume it has other "
            "context."
        )
    )
    dependencies: list[str] = Field(
        description=(
            "List the task_ids of any tasks that must complete before this "
            "task starts. Return an empty list if this task has no "
            "dependencies. Note: in the current implementation all tasks "
            "run in parallel — this field is reserved for future sequential "
            "scheduling."
        )
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        description=(
            "Assess the risk of this change. "
            "'low' means an isolated addition with no impact on existing "
            "behaviour. "
            "'medium' means a change to existing logic that could regress "
            "adjacent functionality. "
            "'high' means a change to a shared interface, public API, "
            "authentication path, or data schema."
        )
    )


class ImplementationPlan(BaseModel):
    """
    The complete implementation plan produced by the Architecture Agent.
    Contains all tasks to execute in Stage 3 and a high-level summary
    of the change's scope and risk.
    """

    tasks: list[Task] = Field(
        description=(
            "The ordered list of implementation tasks. Each task will be "
            "dispatched to a separate Coding Agent. Include one task per "
            "file that needs to change."
        )
    )
    total_files_affected: int = Field(
        description=(
            "The total number of distinct files that will be created or "
            "modified across all tasks. Must equal len(tasks) since each "
            "task addresses exactly one file."
        )
    )
    estimated_complexity: Literal["low", "medium", "high"] = Field(
        description=(
            "Your overall complexity estimate for the full implementation. "
            "'low' means a few isolated additions (< 50 lines total). "
            "'medium' means multiple files touched or meaningful logic "
            "changes (50–200 lines). "
            "'high' means architectural changes, new abstractions, or "
            "schema migrations (200+ lines or cross-cutting concerns)."
        )
    )
    risk_summary: str = Field(
        description=(
            "Write one paragraph summarising the highest-risk aspects of "
            "this implementation plan and what the Coding Agents should be "
            "especially careful about. Name specific files or interfaces "
            "that are sensitive."
        )
    )


# ---------------------------------------------------------------------------
# Stage 3 — Coding Agent output (one per Task; accumulated via reducer)
# ---------------------------------------------------------------------------

class CodeChange(BaseModel):
    """
    The result of one Coding Agent completing one Task. Contains the
    complete new content of the modified file (not a diff). The PR Author
    Agent computes diffs from these full-file contents.
    """

    task_id: str = Field(
        description=(
            "The task_id of the Task this CodeChange fulfils. Must exactly "
            "match one task_id in the ImplementationPlan."
        )
    )
    file_path: str = Field(
        description=(
            "The path (relative to project root) of the file that was "
            "written. Must match the file_path in the corresponding Task."
        )
    )
    content: str = Field(
        description=(
            "The complete, final content of the file after your changes. "
            "This must be the entire file — not a diff, not a fragment. "
            "The file will be written to disk exactly as you return it."
        )
    )
    explanation: str = Field(
        description=(
            "Explain what you changed in this file and why. Describe the "
            "approach taken, any trade-offs made, and anything the reviewer "
            "should pay special attention to. This appears in the PR "
            "description."
        )
    )


# ---------------------------------------------------------------------------
# Stage 4 — Test-Fix Loop output
# ---------------------------------------------------------------------------

class TestReport(BaseModel):
    """
    Summary of the test execution phase produced after the Test-Fix Loop
    completes. Captures both the pre-existing test suite results and the
    results of any newly generated tests.
    """

    existing_tests_passed: bool = Field(
        description=(
            "Set to True if every test in the pre-existing test suite "
            "passed after the code changes were applied. Set to False if "
            "any pre-existing test failed — this is a regression."
        )
    )
    new_tests_generated: int = Field(
        description=(
            "The total number of new test cases written by the test "
            "generation agent for the changed code."
        )
    )
    new_tests_passed: int = Field(
        description=(
            "The number of newly generated test cases that passed on the "
            "final test run."
        )
    )
    new_tests_failed: int = Field(
        description=(
            "The number of newly generated test cases that failed on the "
            "final test run after all fix retries were exhausted."
        )
    )
    failure_details: list[str] = Field(
        description=(
            "For each test that failed on the final run, provide a string "
            "describing: the test name, the assertion that failed, and the "
            "relevant stack trace excerpt. Return an empty list if all "
            "tests passed."
        )
    )
    retry_count: int = Field(
        description=(
            "The number of fix-and-rerun cycles that were executed before "
            "reaching the final state. Zero means tests passed on the "
            "first run with no fixes needed."
        )
    )
    final_status: Literal["passed", "failed", "partial"] = Field(
        description=(
            "The overall outcome of the test phase. "
            "'passed' means all existing and new tests pass. "
            "'partial' means existing tests pass but some new tests fail. "
            "'failed' means one or more existing tests regressed."
        )
    )


# ---------------------------------------------------------------------------
# Stage 5 — Review Agents output (three agents; accumulated via reducer)
# Stage 6 — Review Synthesizer consumes the accumulated list
# ---------------------------------------------------------------------------

class FindingItem(BaseModel):
    """
    A single finding raised by one of the three review agents (security,
    performance, or quality). Findings are accumulated from all three
    reviewers and then synthesised in Stage 6.
    """

    reviewer: Literal["security", "performance", "quality"] = Field(
        description=(
            "The identity of the reviewer that raised this finding. "
            "Use 'security' for the Security Reviewer, 'performance' for "
            "the Performance Reviewer, and 'quality' for the Quality "
            "Reviewer."
        )
    )
    severity: Literal["blocking", "major", "minor", "suggestion"] = Field(
        description=(
            "The severity of this finding. "
            "'blocking' means the PR must not be merged until this is "
            "fixed — e.g. a critical vulnerability or a regression. "
            "'major' means significant quality or correctness issue that "
            "should be fixed before merge if possible. "
            "'minor' means a small issue that can be addressed in a "
            "follow-up. "
            "'suggestion' means an optional improvement."
        )
    )
    title: str = Field(
        description=(
            "A short, specific title for this finding, e.g. "
            "'SQL query built with string concatenation'. "
            "Maximum 100 characters."
        )
    )
    description: str = Field(
        description=(
            "Explain the problem in detail: what the code does, why it is "
            "problematic, and what the concrete risk or impact is. "
            "Be specific — name the function, variable, or pattern involved."
        )
    )
    file_path: str = Field(
        description=(
            "The path (relative to project root) of the file where this "
            "finding was observed."
        )
    )
    line_number: int | None = Field(
        description=(
            "The approximate line number where the issue occurs. "
            "Set to null if the finding applies to the file as a whole "
            "or if a precise line number cannot be determined."
        )
    )
    suggestion: str = Field(
        description=(
            "Provide a concrete, actionable suggestion for resolving this "
            "finding. Ideally include a brief code example or reference to "
            "the correct pattern to use."
        )
    )


class ReviewFindings(BaseModel):
    """
    The consolidated review output produced by the Review Synthesizer in
    Stage 6. All raw findings from the three reviewers are deduplicated,
    grouped by severity, and accompanied by a final recommendation.
    """

    blocking_issues: list[FindingItem] = Field(
        description=(
            "All findings classified as 'blocking' severity after "
            "deduplication. If this list is non-empty, the recommendation "
            "must be 'reject' or 'request_changes'."
        )
    )
    major_issues: list[FindingItem] = Field(
        description=(
            "All findings classified as 'major' severity after "
            "deduplication. These should be addressed before merge."
        )
    )
    minor_issues: list[FindingItem] = Field(
        description=(
            "All findings classified as 'minor' severity after "
            "deduplication. These are low-priority and can be addressed "
            "in a follow-up PR."
        )
    )
    suggestions: list[FindingItem] = Field(
        description=(
            "All findings classified as 'suggestion' after deduplication. "
            "These are optional improvements — do not block the PR on them."
        )
    )
    recommendation: Literal["approve", "request_changes", "reject"] = Field(
        description=(
            "Your final recommendation for this PR. "
            "'approve' if there are no blocking or major issues. "
            "'request_changes' if there are major or blocking issues that "
            "can be fixed with another revision cycle. "
            "'reject' if the approach is fundamentally flawed and a "
            "complete rewrite is needed."
        )
    )
    recommendation_reasoning: str = Field(
        description=(
            "Write one paragraph explaining your recommendation. Reference "
            "the specific findings that drove the decision. If recommending "
            "'request_changes', state clearly what must change for the next "
            "revision to be approvable."
        )
    )


# ---------------------------------------------------------------------------
# LangGraph state — accumulates all stage outputs across the pipeline
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """
    The single shared state object threaded through the entire LangGraph
    pipeline. Each node reads a subset of fields and returns only the
    fields it modifies.

    Fields annotated with `Annotated[list[T], operator.add]` are written
    to by multiple parallel nodes simultaneously; the reducer merges all
    writes into a single accumulated list rather than overwriting.
    """

    # Input
    issue: dict  # Raw GitHub issue payload (title, body, labels, comments)

    # Stage 1 — Requirements Agent
    requirements_spec: RequirementsSpec

    # Stage 2 — Architecture Agent
    implementation_plan: ImplementationPlan
    tasks: list[Task]

    # Stage 3 — Coding Agents (parallel; reducer accumulates across agents)
    code_changes: Annotated[list[CodeChange], operator.add]

    # Stage 4 — Test-Fix Loop
    test_results: TestReport

    # Stage 5 — Review Agents (parallel; reducer accumulates across reviewers)
    review_findings_raw: Annotated[list[FindingItem], operator.add]

    # Stage 6 — Review Synthesizer
    review_findings: ReviewFindings

    # Stage 7 — Human-in-the-Loop Gate
    human_decision: str   # "approve" | "request_changes" | "reject"
    human_feedback: str   # Optional free-text from the engineer

    # Stage 8 — PR Author
    pr: dict  # GitHub PR payload returned by the API after creation

    # Retry and revision tracking
    revision_count: int   # How many times we have looped back to Stage 3
    max_revisions: int    # Maximum allowed revisions before forced termination

    # Pipeline metadata
    pipeline_status: str    # Current status string for monitoring dashboards
    total_cost_usd: float   # Accumulated LLM API cost across all nodes
