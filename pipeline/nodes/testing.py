"""
pipeline/nodes/testing.py — Stage 4: Testing Node (Test-Fix Loop)

Stage 4 of the pipeline sits between the Coding Agents (Stage 3) and the Review
Agents (Stage 5). It takes the accumulated code changes produced by the parallel
coding fan-out, writes them to an isolated temporary directory, and runs pytest
against the test files.

Execution model:
Executes pytest via subprocess in a local temporary directory. While local execution
grants the generated code filesystem access (a known limitation vs an isolated
Docker sandbox container), it requires zero additional infrastructure.
A ponytail comment marks this boundary and names the Docker upgrade path.

Test-fix loop:
This node is called repeatedly by route_after_testing in pipeline/graph.py until
all tests pass or the revision budget (max_revisions) is exhausted. The node itself
does not loop internally; it executes one cycle, attempts targeted LLM fixes on
failing implementation files (preserving test files as source-of-truth expectations),
increments revision_count, and returns updated state to LangGraph.
"""

import os
import time
import shutil
import tempfile
import subprocess
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.state import PipelineState, TestReport, CodeChange
from pipeline.models import coding_model

logger = logging.getLogger(__name__)


FIXER_SYSTEM_PROMPT = """You are a senior software engineer fixing code to make failing tests pass.
The fix must be minimal: change only what is strictly necessary to make the tests pass without breaking other functionality or introducing new issues.

Output the complete, updated file content for the specified file. Do not output diffs, patches, or markdown formatting outside the schema.

The response must be a JSON object with EXACTLY these fields:
- "task_id": string, copy from the input
- "file_path": string, copy from the input
- "content": string, the complete fixed file content
- "explanation": string, what was changed and why

Do NOT use any other field names.
Respond with a valid JSON object that strictly follows the provided schema.

FastAPI HTTP 400 vs 422 — mandatory rule:
If a test fails with "assert 422 == 400" or "assert 400 == 422":
  - Do NOT change the test assertion to expect 422 — that ignores the requirement.
  - Do NOT add a RequestValidationError exception handler — that is a Pydantic workaround.
  - The correct fix is in the implementation file: remove any Pydantic field validators,
    EmailStr, or format-validating types from the request model. Declare the field as
    plain `str`, then validate manually in the route handler using re.match, and raise
    HTTPException(status_code=400, detail="...") directly.
  - The test asserting 400 is correct. The implementation returning 422 is wrong.

CRITICAL CONSTRAINT: Only return files that already exist in the input. Do NOT create new files, new modules, or new helper files. Fix the existing file in-place. One input file → one output file with the same file_path.
"""



def _write_to_tempdir(
    code_changes: list[CodeChange],
    tmpdir: str,
) -> None:
    """
    Writes each CodeChange content to {tmpdir}/{change.file_path},
    creating intermediate directories as needed.
    """
    for change in code_changes:
        full_path = os.path.join(tmpdir, change.file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(change.content)
        logger.debug(
            "Wrote file to temp directory",
            extra={"path": full_path, "bytes": len(change.content)},
        )


def _run_pytest(tmpdir: str) -> tuple[bool, str, int, int]:
    """
    Runs pytest in the given directory using subprocess.
    Returns: (passed: bool, output: str, n_passed: int, n_failed: int)
    """
    # ponytail: local subprocess execution, not Docker sandbox.
    # Risk: generated code has full filesystem access.
    # Upgrade path: replace subprocess.run with Docker SDK call
    # in a future infrastructure step.
    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                tmpdir,
                "-v",
                "--tb=short",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tmpdir,
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0

        n_passed = 0
        n_failed = 0
        for line in output.split("\n"):
            if " passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.rstrip(",") == "passed" and i > 0:
                        try:
                            n_passed = int(parts[i - 1])
                        except ValueError:
                            pass
            if " failed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.rstrip(",") == "failed" and i > 0:
                        try:
                            n_failed = int(parts[i - 1])
                        except ValueError:
                            pass

        return (passed, output, n_passed, n_failed)
    except subprocess.TimeoutExpired:
        return (False, "pytest timed out after 60 seconds", 0, 0)
    except Exception as e:
        return (False, f"pytest execution error: {str(e)}", 0, 0)


def _extract_failures(output: str) -> list[str]:
    """
    Extracts failure messages from pytest output.
    Returns a list of failure description strings (at most 5).
    """
    failures = []
    lines = output.splitlines()
    for line in lines:
        if line.startswith("FAILED "):
            failures.append(line.strip())
            if len(failures) >= 5:
                break

    if not failures:
        for line in lines:
            if line.startswith("E   "):
                failures.append(line.strip())
                if len(failures) >= 5:
                    break

    if not failures:
        return ["No specific failure details extracted from pytest output."]

    return failures


def _generate_fixes(
    code_changes: list[CodeChange],
    failure_details: list[str],
    pytest_output: str,
) -> list[CodeChange]:
    """
    Calls the LLM to generate fixes for failing code.
    Returns an updated list of CodeChange objects.
    Only attempts to fix implementation files (not test files).
    """
    impl_changes = [
        c for c in code_changes
        if not ("test" in c.file_path.lower() or c.file_path.startswith("tests/"))
    ]
    test_changes = [
        c for c in code_changes
        if "test" in c.file_path.lower() or c.file_path.startswith("tests/")
    ]

    fixed_changes: list[CodeChange] = []

    for change in impl_changes:
        human_msg = f"""File to fix: {change.file_path}

Current file content:
{change.content}

Failing test output:
{chr(10).join(failure_details[:3])}

Pytest summary (last 30 lines):
{chr(10).join(pytest_output.split(chr(10))[-30:])}

Fix the code in {change.file_path} to make the failing tests pass. Return the complete fixed file content."""

        structured_model = coding_model.with_structured_output(
            CodeChange, method="json_mode"
        )
        try:
            time.sleep(15)  # rate limit guard
            fixed = structured_model.invoke(
                [
                    SystemMessage(content=FIXER_SYSTEM_PROMPT),
                    HumanMessage(content=human_msg),
                ]
            )
            fixed_changes.append(fixed)
            logger.info(
                "Fix generated",
                extra={"file_path": change.file_path},
            )
        except Exception as e:
            logger.error(
                "Fix generation failed",
                extra={
                    "file_path": change.file_path,
                    "error": str(e),
                },
            )
            # Keep original on failure — don't lose the change
            fixed_changes.append(change)

    # Merge by file_path, restricted to paths that existed in the original
    # input. Hallucinated extra files (not in the original set) are dropped.
    # This prevents the 1→2→4→8 file accumulation across revision cycles.
    original_paths = {c.file_path for c in code_changes}
    merged: dict[str, CodeChange] = {}
    for c in code_changes:
        merged[c.file_path] = c                          # original baseline
    for c in fixed_changes + test_changes:
        if c.file_path in original_paths:                # allowlist only
            merged[c.file_path] = c                      # fixer version wins
    return list(merged.values())


def testing_node(state: PipelineState) -> dict:
    """
    Stage 4: Testing Node.

    Writes code_changes to a temporary directory, runs pytest,
    parses results, and produces a TestReport. If tests fail
    and the revision budget allows, generates fixes via LLM
    and updates code_changes so route_after_testing can loop
    back to this node.

    The test-fix loop is controlled by route_after_testing in
    graph.py — this node does not loop itself. It produces
    updated state; the routing function decides whether to
    call it again.

    Reads:  state["code_changes"], state["revision_count"],
            state["max_revisions"]
    Writes: state["test_results"], state["code_changes"]
            (updated with fixes on failure),
            state["revision_count"] (incremented on fix attempt),
            state["pipeline_status"]
    """
    code_changes = state.get("code_changes", [])
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    time.sleep(15)  # rate limit guard (same as other nodes)

    if not code_changes:
        logger.warning("Testing node called with no code changes")
        empty_report = TestReport(
            existing_tests_passed=False,
            new_tests_generated=0,
            new_tests_passed=0,
            new_tests_failed=0,
            failure_details=["No code changes to test."],
            retry_count=revision_count,
            final_status="failed",
        )
        return {
            "test_results": empty_report,
            "pipeline_status": "testing_complete",
        }

    tmpdir = tempfile.mkdtemp(prefix="sdlc_test_")
    logger.info(
        "Created temp directory for testing",
        extra={"tmpdir": tmpdir, "files": len(code_changes)},
    )

    try:
        _write_to_tempdir(code_changes, tmpdir)

        test_files = [
            c for c in code_changes
            if "test" in c.file_path.lower()
        ]
        new_tests_generated = len(test_files)

        passed, output, n_passed, n_failed = _run_pytest(tmpdir)

        logger.info(
            "Pytest completed",
            extra={
                "passed": passed,
                "n_passed": n_passed,
                "n_failed": n_failed,
                "revision_count": revision_count,
            },
        )

        if passed:
            report = TestReport(
                existing_tests_passed=True,
                new_tests_generated=new_tests_generated,
                new_tests_passed=n_passed,
                new_tests_failed=0,
                failure_details=[],
                retry_count=revision_count,
                final_status="passed",
            )
            return {
                "test_results": report,
                "pipeline_status": "testing_complete",
            }

        failure_details = _extract_failures(output)

        if revision_count < max_revisions:
            logger.info(
                "Tests failed, generating fixes",
                extra={
                    "revision_count": revision_count,
                    "max_revisions": max_revisions,
                    "failures": len(failure_details),
                },
            )
            fixed_changes = _generate_fixes(
                code_changes, failure_details, output
            )
            report = TestReport(
                existing_tests_passed=False,
                new_tests_generated=new_tests_generated,
                new_tests_passed=n_passed,
                new_tests_failed=n_failed,
                failure_details=failure_details,
                retry_count=revision_count,
                final_status="failed",
            )
            return {
                "test_results": report,
                "code_changes": fixed_changes,
                "revision_count": revision_count + 1,
                "pipeline_status": "testing_fixing",
            }
        else:
            logger.warning(
                "Tests failed and revision budget exhausted",
                extra={
                    "revision_count": revision_count,
                    "max_revisions": max_revisions,
                },
            )
            report = TestReport(
                existing_tests_passed=False,
                new_tests_generated=new_tests_generated,
                new_tests_passed=n_passed,
                new_tests_failed=n_failed,
                failure_details=failure_details,
                retry_count=revision_count,
                final_status="failed",
            )
            return {
                "test_results": report,
                "pipeline_status": "testing_complete",
            }
    except Exception as e:
        logger.error(
            "Testing node unexpected failure",
            extra={"error": str(e)},
        )
        error_report = TestReport(
            existing_tests_passed=False,
            new_tests_generated=0,
            new_tests_passed=0,
            new_tests_failed=0,
            failure_details=[f"Testing node error: {str(e)}"],
            retry_count=revision_count,
            final_status="failed",
        )
        return {
            "test_results": error_report,
            "pipeline_status": "testing_complete",
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.info(
            "Cleaned up temp directory",
            extra={"tmpdir": tmpdir},
        )
