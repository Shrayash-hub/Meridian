# Meridian

An autonomous multi-agent pipeline that takes a GitHub issue and carries it through requirements analysis, code generation, testing, code review, human approval, and PR creation without manual intervention at each stage.

---

## Overview

The standard path from a filed bug report or feature request to a merged pull request involves a repetitive sequence of manual steps: read the issue, identify which files are relevant, write the fix, run the tests, fix what the tests break, get a peer review, address the review comments, and finally open the PR. For teams handling high issue volume, most of that work follows predictable patterns that do not require human judgment at every step.

Meridian addresses this by running a structured multi-agent pipeline over each incoming GitHub issue. When an issue is received, the pipeline extracts a structured requirements specification, decomposes the implementation into parallel coding tasks, generates the code, runs pytest against it in a temporary directory, generates targeted fixes on failure, dispatches three simultaneous code reviewers (security, performance, quality), synthesizes their findings, and opens a pull request. The only mandatory human interaction is the approval gate after the review synthesis, where an engineer sees the full review package before anything is pushed.

The orchestration layer is built on LangGraph. LangGraph's `StateGraph` provides deterministic, checkpointed execution across all pipeline stages. The interrupt/resume primitive (`interrupt()` plus `Command(resume=...)`) is what makes the human-in-the-loop gate possible: graph state is fully serialized to the checkpointer when the pipeline pauses at the approval gate, and execution resumes from exactly that point when the engineer responds. The Send API enables the parallel fan-outs at Stage 3 (one coding agent per task) and Stage 5 (three review agents simultaneously), with `operator.add` reducers on the relevant state fields merging the parallel outputs back into a single accumulated list.

Meridian is a working prototype of this pipeline. It runs end-to-end, opens real GitHub PRs, and handles multi-turn fix loops when tests fail. It is not production-ready in the sense that checkpointing is in-memory (process restarts lose all state), generated code runs in a local subprocess with no sandboxing, and the HITL gate is auto-approved by default rather than wired to a review UI. These are documented constraints, each with a clear upgrade path.

---

## Architecture

### Pipeline stages

```
Stage 1:  Requirements Analysis      (sequential)
Stage 2:  Architecture Planning      (sequential)
Stage 3:  Code Generation            (parallel fan-out via Send API, one agent per task)
Stage 4:  Test and Fix Loop          (sequential, loops up to MAX_REVISIONS times)
Stage 5:  Code Review                (parallel fan-out via Send API, three reviewers)
Stage 6:  Review Synthesis           (sequential, aggregates Stage 5 results)
Stage 7:  Human-in-the-Loop Gate     (interrupt/resume, waits for human approval)
Stage 8:  PR Creation                (sequential)
Stage 9:  Memory Update              (sequential, post-pipeline success path only)
```

### Stage descriptions

**Stage 1: Requirements Analysis** (`pipeline/nodes/requirements.py`)

Reads `state["issue"]` (the raw GitHub issue dict with title, number, body, labels, and comments). Before calling the LLM, it calls `search_codebase()` against the FAISS index using the issue title as the query, appending the top-3 relevant code chunks to the prompt if an index exists. It also calls `get_recent_memories()` to inject up to 3 past pipeline runs from the long-term store, so the model can reference recurring review patterns from similar issues. The LLM produces a `RequirementsSpec` with six fields: `summary`, `acceptance_criteria`, `constraints`, `files_likely_affected`, `ambiguities`, and `confidence`. Writes `state["requirements_spec"]` and `state["pipeline_status"]`.

**Stage 2: Architecture Planning** (`pipeline/nodes/architecture.py`)

Reads `state["requirements_spec"]`. Also calls `search_codebase()` against the FAISS index using the requirements summary as the query. The LLM produces an `ImplementationPlan` containing a list of `Task` objects, where each task specifies a single file path, detailed implementation instructions, a risk level, and a task ID slug. The constraint that each task operates on exactly one file is enforced both in the schema (the `file_path` field is a string, not a list) and in the system prompt. Writes `state["implementation_plan"]`, `state["tasks"]`, and `state["pipeline_status"]`. The `tasks` list is what `route_to_coding_agents` reads to construct the Send API fan-out.

**Stage 3: Code Generation** (`pipeline/nodes/coding.py`)

Each coding agent instance receives a `CodingAgentState` slice containing exactly one `Task` and the `RequirementsSpec`. It calls the LLM to produce complete file content for that task, not a diff, and returns a `CodeChange` wrapped in a one-item list. The `operator.add` reducer on `PipelineState["code_changes"]` accumulates the results from all parallel agents as they complete. LangGraph guarantees that the edge from `CODING` to `TESTING` only fires once all parallel coding agents have returned. Writes `state["code_changes"]` (appends one `CodeChange`).

**Stage 4: Test and Fix Loop** (`pipeline/nodes/testing.py`)

Reads `state["code_changes"]`, `state["revision_count"]`, and `state["max_revisions"]`. Writes all `CodeChange` files to a `tempfile.mkdtemp()` directory, then runs `python -m pytest` via `subprocess.run` with a 60-second timeout. If pytest passes, it writes a `TestReport` with `final_status="passed"` and returns. If pytest fails and `revision_count < max_revisions`, it calls `_generate_fixes()`, which sends only the implementation files (those without "test" in their path) to the LLM with the pytest failure output, receives updated `CodeChange` objects, merges them against the original file-path allowlist to prevent hallucinated new files from accumulating, increments `revision_count`, and returns the updated `code_changes` so `route_after_testing` can loop back to this node. When the budget is exhausted, the node returns the failed `TestReport` without generating further fixes, and the pipeline exits the loop and proceeds to review. The temp directory is always removed in the `finally` block. Writes `state["test_results"]`, `state["code_changes"]` (on fix), `state["revision_count"]`, and `state["pipeline_status"]`.

**Stage 5: Code Review** (`pipeline/nodes/review.py`)

Three separate node functions (`security_review_node`, `performance_review_node`, `quality_review_node`) run simultaneously via the Send API. Each receives a `ReviewAgentState` slice with its `reviewer_type`, the full `code_changes` list, and the `requirements_spec`. Each calls the LLM with a specialized system prompt focused on its dimension (injection vulnerabilities and OWASP Top 10 for security; algorithmic complexity, N+1 patterns, and resource leaks for performance; error handling, test coverage, and design consistency for quality) and produces a list of `FindingItem` objects. Each `FindingItem` includes the reviewer identity, severity, title, description, file path, line number, and a concrete suggestion. The nodes use separate named functions rather than a single generic node because LangGraph's parallel execution requires each branch to be a distinct named node in the graph topology. Each function includes `time.sleep()` calls to stagger their LLM requests and avoid burst-rate limit violations. Writes `state["review_findings_raw"]` (appended via `operator.add`).

**Stage 6: Review Synthesis** (`pipeline/nodes/synthesizer.py`)

Reads `state["review_findings_raw"]` (the accumulated list from all three reviewers) and `state["requirements_spec"]`. Groups findings by reviewer, passes the structured summary to the LLM, and produces a unified `ReviewFindings` object with findings bucketed into `blocking_issues`, `major_issues`, `minor_issues`, and `suggestions`. The LLM deduplicates overlapping findings from different reviewers, assigns the higher severity on conflicts, and produces a `recommendation` of `"approve"`, `"request_changes"`, or `"reject"` with a written `recommendation_reasoning`. Writes `state["review_findings"]` and `state["pipeline_status"]`.

**Stage 7: Human-in-the-Loop Gate** (`pipeline/nodes/hitl.py`)

Reads `state["review_findings"]`, `state["requirements_spec"]`, `state["code_changes"]`, `state["revision_count"]`, and `state["max_revisions"]`. Builds a structured interrupt payload containing the review recommendation, reasoning, issue counts by severity, the list of changed files, the requirements summary, remaining revision budget, and the set of valid decisions. Calls `interrupt(payload)`, which serializes the full graph state to the `MemorySaver` checkpointer and raises `GraphInterrupt` (a `BaseException`) to pause execution. The graph remains paused until `Command(resume=human_response)` is received with the same `thread_id`. On resume, the node reads `decision` and `feedback` from the response, validates the decision against the allowed set, and on `"request_changes"` increments `revision_count` before returning. Writes `state["human_decision"]`, `state["human_feedback"]`, `state["revision_count"]`, and `state["pipeline_status"]`.

**Stage 8: PR Creation** (`pipeline/nodes/pr_author.py`)

Reads `state["issue"]`, `state["requirements_spec"]`, `state["code_changes"]`, `state["test_results"]`, and `state["review_findings"]`. Calls the LLM to produce a `PRDescription` with six fields (title, summary, motivation, changes made, test plan, risk assessment), then renders it into a GitHub-flavored markdown PR body. When `DRY_RUN=true`, returns a synthetic PR dict and exits. Otherwise, posts to `POST /repos/{github_repo}/pulls` via httpx, posts each `FindingItem` from the review as a comment on the PR's conversation thread (true diff-position inline comments would require the commit SHA from the GitHub Commits API, noted as the upgrade path), and posts a linking comment on the original issue. Writes `state["pr"]` and `state["pipeline_status"]`.

**Stage 9: Memory Update** (`pipeline/nodes/memory_update.py`)

Runs only on the success path, when `route_after_pr` detects `pipeline_status == "pr_opened"`. Calls `extract_memory()` in `memory/store.py`, which makes an LLM call to distill the completed run into a `PRMemory` object containing the issue type, files changed, approach summary, review patterns to watch for, the human decision, and lessons learned. Saves the result to the `InMemoryStore` under the namespace `("sdlc_pipeline", "pr_patterns")` with a key of the form `"issue-{number}-{YYYYMMDD-HHMMSS}"`. Memory update failure never affects the pipeline result: the node catches all exceptions and returns a status suffix (`pr_opened_memory_skipped`, `pr_opened_memory_failed`) rather than crashing. Writes `state["pipeline_status"]`.

### Graph topology and routing

The graph is a `StateGraph(PipelineState)` compiled with `MemorySaver` as checkpointer and `InMemoryStore` as the long-term memory store.

`route_to_coding_agents` runs as a conditional edge from `ARCHITECTURE`. It reads `state["tasks"]` and returns a `list[Send]` with one `Send(CODING, CodingAgentState(...))` per task. If the task list is empty (architecture agent failure or a degenerate input), it returns `[Send(TESTING, state)]` to skip coding and proceed directly to testing. LangGraph fans all coding agents out simultaneously and fans them back in to the single `TESTING` edge when they all complete.

`route_after_testing` is the conditional edge from `TESTING`. It checks whether `test_results.final_status` is `"passed"` and whether `revision_count < max_revisions`. The check handles both Pydantic objects (live run) and plain dicts (checkpoint deserialization after HITL resume) by trying `getattr` before falling back to dict `.get()`. If tests are still failing and budget remains, it returns the string `"testing"` to loop back to the testing node. Otherwise, it calls `route_to_review_agents()` and returns a `list[Send]` fanning out to all three review agents simultaneously. This is the only routing function in the graph that can return either a string or a list, which LangGraph supports on conditional edges.

`route_after_hitl` routes on `state["human_decision"]`. The value `"approve"` goes to `PR_AUTHOR`. The value `"request_changes"` goes back to `CODING`, re-entering Stage 3 with the current state (which at that point includes any fixed code changes from the test-fix loop). Any other value, including `"reject"` and unrecognized strings, goes to `END`, terminating the pipeline safely.

`route_after_pr` routes on `state["pipeline_status"]`. The value `"pr_opened"` proceeds to `MEMORY_UPDATE` before `END`. Any other value (including `"pr_failed"` and `"pr_dry_run"`) goes directly to `END`, skipping memory update.

---

## Key technical decisions

### FallbackChain

`FallbackChain` is a plain Python class defined in `pipeline/models.py` that wraps a list of LangChain chat models and tries them left to right on every `invoke()` or `with_structured_output().invoke()` call. Each provider is initialized with `max_retries=0` so that SDK-level retry logic does not compete with `FallbackChain`'s own fallback logic. Each provider call runs inside a `ThreadPoolExecutor` with a 35-second wall-clock deadline, enforced because `ChatGoogleGenerativeAI` ignores SDK-level timeout parameters and can hang indefinitely without a thread-level cutoff.

`FallbackChain` was built as a plain class rather than using LiteLLM because the project already depends on LangChain provider packages and adding LiteLLM would have introduced a second provider-abstraction layer with its own model name conventions and configuration surface area. The class is a drop-in replacement for any single LangChain chat model: any node that can call `model.invoke(messages)` or `model.with_structured_output(Schema).invoke(messages)` can use a `FallbackChain` without modification.

The upgrade path documented in the code is to replace `FallbackChain` with LiteLLM when provider diversity beyond Gemini and Groq is needed.

### Multi-key Gemini rotation

`_gemini_all()` in `pipeline/models.py` reads up to three Google API keys (`GOOGLE_API_KEY`, `GOOGLE_API_KEY_2`, `GOOGLE_API_KEY_3`) and returns a separate `ChatGoogleGenerativeAI` instance for each configured key. The comment in the code notes that keys must come from different Google Cloud projects to have independent quota pools: a single GCP project's quota is shared across all keys created under it. All three exported model objects (`review_model`, `coding_model`, `flash_model`) are built as `FallbackChain([*_gemini_all(), _make_groq()])`, meaning the pipeline exhausts all Gemini quota pools before falling back to Groq. Groq is treated as a last resort because its free-tier OTPM (output tokens per minute) limit makes it unsuitable as a primary provider for a pipeline that makes eight or more LLM calls per run.

### State design

`PipelineState` is a `TypedDict` where most fields hold Pydantic models (`RequirementsSpec`, `ImplementationPlan`, `TestReport`, `ReviewFindings`) rather than raw dicts. This keeps structured output parsing and validation at the LLM call boundary and exposes clean typed objects to each node.

The two fields that multiple parallel agents write to simultaneously, `code_changes` and `review_findings_raw`, are declared with `Annotated[list[T], operator.add]`. The `operator.add` annotation tells LangGraph's fan-in logic to concatenate the lists returned by parallel node executions rather than overwriting them. Without this reducer, the last parallel agent to finish would overwrite the results from all earlier ones.

LangGraph's `MemorySaver` checkpointer serializes the entire `PipelineState` to memory on each node completion, including after the `interrupt()` call in the HITL gate. LangGraph produces "Deserializing unregistered type" warnings for the Pydantic model fields when the checkpoint is later deserialized; the fix is to register those types in `allowed_msgpack_modules`, which is the documented upgrade path in the LangGraph serialization documentation.

### HITL implementation

Inside `hitl_node`, calling `interrupt(payload)` does two things at the LangGraph level: it serializes the complete graph state to the checkpointer and raises `GraphInterrupt`, which is a `BaseException` (not `Exception`). Because it is a `BaseException`, a bare `except Exception` block in the caller will not catch it.

The pipeline runner in `ingestion/runner.py` handles this with `except BaseException as e` and `isinstance(e, GraphInterrupt)`. On catching the interrupt, the runner reads the last checkpoint state via `pipeline.get_state(config).values` to log the Phase 1 outcome. Phase 2 is then triggered by calling `pipeline.invoke(Command(resume=human_response), config=config)` with the same `thread_id` that was used in Phase 1. LangGraph resumes execution from the point immediately after the `interrupt()` call inside `hitl_node`, with all prior state intact. The human response dict is the return value of `interrupt()` inside the node function.

This design means that in production, Phase 1 and Phase 2 do not need to run in the same process or even on the same machine, as long as the checkpointer is a shared persistent store. With the current `MemorySaver`, both phases must run in the same process.

### Test-fix loop

The testing node does not loop internally. It runs one complete cycle (write files to temp dir, run pytest, parse results, optionally generate fixes) and returns. The routing function `route_after_testing` decides whether to invoke it again.

When tests fail and the revision budget allows, `_generate_fixes()` sends each implementation file individually to the LLM alongside the pytest failure output. Test files are deliberately excluded from the fix set: the test file is treated as the specification, and the implementation is what changes. After generating fixes, the function merges the updated `CodeChange` objects against the original file-path set (the allowlist). Any file paths the LLM introduces that were not in the original `code_changes` list are silently dropped. This prevents the LLM from hallucinating new files across revision cycles, which would cause the `code_changes` list to grow unboundedly.

The routing function handles two formats for `test_results.final_status` because LangGraph's `MemorySaver` deserializes Pydantic models as plain dicts when the checkpoint is read after a process restart or HITL resume. The routing function uses `getattr(test_results, "final_status", None) or (test_results.get("final_status") if isinstance(test_results, dict) else None)` to handle both cases.

`MAX_REVISIONS` sets the maximum number of fix-and-rerun cycles. The default is 3. When the budget is exhausted, the pipeline exits the loop and proceeds to code review with the most recently fixed (but still failing) code. The engineer sees the full review including the test failures.

### RAG code search

`build_index.py` accepts a path to a codebase directory, walks all Python files in it (skipping `__pycache__`, `.git`, `node_modules`, and virtual environment directories), and chunks each file using Python's stdlib `ast` module. The chunker identifies top-level function definitions, async function definitions, and class definitions, extracts each as a named chunk with source line range metadata, and collects any remaining non-blank lines at module level into a "module_level" chunk. It then calls `FAISS.from_texts()` with `GoogleGenerativeAIEmbeddings` (`text-embedding-004`) to build a FAISS vector index and saves it to the directory specified by `INDEX_PATH` (default: `codebase_index/`).

At runtime, `search_codebase()` is called by both the requirements node and the architecture node before their LLM calls. It loads the FAISS index from disk and runs a similarity search with `k=3`, returning a formatted string of the top-3 matching code chunks. If the `index.faiss` file does not exist in the index directory, `search_codebase()` logs an info message and returns an empty string, and both nodes proceed without RAG context. The choice of the stdlib `ast` module over Tree-sitter is noted in the code: `ast` requires no additional dependencies, and Tree-sitter is the documented upgrade for multi-language codebases or more complex Python parsing.

### In-memory storage and upgrade paths

Three storage mechanisms in the pipeline are in-memory only:

`MemorySaver` (the LangGraph checkpointer) stores the serialized graph state for each `thread_id`. All checkpoints are lost on process restart. The documented upgrade is `SqliteSaver` for single-process persistence or `PostgresCheckpointer` for shared multi-process access.

`InMemoryStore` (the long-term memory store) holds `PRMemory` objects extracted after each successful run and the run metrics recorded by the evaluator. All entries are lost on process restart. The documented upgrade, noted directly in `memory/store.py`, is to replace the `InMemoryStore()` initialization with `AsyncPostgresStore` backed by pgvector. No other code in the module needs to change.

The subprocess execution model in the testing node (`subprocess.run(["python", "-m", "pytest", ...]`) gives generated code full filesystem access on the host machine. The documented upgrade path is to replace the `subprocess.run` call with a Docker SDK call that runs pytest inside an isolated container with no host filesystem access.

---

## Technologies

### Orchestration

- **LangGraph** (`langgraph>=0.2,<0.3`): `StateGraph` for the pipeline topology, `MemorySaver` for checkpointing, `InMemoryStore` for long-term memory, `interrupt()` and `Command` for HITL, and the `Send` API for parallel fan-outs.
- **LangChain Core** (`langchain-core>=0.2,<0.3`): `BaseMessage`, `SystemMessage`, `HumanMessage`, and `BaseChatModel` abstractions that all node code and the `FallbackChain` are written against.

### LLM providers

- **Google Gemini** via `langchain-google-genai` (`>=1.0,<2.0`): `ChatGoogleGenerativeAI` with `model="gemini-3.8-flash"` is the primary provider for all three model objects (`review_model`, `coding_model`, `flash_model`). Up to three independent keys from separate GCP projects can be configured for quota rotation. `GoogleGenerativeAIEmbeddings` with `model="models/text-embedding-004"` is used for RAG indexing and search.
- **Groq** via `langchain-groq` (`>=0.1,<1.0`): `ChatGroq` with `model="openai/gpt-oss-120b"` is the fallback provider in all three `FallbackChain` instances. The `FallbackChain` only routes to Groq when all configured Gemini keys have failed or timed out.

### Web framework

- **FastAPI** (`>=0.111,<1.0`): `ingestion/webhook.py` defines a router with the `POST /webhook` endpoint for GitHub event ingestion. `main.py` mounts this router and adds `/health` and `/metrics` endpoints. Signature verification uses Python's stdlib `hmac` and `hashlib`.
- **Uvicorn** (`>=0.30,<1.0`): ASGI server. Run via `uvicorn main:app` or `python main.py`.

### Testing

- **pytest**: Invoked via `subprocess.run(["python", "-m", "pytest", ...])` inside the testing node. The coding agent generates test files using FastAPI's `TestClient` (from `httpx`) for route handler tests.
- **httpx** (`>=0.27,<1.0`): Used directly in `pr_author.py` for all GitHub REST API calls, and used by FastAPI's `TestClient` internally.

### Vector search

- **FAISS** (`faiss-cpu>=1.7,<2.0`): The vector store for the RAG codebase index. Built once via `build_index.py` and loaded at runtime.
- **LangChain Community** (`langchain-community>=0.2,<0.3`): Provides the `FAISS` wrapper class used in `code_search.py`.
- **GoogleGenerativeAIEmbeddings**: Generates embeddings for both index construction and query-time similarity search.

### GitHub integration

- **httpx** with direct GitHub REST API calls: `pr_author.py` posts to `/repos/{owner}/{repo}/pulls` to open PRs, `/repos/{owner}/{repo}/issues/{pr_number}/comments` to post review findings, and `/repos/{owner}/{repo}/issues/{issue_number}/comments` to link the issue to the new PR. All calls use `Bearer` token authentication and the `application/vnd.github+json` accept header.

### Observability

- **LangSmith**: Enabled by setting `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in the environment. LangChain detects these variables automatically and traces every LLM call in every node, the full graph execution, per-node latency, and error details. No code changes are needed in any node file.

---

## Project structure

```
sdlc-project/
├── main.py                         FastAPI application entry point; mounts webhook router, /health, /metrics
├── build_index.py                  CLI script to build the FAISS codebase index from a local directory
├── requirements.txt                Python dependency pinning for all packages
├── .env.example                    Template for all required and optional environment variables
├── .gitignore                      Excludes secrets, bytecode, virtual environments, and index files
│
├── pipeline/
│   ├── state.py                    Single source of truth for all Pydantic schemas and PipelineState TypedDict
│   ├── graph.py                    Graph construction, node registration, edge wiring, and compilation
│   ├── models.py                   FallbackChain class and all LLM model instantiations (coding_model, review_model, flash_model, embeddings_model)
│   ├── config.py                   build_pipeline_config(): constructs the LangGraph config dict for every pipeline invocation
│   ├── logging_config.py           configure_logging(): structured JSON logging with third-party logger suppression
│   │
│   ├── nodes/
│   │   ├── requirements.py         Stage 1: extracts RequirementsSpec from GitHub issue with RAG and memory injection
│   │   ├── architecture.py         Stage 2: decomposes RequirementsSpec into parallel Task objects via ImplementationPlan
│   │   ├── coding.py               Stage 3 worker: generates complete file content for one Task; runs in parallel per task
│   │   ├── testing.py              Stage 4: writes code to temp dir, runs pytest, generates LLM fixes on failure, loops via router
│   │   ├── review.py               Stage 5: three parallel review agents (security, performance, quality) via Send API
│   │   ├── synthesizer.py          Stage 6: merges and deduplicates raw findings from all three reviewers into ReviewFindings
│   │   ├── hitl.py                 Stage 7: interrupt()/resume HITL gate; pauses graph for human engineer approval
│   │   ├── pr_author.py            Stage 8: generates PR description via LLM, opens GitHub PR, posts review comments
│   │   └── memory_update.py        Stage 9: extracts PRMemory from completed run and saves to InMemoryStore (success path only)
│   │
│   └── tools/
│       └── code_search.py          AST-based Python chunker, FAISS index builder, and search_codebase() function
│
├── ingestion/
│   ├── webhook.py                  FastAPI router for POST /webhook; HMAC-SHA256 signature verification, event filtering
│   └── runner.py                   run_pipeline_for_issue(): two-phase invocation handling Phase 1 (interrupt) and Phase 2 (resume)
│
├── memory/
│   └── store.py                    InMemoryStore singleton, PRMemory schema, extract_memory(), save_memory(), get_recent_memories()
│
└── evaluation/
    └── evaluator.py                record_run_metrics() and get_metrics_summary() for in-memory aggregate quality tracking
```

---

## Setup and installation

### Prerequisites

- Python 3.11 or later (the code uses `list[T]` and `X | Y` type syntax from 3.10+, and `match` statements are not used so 3.10+ is sufficient; 3.11 is recommended)
- `git`

### Clone the repository

```bash
git clone https://github.com/Shrayash-hub/Meridian.git
cd Meridian
```

### Create a virtual environment

```bash
python -m venv .venv
```

On Windows:
```bash
.venv\Scripts\activate
```

On macOS and Linux:
```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set the following:

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key from [aistudio.google.com](https://aistudio.google.com). Primary LLM provider for all pipeline nodes. Free tier allows 20 requests per day per GCP project. |
| `GOOGLE_API_KEY_2` | No | Second Gemini key from a different GCP project. Adds a second independent quota pool to the FallbackChain. |
| `GOOGLE_API_KEY_3` | No | Third Gemini key from a different GCP project. Adds a third independent quota pool. |
| `GROQ_API_KEY` | No | Groq API key. Used as the final fallback after all Gemini keys are exhausted. |
| `GITHUB_TOKEN` | Yes (for real PRs) | GitHub personal access token with `repo` scope. Not needed when `DRY_RUN=true`. |
| `GITHUB_WEBHOOK_SECRET` | Recommended | Secret used to verify HMAC-SHA256 signatures on incoming webhook payloads. If unset, signature verification is skipped with a warning. |
| `GITHUB_REPO` | Yes (for real PRs) | Target repository in `owner/repo` format, e.g. `myorg/myrepo`. |
| `HEAD_BRANCH` | No | Branch the pipeline commits code to (the PR head). Default: `sdlc/automated`. |
| `BASE_BRANCH` | No | Branch to merge the PR into. Default: `main`. |
| `DRY_RUN` | No | Set to `true` to skip all GitHub API calls. The PR author node returns a synthetic PR dict. Default: `true` in the example file. |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable LangSmith tracing of all LLM calls. Default: `false`. |
| `LANGCHAIN_API_KEY` | No | LangSmith API key from [smith.langchain.com](https://smith.langchain.com). Required when `LANGCHAIN_TRACING_V2=true`. |
| `LANGCHAIN_PROJECT` | No | LangSmith project name for grouping traces. Default: `sdlc-pipeline`. |
| `LANGCHAIN_ENDPOINT` | No | LangSmith API endpoint. Default: `https://api.smith.langchain.com`. |
| `AUTO_APPROVE` | No | Set to `true` to auto-approve at the HITL gate without waiting for human input. Set to `false` in production. Default: `true`. |
| `MAX_REVISIONS` | No | Maximum test-fix loop iterations per pipeline run. Default: `3`. |
| `INDEX_PATH` | No | Directory where the FAISS codebase index is stored and loaded from. Default: `codebase_index`. |
| `PIPELINE_VERSION` | No | Version string included in LangSmith trace metadata. Default: `1.0`. |
| `ENVIRONMENT` | No | Environment label included in trace metadata and tags. Default: `development`. |
| `LOG_LEVEL` | No | Python logging level for the application. Default: `INFO`. |
| `HOST` | No | Host address for uvicorn. Default: `0.0.0.0`. |
| `PORT` | No | Port for uvicorn. Default: `8000`. |

---

## Configuration

### MAX_REVISIONS

Controls how many times the test-fix loop can iterate before the pipeline gives up and proceeds to code review with the current (failing) code. The default is 3. Setting it to 1 means the pipeline generates fixes once, reruns pytest once, and then exits the loop regardless of outcome. Setting it to 0 means the testing node runs once and immediately exits to review on any failure, with no fix attempts.

Increasing this value increases the number of LLM calls per run (each fix cycle calls the fixer LLM once per implementation file) and increases total pipeline runtime.

### MAX_TASKS

This variable is not currently read in the pipeline code. The architecture agent's task count is determined entirely by the LLM's decomposition of the issue. To limit LLM calls during development, set `MAX_REVISIONS=1` and use `DRY_RUN=true` rather than trying to cap the task count.

### DRY_RUN

When `DRY_RUN=true`, the PR author node skips all GitHub API calls (opening the PR, posting review comments, linking the issue) and returns a synthetic dict:

```json
{
  "number": 0,
  "html_url": "https://github.com/dry-run/pr/0",
  "title": "<generated title>",
  "body": "<generated body>",
  "state": "open",
  "dry_run": true
}
```

The rest of the pipeline (including memory update) continues normally. Use `DRY_RUN=true` during development to avoid creating real GitHub objects.

### AUTO_APPROVE

When `AUTO_APPROVE=true`, the pipeline runner automatically resumes the HITL gate with `decision: "approve"` without waiting for human input. The HITL interrupt still fires inside the graph, the checkpoint is still written, and the graph still pauses for the duration of Phase 1 teardown and Phase 2 startup, but the human response is injected programmatically.

When `AUTO_APPROVE=false`, the current runner still defaults to approve because no review UI is implemented yet. The comment in `runner.py` notes that production requires storing the Phase 1 result in a database, exposing it via a `/review/{issue_number}` endpoint, and triggering Phase 2 via a `/approve/{issue_number}` endpoint.

### LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY

Set both of these to enable LangSmith tracing:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your key from smith.langchain.com>
```

LangChain detects these variables at import time. No code changes are needed. Every LLM call in every node, the full graph execution tree, per-node latency, token counts, and error tracebacks will appear in the LangSmith UI under the project named by `LANGCHAIN_PROJECT`. Each run is tagged with the issue number and environment from `build_pipeline_config()`.

---

## Building the RAG index

The codebase RAG tool requires a FAISS index built from the target codebase. Build it once before running the pipeline against a new project:

```bash
python build_index.py /path/to/your/project
```

The script walks all `.py` files in the given directory (skipping `__pycache__`, `.git`, `.venv`, and similar), chunks each file by top-level function and class definitions using the stdlib `ast` module, embeds the chunks with `GoogleGenerativeAIEmbeddings`, and saves the FAISS index to the directory specified by `INDEX_PATH` (default: `codebase_index/`).

`GOOGLE_API_KEY` must be set in `.env` before running the indexer, because the embedding model requires it.

If the index does not exist when the pipeline runs, `search_codebase()` logs an info message and returns an empty string. The requirements and architecture nodes continue without RAG context. The pipeline does not fail.

---

## Running the pipeline

### Option 1: End-to-end validation

The project includes `validate_e2e.py` (not committed to the repository, as it is an internal validation script) that constructs a test GitHub issue, runs the full pipeline, and prints the status of each stage.

```bash
python validate_e2e.py
```

Set `DRY_RUN=true` and `AUTO_APPROVE=true` in `.env` before running this to avoid creating real GitHub objects.

### Option 2: Webhook server

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Or:

```bash
python main.py
```

The server listens on `HOST:PORT` (default: `0.0.0.0:8000`). To receive GitHub webhook events locally, expose the server with a tunneling tool:

```bash
ngrok http 8000
```

Set the ngrok HTTPS URL as the webhook URL in your GitHub repository settings under Settings, Webhooks. Set the Content type to `application/json`, choose "Let me select individual events", and enable "Issues". Set a secret and put the same value in `GITHUB_WEBHOOK_SECRET`.

The webhook endpoint is `POST /webhook/webhook`. It accepts GitHub webhook payloads, verifies the `X-Hub-Signature-256` header via HMAC-SHA256 (skipped with a warning if `GITHUB_WEBHOOK_SECRET` is not set), and filters for `issues` events with `action: "opened"`. All other events are returned immediately with `{"status": "ignored"}`. A health check is available at `GET /health` and aggregate metrics at `GET /metrics`.

---

## How a real run works

When a new issue is opened in the configured repository, GitHub sends a POST to `/webhook/webhook` with the issue payload and an HMAC-SHA256 signature. The webhook handler verifies the signature, extracts the issue title, number, body, and labels, and calls `run_pipeline_for_issue()`.

The runner assigns a `thread_id` of the form `issue-{number}-{hex8}` and calls `pipeline.invoke(initial_state, config=config)`. Execution enters Stage 1. The requirements node calls `search_codebase()` with the issue title, retrieving the three most similar code chunks from the FAISS index if it exists. It also retrieves up to three recent `PRMemory` records from the long-term store, so the model can see recurring review patterns from similar past issues. The LLM produces a `RequirementsSpec` with a one-paragraph engineering summary, three to six testable acceptance criteria, any stated constraints, a list of files likely to need changes, and a confidence rating.

Stage 2 takes that spec and produces an `ImplementationPlan`. For a typical feature request touching two files, the architecture agent returns two `Task` objects, each with a slug ID, a file path, and detailed per-file instructions. `route_to_coding_agents` reads the tasks list and constructs two `Send` objects. LangGraph dispatches both coding agents simultaneously. Each agent receives its one task and the full requirements spec, calls the LLM, and returns a complete `CodeChange` with the entire file content. The `operator.add` reducer accumulates both results into `state["code_changes"]` as they complete.

Testing starts once both coding agents have returned. The testing node writes both files to a temporary directory under the system's temp path, runs `python -m pytest /tmp/sdlc_test_xxx -v --tb=short`, and parses the output. If pytest fails, the node calls the fixer LLM with the pytest output and the implementation file content, receives an updated file, merges it against the allowlist, increments `revision_count`, and returns. `route_after_testing` routes back to `TESTING`. The loop continues until all tests pass or `MAX_REVISIONS` is exhausted.

After the loop exits, `route_after_testing` fires three `Send` objects simultaneously: one to `security_review_node`, one to `performance_review_node`, and one to `quality_review_node`. Each reviewer independently calls the LLM with its specialized system prompt and the full `code_changes` list, producing a list of `FindingItem` objects accumulated into `state["review_findings_raw"]`. The synthesizer then reads all accumulated findings, groups them by reviewer, and asks the LLM to deduplicate, bucket by severity, and produce a `recommendation`.

The graph enters Stage 7 and calls `interrupt(payload)`. The payload the engineer sees includes the review recommendation and reasoning, counts of blocking and major issues, the list of changed files, remaining revision budget, and valid decision options. LangGraph raises `GraphInterrupt`, the runner catches it, reads the checkpoint state, and (when `AUTO_APPROVE=true`) immediately calls `pipeline.invoke(Command(resume={"decision": "approve", "feedback": "..."}), config=config)`. The graph resumes inside `hitl_node` at the line after `interrupt()`, reads the decision from the resume value, writes it to state, and returns. `route_after_hitl` sends execution to Stage 8.

The PR author node calls the LLM to produce a structured PR description, formats it into a GitHub-flavored markdown body with sections for summary, motivation, changes made, test plan, and risk assessment. It posts to `POST /repos/{owner}/{repo}/pulls` with the configured `HEAD_BRANCH` and `BASE_BRANCH`. For each `FindingItem` in the review, it posts a comment to the PR conversation thread with the severity tag, title, description, and suggestion. Finally, it posts a comment to the original issue with the PR URL.

On success, `route_after_pr` routes to Stage 9. The memory update node calls the LLM to extract a `PRMemory` describing the issue type, approach, review patterns to watch for, and lessons learned, and saves it to the `InMemoryStore`. Future runs handling similar issues will receive this memory in the Stage 1 prompt.

---

## Known limitations and future improvements

**Subprocess test execution.** The testing node runs pytest via `subprocess.run` in a local temporary directory. Generated code runs with the same filesystem access as the pipeline process. The documented upgrade path is to replace the `subprocess.run` call with a Docker SDK call that runs pytest inside a container with a restricted filesystem.

**In-memory checkpointing.** `MemorySaver` stores graph checkpoints in RAM only. A process restart loses all in-progress pipeline runs, including any paused at the HITL gate. The documented upgrade is `SqliteSaver` for single-process persistence or `PostgresCheckpointer` for multi-process shared state.

**In-memory long-term memory.** `InMemoryStore` holds `PRMemory` records and run metrics in RAM only. All accumulated memory from previous runs is lost on restart. The documented upgrade, specified by a single-line change in `memory/store.py`, is to replace the `InMemoryStore()` initialization with `AsyncPostgresStore` backed by pgvector. Memory retrieval is also currently by recency rather than semantic similarity; the vector store enables similarity-based retrieval over the stored memories.

**Gemini free-tier quota.** The Gemini free tier allows 20 requests per day per Google Cloud project. A single end-to-end pipeline run makes approximately 8-12 LLM calls (requirements, architecture, N coding agents, N fix iterations, 3 reviewers, synthesizer, PR author, memory extraction). Running more than one or two issues per day against a single key exhausts the quota. Adding `GOOGLE_API_KEY_2` and `GOOGLE_API_KEY_3` from separate GCP projects extends this to 60 requests per day across three quota pools before Groq fallback activates.

**Checkpoint deserialization warnings.** LangGraph's `MemorySaver` produces "Deserializing unregistered type" warnings for Pydantic model fields (`RequirementsSpec`, `ImplementationPlan`, `TestReport`, `ReviewFindings`) when checkpoints are read after HITL resume. The data is still deserialized correctly as plain dicts, which is why `route_after_testing` handles both Pydantic objects and dicts for `test_results.final_status`. The fix is to register these types in LangGraph's `allowed_msgpack_modules` configuration.

**No webhook queuing or deduplication.** The webhook handler calls `run_pipeline_for_issue()` synchronously in the request handler. Concurrent incoming issues will each run their own pipeline instances simultaneously, competing for the same LLM quota. The documented upgrade is to move pipeline execution to a background task queue so that the HTTP response is returned immediately and pipelines are executed sequentially or with controlled concurrency.

**PR author commits via Contents API.** The PR author node opens a PR pointing to `HEAD_BRANCH` but does not actually commit the generated files to that branch. It uses the GitHub REST API only to create the PR object, post comments, and link the issue. In a real deployment, the generated code would need to be committed to `HEAD_BRANCH` via the GitHub Contents API or git operations before opening the PR.

---

## LangSmith observability

LangSmith provides a full execution trace for every pipeline run: the graph structure, which nodes ran, their start and end times, the LLM inputs and outputs for every call, token counts, and error details for any node that raised an exception.

Enable it with two environment variables:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your key>
```

Traces appear at [smith.langchain.com](https://smith.langchain.com) under the project named by `LANGCHAIN_PROJECT`. Each run is searchable by issue number via the `issue-{number}` tag applied in `build_pipeline_config()`.
