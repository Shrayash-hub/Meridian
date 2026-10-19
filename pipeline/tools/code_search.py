"""
pipeline/tools/code_search.py — Codebase Search RAG Tool

This module provides AST-based Python code chunking, FAISS vector indexing,
and semantic code search for the SDLC pipeline's RAG capability.

The index is built once via build_index.py (or test suites) and read at
runtime by agent nodes (Requirements and Architecture).

Ponytail architecture:
Uses Python's standard library ast module to chunk code by top-level functions,
classes, and module-level code. While Tree-sitter provides better multi-language
support and parsing of arbitrary language constructs, Python's ast module requires
zero new dependencies and provides clean, native boundary detection for Python
codebases. Tree-sitter is the documented upgrade path when multi-language support
is required.
"""

import os
import ast
import logging
from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import FAISS
from pipeline.models import embeddings_model

logger = logging.getLogger(__name__)

# Default path for the FAISS index.
# Override with INDEX_PATH environment variable.
DEFAULT_INDEX_PATH = "codebase_index"


# ponytail: uses stdlib ast for chunking — zero new dependencies.
# Tree-sitter provides better accuracy for complex/nested code
# and supports non-Python languages. Upgrade when multi-language
# support is needed or ast chunking proves insufficient.
def chunk_python_file(
    file_path: str,
    source_code: str,
) -> list[dict]:
    """
    Chunks a Python source file into semantically meaningful units
    using Python's ast module.

    Returns a list of dicts:
    {
        "content": str,      # the code chunk text
        "metadata": {
            "file_path": str,   # relative file path
            "chunk_type": str,  # "function", "class", or "module"
            "name": str,        # function/class name or "module_level"
            "start_line": int,
            "end_line": int,
        }
    }
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.warning(
            "Failed to parse Python file",
            extra={"file_path": file_path, "error": str(e)},
        )
        return [
            {
                "content": source_code,
                "metadata": {
                    "file_path": file_path,
                    "chunk_type": "module",
                    "name": "unparseable",
                    "start_line": 1,
                    "end_line": len(source_code.splitlines()),
                },
            }
        ]

    lines = source_code.splitlines()
    chunks = []
    covered_lines = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1  # convert to 0-indexed
            end = node.end_lineno  # end_lineno is inclusive

            chunk_lines = lines[start:end]
            content = "\n".join(chunk_lines)

            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"

            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        "file_path": file_path,
                        "chunk_type": chunk_type,
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                    },
                }
            )

            for i in range(start, end):
                covered_lines.add(i)

    module_lines = [
        line for i, line in enumerate(lines)
        if i not in covered_lines and line.strip()
    ]
    if module_lines:
        chunks.insert(
            0,
            {
                "content": "\n".join(module_lines),
                "metadata": {
                    "file_path": file_path,
                    "chunk_type": "module",
                    "name": "module_level",
                    "start_line": 1,
                    "end_line": len(lines),
                },
            },
        )

    return chunks


def index_codebase(
    codebase_path: str,
    index_path: str = DEFAULT_INDEX_PATH,
    extensions: list[str] = None,
) -> int:
    """
    Indexes all Python files in codebase_path and saves a FAISS vector index to index_path.
    Returns the total number of chunks indexed.
    """
    if extensions is None:
        extensions = [".py"]

    all_files = []
    skip_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build"}
    for ext in extensions:
        for p in Path(codebase_path).rglob(f"*{ext}"):
            if not any(part in skip_dirs for part in p.parts):
                all_files.append(p)

    logger.info("Found files to index", extra={"count": len(all_files)})

    all_chunks = []
    for file_path in all_files:
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            relative_path = str(file_path.relative_to(codebase_path))
            chunks = chunk_python_file(relative_path, source)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(
                "Failed to chunk file",
                extra={"file": str(file_path), "error": str(e)},
            )
            continue

    if not all_chunks:
        logger.warning("No chunks produced from codebase")
        return 0

    logger.info("Total chunks produced", extra={"count": len(all_chunks)})

    texts = [c["content"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    logger.info("Building FAISS index", extra={"chunks": len(texts)})

    vectorstore = FAISS.from_texts(
        texts,
        embeddings_model,
        metadatas=metadatas,
    )
    vectorstore.save_local(index_path)

    logger.info(
        "Index saved",
        extra={"index_path": index_path, "chunks": len(texts)},
    )

    return len(texts)


def search_codebase(
    query: str,
    k: int = 3,
    index_path: str = None,
) -> str:
    """
    Searches the FAISS index for code relevant to the query.
    Returns a formatted string of results suitable for injection
    into an agent's human message.
    Returns empty string if no index exists.
    """
    resolved_path = index_path or os.getenv("INDEX_PATH", DEFAULT_INDEX_PATH)

    index_file = Path(resolved_path) / "index.faiss"
    if not index_file.exists():
        logger.info(
            "No codebase index found, skipping search",
            extra={"index_path": resolved_path},
        )
        return ""

    try:
        # allow_dangerous_deserialization=True is safe here because we built
        # the index ourselves from our own codebase — it is not loading untrusted data.
        vectorstore = FAISS.load_local(
            resolved_path,
            embeddings_model,
            allow_dangerous_deserialization=True,
        )
        results = vectorstore.similarity_search(query, k=k)

        if not results:
            return ""

        formatted = ["Relevant existing code from codebase:\n"]
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            formatted.append(
                f"[{i}] {meta.get('file_path', 'unknown')} "
                f"({meta.get('chunk_type', 'code')} "
                f"'{meta.get('name', '')}', "
                f"lines {meta.get('start_line', '?')}"
                f"-{meta.get('end_line', '?')}):"
            )
            formatted.append(f"```python\n{doc.page_content}\n```")
            formatted.append("")

        return "\n".join(formatted)

    except Exception as e:
        logger.warning(
            "Codebase search failed",
            extra={"query": query[:50], "error": str(e)},
        )
        return ""
