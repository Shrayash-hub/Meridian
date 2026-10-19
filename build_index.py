# Run with: python build_index.py <path_to_codebase>
# Example:  python build_index.py /path/to/your/project
# This builds a FAISS vector index of Python files in the
# target directory. Run this once before running the pipeline
# against a new codebase.
# Output: codebase_index/ directory (or INDEX_PATH if set)
# Requires: GOOGLE_API_KEY set in .env

import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from pipeline.tools.code_search import index_codebase

if len(sys.argv) < 2:
    print("Usage: python build_index.py <path_to_codebase>")
    print("Example: python build_index.py /path/to/project")
    sys.exit(1)

codebase_path = sys.argv[1]
index_path = os.getenv("INDEX_PATH", "codebase_index")

if not os.path.exists(codebase_path):
    print(f"Error: path does not exist: {codebase_path}")
    sys.exit(1)

print(f"Indexing: {codebase_path}")
print(f"Index path: {index_path}")

total = index_codebase(codebase_path, index_path)

print(f"\nIndexing complete.")
print(f"Total chunks indexed: {total}")
print(f"Index saved to: {index_path}/")
print(f"\nYou can now run the pipeline against this codebase.")
