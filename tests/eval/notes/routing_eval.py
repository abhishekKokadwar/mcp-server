"""Measure find_tools routing: does each request return the expected tool?

Runs the real find_tools from kubeflow_mcp.core.dynamic_tools (semantic mode). With
sentence-transformers installed it uses embeddings; without, it falls back to keyword
search. Prints top-1 and top-3 accuracy plus the misses.
"""

import json
import sys

from kubeflow_mcp.core import dynamic_tools as dt
from kubeflow_mcp.core.server import create_server

queries = json.load(open(sys.argv[1]))
create_server(persona="platform-admin", mode="semantic")

top1 = top3 = 0
misses = []
method = None
for query, expected in queries:
    res = dt.find_tools(query, top_k=3)
    names = [t["name"] for t in res.get("tools", res.get("results", []))]
    method = res.get("mode", "semantic")
    top1 += names[:1] == [expected]
    top3 += expected in names
    if names[:1] != [expected]:
        misses.append((query, expected, names))

n = len(queries)
print(f"method={method}  top-1 {top1}/{n} ({top1 / n:.0%})  top-3 {top3}/{n} ({top3 / n:.0%})")
for query, expected, names in misses:
    print(f"  MISS  {query!r:62} expected {expected:22} got {names}")
