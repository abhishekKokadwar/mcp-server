# Copyright The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record real model runs for the golden scenarios (Tier 2 groundwork).

A local model (Ollama) talks to the real MCP server through an in-memory client:
it gets the real server instructions, the real tool schemas for the scenario's
persona, and the real ``_meta`` hints. Only the tools that would touch a cluster
are faked, from the scenario's ``cluster`` block. Planning logic on top of them
(``pre_flight``'s tool selection, ``estimate_resources``) runs for real.

The simulated user approves the first preview only when the scenario says
``approve: true``, and that approval is recorded as an explicit event.

Usage:
    python -m tests.eval.golden.harness --model qwen2.5:3b [--repeat 3] [--only ID]
"""

import argparse
import asyncio
import contextlib
import functools
import json
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
from fastmcp import Client

from kubeflow_mcp import trainer
from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse
from kubeflow_mcp.core import server as core_server
from kubeflow_mcp.core.server import create_server
from kubeflow_mcp.trainer.api import training
from tests.eval.golden.scorer import MUTATING_TOOLS, load_scenarios, score

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MAX_STEPS = 12
NUM_CTX = 16384
# Sampling for the next run. main() changes the seed per repeat when temperature > 0.
SAMPLING: dict[str, float | int] = {"temperature": 0, "seed": 7}
APPROVAL_TEXT = "Looks good, go ahead."


# --------------------------------------------------------------------------- fake cluster


def _job(cluster: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((j for j in cluster.get("jobs", []) if j["name"] == name), None)


def _not_found(what: str) -> dict[str, Any]:
    return ToolError(error=f"{what} not found", error_code="RESOURCE_NOT_FOUND").model_dump()


def _fakes(cluster: dict[str, Any]) -> dict[str, Callable[..., dict[str, Any]]]:  # noqa: C901
    """Cluster-facing tools answered from the scenario, in the real response shapes."""
    runtimes = cluster.get("runtimes", [])
    blockers = ["TrainJob CRD not installed"] if cluster.get("trainjob_crd") == "missing" else []

    def check_compatibility(**_):
        return ToolResponse(
            data={
                "compatible": not blockers,
                "blockers": blockers,
                "platform": cluster.get("platform", "kubernetes"),
                "kubernetes_version": "1.30",
            }
        ).model_dump()

    def get_cluster_resources(**_):
        gpus = cluster.get("gpu_total", 0)
        return ToolResponse(
            data={"gpu_total": gpus, "nodes": [{"name": "node-1", "gpus": gpus, "cpu": "16"}]}
        ).model_dump()

    def list_runtimes(**_):
        return ToolResponse(
            data={"runtimes": [{"name": r} for r in runtimes], "count": len(runtimes)}
        ).model_dump()

    def get_runtime(name, include_packages=False, **_):
        if name not in runtimes:
            return _not_found(f"Runtime '{name}'")
        data: dict[str, Any] = {"name": name, "image": f"ghcr.io/kubeflow/{name}:latest"}
        if include_packages:
            data["packages"] = cluster.get("packages", {}).get(name, [])
        return ToolResponse(data=data).model_dump()

    def list_training_jobs(**_):
        jobs = [{"name": j["name"], "status": j["status"]} for j in cluster.get("jobs", [])]
        return ToolResponse(data={"jobs": jobs, "count": len(jobs)}).model_dump()

    def get_training_job(name, **_):
        job = _job(cluster, name)
        if job is None:
            return _not_found(f"Training job '{name}'")
        return ToolResponse(data={"name": name, "status": job["status"]}).model_dump()

    def get_training_logs(name, **_):
        job = _job(cluster, name)
        if job is None:
            return _not_found(f"Training job '{name}'")
        return ToolResponse(data={"job": name, "logs": job.get("logs", "")}).model_dump()

    def get_training_events(name, **_):
        job = _job(cluster, name)
        if job is None:
            return _not_found(f"Training job '{name}'")
        return ToolResponse(data={"job": name, "events": job.get("events", [])}).model_dump()

    def wait_for_training(name, **_):
        return get_training_job(name)

    fakes = {
        "check_compatibility": check_compatibility,
        "get_cluster_resources": get_cluster_resources,
        "list_runtimes": list_runtimes,
        "get_runtime": get_runtime,
        "list_training_jobs": list_training_jobs,
        "get_training_job": get_training_job,
        "get_training_logs": get_training_logs,
        "get_training_events": get_training_events,
        "wait_for_training": wait_for_training,
    }

    def mutating(tool: str) -> Callable[..., dict[str, Any]]:
        def run(confirmed: bool = False, **kwargs):
            if tool == "fine_tune":
                # The real GPU gate, reading the faked get_cluster_resources.
                gpu_err = training._check_gpu_available()
                if gpu_err is not None:
                    return gpu_err
            if not confirmed:
                return PreviewResponse(
                    message=f"Will run {tool}. Set confirmed=True to proceed.", config=kwargs
                ).model_dump()
            if tool in {"fine_tune", "run_custom_training", "run_container_training"}:
                job_name = kwargs.get("name") or "train-job-7f3a"
                return ToolResponse(
                    data={
                        "job_name": job_name,
                        "status": "Created",
                        "message": f"Training job '{job_name}' submitted successfully",
                    }
                ).model_dump()
            return ToolResponse(data={"tool": tool, "applied": True, **kwargs}).model_dump()

        return run

    fakes.update({tool: mutating(tool) for tool in MUTATING_TOOLS})
    return fakes


@contextlib.contextmanager
def fake_cluster(cluster: dict[str, Any]) -> Iterator[None]:
    """Swap cluster-facing tools for fakes with identical signatures, then restore them."""
    fakes = _fakes(cluster)
    lists = [trainer.TOOLS, core_server.HEALTH_TOOLS]
    saved_lists = [list(lst) for lst in lists]
    saved_globals: list[tuple[Any, str, Any]] = []
    try:
        for lst in lists:
            for i, original in enumerate(lst):
                impl = fakes.get(original.__name__)
                if impl is None:
                    continue
                fake = functools.wraps(original)(lambda _impl=impl, **kw: _impl(**kw))
                lst[i] = fake
                module = sys.modules[original.__module__]
                saved_globals.append((module, original.__name__, original))
                setattr(module, original.__name__, fake)
                # Planning helpers import some of these by name into other modules.
                for other in list(sys.modules.values()):
                    if getattr(other, "__name__", "").startswith("kubeflow_mcp.trainer") and (
                        getattr(other, original.__name__, None) is original
                    ):
                        saved_globals.append((other, original.__name__, original))
                        setattr(other, original.__name__, fake)
        yield
    finally:
        for lst, saved in zip(lists, saved_lists, strict=True):
            lst[:] = saved
        for module, name, original in reversed(saved_globals):
            setattr(module, name, original)


# --------------------------------------------------------------------------- model loop


def _tool_spec(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


def _chat(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict:
    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            # Ollama defaults to a 4096-token context; the full-mode prompt is ~5k tokens, so it
            # would be cut before the model sees any tool result.
            "options": {**SAMPLING, "num_ctx": NUM_CTX},
        },
        timeout=300,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("prompt_eval_count", 0) >= NUM_CTX - 1:
        raise RuntimeError(f"prompt filled the {NUM_CTX}-token context; the run would be invalid")
    return body["message"]


def _payload(result: Any) -> Any:
    body = result.structured_content
    if isinstance(body, dict) and set(body) == {"result"}:
        return body["result"]
    if body is not None:
        return body
    return "".join(getattr(block, "text", "") for block in result.content)


def server_instructions(client: Client) -> str:
    """Return the server instructions the model is given as its system prompt.

    A run without them measures nothing, so a missing value is an error here.
    """
    instructions = client.instructions
    if not instructions:
        raise RuntimeError("the server returned no instructions; the run would be invalid")
    return instructions


async def record_run(scenario: dict[str, Any], model: str) -> list[dict[str, Any]]:
    """Drive one scenario with the model and return the recorded events."""
    run: list[dict[str, Any]] = [{"type": "user", "text": scenario["request"]}]
    with fake_cluster(scenario.get("cluster", {})):
        mcp = create_server(persona=scenario.get("persona", "readonly"))
        async with Client(mcp) as client:
            tools = [_tool_spec(t) for t in await client.list_tools()]
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": server_instructions(client)},
                {"role": "user", "content": scenario["request"]},
            ]
            approved = False
            saw_preview = False
            for _ in range(MAX_STEPS):
                msg = _chat(model, messages, tools)
                messages.append(msg)
                calls = msg.get("tool_calls") or []
                if msg.get("content"):
                    run.append({"type": "answer", "text": msg["content"]})
                if not calls:
                    if scenario.get("approve") and saw_preview and not approved:
                        approved = True
                        run.append({"type": "approval"})
                        messages.append({"role": "user", "content": APPROVAL_TEXT})
                        continue
                    break
                for call in calls:
                    name = call["function"]["name"]
                    args = call["function"].get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args or "{}")
                    run.append({"type": "tool", "name": name, "args": args})
                    result = await client.call_tool(name, args, raise_on_error=False)
                    body = _payload(result)
                    if isinstance(body, dict) and body.get("status") == "preview":
                        saw_preview = True
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": name,
                            "content": json.dumps(body, default=str),
                        }
                    )
            else:
                run.append({"type": "step_limit"})
    return run


# --------------------------------------------------------------------------- CLI


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="tests/eval/notes/runs")
    opts = parser.parse_args()

    out = Path(opts.out) / opts.model.replace(":", "_")
    out.mkdir(parents=True, exist_ok=True)
    scenarios = [s for s in load_scenarios() if not opts.only or s["id"] == opts.only]
    passed = total = 0
    for scenario in scenarios:
        for i in range(opts.repeat):
            SAMPLING.update(temperature=opts.temperature, seed=opts.seed + i)
            start = time.monotonic()
            run = await record_run(scenario, opts.model)
            result = score(scenario, run)
            total += 1
            passed += result.passed
            (out / f"{scenario['id']}-{i}.json").write_text(json.dumps(run, indent=1))
            fails = [f"{r.rule}: {r.detail}" for r in result.rules if not r.passed]
            print(
                f"{'PASS' if result.passed else 'FAIL'} {scenario['id']:22} run {i} "
                f"score {result.score:.2f} {time.monotonic() - start:5.1f}s  {'; '.join(fails)}",
                flush=True,
            )
    print(f"\n{opts.model}: {passed}/{total} runs passed all critical rules")


if __name__ == "__main__":
    asyncio.run(main())
