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

"""The fake cluster used for recorded runs must not change what the model sees."""

from fastmcp import Client

from kubeflow_mcp import trainer
from kubeflow_mcp.core import server as core_server
from kubeflow_mcp.core.server import create_server
from kubeflow_mcp.trainer.api import planning
from tests.eval.conftest import payload
from tests.eval.golden.harness import fake_cluster, server_instructions
from tests.eval.golden.scorer import load_scenarios

SCENARIOS = {s["id"]: s for s in load_scenarios()}


async def test_the_model_gets_the_server_instructions():
    """Guards the system prompt: this accessor changed with FastMCP 4."""
    async with Client(create_server(persona="data-scientist")) as client:
        assert "pre_flight" in server_instructions(client)


async def _schemas() -> dict:
    async with Client(create_server(persona="platform-admin")) as client:
        return {t.name: (t.input_schema, t.description) for t in await client.list_tools()}


async def test_tool_schemas_are_unchanged_under_the_fakes():
    real = await _schemas()
    with fake_cluster(SCENARIOS["finetune-no-gpus"]["cluster"]):
        assert await _schemas() == real


async def test_real_planning_logic_runs_on_top_of_the_fakes():
    with fake_cluster(SCENARIOS["finetune-no-gpus"]["cluster"]):
        async with Client(create_server(persona="data-scientist")) as client:
            result = await client.call_tool("pre_flight", {"model": "google/gemma-2b"})
    data = payload(result)["data"]
    assert data["cluster"]["gpu_total"] == 0
    assert data["tool_selection"]["recommended"] == "run_custom_training"


async def test_blockers_come_from_the_scenario():
    with fake_cluster(SCENARIOS["preflight-blockers"]["cluster"]):
        async with Client(create_server(persona="data-scientist")) as client:
            result = await client.call_tool("pre_flight", {})
    body = payload(result)
    assert body["data"]["compatibility"]["blockers"]
    assert "next" not in body["_meta"]


def test_everything_is_restored():
    tools = list(trainer.TOOLS)
    health = list(core_server.HEALTH_TOOLS)
    check = planning.check_compatibility
    with fake_cluster({}):
        assert planning.check_compatibility is not check
    assert trainer.TOOLS == tools
    assert core_server.HEALTH_TOOLS == health
    assert planning.check_compatibility is check
