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

"""Checks the golden scenarios and the scorer against each other.

Every scenario's good example must pass every rule, and every bad example must
fail exactly the rule it names. That keeps the rules meaningful (a rule nothing
can fail is useless) and the examples honest.
"""

import asyncio

import pytest
from fastmcp import Client

from kubeflow_mcp.core.server import create_server
from tests.eval.golden.scorer import RULES, load_scenarios, score

SCENARIOS = load_scenarios()
BAD_CASES = [(s, i) for s in SCENARIOS for i in range(len(s["examples"].get("bad", [])))]


def _ids(case):
    return case["id"] if isinstance(case, dict) else f"{case[0]['id']}-bad{case[1]}"


def test_scenarios_are_well_formed():
    ids = [s["id"] for s in SCENARIOS]
    assert len(ids) == len(set(ids)), "duplicate scenario id"
    for s in SCENARIOS:
        assert s["source"], f"{s['id']}: rules must cite the instruction they come from"
        assert any(r.get("critical") for r in s["rules"]), f"{s['id']}: needs a critical rule"
        for rule in s["rules"]:
            assert len(set(rule) & set(RULES)) == 1, f"{s['id']}: unknown rule {rule}"


def test_every_named_tool_exists():
    """Scenarios must not reference tools the server does not have."""

    async def names() -> set[str]:
        async with Client(create_server(persona="platform-admin")) as c:
            return {t.name for t in await c.list_tools()}

    real = asyncio.run(names()) | {"execute_tool"}
    for s in SCENARIOS:
        runs = [s["examples"]["good"]] + [b["run"] for b in s["examples"].get("bad", [])]
        for event in (e for run in runs for e in run if e["type"] == "tool"):
            assert event["name"] in real, f"{s['id']}: unknown tool {event['name']}"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids)
def test_good_example_passes_every_rule(scenario):
    result = score(scenario, scenario["examples"]["good"])
    failed = [(r.rule, r.detail) for r in result.rules if not r.passed]
    assert failed == [], failed


@pytest.mark.parametrize("case", BAD_CASES, ids=_ids)
def test_bad_example_fails_exactly_its_rule(case):
    scenario, index = case
    bad = scenario["examples"]["bad"][index]
    result = score(scenario, bad["run"])
    failed = sorted({r.rule for r in result.rules if not r.passed})
    assert failed == [bad["breaks"]]
