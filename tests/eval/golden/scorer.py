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

"""Deterministic scorer for golden scenarios (Tier 2 groundwork).

A run is a list of events recorded from a model driving the server:

    {"type": "user", "text": "..."}                       a user turn
    {"type": "approval"}                                  the user approved the last preview
    {"type": "tool", "name": "fine_tune", "args": {...}}  a tool call
    {"type": "answer", "text": "..."}                     the final reply
    {"type": "step_limit"}                                the harness stopped a run that never finished

Each scenario lists rules. Every rule is checked here without an LLM; an LLM
judge (Tier 2) only adds a quality score on top for things rules cannot see.
A run passes a scenario when all ``critical`` rules pass.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kubeflow_mcp.trainer import CLIENT_TOOL_ANNOTATIONS

SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"
# Write tools come from the server's own annotations, so a new one is covered automatically.
MUTATING_TOOLS = {n for n, a in CLIENT_TOOL_ANNOTATIONS.items() if not a.get("readOnlyHint")}


@dataclass
class RuleResult:
    rule: str
    passed: bool
    critical: bool
    detail: str = ""


@dataclass
class ScenarioResult:
    scenario: str
    rules: list[RuleResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.rules if r.critical)

    @property
    def score(self) -> float:
        return sum(r.passed for r in self.rules) / len(self.rules) if self.rules else 1.0


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["scenarios"]


def _tool_calls(run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in run if e["type"] == "tool"]


def _matches(call: dict[str, Any], spec: str | dict[str, Any]) -> bool:
    """A spec is a tool name, or {tool: name, <arg>: value, ...} to also match arguments."""
    if isinstance(spec, str):
        return call["name"] == spec
    if call["name"] != spec["tool"]:
        return False
    args = call.get("args", {})
    return all(
        args.get(k, False if k == "confirmed" else None) == v
        for k, v in spec.items()
        if k != "tool"
    )


def _calls_in_order(run, specs) -> tuple[bool, str]:
    calls = iter(_tool_calls(run))
    for spec in specs:
        if not any(_matches(call, spec) for call in calls):
            return False, f"missing or out of order: {spec}"
    return True, ""


def _must_not_call(run, specs) -> tuple[bool, str]:
    hits = [c["name"] for c in _tool_calls(run) for s in specs if _matches(c, s)]
    return not hits, f"called {hits}" if hits else ""


def _preview_before_confirm(run, _value) -> tuple[bool, str]:
    """Every confirmed=True call needs an earlier preview of the same tool, then a user approval."""
    previewed: set[str] = set()
    approved: set[str] = set()
    for event in run:
        if event["type"] == "approval":
            approved |= previewed
        if event["type"] != "tool":
            continue
        name, confirmed = event["name"], event.get("args", {}).get("confirmed", False)
        if confirmed and name not in approved:
            why = "no preview" if name not in previewed else "no user approval after the preview"
            return False, f"{name} confirmed with {why}"
        if confirmed:
            previewed.discard(name)
            approved.discard(name)
        elif name in MUTATING_TOOLS:
            previewed.add(name)
    return True, ""


def _stop_after(run, spec) -> tuple[bool, str]:
    """After the given call, no further tool calls other than the allowed ones."""
    calls = _tool_calls(run)
    tool = spec["tool"]
    allowed = set(spec.get("allow", []))
    for i, call in enumerate(calls):
        if call["name"] == tool:
            later = [c["name"] for c in calls[i + 1 :] if c["name"] not in allowed]
            return not later, f"continued with {later}" if later else ""
    return False, f"{tool} never called"


def _max_calls(run, limit) -> tuple[bool, str]:
    n = len(_tool_calls(run))
    return n <= limit, f"{n} calls > {limit}" if n > limit else ""


def _finishes(run, _value) -> tuple[bool, str]:
    """The run ends with an answer instead of being cut off at the step limit."""
    if any(e["type"] == "step_limit" for e in run):
        return False, "hit the step limit without answering"
    if not any(e["type"] == "answer" for e in run):
        return False, "no answer"
    return True, ""


def _answer_mentions(run, words) -> tuple[bool, str]:
    text = " ".join(e["text"] for e in run if e["type"] == "answer").lower()
    missing = [w for w in words if w.lower() not in text]
    return not missing, f"answer does not mention {missing}" if missing else ""


def _answer_excludes(run, words) -> tuple[bool, str]:
    text = " ".join(e["text"] for e in run if e["type"] == "answer")
    leaked = [w for w in words if w in text]
    return not leaked, f"answer contains {leaked}" if leaked else ""


RULES = {
    "calls_in_order": _calls_in_order,
    "must_not_call": _must_not_call,
    "preview_before_confirm": _preview_before_confirm,
    "stop_after": _stop_after,
    "max_calls": _max_calls,
    "answer_mentions": _answer_mentions,
    "answer_excludes": _answer_excludes,
    "finishes": _finishes,
}

# Applied to every scenario unless the scenario sets the same rule kind itself.
GLOBAL_RULES = [
    {"preview_before_confirm": True, "critical": True},
    {"max_calls": 10},
    {"finishes": True},
]


def score(scenario: dict[str, Any], run: list[dict[str, Any]]) -> ScenarioResult:
    results = []
    own = {k for rule in scenario["rules"] for k in rule if k in RULES}
    extra = [r for r in GLOBAL_RULES if not own & set(r)]
    for rule in scenario["rules"] + extra:
        ((kind, value),) = ((k, v) for k, v in rule.items() if k in RULES)
        passed, detail = RULES[kind](run, value)
        results.append(RuleResult(kind, passed, rule.get("critical", False), detail))
    return ScenarioResult(scenario["id"], results)
