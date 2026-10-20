#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""The harness around the frozen VLA: the only thing EmbodiedRSI hypotheses are allowed to change.

A skill is one way of invoking the policy (an instruction and a time budget). The plan is the code: which
skills run, in which order and for how long. A code hypothesis replaces the plan, a skill hypothesis replaces
or adds skills. Nothing here can command a motor, so whatever the proposer writes, the robot only ever
executes the trained policy under the existing safety guard, within ``HarnessLimits``.

One built-in step is not a policy call: ``align_base`` runs the table alignment behaviour (the base drives back
to the taught pose at the table, arms and lift held still). A plan may use it only when the runner supports it
(``HarnessLimits.max_align_steps > 0``); its parameters are fixed by the runner, so a hypothesis can only choose
where it goes in the plan and how long it may take.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from .hypothesis_graph import CODE, SKILL, Hypothesis


class HarnessError(ValueError):
    pass


ALIGN_BASE = "align_base"
POLICY, ALIGN = "policy", "align"
ALIGN_INSTRUCTION = "(built-in) align the base to the table"


@dataclass(frozen=True)
class HarnessLimits:
    max_steps: int = 12
    max_total_time_s: float = 180.0
    min_step_time_s: float = 2.0
    max_instruction_chars: int = 200
    # Built-in align_base steps allowed per plan (0 = the runner cannot align the base).
    max_align_steps: int = 0
    align_time_s: float = 90.0


@dataclass(frozen=True)
class Skill:
    instruction: str
    time_s: float


@dataclass(frozen=True)
class Step:
    skill: str
    time_s: float | None = None


@dataclass(frozen=True)
class ResolvedStep:
    skill: str
    instruction: str
    time_s: float
    kind: str = POLICY


def code_payload(plan: Iterable[Step]) -> str:
    return json.dumps({"plan": [{"skill": step.skill, "time_s": step.time_s} for step in plan]})


def skill_payload(skills: dict[str, Skill]) -> str:
    return json.dumps(
        {
            "skills": [
                {"name": name, "instruction": skill.instruction, "time_s": skill.time_s}
                for name, skill in skills.items()
            ]
        }
    )


def total_time_s(steps: Iterable[ResolvedStep]) -> float:
    return sum(step.time_s for step in steps)


@dataclass(frozen=True)
class Harness:
    skills: dict[str, Skill]
    plan: tuple[Step, ...]

    @classmethod
    def single_task(cls, instruction: str, time_s: float) -> "Harness":
        """The un-evolved baseline: one policy call with the task instruction."""
        return cls(skills={"task": Skill(instruction, time_s)}, plan=(Step("task"),))

    def resolve(self, limits: HarnessLimits | None = None) -> tuple[ResolvedStep, ...]:
        """The exact sequence of policy calls this harness makes, or ``HarnessError`` if it is not allowed."""
        limits = limits or HarnessLimits()
        if not 1 <= len(self.plan) <= limits.max_steps:
            raise HarnessError(f"Plan needs 1..{limits.max_steps} steps, got {len(self.plan)}")
        if ALIGN_BASE in self.skills:
            raise HarnessError(f"'{ALIGN_BASE}' is a built-in step and cannot be defined as a skill")
        n_align = sum(step.skill == ALIGN_BASE for step in self.plan)
        if n_align > limits.max_align_steps:
            raise HarnessError(
                f"Plan uses '{ALIGN_BASE}' {n_align} times; this runner allows {limits.max_align_steps}"
            )
        steps = []
        for step in self.plan:
            if step.skill == ALIGN_BASE:
                time_s = limits.align_time_s if not step.time_s else step.time_s
                if (
                    not isinstance(time_s, int | float)
                    or not limits.min_step_time_s <= time_s <= limits.align_time_s
                ):
                    raise HarnessError(
                        f"'{ALIGN_BASE}' needs {limits.min_step_time_s:g}..{limits.align_time_s:g}s, got {time_s}"
                    )
                steps.append(ResolvedStep(ALIGN_BASE, ALIGN_INSTRUCTION, float(time_s), ALIGN))
                continue
            skill = self.skills.get(step.skill)
            if skill is None:
                raise HarnessError(f"Plan uses unknown skill '{step.skill}'")
            time_s = skill.time_s if step.time_s is None else step.time_s
            instruction = skill.instruction.strip()
            if not instruction or len(instruction) > limits.max_instruction_chars:
                raise HarnessError(
                    f"Skill '{step.skill}' needs an instruction of 1..{limits.max_instruction_chars} chars"
                )
            if not isinstance(time_s, int | float) or time_s < limits.min_step_time_s:
                raise HarnessError(f"Step '{step.skill}' must last at least {limits.min_step_time_s}s")
            steps.append(ResolvedStep(step.skill, instruction, float(time_s)))
        if total_time_s(steps) > limits.max_total_time_s:
            raise HarnessError(
                f"Plan lasts {total_time_s(steps):.0f}s, more than the {limits.max_total_time_s:.0f}s allowed"
            )
        return tuple(steps)

    def apply(self, hypotheses: Iterable[Hypothesis]) -> "Harness":
        """This harness with code and skill hypotheses applied (pair nodes carry no payload of their own)."""
        harness = self
        for hypothesis in hypotheses:
            try:
                payload = json.loads(hypothesis.payload)
                if hypothesis.kind == CODE:
                    plan = tuple(Step(str(item["skill"]), item.get("time_s")) for item in payload["plan"])
                    harness = replace(harness, plan=plan)
                elif hypothesis.kind == SKILL:
                    skills = dict(harness.skills)
                    for item in payload["skills"]:
                        skills[str(item["name"])] = Skill(str(item["instruction"]), item["time_s"])
                    harness = replace(harness, skills=skills)
            except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
                raise HarnessError(f"Malformed payload in hypothesis '{hypothesis.id}': {error}") from error
        return harness

    def skill_names_changed_by(self, hypothesis: Hypothesis) -> set[str]:
        if hypothesis.kind != SKILL:
            return set()
        return {str(item["name"]) for item in json.loads(hypothesis.payload)["skills"]}

    def to_dict(self) -> dict[str, Any]:
        return {
            "skills": {
                name: {"instruction": skill.instruction, "time_s": skill.time_s}
                for name, skill in self.skills.items()
            },
            "plan": [{"skill": step.skill, "time_s": step.time_s} for step in self.plan],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Harness":
        return cls(
            skills={name: Skill(**skill) for name, skill in data["skills"].items()},
            plan=tuple(Step(**step) for step in data["plan"]),
        )
