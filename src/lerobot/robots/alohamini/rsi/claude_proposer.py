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

"""Claude as the reasoning half of EmbodiedRSI's Fast System and as its Slow System operator.

One request per round reads the episode records (M1), the Hypothesis Graph (M2) and the current rules (M3),
then returns new code, skill and pair hypotheses plus the revised rules. ``anthropic`` is an optional
dependency and is imported on first use.
"""

import base64
import json
from pathlib import Path
from typing import Any

from .evolution_loop import Proposal, ProposalContext, ProposedHypothesis
from .harness import Skill, Step, code_payload, skill_payload
from .hypothesis_graph import CODE, PAIR, SKILL

SYSTEM_PROMPT = """\
You are improving how a robot uses a frozen vision-language-action (VLA) policy. The policy's weights never \
change. What you can change is the harness around it:

- A skill is one way of calling the policy: a short natural-language instruction and a time budget in \
seconds. The policy was fine-tuned on teleoperated demonstrations, so instructions close to what it saw in \
training tend to work and unusual wording can push it out of distribution.
- The plan is the code: an ordered list of skill calls, each optionally overriding the skill's time budget. \
A skill may appear several times, for example to retry.

You work by proposing hypotheses that the robot then tests physically, and physical episodes are scarce: a \
person resets the scene by hand before every one. Each hypothesis is tested with matched trials, meaning the \
same named initial state is run with and without it, so a hypothesis should be a specific, falsifiable claim \
about why episodes fail and what change fixes it.

Three kinds of hypothesis:
- code: replaces the whole plan. It must only use skills that exist in the current harness, unless it is \
meant to run together with a skill hypothesis that adds them, in which case also propose that pair.
- skill: replaces or adds skills by name. Changing a skill the plan already uses is testable on its own; \
adding a new skill only has an effect through a pair.
- pair: one code and one skill hypothesis claimed to do better together than either alone. Refer to members \
by their key from this response or by the id of an existing active hypothesis.

Use `refines` when a hypothesis is a revision of an existing one, so the history stays traceable. Do not \
re-propose hypotheses that were refuted unless you change them in a way the evidence motivates. Prefer a few \
hypotheses that distinguish between different explanations of the failures over many similar ones, since \
every hypothesis costs episodes to test. If the evidence does not suggest anything worth testing, return no \
hypotheses.

Also maintain the cognition list: short, reusable rules about this robot, policy and task that the evidence \
supports and that should guide later rounds. Return the full revised list each time: keep rules that still \
hold, merge overlapping ones, rewrite ones the new evidence refines, and drop ones it contradicts."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "kind": {"type": "string", "enum": [CODE, SKILL, PAIR]},
                    "claim": {"type": "string"},
                    "refines": {"type": "string", "description": "Existing hypothesis id, or empty."},
                    "plan": {
                        "type": "array",
                        "description": "For kind=code: the new plan. Empty otherwise.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "skill": {"type": "string"},
                                "time_s": {
                                    "type": "number",
                                    "description": "Seconds for this step; 0 uses the skill's own budget.",
                                },
                            },
                            "required": ["skill", "time_s"],
                            "additionalProperties": False,
                        },
                    },
                    "skills": {
                        "type": "array",
                        "description": "For kind=skill: skills to replace or add. Empty otherwise.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "instruction": {"type": "string"},
                                "time_s": {"type": "number"},
                            },
                            "required": ["name", "instruction", "time_s"],
                            "additionalProperties": False,
                        },
                    },
                    "members": {
                        "type": "array",
                        "description": "For kind=pair: [code key or id, skill key or id]. Empty otherwise.",
                        "items": {"type": "string"},
                    },
                },
                "required": ["key", "kind", "claim", "refines", "plan", "skills", "members"],
                "additionalProperties": False,
            },
        },
        "cognition": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["analysis", "hypotheses", "cognition"],
    "additionalProperties": False,
}


def describe_context(context: ProposalContext) -> str:
    graph = context.graph
    state = {
        "task": context.task,
        "harness_limits": vars(context.limits),
        "initial_states": context.init_states,
        "current_harness": context.baseline.to_dict(),
        "hypotheses": [
            {
                "id": node.id,
                "kind": node.kind,
                "status": node.status,
                "claim": node.claim,
                "refines": node.refines,
                "members": list(node.members),
                "payload": json.loads(node.payload) if node.payload else None,
                "supporting_trials": node.n_sup,
                "refuting_trials": node.n_ref,
                "confidence": round(graph.confidence(node.id), 3),
            }
            for node in graph.nodes.values()
        ],
        "works_with": [{"code": c, "skill": s, "joint_gain": g} for (c, s), g in graph.works_with.items()],
        "cognition": context.cognition,
        "episodes": [
            {
                "episode": record.episode,
                "variant": record.variant or ["baseline"],
                "harness_version": record.baseline_version,
                "initial_state": record.init_state,
                "steps": record.steps,
                "success": record.success,
                "duration_s": round(record.duration_s, 1),
                "operator_note": record.note,
            }
            for record in context.records
        ],
    }
    return json.dumps(state, indent=2)


def to_proposal(data: dict[str, Any]) -> Proposal:
    hypotheses = []
    for item in data["hypotheses"]:
        payload = ""
        if item["kind"] == CODE:
            payload = code_payload(Step(step["skill"], step["time_s"] or None) for step in item["plan"])
        elif item["kind"] == SKILL:
            payload = skill_payload(
                {skill["name"]: Skill(skill["instruction"], skill["time_s"]) for skill in item["skills"]}
            )
        hypotheses.append(
            ProposedHypothesis(
                key=item["key"],
                kind=item["kind"],
                claim=item["claim"],
                payload=payload,
                refines=item["refines"] or None,
                members=tuple(item["members"]) if item["kind"] == PAIR else (),
            )
        )
    return Proposal(hypotheses=hypotheses, cognition=list(data["cognition"]))


class ClaudeProposer:
    def __init__(
        self, model: str = "claude-opus-5-5", effort: str = "high", max_failure_images: int = 6, client=None
    ):
        self.model = model
        self.effort = effort
        self.max_failure_images = max_failure_images
        self._client = client
        self.last_analysis = ""

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as error:
                raise ImportError(
                    "ClaudeProposer needs the 'anthropic' package: pip install anthropic"
                ) from error
            self._client = anthropic.Anthropic()
        return self._client

    def _content(self, context: ProposalContext) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        failures = [r for r in context.records if not r.success and r.frames]
        for record in failures[-self.max_failure_images :]:
            frame = Path(record.frames[-1])
            if not frame.exists():
                continue
            content.append(
                {
                    "type": "text",
                    "text": f"Final camera frame of failed episode {record.episode} "
                    f"({'+'.join(record.variant) or 'baseline'}, initial state '{record.init_state}'):",
                }
            )
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(frame.read_bytes()).decode("utf-8"),
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": "Current state of the experiment:\n\n"
                + describe_context(context)
                + "\n\nPropose the next hypotheses to test and return the revised cognition list.",
            }
        )
        return content

    def propose(self, context: ProposalContext) -> Proposal:
        response = self._get_client().beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._content(context)}],
        )
        if response.stop_reason != "end_turn":
            raise RuntimeError(f"Claude stopped with '{response.stop_reason}' instead of a proposal")
        data = json.loads(next(block.text for block in response.content if block.type == "text"))
        self.last_analysis = data["analysis"]
        return to_proposal(data)
