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

"""Hypothesis Graph of EmbodiedRSI (arXiv:2610.10498, Sec. 3.2): the M2 level of its memory.

Nodes are code, skill or pair hypotheses, each with a Beta posterior on its support rate. A REFINES edge
links a hypothesis to the one it revises; a WORKS-WITH edge links a code and a skill hypothesis whose joint
execution beat both of them alone from the same initial states (Eq. 7). A pair hypothesis is the claim that
one code and one skill hypothesis are better together than either alone (the "Pair" column of Fig. 2).

The paper's appendices are not public, so the rule that turns a matched trial into support or refutation
(``classify_matched``) is this module's reading of the main text, not the authors' definition.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CODE = "code"
SKILL = "skill"
PAIR = "pair"

ACTIVE = "active"
PROMOTED = "promoted"
REFUTED = "refuted"
SUPERSEDED = "superseded"
INCONCLUSIVE = "inconclusive"


def binary_entropy(q: float) -> float:
    """h2(q) in bits (Eq. 5)."""
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -q * math.log2(q) - (1.0 - q) * math.log2(1.0 - q)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1.0) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1.0)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def prob_support_rate_above_half(a: float, b: float) -> float:
    """P(theta > 1/2) for theta ~ Beta(a, b), i.e. q_i in the paper."""
    x = 0.5
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + (a + b) * math.log(x))
    if x < (a + 1.0) / (a + b + 2.0):
        cdf = front * _beta_continued_fraction(a, b, x) / a
    else:
        cdf = 1.0 - front * _beta_continued_fraction(b, a, x) / b
    return min(1.0, max(0.0, 1.0 - cdf))


def classify_matched(baseline_success: bool, variant_success: bool) -> bool | None:
    """Evidence from one matched trial: the same initial state run without and with the hypothesis.

    True supports the hypothesis, False refutes it, None is uninformative (the harness already succeeded
    without it, so the trial cannot tell the hypothesis apart from doing nothing).
    """
    if variant_success and not baseline_success:
        return True
    if not variant_success:
        return False
    return None


@dataclass
class Hypothesis:
    id: str
    kind: str
    claim: str
    payload: str = ""
    refines: str | None = None
    members: tuple[str, ...] = ()
    status: str = ACTIVE
    n_sup: int = 0
    n_ref: int = 0

    def __post_init__(self):
        self.members = tuple(self.members)

    @property
    def n_trials(self) -> int:
        return self.n_sup + self.n_ref


@dataclass
class HypothesisGraph:
    alpha: float = 1.0
    beta: float = 1.0
    nodes: dict[str, Hypothesis] = field(default_factory=dict)
    works_with: dict[tuple[str, str], float] = field(default_factory=dict)

    def add(self, hypothesis: Hypothesis) -> Hypothesis:
        if hypothesis.kind not in (CODE, SKILL, PAIR):
            raise ValueError(f"Unknown hypothesis kind '{hypothesis.kind}'")
        member_kinds = tuple(self.nodes[m].kind if m in self.nodes else None for m in hypothesis.members)
        if member_kinds != ((CODE, SKILL) if hypothesis.kind == PAIR else ()):
            raise ValueError(
                f"'{hypothesis.id}': a pair needs existing (code, skill) members and nothing else has members"
            )
        if hypothesis.id in self.nodes:
            raise ValueError(f"Hypothesis '{hypothesis.id}' already exists")
        if hypothesis.refines is not None:
            parent = self.nodes.get(hypothesis.refines)
            if parent is None:
                raise ValueError(f"'{hypothesis.id}' refines unknown hypothesis '{hypothesis.refines}'")
            if parent.kind != hypothesis.kind:
                raise ValueError(f"'{hypothesis.id}' ({hypothesis.kind}) cannot refine a {parent.kind} node")
        self.nodes[hypothesis.id] = hypothesis
        return hypothesis

    def record(self, hypothesis_id: str, supported: bool | None) -> None:
        """Count one matched outcome. ``None`` (uninformative) leaves the posterior untouched."""
        node = self.nodes[hypothesis_id]
        if supported is True:
            node.n_sup += 1
        elif supported is False:
            node.n_ref += 1

    def confidence(self, hypothesis_id: str, extra_sup: int = 0, extra_ref: int = 0) -> float:
        """Posterior mean support rate c_i (Eq. 4)."""
        node = self.nodes[hypothesis_id]
        n_sup = node.n_sup + extra_sup
        return (n_sup + self.alpha) / (n_sup + node.n_ref + extra_ref + self.alpha + self.beta)

    def prob_majority(self, hypothesis_id: str, extra_sup: int = 0, extra_ref: int = 0) -> float:
        """q_i = P(theta_i > 1/2 | D)."""
        node = self.nodes[hypothesis_id]
        return prob_support_rate_above_half(
            self.alpha + node.n_sup + extra_sup, self.beta + node.n_ref + extra_ref
        )

    def active(self) -> list[Hypothesis]:
        return [node for node in self.nodes.values() if node.status == ACTIVE]

    def uncertainty(self) -> float:
        """Exploration uncertainty H_exp in bits (Eq. 5), over the hypotheses still being explored."""
        return sum(binary_entropy(self.prob_majority(node.id)) for node in self.active())

    def lineage(self, hypothesis_id: str) -> list[str]:
        """The hypothesis followed by everything it refines, newest first."""
        chain = []
        current: str | None = hypothesis_id
        while current is not None:
            chain.append(current)
            current = self.nodes[current].refines
        return chain

    def record_joint(
        self, code_id: str, skill_id: str, r_joint: float, r_code: float, r_skill: float
    ) -> float:
        """Joint gain w_ij (Eq. 7) from success rates measured on the same initial states.

        A positive gain is kept as a WORKS-WITH edge; a non-positive one removes any earlier edge.
        """
        if self.nodes[code_id].kind != CODE or self.nodes[skill_id].kind != SKILL:
            raise ValueError("record_joint expects a code hypothesis and a skill hypothesis, in that order")
        gain = r_joint - max(r_code, r_skill)
        if gain > 0:
            self.works_with[(code_id, skill_id)] = gain
        else:
            self.works_with.pop((code_id, skill_id), None)
        return gain

    def partners(self, hypothesis_id: str) -> list[tuple[str, float]]:
        """WORKS-WITH partners of a hypothesis, strongest joint gain first."""
        found = []
        for (code_id, skill_id), gain in self.works_with.items():
            if hypothesis_id == code_id:
                found.append((skill_id, gain))
            elif hypothesis_id == skill_id:
                found.append((code_id, gain))
        return sorted(found, key=lambda pair: -pair[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "nodes": [asdict(node) for node in self.nodes.values()],
            "works_with": [
                {"code": code_id, "skill": skill_id, "gain": gain}
                for (code_id, skill_id), gain in self.works_with.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisGraph":
        graph = cls(alpha=data["alpha"], beta=data["beta"])
        for node in data["nodes"]:
            graph.add(Hypothesis(**node))
        for edge in data["works_with"]:
            graph.works_with[(edge["code"], edge["skill"])] = edge["gain"]
        return graph

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HypothesisGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))
