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

"""Value-of-Information Experiment Selection of EmbodiedRSI (arXiv:2610.10498, Sec. 3.2, Eq. 6).

Each candidate experiment declares the hypotheses its outcome can update. Its score is the expected drop in
exploration uncertainty divided by its physical cost; the best one is run next, until nothing informative
and affordable is left.

The outcome model (Appendix D) and the thresholds (Appendix E) are not public. Here every hypothesis an
experiment tests is assumed to be supported independently with its posterior mean confidence, and the
defaults in ``SelectionConfig`` are placeholders to tune on the robot.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from .hypothesis_graph import HypothesisGraph, binary_entropy


@dataclass(frozen=True)
class CostWeights:
    """w_tau, w_s, w_g, w_r. With the defaults the cost is wall-clock seconds including the reset."""

    time: float = 1.0
    steps: float = 0.0
    gpu: float = 0.0
    reset: float = 1.0


@dataclass(frozen=True)
class Experiment:
    id: str
    hypotheses: tuple[str, ...]
    time_s: float
    n_steps: float = 0.0
    gpu_cost: float = 0.0
    reset_cost: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class SelectionConfig:
    weights: CostWeights = field(default_factory=CostWeights)
    eps_h: float = 0.3
    eps_voi: float = 1e-4
    n_max: int = 10


@dataclass(frozen=True)
class Selection:
    experiment: Experiment | None
    voi: float
    stop_reason: str | None = None


def experiment_cost(experiment: Experiment, weights: CostWeights) -> float:
    """C(t) = w_tau * T + w_s * N_step + w_g * C_GPU + w_r * C_reset."""
    return (
        weights.time * experiment.time_s
        + weights.steps * experiment.n_steps
        + weights.gpu * experiment.gpu_cost
        + weights.reset * experiment.reset_cost
    )


def expected_information_gain(
    graph: HypothesisGraph, experiment: Experiment, n_max: int | None = None
) -> float:
    """G_k(t): expected reduction of H_exp from running the experiment once (Eq. 6), in bits.

    Hypotheses that already reached ``n_max`` evaluations are not updated any more and contribute nothing.
    """
    gain = 0.0
    for hypothesis_id in experiment.hypotheses:
        if n_max is not None and graph.nodes[hypothesis_id].n_trials >= n_max:
            continue
        p_support = graph.confidence(hypothesis_id)
        gain += (
            binary_entropy(graph.prob_majority(hypothesis_id))
            - p_support * binary_entropy(graph.prob_majority(hypothesis_id, extra_sup=1))
            - (1.0 - p_support) * binary_entropy(graph.prob_majority(hypothesis_id, extra_ref=1))
        )
    return gain


def select_experiment(
    graph: HypothesisGraph,
    experiments: Iterable[Experiment],
    config: SelectionConfig | None = None,
    episodes_left: int | None = None,
) -> Selection:
    """t* = argmax VOI(t) = G(t) / C(t), or a stop reason when exploration should end."""
    config = config or SelectionConfig()
    if episodes_left is not None and episodes_left <= 0:
        return Selection(None, 0.0, "budget_exhausted")
    if graph.active() and all(node.n_trials >= config.n_max for node in graph.active()):
        return Selection(None, 0.0, "evaluation_cap")
    if graph.uncertainty() < config.eps_h:
        return Selection(None, 0.0, "uncertainty_resolved")

    best: Experiment | None = None
    best_voi = 0.0
    for experiment in experiments:
        cost = experiment_cost(experiment, config.weights)
        if cost <= 0:
            raise ValueError(f"Experiment '{experiment.id}' has non-positive cost {cost}")
        voi = expected_information_gain(graph, experiment, config.n_max) / cost
        if best is None or voi > best_voi:
            best, best_voi = experiment, voi
    if best is None or best_voi < config.eps_voi:
        return Selection(None, best_voi, "no_informative_experiment")
    return Selection(best, best_voi)
