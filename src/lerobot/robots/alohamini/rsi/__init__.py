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

from .claude_proposer import ClaudeProposer
from .evolution_loop import (
    EpisodeOutcome,
    EvolutionLoop,
    LoopConfig,
    Proposal,
    ProposalContext,
    ProposedHypothesis,
)
from .experiment_selection import (
    CostWeights,
    Experiment,
    Selection,
    SelectionConfig,
    expected_information_gain,
    experiment_cost,
    select_experiment,
)
from .harness import (
    ALIGN,
    ALIGN_BASE,
    POLICY,
    Harness,
    HarnessError,
    HarnessLimits,
    ResolvedStep,
    Skill,
    Step,
    code_payload,
    skill_payload,
    total_time_s,
)
from .hypothesis_graph import (
    ACTIVE,
    CODE,
    INCONCLUSIVE,
    PAIR,
    PROMOTED,
    REFUTED,
    SKILL,
    SUPERSEDED,
    Hypothesis,
    HypothesisGraph,
    binary_entropy,
    classify_matched,
    prob_support_rate_above_half,
)
from .memory import ExperienceRecord, ExperienceStore

__all__ = [
    "ALIGN",
    "ALIGN_BASE",
    "POLICY",
    "ACTIVE",
    "CODE",
    "PAIR",
    "PROMOTED",
    "REFUTED",
    "SKILL",
    "SUPERSEDED",
    "ClaudeProposer",
    "CostWeights",
    "EpisodeOutcome",
    "EvolutionLoop",
    "ExperienceRecord",
    "ExperienceStore",
    "Experiment",
    "Harness",
    "HarnessError",
    "HarnessLimits",
    "Hypothesis",
    "HypothesisGraph",
    "INCONCLUSIVE",
    "LoopConfig",
    "Proposal",
    "ProposalContext",
    "ProposedHypothesis",
    "ResolvedStep",
    "Selection",
    "SelectionConfig",
    "Skill",
    "Step",
    "binary_entropy",
    "classify_matched",
    "code_payload",
    "expected_information_gain",
    "experiment_cost",
    "prob_support_rate_above_half",
    "select_experiment",
    "skill_payload",
    "total_time_s",
]
