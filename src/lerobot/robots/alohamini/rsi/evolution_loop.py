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

"""EmbodiedRSI's self-evolution loop (arXiv:2610.10498, Fig. 1) for a robot with manual resets.

Each round: pick the experiment with the best value of information, run its missing episodes, turn the
matched outcomes into evidence, keep what is confirmed in the baseline harness, and ask the proposer for new
hypotheses when nothing informative is left.

A matched trial here is a slot (initial state, repeat): every variant of one experiment is run from the same
named initial state, which a person restores by hand. Outcomes are cached per slot, so the baseline episode
of a slot is paid for once and shared by every hypothesis tested there. Hypotheses are only tested from
initial states where the current baseline failed, because a slot where it already succeeds cannot show an
improvement. All state is written to ``run_dir`` after each episode, and a loop created on an existing
``run_dir`` resumes it.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .experiment_selection import Experiment, Selection, SelectionConfig, select_experiment
from .harness import Harness, HarnessError, HarnessLimits, ResolvedStep, total_time_s
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
    classify_matched,
)
from .memory import ExperienceRecord, ExperienceStore

logger = logging.getLogger(__name__)


@dataclass
class EpisodeOutcome:
    success: bool
    duration_s: float
    note: str = ""
    frames: list[str] = field(default_factory=list)
    # Base alignments run in the episode, if the runner aligns the base (see behaviors.summarize_alignment).
    base: list[dict] = field(default_factory=list)


# (policy calls to make, initial state to restore first, label of the variant) -> outcome
RunEpisode = Callable[[tuple[ResolvedStep, ...], str, str], EpisodeOutcome]


@dataclass
class ProposedHypothesis:
    """A proposer's hypothesis. ``key`` is its own name for it, so pairs in the same batch can refer to it."""

    key: str
    kind: str
    claim: str
    payload: str = ""
    refines: str | None = None
    members: tuple[str, ...] = ()


@dataclass
class Proposal:
    hypotheses: list[ProposedHypothesis] = field(default_factory=list)
    cognition: list[str] | None = None


@dataclass
class ProposalContext:
    task: str
    baseline: Harness
    limits: HarnessLimits
    init_states: list[str]
    graph: HypothesisGraph
    records: list[ExperienceRecord]
    cognition: list[str]


class Proposer(Protocol):
    def propose(self, context: ProposalContext) -> Proposal: ...


@dataclass
class LoopConfig:
    episode_budget: int = 50
    reset_time_s: float = 30.0
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    limits: HarnessLimits = field(default_factory=HarnessLimits)
    promote_q: float = 0.9
    refute_q: float = 0.1
    min_trials: int = 3
    max_rounds: int = 6
    records_in_context: int = 30


@dataclass
class _Trial:
    node: Hypothesis
    state: str
    repeat: int
    variants: list[tuple[str, ...]]
    missing: list[tuple[str, ...]]
    tested: list[str]


class EvolutionLoop:
    def __init__(
        self,
        run_dir: str | Path,
        task: str,
        baseline: Harness,
        init_states: list[str],
        run_episode: RunEpisode,
        proposer: Proposer,
        config: LoopConfig | None = None,
    ):
        if not init_states:
            raise ValueError("At least one initial state is required")
        self.run_dir = Path(run_dir)
        self.task = task
        self.init_states = list(init_states)
        self.run_episode = run_episode
        self.proposer = proposer
        self.config = config or LoopConfig()
        self.store = ExperienceStore(self.run_dir / "experience.jsonl")

        baseline.resolve(self.config.limits)
        self.baseline = baseline
        self.baseline_version = 0
        self.graph = HypothesisGraph()
        self.cognition: list[str] = []
        self.outcomes: dict[str, bool] = {}
        self.attempts: dict[str, int] = {}
        self.evidenced: set[str] = set()
        self.pair_stats: dict[str, list[int]] = {}
        self.episodes_used = 0
        self.rounds = 0
        self.counters = {CODE: 0, SKILL: 0}
        if (self.run_dir / "state.json").exists():
            self._load()

    # ------------------------------------------------------------------ persistence

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "task": self.task,
            "init_states": self.init_states,
            "baseline": self.baseline.to_dict(),
            "baseline_version": self.baseline_version,
            "graph": self.graph.to_dict(),
            "cognition": self.cognition,
            "outcomes": self.outcomes,
            "attempts": self.attempts,
            "evidenced": sorted(self.evidenced),
            "pair_stats": self.pair_stats,
            "episodes_used": self.episodes_used,
            "rounds": self.rounds,
            "counters": self.counters,
        }
        tmp = self.run_dir / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.run_dir / "state.json")

    def _load(self) -> None:
        state = json.loads((self.run_dir / "state.json").read_text())
        if state["init_states"] != self.init_states:
            raise ValueError(
                f"{self.run_dir} was started with initial states {state['init_states']}; "
                "resume with the same ones or use a new run directory"
            )
        self.baseline = Harness.from_dict(state["baseline"])
        self.baseline_version = state["baseline_version"]
        self.graph = HypothesisGraph.from_dict(state["graph"])
        self.cognition = state["cognition"]
        self.outcomes = state["outcomes"]
        self.attempts = state["attempts"]
        self.evidenced = set(state["evidenced"])
        self.pair_stats = state["pair_stats"]
        self.episodes_used = state["episodes_used"]
        self.rounds = state["rounds"]
        self.counters = state["counters"]

    # ------------------------------------------------------------------ variants and slots

    @property
    def episodes_left(self) -> int:
        return self.config.episode_budget - self.episodes_used

    def _slot_key(self, variant: tuple[str, ...], state: str, repeat: int) -> str:
        name = "+".join(variant) or "baseline"
        return f"b{self.baseline_version}|{name}|{state}|{repeat}"

    def _steps(self, variant: tuple[str, ...]) -> tuple[ResolvedStep, ...] | None:
        """Policy calls of the baseline with these hypotheses applied, or None if that is not runnable."""
        try:
            harness = self.baseline.apply(self.graph.nodes[i] for i in variant)
            return harness.resolve(self.config.limits)
        except HarnessError:
            return None

    def _effective(self, variant: tuple[str, ...]) -> bool:
        """Runnable and actually different from the baseline, so running it can tell us something."""
        steps = self._steps(variant)
        return steps is not None and steps != self._steps(())

    def _trial(self, node: Hypothesis) -> _Trial | None:
        attempt = self.attempts.get(node.id, 0)
        if attempt >= self.config.selection.n_max:
            return None
        failing = [s for s in self.init_states if self.outcomes.get(self._slot_key((), s, 0)) is False]
        if not failing:
            return None
        state = failing[attempt % len(failing)]
        repeat = attempt // len(failing)
        members = node.members if node.kind == PAIR else (node.id,)
        if not self._effective(members):
            return None
        variants: list[tuple[str, ...]] = [()]
        variants += [(m,) for m in members if len(members) > 1 and self._effective((m,))]
        variants.append(members)
        tested = [node.id]
        if node.kind == PAIR:
            tested += [
                m
                for m in members
                if self.graph.nodes[m].status == ACTIVE
                and self._slot_key((m,), state, repeat) not in self.evidenced
            ]
        missing = [v for v in variants if self._slot_key(v, state, repeat) not in self.outcomes]
        return _Trial(node, state, repeat, variants, missing, tested)

    # ------------------------------------------------------------------ evidence

    def _settle(self, trial: _Trial) -> None:
        """Turn the cached outcomes of a complete matched trial into graph evidence."""
        node, state, repeat = trial.node, trial.state, trial.repeat

        def outcome(variant: tuple[str, ...]) -> bool:
            return self.outcomes[self._slot_key(variant, state, repeat)]

        base = outcome(())
        if node.kind == PAIR:
            joint = outcome(node.members)
            alone = [outcome((m,)) if (m,) in trial.variants else base for m in node.members]
            self.graph.record(node.id, classify_matched(max(alone), joint))
            for member, member_alone in zip(node.members, alone, strict=True):
                if member in trial.tested:
                    reference = member_alone if (member,) in trial.variants else joint
                    self.graph.record(member, classify_matched(base, reference))
                    self.evidenced.add(self._slot_key((member,), state, repeat))
            stats = self.pair_stats.setdefault(node.id, [0, 0, 0, 0])
            for index, value in enumerate((1, joint, *alone)):
                stats[index] += int(value)
            n, n_joint, n_code, n_skill = stats
            self.graph.record_joint(*node.members, n_joint / n, n_code / n, n_skill / n)
        else:
            self.graph.record(node.id, classify_matched(base, outcome((node.id,))))
            self.evidenced.add(self._slot_key((node.id,), state, repeat))
        self._close_slot(node)
        self._review()

    def _close_slot(self, node: Hypothesis) -> None:
        self.attempts[node.id] = self.attempts.get(node.id, 0) + 1
        if self.attempts[node.id] >= self.config.selection.n_max and node.status == ACTIVE:
            node.status = INCONCLUSIVE

    def _baseline_succeeded(self, trial: _Trial) -> bool:
        return self.outcomes.get(self._slot_key((), trial.state, trial.repeat), False)

    def _review(self) -> None:
        """Keep confirmed hypotheses in the baseline and drop refuted ones."""
        for node in self.graph.active():
            if node.status != ACTIVE or node.n_trials < self.config.min_trials:
                continue
            q = self.graph.prob_majority(node.id)
            members = node.members if node.kind == PAIR else (node.id,)
            # A hypothesis that only works through a pair is kept by promoting that pair, not on its own.
            if q >= self.config.promote_q and self._effective(members):
                self._promote(node)
            elif q <= self.config.refute_q:
                node.status = REFUTED
                logger.info("Refuted %s: %s", node.id, node.claim)

    def _promote(self, node: Hypothesis) -> None:
        # A pair is kept whole even if a member was refuted on its own: that is what joint gain means.
        promoted = [self.graph.nodes[m] for m in node.members] if node.kind == PAIR else [node]
        self.baseline = self.baseline.apply(promoted)
        self.baseline_version += 1
        ids = {h.id for h in promoted}
        changed_skills = set().union(*(self.baseline.skill_names_changed_by(h) for h in promoted))
        node.status = PROMOTED
        for hypothesis in promoted:
            hypothesis.status = PROMOTED
        # Everything written against the old baseline that touches what just changed is stale, unless it
        # was proposed as a refinement of what was promoted.
        for other in self.graph.active():
            if other.refines in ids:
                continue
            stale_code = other.kind == CODE and any(h.kind == CODE for h in promoted)
            stale_skill = bool(self.baseline.skill_names_changed_by(other) & changed_skills)
            if stale_code or stale_skill:
                other.status = SUPERSEDED
        for other in self.graph.active():
            stale = (PROMOTED, SUPERSEDED)
            if other.kind == PAIR and any(self.graph.nodes[m].status in stale for m in other.members):
                other.status = SUPERSEDED
        logger.info("Promoted %s into baseline v%d: %s", node.id, self.baseline_version, node.claim)

    # ------------------------------------------------------------------ the loop

    def next_experiment(self) -> tuple[Selection, _Trial | None]:
        """VOI selection over the active hypotheses; trials whose episodes are all cached settle for free."""
        while True:
            trials = {}
            for node in self.graph.active():
                trial = self._trial(node)
                if trial is not None:
                    trials[node.id] = trial
            skipped = next((t for t in trials.values() if self._baseline_succeeded(t)), None)
            free = next((t for t in trials.values() if not t.missing), None)
            if skipped is not None:
                self._close_slot(skipped.node)
            elif free is not None:
                self._settle(free)
            else:
                break
            self.save()
        experiments = [
            Experiment(
                id=node_id,
                hypotheses=tuple(trial.tested),
                time_s=sum(total_time_s(self._steps(v)) for v in trial.missing),
                reset_cost=len(trial.missing) * self.config.reset_time_s,
            )
            for node_id, trial in trials.items()
        ]
        selection = select_experiment(self.graph, experiments, self.config.selection, self.episodes_left)
        return selection, trials.get(selection.experiment.id) if selection.experiment else None

    def _run(self, variant: tuple[str, ...], state: str, repeat: int) -> bool:
        steps = self._steps(variant)
        label = "+".join(variant) or "baseline"
        outcome = self.run_episode(steps, state, label)
        self.outcomes[self._slot_key(variant, state, repeat)] = outcome.success
        self.store.append(
            ExperienceRecord(
                episode=self.episodes_used,
                variant=list(variant),
                baseline_version=self.baseline_version,
                init_state=state,
                repeat=repeat,
                steps=[asdict(step) for step in steps],
                success=outcome.success,
                duration_s=outcome.duration_s,
                note=outcome.note,
                frames=outcome.frames,
                timestamp=time.time(),
                base=outcome.base,
            )
        )
        self.episodes_used += 1
        self.save()
        return outcome.success

    def run_experiment(self, trial: _Trial) -> None:
        for variant in trial.missing:
            if self.episodes_left <= 0:
                return
            if self._run(variant, trial.state, trial.repeat) and variant == ():
                self._close_slot(trial.node)
                self.save()
                return
        self._settle(trial)
        self.save()

    def _ensure_baseline(self) -> bool:
        """Run the current baseline once from every initial state; False if the budget ran out first."""
        for state in self.init_states:
            if self._slot_key((), state, 0) not in self.outcomes:
                if self.episodes_left <= 0:
                    return False
                self._run((), state, 0)
        return True

    def propose(self) -> int:
        """Ask the proposer for hypotheses, after making sure it has seen the baseline on every state."""
        if not self._ensure_baseline():
            return 0
        self.rounds += 1
        proposal = self.proposer.propose(
            ProposalContext(
                task=self.task,
                baseline=self.baseline,
                limits=self.config.limits,
                init_states=self.init_states,
                graph=self.graph,
                records=self.store.load()[-self.config.records_in_context :],
                cognition=self.cognition,
            )
        )
        if proposal.cognition is not None:
            self.cognition = list(proposal.cognition)
        added = 0
        ids: dict[str, str] = {}
        for proposed in sorted(proposal.hypotheses, key=lambda p: p.kind == PAIR):
            try:
                added += self._add(proposed, ids)
            except (HarnessError, ValueError, KeyError) as error:
                logger.warning("Rejected proposed hypothesis '%s': %s", proposed.key, error)
        self.save()
        return added

    def _add(self, proposed: ProposedHypothesis, ids: dict[str, str]) -> int:
        refines = ids.get(proposed.refines, proposed.refines) if proposed.refines else None
        if proposed.kind == PAIR:
            members = tuple(ids.get(m, m) for m in proposed.members)
            hypothesis = Hypothesis("+".join(members), PAIR, proposed.claim, members=members)
            if any(self.graph.nodes[m].status != ACTIVE for m in members):
                raise ValueError("pair members must be active hypotheses")
        else:
            hypothesis = Hypothesis(
                f"{proposed.kind[0].upper()}{self.counters[proposed.kind] + 1}",
                proposed.kind,
                proposed.claim,
                payload=proposed.payload,
                refines=refines,
            )
            self.baseline.apply([hypothesis])  # payload must at least parse
        self.graph.add(hypothesis)
        if hypothesis.kind == PAIR:
            if not self._effective(hypothesis.members):
                del self.graph.nodes[hypothesis.id]
                raise HarnessError("pair is not runnable or does not change the baseline")
        else:
            self.counters[hypothesis.kind] += 1
            ids[proposed.key] = hypothesis.id
        return 1

    def run(self) -> str:
        """Evolve until the budget or the hypotheses run out. Returns the reason for stopping."""
        while True:
            if not self._ensure_baseline():
                return "budget_exhausted"
            selection, trial = self.next_experiment()
            if trial is not None:
                logger.info(
                    "Experiment %s on '%s' (VOI %.4f bits/s)", trial.node.id, trial.state, selection.voi
                )
                self.run_experiment(trial)
                continue
            if self.rounds >= self.config.max_rounds:
                return selection.stop_reason or "max_rounds"
            if self.propose() == 0:
                return "budget_exhausted" if self.episodes_left <= 0 else "no_new_hypotheses"

    def summary(self) -> dict[str, Any]:
        return {
            "episodes_used": self.episodes_used,
            "baseline_version": self.baseline_version,
            "baseline": self.baseline.to_dict(),
            "uncertainty_bits": self.graph.uncertainty(),
            "hypotheses": [
                {
                    "id": node.id,
                    "kind": node.kind,
                    "status": node.status,
                    "claim": node.claim,
                    "support": node.n_sup,
                    "refute": node.n_ref,
                    "confidence": round(self.graph.confidence(node.id), 3),
                }
                for node in self.graph.nodes.values()
            ],
            "works_with": [
                {"code": c, "skill": s, "gain": round(g, 3)} for (c, s), g in self.graph.works_with.items()
            ],
            "cognition": self.cognition,
        }
