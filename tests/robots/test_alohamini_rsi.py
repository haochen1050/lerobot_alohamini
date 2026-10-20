import json
import random

import pytest

from lerobot.robots.alohamini.rsi import (
    CODE,
    SKILL,
    CostWeights,
    Experiment,
    Hypothesis,
    HypothesisGraph,
    SelectionConfig,
    classify_matched,
    expected_information_gain,
    experiment_cost,
    prob_support_rate_above_half,
    select_experiment,
)


def make_graph() -> HypothesisGraph:
    graph = HypothesisGraph()
    graph.add(Hypothesis("C0", CODE, "re-detect the object before grasping"))
    graph.add(Hypothesis("S0", SKILL, "say which bowl by colour"))
    return graph


def test_confidence_is_beta_posterior_mean():
    graph = make_graph()
    assert graph.confidence("C0") == pytest.approx(0.5)
    for supported in (True, True, True, False, None):
        graph.record("C0", supported)
    assert graph.nodes["C0"].n_trials == 4
    assert graph.confidence("C0") == pytest.approx(4 / 6)


def test_prob_majority_closed_forms():
    assert prob_support_rate_above_half(1, 1) == pytest.approx(0.5)
    assert prob_support_rate_above_half(2, 1) == pytest.approx(0.75)
    assert prob_support_rate_above_half(1, 3) == pytest.approx(0.125)
    assert prob_support_rate_above_half(7.5, 7.5) == pytest.approx(0.5)


def test_prob_majority_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(0)
    for _ in range(200):
        a, b = rng.uniform(0.3, 40), rng.uniform(0.3, 40)
        assert prob_support_rate_above_half(a, b) == pytest.approx(stats.beta.sf(0.5, a, b), abs=1e-9)


def test_uncertainty_starts_at_one_bit_per_hypothesis_and_falls_with_evidence():
    graph = make_graph()
    assert graph.uncertainty() == pytest.approx(2.0)
    for _ in range(6):
        graph.record("C0", True)
    assert graph.uncertainty() < 1.1


def test_classify_matched():
    assert classify_matched(baseline_success=False, variant_success=True) is True
    assert classify_matched(baseline_success=False, variant_success=False) is False
    assert classify_matched(baseline_success=True, variant_success=False) is False
    assert classify_matched(baseline_success=True, variant_success=True) is None


def test_refines_edges_and_lineage():
    graph = make_graph()
    graph.add(Hypothesis("C1", CODE, "re-detect twice", refines="C0"))
    graph.add(Hypothesis("C2", CODE, "re-detect and re-plan", refines="C1"))
    assert graph.lineage("C2") == ["C2", "C1", "C0"]
    with pytest.raises(ValueError):
        graph.add(Hypothesis("S1", SKILL, "wrong parent kind", refines="C0"))
    with pytest.raises(ValueError):
        graph.add(Hypothesis("C3", CODE, "missing parent", refines="nope"))
    with pytest.raises(ValueError):
        graph.add(Hypothesis("C0", CODE, "duplicate"))


def test_joint_gain_keeps_only_positive_synergy():
    graph = make_graph()
    assert graph.record_joint("C0", "S0", r_joint=0.8, r_code=0.5, r_skill=0.6) == pytest.approx(0.2)
    assert graph.partners("C0") == [("S0", pytest.approx(0.2))]
    assert graph.partners("S0") == [("C0", pytest.approx(0.2))]
    assert graph.record_joint("C0", "S0", r_joint=0.6, r_code=0.5, r_skill=0.6) == pytest.approx(0.0)
    assert graph.partners("C0") == []
    with pytest.raises(ValueError):
        graph.record_joint("S0", "C0", 1.0, 0.0, 0.0)


def test_save_load_roundtrip(tmp_path):
    graph = make_graph()
    graph.add(Hypothesis("C1", CODE, "refined", payload="print(1)", refines="C0"))
    graph.record("C1", True)
    graph.record_joint("C1", "S0", 0.9, 0.5, 0.4)
    graph.save(tmp_path / "graph.json")
    assert HypothesisGraph.load(tmp_path / "graph.json") == graph


def test_expected_gain_is_non_negative_and_matches_enumeration():
    rng = random.Random(1)
    for _ in range(100):
        graph = make_graph()
        for hypothesis_id in graph.nodes:
            graph.nodes[hypothesis_id].n_sup = rng.randint(0, 8)
            graph.nodes[hypothesis_id].n_ref = rng.randint(0, 8)
        experiment = Experiment("pair", ("C0", "S0"), time_s=60)
        before = graph.uncertainty()
        expected_after = 0.0
        for c_sup in (True, False):
            for s_sup in (True, False):
                probability = (graph.confidence("C0") if c_sup else 1 - graph.confidence("C0")) * (
                    graph.confidence("S0") if s_sup else 1 - graph.confidence("S0")
                )
                after = HypothesisGraph.from_dict(graph.to_dict())
                after.record("C0", c_sup)
                after.record("S0", s_sup)
                expected_after += probability * after.uncertainty()
        gain = expected_information_gain(graph, experiment)
        assert gain == pytest.approx(before - expected_after, abs=1e-9)
        assert gain >= -1e-12


def test_selection_prefers_information_per_cost():
    graph = make_graph()
    for _ in range(8):
        graph.record("C0", True)
    code_only = Experiment("code", ("C0",), time_s=60)
    skill_only = Experiment("skill", ("S0",), time_s=60)
    pair = Experiment("pair", ("C0", "S0"), time_s=60)
    slow_pair = Experiment("slow_pair", ("C0", "S0"), time_s=60, reset_cost=600)

    assert select_experiment(graph, [code_only, skill_only]).experiment is skill_only
    assert select_experiment(graph, [code_only, skill_only, slow_pair]).experiment is skill_only
    assert select_experiment(graph, [code_only, skill_only, pair, slow_pair]).experiment is pair
    assert experiment_cost(slow_pair, CostWeights()) == 660


def test_selection_stop_reasons():
    graph = make_graph()
    experiments = [Experiment("pair", ("C0", "S0"), time_s=60)]
    assert select_experiment(graph, experiments, episodes_left=0).stop_reason == "budget_exhausted"
    assert select_experiment(graph, []).stop_reason == "no_informative_experiment"
    assert (
        select_experiment(graph, experiments, SelectionConfig(eps_voi=1.0)).stop_reason
        == "no_informative_experiment"
    )
    assert (
        select_experiment(graph, experiments, SelectionConfig(eps_h=2.5)).stop_reason
        == "uncertainty_resolved"
    )
    for hypothesis_id in graph.nodes:
        for i in range(10):
            graph.record(hypothesis_id, i % 2 == 0)
    assert select_experiment(graph, experiments).stop_reason == "evaluation_cap"
    with pytest.raises(ValueError):
        select_experiment(make_graph(), [Experiment("free", ("C0",), time_s=0)])


def test_capped_hypothesis_contributes_no_gain():
    graph = make_graph()
    for i in range(10):
        graph.record("C0", i % 2 == 0)
    experiment = Experiment("pair", ("C0", "S0"), time_s=60)
    assert expected_information_gain(graph, experiment, n_max=10) == pytest.approx(
        expected_information_gain(graph, Experiment("skill", ("S0",), time_s=60))
    )


# ---------------------------------------------------------------------------- harness and evolution loop

from lerobot.robots.alohamini.rsi import (  # noqa: E402
    ACTIVE,
    PAIR,
    PROMOTED,
    SUPERSEDED,
    EpisodeOutcome,
    EvolutionLoop,
    Harness,
    HarnessError,
    HarnessLimits,
    LoopConfig,
    Proposal,
    ProposedHypothesis,
    Skill,
    Step,
    code_payload,
    skill_payload,
)
from lerobot.robots.alohamini.rsi.claude_proposer import (  # noqa: E402
    OUTPUT_SCHEMA,
    ClaudeProposer,
    describe_context,
    to_proposal,
)

BASELINE = Harness.single_task("put the oreo in the bowl", 30)
STATES = ["left", "centre", "right"]
RED = skill_payload({"task": Skill("put the oreo in the red bowl", 30)})
BLUE = skill_payload({"task": Skill("put the oreo in the blue bowl", 30)})
RETRY = code_payload([Step("task"), Step("task", 15)])


def test_harness_resolve_apply_and_limits():
    steps = BASELINE.resolve()
    assert [(s.skill, s.instruction, s.time_s) for s in steps] == [("task", "put the oreo in the bowl", 30.0)]
    retry = BASELINE.apply([Hypothesis("C1", CODE, "retry", payload=RETRY)])
    assert [s.time_s for s in retry.resolve()] == [30.0, 15.0]
    red = retry.apply([Hypothesis("S1", SKILL, "red", payload=RED)])
    assert {s.instruction for s in red.resolve()} == {"put the oreo in the red bowl"}
    assert Harness.from_dict(red.to_dict()) == red
    assert BASELINE.resolve() == steps

    def bad(payload, kind=CODE):
        return BASELINE.apply([Hypothesis("X", kind, "bad", payload=payload)])

    with pytest.raises(HarnessError):
        bad(code_payload([Step("missing")])).resolve()
    with pytest.raises(HarnessError):
        bad(code_payload([Step("task", 0.5)])).resolve()
    with pytest.raises(HarnessError):
        bad(code_payload([Step("task")] * 13)).resolve()
    with pytest.raises(HarnessError):
        bad(code_payload([Step("task", 100), Step("task", 100)])).resolve()
    with pytest.raises(HarnessError):
        bad(skill_payload({"task": Skill("x" * 201, 10)}), SKILL).resolve()
    with pytest.raises(HarnessError):
        bad("import os; os.system('rm -rf /')")
    with pytest.raises(HarnessError):
        bad('{"plan": "task"}')
    assert BASELINE.resolve(HarnessLimits(max_total_time_s=30)) == steps


class World:
    """The frozen policy only succeeds when told the bowl colour; a retry does not help."""

    def __init__(self):
        self.episodes = []

    def __call__(self, steps, state, label):
        self.episodes.append((label, state, tuple(s.instruction for s in steps)))
        return EpisodeOutcome(success=any("red bowl" in s.instruction for s in steps), duration_s=1.0)


class ScriptedProposer:
    def __init__(self, *rounds):
        self.rounds = list(rounds)
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        return self.rounds.pop(0) if self.rounds else Proposal()


FIRST_ROUND = Proposal(
    hypotheses=[
        ProposedHypothesis("red", SKILL, "the policy needs the bowl colour", payload=RED),
        ProposedHypothesis("blue", SKILL, "the policy needs a colour, any colour", payload=BLUE),
        ProposedHypothesis("retry", CODE, "a second attempt recovers a missed grasp", payload=RETRY),
        ProposedHypothesis("bogus", CODE, "uses a skill that does not exist", payload="not json"),
    ],
    cognition=["Baseline fails from every initial state."],
)


def test_loop_promotes_the_true_hypothesis_and_refutes_the_others(tmp_path):
    world = World()
    proposer = ScriptedProposer(FIRST_ROUND)
    loop = EvolutionLoop(tmp_path, "oreo", BASELINE, STATES, world, proposer, LoopConfig(episode_budget=40))
    reason = loop.run()

    nodes = loop.graph.nodes
    assert set(nodes) == {"S1", "S2", "C1"}
    assert nodes["S1"].status == PROMOTED
    assert (nodes["C1"].n_sup, nodes["C1"].status) == (0, ACTIVE)
    assert nodes["S2"].status == SUPERSEDED
    assert loop.baseline.skills["task"].instruction == "put the oreo in the red bowl"
    assert loop.baseline_version == 1
    assert loop.cognition == ["Baseline fails from every initial state."]
    assert reason == "no_new_hypotheses"
    # Baseline once per state and harness version, shared by every hypothesis: 3 + 7 variant runs + 3.
    records = loop.store.load()
    slots = [(r.baseline_version, tuple(r.variant), r.init_state, r.repeat) for r in records]
    assert len(slots) == len(set(slots)) == loop.episodes_used == len(world.episodes) == 13
    assert [e[0] for e in world.episodes[:3]] == ["baseline"] * 3
    assert all(r.success for r in records if r.baseline_version == 1)
    # The second proposal round saw the promoted harness and the episode records.
    assert proposer.contexts[-1].baseline == loop.baseline
    assert proposer.contexts[-1].records


def test_loop_respects_budget_and_resumes(tmp_path):
    world = World()
    config = LoopConfig(episode_budget=5)
    loop = EvolutionLoop(tmp_path, "oreo", BASELINE, STATES, world, ScriptedProposer(FIRST_ROUND), config)
    assert loop.run() == "budget_exhausted"
    assert loop.episodes_used == 5

    resumed = EvolutionLoop(
        tmp_path, "oreo", BASELINE, STATES, world, ScriptedProposer(), LoopConfig(episode_budget=40)
    )
    assert resumed.episodes_used == 5
    assert resumed.graph == loop.graph
    resumed.run()
    assert resumed.graph.nodes["S1"].status == PROMOTED
    assert len(world.episodes) == 13
    with pytest.raises(ValueError):
        EvolutionLoop(tmp_path, "oreo", BASELINE, ["other"], world, ScriptedProposer())


def test_pair_hypothesis_records_joint_gain(tmp_path):
    class NeedsBoth(World):
        def __call__(self, steps, state, label):
            self.episodes.append((label, state))
            ok = len(steps) == 2 and steps[0].skill == "approach" and "red bowl" in steps[1].instruction
            return EpisodeOutcome(success=ok, duration_s=1.0)

    proposal = Proposal(
        hypotheses=[
            ProposedHypothesis("pair", PAIR, "approach first, then place", members=("plan", "skills")),
            ProposedHypothesis(
                "plan",
                CODE,
                "split into approach and place",
                payload=code_payload([Step("approach"), Step("task")]),
            ),
            ProposedHypothesis(
                "skills",
                SKILL,
                "add an approach skill and name the bowl",
                payload=skill_payload(
                    {"approach": Skill("move to the oreo", 10), "task": Skill("put it in the red bowl", 20)}
                ),
            ),
        ]
    )
    world = NeedsBoth()
    loop = EvolutionLoop(tmp_path, "oreo", BASELINE, STATES, world, ScriptedProposer(proposal))
    loop.run()

    assert {n.status for n in loop.graph.nodes.values()} == {PROMOTED}
    assert loop.graph.nodes["S1"].n_ref == 3
    assert loop.graph.works_with[("C1", "S1")] == pytest.approx(1.0)
    assert [s.skill for s in loop.baseline.resolve()] == ["approach", "task"]
    # The plan alone is not runnable (no 'approach' skill yet), so it never costs an episode.
    assert not any(label == "C1" for label, _ in world.episodes)


def test_claude_proposer_request_and_parsing(tmp_path):
    class FakeMessages:
        def create(self, **kwargs):
            self.kwargs = kwargs
            text = json.dumps(
                {
                    "analysis": "wrong bowl",
                    "hypotheses": [
                        {
                            "key": "a",
                            "kind": "skill",
                            "claim": "name the colour",
                            "refines": "",
                            "plan": [],
                            "skills": [{"name": "task", "instruction": "red bowl", "time_s": 30}],
                            "members": [],
                        },
                        {
                            "key": "b",
                            "kind": "code",
                            "claim": "retry",
                            "refines": "",
                            "plan": [{"skill": "task", "time_s": 0}, {"skill": "task", "time_s": 15}],
                            "skills": [],
                            "members": [],
                        },
                        {
                            "key": "c",
                            "kind": "pair",
                            "claim": "both",
                            "refines": "",
                            "plan": [],
                            "skills": [],
                            "members": ["b", "a"],
                        },
                    ],
                    "cognition": ["rule"],
                }
            )
            block = type("Block", (), {"type": "text", "text": text})()
            return type("Response", (), {"stop_reason": "end_turn", "content": [block]})()

    class FakeClient:
        def __init__(self):
            self.beta = type("Beta", (), {"messages": FakeMessages()})()

    import json

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")

    def run_episode(steps, state, label):
        return EpisodeOutcome(success=False, duration_s=1.0, note="missed the bowl", frames=[str(frame)])

    client = FakeClient()
    loop = EvolutionLoop(
        tmp_path / "run", "oreo", BASELINE, STATES, run_episode, ClaudeProposer(client=client)
    )
    assert loop.propose() == 3
    assert set(loop.graph.nodes) == {"S1", "C1", "C1+S1"}
    assert loop.cognition == ["rule"]
    assert [s.time_s for s in loop.baseline.apply([loop.graph.nodes["C1"]]).resolve()] == [30.0, 15.0]

    kwargs = client.beta.messages.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"]["format"]["schema"] is OUTPUT_SCHEMA
    content = kwargs["messages"][0]["content"]
    assert [block["type"] for block in content] == ["text", "image"] * 3 + ["text"]
    assert "missed the bowl" in content[-1]["text"]
    assert to_proposal({"analysis": "", "hypotheses": [], "cognition": []}).hypotheses == []
    assert "current_harness" in describe_context(loop_context(loop))


def loop_context(loop):
    from lerobot.robots.alohamini.rsi import ProposalContext

    return ProposalContext(
        loop.task, loop.baseline, loop.config.limits, loop.init_states, loop.graph, [], loop.cognition
    )


def test_loop_tests_an_align_base_plan_and_shows_base_records_to_the_proposer(tmp_path):
    from lerobot.robots.alohamini.rsi import ALIGN, ALIGN_BASE

    class DriftingBase(World):
        """The base drifts off its pose while the policy runs: only re-aligning before the task works.

        Every episode starts with an alignment (as the robot runner does) and reports it in `base`."""

        def __call__(self, steps, state, label):
            self.episodes.append((label, state, tuple(s.kind for s in steps)))
            kinds = [s.kind for s in steps]
            base = [{"phase": "start", "state": "BASE_READY", "final": {"distance_cm": 0.8}}]
            base += [{"phase": "step align_base", "state": "BASE_READY"} for k in kinds if k == ALIGN]
            realigned = ALIGN in kinds and kinds.index(ALIGN) < len(kinds) - 1
            return EpisodeOutcome(success=realigned, duration_s=1.0, base=base)

    plan = code_payload([Step("task", 10), Step(ALIGN_BASE, 0), Step("task", 15)])
    proposer = ScriptedProposer(
        Proposal(
            hypotheses=[ProposedHypothesis("realign", CODE, "re-aligning the base fixes it", payload=plan)]
        )
    )
    limits = HarnessLimits(max_align_steps=1, align_time_s=60.0)
    world = DriftingBase()
    loop = EvolutionLoop(
        tmp_path, "oreo", BASELINE, STATES, world, proposer, LoopConfig(episode_budget=12, limits=limits)
    )
    loop.run()

    assert loop.graph.nodes["C1"].status == PROMOTED
    tested = [e for e in world.episodes if e[0] == "C1"]
    assert tested and tested[0][2] == ("policy", ALIGN, "policy")
    records = loop.store.load()
    assert all(r.base and r.base[0]["phase"] == "start" for r in records)
    aligned = next(r for r in records if r.variant == ["C1"])
    assert aligned.steps[1]["kind"] == ALIGN and aligned.steps[1]["time_s"] == 60.0

    described = json.loads(describe_context(proposer.contexts[-1]))
    assert described["harness_limits"]["max_align_steps"] == 1
    assert any("base_alignment" in episode for episode in described["episodes"])
