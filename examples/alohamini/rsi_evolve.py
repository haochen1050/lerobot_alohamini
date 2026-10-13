#!/usr/bin/env python3

"""Run the EmbodiedRSI self-evolution loop (arXiv:2610.10498) around a frozen policy on AlohaMini.

The policy is never updated. Claude proposes changes to the harness around it (which instructions the policy
is called with, in which order and for how long), the loop picks the most informative one to test, and you
restore the named initial state before each episode and label it success or failure afterwards.

Everything is written to --run_dir after every episode; rerun the same command to resume.

Example:
    python examples/alohamini/rsi_evolve.py \
        --policy.path outputs/train/pi05_oreo/checkpoints/last/pretrained_model \
        --task "put the oreo in the red bowl" \
        --init_states "oreo left;oreo centre;oreo right" \
        --run_dir outputs/rsi/oreo --robot.remote_ip 192.168.1.50 --robot.robot_model alohamini2pro \
        --inference.type rtc
"""

import argparse
import inspect
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

import lerobot.robots.alohamini  # noqa: F401 — registers alohamini_client robot type
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini import AlohaMiniClient, AlohaMiniClientConfig
from lerobot.robots.alohamini.rsi import (
    ClaudeProposer,
    EpisodeOutcome,
    EvolutionLoop,
    Harness,
    HarnessLimits,
    LoopConfig,
    ResolvedStep,
    SelectionConfig,
)
from lerobot.rollout.inference.factory import (
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.device_utils import auto_select_torch_device
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

try:
    from .evaluation_safety import EvaluationSafetyGuard, hold_action, stop_inference
except ImportError:
    from evaluation_safety import EvaluationSafetyGuard, hold_action, stop_inference


class RobotEpisodeRunner:
    """One physical episode: restore the scene, run the harness steps through the policy, ask for a label."""

    def __init__(self, args, run_dir: Path):
        self.args = args
        self.frames_dir = run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        device = str(auto_select_torch_device())

        policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        policy_cfg.pretrained_path = args.policy_path
        policy_class = get_policy_class(policy_cfg.type)
        policy = policy_class.from_pretrained(args.policy_path, config=policy_cfg)
        if args.inference_type == "rtc":
            params = inspect.signature(policy_class.predict_action_chunk).parameters
            if not {"inference_delay", "prev_chunk_left_over"}.issubset(params) and not any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            ):
                raise ValueError(f"Policy type '{policy_cfg.type}' does not support RTC inference")
            rtc_config = RTCConfig(
                execution_horizon=args.rtc_execution_horizon,
                max_guidance_weight=args.rtc_max_guidance_weight,
            )
            policy.config.rtc_config = rtc_config
            if hasattr(policy, "init_rtc_processor"):
                policy.init_rtc_processor()
            inference_cfg = RTCInferenceConfig(rtc=rtc_config, queue_threshold=args.rtc_queue_threshold)
        else:
            inference_cfg = SyncInferenceConfig()
        policy = policy.to(device)
        policy.eval()

        self.robot = AlohaMiniClient(
            AlohaMiniClientConfig(remote_ip=args.remote_ip, id=args.robot_id, robot_model=args.robot_model)
        )
        self.robot.connect()

        teleop_action_processor, self.robot_action_processor, self.robot_observation_processor = (
            make_default_processors()
        )
        action_features_hw = self.robot.action_features
        observation_features_hw = self.robot.observation_features
        self.dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=action_features_hw),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=self.robot_observation_processor,
                initial_features=create_initial_features(observation=observation_features_hw),
                use_videos=True,
            ),
        )
        self.ordered_action_keys = list(action_features_hw.keys())
        self.dataset = LeRobotDataset.create(
            repo_id=f"local/rsi_{run_dir.name}",
            root=run_dir / f"dataset_{time.strftime('%Y%m%d_%H%M%S')}",
            fps=args.fps,
            features=self.dataset_features,
            robot_type=self.robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=args.policy_path,
            dataset_stats=self.dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.engine = create_inference_engine(
            inference_cfg,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=ThreadSafeRobot(self.robot),
            hw_features=hw_to_dataset_features(observation_features_hw, "observation"),
            dataset_features=self.dataset_features,
            ordered_action_keys=self.ordered_action_keys,
            task=args.task,
            fps=float(args.fps),
            device=device,
        )
        self.interpolator = ActionInterpolator(multiplier=args.interpolation_multiplier)
        self.control_interval = self.interpolator.get_control_interval(args.fps)
        self.safety_guard = EvaluationSafetyGuard()
        self.start_pose: dict[str, float] | None = None
        self.episode = 0

    def _fresh_observation(self, duration_s: float = 0.5) -> dict:
        """Drain observations queued while waiting on the keyboard so the episode starts on a live frame."""
        deadline = time.monotonic() + duration_s
        observation = self.robot.get_observation()
        while time.monotonic() < deadline:
            observation = self.robot.get_observation()
            precise_sleep(self.control_interval)
        return observation

    def _return_to_start(self) -> None:
        """Move the arms and lift back to where the first episode started, slowly, with the base stopped."""
        observation = self._fresh_observation()
        current = hold_action(observation)
        if self.start_pose is None:
            self.start_pose = current
            return
        n_ticks = max(1, int(self.args.return_time_s / self.control_interval))
        for tick in range(1, n_ticks + 1):
            blend = tick / n_ticks
            action = {k: current[k] + blend * (self.start_pose[k] - current[k]) for k in current}
            self.robot.send_action(self.robot_action_processor((action, observation)))
            precise_sleep(self.control_interval)

    def _run_step(self, step: ResolvedStep) -> dict:
        """The control loop of evaluate_bi.py for one policy call with this step's instruction."""
        robot, engine, interpolator = self.robot, self.engine, self.interpolator
        stop_inference(engine)
        # The engines read the instruction from this attribute on every inference call.
        engine._task = step.instruction
        engine.reset()
        engine.start()
        interpolator.reset()
        engine.resume()
        start = time.perf_counter()
        cached_obs_processed = None
        obs_raw = robot.get_observation()

        while (time.perf_counter() - start) < step.time_s:
            loop_start = time.perf_counter()
            obs_raw = robot.get_observation()
            obs_raw, reason = self.safety_guard.check_observation(robot, obs_raw)
            if reason:
                paused_at = time.perf_counter()
                self.safety_guard.recover(robot, engine, interpolator, None, obs_raw, reason)
                start += time.perf_counter() - paused_at
                cached_obs_processed = None
                continue
            if cached_obs_processed is None or interpolator.needs_new_action():
                obs_processed = self.robot_observation_processor(obs_raw)
                engine.notify_observation(obs_processed)
                cached_obs_processed = obs_processed
            else:
                obs_processed = cached_obs_processed
            obs_frame = build_dataset_frame(self.dataset_features, obs_processed, prefix=OBS_STR)

            if interpolator.needs_new_action():
                action_tensor = engine.get_action(obs_frame)
                if action_tensor is not None:
                    interpolator.add(action_tensor.cpu())

            interp_action = interpolator.get()
            if interp_action is not None:
                action_dict = {k: interp_action[i].item() for i, k in enumerate(self.ordered_action_keys)}
                if robot.send_action(self.robot_action_processor((action_dict, obs_raw))):
                    action_frame = build_dataset_frame(self.dataset_features, action_dict, prefix=ACTION)
                    self.dataset.add_frame({**obs_frame, **action_frame, "task": step.instruction})

            dt = time.perf_counter() - loop_start
            if (sleep_t := self.control_interval - dt) > 0:
                precise_sleep(sleep_t)
        engine.pause()
        return obs_raw

    def _save_frame(self, observation: dict) -> list[str]:
        image = observation.get(self.args.frame_camera)
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            return []
        path = self.frames_dir / f"episode_{time.strftime('%Y%m%d_%H%M%S')}_{self.episode:04d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return [str(path)]

    def __call__(self, steps: tuple[ResolvedStep, ...], state: str, label: str) -> EpisodeOutcome:
        self.episode += 1
        print(f"\n=== Episode {self.episode}: variant '{label}', initial state '{state}' ===")
        for index, step in enumerate(steps, 1):
            print(f"  step {index}: [{step.time_s:.0f}s] {step.instruction}")
        input(f"[RESET] Restore initial state '{state}', clear the workspace, then press Enter to start: ")
        if self.args.return_to_start:
            self._return_to_start()
        else:
            self._fresh_observation()

        started = time.perf_counter()
        observation = {}
        for step in steps:
            observation = self._run_step(step)
        duration_s = time.perf_counter() - started
        self.robot.send_action(hold_action(observation))
        self.dataset.save_episode()
        frames = self._save_frame(observation)

        answer = ""
        while answer not in ("y", "n"):
            answer = input("[LABEL] Did the task succeed? [y/n]: ").strip().lower()
        note = input("[NOTE] What happened, in a few words (optional, helps the next hypotheses): ").strip()
        return EpisodeOutcome(success=answer == "y", duration_s=duration_s, note=note, frames=frames)

    def close(self) -> None:
        self.engine.pause()
        try:
            self.robot.send_action(hold_action(self.robot.last_remote_state))
        finally:
            try:
                self.engine.stop()
            finally:
                try:
                    if self.dataset.has_pending_frames():
                        self.dataset.save_episode()
                finally:
                    try:
                        self.dataset.finalize()
                    finally:
                        self.robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--policy.path", dest="policy_path", type=str)
    parser.add_argument("--task", type=str, help="Instruction the policy was trained with.")
    parser.add_argument(
        "--init_states",
        type=str,
        help="Semicolon-separated names of initial scene layouts you can restore by hand, e.g. marked spots.",
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--summary", action="store_true", help="Print the state of --run_dir and exit.")
    parser.add_argument("--budget", type=int, default=50, help="Total physical episodes for this run.")
    parser.add_argument("--episode_time", type=float, default=60.0, help="Baseline time budget in seconds.")
    parser.add_argument("--max_episode_time", type=float, default=180.0)
    parser.add_argument("--reset_time", type=float, default=30.0, help="Rough manual reset time, for VOI.")
    parser.add_argument("--n_max", type=int, default=10, help="Evaluation cap per hypothesis.")
    parser.add_argument("--claude_model", type=str, default="claude-opus-5-5")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame_camera", type=str, default="forward")
    parser.add_argument(
        "--return_to_start",
        action="store_true",
        help="Before each episode, move the arms and lift back to the pose of the first episode start.",
    )
    parser.add_argument("--return_time_s", type=float, default=4.0)
    parser.add_argument("--robot.remote_ip", dest="remote_ip", type=str, default="127.0.0.1")
    parser.add_argument("--robot.id", dest="robot_id", type=str, default="my_alohamini")
    parser.add_argument(
        "--robot.robot_model",
        dest="robot_model",
        type=str,
        default="alohamini1",
        choices=["alohamini1", "alohamini2", "alohamini2pro"],
    )
    parser.add_argument(
        "--inference.type", dest="inference_type", type=str, default="sync", choices=["sync", "rtc"]
    )
    parser.add_argument(
        "--inference.rtc.execution_horizon", dest="rtc_execution_horizon", type=int, default=10
    )
    parser.add_argument(
        "--inference.rtc.max_guidance_weight", dest="rtc_max_guidance_weight", type=float, default=10.0
    )
    parser.add_argument("--inference.rtc.queue_threshold", dest="rtc_queue_threshold", type=int, default=30)
    parser.add_argument("--interpolation_multiplier", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.summary:
        print((args.run_dir / "state.json").read_text())
        return
    if not (args.policy_path and args.task and args.init_states):
        parser.error("--policy.path, --task and --init_states are required")

    init_states = [state.strip() for state in args.init_states.split(";") if state.strip()]
    config = LoopConfig(
        episode_budget=args.budget,
        reset_time_s=args.reset_time,
        selection=SelectionConfig(n_max=args.n_max),
        limits=HarnessLimits(max_total_time_s=args.max_episode_time),
    )
    proposer = ClaudeProposer(model=args.claude_model)
    proposer._get_client()  # fail on a missing package or credentials before the robot moves
    runner = RobotEpisodeRunner(args, args.run_dir)
    try:
        loop = EvolutionLoop(
            args.run_dir,
            args.task,
            Harness.single_task(args.task, args.episode_time),
            init_states,
            runner,
            proposer,
            config,
        )
        reason = loop.run()
        print(f"\nStopped: {reason}")
        print(json.dumps(loop.summary(), indent=2))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
