from pathlib import Path
import shutil
from typing import Any, cast

import numpy as np
import pytest
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from traffic_drl.environment.make_env import make_vectorized_environment
from traffic_drl.environment.scenario_factory import ScenarioManifest
from traffic_drl.contracts import ScenarioRecord
from traffic_drl.train.callbacks import RobustCheckpointCallback, TrafficMetricsCallback
from traffic_drl.train.train_dqn import build_dqn_model, load_dqn_checkpoint, train_dqn
from traffic_drl.train.scenario_sampler import ScenarioSampler


def test_subproc_vec_env_samples_only_allowed_split(tmp_path: Path):
    if shutil.which("sumo") is None:
        pytest.skip("SUMO executable is unavailable")

    route_file = Path("scenarios/dummy/single-intersection-vhvh.rou.xml")
    manifest = ScenarioManifest(
        [
            ScenarioRecord("train-a", "TR", route_file, 1, 1, 300),
            ScenarioRecord("train-b", "TR", route_file, 2, 2, 300),
            ScenarioRecord("validation-only", "VA", route_file, 3, 3, 300),
        ]
    )
    sampler = ScenarioSampler(manifest, allowed_split="TR", seed=11)
    env = make_vectorized_environment(
        config="configs/environment/dev_single_intersection.yaml",
        route_file=route_file,
        run_id="subproc_sampling_smoke",
        base_dir=tmp_path,
        seed=11,
        wrap=True,
        scenario_sampler=sampler,
        custom_observation=True,
        n_envs=2,
        start_method="spawn",
        norm_obs=True,
        norm_reward=True,
    )

    try:
        normalized_env = cast(VecNormalize, env)
        worker_env = cast(SubprocVecEnv, normalized_env.venv)
        observations = cast(np.ndarray, normalized_env.reset())

        assert len(worker_env.reset_infos) == 2
        assert observations.shape[0] == 2
        sampled_ids = {info["scenario_id"] for info in worker_env.reset_infos}
        assert sampled_ids
        assert sampled_ids <= {"train-a", "train-b"}
        assert "validation-only" not in sampled_ids

        test_actions = [0] * normalized_env.num_envs
        _, test_rewards, test_dones, test_infos = normalized_env.step(test_actions)
        assert test_rewards.shape == (2,)
        assert test_dones.shape == (2,)
        assert len(test_infos) == 2

        model = build_dqn_model(
            cast(Any, normalized_env),
            cast(
                Any,
                {
                    "policy": "MlpPolicy",
                    "buffer_size": 64,
                    "learning_starts": 0,
                    "batch_size": 2,
                    "train_freq": 1,
                    "gradient_steps": 1,
                    "target_update_interval": 10,
                },
            ),
            seed=11,
        )
        checkpoint_dir = tmp_path / "checkpoints"
        callbacks = CallbackList(
            [
                RobustCheckpointCallback(
                    save_freq=100,
                    save_dir=checkpoint_dir,
                    save_replay_buffer=True,
                    save_vec_normalize=True,
                ),
                TrafficMetricsCallback(),
            ]
        )
        train_dqn(model, total_timesteps=8, callback=callbacks)

        assert model.num_timesteps == 8
        assert model.replay_buffer is not None
        assert model.replay_buffer.size() >= 3
        assert normalized_env.norm_obs
        assert normalized_env.norm_reward
        assert (checkpoint_dir / "last" / "model.zip").is_file()
        assert (checkpoint_dir / "last" / "replay_buffer.pkl").is_file()
        assert (checkpoint_dir / "last" / "vec_normalize.pkl").is_file()

        env.close()
        resume_env = make_vectorized_environment(
            config="configs/environment/dev_single_intersection.yaml",
            route_file=route_file,
            run_id="subproc_sampling_resume",
            base_dir=tmp_path,
            seed=11,
            wrap=True,
            scenario_sampler=sampler,
            custom_observation=True,
            n_envs=2,
            start_method="spawn",
            norm_obs=False,
            norm_reward=False,
        )
        try:
            resumed_model = load_dqn_checkpoint(
                checkpoint_dir / "last",
                env=cast(Any, resume_env),
                training=True,
            )
            resumed_env = cast(VecNormalize, resumed_model.get_env())

            assert resumed_env.norm_obs
            assert resumed_env.norm_reward
            assert resumed_model.replay_buffer is not None
            assert resumed_model.replay_buffer.size() >= 3

            train_dqn(
                resumed_model,
                total_timesteps=4,
                reset_num_timesteps=False,
            )
            assert resumed_model.num_timesteps == 12
        finally:
            resume_env.close()
    finally:
        env.close()


def test_subproc_vec_env_trains_and_saves_valid_dqn(tmp_path: Path):
    if shutil.which("sumo") is None:
        pytest.skip("SUMO executable is unavailable")

    env = make_vectorized_environment(
        config="configs/environment/dev_single_intersection.yaml",
        route_file="scenarios/dummy/single-intersection-vhvh.rou.xml",
        run_id="subproc_smoke",
        base_dir=tmp_path,
        seed=11,
        custom_observation=False,
        n_envs=2,
        start_method="spawn",
        norm_obs=False,
        norm_reward=False,
    )

    try:
        model = build_dqn_model(
            cast(Any, env),
            cast(
                Any,
                {
                    "policy": "MlpPolicy",
                    "learning_rate": 1e-3,
                    "buffer_size": 64,
                    "learning_starts": 0,
                    "batch_size": 2,
                    "train_freq": 1,
                    "gradient_steps": 1,
                    "target_update_interval": 10,
                },
            ),
            seed=11,
        )
        initial_parameters = [
            parameter.detach().cpu().numpy().copy()
            for parameter in model.policy.parameters()
        ]
        train_dqn(model, total_timesteps=8)

        assert model.num_timesteps == 8
        assert model.replay_buffer is not None
        assert model.replay_buffer.size() >= 3
        assert model._n_updates > 0
        assert any(
            not np.array_equal(initial, parameter.detach().cpu().numpy())
            for initial, parameter in zip(
                initial_parameters, model.policy.parameters(), strict=True
            )
        )

        model_path = tmp_path / "trained_dqn.zip"
        model.save(str(model_path))
        loaded_model = DQN.load(str(model_path), env=env)
        observations = cast(np.ndarray, env.reset())
        actions, _ = loaded_model.predict(observations, deterministic=True)

        assert model_path.is_file()
        assert observations.shape[0] == 2
        assert actions.shape == (2,)
        action_space = cast(gym.spaces.Discrete, env.action_space)
        assert np.all((0 <= actions) & (actions < action_space.n))
        assert (tmp_path / "subproc_smoke_worker_0").is_dir()
        assert (tmp_path / "subproc_smoke_worker_1").is_dir()
    finally:
        env.close()
