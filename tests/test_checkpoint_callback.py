from collections import deque
import json
from types import SimpleNamespace
from typing import Any, cast

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from traffic_drl.train.callbacks import RobustCheckpointCallback, TrafficMetricsCallback
from traffic_drl.train.train_dqn import build_dqn_model, train_dqn


class ScriptedRewardEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self._episode = -1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode += 1
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        reward = (2.0, 0.0, 0.0)[min(self._episode, 2)]
        return np.zeros(1, dtype=np.float32), reward, True, False, {}


def test_best_checkpoint_uses_completed_episode_when_logger_is_empty():
    callback = RobustCheckpointCallback(save_freq=100, save_dir="unused")
    callback.model = cast(Any, SimpleNamespace(
        logger=SimpleNamespace(name_to_value={}),
        ep_info_buffer=deque(maxlen=100),
    ))
    callback.locals = {
        "infos": [{"episode": {"r": 5.0, "l": 10}, "average_waiting_time": 4.0}]
    }
    saved = []

    def capture_saved_bundle(name: str) -> None:
        saved.append(name)

    callback._save_bundle = capture_saved_bundle

    callback._check_best()

    assert saved == ["best"]
    assert callback._best_reward == 5.0
    assert callback._best_waiting_time == 4.0

    callback.locals = {
        "infos": [{"episode": {"r": 1.0, "l": 10}, "average_waiting_time": 2.0}]
    }
    callback._check_best()
    assert saved == ["best"]

    callback.locals = {
        "infos": [{"episode": {"r": 9.0, "l": 10}, "average_waiting_time": 2.0}]
    }
    callback._check_best()
    assert saved == ["best", "best"]
    assert callback._best_reward == 5.0
    assert callback._best_waiting_time == 2.0


def test_traffic_metrics_callback_averages_worker_metrics():
    records = {}
    callback = TrafficMetricsCallback()
    callback.model = cast(Any, SimpleNamespace(
        logger=SimpleNamespace(record=lambda key, value: records.update({key: value}))
    ))
    callback.locals = {
        "infos": [
            {
                "episode": {"r": 2.0},
                "scenario_id": "train-a",
                "average_waiting_time": 4.0,
                "average_queue_length": 2.0,
                "throughput": 10,
                "phase_switch_rate": 0.2,
                "min_green_violations": 2,
            },
            {
                "episode": {"r": 3.0},
                "scenario_id": "train-b",
                "average_waiting_time": 8.0,
                "average_queue_length": 6.0,
                "throughput": 20,
                "phase_switch_rate": 0.4,
                "min_green_violations": 4,
            },
        ]
    }

    assert callback._on_step()
    assert records["traffic/average_waiting_time"] == 6.0
    assert records["traffic/average_queue_length"] == 4.0
    assert records["traffic/throughput"] == 15.0
    assert np.isclose(records["traffic/phase_switch_rate"], 0.3)
    assert records["traffic/min_green_violations"] == 3.0
    assert records["scenario/id/env_0"] == "train-a"
    assert records["scenario/id/env_1"] == "train-b"


def test_dqn_training_writes_best_and_last_checkpoint_bundles(tmp_path):
    env = DummyVecEnv([lambda: Monitor(ScriptedRewardEnv())])
    config = {
        "policy": "MlpPolicy",
        "learning_rate": 1e-3,
        "buffer_size": 32,
        "learning_starts": 100,
        "batch_size": 4,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_update_interval": 10,
    }
    model = build_dqn_model(env, cast(Any, config), seed=7)
    callback = RobustCheckpointCallback(
        save_freq=100,
        save_dir=tmp_path,
        save_replay_buffer=True,
        save_vec_normalize=False,
    )

    try:
        train_dqn(model, total_timesteps=3, callback=callback)
    finally:
        env.close()

    for checkpoint_name in ("best", "last"):
        checkpoint_dir = tmp_path / checkpoint_name
        assert (checkpoint_dir / "model.zip").is_file()
        assert (checkpoint_dir / "replay_buffer.pkl").is_file()
        assert (checkpoint_dir / "resume_info.json").is_file()

    best_info = json.loads((tmp_path / "best" / "resume_info.json").read_text())
    last_info = json.loads((tmp_path / "last" / "resume_info.json").read_text())
    assert best_info["best_metric_name"] == "ep_rew_mean"
    assert best_info["best_metric_value"] == 2.0
    assert best_info["num_timesteps"] == 1
    assert last_info["num_timesteps"] == 3