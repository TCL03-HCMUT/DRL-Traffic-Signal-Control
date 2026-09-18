"""Stable-Baselines3 DQN training entry points for the Phase 1 pilot.

The public API mirrors the PPO training module
(:mod:`traffic_drl.train.train_ppo`) with the addition of replay-buffer
checkpointing.

Typical usage
-------------
::

    from traffic_drl.config import load_train_config
    from traffic_drl.train.train_dqn import build_dqn_model, train_dqn
    from traffic_drl.train.callbacks import RobustCheckpointCallback

    cfg = load_train_config("configs/train/dqn_phase1.yaml")
    model = build_dqn_model(env, cfg, seed=cfg.experiment.seed)
    callback = RobustCheckpointCallback(save_freq=cfg.training_control.save_freq,
                                        save_dir="outputs/checkpoints")
    model = train_dqn(model, total_timesteps=cfg.training_control.total_timesteps,
                      callback=callback)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .checkpointing import load_checkpoint, save_checkpoint
from ..config import TrainConfig, load_train_config

if TYPE_CHECKING:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def build_dqn_model(
    env: "DummyVecEnv | VecNormalize",
    config: TrainConfig,
    *,
    seed: int | None = None,
    tensorboard_log: str | Path | None = None,
) -> "DQN":
    """Construct an SB3 DQN with a vectorised Gymnasium-compatible environment."""
    from stable_baselines3 import DQN

    kwargs = dict(config.model_hyperparameters)
    policy = kwargs.pop("policy", "MlpPolicy")

    return DQN(
        policy=policy,
        env=env,
        seed=seed if seed is not None else config.experiment.seed,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        **kwargs,
    )


def train_dqn(
    model: "DQN",
    *,
    total_timesteps: int,
    callback: "BaseCallback | None" = None,
    reset_num_timesteps: bool = True,
) -> "DQN":
    """Train a DQN model and return it."""
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps,
    )
    return model


def save_dqn_checkpoint(
    model: "DQN",
    path: str | Path,
    *,
    replay_buffer_path: str | Path | None = None,
    vec_normalize_env: "VecNormalize | None" = None,
    resume_info: dict[str, str | int | float | bool] | None = None,
) -> None:
    """Save model, replay buffer, normalisation stats, and resume metadata."""
    save_checkpoint(
        model,
        path,
        vec_normalize_env=vec_normalize_env,
        save_replay_buffer=replay_buffer_path is not None,
        replay_buffer_path=replay_buffer_path,
        resume_info=resume_info,
    )


def load_dqn_checkpoint(
    path: str | Path,
    *,
    env: "DummyVecEnv | VecNormalize | None" = None,
    replay_buffer_path: str | Path | None = None,
    vec_normalize_stats: str | Path | None = None,
) -> "DQN":
    """Load a DQN checkpoint and optional training state."""
    from stable_baselines3 import DQN as _DQN

    return load_checkpoint(
        _DQN,
        path,
        env=env,
        vec_normalize_path=vec_normalize_stats,
        replay_buffer_path=replay_buffer_path,
    )


def run_dqn_pilot(
    config: TrainConfig | str | Path = "configs/train/dqn_phase1.yaml",
    *,
    checkpoint_dir: str | Path,
    total_timesteps: int = 10_000,
    seed: int = 5,
) -> "DQN":
    """Run the DEV-00 pilot training without treating it as a final result."""
    if isinstance(config, (str, Path)):
        config = load_train_config(config)

    from traffic_drl.environment.make_env import make_dev_environment, make_vectorized_environment, build_reward_fn
    from traffic_drl.environment.scenario_factory import ScenarioManifest
    from traffic_drl.train.callbacks import Phase1PilotCallback
    from traffic_drl.evaluation.test import check_environment, run_smoke_test

    # 1. Smoke test the DEV environment to verify contract
    dev_env = make_dev_environment(config.environment.env_config_path, seed=seed)
    check_environment(dev_env)
    run_smoke_test(dev_env, steps=10, seed=seed)

    # 2. Set up the vectorized environment
    manifest = ScenarioManifest.from_csv(config.environment.manifest_path)
    record = manifest.get("DEV-00")
    
    vec_env = make_vectorized_environment(
        config=config.environment.env_config_path,
        route_file=record.route_file,
        run_id="pilot_dqn",
        base_dir=Path("outputs/runs"),
        seed=seed,
        reward_fn=build_reward_fn(config.reward),
        norm_obs=config.normalisation.norm_obs,
        norm_reward=config.normalisation.norm_reward,
        clip_obs=config.normalisation.clip_obs,
        clip_reward=config.normalisation.clip_reward,
    )

    # 3. Build model
    model = build_dqn_model(vec_env, config, seed=seed, tensorboard_log=checkpoint_dir)

    # 4. Train
    callback = Phase1PilotCallback(
        save_freq=config.training_control.save_freq,
        save_dir=checkpoint_dir,
        save_replay_buffer=True,
        save_vec_normalize=True,
    )
    
    model = train_dqn(
        model, 
        total_timesteps=total_timesteps, 
        callback=callback
    )

    if hasattr(vec_env, "close") and callable(vec_env.close):
        vec_env.close()

    return model
