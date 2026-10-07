"""Stable-Baselines3 PPO training API — reserved for the Phase 2 pipeline.

PPO training is secondary to the DQN deliverable.  This module provides the
same surface as :mod:`traffic_drl.train.train_dqn` so the evaluation pipeline
can call either without special-casing.

TODO (SV2): keep PPO secondary to the minimum DQN deliverable; implement once
Phase 1 DQN evaluation is complete.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .checkpointing import load_checkpoint, save_checkpoint
from ..config import TrainConfig, load_train_config

if TYPE_CHECKING:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def build_ppo_model(
    env: "DummyVecEnv | VecNormalize",
    config: TrainConfig,
    *,
    seed: int | None = None,
    tensorboard_log: str | Path | None = None,
) -> "PPO":
    """Construct an SB3 PPO model.

    Args:
        env: Vectorised (and optionally normalised) Gymnasium environment from
            :func:`~traffic_drl.environment.make_env.make_vectorized_environment`.
        config: Typed training configuration from
            :func:`~traffic_drl.config.load_train_config`.
        seed: Model random seed.  Overrides ``config.experiment.seed`` if
            provided.
        tensorboard_log: Optional TensorBoard log directory.

    Returns:
        PPO: An initialised SB3 PPO model.
    """
    from stable_baselines3 import PPO

    kwargs = dict(config.model_hyperparameters)
    policy = kwargs.pop("policy", "MlpPolicy")

    return PPO(
        policy=policy,
        env=env,
        seed=seed if seed is not None else config.experiment.seed,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        **kwargs,
    )


def train_ppo(
    model: "PPO",
    *,
    total_timesteps: int,
    callback: "BaseCallback | None" = None,
    reset_num_timesteps: bool = True,
) -> "PPO":
    """Train a PPO model and return it.

    Args:
        model: Constructed SB3 PPO instance from :func:`build_ppo_model`.
        total_timesteps: Number of environment transitions to collect.
        callback: Optional SB3 callback.
        reset_num_timesteps: ``True`` for a fresh run; ``False`` when resuming.

    Returns:
        PPO: The trained (or partially trained) PPO model.
    """
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps,
    )
    return model


def save_ppo_checkpoint(
    model: "PPO",
    bundle_dir: str | Path,
    *,
    vec_normalize_env: "VecNormalize | None" = None,
    resume_info: "ResumeInfo | None" = None,
) -> None:
    """Save PPO model and all artifacts required for resuming."""
    save_checkpoint(
        model,
        bundle_dir,
        vec_normalize_env=vec_normalize_env,
        save_replay_buffer=False,  # PPO has no replay buffer
        resume_info=resume_info,
    )


def load_ppo_checkpoint(
    bundle_dir: str | Path,
    *,
    env: "DummyVecEnv | VecNormalize | None" = None,
    training: bool = True,
    restore_rng_state: bool = True,
) -> "PPO":
    """Load a PPO checkpoint with auto-detected VecNormalize."""
    from stable_baselines3 import PPO as _PPO

    return load_checkpoint(
        _PPO,
        bundle_dir,
        env=env,
        training=training,
        restore_rng_state=restore_rng_state,
    )


def run_ppo_pilot(
    config: TrainConfig | str | Path = "configs/train/ppo_dummy.yaml",
    *,
    checkpoint_dir: str | Path,
    total_timesteps: int = 10_000,
    seed: int = 5,
) -> "PPO":
    """Run the DEV-00 pilot training with PPO."""
    if isinstance(config, (str, Path)):
        config = load_train_config(config)

    from traffic_drl.environment.make_env import make_dev_environment, make_vectorized_environment, build_reward_fn
    from traffic_drl.environment.scenario_factory import ScenarioManifest
    from traffic_drl.train.callbacks import RobustCheckpointCallback
    from traffic_drl.evaluation.test import check_environment, run_smoke_test

    # 1. Smoke test the DEV environment
    dev_env = make_dev_environment(config.environment.env_config_path, seed=seed)
    check_environment(dev_env)
    run_smoke_test(dev_env, steps=10, seed=seed)

    # 2. Set up the vectorized environment
    manifest = ScenarioManifest.from_csv(config.environment.manifest_path)
    record = manifest.get("DEV-00")
    
    vec_env = make_vectorized_environment(
        config=config.environment.env_config_path,
        route_file=record.route_file,
        run_id="pilot_ppo",
        base_dir=Path("outputs/runs"),
        seed=seed,
        custom_observation=False,
        reward_fn=build_reward_fn(config.reward),
        norm_obs=config.normalisation.norm_obs,
        norm_reward=config.normalisation.norm_reward,
        clip_obs=config.normalisation.clip_obs,
        clip_reward=config.normalisation.clip_reward,
    )

    # 3. Build model
    model = build_ppo_model(vec_env, config, seed=seed, tensorboard_log=checkpoint_dir)

    # 4. Train
    callback = RobustCheckpointCallback(
        save_freq=config.training_control.save_freq,
        save_dir=checkpoint_dir,
        save_replay_buffer=False, # PPO has no replay buffer
        save_vec_normalize=True,
    )
    
    model = train_ppo(
        model, 
        total_timesteps=total_timesteps, 
        callback=callback
    )

    if hasattr(vec_env, "close") and callable(vec_env.close):
        vec_env.close()

    return model
