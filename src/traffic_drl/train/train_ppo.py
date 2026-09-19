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

    TODO (SV2): map ``config.model_hyperparameters`` to the SB3 PPO constructor;
    keep PPO secondary to the minimum DQN deliverable.

    Returns:
        PPO: An initialised SB3 PPO model.
    """
    raise NotImplementedError


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

    TODO (SV2): call ``model.learn(...)`` and return the trained model.

    Returns:
        PPO: The trained (or partially trained) PPO model.
    """
    raise NotImplementedError


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
